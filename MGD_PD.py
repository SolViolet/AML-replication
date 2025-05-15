import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
import copy, math, random, numpy as np
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image
import os


# 1) Memory-Optimized Model with explicit use_reentrant

class MetaSmoothResNet9(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=2.0,
                 init_scale=2.0, final_scale=0.125):
        super().__init__()
        self.init_scale = init_scale
        self.final_scale = final_scale

        def conv_block(in_c, out_c, pool=False):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.5),
                nn.GELU()
            ]
            if pool: layers.append(nn.AvgPool2d(2))
            return nn.Sequential(*layers)

        ch = [int(64*width_multiplier)]*4
        self.conv1 = conv_block(3, ch[0])
        self.conv2 = conv_block(ch[0], ch[1], pool=True)
        self.res1 = nn.Sequential(conv_block(ch[1], ch[1]),
                                 conv_block(ch[1], ch[1]))
        self.conv3 = conv_block(ch[1], ch[2], pool=True)
        self.conv4 = conv_block(ch[2], ch[3], pool=True)
        self.res2 = nn.Sequential(conv_block(ch[3], ch[3]),
                                 conv_block(ch[3], ch[3]))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch[3], num_classes)
        )

        # Initialize with scales
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, init_scale)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.classifier[-1].weight, final_scale)
        if self.classifier[-1].bias is not None:
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, x):
        # Checkpointed forward pass with use_reentrant=False
        def segment1(x):
            x = self.conv1(x)
            x = self.conv2(x)
            return self.res1(x) + x

        def segment2(x):
            x = self.conv3(x)
            x = self.conv4(x)
            return self.res2(x) + x

        x = torch.utils.checkpoint.checkpoint(segment1, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(segment2, x, use_reentrant=False)
        return self.classifier(x)


# 2) Memory-Efficient ReplayManager

class ReplayManager:
    def __init__(self, model, optimizer, total_steps, k, dataset, batch_size, device):
        self.model = model
        self.optimizer = optimizer
        self.T = total_steps
        self.k = k
        self.dataset = dataset
        self.batch_size = batch_size
        self.dev = device

        # Build k-ary tree structure
        self.segments = []
        def build_segments(start, length):
            if length <= batch_size*2:  # Leaf nodes process 2 batches
                self.segments.append((start, length))
                return
            seg_len = math.ceil(length / k)
            for i in range(k):
                new_start = start + i*seg_len
                new_len = min(seg_len, length - i*seg_len)
                if new_len > 0:
                    build_segments(new_start, new_len)
        build_segments(0, total_steps)

        self.checkpoints = {}
        self.active_path = set()

    def save_checkpoint(self, step_idx):
        for seg_start, seg_len in self.segments:
            if seg_start == step_idx:
                if seg_start not in self.checkpoints:
                    # Prune old checkpoints
                    for key in list(self.checkpoints.keys()):
                        if key not in self.active_path:
                            del self.checkpoints[key]
                    # Save new checkpoint
                    self.checkpoints[seg_start] = (
                        copy.deepcopy(self.model.state_dict()),
                        copy.deepcopy(self.optimizer.state_dict())
                    )
                    self.active_path.add(seg_start)
                break


# _replay_segment Method Fix (Remove torch.no_grad)

    def _replay_segment(self, seg_start, seg_len, criterion):
        m_state, o_state = self.checkpoints[seg_start]

        # Load model with gradient preservation
        self.model.load_state_dict(m_state)
        for param in self.model.parameters():
            param.requires_grad_(True)  # Force gradient tracking

        # Load optimizer with gradient tracking
        self.optimizer.load_state_dict(o_state)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.requires_grad_(True)

        # Track loss as tensor
        total_loss = torch.tensor(0.0, device=self.dev, requires_grad=True)

        for step in range(seg_len):
            batch_idx = (seg_start + step) % math.ceil(len(self.dataset)/self.batch_size)
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))

            # Get batch with gradient tracking
            batch_x = torch.stack([self.dataset[i][0] for i in range(start_idx, end_idx)])
            batch_y = torch.tensor([self.dataset[i][1] for i in range(start_idx, end_idx)])
            batch_x, batch_y = batch_x.to(self.dev), batch_y.to(self.dev)

            # Forward with proper checkpointing
            def compute_loss(x, y):
                with torch.set_grad_enabled(True):
                    return criterion(self.model(x), y)

            loss = torch.utils.checkpoint.checkpoint(
                compute_loss,
                batch_x,
                batch_y,
                use_reentrant=False  # Explicitly use recommended mode
            )
            total_loss = total_loss + loss

            # Backward with gradient control
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            del batch_x, batch_y, loss
            torch.cuda.empty_cache()

        return total_loss


# Modified ReplayManager Methods

    def compute_metagrad(self, val_batch, criterion, poison_imgs):
        xi, yi = [t.to(self.dev) for t in val_batch]

        # Reset model gradients
        self.model.zero_grad()

        # Ensure gradient tracking for all components
        with torch.set_grad_enabled(True):
            # Forward pass with explicit gradient preservation
            loss_T = criterion(self.model(xi), yi)
            theta_bar = torch.autograd.grad(
                loss_T,
                list(self.model.parameters()),
                create_graph=True,
                retain_graph=True
            )

        z_meta = torch.zeros_like(poison_imgs)
        processed_segments = sorted(self.segments, key=lambda x: -x[0])

        for i, (seg_start, seg_len) in enumerate(processed_segments):
            # Ensure poison images track gradients
            poison_imgs = poison_imgs.clone().detach().requires_grad_(True)

            # Replay segment with gradient tracking
            loss_seg = self._replay_segment(seg_start, seg_len, criterion)

            # Compute gradients with strict checking
            grads = torch.autograd.grad(
                outputs=loss_seg,
                inputs=[*self.model.parameters(), poison_imgs],
                grad_outputs=torch.ones_like(loss_seg),
                retain_graph=(i < len(processed_segments)-1),
                allow_unused=True  # Changed from False to True
            )

           # Handle unused gradients by replacing None with zeros
            valid_grads = []
            for g, param in zip(grads, [*self.model.parameters(), poison_imgs]):
                if g is None:
                    valid_grads.append(torch.zeros_like(param))
                else:
                    valid_grads.append(g)

            # Accumulate gradients
            z_meta += valid_grads[-1].detach()

            # Update parameter gradients
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), valid_grads[:-1]):
                    param.grad = grad.detach() if param.grad is None else param.grad + grad.detach()

            del loss_seg, grads
            torch.cuda.empty_cache()

        return z_meta


# 3) Optimized Poisoning Pipeline

def run_mgd_poisoning(epsilon=0.025, meta_steps=50, inner_epochs=12
                      ,
                     poison_lr=0.01, k=2, batch_size=250, model_ckpt='resnet9.pth'):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])

    full_train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    n_train = len(full_train)
    train_size = int(0.8 * n_train)
    train_sub = Subset(full_train, range(train_size))
    val_sub = Subset(full_train, range(train_size, n_train))

    n_poison = int(epsilon * len(train_sub))
    base_imgs = torch.stack([full_train[i][0] for i in range(n_poison)])
    base_imgs = base_imgs.to(dev)  # First move to device
    base_imgs = base_imgs.clone().detach().requires_grad_(True)  # Then make leaf tensor
    base_lbls = [full_train.targets[i] for i in range(n_poison)]

    def evaluate(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                preds = model(xb).argmax(1)
                correct += preds.eq(yb).sum().item()
                total += yb.size(0)
        return 100. * correct / total

    model = MetaSmoothResNet9().to(dev)
    model.load_state_dict(torch.load(model_ckpt, map_location=dev))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    clean_acc = evaluate(model, test_loader)
    print(f"Clean model accuracy: {clean_acc:.2f}%")

    best_acc = 100.0
    scaler = GradScaler()

    for meta_step in range(1, meta_steps + 1):
        torch.cuda.empty_cache()
        print(f"\nMeta Step {meta_step}/{meta_steps}")

        class PoisonedDataset(torch.utils.data.Dataset):
            def __init__(self, clean, imgs, lbls):
                self.clean = clean
                self.imgs = imgs
                self.lbls = lbls
                self.n_poison = len(imgs)
            def __len__(self): return len(self.clean)
            def __getitem__(self, idx):
                if idx < self.n_poison:
                    return self.imgs[idx], self.lbls[idx]
                return self.clean[idx - self.n_poison]

        poison_ds = PoisonedDataset(train_sub, base_imgs, base_lbls)
        poison_loader = DataLoader(poison_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

        model.load_state_dict(torch.load(model_ckpt, map_location=dev))
        opt = optim.SGD(model.parameters(), lr=0.5, momentum=0.85,
                       weight_decay=1e-5, nesterov=True)
        total_steps = inner_epochs * len(poison_loader)
        replay = ReplayManager(model, opt, total_steps, k, poison_ds, batch_size, dev)

        model.train()
        step_idx = 0
        for epoch in range(inner_epochs):
            for xb, yb in poison_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                replay.save_checkpoint(step_idx)
                opt.zero_grad()
                with autocast(device_type='cuda' if 'cuda' in str(dev) else 'cpu'):
                    loss = nn.CrossEntropyLoss()(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                step_idx += 1
                if step_idx % 10 == 0:
                    torch.cuda.empty_cache()

        current_acc = evaluate(model, test_loader)
        print(f"Current test accuracy: {current_acc:.2f}%")
        if current_acc < best_acc:
            best_acc = current_acc
            best_poison = base_imgs.detach().clone()
            torch.save(model.state_dict(), f"worst_model_step{meta_step}.pth")
            print(f"New worst model saved (acc: {best_acc:.2f}%)")

            # Save first 1000 poisoned images from base_imgs
            os.makedirs("best_poisons", exist_ok=True)
            denorm = transforms.Compose([
                transforms.Normalize(mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
                                    std=[1/0.2470, 1/0.2435, 1/0.2616]),
                transforms.Lambda(lambda x: x.clamp(0, 1))
            ])

            # Save first 1000 poison images from base_imgs
            for idx in range(1000):
                img = denorm(base_imgs[idx].float().cpu())
                save_image(img, f"best_poisons/poison_{idx}.png")

            print(f"Saved 1000 poisoned samples from current base_imgs")

        # Compute meta gradient (no torch.no_grad here)
        val_batch = next(iter(val_loader))
        poison_imgs = base_imgs
        z_grad = replay.compute_metagrad(val_batch, nn.CrossEntropyLoss(), poison_imgs)

        updated_imgs = (poison_imgs + poison_lr * z_grad.sign()).clamp(0, 1)
        base_imgs = updated_imgs.detach().cpu()

        del poison_imgs, z_grad, updated_imgs
        torch.cuda.empty_cache()


    print(f"\nFinal worst accuracy achieved: {best_acc:.2f}%")
    print(f"Total accuracy drop: {clean_acc - best_acc:.2f}%")

if __name__ == '__main__':
    run_mgd_poisoning()
    
    
