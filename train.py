import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR

# ResNet-9 with meta-smooth modifications
class MetaSmoothResNet9(nn.Module):
    def __init__(self, num_classes=10, width_multiplier=2.0, 
                 init_scale=2.0, final_scale=0.125):
        super().__init__()
        self.init_scale = init_scale
        self.final_scale = final_scale
        
        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.5),
                nn.GELU()
            ]
            if pool:
                layers.append(nn.AvgPool2d(2))
            return nn.Sequential(*layers)
        
        channels = [int(64 * width_multiplier)] * 4
        self.conv1 = conv_block(3, channels[0])
        self.conv2 = conv_block(channels[0], channels[1], pool=True)
        self.res1 = nn.Sequential(conv_block(channels[1], channels[1]),
                                  conv_block(channels[1], channels[1]))
        
        self.conv3 = conv_block(channels[1], channels[2], pool=True)
        self.conv4 = conv_block(channels[2], channels[3], pool=True)
        self.res2 = nn.Sequential(conv_block(channels[3], channels[3]),
                                  conv_block(channels[3], channels[3]))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[3], num_classes, bias=True)
        )
        # Initialize scales
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, init_scale)
                nn.init.zeros_(m.bias)
        
        nn.init.constant_(self.classifier[-1].weight, final_scale)
        if self.classifier[-1].bias is not None:
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)
        return x

# Data loading
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.125, 0.125)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=250,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=250,
                                         shuffle=False, num_workers=2)

# Model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetaSmoothResNet9(width_multiplier=2.0, 
                         init_scale=2.0, 
                         final_scale=0.125).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.5, 
                     momentum=0.85, 
                     weight_decay=1e-5,
                     nesterov=True)

# Learning rate scheduler
scheduler = OneCycleLR(optimizer, max_lr=0.5,
                       epochs=18,
                       steps_per_epoch=len(trainloader),
                       pct_start=0.25,
                       final_div_factor=10000)

criterion = nn.CrossEntropyLoss()

best_acc = 0.0
for epoch in range(18):
    # Training
    model.train()
    train_loss = 0.0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    
    # Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    train_loss /= len(trainloader)
    test_loss /= len(testloader)
    acc = 100.*correct/total
    
    print(f"Epoch {epoch+1}:")
    print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {acc:.2f}%")
    
    # Save best model
    if acc > best_acc:
        print(f"New best model ({acc:.2f}%)")
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")

print(f"\nBest Accuracy: {best_acc:.2f}%")