import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the same model architecture
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.classifier(x)
        return x

# Load the test data
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=250, shuffle=False, num_workers=2)

# Initialize the model and load the saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetaSmoothResNet9(width_multiplier=2.0, 
                         init_scale=2.0, 
                         final_scale=0.125).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set the model to evaluation mode

# Test the model
criterion = nn.CrossEntropyLoss()
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

test_loss /= len(testloader)
accuracy = 100. * correct / total

print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

# Print class-wise accuracy
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

class_correct = list(0. for _ in range(10))
class_total = list(0. for _ in range(10))

with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        c = (predicted == targets).squeeze()
        
        for i in range(targets.size(0)):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print("\nClass-wise accuracy:")
for i in range(10):
    print(f"{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")