import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# Dataset 불러오기
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train = True,
                                             transform=transform,
                                             download = True)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train = False,
                                            transform = transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = 100,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# VGG Model 생성(간단화됨! block2 한개로 구성)
class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.maxpool = nn.MaxPool2d(4)
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
    self.relu = nn.ReLU(inplace = True)
    self.fc = nn.Linear(1568, 10)
  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x

model = VGG().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

total_step = len(train_loader)
curr_lr = learning_rate

def train(epoch):
  global num_epochs
  global total_step
  model.train()
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print("Epoch [{}/{}], step [{}/{}], Loss: {:4f}".format(epoch, num_epochs, i+1, total_step, loss.item()))

  if (epoch+1) % 20 == 0:
    curr_lr /= 3
    update_lr(optimizer, curr_lr)

def test():
  model.eval()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    
    print('Accuracy of the odel on the test images: {} %'.format(100*correct/total))
  
for epoch in range(1,10):
  train(epoch)
  test()