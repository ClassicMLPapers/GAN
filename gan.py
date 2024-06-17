import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


import torchvision.datasets as datasets
import torchvision.transforms as transforms

BATCH_SIZE = 100

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(device)

transforms = transforms.Compose([ # https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
    transforms.ToTensor(), # Converts PIL to Tensors
                           # 3D Numpy array (heigh, width, channels) to 3D Tensor (channels, height, width)
    transforms.Normalize(mean = 0.5, std = 0.5)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transforms)
# print(train_dataset.data.shape)
# print(test_dataset.data.shape)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# print(train_loader.batch_size)
# print(test_loader.batch_size)

# Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(100, 128, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2)

    def forward(self, z):
        x = F.leaky_relu(self.conv1(z), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.tanh(self.conv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.fc = nn.Linear(128*5*5, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 128*5*5)
        x = F.sigmoid(self.fc(x))
        return x

# Network
z_dim = 100
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2) # 784

G = Generator().to(device)
# print(G)
D = Discriminator().to(device)
# print(D)

#loss
criterion = nn.BCELoss()

#optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr) 
D_optimizer = optim.Adam(D.parameters(), lr=lr)

# Discriminator train
def D_train(x, bs):
    D.zero_grad()

    # train discriminator on real
    x_real = x.view(-1, 1, 28, 28).to(device)
    D_output = D(x_real)
    y_real = torch.ones(D_output.size(0), 1).to(device)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.size(0), z_dim, 1, 1).to(device)  
    x_fake = G(z)
    D_output = D(x_fake)
    y_fake = torch.zeros(D_output.size(0), 1).to(device)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.item()

# Generator Train
def G_train(bs):
    G.zero_grad()

    # train generator
    z = torch.randn(bs, z_dim, 1, 1).to(device)
    x_fake, y_fake = G(z), torch.ones(bs, 1).to(device)

    D_output = D(x_fake)
    G_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

# Training loop
for epoch in range(100):
    for i, batch in enumerate(train_loader):
        x = batch[0] 
        bs = BATCH_SIZE
        bs = x.size(0)
        D_loss = D_train(x, bs)
        G_loss = G_train(bs)
        print('Epoch: ', epoch+1, ', Batch: ', i+1, ', D_loss: ', D_loss, ', G_loss: ', G_loss)
