import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


import torchvision.datasets as datasets
import torchvision.transforms as transforms

BATCH_SIZE = 128

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


