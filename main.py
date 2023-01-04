import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import cv2
dict_label = {"cloudy":0, "desert":1, "green_area":2, "water":3}

def readfile(path, label):
    image_dir = os.listdir(path)
    x = []
    y = []
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x.append(cv2.resize(img, (128, 128)))
        y.append(dict_label[label])
    return x, y

dir = "C:/Users/user/Desktop/data/data"
cloudy_x, cloudy_y = readfile(os.path.join(dir, "cloudy"), "cloudy")
desert_x, desert_y = readfile(os.path.join(dir, "desert"), "desert")
green_area_x, green_area_y = readfile(os.path.join(dir, "green_area"), "green_area")
water_x, water_y = readfile(os.path.join(dir, "water"), "water")

x_train = cloudy_x + desert_x + green_area_x + water_x
y_train = cloudy_y + desert_y + green_area_y + water_y

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        X = self.x[idx]
        Y = self.y[idx]
        if self.transform is not None:
            X = self.transform(X)
        return X, Y

batch_size = 128
train_set = ImgDataset(x_train, y_train, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

model = Classifier()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epoch = 30

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    total = 0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0])
        batch_loss = loss(train_pred, data[1])
        batch_loss.backward()
        optimizer.step()

        total += len(data[0])
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        if i % 50 == 0:
            print(f"train_acc:{train_acc/total:.3f}%, train_loss:{train_loss/total:.5f}%")