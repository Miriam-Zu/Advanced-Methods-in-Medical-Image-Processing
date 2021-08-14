import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

def get_dataloader():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    BATCH_SIZE = 16
    
    train_root = "/home/dsi/zuckerm1/imaging/binary_01/train/"
    train_data = torchvision.datasets.ImageFolder(root=train_root, transform=transform)
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # train_data[i][0] # tensor image
    # train_data[i][1] # class 0,1,2,3

    test_root = "/home/dsi/zuckerm1/imaging/binary_01/test"
    test_data = torchvision.datasets.ImageFolder(root=test_root, transform=transform)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    return trainloader, testloader

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNN, self).__init__()
        # in, out, kernel, stride=1, padding=0
        self.conv1 = nn.Conv2d(3, 96, 11, 4)

        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))

        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1)

        self.d1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3*3*256, 1024)

        self.d2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)

        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        bn = nn.BatchNorm2d(x.shape[1]).to(device)
        x = self.pool(bn(x))

        x = F.relu(self.conv2(x))
        bn2 = nn.BatchNorm2d(x.shape[1]).to(device)
        x = self.pool(bn2(x))

        x = F.relu(self.conv3(x))
        bn3 = nn.BatchNorm2d(x.shape[1]).to(device)
        x = bn3(x)

        x = F.relu(self.conv4(x))
        bn4 = nn.BatchNorm2d(x.shape[1]).to(device)
        x = bn4(x)

        x = F.relu(self.conv5(x))
        bn5 = nn.BatchNorm2d(x.shape[1]).to(device)
        x = self.pool(bn5(x))

        x = x.reshape(x.shape[0], -1)

        x = self.d1(F.relu(self.fc1(x)))

        x = self.d2(F.relu(self.fc2(x)))

        x = F.softmax(self.fc3(x))
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, critirion, train_loader, epoch, epochs):
    model.train()
    correct = 0
    train_loss = 0
    for data, labels in train_loader:
        optimizer.zero_grad()
        data = data.to(device = device)
        labels = labels.to(device=device)
        output = model(data).to(device=device)

        loss = critirion(output, labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
    train_loss /= len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    print('epoch : {}/{}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(epoch + 1, epochs, train_loss, correct,
                                                                                  len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
    return train_loss, accuracy

def validation(valid_loader, model, critirion, test):
    model.eval()
    valid_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in valid_loader:
            data = data.to(device = device)
            target = target.to(device=device)
            output = model(data).to(device=device)

            valid_loss += critirion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            
            if (test):
              target_ = target.view_as(pred).tolist()
              target_ = [k for sub in target_ for k in sub]
              y_true.extend(target_)
              pred_ = pred.tolist()
              pred_ = [k for sub in pred_ for k in sub]
              y_pred.extend(pred_) 

    valid_loss /= len(valid_loader.dataset)
    accuracy = correct/ len(test_loader.dataset)
    if (not test):
      print('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(valid_loss, correct, len(valid_loader.dataset), 100. * correct / len(valid_loader.dataset)))
    if (test):
      print ('test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(valid_loss, correct, len(valid_loader.dataset), 100. * correct / len(valid_loader.dataset)))
      
    return valid_loss, accuracy, y_true, y_pred

def plotdata(loss_train, loss_test, accuracy_train, accuracy_test):
    plt.figure()
    plt.subplot(211)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(num_epochs), loss_train, "r", range(num_epochs), loss_test, "b")
    plt.subplot(212)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(range(num_epochs), accuracy_train, "r", range(num_epochs), accuracy_test, "b")
    plt.show()

# load data
train_loader, test_loader = get_dataloader()

# Initialize network
model = CNN().to(device)

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 0.01
num_epochs = 40

# Loss func
critirion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

loss_train = []
loss_test = []
accuracy_train = []
accuracy_test = []
for epoch in range(num_epochs-1):
        loss_accuracy_train = (train(model, optimizer, critirion, train_loader, epoch, num_epochs))
        loss_train.append(loss_accuracy_train[0])
        accuracy_train.append(loss_accuracy_train[1])
        valid_loss, accuracy, _, _ = validation(test_loader, model, critirion, False)
        loss_test.append(valid_loss)
        accuracy_test.append(accuracy)
loss_accuracy_train = (train(model, optimizer, critirion, train_loader, num_epochs-1, num_epochs))
loss_train.append(loss_accuracy_train[0])
accuracy_train.append(loss_accuracy_train[1])
valid_loss, accuracy, y_true, y_pred = validation(test_loader, model, critirion, True)
loss_test.append(valid_loss)
accuracy_test.append(accuracy)

plotdata(loss_train, loss_test, accuracy_train, accuracy_test)

cm = confusion_matrix(y_true, y_pred, labels = [0, 1])

cm_display = ConfusionMatrixDisplay(cm, display_labels = [0, 1]).plot()

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='binary 1 2').plot()
