import numpy as np
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

# load data function
def get_data():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_root = "./train"
    train_data = torchvision.datasets.ImageFolder(root=train_root, transform=transform)

    test_root = "./test"
    test_data = torchvision.datasets.ImageFolder(root=test_root, transform=transform)

    return train_data, test_data

# perceptron
def perceptron():
    w = torch.from_numpy(np.random.uniform(-0.05, 0.05, [4, 224*224*3]))
    # train
    for e in range(epoch):
        for x,y in train:
            x = x.reshape(-1)
            y = int(y)
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                w[y_hat] = w[y_hat] - x
                w[y] = w[y] + x

    # test
    results = []
    for x,y in test:
        x = x.reshape(-1)
        y = int(y)
        y_hat = np.argmax(np.dot(w, x))
        results.append([y_hat, y])
    return results


if __name__ == '__main__':
        train, test = get_data()
        epoch = 10
        r = perceptron()
        # calc accuracy
        correct = 0
        wrong = 0
        for i in r:
            if i[0] == i[1]:
                correct = correct + 1
            else:
                wrong = wrong + 1
        print(f"epoch {epoch}: {correct / (correct + wrong)}")
        
        # confusion matrix
        y_true = []
        y_pred = []
        for el in r:
            y_pred.append(el[0])
            y_true.append(el[1])
        
        cm = confusion_matrix(y_true, y_pred, labels = [0, 1, 2, 3])
        cm_display = ConfusionMatrixDisplay(cm, display_labels = [0, 1, 2, 3]).plot()
