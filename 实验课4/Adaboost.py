from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import numpy as np

class training_data():
    def __init__(self):
        self.data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )  # 60000 training samples

        self.split = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def dataloader(self, batch_size):

        return self.data[pass]

class test_data():
    def __init__(self):
        self.data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )  # 10000 test samples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def top_1_acc(self, model):
        acc = 0
        for i in range(len(self.data)):
            ans = model.pred(self.data[i][0])
            if ans == self.data[i][1]:
                acc += 1
        return acc / len(self.data)

class Weak_classifier():
    def __init__(self, dim):
        self.dim = dim
        self.w = np.random.random(dim) * 2 -1.0
        self.b = random.random() * 2 - 1.0

    def pred(self, x):
        if np.dot(self.w, x) + self.b >= 0:
            return 1
        else:
            return -1

    def train(self, x, y):
        self.w -= 0.001 * x
        self.b -= 0.001


class Adaboost():
    def __init__(self, num):
        global Weak_classifier
        self.num = num
        self.weak_predictor = []
        for i in range(self.num):
            self.weak_predictor.append(Weak_classifier())
    def



def main():
    pass

if __name__ == "__main__":
    main()