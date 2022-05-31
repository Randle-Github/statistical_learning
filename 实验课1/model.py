import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

loss_his = []
regress_his = []
def gen(i):
    plt.plot(np.arange(i+1), loss_his[:i+1], label = "loss", c = "green")
    # plt.plot(np.arange(i+1), regress_his[:i+1], label = "regression", c = "orange")

def init_gen():
    plt.plot(np.arange(1), regress_his[1], label="regression", c="orange")
    plt.legend()


class Linear_Regression():
    def __init__(self):
        self.W = None # (m+1,1)

    def expand(self, X): # (n,m) -> (n,m+1)
        return np.column_stack((X, np.ones((X.shape[0], 1))))

    def fit(self, X, y, mode = "analytic", alpha = 0.0005, times = 1000, lamb = 0):
        self.X = X
        self.y = y
        expand_X = self.expand(self.X)

        if mode == 'analytic':
            self.W = np.dot(np.dot(np.linalg.inv(np.dot(expand_X.T,expand_X)), expand_X.T), self.y)
            pred = self.predict(self.X)
            loss = self.loss(pred, self.y)
            print(loss)

        elif mode == 'gradient':
            global loss_his, regress_his
            self.W = np.random.random(14)
            for i in range(times):
                print(self.W)
                pred = self.predict(self.X)
                loss = self.loss(pred, self.y)
                print("iter_time: {}, loss: {}".format(i,loss))
                loss_his.append(loss)
                regress_his.append(1-pred.shape[0]*loss/np.sum((self.y-np.mean(self.y))**2))
                self.W = self.W - alpha *(-2. / float(pred.shape[0]) * np.dot(self.y - pred, expand_X) + lamb * self.W)

            print(regress_his[-1])
            fig, _ = plt.subplots()
            plt.title("loss with gradient descent")
            # plt.plot(np.arange(times), loss_his, label = "loss")
            ani = animation.FuncAnimation(fig = fig,func=gen, frames=times-2, interval=50)

            plt.show()

        elif mode == 'likelyhood':
            loss_his = []


    def predict(self, X):
        expand_X = self.expand(X)
        return np.dot(expand_X, self.W)

    def loss(self, pred, target):
        return np.sum((target-pred)**2) / pred.shape[0]
