from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

class dataset():
    def __init__(self):
        self.boston = load_boston()

    def demonstration(self): # 506 * 13
        print(len(self.boston['data']))

    def load(self, mode = 'train', attri = 13):
        train_list = [i for i in range(506) if i%4 != 0]
        test_list = [i for i in range(506) if i % 4 == 0]
        if mode == 'train':
            return (self.boston['data'][train_list])[:,: attri],self.boston['target'][train_list]
        elif mode == 'test':
            return (self.boston['data'][test_list])[:,:attri], self.boston['target'][test_list]
        elif mode == 'all':
            return (self.boston['data'])[:,:attri], self.boston['target']

if __name__ == "__main__":
    dd = dataset()
    dd.demonstration()

