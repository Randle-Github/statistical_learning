import model
import datasets
import matplotlib.pyplot as plt
from utils import PCA_helper

def main():
    dataset = datasets.dataset()
    attri = 13
    n_components = 13
    X, y = dataset.load(mode = 'all', attri = attri)
    X/=100
    pca = PCA_helper(X, n_components = n_components)
    train_X, train_y = dataset.load(mode = 'train', attri = attri)
    test_X, test_y = dataset.load(mode = 'test', attri = attri)
    train_X/=100
    test_X/=100
    train_X = pca._transform(train_X)
    test_X = pca._transform(test_X)
    y/=1000
    test_y/=1000
    train_y/=1000

    classifier = model.Linear_Regression()
    # classifier.fit(train_X, train_y)
    classifier.fit(train_X, train_y, mode = "gradient")
    # print(classifier.loss(classifier.predict(test_X),test_y))

    '''
    plt.title("PCA")
    plt.scatter(train_X, train_y, c = "green", label = "train")
    plt.scatter(test_X, test_y, c = "red", label = "test")
    plt.legend()
    plt.show()
    '''

if __name__ == "__main__":
    main()

    '''
    result:
        analytic = 0.000019616608027106
        gradient = 0.000305431660412052
    '''