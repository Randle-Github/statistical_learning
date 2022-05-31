from sklearn.decomposition import PCA

class PCA_helper():
    def __init__(self,X, n_components=2):
        self.model = PCA(n_components = n_components)
        self.model.fit(X)

    def _transform(self, X):
        return self.model.transform(X)