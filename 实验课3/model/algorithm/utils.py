import numpy as np


def draw(model, X, Y, ax, t, support_vectors=None):
    ax.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap = "spring")
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    xx = np.linspace(x_lim[0], x_lim[1], 30)
    yy = np.linspace(y_lim[0], y_lim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    predict = model.predict(X)
    score = round((predict == Y).sum() / X.shape[0], 2)
    t = t + f" Acc: {score}"
    print(t)
    ax.set_title(t)

    ax.contour(XX, YY, Z, colors='black', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['-', '--', '-'])

    if support_vectors is None:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k', c = "green")
    else:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k', c = "blue")
