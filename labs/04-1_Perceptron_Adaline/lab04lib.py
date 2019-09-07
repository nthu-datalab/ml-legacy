from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

########### plot validation history ############


def plot_validation_history(his, fig_path):
    train_loss = his.history['loss']
    val_loss = his.history['val_loss']

    # visualize training history
    plt.plot(
        range(1, len(train_loss) + 1),
        train_loss, 
        color = 'blue',
        label = 'Train loss'
    )
    plt.plot(
        range(1, len(val_loss) + 1),
        val_loss,
        color = 'red',
        label = 'Val loss'
    )
    plt.legend(loc = "upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig(fig_path, dpi = 300)
    plt.show()


########### print download progress ############
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x = X[y == cl, 0],
            y = X[y == cl, 1],
            alpha = 0.8,
            c = [cmap(idx)],  # Prevents warning
            linewidths = 1,
            marker = markers[idx],
            label = cl,
            edgecolors = 'k'
        )

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c = '',
            alpha = 1.0,
            linewidths = 1,
            marker = 'o',
            s = 55,
            label = 'test set', 
            edgecolors = 'k'
        )
