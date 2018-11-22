
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

def plot_imgs(imgs, n = 10, title = None):
    plt.figure(figsize=(n, 2))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if title is not None:
        plt.title(title, x = -n//2)
    plt.show()

def tangent_vecs(jaco_matrix):
    
    # get jacobian matrix of size (code_size * img_dim)
    
    # get tangent vectors via SVD
    U, s, V = np.linalg.svd(jaco_matrix, full_matrices=False)
    print(s.shape)
    plt.bar(range(s.shape[0]), s, alpha=0.5)
    plt.ylabel('SVD values')
    plt.xlabel('Index')
    plt.tight_layout()
    #plt.savefig('./output/fig-svd.png', dpi=300)
    plt.show()
    
    
    return V
def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = int(n / size)
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])
