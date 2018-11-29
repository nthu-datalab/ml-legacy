"""
Some codes from https://github.com/Newmu/dcgan_code
"""

import math
import random
import scipy.misc
import numpy as np
from numpy.random import shuffle
from scipy.misc import imsave


import tensorflow as tf
import tensorflow.contrib.slim as slim


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)



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
def grayscale_grid_vis(X, nhw, save_path=None):
    (nh, nw) = nhw[0], nhw[1]
    if len(X.shape) == 4:
        X = X.reshape(-1,28,28)    
    h, w = X[0].shape[:2]

    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img



def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps = len(images) / duration)
    