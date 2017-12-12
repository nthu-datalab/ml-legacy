
import numpy as np 
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def bn(X, eps=1e-8, offset = 0, scale = 1):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        var = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        output = tf.nn.batch_normalization(X, mean, var, offset, scale, eps)
    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        var = tf.reduce_mean(tf.square(X-mean), 0)
        output = tf.nn.batch_normalization(X, mean, var, offset, scale, eps)
    else:
        raise NotImplementedError
    return output



def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                  initializer=tf.random_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        return conv

def deconv2d(input_, nf,
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
        name="deconv2d", with_w=False):

    input_shape = tf.shape(input_)
    output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, nf]
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])
        deconv = tf.reshape(deconv, output_shape)


        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                     tf.random_normal_initializer(stddev=stddev))
        if with_w:
            return tf.matmul(input_, matrix), matrix
        else:
            return tf.matmul(input_, matrix)
def bce(x, y):
# Expect logits x
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)