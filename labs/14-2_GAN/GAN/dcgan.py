import os
import time
import math
from glob import glob
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from tqdm import tqdm
from ops import *
from utils import *
import matplotlib.pyplot as plt

class DCGAN(object):
    def __init__(self, sess, input_height=28, input_width=28, crop=True,
         batch_size=128, output_height=28, output_width=28,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name = 'mnist',
         input_fname_pattern='*.jpg', checkpoint_dir='checkpoint', samples_dir = None,
         show_samples = False,
         learning_rate = 2e-4, beta1 = 0.5, epoch = 100):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess

        self.epoch = epoch
        self.batch_size = batch_size
        self.sample_num = 200
        self.learning_rate = learning_rate
        self.beta1 = beta1

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.input_fname_pattern = input_fname_pattern
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.samples_dir = samples_dir
        if not os.path.exists(samples_dir) and sample_dir is not None:
            os.makedirs(samples_dir)
        self.show_samples = show_samples

        self.trX, self.teX = self.load_mnist() 
        self.ntrain = len(self.trX)
        self.c_dim = 1
        self.grayscale = (self.c_dim == 1)
        self.log = {'d_loss': [], 'g_loss': [], 'gen_samples': []}
        self.build_model()

    def build_model(self):

        image_dims = [self.input_height, self.input_width, self.c_dim]
        
        self.inputs = tf.placeholder(
            tf.float32, [None] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [None] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z)
        self.D, self.D_logits_real = self.discriminator(inputs)

        self.D_, self.D_logits_fake = self.discriminator(self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(bce(self.D_logits_real, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(bce(self.D_logits_fake, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(bce(self.D_logits_fake, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                  .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                  .minimize(self.g_loss, var_list=self.g_vars)
        init = tf.global_variables_initializer()
            
        self.sess.run(init)

        
        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        sample_idxs = np.random.randint(low = 0, high = len(self.trX), size = self.sample_num)
        sample_inputs = self.trX[sample_idxs]

        counter = 1
        start_time = time.time()        
        for epoch in range(self.epoch):
            shuffle(self.trX)
            for batch_images in iter_data(self.trX, size = self.batch_size):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                if(batch_images.shape[0] != self.batch_size):
                    continue
                if counter % 2 == 0:
                    # Update D network
                    self.sess.run(d_optim,
                        feed_dict={ 
                            self.inputs: batch_images,
                            self.z: batch_z,
                        })
                else:
                    # Update G network
                    self.sess.run(g_optim,
                        feed_dict={
                            self.z: batch_z, 
                        })
                counter += 1
    
            errD_fake = self.d_loss_fake.eval(session = self.sess, feed_dict = {self.z: batch_z})
            errD_real = self.d_loss_real.eval(session = self.sess, feed_dict = {self.inputs: batch_images})
            errG = self.g_loss.eval(session = self.sess, feed_dict = {self.z: batch_z})
            self.log['d_loss'].append(errD_fake + errD_real)
            self.log['g_loss'].append(errG)
            print("Epoch: [%2d] time: %.2fs, d_loss: %.4f, g_loss: %.4f" \
              % (epoch,time.time() - start_time, errD_fake+errD_real, errG))

            if (epoch + 1) % 1 == 0:
                samples = self.sess.run(
                  self.G,
                  feed_dict={self.z: sample_z,}
                )
                img = grayscale_grid_vis(samples, nhw=(10,20), save_path= self.samples_dir + '/%d.jpg'%epoch)
                self.log['gen_samples'].append(img)
                if self.show_samples:
                    plt.imshow(img, cmap = 'gray')
                    plt.axis('off')
                    plt.show()

            if (epoch+1) % 10 == 0:
                self.save(self.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            x = image

            h0 = lrelu(conv2d(x, self.c_dim, name='d_h0_conv'))

            h1 = lrelu(bn(conv2d(h0, self.df_dim, name='d_h1_conv')))
            h1 = flatten(h1)

            h2 = lrelu(bn(linear(h1, self.dfc_dim, 'd_h2_lin')))

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)


            h0 = tf.nn.relu(bn(linear(z, self.gfc_dim, 'g_h0_lin')))

            h1 = tf.nn.relu(bn(linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [-1, s_h4, s_w4, self.gf_dim * 2])
            
            h2 = tf.nn.relu(bn(deconv2d(h1, nf = self.gf_dim * 2, name='g_h2')))
            return tf.nn.sigmoid(deconv2d(h2, nf = self.c_dim, name='g_h3'))

    def load_mnist(self):
        mnist = learn.datasets.load_dataset("mnist")
        train_data = mnist.train.images.reshape(-1,28,28,1) # Returns np.array
        test_data = mnist.test.images.reshape(-1,28,28,1) # Returns np.array
        return train_data, test_data


    @property
    def model_dir(self):
        return "%s_%s"%(
            self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
