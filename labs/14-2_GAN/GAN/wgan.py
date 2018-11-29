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
from dcgan import DCGAN


class WGAN(DCGAN):
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

        self.d_loss_real = tf.reduce_mean(self.D_logits_real)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_fake)
        
        # self.d_loss_real = tf.reduce_mean(bce(self.D_logits_real, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(bce(self.D_logits_fake, tf.zeros_like(self.D_)))
        
        self.g_loss = -tf.reduce_mean(self.D_logits_fake)
        # self.g_loss = tf.reduce_mean(bce(self.D_logits_fake, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_fake - self.d_loss_real

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()

    def train(self):    
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        with tf.control_dependencies(g_update_ops):
            g_optim = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
            ).minimize(self.g_loss, var_list=self.g_vars)
        with tf.control_dependencies(d_update_ops):
            d_optim = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
            ).minimize(self.d_loss, var_list=self.d_vars)
        # d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
        #           .minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
        #           .minimize(self.g_loss, var_list=self.g_vars)

        clip_ops = []
        for var in self.d_vars:            
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var, 
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)

        
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
                if counter % 6:
                    # Update D network
                    self.sess.run(d_optim,
                        feed_dict={ 
                            self.inputs: batch_images,
                            self.z: batch_z,
                        })
                    self.sess.run(clip_disc_weights)
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