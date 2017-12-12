import numpy as np
import tensorflow as tf
from tqdm import trange
from time import time
from tensorflow.contrib import learn
from utils import *



class AutoEncoder(object):
    def __init__(self, sess, inputs, targets = None, 
                 b1 = 0.5, lr = 1., nf = 16, code_size = 16, 
                 nbatch = 256, niter = 200,
                 cost_function = 'bce', name = 'autoencoder', optimizer = 'adadelta'):
        self.sess = sess
        self.b1 = b1
        self.lr = lr
        self.nf = nf
        self.niter = niter
        self.nbatch = nbatch
        self.inputs = inputs
        self.code_size = code_size
        self.load_dataset()
        if targets is not None:
            self.targets = targets
            self.gen_noisy_data()
        else:
            self.targets = inputs
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.log = {'train_loss':[], 'valid_loss':[]}
        
        self.name = name
        

        self.build_model()
    def build_model(self):
        nf = self.nf
        code_size = self.code_size
        with tf.variable_scope(self.name) as scope:
            self.enc = tf.layers.dense(inputs=self.inputs, units = nf * 16, activation=tf.nn.relu, name='enc')
            self.enc2 = tf.layers.dense(inputs=self.enc, units = nf * 8, activation=tf.nn.relu, name='enc2')
            self.enc3 = tf.layers.dense(inputs=self.enc2, units = nf * 4, activation=tf.nn.relu, name='enc3')
            self.code = tf.layers.dense(inputs=self.enc3, units = code_size, activation=tf.nn.relu, name='code')

            self.dec = tf.layers.dense(inputs=self.code, units = nf * 4, activation=tf.nn.relu, name = 'dec')
            self.dec2 = tf.layers.dense(inputs=self.dec, units = nf * 8, activation=tf.nn.relu, name = 'dec2')
            self.dec3 = tf.layers.dense(inputs=self.dec, units = nf * 16, activation=tf.nn.relu, name = 'dec3')
            self.recon_logits = tf.layers.dense(inputs=self.dec, units = 28 * 28, name = 'recon_logits')
            self.jacobian_op = self.jacobian(self.code, self.inputs)
            if self.cost_function == 'mse':
                self.recon = self.recon_logits
                self.cost = tf.reduce_mean(tf.pow(self.targets - self.recon_logits, 2))
            elif self.cost_function == 'bce':
                self.recon = tf.nn.sigmoid(self.recon_logits)
                self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                                                labels = self.targets, 
                                                                logits = self.recon_logits))
                
            
            else:
                raise NotImplementedError
    def train(self):
        if self.optimizer == 'adadelta':
            self.optim = tf.train.AdadeltaOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adam':
            self.optim = tf.train.AdamOptimizer(self.lr, beta1=self.b1).minimize(self.cost)
        elif self.optimizer == 'rmsprop':
            self.optim = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)
        else:
            raise NotImplementedError
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
                    
        X = self.inputs
        t0 = time()
        if self.targets is not self.inputs:
            print('Denoising autoencoder')
            Y = self.targets
            for epoch in trange(self.niter):
                t = time()
                shuffle(self.trX, self.trX_noisy)
                for batch, noisy_batch in iter_data(self.trX, self.trX_noisy, size = self.nbatch):
                    self.optim.run(session = self.sess, feed_dict={X: noisy_batch, Y: batch})

                idxs = np.random.randint(low = 0, high = len(self.vaX), size = self.nbatch)
                valid_batch = self.vaX[idxs]
                valid_noisy_batch = self.vaX_noisy[idxs]

                self.log['train_loss'].append(self.cost.eval(session = self.sess, 
                                                             feed_dict = {X:noisy_batch, 
                                                                          Y:batch})
                                             )
                self.log['valid_loss'].append(self.cost.eval(session = self.sess, 
                                                             feed_dict = {X:valid_noisy_batch, 
                                                                          Y: valid_batch})
                                             )
            print("final loss %g, total cost time: %.2fs" % (self.cost.eval(session = self.sess, 
                                                                            feed_dict={X: self.teX_noisy, 
                                                                                       Y: self.teX}), 
                                                             time() - t0))
            
        else:
            print('Audoencoder')
            for epoch in trange(self.niter):
                t = time()
                shuffle(self.trX)
                for batch in iter_data(self.trX, size = self.nbatch):
                    self.optim.run(session = self.sess, feed_dict={X: batch})

                idxs = np.random.randint(low = 0, high = len(self.vaX), size = self.nbatch)
                valid_batch = self.vaX[idxs]

                self.log['train_loss'].append(self.cost.eval(session = self.sess, feed_dict = {X:batch}))
                self.log['valid_loss'].append(self.cost.eval(session = self.sess, feed_dict = {X:valid_batch}))
            print("final loss %g, total cost time: %.2fs" % (self.cost.eval(session = self.sess, feed_dict={X: self.teX}), time() - t0))
        
    def load_dataset(self):
        mnist = learn.datasets.load_dataset("mnist")
        self.trX = mnist.train.images # Returns np.array
        self.vaX = mnist.validation.images # Returns np.array
        self.teX = mnist.test.images
        
    def gen_noisy_data(self):
        # Noise scale
        noise_factor = 0.4
        trX_noisy = self.trX + noise_factor * np.random.normal(loc=0., scale=1.0, size=self.trX.shape) 
        vaX_noisy = self.vaX + noise_factor * np.random.normal(loc=0., scale=1.0, size=self.vaX.shape) 
        teX_noisy = self.teX + noise_factor * np.random.normal(loc=0., scale=1.0, size=self.teX.shape) 

        # Range of our dataset is [0,1]
        self.trX_noisy = np.clip(trX_noisy, 0., 1.)
        self.vaX_noisy = np.clip(vaX_noisy, 0., 1.)
        self.teX_noisy = np.clip(teX_noisy, 0., 1.)
        
    def encode(self, inputs):
        
        return self.code.eval(session = self.sess, feed_dict={self.inputs : inputs})
    def reconstruct(self, inputs):
        return self.recon.eval(session = self.sess, feed_dict={self.inputs : inputs})
    
    
    def jacobian(self, y, x):
        # For function f: mapping from single column x to multiple values ys
        # Note: tf.gradients returns sum(dy/dx) for each x in xs, so we need to compute each y seperatedly. 
        jacobian_flat = tf.concat(
          [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y, axis = 1)], axis = 0)
        return jacobian_flat
    
    def get_jaco_matrix(self, xbatch):
        jaco_matrix = []
        for x in xbatch:
            jaco_matrix.append(self.jacobian_op.eval(session = self.sess, feed_dict={self.inputs: x.reshape(1,-1)}).reshape(1,self.code_size,28*28))
        return np.concatenate(jaco_matrix)
                    
        
        