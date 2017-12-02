import os
import sys
from six.moves import urllib
import tarfile
import tensorflow as tf
import numpy as np

def maybe_download_and_extract(dest_directory, url):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    file_name = 'cifar-10-binary.tar.gz'
    file_path = os.path.join(dest_directory, file_name)
    # if have not downloaded yet
    if not os.path.exists(file_path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r%.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush() # flush the buffer
        print('>> Downloading %s ...' % file_name)
        file_path, _ = urllib.request.urlretrieve(url, file_path, _progress)
        file_size = os.stat(file_path).st_size
        print('\r>> Total %d bytes' % file_size)
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        # Open for reading with gzip compression, then extract all
        tarfile.open(file_path, 'r:gz').extractall(dest_directory)
    print('>> Done')

class CNN_Model(object):
    def __init__(self, batch_size, 
                 num_classes, 
                 num_training_example, 
                 num_epoch_per_decay,
                 init_lr,
                 moving_average_decay):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_training_example = num_training_example
        self.num_epoch_per_decay = num_epoch_per_decay
        self.init_lr = init_lr
        self.moving_average_decay = moving_average_decay
    
    def _variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var
    
    def _variable_with_weight_decay(self, name, shape, stddev, wd=0.0):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
        var = self._variable_on_cpu(name, shape, initializer)
        # deal with weight decay
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return var
    
    def inference(self, images):
        """build the model
        Args:
            images with shape [batch_size,24,24,3]
        Return:
            logits with shape [batch_size,10]
        """
        with tf.variable_scope('conv_1') as scope:
            kernel = self._variable_with_weight_decay('weights', [5,5,3,64], 5e-2)
            conv = tf.nn.conv2d(images, kernel, strides=[1,1,1,1], padding="SAME")
            biases = self._variable_on_cpu('bias', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv_1 = tf.nn.relu(pre_activation, name=scope.name)
        # pool_1
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1,3,3,1], strides=[1,2,2,1], 
                                padding='SAME', name='pool_1') 
        # norm_1 (local_response_normalization)
        norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm_1')
        # conv2
        with tf.variable_scope('conv_2') as scope:
            kernel = self._variable_with_weight_decay('weights', [5, 5, 64, 64], 5e-2)
            conv = tf.nn.conv2d(norm_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv_2 = tf.nn.relu(pre_activation, name=scope.name)
        # norm2
        norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm_2')
        # pool2
        pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                               padding='SAME', name='pool_2')
        # FC_1 (fully-connected layer)
        with tf.variable_scope('FC_1') as scope:
            flat_features = tf.reshape(pool_2, [self.batch_size, -1])
            dim = flat_features.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', [dim, 384], 0.04, 0.004)
            biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            FC_1 = tf.nn.relu(tf.matmul(flat_features, weights) + biases, name=scope.name)
        # FC_2
        with tf.variable_scope('FC_2') as scope:
            weights = self._variable_with_weight_decay('weights', [384, 192], 0.04, 0.004)
            biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            FC_2 = tf.nn.relu(tf.matmul(FC_1, weights) + biases, name=scope.name)
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [192, self.num_classes],1/192.0)
            biases = self._variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
            logits = tf.add(tf.matmul(FC_2, weights), biases, name=scope.name)
        return logits

    def loss(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
    
    def train(self, total_loss, global_step):
        num_batches_per_epoch = self.num_training_example / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.num_epoch_per_decay)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.init_lr, global_step, decay_steps, 
                                        decay_rate=0.1, staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        # This step just records the moving average weights but not uses them
        ema = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
        self.ema = ema
        variables_averages_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op




