from __future__ import absolute_import # use absolute import instead of relative import

# '/' for floating point division, '//' for integer division
from __future__ import division  
from __future__ import print_function  # use 'print' as a function

import os

import numpy as np
import tensorflow as tf


from process_data import make_dir, get_batch_gen, process_data

class SkipGramModel:
  """ Build the graph for word2vec model """
  def __init__(self, hparams=None):

    if hparams is None:
        self.hps = get_default_hparams()
    else:
        self.hps = hparams

    # define a variable to record training progress
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    

  def _create_input(self):
    """ Step 1: define input and output """

    with tf.name_scope("data"):
      self.centers = tf.placeholder(tf.int32, [self.hps.num_pairs], name='centers')
      self.targets = tf.placeholder(tf.int32, [self.hps.num_pairs, 1], name='targets')
      dataset = tf.contrib.data.Dataset.from_tensor_slices((self.centers, self.targets))
      dataset = dataset.repeat() # # Repeat the input indefinitely
      dataset = dataset.batch(self.hps.batch_size)
      
        
      self.iterator = dataset.make_initializable_iterator()  # create iterator
      self.center_words, self.target_words = self.iterator.get_next()

  def _create_embedding(self):
    """ Step 2: define weights. 
        In word2vec, it's actually the weights that we care about
    """
    with tf.device('/gpu:0'):
      with tf.name_scope("embed"):
        self.embed_matrix = tf.Variable(tf.random_uniform([self.hps.vocab_size,
                                                           self.hps.embed_size], -1.0, 1.0),
                                                           name='embed_matrix')

  def _create_loss(self):
    """ Step 3 + 4: define the model + the loss function """
    with tf.device('/cpu:0'):
      with tf.name_scope("loss"):
        # Step 3: define the inference
        embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')


        # Step 4: define loss function
        # construct variables for NCE loss
        nce_weight = tf.Variable(
                        tf.truncated_normal([self.hps.vocab_size, self.hps.embed_size],
                                            stddev=1.0 / (self.hps.embed_size ** 0.5)),
                                            name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([self.hps.vocab_size]), name='nce_bias')

        # define loss function to be NCE loss function
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                  biases=nce_bias,
                                                  labels=self.target_words,
                                                  inputs=embed,
                                                  num_sampled=self.hps.num_sampled,
                                                  num_classes=self.hps.vocab_size), name='loss')
  def _create_optimizer(self):
    """ Step 5: define optimizer """
    with tf.device('/gpu:0'):
      self.optimizer = tf.train.AdamOptimizer(self.hps.lr).minimize(self.loss,
                                                         global_step=self.global_step)
  
  def _build_nearby_graph(self):
    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    self.nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nemb = tf.nn.l2_normalize(self.embed_matrix, 1)
    nearby_emb = tf.gather(nemb, self.nearby_word)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    self.nearby_val, self.nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, self.hps.vocab_size))
    

  def _build_eval_graph(self):
    """Build the eval graph."""
    # Eval graph

    # Each analogy task is to predict the 4th word (d) given three
    # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
    # predict d=paris.

    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    self.analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    self.analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    self.analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self.embed_matrix, 1)

    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(nemb, self.analogy_a)  # a's embs
    b_emb = tf.gather(nemb, self.analogy_b)  # b's embs
    c_emb = tf.gather(nemb, self.analogy_c)  # c's embs

    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    target = c_emb + (b_emb - a_emb)

    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, nemb, transpose_b=True)

    # For each question (row in dist), find the top 20 words.
    _, self.pred_idx = tf.nn.top_k(dist, 20)

  def predict(self, sess, analogy):
    """ Predict the top 20 answers for analogy questions """
    idx, = sess.run([self.pred_idx], {
        self.analogy_a: analogy[:, 0],
        self.analogy_b: analogy[:, 1],
        self.analogy_c: analogy[:, 2]
    })
    return idx

  def _create_summaries(self):
    with tf.name_scope("summaries"):
      tf.summary.scalar("loss", self.loss)
      tf.summary.histogram("histogram_loss", self.loss)
      # because you have several summaries, we should merge them all
      # into one op to make it easier to manage
      self.summary_op = tf.summary.merge_all()


  def build_graph(self):
    """ Build the graph for our model """
    self._create_input()
    self._create_embedding()
    self._create_loss()
    self._create_optimizer()
    self._build_eval_graph()
    self._build_nearby_graph()
    self._create_summaries()



def train_model(sess, model, batch_gen, index_words, num_train_steps):
  saver = tf.train.Saver()
  # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

  initial_step = 0
  make_dir('checkpoints') # directory to store checkpoints


  
  sess.run(tf.global_variables_initializer()) # initialize all variables
  ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
  # if that checkpoint exists, restore from checkpoint
  if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)

  total_loss = 0.0 # use this to calculate late average loss in the last SKIP_STEP steps
  writer = tf.summary.FileWriter('graph/lr' + str(model.hps.lr), sess.graph)
  initial_step = model.global_step.eval()
  for index in range(initial_step, initial_step + num_train_steps):
    # feed in new dataset  
    if index % model.hps.new_dataset_every == 0:
      try:
          centers, targets = next(batch_gen)
      except StopIteration: # generator has nothing left to generate
          batch_gen = get_batch_gen(index_words, 
                                    model.hps.skip_window, 
                                    model.hps.num_pairs)
          centers, targets = next(batch_gen)
          print('Finished looking at the whole text')
            
      feed = {
          model.centers: centers,
          model.targets: targets
      }
      _ = sess.run(model.iterator.initializer, feed_dict = feed)
      print('feeding in new dataset')
      
      
    loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op])
    writer.add_summary(summary, global_step=index)
    total_loss += loss_batch
    if (index + 1) % model.hps.skip_step == 0:
        print('Average loss at step {}: {:5.1f}'.format(index, total_loss / model.hps.skip_step))
        total_loss = 0.0
        saver.save(sess, 'checkpoints/skip-gram', index)


def get_default_hparams():
    hparams = tf.contrib.training.HParams(
        num_pairs = 10**6,                # number of (center, target) pairs 
                                          # in each dataset instance
        vocab_size = 10000,
        batch_size = 128,
        embed_size = 300,                 # dimension of the word embedding vectors
        skip_window = 3,                  # the context window
        num_sampled = 100,                # number of negative examples to sample
        lr = 0.005,                       # learning rate
        new_dataset_every = 10**4,        # replace the original dataset every ? steps
        num_train_steps = 2*10**5,        # number of training steps for each feed of dataset
        skip_step = 2000
    )
    return hparams

def main():

  hps = get_default_hparams()
  index_words, dictionary, index_dictionary = process_data(hps.vocab_size)
  batch_gen = get_batch_gen(index_words, hps.skip_window, hps.num_pairs)
                                                          
  model = SkipGramModel(hparams = hps)
  model.build_graph()
  
  
  with tf.Session() as sess:
    
    # feed the model with dataset
    centers, targets = next(batch_gen)
    feed = {
        model.centers: centers,
        model.targets: targets
    }
    sess.run(model.iterator.initializer, feed_dict = feed) # initialize the iterator

    train_model(sess, model, batch_gen, index_words, hps.num_train_steps)
      
if __name__ == '__main__':
  main()
