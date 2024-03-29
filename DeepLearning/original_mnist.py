# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#Matched to architecture in cnn_mnist.py 2 CL and 2 PL with SGD with learning rate of 0.001
def main(_):
  # Import data
  sess = tf.InteractiveSession()
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))  
  y = tf.nn.softmax(tf.matmul(x, W))
  y_ = tf.placeholder(tf.float32, [None, 10])
  sess.run(tf.global_variables_initializer())
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  W_conv1 = weight_variable([5, 5, 1, 32])
  
  x_image = tf.reshape(x, [-1,28,28,1])
  
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))
  h_pool1 = max_pool_2x2(h_conv1)
  W_conv2 = weight_variable([5, 5, 32, 64])
  
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) )
  h_pool2 = max_pool_2x2(h_conv2)
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1))

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  

  y_conv = tf.matmul(h_fc1_drop, W_fc2) 
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy)
  # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # tf.summary.scalar('accuracy', accuracy)


  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/original', sess.graph)

  sess.run(tf.global_variables_initializer())
  def feed_dict():
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(5000):
    batch = mnist.train.next_batch(100)
    if i%100 == 99:
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict())
      train_writer.add_summary(summary, i)
      # train_accuracy = accuracy.eval(feed_dict={
      #     x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, testing accuracy %g"%(i, acc))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})

  print("Final test accuracy %g"%accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)