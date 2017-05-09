import glob
import os
import sys

sys.path.append('..')

import numpy as np
import tensorflow as tf

##############################################
#                Build Graph
##############################################
graph = tf.Graph()
with graph.as_default():
    y_indices = np.array(
        [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
         [2, 0], [2, 1], [2, 2], [3, 0]]
    )
    y_vals = np.array([1, 1, 2, 2, 3, 2, 1, 4, 4, 1, 2, 3, 4])
    y_shape = np.array([4, 5])

    y = tf.cast(tf.SparseTensor(y_indices, y_vals, y_shape), tf.int32)

    pred_indices = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [3, 0], [3, 1]])
    pred_vals = np.array([2, 2, 4, 5, 6, 5])
    pred_shape = np.array([4, 2])
    pred = tf.cast(tf.SparseTensor(pred_indices, pred_vals, pred_shape), tf.int32)

    dist = tf.edit_distance(pred, y, normalize=True)
    final_err_sum = tf.reduce_sum(dist)
    final_err_mean = tf.reduce_mean(dist)

##############################################
#                Run TF Session
##############################################
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    cur_dist, cur_err_sum, cur_err_mean = sess.run([dist, final_err_sum, final_err_mean])
    # _, cur_dist, cur_err = sess.run([dist, final_err],
    #                                 feed_dict={
    #                                     y_indices: np.array(
    #                                         [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
    #                                          [2, 0], [2, 1], [2, 2]]
    #                                     ),
    #                                     y_vals: np.array([1, 1, 2, 2, 3, 2, 1, 4, 4, 1, 2, 3]),
    #                                     y_shape: np.array([3, 5]),
    #                                     pred_indices: np.array([[0, 0], [0, 1], [1, 0], [2, 0]]),
    #                                     pred_vals: np.array([2, 2, 4, 5]),
    #                                     pred_shape: np.array([3, 2])})

    pass
