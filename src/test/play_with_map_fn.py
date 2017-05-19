import glob
import os
import time
import sys

sys.path.append('..')

import numpy as np
import tensorflow as tf


def main():
    graph = tf.Graph()

    def if_0():
        return tf.Variable(10)

    def if_1():
        return tf.Variable(20)

    def func(x):
        return tf.cond(tf.equal(x, tf.Variable(0)), if_0, if_1)

    with graph.as_default():
        mapping = {tf.constant(0, dtype=tf.int32): tf.constant(10, dtype=tf.int32),
                   tf.constant(1, dtype=tf.int32): tf.constant(20, dtype=tf.int32)}
        a = tf.Variable([0, 1, 1, 0], dtype=tf.int32)
        b = tf.map_fn(func, a)
        c = tf.equal(a[0], tf.Variable(0))
        d = tf.equal(a[0], tf.Variable(1))
        e = tf.where(
            tf.Variable([True, False, True]),
            x=tf.Variable([100, 200, 300]),
            y=tf.Variable([1, 2, 3])
        )
        f = tf.reduce_mean(tf.edit_distance(
            tf.cast(tf.SparseTensor(
                tf.Variable([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=tf.int64),
                tf.Variable([1, 3, 5, 10, 12, 13], dtype=tf.int64),
                tf.Variable([2, 3], dtype=tf.int64)
            ), tf.int32),
            tf.cast(tf.SparseTensor(
                tf.Variable([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], dtype=tf.int64),
                tf.Variable([2, 3, 7, 12, 13, 14], dtype=tf.int64),
                tf.Variable([2, 3], dtype=tf.int64)
            ), tf.int32),
            normalize=True
        ))

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        bb, cc, dd, ee, ff = sess.run([b, c, d, e, f])

    from utils import calc_err_rate
    seq_a = ['ace', 'hjk']
    seq_b = ['bcg', 'jkl']
    er = calc_err_rate(seq_a, seq_b, normalize=True)

    pass


if __name__ == '__main__':
    main()
