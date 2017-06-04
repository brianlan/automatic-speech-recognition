import glob
import os
import time
import sys

sys.path.append('..')

import numpy as np
import tensorflow as tf

from utils import SparseTensor, calc_PER
from helpers import RNNCellHelper


label_type = 'phn'

num_epochs = 400
batch_size = 32
num_features = 39  # mfcc feature size
num_rnn_hidden = 256
num_rnn_layers = 4
learning_rate = 0.0001
grad_clip = 1.0

rnn_cell_fn = RNNCellHelper.make_cell_fn('gru')
rnn_cell_activation_fn = RNNCellHelper.make_cell_activation_fn('tanh')

if label_type == 'phn':
    num_classes = 62
elif label_type == 'cha':
    num_classes = 29

train_data_dir = '/home/rlan/dataset/timit_rm_sa/mfcc/train'
train_label_dir = '/home/rlan/dataset/timit_rm_sa/label/train/{}'.format(label_type)
test_data_dir = '/home/rlan/dataset/timit_rm_sa/mfcc/test'
test_label_dir = '/home/rlan/dataset/timit_rm_sa/label/test/{}'.format(label_type)

TENSORBOARD_LOG_DIR = '/home/rlan/tensorboard_log/automatic-speech-recognition/'
CHECKPOINT_DIR = '/home/rlan/model_checkpoints/automatic-speech-recognition/'


def to_sparse_representation(label, batch_idx):
    """Transform a np.array list represented label into sparse representation
    
    :param label: full label data
    :param batch_idx: indices generated for current batch
    :return: a 3-element tuple meets the input criteria of tf.SparseTensor
    """
    indices = []
    vals = []

    for i, idx in enumerate(batch_idx):
        for j, c in enumerate(label[idx]):
            indices.append([i, j])
            vals.append(c)

    shape = [len(batch_idx), np.max(indices, axis=0)[1] + 1]

    return np.array(indices), np.array(vals), np.array(shape)


def create_batches(data, label, max_seq_length, batch_size, rand_idx, mode='train'):
    """Randomly split data into batches according to given batch_size and rand_idx. It is a generator.
    
    :param data: a 3-D np.array, of shape num_samples x seq_length x num_features. 
    :param label: a 2-D np.array, of shape num_samples x label_length.
    :param max_seq_length: global max sequence length, a int value.
    :param batch_size: batch size
    :param rand_idx: split data and label into batches according to the given order described in rand_idx. 
                     it's a 1-D array of size len(data)
    :return: Yields a tuple consists of data and label in a batch.
    """
    num_samples = len(data)
    num_batches = num_samples // batch_size
    for i in xrange(num_batches):
        batch_start_pos = i * batch_size
        batch_end_pos = min((i + 1) * batch_size, num_samples)
        batch_idx = rand_idx[batch_start_pos:batch_end_pos]
        label_in_batch = to_sparse_representation(label, batch_idx)
        data_in_batch = np.zeros((max_seq_length, batch_size, num_features))
        seq_lengths = np.zeros(batch_size)
        for j, idx in enumerate(batch_idx):
            x = data[idx]
            data_in_batch[0:x.shape[1], j, :] = np.reshape(x, (x.shape[1], num_features))
            seq_lengths[j] = x.shape[1]
        yield ((data_in_batch, seq_lengths), label_in_batch)


def normalize(X, miu=None, std=None):
    if miu is None or std is None:
        mode = 'train'
        miu = []
        std = []
    else:
        mode = 'test'

    for f in range(num_features):
        full_features = []

        if mode == 'train':
            for d in X:
                full_features.extend(d[:, f])

            miu.append(np.mean(full_features))
            std.append(np.std(full_features))

        for n in range(len(X)):
            X[n][:, f] = (X[n][:, f] - miu[f]) / std[f]

    return miu, std


def sort_by_length(X, axis=1):
    lengths = [x.shape[axis] for x in X]
    perm = np.argsort(lengths)
    sorted_X = []
    for i in perm:
        sorted_X.append(X[i])

    return sorted_X


def batchnorm_layer(X, offset, scale, is_test, iteration):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(X, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Xbn = tf.nn.batch_normalization(X, m, v, offset, scale, bnepsilon)
    return Xbn, update_moving_everages


def main():
    ##############################################
    #                Load Data
    ##############################################
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for fname in glob.glob(os.path.join(train_data_dir, '*.npy')):
        train_data.append(np.load(fname))

    miu, std = normalize(train_data)
    train_data = sort_by_length(train_data)

    for fname in glob.glob(os.path.join(train_label_dir, '*.npy')):
        train_label.append(np.load(fname))

    for fname in glob.glob(os.path.join(test_data_dir, '*.npy')):
        test_data.append(np.load(fname))

    normalize(test_data, miu=miu, std=std)

    for fname in glob.glob(os.path.join(test_label_dir, '*.npy')):
        test_label.append(np.load(fname))

    ##############################################
    #                Preparation
    ##############################################
    max_seq_length = max([d.shape[1] for d in train_data + test_data])
    num_test_samples = len(test_data)

    (test_data_tensor, test_seq_lengths), (test_label_indices, test_label_vals, test_label_shape) = \
        list(create_batches(test_data,
                            test_label,
                            max_seq_length,
                            num_test_samples,
                            range(num_test_samples)))[0]  # borrow create_batch to transform test data / label

    cur_unixtime = time.time()
    cur_checkpoint_path = os.path.join(CHECKPOINT_DIR, '{:.0f}'.format(cur_unixtime))
    if not os.path.exists(cur_checkpoint_path):
        os.makedirs(cur_checkpoint_path)

    cur_tb_summary_path = os.path.join(TENSORBOARD_LOG_DIR, '{:.0f}'.format(cur_unixtime))
    if not os.path.exists(cur_tb_summary_path):
        os.makedirs(cur_tb_summary_path)

    ##############################################
    #                Build Graph
    ##############################################
    graph = tf.Graph()
    with graph.as_default():
        ##################
        #     INPUT
        ##################
        X_train = tf.placeholder(tf.float32, shape=(max_seq_length, batch_size, num_features), name='X_train')
        X_test = tf.placeholder(tf.float32, shape=(max_seq_length, num_test_samples, num_features), name='X_test')

        y_train_indices = tf.placeholder(tf.int64, shape=(None, 2))
        y_train_vals = tf.placeholder(tf.int64)
        y_train_shape = tf.placeholder(tf.int64, shape=(2, ))
        y_train = tf.cast(tf.SparseTensor(y_train_indices, y_train_vals, y_train_shape), tf.int32)

        y_test_indices = tf.placeholder(tf.int64, shape=(None, 2))
        y_test_vals = tf.placeholder(tf.int64)
        y_test_shape = tf.placeholder(tf.int64, shape=(2,))
        y_test = tf.cast(tf.SparseTensor(y_test_indices, y_test_vals, y_test_shape), tf.int32)

        seq_lengths_train = tf.placeholder(tf.int32, shape=(batch_size, ))
        seq_lengths_test = tf.placeholder(tf.int32, shape=(num_test_samples,))

        ##################
        #     BGRU
        ##################
        def brnn_layer(fw_cell, bw_cell, inputs, seq_lengths, scope=None):
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                        bw_cell,
                                                                        inputs=inputs,
                                                                        dtype=tf.float32,
                                                                        sequence_length=seq_lengths,
                                                                        time_major=True,
                                                                        scope=scope)

            brnn_combined_outputs = output_fw + output_bw
            return brnn_combined_outputs

        def multi_brnn_layer(inputs, seq_lengths, num_layers=1):
            inner_outputs = inputs
            for n in range(num_layers):
                forward_cell = rnn_cell_fn(num_rnn_hidden, activation=rnn_cell_activation_fn)
                backward_cell = rnn_cell_fn(num_rnn_hidden, activation=rnn_cell_activation_fn)
                inner_outputs = brnn_layer(forward_cell, backward_cell, inner_outputs, seq_lengths, 'brnn_{}'.format(n))

            return inner_outputs

        with tf.variable_scope('rlan') as scope:
            brnn_outputs_train = multi_brnn_layer(X_train, seq_lengths_train, num_layers=num_rnn_layers)
            brnn_outputs_train = [tf.reshape(t, shape=(batch_size, num_rnn_hidden)) for t in
                                  tf.split(brnn_outputs_train, max_seq_length, axis=0)]

            # scope.reuse_variables()
            #
            # brnn_outputs_test = multi_brnn_layer(X_test, seq_lengths_test, num_layers=num_rnn_layers)
            # brnn_outputs_test = [tf.reshape(t, shape=(num_test_samples, num_rnn_hidden)) for t in
            #                      tf.split(brnn_outputs_test, max_seq_length, axis=0)]

        # TODO: Learning Rate Decay
        # TODO: Use better initialization
        # TODO: Add BatchNorm
        # TODO: Joint LM-acoustic Model
        # TODO: Implement Demo (audio => text)
        ##################
        #     CTC
        ##################
        # with tf.variable_scope("ctc") as scope:
        with tf.name_scope('fc-layer'):
            fc_W = tf.get_variable('fc_W', initializer=tf.truncated_normal([num_rnn_hidden, num_classes]))
            fc_b = tf.get_variable('fc_b', initializer=tf.truncated_normal([num_classes]))

            logits_train = [tf.matmul(output, fc_W) + fc_b for output in brnn_outputs_train]

        logits3d_train = tf.stack(logits_train)

        # logits_test = [tf.matmul(output, fc_W) + fc_b for output in brnn_outputs_test]
        # logits3d_test = tf.stack(logits_test)

        loss_train = tf.reduce_mean(tf.nn.ctc_loss(y_train, logits3d_train, seq_lengths_train))
        # loss_test = tf.reduce_mean(tf.nn.ctc_loss(y_test, logits3d_test, seq_lengths_test))

        var_trainable_op = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss_train, var_trainable_op), grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, var_trainable_op))

        pred_train = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d_train, seq_lengths_train, merge_repeated=False)[0][0])
        err_rate_train = tf.reduce_mean(tf.edit_distance(pred_train,
                                                         y_train,
                                                         normalize=True))

        # err_rate_train = tf.reduce_mean(tf.edit_distance(tf.SparseTensor(pred_train.indices,
        #                                                                  tf.map_fn(phn_map_func, pred_train.values),
        #                                                                  pred_train.dense_shape),
        #                                                  tf.SparseTensor(y_train.indices,
        #                                                                  tf.map_fn(phn_map_func, y_train.values),
        #                                                                  y_train.dense_shape),
        #                                                  normalize=True))

        # pred_test = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d_test, seq_lengths_test, merge_repeated=False)[0][0])
        # err_rate_test = tf.reduce_mean(tf.edit_distance(pred_test, y_test, normalize=True))

        # tf.summary.scalar('loss_train', loss_train)
        # tf.summary.scalar('loss_test', loss_test)
        tf.summary.scalar('err_rate_train', err_rate_train)
        # tf.summary.scalar('err_rate_test', err_rate_test)
        merged = tf.summary.merge_all()

    ##############################################
    #                Run TF Session
    ##############################################
    tb_file_writer = tf.summary.FileWriter(cur_tb_summary_path, graph)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
        tf.global_variables_initializer().run()
        num_processed_batches = 0
        for epoch in range(num_epochs):
            num_samples = len(train_data)
            perm = range(num_samples) if epoch == 0 else np.random.permutation(num_samples)
            batches = create_batches(train_data, train_label, max_seq_length, batch_size, perm)

            for batch, ((batch_data, batch_seq_lengths), (batch_indices, batch_vals, batch_shape)) in enumerate(batches):
                _, batch_loss, batch_err_rate, batch_pred, summary = \
                    sess.run([optimizer, loss_train, err_rate_train, pred_train, merged],
                             feed_dict={X_train: batch_data,
                                        y_train_indices: batch_indices,
                                        y_train_vals: batch_vals,
                                        y_train_shape: batch_shape,
                                        seq_lengths_train: batch_seq_lengths})
                # _, batch_loss, batch_err_rate, batch_pred, cur_test_loss, cur_test_err_rate, cur_test_pred, summary = \
                #     sess.run([optimizer, loss_train, err_rate_train, pred_train, loss_test, err_rate_test, pred_test, merged],
                #              feed_dict={X_train: batch_data,
                #                         y_train_indices: batch_indices,
                #                         y_train_vals: batch_vals,
                #                         y_train_shape: batch_shape,
                #                         seq_lengths_train: batch_seq_lengths,
                #                         X_test: test_data_tensor,
                #                         y_test_indices: test_label_indices,
                #                         y_test_vals: test_label_vals,
                #                         y_test_shape: test_label_shape,
                #                         seq_lengths_test: test_seq_lengths})

                num_processed_batches += 1
                # err_test = {:.4f} (phn_merged: {:.4f}
                print('[epoch: {}, batch: {}] err_train = {:.4f} (phn_merged: {:.4f}))'.format(
                    epoch,
                    batch,
                    batch_err_rate,
                    calc_PER(SparseTensor(batch_pred.indices, batch_pred.values, batch_pred.dense_shape),
                             SparseTensor(batch_indices, batch_vals, batch_shape)),
                    # cur_test_err_rate,
                    # calc_err_rate(seq_to_single_char_strings(reduce_phoneme(cur_test_pred.indices, cur_test_pred.values, cur_test_pred.dense_shape)),
                    #               seq_to_single_char_strings(reduce_phoneme(test_label_indices, test_label_vals, test_label_shape)))
                ))
                tb_file_writer.add_summary(summary, num_processed_batches)

            saver.save(sess, os.path.join(cur_checkpoint_path, 'model'), global_step=epoch)

            # num_samples_in_batch = max(batch_indices[:, 0])
            # for sample_id in range(num_samples_in_batch):
            #     ground_truth_label_seq = batch_vals[batch_indices[:, 0] == sample_id]
            #     pred_label_seq = batch_pred.values[batch_pred.indices[:, 0] == sample_id]
            #     print('-' * 120)
            #     print('Ground Truth: {}'.format(label_indices_to_characters(ground_truth_label_seq, label_type)))
            #     print('  Prediction: {}'.format(label_indices_to_characters(pred_label_seq, label_type)))

            # for sample_id in range(num_test_samples):
            #     ground_truth_label_seq = test_label_vals[test_label_indices[:, 0] == sample_id]
            #     pred_label_seq = cur_test_pred.values[cur_test_pred.indices[:, 0] == sample_id]
            #     print('-' * 120)
            #     print('Ground Truth: {}'.format(label_indices_to_characters(ground_truth_label_seq, label_type)))
            #     print('  Prediction: {}'.format(label_indices_to_characters(pred_label_seq, label_type)))
            #
            # print('-' * 120)


if __name__ == '__main__':
    main()
