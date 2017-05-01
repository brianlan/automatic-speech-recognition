import glob
import os

import numpy as np
import tensorflow as tf


label_type = 'phn'

num_epochs = 1
batch_size = 5
num_features = 39  # mfcc feature size
num_rnn_hidden = 256
learning_rate = 0.0001
grad_clip = 0.8

if label_type == 'phn':
    num_classes = 62
elif label_type == 'cha':
    num_classes = 28

train_data_dir = '/home/rlan/dataset/timit_lite/mfcc/train'
train_label_dir = '/home/rlan/dataset/timit_lite/label/train/{}'.format(label_type)
# test_data_dir = '/home/rlan/dataset/timit_lite/mfcc/test'
# test_label_dir = '/home/rlan/dataset/timit_lite/label/test/{}'.format(label_type)

TENSORBOARD_LOG_DIR = '/home/rlan/tensorboard_log/automatic-speech-recognition/'


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


def create_batches(data, label, max_seq_length, batch_size, rand_idx):
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
    num_batches = (num_samples - 1) // batch_size + 1
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


def main():
    ##############################################
    #                Load Data
    ##############################################
    train_data = []
    train_label = []
    for fname in glob.glob(os.path.join(train_data_dir, '*.npy')):
        train_data.append(np.load(fname))

    for fname in glob.glob(os.path.join(train_label_dir, '*.npy')):
        train_label.append(np.load(fname))

    ##############################################
    #                Preparation
    ##############################################
    max_seq_length = max([d.shape[1] for d in train_data])

    ##############################################
    #                Build Graph
    ##############################################
    graph = tf.Graph()
    with graph.as_default():
        ##################
        #     INPUT
        ##################
        X_train = tf.placeholder(tf.float32, shape=(max_seq_length, batch_size, num_features), name='X_train')
        y_indices = tf.placeholder(tf.int64, shape=(None, 2))
        y_vals = tf.placeholder(tf.int64)
        y_shape = tf.placeholder(tf.int64, shape=(2, ))
        y_train = tf.cast(tf.SparseTensor(y_indices, y_vals, y_shape), tf.int32)
        seq_lengths = tf.placeholder(tf.int32, shape=(batch_size, ))

        ##################
        #     BGRU
        ##################
        forward_cell = tf.contrib.rnn.GRUCell(num_rnn_hidden, tf.nn.tanh)
        backward_cell = tf.contrib.rnn.GRUCell(num_rnn_hidden, tf.nn.tanh)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                                    backward_cell,
                                                                    inputs=X_train,
                                                                    dtype=tf.float32,
                                                                    sequence_length=seq_lengths,
                                                                    time_major=True)

        brnn_combined_outputs = [tf.reshape(t, shape=(batch_size, num_rnn_hidden)) for t in
                                 tf.split(output_fw + output_bw, max_seq_length, axis=0)]
        ##################
        #     CTC
        ##################
        fc_W = tf.Variable(tf.truncated_normal([num_rnn_hidden, num_classes]), name='fc_W')
        fc_b = tf.Variable(tf.truncated_normal([num_classes]), name='fc_b')

        logits = [tf.matmul(output, fc_W) + fc_b for output in brnn_combined_outputs]
        logits3d = tf.stack(logits)
        loss = tf.reduce_mean(tf.nn.ctc_loss(y_train, logits3d, seq_lengths))
        var_trainable_op = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, var_trainable_op), grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, var_trainable_op))
        predictions = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits3d, seq_lengths, merge_repeated=False)[0][0])
        error_rate = tf.reduce_sum(tf.edit_distance(predictions, y_train, normalize=True))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('error_rate', error_rate)
        merged = tf.summary.merge_all()

    ##############################################
    #                Run TF Session
    ##############################################
    tb_file_writer = tf.summary.FileWriter(TENSORBOARD_LOG_DIR, graph)
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        num_processed_batches = 0
        for epoch in range(num_epochs):
            num_samples = len(train_data)
            batches = create_batches(train_data, train_label, max_seq_length, batch_size, np.random.permutation(num_samples))

            for batch, ((batch_data, batch_seq_lengths), (batch_indices, batch_vals, batch_shape)) in enumerate(batches):
                _, loss, error_rate, summary = sess.run([optimizer, loss, error_rate, merged],
                                                        feed_dict={X_train: batch_data,
                                                                   y_indices: batch_indices,
                                                                   y_vals: batch_vals,
                                                                   y_shape: batch_shape,
                                                                   seq_lengths: batch_seq_lengths})

                num_processed_batches += 1
                print('[epoch: {}, batch: {}] loss = {}, error_rate = {}'.format(epoch, batch, loss, error_rate))
                tb_file_writer.add_summary(summary, num_processed_batches)


if __name__ == '__main__':
    main()
