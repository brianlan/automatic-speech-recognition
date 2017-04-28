import glob
import random
import os

import numpy as np
import tensorflow as tf


num_epochs = 1
batch_size = 5
num_features = 39  # mfcc feature size
label_type = 'phn'
train_data_dir = '/home/rlan/dataset/timit_lite/mfcc/train'
train_label_dir = '/home/rlan/dataset/timit_lite/label/train/{}'.format(label_type)
# test_data_dir = '/home/rlan/dataset/timit_lite/mfcc/test'
# test_label_dir = '/home/rlan/dataset/timit_lite/label/test/{}'.format(label_type)


def to_sparse_representation(label, batch_idx):
    """Transform a np.array list represented label into sparse representation
    
    :param label: full label data
    :param batch_idx: indices generated for current batch
    :return: a 3-element tuple meets the input criteria of tf.sparse_tensor
    """
    indices = []
    vals = []

    for i, idx in enumerate(batch_idx):
        for j, c in enumerate(label[idx]):
            indices.append([i, j])
            vals.append(c)

    shape = [len(batch_idx), np.max(indices, axis=0)[1] + 1]

    return (np.array(indices), np.array(vals), np.array(shape))


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
        for i, idx in enumerate(batch_idx):
            x = data[idx]
            data_in_batch[0:x.shape[1], i, :] = np.reshape(x, (x.shape[1], num_features))
        yield (data_in_batch, label_in_batch)

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
    pass
    ##############################################
    #                Run TF Session
    ##############################################
    for epoch in range(num_epochs):
        num_samples = len(train_data)
        batches = create_batches(train_data, train_label, max_seq_length, batch_size, np.random.permutation(num_samples))

        for (batch_data, batch_label_sparse_repr) in batches:
            print(batch_data.shape)
            print(batch_label_sparse_repr[2])


if __name__ == '__main__':
    main()