import glob
import random
import os

import numpy as np
import tensorflow as tf


num_epochs = 1
batch_size = 5
num_features = 39  # mfcc feature size
label_type = 'cha'
train_data_dir = '/home/rlan/dataset/timit_lite/mfcc/train'
train_label_dir = '/home/rlan/dataset/timit_lite/label/train/{}'.format(label_type)
# test_data_dir = '/home/rlan/dataset/timit_lite/mfcc/test'
# test_label_dir = '/home/rlan/dataset/timit_lite/label/test/{}'.format(label_type)


def create_sparse_tensor_labels(label, batch_idx):
    return [0, 1, 2]


def create_batches(data, label, batch_size, rand_idx):
    """Randomly split data into batches according to given batch_size and rand_idx. It is a generator.
    Args:
        data: a 3-D np.array, of shape num_samples x seq_length x num_features. 
        label: a 2-D np.array, of shape num_samples x label_length.
        batch_size: batch size
        rand_idx: split data and label into batches according to the given order described in rand_idx. 
                  it's a 1-D array of size len(data)

    Returns:
        Yields a tuple consists of data and label in a batch.
    """
    num_samples = len(data)
    max_seq_length = max([d.shape[1] for d in data])
    num_batches = num_samples // batch_size + 1
    for i in xrange(num_batches):
        batch_start_pos = i * batch_size
        batch_end_pos = min((i + 1) * batch_size, num_samples)
        batch_idx = rand_idx[batch_start_pos:batch_end_pos]
        label_in_batch = create_sparse_tensor_labels(label, batch_idx)
        data_in_batch = np.zeros((batch_size, max_seq_length, num_features))
        for i, idx in enumerate(batch_idx):
            x = data[idx]
            data_in_batch[i, 0:x.shape[1], :] = np.reshape(x, (x.shape[1], num_features))
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

    ##############################################
    #                Build Graph
    ##############################################
    pass
    ##############################################
    #                Run TF Session
    ##############################################
    for epoch in range(num_epochs):
        num_samples = len(train_data)
        batches = create_batches(train_data, train_label, batch_size, np.random.permutation(num_samples))

        for (batch_data, batch_label) in batches:
            print(batch_data.shape)
            print(batch_label)



if __name__ == '__main__':
    main()