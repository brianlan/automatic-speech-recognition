from collections import namedtuple

import leven
import numpy as np

from .chr_idx_mapping import IDX_MAPPING


SparseTensor = namedtuple('SparseTensor', 'indices vals shape')


def calc_PER(pred, ground_truth, normalize=True, merge_phn=True):
    """Calculates the Phoneme Error Rate based on python package leven, which produce the same results as 
    tf.edit_distance and tf.reduce_mean based calculation
    
    :param pred: tuple with 3 numpy-typed element representing sparse tensor
    :param ground_truth: tuple with 3 numpy-typed element representing sparse tensor
    :param normalize: if True, the distance between sequence will be divided by the length of the ground_truth length
    :param merge_phn: if True, 61 phonemes will be merged into 39 phonemes, then do the distance calculation
    :return: the PER
    """

    pred_seq_list = seq_to_single_char_strings(sparse_tensor_to_seq_list(pred, merge_phn=merge_phn))
    truth_seq_list = seq_to_single_char_strings(sparse_tensor_to_seq_list(ground_truth, merge_phn=merge_phn))

    assert len(truth_seq_list) == len(pred_seq_list)

    distances = []
    for i in range(len(truth_seq_list)):
        dist_i = leven.levenshtein(pred_seq_list[i], truth_seq_list[i])
        if normalize:
            dist_i /= float(len(truth_seq_list[i]))
        distances.append(dist_i)

    return np.mean(distances)


def seq_to_single_char_strings(seq):
    strings = []
    for s in seq:
        strings.append(''.join([chr(65 + p) for p in s]))

    return strings


def sparse_tensor_to_seq_list(sparse_seq, merge_phn=True):
    phonemes_list = []
    it = 0
    num_samples = np.max(sparse_seq.indices, axis=0)[0] + 1
    for n in range(num_samples):
        cur_sample_indices = sparse_seq.indices[sparse_seq.indices[:, 0] == n, 1]

        if len(cur_sample_indices) == 0:
            seq_length = 0
        else:
            seq_length = np.max(cur_sample_indices) + 1

        seq = sparse_seq.vals[it:it+seq_length]
        _seq = [IDX_MAPPING[p] for p in seq] if merge_phn else seq
        phonemes_list.append(_seq)
        it += seq_length

    return phonemes_list
