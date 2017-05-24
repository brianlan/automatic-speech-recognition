import leven
import numpy as np


PHN_LOOKUP_TABLE = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
                    'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
                    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
                    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

CHA_LOOKUP_TABLE = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', "'", ]

KAIFU_LEE_PHN_MAPPING = {'iy': 'iy', 'ix': 'ix', 'ih': 'ix', 'eh': 'eh', 'ae': 'ae', 'ax': 'ax', 'ah': 'ax',
                         'ax-h': 'ax', 'uw': 'uw', 'ux': 'uw', 'uh': 'uh', 'ao': 'ao', 'aa': 'ao', 'ey': 'ey',
                         'ay': 'ay', 'oy': 'oy', 'aw': 'aw', 'ow': 'ow', 'er': 'er', 'axr': 'er', 'l': 'l', 'el': 'l',
                         'r': 'r', 'w': 'w', 'y': 'y', 'm': 'm', 'em': 'm', 'n': 'n', 'en': 'n', 'nx': 'n', 'ng': 'ng',
                         'eng': 'ng', 'v': 'v', 'f': 'f', 'dh': 'dh', 'th': 'th', 'z': 'z', 's': 's', 'zh': 'zh',
                         'sh': 'zh', 'jh': 'jh', 'ch': 'ch', 'b': 'b', 'p': 'p', 'd': 'd', 'dx': 'dx', 't': 't',
                         'g': 'g', 'k': 'k', 'hh': 'hh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#', 'dcl': 'h#', 'tcl': 'h#',
                         'gcl': 'h#', 'kcl': 'h#', 'q': 'h#', 'epi': 'h#', 'pau': 'h#', 'h#': 'h#'}

KAIFU_LEE_IDX_MAPPING = {0: 3, 1: 1, 2: 5, 3: 3, 4: 4, 5: 5, 6: 5, 7: 22, 8: 8, 9: 9, 10: 27, 11: 11, 12: 12, 13: 27,
                         14: 14, 15: 15, 16: 16, 17: 36, 18: 37, 19: 38, 20: 39, 21: 27, 22: 22, 23: 23, 24: 24, 25: 25,
                         26: 27, 27: 27, 28: 28, 29: 28, 30: 31, 31: 31, 32: 32, 33: 33, 34: 34, 35: 27, 36: 36, 37: 37,
                         38: 38, 39: 39, 40: 38, 41: 41, 42: 42, 43: 43, 44: 27, 45: 27, 46: 27, 47: 47, 48: 48, 49: 60,
                         50: 50, 51: 27, 52: 52, 53: 53, 54: 54, 55: 54, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60}


# for k, v in KAIFU_LEE_PHN_MAPPING.items():
#     KAIFU_LEE_IDX_MAPPING[PHN_LOOKUP_TABLE.index(k)] = PHN_LOOKUP_TABLE.index(v)
def label_indices_to_characters(labels, label_type='phn'):
    if label_type == 'phn':
        lookup = PHN_LOOKUP_TABLE
        chars = ' '.join([lookup[l] for l in labels])
    elif label_type == 'cha':
        lookup = CHA_LOOKUP_TABLE
        chars = ''.join([lookup[l] for l in labels])
    else:
        raise ValueError('Label type {!r} is not supported.'.format(label_type))

    return chars


def phn_map_func(ori_phn_number):
    return PHN_LOOKUP_TABLE.index(KAIFU_LEE_PHN_MAPPING[PHN_LOOKUP_TABLE[ori_phn_number]])


def calc_err_rate(pred, ground_truth, normalize=True):
    assert len(ground_truth) == len(pred)
    distances = []
    for i in range(len(ground_truth)):
        dist_i = leven.levenshtein(pred[i], ground_truth[i])
        if normalize:
            dist_i /= float(len(ground_truth[i]))
        distances.append(dist_i)

    return np.mean(distances)


def seq_to_single_char_strings(seq):
    strings = []
    for s in seq:
        strings.append(''.join([chr(65 + p) for p in s]))

    return strings


def sparse_repr_to_2d_list(indices, vals, shape):
    phonemes_list = []
    it = 0
    num_samples = np.max(indices, axis=0)[0] + 1
    for n in range(num_samples):
        seq_length = np.max(indices[indices[:, 0] == n, 1]) + 1
        phonemes_list.append(vals[it:it + seq_length])
        it += seq_length

    return phonemes_list


def reduce_phoneme(indices, vals, shape):
    phonemes_list = []
    it = 0
    num_samples = np.max(indices, axis=0)[0] + 1
    for n in range(num_samples):
        cur_sample_indices = indices[indices[:, 0] == n, 1]

        if len(cur_sample_indices) == 0:
            seq_length = 0
        else:
            seq_length = np.max(cur_sample_indices) + 1

        seq = vals[it:it+seq_length]
        reduced_seq = [KAIFU_LEE_IDX_MAPPING[p] for p in seq]
        phonemes_list.append(reduced_seq)
        it += seq_length

    return phonemes_list
