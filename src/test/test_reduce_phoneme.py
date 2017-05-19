import numpy as np
from ..utils import calc_err_rate, reduce_phoneme, seq_to_single_char_strings


def test_reduce_phoneme():
    indices = np.array([[0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [2, 0],
                        [2, 1],
                        [2, 2],
                        [2, 3],
                        [2, 4],
                        [2, 5],
                        [2, 6],
                        [2, 7],
                        [3, 0]])
    vals = np.array([1, 2, 7, 7, 2, 2, 2, 5, 6, 7, 5, 4, 0, 0, 4, 1])
    shape = np.array([4, 8])

    # ['ae', 1 5 22 22
    #  'ax',
    #  'er',
    #  'er',
    #
    #  'ax',  5 5 5
    #  'ax',
    #  'ax',
    #
    #  'ax',5 5 22 5 4 3 3  4
    #  'ax',
    #  'er',
    #  'ax',
    #  'aw',
    #  'ao',
    #  'ao',
    #  'aw',
    #
    #  'ae'] 1

    reduced_seq = reduce_phoneme(indices, vals, shape)
    assert reduced_seq == [[1, 5, 22, 22], [5, 5, 5], [5, 5, 22, 5, 4, 3, 3, 4], [1]]
    assert seq_to_single_char_strings(reduced_seq) == ['BFWW', 'FFF', 'FFWFEDDE', 'B']
