import numpy as np

from ..brnn_ctc import to_sparse_representation


def test_to_sparse_representation():
    label = [np.array([1, 2, 10, 9]),
             np.array([2, 2, 2]),
             np.array([5, 6, 7, 8, 9, 0, 0, 4]),
             np.array([1])]

    st_labels = to_sparse_representation(label, range(4))

    assert len(st_labels)
    assert st_labels[0].shape[0] == st_labels[1].shape[0]
    assert np.array_equal(st_labels[0], np.array([[0, 0],
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
                                                  [3, 0]]))
    assert np.array_equal(st_labels[1], np.array([1, 2, 10, 9, 2, 2, 2, 5, 6, 7, 8, 9, 0, 0, 4, 1]))
    assert np.array_equal(st_labels[2], np.array([4, 8]))
