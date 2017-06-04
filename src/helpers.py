import tensorflow as tf

from error import InvalidRNNCellTypeException, InvalidRNNCellActivationTypeException


class RNNCellHelper(object):
    @classmethod
    def make_cell_fn(cls, cell_type):
        if cell_type == 'gru':
            return tf.contrib.rnn.GRUCell

        if cell_type == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell

        raise InvalidRNNCellTypeException("Supported cell type is gru and lstm, but {!r} is given".format(cell_type))

    @classmethod
    def make_cell_activation_fn(cls, activation_type):
        if activation_type == 'tanh':
            return tf.nn.tanh

        raise InvalidRNNCellActivationTypeException(
            "Supported cell activation type is tanh, but {!r} is given".format(activation_type))
