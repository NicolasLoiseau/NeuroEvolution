import numpy as np


class _NeuralNetwork(object):
    """
    bases of NN classes
    """
    def __init__(self, row_nb, column_nb):
        self.column_nb = column_nb
        self.row_nb = row_nb
        self.input_nb = self.column_nb * self.row_nb
        self.output_nb = self.column_nb * self.row_nb  # Do not take into account : will be redefined in child classes
        self.syn0 = 2 * np.random.random((self.input_nb, self.output_nb)) - 1
        self.syn1 = 2 * np.random.random((self.output_nb, self.output_nb)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, input):
        l1 = self.sigmoid(np.dot(input, self.syn0))
        return self.sigmoid(np.dot(l1, self.syn1))

    def mutate(self):
        self.syn0 += (2 * np.random.random((self.input_nb, self.output_nb)) - 1) / 10
        self.syn1 += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 10

    def export(self):
        return {
            'column_nb': self.column_nb,
            'row_nb': self.row_nb,
            'syn0': self.syn0.tolist(),
            'syn1': self.syn1.tolist()
        }


class NeuralNetwork(_NeuralNetwork):
    """
    output layer structure : number of cases + 4 (directions)
    2 outputs taken from output layer : (case to move, direction)
    """
    __name__ = 'NN_add'

    def __init__(self, row_nb, column_nb):
        super().__init__(row_nb=row_nb, column_nb=column_nb)
        self.output_nb = self.column_nb * self.row_nb + 4


class NeuralNetwork_2(_NeuralNetwork):
    """
    output layer structure : number of cases * 4 (directions)
    1 output taken from output layer : case to move in the given direction
    """
    __name__ = 'NN_mul'

    def __init__(self, row_nb, column_nb):
        super().__init__(row_nb=row_nb, column_nb=column_nb)
        self.output_nb = self.column_nb * self.row_nb * 4

