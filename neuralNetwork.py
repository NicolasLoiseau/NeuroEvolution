import numpy as np


class NeuralNetwork(object):
    def __init__(self, row_nb, column_nb, syn0=None, syn1=None, syn2=None):
        self.column_nb = column_nb
        self.row_nb = row_nb
        self.input_nb = self.column_nb * self.row_nb
        self.output_nb = (self.row_nb - 1) * (3 * self.column_nb - 2)
        self.syn0 = syn0 or 2 * np.random.random((self.input_nb, self.output_nb)) - 1
        self.syn1 = syn1 or 2 * np.random.random((self.output_nb, self.output_nb)) - 1
        self.syn2 = syn2 or 2 * np.random.random((self.output_nb, self.output_nb)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, input):
        l1 = self.sigmoid(np.dot(input, self.syn0))
        l2 = self.sigmoid(np.dot(l1, self.syn1))
        return self.sigmoid(np.dot(l2, self.syn2))

    def mutate(self):
        self.syn0 += (2 * np.random.random((self.input_nb, self.output_nb)) - 1) / 10
        self.syn1 += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 10
        self.syn2 += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 10

    def export(self):
        return {
            'column_nb': self.column_nb,
            'row_nb': self.row_nb,
            'syn0': self.syn0.tolist(),
            'syn1': self.syn1.tolist(),
            'syn2': self.syn2.tolist()
        }
