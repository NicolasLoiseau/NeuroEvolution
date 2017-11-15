import numpy as np
import random

np.random.seed(1)


class NeuralNetwork(object):
    def __init__(self, row_nb, column_nb):
        self.column_nb = column_nb
        self.row_nb = row_nb
        self.input_nb = self.column_nb * self.row_nb
        self.output_nb = self.column_nb * self.row_nb + 4
        self.syn0 = 2 * np.random.random((self.input_nb, self.output_nb)) - 1
        self.syn1 = 2 * np.random.random((self.output_nb, self.output_nb)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, input):
        l1 = self.sigmoid(np.dot(input, self.syn0))
        return self.sigmoid(np.dot(l1, self.syn1))

    def mutate(self):
        if random.randint(0, 1):
            self.syn0 += (2 * np.random.random((self.input_nb, self.output_nb)) - 1) / 10
        else:
            self.syn1 += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 10