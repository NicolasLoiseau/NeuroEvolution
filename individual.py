import copy
import json

import numpy as np

from UnderTheLine.kernel import Kernel
from neuralNetwork import NeuralNetwork


class Individual(Kernel):
    """Artificial intelligence."""

    def __init__(self, row_nb, column_nb, cap, intelligence=None):
        super().__init__(row_nb=row_nb, column_nb=column_nb, cap=cap, a=0)
        self.intelligence = intelligence or NeuralNetwork(row_nb, column_nb)
        self.move_mapper = self.build_move_mapper()

    def build_move_mapper(self):
        """Map the neural network output with moves."""
        move_mapper = dict()

        # up movement
        for i in range(1, self.row_nb):
            for j in range(self.column_nb):
                move_mapper[(i - 1) * self.column_nb + j] = [(i, j), (i - 1, j)]

        N = (self.row_nb - 1) * self.column_nb
        # right movement
        for i in range(1, self.row_nb):
            for j in range(self.column_nb - 1):
                move_mapper[N + (i - 1) * (self.column_nb - 1) + j] = [(i, j), (i, j + 1)]

        M = N + (self.row_nb - 1) * (self.column_nb - 1)
        # left movement
        for i in range(1, self.row_nb):
            for j in range(1, self.column_nb):
                move_mapper[M + (i - 1) * (self.column_nb - 1) + j - 1] = [(i, j), (i, j - 1)]
        return move_mapper

    @property
    def input(self):
        normalized = self.skeleton / np.amax(self.skeleton)
        return normalized.reshape((self.column_nb * self.row_nb,))

    def play(self):
        """Play until the game over."""
        self.reset()
        self.fill()
        while not self.game_over:
            self.one_play()
        return self.score

    def one_play(self):
        """Play one move."""
        start_pt, end_pt = self.get_move()
        not_equal = self.skeleton[start_pt] != self.skeleton[end_pt]
        self.remodeling(start_pt, end_pt)
        if not_equal:
            self.refill()
        self.gravity()

    def fusible(self, start_point, end_point):
        """Check if the cases (i, j) and (k, l) are fusible and (i, j) is not empty."""
        condition1 = self[start_point]
        condition2 = self[start_point] + self[end_point] <= self.cap or self[start_point] == self[end_point]
        return condition1 and condition2

    def get_move(self):
        """Find the first possible move according to the neural network output."""
        sorted_index = np.flip(np.argsort(self.neural()), axis=0)
        for index in sorted_index:
            pts = self.move_mapper[index]
            if self.fusible(pts[0], pts[1]):
                return pts[0], pts[1]
        raise Exception

    def neural(self):
        """The neural network output"""
        return self.intelligence(self.input)

    def mutate(self):
        """Perturb the neural network weights"""
        child = copy.deepcopy(self)
        child.intelligence.mutate()
        return child

    def reset(self):
        self.skeleton = np.zeros((self.row_nb, self.column_nb)).astype(int)
        self.cap = self.cap0
        self.score = 0

    def export(self):
        return {
            'column_nb': self.column_nb,
            'row_nb': self.row_nb,
            'cap': self.cap,
            'intelligence': self.intelligence.export()
        }

    def save(self, filepath):
        with open(filepath, 'w') as outfile:
            json.dump(self.export(), outfile)

