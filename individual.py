import copy
import json

import numpy as np

from UnderTheLine.kernel import Kernel
from neuralNetwork import NeuralNetwork


class Individual(Kernel):
    def __init__(self, row_nb, column_nb, cap, intelligence=None):
        super().__init__(row_nb=row_nb, column_nb=column_nb, cap=cap, a=0)
        self.intelligence = intelligence or NeuralNetwork(row_nb, column_nb)
        self.move_mapper = self.build_move_mapper()

    def build_move_mapper(self):
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
        self.reset()
        self.fill()
        while not self.game_over:
            self.one_play()
        return self.score

    def one_play(self):
        start_pt, end_pt = self.get_move()
        not_equal = self.skeleton[start_pt] != self.skeleton[end_pt]
        self.remodeling(start_pt, end_pt)
        if not_equal:
            self.refill()
        self.gravity()

    def in_playground(self, point):
        return -1 < point[0] < self.row_nb and -1 < point[1] < self.column_nb

    def fusible(self, start_point, end_point):
        """Check if the cases (i, j) and (k, l) are fusible.
        :return:"""
        if self.in_playground(start_point) and self.in_playground(end_point):
            condition1 = self[start_point]
            condition2 = self[start_point] + self[end_point] <= self.cap or self[start_point] == self[end_point]
            return condition1 and condition2
        return False

    def get_move(self):
        output = self.neural().tolist()
        map_index = list(map(lambda x: output.index(x), reversed(sorted(output))))
        for index in map_index:
            pts = self.move_mapper[index]
            if self.fusible(pts[0], pts[1]):
                return pts[0], pts[1]
        raise Exception

    @staticmethod
    def euclidian(a, b):
        r = a % b
        return int((a - r) / b), int(r)

    def neural(self):
        return self.intelligence(self.input)

    def mutate(self):
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


if __name__ == '__main__':
    ind = Individual(6, 3, 7)
    # filepath = 'test.json'
    # ind.save(filepath)
    l = list(ind.move_mapper.values())
    n = np.zeros((6,3))
    for j in l:
        n[j[0]] += 1
    print('ok')
