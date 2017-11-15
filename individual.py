import numpy as np

from UnderTheLine.kernel import Kernel
from neuralNetwork import NeuralNetwork


direction_mapper = {3: (1, 0), 2: (0, 1), 0: (-1, 0), 1: (0, -1)}


class Indivudual(Kernel):

    def __init__(self, row_nb, column_nb, cap, a=0):
        super().__init__(row_nb=row_nb, column_nb=column_nb, cap=cap, a=a)
        self.intelligence = NeuralNetwork(row_nb, column_nb)

    @property
    def input(self):
        normalized = self.skeleton / np.amax(self.skeleton)
        return normalized.reshape((self.column_nb * self.row_nb,))

    def play(self):
        self.fill()
        while not self.game_over:
            self.one_play()
        return self.score

    def one_play(self):
        start_pt, end_pt = self.get_move()
        self.remodeling(start_pt, end_pt)
        if self.skeleton[start_pt] != self.skeleton[end_pt]:
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
        output = self.neural()
        playground = output[:-4].tolist()
        map_index_pg = list(map(lambda x: playground.index(x), reversed(sorted(playground))))
        direction = output[-4:].tolist()
        map_index_dir = list(map(lambda x: direction.index(x), reversed(sorted(direction))))
        for index_pg in map_index_pg:
            for index_dir in map_index_dir:
                start_pt = self.euclidian(index_pg, self.column_nb)
                to_add = direction_mapper[index_dir]
                end_pt = start_pt[0] + to_add[0], start_pt[1] + to_add[1]
                if self.fusible(start_pt, end_pt):
                    return start_pt, end_pt
        raise Exception

    @staticmethod
    def euclidian(a, b):
        r = a % b
        return int((a - r) / b), int(r)

    def neural(self):
        return self.intelligence(self.input)
