from individual import Individual
import numpy as np


class Generation:
    def __init__(self, generation_size, row_nb, column_nb, cap):
        self.generation_size = generation_size
        self.individuals = [Individual(row_nb, column_nb, cap) for _ in range(0, generation_size)]
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap = cap
        self.score = np.ones(generation_size)*0

    @property
    def score(self):
        return [self.individuals[i].score for i in range(0, self.generation_size)]

    def reset(self):
        map(self.individuals, lambda x: x.reset())

    def fill(self):
        height = self.row_nb // 2
        sub_skeleton = np.random.randint(self.cap, size=(self.generation_size,height, self.column_nb)) + 1
        for i in range(0,self.generation_size):
            self.individuals[i].skeleton[: self.row_nb - height, :] = 0
            self.individuals[i].skeleton[self.row_nb - height:, :] = sub_skeleton[i]

    def check_level(self, level):
        s = 0
        for i in range(0,self.generation_size):
            s += self.individuals[i].skeleton.sum(axis=1)[:level].sum()
        return s == 0

    @property
    def game_over(self):
        return not self.check_level(1)

    def remodeling(self, start_point, end_point):
        for i in range(0, self.generation_size):
            if self[end_point] != self[start_point]:
                self.individuals[i].update_dashboard(self.individuals[i][start_point] + self.individuals[i][end_point])
                self.individuals[i].skeleton[end_point] += self.individuals[i][start_point]
            else:
                self.individuals[i].update_dashboard(2 * self.individuals[i][start_point])
            self.individuals[i].skeleton[start_point] = 0

    def refill(self, not_equal):
        for i in range (0, self.generation_size):
            if not_equal[i]:
                column = np.random.randint(0, self.column_nb - 1)
                values = [1, 1, 2, 2, self.cap, self.cap - 1]
                uniform = np.random.randint(0, 5)
                value = values[uniform]
                self.individuals[i].skeleton[0, column] = value

    def gravity(self):
        for i in range(0, self.generation_size):
            transposed = self.individuals[i].skeleton.transpose().copy()
            self.individuals[i].skeleton = np.array([self.individuals[i].gravity_line(line) for line in transposed]).transpose()

    def one_play(self):
        start_pt, end_pt = self.get_move()
        not_equal = [self.individuals[i].skeleton[start_pt] != self.individuals[i].skeleton[end_pt] for i in range(0, self.generation_size)]
        self.remodeling(start_pt, end_pt)
        self.refill(not_equal)
        self.gravity()

    def play(self):
        self.reset()
        self.fill()
        while not self.game_over:
            self.one_play()
        return self.score
