import numpy as np

from gravity import gravity_gpu
from neuralNetworkGPU import NeuralNetworkGPU
from timeit import timeit

class Generation:
    def __init__(self, generation_size, row_nb, column_nb, cap, skeletons_gpu):
        self.generation_size = generation_size
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap0 = cap
        self.skeletons = np.zeros((self.generation_size, self.row_nb, self.column_nb)).astype(int)
        self.scores = np.zeros(self.generation_size)
        self.move_mapper = self.build_move_mapper()
        self.intelligence = NeuralNetworkGPU(self.generation_size, self.row_nb, self.column_nb, skeletons_gpu)
        self.moves = np.zeros((self.generation_size, 2, 2))
        self.skeletons_gpu = skeletons_gpu

    @property
    def caps(self):
        return np.ones(self.generation_size) * self.cap0 + self.scores // 100

    def reset(self):
        self.skeletons = np.zeros((self.generation_size, self.row_nb, self.column_nb)).astype(int)
        self.scores = np.zeros(self.generation_size)

    def fill(self):
        height = self.row_nb // 2
        sub_skeleton = np.random.randint(self.cap0, size=(self.generation_size, height, self.column_nb)) + 1
        self.skeletons[:, self.row_nb - height:, :] = sub_skeleton

    @property
    def game_over(self):
        return self.skeletons.sum(axis=2)[:, 0] != 0

    def remodeling(self, play_range):
        for i in play_range:
            if self.skeletons[i][self.end_pt(i)] != self.skeletons[i][self.start_pt(i)]:
                self.scores[i] += (self.skeletons[i][self.start_pt(i)] + self.skeletons[i][self.end_pt(i)])
                self.skeletons[i][self.end_pt(i)] += self.skeletons[i][self.start_pt(i)]
            else:
                self.scores[i] += (2 * self.skeletons[i][self.start_pt(i)])
            self.skeletons[i][self.start_pt(i)] = 0

    def refill(self, not_equal, play_range):
        for i in play_range:
            if not_equal[i]:
                column = np.random.randint(0, self.column_nb - 1)
                values = [1, 1, 2, 2, self.caps[i], self.caps[i] - 1]
                uniform = np.random.randint(0, 5)
                value = values[uniform]
                self.skeletons[i][0, column] = value

    def gravity(self):
        self.skeletons = gravity_gpu(self.skeletons)

    def one_play(self):
        self.moves = self.get_move()
        not_equal = [self.skeletons[i][self.start_pt(i)] != self.skeletons[i][self.end_pt(i)] for i in
                     range(self.generation_size)]
        play_range = np.where(np.invert(self.game_over))[0]
        play_vec = self.game_over.astype(int)
        self.remodeling(play_range)
        self.refill(not_equal, play_range)
        self.gravity()

    def play(self):
        self.reset()
        self.fill()
        while not all(self.game_over):
            self.one_play()
        return self.scores

    def old_get_move(self):
        start_points = np.zeros((self.generation_size, 2))
        end_points = np.zeros((self.generation_size, 2))
        for i in range(self.generation_size):
            start_point, end_point = self.individuals[i].get_move()
            start_points[i] = start_point
            end_points[i] = end_point
        return start_points.astype(int), end_points.astype(int)

    def build_move_mapper(self):
        """Construct the dictionary to map the neural network output with moves."""
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

    def get_move(self):
        """Return the moves choosen by the intelligence for each individual."""
        sorted_index = np.flip(np.apply_along_axis(np.argsort, 2, self.neural()), 2)
        u, v, w = sorted_index.shape
        return np.array([self.best_move(i, index) for i, index in zip(range(u), sorted_index.reshape(u, w))])

    def best_move(self, i, sorted_index):
        """Find the first possible move according to the neural network output."""
        for index in sorted_index:
            pts = self.move_mapper[index]
            if self.fusible(i, pts[0], pts[1]):
                return pts[0], pts[1]
        raise Exception

    def fusible(self, i, start_point, end_point):
        """Check if the cases (i, j) and (k, l) are fusible and (i, j) is not empty."""
        condition1 = self.skeletons[i][start_point]
        condition2 = self.skeletons[i][start_point] + self.skeletons[i][end_point] <= self.caps[i]
        condition3 = self.skeletons[i][start_point] == self.skeletons[i][end_point]
        return condition1 and (condition2 or condition3)

    def neural(self):
        """The neural network output"""
        nn_input = self.skeletons.reshape((self.generation_size, 1, self.row_nb * self.column_nb))
        return self.intelligence(nn_input)

    def start_pt(self, i):
        return self.moves[i, 0, 0], self.moves[i, 0, 1]

    def end_pt(self, i):
        return self.moves[i, 1, 0], self.moves[i, 1, 1]


if __name__ == '__main__':
    gen = Generation(1, 6, 3, 7)
    gen.fill()
    # gravity_gpu(gen.skeletons)
    gen.play()
