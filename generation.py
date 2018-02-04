from individual import Kernel
import numpy as np


class Generation:
    def __init__(self, generation_size, row_nb, column_nb, cap):
        self.generation_size = generation_size
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap0 = cap
        self.skeletons = np.zeros((self.generation_size,self.row_nb, self.column_nb)).astype(int)
        self.scores = np.zeros(self.generation_size)
        self.caps = np.ones(self.generation_size)*self.cap0

    def reset(self):
        self.skeletons = np.zeros((self.generation_size,self.row_nb, self.column_nb)).astype(int)
        self.scores = np.zeros(self.generation_size)
        self.caps = np.ones(self.generation_size)*self.cap0

    def fill(self):
        height = self.row_nb // 2
        sub_skeleton = np.random.randint(self.cap0, size=(self.generation_size, height, self.column_nb)) + 1
        self.skeletons = sub_skeleton

    @property
    def game_over(self):
        return self.skeletons.sum(axis=2)[:, 0] != 0

    def remodeling(self, start_point, end_point, play_range):
        for i in play_range:
            self.individuals[i].remodelling

    def refill(self, not_equal, play_range):
        for i in play_range:
            if not_equal[i]:
                column = np.random.randint(0, self.column_nb - 1)
                values = [1, 1, 2, 2, self.caps[i], self.caps[i] - 1]
                uniform = np.random.randint(0, 5)
                value = values[uniform]
                self.skeletons[i][0, column] = value

    def gravity(self):
        for i in range(self.generation_size):
            self.individuals[i].gravity()


    def one_play(self):
        start_pt, end_pt = self.get_move()
        not_equal = [self.skeletons[i][start_pt[i][0]][start_pt[i][1]] != self.skeletons[i][end_pt[i][0]][end_pt[i][1]] for i in range(self.generation_size)]
        play_range = np.where(np.invert(self.game_over))[0]
        play_vec = self.game_over.astype(int)
        self.remodeling(start_pt, end_pt, play_vec)
        self.refill(not_equal, play_range)
        self.gravity()

    def play(self):
        self.reset()
        self.fill()
        while not all(self.game_over):
            self.one_play()
        return self.scores

    def get_move(self):
        start_points = np.zeros((self.generation_size, 2))
        end_points = np.zeros((self.generation_size, 2))
        for i in range(self.generation_size):
            start_point, end_point = self.individuals[i].get_move()
            start_points[i] = start_point
            end_points[i] = end_point
        return start_points.astype(int), end_points.astype(int)


if __name__ == '__main__':
    for i in range(0,1):
        gen = Generation(3, 6, 3, 7)
        #print([i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i])
        gen.play()


