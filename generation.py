from individual import Individual
import numpy as np


class Generation:
    def __init__(self, generation_size, row_nb, column_nb, cap):
        self.generation_size = generation_size
        self.individuals = [Individual(row_nb, column_nb, cap) for _ in range(0, generation_size)]
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap0 = cap

    @property
    def intelligences(self):
        return [self.individuals[i].intelligence for i in range(self.generation_size)]

    @property
    def skeletons(self):
        return np.array([self.individuals[i].skeleton for i in range(self.generation_size)])

    @property
    def scores(self):
        return [self.individuals[i].score for i in range(self.generation_size)]

    @property
    def caps(self):
        return [self.individuals[i].cap for i in range(self.generation_size)]

    def reset(self):
        for i in range(0, self.generation_size):
            self.individuals[i].reset()

    def fill(self):
        height = self.row_nb // 2
        sub_skeleton = np.random.randint(self.cap0, size=(self.generation_size,height, self.column_nb)) + 1
        for i in range(self.generation_size):
            self.individuals[i].skeleton[: self.row_nb - height, :] = 0
            self.individuals[i].skeleton[self.row_nb - height:, :] = sub_skeleton[i]

    @property
    def game_over(self):
        #return [self.skeletons[i][0].sum() != 0 for i in range(0, self.generation_size)]
        #return [self.individuals[i].skeleton.sum(axis=1)[:1].sum() != 0 for i in range(0, self.generation_size)]
        return self.skeletons.sum(axis=2)[:, 0] != 0

    def remodeling(self, start_point, end_point, play_range):
        for i in play_range:
            if self.skeletons[i][end_point[i][0]][end_point[i][1]] != self.skeletons[i][start_point[i][0]][start_point[i][1]]:
                self.individuals[i].update_dashboard(self.skeletons[i][start_point[i][0]][start_point[i][1]] + self.skeletons[i][end_point[i][0]][end_point[i][1]])
                self.individuals[i].skeleton[end_point[i][0]][end_point[i][1]] += self.skeletons[i][start_point[i][0]][start_point[i][1]]
            else:
                self.individuals[i].update_dashboard(2 * self.skeletons[i][start_point[i][0]][start_point[i][1]])
            self.individuals[i].skeleton[start_point[i][0]][start_point[i][1]] = 0

    def refill(self, not_equal, play_range):
        for i in play_range:
            if not_equal[i]:
                column = np.random.randint(0, self.column_nb - 1)
                values = [1, 1, 2, 2, self.caps[i], self.caps[i] - 1]
                uniform = np.random.randint(0, 5)
                value = values[uniform]
                self.individuals[i].skeleton[0, column] = value

    def gravity(self):
        for i in range(self.generation_size):
            transposed = self.skeletons[i].transpose().copy()
            self.individuals[i].skeleton = np.array([self.individuals[i].gravity_line(line) for line in transposed]).transpose()

    def one_play(self):
        start_pt, end_pt = self.get_move()
        not_equal = [self.skeletons[i][start_pt[i][0]][start_pt[i][1]] != self.skeletons[i][end_pt[i][0]][end_pt[i][1]] for i in range(self.generation_size)]
        play_range = np.where(np.invert(self.game_over))[0]
        self.remodeling(start_pt, end_pt, play_range)
        self.refill(not_equal, play_range)
        self.gravity()

    def play(self):
        self.reset()
        self.fill()
        while not all(self.game_over):
            self.one_play()
            #print(self.skeletons)
            #print(self.game_over)
            #print(self.scores[0])
            #print(self.caps[0])
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

    #gen.fill()
    #print(gen.individuals[0].skeleton)
    #print(gen.individuals[1].skeleton)

