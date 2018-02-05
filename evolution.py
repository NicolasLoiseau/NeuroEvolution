import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from generation import Generation
from individual import Individual
from timeit import timeit


class Evolution:
    def __init__(self, generation_nb, generation_size, game_per_generation, row_nb, column_nb, cap, use_gpu=False):
        self.generation_nb = generation_nb
        self.generation_size = generation_size
        self.game_per_generation = game_per_generation
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap = cap
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.generation = Generation(generation_size, row_nb, column_nb, cap)
        else:
            self.generation = [Individual(row_nb, column_nb, cap) for _ in range(0, generation_size)]
        self.score_mean = list()
        self.score_max = list()
        self.score_min = list()

    @timeit
    def nextgen(self):
        scores = np.zeros(self.generation_size)
        if self.use_gpu:
            for i in range(self.game_per_generation):
                scores += np.array(list(self.generation.play()))
        else:
            for i in range(self.game_per_generation):
                scores += np.array(list(map(lambda ind: ind.play(), self.generation)))
        scores /= self.game_per_generation
        self.score_mean.append(np.mean(scores))
        self.score_max.append(np.max(scores))
        self.score_min.append(np.min(scores))
        index = np.argsort(scores)[:self.generation_size//2].shape
        if self.use_gpu:
            self.generation.intelligence.mutate(index)
        else:
            newgen = [gen for i, gen in zip(range(self.generation_size), self.generation) if i in index]
            for k in range(self.generation_size - len(newgen)):
                newgen.append(newgen[k].mutate())
            self.generation = newgen

    def train(self):
        for i in range(self.generation_nb):
            self.nextgen()
            r = int(50 * (i + 1) / self.generation_nb)
            load_bar = '|' + r * '*' + (50 - r) * '_' + '|'
            maxi = ' max: ' + str(int(self.score_max[-1]))
            print(load_bar + maxi, end='\r')
        self.plot()

    def plot(self):
        x = list(range(self.generation_nb))
        plt.plot(x, self.score_max, color='teal')
        plt.plot(x, self.score_mean, color='orange')
        plt.plot(x, self.score_min, color='green')
        max_patch = mpatches.Patch(color='teal', label='max')
        mean_patch = mpatches.Patch(color='orange', label='mean')
        min_patch = mpatches.Patch(color='green', label='min')

        plt.legend(handles=[max_patch, mean_patch, min_patch])
        plt.show()


if __name__ == '__main__':
    evolution = Evolution(
        generation_nb=100,
        generation_size=50,
        game_per_generation=1,
        row_nb=6,
        column_nb=3,
        cap=7,
        use_gpu=False
    )
    evolution.train()
