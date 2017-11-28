import numpy as np

from individual import Individual
import matplotlib.pyplot as plt

class Evolution:
    def __init__(self, generation_nb, generation_size, game_per_generation, row_nb, column_nb, cap):
        self.generation_nb = generation_nb
        self.generation_size = generation_size
        self.game_per_generation = game_per_generation
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap = cap
        self.generation = [Individual(row_nb, column_nb, cap) for _ in range(0, generation_size)]
        self.score_mean = list()
        self.score_max = list()

    def nextgen(self):
        scores = np.zeros(self.generation_size)
        for i in range(self.game_per_generation):
            scores += np.array(list(map(lambda ind: ind.play(), self.generation)))
        scores /= self.game_per_generation
        self.score_mean.append(np.mean(scores))
        self.score_max.append(np.max(scores))
        median = np.median(scores)
        newgen = [gen for score, gen in zip(scores, self.generation) if score >= median]
        for k in range(self.generation_size - len(newgen)):
            newgen.append(newgen[k].mutate())
        self.generation = newgen

if __name__ == '__main__':

    evolution = Evolution(1, 100, 10, 6, 3, 7)
    for i in range(1000):
        print(i)
        evolution.nextgen()
    plt.plot(evolution.score_max)
    plt.show()
