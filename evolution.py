import json

import matplotlib.pyplot as plt
import numpy as np

from individual import Individual
from neuralNetwork import NeuralNetwork
from generation import Generation


class Evolution:
    def __init__(self, generation_nb, generation_size, game_per_generation, row_nb, column_nb, cap):
        self.generation_nb = generation_nb
        self.generation_size = generation_size
        self.game_per_generation = game_per_generation
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap = cap
        self.generation = Generation(generation_size, row_nb, column_nb, cap)
        self.score_mean = list()
        self.score_max = list()
        self.score_min = list()

    def nextgen(self):
        scores = np.zeros(self.generation_size)
        for i in range(self.game_per_generation):
            scores += np.array(list(self.generation.play()))
        scores /= self.game_per_generation
        self.score_mean.append(np.mean(scores))
        self.score_max.append(np.max(scores))
        self.score_min.append(np.min(scores))
        median = np.median(scores)
        newgen = [gen for score, gen in zip(scores, self.generation) if score >= median]
        for k in range(self.generation_size - len(newgen)):
            newgen.append(newgen[k].mutate())
        self.generation = newgen

    def train(self):
        for i in range(self.generation_nb):
            self.nextgen()
            print(int(100 * i / self.generation_nb), '%')
            print('max: ', int(evolution.score_max[-1]))
        self.plot()

    def plot(self):
        x = list(range(self.generation_nb))
        plt.plot(x, self.score_max, x, self.score_mean, x, self.score_min)
        plt.show()

    @staticmethod
    def load_individual(filepath):
        with open(filepath) as json_data:
            params = json.load(json_data)
        return Individual(row_nb=params['row_nb'],
                          column_nb=params['column_nb'],
                          cap=params['cap'],
                          intelligence=NeuralNetwork(**params['intelligence'])
                          )


if __name__ == '__main__':
    evolution = Evolution(
        generation_nb=200,
        generation_size=50,
        game_per_generation=10,
        row_nb=6,
        column_nb=3,
        cap=7
    )
    evolution.train()
