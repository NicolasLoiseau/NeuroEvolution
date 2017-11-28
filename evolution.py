import numpy as np
from individual import Individual

class Evolution:

    def __init__(self,generation_nb,generation_size, game_per_generation, row_nb, column_nb, cap):
        self.generation_nb = generation_nb
        self.generation_size = generation_size
        self.game_per_generation = game_per_generation
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap = cap
        self.generation = [Individual(row_nb, column_nb, cap) for _ in range(0,generation_size)]

    def nextgen(self):
        scores = []
        newgen = []
        for i in range(self.game_per_generation):
            for j in range(len(self.generation)):
                if i == 0:
                    scores.append(self.generation[j].play())
                else:
                    scores[j] += self.generation[j].play()
        scores = [s / self.game_per_generation for s in scores]
        median = np.median(scores)
        for sg in zip(scores,self.generation):
            if sg[0]>median:
                newgen.append(sg[1])
        survivors_nb = len(newgen)
        for k in range(self.generation_size-survivors_nb):
            newgen.append(newgen[k].mutate)
        return(newgen)

evolution=Evolution(1,100,10,6,3,7)