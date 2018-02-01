from individual import Individual


class Generation:
    def __init__(self, generation_size, row_nb, column_nb, cap):
        self.generation_size = generation_size
        self.individuals = [Individual(row_nb, column_nb, cap) for _ in range(0, generation_size)]

    @property
    def skeletons(self):
        return [self.individuals[i].skeleton for i in range(0, self.generation_size)]
