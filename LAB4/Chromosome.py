from random import randint

def random_permutation(n):
    perm = [i for i in range(n)]
    pos1 = randint(1, n - 1)
    pos2 = randint(1, n - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    return perm


class Chromosome:
    def __init__(self, problemParam=None):
        self.__problemParam = problemParam
        self.__representation = random_permutation(self.__problemParam['noNodes'])
        self.__fitness = 0.0

    @property
    def representation(self):
        return self.__representation

    @property
    def fitness(self):
        return self.__fitness

    @representation.setter
    def representation(self, l=[]):
        self.__representation = l

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    def crossover(self, c):
        gene1 = randint(1, self.__problemParam['noNodes'] - 1)
        gene2 = randint(gene1, self.__problemParam['noNodes'] - 1)
        repres1 = [0]
        for i in range(gene1, gene2):
            repres1.append(c.__representation[i])
        repres2 = [el for el in self.__representation if el not in repres1]
        new_representation = repres1 + repres2
        offspring = Chromosome(self.__problemParam)
        offspring.representation = new_representation
        return offspring

    def mutation(self):
        # swap mutation p=0.1
        r1 = randint(1, 101)
        if r1 <= 10:
            pos1 = randint(1, self.__problemParam['noNodes'] - 1)
            pos2 = randint(1, self.__problemParam['noNodes'] - 1)
            self.__representation[pos1], self.__representation[pos2] = self.__representation[pos2], self.__representation[pos1]

    def __str__(self):
        return "\nChromosome " + str(self.__representation) + " has fit " + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__representation == c.__representation and self.__fitness == c.__fitness
