from random import randint

from Chromosome import Chromosome


def route_cost(route, matr):
    #route - permutarea de noduri
    #matr - matricea de cost
    cost = 0
    for i in range(len(route) - 1):
        cost += matr[route[i]][route[i + 1]]
    cost += matr[route[0]][route[len(route) - 1]]
    return cost

def path_length_destination(path,params):
    start_node, end_node = params['source'], params['destination']
    length = 0
    for i in range(start_node,end_node-1):
        length += params['mat'][path[i]][path[i + 1]]
    length += params['mat'][path[0]][path[params['destination']-1]]
    return length


def route_cost_s_d(route, s, d, matr):
    cost = 0
    for i in range(s, d-1):
        cost += matr[route[i]][route[i + 1]]
    cost += matr[route[0]][route[d - 1]]
    return cost




class GA:
    def __init__(self, param=None, problemParam=None):
        self.__param = param
        self.__problemParam = problemParam
        self.__population = []

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for _ in range(0, self.__param['popSize']):
            c = Chromosome(self.__problemParam)
            self.__population.append(c)

    def evaluation(self):
        for c in self.__population:
            c.fitness = self.__problemParam['function'](c.representation, self.__problemParam['cities'])
            print(c)

    def evaluation1(self):
        for c in self.__population:
            c.fitness = self.__problemParam['function'](c.representation, self.__problemParam['source'],
                                                        self.__problemParam['destination'],
                                                        self.__problemParam['cities'])
            print(c)

    def bestChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if c.fitness < best.fitness:
                best = c
        return best

    def worstChromosome(self):
        best = self.__population[0]
        for c in self.__population:
            if c.fitness > best.fitness:
                best = c
        return best

    def selection(self):
        pos1 = randint(0, self.__param['popSize'] - 1)
        pos2 = randint(0, self.__param['popSize'] - 1)
        if self.__population[pos1].fitness < self.__population[pos2].fitness:
            return pos1
        else:
            return pos2

    def oneGeneration(self):
        newPop = []
        for _ in range(self.__param['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()

    def oneGenerationElitism(self):
        newPop = [self.bestChromosome()]
        for _ in range(self.__param['popSize'] - 1):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation()

    def oneGeneration1(self):
        newPop = []
        for _ in range(self.__param['popSize']):
            p1 = self.__population[self.selection()]
            p2 = self.__population[self.selection()]
            off = p1.crossover(p2)
            off.mutation()
            newPop.append(off)
        self.__population = newPop
        self.evaluation1()
