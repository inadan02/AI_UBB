from GA import GA
from RealChromosome import Chromosome
import networkx as nx
from random import seed
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')


def readFromTxt(filePath):
    f = open(filePath, "r")
    net = {}
    edges = []
    for line in f.readlines():
        elems = line.strip().split()
        edges.append((int(elems[0]), int(elems[1])))
    f.close()
    graph = nx.Graph()
    graph.add_edges_from(edges)
    net['noNodes'] = graph.number_of_nodes()
    net['noEdges'] = graph.number_of_edges()
    mat = []
    mat = nx.to_numpy_matrix(graph)
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(net['noNodes']):
        d = 0
        for j in range(net['noNodes']):
            if (mat.item((i, j)) == 1):
                d += 1
            if (j > i):
                noEdges += mat.item((i, j))
        degrees.append(d)
    net["degrees"] = degrees
    return net


def readGmlFile(fileName):
    if fileName == "data/krebs.gml" or fileName == "data/karate.gml":
        graph = nx.read_gml(fileName, label='id')
    else:
        graph = nx.read_gml(fileName)
    net = {}
    net['noNodes'] = graph.number_of_nodes()
    net['noEdges'] = graph.number_of_edges()
    mat = []
    mat = nx.to_numpy_matrix(graph)
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(net['noNodes']):
        d = 0
        for j in range(net['noNodes']):
            if (mat.item((i, j)) == 1):
                d += 1
            if (j > i):
                noEdges += mat.item((i, j))
        degrees.append(d)
    net["degrees"] = degrees
    return net


# param = readGmlFile("data/football.gml")
# param = readGmlFile("data/lesmis.gml")
# param = readGmlFile("data/dolphins.gml")
# param = readGmlFile("data/krebs.gml")
param = readGmlFile("data/karate.gml")
# param = readGmlFile("data/adjnoun.gml")
# param = readGmlFile("data/primaryschool.gml")
# param = readGmlFile("data/polbooks.gml")
# param = readGmlFile("data/comunitate_map.gml")
# param = readFromTxt("data/classLabelfootball.txt")  # 11 comunitati onegeneration
# param=readFromTxt("data/classLabeldolphins.txt")

MIN = 0
MAX = param['noNodes']


def modularity(communities):
    #functia fitness
    noNodes = param['noNodes']
    mat = param['mat']
    degrees = param['degrees']
    noEdges = param['noEdges']
    M = 2 * noEdges
    Q = 0.0
    for i in range(0, noNodes):
        for j in range(0, noNodes):
            if (communities[i] == communities[j]):
                Q += (mat.item((i, j)) - degrees[i] * degrees[j] / M)
    return Q * 1 / M


seed(1)

gaParam = {'popSize': 300, 'noGen': 300}
problParam = {'min': MIN, 'max': MAX, 'function': modularity, 'noNodes': MAX}

generations = []

ga = GA(gaParam, problParam)
ga.initialisation()
ga.evaluation()

bestestChromosome = Chromosome(problParam)

for generation in range(gaParam['noGen']):
    generations.append(generation)

    # ga.oneGeneration()
    ga.oneGenerationElitism()
    # ga.oneGenerationSteadyState()

    bestChromosome = ga.bestChromosome()
    if bestestChromosome.fitness < bestChromosome.fitness:
        bestestChromosome = bestChromosome

    print('Generation: ' + str(generation) + '; no. communities: ' + str(
        len(Counter(bestChromosome.repres).keys())) + ' fitness: ' + str(bestChromosome.fitness) + '\n')

print("Number of communities: " + str(len(Counter(bestestChromosome.repres).keys())) + '\n')
print("Best fitness: " + str(bestestChromosome.fitness) + '\n')



