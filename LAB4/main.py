import sys
import networkx as nx
from Chromosome import Chromosome
from utils import readFromFile, writeToFile
from GA import GA, route_cost, route_cost_s_d, path_length_destination
import tsplib95 as tspp

from math import factorial


def main():
    #n, tsp, source, destination = readFromFile.read("data/easy_01_tsp.txt")
    #n, tsp, source, destination = readFromFile.read("data/easy_02_tsp.txt")
    #n, tsp, source, destination = readFromFile.read("data/easy_03_tsp.txt")
    #n, tsp, source, destination = readFromFile.read_space("data/hard_01_tsp.txt")
    #n, tsp, source, destination = readFromFile.read("data/medium_01_tsp.txt")
    #n, tsp, source, destination = readFromFile.read("data/medium_02_tsp.txt")
    #n, tsp, source, destination = readFromFile.read("data/medium_03_tsp.txt")
    #n, tsp, source, destination = readFromFile.read("data/medium_04_tsp.txt")
    ##n, tsp, source, destination = readFromFile.read_space("data/hard_02_tsp.txt")
    n, tsp, source, destination = readFromFile.read("data/hard.txt")
    #problme=tspp.load("data/eil101.tsp")
    #tsp=problme.get_graph()
    #g=nx.to_numpy_matrix(tsp)
    ga_param = {'popSize': 300, 'noGen': 300}
    problem_param = {'noNodes': n, 'cities': tsp, 'function': route_cost}

    # ga_param1 = {'popSize': 300, 'noGen': 300}
    # problem_param1 = {'noNodes': n, 'cities': tsp, 'function': route_cost_s_d, 'source': source, 'destination': destination}
    #print(tsp[1][2]['weight'])
    ga = GA(ga_param, problem_param)
    ga.initialisation()
    ga.evaluation()

    # ga1 = GA(ga_param1, problem_param1)
    # ga1.initialisation()
    # ga1.evaluation1()

    best_overall_chromosome = Chromosome(problem_param)
    best_overall_chromosome.fitness = sys.maxsize

    # best_overall_chromosome1 = Chromosome(problem_param1)
    # best_overall_chromosome1.fitness = sys.maxsize
    best_sol = []
    #solution = None
    for generation in range(ga_param['noGen']):

        #ga.oneGeneration()
        ga.oneGenerationElitism()
        bestChromosome = ga.bestChromosome()
        #best_sol.append(bestChromosome)
        print('Best cost in generation ' + str(generation+1) + ' is: ' + str(bestChromosome))
        solution = bestChromosome

        if bestChromosome.fitness < best_overall_chromosome.fitness:
            best_overall_chromosome = bestChromosome
        if bestChromosome.fitness==best_overall_chromosome.fitness:
            best_sol.append(bestChromosome)
            #if bestChromosome not in best_sol:
                #best_sol.append(bestChromosome)
    # best_best=[]
    # for i in best_sol:
    #     if best_sol[i].fitness==best_overall_chromosome.fitness:
    #         best_best.append(best_sol[i])
    print("best solutions:", best_sol)

        #print("Best overall cost: " + str(best_overall_chromosome))

    # solution1 = None
    # for g1 in range(ga_param1['noGen']):
    #     ga1.oneGeneration1()
    #     #ga1.oneGenerationElitism()
    #     bestChromosome1 = ga1.bestChromosome()
    #     print('Best solution in generation ' + str(g1 + 1) + ' is: ' + str(bestChromosome1))
    #     solution1 = bestChromosome1
    #     if bestChromosome1.fitness < best_overall_chromosome1.fitness:
    #         best_overall_chromosome1 = bestChromosome1

    #print(source)
    #print(destination)
    #print(tsp)
    print("Best overall cost: " + str(best_overall_chromosome))


    #writeToFile.write(n, solution.representation, route_cost(solution.representation, tsp), 'output.txt')
    writeToFile.write(n, solution.representation, route_cost(best_overall_chromosome.representation, tsp), 'output.txt')
    #writeToFile.write(n, solution1.representation, route_cost_s_d(solution1.representation,source, destination, tsp),'output.txt')


main()
