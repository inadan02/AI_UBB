import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


def readNet(fileName):
    f = open(fileName, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(" ")
        for j in range(n):
            mat[-1].append(int(elems[j]))
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if (mat[i][j] == 1):
                d += 1
            if (j > i):
                noEdges += mat[i][j]
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    f.close()
    return net


def plotNetwork(net, communities=None):
    if communities is None:
        communities = [1] * net["noNodes"]
    np.random.seed(123)
    A = np.matrix(net["mat"])
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(4, 4))
    nx.draw_networkx_nodes(G, pos, node_size=35, cmap=plt.cm.RdYlBu, node_color=communities)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()


def greedyCommunitiesDetectionByTool(net):
    # Input: a graph
    # Output: list of community index (for every node)

    from networkx.algorithms import community

    A = np.matrix(net["mat"])
    G = nx.from_numpy_matrix(A)
    communities_generator = community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    sorted(map(sorted, top_level_communities))
    communities = [0 for node in range(net['noNodes'])]
    index = 1
    for community in sorted(map(sorted, top_level_communities)):
        for node in community:
            communities[node] = index
        index += 1
    return communities


def readNetFromGml(file):
    net = nx.read_gml(file, label="id")
    matr=nx.adjacency_matrix(net).todense()
    net = {"mat": matr, "noNodes": net.number_of_nodes(), "noEdges": net.number_of_edges(), "degrees": net.degree()}
    return net



def calculate_edge_to_remove(G):
    edges = list(nx.edge_betweenness_centrality(G).items())  # Dictionary of edges with betweenness centrality as the value. {(edge): Q}
    most_crossed = max(edges, key=lambda item: item[1]) # max dupa Q care e valoare din dictionarul de mai sus
    return most_crossed[0]


def greedyCommunitiesDetection(net, number_of_communities):
    A = np.matrix(net["mat"])
    G = nx.from_numpy_matrix(A)
    while len(list(nx.connected_components(G))) < number_of_communities:
        edge_to_remove = calculate_edge_to_remove(G)
        G.remove_edge(edge_to_remove[0], edge_to_remove[1]) # source node, destination node
    communities=[0 for node in range(net["noNodes"])]
    index = 1
    for community in nx.connected_components(G):
        for node in community:
            communities[node] = index
        index += 1
    return communities


def run_tests():
    crtDir = os.getcwd()
    print("Started DOLPHINS test")
    filePath = os.path.join(crtDir, 'C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real', 'dolphins.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished DOLPHINS test")

    print("Started KARATE test")
    filePath = os.path.join(crtDir, 'C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real', 'karate.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished KARATE test")

    print("Started KREBS test")
    filePath = os.path.join(crtDir, 'C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real', 'krebs.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished KREBS test")

    print("Started FOOTBALL test")
    filePath = os.path.join(crtDir, 'C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real', 'football.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished FOOTBALL test")

    print("Started LES MISERABLES test")
    filePath = os.path.join(crtDir, 'data/real', 'lesmis.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished LES MISERABLES test")

    print("Started BOOKS ABOUT US POLITICS")
    filePath = os.path.join(crtDir, 'data/real', 'polbooks.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished BOOKS ABOUT US POLITICS")

    print("Started WORD ADJACENCY test")
    filePath = os.path.join(crtDir, 'data/real', 'adjnoun.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished WORD ADJACENCY test")

    print("Started MAP COMUNITATE test")
    filePath = os.path.join(crtDir, 'data/real', 'comunitate_map.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished MAP COMUNITATE test")

    print("Started PRIMARY SCHOOL test")
    filePath = os.path.join(crtDir, 'data/real', 'primaryschool.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished PRIMARY SCHOOL test")

"""
    print("Started CONDENSED MATTER COLLAB 1999 test")
    filePath = os.path.join(crtDir, 'data/real', 'cond-mat.gml')
    net = readNetFromGml(filePath)
    assert (greedyCommunitiesDetection(net, 2) == greedyCommunitiesDetectionByTool(net))
    print("Finished CONDENSED MATTER COLLAB 1999 test")
"""



if __name__ == '__main__':
    #run_tests()

    network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/dolphins.gml")
    communities = greedyCommunitiesDetection(network, 2)
    plotNetwork(network, communities)

    network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/comunitate_map.gml")
    communities = greedyCommunitiesDetection(network, 3)
    plotNetwork(network, communities)
    print(greedyCommunitiesDetection(network, 3))


    # network1 = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/football.gml")
    # communities1 = greedyCommunitiesDetectionByTool(network1)
    # plotNetwork(network1, communities1)
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/football.gml")
    # communities = greedyCommunitiesDetection(network, 2)
    # plotNetwork(network, communities)
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/football.gml")
    # communities = greedyCommunitiesDetection(network, 4)
    # plotNetwork(network, communities)
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/lesmis.gml")
    # communities = greedyCommunitiesDetection(network, 2)
    # plotNetwork(network, communities)
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/polbooks.gml")
    # communities = greedyCommunitiesDetection(network, 2)
    # plotNetwork(network, communities)
    #
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/comunitate_map.gml")
    # communities = greedyCommunitiesDetectionByTool(network)
    # plotNetwork(network, communities)
    #
    #
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/lesmis.gml")
    # communities = greedyCommunitiesDetection(network, 32)
    # plotNetwork(network, communities)
    #
    # network = readNetFromGml("C:/FACULTATE/ANUL2/SEM2/AI/LAB2/data/real/polbooks.gml")
    # communities = greedyCommunitiesDetection(network, 30)
    # plotNetwork(network, communities)
    # print(greedyCommunitiesDetection(network, 30))
    #
    #














