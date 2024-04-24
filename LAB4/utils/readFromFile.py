def read(fileName):
    file = open(fileName, 'r')
    n = int(file.readline())
    tsp = []
    for i in range(n):
        line = file.readline().split(",")
        costs = []
        for cost in line:
            costs.append(int(cost))
        tsp.append(costs)
    source = int(file.readline())
    destination = int(file.readline())
    file.close()

    return n, tsp, source, destination

def read_space(fileName):
    file = open(fileName, 'r')
    n = int(file.readline())
    tsp = []
    for i in range(n):
        line = file.readline().split()
        costs = []
        for cost in line:
            costs.append(int(cost))
        tsp.append(costs)
    source = int(file.readline())
    destination = int(file.readline())
    file.close()

    return n, tsp, source, destination
