def write(n, route, route_cost, filename):
    file = open(filename, 'w')

    solution = []
    for i in route:
        solution.append(i + 1)

    file.write(str(n) + '\n')
    route_string = ''
    for i in range(len(solution) - 1):
        route_string += str(solution[i]) + ','
    route_string += str(solution[n - 1]) + '\n'

    file.write(str(route_string))
    file.write(str(route_cost) + '\n')

    file.close()
