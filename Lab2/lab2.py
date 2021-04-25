import numpy as np
import random
from matplotlib import pyplot as plt
podgraphs = []

def find_value(graph, start, finish, way):
    current_res = []
    if start == finish:
        way.append(start)
        result = way.copy()
        way.pop()
        return result, True
    if len(way) > len(graph) + 1:
        return [], True
    cur_nodes = graph[start]
    for node in cur_nodes:
        if way.count(node) == 0:
            way.append(start)
            res_of_return, is_found = find_value(graph, node, finish, way)
            if is_found and len(res_of_return) > 0:
                current_res.append(res_of_return)
            elif len(res_of_return) > 0:
                current_res.extend(res_of_return)
            way.pop()

    return current_res, False


def getEdges(graph):
    edges = []
    for i in range(1,len(graph)):
        for j in range(len(graph[i])):
            if i > graph[i][j]:
                edges.append(int(str(graph[i][j]) + str(i)))
            else:
                edges.append(int(str(i) + str(graph[i][j])))
    edges = list(set(edges))
    return sorted(edges)


def getAllPodgraphs(edges):
    for i in range(0, len(edges)):
        if edges[i] != 0:
            tmp = edges[i]
            edges[i] = 0
            podgraphs.append(edges)
            getAllPodgraphs(edges)
            edges[i] = tmp


def waysToEdges(ways, edges):
    res = [[0] * len(edges) for i in range(len(ways))]
    for i in range(len(ways)):
        for j in range(0, len(ways[i]) - 1):
            if ways[i][j] < ways[i][j+1]:
                edge = ways[i][j] * 10 + ways[i][j+1]
            else:
                edge = ways[i][j+1] * 10 + ways[i][j]
            res[i][edges.index(edge)] = 1
        print()
    return res


def getAllGrpahs(m):
    result = []
    for i in range(0, pow(2,m)):
        tmp = [int(i) for i in bin(i)[2:]]
        if len(tmp) < m:
            for i in range(len(tmp), m):
                tmp.insert(0,0)
        result.append(tmp)
    return result


def getPodgraphs(edges, ways):
    res = []
    for i in range(len(ways)):
        max = ways[i].count(1)
        for j in range(len(edges)):
            if max <= edges[j].count(1):
                count = 0
                for k in range(len(edges[j])):
                    if ways[i][k] == 1 and ways[i][k] == edges[j][k]:
                        count += 1
                if count == max:
                    if res.count(edges[j]) == 0:
                        res.append(edges[j])
    return res


def get_result(result, edges):
    res = []
    for i in range(len(result)):
        res.append([edges[j]*result[i][j] for j in range(len(edges))])
    return res


def task2(vectors, n):
    ex = 0.01
    N = 2.25 / pow(ex, 2)
    to_return = []
    for p in range(11):
        p /= 10
        result = 0
        for z in range(int(N)):
            graph = [0 for _ in range(n)]
            for i in range(len(graph)):
                if random.uniform(0, 1) < p:
                    graph[i] = 1
                else:
                    graph[i] = 0
            for i in range(len(vectors)):
                tmp = 0
                cur = 0
                for j in range(len(vectors[i])):
                    if vectors[i][j] == 1:
                        tmp = tmp + 1
                        if vectors[i][j] == graph[j]:
                            cur = cur + 1
                if tmp == cur and tmp != 0 and cur != 0:
                    result += 1
                    break
        to_return.append(result / N)
    return to_return


def lab2_fast(vectors, n):
    ex = 0.01
    l_min = 3
    l_max = n -1
    N = 2.25 / pow(ex, 2)
    to_return = []
    for p in range(0, 11, 1):
        p /= 10
        result = 0
        res = 0
        for z in range(int(N)):
            ves = 0
            graph = [0 for _ in range(n)]
            for i in range(len(graph)):
                if random.uniform(0, 1) < p:
                    graph[i] = 1
                    ves += 1
                else:
                    graph[i] = 0
            graph = np.flip(graph, 0)
            if ves < l_min or ves > l_max:
                res += 1
            if ves > l_max:
                result += 1
            elif ves > l_min:
                for i in range(len(vectors)):
                    tmp = 0
                    cur = 0
                    for j in range(len(vectors[i])):
                        if vectors[i][j] == 1:
                            tmp = tmp + 1
                            if vectors[i][j] == graph[j]:
                                cur = cur + 1
                    if tmp == cur and tmp != 0 and cur != 0:
                        result += 1
                        break
        to_return.append(res/N)
    return to_return




if __name__ == '__main__':
    graph = {
        1: [2, 4, 6],
        2: [1, 3, 4],
        3: [2, 4, 5],
        4: [1, 2, 3, 5],
        5: [3, 4, 6],
        6: [1, 5]
    }
    way = []

    res = []
    rebra = []

    res.extend(find_value(graph, 1, 3, way)[0])
    #print(res)

    edges = getEdges(graph)  # нумерация ребр
    #print(edges)
    ways = waysToEdges(res, edges)  # перевод полученных путей из вершин в ребра
    all_edges = getAllGrpahs(len(edges))  # получение всех возможных подграфов
    res = getPodgraphs(all_edges, ways)  # получение нужных подграфов
    res = get_result(res, edges)


    P = []
    p1 = []
    for p in range(0, 11, 1):
        p /= 10
        result = 0
        for i in range(len(res)):
            sum_0 = 0
            sum_1 = 0
            for j in range(len(res[i])):
                if res[i][j] == 0:
                    sum_0 += 1
                else:
                    sum_1 += 1
            result = result + pow(p, sum_1) * pow((1 - p), sum_0)
        P.append(result)
        y = p + p**2 - p**3
        x1 = ((2 * p ** 2 - p ** 4) + p * (2 * p - p ** 2) - p * (2 * p - p ** 2) * (2 * p ** 2 - p ** 4))
        x2 = ((2 * p - p ** 2) ** 2 + p * (2 * p - p ** 2) - p * (2 * p - p ** 2) * ((2 * p - p ** 2) ** 2))
        x3 = (p ** 2 + (2 * p - p ** 2) * (p + (2 * p - p ** 2) - p * (2 * p - p ** 2)) - p ** 2 * (2 * p - p ** 2) * (
                    p + (2 * p - p ** 2) - p * (2 * p - p ** 2)))
        x4 = ((2 * p - p ** 2 + (2 * p - p ** 2) - (2 * p - p ** 2) * (2 * p - p ** 2)) * (
                    (2 * p - p ** 2) + p - p * (2 * p - p ** 2)))
        p1.append((x1 * (1 - p) + x2 * p) * (1 - p) + (x3 * (1 - p) + x4 * p) * p)

    second = task2(ways, len(all_edges[0]))
    print(second)
    third = lab2_fast(ways, len(all_edges[0]))
    print("Полный перебор" + ' | ' + "Метод декомпозиции" + ' | ' + "Ускоренный метод")
    for i in range(len(P)):
        print(str(P[i]) + ' | ' + str(p1[i]) + ' | ' + str(second[i]))



    plt.figure()
    plt.plot([i/10 for i in range(len(P))], second, label="полный перебор")
    plt.plot([i/10 for i in range(len(P))], P, label="модуляция")
    plt.plot([i / 10 for i in range(len(P))], third, label="Ускоренный")
    plt.legend()
    plt.grid(True)
    #plt.figure()
    # plt.legend()
    # plt.grid(True)
    plt.show()


