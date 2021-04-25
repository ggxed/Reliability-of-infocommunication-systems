import networkx as nx
from matplotlib import pyplot as plt
from itertools import combinations 


def Plot(ver, value, x, y, title):
    plt.figure()
    plt.plot(ver, value)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True)



def graph(N):
	G = nx.Graph()
	G.add_nodes_from(N)
	G.add_edge(1,2)
	G.add_edge(1,4)
	G.add_edge(1,6)
	G.add_edge(1,5)
	G.add_edge(2,3)
	G.add_edge(2,4)
	G.add_edge(3,4)
	G.add_edge(3,5)
	G.add_edge(4,5)
	G.add_edge(5,6)
	return(G)

 #def visual()
# 	G = nx.cubical_graph()
# 	nx.draw(G)   
# 	nx.draw(G,pos=nx.spectral_layout(G), nodecolor='r',edge_color='b')

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

if __name__ =="__main__":
	R = 6 # вершины
	R1 = 10 # ребра
	N = [i + 1 for i in range(R)]
	P = []
	vb = 1
	ve = 3
	spisok = []
	G = nx.Graph()
	G = graph(N)
	#G = nx.cubical_graph()
	nx.draw(G)   
	#plt.savefig('Graph.png')
	for i in range(R1 + 1):
		spisok.append(list(combinations(list(G.edges), i)))
	pG = nx.Graph()
	summ = []
	p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for k in p:
		x1 = ((2*k**2 - k**4) + k* (2*k- k**2) - k * (2*k- k**2)* (2*k**2 - k**4))
		x2 = ((2*k - k**2)** 2 + k *(2*k- k**2) - k * (2*k- k**2)* ((2*k - k**2)** 2))
		x3 = (k ** 2 + (2*k - k**2) * (k + (2*k - k**2) - k * (2*k - k**2)) - k ** 2 * (2*k - k**2)  * (k + (2*k - k**2)  - k * (2*k - k**2) ))
		x4 = ((2 * k - k**2 + (2*k - k**2) - (2*k - k**2) * (2*k - k**2)) * ((2*k - k**2)  + k - k * (2*k - k**2) ))
		P.append((x1*(1-k) + x2*k)*(1-k) + (x3*(1-k) +x4*k)*k)
		#P.append((k**(4)*(2*k- k**2)-k**(3)*(2*k- k**2)- k**(2) * y**(2) - k**(3) + k*y**(2) + k*y)**2)
		sum1 = 0
		for i in spisok:
			for j in range(len(i)):
				pG.add_nodes_from(N)
				for l in range(len(i[j])):
					pG.add_edge(i[j][l][0], i[j][l][1])
				if nx.has_path(pG, vb, ve):
					sum1 += k**pG.number_of_edges() * (1 - k)**(R1 - pG.number_of_edges())
				pG.clear()
		summ.append(sum1)
		sum1 = 0	
	for o in range(len(p)):
		print(str(p[o]) + " : " + str(P[o]) + " : " + str(summ[o]))

	_, ax = plt.subplots()
	ax.plot(p, P)
	ax.plot(p, summ)
	plt.show()
	third = lab2_fast(p, summ)
	plt.plot([i / 10 for i in range(len(P))], third, label="Ускоренный")
	plt.legend()
	plt.grid(True)
	#Plot(p, P, "вероятность", "Декомпозиция", "Через декомпозицию")
	#Plot(p, summ, "вероятность", "подграфы", "Через подграфы")
	plt.show()

