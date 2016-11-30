""" IXXI graph library """
from random import random
from random import sample
from random import choice
from collections import deque
import numpy as np
from math import factorial
import matplotlib.pyplot as plt


def nbr_combination(n, r):
    return factorial(n) / (factorial(r) * factorial(n-r))


def er_np(n, p):
    """ create and returns a Erdos-Renyi G_n,p random graph,
    where n is the number of vertices and p the probability of puting
    an edge between each two vertices.
    """
    assert 0. <= p <= 1.
    graph = Graph({})
    for vertex in range(n):
        graph.add_vertex(vertex)
    for in_vertex in range(n):
        for out_vertex in range(n):
            if in_vertex < out_vertex:
                if random() < p:
                    graph.add_edge((in_vertex, out_vertex))
    return graph


def er_nm(n, m):
    """ Create and returns a Erdos-Renyi G_n,m random graph,
    where n is the number of vertices and m the number of edges.
    """
    m, n = int(m), int(n)
    assert (n * (n-1) / 2) >= m, "Too many edges for the number of vertices"
    graph = Graph({})
    for vertex in range(n):
        graph.add_vertex(vertex)

    for in_vertex, out_vertex in sample([(v_i, v_o)
                                         for v_i in set(range(n))
                                         for v_o in set(range(n))
                                         if v_i < v_o], m):
            graph.add_edge((in_vertex, out_vertex))
    return graph


def unique_choice(seq, not_element, not_seq):
    """ Return 1 unique elements from seq. Must satisfy that this element is different
    to 'not_element' and must not be in 'not_seq'
    """
    target = 0
    while 1:
        target = choice(seq)
        if target != not_element and target not in not_seq:
            break
    return target


def watts_strogatz(n=1000, k=50, beta=0.5):
    """ Create and returns a Watts-Strogatz random graph,
        where n is the number of vertices, k the number of adjacent edges,
        and beta the probability of rewiring.
    """
    assert 0. <= beta <= 1., "Watts-Strogatz graph must have 0. <= beta <= 1."
    assert k % 2 == 0, "Watts-Strogatz graph must have k be an even integer"
    assert 1 < k < n, "Watts-Strogatz graph must have 1 < k < n"
    graph = Graph({})
    graph.add_vertices(n)
    for vertex in graph.vertices():
        for ring_vertex in range(int(k/2)):
            graph.add_edge((vertex, (vertex + (ring_vertex + 1)) % n))
            graph.add_edge((vertex, (vertex - (ring_vertex + 1)) % n))
    if beta > 0.:
        for v_i in graph.vertices():
            for v_j in graph.dict()[v_i]:
                if v_i < v_j and random() <= beta:
                    v_l = unique_choice(graph.vertices(), v_i, graph.dict()[v_i])
                    graph.rewire_edge((v_i, v_j), (v_i, v_l))
    return graph


def unique_sample(seq, m):
    """ Return m unique elements from seq.

    This differs from sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets = set()
    while len(targets) < m:
        x = choice(seq)
        targets.add(x)
    return targets


def barabasi_albert(mo=2, m=2, n=1000):
    """ Create and returns a Barabasi-Albert random graph,
        where n is the number of vertices and k the number of edges for each added vertex.
    """
    assert 1 <= m <= mo <= n, "m < mo < n"
    graph = er_np(mo, 1)
    repeated_vertices = [v for edge in graph.edges() for v in edge]
    for vertex in range(mo, n):
        graph.add_vertex(vertex)
        targets = unique_sample(repeated_vertices, m)
        for v_target in targets:
            graph.add_edge((vertex, v_target))
        repeated_vertices.extend(targets)
        repeated_vertices.extend([vertex]*m)
    return graph

# =========================
#       GRAPH CLASS
# =========================


class Graph(object):
    def __init__(self, graph_dict):
        """ initializes a graph object """
        self.__graph_dict = graph_dict

    def dict(self):
        return self.__graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def sorted_vertices(self):
        """ returns the vertices of a graph """
        return sorted(list(self.__graph_dict.keys()))

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def __len__(self):
        return len(self.vertices())

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
        self.__graph_dict, a key "vertex" with an empty
        list as a value is added to the dictionary.
        Otherwise nothing has to be done.
        """
        if vertex not in self.vertices():
            self.__graph_dict[vertex] = []

    def add_vertices(self, nbr_vertices):
        """ returns the vertices of a graph """
        for vertex in range(nbr_vertices):
            self.add_vertex(vertex)

    def add_edge(self, edge):
        """ It the edge is not in self.__graph_dict,
        the vertices of the edge are added to each other keys
        The function assumes that edge is of type set, tuple or list;
        (no multiple edges)
        """
        if edge[1] not in self.__graph_dict[edge[0]]:
            self.__graph_dict[edge[0]].append(edge[1])
        if edge[0] not in self.__graph_dict[edge[1]]:
            self.__graph_dict[edge[1]].append(edge[0])

    def remove_edge(self, edge):
        """ It the edge is not in self.__graph_dict,
        the vertices of the edge are added to each other keys
        The function assumes that edge is of type set, tuple or list;
        (no multiple edges)
        """
        if edge[1] in self.__graph_dict[edge[0]]:
            self.__graph_dict[edge[0]].remove(edge[1])
        if edge[0] in self.__graph_dict[edge[1]]:
            self.__graph_dict[edge[1]].remove(edge[0])

    def rewire_edge(self, edge1, edge2):
        self.remove_edge(edge1)
        self.add_edge(edge2)

    def __generate_edges(self):
        """ A static method generating the edges of the
        graph "graph". Edges are represented as sets
        two vertices (no loop)
        """
        edges = []
        for vertex_in in self.vertices():
            for vertex_out in self.__graph_dict[vertex_in]:
                if vertex_in < vertex_out:
                    edges.append((vertex_in, vertex_out))
        return edges

    def vertex_degree(self):
        """ returns a list of sets containing the
        name and degree of each vertex of a graph """
        return [(vertex, len(self.__graph_dict[vertex])) for vertex in self.vertices()]

    def degree_sequence(self):
        """ returns as a decreasing list sequence of the vertex degrees of a graph """
        return [degree for _, degree in sorted(self.vertex_degree(), key=lambda x: x[1], reverse=True)]

    def degree_distribution(self):
        """ returns the degree distribution"""
        bincount = np.bincount([len(self.__graph_dict[vertex]) for vertex in self.vertices()])
        return bincount.astype(np.float) / sum(bincount)

    def erdos_gallai(self, sequence):
        """ for a given degree sequence check if it can be
         realised by a simple graph
         returns a boolean"""
        n = len(sequence)
        if sum(sequence) % 2 == 1:
            return False
        for k in range(1, n + 1):
            if not sum(sequence[:k]) <= k * (k - 1) + sum([min(d, k) for d in sequence[k:n]]):
                return False
        return True

    def find_isolated_vertices(self):
        """ returns the list of zero-degree vertices of a graph """
        return [vertex for vertex, degree in self.vertex_degree() if degree == 0]

    def density(self):
        """ returns the density of a graph """
        nbr_edges, nbr_vertices = float(len(self.edges())), float(len(self.vertices()))
        return 2 * nbr_edges / (nbr_vertices * (nbr_vertices - 1))

    def adjacency_matrix(self):
        """ returns the ajacency matrix of a graph
         in the form of a numpy array"""
        v_dict = {}
        for i, v in enumerate(self.vertices()):
            v_dict[v] = i
        n = len(v_dict)
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for v_i, v_j in self.edges():
            i, j = v_dict[v_i], v_dict[v_j]
            adj_matrix[i][j], adj_matrix[j][i] = 1, 1
        return np.array(adj_matrix)

    def global_clustering_coeff(self):
        """ returns the global clustering coefficient of a graph """
        adj_mtrx = self.adjacency_matrix()
        path_length_two = np.linalg.matrix_power(adj_mtrx, 2)
        closed_triple_mtrx = np.multiply(adj_mtrx, path_length_two)
        n = len(self.vertices())
        nbr_closed_triple, nbr_all_triple = 0.0, 0.0  # float because of division
        nbr_closed_triple += sum(closed_triple_mtrx[i][e] for i in range(n) for e in range(n) if i != e)
        nbr_all_triple += sum(path_length_two[i][e] for i in range(n) for e in range(n) if i != e)
        # instead of not computing the diagonal
        # we could have substract np.trace(path_length_two) from nbr_triple
        return nbr_closed_triple / nbr_all_triple if nbr_all_triple != 0 else 0  # avoid 0 division

    def shortest_path(self, a, b):
        """ returns the shortest path distance between two given vertices a, b"""
        queue = deque()
        distance = {a: 0}
        queue.append(a)
        while len(queue) > 0:
            current = queue.popleft()
            for vertex in self.__graph_dict[current]:
                if vertex == b:
                    return distance[current] + 1
                if vertex not in distance:
                    queue.append(vertex)
                    distance[vertex] = distance[current] + 1
        return float("inf")

    def connected_components(self):
        """ returns a list of sets composed of two elements: the vertices list and the size
        of each connected components of a graph """
        components = []
        set_vertices = set(self.vertices())
        queue = deque()
        while len(set_vertices) > 0:
            init = set_vertices.pop()
            visited = {init: True}
            queue.append(init)
            while len(queue) > 0:
                current = queue.popleft()
                for vertex in self.__graph_dict[current]:
                    if vertex not in visited:
                        queue.append(vertex)
                    visited[vertex] = True
            set_vertices -= set(visited.keys())
            components.append(list(visited.keys()))
        return zip(components, [len(e) for e in components])

    def connected_component_elements(self):
        """ returns a list of the vertices list of each connected components of a graph """
        return map(lambda x: x[0], self.connected_components())

    def connected_component_sizes(self):
        """ returns a list of the size of each connected components of a graph """
        return map(lambda x: x[1], self.connected_components())

    def biggest_component_elements(self):
        return max(self.connected_component_elements(), key=len)

    def component_diameter(self, component):
        """ returns the diameter of a given connected component element of a graph"""
        diameter = 0
        for init in component:
            queue = deque()
            distance = {init: 0}
            queue.append(init)
            while len(queue) > 0:
                current = queue.popleft()
                for vertex in self.__graph_dict[current]:
                    if vertex not in distance:
                        queue.append(vertex)
                        distance[vertex] = distance[current] + 1
            diameter = max((diameter, max(distance.values())))

        return diameter

    def component_average_path_length(self, component):
        """ returns the diameter of a given connected component element of a graph"""
        diameter = []
        for init in component:
            queue = deque()
            distance = {init: 0}
            queue.append(init)
            while len(queue) > 0:
                current = queue.popleft()
                for vertex in self.__graph_dict[current]:
                    if vertex not in distance:
                        queue.append(vertex)
                        distance[vertex] = distance[current] + 1
            diameter.extend(distance.values())
        return np.mean([d for d in diameter if d != 0])

    def biggest_component_average_path_length(self):
        """ returns the diameter of the biggest connected component of a graph """
        return self.component_average_path_length(max(self.connected_component_elements(), key=len))

    def forest_diameters(self):
        """ returns the list of the diameter of each connected components of a graph """
        return [self.component_diameter(component) for component in self.connected_component_elements()]

    def biggest_component_diameter(self):
        """ returns the diameter of the biggest connected component of a graph """
        return self.component_diameter(max(self.connected_component_elements(), key=len))

    def component_spanning_tree(self, component):
        """ returns the spanning tree of a given connected component of a graph """
        spanning_tree = Graph({})
        queue = deque()
        spanning_tree.add_vertex(component.pop())
        queue.extend(spanning_tree.vertices())
        while len(queue) > 0:
            current = queue.popleft()
            for vertex in self.__graph_dict[current]:
                if vertex not in spanning_tree.vertices():
                    queue.append(vertex)
                    spanning_tree.add_vertex(vertex)
                    spanning_tree.add_edge((current, vertex))
        return spanning_tree

    def spanning_forest(self):
        """ returns the list of spanning trees of each connected component of a graph """
        return [self.component_spanning_tree(component) for component in self.connected_component_elements()]

    def biggest_component_spanning_tree(self):
        """ returns the spanning tree of a the biggest connected component of a graph """
        return self.component_spanning_tree(max(self.connected_component_elements(), key=lambda c: len(c)))

    def __str__(self):
        """ A better way for printing a graph """
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
            res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res


# =========================
#      IMPORT DATA
# =========================


def file_to_graph(file_path):
    """ import and parse a text file containing an edge list
    then dynamically construct a dictionnary representation of the graph from the edge list"""
    graph_import = Graph({})
    with open(file_path, 'r') as document:
        for line in document:
            vertices = line.split()
            if not vertices:  # empty line?
                continue
            if len(vertices) % 2 == 0:
                graph_import.add_vertex(vertices[0])
                graph_import.add_vertex(vertices[1])
                graph_import.add_edge((vertices[0], vertices[1]))
            else:
                for v in vertices:
                    graph_import.add_vertex(v)
                    if v != vertices[0]:
                        graph_import.add_edge((vertices[0], v))
    return graph_import


# file_list = ('zachary-connected.txt', 'graph_100n_1000m.txt', 'graph_1000n_4000m.txt')
# print(file_list)
# for file_path in file_list:
#     graph = file_to_graph(file_path)
#     print("\n Number of vertices of graph:")
#     print(len(graph.vertices()))
#     print("\n Number of edges of graph:")
#     print(len(graph.edges()))
#     print("\n Density of graph:")
#     print(graph.density())
#     print("\n Diameter of graph:")
#     print(graph.biggest_component_diameter())
#     print("\n Clustering coefficient of the graph:")
#     print(graph.global_clustering_coeff())


# =========================
#          TEST
# =========================

def compare_edge_count(n, p):
    m = int(p * nbr_combination(n, 2))
    return float(len(er_np(n, p).edges())) / m


def plot_compare_edge_count(n, p):
    n_max = max(n, 20)
    x = np.logspace(1, np.log10(n_max), 50)
    plt.scatter(x, [compare_edge_count(int(e), p) for e in x],
                linewidth=2, label="$p={}$".format(p))
    plt.plot((10, n_max), (1, 1), 'r--', label="Theoretical value")
    plt.xlim(10, n_max)
    plt.xscale('log')
    plt.xlabel('$n$')
    plt.ylabel('Compare edge count')
    plt.title("Compare edge count for Erdos-Renyi $G_{n,p}$ \n and $G_{n,m}$ random graph")
    plt.legend()
    plt.show()


def plot_connected_components_erdos_renyi(n):
    p_max = 2.5 / n
    range_p = np.linspace(0, p_max, 100)
    sorted_sizes_list = [[sorted(er_np(int(n), p).connected_component_sizes(), reverse=True) for _ in range(10)]
                         for p in range_p]

    max_len = [np.mean([sorted_sizes[0] for sorted_sizes in sorted_sizes_range])
               for sorted_sizes_range in sorted_sizes_list]
    plt.scatter(range_p, max_len, c='r', label="$Largest\ component$")

    sec_len = [np.mean([(sorted_sizes[1] if len(sorted_sizes) > 1 else 0) for sorted_sizes in sorted_sizes_range])
               for sorted_sizes_range in sorted_sizes_list]
    plt.scatter(range_p, sec_len, c='b', label="$Second\ largest\ component$")

    plt.plot((0, p_max), (np.log(n), np.log(n)), 'r')
    plt.text(p_max*0.90, np.log(n), r'$log(n)$', fontsize=15)
    plt.plot((0, p_max), (n / np.sqrt(n), n / np.sqrt(n)), 'r')
    plt.text(p_max*0.90, n / np.sqrt(n), r'$n^{2/3}$', fontsize=15)
    plt.plot((1./n, 1./n), (0, n), 'b')

    plt.title("$n = {}$".format(n))

    plt.xlabel(r'$p$')
    plt.ylabel(r'$Size\ of\ the\ component$')
    plt.yscale('log')

    plt.plot((0, p_max), (n, n), 'r')
    plt.text(1.01 / n, 1, r'$p=\frac{1}{n}$', fontsize=15)
    plt.xlim(0, p_max)
    plt.ylim(0, n)
    plt.legend()
    plt.show()


def plot_isolated_vertices_erdos_renyi(n):
    th = np.log(n) / n
    p_max = min(1, 1.5 * th)
    p_min = max(0, 0.5 * th)
    range_p = np.linspace(p_min, p_max, 100)
    sorted_sizes_list = [[sorted(er_np(int(n), p).connected_component_sizes(), reverse=True) for _ in range(10)]
                         for p in range_p]

    single_len = [np.mean([len([s for s in sorted_sizes if s == 1]) for sorted_sizes in sorted_sizes_range])
                  for sorted_sizes_range in sorted_sizes_list]
    plt.scatter(range_p, single_len, c='b', label="$Number\ of\ isolated\ vertices$")
    plt.plot((th, th), (0, n), 'b')

    plt.title("$n = {}$".format(n))

    plt.xlabel(r'$p$')
    plt.ylabel(r'$Number\ of\ isolated\ vertices$')

    plt.plot((p_min, p_max), (n, n), 'r')
    plt.xlim(p_min, p_max)
    plt.text(th*1.01, single_len[0]*0.25, r'$p=\frac{log(n)}{n}$', fontsize=15)
    plt.ylim(0, single_len[0])
    plt.legend()
    plt.show()


def plot_degree_distribution_erdos_renyi(n, p):
    # the histogram of the data
    graph = er_np(n, p)
    hist = graph.degree_distribution()

    degrees = range(len(hist))

    plt.scatter(degrees, hist, label='Empirical distribution')
    plt.plot(range(n), [nbr_combination(n-1, e)*(p**e)*((1-p)**(n-e-1)) for e in range(n)],
             'r--', linewidth=1.5, label='Theoretical distribution')

    plt.xscale('log')
    plt.xlim(1, n)
    plt.ylim(0, max(hist))
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    plt.title('Degree distribution')
    plt.legend()

    plt.show()


def plot_watts_strogatz(n, k):
    # the histogram of the data
    list_beta = np.logspace(-4, 0, 20)

    list_graph = [watts_strogatz(n, k, beta) for beta in list_beta]
    wso = watts_strogatz(n, k, 0.)
    max_clustering_coef = wso.global_clustering_coeff()
    list_clustering_coef = [graph.global_clustering_coeff() / max_clustering_coef for graph in list_graph]
    max_diameter = wso.biggest_component_average_path_length()
    list_diameter = [graph.biggest_component_average_path_length() / max_diameter for graph in list_graph]

    plt.scatter(list_beta, list_clustering_coef, c='r', label='Clustering coefficient')
    plt.scatter(list_beta, list_diameter, c='b', label='Average path length')
    plt.legend()

    plt.xscale('log')
    plt.xlabel(r'$\beta$')
    plt.ylim(0, 1)
    plt.xlim(min(list_beta), max(list_beta))
    plt.grid(True)

    plt.show()


def plot_degree_distribution_barabasi_albert(mo, m, n):
    graph = barabasi_albert(mo, m, n)
    hist = graph.degree_distribution()

    degrees = range(len(hist))

    plt.scatter(degrees, hist, label='Empirical distribution')
    plt.plot(degrees[1:], [2.*m*(m+1)/(k*(k+1)*(k+2)) for k in degrees[1:]],
             'r--', linewidth=1.5, label='Theoretical distribution')

    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(degrees[next((i for i, h in enumerate(hist) if h != 0), 1)], len(degrees))
    plt.ylim(min([h for h in hist if h != 0]), max(hist))
    plt.xlabel(r'$k$')
    plt.ylabel(r'$P(k)$')
    plt.title('Degree distribution')
    plt.legend()
    plt.show()


def plot_path_length_barabasi_albert(mo, m, n_max):
    range_n = np.logspace(np.log10(mo), np.log10(n_max), 20)
    list_graph = [barabasi_albert(mo, m, int(n)) for n in range_n]
    list_diameter = [graph.biggest_component_average_path_length() for graph in list_graph]

    plt.scatter(range_n, list_diameter)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\mathrm{Average\ path\ length}$')

    plt.xscale('log')
    plt.xlim(mo, n_max)
    plt.ylim(1, max(list_diameter))

    plt.show()

#plot_connected_components_erdos_renyi(500)
#plot_isolated_vertices_erdos_renyi(500)
plot_compare_edge_count(1000, 0.1)
plot_degree_distribution_erdos_renyi(1000, 0.02)
# plot_watts_strogatz(1000, 10)
# plot_degree_distribution_barabasi_albert(mo=10, m=10, n=5000)
# plot_path_length_barabasi_albert(mo=2, m=2, n_max=1000)
