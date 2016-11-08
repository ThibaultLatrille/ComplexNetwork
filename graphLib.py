""" IXXI graph library """
from random import random
from random import sample
from collections import deque
import numpy as np
from math import factorial
import matplotlib.pyplot as plt

def nCr(n,r):
    return factorial(n) / (factorial(r) * factorial(n-r))

def ER_np(n, p):
    """ create and returns a Erdos-Renyi
    G_n,p random graph -
    where n is the number of vertices
    and p the probability of puting
    an edge between each two vertices
     """
    graph = Graph({})
    for vertex in range(n):
        graph.add_vertex(vertex)
    for in_vertex in range(n):
        for out_vertex in range(n):
            if in_vertex < out_vertex:
                if random() < p:
                    graph.add_edge((in_vertex, out_vertex))
    return graph


def ER_nm(n, m):
    """ create and returns a Erdos-Renyi
    G_n,m random graph -
    where n is the number of vertices
    and m
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

# =========================
#       GRAPH CLASS
# =========================


class Graph(object):
    def __init__(self, graph_dict={}):
        """ initializes a graph object """
        self.__graph_dict = graph_dict

    def dict(self):
        return self.__graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return sorted(list(self.__graph_dict.keys()))

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def __len__(self):
        return len(self.edges())

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
        self.__graph_dict, a key "vertex" with an empty
        list as a value is added to the dictionary.
        Otherwise nothing has to be done.
        """
        if vertex not in self.vertices():
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ It the edge is not in self.__graph_dict,
        the vertices of the edge are added to each other keys
        The function assumes that edge is of type set, tuple or list;
        (no multiple edges)
        """
        if edge[1] not in self.__graph_dict[edge[0]]:
            self.__graph_dict[edge[0]].append(edge[1])
            self.__graph_dict[edge[1]].append(edge[0])

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
        """ returns as a non-increasing list sequence of the vertex degrees of a graph """
        return [degree for _, degree in sorted(self.vertex_degree(), key=lambda x: x[1], reverse=True)]

    def degree_distribution(self):
        """ returns the degree distribution"""
        return 

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
        edges = self.edges()
        n = len(self.vertices())
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and (self.vertices()[i], self.vertices()[j]) in edges:
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
        return zip(components, map(lambda e: len(e), components))

    def connected_component_elements(self):
        """ returns a list of the vertices list of each connected components of a graph """
        return map(lambda x: x[0], self.connected_components())

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
    m = int(p * nCr(n, 2))
    return float(len(ER_np(n, p).edges())) / m


def er_degree_distribution(n, p):
    return list(map(lambda e: nCr(n-1, e)*(p**e)*((1-p)**(n-e-1)), range(n)))


def edge_count_test(p, x):
    return list(map(lambda e: compare_edge_count(int(e), p), x))


n = 100
plt.plot(range(n), er_degree_distribution(n, 0.5), linewidth=2)
plt.xscale('log')
plt.show()

# x = np.logspace(1, 2.5, 100)
# plt.plot(x, edge_count_test(0.05, x), linewidth=2)
# plt.xscale('log')
# plt.show()

print("\n Density of complete graph:")
np_graph = ER_np(50, .5)
print(np_graph.density())

print("\n Density of complete graph:")
nm_graph = ER_nm(3000, int(10000))
print(nm_graph.density())


print(sample(range(10), 3))