""" IXXI graph library """
from random import random
from collections import deque


def random_graph(n, p):
    graph = Graph({})
    for vertex in range(n):
        graph.add_vertex(vertex)
    for in_vertex in range(n):
        for out_vertex in range(n):
            if in_vertex < out_vertex:
                if random() < p:
                    graph.add_edge((in_vertex, out_vertex))
    return graph


class Graph(object):
    def __init__(self, graph_dict={}):
        """ initializes a graph object """
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

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
        """ assumes that edge is of type set, tuple or list;
        (no multiple edges)
        """
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
        return [(vertex, len(self.__graph_dict[vertex])) for vertex in self.vertices()]

    def degree_sequence(self):
        return [degree for _, degree in sorted(self.vertex_degree(), key=lambda x: x[1], reverse=True)]

    def erdos_gallai(self, sequence):
        n = len(sequence)
        if sum(sequence) % 2 == 1:
            return False
        for k in range(1, n+1):
            if sum(sequence[:k]) > k*(k-1) + sum([min(d, k) for d in sequence[k:n]]):
                return False
        return True

    def find_isolated_vertices(self):
        return [vertex for vertex, degree in self.vertex_degree() if degree == 0]

    def density(self):
        nbr_edges, nbr_vertices = len(self.edges()), len(self.vertices())
        return 2 * nbr_edges / (nbr_vertices * (nbr_vertices - 1))

    def connected_components(self):
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
                    if not visited.get(vertex):
                        queue.append(vertex)
                    visited[vertex] = True
            set_vertices -= set(visited.keys())
            components.append(list(visited.keys()))
        return components

    def spanning_tree(self):
        return [self.component_spanning_tree(component).edges() for component in self.connected_components()]

    def component_spanning_tree(self, component):
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

    def __str__(self):
        """ A better way for printing a graph """
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
            res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

if __name__ == "__main__":
    G = {
      "a": ["c", "d", "g"],
      "b": ["c", "f"],
      "c": ["a", "b", "d", "f"],
      "d": ["a", "c", "e", "g"],
      "e": ["d"],
      "f": ["b", "c"],
      "g": ["a", "d"]
    }
    graph = Graph(G)
    print("Vertices of graph:")
    print(graph.vertices())
    print("Edges of graph:")
    print(graph.edges())
    print("Degrees of graph:")
    print(graph.vertex_degree())
    graph.add_vertex("h")
    print("Find isolated vertices:")
    print(graph.find_isolated_vertices())
    print("Degree sequence:")
    print(graph.degree_sequence())
    print("Erdos-Gallai test:")
    print(graph.erdos_gallai(graph.degree_sequence()))

    print("Density of empty graph:")
    empty_graph = random_graph(50, 0.)
    print(empty_graph.density())
    print(empty_graph.erdos_gallai(empty_graph.degree_sequence()))

    print("Density of complete graph:")
    complete_graph = random_graph(50, 1.)
    print(complete_graph.density())
    print(complete_graph.erdos_gallai(complete_graph.degree_sequence()))

    print("Density of random graph:")
    random_graph = random_graph(50, 0.05)
    print(random_graph.density())
    print(random_graph.erdos_gallai(random_graph.degree_sequence()))
    print(random_graph.connected_components())
    print(random_graph.spanning_tree())
