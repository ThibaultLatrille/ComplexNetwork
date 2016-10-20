""" IXXI graph library """
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
