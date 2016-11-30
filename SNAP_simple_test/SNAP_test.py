#----------------------------------
#       Is SNAP installed?
#----------------------------------

status = False
try:
    import snap
    version = snap.Version
    i = snap.TInt(5)
    if i == 5:
        status = True
except:
    pass

if status:
    print "SUCCESS, your version of Snap.py is %s" % (version)
else:
    print "*** ERROR, no working Snap.py was found on your computer"


# ----------------------------------
#           Graph Testing
# ----------------------------------
import snap as s
import random as rand

# Generate undirected Erdos Reyni random graph
# set up vertices and edges
vertices = 20
edges = 15
u_rndm_graph = snap.GenRndGnm(snap.PUNGraph, vertices, edges)

# Draw the graph to a plot, counting vertices
snap.DrawGViz(u_rndm_graph, snap.gvlNeato, "graph_rdm_undirected.png", "Undirected Random Graph", True)

# Plot the out degree distrib
snap.PlotOutDegDistr(u_rndm_graph, "graph_rdm_undirected", "Undirected graph - out-degree Distribution")


# Compute and print the list of all edges
for vertex_in in u_rndm_graph.Nodes():
    for vertex_out_id in vertex_in.GetOutEdges():
        print "edge (%d %d)" % (vertex_in.GetId(), vertex_out_id)
# Save it to an external file
snap.SaveEdgeList(u_rndm_graph, "Rndm_graph.txt", "Save as tab-separated list of edges")

# Compute degree distribution and save it to an external textfile
degree_vertex_count = snap.TIntPrV()
s.GetOutDegCnt(u_rndm_graph, degree_vertex_count)
file = open("graph_rdm_undirected_degree_distrib.txt", "w")
file.write("#----------------------------------\n")
file.write("#       Degree Distribution        \n")
file.write("#----------------------------------\n")
file.write("\n")
for pairs in degree_vertex_count:
     file.write("vertex degree %d: nmbr vertices with such degree %d \n" % (pairs.GetVal1(), pairs.GetVal2()))
file.close()


# Compute the sizes of the connected component and save it to an external file
Components = snap.TCnComV()
snap.GetSccs(u_rndm_graph, Components)
file_2 = open("graph_rdm_undirected_connected_compo_sizes.txt", "w")
file_2.write("#----------------------------------\n")
file_2.write("#   Size of Connected Components   \n")
file_2.write("#----------------------------------\n")
file_2.write("\n")
file_2.write("Total number of different components = %d\n" % len(Components))
file_2.write("\n")
i = 1
for idx, component in enumerate(Components):
        file_2.write("Size of component #%d : %d\n" % (idx, len(component)))
file_2.close()


# Output the average of the shortest paths, adding more edges to the graph if it's not connected
average_shortest_paths = []
v_in_short_paths_len = []
vertices = u_rndm_graph.Nodes()
len_vertices = sum(1 for _ in vertices)
edge_list = list((item.GetSrcNId(), item.GetDstNId()) for item in u_rndm_graph.Edges())
# is the graph connected?
print snap.IsConnected(u_rndm_graph)
while not snap.IsConnected(u_rndm_graph):
    #in_vertex, out_vertex = rand.sample([(v_i, v_o)
    #                                     for v_i in set(range(len_vertices))
    #                                     for v_o in set(range(len_vertices))
    #                                     if v_i != v_o] and v_i not in [cc for cc in Components, m)
    u_rndm_graph.AddEdge(in_vertex, out_vertex)
print "Now graph is fully connected"

for v_in in vertices:
    id_v_in = v_in.GetId()
    set_v_out = filter(lambda x: x != id_v_in, vertices)
    for v_out in set_v_out:
        v_in_short_paths_len.append(snap.GetShortPath(u_rndm_graph, id_v_in, v_out.GetId()))
        #les reduce dessous c'est juste un tricks pour faire des sommes de toute la liste (;
        v_in_average__short_paths_len = reduce(lambda x, y: x+y, v_in_short_paths_len) / float(len(v_in_short_paths_len))
    average_shortest_paths.append(v_in_average__short_paths_len)
global_average_shortest_path = reduce(lambda x, y: x+y, average_shortest_paths) / float(len(average_shortest_paths))
print global_average_shortest_path
