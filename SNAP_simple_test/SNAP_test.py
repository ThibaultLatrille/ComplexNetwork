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

# Generate undirected Erdos Reyni random graph
# set up vertices and edges
vertices = 20
edges = 10
u_rndm_graph = snap.GenRndGnm(snap.PUNGraph, vertices, edges)

# Draw the graph to a plot, counting vertices
snap.DrawGViz(u_rndm_graph, snap.gvlNeato, "graph_rdm_undirected.png", "Undirected Random Graph", True)

# Plot the out degree distrib
snap.PlotOutDegDistr(u_rndm_graph, "graph_rdm_undirected", "Undirected graph - out-degree Distribution")


# Compute and print the list of all edges
for vertex in u_rndm_graph.Nodes():
    for Id in vertex.GetOutEdges():
        print "edge (%d %d)" % (vertex.GetId(), Id)
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
file_2.write("Total number of different components = %d\n" % Components.Len())
file_2.write("\n")
i = 1
for idx, val in enumerate(Components):
        file_2.write("Size of component #%d : %d\n" % (idx, val.Len()))
file_2.close()


