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
# 10 vertices and 15 edges
UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 15)

# Draw the graph to a plot, counting vertices
snap.DrawGViz(UGraph, snap.gvlNeato, "graph_rdm_undirected.png", "Undirected Random Graph", True)

#Plot the out degree distrib
snap.PlotOutDegDistr(UGraph, "graph_rdm_undirected", "Undirected graph - out-degree Distribution")


# Compute and print the list of all edges
for NI in UGraph.Nodes():
    for Id in NI.GetOutEdges():
        print "edge (%d %d)" % (NI.GetId(), Id)


CntV = snap.TIntPrV()
s.GetOutDegCnt(UGraph, CntV)
for p in CntV:
     print "degree %d: count %d" % (p.GetVal1(), p.GetVal2())


#snap.SaveEdgeList(graph_1, "test.txt", "Save as tab-separated list of edges")