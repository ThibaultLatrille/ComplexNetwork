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
#      Random Graph (10, 15)
# ----------------------------------
import snap as s

UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 15)
snap.DrawGViz(UGraph, snap.gvlNeato, "graph_undirected.png", "graph 2", True)


for NI in UGraph.Nodes():
    for Id in NI.GetOutEdges():
        print "edge (%d %d)" % (NI.GetId(), Id)


CntV = snap.TIntPrV()
s.GetOutDegCnt(UGraph, CntV)
for p in CntV:
     print "degree %d: count %d" % (p.GetVal1(), p.GetVal2())


#snap.SaveEdgeList(graph_1, "test.txt", "Save as tab-separated list of edges")