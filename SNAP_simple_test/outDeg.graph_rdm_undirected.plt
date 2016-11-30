#
# Undirected graph - out-degree Distribution. G(10, 15). 4 (0.4000) nodes with out-deg > avg deg (3.0), 0 (0.0000) with >2*avg.deg (Wed Nov 30 19:48:57 2016)
#

set title "Undirected graph - out-degree Distribution. G(10, 15). 4 (0.4000) nodes with out-deg > avg deg (3.0), 0 (0.0000) with >2*avg.deg"
set key bottom right
set logscale xy 10
set format x "10^{%L}"
set mxtics 10
set format y "10^{%L}"
set mytics 10
set grid
set xlabel "Out-degree"
set ylabel "Count"
set tics scale 2
set terminal png font arial 10 size 1000,800
set output 'outDeg.graph_rdm_undirected.png'
plot 	"outDeg.graph_rdm_undirected.tab" using 1:2 title "" with linespoints pt 6
