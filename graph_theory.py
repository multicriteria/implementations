import numpy as np
import networkx as nx
#Helper functions

def neighbors(A,node):
    row = A[node]
    return(np.where(row!=0)[0])


def graph_distance(G,n):
    length= dict(nx.all_pairs_shortest_path_length(G))
    res = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            res[i][j] =  res[j][i] =  length[i][j]
    return res



def adjacent_edges(G,i):
    res = []
    adjacent_edges = list(G.edges(i))
    for (u,v) in adjacent_edges:
        res.append((min(u,v),max(i,v)))
    return res
