import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




#Algorithms
# all algorithms work on box with boundary size 1x1

#Random layout
def random_layout(n):
    return(np.asarray(np.random.random((n,2))))

# Spring-Embedder
def Fruchterman(G,size=0.5):
    return nx.fruchterman_reingold_layout(G,scale=size,center=(size,size))

def Kamada(G,size=0.5):
    return nx.kamada_kawai_layout(G,scale=size,center=(size,size))

# Our Implementations


from OurAlgorithm import Our_alg , Params

from DH import DH

from GDsquared import GD_squared

from Bigangle import bigangle_layout
def Bigangle(G,iterations=50,size=0.5,pos=None):
    return np.asarray(list(bigangle_layout(G,iterations=iterations,pos=pos,scale=size).values()))



# Set weights for individual criteria

crit_weights = { 'stress' :1/6, 
                   'crossangle' :1/6, 
                   'idealedge' : 1/6, 
                   'angularres' : 1/6, 
                   'vertexres' : 1/6,
                   'crossnum' : 1/6 } 

# Create /Load graph -> most algorithms require (G,pos) as input, where G is nx.Graph() and pos is np.array of dimension(|V|,2)

G = nx.complete_graph(8)
pos_r = random_layout(G.number_of_nodes())
pos_f = Fruchterman(G)
pos_k = Kamada(G)
#Transform dict into array
pos_arr = np.asarray(list(pos_k.values()))
#set timelimit
timelimit = 3

pos_big = Bigangle(G,pos=pos_arr)
pos_dh = DH(G,pos=pos_arr,timelimit=timelimit,lambdas=np.array([0.2,0.2,0.2,0.2,0.2]),height = 1.1,width = 1.1) #boundary is further away to normalize boundary contribution
pos_gd = GD_squared(G,pos = pos_arr,timelimit=timelimit,crit_weights = crit_weights)
pos_o = Our_alg(G,pos_arr,timelimit,crit_weights = crit_weights,height=1,width=1)


nx.draw(G,pos_o,with_labels=True)
plt.show()
