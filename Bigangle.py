#This is not completely the BIGANGLE algorithm, since we use FR instead of "normal" spring embedder -> authors differentiate between the two
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from geometry import *
from utils import *



def _rescale_layout(pos, scale=1.):
    # rescale to [0, scale) in each axis

    # Find max length over all dimensions
    maxlim=0
    for i in range(pos.shape[1]):
        pos[:,i] -= pos[:,i].min() # shift min to zero
        maxlim = max(maxlim, pos[:,i].max())
    if maxlim > 0:
        for i in range(pos.shape[1]):
            pos[:,i] *= scale / maxlim
    return pos


def bigangle_layout(G, dim=2, k=None,
                                pos=None,
                                fixed=None,
                                iterations=50,
                                weight='weight',
                                scale=1.0,
                                center=None):


    if len(G) == 0:
        return {}

    if pos is not None:
        pos_arr = pos
    #     # Determine size of existing domain to adjust initial positions
    #     #pos_coords = np.array(list(pos.values()))
    #     pos_coords = pos
    #     min_coords = pos_coords.min(0)
    #     domain_size = pos_coords.max(0) - min_coords
    #     shape = (len(G), dim)
    #     pos_arr = np.random.random(shape) * domain_size + min_coords
    #     for i,n in enumerate(G):
    #         if n in pos:
    #             pos_arr[i] = np.asarray(pos[n])
    # else:
    #     pos_arr=None

    A = nx.to_numpy_matrix(G, weight=weight)

    pos =  _bigangle(A, list(G.edges()),G.number_of_edges(), dim, k, pos_arr, fixed, iterations)

    if fixed is None:
        pos = _rescale_layout(pos, scale)
        if center is not None:
            pos += np.asarray(center) - 0.5 * scale

    return dict(zip(G,pos))




def _bigangle(A,edges,m,dim=2,k=None,pos=None,fixed=None,iterations=50):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np
    try:
        nnodes,_=A.shape
    except AttributeError:
        raise nx.NetworkXError('Expected Adjacency matrix')

    A=np.asarray(A) # make sure we have an array instead of a matrix

    if pos is None:
        # random initial positions
        pos=np.asarray(np.random.random((nnodes,dim)),dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos=pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k=np.sqrt(1.0/nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # Calculate domain in case our fixed positions are bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1]))*0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt=t/float(iterations+1)
    delta = np.zeros((pos.shape[0],pos.shape[0],pos.shape[1]),dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        for i in range(pos.shape[1]):
            delta[:,:,i]= pos[:,i,None]-pos[:,i]
        # distance between points
        distance=np.sqrt((delta**2).sum(axis=-1))
        # enforce minimum distance of 0.01
        distance=np.where(distance<0.01,0.01,distance)
        # displacement "force"
        displacement=np.transpose(np.transpose(delta)*\
                                  (k*k/distance**2-A*distance/k))\
                                  .sum(axis=1)

        max_mag = np.max(np.linalg.norm(displacement,axis=1))
        #dynamically set the values to be of the same magnitude
        # constant factors empirically found out
        k_cos = max_mag*0.05
        k_sin = max_mag*0.3

        crossings = compute_crossings(edges,pos,m)

        for c in crossings:
            ca = crossing_angle_between_segments(c[0],c[1],pos)
            if ca >= 89.9:
              pass
            else:
              force_mag = math.cos(math.radians(ca)) * k_cos  #k_cos
              vec_e1 = np.array([pos[c[0][1]][0] - pos[c[0][0]][0] ,pos[c[0][1]][1] - pos[c[0][0]][1]])
              vec_e2 = np.array([pos[c[1][1]][0] - pos[c[1][0]][0] ,pos[c[1][1]][1] - pos[c[1][0]][1]])
              #normalize
              vec_e1 = vec_e1 / np.linalg.norm(vec_e1)
              vec_e2 = vec_e2 / np.linalg.norm(vec_e2)
              vec_r1 = vec_e1 * -1
              vec_r2 = vec_e2 * -1
              theta_1 = np.degrees(np.arccos(np.dot(vec_e1, vec_e2)/(np.linalg.norm(vec_e1)*np.linalg.norm(vec_e2))))
              theta_2 = np.degrees(np.arccos(np.dot(vec_e1, vec_r2)/(np.linalg.norm(vec_e1)*np.linalg.norm(vec_r2))))
              if theta_1 > theta_2:
                  displacement[c[0][1]] += vec_e2 * force_mag
                  displacement[c[0][0]] += vec_r2 * force_mag
                  displacement[c[1][1]] += vec_e1 * force_mag
                  displacement[c[1][0]] += vec_r1 * force_mag
              else:
                  displacement[c[0][0]] += vec_e2 * force_mag
                  displacement[c[0][1]] += vec_r2 * force_mag
                  displacement[c[1][0]] += vec_e1 * force_mag
                  displacement[c[1][1]] += vec_r1 * force_mag

        for i in range(pos.shape[0]):
            circ_order = circular_ordering_edges(A,i,pos)
            deg = len(circ_order)
            opt_angle = 360 / deg
            if deg > 2:
              for j in range(deg):
                  current_e = circ_order[j]
                  n1 = current_e[0] if current_e[0] != i else current_e[1]
                  next_e = circ_order[(j+1) % deg]
                  n2 = next_e[0] if next_e[0] != i else next_e[1]
                  vec_e1 = np.array([pos[n1][0] - pos[i][0] ,pos[n1][1] - pos[i][1]])
                  vec_e2 = np.array([pos[n2][0] - pos[i][0] ,pos[n2][1] - pos[i][1]])
                  vec_e1 = vec_e1 / np.linalg.norm(vec_e1)
                  vec_e2 = vec_e2 / np.linalg.norm(vec_e2)
                  theta = np.degrees(np.arccos(np.dot(vec_e1, vec_e2)/(np.linalg.norm(vec_e1)*np.linalg.norm(vec_e2))))
                  theta = min(theta,360-theta)
                  force_mag = np.sin(np.radians((opt_angle - theta)/2)) * k_sin
                  displacement[n1]-= np.array([-vec_e1[1],vec_e1[0]]) * force_mag
                  displacement[n2]+= np.array([-vec_e2[1],vec_e2[0]]) * force_mag

            elif deg == 2:
                #Special Case for degree two nodes
              e1 = circ_order[0]
              e2 = circ_order[1]
              n1 = e1[0] if e1[0] != i else e1[1]
              n2 = e2[0] if e2[0] != i else e2[1]
              vec_e1 = np.array([pos[n1][0] - pos[i][0] ,pos[n1][1] - pos[i][1]])
              vec_e2 = np.array([pos[n2][0] - pos[i][0] ,pos[n2][1] - pos[i][1]])
              vec_e1 = vec_e1 / np.linalg.norm(vec_e1)
              vec_e2 = vec_e2 / np.linalg.norm(vec_e2)
              theta = np.degrees(math.acos(np.dot(vec_e1, vec_e2)/(np.linalg.norm(vec_e1)*np.linalg.norm(vec_e2))))
              theta = min(theta,360-theta)
              force_mag = np.sin(np.radians((opt_angle - theta)/2)) * k_sin
              displacement[n1]-= np.array([-vec_e1[1],vec_e1[0]]) * force_mag
              displacement[n2]+= np.array([-vec_e2[1],vec_e2[0]]) * force_mag
        # update positions
        length=np.sqrt((displacement**2).sum(axis=1))
        length=np.where(length<0.01,0.01,length)
        delta_pos=np.transpose(np.transpose(displacement)*t/length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed]=0.0
        pos+=delta_pos
        # cool temperature
        t-=dt
        if fixed is None:
            pos = _rescale_layout(pos)
    return pos
