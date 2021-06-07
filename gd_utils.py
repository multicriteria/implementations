import math
import networkx as nx
import numpy as np
import torch
import time
from geometry import *
from graph_theory import *
from torch import autograd
def compute_crossings(edges,pos,m):
    crossings = []
    for i in range(m-1):
        for j in range(i+1,m):
            if not(edges[i][0] in edges[j] or edges[i][1] in edges[j]):
                p1 = pos[edges[i][0]]
                q1 = pos[edges[i][1]]
                p2 = pos[edges[j][0]]
                q2 = pos[edges[j][1]]
                if do_intersect(p1,q1,p2,q2):
                    crossings.append((edges[i],edges[j]))
    return(crossings,len(crossings))


def node_distance(pos_tensor):
    return(torch.norm(pos_tensor[:, None] - pos_tensor, dim=2))

# Stress
# compute graph-dist matrix once since it never changes
# graph_dist is nxn symmetric matrix with 0 diagonal
# weight is nxn symmetric matrix matrix
def stress_loss(node_dist,graph_dist,weight):
    l = 1 / torch.max(graph_dist)
    return((torch.sum(torch.mm(weight, (node_dist-l*graph_dist)**2))/2)/node_dist.shape[0]**2)


def ideal_edge_length_loss(adj,node_dist,m):
    # filter distances only if edge exists
    edge_lengths = node_dist *  adj
    avg_edge_length = torch.sum(edge_lengths)/m
    edges = edge_lengths[edge_lengths != 0]
    return(torch.sqrt(torch.square((edges - avg_edge_length) / avg_edge_length).sum()/ m))




def edge_pairs_for_crossings(edges):
    pairs = []
    m = len(edges)
    for i in range(m-1):
        for j in range(i+1,m):
            #check that they are not adjacent
            if edges[i][0] not in edges[j]:
                pairs.append((edges[i],edges[j]))
    return pairs


def get_angle_from_cosine(cos_val):
    epsilon=1e-6
    c_angle = torch.acos(torch.clamp(cos_val, -1 + epsilon, 1 - epsilon))
    ang_deg = torch.rad2deg(c_angle)
    if ang_deg-180>=0:
        return (360 -ang_deg)
    else:
        return (min(ang_deg,180-ang_deg))


def cosine_of_angle(e1,e2,pos):
    idx_1 = torch.tensor([e1[0],e1[1]])
    idx_2 = torch.tensor([e2[0],e2[1]])
    s1=pos[idx_1[1]] - pos[idx_1[0]]
    s2=pos[idx_2[1]] - pos[idx_2[0]]
    # create
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return(cos(s1,s2))


def crossing_angle_loss(crossings,pos):
    # assume that crossings contains list of all crossings, i.e. list of tuple of edges
    # squared cosine of crossing angles
    #Quality meassurement Q = max\theta |\theta - 90 deg | / 90 deg -> bounded between [0,1]
    loss = torch.tensor(0).float()
    if len(crossings) == 0:
        return loss
    else:
        worst_angle = 91
        for (e1,e2) in crossings:
            c_a = cosine_of_angle(e1,e2,pos)
            loss+=c_a**2
            current_angle = get_angle_from_cosine(c_a)
            #print(current_angle)
            if current_angle < worst_angle:
                worst_angle = current_angle
                involved_edges = (e1,e2)
    return loss/len(crossings) ,worst_angle





def angular_resolution(adj,n,pos,max_degree,s=1):
#Quality meassurement Q = ANR / 2pi / dmax -> bound it to [0,1]
    min_angle = 181
    loss = torch.tensor(0).float()
    counter = 0
    for j in range(n):
        #for every pair of adjacent edges, compute angle
        # this computes actually a bit too much
        adjacent_edges = adj[j].nonzero(as_tuple=True)[0]
        deg_j = len(adjacent_edges)
        for ind_i in range(deg_j-1):
            for ind_k in range(ind_i+1,deg_j):
                counter+=1
                i = adjacent_edges[ind_i]
                k = adjacent_edges[ind_k]
                c_ijk = (cosine_of_angle((j,i),(j,k),pos))
                phi_ijk = get_angle_from_cosine(c_ijk)
                loss += torch.exp(-phi_ijk)
                if phi_ijk < min_angle:
                    min_angle = phi_ijk
    return(loss,min_angle*max_degree / 90)




def vertex_resolution(node_dist,max_degree,n):
    r = torch.tensor(1/math.sqrt(n))
    loss = torch.tensor(0).float()
    q_vr = 1
    for i in range(n-1):
        for j in range(i+1,n):
            dist_ij = (node_dist[i][j] / (r*max_degree))
            loss += torch.max(torch.tensor(0).float(),(1-dist_ij)**2)
            if dist_ij < q_vr:
                q_vr = dist_ij
                #worst_nodes = (i,j)
    #maybe return worst_nodes
    return(loss / (n*(n-1)/2),q_vr)



#have to look at this again
def crossings_loss(pos_tensor,edge_pairs,n,boundary_lr=0.15):
    rand_bounds = 1/5 * np.random.random_sample((len(edge_pairs),3,1)) -0.1
    boundaries = torch.tensor(rand_bounds,dtype=torch.float, requires_grad=True)
    boundary_optimizer = torch.optim.SGD([boundaries], lr=boundary_lr)
    for b_iter in range(1):
        ones = torch.ones(n, 1)
        x1 = torch.cat((pos_tensor,ones), 1)
        poe_tens = torch.LongTensor(edge_pairs)
        e1 = x1[poe_tens[:,0,:]]
        e2 = x1[poe_tens[:,1,:]]
        pred1 = e1 @ boundaries
        pred2 = e2 @ boundaries
        loss1 = torch.sum(torch.nn.functional.relu(1-pred1*(1)))
        loss2 = torch.sum(torch.nn.functional.relu(1-pred2*(-1)))
        margin = torch.sum(torch.norm(boundaries,p=2,dim=0)[:1])
        loss = loss1+loss2+(margin*0.5)
        loss.backward()
        boundary_optimizer.step()
    ones = torch.ones(n, 1)
    x1 = torch.cat((pos_tensor,ones), 1)
    poe_tens = torch.LongTensor(edge_pairs)
    e1 = x1[poe_tens[:,0,:]]
    e2 = x1[poe_tens[:,1,:]]
    pred1 = e1 @ boundaries
    pred2 = e2 @ boundaries
    loss1 = torch.nn.functional.relu(1-pred1*(1)).sum()
    loss2 = torch.nn.functional.relu(1-pred2*(-1)).sum()
    loss = loss1+loss2
    return loss/(len(edge_pairs)*10)   #*(0.01) why is that times 0.01

    #q is simply number of crosssings.
