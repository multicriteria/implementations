import numpy as np
import math
import networkx as nx
import random
import copy
import time
from utils import *


def graph_energy(lambdas,d_ij,boundary,d_k,cr,g_kl,ft):
    dij_sub = np.sum(lambdas[0] / np.diagonal(d_ij)**2) #remove dist_ii
    a_ij = (np.sum(lambdas[0] / d_ij**2) - dij_sub) / (d_ij.shape[0]**2) #normalize by n^2 -> avg
    m_i = np.sum(lambdas[1]*(1/boundary**2)) / (4*d_ij.shape[0]) #normalize by 4n -> avg
    c_k = np.sum(lambdas[2] * d_k)
    cr_k = np.sum(lambdas[3] * cr)
    if not ft: #inital step
        return(a_ij+m_i+c_k+cr_k)
    else:
        h_kl_sub = 2*g_kl.shape[1] * (lambdas[4] / 0.01**2) #remove dist_ik with k = (i,_) -> 2m entries
        h_kl = (np.sum(lambdas[4] / g_kl**2) - h_kl_sub) / (g_kl.shape[0] * g_kl.shape[1]*50) #normalize by n*m*50 -> avg/50
        return(a_ij+m_i+c_k+cr_k+h_kl)

def vertex_energy(lambdas,d_i,boundary_i,d_adj,cr_i,g_kl,ft):
    dij_sub = lambdas[0] / (0.01**2)
    a_ij = (np.sum(lambdas[0] / d_i**2) - dij_sub) / (d_i.shape[0]**2) #normalize by n^2
    m_i = np.sum(lambdas[1]*(1/boundary_i**2)) / (4*d_i.shape[0])
    c_k = np.sum(lambdas[2] * d_adj)
    cr_k = np.sum(lambdas[3] * cr_i)
    if not ft:
        return(a_ij+m_i+c_k+cr_k)
    else:
        h_kl_sub = 2*g_kl.shape[1] * (lambdas[4] / 0.01**2) #remove dist_ik with k = (i,_) -> 2m entries
        h_kl = (np.sum(lambdas[4] / g_kl**2) - h_kl_sub) / (g_kl.shape[0] * g_kl.shape[1]*50) #normalize by n*m*50 -> avg/50
        return(a_ij+m_i+c_k+cr_k+h_kl)





def DH(G, pos,timelimit,lambdas, height, width):
    #SA method
    # Initialize
    start = time.time()
    elapsed = 0
    # 70% of time for initial loop, 30% for finetuning
    time_initial = 0.7*timelimit
    time_until_cooling = time_initial / 10
    time_fine = timelimit - time_initial
    left_border= bottom_border = 0.1
    best_pos = copy.deepcopy(pos)
    A = nx.to_numpy_matrix(G)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    edges = list(G.edges())
    nodes = list(G.nodes())
    adjacent_edges = [[] for _ in range(len(nodes))]
    for i in range(len(edges)):
        (u,v) = edges[i]
        adjacent_edges[u].append(i)
        adjacent_edges[v].append(i)
    #n x n matrix that saves node distances
    d_ij  = lb(node_distance_total(pos))
    # 1 x n vectors that contain distances of nodes to
    # left (0), right (1),bottom(0),top(1) border
    l_i =  pos[:,0] +left_border
    r_i  = width - pos[:,0]
    b_i =  pos[:,1] + bottom_border
    t_i  = height - pos[:,1]
    boundary = lb(np.column_stack((l_i,r_i,b_i,t_i)))

    #1xm vector that contains the length of all edges
    d_k =  np.asarray([d_ij[u][v] for (u,v) in edges])
    #1xn vector that contains the number of crossings each vertex is involved with
    rec_list,rtree,cn = init_crossings(edges,pos,m)
    #nxm matrix that contains minimum distance between any node and any edge
    g_kl = lb(edge_vertex_data(m,n,pos,edges))
    #to compare against

    old_energy = graph_energy(lambdas,d_ij,boundary,d_k,cn,0,ft=False)
    #1 Initial configuration (pos) and initial temperature
    temperature = 1

    unchanged_iterations=0
    while elapsed < time_initial:
        elapsed_iter = 0
        iter_start = time.time()
        while elapsed_iter < time_until_cooling:
            for idx in range(n):
                changepos = False
                old_pos = copy.deepcopy(pos[idx])
                #Compute previous crossings that the vertex is involved with
                previous_crossings = crossings_of_node(rtree,adjacent_edges[idx],pos,edges)
                part_energy_before = vertex_energy(lambdas,d_ij[idx],boundary[idx],d_k[adjacent_edges[idx]],previous_crossings,0,ft=False)
                #selected index is idx
                #2(a) choose new configuration from neighborhood
                #i.e. sample new point for node idx on !perimeter! of circle
                #OGDF: chooses the initial radius of the disk as half the maximum of width and height of the initial layout
                #Have to decide which to use
                f_1 = np.sqrt(n /(height/width)) / 5.0
                f_2 =  max(width, height)  / 5.0
                circle_radius = f_2 * temperature
                while True:
                    alpha = 2 * math.pi * random.random()
                    px = circle_radius * math.cos(alpha) + old_pos[0]
                    py = circle_radius * math.sin(alpha) + old_pos[1]
                    #sample them inside the box
                    if 0 <= px <= width and 0 <= py <= height:
                        break
                #update position
                pos[idx] = (px, py)
                d_ij[idx] = d_ij[:,idx] = lb(node_distance_node(idx,pos))
                boundary[idx] = lb(np.array([pos[idx][0]+left_border,width - pos[idx][0],pos[idx][1]+bottom_border,height - pos[idx][1]]))
                for adj in adjacent_edges[idx]:
                    u,v = edges[adj][0],edges[adj][1]
                    d_k[adj] =  d_ij[u][v]
                    rtree,rec_list = update_edge_rtree(rtree,edges[adj],adj,pos,rec_list)
                #now that rtree is updated we can compute new crossings
                new_crossings = crossings_of_node(rtree,adjacent_edges[idx],pos,edges)
                part_energy_after= vertex_energy(lambdas,d_ij[idx],boundary[idx],d_k[adjacent_edges[idx]],new_crossings,0,ft=False)
                new_energy = old_energy - part_energy_before + part_energy_after
                if part_energy_after <= part_energy_before:
                    changepos = True
                    best_pos[idx] = copy.copy(pos[idx])
                else:

                    probability = np.exp((old_energy-new_energy)/temperature)
                    if (np.random.random() < probability):
                        changepos = True

                if changepos == True:
                    unchanged_iterations = 0
                    old_energy = new_energy
                    cn = cn - previous_crossings + new_crossings
                else:
                    pos[idx] = old_pos
                    #revert changes
                    d_ij[idx] = d_ij[:,idx] = lb(node_distance_node(idx,pos))
                    boundary[idx] = lb(np.array([pos[idx][0]+left_border,width - pos[idx][0],pos[idx][1]+bottom_border,height - pos[idx][1]]))
                    for adj in adjacent_edges[idx]:
                        u,v = edges[adj][0],edges[adj][1]
                        d_k[adj] =  d_ij[u][v]
                        rtree,rec_list = update_edge_rtree(rtree,edges[adj],adj,pos,rec_list)
                if old_energy == 0:
                    print('Optimal solution found')
                    return pos
            elapsed_iter = time.time() - iter_start
        elapsed = time.time() - start
        unchanged_iterations+= 1
        if unchanged_iterations == 3:
            print('Early break due to no change')
            return pos
        temperature *= 0.75

    #nx.draw(G,pos,with_labels=True)
    #plt.show()
    #Finteuning
    #print('Finetuning')
    g_kl = lb(edge_vertex_data(m,n,pos,edges))
    #compute new energy including edge vertex distance

    old_energy = graph_energy(lambdas,d_ij,boundary,d_k,cn,g_kl,ft=True)
    while elapsed < timelimit:
        for idx in range(n):
            old_pos = copy.copy(pos[idx])
            previous_crossings = crossings_of_node(rtree,adjacent_edges[idx],pos,edges)
            part_energy_before = vertex_energy(lambdas,d_ij[idx],boundary[idx],d_k[adjacent_edges[idx]],previous_crossings,g_kl,ft=True)
            #previous Distances
            g_kl_idx = copy.deepcopy(g_kl[idx])
            g_kl_adj = copy.deepcopy(g_kl[:,adjacent_edges[idx]])
            changepos = False
            #very small, they say pixels!
            circle_radius = 0.001 * max(width, height)
            while True:
                alpha = 2 * math.pi * random.random()
                r = circle_radius * random.random()
                px = r * math.cos(alpha) + old_pos[0]
                py = r * math.sin(alpha) + old_pos[1]
                #sample them inside the box
                if 0 <= px <= width and 0 <= py <= height:
                    break
            pos[idx] = (px, py)
            d_ij[idx] = d_ij[:,idx] = lb(node_distance_node(idx,pos))
            boundary[idx] = lb(np.array([pos[idx][0]+left_border,width - pos[idx][0],pos[idx][1]+bottom_border,height - pos[idx][1]]))

            for adj in adjacent_edges[idx]:
                u,v = edges[adj][0],edges[adj][1]
                d_k[adj] =  d_ij[u][v]
                rtree,rec_list = update_edge_rtree(rtree,edges[adj],adj,pos,rec_list)
                for i in range(n):
                    g_kl[i][adj] =  lb(distance_edge_to_node(edges[adj],i,pos))
            for e in range(m):
                g_kl[idx][e] = lb(distance_edge_to_node(edges[e],idx,pos))
            new_crossings = crossings_of_node(rtree,adjacent_edges[idx],pos,edges)
            part_energy_after = vertex_energy(lambdas,d_ij[idx],boundary[idx],d_k[adjacent_edges[idx]],new_crossings,g_kl,ft=True)
            #only good moves
            if part_energy_after < part_energy_before:
                unchanged_iterations = 0
                old_energy = old_energy - part_energy_before + part_energy_after
                cn = cn - previous_crossings + new_crossings
                best_pos[idx] = copy.copy(pos[idx])
            else:
                pos[idx] = old_pos
                #revert changes
                d_ij[idx] = d_ij[:,idx] = lb(node_distance_node(idx,pos))
                boundary[idx] = lb(np.array([pos[idx][0]+left_border,width - pos[idx][0],pos[idx][1]+bottom_border,height - pos[idx][1]]))
                for adj in adjacent_edges[idx]:
                    u,v = edges[adj][0],edges[adj][1]
                    d_k[adj] =  d_ij[u][v]
                    rtree,rec_list = update_edge_rtree(rtree,edges[adj],adj,pos,rec_list)
                g_kl[idx] = g_kl_idx
                g_kl[:,adjacent_edges[idx]] = g_kl_adj
        elapsed = time.time() - start
    return best_pos
