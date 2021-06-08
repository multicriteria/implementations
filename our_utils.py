import numpy as np
import math
from geometry import *
from graph_theory import *




def stress_of_single_node(node,node_distance,graph_distance,weights,L):
    return (np.sum(np.multiply(weights[node],(node_distance[node]-L*graph_distance[node])**2)))


def circular_ordering_edges(node,pos):
    neigh = node.neighbors
    if len(neigh) == 1:
        return [1]
    circ_list = np.asarray([[0]*2 for j in range(len(neigh))], dtype=object)
    for ind_i in range(len(neigh)):
        i = neigh[ind_i]
        xposi = pos[i][0] - pos[node.id][0]
        yposi = pos[i][1] - pos[node.id][1]
        angle = np.arctan2(yposi, xposi) * 180 / np.pi
        if (angle < 0):
            angle += 360
        circ_list[ind_i][0] = angle
        circ_list[ind_i][1] = (node.id,i)
    circ_list = circ_list[np.argsort(circ_list[:, 0])]
    return circ_list[:,1]


def angular_resolution_for_node(node,pos,circ_order):
    edge_pair = (0,0)
    min_angle = 360
    if len(circ_order) == 1:
        return (360, edge_pair)
    else:
        for ind_i in range(len(circ_order)):
            e1 = circ_order[ind_i]
            e2 =  circ_order[(ind_i+1) % (len(circ_order))]
            phi = angle_between_segments(e1,e2,pos)
            if phi < min_angle:
                min_angle = phi
                edge_pair = (e1,e2)
        return min_angle,edge_pair



def edge_vertex_data(m,n,pos,edges):
    e_v_matrix = np.zeros((n,m))-1
    for i in range(m):
        e = edges[i]
        for j in range(n):
            if j not in e:
                dist = distance_edge_to_node(e,j,pos)
                e_v_matrix[j][i] = dist
    return e_v_matrix


def e_v_update(e_v_matrix,Node,n,m,pos,edges):
    for i in range(len(Node.edges)):
        idx = Node.edges[i]
        e = edges[idx]
        for j in range(n):
            if j not in e:
                dist = distance_edge_to_node(e,j,pos)
                e_v_matrix[j][idx] = dist
    for i in range(m):
        e = edges[i]
        if Node.id not in e:
            dist = distance_edge_to_node(e,Node.id,pos)
            e_v_matrix[Node.id][i] = dist
    return e_v_matrix



def init_crossings(n,m,edges,pos,ca_flag,cn_flag):
    min_cross_angle_edge, cross_number_edges,min_ca,cr = None,None,None,None
    cross_matrix = np.zeros((m,m))
    #2 Crossing Angle / Crossing number
    if ca_flag:
        min_cross_angle_edge = np.zeros((m))+91
    if cn_flag:
        cross_number_edges = np.zeros((m))
    #Build rtree and rec_list
    rec_list = rectangle_list(edges,pos)
    idx_1 = init_rtree(edges,rec_list)
    for i in range(m):
        e1 = edges[i]
        p_i = possible_intersections(e1,pos,idx_1)
        for j in p_i:
            e2 = edges[j]
            if not(e1[0] in e2 or e1[1] in e2):
                p1 = pos[e1[0]]
                q1 = pos[e1[1]]
                p2 = pos[e2[0]]
                q2 = pos[e2[1]]
                if do_intersect(p1,q1,p2,q2):
                    ca = crossing_angle_between_segments(e1,e2,pos)
                    #cross matrix
                    cross_matrix[i][j] = ca
                    #min_cross_angle
                    if ca_flag:
                        min_cross_angle_edge[i] = min(min_cross_angle_edge[i],ca)
                    #cross_number_edges
                    if cn_flag:
                        cross_number_edges[i] +=1

    if ca_flag:
        min_ca = np.min(min_cross_angle_edge)
    if cn_flag:
        cr = np.sum(cross_number_edges)/2
    return (rec_list,idx_1,cross_matrix,min_cross_angle_edge,cross_number_edges,min_ca,cr)


def update_crossings(self,Node,constants,edges,pos,ca_flag,cn_flag):
    for adj in Node.edges:
        previous_crossed_edges = set(np.where(self.cross_matrix[adj] > 0)[0])
        new_crossed_edges = set()
        e1 = edges[adj]
        p_i = possible_intersections(e1,pos,self.rtree)
        for j in p_i:
            e2 = edges[j]
            if not(e1[0] in e2 or e1[1] in e2):
                p1 = pos[e1[0]]
                q1 = pos[e1[1]]
                p2 = pos[e2[0]]
                q2 = pos[e2[1]]
                if do_intersect(p1,q1,p2,q2):
                    new_crossed_edges.add(j)
                    ca = crossing_angle_between_segments(e1,e2,pos)
                    if self.cross_matrix[adj][j] > 0:
                        #they intersected before
                        # do not need to increase cross_number_edges or nodes
                        pass
                    else:
                        if cn_flag:
                            #they start crossing in this iteration
                            self.cross_number_edges[adj] +=1
                            self.cross_number_edges[j] +=1
                    #Now we can update the structures
                    #cross matrix
                    self.cross_matrix[adj][j] = self.cross_matrix[j][adj] = ca
        for j in previous_crossed_edges:
            if j not in new_crossed_edges:
                if cn_flag:
                    self.cross_number_edges[j] -=1
                    self.cross_number_edges[adj] -=1
                self.cross_matrix[adj][j] = self.cross_matrix[j][adj] = 0
    if ca_flag:
        for i in range(constants.m):
            entries = self.cross_matrix[i][self.cross_matrix[i] > 0]
            if len(entries) == 0:
                self.min_cross_angle_edge[i] = 91
            else:
                self.min_cross_angle_edge[i] = np.min(self.cross_matrix[i][self.cross_matrix[i] > 0])
        self.min_ca = np.min(self.min_cross_angle_edge)
    if cn_flag:
        self.cr = np.sum(self.cross_number_edges) / 2


def compute_criteria_new(const,stress,ar,vr,cross,list_of_nodes,init,criteria_weights):
    s_flag,ca_flag,el_flag,ar_flag,vr_flag,cn_flag = criteria_weights > 0
    c_ST,c_CA,c_EL,c_AR,c_EV,c_CN = 0,0,0,0,0,0
    #Compute Criteria initially and normalize
    #1.Stress
    if s_flag:
        stress_val_of_graph = np.sum(stress.vertex_stress)/2
        c_ST = stress_val_of_graph / ((const.n*(const.n-1)/2) * math.sqrt(2))

    #2.Crossing angle
    if ca_flag:
        c_CA = max(0, 1 - (cross.min_ca/90))


    #3.Ideal edge length
    if el_flag:
        c_EL = np.sum(np.abs(stress.edge_lengths - const.L))/(const.m *(math.sqrt(2) - const.L))

    #4.Angular resolution
    if ar_flag:
        min_entry = np.argmin(np.asarray([i for (i,j) in ar.ar_nodes]))
        u_degree = const.max_degree
        c_AR =  1 - (ar.ar_nodes[min_entry][0] / (360 / u_degree))

    #5. Edge-vertex resolution
    if vr_flag:
        short_dist,long_dist = np.min(vr.edge_vertex_data[np.where(vr.edge_vertex_data >= 0)]),np.max(vr.edge_vertex_data)
        c_EV =  1 - short_dist/long_dist

    #6. Crossing number
    if cn_flag:
        if cross.cr == 0:
            c_CN = 0
        else:
            c_CN = min(1,cross.cr / (init.cr+0.00001))


    #Compute initial objective function
    n_crit = np.array([c_ST,c_CA,c_EL,c_AR,c_EV,c_CN])
    return n_crit


def choose_vertex_from_pool(pos,edges,consts,stress,ar,vr,cross,partial_energy,energy,params,criteria_weights):
    s_flag,ca_flag,el_flag,ar_flag,vr_flag,cn_flag = criteria_weights > 0
    s_val,ca_val,el_val,ar_val,vr_val,cn_val = partial_energy > 0
    v_pool = []
    stress_subpool,ca_subpool,el_subpool,ar_subpool,ev_subpool,cr_subpool = [],[],[],[],[],[]
    #Stress
    if s_flag and s_val:
        ind = np.argpartition(stress.vertex_stress, -params.s_k)[-params.s_k:]
        stress_subpool = [ind][0]
    #Ideal edge length -> just add the longest edge
    if el_flag and el_val:
        longest_edge = np.unravel_index(stress.node_dist.argmax(), stress.node_dist.shape)
        shortest_edge = np.unravel_index(stress.node_dist.argmin(), stress.node_dist.shape)
        el_subpool = [longest_edge[0],longest_edge[1],shortest_edge[0],shortest_edge[1]]
    #Angular resolution
    if ar_flag and ar_val:
        min_entry = np.argmin(np.asarray([i for (i,j) in ar.ar_nodes]))
        _,(e1,e2) = ar.ar_nodes[min_entry]
        ar_subpool = [min_entry,e1[0],e1[1],e2[0],e2[1]]

    #EV-Res
    if vr_flag and vr_val:
        valid_idx= np.where(vr.edge_vertex_data >= 0)[0]
        flat_idx = vr.edge_vertex_data[valid_idx].argmin()
        v,e = np.unravel_index(flat_idx,vr.edge_vertex_data.shape)
        ev_subpool = [v,edges[e][0],edges[e][1]]


    if ca_flag and ca_val:
        crit_edges_idx = np.where(cross.min_cross_angle_edge == np.min(cross.min_cross_angle_edge))
        e1 = edges[crit_edges_idx[0][0]]
        e2 = edges[crit_edges_idx[0][1]]
        ca_subpool = [e1[0],e1[1],e2[0],e2[1]]
    if cn_flag and cn_val:
        #Crossing number
        ind = np.argpartition(cross.cross_number_edges,-params.num_most_crossed)[-params.num_most_crossed:]
        resp_crossings = cross.cross_number_edges[ind]
        cr_subpool = []
        for i in range(params.num_most_crossed):
            edge = edges[ind[i]]
            cr_subpool += [edge[0]]*int(resp_crossings[i])
            cr_subpool += [edge[1]]*int(resp_crossings[i])


    v_pool = [stress_subpool,ca_subpool,el_subpool,ar_subpool,ev_subpool,cr_subpool]
    normed_partial_energy = partial_energy / energy
    probs = np.zeros((6))
    for i in range(6):
        probs[i] = np.sum(normed_partial_energy[:i+1])
    #sample random number and check which entry of probs is larger
    sample = np.random.random_sample()
    flag = True
    i = 0
    while flag:
        if probs[i] >= sample:
            flag = False
            subpool_idx = i
        i+=1
    chosen_pool = v_pool[subpool_idx]
    #choose uniformly among these vertices
    return(chosen_pool[np.random.randint(0,len(chosen_pool))])
