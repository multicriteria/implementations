import numpy as np
import math
import networkx as nx
import random
import copy
import matplotlib.pyplot as plt
from geometry import *
from graph_theory import *
from our_utils import *
import time


# Datastructures
# Have fixed order of nodes and edges -> use indices = [0,..,n-1] and [0,m-1] to access them throughout the algorithm
class Node:
    # Edges and neighbors are not Objects, but simply lists of indices
    def __init__(self,id,edges,neighbors):
        self.id = id
        self.edges = edges
        self.neighbors = neighbors

#
class Constants:
    #save all scalar constants in this object
    def __init__(self,G,height,width,static):
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        self.height = height
        self.width = width
        self.max_degree = np.max(np.sum(static.A,axis=1))
        self.L = min(self.width,self.height) / min(np.max(static.graph_dist),100)
        self.d =min(self.width,self.height)/4

class Static_Data:
    #static data structures
    def __init__(self,G):
        self.graph_dist = graph_distance(G,G.number_of_nodes())
        self.A = nx.to_numpy_matrix(G)

#below are dynamic data
#stress + el
class Stress_and_EL_Data:
    def __init__(self,pos,static,constants,weights_stress,nodes,edges,s_flag,el_flag):
        self.node_dist = node_distance_total(pos)
        if el_flag:
            self.edge_lengths = [self.node_dist[e[0]][e[1]] for e in edges]
        if s_flag:
            self.vertex_stress = [stress_of_single_node(u,self.node_dist,static.graph_dist,weights_stress,constants.L) for u in nodes]

    def update(self,Node,pos,static,constants,weights_stress,edges,s_flag,el_flag):
        self.node_dist[Node.id] = self.node_dist[:,Node.id] = node_distance_node(Node.id,pos)
        if s_flag:
            self.vertex_stress[Node.id] = stress_of_single_node(Node.id,self.node_dist,static.graph_dist,weights_stress,constants.L)
        if el_flag:
            for adj in Node.edges:
                u,v = edges[adj]
                self.edge_lengths[adj] = self.node_dist[u][v]

    def overwrite(self,other,Node,s_flag,el_flag):
        self.node_dist[Node.id] = self.node_dist[:,Node.id] = other.node_dist[Node.id]
        if s_flag:
            self.vertex_stress[Node.id] = other.vertex_stress[Node.id]
        if el_flag:
            for adj in Node.edges:
                self.edge_lengths[adj] = other.edge_lengths[adj]





#Edge vertex resolution
class VR_Data:
    def __init__(self,pos,static,constants,edges):
        self.edge_vertex_data = edge_vertex_data(constants.m,constants.n,pos,edges)

    def update(self,Node,pos,constants,edges):
        self.edge_vertex_data = e_v_update(self.edge_vertex_data,Node,constants.n,constants.m,pos,edges)

    def overwrite(self,other,Node):
        self.edge_vertex_data[Node.id] = other.edge_vertex_data[Node.id]
        for adj in Node.edges:
            self.edge_vertex_data[:,adj] = other.edge_vertex_data[:,adj]


#ang res
class AR_Data:
    def __init__(self,pos,static,constants,list_of_nodes,edges,nodes):
        self.circ_order = [circular_ordering_edges(node,pos) for node in list_of_nodes]
        self.ar_nodes = [angular_resolution_for_node(node,pos,self.circ_order[node]) for node in nodes]

    def update(self,Node,pos,list_of_nodes):
        self.circ_order[Node.id] = circular_ordering_edges(Node,pos)
        self.ar_nodes[Node.id] = angular_resolution_for_node(Node.id,pos,self.circ_order[Node.id])
        for neighs in Node.neighbors:
            self.circ_order[neighs] = circular_ordering_edges(list_of_nodes[neighs],pos) #could improve this
            self.ar_nodes[neighs] = angular_resolution_for_node(neighs,pos,self.circ_order[neighs])

    def overwrite(self,other,Node):
        self.circ_order[Node.id] = other.circ_order[Node.id]
        self.ar_nodes[Node.id] = other.ar_nodes[Node.id]
        for neigh in Node.neighbors:
            self.circ_order[neigh]= other.circ_order[neigh]  #could improve this
            self.ar_nodes[neigh] = other.ar_nodes[neigh]



#crossing data
#cross num and cross angle

class Cross_Data:
    def __init__(self,pos,constants,edges,ca_flag,cn_flag):
        self.rec_list,self.rtree,\
        self.cross_matrix,self.min_cross_angle_edge,self.cross_number_edges,\
        self.min_ca,self.cr = init_crossings(constants.n,constants.m,edges,pos,ca_flag,cn_flag)
        #if cn_flag:
    #        self.edges_to_update = []

    def update(self,Node,constants,static,edges,pos,ca_flag,cn_flag):
        for i in range(len(Node.edges)):
            self.rtree,self.rec_list = update_edge_rtree(self.rtree,edges[Node.edges[i]],Node.edges[i],pos,self.rec_list)
        update_crossings(self,Node,constants,edges,pos,ca_flag,cn_flag)

    def overwrite(self,other,Node,constants,edges,pos,ca_flag,cn_flag):
        if ca_flag:
            self.min_cross_angle_edge = copy.deepcopy(other.min_cross_angle_edge)
            self.min_ca = other.min_ca
        for adj in Node.edges:
            self.cross_matrix[adj] = self.cross_matrix[:,adj] = other.cross_matrix[adj]
            #hard to copy this from other
            self.rtree,self.rec_list = update_edge_rtree(self.rtree,edges[adj],adj,pos,self.rec_list)
            if cn_flag:
                self.cross_number_edges[adj] = other.cross_number_edges[adj]
        if cn_flag:
            for i in range(constants.m):
                self.cross_number_edges[i] = other.cross_number_edges[i]
            self.cr = other.cr



class Values:
    def __init__(self,cr):
        self.cr = cr


class Params:
    def __init__(self,num_most_crossed,num_of_rays,s_k):
        self.num_most_crossed = num_most_crossed
        self.num_of_rays = num_of_rays
        self.s_k = s_k




def Our_alg(G,pos,timelimit,crit_weights,height,width,params = Params(4,4,4)):
    #Start time
    start = time.time()
    elapsed = 0
    #Immutable data
    edges = tuple(G.edges())
    nodes = tuple(G.nodes())
    #create list that stores nodes Objects
    adjacent_edges = [[] for _ in range(len(nodes))]
    for i in range(len(edges)):
        (u,v) = edges[i]
        adjacent_edges[u].append(i)
        adjacent_edges[v].append(i)
    list_of_nodes = []
    for i in range(len(nodes)):
        id = i
        neighs = list(G.adj[i])
        u = Node(i,adjacent_edges[i],neighs)
        list_of_nodes.append(u)


    # Init data structures
    static_data = Static_Data(G)
    consts = Constants(G,height,width,static_data)
    criteria_weights = np.asarray(list(crit_weights.values()))
    weights_stress = np.ones((consts.n, consts.n))


    #Chosen Parameters
    s_flag,ca_flag,el_flag,ar_flag,vr_flag,cn_flag = criteria_weights > 0
    stress_el,stress_el_tmp = None,None
    vr,vr_tmp = None,None
    ar,ar_tmp = None,None
    cross,cross_tmp,init_values = None,None,None

        #Datastructures
    # Init normal and tmp version.
    if (s_flag or el_flag):
        stress_el = Stress_and_EL_Data(pos,static_data,consts,weights_stress,nodes,edges,s_flag,el_flag)
        stress_el_tmp = Stress_and_EL_Data(pos,static_data,consts,weights_stress,nodes,edges,s_flag,el_flag)
    if vr_flag:
        vr = VR_Data(pos,static_data,consts,edges)
        vr_tmp = VR_Data(pos,static_data,consts,edges)
    if ar_flag:
        ar = AR_Data(pos,static_data,consts,list_of_nodes,edges,nodes)
        ar_tmp = AR_Data(pos,static_data,consts,list_of_nodes,edges,nodes)
    if (ca_flag or cn_flag):
        cross = Cross_Data(pos,consts,edges,ca_flag,cn_flag)
        cross_tmp = Cross_Data(pos,consts,edges,ca_flag,cn_flag)
        if cn_flag:
            init_values = Values(np.sum(cross.cross_number_edges)/2)




    n_crit = compute_criteria_new(consts,stress_el,ar,vr,cross,list_of_nodes,init_values,criteria_weights)
    partial_obj = criteria_weights * n_crit

    obj_func = np.sum(partial_obj)
    initial = obj_func
    unchanged_iterations = 0

    #keep track of best position and corresponding value
    best_pos = copy.copy(pos)
    best_value = obj_func
    iterations = 0
    while elapsed < timelimit :
        #print(elapsed)
        if obj_func == 0:
            #print('Global optimum')
            return pos
        #Choose vertex
        i = choose_vertex_from_pool(pos,edges,consts,stress_el,ar,vr,cross,partial_obj,obj_func,params,criteria_weights)
        current_node = list_of_nodes[i]
        old_pos = copy.copy(pos[i])

        potential_positions = calc_rays(i, pos,consts.width,consts.height,consts.d,params.num_of_rays)
        random.shuffle(potential_positions)
        index = 0
        not_increased = True
        while not_increased and index < len(potential_positions):

            pos[i] = potential_positions[index]
            if (s_flag or el_flag):
                stress_el_tmp.update(current_node,pos,static_data,consts,weights_stress,edges,s_flag,el_flag)
            if vr_flag:
                vr_tmp.update(current_node,pos,consts,edges)
            if ar_flag:
                ar_tmp.update(current_node,pos,list_of_nodes)
            if (ca_flag or cn_flag):
                cross_tmp.update(current_node,consts,static_data,edges,pos,ca_flag,cn_flag)


            n_crit_new = compute_criteria_new(consts,stress_el_tmp,ar_tmp,vr_tmp,cross_tmp,list_of_nodes,init_values,criteria_weights)
            partial_obj_new = criteria_weights * n_crit_new
            obj_func_new = np.sum(partial_obj_new)

            if obj_func_new < obj_func:

                #We take the position

                if (s_flag or el_flag):
                    stress_el.overwrite(stress_el_tmp,current_node,s_flag,el_flag)
                if vr_flag:
                    vr.overwrite(vr_tmp,current_node)
                if ar_flag:
                    ar.overwrite(ar_tmp,current_node)
                if (ca_flag or cn_flag):
                    cross.overwrite(cross_tmp,current_node,consts,edges,pos,ca_flag,cn_flag)


                not_increased = False
                unchanged_iterations = 0
                partial_obj = partial_obj_new
                obj_func = obj_func_new

                if obj_func_new <= best_value:
                    #if obj_func_new <= 0.98*best_value:
                        #print(obj_func_new)
                    best_value = obj_func_new
                    best_pos = copy.copy(pos)

            index+=1
        if not_increased:
            prob = np.exp(-1*(5 /(unchanged_iterations+1)))
            if  prob >= np.random.random():
                #also update
                unchanged_iterations = 0
                if (s_flag or el_flag):
                    stress_el.overwrite(stress_el_tmp,current_node,s_flag,el_flag)
                if vr_flag:
                    vr.overwrite(vr_tmp,current_node)
                if ar_flag:
                    ar.overwrite(ar_tmp,current_node)
                if (ca_flag or cn_flag):
                    cross.overwrite(cross_tmp,current_node,consts,edges,pos,ca_flag,cn_flag)
                partial_obj = partial_obj_new
                obj_func = np.sum(partial_obj)
            else:
                pos[i]= old_pos
                if (s_flag or el_flag):
                    stress_el_tmp.overwrite(stress_el,current_node,s_flag,el_flag)
                if vr_flag:
                    vr_tmp.overwrite(vr,current_node)
                if ar_flag:
                    ar_tmp.overwrite(ar,current_node)
                if (ca_flag or cn_flag):
                    cross_tmp.overwrite(cross,current_node,consts,edges,pos,ca_flag,cn_flag)
        iterations+=1
        unchanged_iterations+=1

        elapsed = time.time() - start
    return best_pos
