from geometry import *
from graph_theory import *
def compute_crossings(edges,pos,m):
    res = []
    for i in range(m-1):
        for j in range(i+1,m):
            if not(edges[i][0] in edges[j] or edges[i][1] in edges[j]):
                p1 = pos[edges[i][0]]
                q1 = pos[edges[i][1]]
                p2 = pos[edges[j][0]]
                q2 = pos[edges[j][1]]
                if do_intersect(p1,q1,p2,q2):
                    res.append((edges[i],edges[j]))
    return(res)

def circular_ordering_edges(A,n,pos):
    neigh = neighbors(A, n)
    if len(neigh) == 1:
        return [1]
    circ_list = np.asarray([[0]*2 for j in range(len(neigh))], dtype=object)
    for ind_i in range(len(neigh)):
        i = neigh[ind_i]
        xposi = pos[i][0] - pos[n][0]
        yposi = pos[i][1] - pos[n][1]
        angle = np.arctan2(yposi, xposi) * 180 / np.pi
        if (angle < 0):
            angle += 360
        circ_list[ind_i][0] = angle
        circ_list[ind_i][1] = (n,i)
    circ_list = circ_list[np.argsort(circ_list[:, 0])]
    return circ_list[:,1]


def edge_vertex_data(m,n,pos,edges):
    e_v_matrix = np.zeros((n,m))-1
    for i in range(m):
        e = edges[i]
        for j in range(n):
            if j not in e:
                dist = distance_edge_to_node(e,j,pos)
                e_v_matrix[j][i] = dist
    return e_v_matrix



def init_crossings(edges,pos,m):
    rec_list = rectangle_list(edges,pos)
    idx = init_rtree(edges,rec_list)
    cn = 0
    for i in range(m):
        e1 = edges[i]
        p_i = possible_intersections(e1,pos,idx)
        for j in p_i:
            e2 = edges[j]
            if not(e1[0] in e2 or e1[1] in e2):
                p1 = pos[e1[0]]
                q1 = pos[e1[1]]
                p2 = pos[e2[0]]
                q2 = pos[e2[1]]
                if do_intersect(p1,q1,p2,q2):
                    cn+=1
    return rec_list,idx,cn/2

def crossings_of_node(rtree,adj_edges,pos,edges):
    cn = 0
    for adj in adj_edges:
        e1= edges[adj]
        p_i = possible_intersections(e1,pos,rtree)
        for j in p_i:
            e2 = edges[j]
            if not(e1[0] in e2 or e1[1] in e2):
                p1 = pos[e1[0]]
                q1 = pos[e1[1]]
                p2 = pos[e2[0]]
                q2 = pos[e2[1]]
                if do_intersect(p1,q1,p2,q2):
                    cn+=1
    return cn


#lower bound computed matrix to avoid division by zero
def lb(input):
    return(np.where(input <0.01,0.01,input))
