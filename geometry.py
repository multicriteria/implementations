import numpy as np
import networkx as nx
from rtree import index

#Colleciton of geometrical functions between segments (edges) and points (nodes)


#1- Crossings of segmnets

#Return whether two straight line segments cross

def on_segment(p, q, r):
    #Given three colinear points p, q, r, the function checks if
    #point q lies on line segment

    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False



def orientation(p, q, r):
    '''Find orientation of ordered triplet (p, q, r).\n",
    The function returns following values\n",
    0 --> p, q and r are colinear\n",
    1 --> Clockwise\n",
    2 --> Counterclockwise\n",
    '''

    val = ((q[1] - p[1]) * (r[0] - q[0]) -
            (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # colinear\n",
    elif val > 0:
        return 1   # clockwise\n",
    else:
        return 2  # counter-clockwise\n",



def do_intersect(p1,q1,p2,q2):
    '''Main function to check whether the closed line segments p1 - q1 and p2 \n",
       - q2 intersect'''
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    # General case\n",
    if (o1 != o2 and o3 != o4):
        return True

    # Special Cases\n",
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1\n",
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1\n",
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2\n",
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2\n",
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False # Doesn't fall in any of the above cases\n",



# we need to maintain fixed order of edges

def rectangle(edge,pos):
    u,v = edge[0],edge[1]
    u_x,u_y = pos[u][0],pos[u][1]
    v_x,v_y = pos[v][0],pos[v][1]
    left,bottom,right,top = min(u_x,v_x),min(u_y,v_y),max(u_x,v_x),max(u_y,v_y)
    return (left,bottom,right,top)


def rectangle_list(edges,pos):
    rectangle_list = []
    for e in edges:
        rectangle_list.append(rectangle(e,pos))
    return rectangle_list

#Compute intersections with Rtree
def init_rtree(edges,rec_list):
    idx = index.Index()
    for i in range(len(edges)):
        left,bottom,right, top = rec_list[i]
        idx.insert(i,(left,bottom,right,top))
    return idx

#also updates rectangle_list
def update_edge_rtree(rtree_index,edge,edge_index,pos,rectangle_list):
    new_rectangle = rectangle(edge,pos)
    #delete old rectangle
    rtree_index.delete(edge_index,(rectangle_list[edge_index]))
    #add new rectangle
    rtree_index.insert(edge_index,new_rectangle)
    #update rectangle_list
    rectangle_list[edge_index] = new_rectangle
    return rtree_index,rectangle_list



def possible_intersections(edge, pos, rtree_index):
    left,bottom,right, top = rectangle(edge,pos)
    return(list(rtree_index.intersection((left,bottom,right,top))))







#Function


#2. Distances in Euclidean Space

def node_distance_total(pos):
    return(np.linalg.norm(pos[:, None]-pos, axis=2))
def node_distance_node(node,pos):
    return(np.linalg.norm(pos[node, None]-pos, axis=1))

def adjacent_edges_length(node,A,node_dist):
    edge_lengths = np.multiply(A[node],node_dist[node])
    return (np.sum(edge_lengths))

#Edge-vertex- resolution
#http://paulbourke.net/geometry/pointlineplane/
#https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment

def distance_edge_to_node(e,p,pos):##x1, y1, x2, y2, x3, y3): # x3,y3 is the point
    x1, y1 = pos[e[0]]
    x2, y2 = pos[e[1]]
    x3, y3 = pos[p]

    px = x2-x1
    py = y2-y1
    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3
    dist = (dx*dx + dy*dy)**.5
    return dist


#3. Computing angles between segments

#Angle stuff

#Angular resolution
#old
def get_angle_from_cosine(cos_val):
    c_angle = np.arccos(cos_val)
    ang_deg = np.degrees(c_angle)%360
    return (min(ang_deg,360-ang_deg))


def cosine_of_angle(e1,e2,pos):
    idx_1 = np.array([e1[0],e1[1]])
    idx_2 = np.array([e2[0],e2[1]])
    s1=pos[idx_1[1]] - pos[idx_1[0]]
    s2=pos[idx_2[1]] - pos[idx_2[0]]
    cos = np.dot(s1, s2)/(np.linalg.norm(s1)*np.linalg.norm(s2))
    return(cos)



def angle_between_segments(e1,e2,pos):
    #maybe save edges as np.array from the start
    idx_1 = np.array([e1[0],e1[1]])
    idx_2 = np.array([e2[0],e2[1]])
    s1=pos[idx_1[1]] - pos[idx_1[0]]
    s2=pos[idx_2[1]] - pos[idx_2[0]]
    # create

    acos_in_rads = np.dot(s1, s2)/(np.linalg.norm(s1)*np.linalg.norm(s2))
    if acos_in_rads > 1:
        cos = np.degrees(np.arccos(1))
    elif acos_in_rads < -1:
        cos = np.degrees(np.arccos(-1))
    else:
        cos = np.degrees(np.arccos(acos_in_rads))

    return(cos)

def crossing_angle_between_segments(e1,e2,pos):
    cos = angle_between_segments(e1,e2,pos)
    if cos > 90:
        return (180-cos)
    else:
        return(cos)



def calc_rays(i, pos,width,height,d,raycount=8):
    pos_list = []
    angle = 360 / raycount
    rand_rotate = np.random.random() * angle
    scalar = np.random.uniform(0,d,raycount)

    for k in range(raycount):
        new_x = np.cos(((2 * np.pi * angle) / 360) * k + rand_rotate)*scalar[k] + pos[i][0]
        new_y = np.sin(((2 * np.pi * angle) / 360) * k + rand_rotate)*scalar[k] + pos[i][1]
        if new_x >= 0 and new_x <= width  and new_y >= 0 and new_y <= height:
            pos_list.append((new_x,new_y))
    if len(pos_list) > 0:
        return pos_list
    return [pos[i]]
