
from gd_utils import *


def GD_squared(G,pos,timelimit,crit_weights):
    start = time.time()
    elapsed = 0

    pos_tensor = torch.tensor(pos).float().requires_grad_()
    adj_matrix = torch.from_numpy(nx.adjacency_matrix(G).todense())
    max_degree = max([val for (node, val) in G.degree()])
    edges = list(G.edges())
    m = G.number_of_edges()
    n = G.number_of_nodes()

    # Compute static structures
    graph_dist = graph_distance(G,G.number_of_nodes())
    graph_dist_tens = torch.from_numpy(graph_dist).float()
    possible_crossing_edges = edge_pairs_for_crossings(edges)



    weight_for_stress = torch.ones((n,n)).float()

    criteria_weights = {}
    criteria_weights['stress'] = torch.tensor(crit_weights['stress'])
    criteria_weights['crossangle'] = torch.tensor(crit_weights['crossangle'])
    criteria_weights['idealedge'] = torch.tensor(crit_weights['idealedge'])
    criteria_weights['angularres'] =torch.tensor(crit_weights['angularres'])
    criteria_weights['vertexres'] =  torch.tensor(crit_weights['vertexres'])
    criteria_weights['crossnum'] = torch.tensor(crit_weights['crossnum'])


    opt = torch.optim.SGD([pos_tensor], lr=0.001)

    while elapsed < timelimit:
        if criteria_weights['crossangle'] > 0 or criteria_weights['crossnum'] > 0 :
            crossings,crossingnumber = compute_crossings(edges,pos_tensor.tolist(),m)
        node_dist = node_distance(pos_tensor)

        loss = torch.tensor(0.0)
        if criteria_weights['stress'] > 0:
            l = stress_loss(node_dist,graph_dist_tens,weight_for_stress)
            loss += torch.mul(criteria_weights['stress'],l)

        if criteria_weights['crossangle'] > 0 and crossingnumber > 0:
            l,metric_ca= crossing_angle_loss(crossings,pos_tensor)
            loss += torch.mul(criteria_weights['crossangle'],l)

        if criteria_weights['idealedge'] > 0:
            l = ideal_edge_length_loss(adj_matrix,node_dist,m)
            loss += torch.mul(criteria_weights['idealedge'],l)

        if criteria_weights['angularres'] > 0:
            l,metric_ar = angular_resolution(adj_matrix,n,pos_tensor,max_degree,s=1)
            loss += torch.mul(criteria_weights['angularres'],l)

        if criteria_weights['vertexres'] > 0:
            l,metric_vr = vertex_resolution(node_dist,torch.tensor(max_degree).float(),n)
            loss += torch.mul(criteria_weights['vertexres'],l)

        if criteria_weights['crossnum'] > 0 and crossingnumber > 0:
            l = crossings_loss(pos_tensor,possible_crossing_edges,n,boundary_lr=0.15)
            loss += torch.mul(criteria_weights['crossnum'],l)

        if loss == 0:
            small_val = torch.min(pos_tensor,dim=0)
            shifted_pos = torch.add(pos_tensor,torch.abs(small_val.values))
            scale = torch.max(torch.max(shifted_pos,dim=0)[0]).item()
            new_pos = shifted_pos / scale
            new_pos = new_pos.detach().numpy()
            return new_pos

        opt.zero_grad()
        loss.backward()
        opt.step()
        elapsed = time.time() - start


    #shift coordinates such that they are positive and redraw graph with new positions
    small_val = torch.min(pos_tensor,dim=0)
    shifted_pos = torch.add(pos_tensor,torch.abs(small_val.values))
    #maybe scale them to 0/1
    scale = torch.max(torch.max(shifted_pos,dim=0)[0]).item()
    new_pos = shifted_pos / scale
    new_pos = new_pos.detach().numpy()
    return new_pos
