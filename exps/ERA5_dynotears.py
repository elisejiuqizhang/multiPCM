from causalnex.structure.dynotears import from_pandas_dynamic

import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.ERA5.multivar import ERA5MultivarData

# visualize adjacency matrix
import networkx as nx
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

# visualize
import networkx as nx
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE


import argparse
parser=argparse.ArgumentParser("Single experiment of DYNOTEARS on ERA5 data")
parser.add_argument('--data_dir', type=str, default='data_files/data/ERA5')
parser.add_argument('--csv', type=str, default='Timeseries_Upstream.csv')

parser.add_argument('--w_start', type=int, default=12, help="Winter start month")
parser.add_argument('--w_end', type=int, default=2, help="Winter end month")
parser.add_argument('--scaler', type=str, default='std', help="Options are None, minmax/mm, standard/std")
parser.add_argument('--season_rm', type=str, default=None, help="Options are None, daily/d, weekly/w, monthly/m")

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=6000, help='length of input time series')

# name of 3 variables
parser.add_argument('--X_name', type=str, default='tcw')
parser.add_argument('--Y_name', type=str, default='rad')
parser.add_argument('--Z_name', type=str, default='T_2m')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'ERA5_dynotears', 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# var_list
X_name=args.X_name
Y_name=args.Y_name
Z_name=args.Z_name
var_names = [X_name, Y_name, Z_name]

# var_names = ['tcw', 'rad', 'T_adv_950', 'T_2m'] # to be manually set
# var_names = ['tcw', 'terr_rad', 'T_adv_950', 'T_2m']
# var_names = ['tcw', 'solar_rad', 'T_adv_950', 'T_2m']
var_names = ['tcw', 'terr_rad', 'solar_rad', 'T_adv_950', 'T_2m']

n_vars=len(var_names)
# file_name = X_name+'_'+Y_name+'_'+Z_name
for i in range(n_vars):
    if i==0:
        file_name=var_names[i]
    else:
        file_name+='_'+var_names[i]

# load data
dataset=ERA5MultivarData(csv_path=os.path.join(root, args.data_dir, args.csv),
                         var_list=var_names, 
                         winter_start_month=args.w_start, 
                         winter_end_month=args.w_end, 
                         scaler=args.scaler, 
                         season_rm=args.season_rm)
df=dataset.df

# select a random start point
start_idx=random.randint(0, len(df)-args.L)
df=df[start_idx:start_idx+args.L]

data=df.values


def dynotears(data, tau_max=4):
    graph_dict = dict()
    for name in data.columns:
        graph_dict[name] = []

    # sm = from_pandas_dynamic(data, p=tau_max, w_threshold=0.01, lambda_w=0.05, lambda_a=0.05)
    sm = from_pandas_dynamic(data, max_iter=100, p=tau_max, w_threshold=1e-2, lambda_w=0.005, lambda_a=0.005)

    # print(sm.edges)
    # print(sm.pred)

    tname_to_name_dict = dict()
    count_lag = 0
    idx_name = 0
    for tname in sm.nodes:
        tname_to_name_dict[tname] = data.columns[idx_name]
        if count_lag == tau_max:
            idx_name = idx_name +1
            count_lag = -1
        count_lag = count_lag +1

    for ce in sm.edges:
        c = ce[0]
        e = ce[1]
        tc = int(c.partition("lag")[2])
        te = int(e.partition("lag")[2])
        t = tc - te
        if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
            graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))

    # g = sm.to_directed()
    return graph_dict, sm



g=StructureModel()
g.add_edges_from([(var_names[i], var_names[j]) for i in range(n_vars) for j in range(n_vars) if i!=j])
g_dict, g=dynotears(df, tau_max=4)

# save the edges
edges=g.edges # saved in teh fashion of [(nodeName1_lagM, nodeName2_lagN), ...]
with open(os.path.join(save_dir, file_name+'_edges.txt'), 'w') as f:
    f.write(str(edges))

# visualize the graph
viz=plot_structure(g, all_node_attributes=NODE_STYLE.WEAK,all_edge_attributes=EDGE_STYLE.WEAK,)
fig = plt.figure(figsize=(12, 8))
nx.draw_networkx(g, with_labels=True)
plt.savefig(os.path.join(save_dir, file_name+'_graph.png'))
plt.close(fig)


# visualize the graph but in a time-compressed way: as long as there is an edge linking two nodes, whether or not there is time lag, consider this an edge between two nodes - can be uni bi directional
# process the edges as a list of tuples
edge_dict=dict()
for pair in edges:
    cause_name=pair[0][:-5]
    if cause_name not in edge_dict.keys():
        edge_dict[cause_name]=[]
    effect_name=pair[1][:-5]
    if effect_name not in edge_dict[cause_name]:
        edge_dict[cause_name].append(effect_name)

# adjacency matrix
adj_matrix=np.zeros((n_vars,n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        if var_names[i] in edge_dict.keys():
            if var_names[j] in edge_dict[var_names[i]]:
                adj_matrix[i,j]=1

# save the matrix
np.save(os.path.join(save_dir, file_name+'_adj_matrix.npy'), adj_matrix)

# visualize the graph
G=nx.DiGraph()
G.add_nodes_from(var_names)
for i in range(n_vars):
    for j in range(n_vars):
        if adj_matrix[i,j]==1:
            G.add_edge(var_names[i],var_names[j])
nx.draw(G, with_labels=True)
plt.savefig(os.path.join(save_dir, file_name+'_compressed.png'))