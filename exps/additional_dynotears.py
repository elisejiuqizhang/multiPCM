from causalnex.structure.dynotears import from_pandas_dynamic

#https://github.com/ckassaad/causal_discovery_for_time_series
def dynotears(data, tau_max=5):
    graph_dict = dict()
    for name in data.columns:
        graph_dict[name] = []

    sm = from_pandas_dynamic(data, p=tau_max, w_threshold=0.01, lambda_w=0.05, lambda_a=0.05)

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



# ----------------------- main -----------------------

import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

import time # to track runtime

from data_loaders.generated.multivar import MyMultivarData
# visualize
import networkx as nx
from causalnex.structure import StructureModel
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

import matplotlib.pyplot as plt
import io

import argparse
parser=argparse.ArgumentParser("single experiment of causal learning baselines on synthetic data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='5V_indirect', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'dynotears', args.causality_type, 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# file name/names for each type of causality
# 3V_direct has 3 structures, denoted as _1, _2, _3
# 3V_indirect, 3V_both_noCycle and 3V_both_Cycle have only 1 structure, no specification in file names
# 4V_direct, 4V_indirect has 2 structures, denoted as _1, _2
# 4V_both_noCycle and 4V_both_Cycle have 3 structures, denoted as _1, _2, _3 (pas encore fait - note Oct.9)

# get file name list
if args.noiseType!=None and args.noiseType.lower()!='none': # with noise
    if args.causality_type == '3V_direct' or args.causality_type=='4V_both_noCycle' or args.causality_type=='4V_both_Cycle':
        prefix = args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
        file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
    elif args.causality_type == '3V_indirect' or args.causality_type=='3V_both_noCycle' or args.causality_type=='3V_both_Cycle':
        file_names = [args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
    elif args.causality_type == '4V_direct' or args.causality_type=='4V_indirect':
        prefix = args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
        file_names = [prefix+'_1', prefix+'_2']
    else: # beyond 4V, only one case each
        file_names = [args.causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
else: # no noise
    if args.causality_type == '3V_direct' or args.causality_type=='4V_both_noCycle' or args.causality_type=='4V_both_Cycle':
        prefix=args.causality_type+'_noNoise'
        file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
    elif args.causality_type == '3V_indirect' or args.causality_type=='3V_both_noCycle' or args.causality_type=='3V_both_Cycle':
        file_names = [args.causality_type+'_noNoise']
    elif args.causality_type == '4V_direct' or args.causality_type=='4V_indirect':
        prefix=args.causality_type+'_noNoise'
        file_names = [prefix+'_1', prefix+'_2']
    else: # beyond 4V, only one case each
        file_names = [args.causality_type+'_noNoise']

if args.causality_type=='3V_direct' or args.causality_type=='3V_indirect' or args.causality_type=='3V_both_Cycle' or args.causality_type=='3V_both_noCycle' or args.causality_type=='3V_immorality':
    n_vars=3
    var_names=['X','Y','Z']
elif args.causality_type=='4V_direct' or args.causality_type=='4V_indirect' or args.causality_type=='4V_both_Cycle' or args.causality_type=='4V_both_noCycle':
    n_vars=4
    var_names=['W','X','Y','Z']
if args.causality_type=='3V_direct' or args.causality_type=='3V_indirect' or args.causality_type=='3V_both_Cycle' or args.causality_type=='3V_both_noCycle':
    n_vars=3
    var_names=['X','Y','Z']
elif args.causality_type=='4V_direct' or args.causality_type=='4V_indirect' or args.causality_type=='4V_both_Cycle' or args.causality_type=='4V_both_noCycle':
    n_vars=4
    var_names=['W','X','Y','Z']
elif args.causality_type=='5V_direct' or args.causality_type=='5V_indirect' or args.causality_type=='5V_both_Cycle' or args.causality_type=='5V_both_noCycle':
    n_vars=5
    var_names=['V','W','X','Y','Z']
elif args.causality_type=='6V_direct' or args.causality_type=='6V_indirect' or args.causality_type=='6V_both_Cycle' or args.causality_type=='6V_both_noCycle':
    n_vars=6
    var_names=['U','V','W','X','Y','Z']
elif args.causality_type=='7V_direct' or args.causality_type=='7V_indirect' or args.causality_type=='7V_both_Cycle' or args.causality_type=='7V_both_noCycle':
    n_vars=7
    var_names=['T','U','V','W','X','Y','Z']
elif args.causality_type=='8V_direct' or args.causality_type=='8V_indirect' or args.causality_type=='8V_both_Cycle' or args.causality_type=='8V_both_noCycle':
    n_vars=8
    var_names=['S','T','U','V','W','X','Y','Z']


# usage following https://notebook.community/jakobrunge/tigramite/tutorials/tigramite_tutorial_basics
for file_name in file_names:
    # load data
    dataset=MyMultivarData(os.path.join(root,args.data_dir,args.causality_type,file_name+'.csv'))
    df=dataset.df

    # arrange df as the order of var_names
    df=df[var_names]

    # start from a random start point and cut off at L
    start_point=random.randint(0,len(df)-args.L)
    df=df[start_point:start_point+args.L]

    g=StructureModel()
    g.add_edges_from([(var_names[i], var_names[j]) for i in range(n_vars) for j in range(n_vars) if i!=j])
    start_time = time.time()
    g_dict, g=dynotears(df, tau_max=1)
    time_spent = time.time() - start_time

    # save the edges
    edges=g.edges # saved in teh fashion of [(nodeName1_lagM, nodeName2_lagN), ...]
    with open(os.path.join(save_dir, file_name+'_edges.txt'), 'w') as f:
        f.write(str(edges))

    # visualize the graph
    viz=plot_structure(g, all_node_attributes=NODE_STYLE.WEAK,all_edge_attributes=EDGE_STYLE.WEAK,)
    # viz.show(os.path.join(save_dir, file_name+'.html'))
    fig = plt.figure(figsize=(12, 8))
    nx.draw_networkx(g, with_labels=True)
    plt.savefig(os.path.join(save_dir, file_name+'.png'))
    plt.close(fig)

    # save dict and time
    with open(os.path.join(save_dir, file_name+'_dynotears_dict.txt'), 'w') as f:
        f.write(str(g_dict))
    with open(os.path.join(save_dir, file_name+'_time.txt'), 'w') as f:
        f.write(str(time_spent))


    # visualize the graph but in a time-compressed way: as long as there is an edge linking two nodes, whether or not there is time lag, consider this an edge between two nodes - can be uni bi directional
    # process the edges as a list of tuples
    edge_dict=dict()
    for pair in edges:
        cause_name=pair[0][0]
        if cause_name not in edge_dict.keys():
            edge_dict[cause_name]=[]
        effect_name=pair[1][0]
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