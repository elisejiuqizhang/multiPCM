from causallearn.search.ConstraintBased.FCI import *
from causallearn.graph.Graph import Graph
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.utils.FAS import fas
from causallearn.utils.cit import CIT
from causallearn.graph.GraphNode import GraphNode
from typing import List, Tuple, Dict, Set

#helpers
def visualize_graph(graph: Graph, var_names: List[str], save_path: str = None):
    """
    Visualize the causal graph using NetworkX and Matplotlib with variable names.
    :param graph: The causal graph to visualize.
    :param var_names: List of variable names for labeling.
    :param save_path: Path to save the graph image, if specified.
    """
    G = nx.DiGraph()
    
    for edge in graph.get_graph_edges():
        node1 = var_names[int(edge.get_node1().get_name().split("X")[1].split("_")[0]) - 1]
        node2 = var_names[int(edge.get_node2().get_name().split("X")[1].split("_")[0]) - 1]
        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()
        
        if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
            G.add_edge(node1, node2)
        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
            G.add_edge(node2, node1)
        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.ARROW:
            G.add_edge(node1, node2, style="bidirectional")
    
    pos = nx.circular_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, font_weight="bold")
    plt.title("Causal Graph")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
    plt.show()
    plt.close()


def add_lagged_variables(dataset: np.ndarray, max_lag: int) -> Tuple[np.ndarray, List[str]]:
    """
    Create lagged variables for time series data up to a specified lag.
    :param dataset: Original time series data (n_samples, n_features)
    :param max_lag: Maximum lag to consider.
    :return: Lagged dataset and variable names.
    """
    n_samples, n_features = dataset.shape
    lagged_data = []
    var_names = []
    
    for lag in range(max_lag + 1):
        lagged_data.append(dataset[max_lag - lag : n_samples - lag, :])
        var_names += [f"X{j+1}_t-{lag}" if lag > 0 else f"X{j+1}_t" for j in range(n_features)]

    return np.hstack(lagged_data), var_names

def temporal_constraints(graph: Graph, max_lag: int):
    """
    Enforce temporal constraints on the causal graph.
    Edges must respect time ordering.
    :param graph: The current causal graph.
    :param max_lag: Maximum lag.
    """
    for edge in graph.get_graph_edges():
        node1, node2 = edge.get_node1(), edge.get_node2()
        time1 = int(node1.get_name().split("_t-")[1]) if "_t-" in node1.get_name() else 0
        time2 = int(node2.get_name().split("_t-")[1]) if "_t-" in node2.get_name() else 0

        # If time1 > time2, remove or block the edge (no future causes past)
        if time1 > time2:
            graph.remove_edge(edge)

def tsfci(dataset: np.ndarray, max_lag: int, independence_test_method: str = "fisherz", 
          alpha: float = 0.05, depth: int = -1, verbose: bool = False):
    """
    Time Series FCI (tsFCI) algorithm for causal discovery in time series data.
    :param dataset: Original time series data (n_samples, n_features).
    :param max_lag: Maximum time lag to consider.
    :param independence_test_method: Conditional independence test method.
    :param alpha: Significance level for independence tests.
    :param depth: Maximum depth for adjacency search.
    :param verbose: Verbose output.
    :return: Causal graph with temporal constraints applied.
    """
    # Step 1: Create lagged variables
    lagged_data, var_names = add_lagged_variables(dataset, max_lag)
    
    # Step 2: Initialize nodes
    nodes = [GraphNode(name) for name in var_names]

    # Step 3: Perform Fast Adjacency Search (FAS)
    independence_test = CIT(lagged_data, method=independence_test_method)
    graph, sep_sets, _ = fas(lagged_data, nodes, independence_test, alpha, depth=depth, verbose=verbose)

    # Step 4: Reorient all edges to circles
    for edge in graph.get_graph_edges():
        edge.set_endpoint1(Endpoint.CIRCLE)
        edge.set_endpoint2(Endpoint.CIRCLE)

    # Step 5: Enforce temporal constraints
    temporal_constraints(graph, max_lag)

    # Step 6: Apply FCI rules (rule0, rulesR1R2, ruleR3, etc.)
    rule0(graph, nodes, sep_sets, None, verbose)
    change_flag = True
    max_iterations = 100  # Safeguard against infinite loops
    iteration = 0

    while change_flag and iteration < max_iterations:
        change_flag = False  # Reset flag for this iteration
        iteration += 1
        if verbose:
            print(f"Iteration {iteration}: Applying FCI rules...")

        # Apply rules
        change_flag |= rulesR1R2cycle(graph, None, change_flag, verbose)
        change_flag |= ruleR3(graph, sep_sets, None, change_flag, verbose)
        change_flag |= ruleR4B(graph, -1, lagged_data, independence_test, alpha, sep_sets, change_flag, None, verbose)

        if verbose and not change_flag:
            print("No changes detected, stopping rule application.")

    if iteration == max_iterations:
        print("Warning: Maximum iterations reached. Check for rule stability.")

    graph.set_pag(True)
    return graph


# --------------------------- main ---------------------------

import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

import time # to track runtime

from data_loaders.generated.multivar import MyMultivarData
# visualize adjacency matrix
import networkx as nx
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

import argparse
parser=argparse.ArgumentParser("single experiment of causal learning baselines on synthetic data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='4V_indirect', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

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
save_dir = os.path.join(root, 'outputs', 'tsFCI_test', args.causality_type, 'seed'+str(seed))
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

    data=df.values

    # run tsFCI
    max_lag=4
    start_time = time.time()
    graph=tsfci(data, max_lag, independence_test_method='fisherz', alpha=0.05, verbose=False)
    time_spent = time.time() - start_time
    visualize_graph(graph,var_names, save_path=os.path.join(save_dir, file_name+'.png'))
    # save runtime
    with open(os.path.join(save_dir, file_name+'_time.txt'), 'w') as f:
        f.write(str(time_spent))
