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

from data_loaders.ERA5.multivar import ERA5MultivarData

# visualize adjacency matrix
import networkx as nx
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

import argparse
parser=argparse.ArgumentParser("Single experiment of tsFCI on ERA5 data")
parser.add_argument('--data_dir', type=str, default='data_files/data/ERA5')
parser.add_argument('--csv', type=str, default='Timeseries_Upstream.csv')

parser.add_argument('--w_start', type=int, default=12, help="Winter start month")
parser.add_argument('--w_end', type=int, default=2, help="Winter end month")
parser.add_argument('--scaler', type=str, default='std', help="Options are None, minmax/mm, standard/std")
parser.add_argument('--season_rm', type=str, default=None, help="Options are None, daily/d, weekly/w, monthly/m")

parser.add_argument('--seed', type=int, default=197, help='random seed, for sampling a random start point for input time series')

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
save_dir = os.path.join(root, 'outputs', 'ERA5_tsFCI', 'seed'+str(seed))
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

# run tsFCI
max_lag=4
graph=tsfci(data, max_lag, independence_test_method='fisherz', alpha=0.01, verbose=True)
visualize_graph(graph,var_names, save_path=os.path.join(save_dir, file_name+'.png'))
