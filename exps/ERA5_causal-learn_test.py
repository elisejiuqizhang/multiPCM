import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.ERA5.multivar import ERA5MultivarData

import causallearn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.Granger.Granger import Granger

# visualize adjacency matrix
import networkx as nx
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

import argparse
parser=argparse.ArgumentParser("Single experiment of causal learning baselines on ERA5 data")

parser.add_argument('--data_dir', type=str, default='data_files/data/ERA5')
parser.add_argument('--csv', type=str, default='Timeseries_Upstream.csv')

parser.add_argument('--w_start', type=int, default=12, help="Winter start month")
parser.add_argument('--w_end', type=int, default=2, help="Winter end month")
parser.add_argument('--scaler', type=str, default='std', help="Options are None, minmax/mm, standard/std")
parser.add_argument('--season_rm', type=str, default=None, help="Options are None, daily/d, weekly/w, monthly/m")

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=6000, help='length of input time series')

parser.add_argument('--method', type=str, default='granger', help='name of causal learning method: pc, fci, lingam')

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
save_dir = os.path.join(root, 'outputs', 'ERA5_causal-learn_test', args.method, 'seed'+str(seed))
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

# choose a method
if args.method=='pc':
    # independence test method: 'fisherz', 'chisq','gsq', 'kci', 'mv_fisherz' - run all
    # list_ind_test=['fisherz', 'chisq','gsq', 'kci', 'mv_fisherz']
    list_ind_test=['fisherz']
    for ind_test in list_ind_test:
        cg = pc(data=data, alpha=0.05, indep_test=ind_test, stable=True, uc_rule=1, uc_priority=2, mvpc=False, correction_name='MV_Crtn_Fisher_Z', background_knowledge=None, verbose=True, show_progress=True)
        # visualize graphs
        g=cg.G
        pydot_graph = GraphUtils.to_pydot(g, labels=var_names)
        pydot_graph.write_png(os.path.join(save_dir, file_name+f'_pc_{ind_test}_graph.png'))

elif args.method=='fci':
    # independence test method: 'fisherz', 'chisq','gsq', 'kci', 'mv_fisherz'
    list_ind_test=['fisherz', 'chisq','gsq', 'kci', 'mv_fisherz']
    for ind_test in list_ind_test:
        g, edges = fci(dataset=data, independence_test_method=ind_test, alpha=0.05, depth=-1, max_path_length=-1, verbose=True, cache_variables_map=None, show_progress=True)
        # visualize graphs
        pydot_graph = GraphUtils.to_pydot(g, labels=var_names)
        pydot_graph.write_png(os.path.join(save_dir, file_name+f'_fci_{ind_test}_graph.png'))

elif args.method=='ges':
    list_sc_func=['local_score_marginal_general', 'local_score_BIC', 'local_score_BDeu']
    for sc_func in list_sc_func:
        Record = ges(X=data, score_func=sc_func, maxP=4)
        g=Record['G']
        insertUpdate=Record['update1']
        g1=Record['G_step1']
        deleteUpdate=Record['update2']
        g2=Record['G_step2']
        score=Record['score']
        # save results
        with open(os.path.join(save_dir, file_name+f'_ges_{sc_func}_output.txt'), 'w') as f:
            f.write('insertUpdate, deleteUpdate, score\n')
            f.write(','.join([str(x) for x in [insertUpdate, deleteUpdate, score]])+'\n\n')
        # save the graphs
        np.save(os.path.join(save_dir, file_name+f'_ges_{sc_func}_graph.npy'), g)
        np.save(os.path.join(save_dir, file_name+f'_ges_{sc_func}_graph1.npy'), g1)
        np.save(os.path.join(save_dir, file_name+f'_ges_{sc_func}_graph2.npy'), g2)
        # visualize graphs
        pydot_graph = GraphUtils.to_pydot(g, labels=var_names)
        pydot_graph.write_png(os.path.join(save_dir, file_name+f'_ges_{sc_func}_graph.png'))

elif args.method=='lingam':
    # model=lingam.ICALiNGAM(random_state=args.seed, max_iter=int(1e5))
    # model.fit(data)
    # order=model.causal_order_
    # adj_matrix=model.adjacency_matrix_

    model = lingam.VARLiNGAM(random_state=args.seed, prune=True, criterion='bic')
    model.fit(data)
    order=model.causal_order_
    adj_matrix=model.adjacency_matrices_[1]


    # visualization
    G=nx.DiGraph()
    G.add_nodes_from(var_names)
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if adj_matrix[i,j]!=0:
                G.add_edge(var_names[i], var_names[j])
    # save results
    with open(os.path.join(save_dir, file_name+'_order.txt'), 'w') as f:
        f.write(','.join([str(x) for x in order])+'\n\n')
    np.save(os.path.join(save_dir, file_name+'_adj_matrix.npy'), adj_matrix) 
    # visualize graph
    nx.draw(G, with_labels=True)
    plt.savefig(os.path.join(save_dir, file_name+'_lingam_graph.png'))   

elif args.method=='granger':
    model = Granger(maxlag=1)
    coef=model.granger_lasso(data)
    # save the coef
    np.save(os.path.join(save_dir, file_name+'_granger_coef.npy'), coef)
    # visualize graph
    G=nx.DiGraph()
    G.add_nodes_from(var_names)
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if coef[i,j]!=0:
                G.add_edge(var_names[i], var_names[j])
    # visualize graph
    nx.draw(G, with_labels=True)
    plt.savefig(os.path.join(save_dir, file_name+'_granger_graph.png'))
    plt.close()
