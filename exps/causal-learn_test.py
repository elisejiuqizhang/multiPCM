import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

import time # to track runtime

from data_loaders.generated.multivar import MyMultivarData

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
parser=argparse.ArgumentParser("single experiment of causal learning baselines on synthetic data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='4V_indirect', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--method', type=str, default='lingam', help='name of causal learning method: pc, fci, lingam(var-lingam), granger')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'causal-learn_test', args.method, args.causality_type, 'seed'+str(seed))
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
        # list_ind_test=['fisherz', 'chisq','gsq', 'kci', 'mv_fisherz']
        list_ind_test=['fisherz']
        for ind_test in list_ind_test:
            g, edges = fci(dataset=data, independence_test_method=ind_test, alpha=0.05, depth=-1, max_path_length=-1, verbose=True, cache_variables_map=None, show_progress=True)
            # visualize graphs
            pydot_graph = GraphUtils.to_pydot(g, labels=var_names)
            pydot_graph.write_png(os.path.join(save_dir, file_name+f'_fci_{ind_test}_graph.png'))

    elif args.method=='ges':
        # parameters: https://causal-learn.readthedocs.io/en/latest/search_methods_index/Score-based%20causal%20discovery%20methods/GES.html#parameters
        # I use the following score functions: 'local_score_marginal_general', 'local_score_BIC', 'local_score_BDeu'
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
        # model = lingam.ICALiNGAM(random_state=args.seed, max_iter=int(1e5))
        # model.fit(data)
        # order=model.causal_order_
        # adj_matrix=model.adjacency_matrix_

        model = lingam.VARLiNGAM(random_state=args.seed, prune=True, criterion='bic')
        start_time = time.time()
        model.fit(data)
        time_spent = time.time() - start_time
        order=model.causal_order_
        adj_matrix=model.adjacency_matrices_[1]
        # visualization
        G=nx.DiGraph()
        G.add_nodes_from(var_names)
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                # if adj_matrix[i,j]!=0:
                if abs(adj_matrix[i,j])>0.05:
                    G.add_edge(var_names[i], var_names[j])
        # save results
        with open(os.path.join(save_dir, file_name+'_order.txt'), 'w') as f:
            f.write(','.join([str(x) for x in order])+'\n\n')
        np.save(os.path.join(save_dir, file_name+'_adj_matrix.npy'), adj_matrix)
        # visualize graph
        nx.draw(G, with_labels=True)
        plt.savefig(os.path.join(save_dir, file_name+'_lingam_graph.png'))
        plt.close()
        # save the time spent
        with open(os.path.join(save_dir, file_name+'_time.txt'), 'w') as f:
            f.write(str(time_spent))


        # model = lingam.VARLiNGAM(random_state=args.seed, lags=2, prune=True, criterion='bic')
        # start_time = time.time()
        # model.fit(data)
        # total_effects=model.bootstrap(data, n_sampling=20).get_total_causal_effects()
        # time_spent = time.time() - start_time
        # order=model.causal_order_
        # #save total_effects
        # np.save(os.path.join(save_dir, file_name+'_total_effects.npy'), total_effects)
        # # save the time spent
        # with open(os.path.join(save_dir, file_name+'_time.txt'), 'w') as f:
        #     f.write(str(time_spent))

    elif args.method=='granger':
        model = Granger(maxlag=1)
        start_time = time.time()
        coef=model.granger_lasso(data)
        time_spent = time.time() - start_time
        # save the coef
        np.save(os.path.join(save_dir, file_name+'_coef.npy'), coef)
        # visualization
        G=nx.DiGraph()
        G.add_nodes_from(var_names)
        for i in range(coef.shape[0]):
            for j in range(coef.shape[1]):
                # if coef[i,j]!=0:
                if abs(coef[i,j])>0.05:
                    G.add_edge(var_names[i], var_names[j])
        # visualize graph
        nx.draw(G, with_labels=True)
        plt.savefig(os.path.join(save_dir, file_name+'_granger_graph.png'))
        plt.close()
        
        # save the time spent
        with open(os.path.join(save_dir, file_name+'_time.txt'), 'w') as f:
            f.write(str(time_spent))