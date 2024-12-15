import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.generated.multivar import MyMultivarData

from utils.causal_simplex import PCM_simplex
from utils.MXMap import MXMap

import argparse
parser=argparse.ArgumentParser("single experiment of MXMap on simulated data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='4V_both_noCycle', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=197, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--tau', type=int, default=1, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=4, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=10, help="Neighborhood size")

parser.add_argument('--kNN_model', type=str, default='vanilla') # Options are 'PCA' or 'vanilla'
parser.add_argument('--pca_dim', type=int, default=3, help="Number of PCA components")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
parser.add_argument('--bivCCM_thres', type=float, default=1.0, help="Threshold of the ratio (sc1/sc2) for phase 1 (bivariate CCM), used to determine causal direction")
parser.add_argument('--pcm_thres', type=float, default=0.45, help="Threshold of the ratio (sc_indrect/sc_direct) for phase 2 (PCM)")

args=parser.parse_args()

# set seeds
seed = args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir=os.path.join(root, 'outputs', 'MXMap_test', args.causality_type, 'seed'+str(seed), f'tau_{args.tau}_emd_{args.emd}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.kNN_model == 'PCA':
    save_dir = os.path.join(save_dir, f'pca_dim_{args.pca_dim}')
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
else: # no noise
    if args.causality_type == '3V_direct' or args.causality_type=='4V_both_noCycle' or args.causality_type=='4V_both_Cycle':
        prefix=args.causality_type+'_noNoise'
        file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
    elif args.causality_type == '3V_indirect' or args.causality_type=='3V_both_noCycle' or args.causality_type=='3V_both_Cycle':
        file_names = [args.causality_type+'_noNoise']
    elif args.causality_type == '4V_direct' or args.causality_type=='4V_indirect':
        prefix=args.causality_type+'_noNoise'
        file_names = [prefix+'_1', prefix+'_2']


for file_name in file_names:
    # file save name
    file_dir=os.path.join(save_dir, file_name)
    # load data
    dataset=MyMultivarData(os.path.join(root,args.data_dir,args.causality_type,file_name+'.csv'))
    df=dataset.df
    # get the name lists of cause, effect and condition
    var_list = df.columns.tolist()
    # create MXMap object
    if args.kNN_model == 'PCA':
        model=MXMap(df, tau=args.tau, emd=args.emd, score_type=args.score_type, bivCCM_thres=args.bivCCM_thres, pcm_thres=args.pcm_thres, knn=args.knn, L=args.L, method=args.kNN_model, pca_dim=args.pca_dim)
    else:
        model=MXMap(df, tau=args.tau, emd=args.emd, score_type=args.score_type, bivCCM_thres=args.bivCCM_thres, pcm_thres=args.pcm_thres, knn=args.knn, L=args.L, method=args.kNN_model)
    ch=model.fit()
    model.draw_graph(file_dir)

    print('ch:', ch)

    with open(file_dir+'_ch.txt', 'w') as f:
        f.write(str(ch))

    # print the stats in phase one - determine the order
    print('Phase 1 stats:')
    print(model.phase1_stats)
    with open(file_dir+'_phase1_stats.txt', 'w') as f:
        f.write(str(model.phase1_stats))

    # print the stats in phase two - determine the PCM
    print('Phase 2 stats:')
    print(model.phase2_stats)
    with open(file_dir+'_phase2_stats.txt', 'w') as f:
        f.write(str(model.phase2_stats))

