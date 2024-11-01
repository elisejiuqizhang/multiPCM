import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.generated.multivar import MyMultivarData
from utils.causal_simplex import PCM_simplex

import argparse
parser = argparse.ArgumentParser("single experiment of PCM on synthetic data")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='3V_indirect', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--tau', type=int, default=1, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=16, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors for PCM")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
parser.add_argument('--pcm_thres', type=float, default=0.45, help="Threshold for PCM")

# name of cause and effect (each is single variable, the rest are all treated as conditions)
parser.add_argument('--cause', type=str, default='X')
parser.add_argument('--effect', type=str, default='Y')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'multiPCM_test', args.causality_type, 'seed'+str(seed))
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
    # load data
    dataset=MyMultivarData(os.path.join(root,args.data_dir,args.causality_type,file_name+'.csv'))
    df=dataset.df

    # get the name lists of cause, effect and condition
    var_list = df.columns
    list_cause=[args.cause]
    list_effect=[args.effect]
    list_conditions=[var for var in var_list if var!=args.cause and var!=args.effect]

    pcm=PCM_simplex(df=df, causes=list_cause, effects=list_effect, cond=list_conditions, tau=args.tau, emd=args.emd, L=args.L, knn=args.knn)
    output=pcm.causality() # order: sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2

    del pcm

    # (threshold independent) save all outputs first as text
    file_save_name=file_name+f'_L{args.L}__tau{args.tau}_emd{args.emd}_knn{args.knn}_pcmThres{args.pcm_thres}'
    with open(os.path.join(save_dir, file_save_name+'_output.txt'), 'w') as f:
        f.write('sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2\n')
        f.write(','.join([str(x) for x in output])+'\n\n')
    np.save(os.path.join(save_dir, file_save_name+'_output.npy'), output)




    # then (threshold dependent), depend on the score type and threshold, determine the causality
    if args.score_type=='err':
        if output[2]>0.95 and output[2]<1.05: # ratio of indirectError over directError close enough
            # then likely the other variables are not condition
            result_idx=0 # means "not condition"
        else: # then compare with threshold
            if output[2]>=args.pcm_thres: # ratio of indirectError over directError is significantly increased, then likely the other variables are condition
                result_idx=1 # means "condition"
            else:
                result_idx=0

    elif args.score_type=='corr' or args.score_type=='r2':
        if output[5]>0.95 and output[5]<1.05:
            result_idx=0
        else:
            if output[5]<=args.pcm_thres: # if the correlation ratio is significantly decreased, then likely the other variables are condition
                result_idx=1
            else:
                result_idx=0

    # print the result statement to the text file
    with open(os.path.join(save_dir, file_save_name+'_conclus.txt'), 'w') as f:
        if result_idx==0:
            f.write('The other variables are not condition.\n')
        else:
            f.write('The other variables are condition.\n')

    # save the index of the result
    np.save(os.path.join(save_dir, file_save_name+'_result_idx.npy'), result_idx)
