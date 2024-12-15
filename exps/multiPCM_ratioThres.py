import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt

from data_loaders.generated.multivar import MyMultivarData
from utils.causal_simplex import PCM_simplex

import argparse
parser = argparse.ArgumentParser("Process outputs of multiPCMs")
parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='3V_direct', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=3500, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=5e-3, help='noise level')

parser.add_argument('--tau', type=int, default=2, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=6, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors for PCM")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
# parser.add_argument('--pcm_thres', type=float, default=0.45, help="Threshold for PCM")

# name of cause and effect (each is single variable, the rest are all treated as conditions)
parser.add_argument('--cause', type=str, default='X')
parser.add_argument('--effect', type=str, default='Y')

args=parser.parse_args()

# list of PCM thresholds
list_pcm_thres = np.arange(0.05, 1.0, 0.05)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'multiPCM_ratioThres', args.causality_type, str(args.seed))
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
        # file_names = [prefix+'_3']
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
    file_save_name=file_name+f'_L{args.L}__tau{args.tau}_emd{args.emd}_knn{args.knn}'
    with open(os.path.join(save_dir, file_save_name+'_output.txt'), 'w') as f:
        f.write('sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2\n')
        f.write(','.join([str(x) for x in output])+'\n\n')
    np.save(os.path.join(save_dir, file_save_name+'_output.npy'), output)

    # loop over all PCM thresholds
    arr_label=np.zeros(len(list_pcm_thres)) # the predicted label of whether or not this is direct causality

    if args.score_type=='err':
        arr_ratio=np.full((len(list_pcm_thres)), output[2]) # the third output, the ratio of error
        mask=(arr_ratio>0.95)&(arr_ratio<1.05)
        arr_label[arr_ratio>list_pcm_thres]=1 # ratio of indirectError over directError larger than threshold (significantly increased), then likely the other vars are condition
        arr_label[mask]=0 # ratio of indirectCorr over directCorr close enough, then likely the other vars are not condition
        
    elif args.score_type=='corr' or args.score_type=='r2':
        arr_ratio=np.full((len(list_pcm_thres)), output[5]) # the sixth output, the ratio of correlation
        mask=(arr_ratio>0.95)&(arr_ratio<1.05)
        arr_label[arr_ratio<list_pcm_thres]=1 # ratio of indirectCorr over directCorr smaller than threshold (significantly decreased), then likely the other vars are condition
        arr_label[mask]=0 # ratio of indirectCorr over directCorr close enough, then likely the other vars are not condition
        

    # save the labels
    # thres range and labels together
    arr_combined=np.vstack((list_pcm_thres, arr_label))
    np.save(os.path.join(save_dir, file_save_name+'_labels.npy'), arr_combined)

    # plot the ratio and labels
    plt.plot(list_pcm_thres, arr_label, label='label')

    plt.xlabel('PCM threshold')
    plt.ylabel('Label')
    plt.title(f'Predicted labels for causality between {args.cause} and {args.effect} in {file_name}')
    plt.legend()
    plt.savefig(os.path.join(save_dir, file_save_name+'_labels.png'))
    plt.close()