import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
parser = argparse.ArgumentParser("Process outputs of multiPCM with various thresholds of PCM ratio")
parser.add_argument('--data_dir', type=str, default='outputs/multiPCM_ratioThres')
parser.add_argument('--causality_type', type=str, default='both_Cycle', help='Options: direct, indirect, both, both_Cycle, both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='if None, average over all seeds; else an int as random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=3500, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='gNoise', help='type of noise. Options: gNoise, lpNoise, or None')
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

# # list of PCM thresholds
# list_pcm_thres = np.arange(0.1, 0.9, 0.05)

# # list of system names
# if args.causality_type == 'direct':
#     system_names = ['3V_direct_1', '3V_direct_2', '3V_direct_3', '4V_direct_1', '4V_direct_2']
# elif args.causality_type == 'indirect':
#     system_names = ['3V_indirect', '4V_indirect_1', '4V_indirect_2']
# elif args.causality_type == 'both':
#     system_names = ['3V_both_Cycle', '3V_both_noCycle', '4V_both_Cycle_1', '4V_both_Cycle_2', '4V_both_noCycle_1', '4V_both_noCycle_2', '4V_both_noCycle_3']
# elif args.causality_type == 'both_Cycle':
#     system_names = ['3V_both_Cycle', '4V_both_Cycle_1', '4V_both_Cycle_2']
# elif args.causality_type == 'both_noCycle':
#     system_names = ['3V_both_noCycle', '4V_both_noCycle_1', '4V_both_noCycle_2', '4V_both_noCycle_3']

if args.causality_type == 'direct':
    systems = ['3V_direct', '4V_direct']
elif args.causality_type == 'indirect':
    systems = ['3V_indirect', '4V_indirect']
elif args.causality_type == 'both':
    systems = ['3V_both_Cycle', '3V_both_noCycle', '4V_both_Cycle', '4V_both_noCycle']
elif args.causality_type == 'both_Cycle':
    systems = ['3V_both_Cycle', '4V_both_Cycle']
elif args.causality_type == 'both_noCycle':
    systems = ['3V_both_noCycle', '4V_both_noCycle']


# folder to store outputs
save_dir = os.path.join(root,'outputs', 'multiPCM_ratioThres_viz', args.causality_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.noiseType!=None and args.noiseType.lower()!='none':
    save_dir = os.path.join(save_dir, args.noiseType+'_'+args.noiseInjectType+'_'+str(args.noiseLevel))
else:
    save_dir = os.path.join(save_dir, 'noNoise')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read in data
if args.seed!=None: # only plot for one seed
    labels=[]
    sample_names=[]
    for system in systems:
        data_dir = os.path.join(root, args.data_dir, system)
        # get file name
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if system=='3V_direct' or system=='4V_both_noCycle' or system=='4V_both_Cycle':
                prefix = system+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
                file_names =  [prefix+'_1', prefix+'_2', prefix+'_3']
            elif system=='3V_indirect' or system=='3V_both_noCycle' or system=='3V_both_Cycle':
                file_names = [system+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
            elif system=='4V_direct' or system=='4V_indirect':
                prefix = system+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
                file_names = [prefix+'_1', prefix+'_2']
        else:
            if system=='3V_direct' or system=='4V_both_noCycle' or system=='4V_both_Cycle':
                prefix=system+'_noNoise'
                file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
            elif system=='3V_indirect' or system=='3V_both_noCycle' or system=='3V_both_Cycle':
                file_names = [system+'_noNoise']
            elif system=='4V_direct' or system=='4V_indirect':
                prefix=system+'_noNoise'
                file_names = [prefix+'_1', prefix+'_2']

        for file_name in file_names:
            sample_names.append(file_name)
            # load data
            file_name+=f'_L{args.L}__tau{args.tau}_emd{args.emd}_knn{args.knn}'
            label=np.load(os.path.join(data_dir, str(args.seed), file_name+'_labels.npy'))
            labels.append(label[1,:])
    # thres=label[0,:]
    thres=[round(x, 2) for x in label[0,:]]
    labels=np.array(labels)

    # Convert labels into a DataFrame for Seaborn-friendly visualization
    label_df = pd.DataFrame(labels, index=sample_names, columns=thres)

    # Plot 1: Heatmap of Labels
    plt.figure(figsize=(12, 6))
    sns.heatmap(label_df, cmap="coolwarm", cbar_kws={'label': 'Binary Label (0 or 1)'}, linewidths=0.5)
    plt.title('Binary Labels Across Thresholds for Multiple Systems')
    plt.xlabel('Threshold')
    plt.ylabel('System Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_labels.png'))
    plt.show()

    # Plot 2: Step Plot of Binary Labels for Each System
    plt.figure(figsize=(12, 6))
    for i, system_name in enumerate(sample_names):
        plt.step(thres, labels[i] + i, where='post', linewidth=2, label=system_name)
    plt.yticks(np.arange(len(sample_names)), sample_names)
    plt.axvline(np.mean(thres), color='gray', linestyle='--', label='Midpoint Threshold')
    plt.title('Step Plot of Binary Labels Across Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Systems (Offset for Clarity)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stepplot_labels.png'))
    plt.show()

    # Plot 3: Change Heatmap - Highlight Flips in Binary Labels
    plt.figure(figsize=(12, 6))
    label_changes = np.diff(labels, axis=1) != 0  # True where labels flip
    change_df = pd.DataFrame(label_changes, index=sample_names, columns=thres[1:])
    sns.heatmap(change_df, cmap="YlGnBu", cbar_kws={'label': 'Label Flip (Change)'}, linewidths=0.5)
    plt.title('Regions of Label Changes Across Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('System Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'heatmap_changes.png'))
    plt.show()