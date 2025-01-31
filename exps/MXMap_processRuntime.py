# read in the results of the recorded runtime, and record the average runtime across 10 seeds
# also try to get visualization

import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

list_seeds=[97, 197, 297, 397, 497, 597, 697, 797, 897, 997]
list_causality_types=['3V_direct', '3V_indirect', '3V_both_noCycle', '3V_both_Cycle', '4V_direct', '4V_indirect', '4V_both_noCycle', '4V_both_Cycle', '5V_direct', '5V_indirect', '5V_both_noCycle', '5V_both_Cycle', '6V_direct', '6V_indirect', '6V_both_noCycle', '6V_both_Cycle', '7V_direct', '7V_indirect', '7V_both_noCycle', '7V_both_Cycle', '8V_direct']

import argparse
parser=argparse.ArgumentParser("Read and process runtime of MXMap")

parser.add_argument('--read_dir', type=str, default='outputs/MXMap_test')
parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

parser.add_argument('--tau', type=int, default=2, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=6, help="Cross mapping embedding dimension")

parser.add_argument('--kNN_model', type=str, default='vanilla') # Options are 'PCA' or 'vanilla'
parser.add_argument('--pca_dim', type=int, default=3, help="Number of PCA components")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
parser.add_argument('--bivCCM_thres', type=float, default=1.0, help="Threshold of the ratio (sc1/sc2) for phase 1 (bivariate CCM), used to determine causal direction")
parser.add_argument('--pcm_thres', type=float, default=0.45, help="Threshold of the ratio (sc_indrect/sc_direct) for phase 2 (PCM)")

args=parser.parse_args()

# folder to store outputs
save_dir=os.path.join(root, 'outputs', 'MXMap_processRuntime')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# path to read in the results
read_dir=os.path.join(root, args.read_dir)

list_sys_name=[]
list_avg_runtime=[]



for causality_type in list_causality_types:
    # get file names (since certain system contains multiple structures)
    if args.noiseType!=None and args.noiseType.lower()!='none': # with noise
        if causality_type == '3V_direct' or causality_type=='4V_both_noCycle' or causality_type=='4V_both_Cycle':
            prefix = causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
            file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
        elif causality_type == '3V_indirect' or causality_type=='3V_both_noCycle' or causality_type=='3V_both_Cycle':
            file_names = [causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
        elif causality_type == '4V_direct' or causality_type=='4V_indirect':
            prefix = causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}'
            file_names = [prefix+'_1', prefix+'_2']
        else: # beyond 4V, only one case each
            file_names = [causality_type+f'_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}']
    else: # no noise
        if causality_type == '3V_direct' or causality_type=='4V_both_noCycle' or causality_type=='4V_both_Cycle':
            prefix=causality_type+'_noNoise'
            file_names = [prefix+'_1', prefix+'_2', prefix+'_3']
        elif causality_type == '3V_indirect' or causality_type=='3V_both_noCycle' or causality_type=='3V_both_Cycle':
            file_names = [causality_type+'_noNoise']
        elif causality_type == '4V_direct' or causality_type=='4V_indirect':
            prefix=causality_type+'_noNoise'
            file_names = [prefix+'_1', prefix+'_2']
        else: # beyond 4V, only one case each
            file_names = [causality_type+'_noNoise']

    for file_name in file_names:
        list_runtime_by_seed=[]
        for seed in list_seeds:
            # read from .txt file
            runtime_path=os.path.join(read_dir, causality_type, 'seed'+str(seed), f'tau_{args.tau}_emd_{args.emd}', file_name+'_time.txt')
            with open(runtime_path, 'r') as f:
                runtime=float(f.read())
            list_runtime_by_seed.append(runtime)
        list_avg_runtime.append(np.mean(list_runtime_by_seed))
        list_sys_name.append(file_name)

# save the results first
save_path=os.path.join(save_dir, f'avg_runtime_tau{args.tau}_emd{args.emd}.txt')

list_sys_to_plot=[]
for i in range(3, 9):
    var_system = str(i) + "V"
    var_systems = []
    for sys_name in list_sys_name:
        if var_system in sys_name and "_direct" in sys_name and "noNoise" in sys_name:
            var_systems.append(sys_name)
    # print(var_systems[0])
    list_sys_to_plot.append(var_systems[0])

# get the runtime to plot based on corresponding system names
list_avg_runtime_to_plot=[]
for sys_name in list_sys_to_plot:
    index=list_sys_name.index(sys_name)
    list_avg_runtime_to_plot.append(list_avg_runtime[index])

# plot the results - x axis: list_sys_to_plot, y axis: list_avg_runtime
plt.figure(figsize=(12, 8))
# sns.barplot(x=list_sys_to_plot, y=list_avg_runtime_to_plot)
# also plot the trend line
plt.plot(list_sys_to_plot, list_avg_runtime_to_plot, 'r--')
plt.xticks(rotation=45)
plt.ylabel('Average runtime (s)')
plt.title(f'Average runtime of MXMap with tau={args.tau}, emd={args.emd}')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'avg_runtime_tau{args.tau}_emd{args.emd}.png'))
plt.show()

