# read the outputs of single experiments and process the results (as grid search results)

# output name defined as follows:
# if args.noiseType!=None and args.noiseType.lower()!='none':
#     data.to_csv(os.path.join(save_dir,f'4V_both_Cycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}_3.csv'), index=False)
# else:
#     data.to_csv(os.path.join(save_dir,f'4V_both_Cycle_noNoise_3.csv'), index=False)

import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser("Process outputs of multiPCMs")
parser.add_argument('--data_dir', type=str, default='outputs/multiPCM_test')
parser.add_argument('--causality_type', type=str, default='3V_direct', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=3500, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=5e-3, help='noise level')

# parser.add_argument('--tau', type=int, default=1, help="Cross mapping tau-lag")
# parser.add_argument('--emd', type=int, default=16, help="Cross mapping embedding dimension")
# parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors for PCM")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
parser.add_argument('--pcm_thres', type=float, default=0.45, help="Threshold for PCM")

# name of cause and effect (each is single variable, the rest are all treated as conditions)
parser.add_argument('--cause', type=str, default='X')
parser.add_argument('--effect', type=str, default='Y')

args=parser.parse_args()

# lists of variables of interests to do grid search: tau, emd, knn (see bash scripts for range)
# do 2d grids and 3d visualization
list_tau=np.arange(1,9,1)
list_emd=np.arange(2,9,1)
list_knn=np.arange(10,21,5)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'multiPCM_gridSearch', args.causality_type)
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
    # arrays to store results
    # sc1: direct causality score
    # sc2: indirect causality score
    # sc3: ratio of indirect over direct
    # result: the result reflected by index, 0 means not conditioned, 1 means conditioned
    arr_sc1=np.zeros((len(list_tau),len(list_emd),len(list_knn)))
    arr_sc2=np.zeros((len(list_tau),len(list_emd),len(list_knn)))
    arr_ratio=np.zeros((len(list_tau),len(list_emd),len(list_knn)))
    result_idx_arr=np.zeros((len(list_tau),len(list_emd),len(list_knn))) 

    # loop over all files
    for i in range(len(list_tau)):
        for j in range(len(list_emd)):
            for k in range(len(list_knn)):
                tau=list_tau[i]
                emd=list_emd[j]
                knn=list_knn[k]

                # output order: sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2
                if args.score_type=='err':
                    start_idx=0
                elif args.score_type=='corr':
                    start_idx=3
                elif args.score_type=='r2':
                    start_idx=6
                                
                # load data (two files, _output.npy and result_idx.npy)
                all_scores=np.load(os.path.join(root, args.data_dir, args.causality_type, f'seed{args.seed}', file_name+f'_L{args.L}__tau{tau}_emd{emd}_knn{knn}_pcmThres{args.pcm_thres}_output.npy'))
                result_idx=np.load(os.path.join(root, args.data_dir, args.causality_type, f'seed{args.seed}', file_name+f'_L{args.L}__tau{tau}_emd{emd}_knn{knn}_pcmThres{args.pcm_thres}_result_idx.npy'))

                # fill in the arrays
                arr_sc1[i,j,k]=all_scores[start_idx+0]
                arr_sc2[i,j,k]=all_scores[start_idx+1]
                arr_ratio[i,j,k]=all_scores[start_idx+2]
                result_idx_arr[i,j,k]=result_idx

            # save the arrays
            np.save(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_sc1.npy'), arr_sc1)
            np.save(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_sc2.npy'), arr_sc2)
            np.save(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_ratio.npy'), arr_ratio)

            # save the result_idx_arr
            np.save(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_result_idx.npy'), result_idx_arr)

            

    # fix knn, sweep tau and emd, results are 3d scatter plots, where the height reflects value of score
    for k in range(len(list_knn)):
        # deal with arr_sc1
        fig=plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        tau_grid, emd_grid=np.meshgrid(list_tau, list_emd)
        ax.plot_surface(tau_grid, emd_grid, arr_sc1[:,:,k].transpose(), cmap='viridis')
        ax.set_xlabel('tau')
        ax.set_ylabel('emd')
        ax.set_zlabel('sc1')
        ax.set_title(f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_sc1')
        plt.savefig(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_sc1.png'))
        plt.close()

        # deal with arr_sc2
        fig=plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        tau_grid, emd_grid=np.meshgrid(list_tau, list_emd)
        ax.plot_surface(tau_grid, emd_grid, arr_sc2[:,:,k].transpose(), cmap='viridis')
        ax.set_xlabel('tau')
        ax.set_ylabel('emd')
        ax.set_zlabel('sc2')
        ax.set_title(f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_sc2')
        plt.savefig(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_sc2.png'))
        plt.close()

        # deal with arr_ratio
        fig=plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        tau_grid, emd_grid=np.meshgrid(list_tau, list_emd)
        ax.plot_surface(tau_grid, emd_grid, arr_ratio[:,:,k].transpose(), cmap='viridis')
        ax.set_xlabel('tau')
        ax.set_ylabel('emd')
        ax.set_zlabel('ratio')
        ax.set_title(f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_ratio')
        plt.savefig(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_ratio.png'))
        plt.close()

        # deal with result_idx_arr - this will be a binary results, 2d
        two_colors=['blue', 'red']
        fig=plt.figure()
        ax=fig.add_subplot(111)
        tau_grid, emd_grid=np.meshgrid(list_tau, list_emd)
        for i in range(len(list_tau)):
            for j in range(len(list_emd)):
                # ax.scatter(list_tau[i], list_emd[j], c=two_colors[int(result_idx_arr[i,j,k])])
                if result_idx_arr[i,j,k]==0:
                    plt.scatter(list_tau[i], list_emd[j], c='blue', label='X and Y are directly causal, not via conditions')
                else:
                    plt.scatter(list_tau[i], list_emd[j], c='red', label='X and Y are indirectly causal, via conditions')
        ax.set_xlabel('tau')
        ax.set_ylabel('emd')
        ax.set_title(f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_result_idx')
        # plt.legend()
        plt.savefig(os.path.join(save_dir, f'{file_name}_L{args.L}_pcmThres{args.pcm_thres}_result_idx.png'))
        plt.close()

