import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.generated.multivar import MyMultivarData

import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI

# from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr

from tigramite.independence_tests.parcorr_wls import ParCorrWLS   
from tigramite.independence_tests.gpdc import GPDC  
from tigramite.independence_tests.cmiknn import CMIknn  
from tigramite.independence_tests.cmisymb import CMIsymb  
from tigramite.independence_tests.gsquared import Gsquared  
from tigramite.independence_tests.regressionCI import RegressionCI

# from tigramite.independence_tests.gpdc import GPDC # missing dcor?
# from tigramite.independence_tests.cmiknn import CMIknn # missing numba?
# from tigramite.independence_tests.cmisymb import CMIsymb # missing numba?

# from tigramite.models import LinearMediation, Prediction

import argparse
import contextlib

parser = argparse.ArgumentParser("single experiment of PCMCI on synthetic data")

parser.add_argument('--data_dir', type=str, default='data_files/data/gen')
parser.add_argument('--causality_type', type=str, default='4V_both_Cycle', help='Options: 3V_direct, 3V_indirect, 3V_both_Cycle, 3V_both_noCycle, 4V_direct, 4V_indirect, 4V_both_Cycle, 4V_both_noCycle')

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=4000, help='length of input time series')

parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=1e-2, help='noise level')

# parser.add_argument('--corrType', type=str, default='GPDC', help='name of correlation scores')
parser.add_argument('--corrType', type=str, default='ParCorr', help='name of correlation scores: ParCorr, RpbustParCorr')
parser.add_argument('--tau_max', type=int, default=4, help="Max lag value for PCMCI")
parser.add_argument('--alpha', type=float, default=0.05, help="Significance level for PCMCI")

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'PCMCI_test', args.corrType, args.causality_type, 'seed'+str(seed))
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

if args.causality_type=='3V_direct' or args.causality_type=='3V_indirect' or args.causality_type=='3V_both_Cycle' or args.causality_type=='3V_both_noCycle':
    n_vars=3
    var_names=['X','Y','Z']
elif args.causality_type=='4V_direct' or args.causality_type=='4V_indirect' or args.causality_type=='4V_both_Cycle' or args.causality_type=='4V_both_noCycle':
    n_vars=4
    var_names=['W','X','Y','Z']


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

    # transform to tigramite dataframe
    dataframe=pp.DataFrame(df.values, datatime=np.arange(args.L), var_names=var_names)

    # Choose a conditional independence test; plot the lagged unconditional dependencies, e.g., the lagged correlations, to identify which maximal time lag tau_max to choose in the causal algorithm
    save_dir_corrPred = os.path.join(save_dir, 'corrPred')
    if not os.path.exists(save_dir_corrPred):
        os.makedirs(save_dir_corrPred)
    if args.corrType=='ParCorr':
        # parcorr = ParCorr(significance='analytic')
        parcorr=ParCorr(significance='fixed_thres')
    elif args.corrType=='RobustParCorr':
        # parcorr = RobustParCorr(significance='analytic')
        parcorr = RobustParCorr(significance='fixed_thres')
    elif args.corrType=='ParCorrWLS':
        # parcorr = ParCorrWLS(significance='analytic')
        parcorr = ParCorrWLS(significance='fixed_thres')
    elif args.corrType=='GPDC':
        parcorr=GPDC(significance='fixed_thres')
    elif args.corrType=='CMIknn':
        parcorr=CMIknn(significance='fixed_thres')
    elif args.corrType=='CMIsymb':
        parcorr=CMIsymb(significance='fixed_thres')
    else:
        raise ValueError(f'corrType {args.corrType} not recognized')

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

    # # plot the lagged correlations
    # correlations = pcmci.get_lagged_dependencies(tau_max=25, val_only=True)['val_matrix']
    # lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names, 'x_base':5, 'y_base':.5}); 
    # plt.show()
    # plt.savefig(os.path.join(save_dir_corrPred, file_name+'_lagged_correlations.png'))
    # plt.close()


    # Here we let PCMCI choose the optimal value by setting it to pc_alpha=None. 
    # Then PCMCI will optimize this parameter in the ParCorr case by the Akaike Information criterion 
    # among a reasonable default list of values (e.g., pc_alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]).

    # get the terminal outputs and save to txt for this following function
    file_save_name=file_name+f'_L{args.L}_tauMax{args.tau_max}'
    output_file = os.path.join(save_dir, file_save_name + '_output.txt')
    with open(output_file, 'w') as f:
        with contextlib.redirect_stdout(f):
            results = pcmci.run_pcmci(tau_max=args.tau_max, pc_alpha=args.alpha)
            # if args.corrType == 'GPDC' or args.corrType == 'CMIknn' or args.corrType == 'CMIsymb':
            #     results = pcmci.run_pcmci(tau_max=args.tau_max, pc_alpha=args.alpha)
            # else:
            #     results = pcmci.run_pcmci(tau_max=args.tau_max, pc_alpha=None)
    # save the results (p_values, mci_parCorr)
    with open(os.path.join(save_dir, file_save_name+'_p-values.npy'), 'wb') as f:
        f.write(results['p_matrix'].round(3))
    with open(os.path.join(save_dir, file_save_name+'_mci_parCorr.npy'), 'wb') as f:
        f.write(results['val_matrix'].round(3))


    
    link_output_file = os.path.join(save_dir, file_save_name + f'_alpha{args.alpha}_link_output.txt')
    with open(link_output_file, 'w') as f:
        with contextlib.redirect_stdout(f):
            pcmci.print_significant_links(p_matrix = results['p_matrix'],val_matrix = results['val_matrix'],
                                            alpha_level = args.alpha)
    
    # plot the graph based on results[0], which is an array of shape [N, N, tau_max+1]
    tp.plot_graph(graph=results['graph'], val_matrix=results['val_matrix'], var_names=var_names)
    plt.savefig(os.path.join(save_dir, file_save_name+f'_alpha{args.alpha}_graph.png'))
    plt.close()

    tp.plot_time_series_graph(
        figsize=(6, 4),
        val_matrix=results['val_matrix'],
        graph=results['graph'],
        var_names=var_names,
        link_colorbar_label='MCI',
    )
    plt.savefig(os.path.join(save_dir, file_save_name+f'_alpha{args.alpha}_time_series_graph.png'))
    plt.close()