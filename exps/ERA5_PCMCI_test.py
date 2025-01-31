import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.ERA5.multivar import ERA5MultivarData

import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn
# from tigramite.independence_tests.cmisymb import CMIsymb

import argparse
import contextlib

parser = argparse.ArgumentParser("single experiment of PCMCI on ERA5 data")

parser.add_argument('--data_dir', type=str, default='data_files/data/ERA5')
parser.add_argument('--csv', type=str, default='Timeseries_Upstream.csv')

parser.add_argument('--w_start', type=int, default=12, help="Winter start month")
parser.add_argument('--w_end', type=int, default=2, help="Winter end month")
parser.add_argument('--scaler', type=str, default='std', help="Options are None, minmax/mm, standard/std")
parser.add_argument('--season_rm', type=str, default=None, help="Options are None, daily/d, weekly/w, monthly/m")

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=6000, help='length of input time series')
parser.add_argument('--corrType', type=str, default='ParCorr', help='name of correlation scores: ParCorr, RobustParCorr')
parser.add_argument('--tau_max', type=int, default=2, help="Max lag value for PCMCI")
parser.add_argument('--alpha', type=float, default=0.2, help="Significance level for PCMCI")

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
save_dir = os.path.join(root, 'outputs', 'ERA5_PCMCI_test', args.corrType, args.X_name+'_'+args.Y_name+'_'+args.Z_name, 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# var_names
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

# transform to tigramite dataframe
dataframe = pp.DataFrame(df.values, datatime=np.arange(args.L), var_names=var_names)

# choose correlation type

if args.corrType=='ParCorr':
    # parcorr = ParCorr(significance='analytic')
    parcorr=ParCorr(significance='fixed_thres')
elif args.corrType=='RobustParCorr':
    # parcorr = RobustParCorr(significance='analytic')
    parcorr = RobustParCorr(significance='fixed_thres')
elif args.corrType=='ParCorrWLS':
    # parcorr = ParCorrWLS(significance='analytic')
    parcorr = ParCorrWLS(significance='fixed_thres')
# elif args.corrType=='GPDC':
#     parcorr = GPDC(significance='fixed_thres')
# elif args.corrType=='CMIknn':
#     parcorr = CMIknn(significance='fixed_thres')
# elif args.corrType=='CMIsymb':
#     parcorr=CMIsymb(significance='fixed_thres')
else:
    raise ValueError(f'corrType {args.corrType} not recognized')


pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

# # plot the lagged correlations
# correlations = pcmci.get_lagged_dependencies(tau_max=25, val_only=True)['val_matrix']
# lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names': var_names, 'x_base':5, 'y_base':.5})
# # plt.show()
# plt.savefig(os.path.join(save_dir, 'lagged_correlations.png'))
# plt.close()

# get the terminal outputs and save to txt for this following function
file_save_name = f'L{args.L}_tauMax{args.tau_max}_{args.scaler}'
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
        
# plot the graph
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
plt.savefig(os.path.join(save_dir, file_save_name+f'_alpha{args.alpha}_tsgraph.png'))
plt.close()