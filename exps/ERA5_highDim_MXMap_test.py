import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.ERA5.multivar import ERA5MultivarData

from utils.causal_simplex import PCM_simplex       
from utils.MXMap import MXMap

# var_list
# var_names = ['tcw', 'rad', 'T_adv_950', 'T_2m'] # to be manually set
# var_names = ['tcw', 'terr_rad', 'T_adv_950', 'T_2m']
# var_names = ['tcw', 'solar_rad', 'T_adv_950', 'T_2m']
var_names = ['tcw', 'terr_rad', 'solar_rad', 'T_adv_950', 'T_2m']

# var_names = ['tcw', 'T_adv_950', 'T_adv_850', 'T_2m']
# var_names = ['tcw', 'rad', 'T_adv_950', 'T_adv_850', 'T_2m']
# var_names = ['tcw', 'terr_rad', 'solar_rad', 'T_adv_950', 'T_adv_850', 'T_2m']
# var_names = ['tcw', 'rad', 'terr_rad', 'solar_rad', 'T_adv_950', 'T_adv_850', 'T_2m']


file_name = str()
for var in var_names:
    # if not the last one
    if var != var_names[-1]:
        file_name += var + '_'
    else:
        file_name += var

import argparse
parser=argparse.ArgumentParser("single experiment of MXMap on ERA5 data")

parser.add_argument('--data_dir', type=str, default='data_files/data/ERA5')
parser.add_argument('--csv', type=str, default='Timeseries_Upstream.csv')

parser.add_argument('--w_start', type=int, default=12, help="Winter start month")
parser.add_argument('--w_end', type=int, default=2, help="Winter end month")
parser.add_argument('--scaler', type=str, default='std', help="Options are None, minmax/mm, standard/std")
parser.add_argument('--season_rm', type=str, default=None, help="Options are None, daily/d, weekly/w, monthly/m")

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=6000, help='length of input time series')

parser.add_argument('--tau', type=int, default=4, help="Cross mapping tau-lag")
parser.add_argument('--emd', type=int, default=6, help="Cross mapping embedding dimension")
parser.add_argument('--knn', type=int, default=10, help="Neighborhood size")

parser.add_argument('--kNN_model', type=str, default='vanilla') # Options are 'PCA' or 'vanilla'
parser.add_argument('--pca_dim', type=int, default=3, help="Number of PCA components")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
parser.add_argument('--pcm_thres', type=float, default=0.7, help="Threshold of the ratio (sc_indrect/sc_direct) for phase 2 (PCM)")

# # name of 3 variables
# parser.add_argument('--X_name', type=str, default='tcw')
# parser.add_argument('--Y_name', type=str, default='rad')
# parser.add_argument('--Z_name', type=str, default='T_2m')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# folder to store outputs
save_dir=os.path.join(root, 'outputs', f'ERA5_MXMap_test_pcmThres{args.pcm_thres}', 'seed'+str(seed), f'tau_{args.tau}_emd_{args.emd}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.kNN_model == 'PCA':
    save_dir = os.path.join(save_dir, f'pca_dim_{args.pca_dim}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


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

file_dir=os.path.join(save_dir, file_name)

# create MXMap object
if args.kNN_model == 'PCA':
    model=MXMap(df=df, tau=args.tau, emd=args.emd, L=args.L, knn=args.knn, pca_dim=args.pca_dim)
else:
    model=MXMap(df=df, tau=args.tau, emd=args.emd, L=args.L, knn=args.knn)
ch=model.fit()
model.draw_graph(file_dir)

print('ch:', ch)

with open(file_dir+'_ch.txt', 'w') as f:
    f.write(str(ch)+'\n')

# print stats in phase one
print('Phase 1 stats:')
print(model.phase1_stats)
with open(file_dir+'_phase1_stats.txt', 'w') as f:
    f.write(str(model.phase1_stats)+'\n')

# print stats in phase two
print('Phase 2 stats:')
print(model.phase2_stats)
with open(file_dir+'_phase2_stats.txt', 'w') as f:
    f.write(str(model.phase2_stats)+'\n')
    