import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from data_loaders.ERA5.multivar import ERA5MultivarData
from utils.causal_simplex import PCM_simplex

import argparse

parser = argparse.ArgumentParser("single experiment of PCMCI on ERA5 data")

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
parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors for PCM")

parser.add_argument('--score_type', type=str, default='corr', help="Options are err or corr or r2")
# parser.add_argument('--pcm_thres', type=float, default=0.5, help="Threshold for PCM")

# name of 3 variables
parser.add_argument('--X_name', type=str, default='tcw')
parser.add_argument('--Y_name', type=str, default='T_adv_950')
parser.add_argument('--Z_name', type=str, default='rad')

parser.add_argument('--cause', type=str, default='T_adv_950')
parser.add_argument('--effect', type=str, default='rad')

args=parser.parse_args()

# set seeds
seed=args.seed
random.seed(seed)
np.random.seed(seed)

# list of PCM thresholds
list_pcm_thres = np.arange(0.05, 1.0, 0.05)

# folder to store outputs
save_dir = os.path.join(root, 'outputs', 'ERA5_multiPCM_ratioThres', 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# var_list
X_name=args.X_name
Y_name=args.Y_name
Z_name=args.Z_name
var_list = [X_name, Y_name, Z_name]



# load data
dataset=ERA5MultivarData(csv_path=os.path.join(root, args.data_dir, args.csv),
                         var_list=var_list, 
                         winter_start_month=args.w_start, 
                         winter_end_month=args.w_end, 
                         scaler=args.scaler, 
                         season_rm=args.season_rm)
df=dataset.df

# select a random start point
start_idx=random.randint(0, len(df)-args.L)
df=df[start_idx:start_idx+args.L]

list_cause=[args.cause]
list_effect=[args.effect]
list_conditions=[var for var in var_list if var not in list_cause+list_effect]

# update save_dir based on known cause, effect and conditions
save_dir = os.path.join(save_dir, f'cause_{args.cause}_eff_{args.effect}_conds_{list_conditions}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

pcm=PCM_simplex(df=df, causes=list_cause, effects=list_effect, cond=list_conditions, tau=args.tau, emd=args.emd, L=args.L, knn=args.knn)
output=pcm.causality() # order: sc1_error, sc2_error, ratio_error, sc1_corr, sc2_corr, ratio_corr, sc1_r2, sc2_r2, ratio_r2
del pcm

# (threshold independent) save all outputs first as text
file_save_name=f'L{args.L}__tau{args.tau}_emd{args.emd}_knn{args.knn}'
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

elif args.score_type=='corr':
    arr_ratio=np.full((len(list_pcm_thres)), output[5])
    mask=(arr_ratio>0.95)&(arr_ratio<1.05)
    arr_label[arr_ratio<list_pcm_thres]=1
    arr_label[mask]=0

elif args.score_type=='r2':
    arr_ratio=np.full((len(list_pcm_thres)), output[8])
    mask=(arr_ratio>0.95)&(arr_ratio<1.05)
    arr_label[arr_ratio<list_pcm_thres]=1
    arr_label[mask]=0

# save the label
arr_combined=np.vstack((list_pcm_thres, arr_label))
np.save(os.path.join(save_dir, file_save_name+'_labels.npy'), arr_combined)

# plot the ratio and labels
plt.plot(list_pcm_thres, arr_label, label='label')
plt.xlabel('PCM threshold')
plt.ylabel('Label')
plt.legend()
plt.savefig(os.path.join(save_dir, file_save_name+'_labels.png'))
plt.close()

