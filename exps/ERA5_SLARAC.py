import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root)

import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz

import time # to track runtime

from data_loaders.ERA5.multivar import ERA5MultivarData
from utils import tidybench

# visualize adjacency matrix
import networkx as nx
# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

import argparse
parser=argparse.ArgumentParser("Single experiment of causal learning baselines on ERA5 data")

parser.add_argument('--data_dir', type=str, default='data_files/data/ERA5')
parser.add_argument('--csv', type=str, default='Timeseries_Upstream.csv')

parser.add_argument('--w_start', type=int, default=12, help="Winter start month")
parser.add_argument('--w_end', type=int, default=2, help="Winter end month")
parser.add_argument('--scaler', type=str, default='std', help="Options are None, minmax/mm, standard/std")
parser.add_argument('--season_rm', type=str, default=None, help="Options are None, daily/d, weekly/w, monthly/m")

parser.add_argument('--seed', type=int, default=97, help='random seed, for sampling a random start point for input time series')

parser.add_argument('--L', type=int, default=6000, help='length of input time series')

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
save_dir = os.path.join(root, 'outputs', 'ERA5_slarac', 'seed'+str(seed))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# var_list
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

data=df.values

max_lag=4

sc_matrix = tidybench.slarac(data, maxlags=max_lag, post_standardise=True).round(2)


# print(sc_matrix)
np.save(os.path.join(save_dir,file_name+'_sc_matrix.npy'), sc_matrix)

# visualize the matrix
G=nx.DiGraph()
G.add_nodes_from(var_names)
for i in range(n_vars):
    for j in range(n_vars):
        if i!=j:
            # G.add_edge(var_names[i], var_names[j], weight=sc_matrix[i,j])
            if abs(sc_matrix[i,j])>0.5:
                G.add_edge(var_names[i],var_names[j],weight=sc_matrix[i,j])

nx.draw(G, with_labels=True)
plt.savefig(os.path.join(save_dir,file_name+'.png'))
plt.close()

