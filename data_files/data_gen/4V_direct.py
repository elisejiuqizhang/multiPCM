# 4V systems, where X and Y and directly causally connected, the other two vars W and Z are either their parents/children or independent from them, hence partial cross mapping doesn't work.

# case 1: chain W->X->Y->Z
# case 2: W->X, X->Y, X->Z (kinda like a fork)

import os, sys
root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

# save_dir
save_dir=os.path.join(root, 'data_files', 'data', 'gen', '4V_direct')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import random
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser("Generate 3V system data")
parser.add_argument('--seed', type=int, default=97, help='random seed, for initial conditions')
parser.add_argument('--L', type=int, default=10000, help='length of time series')
parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=2e-2, help='noise level')
args=parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# case 1: chain W->X->Y->Z
# W(t+1)=W(t)*(Rw-Rw*W(t))
# X(t+1)=X(t)*(Rx-Rx*X(t)-Awx*W(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Rw=3.7
Rx=3.78
Ry=3.72
Rz=3.7

# Interations/Coupling strength
Awx=0.35
Axy=0.35
Ayz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    W[0]=np.random.uniform(0.001,1)
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    flag_rerun=False

    for t in range(1, args.L):
        # 1. get noise terms for each time step
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise case
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update the system
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Awx*W[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check for numerical stability
        if np.any(np.isnan([W[t],X[t],Y[t],Z[t]])) or np.any(np.isinf([W[t],X[t],Y[t],Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir, f'4V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}_1.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir, '4V_direct_noNoise_1.csv'), index=False)


# case 2: W->X, X->Y, X->Z (kinda like a fork)
# W(t+1)=W(t)*(Rw-Rw*W(t))
# X(t+1)=X(t)*(Rx-Rx*X(t)-Awx*W(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Axz*X(t))

# Autonomous dynamics
Rw=3.7
Rx=3.78
Ry=3.72
Rz=3.7

# Interations/Coupling strength
Awx=0.35
Axy=0.35
Axz=0.35

flag_rerun=True # to start running

while(flag_rerun):
    # empty arrays 
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    W[0]=np.random.uniform(0.001,1)
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    flag_rerun=False

    for t in range(1, args.L):
        # 1. get noise terms for each time step
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise case
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update the system
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Awx*W[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Axz*X[t-1])+noiseZ_add

        # 3. check for numerical stability
        if np.any(np.isnan([W[t],X[t],Y[t],Z[t]])) or np.any(np.isinf([W[t],X[t],Y[t],Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir, f'4V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}_2.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir, '4V_direct_noNoise_2.csv'), index=False)

