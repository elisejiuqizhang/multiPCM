# additional tests - 5V systems
# X-Y causality may be direct, indirect, both (both_Cycle, both_noCycle)

import os, sys
root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

# save_dir
save_dir=os.path.join(root, 'data_files', 'data', 'gen')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_direct=os.path.join(save_dir, '5V_direct')
if not os.path.exists(save_dir_direct):
    os.makedirs(save_dir_direct)
save_dir_indirect=os.path.join(save_dir, '5V_indirect')
if not os.path.exists(save_dir_indirect):
    os.makedirs(save_dir_indirect)
save_dir_both_noCycle=os.path.join(save_dir, '5V_both_noCycle')
if not os.path.exists(save_dir_both_noCycle):
    os.makedirs(save_dir_both_noCycle)
save_dir_both_Cycle=os.path.join(save_dir, '5V_both_Cycle')
if not os.path.exists(save_dir_both_Cycle):
    os.makedirs(save_dir_both_Cycle)


import random
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser("Generate 5V system data")
parser.add_argument('--seed', type=int, default=97, help='random seed, for initial conditions')
parser.add_argument('--L', type=int, default=10000, help='length of time series')
parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=5e-3, help='noise level')
args=parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# 5V_direct: V->W->X->Y->Z
# V(t+1)=V(t)*(Rv-Rv*V(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Avw*V(t))
# X(t+1)=X(t)*(Rx-Rx*X(t)-Awx*W(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Rv=3.7
Rw=3.73
Rx=3.76
Ry=3.74
Rz=3.72

# Interations/Coupling strength
Avw=0.35
Awx=0.35
Axy=0.35
Ayz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    V[0]=np.random.uniform(0.001,1)
    W[0]=np.random.uniform(0.001,1)
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    flag_rerun=False

    for t in range(1,args.L):
        # 1. get noise terms for each time step
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Awx*W[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check if any variable is NaN
        if np.any(np.isnan([V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break
            
# save data
data=pd.DataFrame({'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir_direct, f'5V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir_direct, '5V_direct_noNoise.csv'), index=False)

#5V indirect: V->W->Y->Z, X->W->Y->Z
# V(t+1)=V(t)*(Rv-Rv*V(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Avw*V(t)-Axw*X(t))
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Awy*W(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Rv=3.7
Rw=3.73
Rx=3.76
Ry=3.74
Rz=3.72

# Interations/Coupling strength
Avw=0.35
Axw=0.35
Awy=0.35
Ayz=0.35


flag_rerun=True # to start running

while(flag_rerun):
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    V[0]=np.random.uniform(0.001,1)
    W[0]=np.random.uniform(0.001,1)
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    flag_rerun=False

    for t in range(1,args.L):
        # 1. get noise terms for each time step
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1]-Awx*X[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Awy*W[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add


        # 3. check if any variable is NaN
        if np.any(np.isnan([V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break


# save data
data=pd.DataFrame({'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir_indirect, f'5V_indirect_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir_indirect, '5V_indirect_noNoise.csv'), index=False)


#5V_both_noCycle: V->W->Z->Y, X->W->Z->Y, X->Y
# V(t+1)=V(t)*(Rv-Rv*V(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Avw*V(t)-Axw*X(t))
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t)-Azy*Z(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Awz*W(t))

# Autonomous dynamics
Rv=3.7
Rw=3.73
Rx=3.76
Ry=3.74
Rz=3.72

# Interations/Coupling strength
Avw=0.35
Axw=0.35
Axy=0.35
Azy=0.35
Awz=0.35

flag_rerun=True # to start running

while(flag_rerun):
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    V[0]=np.random.uniform(0.001,1)
    W[0]=np.random.uniform(0.001,1)
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    flag_rerun=False

    for t in range(1,args.L):
        # 1. get noise terms for each time step
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1]-Axw*X[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1]-Azy*Z[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Awz*W[t-1])+noiseZ_add

        # 3. check if any variable is NaN
        if np.any(np.isnan([V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir_both_noCycle, f'5V_both_noCycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir_both_noCycle, '5V_both_noCycle_noNoise.csv'), index=False)


#5V_both_Cycle: X->V->Y->Z->W->V, X->Y
# V(t+1)=V(t)*(Rv-Rv*V(t)-Axv*X(t)-Awv*W(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Azw*Z(t))
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t)-Avy*V(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Rv=3.7
Rw=3.73
Rx=3.76
Ry=3.74
Rz=3.72

# Interations/Coupling strength
Axv=0.35
Awv=0.35
Azw=0.35
Axy=0.35
Avy=0.35
Ayz=0.35

flag_rerun=True # to start running

while(flag_rerun):
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    V[0]=np.random.uniform(0.001,1)
    W[0]=np.random.uniform(0.001,1)
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    flag_rerun=False

    for t in range(1,args.L):
        # 1. get noise terms for each time step
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Axv*X[t-1]-Awv*W[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Azw*Z[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1]-Avy*V[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check if any variable is NaN
        if np.any(np.isnan([V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir_both_Cycle, f'5V_both_Cycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir_both_Cycle, '5V_both_Cycle_noNoise.csv'), index=False)
