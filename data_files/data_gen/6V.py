import os, sys
root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

# save_dir
save_dir=os.path.join(root, 'data_files', 'data', 'gen')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_direct=os.path.join(save_dir, '6V_direct')
if not os.path.exists(save_dir_direct):
    os.makedirs(save_dir_direct)
save_dir_indirect=os.path.join(save_dir, '6V_indirect')
if not os.path.exists(save_dir_indirect):
    os.makedirs(save_dir_indirect)
save_dir_both_noCycle=os.path.join(save_dir, '6V_both_noCycle')
if not os.path.exists(save_dir_both_noCycle):
    os.makedirs(save_dir_both_noCycle)
save_dir_both_Cycle=os.path.join(save_dir, '6V_both_Cycle')
if not os.path.exists(save_dir_both_Cycle):
    os.makedirs(save_dir_both_Cycle)

import random
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser("Generate 6V system data")
parser.add_argument('--seed', type=int, default=97, help='random seed, for initial conditions')
parser.add_argument('--L', type=int, default=10000, help='length of time series')
parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=5e-3, help='noise level')
args=parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# 6V_direct: U->V->W->X->Y->Z
# U(t+1)=U(t)*(Ru-Ru*U(t))
# V(t+1)=V(t)*(Rv-Rv*V(t)-Auv*U(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Avw*V(t))
# X(t+1)=X(t)*(Rx-Rx*X(t)-Awx*W(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Ru=3.7
Rv=3.72
Rw=3.74
Rx=3.76
Ry=3.78
Rz=3.73

# Interations/Coupling strength
Auv=0.35
Avw=0.35
Awx=0.35
Axy=0.35
Ayz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    U[0]=np.random.uniform(0.001,1)
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
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Auv*U[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Awx*W[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_direct, f'6V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_direct, '6V_direct_noNoise.csv'), index=False)

# 6V_indirect: U → V → Z → Y and X → W → Z → Y
# U(t+1)=U(t)*(Ru-Ru*U(t))
# V(t+1)=V(t)*(Rv-Rv*V(t)-Auv*U(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Awx*X(t))
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Ayz*Z(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Azv*V(t)-Azw*W(t))

# Autonomous dynamics
Ru=3.7
Rv=3.72
Rw=3.74
Rx=3.76
Ry=3.78
Rz=3.73

# Interations/Coupling strength
Auv=0.35
Awx=0.35
Ayz=0.35
Azv=0.35
Azw=0.35

flag_rerun=True # to start running
while(flag_rerun):
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    U[0]=np.random.uniform(0.001,1)
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
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Auv*U[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Awx*X[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Ayz*Z[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Azv*V[t-1]-Azw*W[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_indirect, f'6V_indirect_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_indirect, '6V_indirect_noNoise.csv'), index=False)


# 6V_both_noCycle: X → U → V → W → Y and X → Z and X → Y
# U(t+1)=U(t)*(Ru-Ru*U(t)-Axu*X(t))
# V(t+1)=V(t)*(Rv-Rv*V(t)-Auv*U(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Avw*V(t))
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t)-Awy*W(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Axz*X(t))

# Autonomous dynamics
Ru=3.7
Rv=3.72
Rw=3.74
Rx=3.76
Ry=3.78
Rz=3.73

# Interations/Coupling strength
Axu=0.35
Auv=0.35
Avw=0.35
Axy=0.35
Awy=0.35
Axz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    U[0]=np.random.uniform(0.001,1)
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
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1]-Axu*X[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Auv*U[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1]-Awy*W[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Axz*X[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_both_noCycle, f'6V_both_noCycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_both_noCycle, '6V_both_noCycle_noNoise.csv'), index=False)


# 6V_both_Cycle: U → X → Y → W → U and V → Z and U → X → Y → Z
# U(t+1)=U(t)*(Ru-Ru*U(t)-Awu*W(t))
# V(t+1)=V(t)*(Rv-Rv*V(t))
# W(t+1)=W(t)*(Rw-Rw*W(t)-Ayw*Y(t))
# X(t+1)=X(t)*(Rx-Rx*X(t)-Aux*U(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Avz*V(t)-Ayz*Y(t))

# Autonomous dynamics
Ru=3.7
Rv=3.72
Rw=3.74
Rx=3.76
Ry=3.78
Rz=3.73

# Interations/Coupling strength
Awu=0.35
Ayw=0.35
Aux=0.35
Axy=0.35
Avz=0.35
Ayz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    U[0]=np.random.uniform(0.001,1)
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
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1]-Awu*W[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Ayw*Y[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Aux*U[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Avz*V[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_both_Cycle, f'6V_both_Cycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_both_Cycle, '6V_both_Cycle_noNoise.csv'), index=False)




