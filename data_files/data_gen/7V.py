import os, sys
root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

# save_dir
save_dir=os.path.join(root, 'data_files', 'data', 'gen')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_direct=os.path.join(save_dir, '7V_direct')
if not os.path.exists(save_dir_direct):
    os.makedirs(save_dir_direct)
save_dir_indirect=os.path.join(save_dir, '7V_indirect')
if not os.path.exists(save_dir_indirect):
    os.makedirs(save_dir_indirect)
save_dir_both_noCycle=os.path.join(save_dir, '7V_both_noCycle')
if not os.path.exists(save_dir_both_noCycle):
    os.makedirs(save_dir_both_noCycle)
save_dir_both_Cycle=os.path.join(save_dir, '7V_both_Cycle')
if not os.path.exists(save_dir_both_Cycle):
    os.makedirs(save_dir_both_Cycle)

import random
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser("Generate 7V system data")
parser.add_argument('--seed', type=int, default=97, help='random seed, for initial conditions')
parser.add_argument('--L', type=int, default=10000, help='length of time series')
parser.add_argument('--noiseType', type=str, default='None', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=5e-3, help='noise level')
args=parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# 7V_direct: T->U->V->W->X->Y->Z
# T(t+1)=T(t)*(Rt-Rt*T(t))
# U(t+1)=U(t)*(Rt-Rt*U(t)-Atu*T(t))
# V(t+1)=V(t)*(Rt-Rt*V(t)-Auv*U(t))
# W(t+1)=W(t)*(Rt-Rt*W(t)-Avw*V(t))
# X(t+1)=X(t)*(Rt-Rt*X(t)-Awx*W(t))
# Y(t+1)=Y(t)*(Rt-Rt*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rt-Rt*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Rt=3.7
Ru=3.72
Rv=3.73
Rw=3.74
Rx=3.76
Ry=3.75
Rz=3.73

# Interations/Coupling strength
Atu=0.35
Auv=0.35
Avw=0.35
Awx=0.35
Axy=0.35
Ayz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    T=np.zeros(args.L)
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    T[0]=np.random.uniform(0.001,1)
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
                noiseT_mult=1
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseT_add=0
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseT_add=0
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseT_mult=1
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        T[t]=T[t-1]*noiseT_mult*(Rt-Rt*T[t-1])+noiseT_add
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1]-Atu*T[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Auv*U[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Awx*W[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'T':T, 'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_direct, f'7V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_direct, '7V_direct_noNoise.csv'), index=False)


# 7V_indirect: T->X->V->W->Y and T->X->Z->Y and U->X and U->W
# T(t+1)=T(t)*(Rt-Rt*T(t))
# U(t+1)=U(t)*(Rt-Rt*U(t))
# V(t+1)=V(t)*(Rt-Rt*V(t)-Axv*X(t))
# W(t+1)=W(t)*(Rt-Rt*W(t)-Avw*V(t)-Auw*U(t))
# X(t+1)=X(t)*(Rt-Rt*X(t)-Atx*T(t)-Aux*U(t))
# Y(t+1)=Y(t)*(Rt-Rt*Y(t)-Awy*W(t)-Azy*Z(t))
# Z(t+1)=Z(t)*(Rt-Rt*Z(t)-Axz*X(t))

# Autonomous dynamics
Rt=3.7
Ru=3.72
Rv=3.73
Rw=3.74
Rx=3.76
Ry=3.75
Rz=3.73

# Interations/Coupling strength
Axv=0.35
Avw=0.35
Auw=0.35
Atx=0.35
Aux=0.35
Awy=0.35
Azy=0.35
Axz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    T=np.zeros(args.L)
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    T[0]=np.random.uniform(0.001,1)
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
                noiseT_mult=1
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseT_add=0
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseT_add=0
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseT_mult=1
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        T[t]=T[t-1]*noiseT_mult*(Rt-Rt*T[t-1])+noiseT_add
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Axv*X[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Avw*V[t-1]-Auw*U[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Atx*T[t-1]-Aux*U[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Awy*W[t-1]-Azy*Z[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Axz*X[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'T':T, 'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_indirect, f'7V_indirect_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_indirect, '7V_indirect_noNoise.csv'), index=False)


# 7V_both_noCycle: T->X->Y and T->U->W->Z and X->U and W->Y and W->V and T->V
# T(t+1)=T(t)*(Rt-Rt*T(t))
# U(t+1)=U(t)*(Rt-Rt*U(t)-Atu*T(t)-Axu*X(t))
# V(t+1)=V(t)*(Rt-Rt*V(t)-Atv*T(t)-Awv*W(t))
# W(t+1)=W(t)*(Rt-Rt*W(t)-Auw*U(t))
# X(t+1)=X(t)*(Rt-Rt*X(t)-Atx*T(t))
# Y(t+1)=Y(t)*(Rt-Rt*Y(t)-Axy*X(t)-Awy*W(t))
# Z(t+1)=Z(t)*(Rt-Rt*Z(t)-Awz*W(t))

# Autonomous dynamics
Rt=3.7
Ru=3.72
Rv=3.73
Rw=3.74
Rx=3.76
Ry=3.75
Rz=3.73

# Interations/Coupling strength
Atu=0.35
Axu=0.35
Atv=0.35
Awv=0.35
Auw=0.35
Atx=0.35
Axy=0.35
Awy=0.35
Awz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    T=np.zeros(args.L)
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    T[0]=np.random.uniform(0.001,1)
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
                noiseT_mult=1
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseT_add=0
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseT_add=0
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseT_mult=1
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        T[t]=T[t-1]*noiseT_mult*(Rt-Rt*T[t-1])+noiseT_add
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1]-Atu*T[t-1]-Axu*X[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Atv*T[t-1]-Awv*W[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Auw*U[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Atx*T[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1]-Awy*W[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Awz*W[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'T':T, 'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_both_noCycle, f'7V_both_noCycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_both_noCycle, '7V_both_noCycle_noNoise.csv'), index=False)


# 7V_both_Cycle: T->U->V->Y->X->U and T->Z->Y and T->W
# T(t+1)=T(t)*(Rt-Rt*T(t))
# U(t+1)=U(t)*(Rt-Rt*U(t)-Atu*T(t)-Axu*X(t))
# V(t+1)=V(t)*(Rt-Rt*V(t)-Auv*U(t))
# W(t+1)=W(t)*(Rt-Rt*W(t)-Atw*T(t))
# X(t+1)=X(t)*(Rt-Rt*X(t)-Ayx*Y(t))
# Y(t+1)=Y(t)*(Rt-Rt*Y(t)-Azy*Z(t)-Avy*V(t))
# Z(t+1)=Z(t)*(Rt-Rt*Z(t)-Atz*T(t))

# Autonomous dynamics
Rt=3.7
Ru=3.72
Rv=3.73
Rw=3.74
Rx=3.76
Ry=3.75
Rz=3.73

# Interations/Coupling strength
Atu=0.35
Axu=0.35
Auv=0.35
Atw=0.35
Ayx=0.35
Azy=0.35
Avy=0.35
Atz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    T=np.zeros(args.L)
    U=np.zeros(args.L)
    V=np.zeros(args.L)
    W=np.zeros(args.L)
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    T[0]=np.random.uniform(0.001,1)
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
                noiseT_mult=1
                noiseU_mult=1
                noiseV_mult=1
                noiseW_mult=1
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseT_add=0
                noiseU_add=0
                noiseV_add=0
                noiseW_add=0
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseT_add=np.random.normal(0,args.noiseLevel)
                    noiseU_add=np.random.normal(0,args.noiseLevel)
                    noiseV_add=np.random.normal(0,args.noiseLevel)
                    noiseW_add=np.random.normal(0,args.noiseLevel)
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseT_mult=np.random.normal(1,args.noiseLevel)
                    noiseU_mult=np.random.normal(1,args.noiseLevel)
                    noiseV_mult=np.random.normal(1,args.noiseLevel)
                    noiseW_mult=np.random.normal(1,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseT_add=np.random.laplace(0,args.noiseLevel)
                    noiseU_add=np.random.laplace(0,args.noiseLevel)
                    noiseV_add=np.random.laplace(0,args.noiseLevel)
                    noiseW_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseT_mult=np.random.laplace(1,args.noiseLevel)
                    noiseU_mult=np.random.laplace(1,args.noiseLevel)
                    noiseV_mult=np.random.laplace(1,args.noiseLevel)
                    noiseW_mult=np.random.laplace(1,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise
            noiseT_add=0
            noiseU_add=0
            noiseV_add=0
            noiseW_add=0
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseT_mult=1
            noiseU_mult=1
            noiseV_mult=1
            noiseW_mult=1
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2. update system
        T[t]=T[t-1]*noiseT_mult*(Rt-Rt*T[t-1])+noiseT_add
        U[t]=U[t-1]*noiseU_mult*(Ru-Ru*U[t-1]-Atu*T[t-1]-Axu*X[t-1])+noiseU_add
        V[t]=V[t-1]*noiseV_mult*(Rv-Rv*V[t-1]-Auv*U[t-1])+noiseV_add
        W[t]=W[t-1]*noiseW_mult*(Rw-Rw*W[t-1]-Atw*T[t-1])+noiseW_add
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1]-Ayx*Y[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Azy*Z[t-1]-Avy*V[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Atz*T[t-1])+noiseZ_add

        # 3. check if any variable is out of bound
        if np.any(np.isnan([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])) or np.any(np.isinf([T[t], U[t], V[t], W[t], X[t], Y[t], Z[t]])):
            flag_rerun=True
            break

# save data
df=pd.DataFrame({'T':T, 'U':U, 'V':V, 'W':W, 'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    df.to_csv(os.path.join(save_dir_both_Cycle, f'7V_both_Cycle_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}.csv'), index=False)
else:
    df.to_csv(os.path.join(save_dir_both_Cycle, '7V_both_Cycle_noNoise.csv'), index=False)

