# 3V systems, where direct causal interaction exists between X and Y, hence partial xmap through Z shouldn't be possible

# case 1: chain X -> Y -> Z
# case 2: X -> Y, and Z is independent
# case 3: fork Y <- X -> Z

import os, sys
root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

# save_dir
save_dir=os.path.join(root, 'data_files', 'data', 'gen', '3V_direct')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

import random
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser("Generate 3V system data")
parser.add_argument('--seed', type=int, default=97, help='random seed, for initial conditions')
parser.add_argument('--L', type=int, default=10000, help='length of time series')
parser.add_argument('--noiseType', type=str, default='lpNoise', help='type of noise. Options: gNoise, lpNoise, or None')
parser.add_argument('--noiseInjectType', type=str, default='add', help='type of noise injection. Options: add, mult, both')
parser.add_argument('--noiseLevel', type=float, default=2e-2, help='noise level')
args=parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)


# case 1: chain X -> Y -> Z
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*Y(t))

# Autonomous dynamics
Rx = 3.7
Ry = 3.78
Rz = 3.72

# Interations/Coupling strength
Axy=0.35
Ayz=0.35

flag_rerun=True # to start running
while(flag_rerun):
    # empty array with known length
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    # flag set to false to prep for update
    flag_rerun=False

    for t in range(1,args.L):
        # 1, get noise terms for each time step
        # if noiseType is None, then no noise is added; else add noise
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise case
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2, update the system
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*Y[t-1])+noiseZ_add

        # 3. check for numerical stability
        if np.any(np.isnan([X[t],Y[t],Z[t]])) or np.any(np.isinf([X[t],Y[t],Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir, f'3V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}_1.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir, '3V_direct_noNoise_1.csv'), index=False)


# case 2: fork X -> Y, and Z is independent
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t))

# Autonomous dynamics
Rx = 3.7
Ry = 3.78
Rz = 3.72

# Interations/Coupling strength
Axy=0.35

flag_rerun=True # to start running

while(flag_rerun):
    # empty array with known length
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    # flag set to false to prep for update
    flag_rerun=False

    for t in range(1,args.L):
        # 1, get noise terms for each time step
        # if noiseType is None, then no noise is added; else add noise
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise case
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2, update the system
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1])+noiseZ_add

        # 3. check for numerical stability
        if np.any(np.isnan([X[t],Y[t],Z[t]])) or np.any(np.isinf([X[t],Y[t],Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir, f'3V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}_2.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir, '3V_direct_noNoise_2.csv'), index=False)

        
# case 3: fork Y <- X -> Z
# X(t+1)=X(t)*(Rx-Rx*X(t))
# Y(t+1)=Y(t)*(Ry-Ry*Y(t)-Axy*X(t))
# Z(t+1)=Z(t)*(Rz-Rz*Z(t)-Ayz*X(t))

# Autonomous dynamics
Rx = 3.7
Ry = 3.78
Rz = 3.72

# Interations/Coupling strength
Axy=0.35
Ayz=0.35

flag_rerun=True # to start running

while(flag_rerun):
    # empty array with known length
    X=np.zeros(args.L)
    Y=np.zeros(args.L)
    Z=np.zeros(args.L)
    # sample initial conditions
    X[0]=np.random.uniform(0.001,1)
    Y[0]=np.random.uniform(0.001,1)
    Z[0]=np.random.uniform(0.001,1)

    # flag set to false to prep for update
    flag_rerun=False

    for t in range(1,args.L):
        # 1, get noise terms for each time step
        # if noiseType is None, then no noise is added; else add noise
        if args.noiseType!=None and args.noiseType.lower()!='none':
            if args.noiseInjectType=='add':
                noiseX_mult=1
                noiseY_mult=1
                noiseZ_mult=1
                if args.noiseType=='gNoise':
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
            elif args.noiseInjectType=='mult':
                noiseX_add=0
                noiseY_add=0
                noiseZ_add=0
                if args.noiseType=='gNoise':
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
            elif args.noiseInjectType=='both':
                if args.noiseType=='gNoise':
                    noiseX_add=np.random.normal(0,args.noiseLevel)
                    noiseY_add=np.random.normal(0,args.noiseLevel)
                    noiseZ_add=np.random.normal(0,args.noiseLevel)
                    noiseX_mult=np.random.normal(1,args.noiseLevel)
                    noiseY_mult=np.random.normal(1,args.noiseLevel)
                    noiseZ_mult=np.random.normal(1,args.noiseLevel)
                elif args.noiseType=='lpNoise':
                    noiseX_add=np.random.laplace(0,args.noiseLevel)
                    noiseY_add=np.random.laplace(0,args.noiseLevel)
                    noiseZ_add=np.random.laplace(0,args.noiseLevel)
                    noiseX_mult=np.random.laplace(1,args.noiseLevel)
                    noiseY_mult=np.random.laplace(1,args.noiseLevel)
                    noiseZ_mult=np.random.laplace(1,args.noiseLevel)
        else: # no noise case
            noiseX_add=0
            noiseY_add=0
            noiseZ_add=0
            noiseX_mult=1
            noiseY_mult=1
            noiseZ_mult=1

        # 2, update the system
        X[t]=X[t-1]*noiseX_mult*(Rx-Rx*X[t-1])+noiseX_add
        Y[t]=Y[t-1]*noiseY_mult*(Ry-Ry*Y[t-1]-Axy*X[t-1])+noiseY_add
        Z[t]=Z[t-1]*noiseZ_mult*(Rz-Rz*Z[t-1]-Ayz*X[t-1])+noiseZ_add

        # 3. check for numerical stability
        if np.any(np.isnan([X[t],Y[t],Z[t]])) or np.any(np.isinf([X[t],Y[t],Z[t]])):
            flag_rerun=True
            break

# save data
data=pd.DataFrame({'X':X, 'Y':Y, 'Z':Z})
if args.noiseType!=None and args.noiseType.lower()!='none':
    data.to_csv(os.path.join(save_dir, f'3V_direct_{args.noiseType}_{args.noiseInjectType}_{args.noiseLevel}_3.csv'), index=False)
else:
    data.to_csv(os.path.join(save_dir, '3V_direct_noNoise_3.csv'), index=False)



