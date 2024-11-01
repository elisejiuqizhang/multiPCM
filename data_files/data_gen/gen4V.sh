cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/multiPCM/data_files/data_gen

# list_noiseType=('gNoise' 'lpNoise')
list_noiseType=('gNoise')
# list_noiseType=('gNoise')

# list_noiseInjectType=('add' 'mult' 'both')
list_noiseInjectType=('add')

# list_noiseLevel=(5e-3 1e-2 2e-2)
# list_noiseLevel=(5e-3 1e-2)
list_noiseLevel=(1e-2)

# list_noiseType=('None')
# list_noiseInjectType=('None')
# list_noiseLevel=(0)

# export
export NOISETYPE="${list_noiseType[*]}"
export NOISEINJECTTYPE="${list_noiseInjectType[*]}"
export NOISELEVEL="${list_noiseLevel[*]}"

# gnu parallel
# parallel -j 18 python 4V_direct.py --seed 97 --noiseType {1} --noiseInjectType {2} --noiseLevel {3} ::: ${NOISETYPE} ::: ${NOISEINJECTTYPE} ::: ${NOISELEVEL}
parallel -j 18 python 4V_indirect.py --seed 97 --noiseType {1} --noiseInjectType {2} --noiseLevel {3} ::: ${NOISETYPE} ::: ${NOISEINJECTTYPE} ::: ${NOISELEVEL}
# parallel -j 18 python 4V_both_Cycle.py --seed 97 --noiseType {1} --noiseInjectType {2} --noiseLevel {3} ::: ${NOISETYPE} ::: ${NOISEINJECTTYPE} ::: ${NOISELEVEL}
# parallel -j 18 python 4V_both_noCycle.py --seed 97 --noiseType {1} --noiseInjectType {2} --noiseLevel {3} ::: ${NOISETYPE} ::: ${NOISEINJECTTYPE} ::: ${NOISELEVEL}