cd /home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/multiPCM/data_files/data_gen

list_noiseType=('gNoise' 'lpNoise')
list_noiseInjectType=('add' 'mult' 'both')
list_noiseLevel=(5e-3 1e-2)
# list_noiseType=('None')
# list_noiseInjectType=('None')
# list_noiseLevel=(0)

# export
export NOISETYPE="${list_noiseType[*]}"
export NOISEINJECTTYPE="${list_noiseInjectType[*]}"
export NOISELEVEL="${list_noiseLevel[*]}"

# gnu parallel
parallel -j 16 python 8V.py --seed 97 --noiseType {1} --noiseInjectType {2} --noiseLevel {3} ::: ${NOISETYPE} ::: ${NOISEINJECTTYPE} ::: ${NOISELEVEL}