#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $THIS_DIR
cd $THIS_DIR
cfg=$1
if [ ! -f $cfg ]; then
  echo "Config not found!"
fi

gpu_free=($(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader))
IFS=$'\n' gpu_free_sorted=($(sort -r -g <<<"${gpu_free[*]}"))
unset IFS
for i in "${!gpu_free[@]}"; do
    if [ "${gpu_free[$i]}" == "${gpu_free_sorted[0]}" ] && [ -z "${gpu_opt}" ]; then
        gpu_opt=$i
    fi
done
for i in "${!gpu_free[@]}"; do
    if [ "${gpu_free[$i]}" == "${gpu_free_sorted[1]}" ] && [ $i != ${gpu_opt} ]; then
        gpu_second_opt=$i
    fi
done
echo "GPU ${gpu_opt} has the largest free memory (${gpu_free[${gpu_opt}]}MiB)."
export CUDA_VISIBLE_DEVICES=$gpu_opt

bash -c "exec -a python3_cifar10_resnet20_432 python3 train.py app:$cfg"
