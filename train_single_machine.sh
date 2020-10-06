#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
echo $THIS_DIR
cd $THIS_DIR
cfg=$1
bs=${2:-256}
bs_per_gpu=${3:-64}
num_gpus=$(( bs / bs_per_gpu ))
num_nodes=$(( ( num_gpus - 1 ) / 8 + 1 ))
num_proc_per_nodes=$(( num_gpus < 8 ? num_gpus : 8 ))
echo $bs
echo $bs_per_gpu
echo $num_gpus
echo $num_nodes
echo $num_proc_per_nodes
if [ ! -f $cfg ]; then
  echo "Config not found!"
fi
##### move data
mkdir -p data
SRC_DIR=/path/to/datasets/imagenet
SRC_DIR1=/path/to/datasets/ILSVRC2012/tars
#cp -r $SRC_DIR data/
#cat imagenet_classes | parallel -j20 rsync -a $SRC_DIR/train/{} data/train
for f in `cat imagenet_classes`; do
  mkdir -p data/train/$f
done
mkdir -p data/val
cat imagenet_classes | parallel -j10 tar -xf $SRC_DIR1/train/{}.tar -C data/train/{}/
cat imagenet_classes | parallel -j20 rsync -a $SRC_DIR/val/{} data/val
echo $METIS_WORKER_0_HOST
echo $METIS_WORKER_0_PORT
echo $METIS_TASK_INDEX
RANK=0 python3 -m torch.distributed.launch --nproc_per_node=$num_proc_per_nodes train.py app:$cfg bs:$bs
