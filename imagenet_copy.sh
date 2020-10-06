#!/bin/bash
#mkdir -p data
SRC_DIR=/path/to/datasets/imagenet
SRC_DIR1=/path/to/datasets/ILSVRC2012/tars
#cp -r $SRC_DIR data/
#cat imagenet_classes | parallel -j20 rsync -a $SRC_DIR/train/{} data/train
TGT_DIR=/path/to/datasets/imagenet
mkdir -p $TGT_DIR
for f in `cat imagenet_classes`; do
  mkdir -p $TGT_DIR/train/$f
done
mkdir -p $TGT_DIR/val
cat imagenet_classes | parallel -j10 tar -xf $SRC_DIR1/train/{}.tar -C $TGT_DIR/train/{}/
cat imagenet_classes | parallel -j20 rsync -a $SRC_DIR/val/{} $TGT_DIR/val
echo $METIS_WORKER_0_HOST
echo $METIS_WORKER_0_PORT
echo $METIS_TASK_INDEX
python3 -m torch.distributed.launch --nproc_per_node=$num_proc_per_nodes --nnodes=$num_nodes --node_rank=$METIS_TASK_INDEX --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT train.py app:$cfg bs:$bs
