#!/bin/bash
#SBATCH --job-name=bigcode-scaling
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/leandro/logs/bigcode-scaling/bcs-%x-%j.out

set -x -e

source /admin/home/leandro/.bashrc

conda activate megatron

echo "START TIME: $(date)"

SCRIPT_REPO=/fsx/leandro/git/Megatron-LM-BC

pushd $SCRIPT_REPO

LOG_PATH=$SCRIPT_REPO/main_log.txt

# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# File path setup
CHECKPOINT_PATH=/fsx/leandro/experiments/bigcode-scaling/$SLURM_JOB_ID  # Adjust: Directory to store the checkpoints
WEIGHTS_TRAIN=/fsx/loubna/code/bigcode-data-mix/data/train_data_paths.txt.tmp
WEIGHTS_VALID=/fsx/loubna/code/bigcode-data-mix/data/valid_data_paths.txt.tmp
TOKENIZER_FILE=/fsx/leandro/data/tokenizers/tokenizer-the-stack-march-sample-v3-no-prefix-spaces/tokenizer.json # Adjust

mkdir $CHECKPOINT_PATH

GPT_ARGS="\
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 4 \
       --sequence-parallel \
       --num-layers 40 \
       --hidden-size 6144 \
       --num-attention-heads 48 \
       --attention-head-type multiquery \
       --init-method-std 0.01275 \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --micro-batch-size 1 \
       --global-batch-size 512 \
       --lr 0.0003 \
       --min-lr 0.00003 \
       --train-iters 250000 \
       --lr-decay-iters 250000 \
       --lr-decay-style cosine \
       --lr-warmup-iters 2000 \
       --weight-decay .1 \
       --adam-beta2 .95 \
       --clip-grad 1.0 \
       --bf16 \
       --use-flash-attn \
       --fim-rate 0.5 \
       --log-interval 10 \
       --save-interval 2500 \
       --eval-interval 2500 \
       --eval-iters 5 \
       --use-distributed-optimizer \
       --num-layers-per-virtual-pipeline-stage 5 \
"

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"

CMD=" \
    pretrain_gpt.py \
    $GPT_ARGS \
    --tokenizer-type TokenizerFromFile \
    --tokenizer-file $TOKENIZER_FILE \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --train-weighted-split-paths-path $WEIGHTS_TRAIN \
    --valid-weighted-split-paths-path $WEIGHTS_VALID \
    $TENSORBOARD_ARGS \
    --wandb-entity-name lvwerra \
    --wandb-project-name bigcode-scaling \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

echo $CMD

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

export PATH="/usr/local/cuda-11.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD" 2>&1 | tee $LOG_PATH

rm -rf $CHECKPOINT_PATH

echo "END TIME: $(date)"
