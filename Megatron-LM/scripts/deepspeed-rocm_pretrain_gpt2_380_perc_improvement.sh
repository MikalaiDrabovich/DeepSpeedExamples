#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

STATIC_LOSS_SCALE=32768

export PYTHONWARNINGS="ignore"
export HSA_FORCE_FINE_GRAIN_PCIE=1
#export NCCL_ALGO=Ring #Tree,Ring,Collnet
#export NCCL_TREE_THRESHOLD=0
#export NCCL_DEBUG=INFO
#USE_TORCH_DDP = False
#number_checkpoints


config_json="$script_dir/deepspeed-rocm_config_380_perc_improvement.json"
gpt_options=" \
	   --exit-interval 10 \
	   --log-interval 1 \
	   --loss-scale ${STATIC_LOSS_SCALE} \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 60 \
       --hidden-size 1584 \
       --num-attention-heads 24 \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 100000 \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
	   --cache-dir cache \
	   --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --fp16 \
	   --num-workers 2
	   
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="deepspeed.pt --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
