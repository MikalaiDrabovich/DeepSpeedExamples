#! /bin/bash

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export PYTHONWARNINGS="ignore"
export HSA_FORCE_FINE_GRAIN_PCIE=1

STATIC_LOSS_SCALE=32768
#export NCCL_ALGO=Ring #Tree,Ring,Collnet
#export NCCL_TREE_THRESHOLD=0
#export NCCL_DEBUG=INFO

: '
Name			Params		Layers	Hidden_size	Heads	Seq
GPT				0.110B		12		768			12		512
GPT2-medium		0.345B		24		1024		16		1024
GPT2-large		0.774B		36		1280		20		1024
GPT2-xl-c		1.542B		48		1584		24		1024
GPT2-xl			1.542B		48		1600		25		1024
GPT2-8B-c		8.000B		72		3072		24		1024
GPT2-8B			8.000B		72		3072		25		1024
Megatron-​LM		8.300B		72		3072		32		1024
Turing-​NLG		17.00B		78		4256		28		1024
GPT3			175.0B		96		12288		96		2048
'

for NUM_LAYERS in 14 15 
do
	for BS in 1 
	do 
			python -m torch.distributed.launch $DISTRIBUTED_ARGS \
					pretrain_gpt2.py \
					--exit-interval 5 \
					--log-interval 1 \
					--train-iters 10 \
					--loss-scale ${STATIC_LOSS_SCALE} \
					--batch-size ${BS} \
					--num-layers ${NUM_LAYERS} \
					--hidden-size 1584 \
					--num-attention-heads 24 \
					--seq-length 1024 \
					--max-position-embeddings 1024 \
					--train-data wikipedia \
					--lazy-loader \
					--tokenizer-type GPT2BPETokenizer \
					--cache-dir cache \
					--split 949,50,1 \
					--distributed-backend nccl \
					--lr 0.00015 \
					--lr-decay-style cosine \
					--weight-decay 1e-2 \
					--clip-grad 1.0 \
					--warmup .01 \
					--fp16

			echo "NUM LAYERS:${NUM_LAYERS}; BATCH SIZE:${BS}"
	done
done

set +x
