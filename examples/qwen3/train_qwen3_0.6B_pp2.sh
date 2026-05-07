#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1
export GPUS_PER_NODE=2

ENV=dsw
MODEL_SIZE=0.6B
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=128
PAD_LEN=128
PR=bf16
TP=1
PP=2
CP=1
ETP=1
EP=1
SP=false
DO=false
FL=false
AC=sel
OPTIMIZER_OFFLOAD=false
SFT=false
TRAIN_TOKENS_OR_ITERS=640
WARMUP_TOKENS_OR_ITERS=128
SAVE_INTERVAL=100000
DATASET_PATH=/mnt/d/ubuntu/datasets/pretrain/qwen3-datasets/mmap_qwen3_datasets_text_document
VALID_DATASET_PATH=/mnt/d/ubuntu/datasets/pretrain/qwen3-datasets/mmap_qwen3_datasets_text_document
PRETRAIN_CHECKPOINT_PATH=none
OUTPUT_BASEPATH=/mnt/d/ubuntu/logs/output_mcore_qwen3_pretrain_pp2
CONFIG_PATH=/mnt/d/ubuntu/models/Qwen3-0.6B

sh run_mcore_qwen3_parallel.sh \
    $ENV \
    $MODEL_SIZE \
    $BATCH_SIZE \
    $GLOBAL_BATCH_SIZE \
    $LR \
    $MIN_LR \
    $SEQ_LEN \
    $PAD_LEN \
    $PR \
    $TP \
    $PP \
    $CP \
    $ETP \
    $EP \
    $SP \
    $DO \
    $FL \
    $SFT \
    $AC \
    $OPTIMIZER_OFFLOAD \
    $SAVE_INTERVAL \
    $DATASET_PATH \
    $VALID_DATASET_PATH \
    $PRETRAIN_CHECKPOINT_PATH \
    $TRAIN_TOKENS_OR_ITERS \
    $WARMUP_TOKENS_OR_ITERS \
    $OUTPUT_BASEPATH \
    $CONFIG_PATH
