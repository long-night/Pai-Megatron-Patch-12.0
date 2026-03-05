#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
set -ex
# ============ 环境配置 ============
ENV=dsw                          # $1  运行环境: dsw(单机多卡), dlc(多机多卡)

# ============ 模型配置 ============
MODEL_SIZE=110B                  # $3  模型规格: 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, A2.7B (A2.7B是MoE模型)

# ============ Batch Size 配置 ============
MICRO_BATCH_SIZE=1                     # $4  单卡 micro batch size
GLOBAL_BATCH_SIZE=1              # $5  全局 batch size (所有数据并行的总样本数)

# ============ 训练超参 ============
LR=1e-5                          # $6  学习率
MIN_LR=1e-6                      # $7  最小学习率
SEQ_LEN=2048                     # $8  序列长度
PAD_LEN=2048                     # $9  最大 Padding 长度
EXTRA_VOCAB_SIZE=421             # $10 额外词表大小: 小于14B用293, 14B及以上用421

# ============ 训练精度 ============
PR=bf16                          # $11 训练精度: fp16, bf16, fp8

# ============ 并行策略 ============
TP=8                             # $12 Tensor Parallelism (张量并行度)
PP=1                             # $13 Pipeline Parallelism (流水线并行度)

# ============ 优化策略 ============
AC=sel                           # $14 Activation Checkpointing: full, sel, none
DO=false                         # $15 Distributed Optimizer (Zero-1): true, false
FL=false                          # $16 Flash Attention: true, false
SP=false                          # $17 Sequence Parallelism (序列并行): true, false (需TP>1时生效)
TE=false                         # $18 Transformer Engine: true, false
MOE=false                        # $19 MoE 模式: true, false (A2.7B需设为true)

# ============ 训练进度 ============
SAVE_INTERVAL=100000               # $20 保存 checkpoint 间隔 (迭代数)
TRAIN_ITERS=200000
WARMUP_ITERS=1000
TRAIN_TOKENS=$(( $TRAIN_ITERS * $SEQ_LEN * $GLOBAL_BATCH_SIZE ))        # $23 总训练 Token 数
WARMUP_TOKENS=$(( $WARMUP_ITERS * $SEQ_LEN * $GLOBAL_BATCH_SIZE ))          # $24 预热 Token 数

# ============ 数据路径 ============
DATASET_PATH=/mnt/qwen-datasets/qwen1_5_data/wudao_qwenbpe_text_document   # $21 训练数据集路径 (mmap格式)

# ============ 模型路径 ============
CONF_FILE_PATH=/mnt/qwen-ckpts/Qwen1.5-110B
PRETRAIN_CHECKPOINT_PATH=${CONF_FILE_PATH}    # $22 预训练模型路径, 从头训练设为 none

# ============ 输出路径 ============
OUTPUT_BASEPATH=/root/log  # $25 训练输出路径 (checkpoint/tensorboard/log)

# ============ 执行训练 ============
sh run_pretrain_qwen.sh \
    $ENV \
    $MODEL_SIZE \
    $MICRO_BATCH_SIZE \
    $GLOBAL_BATCH_SIZE \
    $LR \
    $MIN_LR \
    $SEQ_LEN \
    $PAD_LEN \
    $PR \
    $TP \
    $PP \
    $AC \
    $DO \
    $FL \
    $SP \
    $TE \
    $MOE \
    $SAVE_INTERVAL \
    $DATASET_PATH \
    $PRETRAIN_CHECKPOINT_PATH \
    $TRAIN_TOKENS \
    $WARMUP_TOKENS \
    $OUTPUT_BASEPATH \
    $CONF_FILE_PATH

