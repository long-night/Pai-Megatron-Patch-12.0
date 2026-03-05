#!/bin/bash
set -ex

# ============ 命令行参数 ============
TP=${1:-8}                       # 第1个参数: Tensor Parallelism
LAYERS=${2:-10}                  # 第2个参数: 层数
PR=${3:-bf16}                    # 第3个参数: 精度 fp32, bf16, fp8

# ============ 环境配置 ============
ENV=dsw                          # 运行环境: dsw(单机多卡), dlc(多机多卡)

# ============ 模型配置 ============
MODEL_SIZE=110B                  # 模型规格

# ============ Batch Size 配置 ============
MICRO_BATCH_SIZE=1               # 单卡 micro batch size
GLOBAL_BATCH_SIZE=1              # 全局 batch size

# ============ 训练超参 ============
LR=1e-5                          # 学习率
MIN_LR=1e-6                      # 最小学习率
SEQ_LEN=2048                     # 序列长度
PAD_LEN=2048                     # 最大 Padding 长度
EXTRA_VOCAB_SIZE=421             # 额外词表大小

# ============ 并行策略 ============
# 根据 TP 设置 CUDA_VISIBLE_DEVICES
if [ "$TP" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ "$TP" -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

PP=1                             # Pipeline Parallelism

# ============ 优化策略 ============
AC=sel                           # Activation Checkpointing: full, sel, none
DO=false                         # Distributed Optimizer (Zero-1)
FL=false                         # Flash Attention
SP=false                         # Sequence Parallelism
TE=false                         # Transformer Engine
MOE=false                        # MoE 模式

# ============ 训练进度 ============
SAVE_INTERVAL=100000             # 保存 checkpoint 间隔
TRAIN_ITERS=200000
WARMUP_ITERS=1000
TRAIN_TOKENS=$(( $TRAIN_ITERS * $SEQ_LEN * $GLOBAL_BATCH_SIZE ))
WARMUP_TOKENS=$(( $WARMUP_ITERS * $SEQ_LEN * $GLOBAL_BATCH_SIZE ))

# ============ 数据路径 ============
DATASET_PATH=/mnt/qwen-datasets/qwen1_5_data/wudao_qwenbpe_text_document

# ============ 模型路径 ============
CONF_FILE_PATH=/mnt/qwen-ckpts/Qwen1.5-110B
PRETRAIN_CHECKPOINT_PATH=${CONF_FILE_PATH}

# ============ 输出路径 ============
# OUTPUT_BASEPATH=/root/log/prof
OUTPUT_BASEPATH=/root/tmp/log/prof

# ============ 执行训练 ============
sh run_pretrain_qwen_prof.sh \
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
    $CONF_FILE_PATH \
    $LAYERS