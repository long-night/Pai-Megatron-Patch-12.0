#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# ============ 环境配置 ============
ENV=dsw                          # $1  运行环境: dsw(单机多卡), dlc(多机多卡)
MODEL_SIZE=0.6B                   # $2  模型规格: 0.6B, 1.7B, 4B, 8B, 14B, 32B, A3B, A22B (A3B A22B是MoE模型)
# ============ Batch Size 配置 ============
BATCH_SIZE=1                     # $3  单个数据并行内的样本数 (micro batch size)
GLOBAL_BATCH_SIZE=1            # $4  所有数据并行的总样本数 (global batch size)
# ============ 训练超参 ============
LR=1e-5                          # $5  学习率
MIN_LR=1e-6                      # $6  最小学习率
SEQ_LEN=128                      # $7  序列长度
PAD_LEN=128                      # $8  Padding 长度
PR=bf16                          # $9  训练精度: fp32,fp16, bf16, fp8
# ============ 并行策略 ============
TP=1                             # $10 Tensor Parallelism (模型并行度)
PP=1                             # $11 Pipeline Parallelism (流水并行度)
CP=1                             # $12 Context Parallelism (上下文并行度)
ETP=1                            # $13 Expert Tensor Parallelism (专家张量并行度)
EP=1                             # $14 Expert Parallelism (专家模型并行度)
SP=true                          # $15 Sequence Parallelism (序列并行): true, false
# ============ 优化策略 ============
DO=false                         # $16 Distributed Optimizer (Zero-1 降显存): true, false
FL=false                          # $17 Flash Attention: true, false
AC=sel                           # $19 Activation Checkpointing: sel, full, offload, false
OPTIMIZER_OFFLOAD=false          # $20 Optimizer Offload: false, 或 0~1 小数表示 offload 比例
# ============ 训练模式 ============
SFT=false                        # $18 是否微调训练: true(SFT), false(预训练)
# ============ 训练进度 ============
TRAIN_TOKENS_OR_ITERS=25600     # $25 训练 Token 数或迭代数, 预训是Tokens，SFT是迭代数 下同
WARMUP_TOKENS_OR_ITERS=100      # $26 预热 Token 数或迭代数
SAVE_INTERVAL=100000              # $21 保存 checkpoint 间隔 (迭代)
# ============ 数据路径 ============
DATASET_PATH=/mnt/d/ubuntu/datasets/pretrain/qwen3-datasets/mmap_qwen3_datasets_text_document       # $22 训练数据集路径
VALID_DATASET_PATH=/mnt/d/ubuntu/datasets/pretrain/qwen3-datasets/mmap_qwen3_datasets_text_document # $23 验证数据集路径
# ============ 模型路径 ============
PRETRAIN_CHECKPOINT_PATH=none         # $24 预训练模型路径
CONFIG_PATH=/mnt/d/ubuntu/models/Qwen3-0.6B         # $28 预训练模型路径
# ============ 输出路径 ============
OUTPUT_BASEPATH=/mnt/d/ubuntu/logs/output_mcore_qwen3_pretrain                   # $27 训练输出日志文件路径
# ============ 执行训练 ============
sh run_mcore_qwen3.sh \
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
