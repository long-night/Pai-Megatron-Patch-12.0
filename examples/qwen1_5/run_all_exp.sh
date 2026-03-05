#!/bin/bash
set -e

# ============ 参数配置 ============
TP_LIST=(8 4)
LAYERS_LIST=(10 8 5 4 2)
PR_LIST=(fp32 bf16)

LOG_DIR=/root/log_prof
RESULT_FILE=/root/log_prof/results.txt

# 创建日志目录
mkdir -p $LOG_DIR

# 清空结果文件，写入表头
echo -e "TP\tLAYERS\tPR\tMaxMemory(GB)" > $RESULT_FILE

exp_id=1

for tp in "${TP_LIST[@]}"; do
    for layers in "${LAYERS_LIST[@]}"; do
        for pr in "${PR_LIST[@]}"; do
            log_file="${LOG_DIR}/exp${exp_id}_tp${tp}_layers${layers}_${pr}.log"
            
            echo "========================================"
            echo "Running Experiment $exp_id: TP=$tp, LAYERS=$layers, PR=$pr"
            echo "Log file: $log_file"
            echo "========================================"
            
            # 执行训练脚本
            ./train_qwen_15_prof.sh $tp $layers $pr > "$log_file" 2>&1 || {
                echo "Warning: Experiment $exp_id failed, check log: $log_file"
            }
            
            # 从日志中提取 max allocated 显存值
            mem_value=$(grep "max allocated:" "$log_file" | tail -1 | sed 's/.*max allocated: \([0-9.]*\).*/\1/' 2>/dev/null || echo "N/A")
            
            # 保留一位小数
            if [[ "$mem_value" != "N/A" && -n "$mem_value" ]]; then
                mem_value=$(printf "%.1f" "$mem_value")
            fi
            
            echo "Max Memory: ${mem_value} GB"
            echo ""
            
            # 写入结果文件
            echo -e "${tp}\t${layers}\t${pr}\t${mem_value}" >> $RESULT_FILE
            
            ((exp_id++))
        done
    done
done

echo "========================================"
echo "All experiments completed! (Total: $((exp_id-1)) experiments)"
echo "Results saved to: $RESULT_FILE"
echo "========================================"
echo ""
cat $RESULT_FILE
