#!/bin/bash

# ==================== 配置区 ====================
# Python 脚本路径
PYTHON_SCRIPT="/mnt/data/xgl/mycode/action_agent/main_gtea.py"

# 运行命令
CMD="python $PYTHON_SCRIPT"

# 输出结果文件（用于判断是否已完成）
OUTPUT_FILE="gtea_val_8B.txt"

# 预期总行数 = CSV行数 + 1（表头）
EXPECTED_LINES=529

# 日志文件
LOG_FILE="gtea_run_log_$(date +%Y%m%d_%H%M%S).txt"

# ==================== 运行循环 ====================

# 记录脚本开始时间（秒）
START_TIME=$(date +%s)

echo "启动自动重跑脚本" | tee "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "目标：$OUTPUT_FILE 达到 $EXPECTED_LINES 行" | tee -a "$LOG_FILE"
echo "命令：$CMD" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

while true; do
    # 获取当前已处理行数（包括表头）
    if [[ -f "$OUTPUT_FILE" ]]; then
        CURRENT_LINES=$(wc -l < "$OUTPUT_FILE")
    else
        CURRENT_LINES=0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 当前进度: $CURRENT_LINES / $EXPECTED_LINES 行" | tee -a "$LOG_FILE"

    # 如果已经完成，退出
    if [[ $CURRENT_LINES -ge $EXPECTED_LINES ]]; then
        # 记录结束时间并计算总耗时（秒）
        END_TIME=$(date +%s)
        TOTAL_SECONDS=$((END_TIME - START_TIME))

        echo "=== 全部完成！总共 $CURRENT_LINES 行 ===" | tee -a "$LOG_FILE"
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
        echo "总耗时: $TOTAL_SECONDS 秒" | tee -a "$LOG_FILE"
        echo "成功！可以查看结果：$OUTPUT_FILE" | tee -a "$LOG_FILE"
        break
    fi

    # 运行一次 Python 脚本
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始运行..." | tee -a "$LOG_FILE"
    $CMD >> "$LOG_FILE" 2>&1

    # 检查退出码
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 139 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到段错误（退出码 139），准备重启..." | tee -a "$LOG_FILE"
    elif [[ $EXIT_CODE -ne 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 程序异常退出（退出码 $EXIT_CODE），准备重启..." | tee -a "$LOG_FILE"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 本轮运行正常结束" | tee -a "$LOG_FILE"
    fi

    sleep 5
done

echo "脚本结束，整个任务已完成！" | tee -a "$LOG_FILE"