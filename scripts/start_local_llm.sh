#!/bin/bash
# 启动本地 llama.cpp 推理服务（Qwen2.5-7B, 端口 8081）
# 用于意图识别等轻量任务，避免消耗 OpenAI API 费用

MODEL_DIR="/Volumes/PS2000/AI项目/面试项目/TravelAgent2/models"
MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

if [ ! -f "$MODEL" ]; then
    echo "❌ 模型未找到: $MODEL"
    echo "   下载中...请稍后重试"
    exit 1
fi

echo "🚀 启动 Qwen2.5-7B (Q4_K_M) on port 8081..."
llama-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8081 \
    -c 4096 \
    -ngl 99 \
    --chat-template qwen \
    --threads 8
