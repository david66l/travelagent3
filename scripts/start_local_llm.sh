#!/bin/bash
# 启动本地 llama.cpp 推理服务（Qwen2.5-7B, 端口 8081）
# 用于意图识别等轻量任务，避免消耗 OpenAI API 费用

MODEL_DIR="/Volumes/PS2000/AI项目/面试项目/TravelAgent2/models"

# IQ4_XS 优先（更快），Q4_K_M 兜底
if [ -f "$MODEL_DIR/Qwen2.5-7B-Instruct-IQ4_XS.gguf" ]; then
    MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-IQ4_XS.gguf"
    MODEL_NAME="IQ4_XS"
elif [ -f "$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf" ]; then
    MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    MODEL_NAME="Q4_K_M"
else
    echo "❌ 未找到模型文件"
    echo "   下载 IQ4_XS: curl -L -O https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-IQ4_XS.gguf"
    exit 1
fi

echo "🚀 启动 Qwen2.5-7B ($MODEL_NAME) on port 8081..."
llama-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8081 \
    -c 2048 \
    -ngl 99 \
    -np 1 \
    -t 6 \
    -tb 6 \
    -b 512 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --mlock
