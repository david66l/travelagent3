#!/bin/bash
# 启动本地 llama.cpp 推理服务（Qwen2.5-7B, 端口 8081）
# 用于意图识别等轻量任务，避免消耗 OpenAI API 费用

MODEL_DIR="/Volumes/PS2000/AI项目/面试项目/TravelAgent2/models"

# 意图/槽位任务用不上 7B：优先更快的 3B（M2 上约快一倍），7B 作兜底。
if [ -f "$MODEL_DIR/Qwen2.5-3B-Instruct-Q4_K_M.gguf" ]; then
    MODEL="$MODEL_DIR/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    MODEL_NAME="3B-Q4_K_M"
elif [ -f "$MODEL_DIR/Qwen2.5-7B-Instruct-IQ4_XS.gguf" ]; then
    MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-IQ4_XS.gguf"
    MODEL_NAME="7B-IQ4_XS"
elif [ -f "$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf" ]; then
    MODEL="$MODEL_DIR/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    MODEL_NAME="7B-Q4_K_M"
else
    echo "❌ 未找到模型文件"
    echo "   下载 3B: curl -L -O https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    exit 1
fi

echo "🚀 启动 Qwen2.5 ($MODEL_NAME) on port 8081..."
# 调整说明：
#  • 去掉 --cache-type-k/v q8_0：M2 Metal 上量化 KV cache 反而拖慢生成，而 ctx 仅
#    2048、省内存意义不大 → 用默认 f16 cache，token 生成更快。
#  • -t 6 → -t 4：GPU 全量 offload(-ngl 99) 时多线程与 GPU 抢资源；M2 仅 4 性能核。
llama-server \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8081 \
    -c 2048 \
    -ngl 99 \
    -np 1 \
    -t 4 \
    -tb 4 \
    -b 512 \
    --mlock
