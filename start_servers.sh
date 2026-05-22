#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook)"

echo "=== 启动视频时序定位推理服务 ==="

# 启动主服务（CPL + CPL-MoE，端口 8100）
echo "[1/2] 启动主推理服务（CPL + CPL-MoE，端口 8100，cpl 环境）..."
conda activate cpl
python "${SCRIPT_DIR}/inference_server.py" &
PID_CPL=$!
echo "  PID: ${PID_CPL}"

# 启动 PPS 服务（端口 8200）
echo "[2/2] 启动 PPS 推理服务（端口 8200，pps 环境）..."
conda activate pps
python "${SCRIPT_DIR}/pps_inference_server.py" &
PID_PPS=$!
echo "  PID: ${PID_PPS}"

echo ""
echo "=== 服务已启动 ==="
echo "  主服务（CPL + CPL-MoE）: http://127.0.0.1:8100  PID: ${PID_CPL}"
echo "  PPS 服务:               http://127.0.0.1:8200  PID: ${PID_PPS}"
echo ""
echo "访问 http://127.0.0.1:8100 使用前端页面。"
echo "按 Ctrl+C 停止所有服务。"

# 等待任意子进程退出
wait
