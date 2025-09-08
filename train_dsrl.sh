#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL=egl
export WANDB_BASE_URL=https://api.bandw.top

# 用法示例：
#   顺序跑： ./run_dsrl.sh lift can
#   并行跑（GPU0,1）： ./run_dsrl.sh --parallel --gpus 0,1 lift can square transport
#   仅单任务： ./run_dsrl.sh lift

# ========== 可配置路径 ==========
BASE_DIR="/mnt/ssd1/data/wys/dppo/log"

# ========== 解析参数 ==========
PARALLEL=0
GPUS=("0")     # 默认只有一张卡

ARGS=()
while (( "$#" )); do
  case "$1" in
    --parallel)
      PARALLEL=1; shift ;;
    --gpus)
      IFS=',' read -r -a GPUS <<< "$2"; shift 2 ;;
    -h|--help)
      echo "用法: $0 [--parallel] [--gpus 0,1,2] {lift|can|square|transport}..."
      exit 0 ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

if [ ${#ARGS[@]} -eq 0 ]; then
  echo "❗ 请至少指定一个任务：{lift|can|square|transport}"
  exit 1
fi

# mkdir -p "$LOG_DIR"

# ========== 每个任务的路径选择 ==========
resolve_paths() {
  local task="$1"
  case "$task" in
    lift)
      BASE_POLICY_PATH="$BASE_DIR/robomimic-pretrain/lift/lift_pre_diffusion_mlp_ta4_td20/2024-06-28_14-47-58/checkpoint/state_5000.pt"
      NORMALIZATION_PATH="$BASE_DIR/robomimic/lift/normalization.npz"
      ;;
    can)
      BASE_POLICY_PATH="$BASE_DIR/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt"
      NORMALIZATION_PATH="$BASE_DIR/robomimic/can/normalization.npz"
      ;;
    square)
      BASE_POLICY_PATH="$BASE_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_ta4_td20/2024-06-28_14-49-03/checkpoint/state_5000.pt"
      NORMALIZATION_PATH="$BASE_DIR/robomimic/square/normalization.npz"
      ;;
    transport)
      BASE_POLICY_PATH="$BASE_DIR/robomimic-pretrain/transport/transport_pre_diffusion_mlp_ta4_td20/2024-06-28_14-50-37/checkpoint/state_5000.pt"
      NORMALIZATION_PATH="$BASE_DIR/robomimic/transport/normalization.npz"
      ;;
    *)
      echo "未知任务: $task （支持 lift|can|square|transport）"
      exit 1 ;;
  esac
}

# ========== 单任务启动函数 ==========
run_task() {
  local task="$1"
  local gpu="$2"

  resolve_paths "$task"

  # 启动前做存在性检查
  if [ ! -f "$BASE_POLICY_PATH" ]; then
    echo "❗ 找不到 base_policy_path: $BASE_POLICY_PATH"
    exit 1
  fi
  if [ ! -f "$NORMALIZATION_PATH" ]; then
    echo "❗ 找不到 normalization_path: $NORMALIZATION_PATH"
    exit 1
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
#   log_file="./logs/robomimic-dsrl/${task}_${ts}.log"

  echo "▶ 任务: $task | GPU: $gpu"
  echo "   base_policy_path: $BASE_POLICY_PATH"
  echo "   normalization_path: $NORMALIZATION_PATH"
#   echo "   日志: $log_file"

  CUDA_VISIBLE_DEVICES="$gpu" \
  python train_dsrl.py \
    --config-path=cfg/robomimic \
    --config-name=dsrl_can.yaml \
    env_name="$task" \
    base_policy_path="$BASE_POLICY_PATH" \
    normalization_path="$NORMALIZATION_PATH" \
    use_wandb=False
}

# ========== 顺序 or 并行调度 ==========
if [ "$PARALLEL" -eq 0 ]; then
  # 顺序运行
  for idx in "${!ARGS[@]}"; do
    task="${ARGS[$idx]}"
    gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
    run_task "$task" "$gpu"
  done
else
  # 并行运行（GPU 轮询分配）
  pids=()
  for idx in "${!ARGS[@]}"; do
    task="${ARGS[$idx]}"
    gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
    run_task "$task" "$gpu" &
    pids+=($!)
    sleep 2  # 避免同时抢显存
  done
  echo "⏳ 等待所有并行任务结束..."
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "✅ 全部任务完成"
fi