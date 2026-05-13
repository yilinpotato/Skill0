#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${1:?Usage: resource_monitor.sh LOG_DIR [PARENT_PID]}"
PARENT_PID="${2:-}"

INTERVAL="${RESOURCE_MONITOR_INTERVAL:-5}"
GPU_WARN_PCT="${GPU_MEMORY_WARN_PCT:-90}"
CPU_WARN_PCT="${CPU_MEMORY_WARN_PCT:-92}"
TRACE_COOLDOWN="${RESOURCE_TRACE_COOLDOWN:-60}"
ENABLE_TRACE_SIGNAL="${ENABLE_RESOURCE_TRACE_SIGNAL:-0}"

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/resource_monitor.log"
EVENT_LOG="$LOG_DIR/resource_events.log"

last_trace_ts=0
TARGET_PGID=""
if [[ -n "$PARENT_PID" ]] && kill -0 "$PARENT_PID" 2>/dev/null; then
  TARGET_PGID="$(ps -o pgid= -p "$PARENT_PID" | tr -d ' ')"
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_header() {
  {
    echo "[$(timestamp)] resource monitor start"
    echo "interval=${INTERVAL}s gpu_warn_pct=${GPU_WARN_PCT} cpu_warn_pct=${CPU_WARN_PCT} trace_cooldown=${TRACE_COOLDOWN}s enable_trace_signal=${ENABLE_TRACE_SIGNAL}"
    echo "parent_pid=${PARENT_PID:-none}"
    echo "target_pgid=${TARGET_PGID:-none}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "RESOURCE_DIAGNOSTICS_DIR=${RESOURCE_DIAGNOSTICS_DIR:-unset}"
  } >> "$MAIN_LOG"
}

gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits >> "$MAIN_LOG" 2>&1 || true
    nvidia-smi pmon -c 1 >> "$MAIN_LOG" 2>&1 || true
  else
    echo "nvidia-smi not found" >> "$MAIN_LOG"
  fi
}

process_snapshot() {
  {
    echo "-- ps python/ray/vllm --"
    ps -u "$USER" -o pid,ppid,pgid,stat,%cpu,%mem,rss,vsz,etime,cmd \
      | awk 'NR==1 || /python|ray|raylet|gcs_server|vllm|main_ppo|WorkerDict|AlfworldWorker/'
    echo "-- top rss --"
    ps -u "$USER" -o pid,ppid,stat,%cpu,%mem,rss,etime,cmd --sort=-rss | head -25
  } >> "$MAIN_LOG" 2>&1 || true
}

cpu_mem_pct() {
  awk '/MemTotal:/ {total=$2} /MemAvailable:/ {avail=$2} END {if (total > 0) printf "%.0f", (total-avail)*100/total; else print 0}' /proc/meminfo
}

max_gpu_mem_pct() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{
        used=$1+0; total=$2+0;
        if (total > 0) {
          pct=used*100/total;
          if (pct > max) max=pct;
        }
      } END {printf "%.0f", max+0}'
}

python_pids() {
  if [[ -n "$TARGET_PGID" ]]; then
    ps -u "$USER" -o pid=,pgid=,cmd= \
      | awk -v pgid="$TARGET_PGID" '$2 == pgid && /python|ray::|WorkerDict|AlfworldWorker|vllm/ && !/resource_monitor/ {print $1}'
  else
    ps -u "$USER" -o pid=,cmd= \
      | awk '/python|ray::|WorkerDict|AlfworldWorker|vllm/ && !/resource_monitor/ {print $1}'
  fi
}

emit_trace_signal() {
  local reason="$1"
  {
    echo "[$(timestamp)] threshold exceeded: ${reason}"
    echo "-- trace signal disabled; set ENABLE_RESOURCE_TRACE_SIGNAL=1 to send SIGUSR1 --"
  } >> "$EVENT_LOG"

  if [[ "$ENABLE_TRACE_SIGNAL" != "1" ]]; then
    return
  fi

  local now
  now="$(date +%s)"
  if (( now - last_trace_ts < TRACE_COOLDOWN )); then
    return
  fi
  last_trace_ts="$now"

  {
    echo "-- signaling python workers with SIGUSR1 --"
  } >> "$EVENT_LOG"

  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[$(timestamp)] SIGUSR1 pid=${pid}" >> "$EVENT_LOG"
      kill -USR1 "$pid" 2>/dev/null || true
    fi
  done < <(python_pids)
}

parent_alive() {
  if [[ -z "$PARENT_PID" ]]; then
    return 0
  fi
  kill -0 "$PARENT_PID" 2>/dev/null
}

log_header

while parent_alive; do
  {
    echo
    echo "===== $(timestamp) ====="
    echo "-- gpu --"
  } >> "$MAIN_LOG"
  gpu_snapshot
  {
    echo "-- mem --"
    free -h
  } >> "$MAIN_LOG" 2>&1 || true
  process_snapshot

  gpu_pct="$(max_gpu_mem_pct)"
  cpu_pct="$(cpu_mem_pct)"
  if (( gpu_pct >= GPU_WARN_PCT )); then
    emit_trace_signal "gpu_memory_pct=${gpu_pct}"
  fi
  if (( cpu_pct >= CPU_WARN_PCT )); then
    emit_trace_signal "cpu_memory_pct=${cpu_pct}"
  fi

  sleep "$INTERVAL"
done

echo "[$(timestamp)] resource monitor stop" >> "$MAIN_LOG"
