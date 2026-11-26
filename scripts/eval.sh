#!/usr/bin/env bash
# Usage: bash scripts/eval.sh {poses|main|baselines|latency|disambig|all}
#   poses      predicted poses only (150 decisions), checked against reference/
#   main       5 tasks x 30 episodes with policy rollouts (the headline result)
#   baselines  Random and Reachability
#   latency    per-decision wall-clock
#   disambig   2-way; plus 3-way if THREE_POLICY_DIR is set
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
TASKS="${TASKS:-TurnOnMicrowave CloseDrawer CloseSingleDoor TurnOnSinkFaucet TurnOnStove}"
SP="0:0,2:2,3:3,5:5,6:6"   # seen layout:style pairs
EPS="${EPS:-6}"
OUT="${OUT:-results}"
CKPT="${CKPT:-checkpoints/moflo/ckpt_best.pt}"
WHAT="${1:-poses}"
has() { [ "$WHAT" = all ] || [ "$WHAT" = "$1" ]; }
case "$WHAT" in poses|main|baselines|latency|disambig|all) ;; *) sed -n 2,8p "$0"; exit 2;; esac

if has poses; then
  for t in $TASKS; do
    python -m moflo.eval.run_moflo_eval --env_name "$t" --scene_pairs "$SP" --episodes_per_layout "$EPS" \
      --seed 11 --flow_ckpt "$CKPT" --out_dir "$OUT/poses" --pose_only
  done
  python scripts/verify.py "$OUT/poses"
fi
if has main; then
  for t in $TASKS; do
    python -m moflo.eval.run_moflo_eval --env_name "$t" --scene_pairs "$SP" --episodes_per_layout "$EPS" \
      --seed 11 --flow_ckpt "$CKPT" --out_dir "$OUT/main"
  done
  python scripts/verify.py "$OUT/main"
fi
if has baselines; then
  for m in random reach; do for t in $TASKS; do
    python -m moflo.eval.run_baselines --env_name "$t" --scene_pairs "$SP" --episodes_per_layout "$EPS" \
      --seed 11 --method "$m" --out_dir "$OUT/baseline_$m"
  done; done
fi
if has latency; then
  python -m moflo.eval.benchmark_latency --flow_ckpt "$CKPT" --env_name TurnOnMicrowave --n_trials 30 --out_path "$OUT/latency.json"
fi
if has disambig; then
  DCKPT="${DCKPT:-checkpoints/moflo_disambig/ckpt_best.pt}"
  python -m moflo.eval.disambiguation.rollout_disambig_drawer_flow --flow_ckpt "$DCKPT" \
    --out_dir "$OUT/disambig_2way" --num_eps 10 --layouts 1,4,7,8
  if [ -n "${THREE_POLICY_DIR:-}" ]; then
    python -m moflo.eval.disambiguation.rollout_disambig_three_flow --flow_ckpt "$DCKPT" \
      --policy_ckpt_dir "$THREE_POLICY_DIR" --out_dir "$OUT/disambig_3way" --num_eps 5 --layouts 7,8,9
  else
    echo "3-way skipped: set THREE_POLICY_DIR (see README)"
  fi
fi
