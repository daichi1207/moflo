#!/usr/bin/env bash
# Train MoFlo. Needs the cached features (see README, "Training").
#   FEATURES=data/features   DISAMBIG_FEATURES=data/disambig_features
#   EPOCHS / DISAMBIG_EPOCHS / OUT override the defaults.
#   WITH_SCORER=1 also trains the score-only ablation head.
set -euo pipefail
cd "$(dirname "$0")/.."
FEATURES="${FEATURES:-data/features}"
DISAMBIG_FEATURES="${DISAMBIG_FEATURES:-data/disambig_features}"
OUT="${OUT:-runs}"
DIRS=""
for t in TurnOnMicrowave CloseDrawer CloseSingleDoor TurnOnSinkFaucet TurnOnStove; do
  [ -d "$FEATURES/$t" ] || { echo "missing $FEATURES/$t (see README, Training)"; exit 1; }
  DIRS="$DIRS,$FEATURES/$t"
done
DIRS="${DIRS#,}"

python -m moflo.train.train_pose_flow \
  --feature_dirs "$DIRS" --oracles_json moflo/data/empirical_oracles.json --out_dir "$OUT/flow" \
  --epochs "${EPOCHS:-150}" --batch_size 128 --num_workers 4 --base_lr 1e-4 \
  --d_model 384 --n_blocks 8 --n_heads 8 --fail_weight 0.3 --n_steps_eval 10 --sigma_path 0.0 \
  --target_mode oracle --param_mode x1 --loss_space velocity --t_beta_a 1.0 --t_beta_b 4.0 --seed 42

if [ -d "$DISAMBIG_FEATURES" ]; then
  python -m moflo.train.train_disambig_flow \
    --data_dir "$DISAMBIG_FEATURES" --out_dir "$OUT/flow_disambig" --epochs "${DISAMBIG_EPOCHS:-250}" \
    --batch_size 128 --base_lr 1e-4 --loss_space velocity --t_beta_a 1.0 --t_beta_b 4.0 \
    --init_from "$OUT/flow/ckpt_best.pt" --seed 42
fi
if [ "${WITH_SCORER:-0}" = 1 ]; then
  python -m moflo.train.train_scorer --feature_dirs "$DIRS" --out_dir "$OUT/scorer"
fi
