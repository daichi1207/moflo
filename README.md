# MoFlo: Language-Conditioned Flow Matching for Policy Mobilization

Code, checkpoints and reference results for the simulation experiments.

## Quick start

```bash
python3.10 -m venv .venv && source .venv/bin/activate
bash scripts/setup.sh          # dependencies, RoboCasa kitchen assets, downstream policies
bash scripts/eval.sh poses     # 150 pose predictions, checked against reference/
bash scripts/eval.sh main      # the 5-task result, with policy rollouts (needs a GPU with EGL)
```

## Training

The cached training features are in the companion archive (`moflo_companion.zip`, 1.6 GB,
on the [Releases page](https://github.com/daichi1207/moflo/releases)). Unpack it in the
repository root to get `data/features/`, `data/disambig_features/` and `ckpts/three_drawer/`, then:

```bash
bash scripts/train.sh
CKPT=runs/flow/ckpt_best.pt OUT=results_retrained bash scripts/eval.sh main
```

## License and third-party material

BSD-3-Clause-Clear (`LICENSE`). RoboCasa, robomimic, MimicGen, OpenShape and the downstream
policies are downloaded from their authors and keep their own licenses. `data/` contains
small excerpts of RoboCasa demonstrations.
