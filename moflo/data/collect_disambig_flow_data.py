"""Collect disambiguation training data for the x1-flow.

For each train layout and each side in {left, right}:
  - instantiate CloseDrawerDisambig (BOTH drawers open + visible)
  - compute the target = the prompted drawer's oracle base pose
  - sample K candidate "current" poses around the two-drawer midpoint
  - render ego RGB-D at each, encode with PointBERT
  - emit a tuple (visual, language="close the {side} drawer",
                  current_pose_norm, target_pose_norm)

Both prompts are emitted for every scene, with DIFFERENT targets — that is the
signal that forces the (already language-conditioned) x1-flow to actually use
the left/right token. Same model architecture; only the data is new.

Output: one HDF5 per (layout, side) under --out_dir, schema:
    visual    (N, 65, 256) float16
    language  (N, Ltok, 512) float16
    pose_cur  (N, 4) float32   # normalized current pose
    pose_tgt  (N, 4) float32   # normalized target (correct drawer oracle)

Usage:
  python -m moflo.data.collect_disambig_flow_data \\
      --layouts 0,3,5,6 --k_per_cell 100 --out_dir data/disambig_flow_features
"""
import os, sys, h5py, click, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")

import robomimic.utils.env_utils as EnvUtils
from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR
from mobipi.utils.env_utils import get_metadata
from mobipi.utils.policy_utils import get_config_for_policy
from moflo.data.collect_rollouts import navigate_and_snap, current_base_pose
from moflo.data.pointbert_encoder import PointBERTRgbEncoder
import moflo.envs.close_drawer_disambig  # noqa: F401



def w2n(xy, t):
    return np.array([(xy[0]-2.5)/2.5, (xy[1]+2.5)/2.5, np.cos(t), np.sin(t)], dtype=np.float32)


def load_env(config, layout_id, target_side):
    em, sm = get_metadata(config)
    env_meta, shape_meta = em[0], sm[0]
    env_meta["env_name"] = "CloseDrawerDisambig"
    env_meta["env_kwargs"]["seed"] = 0
    env_meta["env_kwargs"].update({
        "layout_and_style_ids": [[layout_id, layout_id]],
        "place_robot_for_nav": True, "hard_reset": True,
        "target_side": target_side, "camera_depths": True,
    })
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, env_name="CloseDrawerDisambig", render=False,
        render_offscreen=config.experiment.render_video,
        use_image_obs=shape_meta["use_images"])
    return EnvUtils.wrap_env_from_config(env, config=config)


def target_oracle(env):
    """Pick the interior-facing collision-free placement for the target drawer."""
    cands = env.unwrapped.env.compute_target_oracle_candidates()
    if not cands:
        return None
    # Heuristic: the placement closest to the room interior (largest |y| toward
    # centre). compute_target_oracle_candidates returns reachable placements;
    # the first is typically the interior-facing one.
    return np.asarray(cands[0])


def render_ego_rgbd(env, cam="robot0_agentview_right", h=224, w=224):
    sim = env.unwrapped.env.sim
    rgb = sim.render(camera_name=cam, width=w, height=h)[::-1]
    d = sim.render(camera_name=cam, width=w, height=h, depth=True)
    if isinstance(d, tuple): d = d[1]
    return np.ascontiguousarray(rgb), np.ascontiguousarray(d[::-1])


@click.command()
@click.option("--layouts", default="0,3,5,6")
@click.option("--k_per_cell", default=100, type=int)
@click.option("--out_dir", default="data/disambig_flow_features")
@click.option("--seed", default=7, type=int)
@click.option("--xy_half", default=0.20, type=float)
@click.option("--heading_sigma", default=0.262, type=float)
def main(layouts, k_per_cell, out_dir, seed, xy_half, heading_sigma):
    os.makedirs(out_dir, exist_ok=True)
    layout_ids = [int(x) for x in layouts.split(",")]
    config, _ = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR,
                                      "CloseDrawer", "bc_xfmr", seed=1)
    enc = PointBERTRgbEncoder(device="cuda")

    for L in layout_ids:
        # Get both drawer oracles by instantiating each side once.
        oracles = {}
        for side in ("left", "right"):
            env = load_env(config, L, side)
            env.reset()
            o = target_oracle(env)
            oracles[side] = o
            del env
            if o is None:
                print(f"[L{L}/{side}] no oracle — skipping layout"); break
        if any(oracles[s] is None for s in ("left", "right")):
            continue
        mid = 0.5 * (oracles["left"] + oracles["right"])
        print(f"[L{L}] left={oracles['left']}  right={oracles['right']}  mid={mid}")

        for side in ("left", "right"):
            tgt = oracles[side]
            prompt = f"close the {side} drawer"
            text = enc.encode_text(prompt)                       # (Ltok, 512)
            env = load_env(config, L, side)
            rng = np.random.RandomState(seed * 100 + L * 10 + (0 if side == "left" else 1))

            vis_list, cur_list = [], []
            for k in range(k_per_cell):
                env.reset()
                # Mixed current-pose sampling:
                #  - 65%: n2m noise around the two-drawer MIDPOINT (both drawers
                #    visible -> the flow must use language to disambiguate)
                #  - 35%: n2m noise around the TARGET drawer oracle (fine-grained
                #    within-drawer transport -> precision signal)
                if rng.rand() < 0.65:
                    centre = mid
                else:
                    centre = tgt
                cx = centre[0] + rng.uniform(-xy_half, xy_half)
                cy = centre[1] + rng.uniform(-xy_half, xy_half)
                cth = centre[2] + rng.normal(0, heading_sigma)
                navigate_and_snap(env, np.array([cx, cy, cth]), n_settle_steps=8)
                cur = np.array(current_base_pose(env.unwrapped.env))
                rgb, depth = render_ego_rgbd(env)
                vis = enc.encode_image(rgb, depth_np=depth)       # (65, 256)
                vis_list.append(vis.astype(np.float16))
                cur_list.append(w2n(cur[:2], cur[2]))
                if (k + 1) % 25 == 0:
                    print(f"  L{L}/{side}: {k+1}/{k_per_cell}", flush=True)
            del env

            N = len(vis_list)
            tgt_norm = w2n(tgt[:2], tgt[2])
            out_h5 = os.path.join(out_dir, f"layout{L}_{side}.hdf5")
            with h5py.File(out_h5, "w") as f:
                f.attrs["layout_id"] = L
                f.attrs["side"] = side
                f.attrs["prompt"] = prompt
                f.create_dataset("visual", data=np.stack(vis_list))
                f.create_dataset("language",
                                 data=np.broadcast_to(text[None].astype(np.float16),
                                                      (N,) + text.shape).copy())
                f.create_dataset("pose_cur", data=np.stack(cur_list).astype(np.float32))
                f.create_dataset("pose_tgt",
                                 data=np.broadcast_to(tgt_norm[None], (N, 4)).copy())
            print(f"[L{L}/{side}] wrote {N} tuples -> {out_h5}", flush=True)

    print("[collect-disambig] DONE")


if __name__ == "__main__":
    main()
