"""R2: Baseline methods for base placement.

  --method random   : pick a uniformly-random pose in the ±20cm n2m box around oracle
  --method reach    : pick the closest collision-free pose to oracle (Reachability)
  --method oracle   : navigate directly to ground-truth oracle (policy ceiling)
"""
import os, sys, json, click, numpy as np, torch
import robomimic.utils.train_utils as TrainUtils
from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR
from mobipi.utils.env_utils import load_env, get_metadata, get_env_map_and_default_robot_init_pose
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from moflo.data.collect_rollouts import TASK_CAMERA_NAMES, navigate_and_snap, current_base_pose, sample_candidate_poses


@click.command()
@click.option("--env_name", required=True)
@click.option("--scene_pairs", required=True)
@click.option("--episodes_per_layout", default=6, type=int)
@click.option("--seed", default=11, type=int)
@click.option("--method", required=True, type=click.Choice(["random", "reach", "oracle"]))
@click.option("--out_dir", required=True)
@click.option("--xy_half", default=0.20, type=float)
@click.option("--heading_sigma", default=0.262, type=float)
def main(env_name, scene_pairs, episodes_per_layout, seed, method, out_dir,
         xy_half, heading_sigma):
    os.makedirs(out_dir, exist_ok=True)
    scene_list = [(int(s.split(":")[0]), int(s.split(":")[1])) for s in scene_pairs.split(",")]
    config, _ = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, env_name, "bc_xfmr", seed=1)
    config2, ckpt = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, env_name, "bc_xfmr", seed=1)
    _, rollout_model, _ = load_policy(config2, ckpt)
    cam_names = TASK_CAMERA_NAMES.get(env_name, ["robot0_agentview_right"])
    env_meta_list, _ = get_metadata(config)
    base_cams = env_meta_list[0]["env_kwargs"]["camera_names"]
    all_cams = list(dict.fromkeys(list(base_cams) + cam_names + ["robot0_navview"]))
    succ = total = 0; records = []
    print(f"[baseline] {env_name} method={method}")

    for L, S in scene_list:
        oargs = {"layout_and_style_ids": [[L, S]], "place_robot_for_nav": True,
                 "camera_names": all_cams, "hard_reset": True, "camera_depths": True}
        try: (bf, ff, drp, dro, rs) = get_env_map_and_default_robot_init_pose(config, oargs)
        except Exception as e: print(f"[skip] L{L}: {e}"); continue
        oracle = np.array([drp[0], drp[1], dro[-1]])
        for ep in range(episodes_per_layout):
            ep_seed = seed * 10000 + L * 100 + ep
            env = load_env(config, oargs)
            env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(ep_seed)
            env.reset()
            rng = np.random.RandomState(ep_seed)
            init_poses = sample_candidate_poses(oracle, 1, bf, ff, rs, rng,
                                                 n2m_style=True, n2m_xy_half=xy_half,
                                                 n2m_heading_sigma=heading_sigma)
            navigate_and_snap(env, init_poses[0], n_settle_steps=20)

            # ===== Method-specific pose selection =====
            if method == "oracle":
                # navigate directly to ground-truth oracle pose
                target = oracle.copy()
            elif method == "random":
                # uniformly random in the ±20cm n2m box, with ±15deg heading
                rng2 = np.random.RandomState(ep_seed + 1)
                cands = sample_candidate_poses(oracle, 1, bf, ff, rs, rng2,
                                                n2m_style=True, n2m_xy_half=xy_half,
                                                n2m_heading_sigma=heading_sigma)
                target = cands[0]
            elif method == "reach":
                # closest collision-free pose to oracle out of K candidates
                rng2 = np.random.RandomState(ep_seed + 1)
                K = 64
                cands = sample_candidate_poses(oracle, K, bf, ff, rs, rng2,
                                                n2m_style=True, n2m_xy_half=xy_half,
                                                n2m_heading_sigma=heading_sigma)
                d = np.linalg.norm(cands[:, :2] - oracle[:2], axis=1)
                target = cands[int(np.argmin(d))]

            d_oracle = float(np.linalg.norm(target[:2] - oracle[:2]))
            del env
            env = load_env(config, oargs)
            env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(ep_seed)
            env.reset()
            navigate_and_snap(env, target, n_settle_steps=50)
            logs, _ = TrainUtils.rollout_with_stats(
                policy=rollout_model, envs=[env],
                horizon=[config.experiment.rollout.horizon],
                use_goals=config.use_goals, num_episodes=1,
                render=False, video_dir=None, epoch=f"{method}_{L}_{ep}",
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
                reset_before_rollout=False, del_envs_after_rollouts=True, data_logger=None)
            s = int(logs[env.name]["Success_Rate"] > 0.5)
            total += 1; succ += s
            records.append({"layout": L, "ep": ep, "target": target.tolist(),
                            "oracle": oracle.tolist(), "init_pose": init_poses[0].tolist(),
                            "d_oracle_m": float(d_oracle), "success": s})
            print(f"  L{L} ep{ep}: succ={s} d_oracle={d_oracle*100:.1f}cm  ({succ}/{total} = {100*succ/total:.1f}%)", flush=True)

    out = {"env_name": env_name, "method": method, "success_rate": succ/max(1, total),
           "total": total, "records": records}
    with open(os.path.join(out_dir, f"poc_{env_name}.json"), "w") as f: json.dump(out, f, indent=2)
    print(f"[{method}] {env_name}: SR={100*succ/max(1,total):.1f}%")


if __name__ == "__main__":
    main()
