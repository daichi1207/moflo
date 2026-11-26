"""Evaluate MoFlo on one RoboCasa task.

Each episode places the robot within +-20 cm / ~15 deg of the task's default pose,
renders ego RGB-D, integrates the flow from that pose to get a target base pose,
navigates there, and runs the downstream per-task policy.

    ep_seed = seed * 10000 + layout * 100 + episode

--pose_only stops after the pose prediction (no policy rollout). With the default
seed the predictions can be checked against reference/ (scripts/verify.py).
"""
import os, sys, json, click, numpy as np, torch
import robomimic.utils.train_utils as TrainUtils
from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR
from mobipi.utils.env_utils import load_env, get_metadata, get_env_map_and_default_robot_init_pose
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from moflo.data.collect_rollouts import TASK_CAMERA_NAMES, navigate_and_snap, current_base_pose, sample_candidate_poses
from moflo.data.cache_features import load_paraphrases
from moflo.data.pointbert_encoder import PointBERTRgbEncoder
from moflo.models.pose_flow_predictor import PoseFlowPredictor


def render_ego(env, cn, H=224, W=224):
    sim = env.unwrapped.env.sim
    rgb = sim.render(camera_name=cn[0], width=W, height=H)[::-1]
    d = sim.render(camera_name=cn[0], width=W, height=H, depth=True)
    if isinstance(d, tuple): d = d[1]
    return np.ascontiguousarray(rgb), np.ascontiguousarray(d[::-1])


def load_flow(p):
    ck = torch.load(p, map_location="cuda", weights_only=False); a = ck["args"]
    m = PoseFlowPredictor(visual_dim=a["visual_dim"], language_dim=a["language_dim"],
                          pose_dim=a["pose_dim"], d_model=a["d_model"], n_blocks=a["n_blocks"],
                          n_heads=a["n_heads"], ffn_mult=a["ffn_mult"], dropout=a["dropout"],
                          param_mode=a.get("param_mode", "velocity")).cuda()
    m.load_state_dict(ck["model"]); m.eval()
    return m, a.get("n_steps_eval", 10)


def w2n(xy, t): return np.array([(xy[0]-2.5)/2.5, (xy[1]+2.5)/2.5, np.cos(t), np.sin(t)], dtype=np.float32)
def norm_to_world(pn): return np.array([pn[0]*2.5 + 2.5, pn[1]*2.5 - 2.5, np.arctan2(pn[3], pn[2])])


@click.command()
@click.option("--env_name", required=True)
@click.option("--scene_pairs", required=True)
@click.option("--episodes_per_layout", default=6, type=int)
@click.option("--ep_start", default=0, type=int,
              help="First episode index per layout (default 0). Set to the "
                   "episode count of a previous run to extend it with fresh "
                   "episodes: ep_seed = seed*10^4 + layout*100 + ep, so the "
                   "new episodes are disjoint from and poolable with the old.")
@click.option("--seed", default=11, type=int)
@click.option("--flow_ckpt", required=True)
@click.option("--out_dir", required=True)
@click.option("--n_steps", default=0, type=int, help="Euler steps (0 = use ckpt default)")
@click.option("--lang_mode", type=click.Choice(["normal", "dummy", "zero", "shuffle"]),
              default="normal", help="language ablation (normal = real prompt)")
@click.option("--paraphrase_set", type=click.Choice(["paraphrases", "held_out_paraphrases"]),
              default="paraphrases",
              help="held_out_paraphrases = novel instruction phrasings never seen "
                   "in training (instruction-generalization test)")
@click.option("--xy_half", default=0.20, type=float)
@click.option("--heading_sigma", default=0.262, type=float)
@click.option("--pose_only", is_flag=True, help="predict and record poses; skip the policy rollout")
def main(env_name, scene_pairs, episodes_per_layout, ep_start, seed, flow_ckpt,
         out_dir, n_steps, lang_mode, paraphrase_set, xy_half, heading_sigma, pose_only):
    os.makedirs(out_dir, exist_ok=True)
    scene_list = [(int(s.split(":")[0]), int(s.split(":")[1])) for s in scene_pairs.split(",")]
    config, _ = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, env_name, "bc_xfmr", seed=1)
    if not pose_only:
        config2, ckpt = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, env_name, "bc_xfmr", seed=1)
        _, rollout_model, _ = load_policy(config2, ckpt)
    flow, default_steps = load_flow(flow_ckpt)
    if n_steps <= 0: n_steps = default_steps
    enc = PointBERTRgbEncoder(device="cuda")
    paraphrases = load_paraphrases(env_name, paraphrase_set=paraphrase_set)
    # Language ablation: build the text vector fed to the flow
    if lang_mode == "normal":
        text = torch.from_numpy(enc.encode_text(paraphrases[0])).float().cuda()
        print(f"[moflo] paraphrase_set={paraphrase_set} prompt={paraphrases[0]!r}")
    elif lang_mode == "dummy":
        text = torch.from_numpy(enc.encode_text("perform the task")).float().cuda()
    elif lang_mode == "zero":
        text = torch.zeros_like(torch.from_numpy(enc.encode_text("x")).float().cuda())
    elif lang_mode == "shuffle":
        _OTHER = {"TurnOnMicrowave": "close the drawer", "CloseDrawer": "turn on the stove",
                  "CloseSingleDoor": "turn on the sink faucet",
                  "TurnOnSinkFaucet": "close the cabinet door",
                  "TurnOnStove": "turn on the microwave"}
        text = torch.from_numpy(enc.encode_text(_OTHER[env_name])).float().cuda()
    print(f"[moflo] lang_mode={lang_mode}")
    cam_names = TASK_CAMERA_NAMES.get(env_name, ["robot0_agentview_right"])
    env_meta_list, _ = get_metadata(config)
    base_cams = env_meta_list[0]["env_kwargs"]["camera_names"]
    all_cams = list(dict.fromkeys(list(base_cams) + cam_names + ["robot0_navview"]))
    succ = total = 0; records = []
    print(f"[moflo] {env_name} n_steps={n_steps}")

    for L, S in scene_list:
        oargs = {"layout_and_style_ids": [[L, S]], "place_robot_for_nav": True,
                 "camera_names": all_cams, "hard_reset": True, "camera_depths": True}
        try: (bf, ff, drp, dro, rs) = get_env_map_and_default_robot_init_pose(config, oargs)
        except Exception as e: print(f"[skip] L{L}: {e}"); continue
        oracle = np.array([drp[0], drp[1], dro[-1]])
        for ep in range(ep_start, episodes_per_layout):
            ep_seed = seed * 10000 + L * 100 + ep
            env = load_env(config, oargs)
            env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(ep_seed)
            env.reset()
            rng = np.random.RandomState(ep_seed)
            init_poses = sample_candidate_poses(oracle, 1, bf, ff, rs, rng,
                                                 n2m_style=True, n2m_xy_half=xy_half,
                                                 n2m_heading_sigma=heading_sigma)
            navigate_and_snap(env, init_poses[0], n_settle_steps=20)
            cur = np.array(current_base_pose(env.unwrapped.env))
            rgb, depth = render_ego(env, cam_names)
            vfeat = enc.encode_image(rgb, depth_np=depth)
            visual = torch.from_numpy(np.asarray(vfeat)).float().unsqueeze(0).cuda()
            x0 = torch.from_numpy(w2n(cur[:2], cur[2])).float().unsqueeze(0).cuda()
            with torch.no_grad():
                x1 = flow.integrate(x0, visual, text.unsqueeze(0), n_steps=n_steps)
            target = norm_to_world(x1.cpu().numpy()[0])
            d_oracle = np.linalg.norm(target[:2] - oracle[:2])
            del env
            rec = {"layout": L, "ep": ep, "target": target.tolist(), "oracle": oracle.tolist(),
                   "init_pose": init_poses[0].tolist(), "d_oracle_m": float(d_oracle)}
            if pose_only:
                records.append(rec)
                print(f"  L{L} ep{ep}: target={np.round(target, 4).tolist()}", flush=True)
                continue
            env = load_env(config, oargs)
            env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(ep_seed)
            env.reset()
            navigate_and_snap(env, target, n_settle_steps=50)
            logs, _ = TrainUtils.rollout_with_stats(
                policy=rollout_model, envs=[env],
                horizon=[config.experiment.rollout.horizon],
                use_goals=config.use_goals, num_episodes=1,
                render=False, video_dir=None, epoch=f"flow_{L}_{ep}",
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
                reset_before_rollout=False, del_envs_after_rollouts=True, data_logger=None)
            s = int(logs[env.name]["Success_Rate"] > 0.5)
            total += 1; succ += s
            records.append({**rec, "success": s})
            print(f"  L{L} ep{ep}: succ={s} d_oracle={d_oracle*100:.1f}cm  ({succ}/{total} = {100*succ/total:.1f}%)", flush=True)
    out = {"env_name": env_name, "method": "moflo", "pose_only": pose_only,
           "success_rate": None if pose_only else succ / max(1, total), "total": total,
           "n_steps": n_steps, "seed": seed, "ep_start": ep_start,
           "episodes_per_layout": episodes_per_layout, "records": records}
    with open(os.path.join(out_dir, f"poc_{env_name}.json"), "w") as f: json.dump(out, f, indent=2)
    if not pose_only:
        print(f"[moflo] {env_name}: SR={100*succ/max(1,total):.1f}%")


if __name__ == "__main__":
    main()
