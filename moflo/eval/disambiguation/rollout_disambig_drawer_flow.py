"""Within-task language disambiguation rollout-SR for the x1-flow.

Harder than rollout_disambig_drawer.py: that one gives each method the two
oracle placements to *rank*. Here the x1-flow must *generate* the target pose
from (ego RGB-D + "close the {left|right} drawer") with no candidate set.

Per (layout, side): place the robot at a midpoint pose that sees BOTH open
drawers, render ego RGB-D, run the x1-flow conditioned on the disambiguating
prompt, navigate to the produced pose, run the bc-policy, and check whether the
*prompted* drawer was closed (CloseDrawerDisambig._check_success).

A language-blind predictor scores ~50% (same pose for both prompts -> right on
one, wrong on the other). A model that genuinely uses "left/right" scores well
above 50% AND is balanced across the two prompts.

Usage:
  python -m moflo.eval.disambiguation.rollout_disambig_drawer_flow \\
      --flow_ckpt checkpoints/moflo_disambig/ckpt_best.pt \\
      --out_dir results/disambig_2way --num_eps 10
"""
import os, sys, json, click, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.env_utils as EnvUtils
from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR
from mobipi.utils.env_utils import get_metadata
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from moflo.data.collect_rollouts import navigate_and_snap, current_base_pose
from moflo.data.pointbert_encoder import PointBERTRgbEncoder
from moflo.models.pose_flow_predictor import PoseFlowPredictor
import moflo.envs.close_drawer_disambig  # noqa: F401
from moflo.eval.disambiguation.visibility import target_drawer_visible

from mobipi.macros import MOBIPI_DIR
CALIB = os.path.join(MOBIPI_DIR, "moflo", "data", "calibration", "two_way.json")


def load_flow(p):
    ck = torch.load(p, map_location="cuda", weights_only=False); a = ck["args"]
    m = PoseFlowPredictor(visual_dim=a["visual_dim"], language_dim=a["language_dim"],
                          pose_dim=a["pose_dim"], d_model=a["d_model"], n_blocks=a["n_blocks"],
                          n_heads=a["n_heads"], ffn_mult=a["ffn_mult"], dropout=a["dropout"],
                          param_mode=a.get("param_mode", "velocity")).cuda()
    m.load_state_dict(ck["model"]); m.eval()
    return m, a.get("n_steps_eval", 10)


def w2n(xy, t): return np.array([(xy[0]-2.5)/2.5, (xy[1]+2.5)/2.5, np.cos(t), np.sin(t)], dtype=np.float32)
def n2w(p): return np.array([p[0]*2.5+2.5, p[1]*2.5-2.5, np.arctan2(p[3], p[2])])


def load_disambig_env(config, layout_id, target_side):
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


def render_ego_rgbd(env, cam="robot0_agentview_right", h=224, w=224):
    sim = env.unwrapped.env.sim
    rgb = sim.render(camera_name=cam, width=w, height=h)[::-1]
    d = sim.render(camera_name=cam, width=w, height=h, depth=True)
    if isinstance(d, tuple): d = d[1]
    return np.ascontiguousarray(rgb), np.ascontiguousarray(d[::-1])


@click.command()
@click.option("--flow_ckpt", default="checkpoints/moflo_disambig/ckpt_best.pt")
@click.option("--out_dir", default="results/disambig_2way")
@click.option("--num_eps", default=10, type=int)
@click.option("--seed", default=11, type=int)
@click.option("--layouts", default="1,4,7,8")
@click.option("--start_backoff", default=0.9, type=float,
              help="metres to back the start pose off the drawer row so both drawers are in view")
def main(flow_ckpt, out_dir, num_eps, seed, layouts, start_backoff):
    os.makedirs(out_dir, exist_ok=True)
    config, ckpt = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR,
                                         "CloseDrawer", "bc_xfmr", seed=1)
    _, rollout_model, _ = load_policy(config, ckpt)
    flow, n_steps = load_flow(flow_ckpt)
    enc = PointBERTRgbEncoder(device="cuda")
    calib = json.load(open(CALIB))
    layout_ids = [int(x) for x in layouts.split(",")]
    print(f"[disambig-flow] n_steps={n_steps}")

    all_results = {}
    for lid in layout_ids:
        for side in ("left", "right"):
            key = f"{lid}_{side}"
            if calib.get(key, {}).get("sr", 0) <= 0:
                print(f"[skip] {key} oracle SR={calib.get(key,{}).get('sr')}"); continue
            oracle_l = np.array(calib[f"{lid}_left"]["oracle"])
            oracle_r = np.array(calib[f"{lid}_right"]["oracle"])
            correct = oracle_l if side == "left" else oracle_r
            prompt = f"close the {side} drawer"
            # Start pose: midpoint of the two drawers, backed off so both visible.
            mid = 0.5 * (oracle_l + oracle_r)
            heading = mid[2]
            start = mid.copy()
            start[0] -= start_backoff * np.cos(heading)
            start[1] -= start_backoff * np.sin(heading)

            env = load_disambig_env(config, lid, side)
            text = torch.from_numpy(enc.encode_text(prompt)).float().cuda()
            succs, picks, viss = [], [], []
            for ep in range(num_eps):
                env.reset()
                navigate_and_snap(env, start, n_settle_steps=20)
                cur = np.array(current_base_pose(env.unwrapped.env))
                rgb, depth = render_ego_rgbd(env)
                vis = torch.from_numpy(enc.encode_image(rgb, depth_np=depth)).float().unsqueeze(0).cuda()
                x0 = torch.from_numpy(w2n(cur[:2], cur[2])).float().unsqueeze(0).cuda()
                with torch.no_grad():
                    x1 = flow.integrate(x0, vis, text.unsqueeze(0), n_steps=n_steps)
                target = n2w(x1.cpu().numpy()[0])
                # Which drawer is the produced pose closer to?
                dl = np.linalg.norm(target[:2]-oracle_l[:2])
                dr = np.linalg.norm(target[:2]-oracle_r[:2])
                picked = "left" if dl < dr else "right"
                picks.append(picked)
                env.reset()
                navigate_and_snap(env, target, n_settle_steps=50)
                viss.append(int(target_drawer_visible(env)))
                logs, _ = TrainUtils.rollout_with_stats(
                    policy=rollout_model, envs=[env],
                    horizon=[config.experiment.rollout.horizon],
                    use_goals=config.use_goals, num_episodes=1, render=False,
                    video_dir=None, epoch=f"L{lid}_{side}_ep{ep}",
                    video_skip=config.experiment.get("video_skip", 5),
                    terminate_on_success=config.experiment.rollout.terminate_on_success,
                    reset_before_rollout=False, del_envs_after_rollouts=False,
                    data_logger=None)
                s = int(logs[env.name]["Success_Rate"] > 0.5)
                succs.append(s)
                print(f"  L{lid}/{side} ep{ep}: picked={picked} "
                      f"correct={picked==side} success={s}", flush=True)
            del env
            all_results[key] = {
                "successes": succs, "sr": 100.0*sum(succs)/len(succs),
                "picks": picks,
                "nav_vis": viss, "nav_vis_rate": 100.0*sum(viss)/len(viss),
                "pick_correct_rate": 100.0*sum(p==side for p in picks)/len(picks),
            }
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("x1-flow within-task disambiguation rollout-SR")
    print("="*60)
    srs = [v["sr"] for v in all_results.values()]
    pcs = [v["pick_correct_rate"] for v in all_results.values()]
    for k in sorted(all_results.keys()):
        v = all_results[k]
        print(f"  {k:10s} SR={v['sr']:5.1f}%  pick-correct={v['pick_correct_rate']:5.1f}%")
    print("-"*60)
    print(f"  MEAN       SR={np.mean(srs):5.1f}%  pick-correct={np.mean(pcs):5.1f}%")
    print("  (language-blind baseline = 50%; >50% balanced = genuine disambiguation)")


if __name__ == "__main__":
    main()
