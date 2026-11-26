"""MoFlo three-drawer disambiguation rollout (CloseDrawerDisambigThree), applied
zero-shot with the checkpoint fine-tuned on the two-way left/right contrast.

From a midpoint pose that sees all three drawers: render ego RGB-D, integrate the
flow conditioned on the prompt, navigate to the produced pose, run the downstream
policy, and check whether the prompted drawer was closed. Also reports whether the
prompted drawer is visible at the reached pose (the paper's navigation success).
"""
import os, sys, json, glob, click, numpy as np, torch
os.environ.setdefault("MUJOCO_GL", "egl")

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.algo import algo_factory
from robomimic.config import config_factory as algo_config_factory

from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, MOBIPI_DIR
from mobipi.utils.env_utils import get_metadata
from mobipi.utils.policy_utils import get_config_for_policy, load_policy, replace_dataset_path
from moflo.data.collect_rollouts import navigate_and_snap, current_base_pose
from moflo.data.pointbert_encoder import PointBERTRgbEncoder
from moflo.models.pose_flow_predictor import PoseFlowPredictor
import moflo.envs.close_drawer_disambig_three  # noqa
from moflo.eval.disambiguation.visibility import target_drawer_visible
from moflo.envs.close_drawer_disambig_three import ORDINALS_3


TASKS = {
    "three": dict(
        env_name="CloseDrawerDisambigThree",
        target_kwarg="target_index", targets=[0, 1, 2],
        prompt=lambda t: f"close the {ORDINALS_3[t]} drawer",
        calib=os.path.join(MOBIPI_DIR, "moflo", "data", "calibration", "three_way.json"),
        anchor_keys=lambda lid: [f"{lid}_leftmost", f"{lid}_middle", f"{lid}_rightmost"],
        policy_task="CloseDrawer",
        extra_kwargs={"open_all": False, "randomize_target": False},
    ),
}


def load_flow(p):
    ck = torch.load(p, map_location="cuda", weights_only=False); a = ck["args"]
    m = PoseFlowPredictor(visual_dim=a["visual_dim"], language_dim=a["language_dim"],
                          pose_dim=a["pose_dim"], d_model=a["d_model"], n_blocks=a["n_blocks"],
                          n_heads=a["n_heads"], ffn_mult=a["ffn_mult"], dropout=a["dropout"],
                          param_mode=a.get("param_mode", "x1")).cuda()
    m.load_state_dict(ck["model"]); m.eval()
    return m, a.get("n_steps_eval", 10)


def w2n(xy, t): return np.array([(xy[0]-2.5)/2.5, (xy[1]+2.5)/2.5, np.cos(t), np.sin(t)], dtype=np.float32)
def n2w(p): return np.array([p[0]*2.5+2.5, p[1]*2.5-2.5, np.arctan2(p[3], p[2])])


def load_custom_policy(ckpt_dir):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    models_dir = os.path.join(ckpt_dir, "models")
    ckpt_files = sorted(glob.glob(os.path.join(models_dir, "model_epoch_*.pth")),
                        key=lambda s: int(s.split("_")[-1].split(".")[0]))
    ckpt_path = ckpt_files[-1]
    print(f"[flow-newtasks] custom policy ckpt: {ckpt_path}", flush=True)
    ext_cfg = json.load(open(cfg_path)); config = algo_config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
        for d in config["train"]["data"]:
            d["path"] = replace_dataset_path(d["path"], DATA_ROOT_DIR)
    env_meta_list, shape_meta_list = get_metadata(config)
    shape_meta = shape_meta_list[0]
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    lang_encoder = LangUtils.LangEncoder(device=device)
    trainset, _ = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)
    action_norm = trainset.get_action_normalization_stats()
    model = algo_factory(algo_name=config.algo_name, config=config,
                         obs_key_shapes=shape_meta["all_shapes"],
                         ac_dim=shape_meta["ac_dim"], device=device)
    ck = FileUtils.maybe_dict_from_checkpoint(ckpt_path)
    model.deserialize(ck["model"]); model.set_eval()
    rm = TrainUtils.RolloutPolicy(model, obs_normalization_stats=None,
                                  action_normalization_stats=action_norm,
                                  lang_encoder=lang_encoder)
    return config, rm


def load_env(config, tcfg, lid, target):
    em, sm = get_metadata(config); env_meta, shape_meta = em[0], sm[0]
    env_meta["env_name"] = tcfg["env_name"]
    env_meta["env_kwargs"]["seed"] = 0
    env_meta["env_kwargs"].pop("layout_ids", None)
    env_meta["env_kwargs"].pop("style_ids", None)
    env_meta["env_kwargs"].update({
        "layout_and_style_ids": [[lid, lid]],
        "place_robot_for_nav": True, "hard_reset": True,
        "camera_depths": True,
        tcfg["target_kwarg"]: target,
    })
    env_meta["env_kwargs"].update(tcfg["extra_kwargs"])
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, env_name=tcfg["env_name"], render=False,
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
@click.option("--task", type=click.Choice(["three"]), default="three")
@click.option("--flow_ckpt", default="checkpoints/moflo_disambig/ckpt_best.pt")
@click.option("--policy_ckpt_dir", default=None, help="custom trained policy dir; else standard per-task")
@click.option("--out_dir", required=True)
@click.option("--num_eps", default=5, type=int)
@click.option("--seed", default=11, type=int)
@click.option("--layouts", required=True)
@click.option("--start_backoff", default=0.9, type=float)
def main(task, flow_ckpt, policy_ckpt_dir, out_dir, num_eps, seed, layouts, start_backoff):
    os.makedirs(out_dir, exist_ok=True)
    tcfg = TASKS[task]
    if policy_ckpt_dir:
        config, rollout_model = load_custom_policy(policy_ckpt_dir)
    else:
        config, ckpt = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR,
                                             tcfg["policy_task"], "bc_xfmr", seed=1)
        _, rollout_model, _ = load_policy(config, ckpt)
    flow, n_steps = load_flow(flow_ckpt)
    enc = PointBERTRgbEncoder(device="cuda")
    calib = json.load(open(tcfg["calib"]))
    layout_ids = [int(x) for x in layouts.split(",")]
    print(f"[flow-newtasks] task={task} n_steps={n_steps} flow={flow_ckpt}", flush=True)

    all_results = {}
    for lid in layout_ids:
        akeys = tcfg["anchor_keys"](lid)
        if any(k not in calib for k in akeys):
            print(f"[skip] L{lid}: missing calib anchors {akeys}"); continue
        anchors = [np.array(calib[k]["oracle"], dtype=float) for k in akeys]
        for ti, target in enumerate(tcfg["targets"]):
            prompt = tcfg["prompt"](target)
            mid = np.mean(np.stack(anchors), axis=0)
            heading = mid[2]
            start = mid.copy()
            start[0] -= start_backoff * np.cos(heading)
            start[1] -= start_backoff * np.sin(heading)
            try:
                env = load_env(config, tcfg, lid, target)
            except Exception as e:
                print(f"[skip] L{lid}/{target}: env create fail {e}"); continue
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
                tgt_pose = n2w(x1.cpu().numpy()[0])
                dists = [np.linalg.norm(tgt_pose[:2] - a[:2]) for a in anchors]
                picked_idx = int(np.argmin(dists))
                picks.append(picked_idx)
                env.reset()
                navigate_and_snap(env, tgt_pose, n_settle_steps=50)
                viss.append(int(target_drawer_visible(env)))
                logs, _ = TrainUtils.rollout_with_stats(
                    policy=rollout_model, envs=[env],
                    horizon=[config.experiment.rollout.horizon],
                    use_goals=config.use_goals, num_episodes=1, render=False,
                    video_dir=None, epoch=f"L{lid}_{target}_ep{ep}",
                    video_skip=config.experiment.get("video_skip", 5),
                    terminate_on_success=config.experiment.rollout.terminate_on_success,
                    reset_before_rollout=False, del_envs_after_rollouts=False,
                    data_logger=None)
                s = int(logs[env.name]["Success_Rate"] > 0.5)
                succs.append(s)
                correct = (picked_idx == ti)
                print(f"  L{lid}/{target} ep{ep}: picked_idx={picked_idx} "
                      f"correct={correct} success={s}", flush=True)
            del env
            key = f"{lid}_{target}"
            all_results[key] = {
                "successes": succs, "sr": 100.0*sum(succs)/len(succs),
                "picks": picks,
                "nav_vis": viss, "nav_vis_rate": 100.0*sum(viss)/len(viss),
                "pick_correct_rate": 100.0*sum(p == ti for p in picks)/len(picks),
            }
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*64)
    print(f"MoFlo (x1-flow) disambig rollout-SR -- task={task}")
    print("="*64)
    srs = [v["sr"] for v in all_results.values()]
    pcs = [v["pick_correct_rate"] for v in all_results.values()]
    for k in sorted(all_results.keys()):
        v = all_results[k]
        print(f"  {k:14s} manip-SR={v['sr']:5.1f}%  referent-pick={v['pick_correct_rate']:5.1f}%")
    print("-"*64)
    if srs:
        print(f"  MEAN          manip-SR={np.mean(srs):5.1f}%  referent-pick={np.mean(pcs):5.1f}%")


if __name__ == "__main__":
    main()
