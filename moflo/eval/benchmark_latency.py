"""Measure per-decision wall-clock of MoFlo: frozen encoders + flow integration.

Usage:
  python -m moflo.eval.benchmark_latency --env_name TurnOnMicrowave --n_trials 30
"""
import os, sys, time, json, click, numpy as np, torch
os.environ.setdefault('MUJOCO_GL', 'egl')

from mobipi.macros import POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR
from mobipi.utils.env_utils import load_env, get_metadata, get_env_map_and_default_robot_init_pose
from mobipi.utils.policy_utils import get_config_for_policy
from moflo.data.collect_rollouts import TASK_CAMERA_NAMES, sample_candidate_poses
from moflo.data.cache_features import load_paraphrases
from moflo.data.pointbert_encoder import PointBERTRgbEncoder
from moflo.eval.run_moflo_eval import load_flow, render_ego, w2n


@click.command()
@click.option("--env_name", default="TurnOnMicrowave")
@click.option("--layout", default=0, type=int)
@click.option("--style", default=0, type=int)
@click.option("--n_trials", default=30, type=int)
@click.option("--flow_ckpt", default="checkpoints/moflo/ckpt_best.pt")
@click.option("--n_steps", default=0, type=int, help="Euler steps (0 = use ckpt default)")
@click.option("--out_path", default="results/latency.json")
def main(env_name, layout, style, n_trials, flow_ckpt, n_steps, out_path):
    print(f"[latency] env={env_name} L{layout}_S{style} n={n_trials}", flush=True)
    # Setup env to get a representative observation
    config, _ = get_config_for_policy(POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, env_name, "bc_xfmr", seed=1)
    cam_names = TASK_CAMERA_NAMES.get(env_name, ["robot0_agentview_right"])
    env_meta_list, _ = get_metadata(config)
    base_cams = env_meta_list[0]["env_kwargs"]["camera_names"]
    all_cams = list(dict.fromkeys(list(base_cams) + cam_names + ["robot0_navview"]))
    oargs = {"layout_and_style_ids": [[layout, style]], "place_robot_for_nav": True,
             "camera_names": all_cams, "hard_reset": True, "camera_depths": True}
    (bf, ff, drp, dro, rs) = get_env_map_and_default_robot_init_pose(config, oargs)
    oracle = np.array([drp[0], drp[1], dro[-1]])
    env = load_env(config, oargs)
    env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(0)
    env.reset()
    rgb, depth = render_ego(env, cam_names)

    # Encoders + flow head
    enc = PointBERTRgbEncoder(device="cuda")
    paraphrases = load_paraphrases(env_name)
    text = torch.from_numpy(enc.encode_text(paraphrases[0])).float().cuda()
    flow, default_steps = load_flow(flow_ckpt)
    n_steps = n_steps or default_steps
    cur_world = oracle.copy()
    cur_norm = torch.from_numpy(w2n(cur_world[:2], cur_world[2])).float().unsqueeze(0).cuda()

    def one_inference():
        v = torch.from_numpy(enc.encode_image(rgb, depth_np=depth)).float().unsqueeze(0).cuda()
        with torch.no_grad():
            pred = flow.integrate(cur_norm, v, text.unsqueeze(0), n_steps=n_steps).cpu().numpy()[0]
        return pred

    # Warmup (3 calls)
    for _ in range(3): one_inference()
    torch.cuda.synchronize()

    # Timed trials
    times_ms = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_inference()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        times_ms.append(dt)
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Component breakdown (one big trial)
    n_break = 100
    enc_times = []; pred_times = []
    for _ in range(n_break):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        v = torch.from_numpy(enc.encode_image(rgb, depth_np=depth)).float().unsqueeze(0).cuda()
        torch.cuda.synchronize(); enc_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = flow.integrate(cur_norm, v, text.unsqueeze(0), n_steps=n_steps)
        torch.cuda.synchronize(); pred_times.append((time.perf_counter() - t0) * 1000)

    results = {
        "env_name": env_name, "layout": layout, "style": style, "n_trials": n_trials, "n_steps": n_steps,
        "full_inference_ms": {
            "mean": float(np.mean(times_ms)), "std": float(np.std(times_ms)),
            "median": float(np.median(times_ms)),
            "p5": float(np.percentile(times_ms, 5)),
            "p95": float(np.percentile(times_ms, 95)),
            "min": float(min(times_ms)), "max": float(max(times_ms)),
        },
        "components_ms": {
            "encoder": {"mean": float(np.mean(enc_times)), "std": float(np.std(enc_times))},
            "flow": {"mean": float(np.mean(pred_times)), "std": float(np.std(pred_times))},
        },
        "peak_gpu_mem_mb": float(peak_mem_mb),
    }
    print(json.dumps(results, indent=2))
    if os.path.exists(out_path):
        with open(out_path) as f: existing = json.load(f)
    else:
        existing = {}
    existing[f"{env_name}_L{layout}_S{style}"] = results
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, "w") as f: json.dump(existing, f, indent=2)
    print(f"[latency] saved → {out_path}")


if __name__ == "__main__":
    main()
