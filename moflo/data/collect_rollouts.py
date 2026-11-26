"""
MoFlo data collection: sample candidate SE(2) base poses around the oracle,
teleport the robot, run the per-task BC-Transformer policy, and log
(pose_delta, ego_view, success, language) tuples.

One "rollout record" is keyed by (env_name, layout_id, style_id, seed, rollout_idx)
and stored as a row in a per-task HDF5 shard:

    out_dir/
        TurnOnMicrowave/
            layout0_style0_seed1.hdf5
                rollouts/0000/ego_view  (H, W, 3) uint8
                             /depth     (H, W) float32  (optional)
                             /candidate_pose  (3,) float64   world frame
                             /oracle_pose     (3,) float64   world frame
                             /success         ()  int8
                             /language        ()  str
                             /camera_name     ()  str

Viewpoint augmentation is NOT done here (it's a pure post-process on the
saved ego view + oracle pose); see viewpoint_augment.py.

Usage:
    python -m moflo.data.collect_rollouts \\
        --env_name TurnOnMicrowave \\
        --layout_ids 0,2,3,5,6 \\
        --num_candidates 30 \\
        --seed 1 \\
        --out_dir data/moflo_rollouts
"""
import os
import sys
import json
import h5py
import time
import click
import warnings
import numpy as np
from datetime import datetime

import robomimic.utils.train_utils as TrainUtils
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

from mobipi.macros import (
    POLICY_CKPT_ROOT_DIR, DATA_ROOT_DIR, LOG_ROOT_DIR,
)
from mobipi.utils.io_utils import DualStream, camel_to_snake_case
from mobipi.utils.env_utils import (
    load_env,
    get_env_map_and_default_robot_init_pose,
    get_metadata,
    check_robot_positions,
    compute_robot_angle_from_mat,
    render_image_with_robot_mask,
)
from mobipi.utils.policy_utils import get_config_for_policy, load_policy
from mobipi.utils.nav_utils import move_to_pose as _move_to_pose
import robosuite.utils.transform_utils as T
import mimicgen.utils.pose_utils as PoseUtils


class _DummyCollisionChecker:
    """Always-collision-free collision checker for move_to_pose.

    In ``mobipi.utils.nav_utils.is_collision_free`` the raw score is inverted
    before the "any collision" test:
        scores = ~k_col.compute_score(...)
        return not np.any(scores > 0)
    So to signal "always free", compute_score must return all-True (before the
    inversion, which yields all-False, which yields "not any collision").
    """
    def __init__(self, robot_size=0.5):
        self.robot_size = robot_size

    def compute_score(self, *_args, numpy=False, robot_size=None, **_kwargs):
        poses = _args[2] if len(_args) >= 3 else _kwargs.get("poses")
        poses_arr = np.array(poses).reshape(-1, 3)
        return np.ones(len(poses_arr), dtype=bool)


warnings.filterwarnings(
    "ignore",
    message=".*default value of the antialias parameter.*",
    category=UserWarning,
)


# Per-task cameras (mirrors eval_mobipi.py). Primary (ego) camera is index 0.
# v7: wrist camera (robot0_eye_in_hand) moved to index 0 — closer to N2M's
# wrist-RGB-D setup. Agentview kept as secondary for diagnostic renders.
TASK_CAMERA_NAMES = {
    "CloseDrawer": ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"],
    "CloseSingleDoor": ["robot0_eye_in_hand", "robot0_agentview_right"],
    "TurnOnMicrowave": ["robot0_eye_in_hand", "robot0_agentview_right"],
    "TurnOnSinkFaucet": ["robot0_eye_in_hand", "robot0_agentview_right"],
    "TurnOnStove": ["robot0_eye_in_hand", "robot0_agentview_right"],
}


def sample_candidate_poses(
    oracle_pose,
    num_samples,
    base_fixture_bounds_2d,
    floor_fixture_bounds_2d,
    robot_size,
    rng,
    close_frac=0.55,
    medium_frac=0.30,
    close_radius=0.35,
    medium_radius=0.8,
    far_radius=1.3,
    close_heading_sigma=0.20,     # ~11° std — tight near-oracle heading
    medium_heading_sigma=0.55,    # ~31° std
    far_heading_sigma=1.2,        # ~69° std
    max_tries=5000,
    n2m_style=False,
    n2m_xy_half=0.2,              # ±0.2m uniform → 0.4m × 0.4m box
    n2m_heading_sigma=0.262,      # ~15° std, matches N2M's 15° angular variance
):
    """n2m_style=True switches to the N2M-paper sampling: all candidates drawn
    uniformly in a 0.4 m × 0.4 m box around oracle_pose[:2] with a Gaussian
    ±15° heading offset. All candidates land inside the BC policy's success
    basin, so labels are mostly positive (matches what N2M reports).
    """
    if n2m_style:
        ox, oy, otheta = oracle_pose
        out = []
        tries = 0
        kept = 0
        while kept < num_samples and tries < max_tries:
            tries += 1
            x = ox + rng.uniform(-n2m_xy_half, n2m_xy_half)
            y = oy + rng.uniform(-n2m_xy_half, n2m_xy_half)
            dtheta = rng.normal(0.0, n2m_heading_sigma)
            dtheta = float(np.clip(dtheta, -np.pi, np.pi))
            theta = np.arctan2(np.sin(otheta + dtheta), np.cos(otheta + dtheta))
            valid = check_robot_positions(
                [[x, y]],
                base_fixture_bounds_2d,
                floor_fixture_bounds_2d,
                robot_size,
            )
            if valid[0]:
                out.append([x, y, theta])
                kept += 1
        while kept < num_samples:
            x = ox + rng.uniform(-n2m_xy_half, n2m_xy_half)
            y = oy + rng.uniform(-n2m_xy_half, n2m_xy_half)
            dtheta = rng.normal(0.0, n2m_heading_sigma)
            dtheta = float(np.clip(dtheta, -np.pi, np.pi))
            theta = np.arctan2(np.sin(otheta + dtheta), np.cos(otheta + dtheta))
            out.append([x, y, theta])
            kept += 1
        return np.array(out[:num_samples], dtype=np.float64)

    """Sample ``num_samples`` candidate (x, y, theta) poses around ``oracle_pose``.

    Position distribution: close_frac within close_radius, medium_frac within
    close_radius..medium_radius, remainder within medium_radius..far_radius.

    Heading is distance-conditional: close samples get heading drawn from
    Normal(oracle, close_heading_sigma) wrapped to [-pi, pi]; medium and far
    buckets use progressively wider sigmas. This keeps a nontrivial positive
    fraction in the dataset while still spanning the full SE(2) delta window.

    Candidates are collision-checked against fixtures via check_robot_positions.
    Returns array of shape (num_samples, 3) in WORLD frame.
    """
    n_close = int(round(num_samples * close_frac))
    n_medium = int(round(num_samples * medium_frac))
    n_far = num_samples - n_close - n_medium

    ox, oy, otheta = oracle_pose
    out = []
    tries = 0
    targets = [
        (n_close, 0.0, close_radius, close_heading_sigma),
        (n_medium, close_radius, medium_radius, medium_heading_sigma),
        (n_far, medium_radius, far_radius, far_heading_sigma),
    ]
    for count, rmin, rmax, sigma in targets:
        kept = 0
        while kept < count and tries < max_tries:
            tries += 1
            r = rng.uniform(rmin, rmax)
            phi = rng.uniform(-np.pi, np.pi)
            x = ox + r * np.cos(phi)
            y = oy + r * np.sin(phi)
            # Gaussian heading offset, wrapped to [-pi, pi].
            dtheta = rng.normal(0.0, sigma)
            dtheta = float(np.clip(dtheta, -np.pi, np.pi))
            theta = otheta + dtheta
            theta = np.arctan2(np.sin(theta), np.cos(theta))
            valid = check_robot_positions(
                [[x, y]],
                base_fixture_bounds_2d,
                floor_fixture_bounds_2d,
                robot_size,
            )
            if valid[0]:
                out.append([x, y, theta])
                kept += 1
        if kept < count:
            # relax: allow boundary-only check (no-op if we hit max_tries) so we fill count
            while kept < count:
                r = rng.uniform(rmin, rmax)
                phi = rng.uniform(-np.pi, np.pi)
                x = ox + r * np.cos(phi)
                y = oy + r * np.sin(phi)
                dtheta = rng.normal(0.0, sigma)
                dtheta = float(np.clip(dtheta, -np.pi, np.pi))
                theta = otheta + dtheta
                theta = np.arctan2(np.sin(theta), np.cos(theta))
                out.append([x, y, theta])
                kept += 1
    return np.array(out[:num_samples], dtype=np.float64)


def navigate_and_snap(env, pose_xy_theta, k_col=None, n_settle_steps=50, snap=True):
    """Use Mobi-π's move_to_pose navigation + optional final snap.

    Mirrors the exact sequence used by eval_mobipi.py (nav with arm held at
    reset_joint_pos, then set_robot_base_pose + 50-step physics settle), so the
    BC-Transformer policy sees the same pre-rollout state it was trained from.
    """
    from robocasa.utils.env_utils import set_robot_base_pose
    unwrapped = env.unwrapped.env

    if k_col is None:
        k_col = _DummyCollisionChecker(robot_size=0.5)

    x, y, theta = float(pose_xy_theta[0]), float(pose_xy_theta[1]), float(pose_xy_theta[2])
    target_pose_mat = PoseUtils.make_pose(
        np.array([x, y, 0.0]),
        T.quat2mat(T.axisangle2quat(np.array([0, 0, theta]))),
    )
    _move_to_pose(
        env, target_pose_mat,
        render=False, verbose=False, use_rrt=True, k_col=k_col,
    )

    if snap:
        set_robot_base_pose(unwrapped, [x, y, theta])
        unwrapped.sim.data.qvel[:] = 0.0
        unwrapped.sim.forward()
        for _ in range(n_settle_steps):
            unwrapped.sim.step()


def current_base_pose(unwrapped):
    pos = unwrapped.sim.data.get_body_xpos("mobilebase0_base")
    mat = unwrapped.sim.data.get_body_xmat("mobilebase0_base").reshape(3, 3)
    heading = compute_robot_angle_from_mat(mat)
    return np.array([float(pos[0]), float(pos[1]), float(heading)], dtype=np.float64)


def render_ego_views(env, camera_names, image_h=224, image_w=224):
    """Render each ego camera into an HxWx3 uint8 RGB image."""
    images = {}
    for cam in camera_names:
        _, _, rgb, _ = render_image_with_robot_mask(env, image_h, image_w, camera_name=cam)
        images[cam] = np.ascontiguousarray(rgb).astype(np.uint8)
    return images


def render_ego_depths(env, camera_names, image_h=224, image_w=224):
    """Render each ego camera into an HxW float32 depth image."""
    sim = env.unwrapped.env.sim
    depths = {}
    for cam in camera_names:
        d = sim.render(camera_name=cam, width=image_w, height=image_h, depth=True)
        if isinstance(d, tuple): d = d[1]
        depths[cam] = np.ascontiguousarray(d[::-1]).astype(np.float32)
    return depths


@click.command()
@click.option("--env_name", required=True, type=str)
@click.option("--policy_name", default="bc_xfmr", type=str)
@click.option("--layout_ids", default="0,2,3,5,6", type=str,
              help="Comma-separated layout ids (style id == layout id). "
                   "Ignored if --scene_pairs is given.")
@click.option("--scene_pairs", default=None, type=str,
              help="Explicit (layout,style) pairs as e.g. "
                   "'1:0,1:1,2:0,2:1,...'. Decouples style from layout for "
                   "diversity scaling. Each shard is named "
                   "layout{L}_style{S}_seed{seed}.hdf5.")
@click.option("--randomize_cameras/--no_randomize_cameras", default=False,
              help="Pass randomize_cameras to the env. When true the agent "
                   "view cams jitter per-reset, broadening visual diversity "
                   "without changing the scene layout.")
@click.option("--num_candidates", default=30, type=int,
              help="Candidate rollouts per layout.")
@click.option("--seed", default=1, type=int)
@click.option("--policy_seed", default=None, type=int,
              help="Seed used to pick the bc_xfmr checkpoint. Defaults to "
                   "--seed. Set this when --seed is outside the range of "
                   "trained seeds (e.g. --seed=4 for pose variety with "
                   "--policy_seed=1 to reuse an existing ckpt).")
@click.option("--ckpt_root_dir", default=POLICY_CKPT_ROOT_DIR, type=str)
@click.option("--data_root_dir", default=DATA_ROOT_DIR, type=str)
@click.option("--out_dir", default="data/moflo_rollouts", type=str)
@click.option("--image_size", default=224, type=int)
@click.option("--num_settle_steps", default=50, type=int)
@click.option("--trials_per_candidate", default=1, type=int,
              help="Re-roll each candidate N times; record each per-trial success "
                   "and store the majority-vote label as 'success'. Denoises "
                   "BC-policy label noise at the cost of N× collection time.")
@click.option("--n2m_sampling/--no_n2m_sampling", default=False,
              help="Sample candidates uniformly in a 0.4×0.4 m box + ±15° "
                   "heading around oracle (matches the N2M paper's sampling).")
@click.option("--dry_run", is_flag=True,
              help="Only 3 candidates per layout; skip rollout, save dummy success=-1.")
@click.option("--synthesize_only_oracle", is_flag=True,
              help="Skip both candidate sampling and policy rollouts. Emit "
                   "exactly one rollout per scene at the oracle pose with "
                   "success=1 (assumed). Used for v23-style scene-diversity "
                   "scaling: enumerate (layout, style) combos cheaply, then "
                   "let viewpoint_augment.py expand with viewpoint augs.")
@click.option("--oracle_override_json", default=None, type=str,
              help="Path to JSON {task: {layout_id: [x, y, theta]}} overriding "
                   "the metadata oracle with empirical oracle from training data.")
@click.option("--n2m_xy_half", default=0.2, type=float,
              help="N2M sampling box half-width when n2m_sampling is True.")
@click.option("--n2m_heading_sigma", default=0.262, type=float,
              help="N2M sampling heading std (rad) when n2m_sampling is True.")
def main(
    env_name,
    policy_name,
    layout_ids,
    scene_pairs,
    randomize_cameras,
    num_candidates,
    seed,
    policy_seed,
    ckpt_root_dir,
    data_root_dir,
    out_dir,
    image_size,
    num_settle_steps,
    trials_per_candidate,
    n2m_sampling,
    dry_run,
    synthesize_only_oracle,
    oracle_override_json,
    n2m_xy_half,
    n2m_heading_sigma,
):
    if policy_seed is None:
        policy_seed = seed
    if scene_pairs is not None:
        scene_list = []
        for tok in scene_pairs.split(","):
            l, s = tok.strip().split(":")
            scene_list.append((int(l), int(s)))
        layout_id_list = [l for l, _ in scene_list]
    else:
        layout_id_list = [int(s.strip()) for s in layout_ids.split(",")]
        scene_list = [(l, l) for l in layout_id_list]

    out_task_dir = os.path.join(out_dir, env_name)
    os.makedirs(out_task_dir, exist_ok=True)

    log_path = os.path.join(out_task_dir, f"collect_seed{seed}.log")
    file_stream = open(log_path, "a")
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)

    print(f"\n=== MoFlo data collection: {env_name} seed={seed} "
          f"layouts={layout_id_list} num_candidates={num_candidates} "
          f"dry_run={dry_run} {datetime.now().isoformat()} ===")

    # Load policy once; shared across layouts.
    config, ckpt_path = get_config_for_policy(
        ckpt_root_dir, data_root_dir, env_name, policy_name,
        seed=policy_seed, dataset_name="mg-300",
    )
    if synthesize_only_oracle:
        rollout_model = None
        print("[synthesize_only_oracle] skipping policy load")
    else:
        _, rollout_model, _ = load_policy(config, ckpt_path)

    camera_names = TASK_CAMERA_NAMES.get(
        env_name, ["robot0_agentview_right"],
    )
    ego_cam = camera_names[0]

    for layout_id, style_id in scene_list:
        t_layout = time.time()
        out_h5 = os.path.join(
            out_task_dir, f"layout{layout_id}_style{style_id}_seed{seed}.hdf5",
        )
        if os.path.exists(out_h5):
            print(f"[skip] {out_h5} already exists.")
            continue

        print(f"\n--- layout {layout_id} style {style_id} ---")
        try:
            env_meta_list, _ = get_metadata(config)
        except Exception as e:
            print(f"[skip] (layout {layout_id}, style {style_id}) invalid: {repr(e)}")
            continue
        base_camera_names = env_meta_list[0]["env_kwargs"]["camera_names"]
        all_cameras = list(dict.fromkeys(
            list(base_camera_names) + camera_names + ["robot0_navview"]
        ))
        # place_robot_for_nav=True mirrors eval_mobipi.py: env.reset() puts the
        # robot at a sampled nav pose with the arm in the "home" configuration
        # the BC policy was trained from. We snapshot the arm qpos in the
        # per-candidate loop, teleport the base to the candidate, and hold the
        # arm at that snapshot during settle so the policy sees the expected
        # arm state at rollout start.
        override_args = {
            "layout_and_style_ids": [[layout_id, style_id]],
            "place_robot_for_nav": True,
            "camera_names": all_cameras,
            "hard_reset": True,
            "randomize_cameras": randomize_cameras,
        }
        # Cache env map once; do NOT re-create an env for each candidate below.
        try:
            (
                base_fixture_bounds_2d,
                floor_fixture_bounds_2d,
                default_robot_pos,
                default_robot_ori,
                robot_size,
            ) = get_env_map_and_default_robot_init_pose(
                config, override_args, save_dir=None,
            )
        except Exception as e:
            print(f"[skip] (layout {layout_id}, style {style_id}) env map failed: {repr(e)}")
            continue
        oracle_pose = np.array(
            [default_robot_pos[0], default_robot_pos[1], default_robot_ori[-1]],
            dtype=np.float64,
        )

        # Optional empirical-oracle override
        if oracle_override_json is not None:
            import json as _json
            _table = _json.load(open(oracle_override_json))
            if env_name in _table and str(layout_id) in _table[env_name]:
                _emp = _table[env_name][str(layout_id)]
                oracle_pose = np.array(_emp, dtype=np.float64)
                print(f"[oracle-override] {env_name} L{layout_id}: using empirical oracle {oracle_pose}")
            else:
                print(f"[oracle-override] no entry for {env_name} L{layout_id}; using metadata oracle.")

        rng = np.random.RandomState(seed * 10000 + layout_id)
        if synthesize_only_oracle:
            candidates = np.array([oracle_pose], dtype=np.float64)
        else:
            n = 3 if dry_run else num_candidates
            candidates = sample_candidate_poses(
                oracle_pose,
                n,
                base_fixture_bounds_2d,
                floor_fixture_bounds_2d,
                robot_size,
                rng,
                n2m_style=n2m_sampling,
                n2m_xy_half=n2m_xy_half,
                n2m_heading_sigma=n2m_heading_sigma,
            )

        # Open HDF5 shard; each candidate is one group.
        with h5py.File(out_h5, "w") as f:
            f.attrs["env_name"] = env_name
            f.attrs["layout_id"] = layout_id
            f.attrs["style_id"] = style_id
            f.attrs["seed"] = seed
            f.attrs["oracle_pose"] = oracle_pose
            f.attrs["camera_names"] = np.array(camera_names, dtype=h5py.string_dtype())
            f.attrs["image_size"] = image_size
            rollouts_grp = f.create_group("rollouts")

            for ci, cand in enumerate(candidates):
                t0 = time.time()
                trial_successes = []
                achieved_pose = None
                ego_images = None
                task_lang = ""
                obj_name = ""
                for trial_idx in range(max(1, trials_per_candidate)):
                    try:
                        env = load_env(config, override_args)
                        env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(
                            seed * 10000 + layout_id * 1000 + ci * 10 + trial_idx
                        )
                        env.reset()
                        unwrapped = env.unwrapped.env

                        if trial_idx == 0:
                            task_lang = unwrapped.get_ep_meta().get("lang", "")
                            obj_name = unwrapped.get_object_name()

                        navigate_and_snap(env, cand, n_settle_steps=num_settle_steps)

                        if trial_idx == 0:
                            achieved_pose = current_base_pose(unwrapped)
                            ego_images = render_ego_views(env, camera_names, image_size, image_size)
                            try:
                                ego_depths = render_ego_depths(env, camera_names, image_size, image_size)
                            except Exception as _e:
                                print(f"[warn] depth render failed: {_e}"); ego_depths = None

                        if dry_run:
                            trial_successes.append(-1)
                            break
                        elif synthesize_only_oracle:
                            trial_successes.append(1)
                            break
                        else:
                            all_rollout_logs, _ = TrainUtils.rollout_with_stats(
                                policy=rollout_model,
                                envs=[env],
                                horizon=[config.experiment.rollout.horizon],
                                use_goals=config.use_goals,
                                num_episodes=1,
                                render=False,
                                video_dir=None,
                                epoch=f"moflo_collect_c{ci}_t{trial_idx}",
                                video_skip=config.experiment.get("video_skip", 5),
                                terminate_on_success=config.experiment.rollout.terminate_on_success,
                                reset_before_rollout=False,
                                del_envs_after_rollouts=True,
                                data_logger=None,
                            )
                            trial_successes.append(
                                int(all_rollout_logs[env.name]["Success_Rate"] > 0.5)
                            )
                    except Exception as e:
                        print(f"[warn] candidate {ci} trial {trial_idx} failed: {repr(e)}")
                        try:
                            del env
                        except Exception:
                            pass
                        continue

                if len(trial_successes) == 0 or achieved_pose is None:
                    print(f"[warn] candidate {ci} produced no usable trials; skipping")
                    continue

                # Majority-vote label; also store the full trial vector.
                if dry_run:
                    success = -1
                    success_rate = -1.0
                elif synthesize_only_oracle:
                    success = 1
                    success_rate = 1.0
                else:
                    success_rate = float(sum(trial_successes)) / len(trial_successes)
                    success = int(success_rate > 0.5)

                rgrp = rollouts_grp.create_group(f"{ci:04d}")
                rgrp.create_dataset("ego_view", data=ego_images[ego_cam])
                for cam in camera_names[1:]:
                    rgrp.create_dataset(f"ego_view_{cam}", data=ego_images[cam])
                if ego_depths is not None:
                    rgrp.create_dataset("ego_depth", data=ego_depths[ego_cam])
                    for cam in camera_names[1:]:
                        rgrp.create_dataset(f"ego_depth_{cam}", data=ego_depths[cam])
                rgrp.create_dataset("candidate_pose", data=cand)
                rgrp.create_dataset("achieved_pose", data=achieved_pose)
                rgrp.create_dataset("oracle_pose", data=oracle_pose)
                rgrp.create_dataset(
                    "trial_successes",
                    data=np.array(trial_successes, dtype=np.int8),
                )
                rgrp.attrs["success"] = success
                rgrp.attrs["success_rate"] = success_rate
                rgrp.attrs["num_trials"] = len(trial_successes)
                rgrp.attrs["language"] = task_lang
                rgrp.attrs["object_name"] = obj_name
                rgrp.attrs["ego_camera"] = ego_cam
                print(
                    f"  cand {ci+1}/{len(candidates)}: "
                    f"pose={np.round(achieved_pose, 3).tolist()} "
                    f"trials={trial_successes} rate={success_rate:.2f} "
                    f"label={success} dt={time.time() - t0:.1f}s"
                )

        print(f"[layout {layout_id}] wrote {out_h5} in {time.time() - t_layout:.1f}s")

    print(f"\n=== done {env_name} seed={seed} at {datetime.now().isoformat()} ===")


if __name__ == "__main__":
    main()
