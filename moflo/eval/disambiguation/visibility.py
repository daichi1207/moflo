"""Geometric visibility check: is the prompted drawer in the robot's head camera
view at the current (navigated) base pose?

"Visible" = the target drawer's world position falls inside the
robot0_agentview_right camera frustum and is in front of the camera. This is the
camera the disambiguation flow consumes as its ego-centric observation, so it is
exactly "in the model's input view." Occlusion is not modelled (the kitchen
fixtures are in open view from the start pose), so this is an in-frustum test.
"""
import numpy as np


def target_drawer_visible(env, cam="robot0_agentview_right"):
    base = env.unwrapped.env
    drawer = base._target_drawer
    pos = np.asarray(drawer.pos, dtype=float)
    sim = base.sim
    cid = sim.model.camera_name2id(cam)
    cpos = np.asarray(sim.data.cam_xpos[cid], dtype=float)
    cmat = np.asarray(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
    fovy = float(sim.model.cam_fovy[cid]) * np.pi / 180.0
    p_cam = cmat.T @ (pos - cpos)          # world -> camera frame
    z = p_cam[2]                           # MuJoCo cam looks down -z
    if z >= 0:
        return False
    half = np.tan(fovy / 2.0)              # square (224x224) render aspect
    u, v = p_cam[0] / (-z), p_cam[1] / (-z)
    return bool(abs(u) <= half and abs(v) <= half)
