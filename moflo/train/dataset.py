"""
PyTorch dataset of cached SigLIP features for MoFlo.

Loads HDF5 shards produced by cache_features.py and returns tuples of
  (visual_tokens, language_tokens, pose_delta_local, success)

Supports two modes via `split`:
  - 'success':   only tuples with success == 1 (for CFM warm-start).
  - 'all':       all tuples (for AWM fine-tuning / analysis).
"""
import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MoFloFeatureDataset(Dataset):
    def __init__(self, feature_dirs, split="success", target_mode="delta"):
        """
        feature_dirs: list of per-task directories containing layout*.hdf5 shards.
        split: 'success' or 'all'.
        target_mode: 'delta' (local Δ in query frame) or 'world' (absolute
                     pose in world frame; v5 N2M-style). 'world' requires
                     shards that carry a 'pose_abs_world' dataset.
        """
        assert split in ("success", "all")
        assert target_mode in ("delta", "world")
        self.split = split
        self.target_mode = target_mode
        self.index = []   # list of (shard_path, rkey, akey, success)
        for d in feature_dirs:
            shard_paths = sorted(glob.glob(os.path.join(d, "*.hdf5")))
            for sp in shard_paths:
                with h5py.File(sp, "r") as f:
                    tok = f["tokens"]
                    for rkey in tok.keys():
                        r = tok[rkey]
                        for akey in r.keys():
                            if akey in ("__phantom__",):
                                continue
                            if not akey.isdigit() and not (akey.replace("_", "").isalnum()):
                                continue
                            try:
                                succ = int(r[akey].attrs["success"])
                            except KeyError:
                                continue
                            if split == "success" and succ != 1:
                                continue
                            self.index.append((sp, rkey, akey, succ))
        if len(self.index) == 0:
            raise RuntimeError(
                f"No samples found for split={split} in {feature_dirs}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sp, rkey, akey, succ = self.index[idx]
        with h5py.File(sp, "r") as f:
            g = f[f"tokens/{rkey}/{akey}"]
            visual = np.array(g["visual"][()], dtype=np.float32)
            language = np.array(g["language"][()], dtype=np.float32)
            if self.target_mode == "world":
                if "pose_abs_world" not in g:
                    raise KeyError(
                        f"pose_abs_world missing in {sp}:{rkey}/{akey}; "
                        "re-run viewpoint_augment + cache_features from v5."
                    )
                pose = np.array(g["pose_abs_world"][()], dtype=np.float32)
            else:
                pose = np.array(g["pose_delta_local"][()], dtype=np.float32)
        return (
            torch.from_numpy(visual),
            torch.from_numpy(language),
            torch.from_numpy(pose),
            torch.tensor(succ, dtype=torch.float32),
        )


def moflo_collate(batch):
    """Pad variable-length token sequences to the max in the batch."""
    visuals, languages, poses, succs = zip(*batch)
    N_v = max(v.shape[0] for v in visuals)
    N_l = max(l.shape[0] for l in languages)
    d_v = visuals[0].shape[1]
    d_l = languages[0].shape[1]
    B = len(batch)
    V = torch.zeros(B, N_v, d_v)
    L = torch.zeros(B, N_l, d_l)
    V_mask = torch.ones(B, N_v, dtype=torch.bool)
    L_mask = torch.ones(B, N_l, dtype=torch.bool)
    for i, (v, l) in enumerate(zip(visuals, languages)):
        V[i, :v.shape[0]] = v
        V_mask[i, :v.shape[0]] = False
        L[i, :l.shape[0]] = l
        L_mask[i, :l.shape[0]] = False
    cond_mask = torch.cat([V_mask, L_mask], dim=1)  # True = PAD
    P = torch.stack(poses, dim=0)
    S = torch.stack(succs, dim=0)
    return V, L, cond_mask, P, S
