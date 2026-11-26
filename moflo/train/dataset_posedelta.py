"""Dataset for delta pose prediction.

Returns (visual, language, current_pose, target_pose, weight).
- current_pose: the pose at which the visual was rendered (= candidate pose in v19_pointbert)
- target_pose: oracle pose (normalized)
- weight: 1.0 if success else fail_weight

The model should learn: given visual at current_pose, what's the offset to oracle?
This works because the OFFSET (oracle - current) varies across samples (since current varies),
so the model gets diverse offset training signal.
"""
import h5py, glob, os, json, numpy as np, torch
from torch.utils.data import Dataset


def w2n_world(xy, t):
    return np.array([(xy[0]-2.5)/2.5, (xy[1]+2.5)/2.5, np.cos(t), np.sin(t)], dtype=np.float32)


class PoseDeltaDataset(Dataset):
    def __init__(self, feature_dirs, oracles_json_path, fail_weight=0.3):
        self.oracles = json.load(open(oracles_json_path))
        self.fail_weight = fail_weight
        self.index = []
        for d in feature_dirs:
            task = os.path.basename(d.rstrip("/"))
            if task not in self.oracles:
                print(f"[warn] no oracles for {task}; skip"); continue
            shard_paths = sorted(glob.glob(os.path.join(d, "*.hdf5")))
            for sp in shard_paths:
                with h5py.File(sp, "r") as f:
                    L = int(f.attrs.get("layout_id", -1))
                    if str(L) not in self.oracles[task]: continue
                    oracle_world = self.oracles[task][str(L)]
                    oracle_norm = w2n_world(oracle_world[:2], oracle_world[2])
                    if "tokens" not in f: continue
                    for rkey in f["tokens"]:
                        rgroup = f[f"tokens/{rkey}"]
                        for akey in rgroup:
                            agroup = rgroup[akey]
                            if "pose_abs_world" not in agroup: continue
                            succ = int(agroup.attrs.get("success", 0))
                            self.index.append((sp, rkey, akey, succ, oracle_norm))
        print(f"[PoseDeltaDataset] {len(self.index)} tuples")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        sp, rkey, akey, succ, oracle_norm = self.index[idx]
        with h5py.File(sp, "r") as f:
            g = f[f"tokens/{rkey}/{akey}"]
            visual = np.array(g["visual"][()], dtype=np.float32)
            language = np.array(g["language"][()], dtype=np.float32)
            current_pose = np.array(g["pose_abs_world"][()], dtype=np.float32)  # 4D normalized
        weight = 1.0 if succ else self.fail_weight
        return (
            torch.from_numpy(visual),
            torch.from_numpy(language),
            torch.from_numpy(current_pose),
            torch.from_numpy(oracle_norm),
            torch.tensor(weight, dtype=torch.float32),
        )


def pose_delta_collate(batch):
    visuals, languages, cur_poses, targets, weights = zip(*batch)
    N_v = max(v.shape[0] for v in visuals)
    N_l = max(l.shape[0] for l in languages)
    d_v = visuals[0].shape[1]; d_l = languages[0].shape[1]
    B = len(batch)
    V = torch.zeros(B, N_v, d_v); L = torch.zeros(B, N_l, d_l)
    V_mask = torch.ones(B, N_v, dtype=torch.bool); L_mask = torch.ones(B, N_l, dtype=torch.bool)
    for i, (v, l) in enumerate(zip(visuals, languages)):
        V[i, :v.shape[0]] = v; V_mask[i, :v.shape[0]] = False
        L[i, :l.shape[0]] = l; L_mask[i, :l.shape[0]] = False
    cond_mask = torch.cat([V_mask, L_mask], dim=1)
    CP = torch.stack(cur_poses, dim=0)
    T = torch.stack(targets, dim=0)
    W = torch.stack(weights, dim=0)
    return V, L, cond_mask, CP, T, W
