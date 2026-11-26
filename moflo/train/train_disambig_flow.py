"""Train the x1-flow on disambiguation data.

Same model (PoseFlowPredictor, param_mode=x1) and same CFM x1-prediction loss
as train_pose_flow.py. The ONLY difference is the dataset: 2-drawer scenes
where the left/right token determines the target. No architecture change.

Usage:
  python -m moflo.train.train_disambig_flow \\
      --data_dir data/disambig_flow_features \\
      --out_dir logs_moflo/pose_flow_x1_disambig
"""
import os, sys, math, glob, json, click, numpy as np, torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from mobipi.utils.io_utils import DualStream
from moflo.models.pose_flow_predictor import PoseFlowPredictor


class DisambigFlowDataset(Dataset):
    """Reads the per-(layout,side) HDF5 shards from collect_disambig_flow_data.
    Returns (visual, language, pose_cur, pose_tgt, weight=1.0)."""
    def __init__(self, data_dir):
        import h5py
        self.shards = []
        self.index = []
        for sp in sorted(glob.glob(os.path.join(data_dir, "*.hdf5"))):
            with h5py.File(sp, "r") as f:
                n = f["visual"].shape[0]
            self.shards.append(sp)
            for i in range(n):
                self.index.append((sp, i))
        print(f"[DisambigFlowDataset] {len(self.index)} tuples "
              f"from {len(self.shards)} shards")
        self._cache = {}

    def _get(self, sp):
        import h5py
        if sp not in self._cache:
            self._cache[sp] = h5py.File(sp, "r")
        return self._cache[sp]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sp, i = self.index[idx]
        f = self._get(sp)
        visual = np.asarray(f["visual"][i], dtype=np.float32)
        language = np.asarray(f["language"][i], dtype=np.float32)
        cur = np.asarray(f["pose_cur"][i], dtype=np.float32)
        tgt = np.asarray(f["pose_tgt"][i], dtype=np.float32)
        return (torch.from_numpy(visual), torch.from_numpy(language),
                torch.from_numpy(cur), torch.from_numpy(tgt),
                torch.tensor(1.0, dtype=torch.float32))


def collate(batch):
    vs, ls, cps, tgs, ws = zip(*batch)
    Nv = max(v.shape[0] for v in vs); Nl = max(l.shape[0] for l in ls)
    dv, dl = vs[0].shape[1], ls[0].shape[1]
    B = len(batch)
    V = torch.zeros(B, Nv, dv); L = torch.zeros(B, Nl, dl)
    Vm = torch.ones(B, Nv, dtype=torch.bool); Lm = torch.ones(B, Nl, dtype=torch.bool)
    for i, (v, l) in enumerate(zip(vs, ls)):
        V[i, :v.shape[0]] = v; Vm[i, :v.shape[0]] = False
        L[i, :l.shape[0]] = l; Lm[i, :l.shape[0]] = False
    cond_mask = torch.cat([Vm, Lm], dim=1)
    return V, L, cond_mask, torch.stack(cps), torch.stack(tgs), torch.stack(ws)


def cosine_lr(step, total, base, final):
    if total <= 0: return base
    t = min(max(step/total, 0.0), 1.0)
    return final + 0.5*(base-final)*(1+math.cos(math.pi*t))


@click.command()
@click.option("--data_dir", default="data/disambig_flow_features")
@click.option("--out_dir", default="runs/flow_disambig")
@click.option("--epochs", default=200, type=int)
@click.option("--batch_size", default=128, type=int)
@click.option("--base_lr", default=1e-4, type=float)
@click.option("--final_lr", default=1e-6, type=float)
@click.option("--weight_decay", default=0.01, type=float)
@click.option("--val_frac", default=0.1, type=float)
@click.option("--d_model", default=384, type=int)
@click.option("--n_blocks", default=8, type=int)
@click.option("--n_heads", default=8, type=int)
@click.option("--ffn_mult", default=4, type=int)
@click.option("--dropout", default=0.1, type=float)
@click.option("--n_steps_eval", default=10, type=int)
@click.option("--init_from", default="", type=str,
              help="optional warm-start checkpoint (e.g. the main x1-flow)")
@click.option("--loss_space", type=click.Choice(["pose", "velocity"]), default="pose",
              help="pose: ||x1_hat - x1||^2. velocity: ||(x1_hat - x_t)/(1-t) - (x1 - x0)||^2 (canonical CFM).")
@click.option("--t_beta_a", default=1.0, type=float,
              help="Beta(a, b) shape for timestep sampling. Defaults a=b=1 -> Uniform(0,1).")
@click.option("--t_beta_b", default=1.0, type=float,
              help="Beta(a, b) shape b parameter; see --t_beta_a.")
@click.option("--seed", default=42, type=int)
def main(data_dir, out_dir, epochs, batch_size, base_lr, final_lr, weight_decay,
         val_frac, d_model, n_blocks, n_heads, ffn_mult, dropout, n_steps_eval,
         init_from, loss_space, t_beta_a, t_beta_b, seed):
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = DualStream(sys.__stdout__, open(os.path.join(out_dir, "train.log"), "a"))
    print(f"=== Disambig x1-flow training {datetime.utcnow().isoformat()} "
          f"loss_space={loss_space} t_dist=Beta({t_beta_a},{t_beta_b}) ===")
    torch.manual_seed(seed); np.random.seed(seed)
    use_beta = (t_beta_a != 1.0) or (t_beta_b != 1.0)
    t_beta = torch.distributions.Beta(torch.tensor(float(t_beta_a)),
                                      torch.tensor(float(t_beta_b))) if use_beta else None

    ds = DisambigFlowDataset(data_dir)
    n_val = max(1, int(len(ds)*val_frac)); n_train = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                    collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
                    collate_fn=collate)
    s0 = train_ds[0]
    visual_dim, language_dim, pose_dim = s0[0].shape[-1], s0[1].shape[-1], s0[2].shape[-1]
    print(f"visual_dim={visual_dim} language_dim={language_dim} pose_dim={pose_dim}")

    model = PoseFlowPredictor(visual_dim=visual_dim, language_dim=language_dim,
                              pose_dim=pose_dim, d_model=d_model, n_blocks=n_blocks,
                              n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
                              param_mode="x1").cuda()
    if init_from and os.path.exists(init_from):
        ck = torch.load(init_from, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"warm-started from {init_from}")

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(tl))
    step = 0; best = 1e9; history = []
    for ep in range(1, epochs+1):
        model.train(); ls = 0; nb = 0
        for V, L, cm, CP, T, W in tl:
            V, L, cm, CP, T, W = V.cuda(), L.cuda(), cm.cuda(), CP.cuda(), T.cuda(), W.cuda()
            if use_beta:
                t = t_beta.sample((CP.shape[0],)).to(CP.device).clamp(1e-4, 1 - 1e-4)
            else:
                t = torch.rand(CP.shape[0], device=CP.device)
            x_t = (1-t[:, None])*CP + t[:, None]*T
            if loss_space == "pose":
                x1_pred = model.predict_x1(x_t, t, V, L, kv_mask=cm)
                err = ((x1_pred - T)**2).sum(-1)
            else:
                v_target = T - CP
                v_pred = model(x_t, t, V, L, kv_mask=cm)
                err = ((v_pred - v_target)**2).sum(-1)
            loss = (err*W).sum() / (W.sum()+1e-6)
            lr = cosine_lr(step, total_steps, base_lr, final_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls += loss.item(); nb += 1; step += 1
        # val: integrated end-pose error
        model.eval(); vxy = 0; vn = 0
        with torch.no_grad():
            for V, L, cm, CP, T, W in vl:
                V, L, cm, CP, T = V.cuda(), L.cuda(), cm.cuda(), CP.cuda(), T.cuda()
                xe = model.integrate(CP, V, L, n_steps=n_steps_eval, kv_mask=cm)
                e = (((xe[:, :2]-T[:, :2])*2.5)**2).sum(-1).sqrt()
                vxy += e.sum().item(); vn += e.shape[0]
        v_xy = vxy/max(1, vn)
        print(f"[ep {ep}/{epochs}] train_loss={ls/max(1,nb):.4f} val_xy_err={v_xy*100:.2f}cm", flush=True)
        history.append(dict(epoch=ep, train_loss=ls/max(1,nb), val_xy_err_m=v_xy))
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"), indent=2)
        if v_xy < best:
            best = v_xy
            torch.save({"model": model.state_dict(),
                        "args": dict(visual_dim=visual_dim, language_dim=language_dim,
                                     pose_dim=pose_dim, d_model=d_model, n_blocks=n_blocks,
                                     n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
                                     n_steps_eval=n_steps_eval, param_mode="x1",
                                     loss_space=loss_space,
                                     t_beta_a=t_beta_a, t_beta_b=t_beta_b)},
                       os.path.join(out_dir, "ckpt_best.pt"))
    print(f"=== done. best_val_xy_err={best*100:.2f}cm ===")


if __name__ == "__main__":
    main()
