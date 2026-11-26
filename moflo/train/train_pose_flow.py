"""Train PoseFlowPredictor — conditional flow matching for SE(2) base placement.

Same data + backbone as train_pose_delta.py, but the head learns a velocity
field instead of a 1-step residual. Straight-line (rectified-flow) CFM:

    x_0      = current_pose          (the n2m start pose)
    x_1      = oracle pose
    t ~ U(0,1)
    x_t      = (1 - t) x_0 + t x_1
    v_target = x_1 - x_0
    L        = w * || v_theta(x_t, t, c) - v_target ||^2

Validation reports val_xy_err by ODE-integrating from x_0 (n_steps Euler).
"""
import os, sys, math, click, json, numpy as np, torch
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from mobipi.utils.io_utils import DualStream
from moflo.models.pose_flow_predictor import PoseFlowPredictor
from moflo.train.dataset_posedelta import PoseDeltaDataset, pose_delta_collate


def cosine_lr(step, total, base, final):
    if total <= 0: return base
    t = min(max(step/total, 0.0), 1.0)
    return final + 0.5*(base-final)*(1+math.cos(math.pi*t))


@click.command()
@click.option("--feature_dirs", required=True)
@click.option("--oracles_json", required=True)
@click.option("--out_dir", required=True)
@click.option("--epochs", default=120, type=int)
@click.option("--batch_size", default=128, type=int)
@click.option("--num_workers", default=4, type=int)
@click.option("--base_lr", default=1e-4, type=float)
@click.option("--final_lr", default=1e-6, type=float)
@click.option("--weight_decay", default=0.01, type=float)
@click.option("--grad_clip", default=1.0, type=float)
@click.option("--val_frac", default=0.1, type=float)
@click.option("--d_model", default=384, type=int)
@click.option("--n_blocks", default=8, type=int)
@click.option("--n_heads", default=8, type=int)
@click.option("--ffn_mult", default=4, type=int)
@click.option("--dropout", default=0.1, type=float)
@click.option("--fail_weight", default=0.3, type=float)
@click.option("--n_steps_eval", default=10, type=int, help="Euler steps for val integration")
@click.option("--sigma_path", default=0.0, type=float,
              help="optional Gaussian jitter on x_t (0 = pure rectified flow)")
@click.option("--target_mode", type=click.Choice(["oracle", "awm_basin"]), default="oracle",
              help="oracle: transport always to the empirical oracle (plain CFM). "
                   "awm_basin: advantage-weighted -- successful candidates are their "
                   "own target (zero velocity, 'stay in the basin'); failed candidates "
                   "transport to the oracle. Success weighted higher.")
@click.option("--awm_succ_weight", default=1.0, type=float,
              help="loss weight for success tuples in awm_basin mode")
@click.option("--param_mode", type=click.Choice(["velocity", "x1"]), default="velocity",
              help="velocity: regress the CFM velocity field. "
                   "x1: regress the endpoint x1_hat (loss in pose-space, more "
                   "stable; at t=0 reduces to direct regression).")
@click.option("--loss_space", type=click.Choice(["auto", "pose", "velocity"]), default="auto",
              help="space the L2 loss is taken in. "
                   "auto: pose-loss in x1 mode, velocity-loss in velocity mode (the existing diagonals). "
                   "pose: ||x1_hat - x1||^2. "
                   "velocity: ||v_pred - v_target||^2 where v_target = x1 - x0. "
                   "x1-param + velocity-loss isolates the parameterization from the implicit "
                   "(1-t)^2 timestep weighting of x1-param + pose-loss; gradients are upweighted "
                   "near t=1 (denominator 1/(1-t) clamped at 1e-3).")
@click.option("--t_beta_a", default=1.0, type=float,
              help="Beta(a, b) shape for timestep sampling. Defaults a=b=1 -> Uniform(0,1). "
                   "a<b concentrates mass near t=0 (e.g., Beta(1,4) has mean 0.2, mode 0). "
                   "Useful with velocity-loss: combined with the 1/(1-t)^2 weighting, the "
                   "effective pose-space weight becomes (a+b)*(1-t)^(b-1)*t^(a-1)/(1-t)^2 -- "
                   "Beta(1,4) gives 4(1-t), aligning loss emphasis with deployment-time small-t.")
@click.option("--t_beta_b", default=1.0, type=float,
              help="Beta(a, b) shape b parameter; see --t_beta_a.")
@click.option("--seed", default=42, type=int)
def main(feature_dirs, oracles_json, out_dir, epochs, batch_size, num_workers,
         base_lr, final_lr, weight_decay, grad_clip, val_frac,
         d_model, n_blocks, n_heads, ffn_mult, dropout, fail_weight,
         n_steps_eval, sigma_path, target_mode, awm_succ_weight, param_mode,
         loss_space, t_beta_a, t_beta_b, seed):
    os.makedirs(out_dir, exist_ok=True)
    file_stream = open(os.path.join(out_dir, "train.log"), "a")
    sys.stdout = DualStream(sys.__stdout__, file_stream)
    if loss_space == "auto":
        loss_space = "pose" if param_mode == "x1" else "velocity"
    print(f"=== PoseFlow (CFM) training {datetime.utcnow().isoformat()} "
          f"param_mode={param_mode} loss_space={loss_space} "
          f"t_dist=Beta({t_beta_a},{t_beta_b}) "
          f"sigma_path={sigma_path} n_steps_eval={n_steps_eval} ===")
    torch.manual_seed(seed); np.random.seed(seed)
    use_beta = (t_beta_a != 1.0) or (t_beta_b != 1.0)
    t_beta = torch.distributions.Beta(torch.tensor(float(t_beta_a)),
                                      torch.tensor(float(t_beta_b))) if use_beta else None

    feat_dirs = [d.strip() for d in feature_dirs.split(",") if d.strip()]
    ds = PoseDeltaDataset(feat_dirs, oracles_json, fail_weight=fail_weight)
    n_val = int(len(ds)*val_frac); n_train = len(ds)-n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=pose_delta_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=pose_delta_collate)
    sample = train_ds[0]
    visual_dim = sample[0].shape[-1]; language_dim = sample[1].shape[-1]
    pose_dim = sample[2].shape[-1]
    print(f"visual_dim={visual_dim} language_dim={language_dim} pose_dim={pose_dim}")

    model = PoseFlowPredictor(visual_dim=visual_dim, language_dim=language_dim,
                              pose_dim=pose_dim, d_model=d_model, n_blocks=n_blocks,
                              n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
                              param_mode=param_mode).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    def awm_endpoint(CP, T, W):
        """In awm_basin mode the per-sample flow endpoint x1 is:
          - CP itself if the candidate succeeded (W ~ 1.0) -> zero velocity, "stay"
          - the oracle T if it failed -> transport into the basin.
        Returns (x1, loss_weight)."""
        if target_mode == "oracle":
            return T, W
        is_succ = (W >= 0.999)
        x1 = torch.where(is_succ[:, None], CP, T)
        # Success tuples define the basin -> weight them up.
        lw = torch.where(is_succ, torch.full_like(W, awm_succ_weight), W)
        return x1, lw

    total_steps = epochs * max(1, len(train_loader))
    step = 0; best_val = 1e9; history = []
    for epoch in range(1, epochs+1):
        model.train(); loss_sum = 0; n_b = 0
        for batch in train_loader:
            V, L, cond_mask, CP, T, W = batch
            V = V.cuda(); L = L.cuda(); CP = CP.cuda(); T = T.cuda(); W = W.cuda()
            cond_mask = cond_mask.cuda()
            B = CP.shape[0]
            X1, LW = awm_endpoint(CP, T, W)
            if use_beta:
                t = t_beta.sample((B,)).to(CP.device).clamp(1e-4, 1 - 1e-4)
            else:
                t = torch.rand(B, device=CP.device)
            x_t = (1 - t[:, None]) * CP + t[:, None] * X1
            if sigma_path > 0:
                # Variance-preserving jitter that vanishes at both endpoints
                scale = sigma_path * torch.sqrt(t * (1 - t))[:, None]
                x_t = x_t + scale * torch.randn_like(x_t)
            if loss_space == "pose":
                x1_pred = model.predict_x1(x_t, t, V, L, kv_mask=cond_mask)
                err = ((x1_pred - X1) ** 2).sum(dim=-1)
            else:
                v_target = X1 - CP
                v_pred = model(x_t, t, V, L, kv_mask=cond_mask)
                err = ((v_pred - v_target) ** 2).sum(dim=-1)
            loss = (err * LW).sum() / (LW.sum() + 1e-6)
            lr = cosine_lr(step, total_steps, base_lr, final_lr)
            for g in opt.param_groups: g["lr"] = lr
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            loss_sum += loss.item(); n_b += 1; step += 1
        train_loss = loss_sum/max(1, n_b)

        # Validation: integrate the ODE and measure end-pose error
        model.eval(); v_loss = 0; v_n = 0
        v_xy_sum = 0; v_head_sum = 0; v_n_samp = 0
        with torch.no_grad():
            for batch in val_loader:
                V, L, cond_mask, CP, T, W = batch
                V = V.cuda(); L = L.cuda(); CP = CP.cuda(); T = T.cuda(); W = W.cuda()
                cond_mask = cond_mask.cuda()
                B = CP.shape[0]
                X1, LW = awm_endpoint(CP, T, W)
                if use_beta:
                    t = t_beta.sample((B,)).to(CP.device).clamp(1e-4, 1 - 1e-4)
                else:
                    t = torch.rand(B, device=CP.device)
                x_t = (1 - t[:, None]) * CP + t[:, None] * X1
                if loss_space == "pose":
                    x1_pred = model.predict_x1(x_t, t, V, L, kv_mask=cond_mask)
                    err = ((x1_pred - X1) ** 2).sum(dim=-1)
                else:
                    v_target = X1 - CP
                    v_pred = model(x_t, t, V, L, kv_mask=cond_mask)
                    err = ((v_pred - v_target) ** 2).sum(dim=-1)
                v_loss += ((err * LW).sum() / (LW.sum() + 1e-6)).item(); v_n += 1
                # End-pose error via integration (measured vs oracle: a known
                # basin point, so a valid checkpoint-selection metric for both modes)
                x_end = model.integrate(CP, V, L, n_steps=n_steps_eval,
                                        kv_mask=cond_mask)
                xy_err = (((x_end[:, :2] - T[:, :2]) * 2.5) ** 2).sum(dim=-1).sqrt()
                v_xy_sum += xy_err.sum().item(); v_n_samp += xy_err.shape[0]
                cos_diff = (x_end[:, 2]*T[:, 2] + x_end[:, 3]*T[:, 3]).clamp(-1, 1)
                v_head_sum += (torch.acos(cos_diff) * 180/3.14159).sum().item()
        val_loss = v_loss/max(1, v_n)
        val_xy_err = v_xy_sum/max(1, v_n_samp)
        val_head_err = v_head_sum/max(1, v_n_samp)
        print(f"[ep {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_xy_err={val_xy_err*100:.1f}cm val_head_err={val_head_err:.1f}deg", flush=True)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            val_xy_err_m=val_xy_err, val_head_err_deg=val_head_err))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        # Select on integrated end-pose error (the metric we actually deploy)
        if val_xy_err < best_val:
            best_val = val_xy_err
            torch.save({"model": model.state_dict(),
                        "args": dict(visual_dim=visual_dim, language_dim=language_dim,
                                     pose_dim=pose_dim, d_model=d_model, n_blocks=n_blocks,
                                     n_heads=n_heads, ffn_mult=ffn_mult, dropout=dropout,
                                     n_steps_eval=n_steps_eval, sigma_path=sigma_path,
                                     target_mode=target_mode, awm_succ_weight=awm_succ_weight,
                                     param_mode=param_mode, loss_space=loss_space,
                                     t_beta_a=t_beta_a, t_beta_b=t_beta_b)},
                       os.path.join(out_dir, "ckpt_best.pt"))
    print(f"=== done. best_val_xy_err={best_val*100:.2f}cm ===")


if __name__ == "__main__":
    main()
