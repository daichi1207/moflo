"""N2M-style binary scorer training.

Trains PoseScorer to predict success/failure on (visual, language,
pose) tuples drawn from feature shards. Uses split="all" so we get
both successful and failed candidates as positives/negatives.
"""
import os
import sys
import math
import time
import click
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, random_split

from mobipi.utils.io_utils import DualStream
from moflo.models.scorer import PoseScorer
from moflo.train.dataset import MoFloFeatureDataset, moflo_collate


def cosine_lr(step, total_steps, base_lr, final_lr):
    if total_steps <= 0:
        return base_lr
    t = min(max(step / total_steps, 0.0), 1.0)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t))


@click.command()
@click.option("--feature_dirs", required=True, type=str)
@click.option("--out_dir", default="logs_moflo/scorer", type=str)
@click.option("--epochs", default=50, type=int)
@click.option("--batch_size", default=128, type=int)
@click.option("--num_workers", default=4, type=int)
@click.option("--base_lr", default=1e-4, type=float)
@click.option("--final_lr", default=1e-5, type=float)
@click.option("--weight_decay", default=0.01, type=float)
@click.option("--grad_clip", default=1.0, type=float)
@click.option("--val_frac", default=0.1, type=float)
@click.option("--d_model", default=256, type=int)
@click.option("--n_blocks", default=6, type=int)
@click.option("--n_heads", default=8, type=int)
@click.option("--ffn_mult", default=4, type=int)
@click.option("--dropout", default=0.1, type=float)
@click.option("--mixed_precision", default="bf16",
              type=click.Choice(["none", "fp16", "bf16"]))
@click.option("--log_every", default=50, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--target_mode", default="world",
              type=click.Choice(["delta", "world"]),
              help="Pose representation. World is recommended for the scorer.")
def main(feature_dirs, out_dir, epochs, batch_size, num_workers, base_lr,
         final_lr, weight_decay, grad_clip, val_frac, d_model, n_blocks,
         n_heads, ffn_mult, dropout, mixed_precision, log_every, seed,
         target_mode):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    file_stream = open(log_path, "a")
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)

    print(f"\n=== MoFlo scorer training {datetime.now().isoformat()} ===")
    print(f"Args: {locals()}")
    torch.manual_seed(seed); np.random.seed(seed)

    feature_dir_list = [d.strip() for d in feature_dirs.split(",") if d.strip()]
    # split="all" so we get both pos and neg candidates
    dataset = MoFloFeatureDataset(feature_dir_list, split="all", target_mode=target_mode)
    print(f"Dataset size (all): {len(dataset)} target_mode={target_mode}")
    # Class balance check via __getitem__ over a small sample
    pos = neg = 0
    for i in range(min(2000, len(dataset))):
        _, _, _, s = dataset[i]
        if int(s.item()) == 1: pos += 1
        else: neg += 1
    print(f"sample class balance (first 2000): pos={pos} neg={neg} pos_frac={pos / max(1, pos+neg):.2%}")

    n_val = max(1, int(val_frac * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    effective_batch = min(batch_size, max(1, len(train_set)))
    train_loader = DataLoader(train_set, batch_size=effective_batch, shuffle=True,
                              num_workers=num_workers, collate_fn=moflo_collate,
                              drop_last=(len(train_set) >= effective_batch * 2),
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=moflo_collate,
                            drop_last=False, persistent_workers=num_workers > 0)

    V0, L0, _, _, _ = next(iter(train_loader))
    visual_dim = V0.shape[-1]; language_dim = L0.shape[-1]
    print(f"visual_dim={visual_dim} language_dim={language_dim}")

    model = PoseScorer(pose_dim=4, visual_dim=visual_dim, language_dim=language_dim,
                       d_model=d_model, n_blocks=n_blocks, n_heads=n_heads,
                       ffn_mult=ffn_mult, dropout=dropout).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PoseScorer params: {n_params / 1e6:.2f}M")

    # If positives are rare, weight them up.
    pos_weight = torch.tensor(max(1.0, neg / max(1, pos))).cuda()
    print(f"pos_weight={pos_weight.item():.2f}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    amp_dtype = {"none": None, "fp16": torch.float16, "bf16": torch.bfloat16}[mixed_precision]

    step = 0; best_val = float("inf"); history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train(); losses = []; correct = 0; total = 0
        for V, L, cmask, P, S in train_loader:
            V = V.cuda(non_blocking=True); L = L.cuda(non_blocking=True)
            cmask = cmask.cuda(non_blocking=True); P = P.cuda(non_blocking=True)
            S = S.cuda(non_blocking=True)
            lr = cosine_lr(step, total_steps, base_lr, final_lr)
            for pg in optim.param_groups: pg["lr"] = lr
            optim.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(V, L, P, cond_mask=cmask)
                    loss = bce(logits, S)
            else:
                logits = model(V, L, P, cond_mask=cmask)
                loss = bce(logits, S)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            losses.append(loss.item())
            with torch.no_grad():
                pred = (logits > 0).float()
                correct += (pred == S).sum().item(); total += S.numel()
            if step % log_every == 0:
                print(f"[ep {epoch}/{epochs} step {step}] lr={lr:.2e} loss={loss.item():.4f}")
            step += 1

        # val
        model.eval()
        v_losses = []; v_correct = 0; v_total = 0; v_pos_correct = 0; v_pos_total = 0
        with torch.no_grad():
            for V, L, cmask, P, S in val_loader:
                V = V.cuda(); L = L.cuda(); cmask = cmask.cuda(); P = P.cuda(); S = S.cuda()
                if amp_dtype is not None:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(V, L, P, cond_mask=cmask)
                        loss = bce(logits, S)
                else:
                    logits = model(V, L, P, cond_mask=cmask)
                    loss = bce(logits, S)
                v_losses.append(loss.item())
                pred = (logits > 0).float()
                v_correct += (pred == S).sum().item(); v_total += S.numel()
                v_pos_correct += ((pred == 1) & (S == 1)).sum().item()
                v_pos_total += (S == 1).sum().item()
        val_loss = float(np.mean(v_losses)) if v_losses else float("nan")
        train_loss = float(np.mean(losses))
        train_acc = correct / max(1, total)
        val_acc = v_correct / max(1, v_total)
        recall = v_pos_correct / max(1, v_pos_total)
        print(f"[ep {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2%} pos_recall={recall:.2%}")
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_acc=train_acc, val_acc=val_acc, pos_recall=recall))
        if val_loss < best_val:
            best_val = val_loss
            ckpt = dict(
                model=model.state_dict(),
                args=dict(pose_dim=4, visual_dim=visual_dim, language_dim=language_dim,
                          d_model=d_model, n_blocks=n_blocks, n_heads=n_heads,
                          ffn_mult=ffn_mult, dropout=dropout, target_mode=target_mode),
            )
            torch.save(ckpt, os.path.join(out_dir, "ckpt_best.pt"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"=== done. best_val={best_val:.4f} elapsed={time.time() - t_start:.1f}s ===")


if __name__ == "__main__":
    main()
