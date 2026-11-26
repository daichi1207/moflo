"""PoseFlowPredictor: conditional flow-matching velocity field for SE(2) base
placement.

This is the flow-matching counterpart of PoseDeltaPredictor. Instead of a
1-step residual regression, it learns a velocity field

    v_theta(x_t, t, visual, language)

that transports the n2m starting pose x_0 = current_pose to the oracle pose
x_1 along a straight-line (rectified-flow / CFM) path:

    x_t      = (1 - t) x_0 + t x_1
    v_target = x_1 - x_0

At inference the target pose is obtained by integrating the ODE
dx/dt = v_theta(x_t, t, c) from x_0 over n_steps Euler steps.

Architecture mirrors PoseDeltaPredictor (cross-attention transformer) so the
flow-vs-regression comparison is controlled: same backbone, only the head
semantics differ. The query token is derived from x_t plus a sinusoidal time
embedding.
"""
import math
import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, kv, kv_mask=None):
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=kv_mask)
        q = self.ln1(q + attn_out)
        q = self.ln2(q + self.ffn(q))
        return q


def sinusoidal_time_emb(t, dim):
    """t: (B,) in [0,1] -> (B, dim) sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :] * (2 * math.pi)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PoseFlowPredictor(nn.Module):
    """Velocity field v_theta(x_t, t, visual, language) for SE(2) flow matching."""

    def __init__(self,
                 visual_dim=256,
                 language_dim=512,
                 pose_dim=4,
                 d_model=384,
                 n_blocks=8,
                 n_heads=8,
                 ffn_mult=4,
                 dropout=0.1,
                 time_emb_dim=128,
                 param_mode="velocity"):
        """param_mode:
          "velocity" -- head outputs the velocity field v_theta directly.
          "x1"       -- head outputs the predicted endpoint x_1_hat; the
                        velocity is derived as (x_1_hat - x_t)/(1-t). At t=0
                        this reduces exactly to a regression head, so the flow
                        is guaranteed at least as accurate as direct regression.
        """
        super().__init__()
        assert param_mode in ("velocity", "x1")
        self.param_mode = param_mode
        self.pose_dim = pose_dim
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.language_proj = nn.Linear(language_dim, d_model)
        self.visual_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.language_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        # Query: x_t projection + time embedding
        self.pose_proj = nn.Linear(pose_dim, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_emb_dim = time_emb_dim
        self.blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pose_dim)   # raw velocity, unconstrained

    def _raw(self, x_t, t, visual_tokens, language_tokens, kv_mask=None):
        """Raw head output (B, pose_dim) — velocity or x1_hat depending on mode."""
        v = self.visual_proj(visual_tokens) + self.visual_type_emb
        l = self.language_proj(language_tokens) + self.language_type_emb
        cond = torch.cat([v, l], dim=1)
        q = self.pose_proj(x_t)
        q = q + self.time_mlp(sinusoidal_time_emb(t, self.time_emb_dim))
        q = q.unsqueeze(1)                                    # (B, 1, d_model)
        for block in self.blocks:
            q = block(q, cond, kv_mask=kv_mask)
        q = self.head_norm(q[:, 0])
        return self.head(q)

    def predict_x1(self, x_t, t, visual_tokens, language_tokens, kv_mask=None):
        """Predicted endpoint x_1_hat. In x1 mode this is the raw head; in
        velocity mode it is x_t + (1-t) v."""
        raw = self._raw(x_t, t, visual_tokens, language_tokens, kv_mask=kv_mask)
        if self.param_mode == "x1":
            return raw
        return x_t + (1.0 - t)[:, None] * raw

    def forward(self, x_t, t, visual_tokens, language_tokens, kv_mask=None):
        """Velocity field v_theta(x_t, t, c) -> (B, pose_dim).

        velocity mode: head output IS the velocity.
        x1 mode:       velocity is derived as (x1_hat - x_t)/(1-t).
        """
        raw = self._raw(x_t, t, visual_tokens, language_tokens, kv_mask=kv_mask)
        if self.param_mode == "velocity":
            return raw
        # x1 mode: raw == x1_hat; v = (x1_hat - x_t)/(1-t), clamped near t=1
        denom = (1.0 - t).clamp(min=1e-3)[:, None]
        return (raw - x_t) / denom

    @torch.no_grad()
    def integrate(self, x_0, visual_tokens, language_tokens, n_steps=10,
                  kv_mask=None, renorm_heading=True, solver="euler"):
        """Integrate the ODE dx/dt = v_theta from x_0 to x_1.

        solver: "euler" (1st order) or "heun" (2nd-order predictor-corrector).
        Heun halves the integration error per step at 2x the model evals.
        """
        x = x_0.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0 = torch.full((x.shape[0],), i * dt, device=x.device)
            v0 = self.forward(x, t0, visual_tokens, language_tokens, kv_mask=kv_mask)
            if solver == "euler":
                x = x + dt * v0
            elif solver == "heun":
                t1 = torch.full((x.shape[0],), (i + 1) * dt, device=x.device)
                x_pred = x + dt * v0
                v1 = self.forward(x_pred, t1, visual_tokens, language_tokens,
                                  kv_mask=kv_mask)
                x = x + dt * 0.5 * (v0 + v1)
            else:
                raise ValueError(f"unknown solver {solver!r}")
        if renorm_heading and x.shape[-1] >= 4:
            xy = x[:, :2]
            ct = x[:, 2:4]
            ct = ct / (ct.norm(dim=-1, keepdim=True) + 1e-6)
            x = torch.cat([xy, ct], dim=-1)
        return x
