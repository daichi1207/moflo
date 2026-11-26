"""N2M-style pose scorer.

Replaces MoFlo's flow-matching parameterization with a discrete
binary classifier: given (ego visual tokens, language tokens,
candidate pose), predict whether the policy will succeed at that
pose. At inference we sample many candidate poses (N2M-style or
uniformly), score each, and pick the argmax.

Architecture mirrors FlowDenoiser (cross-attn over visual+language)
without the time conditioning — the scorer is time-free, just
scoring static (cond, pose) pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlockNoTime(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h, cond, cond_mask=None):
        # cross-attention from h (queries) to cond (keys/values)
        q = self.norm1(h)
        kv = cond
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=cond_mask, need_weights=False)
        h = h + self.drop(attn_out)
        h = h + self.drop(self.ffn(self.norm2(h)))
        return h


class PoseScorer(nn.Module):
    """Score candidate poses for likelihood of policy success.

    Input:
        visual_tokens   (B, N_v, visual_dim)
        language_tokens (B, N_l, language_dim)
        pose            (B, pose_dim) — candidate pose to score
    Output:
        logit (B,) — pre-sigmoid scalar, higher = more likely success
    """
    def __init__(
        self,
        pose_dim=4,
        visual_dim=768,
        language_dim=768,
        d_model=256,
        n_blocks=6,
        n_heads=8,
        ffn_mult=4,
        dropout=0.1,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.language_proj = nn.Linear(language_dim, d_model)
        self.visual_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.language_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            CrossAttnBlockNoTime(d_model, n_heads, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def _build_cond(self, visual_tokens, language_tokens):
        v = self.visual_proj(visual_tokens) + self.visual_type_emb
        l = self.language_proj(language_tokens) + self.language_type_emb
        return torch.cat([v, l], dim=1)

    def forward(self, visual_tokens, language_tokens, pose, cond_mask=None):
        cond = self._build_cond(visual_tokens, language_tokens)
        h = self.pose_embed(pose).unsqueeze(1)  # (B, 1, d_model)
        for blk in self.blocks:
            h = blk(h, cond, cond_mask=cond_mask)
        h = self.out_norm(h).squeeze(1)
        return self.out(h).squeeze(-1)
