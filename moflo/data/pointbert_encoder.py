"""Point-BERT (OpenShape PPAT scaling=1) encoder for MoFlo.

Generates a point cloud from RGB-D + camera intrinsics, feeds through the
OpenShape pretrained PPAT (Point Patch Attention Transformer) — returns
patch tokens for cross-attention with the MoFlo scorer / flow.

Modified forward: returns ALL tokens (cls + 64 patches = 65 × 256) rather
than just the CLS token (which is what OpenShape's CLIP-aligned head needs).

Model code: https://github.com/Colin97/OpenShape_code (cloned by scripts/setup.sh).
Weights: OpenShape/openshape-pointbert-shapenet on Hugging Face (5M params).
"""
import os, sys, numpy as np, torch
import torch.nn as nn

# OpenShape model code: external/OpenShape_code (see scripts/setup.sh), or set
# OPENSHAPE_CODE_DIR.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POINTBERT_REPO = "OpenShape/openshape-pointbert-shapenet"
POINTBERT_REVISION = "53b3e13bcddf780ae524a8897f235d4c5ff59fbd"


def default_pointbert_ckpt():
    """Pinned PointBERT weights (61 MB, cached by huggingface_hub)."""
    if os.environ.get("MOFLO_POINTBERT_CKPT"):
        return os.environ["MOFLO_POINTBERT_CKPT"]
    from huggingface_hub import hf_hub_download
    return hf_hub_download(POINTBERT_REPO, "model.pt", revision=POINTBERT_REVISION)


def _resolve_openshape_src():
    p = os.environ.get("OPENSHAPE_CODE_DIR") or os.path.join(_ROOT, "external", "OpenShape_code")
    if not os.path.isfile(os.path.join(p, "src", "models", "ppat.py")):
        raise FileNotFoundError(
            f"OpenShape code not found at {p}. Run scripts/setup.sh or set OPENSHAPE_CODE_DIR.")
    return p


OPENSHAPE_SRC = _resolve_openshape_src()
sys.path.insert(0, os.path.join(OPENSHAPE_SRC, "src"))

# Camera intrinsics (robosuite default 45 deg vertical FOV at 224x224).
DEFAULT_FX = 270.4
DEFAULT_FY = 270.4
DEFAULT_CX = 112.0
DEFAULT_CY = 112.0
DEPTH_NEAR = 0.01
DEPTH_FAR = 8.0
N_POINTS = 2048   # PPAT expects ~2k points


def rgbd_to_point_cloud(rgb_np, depth_raw, n_points=N_POINTS,
                        fx=DEFAULT_FX, fy=DEFAULT_FY, cx=DEFAULT_CX, cy=DEFAULT_CY,
                        depth_near=DEPTH_NEAR, depth_far=DEPTH_FAR):
    """Back-project a 224x224 depth image to a point cloud, attach RGB.
    Returns xyz (N,3) float32 and features (N,6) = xyz + rgb_normalized."""
    d = np.asarray(depth_raw, dtype=np.float32)
    z = depth_near + (depth_far - depth_near) * d
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz_full = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)  # (HW, 3)
    rgb = np.asarray(rgb_np, dtype=np.float32).reshape(-1, 3) / 255.0  # (HW, 3)
    # Filter invalid depth (z too close to depth_far suggests no return)
    valid = (z.flatten() > depth_near + 0.01) & (z.flatten() < depth_far - 0.5)
    xyz_full = xyz_full[valid]; rgb = rgb[valid]
    if len(xyz_full) == 0:
        # Fallback: tile with zeros
        xyz_full = np.zeros((n_points, 3), dtype=np.float32)
        rgb = np.zeros((n_points, 3), dtype=np.float32)
    # Sample n_points (random subsampling — fast)
    if len(xyz_full) >= n_points:
        idx = np.random.choice(len(xyz_full), n_points, replace=False)
    else:
        idx = np.random.choice(len(xyz_full), n_points, replace=True)
    xyz = xyz_full[idx].astype(np.float32)
    rgb_pts = rgb[idx].astype(np.float32)
    features = np.concatenate([xyz, rgb_pts], axis=-1)  # (N, 6)
    return xyz, features


class PointBERTEncoder(nn.Module):
    """OpenShape PPAT scaling=1, modified to return all 65 tokens (cls+64).
    Frozen at inference (weights from pretrained shapenet ckpt)."""

    OPENSHAPE_SRC = OPENSHAPE_SRC

    def __init__(self, ckpt_path, device="cuda"):
        super().__init__()
        # Import PPAT directly to bypass models/__init__.py (loads Minkowski),
        # and stub out dgl.geometry.farthest_point_sampler with a torch FPS.
        import importlib.util, types
        dgl_stub = types.ModuleType("dgl")
        dgl_geom = types.ModuleType("dgl.geometry")
        def _torch_fps(xyz, npoint):
            """Vectorized farthest point sampling. xyz: (B, N, 3) → (B, npoint) indices."""
            B, N, _ = xyz.shape
            centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
            distance = torch.ones(B, N, device=xyz.device) * 1e10
            farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
            for i in range(npoint):
                centroids[:, i] = farthest
                centroid = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
                d = ((xyz - centroid) ** 2).sum(-1)
                mask = d < distance; distance[mask] = d[mask]
                farthest = torch.max(distance, dim=-1)[1]
            return centroids
        dgl_geom.farthest_point_sampler = _torch_fps
        dgl_stub.geometry = dgl_geom
        sys.modules["dgl"] = dgl_stub
        sys.modules["dgl.geometry"] = dgl_geom

        # Create a fake 'models' package so relative imports in ppat.py work,
        # but skip its real __init__.py (which loads Minkowski).
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [os.path.join(self.OPENSHAPE_SRC, "src", "models")]
        sys.modules["models"] = models_pkg
        spec_pn = importlib.util.spec_from_file_location(
            "models.pointnet_util", os.path.join(self.OPENSHAPE_SRC, "src", "models", "pointnet_util.py"))
        pnu = importlib.util.module_from_spec(spec_pn)
        sys.modules["models.pointnet_util"] = pnu
        spec_pn.loader.exec_module(pnu)

        spec_pp = importlib.util.spec_from_file_location(
            "models.ppat", os.path.join(self.OPENSHAPE_SRC, "src", "models", "ppat.py"))
        ppat_mod = importlib.util.module_from_spec(spec_pp)
        sys.modules["models.ppat"] = ppat_mod
        spec_pp.loader.exec_module(ppat_mod)
        PointPatchTransformer = ppat_mod.PointPatchTransformer
        # scaling=1 args from OpenShape ppat.make()
        self.ppat = PointPatchTransformer(
            dim=256, depth=6, heads=4, mlp_dim=1024,
            sa_dim=96, patches=64, prad=0.4, nsamp=256, in_dim=6,
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ck["state_dict"]
        ppat_sd = {k.replace("module.ppat.", ""): v for k, v in sd.items()
                   if k.startswith("module.ppat.")}
        missing, unexpected = self.ppat.load_state_dict(ppat_sd, strict=False)
        if missing: print(f"[PointBERT] WARN missing keys: {missing[:5]}")
        if unexpected: print(f"[PointBERT] WARN unexpected keys: {unexpected[:5]}")
        self.ppat.eval()
        for p in self.ppat.parameters(): p.requires_grad_(False)
        self.device = device

    def _forward_all_tokens(self, xyz, features):
        """Patched PPAT forward returning ALL tokens (cls + patches)."""
        import torch_redstone as rst
        m = self.ppat
        m.sa.npoint = m.patches  # eval mode, no patch dropout
        centroids, feature = m.sa(xyz, features)
        x = m.lift(torch.cat([centroids, feature], dim=1))
        x = rst.supercat([m.cls_token, x], dim=-2)
        centroids_pad = rst.supercat([centroids.new_zeros(1), centroids], dim=-1)
        centroid_delta = centroids_pad.unsqueeze(-1) - centroids_pad.unsqueeze(-2)
        x = m.transformer(x, centroid_delta)
        return x  # (B, 65, 256)

    @torch.no_grad()
    def encode(self, xyz_np, features_np):
        """xyz: (N,3) np.float32, features: (N,6) np.float32 →
        returns (65, 256) np.float16 token features."""
        xyz = torch.from_numpy(xyz_np[None]).to(self.device)  # (1, N, 3)
        feat = torch.from_numpy(features_np[None]).to(self.device)  # (1, N, 6)
        # PPAT expects xyz: (B, 3, N), features: (B, C, N) per Projected.forward
        xyz_t = xyz.transpose(-1, -2).contiguous()
        feat_t = feat.transpose(-1, -2).contiguous()
        tokens = self._forward_all_tokens(xyz_t, feat_t)
        return tokens[0].cpu().numpy().astype(np.float16)  # (65, 256)


class PointBERTRgbEncoder:
    """Composite: PointBERT (RGB-D point cloud) tokens + CLIP text tokens.
    Drop-in replacement for the rgbd encoder in cache_features schema."""

    def __init__(self, pointbert_ckpt=None, text_name="openai/clip-vit-base-patch16",
                 device="cuda"):
        from transformers import CLIPTokenizer, CLIPTextModelWithProjection
        self.pb = PointBERTEncoder(pointbert_ckpt or default_pointbert_ckpt(), device=device)
        self.clip_tok = CLIPTokenizer.from_pretrained(text_name)
        self.clip_txt = CLIPTextModelWithProjection.from_pretrained(text_name).to(device).eval()
        self.device = device

    def encode_image(self, rgb_np, depth_np=None):
        if depth_np is None:
            raise ValueError("PointBERT encoder requires depth")
        xyz, feat = rgbd_to_point_cloud(rgb_np, depth_np)
        return self.pb.encode(xyz, feat)  # (65, 256)

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.clip_tok(text, return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
        hidden = self.clip_txt.text_model(**inputs).last_hidden_state[0]
        return hidden.cpu().numpy().astype(np.float16)
