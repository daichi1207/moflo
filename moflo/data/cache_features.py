"""
CLIP / SigLIP feature caching for MoFlo.

Reads augmented HDF5 shards (from viewpoint_augment.py), runs the vision
+ text towers of a CLIP-family model on each ego_view + language
paraphrase, and writes cached tokens to disk.

Per task we emit a *single* feature HDF5 per shard:

    data/moflo_features/<env_name>/layout{L}_style{S}_seed{seed}.hdf5
        tokens/<rollout>/<aug>/visual     (N_v, d_v)   float16
                            /language    (N_l, d_l)   float16
                            /pose_delta_local (4,)    float32
                            /success    ()            int8

d_v and d_l depend on the model (CLIP-base: 768 / 512; SigLIP-base: 768 / 768).

Language tokens are picked per-sample. With ``--positional_lang_prob > 0`` we
mix in the env's positional ``@language`` string (e.g. "close the left
drawer", "turn on the rear left burner") so the scorer/flow can learn to
disambiguate within-task. Otherwise we fall back to a round-robin over the
``paraphrases`` from ``moflo/configs/tasks.yaml``.

Usage:
    python -m moflo.data.cache_features \\
        --env_name TurnOnMicrowave \\
        --in_dir data/moflo_augmented \\
        --out_dir data/moflo_features \\
        --model_name openai/clip-vit-base-patch16

Note: the transformers version shipped in the shared venv is 4.36, which
predates ``model_type=siglip``; SigLIP weights cannot be loaded directly.
We default to CLIP here and keep the SigLIP branch behind the model_name
(will error at load time if transformers is too old).
"""
import os
import sys
import click
import h5py
import time
import yaml
import numpy as np
import torch
from datetime import datetime
from PIL import Image

from mobipi.utils.io_utils import DualStream

_TASKS_YAML = os.path.join(os.path.dirname(__file__), "..", "configs", "tasks.yaml")


def load_paraphrases(env_name, paraphrase_set="paraphrases"):
    """Load paraphrases for a task.

    paraphrase_set:
      - "paraphrases" (default): the training paraphrases under tasks[env_name].paraphrases
      - "held_out_paraphrases": held-out paraphrases under top-level
        held_out_paraphrases[env_name]; used for the paraphrase-generalization
        eval in Section 5.6 of the paper. Falls back to training paraphrases
        if no held-out set is defined for the env.
    """
    with open(_TASKS_YAML, "r") as f:
        cfg = yaml.safe_load(f)
    if paraphrase_set == "held_out_paraphrases":
        ho = cfg.get("held_out_paraphrases", {})
        if env_name in ho and ho[env_name]:
            return ho[env_name]
        # Fallback: warn-on-stdout and use training paraphrases.
        print(f"[load_paraphrases] WARNING: no held_out_paraphrases for {env_name}, "
              f"falling back to training paraphrases")
    tasks = cfg["tasks"]
    if env_name not in tasks:
        return [env_name.lower()]
    return tasks[env_name]["paraphrases"]


class SiglipEncoder:
    """Wrap a CLIP-family HF model (CLIP / SigLIP) for visual + language tokens.

    Uses patch/subword tokens from the respective towers (pre-pooling) via
    ``last_hidden_state``. Returned as float16 numpy arrays for storage.

    Despite the class name (kept for back-compat with existing imports), this
    now defaults to CLIP because transformers 4.36 in the venv does not
    include SigLIP support. Pass any SigLIP checkpoint to try SigLIP if you
    upgrade transformers later.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch16", device="cuda",
                 dtype=torch.float32):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.dtype = dtype
        print(f"[SiglipEncoder] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device).eval()
        print(f"[SiglipEncoder] Ready on {device}.")

    @torch.no_grad()
    def encode_image(self, rgb_np):
        """rgb_np: (H, W, 3) uint8. Returns (N_v, d) float16 numpy."""
        img = Image.fromarray(rgb_np)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        vision = self.model.vision_model(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            output_hidden_states=False,
            return_dict=True,
        )
        feats = vision.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats

    @torch.no_grad()
    def encode_text(self, text):
        """text: str. Returns (N_l, d) float16 numpy."""
        # CLIP's processor auto-caps to 77 context; SigLIP uses max_length padding.
        inputs = self.processor(
            text=[text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        text_kwargs = dict(input_ids=inputs["input_ids"],
                           output_hidden_states=False,
                           return_dict=True)
        if "attention_mask" in inputs:
            text_kwargs["attention_mask"] = inputs["attention_mask"]
        text_out = self.model.text_model(**text_kwargs)
        feats = text_out.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats


class DinoV2Encoder:
    """DINOv2 visual + CLIP text. Use for v6 encoder-isolation experiment.

    DINOv2 gives stronger geometric/spatial features than CLIP ViT because
    it is self-supervised on natural images (not text-aligned), so patch
    tokens encode more geometry per patch. Standard drop-in: 257 patches
    × 768 d for dinov2-base, same shape/dim as CLIP ViT-B/16.

    Text branch falls back to the CLIP model so we don't have to swap the
    language encoder too — visual is what we're testing.
    """

    def __init__(self, dino_name="facebook/dinov2-base",
                 text_name="openai/clip-vit-base-patch16",
                 device="cuda", dtype=torch.float32):
        from transformers import AutoModel, AutoImageProcessor, AutoProcessor
        self.device = device
        self.dtype = dtype
        print(f"[DinoV2Encoder] Loading visual {dino_name}...")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_name)
        self.dino = AutoModel.from_pretrained(
            dino_name, torch_dtype=dtype,
        ).to(device).eval()
        print(f"[DinoV2Encoder] Loading text {text_name}...")
        self.clip_processor = AutoProcessor.from_pretrained(text_name)
        self.clip = AutoModel.from_pretrained(
            text_name, torch_dtype=dtype,
        ).to(device).eval()
        print(f"[DinoV2Encoder] Ready on {device}.")

    @torch.no_grad()
    def encode_image(self, rgb_np):
        img = Image.fromarray(rgb_np)
        inputs = self.dino_processor(images=img, return_tensors="pt").to(self.device)
        out = self.dino(pixel_values=inputs["pixel_values"].to(self.dtype))
        feats = out.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.clip_processor(
            text=[text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        text_kwargs = dict(input_ids=inputs["input_ids"],
                           output_hidden_states=False,
                           return_dict=True)
        if "attention_mask" in inputs:
            text_kwargs["attention_mask"] = inputs["attention_mask"]
        text_out = self.clip.text_model(**text_kwargs)
        feats = text_out.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats


class DinoSiglipEncoder:
    """DINOv2 visual + SigLIP text (via open_clip). Use for v14 ablation.

    Motivation: SigLIP text is jointly trained with visual for contrastive
    alignment on 4 B web image-text pairs, so its text tokens should be
    more task-aware than CLIP's. We keep DINOv2 for visual because it gave
    us the geometric features needed to localize per-layout oracles.

    Returns DINOv2 patch tokens (257 × 768) for images and SigLIP
    subword tokens (64 × 768) for text.
    """

    def __init__(self, dino_name="facebook/dinov2-base",
                 siglip_clip_name="ViT-B-16-SigLIP",
                 siglip_pretrained="webli",
                 device="cuda", dtype=torch.float32):
        from transformers import AutoModel, AutoImageProcessor
        import open_clip
        self.device = device
        self.dtype = dtype
        print(f"[DinoSiglipEncoder] Loading visual {dino_name}...")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_name)
        self.dino = AutoModel.from_pretrained(
            dino_name, torch_dtype=dtype,
        ).to(device).eval()
        print(f"[DinoSiglipEncoder] Loading text {siglip_clip_name} "
              f"({siglip_pretrained})...")
        self.siglip, _, _ = open_clip.create_model_and_transforms(
            siglip_clip_name, pretrained=siglip_pretrained,
        )
        self.siglip = self.siglip.to(device).eval()
        self.siglip_tokenizer = open_clip.get_tokenizer(siglip_clip_name)
        print(f"[DinoSiglipEncoder] Ready on {device}.")

    @torch.no_grad()
    def encode_image(self, rgb_np):
        img = Image.fromarray(rgb_np)
        inputs = self.dino_processor(images=img, return_tensors="pt").to(self.device)
        out = self.dino(pixel_values=inputs["pixel_values"].to(self.dtype))
        feats = out.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats

    @torch.no_grad()
    def encode_text(self, text):
        """text: str → (N_l=64, d=768) float16 SigLIP subword tokens."""
        tokens = self.siglip_tokenizer([text]).to(self.device)
        text_mod = self.siglip.text
        x = text_mod.token_embedding(tokens)
        x = x + text_mod.positional_embedding
        x = x.permute(1, 0, 2)
        x = text_mod.transformer(x, attn_mask=text_mod.attn_mask)
        x = x.permute(1, 0, 2)
        x = text_mod.ln_final(x)
        return x[0].to(torch.float16).cpu().numpy()


class MaeEncoder:
    """MAE-base visual + CLIP text. Used as the EXP-D encoder swap in v40.

    facebook/vit-mae-base is a masked-autoencoder ViT-B/16 (768 d, 197 tokens
    at full input). We set mask_ratio=0 so the encoder returns all 197 patch
    tokens during inference (the default 0.75 mask ratio is for pre-training,
    not feature extraction).

    Text branch reuses CLIP, matching the DINOv2 setup, so the only thing we
    swap is the visual encoder (the ablation we want to isolate).
    """

    def __init__(self, mae_name="facebook/vit-mae-base",
                 text_name="openai/clip-vit-base-patch16",
                 device="cuda", dtype=torch.float32):
        from transformers import AutoModel, AutoImageProcessor, AutoProcessor
        self.device = device
        self.dtype = dtype
        print(f"[MaeEncoder] Loading visual {mae_name} (mask_ratio=0)...")
        self.mae_processor = AutoImageProcessor.from_pretrained(mae_name)
        self.mae = AutoModel.from_pretrained(
            mae_name, torch_dtype=dtype,
        ).to(device).eval()
        # Disable random masking so we keep all 197 tokens at inference.
        self.mae.config.mask_ratio = 0.0
        print(f"[MaeEncoder] Loading text {text_name}...")
        self.clip_processor = AutoProcessor.from_pretrained(text_name)
        self.clip = AutoModel.from_pretrained(
            text_name, torch_dtype=dtype,
        ).to(device).eval()
        print(f"[MaeEncoder] Ready on {device}.")

    @torch.no_grad()
    def encode_image(self, rgb_np):
        img = Image.fromarray(rgb_np)
        inputs = self.mae_processor(images=img, return_tensors="pt").to(self.device)
        out = self.mae(pixel_values=inputs["pixel_values"].to(self.dtype))
        feats = out.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats

    @torch.no_grad()
    def encode_text(self, text):
        inputs = self.clip_processor(
            text=[text], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        text_kwargs = dict(input_ids=inputs["input_ids"],
                           output_hidden_states=False,
                           return_dict=True)
        if "attention_mask" in inputs:
            text_kwargs["attention_mask"] = inputs["attention_mask"]
        text_out = self.clip.text_model(**text_kwargs)
        feats = text_out.last_hidden_state[0].to(torch.float16).cpu().numpy()
        return feats


def build_encoder(model_name, device="cuda"):
    """Factory:
    - model_name starting with 'dinov2-siglip' → DinoSiglipEncoder (DINOv2 visual + SigLIP text)
    - containing 'dinov2'                      → DinoV2Encoder (DINOv2 visual + CLIP text)
    - containing 'vit-mae' or 'mae-base'        → MaeEncoder (MAE-base visual + CLIP text; v40 EXP-D)
    - else                                     → SiglipEncoder (CLIP-family from transformers)
    """
    name = model_name.lower()
    if "dinov2-siglip" in name or "dino-siglip" in name or name.startswith("siglip:"):
        return DinoSiglipEncoder(device=device)
    if "dinov2" in name or "dino-v2" in name:
        return DinoV2Encoder(dino_name=model_name, device=device)
    if "vit-mae" in name or "mae-base" in name:
        return MaeEncoder(mae_name=model_name, device=device)
    return SiglipEncoder(model_name=model_name, device=device)


@click.command()
@click.option("--env_name", required=True, type=str)
@click.option("--in_dir", default="data/moflo_augmented", type=str)
@click.option("--out_dir", default="data/moflo_features", type=str)
@click.option("--model_name", default="facebook/dinov2-base", type=str)
@click.option("--device", default="cuda", type=str)
@click.option("--positional_lang_prob", default=0.0, type=float,
              help="Probability of using the env's positional @language string "
                   "(e.g. 'close the left drawer') instead of a paraphrase. "
                   "0.0 = pure paraphrase round-robin (legacy v15/v20 behavior). "
                   "1.0 = always positional. 0.5 = 50/50 mix that retains "
                   "paraphrase diversity while teaching positional grammar.")
@click.option("--seed", default=0, type=int,
              help="RNG seed for the positional vs paraphrase coin flip.")
def main(env_name, in_dir, out_dir, model_name, device, positional_lang_prob, seed):
    in_task_dir = os.path.join(in_dir, env_name)
    out_task_dir = os.path.join(out_dir, env_name)
    os.makedirs(out_task_dir, exist_ok=True)

    log_path = os.path.join(out_task_dir, "cache.log")
    file_stream = open(log_path, "a")
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)

    print(f"\n=== MoFlo feature caching: {env_name} "
          f"model={model_name} positional_lang_prob={positional_lang_prob} "
          f"{datetime.now().isoformat()} ===")

    paraphrases = load_paraphrases(env_name)
    print(f"Paraphrases ({len(paraphrases)}): {paraphrases}")

    encoder = build_encoder(model_name, device=device)

    # Pre-compute text features for paraphrases (tiny, one per paraphrase).
    text_cache = [encoder.encode_text(p) for p in paraphrases]
    # Lazy-cache for positional strings discovered in @language attrs.
    positional_text_cache = {}

    def _get_positional_text(s):
        if s not in positional_text_cache:
            positional_text_cache[s] = encoder.encode_text(s)
        return positional_text_cache[s]

    shard_files = sorted([f for f in os.listdir(in_task_dir) if f.endswith(".hdf5")])
    print(f"Found {len(shard_files)} shards.")

    rng = np.random.RandomState(seed)

    for shard in shard_files:
        in_h5 = os.path.join(in_task_dir, shard)
        out_h5 = os.path.join(out_task_dir, shard)
        if os.path.exists(out_h5):
            print(f"[skip] {out_h5} exists.")
            continue
        t_shard = time.time()
        n_tuples = 0
        n_pos = 0
        with h5py.File(in_h5, "r") as src, h5py.File(out_h5, "w") as dst:
            dst.attrs["env_name"] = env_name
            dst.attrs["layout_id"] = int(src.attrs["layout_id"])
            dst.attrs["style_id"] = int(src.attrs["style_id"])
            dst.attrs["seed"] = int(src.attrs["seed"])
            dst.attrs["model_name"] = model_name
            dst.attrs["paraphrases"] = np.array(paraphrases, dtype=h5py.string_dtype())
            dst.attrs["positional_lang_prob"] = float(positional_lang_prob)
            tok_grp = dst.create_group("tokens")
            tuples_grp = src["tuples"]
            for rkey in sorted(tuples_grp.keys()):
                rsrc = tuples_grp[rkey]
                rdst = tok_grp.create_group(rkey)
                rdst.attrs["success"] = int(rsrc.attrs["success"])
                positional_lang = str(rsrc.attrs["language"])
                rdst.attrs["language"] = positional_lang
                for ai, akey in enumerate(sorted([k for k in rsrc.keys()])):
                    asrc = rsrc[akey]
                    rgb = asrc["ego_view"][()]
                    visual = encoder.encode_image(rgb)
                    use_positional = (
                        positional_lang_prob > 0.0
                        and len(positional_lang) > 0
                        and rng.random_sample() < positional_lang_prob
                    )
                    if use_positional:
                        text_feats = _get_positional_text(positional_lang)
                        para_idx = -1
                        n_pos += 1
                    else:
                        para_idx = ai % len(paraphrases)
                        text_feats = text_cache[para_idx]
                    agrp = rdst.create_group(akey)
                    if "pose_abs_world" in asrc:
                        agrp.create_dataset(
                            "pose_abs_world",
                            data=np.asarray(asrc["pose_abs_world"][()], dtype=np.float32),
                        )
                    agrp.create_dataset("visual", data=visual, compression="gzip", compression_opts=3)
                    agrp.create_dataset("language", data=text_feats, compression="gzip", compression_opts=3)
                    agrp.create_dataset(
                        "pose_delta_local",
                        data=np.asarray(asrc["pose_delta_local"][()], dtype=np.float32),
                    )
                    agrp.attrs["success"] = int(asrc.attrs["success"])
                    agrp.attrs["paraphrase_idx"] = int(para_idx)
                    agrp.attrs["used_positional_lang"] = int(use_positional)
                    n_tuples += 1
            dst.attrs["total_tuples"] = n_tuples
            dst.attrs["positional_lang_count"] = n_pos
        print(f"[{shard}] {n_tuples} tuples ({n_pos} positional, "
              f"{n_tuples - n_pos} paraphrase) in {time.time() - t_shard:.1f}s")

    print(f"=== done {env_name} at {datetime.now().isoformat()} ===")


if __name__ == "__main__":
    main()
