"""Data / checkpoint locations. Defaults live under the release root; each can be
overridden with the environment variable named below."""
import os

MOBIPI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POLICY_CKPT_ROOT_DIR = os.environ.get("MOFLO_POLICY_CKPT_ROOT", os.path.join(MOBIPI_DIR, "ckpts"))
LOG_ROOT_DIR = os.environ.get("MOFLO_LOG_ROOT", os.path.join(MOBIPI_DIR, "logs"))
DATA_ROOT_DIR = os.environ.get("MOFLO_DATA_ROOT", os.path.join(MOBIPI_DIR, "data"))
