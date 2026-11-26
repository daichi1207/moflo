"""Download the five downstream RoboCasa policies (bc_xfmr, seed 1; 1.1 GB) from the
public Mobi-pi release into ckpts/robocasa/bc_xfmr and check their SHA-256."""
import glob, hashlib, os, sys, urllib.request, zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mobipi.macros import POLICY_CKPT_ROOT_DIR

BASE = "https://download.cs.stanford.edu/juno/mobipi/pi/"
# task: (name used by the download, sha256 of model_epoch_1000.pth)
POLICIES = {
    "TurnOnMicrowave":  ("TurnOnMicrowave", "eae33e3dbfb4940b24249d78cc681f745904c0dc4154d74a719cd5c365b0bbe8"),
    "CloseDrawer":      ("CloseDrawer",     "e9dce6bf82b225ccdedcdfdde1b6915c0a033c12d04d7c2830a9a71b00c0de16"),
    "CloseSingleDoor":  ("CloseSingleDoor", "6cafee55eaf087a93b6e604d072da459c6200b15616f14c32120e29f32be9852"),
    "TurnOnSinkFaucet": ("TurnOnFaucet",    "0d212f6239b73beae5ec54a0ef0131fe2974e0b9bd57347599e3800a3956ab8c"),
    "TurnOnStove":      ("TurnOnStove",     "5006e920bff5763ff322729646240476cc4b3326e156e915129d50d6b6ab23e2"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


root = os.path.join(POLICY_CKPT_ROOT_DIR, "robocasa", "bc_xfmr")
os.makedirs(root, exist_ok=True)
for task, (name, want) in POLICIES.items():
    pattern = os.path.join(root, f"04-12-{task}", f"seed_1_{task}_mg-300", "*", "models", "model_epoch_1000.pth")
    if not glob.glob(pattern):
        z = os.path.join(root, f"{task}.zip")
        print(f"downloading {task} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE}04-12-{name}_seed_1_{task}_mg-300.zip", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        os.remove(z)
        if name != task:  # the faucet archive is named differently from its task
            os.rename(os.path.join(root, f"04-12-{name}"), os.path.join(root, f"04-12-{task}"))
    got = sha256(glob.glob(pattern)[0])
    if got != want:
        sys.exit(f"{task}: checksum mismatch ({got})")
    print(f"{task}: ok")
