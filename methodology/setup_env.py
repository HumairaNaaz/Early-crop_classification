"""Install PyTorch (GPU build) and the other packages into this project's .venv.

Run inside the project's virtual environment; safe to run again (it skips what is already installed).
Uses only the standard library, so it runs before anything is installed.

The PyTorch build is chosen from the NVIDIA driver and the GPU generation:
    driver older than 528        -> CUDA 11.8 build
    RTX 50xx (compute cap. 12.x) -> CUDA 12.8 build
    everything else              -> CUDA 12.6 build
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "logs" / "setup.log"
# No version pin: the newest build on the index matches whatever Python version the .venv has
# (a fixed old version has no wheels for new Python releases).


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    with LOG.open("a", encoding="utf-8") as fh:
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
    return proc.wait()


def gpu_info() -> tuple[str, float, float] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    name, cc, driver = [s.strip() for s in out[0].split(",")]
    return name, float(cc), float(driver.split(".")[0])


def torch_ok() -> bool:
    code = "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def main() -> int:
    log(f"Python {sys.version.split()[0]} at {sys.executable}")
    try:
        import ssl  # noqa: F401  (pip needs it; broken in some Anaconda-based environments)
    except ImportError:
        log("ERROR: this Python cannot use SSL, so pip cannot download packages. "
            "Delete the .venv folder and install Python from python.org, then run setup again.")
        return 1

    info = gpu_info()
    if info is None:
        log("ERROR: nvidia-smi did not answer, so no NVIDIA GPU / driver was found. "
            "Install or update the NVIDIA driver and run setup again.")
        return 1
    name, cc, driver = info
    log(f"GPU: {name}, compute capability {cc}, driver {driver:.0f}")

    if torch_ok():
        log("PyTorch with CUDA is already installed.")
    else:
        if driver < 528:
            index = "cu118"
        elif cc >= 12.0:
            index = "cu128"
        else:
            index = "cu126"
        log(f"Installing PyTorch ({index} build, about 2.5 GB)...")
        rc = run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        rc = rc or run([sys.executable, "-m", "pip", "install", "torch",
                        "--index-url", f"https://download.pytorch.org/whl/{index}"])
        if rc != 0:
            log("ERROR: PyTorch installation failed (see above).")
            return 1
        if not torch_ok():
            log("ERROR: PyTorch is installed but cannot see the GPU.")
            return 1

    ver = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                         capture_output=True, text=True).stdout.strip()
    log(f"PyTorch {ver} sees the GPU.")
    rc = run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    if rc != 0:
        log("ERROR: installing requirements.txt failed (see above).")
        return 1
    log("Environment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
