"""Windows CUDA and float64 second-derivative preflight for formal PINN runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; formal PINN jobs must not start")
    x = torch.tensor([[0.3]], dtype=torch.float64, device="cuda", requires_grad=True)
    y = (x**3).sum()
    first = torch.autograd.grad(y, x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    torch.cuda.synchronize()
    if not torch.allclose(second, 6.0 * x.detach(), atol=1e-12):
        raise RuntimeError("float64 second-derivative CUDA preflight failed")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        driver = f"unavailable: {exc}"
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": driver,
        "float64_second_derivative": "PASS",
    }
    output = Path("results/08_pinn_gap/00_protocol")
    output.mkdir(parents=True, exist_ok=True)
    (output / "windows_hardware_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
