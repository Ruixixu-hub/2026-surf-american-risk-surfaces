"""Windows CUDA, float64, and DeepONet contraction preflight."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from american_risk_surfaces.deeponet.model import PositivePremiumDeepONet


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; formal DeepONet jobs must not start")
    model = PositivePremiumDeepONet(32).to(device="cuda", dtype=torch.float64)
    branch = torch.randn((2, 4), device="cuda", dtype=torch.float64)
    trunk = torch.randn((14280, 2), device="cuda", dtype=torch.float64)
    output = model(branch, trunk)
    coordinate = torch.randn((257, 2), device="cuda", dtype=torch.float64, requires_grad=True)
    branch_one = model.encode_branch(branch[:1])
    raw = model.contract(branch_one, model.encode_trunk(coordinate))
    first = torch.autograd.grad(raw.sum(), coordinate, create_graph=True)[0][:, 0]
    second = torch.autograd.grad(first.sum(), coordinate)[0][:, 0]
    torch.cuda.synchronize()
    if output.shape != (2, 14280) or not torch.all(torch.isfinite(output)):
        raise RuntimeError("float64 Cartesian-product DeepONet preflight failed")
    if not torch.all(torch.isfinite(first)) or not torch.all(torch.isfinite(second)):
        raise RuntimeError("float64 DeepONet AD Greek preflight failed")
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True, check=True, text=True,
        ).stdout.strip()
    except Exception as exc:
        driver = f"unavailable: {exc}"
    payload = {
        "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0), "nvidia_smi": driver,
        "float64_deeponet_shape": list(output.shape),
        "float64_second_coordinate_derivative": "PASS", "status": "PASS",
    }
    output_dir = Path("results/12_positive_premium_deeponet/00_protocol")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "windows_hardware_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
