#!/usr/bin/env python
from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "skimage", "pywt", "cv2",
    "PIL", "pydicom", "torch", "torchvision", "timm", "yaml", "tqdm",
    "matplotlib",
]

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
for name in PACKAGES:
    try:
        m = importlib.import_module(name)
        version = getattr(m, "__version__", "installed")
        print(f"{name:12s} {version}")
    except Exception as exc:
        print(f"{name:12s} ERROR: {exc}")

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
except Exception:
    pass
