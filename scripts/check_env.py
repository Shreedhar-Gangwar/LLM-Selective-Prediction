"""Environment check — run this FIRST, before anything else.

Verifies the machine can actually run the project: Python version, PyTorch with CUDA,
the GPU and its free VRAM, and that the key libraries import. Prints a PASS/WARN/FAIL
summary and exits non-zero if a hard requirement is missing.

Usage:  python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import sys

MIN_FREE_VRAM_GB = 6.0  # a 4B model in 4-bit needs ~3-4 GB; leave headroom for KV cache


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    hard_fail = False
    print("Environment check")
    print("-" * 60)

    # Python
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        _ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _fail(f"Python {v.major}.{v.minor} — need >= 3.10")
        hard_fail = True

    # Core imports
    for mod in ("torch", "transformers", "datasets", "bitsandbytes"):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "unknown")
            _ok(f"{mod} {ver}")
        except Exception as e:  # noqa: BLE001
            _fail(f"could not import {mod}: {e}")
            if mod in ("torch", "transformers"):
                hard_fail = True

    # CUDA / GPU
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb, total_gb = free_b / 1e9, total_b / 1e9
            _ok(f"CUDA available — GPU: {name}")
            _ok(f"VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
            if free_gb < MIN_FREE_VRAM_GB:
                _warn(
                    f"only {free_gb:.1f} GB free; close other GPU apps or use a smaller "
                    f"model / shorter context. Need ~{MIN_FREE_VRAM_GB:.0f} GB comfortable."
                )
        else:
            _fail(
                "CUDA not available. On a laptop 4060 this usually means the CPU-only "
                "torch wheel is installed. Reinstall torch with a CUDA index-url "
                "(see SETUP.md), and on Windows run inside WSL2."
            )
            hard_fail = True
    except Exception as e:  # noqa: BLE001
        _fail(f"torch/CUDA check errored: {e}")
        hard_fail = True

    print("-" * 60)
    if hard_fail:
        print("RESULT: FAIL — fix the items above before running the project.")
        return 1
    print("RESULT: PASS — environment looks good. Next: python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
