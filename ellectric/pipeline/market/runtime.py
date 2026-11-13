"""Runtime compatibility helpers for the approved CPU model stack."""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from typing import Any


def import_lightgbm() -> Any:
    """Import LightGBM, preloading an already-installed OpenMP runtime if needed."""
    try:
        return importlib.import_module("lightgbm")
    except OSError as error:
        if "libgomp.so.1" not in str(error) or os.name == "nt":
            raise

    candidates = []
    configured = os.environ.get("ELLECTRIC_OPENMP_LIBRARY")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path("/usr/local/lib/ollama/libgomp.so.1.0.0"),
            Path("/usr/lib/x86_64-linux-gnu/libgomp.so.1"),
            Path("/lib/x86_64-linux-gnu/libgomp.so.1"),
        ]
    )
    for library in candidates:
        if library.is_file():
            ctypes.CDLL(str(library), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            return importlib.import_module("lightgbm")
    raise RuntimeError(
        "LightGBM 需要 libgomp.so.1；可安装系统运行库，或设置 "
        "ELLECTRIC_OPENMP_LIBRARY 指向已存在的 libgomp.so.1 文件。"
    )
