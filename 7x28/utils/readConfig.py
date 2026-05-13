"""
utils/readConfig.py

Loads project config from configs/config.yaml (or environment variables).

Expected configs/config.yaml format::

    OPENAI_API_KEY:   "gsk_..."
    OPENAI_API_MODEL: "llama-3.1-8b-instant"
    OPENAI_API_BASE:  "https://api.groq.com/openai/v1"
    OPENAI_PROXY:     ""        # leave blank if not needed

Source: iLLM-TSC2 (utils/readConfig.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


def read_config(config_path: Optional[str] = None) -> dict:
    """
    Load and return the configuration dictionary.

    Search order
    ------------
    1. ``config_path`` argument (if provided)
    2. ``configs/config.yaml`` relative to the project root
    3. ``config.yaml`` in the current working directory
    4. ``config.yaml`` next to this file (utils/)
    5. Environment variables fallback

    Returns
    -------
    dict with keys: OPENAI_API_KEY, OPENAI_API_MODEL, OPENAI_API_BASE, OPENAI_PROXY
    """
    cfg: dict = {}

    candidates = []
    if config_path:
        candidates.append(Path(config_path))

    project_root = Path(__file__).resolve().parent.parent
    candidates += [
        project_root / "configs" / "config.yaml",
        Path.cwd() / "configs" / "config.yaml",
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent / "config.yaml",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    cfg.update(loaded)
            break

    # Fallback to environment variables
    for key in ("OPENAI_API_KEY", "OPENAI_API_MODEL", "OPENAI_API_BASE", "OPENAI_PROXY"):
        if not cfg.get(key):
            env_val = os.environ.get(key, "")
            if env_val:
                cfg[key] = env_val

    # Defaults
    cfg.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")
    cfg.setdefault("OPENAI_PROXY", "")

    # Validate required keys
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_API_MODEL") if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"Missing required config keys: {missing}\n"
            "Create configs/config.yaml with:\n"
            "  OPENAI_API_KEY:   'gsk-...'\n"
            "  OPENAI_API_MODEL: 'llama-3.1-8b-instant'\n"
            "  OPENAI_API_BASE:  'https://api.groq.com/openai/v1'\n"
            "  OPENAI_PROXY:     ''\n"
        )

    return cfg
