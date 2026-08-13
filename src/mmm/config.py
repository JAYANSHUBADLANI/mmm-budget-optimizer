"""Configuration loading.

I load config/config.yaml once and expose it as a dict like object with attribute
access, so call sites read as cfg.channels.paid rather than cfg["channels"]["paid"].
Paths are resolved against the repository root so scripts work from any directory.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root, which is two levels above this file's package."""
    return Path(__file__).resolve().parents[2]


class DotDict(dict):
    """Dict that also supports attribute access, recursively.

    I use this instead of a dataclass because the config grows over the four phases
    and I did not want to keep editing a schema class every time I added a key.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return _wrap(value)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested key with a dotted string, for example 'model.bayesian.draws'."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return _wrap(node)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return DotDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike[str] | None = None) -> DotDict:
    """Load and cache the YAML config."""
    cfg_path = Path(path) if path is not None else repo_root() / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = DotDict(raw)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(relative: str) -> Path:
    """Resolve a config path value against the repo root and create parent folders."""
    target = repo_root() / relative
    target.mkdir(parents=True, exist_ok=True)
    return target


def paid_channels(cfg: DotDict | None = None) -> list[str]:
    cfg = cfg or load_config()
    return list(cfg["channels"]["paid"])


def all_media_columns(cfg: DotDict | None = None) -> list[str]:
    """Paid channels plus organic controls, which is every media volume column."""
    cfg = cfg or load_config()
    return list(cfg["channels"]["paid"]) + list(cfg["channels"]["organic_controls"])


def cpm_rates(cfg: DotDict | None = None) -> dict[str, float]:
    """Return the assumed cost per thousand impressions for each channel.

    These are assumptions I chose and documented, not measured costs. See
    docs/cpm_assumptions.md.
    """
    cfg = cfg or load_config()
    return {k: float(v["value"]) for k, v in cfg["cpm"].items()}
