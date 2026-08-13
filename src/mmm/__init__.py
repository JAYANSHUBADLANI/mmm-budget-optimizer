"""Hierarchical Bayesian marketing mix model with a constrained budget optimiser."""

__version__ = "0.4.0"

from .config import DotDict, load_config, paid_channels, repo_root

__all__ = ["DotDict", "load_config", "paid_channels", "repo_root", "__version__"]
