#!/usr/bin/env python3
"""Download the three datasets.

Run this once after cloning. Nothing else in the pipeline reaches out to the network,
apart from the optional enrichment fetchers, which cache their results.

  python scripts/fetch_data.py              downloads everything it can
  python scripts/fetch_data.py --primary    primary Kaggle dataset only
  python scripts/fetch_data.py --benchmark  Robyn RData only

Credentials. The two Kaggle datasets need an API token. Create one at
https://www.kaggle.com/settings, then either place kaggle.json at ~/.kaggle/kaggle.json or
set KAGGLE_USERNAME and KAGGLE_KEY in your .env file. Without them the Kaggle steps are
skipped with a clear message rather than failing halfway through.

If a download is unavailable, run scripts/make_replica_dataset.py to generate a schema
faithful stand in so the rest of the pipeline still runs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm.config import load_config, repo_root  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("fetch_data")


def load_env() -> None:
    """Load .env if python-dotenv is available, otherwise rely on the shell environment."""
    try:
        from dotenv import load_dotenv

        env_path = repo_root() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Loaded environment from %s", env_path)
    except ImportError:
        logger.info("python-dotenv not installed, reading credentials from the shell only")


def kaggle_available() -> bool:
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (has_json or has_env):
        logger.warning(
            "No Kaggle credentials found. Place kaggle.json at ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY in .env, then rerun."
        )
        return False
    try:
        import kaggle  # noqa: F401
    except ImportError:
        logger.warning("The kaggle package is not installed. Run 'pip install kaggle'.")
        return False
    except OSError as exc:
        logger.warning("Kaggle authentication failed: %s", exc)
        return False
    return True


def download_kaggle_dataset(slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset into dest."""
    if not kaggle_available():
        return False
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading %s into %s", slug, dest)
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)

    # Older kaggle client versions leave the archive behind rather than unzipping.
    for archive in dest.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()

    files = sorted(p.name for p in dest.rglob("*") if p.is_file())
    logger.info("Files now present in %s: %s", dest, files)
    return bool(files)


def download_benchmark(url: str, dest: Path, filename: str) -> bool:
    """Download the Robyn RData file.

    pyreadr reads .RData in Python, so this project never needs an R installation.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests is not installed, cannot download the benchmark file")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    target = dest / filename
    if target.exists():
        logger.info("Benchmark file already present at %s", target)
        return True
    try:
        logger.info("Downloading %s", url)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
        logger.info("Saved %s, %.1f KB", target, target.stat().st_size / 1024)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Benchmark download failed: %s. Download the file manually from %s and place it "
            "at %s, then rerun.",
            exc,
            url,
            target,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument("--primary", action="store_true", help="primary Kaggle dataset only")
    parser.add_argument("--secondary", action="store_true", help="secondary Kaggle dataset only")
    parser.add_argument("--benchmark", action="store_true", help="Robyn RData only")
    args = parser.parse_args()

    run_all = not (args.primary or args.secondary or args.benchmark)
    load_env()
    cfg = load_config()
    raw = repo_root() / cfg["paths"]["raw"]
    external = repo_root() / cfg["paths"]["external"]

    status: dict[str, bool] = {}

    if run_all or args.primary:
        status["primary"] = download_kaggle_dataset(
            str(cfg["data"]["primary"]["kaggle_slug"]), raw
        )
    if run_all or args.secondary:
        status["secondary"] = download_kaggle_dataset(
            str(cfg["data"]["secondary"]["kaggle_slug"]), raw / "secondary"
        )
    if run_all or args.benchmark:
        status["benchmark"] = download_benchmark(
            str(cfg["data"]["benchmark"]["source_url"]),
            external,
            str(cfg["data"]["benchmark"]["filename"]),
        )

    logger.info("Download summary: %s", status)
    if not status.get("primary", True):
        logger.warning(
            "The primary dataset is missing. Run 'python scripts/make_replica_dataset.py' to "
            "generate a schema faithful stand in so the pipeline still runs, or download "
            "%s manually from Kaggle into %s.",
            cfg["data"]["primary"]["filename"],
            raw,
        )
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
