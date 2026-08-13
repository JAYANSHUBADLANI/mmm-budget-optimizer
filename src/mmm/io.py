"""Loading and saving. Everything that touches disk lives here.

The three datasets are deliberately kept in separate loaders. The primary dataset is the
analysis. The secondary dataset is a generalisation check. The benchmark dataset is a
correctness check. They are never concatenated, because stacking unrelated panels with
different channels, currencies and time spans would produce a table that looks like data
and means nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DotDict, load_config, repo_root

logger = logging.getLogger(__name__)

# Column name variants I have seen for the primary dataset across Kaggle versions.
# Normalising here means the rest of the codebase can assume one schema.
PRIMARY_COLUMN_ALIASES: dict[str, str] = {
    "division": "Division",
    "calendar_week": "Calendar_Week",
    "paid_views": "Paid_Views",
    "organic_views": "Organic_Views",
    "google_impressions": "Google_Impressions",
    "email_impressions": "Email_Impressions",
    "facebook_impressions": "Facebook_Impressions",
    "affiliate_impressions": "Affiliate_Impressions",
    "overall_views": "Overall_Views",
    "sales": "Sales",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in PRIMARY_COLUMN_ALIASES:
            renames[col] = PRIMARY_COLUMN_ALIASES[key]
    return df.rename(columns=renames)


def _find_file(folder: Path, filename: str) -> Path:
    """Locate a file, tolerating the case and separator variations Kaggle exports use."""
    direct = folder / filename
    if direct.exists():
        return direct
    stem = Path(filename).stem.lower().replace("_", "").replace("-", "")
    for candidate in folder.rglob("*"):
        if not candidate.is_file():
            continue
        cand_stem = candidate.stem.lower().replace("_", "").replace("-", "")
        if cand_stem == stem and candidate.suffix.lower() == Path(filename).suffix.lower():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} under {folder}. "
        "Run 'python scripts/fetch_data.py' to download it, or "
        "'python scripts/make_replica_dataset.py' to generate the schema faithful stand in."
    )


def load_primary_raw(cfg: DotDict | None = None) -> pd.DataFrame:
    """Load the primary panel exactly as it sits on disk, with no cleaning applied.

    I keep the raw load separate from cleaning so the data quality report can compare
    the before and after states and quantify what the cleaning step actually changed.
    """
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["raw"]
    path = _find_file(folder, cfg["data"]["primary"]["filename"])
    df = pd.read_csv(path)
    df = _normalise_columns(df)
    logger.info("Loaded primary raw data from %s with shape %s", path, df.shape)
    return df


def parse_primary_dates(df: pd.DataFrame, cfg: DotDict | None = None) -> pd.DataFrame:
    """Parse Calendar_Week into a real datetime.

    The source ships dates as strings. I try the configured format first, then fall back
    to pandas inference, and I fail loudly rather than silently producing NaT rows,
    because a silent date failure would quietly destroy the time ordering the model needs.
    """
    cfg = cfg or load_config()
    date_col = cfg["data"]["primary"]["date_col"]
    fmt = cfg["data"]["primary"].get("date_format")
    out = df.copy()
    parsed = pd.to_datetime(out[date_col], format=fmt, errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(out[date_col], errors="coerce", dayfirst=False)
    n_bad = int(parsed.isna().sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} values in {date_col} could not be parsed as dates. "
            "Inspect the raw file before continuing, do not drop them blindly."
        )
    out[date_col] = parsed
    return out


def load_secondary_raw(cfg: DotDict | None = None) -> pd.DataFrame:
    """Load the secondary Kaggle dataset used only for the generalisation check."""
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["raw"] / "secondary"
    path = _find_file(folder, cfg["data"]["secondary"]["filename"])
    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    logger.info("Loaded secondary data from %s with shape %s", path, df.shape)
    return df


def load_benchmark_raw(cfg: DotDict | None = None) -> pd.DataFrame:
    """Load Robyn's dt_simulated_weekly from the RData file using pyreadr.

    pyreadr reads .RData directly, so there is no R installation in this project.
    """
    cfg = cfg or load_config()
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required to read the Robyn benchmark file. Install it with "
            "'pip install pyreadr'."
        ) from exc

    folder = repo_root() / cfg["paths"]["external"]
    path = _find_file(folder, cfg["data"]["benchmark"]["filename"])
    result = pyreadr.read_r(str(path))
    obj = cfg["data"]["benchmark"]["object_name"]
    if obj in result:
        df = result[obj]
    else:
        # pyreadr keys on the object name stored inside the file, which has occasionally
        # differed from the file name across Robyn releases.
        first_key = next(iter(result))
        logger.warning("Object %s not found in RData, falling back to %s", obj, first_key)
        df = result[first_key]
    df = pd.DataFrame(df)
    logger.info("Loaded benchmark data from %s with shape %s", path, df.shape)
    return df


def save_table(df: pd.DataFrame, name: str, cfg: DotDict | None = None) -> Path:
    """Write a result table to reports/tables as CSV and return the path."""
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["tables"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote table %s", path)
    return path


def save_json(payload: dict[str, Any], name: str, cfg: DotDict | None = None) -> Path:
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["tables"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote json %s", path)
    return path


def save_processed(df: pd.DataFrame, name: str, cfg: DotDict | None = None) -> Path:
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["processed"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        # No parquet engine installed. CSV keeps the pipeline working, at the cost of
        # losing dtypes, so I log it rather than letting it pass unnoticed.
        path = folder / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.warning("Parquet engine unavailable, wrote CSV instead: %s", path)
    return path


def load_processed(name: str, cfg: DotDict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    folder = repo_root() / cfg["paths"]["processed"]
    parquet = folder / f"{name}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = folder / f"{name}.csv"
    if csv.exists():
        frame = pd.read_csv(csv)
        if "Calendar_Week" in frame.columns:
            frame["Calendar_Week"] = pd.to_datetime(frame["Calendar_Week"], errors="coerce")
        return frame
    raise FileNotFoundError(f"No processed dataset named {name} in {folder}")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
