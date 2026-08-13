#!/usr/bin/env python3
"""Generate a schema faithful stand in for the primary dataset.

Why this exists. The primary dataset comes from Kaggle and needs an API token. Rather than
have the repository fail to run for anyone without one, this script produces a file with
the identical schema, shape and date range, so every downstream step works immediately.

Read this before quoting any number produced from the stand in. This is simulated data. It
is generated from a known data generating process that I control, so the model will recover
the parameters I put in. That makes it a useful test of whether the code is correct, and it
makes it worthless as evidence about real advertising. Any figure in the README that came
from the stand in is labelled as such.

Two things the generator does deliberately:

1. It writes the ground truth parameters to data/raw/replica_ground_truth.json, so
   scripts/run_phase1.py can compare the recovered posteriors against the values that
   generated the data. Parameter recovery on simulated data is the cleanest available check
   that the adstock and saturation implementations are right.

2. It reproduces the Division Z duplication bug found in the real file, so the data quality
   step has something to catch. Pass --no-bug to generate a clean file.
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm.config import load_config, repo_root  # noqa: E402
from mmm.features import geometric_adstock, hill_saturation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("make_replica")

CHANNELS = [
    "Google_Impressions",
    "Facebook_Impressions",
    "Email_Impressions",
    "Affiliate_Impressions",
    "Paid_Views",
]

# Ground truth. The generator uses these, and run_phase1.py checks the posterior against
# them, so a change here has to show up in the recovered parameters or something is wrong.
TRUE_DECAY = {
    "Google_Impressions": 0.20,   # search intent decays fast
    "Facebook_Impressions": 0.55,
    "Email_Impressions": 0.30,
    "Affiliate_Impressions": 0.45,
    "Paid_Views": 0.65,           # video builds and lingers
}
TRUE_HALF_SAT = {
    "Google_Impressions": 0.35,
    "Facebook_Impressions": 0.45,
    "Email_Impressions": 0.25,
    "Affiliate_Impressions": 0.55,
    "Paid_Views": 0.40,
}
TRUE_BETA_MEAN = {
    "Google_Impressions": 0.30,
    "Facebook_Impressions": 0.18,
    "Email_Impressions": 0.08,
    "Affiliate_Impressions": 0.12,
    "Paid_Views": 0.10,
}
TRUE_BETA_SD = 0.25  # lognormal spread of the division level effects
BASE_SCALE = 1_200_000.0


def generate(
    n_divisions: int = 26,
    n_weeks: int = 113,
    start: str = "2018-01-06",
    seed: int = 42,
    inject_bug: bool = True,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    divisions = list(string.ascii_uppercase[:n_divisions])
    dates = pd.date_range(start=start, periods=n_weeks, freq="7D")

    # Division level scale, so the panel has the heterogeneity the hierarchy exists to pool.
    div_scale = rng.lognormal(mean=0.0, sigma=0.45, size=n_divisions)
    div_beta = {
        ch: np.exp(np.log(TRUE_BETA_MEAN[ch]) + rng.normal(0, TRUE_BETA_SD, n_divisions))
        for ch in CHANNELS
    }

    t = np.arange(n_weeks)
    seasonal = (
        0.10 * np.sin(2 * np.pi * t / 52.18)
        + 0.05 * np.cos(2 * np.pi * t / 52.18)
        + 0.04 * np.sin(4 * np.pi * t / 52.18)
    )
    trend = 0.08 * t / n_weeks

    rows = []
    for d, division in enumerate(divisions):
        # Each channel gets a persistent division level mean plus weekly variation, plus
        # occasional campaign bursts, which is what gives the difference in differences
        # check something to detect.
        media = {}
        for ch in CHANNELS:
            level = rng.lognormal(mean=11.0, sigma=0.5) * div_scale[d]
            weekly = level * np.exp(rng.normal(0, 0.35, n_weeks))
            n_bursts = rng.integers(0, 3)
            for _ in range(n_bursts):
                start_wk = int(rng.integers(10, n_weeks - 20))
                length = int(rng.integers(6, 16))
                weekly[start_wk : start_wk + length] *= rng.uniform(1.6, 3.0)
            # Genuine dark weeks, so the zero inflation diagnostic has something to find.
            dark = rng.random(n_weeks) < 0.04
            weekly[dark] = 0.0
            media[ch] = weekly

        organic = (
            rng.lognormal(mean=10.5, sigma=0.4, size=n_weeks)
            * div_scale[d]
            * (1.0 + 0.25 * seasonal)
        )
        paid_views = media["Paid_Views"]

        response = np.zeros(n_weeks)
        for ch in CHANNELS:
            scaled = media[ch] / max(media[ch].max(), 1.0)
            adstocked = geometric_adstock(scaled, TRUE_DECAY[ch], max_lag=8)
            saturated = hill_saturation(adstocked, TRUE_HALF_SAT[ch], slope=1.0)
            response += div_beta[ch][d] * saturated

        base = BASE_SCALE * div_scale[d]
        sales = base * (0.55 + response + seasonal + trend)
        sales = sales * np.exp(rng.normal(0, 0.06, n_weeks))  # multiplicative noise
        sales = np.clip(sales, 1.0, None)

        for w in range(n_weeks):
            rows.append(
                {
                    "Division": division,
                    "Calendar_Week": dates[w].strftime("%m/%d/%Y"),
                    "Paid_Views": round(float(paid_views[w]), 2),
                    "Organic_Views": round(float(organic[w]), 2),
                    "Google_Impressions": round(float(media["Google_Impressions"][w]), 2),
                    "Email_Impressions": round(float(media["Email_Impressions"][w]), 2),
                    "Facebook_Impressions": round(float(media["Facebook_Impressions"][w]), 2),
                    "Affiliate_Impressions": round(float(media["Affiliate_Impressions"][w]), 2),
                    "Overall_Views": round(float(paid_views[w] + organic[w]), 2),
                    "Sales": round(float(sales[w]), 2),
                }
            )

    df = pd.DataFrame(rows)

    if inject_bug:
        # Reproduce the duplication found in the real file: Division Z appears twice.
        z_rows = df[df["Division"] == "Z"].copy()
        df = pd.concat([df, z_rows], ignore_index=True)
        logger.warning(
            "Injected the Division Z duplication bug, %d extra rows. The data quality step "
            "in src/mmm/validation.py is what catches it.",
            len(z_rows),
        )

    ground_truth = {
        "note": (
            "Parameters used to generate the stand in dataset. Any figure derived from this "
            "file is a test of the code, not a finding about advertising."
        ),
        "n_divisions": n_divisions,
        "n_weeks": n_weeks,
        "start_week": start,
        "seed": seed,
        "bug_injected": inject_bug,
        "true_decay": TRUE_DECAY,
        "true_half_sat": TRUE_HALF_SAT,
        "true_beta_channel_mean": TRUE_BETA_MEAN,
        "true_beta_sd_log": TRUE_BETA_SD,
        "overall_views_is_sum_of": ["Paid_Views", "Organic_Views"],
    }
    return df, ground_truth


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the stand in primary dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bug", action="store_true", help="generate without the Division Z bug")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = repo_root() / cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / str(cfg["data"]["primary"]["filename"])

    if target.exists() and not args.force:
        logger.error(
            "%s already exists. If that is the real Kaggle file, leave it alone. Pass --force "
            "to overwrite it with the stand in.",
            target,
        )
        return 1

    df, ground_truth = generate(seed=args.seed, inject_bug=not args.no_bug)
    df.to_csv(target, index=False)

    truth_path = raw_dir / "replica_ground_truth.json"
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(ground_truth, handle, indent=2)

    notice = raw_dir / "REPLICA_NOTICE.txt"
    notice.write_text(
        "The file sample_media_spend.csv in this folder was generated by\n"
        "scripts/make_replica_dataset.py. It is simulated data with the same schema, shape\n"
        "and date range as the Kaggle dataset, not the Kaggle dataset itself.\n\n"
        "Delete this notice and the CSV, then run 'python scripts/fetch_data.py --primary'\n"
        "to replace it with the real file.\n",
        encoding="utf-8",
    )

    logger.info("Wrote %s with %d rows and %d columns", target, len(df), df.shape[1])
    logger.info("Divisions: %d, weeks: %d", df["Division"].nunique(), df["Calendar_Week"].nunique())
    logger.info("Ground truth written to %s", truth_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
