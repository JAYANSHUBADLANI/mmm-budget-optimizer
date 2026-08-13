#!/usr/bin/env python3
"""Phase 2: benchmark validation, generalisation check, causal validation.

  python scripts/run_phase2.py --smoke
  python scripts/run_phase2.py --only causal

Each of the three parts answers a different objection:

  benchmark        is the implementation correct
  generalisation   does the method travel to a different channel mix
  causal           are the estimates plausibly causal rather than correlational

Any part whose input data is missing is skipped with a message rather than failing the run,
because the Kaggle and GitHub downloads are the parts most likely to be unavailable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmm import io  # noqa: E402
from mmm.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("phase2")


def run_benchmark_part(cfg, smoke: bool) -> dict:
    from mmm.benchmark_robyn import run_benchmark

    logger.info("Benchmark check against Robyn dt_simulated_weekly")
    try:
        results = run_benchmark(cfg, smoke=smoke)
    except FileNotFoundError as exc:
        logger.warning("Benchmark skipped: %s", exc)
        return {"status": "skipped", "reason": str(exc)}
    except ImportError as exc:
        logger.warning("Benchmark skipped, pyreadr is not installed: %s", exc)
        return {"status": "skipped", "reason": str(exc)}

    io.save_table(results["contributions"], "benchmark_contributions", cfg)
    io.save_table(results["transform_parameters"], "benchmark_transforms", cfg)
    io.save_table(results["metrics"], "benchmark_metrics", cfg)
    io.save_json(results["checks"], "benchmark_checks", cfg)
    logger.info("Benchmark checks: %s", results["checks"])
    return results["checks"]


def run_generalisation_part(cfg, smoke: bool) -> dict:
    from mmm.generalization import run_generalization

    logger.info("Generalisation check on the secondary dataset")
    try:
        results = run_generalization(cfg, smoke=smoke)
    except FileNotFoundError as exc:
        logger.warning("Generalisation skipped: %s", exc)
        return {"status": "skipped", "reason": str(exc)}

    io.save_table(results["contributions"], "secondary_contributions", cfg)
    io.save_table(results["metrics"], "secondary_metrics", cfg)
    io.save_table(results["channels"], "secondary_channels", cfg)
    io.save_json(results["checks"], "secondary_checks", cfg)
    logger.info("Generalisation checks: %s", results["checks"])
    return results["checks"]


def run_causal_part(cfg, smoke: bool) -> dict:
    from mmm.causal import (
        difference_in_differences,
        geo_holdout_validation,
        parallel_trends_test,
        placebo_test,
    )

    panel = io.load_processed("primary_clean", cfg)
    paid = [c for c in cfg["channels"]["paid"] if c in panel.columns]
    controls = [c for c in cfg["channels"]["organic_controls"] if c in panel.columns]

    logger.info("Geo holdout validation")
    geo = geo_holdout_validation(panel, paid, controls, cfg, smoke=smoke)
    io.save_table(geo.to_frame(), "causal_geo_holdout", cfg)
    io.save_json(
        {"holdout_divisions": geo.holdout_divisions, "notes": geo.notes},
        "causal_geo_holdout_notes",
        cfg,
    )

    did_rows, pt_rows, placebo_rows = [], [], []
    for channel in paid:
        logger.info("Difference in differences on %s", channel)
        did_rows.append(difference_in_differences(panel, channel, cfg))
        pt_rows.append(parallel_trends_test(panel, channel, cfg))
        placebo_rows.append(placebo_test(panel, channel, cfg, n_placebos=50))

    io.save_table(pd.DataFrame(did_rows), "causal_did", cfg)
    io.save_table(pd.DataFrame(pt_rows), "causal_parallel_trends", cfg)
    io.save_table(pd.DataFrame(placebo_rows), "causal_placebo", cfg)

    ok = [r for r in did_rows if r.get("status") == "ok"]
    return {
        "geo_holdout_divisions": geo.holdout_divisions,
        "geo_holdout_notes": geo.notes,
        "n_channels_with_did_estimate": len(ok),
        "n_channels_skipped_for_too_few_treated": len(did_rows) - len(ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 pipeline")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only",
        choices=["benchmark", "generalisation", "causal"],
        help="run a single part",
    )
    args = parser.parse_args()

    cfg = load_config()
    started = time.time()
    summary: dict[str, object] = {"smoke_mode": args.smoke}

    if args.only in (None, "benchmark"):
        summary["benchmark"] = run_benchmark_part(cfg, args.smoke)
    if args.only in (None, "generalisation"):
        summary["generalisation"] = run_generalisation_part(cfg, args.smoke)
    if args.only in (None, "causal"):
        summary["causal"] = run_causal_part(cfg, args.smoke)

    summary["runtime_seconds"] = round(time.time() - started, 1)
    io.save_json(summary, "phase2_summary", cfg)
    logger.info("Phase 2 complete in %.1f seconds", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
