"""Run the final 3 remaining exp3 experiments sequentially.

context_48, context_64 (with reduced batch_size), transition_head.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import traceback

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.expanduser("~/.wavecast/audit/exp3")


def _exists(name: str) -> bool:
    return os.path.isfile(os.path.join(RESULTS_DIR, f"{name}_results.json"))


def _verdict(name: str) -> str:
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    try:
        d = json.load(open(path))
        return d.get("verdict", "?")
    except Exception:
        return "?"


def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_context_48():
    from scripts.exp3.test_context import _make_context_builder
    from scripts.exp3.evaluate_representation import evaluate_representation

    builder = _make_context_builder(48)
    return evaluate_representation(
        name="context_48",
        build_dataset_fn=builder,
        n_seeds=3,
        model_config={"batch_size": 16},
        run_baseline=True,
    )


def run_context_64():
    from scripts.exp3.test_context import _make_context_builder
    from scripts.exp3.evaluate_representation import evaluate_representation

    builder = _make_context_builder(64)
    return evaluate_representation(
        name="context_64",
        build_dataset_fn=builder,
        n_seeds=3,
        model_config={"batch_size": 8},
        run_baseline=True,
    )


def run_transition_head():
    from scripts.exp3.test_transition_head import run_transition_head_experiment
    return run_transition_head_experiment()


EXPERIMENTS = [
    ("context_48", run_context_48),
    ("context_64", run_context_64),
    ("transition_head", run_transition_head),
]

if __name__ == "__main__":
    start = time.time()
    total = len(EXPERIMENTS)
    done = 0
    skipped = 0
    failed = 0

    logger.info("=" * 70)
    logger.info("FINAL 3 EXP3 RUNNER — %d experiments", total)
    logger.info("=" * 70)
    sys.stdout.flush()

    for name, run_fn in EXPERIMENTS:
        if _exists(name):
            logger.info("SKIP %s — results already exist (%s)", name, _verdict(name))
            skipped += 1
            done += 1
            continue

        # Clear GPU between experiments
        _clear_gpu()

        logger.info("START %s [%d/%d]", name, done + 1, total)
        sys.stdout.flush()
        t0 = time.time()
        try:
            result = run_fn()
            elapsed = time.time() - t0
            verdict = result.get("verdict", "?") if isinstance(result, dict) else "?"
            logger.info("DONE %s in %.0fs — %s", name, elapsed, verdict)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("FAIL %s after %.0fs: %s", name, elapsed, e)
            traceback.print_exc()
            failed += 1
        done += 1
        sys.stdout.flush()

    total_time = time.time() - start
    hrs = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    logger.info("=" * 70)
    logger.info("ALL DONE in %dh %dm — %d ok, %d failed, %d skipped",
                hrs, mins, done - failed - skipped, failed, skipped)
    logger.info("=" * 70)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith("_results.json"):
            name = f.replace("_results.json", "")
            d = json.load(open(os.path.join(RESULTS_DIR, f)))
            v = d.get("verdict", "?")
            for key in ["challenger_avg", "regression_avg"]:
                if key in d:
                    m = d[key]
                    print(f"  {name:20s} {v[:50]:50s} "
                          f"econ={m.get('econ_dir',0):.1%} trans={m.get('transition_acc',0):.1%} "
                          f"flat={m.get('pred_dist',{}).get('flat',0):.1%}")
                    break
            else:
                print(f"  {name:20s} {v}")
