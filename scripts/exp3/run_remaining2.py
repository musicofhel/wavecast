"""Run remaining failed exp3 experiments with GPU memory cleanup."""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time

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


def _clear_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("GPU memory cleared: %.0f MB used",
                torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0)


def run_context(ctx_len: int):
    """Run a single context length experiment with GPU cleanup."""
    _clear_gpu()
    from scripts.exp3.test_context import _make_context_builder
    from scripts.exp3.evaluate_representation import evaluate_representation
    builder = _make_context_builder(ctx_len)
    return evaluate_representation(
        name=f"context_{ctx_len}",
        build_dataset_fn=builder,
        n_seeds=3,
        run_baseline=True,
    )


def run_transition_head():
    """Run transition head experiment."""
    _clear_gpu()
    from scripts.exp3.test_transition_head import run_transition_head_experiment
    return run_transition_head_experiment()


EXPERIMENTS = [
    ("context_48", lambda: run_context(48)),
    ("context_64", lambda: run_context(64)),
    ("transition_head", run_transition_head),
]

if __name__ == "__main__":
    start = time.time()
    logger.info("=" * 70)
    logger.info("REMAINING EXP3 RUNNER v2 — %d experiments", len(EXPERIMENTS))
    logger.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("=" * 70)

    _clear_gpu()
    done = 0
    failed = 0

    for name, run_fn in EXPERIMENTS:
        if _exists(name):
            logger.info("SKIP %s — already exists", name)
            done += 1
            continue

        logger.info("START %s", name)
        t0 = time.time()
        try:
            result = run_fn()
            elapsed = time.time() - t0
            verdict = result.get("verdict", "?") if isinstance(result, dict) else "?"
            logger.info("DONE %s in %.0fs — %s", name, elapsed, verdict)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("FAIL %s after %.0fs: %s", name, elapsed, e)
            import traceback
            traceback.print_exc()
            failed += 1
        done += 1
        _clear_gpu()
        sys.stdout.flush()

    total_time = time.time() - start
    logger.info("=" * 70)
    logger.info("ALL DONE in %.0fm — %d failed", total_time / 60, failed)

    # Print all results
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith("_results.json"):
            name = f.replace("_results.json", "")
            d = json.load(open(os.path.join(RESULTS_DIR, f)))
            v = d.get("verdict", "?")
            for key in ["challenger_avg", "regression_avg", "best_avg"]:
                if key in d:
                    m = d[key]
                    print(f"  {name:20s} econ={m.get('econ_dir',0):.1%} trans={m.get('transition_acc',0):.1%} "
                          f"flat={m.get('pred_dist',{}).get('flat',0):.1%} → {v[:50]}")
                    break
