"""Run the 3 remaining experiments sequentially with GPU cleanup.

Launch with: nohup python -u scripts/exp3/run_final.py > /tmp/exp3_final.log 2>&1 &
"""
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


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def exists(name):
    return os.path.isfile(os.path.join(RESULTS_DIR, f"{name}_results.json"))


def run_exp(name, fn):
    if exists(name):
        logger.info("SKIP %s (exists)", name)
        return
    logger.info("START %s", name)
    clear_gpu()
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        v = result.get("verdict", "?") if isinstance(result, dict) else "?"
        logger.info("DONE %s in %.0fs: %s", name, elapsed, v)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("FAIL %s after %.0fs: %s", name, elapsed, e)
        traceback.print_exc()
    clear_gpu()


if __name__ == "__main__":
    logger.info("=== FINAL EXP3 RUNNER ===")
    logger.info("GPU: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # 1. context_48
    def do_context_48():
        from scripts.exp3.test_context import _make_context_builder
        from scripts.exp3.evaluate_representation import evaluate_representation
        return evaluate_representation(
            name="context_48",
            build_dataset_fn=_make_context_builder(48),
            n_seeds=3,
            run_baseline=True,
        )
    run_exp("context_48", do_context_48)

    # 2. context_64
    def do_context_64():
        from scripts.exp3.test_context import _make_context_builder
        from scripts.exp3.evaluate_representation import evaluate_representation
        return evaluate_representation(
            name="context_64",
            build_dataset_fn=_make_context_builder(64),
            n_seeds=3,
            run_baseline=True,
        )
    run_exp("context_64", do_context_64)

    # 3. transition_head
    def do_transition():
        from scripts.exp3.test_transition_head import run_transition_head_experiment
        return run_transition_head_experiment()
    run_exp("transition_head", do_transition)

    logger.info("=== ALL DONE ===")

    # Summary
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith("_results.json"):
            d = json.load(open(os.path.join(RESULTS_DIR, f)))
            n = f.replace("_results.json", "")
            v = d.get("verdict", "?")
            for k in ["challenger_avg", "regression_avg", "best_avg"]:
                if k in d:
                    m = d[k]
                    logger.info("  %s: econ=%.1f%% trans=%.1f%% flat=%.1f%% -> %s",
                                n, m.get("econ_dir", 0) * 100, m.get("transition_acc", 0) * 100,
                                m.get("pred_dist", {}).get("flat", 0) * 100, v[:50])
                    break
