import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import benchmark, monitors

logger = logging.getLogger("data_collection")


def run_measurement(
    episode: dict,
    condition: dict,
    cfg: dict,
    prompt_template: str,
    phase: str,
    repeat: int,
    request_rate_rps=None,
    request_index=None,
):
    """Run one monitor measurement and return its JSON record."""

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    record = {
        "timestamp": started_at,
        "phase": phase,
        "repeat": repeat,
        "request_index": request_index,
        "episode_id": episode["id"],
        "category": episode["category"],
        "label": episode["label"],
        "condition": condition["name"],
        "backend": cfg["monitor"]["backend"],
        "request_rate_rps": request_rate_rps,
    }

    try:
        score = monitors.score_episode(
            episode,
            condition["visible_fields"],
            cfg["monitor"],
            prompt_template,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000

        record.update({
            "status": "success",
            "score": int(score),
            "latency_ms": round(elapsed_ms, 4),
            "error_type": None,
            "error_message": None,
        })

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000

        record.update({
            "status": "error",
            "score": None,
            "latency_ms": round(elapsed_ms, 4),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
        })

        logger.exception(
            "Measurement failed: episode=%s, condition=%s",
            episode["id"],
            condition["name"],
        )

    return record


def collect(cfg: dict, output_path: str) -> dict:
    """Collect warmup and benchmark measurements at different request rates."""

    episodes = benchmark.load_episodes(cfg["data"]["episodes_path"])

    prompt_template = Path(
        cfg["monitor"]["prompt_template"]
    ).read_text(encoding="utf-8")

    collection_cfg = cfg["collection"]

    warmup_repeats = int(collection_cfg["warmup_repeats"])
    benchmark_repeats = int(collection_cfg["benchmark_repeats"])

    request_rates = [
        float(rate)
        for rate in collection_cfg["request_rates_rps"]
    ]

    if warmup_repeats < 0 or benchmark_repeats < 1:
        raise ValueError("Invalid collection repeat settings")

    if not request_rates or any(rate <= 0 for rate in request_rates):
        raise ValueError("Request rates must be positive")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_errors = 0
    total_benchmark = 0
    run_started = datetime.now(timezone.utc).isoformat()

    with output_file.open("w", encoding="utf-8") as file:

        # STEP 1: Warm-up without load scheduling.
        logger.info("Starting warmup phase")

        for repeat in range(1, warmup_repeats + 1):
            for episode in episodes:
                for condition in cfg["conditions"]:

                    record = run_measurement(
                        episode,
                        condition,
                        cfg,
                        prompt_template,
                        phase="warmup",
                        repeat=repeat,
                    )

                    record["run_started"] = run_started
                    record["achieved_rps"] = None

                    file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    if record["status"] == "success":
                        total_success += 1
                    else:
                        total_errors += 1

        # STEP 2: Benchmark at each target request rate.
        for target_rps in request_rates:

            logger.info(
                "Starting benchmark at target rate: %s RPS",
                target_rps,
            )

            batch_records = []
            request_number = 0
            interval = 1.0 / target_rps
            batch_start = time.perf_counter()

            for repeat in range(1, benchmark_repeats + 1):
                for episode in episodes:
                    for condition in cfg["conditions"]:

                        # Schedule each request at the target interval.
                        scheduled_time = (
                            batch_start + request_number * interval
                        )

                        wait_time = (
                            scheduled_time - time.perf_counter()
                        )

                        if wait_time > 0:
                            time.sleep(wait_time)

                        record = run_measurement(
                            episode,
                            condition,
                            cfg,
                            prompt_template,
                            phase="benchmark",
                            repeat=repeat,
                            request_rate_rps=target_rps,
                            request_index=request_number + 1,
                        )

                        record["run_started"] = run_started
                        batch_records.append(record)
                        request_number += 1

            batch_elapsed = time.perf_counter() - batch_start

            achieved_rps = (
                len(batch_records) / batch_elapsed
                if batch_elapsed > 0
                else 0
            )

            # Save actual achieved rate for every record in this batch.
            for record in batch_records:
                record["achieved_rps"] = round(achieved_rps, 4)
                record["batch_elapsed_seconds"] = round(
                    batch_elapsed, 4
                )

                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                total_benchmark += 1

                if record["status"] == "success":
                    total_success += 1
                else:
                    total_errors += 1

            file.flush()

            logger.info(
                "Target RPS: %s | Achieved RPS: %.4f | Measurements: %d",
                target_rps,
                achieved_rps,
                len(batch_records),
            )

    summary = {
        "output_file": str(output_file),
        "episodes": len(episodes),
        "conditions": len(cfg["conditions"]),
        "warmup_repeats": warmup_repeats,
        "benchmark_repeats": benchmark_repeats,
        "request_rates_rps": request_rates,
        "benchmark_measurements": total_benchmark,
        "successful_measurements": total_success,
        "failed_measurements": total_errors,
    }

    logger.info("Collection summary: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Collect monitor latency measurements"
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--output",
        default="data/raw/test_run_load.jsonl",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = yaml.safe_load(
        Path(args.config).read_text(encoding="utf-8")
    )

    summary = collect(cfg, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()