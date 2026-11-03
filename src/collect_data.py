import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import benchmark, monitors

logger = logging.getLogger("data_collection")


def collect(cfg: dict, output_path: str) -> dict:
    """Run warmup and benchmark measurements and save them as JSONL."""

    episodes = benchmark.load_episodes(cfg["data"]["episodes_path"])

    prompt_template = Path(
        cfg["monitor"]["prompt_template"]
    ).read_text(encoding="utf-8")

    collection_cfg = cfg["collection"]
    warmup_repeats = int(collection_cfg["warmup_repeats"])
    benchmark_repeats = int(collection_cfg["benchmark_repeats"])

    if warmup_repeats < 0 or benchmark_repeats < 1:
        raise ValueError("Invalid collection repeat settings")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_success = 0
    total_errors = 0
    run_started = datetime.now(timezone.utc).isoformat()

    with output_file.open("w", encoding="utf-8") as file:

        for phase, repeats in (
            ("warmup", warmup_repeats),
            ("benchmark", benchmark_repeats),
        ):
            logger.info(
                "Starting %s phase: %d repeats",
                phase,
                repeats,
            )

            for repeat in range(1, repeats + 1):
                for episode in episodes:
                    for condition in cfg["conditions"]:

                        started = time.perf_counter()
                        record = {
                            "timestamp": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "run_started": run_started,
                            "phase": phase,
                            "repeat": repeat,
                            "episode_id": episode["id"],
                            "category": episode["category"],
                            "label": episode["label"],
                            "condition": condition["name"],
                            "backend": cfg["monitor"]["backend"],
                        }

                        try:
                            score = monitors.score_episode(
                                episode,
                                condition["visible_fields"],
                                cfg["monitor"],
                                prompt_template,
                            )

                            elapsed_ms = (
                                time.perf_counter() - started
                            ) * 1000

                            record.update({
                                "status": "success",
                                "score": int(score),
                                "latency_ms": round(elapsed_ms, 4),
                                "error_type": None,
                                "error_message": None,
                            })

                            total_success += 1

                        except Exception as exc:
                            elapsed_ms = (
                                time.perf_counter() - started
                            ) * 1000

                            record.update({
                                "status": "error",
                                "score": None,
                                "latency_ms": round(elapsed_ms, 4),
                                "error_type": type(exc).__name__,
                                "error_message": str(exc)[:500],
                            })

                            total_errors += 1
                            logger.exception(
                                "Measurement failed: episode=%s, condition=%s",
                                episode["id"],
                                condition["name"],
                            )

                        file.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )

            file.flush()

    summary = {
        "output_file": str(output_file),
        "episodes": len(episodes),
        "conditions": len(cfg["conditions"]),
        "warmup_repeats": warmup_repeats,
        "benchmark_repeats": benchmark_repeats,
        "successful_measurements": total_success,
        "failed_measurements": total_errors,
        "benchmark_measurements_expected": (
            len(episodes)
            * len(cfg["conditions"])
            * benchmark_repeats
        ),
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
        default="data/raw/test_run.jsonl",
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
