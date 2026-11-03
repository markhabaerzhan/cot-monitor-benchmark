"""Process benchmark JSONL measurements by request rate and condition."""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("data_analysis")


def load_measurements(input_path):
    """Read JSONL records and count malformed lines."""
    records = []
    malformed_lines = 0

    with Path(input_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_lines += 1
                logger.error("Invalid JSON at line %d", line_number)

    return records, malformed_lines


def calculate_statistics(values, bootstrap_resamples, seed):
    """Calculate descriptive statistics and bootstrap CI."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {
            "mean_ms": None,
            "std_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
            "bootstrap_ci_lower_ms": None,
            "bootstrap_ci_upper_ms": None,
            "outlier_count": 0,
            "iqr_lower_fence_ms": None,
            "iqr_upper_fence_ms": None,
        }

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outlier_count = int(
        np.sum((values < lower_fence) | (values > upper_fence))
    )

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_resamples)

    for index in range(bootstrap_resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means[index] = np.mean(sample)

    ci_lower, ci_upper = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    return {
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "bootstrap_ci_lower_ms": float(ci_lower),
        "bootstrap_ci_upper_ms": float(ci_upper),
        "outlier_count": outlier_count,
        "iqr_lower_fence_ms": float(lower_fence),
        "iqr_upper_fence_ms": float(upper_fence),
    }


def process_data(config_path, input_path, output_path):
    """Calculate latency statistics for each RPS and condition."""

    config = yaml.safe_load(
        Path(config_path).read_text(encoding="utf-8")
    )

    records, malformed_lines = load_measurements(input_path)

    seed = int(config["experiment"]["seed"])
    bootstrap_resamples = int(
        config["statistics"]["bootstrap_resamples"]
    )

    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1")

    conditions = [
        condition["name"] for condition in config["conditions"]
    ]

    request_rates = [
        float(rate)
        for rate in config["collection"]["request_rates_rps"]
    ]

    benchmark_records = [
        record for record in records
        if record.get("phase") == "benchmark"
    ]

    warmup_count = sum(
        record.get("phase") == "warmup"
        for record in records
    )

    unknown_phase_count = (
        len(records) - warmup_count - len(benchmark_records)
    )

    summary_rows = []
    group_index = 0

    for target_rps in request_rates:
        for condition in conditions:

            group_records = [
                record for record in benchmark_records
                if record.get("condition") == condition
                and record.get("request_rate_rps") is not None
                and np.isclose(
                    float(record["request_rate_rps"]),
                    target_rps,
                )
            ]

            successful_records = [
                record for record in group_records
                if record.get("status") == "success"
            ]

            error_count = len(group_records) - len(successful_records)

            valid_records = []

            for record in successful_records:
                try:
                    latency = float(record["latency_ms"])

                    if np.isfinite(latency) and latency >= 0:
                        valid_records.append(latency)

                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Invalid latency for episode %s",
                        record.get("episode_id"),
                    )

            stats = calculate_statistics(
                valid_records,
                bootstrap_resamples,
                seed + group_index,
            )

            achieved_values = [
                float(record["achieved_rps"])
                for record in group_records
                if record.get("achieved_rps") is not None
            ]

            achieved_rps = (
                float(np.mean(achieved_values))
                if achieved_values else None
            )

            error_rate = (
                error_count / len(group_records)
                if group_records else 0.0
            )

            row = {
                "request_rate_rps": target_rps,
                "achieved_rps": achieved_rps,
                "condition": condition,
                "benchmark_measurements": len(group_records),
                "successful_measurements": len(successful_records),
                "failed_measurements": error_count,
                "error_rate": error_rate,
                "valid_latency_measurements": len(valid_records),
                "warmup_measurements_excluded": warmup_count,
                "malformed_json_lines": malformed_lines,
                "unknown_phase_records": unknown_phase_count,
                "bootstrap_resamples": bootstrap_resamples,
                "mean_ci_method": "percentile bootstrap 95%",
                "outlier_method": "IQR 1.5 rule",
                **stats,
            }

            summary_rows.append(row)
            group_index += 1

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        with output_file.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(summary_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    total_errors = sum(
        row["failed_measurements"]
        for row in summary_rows
    )

    result = {
        "input_file": str(input_path),
        "output_file": str(output_file),
        "total_json_records": len(records),
        "warmup_records_excluded": warmup_count,
        "benchmark_records": len(benchmark_records),
        "request_rates_analyzed": len(request_rates),
        "conditions_analyzed": len(conditions),
        "summary_rows": len(summary_rows),
        "malformed_json_lines": malformed_lines,
        "failed_measurements": total_errors,
    }

    logger.info("Analysis completed: %s", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark latency measurements"
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--input",
        default="data/raw/test_run_load.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/processed/metrics_summary.csv",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    result = process_data(
        args.config,
        args.input,
        args.output,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()