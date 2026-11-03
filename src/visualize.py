import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


INPUT_JSONL = Path("data/raw/test_run_load.jsonl")
OUTPUT_DIR = Path("reports/figures")

# Colorblind-friendly colors (Okabe-Ito palette)
COLORS = {
    "output_only": "#0072B2",
    "cot_plus_output": "#D55E00",
}

LINE_STYLES = {
    "P50": "-",
    "P95": "--",
    "P99": ":",
}


def load_latency_data():
    """Load successful benchmark measurements grouped by RPS and condition."""

    measurements = {}

    with INPUT_JSONL.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record.get("phase") != "benchmark"
                or record.get("status") != "success"
            ):
                continue

            rps = float(record["request_rate_rps"])
            condition = record["condition"]
            latency = float(record["latency_ms"])

            if not np.isfinite(latency) or latency < 0:
                continue

            measurements.setdefault(
                (condition, rps), []
            ).append(latency)

    return measurements


def save_figure(fig, filename):
    """Save figure in high-resolution PNG and vector PDF formats."""

    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_latency_vs_load(measurements):
    """Plot P50, P95 and P99 latency against request rate."""

    conditions = list(COLORS.keys())

    request_rates = sorted({
        rps for condition, rps in measurements
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in conditions:

        medians = []
        p95_values = []
        p99_values = []
        q25_values = []
        q75_values = []

        for rps in request_rates:
            values = measurements.get((condition, rps), [])

            if not values:
                raise ValueError(
                    f"No measurements for {condition} at {rps} RPS"
                )

            q25, median, q75, p95, p99 = np.percentile(
                values, [25, 50, 75, 95, 99]
            )

            q25_values.append(q25)
            medians.append(median)
            q75_values.append(q75)
            p95_values.append(p95)
            p99_values.append(p99)

        color = COLORS[condition]

        # P25-P75 percentile interval
        ax.fill_between(
            request_rates,
            q25_values,
            q75_values,
            color=color,
            alpha=0.15,
            label=f"{condition} P25-P75",
        )

        # Median latency
        ax.plot(
            request_rates,
            medians,
            color=color,
            linestyle=LINE_STYLES["P50"],
            marker="o",
            linewidth=2,
            markersize=7,
            label=f"{condition} P50",
        )

        # P95 latency
        ax.plot(
            request_rates,
            p95_values,
            color=color,
            linestyle=LINE_STYLES["P95"],
            marker="s",
            linewidth=1.8,
            markersize=6,
            label=f"{condition} P95",
        )

        # P99 latency
        ax.plot(
            request_rates,
            p99_values,
            color=color,
            linestyle=LINE_STYLES["P99"],
            marker="^",
            linewidth=1.8,
            markersize=6,
            label=f"{condition} P99",
        )

    ax.set_xlabel("Request Rate (requests/sec)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)

    ax.set_title(
        "Latency vs Request Rate",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks(request_rates)
    ax.set_ylim(bottom=0)

    ax.grid(
        axis="both",
        linestyle="--",
        alpha=0.4,
    )

    ax.legend(
        fontsize=9,
        ncol=2,
        frameon=True,
    )

    fig.tight_layout()

    save_figure(fig, "fig1_latency_vs_load")


def plot_latency_distribution(measurements):
    """Compare latency distributions and display outliers."""

    conditions = list(COLORS.keys())

    values = [
        [
            latency
            for (condition, rps), latencies in measurements.items()
            if condition == name
            for latency in latencies
        ]
        for name in conditions
    ]

    if any(not group for group in values):
        raise ValueError("No latency data available for boxplot")

    fig, ax = plt.subplots(figsize=(9, 6))

    boxplot = ax.boxplot(
        values,
        tick_labels=conditions,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 2},
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
        },
        flierprops={
            "marker": "o",
            "markersize": 4,
            "alpha": 0.6,
        },
    )

    for patch, condition in zip(
        boxplot["boxes"], conditions
    ):
        patch.set_facecolor(COLORS[condition])
        patch.set_alpha(0.55)

    ax.set_xlabel("Monitoring Condition", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)

    ax.set_title(
        "Distribution of Benchmark Latency",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()

    save_figure(fig, "fig2_distribution_boxplot")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    measurements = load_latency_data()

    if not measurements:
        raise ValueError("No benchmark data found")

    plot_latency_vs_load(measurements)
    plot_latency_distribution(measurements)

    print("Scientific plots created successfully.")

    print(
        "Conditions:",
        sorted({condition for condition, rps in measurements}),
    )

    print(
        "Request rates:",
        sorted({rps for condition, rps in measurements}),
    )

    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"Created: {path}")


if __name__ == "__main__":
    main()