"""Entry point: python -m src.main --config configs/config.yaml"""
import argparse
import logging
from pathlib import Path

import yaml

from . import benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="CoT vs output-only monitor benchmark")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{cfg['output']['run_name']}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")])

    result = benchmark.run(cfg)
    path = benchmark.save(result, cfg["output"]["dir"], cfg["output"]["run_name"])
    logging.getLogger("main").info("Results written to %s (log: %s)", path, log_path)


if __name__ == "__main__":
    main()
