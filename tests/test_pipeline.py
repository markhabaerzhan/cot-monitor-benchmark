import json
import unittest
from pathlib import Path

import yaml

from src import benchmark, metrics, stats

ROOT = Path(__file__).resolve().parents[1]


class MetricsTests(unittest.TestCase):
    def test_auroc_perfect_and_tied(self):
        self.assertEqual(metrics.auroc([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertEqual(metrics.auroc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5)

    def test_threshold_respects_fpr(self):
        safe = list(range(20))  # 0..19
        t = metrics.threshold_at_fpr(safe, 0.05)
        self.assertLessEqual(sum(s >= t for s in safe) / 20, 0.05)

    def test_prf(self):
        r, p, f = metrics.prf(tp=3, fp=1, fn=1)
        self.assertAlmostEqual(r, 0.75)
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(f, 0.75)


class StatsTests(unittest.TestCase):
    def test_mcnemar_counts(self):
        res = stats.rq3_mcnemar([1, 0, 0, 1], [1, 1, 1, 1])
        self.assertEqual((res["b01_only_cot"], res["b10_only_output"]), (2, 0))

    def test_binomial(self):
        self.assertLess(stats.rq2_binomial(20, 20, 0.7)["p_value"], 0.05)


class SmokeTest(unittest.TestCase):
    def test_sample_run(self):
        cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text())
        for k in ("episodes_path",):
            cfg["data"][k] = str(ROOT / cfg["data"][k])
        cfg["monitor"]["prompt_template"] = str(ROOT / cfg["monitor"]["prompt_template"])
        res = benchmark.run(cfg)
        self.assertEqual(res["n_episodes"], 10)
        self.assertEqual(set(res["conditions"]), {"output_only", "cot_plus_output"})
        json.dumps(res, default=float)  # serializable
        # keyword placeholder reads CoT -> must detect at least as much as output-only
        cot = res["conditions"]["cot_plus_output"]["overall"]["recall_at_fpr"]
        out = res["conditions"]["output_only"]["overall"]["recall_at_fpr"]
        self.assertGreaterEqual(cot, out)


if __name__ == "__main__":
    unittest.main()
