import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SHAPELLM_ENABLE_SEMANTIC_METRICS", "0")

from eval.config import EvalConfig
from eval.metrics import compute_text_metrics
from eval.pruners import PRUNER_REGISTRY, ensure_pruners_loaded, pruner_load_errors
from eval.run_eval import validate_eva01_eval_config
from eval.score_eval_run import score_from_summary_file
from eval.utils import aggregate_summary, compute_pct_of_full


class EvalIntegrationTests(unittest.TestCase):
    def test_pruner_registry_contains_baselines(self):
        ensure_pruners_loaded()
        expected = {
            "no_pruning",
            "random",
            "uniform",
            "divprune",
            "apet",
            "otprune",
            "tome",
            "fastv_mesh",
        }
        self.assertTrue(expected.issubset(set(PRUNER_REGISTRY)))
        self.assertEqual({}, pruner_load_errors())

    def test_text_metrics_without_semantic_models(self):
        scores = compute_text_metrics("a red chair with four legs", ["red chair", "blue car"])
        self.assertGreater(scores["rouge_l"], 0.0)
        self.assertIn("bleu_1", scores)
        self.assertIn(scores["sentence_bert"], (None, 0.0))
        self.assertIn(scores["simcse"], (None, 0.0))

    def test_summary_aggregates_semantic_metrics_and_backend(self):
        rows = [
            {
                "model_backend": "shapellm",
                "pruner": "no_pruning",
                "keep_ratio": 1.0,
                "bleu_1": 0.5,
                "bleu_2": 0.4,
                "bleu_3": 0.3,
                "bleu_4": 0.2,
                "rouge_l": 0.6,
                "sentence_bert": 0.8,
                "simcse": 0.7,
                "generation_time_sec": 1.0,
                "num_input_tokens": 10,
                "num_output_tokens": 2,
                "num_tokens_pruned": 1024,
                "pruner_tflops": 0.0,
                "llm_prefill_tflops": 1.0,
                "llm_decode_tflops": 0.1,
                "llm_total_tflops": 1.1,
                "total_tflops": 1.1,
            },
            {
                "model_backend": "shapellm",
                "pruner": "no_pruning",
                "keep_ratio": 1.0,
                "bleu_1": 0.7,
                "bleu_2": 0.6,
                "bleu_3": 0.5,
                "bleu_4": 0.4,
                "rouge_l": 0.8,
                "sentence_bert": 0.9,
                "simcse": 0.6,
                "generation_time_sec": 2.0,
                "num_input_tokens": 12,
                "num_output_tokens": 4,
                "num_tokens_pruned": 1024,
                "pruner_tflops": 0.0,
                "llm_prefill_tflops": 1.2,
                "llm_decode_tflops": 0.2,
                "llm_total_tflops": 1.4,
                "total_tflops": 1.4,
            },
        ]
        summary = compute_pct_of_full(aggregate_summary(rows))
        self.assertEqual(1, len(summary))
        row = summary[0]
        self.assertEqual("shapellm", row["model_backend"])
        self.assertAlmostEqual(0.85, row["sentence_bert_mean"])
        self.assertAlmostEqual(0.65, row["simcse_mean"])
        self.assertAlmostEqual(100.0, row["sentence_bert_pct_of_full"])

    def test_score_eval_run_exposes_semantic_extras_without_changing_score(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "summary.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "pruner",
                        "keep_ratio",
                        "rouge_l_mean",
                        "bleu_4_mean",
                        "bleu_1_mean",
                        "sentence_bert_mean",
                        "simcse_mean",
                        "generation_time_sec_mean",
                        "total_tflops_mean",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pruner": "no_pruning",
                        "keep_ratio": "1.0",
                        "rouge_l_mean": "0.6",
                        "bleu_4_mean": "0.2",
                        "bleu_1_mean": "0.5",
                        "sentence_bert_mean": "0.9",
                        "simcse_mean": "0.8",
                        "generation_time_sec_mean": "1.0",
                        "total_tflops_mean": "2.0",
                    }
                )
            score, extras = score_from_summary_file(path, "no_pruning")
        self.assertAlmostEqual(0.46, score)
        self.assertAlmostEqual(0.9, extras["sentence_bert_mean"])
        self.assertAlmostEqual(0.8, extras["simcse_mean"])

    def test_eva01_validation_rejects_pruning(self):
        cfg = EvalConfig(model_backend="eva01", pruners=["random"], keep_ratios=[0.5])
        with self.assertRaises(ValueError):
            validate_eva01_eval_config(cfg)

        cfg = EvalConfig(model_backend="eva01", pruners=["no_pruning"], keep_ratios=[1.0])
        validate_eva01_eval_config(cfg)

    def test_eva01_mock_model_allows_baseline_pruners(self):
        cfg = EvalConfig(
            model_backend="eva01",
            pruners=["random", "uniform", "divprune", "apet", "otprune", "tome", "fastv_mesh"],
            keep_ratios=[0.75, 0.5],
            mock_model=True,
        )
        validate_eva01_eval_config(cfg)

    def test_eva01_missing_dependency_error_is_lazy(self):
        if importlib.util.find_spec("eva01") is not None:
            self.skipTest("eva01 package is installed in this environment")
        from eval.eva01_backend import load_eva01_model

        with self.assertRaises(ImportError):
            load_eva01_model(device=__import__("torch").device("cpu"))


if __name__ == "__main__":
    unittest.main()
