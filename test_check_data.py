"""Synthetic layout checks only. No EEG, image, tensor or model files are read."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from check_data import inspect_layout


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {
            "subjects": ["sub-01", "sub-02"],
            "metadata_path": "data/{subject}/metadata.parquet",
            "eeg_trials_train": "data/{subject}/train.npy",
            "eeg_trials_test": "data/{subject}/test.npy",
            "features_train": "features/train.pt",
            "features_test": "features/test.pt",
        }

    def create(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def populate(self):
        for field, template in self.config.items():
            if field != "subjects":
                for subject in self.config["subjects"]:
                    self.create(template.format(subject=subject))

    def cli(self, *extra):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config))
        return subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_data.py")),
             "--config", str(path), "--root", str(self.root), *extra],
            capture_output=True, text=True,
        )

    def test_collects_all_missing_without_duplicate_shared_features(self):
        result = inspect_layout(self.config, self.root)
        self.assertEqual((result["checked"], result["missing"]), (8, 8))

    def test_complete_layout_does_not_claim_content_validation(self):
        self.populate()
        result = inspect_layout(self.config, self.root)
        self.assertEqual(result["missing"], 0)
        self.assertIn("not validated", result["scope"])

    def test_directories_are_not_files(self):
        (self.root / "features/train.pt").mkdir(parents=True)
        result = inspect_layout(self.config, self.root)
        self.assertEqual(result["missing"], 8)

    def test_subject_selection_limits_required_paths(self):
        result = inspect_layout(self.config, self.root, ["sub-02"])
        self.assertEqual(result["checked"], 5)
        self.assertFalse(any("sub-01" in r["path"] for r in result["files"]))

    def test_unknown_subject_rejected(self):
        with self.assertRaises(ValueError):
            inspect_layout(self.config, self.root, ["sub-03"])

    def test_bad_configs_rejected(self):
        for config in [[], {}, {"subjects": []}, {"subjects": [1]},
                       {"subjects": ["x", "x"]}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                inspect_layout(config, self.root)

    def test_missing_or_non_string_paths_rejected(self):
        for value in [None, "", 3]:
            config = copy.deepcopy(self.config)
            config["features_train"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect_layout(config, self.root)

    def test_unexpected_template_fields_rejected(self):
        for value in ["{unknown}.npy", "{subject.__class__}", "{subject!r}",
                      "{subject:>9}", "{broken"]:
            config = copy.deepcopy(self.config)
            config["eeg_trials_train"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                inspect_layout(config, self.root)

    def test_absolute_paths_are_not_rebased(self):
        path = self.root / "absolute.pt"
        path.touch()
        self.config["features_train"] = str(path)
        result = inspect_layout(self.config, self.root / "elsewhere")
        entry = next(r for r in result["files"] if r["path"] == str(path))
        self.assertTrue(entry["exists"])

    def test_layout_inspection_does_not_open_data(self):
        self.populate()
        with patch("builtins.open", side_effect=AssertionError("unexpected read")), \
             patch.object(Path, "open", side_effect=AssertionError("unexpected read")):
            self.assertEqual(inspect_layout(self.config, self.root)["missing"], 0)

    def test_cli_missing_exit_one_json_lists_all_inputs(self):
        proc = self.cli()
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["missing"], 8)
        self.assertEqual(proc.stderr, "")

    def test_cli_available_exit_zero(self):
        self.populate()
        proc = self.cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["missing"], 0)

    def test_cli_invalid_subject_exit_two(self):
        proc = self.cli("--subjects", "not-a-subject")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("selected subjects", proc.stderr)

    def test_cli_invalid_json_exit_two(self):
        config = self.root / "bad.json"
        config.write_text("{")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_data.py")),
             "--config", str(config)], capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("Input check failed", proc.stderr)

    def test_metadata_alternative_is_not_silently_accepted(self):
        self.create("data/sub-01/metadata_categories.parquet")
        result = inspect_layout(self.config, self.root, ["sub-01"])
        self.assertEqual(result["missing"], 5)

    def test_no_mutation_of_caller_config(self):
        previous = copy.deepcopy(self.config)
        inspect_layout(self.config, self.root, ["sub-01"])
        self.assertEqual(previous, self.config)


if __name__ == "__main__":
    unittest.main()
