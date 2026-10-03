import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from acoustic_cnn.core import (Config, load_config, metrics, read_dataset,
    resolve_audio, select_thresholds)
from acoustic_cnn.model import positive_weights


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "windows.csv"
        self.frame = pd.DataFrame(dict(recording_id=["001", "001", "002", "002"],
            audio_file=["a.wav", "a.wav", "b.wav", "b.wav"],
            start_seconds=[0, 5, 0, 5], split=["train", "train", "val", "val"],
            bird=[1, 0, 0, 1], frog=[0, 1, 1, 0], insect=[1, 1, 0, 1]))

    def read(self):
        self.frame.to_csv(self.path, index=False)
        return read_dataset(self.path, Config())

    def test_preserve_recording_identifiers(self):
        self.assertEqual(self.read().recording_id.iloc[0], "001")

    def test_recording_leakage_including_test(self):
        self.frame.loc[1, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.read()

    def test_audio_file_alias_leakage(self):
        self.frame.loc[2:, "audio_file"] = "a.wav"
        with self.assertRaisesRegex(ValueError, "multiple recording"):
            self.read()

    def test_missing_labels(self):
        self.frame.loc[1, "bird"] = np.nan
        with self.assertRaises(ValueError):
            self.read()

    def test_nonbinary_labels(self):
        self.frame.loc[1, "bird"] = 2
        with self.assertRaises(ValueError):
            self.read()

    def test_invalid_start(self):
        self.frame["start_seconds"] = self.frame["start_seconds"].astype(float)
        self.frame.loc[1, "start_seconds"] = float("inf")
        with self.assertRaises(ValueError):
            self.read()

    def test_duplicate_windows(self):
        self.frame.loc[1, "start_seconds"] = 0
        with self.assertRaises(ValueError):
            self.read()

    def test_path_escape(self):
        with self.assertRaises(ValueError):
            resolve_audio(self.temporary.name, "../secret.wav")

    def test_config_shape(self):
        self.assertEqual(Config().input_shape, (128, 469, 1))
        with self.assertRaises(ValueError):
            Config(fmax=30000)

    def test_unknown_config_setting(self):
        path = Path(self.temporary.name) / "config.json"
        path.write_text(json.dumps({"window_second": 5}))
        with self.assertRaises(ValueError):
            load_config(path)

    def test_thresholds_and_support(self):
        y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        probabilities = np.array([[.1, .8], [.9, .1], [.7, .7], [.2, .2]])
        thresholds = select_thresholds(y, probabilities)
        rows, global_scores = metrics(y, probabilities, thresholds, ["bird", "frog"])
        self.assertEqual([row["support"] for row in rows], [2, 2])
        self.assertEqual(global_scores["macro_f1"], 1)

    def test_thresholds_require_two_outcomes(self):
        with self.assertRaises(ValueError):
            select_thresholds(np.ones((3, 2)), np.full((3, 2), .8))

    def test_degenerate_test_class_is_reported(self):
        rows, _ = metrics(np.array([[0, 1], [0, 0]]), np.array([[.1, .9], [.2, .1]]),
                          [.5, .5], ["bird", "frog"])
        self.assertIsNone(rows[0]["roc_auc"])
        self.assertIsNone(rows[0]["average_precision"])

    def test_positive_weights(self):
        np.testing.assert_allclose(positive_weights(np.array([[1, 0], [0, 1]])), [1, 1])
        with self.assertRaises(ValueError):
            positive_weights(np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
