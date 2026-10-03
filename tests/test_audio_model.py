from dataclasses import asdict
from pathlib import Path
import importlib.util
import tempfile
import unittest
import numpy as np
from acoustic_cnn.core import Config, file_digest, write_json


@unittest.skipUnless(importlib.util.find_spec("tensorflow") and importlib.util.find_spec("librosa"),
                     "TensorFlow and librosa are required")
class AudioModelTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(sample_rate=8000, window_seconds=.5, n_fft=256,
            hop_length=128, n_mels=32, fmax=4000, low_frequency_hz=1000)

    def test_preprocessing_shape_and_augmentation(self):
        from acoustic_cnn.audio import augment, mel_spectrogram
        samples = np.sin(2 * np.pi * 500 * np.arange(4000) / 8000).astype(np.float32)
        mel = mel_spectrogram(samples, self.config)
        self.assertEqual(mel.shape, (32, 32, 1))
        output = augment(mel, self.config, np.random.default_rng(42))
        self.assertEqual(output.shape, mel.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertFalse(np.array_equal(mel, output))

    def test_model_serialization_inference_resume_and_stale_config(self):
        import soundfile as sf
        from acoustic_cnn.model import build_model, weighted_loss
        from acoustic_cnn.pipeline import inference, load_bundle
        import tensorflow as tf
        model = build_model(self.config)
        model.compile(optimizer="adam", loss=weighted_loss(np.ones(3)))
        x = np.zeros((2, *self.config.input_shape), dtype=np.float32)
        y = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        self.assertTrue(np.isfinite(model.train_on_batch(x, y)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            sf.write(audio / "synthetic.wav", np.sin(np.arange(8000)).astype(np.float32) * .1, 8000)
            uncompiled = tf.keras.Model(model.inputs, model.outputs)
            model_path, bundle_path = root / "model.keras", root / "bundle.json"
            uncompiled.save(model_path)
            bundle = dict(schema_version=1, config=asdict(self.config), model_sha256=file_digest(model_path),
                thresholds={name: .5 for name in self.config.classes}, threshold_source="validation")
            write_json(bundle_path, bundle)
            loaded, _, _, _ = load_bundle(model_path, bundle_path)
            np.testing.assert_allclose(loaded.predict_on_batch(x), model.predict_on_batch(x), atol=1e-6)
            output = root / "output"
            inference(audio, model_path, bundle_path, output)
            before = (output / "predictions.csv").read_bytes()
            csv_path = next((output / "per_file").glob("*.csv"))
            modification_time = csv_path.stat().st_mtime_ns
            inference(audio, model_path, bundle_path, output)
            self.assertEqual(before, (output / "predictions.csv").read_bytes())
            self.assertEqual(modification_time, csv_path.stat().st_mtime_ns)
            # Corrupted cached output must be regenerated.
            csv_path.write_text("corrupted")
            inference(audio, model_path, bundle_path, output)
            self.assertEqual(before, (output / "predictions.csv").read_bytes())
            # A changed recording with the same name must not reuse the cache.
            import json
            old_manifest = json.loads((output / "manifest.json").read_text())
            sf.write(audio / "synthetic.wav", np.zeros(8000, dtype=np.float32), 8000)
            inference(audio, model_path, bundle_path, output)
            new_manifest = json.loads((output / "manifest.json").read_text())
            key = next(iter(old_manifest))
            self.assertNotEqual(old_manifest[key]["audio_sha256"], new_manifest[key]["audio_sha256"])
            bundle["thresholds"]["bird"] = .7
            write_json(bundle_path, bundle)
            with self.assertRaisesRegex(ValueError, "different model"):
                inference(audio, model_path, bundle_path, output)


if __name__ == "__main__":
    unittest.main()
