"""Training, held-out evaluation and resumable directory inference."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import numpy as np
import pandas as pd
from .audio import augment, load_window, mel_spectrogram
from .core import (Config, file_digest, metrics, read_dataset, resolve_audio,
                   select_thresholds, write_json)
from .model import build_model, positive_weights, weighted_loss


def make_sequence(frame, audio_root, config, training=False):
    import tensorflow as tf
    class WindowSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.frame = frame.reset_index(drop=True)
            self.indices = np.arange(len(frame))
            self.epoch = 0
            self.paths = {f: resolve_audio(audio_root, f) for f in frame.audio_file.unique()}
            if training:
                self._shuffle()

        def _shuffle(self):
            np.random.default_rng(config.seed + self.epoch).shuffle(self.indices)

        def __len__(self):
            return math.ceil(len(self.frame) / config.batch_size)

        def on_epoch_end(self):
            if training:
                self.epoch += 1
                self._shuffle()

        def __getitem__(self, batch_index):
            ids = self.indices[batch_index * config.batch_size:(batch_index + 1) * config.batch_size]
            batch = self.frame.iloc[ids]
            inputs = []
            for row_index, row in batch.iterrows():
                audio = load_window(self.paths[row.audio_file], row.start_seconds, config)
                mel = mel_spectrogram(audio, config)
                if training:
                    rng = np.random.default_rng(config.seed + self.epoch * 1_000_003 + int(row_index))
                    mel = augment(mel, config, rng)
                inputs.append(mel)
            return np.stack(inputs), batch[list(config.classes)].to_numpy(dtype=np.float32)
    return WindowSequence()


def predict_sequence(model, sequence):
    # Avoid implicit shuffle or tf.data worker behavior during metric extraction.
    return np.concatenate([np.asarray(model.predict_on_batch(sequence[i][0]))
                           for i in range(len(sequence))])


def save_evaluation(frame, probabilities, thresholds, config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    labels = frame[list(config.classes)].to_numpy(dtype=int)
    rows, scores = metrics(labels, probabilities, thresholds, config.classes)
    pd.DataFrame(rows).to_csv(destination / "class_metrics.csv", index=False)
    write_json(destination / "global_metrics.json", scores)
    predictions = frame[["recording_id", "audio_file", "start_seconds"]].copy()
    for index, name in enumerate(config.classes):
        predictions[name] = labels[:, index]
        predictions[f"prob_{name}"] = probabilities[:, index]
        predictions[f"pred_{name}"] = (probabilities[:, index] >= thresholds[index]).astype(int)
    predictions.to_csv(destination / "predictions.csv", index=False)


def train(dataset, audio_root, config, output):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.seed)
    frame = read_dataset(dataset, config, audio_root)
    training = frame[frame.split == "train"].reset_index(drop=True)
    validation = frame[frame.split == "val"].reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError("Training and validation splits must both be nonempty.")
    for name in config.classes:
        if validation[name].nunique() != 2:
            raise ValueError(f"Validation needs positive and negative examples of {name}.")
    weights = positive_weights(training[list(config.classes)].to_numpy())
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output must be empty; choose a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_config.json", asdict(config))
    train_sequence = make_sequence(training, audio_root, config, training=True)
    validation_sequence = make_sequence(validation, audio_root, config)
    model = build_model(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate), loss=weighted_loss(weights))
    # Save weights only: the custom loss is not required for checkpoint loading.
    checkpoint = output / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience,
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
            min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_loss",
            save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(train_sequence, validation_data=validation_sequence,
        epochs=config.epochs, callbacks=callbacks, shuffle=False)
    pd.DataFrame(history.history).to_csv(output / "training_history.csv", index=False)
    if not checkpoint.is_file():
        raise RuntimeError("No valid checkpoint was saved; inspect the training loss.")
    model.load_weights(checkpoint)
    # Export an uncompiled model so inference never depends on loss serialization.
    inference_model = tf.keras.Model(model.inputs, model.outputs, name=model.name)
    model_path = output / "model.keras"
    inference_model.save(model_path)
    probabilities = predict_sequence(inference_model, validation_sequence)
    labels = validation[list(config.classes)].to_numpy(dtype=int)
    thresholds = select_thresholds(labels, probabilities)
    bundle = dict(schema_version=1, config=asdict(config),
        thresholds=dict(zip(config.classes, map(float, thresholds))),
        model_sha256=file_digest(model_path), threshold_source="validation",
        positive_weights=dict(zip(config.classes, map(float, weights))))
    write_json(output / "model_bundle.json", bundle)
    save_evaluation(validation, probabilities, thresholds, config, output / "validation")
    print(f"Model and validation-selected thresholds saved to {output.resolve()}")


def load_bundle(model_path, bundle_path):
    import tensorflow as tf
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or file_digest(model_path) != bundle["model_sha256"]:
        raise ValueError("The model file does not match its bundle.")
    config = Config(**{**bundle["config"], "classes": tuple(bundle["config"]["classes"])})
    thresholds = np.array([bundle["thresholds"][name] for name in config.classes], dtype=float)
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Invalid bundle thresholds.")
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != config.input_shape or model.output_shape[-1] != len(config.classes):
        raise ValueError("Model shape differs from bundle configuration.")
    model.trainable = False
    return model, config, thresholds, bundle


def evaluate(dataset, audio_root, model_path, bundle_path, output):
    model, config, thresholds, _ = load_bundle(model_path, bundle_path)
    frame = read_dataset(dataset, config, audio_root)
    test = frame[frame.split == "test"].reset_index(drop=True)
    if test.empty:
        raise ValueError("No held-out test windows found.")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Evaluation output must be empty.")
    probabilities = predict_sequence(model, make_sequence(test, audio_root, config))
    save_evaluation(test, probabilities, thresholds, config, destination)
    print("Test evaluated using frozen validation thresholds.")


def inference(audio_root, model_path, bundle_path, output):
    import librosa
    import hashlib
    model, config, thresholds, bundle = load_bundle(model_path, bundle_path)
    root, output = Path(audio_root).resolve(), Path(output)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not paths:
        raise ValueError("No WAV files found.")
    job = {"bundle": bundle, "inference_policy": "complete_nonoverlapping_windows_v1"}
    identity_path = output / "inference_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != job:
            raise ValueError("Output belongs to a different model or configuration. Use a new directory.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty and has no inference identity. Use a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(identity_path, job)
    per_file = output / "per_file"
    per_file.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    completed = []
    failures = []
    columns = ["audio_file", "window_id", "start_seconds", "end_seconds"]
    columns += [f"prob_{c}" for c in config.classes] + [f"pred_{c}" for c in config.classes]
    columns += ["n_classes_predicted"]
    for path in paths:
        relative = path.relative_to(root).as_posix()
        key = hashlib.sha256(relative.encode()).hexdigest()
        target = per_file / f"{key}.csv"
        try:
            audio_hash = file_digest(path)
            previous = manifest.get(key, {})
            if (previous.get("status") == "complete" and previous.get("audio_sha256") == audio_hash
                and target.is_file() and previous.get("csv_sha256") == file_digest(target)):
                completed.append(target)
                print(f"Already complete: {relative}")
                continue
            audio, _ = librosa.load(path, sr=config.sample_rate, mono=True)
            count = len(audio) // config.samples
            if not count:
                raise ValueError("No complete audio windows.")
            temporary = target.with_suffix(".tmp.csv")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                for first in range(0, count, config.batch_size):
                    ids = list(range(first, min(first + config.batch_size, count)))
                    batch = np.stack([mel_spectrogram(audio[i * config.samples:(i + 1) * config.samples],
                        config) for i in ids])
                    probabilities = np.asarray(model.predict_on_batch(batch))
                    if probabilities.shape != (len(ids), len(config.classes)) or not np.isfinite(probabilities).all():
                        raise ValueError("Invalid model outputs.")
                    predictions = (probabilities >= thresholds).astype(int)
                    for i, p, prediction in zip(ids, probabilities, predictions):
                        writer.writerow([relative, i + 1, i * config.samples / config.sample_rate,
                            (i + 1) * config.samples / config.sample_rate, *map(float, p),
                            *map(int, prediction), int(prediction.sum())])
            temporary.replace(target)
            manifest[key] = {"audio_file": relative, "status": "complete", "audio_sha256": audio_hash,
                "csv_sha256": file_digest(target), "windows": count,
                "discarded_tail_seconds": (len(audio) % config.samples) / config.sample_rate}
            completed.append(target)
            print(f"Processed {relative}: {count} windows")
        except Exception as error:
            manifest[key] = {"audio_file": relative, "status": "error", "message": str(error)}
            failures.append(relative)
            print(f"Failed {relative}: {error}")
        write_json(manifest_path, manifest)
    if completed:
        target = output / "predictions.csv"
        temporary = target.with_suffix(".tmp.csv")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for path in completed:
                with path.open(newline="", encoding="utf-8") as source:
                    reader = csv.reader(source)
                    if next(reader) != columns:
                        raise ValueError("Incompatible per-file output columns.")
                    writer.writerows(reader)
        temporary.replace(target)
    if failures:
        raise RuntimeError(f"{len(failures)} WAV files failed. Inspect manifest.json and rerun after fixing them.")
