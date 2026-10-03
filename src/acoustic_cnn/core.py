"""Configuration, dataset validation, metrics and output identity."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
    precision_recall_curve, precision_recall_fscore_support, roc_auc_score)


@dataclass(frozen=True)
class Config:
    classes: tuple = ("bird", "frog", "insect")
    sample_rate: int = 48000
    window_seconds: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmax: float = 24000
    seed: int = 42
    batch_size: int = 16
    epochs: int = 40
    patience: int = 5
    learning_rate: float = 0.0003
    l2: float = 0.0001
    frequency_mask_probability: float = 0.8
    time_mask_probability: float = 0.7
    low_frequency_probability: float = 0.5
    max_frequency_mask: int = 12
    max_time_mask: int = 45
    low_frequency_hz: float = 2000
    low_blend_min: float = 0.35
    low_blend_max: float = 0.85

    def __post_init__(self):
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError("Use at least two unique class names.")
        if any(not isinstance(c, str) or not c.isidentifier() for c in self.classes):
            raise ValueError("Class names must be valid Python identifiers.")
        reserved = {"recording_id", "audio_file", "start_seconds", "split"}
        if reserved.intersection(self.classes):
            raise ValueError("Class names conflict with metadata columns.")
        integers = (self.sample_rate, self.n_fft, self.hop_length, self.n_mels,
                    self.batch_size, self.epochs, self.patience,
                    self.max_frequency_mask, self.max_time_mask)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in integers):
            raise ValueError("Audio, batch, epoch and mask sizes must be positive integers.")
        if not np.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite.")
        if self.samples < 1 or not 0 < self.fmax <= self.sample_rate / 2:
            raise ValueError("Invalid audio length or frequency limit.")
        if self.n_mels < 4 or self.frames < 4:
            raise ValueError("The CNN requires at least four frequency bins and frames.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not np.isfinite(self.learning_rate) or not np.isfinite(self.l2) or not 0 < self.learning_rate or self.l2 < 0:
            raise ValueError("Invalid learning rate or regularization.")
        probabilities = (self.frequency_mask_probability, self.time_mask_probability,
                         self.low_frequency_probability, self.low_blend_min, self.low_blend_max)
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError("Probabilities and blend factors must be in [0, 1].")
        if self.low_blend_min > self.low_blend_max or not 0 < self.low_frequency_hz < self.fmax:
            raise ValueError("Invalid low-frequency augmentation settings.")

    @property
    def samples(self):
        return int(round(self.sample_rate * self.window_seconds))

    @property
    def frames(self):
        # librosa centered STFT, including odd n_fft.
        return 1 + (self.samples + 2 * (self.n_fft // 2) - self.n_fft) // self.hop_length

    @property
    def input_shape(self):
        return self.n_mels, self.frames, 1


def load_config(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if "classes" in values:
        values["classes"] = tuple(values["classes"])
    return Config(**values)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_audio(root, relative_path):
    root = Path(root).resolve()
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise ValueError("audio_file must be relative to the audio directory.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("audio_file must stay inside the audio directory.")
    if not path.is_file() or path.suffix.lower() != ".wav":
        raise FileNotFoundError(f"WAV file not found: {relative_path}")
    return path


def read_dataset(path, config, audio_root=None):
    frame = pd.read_csv(path, dtype={"recording_id": str, "audio_file": str})
    required = {"recording_id", "audio_file", "start_seconds", "split", *config.classes}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("The dataset must be nonempty without missing required values.")
    frame["split"] = frame["split"].astype(str).str.lower().str.strip()
    if not set(frame["split"]) <= {"train", "val", "test"}:
        raise ValueError("split must be train, val or test.")
    if (frame.groupby("recording_id")["split"].nunique() > 1).any():
        raise ValueError("Recording leakage: a recording occurs in multiple splits.")
    if (frame.groupby("recording_id")["audio_file"].nunique() > 1).any():
        raise ValueError("Each recording_id must identify exactly one audio file.")
    if (frame.groupby("audio_file")["recording_id"].nunique() > 1).any():
        raise ValueError("The same audio file has multiple recording identifiers.")
    frame["start_seconds"] = pd.to_numeric(frame["start_seconds"], errors="raise")
    starts = frame["start_seconds"].to_numpy(dtype=float)
    if not np.isfinite(starts).all() or (starts < 0).any():
        raise ValueError("Window starts must be finite and nonnegative.")
    if frame.duplicated(["audio_file", "start_seconds"]).any():
        raise ValueError("Duplicate audio windows found.")
    for name in config.classes:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"Labels for {name} must be 0 or 1.")
    if audio_root is not None:
        for relative in frame["audio_file"].unique():
            resolve_audio(audio_root, relative)
    return frame


def check_probabilities(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape != probabilities.shape or labels.ndim != 2 or not len(labels):
        raise ValueError("Labels and probabilities must have matching nonempty 2D shapes.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    return labels, probabilities


def select_thresholds(labels, probabilities):
    labels, probabilities = check_probabilities(labels, probabilities)
    thresholds = []
    for index in range(labels.shape[1]):
        y = labels[:, index]
        if len(np.unique(y)) != 2:
            raise ValueError("Threshold selection requires positives and negatives for every class.")
        precision, recall, candidates = precision_recall_curve(y, probabilities[:, index])
        score = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        thresholds.append(float(candidates[np.argmax(score)]))
    return np.asarray(thresholds)


def metrics(labels, probabilities, thresholds, classes):
    labels, probabilities = check_probabilities(labels, probabilities)
    thresholds = np.asarray(thresholds)
    if thresholds.shape != (len(classes),) or labels.shape[1] != len(classes):
        raise ValueError("Classes, thresholds and output dimensions do not match.")
    if not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Invalid thresholds.")
    predictions = (probabilities >= thresholds).astype(int)
    rows = []
    for index, name in enumerate(classes):
        y, p = labels[:, index], probabilities[:, index]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions[:, index], average="binary", zero_division=0)
        rows.append(dict(class_name=name, threshold=float(thresholds[index]),
            precision=float(precision), recall=float(recall), f1=float(f1),
            support=int(y.sum()),
            roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            average_precision=float(average_precision_score(y, p)) if y.sum() else None))
    global_scores = {f"{average}_f1": float(f1_score(labels, predictions,
        average=average, zero_division=0)) for average in ("micro", "macro", "weighted", "samples")}
    return rows, global_scores
