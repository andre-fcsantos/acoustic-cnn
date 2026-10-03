"""Shared training and inference preprocessing and v2 augmentation."""
import numpy as np


def load_window(path, start, config):
    import librosa
    audio, _ = librosa.load(path, sr=config.sample_rate, mono=True,
                           offset=float(start), duration=config.window_seconds)
    if not len(audio):
        raise ValueError("Audio window is empty; check its start time.")
    # Retain the original training padding policy for partial windows.
    if len(audio) < config.samples:
        audio = np.pad(audio, (0, config.samples - len(audio)))
    return audio[:config.samples].astype(np.float32)


def mel_spectrogram(audio, config):
    import librosa
    if len(audio) != config.samples or not np.isfinite(audio).all():
        raise ValueError("Audio must contain one finite, fixed-length window.")
    mel = librosa.feature.melspectrogram(y=audio, sr=config.sample_rate,
        n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels,
        fmax=config.fmax, power=2.0, center=True)
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape != config.input_shape[:2]:
        raise ValueError(f"Unexpected Mel shape: {mel.shape}")
    return mel.astype(np.float32)[..., None]


def augment(mel, config, rng):
    import librosa
    out = mel[..., 0].copy()
    if rng.random() <= config.frequency_mask_probability:
        for _ in range(2):
            width = int(rng.integers(1, min(config.max_frequency_mask, out.shape[0]) + 1))
            start = int(rng.integers(0, out.shape[0] - width + 1))
            out[start:start + width] = np.median(out, axis=0, keepdims=True)
    if rng.random() <= config.time_mask_probability:
        for _ in range(2):
            width = int(rng.integers(1, min(config.max_time_mask, out.shape[1]) + 1))
            start = int(rng.integers(0, out.shape[1] - width + 1))
            out[:, start:start + width] = np.median(out, axis=1, keepdims=True)
    low = librosa.mel_frequencies(n_mels=config.n_mels, fmin=0, fmax=config.fmax) <= config.low_frequency_hz
    if low.any() and (~low).any() and rng.random() <= config.low_frequency_probability:
        alpha = rng.uniform(config.low_blend_min, config.low_blend_max)
        background = np.median(out[~low], axis=0, keepdims=True)
        out[low] = (1 - alpha) * out[low] + alpha * background
    return out.astype(np.float32)[..., None]
