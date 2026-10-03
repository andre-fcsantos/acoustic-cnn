"""Generate a technical smoke-test dataset, not ecological observations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from acoustic_cnn.core import write_json


def generate(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Synthetic output must be empty.")
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rate, seconds = 8000, 0.5
    rng = np.random.default_rng(42)
    times = np.arange(int(rate * seconds)) / rate
    rows = []
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        for recording in range(count):
            name = f"synthetic_{split}_{recording:02}.wav"
            windows = []
            for window in range(8):
                labels = [(window >> i) & 1 for i in range(3)]
                audio = rng.normal(0, 0.01, size=len(times))
                for present, frequency in zip(labels, (500, 1200, 2600)):
                    if present:
                        audio += 0.1 * np.sin(2 * np.pi * frequency * times)
                windows.append(audio)
                rows.append(dict(recording_id=Path(name).stem, audio_file=name,
                    start_seconds=window * seconds, split=split,
                    bird=labels[0], frog=labels[1], insect=labels[2]))
            sf.write(audio_dir / name, np.concatenate(windows), rate, subtype="PCM_16")
    pd.DataFrame(rows).to_csv(output / "windows.csv", index=False)
    write_json(output / "config.json", dict(classes=["bird", "frog", "insect"],
        sample_rate=rate, window_seconds=seconds, n_fft=256, hop_length=128,
        n_mels=32, fmax=4000, low_frequency_hz=1000, epochs=1, batch_size=8))
    print(f"Synthetic smoke-test data saved to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    generate(parser.parse_args().output)
