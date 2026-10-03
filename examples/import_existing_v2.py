"""Build a private bundle from the original v2 JSON without retraining."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
from acoustic_cnn.core import Config, file_digest, write_json
from acoustic_cnn.pipeline import load_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old = json.loads(Path(args.legacy_config).read_text(encoding="utf-8"))
    mapping = {"AVE": "bird", "ANURO": "frog", "INSETO": "insect"}
    if set(old["classes"]) != set(mapping):
        raise ValueError("This migration helper supports the original three v2 classes only.")
    config = Config(classes=tuple(mapping[c] for c in old["classes"]),
        sample_rate=old["sr"], window_seconds=old["duracao_s"],
        n_fft=old["n_fft"], hop_length=old["hop_length"], n_mels=old["n_mels"],
        fmax=old["fmax"], seed=old.get("seed", 42), batch_size=old.get("batch_size", 16))
    if config.frames != old["n_frames"]:
        raise ValueError("Legacy frame count does not match its preprocessing settings.")
    bundle = dict(schema_version=1, config=asdict(config),
        model_sha256=file_digest(args.model), threshold_source="validation",
        thresholds={mapping[c]: old["thresholds_validation"][c] for c in old["classes"]})
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "bundle.json"
        write_json(temporary, bundle)
        load_bundle(args.model, temporary)
    write_json(destination, bundle)
    print(f"Existing model verified. Private bundle saved to {destination.resolve()}")


if __name__ == "__main__":
    main()
