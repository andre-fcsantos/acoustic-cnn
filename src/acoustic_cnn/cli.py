"""Command line interface; importing modules does not start an analysis."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
from .core import load_config, read_dataset, write_json


def main():
    parser = argparse.ArgumentParser(description="Multilabel acoustic CNN pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check a window dataset and recording splits")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--config", required=True)
    validate.add_argument("--audio-dir")
    split = commands.add_parser("split", help="Create recording-level splits for a NEW dataset")
    split.add_argument("--dataset", required=True)
    split.add_argument("--config", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--group-column", default="recording_id")
    for command in ("train", "evaluate", "infer"):
        sub = commands.add_parser(command)
        sub.add_argument("--audio-dir", required=True)
        sub.add_argument("--output", required=True)
        if command != "infer":
            sub.add_argument("--dataset", required=True)
        if command == "train":
            sub.add_argument("--config", required=True)
        else:
            sub.add_argument("--model", required=True)
            sub.add_argument("--bundle", required=True)
    migration = commands.add_parser("convert-dataset", help="Convert the original private window CSV")
    migration.add_argument("--dataset", required=True)
    migration.add_argument("--output", required=True)
    migration.add_argument("--config", required=True)
    migration.add_argument("--audio-dir", required=True)
    migration.add_argument("--class-map", required=True,
        help='JSON mapping of original to configured labels, e.g. {"OLD_A":"bird"}')
    bundle = commands.add_parser("bundle-model", help="Associate a PRIVATE existing model with its saved thresholds")
    bundle.add_argument("--model", required=True)
    bundle.add_argument("--config", required=True)
    bundle.add_argument("--thresholds", required=True,
        help="JSON dictionary of class names to validation-selected thresholds")
    bundle.add_argument("--output", required=True)
    report = commands.add_parser("report", help="Plot diagnostics from labelled predictions")
    report.add_argument("--predictions", required=True)
    report.add_argument("--config", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--history")
    args = parser.parse_args()
    if args.command == "validate":
        frame = read_dataset(args.dataset, load_config(args.config), args.audio_dir)
        print(frame.groupby("split").agg(windows=("recording_id", "size"),
                                        recordings=("recording_id", "nunique")))
    elif args.command == "split":
        from sklearn.model_selection import GroupShuffleSplit
        config = load_config(args.config)
        frame = pd.read_csv(args.dataset, dtype={"recording_id": str, "audio_file": str})
        if "split" in frame:
            raise ValueError("Dataset already has split assignments. Preserve the existing experiment.")
        if args.group_column not in frame or frame[args.group_column].isna().any():
            raise ValueError("Missing grouping column or values.")
        if frame[args.group_column].nunique() < 5:
            raise ValueError("Use at least five independent groups for automatic splitting.")
        train, other = next(GroupShuffleSplit(n_splits=1, train_size=0.7,
            random_state=config.seed).split(frame, groups=frame[args.group_column]))
        rest = frame.iloc[other]
        val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.5,
            random_state=config.seed).split(rest, groups=rest[args.group_column]))
        frame["split"] = "train"
        frame.loc[frame.index[other[val]], "split"] = "val"
        frame.loc[frame.index[other[test]], "split"] = "test"
        # This command is for new experiments; no stratification is implied.
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        # Validate before saving, including one split per recording.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "dataset.csv"
            frame.to_csv(temporary, index=False)
            read_dataset(temporary, config)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    elif args.command == "convert-dataset":
        from .core import resolve_audio
        config = load_config(args.config)
        frame = pd.read_csv(args.dataset, dtype={"gravacao_id": str, "IN FILE": str})
        mapping = json.loads(args.class_map)
        if set(mapping.values()) != set(config.classes) or len(mapping) != len(config.classes):
            raise ValueError("Class mapping must cover every configured class exactly once.")
        frame = frame.rename(columns={"gravacao_id": "recording_id", "IN FILE": "audio_file",
            "inicio_janela_s": "start_seconds", **mapping})
        needed = ["recording_id", "audio_file", "start_seconds", "split", *config.classes]
        if not set(needed) <= set(frame):
            raise ValueError("Missing original window dataset columns.")
        # Resolve exact relative paths first; unique basenames are a migration fallback.
        root = Path(args.audio_dir).resolve()
        index = {}
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".wav":
                index.setdefault(path.name.casefold(), []).append(path)
        def relative(name):
            name = str(name).replace("\\", "/")
            try:
                return resolve_audio(root, name).relative_to(root).as_posix()
            except (ValueError, FileNotFoundError):
                candidates = index.get(Path(name).name.casefold(), [])
                if len(candidates) != 1:
                    raise ValueError(f"Expected one unambiguous WAV match for {name}")
                return candidates[0].relative_to(root).as_posix()
        frame["audio_file"] = frame.audio_file.map(relative)
        destination = Path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "dataset.csv"
            frame[needed].to_csv(temporary, index=False)
            read_dataset(temporary, config, root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame[needed].to_csv(destination, index=False)
    elif args.command == "bundle-model":
        from .core import file_digest
        from .pipeline import load_bundle
        config = load_config(args.config)
        thresholds = json.loads(Path(args.thresholds).read_text())
        if set(thresholds) != set(config.classes):
            raise ValueError("Threshold names must match configured classes and model output order.")
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        value = dict(schema_version=1, config=asdict(config), thresholds=thresholds,
                     model_sha256=file_digest(args.model), threshold_source="validation")
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "bundle.json"
            write_json(temporary, value)
            load_bundle(args.model, temporary)
        write_json(args.output, value)
    elif args.command == "report":
        from .report import plot_report
        plot_report(args.predictions, load_config(args.config), args.output, args.history)
    else:
        from .pipeline import train, evaluate, inference
        if args.command == "train":
            train(args.dataset, args.audio_dir, load_config(args.config), args.output)
        elif args.command == "evaluate":
            evaluate(args.dataset, args.audio_dir, args.model, args.bundle, args.output)
        else:
            inference(args.audio_dir, args.model, args.bundle, args.output)


if __name__ == "__main__":
    main()
