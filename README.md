# Acoustic CNN Pipeline

A configurable Python pipeline for multilabel acoustic classification, developed
by André Felipe Carneiro dos Santos. The public example uses the labels `bird`,
`frog`, and `insect`; users provide their own reviewed labels and recordings.
This is group-level acoustic classification, not species identification.

The CNN v2 uses six convolutional layers, batch normalization, two pooling stages,
global average pooling, dropout and independent sigmoid outputs. Multiple classes
can be present in the same window. The training workflow uses moderately weighted
binary cross-entropy, training-only augmentation, and thresholds selected on the
validation split. No pretrained model or research-specific weights are included.

## Quick start in VS Code (Windows)

Extract this project into a **new folder**, outside your existing research project.
Open that folder using **File → Open Folder**. Open **Terminal → New Terminal**.
The example below uses Python 3.12, the Python version used for local verification.
TensorFlow support depends on the Python version and operating system; keep your
existing working research environment separate.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m acoustic_cnn --help
```

Use **Python: Select Interpreter** in the command palette to select this virtual
environment. The explicit Python paths above also work without activation.

## Try the pipeline with synthetic audio

The generated audio consists of artificial tones and noise. It verifies software
execution and does not demonstrate accuracy on ecological recordings.

```powershell
.venv\Scripts\python.exe examples/generate_synthetic.py --output data/synthetic
.venv\Scripts\python.exe -m acoustic_cnn validate --dataset data/synthetic/windows.csv --config data/synthetic/config.json --audio-dir data/synthetic/audio
.venv\Scripts\python.exe -m acoustic_cnn train --dataset data/synthetic/windows.csv --config data/synthetic/config.json --audio-dir data/synthetic/audio --output models/demo
.venv\Scripts\python.exe -m acoustic_cnn evaluate --dataset data/synthetic/windows.csv --audio-dir data/synthetic/audio --model models/demo/model.keras --bundle models/demo/model_bundle.json --output outputs/demo_test
.venv\Scripts\python.exe -m acoustic_cnn infer --audio-dir data/synthetic/audio --model models/demo/model.keras --bundle models/demo/model_bundle.json --output outputs/demo_inference
.venv\Scripts\python.exe -m acoustic_cnn report --predictions outputs/demo_test/predictions.csv --config data/synthetic/config.json --history models/demo/training_history.csv --output outputs/demo_figures
```

Training writes `training_history.csv`, the best weights, an uncompiled
`model.keras`, a `model_bundle.json`, and validation predictions and metrics.
Evaluation uses the **test** split with frozen validation thresholds. Directory
inference writes one CSV per recording, a resumable manifest and `predictions.csv`.
Every writing command requires an explicit output path.
The report command creates class confusion matrices, ROC and precision–recall
curves, and optionally a training-history plot. These plots use labelled evaluation
predictions; unlabelled inference cannot establish accuracy.

## Prepare your own dataset

Use a CSV with one row per labelled audio window:

| Column | Meaning |
| --- | --- |
| `recording_id` | Unique recording identifier shared by its windows |
| `audio_file` | Exact WAV path relative to `--audio-dir`, including subdirectories |
| `start_seconds` | Nonnegative start time of the window |
| `split` | `train`, `val` or `test` |
| One column per configured class | Binary presence label: `0` or `1` |

See `examples/windows_schema.csv` for the schema; its filenames are fictional and
no corresponding audio is included. Copy `configs/example.json` and edit it for
your analysis. Labels must come from your annotation process; unannotated windows
must not automatically be treated as verified negatives.

Keep all windows from each recording in one split. The validator also rejects
multiple recording identifiers for the same audio path, duplicate windows, missing
values and invalid labels. Separate copied audio files with different paths are
not detected as duplicates by this dataset validator; deduplicate your recordings
before assigning groups.

For a **new** dataset without a split column, you may run:

```powershell
.venv\Scripts\python.exe -m acoustic_cnn split --dataset data/windows_unsplit.csv --config configs/example.json --output data/windows_split.csv
```

This creates approximate 70/15/15 splits by recording groups. It does not
stratify by class, campaign or site. Inspect class balance afterwards. Use
`--group-column site_id` to keep all recordings from a site together when testing
generalization to unseen sites. Do not regenerate an existing experimental split.

## Reuse an existing private v2 model

You can validate and use an existing model without retraining. Keep the model and
its original `config_cnn_v2.json` in a local folder. The migration helper reads the
saved validation thresholds and preserves the original output order while mapping
the labels to English. It does not guess thresholds or replace the weights.

```powershell
.venv\Scripts\python.exe examples/import_existing_v2.py --model private/cnn_multilabel_v2.keras --legacy-config private/config_cnn_v2.json --output private/model_bundle.json
.venv\Scripts\python.exe -m acoustic_cnn infer --audio-dir data/audio --model private/cnn_multilabel_v2.keras --bundle private/model_bundle.json --output outputs/private_inference
```

Keep those files private. If you do not have the original saved configuration,
recover it before using the model; the output shape cannot establish class order
or threshold values. Use the same compatible environment that originally saved
your model if deserialization fails.

For the original window CSV, the optional converter preserves the existing split
and omits extra research-specific columns:

```powershell
.venv\Scripts\python.exe -m acoustic_cnn convert-dataset --dataset private/dataset_janelas_multilabel_split.csv --audio-dir data/audio --config configs/example.json --class-map '{"AVE":"bird","ANURO":"frog","INSETO":"insect"}' --output data/windows.csv
```

If your Windows shell alters the quotation marks in that JSON argument, run the
command in the VS Code PowerShell terminal and keep the outer single quotes.

## Resume inference safely

Rerun the same `infer` command. Existing outputs are reused only if the model,
bundle, audio content and output CSV checksums match. Changing model weights or
thresholds requires a new output directory. Changed recordings are reprocessed.
Only currently present input recordings enter the consolidated CSV. A damaged or
missing per-recording CSV is regenerated. File errors are recorded in the manifest
and cause a nonzero exit status after processing the other files.

The current implementation loads one full recording into memory at a time,
then computes predictions in batches. Inference uses complete nonoverlapping
windows and reports the discarded tail duration. Training retains the original
short-window zero-padding policy. Resampling a full file and loading an individual
offset can differ near a window boundary; use audio at the configured sample rate
when strict boundary consistency is required.

## Scientific interpretation

Validation results used for threshold selection are development results, not an
independent estimate of accuracy. Report held-out test metrics separately. A
recording-level split does not establish transferability to unseen sites,
campaigns, devices or species. Choose the grouping unit to match your research
question and review domain shifts using independent labelled recordings.

Sigmoid outputs are model scores; this pipeline does not establish probability
calibration. Class weights, low-frequency augmentation and decision thresholds
are methodological choices that require validation. When a test class has only
one outcome, ROC AUC is reported as undefined rather than invented.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests check recording leakage, labels, thresholds, class support, preprocessing,
model serialization and inference resumption. Audio/model tests require the
installed TensorFlow and librosa dependencies. See `docs/VALIDATION.md` for the
checks completed during this refactor.

## Project organization

- `src/acoustic_cnn/`: reusable preprocessing, CNN, training, evaluation and CLI.
- `configs/`: editable generic settings.
- `examples/`: schema, synthetic demonstration and private-model migration helper.
- `tests/`: executable checks.
- `docs/`: migration notes and verification record.

The original experimental scripts, unpublished observations, trained weights,
study-specific thresholds and research results are not part of this public bundle.
They remain in the original research project. This software release does not
rerun or replace that experiment.

## Authorship and reuse

Developed by André Felipe Carneiro dos Santos.

The source code is distributed under the [MIT License](LICENSE).
Copyright (c) 2026 André Felipe Carneiro dos Santos. Reuse and redistribution
must retain the copyright and license notice.

The license applies to the software distributed in this repository.
Private recordings, unpublished datasets and research model weights are not
distributed here.
