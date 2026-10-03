# Migration from the original research scripts

## Source mapping

| Original component | Public component |
| --- | --- |
| `cnn_multilabel_v2_epoch17_backup.keras.py` architecture and augmentation | `model.py`, `audio.py` |
| `11_retomar_cnn_v2.py` corrected class support and validation metrics | `core.py`, `pipeline.py` |
| `04d_inferencia_completa_cnn_v2.py` directory inference and resumable outputs | `pipeline.py` |
| `02_preparacao_dataset.py` frozen window dataset | `convert-dataset` preserves its existing labels and split |
| Experimental plots, ablations, RF baselines and study synthesis | Kept in the private original project |

## Preserved v2 design

Default audio settings: 48 kHz mono, 5-second windows, FFT 2048, hop 512,
128 Mel bins, maximum frequency 24 kHz and 469 frames. Mel power is converted to
relative dB using the per-window maximum. The convolutional blocks have 16, 32
and 64 filters, with two convolutions each. Pooling follows the first two blocks.
Spatial dropout is 0.10; the dense hidden layer has 64 units and dropout 0.40.
Outputs are independent sigmoid scores. Positive weights use the square root
of the training negative-to-positive ratio. Augmentation is training-only.

## Deliberate software changes

- English public configuration, messages and documentation.
- Explicit input and output paths; no search for a study-specific project root.
- Shared preprocessing implementation for training and inference.
- All provided split assignments checked, including the test split.
- Class support counted directly from the true positive labels. The old training
  file called `int(support)` after requesting averaged binary metrics; the
  later resumption script had already corrected this.
- Ambiguous same-name WAV files rejected during migration. Equal file sizes are
  not evidence of equal contents.
- Thresholds read from a model-associated bundle, not copied into inference code.
- Model and output checksums prevent unintended reuse across inference runs.
- Model output dimension follows the number of configured classes.
- Undefined test AUC is represented as missing.
- Fresh training reloads the best checkpoint before exporting and selecting thresholds.

The new training command starts a fresh run; it does not implement historical
epoch-17 resumption or claim to recreate the original checkpoint. The existing
checkpoint can be used privately through the migration helper. Group splitting
for new data does not reproduce the campaign-stratified historical split. Always
use the original frozen split when comparing against the research experiment.

## Before using private recordings

1. Retain a backup of the complete original project and its environment.
2. Convert the existing dataset without changing its labels or split assignments.
3. Import the saved model and original configuration; verify class order.
4. Compare predictions for a small set of private recordings against the original
   inference script in the same environment before applying the refactor widely.
5. Keep audio, metadata, weights and generated predictions in ignored local folders.

The public pipeline accepts reviewed window labels. It does not reproduce the
study-specific cluster selection or automatically infer negative labels from
missing annotations. Those annotation decisions remain part of the private
research workflow.
