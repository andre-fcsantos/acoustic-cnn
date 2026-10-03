# Verification record

Verification date: 2026-10-03.

## Environment actually exercised

Linux, Python 3.12.14, TensorFlow CPU 2.21.0, Keras 3.15.1,
librosa 1.0.0, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0,
Matplotlib 3.10.8 and soundfile 0.14.0. Windows was not exercised in this
environment. Use the README setup steps and verify locally before applying the
refactor to the original private study.

## Completed checks

- All 16 executable unittest cases passed, including a real TensorFlow batch
  training step, model serialization/reloading, audio preprocessing and
  augmentation, recording leakage, invalid labels and threshold selection.
- Inference resumption reused an unchanged output, regenerated a corrupted
  per-recording CSV, reprocessed changed audio and rejected changed thresholds
  in the same output directory.
- The complete synthetic CLI demonstration trained for one epoch, saved the
  best checkpoint, selected validation thresholds, evaluated 16 held-out test
  windows and produced 64 directory-inference windows from eight synthetic WAVs.
- Reporting generated confusion matrices, ROC and precision–recall curves,
  and a training-history figure. The confusion matrices were visually inspected.
- The existing-model migration helper accepted a synthetic legacy-format v2
  configuration and verified its model shape and checksum.
- Legacy dataset conversion preserved all synthetic labels and existing split
  assignments, confirmed by exact dataframe comparison.

These are software checks. They are not scientific performance claims. The
synthetic data consists of simple artificial tones and noise, and one training
epoch is used only to verify the workflow.

## Remaining private validation

The uploaded ZIP contained source scripts, but no trained model weights or audio
recordings. Prediction equivalence against the original research checkpoint was
therefore not tested. Run the existing-model migration helper and compare a small
private set of recordings with the original inference script before replacing
the research workflow. This release does not include the original RF comparison,
annotation/cluster-selection procedure or study-specific figures.

TensorFlow printed informational oneDNN messages and retracing warnings during
repeated short tests. Keras emitted an upstream NumPy array-conversion
deprecation warning. These did not prevent the tests or synthetic workflow from
completing.

## API references consulted

- [TensorFlow ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
- [scikit-learn binary precision, recall, F-score and support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)

The project is a prepared software release candidate. It has not been uploaded
to a GitHub repository or a Python package registry as part of this verification.
