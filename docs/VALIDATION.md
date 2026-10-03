# Verification record

Verification date: 2026-10-03.

## Environments exercised

The initial verification used Linux, Python 3.12.14, TensorFlow CPU 2.21.0,
Keras 3.15.1, librosa 1.0.0, NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0, Matplotlib 3.10.8 and soundfile 0.14.0.

The maintainer subsequently ran the pipeline on Windows with Python 3.13.
The 16 unittest cases passed, and the synthetic training, evaluation,
reporting and inference commands completed. Exact Windows dependency versions
were not captured in this record; this is not a claim of compatibility with
every Windows or Python configuration.

## Completed checks

- All 16 executable unittest cases passed on Linux and on the maintainer's
  Windows environment, including a real TensorFlow batch training step,
  model serialization/reloading, audio preprocessing and augmentation,
  recording leakage, invalid labels and threshold selection.
- Inference resumption reused an unchanged output, regenerated a corrupted
  per-recording CSV, reprocessed changed audio and rejected changed thresholds
  in the same output directory.
- The Linux synthetic CLI demonstration trained for one epoch, saved the
  best checkpoint, selected validation thresholds, evaluated 16 held-out test
  windows and produced 64 directory-inference windows from eight synthetic WAVs.
  The maintainer also exercised the synthetic workflow on Windows.
- Reporting generated confusion matrices, ROC and precision–recall curves,
  and a training-history figure. The Linux confusion matrices were visually
  inspected.
- The existing-model migration helper accepted a synthetic legacy-format v2
  configuration and verified its model shape and checksum.
- Legacy dataset conversion preserved all synthetic labels and existing split
  assignments, confirmed by exact dataframe comparison.

These are software checks. They are not scientific performance claims.
The synthetic data consists of artificial tones and noise, and one training
epoch is used only to verify execution.

## Comparison with the original private model

The maintainer ran the refactored inference pipeline locally on Windows using
the original private CNN v2 checkpoint. The bundle preserved the original
class order and the thresholds saved in the original inference configuration.
The model was not retrained.

The maintainer compared two recordings against their existing inference CSVs:
one containing six complete five-second windows and another containing
360 complete five-second windows (30 minutes). Window identifiers and start/end
times were checked before comparing the corresponding probabilities and labels.

| Class | Maximum absolute score difference: 6 windows | Maximum absolute score difference: 360 windows | Changed binary predictions in either comparison |
| --- | ---: | ---: | ---: |
| bird | 1.724701e-08 | 1.147881e-08 | 0 |
| frog | 3.814354e-09 | 6.966209e-09 | 0 |
| insect | 9.492493e-09 | 1.459770e-08 | 0 |

All scores matched within an absolute tolerance of 1e-6 with relative tolerance
set to zero. These figures are based on the maintainer's local comparison
outputs. The recordings, checkpoint, thresholds, file identifiers and study
locations remain private and are not distributed in this repository.

These comparisons support inference equivalence for the two checked recordings
under the tested configuration. They do not establish classification accuracy,
equivalence across every recording or environment, or ecological generalization.
Independent labelled test data are needed to assess scientific performance.

## Scope and runtime messages

The public pipeline does not include the original RF comparison,
annotation/cluster-selection procedure or study-specific figures.

TensorFlow printed informational oneDNN messages and retracing warnings during
repeated short tests. Keras emitted an upstream NumPy array-conversion
deprecation warning. Windows reported that native GPU support was unavailable
for the installed TensorFlow version. These messages did not prevent the
reported checks from completing.

## Publication and automated checks

The source code was published at
[andre-fcsantos/acoustic-cnn](https://github.com/andre-fcsantos/acoustic-cnn)
on 2026-10-03. GitHub Actions runs the unittest suite on Ubuntu with Python 3.12
for pushes and pull requests; current results are available on the
[Actions page](https://github.com/andre-fcsantos/acoustic-cnn/actions).

This record does not claim publication to a Python package registry.

## API references consulted

- [TensorFlow ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
- [scikit-learn binary precision, recall, F-score and support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
