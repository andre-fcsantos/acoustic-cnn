"""Plots of labelled validation or test predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from .core import check_probabilities


def plot_report(predictions, config, output, history=None):
    import matplotlib.pyplot as plt
    frame = pd.read_csv(predictions)
    required = [*config.classes, *[f"prob_{c}" for c in config.classes],
                *[f"pred_{c}" for c in config.classes]]
    if not set(required) <= set(frame):
        raise ValueError("Report requires labelled validation/test predictions, not unlabelled inference.")
    y = frame[list(config.classes)].to_numpy()
    probabilities = frame[[f"prob_{c}" for c in config.classes]].to_numpy()
    check_probabilities(y, probabilities)
    predictions = frame[[f"pred_{c}" for c in config.classes]].to_numpy()
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Predictions must be binary.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Report output must be empty.")
    output.mkdir(parents=True, exist_ok=True)
    # rc_context restores global graphical settings after generating the figures.
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        figure, axes = plt.subplots(1, len(config.classes), figsize=(4 * len(config.classes), 3.6), squeeze=False)
        for index, name in enumerate(config.classes):
            axis = axes[0, index]
            counts = confusion_matrix(y[:, index], predictions[:, index], labels=[0, 1])
            axis.imshow(counts, cmap="Blues")
            for row in range(2):
                for column in range(2):
                    axis.text(column, row, str(counts[row, column]), ha="center", va="center",
                        color="white" if counts[row, column] > counts.max() / 2 else "black")
            axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Absent", "Present"],
                yticklabels=["Absent", "Present"], xlabel="Predicted", ylabel="Observed", title=name)
        figure.tight_layout()
        figure.savefig(output / "confusion_matrices.png", dpi=200)
        plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        for index, name in enumerate(config.classes):
            if len(np.unique(y[:, index])) == 2:
                false_positive, true_positive, _ = roc_curve(y[:, index], probabilities[:, index])
                axes[0].plot(false_positive, true_positive, label=name)
                precision, recall, _ = precision_recall_curve(y[:, index], probabilities[:, index])
                axes[1].plot(recall, precision, label=name)
        axes[0].plot([0, 1], [0, 1], "--", color="0.6")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves")
        for axis in axes:
            axis.set(xlim=(0, 1), ylim=(0, 1.02))
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output / "discrimination_curves.png", dpi=200)
        plt.close(figure)
        if history:
            values = pd.read_csv(history)
            if not {"loss", "val_loss"} <= set(values):
                raise ValueError("Training history requires loss and val_loss.")
            figure, axis = plt.subplots(figsize=(6, 4))
            for column, label in (("loss", "Training"), ("val_loss", "Validation")):
                axis.plot(np.arange(1, len(values) + 1), values[column], label=label)
            axis.set(xlabel="Epoch", ylabel="Loss (including regularization)")
            axis.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(output / "training_history.png", dpi=200)
            plt.close(figure)
    print(f"Diagnostic figures saved to {output.resolve()}")
