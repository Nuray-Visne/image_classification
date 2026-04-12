"""Plot test loss and accuracy: scratch vs ImageNet transfer (first N epochs).

Expects:
  results/resnet50_scratch_history.csv
  results/resnet50_transfer_history.csv

Run from repo root:  python src/plot_scratch_vs_transfer.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COMPARE_FIRST_N = 10


def main():
    repo_root = Path(__file__).resolve().parent.parent
    results_root = repo_root / "results"
    scratch_csv = results_root / "resnet50_scratch_history.csv"
    transfer_csv = results_root / "resnet50_transfer_history.csv"
    out_path = results_root / "scratch_vs_transfer_first10.png"

    for p in (scratch_csv, transfer_csv):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p.name}. Run train_resnet50_scratch.py and train_resnet50_transfer_learning first."
            )

    df_s = pd.read_csv(scratch_csv).head(COMPARE_FIRST_N)
    df_t = pd.read_csv(transfer_csv).head(COMPARE_FIRST_N)
    n = min(len(df_s), len(df_t), COMPARE_FIRST_N)
    df_s = df_s.iloc[:n]
    df_t = df_t.iloc[:n]
    ep = df_s["epoch"].values

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, df_s["test_loss"], label="Scratch (test)", marker="o", markersize=3)
    ax[0].plot(ep, df_t["test_loss"], label="Transfer (test)", marker="o", markersize=3)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title(f"Test loss: scratch vs transfer (first {n} epochs)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(ep, df_s["test_acc"], label="Scratch (test)", marker="o", markersize=3)
    ax[1].plot(ep, df_t["test_acc"], label="Transfer (test)", marker="o", markersize=3)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title(f"Test accuracy: scratch vs transfer (first {n} epochs)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
