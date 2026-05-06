"""
visualize.py – Training plots, embedding visualization, and final results table.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

import config


# ─── Color palette ───────────────────────────────────────────────────────────
COLORS = plt.cm.tab10.colors


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(history: dict, run_name: str):
    """
    Generates a figure with:
      - Train & Val loss
      - Val accuracy
      - Distillation loss (if present)
    """
    has_distil = "distill_loss" in history and len(history["distill_loss"]) > 0
    n_plots    = 3 if has_distil else 2
    fig, axes  = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    fig.suptitle(f"Training Curves – {run_name}", fontsize=13)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"],   label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], color="tab:green", label="val acc")
    if "val_f1" in history:
        ax.plot(epochs, history["val_f1"], color="tab:orange",
                linestyle="--", label="val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Accuracy / F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Distillation loss ─────────────────────────────────────────────────
    if has_distil:
        ax = axes[2]
        ax.plot(epochs, history["distill_loss"], color="tab:red",
                label="distil loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Distillation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, f"{run_name}_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[viz] Curves saved: {path}")


# ─── Embedding extraction ────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader,
                       device: torch.device,
                       max_samples: int = config.VIZ_SAMPLES):
    """Extract embeddings and labels from a loader."""
    model.eval()
    all_embs, all_labels = [], []
    total = 0

    for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
        imgs, labels = batch[0].to(device), batch[1]
        out = model(imgs)

        emb = out[1] if isinstance(out, (tuple, list)) else model.get_embedding(imgs)

        all_embs.append(emb.cpu())
        all_labels.append(labels)

        total += imgs.size(0)
        if total >= max_samples:
            break

    embs   = torch.cat(all_embs,   dim=0)[:max_samples]
    labels = torch.cat(all_labels, dim=0)[:max_samples]

    return embs.numpy(), labels.numpy()


# ─── PCA / t-SNE ─────────────────────────────────────────────────────────────

def _reduce_embeddings(embs: np.ndarray, method: str = "pca") -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=config.SEED)
    else:
        reducer = TSNE(n_components=2, random_state=config.SEED,
                       perplexity=30, n_iter=1000)
    return reducer.fit_transform(embs)


# ─── Embedding plots (single view) ───────────────────────────────────────────

def plot_embedding_comparison(embs_dict: dict, labels: np.ndarray,
                             title: str, filename: str,
                             method: str = "pca"):
    """
    embs_dict: {"Name": np.ndarray (N, D), ...}
    Draws side-by-side plots colored by class.
    """
    n = len(embs_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13)

    mask = labels < config.VIZ_CLASSES

    for ax, (name, embs) in zip(axes, embs_dict.items()):
        reduced = _reduce_embeddings(embs[mask], method=method)
        lbs     = labels[mask]

        for cls in range(config.VIZ_CLASSES):
            idx = lbs == cls
            ax.scatter(reduced[idx, 0], reduced[idx, 1],
                       s=8, alpha=0.6,
                       color=COLORS[cls % 10],
                       label=str(cls))

        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=7,
                  loc="best", title="Class", ncol=2)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[viz] Embeddings saved: {path}")


# ─── Embedding plots (multi views with fewer classes) ─────────────────────────

def plot_embedding_multi_views(embs_dict: dict, labels: np.ndarray,
                              title_prefix: str,
                              filename_prefix: str,
                              class_splits=(3, 5, 10),
                              method="pca"):
    """
    Generates multiple embedding plots using fewer classes.
    """

    for n_classes in class_splits:
        mask = labels < n_classes

        n = len(embs_dict)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"{title_prefix} – First {n_classes} classes", fontsize=13)

        for ax, (name, embs) in zip(axes, embs_dict.items()):
            reduced = _reduce_embeddings(embs[mask], method=method)
            lbs     = labels[mask]

            for cls in range(n_classes):
                idx = lbs == cls
                ax.scatter(reduced[idx, 0], reduced[idx, 1],
                           s=8, alpha=0.6,
                           color=COLORS[cls % 10],
                           label=str(cls))

            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(markerscale=2, fontsize=7,
                      loc="best", title="Class", ncol=2)

        plt.tight_layout()
        path = os.path.join(config.OUTPUT_DIR,
                            f"{filename_prefix}_{n_classes}cls.png")
        plt.savefig(path, dpi=120)
        plt.close()

        print(f"[viz] Embeddings ({n_classes} classes) saved: {path}")


# ─── Final results table and plot ─────────────────────────────────────────────

def plot_final_results(results: list):
    """
    results: list of dicts with keys:
      model, training_type, val_accuracy, test_accuracy, parameters
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(results) * 0.6 + 2)))

    # ── Table ─────────────────────────────────────────────────────────────
    ax_tbl = axes[0]
    ax_tbl.axis("off")

    col_labels = ["Model", "Type", "Val Acc", "Test Acc", "Parameters"]

    rows = []
    for r in results:
        rows.append([
            r["model"],
            r["training_type"],
            f"{r['val_accuracy']:.4f}",
            f"{r['test_accuracy']:.4f}",
            f"{r['parameters']:,}",
        ])

    tbl = ax_tbl.table(cellText=rows,
                       colLabels=col_labels,
                       cellLoc="center",
                       loc="center")

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    ax_tbl.set_title("Results Table", fontsize=12, pad=10)

    # ── Accuracy plot ─────────────────────────────────────────────────────
    ax_plt = axes[1]

    model_names = list({r["model"] for r in results})
    x = np.arange(len(model_names))
    width = 0.35

    base_accs = []
    distil_accs = []

    for m in model_names:
        bline = next((r for r in results
                      if r["model"] == m and r["training_type"] == "Baseline"), None)
        dline = next((r for r in results
                      if r["model"] == m and r["training_type"] == "Distillation"), None)

        base_accs.append(bline["test_accuracy"] if bline else 0)
        distil_accs.append(dline["test_accuracy"] if dline else 0)

    ax_plt.bar(x - width / 2, base_accs, width,
               label="Baseline", color="steelblue")
    ax_plt.bar(x + width / 2, distil_accs, width,
               label="Distillation", color="darkorange")

    ax_plt.set_xticks(x)
    ax_plt.set_xticklabels(model_names, rotation=15, ha="right")
    ax_plt.set_ylabel("Test Accuracy")
    ax_plt.set_title("Baseline vs Distillation by Model")
    ax_plt.legend()
    ax_plt.grid(axis="y", alpha=0.3)
    ax_plt.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "final_results.png")
    plt.savefig(path, dpi=120)
    plt.close()

    print(f"[viz] Final results saved: {path}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(config.OUTPUT_DIR, "final_results.csv")

    with open(csv_path, "w") as f:
        f.write("model,training_type,val_accuracy,test_accuracy,parameters\n")
        for r in results:
            f.write(f"{r['model']},{r['training_type']},"
                    f"{r['val_accuracy']:.6f},{r['test_accuracy']:.6f},"
                    f"{r['parameters']}\n")

    print(f"[viz] CSV saved: {csv_path}")
