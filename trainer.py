"""
trainer.py – Loops de entrenamiento para baseline y destilación.

Baseline:   L = CrossEntropy(logits, labels)
Distillation: L = α * CrossEntropy + β * MSE(proj_emb, teacher_emb)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from typing import Optional

import config
from models import count_parameters


# ─── Dataset que adjunta embeddings del teacher ───────────────────────────────

class EmbeddingDataset(Dataset):
    """
    Wrapper sobre un Subset de ImageFolder que adjunta el embedding del teacher.
    Devuelve (img, label, teacher_embedding).
    """

    def __init__(self, base_subset: Subset, teacher_embeddings: torch.Tensor):
        """
        base_subset          : Subset de ImageFolder (con augmentación)
        teacher_embeddings   : Tensor (N, 2048) en el mismo orden que base_subset
        """
        self.base   = base_subset
        self.embeds = teacher_embeddings   # ya en CPU

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        emb        = self.embeds[idx]
        return img, label, emb


# ─── Funciones de pérdida ─────────────────────────────────────────────────────

def distill_loss(student_proj: torch.Tensor,
                 teacher_emb: torch.Tensor,
                 mode: str = config.DISTILL_LOSS) -> torch.Tensor:
    """L_distill entre embedding proyectado del estudiante y embedding del teacher."""
    if mode == "mse":
        return F.mse_loss(student_proj, teacher_emb)
    elif mode == "cosine":
        # 1 - cosine_similarity (convertida en pérdida)
        return (1 - F.cosine_similarity(student_proj, teacher_emb, dim=1)).mean()
    else:
        raise ValueError(f"distill_loss mode desconocido: {mode}")


# ─── Optimizer factory ────────────────────────────────────────────────────────

def _make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if config.OPTIMIZER == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        momentum=0.9,
    )


# ─── Métricas ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    """Devuelve accuracy y f1 sobre el loader dado."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        out    = model(imgs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1": float(f1)}


# ─── Entrenamiento baseline ───────────────────────────────────────────────────

def train_baseline(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   device: torch.device,
                   run_name: str) -> dict:
    """
    Entrenamiento estándar con Cross-Entropy.
    Devuelve historial de métricas.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [],
    }
    best_acc  = 0.0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}.pth")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader,
                          desc=f"[{run_name}] Época {epoch}/{config.NUM_EPOCHS}",
                          leave=False):
            imgs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out    = model(imgs)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Val
        val_metrics = evaluate(model, val_loader, device)
        val_loss    = _val_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"[{run_name}] Época {epoch:3d} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_metrics['f1']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[{run_name}] Mejor val_acc={best_acc:.4f}")
    return history


# ─── Entrenamiento con destilación ───────────────────────────────────────────

def train_distillation(model: nn.Module,
                       train_loader: DataLoader,   # DataLoader de EmbeddingDataset
                       val_loader: DataLoader,
                       device: torch.device,
                       run_name: str) -> dict:
    """
    Entrenamiento con destilación de embeddings.
    train_loader DEBE producir (img, label, teacher_emb).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [],
        "distill_loss": [],
    }
    best_acc  = 0.0
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}.pth")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_total  = 0.0
        running_cls    = 0.0
        running_distil = 0.0

        for imgs, labels, teacher_embs in tqdm(
                train_loader,
                desc=f"[{run_name}] Época {epoch}/{config.NUM_EPOCHS}",
                leave=False):
            imgs        = imgs.to(device)
            labels      = labels.to(device)
            teacher_embs = teacher_embs.to(device)

            optimizer.zero_grad()
            logits, _, proj = model(imgs)

            l_cls     = criterion(logits, labels)
            l_distil  = distill_loss(proj, teacher_embs)
            loss      = config.ALPHA * l_cls + config.BETA * l_distil

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_total  += loss.item()     * bs
            running_cls    += l_cls.item()    * bs
            running_distil += l_distil.item() * bs

        scheduler.step()
        n = len(train_loader.dataset)
        train_loss  = running_total  / n
        distil_loss = running_distil / n

        val_metrics = evaluate(model, val_loader, device)
        val_loss    = _val_loss_cls(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["distill_loss"].append(distil_loss)

        print(f"[{run_name}] Época {epoch:3d} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"distil={distil_loss:.4f} | "
              f"acc={val_metrics['accuracy']:.4f} | "
              f"f1={val_metrics['f1']:.4f}")

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[{run_name}] Mejor val_acc={best_acc:.4f}")
    return history


# ─── Helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _val_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        out    = model(imgs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss   = criterion(logits, labels)
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _val_loss_cls(model, loader, criterion, device):
    """Val loss usando solo CrossEntropy (para modelos de distilación)."""
    return _val_loss(model, loader, criterion, device)
