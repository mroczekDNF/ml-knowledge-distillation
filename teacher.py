"""
teacher.py – Fine-tuning del teacher en Tiny ImageNet y caché de embeddings.

CRÍTICO:
  1) El teacher DEBE ser entrenado en Tiny ImageNet antes de cualquier destilación.
  2) Los embeddings se calculan UNA SOLA VEZ y se guardan en disco.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models import get_teacher, count_parameters


# ─── Utilidad: construir optimizer ───────────────────────────────────────────

def _make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    if config.OPTIMIZER == "adam":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=config.WEIGHT_DECAY,
        )
    return torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config.WEIGHT_DECAY, momentum=0.9,
    )


# ─── Fine-tuning ──────────────────────────────────────────────────────────────

def finetune_teacher(train_loader: DataLoader,
                     val_loader: DataLoader,
                     device: torch.device) -> nn.Module:
    """
    Hace fine-tuning de ResNet-50 en Tiny ImageNet.
    Guarda el checkpoint en TEACHER_CKPT.
    Carga el checkpoint si ya existe.
    """
    teacher = get_teacher().to(device)

    if os.path.isfile(config.TEACHER_CKPT):
        print(f"[teacher] Cargando checkpoint: {config.TEACHER_CKPT}")
        teacher.load_state_dict(
            torch.load(config.TEACHER_CKPT, map_location=device)
        )
        teacher.eval()
        return teacher

    print(f"[teacher] Parámetros: {count_parameters(teacher):,}")
    print(f"[teacher] Iniciando fine-tuning ({config.TEACHER_EPOCHS} épocas) …")

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(teacher, config.TEACHER_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.TEACHER_EPOCHS
    )

    best_acc  = 0.0

    for epoch in range(1, config.TEACHER_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        teacher.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"[Teacher] Época {epoch}/{config.TEACHER_EPOCHS}",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = teacher(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # ── Val ────────────────────────────────────────────────────────────
        val_acc = _evaluate(teacher, val_loader, device)
        print(f"[teacher] Época {epoch:3d} | loss_train={train_loss:.4f} "
              f"| val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(teacher.state_dict(), config.TEACHER_CKPT)

    print(f"[teacher] Fine-tuning completado. Mejor val_acc={best_acc:.4f}")
    teacher.load_state_dict(torch.load(config.TEACHER_CKPT, map_location=device))
    teacher.eval()
    return teacher


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader,
              device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ─── Caché de embeddings ──────────────────────────────────────────────────────

@torch.no_grad()
def cache_teacher_embeddings(teacher: nn.Module,
                              train_loader: DataLoader,
                              embed_loader: DataLoader,
                              device: torch.device) -> dict:
    """
    Extrae los embeddings del teacher sobre el conjunto de train UNA SOLA VEZ
    y los guarda en disco. Devuelve un dict {global_index: embedding_tensor}.

    Usamos embed_loader (sin augmentación) para obtener embeddings reproducibles.
    """
    if os.path.isfile(config.TEACHER_EMBED_CACHE):
        print(f"[teacher] Cargando embeddings cacheados: {config.TEACHER_EMBED_CACHE}")
        return torch.load(config.TEACHER_EMBED_CACHE, map_location="cpu")

    print("[teacher] Calculando y cacheando embeddings del teacher …")
    teacher.eval()

    all_embeddings = []   # lista de tensores (B, D)
    all_indices    = []   # índices globales

    global_idx = 0
    for imgs, _ in tqdm(embed_loader, desc="[Teacher] Embeddings"):
        imgs = imgs.to(device)
        _, emb = teacher(imgs)
        all_embeddings.append(emb.cpu())
        batch_size = imgs.size(0)
        all_indices.extend(range(global_idx, global_idx + batch_size))
        global_idx += batch_size

    embeddings_tensor = torch.cat(all_embeddings, dim=0)   # (N, 2048)

    cache = {
        "embeddings": embeddings_tensor,   # Tensor (N, 2048)
        "indices":    all_indices,          # lista de enteros
    }
    torch.save(cache, config.TEACHER_EMBED_CACHE)
    print(f"[teacher] Embeddings guardados: {embeddings_tensor.shape}")
    return cache
