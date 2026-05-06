"""
dataset.py – Pipeline completo de Tiny ImageNet con descarga automática.

Estructura esperada después de la descarga:
    data/tiny-imagenet-200/
        train/   <class_id>/images/*.JPEG
        val/     images/*.JPEG   +  val_annotations.txt
        test/    images/*.JPEG
        wnids.txt
        words.txt
"""

import os
import shutil
import zipfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

import config


# ─── Normalización estándar ImageNet ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    """Devuelve las transformaciones según el split (train / val / test)."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ─── Descarga y extracción ────────────────────────────────────────────────────

def _reporthook(count, block_size, total_size):
    """Muestra progreso durante la descarga."""
    if total_size > 0:
        pct = min(count * block_size * 100 / total_size, 100)
        mb_done = count * block_size / 1e6
        mb_total = total_size / 1e6
        print(f"\r  Descargando… {mb_done:.1f}/{mb_total:.1f} MB  ({pct:.1f}%)",
              end="", flush=True)


def download_tiny_imagenet():
    """Descarga y extrae Tiny ImageNet si no existe."""
    if os.path.isdir(config.DATASET_ROOT):
        print("[dataset] Tiny ImageNet ya existe, omitiendo descarga.")
        return

    print(f"[dataset] Descargando Tiny ImageNet desde {config.DATASET_URL} …")
    urllib.request.urlretrieve(config.DATASET_URL, config.DATASET_ZIP,
                               reporthook=_reporthook)
    print()

    print("[dataset] Extrayendo …")
    with zipfile.ZipFile(config.DATASET_ZIP, "r") as zf:
        zf.extractall(config.DATA_DIR)
    os.remove(config.DATASET_ZIP)
    print("[dataset] Extracción completada.")

    _fix_val_structure()
    print("[dataset] Dataset listo.")


def _fix_val_structure():
    """
    El split val de Tiny ImageNet usa un único directorio flat.
    Reorganizamos a la estructura class_id/images/*.JPEG que espera ImageFolder.
    """
    val_dir     = os.path.join(config.DATASET_ROOT, "val")
    img_dir     = os.path.join(val_dir, "images")
    ann_file    = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.isfile(ann_file):
        return   # ya reorganizado

    print("[dataset] Reorganizando directorio de validación …")
    # Leer anotaciones: filename <tab> class_id <tab> …
    with open(ann_file) as f:
        lines = f.readlines()

    for line in lines:
        parts    = line.strip().split("\t")
        filename = parts[0]
        class_id = parts[1]
        src      = os.path.join(img_dir, filename)
        dst_dir  = os.path.join(val_dir, class_id, "images")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, os.path.join(dst_dir, filename))

    # Limpiar el directorio flat ya vacío
    if os.path.isdir(img_dir) and not os.listdir(img_dir):
        os.rmdir(img_dir)


# ─── Datasets ─────────────────────────────────────────────────────────────────

def get_datasets():
    """
    Devuelve (train_dataset, val_dataset, test_dataset).
    El split test de Tiny ImageNet no tiene etiquetas públicas; usamos
    una parte del train como test y el val oficial como validación.
    """
    download_tiny_imagenet()

    train_full = ImageFolder(
        root=os.path.join(config.DATASET_ROOT, "train"),
        transform=get_transforms("train"),
    )
    val_dataset = ImageFolder(
        root=os.path.join(config.DATASET_ROOT, "val"),
        transform=get_transforms("val"),
    )

    # Dividir train en 90 % train / 10 % test usando índices fijos (reproducible)
    n = len(train_full)
    rng = torch.Generator().manual_seed(config.SEED)
    indices = torch.randperm(n, generator=rng).tolist()
    split   = int(0.9 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = Subset(train_full, train_idx)
    test_dataset  = Subset(
        ImageFolder(
            root=os.path.join(config.DATASET_ROOT, "train"),
            transform=get_transforms("test"),
        ),
        test_idx,
    )

    print(f"[dataset] Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(train_ds, val_ds, test_ds):
    """Devuelve DataLoaders para los tres splits."""
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ─── Loader para cacheo de embeddings (sin augmentación) ─────────────────────

def get_embed_loader(dataset):
    """
    Loader sin augmentación para extraer embeddings reproducibles del teacher.
    El dataset de train tiene augmentación; necesitamos uno sin ella.
    """
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
        # Crear nuevo Subset sobre dataset sin augmentación
        clean_base = ImageFolder(
            root=base.root,
            transform=get_transforms("val"),
        )
        clean_ds = Subset(clean_base, indices)
    else:
        clean_ds = ImageFolder(
            root=dataset.root,
            transform=get_transforms("val"),
        )

    return DataLoader(
        clean_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
