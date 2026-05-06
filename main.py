"""
main.py – Script principal de orquestación del experimento de Knowledge Distillation.

Flujo:
  1. Descarga y prepara Tiny ImageNet
  2. Fine-tuning del teacher (ResNet-50)
  3. Caché de embeddings del teacher
  4. Para cada modelo estudiante:
       a) Entrenamiento baseline
       b) Entrenamiento con destilación
  5. Evaluación en test
  6. Visualizaciones y tabla final
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader

import config
from dataset import (
    get_datasets, get_dataloaders, get_embed_loader,
)
from models import get_student, get_teacher, count_parameters
from teacher import finetune_teacher, cache_teacher_embeddings
from trainer import EmbeddingDataset, train_baseline, train_distillation, evaluate
from visualize import (
    plot_training_curves,
    plot_embedding_comparison,
    extract_embeddings,
    plot_final_results,
)


# ─── Reproducibilidad ─────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Device: {device}")

    # ── 1. Dataset ────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = get_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds
    )

    # ── 2. Teacher fine-tuning ────────────────────────────────────────────
    teacher = finetune_teacher(train_loader, val_loader, device)
    teacher_params = count_parameters(teacher)
    print(f"[main] Teacher parameters: {teacher_params:,}")

    # Congelar teacher
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # ── 3. Caché de embeddings del teacher ───────────────────────────────
    embed_loader = get_embed_loader(train_ds)
    cache        = cache_teacher_embeddings(teacher, train_loader,
                                            embed_loader, device)
    teacher_embs = cache["embeddings"]   # Tensor (N, 2048), CPU
    print(f"[main] Cached embeddings: {teacher_embs.shape}")

    # ── 4. Preparar loader de distilación ────────────────────────────────
    embed_train_ds = EmbeddingDataset(train_ds, teacher_embs)
    embed_train_loader = DataLoader(
        embed_train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ── 5. Experimentos ───────────────────────────────────────────────────
    results = []
    all_histories = {}

    for model_name in config.STUDENT_MODELS:
        print(f"\n{'='*60}")
        print(f" STUDENT: {model_name}")
        print(f"{'='*60}")

        # ── 5a. Baseline ──────────────────────────────────────────────────
        run_base = f"{model_name}_baseline"
        student_base = get_student(model_name)
        n_params = count_parameters(student_base)
        print(f"[main] {model_name} parameters: {n_params:,}")

        history_base = train_baseline(
            student_base, train_loader, val_loader, device, run_base
        )
        all_histories[run_base] = history_base
        plot_training_curves(history_base, run_base)

        val_metrics_base  = evaluate(student_base, val_loader,  device)
        test_metrics_base = evaluate(student_base, test_loader, device)
        print(f"[{run_base}] Val:  acc={val_metrics_base['accuracy']:.4f}  "
              f"f1={val_metrics_base['f1']:.4f}")
        print(f"[{run_base}] Test: acc={test_metrics_base['accuracy']:.4f}  "
              f"f1={test_metrics_base['f1']:.4f}")
        results.append({
            "model":         model_name,
            "training_type": "Baseline",
            "val_accuracy":  val_metrics_base["accuracy"],
            "test_accuracy": test_metrics_base["accuracy"],
            "parameters":    n_params,
        })

        # ── 5b. Distillation ───────────────────────────────────────────────
        run_kd = f"{model_name}_distillation"
        student_kd = get_student(model_name)

        history_kd = train_distillation(
            student_kd, embed_train_loader, val_loader, device, run_kd
        )
        all_histories[run_kd] = history_kd
        plot_training_curves(history_kd, run_kd)

        val_metrics_kd  = evaluate(student_kd, val_loader,  device)
        test_metrics_kd = evaluate(student_kd, test_loader, device)
        print(f"[{run_kd}] Val:  acc={val_metrics_kd['accuracy']:.4f}  "
              f"f1={val_metrics_kd['f1']:.4f}")
        print(f"[{run_kd}] Test: acc={test_metrics_kd['accuracy']:.4f}  "
              f"f1={test_metrics_kd['f1']:.4f}")
        results.append({
            "model":         model_name,
            "training_type": "Distillation",
            "val_accuracy":  val_metrics_kd["accuracy"],
            "test_accuracy": test_metrics_kd["accuracy"],
            "parameters":    n_params,
        })

        # ── 5c. Visualización de embeddings ────────────────────────────────
        print(f"[main] Generating visualization of embeddings for {model_name} …")
        # Usamos val_loader (sin augmentación, etiquetas conocidas)
        embs_teacher, labels = extract_embeddings(teacher,     val_loader, device)
        embs_base,    _      = extract_embeddings(student_base, val_loader, device)
        embs_kd,      _      = extract_embeddings(student_kd,   val_loader, device)

        # Teacher vs estudiantes
        plot_embedding_comparison(
            {"Teacher (ResNet-50)": embs_teacher,
             f"Baseline ({model_name})": embs_base,
             f"Distillation ({model_name})": embs_kd},
            labels,
            title=f"Embeddings visualization – {model_name}",
            filename=f"{model_name}_embeddings_pca.png",
            method="pca",
        )

    # ── 6. Tabla final ────────────────────────────────────────────────────
    # Añadir teacher a los resultados para referencia
    teacher_val  = evaluate(teacher, val_loader,  device)
    teacher_test = evaluate(teacher, test_loader, device)
    results.insert(0, {
        "model":         "ResNet-50 (Teacher)",
        "training_type": "Fine-tuned",
        "val_accuracy":  teacher_val["accuracy"],
        "test_accuracy": teacher_test["accuracy"],
        "parameters":    teacher_params,
    })
    print(f"\n[teacher] Val:  acc={teacher_val['accuracy']:.4f}  "
          f"f1={teacher_val['f1']:.4f}")
    print(f"[teacher] Test: acc={teacher_test['accuracy']:.4f}  "
          f"f1={teacher_test['f1']:.4f}")

    plot_final_results(results)

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"{'Model':<30} {'Type':<15} {'Val Acc':>8} {'Test Acc':>9} {'Params':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<30} {r['training_type']:<15} "
              f"{r['val_accuracy']:>8.4f} {r['test_accuracy']:>9.4f} "
              f"{r['parameters']:>12,}")
    print("="*60)
    print(f"\n[main] Experiments completed. Outputs in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
