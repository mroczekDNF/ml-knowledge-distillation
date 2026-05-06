"""
config.py – Configuración central del experimento de Knowledge Distillation.
Todos los hiperparámetros están aquí para garantizar comparaciones justas.
"""

import os

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")

for d in [DATA_DIR, CACHE_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_URL     = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_ZIP     = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
DATASET_ROOT    = os.path.join(DATA_DIR, "tiny-imagenet-200")
NUM_CLASSES     = 200
IMAGE_SIZE      = 64   # Tiny ImageNet native size

# ─── Reproducibilidad ─────────────────────────────────────────────────────────
SEED = 42

# ─── Entrenamiento (idéntico para TODOS los experimentos) ─────────────────────
#BATCH_SIZE      = 128
BATCH_SIZE      = 256
NUM_EPOCHS      = 2
#NUM_EPOCHS      = 150
LEARNING_RATE   = 1e-3
OPTIMIZER       = "adam"   # "adam" | "sgd"
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 4

# ─── Fine-tuning del teacher ──────────────────────────────────────────────────
#TEACHER_EPOCHS  = 20
TEACHER_EPOCHS  = 5
TEACHER_LR      = 1e-4    # LR más bajo para fine-tuning
TEACHER_CKPT    = os.path.join(CHECKPOINT_DIR, "teacher_resnet50.pth")

# ─── Knowledge Distillation ───────────────────────────────────────────────────
ALPHA           = 0.5    # peso de L_cls  (cross-entropy)
BETA            = 0.5    # peso de L_distill (MSE embeddings)
DISTILL_LOSS    = "mse"  # "mse" | "cosine"
EMBED_DIM       = 512    # dimensión del embedding de los estudiantes

# ─── Caché de embeddings del teacher ─────────────────────────────────────────
TEACHER_EMBED_CACHE = os.path.join(CACHE_DIR, "teacher_embeddings.pt")

# ─── Modelos estudiante disponibles ───────────────────────────────────────────
STUDENT_MODELS  = ["resnet18", "mobilenet_v3_small"]

# ─── Visualización embeddings ─────────────────────────────────────────────────
VIZ_SAMPLES     = 1000   # muestras para PCA / t-SNE
VIZ_CLASSES     = 10     # primeras N clases para colorear
