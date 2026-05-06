"""
models.py – Definición de todos los modelos con extracción de embeddings explícita.

Arquitectura:
    backbone → global avg pooling → embedding → classifier

El teacher (ResNet-50) expone embedding de dim=2048.
Los estudiantes exponen embedding de dim=512 + proyección lineal 512→2048
para alinearse con el espacio del teacher durante la destilación.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


TEACHER_EMBED_DIM = 2048   # ResNet-50 antes del FC


# ─── Teacher: ResNet-50 ───────────────────────────────────────────────────────

class TeacherResNet50(nn.Module):
    """
    ResNet-50 pre-entrenado en ImageNet, ajustado a Tiny ImageNet (200 clases).
    Expone el embedding de 2048-d antes del clasificador.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        try:
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            # Fallback sin pesos preentrenados si no hay acceso a internet
            base = models.resnet50(weights=None)

        # Backbone sin la capa FC original
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = TEACHER_EMBED_DIM
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.pool(feat)
        return feat.flatten(1)   # (B, 2048)

    def forward(self, x: torch.Tensor):
        emb = self.get_embedding(x)
        return self.classifier(emb), emb


# ─── Estudiante: ResNet-18 ────────────────────────────────────────────────────

class StudentResNet18(nn.Module):
    """
    ResNet-18 (mucho más pequeño que ResNet-50).
    Backbone + embedding de 512-d + proyección a 2048 para KD.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 embed_dim: int = config.EMBED_DIM):
        super().__init__()
        base = models.resnet18(weights=None)   # sin preentrenamiento

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim  = embed_dim             # 512
        self.projection = nn.Linear(embed_dim, TEACHER_EMBED_DIM)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.pool(feat)
        return feat.flatten(1)   # (B, 512)

    def get_projected_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding proyectado al espacio del teacher (2048-d)."""
        return self.projection(self.get_embedding(x))

    def forward(self, x: torch.Tensor):
        emb  = self.get_embedding(x)
        proj = self.projection(emb)
        return self.classifier(emb), emb, proj


# ─── Estudiante: MobileNetV3-Small ───────────────────────────────────────────

class StudentMobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small: aún más compacto.
    La última capa de características produce 576-d; proyectamos a embed_dim=512
    y luego a 2048 para alinearse con el teacher.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES,
                 embed_dim: int = config.EMBED_DIM):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)

        # El backbone de MobileNetV3 termina en features
        self.backbone  = base.features          # salida (B, 576, H, W)
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        backbone_out   = 576

        self.embed_layer = nn.Sequential(
            nn.Linear(backbone_out, embed_dim),
            nn.Hardswish(),
        )
        self.embed_dim   = embed_dim
        self.projection  = nn.Linear(embed_dim, TEACHER_EMBED_DIM)
        self.classifier  = nn.Linear(embed_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.embed_layer(feat)   # (B, 512)

    def get_projected_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.get_embedding(x))

    def forward(self, x: torch.Tensor):
        emb  = self.get_embedding(x)
        proj = self.projection(emb)
        return self.classifier(emb), emb, proj


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_teacher() -> TeacherResNet50:
    return TeacherResNet50(num_classes=config.NUM_CLASSES)


def get_student(name: str):
    """Devuelve instancia del modelo estudiante según nombre."""
    registry = {
        "resnet18":           StudentResNet18,
        "mobilenet_v3_small": StudentMobileNetV3Small,
    }
    if name not in registry:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(registry)}")
    return registry[name](num_classes=config.NUM_CLASSES,
                          embed_dim=config.EMBED_DIM)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
