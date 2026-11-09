"""Model zoo: every architecture is built through timm with identical head handling,
so the ONLY thing that changes between runs in Experiment A is the architecture."""
import timm

MODEL_ZOO = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "mobilenetv3_small": "mobilenetv3_small_100",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "vit_tiny": "vit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
}


def build_model(name: str, num_classes: int, pretrained: bool = True):
    if name not in MODEL_ZOO:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_ZOO)}")
    return timm.create_model(MODEL_ZOO[name], pretrained=pretrained, num_classes=num_classes)
