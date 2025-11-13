
"""
Reusable utilities for Oxford Flowers 102 experiments (Assignment Section F: Flowers Recognition).
Consolidates common components used by baseline, tuning, and ensemble pipelines.

Contents
1) Determinism and Device/AMP
2) Data: transforms and deterministic DataLoader builders
3) Model: ResNet50 backbone with a cosine classifier head
4) Checkpoint: safe backbone load and optional FC-to-cosine transplant
5) Training/Evaluation: mixed precision, early stopping, metrics helpers
6) Inference: single-image top-k prediction
7) Logging and few-shot helpers
8) Ensemble: probability averaging across models

Notes
- This module performs no I/O or training at import time.
"""

import os
import platform
import random
import math
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Flowers102
from torchvision.models import resnet50
try:
    from torchvision.models import ResNet50_Weights
    _RESNET_WTS = ResNet50_Weights.IMAGENET1K_V2
except Exception:
    _RESNET_WTS = "IMAGENET1K_V2"

# =============================================================================
# 1. Determinism and device/AMP
# =============================================================================

def seed_all(seed: int = 1029, strict_determinism: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs. Optionally enforce deterministic algorithms.
    Call before creating CUDA tensors when strict_determinism is True.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # ":16:8" is also valid
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        torch.use_deterministic_algorithms(True, warn_only=False)


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    amp_enabled: bool
    non_blocking_io: bool

def get_device_config() -> DeviceConfig:
    """
    Select compute device and AMP policy. Use MPS on macOS when available, otherwise CUDA, otherwise CPU.
    Enable AMP on CUDA only. Use non_blocking host-to-device copies on CUDA only.
    """
    is_mac = (platform.system() == "Darwin")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if is_mac and mps_ok:
        return DeviceConfig(torch.device("mps"), amp_enabled=False, non_blocking_io=False)
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        return DeviceConfig(torch.device("cuda:0"), amp_enabled=True, non_blocking_io=True)
    return DeviceConfig(torch.device("cpu"), amp_enabled=False, non_blocking_io=False)


# =============================================================================
# 2. Data
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _seed_worker(worker_id: int) -> None:
    """
    Initialize worker RNGs for deterministic DataLoader behavior.
    """
    worker_seed = (torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_dataloaders(
    root: str = "data",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment: bool = True,
    seed: int = 1029,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create deterministic train/val/test DataLoaders for Flowers102.
    Use a fixed torch.Generator on CPU to make shuffle order reproducible.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = Flowers102(root=root, split="train", download=True, transform=train_tf)
    val_set   = Flowers102(root=root, split="val",   download=True, transform=eval_tf)
    test_set  = Flowers102(root=root, split="test",  download=True, transform=eval_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, persistent_workers=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, persistent_workers=False,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# 3. Model
# =============================================================================

class CosineClassifier(nn.Module):
    """
    Cosine classifier over L2-normalized features and weights.
    Apply a scale factor s to sharpen softmax responses.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0) -> None:
        super().__init__()
        self.s = float(s)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x)
        w = F.normalize(self.weight)
        return self.s * (x @ w.t())

def build_resnet50_cosine(num_classes: int = 102, pretrained: bool = True, device: Optional[torch.device] = None) -> nn.Module:
    """
    Build a ResNet-50 backbone with identity final FC and a cosine classifier head.
    """
    base = resnet50(weights=_RESNET_WTS if pretrained else None)
    base.fc = nn.Identity()
    model = nn.Sequential(base, CosineClassifier(2048, num_classes))
    if device is None:
        device = get_device_config().device
    return model.to(device)


# =============================================================================
# 4. Checkpoint loading
# =============================================================================

def safe_load_backbone_and_transplant_head(
    model: nn.Module,
    ckpt_path: str,
    device: Optional[torch.device] = None,
    cosine_scale: float = 30.0,
) -> None:
    """
    Load a checkpoint into model[0] (backbone) with strict=False.
    If an FC weight exists and matches the cosine head shape, copy its normalized values
    into model[1].weight and set the cosine scale.
    """
    if device is None:
        device = get_device_config().device
    if not (ckpt_path and os.path.exists(ckpt_path)):
        print(f"[INFO] No checkpoint loaded (missing path: {ckpt_path})")
        return

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Load backbone non-strictly
    model[0].load_state_dict(state, strict=False)
    print(f"[OK] Loaded backbone from: {ckpt_path}")

    # Try transplant FC->cosine head
    def first_present(d: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
        for k in keys:
            if k in d:
                return k
        return None

    w_key = first_present(state, ["fc.1.weight", "fc.weight"])
    if w_key is not None:
        W = state[w_key]
        if hasattr(model[1], "weight") and W.shape == model[1].weight.shape:
            with torch.no_grad():
                model[1].weight.copy_(F.normalize(W, dim=1))
                if hasattr(model[1], "s"):
                    model[1].s = float(cosine_scale)
            print(f"[OK] Transplanted {w_key} → cosine (normalized, s={cosine_scale}).")
        else:
            print(f"[INFO] Skip transplant due to shape mismatch: {tuple(W.shape)} vs {tuple(getattr(model[1],'weight').shape)}")
    else:
        print("[INFO] No FC weights found to transplant.")


# =============================================================================
# 5. Training and evaluation
# =============================================================================

@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-4
    patience: int = 5
    weight_decay: float = 0.0
    label_smoothing: float = 0.0
    ckpt_path: str = "ckpt/best.pth"

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig = TrainConfig(),
    device_cfg: Optional[DeviceConfig] = None,
    epoch_log_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    trial_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train with mixed precision on CUDA and FP32 elsewhere.
    Apply early stopping on validation loss; break ties by higher validation accuracy.
    """
    from torch.cuda.amp import autocast, GradScaler
    device_cfg = device_cfg or get_device_config()
    device = device_cfg.device
    model = model.to(device)
    os.makedirs(os.path.dirname(cfg.ckpt_path), exist_ok=True)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(enabled=device_cfg.amp_enabled)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=device_cfg.non_blocking_io)
            labels = labels.to(device, non_blocking=device_cfg.non_blocking_io)

            optimizer.zero_grad(set_to_none=True)
            if device_cfg.amp_enabled:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=device_cfg.non_blocking_io)
                labels = labels.to(device, non_blocking=device_cfg.non_blocking_io)
                if device_cfg.amp_enabled:
                    with autocast():
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d} | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

        improved = (val_loss < best_val_loss) or (math.isclose(val_loss, best_val_loss, rel_tol=0.0, abs_tol=1e-12) and val_acc > best_val_acc)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), cfg.ckpt_path)
            print(f"Saved best model → {cfg.ckpt_path}")
        else:
            patience_counter += 1
            print(f" EarlyStopping counter {patience_counter}/{cfg.patience}")
            if patience_counter >= cfg.patience:
                print(" Early stopping triggered.")
                break

        if epoch_log_cb is not None:
            epoch_log_cb({
                "trial_id": trial_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            })

    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "ckpt_path": cfg.ckpt_path,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    amp_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Compute accuracy and macro/weighted aggregates from the classification report.
    """
    from sklearn.metrics import classification_report, accuracy_score
    from torch.cuda.amp import autocast

    if device is None or amp_enabled is None:
        dc = get_device_config()
        device = dc.device if device is None else device
        amp_enabled = dc.amp_enabled if amp_enabled is None else amp_enabled

    model = model.to(device).eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for imgs, labels in data_loader:
        imgs = imgs.to(device)
        if amp_enabled:
            with autocast():
                logits = model(imgs)
        else:
            logits = model(imgs)
        preds = logits.argmax(1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = float(accuracy_score(all_labels, all_preds))
    rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "macro_avg": {
            "precision": float(rep.get("macro avg", {}).get("precision", 0.0)),
            "recall": float(rep.get("macro avg", {}).get("recall", 0.0)),
            "f1": float(rep.get("macro avg", {}).get("f1-score", 0.0)),
        },
        "weighted_avg": {
            "precision": float(rep.get("weighted avg", {}).get("precision", 0.0)),
            "recall": float(rep.get("weighted avg", {}).get("recall", 0.0)),
            "f1": float(rep.get("weighted avg", {}).get("f1-score", 0.0)),
        },
    }


# =============================================================================
# 6. Inference
# =============================================================================

def predict_image(
    model: nn.Module,
    img_path: str,
    device: Optional[torch.device] = None,
    topk: int = 5,
    resize: int = 256,
    center_crop: int = 224,
) -> List[Tuple[int, float]]:
    """
    Predict top-k classes for a single RGB image using ImageNet normalization.
    """
    from PIL import Image

    tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)

    if device is None:
        device = get_device_config().device
    model.eval().to(device)
    with torch.no_grad():
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)[0]
        vals, inds = probs.topk(topk)
    return [(int(i), float(v)) for v, i in zip(inds, vals)]
