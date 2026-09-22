import argparse
import os
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import EmbryoMultiTaskDataset, LabelMaps, build_label_maps
from model import EmbryoMultiTaskModel


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt).pow(self.gamma) * log_pt
        if self.weight is not None:
            alpha = self.weight.gather(0, targets)
            loss = loss * alpha
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterions: Dict[str, nn.Module],
    loss_weights: Tuple[float, float, float],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_uncertainty_weighting: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_stage_acc = 0.0
    total_icm_acc = 0.0
    total_te_acc = 0.0

    for images, stage, icm, te in loader:
        images = images.to(device)
        stage = stage.to(device)
        icm = icm.to(device)
        te = te.to(device)

        optimizer.zero_grad(set_to_none=True)
        stage_logits, icm_logits, te_logits = model(images)

        loss_stage = criterions["stage"](stage_logits, stage)
        loss_icm = criterions["icm"](icm_logits, icm)
        loss_te = criterions["te"](te_logits, te)

        if use_uncertainty_weighting and hasattr(model, "log_vars"):
            log_vars = torch.clamp(model.log_vars, min=-2.0)
            precision_stage = torch.exp(-log_vars[0])
            precision_icm = torch.exp(-log_vars[1])
            precision_te = torch.exp(-log_vars[2])
            loss = (
                precision_stage * loss_stage + log_vars[0]
                + precision_icm * loss_icm + log_vars[1]
                + precision_te * loss_te + log_vars[2]
            )
        else:
            loss = (
                loss_weights[0] * loss_stage
                + loss_weights[1] * loss_icm
                + loss_weights[2] * loss_te
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_stage_acc += _accuracy(stage_logits, stage)
        total_icm_acc += _accuracy(icm_logits, icm)
        total_te_acc += _accuracy(te_logits, te)

    steps = max(1, len(loader))
    return {
        "loss": total_loss / steps,
        "stage_acc": total_stage_acc / steps,
        "icm_acc": total_icm_acc / steps,
        "te_acc": total_te_acc / steps,
    }


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterions: Dict[str, nn.Module],
    loss_weights: Tuple[float, float, float],
    device: torch.device,
    use_uncertainty_weighting: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_stage_acc = 0.0
    total_icm_acc = 0.0
    total_te_acc = 0.0

    for images, stage, icm, te in loader:
        images = images.to(device)
        stage = stage.to(device)
        icm = icm.to(device)
        te = te.to(device)

        stage_logits, icm_logits, te_logits = model(images)
        loss_stage = criterions["stage"](stage_logits, stage)
        loss_icm = criterions["icm"](icm_logits, icm)
        loss_te = criterions["te"](te_logits, te)

        if use_uncertainty_weighting and hasattr(model, "log_vars"):
            log_vars = torch.clamp(model.log_vars, min=-2.0)
            precision_stage = torch.exp(-log_vars[0])
            precision_icm = torch.exp(-log_vars[1])
            precision_te = torch.exp(-log_vars[2])
            loss = (
                precision_stage * loss_stage + log_vars[0]
                + precision_icm * loss_icm + log_vars[1]
                + precision_te * loss_te + log_vars[2]
            )
        else:
            loss = (
                loss_weights[0] * loss_stage
                + loss_weights[1] * loss_icm
                + loss_weights[2] * loss_te
            )

        total_loss += loss.item()
        total_stage_acc += _accuracy(stage_logits, stage)
        total_icm_acc += _accuracy(icm_logits, icm)
        total_te_acc += _accuracy(te_logits, te)

    steps = max(1, len(loader))
    return {
        "loss": total_loss / steps,
        "stage_acc": total_stage_acc / steps,
        "icm_acc": total_icm_acc / steps,
        "te_acc": total_te_acc / steps,
    }


def _compute_class_weights(
    dataset: EmbryoMultiTaskDataset, num_classes: int, label_index: int
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, stage, icm, te in dataset.samples:
        label = (stage, icm, te)[label_index]
        counts[label] += 1.0
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean().clamp(min=1e-6)
    return weights


def _build_sample_weights(
    dataset: EmbryoMultiTaskDataset,
    icm_weights: torch.Tensor,
    te_weights: torch.Tensor,
) -> torch.Tensor:
    weights = []
    for _, _, icm, te in dataset.samples:
        weights.append(0.5 * icm_weights[icm] + 0.5 * te_weights[te])
    return torch.tensor(weights, dtype=torch.float32)


def _build_label_maps(dataset_root: str) -> LabelMaps:
    label_root = os.path.join(dataset_root, "Training", "T_labeled_data", "microscope")
    if not os.path.exists(label_root):
        label_root = os.path.join(dataset_root, "Training", "T_labeled_data")
    return build_label_maps(label_root)


def _build_dataloaders(
    dataset_root: str,
    label_maps: LabelMaps,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
    aug_strength: float,
    aug_advanced: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = EmbryoMultiTaskDataset(
        dataset_root=dataset_root,
        split="Training",
        label_maps=label_maps,
        is_train=True,
        aug_strength=aug_strength,
        aug_advanced=aug_advanced,
    )
    val_dataset = EmbryoMultiTaskDataset(
        dataset_root=dataset_root,
        split="Validation",
        label_maps=label_maps,
        is_train=False,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty after filtering labels.")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after filtering labels.")

    if use_weighted_sampler:
        icm_weights = _compute_class_weights(
            train_dataset, len(label_maps.icm), 1
        )
        te_weights = _compute_class_weights(
            train_dataset, len(label_maps.te), 2
        )
        sample_weights = _build_sample_weights(
            train_dataset, icm_weights, te_weights
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _train_phase(
    model: EmbryoMultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    criterions: Dict[str, nn.Module],
    loss_weights: Tuple[float, float, float],
    device: torch.device,
    phase_name: str,
    save_dir: str,
    lr_warmup: int = 0,
    warmup_target: Tuple[float, float] = (0.0, 0.0),
    use_uncertainty_weighting: bool = False,
) -> None:
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        if lr_warmup > 0 and epoch <= lr_warmup:
            ratio = epoch / lr_warmup
            for group in optimizer.param_groups:
                if group.get("name") == "backbone":
                    group["lr"] = warmup_target[0] * ratio
                elif group.get("name") == "heads":
                    group["lr"] = warmup_target[1] * ratio
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            criterions,
            loss_weights,
            optimizer,
            device,
            use_uncertainty_weighting,
        )
        val_metrics = _validate(
            model,
            val_loader,
            criterions,
            loss_weights,
            device,
            use_uncertainty_weighting,
        )
        scheduler.step()

        print(
            f"[{phase_name}] Epoch {epoch}/{epochs} "
            f"Train Loss {train_metrics['loss']:.4f} "
            f"Val Loss {val_metrics['loss']:.4f} "
            f"Val Acc (Stage/ICM/TE) "
            f"{val_metrics['stage_acc']:.3f}/"
            f"{val_metrics['icm_acc']:.3f}/"
            f"{val_metrics['te_acc']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_maps": {
                        "stage": train_loader.dataset.label_maps.stage,
                        "icm": train_loader.dataset.label_maps.icm,
                        "te": train_loader.dataset.label_maps.te,
                    },
                },
                os.path.join(save_dir, f"best_{phase_name}.pt"),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "embyo_dataset"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--shared-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--use-task-adapters", action="store_true")
    parser.add_argument("--use-uncertainty-weighting", action="store_true")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--aug-strength", type=float, default=0.2)
    parser.add_argument("--aug-advanced", action="store_true")
    parser.add_argument("--stage-weight", type=float, default=1.0)
    parser.add_argument("--icm-weight", type=float, default=1.0)
    parser.add_argument("--te-weight", type=float, default=1.0)
    parser.add_argument("--use-stage-weights", action="store_true")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--icm-loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--te-loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--finetune-warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-backbone-start-lr", type=float, default=1e-6)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_maps = _build_label_maps(args.dataset_root)
    train_loader, val_loader = _build_dataloaders(
        dataset_root=args.dataset_root,
        label_maps=label_maps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
        aug_strength=args.aug_strength,
        aug_advanced=args.aug_advanced,
    )

    model = EmbryoMultiTaskModel(
        num_stage_classes=len(label_maps.stage),
        num_icm_classes=len(label_maps.icm),
        num_te_classes=len(label_maps.te),
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        shared_dim=args.shared_dim,
        head_hidden_dim=args.head_hidden_dim,
        adapter_dim=args.adapter_dim,
        use_task_adapters=args.use_task_adapters,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
    ).to(device)

    stage_weight_tensor = None
    icm_weight_tensor = None
    te_weight_tensor = None
    if args.use_stage_weights or args.use_class_weights:
        stage_weight_tensor = _compute_class_weights(
            train_loader.dataset, len(label_maps.stage), 0
        )
    if args.use_class_weights:
        icm_weight_tensor = _compute_class_weights(
            train_loader.dataset, len(label_maps.icm), 1
        )
        te_weight_tensor = _compute_class_weights(
            train_loader.dataset, len(label_maps.te), 2
        )

    stage_weight = stage_weight_tensor.to(device) if stage_weight_tensor is not None else None
    icm_weight = icm_weight_tensor.to(device) if icm_weight_tensor is not None else None
    te_weight = te_weight_tensor.to(device) if te_weight_tensor is not None else None

    stage_criterion = nn.CrossEntropyLoss(
        weight=stage_weight,
        label_smoothing=args.label_smoothing,
    )
    if args.icm_loss == "focal":
        icm_criterion = FocalLoss(gamma=args.focal_gamma, weight=icm_weight)
    else:
        icm_criterion = nn.CrossEntropyLoss(
            weight=icm_weight,
            label_smoothing=args.label_smoothing,
        )
    if args.te_loss == "focal":
        te_criterion = FocalLoss(gamma=args.focal_gamma, weight=te_weight)
    else:
        te_criterion = nn.CrossEntropyLoss(
            weight=te_weight,
            label_smoothing=args.label_smoothing,
        )

    criterions = {
        "stage": stage_criterion,
        "icm": icm_criterion,
        "te": te_criterion,
    }
    loss_weights = (args.stage_weight, args.icm_weight, args.te_weight)

    if args.freeze_epochs > 0:
        model.freeze_backbone(True)
        head_params = (
            list(model.stage_head.parameters())
            + list(model.icm_head.parameters())
            + list(model.te_head.parameters())
        )
        if getattr(model, "use_task_adapters", False):
            head_params += (
                list(model.stage_adapter.parameters())
                + list(model.icm_adapter.parameters())
                + list(model.te_adapter.parameters())
            )
        if args.use_uncertainty_weighting and hasattr(model, "log_vars"):
            head_params += [model.log_vars]
        optimizer = Adam(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        _train_phase(
            model,
            train_loader,
            val_loader,
            args.freeze_epochs,
            optimizer,
            scheduler,
            criterions,
            loss_weights,
            device,
            "freeze",
            args.save_dir,
            use_uncertainty_weighting=args.use_uncertainty_weighting,
        )

    fine_tune_epochs = max(0, args.epochs - args.freeze_epochs)
    if fine_tune_epochs > 0:
        model.freeze_backbone(False)
        optimizer = Adam(
            [
                {
                    "params": model.backbone.parameters(),
                    "lr": args.finetune_backbone_start_lr,
                    "name": "backbone",
                },
                {
                    "params": (
                        list(model.stage_head.parameters())
                        + list(model.icm_head.parameters())
                        + list(model.te_head.parameters())
                    ),
                    "lr": args.head_lr,
                    "name": "heads",
                },
            ],
            weight_decay=args.weight_decay,
        )
        if getattr(model, "use_task_adapters", False):
            optimizer.add_param_group(
                {
                    "params": (
                        list(model.stage_adapter.parameters())
                        + list(model.icm_adapter.parameters())
                        + list(model.te_adapter.parameters())
                    ),
                    "lr": args.head_lr,
                    "name": "adapters",
                }
            )
        if args.use_uncertainty_weighting and hasattr(model, "log_vars"):
            optimizer.add_param_group(
                {"params": [model.log_vars], "lr": args.head_lr, "name": "loss"}
            )
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        _train_phase(
            model,
            train_loader,
            val_loader,
            fine_tune_epochs,
            optimizer,
            scheduler,
            criterions,
            loss_weights,
            device,
            "finetune",
            args.save_dir,
            lr_warmup=args.finetune_warmup_epochs,
            warmup_target=(args.backbone_lr, args.head_lr),
            use_uncertainty_weighting=args.use_uncertainty_weighting,
        )


if __name__ == "__main__":
    main()
