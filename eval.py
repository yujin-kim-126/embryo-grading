"""저장된 체크포인트로 Validation 세트 평가 후 최종 결과 출력."""
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import EmbryoMultiTaskDataset, LabelMaps
from model import EmbryoMultiTaskModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "checkpoints", "best_finetune.pt"),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "embyo_dataset"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="eval_result.txt 저장 경로 (미지정 시 프로젝트 루트)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"]
    shared_dim = state["shared_mlp.0.weight"].shape[0]
    head_hidden_dim = state["stage_head.net.0.weight"].shape[0]
    use_task_adapters = any(k.startswith("stage_adapter") for k in state.keys())
    use_uncertainty_weighting = "log_vars" in state
    adapter_dim = 256
    if use_task_adapters:
        adapter_dim = state["stage_adapter.net.0.weight"].shape[0]
    label_maps = LabelMaps(
        stage=ckpt["label_maps"]["stage"],
        icm=ckpt["label_maps"]["icm"],
        te=ckpt["label_maps"]["te"],
    )

    model = EmbryoMultiTaskModel(
        num_stage_classes=len(label_maps.stage),
        num_icm_classes=len(label_maps.icm),
        num_te_classes=len(label_maps.te),
        pretrained=False,
        dropout=0.5,
        shared_dim=shared_dim,
        head_hidden_dim=head_hidden_dim,
        adapter_dim=adapter_dim,
        use_task_adapters=use_task_adapters,
        use_uncertainty_weighting=use_uncertainty_weighting,
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    val_dataset = EmbryoMultiTaskDataset(
        dataset_root=args.dataset_root,
        split="Validation",
        label_maps=label_maps,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterions = {
        "stage": nn.CrossEntropyLoss(),
        "icm": nn.CrossEntropyLoss(),
        "te": nn.CrossEntropyLoss(),
    }
    loss_weights = (1.0, 1.0, 1.0)

    total_loss = 0.0
    correct_stage = 0
    correct_icm = 0
    correct_te = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for images, stage, icm, te in val_loader:
            b = images.size(0)
            images = images.to(device)
            stage = stage.to(device)
            icm = icm.to(device)
            te = te.to(device)
            stage_logits, icm_logits, te_logits = model(images)
            loss_stage = criterions["stage"](stage_logits, stage)
            loss_icm = criterions["icm"](icm_logits, icm)
            loss_te = criterions["te"](te_logits, te)
            loss = loss_weights[0] * loss_stage + loss_weights[1] * loss_icm + loss_weights[2] * loss_te
            total_loss += loss.item()
            correct_stage += (stage_logits.argmax(1) == stage).sum().item()
            correct_icm += (icm_logits.argmax(1) == icm).sum().item()
            correct_te += (te_logits.argmax(1) == te).sum().item()
            total += b
            n_batches += 1

    steps = max(1, n_batches)
    n_samples = max(1, total)
    loss_val = total_loss / steps
    stage_acc = correct_stage / n_samples
    icm_acc = correct_icm / n_samples
    te_acc = correct_te / n_samples

    lines = [
        "=== 최종 검증 결과 (Validation) ===",
        f"  Loss:        {loss_val:.4f}",
        f"  Stage Acc:   {stage_acc:.4f} ({100 * stage_acc:.2f}%)",
        f"  ICM Acc:     {icm_acc:.4f} ({100 * icm_acc:.2f}%)",
        f"  TE Acc:      {te_acc:.4f} ({100 * te_acc:.2f}%)",
        f"  샘플 수:     {len(val_dataset)}",
    ]
    out = "\n".join(lines)
    print(out)

    result_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(result_dir, "eval_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"\n결과 저장: {result_path}")


if __name__ == "__main__":
    main()
