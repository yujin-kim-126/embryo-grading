from typing import Tuple

import torch
from torch import nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
except Exception:  # pragma: no cover - compatibility fallback
    from torchvision.models import resnet50
    ResNet50_Weights = None


class MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskAdapter(nn.Module):
    def __init__(self, dim: int, adapter_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(adapter_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbryoMultiTaskModel(nn.Module):
    def __init__(
        self,
        num_stage_classes: int,
        num_icm_classes: int,
        num_te_classes: int,
        pretrained: bool = True,
        dropout: float = 0.5,
        shared_dim: int = 1024,
        head_hidden_dim: int = 512,
        adapter_dim: int = 256,
        use_task_adapters: bool = True,
        use_uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = self._build_backbone(pretrained)
        self.shared_mlp = nn.Sequential(
            nn.Linear(2048, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.use_task_adapters = use_task_adapters
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_task_adapters:
            self.stage_adapter = TaskAdapter(shared_dim, adapter_dim, dropout=dropout)
            self.icm_adapter = TaskAdapter(shared_dim, adapter_dim, dropout=dropout)
            self.te_adapter = TaskAdapter(shared_dim, adapter_dim, dropout=dropout)
        self.stage_head = MLPHead(
            shared_dim, head_hidden_dim, num_stage_classes, dropout=dropout
        )
        self.icm_head = MLPHead(
            shared_dim, head_hidden_dim, num_icm_classes, dropout=dropout
        )
        self.te_head = MLPHead(
            shared_dim, head_hidden_dim, num_te_classes, dropout=dropout
        )
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))

    def _build_backbone(self, pretrained: bool) -> nn.Module:
        if ResNet50_Weights is None:
            backbone = resnet50(pretrained=pretrained)
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()
        return backbone

    def freeze_backbone(self, freeze: bool = True) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        shared = self.shared_mlp(features)
        if self.use_task_adapters:
            stage_feat = self.stage_adapter(shared)
            icm_feat = self.icm_adapter(shared)
            te_feat = self.te_adapter(shared)
        else:
            stage_feat = icm_feat = te_feat = shared
        stage_logits = self.stage_head(stage_feat)
        icm_logits = self.icm_head(icm_feat)
        te_logits = self.te_head(te_feat)
        return stage_logits, icm_logits, te_logits
