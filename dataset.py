import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LabelMaps:
    stage: Dict[str, int]
    icm: Dict[str, int]
    te: Dict[str, int]


def _find_label_root(dataset_root: str, split: str) -> str:
    if split.lower() == "training":
        candidate = os.path.join(dataset_root, "Training", "T_labeled_data", "microscope")
        if os.path.exists(candidate):
            return candidate
        return os.path.join(dataset_root, "Training", "T_labeled_data")

    candidate = os.path.join(dataset_root, "Validation", "V_labeled_data", "label", "microscope")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(dataset_root, "Validation", "V_labeled_data")


def _find_image_root(dataset_root: str, split: str) -> str:
    if split.lower() == "training":
        candidate = os.path.join(dataset_root, "Training", "T_source_data", "microscope")
        if os.path.exists(candidate):
            return candidate
        return os.path.join(dataset_root, "Training", "T_source_data")

    candidate = os.path.join(dataset_root, "Validation", "V_source_data", "image", "microscope")
    if os.path.exists(candidate):
        return candidate
    return os.path.join(dataset_root, "Validation", "V_source_data")


def _iter_label_files(label_root: str) -> List[str]:
    label_paths: List[str] = []
    for root, _, files in os.walk(label_root):
        for name in files:
            if name.endswith(".json"):
                label_paths.append(os.path.join(root, name))
    return label_paths


def _parse_labels(label_path: str) -> Optional[Tuple[str, str, str]]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    properties = data.get("categories", {}).get("properties", [])
    if not properties:
        return None

    props = properties[0]
    stage = props.get("stage")
    icm = props.get("ICM")
    te = props.get("TE")
    if stage is None or icm is None or te is None:
        return None
    return stage, icm, te


def build_label_maps(label_root: str) -> LabelMaps:
    stage_set, icm_set, te_set = set(), set(), set()
    for label_path in _iter_label_files(label_root):
        parsed = _parse_labels(label_path)
        if parsed is None:
            continue
        stage, icm, te = parsed
        stage_set.add(stage)
        icm_set.add(icm)
        te_set.add(te)

    stage_map = {label: idx for idx, label in enumerate(sorted(stage_set))}
    icm_map = {label: idx for idx, label in enumerate(sorted(icm_set))}
    te_map = {label: idx for idx, label in enumerate(sorted(te_set))}
    return LabelMaps(stage=stage_map, icm=icm_map, te=te_map)


def _build_image_index(image_root: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(image_root):
        for name in files:
            if name.endswith(".png"):
                base = os.path.splitext(name)[0]
                index[base] = os.path.join(root, name)
    return index


def build_transforms(
    is_train: bool, aug_strength: float = 0.2, aug_advanced: bool = False
) -> transforms.Compose:
    if is_train:
        aug_list = []
        if aug_advanced:
            aug_list.extend(
                [
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)
                    ),
                    transforms.RandomAdjustSharpness(1.5, p=0.3),
                ]
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                *aug_list,
                transforms.ColorJitter(
                    brightness=aug_strength,
                    contrast=aug_strength,
                    saturation=aug_strength,
                    hue=min(0.1, aug_strength),
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class EmbryoMultiTaskDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        label_maps: LabelMaps,
        transform: Optional[transforms.Compose] = None,
        is_train: Optional[bool] = None,
        aug_strength: float = 0.2,
        aug_advanced: bool = False,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.label_root = _find_label_root(dataset_root, split)
        self.image_root = _find_image_root(dataset_root, split)
        self.label_maps = label_maps
        if is_train is None:
            is_train = split.lower() == "training"
        self.transform = transform or build_transforms(
            is_train=is_train,
            aug_strength=aug_strength,
            aug_advanced=aug_advanced,
        )

        self.samples = self._build_samples()

    def _build_samples(self) -> List[Tuple[str, int, int, int]]:
        label_paths = _iter_label_files(self.label_root)
        image_index: Optional[Dict[str, str]] = None
        samples: List[Tuple[str, int, int, int]] = []

        for label_path in label_paths:
            parsed = _parse_labels(label_path)
            if parsed is None:
                continue

            stage, icm, te = parsed
            if (
                stage not in self.label_maps.stage
                or icm not in self.label_maps.icm
                or te not in self.label_maps.te
            ):
                continue

            rel_path = os.path.relpath(label_path, self.label_root)
            rel_base = os.path.splitext(rel_path)[0] + ".png"
            image_path = os.path.join(self.image_root, rel_base)

            if not os.path.exists(image_path):
                if image_index is None:
                    image_index = _build_image_index(self.image_root)
                base = os.path.splitext(os.path.basename(label_path))[0]
                image_path = image_index.get(base)

            if image_path is None or not os.path.exists(image_path):
                continue

            samples.append(
                (
                    image_path,
                    self.label_maps.stage[stage],
                    self.label_maps.icm[icm],
                    self.label_maps.te[te],
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, stage, icm, te = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, stage, icm, te
