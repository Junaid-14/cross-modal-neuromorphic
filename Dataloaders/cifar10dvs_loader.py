"""
CIFAR10-DVS Dataset Loader
DVS event-based version of CIFAR-10 (10 classes, 128×128).
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn.functional as F
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10dvs_loaders(
    batch_size: int = 32,
    time_steps: int = 25,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    save_to: str = "./ICONS_M7/datasets",
    time_window: int = 1000,
    target_spatial_size: Tuple[int, int] | None = (34, 34),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR10-DVS train, validation, and test data loaders.

    CIFAR10-DVS does not provide a built-in train/val/test split, so one is
    created deterministically via ``random_split``.

    Args:
        batch_size: Batch size for training.
        time_steps: Number of time steps for temporal encoding.
        num_workers: Number of worker processes for data loading.
        pin_memory: If True, use pinned memory for faster GPU transfer.
        persistent_workers: If True, workers stay alive between epochs.
        save_to: Directory to download/store the dataset.
        time_window: Window size (in microseconds) for event framing.
        target_spatial_size: If provided, resize frames to this spatial size
            via adaptive average pooling (default: 34×34 to match the
            VisualBackbone default).
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
        seed: Random seed for the split.

    Returns:
        train_loader, val_loader, test_loader
    """
    if num_workers is None:
        num_workers = min(8, os.cpu_count())
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if num_workers == 0 or num_workers == 1:
        persistent_workers = False

    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size  # (128, 128, 2)
    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        ]
    )

    full_dataset = tonic.datasets.CIFAR10DVS(
        save_to=save_to, transform=frame_transform
    )

    n_total = len(full_dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    def collate_fn(batch):
        frames, labels = zip(*batch)
        frames_list = []
        for f in frames:
            f_tensor = torch.as_tensor(f, dtype=torch.float32)
            # Optional spatial resize to match backbone input
            if target_spatial_size is not None and f_tensor.shape[-2:] != target_spatial_size:
                f_tensor = F.adaptive_avg_pool2d(
                    f_tensor.view(-1, *f_tensor.shape[-3:]),
                    target_spatial_size,
                ).view(f_tensor.shape[0], *f_tensor.shape[1:-2], *target_spatial_size)
            if f_tensor.shape[0] >= time_steps:
                frames_list.append(f_tensor[:time_steps])
            else:
                padding = torch.zeros(
                    time_steps - f_tensor.shape[0],
                    *f_tensor.shape[1:],
                    dtype=torch.float32,
                )
                frames_list.append(torch.cat([f_tensor, padding], dim=0))
        frames_tensor = torch.stack(frames_list, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return frames_tensor, labels_tensor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(
        f"✓ CIFAR10-DVS loaded: {n_train} train, {n_val} val, {n_test} test"
    )
    return train_loader, val_loader, test_loader
