"""
N-TIDIGITS18 Dataset Loader
Spiking spoken digits (single-digit utterances, 10 classes, 64 channels).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import tonic
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DenseNTIDIGITS18Dataset(Dataset):
    """Pre-converted N-TIDIGITS18 dataset with dense spike tensors."""

    def __init__(self, data: List[Tuple[torch.Tensor, int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        spikes, label = self.data[idx]
        return spikes, torch.tensor(label, dtype=torch.long)


def _events_to_dense(events, channels: int = 64, time_bins: int = 100) -> torch.Tensor:
    dense = torch.zeros(time_bins, channels)

    if len(events) > 0:
        max_time = events["t"].max() if len(events) > 0 else 1
        time_indices = (events["t"] / max_time * (time_bins - 1)).astype(int)
        channel_indices = events["x"].astype(int)

        for t, c in zip(time_indices, channel_indices):
            if 0 <= t < time_bins and 0 <= c < channels:
                dense[t, c] = 1.0

    return dense.unsqueeze(1).unsqueeze(1)


def get_ntidigits18_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
    save_to: str = "./ICONS_M7/datasets",
    time_bins: int = 100,
    single_digits: bool = True,
    use_cache: bool = True,
    cache_path: Path | None = None,
    force_recreate: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get N-TIDIGITS18 train and test data loaders with optional disk caching.

    Args:
        batch_size: Batch size for training.
        num_workers: Number of worker processes for data loading.
        save_to: Directory to download/store dataset.
        time_bins: Number of time bins for temporal encoding.
        single_digits: If True, use only single-digit utterances (10 classes).
        use_cache: If True, use cached converted data if available.
        cache_path: Path to cache file (default: ./data/ntidigits18_cache.pt).
        force_recreate: If True, ignore existing cache and recreate.

    Returns:
        train_loader, test_loader
    """
    if cache_path is None:
        cache_path = Path(save_to) / "ntidigits18_cache.pt"

    if use_cache and not force_recreate and cache_path.exists():
        print(f"Loading cached N-TIDIGITS18 data from {cache_path}")
        train_data, test_data = torch.load(cache_path)
    else:
        train_dataset = tonic.datasets.NTIDIGITS18(
            save_to=save_to, train=True, single_digits=single_digits
        )
        test_dataset = tonic.datasets.NTIDIGITS18(
            save_to=save_to, train=False, single_digits=single_digits
        )

        channels = train_dataset.sensor_size[0]  # (64, 1, 1)

        train_data = []
        for events, label in tqdm(train_dataset, desc="N-TIDIGITS18 train -> dense"):
            train_data.append((_events_to_dense(events, channels, time_bins), label))

        test_data = []
        for events, label in tqdm(test_dataset, desc="N-TIDIGITS18 test -> dense"):
            test_data.append((_events_to_dense(events, channels, time_bins), label))

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((train_data, test_data), cache_path)
            print(f"Saved N-TIDIGITS18 cache to {cache_path}")

    train_loader = DataLoader(
        DenseNTIDIGITS18Dataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        DenseNTIDIGITS18Dataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    print(
        f"✓ N-TIDIGITS18 loaded: {len(train_data)} train, {len(test_data)} test"
    )
    return train_loader, test_loader
