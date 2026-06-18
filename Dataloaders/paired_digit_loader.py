"""
PairedDigitLoader — class-balanced, label-matched cross-modal loader for N-MNIST + SHD.

N-MNIST labels (0-9) pass through unchanged.
SHD labels (0-19) collapse to digit identity via shd_to_identity(): English 0-9 and German 0-9 both map to 0-9.

DVS-Gesture is intentionally excluded (different concept set).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
#  SHD → digit-identity mapping
# ═══════════════════════════════════════════════════════════════════════════════

def shd_to_identity(shd_label: int) -> int:
    """
    Map SHD label (0-19) to digit identity (0-9).

    SHD has 20 classes: spoken digits 0-9 in English (labels 0-9) and German (labels 10-19).
    This function removes the language dimension, collapsing both language variants
    onto the shared digit-identity space 0-9.

    Args:
        shd_label: integer in [0, 19]

    Returns:
        digit identity in [0, 9]

    Raises:
        ValueError: if shd_label is outside [0, 19]

    Reference:
        Cramer et al. (2020), "The Heidelberg Spiking Data Sets for the Systematic
        Evaluation of Spiking Neural Networks", IEEE TNNLS.
        Labels 0-9  → English digits 0-9
        Labels 10-19 → German digits 0-9
    """
    if not (0 <= shd_label <= 19):
        raise ValueError(f"SHD label must be in [0, 19], got {shd_label}")
    return int(shd_label % 10)


def nmnist_to_identity(nmnist_label: int) -> int:
    """
    N-MNIST labels are already digit identities 0-9.
    This function exists for symmetry and defensive validation.
    """
    if not (0 <= nmnist_label <= 9):
        raise ValueError(f"N-MNIST label must be in [0, 9], got {nmnist_label}")
    return int(nmnist_label)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET WRAPPER for SHD with mapped labels
# ═══════════════════════════════════════════════════════════════════════════════

class SHDDigitIdentityDataset(Dataset):
    """
    Wraps an SHD dataset (or any dataset yielding (tensor, int_label)) and remaps
    labels to digit identity 0-9 via shd_to_identity().
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        # Pre-index all samples by identity label for fast class-balanced access
        self.by_identity: Dict[int, List[int]] = {i: [] for i in range(10)}
        for idx in range(len(base_dataset)):
            _, label = base_dataset[idx]
            identity = shd_to_identity(int(label))
            self.by_identity[identity].append(idx)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, label = self.base_dataset[idx]
        identity = shd_to_identity(int(label))
        return data, identity

    def samples_for_identity(self, identity: int) -> List[int]:
        """Return list of dataset indices belonging to the given digit identity."""
        if not (0 <= identity <= 9):
            raise ValueError(f"identity must be in [0, 9], got {identity}")
        return self.by_identity[identity]


# ═══════════════════════════════════════════════════════════════════════════════
#  PAIRED DIGIT DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class PairedDigitDataset(Dataset):
    """
    Yokes N-MNIST (visual) and SHD (audio) by shared digit identity 0-9.

    Each item returned is **label-matched but not instance-paired**:
    - visual_item:  a random (or cycled) sample from N-MNIST with digit == c
    - audio_item:   a random (or cycled) sample from SHD with digit == c
    - shared_label: the digit identity c ∈ [0, 9]

    The dataset length is ``samples_per_class * 10`` (one visual+audio pair per
    class per slot).  When iterated with a DataLoader of batch_size divisible by
    10, every batch is perfectly class-balanced.
    """

    def __init__(
        self,
        visual_dataset: Dataset,
        audio_dataset: Dataset,
        samples_per_class: int = 500,
        visual_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Args:
            visual_dataset: N-MNIST dataset (or any yielding (tensor, label_in_0_9))
            audio_dataset:  SHD dataset (or any yielding (tensor, label_in_0_19))
            samples_per_class: number of pairs to synthesise per digit identity.
                               This caps the dataset length at ``samples_per_class * 10``.
            visual_transform: optional transform applied to visual tensor
            audio_transform:  optional transform applied to audio tensor
            seed: RNG seed for shuffling indices within each class
        """
        self.visual_dataset = visual_dataset
        self.audio_dataset = SHDDigitIdentityDataset(audio_dataset)
        self.samples_per_class = samples_per_class
        self.visual_transform = visual_transform
        self.audio_transform = audio_transform

        # Index visual dataset by identity 0-9
        self.visual_by_id: Dict[int, List[int]] = {i: [] for i in range(10)}
        for idx in range(len(visual_dataset)):
            _, label = visual_dataset[idx]
            identity = nmnist_to_identity(int(label))
            self.visual_by_id[identity].append(idx)

        # Determine max feasible samples_per_class
        min_visual = min(len(v) for v in self.visual_by_id.values())
        min_audio = min(len(v) for v in self.audio_dataset.by_identity.values())
        self._feasible_spc = min(min_visual, min_audio, samples_per_class)

        if self._feasible_spc < samples_per_class:
            print(
                f"⚠️  PairedDigitDataset: requested samples_per_class={samples_per_class}, "
                f"but limiting to {self._feasible_spc} (visual min={min_visual}, audio min={min_audio})"
            )
            self.samples_per_class = self._feasible_spc

        # Build per-class index tables by cycling through shuffled pools.
        rng = np.random.default_rng(seed)
        self.visual_indices: Dict[int, np.ndarray] = {}
        self.audio_indices: Dict[int, np.ndarray] = {}

        for c in range(10):
            vis_pool = np.array(self.visual_by_id[c])
            aud_pool = np.array(self.audio_dataset.by_identity[c])
            rng.shuffle(vis_pool)
            rng.shuffle(aud_pool)
            vis_repeated = np.tile(vis_pool, math.ceil(self.samples_per_class / len(vis_pool)))
            aud_repeated = np.tile(aud_pool, math.ceil(self.samples_per_class / len(aud_pool)))
            self.visual_indices[c] = vis_repeated[: self.samples_per_class]
            self.audio_indices[c] = aud_repeated[: self.samples_per_class]

        # Build a class-balanced global index order.
        # Each "slot" is one sample from each of the 10 classes.
        # We shuffle the order of classes within each slot, then shuffle slot order,
        # producing a flat list where every contiguous block of 10 is class-balanced.
        slots = []
        for s in range(self.samples_per_class):
            class_order = rng.permutation(10)
            slots.append([int(c + s * 10) for c in class_order])
        rng.shuffle(slots)
        self._flat_indices: List[int] = [idx for slot in slots for idx in slot]

    def __len__(self) -> int:
        return self.samples_per_class * 10

    def _decode_flat(self, flat_idx: int) -> Tuple[int, int]:
        """Map a flat index to (slot, class)."""
        # The flat index was constructed as: slot * 10 + class
        # But because we shuffled slots and class orders, we need to reverse-map.
        # Actually, _flat_indices[i] stores the original (slot*10 + class) value.
        # So flat_idx is already slot*10 + class.
        c = flat_idx % 10
        s = flat_idx // 10
        return s, c

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            (visual_tensor, audio_tensor, shared_label)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")

        flat_idx = self._flat_indices[idx]
        s, c = self._decode_flat(flat_idx)

        vis_idx = int(self.visual_indices[c][s])
        aud_idx = int(self.audio_indices[c][s])

        vis_data, _ = self.visual_dataset[vis_idx]
        aud_data, _ = self.audio_dataset[aud_idx]

        if self.visual_transform is not None:
            vis_data = self.visual_transform(vis_data)
        if self.audio_transform is not None:
            aud_data = self.audio_transform(aud_data)

        if not isinstance(vis_data, torch.Tensor):
            vis_data = torch.as_tensor(vis_data, dtype=torch.float32)
        if not isinstance(aud_data, torch.Tensor):
            aud_data = torch.as_tensor(aud_data, dtype=torch.float32)

        return vis_data, aud_data, c


# ═══════════════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTION for (visual, audio, label) tuples
# ═══════════════════════════════════════════════════════════════════════════════

def paired_digit_collate_fn(batch):
    """
    Collate a list of (visual, audio, label) tuples into batched tensors.

    Args:
        batch: list of tuples (vis_tensor, aud_tensor, label_int)

    Returns:
        (visual_batch, audio_batch, label_batch) where visual_batch and audio_batch
        are stacked along the batch dimension, and label_batch is a 1-D long tensor.
    """
    visuals, audios, labels = zip(*batch)
    return (
        torch.stack(visuals, dim=0),
        torch.stack(audios, dim=0),
        torch.tensor(labels, dtype=torch.long),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_paired_digit_loaders(
    visual_dataset: Dataset,
    audio_dataset: Dataset,
    batch_size: int = 30,
    samples_per_class: int = 500,
    num_workers: int = 0,
    seed: int = 42,
    visual_transform: Optional[Callable] = None,
    audio_transform: Optional[Callable] = None,
) -> DataLoader:
    """
    Build a single DataLoader for paired visual-audio digit training.

    Args:
        visual_dataset: N-MNIST (or equivalent visual-digit dataset)
        audio_dataset:  SHD (or equivalent audio-digit dataset with labels 0-19)
        batch_size:     should be divisible by 10 for perfect class balance
        samples_per_class: cap on pairs per digit identity
        num_workers:    DataLoader workers
        seed:           RNG seed for internal shuffling
        visual_transform: optional transform on visual samples
        audio_transform:  optional transform on audio samples

    Returns:
        DataLoader yielding (visual_batch, audio_batch, shared_labels)
    """
    if batch_size % 10 != 0:
        print(
            f"⚠️  batch_size={batch_size} is not divisible by 10; "
            "batches will not be perfectly class-balanced."
        )

    paired_ds = PairedDigitDataset(
        visual_dataset=visual_dataset,
        audio_dataset=audio_dataset,
        samples_per_class=samples_per_class,
        visual_transform=visual_transform,
        audio_transform=audio_transform,
        seed=seed,
    )

    # shuffle=False because PairedDigitDataset already shuffles its internal order
    loader = DataLoader(
        paired_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=paired_digit_collate_fn,
    )
    return loader
