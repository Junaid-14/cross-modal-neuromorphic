"""Unit tests for PairedDigitLoader and SHD→digit-identity mapping."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from Dataloaders.paired_digit_loader import (
    shd_to_identity,
    nmnist_to_identity,
    SHDDigitIdentityDataset,
    PairedDigitDataset,
    paired_digit_collate_fn,
    get_paired_digit_loaders,
)


# ---------------------------------------------------------------------------
# Helpers: tiny fake datasets
# ---------------------------------------------------------------------------

class FakeNMNISTDataset(Dataset):
    """Fake N-MNIST: 100 samples, 10 per class, labels 0-9."""

    def __init__(self, n_per_class: int = 10):
        self.n_per_class = n_per_class
        self.data = []
        for c in range(10):
            for i in range(n_per_class):
                # Visual tensor: (T=25, C=2, H=34, W=34)
                tensor = torch.randn(25, 2, 34, 34) * 0.1 + c
                self.data.append((tensor, c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FakeSHDDataset(Dataset):
    """Fake SHD: 200 samples, 10 per SHD label (0-19) → 20 per digit identity."""

    def __init__(self, n_per_shd_label: int = 10):
        self.n_per_shd_label = n_per_shd_label
        self.data = []
        for shd_label in range(20):
            for i in range(n_per_shd_label):
                # Audio tensor: (T=100, C=1, H=1, W=700)
                tensor = torch.randn(100, 1, 1, 700) * 0.1 + shd_label
                self.data.append((tensor, shd_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# SHD → identity mapping
# ---------------------------------------------------------------------------

class TestSHDToIdentity:
    """Verify the SHD label→digit-identity mapping against the dataset docs."""

    def test_english_digits(self):
        """Labels 0-9 are English digits 0-9 → identity = label."""
        for lbl in range(10):
            assert shd_to_identity(lbl) == lbl

    def test_german_digits(self):
        """Labels 10-19 are German digits 0-9 → identity = label - 10."""
        for lbl in range(10, 20):
            assert shd_to_identity(lbl) == lbl - 10

    def test_all_map_to_0_9(self):
        """Every valid SHD label maps into [0, 9]."""
        identities = [shd_to_identity(lbl) for lbl in range(20)]
        assert all(0 <= i <= 9 for i in identities)
        assert set(identities) == set(range(10))

    def test_invalid_labels_raise(self):
        """Out-of-range labels must raise ValueError."""
        with pytest.raises(ValueError):
            shd_to_identity(-1)
        with pytest.raises(ValueError):
            shd_to_identity(20)
        with pytest.raises(ValueError):
            shd_to_identity(100)

    def test_consistency_with_modulo(self):
        """For all valid labels, identity == label % 10."""
        for lbl in range(20):
            assert shd_to_identity(lbl) == lbl % 10

    def test_nmnist_identity_pass_through(self):
        """N-MNIST labels 0-9 pass through unchanged."""
        for lbl in range(10):
            assert nmnist_to_identity(lbl) == lbl

    def test_nmnist_invalid_raises(self):
        with pytest.raises(ValueError):
            nmnist_to_identity(10)
        with pytest.raises(ValueError):
            nmnist_to_identity(-1)


# ---------------------------------------------------------------------------
# SHDDigitIdentityDataset wrapper
# ---------------------------------------------------------------------------

class TestSHDDigitIdentityDataset:
    def test_label_remapping(self):
        base = FakeSHDDataset(n_per_shd_label=5)
        wrapped = SHDDigitIdentityDataset(base)

        for idx in range(len(base)):
            _, original_label = base[idx]
            _, identity = wrapped[idx]
            assert identity == shd_to_identity(original_label)
            assert 0 <= identity <= 9

    def test_by_identity_indexing(self):
        base = FakeSHDDataset(n_per_shd_label=5)
        wrapped = SHDDigitIdentityDataset(base)

        for c in range(10):
            indices = wrapped.samples_for_identity(c)
            # 2 SHD labels map to each identity → 2 * 5 = 10 samples
            assert len(indices) == 10
            for idx in indices:
                _, lbl = wrapped[idx]
                assert lbl == c


# ---------------------------------------------------------------------------
# PairedDigitDataset
# ---------------------------------------------------------------------------

class TestPairedDigitDataset:
    def test_length(self):
        vis = FakeNMNISTDataset(n_per_class=20)
        aud = FakeSHDDataset(n_per_shd_label=10)  # 20 per identity
        ds = PairedDigitDataset(vis, aud, samples_per_class=15, seed=0)
        assert len(ds) == 15 * 10

    def test_items_are_tensors(self):
        vis = FakeNMNISTDataset(n_per_class=10)
        aud = FakeSHDDataset(n_per_shd_label=10)
        ds = PairedDigitDataset(vis, aud, samples_per_class=5, seed=0)

        for idx in range(len(ds)):
            v, a, lbl = ds[idx]
            assert isinstance(v, torch.Tensor)
            assert isinstance(a, torch.Tensor)
            assert isinstance(lbl, int)
            assert 0 <= lbl <= 9

    def test_class_balanced_over_full_epoch(self):
        """Iterate the full dataset; every identity should appear equally often."""
        vis = FakeNMNISTDataset(n_per_class=50)
        aud = FakeSHDDataset(n_per_shd_label=50)
        ds = PairedDigitDataset(vis, aud, samples_per_class=30, seed=0)

        counts = {c: 0 for c in range(10)}
        for idx in range(len(ds)):
            _, _, lbl = ds[idx]
            counts[lbl] += 1

        assert all(c == 30 for c in counts.values()), f"Class counts unequal: {counts}"

    def test_visual_audio_shapes(self):
        vis = FakeNMNISTDataset(n_per_class=10)
        aud = FakeSHDDataset(n_per_shd_label=10)
        ds = PairedDigitDataset(vis, aud, samples_per_class=5, seed=0)

        v, a, lbl = ds[0]
        assert v.shape == (25, 2, 34, 34)
        assert a.shape == (100, 1, 1, 700)

    def test_label_matched_not_instance_paired(self):
        """
        For a given returned item, visual and audio share the same label
        but are drawn from independent index pools.
        """
        vis = FakeNMNISTDataset(n_per_class=100)
        aud = FakeSHDDataset(n_per_shd_label=50)
        ds = PairedDigitDataset(vis, aud, samples_per_class=50, seed=0)

        # Collect all visual and audio base-dataset indices used for class 3
        vis_indices = set()
        aud_indices = set()
        for idx in range(len(ds)):
            _, _, lbl = ds[idx]
            if lbl == 3:
                # Reconstruct indices via internal tables
                s = idx // 10
                vis_idx = int(ds.visual_indices[3][s])
                aud_idx = int(ds.audio_indices[3][s])
                vis_indices.add(vis_idx)
                aud_indices.add(aud_idx)

        # Visual and audio indices should come from disjoint pools
        # (since the base datasets are different objects)
        assert len(vis_indices) > 1, "Expected multiple distinct visual samples for class 3"
        assert len(aud_indices) > 1, "Expected multiple distinct audio samples for class 3"


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

class TestPairedDigitCollate:
    def test_stack_shapes(self):
        batch = [
            (torch.randn(25, 2, 34, 34), torch.randn(100, 1, 1, 700), i % 10)
            for i in range(30)
        ]
        v_batch, a_batch, lbl_batch = paired_digit_collate_fn(batch)
        assert v_batch.shape == (30, 25, 2, 34, 34)
        assert a_batch.shape == (30, 100, 1, 1, 700)
        assert lbl_batch.shape == (30,)
        assert lbl_batch.dtype == torch.long


# ---------------------------------------------------------------------------
# Factory / DataLoader integration
# ---------------------------------------------------------------------------

class TestGetPairedDigitLoaders:
    def test_batch_class_balance(self):
        """
        With batch_size = 30 and 10 classes, every batch should contain
        exactly 3 samples per digit identity.
        """
        vis = FakeNMNISTDataset(n_per_class=100)
        aud = FakeSHDDataset(n_per_shd_label=50)
        loader = get_paired_digit_loaders(
            visual_dataset=vis,
            audio_dataset=aud,
            batch_size=30,
            samples_per_class=90,
            num_workers=0,
            seed=42,
        )

        checked = 0
        for v_batch, a_batch, lbl_batch in loader:
            assert v_batch.shape[0] == 30
            assert a_batch.shape[0] == 30
            assert lbl_batch.shape[0] == 30

            # Count occurrences per identity
            unique, counts = torch.unique(lbl_batch, return_counts=True)
            assert set(unique.tolist()) == set(range(10)), f"Missing classes: {unique.tolist()}"
            assert torch.all(counts == 3), f"Class counts not equal to 3: {counts.tolist()}"

            checked += 1
            if checked >= 3:
                break

    def test_warning_on_imbalanced_batch_size(self, capsys):
        """batch_size not divisible by 10 should print a warning."""
        vis = FakeNMNISTDataset(n_per_class=50)
        aud = FakeSHDDataset(n_per_shd_label=25)
        _ = get_paired_digit_loaders(
            visual_dataset=vis,
            audio_dataset=aud,
            batch_size=32,
            samples_per_class=50,
            num_workers=0,
            seed=0,
        )
        captured = capsys.readouterr()
        assert "not divisible by 10" in captured.out


# ---------------------------------------------------------------------------
# Acceptance: real SHD file mapping verification
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("ICONS_M7/datasets/SHD/shd_train.h5").exists(),
    reason="Real SHD data not present; run only when dataset is available",
)
class TestRealSHDMapping:
    def test_real_shd_labels_cover_all_20(self):
        import h5py

        with h5py.File("ICONS_M7/datasets/SHD/shd_train.h5", "r") as f:
            labels = np.array(f["labels"])

        unique = np.unique(labels)
        assert len(unique) == 20
        assert unique.min() == 0
        assert unique.max() == 19

    def test_real_shd_identity_distribution(self):
        import h5py

        with h5py.File("ICONS_M7/datasets/SHD/shd_train.h5", "r") as f:
            labels = np.array(f["labels"])

        identities = np.array([shd_to_identity(lbl) for lbl in labels])
        unique_id, counts_id = np.unique(identities, return_counts=True)

        # Every identity 0-9 should be present
        assert set(unique_id.tolist()) == set(range(10))

        # Each identity should have ~2× the per-label count (English + German)
        # Total 8156 samples / 10 identities ≈ 815 each
        assert all(c > 700 for c in counts_id), f"Some identity under-represented: {counts_id}"
        assert all(c < 900 for c in counts_id), f"Some identity over-represented: {counts_id}"
