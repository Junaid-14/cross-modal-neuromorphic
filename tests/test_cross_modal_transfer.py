"""Tests for utils/cross_modal_transfer.py — Paper 1 engram protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from utils.cross_modal_transfer import (
    _shd_to_identity,
    extract_shared_embeddings,
    evaluate_cross_modal_transfer,
)
from m7_trimodal_run_provenance import (
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    get_device,
)


# ── Helpers ───────────────────────────────────────────────────────────────

class FakeVisualDataset(Dataset):
    """Fake N-MNIST: 100 samples, 10 per class."""
    def __init__(self, n_per_class=10):
        self.data = []
        for c in range(10):
            for _ in range(n_per_class):
                self.data.append((torch.randn(25, 2, 34, 34) * 0.1 + c * 0.05, c))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class FakeAudioDataset(Dataset):
    """Fake SHD: 200 samples, 20 per SHD label (10 per identity)."""
    def __init__(self, n_per_shd_label=10):
        self.data = []
        for shd_label in range(20):
            for _ in range(n_per_shd_label):
                self.data.append((torch.randn(100, 1, 1, 700) * 0.1 + shd_label * 0.05, shd_label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def _make_model(cfg=None):
    if cfg is None:
        cfg = ExperimentConfig(
            seed=42, task_order=["nmnist", "shd", "dvs"],
            shared_dim=64, align_shared_dim=16, num_patterns=8,
            device="cpu",
        )
    tasks = {
        "nmnist": {"num_classes": 10, "type": "visual"},
        "shd": {"num_classes": 20, "type": "audio"},
        "dvs": {"num_classes": 11, "type": "visual_motion"},
    }
    return M7_ContinualAdaptiveModel(tasks=tasks, cfg=cfg).to(get_device(cfg)), cfg


# ── SHD mapping unit test ─────────────────────────────────────────────────

class TestSHDIdentityMapping:
    def test_all_valid_labels(self):
        for lbl in range(20):
            identity = _shd_to_identity(lbl)
            assert 0 <= identity <= 9
            assert identity == lbl % 10

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _shd_to_identity(-1)
        with pytest.raises(ValueError):
            _shd_to_identity(20)


# ── Extraction test ───────────────────────────────────────────────────────

class TestExtractSharedEmbeddings:
    def test_visual_extraction_shape(self):
        model, cfg = _make_model()
        ds = FakeVisualDataset(n_per_class=5)
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        embs, lbls = extract_shared_embeddings(
            model, loader, "nmnist", "nmnist", get_device(cfg)
        )
        assert embs.shape[1] == cfg.align_shared_dim
        assert len(embs) == len(ds)
        assert set(np.unique(lbls).tolist()) == set(range(10))

    def test_audio_extraction_maps_labels(self):
        model, cfg = _make_model()
        ds = FakeAudioDataset(n_per_shd_label=5)
        loader = DataLoader(ds, batch_size=20, shuffle=False)
        embs, lbls = extract_shared_embeddings(
            model, loader, "shd", "shd", get_device(cfg), map_labels=_shd_to_identity
        )
        assert set(np.unique(lbls).tolist()) == set(range(10))
        assert len(embs) == len(ds)

    def test_max_samples_respected(self):
        model, cfg = _make_model()
        ds = FakeVisualDataset(n_per_class=20)
        loader = DataLoader(ds, batch_size=10, shuffle=False)
        embs, lbls = extract_shared_embeddings(
            model, loader, "nmnist", "nmnist", get_device(cfg), max_samples=50
        )
        assert len(embs) == 50
        assert len(lbls) == 50


# ── Full protocol test ────────────────────────────────────────────────────

class TestEvaluateCrossModalTransfer:
    def test_returns_expected_keys(self):
        model, cfg = _make_model()
        vis_ds = FakeVisualDataset(n_per_class=20)
        aud_ds = FakeAudioDataset(n_per_shd_label=20)
        vis_loader = DataLoader(vis_ds, batch_size=20, shuffle=False)
        aud_loader = DataLoader(aud_ds, batch_size=20, shuffle=False)

        result = evaluate_cross_modal_transfer(
            model, vis_loader, aud_loader, get_device(cfg), samples_per_class=20
        )
        expected_keys = {
            "v2a", "a2v", "prototype_alignment",
            "prototype_alignment_off_diag",
            "sil_visual", "sil_audio", "chance",
        }
        assert set(result.keys()) == expected_keys
        assert result["chance"] == pytest.approx(0.10, abs=0.001)

    def test_untrained_model_near_chance(self):
        """
        Paper 1 reproduction: a model WITHOUT alignment should show
        V2A ~ chance and no discriminative prototype alignment.
        """
        model, cfg = _make_model()
        vis_ds = FakeVisualDataset(n_per_class=50)
        aud_ds = FakeAudioDataset(n_per_shd_label=50)
        vis_loader = DataLoader(vis_ds, batch_size=30, shuffle=False)
        aud_loader = DataLoader(aud_ds, batch_size=30, shuffle=False)

        result = evaluate_cross_modal_transfer(
            model, vis_loader, aud_loader, get_device(cfg), samples_per_class=50
        )
        # Untrained model → near chance, no discriminative alignment
        assert result["v2a"] < 0.20, f"Untrained V2A {result['v2a']:.3f} should be < 0.20"
        assert result["a2v"] < 0.20, f"Untrained A2V {result['a2v']:.3f} should be < 0.20"
        # For random embeddings all prototypes are ~equally similar;
        # the discriminative signal is the ratio (should be ~1.0) not the absolute cosine.
        ratio = result["prototype_alignment"] / (result["prototype_alignment_off_diag"] + 1e-8)
        assert 0.8 <= ratio <= 1.3, \
            f"Untrained alignment ratio {ratio:.2f}x should be ~1.0 (no discriminative alignment)"
        print(f"\n   Untrained baseline: V2A={result['v2a']:.3f}, "
              f"align_ratio={ratio:.2f}x → near chance ✓")

    def test_silhouette_scores_valid(self):
        model, cfg = _make_model()
        vis_ds = FakeVisualDataset(n_per_class=20)
        aud_ds = FakeAudioDataset(n_per_shd_label=20)
        vis_loader = DataLoader(vis_ds, batch_size=20, shuffle=False)
        aud_loader = DataLoader(aud_ds, batch_size=20, shuffle=False)

        result = evaluate_cross_modal_transfer(
            model, vis_loader, aud_loader, get_device(cfg), samples_per_class=20
        )
        assert -1.0 <= result["sil_visual"] <= 1.0
        assert -1.0 <= result["sil_audio"] <= 1.0
