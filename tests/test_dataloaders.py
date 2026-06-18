"""Smoke tests for dataloaders (uses mocked tonic datasets to avoid downloads)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from Dataloaders.cifar10dvs_loader import get_cifar10dvs_loaders
from Dataloaders.dvs_gesture_loader import get_dvs_gesture_loaders
from Dataloaders.nmnist_loader import get_nmnist_loaders
from Dataloaders.ntidigits18_loader import get_ntidigits18_loaders
from Dataloaders.shd_loader import get_shd_loaders
from Dataloaders.ssc_loader import get_ssc_loaders


# ---------------------------------------------------------------------------
# Helpers: tiny fake tonic datasets
# ---------------------------------------------------------------------------

class FakeNMNIST:
    sensor_size = (34, 34, 2)

    def __init__(self, save_to, transform, train):
        self.transform = transform
        self.train = train

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        # Return pre-transformed frames (mimics what tonic ToFrame returns: T, C, H, W)
        frames = np.random.rand(20, 2, 34, 34).astype(np.float32)
        return frames, idx % 10


class FakeSHD:
    def __init__(self, save_to, train):
        self.train = train

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        events = np.array(
            [(i, 0, 0, i * 100) for i in range(10)],
            dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")],
        )
        return events, idx % 20


class FakeDVSGesture:
    sensor_size = (128, 128, 2)

    def __init__(self, save_to, transform, train):
        self.transform = transform
        self.train = train

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        # Tonic ToFrame returns (T, C, H, W)
        frames = np.random.rand(20, 2, 128, 128).astype(np.float32)
        return frames, idx % 11


class FakeSSC:
    sensor_size = (700,)
    classes = list(range(35))

    def __init__(self, save_to, train):
        self.train = train

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        events = np.array(
            [(i, 0, 0, i * 100) for i in range(10)],
            dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")],
        )
        return events, idx % 35


class FakeCIFAR10DVS:
    sensor_size = (128, 128, 2)

    def __init__(self, save_to, transform=None, target_transform=None, transforms=None):
        self.transform = transform

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        frames = np.random.rand(20, 2, 128, 128).astype(np.float32)
        return frames, idx % 10


class FakeNTIDIGITS18:
    sensor_size = (64, 1, 1)

    def __init__(self, save_to, train=True, single_digits=False):
        self.train = train
        self.single_digits = single_digits

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        events = np.array(
            [(i, 0, 0, i * 100) for i in range(10)],
            dtype=[("x", "<i8"), ("y", "<i8"), ("p", "<i8"), ("t", "<i8")],
        )
        return events, idx % 10


# ---------------------------------------------------------------------------
# N-MNIST
# ---------------------------------------------------------------------------

class TestNMNISTLoader:
    def test_loader_shapes(self):
        with patch("Dataloaders.nmnist_loader.tonic.datasets.NMNIST", FakeNMNIST):
            train_loader, test_loader = get_nmnist_loaders(
                batch_size=2, time_steps=25, num_workers=0, pin_memory=False
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 25, 2, 34, 34)
            assert y.shape == (2,)
            assert y.dtype == torch.long

            x_test, y_test = next(iter(test_loader))
            assert x_test.shape[0] <= 2  # drop_last=False for test

    def test_collate_pad(self):
        """Frames with fewer than time_steps should be padded."""
        with patch("Dataloaders.nmnist_loader.tonic.datasets.NMNIST", FakeNMNIST):
            train_loader, _ = get_nmnist_loaders(
                batch_size=2, time_steps=30, num_workers=0, pin_memory=False
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 30, 2, 34, 34)

    def test_collate_truncate(self):
        """Frames with more than time_steps should be truncated."""
        with patch("Dataloaders.nmnist_loader.tonic.datasets.NMNIST", FakeNMNIST):
            train_loader, _ = get_nmnist_loaders(
                batch_size=2, time_steps=10, num_workers=0, pin_memory=False
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 10, 2, 34, 34)


# ---------------------------------------------------------------------------
# SHD
# ---------------------------------------------------------------------------

class TestSHDLoader:
    def test_loader_shapes(self, tmp_path):
        cache_path = tmp_path / "shd_test_cache.pt"
        with patch("Dataloaders.shd_loader.tonic.datasets.SHD", FakeSHD):
            train_loader, test_loader = get_shd_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                force_recreate=True,
            )
            x, y = next(iter(train_loader))
            # SHD dense format: (time_bins, 1, 1, channels)
            assert x.shape == (2, 100, 1, 1, 700)
            assert y.dtype == torch.long

    def test_cache_load(self, tmp_path):
        cache_path = tmp_path / "shd_test_cache.pt"
        with patch("Dataloaders.shd_loader.tonic.datasets.SHD", FakeSHD):
            # First call creates cache
            get_shd_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                force_recreate=True,
            )
            # Second call loads from cache (no mock needed here since cache exists)
            train_loader, _ = get_shd_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                use_cache=True,
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 100, 1, 1, 700)


# ---------------------------------------------------------------------------
# DVS-Gesture
# ---------------------------------------------------------------------------

class TestDVSGestureLoader:
    def test_loader_shapes(self):
        with patch("Dataloaders.dvs_gesture_loader.tonic.datasets.DVSGesture", FakeDVSGesture):
            train_loader, test_loader = get_dvs_gesture_loaders(
                batch_size=2,
                time_steps=25,
                num_workers=0,
                pin_memory=False,
                target_spatial_size=(128, 128),
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 25, 2, 128, 128)
            assert y.dtype == torch.long

    def test_resize(self):
        with patch("Dataloaders.dvs_gesture_loader.tonic.datasets.DVSGesture", FakeDVSGesture):
            train_loader, _ = get_dvs_gesture_loaders(
                batch_size=2,
                time_steps=25,
                num_workers=0,
                pin_memory=False,
                target_spatial_size=(64, 64),
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 25, 2, 64, 64)


# ---------------------------------------------------------------------------
# SSC
# ---------------------------------------------------------------------------

class TestSSCLoader:
    def test_loader_shapes(self, tmp_path):
        cache_path = tmp_path / "ssc_test_cache.pt"
        with patch("Dataloaders.ssc_loader.tonic.datasets.SSC", FakeSSC):
            train_loader, test_loader = get_ssc_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                force_recreate=True,
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 100, 1, 1, 700)
            assert y.dtype == torch.long


# ---------------------------------------------------------------------------
# CIFAR10-DVS
# ---------------------------------------------------------------------------

class TestCIFAR10DVSLoader:
    def test_loader_shapes(self):
        with patch(
            "Dataloaders.cifar10dvs_loader.tonic.datasets.CIFAR10DVS", FakeCIFAR10DVS
        ):
            train_loader, val_loader, test_loader = get_cifar10dvs_loaders(
                batch_size=2,
                time_steps=25,
                num_workers=0,
                pin_memory=False,
                target_spatial_size=(34, 34),
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 25, 2, 34, 34)
            assert y.dtype == torch.long

    def test_no_resize(self):
        with patch(
            "Dataloaders.cifar10dvs_loader.tonic.datasets.CIFAR10DVS", FakeCIFAR10DVS
        ):
            train_loader, _, _ = get_cifar10dvs_loaders(
                batch_size=2,
                time_steps=25,
                num_workers=0,
                pin_memory=False,
                target_spatial_size=None,
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 25, 2, 128, 128)


# ---------------------------------------------------------------------------
# N-TIDIGITS18
# ---------------------------------------------------------------------------

class TestNTIDIGITS18Loader:
    def test_loader_shapes(self, tmp_path):
        cache_path = tmp_path / "ntidigits18_test_cache.pt"
        with patch(
            "Dataloaders.ntidigits18_loader.tonic.datasets.NTIDIGITS18", FakeNTIDIGITS18
        ):
            train_loader, test_loader = get_ntidigits18_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                force_recreate=True,
                single_digits=True,
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 100, 1, 1, 64)
            assert y.dtype == torch.long

    def test_cache_load(self, tmp_path):
        cache_path = tmp_path / "ntidigits18_test_cache.pt"
        with patch(
            "Dataloaders.ntidigits18_loader.tonic.datasets.NTIDIGITS18", FakeNTIDIGITS18
        ):
            get_ntidigits18_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                force_recreate=True,
            )
            train_loader, _ = get_ntidigits18_loaders(
                batch_size=2,
                num_workers=0,
                save_to=str(tmp_path),
                cache_path=cache_path,
                use_cache=True,
            )
            x, y = next(iter(train_loader))
            assert x.shape == (2, 100, 1, 1, 64)
