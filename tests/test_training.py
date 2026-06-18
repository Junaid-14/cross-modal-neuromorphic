"""Smoke tests for the training pipeline."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from Models.registry import build_model
from training.pipeline import train_model, train_step
from training.settings import TrainingConfig


@pytest.fixture
def tiny_visual_dataset():
    """Synthetic visual dataset: 8 samples, 2 classes."""
    x = torch.randn(8, 25, 2, 34, 34)
    y = torch.randint(0, 10, (8,))
    return TensorDataset(x, y)


@pytest.fixture
def tiny_audio_dataset():
    """Synthetic audio dataset: 8 samples, 2 classes."""
    x = torch.randn(8, 100, 700)
    y = torch.randint(0, 20, (8,))
    return TensorDataset(x, y)


class TestTrainStep:
    def test_train_step_runs(self, tiny_visual_dataset):
        model = build_model("baseline", input_type="nmnist", num_classes=10)
        loader = DataLoader(tiny_visual_dataset, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, ce_loss, con_loss, acc = train_step(
            model, loader, optimizer, ce_loss_fn, None, 0.0, device, 1.0
        )
        assert isinstance(loss, float)
        assert 0 <= acc <= 100

    def test_train_step_with_contrastive(self, tiny_visual_dataset):
        from training.losses import SupervisedContrastiveLoss

        model = build_model("scl", input_type="nmnist", num_classes=10)
        loader = DataLoader(tiny_visual_dataset, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        con_loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        device = torch.device("cpu")

        loss, ce_loss, con_loss, acc = train_step(
            model, loader, optimizer, ce_loss_fn, con_loss_fn, 0.1, device, 1.0
        )
        assert isinstance(loss, float)
        assert con_loss > 0 or con_loss == 0  # just ensure it runs


class TestTestStep:
    def test_test_step_runs(self, tiny_visual_dataset):
        from training.pipeline import test_step as _test_step

        model = build_model("baseline", input_type="nmnist", num_classes=10)
        loader = DataLoader(tiny_visual_dataset, batch_size=2)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")

        loss, ce_loss, con_loss, acc = _test_step(
            model, loader, ce_loss_fn, None, 0.0, device
        )
        assert isinstance(loss, float)
        assert 0 <= acc <= 100


class TestTrainModel:
    def test_one_epoch_visual(self, tiny_visual_dataset, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        model = build_model("baseline", input_type="nmnist", num_classes=10)
        train_loader = DataLoader(tiny_visual_dataset, batch_size=2)
        test_loader = DataLoader(tiny_visual_dataset, batch_size=2)
        config = TrainingConfig(
            num_epochs=1, patience=1, checkpoint_dir=checkpoint_dir
        )
        device = torch.device("cpu")

        trained_model, history, best_acc = train_model(
            model,
            train_loader,
            test_loader,
            "test_baseline",
            "nmnist",
            use_contrastive=False,
            device=device,
            config=config,
        )
        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert 0 <= best_acc <= 100
        # Note: checkpoint is only written if val_acc improves; with random data this is not guaranteed

    def test_one_epoch_audio(self, tiny_audio_dataset, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        model = build_model("baseline", input_type="shd", num_classes=20)
        train_loader = DataLoader(tiny_audio_dataset, batch_size=2)
        test_loader = DataLoader(tiny_audio_dataset, batch_size=2)
        config = TrainingConfig(
            num_epochs=1, patience=1, checkpoint_dir=checkpoint_dir
        )
        device = torch.device("cpu")

        trained_model, history, best_acc = train_model(
            model,
            train_loader,
            test_loader,
            "test_baseline_audio",
            "shd",
            use_contrastive=False,
            device=device,
            config=config,
        )
        assert len(history["train_loss"]) == 1
        assert 0 <= best_acc <= 100

    def test_early_stopping(self, tiny_visual_dataset, tmp_path):
        checkpoint_dir = tmp_path / "checkpoints"
        model = build_model("baseline", input_type="nmnist", num_classes=10)
        train_loader = DataLoader(tiny_visual_dataset, batch_size=2)
        test_loader = DataLoader(tiny_visual_dataset, batch_size=2)
        # Patience=0 means it should stop after first epoch if no improvement
        config = TrainingConfig(
            num_epochs=10, patience=0, checkpoint_dir=checkpoint_dir
        )
        device = torch.device("cpu")

        trained_model, history, best_acc = train_model(
            model,
            train_loader,
            test_loader,
            "test_early_stop",
            "nmnist",
            use_contrastive=False,
            device=device,
            config=config,
        )
        # Should have stopped early (after 1 epoch because patience=0)
        assert len(history["train_loss"]) == 1
