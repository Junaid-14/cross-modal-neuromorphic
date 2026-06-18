"""Smoke tests for all registered model variants."""

import pytest
import torch

from Models.registry import build_model


VISUAL_INPUT = torch.randn(2, 25, 2, 34, 34)
AUDIO_INPUT = torch.randn(2, 100, 700)
NTIDIGITS_INPUT = torch.randn(2, 100, 64)
CIFAR10DVS_INPUT = torch.randn(2, 25, 2, 34, 34)

MODEL_VARIANTS = ["baseline", "scl", "hopfield", "hgrn", "hybrid"]


class TestModelForward:
    """Forward-pass shape tests for every model variant and modality."""

    @pytest.mark.parametrize("variant", MODEL_VARIANTS)
    def test_visual_forward(self, variant):
        model = build_model(variant, input_type="nmnist", num_classes=10)
        spk_out, features = model(VISUAL_INPUT)
        assert spk_out.shape == (2, 10)
        assert features.shape == (2, 512)

    @pytest.mark.parametrize("variant", MODEL_VARIANTS)
    def test_audio_forward(self, variant):
        model = build_model(variant, input_type="shd", num_classes=20)
        spk_out, features = model(AUDIO_INPUT)
        assert spk_out.shape == (2, 20)
        assert features.shape == (2, 512)

    @pytest.mark.parametrize("variant", MODEL_VARIANTS)
    def test_cifar10dvs_forward(self, variant):
        model = build_model(variant, input_type="cifar10dvs", num_classes=10)
        spk_out, features = model(CIFAR10DVS_INPUT)
        assert spk_out.shape == (2, 10)
        assert features.shape == (2, 512)

    @pytest.mark.parametrize("variant", MODEL_VARIANTS)
    def test_ntidigits18_forward(self, variant):
        model = build_model(variant, input_type="ntidigits18", num_classes=10, input_size=64)
        spk_out, features = model(NTIDIGITS_INPUT)
        assert spk_out.shape == (2, 10)
        assert features.shape == (2, 512)

    @pytest.mark.parametrize("variant", MODEL_VARIANTS)
    def test_forward_batch_size_one(self, variant):
        model = build_model(variant, input_type="nmnist", num_classes=10)
        x = torch.randn(1, 25, 2, 34, 34)
        spk_out, features = model(x)
        assert spk_out.shape == (1, 10)
        assert features.shape == (1, 512)


class TestModelRegistry:
    """Tests for the model registry / factory."""

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown model variant"):
            build_model("nonexistent", input_type="nmnist")
