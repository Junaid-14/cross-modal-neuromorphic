"""Smoke tests for SNN backbones."""

import pytest
import torch

from Models.backbones import AudioBackbone, VisualBackbone, build_backbone


class TestVisualBackbone:
    """Tests for the conv-based visual backbone."""

    def test_forward_shape(self):
        model = VisualBackbone(input_channels=2, num_classes=10, spatial_size=(34, 34))
        x = torch.randn(2, 25, 2, 34, 34)
        spk_sum, features = model(x)
        assert spk_sum.shape == (2, 10)
        assert features.shape == (2, 512)

    def test_forward_batch_size_one(self):
        model = VisualBackbone(input_channels=2, num_classes=10, spatial_size=(34, 34))
        x = torch.randn(1, 25, 2, 34, 34)
        spk_sum, features = model(x)
        assert spk_sum.shape == (1, 10)
        assert features.shape == (1, 512)

    def test_different_spatial_size(self):
        model = VisualBackbone(input_channels=2, num_classes=5, spatial_size=(64, 64))
        x = torch.randn(2, 25, 2, 64, 64)
        spk_sum, features = model(x)
        assert spk_sum.shape == (2, 5)
        assert features.shape == (2, 512)


class TestAudioBackbone:
    """Tests for the FC-based audio backbone."""

    def test_forward_shape_3d(self):
        model = AudioBackbone(input_size=700, num_classes=20)
        x = torch.randn(2, 100, 700)
        spk_sum, features = model(x)
        assert spk_sum.shape == (2, 20)
        assert features.shape == (2, 512)

    def test_forward_shape_5d(self):
        model = AudioBackbone(input_size=700, num_classes=20)
        x = torch.randn(2, 100, 1, 1, 700)
        spk_sum, features = model(x)
        assert spk_sum.shape == (2, 20)
        assert features.shape == (2, 512)

    def test_forward_batch_size_one(self):
        model = AudioBackbone(input_size=700, num_classes=20)
        x = torch.randn(1, 100, 700)
        spk_sum, features = model(x)
        assert spk_sum.shape == (1, 20)
        assert features.shape == (1, 512)


class TestBuildBackbone:
    """Tests for the backbone factory function."""

    def test_nmnist_returns_visual(self):
        model = build_backbone("nmnist", num_classes=10)
        assert isinstance(model, VisualBackbone)

    def test_dvs_gesture_returns_visual(self):
        model = build_backbone("dvs_gesture", num_classes=11)
        assert isinstance(model, VisualBackbone)

    def test_shd_returns_audio(self):
        model = build_backbone("shd", num_classes=20)
        assert isinstance(model, AudioBackbone)

    def test_ssc_returns_audio(self):
        model = build_backbone("ssc", num_classes=35)
        assert isinstance(model, AudioBackbone)

    def test_cifar10dvs_returns_visual(self):
        model = build_backbone("cifar10dvs", num_classes=10, spatial_size=(34, 34))
        assert isinstance(model, VisualBackbone)

    def test_ntidigits18_returns_audio(self):
        model = build_backbone("ntidigits18", num_classes=10, input_size=64)
        assert isinstance(model, AudioBackbone)

    def test_unknown_input_type_raises(self):
        with pytest.raises(ValueError, match="Unknown input_type"):
            build_backbone("unknown")
