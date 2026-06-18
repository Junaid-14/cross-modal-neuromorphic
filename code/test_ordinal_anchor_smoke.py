#!/usr/bin/env python3
"""Quick smoke test for run_ordinal_anchor.py: Stage 0 + 1 seed MLP."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

from run_ordinal_anchor import (
    ExperimentConfig,
    M7_ContinualAdaptiveModel,
    _build_digit_loaders,
    _load_cached_features,
    compute_rdms,
    get_device,
    run_one_seed,
)


def main():
    cfg = ExperimentConfig(
        seed=42,
        task_order=["nmnist", "shd", "dvs"],
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=32,
        max_epochs=50,
        patience=12,
        gradient_clip=1.0,
        beta=0.9,
        dropout=0.2,
        hidden_dim=512,
        shared_dim=512,
        num_patterns=100,
        hopfield_temperature=1.0,
        num_gru_layers=2,
        num_workers=0,
        time_steps=25,
        pin_memory=True,
        use_contrastive=True,
        contrastive_temperature=0.07,
        contrastive_weight=0.1,
        use_replay=False,
        replay_batch=32,
        replay_weight=0.3,
        replay_samples_per_class=50,
        device="auto",
        align_cross_modal=False,
        align_shared_dim=512,
        align_weight=0.5,
        align_temperature=0.1,
        align_use_orthogonal=False,
        align_orthogonal_weight=0.01,
    )
    device = get_device(cfg)
    print(f"Device: {device}")

    print("\nBuilding loaders...")
    loaders = _build_digit_loaders(cfg)

    print("\nLoading cached features...")
    cached = _load_cached_features(device)

    print("\n=== STAGE 0: RDM diagnostic ===")
    rdm = compute_rdms(cached)
    print(f"Vision↔Audio RDM r = {rdm['vision_audio_rdm_r']:.4f}, p = {rdm['vision_audio_rdm_p']:.4f}")
    print(f"Vision magnitude rho = {rdm['vision_magnitude_rho']:.4f}, p = {rdm['vision_magnitude_p']:.4f}")
    print(f"Audio magnitude rho = {rdm['audio_magnitude_rho']:.4f}, p = {rdm['audio_magnitude_p']:.4f}")

    print("\n=== STAGE 1: one seed / MLP ===")
    res = run_one_seed(42, cfg, loaders, cached, device, head_type="mlp")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
