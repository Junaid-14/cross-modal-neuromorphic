#!/usr/bin/env python3
"""
M7 Ablation Study Runner
========================
Trains all registered model variants on all supported neuromorphic datasets
and logs: accuracy, spike rate, sparsity, and timestep count.

All experiments use fixed seeds for reproducibility.

Example:
    python code/run_ablation.py --datasets nmnist shd --models baseline scl --epochs 5
    python code/run_ablation.py --quick  # 1 epoch, 1 seed, tiny smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import snntorch as snn

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
from Models.registry import build_model
from training.pipeline import train_model
from training.settings import TrainingConfig

# Dataloaders
from Dataloaders.cifar10dvs_loader import get_cifar10dvs_loaders
from Dataloaders.dvs_gesture_loader import get_dvs_gesture_loaders
from Dataloaders.nmnist_loader import get_nmnist_loaders
from Dataloaders.ntidigits18_loader import get_ntidigits18_loaders
from Dataloaders.shd_loader import get_shd_loaders
from Dataloaders.ssc_loader import get_ssc_loaders


# ------------------------------------------------------------------
# Limited loader helper (for quick smoke tests)
# ------------------------------------------------------------------

class _LimitedLoader:
    """Wraps a DataLoader to yield at most N batches."""

    def __init__(self, loader: DataLoader, max_batches: int | None):
        self._loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset

    def __iter__(self):
        if self.max_batches is None:
            yield from self._loader
        else:
            for i, batch in enumerate(self._loader):
                if i >= self.max_batches:
                    break
                yield batch

    def __len__(self):
        if self.max_batches is None:
            return len(self._loader)
        return min(self.max_batches, len(self._loader))


# ------------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------------


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "nmnist": {
        "loader_fn": get_nmnist_loaders,
        "kwargs": {"num_workers": 0},
        "input_type": "nmnist",
        "num_classes": 10,
        "input_channels": 2,
        "input_size": 700,
        "spatial_size": (34, 34),
        "modality": "visual",
    },
    "shd": {
        "loader_fn": get_shd_loaders,
        "kwargs": {"num_workers": 0},
        "input_type": "shd",
        "num_classes": 20,
        "input_channels": 2,
        "input_size": 700,
        "spatial_size": (34, 34),
        "modality": "audio",
    },
    "dvs_gesture": {
        "loader_fn": get_dvs_gesture_loaders,
        "kwargs": {"num_workers": 0},
        "input_type": "dvs_gesture",
        "num_classes": 11,
        "input_channels": 2,
        "input_size": 700,
        "spatial_size": (34, 34),
        "modality": "visual_motion",
    },
    "ssc": {
        "loader_fn": get_ssc_loaders,
        "kwargs": {"num_workers": 0},
        "input_type": "ssc",
        "num_classes": 35,
        "input_channels": 2,
        "input_size": 700,
        "spatial_size": (34, 34),
        "modality": "audio",
    },
    "cifar10dvs": {
        "loader_fn": get_cifar10dvs_loaders,
        "kwargs": {"num_workers": 0, "target_spatial_size": (34, 34)},
        "input_type": "cifar10dvs",
        "num_classes": 10,
        "input_channels": 2,
        "input_size": 700,
        "spatial_size": (34, 34),
        "modality": "visual",
    },
    "ntidigits18": {
        "loader_fn": get_ntidigits18_loaders,
        "kwargs": {"num_workers": 0, "single_digits": True},
        "input_type": "ntidigits18",
        "num_classes": 10,
        "input_channels": 2,
        "input_size": 64,
        "spatial_size": (34, 34),
        "modality": "audio",
    },
}

MODEL_VARIANTS = ["baseline", "scl", "hopfield", "hgrn", "hybrid"]
DEFAULT_SEEDS = [42, 123, 456]

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Spike metrics
# ------------------------------------------------------------------

class SpikeRecorder:
    """Records spike tensors from every snn.Leaky layer during forward passes."""

    def __init__(self):
        self.spikes: Dict[str, List[torch.Tensor]] = {}
        self._hooks: List[Any] = []

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            # snn.Leaky returns (spk, mem) when output=True, else just spk
            if isinstance(out, tuple):
                spk = out[0]
            else:
                spk = out
            self.spikes.setdefault(name, []).append(spk.detach().cpu())
        return hook

    def attach(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, snn.Leaky):
                self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        all_flat: List[torch.Tensor] = []

        for name, spike_list in self.spikes.items():
            cat = torch.cat(spike_list, dim=0)  # (B*T, ...)
            flat = cat.flatten()
            total = flat.numel()
            spikes = flat.sum().item()
            rate = spikes / total if total > 0 else 0.0
            metrics[name] = {
                "spike_rate": round(rate, 6),
                "sparsity": round(1.0 - rate, 6),
            }
            all_flat.append(flat)

        if all_flat:
            global_flat = torch.cat(all_flat)
            g_total = global_flat.numel()
            g_spikes = global_flat.sum().item()
            g_rate = g_spikes / g_total if g_total > 0 else 0.0
            metrics["global"] = {
                "spike_rate": round(g_rate, 6),
                "sparsity": round(1.0 - g_rate, 6),
            }

        return metrics


def measure_spike_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Run a limited forward pass and return per-layer + global spike metrics."""
    recorder = SpikeRecorder()
    recorder.attach(model)
    model.eval()

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= max_batches:
                break
            data = data.to(device)
            _ = model(data)

    recorder.detach()
    return recorder.compute_metrics()


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------

def run_single_experiment(
    dataset_name: str,
    model_variant: str,
    seed: int,
    device: torch.device,
    num_epochs: int,
    patience: int,
    batch_size: int,
    use_contrastive: bool,
    lr: float = 1e-3,
    quiet: bool = False,
    max_train_batches: int | None = None,
    max_test_batches: int | None = None,
) -> Dict[str, Any]:
    """Train one model on one dataset with one seed and return metrics."""
    ds_cfg = DATASET_REGISTRY[dataset_name]
    torch.manual_seed(seed)

    # Build loaders
    loader_fn: Callable = ds_cfg["loader_fn"]
    loader_kwargs = dict(ds_cfg["kwargs"])
    loader_kwargs["batch_size"] = batch_size
    loader_kwargs["save_to"] = "./ICONS_M7/datasets"

    loaders = loader_fn(**loader_kwargs)
    if dataset_name == "cifar10dvs":
        train_loader, val_loader, test_loader = loaders
        # Use val for early-stopping monitoring, test for final metrics
    else:
        train_loader, test_loader = loaders
        val_loader = test_loader

    # Optionally limit batches for smoke tests
    if max_train_batches is not None:
        train_loader = _LimitedLoader(train_loader, max_train_batches)
    if max_test_batches is not None:
        test_loader = _LimitedLoader(test_loader, max_test_batches)
        val_loader = test_loader

    # Determine timesteps from a sample
    sample_x, _ = next(iter(test_loader))
    timesteps = sample_x.shape[1]

    # Build model
    model = build_model(
        variant=model_variant,
        input_type=ds_cfg["input_type"],
        num_classes=ds_cfg["num_classes"],
        input_channels=ds_cfg.get("input_channels", 2),
        input_size=ds_cfg.get("input_size", 700),
        spatial_size=ds_cfg.get("spatial_size", (34, 34)),
    ).to(device)

    config = TrainingConfig(
        learning_rate=lr,
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_dir=RESULTS_DIR / "checkpoints" / f"{dataset_name}_{model_variant}_seed{seed}",
    )

    tag = f"[{dataset_name}] {model_variant} | seed={seed}"
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Running: {tag}")
        print(f"  Device: {device}")
        print(f"  Epochs: {num_epochs}, Patience: {patience}, Batch: {batch_size}")
        print(f"  Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")
        print(f"  Timesteps: {timesteps}")

    t0 = time.time()
    trained_model, history, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        model_name=f"{dataset_name}_{model_variant}_s{seed}",
        dataset_name=dataset_name,
        use_contrastive=use_contrastive,
        device=device,
        config=config,
    )
    train_time = time.time() - t0

    # Spike metrics on test set
    spike_metrics = measure_spike_metrics(
        trained_model, test_loader, device, max_batches=20
    )
    global_spike = spike_metrics.get("global", {})

    result = {
        "dataset": dataset_name,
        "model_variant": model_variant,
        "seed": seed,
        "modality": ds_cfg["modality"],
        "num_classes": ds_cfg["num_classes"],
        "timesteps": timesteps,
        "best_val_acc": round(best_val_acc, 4),
        "epochs_trained": len(history.get("train_loss", [])),
        "train_time_sec": round(train_time, 2),
        "spike_rate": global_spike.get("spike_rate", None),
        "sparsity": global_spike.get("sparsity", None),
        "per_layer_spikes": json.dumps(spike_metrics),
    }

    if not quiet:
        print(f"  Best val acc: {result['best_val_acc']:.2f}%")
        print(f"  Spike rate:   {result['spike_rate']:.4f}")
        print(f"  Sparsity:     {result['sparsity']:.4f}")
        print(f"  Time:         {result['train_time_sec']:.1f}s")

    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M7 Ablation Study Runner")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=list(DATASET_REGISTRY.keys()),
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_VARIANTS,
        default=MODEL_VARIANTS,
        help="Model variants to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use-contrastive", action="store_true", help="Enable SCL loss")
    parser.add_argument("--device", type=str, default="auto", help="cpu/cuda/mps/auto")
    parser.add_argument("--output", type=str, default="results/ablation_results.csv", help="CSV output path")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1 epoch, 1 seed, 1 model, 1 dataset, 10 train batches")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches per epoch")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Limit test batches per evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run prints")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Quick mode overrides
    if args.quick:
        args.datasets = [args.datasets[0]]
        args.models = [args.models[0]]
        args.seeds = [42]
        args.epochs = 1
        args.patience = 1
        if args.max_train_batches is None:
            args.max_train_batches = 10
        if args.max_test_batches is None:
            args.max_test_batches = 5
        print("QUICK MODE: 1 epoch, 1 seed, 1 model, 1 dataset, limited batches")

    total = len(args.datasets) * len(args.models) * len(args.seeds)
    print(f"Total experiments: {total}")
    print(f"Datasets: {args.datasets}")
    print(f"Models:   {args.models}")
    print(f"Seeds:    {args.seeds}")

    results: List[Dict[str, Any]] = []
    completed = 0

    for dataset_name in args.datasets:
        for model_variant in args.models:
            for seed in args.seeds:
                completed += 1
                print(f"\n[{completed}/{total}] Starting...")
                try:
                    row = run_single_experiment(
                        dataset_name=dataset_name,
                        model_variant=model_variant,
                        seed=seed,
                        device=device,
                        num_epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        use_contrastive=args.use_contrastive,
                        lr=args.lr,
                        quiet=args.quiet,
                        max_train_batches=args.max_train_batches,
                        max_test_batches=args.max_test_batches,
                    )
                    results.append(row)
                except Exception as exc:
                    print(f"ERROR in {dataset_name}/{model_variant}/seed{seed}: {exc}", file=sys.stderr)
                    results.append({
                        "dataset": dataset_name,
                        "model_variant": model_variant,
                        "seed": seed,
                        "error": str(exc),
                    })

                # Incremental save after every run
                if results:
                    _save_csv(results, args.output)

    print(f"\n{'='*60}")
    print(f"Done. Results saved to {args.output}")
    print(f"Successful runs: {sum(1 for r in results if 'error' not in r)} / {total}")


def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
