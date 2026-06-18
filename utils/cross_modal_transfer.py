"""
Cross-Modal Transfer Test — Paper 1 Engram Protocol (shared-subspace edition)

With the model frozen:
1. Extract shared-subspace embeddings for N-MNIST (visual) and SHD (audio, mapped to 0–9).
2. Train a logistic-regression linear probe on VISUAL shared embeddings;
   freeze it; evaluate on AUDIO shared embeddings → Visual→Audio transfer.
3. Repeat in reverse → Audio→Visual transfer.
4. Compute cross-modal cosine alignment between per-class prototypes.
5. Compute silhouette scores per modality on the shared embeddings.

Returns:
    {
        "v2a": Visual→Audio transfer accuracy (0-1),
        "a2v": Audio→Visual transfer accuracy (0-1),
        "prototype_alignment": mean diagonal cosine similarity of class prototypes,
        "prototype_alignment_off_diag": mean off-diagonal cosine similarity,
        "sil_visual": silhouette score on visual shared embeddings,
        "sil_audio": silhouette score on audio shared embeddings,
        "chance": 1/num_classes,
    }

Acceptance:
    - Without alignment: V2A ~ chance (~0.10), prototype_alignment ~ 0
    - With alignment: V2A >> chance, prototype_alignment > 0
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Reuse SHD label mapping without circular import
def _shd_to_identity(shd_label: int) -> int:
    if not (0 <= shd_label <= 19):
        raise ValueError(f"SHD label must be in [0, 19], got {shd_label}")
    return int(shd_label % 10)


def extract_shared_embeddings(
    model,
    loader,
    modality: str,
    task_name: str,
    device,
    max_samples: int | None = None,
    map_labels=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate a DataLoader and extract shared-subspace embeddings.

    Args:
        model: M7_ContinualAdaptiveModel (or any with forward_with_shared)
        loader: PyTorch DataLoader
        modality: e.g. "nmnist" or "shd"
        task_name: e.g. "nmnist" or "shd"
        device: torch device
        max_samples: cap total samples (None = all)
        map_labels: callable to remap labels (e.g. _shd_to_identity for SHD)

    Returns:
        (embeddings_array, labels_array) as numpy arrays
    """
    model.eval()
    all_embs, all_lbls = [], []
    n = 0
    with torch.no_grad():
        for data, labels in tqdm(loader, desc=f"  Extract {modality}", leave=False):
            data = data.to(device)
            # forward_with_shared returns (logits, gate, full, shared)
            _, _, _, shared = model.forward_with_shared(data, modality, task_name)
            shared = shared.cpu().numpy()

            if map_labels is not None:
                labels = torch.tensor([map_labels(int(l)) for l in labels])

            labels = labels.numpy()

            all_embs.append(shared)
            all_lbls.append(labels)
            n += shared.shape[0]
            if max_samples is not None and n >= max_samples:
                break

    embeddings = np.vstack(all_embs)
    labels = np.concatenate(all_lbls)
    if max_samples is not None:
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]
    return embeddings, labels


def evaluate_cross_modal_transfer(
    model,
    visual_loader,
    audio_loader,
    device,
    num_classes: int = 10,
    samples_per_class: int = 100,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Run the full Paper-1 engram protocol on the shared subspace.

    Args:
        model: trained M7_ContinualAdaptiveModel
        visual_loader: N-MNIST test DataLoader
        audio_loader: SHD test DataLoader
        device: torch device
        num_classes: digit classes (default 10)
        samples_per_class: max samples to keep per class for analysis
        random_state: sklearn RNG seed

    Returns:
        dict with V2A, A2V, prototype_alignment, sil_visual, sil_audio, chance
    """
    print("\n" + "=" * 70)
    print("🔗 CROSS-MODAL TRANSFER TEST (Paper 1 Engram Protocol)")
    print("=" * 70)

    # ── 1. Extract shared embeddings ───────────────────────────────────────
    print("\n1. Extracting shared-subspace embeddings...")
    vis_emb, vis_lbl = extract_shared_embeddings(
        model, visual_loader, "nmnist", "nmnist", device, map_labels=None
    )
    aud_emb, aud_lbl = extract_shared_embeddings(
        model, audio_loader, "shd", "shd", device, map_labels=_shd_to_identity
    )

    # Stratify to balanced per-class counts
    def _balance(embs, lbls, spc):
        out_embs, out_lbls = [], []
        for c in range(num_classes):
            mask = lbls == c
            idx = np.where(mask)[0]
            if len(idx) > spc:
                rng = np.random.default_rng(random_state + c)
                idx = rng.choice(idx, size=spc, replace=False)
            out_embs.append(embs[idx])
            out_lbls.append(lbls[idx])
        return np.vstack(out_embs), np.concatenate(out_lbls)

    vis_emb, vis_lbl = _balance(vis_emb, vis_lbl, samples_per_class)
    aud_emb, aud_lbl = _balance(aud_emb, aud_lbl, samples_per_class)

    print(f"   Visual shared:   {vis_emb.shape}")
    print(f"   Audio shared:    {aud_emb.shape}")

    # ── 2. Linear-probe transfer ───────────────────────────────────────────
    print("\n2. Linear-probe cross-modal transfer...")

    # V2A: train on visual, test on audio
    clf_v2a = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_v2a.fit(vis_emb, vis_lbl)
    v2a_acc = clf_v2a.score(aud_emb, aud_lbl)

    # A2V: train on audio, test on visual
    clf_a2v = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_a2v.fit(aud_emb, aud_lbl)
    a2v_acc = clf_a2v.score(vis_emb, vis_lbl)

    chance = 1.0 / num_classes
    print(f"   Visual → Audio:  {v2a_acc * 100:.2f}%  (chance={chance * 100:.1f}%)")
    print(f"   Audio → Visual:  {a2v_acc * 100:.2f}%  (chance={chance * 100:.1f}%)")

    # ── 3. Prototype cosine alignment ──────────────────────────────────────
    print("\n3. Per-class prototype alignment...")
    prototypes_v = np.array([vis_emb[vis_lbl == c].mean(axis=0) for c in range(num_classes)])
    prototypes_a = np.array([aud_emb[aud_lbl == c].mean(axis=0) for c in range(num_classes)])

    # L2-normalise prototypes before cosine
    prototypes_v = prototypes_v / (np.linalg.norm(prototypes_v, axis=1, keepdims=True) + 1e-8)
    prototypes_a = prototypes_a / (np.linalg.norm(prototypes_a, axis=1, keepdims=True) + 1e-8)

    alignment_matrix = cosine_similarity(prototypes_v, prototypes_a)
    same_class_sim = float(alignment_matrix.diagonal().mean())
    off_diag_mask = ~np.eye(num_classes, dtype=bool)
    diff_class_sim = float(alignment_matrix[off_diag_mask].mean())

    print(f"   Same-class cos:     {same_class_sim:.4f}")
    print(f"   Off-diagonal cos:   {diff_class_sim:.4f}")
    print(f"   Alignment ratio:    {same_class_sim / (diff_class_sim + 1e-8):.2f}x")

    # ── 4. Silhouette scores ───────────────────────────────────────────────
    print("\n4. Silhouette scores (shared subspace)...")
    sil_vis = silhouette_score(vis_emb, vis_lbl) if len(np.unique(vis_lbl)) > 1 else 0.0
    sil_aud = silhouette_score(aud_emb, aud_lbl) if len(np.unique(aud_lbl)) > 1 else 0.0
    print(f"   Visual:  {sil_vis:.4f}")
    print(f"   Audio:   {sil_aud:.4f}")

    # ── 5. Sanity assertions ───────────────────────────────────────────────
    print("\n5. Sanity checks...")
    if v2a_acc < chance + 0.05:
        print(f"   ⚠️  V2A near chance ({v2a_acc * 100:.1f}%): no cross-modal alignment detected")
    else:
        print(f"   ✅ V2A above chance: {v2a_acc * 100:.1f}%")

    if same_class_sim < 0.05:
        print(f"   ⚠️  Prototype alignment near zero: shared space is not aligned")
    else:
        print(f"   ✅ Prototype alignment: {same_class_sim:.3f}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋 CROSS-MODAL TRANSFER SUMMARY")
    print("=" * 70)
    print(f"   V2A transfer:       {v2a_acc * 100:.2f}%")
    print(f"   A2V transfer:       {a2v_acc * 100:.2f}%")
    print(f"   Prototype align:    {same_class_sim:.4f}")
    print(f"   Silhouette (vis):   {sil_vis:.4f}")
    print(f"   Silhouette (aud):   {sil_aud:.4f}")
    print(f"   Chance:             {chance * 100:.1f}%")

    return {
        "v2a": float(v2a_acc),
        "a2v": float(a2v_acc),
        "prototype_alignment": same_class_sim,
        "prototype_alignment_off_diag": diff_class_sim,
        "sil_visual": float(sil_vis),
        "sil_audio": float(sil_aud),
        "chance": float(chance),
    }
