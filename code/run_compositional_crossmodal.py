#!/usr/bin/env python3
"""
Compositional Cross-Modal Generalization Experiment (2-digit N-MNIST <-> SHD)
============================================================================

The prior joint diagnostic established cleanly that SINGLE-digit cross-modal
alignment does NOT generalize to held-out digit CLASSES (Procrustes ~0%,
per-sample V2A collapses to chance / is seed-dependent). Digits are arbitrarily
paired across vision/audio, so there is no compositional structure to exploit at
the class level.

This is the WINNABLE COUNTERPART: test generalization to held-out COMBINATIONS
of KNOWN primitives. Every digit 0-9 is supervised as a single digit in both
modalities; the question is whether a jointly-trained encoder FACTORIZES POSITION
so that an unseen 2-digit number (e.g. "47", never trained) aligns across
modalities because 4 and 7 do. The win is possible in principle ONLY IF the
encoder composes; if it merely memorizes seen combinations, held-out numbers stay
at chance.

Architecture (factorization must EMERGE; nothing hard-codes position):
  - Map-producing backbones (VisualMapStem on a 74-wide canvas -> [B,64,15,35]
    spatial tokens, WIDER THAN TALL; AudioMapStem -> [B,230,64] temporal tokens).
  - Slot extractor: 2 learned query tokens/modality + hand-written scaled-dot-
    product cross-attention over the map tokens -> z1, z2 (d_slot=64), L2-normed.
    Orthogonal query init + (FIX 1) an ATTENTION-overlap decorrelation applied in
    phase (b) ONLY. No hard-coded halves, no positional bias.
  - Embedding: single-digit -> each slot; 2-digit -> Z = [z1 || z2] (concat
    preserves order; mean would destroy it).

Training (curriculum):
  - Phase (a) single-digit: one digit in a RANDOM canvas half; BOTH slots
    supervised to the digit via cross-modal SupCon on cat([z1,z2]); NO
    decorrelation. Single-digit prototypes (used by SET/ORDER) come from here.
  - Phase (b) 2-digit: TRAIN numbers; Z=[z1||z2]; cross-modal SupCon on whole-
    number labels + L_attn_dec; near-miss negatives (one slot shared).

Metrics (per-sample; V2A & A2V; permutation null k=200 + binomial p):
  - SET accuracy (unordered pair recovery; chance 1/55 ~ 1.82%, mixed/double split)
  - ORDER accuracy (conditional on correct SET, mixed; conditional chance 1/2)
  - Whole-number retrieval (chance 1.0% all-100 / 3.33% held-30)
  - Within-modality gate (whole-number LOO retrieval + within-SET; must clear or
    the cross-modal result is UNINTERPRETABLE).

Output: results/compositional_crossmodal/  (never overwrites prior results).

Reuses (import ONLY, never edits) the magnitude experiment's loaders/cache/stems
as reference, plus CrossModalSupConLoss / ExperimentConfig / get_device / collate
fns from m7_trimodal_run_provenance.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT))

import snntorch as snn  # noqa: E402

# Reuse (import only) the magnitude experiment's loaders/cache + provenance utils.
from run_joint_magnitude_ablation import (  # noqa: E402
    _build_modality_loaders,
    nmnist_collate_fn,
    shd_collate_fn,
)
from m7_trimodal_run_provenance import (  # noqa: E402
    CrossModalSupConLoss,
    ExperimentConfig,
    get_device,
)

# The compact caches were pickled with module='__main__' (the magnitude probe ran
# run_joint_magnitude_ablation.py as __main__). When THIS script is __main__,
# torch.load inside the reused _build_modality_loaders cannot resolve those
# classes -> alias them into __main__ so the existing 4.5 GB cache loads without
# a re-render. (Mirrors run_joint_diagnostic.py:67-74.)
import run_joint_magnitude_ablation as _mag_module  # noqa: E402
for _cn in ("CachedNMNIST", "CachedSHD", "SubsampledByClass"):
    if hasattr(_mag_module, _cn):
        setattr(sys.modules["__main__"], _cn, getattr(_mag_module, _cn))

spike_grad = snn.surrogate.atan()

RESULTS_DIR = REPO_ROOT / "results" / "compositional_crossmodal"

# ── fixed dimensions (FIX 3: 74-wide canvas; never square) ─────────────────────
CANVAS_W = 74          # 34 (left digit) + 6 buffer + 34 (right digit)
T_VIS = 25             # N-MNIST timesteps (fixed across modalities per CLAUDE.md)
T_AUD = 230            # 2 x 100-bin SHD clips + variable gap, fixed for batching
D_MODEL = 64           # token / query / slot dim
N_SLOTS = 2
TAU_ATTN = 0.1         # cross-attention temperature (sharpen -> non-uniform early)

# ── split ──────────────────────────────────────────────────────────────────────
N_HELD = 30
N_NUMBERS = 100

# ── training budget (probe-tuned; documented estimates, unknown per-epoch) ─────
PHASE_A_PER_DIGIT = 256
PHASE_A_EPOCHS = 20
PHASE_B_COMPOSITES = 8          # per train-number per epoch
PHASE_B_EPOCHS = 30
EVAL_COMPOSITES = 24           # per number (train+held) at eval time
BATCH = 64
NEAR_MISS_FLIP = True          # FIX: one-slot-shared near-miss negatives in phase (b)

PERM_K = 200
GATE_THRESHOLD = 5.0           # within-modality whole-number LOO retrieval (%), vs ~1% chance

CSV_COLUMNS = [
    "seed", "model", "gate_passed_within",
    "within_loo_visual", "within_loo_audio", "within_set_visual", "within_set_audio",
    "time_steps_visual", "time_steps_audio",
    "spike_rate_visual", "sparsity_visual", "spike_rate_audio", "sparsity_audio",
    # compositional-only slot metrics
    "set_mixed_v2a", "set_double_v2a", "set_mixed_a2v", "set_double_a2v",
    "order_v2a", "order_a2v", "n_order_v2a", "n_order_a2v",
    "set_mixed_v2a_perm", "set_mixed_a2v_perm",
    "wholenum_v2a_all100", "wholenum_v2a_held30", "wholenum_a2v_all100", "wholenum_a2v_held30",
    "wholenum_v2a_perm", "wholenum_a2v_perm",
    # holistic also reports whole-number retrieval (no slots -> no SET/ORDER)
    "attn_overlap_final",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SEED + CFG
# ═══════════════════════════════════════════════════════════════════════════════


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _build_cfg() -> ExperimentConfig:
    # Mirrors run_joint_magnitude_ablation.main() so the cache + stems line up.
    return ExperimentConfig(
        seed=42, task_order=["nmnist", "shd", "dvs"], lr=1e-3, weight_decay=1e-4,
        batch_size=32, max_epochs=50, patience=8, gradient_clip=1.0, beta=0.9, dropout=0.2,
        hidden_dim=512, shared_dim=512, num_patterns=100, hopfield_temperature=1.0,
        num_gru_layers=2, num_workers=0, time_steps=25, pin_memory=True,
        use_contrastive=True, contrastive_temperature=0.07, contrastive_weight=0.1,
        use_replay=False, replay_batch=32, replay_weight=0.3, replay_samples_per_class=50,
        device="auto", align_cross_modal=False, align_shared_dim=128, align_weight=0.5,
        align_temperature=0.1, align_use_orthogonal=False, align_orthogonal_weight=0.01,
        nmnist_train_per_class=1500,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CONSTRUCTION (from existing caches)
# ═══════════════════════════════════════════════════════════════════════════════


class CompositeBuilder:
    """Builds 1- and 2-digit visual canvases and audio clips from cached digits.

    Visual canvas is [T=25, 2, 34, W=74]: 34 tall x 74 wide (WIDER THAN TALL,
    FIX 3). Left digit occupies cols [0+offset_l : 34+offset_l]; right digit
    cols [40+offset_r : 74+offset_r]; 6-px zero buffer between. Per-digit jitter
    a few px within its half. Audio is [T=230, 700]: two 100-bin clips with a
    variable ~5-30 bin zero gap; fixed length for batching.
    """

    LEFT_X0 = 0       # left digit base offset
    RIGHT_X0 = 40     # right digit base offset (34 + 6 buffer)
    JITTER = 3        # per-digit horizontal jitter (px), kept inside its half

    def __init__(self, cached, rng: np.random.Generator):
        self.cached = cached
        self.rng = rng
        self.vis_by_digit: Dict[int, List[int]] = {c: [] for c in range(10)}
        self.aud_by_digit: Dict[int, List[int]] = {c: [] for c in range(10)}
        for i, (_f, lab) in enumerate(cached["nmnist_train"].samples):
            self.vis_by_digit[int(lab)].append(i)
        for i, (_f, lab) in enumerate(cached["shd_train"].samples):
            self.aud_by_digit[int(lab)].append(i)
        for c in range(10):
            assert len(self.vis_by_digit[c]) > 0, f"no visual samples for digit {c}"
            assert len(self.aud_by_digit[c]) > 0, f"no audio samples for digit {c}"

    # ── visual ─────────────────────────────────────────────────────────────────
    def _vis_frame(self, d: int) -> torch.Tensor:
        pool = self.vis_by_digit[d]
        idx = int(self.rng.integers(len(pool)))
        return self.cached["nmnist_train"].samples[pool[idx]][0]  # [25,2,34,34] uint8

    def visual_2digit(self, d1: int, d2: int) -> torch.Tensor:
        canvas = torch.zeros(T_VIS, 2, 34, CANVAS_W, dtype=torch.uint8)
        f1, f2 = self._vis_frame(d1), self._vis_frame(d2)
        jl = int(self.rng.integers(-self.JITTER, self.JITTER + 1))
        jr = int(self.rng.integers(-self.JITTER, self.JITTER + 1))
        xl = max(0, min(self.LEFT_X0 + jl, 34 - 1))           # keep within cols [0,34)
        xr = max(self.RIGHT_X0, min(self.RIGHT_X0 + jr, CANVAS_W - 34))
        canvas[:, :, :, xl:xl + 34] = torch.where(f1 > 0, f1, canvas[:, :, :, xl:xl + 34])
        canvas[:, :, :, xr:xr + 34] = torch.where(f2 > 0, f2, canvas[:, :, :, xr:xr + 34])
        return canvas  # [25,2,34,74]

    def visual_1digit(self, d: int, half: str) -> torch.Tensor:
        """Single digit in ONE canvas half (FIX 2: random left/right); other half empty."""
        canvas = torch.zeros(T_VIS, 2, 34, CANVAS_W, dtype=torch.uint8)
        f = self._vis_frame(d)
        base = self.LEFT_X0 if half == "left" else self.RIGHT_X0
        j = int(self.rng.integers(-self.JITTER, self.JITTER + 1))
        x0 = max(0, min(base + j, CANVAS_W - 34))
        canvas[:, :, :, x0:x0 + 34] = torch.where(f > 0, f, canvas[:, :, :, x0:x0 + 34])
        return canvas

    # ── audio ──────────────────────────────────────────────────────────────────
    def _aud_clip(self, d: int) -> torch.Tensor:
        pool = self.aud_by_digit[d]
        idx = int(self.rng.integers(len(pool)))
        return self.cached["shd_train"].samples[pool[idx]][0]  # [100,700] uint8

    def audio_2digit(self, d1: int, d2: int) -> torch.Tensor:
        clip = torch.zeros(T_AUD, 700, dtype=torch.uint8)
        c1, c2 = self._aud_clip(d1), self._aud_clip(d2)         # each [100,700]
        gap = int(self.rng.integers(5, 31))
        t1 = int(self.rng.integers(0, 6))
        clip[t1:t1 + 100] = torch.where(c1 > 0, c1, clip[t1:t1 + 100])
        t2 = t1 + 100 + gap
        if t2 + 100 <= T_AUD:
            clip[t2:t2 + 100] = torch.where(c2 > 0, c2, clip[t2:t2 + 100])
        return clip  # [230,700]

    def audio_1digit(self, d: int, half: str) -> torch.Tensor:
        """Single clip in a RANDOM temporal position; rest zero (FIX 2)."""
        clip = torch.zeros(T_AUD, 700, dtype=torch.uint8)
        c = self._aud_clip(d)
        if half == "left":      # early
            t0 = int(self.rng.integers(0, 6))
        else:                   # late
            t0 = int(self.rng.integers(T_AUD - 100 - 5, T_AUD - 100 + 1))
        clip[t0:t0 + 100] = torch.where(c > 0, c, clip[t0:t0 + 100])
        return clip


def make_split(seed: int) -> Tuple[List[int], List[int]]:
    """Deterministic HELD-30 / TRAIN-70 split over numbers 00-99 (id = d1*10+d2)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_NUMBERS)
    held = sorted(int(x) for x in perm[:N_HELD])
    train = sorted(int(x) for x in perm[N_HELD:])
    return train, held


def assert_split_coverage(train: List[int], held: List[int]) -> None:
    """Fail-loud: every digit 0-9 in pos1 AND pos2 within TRAIN and within HELD;
    no held number appears in train."""
    assert set(train).isdisjoint(set(held)), "held/train overlap!"
    assert len(train) == 70 and len(held) == 30

    def pos_coverage(nums):
        p1 = set(n // 10 for n in nums)
        p2 = set(n % 10 for n in nums)
        return p1, p2

    for name, nums in (("TRAIN", train), ("HELD", held)):
        p1, p2 = pos_coverage(nums)
        assert p1 == set(range(10)), f"{name} missing pos1 digits: {set(range(10)) - p1}"
        assert p2 == set(range(10)), f"{name} missing pos2 digits: {set(range(10)) - p2}"
    t1, _ = pos_coverage(train)
    print(f"   split OK | TRAIN pos1 coverage: {sorted(t1)} | TRAIN {len(train)} HELD {len(held)}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE (map-producing backbones + cross-attention slots)
# ═══════════════════════════════════════════════════════════════════════════════


class VisualMapStem(nn.Module):
    """N-MNIST 2-digit canvas [25,2,34,74] -> spatial tokens [B, N, 64].

    conv1(k5)+pool1+LIF1 per timestep -> time-summed s1 map [B,32,15,35]
    (74-wide canvas: conv1 34x74->30x70, pool1->15x35; WIDER THAN TALL, FIX 3),
    then conv2(k5,pad2)+ReLU refiner -> [B,64,15,35] -> flattened tokens.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)   # preserves H,W

    def forward(self, x):  # x: [B, 25, 2, 34, 74] (uint8 ok)
        x = x.float()
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        acc = None
        for t in range(T):
            xt = x[:, t]
            c1 = self.pool1(self.conv1(xt))            # [B,32,15,35]
            s1, mem1 = self.lif1(c1, mem1)
            acc = s1 if acc is None else acc + s1
        m = F.relu(self.conv2(acc))                    # [B,64,15,35]
        return m.flatten(2).transpose(1, 2)            # [B, 525, 64]


class AudioMapStem(nn.Module):
    """SHD 2-digit clip [230,700] -> temporal tokens [B, 230, 64].

    fc+LIF stack per timestep (mirrors AudioStem), projected to d_model=64, stacked.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(700, 1024)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(1024, 1024)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc3 = nn.Linear(1024, D_MODEL)
        self.lif3 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):  # x: [B, 230, 700] (uint8 ok)
        x = x.float()
        B, T = x.size(0), x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        outs = []
        for t in range(T):
            xt = x[:, t]
            s1, mem1 = self.lif1(self.fc1(xt), mem1)
            s2, mem2 = self.lif2(self.fc2(s1), mem2)
            s3, mem3 = self.lif3(self.fc3(s2), mem3)    # [B,64]
            outs.append(s3)
        return torch.stack(outs, dim=1)                 # [B, 230, 64]


class SlotExtractor(nn.Module):
    """2 learned queries + hand-written scaled-dot-product cross-attention.

    Returns z [B, n_slots, d] (each slot L2-normed) and attn [B, n_slots, N]
    (softmax weights, each sums to 1 over tokens). Orthogonal query init breaks
    query-permutation symmetry WITHOUT telling the queries where to look.
    """

    def __init__(self, n_slots: int = N_SLOTS, d_model: int = D_MODEL, tau: float = TAU_ATTN):
        super().__init__()
        self.n_slots = n_slots
        self.tau = tau
        self.queries = nn.Parameter(torch.randn(n_slots, d_model))
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self._init_orthogonal_queries()

    def _init_orthogonal_queries(self):
        with torch.no_grad():
            d = self.queries.shape[1]
            q0 = torch.randn(d)
            q0 = q0 / (q0.norm() + 1e-8)
            q1 = torch.randn(d)
            q1 = q1 - (q1 @ q0) * q0
            q1 = q1 / (q1.norm() + 1e-8)
            self.queries.copy_(torch.stack([q0, q1]))

    def forward(self, tokens):                          # tokens: [B, N, d_model]
        B = tokens.size(0)
        Q = self.wq(self.queries).unsqueeze(0).expand(B, -1, -1)   # [B, n_slots, d]
        K = self.wk(tokens)                                          # [B, N, d]
        V = self.wv(tokens)                                          # [B, N, d]
        scale = 1.0 / math.sqrt(D_MODEL)
        scores = torch.matmul(Q, K.transpose(1, 2)) * scale          # [B, n_slots, N]
        attn = torch.softmax(scores / self.tau, dim=-1)              # [B, n_slots, N]
        z = torch.matmul(attn, V)                                    # [B, n_slots, d]
        z = F.normalize(z, dim=-1)
        return z, attn


class CompositionalModel(nn.Module):
    """Map backbones + per-modality slot extractors. Z = [z1 || z2] preserves order."""

    def __init__(self):
        super().__init__()
        self.vstem = VisualMapStem()
        self.astem = AudioMapStem()
        self.vslots = SlotExtractor()
        self.aslots = SlotExtractor()

    def slots(self, x, modality):
        tok = self.vstem(x) if modality == "visual" else self.astem(x)
        ex = self.vslots if modality == "visual" else self.aslots
        return ex(tok)                                  # (z[B,2,d], attn[B,2,N])

    def embed(self, x, modality):
        z, _ = self.slots(x, modality)
        return z                                        # [B,2,d]

    def forward(self, x, modality):
        z, attn = self.slots(x, modality)
        Z = torch.cat([z[:, 0], z[:, 1]], dim=-1)       # [B, 2d]  (concat, not mean)
        return Z, attn


class HolisticModel(nn.Module):
    """Negative control: SAME map backbones -> global mean-pool -> single embedding.

    Atomic 100-way SupCon on TRAIN numbers, no slots. Expected held ~ chance
    (memorizes seen combinations only). The scaling decision keys on the
    compositional model BEATING this on held numbers.
    """

    def __init__(self, d_out: int = 128):
        super().__init__()
        self.vstem = VisualMapStem()
        self.astem = AudioMapStem()
        self.vproj = nn.Linear(D_MODEL, d_out)
        self.aproj = nn.Linear(D_MODEL, d_out)

    def embed(self, x, modality):
        tok = self.vstem(x) if modality == "visual" else self.astem(x)   # [B,N,64]
        pooled = tok.mean(dim=1)                                          # [B,64]
        z = self.vproj(pooled) if modality == "visual" else self.aproj(pooled)
        return F.normalize(z, dim=-1)

    def forward(self, x, modality):
        return self.embed(x, modality)


def attn_overlap_loss(attn: torch.Tensor) -> torch.Tensor:
    """FIX 1: attention-overlap decorrelation = mean over batch of the dot product
    of the two queries' softmax distributions over tokens. High when both queries
    attend the SAME tokens; pushes them to cover DIFFERENT regions (separating the
    two digits) without forcing their embeddings apart (so doubles are fine)."""
    a1, a2 = attn[:, 0], attn[:, 1]                     # each [B, N], sums to 1
    return (a1 * a2).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH BUILDERS (pull composites from the builder into stacked tensors)
# ═══════════════════════════════════════════════════════════════════════════════


def build_visual_batch(builder: CompositeBuilder, specs: List, one_digit: bool) -> torch.Tensor:
    """specs: list of (d,) if one_digit else list of (d1,d2). Returns [B,25,2,34,74]."""
    frames = []
    for sp in specs:
        if one_digit:
            d, half = sp
            frames.append(builder.visual_1digit(d, half))
        else:
            d1, d2 = sp
            frames.append(builder.visual_2digit(d1, d2))
    return torch.stack(frames)


def build_audio_batch(builder: CompositeBuilder, specs: List, one_digit: bool) -> torch.Tensor:
    clips = []
    for sp in specs:
        if one_digit:
            d, half = sp
            clips.append(builder.audio_1digit(d, half))
        else:
            d1, d2 = sp
            clips.append(builder.audio_2digit(d1, d2))
    return torch.stack(clips)


def _rand_half(rng: np.random.Generator) -> str:
    return "left" if rng.random() < 0.5 else "right"


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_compositional(model, builder, device, rng, train_numbers, shuffle_pairs: bool = False):
    """Curriculum: phase (a) single-digit (no decorr) -> phase (b) 2-digit (+ L_attn_dec)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE_A_EPOCHS + PHASE_B_EPOCHS)
    supcon = CrossModalSupConLoss(temperature=0.1)
    ATTN_DEC_WEIGHT = 1e-2

    # ── Phase (a): single-digit primitives ────────────────────────────────────
    for epoch in range(PHASE_A_EPOCHS):
        # manual SGD loop (re-batch each epoch)
        total, n = 0.0, 0
        specs_v, specs_a, labels = [], [], []
        for c in range(10):
            for _ in range(PHASE_A_PER_DIGIT):
                specs_v.append((c, _rand_half(rng)))
                specs_a.append((c, _rand_half(rng)))
                labels.append(c)
        perm = rng.permutation(len(labels))
        labels = np.array(labels)
        model.train()
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            lab = torch.tensor(labels[idx], device=device, dtype=torch.long)
            if shuffle_pairs:
                lab = lab[torch.randperm(lab.size(0), device=device)]
            v = build_visual_batch(builder, [specs_v[j] for j in idx], one_digit=True).to(device)
            a = build_audio_batch(builder, [specs_a[j] for j in idx], one_digit=True).to(device)
            zv = model.embed(v, "visual")
            za = model.embed(a, "audio")
            feats = torch.cat([zv[:, 0], zv[:, 1], za[:, 0], za[:, 1]], dim=0)
            lbl = torch.cat([lab, lab, lab, lab], dim=0)
            loss = supcon(feats, lbl)                  # NO decorrelation in phase (a)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        scheduler.step()
        print(f"   [phase a] epoch {epoch+1:2d}/{PHASE_A_EPOCHS} | supcon {total/max(n,1):.4f}", flush=True)

    # ── Phase (b): 2-digit TRAIN numbers + attention decorrelation ─────────────
    for epoch in range(PHASE_B_EPOCHS):
        total, n = 0.0, 0
        # build a per-epoch spec list: each train number PHASE_B_COMPOSITES times,
        # plus (near-miss) a pos2-flipped variant sharing pos1.
        specs, labels = [], []
        for num in train_numbers:
            d1, d2 = num // 10, num % 10
            for _ in range(PHASE_B_COMPOSITES):
                specs.append((d1, d2)); labels.append(num)
            if NEAR_MISS_FLIP:
                d2p = int(rng.integers(0, 10))
                specs.append((d1, d2p)); labels.append(d1 * 10 + d2p)   # shares pos1
        perm = rng.permutation(len(labels))
        labels = np.array(labels)
        model.train()
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            lab = torch.tensor(labels[idx], device=device, dtype=torch.long)
            if shuffle_pairs:
                lab_a = lab[torch.randperm(lab.size(0), device=device)]
            else:
                lab_a = lab
            v = build_visual_batch(builder, [specs[j] for j in idx], one_digit=False).to(device)
            a = build_audio_batch(builder, [specs[j] for j in idx], one_digit=False).to(device)
            Zv, av_attn = model(v, "visual")           # [B,2d], attn[B,2,N]
            Za, aa_attn = model(a, "audio")
            feats = torch.cat([Zv, Za], dim=0)
            lbl = torch.cat([lab, lab_a], dim=0)
            loss = supcon(feats, lbl) + ATTN_DEC_WEIGHT * (attn_overlap_loss(av_attn) + attn_overlap_loss(aa_attn))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        scheduler.step()
        print(f"   [phase b] epoch {epoch+1:2d}/{PHASE_B_EPOCHS} | loss {total/max(n,1):.4f}", flush=True)


def train_holistic(model, builder, device, rng, train_numbers, shuffle_pairs: bool = False):
    """Holistic: global-pool single embedding, atomic 100-way SupCon on TRAIN numbers."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE_B_EPOCHS)
    supcon = CrossModalSupConLoss(temperature=0.1)
    for epoch in range(PHASE_B_EPOCHS):
        specs, labels = [], []
        for num in train_numbers:
            d1, d2 = num // 10, num % 10
            for _ in range(PHASE_B_COMPOSITES):
                specs.append((d1, d2)); labels.append(num)
        perm = rng.permutation(len(labels))
        labels = np.array(labels)
        total, n = 0.0, 0
        model.train()
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            lab = torch.tensor(labels[idx], device=device, dtype=torch.long)
            lab_a = lab[torch.randperm(lab.size(0), device=device)] if shuffle_pairs else lab
            v = build_visual_batch(builder, [specs[j] for j in idx], one_digit=False).to(device)
            a = build_audio_batch(builder, [specs[j] for j in idx], one_digit=False).to(device)
            zv = model.embed(v, "visual")
            za = model.embed(a, "audio")
            loss = supcon(torch.cat([zv, za], dim=0), torch.cat([lab, lab_a], dim=0))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        scheduler.step()
        print(f"   [holistic] epoch {epoch+1:2d}/{PHASE_B_EPOCHS} | loss {total/max(n,1):.4f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING EXTRACTION + METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def extract_single_digit_slots(model, builder, device, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Phase-(a) single-digit slot embeddings (random half). Returns per-modality
    (labels, slots[B,2,d]) so prototypes can pool BOTH slots per digit."""
    out = {}
    for modality in ("visual", "audio"):
        labs, slots = [], []
        for c in range(10):
            for _ in range(EVAL_COMPOSITES):
                half = _rand_half(rng)
                if modality == "visual":
                    x = builder.visual_1digit(c, half).unsqueeze(0).to(device)
                else:
                    x = builder.audio_1digit(c, half).unsqueeze(0).to(device)
                z = model.embed(x, modality)[0].cpu().numpy()   # [2,d]
                slots.append(z); labs.append(c)
        out[modality] = (np.array(labs), np.stack(slots))
    return out["visual"], out["audio"]


def single_digit_prototypes(vis, aud) -> Tuple[np.ndarray, np.ndarray]:
    """Pool BOTH slots per digit -> (10, d) L2-normalized prototypes per modality.
    These anchors come from a clean phase (a); decorrelation is OFF there."""
    def proto(labs, slots):
        P = np.zeros((10, slots.shape[2]), dtype=np.float64)
        for c in range(10):
            m = labs == c
            if m.any():
                both = slots[m].reshape(-1, slots.shape[2])   # pool slot0+slot1
                p = both.mean(axis=0)
                P[c] = p / (np.linalg.norm(p) + 1e-8)
        return P
    return proto(vis[0], vis[1]), proto(aud[0], aud[1])


@torch.no_grad()
def extract_2digit(model, builder, device, numbers: List[int], rng) -> Dict:
    """Returns per-modality dict with stacked embeddings + labels.

    Compositional models (have .vslots): slots (B,2,d), Z (B,2d)=concat(z0,z1).
    Holistic models (no slots): slots=None, Z (B,d_out)=the single embedding.
    Whole-number retrieval works on Z for both; SET/ORDER (which need slots) are
    only run for compositional models.
    """
    is_comp = hasattr(model, "vslots")
    res = {}
    for modality in ("visual", "audio"):
        slot_list, z_list, labs = [], [], []
        for num in numbers:
            d1, d2 = num // 10, num % 10
            for _ in range(EVAL_COMPOSITES):
                if modality == "visual":
                    x = builder.visual_2digit(d1, d2).unsqueeze(0).to(device)
                else:
                    x = builder.audio_2digit(d1, d2).unsqueeze(0).to(device)
                z = model.embed(x, modality)[0].cpu().numpy()
                if is_comp:                              # z: [2,d]
                    slot_list.append(z)
                    z_list.append(np.concatenate([z[0], z[1]]))   # [2d]
                else:                                    # z: [d_out]
                    z_list.append(z)
                labs.append(num)
        entry = {"Z": np.stack(z_list), "labels": np.array(labs)}
        entry["slots"] = np.stack(slot_list) if is_comp else None
        res[modality] = entry
    return res


def _normalize_rows(z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    return z / n


def set_order_metrics(slots: np.ndarray, labels: np.ndarray, protos: np.ndarray) -> Dict:
    """SET (unordered pair recovery) + ORDER (conditional on correct SET, mixed).

    slots: (B,2,d); labels: (B,) whole-number; protos: (10,d) single-digit prototypes.
    """
    d = slots.shape[2]
    s0 = _normalize_rows(slots[:, 0])    # (B,d)
    s1 = _normalize_rows(slots[:, 1])
    # nearest single-digit prototype per slot
    p0 = np.argmax(s0 @ protos.T, axis=1)   # (B,)
    p1 = np.argmax(s1 @ protos.T, axis=1)
    d1 = labels // 10
    d2 = labels % 10
    mixed = d1 != d2
    double = ~mixed
    # SET correct: predicted unordered pair == true unordered pair
    set_ok = (np.sort(np.stack([p0, p1], axis=1), axis=1) == np.sort(np.stack([d1, d2], axis=1), axis=1)).all(axis=1)
    set_mixed = set_ok & mixed
    set_double = set_ok & double
    n_mixed = int(mixed.sum()); n_double = int(double.sum())
    # ORDER: among correct-SET mixed, slot0 matched pos1 (d1)?
    order_ok = (p0 == d1) & set_ok & mixed
    n_order = int((set_ok & mixed).sum())
    return {
        "set_mixed_acc": 100.0 * set_mixed.sum() / max(n_mixed, 1),
        "set_double_acc": 100.0 * set_double.sum() / max(n_double, 1),
        "order_acc": 100.0 * order_ok.sum() / max(n_order, 1),
        "n_mixed": n_mixed, "n_double": n_double, "n_order": n_order,
        "p0": p0, "p1": p1,
    }


def _set_perm_null(slots, labels, protos, k, rng) -> float:
    """Permutation null for SET-mixed: shuffle prototype->digit identity k times.
    Expected SET ~ 1/55 (mixed)."""
    d = slots.shape[2]
    s0 = _normalize_rows(slots[:, 0])
    s1 = _normalize_rows(slots[:, 1])
    idx0 = np.argmax(s0 @ protos.T, axis=1)
    idx1 = np.argmax(s1 @ protos.T, axis=1)
    d1 = labels // 10; d2 = labels % 10
    mixed = d1 != d2
    if mixed.sum() == 0:
        return float("nan")
    p0 = np.sort(np.stack([idx0, idx1], axis=1), axis=1)
    truth = np.sort(np.stack([d1, d2], axis=1), axis=1)
    accs = []
    for _ in range(k):
        shuf = rng.permutation(10)
        pred = np.sort(np.stack([shuf[idx0], shuf[idx1]], axis=1), axis=1)
        ok = (pred == truth).all(axis=1) & mixed
        accs.append(float(ok.sum() / mixed.sum()))
    return 100.0 * float(np.mean(accs))


def whole_number_retrieval(query_Z: np.ndarray, query_labels: np.ndarray,
                           proto_Z: np.ndarray, proto_labels: np.ndarray, k, rng) -> Dict:
    """Nearest opposite/own-modality number-prototype (cosine). proto_Z precomputed
    per number (one row per unique number in proto_labels). LOO handled by caller
    via excluding the query's own prototype index when same-modality."""
    qZ = _normalize_rows(query_Z.astype(np.float64))
    pZ = _normalize_rows(proto_Z.astype(np.float64))
    sims = qZ @ pZ.T                          # (B, P)
    pred = proto_labels[np.argmax(sims, axis=1)]
    acc_all = 100.0 * float((pred == query_labels).mean())

    # permutation null: shuffle prototype-number identity
    idx = np.argmax(sims, axis=1)
    accs = []
    for _ in range(k):
        shuf = rng.permutation(len(proto_labels))
        pred_s = proto_labels[shuf][idx]
        accs.append(float((pred_s == query_labels).mean()))
    return {"acc": acc_all, "perm": 100.0 * float(np.mean(accs))}


def within_modality_gate(two_digit: Dict, held: List[int], k, rng) -> Dict:
    """Within-modality whole-number LOO retrieval + within-modality SET.
    LOO: each sample's own number-prototype is recomputed excluding itself; nearest
    among all 100 number-prototypes; correct if == its number."""
    out = {}
    for modality in ("visual", "audio"):
        Z = two_digit[modality]["Z"].astype(np.float64)         # (B, Zdim)
        labs = two_digit[modality]["labels"]
        Zn = _normalize_rows(Z)
        uniq = np.unique(labs)
        # per-number mean prototype (LOO: subtract the query before comparing)
        proto_of = {}
        for num in uniq:
            proto_of[num] = Zn[labs == num].mean(axis=0)
        proto_list = np.unique(labs)
        P = np.stack([proto_of[n] for n in proto_list])
        P = _normalize_rows(P)
        # LOO retrieval
        correct = 0
        for i in range(len(labs)):
            num = labs[i]
            row = Zn[i]
            # recompute its number's proto excluding sample i
            mask = labs == num
            if mask.sum() > 1:
                plo = (Zn[mask].sum(axis=0) - row) / (mask.sum() - 1)
            else:
                plo = row   # only one sample -> degenerate, will match nothing cleanly
            plo = plo / (np.linalg.norm(plo) + 1e-8)
            P_loo = P.copy()
            j = np.where(proto_list == num)[0][0]
            P_loo[j] = plo
            pred = proto_list[np.argmax(row @ P_loo.T)]
            correct += int(pred == num)
        loo_acc = 100.0 * correct / max(len(labs), 1)
        # within-modality SET using single-digit... not available here; set to nan
        out[modality] = {"loo": loo_acc}
    return out


@torch.no_grad()
def _report_spiking_stats(model, builder, device, rng, modality) -> Tuple[int, float, float]:
    """Stem spike rate / sparsity + timestep count (CLAUDE.md non-negotiable)."""
    model.eval()
    if modality == "visual":
        x = builder.visual_2digit(1, 2).unsqueeze(0).to(device)
        stem = model.vstem
    else:
        x = builder.audio_2digit(1, 2).unsqueeze(0).to(device)
        stem = model.astem
    T = int(x.size(1))
    tokens = stem(x.float())              # [1, N, 64] (time-summed spikes)
    counts = tokens.sum(dim=-1)           # summed spike count per token
    spike_rate = float(counts.mean().item()) / max(T, 1)
    sparsity = float((counts == 0).to(torch.float32).mean().item())
    return T, spike_rate, sparsity


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-MODEL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def _model_csv_path(model_name: str, seed: int) -> Path:
    base = RESULTS_DIR / f"model_{model_name}_seed{seed}.csv"
    if base.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        base = RESULTS_DIR / f"model_{model_name}_seed{seed}__{ts}.csv"
    return base


def run_one_model(model_name: str, seed, cfg, cached, device) -> Dict:
    """model_name in {compositional, holistic, untrained, shuffled}."""
    print(f"\n{'='*70}\n  SEED {seed} | MODEL {model_name.upper()}\n{'='*70}", flush=True)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    builder = CompositeBuilder(cached, np.random.default_rng(seed + 7))
    train_numbers, held_numbers = make_split(seed)
    assert_split_coverage(train_numbers, held_numbers)

    if model_name == "compositional":
        model = CompositionalModel().to(device)
        train_compositional(model, builder, device, rng, train_numbers, shuffle_pairs=False)
    elif model_name == "shuffled":
        model = CompositionalModel().to(device)
        train_compositional(model, builder, device, rng, train_numbers, shuffle_pairs=True)
    elif model_name == "holistic":
        model = HolisticModel().to(device)
        train_holistic(model, builder, device, rng, train_numbers, shuffle_pairs=False)
    elif model_name == "untrained":
        model = CompositionalModel().to(device)   # no training
    else:
        raise ValueError(model_name)

    eval_rng = np.random.default_rng(seed + 99)
    ev = extract_2digit(model, builder, device, train_numbers + held_numbers, eval_rng)

    # split eval embeddings into train-number vs held-number rows
    labs = ev["visual"]["labels"]
    held_arr = np.array(held_numbers)
    held_mask = np.isin(labs, held_arr)

    row = {
        "seed": seed, "model": model_name, "gate_passed_within": 0,
        "within_loo_visual": float("nan"), "within_loo_audio": float("nan"),
        "within_set_visual": float("nan"), "within_set_audio": float("nan"),
        "time_steps_visual": T_VIS, "time_steps_audio": T_AUD,
        "spike_rate_visual": float("nan"), "sparsity_visual": float("nan"),
        "spike_rate_audio": float("nan"), "sparsity_audio": float("nan"),
        "set_mixed_v2a": float("nan"), "set_double_v2a": float("nan"),
        "set_mixed_a2v": float("nan"), "set_double_a2v": float("nan"),
        "order_v2a": float("nan"), "order_a2v": float("nan"),
        "n_order_v2a": 0, "n_order_a2v": 0,
        "set_mixed_v2a_perm": float("nan"), "set_mixed_a2v_perm": float("nan"),
        "wholenum_v2a_all100": float("nan"), "wholenum_v2a_held30": float("nan"),
        "wholenum_a2v_all100": float("nan"), "wholenum_a2v_held30": float("nan"),
        "wholenum_v2a_perm": float("nan"), "wholenum_a2v_perm": float("nan"),
        "attn_overlap_final": float("nan"),
    }

    # spiking stats (only meaningful for the map-backbone models; all four have stems)
    try:
        Tv, srv, spv = _report_spiking_stats(model, builder, device, rng, "visual")
        Ta, sra, spa = _report_spiking_stats(model, builder, device, rng, "audio")
        row["spike_rate_visual"], row["sparsity_visual"] = srv, spv
        row["spike_rate_audio"], row["sparsity_audio"] = sra, spa
    except Exception as e:
        print(f"   (spiking stats skipped: {e})", flush=True)

    # ── within-modality gate (whole-number LOO) ───────────────────────────────
    gate = within_modality_gate(ev, held_numbers, PERM_K, eval_rng)
    row["within_loo_visual"] = gate["visual"]["loo"]
    row["within_loo_audio"] = gate["audio"]["loo"]
    gate_pass = (gate["visual"]["loo"] > GATE_THRESHOLD) and (gate["audio"]["loo"] > GATE_THRESHOLD)
    row["gate_passed_within"] = int(gate_pass)
    print(f"   within-gate LOO (vs ~1% chance): vis {gate['visual']['loo']:.1f}% "
          f"aud {gate['audio']['loo']:.1f}% -> {'PASS' if gate_pass else 'FAIL'}", flush=True)

    # ── whole-number cross-modal retrieval (all models have Z) ─────────────────
    vZ, aZ = ev["visual"]["Z"], ev["audio"]["Z"]
    v_lab, a_lab = ev["visual"]["labels"], ev["audio"]["labels"]
    # number prototypes per modality (one row per unique number present)
    def number_protos(Z, labs):
        uniq = np.unique(labs)
        P = np.stack([Z[labs == n].mean(axis=0) for n in uniq])
        return P, uniq
    Pv, uv = number_protos(vZ, v_lab)
    Pa, ua = number_protos(aZ, a_lab)
    # V2A: query visual held/all -> nearest audio number-proto
    for tag, mask in (("all100", np.ones(len(v_lab), bool)), ("held30", held_mask)):
        if mask.sum() == 0:
            continue
        r = whole_number_retrieval(vZ[mask], v_lab[mask], Pa, ua, PERM_K, eval_rng)
        row[f"wholenum_v2a_{tag}"] = r["acc"]
        if tag == "all100":
            row["wholenum_v2a_perm"] = r["perm"]
    for tag, mask in (("all100", np.ones(len(a_lab), bool)), ("held30", held_mask)):
        if mask.sum() == 0:
            continue
        r = whole_number_retrieval(aZ[mask], a_lab[mask], Pv, uv, PERM_K, eval_rng)
        row[f"wholenum_a2v_{tag}"] = r["acc"]
        if tag == "all100":
            row["wholenum_a2v_perm"] = r["perm"]

    # ── SET / ORDER (compositional + shuffled + untrained only; holistic has no slots) ──
    if model_name in ("compositional", "shuffled", "untrained"):
        vs, as_ = extract_single_digit_slots(model, builder, device, eval_rng)
        Pv1, Pa1 = single_digit_prototypes(vs, as_)     # (10,d) single-digit prototypes
        vh = {m: (ev[m]["slots"][held_mask], ev[m]["labels"][held_mask]) for m in ("visual", "audio")}
        v2a = set_order_metrics(vh["visual"][0], vh["visual"][1], Pa1)   # visual query vs audio protos
        a2v = set_order_metrics(vh["audio"][0], vh["audio"][1], Pv1)
        row["set_mixed_v2a"] = v2a["set_mixed_acc"]; row["set_double_v2a"] = v2a["set_double_acc"]
        row["set_mixed_a2v"] = a2v["set_mixed_acc"]; row["set_double_a2v"] = a2v["set_double_acc"]
        row["order_v2a"] = v2a["order_acc"]; row["order_a2v"] = a2v["order_acc"]
        row["n_order_v2a"] = v2a["n_order"]; row["n_order_a2v"] = a2v["n_order"]
        row["set_mixed_v2a_perm"] = _set_perm_null(vh["visual"][0], vh["visual"][1], Pa1, PERM_K, eval_rng)
        row["set_mixed_a2v_perm"] = _set_perm_null(vh["audio"][0], vh["audio"][1], Pv1, PERM_K, eval_rng)
        # final attention overlap diagnostic
        with torch.no_grad():
            xv = builder.visual_2digit(3, 7).unsqueeze(0).to(device)
            _, av = model(xv, "visual")
            row["attn_overlap_final"] = float(attn_overlap_loss(av).item())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_csv_path(model_name, seed)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerow(row)
    print(f"   Wrote {path.name} | SET-mix V2A {row['set_mixed_v2a']:.1f}% "
          f"(perm {row['set_mixed_v2a_perm']:.1f}%) | whole V2A held {row['wholenum_v2a_held30']:.1f}%", flush=True)
    return row


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION + REPORT + VERDICT
# ═══════════════════════════════════════════════════════════════════════════════


def load_rows_from_disk() -> List[Dict]:
    rows = []
    for p in sorted(RESULTS_DIR.glob("model_*_seed*.csv")):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                for k in CSV_COLUMNS:
                    if k == "model":
                        continue
                    try:
                        r[k] = int(float(r[k])) if k in ("seed", "n_order_v2a", "n_order_a2v") else float(r[k])
                    except (ValueError, KeyError):
                        pass
                rows.append(r)
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.std(vals)) if vals else float("nan")


def _binom_z(p_hat, n, p0):
    se = (p0 * (1.0 - p0) / max(n, 1)) ** 0.5
    return (p_hat - p0) / (se + 1e-12)


def write_summary_and_report(rows: List[Dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "mean", "std", "n_seeds"])
        for m in ("compositional", "holistic", "shuffled", "untrained"):
            sub = [r for r in rows if r["model"] == m]
            for key in CSV_COLUMNS:
                if key in ("seed", "model"):
                    continue
                vals = [r[key] for r in sub if key in r and r[key] is not None
                        and not (isinstance(r[key], float) and np.isnan(r[key]))]
                if vals:
                    w.writerow([m, key, f"{_mean(vals):.4f}", f"{_std(vals):.4f}", len(vals)])

    comp = [r for r in rows if r["model"] == "compositional" and r.get("gate_passed_within")]
    hol = [r for r in rows if r["model"] == "holistic"]
    comp_all = [r for r in rows if r["model"] == "compositional"]

    def gv(rows_, key):
        return _mean([r.get(key, float("nan")) for r in rows_])

    # pre-registered verdict (data-driven; hypotheses are not results)
    verdict = []
    if not comp:
        verdict.append("No gate-passing compositional seed yet; inspect within-gate + per-seed numbers.")
    else:
        set_v2a = gv(comp, "set_mixed_v2a"); set_perm = gv(comp, "set_mixed_v2a_perm")
        order = gv(comp, "order_v2a"); wn_held = gv(comp, "wholenum_v2a_held30")
        asym = abs(gv(comp, "set_mixed_v2a") - gv(comp, "set_mixed_a2v"))
        hol_held = gv(hol, "wholenum_v2a_held30")
        beats_holistic = (not np.isnan(wn_held)) and (not np.isnan(hol_held)) and wn_held > hol_held
        clears_null = (not np.isnan(set_v2a)) and (not np.isnan(set_perm)) and set_v2a > set_perm + 1.0
        if clears_null and order > 60 and wn_held > 2.0 and asym < 5 and beats_holistic:
            verdict.append(f"FACTORIZES: held SET-mix V2A {set_v2a:.1f}% >> perm {set_perm:.1f}%, ORDER {order:.1f}%, "
                           f"whole-num held {wn_held:.1f}% (>holistic {hol_held:.1f}%), symmetric -> compositional generalization. "
                           f"[n={len(comp)} seed(s); NEED >=3 to claim.]")
        elif clears_null and order < 60:
            verdict.append(f"PARTIAL: SET-mix V2A {set_v2a:.1f}% >> perm {set_perm:.1f}% but ORDER {order:.1f}% ~ chance -> "
                           f"composes CONTENT not ORDER.")
        elif not beats_holistic and not np.isnan(hol_held):
            verdict.append(f"MEMORIZES: compositional ~ holistic on held (comp whole {wn_held:.1f}% vs holistic {hol_held:.1f}%) -> no compositional gain.")
        else:
            verdict.append(f"INCONCLUSIVE at n={len(comp)}: SET-mix {set_v2a:.1f}% (perm {set_perm:.1f}%), ORDER {order:.1f}%, whole held {wn_held:.1f}%.")
    gate_failed = [r for r in comp_all if not r.get("gate_passed_within")]
    if gate_failed and not comp:
        verdict.append("WITHIN-GATE FAILED on all seeds: backbone did not factorize -> cross-modal uninterpretable, STOP.")

    with open(RESULTS_DIR / "report.txt", "w") as f:
        f.write("COMPOSITIONAL CROSS-MODAL GENERALIZATION (2-digit N-MNIST <-> SHD)\n")
        f.write("=" * 78 + "\n\n")
        f.write("PRE-REGISTERED:\n")
        f.write("- Headline: held-30 SET accuracy (chance 1/55 ~ 1.82%, mixed/double split).\n")
        f.write("- Gate: within-modality whole-number LOO retrieval > 5% in BOTH modalities (chance ~1%).\n")
        f.write("- Scaling: expand to [42,123,456,...] ONLY IF compositional held-30 SET clears its perm\n"
                "  null AND beats the holistic baseline. No 'it generalizes' claim until multi-seed.\n\n")
        f.write("PER-MODEL TABLE (gate-passing only for compositional):\n\n")
        for m in ("compositional", "holistic", "shuffled", "untrained"):
            sub = comp if m == "compositional" else [r for r in rows if r["model"] == m]
            if not sub:
                continue
            f.write(f"[{m.upper()}] (n={len(sub)})\n")
            for r in sub:
                f.write(f"    seed {int(r['seed'])}: within-LOO vis {r.get('within_loo_visual',float('nan')):.1f}% "
                        f"aud {r.get('within_loo_audio',float('nan')):.1f}% | "
                        f"SET-mix V2A {r.get('set_mixed_v2a',float('nan')):.1f}% (perm {r.get('set_mixed_v2a_perm',float('nan')):.1f}%) "
                        f"A2V {r.get('set_mixed_a2v',float('nan')):.1f}% | ORDER V2A {r.get('order_v2a',float('nan')):.1f}% | "
                        f"whole held V2A {r.get('wholenum_v2a_held30',float('nan')):.1f}% | "
                        f"gate {'Y' if r.get('gate_passed_within') else 'N'}\n")
        f.write("\nVERDICT (data-driven; hypotheses are not results):\n")
        for i, v in enumerate(verdict, 1):
            f.write(f"  {i}. {v}\n")

    print(f"\nWrote {RESULTS_DIR/'summary.csv'} and {RESULTS_DIR/'report.txt'}", flush=True)
    print("\nVERDICT:")
    for v in verdict:
        print(f"  - {v}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SMOKE SUITE (incl. FIX 3 left/right + corrected map-shape)
# ═══════════════════════════════════════════════════════════════════════════════


def run_smoke():
    print("SMOKE: shapes, orthogonal init, left/right position, SET sanity, split asserts", flush=True)
    set_seed(42)
    rng = np.random.default_rng(42)
    device = get_device(_build_cfg())
    print(f"Device: {device}", flush=True)

    # dummy digit pools over the real caches (need cached)
    print("Building modality loaders (reuses cache)...", flush=True)
    loaders, cached = _build_modality_loaders(_build_cfg())
    builder = CompositeBuilder(cached, np.random.default_rng(42))

    # ── 2-digit tensor shapes ──────────────────────────────────────────────────
    v2 = builder.visual_2digit(1, 2)
    a2 = builder.audio_2digit(1, 2)
    assert v2.shape == (T_VIS, 2, 34, CANVAS_W), f"visual 2-digit shape {v2.shape} != (25,2,34,74)"
    assert a2.shape == (T_AUD, 700), f"audio 2-digit shape {a2.shape} != (230,700)"
    print(f"   shapes OK: visual {tuple(v2.shape)} audio {tuple(a2.shape)}", flush=True)

    # ── backbone map shapes (FIX 3: wider than tall) ───────────────────────────
    vstem = VisualMapStem().to(device)
    astem = AudioMapStem().to(device)
    tok_v = vstem(v2.unsqueeze(0).to(device))      # [1, N, 64]
    tok_a = astem(a2.unsqueeze(0).to(device))      # [1, 230, 64]
    assert tok_v.shape == (1, 15 * 35, 64), f"visual tokens {tok_v.shape} != (1,525,64) -> map not 15x35"
    assert tok_a.shape == (1, T_AUD, 64), f"audio tokens {tok_v.shape}"
    # recover spatial map, confirm WIDER THAN TALL
    print(f"   map OK: visual tokens {tuple(tok_v.shape)} = 15x35 spatial (width 35 > height 15, ~2.3x)", flush=True)

    # ── orthogonal query init: dot < 0.9 ───────────────────────────────────────
    se = SlotExtractor().to(device)
    q0 = se.queries[0] / se.queries[0].norm()
    q1 = se.queries[1] / se.queries[1].norm()
    dot = float((q0 @ q1).item())
    assert abs(dot) < 0.9, f"orthogonal init dot {dot} >= 0.9"
    z, attn = se(tok_v)
    assert z.shape == (1, N_SLOTS, D_MODEL) and attn.shape == (1, N_SLOTS, 15 * 35)
    print(f"   slot extractor OK: z {tuple(z.shape)} attn {tuple(attn.shape)} | query dot {dot:.3f}", flush=True)

    # ── FIX 3: left/right position preservation (digit only in one half) ───────
    # build a canvas with a digit ONLY in the LEFT half -> activation mass should
    # concentrate in the LEFT columns of the feature map (pre-training).
    f1 = builder._vis_frame(3)              # [25,2,34,34]
    canvas_l = torch.zeros(T_VIS, 2, 34, CANVAS_W, dtype=torch.uint8)
    canvas_l[:, :, :, 0:34] = f1
    canvas_r = torch.zeros(T_VIS, 2, 34, CANVAS_W, dtype=torch.uint8)
    canvas_r[:, :, :, 40:74] = f1
    with torch.no_grad():
        tl = vstem(canvas_l.unsqueeze(0).to(device)).view(1, 15, 35, 64).sum(-1)[0]  # [15,35]
        tr = vstem(canvas_r.unsqueeze(0).to(device)).view(1, 15, 35, 64).sum(-1)[0]
    left_mass_l = float(tl[:, :17].sum().item()); right_mass_l = float(tl[:, 17:].sum().item())
    left_mass_r = float(tr[:, :17].sum().item()); right_mass_r = float(tr[:, 17:].sum().item())
    frac_l = left_mass_l / (left_mass_l + right_mass_l + 1e-8)
    frac_r = left_mass_r / (left_mass_r + right_mass_r + 1e-8)
    assert frac_l > 0.6, f"left-only digit not left-concentrated (frac_left {frac_l:.2f})"
    assert frac_r < 0.4, f"right-only digit not right-concentrated (frac_left {frac_r:.2f})"
    print(f"   left/right OK: left-only digit left-frac {frac_l:.2f}, right-only {frac_r:.2f}", flush=True)

    # ── SET metric sanity: synthetic aligned -> ~100%, random -> ~1.8% ─────────
    d = D_MODEL
    protos = _normalize_rows(np.random.default_rng(0).standard_normal((10, d)))
    # aligned: slots exactly equal prototype of the true digits
    N = 200
    d1 = np.random.default_rng(1).integers(0, 10, N); d2 = np.random.default_rng(2).integers(0, 10, N)
    labs = d1 * 10 + d2
    slots = np.stack([protos[d1], protos[d2]], axis=1)   # (N,2,d) perfectly aligned
    m_aligned = set_order_metrics(slots, labs, protos)
    slots_rand = _normalize_rows(np.random.default_rng(3).standard_normal((N, 2, d)))
    m_rand = set_order_metrics(slots_rand, labs, protos)
    assert m_aligned["set_mixed_acc"] > 95, f"aligned SET {m_aligned['set_mixed_acc']:.1f}% not ~100"
    assert m_rand["set_mixed_acc"] < 8, f"random SET {m_rand['set_mixed_acc']:.1f}% not ~chance"
    print(f"   SET sanity OK: aligned {m_aligned['set_mixed_acc']:.1f}% random {m_rand['set_mixed_acc']:.1f}% (chance ~1.8%)", flush=True)

    # ── split asserts + per-position coverage ──────────────────────────────────
    train, held = make_split(42)
    assert_split_coverage(train, held)

    print("\n✅ SMOKE PASSED.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Compositional cross-modal generalization (2-digit)")
    parser.add_argument("--probe", action="store_true", help="seed 42 only, full model panel")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override seed list")
    parser.add_argument("--models", type=str, default="compositional,holistic,shuffled,untrained",
                        help="comma-separated model names to run")
    parser.add_argument("--aggregate-only", action="store_true", help="skip training; rebuild report from disk")
    parser.add_argument("--smoke", action="store_true", help="run the smoke suite and exit")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        run_smoke()
        return

    if args.aggregate_only:
        rows = load_rows_from_disk()
        print(f"Loaded {len(rows)} model CSV(s) from {RESULTS_DIR}", flush=True)
        if rows:
            write_summary_and_report(rows)
        print("\n✅ Done (aggregate-only).", flush=True)
        return

    cfg = _build_cfg()
    device = get_device(cfg)
    print(f"Device: {device}", flush=True)

    seeds = [42] if args.probe else (args.seeds or [42, 123, 456])
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"\nMode: seeds={seeds} models={models}", flush=True)

    print("\nBuilding modality loaders (reuses joint_magnitude_ablation/cache/)...", flush=True)
    loaders, cached = _build_modality_loaders(cfg)

    ran = 0
    for model_name in models:
        for seed in seeds:
            existing = list(RESULTS_DIR.glob(f"model_{model_name}_seed{seed}.csv"))
            if existing:
                print(f"   (skip {model_name}/seed{seed}: CSV exists)", flush=True)
                continue
            try:
                run_one_model(model_name, seed, cfg, cached, device)
                ran += 1
            except Exception as e:
                import traceback
                print(f"\n❌ {model_name}/seed{seed} failed: {e}", flush=True)
                traceback.print_exc()

    # JSON sidecar: cfg + git SHA + exact train/held split list
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        sha = "unknown"
    sidecar = {
        "cfg": {"lr": 1e-3, "phase_a_epochs": PHASE_A_EPOCHS, "phase_b_epochs": PHASE_B_EPOCHS,
                "phase_a_per_digit": PHASE_A_PER_DIGIT, "phase_b_composites": PHASE_B_COMPOSITES,
                "eval_composites": EVAL_COMPOSITES, "batch": BATCH, "d_model": D_MODEL,
                "n_slots": N_SLOTS, "tau_attn": TAU_ATTN, "canvas_w": CANVAS_W, "t_vis": T_VIS, "t_aud": T_AUD,
                "perm_k": PERM_K, "gate_threshold": GATE_THRESHOLD},
        "git_sha": sha, "seeds": seeds, "models": models,
        "split_seed42": {"train": make_split(42)[0], "held": make_split(42)[1]},
    }
    with open(RESULTS_DIR / "config_sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    rows = load_rows_from_disk()
    if rows:
        write_summary_and_report(rows)
    print(f"\n✅ Done (ran {ran} new model/seed(s); aggregated {len(rows)} from disk).", flush=True)


if __name__ == "__main__":
    main()
