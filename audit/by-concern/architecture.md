# Architecture & duplication

_Generated 2026-07-15. 30 canonical findings in categories: architecture, duplication. Duplicates omitted (see `findings.jsonl`)._

Severity: HIGH 3, MED 16, LOW 10, INFO 1


## HIGH (3)

#### `PROV-002` [HIGH] Default forward path silently routes features through proj_shared/proj_private (512->128+384->512), diverging from the documented architecture and from the orphan copy
`code/m7_trimodal_run_provenance.py:362-399` · architecture · status:live · effort:medium · confidence:high
- **What:** forward() (401-407) always calls forward_with_shared() (372-399), which always calls _split_shared_private() (394): shared=proj_shared(features) (364), private=proj_private(features) (368), full=cat([shared,private]) (369). The memory stack then consumes full, not the raw backbone features (gate=adaptive_gate(full) 395, hop=hopfield(full) 396, hgrn=hgrn(full) 397). proj_shared and proj_private are independent trainable nn.Linear layers, so full = W*features + b is an EXTRA learned 512->512 reparametrisation inserted before Hopfield/HGRN/gate. This projection is documented (forward_with_shared docstring, 372-376) as used only 'during cross-modal alignment training', yet it is active on every forward pass even with align_cross_modal=False (the default, 1321). The orphan copy code/m7_trimodal_run.py feeds RAW features to the same modules (m7_trimodal_run.py:338-340), and CLAUDE.md documents the pipeline as backbone->Hopfield||HGRN->gate with no shared/private split. So the shared hub runs a different, higher-capacity model than both the docs and the nominal reproducibility entry point, changing numerics.
- **Evidence:** self.proj_shared=nn.Linear(cfg.shared_dim, cfg.align_shared_dim) built unconditionally (337); _split_shared_private returns full=cat([shared,private]) (369); forward_with_shared uses gate/hop/hgrn on full (395-398); forward delegates to forward_with_shared (406). Contrast orphan m7_trimodal_run.py:338-340 which uses raw features.
- **Fix:** Only apply the shared/private projection when cfg.align_cross_modal (or an explicit flag) is set; otherwise feed raw backbone features to gate/hopfield/hgrn so the default model matches the documented architecture and the orphan. If the projection is intended to be permanent, document it in CLAUDE.md/EXPERIMENTAL_MANIFEST and reconcile the orphan copy. Any change here moves numbers: call it out with before/after.
- **Fix at:** `code/m7_trimodal_run_provenance.py:362-399`, `code/m7_trimodal_run_provenance.py:336-341`
- _related: PROV-003, PROV-007_

#### `PROV-007` [HIGH] Two divergent copies of the model+trainer share class/function names and output filenames but differ in model math, metrics, CSV schema, freezing, and import-safety
`code/m7_trimodal_run_provenance.py:1-1379` · duplication · status:live · effort:large · confidence:high
- **What:** m7_trimodal_run_provenance.py (the shared hub, referenced by 19 files) and code/m7_trimodal_run.py (the orphan, imported by nobody) define the SAME public names — M7_ContinualAdaptiveModel, ContinualLearner, EmbeddingReplayBuffer, SNN_Backbone, run_one_seed, TASKS, collate fns — yet have DIVERGED substantially: (1) provenance inserts a proj_shared/proj_private reparametrisation into the forward path, the orphan does not (PROV-002); (2) provenance freezes completed backbones/heads via _freeze_completed_tasks (550-568), the orphan freezes nothing so it trains all params every task; (3) provenance defines 'fwt' as zero_shot - random_init (1066) while the orphan defines 'fwt' as after - zero_shot (m7_trimodal_run.py:616) — a different, mislabeled quantity (PROV-008); (4) different CSV schema and filename collision (PROV-006); (5) provenance is import-safe with a __main__ guard, the orphan runs a 4-5h job at import. Both write ./ICONS_M7/results/cl_trimodal_results.csv and identical checkpoint names (final_{task}_{task}_seed{seed}_{condition}.pt), so their artifacts are mutually confusable while their numerics differ. Edits to one will not propagate to the other.
- **Evidence:** Shared names across both files; _freeze_completed_tasks present only in provenance (550-568); fwt formulas differ (1066 vs m7_trimodal_run.py:616); identical checkpoint naming (1) prov 992/1041/1093 (2) orphan 536/562/589; __main__ guard prov 1378 vs orphan top-level run 707-711.
- **Fix:** Collapse to a single source of truth: keep the provenance hub, delete or clearly quarantine the orphan (it is imported by nobody), or extract the shared model/trainer into one module both import. At minimum stop them writing identical filenames.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1-1379`, `code/m7_trimodal_run.py:1-720`
- _related: PROV-002, PROV-005, PROV-006, PROV-008_

#### `JNT-005` [HIGH] CachedNMNIST/CachedSHD pickled as __main__ couples the cache to run-as-main and forces alias hacks downstream
`code/run_joint_magnitude_ablation.py:85-129` · architecture · status:live · effort:medium · confidence:high
- **What:** Dataset classes defined in this run-as-__main__ script are pickled whole via torch.save; any other process must recreate them under __main__ to torch.load the cache. This coupling has already forced identical alias hacks in two downstream scripts. weights_only=False is also an unpickle code-exec surface.
- **Evidence:** CachedNMNIST/CachedSHD (85,105) and SubsampledByClass (132) are saved with torch.save(cache_obj, path) (184-185,197-198) and reloaded with torch.load(..., weights_only=False) (172-173,190-191). Pickled under 'python run_joint_magnitude_ablation.py' the class path is __main__.CachedNMNIST. Confirmed downstream aliasing: run_compositional_crossmodal.py:92-95 and run_joint_diagnostic.py:71-74 both do 'import run_joint_magnitude_ablation as _mag_module; for _cn in (...): setattr(sys.modules["__main__"], _cn, getattr(_mag_module,_cn))' with comments blaming this exact pickling.
- **Fix:** Move CachedNMNIST/CachedSHD/SubsampledByClass into an importable module (e.g. code/joint_datasets.py) so the pickle path is stable and importers need no __main__ aliasing; cache tensor state rather than whole Dataset objects and prefer weights_only=True on a tensor-only cache.
- **Fix at:** `code/run_joint_magnitude_ablation.py:85-129`, `code/run_joint_magnitude_ablation.py:172-198`, `code/run_compositional_crossmodal.py:92-95`, `code/run_joint_diagnostic.py:71-74`


## MED (16)

#### `ARCH-011` [MED] ConvFrontend class copy-pasted across 5 files (4 archived + 1 live)
`code/archive/verify_full.py:26-46` · duplication · status:archive · effort:medium · confidence:high
- **What:** An identical ConvFrontend nn.Module (conv1 2->32 k5, pool, LIF beta=0.9; conv2 32->64 k5, pool, LIF; flatten; sum over time) is duplicated verbatim in five places, including the LIVE code/final_evaluation.py. Five divergent copies of the visual backbone frontend are a maintenance hazard: a numerics fix to one (e.g. reset semantics, beta) will silently not propagate to the others.
- **Evidence:** verify_full.py:26-46; verify_simulators.py:27-47; verify_brian2_raw.py:30-50; debug_brian2_exact.py:19-39; code/final_evaluation.py:33-53 (LIVE). All byte-identical bodies.
- **Fix:** Extract ONE ConvFrontend into a shared, import-safe module (e.g. code/spinnaker_frontend.py) and import it in final_evaluation.py; retire the four archived copies with the files. Verify the extraction is numerically inert (same beta, same reset_hidden order) before touching the live final_evaluation.py.
- **Fix at:** `code/archive/verify_full.py:26-46`, `code/archive/verify_simulators.py:27-47`, `code/archive/verify_brian2_raw.py:30-50`, `code/archive/debug_brian2_exact.py:19-39`, `code/final_evaluation.py:33-53`
- _related: ARCH-003, ARCH-004, ARCH-005, ARCH-007_

#### `DEAD-004` [MED] m7_mps.py has drifted from its generator m7_raw.py and is no longer reproducible by convert_mps.py
`code/m7_mps.py:2-23` · architecture · status:dead · effort:trivial · confidence:high
- **What:** m7_mps.py is meant to be a pure machine translation of m7_raw.py via convert_mps.py, but the two have diverged: the !pip lines are COMMENTED in m7_mps.py (L2-4/L17-23) yet UNCOMMENTED in m7_raw.py, and convert_mps.py never edits pip lines - so the current m7_mps.py cannot have been produced by running convert_mps.py on the current m7_raw.py. Regenerating is also unsafe because convert_mps.py's exact-string targets are themselves stale (see DEAD-011). Treat m7_mps.py as an orphaned snapshot, not a derived file that can be rebuilt.
- **Evidence:** m7_raw.py L2 '!pip...' vs m7_mps.py L2 '# !pip...'; convert_mps.py performs no pip edits (only device/seed/cache/print blocks, L14-167).
- **Fix:** Delete m7_raw.py and m7_mps.py together; do not attempt to reconcile or regenerate. Maintained source is m7_trimodal_run_provenance.py.
- **Fix at:** `code/m7_mps.py:1-3900`, `code/m7_raw.py:1-3877`
- _related: DEAD-011_

#### `RUN-018` [MED] Backbone fc1=Linear(64*5*5,1024) hardcodes a 34x34 input, forcing DVS to be pooled to 34x34 in collate
`code/m7_trimodal_run.py:242` · architecture · status:live · effort:medium · confidence:high
- **What:** SNN_Backbone.fc1 assumes the post-conv spatial size that results only from a 34x34 input (conv5->30, pool->15, conv5->11, pool->5 => 64*5*5=1600). To satisfy this fixed fan-in, dvs_collate_fn adaptively pools native 128x128 DVS frames down to 34x34 (98-102), a large downsampling for a visual-motion task. The coupling is implicit: no assertion, nothing derives the fan-in from input size, so any input-resolution change silently breaks the layer. (This 1600x1024 fan-in is also the basis of the SpiNNaker core budget.)
- **Evidence:** line 242 'self.fc1 = nn.Linear(64 * 5 * 5, 1024)'; dvs collate pooling to (34,34) at lines 98-102
- **Fix:** Derive the conv output size dynamically (lazy linear or a shape probe) or assert the expected input size; document why DVS is pooled to 34x34 and the information loss it implies.
- **Fix at:** `code/m7_trimodal_run.py:242`, `code/m7_trimodal_run.py:88-104`

#### `RUN-023` [MED] _get_features and forward duplicate the whole backbone feature-extraction, including the audio unroll verbatim
`code/m7_trimodal_run.py:302-342` · duplication · status:live · effort:small · confidence:high
- **What:** The audio-backbone temporal unroll in _get_features (307-318) is a verbatim copy of the same block in forward (326-337), and the visual branch '_, features = bb(x)' is duplicated too. Any change to feature extraction (e.g. adding spike-rate logging per RUN-011, or fixing timesteps per RUN-014) must be made in two places, which is exactly how numeric drift between the replay path and the training path is introduced.
- **Evidence:** lines 302-319 (_get_features) vs 321-342 (forward) contain the same isinstance branch and audio for-loop
- **Fix:** Extract a single _extract_features(x, modality) helper and call it from both _get_features and forward.
- **Fix at:** `code/m7_trimodal_run.py:302-342`
- _related: RUN-022_

#### `SNN-013` [MED] Two divergent M7 entrypoints produce non-comparable numbers
`code/m7_trimodal_run.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** m7_trimodal_run.py vs m7_trimodal_run_provenance.py differ in RNG order (SNN-001), fwt/forgetting definitions (SNN-002/010), CSV schema (SNN-003), and seed count (SNN-011). CLAUDE.md names m7_trimodal_run.py the entrypoint and the canonical loader matches its schema, yet every run_* script imports the provenance hub - so it is ambiguous which file produced any published number, and the two are not numerically interchangeable.
- **Evidence:** schema/metric/seed differences enumerated across SNN-001/002/003/010/011
- **Fix:** Declare one source-of-record, have the other import from it (no duplicated run_one_seed/model code), and reconcile metric definitions.
- **Fix at:** `code/m7_trimodal_run.py:1`, `code/m7_trimodal_run_provenance.py:1`
- _related: SNN-002, SNN-003, SNN-011_

#### `CLE-030` [MED] build_full_loaders and SHD/N-MNIST loader helpers are duplicated across the cluster
`code/run_cl_confirmation_fixed.py:69-179` · duplication · status:live · effort:large · confidence:high
- **What:** build_full_loaders (69-179) is nearly identical between run_cl_confirmation_fixed.py and run_cl_rehearsal.py (47-157). The _events_to_dense SHD-densify helper is copy-pasted into 4 files (run_cl_confirmation_fixed 101-111, run_cl_rehearsal 79-89, eval_transfer_fixed 46-56, run_rehearsal_transfer_eval_only 65-75), and build_nmnist_test_loader/build_shd_test_loader are duplicated between eval_transfer_fixed and phase1_sanity_check. Divergent edits to any copy (transform windows, split seed, densify logic) silently produce mismatched preprocessing across scripts that are supposed to be comparable.
- **Evidence:** build_full_loaders present in two files with matching bodies; _events_to_dense present in 4 files; build_*_test_loader present in eval_transfer_fixed.py:33-58 and phase1_sanity_check.py:34-42.
- **Fix:** Extract the loaders and _events_to_dense into a shared, import-safe module (alongside m7_trimodal_run_provenance) and import them; do not re-declare per script.
- **Fix at:** `code/run_cl_confirmation_fixed.py:69-179`, `code/run_cl_rehearsal.py:47-157`, `code/eval_transfer_fixed.py:33-58`, `code/run_rehearsal_transfer_eval_only.py:49-77`, `code/phase1_sanity_check.py:34-42`
- _related: CLE-031, CLE-032_

#### `CMP-013` [MED] File is a structural clone of run_joint_diagnostic.py and the magnitude harness
`code/run_compositional_crossmodal.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** set_seed, _build_cfg, _mean/_std, _binom_z, load_rows_from_disk, write_summary_and_report, _report_spiking_stats, and the main/aggregate scaffolding are near-duplicates of run_joint_diagnostic.py and run_joint_magnitude_ablation.py. The two audited files together are ~95 KB of largely copy-pasted experiment harness, so a fix to one (e.g. the dedup or the spike-metric bug) must be applied in several places.
- **Evidence:** set_seed(146), _build_cfg(153), _mean/_std(969), _binom_z(979), load_rows_from_disk(953), write_summary_and_report(984) duplicate the sibling
- **Fix:** Extract a shared, importable, __main__-guarded harness module (seed, cfg, aggregation, spiking-stats) and import it from both experiment scripts.
- **Fix at:** `code/run_compositional_crossmodal.py:146-165`, `code/run_compositional_crossmodal.py:969-997`, `code/run_compositional_crossmodal.py:795-809`, `code/run_joint_diagnostic.py:130-135`, `code/run_joint_diagnostic.py:441-505`
- _related: CMP-024_

#### `ALGN-007` [MED] Audio backbone forward hand-reimplemented, duplicating the model (silent-drift risk)
`code/run_correspondence_anchor_v2.py:190-207` · duplication · status:live · effort:small · confidence:high
- **What:** CorrespondenceModelV2._get_backbone_features (and the dead _extract_backbone_features) manually re-run the SHD audio ModuleDict forward (bb['lif1'](bb['fc1'](xt)) ... fc3), duplicating M7_ContinualAdaptiveModel._get_features/forward_with_shared in m7_trimodal_run_provenance.py:343-399. It is numerically consistent with the model TODAY (verified: audio ModuleDict is fc1->1024,fc2->1024,fc3->512 with lif1-3), but if the provenance backbone changes, this copy silently produces wrong features with no error.
- **Evidence:** lines 190-207 mem1=bb['lif1'].init_leaky(); s3=bb['lif3'](bb['fc3'](s2)); provenance _get_features m7_trimodal_run_provenance.py:343-360 is identical logic
- **Fix:** Call the base model's feature extractor (e.g. a public _get_features) instead of re-deriving the forward pass in each script.
- **Fix at:** `code/run_correspondence_anchor_v2.py:190-207`, `code/run_correspondence_anchor_v2.py:120-144`, `code/run_ordinal_anchor.py:282-299`
- _related: ALGN-006, ALGN-018_

#### `ALGN-008` [MED] Cluster-wide helper duplication: no shared alignment library
`code/run_correspondence_anchor_v2.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** Core alignment helpers are copy-pasted across the run_* fleet instead of centralized. compute_held_out_metrics has 3 definitions (run_correspondence_anchor_v2.py:372, run_joint_magnitude_ablation.py:572, run_joint_shared_encoder.py:653 [different signature]); _hash_params has 2 (run_correspondence_anchor_v2.py:59, code/archive/run_correspondence_anchor.py:65); _extract_backbone_features 2 (v2:120, archive:139); build_paired_feature_dataset and train_correspondence in both v2 and archive; run_procrustes in v2:450 and run_joint_shared_encoder.py:704 plus run_procrustes_z in run_joint_magnitude_ablation.py:648; _events_to_dense re-defined in ~8 files. Divergent copies will silently disagree.
- **Evidence:** grep -rn 'def compute_held_out_metrics|def _hash_params|def run_procrustes|def _extract_backbone_features|def build_paired_feature_dataset|def _events_to_dense' code/ (see counts above)
- **Fix:** Extract shared alignment/metric helpers into one module (e.g. code/alignment_common.py or utils/) and import everywhere; delete archived duplicates from the live import surface.
- **Fix at:** `code/run_correspondence_anchor_v2.py:59-64`, `code/run_correspondence_anchor_v2.py:120-144`, `code/run_correspondence_anchor_v2.py:241-275`, `code/run_correspondence_anchor_v2.py:278-365`, `code/run_correspondence_anchor_v2.py:372-443`, `code/run_correspondence_anchor_v2.py:450-517`, `code/run_ordinal_anchor.py:92-102`, `code/run_correspondence_anchor_v2.py:93-103`, `code/run_digit_only_alignment.py:122-132`
- _related: ALGN-006, ALGN-007, ALGN-009, ALGN-018_

#### `ALGN-030` [MED] Digit-only script couples to the legacy bi-modal Dataloaders package (boundary)
`code/run_digit_only_alignment.py:152` · architecture · status:live · effort:medium · confidence:medium
- **What:** build_digit_loaders does 'from Dataloaders.paired_digit_loader import get_paired_digit_loaders' (line 152). Dataloaders/ is the legacy bi-modal (Paper 2) package that CLAUDE.md and python-numerics.md place OUTSIDE the M7 reproducibility boundary (m7_trimodal_run.py must not import it). The import resolves (the module exists), but it couples an M7-era alignment study to legacy code, so a change there silently affects these results.
- **Evidence:** line 152 from Dataloaders.paired_digit_loader import get_paired_digit_loaders; Dataloaders/paired_digit_loader.py exists at repo root (legacy package)
- **Fix:** Vendor the paired-loader logic into the M7 code/ surface or into the shared provenance module so the alignment studies do not depend on the legacy package.
- **Fix at:** `code/run_digit_only_alignment.py:151-172`
- _related: ALGN-008_

#### `CMP-024` [MED] File is a structural clone of run_compositional_crossmodal.py and the magnitude harness
`code/run_joint_diagnostic.py:1` · architecture · status:live · effort:large · confidence:high
- **What:** set_seed, _build_cfg, _mean/_std, _binom_z, load_rows_from_disk, write_summary_and_report, _report_spiking_stats and the main/aggregate-only scaffolding are near-duplicates of the sibling file and run_joint_magnitude_ablation.py. Shared bugs (dedup, spike-metric semantics, dead-config) must be fixed in multiple copies.
- **Evidence:** set_seed(130), _build_cfg(624), _mean/_std(441), _binom_z(451), load_rows_from_disk(469), write_summary_and_report(487) mirror the sibling
- **Fix:** Extract a shared importable harness module and import it from both scripts.
- **Fix at:** `code/run_joint_diagnostic.py:130-135`, `code/run_joint_diagnostic.py:441-505`, `code/run_joint_diagnostic.py:336-349`, `code/run_compositional_crossmodal.py:146-165`
- _related: CMP-013_

#### `JNT-008` [MED] VisualStem/AudioStem re-implemented independently of the provenance backbone
`code/run_joint_magnitude_ablation.py:288-352` · duplication · status:live · effort:medium · confidence:high
- **What:** The visual and audio SNN stems are hand-copied with divergent final dimensions instead of reusing the provenance backbone, creating a third drifting copy of the same architecture across the repo.
- **Evidence:** VisualStem (288-324) and AudioStem (327-352) duplicate the provenance canonical stems (m7_trimodal_run_provenance.py:258-297 visual SNN_Backbone; 307-317 audio ModuleDict) but end at 256-d with 4 LIFs. run_joint_shared_encoder.py:280-352 is a third variant (adds fc2 1024->512 and fc_proj, 5 LIFs). No shared source of truth.
- **Fix:** Extract one parameterized visual conv stem and audio MLP stem into an importable module and consume it from provenance and both joint scripts; if the joint experiments deliberately use a smaller stem, document that deviation in one place.
- **Fix at:** `code/run_joint_magnitude_ablation.py:288-352`, `code/m7_trimodal_run_provenance.py:258-297`, `code/m7_trimodal_run_provenance.py:307-317`, `code/run_joint_shared_encoder.py:280-352`
- _related: JNT-018_

#### `JNT-017` [MED] Parallel Config dataclass re-declares ExperimentConfig hyperparameters (config-drift hazard)
`code/run_joint_shared_encoder.py:72-88` · architecture · status:live · effort:medium · confidence:high
- **What:** A second config dataclass re-declares shared hyperparameters with its own defaults, so values like beta or lr can silently diverge between M7 and this experiment.
- **Evidence:** Config (72-88) re-declares lr=1e-3, weight_decay=1e-4, beta=0.9, batch_size, num_workers and seeds, parallel to ExperimentConfig (m7_trimodal_run_provenance.py:33-74). Two independent sources of truth for the same knobs.
- **Fix:** Compose or derive from ExperimentConfig for the shared hyperparameters, keeping only experiment-unique fields (d_stem, d_embed, supcon_temperature, samples_per_class, epochs) local, so one edit governs both.
- **Fix at:** `code/run_joint_shared_encoder.py:72-88`, `code/m7_trimodal_run_provenance.py:33-74`
- _related: JNT-013_

#### `JNT-018` [MED] Own VisualStem/AudioStem duplicate the provenance backbone and diverge from the ablation's stems
`code/run_joint_shared_encoder.py:280-352` · duplication · status:live · effort:medium · confidence:high
- **What:** A third copy of the visual/audio SNN stems, structurally different from both the provenance backbone and the magnitude ablation's stems.
- **Evidence:** VisualStem (280-320) and AudioStem (323-352) duplicate provenance (m7_trimodal_run_provenance.py:258-297,307-317) and differ from run_joint_magnitude_ablation.py:288-352 (this variant adds fc2 1024->512 and fc_proj 512->d_stem, 5 LIFs vs the ablation's 4).
- **Fix:** Consume one shared parameterized stem module (see JNT-008) instead of a per-script copy.
- **Fix at:** `code/run_joint_shared_encoder.py:280-352`, `code/m7_trimodal_run_provenance.py:258-297`, `code/run_joint_magnitude_ablation.py:288-352`
- _related: JNT-008_

#### `ALGN-018` [MED] OrdinalCorrespondenceModel duplicates CorrespondenceModelV2 except for anchors
`code/run_ordinal_anchor.py:247-326` · duplication · status:live · effort:small · confidence:high
- **What:** OrdinalCorrespondenceModel (247-326) is a near-verbatim copy of CorrespondenceModelV2 (run_correspondence_anchor_v2.py:164-234): identical head construction, _get_backbone_features (282-299), project (301-306), forward_with_shared (308-314), task_accuracy (316-326). The only real difference is anchors = build_ordinal_anchors(...) instead of the QR construction. Two full copies of the audio-forward re-implementation (ALGN-007) live here too.
- **Evidence:** lines 282-326 mirror run_correspondence_anchor_v2.py:190-234; only the register_buffer('anchors', ...) source differs (line 277 vs v2:182-186)
- **Fix:** Make OrdinalCorrespondenceModel subclass CorrespondenceModelV2 and override only anchor construction, or parameterize the anchor builder.
- **Fix at:** `code/run_ordinal_anchor.py:247-326`, `code/run_correspondence_anchor_v2.py:164-234`
- _related: ALGN-007, ALGN-008_

#### `CLE-005` [MED] cfg is a hand-copied duplicate of run_cl_rehearsal.py's config (drift hazard)
`code/run_rehearsal_transfer_eval_only.py:21-35` · duplication · status:live · effort:small · confidence:high
- **What:** The ExperimentConfig block (22-35) is a manual copy of the config in run_cl_rehearsal.py, flagged by its own comment `# --- Reconstruct config (must match run_cl_rehearsal.py) ---` (21). A hand-maintained 'must match' duplicate is a classic drift hazard, and it has already drifted in effect (see the replay_weight discrepancy, CLE-003). If the source config changes, this eval silently evaluates under a different config than the run that produced the checkpoint.
- **Evidence:** Line 21 comment plus the literal block 22-35, which mirrors run_cl_rehearsal.py:377-390.
- **Fix:** Load the config that was saved alongside the checkpoint (the run writes config_*.json / master_config.json) via ExperimentConfig(**json.load(...)) rather than re-declaring it, as eval_transfer_fixed.py and phase1_sanity_check.py already do.
- **Fix at:** `code/run_rehearsal_transfer_eval_only.py:21-35`
- _related: CLE-032_


## LOW (10)

#### `FIG-027` [LOW] final_evaluation.py is misfiled: a hardware-verify script in code/ root, not with the archive cluster
`code/final_evaluation.py:1` · architecture · status:live · effort:trivial · confidence:high
- **What:** final_evaluation.py is a SpiNNaker discrete-time-simulator verification script, but it lives in code/ root while its peer hardware-verify scripts are in code/archive/: verify_brian2_raw.py, verify_simulators.py, verify_full.py, verify_tiny.py, debug_brian2_exact.py (confirmed by listing). It even references 'raw Brian2' verification (307) that corresponds to code/archive/verify_brian2_raw.py. Its presence in the main code/ directory, named 'final', implies it is a current M7 deliverable rather than an archived hardware-verification experiment.
- **Evidence:** code/archive/ contains verify_brian2_raw.py, verify_simulators.py, debug_brian2_exact.py; final_evaluation.py is in code/ root
- **Fix:** Move code/final_evaluation.py into code/archive/ alongside the other verify_*/debug_brian2 scripts (or into a dedicated hardware-verify directory).
- **Fix at:** `code/final_evaluation.py:1`
- _related: FIG-020_

#### `FIG-028` [LOW] ConvFrontend duplicates FullModel's conv layers, and both re-implement m7's SNN_Backbone
`code/final_evaluation.py:33-85` · architecture · status:live · effort:small · confidence:high
- **What:** ConvFrontend (33-53) redefines conv1/pool1/lif1/conv2/pool2/lif2 identical to the first half of FullModel (59-64), and both re-implement the backbone that m7_trimodal_run.py defines as SNN_Backbone (conv1 2->32 k5, conv2 32->64 k5, fc1 1600->1024, fc2 1024->512). The architecture is stated three times; a backbone change (e.g. the 2-vs-3 conv issue resolved in contested-facts.md) must be mirrored here or these silently diverge from the trained model.
- **Evidence:** ConvFrontend and FullModel both define conv1/conv2/lif1/lif2 with the same params
- **Fix:** Have ConvFrontend reuse FullModel's sub-modules (or share a single backbone definition) instead of re-declaring the conv stack; ideally import the canonical backbone rather than copying it.
- **Fix at:** `code/final_evaluation.py:33-53`, `code/final_evaluation.py:56-85`
- _related: FIG-017_

#### `FIG-017` [LOW] box() helper and the entire Figure 5 architecture schematic are duplicated across both figure scripts
`code/generate_figures.py:185-252` · duplication · status:live · effort:small · confidence:high
- **What:** The box() helper (generate_figures.py 192-197 vs generate_figures_3seed.py 339-343) and the whole Figure-5 schematic block (generate_figures.py 185-252 vs generate_figures_3seed.py 330-378) are near-identical copies. This is dead duplication maintained in two places and part of why the two scripts collide (FIG-001).
- **Evidence:** def box(...) and Figure 5 present verbatim in both files
- **Fix:** After retiring generate_figures.py (FIG-008) the duplication disappears. If both must exist, factor box()/the schematic into a shared module.
- **Fix at:** `code/generate_figures.py:185-252`, `code/generate_figures_3seed.py:330-378`
- _related: FIG-008_

#### `CLE-022` [LOW] forgetting_T2_T3 and total_forgetting_T2_T3 are computed identically (always equal)
`code/run_cl_confirmation_fixed.py:328-330` · duplication · status:live · effort:trivial · confidence:high
- **What:** forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3 (328) and total_forgetting_t2_t3 = acc_t2_after - acc_t2_after_t3 (330) are the same expression, so the two CSV columns are always byte-identical -- redundant, and asymmetric with the T1 pair where the two columns genuinely differ (327 vs 329). Either total_forgetting_T2_T3 was meant to be a different quantity, or one column is superfluous. Same in run_cl_rehearsal.py:337/339.
- **Evidence:** Lines 328 and 330 are identical expressions.
- **Fix:** Drop the redundant column or, if a distinct quantity was intended (e.g. an incremental T2 measure), compute it correctly. Mirror in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:328-330`, `code/run_cl_rehearsal.py:337-339`
- _related: CLE-021_

#### `CLE-032` [LOW] ExperimentConfig literal block triplicated across the run/eval scripts
`code/run_cl_confirmation_fixed.py:364-377` · duplication · status:live · effort:medium · confidence:high
- **What:** The same ~14-line ExperimentConfig(...) literal is declared three times: run_cl_confirmation_fixed.py (364-377), run_cl_rehearsal.py (377-390), and run_rehearsal_transfer_eval_only.py (22-35). They must stay in lockstep for the eval scripts to match the runs, but nothing enforces it (see the replay_weight drift, CLE-003). 
- **Evidence:** Three near-identical ExperimentConfig literal blocks at the cited locations.
- **Fix:** Define the base config once (a factory or a JSON that all three load) and have the eval scripts read the config saved with the checkpoint rather than re-declaring it.
- **Fix at:** `code/run_cl_confirmation_fixed.py:364-377`, `code/run_cl_rehearsal.py:377-390`, `code/run_rehearsal_transfer_eval_only.py:22-35`
- _related: CLE-005, CLE-030_

#### `JNT-011` [LOW] _events_to_dense_shd duplicates SHD densification present in three other files
`code/run_joint_magnitude_ablation.py:72-82` · duplication · status:live · effort:small · confidence:high
- **What:** The SHD event-to-dense conversion exists in at least four near-identical copies across the code tree, with subtly different output shapes.
- **Evidence:** _events_to_dense_shd (72-82) returns (T,C) 2-D; run_joint_shared_encoder.py:258-268 returns (T,1,1,C); m7_trimodal_run_provenance.py:1200-1210 and run_correspondence_anchor_v2.py:93-103 return (T,1,1,C),label. Same algorithm, four sites.
- **Fix:** Centralize one parameterized SHD densification helper (bins/channels/shape) in a shared module and import it everywhere.
- **Fix at:** `code/run_joint_magnitude_ablation.py:72-82`, `code/run_joint_shared_encoder.py:258-268`, `code/m7_trimodal_run_provenance.py:1200-1210`
- _related: JNT-027_

#### `JNT-026` [LOW] Own nmnist_collate_fn/shd_collate_fn duplicate the provenance collates
`code/run_joint_shared_encoder.py:241-273` · duplication · status:live · effort:small · confidence:high
- **What:** This file re-implements the collate functions instead of importing them, unlike the magnitude ablation which imports the provenance versions.
- **Evidence:** nmnist_collate_fn (241-253) and shd_collate_fn (271-273) duplicate m7_trimodal_run_provenance.py:107-123. This file imports only CrossModalSupConLoss and spike_grad from provenance (47-50); run_joint_magnitude_ablation.py:49-55 imports the provenance collates.
- **Fix:** Import the provenance collate functions instead of re-defining them (as the magnitude ablation does).
- **Fix at:** `code/run_joint_shared_encoder.py:241-273`, `code/m7_trimodal_run_provenance.py:107-123`
- _related: JNT-011_

#### `JNT-027` [LOW] _events_to_dense is another SHD densification copy with a third shape-normalization site
`code/run_joint_shared_encoder.py:258-268` · duplication · status:live · effort:small · confidence:high
- **What:** Another copy of the SHD densifier, producing a (T,1,1,C) tensor that is then squeezed in two more places for one logical tensor.
- **Evidence:** _events_to_dense (258-268) returns (100,1,1,700); shd_collate_fn squeezes to (B,100,700) (273); PairedDigitDatasetFiltered.__getitem__ squeezes again (226-227). See JNT-011 for the other copies.
- **Fix:** Centralize densification and settle on one output shape to remove the repeated squeezes.
- **Fix at:** `code/run_joint_shared_encoder.py:258-268`, `code/run_joint_shared_encoder.py:226-227`, `code/run_joint_shared_encoder.py:273`
- _related: JNT-011_

#### `TST-011` [LOW] Local TASKS is a hand-copied duplicate of the canonical module-level TASKS (silent-drift risk)
`code/test_cross_modal_alignment.py:61-65` · duplication · status:live · effort:trivial · confidence:high
- **What:** The test defines its own TASKS dict (lines 61-65) instead of importing the canonical TASKS from m7_trimodal_run_provenance (lines 955-959). The two are byte-for-byte equivalent TODAY (nmnist:10/visual, shd:20/audio, dvs:11/visual_motion), so there is no current bug, but it is a duplicated source of truth: if the real TASKS schema or class counts change, this test keeps building the model against a stale task spec and will not detect the drift. (Note: it is not literal Python shadowing, since this file does not import TASKS — it is a divergent copy.)
- **Evidence:** code/test_cross_modal_alignment.py:61-65 local TASKS; identical structure to m7_trimodal_run_provenance.py:955-959.
- **Fix:** Import TASKS from m7_trimodal_run_provenance and use it, or build the test's TASKS from that source, so there is a single source of truth.
- **Fix at:** `code/test_cross_modal_alignment.py:61-65`
- _related: TST-020_

#### `TST-020` [LOW] Inline ExperimentConfig duplicates the full production hyperparameter set
`code/test_ordinal_anchor_smoke.py:23-56` · duplication · status:live · effort:small · confidence:medium
- **What:** The test hand-writes a 34-field ExperimentConfig (lines 23-56) restating essentially all M7 production hyperparameters (num_patterns=100, time_steps=25, beta=0.9, hidden_dim=512, shared_dim=512, contrastive_temperature=0.07, replay_weight=0.3, etc.). Most are already the dataclass defaults (m7_trimodal_run_provenance.py:34-74), so this is redundant, and it is a second place these production values are pinned — if the real defaults change, this smoke config silently diverges. A smoke test should use small/cheap values (small dims, few epochs), not mirror the full production config.
- **Evidence:** code/test_ordinal_anchor_smoke.py:23-56 inline ExperimentConfig with production values; same fields/defaults defined in m7_trimodal_run_provenance.py:34-74.
- **Fix:** Only override the fields the smoke test needs (and make them small); rely on dataclass defaults for the rest. Avoid re-pinning production hyperparameters in a test.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:23-56`
- _related: TST-011_


## INFO (1)

#### `PROV-027` [INFO] Shared hub contains the SpiNNaker-non-portable ops (LayerNorm, Softmax gate, HGRN multiplicative reset r*h_prev)
`code/m7_trimodal_run_provenance.py:222-255` · architecture · status:live · effort:trivial · confidence:high
- **What:** The model defined here is what the SpiNNaker port must map. It uses nn.LayerNorm inside Hopfield (228) and HGRN (243-245) and in each head (332); a Softmax gate (327); and the HGRN multiplicative reset r * h_prev (254). Per .claude/rules/spinnaker.md and CLAUDE.md, LayerNorm and Softmax are not native to sPyNNaker and the multiplicative reset r*h_prev is flagged as not expressible (risk R3, threatening the novelty claim). This is not a code defect; it is recorded so the port work references the actual hub source rather than the orphan.
- **Evidence:** LayerNorm in Hopfield (228), HGRN (243-245), heads (332); Softmax gate (327); h_tilde uses torch.cat([x, r * h_prev]) (254).
- **Fix:** No code change here. Use these exact lines as the reference when assessing SpiNNaker expressibility; do not assert portability without a board run (mapping != inference != hardware).
- **Fix at:** `code/m7_trimodal_run_provenance.py:222-255`, `code/m7_trimodal_run_provenance.py:326-333`
