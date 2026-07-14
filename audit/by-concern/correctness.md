# Correctness / logic bugs

_Generated 2026-07-15. 48 canonical findings in categories: correctness. Duplicates omitted (see `findings.jsonl`)._

Severity: CRITICAL 5, HIGH 8, MED 19, LOW 14, INFO 2


## CRITICAL (5)

#### `RUN-002` [CRITICAL] Replay sample() filters TASK ids against a CLASS-keyed store and draws <=1 embedding per class
`code/m7_trimodal_run.py:380-392` · correctness · status:live · effort:medium · confidence:high
- **What:** _store is keyed by class label (0-19), but callers pass replay_task_ids=[1] (T2) and [1,2] (T3) meaning TASK numbers. sample() computes 'available = [c for c in task_ids if c in self._store]', so it keeps only the class(es) whose integer label equals a task id: T2 replay uses only class 1, T3 replay uses only classes 1 and 2. It then draws exactly one embedding per selected class, so a requested replay_batch of 32 yields at most 1-2 samples. self._task_labels (the map that would make this correct) is never consulted.
- **Evidence:** line 383 'available = [c for c in task_ids if c in self._store]'; line 386 'np.random.choice(available, size=min(batch_size, len(available)), replace=False)'; lines 388-391 one 'idx = np.random.randint(len(self._store[c]))' per class; callers 569 (task_ids=[1]) and 596 ([1,2])
- **Fix:** Select classes via _task_labels (classes whose task id is requested) and sample multiple embeddings per class up to batch_size, instead of one per class. Cloned in the provenance sibling.
- **Fix at:** `code/m7_trimodal_run.py:380-392`, `code/m7_trimodal_run.py:569`, `code/m7_trimodal_run.py:596`, `code/m7_trimodal_run_provenance.py:446-458`
- _related: RUN-003, RUN-004, RUN-030_

#### `RUN-003` [CRITICAL] populate() uses label_offset=0 for every task, so SHD classes overwrite N-MNIST in the class-keyed store
`code/m7_trimodal_run.py:361-378` · correctness · status:live · effort:medium · confidence:high
- **What:** _store is keyed by 'c = int(lbl.item()) + label_offset' and populate() is called with label_offset=0 for T1 (nmnist, classes 0-9) and T2 (shd, classes 0-19). SHD classes 0-9 overwrite the stored N-MNIST embeddings under the same keys. By T3, replay 'of T1 and T2' contains no N-MNIST embeddings at all, so T1 gets zero replay protection during T3. Combined with RUN-002 the replay signal is both misrouted and largely destroyed.
- **Evidence:** line 369 'c = int(lbl.item()) + label_offset'; call sites line 553 '...task_id=1, label_offset=0' and line 584 '...task_id=2, label_offset=0'
- **Fix:** Key the store so class collisions across tasks are impossible (e.g. by (task_id, class) or a per-task offset) and make sample() task-aware (RUN-002). A real per-task label_offset is the minimal fix. Cloned in the provenance sibling.
- **Fix at:** `code/m7_trimodal_run.py:361-378`, `code/m7_trimodal_run.py:553`, `code/m7_trimodal_run.py:584`, `code/m7_trimodal_run_provenance.py:427-444`, `code/m7_trimodal_run_provenance.py:1011`, `code/m7_trimodal_run_provenance.py:1079`
- _related: RUN-002, RUN-030_

#### `PROV-001` [CRITICAL] EmbeddingReplayBuffer replay is fundamentally broken: task_ids used as class keys, label_offset=0 collides tasks, wrong head/labels
`code/m7_trimodal_run_provenance.py:446-458` · correctness · status:live · effort:medium · confidence:high
- **What:** populate() keys the store by raw class label c=int(lbl.item())+label_offset (435) with label_offset=0 at BOTH call sites (1011, 1079), so SHD classes 0-9 OVERWRITE N-MNIST classes 0-9 in the shared _store dict (self._store[c]=... at 443). sample(task_ids) then treats task_ids as store KEYS, not task ids: available=[c for c in task_ids if c in self._store] (449). Call sites pass replay_task_ids=[1] (1050) and [1,2] (1102), so replay draws only class-index 1 (and 2) embeddings from whatever task last wrote those keys, never a task's worth of samples and never N-MNIST (it was overwritten). The sampled labels are the class indices (457) and are fed to forward_from_embeddings(embs, task_name) through the CURRENT task's head (611-615), so replay optimizes the wrong head against wrong labels. _task_labels[c]=task_id (444) records the true task but sample() never consults it. Net effect: 'embedding replay' rehearses one-to-two wrong-task embeddings through the current head, so it cannot prevent forgetting of prior tasks. This is the hub imported by 19 files, so the defect propagates everywhere.
- **Evidence:** key c=int(lbl.item())+label_offset (435); self._store[c]=torch.stack(samples) (443); self._task_labels[c]=task_id (444, unused by sample); available=[c for c in task_ids if c in self._store] (449); label_offset=0 (1011,1079); replay_task_ids=[1] (1050), [1,2] (1102); forward_from_embeddings(replay_embs, task_name) (612-614). The CORRECT pattern is present in the same file: RawInputReplayBuffer.sample filters by task_names (498-508).
- **Fix:** Key the buffer per task so tasks cannot collide (offset labels by task, or nest by task_id); in sample() select stored classes via _task_labels==requested task_id (mirror RawInputReplayBuffer.sample, 498); replay each sample through its OWN task head with its own labels, not the current task_name. Same defect and the same fix apply to the orphan clone code/m7_trimodal_run.py.
- **Fix at:** `code/m7_trimodal_run_provenance.py:435`, `code/m7_trimodal_run_provenance.py:444`, `code/m7_trimodal_run_provenance.py:446-458`, `code/m7_trimodal_run_provenance.py:611-615`, `code/m7_trimodal_run_provenance.py:1011`, `code/m7_trimodal_run_provenance.py:1050`, `code/m7_trimodal_run_provenance.py:1079`, `code/m7_trimodal_run_provenance.py:1102`, `code/m7_trimodal_run.py:380-392`, `code/m7_trimodal_run.py:553`, `code/m7_trimodal_run.py:569`, `code/m7_trimodal_run.py:584`, `code/m7_trimodal_run.py:596`
- _related: PROV-003_

#### `ALGN-001` [CRITICAL] Held-out training path is doubly broken: CE reduced before mask, and 6 anchors vs labels 0-9
`code/run_correspondence_anchor_v2.py:311-324` · correctness · status:live · effort:small · confidence:high
- **What:** train_correspondence uses ce=nn.CrossEntropyLoss() (default reduction='mean', line 287). In the held-out config, run_one_seed builds corr_ho with num_anchors=6 (line 575) so model.anchors is (6, anchor_dim) and logits_vis/logits_aud have 6 columns (315-316), but the training set from build_paired_feature_dataset contains labels 0-9 (loops range(10), lines 251,261). ce(6-col logits, target=9) raises RuntimeError 'Target out of bounds', so held-out training crashes every seed. Even if it did not crash, the mask is applied AFTER the mean reduction: loss is already a scalar, so loss=loss*mask.float() broadcasts to (B,) and loss.sum()/mask.sum() collapses back to the same scalar, making the held-out exclusion a no-op. main() wraps run_one_seed in try/except (669-672), so the crash is swallowed and NO per-seed CSV rows are ever written; the script's central 'held-out 6-9 retrieval' deliverable never runs. The paper's held-out-retrieval-at-chance numbers cannot have been produced by this code path.
- **Evidence:** line 287 ce=nn.CrossEntropyLoss(); lines 318-324 loss_vis=ce(logits_vis,labels); loss=loss_vis+loss_aud; if len(held_out_classes)>0: loss=loss*mask.float(); loss=loss.sum()/(mask.sum().clamp(min=1)); line 575 corr_ho=...num_anchors=6; lines 251,261 build_paired_feature_dataset loops range(10) so labels are 0-9
- **Fix:** Filter logits and labels by the keep-mask BEFORE calling ce (or set reduction='none' and average only the kept per-sample losses). The held-out config must train on 0-5 samples only so targets are in [0,5] and match the 6 anchors; masking the data also removes the target-out-of-bounds crash.
- **Fix at:** `code/run_correspondence_anchor_v2.py:287`, `code/run_correspondence_anchor_v2.py:311-324`, `code/run_correspondence_anchor_v2.py:575-579`
- _related: ALGN-015_

#### `ALGN-015` [CRITICAL] Stage 1 inherits the broken held-out training from v2
`code/run_ordinal_anchor.py:51-58` · correctness · status:live · effort:small · confidence:high
- **What:** run_ordinal_anchor imports train_correspondence and compute_held_out_metrics from run_correspondence_anchor_v2 (lines 51-58) and drives the identical held-out path: corr_ho = OrdinalCorrespondenceModel(..., num_anchors=6) (483-485) trained via train_correspondence(..., held_out_classes=(6,7,8,9)) (487-497) on labels 0-9. This hits the same CE-target-out-of-bounds crash plus no-op mask described in ALGN-001, and main() swallows it via try/except (680-684). Stage 1 (the ordinal held-out retrieval that is the script's entire reason to exist) never completes.
- **Evidence:** lines 51-58 from run_correspondence_anchor_v2 import (... train_correspondence, compute_held_out_metrics); lines 483-497 corr_ho num_anchors=6 + train_correspondence held_out_classes=(6,7,8,9); build_paired_feature_dataset (imported) yields labels 0-9
- **Fix:** Fix train_correspondence per ALGN-001; then Stage 1 here runs. No separate change needed once the shared helper is corrected.
- **Fix at:** `code/run_ordinal_anchor.py:483-497`, `code/run_correspondence_anchor_v2.py:311-324`
- _related: ALGN-001_


## HIGH (8)

#### `ARCH-015` [HIGH] Held-out-class masking is a no-op: held-out digits are still trained on
`code/archive/run_correspondence_anchor.py:412-419` · correctness · status:archive · effort:small · confidence:high
- **What:** train_correspondence uses ce = nn.CrossEntropyLoss() with default reduction='mean', so loss_vis/loss_aud/loss are SCALARS (lines 412-414). The held-out masking then does loss = loss * mask.float() (scalar broadcast to a per-sample vector) then loss.sum()/mask.sum() (lines 418-419), which algebraically returns the original scalar mean over ALL samples — the held-out classes were never excluded. Compounding this, build_paired_feature_dataset accepts held_out_classes (line 299) but never uses it, so held-out samples are also present in the training set. Net effect: the step-7 'held-out-class' test (digits 8,9, lines 616-628) evaluates alignment on classes the model was actually trained on, invalidating that result and the derived 'discovered vs engineered' verdict (line 651).
- **Evidence:** nn.CrossEntropyLoss() default mean at line 377; scalar loss lines 412-414; scalar*mask then sum/sum lines 418-419; held_out_classes param unused in build_paired_feature_dataset (declared line 299); held-out run at lines 622-628.
- **Fix:** To actually hold classes out: use reduction='none' to get per-sample losses BEFORE masking, or filter held-out samples out of the dataset in build_paired_feature_dataset. As-is the held-out numbers are meaningless. For retirement this is a primary reason v1 was superseded (ARCH-001) — confirm v2 fixed it.
- **Fix at:** `code/archive/run_correspondence_anchor.py:377`, `code/archive/run_correspondence_anchor.py:412-419`, `code/archive/run_correspondence_anchor.py:299`
- _related: ARCH-001, ARCH-017_

#### `CLE-009` [HIGH] Empty-results DataFrame raises KeyError before the 'in' guard can protect it
`code/eval_transfer_fixed.py:126` · correctness · status:live · effort:small · confidence:high
- **What:** If both checkpoints are missing but the config loaded (the 'only checkpoints missing' path), the loop skips every stage and results stays []. Then df = pd.DataFrame([]) has no columns, and the verdict line evaluates the ternary condition `"post_T2" in df["stage"].values` first -- df["stage"] raises KeyError: 'stage' before the membership test or the `else 0` fallback can run. So the intended empty-case fallback never fires; the script crashes.
- **Evidence:** Line 120 `df = pd.DataFrame(results)`; line 126 `... if "post_T2" in df["stage"].values else 0`. With results==[] the DataFrame has no 'stage' column.
- **Fix:** Handle the empty case before indexing: e.g. `if df.empty or "post_T2" not in df["stage"].values: post_t2_v2a = None` and branch the verdict on None (report 'no data' rather than a number).
- **Fix at:** `code/eval_transfer_fixed.py:120-137`
- _related: CLE-010_

#### `RUN-004` [HIGH] Replayed embeddings pass through the CURRENT task head, not the originating task head
`code/m7_trimodal_run.py:453` · correctness · status:live · effort:medium · confidence:high
- **What:** During task T, replay is scored via 'forward_from_embeddings(replay_embs, task_name)' where task_name is the current task. Old-task embeddings are pushed through the current head and trained against old-class integer labels. This does nothing to preserve the previous task's head (the purpose of rehearsal) and instead trains the current head on foreign-class targets. forward_from_embeddings has no way to select the originating head.
- **Evidence:** line 453 'replay_logits = self.model.forward_from_embeddings(replay_embs, task_name)'; def 344-349 uses 'self.heads[task_name]'
- **Fix:** Store the originating task per embedding (RUN-030) and route each replayed embedding through its own head; forward_from_embeddings must accept a per-group task identity. Cloned in the provenance sibling.
- **Fix at:** `code/m7_trimodal_run.py:344-349`, `code/m7_trimodal_run.py:451-454`, `code/m7_trimodal_run_provenance.py:409-414`, `code/m7_trimodal_run_provenance.py:614`
- _related: RUN-002, RUN-003, RUN-030_

#### `RUN-012` [HIGH] HGRN reset gate is inert: h_prev is always zero, so r*h_prev is 0 and W_r/norm_r get no gradient
`code/m7_trimodal_run.py:223-231` · correctness · status:live · effort:medium · confidence:high
- **What:** ImprovedHGRNGate is only ever called with a single argument (self.hgrn(features) at 340, self.hgrn(embs) at 347), so h_prev defaults to zeros on every call and the gate is never used recurrently. With h_prev=0, 'r * h_prev' is identically 0, h_tilde reduces to tanh(norm_h(W_h([x,0]))) and the output is 'z * h_tilde'. The reset gate r (and W_r, norm_r) has zero effect on the output and receives zero gradient, so those parameters stay at init. The multiplicative reset r*h_prev is exactly the operation flagged as the central SpiNNaker risk (R3) and a novelty of the paper - and in the model as run it does nothing.
- **Evidence:** line 226 'h_prev = torch.zeros(batch_size, ...)'; line 230 'h_tilde = torch.tanh(self.norm_h(self.W_h(torch.cat([x, r * h_prev], dim=-1))))'; line 231 'return (1 - z) * h_prev + z * h_tilde'; call sites 340 and 347 pass no h_prev
- **Fix:** Decide what the HGRN should do: if recurrence is intended, thread a real hidden state; if not, remove the reset gate and its weights and revise the novelty/SpiNNaker-R3 claims. Either way it must not be described as a working multiplicative-reset recurrent gate.
- **Fix at:** `code/m7_trimodal_run.py:214-231`, `code/m7_trimodal_run.py:340`, `code/m7_trimodal_run.py:347`
- _related: RUN-013_

#### `PROV-003` [HIGH] Replay path (forward_from_embeddings) bypasses _split_shared_private, so replayed embeddings hit a different subgraph than live inputs
`code/m7_trimodal_run_provenance.py:409-414` · correctness · status:live · effort:small · confidence:high
- **What:** Live inputs are transformed by _split_shared_private into full before gate/hopfield/hgrn (394-398). But forward_from_embeddings(embs, task_name) (409-414) applies adaptive_gate/hopfield/hgrn DIRECTLY to the stored embeddings, skipping proj_shared/proj_private entirely. The stored embeddings are raw backbone features (populate calls model._get_features, 433, which returns backbone output). So replay trains gate/Hopfield/HGRN/heads on raw-feature distribution while inference uses the projected full distribution — a train/inference mismatch that corrupts the replay gradient even if PROV-001 were fixed. This inconsistency is PROVENANCE-SPECIFIC: in the orphan both live and replay paths use raw features (m7_trimodal_run.py:338-340 vs 344-349), so they are consistent there. The bug was introduced by the proj insertion in PROV-002.
- **Evidence:** forward_from_embeddings: gate=adaptive_gate(embs), hop=hopfield(embs) on raw embs (410-411); live path uses full via _split_shared_private (394-398); populate stores model._get_features(...) raw features (433, 359-360).
- **Fix:** Route replayed embeddings through the same _split_shared_private transform used for live inputs before gate/hopfield/hgrn, OR store the post-projection full representation in the buffer. Keep replay and inference on the identical subgraph.
- **Fix at:** `code/m7_trimodal_run_provenance.py:409-414`, `code/m7_trimodal_run_provenance.py:427-444`
- _related: PROV-002, PROV-001_

#### `CMP-001` [HIGH] Near-miss negatives leak held-out numbers into training
`code/run_compositional_crossmodal.py:538-540` · correctness · status:live · effort:small · confidence:high
- **What:** Phase-(b) near-miss augmentation draws d2p = rng.integers(0,10) and appends a composite with label d1*10+d2p, with NO exclusion of held_numbers. assert_split_coverage guarantees every pos1 digit appears in HELD, so near-miss composites can equal held-out numbers. With shuffle_pairs=False the visual and audio composites at the same batch index share that label, so the held-out number is trained as a labeled cross-modal SupCon POSITIVE pair. Held-30 metrics are therefore contaminated and a positive compositional-generalization result cannot be distinguished from memorization - the exact failure mode the docstring (16-18) claims to exclude.
- **Evidence:** d2p = int(rng.integers(0, 10)); specs.append((d1, d2p)); labels.append(d1 * 10 + d2p)   # shares pos1
- **Fix:** Reject-sample d2p so that d1*10+d2p is not in held_numbers (or give near-miss items a label outside the evaluated number space), and assert generated near-miss labels are disjoint from held before training.
- **Fix at:** `code/run_compositional_crossmodal.py:534-540`

#### `ALGN-016` [HIGH] Invalid Mantel permutation null (permutes vectorized triangle, not objects)
`code/run_ordinal_anchor.py:147-168` · correctness · status:live · effort:small · confidence:medium
- **What:** _mantel_test permutes the vectorized upper triangle yv directly (line 161 perm = rng.permutation(len(yv)); yp = yv[perm]) and correlates. A valid Mantel test permutes the n underlying objects (rows AND columns of one RDM together) and recomputes the upper triangle; permuting the 45 distance entries independently breaks the distance-matrix dependency structure and yields an invalid permutation p-value. The observed r (line 157) is fine, but the reported vision_audio_rdm_p is not a valid Mantel p. This diagnostic underpins the 'RDMs uncorrelated (Mantel r, p)' claim.
- **Evidence:** lines 159-167 rng=np.random.default_rng(seed); for _ in range(permutations): perm=rng.permutation(len(yv)); yp=yv[perm]; rp=corrcoef(xv,yp); p_value=(count+1)/(permutations+1); called at line 197 on the 10x10 RDMs
- **Fix:** Permute the n=10 objects: draw a permutation of indices, apply it symmetrically to one RDM (rdm[np.ix_(perm,perm)]), then re-extract the upper triangle and correlate.
- **Fix at:** `code/run_ordinal_anchor.py:147-168`

#### `TST-016` [HIGH] ImportError on import: _build_digit_loaders is not exported by run_ordinal_anchor
`code/test_ordinal_anchor_smoke.py:14` · correctness · status:live · effort:trivial · confidence:high
- **What:** The test does `from run_ordinal_anchor import (..., _build_digit_loaders, ...)` (line 14), but run_ordinal_anchor neither defines nor re-exports _build_digit_loaders. That symbol is defined in run_correspondence_anchor_v2.py (line 67); run_ordinal_anchor imports several helpers from that module (lines 51-58: _hash_params, _load_cached_features, build_paired_feature_dataset, compute_held_out_metrics, run_procrustes, train_correspondence) but NOT _build_digit_loaders. run_ordinal_anchor only defines a differently-named _build_test_loaders (line 75). Result: `ImportError: cannot import name '_build_digit_loaders' from 'run_ordinal_anchor'`. The module raises before main() ever runs, so the __main__ guard is moot and this is the actual, live crash — earlier than the missing-data/checkpoint issues.
- **Evidence:** code/test_ordinal_anchor_smoke.py:14 imports _build_digit_loaders from run_ordinal_anchor; run_ordinal_anchor.py:51-58 import list omits it; _build_digit_loaders defined in run_correspondence_anchor_v2.py:67; run_ordinal_anchor.py:75 defines only _build_test_loaders.
- **Fix:** Import _build_digit_loaders from run_correspondence_anchor_v2 (where it lives), or add it to run_ordinal_anchor's re-export block (run_ordinal_anchor.py:51-58). Confirm the intended loader: run_ordinal_anchor exposes _build_test_loaders, which may be the one actually wanted.
- **Fix at:** `code/test_ordinal_anchor_smoke.py:11-19`, `code/run_ordinal_anchor.py:51-58`
- _related: TST-017, TST-018_


## MED (19)

#### `ARCH-018` [MED] forgetting_T1_T3 stores the incremental T2->T3 drop, not total T1->T3 forgetting
`code/archive/run_cl_confirmation.py:412` · correctness · status:archive · effort:trivial · confidence:medium
- **What:** forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 (line 412, self-labeled 'incremental T2->T3'). Its name implies total forgetting of T1 through T3, but total T1 forgetting is total_forgetting_T1_T3 = acc_t1_after - acc_t1_after_t3 (line 414). A downstream reader/aggregator that treats the column named forgetting_T1_T3 as total forgetting will understate/mislabel the headline forgetting metric — sensitive given RULE 5 (the forgetting claim must be reported honestly).
- **Evidence:** forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 line 412; total_forgetting_T1_T3 = acc_t1_after - acc_t1_after_t3 line 414; both written to CSV lines 437,439 and re-read by aggregate_confirmation_results metrics list lines 486-487.
- **Fix:** Rename the column to forgetting_T1_incremental_T2_T3 (or similar) so the incremental vs total distinction is unambiguous in the CSV consumed downstream. Retirement: note the ambiguity for anyone mining old cl_confirmation CSVs.
- **Fix at:** `code/archive/run_cl_confirmation.py:412`, `code/archive/run_cl_confirmation.py:437`
- _related: ARCH-019_

#### `DEAD-011` [MED] convert_mps.py's exact-string .replace() transforms silently no-op when m7_raw.py drifts
`code/convert_mps.py:70-108` · correctness · status:dead · effort:small · confidence:high
- **What:** The port relies on brittle literal .replace()/regex edits with no verification that any substitution matched. Two are already stale versus current m7_raw.py: (a) the EmbeddingReplayBuffer target expects a 'device=None' third line (convert_mps L91-93) but m7_raw L1774-1776 has 'device: str = 'cuda'', and the regex at L70-84 only matches bare device='cuda' (not the annotated form), so neither the regex nor the exact block fires and auto-detect is never inserted; (b) the ContinualLearner target expects '... device=None, lr=1e-3, weight_decay=1e-4, gradient_clip=1.0' (L102-103) but m7_raw L1883-1884 is '... device='cuda', replay_batch=32, replay_weight=0.3'. A regenerated m7_mps.py would silently keep a CUDA default and break on MPS/CPU, with no error raised. This is why regeneration is unsafe (DEAD-004).
- **Evidence:** convert_mps L70-84 regex; L91-99 old_er_init; L102-108 old_cl_init; vs m7_raw L1774-1776 (device: str = 'cuda') and L1883-1884 (device='cuda', replay_batch=32, replay_weight=0.3).
- **Fix:** Delete. If a similar tool is ever needed, assert each replacement changed the text (count matches) and fail loudly on zero matches; never patch a moving source with literal strings.
- **Fix at:** `code/convert_mps.py:70-108`
- _related: DEAD-004, DEAD-009_

#### `CLE-010` [MED] Misleading 'transfer did not survive' verdict when post_T2 is simply absent
`code/eval_transfer_fixed.py:126-137` · correctness · status:live · effort:small · confidence:high
- **What:** When df is non-empty (e.g. only post_T3 present) but has no post_T2 row, the ternary falls to `else 0`, so post_t2_v2a=0 and the code prints the failure branch: 'Transfer DID NOT survive. V2A = 0.0% (near chance).' This reports a fabricated 0.0% and a scientific conclusion for a condition that was never evaluated -- absence of data is silently rendered as a negative result.
- **Evidence:** Line 126 `else 0`; lines 134-137 print the failure verdict using post_t2_v2a.
- **Fix:** Distinguish 'not evaluated' (None) from a measured low value; only emit a verdict when post_T2 was actually scored.
- **Fix at:** `code/eval_transfer_fixed.py:126-137`
- _related: CLE-009_

#### `FIG-022` [MED] torch.load of a missing checkpoint crashes the only run path; CWD-relative path; name lacks seed
`code/final_evaluation.py:207` · correctness · status:live · effort:small · confidence:high
- **What:** main() calls torch.load('ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt', map_location='cpu') (207). The repo has NO checkpoints, so the script raises FileNotFoundError immediately - the entire evaluation depends on this file. The path is CWD-relative (no './'; resolves only from code/, and note it uses 'ICONS_M7/...' whereas the figure scripts use './ICONS_M7'). The filename 'final_nmnist_nmnist_replay.pt' also lacks the seed field of the documented pattern final_{task}_{task}_seed{seed}_{condition}.pt. Unlike generate_figures_3seed.py (which sys.exits with guidance on a missing CSV), this gives a bare traceback. Minor: torch.load is called without weights_only=; torch 2.11 (pinned) defaults weights_only=True (secure-by-default), which also means a non-tensor-only checkpoint would fail here.
- **Evidence:** torch.load('ICONS_M7/results/checkpoints/final_nmnist_nmnist_replay.pt', map_location='cpu')
- **Fix:** Check the path exists and emit an informative message (mirror the CSV-missing pattern); make the path './ICONS_M7/...'; confirm the intended checkpoint filename/seed convention; pass weights_only explicitly.
- **Fix at:** `code/final_evaluation.py:207`
- _related: FIG-009_

#### `FIG-023` [MED] collect_stratified silently under-collects if a class has <200 samples; assumes batch_size=1; KeyError if label>=10
`code/final_evaluation.py:88-99` · correctness · status:live · effort:small · confidence:high
- **What:** collect_stratified (88-99) fills up to per_class (200) samples/class, breaking early once all classes are satisfied (97-98). If any class has fewer than per_class samples in the loader, the all(...) condition never becomes True and the loop simply exhausts the finite loader and returns fewer samples than requested - with NO error - while main() still prints '200/class' (200/223). It also assumes batch_size=1 via int(y.item()) (92) - y.item() raises if a batch has >1 element (batch_size=1 is set at 221, so this is a coupled assumption). counts is keyed 0..9 (90); a label >=10 would KeyError. Not an infinite loop (the for-loop is bounded by the finite loader).
- **Evidence:** counts = {i: 0 for i in range(10)}; break when all(c >= per_class ...)
- **Fix:** After the loop, assert each class reached per_class (or report actual per-class counts); derive the class set from the data rather than hardcoding range(10); document the batch_size=1 requirement.
- **Fix at:** `code/final_evaluation.py:88-99`

#### `RUN-016` [MED] SupervisedContrastiveLoss includes the self/diagonal term in both the positive mask and the denominator
`code/m7_trimodal_run.py:190-199` · correctness · status:live · effort:small · confidence:medium
- **What:** The positive mask is 'labels==labels' which is 1 on the diagonal, and the log-prob denominator sums over all j including j==i, with no logits_mask to remove self-contrast. Standard SupCon (Khosla et al.) excludes the anchor itself from both positives and denominator. Including self gives every anchor a trivial always-satisfied positive (its own normalized similarity, the largest term after max-subtraction), diluting the contrastive gradient and biasing the loss, most visibly for classes appearing once per batch.
- **Evidence:** line 193 'mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()'; lines 196-198 build exp_logits and mean_log_prob without masking the diagonal
- **Fix:** Build a logits_mask that removes self from the denominator and from mask before the mean, matching the canonical SupCon formulation; this will move the loss and thus results, so flag it as a deliberate numeric change.
- **Fix at:** `code/m7_trimodal_run.py:190-199`

#### `PROV-013` [MED] The 'per-seed' CSV actually stores the entire accumulated master table, not that seed's row
`code/m7_trimodal_run_provenance.py:858-860` · correctness · status:live · effort:trivial · confidence:high
- **What:** save_seed_result reassigns df to the full master (existing rows + new row) at 855, writes the master (856), then writes df again to cl_results_seed{seed}_{condition}.csv (859-860). Because df is the concatenated master, the file labelled per-seed contains EVERY seed/condition accumulated so far, not one seed. The comment 'Also save per-seed CSV for clean aggregation' (858) is therefore false, and downstream code expecting one row per file gets a growing full history mislabeled by whichever seed happened to be written last.
- **Evidence:** df = pd.concat([existing, df], ignore_index=True) (855); df.to_csv(csv_path) master (856); seed_csv = ... (859); df.to_csv(seed_csv) writes the SAME master df (860).
- **Fix:** Write only the current seed's row to the per-seed file: build a fresh single-row frame (e.g. pd.DataFrame([result])) for seed_csv instead of reusing the concatenated df.
- **Fix at:** `code/m7_trimodal_run_provenance.py:849-860`
- _related: PROV-006_

#### `PROV-014` [MED] SupCon losses count the anchor itself as a positive (diagonal not masked); the CrossModal comment claiming self contributes zero is wrong
`code/m7_trimodal_run_provenance.py:156-203` · correctness · status:live · effort:small · confidence:medium
- **What:** SupervisedContrastiveLoss.forward builds mask = eq(labels_i, labels_j) which is True on the diagonal (159), and mean_log_prob = (mask*log_prob).sum(1)/(mask.sum(1)+eps) (164), so each anchor's own similarity is counted in both numerator and denominator of the positive average. Standard SupCon (Khosla et al. 2020) excludes the anchor from its positive set P(i); including it biases the loss toward the trivially-satisfied self term and dilutes the true-positive gradient. This loss is ACTIVE in the default run (scl_weight=0.1, use_scl=True, 608-609). CrossModalSupConLoss repeats the same self-as-positive construction (192,198,202) and its comment (201) asserts 'self gives 0 contribution since mask*log_prob on diagonal is log(1)=0' — that is incorrect: the diagonal log_prob is log-softmax of the self position (<=0), not 0. The CrossModal loss only runs under align_cross_modal (default off), but the misleading comment and the shared bug pattern should be fixed.
- **Evidence:** SCL: mask includes diagonal (159); mean over positives incl. self (164); used in train_task (585,609). XMSCL: pos_mask=mask (198), denom incl. self (200), wrong comment (201).
- **Fix:** Mask the diagonal out of the positive set (set self-similarity to -inf before softmax and zero it in pos_mask/pos-count) to match Khosla SupCon; correct or delete the false comment at 201. Numerics-affecting: flag before/after if applied.
- **Fix at:** `code/m7_trimodal_run_provenance.py:156-165`, `code/m7_trimodal_run_provenance.py:181-203`, `code/m7_trimodal_run.py:186-199`

#### `DEAD-014` [MED] modify_notebook.py sets a cell's source to a list of single characters (list(str))
`code/modify_notebook.py:558` · correctness · status:dead · effort:trivial · confidence:high
- **What:** src_text is a STRING (built at L207 as ''.join(src) and mutated via .replace()). At L558 the modified source is stored as cells[cl_cell_idx]['source'] = list(src_text), which splits the string into a list of individual CHARACTERS, not lines. The resulting notebook cell 'source' would be thousands of one-character strings - malformed for nbformat and unusable. Unreachable today because L9 crashes first (DEAD-012), but a genuine logic defect unique to this file.
- **Evidence:** L207 src_text = ''.join(src); L558 cells[cl_cell_idx]['source'] = list(src_text).
- **Fix:** Use src_text.splitlines(keepends=True) (or store the string directly) instead of list(...). Moot if the file is deleted.
- **Fix at:** `code/modify_notebook.py:558`
- _related: DEAD-012, DEAD-013_

#### `ABL-001` [MED] _save_csv opens in 'w' mode, silently discarding prior ablation rows on a narrower re-run
`code/run_ablation.py:461-470` · correctness · status:live · effort:small · confidence:medium
- **What:** run_ablation.py writes results with _save_csv opening the output CSV in write ('w') mode, so re-running with a narrower --datasets/--models/--seeds selection overwrites and discards all previously computed ablation rows instead of merging. This is the inverse of the append-without-dedup hazard elsewhere: here legitimate prior results are silently lost. (Added by the completeness-critic pass. run_ablation.py is otherwise the one script that seeds correctly with [42,123,456], measures real spike metrics via SpikeRecorder/measure_spike_metrics, and has a __main__ guard - it wires to the legacy bi-modal package, not M7.)
- **Evidence:** _save_csv opens the CSV in 'w' (overwrite) mode at ~461-470; no merge with existing rows keyed by (dataset,model,seed)
- **Fix:** Merge new rows with any existing CSV keyed by (dataset,model,seed) before writing (upsert), or append then drop_duplicates on that key; do not blindly overwrite.
- **Fix at:** `code/run_ablation.py:461-470`

#### `CLE-021` [MED] forgetting_T1_T3 is the incremental post-T2->post-T3 drop, not total T1 forgetting the name implies
`code/run_cl_confirmation_fixed.py:327` · correctness · status:live · effort:small · confidence:high
- **What:** forgetting_T1_T3 = acc_t1_after_t2 - acc_t1_after_t3 (327) measures only the T1 drop that occurs during T3 (between post-T2 and post-T3). The name reads as total T1->T3 forgetting, which is actually a separate column total_forgetting_T1_T3 = acc_t1_after - acc_t1_after_t3 (329). This is a real mislabel risk: a summary or paper could quote 'T1_T3 forgetting' and use the incremental value. Identical construction in run_cl_rehearsal.py:336.
- **Evidence:** Line 327 forgetting_t1_t3 = acc_t1_after_t2 - acc_t1_after_t3; line 329 total_forgetting_t1_t3 = acc_t1_after - acc_t1_after_t3.
- **Fix:** Rename to forgetting_T1_during_T3 (incremental) or make the pair unambiguous in a data dictionary; ensure downstream analysis and paper text pick the intended one. Mirror in run_cl_rehearsal.py.
- **Fix at:** `code/run_cl_confirmation_fixed.py:326-329`, `code/run_cl_rehearsal.py:335-338`
- _related: CLE-022_

#### `CMP-004` [MED] Per-digit horizontal jitter is inert for the right digit and boundary-violating for the left
`code/run_compositional_crossmodal.py:206-225` · correctness · status:live · effort:small · confidence:medium
- **What:** Per-digit jitter is far weaker than the boundary comment implies. The RIGHT digit is fully inert: with RIGHT_X0=40 and the min/max clamp, xr==40 for every jitter in [-3,3]. The LEFT digit does jitter but only over xl in {0,1,2,3} (LEFT_X0=0, JITTER=3), spanning cols [3,37); it clips ~3px into the 6px inter-digit buffer (cols 34-39) but never reaches the right region (col>=40). The code's boundary comment is technically violated, but by ~3px, not by a large cross-region slide.
- **Evidence:** RIGHT_X0=40 + clamp => xr==40 for all jitter; LEFT_X0=0, JITTER=3 => xl in {0..3}, span [3,37), max 3px into the 6px buffer
- **Fix:** Make the augmentation actually jitter both digits within intended bounds (the right digit is pinned; the left moves <=3px); if a buffer-crossing near-miss is intended, widen the left range deliberately and assert it.
- **Fix at:** `code/run_compositional_crossmodal.py:206-215`, `code/run_compositional_crossmodal.py:217-225`

#### `CMP-005` [MED] 'shuffled' negative control is defined inconsistently across phases
`code/run_compositional_crossmodal.py:511-556` · correctness · status:live · effort:small · confidence:medium
- **What:** Phase (a) applies ONE permutation to a single label vector reused for both modalities (lab -> cat([lab,lab,lab,lab])), preserving cross-modal pairing while breaking the label-to-data correspondence. Phase (b) shuffles the AUDIO labels with an INDEPENDENT permutation (lab_a) while visual keeps lab, which breaks the cross-modal pairing itself. The negative control therefore means two different things across the curriculum, muddying its interpretation as a baseline.
- **Evidence:** phase a: if shuffle_pairs: lab = lab[torch.randperm(...)]  vs  phase b: lab_a = lab[torch.randperm(...)]
- **Fix:** Choose a single shuffle semantics (e.g. permute the cross-modal correspondence, or permute labels-vs-data) and apply it identically in both phases.
- **Fix at:** `code/run_compositional_crossmodal.py:511-519`, `code/run_compositional_crossmodal.py:547-556`

#### `CMP-007` [MED] assert_split_coverage can fail for arbitrary seeds and is then silently swallowed per-seed
`code/run_compositional_crossmodal.py:265-281` · correctness · status:live · effort:small · confidence:medium
- **What:** assert_split_coverage requires all ten digits to appear in pos1 AND pos2 within BOTH the 70-number train set and the 30-number held set. A random 30/70 permutation split does not guarantee full per-position coverage for every seed. On failure the assert raises, and main's broad try/except prints a traceback and drops that seed, so a nominal 3-seed run can silently become fewer seeds.
- **Evidence:** assert p1 == set(range(10)), f'{name} missing pos1 digits: ...'  (raised inside run_one_model, caught by main)
- **Fix:** Resample the split in make_split until coverage holds (loop), or restrict to seeds verified to satisfy coverage; do not rely on the broad except to hide coverage failures.
- **Fix at:** `code/run_compositional_crossmodal.py:265-281`, `code/run_compositional_crossmodal.py:256-262`, `code/run_compositional_crossmodal.py:1195-1201`

#### `JNT-004` [MED] Leakage assert uses a non-element-wise idiom that does not mean what it reads
`code/run_joint_magnitude_ablation.py:488` · correctness · status:live · effort:trivial · confidence:high
- **What:** The leakage guard checks 'any element >=6 AND any element <=9' rather than 'any element in [6,9]'. It happens to catch a leak for the 0-9 label range but is the wrong idiom and can false-pass other ranges.
- **Evidence:** assert not ((labels >= 6).any() and (labels <= 9).any()) at 488 (duplicated at 776, 778). Because (labels>=6).any() and (labels<=9).any() are evaluated independently, the intended element-wise membership test ((labels>=6)&(labels<=9)).any() is not what runs. For labels in 0-9 a label 8 trips both terms so a leak is caught, but the idiom is incorrect and fragile.
- **Fix:** Replace all three with element-wise: assert not ((labels >= 6) & (labels <= 9)).any().
- **Fix at:** `code/run_joint_magnitude_ablation.py:488`, `code/run_joint_magnitude_ablation.py:776`, `code/run_joint_magnitude_ablation.py:778`
- _related: JNT-032_

#### `JNT-020` [MED] SHD train cache uses identity labels (%10) but test cache stores raw 0-19; reconciled only post-hoc
`code/run_joint_shared_encoder.py:414-472` · correctness · status:live · effort:small · confidence:high
- **What:** The train and test SHD caches use different label conventions under the same filename convention; correctness depends on a single post-hoc %10 that is easy to remove or overlook.
- **Evidence:** Train cache stores int(l) % 10 (414); test cache stores raw int(l) (472). run_one_seed fixes only the test side later: aud_lbl_all = aud_lbl_all % 10 (848). Without that %10 the HELD_OUT mask (6,7,8,9) drops German 6-9 (labels 16-19) and TRAIN mask drops German 0-5; it is load-bearing and asymmetric with the train convention.
- **Fix:** Apply %10 at cache-build time for BOTH train and test (line 472) so the on-disk convention is uniform; then the 848 %10 becomes a harmless no-op that can be removed.
- **Fix at:** `code/run_joint_shared_encoder.py:414`, `code/run_joint_shared_encoder.py:472`, `code/run_joint_shared_encoder.py:848`

#### `JNT-024` [MED] PairedDigitDatasetFiltered encodes flat index by class VALUE but decodes by positional INDEX
`code/run_joint_shared_encoder.py:182-196` · correctness · status:live · effort:small · confidence:medium
- **What:** The flat-index encode/decode is asymmetric: it is only correct when the class list is exactly 0..N-1, and silently mis-decodes for any other class list.
- **Evidence:** Encode uses the class value: int(self.classes[c] + s*self.num_classes) (186). Decode uses the position: c = flat_idx % self.num_classes; s = flat_idx // self.num_classes (193-196), then indexes self.visual_indices[c] keyed by class value (203). Correct only for classes==(0,1,2,3,4,5); always called with TRAIN_CLASSES so it works today but is a latent bug for non-zero-based/non-contiguous class lists.
- **Fix:** Encode with the positional index (c_index + s*num_classes) and map to the class value on decode via self.classes[c_index]; or assert self.classes == tuple(range(num_classes)).
- **Fix at:** `code/run_joint_shared_encoder.py:182-196`

#### `TST-010` [MED] Flaky assertion: cross-modal loss must strictly decrease over just 2 epochs on near-noise fake data
`code/test_cross_modal_alignment.py:142-143` · correctness · status:live · effort:small · confidence:high
- **What:** Line 142 asserts history['xmscl_loss'][1] < history['xmscl_loss'][0] — i.e. the CrossModalSupCon loss must strictly drop between epoch 1 and epoch 2. The data is FakePairedDataset, where per-class signal is c*0.05 added onto randn*0.1 noise (lines 37-39), i.e. mostly noise; optimization uses AdamW with CosineAnnealingLR(T_max=num_epochs=2) (m7_trimodal_run_provenance.py:702). Over only two epochs on near-noise data a strict monotonic decrease is not guaranteed, so this test can fail spuriously on some seeds/runs — a false alarm rather than a real regression signal.
- **Evidence:** code/test_cross_modal_alignment.py:142-143 strict `<` assertion; FakePairedDataset signal c*0.05 over 0.1 noise at lines 37-39; num_epochs=2 at line 132; CosineAnnealingLR(T_max=num_epochs) in align_cross_modal (m7_trimodal_run_provenance.py:702).
- **Fix:** Make the check robust: assert loss[-1] <= loss[0] with a tolerance, or assert a downward trend over more epochs, or compare mean of last-half vs first-half. Seed the run and/or increase epochs so the trend is reliable.
- **Fix at:** `code/test_cross_modal_alignment.py:130-143`

#### `TST-004` [MED] The 'key assertion' is not an assertion — it is a non-enforcing if/else that always passes
`code/test_provenance_smoke.py:89-93` · correctness · status:live · effort:trivial · confidence:high
- **What:** The comment on lines 87-88 calls the near-chance zero-shot check 'The key assertion', but lines 89-93 implement it as an `if abs(...) < 3.0: print(ok) else: print(warning)` — it never fails the test. The module still prints 'SMOKE TEST PASSED' (line 109) regardless of whether SHD zero-shot is near chance. So the single most important property this smoke test claims to verify (no spurious cross-modal signal / no data leakage) is not enforced. Compounding it, the 3.0 pp band is tighter than the noise floor of a ~40-sample evaluate() (DummyLoader(10,20), line 53), so the warning branch fires often yet nothing fails.
- **Evidence:** code/test_provenance_smoke.py:87-88 comment 'The key assertion'; lines 89-93 if/else that only prints; line 109 prints PASSED unconditionally; evaluate() sample count ~40 from dummy_shd = DummyLoader(10, 20) at line 53.
- **Fix:** Promote the check to a real assertion (assert abs(acc_shd_zero - chance_shd) < TOL) with a defensible TOL and larger sample count, or explicitly document why it is only a warning. Do not label a warning as the 'key assertion'.
- **Fix at:** `code/test_provenance_smoke.py:87-93`
- _related: TST-003_


## LOW (14)

#### `FIG-025` [LOW] argmax over all-zero head spike counts silently predicts class 0
`code/final_evaluation.py:183-184` · correctness · status:live · effort:small · confidence:high
- **What:** preds = np.argmax(head_counts, axis=1) (183). For any sample with zero total head spikes, head_counts row is all zeros and np.argmax returns index 0 - the sample is silently classified as class 0, biasing accuracy/confusion toward class 0. zero_count (182) records how many such samples occur and the report prints the total (291), but the predictions still default to 0 rather than being marked no-prediction. The HONEST STATEMENT relies on 'Confirms 0 all-zero' - if that count is ever nonzero, accuracy already silently absorbed those as class-0 guesses.
- **Evidence:** preds = np.argmax(head_counts, axis=1) with zero_count tracked separately
- **Fix:** Mark all-zero samples as an explicit no-prediction sentinel (e.g. -1) and exclude/flag them in accuracy and confusion, instead of letting argmax default to class 0.
- **Fix at:** `code/final_evaluation.py:181-184`
- _related: FIG-026_

#### `FIG-013` [LOW] Figure 4 'Raw dense frames' bar uses DVS pooled 34x34, not raw 128x128
`code/generate_figures_3seed.py:294` · correctness · status:live · effort:trivial · confidence:medium
- **What:** FRAME_ELEMS['dvs'] = 25*2*34*34 (294) uses the POOLED 34x34 resolution (comment: 'pooled 128->34 to match the visual backbone'), but the bar it feeds is labeled 'Raw dense frames (50/class)' (313). DVS-Gesture raw frames are 2x128x128 (per CLAUDE.md), ~14x more elements. So the 'raw' baseline for DVS is understated, which understates the raw total and hence the compression ratio (322) - conservative/anti-flattering, not an inflating error - but internally inconsistent with the 'Raw dense frames' label.
- **Evidence:** 'dvs': 25 * 2 * 34 * 34 under a bar labeled 'Raw dense frames'
- **Fix:** Either compute the DVS raw-frame term at native 128x128 to match the 'raw' label, or relabel the bar to clarify it uses the pooled resolution the model consumes.
- **Fix at:** `code/generate_figures_3seed.py:294`, `code/generate_figures_3seed.py:313`
- _related: FIG-012_

#### `FIG-016` [LOW] Summary-row mask at line 65 is fragile: KeyError if 'step' absent; nullable-boolean mask risk under pandas 3.0 string dtype
`code/generate_figures_3seed.py:65` · correctness · status:live · effort:small · confidence:low
- **What:** summary = df[df['step'].isna() | (df['step'].astype(str).str.strip()=='')].copy() (65). Two fragilities: (1) if the CSV lacks a 'step' column, df['step'] raises KeyError with no friendly message - the dual-schema contract is assumed, not checked. (2) Under pandas 3.0 (pinned 3.0.2), if 'step' is inferred as the new string dtype rather than float, str.strip()=='' yields a nullable BooleanArray; OR-ing/masking that with possible <NA> can raise 'Cannot mask with non-boolean array containing NA'. In practice blank CSV cells usually parse to float NaN (safe path), so this is conditional - confidence low - but worth a guard.
- **Evidence:** df[df['step'].isna() | (df['step'].astype(str).str.strip() == '')]
- **Fix:** Guard for the 'step' column's presence and coerce the mask to plain numpy bool (fillna(False) / .to_numpy(dtype=bool)) before indexing.
- **Fix at:** `code/generate_figures_3seed.py:65`
- _related: FIG-014_

#### `RUN-033` [LOW] events_to_dense: binary occupancy, redundant max_time guard, and divide-by-zero when all timestamps are equal
`code/m7_trimodal_run.py:138-148` · correctness · status:live · effort:trivial · confidence:medium
- **What:** The dense encoding sets dense[t,c]=1.0, so repeated events at the same (t,c) do not accumulate (occupancy, not count). max_time is guarded by 'if len(events)>0 else 1' but the real risk is max_time==0 (all events at t=0), which makes 'events[t]/max_time' divide by zero and yields invalid indices after astype(int). The guard checks length, not max_time.
- **Evidence:** line 142 'max_time = events["t"].max() if len(events) > 0 else 1'; line 143 divides by max_time; line 147 'dense[t, c] = 1.0'
- **Fix:** Guard max_time with 'max(events["t"].max(), 1)', and decide deliberately between occupancy and event-count encoding.
- **Fix at:** `code/m7_trimodal_run.py:138-148`
- _related: RUN-017_

#### `PROV-021` [LOW] EmbeddingReplayBuffer default device picks mps/cpu only, ignoring CUDA
`code/m7_trimodal_run_provenance.py:418-423` · correctness · status:live · effort:trivial · confidence:medium
- **What:** When device is None the buffer defaults to mps if available else cpu (419-420), never cuda. run_one_seed passes an explicit device (975) so the default is not hit there, but any of the 19 importers that construct EmbeddingReplayBuffer without a device on a CUDA host would place replay embeddings on CPU while the model is on CUDA, causing a device-mismatch error in forward_from_embeddings.
- **Evidence:** device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') (420); explicit device passed at 975.
- **Fix:** Mirror get_device precedence (mps -> cuda -> cpu) in the buffer default, or make device a required argument.
- **Fix at:** `code/m7_trimodal_run_provenance.py:418-423`

#### `PROV-024` [LOW] _events_to_dense divides by max_time which can be zero, producing NaN time indices
`code/m7_trimodal_run_provenance.py:1200-1210` · correctness · status:live · effort:trivial · confidence:medium
- **What:** In _events_to_dense, time_indices = (events['t'] / max_time * (time_bins-1)).astype(int) (1205) where max_time = events['t'].max() (1204). If all events share t=0 (or a single event at t=0), max_time is 0 and the division yields inf/NaN, and astype(int) on NaN is undefined behaviour. The len(events)>0 guard (1203) does not cover the all-zero-timestamp case, and the inner 'if len(events)>0 else 1' (1204) is redundant since it is already inside the len>0 block.
- **Evidence:** max_time = events['t'].max() if len(events) > 0 else 1 (1204); time_indices = (events['t'] / max_time * (time_bins - 1)).astype(int) (1205).
- **Fix:** Guard with max_time = max(events['t'].max(), 1) (or skip the sample) so the divisor is never zero; drop the redundant inner conditional.
- **Fix at:** `code/m7_trimodal_run_provenance.py:1200-1210`

#### `CLE-018` [LOW] _is_complete_result treats a header-only CSV as complete (false skip)
`code/run_cl_confirmation_fixed.py:53-60` · correctness · status:live · effort:trivial · confidence:medium
- **What:** _is_complete_result checks only that all EXPECTED_COLUMNS are present as columns (58), not that any row exists. A CSV written with a header but zero data rows (or a truncated/failed prior write that still has the header) satisfies the check, so run_confirmation_cell prints 'SKIP' (186) and returns {} -- the cell is silently skipped despite having produced no result.
- **Evidence:** Line 57-58 `df = pd.read_csv(csv_path); return all(col in df.columns ...)`; line 185-187 skip logic.
- **Fix:** Also require len(df) >= 1 (and ideally that the single expected row's key fields are non-null) before treating the result as complete.
- **Fix at:** `code/run_cl_confirmation_fixed.py:53-60`

#### `CLE-028` [LOW] Backbone-drift check indexes sd_t3[k] by sd_t1's keys without a membership check
`code/run_cl_rehearsal.py:324-328` · correctness · status:live · effort:trivial · confidence:low
- **What:** The drift loop iterates keys of sd_t1 and computes (sd_t1[k] - sd_t3[k]).norm() (326-327) with no check that k exists in sd_t3. If the two checkpoints have different key sets (e.g. arch change between saves), this raises KeyError. Low likelihood because both come from the same model class, hence low confidence, but the sibling phase1_sanity_check.compare_state_dicts (65-66) does guard for the missing key -- an inconsistency worth aligning.
- **Evidence:** Lines 324-328; contrast phase1_sanity_check.py:65-66 which handles `k not in sd2`.
- **Fix:** Skip or record keys missing from sd_t3 (mirror compare_state_dicts) before subtracting.
- **Fix at:** `code/run_cl_rehearsal.py:324-328`

#### `CMP-010` [LOW] Eval composites use TRAIN-set exemplars and differ across models
`code/run_compositional_crossmodal.py:825-849` · correctness · status:live · effort:medium · confidence:medium
- **What:** CompositeBuilder samples digit exemplars from cached['nmnist_train']/['shd_train'], and eval reuses the same builder whose RNG (seed+7) has been advanced by a model-specific amount of training (compositional 2 phases, holistic 1, untrained 0). So the four models are evaluated on different exemplar draws, cross-model comparison is not on identical inputs, and no unseen-EXEMPLAR generalization is tested (only unseen combinations).
- **Evidence:** builder = CompositeBuilder(cached, np.random.default_rng(seed + 7)); ... ev = extract_2digit(model, builder, device, ...)
- **Fix:** Use a separate fixed-RNG builder for evaluation, independent of training draws, and snapshot the eval spec list so every model sees identical inputs; consider held-out exemplars for a stronger test.
- **Fix at:** `code/run_compositional_crossmodal.py:830`, `code/run_compositional_crossmodal.py:848-849`, `code/run_compositional_crossmodal.py:187-198`

#### `ALGN-011` [LOW] _events_to_dense: dead else-branch and divide-by-zero when all events at t=0
`code/run_correspondence_anchor_v2.py:93-103` · correctness · status:live · effort:trivial · confidence:medium
- **What:** Inside if len(events)>0, max_time = events['t'].max() if len(events)>0 else 1 has a dead else (already guarded). If every event has t==0 (max_time==0), time_indices = (events['t']/max_time*(time_bins-1)).astype(int) divides by zero -> inf/nan -> astype(int) is undefined behaviour. Copy exists in run_ordinal_anchor.py:92 and run_digit_only_alignment.py:122.
- **Evidence:** lines 97-98 max_time = events['t'].max() if len(events)>0 else 1; time_indices=(events['t']/max_time*(time_bins-1)).astype(int)
- **Fix:** Guard max_time with max(int(events['t'].max()),1); remove the redundant conditional.
- **Fix at:** `code/run_correspondence_anchor_v2.py:97-98`, `code/run_ordinal_anchor.py:97-98`, `code/run_digit_only_alignment.py:127-128`
- _related: ALGN-008_

#### `ALGN-031` [LOW] aggregate_digit_results silently drops cells missing any EXPECTED_COLUMN
`code/run_digit_only_alignment.py:365-374` · correctness · status:live · effort:trivial · confidence:medium
- **What:** aggregate_digit_results skips any per-cell CSV that lacks all EXPECTED_COLUMNS (368-374, printing 'Skipping ... incomplete columns'). A cell written by an older schema (or partially, though atomic writes make partials unlikely) is silently excluded from the summary and Wilcoxon, so n can drop without the headline changing. Note the positive: because each cell is one atomically-overwritten file (_atomic_write_csv, 197-201), reruns do NOT accumulate duplicate rows here - unlike run_cross_modal_alignment_study (ALGN-022).
- **Evidence:** lines 368-374 if all(col in df.columns for col in EXPECTED_COLUMNS): dfs.append(df) else print('Skipping ... incomplete columns'); EXPECTED_COLUMNS defined 73-82; _atomic_write_csv 197-201
- **Fix:** Log skipped cells at WARN with the missing column names and surface the final n per condition so silent shrinkage is visible.
- **Fix at:** `code/run_digit_only_alignment.py:364-378`
- _related: ALGN-022_

#### `JNT-030` [LOW] best_state can be undefined -> NameError if first-epoch val loss is NaN or epochs==0
`code/run_joint_shared_encoder.py:805-839` · correctness · status:live · effort:trivial · confidence:medium
- **What:** The best-model restore assumes the improvement branch ran at least once; a NaN first-epoch validation loss (possible in SNN training) or epochs==0 leaves best_state unbound.
- **Evidence:** best_val_loss=inf (805); best_state assigned only inside 'if val_stats[loss] < best_val_loss' (828-831); model.load_state_dict(best_state) at 839. NaN < inf is False, so a NaN first epoch never sets best_state.
- **Fix:** Initialize best_state = copy.deepcopy(model.state_dict()) before the training loop.
- **Fix at:** `code/run_joint_shared_encoder.py:805-839`

#### `JNT-031` [LOW] Shuffled-control permutation index built on CPU while labels are on the MPS device
`code/run_joint_shared_encoder.py:512-514` · correctness · status:live · effort:trivial · confidence:low
- **What:** The label-permutation index is created on the default (CPU) device and used to index a device tensor, which can raise on MPS/CUDA depending on the torch version and would crash the unguarded shuffled-control run.
- **Evidence:** perm = torch.randperm(all_labels.size(0)) (513, and 560 in validate) with all_labels on DEVICE (mps); all_labels[perm] indexes across devices. The shuffled control is called unguarded in main (930), so a device error there aborts before summaries are written. torch is pinned at 2.11.0.
- **Fix:** Create the permutation on the tensor's device: torch.randperm(n, device=all_labels.device); optionally guard the shuffled-control call.
- **Fix at:** `code/run_joint_shared_encoder.py:512-514`, `code/run_joint_shared_encoder.py:559-561`
- _related: JNT-001_

#### `TST-013` [LOW] Orthogonal-penalty test uses a raw slice, not the model's learned proj_private
`code/test_cross_modal_alignment.py:102-106` · correctness · status:live · effort:small · confidence:medium
- **What:** private_vis is computed by slicing full_vis[:, cfg.align_shared_dim:] (line 102), i.e. the tail dimensions of the full feature, whereas the model's actual private subspace is a learned nn.Linear projection self.proj_private (m7_trimodal_run_provenance.py:339). So the penalty exercised here is on a different quantity than the one the model uses in training, and (given the vacuous >=0 assertion, TST-012) the test would not catch a bug in proj_private wiring. It validates orthogonal_penalty's arithmetic but not the model's shared/private split.
- **Evidence:** code/test_cross_modal_alignment.py:102 `private_vis = full_vis[:, cfg.align_shared_dim:]`; model's proj_private is a learned Linear at m7_trimodal_run_provenance.py:339.
- **Fix:** Obtain private from the model's proj_private path (or from the model's own split helper) so the test reflects what training actually uses, and add a meaningful assertion on it.
- **Fix at:** `code/test_cross_modal_alignment.py:102-106`
- _related: TST-012_


## INFO (2)

#### `ALGN-013` [INFO] Retrieval candidate-set sizes differ across analyses; magic verdict threshold
`code/run_correspondence_anchor_v2.py:419-435` · correctness · status:live · effort:small · confidence:medium
- **What:** compute_held_out_metrics v2a/a2v retrieval ranks among only the 4 held-out classes (chance 0.25, lines 419-435), while run_procrustes ranks among all 10 classes (chance 0.1, all_classes at line 489). The two retrieval numbers are reported side by side in report.txt yet are not comparable. The verdict threshold '0.25 + 0.1' (line 592) is a magic number; with only 4 held-out classes the retrieval statistic is quantised to {0,.25,.5,.75,1} and is high-variance across seeds.
- **Evidence:** lines 419-435 (4-way retrieval); line 489 all_classes = list(train_classes)+list(held_out_classes) (10-way); line 592 held_out['held_out_v2a_retrieval'] > 0.25 + 0.1
- **Fix:** Fix candidate sets to a documented common protocol and name the chance level explicitly; replace 0.25+0.1 with a named constant.
- **Fix at:** `code/run_correspondence_anchor_v2.py:419-435`, `code/run_correspondence_anchor_v2.py:489-508`, `code/run_correspondence_anchor_v2.py:592`

#### `CMP-027` [INFO] The 2x2 SHARED-vs-SEPARATE design correctly isolates classifier-induced z-alignment (positive note)
`code/run_joint_diagnostic.py:97-127` · correctness · status:live · effort:trivial · confidence:high
- **What:** Retrieval is measured on encoder embeddings z (per_sample_retrieval and compute_held_out_metrics both call model.embed), not on classifier logits. In the SHARED cell the phase-2 within-modality CE backpropagates through the single shared_classifier for BOTH modalities, so the shared head induces cross-modal alignment of the 6-9 embeddings with no cross-modal supervision - a valid positive control for the leak mechanism, absent in the SEPARATE cell. The new per-sample retrieval with TRAIN prototypes (282-333) is a genuine improvement over the transductive quantized 4-class metric it replaces. Recorded as an integrity-positive; no fix.
- **Evidence:** SharedHeadJointModel.forward uses one shared_classifier; train_model_diag Pass 2 (204-213) CE through it for both modalities; retrieval uses model.embed (z)
- **Fix:** No action required.
- **Fix at:** `code/run_joint_diagnostic.py:97-127`, `code/run_joint_diagnostic.py:204-213`, `code/run_joint_diagnostic.py:282-333`
