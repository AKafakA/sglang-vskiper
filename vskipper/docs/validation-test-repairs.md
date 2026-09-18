# Reference validation: test maintenance

The frozen control is `461f0af45e457391515cf2e7bb50f92c0fb77b5e`.
Its unmodified runtime passed the serving import sweep (25 modules in two
postures). The first complete control pytest run recorded 752 passes,
71 failures and one guard-page skip. The repairs below align test fixtures
with the control's existing interfaces; runtime and numerical-analysis
implementations, input data and numerical tolerances are unchanged.

| Test family | Original failures | Evidence and repair |
| --- | ---: | --- |
| Module import safety | 1 | Static precheck did not bind tuple loop targets within the loop body. Add lexical binding and positive/negative regression cases; retain the real import sweep. |
| Commit-overlap startup | 1 | Missing host fixture and environment-selected assumptions for pinned-off modes. Supply host fixture, preserve all dependency guard cases through explicit unit seams, and separately check the actual retired-mode readers stay off. |
| Coverage dense | 8 | Three phase cases and two dispatch cases used the retired JSON environment API; construct typed dispatch fixtures and use unset/off for the actual switch. Fused evidence is pinned off, low-row policy is fixed off, and production selects the stock arm. Check those current contracts explicitly. |
| Execution-mode startup | 1 | Tests used environment selection and omitted the host fixture. Check actual stock/routed arm startup, full-graph selection, inconsistent eager/graph rejection, layer identity, topology rejection and all removed-flag shapes. |
| Prefill cuBLAS reference | 4 | Removed `_full_dual_mlp` helper raised before numerical comparison. Compute the independent dense two-branch reference in the test; keep original shapes, count assertions and tolerances. |
| Shared-band arm fields | 1 | Expected nine twins; current declaration contains the full twelve-point grid. Assert the exact rate/depth set. |
| Expected runtime generator | 1 | Prefix selection included derived twins and Qwen arms. Select the twelve base grid names exactly. |
| Paired analysis | 10 | Fixtures named `integrated_it4` while the CLI defaults to `vskipper`. Pass the fixture treatment explicitly; keep all refusal, interval and direction checks. |
| Paired campaign work gate | 4 | Driver now requires an explicit treatment argument. Supply it; preserve equal-work, unequal-work, verdict-file and missing-artifact cases. |
| Paper table | 10 | Fixture omitted p95 and makespan fields now required by the unchanged emitter. Add those fixture fields; preserve corruption and missing/stale-row checks. |
| Integer knee | 2 | Assertions predated the sustained-growth rule. Assert that an isolated blip is rejected and both applicable bracket guards fire. |
| Quality token identity | 1 | Literal source search predated protocol-specific command construction. Exercise the command builder for raw token-ID completion and chat-message/template protocols. |
| Random-skip sweep | 27 | Broad prefix captured derived twins/Qwen arms, then failed grid size and name parsing. Select exactly the twelve base names; separate arm-field tests cover twins and Qwen declarations. |

Before these 71 findings, shared-state test collection failed because it passed
`None` to the armed attestation runner API. That fixture now supplies the
required runner fields, explicitly exercises armed attestation and exposes a
pytest assertion. Counter-reset and object-identity assertions remain.

## Same repaired tests, unchanged control runtime

`validate_reference.py` builds a fresh control test overlay. Runtime directories
link to the hash-verified control; original control files remain intact. The
enumerated repaired tests come from the committed candidate. The adapter
translates import namespaces and relocated paths, recording original-control,
committed-fixture and adapted-fixture hashes plus each translation. It preserves
the original failing report and requires the repaired suite to pass on both
trees. A baseline failure is not silently accepted.

The candidate-specific source-layout checks run on the candidate. Both trees
retain their full real-import sweeps. A `tests-only` pass is one scoped gate;
release additionally requires all six paired serving cases, packaged-resource
checks and complete numerical reproduction.

The guard-page probe can skip when the allocator provides no unmapped page.
Its memory-safety requirement remains outstanding until the corresponding
compute-sanitizer check passes. A skip is not a memory-safety pass.
