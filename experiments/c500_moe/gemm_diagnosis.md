# mcTlass performance diagnosis (C500, 2026-10-29)

Controlled GPU-event timing of each actual expert GEMM in the same MoE call. Seven rounds per variant, rotated order; all tested GEMM output tensors exactly equal the official output. `SystemExit` after the second stage is intentional. Inputs, routing and GPU buffers are shared. Official implementation is prebuilt `/opt/maca/lib/libmctlassEx.so` (SHA256 c6862540e1a7ddd1d285352f3c5b072fdb644d0cc7c268821dc0c7012d14a521). Locally rebuilt variants use SDK 3.8.2.6 headers and the C-callable frontend wrapper. Do not test unmodified rebuild at T=131072: signed overflow causes invalid access.

At T=131072: official gate/up 117.3903 ms, patched 132.6377 ms; official down 69.0120 ms, patched 76.0804 ms. Combined median gaps 22.3158 ms, consistent with 22.56 ms whole-MoE gap but independently measured and medians not strictly additive.

At T=65536: official gate 60.8115 ms, unmodified rebuilt 66.4259, input-only 68.1121, output-only 66.5779, both patches 68.2527. Down respectively 35.7079, 38.7866, 39.0738, 38.9732, 39.2922. This localizes slowdown to both the official-vs-rebuilt difference and additional 64-bit input indexing, especially gate; input-only and output-only changes are not necessarily additive. Rebuilt O2 vs O3 does not materially change times in this experiment.

mcTracer `comparison-30941.json` shows the official and rebuilt kernel symbols differ internally though both have threadblock 128x128x256 and warp 32x128x256. The version/build/implementation differences are not isolated from one another: this does not establish exactly which compiler flag or instruction explains the official-rebuilt gap. More source/build provenance and profiling would be needed.

## Follow-up: resource and compiler evidence

The same gate/up grid is (8, 36, 432) with 256 threads and 64 KiB shared memory; down is (8, 40, 432), same block and shared memory. mcTracer records 248 registers/thread for the official kernel, 254 for the unmodified SDK rebuild, and 256 for the patched rebuild; all show zero private memory. These are observations, not proof that register usage causes the slowdown.

Official and rebuilt fatbins were extracted with llvm-objcopy and clang-offload-bundler. Exact kernel symbols from mcTracer were found in the xcore1000 LLVM IR. Official and rebuilt gate kernels each contain 512 static MMA intrinsics. Official uses llvm.mxc.griddim.id.x; the rebuilt uses dispatch metadata and udiv to compute grid size. Recompiling with __USE_DIM_INTRINSIC__ did not materially change stage timings at T=65536 or 131072, so this difference does not explain the measured regression. Results: compare_dim_65536.json and compare_dim_131072.json. The official and SDK kernels have different template signatures and 248 vs 254 reported registers, but source version and precise generated instruction-level cause of their performance gap remain unproven.

## Instruction scheduling flag ablation

At T=131072, comparing official prebuilt and corrected experimental GEMM in rotated order, medians in milliseconds: official gate/down 118.2735/69.3030; experimental default 132.8207/76.1303; experimental with `-mllvm -metaxgpu-igroup=true` 126.0150/74.1484; experimental with `-mllvm -misched-postra=true` 132.8161/76.0952. Both flags together 126.7645/74.0847. All tested full GEMM outputs were exactly equal in this experiment. Thus igroup materially reduces the experimental regression for these cases; postra alone did not. These flags were selected for controlled testing; the actual build options of the official binary are not established.

Seven-round whole-MoE test with igroup, original activation and guarded uint32 quantization: T=131072 median 236.3947 ms (235.47 effective TOPS), compared with earlier default patched median 245.0836 ms (227.12 TOPS) and official prebuilt baseline 222.5260 ms (250.14 TOPS). These whole-MoE runs were separate, not interleaved. Three sampled CPU reference checks passed; no whole-output reference comparison was done for this timing run.

The cause of the remaining gap to the official prebuilt implementation is unresolved. Source and build metadata, register usage, and hardware execution may differ. The current image lacks mcProfiler, limiting counter-based investigation.

## Mixed index experiment

In the specialized 128x128x256 kernel, input gather offsets use uint32 under a host-side input-numel bound and output stores stay uint64; built with -mllvm -metaxgpu-igroup=true. Compared using the same buffers and rotated runs, T=131072 gate medians: official 118.0316 ms, fixed uint64+igroup 125.9505 ms, mixed 121.8542 ms; down 69.1546, 74.0575, 74.0995 ms. For T=262144 the official down fails with invalid access, so no full two-stage same-buffer official comparison exists. Separate full MoE output comparisons against official 16384-token chunks passed exact at T=155345 and 262144, tested E=384, H=5120, F=2304, topk6. One-launch native SiLU and guarded uint32 quantization used. Full MoE mixed medians: 232.2227 ms, 239.70 effective TOPS at T=131072, and 460.7388 ms, 241.62 TOPS at T=262144. CPU sample checks passed. Scope is experimental; input numel guard is necessary but other parameters, layouts, sentinel handling, uint32 intermediate arithmetic and 2^32 boundaries need further validation. Do not install.

## Load scheduling experiment

The mixed-index + igroup build with `-mllvm -metaxgpu-ldg-bsm-mtreg-reduce=true` produced T=131072 whole-MoE median 229.5171 ms (seven rounds, range/median 1.16%, three CPU sample checks passed), versus 232.2227 ms for the earlier mixed-index + igroup run; these are separate runs, not an interleaved comparison. The event timing includes the full post-routing expert path, guarded uint32 quantization, single native activation, and excludes routing. At T=262144, the full output is elementwise identical to the original mcTlass path evaluated in 16 chunks with the same quantization, inputs and routes. Files: reduce_native_perf_131072.json and reduce_native_compare_262144.json. Only the specified shape/configuration was verified; this is not evidence of general safety at uint32 boundaries. The remaining gap to the prebuilt original kernel has not been attributed to a specific instruction or compiler flag.

## Comparison scope correction

The full-MoE 222.5260 ms uint32-quant baseline uses the original activation (full_moe_u32_guarded.py); the 229.5171 ms experimental result uses the separately implemented 64-bit native activation (full_moe_reduce_native.py). Thus the previous full-MoE comparison also changes activation, and cannot isolate the GEMM effect. They were separate sessions as well. For GEMM-only conclusions use the interleaved, same-input/same-output-buffer stage experiment `compare_reg_variants.json`: T=131072 original vs ldg_reduce gate/up 114.7907 vs 120.0056 ms, down 69.1103 vs 72.9521 ms, seven rounds; full GEMM outputs were exactly equal for each variant in this test. For the original gate/up run range/median was 3.03%, so do not overstate precision. Attributing this remaining gap to a specific instruction remains unresolved.

## Preserved kernel LLVM IR for offline analysis

Extracted the exact four kernel functions identified by mcTracer from the saved official and unmodified rebuilt device LLVM IR, without running the GPU: official_36.ll, rebuilt_36.ll, official_40.ll, rebuilt_40.ll. For both gate/up and down, static call sites include 512 MMA, 32 predicated global-to-shared vector loads and 8 barrier intrinsics in both variants. The official has eight more `llvm.mxc.icmp.i64.i32` static call sites than the rebuilt kernel for both stages; this alone cannot explain its faster execution. Static IR counts are not dynamic instruction counts; compiler scheduling, final machine code and loop execution frequency are not established by these counts. These four extracted functions omit the module declarations/metadata and are for inspection, not standalone compilation.

## Mixed-index input-bound audit

The mixed-index patch narrows only ldgA_offs input gather arithmetic to uint32; output offsets remain widened. Its present wrapper checks A.numel() <= 2^32, separately for each GEMM. For the tested H=5120, F=2304, topk=6, unchunked contiguous inputs require T*5120 <= 2^32 for gate/up and 6*T*2304 <= 2^32 for down. These necessary numerical bounds are T <= 838860 and T <= 310689 respectively. T=262144 satisfies both; T=524288 fails the down bound and must be rejected or dispatched to a safe wider-offset fallback. These are necessary element-count bounds only, not a proof of safety: intermediate unsigned arithmetic, predicated/sentinel accesses, vector widths, strides, and different shapes still require auditing.

## Additional load-path difference in saved IR

In each of the saved gate/up and down functions, the original prebuilt kernel contains eight `llvm.mxc.ldg.predicator.bsm.v1i16` call sites near initialization; none are present in the extracted unmodified rebuilt functions. For gate/up these source addresses derive from the input pointer, routed token indices and K; predicates include both a tile-related condition and valid-token check. This is concrete evidence that their input-load paths differ, even before index widening. Both versions still have 32 v4i32 predicated BSM load call sites and 512 MMA call sites. The v1i16 loads are predicated, so their runtime frequency and impact cannot be inferred from their static existence, and their exact role needs source/ISA or counter validation. They must not be presented as a proven cause of the slowdown.

## Predicate clarification for original-only loads

On closer inspection, the earlier phrase “tile-related condition” was inaccurate. In the original gate/up IR, `%33 = threadIdx.x & 63` and `%94 = (%33 == 0)` feed the first v1i16 load predicate along with the row-valid comparison `%68 < %3`; the corresponding down function uses `%35 = threadIdx.x & 63` and the same lane-zero test. Thus these eight static input-load call sites are restricted to lane 0 in each 64-thread wave and a validity predicate, not shown to be conditional on a K-tail. For valid routed rows at least one predicate can be logically true, but we have not verified the exact intrinsic lowering or dynamic timing. This does not establish that these loads explain the performance difference.

## Expert-block early-exit difference

The saved original gate/up LLVM IR checks the selected expert index (`%44`) for being negative and returns early when negative, after the padded-block bound check (`official_36.ll`, labels 39 and 46). The extracted unmodified rebuild function (`rebuilt_36.ll`) has the padded-block bound check but no equivalent early return at that point; its selected expert index `%53` is sign-extended and enters address calculation. This is a control-flow distinction and a possible correctness/dispatch compatibility issue for negative-expert entries, not proof of an overhead source in the tested routes. Counts and semantics of negative IDs in the tested execution have not been captured, so the runtime performance or safety effect remains unknown. Do not infer either that this branch explains the slowdown or that negative IDs are supported by the rebuild.

## Reproduced T=131072 routing input

Regenerated the exact NumPy 2.2.6 routes with seed 902072 using benchmark_long.py and verified SHA256 0b68de68545357965371313a5dbb943db1ad335f71f7319d95fda9944480e219 against the saved measurement. All 384 experts were selected; routed counts range from 1902 to 2179, and raw routed expert IDs are 0 through 383. This excludes negative IDs in the input routing array and empty experts for that test, but does not prove the separately constructed padded per-block expert-ID buffer contains no negative entries; that buffer was not captured. The negative-expert early-exit hypothesis thus remains unverified.

## Public mcoplib alignment source clue (not proven installed binary)

Local mcoplib commit e1cbf1d47660be6807cd0096ff811c86ee364401, op/vllm/moe/moe_align_sum_kernels.cu, generic `_moe_align_block_size`, writes expert_ids = threadIdx.x for all blocks below total_tokens_post_pad and fills the remaining capacity with inactive_expert_id; the regular kernel passes -1. The original and rebuilt GEMMs both have an early block bound check against padded rows. If the installed binary uses equivalent logic and the call has no expert-map remapping, valid block IDs will be nonnegative and the negative-expert check should not explain the tested difference. The installed binary has not been matched to this source revision and the per-block array was not captured, so this is conditional rather than an observed runtime result.

## CPU-only machine-code tooling check

On the currently accessible SDK 3.8.2 instance, following /opt/maca symlinks finds no llvm-objdump, llc, or mcProfiler executable under the MACA/MX driver trees. mxcc identifies the saved original bitcode target as mxc-metax-macahca, but rejects standard clang `-S`, `-target`, and `-mcpu` arguments; direct assembly reproduction with those arguments is therefore unavailable. /usr/bin/objdump exists but XCore disassembly capability is not established. This does not prove disassembly impossible with another tool or SDK. No GPU work was run in this check.

## Seven-round ablation review (T=65536)

Using the saved seven raw samples in compare_index_variants.json, the original-to-unmodified-rebuild median differences are +5.6143 ms gate/up and +3.0787 ms down. Relative to that same rebuild, the input-only 64-bit change adds +1.6863 and +0.2872 ms; output-only adds +0.1521 and +0.1866 ms; both changes together add +1.8268 and +0.5056 ms. Rebuilt variants are identical source/build options other than the isolated changes; outputs match bit-for-bit at this shape. The original-to-rebuild difference cannot be assigned solely to compiler flags: kernel source/template implementations also differ. Median differences are not strictly additive and seven rounds in one session cannot rule out clock/order effects. Do not extrapolate to other token counts or claim universal attribution.

## mxcc assembly option check

`mxcc --help` documents `-dlink-asm` as generating device-link-phase assembly, and `-Xdevice` for device arguments. A direct invocation with `-dlink-asm` on the extracted official_bc/18.bc did not produce the requested assembly artifact. The correct input format and driver pipeline remain unverified; even recompiled bitcode would not by itself prove correspondence with the machine code executed by the prebuilt binary. No GPU work was run.

The system GNU objdump 2.38 `-i` lists x86 targets but no mxc/XCore target, so it cannot be assumed to decode C500 code. A tiny MACA kernel compiled successfully with mxcc `-dlink-asm -c`, but produced only the host object; an additional attempted `-dlink` step is not recognized. No conclusion about the flag in a supported device-link workflow. No GPU was used.

## Public compiler workflow reference

MetaX developer blog https://developer.metax-tech.com/blog/w41wptpgwba1x7twjctj8kk8 describes disassembling the host stub using `mxcc -c -dlink-asm ...` followed by GNU objdump, and describes GPU machine assembly separately as requiring llvm-objdump. This supports why the prior small-kernel invocation produced a host object, not immediately readable XCore assembly. Search index text was accessible; opening the page timed out. No compatible GPU llvm-objdump has been found in this instance.

## Top-k specialization check

The captured gate/up call uses topk=6, down uses topk=1. The SDK frontend snapshot branches on whether topk is a power of two for this 128x128x256 configuration. The extracted IR contains 25 static `sdiv` operations for both official and rebuilt gate/up kernels, and one for both down kernels (plus one extra udiv in the rebuilt code related to grid derivation). This is consistent with both implementations using different top-k specializations for gate/up vs down, not evidence of a top-k division mismatch between the two variants. Static counts do not prove instruction-level equivalence or runtime cost.

## Public source availability check

The locally available public MetaX-MACA/mcTlass checkout is branch 2.20.2, and a full-text search found no MacaMoeMmaMultistage or MacaGemmMoe definitions. Public web searches for these exact symbols and the specialized 128x128x256 header returned no indexed match. The official GitHub repository confirms default branch 2.20.2; live git branch enumeration failed with a connection reset, so other branches or unindexed sources cannot be ruled out. The source and exact build configuration of the precompiled mctlassEx GPU kernel remain unavailable, preventing a full matched source/build comparison.

## Existing mcProfiler data scope

The archived c500_fullcard_profiler_compare_20261023 data is for standalone 2048-square naive GEMM and mcBLAS, not the expert GEMM versions diagnosed here. It cannot establish relative stall, cache or instruction counters for original vs experimental mcTlass. Its README reports counter inconsistencies; the older full64_mcprofiler_20261022T024827Z README separately records an ISU/CE wave undercount. Both limit interpreting any future comparison without counter validation. The archived CLI version was 3.8.1.4 on a different MACA image; compatibility with the current 3.8.2.6 image is unverified.
