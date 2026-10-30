# C500 W8A8 MoE experiments

Experimental scripts use synthetic inputs, precomputed Top-6 routes, and `fused_experts_impl`; routing and communication are excluded. They were tested on a particular C500 / MACA 3.8.2.6 environment and an older vLLM-metax snapshot, not validated against this checkout. Do not install or enable them in production.

For an unchunked experiment, set both `TEST_MOE_CHUNK_SIZE` and `VLLM_FUSED_MOE_CHUNK_SIZE` to the selected token count before launching the benchmark. This is an experiment configuration, not a change to the library default. `benchmark_long.py` expects the vLLM and C500 environment and a pre-existing output directory. Large inputs may overflow or exhaust memory.

`full_moe_u32_guarded.py` replaces the per-token Triton quantizer in the benchmark process only, using uint32 row and stride calculations with a contiguous-input element-count guard. It does not modify the installed quantizer. The guard alone is not proof of safety for arbitrary strides, vector tails or near-2^32 offsets.

The 64-bit GEMM offset patches target SDK 3.8.2.6 headers, and the C ABI wrapper expects a separately built `libexperimental_moe.so`. The benchmark monkey-patches only its own process. The SDK files, built library and a portable build procedure are not included; the diagnostic notes preserve observed timing and limitations.

The native SiLU experiment is under `native_activation_i64/` and replaces the activation only in the benchmark process. See its README for input restrictions, source provenance and tested shapes. The full MoE helper hard-codes the old remote library path and requires adaptation before reuse.
