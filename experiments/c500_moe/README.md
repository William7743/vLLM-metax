# C500 W8A8 MoE experiments

Experimental scripts use synthetic inputs, precomputed Top-6 routes, and `fused_experts_impl`; routing and communication are excluded. They were tested on a particular C500 / MACA 3.8.2.6 environment and an older vLLM-metax snapshot, not validated against this checkout. Do not install or enable them in production.

For an unchunked experiment, set both `TEST_MOE_CHUNK_SIZE` and `VLLM_FUSED_MOE_CHUNK_SIZE` to the selected token count before launching the benchmark. This is an experiment configuration, not a change to the library default. `benchmark_long.py` expects the vLLM and C500 environment and a pre-existing output directory. Large inputs may overflow or exhaust memory.
