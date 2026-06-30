"""End-to-end serving benchmarks for xla-recompile-lens.

`llm_prefill_serving` measures the real wall-time cost of XLA recompilation for
LLM prefill under a realistic prompt-length distribution, and the speedup from
data-derived sequence bucketing (see `xla_recompile_lens.bucket_advisor`).
"""
