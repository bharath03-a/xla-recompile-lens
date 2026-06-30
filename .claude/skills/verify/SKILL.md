---
name: verify
description: Run the full local verification loop for xla-recompile-lens — lint, tests, and the CPU example demos. Use before committing or when asked to confirm the project is healthy.
---

# verify

Run every check that does not require a TPU. All of this works on CPU.

## Steps (run in order, stop and report on first failure)

1. **Sync deps** (idempotent):
   ```bash
   uv sync --extra viz
   ```

2. **Lint:**
   ```bash
   uv run ruff check .
   ```

3. **Tests:**
   ```bash
   uv run pytest
   ```
   Expect all tests to pass. They are CPU-only by design.

4. **Smoke-run the examples** (they must exit 0 and print sensible numbers):
   ```bash
   uv run python examples/see_it_work.py
   uv run python examples/plugin_demo.py
   uv run python -m fusion_bench.demo
   uv run xla-recompile-lens --demo
   ```

## Pass criteria

- ruff: "All checks passed!"
- pytest: all green
- `see_it_work.py`: ~99 recompiles without fix, ~2 with fix
- `plugin_demo.py`: WITHOUT > WITH recompiles (fix reduces the count)
- `fusion_bench.demo`: greedy fast, planner slower; verdict line printed

## On failure

Report the exact failing command and its output. Do not "fix the test to make
it pass" — fix the implementation unless the test is demonstrably wrong. Never
weaken an honesty/measurement assertion to get green.
