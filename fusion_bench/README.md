# fusion_bench — does multi-agent planning belong in the compiler?

TorchTPU's **Fused Eager** mode groups consecutive ops into denser fused XLA
subgraphs on the fly. Choosing the group boundaries is an optimization problem —
and it is the *same shape* as my MLSys 2026 winning entry: planning a schedule
for a memory-constrained DAG.

The obvious move is to point my multi-agent planner at it. This experiment
instead asks the question a compiler engineer would ask first: **should you?**

## The setup

Model a forward pass as a linear chain of ops, each with an output size. A
"fusion group" is a contiguous run of ops fused into one subgraph; a group must
fit a fast-memory (SRAM) budget. Two strategies, same graph, same group count:

| Strategy | Algorithm | Cost |
|----------|-----------|------|
| `greedy_fusion` | First-fit, one pass | **O(n)**, deterministic |
| `balanced_fusion` | DP minimizing the largest group ("split array largest sum") — stands in for the expensive global / multi-agent search | **O(n²·k)** |

## The result (`python -m fusion_bench.demo`)

```
classical greedy (O(n))          groups=8  peak=15,000,000 bytes  decide=3.7 us
global planner (DP, agent-like)  groups=8  peak=14,000,000 bytes  decide=513.9 us

Planner improves peak balance by 6.7% but is 140x slower per decision.
```

## The lesson

The global optimizer *does* produce a better grouping — but only modestly, and
at ~140× the decision latency. A compiler makes fusion decisions constantly; a
policy that is two orders of magnitude slower per decision is disqualified from
the hot path no matter how slightly better its output is.

So the honest conclusion: **the classical greedy scheduler belongs in the
compile loop. The multi-agent / global-search approach is an *offline* tool** —
ideal for exploring the space and *discovering* cheap heuristics that the
in-loop scheduler can then encode. That is exactly the right role for the
planning method, and naming that boundary is the point of this experiment.

## Why this is here

Reusing my competition's hammer would be the easy story. Showing *where it
fits and where it doesn't* — with measured latency, not vibes — is the honest
one, and the one that matters for a compiler team.
