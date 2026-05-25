# dual-stream

## The Problem

A flat ReAct agent given a 100+ step task accumulates tool outputs, environment responses, and intermediate reasoning into a single buffer. The compressor treats everything uniformly. Past step ~50, the original goal and active subgoal state are diluted below the effective attention horizon. The agent asks what it was supposed to do. This is not a model failure — it is a context architecture failure.

## The Fix

Two buffers instead of one.

**GoalStream** — append-only, verifier-gated. Contains the root task and active subgoal stack. Never compressed. Every write passes three deterministic checks before committing.

**ObservationStream** — lossy FIFO. Tool outputs and environment responses. Oldest entries evicted when budget is exceeded.

The compressor is physically incapable of touching the GoalStream. This is an architectural guarantee, not a behavioral one.

## The Verifier

Three deterministic checks on every proposed goal write. No embeddings. No LLM. Runtime under 1ms each.

1. **Scope narrowing** — proposed subgoal must be strictly narrower than its parent. Substring check.
2. **Redundancy** — normalized exact match against completed entries.
3. **Spec consistency** — at least one term from the original task spec must appear in the proposed subgoal.

Failed writes are rejected and a notice is injected into the ObservationStream so the agent can self-correct. The GoalStream is never mutated on rejection.

## What We Built

```
DualStreamAgent
├── GoalStream           ← verified writes only, never compressed
│   └── VerifierGate     ← 3 pure checks, no network, <1ms
├── ObservationStream    ← FIFO eviction, oldest-first
│   └── SlidingWindow    ← accepts ObservationStream only (enforced by type)
├── Assembler            ← GoalStream first (high-attention prefix), then ObsStream
└── Backend              ← OpenAI-compatible, temperature=0, exponential backoff

Baselines (identical loop, different context handling only):
├── FlatSlidingWindowAgent   ← single FIFO buffer, same total budget
└── FlatSinkRecentAgent      ← first S tokens + most recent N (StreamingLLM-style)
```

## Preliminary Results

**Not finalized. n=3 smoke run, GPU instance was terminated before the full 50-task sweep completed. Treat these as directional only.**

Qwen/Qwen3-32B (FP8) on H100. WebArena navigation tasks and τ-bench tool-call tasks.

| Condition | WebArena | τ-bench |
|---|---|---|
| dual_stream | 1.00 | 1.00 |
| flat_sliding_window | 0.00 | 0.00 |
| flat_sink_recent | 0.00 | 0.00 |
| verifier_off | 1.00 | 1.00 |

The verifier_off condition matching dual_stream suggests the stream separation itself — not the verifier — is what matters at short horizons. The verifier is expected to differentiate on real long-horizon tasks where goal corruption and redundancy become problems at depth.

GoalStream overhead was 6–20% in the smoke due to tasks completing in 3–8 steps with sparse observations. At N=100 with longer trajectories it dropped to ~6%, approaching the <5% target. Full 50-task sweep needed to confirm.

## Hypothesis

The expected pattern in the full results:
- Flat baselines: completion drops significantly between N=50 and N=100 as goal state gets compressed away
- Dual-stream: holds within ~5pp of its N=50 rate through N=200
- Verifier-off: tracks the flat baselines, not dual-stream

Secondary claim: GoalStream overhead stays under 5% of total context at all N. Without this the argument reduces to "allocate more budget to the system prompt."

## Setup

```bash
pip install -e ".[dev]"
```

Requires an OpenAI-compatible endpoint:

```bash
pip install "vllm>=0.8.0"
vllm serve Qwen/Qwen3-32B --quantization fp8 --port 8000
```

```
OPENAI_API_KEY=local
OPENAI_BASE_URL=http://localhost:8000/v1
MODEL_NAME=Qwen/Qwen3-32B
```

```bash
make test                                          # 42 tests, no API calls
python -m experiments.run_sweep --tasks 3 --results-dir /tmp/smoke
python -m experiments.run_sweep --tasks 50
make figures
```

## Related Work

[TokenDance](https://arxiv.org/abs/2604.03143) (Apr 2026) — KV cache deduplication for multi-agent systems. The GoalStream root entry is a natural prefix cache target: shared across agents working on subtasks of the same root goal, computed once, reused N times.

[HORIZON](https://arxiv.org/abs/2504.10865) (2025) — Empirically shows failure cliffs exist as horizon grows, does not propose an architectural fix.

[StreamingLLM](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023) — Attention sink + recent tokens. Implemented here as the `flat_sink_recent` baseline.

[Agent Memory Below the Prompt](https://arxiv.org/abs/2502.06975) (Feb 2026) — Persistent quantized KV cache for multi-session agent state. Orthogonal; relevant for GoalStream persistence across sessions.
