# dual-stream

A Python library that wraps any OpenAI-compatible agent loop and enforces two-stream context management at inference time. No retraining. No weight changes. Drop-in.

## The Idea

Long-horizon agents fail in a specific way: a flat ReAct loop accumulates tool outputs, environment responses, and intermediate reasoning into a single buffer. The compressor treats everything uniformly. Past step ~50, the original goal and active subgoal state are diluted below the effective attention horizon. The agent asks what it was supposed to do. This is not a model failure — it is a context architecture failure.

The fix is structural separation. Instead of one buffer, use two:

- **GoalStream** — append-only, verifier-gated. Contains the root task and active subgoal stack. Never compressed. Never evicted. Every write must pass three deterministic checks before it commits.
- **ObservationStream** — lossy FIFO. Contains tool outputs and environment responses. Oldest entries evicted when budget is exceeded.

The compressor is physically incapable of touching the GoalStream. This is an architectural guarantee, not a behavioral one. The agent cannot forget what it was doing because the data structure that holds that information is unreachable by the compressor.

## Hypothesis

Separating goal state from observation history at the context buffer level — and verifying writes to the goal buffer before they commit — prevents subgoal loss in flat agents without touching model weights.

**Expected pattern in results:**
- Flat sliding-window baseline: completion rate drops significantly between N=50 and N=100 as the goal state gets compressed away
- Flat sink+recent baseline: similar drop, possibly at a different N
- Dual-stream: holds within ~5pp of its N=50 rate through N=200
- Verifier-off ablation: tracks the flat baselines, not dual-stream — proving the stream separation matters, not just the verifier

**Secondary claim:** GoalStream overhead stays under 5% of total context at all N. Without this, the argument is just "allocate more budget to the system prompt." The overhead claim makes it a fair comparison.

## The Verifier

Three deterministic, sub-millisecond checks on every proposed goal write:

1. **Scope narrowing** — the proposed subgoal must be strictly narrower than its parent. No embeddings. No LLM. Substring check only.
2. **Redundancy** — normalized exact match against completed entries. No fuzzy matching.
3. **Spec consistency** — at least one term from the original task spec must appear in the proposed subgoal.

If any check fails, the write is rejected and a notice is injected into the ObservationStream so the agent can self-correct. The GoalStream is never mutated on rejection.

## Architecture

```
DualStreamAgent
├── GoalStream           ← verified writes only, never compressed
│   └── VerifierGate     ← 3 pure checks, no network, <1ms
├── ObservationStream    ← FIFO eviction, oldest-first
│   └── SlidingWindow    ← accepts ObservationStream only (enforced by type)
├── Assembler            ← GoalStream first (high-attention prefix), then ObsStream
└── Backend              ← OpenAI-compatible, temperature=0, exponential backoff

Baselines (structurally identical loop, different context handling only):
├── FlatSlidingWindowAgent  ← single FIFO buffer, same total budget
└── FlatSinkRecentAgent     ← StreamingLLM-style: first S tokens + most recent N
```

## Current Results

**Status: preliminary smoke test only, not finalized.**

These are from a 3-task smoke run on Qwen/Qwen3-32B (FP8) on an H100, using deterministic fixture tasks that simulate WebArena navigation and τ-bench tool calls. The full 50-task sweep was interrupted when the GPU instance was terminated. Numbers below are directionally interesting but statistically meaningless at n=3.

| Benchmark | Condition | N=10 | N=20 | N=50 | N=100 |
|---|---|---|---|---|---|
| WebArena | dual_stream | 1.00 | 1.00 | 1.00 | 1.00 |
| WebArena | flat_sliding_window | 0.00 | 0.00 | 0.00 | 0.00 |
| WebArena | flat_sink_recent | 0.00 | 0.00 | 0.00 | 0.00 |
| WebArena | verifier_off | 1.00 | 1.00 | 1.00 | 1.00 |
| τ-bench | dual_stream | 1.00 | 1.00 | 1.00 | 1.00 |
| τ-bench | flat_sliding_window | 0.00 | 0.00 | 0.00 | 0.00 |
| τ-bench | flat_sink_recent | 0.00 | 0.00 | 0.00 | 0.00 |
| τ-bench | verifier_off | 1.00 | 1.00 | 1.00 | 1.00 |

The verifier_off condition matching dual_stream is interesting: it suggests the stream separation itself (not the verifier) is what matters for these short fixture tasks. The verifier is expected to differentiate more on real long-horizon tasks where goal corruption and redundancy become problems at depth.

GoalStream overhead ranged from 6–20% in the smoke — higher than the target due to tasks completing in 3–8 steps with sparse observations. At N=100 with longer trajectories, overhead dropped to ~6%, approaching the <5% target.

**These numbers are not ready to cite. Run the full sweep first.**

## Setup

```bash
pip install -e ".[dev]"
```

Requires an OpenAI-compatible backend. For local inference:

```bash
pip install "vllm>=0.8.0"
vllm serve Qwen/Qwen3-32B --quantization fp8 --port 8000
```

Set environment variables:

```
OPENAI_API_KEY=local
OPENAI_BASE_URL=http://localhost:8000/v1
MODEL_NAME=Qwen/Qwen3-32B
```

## Running

```bash
# Unit tests (no API calls)
make test

# Smoke sweep: 3 tasks, all conditions, all step budgets
python -m experiments.run_sweep --tasks 3 --results-dir /tmp/smoke

# Full sweep: 50 tasks
python -m experiments.run_sweep --tasks 50

# Figures from populated results/
make figures
```

## References

- **HORIZON** (2025) — empirically shows failure cliffs exist as horizon grows; does not propose an architectural fix. We do.
- **StreamingLLM** (Xiao et al., 2023) — attention sink + recent tokens; implemented as the `flat_sink_recent` baseline.
- **TokenDance** (arxiv 2604.03143) — prefix caching across agents; the GoalStream root entry is a natural prefix cache target.
- **RLM** — GoalStream active_stack() at any step is isomorphic to the RLM call stack. The connection is intuitive; making it rigorous is listed as a post-paper extension.
- **AppWorld** (Trivedi et al., ACL 2024 Best Resource Paper) — considered and dropped as a primary benchmark because Qwen2.5-14B scored 0% even on test_normal, making the comparison impossible without a much larger model.
