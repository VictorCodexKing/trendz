# Trendz

Trendz is an autonomous multi-agent clip generation system. It runs a pipeline
end-to-end without human intervention: it discovers trends, sources or generates
video content, edits clips, publishes them to social platforms, and then learns
from the results to improve future runs.

## Architecture overview

The pipeline is a chain of specialized agents coordinated by an orchestrator.
The nine agent roles are:

0. **Orchestrator** - owns run lifecycle, dispatch, concurrency, and retries.
1. **Trend Scout** - discovers and scores trends across sources.
2. **Content Strategist** - turns trends into concrete clip briefs (fan-out).
3. **Clip Factory** - one worker per brief; script/voice, visual, caption, assembly.
4. **Quality & Safety Gate** - automated per-clip technical and safety gate.
5. **Scheduler & Optimizer** - decides when and where each clip posts.
6. **Publisher** - one publisher per platform (fan-out per platform), publishing
   to YouTube Shorts, Instagram, TikTok, and Facebook.
7. **Performance Analyst** - collects post metrics and attributes outcomes.
8. **Learning / Memory Store** - self-improvement feedback loop and shared memory.

All eight stages (Trend Scout, Content Strategist, Clip Factory, Quality & Safety
Gate, Scheduler & Optimizer, Publisher, Performance Analyst, and Learning /
Memory Store) are implemented today, so the pipeline is complete end to end:
discover -> plan -> render -> gate -> schedule -> publish -> analyze -> learn ->
feed back. A full offline run discovers and ranks trends, plans and renders
clips, gates them, schedules and publishes each post per platform (YouTube
Shorts, Instagram, TikTok, Facebook), collects each succeeded post's metrics and
attributes the outcome back to its clip/brief/trend, then folds those outcomes
into the persisted learnings and returns a `RunResult` bundling the per-platform,
per-post `PerformanceReports` and the stage-8 `LearningState`.

Stage 8, the Learning / Memory Store, closes the self-improvement loop. It folds
the Performance Analyst's `PerformanceReports` into a persisted `LearningState`
via a deterministic fixed-learning-rate reinforcement update (an
exponential-moving-average nudge toward the observed, normalized reward) plus a
seeded epsilon-greedy multi-armed bandit that balances explore and exploit across
the per-platform arms. The learnings are shaped to feed straight back into the
earlier agents: updated scoring weights into the **Trend Scout**, per-platform
format/hook priors into the **Content Strategist**, an updated engagement
threshold into the **Quality & Safety Gate**, and per-platform timing hints into
the **Scheduler & Optimizer** (`Orchestrator.next_run_agents` maps a
`LearningState` into those real tunable configs). Because the store persists
across runs, applying reports repeatedly accumulates the priors, so the pipeline
improves over time. Everything stays fully offline and deterministic (no network,
API keys, wall-clock, or unseeded randomness). The remaining work is
cross-cutting run-lifecycle hardening (retries/backoff, timeouts, circuit
breakers, budget guards, and a dead-letter queue), not a missing pipeline stage.

See the full design in [docs/DESIGN.md](docs/DESIGN.md).

## Install

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv
uv pip install -e '.[dev]'
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Run and test

```bash
pytest              # run the test suite
ruff check .        # lint
black --check .     # format check
mypy src            # type check
```
