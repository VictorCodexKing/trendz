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
6. **Publisher** - one publisher per platform (fan-out per platform).
7. **Performance Analyst** - collects post metrics and attributes outcomes.
8. **Learning / Memory Store** - self-improvement feedback loop and shared memory.

Stages 1-5 (Trend Scout, Content Strategist, Clip Factory, Quality & Safety
Gate, and Scheduler & Optimizer) are implemented today: a full offline run
returns a per-platform `PublishPlan`. Stages 6-8 are scaffolded and planned.

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
