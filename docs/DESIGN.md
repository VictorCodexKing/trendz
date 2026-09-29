# Trendz - Autonomous Multi-Agent Clip Generation System

## System Goal
A pipeline that runs end-to-end without human intervention: it discovers trends, sources or generates video content, edits clips, and publishes them to social platforms — then learns from the results to improve future runs.

## Agents
0. Orchestrator (conductor): owns run lifecycle, triggers on schedule or trend-spike, manages shared job state, dispatches work, enforces concurrency limits, handles retries/timeouts. Receives a run trigger + global config; decides clip count, concurrency degree, which agents to invoke, whether to abort; passes a RunContext (run ID, config, budget ledger) to Trend Scout.
1. Trend Scout: discovers trends across sources (platform trending APIs, hashtags, Reddit/X, Google Trends, RSS, competitors). Receives RunContext + niche filters + learning-store signals; scores trends on velocity, relevance, saturation, shelf life; passes a ranked TrendList.
2. Content Strategist: turns trends into concrete clip briefs. Receives TrendList; decides angle/hook, format, target length per platform, caption/CTA, hashtags, sourced vs AI footage; passes a list of ClipBrief objects (fan-out boundary).
3. Clip Factory (fan-out, concurrency core): one Clip Worker sub-agent per brief, concurrent up to a limit. Each worker coordinates parallel sub-steps: Script/Voice (script + TTS), Visual (stock/AI B-roll), Caption/Subtitle, Assembly (stitch audio+video+captions+music, format per platform aspect ratio 9:16/1:1/16:9). Receives one ClipBrief; passes a RenderedClip.
4. Quality & Safety Gate: automated gatekeeper per clip. Checks technical (resolution, audio, no black frames, length), content safety (brand safety, copyright/music licensing, platform policy), predicted-engagement score. Approved → Scheduler; rejected → back to Clip Factory with reason (bounded retries) or dropped with logged reason.
5. Scheduler & Optimizer: decides when/where each clip posts. Receives approved clips + audience-timing data; decides optimal post time per platform/timezone, ordering, spacing, A/B variants; passes a PublishPlan.
6. Publisher (fan-out per platform): one Platform Publisher sub-agent per platform (TikTok, Instagram Reels, YouTube Shorts, X), concurrent. Receives its platform slice; handles API quirks/rate limits; passes PostResults.
7. Performance Analyst: after a dwell period, collects metrics per post. Receives PostResults + pulls views, watch time, likes, shares, comments, follows; attributes outcomes to originating brief/trend; passes PerformanceReports.
8. Learning / Memory Store (self-improvement): shared long-term memory. Receives PerformanceReports; produces updated scoring weights/priors (format wins, best post times, high-signal sources, hook performance) via reinforcement-style updates + multi-armed-bandit exploration/exploitation; feeds back into Trend Scout, Content Strategist, Quality Gate, Scheduler.

## Concurrency Model
Fan-out points: Content Strategist → N briefs, Publisher → M platforms, each an independent parallel job. Within each Clip Worker, script/voice, visual, and caption run in parallel (fork-join; only Assembly waits). Orchestrator enforces a bounded worker pool (limited by API rate limits and render/GPU budget). Quality Gate and Analyst are embarrassingly parallel (one task per clip/post). Net total time ≈ slowest single clip, not the sum.

## Error Handling
- Per-step: retry with exponential backoff; on repeated failure substitute fallback (stock footage instead of AI render, alternate TTS voice).
- Per-clip: drop and continue after bounded retries; one bad clip never blocks the run; log reason for Analyst.
- External APIs: circuit breakers + rate-limit-aware queues per platform; degrade gracefully (queue if platform down).
- Idempotency: every job carries a unique ID; publishing is idempotent so retries never double-post.
- Budget/quota guards: Orchestrator tracks spend and quota; aborts/throttles before overruns.
- Dead-letter queue: permanently failed jobs land in a DLQ with full context.
- Observability: structured logs, per-agent metrics, alerts on abnormal failure rates.

## Self-Improvement
Closed feedback loop (Analyst → Learning Store → planning agents). Exploration vs exploitation via multi-armed bandits. Attribution ties each metric back to specific trend, hook, format, post-time. Failure learning: Quality-Gate rejections and DLQ entries become signal to avoid repeating rejected content.

## Suggested Tech Mapping
- Orchestration: workflow engine with parallel task support (Temporal, Airflow, or message queue + worker pool).
- Agent framework: multi-agent framework (LangGraph, CrewAI, or custom) with an LLM per reasoning agent.
- Media: FFmpeg for assembly, a TTS provider, a text-to-video/stock provider.
- State/memory: job store (Postgres/Redis) + vector/metrics store for the Learning module.
