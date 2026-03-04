# Proposal: Cost-Based Scoring — Pass the Quality Bar, Win on Cost

| Field       | Value                  |
|-------------|------------------------|
| Author      | @ningr                 |
| Created     | 2026-03-03             |
| Status      | **Draft**              |

---

## Problem

The current scoring system produces a single blended 0–1 score mixing safety, correctness, efficiency, and structure. This has two problems:

1. **"Score: 0.91" means nothing to enterprises.** The people who would buy optimized policy packs care about dollars, not abstract scores. "We took your agent cost from $0.16/task to $0.04/task" is a story that sells itself.

2. **The optimization target has a ceiling.** Score maxes out at 1.0. Once a miner hits 0.95, there's almost no headroom for challengers (especially with delta=0.05). Cost has no floor — miners can always find ways to use fewer tokens, creating permanent optimization pressure.

3. **Cost is the missing metric.** ClawBench tracks tool calls and rubric checks, but never reports how many tokens the agent used or what the episode cost in dollars. This is the single most important metric for enterprise adoption and it's completely invisible.

## Goal

Change the incentive structure so that:

```
Rubric checks = binary pass/fail bar per scenario (pass all or disqualified)
Cost (tokens → dollars) = the only competitive axis (lower wins)
```

Each scenario defines a set of rubric checks (safety, correctness, structure). A scenario is either PASS (all required checks pass) or FAIL. A miner must pass **every** scenario to compete. Failing any scenario = disqualified (effective cost = infinity). Among qualified miners, the cheapest one wins.

---

## Design: Per-Scenario Quality Gate + Cost Competition

### Core Rule

```
For each scenario:
    scenario_passed[hotkey][scenario] = ALL required rubric checks pass (binary 0/1)

qualified = ALL scenarios pass for this miner (every scenario = 1)
    if ANY scenario fails → disqualified (effective cost = ∞)

Among qualified miners:
    winner = lowest total_cost[hotkey]   (average $/episode across all scenarios)

If no miner is qualified:
    no winner — all weights = 0 (uniform weights across UIDs)
```

### Per-Scenario Pass/Fail

Each scenario defines rubric checks. A scenario is PASS (1) or FAIL (0) — there is no partial score, no threshold, no 0.85 vs 0.90 distinction. Either the pack handles the scenario correctly or it doesn't.

| Scenario | Result | Meaning |
|----------|:------:|---------|
| `client_escalation` | 0 or 1 | All safety + correctness checks pass? |
| `inbox_to_action` | 0 or 1 | All safety + correctness checks pass? |
| `morning_brief` | 0 or 1 | All correctness checks pass? |
| `team_standup` | 0 or 1 | All correctness checks pass? |
| `inbox_triage` | 0 or 1 | All correctness checks pass? |

**All 5 must be 1.** Failing any single scenario = disqualified. No averaging, no compensating. This is simple and unambiguous.

**Which checks are "required"?** All checks in the `safety` and `correctness` categories are required for pass/fail. `efficiency` and `structure` checks are informational only — they don't affect the gate. This keeps the bar focused on "does the agent do the right thing?" and leaves "how cheaply?" to cost competition.

### Cost Measurement

After each episode, capture cumulative token usage from the OpenClaw session:

```
cost_usd = (input_tokens × input_price
          + output_tokens × output_price
          + cache_read_tokens × cache_read_price
          + cache_write_tokens × cache_write_price) / 1_000_000
```

**Fresh session per scenario**: Each scenario evaluation starts a new OpenClaw session with an empty cache. No prompt cache is carried over from a previous scenario or a previous miner's evaluation. This ensures cost reflects the real cost an enterprise would see running the task cold, and prevents one miner's evaluation from benefiting from another's cached context.

Within a single scenario, the agent's multi-turn tool-use loop may generate cache hits (the system prompt and early messages get cached across LLM roundtrips within the same episode). This is normal and reflects real-world usage — the cost formula captures it accurately.

The per-episode cost is smoothed via EMA:

```
ema_cost[hotkey][scenario] = α × new_cost + (1 - α) × ema_cost[hotkey][scenario]

total_cost[hotkey] = Σ(w_i × ema_cost[hotkey][scenario_i]) / Σ(w_i)
```

Where `w_i` is the scenario weight from YAML.

Pass/fail qualification uses the existing **majority-vote consensus** (`evaluate_pack_consensus`): each rubric check is majority-voted across multiple runs. A scenario passes only if all required checks pass in the majority-voted result. This handles LLM non-determinism — a check that passes 2/3 times counts as passed; a check that fails 2/3 times counts as failed.

### Model Normalization

All evaluations use the same model (currently `claude-sonnet-4-5`) configured by the validator. Since every miner's pack is evaluated on the same model by the same validator, cost comparisons are apples-to-apples. No normalization needed.

If we later allow miners to target different models, we'd need a reference pricing table. For now, the validator's model is the standard.

### Winner Selection (Updated)

```python
# 1. Check qualification (all scenarios must pass)
qualified_miners = []
for hotkey in active_miners:
    qualified = True
    for scenario in all_scenarios:
        # Binary: did all required checks (safety + correctness) pass?
        if not scenario_passed[hotkey][scenario]:
            qualified = False
            break
    if qualified:
        qualified_miners.append(hotkey)

# 2. Select winner by cost (lower = better)
if qualified_miners:
    ranked = sorted(qualified_miners, key=lambda h: total_cost[h])

    # First-mover protection: incumbent keeps crown unless
    # challenger is meaningfully cheaper
    if current_winner in qualified_miners:
        current_cost = total_cost[current_winner]
        challenger_cost = total_cost[ranked[0]]
        if challenger_cost >= current_cost * (1 - COST_DELTA):
            winner = current_winner  # incumbent holds
        else:
            winner = ranked[0]       # challenger is significantly cheaper
    else:
        winner = ranked[0]
else:
    # No one qualified — no winner, set uniform weights
    winner = None
```

### First-Mover Protection (Cost-Based)

Since the competitive axis is now cost (lower = better), the delta threshold is multiplicative:

```
new_cost < incumbent_cost × (1 - δ)

Example (δ = 0.10):
  Incumbent cost:  $0.050/episode
  Challenger must: < $0.045/episode  (10% cheaper to dethrone)
```

There is no score comparison for first-mover protection. Score is binary pass/fail — either you're qualified or you're not. Among qualified miners, only cost matters.

A 10% cost delta because:
- Token counts have lower variance than rubric checks, so a smaller margin is meaningful
- But 5% cost difference could be pure noise from LLM variance, so 10% is safer
- This prevents "I shaved 2 tokens" from dethroning the incumbent

---

## Why This Design

### Comparison of Approaches

| Dimension | Blended Score (current) | Pass-Bar + Cost (proposed) | Quality/Cost Ratio | Tiered (bar + score + cost) |
|---|---|---|---|---|
| **Intuitive?** | No — "0.87 score" | **Yes — "$0.16 → $0.04"** | Somewhat | Medium |
| **Optimization target** | Vague | **Crystal clear: reduce cost** | Two variables | Unclear priority |
| **Enterprise appeal** | Low | **High — directly maps to ROI** | Medium | Medium |
| **Gaming risk** | Low-medium | Low (bar prevents junk) | High (trade quality for cheapness) | Low |
| **Quality incentive** | Strong (every point counts) | **Binary: pass all checks or disqualified** | Soft (quality in numerator) | Mixed |
| **Safety guarantee** | Soft (weighted higher) | **Absolute: fail 1 check = out** | None (can trade safety for cost) | Hard for safety, soft for rest |
| **Non-determinism** | EMA smooths score | **Binary checks + majority vote; cost is stable** | Compounds two noisy signals | Score boundary is noisy |
| **Leaderboard clarity** | "Score: 0.87" (so what?) | **"$0.04/task, -75%"** | "42 points per dollar" (weird) | Complex table |
| **Unbounded optimization?** | No (ceiling at 1.0) | **Yes (cost can always go lower)** | Yes | Partially |
| **Stagnation risk** | High (hit 0.95, done) | **Low (always room to cut cost)** | Low | Medium |

### Why Not Quality/Cost Ratio?

The ratio `score / cost` sounds elegant but is dangerous:

```
Pack A: score=0.99, cost=$0.10 → ratio = 9.9
Pack B: score=0.50, cost=$0.01 → ratio = 50.0   ← "wins" with terrible quality
```

A pack that half-asses every task but uses almost no tokens would dominate. The quality gate prevents this.

### Why Not Keep Score as the Main Axis?

1. **Ceiling problem.** Current top scores are 0.90+. With delta=0.05, challengers need 0.95+. With harder checks (proposal 0227), maybe there's 0.20 of headroom. But cost has unlimited headroom — $0.10 → $0.05 → $0.02 → $0.01. The optimization never stops.

2. **Miners will explore creative strategies.** When cost is the target, miners are incentivized to:
   - Write ultra-concise system prompts (fewer input tokens)
   - Minimize tool-use roundtrips (fewer LLM calls per episode)
   - Use stop rules aggressively (don't over-think simple tasks)
   - Eventually: distill to smaller/cheaper models (when model selection is enabled)
   - This is exactly the behavior we want for enterprise value

3. **Validator consensus improves.** Token counts vary ~10% between runs. Rubric checks can flip pass/fail across runs (5-10% variance). Cost-based ranking produces more stable winner selection, meaning fewer cross-validator disagreements and better YC3 convergence.

### The "Bare Minimum Quality" Concern

> Won't miners just barely pass the checks with the cheapest possible response?

**Passing all checks IS quality.** The bar is binary — pass every required check or you're out. There's no "barely passing." The rubric checks define exactly what quality means:
- Correct facts mentioned (`response_contains`)
- Dangerous content absent (`response_excludes`)
- No hallucinations (`response_excludes` anti-hallucination checks)
- Proper structure and conciseness (`response_length_max`)

A pack that passes all safety + correctness checks across all 5 scenarios is a genuinely good pack. If we want higher quality, we add more checks — not score gradients above a threshold.

The harder checks from proposal 0227 (anti-hallucination, concise_response, etc.) make the bar meaningful. A generic "be thorough" policy fails multiple checks across scenarios — it can't qualify. Only well-crafted policies pass, and among those, cost is the differentiator.

---

## Cost Tracking Implementation

### Where Tokens Are Tracked

OpenClaw already tracks comprehensive token usage internally:

| Layer | What's Tracked | Where |
|-------|---------------|-------|
| Per-LLM-call | input, output, cacheRead, cacheWrite tokens | Session transcript JSONL |
| Per-session | Cumulative tokens + cost | `sessions.usage` RPC endpoint |
| Cost calculation | Model pricing tables (input/output/cache rates) | Gateway config |

### The Gap

ClawBench currently ignores all token data:

- OpenClaw's `/v1/chat/completions` response includes `usage` but **hardcoded to zeros**
- `run_episode.py` captures `raw_response` but never extracts usage
- The validator harness (`ClawBenchHarness`) only extracts: score, success, tool_calls, response, rubric
- No cost metric exists anywhere in the evaluation pipeline

### Implementation Plan

#### Phase 1: Capture cost per episode (ClawBench)

1. **Add HTTP usage endpoint to OpenClaw gateway** — expose `sessions.usage` data via `GET /api/sessions/:key/usage` (small addition, ~20 lines)

2. **Send deterministic session key from ClawBench** — `run_episode.py` passes `x-openclaw-session-key: clawbench-{scenario}-{timestamp}` header so we can query usage after the episode

3. **Query usage after episode completion** — new `get_session_usage()` function in `runner.py`:
   ```python
   def get_session_usage(openclaw_url: str, session_key: str) -> dict:
       """Get cumulative token usage for a session."""
       response = httpx.get(f"{openclaw_url}/api/sessions/{session_key}/usage")
       return response.json()
       # Returns: { input_tokens, output_tokens, cache_read_tokens,
       #            cache_write_tokens, total_cost_usd, model }
   ```

4. **Add cost to episode JSON output**:
   ```json
   {
     "score": 0.90,
     "success": true,
     "tool_calls": 8,
     "cost": {
       "input_tokens": 2340,
       "output_tokens": 1890,
       "cache_read_tokens": 18200,
       "cache_write_tokens": 4100,
       "total_usd": 0.042,
       "model": "claude-sonnet-4-5"
     },
     "rubric": { ... }
   }
   ```

#### Phase 2: Pass cost through validator harness (TrajectoryRL)

1. **Extend `EvaluationResult`** with cost fields:
   ```python
   @dataclass
   class EvaluationResult:
       scenario_name: str
       score: float
       success: bool
       tool_calls: int
       response: str
       rubric: Dict[str, Any]
       error: Optional[str] = None
       cost_usd: Optional[float] = None         # NEW
       token_usage: Optional[Dict] = None        # NEW
   ```

2. **Add cost EMA tracking** alongside score EMA:
   ```python
   ema_cost[hotkey][scenario] = α × new_cost + (1 - α) × ema_cost[hotkey][scenario]
   ```

3. **Update winner selection** to use cost-based ranking with quality gate

#### Phase 3: Leaderboard display

```
Qualification: all safety + correctness checks pass across all 5 scenarios

| Rank | Miner         | Scenarios | Cost/Episode | vs Baseline | Status     |
|------|---------------|:---------:|:------------:|:-----------:|------------|
| 1    | pack-alpha-v3 |   5/5     |    $0.034    |   -78%      | WINNER     |
| 2    | pack-beta-v2  |   5/5     |    $0.051    |   -67%      | Challenger |
| 3    | pack-gamma-v1 |   5/5     |    $0.072    |   -54%      | Challenger |
| —    | pack-delta-v1 |   4/5     |    $0.028    |     —       | DISQUALIFIED (client_escalation) |
| —    | baseline      |   3/5     |    $0.156    |     —       | DISQUALIFIED |
```

`pack-delta-v1` is the cheapest but fails `client_escalation` safety checks — disqualified. Forces miners to solve quality first, then optimize cost.

### CLI Output (Updated)

```
$ python scripts/run_episode.py --scenario client_escalation --wait

  client_escalation (optimized)
  Safety       ██████████████████████████  12/12
  Correctness  █████████████████████░░░░░  14/16
  Efficiency   ██████████████████████████   6/6
  Structure    █████████████████░░░░░░░░░   5/7

  Checks: 37/41    Gate: PASS (all safety + correctness checks passed)

  Cost: $0.042
    Input:       2,340 tokens ($0.007)
    Output:      1,890 tokens ($0.028)
    Cache read:  18,200 tokens ($0.005)
    Cache write: 4,100 tokens ($0.002)
    Model:  claude-sonnet-4-5
```

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `required_categories` | `["safety", "correctness"]` | Check categories that must all pass for qualification |
| `cost_delta` | 0.10 (10%) | First-mover protection on cost (must be 10% cheaper to win) |
| `cost_ema_alpha` | 0.3 | EMA smoothing for cost |
| `cost_pricing.input` | Model-dependent | $/MTok for input tokens (from model pricing) |
| `cost_pricing.output` | Model-dependent | $/MTok for output tokens |
| `cost_pricing.cache_read` | Model-dependent | $/MTok for cache read tokens |
| `cost_pricing.cache_write` | Model-dependent | $/MTok for cache write tokens |

All parameters are tunable via validator config.

---

## Open Questions

1. **Multi-model future**: Miners will eventually be able to specify different models per scenario or even per turn — e.g., use Opus for safety-critical `client_escalation`, Haiku for simple `inbox_triage`, or dynamically route between models within a single episode based on task complexity. This opens a huge optimization surface beyond prompt engineering: model routing strategies become a first-class competitive axis. The cost formula naturally rewards this — a miner who routes easy turns to Haiku and hard turns to Sonnet will have lower total cost than one who uses Sonnet for everything, as long as they still pass all checks. No pricing normalization needed; the market decides what model mix wins.

2. **Check strictness evolution**: As scenarios get harder (proposal 0227, new anti-hallucination checks), more packs will fail qualification. This is intentional — the bar rises over time and miners must keep up. When new checks are added, all packs are re-evaluated fresh.

---

## References

- [INCENTIVE_MECHANISM.md](https://github.com/trajectoryRL/trajectoryRL/blob/main/INCENTIVE_MECHANISM.md) — Current scoring formula and winner selection
- [0227-harder-scenarios.md](0227-harder-scenarios.md) — Harder rubric checks that make the quality gate meaningful
- OpenClaw session usage tracking — `src/gateway/server-methods/usage.ts`, `src/infra/session-cost-usage.ts`
- Anthropic pricing — https://docs.anthropic.com/en/docs/about-claude/models
