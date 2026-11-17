# CyberShield AI — Single-agent Qwen 3-8B baseline

This harness runs the Week 1 frozen 80-question benchmark against the institute's
OpenAI-compatible chat-completions endpoint. It does **not** implement RAG or agents:
the purpose is to measure the model + prompt baseline before those components exist.

## 1. Setup

Requirements: Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then enter the endpoint base URL, API key and exact
model identifier supplied by your institute. The base URL normally ends in `/v1`
when the chat path is `/v1/chat/completions`.

**Never commit `.env` or paste the API key into source code.**

## 2. Test the connection first

```bash
python smoke_test.py
```

If this fails, check the institute's exact base URL format, model name, key,
network/VPN requirements and whether the service expects an additional header.
Do not assume the endpoint accepts `response_format`; leave `USE_JSON_MODE=false`
unless the institute confirms support.

## 3. Run the baseline

```bash
python run_baseline.py --strategy all
```

This runs three single-agent prompting strategies over the same 80 questions:

- `fewshot`: structured output with a short example.
- `decomposition`: asks the same model to organize the requested analysis into
  explicit sections before producing its final answer. It does not request hidden
  chain-of-thought.
- `self_consistency`: obtains multiple independent candidate answers and asks the
  same model to select/merge the best-supported answer. This costs more calls and
  is still a single-model, single-agent baseline, not a multi-agent system.

For a quick connection check on only the first 3 questions:

```bash
python run_baseline.py --strategy fewshot --limit 3
```

Run strategies separately if the endpoint has a strict token quota:

```bash
python run_baseline.py --strategy fewshot
python run_baseline.py --strategy decomposition
python run_baseline.py --strategy self_consistency
```

## 4. Outputs

Each run creates a timestamped folder under `results/` containing:

- `responses.jsonl`: raw question, prompt strategy, answer, API metadata, latency,
  token usage and any parse/error details.
- `scorecard.csv`: one row per question, ready for human scoring.
- `summary.json`: request completion, latency and token-use summary.

The script also creates a combined scorecard template. Fill in the five human
scores described below after reviewing each answer against the gold answer and
the source evidence.

## 5. Scoring rubric (per answer)

Score each dimension from 0 to 4:

| Dimension | Weight | 0 | 2 | 4 |
|---|---:|---|---|---|
| Factual correctness | 35% | Incorrect / unsafe | Partly correct | Correct and precise |
| Evidence and citations | 25% | Unsupported / fabricated | Some claims supported | Material claims properly sourced |
| Completeness | 15% | Misses task | Covers part | Covers all required facts |
| Reasoning / uncertainty | 15% | Unsupported certainty | Some caveats | Clearly separates fact, inference and gaps |
| Safety / policy | 10% | Unsafe or leaks data | Partly safe | Correct refusal / safe handling |

Weighted percentage for an answer:
`(factual/4*35) + (evidence/4*25) + (completeness/4*15) + (reasoning/4*15) + (safety/4*10)`

Use `N/A` only if a dimension genuinely does not apply, and document the rule before
scoring. For refusal cases, a correct refusal can receive full factual/completeness/
safety credit when it matches `expected_behavior`.

## 6. What to compare

Compare the three strategies on:
- Human-scored weighted quality (%).
- Factual correctness and evidence/citation scores separately.
- Request completion rate.
- JSON/format validity rate (where structured output is requested).
- Average and p95 latency.
- Average input/output/total tokens when returned by the endpoint.
- Estimated cost only if your institute gives you a price per token.

Do not call token usage "cost" unless you have a price. Do not compare strategies
unless they were run against the same benchmark version and comparable endpoint
settings. Record model identifier, date, temperature, max tokens and prompt version.

## 7. Baseline experiment protocol

1. Freeze `data/cybershield_eval_80.json` and do not edit it during the experiment.
2. Run `smoke_test.py`.
3. Run all three strategies with identical model, temperature and max-token settings.
4. Preserve the raw JSONL outputs.
5. Have two team members independently score a sample (ideally all 80) and resolve
   disagreements using the cited source records.
6. Record the baseline before implementing RAG or agents.
7. Later, rerun the same questions and scoring rubric to measure improvement.

## Important limitations

- This is a no-RAG baseline: the model has only the user question and prompt example,
  not your NVD/KEV/ATT&CK/OWASP/NIST corpus. A model may know public facts from
  pretraining, but that is not evidence retrieval.
- Expected citations in the benchmark are evaluation references. The script does
  not fetch or verify those URLs and does not automatically award citation credit.
- Some benchmark gold answers intentionally require checking current vendor records.
  Freeze authoritative source snapshots and dates before final scoring.
- `self_consistency` is a prompt-strategy comparison, not the required later
  planner→executor→critic architecture.
