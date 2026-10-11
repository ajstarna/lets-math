# lets-math

A small, reproducible study of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) on math problems with deterministic answer verification.

See [RESULTS.md](RESULTS.md) for the running experiment record.

## Research question

How much does outcome-supervised GRPO improve a small Qwen model over its base and SFT-only baselines, and does that improvement survive controlled distribution shifts?

## Intended experiment matrix

1. Base model evaluation.
2. SFT on worked math solutions.
3. SFT followed by GRPO with exact-answer reward.
4. Small ablations: GRPO group size and KL coefficient.

GSM8K is a useful baseline but is contamination-prone for modern models. The main generalization evidence should come from a generated math environment with held-out templates and difficulty ranges.

## Planned layout

- `src/lets_math/`: training, evaluation, data, reward, and experiment code.
- `tests/`: verifier and training-loop tests.
- `outputs/` (ignored): checkpoints, predictions, and run artifacts.
- `data/` (ignored): downloaded/generated local data.

## Setup

```bash
uv sync --dev
uv run lets-math inspect-data --show-token-stats
```

Run a small local SFT smoke test:

```bash
uv run lets-math train-sft \
  --output-dir outputs/sft-smoke \
  --max-steps 2 --validate-every 1 --log-every 1
```

Add `--wandb-project lets-math` to log a run to Weights & Biases.

Evaluate a base checkpoint or a saved adapter with deterministic greedy decoding:

```bash
uv run lets-math evaluate --examples 100
uv run lets-math evaluate --adapter outputs/sft-smoke/checkpoint-best --examples 100
```

Optional Weights & Biases credentials belong in an ignored `.env` file.

## Status

Repository scaffold only. First implementation target: a strict final-answer extractor and verifier with tests.
