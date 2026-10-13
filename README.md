# lets-math

Fine-tuning Qwen2.5-1.5B on GSM8K, then training it with GRPO and checked final answers.

I built the SFT and GRPO loops in PyTorch to understand what's happening inside training. Transformers and PEFT handle model loading, generation, and LoRA. All runs so far use one RTX 5070 Ti (16 GB).

## Results

| Model | Exact-answer accuracy | Valid answer format |
|---|---:|---:|
| Base Qwen2.5-1.5B | 29.7% | 44.5% |
| SFT, ~1 epoch | 71.0% | 100.0% |
| SFT + GRPO, 400 updates | **74.9%** | 99.7% |

Evaluated on 748 held-out validation examples with greedy decoding and a 512-token cap. Training uses the other 6,725 examples from a fixed GSM8K train split. The official test set is untouched.

GRPO added 29 correct answers over SFT on the full validation split (+3.9 percentage points). See [RESULTS.md](RESULTS.md) for experiment details. GSM8K may be present in the base model's pretraining data, so these are benchmark adaptation results, not evidence of general reasoning ability.

## Implementation

- **SFT:** completion-only loss, LoRA, gradient accumulation, validation, and resumable checkpoints.
- **GRPO:** grouped rollouts, normalized advantages, clipped policy updates, and KL against a frozen SFT reference adapter.
- **Rewards:** `+1` for an exact answer, `+0.05` for valid trailing `#### <answer>` format.
- **Tracking:** local metrics and per-example evaluation JSON; optional W&B.

## Getting started

Uses Python 3.14 and PyTorch with CUDA 12.8 on Linux.

```bash
uv sync --dev
uv run pytest -q
uv run lets-math inspect-data --show-token-stats --examples 0

# Train an SFT adapter
uv run lets-math train-sft --output-dir outputs/sft --max-steps 840

# Run GRPO from that adapter
uv run lets-math train-grpo \
  --adapter outputs/sft/checkpoint-best \
  --output-dir outputs/grpo \
  --max-steps 400 --group-size 8 --updates-per-rollout 2

# Evaluate the best GRPO adapter on the full validation split
uv run lets-math evaluate \
  --adapter outputs/grpo/checkpoint-best \
  --examples 748 --max-new-tokens 512 --generation-batch-size 16 \
  --show-examples 0 --output-json outputs/evals/grpo-validation.json
```

Omit `--adapter` for base-model evaluation. Checkpoints and run artifacts are local under ignored `outputs/`; trained adapters are not bundled. Use `--help` on each command for options.

## Code

Under [`src/lets_math/`](src/lets_math/): `data.py` handles loading and splits, `sft.py` and `grpo.py` contain the training loops, `evaluate.py` generates and scores answers, and `answers.py` defines the final-answer parser.
