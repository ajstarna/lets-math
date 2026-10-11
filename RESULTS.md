# Experiment Results

This document is the running, human-readable record of experiment results. Detailed local evaluation artifacts, including per-example predictions, are saved under ignored `outputs/evals/`.

## Evaluation protocol

- **Dataset:** `openai/gsm8k` (`main` configuration).
- **Development split:** deterministic 90/10 split of GSM8K train with seed 42: 6,725 experiment-train examples and 748 validation examples.
- **Generation evaluation:** greedy decoding (`do_sample=False`), `max_new_tokens=512`, generation batch size 16.
- **Metric:** strict trailing `#### <answer>` extraction and exact string match to GSM8K's gold final answer.
- **Official GSM8K test:** untouched so far. These are development results, not final test results.

## Base model versus one-epoch SFT

| Condition | Exact answer accuracy | Valid `####` format | Mean completion tokens | Truncation rate |
|---|---:|---:|---:|---:|
| Base Qwen2.5 1.5B | 222 / 748 (**29.7%**) | 333 / 748 (44.5%) | 196.2 | 1.2% |
| SFT, approximately 1 epoch | 531 / 748 (**71.0%**) | 748 / 748 (100.0%) | 121.2 | 0.0% |
| SFT delta | **+41.3 pp** | +55.5 pp | -75.0 | -1.2 pp |

### SFT configuration

- Base checkpoint: `Qwen/Qwen2.5-1.5B`.
- LoRA target modules: `q_proj`, `v_proj`; rank 8; alpha 16; dropout 0.05.
- Learning rate: `1e-4`.
- Maximum SFT length: 576 tokens.
- Physical batch size: 1; gradient accumulation: 8; effective batch size: 8.
- Training: 840 optimizer steps, approximately one pass over experiment-train.
- Best validation-loss checkpoint: final step 840, validation loss 0.3184 (initial loss 0.4460).
- Wall-clock training time: 12m 01s on an RTX 5070 Ti.

### Local detailed artifacts

- `outputs/evals/base-qwen2.5-1.5b-validation-748-greedy-512-batch16.json`
- `outputs/evals/sft-qwen2.5-1.5b-gsm8k-1epoch-validation-748-greedy-512-batch16.json`

## Interpretation

SFT substantially improves held-out exact-answer accuracy and makes output formatting fully reliable under this evaluation setup. It also produces shorter completions, which explains why the SFT evaluation ran faster than the base-model evaluation.

The next primary comparison is SFT versus SFT + hand-rolled GRPO, using the same experiment-train prompt pool and the same held-out validation protocol.
