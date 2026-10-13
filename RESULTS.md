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

## SFT + GRPO

| Condition | Exact answer accuracy | Valid `####` format | Mean completion tokens | Truncation rate |
|---|---:|---:|---:|---:|
| SFT | 531 / 748 (71.0%) | 748 / 748 (100.0%) | 121.2 | 0.0% |
| SFT + GRPO | 560 / 748 (**74.9%**) | 746 / 748 (99.7%) | 124.0 | 2 / 748 (0.3%) |

GRPO improved full-validation accuracy by **29 answers / 3.9 percentage points** over SFT in this single run. Repeated seeds and ablations are still needed to establish robustness.

### GRPO configuration

- Initialization: the one-epoch SFT loss-selected adapter above.
- 400 optimizer updates; 2 updates per rollout; 200 training prompts.
- Prompt batch size 1; group size 8; 1,600 sampled completions.
- Rollout cap 384 tokens; sampling temperature 0.7.
- Learning rate `1e-5`; KL coefficient 0.04; clip epsilon 0.2; gradient norm cap 1.0.
- Reward: `+1` for exact correctness and `+0.05` for valid final-answer format.
- Reference policy: frozen initial SFT LoRA weights.
- Checkpoint selection: greedy accuracy on the fixed first 100 validation examples, every 100 updates, with a 512-token cap and generation batch size 16.
- Final probe accuracy: 74%, versus 68% before GRPO. Full validation evaluated the best saved adapter.
- Training wall time: 19m 25s on an RTX 5070 Ti.
- This run preceded GRPO resumable-state support; its last checkpoint is adapter-only.

### Local detailed artifact

- `outputs/evals/grpo-qwen2.5-1.5b-g8-400-validation-748-greedy-512-batch16.json`

The official test set remains untouched. Full validation includes the 100 examples used for checkpoint selection, so it is a development comparison, not an independent final test.
