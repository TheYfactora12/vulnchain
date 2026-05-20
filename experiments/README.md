# Experiments

This directory stores experiment logs, run results, and analysis outputs.

## Directory Structure

```
experiments/
└── logs/
    ├── eval_{model_name}_{split}.jsonl   # VulnChain-Eval harness outputs
    ├── injection_test_results.jsonl      # Adversarial injection stress test
    └── stress_test_report.md             # Summary of all stress test results
```

## Experiment Naming Convention

W&B run names follow: `{stage}-v{version}-{key_param}`

Examples:
- `sft-v1-r16` — SFT run 1, LoRA rank 16
- `sft-v2-r64` — SFT run 2, LoRA rank 64
- `dpo-v1-beta0.1` — DPO run 1, beta=0.1
- `dpo-v2-beta0.05` — DPO run 2, beta=0.05

## Log files are gitignored

Large log files and model output JSONL files are excluded from version control.
To reproduce: run `eval/harness.py` with the published checkpoint.
