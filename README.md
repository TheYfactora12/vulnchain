# VulnChain

> Post-training a 7B language model for multi-step API vulnerability chain reasoning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Coming%20Soon-orange)](https://huggingface.co/)

---

## What Is This?

VulnChain is an independent ML research project exploring whether a small (7B parameter) language model can be post-trained to reason about multi-step API vulnerability chains — the kind of chained exploit reasoning that scanners miss and humans run out of time to find.

The project produces three artifacts:

1. **VulnChain-SFT** — A LoRA fine-tuned 7B model trained on API exploit chain reasoning traces
2. **VulnChain-Eval** — A custom evaluation harness measuring multi-step exploit quality (ECVR, FCR, TBE, SCS)
3. **VulnChain-DPO** — A preference-optimized variant using DPO with rule-based verifier signals

No working exploit code is generated or published. All training data is pseudo-malicious reasoning traces grounded in OWASP API Security Top 10 (2023).

---

## Repository Structure

```
vulnchain/
├── data/
│   ├── seeds/              # Seed vulnerability scenarios (OWASP-aligned)
│   ├── generated/          # Synthetic reasoning traces (gitignored)
│   └── scripts/
│       ├── generate_traces.py      # Teacher-model synthetic data pipeline
│       ├── validate_chains.py      # Chain structure validator
│       └── build_preference_pairs.py  # DPO pair construction
├── training/
│   ├── sft_train.py        # Supervised fine-tuning (Unsloth + TRL)
│   ├── dpo_train.py        # DPO alignment training
│   └── configs/
│       ├── sft_config.yaml
│       └── dpo_config.yaml
├── eval/
│   ├── harness.py          # VulnChain-Eval harness (ECVR, FCR, TBE, SCS)
│   ├── metrics.py          # Metric definitions and scorers
│   ├── benchmarks/
│   │   ├── cybench_runner.py
│   │   └── cybergym_runner.py
│   └── stress_tests/
│       ├── distribution_shift.py
│       ├── hallucination_test.py
│       ├── chain_length_test.py
│       ├── adversarial_injection.py
│       └── contamination_check.py
├── experiments/
│   └── logs/               # W&B run exports (gitignored)
├── paper/
│   ├── vulnchain_preprint.md   # Research writeup (paper-style)
│   └── model_card.md           # HuggingFace model card
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_quickstart.ipynb
│   ├── 03_eval_harness_demo.ipynb
│   └── 04_stress_test_analysis.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── ROADMAP.md
```

---

## Quickstart

```bash
git clone https://github.com/TheYfactora12/vulnchain.git
cd vulnchain
pip install -r requirements.txt

# Generate synthetic training data
python data/scripts/generate_traces.py --count 100 --output data/generated/traces_v1.jsonl

# Validate chain structure
python data/scripts/validate_chains.py --input data/generated/traces_v1.jsonl

# Run SFT training
python training/sft_train.py --config training/configs/sft_config.yaml

# Run eval harness
python eval/harness.py --model ./outputs/sft_checkpoint --split test
```

---

## Evaluation Metrics (VulnChain-Eval)

| Metric | Abbreviation | Description |
|--------|-------------|-------------|
| Exploit Chain Validity Rate | ECVR | % of outputs parseable as structured multi-step chains |
| False Chain Rate | FCR | % of chains proposing non-exploitable sequences |
| Token Budget Efficiency | TBE | Chain completeness score per 1,000 tokens |
| Step Coverage Score | SCS | % of relevant OWASP categories identified and chained |

---

## Research Framing

This project is designed to fill a gap noted in current LLM offensive security research: most work either uses general-purpose models without domain post-training, or builds proprietary systems without releasing evaluation methodology. VulnChain aims to publish both the model adapter and the evaluation harness as reproducible artifacts.

See [ROADMAP.md](ROADMAP.md) for the full project timeline.
See [paper/vulnchain_preprint.md](paper/vulnchain_preprint.md) for the evolving research writeup.

---

## Safety and Responsible Disclosure

This project does not publish working exploits, complete attack scripts, or vulnerability-specific payloads. All training data represents *reasoning traces* — structured thinking about how vulnerability categories chain together — not operational exploit code. Model outputs should be evaluated in controlled, authorized environments only.

---

## Status

- [x] Repository initialized
- [x] Project structure scaffolded
- [ ] Seed dataset construction (Phase 1)
- [ ] Synthetic data generation pipeline (Phase 1)
- [ ] SFT training run (Phase 2)
- [ ] VulnChain-Eval harness (Phase 3)
- [ ] DPO alignment (Phase 4)
- [ ] Stress test battery (Phase 5)
- [ ] arXiv preprint submission (Phase 6)

---

## License

MIT License. See [LICENSE](LICENSE).
