---
language:
- en
license: mit
tags:
- security
- cybersecurity
- offensive-security
- fine-tuned
- lora
- api-security
- owasp
base_model: mistralai/Mistral-7B-Instruct-v0.3
---

# VulnChain-7B

## Model Description

VulnChain-7B is a LoRA-adapted 7B language model fine-tuned for multi-step API vulnerability chain reasoning. It is trained to produce structured, step-by-step exploit reasoning traces grounded in the OWASP API Security Top 10 (2023) taxonomy.

This model is a research artifact produced as part of the VulnChain independent research project.

## Intended Use

- **Intended users**: Security researchers, red team practitioners, AI safety researchers
- **Intended use cases**: Educational exploit chain reasoning, security model evaluation, AI-assisted vulnerability assessment in authorized environments
- **Out-of-scope uses**: Unauthorized penetration testing, production exploit generation, use against systems without explicit permission

## Training Data

Fine-tuned on VulnChain-Instruct: a dataset of ~[N] synthetic exploit chain reasoning traces generated from OWASP API Security Top 10 (2023) seed scenarios. Training data consists of reasoning traces only — no working exploit code or operational attack scripts.

## Evaluation

Evaluated using VulnChain-Eval harness:

| Metric | Base Model | VulnChain-SFT | VulnChain-DPO |
|--------|-----------|--------------|---------------|
| ECVR ↑ | — | — | — |
| FCR ↓ | — | — | — |
| TBE ↑ | — | — | — |
| SCS ↑ | — | — | — |

## Known Limitations

- Trained on synthetic data; real-world generalization requires further evaluation
- Performance degrades on non-API surfaces (web, mobile) — distribution shift documented in stress tests
- Chain validity decreases at > 6-step horizons — see stress test report
- Should not be used as the sole signal in any security assessment

## Safety and Responsible Use

This model is for authorized security research only. Do not deploy against systems you do not own or have explicit permission to test. See the full responsible disclosure section in the [research paper](paper/vulnchain_preprint.md).
