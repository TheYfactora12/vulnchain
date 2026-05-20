# VulnChain-7B: Post-Training a Small Language Model for Multi-Step API Vulnerability Chain Reasoning

**Status:** Work in Progress — Living Document  
**Author:** [Your Name]  
**Affiliation:** Independent Research  
**Date:** May 2026  

---

## Abstract

*[To be written after experiments are complete. Target: 150–200 words covering motivation, method, key results, and contribution.]*

---

## 1. Introduction

Modern API attack surfaces present a multi-step reasoning challenge that general-purpose language models are poorly equipped to handle. Effective exploit discovery requires chaining observations across authentication flows, authorization controls, object references, and business logic constraints — a form of structured reasoning that differs substantially from the code generation and question-answering tasks that dominate LLM benchmarks.

Recent work on LLM-based offensive security has focused on agent scaffolding and prompt engineering rather than model-level post-training. This leaves open the question of whether domain-specific supervised fine-tuning and preference optimization can materially improve a model's exploit chain reasoning capability.

This paper presents VulnChain-7B, a LoRA fine-tuned 7B language model trained on a novel dataset of structured API exploit reasoning traces. We make three contributions:

1. **VulnChain-Instruct**: A dataset of [N] synthetic exploit chain reasoning traces grounded in OWASP API Security Top 10 (2023), generated via a teacher-model pipeline and validated by a structural verifier.
2. **VulnChain-Eval**: A novel evaluation harness measuring four metrics specific to multi-step exploit reasoning (ECVR, FCR, TBE, SCS) where no public benchmark exists.
3. **Empirical findings**: Quantitative comparison of base model, SFT, and DPO model variants across all four metrics and three external benchmarks.

---

## 2. Related Work

*[To be written. Cover: LLM offensive security agents, Cybench, CyberGym, SEC-bench, CyberLLMInstruct, post-training methods (SFT, DPO, GRPO), safety trade-offs in security fine-tuning.]*

---

## 3. Dataset Construction

### 3.1 Seed Scenario Design

*[Describe the 40–60 seed scenarios, OWASP coverage, difficulty distribution.]*

### 3.2 Synthetic Generation Pipeline

*[Describe teacher model, prompt template, generation temperature, filtering.]*

### 3.3 Structural Validation

*[Describe validate_chains.py methodology, pass rate, distribution statistics.]*

### 3.4 Dataset Statistics

| Split | Samples | Avg Chain Length | Avg Output Words | OWASP Categories Covered |
|-------|---------|-----------------|-----------------|-------------------------|
| Train | — | — | — | — |
| Eval | — | — | — | — |
| Test | — | — | — | — |

---

## 4. Training Methodology

### 4.1 Base Model

*[Document chosen base model, justification, 4-bit quantization approach.]*

### 4.2 Supervised Fine-Tuning (SFT)

*[LoRA configuration, hyperparameter grid, training compute, W&B run IDs.]*

### 4.3 Direct Preference Optimization (DPO)

*[Preference pair construction methodology, beta selection, training config.]*

---

## 5. VulnChain-Eval Harness

### 5.1 Metric Definitions

**ECVR (Exploit Chain Validity Rate)**  
*[Formal definition, parser specification, validation methodology.]*

**FCR (False Chain Rate)**  
*[Ordering constraint rules, rationale for each constraint, edge cases.]*

**TBE (Token Budget Efficiency)**  
*[Formula, target word count rationale, normalization approach.]*

**SCS (Step Coverage Score)**  
*[Ground truth sourcing, partial credit handling.]*

### 5.2 Contamination Analysis

*[MinHash LSH results, Jaccard threshold, overlap statistics.]*

### 5.3 Human Eval Calibration

*[Inter-rater reliability (Pearson r between ECVR and human scores), sample size.]*

---

## 6. Results

### 6.1 VulnChain-Eval Results

| Model | ECVR ↑ | FCR ↓ | TBE ↑ | SCS ↑ | Composite ↑ |
|-------|--------|-------|-------|-------|-------------|
| Base (no fine-tuning) | — | — | — | — | — |
| VulnChain-SFT | — | — | — | — | — |
| VulnChain-DPO | — | — | — | — | — |

### 6.2 External Benchmark Results

| Model | Cybench | CyberGym | MMLU (preservation) |
|-------|---------|----------|--------------------|
| Base | — | — | — |
| VulnChain-SFT | — | — | — |
| VulnChain-DPO | — | — | — |

### 6.3 Stress Test Results

*[Distribution shift, hallucination, chain length, injection, contamination.]*

---

## 7. Discussion

*[Interpret results. What does the ECVR delta tell us? Why does DPO improve/not improve FCR? Where does the model fail at long horizons?]*

---

## 8. Limitations

*[Compute constraints, synthetic data quality, eval metric coverage, generalization to non-API surfaces.]*

---

## 9. Responsible Disclosure and Safety

All training data consists of reasoning traces describing vulnerability chaining logic, not operational exploit code. The model is not evaluated against live production systems. Model weights are released as LoRA adapters intended for security research in authorized environments only.

The safety/capability trade-off in security fine-tuning is a known concern. Readers are directed to the CyberLLMInstruct safety analysis (Adelsamir et al., 2025) for a detailed treatment of this trade-off.

---

## 10. Conclusion

*[To be written after experiments are complete.]*

---

## Appendix A: Experiment Logs

*[W&B run IDs, loss curves, hyperparameter tables.]*

## Appendix B: Example Model Outputs

*[5–10 example outputs per model variant, with metric scores.]*
