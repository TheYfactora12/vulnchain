"""
VulnChain-Eval Harness

Unified evaluation runner. Loads a model checkpoint and evaluates
against the VulnChain-Eval test set.

Usage:
    python eval/harness.py --model ./outputs/sft_best --split test
    python eval/harness.py --model ./outputs/dpo_v1_final --split test --compare ./outputs/sft_best
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel

from eval.metrics import score_sample


def load_eval_data(split: str = "test") -> list:
    path = f"data/generated/{split}.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def run_model_eval(model_path: str, samples: list, max_new_tokens: int = 512) -> list:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    for sample in tqdm(samples, desc=f"Evaluating {Path(model_path).name}"):
        prompt = f"{sample['instruction']}\n\n{sample['input']}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Greedy for reproducibility
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        scored = score_sample(
            {"output": generated},
            ground_truth_categories=sample.get("owasp_categories", []),
        )
        results.append({
            "id": sample.get("id", "?"),
            "generated": generated,
            "ground_truth_categories": sample.get("owasp_categories", []),
            **scored,
        })
    return results


def print_report(model_name: str, results: list):
    metrics = ["ecvr", "fcr", "tbe", "scs", "composite"]
    print(f"\n{'='*60}")
    print(f"VulnChain-Eval Report: {model_name}")
    print(f"Samples: {len(results)}")
    print(f"{'='*60}")
    for m in metrics:
        vals = [r[m] for r in results]
        label = "(lower=better)" if m == "fcr" else ""
        print(f"  {m.upper():12s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  {label}")
    print(f"{'='*60}")


def main(model_path: str, split: str, compare_path: str = None):
    samples = load_eval_data(split)
    print(f"Loaded {len(samples)} {split} samples")

    results_a = run_model_eval(model_path, samples)
    print_report(Path(model_path).name, results_a)

    if compare_path:
        results_b = run_model_eval(compare_path, samples)
        print_report(Path(compare_path).name, results_b)

        # Delta report
        print("\nDelta (model_a - model_b):")
        for m in ["ecvr", "fcr", "scs", "composite"]:
            delta = np.mean([r[m] for r in results_a]) - np.mean([r[m] for r in results_b])
            direction = "↑" if (m != "fcr" and delta > 0) or (m == "fcr" and delta < 0) else "↓"
            print(f"  {m.upper():12s}  {delta:+.4f} {direction}")

    # Save results
    out_path = f"experiments/logs/eval_{Path(model_path).name}_{split}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results_a:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    main(args.model, args.split, args.compare)
