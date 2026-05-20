"""
VulnChain Preference Pair Construction for DPO

Generates (chosen, rejected) pairs from SFT model outputs
using the VulnChain-Eval verifier as the signal.

Usage:
    python data/scripts/build_preference_pairs.py \
        --model ./outputs/sft_best \
        --prompts data/generated/eval.jsonl \
        --output data/generated/preference_pairs.jsonl \
        --samples_per_prompt 4
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from unsloth import FastLanguageModel

# Import our eval metrics
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from eval.metrics import score_sample


def generate_completions(model, tokenizer, prompt: str, n: int = 4, max_new_tokens: int = 512) -> list:
    """Generate n completions for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    completions = []
    for _ in range(n):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(decoded)
    return completions


def main(model_path: str, prompts_path: str, output_path: str, samples_per_prompt: int):
    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line.strip()))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pairs_written = 0
    pairs_discarded = 0

    with open(output_path, "w") as out:
        for sample in tqdm(prompts, desc="Building preference pairs"):
            prompt = f"{sample['instruction']}\n\n{sample['input']}"
            completions = generate_completions(model, tokenizer, prompt, n=samples_per_prompt)

            # Score each completion
            scores = [score_sample({"output": c})['composite'] for c in completions]

            best_idx = scores.index(max(scores))
            worst_idx = scores.index(min(scores))

            if best_idx == worst_idx:
                pairs_discarded += 1
                continue

            score_gap = scores[best_idx] - scores[worst_idx]
            if score_gap < 0.2:  # Discard too-similar pairs
                pairs_discarded += 1
                continue

            pair = {
                "prompt": prompt,
                "chosen": completions[best_idx],
                "rejected": completions[worst_idx],
                "chosen_score": scores[best_idx],
                "rejected_score": scores[worst_idx],
                "score_gap": score_gap,
            }
            out.write(json.dumps(pair) + "\n")
            pairs_written += 1

    print(f"\nPreference pairs written: {pairs_written}")
    print(f"Discarded (too similar): {pairs_discarded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples_per_prompt", type=int, default=4)
    args = parser.parse_args()
    main(args.model, args.prompts, args.output, args.samples_per_prompt)
