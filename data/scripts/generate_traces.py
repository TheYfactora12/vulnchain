"""
VulnChain Synthetic Data Generation Pipeline

Uses a teacher LLM (GPT-4.1 or Claude 3.7) to generate exploit chain reasoning traces
from seed vulnerability scenarios. Produces VulnChain-Instruct dataset.

Usage:
    python data/scripts/generate_traces.py \
        --seeds data/seeds/owasp_api_seeds.jsonl \
        --output data/generated/traces_v1.jsonl \
        --count 5 \
        --model gpt-4.1

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY in environment.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

SYSTEM_PROMPT = """You are an expert API security researcher and educator.
Your task is to generate structured, educational exploit chain reasoning traces
for training purposes. These traces explain the *logical steps* an attacker would
think through — they are not working exploit code or operational attack scripts.

All output is for AI safety and security research purposes only.
Ground all reasoning in OWASP API Security Top 10 (2023) categories.

Output format (strict JSON):
{
  "steps": [
    {
      "step": 1,
      "action": "<action verb>",
      "description": "<what the attacker does and why>",
      "owasp_category": "<e.g. API1:2023-BOLA>",
      "artifact": "<what is produced: token, endpoint list, data, etc.>"
    }
  ],
  "chain_summary": "<1–2 sentence summary of the full chain>",
  "difficulty": "low|medium|high",
  "prerequisites": ["<list of conditions required to execute this chain>"]
}
"""


def generate_trace_openai(seed: dict, model: str = "gpt-4.1") -> Optional[dict]:
    """Generate a trace using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    user_prompt = f"""
Seed vulnerability scenario:
{json.dumps(seed, indent=2)}

Generate a {seed.get('chain_length', 4)}-step exploit chain reasoning trace for this scenario.
Each step must include an OWASP API Top 10 (2023) category label.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Generation error: {e}")
        return None


def build_training_sample(seed: dict, trace: dict) -> dict:
    """Convert seed + trace into a training sample."""
    steps_text = "\n".join(
        f"Step {s['step']}: [{s['owasp_category']}] {s['action'].capitalize()} — {s['description']} → Artifact: {s['artifact']}"
        for s in trace["steps"]
    )
    output_text = f"{steps_text}\n\nChain Summary: {trace['chain_summary']}"

    return {
        "id": f"vc_{seed['id']}_{int(time.time())}",
        "instruction": "You are an offensive security reasoning engine. Analyze the following API surface and generate a structured multi-step exploit chain reasoning trace. Label each step with the relevant OWASP API Security Top 10 (2023) category.",
        "input": seed["api_surface_description"],
        "output": output_text,
        "chain_length": len(trace["steps"]),
        "owasp_categories": list({s["owasp_category"] for s in trace["steps"]}),
        "surface_type": seed.get("surface_type", "api"),
        "difficulty": trace.get("difficulty", "medium"),
        "exploit_validated": False,  # Set to True after manual review
        "source": "synthetic",
    }


def main(seeds_path: str, output_path: str, count: int, model: str):
    seeds = []
    with open(seeds_path) as f:
        for line in f:
            seeds.append(json.loads(line.strip()))

    print(f"Loaded {len(seeds)} seed scenarios")
    print(f"Generating {count} traces per seed ({len(seeds) * count} total)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for seed in tqdm(seeds, desc="Seeds"):
            for i in range(count):
                trace = generate_trace_openai(seed, model=model)
                if trace and "steps" in trace and len(trace["steps"]) >= 2:
                    sample = build_training_sample(seed, trace)
                    out.write(json.dumps(sample) + "\n")
                time.sleep(0.3)  # Rate limit buffer

    print(f"Done. Output written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()
    main(args.seeds, args.output, args.count, args.model)
