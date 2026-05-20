"""
VulnChain Benchmark Contamination Check

Uses MinHash LSH to detect overlap between VulnChain training data
and public benchmark datasets (Cybench, CyberGym).

Usage:
    python eval/stress_tests/contamination_check.py \
        --train data/generated/train.jsonl \
        --benchmark data/benchmarks/cybench_tasks.jsonl \
        --threshold 0.3
"""

import argparse
import json

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def text_to_shingles(text: str, k: int = 5) -> set:
    """Convert text to k-character shingles for MinHash."""
    text = text.lower().strip()
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def build_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in text_to_shingles(text):
        m.update(shingle.encode("utf8"))
    return m


def main(train_path: str, benchmark_path: str, threshold: float):
    # Load datasets
    train_data = []
    with open(train_path) as f:
        for line in f:
            s = json.loads(line)
            train_data.append((s.get("id", "?"), s.get("input", "") + " " + s.get("output", "")))

    benchmark_data = []
    with open(benchmark_path) as f:
        for line in f:
            s = json.loads(line)
            benchmark_data.append((s.get("id", "?"), s.get("description", "") + " " + s.get("prompt", "")))

    print(f"Training samples:   {len(train_data)}")
    print(f"Benchmark samples:  {len(benchmark_data)}")

    # Build LSH index from training data
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for idx, (doc_id, text) in enumerate(tqdm(train_data, desc="Building LSH index")):
        m = build_minhash(text)
        lsh.insert(f"train_{idx}", m)

    # Query with benchmark items
    contaminated = []
    for bench_id, text in tqdm(benchmark_data, desc="Checking contamination"):
        m = build_minhash(text)
        results = lsh.query(m)
        if results:
            contaminated.append({
                "benchmark_id": bench_id,
                "matching_train_ids": [train_data[int(r.split("_")[1])][0] for r in results],
            })

    contamination_rate = len(contaminated) / len(benchmark_data) * 100 if benchmark_data else 0

    print(f"\nContamination Check Results (threshold={threshold})")
    print(f"  Contaminated benchmark items: {len(contaminated)} / {len(benchmark_data)} ({contamination_rate:.1f}%)")

    if contaminated:
        print("\nSample contaminated pairs:")
        for pair in contaminated[:5]:
            print(f"  Benchmark: {pair['benchmark_id']} — matches train: {pair['matching_train_ids'][:2]}")
        print("\n  ACTION: Remove or rephrase overlapping training samples before publishing eval results.")
    else:
        print("  PASS: No significant overlap detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    main(args.train, args.benchmark, args.threshold)
