"""
VulnChain Chain Structure Validator

Validates that generated training samples meet structural requirements
before being used for fine-tuning.

Usage:
    python data/scripts/validate_chains.py \
        --input data/generated/traces_v1.jsonl \
        --output data/generated/traces_v1_validated.jsonl \
        --report
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple

# OWASP API Security Top 10 (2023) valid labels
VALID_OWASP_LABELS = {
    "API1:2023-BOLA",
    "API2:2023-Authentication",
    "API3:2023-BOPLA",
    "API4:2023-Unrestricted-Resource-Consumption",
    "API5:2023-BFLA",
    "API6:2023-Unrestricted-Access-to-Sensitive-Business-Flows",
    "API7:2023-SSRF",
    "API8:2023-Security-Misconfiguration",
    "API9:2023-Improper-Inventory-Management",
    "API10:2023-Unsafe-Consumption-of-APIs",
}

ACTION_VERBS = {
    "enumerate", "discover", "exploit", "exfiltrate", "bypass", "inject",
    "manipulate", "probe", "authenticate", "escalate", "forge", "intercept",
    "redirect", "scan", "fuzz", "replay", "chain", "extract",
}

# Logical step ordering constraints
# Key step type cannot appear before value step types (simplified)
ORDER_CONSTRAINTS = {
    "exfiltrate": {"enumerate", "discover", "probe", "scan"},  # Must recon first
    "escalate": {"authenticate", "bypass", "exploit"},  # Must have initial access
}


def validate_sample(sample: dict) -> Tuple[bool, list]:
    """
    Returns (is_valid, list_of_failure_reasons).
    """
    failures = []

    # 1. Required fields
    for field in ["instruction", "input", "output", "chain_length"]:
        if field not in sample or not sample[field]:
            failures.append(f"Missing required field: {field}")

    if failures:
        return False, failures

    output = sample["output"]
    
    # 2. Minimum step count
    step_pattern = re.findall(r"Step \d+:", output)
    if len(step_pattern) < 2:
        failures.append(f"Fewer than 2 steps detected: found {len(step_pattern)}")

    # 3. OWASP label presence
    found_owasp = re.findall(r"API\d+:2023-[\w-]+", output)
    if not found_owasp:
        failures.append("No OWASP API Top 10 (2023) labels found in output")
    else:
        invalid = [l for l in found_owasp if l not in VALID_OWASP_LABELS]
        if invalid:
            failures.append(f"Invalid OWASP labels: {invalid}")

    # 4. Action verb presence
    output_lower = output.lower()
    found_verbs = [v for v in ACTION_VERBS if v in output_lower]
    if len(found_verbs) < 2:
        failures.append(f"Too few action verbs: found {found_verbs}")

    # 5. Output length
    token_estimate = len(output.split())
    if token_estimate < 80:
        failures.append(f"Output too short: ~{token_estimate} words")
    if token_estimate > 1200:
        failures.append(f"Output too long: ~{token_estimate} words — may waste context")

    # 6. Chain length consistency
    if len(step_pattern) != sample.get("chain_length", 0):
        # Soft warning, not a failure
        pass

    return len(failures) == 0, failures


def main(input_path: str, output_path: str, report: bool):
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    valid, invalid = [], []
    failure_log = []

    for sample in samples:
        is_valid, failures = validate_sample(sample)
        if is_valid:
            valid.append(sample)
        else:
            invalid.append(sample)
            failure_log.append({"id": sample.get("id", "?"), "failures": failures})

    # Write validated output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in valid:
            f.write(json.dumps(s) + "\n")

    validity_rate = len(valid) / len(samples) * 100 if samples else 0
    print(f"\nValidation Results")
    print(f"  Total samples:   {len(samples)}")
    print(f"  Valid:           {len(valid)} ({validity_rate:.1f}%)")
    print(f"  Invalid:         {len(invalid)}")
    print(f"  Pass threshold:  85% — {'PASS' if validity_rate >= 85 else 'FAIL — regenerate before training'}")

    if report and failure_log:
        print("\nTop failure reasons:")
        reason_counts = {}
        for entry in failure_log:
            for reason in entry["failures"]:
                key = reason.split(":")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}x  {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    main(args.input, args.output, args.report)
