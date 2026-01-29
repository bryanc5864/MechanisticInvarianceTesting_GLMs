#!/usr/bin/env python3
"""Experiment 3.2: Spacing Sensitivity
Test if model knows optimal 17bp spacing between -35 and -10.
"""

import json
import random
import numpy as np
import torch
from pathlib import Path

random.seed(42)
np.random.seed(42)


def generate_background(length, at_fraction=0.55):
    """Generate sequence with specified AT content."""
    seq = []
    for _ in range(length):
        if random.random() < at_fraction:
            seq.append(random.choice('AT'))
        else:
            seq.append(random.choice('GC'))
    return ''.join(seq)


def generate_promoter_with_spacing(spacing):
    """Generate promoter with specified spacing between -35 and -10."""
    # -35 box at fixed position 30
    minus_35_start = 30
    minus_35 = "TTGACA"  # 6bp

    # -10 box position depends on spacing
    minus_10_start = minus_35_start + 6 + spacing
    minus_10 = "TATAAT"  # 6bp (intact for this test)

    # Need sequence long enough
    total_len = max(100, minus_10_start + 6 + 20)

    seq = list(generate_background(total_len))

    # Insert -35 box
    for i, base in enumerate(minus_35):
        seq[minus_35_start + i] = base

    # Insert -10 box
    for i, base in enumerate(minus_10):
        seq[minus_10_start + i] = base

    return ''.join(seq[:100])  # Truncate to 100bp


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Spacings to test (optimal is 17bp)
    spacings = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    n_per_spacing = 50

    # Generate sequences
    sequences = {}
    for spacing in spacings:
        sequences[spacing] = [generate_promoter_with_spacing(spacing) for _ in range(n_per_spacing)]

    # Load HyenaDNA
    print("Loading HyenaDNA...")
    tokenizer = AutoTokenizer.from_pretrained(
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "LongSafari/hyenadna-medium-160k-seqlen-hf",
        trust_remote_code=True
    )
    model.to('cuda')
    model.eval()

    def compute_ll(sequence):
        with torch.no_grad():
            inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to('cuda')
            outputs = model(input_ids)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
            target_ids = input_ids[0, 1:]
            ll = log_probs.gather(1, target_ids.unsqueeze(1)).sum().item()
        return ll

    # Compute LL for all sequences
    print("Computing log-likelihoods...")
    results = {}
    for spacing in spacings:
        print(f"  Spacing = {spacing}bp...")
        lls = [compute_ll(s) for s in sequences[spacing]]
        results[spacing] = {
            'mean': np.mean(lls),
            'std': np.std(lls),
            'values': lls
        }

    # Print results
    print("\n" + "="*70)
    print("EXPERIMENT 3.2: SPACING SENSITIVITY RESULTS")
    print("="*70)

    print("\nMean LL by spacing (optimal is 17bp):")
    print("-"*50)
    print(f"{'Spacing':<12} {'Mean LL':<15} {'Std':<10} {'vs Optimal':<12}")
    print("-"*50)

    optimal_mean = results[17]['mean']
    for spacing in spacings:
        mean = results[spacing]['mean']
        std = results[spacing]['std']
        diff = mean - optimal_mean
        marker = " <-- OPTIMAL" if spacing == 17 else ""
        print(f"{spacing:<12} {mean:<15.3f} {std:<10.3f} {diff:+.3f}{marker}")

    # Find peak
    means = [results[s]['mean'] for s in spacings]
    peak_spacing = spacings[np.argmax(means)]

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    print(f"\n1. Peak LL spacing:       {peak_spacing}bp")
    print(f"   (Optimal is 17bp)")

    print(f"\n2. Range of means:")
    print(f"   Max: {max(means):.3f} (at {peak_spacing}bp)")
    print(f"   Min: {min(means):.3f} (at {spacings[np.argmin(means)]}bp)")
    print(f"   Difference: {max(means) - min(means):.3f}")

    # If mechanistic, expect clear peak at 17 with decay
    # If not, expect flat or noisy

    from scipy import stats
    corr, p = stats.pearsonr(spacings, means)
    print(f"\n3. Correlation (spacing vs LL): r = {corr:.3f}, p = {p:.4f}")

    # Quadratic fit (expect peak at 17)
    coeffs = np.polyfit(spacings, means, 2)
    peak_from_fit = -coeffs[1] / (2 * coeffs[0])
    print(f"4. Peak from quadratic fit: {peak_from_fit:.1f}bp")

    # Save results
    output_path = Path('data/results/spacing_results.json')
    save_results = {str(k): {'mean': v['mean'], 'std': v['std']} for k, v in results.items()}
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
