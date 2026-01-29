#!/usr/bin/env python3
"""Experiment 3.1: Fine-Grained Positional Sweep
Test if model cares about UP element position.
"""

import json
import random
import numpy as np
import torch
from pathlib import Path

random.seed(42)
np.random.seed(42)

MINUS_35_START = 30
MINUS_10_START = 53


def generate_background(length, at_fraction=0.55):
    """Generate sequence with specified AT content."""
    seq = []
    for _ in range(length):
        if random.random() < at_fraction:
            seq.append(random.choice('AT'))
        else:
            seq.append(random.choice('GC'))
    return ''.join(seq)


def generate_promoter_with_up_at_position(up_position, up_element="AAAAAAGAG"):
    """Generate promoter with UP element at specified position."""
    seq = list(generate_background(100))

    # Insert -35 box (fixed at position 30)
    minus_35 = "TTGACA"
    for i, base in enumerate(minus_35):
        seq[MINUS_35_START + i] = base

    # Insert -10 box (broken, fixed at position 53)
    minus_10 = "TGTAAT"
    for i, base in enumerate(minus_10):
        seq[MINUS_10_START + i] = base

    # Insert extended -10 (fixed at position 50)
    ext_10 = "TGT"
    for i, base in enumerate(ext_10):
        seq[50 + i] = base

    # Insert UP element at variable position
    if up_position is not None:
        for i, base in enumerate(up_element):
            if up_position + i < 100:
                seq[up_position + i] = base

    return ''.join(seq)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Positions to test (correct position is 15)
    positions = [0, 5, 10, 15, 20, 25, 35, 45, 60, 70, 80, 90, None]
    n_per_position = 50

    # Generate sequences
    sequences = {}
    for pos in positions:
        sequences[pos] = [generate_promoter_with_up_at_position(pos) for _ in range(n_per_position)]

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
    for pos in positions:
        pos_label = str(pos) if pos is not None else "None"
        print(f"  UP position = {pos_label}...")
        lls = [compute_ll(s) for s in sequences[pos]]
        results[pos_label] = {
            'mean': np.mean(lls),
            'std': np.std(lls),
            'values': lls
        }

    # Print results
    print("\n" + "="*70)
    print("EXPERIMENT 3.1: POSITIONAL SWEEP RESULTS")
    print("="*70)

    print("\nMean LL by UP element position:")
    print("-"*50)
    print(f"{'Position':<12} {'Mean LL':<15} {'Std':<10} {'vs Correct':<12}")
    print("-"*50)

    correct_mean = results['15']['mean']
    for pos in positions:
        pos_label = str(pos) if pos is not None else "None"
        mean = results[pos_label]['mean']
        std = results[pos_label]['std']
        diff = mean - correct_mean
        marker = " <-- CORRECT" if pos == 15 else ""
        print(f"{pos_label:<12} {mean:<15.3f} {std:<10.3f} {diff:+.3f}{marker}")

    # Statistical test: Is correct position significantly better?
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    from scipy import stats

    correct_lls = results['15']['values']

    print("\nPairwise t-tests vs correct position (15):")
    for pos in positions:
        if pos == 15:
            continue
        pos_label = str(pos) if pos is not None else "None"
        other_lls = results[pos_label]['values']
        t_stat, p_val = stats.ttest_ind(correct_lls, other_lls)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  15 vs {pos_label:<5}: t={t_stat:+.2f}, p={p_val:.4f} {sig}")

    # Key question: Does position matter?
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Compare correct (15) vs None
    none_mean = results['None']['mean']
    print(f"\n1. UP element effect (15 vs None): {correct_mean - none_mean:+.3f}")

    # Compare correct (15) vs far position (70)
    far_mean = results['70']['mean']
    print(f"2. Position effect (15 vs 70):      {correct_mean - far_mean:+.3f}")

    # Is there a peak at correct position?
    means = [results[str(p) if p is not None else 'None']['mean'] for p in positions[:-1]]
    peak_pos = positions[np.argmax(means)]
    print(f"3. Peak LL position:                {peak_pos}")
    print(f"   (Correct position is 15)")

    # Save results
    output_path = Path('data/results/positional_sweep_results.json')
    save_results = {k: {'mean': v['mean'], 'std': v['std']} for k, v in results.items()}
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
