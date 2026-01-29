# MIT Benchmark Results

## Overview

This benchmark evaluates whether genomic language models understand regulatory compensation in E. coli σ70 promoters. The primary metric is the **Compensation Sensitivity Score (CSS)**, which measures how often a model scores compensated sequences higher than broken sequences.

- **CSS > 0.5**: Model recognizes compensation
- **CSS = 0.5**: Model cannot distinguish (random baseline)
- **CSS < 0.5**: Model penalizes compensated sequences

## Model Comparison

| Model | CSS | 95% CI | p-value | Significant |
|-------|-----|--------|---------|-------------|
| HyenaDNA | **0.630** | [0.530, 0.720] | **0.0043** | **Yes** |
| NT-500M | 0.540 | [0.450, 0.630] | 0.2132 | No |
| Random | 0.500 | [0.400, 0.610] | 0.5000 | No (baseline) |
| GROVER | 0.460 | [0.370, 0.550] | 0.7868 | No |
| k-mer | 0.430 | [0.330, 0.520] | 0.9187 | No |
| PWM | 0.000 | [0.000, 0.000] | 1.0000 | No |

## Full Metrics

| Model | CSS | MES (Natural) | MES (Synthetic) | CIR | CM | SCR |
|-------|-----|---------------|-----------------|-----|-----|-----|
| HyenaDNA | 0.630 | -0.01 | -0.34 | 32.75 | -1.34 | 0.480 |
| NT-500M | 0.540 | -0.00 | -0.10 | 106.70 | -1.55 | 0.400 |
| Random | 0.500 | -0.14 | -0.04 | 0.31 | 3.88 | 0.460 |
| GROVER | 0.460 | -0.02 | -0.03 | 1.50 | 3.85 | 0.480 |
| k-mer | 0.430 | -0.17 | 0.11 | -0.64 | -2.37 | 0.500 |
| PWM | 0.000 | 0.70 | 10.00 | 14.22 | 0.00 | 0.000 |

### Metric Definitions

- **CSS (Compensation Sensitivity Score)**: Fraction of cases where LL(compensated) > LL(broken)
- **MES (Motif Effect Size)**: Cohen's d for intact vs broken sequences
- **CIR (Context Independence Ratio)**: MES_synthetic / MES_natural
- **CM (Compensation Magnitude)**: Fraction of likelihood recovery from compensation
- **SCR (Scramble Control Ratio)**: Fraction where structured > scrambled compensation

## Statistical Tests

| Test | p-value | FDR-corrected | Significant |
|------|---------|---------------|-------------|
| HyenaDNA CSS vs 0.5 | 0.0043 | 0.0303 | Yes |
| NT-500M CSS vs 0.5 | 0.2132 | 0.7463 | No |
| Random CSS vs 0.5 | 0.5000 | 0.8750 | No |
| GROVER CSS vs 0.5 | 0.7868 | 1.0000 | No |
| k-mer CSS vs 0.5 | 0.9187 | 1.0000 | No |
| PWM CSS vs 0.5 | 1.0000 | 1.0000 | No |
| All models vs 0.5 | 0.4520 | 0.8750 | No |

## Key Findings

1. **HyenaDNA is the only model showing statistically significant compensation sensitivity** (CSS=0.630, p=0.0043, FDR-corrected p=0.0303)

2. **Nucleotide Transformer (NT-500M) shows a positive trend** (CSS=0.540) but does not reach statistical significance at α=0.05

3. **GROVER and k-mer baselines perform at or below random chance**, suggesting they do not recognize regulatory compensation

4. **PWM baseline achieves CSS=0.000** because it evaluates only the -35 and -10 boxes, giving identical scores to broken and compensated sequences (both have the same broken -10)

5. **The Scramble Control Ratio (SCR) is near 0.5 for all models**, indicating that models respond similarly to structured and scrambled compensatory elements

## Sequence Classes

| Class | Name | N | Description |
|-------|------|---|-------------|
| A | Natural Intact | 100 | Real promoters with strong -10 box |
| B | Natural Broken | 100 | Real promoters with mutated -10, no compensation |
| C | Synthetic Intact | 100 | Consensus -35 (TTGACA) and -10 (TATAAT) |
| D | Synthetic Broken | 100 | Consensus -35, broken -10 (TGTAAT) |
| E | Synthetic Compensated | 100 | Broken -10 + UP element + extended -10 |
| F | Over-Compensated | 50 | Broken -10 + all compensatory elements |
| G | Natural Compensated | 50 | Real promoters with compensation |
| H | Scrambled Control | 50 | Same composition as E, scrambled motifs |

**Total: 650 sequences**

## Figures

### CSS Comparison
![CSS Comparison](figures/css_comparison.png)

### Metrics Heatmap
![Metrics Heatmap](figures/metrics_heatmap.png)

### CSS vs MES
![CSS vs MES](figures/css_vs_mes.png)

## Interpretation

HyenaDNA appears to have learned something about regulatory compensation in E. coli σ70 promoters. It scores sequences with compensatory elements (UP element + extended -10) higher than those with just a broken -10 box, even though both have the same damaged core promoter element.

This suggests that HyenaDNA may have captured some mechanistic understanding of how bacterial promoters function, rather than simply memorizing sequence patterns. However, the low SCR scores indicate that even HyenaDNA may be responding more to nucleotide composition than to specific motif structure.

## Reproducibility

To reproduce these results:

```bash
# Generate sequences
python scripts/generate_sequences.py --output data/sequences/

# Run inference (requires GPU)
python scripts/run_inference.py \
    --sequences data/sequences/all_sequences.json \
    --models hyenadna,nt_500m,grover,kmer,pwm,random \
    --output data/results/

# Compute metrics
python scripts/compute_metrics.py --results data/results/

# Generate analysis
python scripts/analyze_results.py --metrics data/results/metrics.json --output figures/
```
