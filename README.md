# MIT Benchmark: Mechanistic Invariance Test for Genomic Language Models

This benchmark evaluates whether genomic language models (GLMs) understand regulatory compensation in E. coli σ70 promoters.

## Overview

The **Mechanistic Invariance Test (MIT)** probes whether GLMs have learned the mechanistic principle that promoter function can be maintained through compensatory regulatory elements, even when primary motifs are damaged.

### Key Insight

In E. coli σ70 promoters:
- The **-10 box** (consensus: TATAAT) is critical for σ70 recognition
- A **broken -10 box** (e.g., TGTAAT) dramatically reduces promoter strength
- However, **compensatory elements** (UP element, extended -10) can restore function

A model that truly understands promoter biology should recognize that:
> Broken + Compensated ≈ Intact

A model that only memorizes surface-level motifs will not understand this equivalence.

## Sequence Classes

| Class | Name | Description | N |
|-------|------|-------------|---|
| A | Natural Intact | Real promoters with strong -10 | 100 |
| B | Natural Broken | Real promoters with weak -10, no compensation | 100 |
| C | Synthetic Intact | Consensus -35/-10 elements | 100 |
| D | Synthetic Broken | Consensus -35, broken -10 | 100 |
| E | Synthetic Compensated | Broken -10 + UP element + extended -10 | 100 |
| F | Over-Compensated | Broken -10 + all compensatory elements | 50 |
| G | Natural Compensated | Real promoters with compensation | 50 |
| H | Scrambled Control | Same composition as E, scrambled motifs | 50 |

## Primary Metric: CSS

**Compensation Sensitivity Score (CSS)** measures how often a model scores compensated sequences higher than broken sequences:

```
CSS = P(LL(compensated) > LL(broken))
```

- **CSS = 0.5**: Model cannot distinguish (random baseline)
- **CSS > 0.5**: Model recognizes compensation
- **CSS = 1.0**: Perfect compensation sensitivity

## Installation

```bash
# Clone repository
cd MITproject

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install evo2 for Evo2 model
pip install evo2
```

## Usage

### 1. Generate Sequences

```bash
python scripts/generate_sequences.py --output data/sequences/
```

This generates 650 sequences across 8 classes.

### 2. Run Inference

```bash
# Run baseline models (fast)
python scripts/run_inference.py \
    --sequences data/sequences/all_sequences.json \
    --models kmer,pwm,random \
    --output data/results/

# Run GLMs (requires GPU)
python scripts/run_inference.py \
    --sequences data/sequences/all_sequences.json \
    --models dnabert2,nt_500m,grover \
    --output data/results/ \
    --gpu 0
```

### 3. Compute Metrics

```bash
python scripts/compute_metrics.py \
    --results data/results/ \
    --output data/results/metrics.json
```

### 4. Analyze Results

```bash
python scripts/analyze_results.py \
    --metrics data/results/metrics.json \
    --output figures/
```

## Models Supported

| Model | Type | Size | GPU Memory |
|-------|------|------|------------|
| Evo2-1B | Autoregressive | 2GB | ~4GB |
| DNABERT-2 | Masked LM | 230MB | ~2GB |
| Nucleotide Transformer 500M | Masked LM | 1GB | ~3GB |
| HyenaDNA-medium | Autoregressive | 500MB | ~2GB |
| GROVER | Masked LM | ~500MB | ~2GB |
| Caduceus | Masked LM | ~300MB | ~2GB |
| k-mer baseline | Statistical | - | CPU |
| PWM baseline | Statistical | - | CPU |

## Project Structure

```
MITproject/
├── mit_benchmark/
│   ├── sequences/
│   │   ├── generator.py      # Sequence generation
│   │   ├── motifs.py         # Promoter motif definitions
│   │   └── natural.py        # Natural promoter data
│   ├── models/
│   │   ├── base.py           # Abstract model interface
│   │   ├── autoregressive.py # Evo2, HyenaDNA wrappers
│   │   ├── masked_lm.py      # DNABERT-2, NT, GROVER, Caduceus
│   │   └── baselines.py      # k-mer, PWM baselines
│   ├── evaluation/
│   │   ├── metrics.py        # CSS, MES, CIR, CM, SCR
│   │   └── analysis.py       # Statistical tests, plotting
│   └── utils/
│       └── config.py         # Configuration
├── scripts/
│   ├── generate_sequences.py
│   ├── run_inference.py
│   ├── compute_metrics.py
│   └── analyze_results.py
├── data/
│   ├── sequences/            # Generated sequences
│   └── results/              # Model predictions
├── figures/                  # Generated plots
├── requirements.txt
└── README.md
```

## Metrics Reference

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| CSS | Compensation Sensitivity Score | [0, 1] | >0.5 = recognizes compensation |
| MES | Motif Effect Size (Cohen's d) | (-∞, ∞) | Higher = better motif discrimination |
| CIR | Context Independence Ratio | [0, ∞) | ~1 = consistent across contexts |
| CM | Compensation Magnitude | (-∞, ∞) | 1 = full recovery |
| SCR | Scramble Control Ratio | [0, 1] | >0.5 = recognizes structure |

## Citation

If you use this benchmark, please cite:

```bibtex
@software{mit_benchmark,
  title={MIT Benchmark: Mechanistic Invariance Test for Genomic Language Models},
  year={2024},
}
```

## License

MIT License
