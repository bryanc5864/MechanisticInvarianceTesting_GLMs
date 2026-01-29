"""Autoregressive model wrappers (Evo2, HyenaDNA)."""

import torch
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseGLM


class Evo2Wrapper(BaseGLM):
    """Wrapper for Evo2 autoregressive genomic language model.

    Evo2 is trained autoregressively on DNA sequences and can compute
    log-likelihoods by summing log P(token_i | tokens_<i).
    """

    def __init__(
        self,
        model_name: str = "evo2_1b_base",
        device: str = "cuda",
    ):
        """Initialize Evo2 wrapper.

        Args:
            model_name: Evo2 model variant (evo2_1b_base, evo2_7b_base, etc.)
            device: Device for inference
        """
        super().__init__(model_name=model_name, device=device)

    def load_model(self) -> None:
        """Load Evo2 model."""
        try:
            from evo2 import Evo2
        except ImportError:
            raise ImportError(
                "evo2 package not found. Install with: pip install evo2"
            )

        print(f"Loading Evo2 model: {self.model_name}")
        self.model = Evo2(self.model_name)
        self.model.model.to(self.device)
        self.model.model.eval()
        self._loaded = True
        print(f"Evo2 model loaded on {self.device}")

    def compute_log_likelihood(self, sequence: str) -> float:
        """Compute log-likelihood for a DNA sequence.

        Uses autoregressive factorization:
        log P(x) = sum_i log P(x_i | x_<i)

        Args:
            sequence: DNA sequence (A, C, G, T)

        Returns:
            Log-likelihood score
        """
        if not self._loaded:
            self.load_model()

        with torch.no_grad():
            # Evo2 expects uppercase DNA sequence
            sequence = sequence.upper()

            # Get logits from the model
            # Evo2 returns logits of shape (batch, seq_len, vocab_size)
            outputs = self.model(sequence)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            # Convert sequence to token IDs
            # Evo2 uses single nucleotide tokens: A=0, C=1, G=2, T=3 (typically)
            token_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            tokens = torch.tensor([token_map.get(c, 0) for c in sequence])
            tokens = tokens.to(self.device)

            # Compute log probabilities
            # logits[i] predicts token at position i+1
            log_probs = torch.log_softmax(logits[0, :-1, :4], dim=-1)  # Only DNA tokens

            # Sum log P(x_i | x_<i) for i > 0
            target_tokens = tokens[1:]  # Shift targets
            ll = log_probs.gather(1, target_tokens.unsqueeze(1)).sum().item()

            return ll

    def compute_batch_log_likelihoods(
        self,
        sequences: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """Compute log-likelihoods for multiple sequences.

        Args:
            sequences: List of DNA sequences
            batch_size: Batch size (Evo2 processes one at a time for variable lengths)

        Returns:
            List of log-likelihood scores
        """
        results = []
        for seq in tqdm(sequences, desc=f"Evo2 inference"):
            results.append(self.compute_log_likelihood(seq))
        return results


class HyenaDNAWrapper(BaseGLM):
    """Wrapper for HyenaDNA autoregressive model.

    HyenaDNA uses Hyena operators for efficient long-range modeling.
    """

    def __init__(
        self,
        model_name: str = "LongSafari/hyenadna-medium-160k-seqlen-hf",
        device: str = "cuda",
    ):
        """Initialize HyenaDNA wrapper.

        Args:
            model_name: HuggingFace model ID
            device: Device for inference
        """
        super().__init__(model_name=model_name, device=device)

    def load_model(self) -> None:
        """Load HyenaDNA model from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading HyenaDNA model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"HyenaDNA model loaded on {self.device}")

    def compute_log_likelihood(self, sequence: str) -> float:
        """Compute log-likelihood using autoregressive factorization.

        Args:
            sequence: DNA sequence

        Returns:
            Log-likelihood score
        """
        if not self._loaded:
            self.load_model()

        with torch.no_grad():
            sequence = sequence.upper()

            # Tokenize
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = inputs["input_ids"].to(self.device)

            # Forward pass
            outputs = self.model(input_ids)
            logits = outputs.logits

            # Compute log-likelihood
            # logits[0, i, :] predicts token at position i+1
            log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
            target_ids = input_ids[0, 1:]

            ll = log_probs.gather(1, target_ids.unsqueeze(1)).sum().item()

            return ll

    def compute_batch_log_likelihoods(
        self,
        sequences: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """Compute log-likelihoods for multiple sequences.

        Args:
            sequences: List of DNA sequences
            batch_size: Batch size for processing

        Returns:
            List of log-likelihood scores
        """
        if not self._loaded:
            self.load_model()

        results = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(sequences), batch_size), desc="HyenaDNA inference"):
            batch = sequences[i:i + batch_size]

            # For variable-length sequences, process one at a time
            # (padding would affect likelihood computation)
            for seq in batch:
                results.append(self.compute_log_likelihood(seq))

        return results
