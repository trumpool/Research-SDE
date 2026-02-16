"""
Semantic Encoder Module

Encodes text using RoBERTa-wwm-ext and projects to latent space.
Reference: Section 3.1 of the paper.
"""

import torch
import torch.nn as nn
from typing import Optional


class SemanticEncoder(nn.Module):
    """
    Semantic encoder using RoBERTa with linear projection.

    Maps text sequences to dense semantic vectors:
        T_i -> RoBERTa -> h^[CLS] -> W_p h + b_p -> x_i ∈ R^{d_in}

    Args:
        d_latent: Dimension of the projected semantic space (default: 32)
        pretrained_model: HuggingFace model name (default: chinese-roberta-wwm-ext)
        freeze_bert: Whether to freeze BERT parameters (default: False)
        dropout: Dropout probability for projection layer (default: 0.1)
    """

    def __init__(
        self,
        d_latent: int = 32,
        pretrained_model: str = "hfl/chinese-roberta-wwm-ext",
        freeze_bert: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.pretrained_model = pretrained_model

        # Load pretrained RoBERTa
        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(pretrained_model)
            self.bert_dim = self.bert.config.hidden_size  # typically 768
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Semantic projection layer: R^768 -> R^{d_latent}
        # Equation (2): x_i = W_p h^[CLS] + b_p
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert_dim, d_latent),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection layer weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text to semantic vectors.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            x: Projected semantic vectors [batch_size, d_latent]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Extract [CLS] token embedding (sentence representation)
        # Equation (1): H_i = RoBERTa(Tokenizer(T_i))
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]

        # Project to latent space
        # Equation (2): x_i = W_p h^[CLS] + b_p
        x = self.projection(cls_embedding)  # [batch, d_latent]

        return x

    def encode_texts(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 128,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method to encode raw text strings.

        Args:
            texts: List of text strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (default: 128)
            device: Target device

        Returns:
            x: Semantic vectors [batch_size, d_latent]
        """
        if device is None:
            device = next(self.parameters()).device

        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        return self.forward(input_ids, attention_mask)


class LightweightEncoder(nn.Module):
    """
    Lightweight encoder without pretrained BERT.

    Useful for quick prototyping or when pretrained models are unavailable.
    Uses learnable embeddings + Transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        d_latent: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        max_length: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.projection = nn.Linear(d_model, d_latent)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Transformer encoding
        if attention_mask is not None:
            # Convert attention mask to transformer format (True = ignore)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling + projection
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        return self.projection(x)
