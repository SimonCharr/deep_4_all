"""
Modèle Baseline : Oracle de la Guilde (Non Optimal)

Ce modèle est VOLONTAIREMENT non optimal.
Les étudiants doivent identifier et corriger les problèmes !

Contient:
- GuildOracle : MLP pour prédiction de survie (stats → survie)
- DungeonOracle : LSTM pour prédiction de survie (séquence d'événements → survie)
"""

import torch
import torch.nn as nn


# ============================================================================
# TP2 : Modèle MLP pour stats d'aventuriers
# ============================================================================


class GuildOracle(nn.Module):
    """
    Modèle pour prédire la survie des aventuriers.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 6, dropout: float = 0.5):
        """
        Args:
            input_dim: Nombre de features (8 stats)
            hidden_dim: Dimension des couches cachées
            dropout: Taux de dropout pour régularisation
        """
        super().__init__()
        # Architecture simple avec dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités de survie."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les prédictions binaires."""
        proba = self.predict_proba(x)
        return (proba > 0.5).float()


# ============================================================================
# TP3 : Modèle LSTM pour séquences de donjon
# ============================================================================


class DungeonOracle(nn.Module):
    """
    Modèle Transformer pour prédire la survie à partir d'une séquence d'événements.

    Architecture : Embedding + Positional Encoding + Transformer Encoder + Classifier

    Avantages par rapport au LSTM baseline :
    1. Self-attention capture les dépendances d'ordre (Potion avant Dragon)
    2. Positional encoding capture les dépendances long-terme (Amulette au début)
    3. Très compact : peu de paramètres pour de bonnes performances
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 32,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            mode: str = "transformer",
            bidirectional: bool = False,
            padding_idx: int = 0,
            max_length: int = 140,
            nhead: int = 4,
            ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mode = mode.lower().strip()
        self.padding_idx = padding_idx
        self.embed_dim = embed_dim
        self.max_length = max_length

        # Embedding
        self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=padding_idx
                )

        if self.mode == "transformer":
            # Positional encoding (learnable)
            self.pos_embedding = nn.Embedding(max_length, embed_dim)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation='gelu',
                    )
            self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                    )

            # Classifier
            self.classifier = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                    )

        elif self.mode == "lstm":
            self.rnn = nn.LSTM(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                    )
            classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.classifier = nn.Sequential(
                    nn.LayerNorm(classifier_input_dim),
                    nn.Linear(classifier_input_dim, 1)
                    )

        elif self.mode == "rnn":
            self.rnn = nn.RNN(
                    input_size=embed_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                    )
            classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.classifier = nn.Sequential(
                    nn.Linear(classifier_input_dim, 1)
                    )

        else:  # linear
            self.solo_embeddings = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(max_length * embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                    )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = x.size()

        embedded = self.embedding(x)

        if self.mode == "transformer":
            # Positional encoding
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            positions = positions.clamp(max=self.max_length - 1)
            embedded = embedded + self.pos_embedding(positions)

            # Padding mask: True = ignore
            padding_mask = (x == self.padding_idx)

            # Transformer
            output = self.transformer(embedded, src_key_padding_mask=padding_mask)

            # Mean pooling over non-padded tokens
            mask = (~padding_mask).unsqueeze(-1).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            logits = self.classifier(pooled)
            return logits

        elif self.mode in ("lstm", "rnn"):
            if self.mode == "lstm":
                output, (hidden, cell) = self.rnn(embedded)
            else:
                output, hidden = self.rnn(embedded)

            if self.bidirectional:
                hidden_forward = hidden[-2]
                hidden_backward = hidden[-1]
                final_hidden = torch.cat([hidden_forward, hidden_backward], dim=1)
            else:
                final_hidden = hidden[-1]

            logits = self.classifier(final_hidden)
            return logits

        else:
            return self.solo_embeddings(embedded)

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, lengths)
            return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        proba = self.predict_proba(x, lengths)
        return (proba > 0.5).float()

    def get_embeddings(self) -> torch.Tensor:
        return self.embedding.weight.detach().clone()


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module):
    """Affiche un résumé du modèle."""
    print("=" * 50)
    print("Résumé du modèle")
    print("=" * 50)
    print(model)
    print("-" * 50)
    print(f"Nombre de paramètres : {count_parameters(model):,}")
    print("=" * 50)
