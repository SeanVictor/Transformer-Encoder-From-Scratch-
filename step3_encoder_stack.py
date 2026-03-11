"""
Passo 3: Empilhando tudo — N=6 camadas do Encoder

Fluxo por camada:
    1. X_att   = SelfAttention(X)
    2. X_norm1 = LayerNorm(X + X_att)
    3. X_ffn   = FFN(X_norm1)
    4. X_out   = LayerNorm(X_norm1 + X_ffn)
    5. X       = X_out   ← vira input da próxima camada
"""

import numpy as np
from step2_math_engine import ScaledDotProductAttention, FeedForwardNetwork, add_and_norm


# ─────────────────────────────────────────────────────────────
# Uma camada do Encoder
# ─────────────────────────────────────────────────────────────

class EncoderLayer:
    """
    Bloco único do Encoder:
        Sub-camada 1 → Self-Attention + Add&Norm
        Sub-camada 2 → FFN            + Add&Norm
    """

    def __init__(self, d_model, d_ff=None):
        self.attention = ScaledDotProductAttention(d_model)
        self.ffn       = FeedForwardNetwork(d_model, d_ff)

    def forward(self, X):
        """
        X: (batch, seq_len, d_model)
        Retorna tensor de mesma shape com representações contextualizadas.
        """
        # ── Sub-camada 1: Self-Attention ──────────────────────
        X_att, _ = self.attention.forward(X)
        X_norm1  = add_and_norm(X, X_att)           # LayerNorm(X + Attention(X))

        # ── Sub-camada 2: Feed-Forward ────────────────────────
        X_ffn  = self.ffn.forward(X_norm1)
        X_out  = add_and_norm(X_norm1, X_ffn)       # LayerNorm(X_norm1 + FFN(X_norm1))

        return X_out


# ─────────────────────────────────────────────────────────────
# Stack completo de N camadas
# ─────────────────────────────────────────────────────────────

class TransformerEncoder:
    """
    Empilha N camadas idênticas de EncoderLayer.
    Cada camada tem seus próprios pesos independentes.
    """

    def __init__(self, d_model, n_layers=6, d_ff=None):
        self.d_model  = d_model
        self.n_layers = n_layers
        # Cada camada é uma instância separada (pesos independentes)
        self.layers = [EncoderLayer(d_model, d_ff) for _ in range(n_layers)]

    def forward(self, X):
        """
        X: (batch, seq_len, d_model)
        Retorna Z de mesma shape após N camadas de contextualização.
        """
        print(f"\n{'='*55}")
        print(f"  Transformer Encoder — Forward Pass")
        print(f"{'='*55}")
        print(f"  Input shape : {X.shape}  (Batch={X.shape[0]}, "
              f"Tokens={X.shape[1]}, d_model={X.shape[2]})")
        print(f"  Camadas     : {self.n_layers}")
        print(f"{'='*55}")

        current = X
        for i, layer in enumerate(self.layers, start=1):
            current = layer.forward(current)
            print(f"  Camada {i}/6 → shape: {current.shape}  "
                  f"| mean={current.mean():.4f}  std={current.std():.4f}")

        print(f"{'='*55}")
        print(f"  Output shape: {current.shape}  ✓ dimensões preservadas")
        print(f"{'='*55}\n")
        return current


# ─────────────────────────────────────────────────────────────
# Execução direta deste módulo → smoke-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    D_MODEL = 64
    BATCH, SEQ = 1, 5

    X_test = np.random.randn(BATCH, SEQ, D_MODEL)
    encoder = TransformerEncoder(d_model=D_MODEL, n_layers=6)
    Z = encoder.forward(X_test)

    assert Z.shape == X_test.shape, (
        f"ERRO: shape de saída {Z.shape} ≠ shape de entrada {X_test.shape}"
    )
    print("Sanity check OK: shape de entrada == shape de saída")
