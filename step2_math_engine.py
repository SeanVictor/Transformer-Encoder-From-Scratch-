"""
Passo 2: O Motor Matemático
- Scaled Dot-Product Attention
- Conexões Residuais + LayerNorm
- Feed-Forward Network (FFN)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# 2.1  Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────

def softmax(x):
    """
    Softmax numericamente estável ao longo do último eixo.
    Subtrai o máximo antes de np.exp para evitar overflow.
    """
    # x pode ser (batch, seq, seq)
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class ScaledDotProductAttention:
    """
    Implementa: Attention(Q,K,V) = softmax( QK^T / sqrt(d_k) ) V
    """

    def __init__(self, d_model):
        self.d_model = d_model
        # Pesos de projeção: (d_model, d_model)
        self.W_Q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def forward(self, X):
        """
        X: (batch, seq_len, d_model)
        Retorna:
            output       : (batch, seq_len, d_model)
            attn_weights : (batch, seq_len, seq_len)  – para inspeção
        """
        # 1. Projeções lineares → Q, K, V
        Q = X @ self.W_Q   # (batch, seq, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        d_k = self.d_model  # dimensão das chaves

        # 2. Produto escalar Q · K^T
        # K.transpose(0,2,1): (batch, d_model, seq)
        scores = Q @ K.transpose(0, 2, 1)  # (batch, seq, seq)

        # 3. Scaling
        scores = scores / np.sqrt(d_k)

        # 4. Softmax
        attn_weights = softmax(scores)  # (batch, seq, seq)

        # 5. Soma ponderada de V
        output = attn_weights @ V       # (batch, seq, d_model)

        return output, attn_weights


# ─────────────────────────────────────────────────────────────
# 2.2  LayerNorm  +  Conexão Residual
# ─────────────────────────────────────────────────────────────

def layer_norm(X, epsilon=1e-6):
    """
    Normalização de camada sobre o último eixo (features).
    LayerNorm opera por token (linha), diferente do BatchNorm.

    X: (batch, seq_len, d_model)
    """
    mean = np.mean(X, axis=-1, keepdims=True)          # (batch, seq, 1)
    var  = np.var(X,  axis=-1, keepdims=True)          # (batch, seq, 1)
    X_norm = (X - mean) / np.sqrt(var + epsilon)       # (batch, seq, d_model)
    return X_norm


def add_and_norm(X, sublayer_output, epsilon=1e-6):
    """
    Conexão residual seguida de LayerNorm:
        Output = LayerNorm(X + Sublayer(X))
    """
    X_res = X + sublayer_output        # Add  (residual connection)
    return layer_norm(X_res, epsilon)  # Norm


# ─────────────────────────────────────────────────────────────
# 2.3  Feed-Forward Network (FFN)
# ─────────────────────────────────────────────────────────────

class FeedForwardNetwork:
    """
    FFN(x) = max(0, x·W1 + b1)·W2 + b2

    Expande d_model → d_ff (tipicamente 4×d_model) e contrai de volta.
    """

    def __init__(self, d_model, d_ff=None):
        if d_ff is None:
            d_ff = d_model * 4   # Paper usa d_ff = 2048 (= 4 × 512)

        self.d_model = d_model
        self.d_ff    = d_ff

        # Inicialização He (adequada para ReLU)
        self.W1 = np.random.randn(d_model, d_ff)    * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)

        self.W2 = np.random.randn(d_ff,    d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        """
        X: (batch, seq_len, d_model)
        """
        # 1ª camada linear + ReLU
        hidden = np.maximum(0, X @ self.W1 + self.b1)   # (batch, seq, d_ff)

        # 2ª camada linear  (contração)
        output = hidden @ self.W2 + self.b2              # (batch, seq, d_model)

        return output


# ─────────────────────────────────────────────────────────────
# Smoke-test rápido das funções isoladas
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    D_MODEL = 64
    BATCH, SEQ = 1, 5

    X_test = np.random.randn(BATCH, SEQ, D_MODEL)

    # Atenção
    attn = ScaledDotProductAttention(D_MODEL)
    out_attn, weights = attn.forward(X_test)
    print(f"[Attention] input : {X_test.shape}  →  output : {out_attn.shape}")
    print(f"[Attention] pesos de atenção shape: {weights.shape}")

    # Add & Norm
    out_norm = add_and_norm(X_test, out_attn)
    print(f"[Add&Norm]  output: {out_norm.shape}")

    # FFN
    ffn = FeedForwardNetwork(D_MODEL)
    out_ffn = ffn.forward(out_norm)
    print(f"[FFN]       input : {out_norm.shape}  →  output : {out_ffn.shape}")
