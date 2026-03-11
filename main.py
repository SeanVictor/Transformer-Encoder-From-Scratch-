"""
main.py  —  Transformer Encoder "From Scratch"
Disciplina: Tópicos em Inteligência Artificial – 2026.1
Instituição: iCEV

Executa o pipeline completo:
    Passo 1 → Preparação dos dados
    Passo 2 → Motor matemático (importado pelos módulos)
    Passo 3 → Forward pass pelas 6 camadas do Encoder
"""

import numpy as np
import pandas as pd

from step2_math_engine  import ScaledDotProductAttention, FeedForwardNetwork, add_and_norm
from step3_encoder_stack import TransformerEncoder

# ─────────────────────────────────────────────────────────────
# Hiperparâmetros
# ─────────────────────────────────────────────────────────────
D_MODEL    = 64    # Paper: 512. Reduzido para CPU.
N_LAYERS   = 6    # N = 6 conforme o paper
D_FF       = D_MODEL * 4   # 256 (paper: 2048)
BATCH_SIZE = 1
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PASSO 1: Preparação dos Dados
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PASSO 1 — Preparação dos Dados")
print("="*55)

vocab_data = {
    "palavra": ["<PAD>", "<UNK>", "o", "banco", "bloqueou",
                "cartao", "meu", "foi", "para", "casa",
                "eu", "preciso", "de", "dinheiro", "urgente"],
    "id": list(range(15))
}
vocab_df = pd.DataFrame(vocab_data).set_index("palavra")
word2id  = vocab_df["id"].to_dict()
VOCAB_SIZE = len(vocab_df)

print(f"\nVocabulário ({VOCAB_SIZE} palavras):")
print(vocab_df.to_string())

# Frase de entrada
frase     = "o banco bloqueou meu cartao"
tokens    = frase.lower().split()
token_ids = [word2id.get(t, word2id["<UNK>"]) for t in tokens]
SEQ_LEN   = len(token_ids)

print(f"\nFrase  : '{frase}'")
print(f"Tokens : {tokens}")
print(f"IDs    : {token_ids}")

# Tabela de Embeddings  (vocab_size, d_model)
embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL)
print(f"\nEmbedding table shape: {embedding_table.shape}")

# Tensor X: (Batch, SeqLen, d_model)
X = embedding_table[token_ids][np.newaxis, :, :]
print(f"Tensor X shape       : {X.shape}  "
      f"(Batch={X.shape[0]}, Tokens={X.shape[1]}, d_model={X.shape[2]})")


# ─────────────────────────────────────────────────────────────
# PASSO 2 + 3: Motor Matemático → Encoder Stack
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PASSO 2 & 3 — Encoder Forward Pass")
print("="*55)

encoder = TransformerEncoder(d_model=D_MODEL, n_layers=N_LAYERS, d_ff=D_FF)
Z = encoder.forward(X)

# ─────────────────────────────────────────────────────────────
# Validação de Sanidade
# ─────────────────────────────────────────────────────────────
print("="*55)
print("  VALIDAÇÃO DE SANIDADE")
print("="*55)
assert Z.shape == X.shape, \
    f"FALHOU: {Z.shape} ≠ {X.shape}"
print(f"  Shape de entrada : {X.shape}")
print(f"  Shape de saída   : {Z.shape}")
print(f"  ✓ Dimensões preservadas — Tensor Z gerado com sucesso!\n")

# ─────────────────────────────────────────────────────────────
# Inspeciona o Vetor Z (representações contextualizadas)
# ─────────────────────────────────────────────────────────────
print("="*55)
print("  VETOR Z — Representações Contextualizadas")
print("="*55)
for i, token in enumerate(tokens):
    vec = Z[0, i, :]  # vetor do token i
    print(f"  '{token:10s}'  |  mean={vec.mean():.4f}  "
          f"std={vec.std():.4f}  norm={np.linalg.norm(vec):.4f}")
print()
