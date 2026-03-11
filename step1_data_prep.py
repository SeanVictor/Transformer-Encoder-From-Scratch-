"""
Passo 1: Preparação dos Dados
- Vocabulário com pandas
- Conversão de frase para IDs
- Tabela de Embeddings com numpy
- Tensor de entrada X no formato (BatchSize, SequenceLength, d_model)
"""

import numpy as np
import pandas as pd

# Semente para reprodutibilidade
np.random.seed(42)

# Hiperparâmetros
D_MODEL = 64  # Paper usa 512; usamos 64 para CPU

# ─────────────────────────────────────────
# 1. Vocabulário como DataFrame do pandas
# ─────────────────────────────────────────
vocab_data = {
    "palavra": ["<PAD>", "<UNK>", "o", "banco", "bloqueou", "cartao",
                "meu", "foi", "para", "casa", "eu", "preciso", "de",
                "dinheiro", "urgente"],
    "id": list(range(15))
}
vocab_df = pd.DataFrame(vocab_data).set_index("palavra")

print("=== Vocabulário ===")
print(vocab_df)
print()

# Dicionário auxiliar para lookup rápido
word2id = vocab_df["id"].to_dict()
id2word = {v: k for k, v in word2id.items()}

VOCAB_SIZE = len(vocab_df)

# ─────────────────────────────────────────
# 2. Frase de entrada → lista de IDs
# ─────────────────────────────────────────
frase = "o banco bloqueou meu cartao"
tokens = frase.lower().split()
token_ids = [word2id.get(t, word2id["<UNK>"]) for t in tokens]

print(f"Frase : '{frase}'")
print(f"Tokens: {tokens}")
print(f"IDs   : {token_ids}")
print()

# ─────────────────────────────────────────
# 3. Tabela de Embeddings simulada
# ─────────────────────────────────────────
# Shape: (vocab_size, d_model)
embedding_table = np.random.randn(VOCAB_SIZE, D_MODEL)

print(f"Tabela de Embeddings shape: {embedding_table.shape}")
print()

# ─────────────────────────────────────────
# 4. Tensor de entrada X: (Batch, SeqLen, d_model)
# ─────────────────────────────────────────
BATCH_SIZE = 1
SEQ_LEN = len(token_ids)

# Busca os embeddings das palavras da frase
X = embedding_table[token_ids]          # (SeqLen, d_model)
X = X[np.newaxis, :, :]                 # (1, SeqLen, d_model) → adiciona dim de batch

print(f"Shape do tensor X (entrada do Encoder): {X.shape}")
print(f"  Batch={X.shape[0]}, SeqLen={X.shape[1]}, d_model={X.shape[2]}")
