Transformer Encoder "From Scratch"

**Disciplina:** Tópicos em Inteligência Artificial – 2026.1  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior  

---

Descrição

Implementação do **Forward Pass** de um bloco Encoder completo do Transformer,
baseado no artigo original *"Attention Is All You Need"* (Vaswani et al., 2017),
construído **sem** PyTorch, TensorFlow ou Keras — apenas `Python 3.x`, `numpy` e `pandas`.

O pipeline recebe uma frase em português, a converte em embeddings e a passa por
**N = 6 camadas idênticas** do Encoder, produzindo o vetor de representação
contextualizada **Z** com as mesmas dimensões de entrada.

---

 Estrutura do Repositório

```
transformer_encoder/
│
├── main.py               # Script principal — executa o pipeline completo
├── step1_data_prep.py    # Passo 1: preparação de dados e embeddings
├── step2_math_engine.py  # Passo 2: Attention, LayerNorm, FFN
├── step3_encoder_stack.py# Passo 3: EncoderLayer e TransformerEncoder (N=6)
├── requirements.txt      # Dependências do projeto
└── README.md             # Este arquivo
```

---

 Arquitetura Implementada

```
Frase (texto)
     ↓
[Tokenização + Lookup de IDs]
     ↓
[Tabela de Embeddings]  →  X: (Batch=1, SeqLen, d_model=64)
     ↓
┌─────────────────────────────────────────┐
│          Encoder Layer × 6              │
│                                         │
│  X_att   = ScaledDotProductAttention(X) │
│  X_norm1 = LayerNorm(X + X_att)         │
│  X_ffn   = FFN(X_norm1)                 │
│  X_out   = LayerNorm(X_norm1 + X_ffn)   │
└─────────────────────────────────────────┘
     ↓
Z: (Batch=1, SeqLen, d_model=64)  ← representações contextualizadas
```

 Componentes implementados

| Módulo | Componente | Descrição |
|--------|------------|-----------|
| `step2_math_engine.py` | `softmax()` | Softmax numericamente estável (sem biblioteca) |
| `step2_math_engine.py` | `ScaledDotProductAttention` | `Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V` |
| `step2_math_engine.py` | `layer_norm()` | Normalização por features (último eixo) |
| `step2_math_engine.py` | `add_and_norm()` | Conexão residual + LayerNorm |
| `step2_math_engine.py` | `FeedForwardNetwork` | `FFN(x) = max(0, xW₁+b₁)W₂+b₂` |
| `step3_encoder_stack.py` | `EncoderLayer` | Um bloco completo do Encoder |
| `step3_encoder_stack.py` | `TransformerEncoder` | Stack de N=6 camadas |

 Hiperparâmetros

| Parâmetro | Paper original | Este projeto |
|-----------|---------------|--------------|
| `d_model` | 512 | **64** (CPU) |
| `d_ff`    | 2048 | **256** (4 × d_model) |
| `N` (camadas) | 6 | **6** |

---

 Pré-requisitos

- Python 3.8 ou superior
- pip

---

## Como rodar

 1. Clone o repositório

```bash
git clone https://github.com/<seu-usuario>/transformer-encoder-from-scratch.git
cd transformer-encoder-from-scratch
```

2. Instale as dependências

```bash
pip install -r requirements.txt
```

 3. Execute o pipeline completo

```bash
python main.py


A saída esperada mostra:
- O vocabulário e a conversão da frase para IDs
- O shape do tensor X de entrada
- O progresso pelas 6 camadas do Encoder
- A validação de sanidade (shape entrada == shape saída)
- Os vetores Z contextualizados por token

 4. Execute os módulos individualmente 

`bash
python step1_data_prep.py      # Apenas preparação de dados
python step2_math_engine.py    # Smoke-test dos componentes matemáticos
python step3_encoder_stack.py  # Smoke-test do stack do Encoder




 Saída Esperada

  PASSO 1 — Preparação dos Dados
Vocabulário (15 palavras): ...
Frase  : 'o banco bloqueou meu cartao'
Tokens : ['o', 'banco', 'bloqueou', 'meu', 'cartao']
IDs    : [2, 3, 4, 6, 5]
Tensor X shape: (1, 5, 64)

  Transformer Encoder — Forward Pass
  Camada 1/6 → shape: (1, 5, 64)
  ...
  Camada 6/6 → shape: (1, 5, 64)

  ✓ Dimensões preservadas — Tensor Z gerado com sucesso!
```



Nota de Integridade Acadêmica

Este projeto foi desenvolvido de forma autoral. Ferramentas de IA Generativa
(Claude) foram consultadas para fins de brainstorming sobre sintaxe do NumPy, 
criação deste readme  e revisão das equações matemáticas do paper, conforme 
permitido pelo contrato pedagógico da disciplina. Todo o código foi escrito e
compreendido pelo aluno.

