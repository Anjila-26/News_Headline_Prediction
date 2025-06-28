# Transformer Encoder-Decoder for Text Summarization

A complete implementation of the Transformer encoder-decoder architecture from scratch using PyTorch for text summarization tasks. This project demonstrates the core components of the "Attention is All You Need" paper applied to headline generation from news article descriptions.

## 🎯 Overview

This project implements a complete Transformer encoder-decoder model for text summarization, specifically for generating news headlines from article descriptions. The implementation includes all core components:

- **Input Embeddings**: Token-to-vector conversion with scaling
- **Positional Encoding**: Sinusoidal position information
- **Multi-Head Attention**: Parallel attention mechanisms
- **Feed-Forward Networks**: Position-wise transformations
- **Encoder Layers**: Self-attention and feed-forward sublayers
- **Decoder Layers**: Masked self-attention, cross-attention, and feed-forward
- **Complete Training Pipeline**: Loss computation, optimization, and evaluation

## 🏗️ Architecture

### High-Level Architecture

```
Input Text (Description) → Encoder → Context Representation
                                          ↓
Target Text (Headline) → Decoder → Output Probabilities
```

### Detailed Component Flow

```
┌─────────────────────┐    ┌─────────────────────┐
│     ENCODER         │    │      DECODER        │
├─────────────────────┤    ├─────────────────────┤
│ Input Embeddings    │    │ Output Embeddings   │
│ Positional Encoding │    │ Positional Encoding │
│                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ Encoder Layer 1 │ │    │ │ Decoder Layer 1 │ │
│ │ - Self Attention│ │    │ │ - Masked Attn   │ │
│ │ - Feed Forward  │ │    │ │ - Cross Attn    │ │
│ └─────────────────┘ │    │ │ - Feed Forward  │ │
│         ...         │───▶│ └─────────────────┘ │
│ ┌─────────────────┐ │    │         ...         │
│ │ Encoder Layer N │ │    │ ┌─────────────────┐ │
│ └─────────────────┘ │    │ │ Decoder Layer N │ │
└─────────────────────┘    │ └─────────────────┘ │
                           │                     │
                           │ Linear + Softmax    │
                           └─────────────────────┘
```

## 📊 Dataset

The project uses the **News Category Dataset v3** which contains:
- **Source**: News article short descriptions
- **Target**: News headlines
- **Size**: 50,000 samples (configurable)
- **Preprocessing**: Text cleaning, tokenization, vocabulary building

### Data Processing Pipeline

1. **Text Cleaning**: Removes URLs, HTML tags, special characters
2. **Tokenization**: NLTK word tokenization
3. **Vocabulary Building**: Creates word-to-index mappings with special tokens
4. **Encoding**: Converts text to numerical sequences with padding/truncation

### Special Tokens

- `<PAD>`: Padding token (ID: 0)
- `<UNK>`: Unknown words (ID: 1)
- `<BOS>`: Beginning of sequence (ID: 2)
- `<EOS>`: End of sequence (ID: 3)

## 🔧 Implementation Details

### Model Configuration

```python
model = Transformer(
    vocab_size=len(word2idx),    # Dynamic based on dataset
    d_model=512,                 # Embedding dimension
    num_heads=8,                 # Number of attention heads
    num_layers=2,                # Number of encoder/decoder layers
    d_ff=2048,                   # Feed-forward hidden dimension
    max_seq_length=64,           # Maximum sequence length
    dropout=0.1                  # Dropout rate
)
```

### Training Configuration

- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: CrossEntropyLoss (ignoring padding tokens)
- **Batch Size**: 16
- **Epochs**: 5
- **Device**: CUDA if available, otherwise CPU

## 🧩 Components

### 1. Input Embeddings

Converts discrete token IDs into dense vector representations:

```python
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Key Features**:
- Learnable embedding table
- Scaled by √d_model for better convergence
- Maps tokens to continuous vector space

### 2. Positional Encoding

Adds position information using sinusoidal functions:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Benefits**:
- Unique encoding for each position
- Relative position relationships
- No additional parameters to learn

### 3. Multi-Head Attention

Implements the core attention mechanism:

```python
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**Process**:
1. **Linear Projections**: Q, K, V from input
2. **Head Splitting**: Divide into multiple attention heads
3. **Scaled Dot-Product**: Compute attention scores
4. **Softmax**: Normalize attention weights
5. **Value Aggregation**: Weighted sum of values
6. **Head Concatenation**: Combine all heads
7. **Output Projection**: Final linear transformation

### 4. Feed-Forward Network

Position-wise transformation applied to each token:

```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Architecture**:
- Two linear layers with ReLU activation
- Expansion to larger dimension (d_ff)
- Projection back to model dimension

### 5. Encoder Layer

Combines self-attention and feed-forward with residual connections:

```
x' = LayerNorm(x + MultiHeadAttention(x, x, x))
output = LayerNorm(x' + FeedForward(x'))
```

### 6. Decoder Layer

Three sublayers with masked attention and cross-attention:

```
x' = LayerNorm(x + MaskedSelfAttention(x, x, x))
x'' = LayerNorm(x' + CrossAttention(x', encoder_out, encoder_out))
output = LayerNorm(x'' + FeedForward(x''))
```

## 🎯 Training

### Training Loop

1. **Forward Pass**: Process input through encoder-decoder
2. **Loss Calculation**: CrossEntropy with padding mask
3. **Backward Pass**: Gradient computation
4. **Optimization**: Adam optimizer step
5. **Evaluation**: BLEU and ROUGE metrics

### Teacher Forcing

During training, the decoder receives the ground truth previous tokens rather than its own predictions, accelerating convergence.

### Masking

- **Padding Mask**: Ignore padding tokens in attention
- **Causal Mask**: Prevent decoder from seeing future tokens
- **Cross Mask**: Optional masking for encoder-decoder attention

## 📈 Evaluation

### Metrics

1. **Loss**: CrossEntropy loss on validation set
2. **BLEU Score**: N-gram overlap with reference
3. **ROUGE-L**: Longest common subsequence F1 score

### Inference

Greedy decoding generates summaries token by token:

```python
def greedy_decode(model, src, word2idx, idx2word, max_len=64):
    # Encode source
    encoder_output = model.encoder(src)
    
    # Initialize decoder input
    ys = torch.tensor([[start_token]])
    
    # Generate tokens sequentially
    for _ in range(max_len - 1):
        out = model.decoder(ys, encoder_output, mask)
        next_token = out[:, -1, :].argmax(dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
    
    return ys
```

## 🚀 Usage

### Setup

```bash
# Install dependencies
pip install torch torchvision nltk rouge-score pandas numpy matplotlib

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Running the Notebook

1. **Data Loading**: Load and preprocess the News Category Dataset
2. **Vocabulary Building**: Create word-to-index mappings
3. **Model Training**: Train the Transformer model
4. **Evaluation**: Assess performance with metrics
5. **Inference**: Generate summaries for new inputs

### Example Usage

```python
# Load data
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)

# Build vocabulary
word2idx, idx2word, word_freq = build_vocab(texts, min_freq=1)

# Create model
model = Transformer(vocab_size=len(word2idx), d_model=512, ...)

# Train model
train_model(model, train_loader, val_loader, num_epochs=5)

# Generate summary
summary = generate_summary(model, input_text, word2idx, idx2word)
```

## 📦 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
nltk>=3.6
rouge-score>=0.1.2
tqdm>=4.62.0
```

## 📊 Results

The model generates headlines from news descriptions with varying quality:

### Example Results

**Input**: "scientists discover new species of ancient fish fossil dating back millions years"
**Generated**: "new species fish discovered"
**Reference**: "ancient fish species discovered by scientists"

**Input**: "technology companies investing heavily artificial intelligence research development"
**Generated**: "companies invest artificial intelligence"
**Reference**: "tech giants boost ai investment"

### Performance Metrics

- **Training Loss**: Decreases over epochs
- **BLEU Score**: Measures n-gram overlap
- **ROUGE-L**: Evaluates longest common subsequence

## 🧮 Mathematical Foundations

### Attention Mechanism

The scaled dot-product attention is defined as:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- Q (queries), K (keys), V (values) are the input matrices
- d_k is the dimension of the key vectors
- Scaling prevents softmax saturation in high dimensions

### Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Positional Encoding

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### Layer Normalization

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta$$

Where μ and σ are the mean and standard deviation of the input.

## 🔍 Key Features

- **Complete Implementation**: All Transformer components from scratch
- **Educational Focus**: Detailed comments and explanations
- **Modular Design**: Reusable components for different tasks
- **Comprehensive Evaluation**: Multiple metrics and inference methods
- **Real Dataset**: News summarization on actual data
- **GPU Support**: CUDA acceleration when available

## 🎓 Learning Objectives

This implementation helps understand:

1. **Attention Mechanisms**: How models focus on relevant information
2. **Transformer Architecture**: Encoder-decoder structure and data flow
3. **Sequence-to-Sequence Learning**: Mapping variable-length inputs to outputs
4. **Text Preprocessing**: Tokenization, vocabulary, and encoding
5. **Training Dynamics**: Loss functions, optimization, and evaluation
6. **Neural Language Models**: How transformers generate text

## 🔬 Potential Improvements

- **Beam Search**: Better decoding than greedy search
- **Label Smoothing**: Regularization technique for training
- **Learning Rate Scheduling**: Adaptive learning rates
- **Larger Models**: More layers and attention heads
- **Advanced Optimizers**: Learning rate warmup and decay
- **Data Augmentation**: Increase training data diversity

## 📚 References

1. Vaswani, A., et al. (2017). "Attention is All You Need." *NIPS*.
2. The Illustrated Transformer - Jay Alammar
3. PyTorch Transformer Tutorial
4. News Category Dataset - Kaggle

## 📄 License

This project is for educational purposes. Dataset attribution to original creators.

---

*This implementation serves as a comprehensive guide to understanding and implementing Transformer models from scratch, providing both theoretical foundations and practical experience with modern NLP architectures.*
