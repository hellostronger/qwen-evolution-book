# 第6章 关键技术创新详解

> 本章要点：GQA、RoPE、SwiGLU、RMSNorm、Tokenizer 的技术原理与实现。

## 6.1 GQA (Grouped Query Attention)

### 核心原理

已在 [第4章](/chapters/04-qwen2) 详细讲解。

### 数学表达

```
传统MHA:
Attention(Q, K, V) = softmax(QK^T / √d) V

GQA:
- Q分成G组，每组Ng个query heads
- 每组共享1个K和1个V head
- KV heads数量 = Q heads数量 / Ng
```

### 性能影响量化

| 指标 | MHA (32 heads) | GQA (8 KV heads) | 比值 |
|-----|----------------|------------------|------|
| KV Cache大小 | 32 × seq_len × d | 8 × seq_len × d | 25% |
| Attention计算 | 32 × Q × K | 32 × Q × (K/4) | 75% |
| 并行度 | 32独立 | 8组 × 4共享 | - |

---

## 6.2 RoPE (Rotary Position Embedding)

### 原理概述

**旋转位置编码**通过将位置信息编码为旋转矩阵：

```
将位置m编码为旋转角度θ:
θ_m = m × θ_base

对向量[x,y]应用旋转:
[x', y'] = [x cos(θ) - y sin(θ), x sin(θ) + y cos(θ)]

扩展到高维：每两个维度组成一组旋转
```

### 数学公式

```
对于位置m的第i个维度对：
f(x, m) = [x_i cos(mθ_i) - x_{i+1} sin(mθ_i),
           x_i sin(mθ_i) + x_{i+1} cos(mθ_i)]

其中 θ_i = θ_base^{-2i/d}
```

### 优势对比

| 特性 | 绝对位置编码 | 相对位置编码 | RoPE |
|-----|-------------|-------------|------|
| 位置信息 | 绝对固定 | 相对偏移 | 相对旋转 |
| 长序列支持 | 差 | 中 | **好** |
| 外推能力 | 无 | 有 | **强** |
| 计算复杂度 | 低 | 中 | 中 |

---

## 6.3 SwiGLU 激活函数

### 公式

```
SwiGLU(x, W, V) = Swish(xW) ⊙ (xV)

其中:
- Swish(x) = x ⊙ sigmoid(x)
- ⊙ 表示逐元素乘积
- W, V 是两个独立的线性变换
```

### 与其他激活函数对比

| 激活函数 | 公式 | 特点 |
|---------|------|------|
| ReLU | max(0, x) | 简单，但有"死亡神经元"问题 |
| GELU | x ⊙ Φ(x) | 平滑，Transformer常用 |
| Swish | x ⊙ sigmoid(βx) | 可学习β，更灵活 |
| **SwiGLU** | Swish(xW) ⊙ (xV) | 双线性+门控，性能更好 |

### 性能提升

论文显示：SwiGLU 在相同参数下比 GELU 提升约 2-3% 的性能。

---

## 6.4 RMSNorm 归一化

### 公式对比

```
LayerNorm:
y = (x - μ) / σ ⊙ γ + β

RMSNorm:
y = x / RMS(x) ⊙ γ

其中 RMS(x) = √(Σ x_i^2 / d)
```

### 优势

| 指标 | LayerNorm | RMSNorm |
|-----|-----------|---------|
| 计算量 | 高（需均值+方差） | 低（仅RMS） |
| 参数量 | γ + β | 仅γ |
| 稳定性 | 好 | **更好** |
| 速度 | 中 | **快** |

---

## 6.5 Tokenizer 设计

### Qwen Tokenizer 特点

```
基础: tiktoken (OpenAI)
词表: ~151K tokens
编码: BPE (Byte Pair Encoding)

优化:
- 中文高效压缩
- 英文基础token保留
- 代码特殊token增强
- 数学符号支持
```

### Token效率对比

| 语言/类型 | 平均token/字符 | 说明 |
|----------|---------------|------|
| 中文 | 0.5-0.8 | 高效压缩 |
| 英文 | 0.25-0.35 | 标准效率 |
| 代码 | 0.3-0.5 | 增强支持 |
| 数学公式 | 0.4-0.6 | 符号token |

---

## 小结

- **GQA**: KV Cache减少75%，性能损失<2%
- **RoPE**: 支持长序列，外推能力强
- **SwiGLU**: 双线性门控激活，性能+2-3%
- **RMSNorm**: 计算更快，稳定性更好
- **Tokenizer**: 中英代码数学全面优化

---

## 参考文献

1. RoPE Paper: https://arxiv.org/abs/2104.09864
2. SwiGLU Paper: https://arxiv.org/abs/2002.05202
3. RMSNorm Paper: https://arxiv.org/abs/1910.07467