# 第4章 Qwen-2：架构突破（2024年6月）

> 本章要点：基于 arXiv:2407.10671 技术报告深度分析 GQA 创新。

---

## 4.1 技术报告分析

### 报告信息

- **标题**: Qwen2 Technical Report
- **arXiv ID**: 2407.10671
- **发布时间**: 2024年7月
- **链接**: https://arxiv.org/abs/2407.10671
- **作者**: Qwen Team, Alibaba Cloud

### 报告摘要

**关键声明**:
- Qwen2 是一个从 0.5B 到 72B 参数的大型语言模型家族
- 包含 base 模型和 instruct 模型
- 在 30+ 语言的大规模多语言数据集上预训练
- instruct 模型在 MMLU、HumanEval、GSM8K 上超越大多数开源模型
- 72B 模型在多个 benchmark 上达到 GPT-4 级别性能

---

## 4.2 核心改进详解

### 改进 1: GQA (Grouped Query Attention) ⭐核心创新

#### 问题背景

**论文原文分析**:

> "传统的 Multi-Head Attention (MHA) 在推理时需要为每个 attention head 存储独立的 Key 和 Value cache，导致内存占用随序列长度线性增长。"

**数学分析**:

```
MHA KV Cache 计算:

设:
- n_heads = 32 (Qwen-2-7B)
- head_dim = 128
- seq_len = L

KV Cache 大小 = 2 × n_heads × head_dim × L × 2 bytes (FP16)
             = 2 × 32 × 128 × L × 2
             = 8192 × L bytes

当 L = 4096:
KV Cache = 32 MB per token layer

当 L = 32K:
KV Cache = 256 MB per token layer  ← 内存瓶颈!
```

#### GQA 解决方案

**论文核心公式**:

```
GQA 分组策略:

设:
- n_query_heads = 32
- n_kv_heads = 8  (减少4倍)
- group_size = n_query_heads / n_kv_heads = 4

每个 KV head 服务 4 个 query heads

KV Cache 大小 = 2 × n_kv_heads × head_dim × L × 2 bytes
             = 2 × 8 × 128 × L × 2
             = 2048 × L bytes  ← 减少到原来的 25%
```

#### 实现细节

```python
# Qwen-2 GQA 实现核心代码

class Qwen2Attention(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 32
        self.num_kv_heads = config.num_key_value_heads  # 8
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Query projection: 32 heads
        self.q_proj = nn.Linear(self.hidden_size, 
                                 self.num_heads * self.head_dim, 
                                 bias=False)
        
        # Key/Value projection: 8 heads (shared)
        self.k_proj = nn.Linear(self.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(self.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                self.hidden_size,
                                bias=False)
    
    def forward(self, hidden_states, past_key_value=None):
        batch, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        # Expand KV to match query groups
        # Key: [batch, seq, 8, 128] → [batch, seq, 32, 128]
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, 32, seq, 128]
        k = k.transpose(1, 2)  # [batch, 32, seq, 128]
        v = v.transpose(1, 2)  # [batch, 32, seq, 128]
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)
```

#### 性能影响量化

| 指标 | MHA | GQA | 改进 |
|-----|-----|-----|------|
| KV Cache | 100% | 25% | **-75%** |
| 推理速度 | 基准 | +15% | **更快** |
| 显存占用 | 基准 | -60% | **大幅降低** |
| MMLU性能 | 基准 | -1.5% | **可接受损失** |

---

## 4.3 训练数据扩展

### 数据规模对比

| 版本 | 训练Token数 | 多语言覆盖 |
|-----|------------|-----------|
| Qwen-1.5 | ~6T | 20+ |
| **Qwen-2** | **~7T** | **30+** |

### 多语言支持

**论文声明**: "pretrained on a large-scale multilingual dataset covering over 30 languages"

```
主要支持语言:

┌────────────────────────────────────────────────────┐
│           Qwen-2 多语言支持                          │
├────────────────────────────────────────────────────┤
│                                                    │
│  核心语言:                                          │
│  ├── 中文 (简体/繁体)                              │
│  ├── 英文                                          │
│  ├── 日语                                          │
│  ├── 韩语                                          │
│                                                    │
│  扩展语言:                                          │
│  ├── 欧洲语言: 德/法/意/西/俄等                     │
│  ├── 亚洲语言: 越/泰/印/马来等                      │
│  ├── 其他: 阿/希等                                  │
│                                                    │
│  训练策略:                                          │
│  ├── 多语言数据混合                                │
│  ├── 语言平衡采样                                  │
│  ├── 质量优先                                      │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 4.4 完整Benchmark数据

### 论文原始数据

#### Qwen2-7B

| Benchmark | Score | 对比Qwen1.5-7B |
|-----------|-------|---------------|
| MMLU | 70.3 | +9.3 |
| MATH | 51.1 | +29.7 ⭐ |
| HumanEval | 45.4 | +8.8 |
| GSM8K | 89.0 | +47 |

#### Qwen2-72B

| Benchmark | Score | 对比Qwen1.5-72B |
|-----------|-------|----------------|
| MMLU | 84.2 | +7.0 |
| MATH | 58.0 | +21.9 ⭐ |
| HumanEval | 64.1 | +11.3 |
| GSM8K | 89.0 | +14 |

### 性能跃升分析

```
MATH 提升分析:

Qwen1.5 → Qwen2:
├── 7B: 21.4 → 51.1 (+29.7分, +138%)
├── 72B: 36.1 → 58.0 (+21.9分, +60%)

原因分析:
├── 训练数据质量提升
├── 数学专项数据增加
├── 模型架构优化 (GQA)
```

---

## 小结

- **技术报告**: arXiv:2407.10671, 系统性架构阐述
- **GQA创新**: KV Cache减少75%, 性能损失<2%
- **训练数据**: 7T tokens, 30+语言覆盖
- **性能跃升**: MATH提升最显著 (+21-30分)
- **论文验证**: 与GPT-4级别相当

---

## 参考文献

1. Qwen2 Technical Report: https://arxiv.org/abs/2407.10671
2. GQA Paper: https://arxiv.org/abs/2305.13245