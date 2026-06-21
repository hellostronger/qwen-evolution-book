# 第4章 Qwen-2：架构突破（2024年6月）

> 本章要点：GQA (Grouped Query Attention) 核心创新与实现细节分析。

## 4.1 发布背景

### 时间节点

- **发布时间**：2024年6月
- **迭代周期**：Qwen-1.5发布后4个月
- **定位**：架构级突破版本

### 技术报告

- **arXiv 论文**：[2407.10671] Qwen2 Technical Report
- **论文链接**：https://arxiv.org/abs/2407.10671
- **作者**：Qwen Team, Alibaba Cloud

---

## 4.2 GQA 核心创新 ⭐

### 问题背景：MHA的局限性

**传统 Multi-Head Attention (MHA)**：

```
┌────────────────────────────────────────────────────┐
│              MHA 结构                               │
├────────────────────────────────────────────────────┤
│                                                    │
│  Input: [batch, seq_len, hidden_dim]              │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 32 Query Heads                               │  │
│  │ 每个head: [batch, seq_len, head_dim=128]    │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 32 Key Heads  ← 需要存储KV cache             │  │
│  │ 每个head: [batch, seq_len, head_dim=128]    │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 32 Value Heads ← 需要存储KV cache            │  │
│  │ 每个head: [batch, seq_len, head_dim=128]    │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  KV Cache Size:                                   │
│  batch × seq_len × 32 × 128 × 2 (K+V)            │
│  = 较大的内存占用                                 │
│                                                    │
└────────────────────────────────────────────────────┘
```

**MHA推理时的内存瓶颈**：

```
问题：推理需要缓存所有历史token的KV

假设场景：
- batch_size = 1
- seq_len = 4096 (长对话)
- num_heads = 32
- head_dim = 128

KV Cache = 2 × 32 × 128 × 4096 × 2 bytes (FP16)
         ≈ 64 MB

如果 seq_len = 32K：
KV Cache ≈ 512 MB （仅一个head的开销）
```

### GQA解决方案

**Grouped Query Attention (GQA)**：

```
┌────────────────────────────────────────────────────┐
│              GQA 结构                               │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 32 Query Heads                               │  │
│  │ 分成8组，每组4个query heads                  │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 8 Key Heads  ← 每组共享1个KV head            │  │
│  │ (减少了4倍!)                                 │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │ 8 Value Heads                                │  │
│  │ (减少了4倍!)                                 │  │
│  └─────────────────────────────────────────────┘  │
│                                                    │
│  KV Cache Size:                                   │
│  batch × seq_len × 8 × 128 × 2 (K+V)             │
│  = 减少到原来的 1/4                               │
│                                                    │
└────────────────────────────────────────────────────┘
```

### GQA 实现代码

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """
    GQA实现 - Qwen2使用的注意力机制
    
    Args:
        hidden_dim: 总hidden维度
        num_query_heads: query head数量 (如32)
        num_kv_heads: KV head数量 (如8, 为query_heads的分组数)
        head_dim: 每个head的维度
    """
    
    def __init__(self, hidden_dim, num_query_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.group_size = num_query_heads // num_kv_heads
        
        # Query projection: 32 heads
        self.q_proj = nn.Linear(hidden_dim, num_query_heads * head_dim, bias=False)
        
        # Key/Value projection: 8 heads (共享)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_query_heads * head_dim, hidden_dim, bias=False)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. Reshape for attention
        # Query: [batch, seq_len, 32, head_dim]
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        
        # Key/Value: [batch, seq_len, 8, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 3. Expand KV to match Q groups
        # 每个KV head复制到group_size个query heads
        # Key: [batch, seq_len, 32, head_dim] (通过repeat)
        k = k.repeat_interleave(self.group_size, dim=2)
        v = v.repeat_interleave(self.group_size, dim=2)
        
        # 4. Compute attention
        # ... (标准attention计算)
        
        # 5. Output projection
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output, present_key_value
```

### 效果对比

| 维度 | MHA | GQA | 提升 |
|-----|-----|-----|------|
| **KV Cache 大小** | 100% | 25% | **减少75%** |
| **推理速度** | 基准 | +15% | **更快** |
| **显存占用** | 基准 | -60% | **大幅降低** |
| **性能损失** | - | <2% | **可接受** |

### 为什么性能损失很小？

**数学直觉**：

GQA 的关键在于：同一个 group 内的 query heads 关注相同的 KV head。

- 在许多情况下，不同位置的 query heads 关注的信息高度相似
- 将相似的 heads 分组共享 KV，损失很小
- 这是 Llama-2、Mistral 等模型也采用的策略

```
直观理解：

假设32个query heads中：
- Heads 0-3 主要关注"语法结构" → 共享KV head 0
- Heads 4-7 主要关注"语义内容" → 共享KV head 1
- Heads 8-11 主要关注"位置信息" → 共享KV head 2
- ...

这种分组减少了冗余计算，同时保留了足够的注意力多样性
```

---

## 4.3 其他架构改进

### 扩展上下文窗口

Qwen-2 支持更长的上下文长度：

| 模型 | Qwen-1.5 | Qwen-2 |
|-----|----------|--------|
| 7B | 4K | 32K |
| 72B | 32K | 128K |

**实现方式**：
- RoPE 外推 + 动态NTK缩放
- 长序列训练数据增强

### 训练数据规模

| 指标 | Qwen-1.5 | Qwen-2 |
|-----|----------|--------|
| 训练token数 | ~6T | ~7T |
| 多语言覆盖 | 20+ | 30+ |

---

## 4.4 Benchmark 跃升

### 完整数据对比（基于搜索结果）

**Qwen-1.5-72B vs Qwen-2-72B**：

| Benchmark | Qwen-1.5-72B | Qwen-2-72B | 提升 |
|-----------|-------------|------------|------|
| **MMLU** | 77.2 | **84.2** | +7.0 |
| **MATH** | 36.1 | **58.0** | +21.9 ⭐ |
| **HumanEval** | 52.8 | **64.1** | +11.3 |
| **GSM8K** | ~75 | ~89 | +14 |

**Qwen2-7B vs Qwen-1.5-7B**：

| Benchmark | Qwen-1.5-7B | Qwen-2-7B | 提升 |
|-----------|-------------|-----------|------|
| MMLU | 61.0 | 70.3 | +9.3 |
| MATH | 21.4 | 51.1 | +29.7 ⭐ |
| HumanEval | 36.6 | 45.4 | +8.8 |

### 关键跃升点

1. **MATH 提升 21-30分**：数学能力大幅增强
2. **MMLU 提升 7-9分**：通用知识显著提高
3. **HumanEval 提升 8-11分**：编程能力明显进步

---

## 小结

- **核心创新**：GQA减少KV Cache 75%，推理效率大幅提升
- **实现细节**：32 query heads → 8 KV heads分组共享
- **性能损失**：<2%，可接受
- **Benchmark跃升**：MATH提升最显著（+21.9）

---

## 参考文献

1. Qwen2 Technical Report: https://arxiv.org/abs/2407.10671
2. GQA Paper: https://arxiv.org/abs/2305.13245
3. Qwen2 Official Blog: https://qwenlm.github.io/blog/qwen2/

> 参见：[第6章 GQA技术详解](/chapters/06-technical-deep)