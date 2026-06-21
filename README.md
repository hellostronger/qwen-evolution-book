# Qwen 模型演进史

> 从 Qwen-1 到 Qwen 3.7，阿里通义千问大模型的技术演进完整记录。

## 完整版本时间线

```
2023.08  Qwen-1     7B/14B/72B    初代开源模型
2024.02  Qwen-1.5   多尺寸        Tokenizer优化、训练效率提升
2024.06  Qwen-2     0.5B-72B      GQA架构突破
2024.09  Qwen-2.5   多尺寸+专项    Math/Coder专项模型
2026.XX  Qwen-3     1.7B-30B-A3B  MoE架构引入
2026.XX  Qwen-3.5   2B-35B-A3B    成熟优化版本
2026.06  Qwen-3.7   Plus          多模态智能体
```

## 目录

- [第1章：诞生背景](chapters/01-origins.md)
- [第2章：Qwen-1 基石](chapters/02-qwen1.md)
- [第3章：Qwen-1.5 快速迭代](chapters/03-qwen1-5.md)
- [第4章：Qwen-2 GQA突破](chapters/04-qwen2.md)
- [第5章：Qwen-2.5 成熟之作](chapters/05-qwen2-5.md)
- [第6章：Qwen 3 系列 MoE时代](chapters/06-qwen3.md)
- [第7章：技术详解](chapters/07-technical.md)
- [第8章：竞品对比](chapters/08-comparison.md)

## 关键技术演进

| 版本 | 架构创新 | 关键改进 |
|-----|---------|---------|
| Qwen-1 | Transformer + RoPE | 初代架构奠基 |
| Qwen-1.5 | Tokenizer优化 | 效率提升 |
| Qwen-2 | **GQA** | KV Cache减少75% |
| Qwen-2.5 | 专项模型 | Math/Coder优化 |
| Qwen-3 | **MoE** | 30B总/3B活跃 |
| Qwen-3.7 | 多模态统一 | 视觉-语言智能体 |

## 数据来源

- Qwen 官方博客: https://qwen.ai
- Qwen-Scope 技术报告: arxiv:2605.11887
- GitHub: https://github.com/QwenLM

## License

MIT