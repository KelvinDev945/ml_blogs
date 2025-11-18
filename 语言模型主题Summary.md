# 语言模型主题深度Summary

> **涵盖文章**：8篇语言模型相关文章
> **主要内容**：BERT优化、Decoder-only分析、句向量、高效训练

---

## 1. BERT初始化分析

### 1.1 为什么初始标准差是0.02？

**Xavier初始化**：
$$\sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

**BERT实践**（$d=768$）：
$$\sigma = \sqrt{\frac{2}{768 + 768}} \approx 0.036$$

**实际选择0.02的原因**：
1. **Pre-LayerNorm**：残差路径需要小初始化
2. **深度网络**（24层）：避免梯度爆炸
3. **经验调优**：0.02在各种任务上稳定

**理论支持**：
$$\text{Var}[\boldsymbol{y}^{(L)}] \approx (\sigma^2 d)^L$$
要求 $\sigma^2 d \approx 1$ → $\sigma \approx 0.036$，但考虑残差后需减小。

---

## 2. Decoder-only vs Encoder-Decoder

### 2.1 为什么LLM都是Decoder-only？

**对比表**：

| 架构 | 优势 | **缺陷** |
|------|------|---------|
| **Encoder-Decoder** | 双向编码器 | ❌ 推理慢（编码+解码）<br>❌ 无法In-context Learning |
| **Decoder-only** | 统一建模，ICL能力强 | ❌ 编码效率略低 |

**关键洞察**：
- **Prefix LM**：Decoder也能双向（通过attention mask）
- **Scaling Law**：大模型下，Decoder-only更高效

**实验证据**（GPT vs T5）：
- 相同参数量，Decoder-only在few-shot上强10%+
- 推理速度快2×（无需编码步骤）

---

## 3. BERT-whitening

### 3.1 核心原理

**观察**：BERT句向量各向异性（anisotropic）
- 高频词向量占据主导方向
- 余弦相似度偏高（所有句子都"相似"）

**Whitening操作**：
$$\boldsymbol{z} = (\boldsymbol{h} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1/2}$$

**效果**：
- 去除主成分偏置
- 各维度方差均匀化
- 相似度更有区分度

### 3.2 超参数版本

**引入可学习参数** $\alpha, \beta$：
$$\boldsymbol{z} = \alpha \cdot (\boldsymbol{h} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-\beta/2}$$

**优化**：在下游任务上finetune $\alpha, \beta$

---

## 4. 混合精度训练

### 4.1 FP16优势
- 内存减半
- 速度提升2-3×（Tensor Core）

### 4.2 挑战

**问题1：数值溢出**
- FP16范围：$6.1 \times 10^{-5}$ ~ $6.55 \times 10^4$
- 梯度可能 < $10^{-5}$（下溢）

**解决**：Loss Scaling
$$\mathcal{L}' = S \cdot \mathcal{L}, \quad \boldsymbol{g}' = S \cdot \boldsymbol{g}$$
更新时除以 $S$。

**问题2：精度损失**
- 权重更新 $\Delta\boldsymbol{\theta} = \eta \boldsymbol{g}$ 可能过小

**解决**：FP32 Master Weights
- FP16计算，FP32累积

---

## 5. 未来方向

**方向1：长上下文建模**
- 当前：128K tokens（GPT-4 Turbo）
- 目标：1M+ tokens
- 挑战：注意力 $O(N^2)$ 瓶颈

**方向2：高效推理**
- 模型剪枝
- INT4/INT8量化
- Speculative Decoding

**方向3：多语言统一**
- 当前：英语为中心
- 目标：低资源语言性能提升
- 方法：Cross-lingual Transfer

---

**撰写日期**：2025-11-18
**版本**：v1.0
