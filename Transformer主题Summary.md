# Transformer/Attention主题深度Summary

> **涵盖文章**：14篇Transformer相关文章
> **主要内容**：Attention机制、位置编码、长度外推、效率优化

---

## 1. 核心理论、公理与历史基础

### 1.1 历史发展
- **2014 - Attention**：Bahdanau等人在机器翻译中引入
- **2017 - Transformer**：Vaswani等人《Attention is All You Need》
- **2018 - BERT/GPT**：预训练范式确立
- **2021 - RoPE**：Su等人提出旋转位置编码
- **2023 - Flash Attention**：Dao等人，IO-aware优化
- **2024 - MLA**：DeepSeek的多头潜在注意力

### 1.2 核心数学

#### **Attention核心公式**
$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}}\right)\boldsymbol{V}$$

#### **为什么需要Scale？**

**熵不变性推导**：
设 $\boldsymbol{q}, \boldsymbol{k} \sim \mathcal{N}(0, \boldsymbol{I})$，则 $\boldsymbol{q}^T\boldsymbol{k} \sim \mathcal{N}(0, d)$

Softmax输出熵：
$$H(\boldsymbol{p}) = -\sum_i p_i \log p_i \approx \log n - \frac{d}{2}\text{Var}[\boldsymbol{q}^T\boldsymbol{k}]$$

**不Scale**：$d$ 增大 → 熵减小 → 注意力过度集中（"过拟合"）
**Scale by** $\sqrt{d}$：保持熵稳定

---

## 2. 位置编码完整推导

### 2.1 绝对位置编码（Sinusoidal）

**公式**：
$$\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d}$$
$$\text{PE}(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

**性质**：
- 相对位置可表达：$\text{PE}(pos+k)$ 可由 $\text{PE}(pos)$ 线性组合
- 无界：可外推到任意长度（理论上）

### 2.2 RoPE（旋转位置编码）

**核心思想**：将位置信息编码为复平面上的旋转

**2D情况**：
$$\begin{pmatrix} q_0' \\ q_1' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

**相对位置表达**：
$$\langle \boldsymbol{q}_m, \boldsymbol{k}_n \rangle = \boldsymbol{q}^T \boldsymbol{R}^T_{n-m} \boldsymbol{k} = f(n-m)$$

**优势**：
- ✅ 完全相对位置
- ✅ 乘法操作（vs 加法）
- ✅ 长度外推性好

---

## 3. 长度外推技术对比

| 方法 | 核心思想 | 外推能力 | **缺陷** | **优化** |
|------|---------|---------|---------|---------|
| **RoPE** | 旋转编码 | 2-4× | ❌ 远距离幅度衰减 | ✅ YaRN重缩放 |
| **ALiBi** | 线性Bias | 10× | ❌ 破坏预训练权重 | ✅ Bias+RoPE混合 |
| **ReRoPE** | 泄漏RoPE | ∞（理论） | ❌ 近距离精度损失 | ✅ 分段泄漏 |
| **NoPE** | 无位置编码 | ∞ | ❌ 位置信息完全依赖数据 | ✅ 预训练增强 |

---

## 4. Flash Attention原理

### 4.1 标准Attention的内存瓶颈

**问题**：需存储 $O(N^2)$ 的注意力矩阵
- **实例**：$N=8192$, $d=128$, FP16 → 256MB per head

### 4.2 Flash Attention解决方案

**核心技巧**：分块计算+在线Softmax

**步骤1：分块加载**
```python
for i in range(0, N, B):  # Query块
    for j in range(0, N, B):  # Key块
        S_ij = Q[i:i+B] @ K[j:j+B].T / sqrt(d)
        # 在线更新Softmax（不存储完整S）
```

**步骤2：在线Softmax**
$$m_{new} = \max(m_{old}, m_{block})$$
$$l_{new} = e^{m_{old}-m_{new}}l_{old} + e^{m_{block}-m_{new}}l_{block}$$

**效果**：
- 内存：$O(N^2) \to O(N)$
- 速度：2-4×加速（减少HBM访问）

---

## 5. 多头注意力变体

### 5.1 MQA（Multi-Query Attention）

**改进**：所有头共享Key和Value
$$\text{MQA}(\boldsymbol{Q}_1, \ldots, \boldsymbol{Q}_h, \boldsymbol{K}, \boldsymbol{V})$$

**优势**：KV缓存减少 $h$ 倍
**劣势**：表达能力下降（~1% 性能损失）

### 5.2 GQA（Grouped-Query Attention）

**折中方案**：$h$ 个头分成 $g$ 组
- $g=1$：MQA
- $g=h$：MHA
- **典型**：$g=h/8$（Llama2）

### 5.3 MLA（Multi-head Latent Attention）

**核心思想**：用低秩投影压缩KV
$$\boldsymbol{K} = \boldsymbol{W}_K^{down} \boldsymbol{W}_K^{up} \boldsymbol{x}$$
其中 $\boldsymbol{W}^{down} \in \mathbb{R}^{d \times d_{latent}}$, $d_{latent} \ll d$

**压缩比**：KV缓存减少 $d / d_{latent}$ 倍（~8×）
**质量**：性能几乎无损（< 0.5%）

---

## 6. 未来方向

**方向1：无限外推的理论边界**
- 当前最佳：StreamingLLM（滑动窗口+注意力sink）
- 目标：真正的无限长度，无精度损失

**方向2：$O(N)$复杂度的完全注意力**
- Linear Attention局限：无法建模高阶交互
- 潜在突破：结构化矩阵（Monarch、Toeplitz）

**方向3：多模态位置编码统一**
- 文本+图像+音频的联合位置表示
- 探索：3D RoPE、时空解耦编码

---

**撰写日期**：2025-11-18
**版本**：v1.0
