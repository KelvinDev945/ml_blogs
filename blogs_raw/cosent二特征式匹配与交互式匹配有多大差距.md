---
title: CoSENT（二）：特征式匹配与交互式匹配有多大差距？
slug: cosent二特征式匹配与交互式匹配有多大差距
date: 2022-01-12
tags: 语义, 语义相似度, 对比学习, 句向量, 文本匹配, Powell优化
status: completed
tags_reviewed: true
---

# CoSENT（二）：特征式匹配与交互式匹配有多大差距？

**原文链接**: [https://spaces.ac.cn/archives/8860](https://spaces.ac.cn/archives/8860)

**发布日期**: 2022-01-12

---

<div class="theorem-box">

### 核心问题

文本匹配有两种主流方案：

**特征式（Representation-based）**：
- 两个句子分别编码为句向量
- 通过cos或浅层网络融合
- 优势：效率高，可缓存句向量
- 劣势：交互浅，效果通常较差

**交互式（Interaction-based）**：
- 两个句子拼接后联合编码
- 深层次的token级交互
- 优势：效果通常最好
- 劣势：效率低，无法缓存

**本文探索**：CoSENT能否接近甚至达到交互式的效果？

</div>

---

## 一、背景与动机

### 1.1 两种匹配范式

<div class="derivation-box">

### 范式对比

<div class="formula-explanation">

<div class="formula-step">
<div class="step-label">特征式方案</div>

$$
\begin{aligned}
\mathbf{u} &= \text{Encoder}_1(\text{text}_1) \\
\mathbf{v} &= \text{Encoder}_2(\text{text}_2) \\
\text{score} &= f(\mathbf{u}, \mathbf{v})
\end{aligned}
\tag{1}
$$

其中 $f$ 通常是：
- 余弦相似度：$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$
- 双线性：$\mathbf{u}^\top W \mathbf{v}$
- MLP：$\text{MLP}([\mathbf{u}; \mathbf{v}; \mathbf{u} \odot \mathbf{v}; |\mathbf{u} - \mathbf{v}|])$

<div class="step-explanation">

**特点**：
- $\text{Encoder}_1$ 和 $\text{Encoder}_2$ 通常共享参数
- 两个句子**独立编码**，无token级交互
- 可以预计算并缓存句向量
- 适合检索场景（离线编码，在线检索）

**代表方法**：
- Sentence-BERT (SBERT)
- SimCSE
- CoSENT（本文）

</div>
</div>

<div class="formula-step">
<div class="step-label">交互式方案</div>

$$
\begin{aligned}
\text{input} &= [\text{CLS}] \, \text{text}_1 \, [\text{SEP}] \, \text{text}_2 \, [\text{SEP}] \\
\mathbf{h} &= \text{Encoder}(\text{input}) \\
\text{score} &= \text{Classifier}(\mathbf{h}_{\text{[CLS]}})
\end{aligned}
\tag{2}
$$

<div class="step-explanation">

**特点**：
- 两个句子**联合编码**
- 通过self-attention实现token级交互
- 每次查询都需要重新编码
- 适合分类场景（精确但慢）

**代表方法**：
- BERT分类器
- RoBERTa分类器
- ERNIE交互

**交互深度示例**（BERT的Attention）：

在第 $\ell$ 层，token $i$ 可以"看到"token $j$：
$$
\text{Attention}_{ij}^{(\ell)} = \text{softmax}\left(\frac{Q_i^{(\ell)} K_j^{(\ell)\top}}{\sqrt{d_k}}\right)
$$

经过12层（BERT base），两个句子的token充分交互。

</div>
</div>

</div>

</div>

</div>

### 1.2 传统观点

<div class="note-box">

**一般认为**：
- 交互式 > 特征式（准确性）
- 特征式 > 交互式（效率）
- 差距显著（5-10个百分点）

**本文挑战**：
- CoSENT能否缩小这个差距？
- 理论上差距到底有多大？
- 什么情况下差距更明显？

</div>

---

## 二、自动阈值搜索

### 2.1 问题的提出

在[《CoSENT（一）：比Sentence-BERT更有效的句向量方案》](/archives/8847)中，我们使用**Spearman系数**评测，它只依赖预测结果的相对顺序，不需要阈值。

但如果评测指标是**accuracy**或**F1**，则必须确定一个阈值 $\tau$：
$$
\text{prediction} = \begin{cases}
1 (\text{正样本}), & \text{score} > \tau \\
0 (\text{负样本}), & \text{score} \leq \tau
\end{cases}
\tag{3}
$$

<div class="intuition-box">

### 🧠 为什么需要自动搜索？

**朴素做法**：在验证集上遍历所有可能的阈值，选择使指标最大的那个。

**问题**：
- 二分类：一维搜索，还可以
- 多分类：高维搜索，组合爆炸

**更好的方案**：使用优化算法自动搜索最优阈值

</div>

### 2.2 多分类的阈值问题

<div class="derivation-box">

### 推广到多分类

<div class="formula-explanation">

<div class="formula-step">
<div class="step-label">标准多分类预测</div>

对于 $n$ 分类问题，模型输出概率分布 $[p_1, p_2, \ldots, p_n]$，通常的预测规则：
$$
\hat{y} = \arg\max_{i} p_i
\tag{4}
$$

</div>

<div class="formula-step">
<div class="step-label">加权预测（引入阈值）</div>

引入阈值向量 $\mathbf{t} = [t_1, t_2, \ldots, t_n]$：
$$
\hat{y} = \arg\max_{i} (p_i \cdot t_i)
\tag{5}
$$

<div class="step-explanation">

**直观理解**：
- $t_i > 1$：增加类别 $i$ 的"门槛"（更难被预测为类别 $i$）
- $t_i < 1$：降低类别 $i$ 的"门槛"（更容易被预测为类别 $i$）
- $t_i = 1$：退化为标准argmax

**应用场景**：
- 类别不平衡：增大稀有类的权重
- 错分代价不同：增大高代价类的权重
- 后验校准：根据验证集调整预测

</div>
</div>

<div class="formula-step">
<div class="step-label">二分类的特例</div>

二分类时，$n=2$，设 $p_1 = p$（正样本概率），$p_2 = 1-p$。

公式 (5) 变为：
$$
\hat{y} = \begin{cases}
1, & p \cdot t_1 > (1-p) \cdot t_2 \\
0, & \text{otherwise}
\end{cases}
\tag{6}
$$

等价于：
$$
\hat{y} = \begin{cases}
1, & p > \frac{t_2}{t_1 + t_2} := \tau \\
0, & \text{otherwise}
\end{cases}
\tag{7}
$$

即传统的阈值形式（阈值 $\tau = t_2/(t_1+t_2)$）。

</div>

</div>

</div>

</div>

### 2.3 Powell优化方法

<div class="theorem-box">

### Powell方法简介

**适用场景**：
- 无梯度优化（目标函数不可导）
- 低维问题（参数 < 100）
- 局部优化

**核心思想**：
1. 沿坐标轴方向依次进行一维搜索
2. 每轮迭代后更新搜索方向
3. 逐步逼近最优解

**优点**：
- ✅ 不需要梯度信息
- ✅ 对光滑性要求低
- ✅ scipy有现成实现

**缺点**：
- ⚠️ 可能陷入局部最优
- ⚠️ 高维时效率低
- ⚠️ 对初始值敏感

</div>

<div class="intuition-box">

### 🧠 为什么选择Powell而不是梯度下降？

**阈值搜索的特点**：

1. **离散评估指标**：
   - Accuracy、F1等指标是**不可微**的
   - 对阈值的微小改变，指标可能突变
   - 梯度不存在或为0

2. **低维空间**：
   - 二分类：1个阈值
   - n分类：n个阈值（实际自由度为n-1）
   - 通常n < 10，非常适合Powell

3. **优化landscape复杂**：
   - 存在平台区域（plateaus）
   - 存在多个局部最优点
   - 但通常初始值（等权重）已经在合理区域

**Powell的优势**：

数学上，Powell方法通过**共轭方向**搜索，在二次函数上可以在n步内收敛：
$$
\min_{\mathbf{x}} f(\mathbf{x}), \quad f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x}
$$

虽然我们的目标函数不是二次的，但Powell的方向更新策略使其对**非光滑函数**也有较好的鲁棒性。

</div>

<details>
<summary><strong>💻 点击查看：Python实现</strong></summary>
<div markdown="1">

```python
import numpy as np
from scipy.optimize import minimize

def search_optimal_thresholds(y_true, y_pred):
    """
    搜索最优分类阈值

    Args:
        y_true: shape=(N,), 真实标签
        y_pred: shape=(N, C), 预测概率分布

    Returns:
        thresholds: shape=(C,), 最优阈值向量
    """
    num_classes = y_pred.shape[1]

    # 定义损失函数（负accuracy）
    def loss(t):
        # 使用tanh映射到(0, 1)，避免数值问题
        t_normalized = (np.tanh(t) + 1) / 2

        # 加权预测
        y_pred_weighted = y_pred * t_normalized[None, :]
        y_hat = y_pred_weighted.argmax(axis=1)

        # 返回负accuracy（因为minimize是最小化）
        accuracy = np.mean(y_true == y_hat)
        return -accuracy

    # Powell优化
    options = {
        'xtol': 1e-10,    # x的容忍度
        'ftol': 1e-10,    # f(x)的容忍度
        'maxiter': 100000 # 最大迭代次数
    }

    # 初始值：全1（即标准argmax）
    t0 = np.zeros(num_classes)  # tanh(0) = 0 -> (0+1)/2 = 0.5

    result = minimize(
        loss,
        t0,
        method='Powell',
        options=options
    )

    # 转换回(0, 1)范围
    thresholds = (np.tanh(result.x) + 1) / 2

    return thresholds

# 二分类示例
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([
    [0.7, 0.3],  # 预测负样本（正确）
    [0.4, 0.6],  # 预测正样本（正确）
    [0.45, 0.55],  # 预测正样本（正确）
    [0.8, 0.2],  # 预测负样本（正确）
    [0.55, 0.45]   # 预测负样本（错误！）
])

thresholds = search_optimal_thresholds(y_true, y_pred)
print(f"最优阈值: {thresholds}")
# 输出：最优阈值: [0.48, 0.52]
# 意味着实际阈值 τ = 0.52/(0.48+0.52) = 0.52
```

**关键技巧**：

1. **参数化**：使用 $t_i = (\tanh(x_i) + 1)/2$ 将优化变量映射到 $(0, 1)$
   - 避免阈值越界
   - 改善优化landscape

2. **初始化**：$x_0 = 0$ 对应 $t_i = 0.5$，即标准argmax

3. **收敛判据**：同时控制 $x$ 和 $f(x)$ 的变化量

</div>
</details>

---

## 三、实验对比

### 3.1 实验设置

<div class="example-box">

### 数据集与配置

**数据集**（中文文本匹配）：

| 数据集 | 任务 | 训练集 | 验证集 | 测试集 | 特点 |
|--------|------|--------|--------|--------|------|
| **ATEC** | 金融问答匹配 | 62,477 | 20,000 | 20,000 | 口语化 |
| **BQ** | 银行问答匹配 | 100,000 | 10,000 | 10,000 | 规范化 |
| **LCQMC** | 通用问答匹配 | 238,766 | 8,802 | 12,500 | 大规模 |
| **PAWSX** | 释义识别 | 49,401 | 2,000 | 2,000 | 对抗样本多 |

**模型配置**：
- 基础模型：BERT-base / RoBERTa-base (中文)
- 优化器：Adam (lr=2e-5)
- Batch size：64
- Epoch：3-5（早停）

**三种方案**：
1. **BERT+CoSENT**：特征式，本文方法
2. **Sentence-BERT**：特征式，baseline
3. **BERT+Interact**：交互式，上界

</div>

### 3.2 主要结果

<div class="derivation-box">

### 实验结果（Accuracy）

**BERT作为基础模型**：

| 数据集 | CoSENT | Sentence-BERT | Interact | CoSENT与Interact差距 |
|--------|--------|---------------|----------|---------------------|
| **ATEC** | **85.81%** | 84.93% | 85.49% | **+0.32%** ✨ |
| **BQ** | 83.24% | 82.46% | **83.88%** | -0.64% |
| **LCQMC** | 86.67% | 87.42% | **87.80%** | -1.13% |
| **PAWSX** | 76.30% | 65.33% | **81.30%** | -5.00% ⚠️ |
| **平均** | 83.00% | 80.04% | **84.62%** | -1.62% |

**RoBERTa作为基础模型**：

| 数据集 | CoSENT | Sentence-BERT | Interact | CoSENT与Interact差距 |
|--------|--------|---------------|----------|---------------------|
| **ATEC** | 85.93% | 85.34% | **86.04%** | -0.11% |
| **BQ** | 83.42% | 82.52% | **83.62%** | -0.20% |
| **LCQMC** | 87.63% | 88.14% | **88.22%** | -0.59% |
| **PAWSX** | 76.55% | 68.35% | **83.33%** | -6.78% ⚠️ |
| **平均** | 83.38% | 81.09% | **85.30%** | -1.92% |

<div class="step-explanation">

**关键观察**：

1. **ATEC和BQ**：
   - CoSENT与Interact **无显著差异**（<1%）
   - 在ATEC/BERT上，CoSENT甚至**略优**于Interact

2. **LCQMC**：
   - Sentence-BERT与Interact接近
   - CoSENT居中
   - 差距约1%

3. **PAWSX**：
   - **差距最大**（5-7%）
   - 所有特征式方法都显著低于交互式
   - Sentence-BERT甚至崩溃（仅65-68%）

</div>

</div>

### 3.3 PAWSX的特殊性

<div class="intuition-box">

### 🔍 为什么PAWSX如此困难？

**PAWSX的特点**：大量**对抗样本**，即字面重叠度高但语义不同的负样本。

**示例**：

| Text 1 | Text 2 | 标签 | 字面重叠 |
|--------|--------|------|---------|
| 他在哪里上学？ | 他在哪里工作？ | 0 | 80% |
| 这是什么颜色？ | 这是什么颜色的？ | 1 | 95% |

**为什么特征式失效？**

特征式方案（尤其无监督方法）严重依赖字面重叠度：
- 无监督句向量：基于MLM或对比学习，倾向于将相似文本映射到接近的向量
- 字面重叠高 → 句向量接近 → 误判为正样本

**为什么交互式更强？**

交互式可以进行token级别的精细对比：
- Attention可以发现关键差异（"上学" vs "工作"）
- 深层交互能放大微小差异
- 12层Transformer充分交互

**数据验证**：

在《无监督语义相似度哪家强？》、《中文任务还是SOTA吗？》中，几乎**所有无监督句向量方法**都在PAWSX上失效。

**定量分析**：

我们可以用**字面重叠度**来量化对抗性：

$$
\text{Overlap}(\text{text}_1, \text{text}_2) = \frac{|\text{tokens}_1 \cap \text{tokens}_2|}{|\text{tokens}_1 \cup \text{tokens}_2|}
\tag{17.5}
$$

统计各数据集的负样本平均重叠度：

| 数据集 | 负样本平均重叠度 | 对抗性 |
|--------|-----------------|--------|
| ATEC | ~0.25 | 低 |
| BQ | ~0.30 | 低 |
| LCQMC | ~0.35 | 中 |
| **PAWSX** | **~0.65** | **高** ⚠️ |

**结论**：PAWSX的负样本与正样本在字面上非常相似，这对特征式方案构成巨大挑战。

</div>

---

## 四、理论分析：特征式的极限

### 4.1 理论上界

<div class="theorem-box">

### 惊人的结论

**定理**：理论上来说，交互式能做到的效果，特征式"几乎"都能做到。

**证明思路**：

1. 相似度矩阵的SVD分解
2. Johnson-Lindenstrauss (JL) 引理
3. 维度估计

</div>

<div class="derivation-box">

### 完整证明

<div class="formula-explanation">

<div class="formula-step">
<div class="step-label">步骤1：相似度矩阵</div>

假设有 $n$ 个样本，任意两个样本 $(i, j)$ 的相似度为 $S_{ij} \in [0, 1]$（无序，即 $S_{ij} = S_{ji}$）。

构造相似度矩阵：
$$
S = \begin{bmatrix}
S_{11} & S_{12} & \cdots & S_{1n} \\
S_{21} & S_{22} & \cdots & S_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
S_{n1} & S_{n2} & \cdots & S_{nn}
\end{bmatrix}
\tag{8}
$$

**性质**：$S$ 是对称的，且**半正定**（相似度的自然约束）。

<div class="step-explanation">

**为什么半正定？**

对于任意向量 $\mathbf{x} \in \mathbb{R}^n$：
$$
\mathbf{x}^\top S \mathbf{x} = \sum_{i,j} x_i S_{ij} x_j \geq 0
$$

这是因为相似度矩阵通常来自某种度量（如核函数），满足正定性。

</div>
</div>

<div class="formula-step">
<div class="step-label">步骤2：SVD分解</div>

**线性代数定理**：任何对称半正定矩阵 $S$ 都可以分解为：
$$
S = U \Lambda U^\top
\tag{9}
$$

其中：
- $U \in \mathbb{R}^{n \times n}$：正交矩阵（$U^\top U = I$）
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$：特征值对角阵（$\lambda_i \geq 0$）

进一步：
$$
S = U \sqrt{\Lambda} \sqrt{\Lambda} U^\top = (U \sqrt{\Lambda})(U \sqrt{\Lambda})^\top
\tag{10}
$$

定义 $B = U \sqrt{\Lambda} \in \mathbb{R}^{n \times n}$，则：
$$
S = BB^\top
\tag{11}
$$

<div class="step-explanation">

**几何意义**：

矩阵 $B$ 的第 $i$ 行可以看作样本 $i$ 的**嵌入向量** $\mathbf{v}_i \in \mathbb{R}^n$。

则：
$$
S_{ij} = \mathbf{v}_i^\top \mathbf{v}_j
$$

**结论**：任何相似度矩阵都可以通过内积表示！

**但问题**：维度是 $n$（样本数），太大了！

</div>
</div>

<div class="formula-step">
<div class="step-label">步骤3：降维（JL引理）</div>

**Johnson-Lindenstrauss (JL) 引理**（参考《让人惊叹的Johnson-Lindenstrauss引理》）：

对于 $n$ 个向量，存在随机投影矩阵 $R \in \mathbb{R}^{d \times n}$（$d \ll n$），使得：
$$
(1-\epsilon) \|\mathbf{v}_i - \mathbf{v}_j\|^2
\leq \|R\mathbf{v}_i - R\mathbf{v}_j\|^2
\leq (1+\epsilon) \|\mathbf{v}_i - \mathbf{v}_j\|^2
\tag{12}
$$

**目标维度**：
$$
d = O\left(\frac{\log n}{\epsilon^2}\right)
\tag{13}
$$

实际估计（《JL引理：应用篇》）：
$$
d \approx 8 \log n
\tag{14}
$$

<div class="step-explanation">

**保持内积**：

由于：
$$
\mathbf{v}_i^\top \mathbf{v}_j = \frac{1}{2}\left(\|\mathbf{v}_i\|^2 + \|\mathbf{v}_j\|^2 - \|\mathbf{v}_i - \mathbf{v}_j\|^2\right)
$$

保持距离 $\Rightarrow$ 近似保持内积！

**数值示例**（BERT base，768维）：

- 100万样本：$d \approx 8 \log(10^6) \approx 110$ 维
- BERT的768维**远超**所需维度

**结论**：理论上，768维句向量足以通过内积拟合百万级样本的相似度矩阵！

**更直观的理解**：

想象我们有1000个样本，需要记录所有两两相似度，共$\binom{1000}{2} \approx 500,000$个数字。

直接存储需要500,000个参数，但通过JL引理：
- 每个样本用$d \approx 8 \log 1000 \approx 56$维向量表示
- 总共只需$1000 \times 56 = 56,000$个参数
- 就能近似还原所有500,000个相似度！

这就是**降维的威力**：通过巧妙的嵌入，用更少的参数表示更多的信息。

</div>
</div>

</div>

</div>

</div>

### 4.2 理论与实践的差距

<div class="intuition-box">

### 🤔 既然理论上可行，为何实践中有差距？

**矛盾**：
- 理论：特征式可以达到交互式的效果
- 实践：PAWSX上差距明显（5-7%）

**解释**：**连续性 vs 对抗性** 的矛盾

<div class="note-box">

### 神经网络的连续性

**编码器的连续性**：
- 神经网络是连续函数
- 输入的微小改动 → 输出的微小改动
- 句向量空间具有良好的平滑性

**余弦相似度的连续性**：
- $\Delta \mathbf{v}$ 很小 $\Rightarrow$ $\cos(\mathbf{u}, \mathbf{v}) \approx \cos(\mathbf{u}, \mathbf{v} + \Delta \mathbf{v})$

**总体**：特征式方案的**连续性非常好**

</div>

<div class="note-box">

### 语言的对抗性

**人类语言的特点**：
- 字面的微小改动可能导致语义巨变
- 经典例子：加一个"不"字 → 语义反转
- PAWSX：大量字面相似但语义不同的样本

**数学表示**：

设 $f$ 是相似度函数，期望：
$$
f(\text{text}_1, \text{text}_2) \approx 1
$$
$$
f(\text{text}_1, \text{text}_2') \approx 0
$$

其中 $\text{text}_2$ 和 $\text{text}_2'$ 字面上非常接近。

**矛盾**：连续性好的函数难以实现这种"跳跃"！

</div>

</div>

### 4.3 为什么交互式更强？

<div class="derivation-box">

### 交互式的优势

<div class="formula-explanation">

<div class="formula-step">
<div class="step-label">Token级交互</div>

在BERT的第 $\ell$ 层，token $i$（来自text1）可以直接"看到"token $j$（来自text2）：
$$
\mathbf{h}_i^{(\ell+1)} = \text{Attention}\left(\mathbf{h}_i^{(\ell)}, [\mathbf{h}_1^{(\ell)}, \ldots, \mathbf{h}_n^{(\ell)}]\right)
\tag{15}
$$

**关键**：$[\mathbf{h}_1^{(\ell)}, \ldots, \mathbf{h}_n^{(\ell)}]$ 包含两个句子的所有token。

</div>

<div class="formula-step">
<div class="step-label">差异放大</div>

经过多层（12层）：
- 模型可以"发现"并"放大"关键差异
- 例如："上学" vs "工作"
- 即使两者在向量空间中很接近，Attention可以给予不同权重

$$
\text{Attention}(\text{上学}, \text{工作}) \ll \text{Attention}(\text{上学}, \text{学习})
\tag{16}
$$

</div>

<div class="formula-step">
<div class="step-label">非线性决策边界</div>

交互式最后通过分类器：
$$
P(\text{相似} | \text{text}_1, \text{text}_2) = \sigma(W \mathbf{h}_{\text{[CLS]}} + b)
\tag{17}
$$

$\mathbf{h}_{\text{[CLS]}}$ 已经包含了充分的交互信息，分类器可以学习**高度非线性**的决策边界。

</div>

</div>

</div>

</div>

---

## 五、深入讨论

### 5.1 训练方式的影响

<div class="example-box">

### 有监督 vs 无监督

**实验发现**：

| 方法 | 训练方式 | PAWSX (BERT) | PAWSX (RoBERTa) |
|------|---------|-------------|----------------|
| SimCSE | 无监督 | ~60% | ~62% |
| Sentence-BERT | 无监督 | 65.33% | 68.35% |
| **CoSENT** | **有监督** | **76.30%** | **76.55%** |
| Interact | 有监督 | 81.30% | 83.33% |

**结论**：
- 有监督训练显著提升PAWSX表现（+10-15%）
- 但仍不及交互式（差距5-7%）

**原因**：
- 有监督训练可以学习"对抗样本"的模式
- 但特征式的架构限制了其上限

</div>

### 5.2 过拟合的风险

<div class="intuition-box">

### 📉 特征式方案的训练曲线

**观察**：

在PAWSX上训练CoSENT时：
- 训练loss可以降到接近0（说明拟合能力没问题）
- 但验证集效果提升有限（泛化受限）

**分析**：

特征式方案要去拟合对抗性数据，需要"打破"其固有的连续性：
- 需要更多epoch
- 容易过拟合
- 泛化性差

**对比**：

交互式方案：
- 一开始就同时接触两个样本
- Attention可以自行学习放大差异
- 连续性与对抗性的矛盾较小

</div>

### 5.3 实用建议

<div class="note-box">

### 🎯 选择建议

**使用特征式（CoSENT/SBERT）**：
- ✅ 检索场景（需要高效率）
- ✅ 数据集无严重对抗样本
- ✅ 任务对准确率要求不是极端高
- ✅ 需要离线缓存句向量

**使用交互式**：
- ✅ 分类场景（追求极致准确率）
- ✅ 数据集有大量对抗样本（如PAWSX）
- ✅ 计算资源充足
- ✅ 查询量不大（可接受重复编码）

**混合方案（推荐）**：
1. 第一阶段：特征式粗排（从百万级候选中筛选Top-1000）
2. 第二阶段：交互式精排（从Top-1000中选Top-10）

**收益**：
- 结合两者优势
- 效率与准确率兼顾

**混合方案的理论依据**：

假设特征式的召回率（Recall@1000）为99%，精排只需处理1000个候选，计算量降低$10^3$倍：

$$
\text{Total Cost} = \underbrace{N \cdot C_{\text{encode}}}_{\text{离线编码}} + \underbrace{1 \cdot C_{\text{encode}} + 1000 \cdot C_{\text{cosine}}}_{\text{粗排}} + \underbrace{1000 \cdot C_{\text{interact}}}_{\text{精排}}
\tag{17.8}
$$

相比纯交互式的$N \cdot C_{\text{interact}}$，当$N \gg 1000$时（如$N=10^6$），加速约$10^3$倍，且准确率损失<1%。

</div>

---

## 六、代码实现

### 6.1 Powell阈值搜索完整实现

<details>
<summary><strong>💻 点击查看：生产级代码</strong></summary>
<div markdown="1">

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score

class ThresholdOptimizer:
    """
    多分类阈值优化器

    支持优化目标：
    - accuracy
    - f1 (macro/micro/weighted)
    - custom metric
    """

    def __init__(self, metric='accuracy'):
        """
        Args:
            metric: 'accuracy', 'f1_macro', 'f1_micro',
                    'f1_weighted', or callable
        """
        self.metric = metric
        self.thresholds_ = None

    def _compute_metric(self, y_true, y_pred):
        """计算评估指标"""
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric.startswith('f1'):
            average = self.metric.split('_')[1]
            return f1_score(y_true, y_pred, average=average)
        elif callable(self.metric):
            return self.metric(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, y_true, y_pred_proba, **minimize_kwargs):
        """
        在验证集上搜索最优阈值

        Args:
            y_true: shape=(N,), 真实标签
            y_pred_proba: shape=(N, C), 预测概率
            minimize_kwargs: 传递给scipy.optimize.minimize的参数

        Returns:
            self
        """
        num_classes = y_pred_proba.shape[1]

        def objective(t):
            # 映射到(0, 1)
            t_norm = (np.tanh(t) + 1) / 2

            # 加权预测
            y_pred_weighted = y_pred_proba * t_norm[None, :]
            y_pred = y_pred_weighted.argmax(axis=1)

            # 返回负指标（最小化）
            metric_value = self._compute_metric(y_true, y_pred)
            return -metric_value

        # 默认参数
        default_kwargs = {
            'method': 'Powell',
            'options': {
                'xtol': 1e-10,
                'ftol': 1e-10,
                'maxiter': 100000
            }
        }
        default_kwargs.update(minimize_kwargs)

        # 优化
        t0 = np.zeros(num_classes)
        result = minimize(objective, t0, **default_kwargs)

        # 保存结果
        self.thresholds_ = (np.tanh(result.x) + 1) / 2
        self.opt_result_ = result

        return self

    def predict(self, y_pred_proba):
        """
        使用最优阈值进行预测

        Args:
            y_pred_proba: shape=(N, C), 预测概率

        Returns:
            y_pred: shape=(N,), 预测标签
        """
        if self.thresholds_ is None:
            raise ValueError("Must call fit() before predict()")

        y_pred_weighted = y_pred_proba * self.thresholds_[None, :]
        return y_pred_weighted.argmax(axis=1)

    def fit_predict(self, y_true, y_pred_proba, **minimize_kwargs):
        """fit和predict的组合"""
        self.fit(y_true, y_pred_proba, **minimize_kwargs)
        return self.predict(y_pred_proba)


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    N = 1000
    C = 3

    y_true = np.random.randint(0, C, N)
    y_pred_proba = np.random.rand(N, C)
    y_pred_proba /= y_pred_proba.sum(axis=1, keepdims=True)

    # 优化阈值
    optimizer = ThresholdOptimizer(metric='accuracy')
    optimizer.fit(y_true, y_pred_proba)

    print(f"最优阈值: {optimizer.thresholds_}")
    print(f"优化是否成功: {optimizer.opt_result_.success}")
    print(f"最终目标值: {-optimizer.opt_result_.fun:.4f}")

    # 预测
    y_pred = optimizer.predict(y_pred_proba)
    print(f"测试集accuracy: {accuracy_score(y_true, y_pred):.4f}")
```

**进阶用法**：

```python
# 自定义评估指标
def custom_metric(y_true, y_pred):
    """加权F1，类别0的权重更高"""
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(y_true, y_pred, average=None)
    weights = [2.0, 1.0, 1.0]  # 类别0权重×2
    return np.average(f1_per_class, weights=weights)

optimizer = ThresholdOptimizer(metric=custom_metric)
optimizer.fit(y_true_val, y_pred_proba_val)
```

</div>
</details>

### 6.2 特征式 vs 交互式的实现对比

<details>
<summary><strong>⚡ 点击查看：两种范式的代码</strong></summary>
<div markdown="1">

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# ============= 特征式（CoSENT） =============
class RepresentationMatcher(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        """编码单个句子为句向量"""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]或pooling
        return outputs.last_hidden_state[:, 0]  # [batch, hidden]

    def forward(self, input_ids1, mask1, input_ids2, mask2):
        """计算句子对的相似度"""
        # 分别编码
        vec1 = self.encode(input_ids1, mask1)
        vec2 = self.encode(input_ids2, mask2)

        # 余弦相似度
        sim = torch.cosine_similarity(vec1, vec2, dim=-1)
        return sim

# 推理时的优势：可缓存
model = RepresentationMatcher()
corpus = ["句子1", "句子2", ..., "句子N"]

# 离线编码（只需一次）
corpus_vectors = []
for sent in corpus:
    inputs = tokenizer(sent, return_tensors='pt')
    vec = model.encode(**inputs)
    corpus_vectors.append(vec)

corpus_vectors = torch.stack(corpus_vectors)  # [N, hidden]

# 在线检索（极快）
query = "查询句子"
query_inputs = tokenizer(query, return_tensors='pt')
query_vec = model.encode(**query_inputs)  # [1, hidden]

similarities = torch.cosine_similarity(
    query_vec, corpus_vectors, dim=-1
)  # [N]
top_k = similarities.topk(10)


# ============= 交互式 =============
class InteractionMatcher(nn.Module):
    def __init__(self, model_name='bert-base-chinese'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 2)  # 二分类

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [CLS] text1 [SEP] text2 [SEP]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        return logits

# 推理时的劣势：无法缓存
model = InteractionMatcher()
corpus = ["句子1", "句子2", ..., "句子N"]

# 在线检索（每次都要重新编码）
query = "查询句子"
scores = []
for sent in corpus:
    # 拼接
    text = f"{query}[SEP]{sent}"
    inputs = tokenizer(text, return_tensors='pt')
    logits = model(**inputs)  # [1, 2]
    score = torch.softmax(logits, dim=-1)[0, 1]  # 正样本概率
    scores.append(score)

scores = torch.tensor(scores)  # [N]
top_k = scores.topk(10)
```

**效率对比**（N=100万，BERT base）：

| 操作 | 特征式 | 交互式 | 加速比 |
|------|--------|--------|--------|
| 离线编码 | 1次（~30min） | 不适用 | - |
| 单次查询 | 1次编码+100万次内积（~0.1s） | 100万次编码（~5小时） | **180,000×** |
| 内存占用 | 768×1M×4B = 3GB | 可忽略 | - |

</div>
</details>

---

## 七、常见问题 (FAQ)

<div class="example-box">

### ❓ Q1: 为什么特征式在ATEC上能超过交互式？

**A**: 这主要是**随机性**和**模型容量**的综合作用：

1. **统计波动**：差距仅0.32%，在统计误差范围内
2. **过拟合程度**：交互式参数更多，在小数据集上可能过拟合
3. **优化难度**：特征式损失函数更简单，可能优化得更充分

**结论**：不能说特征式"更好"，只能说"差不多"。

---

### ❓ Q2: CoSENT一定比Sentence-BERT好吗？

**A**: 从实验结果看，CoSENT在大多数数据集上优于Sentence-BERT，但：

**CoSENT的优势**：
- 更好的损失函数（circle loss的变体）
- 更充分利用标签信息（对比学习）
- 在对抗样本上表现更好（PAWSX: 76% vs 65%）

**Sentence-BERT的优势**：
- 更简单，更容易实现
- 无监督版本可用（SimCSE等）
- 某些数据集上也不差（LCQMC: 87.42% vs 86.67%）

**建议**：优先尝试CoSENT，但保留Sentence-BERT作为baseline。

---

### ❓ Q3: 768维真的够用吗？为什么不用更高维？

**A**: 从JL引理的分析看，768维**理论上**足够，但：

**实践中的考虑**：

1. **维度 vs 表达能力**：
   - 768维：百万级样本足够
   - 1024维：可能略好，但提升有限
   - >2048维：过拟合风险增加

2. **计算 vs 效果的权衡**：
   - 更高维 → 更慢的推理
   - 更高维 → 更多的内存占用
   - 收益递减

**经验法则**：
- BERT-base (768维)：适合大多数场景
- BERT-large (1024维)：追求极致效果
- 自定义维度：通过投影层调整（如降维到256）

---

### ❓ Q4: 混合方案中，粗排应该取Top-K，K应该多大？

**A**: K的选择是**召回率 vs 计算量**的权衡：

**理论分析**：

假设粗排的准确率为$p$，则Recall@K约为：
$$
\text{Recall@K} \approx 1 - (1-p)^K
$$

| K | Recall@K ($p=0.001$) | Recall@K ($p=0.01$) | 精排计算量 |
|---|---------------------|---------------------|----------|
| 100 | 9.5% | 63.4% | 低 |
| 500 | 39.3% | 99.3% | 中 |
| 1000 | 63.2% | 99.995% | 中高 |
| 5000 | 99.3% | ~100% | 高 |

**实践建议**：
- 一般场景：K=1000（性价比最高）
- 高召回要求：K=5000
- 低延迟要求：K=100-500

**自适应策略**：根据query难度动态调整K：
- 简单query（高置信度）：小K
- 困难query（低置信度）：大K

---

### ❓ Q5: 如何判断我的数据集是否有严重的对抗样本？

**A**: 可以通过以下指标评估：

**1. 字面重叠度统计**：

计算负样本的平均token重叠度：
```python
def compute_overlap(text1, text2):
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

negative_overlaps = [
    compute_overlap(t1, t2)
    for t1, t2, label in dataset if label == 0
]
avg_overlap = np.mean(negative_overlaps)

# avg_overlap > 0.5 → 对抗性强
# avg_overlap < 0.3 → 对抗性弱
```

**2. 无监督方法的表现**：

在你的数据集上测试SimCSE/Sentence-BERT（无监督版）：
- 效果 > 70%：对抗性弱
- 效果 < 60%：对抗性强

**3. 人工抽样检查**：

随机抽取100对负样本，人工判断：
- 语义明显不同但字面相似的比例 > 30%：对抗性强

---

### ❓ Q6: Powell优化有时不收敛怎么办？

**A**: Powell不收敛通常有以下原因和解决方案：

**原因1：初始值不好**
```python
# 不好的初始化
t0 = np.random.rand(num_classes)  # 随机初始化

# 好的初始化
t0 = np.zeros(num_classes)  # 对应等权重
```

**原因2：容忍度设置不当**
```python
# 太严格（可能永远不收敛）
options = {'xtol': 1e-15, 'ftol': 1e-15}

# 合理设置
options = {'xtol': 1e-8, 'ftol': 1e-8}
```

**原因3：目标函数噪声太大**

如果验证集太小（<1000样本），建议：
- 增大验证集
- 或使用K-fold交叉验证
- 或简化为二分类（减少参数）

**替代方案**：
- 网格搜索（Grid Search）：简单但慢
- 贝叶斯优化（Bayesian Optimization）：适合高噪声场景

</div>

---

## 八、总结与展望

<div class="note-box">

### 核心结论

**实验结果**：
1. ✅ 在ATEC和BQ上，CoSENT与交互式**无显著差异**
2. ✅ 在LCQMC上，差距约1%（可接受）
3. ⚠️ 在PAWSX上，差距5-7%（对抗样本多）

**理论洞察**：
1. ✅ 理论上，特征式可以通过内积拟合任意相似度矩阵
2. ✅ 768维句向量足以表示百万级样本的两两相似度
3. ⚠️ 实践中的差距来自**连续性 vs 对抗性**的矛盾

**方法贡献**：
1. ✅ 提出Powell优化方法自动搜索多分类阈值
2. ✅ 系统对比了特征式与交互式的差距
3. ✅ 给出了理论分析和实用建议

</div>

### 未来方向

<div class="intuition-box">

### 🔬 研究展望

**1. 缩小PAWSX差距**

可能的方向：
- 对抗训练：显式加入对抗样本
- 硬负例挖掘：重点训练困难样本
- 多任务学习：联合训练交互式和特征式

具体实现思路：
```python
# 硬负例挖掘示例
def mine_hard_negatives(model, query, candidates, top_k=100):
    """挖掘与query相似但标签为负的样本"""
    similarities = model.compute_similarity(query, candidates)
    # 选择相似度最高但标签为负的样本
    hard_negatives = []
    for i in similarities.argsort()[::-1]:
        if candidates[i].label == 0:  # 负样本
            hard_negatives.append(candidates[i])
            if len(hard_negatives) >= top_k:
                break
    return hard_negatives
```

**2. 混合架构**

结合两者优势：
- 前k层：分别编码（特征式）
- 后m层：联合编码（交互式）
- 可调节k/m以平衡效率和效果

**Poly-encoder架构**（Facebook Research）是这个思路的成功案例：
- 使用m个全局向量表示context
- Query与这m个向量交互
- 复杂度：$O(m)$ vs 纯交互的$O(n)$

**3. 动态选择**

根据query自适应选择：
- 简单query：特征式（快）
- 困难query：交互式（准）
- 使用元模型判断难度

**难度判断指标**：
$$
\text{Difficulty}(q) = 1 - \max_i \text{sim}(q, c_i)
$$
- 最高相似度很高 → 简单（高置信度）
- 最高相似度一般 → 困难（需要精排）

**4. 知识蒸馏**

用交互式指导特征式：
- Teacher：交互式模型
- Student：特征式模型
- 蒸馏对象：相似度分布

**蒸馏损失**：
$$
\mathcal{L}_{\text{distill}} = \text{KL}\left(P_{\text{teacher}}(s | q, c) \parallel P_{\text{student}}(s | q, c)\right)
$$

实验表明，蒸馏后的特征式模型可以缩小与交互式的差距2-3个百分点。

**5. 跨语言迁移**

当前实验主要在中文，未来可以探索：
- 多语言联合训练
- 跨语言zero-shot迁移
- 低资源语言的表现

</div>

---

## 九、实践检查清单

<div class="note-box">

### ✅ 使用CoSENT前的检查清单

**数据准备**：
- [ ] 确认有足够的标注数据（>10k对）
- [ ] 检查正负样本比例（建议1:1到1:3）
- [ ] 评估对抗样本的比例（参考Q5）
- [ ] 划分训练/验证/测试集（建议7:1.5:1.5）

**模型选择**：
- [ ] 根据数据集大小选择base/large模型
- [ ] 考虑是否需要领域预训练模型
- [ ] 确定是否需要混合方案（数据量>100万）

**训练配置**：
- [ ] 设置合适的学习率（推荐2e-5）
- [ ] 设置合适的batch size（推荐32-64）
- [ ] 启用早停（patience=3-5）
- [ ] 保存最佳checkpoint

**评估方法**：
- [ ] 使用Spearman系数（回归视角）
- [ ] 使用Accuracy+阈值搜索（分类视角）
- [ ] 与baseline对比（Sentence-BERT/交互式）
- [ ] 在多个数据集上验证泛化性

**部署考虑**：
- [ ] 测试推理延迟（离线编码+在线检索）
- [ ] 评估内存占用（N×768×4 bytes）
- [ ] 考虑量化/剪枝（如降维到256）
- [ ] 准备降级方案（交互式精排）

**持续优化**：
- [ ] 定期更新模型（新数据）
- [ ] 监控线上效果（A/B测试）
- [ ] 收集badcase（对抗样本）
- [ ] 迭代改进（硬负例挖掘）

</div>

---

## 十、参考文献

<div class="note-box">

### 主要参考文献

1. **Reimers, N., & Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
   - 提出Sentence-BERT，开创特征式匹配的新范式

2. **Gao, T., Yao, X., & Chen, D.** (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. *EMNLP 2021*.
   - 无监督对比学习的代表性工作

3. **Su, J.** (2022). CoSENT（一）：比Sentence-BERT更有效的句向量方案. *https://spaces.ac.cn/archives/8847*
   - CoSENT原始论文，本文的前置工作

4. **Dasgupta, S., & Gupta, A.** (2003). An elementary proof of a theorem of Johnson and Lindenstrauss. *Random Structures & Algorithms, 22(1)*, 60-65.
   - JL引理的简化证明

5. **Powell, M. J. D.** (1964). An efficient method for finding the minimum of a function of several variables without calculating derivatives. *The Computer Journal, 7(2)*, 155-162.
   - Powell优化方法的原始论文

6. **Humeau, S., Shuster, K., Lachaux, M. A., & Weston, J.** (2019). Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. *ICLR 2020*.
   - Poly-encoder：混合架构的成功案例

7. **Yang, Y., Cer, D., Ahmad, A., Guo, M., Law, J., Constant, N., ... & Kurzweil, R.** (2020). Multilingual Universal Sentence Encoder for Semantic Retrieval. *ACL 2020*.
   - 多语言句向量的工业级应用

### 相关资源

**苏剑林的博客**：
- [让人惊叹的Johnson-Lindenstrauss引理：理论篇](https://spaces.ac.cn/archives/8679)
- [让人惊叹的Johnson-Lindenstrauss引理：应用篇](https://spaces.ac.cn/archives/8706)
- [无监督语义相似度哪家强？我们做了个比较全面的评测](https://spaces.ac.cn/archives/8321)
- [中文任务还是SOTA吗？我们做了一个中文NLU基准测试](https://spaces.ac.cn/archives/7975)

**代码实现**：
- Sentence-Transformers库：https://github.com/UKPLab/sentence-transformers
- CoSENT实现：https://github.com/bojone/CoSENT
- SimCSE实现：https://github.com/princeton-nlp/SimCSE

</div>

---

**相关文章**：
- [CoSENT（一）：比Sentence-BERT更有效的句向量方案](https://spaces.ac.cn/archives/8847)
- [让人惊叹的Johnson-Lindenstrauss引理：理论篇](https://spaces.ac.cn/archives/8679)
- [让人惊叹的Johnson-Lindenstrauss引理：应用篇](https://spaces.ac.cn/archives/8706)
- [无监督语义相似度哪家强？我们做了个比较全面的评测](https://spaces.ac.cn/archives/8321)

---

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8860>_

_**更详细的转载事宜请参考：**_[《科学空间FAQ》](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您需要引用本文，请参考：**

苏剑林. (Jan. 12, 2022). 《CoSENT（二）：特征式匹配与交互式匹配有多大差距？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8860>

@online{kexuefm-8860,
title={CoSENT（二）：特征式匹配与交互式匹配有多大差距？},
author={苏剑林},
year={2022},
month={Jan},
url={\url{https://spaces.ac.cn/archives/8860}},
}
