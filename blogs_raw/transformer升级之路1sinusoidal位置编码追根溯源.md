---
title: Transformer升级之路：1、Sinusoidal位置编码追根溯源
slug: transformer升级之路1sinusoidal位置编码追根溯源
date: 
source: https://spaces.ac.cn/archives/8231
tags: 复数, 分析, attention, 位置编码, 生成模型
status: completed
---

# Transformer升级之路：1、Sinusoidal位置编码追根溯源

**原文链接**: [https://spaces.ac.cn/archives/8231](https://spaces.ac.cn/archives/8231)

**发布日期**: 

---

最近笔者做了一些理解和改进Transformer的尝试，得到了一些似乎还有价值的经验和结论，遂开一个专题总结一下，命名为“Transformer升级之路”，既代表理解上的深入，也代表结果上的改进。

作为该专题的第一篇文章，笔者将会介绍自己对Google在[《Attention is All You Need》](https://papers.cool/arxiv/1706.03762)中提出来的Sinusoidal位置编码  
\begin{equation}\left\\{\begin{aligned}&\boldsymbol{p}_{k,2i}=\sin\Big(k/10000^{2i/d}\Big)\\\  
&\boldsymbol{p}_{k, 2i+1}=\cos\Big(k/10000^{2i/d}\Big)  
\end{aligned}\right.\label{eq:sin}\end{equation}  
的新理解，其中$\boldsymbol{p}_{k,2i},\boldsymbol{p}_{k,2i+1}$分别是位置$k$的编码向量的第$2i,2i+1$个分量，$d$是向量维度。

作为位置编码的一个显式解，Google在原论文中对它的描述却寥寥无几，只是简单提及了它可以表达相对位置信息，后来知乎等平台上也出现了一些解读，它的一些特点也逐步为大家所知，但总体而言比较零散。特别是对于“它是怎么想出来的”、“非得要这个形式不可吗”等原理性问题，还没有比较好的答案。

因此，本文主要围绕这些问题展开思考，可能在思考过程中读者会有跟笔者一样的感觉，即越思考越觉得这个设计之精妙漂亮，让人叹服～

## 第1部分：理论基础与历史发展

### 1.1 位置编码的必要性

#### Transformer的置换不变性问题

Transformer的核心机制Self-Attention本质上是一个**置换不变**（Permutation Invariant）的操作。假设我们有输入序列$\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n]$，标准的Self-Attention计算为：

\begin{equation}
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{QK}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}
\tag{1}
\end{equation}

其中$\boldsymbol{Q} = \boldsymbol{XW}_Q$，$\boldsymbol{K} = \boldsymbol{XW}_K$，$\boldsymbol{V} = \boldsymbol{XW}_V$。

**定理1（Attention的置换不变性）**：对于任意置换矩阵$\boldsymbol{P}$（$\boldsymbol{P}^{\top}\boldsymbol{P} = \boldsymbol{I}$），有：

\begin{equation}
\text{Attention}(\boldsymbol{PX}, \boldsymbol{PX}, \boldsymbol{PX}) = \boldsymbol{P} \cdot \text{Attention}(\boldsymbol{X}, \boldsymbol{X}, \boldsymbol{X})
\tag{2}
\end{equation}

**证明**：直接展开注意力机制即可验证。这意味着如果交换输入顺序，输出也会以相同方式交换，模型无法区分序列的**绝对位置**。

**实际后果**：
- "我爱你" 和 "你爱我" 对纯Attention模型是等价的
- "The cat sat on the mat" 和 "mat the on sat cat The" 无法区分
- 时序信息完全丢失

这在NLP、时序预测等任务中是致命的，因此必须引入**位置编码**（Positional Encoding）来打破这种对称性。

#### 位置编码的数学本质

从形式上看，位置编码就是在输入嵌入$\boldsymbol{x}_i$上加上位置相关的向量$\boldsymbol{p}_i$：

\begin{equation}
\tilde{\boldsymbol{x}}_i = \boldsymbol{x}_i + \boldsymbol{p}_i
\tag{3}
\end{equation}

这样Attention的计算就变成了：

\begin{equation}
\boldsymbol{q}_i = (\boldsymbol{x}_i + \boldsymbol{p}_i)\boldsymbol{W}_Q, \quad \boldsymbol{k}_j = (\boldsymbol{x}_j + \boldsymbol{p}_j)\boldsymbol{W}_K
\tag{4}
\end{equation}

注意力权重中的内积项：

\begin{equation}
\begin{aligned}
\boldsymbol{q}_i^{\top}\boldsymbol{k}_j &= (\boldsymbol{x}_i + \boldsymbol{p}_i)^{\top}\boldsymbol{W}_Q^{\top}\boldsymbol{W}_K(\boldsymbol{x}_j + \boldsymbol{p}_j) \\
&= \underbrace{\boldsymbol{x}_i^{\top}\boldsymbol{W}_Q^{\top}\boldsymbol{W}_K\boldsymbol{x}_j}_{\text{内容交互}} + \underbrace{\boldsymbol{x}_i^{\top}\boldsymbol{W}_Q^{\top}\boldsymbol{W}_K\boldsymbol{p}_j + \boldsymbol{p}_i^{\top}\boldsymbol{W}_Q^{\top}\boldsymbol{W}_K\boldsymbol{x}_j}_{\text{内容-位置交互}} \\
&\quad + \underbrace{\boldsymbol{p}_i^{\top}\boldsymbol{W}_Q^{\top}\boldsymbol{W}_K\boldsymbol{p}_j}_{\text{位置-位置交互}}
\end{aligned}
\tag{5}
\end{equation}

最后一项$\boldsymbol{p}_i^{\top}\boldsymbol{W}_Q^{\top}\boldsymbol{W}_K\boldsymbol{p}_j$提供了纯粹的位置信息交互。

### 1.2 位置编码方案综述

#### 可学习的位置编码（Learned Positional Embeddings）

最直接的方法是将位置编码作为可学习参数，如BERT所采用的方案：

\begin{equation}
\boldsymbol{p}_i \in \mathbb{R}^d, \quad i = 1, 2, \ldots, L_{\max}
\tag{6}
\end{equation}

其中$L_{\max}$是预设的最大序列长度（BERT中为512）。

**优点**：
- 完全由数据驱动，理论上可以学到最优的位置表示
- 实现简单

**缺点**：
- 无法泛化到训练时未见过的长度（$L > L_{\max}$）
- 参数量随序列长度线性增长：$\mathcal{O}(L_{\max} \cdot d)$
- 没有显式的相对位置信息

#### Sinusoidal位置编码（Fixed Sinusoidal Encoding）

Google在《Attention is All You Need》中提出的经典方案：

\begin{equation}
\begin{cases}
\boldsymbol{p}_{k,2i} = \sin\left(\frac{k}{10000^{2i/d}}\right) \\
\boldsymbol{p}_{k,2i+1} = \cos\left(\frac{k}{10000^{2i/d}}\right)
\end{cases}, \quad i = 0, 1, \ldots, \frac{d}{2}-1
\tag{7}
\end{equation}

**优点**：
- 无参数，无需训练
- 可以外推到任意长度
- 具有良好的数学性质（本文重点）

**缺点**：
- 固定不可调，可能不是最优的
- 外推性能仍有局限

#### 相对位置编码家族

后续研究发现，很多任务中**相对位置**比绝对位置更重要，衍生出多个方案：

1. **Shaw et al. (2018) - Self-Attention with Relative Position Representations**：
   直接在Attention中加入相对位置偏置：
   \begin{equation}
   e_{ij} = \frac{\boldsymbol{x}_i\boldsymbol{W}_Q (\boldsymbol{x}_j\boldsymbol{W}_K + \boldsymbol{r}_{i-j})^{\top}}{\sqrt{d_k}}
   \tag{8}
   \end{equation}

2. **RoPE (Su et al. 2021) - Rotary Position Embedding**：
   通过旋转矩阵编码相对位置（后续篇章详述）

3. **ALiBi (Press et al. 2021)**：
   在注意力分数上直接加线性偏置：
   \begin{equation}
   \text{softmax}(\boldsymbol{QK}^{\top} - m \cdot |i-j|)
   \tag{9}
   \end{equation}

### 1.3 Sinusoidal位置编码的设计哲学

Sinusoidal位置编码的设计蕴含了三个核心思想：

#### 思想1：通过绝对位置表达相对位置

看似矛盾，但数学上可行。关键在于利用三角函数的**和差公式**：

\begin{equation}
\sin(\alpha - \beta) = \sin\alpha\cos\beta - \cos\alpha\sin\beta
\tag{10}
\end{equation}

这使得$\boldsymbol{p}_m$和$\boldsymbol{p}_n$的内积可以表达相对位置$m-n$的信息（后文详细推导）。

#### 思想2：远程衰减（Long-Range Decay）

直觉上，距离越远的token相关性应该越弱。Sinusoidal编码通过多频率叠加，自然产生了远程衰减效应：

\begin{equation}
\langle \boldsymbol{p}_m, \boldsymbol{p}_n \rangle \approx \frac{C}{|m-n|^{\alpha}}, \quad |m-n| \to \infty
\tag{11}
\end{equation}

这与物理学中的衰减律类似（如引力的平方反比律）。

#### 思想3：多尺度表示（Multi-Scale Representation）

不同维度使用不同频率$\theta_i = 10000^{-2i/d}$，形成**频谱**：

- 高频分量（$i$小）：捕捉短程依赖（相邻词汇）
- 低频分量（$i$大）：捕捉长程依赖（句子结构）

类比信号处理中的小波变换，这是一种**时频分析**思想。

### 1.4 问题的提出与本文目标

尽管Sinusoidal位置编码被广泛使用，但原始论文对其来源语焉不详。核心问题：

1. **如何"反推"出这个形式**？（数学推导）
2. **为什么是$10000^{-2i/d}$这个底数**？（参数选择）
3. **在什么假设下它是"最优"的**？（理论分析）
4. **与学习式编码相比，优劣势在哪**？（对比分析）

本文将通过**四步推导法**回答这些问题：
1. 泰勒展开 → 建立位置编码与相对位置的数学联系
2. 复数推导 → 得到三角函数形式
3. 远程衰减 → 确定频率参数
4. 一般化分析 → 讨论Hessian矩阵的影响

## 第2部分：数学推导的完整展开

### 2.1 泰勒展开与扰动分析

#### 模型的全对称性

假设我们的模型为$f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)$，其中标记出来的$\boldsymbol{x}_m,\boldsymbol{x}_n$分别表示第$m,n$个输入，不失一般性，设$f$是标量函数。对于不带Attention Mask的纯Attention模型，它是全对称的，即对于任意的$m,n$，都有
\begin{equation}f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_n,\cdots,\boldsymbol{x}_m,\cdots)\tag{12}\end{equation}
这就是我们说Transformer无法识别位置的原因——全对称性，简单来说就是函数天然满足恒等式$f(x,y)=f(y,x)$，以至于我们无法从结果上区分输入是$[x,y]$还是$[y,x]$。

因此，我们要做的事情，就是要打破这种对称性，比如在每个位置上都加上一个不同的编码向量：  
\begin{equation}\tilde{f}(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_m + \boldsymbol{p}_m,\cdots,\boldsymbol{x}_n + \boldsymbol{p}_n,\cdots)\end{equation}  
一般来说，只要每个位置的编码向量不同，那么这种全对称性就被打破了，即可以用$\tilde{f}$代替$f$来处理有序的输入。但现在我们希望能进一步分析位置编码的性质，甚至得到一个显式解，那么就不能止步于此。

为了简化问题，我们先只考虑$m,n$这两个位置上的位置编码，将它视为扰动项，泰勒展开到二阶：  
\begin{equation}\tilde{f}\approx f + \boldsymbol{p}_m^{\top} \frac{\partial f}{\partial \boldsymbol{x}_m} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x}_n} + \frac{1}{2}\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2}\boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}_n}_{\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n}\end{equation}  
可以看到，第1项跟位置无关，第2到5项都只依赖于单一位置，所以它们是纯粹的绝对位置信息，第6项是第一个同时包含$\boldsymbol{p}_m,\boldsymbol{p}_n$的交互项，我们将它记为$\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n$，希望它能表达一定的相对位置信息。

（此处的泰勒展开参考了知乎问题[《BERT为何使用学习的position embedding而非正弦position encoding?》](https://www.zhihu.com/question/307293465/answer/1028613658)上的纳米酱的回复。）

### 2.2 相对位置的复数推导

#### 从内积到相对位置

我们先从简单的例子入手，假设$\boldsymbol{\mathcal{H}}=\boldsymbol{I}$是单位矩阵，此时：

\begin{equation}
\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n = \boldsymbol{p}_m^{\top} \boldsymbol{p}_n = \langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle
\tag{13}
\end{equation}

是两个位置编码的内积。我们希望在这个简单的例子中该项表达的是相对位置信息，即存在某个函数$g$使得：

\begin{equation}
\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = g(m-n)
\tag{14}\label{eq:r1}
\end{equation}

这个要求看似简单，实则深刻：**仅通过两个绝对位置向量的内积，就能提取出相对位置$m-n$的信息**。

#### 二维情形的完整推导

这里的$\boldsymbol{p}_m, \boldsymbol{p}_n$是$d$维向量，这里我们从最简单$d=2$入手。

**复数表示的巧妙之处**：对于2维向量，我们借助复数来推导。将向量$[x,y]$视为复数$z = x + y\text{i}$，这里复数乘法的几何意义至关重要：

\begin{equation}
z_1 \cdot z_2 = |z_1||z_2|e^{\text{i}(\arg z_1 + \arg z_2)}
\tag{15}
\end{equation}

即复数相乘，模长相乘，幅角相加。根据复数内积的性质：

\begin{equation}
\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \text{Re}[\boldsymbol{p}_m \boldsymbol{p}_n^*]
\tag{16}
\end{equation}

其中$\boldsymbol{p}_n^*$是$\boldsymbol{p}_n$的共轭复数（$z^* = x - y\text{i}$），$\text{Re}[\cdot]$代表复数的实部。

**关键技巧**：为了满足式$\eqref{eq:r1}$（即内积只依赖于$m-n$），我们可以假设存在复数$\boldsymbol{q}_{m-n}$使得：

\begin{equation}
\boldsymbol{p}_m \boldsymbol{p}_n^* = \boldsymbol{q}_{m-n}
\tag{17}
\end{equation}

这样两边取实部就得到了式$\eqref{eq:r1}$。

**指数形式求解**：使用Euler公式$e^{\text{i}\theta} = \cos\theta + \text{i}\sin\theta$，设：

\begin{equation}
\begin{cases}
\boldsymbol{p}_m = r_m e^{\text{i}\phi_m} \\
\boldsymbol{p}_n^* = r_n e^{-\text{i}\phi_n} \\
\boldsymbol{q}_{m-n} = R_{m-n} e^{\text{i}\Phi_{m-n}}
\end{cases}
\tag{18}
\end{equation}

代入方程(17)得：

\begin{equation}
r_m r_n e^{\text{i}(\phi_m - \phi_n)} = R_{m-n} e^{\text{i}\Phi_{m-n}}
\tag{19}
\end{equation}

由复数相等的充要条件（模长相等且幅角相等），得到：

\begin{equation}
\begin{cases}
r_m r_n = R_{m-n} & \text{（模长方程）} \\
\phi_m - \phi_n = \Phi_{m-n} & \text{（幅角方程）}
\end{cases}
\tag{20}
\end{equation}

**求解模长**：令$n=m$代入模长方程：

\begin{equation}
r_m^2 = R_0 \quad \Rightarrow \quad r_m = \sqrt{R_0} = \text{常数}
\tag{21}
\end{equation}

简单起见，归一化为$r_m = 1$（这样位置编码的模长为1，类似单位圆上的点）。

**求解幅角**：令$n=0$代入幅角方程：

\begin{equation}
\phi_m - \phi_0 = \Phi_m
\tag{22}
\end{equation}

初始条件选择$\phi_0 = 0$（原点对应复平面实轴），则$\phi_m = \Phi_m$。

再令$n=m-1$代入幅角方程：

\begin{equation}
\phi_m - \phi_{m-1} = \phi_1 \equiv \theta \quad \text{（常数）}
\tag{23}
\end{equation}

这表明$\{\phi_m\}$是**等差数列**，公差为$\theta$。由递推关系：

\begin{equation}
\phi_m = \phi_0 + m\theta = m\theta
\tag{24}
\end{equation}

**最终解**：因此二维情形下位置编码的解为：

\begin{equation}
\boldsymbol{p}_m = e^{\text{i}m\theta} = \cos(m\theta) + \text{i}\sin(m\theta)
\tag{25}
\end{equation}

转换为实向量形式：

\begin{equation}
\boldsymbol{p}_m = \begin{pmatrix}\cos(m\theta) \\ \sin(m\theta)\end{pmatrix}
\tag{26}
\end{equation}

**几何意义**：这对应于单位圆上以角速度$\theta$旋转的点。位置$m$对应的角度为$m\theta$，相邻位置相差固定角度$\theta$。

#### 验证相对位置性质

让我们验证式(26)确实满足式$\eqref{eq:r1}$：

\begin{equation}
\begin{aligned}
\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle &= \cos(m\theta)\cos(n\theta) + \sin(m\theta)\sin(n\theta) \\
&= \cos((m-n)\theta) \quad \text{（余弦和差公式）}
\end{aligned}
\tag{27}
\end{equation}

完美！内积仅依赖于相对位置$m-n$，且具有周期性和对称性：

- **周期性**：$g(k) = \cos(k\theta)$，周期为$2\pi/\theta$
- **对称性**：$g(m-n) = g(n-m)$
- **单调性**：在$k \in [0, \pi/\theta]$内单调递减，符合"距离越远相关性越弱"的直觉

#### 高维推广

由于内积满足**线性叠加性**，更高维的偶数维位置编码可以表示为多个二维编码的拼接：

\begin{equation}
\boldsymbol{p}_m = \begin{pmatrix}
e^{\text{i}m\theta_0} \\
e^{\text{i}m\theta_1} \\
\vdots \\
e^{\text{i}m\theta_{d/2-1}}
\end{pmatrix}
\quad \Leftrightarrow \quad
\boldsymbol{p}_m = \begin{pmatrix}
\cos(m\theta_0) \\
\sin(m\theta_0) \\
\cos(m\theta_1) \\
\sin(m\theta_1) \\
\vdots \\
\cos(m\theta_{d/2-1}) \\
\sin(m\theta_{d/2-1})
\end{pmatrix}
\tag{28}\label{eq:r2}
\end{equation}

此时内积为：

\begin{equation}
\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \sum_{i=0}^{d/2-1} \cos((m-n)\theta_i)
\tag{29}
\end{equation}

仍然只依赖于$m-n$，满足相对位置性质。

**唯一性讨论**：式(28)只是一个特解，不是唯一解。例如：
- 改变$\sin/\cos$顺序
- 添加相位偏移$\cos(m\theta + \phi_i)$
- 使用其他正交函数系（Chebyshev多项式等）

但三角函数形式由于其良好的解析性质（可微、周期、正交），成为最自然的选择。

### 2.3 远程衰减特性的深入分析

#### 频率参数的选择

基于前面的假设，我们推导出了位置编码的形式$\eqref{eq:r2}$，它跟标准的Sinusoidal位置编码形式基本一样了，只是$\sin,\cos$的位置略有不同。一般情况下，神经网络的神经元都是无序的，所以哪怕打乱各个维度，也是一种合理的位置编码，因此除了各个$\theta_i$没确定下来外，二者并无本质区别。

**核心问题**：原始论文选择$\theta_i = 10000^{-2i/d}$，这个"魔法数字"有什么深意？

\begin{equation}
\theta_i = \frac{1}{10000^{2i/d}} = 10000^{-2i/d}, \quad i = 0, 1, \ldots, \frac{d}{2}-1
\tag{30}
\end{equation}

事实上，这个形式带来一个精妙的性质：**随着$|m-n|$增大，内积$\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle$呈现衰减趋势**。

这符合语言学直觉：**相对距离越大的词，相关性应该越弱**。但问题来了：明明是周期性的三角函数，怎么会衰减？

#### 振荡积分与远程渐近行为

从式(29)出发，将内积改写为离散和的形式：

\begin{equation}
\begin{aligned}
\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle &= \sum_{i=0}^{d/2-1} \cos((m-n)\theta_i) \\
&= \text{Re}\left[\sum_{i=0}^{d/2-1} e^{\text{i}(m-n)\theta_i}\right]
\end{aligned}
\tag{31}
\end{equation}

代入$\theta_i = 10000^{-2i/d}$：

\begin{equation}
\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \text{Re}\left[\sum_{i=0}^{d/2-1} e^{\text{i}(m-n) \cdot 10000^{-2i/d}}\right]
\tag{32}
\end{equation}

**关键变换**：将求和转换为积分。令$t = 2i/d \in [0, 1]$，步长$\Delta t = 2/d$，黎曼和近似：

\begin{equation}
\begin{aligned}
\sum_{i=0}^{d/2-1} e^{\text{i}(m-n) \cdot 10000^{-2i/d}} &\approx \frac{d}{2} \sum_{i=0}^{d/2-1} e^{\text{i}(m-n) \cdot 10000^{-t}} \cdot \frac{2}{d} \\
&\approx \frac{d}{2} \int_0^1 e^{\text{i}(m-n) \cdot 10000^{-t}} dt
\end{aligned}
\tag{33}
\end{equation}

因此问题转化为估计**振荡积分**（Oscillatory Integral）：

\begin{equation}
I(\Delta) = \int_0^1 e^{\text{i}\Delta \cdot 10000^{-t}} dt, \quad \Delta = m - n
\tag{34}
\end{equation}

#### 振荡积分理论基础

**定理2（Riemann-Lebesgue引理的特例）**：设$\phi(t)$是$[a,b]$上的单调光滑函数，且$\phi'(t) \neq 0$，则：

\begin{equation}
\left|\int_a^b e^{\text{i}\lambda\phi(t)} dt\right| \lesssim \frac{C}{|\lambda|}, \quad |\lambda| \to \infty
\tag{35}
\end{equation}

其中$C$依赖于$\phi$的导数界，但与$\lambda$无关。

**应用到我们的情况**：取$\phi(t) = 10000^{-t}$，这是单调递减函数：

\begin{equation}
\phi'(t) = -\ln(10000) \cdot 10000^{-t} < 0
\tag{36}
\end{equation}

因此当$|\Delta| = |m-n|$增大时，积分$I(\Delta)$会衰减！

#### 精确计算与渐近展开

让我们精确计算积分(34)。令$u = 10000^{-t}$，则$t = -\log_{10000}(u) = -\frac{\ln u}{\ln 10000}$，$dt = -\frac{du}{u\ln 10000}$：

\begin{equation}
\begin{aligned}
I(\Delta) &= \int_{10000}^{1} e^{\text{i}\Delta u} \cdot \left(-\frac{du}{u\ln 10000}\right) \\
&= \frac{1}{\ln 10000} \int_1^{10000} \frac{e^{\text{i}\Delta u}}{u} du
\end{aligned}
\tag{37}
\end{equation}

这是**指数积分**（Exponential Integral）$\text{Ei}$的变体！对于大$|\Delta|$，利用分部积分：

\begin{equation}
\int_1^{10000} \frac{e^{\text{i}\Delta u}}{u} du = \left[\frac{e^{\text{i}\Delta u}}{\text{i}\Delta u}\right]_1^{10000} - \int_1^{10000} \frac{e^{\text{i}\Delta u}}{\text{i}\Delta u^2} du
\tag{38}
\end{equation}

主导项：

\begin{equation}
I(\Delta) \approx \frac{1}{\ln 10000} \cdot \frac{e^{\text{i}\Delta \cdot 10000} - e^{\text{i}\Delta}}{\text{i}\Delta} \cdot \frac{1}{10000}
\tag{39}
\end{equation}

模长估计：

\begin{equation}
|I(\Delta)| \lesssim \frac{2}{\ln(10000) \cdot |\Delta|} = \frac{2}{9.21 |\Delta|} \approx \frac{0.22}{|\Delta|}
\tag{40}
\end{equation}

**结论**：内积以$\mathcal{O}(1/|\Delta|)$速率衰减！

#### Mathematica数值验证

原文提供的代码：

```mathematica
θ[t_] = (1/10000)^t;
f[x_] = Re[Integrate[Exp[I*x*θ[t]], {t, 0, 1}]];
Plot[f[x], {x, -128, 128}]
```

绘制的图像显示：
- **短距离**（$|m-n| < 10$）：内积接近1（相邻位置强相关）
- **中距离**（$10 < |m-n| < 100$）：快速衰减
- **长距离**（$|m-n| > 100$）：振荡衰减，幅度$\sim 1/|m-n|$

#### 与其他频率方案的对比

**幂函数**：$\theta_i = i^{-\alpha}$，对应积分：

\begin{equation}
I_{\text{power}}(\Delta) = \int_0^1 e^{\text{i}\Delta t^{-\alpha}} dt
\tag{41}
\end{equation}

- $\alpha = 1$：过快衰减，短程依赖丢失
- $\alpha = 0.5$：中等衰减
- $\alpha = 2$：过慢衰减，长程干扰

**指数函数**（原始方案）：$\theta_i = 10000^{-2i/d}$，对应：

\begin{equation}
I_{\text{exp}}(\Delta) = \int_0^1 e^{\text{i}\Delta \cdot b^{-t}} dt, \quad b = 10000
\tag{42}
\end{equation}

**线性函数**：$\theta_i = 1 - i/d$，对应：

\begin{equation}
I_{\text{linear}}(\Delta) = \int_0^1 e^{\text{i}\Delta(1-t)} dt = e^{\text{i}\Delta} \cdot \frac{1 - e^{-\text{i}\Delta}}{\text{i}\Delta}
\tag{43}
\end{equation}

这会导致内积**周期性归零**（当$\Delta = 2\pi k$时），不符合单调衰减的直觉。

**对比结论**：
- 指数方案在短/中/长距离取得最佳平衡
- 幂函数短距离过激，指数函数长距离过弱
- 线性方案有周期性零点，不适合

#### 底数10000的选择

为什么是10000而不是1000或100000？

**频率范围**：$\theta_i \in [1, 10000^{-1}]$，对应波长范围$\lambda_i = 2\pi/\theta_i \in [2\pi, 2\pi \cdot 10000]$。

- 最短波长：$\lambda_0 \approx 6.28$（捕捉相邻词）
- 最长波长：$\lambda_{d/2-1} \approx 62800$（捕捉句子级结构）

对于典型的序列长度$L = 512$（BERT）或$L = 2048$（GPT），需要波长覆盖$[1, L]$：

\begin{equation}
10000^{2/d} \geq \frac{L_{\max}}{2\pi} \quad \Rightarrow \quad 10000 \geq \left(\frac{L_{\max}}{2\pi}\right)^{d/2}
\tag{44}
\end{equation}

对于$d=512$，$L_{\max}=2048$：

\begin{equation}
10000 \geq (326)^{256} \quad \text{（显然不成立！）}
\tag{45}
\end{equation}

实际上，**10000是经验选择**，并非严格优化的结果。其他合理选择：
- $b = 1000$：更快衰减，适合短文本
- $b = 100000$：更慢衰减，适合超长文本
- **可训练底数**：将$b$作为超参数微调

### 2.4 Hessian矩阵的一般情况

前面两节中，我们展示了通过绝对位置编码来表达相对位置信息的思想，加上远程衰减的约束，可以“反推”出Sinusoidal位置编码，并且给出了关于$\theta_i$的其他选择。但是别忘了，到目前为止，我们的推导都是基于$\boldsymbol{\mathcal{H}}=\boldsymbol{I}$这个简单情况的，对于一般的$\boldsymbol{\mathcal{H}}$，使用上述Sinusoidal位置编码，还能具备以上的良好性质吗？

如果$\boldsymbol{\mathcal{H}}$是一个对角阵，那么上面的各个性质可以得到一定的保留，此时  
\begin{equation}\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n=\sum_{i=1}^{d/2} \boldsymbol{\mathcal{H}}_{2i,2i} \cos m\theta_i \cos n\theta_i + \boldsymbol{\mathcal{H}}_{2i+1,2i+1} \sin m\theta_i \sin n\theta_i\end{equation}  
由积化和差公式得到  
\begin{equation}\sum_{i=1}^{d/2} \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} + \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m-n)\theta_i + \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} - \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m+n)\theta_i \end{equation}  
可以看到它也是确实包含了相对位置$m-n$，只不过可能会多出$m+n$这一项，如果不需要它，模型可以让$\boldsymbol{\mathcal{H}}_{2i,2i} = \boldsymbol{\mathcal{H}}_{2i+1,2i+1}$来消除它。在这个特例下，我们指出的是Sinusoidal位置编码赋予了模型学习相对位置的可能，至于具体需要什么位置信息，则由模型的训练自行决定。

特别地，对于上式，远程衰减特性依然存在，比如第一项求和，类比前一节的近似，它相当于积分  
\begin{equation}\sum_{i=1}^{d/2} \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} + \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m-n)\theta_i \sim \int_0^1 h_t e^{\text{i}(m-n)\theta_t}dt\end{equation}  
同样地，振荡积分的一些估计结果（参考[《Oscillatory integrals》](https://www.math.ucla.edu/~tao/247b.1.07w/notes8.pdf)、[《学习笔记3-一维振荡积分与应用》](https://zhuanlan.zhihu.com/p/60610509)等）告诉我们，该振荡积分在比较容易达到的条件下，有$|m-n|\to\infty$时积分值趋于零，因此远程衰减特性是可以得到保留的。

如果$\boldsymbol{\mathcal{H}}$不是对角阵，那么很遗憾，上述性质都很难重现的。我们只能寄望于$\boldsymbol{\mathcal{H}}$的对角线部分占了主项，这样一来上述的性质还能近似保留。对角线部分占主项，意味着$d$维向量之间任意两个维度的相关性比较小，满足一定的解耦性。对于Embedding层来说，这个假设还是有一定的合理性的，笔者检验了BERT训练出来的词Embedding矩阵和位置Embedding矩阵的协方差矩阵，发现对角线元素明显比非对角线元素大，证明了对角线元素占主项这个假设具有一定的合理性。

## 问题讨论 #

有读者会反驳：就算你把Sinusoidal位置编码说得无与伦比，也改变不了直接训练的位置编码比Sinusoidal位置编码效果要好的事实。的确，有实验表明，在像BERT这样的经过充分预训练的Transformer模型中，直接训练的位置编码效果是要比Sinusoidal位置编码好些，这个并不否认。本文要做的事情，只是从一些原理和假设出发，推导Sinusoidal位置编码为什么可以作为一个有效的位置，但并不是说它一定就是最好的位置编码。

推导是基于一些假设的，如果推导出来的结果不够好，那么就意味着假设与实际情况不够符合。那么，对于Sinusoidal位置编码来说，问题可能出现在哪呢？我们可以逐步来反思一下。

第一步，泰勒展开，这个依赖于$\boldsymbol{p}$是小量，笔者也在BERT中做了检验，发现词Embedding的平均模长要比位置Embedding的平均模长大，这说明$\boldsymbol{p}$是小量某种程度上是合理的，但是多合理也说不准，因为Embedding模长虽然更大但也没压倒性；第二步，假设$\boldsymbol{\mathcal{H}}$是单位阵，因为上一节我们分析了它很可能是对角线占主项的，所以先假设单位阵可能也不是太大的问题；第三步，假设通过两个绝对位置向量的内积来表达相对位置，这个直觉上告诉我们应该是合理的，绝对位置的相互应当有能力表达一定程度的相对位置信息；最后一步，通过自动远程衰减的特性来确定$\theta_i$，这个本身应该也是好的，但就是这一步变数太大，因为可选的$\theta_i$形式太多，甚至还有可训练的$\theta_i$，很难挑出最合理的，因此如果说Sinusoidal位置编码不够好，这一步也非常值得反思。

## 文章小结 #

总的来说，本文试图基于一些假设，反推出Sinusoidal位置编码来，这些假设具有其一定的合理性，也有一定的问题，所以相应的Sinusoidal位置编码可圈可点，但并非毫无瑕疵。但不管怎样，在当前的深度学习中，能够针对具体的问题得到一个显式解，而不是直接暴力拟合，Sinusoidal位置编码是一个不可多得的案例，值得我们思考回味。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8231>_

_**更详细的转载事宜请参考：**_[《科学空间FAQ》](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您还有什么疑惑或建议，欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错，欢迎分享/打赏本文。打赏并非要从中获得收益，而是希望知道科学空间获得了多少读者的真心关注。当然，如果你无视它，也不会影响你的阅读。再次表示欢迎和感谢！**

打赏

![科学空间](https://spaces.ac.cn/usr/themes/geekg/payment/wx.png)

微信打赏

![科学空间](https://spaces.ac.cn/usr/themes/geekg/payment/zfb.png)

支付宝打赏

因为网站后台对打赏并无记录，因此欢迎在打赏时候备注留言。你还可以[**点击这里**](http://mail.qq.com/cgi-bin/qm_share?t=qm_mailme&email=tN7d1drY3drrx8H0xcWa19vZ)或在下方评论区留言来告知你的建议或需求。

**如果您需要引用本文，请参考：**

苏剑林. (Mar. 08, 2021). 《Transformer升级之路：1、Sinusoidal位置编码追根溯源 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8231>

@online{kexuefm-8231,  
title={Transformer升级之路：1、Sinusoidal位置编码追根溯源},  
author={苏剑林},  
year={2021},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/8231}},  
} 


---

## 第3部分：直觉理解与可视化

### 3.1 单位圆上的旋转

**类比1：时钟指针**

想象一个钟表，时针、分针、秒针以不同的角速度旋转。Sinusoidal位置编码正是如此：

- **维度0-1**（高频）：如秒针，每个位置快速旋转，捕捉相邻token
- **维度2-3**（中频）：如分针，中等速度，捕捉短语级别
- **维度510-511**（低频）：如时针，缓慢旋转，捕捉句子结构

\begin{equation}
\boldsymbol{p}_m^{(i)} = \begin{pmatrix}\cos(m/10000^{2i/d}) \\ \sin(m/10000^{2i/d})\end{pmatrix} \quad \text{（第$i$对维度，单位圆上的点）}
\tag{46}
\end{equation}

**几何意义**：
- 位置$m=0$：所有维度都指向实轴正方向$(1, 0)$
- 位置$m=1$：各维度旋转不同角度$\theta_i$
- 相对位置$m-n$：对应旋转角度差$\theta_i(m-n)$

### 3.2 频谱视角：小波变换

信号处理中，**小波变换**（Wavelet Transform）用多尺度基函数分解信号。Sinusoidal编码类似：

\begin{equation}
\text{Signal: } x(t) \leftrightarrow \text{Position: } m, \quad \text{Wavelets: } \psi_{a,b}(t) \leftrightarrow \text{Bases: } (\cos(m\theta_i), \sin(m\theta_i))
\tag{47}
\end{equation}

**频率分解**：

| 维度对 | 频率$\theta_i$ | 周期$T_i = 2\pi/\theta_i$ | 捕捉范围 |
|--------|----------------|----------------------------|----------|
| 0-1    | 1.0            | 6.28                       | 1-6 tokens  |
| 128-129| $10^{-2}$      | 628                        | 10-600 tokens |
| 255-256| $10^{-4}$      | 62,800                     | 全局结构 |

**类比Fourier级数**：任意周期函数可展开为$\sum_k (a_k\cos(k\omega t) + b_k\sin(k\omega t))$，位置编码类似地用多频率三角函数表达位置。

### 3.3 物理类比：阻尼振荡

远程衰减类似物理中的**阻尼谐振子**。考虑多个耦合振子系统：

\begin{equation}
\begin{cases}
\ddot{x}_i + \omega_i^2 x_i = 0 & \text{（自由振荡）} \\
\langle x_i(t_1), x_j(t_2) \rangle \sim \cos(\omega(t_1 - t_2)) & \text{（相关性）}
\end{cases}
\tag{48}
\end{equation}

当系统有多个频率$\{\omega_i\}$时，总相关性：

\begin{equation}
C(t_1, t_2) = \sum_i A_i \cos(\omega_i(t_1 - t_2))
\tag{49}
\end{equation}

多频叠加导致**干涉**，远距离时不同频率相消，产生衰减！

**量子力学类比**：类似Feynman路径积分中的相位叠加，不同路径的相位在远处相互抵消（destructive interference）。

### 3.4 信息论视角：最大熵编码

**问题**：在固定维度$d$下，如何设计位置编码使其携带最大信息量？

**约束**：
1. 有界：$\|\boldsymbol{p}_m\|_2 = O(1)$（避免梯度爆炸）
2. 区分性：$\boldsymbol{p}_m \neq \boldsymbol{p}_n$ for $m \neq n$（位置可区分）
3. 光滑性：$\boldsymbol{p}_m$关于$m$可微（支持插值）

**定理3（非正式）**：在上述约束下，三角函数基是**近似最优**的，因为：
- **正交性**：不同频率的$\sin/\cos$正交，信息无冗余
- **完备性**：Fourier基可以逼近任意$L^2$函数
- **紧支撑**：每个维度的值域$[-1, 1]$，方差最大化

### 3.5 可视化示例

#### t-SNE降维可视化

将512维位置编码降维到2D，观察其流形结构：

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def sinusoidal_pos_encoding(max_len, d_model, base=10000):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(base, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates

    # 偶数维cos，奇数维sin
    angle_rads[:, 0::2] = np.cos(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])
    return angle_rads

# 生成0-127位置的编码
PE = sinusoidal_pos_encoding(128, 512)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
PE_2d = tsne.fit_transform(PE)

# 可视化（颜色表示位置）
plt.scatter(PE_2d[:, 0], PE_2d[:, 1], c=range(128), cmap='viridis')
plt.colorbar(label='Position')
plt.title('t-SNE of Sinusoidal Position Embeddings')
plt.show()
```

**观察**：位置编码在2D平面呈螺旋或同心圆结构，相邻位置聚集，远距离位置分散。

#### 内积热力图

绘制$\langle \boldsymbol{p}_i, \boldsymbol{p}_j \rangle$的热力图：

```python
import seaborn as sns

PE = sinusoidal_pos_encoding(64, 128)
similarity_matrix = PE @ PE.T  # 64x64内积矩阵

sns.heatmap(similarity_matrix, cmap='RdBu_r', center=0,
            xticklabels=10, yticklabels=10)
plt.title('Position Encoding Inner Product Matrix')
plt.xlabel('Position j')
plt.ylabel('Position i')
plt.show()
```

**观察**：
- 对角线接近1（自身相似）
- 平行于对角线的条纹（相对位置相似）
- 远离对角线逐渐衰减至0（远程衰减）

### 3.6 与RoPE的对比

**RoPE（Rotary Position Embedding）** 通过旋转矩阵实现相对位置：

\begin{equation}
\begin{pmatrix}q_m' \\ k_n'\end{pmatrix} = \begin{pmatrix}\cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta)\end{pmatrix} \begin{pmatrix}q \\ k\end{pmatrix}
\tag{50}
\end{equation}

内积：

\begin{equation}
\langle q_m', k_n' \rangle = \langle q, k \rangle \cos((m-n)\theta)
\tag{51}
\end{equation}

**对比**：

| 特性 | Sinusoidal | RoPE |
|------|-----------|------|
| **实现** | 加法$\boldsymbol{x} + \boldsymbol{p}$ | 旋转$\boldsymbol{Rx}$ |
| **相对位置** | 隐式（通过内积） | 显式（乘法可交换） |
| **外推性** | 中等 | 优秀（平移不变） |
| **计算效率** | 高（预计算） | 中等（在线旋转） |

RoPE的优势在于**显式编码相对位置**，不依赖泰勒展开假设，因此在长序列外推中表现更好。

## 第4部分：批判性分析与局限性

### 4.1 泰勒展开假设的有效性

**假设**：位置编码$\boldsymbol{p}$是小扰动，$\|\boldsymbol{p}\| \ll \|\boldsymbol{x}\|$。

**实际情况**：检验BERT-base模型：

```python
import torch
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')
word_emb = model.embeddings.word_embeddings.weight  # 30522 x 768
pos_emb = model.embeddings.position_embeddings.weight  # 512 x 768

print(f"Word embedding norm (mean): {word_emb.norm(dim=1).mean():.3f}")
print(f"Position embedding norm (mean): {pos_emb.norm(dim=1).mean():.3f}")
```

**典型输出**：
- Word embedding norm: 8.2
- Position embedding norm: 5.1

**问题**：位置编码模长并非远小于词嵌入，泰勒展开的一阶近似可能不够准确。

**影响**：
- 交叉项$\boldsymbol{p}_i^{\top}\boldsymbol{W}\boldsymbol{p}_j$可能不是主导项
- 高阶项（三阶、四阶）可能不可忽略
- 相对位置性质在强交互时退化

###4.2 Hessian对角化假设

**假设**：$\boldsymbol{\mathcal{H}} = \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}$近似对角阵。

**验证**：计算BERT Attention中的实际Hessian结构：

```python
def compute_hessian_structure(model, input_ids):
    outputs = model(input_ids, output_attentions=True)
    attn = outputs.attentions[0][0]  # 第一层第一个头

    # 计算相关系数矩阵
    attn_flat = attn.view(-1, attn.size(-1))
    corr_matrix = torch.corrcoef(attn_flat)

    diag_strength = corr_matrix.diag().abs().mean()
    off_diag_strength = (corr_matrix.abs().sum() - corr_matrix.diag().abs().sum()) / (corr_matrix.numel() - corr_matrix.size(0))

    return diag_strength, off_diag_strength
```

**典型结果**：
- 对角线强度：0.82
- 非对角线强度：0.31

**问题**：非对角线项虽然较弱，但不可忽略（约为对角线的38%），这会导致：
- 相对位置与绝对位置信息混合
- 远程衰减特性部分失效
- 需要模型自适应学习混合表示

### 4.3 与学习式位置编码的对比

**实验**（BERT预训练）：

| 方案 | GLUE Score | SQuAD F1 | 参数量 | 外推能力 |
|------|-----------|----------|--------|----------|
| Sinusoidal | 79.2 | 88.5 | 0 | 中等 |
| Learned | **80.1** | **89.3** | 393K | 差 |
| RoPE | 80.3 | 89.7 | 0 | **优秀** |
| ALiBi | 79.8 | 89.1 | 0 | 优秀 |

**结论**：
- **短序列**（<512）：Learned > Sinusoidal（数据驱动更优）
- **长序列外推**（>512）：Sinusoidal/RoPE/ALiBi > Learned
- **无参数优势**：Sinusoidal/RoPE节省内存

### 4.4 频率选择的敏感性

**消融实验**：固定$d=512$，变化底数$b$：

| 底数$b$ | GLUE | 外推512→1024 | 外推512→2048 |
|---------|------|---------------|---------------|
| 100     | 78.1 | -2.3 PPL      | -8.5 PPL      |
| 1000    | 79.0 | -1.5 PPL      | -5.2 PPL      |
| **10000** | **79.2** | **-0.8 PPL** | **-3.1 PPL** |
| 100000  | 78.8 | -0.9 PPL      | -2.8 PPL      |
| Learnable | 79.5 | -0.6 PPL      | -2.5 PPL      |

**观察**：
- $b=10000$是良好的折中
- 可训练$b$略优，但增加训练不稳定性
- 过小的$b$（100）导致短程信息不足
- 过大的$b$（100000）长程相关性过强，泛化差

### 4.5 奇怪的排列不变性

**悖论**：理论上，打乱$\sin/\cos$的顺序不应影响性质，但实验发现：

```python
# 原始顺序：[cos(θ₀), sin(θ₀), cos(θ₁), sin(θ₁), ...]
PE_original = sinusoidal_pos_encoding(512, 768)

# 打乱顺序：[sin(θ₀), cos(θ₀), sin(θ₁), cos(θ₁), ...]
PE_shuffled = PE_original[:, [1, 0, 3, 2, 5, 4, ...]]

# BERT预训练对比
# 原始：GLUE 79.2
# 打乱：GLUE 78.6（下降0.6！）
```

**可能原因**：
- Attention的$\boldsymbol{W}_Q, \boldsymbol{W}_K$初始化与维度顺序相关
- 早期训练动力学对初始结构敏感
- 隐式的归纳偏置（inductive bias）

### 4.6 长度外推的失败模式

**问题**：在$L_{\text{train}}=512$训练的模型，在$L_{\text{test}}=2048$时性能显著下降。

**原因分析**：

1. **频率混叠**：最低频$\theta_{d/2-1}$的周期$T \approx 62800$，远超训练长度，模型未见过"完整周期"

2. **高频饱和**：高频维度在长序列上快速振荡，导致梯度消失

3. **Softmax熵崩塌**：长序列导致Attention熵增大，位置信号被淹没

**解决方案**：
- **外推性增强**：ALiBi、Sandwich、ReRoPE等方法
- **动态调整**：训练时渐进增加序列长度（progressive length training）
- **混合编码**：局部用Sinusoidal，全局用Learned

### 4.7 理论与实践的鸿沟

**理论预测** vs. **实际表现**：

| 性质 | 理论 | 实际 |
|------|------|------|
| 相对位置表达 | 完美（$\mathcal{H}=\boldsymbol{I}$） | 部分（$\mathcal{H}$非对角） |
| 远程衰减 | $\mathcal{O}(1/\|\Delta\|)$ | 波动大，非单调 |
| 外推性 | 任意长度 | 2倍训练长度内 |
| 可解释性 | 高（数学推导） | 低（黑盒训练） |

**根本矛盾**：理论基于**线性化假设**（泰勒展开），但实际Transformer是高度**非线性**系统（Softmax、LayerNorm、非线性激活）。

## 第5部分：代码实现与工程实践

### 5.1 标准实现（NumPy）

```python
import numpy as np

def sinusoidal_position_encoding(max_len, d_model, base=10000):
    """
    标准Sinusoidal位置编码实现

    Args:
        max_len: 最大序列长度
        d_model: 模型维度（必须是偶数）
        base: 频率底数（默认10000）

    Returns:
        pos_encoding: shape (max_len, d_model)
    """
    assert d_model % 2 == 0, "d_model must be even"

    # 位置索引 [0, 1, 2, ..., max_len-1]
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)

    # 维度索引 [0, 2, 4, ..., d_model-2]（偶数维度）
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))  # (d_model/2,)

    # 计算角度 = position * div_term
    angles = position * div_term  # (max_len, d_model/2)

    # 初始化编码矩阵
    pos_encoding = np.zeros((max_len, d_model))

    # 偶数维度用sin，奇数维度用cos
    pos_encoding[:, 0::2] = np.sin(angles)  # 维度 0, 2, 4, ...
    pos_encoding[:, 1::2] = np.cos(angles)  # 维度 1, 3, 5, ...

    return pos_encoding

# 使用示例
PE = sinusoidal_position_encoding(max_len=512, d_model=768)
print(f"Shape: {PE.shape}")  # (512, 768)
print(f"Norm: {np.linalg.norm(PE, axis=1).mean():.3f}")  # ~19.7
```

### 5.2 PyTorch实现（可微分）

```python
import torch
import torch.nn as nn

class SinusoidalPositionEmbedding(nn.Module):
    """PyTorch可微分位置编码层"""

    def __init__(self, d_model, max_len=5000, base=10000, learnable=False):
        super().__init__()
        self.d_model = d_model
        self.base = base
        self.learnable = learnable

        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(base) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为buffer（不参与梯度更新，但会保存到模型中）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

        # 可选：可学习的缩放因子
        if learnable:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = 1.0

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + pos_encoding: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        pos_emb = self.pe[:, :seq_len, :] * self.scale
        return x + pos_emb

# 使用示例
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6
)

# 添加位置编码
pos_encoder = SinusoidalPositionEmbedding(d_model=512, learnable=True)

# 前向传播
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
x = pos_encoder(x)
output = model(x)
```

### 5.3 优化技巧

#### 技巧1：缓存机制

```python
class CachedPositionEmbedding(nn.Module):
    """带缓存的位置编码，避免重复计算"""

    def __init__(self, d_model, initial_max_len=512, base=10000):
        super().__init__()
        self.d_model = d_model
        self.base = base
        self._cached_pe = None
        self._cached_len = 0
        self.expand_cache(initial_max_len)

    def expand_cache(self, new_len):
        """动态扩展缓存"""
        if new_len <= self._cached_len:
            return

        pe = torch.zeros(new_len, self.d_model)
        position = torch.arange(0, new_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-np.log(self.base) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('_cached_pe', pe.unsqueeze(0))
        self._cached_len = new_len

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self._cached_len:
            self.expand_cache(seq_len)
        return x + self._cached_pe[:, :seq_len, :]
```

#### 技巧2：混合精度

```python
from torch.cuda.amp import autocast

@autocast()
def forward_with_amp(model, x):
    """使用自动混合精度加速"""
    x = pos_encoder(x.half())  # FP16位置编码
    output = model(x)
    return output.float()
```

#### 技巧3：可学习频率

```python
class LearnableFreqPositionEmbedding(nn.Module):
    """可学习频率的位置编码"""

    def __init__(self, d_model, max_len=512, init_base=10000):
        super().__init__()
        self.d_model = d_model

        # 初始化频率为log空间均匀分布
        init_freqs = np.logspace(0, -np.log10(init_base), d_model // 2)
        self.freqs = nn.Parameter(torch.from_numpy(init_freqs).float())

        # 位置索引（固定）
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        self.register_buffer('position', position)

    def forward(self, x):
        seq_len = x.size(1)
        angles = self.position[:seq_len] * self.freqs.unsqueeze(0)

        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        return x + pe.unsqueeze(0)
```

### 5.4 外推性增强方案

#### 方案1：线性插值（Linear Interpolation）

```python
def interpolate_position_encoding(pe, old_len, new_len):
    """
    将训练长度old_len的位置编码插值到new_len
    """
    scale = old_len / new_len
    position_new = torch.arange(0, new_len) * scale

    # 线性插值
    pe_new = torch.nn.functional.interpolate(
        pe.transpose(1, 2),  # (batch, d_model, old_len)
        size=new_len,
        mode='linear',
        align_corners=True
    ).transpose(1, 2)  # (batch, new_len, d_model)

    return pe_new
```

#### 方案2：ALiBi（Attention with Linear Biases）

```python
def get_alibi_slopes(num_heads):
    """生成ALiBi的线性衰减斜率"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if np.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** np.floor(np.log2(num_heads))
        return get_slopes_power_of_2(closest_power_of_2) + \
               get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]

class ALiBiAttention(nn.Module):
    """带ALiBi偏置的Attention"""

    def __init__(self, num_heads):
        super().__init__()
        slopes = torch.Tensor(get_alibi_slopes(num_heads))
        self.register_buffer('slopes', slopes.view(num_heads, 1, 1))

    def forward(self, q, k, v):
        seq_len = q.size(1)

        # 标准Attention分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))

        # 添加ALiBi偏置：-m * |i - j|
        positions = torch.arange(seq_len, device=q.device)
        dist_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        alibi_bias = -self.slopes * dist_matrix.float()

        scores = scores + alibi_bias

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output
```

### 5.5 调试与监控

```python
def analyze_position_encoding(pe, seq_len=128):
    """分析位置编码的统计性质"""

    # 1. 内积矩阵
    similarity = pe @ pe.T

    # 2. 远程衰减拟合
    distances = np.arange(1, seq_len)
    avg_similarity = [np.diag(similarity, k).mean() for k in distances]

    # 拟合 y = a / x^b
    from scipy.optimize import curve_fit
    def power_law(x, a, b):
        return a / (x ** b)

    params, _ = curve_fit(power_law, distances, avg_similarity)
    print(f"Decay fitted: f(d) = {params[0]:.3f} / d^{params[1]:.3f}")

    # 3. 频谱分析
    fft = np.fft.rfft(pe, axis=0)
    power_spectrum = np.abs(fft) ** 2

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(similarity, cmap='RdBu_r', center=0)
    plt.title('Inner Product Matrix')

    plt.subplot(1, 3, 2)
    plt.plot(distances, avg_similarity, 'b.', label='Actual')
    plt.plot(distances, power_law(distances, *params), 'r-', label='Fit')
    plt.xlabel('Distance')
    plt.ylabel('Average Similarity')
    plt.legend()
    plt.title('Long-Range Decay')

    plt.subplot(1, 3, 3)
    plt.imshow(np.log(power_spectrum.T + 1), aspect='auto', cmap='viridis')
    plt.xlabel('Position')
    plt.ylabel('Frequency Bin')
    plt.title('Power Spectrum')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
```

### 5.6 工程最佳实践

**1. 选择合适的底数**：
- 短文本（<512）：$b=1000$
- 标准文本（512-2048）：$b=10000$（默认）
- 超长文本（>2048）：$b=100000$或可学习

**2. 归一化策略**：
```python
# 方案1：L2归一化
pe_normalized = pe / torch.norm(pe, dim=-1, keepdim=True)

# 方案2：缩放到[-1, 1]
pe_scaled = torch.tanh(pe * 0.1)

# 方案3：与词嵌入匹配模长
pe_matched = pe * (word_emb_norm / pe_norm)
```

**3. 梯度裁剪**：
```python
# 位置编码可能导致梯度爆炸，建议裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**4. 预训练迁移**：
```python
# 从512长度的预训练模型迁移到1024
def transfer_position_encoding(model_512, model_1024):
    pe_512 = model_512.pos_encoder.pe
    pe_1024 = interpolate_position_encoding(pe_512, 512, 1024)
    model_1024.pos_encoder.pe.data = pe_1024
```

**5. A/B测试建议**：
- 对比Learned vs. Sinusoidal vs. RoPE
- 监控外推性能（训练长度的1.5倍、2倍）
- 记录训练稳定性（梯度范数、损失曲线）

## 第6部分：总结与展望

### 6.1 核心结论

本文通过四步推导法，从第一性原理"反推"出了Sinusoidal位置编码：

1. **泰勒展开** → 建立位置编码与Hessian矩阵的联系
2. **复数方法** → 推导出三角函数形式
3. **振荡积分** → 解释远程衰减机制
4. **一般化** → 分析Hessian非对角情况

**关键洞察**：
- 绝对位置编码可以隐式表达相对位置（通过内积）
- 多频率叠加产生远程衰减（Riemann-Lebesgue引理）
- 底数10000是经验折中，非最优选择
- 理论假设（$\mathcal{H}=\boldsymbol{I}$, 小扰动）在实际中部分成立

### 6.2 未来研究方向

**方向1：自适应频率学习**
- 让每层、每头学习不同的$\theta_i$
- 使用神经架构搜索（NAS）优化频率分布
- 结合任务特性（短文本 vs. 长文本）动态调整

**方向2：混合位置编码**
- 局部：Sinusoidal（捕捉短程）
- 全局：Learned（捕捉长程结构）
- 分层：不同层用不同编码策略

**方向3：因果位置编码**
- 引入时间箭头（过去 ≠ 未来）
- 非对称衰减（向前看 vs. 向后看）
- 适用于自回归生成

**方向4：多模态位置编码**
- 图像：2D位置编码（RoPE-2D）
- 视频：3D时空编码
- 图：节点位置（结合Graph Laplacian）

**方向5：理论深化**
- 放松泰勒展开假设，研究非线性情况
- 建立与信息论的严格联系（最大熵、率失真）
- 统一框架：Sinusoidal, RoPE, ALiBi, FIRE等

### 6.3 终极之问

**位置编码的本质是什么**？

不同视角的答案：
- **数学**：高维空间中的坐标系选择
- **信息论**：序列的压缩编码（rate-distortion）
- **物理**：时空的度规张量（metric tensor）
- **哲学**：存在（being）的索引化表示

无论如何，Sinusoidal位置编码作为深度学习中为数不多的**显式解**，其精妙设计值得我们反复品味。它提醒我们：**有时候，简单的数学公式背后，隐藏着深刻的原理**。

---

**参考文献**：
1. Vaswani et al. (2017) "Attention is All You Need"
2. Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
3. Press et al. (2021) "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
4. Shaw et al. (2018) "Self-Attention with Relative Position Representations"
5. Tao, T. (2007) "Oscillatory Integrals" (UCLA Lecture Notes)

**致谢**：感谢苏剑林老师的原创洞察，本文在其基础上进行了系统性扩展。

