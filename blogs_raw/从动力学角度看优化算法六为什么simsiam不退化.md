---
title: 从动力学角度看优化算法（六）：为什么SimSiam不退化？
slug: 从动力学角度看优化算法六为什么simsiam不退化
date: 2020-12-11
source: https://spaces.ac.cn/archives/7980
tags: 动力学, 优化, 无监督, 生成模型, attention
status: completed
tags_reviewed: true
---

# 从动力学角度看优化算法（六）：为什么SimSiam不退化？

**原文链接**: [https://spaces.ac.cn/archives/7980](https://spaces.ac.cn/archives/7980)

---

## 1. 核心理论、公理与历史基础

### 1.1 跨学科根源：从负采样到对称性破缺

自监督学习（Self-Supervised Learning）的终极幽灵是**“表征坍缩（Representation Collapse）”**：如果没有显式的排斥力，模型会发现最简单的办法是让所有图片的特征向量都变成同一个常数（如全零），此时损失函数虽然最小，但表征彻底失效。

*   **对比学习 (Contrastive Learning)**：如 SimCLR，引入海量的负样本作为“排斥力”。
*   **非对比学习 (Non-contrastive Learning)**：BYOL 和 SimSiam 挑战了这一常识。它们证明了：即便没有负样本，模型依然可以不坍缩。
*   **动力系统视角**：SimSiam 的成功本质上是优化路径中的**对称性破缺**。通过人为制造快慢不一的演化模块，系统在滑向平凡解（坍缩）的过程中被截断了。

### 1.2 历史编年史

1.  **2009 - 深度架构先验**：随机初始化的 CNN 就已经具备了捕捉边缘和纹理的初步能力。
2.  **2017 - Deep Image Prior**：证明了网络结构本身就是一种强力的正则化，对自然图像有偏好。
3.  **2020 - BYOL**：Google 提出使用动量编码器（EMA）实现非对比学习，引发了关于“为什么不坍缩”的大辩论。
4.  **2021 - SimSiam 的极致简化**：何恺明团队证明了即便没有动量编码器，只要有 **Stop-gradient**，同样有效。
5.  **2022-2024 - 动力学解释的完善**：研究者开始意识到，优化算法（梯度下降）本身就是自监督成功的关键变量。

### 1.3 严谨公理化

<div class="theorem-box">

### 核心公理体系：SimSiam 不坍缩三要素

**公理 1 (一致性约束)**：正样本对 $T_1(x), T_2(x)$ 的表示必须尽可能重合。
**公理 2 (Predictor 引入)**：支路间必须存在一个非线性的预测器 $h$，打破恒等映射。
**公理 3 (停止梯度算子)**：梯度的流动必须是不对称的。
\begin{equation} \nabla_{\theta} \| h_{\boldsymbol{\varphi}}(z_1) - \text{stop\_grad}(z_2) \|^2 \tag{1} \end{equation}

</div>

### 1.4 设计哲学：快与慢的博弈

SimSiam 的设计哲学是：**“跑得比坍缩快。”** 
坍缩是一个长期的、结构性的趋势。如果模型中的某个部分（Predictor）能够以极快的速度完成对目标（Encoder 输出）的局部拟合，那么推动 Encoder 整体坍缩的梯度压力就会迅速消散。这就像是在流沙沉没之前，我们先在表面铺好了一层轻质甲板。

---

## 2. 严谨的核心数学推导

本节将通过动力学方程组，定量揭示 Stop-gradient 如何拦截坍缩过程。

### 2.1 建立 Siamese 动力学模型

设编码器参数为 $\boldsymbol{\theta}$，预测器参数为 $\boldsymbol{\varphi}$。损失函数为：
\begin{equation}
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\varphi}) = \mathbb{E}_{x, \mathcal{T}_1, \mathcal{T}_2} \left[ \| h_{\boldsymbol{\varphi}}(f_{\boldsymbol{	heta}}(\mathcal{T}_1(x))) - f_{\boldsymbol{	heta}}(\mathcal{T}_2(x)) \|^2 \right] \tag{2}
\end{equation}

<div class="derivation-box">

### 推导：有无 Stop-gradient 的梯度流对比

**情形 A：无 Stop-gradient（对称更新）**
参数 $\boldsymbol{\theta}$ 的演化速度取决于两边的梯度：
\begin{equation}
\dot{\boldsymbol{\theta}} = -\left( \underbrace{\frac{\partial \mathcal{L}}{\partial f_1} \frac{\partial f_1}{\partial \boldsymbol{\theta}}}_{\text{支路1}} + \underbrace{\frac{\partial \mathcal{L}}{\partial f_2} \frac{\partial f_2}{\partial \boldsymbol{\theta}}}_{\text{支路2}} \right) \tag{3}
\end{equation}
由于两边方向一致，$\boldsymbol{\theta}$ 会获得双倍的动力冲向常数解。

**情形 B：有 Stop-gradient (SimSiam)**
支路 2 的梯度被切断，动力学变为：
\begin{equation}
\dot{\boldsymbol{\theta}} = -\frac{\partial \mathcal{L}}{\partial f_1} \frac{\partial f_1}{\partial \boldsymbol{\theta}} \tag{4}
\end{equation}
同时，预测器 $\boldsymbol{\varphi}$ 的演化为：
\begin{equation}
\dot{\boldsymbol{\varphi}} = -\frac{\partial \mathcal{L}}{\partial h} \frac{\partial h}{\partial \boldsymbol{\varphi}} \tag{5}
\end{equation}

</div>

### 2.2 玩具模型分析：标量演化模拟

为了看清本质，我们假设 $f_{\theta}(x) = \theta x$（线性编码），$h_{\varphi}(z) = \varphi z$（线性预测）。

<div class="derivation-box">

### 推导：坍缩速度的定量计算

设目标是最小化 $\frac{1}{2}(\varphi \theta - \theta)^2$。

**没有 Stop-grad 时**：
\begin{equation}
\dot{\theta} = -(\varphi \theta - \theta) \varphi = -\theta \varphi (\varphi - 1) \tag{6}
\end{equation}
如果初始时 $\varphi$ 还没学好（例如 $\varphi < 1$），那么 $\dot{\theta}$ 会让 $\theta \to 0$。一旦 $\theta=0$，特征全失，无法挽回。

**有 Stop-grad 时**：
由于 Predictor $\varphi$ 位于输出层，其学习路径更短，**动力学极快**。
\begin{equation}
\dot{\boldsymbol{\varphi}} = -(\varphi \theta - \theta) \theta = -\theta^2 (\varphi - 1) \tag{7}
\end{equation}
由于 $\dot{\boldsymbol{\varphi}}$ 的收敛常数是 $\theta^2$（通常大于零且较稳定），$\varphi$ 会以指数级速度 $e^{-\theta^2 t}$ 趋向于 1。
**关键点**：当 $\varphi$ 迅速到达 1 时，(6) 式中的动力 $(\varphi - 1)$ 变为 0。
这意味着：**Encoder 还没来得及滑到 0，驱动它滑动的力就已经被 Predictor 抵消了。**

</div>

### 2.3 深度展开分析：隐式方差补偿

如果将 SimSiam 看作一个 EM 算法（Expectation-Maximization），我们可以得到更有趣的发现。

<div class="formula-explanation">

### 损失函数的一阶泰勒展开

假设数据增强 $\mathcal{T}(x) = x + \Delta x$，其中 $\Delta x$ 是小扰动。

<div class="formula-step">
<div class="step-label">1. 目标中心化</div>
对于目标项 $f_{\theta}(\mathcal{T}_2(x))$，其平均值为 $\bar{z} = f_{\theta}(\bar{x})$。
</div>

<div class="formula-step">
<div class="step-label">2. 展开预测误差</div>
\begin{equation}
\mathcal{L}(\theta) \approx \mathbb{E}_{x, \Delta x} \left[ \left\Vert \boldsymbol{J}_{\theta}(x) \Delta x \right\|^2 \right] \tag{8}
\end{equation}
其中 $\boldsymbol{J}_{\theta}$ 是编码器的雅可比矩阵（特征灵敏度）。
</div>

<div class="formula-step">
<div class="step-label">3. 几何意义</div>
SimSiam 实际上在寻找一个特征映射，使得它对常见的图像变换（数据增强）具有低敏感度，同时通过 Predictor 的解耦效应，在不牺牲表示维度（即不坍缩）的前提下实现这一点。
</div>

</div>

---

## 3. 数学直觉、几何视角与多维类比

<div class="intuition-box">

### 🧠 直觉理解：影子球与快速捕捉手 🎾

想象你在和一个影子（Predictor）玩抛接球。

1.  **坍缩（全梯度）**：你和影子都在拼命往地板（零点）缩。因为你们动作一致，最后你们都会变成地板上的一个点。
2.  **SimSiam 不坍缩**：
    *   你（Encoder）动得很慢。
    *   影子（Predictor）是一个身手极快的捕捉手。
    *   **Stop-gradient**：你抛球时，影子必须停下来接，不能反过来拽你。
    *   **结果**：每当你稍微偏离一点方向，影子由于动作极快，会在你还没动下一脚之前就站在了球的落点上。既然影子已经接到了球（Loss 变小），你就没有动力继续往地板缩了。你停在了半路，保住了你的位置（特征）。

</div>

### 3.2 几何视角：能量盆地的脊线驻留

在特征空间中，坍缩是一个深不见底的中心黑洞。
- **对比学习**：是在黑洞周围修了一圈挡板（负样本）。
- **SimSiam**：是利用动力学在黑洞边缘建立了一个“动态平衡轨道”。通过切断梯度，我们将原本垂直落入黑洞的力，转化为了在轨道上切向运动的力。这种现象在非线性物理中被称为**“吸引子的拓扑改变”**。

---

## 4. 方法论变体、批判性比较与优化

### 4.1 全量对比表

| 模型 | 防坍缩机制 | 核心组件 | **致命缺陷** |
| :--- | :--- | :--- | :--- |
| **SimCLR** | 负样本对齐 | 大 Batch Size | ❌ 计算开销极大 |
| **BYOL** | 动量预测 | EMA 编码器 | ❌ 理论证明复杂 |
| **SimSiam** | **动力学解耦** | **Stop-grad + Predictor** | ❌ **对 BN 极度依赖** |
| **VICReg** | 协方差约束 | Variance Regularization | ❌ 参数调优困难 |

### 4.2 深度批判：SimSiam 的“伪科学”陷阱

虽然实验结果惊艳，但 SimSiam 的理论基础存在三个脆弱点：

1.  **致命缺陷 1：Batch Normalization (BN) 的隐式对比**
    *   **分析**：如果去掉 BN，SimSiam 会瞬间坍缩。
    *   **真相**：BN 在 Batch 维度上的均值和方差计算，实际上提供了一种隐式的“负样本”效应，强迫同一个 Batch 内的特征不能全等。**SimSiam 的成功有一半是属于 BN 的。**
2.  **致命缺陷 2：Predictor 的架构黑箱**
    *   **问题**：Predictor 如果太深，收敛极慢；如果太浅，无法打破对称性。
    *   **局限**：目前没有数学公式能计算出针对特定主干网络的最优 Predictor 深度。
3.  **致命缺陷 3：特征冗余 (Redundancy)**
    *   由于没有去相关的显式约束，SimSiam 学到的 2048 维特征中，可能只有极少数维度是有信息的，其余维度高度相关。

### 4.3 优化演进

*   **Barlow Twins**：通过让互协方差矩阵逼近单位阵，从数学上彻底消除了坍缩的可能性，不再依赖动力学巧合。
*   **DINO**：将 SimSiam 的思想应用到 Transformer 中，利用中心化（Centering）和锐化（Sharpening）替代 BN，实现了更高质量的无监督学习。

---

## 5. 工程实践、路线图与未来展望

### 5.1 炼丹师 Checkpoint：PyTorch 实现核心代码

```python
# SimSiam 的精髓在于这两行逻辑
def simsiam_loss(p1, z2, p2, z1):
    # p1, p2 是 Predictor 的输出
    # z1, z2 是 Encoder 的输出
    
    # 核心：D(p1, z2) 和 D(p2, z1) 
    # z 支路必须调用 .detach()，这对应公式 (1) 中的 stop_gradient
    loss = D(p1, z2.detach()) * 0.5 + D(p2, z1.detach()) * 0.5
    return loss
```

### 5.2 未来研究子问题

#### **方向 1：大模型（LLM）中的自监督坍缩**
- **背景**：Next-token prediction 本质上是带标签的，但隐藏层的表征是否会发生局部坍缩？
- **子问题**：能否借鉴 SimSiam 的 Predictor 结构来增加大模型表征的熵？

#### **方向 2：无需 BN 的动力学解耦**
- **挑战点**：寻找一种替代 BN 的算子，使其在保持 SimSiam 简洁性的同时，能够适配各种网络架构。

#### **方向 3：SimSiam 在扩散模型中的应用**
- **愿景**：利用快慢动力学，在扩散模型的去噪过程中隐式地学习更高阶的语义特征，而非单纯的像素重建。

---

**总结**：SimSiam 告诉我们，**对称性是优化的动力，而打破对称性则是进化的契机。** 停止梯度不仅仅是一个编程技巧，它是我们在无监督的荒野中，利用数学张力搭建起的一座防止模型滑向虚无的桥梁。