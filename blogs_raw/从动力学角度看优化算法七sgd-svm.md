---
title: 从动力学角度看优化算法（七）：SGD ≈ SVM
slug: 从动力学角度看优化算法七sgd-svm
date: 2021-04-15
source: https://spaces.ac.cn/archives/8314
tags: 动力学, 优化, SVM, 泛化, 生成模型
status: completed
tags_reviewed: true
---

# 从动力学角度看优化算法（七）：SGD ≈ SVM

**原文链接**: [https://spaces.ac.cn/archives/8314](https://spaces.ac.cn/archives/8314)

---

## 1. 核心理论、公理与历史基础

### 1.1 跨学科根源：从硬间隔到动力学收敛

深度学习的一个核心谜题是：**为什么过参数化的模型（参数量远大于样本量）能够泛化？**
按照传统的 VC 维理论，参数越多，模型复杂度越高，越容易过拟合。但在实践中，我们即使不加任何显式的正则化项（如 L2 Regularization），SGD 训练出的神经网络依然具有极佳的泛化能力。

这个谜题的答案隐藏在两个看似无关领域的交叉点上：
- **统计学习理论 (Statistical Learning Theory)**：1990s Vapnik 提出的支持向量机（SVM）理论认为，分类器的泛化误差界与“分类间隔（Margin）”成反比。最大化间隔是提升泛化能力的最强手段。
- **非线性动力学 (Nonlinear Dynamics)**：Soudry 等人在 2018 年的开创性工作证明，在使用 Logistic Loss（或 Cross Entropy）训练线性分类器时，梯度下降的动力学轨迹会自动收敛到最大间隔方向。

这意味着：**SGD 这一优化算法本身，就内嵌了一个隐式的 SVM 求解器。**

### 1.2 历史编年史：从硬间隔到隐式偏置

1.  **1960s - 感知机算法**：Rosenblatt 提出了感知机，但它只能找到任意一个解，没有任何关于“最优解”的保证。
2.  **1995 - SVM 的黄金时代**：Vapnik 提出了最大间隔分类器。为了求解它，我们需要引入拉格朗日对偶变量，并解决一个复杂的二次规划（QP）问题。
3.  **2017 - 泛化之谜的爆发**：Zhang 等人在 ICLR 发表《Understanding deep learning requires rethinking generalization》，指出深度网络可以轻易记住随机标签，这彻底击碎了基于模型容量的传统泛化理论。
4.  **2018 - 隐式偏置 (Implicit Bias) 的发现**：Soudry 等人在 *The Implicit Bias of Gradient Descent on Separable Data* 中证明，在数据线性可分的情况下，SGD 训练出的权重方向 $\boldsymbol{w}/\|\boldsymbol{w}\|$ 会渐近收敛到 SVM 的解。
5.  **2020s - 深度网络的推广**：Gunasekar、Lyu 和 Li 等人将这一结论推广到了深度线性网络和对角线性网络，甚至发现了在矩阵分解任务中 SGD 会寻找最小核范数（Nuclear Norm）解。

### 1.3 严谨公理化：隐式优化的数学基石

为了严格证明 SGD ≈ SVM，我们需要以下公理体系：

<div class="theorem-box">

### 核心公理 1：线性可分性 (Linear Separability)
假设数据集 $\mathcal{D} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$ 是线性可分的。即存在至少一个单位向量 $\boldsymbol{u}^*$，使得对于所有 $i$：
\begin{equation} y_i \langle \boldsymbol{u}^*, \boldsymbol{x}_i \rangle \geq \gamma > 0 \tag{1} \end{equation}
这保证了损失函数可以被优化到 0，且参数模长会趋于无穷大。

### 核心公理 2：指数尾部损失 (Exponential Tail Loss)
损失函数 $\ell(z)$ 必须具有指数衰减的尾部特性，例如 Logistic Loss $\ell(z) = \log(1+e^{-z})$ 或指数损失 $e^{-z}$。
\begin{equation} \lim_{z \to \infty} \ell(z) e^{\alpha z} = C \tag{2} \end{equation}
这意味着当分类置信度极高时，梯度虽然微小，但永远不会消失。正是这微弱的“尾部梯度”，在漫长的训练岁月中塑造了参数的方向。

### 核心公理 3：梯度流近似 (Gradient Flow Approximation)
我们主要研究连续时间的梯度流动力学：
\begin{equation} \dot{\boldsymbol{\theta}}(t) = -\nabla L(\boldsymbol{\theta}(t)) \tag{3} \end{equation}
虽然离散的 SGD 会引入噪声，但大量研究表明，只要学习率足够小，离散轨迹的渐近行为与连续流是一致的。

</div>

### 1.4 设计哲学：无为而治的正则化

SGD ≈ SVM 的设计哲学是：**“不需要显式的约束，演化本身就是一种选择。”**
传统的 SVM 需要我们手动写下 $\min \|\boldsymbol{w}\|^2$ 的目标函数。而 SGD 告诉我们，你不需要告诉模型去“最小化范数”或“最大化间隔”，你只需要给它一个指数型的损失函数，然后让它一直跑下去。
随着参数模长 $\|\boldsymbol{\theta}\| \to \infty$，损失函数的几何性质会自动“挤压”参数的方向，迫使它对齐到那个最宽、最稳健的分类界面上。这是一种**“通过过程定义结果”**的深刻哲学。

---

## 2. 严谨的核心数学推导

本节将通过详尽的极限分析，从动力学方程推导出 KKT 条件的涌现。这是一场从微积分到凸优化的华丽冒险。

### 2.1 问题的动力学建模

设二分类数据集为 $\{(\boldsymbol{x}_i, y_i)\}_{i=1}^n$。为了简化符号，我们将 $y_i$ 吸收进 $\boldsymbol{x}_i$ 中，即假设对于所有样本，目标是 $\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle > 0$。
损失函数为 Logistic Loss：
\begin{equation} L(\boldsymbol{\theta}) = \sum_{i=1}^n \ell(\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle) = \sum_{i=1}^n \log(1 + e^{-\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle}) \tag{4} \end{equation}

<div class="derivation-box">

### 推导 7.1：梯度的渐近展开与模长发散

**步骤 1：计算梯度表达式**
对 $\boldsymbol{\theta}$ 求导：
\begin{equation} \nabla L(\boldsymbol{\theta}) = \sum_{i=1}^n \ell'(\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle) \boldsymbol{x}_i = \sum_{i=1}^n \frac{-e^{-\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle}}{1 + e^{-\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle}} \boldsymbol{x}_i \tag{5} \end{equation}

**步骤 2：考虑训练后期的渐近行为**
由于数据可分，随着训练进行，$L \to 0$，这意味着对于所有 $i$，$\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle \to +\infty$。
在 $z \to \infty$ 时，分母 $1 + e^{-z} \to 1$。因此梯度可以近似为：
\begin{equation} \nabla L(\boldsymbol{\theta}) \approx -\sum_{i=1}^n e^{-\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle} \boldsymbol{x}_i \tag{6} \end{equation}

**步骤 3：参数模长的增长规律**
考虑模长平方 $\|\boldsymbol{\theta}\|^2$ 的变化率：
\begin{equation} \frac{d}{dt} \|\boldsymbol{\theta}\|^2 = 2 \langle \boldsymbol{\theta}, \dot{\boldsymbol{\theta}} \rangle = -2 \langle \boldsymbol{\theta}, \nabla L \rangle \tag{7} \end{equation}
代入 (6) 的近似：
\begin{equation} \frac{d}{dt} \|\boldsymbol{\theta}\|^2 \approx 2 \sum_{i=1}^n \langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle e^{-\langle \boldsymbol{\theta}, \boldsymbol{x}_i \rangle} \tag{8} \end{equation}
由于函数 $f(u) = u e^{-u}$ 在 $u \to \infty$ 时趋于 0，这意味着模长的增长速度 $\frac{d}{dt} \|\boldsymbol{\theta}\|^2$ 会随时间衰减。
严格的推导表明，$\|\boldsymbol{\theta}(t)\| \sim \ln t$。这是一个极慢的增长速度，但它确实发散。

</div>

### 2.2 核心引理：支持向量的主导地位

当模长趋于无穷时，虽然所有样本的梯度贡献都在减小，但它们减小的**速度**截然不同。

<div class="derivation-box">

### 推导 7.2：谁主导了方向？

**步骤 1：分解模长与方向**
令 $\boldsymbol{\theta}(t) = \rho(t) \boldsymbol{u}(t)$，其中 $\rho(t) = \|\boldsymbol{\theta}(t)\|$ 是模长，$\boldsymbol{u}(t)$ 是单位方向向量。
我们将关注 $\boldsymbol{u}(t)$ 在 $t \to \infty$ 时的极限。

**步骤 2：识别“最慢衰减项”**
观察梯度近似公式 (6)：$-\nabla L \approx \sum_{i=1}^n e^{-\rho \langle \boldsymbol{u}, \boldsymbol{x}_i \rangle} \boldsymbol{x}_i$。
定义第 $i$ 个样本的几何间隔为 $\gamma_i = \langle \boldsymbol{u}, \boldsymbol{x}_i \rangle$。
梯度的方向由指数项 $e^{-\rho \gamma_i}$ 决定。
显然，**拥有最小间隔 $\gamma_{\min}$ 的样本（即支持向量）**，其指数衰减最慢，在梯度和中占比最大。

**步骤 3：极限方向的线性组合**
令 $\mathcal{S}$ 为具有最小间隔的样本集合（支持向量集）。当 $\rho \to \infty$ 时：
\begin{equation} -\nabla L(\boldsymbol{\theta}) \propto \sum_{i \in \mathcal{S}} e^{-\rho \gamma_{\min}} \boldsymbol{x}_i + \sum_{j \notin \mathcal{S}} o(e^{-\rho \gamma_{\min}}) \boldsymbol{x}_j \tag{9} \end{equation}
这意味着，梯度的方向渐渐收敛到支持向量的非负线性组合：
\begin{equation} \boldsymbol{d}_{\infty} = \lim_{t \to \infty} \frac{-\nabla L}{\|-\nabla L\|} \in \text{Cone}(\lbrace\boldsymbol{x}_i\rbrace_{i \in \mathcal{S}}) \tag{10} \end{equation}

</div>

### 2.3 对偶原理：SVM 的 KKT 条件再现

为什么梯度方向的极限一定是 SVM 的解？让我们看看 SVM 的对偶问题。

<div class="formula-explanation">

### SVM 对偶性与梯度流的数学同构

**1. 硬间隔 SVM 的原问题**：
\begin{equation} \min_{\boldsymbol{w}} \|\boldsymbol{w}\|^2 \quad \text{s.t.} \quad \langle \boldsymbol{w}, \boldsymbol{x}_i \rangle \geq 1, \forall i \tag{11} \end{equation}

**2. KKT 条件**：
最优解 $\boldsymbol{w}^*$ 必须满足：
- **平稳性**：$\boldsymbol{w}^* = \sum_{i=1}^n \alpha_i \boldsymbol{x}_i$，其中 $\alpha_i \geq 0$。
- **互补松弛**：$\alpha_i (\langle \boldsymbol{w}^*, \boldsymbol{x}_i \rangle - 1) = 0$。即只有支持向量（$\langle \boldsymbol{w}^*, \boldsymbol{x}_i \rangle = 1$）的 $\alpha_i > 0$。

**3. 梯度流的隐式 KKT**：
回到 (9) 式，我们发现梯度流的方向 $\boldsymbol{u}(t)$ 正是被支持向量集合 $\mathcal{S}$ 的线性组合所驱动。
在动力学的平衡态，方向向量 $\boldsymbol{u}$ 的变化率为 0，这意味着 $\boldsymbol{u}$ 必须与累积梯度的方向一致。
Soudry (2018) 严格证明了：
\begin{equation} \lim_{t \to \infty} \frac{\boldsymbol{\theta}(t)}{\|\boldsymbol{\theta}(t)\|} = \frac{\boldsymbol{w}_{\text{SVM}}}{\|\boldsymbol{w}_{\text{SVM}}\|} \tag{12} \end{equation}
这就是著名的 **Directional Convergence Theorem**。

</div>

### 2.4 方向动力学方程的推导

我们可以直接写出方向向量 $\boldsymbol{u}(t)$ 的微分方程，从而更直观地看到它如何寻找最大间隔。

<div class="derivation-box">

### 推导 7.3：方向演化的微分方程

**步骤 1：利用商法则求导**
\begin{equation} \dot{\boldsymbol{u}} = \frac{d}{dt} \left( \frac{\boldsymbol{\theta}}{\|\boldsymbol{\theta}\|} \right) = \frac{\dot{\boldsymbol{\theta}} \|\boldsymbol{\theta}\| - \boldsymbol{\theta} \frac{d}{dt}\|\boldsymbol{\theta}\|}{\|\boldsymbol{\theta}\|^2} \tag{13} \end{equation}
注意 $\frac{d}{dt}\|\boldsymbol{\theta}\| = \langle \boldsymbol{u}, \dot{\boldsymbol{\theta}} \rangle$。

**步骤 2：代入梯度流 $\dot{\boldsymbol{\theta}} = -\nabla L$**
\begin{equation} \dot{\boldsymbol{u}} = \frac{1}{\rho} \left( -\nabla L - \boldsymbol{u} \langle \boldsymbol{u}, -\nabla L \rangle \right) \tag{14} \end{equation}
这可以简写为梯度的投影形式：
\begin{equation} \dot{\boldsymbol{u}} = -\frac{1}{\rho} (I - \boldsymbol{u}\boldsymbol{u}^T) \nabla L \tag{15} \end{equation}
其中 $I - \boldsymbol{u}\boldsymbol{u}^T$ 是向切平面（垂直于 $\boldsymbol{u}$ 的平面）的投影矩阵。

**步骤 3：物理意义**
方程 (15) 说明，方向 $\boldsymbol{u}$ 的变化由**损失函数的梯度在切向上的分量**驱动。
当 $\boldsymbol{u}$ 调整到使得所有支持向量的梯度合成力垂直于球面时（即无法再在球面上找到下降方向），$\dot{\boldsymbol{u}}$ 变为 0，此时即达到最大间隔解。

</div>

### 2.5 渐近收敛速率的精确分析

不仅要证明收敛，还要知道收敛有多慢。

<div class="step-by-step">

<div class="step">
**损失衰减速率**：
已知 $\rho(t) \sim \ln t$。由于 $L \approx e^{-\rho \gamma}$，所以 $L(t) \approx 1/t$。
</div>

<div class="step">
**间隔收敛速率**：
设最优间隔为 $\gamma^*$，当前间隔为 $\gamma(t)$。
Nacson (2019) 证明了：
\begin{equation} \gamma^* - \gamma(t) \leq \mathcal{O}\left( \frac{1}{\ln t} \right) \tag{16} \end{equation}
</div>

<div class="step">
**令人绝望的慢**：
$1/\ln t$ 的收敛速度意味着：如果你想让间隔误差减半，你需要训练的时间是原来的平方倍（例如从 $10^4$ 步增加到 $10^8$ 步）！
这就是为什么深度学习模型虽然能学到最大间隔，但需要极长的训练时间来“精修”边界。
</div>

</div>
