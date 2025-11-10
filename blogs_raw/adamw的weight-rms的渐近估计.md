---
title: AdamW的Weight RMS的渐近估计
slug: adamw的weight-rms的渐近估计
date: 2025-10-01
source: https://spaces.ac.cn/archives/11307
tags: 详细推导, 优化
status: pending
---
# AdamW的Weight RMS的渐近估计

**原文链接**: [https://spaces.ac.cn/archives/11307](https://spaces.ac.cn/archives/11307)

**发布日期**: 2025-10-01

---

在[《为什么Adam的Update RMS是0.2？》](https://kexue.fm/archives/11267)中，我们用平均场近似估计了Adam的Update RMS。不久后，读者 [@EIFY](https://x.com/EIFY/status/1965888629814988984) 指出相同的结果已经出现在论文[《Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks》](https://arxiv.org/abs/2305.17212)中。阅读后，笔者发现其中不仅包含了Update RMS的估计，还包含了Weight RMS的估计。

也就是说，AdamW训出来的模型，其权重的RMS是可以事先估计出来一个渐近结果的。大家会不会觉得这个结论有点意外？反正笔者第一次看到它是颇为意外的，直觉上权重模长是模型根据训练集自己学出来的，结果它告诉我这已经隐藏在优化器的超参中，可谓很反直觉了。

这篇文章我们还是用平均场近似方法，来复现对Weight RMS的渐近估计。

[[...]](https://spaces.ac.cn/archives/11307 "AdamW的Weight RMS的渐近估计")


---

## 公式推导与注释

### 1. AdamW优化器的完整数学定义

AdamW是Adam优化器的变体，引入了解耦的权重衰减（Decoupled Weight Decay）。其更新规则如下：

给定参数$\theta_t$，在第$t$步的更新过程为：

$$
\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
\end{aligned}
$$

其中：
- $\alpha$是学习率（learning rate）
- $\beta_1, \beta_2$是动量参数，通常取$\beta_1=0.9, \beta_2=0.999$
- $\epsilon$是数值稳定项，通常取$10^{-8}$
- $\lambda$是权重衰减系数（weight decay coefficient）
- $g_t$是梯度，$m_t$是一阶矩估计，$v_t$是二阶矩估计

**关键区别**：与Adam不同，AdamW的权重衰减项$\lambda \theta_{t-1}$是在自适应学习率之外单独应用的，而不是作为梯度的一部分加入$L_2$正则。

### 2. Weight RMS的定义与意义

权重的均方根（Root Mean Square, RMS）定义为：

$$
\text{RMS}(\theta) = \sqrt{\frac{1}{d}\sum_{i=1}^d \theta_i^2} = \sqrt{\mathbb{E}[\theta^2]}
$$

其中$d$是参数维度。在深度学习中，Weight RMS反映了：

1. **模型容量的使用程度**：更大的RMS通常意味着模型使用了更多的表达能力
2. **正则化强度的体现**：权重衰减会限制RMS的增长
3. **优化动态的表征**：RMS随训练的演化反映了优化过程

**核心问题**：给定AdamW的超参数$(\alpha, \beta_1, \beta_2, \lambda)$，能否预测训练收敛后权重的渐近RMS？

### 3. 平均场近似的基本假设

平均场近似（Mean Field Approximation）是统计物理中的经典方法，用于处理高维系统。在优化器分析中，我们做如下假设：

**假设1（统计独立性）**：不同参数分量在统计意义上独立且同分布（i.i.d.）

**假设2（梯度统计）**：梯度$g_t$可以分解为
$$
g_t = \mu_g + \sigma_g \xi_t
$$
其中$\mu_g$是均值（通常在收敛时趋于0），$\sigma_g$是标准差，$\xi_t \sim \mathcal{N}(0, 1)$是标准正态噪声。

**假设3（稳态假设）**：在$t \to \infty$时，系统达到统计稳态，各量的统计分布不再随时间变化。

这些假设允许我们从单个参数的演化推导整体行为。

### 4. 渐近分析框架

在$t \to \infty$时，偏置修正因子趋于1：
$$
1 - \beta_1^t \to 1, \quad 1 - \beta_2^t \to 1
$$

因此$\hat{m}_t \to m_t$，$\hat{v}_t \to v_t$。更新方程简化为：

$$
\theta_t = \theta_{t-1} - \alpha \left(\frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

重新整理：
$$
\theta_t = (1 - \alpha\lambda)\theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

定义**有效衰减率**：
$$
\gamma = 1 - \alpha\lambda
$$

则：
$$
\theta_t = \gamma \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

这是一个包含递归和随机噪声的动力学方程。

### 5. 一阶矩$m_t$的渐近行为

一阶矩的递推：
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

这是一个指数移动平均（EMA）。其稳态解满足：

假设$g_t$是平稳随机过程，均值为$\mu_g$，则：
$$
\mathbb{E}[m_\infty] = \frac{1-\beta_1}{1-\beta_1} \mu_g = \mu_g
$$

在收敛附近（接近局部极小值），梯度均值$\mu_g \approx 0$，因此：
$$
\mathbb{E}[m_\infty] \approx 0
$$

但$m_t$仍有波动。计算方差：
$$
\text{Var}(m_t) = \beta_1^2 \text{Var}(m_{t-1}) + (1-\beta_1)^2 \sigma_g^2
$$

稳态时$\text{Var}(m_\infty) = \text{Var}(m_{\infty-1})$，解得：
$$
\text{Var}(m_\infty) = \frac{(1-\beta_1)^2}{1-\beta_1^2} \sigma_g^2 = \frac{1-\beta_1}{1+\beta_1} \sigma_g^2
$$

对于$\beta_1 = 0.9$：
$$
\text{Var}(m_\infty) = \frac{0.1}{1.9} \sigma_g^2 \approx 0.053 \sigma_g^2
$$

### 6. 二阶矩$v_t$的渐近行为

二阶矩的递推：
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

假设$\mathbb{E}[g_t^2] = \mu_g^2 + \sigma_g^2 \approx \sigma_g^2$（因为$\mu_g \approx 0$），稳态时：
$$
\mathbb{E}[v_\infty] = \sigma_g^2
$$

这是一个关键结果：**二阶矩估计收敛到梯度的方差**。

进一步，我们需要$v_\infty$本身的波动。由于$g_t^2$的方差与四阶矩有关，假设梯度服从高斯分布，则：
$$
\mathbb{E}[g_t^4] = 3\sigma_g^4
$$

可以证明：
$$
\text{Var}(v_\infty) = \frac{1-\beta_2}{1+\beta_2} \cdot 2\sigma_g^4
$$

但在实际应用中，由于$\beta_2 = 0.999$非常接近1，$v_t$的波动相对较小，我们可以近似：
$$
v_t \approx \mathbb{E}[v_\infty] = \sigma_g^2
$$

### 7. 权重动力学的简化

将上述结果代入权重更新方程：
$$
\theta_t = \gamma \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

忽略$\epsilon$（假设$v_t \gg \epsilon^2$）：
$$
\theta_t \approx \gamma \theta_{t-1} - \alpha \frac{m_t}{\sqrt{\sigma_g^2}} = \gamma \theta_{t-1} - \frac{\alpha}{\sigma_g} m_t
$$

定义标准化的一阶矩：
$$
\tilde{m}_t = \frac{m_t}{\sigma_g}
$$

则：
$$
\theta_t = \gamma \theta_{t-1} - \alpha \tilde{m}_t
$$

这是一个线性随机差分方程。

### 8. 稳态方程的推导

在稳态下，$\theta_t$的分布不随时间变化。对方程两边取平方并求期望：

$$
\mathbb{E}[\theta_\infty^2] = \gamma^2 \mathbb{E}[\theta_{\infty-1}^2] + \alpha^2 \mathbb{E}[\tilde{m}_\infty^2] - 2\alpha\gamma \mathbb{E}[\theta_{\infty-1} \tilde{m}_\infty]
$$

**关键假设（平均场解耦）**：假设$\theta_{t-1}$与$m_t$在统计上独立（这是平均场近似的核心），则：
$$
\mathbb{E}[\theta_{\infty-1} \tilde{m}_\infty] = \mathbb{E}[\theta_{\infty-1}] \mathbb{E}[\tilde{m}_\infty] \approx 0
$$

因为$\mathbb{E}[\tilde{m}_\infty] = 0$。

因此：
$$
\mathbb{E}[\theta_\infty^2] = \gamma^2 \mathbb{E}[\theta_\infty^2] + \alpha^2 \mathbb{E}[\tilde{m}_\infty^2]
$$

利用$\text{Var}(\tilde{m}_\infty) = \frac{1-\beta_1}{1+\beta_1}$（从第5节，除以$\sigma_g^2$）：
$$
\mathbb{E}[\tilde{m}_\infty^2] = \frac{1-\beta_1}{1+\beta_1}
$$

代入稳态方程：
$$
\mathbb{E}[\theta_\infty^2] = \gamma^2 \mathbb{E}[\theta_\infty^2] + \alpha^2 \frac{1-\beta_1}{1+\beta_1}
$$

解得：
$$
\mathbb{E}[\theta_\infty^2] = \frac{\alpha^2}{1-\gamma^2} \cdot \frac{1-\beta_1}{1+\beta_1}
$$

### 9. 权重衰减的影响

回忆$\gamma = 1 - \alpha\lambda$，则：
$$
1 - \gamma^2 = 1 - (1-\alpha\lambda)^2 = 2\alpha\lambda - \alpha^2\lambda^2 = \alpha\lambda(2 - \alpha\lambda)
$$

当$\alpha\lambda \ll 1$（通常情况），近似：
$$
1 - \gamma^2 \approx 2\alpha\lambda
$$

代入：
$$
\mathbb{E}[\theta_\infty^2] \approx \frac{\alpha^2}{2\alpha\lambda} \cdot \frac{1-\beta_1}{1+\beta_1} = \frac{\alpha}{2\lambda} \cdot \frac{1-\beta_1}{1+\beta_1}
$$

### 10. Weight RMS的渐近公式

权重的RMS定义为$\sqrt{\mathbb{E}[\theta_\infty^2]}$，因此：

$$
\boxed{\text{RMS}(\theta_\infty) = \sqrt{\frac{\alpha}{2\lambda} \cdot \frac{1-\beta_1}{1+\beta_1}}}
$$

这是**核心结果**！注意以下几点：

1. **与学习率的关系**：$\text{RMS} \propto \sqrt{\alpha}$，学习率越大，权重RMS越大
2. **与权重衰减的关系**：$\text{RMS} \propto \frac{1}{\sqrt{\lambda}}$，权重衰减越强，RMS越小
3. **与梯度统计无关**：惊人的是，$\sigma_g$消失了！RMS只依赖于优化器超参
4. **与二阶矩参数$\beta_2$无关**：只与一阶矩参数$\beta_1$有关

### 11. 数值验证

对于典型的AdamW超参数：
- $\alpha = 10^{-3}$
- $\lambda = 0.01$
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$

计算：
$$
\frac{1-\beta_1}{1+\beta_1} = \frac{0.1}{1.9} \approx 0.0526
$$

$$
\text{RMS} = \sqrt{\frac{10^{-3}}{2 \times 0.01} \times 0.0526} = \sqrt{\frac{0.001}{0.02} \times 0.0526} = \sqrt{0.05 \times 0.0526} \approx \sqrt{0.00263} \approx 0.051
$$

这意味着典型训练后的权重RMS约为**0.05**量级。

### 12. 不同学习率-权重衰减比例的分析

定义比例$r = \frac{\alpha}{\lambda}$，则：
$$
\text{RMS} = \sqrt{\frac{r}{2} \cdot \frac{1-\beta_1}{1+\beta_1}}
$$

这表明：
- 若保持$\frac{\alpha}{\lambda}$恒定，改变$\alpha$和$\lambda$的绝对值不会改变渐近RMS
- 例如：$(\alpha=0.001, \lambda=0.01)$与$(\alpha=0.01, \lambda=0.1)$会产生相同的RMS

这与实验观察一致：**学习率和权重衰减的比例决定了模型的最终尺度**。

### 13. 与Adam（无权重衰减）的对比

对于标准Adam（$\lambda=0$），上述推导中$\gamma = 1$，稳态方程变为：
$$
\mathbb{E}[\theta_\infty^2] = \mathbb{E}[\theta_\infty^2] + \alpha^2 \frac{1-\beta_1}{1+\beta_1}
$$

这个方程无解（除非$\alpha=0$），意味着：

**Adam不存在稳态！** 权重会持续增长（或减小），没有均衡点。这是为什么实践中Adam需要其他正则化手段（如梯度裁剪、层归一化）的原因。

只有引入权重衰减（AdamW），系统才能达到动态平衡：
- 权重衰减提供**收缩力**，使权重趋向0
- 梯度更新提供**扩张力**，使权重偏离0
- 平衡点就是我们推导的RMS

### 14. 更精确的分析：考虑$\epsilon$的影响

之前我们忽略了$\epsilon$，现在考虑其影响。更新方程：
$$
\theta_t = \gamma \theta_{t-1} - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}
$$

定义有效的自适应学习率：
$$
\alpha_{\text{eff}} = \frac{\alpha}{\sqrt{v_t} + \epsilon} \approx \frac{\alpha}{\sqrt{\sigma_g^2} + \epsilon}
$$

当$\sigma_g \gg \epsilon$时，$\alpha_{\text{eff}} \approx \frac{\alpha}{\sigma_g}$，与之前分析一致。

当$\sigma_g \ll \epsilon$（接近完美收敛），$\alpha_{\text{eff}} \approx \frac{\alpha}{\epsilon}$，此时更新步长由$\epsilon$控制，防止数值不稳定。

在实践中，$\epsilon = 10^{-8}$非常小，对大多数层$\sigma_g \gg \epsilon$，因此我们的近似是合理的。

### 15. 层级依赖的RMS

不同层的梯度统计$\sigma_g$可能不同。但由于最终公式：
$$
\text{RMS} = \sqrt{\frac{\alpha}{2\lambda} \cdot \frac{1-\beta_1}{1+\beta_1}}
$$

**与$\sigma_g$无关**，这意味着：

**所有层的权重RMS渐近到相同的值！**（前提是使用相同的$\alpha, \lambda, \beta_1$）

这是一个强有力的结论。在实践中：
- 不同层可能有不同的初始尺度
- 但经过足够长的AdamW训练，它们会收敛到相同的RMS
- 这称为**旋转平衡**（Rotational Equilibrium），即论文标题所指

### 16. 权重初始化的影响

考虑初始条件$\theta_0$。从递推关系：
$$
\theta_t = \gamma^t \theta_0 + \sum_{k=1}^t \gamma^{t-k} \left(-\alpha \frac{m_k}{\sqrt{v_k}}\right)
$$

第一项$\gamma^t \theta_0$以指数速度衰减（因为$\gamma < 1$）：
$$
\gamma^t = (1-\alpha\lambda)^t \approx e^{-\alpha\lambda t}
$$

**遗忘时间尺度**：
$$
\tau_{\text{forget}} = \frac{1}{\alpha\lambda}
$$

对于$\alpha=0.001, \lambda=0.01$：
$$
\tau_{\text{forget}} = \frac{1}{10^{-5}} = 10^5 \text{ 步}
$$

在约$10^5$步后，初始化的影响基本消失，权重完全由优化器超参决定。

### 17. 多参数组的扩展

在实践中，不同参数组可能有不同的$\alpha, \lambda$（如学习率缩放、层级权重衰减）。

对于第$i$组参数，其RMS为：
$$
\text{RMS}_i = \sqrt{\frac{\alpha_i}{2\lambda_i} \cdot \frac{1-\beta_1}{1+\beta_1}}
$$

这允许**显式控制不同层的尺度**：
- 若希望某层权重更大，增大$\frac{\alpha_i}{\lambda_i}$
- 若希望某层权重更小，减小$\frac{\alpha_i}{\lambda_i}$

例如，对于输出层，可以设置更小的$\frac{\alpha}{\lambda}$以获得更小的权重，有助于稳定训练。

### 18. 训练动态的时间演化

虽然我们关注渐近值，但训练过程中RMS如何演化？

从差分方程的解：
$$
\mathbb{E}[\theta_t^2] = \gamma^{2t} \mathbb{E}[\theta_0^2] + \frac{1-\gamma^{2t}}{1-\gamma^2} \alpha^2 \frac{1-\beta_1}{1+\beta_1}
$$

随着$t \to \infty$，$\gamma^{2t} \to 0$，第一项消失。RMS的时间演化：
$$
\text{RMS}(t) = \sqrt{\gamma^{2t} \text{RMS}_0^2 + (1-\gamma^{2t}) \text{RMS}_\infty^2}
$$

这是从初始RMS到渐近RMS的指数衰减/增长：
- 若$\text{RMS}_0 > \text{RMS}_\infty$：权重逐渐收缩
- 若$\text{RMS}_0 < \text{RMS}_\infty$：权重逐渐扩张

时间常数为$\tau_{\text{eq}} = \frac{1}{2\alpha\lambda}$。

### 19. 实验验证策略

理论预测可以通过以下实验验证：

**实验1：改变$\alpha/\lambda$比例**
- 固定$\beta_1=0.9$，尝试不同的$(\alpha, \lambda)$组合
- 测量收敛后的权重RMS
- 验证$\text{RMS} \propto \sqrt{\alpha/\lambda}$

**实验2：改变$\beta_1$**
- 固定$\alpha, \lambda$，改变$\beta_1$（如0.8, 0.9, 0.95）
- 验证$\text{RMS} \propto \sqrt{\frac{1-\beta_1}{1+\beta_1}}$

**实验3：不同初始化**
- 使用不同的权重初始化（如Xavier, He）
- 验证经过足够步数后RMS收敛到相同值

**实验4：层级一致性**
- 在深度网络中，测量每层的RMS
- 验证所有层收敛到相同的RMS（使用相同超参时）

### 20. 理论的适用范围和局限

**适用条件**：
1. 训练接近收敛（梯度均值$\approx 0$）
2. 参数维度足够高（平均场近似有效）
3. 梯度噪声不退化（$\sigma_g$不趋于0）
4. $\alpha\lambda \ll 1$（小步长近似）

**局限性**：
1. **训练早期**：理论预测稳态，不适用于训练初期的瞬态行为
2. **梯度相关性**：实际中不同参数的梯度可能相关，平均场假设失效
3. **非高斯梯度**：若梯度分布严重非高斯，方差分析可能不准确
4. **学习率调度**：理论假设固定超参，学习率衰减会改变结果
5. **稀疏梯度**：对于嵌入层等稀疏更新，需要更精细的建模

### 21. 与其他正则化的交互

AdamW的权重衰减可以视为$L_2$正则化的一种形式。考虑其他正则化：

**Dropout**：引入随机性，增大梯度方差$\sigma_g$，但不改变公式（因为RMS与$\sigma_g$无关）

**Batch Normalization**：归一化激活，间接影响梯度统计，可能改变$\sigma_g$的层级分布

**Gradient Clipping**：限制梯度范数，相当于改变梯度分布的尾部，可能影响$\mathbb{E}[m_t^2]$

**Layer-wise Learning Rate**：不同层的$\alpha_i$不同，导致层级RMS差异

综合正则化的影响需要更复杂的分析。

### 22. 深度学习理论的启示

这个分析揭示了几个深刻的理论问题：

**1. 优化器的隐式偏好**：优化器不仅决定"如何收敛"，还决定"收敛到哪里"。AdamW隐式地选择了特定RMS的解。

**2. 超参数的双重作用**：
   - $\alpha$：控制收敛速度 + 控制权重尺度
   - $\lambda$：控制正则化强度 + 控制权重尺度

**3. 尺度不变性的破缺**：虽然损失函数可能对权重尺度不敏感（如ReLU网络），AdamW打破了这种对称性，选择了特定尺度。

**4. 训练与泛化的联系**：权重RMS与泛化性能相关（更大的权重可能过拟合），理论预测帮助我们理解超参数如何影响泛化。

### 23. 推广到其他优化器

类似的分析可应用于其他优化器：

**SGD with Momentum + Weight Decay**：
$$
\text{RMS} \propto \sqrt{\frac{\alpha}{\lambda}}
$$
（与AdamW类似，但系数不同）

**RMSprop + Weight Decay**：
$$
\text{RMS} \propto \sqrt{\frac{\alpha}{\lambda} \cdot f(\beta_2)}
$$
（依赖于二阶矩参数）

**Lion Optimizer**：需要不同的分析框架，因为其更新规则基于符号而非幅度

这些分析有助于统一理解不同优化器的行为。

### 24. 理论公式的实用建议

基于推导的公式，我们可以提供实用的超参数选择建议：

**选择1：固定目标RMS**
若希望权重RMS约为$R$，则选择：
$$
\lambda = \frac{\alpha}{2R^2} \cdot \frac{1-\beta_1}{1+\beta_1}
$$

例如，目标$R=0.1$，$\alpha=0.001$，$\beta_1=0.9$：
$$
\lambda = \frac{0.001}{2 \times 0.01} \times 0.0526 = 0.00263 \approx 0.003
$$

**选择2：参数量自适应**
对于大模型，可以根据参数量调整$\lambda$，保持RMS恒定。

**选择3：学习率预热与权重衰减**
在预热阶段，$\alpha$增大，为保持RMS稳定，可以同步增大$\lambda$。

### 25. 结论与展望

通过平均场近似，我们推导出AdamW的权重RMS渐近公式：

$$
\boxed{\text{RMS}(\theta_\infty) = \sqrt{\frac{\alpha}{2\lambda} \cdot \frac{1-\beta_1}{1+\beta_1}}}
$$

**关键发现**：
1. RMS完全由优化器超参决定，与梯度统计、初始化无关（渐近地）
2. $\text{RMS} \propto \sqrt{\alpha/\lambda}$，学习率-权重衰减比例是核心
3. 所有层收敛到相同RMS（使用相同超参时）
4. Adam（无权重衰减）不存在稳态，必须引入$\lambda > 0$

**未来方向**：
- 扩展到非平稳设置（学习率调度）
- 考虑梯度相关性的影响
- 分析瞬态动力学（训练早期）
- 与神经网络架构（如Transformer）的具体交互
- 泛化界的理论联系

这一理论框架为理解和设计优化器提供了新的视角，也为超参数调优提供了理论指导。

