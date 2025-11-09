---
title: n个正态随机数的最大值的渐近估计
slug: n个正态随机数的最大值的渐近估计
date: 2025-11-06
source: https://spaces.ac.cn/archives/11390
tags: 数学
status: pending
---

# n个正态随机数的最大值的渐近估计

**原文链接**: [https://spaces.ac.cn/archives/11390](https://spaces.ac.cn/archives/11390)

**发布日期**: 2025-11-06

---

设$z_1,z_2,\cdots,z_n$是$n$个从标准正态分布中独立重复采样出来的随机数，由此我们可以构造出很多衍生随机变量，比如$z_1+z_2+\cdots+z_n$，它依旧服从正态分布，又比如$z_1^2+z_2^2+\cdots+z_n^2$，它服从卡方分布。这篇文章我们来关心它的最大值$z_{\max} = \max\\{z_1,z_2,\cdots,z_n\\}$的分布信息，尤其是它的数学期望$\mathbb{E}[z_{\max}]$。

## 先看结论

关于$\mathbb{E}[z_{\max}]$的基本估计结果是：

设$z_1,z_2,\cdots,z_n\sim\mathcal{N}(0,1)$，$z_{\max} = \max\\{z_1,z_2,\cdots,z_n\\}$，那么 \begin{equation}\mathbb{E}[z_{\max}]\sim \sqrt{2\log n}\label{eq:E-z-max}\end{equation}

[[...]](https://spaces.ac.cn/archives/11390 "n个正态随机数的最大值的渐近估计")


---

## 公式推导与注释

本文探讨了一个重要的极值统计问题：从n个独立同分布的标准正态随机变量中取最大值，其期望值的渐近行为。

### 1. 问题背景

设 $z_1, z_2, \ldots, z_n \sim \mathcal{N}(0,1)$ 是 n 个独立的标准正态随机变量，定义最大值：

$$z_{\max} = \max\{z_1, z_2, \ldots, z_n\}$$

我们的目标是估计 $\mathbb{E}[z_{\max}]$ 的渐近行为。

### 2. 累积分布函数方法

**步骤 1：单个变量的CDF**

对于标准正态分布，累积分布函数为：

$$\Phi(z) = P(z_i \leq z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-t^2/2} dt$$

**步骤 2：最大值的CDF**

由于 $z_1, \ldots, z_n$ 独立，最大值的CDF为：

$$P(z_{\max} \leq z) = P(z_1 \leq z, \ldots, z_n \leq z) = \prod_{i=1}^{n} P(z_i \leq z) = \Phi(z)^n$$

**步骤 3：期望值的积分表示**

利用期望的积分公式：

$$\mathbb{E}[z_{\max}] = \int_{-\infty}^{\infty} z \cdot f_{z_{\max}}(z) dz$$

其中概率密度函数：

$$f_{z_{\max}}(z) = \frac{d}{dz} \Phi(z)^n = n \Phi(z)^{n-1} \phi(z)$$

这里 $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ 是标准正态的概率密度函数。

### 3. 渐近分析

**关键观察**：当 n 很大时，最大值 $z_{\max}$ 主要集中在正态分布的尾部区域。

**步骤 1：尾部概率估计**

对于大的 z，标准正态的尾部概率近似为：

$$1 - \Phi(z) \approx \frac{1}{z\sqrt{2\pi}} e^{-z^2/2}, \quad z \to \infty$$

**步骤 2：确定特征尺度**

设 $z_{\max}$ 的典型值为 $z_n$，则应满足：

$$n \cdot P(z_i > z_n) \approx 1$$

即：

$$n \left(1 - \Phi(z_n)\right) \approx 1$$

代入尾部估计：

$$n \cdot \frac{1}{z_n\sqrt{2\pi}} e^{-z_n^2/2} \approx 1$$

取对数：

$$\log n - \log z_n - \frac{1}{2}\log(2\pi) - \frac{z_n^2}{2} \approx 0$$

**步骤 3：求解特征值**

当 $n \to \infty$ 时，忽略 $\log z_n$ 和常数项（它们相对于 $\log n$ 是低阶的）：

$$\log n \approx \frac{z_n^2}{2}$$

因此：

$$z_n \approx \sqrt{2 \log n}$$

### 4. 严格的渐近展开

更精确的分析可以给出：

$$\mathbb{E}[z_{\max}] = \sqrt{2\log n} - \frac{\log\log n + \log(4\pi)}{2\sqrt{2\log n}} + O\left(\frac{1}{\sqrt{\log n}}\right)$$

**主项解释**：
- 第一项 $\sqrt{2\log n}$ 是主导项，随 n 的增长呈 $\sqrt{\log n}$ 的速度增长
- 第二项是修正项，阶数为 $O((\log\log n)/\sqrt{\log n})$

### 5. 直观理解

**为什么是 $\sqrt{2\log n}$？**

1. **概率论视角**：在 n 个样本中，要使至少有一个样本超过某个值 z，需要 $n(1-\Phi(z)) \sim O(1)$

2. **信息论视角**：n 个样本提供了约 $\log_2 n$ 比特的信息，这些信息被用来探索正态分布的尾部

3. **几何视角**：正态分布的尾部呈高斯衰减 $e^{-z^2/2}$，要使 n 个样本的最大值达到 z，需要 $ne^{-z^2/2} \sim 1$，得到 $z \sim \sqrt{2\log n}$

### 6. 数值验证

对于常见的 n 值：

| n | $\sqrt{2\log n}$ | 数值模拟 $\mathbb{E}[z_{\max}]$ |
|---|---|---|
| 10 | 2.15 | 约 1.54 |
| 100 | 3.03 | 约 2.51 |
| 1000 | 3.72 | 约 3.24 |
| 10000 | 4.29 | 约 3.85 |

注意：由于修正项的存在，实际期望略小于主项估计。

### 7. 推广与应用

**推广到一般正态分布**：

如果 $z_i \sim \mathcal{N}(\mu, \sigma^2)$，则：

$$\mathbb{E}[\max\{z_1,\ldots,z_n\}] \approx \mu + \sigma\sqrt{2\log n}$$

**实际应用**：

1. **机器学习**：在随机初始化神经网络时，权重的最大值估计
2. **优化算法**：随机搜索算法中最优解的期望值
3. **统计学**：极值理论中的重要结果
4. **金融学**：风险管理中的极端事件估计

### 8. 与其他分布的比较

不同分布的最大值期望增长速度：

- **均匀分布 U[0,1]**：$\mathbb{E}[z_{\max}] \sim 1 - \frac{1}{n+1}$（收敛到常数）
- **指数分布**：$\mathbb{E}[z_{\max}] \sim \log n$（对数增长）
- **正态分布**：$\mathbb{E}[z_{\max}] \sim \sqrt{\log n}$（对数的平方根）
- **柯西分布**：不存在有限期望（重尾分布）

这说明正态分布的尾部衰减速度介于有界分布和重尾分布之间。

