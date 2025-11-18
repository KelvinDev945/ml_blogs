# 优化器主题深度Summary

> **涵盖文章**：16篇优化器相关文章
> **主要内容**：SGD、Adam系列、Lion、Tiger、Muon、梯度分析、学习率Scaling

---

## 1. 核心理论、公理与历史基础

### 1.1 历史发展

- **1951 - SGD**：Robbins-Monro随机逼近算法
- **1964 - 动量法**：Polyak提出Heavy-Ball方法
- **1983 - AdaGrad**：自适应学习率的开端
- **2014 - Adam**：Kingma & Ba，结合动量与自适应学习率
- **2017 - AdamW**：Loshchilov & Hutter，解耦权重衰减
- **2023 - Lion**：Google搜索出的sign-based优化器
- **2023 - Tiger**：极致抠门的优化器（仅符号+EMA）
- **2024 - Muon**：矩阵优化器，基于msign算子

### 1.2 核心公理

#### **公理1：梯度下降方向定理**
对于可微函数 $f(\boldsymbol{\theta})$，负梯度方向 $-\nabla f$ 是局部最速下降方向：
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)$$

#### **公理2：随机梯度无偏性**
$$\mathbb{E}_{\mathcal{B}}[\nabla_{\mathcal{B}} f] = \nabla f$$
其中 $\mathcal{B}$ 是mini-batch。

#### **公理3：Polyak-Łojasiewicz (PL)不等式**
若满足 $\|\nabla f(\boldsymbol{\theta})\|^2 \geq 2\mu(f(\boldsymbol{\theta}) - f^*)$，则SGD指数收敛。

---

## 2. 严谨的核心数学推导

### 2.1 Adam优化器推导

**步骤1：一阶矩估计（动量）**
$$\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t$$

**步骤2：二阶矩估计（自适应学习率）**
$$\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^2$$

**步骤3：偏差修正**
$$\hat{\boldsymbol{m}}_t = \frac{\boldsymbol{m}_t}{1-\beta_1^t}, \quad \hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1-\beta_2^t}$$

**步骤4：参数更新**
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}$$

**推导依据**：近似牛顿法，$\boldsymbol{H}^{-1} \approx \text{diag}(1/\sqrt{\boldsymbol{v}})$

### 2.2 AdamW权重衰减推导

**问题**：L2正则化 $\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$ 在Adam中的梯度被自适应项缩放，导致衰减强度不一致。

**解耦方案**：
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} - \eta \lambda \boldsymbol{\theta}_t$$

**关键洞察**：权重衰减直接作用于参数，不经过动量和自适应项。

### 2.3 Muon（矩阵优化器）推导

**核心思想**：将梯度 $\boldsymbol{G} \in \mathbb{R}^{n \times m}$ 视为矩阵，进行正交归一化。

**步骤1：Newton-Raphson谱归一化**
$$\boldsymbol{X}_{t+1} = \frac{3}{2}\boldsymbol{X}_t - \frac{1}{2}\boldsymbol{X}_t (\boldsymbol{X}_t^T \boldsymbol{X}_t) \boldsymbol{X}_t$$

**步骤2：msign正交化**
$$\text{msign}(\boldsymbol{G}) = \boldsymbol{G}(\boldsymbol{G}^T\boldsymbol{G})^{-1/2}$$

**步骤3：Muon更新**
$$\boldsymbol{m}_t = \beta \boldsymbol{m}_{t-1} + (1-\beta)\boldsymbol{g}_t$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \text{msign}(\boldsymbol{m}_t)$$

**几何意义**：在Stiefel流形上最速下降（正交约束下的梯度下降）。

---

## 3. 数学直觉、多角度解释与类比

### 3.1 Adam的"自适应步长"类比

**场景**：爬山时调整步幅。

- **平坦区域**（梯度小）：$\boldsymbol{v}_t$ 小 → 步长大 → 快速前进
- **陡峭区域**（梯度大）：$\boldsymbol{v}_t$ 大 → 步长小 → 谨慎前进
- **震荡方向**：$\boldsymbol{m}_t$ 平滑抵消 → 稳定前进

### 3.2 Muon的"矩阵罗盘"类比

**传统优化器**：向量指南针（只看梯度方向）
**Muon**：矩阵罗盘（考虑梯度的结构信息）

- **列空间**：参数的主要变化方向
- **msign**：找到"最佳旋转"，保持结构的同时归一化

### 3.3 学习率与Batch Size的"平均场"类比

**小Batch**：
- 噪声大（随机性强）
- 需要小学习率（避免发散）
- 类比：颠簸路面开车，速度要慢

**大Batch**：
- 噪声小（梯度准确）
- 可用大学习率（加速收敛）
- 类比：高速公路，可以加速

**Scaling Law**（线性）：$\eta \propto B$

---

## 4. 方法论变体、批判性比较与优化

### 4.1 优化器对比表

| 优化器 | 内存 | 计算 | **核心缺陷** | **优化方向** |
|--------|------|------|------------|-------------|
| **SGD+Momentum** | 1× | 低 | ❌ 需精细调参<br>❌ 对初始化敏感 | ✅ Nesterov加速<br>✅ 学习率warm-up |
| **Adam** | 2× | 中 | ❌ 大Batch下性能差<br>❌ 权重衰减耦合 | ✅ AdamW解耦<br>✅ Adafactor降内存 |
| **Lion** | 1× | 低 | ❌ 震荡风险<br>❌ Embedding层异常 | ✅ 动态$\lambda$<br>✅ 分层学习率 |
| **Muon** | 1× | 高 | ❌ 仅适用矩阵层<br>❌ 需大Batch | ✅ QK-Clip稳定<br>✅ 混合优化器 |

### 4.2 Adam的批判性分析

**缺陷1：Update RMS不灵活**
- **问题**：$\text{RMS}(\Delta\boldsymbol{\theta}) \approx 0.2$ 是硬编码的
- **影响**：不同层的最优RMS可能不同（Embedding vs FFN）
- **优化**：动态RMS估计 $\alpha_t = \eta \frac{\|\boldsymbol{\theta}_t\|}{\sqrt{\boldsymbol{v}_t}}$

**缺陷2：$\epsilon$ 的Scaling Law不明确**
- **问题**：$\epsilon = 10^{-8}$ 在不同模型尺度下表现不一
- **优化**：自适应$\epsilon_t = c \cdot \text{mean}(\sqrt{\boldsymbol{v}_t})$

### 4.3 Muon的批判性分析

**缺陷1：小奇异值不稳定**
- **问题**：msign对接近奇异的矩阵放大噪声
- **表现**：$\sigma_{\min} < 0.01$ 时，更新方向随机
- **优化**：QK-Clip裁剪奇异值到 $[\alpha, \beta]$

**缺陷2：计算开销**
- **问题**：Newton-Schulz迭代需6次矩阵乘法
- **对比**：Adam只需逐元素操作
- **优化**：混合策略（Embedding用Adam，Transformer用Muon）

---

## 5. 学习路线图与未来展望

### 5.1 必备知识

1. **凸优化理论**：强凸性、Lipschitz连续、收敛速度分析
2. **随机优化**：SGD方差分析、重要性采样
3. **线性代数**：条件数、谱范数、Hessian矩阵
4. **流形优化**：Riemannian梯度、测地线

### 5.2 未来方向

**方向1：二阶信息的高效利用**
- **问题**：牛顿法需 $O(n^3)$ 计算Hessian逆
- **优化**：
  - BFGS/L-BFGS低秩近似
  - Shampoo块对角近似
  - K-FAC Kronecker分解：$\boldsymbol{H} \approx \boldsymbol{A} \otimes \boldsymbol{B}$

**方向2：自适应学习率的理论基础**
- **研究空白**：为什么 $\sqrt{\boldsymbol{v}_t}$ 而不是 $\boldsymbol{v}_t$？
- **猜想**：与Hessian对角元的平方根相关
- **验证**：在不同曲率条件下的收敛性证明

**方向3：大Batch训练的突破**
- **目标**：Batch Size > 100K，保持小Batch性能
- **挑战**：泛化gap、Sharp minima
- **方向**：
  - LARS/LAMB层自适应Scaling
  - Ghost Normalization虚拟小Batch
  - Lookhead多步验证

---

**撰写日期**：2025-11-18
**版本**：v1.0
