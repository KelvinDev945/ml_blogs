# 概率统计主题深度Summary

> **涵盖文章**：10篇概率统计相关文章
> **主要内容**：贝叶斯推断、Viterbi采样、Softmax替代、概率不等式

---

## 1. 核心理论

### 1.1 贝叶斯推断公理
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**CAN方法应用**：利用先验分布校正分类器
$$\hat{y} = \arg\max_c \frac{P(c|x)}{P(c)} \cdot P_{prior}(c)$$

### 1.2 Viterbi采样

**Viterbi解码**（确定性）：
$$\boldsymbol{y}^* = \arg\max_{\boldsymbol{y}} P(\boldsymbol{y}|\boldsymbol{x})$$

**Viterbi采样**（随机性）：
- 步骤1：计算前向概率 $\alpha_t(s)$
- 步骤2：从后向前采样 $s_t \sim P(s_t | s_{t+1}, \boldsymbol{x})$

**完美采样**：拒绝采样保证严格分布
$$P_{accept} = \frac{P(\boldsymbol{y}|\boldsymbol{x})}{M \cdot Q(\boldsymbol{y})}$$

---

## 2. Softmax替代品对比

| 方法 | 公式 | 优点 | **缺陷** | **优化** |
|------|------|------|---------|---------|
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | 标准，可微 | ❌ 过度自信<br>❌ 计算expensive | ✅ Temperature缩放<br>✅ Label smoothing |
| **Sparsemax** | $\arg\min \|\boldsymbol{p} - \boldsymbol{x}\|^2$ | 稀疏输出 | ❌ 不光滑 | ✅ α-entmax平滑 |
| **Gumbel-Softmax** | $\frac{e^{(x_i + g_i)/\tau}}{\sum e^{(x_j+g_j)/\tau}}$ | 可微采样 | ❌ Temperature敏感 | ✅ 自适应$\tau$ |
| **熵归一化** | 约束 $H(\boldsymbol{p}) = H_0$ | 控制不确定性 | ❌ 非凸优化 | ✅ 拉格朗日乘子法 |

---

## 3. 概率不等式

### 3.1 经典不等式

**Chebyshev不等式**：
$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

**Hoeffding不等式**（有界变量）：
$$P(|\bar{X} - \mu| \geq t) \leq 2e^{-2nt^2/(b-a)^2}$$

### 3.2 博客中的概率不等式

**待证明**：
$$P\left(\sum_{i=1}^n X_i \geq \alpha\right) \leq \frac{\mathbb{E}[\sum X_i]}{\alpha}$$

**证明**（Markov不等式）：
对非负随机变量 $Y$：
$$P(Y \geq \alpha) \leq \frac{\mathbb{E}[Y]}{\alpha}$$

**应用**：MoE负载均衡分析

---

## 4. 未来方向

**方向1：因果推断**
- 从相关性到因果性
- Do-calculus、反事实推理
- 应用：公平ML、可解释AI

**方向2：分布式概率估计**
- 联邦学习下的密度估计
- 差分隐私+贝叶斯推断

**方向3：非参数方法**
- Dirichlet过程
- 高斯过程
- Neural Processes

---

**撰写日期**：2025-11-18
**版本**：v1.0
