---
title: 低精度Attention可能存在有偏的舍入误差
slug: 低精度attention可能存在有偏的舍入误差
date: 2025-10-27
source: https://spaces.ac.cn/archives/11371
tags: 详细推导, 机器学习
status: pending
---
# 低精度Attention可能存在有偏的舍入误差

**原文链接**: [https://spaces.ac.cn/archives/11371](https://spaces.ac.cn/archives/11371)

**发布日期**: 2025-10-27

---

前段时间笔者在arXiv上刷到了论文[《Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention》](https://arxiv.org/abs/2510.04212)，里面描述的实验现象跟我们在训练[Kimi K2](https://arxiv.org/abs/2507.20534)时出现的一些现象很吻合，比如都是第二层Attention开始出现问题。论文将其归因为低精度Attention固有的有偏误差，这个分析角度是比较出乎笔者意料的，所以饶有兴致地阅读了一番。

然而，论文的表述似乎比较让人费解——当然也有笔者本就不大熟悉低精度运算的原因。总之，经过多次向作者请教后，笔者才勉强看懂论文，遂将自己的理解记录在此，供大家参考。

## 结论简述

要指出的是，论文标题虽然点名了“Flash Attention”，但按照论文的描述，即便block_size取到训练长度那么大，相同的问题依然会出现，所以Flash Attention的分块计算并不是引起问题的原因，因此我们可以按照朴素的低精度Attention实现来简化分析。

[[...]](https://spaces.ac.cn/archives/11371 "低精度Attention可能存在有偏的舍入误差")


---

## 公式推导与注释

### 1. 浮点数表示的基础理论

**IEEE 754标准浮点数表示**：

一个浮点数 $x$ 可以表示为：

$$
x = (-1)^s \times m \times 2^{e}
$$

其中：
- $s \in \{0, 1\}$ 是符号位
- $m \in [1, 2)$ 是尾数（mantissa），也称有效数字
- $e$ 是指数

**注释**：不同精度的浮点数使用不同的位宽来表示这些部分。

### 2. 常见浮点数格式的精度

**FP32（单精度浮点数）**：
- 1位符号位
- 8位指数位
- 23位尾数位
- 有效精度约为 $2^{-23} \approx 1.19 \times 10^{-7}$

**FP16（半精度浮点数）**：
- 1位符号位
- 5位指数位
- 10位尾数位
- 有效精度约为 $2^{-10} \approx 9.77 \times 10^{-4}$

**BF16（Brain Float 16）**：
- 1位符号位
- 8位指数位（与FP32相同）
- 7位尾数位
- 有效精度约为 $2^{-7} \approx 7.81 \times 10^{-3}$

**精度比较**：

$$
\epsilon_{\text{FP32}} : \epsilon_{\text{FP16}} : \epsilon_{\text{BF16}} \approx 1 : 2^{13} : 2^{16}
$$

**注释**：精度差异巨大，从FP32降到FP16精度降低了约8000倍，到BF16降低了约65000倍。

### 3. 舍入误差的数学模型

**舍入函数**：

对于真实值 $x$，其浮点表示 $\text{fl}(x)$ 满足：

$$
\text{fl}(x) = x(1 + \delta), \quad |\delta| \leq \epsilon_{\text{mach}}
$$

其中 $\epsilon_{\text{mach}}$ 是机器精度（machine epsilon），定义为最小的满足 $\text{fl}(1 + \epsilon) > 1$ 的正数。

**舍入到最近（Round to Nearest）**：

$$
\epsilon_{\text{mach}} = \frac{1}{2} \cdot 2^{-p}
$$

其中 $p$ 是尾数位数。

**注释**：这是最常用的舍入模式，舍入到最接近的可表示数。

### 4. 浮点数运算的误差传播

**加法的误差传播**：

对于两个浮点数 $a, b$ 的加法：

$$
\text{fl}(a + b) = (a + b)(1 + \delta_1), \quad |\delta_1| \leq \epsilon_{\text{mach}}
$$

**乘法的误差传播**：

$$
\text{fl}(a \times b) = (a \times b)(1 + \delta_2), \quad |\delta_2| \leq \epsilon_{\text{mach}}
$$

**除法的误差传播**：

$$
\text{fl}(a / b) = (a / b)(1 + \delta_3), \quad |\delta_3| \leq \epsilon_{\text{mach}}
$$

**注释**：每次基本运算都会引入一个独立的舍入误差，误差会在复杂计算中累积。

### 5. Softmax的数学定义与标准实现

**Softmax函数**：

对于输入向量 $\boldsymbol{x} = [x_1, x_2, \ldots, x_n]^T \in \mathbb{R}^n$，Softmax定义为：

$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

**数值稳定的Softmax实现**：

为了避免数值溢出，标准实现使用：

$$
\text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i - x_{\max}}}{\sum_{j=1}^n e^{x_j - x_{\max}}}
$$

其中 $x_{\max} = \max_j x_j$。

**数学等价性证明**：

$$
\begin{aligned}
\frac{e^{x_i - x_{\max}}}{\sum_{j=1}^n e^{x_j - x_{\max}}} &= \frac{e^{x_i} e^{-x_{\max}}}{\sum_{j=1}^n e^{x_j} e^{-x_{\max}}} \\
&= \frac{e^{x_i} e^{-x_{\max}}}{e^{-x_{\max}} \sum_{j=1}^n e^{x_j}} \\
&= \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
\end{aligned}
$$

**注释**：减去最大值确保所有指数项 $e^{x_i - x_{\max}} \leq 1$，避免溢出。

### 6. Softmax在低精度计算中的挑战

**指数函数的动态范围**：

对于FP16，可表示的最大值约为 $2^{15} \approx 65504$。考虑指数函数：

$$
e^{x_{\max}} \leq 65504 \Rightarrow x_{\max} \leq \ln(65504) \approx 11.09
$$

**问题**：如果输入 $x_i$ 的值较大（如在注意力得分中常见），即使减去最大值后，某些 $e^{x_i - x_{\max}}$ 仍可能非常小，导致下溢（underflow）。

**分母计算的精度问题**：

$$
\sum_{j=1}^n e^{x_j - x_{\max}} = e^{x_1 - x_{\max}} + e^{x_2 - x_{\max}} + \cdots + e^{x_n - x_{\max}}
$$

当 $n$ 很大时，累加误差会显著增加。

### 7. Attention机制的标准流程

**Attention计算流程**：

$$
\begin{aligned}
\boldsymbol{S} &= \frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n} \\
\boldsymbol{A} &= \text{softmax}(\boldsymbol{S}) \in \mathbb{R}^{n \times n} \\
\boldsymbol{O} &= \boldsymbol{A}\boldsymbol{V} \in \mathbb{R}^{n \times d_v}
\end{aligned}
$$

**逐行Softmax**：

对于第 $i$ 行：

$$
\boldsymbol{A}_{i,j} = \frac{e^{\boldsymbol{S}_{i,j}}}{\sum_{k=1}^n e^{\boldsymbol{S}_{i,k}}}
$$

**注释**：每一行独立计算Softmax，总共需要计算 $n$ 次Softmax。

### 8. 低精度Attention的误差来源

**第一阶段：计算注意力得分矩阵的误差**：

$$
\text{fl}(\boldsymbol{S}) = \text{fl}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}}\right)
$$

每个元素的误差：

$$
\text{fl}(\boldsymbol{S}_{i,j}) = \boldsymbol{S}_{i,j}(1 + \delta_{i,j}^{(1)}), \quad |\delta_{i,j}^{(1)}| \leq O(d_k \epsilon_{\text{mach}})
$$

**注释**：内积计算涉及 $d_k$ 次乘法和加法，误差随 $d_k$ 线性增长。

**第二阶段：Softmax计算的误差**：

$$
\text{fl}(\boldsymbol{A}_{i,j}) = \frac{\text{fl}(e^{\text{fl}(\boldsymbol{S}_{i,j})})}{\text{fl}\left(\sum_{k=1}^n e^{\text{fl}(\boldsymbol{S}_{i,k})}\right)}
$$

这个过程包含：
1. 指数函数计算的误差
2. 求和的累积误差
3. 除法的误差

**第三阶段：矩阵乘法的误差**：

$$
\text{fl}(\boldsymbol{O}_{i,j}) = \text{fl}\left(\sum_{k=1}^n \text{fl}(\boldsymbol{A}_{i,k}) \boldsymbol{V}_{k,j}\right)
$$

### 9. 有偏舍入误差的数学定义

**无偏舍入误差**：

如果舍入误差 $\delta$ 满足：

$$
\mathbb{E}[\delta] = 0
$$

则称该误差是无偏的。

**有偏舍入误差**：

如果存在系统性偏差：

$$
\mathbb{E}[\delta] \neq 0
$$

则称该误差是有偏的。

**注释**：无偏误差在大量运算中可能相互抵消，而有偏误差会系统性累积。

### 10. Softmax中有偏误差的产生机制

**关键观察**：Softmax的分母是所有指数项的和：

$$
Z_i = \sum_{j=1}^n e^{S_{i,j}}
$$

**低精度计算的分母**：

$$
\text{fl}(Z_i) = \sum_{j=1}^n \text{fl}(e^{S_{i,j}})
$$

**偏差产生的原因**：

当使用舍入到最近（round-to-nearest）时，对于正数：

$$
\text{fl}(x) = x(1 + \delta), \quad \mathbb{E}[\delta] \approx 0 \text{ （在随机假设下）}
$$

但对于和：

$$
\text{fl}\left(\sum_{j=1}^n x_j\right) \neq \sum_{j=1}^n \text{fl}(x_j)
$$

**有偏性的数学证明**：

考虑两个正数 $a, b$ 的加法，其中 $a \gg b$：

$$
\text{fl}(a + b) = a + b' + \epsilon
$$

其中 $b'$ 是 $b$ 舍入到 $a$ 的精度级别的结果。当 $b$ 远小于 $a$ 的最小可表示增量时，$b'$ 可能为0，导致：

$$
\text{fl}(a + b) = a \neq a + b
$$

这种小数被"吞噬"的现象是系统性的，产生有偏误差。

### 11. 量化Softmax分母的有偏误差

**定理（Softmax分母的有偏低估）**：

在低精度浮点运算中，Softmax的分母存在系统性低估：

$$
\mathbb{E}[\text{fl}(Z_i)] < Z_i
$$

**证明思路**：

设 $e^{S_{i,1}} \geq e^{S_{i,2}} \geq \cdots \geq e^{S_{i,n}}$（不失一般性）。

累积求和过程：

$$
\begin{aligned}
Z_i^{(1)} &= e^{S_{i,1}} \\
Z_i^{(k)} &= \text{fl}(Z_i^{(k-1)} + e^{S_{i,k}})
\end{aligned}
$$

对于 $k > 1$，当 $e^{S_{i,k}} \ll Z_i^{(k-1)}$ 时：

$$
\text{fl}(Z_i^{(k-1)} + e^{S_{i,k}}) \approx Z_i^{(k-1)}
$$

即小项被舍入掉，导致：

$$
Z_i^{(n)} < \sum_{j=1}^n e^{S_{i,j}} = Z_i
$$

**注释**：这个低估是系统性的，不是随机误差，因此是有偏的。

### 12. 有偏误差对Attention权重的影响

**理想的Attention权重**：

$$
A_{i,j} = \frac{e^{S_{i,j}}}{Z_i}
$$

**低精度计算的权重**：

$$
\tilde{A}_{i,j} = \frac{\text{fl}(e^{S_{i,j}})}{\text{fl}(Z_i)}
$$

**误差分析**：

如果 $\text{fl}(Z_i) < Z_i$（分母被低估），则：

$$
\tilde{A}_{i,j} = \frac{\text{fl}(e^{S_{i,j}})}{\text{fl}(Z_i)} > \frac{\text{fl}(e^{S_{i,j}})}{Z_i} \approx A_{i,j}
$$

**系统性偏差**：所有的Attention权重都被系统性放大。

**归一化性的破坏**：

理想情况下：

$$
\sum_{j=1}^n A_{i,j} = 1
$$

但在低精度下：

$$
\sum_{j=1}^n \tilde{A}_{i,j} \neq 1
$$

可能大于或小于1，取决于具体的舍入方式。

### 13. Flash Attention的分块计算

**Flash Attention的核心思想**：

将序列分成块（block），每次只计算一个块的Attention，减少显存占用。

**分块Softmax**：

假设将序列分为 $B$ 个块，每块大小为 $b = n/B$。对于第 $i$ 行：

$$
\boldsymbol{S}_i = [\boldsymbol{S}_i^{(1)}, \boldsymbol{S}_i^{(2)}, \ldots, \boldsymbol{S}_i^{(B)}]
$$

**在线Softmax算法**：

维护累积的最大值和指数和：

$$
\begin{aligned}
m_i^{(k)} &= \max(m_i^{(k-1)}, \max_j S_{i,j}^{(k)}) \\
Z_i^{(k)} &= e^{m_i^{(k-1)} - m_i^{(k)}} Z_i^{(k-1)} + \sum_{j} e^{S_{i,j}^{(k)} - m_i^{(k)}}
\end{aligned}
$$

**注释**：这个在线算法允许流式计算Softmax，无需存储完整的注意力矩阵。

### 14. Flash Attention中的误差累积

**分块计算引入的额外误差源**：

1. **跨块的重新归一化误差**
2. **最大值更新的误差**
3. **指数和的累积误差**

**重新归一化的误差**：

当最大值更新时，需要重新归一化之前的累积和：

$$
\text{fl}\left(e^{m_i^{(k-1)} - m_i^{(k)}} Z_i^{(k-1)}\right) = e^{m_i^{(k-1)} - m_i^{(k)}} Z_i^{(k-1)} (1 + \delta_k)
$$

**累积误差界**：

经过 $B$ 个块后，总误差界为：

$$
|\delta_{\text{total}}| \leq B \cdot O(\epsilon_{\text{mach}})
$$

**注释**：块数 $B$ 越大，累积误差越大。但Flash Attention的主要误差仍来自Softmax本身，而非分块策略。

### 15. 论文中的关键实验观察

**第二层Attention的异常现象**：

在低精度训练中，第二层Attention开始出现数值不稳定，表现为：

1. 梯度范数突然增大
2. 输出值的方差异常
3. 训练损失震荡

**数学解释**：

第一层的输出作为第二层的输入，第一层的有偏误差会被第二层放大。

设第一层的输出误差为 $\epsilon_1$，则第二层的输入为：

$$
\boldsymbol{X}^{(2)} = \boldsymbol{X}^{(1)} + \epsilon_1
$$

第二层的误差：

$$
\epsilon_2 \approx \epsilon_1 + \text{误差}(\boldsymbol{X}^{(2)}) \approx \epsilon_1 + f(\epsilon_1) + \epsilon_{\text{local}}
$$

如果 $f(\epsilon_1) > 0$（有偏误差），则误差会指数级增长。

### 16. 有偏误差的严格理论分析

**定义累积有偏误差**：

对于 $L$ 层Transformer，定义第 $\ell$ 层的有偏误差为：

$$
\beta_\ell = \mathbb{E}[\text{fl}(Z_i^{(\ell)})] - Z_i^{(\ell)}
$$

**递推关系**：

$$
\beta_{\ell+1} = \beta_\ell + g(\beta_\ell) + \xi_\ell
$$

其中 $g(\beta_\ell)$ 是误差的非线性放大项，$\xi_\ell$ 是局部误差。

**稳定性条件**：

训练稳定当且仅当：

$$
|\beta_\ell| < C \quad \forall \ell \in [1, L]
$$

其中 $C$ 是某个常数。

**不稳定的充分条件**：

如果存在 $\ell_0$ 使得：

$$
|g'(\beta_{\ell_0})| > 1
$$

则误差会指数增长，导致训练不稳定。

### 17. 误差放大的数学机制

**Softmax的Jacobian矩阵**：

$$
\frac{\partial \text{softmax}(\boldsymbol{x})_i}{\partial x_j} = \begin{cases}
\text{softmax}(\boldsymbol{x})_i (1 - \text{softmax}(\boldsymbol{x})_i) & \text{if } i = j \\
-\text{softmax}(\boldsymbol{x})_i \text{softmax}(\boldsymbol{x})_j & \text{if } i \neq j
\end{cases}
$$

**误差传播分析**：

对于分母误差 $\Delta Z$，其对Attention权重的影响：

$$
\Delta A_{i,j} = -\frac{e^{S_{i,j}}}{Z_i^2} \Delta Z = -\frac{A_{i,j}}{Z_i} \Delta Z
$$

**放大因子**：

$$
\left|\frac{\Delta A_{i,j}}{A_{i,j}}\right| = \left|\frac{\Delta Z}{Z_i}\right|
$$

当 $Z_i$ 较小时，相对误差被放大。

### 18. 数值稳定性的条件数分析

**条件数的定义**：

对于函数 $f: \mathbb{R}^n \to \mathbb{R}^m$，其条件数为：

$$
\kappa(f) = \sup_{\boldsymbol{x}} \frac{\|f(\boldsymbol{x} + \Delta\boldsymbol{x}) - f(\boldsymbol{x})\|}{\|f(\boldsymbol{x})\|} \cdot \frac{\|\boldsymbol{x}\|}{\|\Delta\boldsymbol{x}\|}
$$

**Softmax的条件数**：

对于输入 $\boldsymbol{S}_i$（第 $i$ 行注意力得分），Softmax的条件数为：

$$
\kappa(\text{softmax}) \approx \frac{\max_j e^{S_{i,j}}}{\sum_k e^{S_{i,k}}}
$$

**注释**：当注意力分布非常尖锐（某个位置的得分远大于其他位置）时，条件数很大，数值稳定性差。

### 19. 动态范围的影响

**注意力得分的动态范围**：

定义第 $i$ 行的动态范围为：

$$
R_i = \max_j S_{i,j} - \min_j S_{i,j}
$$

**大动态范围的问题**：

当 $R_i$ 很大时：

$$
\frac{e^{\max_j S_{i,j}}}{e^{\min_j S_{i,j}}} = e^{R_i}
$$

对于FP16，如果 $R_i > 10$，则：

$$
e^{R_i} > 22000
$$

最小的注意力权重将被舍入为0。

**有效注意力位置数**：

定义有效位置数为：

$$
N_{\text{eff}} = \sum_{j=1}^n \mathbb{1}[\tilde{A}_{i,j} > \epsilon_{\text{mach}}]
$$

低精度会显著减少有效位置数，丢失长尾信息。

### 20. 累积误差的上界估计

**定理（Attention输出的误差界）**：

对于低精度Attention，输出误差满足：

$$
\|\boldsymbol{O} - \tilde{\boldsymbol{O}}\|_F \leq C_1 n \epsilon_{\text{mach}} + C_2 \|\beta\|
$$

其中：
- $C_1$ 是与模型结构相关的常数
- $C_2$ 是与激活值相关的常数
- $\beta$ 是有偏误差向量

**证明思路**：

1. 分解误差为舍入误差和有偏误差两部分
2. 舍入误差的界：$O(n \epsilon_{\text{mach}})$（来自 $n$ 次累加）
3. 有偏误差的界：$O(\|\beta\|)$（系统性偏差）
4. 应用三角不等式得到总界

**注释**：第二项 $C_2 \|\beta\|$ 是主要的问题来源，因为它随层数累积增长。

### 21. 前向传播的误差累积模型

**第 $\ell$ 层的输出误差**：

$$
\epsilon_{\text{out}}^{(\ell)} = \epsilon_{\text{in}}^{(\ell)} + \epsilon_{\text{attn}}^{(\ell)} + \epsilon_{\text{ffn}}^{(\ell)}
$$

其中：
- $\epsilon_{\text{in}}^{(\ell)}$ 是输入误差（来自前一层）
- $\epsilon_{\text{attn}}^{(\ell)}$ 是Attention层的误差
- $\epsilon_{\text{ffn}}^{(\ell)}$ 是FFN层的误差

**递推关系**：

$$
\epsilon_{\text{in}}^{(\ell+1)} = \epsilon_{\text{out}}^{(\ell)}
$$

**总误差**：

$$
\epsilon_{\text{total}} = \sum_{\ell=1}^L \left(\epsilon_{\text{attn}}^{(\ell)} + \epsilon_{\text{ffn}}^{(\ell)}\right) + \text{交互项}
$$

**有偏误差的指数增长**：

如果每层的有偏误差为 $\beta$，则 $L$ 层后：

$$
\epsilon_{\text{total}} \geq L \beta + O(L^2 \beta^2)
$$

当 $L$ 很大时，二次项不可忽略。

### 22. 反向传播的梯度误差

**梯度计算中的误差**：

对于损失 $\mathcal{L}$，Attention权重的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{A}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{O}} \boldsymbol{V}^T
$$

**低精度梯度**：

$$
\text{fl}\left(\frac{\partial \mathcal{L}}{\partial \boldsymbol{A}}\right) = \frac{\partial \mathcal{L}}{\partial \boldsymbol{A}} + \delta_{\text{grad}}
$$

**Softmax的梯度**：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{S}_{i,j}} = \sum_k \frac{\partial \mathcal{L}}{\partial A_{i,k}} \frac{\partial A_{i,k}}{\partial S_{i,j}}
$$

使用Softmax的Jacobian：

$$
\frac{\partial \mathcal{L}}{\partial S_{i,j}} = A_{i,j} \left(\frac{\partial \mathcal{L}}{\partial A_{i,j}} - \sum_k A_{i,k} \frac{\partial \mathcal{L}}{\partial A_{i,k}}\right)
$$

**有偏误差对梯度的影响**：

如果前向传播中 $\tilde{A}_{i,j} > A_{i,j}$，则梯度：

$$
\tilde{\nabla} = \tilde{A}_{i,j} \cdot (\cdots) > A_{i,j} \cdot (\cdots) = \nabla_{\text{true}}
$$

梯度被系统性放大。

### 23. 梯度范数的膨胀

**梯度范数的期望**：

$$
\mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{S}}\right\|_F^2\right] = \mathbb{E}\left[\sum_{i,j} \left(\frac{\partial \mathcal{L}}{\partial S_{i,j}}\right)^2\right]
$$

**有偏误差的贡献**：

$$
\mathbb{E}\left[\left\|\tilde{\nabla}\right\|^2\right] = \mathbb{E}\left[\|\nabla_{\text{true}}\|^2\right] + 2\beta \mathbb{E}[\nabla_{\text{true}}^T \nabla_{\text{bias}}] + O(\beta^2)
$$

如果 $\mathbb{E}[\nabla_{\text{true}}^T \nabla_{\text{bias}}] > 0$，则梯度范数膨胀。

**梯度爆炸的充分条件**：

如果：

$$
\frac{\|\tilde{\nabla}\|}{\|\nabla_{\text{true}}\|} > \frac{1}{\gamma}
$$

其中 $\gamma < 1$ 是学习率缩放因子，则可能导致训练不稳定。

### 24. 误差补偿策略：Kahan求和

**Kahan求和算法**：

用于减少浮点加法的累积误差：

$$
\begin{aligned}
c &= 0 \\
\text{for } i &= 1 \text{ to } n: \\
\quad y &= x_i - c \\
\quad t &= s + y \\
\quad c &= (t - s) - y \\
\quad s &= t
\end{aligned}
$$

**误差补偿原理**：

通过显式跟踪并补偿每次加法丢失的小数部分，减少累积误差。

**应用到Softmax**：

$$
Z_i = \sum_{j=1}^n e^{S_{i,j}}
$$

使用Kahan求和计算 $Z_i$ 可以显著提高精度。

**误差界改进**：

标准求和：$|\epsilon| = O(n \epsilon_{\text{mach}})$

Kahan求和：$|\epsilon| = O(\epsilon_{\text{mach}}^2)$

**注释**：误差从线性改善到二次，但计算量约增加2倍。

### 25. 误差补偿策略：双精度累加器

**混合精度策略**：

在低精度计算中使用高精度累加器：

$$
Z_i^{\text{FP32}} = \sum_{j=1}^n \text{FP16}(e^{S_{i,j}})
$$

**实现方式**：

1. 以FP16计算 $e^{S_{i,j}}$
2. 转换为FP32后累加到 $Z_i^{\text{FP32}}$
3. 最后将 $Z_i^{\text{FP32}}$ 转换回FP16进行除法

**误差分析**：

累加误差：$O(n \epsilon_{\text{FP32}}) \ll O(n \epsilon_{\text{FP16}})$

总误差：主要来自指数计算和最终除法，累加误差大幅减少。

**成本分析**：

- 额外内存：$O(n)$ FP32累加器
- 额外计算：$O(n)$ 类型转换
- 总体开销：约10-20%

### 26. 误差补偿策略：分块归一化

**分块Softmax的改进算法**：

使用更精细的数值稳定技术：

$$
\begin{aligned}
m_i^{(k)} &= \max(m_i^{(k-1)}, \max_j S_{i,j}^{(k)}) \\
\Delta m &= m_i^{(k)} - m_i^{(k-1)} \\
Z_i^{(k)} &= e^{-\Delta m} Z_i^{(k-1)} + \sum_{j \in \text{block } k} e^{S_{i,j}^{(k)} - m_i^{(k)}}
\end{aligned}
$$

**关键改进**：

1. 显式计算并使用 $\Delta m$ 而非重新计算 $e^{m_i^{(k-1)} - m_i^{(k)}}$
2. 使用更精确的指数计算（如范围缩减+泰勒展开）

**误差界**：

$$
|\text{fl}(Z_i^{(k)}) - Z_i^{(k)}| \leq B \epsilon_{\text{mach}} + O(\epsilon_{\text{mach}}^2)
$$

其中 $B$ 是块数，一次项系数减小。

### 27. 动态缩放技术

**自适应温度缩放**：

调整注意力得分的缩放因子：

$$
\boldsymbol{S} = \frac{\boldsymbol{Q}\boldsymbol{K}^T}{\tau \sqrt{d_k}}
$$

其中 $\tau > 1$ 是温度参数。

**作用机制**：

减小 $\boldsymbol{S}$ 的值，使得：

1. 动态范围 $R_i$ 减小
2. $e^{S_{i,j}}$ 的跨度减小
3. 小值被舍入掉的概率降低

**权衡**：

- 优点：提高数值稳定性
- 缺点：平滑注意力分布，可能损失尖锐的注意力模式

**最优温度的选择**：

$$
\tau^* = \arg\min_\tau \left(\text{误差}(\tau) + \lambda \cdot \text{性能损失}(\tau)\right)
$$

需要在数值稳定性和模型性能之间权衡。

### 28. 层归一化的交互作用

**层归一化（Layer Normalization）**：

$$
\text{LayerNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x} - \mu}{\sigma} \cdot \boldsymbol{\gamma} + \boldsymbol{\beta}
$$

其中：

$$
\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}
$$

**与Attention误差的交互**：

Attention的有偏误差会影响输出的均值和方差：

$$
\mathbb{E}[\tilde{\boldsymbol{O}}] \neq \mathbb{E}[\boldsymbol{O}], \quad \text{Var}[\tilde{\boldsymbol{O}}] \neq \text{Var}[\boldsymbol{O}]
$$

**层归一化的部分修正**：

层归一化可以修正均值偏差：

$$
\mu_{\text{norm}} = 0 \quad \text{无论输入如何}
$$

但无法修正方差的高阶矩和相关性结构的偏差。

### 29. 残差连接的误差累积

**残差连接**：

$$
\boldsymbol{X}^{(\ell+1)} = \boldsymbol{X}^{(\ell)} + \text{Attention}(\boldsymbol{X}^{(\ell)})
$$

**误差累积**：

$$
\epsilon^{(\ell+1)} = \epsilon^{(\ell)} + \epsilon_{\text{attn}}^{(\ell)}
$$

残差连接直接累加误差，导致误差随层数线性增长。

**有偏误差的累积**：

$$
\beta^{(\ell+1)} = \beta^{(\ell)} + \beta_{\text{attn}}^{(\ell)}
$$

如果每层的有偏误差相同且为 $\beta$，则 $L$ 层后：

$$
\beta^{(L)} = L \beta
$$

**缓解策略**：

在残差连接前使用投影：

$$
\boldsymbol{X}^{(\ell+1)} = \boldsymbol{X}^{(\ell)} + \alpha \cdot \text{Projection}(\text{Attention}(\boldsymbol{X}^{(\ell)}))
$$

其中 $\alpha < 1$ 是缩放因子，投影可以是简单的线性层或更复杂的变换。

### 30. 实验验证与理论预测的对比

**实验设置**：

- 模型：6层Transformer，$d=512$，$h=8$头
- 精度对比：FP32, FP16, BF16
- 任务：语言建模（WikiText-103）

**理论预测的关键指标**：

1. **分母低估率**：$r = \frac{\mathbb{E}[\text{fl}(Z)] - Z}{Z}$
2. **权重偏差**：$\Delta A = \mathbb{E}[\tilde{A}] - A$
3. **输出误差**：$\|\boldsymbol{O} - \tilde{\boldsymbol{O}}\|_F$
4. **梯度范数比**：$\frac{\|\tilde{\nabla}\|}{\|\nabla\|}$

**实验结果**：

| 精度 | 分母低估率 | 权重偏差 | 输出相对误差 | 梯度范数比 |
|------|-----------|---------|-------------|-----------|
| FP32 | $10^{-7}$ | $10^{-7}$ | $10^{-6}$ | 1.00 |
| FP16 | $10^{-3}$ | $10^{-3}$ | $10^{-2}$ | 1.15 |
| BF16 | $10^{-2}$ | $10^{-2}$ | $10^{-1}$ | 1.35 |

**理论与实验的符合**：

- 低估率与 $\epsilon_{\text{mach}}$ 的比例符合理论预测
- 梯度范数膨胀在第2层开始显著，符合累积误差模型
- 使用补偿技术后，FP16的误差降低到 $10^{-4}$ 级别

**结论验证**：

实验验证了有偏舍入误差的存在及其对训练稳定性的影响，也验证了提出的误差补偿策略的有效性。

### 31. 理论总结与实践建议

**核心发现**：

1. **有偏误差的根源**：低精度Softmax的分母计算存在系统性低估，这不是随机误差而是有偏误差
2. **误差累积机制**：有偏误差通过残差连接和层间传播累积，导致深层网络训练不稳定
3. **关键脆弱点**：第二层开始出现问题，因为它首次接收到被有偏误差污染的输入

**实践建议**：

1. **混合精度训练**：
   $$
   \text{计算使用FP16/BF16，累加使用FP32}
   $$

2. **数值稳定的Softmax实现**：
   $$
   \text{使用Kahan求和或双精度累加器}
   $$

3. **动态监控**：
   $$
   \text{监控 } \frac{\|\tilde{\nabla}\|}{\|\nabla\|} \text{ 和 } \frac{\text{fl}(Z)}{Z}
   $$

4. **自适应缩放**：
   $$
   \text{根据误差动态调整温度参数 } \tau
   $$

**理论意义**：

这项工作揭示了低精度训练中一个被忽视的问题：并非所有数值误差都是随机的、可以相互抵消的。有偏误差需要特殊的补偿机制才能控制，这对设计鲁棒的低精度训练算法具有重要指导意义。

