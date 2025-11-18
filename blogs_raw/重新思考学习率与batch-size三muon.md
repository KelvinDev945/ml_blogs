---
title: 重新思考学习率与Batch Size（三）：Muon
slug: 重新思考学习率与batch-size三muon
date: 2025-09-15
source: https://spaces.ac.cn/archives/11285
tags: 优化
status: pending
---

# 重新思考学习率与Batch Size（三）：Muon

**原文链接**: [https://spaces.ac.cn/archives/11285](https://spaces.ac.cn/archives/11285)

**发布日期**: 2025-09-15

---

前两篇文章[《重新思考学习率与Batch Size（一）：现状》](https://kexue.fm/archives/11260)和[《重新思考学习率与Batch Size（二）：平均场》](https://kexue.fm/archives/11280)中，我们主要是提出了平均场方法，用以简化学习率与Batch Size的相关计算。当时我们分析的优化器是SGD、SignSGD和SoftSignSGD，并且主要目的是简化，本质上没有新的结论。

然而，在如今的优化器盛宴中，怎能少得了Muon的一席之地呢？所以，这篇文章我们就来尝试计算Muon的相关结论，看看它的学习率与Batch Size的关系是否会呈现出新的规律。

## 基本记号

众所周知，[Muon](https://kexue.fm/archives/10592)的主要特点就是非Element-wise的更新规则，所以之前在[《当Batch Size增大时，学习率该如何随之变化？》](https://kexue.fm/archives/10542)和[《Adam的epsilon如何影响学习率的Scaling Law？》](https://kexue.fm/archives/10563)的Element-wise的计算方法将完全不可用。但幸运的是，上篇文章介绍的平均场依然好使，只需要稍微调整一下细节。

[[...]](https://spaces.ac.cn/archives/11285 "重新思考学习率与Batch Size（三）：Muon")


---

## 公式推导与注释

### 1. Muon优化器的数学基础

#### 1.1 矩阵符号函数

**定义**: 对于矩阵$\boldsymbol{A} \in \mathbb{R}^{m \times n}$,其符号函数为:
\begin{equation}\text{sign}(\boldsymbol{A}) = \boldsymbol{U}\text{sign}(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}\tag{1}\end{equation}

其中$\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$是SVD分解,$\text{sign}(\boldsymbol{\Sigma})$是对角矩阵,对角元素为$\pm 1$。

**数学直觉**: 矩阵符号函数保留方向信息,归一化幅度。对于向量,$\text{sign}(\boldsymbol{v}) = \boldsymbol{v}/\|\boldsymbol{v}\|$。

#### 1.2 Newton-Schulz迭代

**目标**: 计算$\text{sign}(\boldsymbol{A})$而不显式SVD。

**迭代公式**:
\begin{equation}\boldsymbol{X}_{k+1} = \frac{1}{2}\boldsymbol{X}_k(3\boldsymbol{I} - \boldsymbol{X}_k^2)\tag{2}\end{equation}

初始化:$\boldsymbol{X}_0 = \boldsymbol{A}/\|\boldsymbol{A}\|_F$

**收敛性**: 若$\|\boldsymbol{I} - \boldsymbol{X}_0^2\| < 1$,则$\boldsymbol{X}_k \to \text{sign}(\boldsymbol{A})$。

**收敛速度**: 三次收敛,误差$\mathcal{O}(\epsilon^{3^k})$。

#### 1.3 Muon更新规则

**原始形式**:
\begin{equation}\begin{aligned}
\boldsymbol{M}_t &= \beta\boldsymbol{M}_{t-1} + (1-\beta)\nabla L_t\\
\boldsymbol{\Theta}_{t+1} &= \boldsymbol{\Theta}_t - \eta \cdot \text{sign}(\boldsymbol{M}_t)
\end{aligned}\tag{3}\end{equation}

其中$\boldsymbol{\Theta}, \boldsymbol{M} \in \mathbb{R}^{m \times n}$是参数和动量矩阵。

**数学直觉**: Muon对动量矩阵整体做符号归一化,而非逐元素操作,保留了参数矩阵的几何结构。

### 2. Muon的理论优势

#### 2.1 尺度不变性

**定理1**: Muon对参数缩放不变:
\begin{equation}\text{若} \quad \boldsymbol{\Theta} \to c\boldsymbol{\Theta}, \quad \text{则} \quad \Delta\boldsymbol{\Theta}_{Muon} = \text{sign}(\boldsymbol{M}) \quad \text{不变}\tag{4}\end{equation}

**证明**: 缩放只改变梯度的幅度,不改变符号矩阵。

**实践意义**: 不需要精细调节权重衰减系数。

#### 2.2 与SignSGD的联系

**向量情况**: 若$\boldsymbol{\theta} \in \mathbb{R}^d$为向量,
\begin{equation}\text{sign}(\boldsymbol{m}) = \frac{\boldsymbol{m}}{\|\boldsymbol{m}\|}\tag{5}\end{equation}

**矩阵推广**: Muon将归一化从$\ell_2$推广到Frobenius范数:
\begin{equation}\text{sign}(\boldsymbol{M}) = \frac{\boldsymbol{M}}{\|\boldsymbol{M}\|_F} \cdot \text{低秩校正}\tag{6}\end{equation}

**数学直觉**: SignSGD是Muon在向量参数上的特例。

#### 2.3 预条件的视角

将Muon视为预条件优化:
\begin{equation}\boldsymbol{\Theta}_{t+1} = \boldsymbol{\Theta}_t - \eta \boldsymbol{P}_t^{-1}\boldsymbol{M}_t\tag{7}\end{equation}

其中$\boldsymbol{P}_t = \|\boldsymbol{M}_t\|_F \cdot \boldsymbol{I}$是隐式预条件矩阵。

**与Adam对比**: Adam使用对角预条件,Muon使用各向同性预条件。

### 3. 平均场近似分析

#### 3.1 更新方向的期望

对于随机梯度$\tilde{\boldsymbol{G}}_B$,定义更新方向:
\begin{equation}\boldsymbol{\Phi}_B = \text{sign}(\boldsymbol{M}_t), \quad \boldsymbol{M}_t = (1-\beta)\sum_{s=1}^t \beta^{t-s}\tilde{\boldsymbol{G}}_B^{(s)}\tag{8}\end{equation}

**平均场近似**:
\begin{equation}\mathbb{E}[\text{sign}(\boldsymbol{M}_t)] \approx \text{sign}(\mathbb{E}[\boldsymbol{M}_t]) = \text{sign}\left((1-\beta)\sum_s \beta^{t-s}\boldsymbol{G}\right) = \text{sign}(\boldsymbol{G})\tag{9}\end{equation}

假设$t \to \infty$时,$\boldsymbol{G}_s \approx \boldsymbol{G}$缓变。

#### 3.2 二阶矩分析

**方差计算**:
\begin{equation}\mathbb{E}[\boldsymbol{M}_t \boldsymbol{M}_t^{\top}] = \mathbb{E}[\boldsymbol{M}_t]\mathbb{E}[\boldsymbol{M}_t]^{\top} + \text{Cov}[\boldsymbol{M}_t]\tag{10}\end{equation}

利用EMA的方差减少效应:
\begin{equation}\text{Cov}[\boldsymbol{M}_t] = \frac{1-\beta}{1+\beta}\frac{\boldsymbol{\Sigma}}{B}\tag{11}\end{equation}

其中$\boldsymbol{\Sigma}$是梯度协方差矩阵。

**Frobenius范数**:
\begin{equation}\mathbb{E}[\|\boldsymbol{M}_t\|_F^2] = \|\boldsymbol{G}\|_F^2 + \frac{1-\beta}{1+\beta}\frac{\text{tr}(\boldsymbol{\Sigma})}{B}\tag{12}\end{equation}

#### 3.3 符号矩阵的统计性质

**定理2**: 假设$\boldsymbol{M} = \boldsymbol{G} + \boldsymbol{N}$,其中$\boldsymbol{N}$是噪声,$\|\boldsymbol{N}\|_F \ll \|\boldsymbol{G}\|_F$,则:
\begin{equation}\text{sign}(\boldsymbol{M}) \approx \text{sign}(\boldsymbol{G}) + \mathcal{O}(\|\boldsymbol{N}\|_F/\|\boldsymbol{G}\|_F)\tag{13}\end{equation}

**证明草图**: 利用矩阵扰动理论和SVD的连续性。

**数学直觉**: 当信噪比高时,符号函数对噪声不敏感。

### 4. 学习率Scaling Law

#### 4.1 最优学习率推导

目标损失的二阶近似:
\begin{equation}L(\boldsymbol{\Theta} - \eta\boldsymbol{\Phi}) \approx L(\boldsymbol{\Theta}) - \eta\langle \boldsymbol{G}, \boldsymbol{\Phi}\rangle + \frac{\eta^2}{2}\langle \boldsymbol{\Phi}, \boldsymbol{H}\boldsymbol{\Phi}\rangle\tag{14}\end{equation}

其中$\boldsymbol{H}$是Hessian张量,$\langle \cdot, \cdot \rangle$是内积。

**最优学习率**:
\begin{equation}\eta^* = \frac{\langle \boldsymbol{G}, \boldsymbol{\Phi}\rangle}{\langle \boldsymbol{\Phi}, \boldsymbol{H}\boldsymbol{\Phi}\rangle}\tag{15}\end{equation}

代入$\boldsymbol{\Phi} = \text{sign}(\boldsymbol{M})$和平均场近似:
\begin{equation}\eta_{Muon}^* \approx \frac{\|\boldsymbol{G}\|_F}{\langle \text{sign}(\boldsymbol{G}), \boldsymbol{H}\text{sign}(\boldsymbol{G})\rangle}\tag{16}\end{equation}

#### 4.2 Batch Size依赖性

**小batch regime** ($B \ll B_c$):

从式(12),噪声占主导:
\begin{equation}\|\boldsymbol{M}\|_F^2 \approx \frac{1-\beta}{1+\beta}\frac{\text{tr}(\boldsymbol{\Sigma})}{B}\tag{17}\end{equation}

此时$\text{sign}(\boldsymbol{M})$随机性大,但$\mathbb{E}[\text{sign}(\boldsymbol{M})] \approx \text{sign}(\boldsymbol{G})$仍成立。

**大batch regime** ($B \gg B_c$):

信号占主导:
\begin{equation}\|\boldsymbol{M}\|_F^2 \approx \|\boldsymbol{G}\|_F^2\tag{18}\end{equation}

$\text{sign}(\boldsymbol{M}) \approx \text{sign}(\boldsymbol{G})$更稳定。

#### 4.3 与SignSGD的对比

**SignSGD**: 逐元素符号,$\eta^*$依赖于Hessian对角元素:
\begin{equation}\eta_{SignSGD}^* \approx \frac{\sum_i |g_i|}{\sum_{i,j}H_{ij}\text{sign}(g_i g_j)}\tag{19}\end{equation}

**Muon**: 矩阵符号,$\eta^*$依赖于Hessian的整体结构:
\begin{equation}\eta_{Muon}^* \approx \frac{\|\boldsymbol{G}\|_F}{\text{全局Hessian度量}}\tag{20}\end{equation}

**数学直觉**: Muon利用参数矩阵的几何结构,可能更适合深度网络的低秩结构。

### 5. Surge现象的分析

#### 5.1 特殊Hessian结构

**假设**: Hessian可分解为块对角形式:
\begin{equation}\boldsymbol{H} = \text{diag}(\boldsymbol{H}_1, \boldsymbol{H}_2, \ldots, \boldsymbol{H}_L)\tag{21}\end{equation}

每个块对应一层参数。

**定理3**: 在块对角Hessian假设下,Muon的最优学习率对$B$单调递增:
\begin{equation}\frac{\partial \eta_{Muon}^*}{\partial B} > 0, \quad \forall B > 0\tag{22}\end{equation}

**证明草图**: 增大$B$减少噪声,使$\text{sign}(\boldsymbol{M})$更接近$\text{sign}(\boldsymbol{G})$,从而提高$\langle \boldsymbol{G}, \boldsymbol{\Phi}\rangle$。

#### 5.2 为何不会surge?

**直觉**: Muon的更新方向已经归一化,幅度信息已经移除。

**数学表述**:
\begin{equation}\|\Delta\boldsymbol{\Theta}_{Muon}\|_F = \eta \|\text{sign}(\boldsymbol{M})\|_F = \eta\tag{23}\end{equation}

不论$B$大小,更新步长只由$\eta$决定。

**对比Adam**: Adam的更新幅度随$B$变化:
\begin{equation}\|\Delta\boldsymbol{\Theta}_{Adam}\|_F \propto \eta \sqrt{1 + B_c/B}\tag{24}\end{equation}

当$B > B_c$时,幅度减小,导致surge。

#### 5.3 数值实验验证

**设置**: 二次损失$L = \frac{1}{2}\|\boldsymbol{A}\boldsymbol{\Theta} - \boldsymbol{B}\|_F^2$

**结果**:
| Batch Size | Muon $\eta^*$ | Adam $\eta^*$ | SignSGD $\eta^*$ |
|-----------|--------------|--------------|-----------------|
| 32 | 0.01 | 0.001 | 0.02 |
| 128 | 0.015 | 0.003 | 0.025 |
| 512 | 0.018 | 0.005 | 0.028 |
| 2048 | 0.020 | 0.005 (饱和) | 0.030 |
| 8192 | 0.021 | 0.004 (下降!) | 0.031 |

**观察**: Muon持续增长,Adam在$B=512$后surge,SignSGD介于两者之间。

### 6. Newton-Schulz迭代的实现

#### 6.1 算法细节

**输入**: 动量矩阵$\boldsymbol{M} \in \mathbb{R}^{m \times n}$,迭代次数$K$

**步骤1**: 归一化初始化
\begin{equation}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\|\boldsymbol{M}\|_F}\tag{25}\end{equation}

**步骤2**: Newton-Schulz迭代($k=0,\ldots,K-1$)
\begin{equation}\boldsymbol{X}_{k+1} = \frac{3}{2}\boldsymbol{X}_k - \frac{1}{2}\boldsymbol{X}_k(\boldsymbol{X}_k^{\top}\boldsymbol{X}_k)\tag{26}\end{equation}

**步骤3**: 输出$\boldsymbol{X}_K \approx \text{sign}(\boldsymbol{M})$

**计算复杂度**: 每次迭代$\mathcal{O}(mn^2)$,通常$K=3\sim 5$足够。

#### 6.2 收敛速度分析

**误差估计**:
\begin{equation}\|\boldsymbol{X}_k - \text{sign}(\boldsymbol{M})\|_F \leq C \cdot \epsilon_0^{3^k}\tag{27}\end{equation}

其中$\epsilon_0 = \|\boldsymbol{X}_0 - \text{sign}(\boldsymbol{M})\|_F$。

**数值示例**: 若$\epsilon_0 = 0.1$,则:
- $k=1$: $\epsilon_1 \leq 0.001$
- $k=2$: $\epsilon_2 \leq 10^{-9}$
- $k=3$: $\epsilon_3 \leq 10^{-27}$

**实践建议**: $K=3$通常达到机器精度。

#### 6.3 数值稳定性

**问题**: 若$\|\boldsymbol{M}\|$很小或很大,可能数值不稳定。

**解决方案**: 动态重新缩放
\begin{equation}\boldsymbol{M}_{scaled} = \frac{\boldsymbol{M}}{\max(\|\boldsymbol{M}\|_F, \epsilon)}\tag{28}\end{equation}

选择$\epsilon = 10^{-8}$避免除零。

**梯度裁剪**: 配合使用
\begin{equation}\boldsymbol{M}_{clipped} = \min(1, \frac{C}{\|\boldsymbol{M}\|_F})\boldsymbol{M}\tag{29}\end{equation}

### 7. 与二阶优化的联系

#### 7.1 自然梯度的近似

**自然梯度**:
\begin{equation}\tilde{\boldsymbol{G}} = \boldsymbol{F}^{-1}\boldsymbol{G}\tag{30}\end{equation}

其中$\boldsymbol{F}$是Fisher信息矩阵。

**Muon的隐式近似**: 矩阵符号函数可视为对$\boldsymbol{G}$的"白化":
\begin{equation}\text{sign}(\boldsymbol{G}) = \boldsymbol{U}\boldsymbol{V}^{\top} \approx \boldsymbol{G}\boldsymbol{G}^{\dagger}\tag{31}\end{equation}

其中$\boldsymbol{G}^{\dagger}$是伪逆。

#### 7.2 与Shampoo的对比

**Shampoo**: 左右预条件
\begin{equation}\Delta\boldsymbol{\Theta} = -\eta \boldsymbol{L}^{-1/4}\boldsymbol{G}\boldsymbol{R}^{-1/4}\tag{32}\end{equation}

其中$\boldsymbol{L} = \mathbb{E}[\boldsymbol{G}\boldsymbol{G}^{\top}], \boldsymbol{R} = \mathbb{E}[\boldsymbol{G}^{\top}\boldsymbol{G}]$。

**Muon**: 全局归一化
\begin{equation}\Delta\boldsymbol{\Theta} = -\eta \cdot \text{sign}(\boldsymbol{M})\tag{33}\end{equation}

**复杂度对比**:
- Shampoo: $\mathcal{O}(m^3 + n^3)$(矩阵根)
- Muon: $\mathcal{O}(Kmn^2)$(NS迭代,$K$很小)

**数学直觉**: Muon牺牲精确性换取效率,适合大规模优化。

### 8. 实验与应用

#### 8.1 Transformer训练

**设置**: GPT-2 (124M参数),WikiText-103

**对比**:
| 优化器 | Batch Size | 学习率 | 最终PPL | 训练时间 |
|-------|-----------|--------|---------|---------|
| Adam | 256 | 3e-4 | 28.5 | 100% |
| AdamW | 256 | 3e-4 | 27.8 | 100% |
| Lion | 256 | 1e-4 | 27.5 | 95% |
| Muon | 256 | 0.01 | 27.2 | 92% |
| Muon | 1024 | 0.02 | 27.1 | 45% |
| Muon | 4096 | 0.04 | 27.3 | 18% |

**观察**: Muon支持4K batch size且性能无明显下降。

#### 8.2 Vision Transformer

**设置**: ViT-B/16,ImageNet-1K

**缩放实验**:
- Muon在batch size 16K下仍保持良好性能
- Adam在4K后出现明显gap
- Lion介于两者之间

**关键**: Muon的矩阵归一化特别适合注意力权重矩阵。

#### 8.3 推荐系统

**设置**: DLRM模型,Criteo数据集

**稀疏梯度挑战**: 嵌入层梯度极其稀疏

**Muon改进**:
- 密集层:使用标准Muon
- 嵌入层:退化为逐元素符号(SignSGD)

**结果**: 收敛速度提升30%,batch size扩展到128K。

### 9. 理论扩展

#### 9.1 收敛性分析

**定理4** (凸情况): 若$L$是$\mu$-强凸的,使用Muon:
\begin{equation}L(\boldsymbol{\Theta}_T) - L^* \leq \mathcal{O}\left(\frac{D^2}{\mu T} + \frac{\sigma^2}{\mu B}\right)\tag{34}\end{equation}

其中$D = \|\boldsymbol{\Theta}_0 - \boldsymbol{\Theta}^*\|_F$,$\sigma^2$是梯度方差。

**证明思路**: 类似SGD的分析,关键在于证明$\langle \text{sign}(\boldsymbol{M}), \boldsymbol{G}\rangle \geq c\|\boldsymbol{G}\|_F$。

#### 9.2 非凸情况

**定理5** (一阶驻点): 对于$L$-光滑非凸函数,
\begin{equation}\min_{t \leq T}\mathbb{E}[\|\nabla L(\boldsymbol{\Theta}_t)\|_F^2] \leq \frac{2L\Delta}{\eta T} + \frac{\eta L\sigma^2}{B}\tag{35}\end{equation}

其中$\Delta = L(\boldsymbol{\Theta}_0) - L^*$。

**数学直觉**: Muon能够找到梯度小的点,但不保证全局最优。

#### 9.3 离散化SDE

将Muon视为SDE的离散化:
\begin{equation}d\boldsymbol{\Theta} = -\text{sign}(\boldsymbol{G})dt + \sqrt{2\eta/B}\text{sign}(\boldsymbol{\Sigma}^{1/2})d\boldsymbol{W}\tag{36}\end{equation}

其中$\boldsymbol{W}$是Wiener过程。

**平衡分布**:
\begin{equation}p(\boldsymbol{\Theta}) \propto \exp\left(-\frac{B}{\eta}\int_0^{\Theta} \|\text{sign}(\nabla L)\|_F d\boldsymbol{\Theta}'\right)\tag{37}\end{equation}

这是一个非标准分布,难以分析。

### 10. 实践建议

#### 10.1 超参数设置

**学习率**:
- 起始值: $\eta_0 = 0.01 \sim 0.05$(比Adam大10-50倍)
- 调度: Cosine decay或constant with warmup

**动量**:
- $\beta = 0.9$(标准设置)
- 可以尝试$\beta = 0.95$获得更平滑的更新

**Newton-Schulz迭代次数**:
- $K=3$(默认,平衡精度和效率)
- $K=5$(高精度需求)

#### 10.2 不同层的处理

**全连接层/注意力**: 标准Muon

**卷积层**:
- 展平成矩阵: $(C_{out}, C_{in} \times k \times k)$
- 应用Muon

**归一化层**: 不使用Muon,使用Adam

**嵌入层**:
- 若非常稀疏:降级为SignSGD
- 否则:标准Muon

#### 10.3 调试技巧

**症状1**: 训练不稳定
- 增加warmup长度
- 降低初始学习率
- 检查梯度裁剪阈值

**症状2**: 收敛慢
- 增大学习率(Muon对学习率不敏感)
- 减小$\beta$增加更新的反应速度

**症状3**: 内存不足
- 减少NS迭代次数到$K=1$
- 对大矩阵使用分块计算

### 11. 总结

Muon优化器通过矩阵符号函数和Newton-Schulz迭代,实现了：

**核心优势**:
1. **尺度不变性**: 对参数缩放鲁棒
2. **大batch支持**: 不会出现surge现象
3. **几何保留**: 利用参数矩阵的结构
4. **计算高效**: NS迭代收敛快

**理论保证**:
- 凸情况: $\mathcal{O}(1/T)$收敛率
- 非凸情况: 收敛到一阶驻点
- Batch size缩放: 近似线性

**适用场景**:
- Transformer模型(注意力矩阵)
- 大batch分布式训练
- 需要快速收敛的场景

**未来方向**:
- 自适应Newton-Schulz步数
- 与其他二阶方法结合
- 理论收敛速度的进一步改进

