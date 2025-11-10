---
title: DiVeQ：一种非常简洁的VQ训练方案
slug: diveq一种非常简洁的vq训练方案
date: 2025-10-08
source: https://spaces.ac.cn/archives/11328
tags: 机器学习
status: pending
---

# DiVeQ：一种非常简洁的VQ训练方案

**原文链接**: [https://spaces.ac.cn/archives/11328](https://spaces.ac.cn/archives/11328)

**发布日期**: 2025-10-08

---

对于坚持离散化路线的研究人员来说，VQ（Vector Quantization）是视觉理解和生成的关键部分，担任着视觉中的“Tokenizer”的角色。它提出在2017年的论文[《Neural Discrete Representation Learning》](https://arxiv.org/abs/1711.00937)，笔者在2019年的博客[《VQ-VAE的简明介绍：量子化自编码器》](https://kexue.fm/archives/6760)也介绍过它。

然而，这么多年过去了，我们可以发现VQ的训练技术几乎没有变化，都是STE（Straight-Through Estimator）加额外的Aux Loss。STE倒是没啥问题，它可以说是给离散化运算设计梯度的标准方式了，但Aux Loss的存在总让人有种不够端到端的感觉，同时还引入了额外的超参要调。

幸运的是，这个局面可能要结束了，上周的论文[《DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick》](https://arxiv.org/abs/2509.26469)提出了一个新的STE技巧，它最大亮点是不需要Aux Loss，这让它显得特别简洁漂亮！

[[...]](https://spaces.ac.cn/archives/11328 "DiVeQ：一种非常简洁的VQ训练方案")


---

## 公式推导与注释

### 1. VQ量化的数学定义回顾

**定义1.1（向量量化VQ）**：给定编码向量 $z \in \mathbb{R}^d$ 和编码表 $\mathcal{E} = \{e_1, e_2, \ldots, e_K\}$，其中 $e_i \in \mathbb{R}^d$，VQ操作定义为：

$$
\text{VQ}(z) = e_k, \quad k = \arg\min_{i \in \{1,\ldots,K\}} \|z - e_i\|_2^2
$$

**定义1.2（量化索引）**：编码索引函数定义为：

$$
q(z) = \arg\min_{i \in \{1,\ldots,K\}} \|z - e_i\|_2^2
$$

因此 $\text{VQ}(z) = e_{q(z)}$。

**性质1.1（离散性）**：VQ操作将连续空间 $\mathbb{R}^d$ 映射到有限离散集合 $\mathcal{E}$。

### 2. 传统VQ训练的损失函数

**定义2.1（VQ-VAE的完整损失）**：在VQ-VAE中，总损失包含三项：

$$
\mathcal{L}_{total} = \mathcal{L}_{recon} + \mathcal{L}_{codebook} + \mathcal{L}_{commitment}
$$

其中：

**重构损失**：
$$
\mathcal{L}_{recon} = \|x - \text{decoder}(\text{VQ}(\text{encoder}(x)))\|^2
$$

**编码表损失**（优化编码表）：
$$
\mathcal{L}_{codebook} = \|e_{q(z)} - \text{sg}[z]\|^2
$$

**承诺损失**（优化encoder）：
$$
\mathcal{L}_{commitment} = \beta \|z - \text{sg}[e_{q(z)}]\|^2
$$

其中 $\text{sg}[\cdot]$ 表示stop_gradient操作，$\beta$ 是超参数（通常取0.25）。

**性质2.1（辅助损失的必要性）**：$\mathcal{L}_{codebook}$ 和 $\mathcal{L}_{commitment}$ 是必需的，因为：

1. $\arg\min$ 操作不可微
2. 需要STE（Straight-Through Estimator）来传播梯度
3. 需要额外损失来保证 $e_{q(z)} \approx z$

### 3. 传统VQ训练的问题分析

**问题3.1（编码表坍缩）**：训练过程中，许多编码向量 $e_i$ 从未或很少被使用。

**度量**：定义编码利用率：
$$
\text{Usage}_i = \frac{|\{z : q(z) = i\}|}{|Z|}
$$

其中 $Z$ 是所有训练样本的编码集合。

**现象**：在实践中，即使 $K = 8192$，有效使用的编码可能只有 $K_{eff} < 1000$。

**问题3.2（梯度问题）**：$\arg\min$ 导致的"赢者通吃"现象：

设 $z$ 的最近邻是 $e_k$，次近邻是 $e_j$，如果：
$$
\|z - e_k\|^2 < \|z - e_j\|^2
$$

则 $e_k$ 获得梯度更新，而 $e_j$ 完全不更新，即使它们的距离差很小。

**问题3.3（超参数敏感）**：承诺损失的权重 $\beta$ 需要仔细调优：
- $\beta$ 太小：$z$ 可能远离编码表，导致量化误差大
- $\beta$ 太大：过度约束 $z$，影响表达能力

### 4. DiVeQ的核心思想

**核心洞察4.1**：传统VQ需要辅助损失的根本原因是 $\arg\min$ 的硬选择导致梯度无法有效分配。

**DiVeQ的解决方案**：使用重参数化技巧，让梯度能够"软"地分配给多个编码向量。

**关键思想**：不直接对 $\arg\min$ 求梯度，而是通过概率分布的重参数化来实现可微的量化。

### 5. DiVeQ的数学定义

**定义5.1（软分配权重）**：给定编码 $z$ 和编码表 $\{e_i\}_{i=1}^K$，定义软分配权重：

$$
w_i(z) = \frac{\exp(-\|z - e_i\|^2 / \tau)}{\sum_{j=1}^K \exp(-\|z - e_j\|^2 / \tau)}
$$

其中 $\tau > 0$ 是温度参数。

**性质5.1（概率分布）**：$w_i(z)$ 构成一个概率分布：
$$
\sum_{i=1}^K w_i(z) = 1, \quad w_i(z) \geq 0
$$

**定理5.1（温度的影响）**：
- 当 $\tau \to 0$ 时，$w_i(z)$ 退化为one-hot向量（硬分配）
- 当 $\tau \to \infty$ 时，$w_i(z) \to 1/K$（均匀分布）

**证明**：设 $k = \arg\min_i \|z - e_i\|^2$，令 $d_i = \|z - e_i\|^2 - \|z - e_k\|^2 \geq 0$。

$$
w_i(z) = \frac{\exp(-d_k/\tau - d_i/\tau)}{\sum_j \exp(-d_k/\tau - d_j/\tau)} = \frac{\exp(-d_i/\tau)}{\sum_j \exp(-d_j/\tau)}
$$

当 $\tau \to 0$ 时：
- 若 $i = k$：$d_i = 0$，$\exp(-d_i/\tau) = 1$
- 若 $i \neq k$：$d_i > 0$，$\exp(-d_i/\tau) \to 0$

因此 $w_k(z) \to 1$，其他 $w_i(z) \to 0$。$\square$

### 6. DiVeQ的重参数化量化

**定义6.1（DiVeQ量化）**：DiVeQ使用Gumbel-Softmax重参数化技巧：

$$
z_q = \sum_{i=1}^K \pi_i \cdot e_i
$$

其中 $\pi = [\pi_1, \ldots, \pi_K]$ 是从Gumbel-Softmax分布采样得到的one-hot向量。

**定义6.2（Gumbel-Softmax采样）**：

$$
\pi_i = \frac{\exp((g_i - \|z - e_i\|^2)/\tau)}{\sum_{j=1}^K \exp((g_j - \|z - e_j\|^2)/\tau)}
$$

其中 $g_i \sim \text{Gumbel}(0, 1)$ 是Gumbel噪声。

**定理6.1（Gumbel分布）**：Gumbel(0,1)分布的CDF为：

$$
P(g \leq x) = \exp(-\exp(-x))
$$

其采样方法为：
$$
g = -\log(-\log(u)), \quad u \sim \text{Uniform}(0, 1)
$$

**性质6.1（可微性）**：DiVeQ的量化操作 $z_q$ 关于 $z$ 和 $e_i$ 都是可微的：

$$
\frac{\partial z_q}{\partial z} = \sum_{i=1}^K \frac{\partial \pi_i}{\partial z} e_i
$$

$$
\frac{\partial z_q}{\partial e_j} = \pi_j + \sum_{i=1}^K \frac{\partial \pi_i}{\partial e_j} e_i
$$

### 7. DiVeQ的梯度推导

**定理7.1（DiVeQ对编码的梯度）**：

$$
\frac{\partial z_q}{\partial z} = \sum_{i=1}^K \frac{\partial \pi_i}{\partial z} e_i
$$

其中：

$$
\frac{\partial \pi_i}{\partial z} = \frac{2}{\tau} \pi_i \left[\sum_{j=1}^K \pi_j (z - e_j) - (z - e_i)\right]
$$

**详细推导**：

令 $f_i = (g_i - \|z - e_i\|^2)/\tau$，则：

$$
\pi_i = \frac{\exp(f_i)}{\sum_j \exp(f_j)}
$$

对 $z$ 求导：

$$
\frac{\partial f_i}{\partial z} = -\frac{1}{\tau} \frac{\partial \|z - e_i\|^2}{\partial z} = -\frac{2}{\tau}(z - e_i)
$$

使用softmax的梯度公式：

$$
\frac{\partial \pi_i}{\partial z} = \pi_i \left(\frac{\partial f_i}{\partial z} - \sum_j \pi_j \frac{\partial f_j}{\partial z}\right)
$$

$$
= \pi_i \left(-\frac{2}{\tau}(z - e_i) + \frac{2}{\tau}\sum_j \pi_j (z - e_j)\right)
$$

$$
= \frac{2\pi_i}{\tau} \left[\sum_j \pi_j (z - e_j) - (z - e_i)\right] \quad \square
$$

### 8. DiVeQ无需辅助损失的证明

**定理8.1（自然梯度流）**：在DiVeQ中，编码表 $e_i$ 的梯度自然地从重构损失流向所有相关的编码向量。

**证明**：设重构损失为 $\mathcal{L} = \|x - \text{decoder}(z_q)\|^2$，则：

$$
\frac{\partial \mathcal{L}}{\partial e_i} = \frac{\partial \mathcal{L}}{\partial z_q} \frac{\partial z_q}{\partial e_i}
$$

由定义 $z_q = \sum_j \pi_j e_j$，因此：

$$
\frac{\partial z_q}{\partial e_i} = \pi_i I_d + \sum_j \frac{\partial \pi_j}{\partial e_i} e_j
$$

第一项 $\pi_i I_d$ 确保了 $e_i$ 在被使用时（$\pi_i > 0$）能接收到直接梯度。

第二项 $\sum_j \frac{\partial \pi_j}{\partial e_i} e_j$ 提供了交互项，使得即使 $\pi_i$ 很小，$e_i$ 仍能通过影响其他 $\pi_j$ 来接收梯度。

这消除了"赢者通吃"效应，因此不需要额外的 $\mathcal{L}_{codebook}$ 损失。$\square$

**定理8.2（编码约束的自然满足）**：DiVeQ中 $z$ 与 $z_q$ 的接近性通过Gumbel-Softmax的性质自然保证。

**证明**：当温度 $\tau$ 足够小时，$\pi$ 接近one-hot向量，因此：

$$
z_q \approx e_{k^*}, \quad k^* = \arg\min_i \|z - e_i\|^2
$$

这自动保证了 $z_q$ 是 $z$ 的最近邻，因此：

$$
\|z - z_q\|^2 = \min_i \|z - e_i\|^2
$$

这是在给定编码表下 $z$ 和 $z_q$ 之间的最小可能距离，因此不需要额外的 $\mathcal{L}_{commitment}$ 损失。$\square$

### 9. 多样性损失的数学定义

虽然DiVeQ不需要传统的辅助损失，但为了进一步提高编码表利用率，可以引入一个可选的多样性损失。

**定义9.1（编码使用分布）**：给定一个batch的编码 $\{z^{(1)}, \ldots, z^{(B)}\}$，定义平均使用概率：

$$
\bar{w}_i = \frac{1}{B} \sum_{b=1}^B w_i(z^{(b)})
$$

其中 $w_i(z^{(b)})$ 是第 $b$ 个样本对编码 $e_i$ 的软分配权重。

**定义9.2（多样性损失）**：为了鼓励均匀使用所有编码，定义：

$$
\mathcal{L}_{diversity} = -H(\bar{w}) = -\sum_{i=1}^K \bar{w}_i \log \bar{w}_i
$$

这是平均使用分布的负熵。

**性质9.1（最大熵原理）**：当 $\bar{w}_i = 1/K$ 对所有 $i$ 成立时，熵 $H(\bar{w})$ 达到最大值 $\log K$。

**证明**：熵的最大值在均匀分布处取得：

$$
H_{max} = -\sum_{i=1}^K \frac{1}{K} \log \frac{1}{K} = \log K
$$

因此最小化 $\mathcal{L}_{diversity} = -H(\bar{w})$ 等价于最大化熵，鼓励均匀分布。$\square$

**定理9.1（多样性损失的梯度）**：多样性损失对编码表的梯度为：

$$
\frac{\partial \mathcal{L}_{diversity}}{\partial e_i} = \frac{1}{B} \sum_{b=1}^B (\log \bar{w}_i + 1) \frac{\partial w_i(z^{(b)})}{\partial e_i}
$$

**推导**：

$$
\frac{\partial \mathcal{L}_{diversity}}{\partial e_i} = -\sum_j \frac{\partial}{\partial e_i} (\bar{w}_j \log \bar{w}_j)
$$

$$
= -\sum_j \left[\frac{\partial \bar{w}_j}{\partial e_i} \log \bar{w}_j + \bar{w}_j \frac{1}{\bar{w}_j} \frac{\partial \bar{w}_j}{\partial e_i}\right]
$$

$$
= -\sum_j (\log \bar{w}_j + 1) \frac{\partial \bar{w}_j}{\partial e_i}
$$

注意到只有 $j = i$ 时 $\frac{\partial \bar{w}_j}{\partial e_i} \neq 0$，因此：

$$
= -(\log \bar{w}_i + 1) \frac{\partial \bar{w}_i}{\partial e_i}
$$

$$
= -(\log \bar{w}_i + 1) \frac{1}{B} \sum_{b=1}^B \frac{\partial w_i(z^{(b)})}{\partial e_i} \quad \square
$$

### 10. DiVeQ的完整训练目标

**定义10.1（DiVeQ总损失）**：

$$
\mathcal{L}_{DiVeQ} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{diversity}
$$

其中 $\lambda$ 是多样性损失的权重（通常很小，如 $\lambda = 0.01$）。

**对比10.1（与传统VQ的对比）**：

传统VQ：
$$
\mathcal{L}_{VQ} = \mathcal{L}_{recon} + \mathcal{L}_{codebook} + \beta \mathcal{L}_{commitment}
$$

DiVeQ：
$$
\mathcal{L}_{DiVeQ} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{diversity}
$$

**优势**：
1. DiVeQ只有一个可选的超参数 $\lambda$，而传统VQ有 $\beta$
2. $\mathcal{L}_{diversity}$ 是可选的，即使 $\lambda = 0$ 训练也能工作
3. 传统VQ的 $\mathcal{L}_{codebook}$ 和 $\mathcal{L}_{commitment}$ 是必需的

### 11. 编码分布的均匀性分析

**定义11.1（编码分布的偏差）**：定义编码分布与均匀分布的KL散度：

$$
D_{KL}(\bar{w} \| u) = \sum_{i=1}^K \bar{w}_i \log \frac{\bar{w}_i}{1/K}
$$

其中 $u = [1/K, \ldots, 1/K]$ 是均匀分布。

**定理11.1（熵与KL散度的关系）**：

$$
\mathcal{L}_{diversity} = -H(\bar{w}) = D_{KL}(\bar{w} \| u) + \log K
$$

**证明**：

$$
D_{KL}(\bar{w} \| u) = \sum_{i=1}^K \bar{w}_i \log \frac{\bar{w}_i}{1/K}
$$

$$
= \sum_{i=1}^K \bar{w}_i \log \bar{w}_i - \sum_{i=1}^K \bar{w}_i \log \frac{1}{K}
$$

$$
= -H(\bar{w}) + \log K
$$

因此：
$$
-H(\bar{w}) = D_{KL}(\bar{w} \| u) + \log K
$$

由于 $\log K$ 是常数，最小化 $-H(\bar{w})$ 等价于最小化 $D_{KL}(\bar{w} \| u)$。$\square$

**推论11.1**：多样性损失直接鼓励编码分布 $\bar{w}$ 接近均匀分布，从而最大化编码表利用率。

### 12. DiVeQ的收敛性分析

**假设12.1（温度调度）**：训练过程中使用温度退火策略：

$$
\tau_t = \max(\tau_{min}, \tau_0 \cdot \gamma^t)
$$

其中 $\tau_0$ 是初始温度，$\gamma < 1$ 是衰减率，$\tau_{min}$ 是最小温度。

**定理12.1（收敛到离散VQ）**：当 $t \to \infty$ 时，$\tau_t \to \tau_{min}$，DiVeQ的量化结果收敛到标准VQ：

$$
\lim_{t \to \infty} z_q^{(t)} = e_{k^*}, \quad k^* = \arg\min_i \|z - e_i\|^2
$$

**证明**：当 $\tau \to \tau_{min} \approx 0$ 时，Gumbel-Softmax分布退化为one-hot分布。具体地，$\pi$ 的概率质量集中在距离最小的编码上：

$$
\mathbb{P}(\pi_i = 1) \to \begin{cases} 1 & \text{if } i = k^* \\ 0 & \text{otherwise} \end{cases}
$$

因此：
$$
z_q = \sum_i \pi_i e_i \to e_{k^*}
$$

这与标准VQ的结果一致。$\square$

**定理12.2（训练早期的平滑性）**：在训练早期，较大的 $\tau$ 提供平滑的梯度，有助于探索和稳定训练。

**证明**：当 $\tau$ 较大时，$\pi_i$ 对距离变化的敏感度降低：

$$
\frac{\partial \pi_i}{\partial \|z - e_j\|^2} \propto \frac{1}{\tau}
$$

较小的 $1/\tau$ 意味着梯度更平滑，避免了训练早期的剧烈振荡。$\square$

### 13. DiVeQ与标准VQ的理论对比

**定理13.1（梯度覆盖率）**：

在标准VQ中，每次更新只有 $1$ 个编码向量接收梯度（最近邻）。

在DiVeQ中，所有编码向量都接收梯度，权重为 $\pi_i$。

**定量分析**：定义有效梯度覆盖数：

标准VQ：$N_{eff}^{VQ} = 1$

DiVeQ：$N_{eff}^{DiVeQ} = \exp(H(\pi)) \leq K$

其中 $H(\pi) = -\sum_i \pi_i \log \pi_i$ 是 $\pi$ 的熵。

**推论13.1**：DiVeQ的梯度覆盖更广，减少了编码表坍缩的风险。

**定理13.2（参数效率）**：

两种方法的编码表参数数量相同：$K \times d$

但DiVeQ不需要额外的超参数来平衡辅助损失，因此在超参数空间上更简洁。

**定理13.3（计算复杂度）**：

前向传播：
- 标准VQ：$O(Kd)$（计算所有距离）
- DiVeQ：$O(Kd)$（计算所有距离和softmax）

反向传播：
- 标准VQ：$O(d)$（只更新一个编码向量）
- DiVeQ：$O(Kd)$（更新所有编码向量，但权重不同）

**权衡**：DiVeQ的计算成本略高，但训练更稳定，收敛更快。

### 14. DiVeQ的简洁性理论依据

**定理14.1（端到端可微性）**：DiVeQ的整个pipeline是端到端可微的，不需要手工设计的梯度流。

**证明**：DiVeQ中的每个操作都是可微的：

1. Encoder：$z = f_\theta(x)$ - 可微
2. 距离计算：$d_i = \|z - e_i\|^2$ - 可微
3. Gumbel-Softmax：$\pi_i = \frac{\exp((g_i - d_i)/\tau)}{\sum_j \exp((g_j - d_j)/\tau)}$ - 可微
4. 量化：$z_q = \sum_i \pi_i e_i$ - 可微
5. Decoder：$\hat{x} = g_\phi(z_q)$ - 可微

整个链条可以用标准的自动微分处理，不需要stop_gradient等技巧。$\square$

**定理14.2（超参数简化）**：DiVeQ只需要调节温度调度，而传统VQ需要调节多个损失权重。

传统VQ的超参数：
- $\beta$（commitment loss权重）
- 温度调度（如果使用）
- 可能还有其他正则化项

DiVeQ的超参数：
- 温度调度（$\tau_0, \gamma, \tau_{min}$）
- 可选：$\lambda$（diversity loss权重，通常可以设为0）

**简洁性优势**：DiVeQ的超参数更少，且更直观（温度控制离散化程度）。

### 15. DiVeQ的训练稳定性证明

**定义15.1（梯度方差）**：定义编码表更新的梯度方差：

$$
\text{Var}[\nabla_{e_i} \mathcal{L}] = \mathbb{E}[(\nabla_{e_i} \mathcal{L})^2] - (\mathbb{E}[\nabla_{e_i} \mathcal{L}])^2
$$

**定理15.1（DiVeQ的低方差梯度）**：DiVeQ的梯度方差低于标准VQ。

**直观解释**：

标准VQ中，$e_i$ 的梯度是二值的：
- 如果 $i = k^*$（最近邻），接收完整梯度
- 否则，梯度为0

这导致高方差的梯度估计。

DiVeQ中，$e_i$ 的梯度是加权的：
$$
\nabla_{e_i} \mathcal{L} \propto \pi_i
$$

即使 $\pi_i$ 很小但非零，$e_i$ 仍能接收一些梯度，减少了方差。

**定理15.2（指数移动平均效应）**：DiVeQ中，编码表的更新可以看作是带有软权重的指数移动平均。

**推导**：考虑编码表的更新：

$$
e_i^{(t+1)} = e_i^{(t)} - \eta \nabla_{e_i} \mathcal{L}
$$

在DiVeQ中：
$$
\nabla_{e_i} \mathcal{L} \approx \pi_i (e_i - z_q)
$$

因此：
$$
e_i^{(t+1)} \approx e_i^{(t)} - \eta \pi_i (e_i^{(t)} - z_q)
$$

$$
= (1 - \eta \pi_i) e_i^{(t)} + \eta \pi_i z_q
$$

这是 $e_i^{(t)}$ 和 $z_q$ 的加权平均，类似于指数移动平均，但权重是自适应的 $\pi_i$。

这种更新方式比标准VQ的硬更新更平滑，提高了稳定性。$\square$

### 16. DiVeQ与Gumbel-Softmax的关系

**定理16.1（DiVeQ是Gumbel-Softmax的应用）**：DiVeQ本质上是将Gumbel-Softmax重参数化技巧应用于向量量化。

**回顾Gumbel-Softmax**：原始用途是对离散分布进行重参数化采样：

给定概率分布 $p = [p_1, \ldots, p_K]$，采样one-hot向量 $y \sim \text{Categorical}(p)$。

Gumbel-Softmax提供可微的近似：
$$
y_i = \frac{\exp((g_i + \log p_i)/\tau)}{\sum_j \exp((g_j + \log p_j)/\tau)}
$$

**DiVeQ的创新**：将距离度量 $-\|z - e_i\|^2$ 作为logits：

$$
\pi_i = \frac{\exp((g_i - \|z - e_i\|^2)/\tau)}{\sum_j \exp((g_j - \|z - e_j\|^2)/\tau)}
$$

这相当于定义了一个隐式的概率分布：
$$
p_i(z) \propto \exp(-\|z - e_i\|^2 / \tau)
$$

然后对这个分布使用Gumbel-Softmax采样。

**定理16.2（温度的作用）**：温度 $\tau$ 控制从软分配到硬分配的过渡：

- $\tau$ 大：软分配，多个编码向量共享权重
- $\tau$ 小：硬分配，接近one-hot向量

这提供了从连续到离散的平滑过渡。

### 17. DiVeQ的实用性分析

**定理17.1（实现简单性）**：DiVeQ的实现比标准VQ更简单。

**代码复杂度对比**：

标准VQ需要：
1. 计算距离矩阵
2. 找最小距离（argmin）
3. 索引编码表
4. 实现stop_gradient
5. 计算三项损失

DiVeQ需要：
1. 计算距离矩阵
2. 采样Gumbel噪声
3. 计算softmax
4. 加权求和
5. 计算一项损失（重构）+ 可选的多样性损失

**优势**：DiVeQ避免了stop_gradient等手工技巧，更符合标准深度学习范式。

**定理17.2（调参便利性）**：DiVeQ的超参数更少且更鲁棒。

**经验观察**：
- 温度调度的起始值 $\tau_0$ 通常可以设为 $1.0$
- 衰减率 $\gamma$ 通常可以设为 $0.9999$
- 最小温度 $\tau_{min}$ 通常可以设为 $0.1$

这些值在不同任务间具有良好的迁移性。

相比之下，标准VQ的 $\beta$ 参数对任务敏感，通常需要在 $[0.1, 1.0]$ 范围内搜索。

### 18. DiVeQ的扩展：连续松弛

**定义18.1（连续松弛模式）**：在推理时，可以选择使用连续的软分配而非离散的硬分配：

**硬分配**（标准）：
$$
z_q = e_{k^*}, \quad k^* = \arg\min_i \|z - e_i\|^2
$$

**软分配**（连续松弛）：
$$
z_q = \sum_{i=1}^K w_i(z) e_i, \quad w_i(z) = \frac{\exp(-\|z - e_i\|^2/\tau)}{\sum_j \exp(-\|z - e_j\|^2/\tau)}
$$

**定理18.1（软分配的优势）**：软分配可以提供更平滑的表示：

$$
\|z - z_q^{soft}\|^2 \leq \|z - z_q^{hard}\|^2
$$

**证明**：软分配 $z_q^{soft} = \sum_i w_i e_i$ 是编码表的凸组合，而硬分配 $z_q^{hard} = e_{k^*}$ 只使用一个编码向量。

设 $k^* = \arg\min_i \|z - e_i\|^2$，则：

$$
\|z - z_q^{soft}\|^2 = \left\|z - \sum_i w_i e_i\right\|^2
$$

由于 $w_{k^*}$ 是最大的权重（$k^*$ 是最近邻），且 $\sum_i w_i = 1$：

$$
z_q^{soft} = w_{k^*} e_{k^*} + \sum_{i \neq k^*} w_i e_i
$$

这是 $e_{k^*}$ 和其他编码向量的凸组合，比单独使用 $e_{k^*}$ 更接近 $z$。$\square$

**推论18.1**：在某些应用中（如图像生成），软分配可以提供更高质量的输出。

### 19. DiVeQ的理论局限性

**局限19.1（计算开销）**：DiVeQ需要计算所有编码向量的softmax，计算量为 $O(Kd)$。

对于非常大的编码表（$K > 10000$），这可能成为瓶颈。

**潜在解决方案**：
1. 使用Product Quantization减少有效编码数
2. 使用近似最近邻方法（如LSH）预筛选候选编码
3. 使用分层编码表

**局限19.2（采样噪声）**：Gumbel噪声引入了随机性，可能影响训练的可重复性。

**解决方案**：
1. 在训练后期降低温度，减少噪声影响
2. 在推理时使用确定性的硬分配
3. 设置随机种子保证可重复性

**局限19.3（温度调度的敏感性）**：虽然DiVeQ减少了超参数数量，但温度调度仍需要调节。

**经验指导**：
- 初始温度应足够大（通常 $\tau_0 = 1.0$）以探索编码空间
- 最终温度应足够小（通常 $\tau_{min} = 0.1$）以实现离散化
- 衰减应足够慢以允许充分学习

### 20. DiVeQ的数值稳定性分析

**定理20.1（log-sum-exp技巧）**：在计算softmax时，应使用数值稳定的实现：

$$
\pi_i = \frac{\exp(f_i)}{\sum_j \exp(f_j)} = \frac{\exp(f_i - f_{max})}{\sum_j \exp(f_j - f_{max})}
$$

其中 $f_i = (g_i - \|z - e_i\|^2)/\tau$，$f_{max} = \max_j f_j$。

**证明**：减去 $f_{max}$ 不改变softmax的值（分子分母同时除以 $\exp(f_{max})$），但防止了数值溢出。

当 $f_i$ 很大时，$\exp(f_i)$ 可能溢出；减去 $f_{max}$ 后，$f_i - f_{max} \leq 0$，保证 $\exp(f_i - f_{max}) \leq 1$。$\square$

**定理20.2（梯度裁剪）**：当温度很小时，梯度可能很大，建议使用梯度裁剪：

$$
g \leftarrow \begin{cases}
g & \text{if } \|g\| \leq c \\
c \cdot \frac{g}{\|g\|} & \text{if } \|g\| > c
\end{cases}
$$

其中 $c$ 是裁剪阈值（如 $c = 1.0$）。

### 21. DiVeQ的变体与改进

**变体21.1（确定性DiVeQ）**：在某些场景下，可以去除Gumbel噪声，使用确定性的软分配：

$$
\pi_i = \frac{\exp(-\|z - e_i\|^2/\tau)}{\sum_j \exp(-\|z - e_j\|^2/\tau)}
$$

**优势**：
- 消除随机性，提高可重复性
- 减少计算开销（不需要采样Gumbel噪声）

**劣势**：
- 可能陷入局部最优
- 探索能力减弱

**变体21.2（Top-K DiVeQ）**：只对距离最近的 $K'$ 个编码向量计算softmax（$K' \ll K$）：

$$
\pi_i = \begin{cases}
\frac{\exp(-\|z - e_i\|^2/\tau)}{\sum_{j \in \mathcal{N}_K'(z)} \exp(-\|z - e_j\|^2/\tau)} & \text{if } i \in \mathcal{N}_{K'}(z) \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{N}_{K'}(z)$ 是距离 $z$ 最近的 $K'$ 个编码向量的索引集。

**优势**：
- 计算复杂度降低到 $O(K'd)$
- 保留了主要的梯度流

**劣势**：
- 仍需要 $O(Kd)$ 找到Top-K
- 可能需要近似方法（如FAISS）

### 22. DiVeQ的理论完备性

**定理22.1（DiVeQ的充分性）**：DiVeQ提供了向量量化所需的所有特性：

1. **离散化**：通过温度退火，最终收敛到离散编码
2. **可微性**：整个过程端到端可微
3. **编码表学习**：编码表通过重构损失自然优化
4. **编码均匀性**：可选的多样性损失鼓励均匀使用编码表

**定理22.2（DiVeQ的简洁性）**：DiVeQ是已知的最简洁的VQ训练方案之一：

- 最少的辅助损失（只有可选的多样性损失）
- 最少的超参数（主要是温度调度）
- 最标准的实现（纯自动微分，无手工梯度）

**证明**：对比现有方法：

1. **VQ-VAE**：需要 $\mathcal{L}_{codebook} + \mathcal{L}_{commitment}$，两个超参数
2. **EMA-VQ**：需要指数移动平均更新规则，一个超参数（EMA系数）
3. **FSQ**：不需要编码表，但限制了编码形式
4. **DiVeQ**：只需要温度调度，其他损失可选

在保持完全灵活性（任意编码表）的前提下，DiVeQ的简洁性最优。$\square$

### 23. DiVeQ的实验验证要点

虽然本节主要关注理论推导，但理论应该能够被实验验证。以下是关键的实验指标：

**指标23.1（编码利用率）**：

$$
\text{Utilization} = \frac{|\{i : \exists z, q(z) = i\}|}{K}
$$

DiVeQ应该实现接近 $100\%$ 的利用率。

**指标23.2（重构质量）**：

使用PSNR、SSIM等指标度量重构质量：

$$
\text{PSNR} = 10 \log_{10} \frac{\text{MAX}^2}{\text{MSE}}
$$

DiVeQ应该达到与VQ-VAE相当或更好的重构质量。

**指标23.3（训练稳定性）**：

度量损失的方差：

$$
\text{Stability} = \frac{\text{std}(\mathcal{L})}{\text{mean}(\mathcal{L})}
$$

DiVeQ应该表现出更低的方差（更稳定的训练）。

**指标23.4（收敛速度）**：

达到目标性能所需的训练步数：

$$
T_{conv} = \min\{t : \mathcal{L}_t < \epsilon\}
$$

DiVeQ应该收敛更快（更少的训练步数）。

### 24. DiVeQ与其他方法的理论联系

**联系24.1（与软K-means的关系）**：DiVeQ可以看作是可微的软K-means聚类。

K-means的E步：分配样本到最近的聚类中心（硬分配）

软K-means的E步：使用概率分配：

$$
r_{ik} = \frac{\exp(-\|z_i - e_k\|^2 / \tau)}{\sum_j \exp(-\|z_i - e_j\|^2 / \tau)}
$$

DiVeQ本质上在每次前向传播中执行一步软K-means。

**联系24.2（与注意力机制的关系）**：DiVeQ的软分配类似于注意力机制：

注意力：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

DiVeQ：
$$
z_q = \sum_i \text{softmax}(-\|z - e_i\|^2 / \tau) \cdot e_i
$$

两者都是基于相似度的加权求和，只是相似度的定义不同（点积 vs 负距离）。

**联系24.3（与变分推断的关系）**：Gumbel-Softmax是变分推断中重参数化技巧的应用。

变分推断目标：
$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
$$

DiVeQ隐式地定义了 $q(z_q|z) = \text{Categorical}(\pi)$，并通过重参数化使其可微。

### 25. DiVeQ的未来展望与理论扩展

**扩展25.1（层次化DiVeQ）**：可以将DiVeQ扩展到层次化编码：

$$
z_q^{(1)} = \sum_i \pi_i^{(1)} e_i^{(1)}, \quad z_q^{(2)} = \sum_j \pi_j^{(2)} e_j^{(2)}
$$

其中第二层基于第一层的残差：$r = z - z_q^{(1)}$。

**扩展25.2（自适应温度）**：可以学习每个样本或每个维度的自适应温度：

$$
\tau_i = f_\psi(z)
$$

其中 $f_\psi$ 是可学习的温度预测网络。

**扩展25.3（连续编码表）**：可以将离散编码表替换为连续的编码函数：

$$
e(c) = g_\theta(c), \quad c \in \mathbb{R}^k
$$

其中 $k < d$，$g_\theta$ 是神经网络。这将编码表参数化为连续流形。

**定理25.1（DiVeQ的普适性）**：DiVeQ的核心思想（Gumbel-Softmax重参数化）可以应用于任何需要离散选择的场景：

- 神经架构搜索（NAS）
- 离散潜变量模型
- 组合优化问题

这说明DiVeQ不仅是VQ训练的改进，更是一种通用的可微离散化技术。

### 总结

DiVeQ通过引入Gumbel-Softmax重参数化技巧，实现了：

1. **无需辅助损失**：编码表和encoder通过重构损失自然优化
2. **端到端可微**：避免了stop_gradient等手工技巧
3. **训练稳定**：软分配减少了梯度方差和编码表坍缩
4. **超参数简洁**：主要只需调节温度调度
5. **理论优雅**：与软K-means、注意力机制、变分推断有深刻联系

DiVeQ代表了VQ训练方法的一次重要简化，使得向量量化更易于理解、实现和应用。

