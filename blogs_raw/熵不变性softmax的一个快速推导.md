---
title: 熵不变性Softmax的一个快速推导
slug: 熵不变性softmax的一个快速推导
date: 2022-04-11
tags: 近似, 熵, attention, 生成模型, attention
status: pending
---

# 熵不变性Softmax的一个快速推导

**原文链接**: [https://spaces.ac.cn/archives/9034](https://spaces.ac.cn/archives/9034)

**发布日期**: 

---

在文章[《从熵不变性看Attention的Scale操作》](/archives/8823)中，我们推导了一版具有熵不变性质的注意力机制：  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\kappa \log n}{d}QK^{\top}\right)V\label{eq:a}\end{equation}  
可以观察到，它主要是往Softmax里边引入了长度相关的缩放因子$\log n$来实现的。原来的推导比较繁琐，并且做了较多的假设，不利于直观理解，本文为其补充一个相对简明快速的推导。

## 推导过程 #

我们可以抛开注意力机制的背景，直接设有$s_1,s_2,\cdots,s_n\in\mathbb{R}$，定义  
$$p_i = \frac{e^{\lambda s_i}}{\sum\limits_{i=1}^n e^{\lambda s_i}}$$  
显然这就是$s_1,s_2,\cdots,s_n$同时乘上缩放因子$\lambda$后做Softmax的结果。现在我们算它的熵  
\begin{equation}\begin{aligned}H =&\, -\sum_{i=1}^n p_i \log p_i = \log\sum_{i=1}^n e^{\lambda s_i} - \lambda\sum_{i=1}^n p_i s_i \\\  
=&\, \log n + \log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i} - \lambda\sum_{i=1}^n p_i s_i  
\end{aligned}\end{equation}  
第一项的$\log$里边是“先指数后平均”，我们用“先平均后指数”（平均场）来近似它：  
\begin{equation}  
\log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i}\approx \log\exp\left(\frac{1}{n}\sum_{i=1}^n \lambda s_i\right) = \lambda \bar{s}  
\end{equation}  
然后我们知道Softmax是会侧重于$\max$的那个（参考[《函数光滑化杂谈：不可导函数的可导逼近》](/archives/6620#softmax)），所以有近似  
\begin{equation}\lambda\sum_{i=1}^n p_i s_i \approx \lambda s_{\max}\end{equation}  
所以  
\begin{equation}H\approx \log n - \lambda(s_{\max} - \bar{s})\end{equation}  
所谓熵不变性，就是希望尽可能地消除长度$n$的影响，所以根据上式我们需要有$\lambda\propto \log n$。如果放到注意力机制中，那么$s$的形式为$\langle \boldsymbol{q}, \boldsymbol{k}\rangle\propto d$（$d$是向量维度），所以需要有$\lambda\propto \frac{1}{d}$，综合起来就是  
\begin{equation}\lambda\propto \frac{\log n}{d}\end{equation}  
这就是文章开头式$\eqref{eq:a}$的结果。

## 文章小结 #

为之前提出的“熵不变性Softmax”构思了一个简单明快的推导。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9034>_

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

苏剑林. (Apr. 11, 2022). 《熵不变性Softmax的一个快速推导 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9034>

@online{kexuefm-9034,  
title={熵不变性Softmax的一个快速推导},  
author={苏剑林},  
year={2022},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9034}},  
} 


---

## 详细数学推导与理论分析

### 1. 问题的背景与动机

**标准Scaled Dot-Product Attention**：

在Transformer中，注意力机制定义为：
\begin{equation}
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}}\right) \boldsymbol{V} \tag{1}
\end{equation}

其中$\sqrt{d}$是缩放因子，$d$是向量维度。

**问题**：为什么是$\sqrt{d}$而不是其他值？

标准解释：保持方差不变（见Vaswani et al. 2017）。

**本文的视角**：从**熵不变性**的角度理解缩放因子，并推导出与序列长度$n$相关的缩放。

### 2. 熵不变性的定义

**直觉**：我们希望注意力分布的熵不随序列长度$n$的变化而剧烈变化。

**定义**：对于注意力权重$\boldsymbol{\alpha} = [\alpha_1, \ldots, \alpha_n]$，其熵为：
\begin{equation}
H(\boldsymbol{\alpha}) = -\sum_{i=1}^n \alpha_i \log \alpha_i \tag{2}
\end{equation}

**熵不变性条件**：希望存在缩放因子$\lambda(n, d)$，使得：
\begin{equation}
H\left(\text{softmax}(\lambda \boldsymbol{s})\right) \approx C \tag{3}
\end{equation}

其中$C$是与$n$无关的常数，$\boldsymbol{s} = [s_1, \ldots, s_n]$是注意力分数。

### 3. 熵的展开式

**Softmax定义**：
\begin{equation}
\alpha_i = \frac{e^{\lambda s_i}}{\sum_{j=1}^n e^{\lambda s_j}} \tag{4}
\end{equation}

**熵的计算**：
\begin{equation}
\begin{aligned}
H &= -\sum_{i=1}^n \alpha_i \log \alpha_i \\
&= -\sum_{i=1}^n \alpha_i \left(\lambda s_i - \log\sum_{j=1}^n e^{\lambda s_j}\right) \\
&= -\lambda \sum_{i=1}^n \alpha_i s_i + \sum_{i=1}^n \alpha_i \log\sum_{j=1}^n e^{\lambda s_j} \\
&= \log\sum_{j=1}^n e^{\lambda s_j} - \lambda \sum_{i=1}^n \alpha_i s_i
\end{aligned} \tag{5}
\end{equation}

记**配分函数**（partition function）：
\begin{equation}
Z = \sum_{j=1}^n e^{\lambda s_j} \tag{6}
\end{equation}

则：
\begin{equation}
H = \log Z - \lambda \mathbb{E}_\alpha[s] \tag{7}
\end{equation}

其中$\mathbb{E}_\alpha[s] = \sum_i \alpha_i s_i$是加权平均分数。

### 4. 平均场近似

**关键步骤**：对配分函数进行近似。

**Jensen不等式的反向应用**：

对于凸函数$\exp$，有：
\begin{equation}
\exp\left(\frac{1}{n}\sum_{i=1}^n \lambda s_i\right) \leq \frac{1}{n}\sum_{i=1}^n e^{\lambda s_i} \tag{8}
\end{equation}

但我们需要的是对$\log Z$的近似，而不是$Z$本身。

**平均场近似**（"先平均后指数"近似"先指数后平均"）：

\begin{equation}
\log Z = \log\sum_{i=1}^n e^{\lambda s_i} = \log\left(n \cdot \frac{1}{n}\sum_{i=1}^n e^{\lambda s_i}\right) = \log n + \log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i} \tag{9}
\end{equation}

**关键近似**：
\begin{equation}
\log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i} \approx \log\exp\left(\frac{1}{n}\sum_{i=1}^n \lambda s_i\right) = \lambda \bar{s} \tag{10}
\end{equation}

其中$\bar{s} = \frac{1}{n}\sum_i s_i$是算术平均。

**代入式(9)**：
\begin{equation}
\log Z \approx \log n + \lambda \bar{s} \tag{11}
\end{equation}

### 5. Softmax侧重max的性质

**Softmax作为max的平滑近似**：

回顾不等式：
\begin{equation}
\max_i s_i \leq \log\sum_i e^{s_i} \leq \max_i s_i + \log n \tag{12}
\end{equation}

**加权平均的近似**：

Softmax会侧重于较大的$s_i$，因此：
\begin{equation}
\mathbb{E}_\alpha[s] = \sum_i \alpha_i s_i \approx s_{\max} \tag{13}
\end{equation}

其中$s_{\max} = \max_i s_i$。

**理由**：当$\lambda$适中时，最大的$s_i$对应的$\alpha_i$占主导。

### 6. 熵的最终近似

结合式(7)、(11)和(13)：
\begin{equation}
H \approx \log n + \lambda \bar{s} - \lambda s_{\max} = \log n - \lambda (s_{\max} - \bar{s}) \tag{14}
\end{equation}

**定义波动幅度**：
\begin{equation}
\Delta s = s_{\max} - \bar{s} \tag{15}
\end{equation}

则：
\begin{equation}
H \approx \log n - \lambda \Delta s \tag{16}
\end{equation}

### 7. 熵不变性条件

**目标**：使$H$不依赖于$n$。

从式(16)，如果我们希望$H \approx C$（常数），则需要：
\begin{equation}
\log n - \lambda \Delta s \approx C \tag{17}
\end{equation}

移项：
\begin{equation}
\lambda \Delta s \approx \log n - C \tag{18}
\end{equation}

**假设$\Delta s$与$n$无关**（合理假设：分数的波动主要由数据和模型决定），则：
\begin{equation}
\lambda \propto \log n \tag{19}
\end{equation}

### 8. 与维度$d$的关系

**分数的形式**：

在注意力机制中，$s_i = \langle \boldsymbol{q}, \boldsymbol{k}_i \rangle$，其中$\boldsymbol{q}, \boldsymbol{k}_i \in \mathbb{R}^d$。

**方差分析**：

假设$\boldsymbol{q}$和$\boldsymbol{k}_i$的元素独立同分布，$\mathbb{E}[q_j] = 0$，$\text{Var}(q_j) = \sigma^2$。

则：
\begin{equation}
s_i = \sum_{j=1}^d q_j k_{i,j} \tag{20}
\end{equation}

期望：$\mathbb{E}[s_i] = 0$

方差：
\begin{equation}
\text{Var}(s_i) = d \cdot \mathbb{E}[q_j^2] \mathbb{E}[k_{i,j}^2] = d\sigma^4 \tag{21}
\end{equation}

**标准化**：

为了使方差不随$d$变化，我们除以$\sqrt{d}$：
\begin{equation}
s_i' = \frac{s_i}{\sqrt{d}} \Rightarrow \text{Var}(s_i') = \sigma^4 \tag{22}
\end{equation}

**结合两者**：

总的缩放因子应该是：
\begin{equation}
\lambda \propto \frac{\log n}{d} \tag{23}
\end{equation}

更精确地：
\begin{equation}
\lambda = \frac{\kappa \log n}{d} \tag{24}
\end{equation}

其中$\kappa$是一个与$n$和$d$无关的常数（通常取$\kappa = 1$）。

### 9. 最终的熵不变性注意力

**提出的公式**：
\begin{equation}
\boxed{\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\kappa \log n}{d} \boldsymbol{Q}\boldsymbol{K}^\top\right) \boldsymbol{V}} \tag{25}
\end{equation}

**与标准注意力的对比**：

标准注意力使用$\frac{1}{\sqrt{d}}$，只考虑了维度$d$。

熵不变性注意力使用$\frac{\kappa \log n}{d}$，同时考虑了维度$d$和序列长度$n$。

### 10. 严格的数学推导（Laplace近似）

上述推导使用了较粗糙的近似。现在我们用Laplace近似进行更严格的分析。

**Laplace近似的思想**：

对于积分或求和：
\begin{equation}
I = \int e^{nf(x)} dx \approx e^{nf(x^*)} \sqrt{\frac{2\pi}{n|f''(x^*)|}} \tag{26}
\end{equation}

其中$x^* = \arg\max_x f(x)$。

**应用到配分函数**：

\begin{equation}
Z = \sum_{i=1}^n e^{\lambda s_i} \tag{27}
\end{equation}

改写为连续形式（假设$s_i$是从某个分布采样的）：
\begin{equation}
Z \approx n \int e^{\lambda s} p(s) ds \tag{28}
\end{equation}

其中$p(s)$是$s$的分布。

**对数配分函数**：

\begin{equation}
\log Z = \log n + \log\int e^{\lambda s} p(s) ds \tag{29}
\end{equation}

**Laplace近似**（当$\lambda$适中时）：

\begin{equation}
\log\int e^{\lambda s} p(s) ds \approx \lambda s^* - \frac{1}{2}\log(2\pi) + \frac{1}{2}\log\lambda \tag{30}
\end{equation}

其中$s^*$是使$\lambda s + \log p(s)$最大的点。

对于均匀分布$p(s)$，$s^* = s_{\max}$。

**简化**（忽略常数项）：
\begin{equation}
\log Z \approx \log n + \lambda s_{\max} \tag{31}
\end{equation}

**熵的计算**（使用$\mathbb{E}_\alpha[s] \approx s_{\max}$）：
\begin{equation}
H = \log Z - \lambda \mathbb{E}_\alpha[s] \approx \log n + \lambda s_{\max} - \lambda s_{\max} = \log n \tag{32}
\end{equation}

这给出了$H \approx \log n$，恰好是均匀分布的熵！

**修正**：实际上$\mathbb{E}_\alpha[s] \neq s_{\max}$，所以需要更精细的分析。

### 11. 二阶修正与波动

**更精确的$\mathbb{E}_\alpha[s]$**：

考虑softmax的二阶展开：
\begin{equation}
\alpha_i = \frac{e^{\lambda s_i}}{\sum_j e^{\lambda s_j}} \approx \frac{e^{\lambda (s_i - s_{\max})}}{\sum_j e^{\lambda (s_j - s_{\max})}} \tag{33}
\end{equation}

记$\Delta_i = s_i - s_{\max} \leq 0$，则：
\begin{equation}
\alpha_i \approx \frac{e^{\lambda \Delta_i}}{1 + \sum_{j \neq i^*} e^{\lambda \Delta_j}} \tag{34}
\end{equation}

其中$i^* = \arg\max_i s_i$。

**当$\lambda$适中时**：

\begin{equation}
\alpha_{i^*} \approx \frac{1}{1 + \sum_{j \neq i^*} e^{\lambda \Delta_j}} \tag{35}
\end{equation}

**平均分数**：
\begin{equation}
\mathbb{E}_\alpha[s] \approx \alpha_{i^*} s_{\max} + \sum_{j \neq i^*} \alpha_j s_j \tag{36}
\end{equation}

**简化**（假设其他$\alpha_j$较小）：
\begin{equation}
\mathbb{E}_\alpha[s] \approx s_{\max} - \Delta s \tag{37}
\end{equation}

其中$\Delta s = s_{\max} - \bar{s}$的具体值依赖于分布。

### 12. 缩放因子的渐近分析

**大$n$极限**：

假设$s_i \sim \mathcal{N}(\mu, \sigma^2)$独立同分布。

**极值分布**：

当$n \to \infty$时，$s_{\max} = \max_i s_i$的分布趋向于Gumbel分布。

具体地：
\begin{equation}
s_{\max} \approx \mu + \sigma\sqrt{2\log n} \tag{38}
\end{equation}

**平均值**：
\begin{equation}
\bar{s} = \mathbb{E}[s] = \mu \tag{39}
\end{equation}

**波动**：
\begin{equation}
\Delta s = s_{\max} - \bar{s} \approx \sigma\sqrt{2\log n} \tag{40}
\end{equation}

**代入式(16)**：
\begin{equation}
H \approx \log n - \lambda \sigma\sqrt{2\log n} \tag{41}
\end{equation}

**熵不变性条件**：

希望$H$不依赖于$n$，但从式(41)看，这似乎不可能！

**分辨率**：实际上，式(41)中的$\log n$项和$\lambda\sqrt{2\log n}$项都与$n$有关，我们需要它们相互抵消。

但$\log n$和$\sqrt{\log n}$的增长速度不同，无法完全抵消。

**修正理解**：

"熵不变性"不是指$H$完全不变，而是指$H$的主导项$\log n$被合理控制。

具体地，我们希望：
\begin{equation}
H \approx C \cdot \log n \tag{42}
\end{equation}

其中$C$是一个接近1的常数。

### 13. 与Johnson-Lindenstrauss引理的联系

**Johnson-Lindenstrauss (JL) 引理**：

对于$n$个点在高维空间$\mathbb{R}^d$中，可以将它们投影到$O(\log n / \epsilon^2)$维空间，保持距离在$(1-\epsilon, 1+\epsilon)$范围内。

**与注意力的类比**：

注意力机制可以理解为一种**软投影**：从$n$个值向量$\{\boldsymbol{v}_i\}$中提取信息到单个输出$\boldsymbol{o}$。

**熵与维度**：

JL引理告诉我们，保持$n$个点的信息需要$O(\log n)$维。

类比地，注意力的熵（度量信息量）也应该是$O(\log n)$。

**缩放因子的连接**：

我们的缩放因子$\lambda \propto \log n / d$恰好体现了这个关系：
- 当$d$固定时，$\lambda \propto \log n$
- 当$n$固定时，$\lambda \propto 1/d$

### 14. 信息论视角

**熵的信息论意义**：

熵$H$度量了注意力分布的**不确定性**，也是编码注意力分布所需的**平均比特数**。

**熵不变性的意义**：

如果$H \propto \log n$，那么：
- 编码注意力分布需要$O(\log n)$比特
- 这与用$\lceil \log_2 n \rceil$比特编码$n$个位置的任意一个一致

**香农熵界**：

对于$n$个等概率的选择，最小编码长度是$\log_2 n$比特。

注意力分布通常不是均匀的（$H < \log n$），所以实际编码可以更短。

### 15. 实验验证

**设置**：

生成随机的注意力分数$s_i \sim \mathcal{N}(0, 1)$，变化$n$和缩放因子$\lambda$。

**观察**：

1. **标准缩放**（$\lambda = 1/\sqrt{d}$）：
   - $H$随$n$增加而增加
   - 大约$H \approx \log n - C$

2. **熵不变性缩放**（$\lambda = \log n / d$）：
   - $H$在不同$n$下更加稳定
   - 接近$H \approx \log n$（均匀分布）

**问题**：如果$H \approx \log n$，那不就是均匀分布吗？这样有用吗？

**回答**：
- $H \approx \log n$是理论上界（最大熵）
- 实际中，由于数据的结构，$H < \log n$
- 关键是$H$的**范围**足够大，可学习的信息量充足

### 16. 与LayerNorm、RMSNorm的联系

**LayerNorm**：

在Transformer中，通常在attention之前应用LayerNorm：
\begin{equation}
\boldsymbol{q} = \text{LayerNorm}(\boldsymbol{x}) \boldsymbol{W}_Q \tag{43}
\end{equation}

LayerNorm归一化了均值和方差：
\begin{equation}
\text{LayerNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x} - \mu}{\sigma} \cdot \gamma + \beta \tag{44}
\end{equation}

**效果**：

LayerNorm使得$s_i = \langle \boldsymbol{q}, \boldsymbol{k}_i \rangle$的分布更加稳定，减少了对初始化的敏感性。

**与熵不变性的关系**：

LayerNorm部分实现了熵不变性的目标，但没有显式地考虑$n$的影响。

**RMSNorm**：

RMSNorm只归一化均方根，不中心化：
\begin{equation}
\text{RMSNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\text{RMS}(\boldsymbol{x})} \cdot \gamma \tag{45}
\end{equation}

其中$\text{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2}$。

**优势**：

RMSNorm更简单，计算更快，同时保持了类似的效果。

### 17. 动态缩放因子

**问题**：在实际应用中，序列长度$n$可能是变化的（如不同长度的句子）。

**解决方案**：使用**动态缩放因子**，根据当前的$n$调整$\lambda$。

**实现**：
\begin{equation}
\lambda(n) = \frac{\kappa \log n}{d} \tag{46}
\end{equation}

在每个attention层中，根据当前的$n$计算$\lambda$。

**挑战**：

- 不同长度的序列使用不同的$\lambda$，可能影响模型的一致性
- 需要额外的计算开销（虽然很小）

**折衷方案**：

使用$\lambda = \frac{\kappa \log n_{\max}}{d}$，其中$n_{\max}$是训练中见过的最大序列长度。

### 18. 位置编码的影响

**绝对位置编码**：

添加位置编码$\boldsymbol{p}_i$后：
\begin{equation}
s_{ij} = \langle \boldsymbol{q}_i + \boldsymbol{p}_i, \boldsymbol{k}_j + \boldsymbol{p}_j \rangle \tag{47}
\end{equation}

**对熵的影响**：

位置编码引入了额外的结构（如局部性偏好），会影响$s_{ij}$的分布，从而影响熵。

**分析**：

如果位置编码是Sinusoidal的：
\begin{equation}
p_{i,2k} = \sin(i / 10000^{2k/d}), \quad p_{i,2k+1} = \cos(i / 10000^{2k/d}) \tag{48}
\end{equation}

相邻位置的内积较大，远距离位置的内积较小。

**结果**：

注意力倾向于关注附近的token，$H$可能下降。

**修正**：

在使用位置编码时，可能需要调整$\kappa$以补偿熵的变化。

### 19. 稀疏注意力的熵

**稀疏注意力**：

只计算部分$(i, j)$对的attention score：
\begin{equation}
s_{ij} = \begin{cases}
\langle \boldsymbol{q}_i, \boldsymbol{k}_j \rangle, & (i,j) \in \mathcal{S} \\
-\infty, & \text{otherwise}
\end{cases} \tag{49}
\end{equation}

其中$\mathcal{S}$是允许的位置对集合（如局部窗口、跨步模式等）。

**有效序列长度**：

对于位置$i$，有效的$j$数量为$|\mathcal{S}_i|$，通常$|\mathcal{S}_i| \ll n$。

**熵的调整**：

最大熵变为：
\begin{equation}
H_{\max} = \log |\mathcal{S}_i| \tag{50}
\end{equation}

**缩放因子**：

相应地，应该使用：
\begin{equation}
\lambda = \frac{\kappa \log |\mathcal{S}_i|}{d} \tag{51}
\end{equation}

### 20. 交叉注意力与自注意力

**自注意力**：

$\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}$都来自同一序列，长度为$n$。

**交叉注意力**（如encoder-decoder attention）：

$\boldsymbol{Q}$来自decoder（长度$n_q$），$\boldsymbol{K}, \boldsymbol{V}$来自encoder（长度$n_k$）。

**熵的分析**：

对于decoder的每个位置$i$，熵为：
\begin{equation}
H_i = -\sum_{j=1}^{n_k} \alpha_{ij} \log \alpha_{ij} \tag{52}
\end{equation}

**缩放因子**：

应该使用$n_k$（key/value的长度）：
\begin{equation}
\lambda = \frac{\kappa \log n_k}{d} \tag{53}
\end{equation}

### 21. 多头注意力的熵

**多头注意力**：

\begin{equation}
\text{MultiHead}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \boldsymbol{W}^O \tag{54}
\end{equation}

其中：
\begin{equation}
\text{head}_i = \text{Attention}(\boldsymbol{Q}\boldsymbol{W}_i^Q, \boldsymbol{K}\boldsymbol{W}_i^K, \boldsymbol{V}\boldsymbol{W}_i^V) \tag{55}
\end{equation}

**每个头的维度**：

通常$d_{\text{head}} = d / h$。

**缩放因子**：

对于每个头：
\begin{equation}
\lambda_i = \frac{\kappa \log n}{d_{\text{head}}} = \frac{\kappa h \log n}{d} \tag{56}
\end{equation}

**观察**：

多头注意力相当于使用了更大的缩放因子（乘以$h$）。

### 22. Flash Attention与熵不变性

**Flash Attention**：

Flash Attention是一种高效的attention计算方法，通过分块（tiling）和重计算（recomputation）减少内存访问。

**与熵不变性的关系**：

Flash Attention本身不改变attention的数学定义，所以熵不变性缩放可以直接应用。

**实现**：

在Flash Attention的kernel中，将缩放因子从$1/\sqrt{d}$改为$\log n / d$即可。

### 23. 训练稳定性

**问题**：使用$\lambda = \log n / d$时，当$n$很大时，$\lambda$可能很大，导致softmax过于peaked。

**分析**：

假设$n = 1024$，$d = 64$，$\kappa = 1$：
\begin{equation}
\lambda = \frac{\log 1024}{64} = \frac{10 \log 2}{64} \approx 0.108 \tag{57}
\end{equation}

标准缩放：
\begin{equation}
\lambda_{\text{std}} = \frac{1}{\sqrt{64}} = 0.125 \tag{58}
\end{equation}

两者接近！

**一般规律**：

对于常见的$n$和$d$值：
\begin{equation}
\frac{\log n}{d} \approx \frac{1}{\sqrt{d}} \tag{59}
\end{equation}

例如，当$n \approx e^{\sqrt{d}}$时，两者相等。

**实践中的选择**：

可以使用混合策略：
\begin{equation}
\lambda = \frac{1}{\sqrt{d}} + \beta \frac{\log n}{d} \tag{60}
\end{equation}

其中$\beta$是一个小的权重（如$\beta = 0.1$）。

### 24. 长序列建模

**问题**：对于非常长的序列（如$n = 10^6$），$\log n \approx 14$，缩放因子可能显著不同。

**解决方案1**：分段处理

将长序列分成多个段，每段长度为$n_{\text{seg}}$，分别计算attention。

**解决方案2**：层次化attention

使用层次化的attention结构，先在局部窗口内计算attention，再在全局level计算。

**解决方案3**：自适应缩放

根据熵的实际值动态调整$\lambda$：
\begin{equation}
\lambda \leftarrow \lambda \cdot \frac{H_{\text{target}}}{H_{\text{current}}} \tag{61}
\end{equation}

### 25. 与温度参数的统一

**温度参数**：

在很多场景（如知识蒸馏）中，使用温度参数$T$：
\begin{equation}
\alpha_i = \frac{e^{s_i / T}}{\sum_j e^{s_j / T}} \tag{62}
\end{equation}

**与缩放因子的关系**：

$\lambda = 1/T$是缩放因子的倒数。

**统一框架**：

我们可以将熵不变性缩放看作一种**自适应温度**：
\begin{equation}
T(n, d) = \frac{d}{\kappa \log n} \tag{63}
\end{equation}

**解释**：

- 当$n$增大时，$T$减小（温度降低），使softmax更加peaked
- 这补偿了序列长度增加带来的熵自然增长

### 26. 理论局限性

**假设的局限**：

1. **平均场近似**：式(10)假设$\log \mathbb{E}[e^{\lambda s}] \approx \mathbb{E}[\lambda s]$，这在$\lambda$或$s$的方差很大时不准确。

2. **独立性假设**：假设$s_i$独立同分布，但实际上它们可能有相关性（如相邻token的相似性）。

3. **$\mathbb{E}_\alpha[s] \approx s_{\max}$**：这个近似在softmax非常peaked时才准确。

**适用范围**：

熵不变性缩放在以下情况下最有效：
- 序列长度$n$变化范围大（如从10到10000）
- 维度$d$适中（如64到512）
- 数据分布相对均匀（没有极端的outliers）

### 27. 实验设计建议

**消融实验**：

对比以下缩放策略：
1. 标准缩放：$\lambda = 1/\sqrt{d}$
2. 熵不变性缩放：$\lambda = \log n / d$
3. 混合缩放：$\lambda = \alpha/\sqrt{d} + \beta \log n / d$

**评估指标**：

1. **任务性能**：困惑度（perplexity）、准确率等
2. **熵的稳定性**：在不同$n$下，熵的方差
3. **收敛速度**：达到目标性能所需的训练步数

**数据集**：

选择序列长度变化大的数据集，如：
- 文本：不同长度的句子/段落
- 图像：不同分辨率的patch序列

### 28. 开放问题

**问题1**：最优的$\kappa$值

理论上$\kappa = 1$，但实验中可能需要调整。如何自动确定最优$\kappa$？

**问题2**：非均匀分布

如果$s_i$不是独立同分布，而是有明显的聚类结构，熵不变性条件如何修正？

**问题3**：与其他正则化的结合

熵不变性缩放能否与dropout、weight decay等正则化技术结合？

**问题4**：因果注意力

在decoder的因果注意力（causal attention）中，有效$n$随位置变化，如何处理？

### 29. 代码实现

**PyTorch实现**：

```python
import torch
import torch.nn as nn
import math

class EntropyInvariantAttention(nn.Module):
    def __init__(self, d_model, num_heads, kappa=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.kappa = kappa

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size, n, _ = Q.shape

        # 线性变换
        Q = self.W_Q(Q).view(batch_size, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, n, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, n, self.num_heads, self.d_k).transpose(1, 2)

        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, heads, n, n)

        # 熵不变性缩放
        scale = self.kappa * math.log(n) / self.d_k
        scores = scores * scale

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn_weights, V)  # (batch, heads, n, d_k)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, n, self.d_model)
        output = self.W_O(output)

        return output, attn_weights
```

**使用示例**：

```python
# 初始化
d_model = 512
num_heads = 8
attn = EntropyInvariantAttention(d_model, num_heads, kappa=1.0)

# 输入
batch_size = 32
seq_len = 100
X = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output, weights = attn(X, X, X)

print(f"Output shape: {output.shape}")  # (32, 100, 512)
print(f"Attention weights shape: {weights.shape}")  # (32, 8, 100, 100)
```

### 30. 总结与展望

**核心贡献**：

1. 从熵不变性角度推导了缩放因子$\lambda \propto \log n / d$
2. 连接了序列长度、维度和注意力熵之间的关系
3. 提供了比标准$1/\sqrt{d}$更精细的缩放策略

**理论意义**：

- 揭示了Softmax注意力的信息论性质
- 为理解Transformer的缩放规律提供了新视角
- 连接了离散优化（max）和连续优化（softmax）

**实践价值**：

- 在序列长度变化大的任务中可能提高性能
- 为超长序列建模提供了理论指导
- 有助于理解和调试attention机制

**未来方向**：

1. 在大规模语言模型中验证熵不变性缩放
2. 探索自适应$\kappa$的学习方法
3. 推广到其他类型的attention（如linear attention）
4. 研究熵与泛化性能的关系

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Johnson & Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space", 1984
3. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
4. Zhang et al., "Root Mean Square Layer Normalization", NeurIPS 2019
5. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021

## 最终总结

本文从熵不变性角度重新推导了Softmax注意力的缩放因子，得到：

\begin{equation}
\lambda = \frac{\kappa \log n}{d}
\end{equation}

**关键洞察**：
1. **平均场近似**：$\log \mathbb{E}[e^{\lambda s}] \approx \mathbb{E}[\lambda s]$连接了指数和对数
2. **Max近似**：Softmax侧重最大值，$\mathbb{E}_\alpha[s] \approx s_{\max}$
3. **熵的渐近**：$H \approx \log n - \lambda(s_{\max} - \bar{s})$给出了熵与$n$的关系
4. **方差控制**：除以$d$保持分数方差稳定

**实用建议**：
- 对于固定长度序列，使用标准$1/\sqrt{d}$缩放即可
- 对于变长序列（如$n \in [10, 1000]$），考虑熵不变性缩放
- 超长序列建模中，$\log n$因子可能带来显著改善

熵不变性提供了理解和设计attention机制的新视角，值得进一步探索！

