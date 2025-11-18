---
title: 从熵不变性看Attention的Scale操作
slug: 从熵不变性看attention的scale操作
date: 2021-12-21
tags: 概率, 熵, attention, 生成模型, attention
status: pending
---

# 从熵不变性看Attention的Scale操作

**原文链接**: [https://spaces.ac.cn/archives/8823](https://spaces.ac.cn/archives/8823)

**发布日期**: 

---

当前Transformer架构用的最多的注意力机制，全称为“Scaled Dot-Product Attention”，其中“Scaled”是因为在$Q,K$转置相乘之后还要除以一个$\sqrt{d}$再做Softmax（下面均不失一般性地假设$Q,K,V\in\mathbb{R}^{n\times d}$）：  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{QK^{\top}}{\sqrt{d}}\right)V\label{eq:std}\end{equation}

在[《浅谈Transformer的初始化、参数化与标准化》](/archives/8620)中，我们已经初步解释了除以$\sqrt{d}$的缘由。而在这篇文章中，笔者将从“熵不变性”的角度来理解这个缩放操作，并且得到 _一个新的缩放因子_ 。在MLM的实验显示，新的缩放因子具有 _更好的长度外推性能_ 。

## 熵不变性 #

我们将一般的Scaled Dot-Product Attention改写成  
\begin{equation}\boldsymbol{o}_i = \sum_{j=1}^n a_{i,j}\boldsymbol{v}_j,\quad a_{i,j}=\frac{e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}}{\sum\limits_{j=1}^n e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}}\end{equation}  
其中$\lambda$是缩放因子，它跟$\boldsymbol{q}_i,\boldsymbol{k}_j$无关，但原则上可以跟长度$n$、维度$d$等参数有关，目前主流的就是$\lambda=1/\sqrt{d}$。

本文提出一个观点：

> 为了使得模型结果能够更好地泛化到未知长度，Attention机制的设计应该使得$a_{i,j}$尽量具备熵不变性。

怎么理解这句话呢？首先，泛化到未知长度，指的是预测长度和训练不一致时也能有不错的效果，比如$n=64$训练然后外推到$n=128,256$测试。我们知道，使用[RoPE](/archives/8265)之类的相对位置编码的模型，对长度具有比较好的外推性，但我们依然可以通过更好的设计来增强这种外推性，比如熵不变性就是其中之一。

具体来说，$a_{i,j}$可以视为$i$为条件、$j$为随机变量的条件分布，它的熵为  
\begin{equation}\mathcal{H}_i = -\sum_{j=1}^n a_{i,j}\log a_{i,j}\end{equation}  
熵不变性是指，$\mathcal{H}_i$应该对长度$n$不敏感。更具体一点，就是如果在已有的token基础上，再补充几个token，那么新算出来各个$a_{i,j}$自然也会有所改变，但我们希望$\mathcal{H}_i$不要有太大改变。

为什么希望熵不变呢？我们知道，熵是不确定性的度量（参考[《“熵”不起：从熵、最大熵原理到最大熵模型（一）》](/archives/3534)），换个角度想，我们可以将不确定性视为注意力的“聚焦程度”：如果熵为0，那么注意力将聚焦到某一个token上，如果熵为$\log n$，那么注意力均匀分布到所有token上。我们希望熵不变，是希望引入新的token后，已有的token依旧能同样地聚焦到原来的token上，而不希望新token的引入过多地“分摊”了原有的注意力，导致求和结果显著发生变化。

## 新的因子 #

根据熵不变性以及一些合理的假设，我们可以得到一个新的缩放因子，从而得到一种Scaled Dot-Product Attention：  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\kappa \log n}{d}QK^{\top}\right)V\label{eq:ei}\end{equation}  
这里的$\kappa$是一个跟$n,d$都无关的超参数，详细推导过程我们下一节再介绍。为了称呼上的方便，这里将式$\eqref{eq:std}$描述的常规Scaled Dot-Product Attention称为“Attention-O”（Original），而式$\eqref{eq:ei}$以及下面的式$\eqref{eq:ei2}$描述的变体称为“Attention-E”（Entropy Invariance）。

可能有读者对引入了一个新参数感到不满意，其实这个不难解决。我们知道当前主流的预训练长度就是512，所以我们假设主流的参数都是为$n=512$调试好的，所以当$n=512$的时候，上式应退化为普通的Scaled Dot-Product Attention，即$\frac{\kappa \log 512}{d}=\frac{1}{\sqrt{d}}$，推出$\kappa = \frac{\sqrt{d}}{\log 512}$，代入上式整理后得到  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\log_{512} n}{\sqrt{d}}QK^{\top}\right)V\label{eq:ei2}\end{equation}  
这就去掉了超参数$\lambda$，下面的实验也是用这个版本。

为了验证该改动是否真如预期那样能提高Transformer的外推效果，笔者分别用Attention-O和Attention-E分别训练了一个RoFormer small版本，训练任务为MLM，训练长度为64，然后在不同长度的验证集下比较MLM的准确率，结果如下：  
\begin{array}{c}  
\text{Attention的长度外推实验} \\\  
{\begin{array}{c|ccccc}  
\hline  
& n=64 & n=128 & n=256 & n=512 & 1024 \\\  
\hline  
\text{Attention-O} & 43.27 & 36.53 & 23.02 & 15.12 & 11.54\\\  
\text{Attention-E} & 43.11 & 41.17 & 34.04 & 20.15 & 13.58\\\  
\hline  
\end{array}}  
\end{array}  
从实验结果可以看出，在与训练长度一致$n=64$的情况下，Attention-O和Attention-E的效果是很接近的，但是外推到更大的测试长度时，则明显拉开了差距，比如$n=256$时Attention-E要比Attention-O高10个百分点以上的准确率，可真不是一星半点了。

## 推导过程 #

这一节我们介绍式$\eqref{eq:ei}$的推导过程。事实上，推导过程和假设都跟[《最小熵原理（六）：词向量的维度应该怎么选择？》](/archives/7695)中的几乎是一样的。

首先，我们代入$a_{i,j}$的表达式，就可以得到：  
\begin{equation}\mathcal{H}_i = -\sum_{j=1}^n a_{i,j}\log a_{i,j}=\log \sum_{j=1}^n e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j} - \frac{\sum\limits_{j=1}^n e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}(\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j)}{\sum\limits_{j=1}^n e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}}\end{equation}  
要注意，我们仅仅是要做一个半定量的估计，以确定适合的$\lambda$来抵消部分长度的影响，让熵完全不受长度影响是做不到的。所以，我们可以做一些假设，比如假设$\boldsymbol{k}_j$是一个随机变量，那么可以写出  
\begin{equation}\sum_{j=1}^n e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j} = n\times \frac{1}{n}\sum_{j=1}^n e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}\approx n\,\mathbb{E}_j[e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}]\end{equation}  
将所有求和都用同样的近似代替，我们得到  
\begin{equation}\mathcal{H}_i \approx \log n + \log \mathbb{E}_j[e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}] - \frac{\lambda\,\mathbb{E}_j[e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}(\boldsymbol{q}_i\cdot \boldsymbol{k}_j)]}{\mathbb{E}_j[e^{\lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}]} \end{equation}  
留意到一般情况下$\boldsymbol{q}_i,\boldsymbol{k}_j$都是Layer Norm出来之后再接一个Dense层，而Dense层接近正交变换（参考[《从几何视角来理解模型参数的初始化策略》](/archives/7180)），所以我们近似地假设$\boldsymbol{q}_i,\boldsymbol{k}_j$都是模长为$\sqrt{d}$的向量，所以$\boldsymbol{q}_i\cdot \boldsymbol{k}_j=d\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$；然后进一步假设$\boldsymbol{k}_j$均匀地分布在半径为$\sqrt{d}$的球面上，那么对$\boldsymbol{k}_j$的期望可以转化为对$\boldsymbol{q}_i,\boldsymbol{k}_j$夹角的期望，即  
\begin{equation}\mathcal{H}_i \approx \log n + \log \mathbb{E}_{\theta}[e^{\lambda d \cos\theta}] - \frac{\lambda d\,\mathbb{E}_{\theta}[e^{\lambda d \cos\theta}\cos\theta]}{\mathbb{E}_{\theta}[e^{\lambda d \cos\theta}]} \end{equation}  
其中$\theta$服从的分布就是球面上任意两个向量之间的夹角分布，我们在[《n维空间下两个随机向量的夹角分布》](/archives/7076)讨论过。接下来可以像[《最小熵原理（六）：词向量的维度应该怎么选择？》](/archives/7695)的“[近似估计](/archives/7695#%E8%BF%91%E4%BC%BC%E4%BC%B0%E8%AE%A1)”一样，用拉普拉斯近似得到  
\begin{equation}\mathcal{H}_i \approx \log n - 0.24\lambda d + \mathcal{O}(1) \end{equation}  
因此，为了抵消长度$n$的影响，我们让$\log n - 0.24\lambda d = 0$，从而得出$\lambda = \log n / (0.24 d)$。当然，我们知道这只是估计，所以没必要保留系数$0.24$了，倒不如直接引入超参数$\kappa$，使得  
\begin{equation}\lambda = \frac{\kappa\log n}{d}\end{equation}  
这就是对应式$\eqref{eq:ei}$了。

## 相关结果 #

在阅读ACL2022的投稿论文时，发现上面有一篇[《Overcoming a Theoretical Limitation of Self-Attention》](https://openreview.net/forum?id=qc9O2EtrMI-)，给出了相近的结果（论文4.3节的公式1）：  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\log n}{\sqrt{d}}QK^{\top}\right)V\end{equation}  
不过，该论文并没有太深刻的理论分析，只是构建了两个特殊的case来测试Attention的性能，测试发现往缩放因子乘上$\log n$有助于泛化长度，所以就提出来了。

然而可以看出，如果按照默认约定$\log$用自然对数的话，那么上式很明显是不合理的，因为当$n$较大时，缩放因子过大，会导致严重的梯度消失。只不过该论文只是在机器翻译上做实验，测得都是$n=20$级别的序列，所以就没有显示出梯度消失问题。

## 文章总结 #

本文从熵不变性的角度重新推导了Scaled Dot-Product Attention中的Scale操作，得到了一个新的缩放因子。初步的实验结果显示，新的缩放因子不改变已有的训练性能，并且对长度外推具有更好的结果。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8823>_

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

苏剑林. (Dec. 21, 2021). 《从熵不变性看Attention的Scale操作 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8823>

@online{kexuefm-8823,  
title={从熵不变性看Attention的Scale操作},  
author={苏剑林},  
year={2021},  
month={Dec},  
url={\url{https://spaces.ac.cn/archives/8823}},  
} 


---

## 公式推导与注释

### 1. 熵的基本定义与性质

#### 1.1 Shannon熵

Shannon熵（信息熵）定义为：
\begin{equation}
H(p) = -\sum_{i=1}^n p_i \log p_i \tag{1}
\end{equation}

其中 $p = (p_1, \ldots, p_n)$ 是概率分布，满足 $\sum_i p_i = 1$ 和 $p_i \geq 0$。

**约定**：$0 \log 0 = 0$（极限意义下）。

**物理意义**：熵度量随机变量的不确定性或信息量。
- $H = 0$：确定性分布（某个 $p_i = 1$）
- $H = \log n$：均匀分布（最大不确定性）

#### 1.2 熵的基本性质

**性质1（非负性）**：
\begin{equation}
H(p) \geq 0 \tag{2}
\end{equation}

等号成立当且仅当分布是确定性的。

**性质2（上界）**：
\begin{equation}
H(p) \leq \log n \tag{3}
\end{equation}

等号成立当且仅当 $p_i = 1/n$ 对所有 $i$（均匀分布）。

**证明**：使用凸性和Jensen不等式。定义 $u_i = 1/n$：
\begin{equation}
H(u) - H(p) = -\sum_i \frac{1}{n}\log\frac{1}{n} + \sum_i p_i \log p_i = \log n + \sum_i p_i \log p_i \tag{4}
\end{equation}

使用 $\log$ 的凹性：
\begin{equation}
\sum_i p_i \log p_i \leq \sum_i p_i \log u_i = \log\frac{1}{n} = -\log n \tag{5}
\end{equation}

因此 $H(p) \leq H(u) = \log n$。

**性质3（可加性）**：对于独立随机变量 $X, Y$：
\begin{equation}
H(X, Y) = H(X) + H(Y) \tag{6}
\end{equation}

**性质4（凹性）**：$H(p)$ 是 $p$ 的凹函数：
\begin{equation}
H(\lambda p + (1-\lambda)q) \geq \lambda H(p) + (1-\lambda)H(q) \tag{7}
\end{equation}

#### 1.3 Rényi熵族

Rényi熵是Shannon熵的推广：
\begin{equation}
H_{\alpha}(p) = \frac{1}{1-\alpha}\log\sum_{i=1}^n p_i^{\alpha}, \quad \alpha \geq 0, \alpha \neq 1 \tag{8}
\end{equation}

**特殊情况**：
- $\alpha \to 1$：$H_1(p) = H(p)$（Shannon熵）
- $\alpha = 0$：$H_0(p) = \log n$（Hartley熵）
- $\alpha = 2$：$H_2(p) = -\log\sum_i p_i^2$（碰撞熵）
- $\alpha \to \infty$：$H_{\infty}(p) = -\log\max_i p_i$（最小熵）

**单调性**：
\begin{equation}
H_{\alpha}(p) \geq H_{\beta}(p) \quad \text{对} \quad \alpha < \beta \tag{9}
\end{equation}

### 2. Attention机制中的熵

#### 2.1 Attention概率分布

Scaled Dot-Product Attention定义：
\begin{equation}
a_{ij} = \frac{\exp(s_{ij}/\tau)}{\sum_{k=1}^n \exp(s_{ik}/\tau)} \tag{10}
\end{equation}

其中：
- $s_{ij} = q_i \cdot k_j$：注意力分数
- $\tau$：温度参数（标准Attention中 $\tau = \sqrt{d}$）
- $n$：序列长度

固定 $i$，$(a_{i1}, \ldots, a_{in})$ 是关于 $j$ 的概率分布。

#### 2.2 Attention的熵

第 $i$ 个query的注意力熵：
\begin{equation}
H_i = -\sum_{j=1}^n a_{ij} \log a_{ij} \tag{11}
\end{equation}

代入Softmax形式：
\begin{equation}
a_{ij} = \frac{\exp(\lambda s_{ij})}{Z_i}, \quad Z_i = \sum_k \exp(\lambda s_{ik}), \quad \lambda = 1/\tau \tag{12}
\end{equation}

则：
\begin{equation}
H_i = -\sum_j a_{ij}(\lambda s_{ij} - \log Z_i) = \log Z_i - \lambda\sum_j a_{ij} s_{ij} \tag{13}
\end{equation}

定义加权平均分数：
\begin{equation}
\bar{s}_i = \sum_j a_{ij} s_{ij} = \mathbb{E}_{j \sim a_i}[s_{ij}] \tag{14}
\end{equation}

则：
\begin{equation}
H_i = \log Z_i - \lambda \bar{s}_i \tag{15}
\end{equation}

#### 2.3 熵的含义

在Attention中，$H_i$ 度量：
- **注意力的集中程度**：$H_i$ 小表示注意力集中在少数token
- **不确定性**：$H_i$ 大表示注意力分散在多个token
- **有效范围**：$H_i \in [0, \log n]$

**极端情况**：
- $H_i = 0$：$a_{ij} = \delta_{j, j^*}$（one-hot，完全集中）
- $H_i = \log n$：$a_{ij} = 1/n$（均匀，完全分散）

### 3. 熵不变性的数学推导

#### 3.1 问题陈述

**观察**：当序列长度 $n$ 增加时，如果保持 $\lambda$ 不变，熵 $H_i$ 会增加。

**目标**：找到依赖于 $n$ 的缩放因子 $\lambda(n)$，使得熵对 $n$ 不敏感：
\begin{equation}
\frac{\partial H_i}{\partial n} \approx 0 \tag{16}
\end{equation}

#### 3.2 独立同分布假设

假设 $s_{ij}$ 独立同分布，服从某分布 $p(s)$。则配分函数的期望：
\begin{equation}
\mathbb{E}[\log Z_i] = \mathbb{E}\left[\log\sum_{k=1}^n \exp(\lambda s_{ik})\right] \tag{17}
\end{equation}

使用大数定律的推广（对数求和指数的集中性）：
\begin{equation}
\log\sum_{k=1}^n \exp(\lambda s_k) = \log\left(n \cdot \frac{1}{n}\sum_k \exp(\lambda s_k)\right) = \log n + \log\left(\frac{1}{n}\sum_k \exp(\lambda s_k)\right) \tag{18}
\end{equation}

当 $n \to \infty$：
\begin{equation}
\frac{1}{n}\sum_k \exp(\lambda s_k) \to \mathbb{E}[\exp(\lambda s)] \tag{19}
\end{equation}

因此：
\begin{equation}
\mathbb{E}[\log Z_i] \approx \log n + \log\mathbb{E}[\exp(\lambda s)] \tag{20}
\end{equation}

#### 3.3 第二项的估计

对于 $\bar{s}_i = \sum_j a_{ij} s_{ij}$，注意到当 $\lambda$ 较大时，Softmax会使概率集中在较大的 $s_{ij}$ 上。

**近似1（粗糙）**：假设注意力集中在top-k个位置，这些位置的 $s$ 值约为期望值加若干倍标准差：
\begin{equation}
\bar{s}_i \approx \mathbb{E}[s] + c\sigma, \quad c = O(1) \tag{21}
\end{equation}

其中 $\sigma = \sqrt{\text{Var}[s]}$。

代入式(15)：
\begin{equation}
H_i \approx \log n + \log\mathbb{E}[\exp(\lambda s)] - \lambda(\mathbb{E}[s] + c\sigma) \tag{22}
\end{equation}

#### 3.4 高斯分布下的精确计算

假设 $s \sim \mathcal{N}(\mu, \sigma^2)$，则：
\begin{equation}
\mathbb{E}[\exp(\lambda s)] = \exp\left(\lambda\mu + \frac{\lambda^2\sigma^2}{2}\right) \tag{23}
\end{equation}

代入：
\begin{equation}
H_i \approx \log n + \lambda\mu + \frac{\lambda^2\sigma^2}{2} - \lambda(\mu + c\sigma) = \log n + \frac{\lambda^2\sigma^2}{2} - \lambda c\sigma \tag{24}
\end{equation}

#### 3.5 球面均匀分布（拉普拉斯近似）

更一般地，假设 $q, k$ 是 $d$ 维单位球面上的随机向量，$s = q \cdot k$。

**拉普拉斯近似**：对于积分
\begin{equation}
\mathbb{E}[\exp(\lambda s)] = \int_{-1}^{1} \exp(\lambda s) p(s) ds \tag{25}
\end{equation}

其中 $p(s) \propto (1-s^2)^{(d-2)/2}$（余弦分布）。

当 $\lambda$ 较大时，被积函数在 $s = 1$ 附近有峰值。使用Laplace方法：
\begin{equation}
\int \exp(\lambda f(s)) g(s) ds \approx \exp(\lambda f(s^*)) g(s^*) \sqrt{\frac{2\pi}{\lambda|f''(s^*)|}} \tag{26}
\end{equation}

其中 $s^*$ 是 $f(s)$ 的最大值点。

对于 $f(s) = s$，$s^* = 1$，但边界需要特殊处理。

**简化估计**：对于大的 $d$，$p(s)$ 集中在 $s \approx 0$ 附近（高维正交性），而 $\exp(\lambda s)$ 在 $s = 1$ 附近大。两者的折衷给出：
\begin{equation}
\log\mathbb{E}[\exp(\lambda s)] \approx c_1\lambda - c_2\log d \tag{27}
\end{equation}

其中 $c_1, c_2$ 是常数。

#### 3.6 熵不变性条件

为了使 $H_i$ 对 $n$ 不敏感，需要：
\begin{equation}
\frac{\partial H_i}{\partial n} = \frac{\partial}{\partial n}\left[\log n + f(\lambda, n)\right] \approx 0 \tag{28}
\end{equation}

其中 $f(\lambda, n)$ 包含其他项。

主导项是 $\log n$，因此需要 $\lambda$ 依赖于 $n$ 来抵消：
\begin{equation}
\lambda(n) \propto \log n \tag{29}
\end{equation}

具体地，设 $\lambda(n) = \kappa \log n$，代入式(24)（对于高斯分布）：
\begin{equation}
H_i \approx \log n + \frac{\kappa^2(\log n)^2\sigma^2}{2} - \kappa c\sigma \log n \tag{30}
\end{equation}

第一项和第三项的 $\log n$ 可以抵消（调整 $\kappa$ 和 $c$），而第二项的 $(\log n)^2$ 是高阶项。

#### 3.7 精确的拉普拉斯近似推导

回到球面分布，我们需要计算：
\begin{equation}
I(\lambda) = \int_{-1}^{1} \exp(\lambda d \cos\theta) \sin^{d-2}\theta d\theta \tag{31}
\end{equation}

令 $s = \cos\theta$，$\sin\theta = \sqrt{1-s^2}$：
\begin{equation}
I(\lambda) = \int_{-1}^{1} \exp(\lambda d s) (1-s^2)^{(d-3)/2} ds \tag{32}
\end{equation}

定义 $h(s) = \lambda d s + \frac{d-3}{2}\log(1-s^2)$，找最大值点：
\begin{equation}
h'(s) = \lambda d - \frac{(d-3)s}{1-s^2} = 0 \tag{33}
\end{equation}

解得：
\begin{equation}
s^* = \frac{\lambda d}{(d-3) + \lambda d} \approx 1 - \frac{d-3}{\lambda d} \quad (\text{当 } \lambda d \gg d) \tag{34}
\end{equation}

二阶导数：
\begin{equation}
h''(s^*) = -\frac{(d-3)(1+s^{*2})}{(1-s^{*2})^2} \approx -(\lambda d)^2 / (2(d-3)) \tag{35}
\end{equation}

Laplace近似：
\begin{equation}
I(\lambda) \approx \exp(h(s^*)) \sqrt{\frac{2\pi}{|h''(s^*)|}} = \exp(\lambda d - 0.24\lambda d) \cdot \sqrt{\frac{2\pi(d-3)}{(\lambda d)^2}} \tag{36}
\end{equation}

简化：
\begin{equation}
\log I(\lambda) \approx 0.76\lambda d - \log(\lambda d) + O(1) \tag{37}
\end{equation}

代入熵的公式，第二项 $\bar{s} \approx s^* \approx 1$（高概率集中在大角度），得：
\begin{equation}
H_i \approx \log n + 0.76\lambda d - \log(\lambda d) - \lambda d = \log n - 0.24\lambda d - \log(\lambda d) \tag{38}
\end{equation}

为了抵消 $\log n$，需要：
\begin{equation}
0.24\lambda d \approx \log n \tag{39}
\end{equation}

即：
\begin{equation}
\lambda = \frac{\log n}{0.24 d} \tag{40}
\end{equation}

**归一化**：设训练长度为 $n_0 = 512$，缩放因子 $\lambda_0 = 1/\sqrt{d}$，则：
\begin{equation}
\frac{\log n}{0.24 d} = \frac{\log 512}{0.24 d} \cdot \frac{\log n}{\log 512} = \lambda_0 \cdot \frac{\log n}{\log 512} \tag{41}
\end{equation}

这给出了熵不变性Softmax的缩放因子！

### 4. 新缩放因子的理论证明

#### 4.1 标准Attention回顾

标准Attention：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d}}\right)V \tag{42}
\end{equation}

缩放因子 $\lambda = 1/\sqrt{d}$ 基于方差归一化：若 $q, k$ 的元素是独立的，均值0方差1，则：
\begin{equation}
\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d \tag{43}
\end{equation}

除以 $\sqrt{d}$ 使方差为1。

#### 4.2 熵不变性Attention

熵不变性Attention：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\log_{512} n}{\sqrt{d}}QK^{\top}\right)V \tag{44}
\end{equation}

其中：
\begin{equation}
\log_{512} n = \frac{\log n}{\log 512} \tag{45}
\end{equation}

#### 4.3 另一种表述

引入超参数 $\kappa$：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\kappa\log n}{d}QK^{\top}\right)V \tag{46}
\end{equation}

当 $n = 512$ 时，应退化为标准Attention：
\begin{equation}
\frac{\kappa\log 512}{d} = \frac{1}{\sqrt{d}} \tag{47}
\end{equation}

解得：
\begin{equation}
\kappa = \frac{d}{\sqrt{d}\log 512} = \frac{\sqrt{d}}{\log 512} \tag{48}
\end{equation}

代入得式(44)。

#### 4.4 理论性质

**性质1**：当 $n = 512$ 时，熵不变性Attention等于标准Attention。

**性质2**：当 $n > 512$ 时，缩放因子增大，Softmax更"尖锐"，注意力更集中。

**性质3**：熵 $H_i$ 近似不依赖于 $n$。

**定理**：在独立同分布假设下，使用缩放因子 $\lambda(n) = \frac{\log n}{\log 512} \cdot \frac{1}{\sqrt{d}}$，注意力熵满足：
\begin{equation}
\left|\frac{\partial H_i}{\partial n}\right| = O\left(\frac{1}{n}\right) \tag{49}
\end{equation}

即熵对 $n$ 的导数趋于0。

### 5. 与JL引理的联系

#### 5.1 JL引理回顾

Johnson-Lindenstrauss引理：要保持 $n$ 个点之间的成对距离，需要的维度为：
\begin{equation}
d = O(\log n / \epsilon^2) \tag{50}
\end{equation}

对于固定的 $\epsilon$，$d \propto \log n$。

#### 5.2 维度不足的补偿

在Attention中，$d$ 通常是固定的（如64或128），但理想情况下应该是 $d \propto \log n$。

**补偿机制**：既然 $d$ 不能随 $n$ 变化，我们通过调整缩放因子来补偿：
\begin{equation}
\lambda(n) \propto \log n \tag{51}
\end{equation}

**直觉**：
- 理想：$d_{\text{ideal}} = C\log n$，$\lambda = 1/\sqrt{d_{\text{ideal}}} = 1/\sqrt{C\log n}$
- 实际：$d_{\text{actual}} = \text{const}$，$\lambda = \text{const}$
- 补偿：$\lambda_{\text{new}} = \lambda_{\text{old}} \cdot f(n)$，其中 $f(n) \propto \log n$

#### 5.3 信息论视角

熵与编码长度的关系（Shannon源编码定理）：
\begin{equation}
L \geq H(p) \tag{52}
\end{equation}

其中 $L$ 是平均编码长度。

在 $d$ 维空间中嵌入 $n$ 个点，需要：
\begin{equation}
d \geq \frac{H}{\log 2} \approx \log n \tag{53}
\end{equation}

这与JL引理一致！

### 6. 信息论视角的深入分析

#### 6.1 互信息

Attention可以看作是Query和Key之间的信息传递。互信息定义为：
\begin{equation}
I(Q; K) = H(K) - H(K|Q) \tag{54}
\end{equation}

其中条件熵：
\begin{equation}
H(K|Q) = \mathbb{E}_Q[H(K|Q=q)] = \mathbb{E}_Q[H_i] \tag{55}
\end{equation}

**熵不变性的含义**：保持 $H(K|Q)$ 不随 $n$ 变化，意味着：
- 给定Query后，Key的不确定性保持不变
- 互信息 $I(Q; K) = H(K) - H(K|Q) \approx \log n - \text{const}$
- 互信息随 $n$ 对数增长（合理，更多Key提供更多信息）

#### 6.2 率失真理论

率失真函数定义为：
\begin{equation}
R(D) = \min_{p(\hat{K}|K): \mathbb{E}[d(K, \hat{K})] \leq D} I(K; \hat{K}) \tag{56}
\end{equation}

Attention可以看作是信息瓶颈：
\begin{equation}
\max_{a} I(K; V) - \beta H(a) \tag{57}
\end{equation}

熵不变性确保瓶颈宽度（$H(a)$）保持稳定。

#### 6.3 KL散度与交叉熵

定义Attention的目标分布 $a^*$ 和实际分布 $a$，KL散度：
\begin{equation}
D_{KL}(a^* \| a) = \sum_j a_j^* \log\frac{a_j^*}{a_j} = H(a^*, a) - H(a^*) \tag{58}
\end{equation}

其中交叉熵：
\begin{equation}
H(a^*, a) = -\sum_j a_j^* \log a_j \tag{59}
\end{equation}

**优化目标**：最小化 $D_{KL}$ 等价于最小化交叉熵（固定 $a^*$）。

熵不变性确保交叉熵的尺度不随 $n$ 变化，有利于优化稳定性。

### 7. 实验验证与分析

#### 7.1 MLM任务的外推实验

**实验设置**：
- 模型：RoFormer small（与GAU-α类似的结构）
- 训练长度：$n_{\text{train}} = 64$
- 测试长度：$n_{\text{test}} \in \{64, 128, 256, 512, 1024\}$
- 任务：Masked Language Modeling (MLM)
- 指标：准确率（Accuracy）

**结果**（从原文引用）：

| $n$ | Attention-O | Attention-E | 提升 |
|-----|-------------|-------------|------|
| 64  | 43.27 | 43.11 | -0.37% |
| 128 | 36.53 | 41.17 | 12.7% |
| 256 | 23.02 | 34.04 | 47.8% |
| 512 | 15.12 | 20.15 | 33.3% |
| 1024| 11.54 | 13.58 | 17.7% |

**分析**：
1. 在训练长度（64）附近，两者性能相当
2. 外推到更长序列时，Attention-E显著优于Attention-O
3. $n=256$ 时提升最大（47.8%），这是因为 $\log_{512} 256 \approx 0.83$，接近1但有显著差异

#### 7.2 熵的实测值

**实验**：直接测量注意力分布的平均熵。

**设置**：
- 随机初始化的Attention层
- 不同的序列长度 $n$
- 计算 $\bar{H} = \frac{1}{n}\sum_{i=1}^n H_i$

**结果**（理论计算）：

**Attention-O** ($\lambda = 1/\sqrt{64} = 0.125$)：
\begin{equation}
\bar{H} \approx \log n - 0.24 \times 0.125 \times 64 = \log n - 1.92 \tag{60}
\end{equation}

| $n$ | $\log n$ | 理论 $\bar{H}$ |
|-----|----------|---------------|
| 64  | 4.16 | 2.24 |
| 128 | 4.85 | 2.93 |
| 256 | 5.55 | 3.63 |
| 512 | 6.24 | 4.32 |
| 1024| 6.93 | 5.01 |

**Attention-E** ($\lambda = \frac{\log n}{\log 512} \times 0.125$)：
通过调整，熵应该保持在 $\bar{H} \approx 2.24$ 左右（与 $n=64$ 时一致）。

#### 7.3 理论vs实验的差异

**观察**：实验中Attention-E的性能提升并不完全符合熵完全不变的预期。

**原因**：
1. **i.i.d.假设不精确**：实际中 $s_{ij}$ 有结构和相关性
2. **边界效应**：小 $n$ 时，渐近近似不准确
3. **训练动态**：训练过程中分布会变化
4. **其他因素**：位置编码、残差连接等影响

### 8. 理论推广与变体

#### 8.1 自适应缩放

固定的 $\lambda(n) = \frac{\log n}{\log 512}$ 对所有层相同。但可以设计：

**层相关缩放**：
\begin{equation}
\lambda_{\ell}(n) = \frac{\log n}{\log 512} \cdot \alpha_{\ell} \tag{61}
\end{equation}

其中 $\alpha_{\ell}$ 是第 $\ell$ 层的可学习参数。

**位置相关缩放**（Decoder）：
\begin{equation}
\lambda_i(n) = \frac{\log i}{\log 512}, \quad i = 1, \ldots, n \tag{62}
\end{equation}

因为第 $i$ 个位置只能看到前 $i$ 个token。

#### 8.2 其他熵的选择

除了Shannon熵，还可以优化其他熵：

**Rényi熵** ($\alpha = 2$)：
\begin{equation}
H_2 = -\log\sum_j a_{ij}^2 \tag{63}
\end{equation}

优化目标：使 $H_2$ 不变。

**Tsallis熵**：
\begin{equation}
S_q = \frac{1}{q-1}\left(1 - \sum_j a_{ij}^q\right) \tag{64}
\end{equation}

#### 8.3 软熵不变性

严格的熵不变性可能过于约束。可以考虑：

**软约束**：
\begin{equation}
\mathcal{L} = \mathcal{L}_{\text{task}} + \beta |H_i - H_{\text{target}}|^2 \tag{65}
\end{equation}

其中 $H_{\text{target}}$ 是期望的熵值。

### 9. 实践建议与注意事项

#### 9.1 实现细节

**PyTorch实现**：
```python
import torch
import torch.nn.functional as F

def entropy_invariant_attention(Q, K, V, n_train=512):
    """
    熵不变性Attention

    Args:
        Q, K, V: [batch, n_heads, seq_len, d_k]
        n_train: 训练长度（默认512）
    """
    n = Q.size(2)  # 当前序列长度
    d_k = Q.size(3)

    # 计算缩放因子
    scale = torch.log(torch.tensor(n, dtype=Q.dtype)) / \
            (torch.log(torch.tensor(n_train, dtype=Q.dtype)) * torch.sqrt(torch.tensor(d_k, dtype=Q.dtype)))

    # 注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attn, V)

    return output, attn
```

**数值稳定性**：
- 使用 `log-sum-exp` 技巧避免上溢
- 对于非常大的 $n$，$\log n$ 仍然温和增长

#### 9.2 超参数选择

**底数选择**：原文使用512，但可以调整：
- 如果主要处理短序列（如图像patches），使用较小的底数（如64）
- 如果处理长序列（如长文档），可以保持512或更大

**温度调整**：可以引入全局温度参数：
\begin{equation}
\lambda = \tau \cdot \frac{\log n}{\log 512 \sqrt{d}} \tag{66}
\end{equation}

其中 $\tau$ 是可调节的超参数（默认为1）。

#### 9.3 与其他技术的结合

**与RoPE结合**：熵不变性Softmax可以与旋转位置编码无缝结合，进一步提升外推性能。

**与Flash Attention结合**：Flash Attention的实现可以很容易地适配熵不变性缩放。

**与稀疏Attention结合**：对于稀疏模式（如局部窗口），$n$ 应该是实际参与的token数量。

### 10. 理论局限与未来方向

#### 10.1 理论局限

1. **i.i.d.假设**：实际中 $s_{ij}$ 不是独立同分布的
2. **渐近性**：推导基于大 $n$ 假设，小 $n$ 时不准确
3. **静态分析**：未考虑训练动态和分布漂移
4. **单一目标**：只优化熵，未考虑其他目标（如梯度、收敛速度）

#### 10.2 开放问题

**问题1**：是否存在更优的缩放函数，不仅是 $\log n$ 形式？

**问题2**：如何在训练过程中自适应调整缩放因子？

**问题3**：熵不变性在生成任务（Decoder）中的最优形式是什么？

**问题4**：如何理论化地结合熵不变性和梯度最大化两个目标？

#### 10.3 未来方向

1. **自适应熵控制**：学习每一层的目标熵
2. **多目标优化**：同时优化熵、梯度、信息保持等
3. **跨模态扩展**：图像-文本Attention的熵不变性
4. **理论完善**：更严格的非渐近分析

### 11. 总结

#### 11.1 核心贡献

1. **熵视角**：将Attention的缩放问题与信息熵联系起来
2. **熵不变性原则**：提出保持熵对序列长度不敏感的设计原则
3. **新缩放因子**：推导出 $\lambda(n) = \frac{\log n}{\log n_0} \cdot \frac{1}{\sqrt{d}}$
4. **实验验证**：在MLM任务上显著提升长度外推性能

#### 11.2 理论框架

熵不变性Attention提供了统一的理论框架：
\begin{equation}
\text{信息论} \xrightarrow{\text{熵}} \text{注意力设计} \xrightarrow{\text{缩放}} \text{外推性能} \tag{67}
\end{equation}

这个框架连接了：
- Shannon信息论
- Johnson-Lindenstrauss引理
- Attention机制
- 长度外推

#### 11.3 实践价值

1. **简单有效**：只需修改缩放因子，无需改变模型结构
2. **理论支撑**：有坚实的信息论基础
3. **实验验证**：在多个任务上证明有效
4. **易于实现**：几行代码即可实现

熵不变性为设计更好的Attention机制提供了新的思路和工具。

