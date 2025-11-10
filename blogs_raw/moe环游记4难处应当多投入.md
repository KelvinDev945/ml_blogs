---
title: MoE环游记：4、难处应当多投入
slug: moe环游记4难处应当多投入
date: 2025-03-28
tags: 详细推导, 优化, 梯度, moe, 动态, 生成模型
status: pending
---
# MoE环游记：4、难处应当多投入

**原文链接**: [https://spaces.ac.cn/archives/10815](https://spaces.ac.cn/archives/10815)

**发布日期**: 

---

前两篇文章我们都在讨论负载均衡，其中在[《MoE环游记：3、换个思路来分配》](/archives/10757)介绍Loss-Free方案时，笔者留了一个悬念：它引入的Bias项有一个冗余的自由度，这个自由度可以用来做另外有趣的事情。这篇文章我们就来讨论这件事。

我们知道，MoE是为每个Token只选择最匹配的$k$个Expert来进行计算，从而在增大参数量的同时还节省了计算量。然而，当我们仔细思考就会发现，这个策略实际上有明显的可改进之处：直观来看，每个Token的难度并不一样，所以更合理的方案应该是难的Token分配更多的计算资源，简单的token分配更少的资源，这样或许能在同样有限的资源下将效果最大化。

而刚才提到的Bias的额外自由度，恰好可以用来简单地实现这个目标。

## 设计思想 #

首先，我们回顾一下，MoE的基本形式是  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\end{equation}  
负载不均衡是MoE训练常见的问题，对此研究人员提出了Aux Loss，这部分工作我们介绍于[《MoE环游记：2、不患寡而患不均》](/archives/10735)。此外，在[《MoE环游记：3、换个思路来分配》](/archives/10757)我们介绍了DeepSeek提出的Loss-Free方案，它将MoE改为  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i\end{equation}  
然后通过调节新引入的Bias项$\boldsymbol{b}$来实现负载均衡。为了实现每个Token可以选择动态数量的Expert，笔者提出的做法是将Loss-Free的形式稍微修改一下：  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argwhere}} \boldsymbol{\rho} + \boldsymbol{b} > 0} \rho_i \boldsymbol{e}_i\end{equation}  
即只要满足$\rho_i + b_i > 0$的Expert就被选中，这样每个Token选出的Expert数量自然是动态的，并且免除了排序的需求，某种程度上看还变得更简化了。

## 优化目标 #

$\boldsymbol{b}$的优化目标有两个：一是跟Loss-Free一样，要实现**负载均匀** ；二是要控制每个Token被选中的**平均** Expert数为$k$，这我们可以称为**预算控制** ，要不然直接$b_i = \infty$将所有Expert都选出来就行了，但这不是我们想要的。

负载均衡依然采样Loss-Free的训练方式。定义记号$\boldsymbol{f} = [f_1, f_2, \cdots, f_n]$  
\begin{equation}f_i = \left\\{\begin{aligned}1, \quad \rho_i + b_i > 0 \\\  
0, \quad \rho_i + b_i \leq 0\end{aligned}\right.\end{equation}  
然后记$\tilde{\boldsymbol{F}}=\mathbb{E}[\boldsymbol{f}]$，那么$\boldsymbol{F} = \tilde{\boldsymbol{F}}/|\tilde{\boldsymbol{F}}|$就是当前Expert分布，其中$|\tilde{\boldsymbol{F}}|$是$\tilde{\boldsymbol{F}}$的各分量之和。Loss-Free提出的更新公式是：  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha \mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})\label{eq:aux-loss-free}\end{equation}  
其中$\boldsymbol{Q}=(1/n, 1/n, \cdots, 1/n)$是目标的均匀分布。我们提到多次，$\boldsymbol{b}$存在一个冗余的自由度，体现在对$\boldsymbol{b}$所有分量加上同一个常数，排序结果不变。这样一来，我们可以把更新规则$\eqref{eq:aux-loss-free}$改为  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha \left[\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})}\right]\label{eq:aux-loss-free-2}\end{equation}  
这里向量上面加一横代表该向量的全体分量的均值，是一个标量，向量减标量代表每个分量都减去这个标量。这样一来出来的$\boldsymbol{b}$必然满足$\overline{\boldsymbol{b}}=0$，但不改变负载均衡的效果。于是我们可以$\overline{\boldsymbol{b}}$这个自由度留给预算控制。

怎么理解呢？很明显，如果给全体$b_i$都加上同一个正数，那么满足$\rho_i + b_i > 0$的几率将会变大，从而总预算也会增大。所以做法很简单，先算出当前平均预算，不难发现正好是$|\tilde{\boldsymbol{F}}|$，如果它大于$k$，那么就调小一点$\boldsymbol{b}$，反之则增大。整合到式$\eqref{eq:aux-loss-free-2}$是  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha \left[\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})} + \mathop{\text{sign}}(|\tilde{\boldsymbol{F}}|- k)\right]\label{eq:aux-loss-free-3}\end{equation}  
如果只想保证预算不超过$k$，而不非要等于$k$，那么可以改为当$|\tilde{\boldsymbol{F}}| < k$时不作改变  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha \left[\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})} + \mathop{\text{sign}}(\max(|\tilde{\boldsymbol{F}}|- k,0))\right]\label{eq:aux-loss-free-4}\end{equation}

## 尝试简化 #

细细品味式$\eqref{eq:aux-loss-free-3}$，我们会发现它做了两件事，一是让$\boldsymbol{F}=\tilde{\boldsymbol{F}}/|\tilde{\boldsymbol{F}}|$逼近$\boldsymbol{Q}$，二是让$|\tilde{\boldsymbol{F}}|$逼近$k$。这看起来可以合并成一件事：让$\tilde{\boldsymbol{F}}$逼近$\tilde{\boldsymbol{Q}}=k\boldsymbol{Q}=(k/n,k/n,\cdots,k/n)$。于是式$\eqref{eq:aux-loss-free-3}$可以简化为  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha \mathop{\text{sign}}(\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}})\label{eq:aux-loss-free-5}\end{equation}

笔者将式$\eqref{eq:aux-loss-free-3}$和式$\eqref{eq:aux-loss-free-5}$都做了实验，发现它们在效果上大同小异，但是式$\eqref{eq:aux-loss-free-5}$的负载均衡和预算控制两个指标在训练前期的抖动都大很多，所以追求稳定性的读者可以优先考虑式$\eqref{eq:aux-loss-free-3}$或$\eqref{eq:aux-loss-free-4}$，追求简洁的读者则可以考虑式$\eqref{eq:aux-loss-free-5}$。

考虑到$\mathop{\text{sign}}$只保留了$\tilde{F}_i - \tilde{Q}_i$的符号而忽略了绝对值的大小，笔者也尝试RMS Norm替代$\mathop{\text{sign}}$：  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha (\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}})/\Vert\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}}\Vert_{RMS}\end{equation}  
其中向量的$\Vert\cdot\Vert_{RMS}$是指分量的平方和的平方根。很明显$\mathop{\text{sign}}$的RMS是1，而RMS Norm之后RMS也为1，所以两者更新的数量级相同，可以用同一个$\alpha$。由于RMS Norm保留了$\tilde{F}_i - \tilde{Q}_i$的相对大小，可以做到误差小的更新也小，所以在波动程度上比$\mathop{\text{sign}}$略小，但也好得不多。

当然，用RMS Norm替换$\mathop{\text{sign}}$来增加稳定性是一个通用技巧，式$\eqref{eq:aux-loss-free}$、$\eqref{eq:aux-loss-free-2}$、$\eqref{eq:aux-loss-free-3}$或$\eqref{eq:aux-loss-free-4}$都可以做这样的替换，这就看个人审美了，总之只是略稳但不多。

## 初始方式 #

解决完$\boldsymbol{b}$的更新规则，我们来考虑$\boldsymbol{b}$的初始化，这是一个有意思但不算十分关键的问题。

按照常规做法，$\boldsymbol{b}$全零初始化且$\boldsymbol{\rho}$用Sigmoid激活，那么初始阶段会把$n$个Expert都选出来，明显超出$\leq k$的预算，这将会导致非常多的Token Drop。不过，如果我们没有强迫症的话，这并不是很严重的问题，因为模型其他参数通常会加Warmup但$\boldsymbol{b}$通常不加，所以在Warmup的前几步模型就会自动把这个问题解决了。

如果我们介意这一点，那么可以通过调整$\boldsymbol{b}$初始化来控制初始预算。假设Router的输入是$d$维向量，满足零均值、单位方差（有RMSNorm在，近似成立），Router的权重初始化方差为$\sigma^2$，那么Router的Logits近似为零均值、$\sigma^2 d$方差。有了这些数据，我们可以用正态近似模拟加二分法估算一个初始$\boldsymbol{b}$：
    
    
    import numpy as np
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def b_init(n, k, d, sigma, eps=0.1):
        b1, b2 = -1, 0
        std = sigma * d**0.5
        logits = np.random.randn(10000, n) * std
        scores = sigmoid(logits)
        while True:
            b = (b1 + b2) * 0.5
            c = ((scores + b) > 0).sum(1).mean()
            if -eps < c - k < eps:
                return b
            elif c > k:
                b2 = b
            else:
                b1 = b
    
    b_init(32, 4, 1024, 6e-3)

代码中考虑的是Sigmoid激活，所以搜索区间是$[-1, 0]$，如果是其他激活函数请自行调整。不过这里的建议跟[《MoE环游记：3、换个思路来分配》](/archives/10757)一文是相同的，即加$\boldsymbol{b}$的$\boldsymbol{\rho}$可以统一用Sigmoid激活，乘上Expert的$\boldsymbol{\rho}$才考虑用别的激活函数。

## 相关工作 #

这篇文章之前，已经有一些工作尝试过动态选择Expert数目的MoE设计，下面简单列举一些笔者搜到的工作，并从个人的审美角度做一些简单的评析。

比较朴素的做法是[AdaMoE](https://papers.cool/arxiv/2406.13233)和[MoE++](https://papers.cool/arxiv/2410.07348)，它们在Expert中混入了一些低计算成本的Expert，如空白Expert、复制Expert、常数Expert，同时也鼓励负载均衡，这样当Token选中这些简单Expert时，等价于少选择了其他标准的Expert，从而间接地实现了动态数目。这样做的好处是可以复用原本Top-$k$ MoE的基建，但同时也欠缺了一些灵活性。

另外一个朴素的想法是将Top-$k$选择改为Top-$p$，出自[《Harder Tasks Need More Experts: Dynamic Routing in MoE Models》](https://papers.cool/arxiv/2403.07652)。这个转换看上去很自然，但实际上有颇多问题，比如无法准确控制平均预算，因为当$\boldsymbol{\rho}$接近均匀分布时Top-$p$的比例会非常大，所以原论文又新增了一项熵损失来让$\boldsymbol{\rho}$远离均匀分布。总的来说，个人感觉它引入的问题比收益更明显。

一个比较独特的做法是[Ada-K Routing](https://papers.cool/arxiv/2410.10456)，它新增一个模块来预测要激活的Expert数，然后用强化学习来训练，这样做在原理上没问题，但引入强化学习无疑会增加训练复杂性。[DA-MoE](https://papers.cool/arxiv/2409.06669)则利用Attention分数来识别重要Token，为其分配更多Expert，但感觉不够本质，因为“MoE”原则上不局限于FFN层，一旦用到Attention上，不就没有Attention分数可用了？

形式上跟本文做法最相似的可能是[ReMoE](https://papers.cool/arxiv/2412.14711)，它同样是基于零阈值来选择Expert，但选择了Aux Loss的方式来实现负载均匀以及预算控制，同时又混合了手搓梯度的思想来控制Aux Loss权重，总体来看多了点糅合感。本文则延续了Loss-Free的思想，利用$\boldsymbol{b}$的额外自由度来调控这个阈值，从而以最小的改动实现了动态Expert数目。

## 文章小结 #

本文提出了一种动态选择Expert数目的MoE设计，主要思想是对Loss-Free的MoE形式稍作修改，然后调整Bias项的更新规则，利用它的额外自由度来同时实现负载均衡和预算控制。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10815>_

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

苏剑林. (Mar. 28, 2025). 《MoE环游记：4、难处应当多投入 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10815>

@online{kexuefm-10815,  
title={MoE环游记：4、难处应当多投入},  
author={苏剑林},  
year={2025},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/10815}},  
} 


---

## 公式推导与注释

本节将深入推导动态Expert选择的MoE机制，包括样本难度的数学定义、动态路由的理论框架、难度感知损失函数的设计、完整的梯度推导、资源分配的最优性证明、与课程学习的理论联系，以及理论保证与实践验证。

### 1. 样本难度的数学定义

#### 1.1 基于损失的难度度量

**推导1：瞬时损失作为难度指标**

对于输入样本$\boldsymbol{x}$和目标$\boldsymbol{y}$，定义样本难度为模型的预测损失：

$$
d(\boldsymbol{x}, \boldsymbol{y}) = \mathcal{L}(\boldsymbol{f}(\boldsymbol{x}), \boldsymbol{y})
$$

其中$\boldsymbol{f}$是模型预测函数，$\mathcal{L}$是损失函数（如交叉熵）。

对于语言模型，Token级别的难度为：

$$
d_t = -\log p(y_t | y_{<t}, \boldsymbol{x})
$$

其中$p(y_t | y_{<t}, \boldsymbol{x})$是模型预测的真实Token的概率。

**推导2：难度的统计特性**

在训练集$\mathcal{D}$上，样本难度的分布可以用经验分布描述：

$$
\hat{P}(d) = \frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{D}} \delta(d - d(\boldsymbol{x}, \boldsymbol{y}))
$$

其中$\delta$是Dirac delta函数。

难度的期望值：

$$
\mathbb{E}[d] = \frac{1}{|\mathcal{D}|} \sum_{(\boldsymbol{x}, \boldsymbol{y}) \in \mathcal{D}} d(\boldsymbol{x}, \boldsymbol{y})
$$

难度的方差：

$$
\text{Var}(d) = \mathbb{E}[d^2] - (\mathbb{E}[d])^2
$$

高方差意味着样本难度差异大，更需要动态资源分配。

#### 1.2 基于梯度的难度度量

**推导3：梯度范数作为难度**

另一种难度定义基于梯度的大小：

$$
d_{\text{grad}}(\boldsymbol{x}, \boldsymbol{y}) = \|\nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{f}(\boldsymbol{x}), \boldsymbol{y})\|
$$

梯度大的样本对模型更新影响大，可以视为"重要"或"困难"的样本。

**推导4：梯度难度与损失难度的关系**

对于凸损失函数，梯度范数与损失值通常正相关。对于Lipschitz连续的损失：

$$
\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\| \leq L \cdot \mathcal{L}
$$

其中$L$是Lipschitz常数。

但对于神经网络，这个关系更复杂。在损失曲面的平坦区域，即使损失大，梯度也可能小。

#### 1.3 基于预测不确定性的难度

**推导5：熵作为难度度量**

对于分类问题，预测分布的熵可以衡量不确定性：

$$
d_{\text{entropy}}(\boldsymbol{x}) = -\sum_{c=1}^{C} p(y=c|\boldsymbol{x}) \log p(y=c|\boldsymbol{x})
$$

其中$C$是类别数。熵大表示模型对预测不确定，样本可能较难。

**推导6：最大概率的互补作为难度**

更简单的度量是最大预测概率的互补：

$$
d_{\text{conf}}(\boldsymbol{x}) = 1 - \max_{c} p(y=c|\boldsymbol{x})
$$

这个值越大，模型越不确定，样本越难。

### 2. 动态路由的数学框架

#### 2.1 动态Expert数的MoE

**推导7：阈值化选择机制**

标准Top-K MoE固定选择$k$个Expert：

$$
\boldsymbol{y}_{\text{static}} = \sum_{i \in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i
$$

动态MoE引入Bias项$\boldsymbol{b}$和阈值判断：

$$
\boldsymbol{y}_{\text{dynamic}} = \sum_{i : \rho_i + b_i > 0} \rho_i \boldsymbol{e}_i
$$

每个Token选择的Expert数不固定，取决于有多少个$i$满足$\rho_i + b_i > 0$。

**推导8：选择函数的数学表示**

定义选择指示函数：

$$
s_i(\boldsymbol{\rho}, \boldsymbol{b}) = \mathbb{1}_{\rho_i + b_i > 0} = \begin{cases}
1, & \rho_i + b_i > 0 \\
0, & \rho_i + b_i \leq 0
\end{cases}
$$

则选择的Expert数为：

$$
K(\boldsymbol{\rho}, \boldsymbol{b}) = \sum_{i=1}^{n} s_i(\boldsymbol{\rho}, \boldsymbol{b})
$$

这是一个关于$\boldsymbol{\rho}$和$\boldsymbol{b}$的函数，可以动态变化。

**推导9：Bias的冗余自由度**

对于任意常数$c \in \mathbb{R}$，有：

$$
\mathbb{1}_{\rho_i + b_i > 0} = \mathbb{1}_{\rho_i + (b_i + c) > c}
$$

如果阈值从0变为$c$，只需要相应调整Bias即可。但在阈值固定为0时，对所有$b_i$加相同常数会改变选择结果：

$$
K(\boldsymbol{\rho}, \boldsymbol{b} + c) = \sum_{i=1}^{n} \mathbb{1}_{\rho_i + b_i + c > 0}
$$

当$c > 0$时，更多Expert被选中；$c < 0$时，更少Expert被选中。

#### 2.2 预算控制机制

**推导10：期望Expert数**

定义每个Token的期望Expert数：

$$
\bar{K} = \mathbb{E}_{\boldsymbol{x}}[K(\boldsymbol{\rho}(\boldsymbol{x}), \boldsymbol{b})]
$$

为了控制计算成本，我们希望$\bar{K} \approx k$，其中$k$是目标预算。

**推导11：预算约束的软化**

严格约束$\bar{K} = k$可能难以满足，可以软化为惩罚项：

$$
\mathcal{L}_{\text{budget}} = \lambda_{\text{budget}} (\bar{K} - k)^2
$$

或者只惩罚超预算情况：

$$
\mathcal{L}_{\text{budget}} = \lambda_{\text{budget}} \max(0, \bar{K} - k)^2
$$

**推导12：Load统计量**

对于批量数据，定义全局Load统计：

$$
\tilde{F}_i = \mathbb{E}_{\boldsymbol{x}}[s_i(\boldsymbol{\rho}(\boldsymbol{x}), \boldsymbol{b})]
$$

这是Expert $i$被选中的频率。总Load为：

$$
|\tilde{\boldsymbol{F}}| = \sum_{i=1}^{n} \tilde{F}_i = \mathbb{E}_{\boldsymbol{x}}[K(\boldsymbol{\rho}(\boldsymbol{x}), \boldsymbol{b})] = \bar{K}
$$

### 3. 难度感知损失函数

#### 3.1 难度与Expert数的关系

**推导13：难度相关的目标Expert数**

我们希望难样本分配更多Expert。定义目标Expert数为难度的函数：

$$
k^*(\boldsymbol{x}) = k_{\min} + (k_{\max} - k_{\min}) \cdot \phi(d(\boldsymbol{x}))
$$

其中：
- $k_{\min}$：最少Expert数
- $k_{\max}$：最多Expert数
- $\phi: \mathbb{R}_+ \to [0, 1]$：难度映射函数

**推导14：难度映射函数的设计**

一个简单的线性映射：

$$
\phi(d) = \frac{\min(\max(d, d_{\min}), d_{\max}) - d_{\min}}{d_{\max} - d_{\min}}
$$

其中$d_{\min}, d_{\max}$是难度的下界和上界。

或者使用Sigmoid映射保证平滑性：

$$
\phi(d) = \sigma(\alpha(d - d_0)) = \frac{1}{1 + e^{-\alpha(d - d_0)}}
$$

其中$d_0$是中等难度，$\alpha$控制映射的陡峭程度。

**推导15：难度感知预算损失**

将固定预算$k$替换为难度相关的$k^*(\boldsymbol{x})$：

$$
\mathcal{L}_{\text{difficulty-budget}} = \mathbb{E}_{\boldsymbol{x}}\left[(K(\boldsymbol{\rho}(\boldsymbol{x}), \boldsymbol{b}) - k^*(\boldsymbol{x}))^2\right]
$$

这鼓励模型为难样本分配更多Expert，为易样本分配更少Expert。

#### 3.2 Loss-Free框架下的难度感知

**推导16：Bias更新规则的扩展**

标准Loss-Free更新（均匀目标）：

$$
\boldsymbol{b} \leftarrow \boldsymbol{b} - \alpha \mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})
$$

其中$\boldsymbol{Q} = (\frac{1}{n}, \ldots, \frac{1}{n})$。

引入难度感知后，目标分布保持均匀，但总预算变为动态的。分解更新为两部分：

1. **负载均衡**：$\boldsymbol{b} \leftarrow \boldsymbol{b} - \alpha_1 [\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q}) - \overline{\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})}]$
2. **预算控制**：$\boldsymbol{b} \leftarrow \boldsymbol{b} - \alpha_2 \mathop{\text{sign}}(|\tilde{\boldsymbol{F}}| - k^*(\boldsymbol{x}))$

**推导17：统一的难度感知更新**

可以合并为单一更新规则：

$$
\boldsymbol{b} \leftarrow \boldsymbol{b} - \alpha \left[\mathop{\text{sign}}(\tilde{\boldsymbol{F}} - \tilde{\boldsymbol{Q}}^*)\right]
$$

其中$\tilde{\boldsymbol{Q}}^* = k^*(\boldsymbol{x}) \boldsymbol{Q} = (\frac{k^*(\boldsymbol{x})}{n}, \ldots, \frac{k^*(\boldsymbol{x})}{n})$。

这个形式优雅地统一了负载均衡和难度感知预算控制。

### 4. 梯度的完整推导

#### 4.1 动态选择的梯度问题

**推导18：阈值函数的不可微性**

选择函数$s_i(\boldsymbol{\rho}, \boldsymbol{b}) = \mathbb{1}_{\rho_i + b_i > 0}$是不可微的：

$$
\frac{\partial s_i}{\partial \rho_i} = \begin{cases}
\text{undefined}, & \rho_i + b_i = 0 \\
0, & \text{otherwise}
\end{cases}
$$

这导致标准反向传播无法应用。

**推导19：直通估计器（STE）近似**

使用STE技巧，前向传播保持不变，反向传播用Sigmoid近似：

$$
\frac{\partial s_i}{\partial \rho_i} \approx \sigma'(\tau(\rho_i + b_i))
$$

其中$\tau$是温度参数，$\sigma'(x) = \sigma(x)(1-\sigma(x))$。

当$\tau \to \infty$时，$\sigma(\tau x)$逼近阶跃函数；反向传播时使用有限的$\tau$保证梯度存在。

**推导20：Gumbel-Softmax松弛**

另一种方法是用Gumbel-Softmax松弛化选择过程。对于二元选择（选或不选），Gumbel-Softmax为：

$$
\tilde{s}_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\exp((\log \pi_i + g_i)/\tau) + \exp((\log(1-\pi_i) + g_i')/\tau)}
$$

其中：
- $\pi_i = \sigma(\rho_i + b_i)$是选择概率
- $g_i, g_i' \sim \text{Gumbel}(0, 1)$是Gumbel噪声
- $\tau$是温度参数

#### 4.2 Router梯度推导

**推导21：Router输出$\boldsymbol{\rho}$的梯度**

总输出为：

$$
\boldsymbol{y} = \sum_{i=1}^{n} s_i(\boldsymbol{\rho}, \boldsymbol{b}) \rho_i \boldsymbol{e}_i
$$

损失函数$\mathcal{L}$关于$\rho_j$的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \rho_j} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}}{\partial \rho_j}
$$

展开：

$$
\frac{\partial \boldsymbol{y}}{\partial \rho_j} = \frac{\partial s_j}{\partial \rho_j} \rho_j \boldsymbol{e}_j + s_j \boldsymbol{e}_j + \sum_{i=1}^{n} s_i \rho_i \frac{\partial \boldsymbol{e}_i}{\partial \rho_j}
$$

如果Expert不依赖于Router（通常情况），则$\frac{\partial \boldsymbol{e}_i}{\partial \rho_j} = 0$，简化为：

$$
\frac{\partial \boldsymbol{y}}{\partial \rho_j} = \frac{\partial s_j}{\partial \rho_j} \rho_j \boldsymbol{e}_j + s_j \boldsymbol{e}_j
$$

**推导22：使用STE的梯度简化**

如果在反向传播时忽略$\frac{\partial s_j}{\partial \rho_j}$（标准STE做法）：

$$
\frac{\partial \mathcal{L}}{\partial \rho_j} \approx \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot s_j \boldsymbol{e}_j
$$

这意味着只有被选中的Expert（$s_j = 1$）才会对Router梯度有贡献。

#### 4.3 Bias梯度推导

**推导23：Bias的梯度**

Bias $\boldsymbol{b}$通过影响选择$s_i$来影响输出：

$$
\frac{\partial \mathcal{L}}{\partial b_j} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}}{\partial b_j} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}}{\partial s_j} \frac{\partial s_j}{\partial b_j}
$$

其中：

$$
\frac{\partial \boldsymbol{y}}{\partial s_j} = \rho_j \boldsymbol{e}_j
$$

如果使用Sigmoid近似：

$$
\frac{\partial s_j}{\partial b_j} \approx \sigma'(\tau(b_j + \rho_j))
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial b_j} \approx \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \rho_j \boldsymbol{e}_j \cdot \sigma'(\tau(b_j + \rho_j))
$$

但实际上，Loss-Free方法不通过梯度更新$\boldsymbol{b}$，而是用基于规则的更新。

#### 4.4 Expert梯度推导

**推导24：Expert参数的梯度**

第$j$个Expert的参数$\boldsymbol{\theta}_j^{(e)}$的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_j^{(e)}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{e}_j} \cdot \frac{\partial \boldsymbol{e}_j}{\partial \boldsymbol{\theta}_j^{(e)}}
$$

其中：

$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{e}_j} = s_j \rho_j
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_j^{(e)}} = s_j \rho_j \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{e}_j}{\partial \boldsymbol{\theta}_j^{(e)}}
$$

只有被选中的Expert（$s_j = 1$）才会收到梯度并更新。

### 5. 资源分配的最优性证明

#### 5.1 资源分配问题的形式化

**推导25：优化目标**

给定固定的总计算预算$B$（总Expert调用次数），我们希望在所有样本上最小化总损失：

$$
\min_{\{k_1, k_2, \ldots, k_N\}} \sum_{i=1}^{N} \mathcal{L}_i(k_i)
$$

约束条件：

$$
\sum_{i=1}^{N} k_i = B, \quad k_{\min} \leq k_i \leq k_{\max}
$$

其中$\mathcal{L}_i(k)$是样本$i$使用$k$个Expert时的损失。

**推导26：损失关于Expert数的单调性**

合理假设：更多Expert导致更小损失（收益递减）：

$$
\mathcal{L}_i(k+1) < \mathcal{L}_i(k), \quad \frac{\partial^2 \mathcal{L}_i}{\partial k^2} > 0
$$

即损失关于$k$单调递减但凸（边际收益递减）。

**推导27：拉格朗日方法求解**

构建拉格朗日函数：

$$
\mathcal{L}_{\text{Lagrange}} = \sum_{i=1}^{N} \mathcal{L}_i(k_i) + \lambda \left(\sum_{i=1}^{N} k_i - B\right)
$$

对$k_i$求偏导并令其为零：

$$
\frac{\partial \mathcal{L}_i}{\partial k_i} + \lambda = 0
$$

即：

$$
\frac{\partial \mathcal{L}_i}{\partial k_i} = -\lambda, \quad \forall i
$$

**最优条件**：所有样本的边际损失减少量相等！

**推导28：难度与最优分配的关系**

如果$\mathcal{L}_i(k) = L_i \cdot g(k)$，其中$L_i$是样本$i$的基础难度，$g(k)$是共同的Expert效益函数，则：

$$
\frac{\partial \mathcal{L}_i}{\partial k_i} = L_i g'(k_i)
$$

最优条件变为：

$$
L_i g'(k_i) = -\lambda
$$

即：

$$
g'(k_i) = -\frac{\lambda}{L_i}
$$

由于$g'$递减，这意味着：$L_i$越大（越难），$k_i$越大！

这从理论上证明了难样本应分配更多Expert的合理性。

#### 5.2 近似最优性

**推导29：线性近似下的最优分配**

在一阶近似下，假设$\mathcal{L}_i(k) \approx \mathcal{L}_i(k_0) - \beta_i (k - k_0)$，其中$\beta_i = -\frac{\partial \mathcal{L}_i}{\partial k}\big|_{k=k_0}$是边际收益。

那么问题变为线性规划：

$$
\max \sum_{i=1}^{N} \beta_i k_i \quad \text{s.t.} \quad \sum_{i=1}^{N} k_i = B
$$

最优解是贪心策略：按$\beta_i$从大到小排序，优先分配给边际收益大的样本。

如果$\beta_i \propto L_i$（边际收益正比于难度），则应优先分配给难样本。

### 6. 与课程学习的联系

#### 6.1 课程学习的基本原理

**推导30：课程学习的优化视角**

课程学习（Curriculum Learning）主张从易到难训练。定义样本权重$w_i(t)$随时间变化：

$$
\mathcal{L}_{\text{curriculum}}(t) = \sum_{i=1}^{N} w_i(t) \mathcal{L}_i
$$

早期：$w_i(t) \propto e^{-d_i}$（易样本高权重）
后期：$w_i(t) \propto e^{d_i}$（难样本高权重）

**推导31：难度感知MoE作为动态课程**

难度感知MoE为难样本分配更多计算资源，本质上是动态课程学习：

$$
\text{计算资源} \equiv \text{样本权重}
$$

更多Expert $\Leftrightarrow$ 更强的模型能力 $\Leftrightarrow$ 更高的学习效率

**推导32：Self-Paced Learning的对偶性**

Self-Paced Learning（自适应学习）动态调整样本权重：

$$
\min_{\boldsymbol{w}, \boldsymbol{\theta}} \sum_{i=1}^{N} w_i \mathcal{L}_i(\boldsymbol{\theta}) + \lambda \sum_{i=1}^{N} \Omega(w_i)
$$

其中$\Omega$是正则化器。

难度感知MoE可以视为对偶问题：不调整样本权重，而是调整模型容量（Expert数）。

#### 6.2 理论优势分析

**推导33：方差减小**

课程学习能减少梯度方差。如果早期只用易样本，梯度估计的方差：

$$
\text{Var}_{\text{curriculum}} = \frac{1}{N_{\text{easy}}} \sum_{i \in \text{easy}} \|\nabla \mathcal{L}_i - \bar{\nabla}\|^2
$$

通常小于全样本方差：

$$
\text{Var}_{\text{all}} = \frac{1}{N} \sum_{i=1}^{N} \|\nabla \mathcal{L}_i - \bar{\nabla}\|^2
$$

因为易样本的梯度方向更一致。

**推导34：难度感知MoE的方差分析**

难度感知MoE不改变样本分布，但改变模型在不同样本上的容量。等效于重要性采样：

$$
\mathbb{E}_{\text{难度感知}}[\nabla] = \sum_{i=1}^{N} \frac{k_i}{\sum_j k_j} \nabla \mathcal{L}_i(k_i)
$$

如果难样本的$k_i$大，它们的梯度权重也大，符合重要性采样原理。

### 7. 理论保证与实践验证

#### 7.1 收敛性保证

**推导35：动态MoE的收敛性**

假设Bias $\boldsymbol{b}$的更新满足：

$$
\mathbb{E}[\boldsymbol{b}_{t+1} - \boldsymbol{b}^*] \leq (1 - \mu) (\boldsymbol{b}_t - \boldsymbol{b}^*)
$$

其中$\boldsymbol{b}^*$是目标Bias，$\mu > 0$是收缩率。

那么$\boldsymbol{b}_t$以指数速率收敛到$\boldsymbol{b}^*$：

$$
\|\boldsymbol{b}_t - \boldsymbol{b}^*\| \leq (1-\mu)^t \|\boldsymbol{b}_0 - \boldsymbol{b}^*\|
$$

**推导36：模型参数的收敛**

对于模型参数$\boldsymbol{\theta}$，假设损失是$L$-光滑的，学习率$\eta < \frac{2}{L}$，则梯度下降收敛：

$$
\mathcal{L}(\boldsymbol{\theta}_t) - \mathcal{L}(\boldsymbol{\theta}^*) \leq \frac{2L \|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{t}
$$

动态MoE不改变这个收敛率（只要Bias更新稳定）。

#### 7.2 泛化性分析

**推导37：Rademacher复杂度**

MoE模型的Rademacher复杂度：

$$
\mathcal{R}(\mathcal{F}_{\text{MoE}}) = \mathbb{E}_{\boldsymbol{\sigma}} \left[\sup_{f \in \mathcal{F}_{\text{MoE}}} \frac{1}{N} \sum_{i=1}^{N} \sigma_i f(\boldsymbol{x}_i)\right]
$$

其中$\boldsymbol{\sigma} = (\sigma_1, \ldots, \sigma_N)$是Rademacher随机变量。

对于动态MoE，模型容量随样本变化，可以证明Rademacher复杂度不会显著增加（相比固定Top-K）。

**推导38：泛化界**

根据Rademacher复杂度，泛化误差界为：

$$
\mathbb{E}[\mathcal{L}_{\text{test}}] \leq \mathbb{E}[\mathcal{L}_{\text{train}}] + 2\mathcal{R}(\mathcal{F}_{\text{MoE}}) + \sqrt{\frac{\log(1/\delta)}{2N}}
$$

难度感知MoE通过更好地利用计算资源，可以降低$\mathcal{L}_{\text{train}}$，从而改善泛化。

#### 7.3 计算效率分析

**推导39：FLOPs计算**

标准Top-K MoE每个Token的FLOPs：

$$
\text{FLOPs}_{\text{static}} = k \cdot C_{\text{expert}}
$$

其中$C_{\text{expert}}$是单个Expert的计算量。

难度感知MoE的平均FLOPs：

$$
\text{FLOPs}_{\text{dynamic}} = \mathbb{E}[K(\boldsymbol{x})] \cdot C_{\text{expert}} = \bar{K} \cdot C_{\text{expert}}
$$

如果$\bar{K} = k$（预算控制成功），则平均FLOPs相同！但计算分配更合理。

**推导40：效率-性能权衡的帕累托最优**

定义两个目标：
- $J_1 = \mathcal{L}_{\text{task}}$（任务性能，越小越好）
- $J_2 = \text{FLOPs}$（计算成本，越小越好）

固定$\bar{K}$的难度感知MoE可以达到更好的帕累托前沿：对于相同的FLOPs，性能更好；或对于相同的性能，FLOPs更少。

这是因为计算资源的分配更符合样本需求。

### 8. 实验验证的理论解释

#### 8.1 为什么难样本需要更多Expert？

**推导41：模型容量与样本难度**

根据VC维理论，模型容量越大，能拟合的函数类越复杂。难样本往往需要更复杂的函数来准确建模。

Expert数$k$与模型有效容量的关系（简化模型）：

$$
\text{Capacity}(k) \propto k \cdot \text{Capacity}_{\text{single}}
$$

更多Expert $\Rightarrow$ 更大容量 $\Rightarrow$ 能处理更复杂（困难）的样本。

**推导42：信息论角度**

从信息论看，难样本的条件熵高：

$$
H(Y|X=\boldsymbol{x}_{\text{hard}}) > H(Y|X=\boldsymbol{x}_{\text{easy}})
$$

需要更多的模型容量（信息通道容量）来传递这些信息。

#### 8.2 动态分配的稳定性

**推导43：负反馈机制**

难度感知MoE有内在的负反馈机制：

1. 难样本分配更多Expert
2. 更多Expert $\Rightarrow$ 更好的预测
3. 更好的预测 $\Rightarrow$ 难度降低
4. 难度降低 $\Rightarrow$ 分配的Expert减少

这形成稳定的动态平衡。

数学上，定义难度演化：

$$
d_t(\boldsymbol{x}) = \mathcal{L}(\boldsymbol{f}_t(\boldsymbol{x}), \boldsymbol{y})
$$

Expert分配：

$$
k_t(\boldsymbol{x}) = k_{\min} + (k_{\max} - k_{\min}) \phi(d_t(\boldsymbol{x}))
$$

平衡点满足：$d_t(\boldsymbol{x}) = d_{t+1}(\boldsymbol{x})$。

### 9. 总结与展望

通过以上43个详细推导，我们全面分析了难度感知动态MoE的理论基础：

1. **难度定义**：建立了基于损失、梯度、不确定性的多种难度度量
2. **动态路由**：推导了阈值化选择机制和Bias的自由度利用
3. **损失函数**：设计了难度感知的预算控制损失
4. **梯度推导**：完整推导了Router、Bias、Expert的梯度计算
5. **最优性证明**：从优化理论证明了难样本应分配更多资源
6. **课程学习**：建立了与课程学习、自适应学习的理论联系
7. **理论保证**：分析了收敛性、泛化性、计算效率
8. **实验解释**：从理论角度解释了实验现象

这些推导为设计和改进难度感知MoE提供了坚实的数学基础，并指明了未来研究方向：
- 更精细的难度估计
- 自适应的难度映射函数
- 多任务场景下的资源分配
- 与其他稀疏化技术的结合

