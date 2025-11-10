---
title: MoE环游记：3、换个思路来分配
slug: moe环游记3换个思路来分配
date: 2025-03-05
tags: 最优, 损失函数, 梯度, moe, 生成模型
status: pending
---

# MoE环游记：3、换个思路来分配

**原文链接**: [https://spaces.ac.cn/archives/10757](https://spaces.ac.cn/archives/10757)

**发布日期**: 

---

这篇文章我们继续探讨MoE的负载均衡问题。在上一篇文章[《MoE环游记：2、不患寡而患不均》](/archives/10735)中，我们主要讨论了通过Aux Loss来促进负载均衡的思路。Aux Loss固然简单直观，但它也有一个明显的缺点——权重不好调——调低了无法促进均衡，调高了容易损害LM Loss，所以业界一直有寻找替代方案的尝试。

本文要分享的是名为“Loss-Free”的方案，由DeepSeek在[《Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts》](https://papers.cool/arxiv/2408.15664)提出。和DeepSeek众多耀眼的开源作品相比，这篇论文也许不算起眼，但在笔者看来，它潜在的学术影响力可能远超其他工作，因为所提方法不仅简单有效，而且极具普适性，堪称经典。

## 方法大意 #

面对负载不均衡，Aux Loss的应对思路是通过额外的损失引导Router给出均衡的打分，而Loss-Free的想法则是换个新的分配思路，即不改变Router现有打分结果，而是改变$\mathop{\text{argtop}}_k \boldsymbol{\rho}$这个分配方式。

其实这个方向此前也有过一些努力。比如2021年Facebook提出了[BASE Layer](https://papers.cool/arxiv/2103.16716)，将Expert的分配视为[线性指派问题](https://en.wikipedia.org/wiki/Assignment_problem)，即以负载均衡为约束条件，求在该约束之下Router总打分尽可能高的分配结果，这可以用[匈牙利算法](https://en.wikipedia.org/wiki/Hungarian_algorithm)等来解决。但该方案需要知道全体Token的打分，所以对于自回归式LLM来说，它只适用于训练，推理还是只能用$\mathop{\text{argtop}}_k \boldsymbol{\rho}$，训练推理存在不一致性，并且由于目前求解算法的限制，它只适用于$k=1$的场景。

相比之下，Loss-Free的做法非常简单且有效，它留意到一个事实，即我们总可以引入一个偏置项$\boldsymbol{b}$，使得$\mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}$的分配是均衡的，所以它将MoE的形式改为  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i\end{equation}  
这里的$\boldsymbol{b}$是输入无关的向量，由训练过程确定下来，训练完后它就保持不变，因此推理阶段也可以用，换言之训练和推理具有一致的形式。注意乘以$\boldsymbol{e}_i$的还是$\rho_i$而不是$\rho_i + b_i$，也就是说$\boldsymbol{b}$仅仅参与分配过程而不参与MoE的前向计算，所以我们对$\boldsymbol{b}$或$\boldsymbol{\rho} + \boldsymbol{b}$的正负性都没有特殊要求。

## 手搓梯度 #

怎么训练$\boldsymbol{b}$呢？我们知道，$\boldsymbol{b}$的优化方向自然是促进负载均衡，为此按照上一篇的记号，我们先定义$\boldsymbol{f}=[f_1,f_2,\cdots,f_n]$：  
\begin{equation}f_i = \left\\{\begin{aligned}1/k, \quad i\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}+\boldsymbol{b} \\\  
0, \quad i\not\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}+\boldsymbol{b}\end{aligned}\right.\end{equation}  
以及$\boldsymbol{F}=\mathbb{E}[\boldsymbol{f}]$，这里的$\boldsymbol{F}$自然就是在$\boldsymbol{b}$偏置下Expert当前的负载分布了。借着我们定义均匀分布为$\boldsymbol{Q}=(1/n,1/n,\cdots,1/n)$，那么负载均衡就相当于最小化  
\begin{equation}\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (F_i - 1/n)^2\end{equation}  
这个目标是不可导的，但有了上一篇的经验，我们知道STE（Straight-Through Estimator）可以解决这个问题。STE的关键是找一个可导且跟$\boldsymbol{F}$具有同增减趋势的量作为$\boldsymbol{F}$的光滑近似，这里我们的优化参数只有$\boldsymbol{b}$，而它正好具有我们期望的性质（增大$b_i$，$i$被选中的概率就更高，那么$F_i$就更大），所以答案就呼之欲出了：  
\begin{equation}\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{b} + \text{sg}[\boldsymbol{F}-\boldsymbol{b}] - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (b_i + \text{sg}[F_i - b_i] - 1/n)^2\end{equation}  
它的梯度是  
\begin{equation}\nabla_{\boldsymbol{b}}\mathcal{L}_{\text{aux}} = \frac{1}{2}\nabla_{\boldsymbol{b}}\Vert\boldsymbol{b} + \text{sg}[\boldsymbol{F}-\boldsymbol{b}] - \boldsymbol{Q}\Vert^2 = \boldsymbol{F} - \boldsymbol{Q}\end{equation}  
所以用梯度下降（SGD）来更新$\boldsymbol{b}$就是  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha (\boldsymbol{F} - \boldsymbol{Q})\end{equation}  
这里$\alpha$是$\boldsymbol{b}$的学习率。不过Loss-Free最终选择的更新规则略有不同，它选择的是符号梯度下降（SignSGD）：  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha \mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})\label{eq:aux-loss-free}\end{equation}  
这个结果其实也很好理解，就是如果$F_i$比$1/n$大，那么就调小一点$b_i$，否则就增大一点$b_i$。

## 改良版本 #

除了加$\mathop{\text{sign}}$的符号梯度下降外，笔者发现直接对$\boldsymbol{F} - \boldsymbol{Q}$做RMS Norm（即Normalized SGD），在相同的$\alpha$下往往能达到更好的均衡效果：  
\begin{equation}\boldsymbol{b}\leftarrow \boldsymbol{b} - \alpha\frac{\boldsymbol{F} - \boldsymbol{Q}}{\text{RMS}(\boldsymbol{F} - \boldsymbol{Q})}\end{equation}  
这里的$\text{RMS}$是“Root Mean Square”，定义为  
\begin{equation}\text{RMS}(\boldsymbol{F} - \boldsymbol{Q}) = \sqrt{\frac{1}{n}\sum_{i=1}^n (F_i - Q_i)^2}\end{equation}  
不难看出，加$\mathop{\text{sign}}$后的$\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})$和加RMS Norm后的$\frac{\boldsymbol{F} - \boldsymbol{Q}}{\text{RMS}(\boldsymbol{F} - \boldsymbol{Q})}$，它们的$\text{RMS}$都是1，因此它们俩尺度上是大致相同的，所以我们可以使用相同的$\alpha$。

简单来说，$\mathop{\text{sign}}$的问题在于不论$F_i$与目标$Q_i$的远近都使用同样的更新幅度，这导致原本就已经跟$Q_i$比较接近的$F_i$反而容易偏离原本已经达到的均衡，从而产生震荡；而RMS Norm则保留了$F_i-Q_i$之间的相对大小，更新幅度更加自适应一些，理论上更有助于促进均衡，实测效果也多是它更好。

## 一脉相承 #

原论文在介绍Loss-Free时，并没有上述Aux Loss的推导过程，而是直接给出式$\eqref{eq:aux-loss-free}$的更新规则，给人的感觉是给$\boldsymbol{b}$“手搓”了梯度$\mathop{\text{sign}}(\boldsymbol{F} - \boldsymbol{Q})$，这也是它Loss-Free这个名字的来源。

然而，从本文给出的推导可以看出，更新规则$\eqref{eq:aux-loss-free}$也完全可以从Aux Loss视角得到，两者是一脉相承的。看起来Loss-Free最直接的好处是不用调Aux Loss权重了，但它实际上也有个学习率参数$\alpha$要调，尽管原论文已经帮我们搜好$\alpha=0.001$这个默认值，但不可否认这个超参数是存在的。

在笔者看来，Loss-Free的本质创新并不是没有Aux Loss，而是隔离了Aux Loss和LM Loss的优化参数，从而达到了负载均衡和模型能力两不误的效果。其中最关键一步，是留意到“一个偏置项足以达到负载均衡”这一事实，然后就让Aux Loss只优化新引入的偏置$\boldsymbol{b}$，而LM Loss则优化剩余参数，让Aux Loss对LM Loss的负面作用降到最低。

相比之下，常规的Aux Loss方案需要全体参数来促进负载均衡，而LM Loss优化的也是全体参数，两者的优化方向可能并不完全兼容，因此想找到一个最优的平衡点相对来说就更为困难。所以，Loss-Free基于“一个偏置项足以达到负载均衡”将两个Loss的优化参数隔离开来，是负载均衡问题的一个绝妙的解决办法。

## 相关细节 #

尽管Loss-Free已经足够简单明了，但是在使用的时候还要稍微注意一些细节。

首先，对于每个Batch的数据，我们应当先根据LM Loss来更新模型参数，然后再根据式$\eqref{eq:aux-loss-free}$来更新$\boldsymbol{b}$。这是因为$\boldsymbol{b}$的更新依赖于全体Token的统计信息$\boldsymbol{F}$，先更新$\boldsymbol{b}$再更新模型其余参数的话，原则上会有泄漏未来信息的风险。虽然直观看来就一个向量$\boldsymbol{b}$泄漏不了多少信息，但这个风险终归是存在的，因此要尽量去规避它。

其次，刚才我们说原论文已经调好$\alpha=0.001$，但这个结果可能跟原论文用Sigmoid作为Router $\boldsymbol{\rho}$激活函数的选择是绑定的。原因也不难想，经过Sigmoid后，每个$\rho_i$相对比较独立，并且都在$(0,1)$内，$\alpha=0.001$相当于说每一步的更新幅度约为千分之一，如果换Softmax、ReLU或者其他激活函数，那么就可能需要重调$\alpha$了。

针对这个问题，笔者建议的做法是解耦Gate和Bias所用的激活函数，即  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}^{(\sigma)} + \boldsymbol{b}} \rho_i^{(h)} \boldsymbol{e}_i\end{equation}  
其中$\boldsymbol{\rho}^{(\sigma)} = \sigma(\boldsymbol{x}\boldsymbol{W}^{(R)}), \boldsymbol{\rho}^{(h)} = h(\boldsymbol{x}\boldsymbol{W}^{(R)})$，$\sigma(\cdot)$是Sigmoid函数，$h(\cdot)$是任意单调且值域非负的函数，说白了就是加上$\boldsymbol{b}$的是Sigmoid激活的打分，这样我们就可以复用$\alpha=0.001$，至于乘上Expert的Gate，我们可以用其他激活函数，只要它的单调性跟Sigmoid一致就行。

此外，由于更新规则$\eqref{eq:aux-loss-free}$加了$\text{sign}$函数，因此有可能训出绝对值大于1的$b_i$，整体绝对值还可能越来越大，这些都是正常的，对模型效果不会有影响。实际上$\boldsymbol{b}$有一个冗余的自由度，因为全体$b_i$都加上同一个常数后，$\mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}$的结果不变。这个额外的自由度我们可以用来做其他好玩的事情（且听下回分解）。

## 延伸思考 #

除了MoE的负载均衡之外，Loss-Free的思想还可以应用到很多类似问题，比如VQ-VQE的编码表坍缩（Codebook Collapse），就可以用同样思路解决，而且相比之前介绍的“[旋转技巧](/archives/10489)”、“[线性变换技巧](/archives/10519)”显得更自然和普适。事实上，本文开篇的评价“Loss-Free潜在的学术影响力可能远超其他工作”，正是基于Loss-Free的普适性考虑的。

抛开具体的应用背景，从数学上来看，Loss-Free的贡献可以理解为给出了用梯度下降来求解指派问题的方法。一个经典的线性指派问题可以表示为：  
\begin{equation}\min_f \sum_{i=1}^n c_{i, f(i)}\end{equation}  
其中$c_{i,j}$是给定的成本函数，$f$是$\\{1,2,\cdots,n\\}$到自身的双射。放到本文的背景下，$c_{i,j}$不就相当于$n$个Token、$n$个Expert的打分，所求$f$不就是一个负载均衡的分配方案？求解此类问题的一般想法是在满足约束条件的空间里搜索尽可能优的解，而Loss-Free则反过来，先构建一个最优但不一定满足约束条件的解：  
\begin{equation}f(i) = \mathop{\text{argmin}}_j c_{i,j}\end{equation}  
这个解在分数上肯定是最优的，但不一定满足双射的条件，这里不满足双射就等价于负载不均衡。于是我们引入偏置  
\begin{equation}f(i) = \mathop{\text{argmin}}_j c_{i,j} + b_j\end{equation}  
$b_j$初始化为零，然后根据式$\eqref{eq:aux-loss-free}$来更新，更新规则说白了就是哪个$j$出现出现次数多，那减少相应的$b_j$，反之增加，直到出现双射为止。

## 文章小结 #

本文介绍了MoE负载均衡问题的Loss-Free方法，它由DeepSeek提出，其核心在于通过引入一个简单的偏置项来实现负载均衡。本文进一步思考了它与Aux Loss的联系，以及它在类似数学问题上的应用潜力。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10757>_

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

苏剑林. (Mar. 05, 2025). 《MoE环游记：3、换个思路来分配 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10757>

@online{kexuefm-10757,  
title={MoE环游记：3、换个思路来分配},  
author={苏剑林},  
year={2025},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/10757}},  
} 


---

## 公式推导与注释

本节提供MoE负载均衡问题的深入数学分析，从最优化理论、深度学习和实际应用等多个角度解释Loss-Free方法的理论基础。

### 1. MoE的数学模型

#### 1.1 基本形式

标准的MoE（Mixture of Experts）模型可以形式化表示为：

$$
\boldsymbol{y} = \sum_{i=1}^{n} g_i(\boldsymbol{x}) \boldsymbol{e}_i(\boldsymbol{x})
$$

其中：
- $\boldsymbol{x} \in \mathbb{R}^d$ 是输入token的表示
- $n$ 是专家（Expert）的总数
- $\boldsymbol{e}_i: \mathbb{R}^d \to \mathbb{R}^{d'}$ 是第 $i$ 个专家网络
- $g_i: \mathbb{R}^d \to \mathbb{R}$ 是门控（Gating）函数，满足 $\sum_{i=1}^n g_i(\boldsymbol{x}) = 1$

#### 1.2 Top-K稀疏化MoE

为了计算效率，实际应用中通常采用Top-K稀疏化的MoE：

$$
\boldsymbol{y} = \sum_{i \in \mathcal{T}_k(\boldsymbol{x})} \tilde{g}_i(\boldsymbol{x}) \boldsymbol{e}_i(\boldsymbol{x})
$$

其中 $\mathcal{T}_k(\boldsymbol{x}) = \mathop{\text{argtop}}_k \{\rho_1(\boldsymbol{x}), \ldots, \rho_n(\boldsymbol{x})\}$ 是选择得分最高的 $k$ 个专家的索引集合，而

$$
\tilde{g}_i(\boldsymbol{x}) = \frac{\rho_i(\boldsymbol{x})}{\sum_{j \in \mathcal{T}_k(\boldsymbol{x})} \rho_j(\boldsymbol{x})}
$$

是重归一化的门控权重。这里 $\boldsymbol{\rho}(\boldsymbol{x}) = [\rho_1(\boldsymbol{x}), \ldots, \rho_n(\boldsymbol{x})]^T$ 是路由器（Router）的输出得分。

**路由器的参数化**：

$$
\boldsymbol{\rho}(\boldsymbol{x}) = h(\boldsymbol{x} \boldsymbol{W}^{(R)})
$$

其中 $\boldsymbol{W}^{(R)} \in \mathbb{R}^{d \times n}$ 是路由器的权重矩阵，$h(\cdot)$ 是激活函数（如Softmax、Sigmoid、ReLU等）。

### 2. 专家分配问题的最优化表述

#### 2.1 负载均衡作为约束优化问题

考虑一个批次（batch）中有 $m$ 个token，记为 $\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_m\}$。专家分配问题可以表述为以下优化问题：

$$
\begin{align}
\max_{\{S_i\}_{i=1}^{m}} \quad & \sum_{t=1}^{m} \sum_{i \in S_t} \rho_i(\boldsymbol{x}_t) \\
\text{s.t.} \quad & |S_t| = k, \quad \forall t = 1, \ldots, m \\
& \left|\bigcup_{t: i \in S_t} \{t\}\right| \leq C, \quad \forall i = 1, \ldots, n
\end{align}
$$

其中：
- $S_t \subseteq \{1, \ldots, n\}$ 是分配给第 $t$ 个token的专家集合
- 第一个约束保证每个token恰好选择 $k$ 个专家
- 第二个约束是容量约束（Capacity Constraint），限制每个专家最多处理 $C$ 个token
- $C = \lceil \frac{mk}{n} \cdot \gamma \rceil$ 是容量上限，$\gamma \geq 1$ 是容量因子（Capacity Factor）

**当 $\gamma = 1$ 时的完美均衡**：容量恰好等于 $C = \frac{mk}{n}$，这意味着所有专家的负载完全均衡。

#### 2.2 线性指派问题（Linear Assignment Problem）

当 $k=1$ 时，上述问题退化为经典的线性指派问题：

$$
\begin{align}
\min_{A \in \mathcal{P}} \quad & \sum_{t=1}^{m} \sum_{i=1}^{n} c_{ti} a_{ti} \\
\text{s.t.} \quad & \sum_{i=1}^{n} a_{ti} = 1, \quad \forall t \\
& \sum_{t=1}^{m} a_{ti} \leq C, \quad \forall i \\
& a_{ti} \in \{0, 1\}
\end{align}
$$

其中 $c_{ti} = -\rho_i(\boldsymbol{x}_t)$ 是成本矩阵，$a_{ti}$ 是指示变量（token $t$ 是否分配给专家 $i$）。

这个问题可以用**匈牙利算法**（Hungarian Algorithm）或**Kuhn-Munkres算法**在 $O(m^3)$ 时间内求解，但存在以下限制：
1. 仅适用于 $k=1$ 的情况
2. 需要全局信息（整个batch的所有token），不适合自回归推理
3. 计算复杂度较高

### 3. 负载均衡约束的数学表征

#### 3.1 负载分布的定义

对于单个token $\boldsymbol{x}$，定义其专家选择的one-hot向量：

$$
f_i = \begin{cases}
\frac{1}{k}, & i \in \mathop{\text{argtop}}_k \boldsymbol{\rho}(\boldsymbol{x}) \\
0, & \text{otherwise}
\end{cases}
$$

这里除以 $k$ 是为了归一化，使得 $\sum_{i=1}^{n} f_i = 1$。

**批次级别的负载分布**：对于包含 $m$ 个token的批次，专家 $i$ 的平均负载为：

$$
F_i = \frac{1}{m} \sum_{t=1}^{m} f_i^{(t)} = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[f_i(\boldsymbol{x})]
$$

其中 $\mathcal{D}$ 是训练数据分布。

**理想均匀分布**：

$$
Q_i = \frac{1}{n}, \quad \forall i = 1, \ldots, n
$$

满足 $\sum_{i=1}^{n} Q_i = 1$。

#### 3.2 负载不均衡的度量

常用的度量包括：

**（1）L2距离（欧氏距离）**：

$$
\mathcal{D}_2(\boldsymbol{F}, \boldsymbol{Q}) = \|\boldsymbol{F} - \boldsymbol{Q}\|_2 = \sqrt{\sum_{i=1}^{n} (F_i - Q_i)^2}
$$

**（2）KL散度（Kullback-Leibler Divergence）**：

$$
\mathcal{D}_{\text{KL}}(\boldsymbol{F} \| \boldsymbol{Q}) = \sum_{i=1}^{n} F_i \log \frac{F_i}{Q_i}
$$

**（3）变异系数（Coefficient of Variation）**：

$$
\text{CV}(\boldsymbol{F}) = \frac{\sqrt{\text{Var}(\boldsymbol{F})}}{\mathbb{E}[\boldsymbol{F}]} = \frac{\sqrt{\frac{1}{n}\sum_{i=1}^{n} (F_i - \frac{1}{n})^2}}{\frac{1}{n}} = \sqrt{n \sum_{i=1}^{n} (F_i - \frac{1}{n})^2}
$$

本文采用L2距离的平方作为优化目标，这在数学上更加方便处理。

### 4. 路由策略的数学分析

#### 4.1 常见激活函数的性质

**（1）Softmax激活**：

$$
\rho_i^{(\text{softmax})}(\boldsymbol{x}) = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)}
$$

其中 $z_i = \boldsymbol{x}^T \boldsymbol{w}_i^{(R)}$ 是第 $i$ 个专家的logit。

**性质**：
- 概率约束：$\sum_{i=1}^{n} \rho_i = 1$
- 值域：$\rho_i \in (0, 1)$
- 梯度：$\frac{\partial \rho_i}{\partial z_j} = \rho_i (\delta_{ij} - \rho_j)$，其中 $\delta_{ij}$ 是Kronecker delta

**（2）Sigmoid激活**：

$$
\rho_i^{(\text{sigmoid})}(\boldsymbol{x}) = \sigma(z_i) = \frac{1}{1 + \exp(-z_i)}
$$

**性质**：
- 独立性：每个 $\rho_i$ 独立计算
- 值域：$\rho_i \in (0, 1)$
- 梯度：$\frac{\partial \rho_i}{\partial z_i} = \rho_i(1 - \rho_i)$
- **不满足概率约束**：$\sum_{i=1}^{n} \rho_i \neq 1$（一般情况）

**（3）ReLU激活**：

$$
\rho_i^{(\text{ReLU})}(\boldsymbol{x}) = \max(0, z_i)
$$

**性质**：
- 值域：$\rho_i \in [0, +\infty)$
- 稀疏性：可能产生很多零值
- 梯度：$\frac{\partial \rho_i}{\partial z_i} = \mathbb{1}_{z_i > 0}$（几乎处处可导）

#### 4.2 Top-K选择的非平滑性

Top-K操作定义为：

$$
\mathop{\text{argtop}}_k \boldsymbol{\rho} = \{i_1, \ldots, i_k\}
$$

其中 $\rho_{i_1} \geq \rho_{i_2} \geq \cdots \geq \rho_{i_k} \geq \rho_j$ 对所有 $j \notin \{i_1, \ldots, i_k\}$。

**问题**：这是一个**离散的、不可微的**操作：
- 当 $\rho_i$ 发生微小变化时，选择集合 $\mathcal{T}_k$ 可能发生突变
- 标准的反向传播无法直接应用

**梯度近似方法**：

（1）**直通估计器（Straight-Through Estimator, STE）**：

$$
\frac{\partial \mathcal{L}}{\partial \rho_i} \approx \begin{cases}
\frac{\partial \mathcal{L}}{\partial \tilde{\rho}_i}, & i \in \mathcal{T}_k \\
0, & \text{otherwise}
\end{cases}
$$

其中 $\tilde{\rho}_i$ 是归一化后的门控权重。

（2）**Gumbel-Softmax松弛**：用连续分布近似离散采样。

（3）**本文方法（Loss-Free）**：引入偏置项 $\boldsymbol{b}$，将Top-K的输入从 $\boldsymbol{\rho}$ 改为 $\boldsymbol{\rho} + \boldsymbol{b}$。

### 5. Loss-Free方法的最优化理论

#### 5.1 偏置项的引入

**核心观察**：对于任意给定的路由得分 $\boldsymbol{\rho}$，总存在一个偏置向量 $\boldsymbol{b} \in \mathbb{R}^n$，使得基于 $\boldsymbol{\rho} + \boldsymbol{b}$ 的Top-K选择达到负载均衡。

**数学表述**：存在 $\boldsymbol{b}^* \in \mathbb{R}^n$，使得

$$
\mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{b}^*)\right] = \boldsymbol{Q}
$$

其中

$$
f_i(\boldsymbol{x}; \boldsymbol{b}) = \begin{cases}
\frac{1}{k}, & i \in \mathop{\text{argtop}}_k (\boldsymbol{\rho}(\boldsymbol{x}) + \boldsymbol{b}) \\
0, & \text{otherwise}
\end{cases}
$$

**存在性证明（构造性）**：

考虑映射 $\Phi: \mathbb{R}^n \to \mathbb{R}^n$：

$$
\Phi(\boldsymbol{b}) = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}\left[\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{b})\right] - \boldsymbol{Q}
$$

目标是找到 $\boldsymbol{b}^*$ 使得 $\Phi(\boldsymbol{b}^*) = \boldsymbol{0}$。

通过以下观察可以证明解的存在性：
1. 当 $b_i \to +\infty$ 时，专家 $i$ 总是被选中（对所有token），因此 $F_i \to 1 > Q_i$
2. 当 $b_i \to -\infty$ 时，专家 $i$ 从不被选中，因此 $F_i \to 0 < Q_i$
3. $\Phi_i(\boldsymbol{b})$ 关于 $b_i$ 单调递增（几乎处处）
4. 由中间值定理和不动点定理，存在 $\boldsymbol{b}^*$ 使得 $\Phi(\boldsymbol{b}^*) = \boldsymbol{0}$

#### 5.2 优化目标的构造

定义损失函数：

$$
\mathcal{L}_{\text{balance}}(\boldsymbol{b}) = \frac{1}{2} \|\boldsymbol{F}(\boldsymbol{b}) - \boldsymbol{Q}\|_2^2 = \frac{1}{2} \sum_{i=1}^{n} (F_i(\boldsymbol{b}) - Q_i)^2
$$

其中 $\boldsymbol{F}(\boldsymbol{b}) = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[\boldsymbol{f}(\boldsymbol{x}; \boldsymbol{b})]$。

**目标**：最小化 $\mathcal{L}_{\text{balance}}(\boldsymbol{b})$ 以达到 $\boldsymbol{F}(\boldsymbol{b}) \approx \boldsymbol{Q}$。

**困难**：$\boldsymbol{F}(\boldsymbol{b})$ 关于 $\boldsymbol{b}$ 不可微（因为涉及 $\mathop{\text{argtop}}_k$ 操作）。

#### 5.3 直通估计器（STE）的应用

**观察**：尽管 $F_i(\boldsymbol{b})$ 不可微，但我们可以利用以下性质：
- 增大 $b_i$ 会增加专家 $i$ 被选中的概率，从而增大 $F_i$
- 减小 $b_i$ 会减少专家 $i$ 被选中的概率，从而减小 $F_i$

**STE构造**：将损失函数重写为

$$
\mathcal{L}_{\text{balance}}(\boldsymbol{b}) = \frac{1}{2} \|\boldsymbol{b} + \text{sg}[\boldsymbol{F}(\boldsymbol{b}) - \boldsymbol{b}] - \boldsymbol{Q}\|_2^2
$$

其中 $\text{sg}[\cdot]$ 表示stop-gradient操作（在反向传播时视为常数）。

**梯度计算**：

$$
\begin{align}
\nabla_{\boldsymbol{b}} \mathcal{L}_{\text{balance}} &= \nabla_{\boldsymbol{b}} \frac{1}{2} \|\boldsymbol{b} + \text{sg}[\boldsymbol{F} - \boldsymbol{b}] - \boldsymbol{Q}\|_2^2 \\
&= \nabla_{\boldsymbol{b}} \frac{1}{2} \|\boldsymbol{F} - \boldsymbol{Q}\|_2^2 \quad (\text{视} \boldsymbol{F} \text{为常数}) \\
&= \boldsymbol{F} - \boldsymbol{Q}
\end{align}
$$

这个梯度形式简单且直观：
- 如果 $F_i > Q_i$（专家 $i$ 过载），则 $\nabla_{b_i} \mathcal{L} > 0$，梯度下降会减小 $b_i$
- 如果 $F_i < Q_i$（专家 $i$ 欠载），则 $\nabla_{b_i} \mathcal{L} < 0$，梯度下降会增大 $b_i$

#### 5.4 梯度下降更新规则

**标准SGD**：

$$
\boldsymbol{b}^{(t+1)} = \boldsymbol{b}^{(t)} - \alpha (\boldsymbol{F}^{(t)} - \boldsymbol{Q})
$$

其中 $\alpha > 0$ 是学习率，$\boldsymbol{F}^{(t)}$ 是第 $t$ 步的负载统计。

**符号梯度下降（SignSGD）**（原论文采用）：

$$
\boldsymbol{b}^{(t+1)} = \boldsymbol{b}^{(t)} - \alpha \, \text{sign}(\boldsymbol{F}^{(t)} - \boldsymbol{Q})
$$

其中

$$
\text{sign}(x) = \begin{cases}
+1, & x > 0 \\
0, & x = 0 \\
-1, & x < 0
\end{cases}
$$

**优点**：
- 更新幅度统一，不受 $|F_i - Q_i|$ 绝对值的影响
- 类似于Adam等自适应优化器的效果
- 鲁棒性更强，不容易因为某个专家负载极度不均而产生过大的更新

**缺点**：
- 忽略了不均衡程度的信息
- 可能在接近最优点时产生震荡

#### 5.5 RMS归一化梯度下降（改进版）

本文作者提出的改进方案：

$$
\boldsymbol{b}^{(t+1)} = \boldsymbol{b}^{(t)} - \alpha \frac{\boldsymbol{F}^{(t)} - \boldsymbol{Q}}{\text{RMS}(\boldsymbol{F}^{(t)} - \boldsymbol{Q})}
$$

其中RMS（Root Mean Square）定义为：

$$
\text{RMS}(\boldsymbol{v}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} v_i^2}
$$

**性质分析**：

（1）**尺度归一化**：归一化后的梯度的RMS值恒为1：

$$
\text{RMS}\left(\frac{\boldsymbol{v}}{\text{RMS}(\boldsymbol{v})}\right) = 1
$$

这与 $\text{sign}(\boldsymbol{v})$ 的 $\ell_2$ 范数为 $\sqrt{n}$ 不同，但两者都实现了尺度归一化。

（2）**保留相对信息**：与SignSGD不同，RMS归一化保留了各分量的相对大小关系：

$$
\frac{F_i - Q_i}{\text{RMS}(\boldsymbol{F} - \boldsymbol{Q})} \propto (F_i - Q_i)
$$

这意味着不均衡更严重的专家会得到更大的调整。

（3）**减少震荡**：在接近收敛时，$\|\boldsymbol{F} - \boldsymbol{Q}\|$ 较小，归一化梯度的各分量也相应变小，有助于稳定收敛。

**与标准SGD的关系**：

RMS归一化梯度下降可以看作是自适应学习率的SGD：

$$
\boldsymbol{b}^{(t+1)} = \boldsymbol{b}^{(t)} - \frac{\alpha}{\text{RMS}(\boldsymbol{F}^{(t)} - \boldsymbol{Q})} (\boldsymbol{F}^{(t)} - \boldsymbol{Q})
$$

相当于有效学习率为 $\alpha_{\text{eff}} = \frac{\alpha}{\text{RMS}(\boldsymbol{F} - \boldsymbol{Q})}$。

### 6. 最优传输理论的联系

#### 6.1 最优传输问题（Optimal Transport）

负载均衡问题可以从最优传输（OT）的角度来理解。给定两个概率分布：
- 源分布：$\boldsymbol{\mu} = \boldsymbol{F}$（当前负载分布）
- 目标分布：$\boldsymbol{\nu} = \boldsymbol{Q}$（均匀分布）

最优传输问题寻找一个传输计划 $\boldsymbol{\pi} \in \mathbb{R}^{n \times n}$，使得

$$
\begin{align}
\min_{\boldsymbol{\pi}} \quad & \sum_{i,j=1}^{n} C_{ij} \pi_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{n} \pi_{ij} = \mu_i, \quad \forall i \\
& \sum_{i=1}^{n} \pi_{ij} = \nu_j, \quad \forall j \\
& \pi_{ij} \geq 0
\end{align}
$$

其中 $C_{ij}$ 是从位置 $i$ 传输到位置 $j$ 的成本。

**特殊情况**：当 $C_{ij} = \|i - j\|^2$ 时，这是经典的**Wasserstein-2距离**：

$$
W_2(\boldsymbol{\mu}, \boldsymbol{\nu}) = \left(\min_{\boldsymbol{\pi} \in \Pi(\boldsymbol{\mu}, \boldsymbol{\nu})} \sum_{i,j} C_{ij} \pi_{ij}\right)^{1/2}
$$

#### 6.2 熵正则化最优传输

为了提高计算效率，引入**熵正则化**：

$$
\min_{\boldsymbol{\pi}} \quad \sum_{i,j=1}^{n} C_{ij} \pi_{ij} + \epsilon H(\boldsymbol{\pi})
$$

其中熵项定义为：

$$
H(\boldsymbol{\pi}) = -\sum_{i,j=1}^{n} \pi_{ij} \log \pi_{ij}
$$

参数 $\epsilon > 0$ 控制正则化强度。

**优点**：
- 最优解具有 $\pi_{ij}^* = u_i K_{ij} v_j$ 的形式，其中 $K_{ij} = \exp(-C_{ij}/\epsilon)$
- 可以用**Sinkhorn算法**高效求解

#### 6.3 Sinkhorn算法

Sinkhorn算法是求解熵正则化OT问题的经典迭代算法。

**初始化**：

$$
\boldsymbol{u}^{(0)} = \mathbf{1}_n, \quad \boldsymbol{v}^{(0)} = \mathbf{1}_n
$$

**迭代更新**（第 $t$ 次迭代）：

$$
\begin{align}
u_i^{(t+1)} &= \frac{\mu_i}{\sum_{j=1}^{n} K_{ij} v_j^{(t)}} = \frac{\mu_i}{(\boldsymbol{K} \boldsymbol{v}^{(t)})_i} \\
v_j^{(t+1)} &= \frac{\nu_j}{\sum_{i=1}^{n} K_{ij} u_i^{(t+1)}} = \frac{\nu_j}{(\boldsymbol{K}^T \boldsymbol{u}^{(t+1)})_j}
\end{align}
$$

**收敛性**：Sinkhorn算法线性收敛到最优解，收敛速度与 $\epsilon$ 有关。

**与Loss-Free的联系**：Loss-Free的偏置项 $\boldsymbol{b}$ 可以看作是一种隐式的最优传输调整：
- $\boldsymbol{b}$ 的作用是调整选择概率，使得负载分布从 $\boldsymbol{F}$ 移动到 $\boldsymbol{Q}$
- 这类似于OT中的dual variable（对偶变量）

**对偶形式**：最优传输的对偶问题为

$$
\max_{\boldsymbol{\phi}, \boldsymbol{\psi}} \quad \sum_{i=1}^{n} \mu_i \phi_i + \sum_{j=1}^{n} \nu_j \psi_j
$$

满足 $\phi_i + \psi_j \leq C_{ij}$。偏置项 $\boldsymbol{b}$ 可以理解为对偶变量的离散化近似。

### 7. 梯度计算与反向传播

#### 7.1 完整的MoE前向传播

考虑带偏置的MoE前向过程：

$$
\begin{align}
\boldsymbol{z} &= \boldsymbol{x} \boldsymbol{W}^{(R)} \in \mathbb{R}^n \\
\boldsymbol{\rho} &= h(\boldsymbol{z}) \in \mathbb{R}^n \\
\boldsymbol{\rho}' &= \boldsymbol{\rho} + \boldsymbol{b} \\
\mathcal{T}_k &= \mathop{\text{argtop}}_k \boldsymbol{\rho}' \\
\tilde{\boldsymbol{\rho}} &= \text{normalize}(\boldsymbol{\rho}|_{\mathcal{T}_k}) \\
\boldsymbol{y} &= \sum_{i \in \mathcal{T}_k} \tilde{\rho}_i \boldsymbol{e}_i(\boldsymbol{x})
\end{align}
$$

#### 7.2 参数的梯度

**（1）专家网络参数的梯度**：

对于 $i \in \mathcal{T}_k$：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_i^{(E)}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{e}_i} \cdot \frac{\partial \boldsymbol{e}_i}{\partial \boldsymbol{\theta}_i^{(E)}} = \tilde{\rho}_i \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{e}_i}{\partial \boldsymbol{\theta}_i^{(E)}}
$$

对于 $i \notin \mathcal{T}_k$：梯度为零（因为该专家未参与计算）。

**（2）路由器参数的梯度**：

使用STE近似：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}^{(R)}} \approx \sum_{i \in \mathcal{T}_k} \frac{\partial \mathcal{L}}{\partial \tilde{\rho}_i} \cdot \frac{\partial \tilde{\rho}_i}{\partial \rho_i} \cdot \frac{\partial \rho_i}{\partial \boldsymbol{z}} \cdot \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}^{(R)}}
$$

其中归一化的梯度为：

$$
\frac{\partial \tilde{\rho}_i}{\partial \rho_j} = \begin{cases}
\frac{1}{Z} - \frac{\rho_i}{Z^2}, & i = j \in \mathcal{T}_k \\
-\frac{\rho_i}{Z^2}, & i \neq j, \, i,j \in \mathcal{T}_k \\
0, & \text{otherwise}
\end{cases}
$$

其中 $Z = \sum_{j \in \mathcal{T}_k} \rho_j$。

**（3）偏置项的梯度**：

关键是 $\boldsymbol{b}$ **不参与前向计算的权重**，只参与Top-K选择。因此：

$$
\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{b}} = \boldsymbol{0}
$$

换言之，$\boldsymbol{b}$ 对主损失函数 $\mathcal{L}_{\text{LM}}$ 的梯度为零！

这正是Loss-Free的核心优势：$\boldsymbol{b}$ 的优化完全独立于主任务，不会干扰语言模型的训练。

#### 7.3 偏置项的独立更新

$\boldsymbol{b}$ 的更新不通过 $\mathcal{L}_{\text{LM}}$ 的梯度，而是通过负载均衡目标：

$$
\boldsymbol{b}^{(t+1)} = \boldsymbol{b}^{(t)} - \alpha \nabla_{\boldsymbol{b}} \mathcal{L}_{\text{balance}} = \boldsymbol{b}^{(t)} - \alpha (\boldsymbol{F}^{(t)} - \boldsymbol{Q})
$$

**更新顺序**：
1. 前向传播：计算 $\mathcal{L}_{\text{LM}}$ 和统计 $\boldsymbol{F}$
2. 反向传播：更新 $\boldsymbol{W}^{(R)}$ 和 $\{\boldsymbol{\theta}_i^{(E)}\}$
3. 独立更新 $\boldsymbol{b}$

这个顺序保证了：
- $\boldsymbol{b}$ 的更新基于当前batch的真实负载统计
- 避免了"泄漏未来信息"的风险

### 8. 容量因子（Capacity Factor）的影响

#### 8.1 容量因子的定义

容量因子 $\gamma \geq 1$ 决定了每个专家的最大容量：

$$
C = \left\lceil \frac{mk}{n} \cdot \gamma \right\rceil
$$

其中：
- $m$：batch中的token数
- $k$：每个token选择的专家数
- $n$：专家总数
- $\frac{mk}{n}$：完美均衡时每个专家的平均负载

#### 8.2 不同 $\gamma$ 值的影响

**（1）$\gamma = 1$（理想情况）**：

- 容量恰好等于平均负载
- **无溢出空间**：要求严格的负载均衡
- 优点：计算资源利用最优
- 缺点：可能导致token overflow（某些token无法分配到k个专家）

**（2）$\gamma > 1$（实际常用）**：

- 提供了 $(\gamma - 1) \times 100\%$ 的冗余容量
- 允许一定程度的负载不均衡
- 常用值：$\gamma = 1.25$ 或 $\gamma = 1.5$

**token溢出概率分析**：

假设负载分布的标准差为 $\sigma_F$，则专家 $i$ 的实际负载 $L_i$ 可以近似为：

$$
L_i \sim \mathcal{N}\left(\frac{mk}{n}, \sigma_F^2 m\right)
$$

溢出概率：

$$
P(\text{overflow}) = P\left(L_i > C\right) = P\left(L_i > \frac{mk}{n} \gamma\right)
$$

标准化后：

$$
P(\text{overflow}) \approx 1 - \Phi\left(\frac{(\gamma - 1) mk/n}{\sigma_F \sqrt{m}}\right)
$$

其中 $\Phi(\cdot)$ 是标准正态分布的CDF。

**结论**：
- $\gamma$ 越大，溢出概率越低
- $\sigma_F$ 越小（负载越均衡），溢出概率越低
- Loss-Free通过减小 $\sigma_F$ 来降低所需的 $\gamma$

#### 8.3 容量约束下的修正

当存在容量约束时，分配策略需要修改为：

$$
\mathcal{T}_k(\boldsymbol{x}_t; \{\ell_i\}) = \begin{cases}
\mathop{\text{argtop}}_k (\boldsymbol{\rho}_t + \boldsymbol{b}) & \text{if no overflow} \\
\text{greedy allocation} & \text{otherwise}
\end{cases}
$$

其中 $\ell_i$ 是专家 $i$ 的当前负载计数。

**贪心分配算法**：按 $\rho_{ti} + b_i$ 降序遍历，跳过已满的专家：

```
for i in argsort(ρ_t + b, descending=True):
    if ℓ_i < C and |T_k| < k:
        T_k.add(i)
        ℓ_i += 1
```

### 9. 损失函数设计（辅助损失对比）

#### 9.1 传统辅助损失（Aux Loss）

**形式1：均衡损失**（Switch Transformer）：

$$
\mathcal{L}_{\text{aux}}^{(1)} = \alpha \sum_{i=1}^{n} F_i P_i
$$

其中 $P_i = \frac{1}{m} \sum_{t=1}^{m} \rho_{ti}$ 是专家 $i$ 的平均得分。

**优化目标**：最小化负载 $F_i$ 与得分 $P_i$ 的乘积，促使高得分的专家被更多选择。

**形式2：变异系数损失**：

$$
\mathcal{L}_{\text{aux}}^{(2)} = \alpha \cdot \text{Var}(\boldsymbol{F}) = \alpha \sum_{i=1}^{n} (F_i - \frac{1}{n})^2
$$

**总损失**：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{aux}}
$$

**问题**：
- 权重 $\alpha$ 难以调节
- $\mathcal{L}_{\text{aux}}$ 的梯度会影响所有参数（包括路由器和专家网络）
- 可能损害主任务性能

#### 9.2 Loss-Free的优势

**对比表**：

| 特性 | 传统Aux Loss | Loss-Free |
|------|-------------|-----------|
| 优化参数 | 全体参数 | 仅偏置 $\boldsymbol{b}$ |
| 对主任务影响 | 直接影响 $\mathcal{L}_{\text{LM}}$ | 不影响前向权重 |
| 超参数 | 权重 $\alpha$ | 学习率 $\alpha$ |
| 训练/推理一致性 | 一致 | 一致 |
| 实现复杂度 | 中等 | 低 |

**数学分析**：

（1）**参数空间分离**：

传统方法：$\theta = \{\boldsymbol{W}^{(R)}, \{\boldsymbol{\theta}_i^{(E)}\}\}$ 同时优化两个目标

$$
\theta^* = \arg\min_{\theta} \left[\mathcal{L}_{\text{LM}}(\theta) + \alpha \mathcal{L}_{\text{aux}}(\theta)\right]
$$

Loss-Free：参数分为两组
- 主参数：$\theta = \{\boldsymbol{W}^{(R)}, \{\boldsymbol{\theta}_i^{(E)}\}\}$ 优化 $\mathcal{L}_{\text{LM}}$
- 辅助参数：$\boldsymbol{b}$ 优化 $\mathcal{L}_{\text{balance}}$

$$
\begin{align}
\theta^* &= \arg\min_{\theta} \mathcal{L}_{\text{LM}}(\theta; \boldsymbol{b}) \\
\boldsymbol{b}^* &= \arg\min_{\boldsymbol{b}} \mathcal{L}_{\text{balance}}(\boldsymbol{b})
\end{align}
$$

（2）**优化景观（Optimization Landscape）**：

传统方法的梯度：

$$
\nabla_{\theta} \mathcal{L}_{\text{total}} = \nabla_{\theta} \mathcal{L}_{\text{LM}} + \alpha \nabla_{\theta} \mathcal{L}_{\text{aux}}
$$

两个梯度可能方向冲突，导致优化困难。

Loss-Free的梯度：

$$
\begin{align}
\nabla_{\theta} \mathcal{L}_{\text{LM}} &\quad \text{(主任务)} \\
\nabla_{\boldsymbol{b}} \mathcal{L}_{\text{balance}} = \boldsymbol{F} - \boldsymbol{Q} &\quad \text{(负载均衡)}
\end{align}
$$

两者正交，互不干扰。

### 10. 收敛性分析与理论保证

#### 10.1 偏置更新的收敛性

考虑连续时间的梯度流（Gradient Flow）：

$$
\frac{d\boldsymbol{b}(t)}{dt} = -(\boldsymbol{F}(\boldsymbol{b}(t)) - \boldsymbol{Q})
$$

**Lyapunov函数**：定义

$$
V(\boldsymbol{b}) = \frac{1}{2} \|\boldsymbol{F}(\boldsymbol{b}) - \boldsymbol{Q}\|_2^2
$$

其沿梯度流的导数为：

$$
\frac{dV}{dt} = (\boldsymbol{F} - \boldsymbol{Q})^T \frac{d\boldsymbol{F}}{dt} = (\boldsymbol{F} - \boldsymbol{Q})^T \frac{\partial \boldsymbol{F}}{\partial \boldsymbol{b}} \frac{d\boldsymbol{b}}{dt}
$$

假设 $\frac{\partial F_i}{\partial b_i} > 0$（增大 $b_i$ 增加 $F_i$），并且 $\frac{\partial F_i}{\partial b_j} \approx 0$ 对 $i \neq j$，则

$$
\frac{dV}{dt} \approx -\sum_{i=1}^{n} (F_i - Q_i) \frac{\partial F_i}{\partial b_i} (F_i - Q_i) = -\sum_{i=1}^{n} \frac{\partial F_i}{\partial b_i} (F_i - Q_i)^2 < 0
$$

因此 $V(\boldsymbol{b}(t))$ 单调递减，系统收敛到 $\boldsymbol{F} = \boldsymbol{Q}$。

#### 10.2 离散化误差

实际的SignSGD或RMS-SGD是离散化的：

$$
\boldsymbol{b}^{(t+1)} = \boldsymbol{b}^{(t)} - \alpha \boldsymbol{g}^{(t)}
$$

其中 $\boldsymbol{g}^{(t)}$ 是归一化梯度。

**稳定性条件**：要求学习率满足

$$
\alpha < \frac{2}{\lambda_{\max}(\boldsymbol{H})}
$$

其中 $\boldsymbol{H} = \frac{\partial^2 V}{\partial \boldsymbol{b}^2}$ 是Hessian矩阵。

对于本问题，由于 $\frac{\partial F_i}{\partial b_i}$ 通常较小（负载变化缓慢），$\alpha = 0.001$ 远小于稳定性上界，因此收敛有保证。

#### 10.3 随机性的影响

实际训练中，$\boldsymbol{F}$ 是基于mini-batch估计的，存在方差：

$$
\boldsymbol{F}_{\text{batch}} = \boldsymbol{F}_{\text{true}} + \boldsymbol{\epsilon}
$$

其中 $\mathbb{E}[\boldsymbol{\epsilon}] = \boldsymbol{0}$，$\text{Var}(\epsilon_i) = \frac{\sigma_i^2}{m}$。

**影响**：
- 更新存在噪声，导致 $\boldsymbol{b}$ 在最优点附近振荡
- 振荡幅度与batch size $m$ 成反比
- 可以通过增大 $m$ 或使用moving average来减小方差

### 11. 实验结果的理论解释

#### 11.1 负载均衡指标

常用指标：

**（1）负载变异系数**：

$$
\text{CV}(\boldsymbol{F}) = \frac{\sqrt{\text{Var}(\boldsymbol{F})}}{\mathbb{E}[\boldsymbol{F}]} = \sqrt{n \sum_{i=1}^{n} (F_i - \frac{1}{n})^2}
$$

**（2）最大/最小负载比**：

$$
\text{Ratio} = \frac{\max_i F_i}{\min_i F_i}
$$

理想情况：$\text{Ratio} = 1$

**（3）熵**：

$$
H(\boldsymbol{F}) = -\sum_{i=1}^{n} F_i \log F_i
$$

最大值：$H(\boldsymbol{Q}) = \log n$（均匀分布）

#### 11.2 性能改进的理论原因

**（1）计算效率**：

均衡负载下，总计算时间为：

$$
T_{\text{total}} = \max_{i} T_i \approx \frac{mk}{n} \cdot T_{\text{expert}}
$$

不均衡时：

$$
T_{\text{total}} = \max_{i} L_i \cdot T_{\text{expert}} \gg \frac{mk}{n} \cdot T_{\text{expert}}
$$

**加速比**：

$$
\text{Speedup} = \frac{\max_i L_i^{\text{before}}}{\max_i L_i^{\text{after}}} = \frac{\max_i L_i^{\text{before}}}{mk/n}
$$

**（2）模型质量**：

传统Aux Loss会损害主任务，因为：

$$
\nabla_{\theta} \mathcal{L}_{\text{LM}} \quad \text{与} \quad \nabla_{\theta} \mathcal{L}_{\text{aux}} \quad \text{可能反向}
$$

导致妥协解：

$$
\theta^* \neq \arg\min_{\theta} \mathcal{L}_{\text{LM}}
$$

Loss-Free避免了这个问题：

$$
\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{LM}}(\theta; \boldsymbol{b}^*)
$$

**（3）专家专业化（Expert Specialization）**：

负载均衡有助于每个专家学习不同的模式：

$$
\min_{\{\boldsymbol{\theta}_i^{(E)}\}} \mathbb{E}\left[\sum_{i \in \mathcal{T}_k(\boldsymbol{x})} \tilde{\rho}_i(\boldsymbol{x}) \mathcal{L}(\boldsymbol{e}_i(\boldsymbol{x}), \boldsymbol{y})\right]
$$

均衡负载确保每个专家都有足够的训练样本，避免某些专家"饿死"。

#### 11.3 消融实验分析

**（1）不同激活函数的影响**：

- Sigmoid：$\rho_i \in (0,1)$，独立，适合 $\alpha=0.001$
- Softmax：$\sum_i \rho_i = 1$，相互依赖，可能需要调整 $\alpha$
- ReLU：$\rho_i \in [0, \infty)$，尺度不固定，需要仔细调整

**（2）SignSGD vs RMS-SGD**：

实验表明RMS-SGD通常更好：
- 更平滑的收敛曲线
- 更低的最终CV值
- 对学习率 $\alpha$ 的鲁棒性更强

**理论解释**：RMS归一化保留了梯度的方向信息，而Sign仅保留符号，丢失了幅度信息。

**（3）学习率 $\alpha$ 的影响**：

- $\alpha$ 太小：收敛慢，可能在训练结束前未达到均衡
- $\alpha$ 太大：可能产生振荡或过冲
- 最优 $\alpha$ 依赖于batch size、专家数等

**经验法则**：

$$
\alpha \approx \frac{C_0}{\sqrt{n}}
$$

其中 $C_0 \approx 0.03$ 对Sigmoid激活。

### 12. 扩展应用：VQ-VAE的编码表坍缩

#### 12.1 问题描述

Vector Quantization VAE（VQ-VAE）中的编码表坍缩（Codebook Collapse）：某些编码向量从不被使用，导致编码表的有效容量减小。

**形式化**：设编码表 $\{\boldsymbol{c}_1, \ldots, \boldsymbol{c}_K\}$，编码过程为

$$
\text{encode}(\boldsymbol{z}) = \arg\min_{i} \|\boldsymbol{z} - \boldsymbol{c}_i\|_2
$$

**问题**：某些 $\boldsymbol{c}_i$ 可能永远不被选中。

#### 12.2 Loss-Free解决方案

引入偏置 $\boldsymbol{b} = [b_1, \ldots, b_K]$：

$$
\text{encode}(\boldsymbol{z}; \boldsymbol{b}) = \arg\min_{i} \|\boldsymbol{z} - \boldsymbol{c}_i\|_2 + b_i
$$

**目标**：均衡使用频率 $\boldsymbol{F} = [F_1, \ldots, F_K]$，其中

$$
F_i = \mathbb{E}_{\boldsymbol{z}}[\mathbb{1}_{\text{encode}(\boldsymbol{z}; \boldsymbol{b}) = i}]
$$

**更新规则**：

$$
\boldsymbol{b} \leftarrow \boldsymbol{b} - \alpha (\boldsymbol{F} - \boldsymbol{Q})
$$

其中 $\boldsymbol{Q} = (\frac{1}{K}, \ldots, \frac{1}{K})$。

**优势**：
- 不需要修改VQ-VAE的核心损失
- 训练和推理一致
- 自动适应数据分布

### 总结

Loss-Free方法的数学本质是：
1. **参数空间分离**：通过引入偏置 $\boldsymbol{b}$，将负载均衡与主任务优化解耦
2. **直通估计器**：利用STE处理不可微的Top-K操作
3. **自适应调整**：通过梯度下降自动调整负载分布
4. **最优传输视角**：隐式求解从当前分布到均匀分布的最优传输

这种方法在理论上优雅，实践上有效，具有广泛的应用潜力。

