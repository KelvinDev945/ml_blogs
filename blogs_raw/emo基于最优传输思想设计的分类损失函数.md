---
title: EMO：基于最优传输思想设计的分类损失函数
slug: emo基于最优传输思想设计的分类损失函数
date: 2023-10-13
tags: 概率, 优化, 损失函数, 最优传输, 生成模型
status: completed
---

# EMO：基于最优传输思想设计的分类损失函数

**原文链接**: [https://spaces.ac.cn/archives/9797](https://spaces.ac.cn/archives/9797)

**发布日期**: 

---

众所周知，分类任务的标准损失是交叉熵（Cross Entropy，等价于最大似然MLE，即Maximum Likelihood Estimation），它有着简单高效的特点，但在某些场景下也暴露出一些问题，如偏离评价指标、过度自信等，相应的改进工作也有很多，此前我们也介绍过一些，比如[《再谈类别不平衡问题：调节权重与魔改Loss的对比联系》](/archives/7708)、[《如何训练你的准确率？》](/archives/9098)、[《缓解交叉熵过度自信的一个简明方案》](/archives/9526)等。由于LLM的训练也可以理解为逐token的分类任务，默认损失也是交叉熵，因此这些改进工作在LLM流行的今天依然有一定的价值。

在这篇文章中，我们介绍一篇名为[《EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling》](https://papers.cool/arxiv/2310.04691)的工作，它基于最优传输思想提出了新的改进损失函数EMO，声称能大幅提高LLM的微调效果。其中细节如何？让我们一探究竟。

## 概率散度 #

假设$p_i$是模型预测的第$i$个类别的概率，$i=1,2,\cdots,n$，$t$则是目标类别，那么交叉熵损失为  
\begin{equation}\mathcal{L} = - \log p_t\end{equation}  
如果将标签$t$用one hot形式的分布$\tau$表示出来（即$\tau_t=1,\tau_i=0|i\neq t, i\in[1,n]$），那么它可以重写成  
\begin{equation}\mathcal{L} = - \sum_i \tau_i\log p_i\end{equation}  
这个形式同时适用于非one hot的标签$\tau$（即软标签），它等价于优化$\tau,p$的KL散度：  
\begin{equation}KL(\tau\Vert p) = \sum_i \tau_i\log \frac{\tau_i}{p_i} = \color{skyblue}{\sum_i \tau_i\log \tau_i} - \sum_i \tau_i\log p_i\end{equation}  
当$\tau$给定时，最右端第一项就是一个常数，所以它跟交叉熵目标是等价的。

这个结果表明，我们在做MLE，或者说以交叉熵为损失时，实则就是在最小化目标分布和预测分布的KL散度。由于KL散度的一般推广是f散度（参考[《f-GAN简介：GAN模型的生产车间》](/archives/6016#f%E6%95%A3%E5%BA%A6)），所以很自然想到换用其他f散度或许有改良作用。事实上，确实有不少工作是按照这个思路进行的，比如[《缓解交叉熵过度自信的一个简明方案》](/archives/9526)介绍的方法，其论文的出发点是“Total Variation距离”，也是f散度的一种。

## 最优传输 #

不过，每种f散度或多或少有些问题，要说概率分布之间的理想度量，当属基于最优传输思想的“推土机距离（Earth Mover's Distance，EMD）”，不了解的读者可以参考一下笔者之前写的[《从Wasserstein距离、对偶理论到WGAN》](/archives/6280)。

简单来说，推土机距离定义为两个分布之间的最优传输成本：  
\begin{equation}\mathcal{C}[p,\tau]=\inf_{\gamma\in \Pi[p,\tau]} \sum_{i,j} \gamma_{i,j} c_{i,j} \end{equation}  
这里的$\gamma\in \Pi[p,\tau]$说的是$\gamma$是任意以$p,\tau$为边缘分布的联合分布，$c_{i,j}$是实现给定的成本函数，代表“从$i$搬运到$j$的成本”，$\inf$是下确界，意思就是说将最低的运输成本作为$p,\tau$之间的差异度量。正如基于f散度的Vanilla GAN换成基于最优传输的Wasserstein GAN能够更好的收敛性质，我们期望如果将分类的损失函数换成两个分布的W距离，也能收敛到更好的结果。

当$\tau$是one hot分布时，目标分布就是一个点$t$，那么就无所谓最不最优了，传输方案就只有一个，即把$p$的所有东西都搬到同一个点$t$，所以此时就有  
\begin{equation}\mathcal{C}[p,\tau]= \sum_i p_i c_{i,t} \label{eq:emo}\end{equation}

如果$\tau$是一般的软标签分布，那么$\mathcal{C}[p,\tau]$的计算是一个线性规划问题，求解起来比较复杂，由于$p_i \tau_j$所定义的分布也属于$\Pi[p,\tau]$，那么我们有  
\begin{equation}\mathcal{C}[p,\tau]=\inf_{\gamma\in \Pi[p,\tau]} \sum_{i,j} \gamma_{i,j} c_{i,j} \leq \sum_{i,j} p_i \tau_j c_{i,j} \end{equation}  
这是一个容易计算的上界，也可以作为优化目标，式$\eqref{eq:emo}$则对应$\tau_j = \delta_{j,t}$，其中$\delta$是“[克罗内克δ函数](https://en.wikipedia.org/wiki/Kronecker_delta)”。

## 成本函数 #

现在回到原论文所关心的场景——LLM的微调，包括二次预训练和微调到下游任务等。正如本文开头所述，LLM的训练可以理解为逐token的分类任务（类别即所有token），每个标签是one hot的，所以适用于式$\eqref{eq:emo}$。

式$\eqref{eq:emo}$还差成本函数$c_{i,t}$还没定下来。如果简单地认为只要$i\neq t$，那么成本都是1，即$c_{i,t}=1 - \delta_{i,t}$，那么  
\begin{equation}\mathcal{C}[p,\tau]= \sum_i p_i c_{i,t} = \sum_i (p_i - p_i \delta_{i, t}) = 1 - p_t\end{equation}  
这其实就是在最大化准确率的光滑近似（参考[《函数光滑化杂谈：不可导函数的可导逼近》](/archives/6620#%E6%AD%A3%E7%A1%AE%E7%8E%87)）。但直觉上，所有$i\neq t$都给予同样程度的惩罚似乎过于简单了，理想情况下应该根据相似度来给每个不同的$i$设计不同的成本，即相似度越大，传输成本越低，那么我们可以将传输成本设计为  
\begin{equation}c_{i,t} = 1 - \cos(\boldsymbol{e}_i,\boldsymbol{e}_t) = 1 - \left\langle\frac{\boldsymbol{e}_i}{\Vert\boldsymbol{e}_i\Vert}, \frac{\boldsymbol{e}_t}{\Vert\boldsymbol{e}_t\Vert}\right\rangle\end{equation}  
这里的$\boldsymbol{e}_i,\boldsymbol{e}_t$是事先获取到Token Embedding，原论文是将预训练模型的LM Head作为Token Embedding的，并且根据最优传输的定义成本函数是要实现给定的，因此计算相似度的Token Embedding要在训练过程中固定不变。

有了成本函数后，我们就可以计算  
\begin{equation}\mathcal{C}[p,\tau]= \sum_i p_i c_{i,t} = \sum_i \left(p_i - p_i \left\langle\frac{\boldsymbol{e}_i}{\Vert\boldsymbol{e}_i\Vert}, \frac{\boldsymbol{e}_t}{\Vert\boldsymbol{e}_t\Vert}\right\rangle\right) = 1 - \left\langle \sum_i p_i \frac{\boldsymbol{e}_i}{\Vert\boldsymbol{e}_i\Vert}, \frac{\boldsymbol{e}_t}{\Vert\boldsymbol{e}_t\Vert}\right\rangle\end{equation}  
这就是EMO（**E** arth **M** over Distance **O** ptimization）最终的训练损失。由于embedding_size通常远小于vocab_size，所以先算$\sum\limits_i p_i \frac{\boldsymbol{e}_i}{\Vert\boldsymbol{e}_i\Vert}$能明显降低计算量。

## 实验效果 #

由于笔者对LLM的研究还处于预训练阶段，还未涉及到微调，所以暂时没有自己的实验结果，只能先跟大家一起看看原论文的实验。不得不说，原论文的实验结果还是比较惊艳的。

首先，是小模型上的继续预训练实验，相比交叉熵（MLE）的提升最多的有10个点，并且是全面SOTA：  


[![小模型上的继续预训练对比实验](/usr/uploads/2023/10/765995927.png)](/usr/uploads/2023/10/765995927.png "点击查看原图")

小模型上的继续预训练对比实验

值得一提的是，这里的评价指标是MAUVE，越大越好，它提出自[《MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers》](https://papers.cool/arxiv/2102.01454)，是跟人工评价最相关的自动评测指标之一。此外，对比方法的TaiLr我们曾在[《缓解交叉熵过度自信的一个简明方案》](/archives/9526)简单介绍过。

可能有读者想EMO更好是不是单纯因为评价指标选得好？并不是，让人意外的是，EMO训练的模型，甚至PPL都更好（PPL跟MLE更相关）：  


[![不同评价指标的对比](/usr/uploads/2023/10/254031621.png)](/usr/uploads/2023/10/254031621.png "点击查看原图")

不同评价指标的对比

然后是将LLAMA-7B/13B微调到下游任务做Few Shot的效果，同样很出色：  


[![LLAMA-7B:13B微调到下游任务的效果](/usr/uploads/2023/10/760182866.png)](/usr/uploads/2023/10/760182866.png "点击查看原图")

LLAMA-7B:13B微调到下游任务的效果

最后对比了不同模型规模和数据规模的效果，显示出EMO在不同模型和数据规模上都有不错的表现：  


[![不同模型规模/数据规模上的效果](/usr/uploads/2023/10/1659559401.png)](/usr/uploads/2023/10/1659559401.png "点击查看原图")

不同模型规模/数据规模上的效果

## 个人思考 #

总的来说，原论文的“成绩单”还是非常漂亮的，值得一试。唯一的疑虑可能是原论文的实验数据量其实都不算大，不清楚进一步增大数据量后是否会缩小EMO和MLE的差距。

就笔者看来，EMO之所以能取得更好的结果，是因为它通过Embedding算相似度，来为“近义词”分配了更合理的损失，从而使得模型的学习更加合理。因为虽然形式上LLM也是分类任务，但它并不是一个简单的对与错问题，并不是说下一个预测的token跟标签token不一致，句子就不合理了，因此引入语义上的相似度来设计损失对LLM的训练是有帮助的。可以进一步猜测的是，vocab_size越大、token颗粒度越大的情况下，EMO的效果应该越好，因为vocab_size大了“近义词”就可能越多。

当然，引入语义相似度也导致了EMO不适用于从零训练，因为它需要一个训练好的LM Head作为Token Embedding。当然，一个可能的解决方案是考虑用其他方式，比如经典的Word2Vec来事先训练好Token Embedding，但这可能会有一个风险，即经典方式训练的Token Embedding是否会降低LLM能力的天花板（毕竟存在不一致性）。

此外，即便Token Embedding没问题，从零预训练时单纯用EMO可能还存在收敛过慢的问题，这是因为根据笔者在[《如何训练你的准确率？》](/archives/9098)的末尾提出的损失函数视角：

> 首先寻找评测指标的一个光滑近似，最好能表达成每个样本的期望形式，然后将错误方向的误差逐渐拉到无穷大（保证模型能更关注错误样本），但同时在正确方向保证与原始形式是一阶近似。

也就是说，为了保证（从零训练的）收敛速度，错误方向的损失最好能拉到无穷大，而EMO显然不满足这一点，因此将EMO用于从零训练的时候，大概率是EMO与MLE的某个加权组合，才能平衡收敛速度和最终效果。

## 文章小结 #

本文介绍了交叉熵损失的一个新的“替代品”——基于最优传输思想的EMO，与以往的小提升不同，EMO在LLM的微调实验中取得了较为明显的提升。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9797>_

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

苏剑林. (Oct. 13, 2023). 《EMO：基于最优传输思想设计的分类损失函数 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9797>

@online{kexuefm-9797,  
title={EMO：基于最优传输思想设计的分类损失函数},  
author={苏剑林},  
year={2023},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/9797}},  
} 


---

## 公式推导与注释

### 1. 最优传输理论基础

**Monge问题（1781）**：最优传输理论起源于Monge的"土方搬运"问题。给定两个概率测度$\mu$和$\nu$，寻找最优的运输方案$T: \mathcal{X} \to \mathcal{Y}$使得运输成本最小：

\begin{equation}
\inf_{T: T_{\#}\mu = \nu} \int_{\mathcal{X}} c(x, T(x)) d\mu(x) \tag{1}
\end{equation}

其中$c(x,y)$是从$x$到$y$的运输成本，$T_{\#}\mu$表示测度$\mu$在$T$下的推前（pushforward）。

**Kantorovich松弛（1942）**：Kantorovich将Monge问题松弛为寻找联合分布$\gamma \in \Pi(\mu, \nu)$：

\begin{equation}
\mathcal{C}[\mu, \nu] = \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\gamma(x, y) \tag{2}
\end{equation}

其中$\Pi(\mu, \nu)$是所有边缘分布为$\mu$和$\nu$的联合分布集合：

\begin{equation}
\Pi(\mu, \nu) = \left\{\gamma \in \mathcal{P}(\mathcal{X} \times \mathcal{Y}): \int_{\mathcal{Y}} d\gamma = \mu, \int_{\mathcal{X}} d\gamma = \nu\right\} \tag{3}
\end{equation}

**离散化**：对于离散分布$p = (p_1, \ldots, p_n)$和$\tau = (\tau_1, \ldots, \tau_n)$，最优传输问题变为线性规划：

\begin{equation}
\mathcal{C}[p, \tau] = \min_{\Gamma \in \mathbb{R}^{n \times n}} \sum_{i,j=1}^{n} \Gamma_{ij} C_{ij} \tag{4}
\end{equation}

约束条件：

\begin{equation}
\begin{cases}
\sum_{j=1}^{n} \Gamma_{ij} = p_i, & \forall i \\
\sum_{i=1}^{n} \Gamma_{ij} = \tau_j, & \forall j \\
\Gamma_{ij} \geq 0, & \forall i,j
\end{cases} \tag{5}
\end{equation}

其中$C_{ij} = c(i,j)$是成本矩阵。

### 2. Wasserstein距离详解

**$p$-Wasserstein距离**：对于$p \geq 1$，$p$-Wasserstein距离定义为：

\begin{equation}
W_p(\mu, \nu) = \left(\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y)^p d\gamma(x,y)\right)^{1/p} \tag{6}
\end{equation}

其中$d(x,y)$是基础度量空间上的距离。

**常用情形**：

1. **1-Wasserstein距离**（Earth Mover's Distance）：

   \begin{equation}
   W_1(\mu, \nu) = \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y) d\gamma(x,y) \tag{7}
   \end{equation}

2. **2-Wasserstein距离**：

   \begin{equation}
   W_2(\mu, \nu) = \sqrt{\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y)^2 d\gamma(x,y)} \tag{8}
   \end{equation}

**Kantorovich-Rubinstein对偶**：

\begin{equation}
W_1(\mu, \nu) = \sup_{f: \|f\|_L \leq 1} \left\{\int f d\mu - \int f d\nu\right\} \tag{9}
\end{equation}

其中$\|f\|_L$是Lipschitz常数。这是WGAN的理论基础。

**离散1-Wasserstein距离**：

\begin{equation}
W_1(p, \tau) = \min_{\Gamma \geq 0} \sum_{i,j} \Gamma_{ij} d_{ij}, \quad \text{s.t. } \Gamma \mathbf{1} = p, \Gamma^T \mathbf{1} = \tau \tag{10}
\end{equation}

### 3. EMO损失函数的推导

**从分类到最优传输**：

标准交叉熵损失：

\begin{equation}
\mathcal{L}_{\text{CE}} = -\log p_t = -\log \frac{e^{z_t}}{\sum_i e^{z_i}} \tag{11}
\end{equation}

KL散度形式：

\begin{equation}
\mathcal{L}_{\text{CE}} = \text{KL}(\tau \| p) = \sum_i \tau_i \log \frac{\tau_i}{p_i} = -\sum_i \tau_i \log p_i + \text{const} \tag{12}
\end{equation}

**问题**：KL散度不考虑类别之间的语义距离，所有错误类别的惩罚相同。

**EMO的动机**：使用最优传输距离$W_1(p, \tau)$替代KL散度，考虑类别间的语义相似性。

**One-hot目标的简化**：

当$\tau = e_t$（one-hot）时，最优传输方案唯一：将$p$的所有质量都运送到$t$，成本为：

\begin{equation}
\mathcal{C}[p, \tau] = \sum_{i=1}^{n} p_i c_{i,t} \tag{13}
\end{equation}

其中$c_{i,t}$是从类别$i$到目标类别$t$的运输成本。

**成本函数设计**：

基于embedding余弦距离：

\begin{equation}
c_{i,j} = 1 - \cos(\boldsymbol{e}_i, \boldsymbol{e}_j) = 1 - \frac{\langle \boldsymbol{e}_i, \boldsymbol{e}_j \rangle}{\|\boldsymbol{e}_i\| \|\boldsymbol{e}_j\|} \tag{14}
\end{equation}

其中$\{\boldsymbol{e}_i\}_{i=1}^n$是预训练的token embeddings（固定不变）。

**EMO损失**：

\begin{equation}
\mathcal{L}_{\text{EMO}} = \sum_{i=1}^{n} p_i c_{i,t} = \sum_{i=1}^{n} p_i (1 - \cos(\boldsymbol{e}_i, \boldsymbol{e}_t)) \tag{15}
\end{equation}

展开：

\begin{equation}
\mathcal{L}_{\text{EMO}} = 1 - \sum_{i=1}^{n} p_i \cos(\boldsymbol{e}_i, \boldsymbol{e}_t) = 1 - \left\langle \sum_{i=1}^{n} p_i \frac{\boldsymbol{e}_i}{\|\boldsymbol{e}_i\|}, \frac{\boldsymbol{e}_t}{\|\boldsymbol{e}_t\|} \right\rangle \tag{16}
\end{equation}

定义加权平均embedding：

\begin{equation}
\bar{\boldsymbol{e}}_p = \sum_{i=1}^{n} p_i \frac{\boldsymbol{e}_i}{\|\boldsymbol{e}_i\|} \tag{17}
\end{equation}

则：

\begin{equation}
\mathcal{L}_{\text{EMO}} = 1 - \cos(\bar{\boldsymbol{e}}_p, \boldsymbol{e}_t) \tag{18}
\end{equation}

**几何解释**：EMO损失衡量预测分布的加权平均embedding与目标embedding的余弦距离。

### 4. 梯度计算

**损失关于logits的梯度**：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z_j} = \frac{\partial}{\partial z_j} \left(\sum_i p_i c_{i,t}\right) \tag{19}
\end{equation}

利用$p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$，有：

\begin{equation}
\frac{\partial p_j}{\partial z_k} = p_j (\delta_{jk} - p_k) \tag{20}
\end{equation}

其中$\delta_{jk}$是Kronecker delta。

代入得：

\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z_j} &= \sum_i \frac{\partial p_i}{\partial z_j} c_{i,t} \\
&= \sum_i p_i (\delta_{ij} - p_j) c_{i,t} \\
&= p_j (c_{j,t} - \sum_i p_i c_{i,t}) \\
&= p_j (c_{j,t} - \mathcal{L}_{\text{EMO}})
\end{aligned} \tag{21}
\end{equation}

**与交叉熵梯度的对比**：

交叉熵：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_j} = p_j - \tau_j = p_j - \delta_{jt} \tag{22}
\end{equation}

EMO：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z_j} = p_j (c_{j,t} - \mathcal{L}_{\text{EMO}}) \tag{23}
\end{equation}

**关键区别**：

1. **加权**：EMO梯度乘以相对成本$(c_{j,t} - \mathcal{L}_{\text{EMO}})$
2. **自适应**：成本低于平均的类别梯度为负（鼓励），成本高于平均的梯度为正（抑制）
3. **语义感知**：近义词（$c_{j,t}$小）受到的惩罚小

### 5. 与KL散度等散度的对比

**f-散度族**：

一般形式：

\begin{equation}
D_f(p \| \tau) = \sum_i \tau_i f\left(\frac{p_i}{\tau_i}\right) \tag{24}
\end{equation}

常见实例：

1. **KL散度**：$f(t) = t \log t$

   \begin{equation}
   \text{KL}(p \| \tau) = \sum_i p_i \log \frac{p_i}{\tau_i} \tag{25}
   \end{equation}

2. **反KL散度**：$f(t) = -\log t$

   \begin{equation}
   \text{KL}(\tau \| p) = \sum_i \tau_i \log \frac{\tau_i}{p_i} \tag{26}
   \end{equation}

3. **Total Variation**：$f(t) = |t-1|$

   \begin{equation}
   \text{TV}(p, \tau) = \frac{1}{2}\sum_i |p_i - \tau_i| \tag{27}
   \end{equation}

4. **Hellinger距离**：$f(t) = (\sqrt{t} - 1)^2$

   \begin{equation}
   H^2(p, \tau) = \sum_i (\sqrt{p_i} - \sqrt{\tau_i})^2 \tag{28}
   \end{equation}

**Wasserstein vs f-散度**：

| 性质 | f-散度 | Wasserstein距离 |
|------|--------|----------------|
| 度量结构 | 非度量（不满足三角不等式） | 真度量 |
| 弱收敛 | 不连续 | 连续（弱拓扑） |
| 语义感知 | 否 | 是（通过成本矩阵） |
| 计算复杂度 | $O(n)$ | $O(n^3)$或$O(n^2/\varepsilon^2)$ |

**定理（弱收敛）**：设$\{p_n\}$弱收敛到$p$，则：

\begin{equation}
W_1(p_n, p) \to 0, \quad \text{但} \quad \text{KL}(p_n \| p) \not\to 0 \text{ (一般)} \tag{29}
\end{equation}

这解释了为什么Wasserstein距离在GAN训练中比f-散度更稳定。

### 6. Sinkhorn算法详解

**熵正则化最优传输**：

标准最优传输求解困难（$O(n^3 \log n)$），Cuturi (2013) 提出熵正则化：

\begin{equation}
\mathcal{C}_{\varepsilon}[p, \tau] = \min_{\Gamma \in \Pi(p, \tau)} \left\{\sum_{ij} \Gamma_{ij} C_{ij} - \varepsilon H(\Gamma)\right\} \tag{30}
\end{equation}

其中熵为：

\begin{equation}
H(\Gamma) = -\sum_{ij} \Gamma_{ij} (\log \Gamma_{ij} - 1) \tag{31}
\end{equation}

**Sinkhorn-Knopp算法**：

最优解具有形式：

\begin{equation}
\Gamma^* = \text{diag}(u) K \text{diag}(v) \tag{32}
\end{equation}

其中$K_{ij} = e^{-C_{ij}/\varepsilon}$是Gibbs核，$u, v$通过迭代求解：

\begin{equation}
\begin{cases}
u^{(k+1)} = \frac{p}{K v^{(k)}} \\
v^{(k+1)} = \frac{\tau}{K^T u^{(k+1)}}
\end{cases} \tag{33}
\end{equation}

**对数空间稳定版本**：

定义$\alpha = \varepsilon \log u$，$\beta = \varepsilon \log v$，迭代：

\begin{equation}
\begin{cases}
\alpha^{(k+1)} = \varepsilon \log p - \varepsilon \log\left(\sum_j e^{(\beta^{(k)}_j - C_{ij})/\varepsilon}\right) \\
\beta^{(k+1)} = \varepsilon \log \tau - \varepsilon \log\left(\sum_i e^{(\alpha^{(k+1)}_i - C_{ij})/\varepsilon}\right)
\end{cases} \tag{34}
\end{equation}

**Log-sum-exp技巧**：

\begin{equation}
\text{LSE}(x) = \log\sum_i e^{x_i} = \max_i x_i + \log\sum_i e^{x_i - \max_i x_i} \tag{35}
\end{equation}

**收敛速率**：

Sinkhorn算法以线性速率收敛：

\begin{equation}
\|\Gamma^{(k)} - \Gamma^*\|_F \leq C \cdot \rho^k, \quad \rho < 1 \tag{36}
\end{equation}

其中$\rho$依赖于$\varepsilon$和边缘分布的正则性。

**EMO中的应用**：

虽然EMO在one-hot情况下不需要Sinkhorn，但在soft label扩展中可以使用：

\begin{equation}
\mathcal{L}_{\text{EMO-soft}} = \min_{\Gamma \in \Pi(p, \tau)} \sum_{ij} \Gamma_{ij} C_{ij} \tag{37}
\end{equation}

用Sinkhorn近似求解。

### 7. 信息论解释

**互信息视角**：

定义联合分布$\gamma(i,j) = p_i \tau_j$（独立假设），则：

\begin{equation}
I(p; \tau) = \sum_{ij} \gamma(i,j) \log \frac{\gamma(i,j)}{p_i \tau_j} = 0 \tag{38}
\end{equation}

独立分布互信息为0，表示没有利用类别间关系。

**运输计划的互信息**：

最优传输计划$\Gamma^*$定义的互信息：

\begin{equation}
I(\Gamma^*) = \sum_{ij} \Gamma^*_{ij} \log \frac{\Gamma^*_{ij}}{p_i \tau_j} \tag{39}
\end{equation}

衡量了运输方案中编码的类别关联信息。

**熵正则化的作用**：

\begin{equation}
\mathcal{C}_{\varepsilon} = \mathcal{C} + \varepsilon \cdot I(\Gamma) \tag{40}
\end{equation}

$\varepsilon$控制运输方案的"随机性"：
- $\varepsilon \to 0$：确定性运输（可能不稳定）
- $\varepsilon \to \infty$：完全随机（$\Gamma \to p \otimes \tau$）

**条件熵**：

给定预测$p$，目标的条件熵：

\begin{equation}
H(\tau | p) = -\sum_{ij} \Gamma^*_{ij} \log \Gamma^*_{ij|i} \tag{41}
\end{equation}

其中$\Gamma^*_{ij|i} = \Gamma^*_{ij} / p_i$。

### 8. 概率论视角

**生成模型解释**：

将EMO视为生成过程：
1. 从预测分布$p$采样类别$i$
2. 以成本$c_{i,t}$"校正"到目标$t$

期望成本即EMO损失。

**Bayes风险**：

定义决策规则$\delta: \mathcal{X} \to \mathcal{Y}$，Bayes风险为：

\begin{equation}
R(\delta) = \mathbb{E}_{x \sim \mathcal{D}}[\mathbb{E}_{y \sim p(\cdot|x)}[c(\delta(x), y)]] \tag{42}
\end{equation}

EMO损失是Bayes风险在成本$c_{i,t}$下的实例化。

**Bayes最优分类器**：

\begin{equation}
\delta^*(x) = \arg\min_{\hat{y}} \sum_{y} p(y|x) c(\hat{y}, y) \tag{43}
\end{equation}

当$c = 0$-$1$损失时，退化为MAP分类器。

**后验校准**：

EMO隐式地鼓励后验校准：

\begin{equation}
p(\hat{y} = y | x) \approx \frac{\#\{x_i: \hat{y}_i = y\}}{N} \tag{44}
\end{equation}

通过惩罚语义上远离目标的预测。

### 9. 几何理解

**Embedding空间的流形结构**：

Token embeddings $\{\boldsymbol{e}_i\}$定义了vocabulary流形$\mathcal{M} \subset \mathbb{R}^d$。

**测地距离**：

在embedding空间中，成本$c_{i,j}$近似测地距离：

\begin{equation}
c_{i,j} \approx d_{\mathcal{M}}(\boldsymbol{e}_i, \boldsymbol{e}_j) \tag{45}
\end{equation}

**重心问题**：

EMO优化等价于寻找Wasserstein重心：

\begin{equation}
\arg\min_{\boldsymbol{e}} \sum_i p_i d(\boldsymbol{e}, \boldsymbol{e}_i) \tag{46}
\end{equation}

其解为：

\begin{equation}
\boldsymbol{e}^* = \frac{\sum_i p_i \boldsymbol{e}_i}{\|\sum_i p_i \boldsymbol{e}_i\|} \tag{47}
\end{equation}

（归一化后的加权平均）

**曲率的影响**：

在正曲率空间（如超球面），Wasserstein距离满足：

\begin{equation}
W_2^2(p, \tau) \leq \frac{1}{\kappa} \left(1 - \cos(\sqrt{\kappa} \cdot d_{\text{Euclidean}}(p, \tau))\right) \tag{48}
\end{equation}

其中$\kappa > 0$是曲率。

### 10. 理论性质

**性质1：非负性与恒等性**

\begin{equation}
\mathcal{L}_{\text{EMO}} \geq 0, \quad \mathcal{L}_{\text{EMO}} = 0 \Leftrightarrow p = e_t \tag{49}
\end{equation}

证明：$\mathcal{L}_{\text{EMO}} = 0$当且仅当$\bar{\boldsymbol{e}}_p = \boldsymbol{e}_t$，即$p = e_t$。

**性质2：Lipschitz连续性**

\begin{equation}
|\mathcal{L}_{\text{EMO}}(p_1) - \mathcal{L}_{\text{EMO}}(p_2)| \leq L \|p_1 - p_2\|_1 \tag{50}
\end{equation}

其中$L = \max_{i,j} c_{i,j} \leq 2$（余弦距离界）。

**性质3：凸性**

$\mathcal{L}_{\text{EMO}}$关于$p$是凸函数：

\begin{equation}
\mathcal{L}_{\text{EMO}}(\lambda p_1 + (1-\lambda)p_2) \leq \lambda \mathcal{L}_{\text{EMO}}(p_1) + (1-\lambda)\mathcal{L}_{\text{EMO}}(p_2) \tag{51}
\end{equation}

证明：余弦相似度是线性的，损失是余弦相似度的凸函数。

**性质4：梯度范数界**

\begin{equation}
\left\|\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z}\right\|_2 \leq \sqrt{n} \cdot \max_i |c_{i,t}| \leq 2\sqrt{n} \tag{52}
\end{equation}

**性质5：Hessian矩阵**

\begin{equation}
H_{jk} = \frac{\partial^2 \mathcal{L}_{\text{EMO}}}{\partial z_j \partial z_k} = p_j(\delta_{jk} - p_k)(c_{j,t} - c_{k,t}) \tag{53}
\end{equation}

### 11. 数值稳定性

**问题1：Embedding归一化**

当$\|\boldsymbol{e}_i\|$差异很大时，余弦相似度不稳定。

**解决方案**：预处理时L2归一化所有embeddings

\begin{equation}
\tilde{\boldsymbol{e}}_i = \frac{\boldsymbol{e}_i}{\max(\|\boldsymbol{e}_i\|, \epsilon)}, \quad \epsilon = 10^{-8} \tag{54}
\end{equation}

**问题2：数值下溢**

当$p_i$非常小时，$\sum_i p_i \boldsymbol{e}_i$可能下溢。

**解决方案**：在log space计算

\begin{equation}
\log \|\bar{\boldsymbol{e}}_p\| = \text{LSE}(\log p + \log \|\boldsymbol{e}\|) \tag{55}
\end{equation}

**问题3：梯度消失**

当预测非常准确时，梯度接近0。

**缓解**：结合交叉熵

\begin{equation}
\mathcal{L}_{\text{hybrid}} = (1-\alpha) \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{EMO}}, \quad \alpha \in [0,1] \tag{56}
\end{equation}

### 12. 具体计算示例

**设置**：
- Vocabulary size = 4
- Embedding dim = 2
- Target class = 2
- Logits $z = [1.0, 2.0, 0.5, 0.3]$

**步骤1：计算Softmax**

\begin{equation}
\begin{aligned}
Z &= \sum_i e^{z_i} = e^{1.0} + e^{2.0} + e^{0.5} + e^{0.3} = 2.718 + 7.389 + 1.649 + 1.350 = 13.106 \\
p &= [0.207, 0.564, 0.126, 0.103]
\end{aligned} \tag{57}
\end{equation}

**步骤2：Embeddings（归一化）**

\begin{equation}
\begin{aligned}
\boldsymbol{e}_1 &= [0.6, 0.8] \\
\boldsymbol{e}_2 &= [1.0, 0.0] \\
\boldsymbol{e}_3 &= [0.6, -0.8] \\
\boldsymbol{e}_4 &= [-0.8, 0.6]
\end{aligned} \tag{58}
\end{equation}

**步骤3：计算成本**

\begin{equation}
\begin{aligned}
c_{1,2} &= 1 - \langle [0.6, 0.8], [1.0, 0.0] \rangle = 1 - 0.6 = 0.4 \\
c_{2,2} &= 1 - 1.0 = 0.0 \\
c_{3,2} &= 1 - 0.6 = 0.4 \\
c_{4,2} &= 1 - (-0.8) = 1.8
\end{aligned} \tag{59}
\end{equation}

**步骤4：EMO损失**

\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{EMO}} &= \sum_i p_i c_{i,2} \\
&= 0.207 \times 0.4 + 0.564 \times 0 + 0.126 \times 0.4 + 0.103 \times 1.8 \\
&= 0.083 + 0 + 0.050 + 0.185 \\
&= 0.318
\end{aligned} \tag{60}
\end{equation}

**步骤5：梯度**

\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial z_1} &= p_1(c_{1,2} - \mathcal{L}_{\text{EMO}}) = 0.207 \times (0.4 - 0.318) = 0.017 \\
\frac{\partial \mathcal{L}}{\partial z_2} &= 0.564 \times (0 - 0.318) = -0.179 \\
\frac{\partial \mathcal{L}}{\partial z_3} &= 0.126 \times (0.4 - 0.318) = 0.010 \\
\frac{\partial \mathcal{L}}{\partial z_4} &= 0.103 \times (1.8 - 0.318) = 0.153
\end{aligned} \tag{61}
\end{equation}

**对比交叉熵**：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = p - e_2 = [0.207, -0.436, 0.126, 0.103] \tag{62}
\end{equation}

**观察**：EMO对语义相近的类别（1,3）惩罚更轻，对语义远离的类别（4）惩罚更重。

### 13. 与其他损失函数的对比

| 损失函数 | 形式 | 语义感知 | 计算复杂度 | 适用场景 |
|----------|------|----------|------------|----------|
| 交叉熵 | $-\log p_t$ | 否 | $O(n)$ | 通用分类 |
| Focal Loss | $-(1-p_t)^\gamma \log p_t$ | 否 | $O(n)$ | 类别不平衡 |
| Label Smoothing | $\sum_i \tau_i' \log p_i$ | 弱（uniform smoothing） | $O(n)$ | 正则化 |
| EMO | $\sum_i p_i c_{i,t}$ | 是（embedding-based） | $O(nd)$ | LLM、大词表 |
| Sinkhorn Loss | $\min_{\Gamma} \langle \Gamma, C \rangle$ | 是 | $O(n^2 K)$ | Soft labels |

### 14. 实践建议

**Embedding选择**：

1. **预训练Embeddings**：使用模型的输出层权重

   \begin{equation}
   \boldsymbol{e}_i = W_{\text{out}}[i, :] \tag{63}
   \end{equation}

2. **Word2Vec/GloVe**：对于没有预训练模型的情况

3. **Contextualized Embeddings**：对每个token取平均

**超参数**：

1. **混合系数$\alpha$**（hybrid loss）：

   - 预训练from scratch：$\alpha = 0.1 \sim 0.3$
   - Fine-tuning：$\alpha = 0.5 \sim 0.9$

2. **Embedding固定 vs 微调**：

   - 固定：保持成本矩阵不变（理论保证）
   - 微调：可能提升性能但失去理论基础

**实现技巧**：

```python
# 伪代码
def emo_loss(logits, targets, embeddings):
    # logits: [B, V], targets: [B], embeddings: [V, D]
    p = softmax(logits, dim=-1)  # [B, V]

    # 归一化embeddings
    emb_norm = F.normalize(embeddings, dim=-1)  # [V, D]

    # 计算加权平均embedding
    weighted_emb = torch.matmul(p, emb_norm)  # [B, D]

    # 目标embedding
    target_emb = emb_norm[targets]  # [B, D]

    # 余弦相似度
    cos_sim = (weighted_emb * target_emb).sum(dim=-1)  # [B]

    # EMO loss
    loss = 1 - cos_sim

    return loss.mean()
```

**调试checklist**：

1. 验证embeddings已归一化
2. 检查成本矩阵对称性和三角不等式
3. 监控损失值范围（应在$[0, 2]$）
4. 对比梯度方向与交叉熵

### 15. 理论扩展

**扩展1：多标签EMO**

\begin{equation}
\mathcal{L}_{\text{multi-EMO}} = \sum_{t \in \mathcal{T}} w_t \sum_i p_i c_{i,t} \tag{64}
\end{equation}

其中$\mathcal{T}$是目标类别集，$w_t$是权重。

**扩展2：Soft Label EMO**

\begin{equation}
\mathcal{L}_{\text{soft-EMO}} = \sum_{i,j} \Gamma^*_{ij} c_{i,j} \tag{65}
\end{equation}

使用Sinkhorn求解$\Gamma^*$。

**扩展3：成本学习**

将成本矩阵$C$作为参数联合优化：

\begin{equation}
\min_{\theta, C} \mathbb{E}[\mathcal{L}_{\text{EMO}}(p_\theta, C)] + \lambda R(C) \tag{66}
\end{equation}

其中$R(C)$是正则项（如鼓励度量性质）。

---

**总结**：本节全面介绍了基于最优传输理论的EMO损失函数，从理论基础（Monge-Kantorovich问题）到实际应用（LLM微调），涵盖Wasserstein距离、Sinkhorn算法、信息论解释、几何视角等多个方面。详细的推导和实例展示了EMO如何通过考虑类别间语义距离来改进标准交叉熵损失。


---

## 公式推导与注释

### 1. 最优传输理论基础

**Monge问题（1781）**：最优传输理论起源于Monge的"土方搬运"问题。给定两个概率测度$\mu$和$\nu$，寻找最优的运输方案$T: \mathcal{X} \to \mathcal{Y}$使得运输成本最小：

\begin{equation}
\inf_{T: T_{\#}\mu = \nu} \int_{\mathcal{X}} c(x, T(x)) d\mu(x) \tag{1}
\end{equation}

其中$c(x,y)$是从$x$到$y$的运输成本，$T_{\#}\mu$表示测度$\mu$在$T$下的推前（pushforward）。

**Kantorovich松弛（1942）**：Kantorovich将Monge问题松弛为寻找联合分布$\gamma \in \Pi(\mu, \nu)$：

\begin{equation}
\mathcal{C}[\mu, \nu] = \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) d\gamma(x, y) \tag{2}
\end{equation}

其中$\Pi(\mu, \nu)$是所有边缘分布为$\mu$和$\nu$的联合分布集合：

\begin{equation}
\Pi(\mu, \nu) = \left\{\gamma \in \mathcal{P}(\mathcal{X} \times \mathcal{Y}): \int_{\mathcal{Y}} d\gamma = \mu, \int_{\mathcal{X}} d\gamma = \nu\right\} \tag{3}
\end{equation}

**离散化**：对于离散分布$p = (p_1, \ldots, p_n)$和$\tau = (\tau_1, \ldots, \tau_n)$，最优传输问题变为线性规划：

\begin{equation}
\mathcal{C}[p, \tau] = \min_{\Gamma \in \mathbb{R}^{n \times n}} \sum_{i,j=1}^{n} \Gamma_{ij} C_{ij} \tag{4}
\end{equation}

约束条件：

\begin{equation}
\begin{cases}
\sum_{j=1}^{n} \Gamma_{ij} = p_i, & \forall i \\
\sum_{i=1}^{n} \Gamma_{ij} = \tau_j, & \forall j \\
\Gamma_{ij} \geq 0, & \forall i,j
\end{cases} \tag{5}
\end{equation}

其中$C_{ij} = c(i,j)$是成本矩阵。

### 2. Wasserstein距离详解

**$p$-Wasserstein距离**：对于$p \geq 1$，$p$-Wasserstein距离定义为：

\begin{equation}
W_p(\mu, \nu) = \left(\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y)^p d\gamma(x,y)\right)^{1/p} \tag{6}
\end{equation}

其中$d(x,y)$是基础度量空间上的距离。

**常用情形**：

1. **1-Wasserstein距离**（Earth Mover's Distance）：

   \begin{equation}
   W_1(\mu, \nu) = \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y) d\gamma(x,y) \tag{7}
   \end{equation}

2. **2-Wasserstein距离**：

   \begin{equation}
   W_2(\mu, \nu) = \sqrt{\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{Y}} d(x,y)^2 d\gamma(x,y)} \tag{8}
   \end{equation}

**Kantorovich-Rubinstein对偶**：

\begin{equation}
W_1(\mu, \nu) = \sup_{f: \|f\|_L \leq 1} \left\{\int f d\mu - \int f d\nu\right\} \tag{9}
\end{equation}

其中$\|f\|_L$是Lipschitz常数。这是WGAN的理论基础。

**离散1-Wasserstein距离**：

\begin{equation}
W_1(p, \tau) = \min_{\Gamma \geq 0} \sum_{i,j} \Gamma_{ij} d_{ij}, \quad \text{s.t. } \Gamma \mathbf{1} = p, \Gamma^T \mathbf{1} = \tau \tag{10}
\end{equation}

### 3. EMO损失函数的推导

**从分类到最优传输**：

标准交叉熵损失：

\begin{equation}
\mathcal{L}_{\text{CE}} = -\log p_t = -\log \frac{e^{z_t}}{\sum_i e^{z_i}} \tag{11}
\end{equation}

KL散度形式：

\begin{equation}
\mathcal{L}_{\text{CE}} = \text{KL}(\tau \| p) = \sum_i \tau_i \log \frac{\tau_i}{p_i} = -\sum_i \tau_i \log p_i + \text{const} \tag{12}
\end{equation}

**问题**：KL散度不考虑类别之间的语义距离，所有错误类别的惩罚相同。

**EMO的动机**：使用最优传输距离$W_1(p, \tau)$替代KL散度，考虑类别间的语义相似性。

**One-hot目标的简化**：

当$\tau = e_t$（one-hot）时，最优传输方案唯一：将$p$的所有质量都运送到$t$，成本为：

\begin{equation}
\mathcal{C}[p, \tau] = \sum_{i=1}^{n} p_i c_{i,t} \tag{13}
\end{equation}

其中$c_{i,t}$是从类别$i$到目标类别$t$的运输成本。

**成本函数设计**：

基于embedding余弦距离：

\begin{equation}
c_{i,j} = 1 - \cos(\boldsymbol{e}_i, \boldsymbol{e}_j) = 1 - \frac{\langle \boldsymbol{e}_i, \boldsymbol{e}_j \rangle}{\|\boldsymbol{e}_i\| \|\boldsymbol{e}_j\|} \tag{14}
\end{equation}

其中$\{\boldsymbol{e}_i\}_{i=1}^n$是预训练的token embeddings（固定不变）。

**EMO损失**：

\begin{equation}
\mathcal{L}_{\text{EMO}} = \sum_{i=1}^{n} p_i c_{i,t} = \sum_{i=1}^{n} p_i (1 - \cos(\boldsymbol{e}_i, \boldsymbol{e}_t)) \tag{15}
\end{equation}

展开：

\begin{equation}
\mathcal{L}_{\text{EMO}} = 1 - \sum_{i=1}^{n} p_i \cos(\boldsymbol{e}_i, \boldsymbol{e}_t) = 1 - \left\langle \sum_{i=1}^{n} p_i \frac{\boldsymbol{e}_i}{\|\boldsymbol{e}_i\|}, \frac{\boldsymbol{e}_t}{\|\boldsymbol{e}_t\|} \right\rangle \tag{16}
\end{equation}

定义加权平均embedding：

\begin{equation}
\bar{\boldsymbol{e}}_p = \sum_{i=1}^{n} p_i \frac{\boldsymbol{e}_i}{\|\boldsymbol{e}_i\|} \tag{17}
\end{equation}

则：

\begin{equation}
\mathcal{L}_{\text{EMO}} = 1 - \cos(\bar{\boldsymbol{e}}_p, \boldsymbol{e}_t) \tag{18}
\end{equation}

**几何解释**：EMO损失衡量预测分布的加权平均embedding与目标embedding的余弦距离。

### 4. 梯度计算

**损失关于logits的梯度**：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z_j} = \frac{\partial}{\partial z_j} \left(\sum_i p_i c_{i,t}\right) \tag{19}
\end{equation}

利用$p_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$，有：

\begin{equation}
\frac{\partial p_j}{\partial z_k} = p_j (\delta_{jk} - p_k) \tag{20}
\end{equation}

其中$\delta_{jk}$是Kronecker delta。

代入得：

\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z_j} &= \sum_i \frac{\partial p_i}{\partial z_j} c_{i,t} \\
&= \sum_i p_i (\delta_{ij} - p_j) c_{i,t} \\
&= p_j (c_{j,t} - \sum_i p_i c_{i,t}) \\
&= p_j (c_{j,t} - \mathcal{L}_{\text{EMO}})
\end{aligned} \tag{21}
\end{equation}

**与交叉熵梯度的对比**：

交叉熵：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_j} = p_j - \tau_j = p_j - \delta_{jt} \tag{22}
\end{equation}

EMO：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z_j} = p_j (c_{j,t} - \mathcal{L}_{\text{EMO}}) \tag{23}
\end{equation}

**关键区别**：

1. **加权**：EMO梯度乘以相对成本$(c_{j,t} - \mathcal{L}_{\text{EMO}})$
2. **自适应**：成本低于平均的类别梯度为负（鼓励），成本高于平均的梯度为正（抑制）
3. **语义感知**：近义词（$c_{j,t}$小）受到的惩罚小

### 5. 与KL散度等散度的对比

**f-散度族**：

一般形式：

\begin{equation}
D_f(p \| \tau) = \sum_i \tau_i f\left(\frac{p_i}{\tau_i}\right) \tag{24}
\end{equation}

常见实例：

1. **KL散度**：$f(t) = t \log t$

   \begin{equation}
   \text{KL}(p \| \tau) = \sum_i p_i \log \frac{p_i}{\tau_i} \tag{25}
   \end{equation}

2. **反KL散度**：$f(t) = -\log t$

   \begin{equation}
   \text{KL}(\tau \| p) = \sum_i \tau_i \log \frac{\tau_i}{p_i} \tag{26}
   \end{equation}

3. **Total Variation**：$f(t) = |t-1|$

   \begin{equation}
   \text{TV}(p, \tau) = \frac{1}{2}\sum_i |p_i - \tau_i| \tag{27}
   \end{equation}

4. **Hellinger距离**：$f(t) = (\sqrt{t} - 1)^2$

   \begin{equation}
   H^2(p, \tau) = \sum_i (\sqrt{p_i} - \sqrt{\tau_i})^2 \tag{28}
   \end{equation}

**Wasserstein vs f-散度**：

| 性质 | f-散度 | Wasserstein距离 |
|------|--------|----------------|
| 度量结构 | 非度量（不满足三角不等式） | 真度量 |
| 弱收敛 | 不连续 | 连续（弱拓扑） |
| 语义感知 | 否 | 是（通过成本矩阵） |
| 计算复杂度 | $O(n)$ | $O(n^3)$或$O(n^2/\varepsilon^2)$ |

**定理（弱收敛）**：设$\{p_n\}$弱收敛到$p$，则：

\begin{equation}
W_1(p_n, p) \to 0, \quad \text{但} \quad \text{KL}(p_n \| p) \not\to 0 \text{ (一般)} \tag{29}
\end{equation}

这解释了为什么Wasserstein距离在GAN训练中比f-散度更稳定。

### 6. Sinkhorn算法详解

**熵正则化最优传输**：

标准最优传输求解困难（$O(n^3 \log n)$），Cuturi (2013) 提出熵正则化：

\begin{equation}
\mathcal{C}_{\varepsilon}[p, \tau] = \min_{\Gamma \in \Pi(p, \tau)} \left\{\sum_{ij} \Gamma_{ij} C_{ij} - \varepsilon H(\Gamma)\right\} \tag{30}
\end{equation}

其中熵为：

\begin{equation}
H(\Gamma) = -\sum_{ij} \Gamma_{ij} (\log \Gamma_{ij} - 1) \tag{31}
\end{equation}

**Sinkhorn-Knopp算法**：

最优解具有形式：

\begin{equation}
\Gamma^* = \text{diag}(u) K \text{diag}(v) \tag{32}
\end{equation}

其中$K_{ij} = e^{-C_{ij}/\varepsilon}$是Gibbs核，$u, v$通过迭代求解：

\begin{equation}
\begin{cases}
u^{(k+1)} = \frac{p}{K v^{(k)}} \\
v^{(k+1)} = \frac{\tau}{K^T u^{(k+1)}}
\end{cases} \tag{33}
\end{equation}

**对数空间稳定版本**：

定义$\alpha = \varepsilon \log u$，$\beta = \varepsilon \log v$，迭代：

\begin{equation}
\begin{cases}
\alpha^{(k+1)} = \varepsilon \log p - \varepsilon \log\left(\sum_j e^{(\beta^{(k)}_j - C_{ij})/\varepsilon}\right) \\
\beta^{(k+1)} = \varepsilon \log \tau - \varepsilon \log\left(\sum_i e^{(\alpha^{(k+1)}_i - C_{ij})/\varepsilon}\right)
\end{cases} \tag{34}
\end{equation}

**Log-sum-exp技巧**：

\begin{equation}
\text{LSE}(x) = \log\sum_i e^{x_i} = \max_i x_i + \log\sum_i e^{x_i - \max_i x_i} \tag{35}
\end{equation}

**收敛速率**：

Sinkhorn算法以线性速率收敛：

\begin{equation}
\|\Gamma^{(k)} - \Gamma^*\|_F \leq C \cdot \rho^k, \quad \rho < 1 \tag{36}
\end{equation}

其中$\rho$依赖于$\varepsilon$和边缘分布的正则性。

**EMO中的应用**：

虽然EMO在one-hot情况下不需要Sinkhorn，但在soft label扩展中可以使用：

\begin{equation}
\mathcal{L}_{\text{EMO-soft}} = \min_{\Gamma \in \Pi(p, \tau)} \sum_{ij} \Gamma_{ij} C_{ij} \tag{37}
\end{equation}

用Sinkhorn近似求解。

### 7. 信息论解释

**互信息视角**：

定义联合分布$\gamma(i,j) = p_i \tau_j$（独立假设），则：

\begin{equation}
I(p; \tau) = \sum_{ij} \gamma(i,j) \log \frac{\gamma(i,j)}{p_i \tau_j} = 0 \tag{38}
\end{equation}

独立分布互信息为0，表示没有利用类别间关系。

**运输计划的互信息**：

最优传输计划$\Gamma^*$定义的互信息：

\begin{equation}
I(\Gamma^*) = \sum_{ij} \Gamma^*_{ij} \log \frac{\Gamma^*_{ij}}{p_i \tau_j} \tag{39}
\end{equation}

衡量了运输方案中编码的类别关联信息。

**熵正则化的作用**：

\begin{equation}
\mathcal{C}_{\varepsilon} = \mathcal{C} + \varepsilon \cdot I(\Gamma) \tag{40}
\end{equation}

$\varepsilon$控制运输方案的"随机性"：
- $\varepsilon \to 0$：确定性运输（可能不稳定）
- $\varepsilon \to \infty$：完全随机（$\Gamma \to p \otimes \tau$）

**条件熵**：

给定预测$p$，目标的条件熵：

\begin{equation}
H(\tau | p) = -\sum_{ij} \Gamma^*_{ij} \log \Gamma^*_{ij|i} \tag{41}
\end{equation}

其中$\Gamma^*_{ij|i} = \Gamma^*_{ij} / p_i$。

### 8. 概率论视角

**生成模型解释**：

将EMO视为生成过程：
1. 从预测分布$p$采样类别$i$
2. 以成本$c_{i,t}$"校正"到目标$t$

期望成本即EMO损失。

**Bayes风险**：

定义决策规则$\delta: \mathcal{X} \to \mathcal{Y}$，Bayes风险为：

\begin{equation}
R(\delta) = \mathbb{E}_{x \sim \mathcal{D}}[\mathbb{E}_{y \sim p(\cdot|x)}[c(\delta(x), y)]] \tag{42}
\end{equation}

EMO损失是Bayes风险在成本$c_{i,t}$下的实例化。

**Bayes最优分类器**：

\begin{equation}
\delta^*(x) = \arg\min_{\hat{y}} \sum_{y} p(y|x) c(\hat{y}, y) \tag{43}
\end{equation}

当$c = 0$-$1$损失时，退化为MAP分类器。

**后验校准**：

EMO隐式地鼓励后验校准：

\begin{equation}
p(\hat{y} = y | x) \approx \frac{\#\{x_i: \hat{y}_i = y\}}{N} \tag{44}
\end{equation}

通过惩罚语义上远离目标的预测。

### 9. 几何理解

**Embedding空间的流形结构**：

Token embeddings $\{\boldsymbol{e}_i\}$定义了vocabulary流形$\mathcal{M} \subset \mathbb{R}^d$。

**测地距离**：

在embedding空间中，成本$c_{i,j}$近似测地距离：

\begin{equation}
c_{i,j} \approx d_{\mathcal{M}}(\boldsymbol{e}_i, \boldsymbol{e}_j) \tag{45}
\end{equation}

**重心问题**：

EMO优化等价于寻找Wasserstein重心：

\begin{equation}
\arg\min_{\boldsymbol{e}} \sum_i p_i d(\boldsymbol{e}, \boldsymbol{e}_i) \tag{46}
\end{equation}

其解为：

\begin{equation}
\boldsymbol{e}^* = \frac{\sum_i p_i \boldsymbol{e}_i}{\|\sum_i p_i \boldsymbol{e}_i\|} \tag{47}
\end{equation}

（归一化后的加权平均）

**曲率的影响**：

在正曲率空间（如超球面），Wasserstein距离满足：

\begin{equation}
W_2^2(p, \tau) \leq \frac{1}{\kappa} \left(1 - \cos(\sqrt{\kappa} \cdot d_{\text{Euclidean}}(p, \tau))\right) \tag{48}
\end{equation}

其中$\kappa > 0$是曲率。

### 10. 理论性质

**性质1：非负性与恒等性**

\begin{equation}
\mathcal{L}_{\text{EMO}} \geq 0, \quad \mathcal{L}_{\text{EMO}} = 0 \Leftrightarrow p = e_t \tag{49}
\end{equation}

证明：$\mathcal{L}_{\text{EMO}} = 0$当且仅当$\bar{\boldsymbol{e}}_p = \boldsymbol{e}_t$，即$p = e_t$。

**性质2：Lipschitz连续性**

\begin{equation}
|\mathcal{L}_{\text{EMO}}(p_1) - \mathcal{L}_{\text{EMO}}(p_2)| \leq L \|p_1 - p_2\|_1 \tag{50}
\end{equation}

其中$L = \max_{i,j} c_{i,j} \leq 2$（余弦距离界）。

**性质3：凸性**

$\mathcal{L}_{\text{EMO}}$关于$p$是凸函数：

\begin{equation}
\mathcal{L}_{\text{EMO}}(\lambda p_1 + (1-\lambda)p_2) \leq \lambda \mathcal{L}_{\text{EMO}}(p_1) + (1-\lambda)\mathcal{L}_{\text{EMO}}(p_2) \tag{51}
\end{equation}

证明：余弦相似度是线性的，损失是余弦相似度的凸函数。

**性质4：梯度范数界**

\begin{equation}
\left\|\frac{\partial \mathcal{L}_{\text{EMO}}}{\partial z}\right\|_2 \leq \sqrt{n} \cdot \max_i |c_{i,t}| \leq 2\sqrt{n} \tag{52}
\end{equation}

**性质5：Hessian矩阵**

\begin{equation}
H_{jk} = \frac{\partial^2 \mathcal{L}_{\text{EMO}}}{\partial z_j \partial z_k} = p_j(\delta_{jk} - p_k)(c_{j,t} - c_{k,t}) \tag{53}
\end{equation}

### 11. 数值稳定性

**问题1：Embedding归一化**

当$\|\boldsymbol{e}_i\|$差异很大时，余弦相似度不稳定。

**解决方案**：预处理时L2归一化所有embeddings

\begin{equation}
\tilde{\boldsymbol{e}}_i = \frac{\boldsymbol{e}_i}{\max(\|\boldsymbol{e}_i\|, \epsilon)}, \quad \epsilon = 10^{-8} \tag{54}
\end{equation}

**问题2：数值下溢**

当$p_i$非常小时，$\sum_i p_i \boldsymbol{e}_i$可能下溢。

**解决方案**：在log space计算

\begin{equation}
\log \|\bar{\boldsymbol{e}}_p\| = \text{LSE}(\log p + \log \|\boldsymbol{e}\|) \tag{55}
\end{equation}

**问题3：梯度消失**

当预测非常准确时，梯度接近0。

**缓解**：结合交叉熵

\begin{equation}
\mathcal{L}_{\text{hybrid}} = (1-\alpha) \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{EMO}}, \quad \alpha \in [0,1] \tag{56}
\end{equation}

### 12. 具体计算示例

**设置**：
- Vocabulary size = 4
- Embedding dim = 2
- Target class = 2
- Logits $z = [1.0, 2.0, 0.5, 0.3]$

**步骤1：计算Softmax**

\begin{equation}
\begin{aligned}
Z &= \sum_i e^{z_i} = e^{1.0} + e^{2.0} + e^{0.5} + e^{0.3} = 2.718 + 7.389 + 1.649 + 1.350 = 13.106 \\
p &= [0.207, 0.564, 0.126, 0.103]
\end{aligned} \tag{57}
\end{equation}

**步骤2：Embeddings（归一化）**

\begin{equation}
\begin{aligned}
\boldsymbol{e}_1 &= [0.6, 0.8] \\
\boldsymbol{e}_2 &= [1.0, 0.0] \\
\boldsymbol{e}_3 &= [0.6, -0.8] \\
\boldsymbol{e}_4 &= [-0.8, 0.6]
\end{aligned} \tag{58}
\end{equation}

**步骤3：计算成本**

\begin{equation}
\begin{aligned}
c_{1,2} &= 1 - \langle [0.6, 0.8], [1.0, 0.0] \rangle = 1 - 0.6 = 0.4 \\
c_{2,2} &= 1 - 1.0 = 0.0 \\
c_{3,2} &= 1 - 0.6 = 0.4 \\
c_{4,2} &= 1 - (-0.8) = 1.8
\end{aligned} \tag{59}
\end{equation}

**步骤4：EMO损失**

\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{EMO}} &= \sum_i p_i c_{i,2} \\
&= 0.207 \times 0.4 + 0.564 \times 0 + 0.126 \times 0.4 + 0.103 \times 1.8 \\
&= 0.083 + 0 + 0.050 + 0.185 \\
&= 0.318
\end{aligned} \tag{60}
\end{equation}

**步骤5：梯度**

\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial z_1} &= p_1(c_{1,2} - \mathcal{L}_{\text{EMO}}) = 0.207 \times (0.4 - 0.318) = 0.017 \\
\frac{\partial \mathcal{L}}{\partial z_2} &= 0.564 \times (0 - 0.318) = -0.179 \\
\frac{\partial \mathcal{L}}{\partial z_3} &= 0.126 \times (0.4 - 0.318) = 0.010 \\
\frac{\partial \mathcal{L}}{\partial z_4} &= 0.103 \times (1.8 - 0.318) = 0.153
\end{aligned} \tag{61}
\end{equation}

**对比交叉熵**：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = p - e_2 = [0.207, -0.436, 0.126, 0.103] \tag{62}
\end{equation}

**观察**：EMO对语义相近的类别（1,3）惩罚更轻，对语义远离的类别（4）惩罚更重。

### 13. 与其他损失函数的对比

| 损失函数 | 形式 | 语义感知 | 计算复杂度 | 适用场景 |
|----------|------|----------|------------|----------|
| 交叉熵 | $-\log p_t$ | 否 | $O(n)$ | 通用分类 |
| Focal Loss | $-(1-p_t)^\gamma \log p_t$ | 否 | $O(n)$ | 类别不平衡 |
| Label Smoothing | $\sum_i \tau_i' \log p_i$ | 弱（uniform smoothing） | $O(n)$ | 正则化 |
| EMO | $\sum_i p_i c_{i,t}$ | 是（embedding-based） | $O(nd)$ | LLM、大词表 |
| Sinkhorn Loss | $\min_{\Gamma} \langle \Gamma, C \rangle$ | 是 | $O(n^2 K)$ | Soft labels |

### 14. 实践建议

**Embedding选择**：

1. **预训练Embeddings**：使用模型的输出层权重

   \begin{equation}
   \boldsymbol{e}_i = W_{\text{out}}[i, :] \tag{63}
   \end{equation}

2. **Word2Vec/GloVe**：对于没有预训练模型的情况

3. **Contextualized Embeddings**：对每个token取平均

**超参数**：

1. **混合系数$\alpha$**（hybrid loss）：

   - 预训练from scratch：$\alpha = 0.1 \sim 0.3$
   - Fine-tuning：$\alpha = 0.5 \sim 0.9$

2. **Embedding固定 vs 微调**：

   - 固定：保持成本矩阵不变（理论保证）
   - 微调：可能提升性能但失去理论基础

**实现技巧**：

```python
# 伪代码
def emo_loss(logits, targets, embeddings):
    # logits: [B, V], targets: [B], embeddings: [V, D]
    p = softmax(logits, dim=-1)  # [B, V]

    # 归一化embeddings
    emb_norm = F.normalize(embeddings, dim=-1)  # [V, D]

    # 计算加权平均embedding
    weighted_emb = torch.matmul(p, emb_norm)  # [B, D]

    # 目标embedding
    target_emb = emb_norm[targets]  # [B, D]

    # 余弦相似度
    cos_sim = (weighted_emb * target_emb).sum(dim=-1)  # [B]

    # EMO loss
    loss = 1 - cos_sim

    return loss.mean()
```

**调试checklist**：

1. 验证embeddings已归一化
2. 检查成本矩阵对称性和三角不等式
3. 监控损失值范围（应在$[0, 2]$）
4. 对比梯度方向与交叉熵

### 15. 理论扩展

**扩展1：多标签EMO**

\begin{equation}
\mathcal{L}_{\text{multi-EMO}} = \sum_{t \in \mathcal{T}} w_t \sum_i p_i c_{i,t} \tag{64}
\end{equation}

其中$\mathcal{T}$是目标类别集，$w_t$是权重。

**扩展2：Soft Label EMO**

\begin{equation}
\mathcal{L}_{\text{soft-EMO}} = \sum_{i,j} \Gamma^*_{ij} c_{i,j} \tag{65}
\end{equation}

使用Sinkhorn求解$\Gamma^*$。

**扩展3：成本学习**

将成本矩阵$C$作为参数联合优化：

\begin{equation}
\min_{\theta, C} \mathbb{E}[\mathcal{L}_{\text{EMO}}(p_\theta, C)] + \lambda R(C) \tag{66}
\end{equation}

其中$R(C)$是正则项（如鼓励度量性质）。

---

**总结**：本节全面介绍了基于最优传输理论的EMO损失函数，从理论基础（Monge-Kantorovich问题）到实际应用（LLM微调），涵盖Wasserstein距离、Sinkhorn算法、信息论解释、几何视角等多个方面。详细的推导和实例展示了EMO如何通过考虑类别间语义距离来改进标准交叉熵损失。

