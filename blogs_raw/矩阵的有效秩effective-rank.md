---
title: 矩阵的有效秩（Effective Rank）
slug: 矩阵的有效秩effective-rank
date: 2025-04-10
tags: 矩阵, 熵, 稀疏, 低秩, 生成模型
status: pending
---

# 矩阵的有效秩（Effective Rank）

**原文链接**: [https://spaces.ac.cn/archives/10847](https://spaces.ac.cn/archives/10847)

**发布日期**: 

---

秩（Rank）是线性代数中的重要概念，它代表了矩阵的内在维度。然而，数学上对秩的严格定义，很多时候并不完全适用于数值计算场景，因为秩等于非零奇异值的个数，而数学上对“等于零”这件事的理解跟数值计算有所不同，数学上的“等于零”是绝对地、严格地等于零，哪怕是$10^{-100}$也是不等于零，但数值计算不一样，很多时候$10^{-10}$就可以当零看待。

因此，我们希望将秩的概念推广到更符合数值计算特性的形式，这便是有效秩（Effective Rank）概念的由来。

## 误差截断 #

需要指出的是，目前学术界对有效秩并没有统一的定义，接下来我们介绍的是一些从不同角度切入来定义有效秩的思路。对于实际问题，读者可以自行选择适合的定义来使用。

接下来我们主要从奇异值的角度来考虑秩。对于矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，不失一般性，设$n\leq m$，那么它的标准秩为  
\begin{equation}\mathop{\text{rank}}(\boldsymbol{M}) \triangleq \max\\{i\,|\,\sigma_i > 0\\}\leq n\end{equation}  
其中$\sigma_1\geq \sigma_2\geq \cdots\geq\sigma_n \geq 0$是$\boldsymbol{M}$的奇异值。直观上，有效秩的概念就是为了将接近于零的奇异值当成零处理，所以有效秩的一个基本属性是不超过标准秩，并且特殊情况下能退化成标准秩。满足该属性的一个简单的定义是  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M},\epsilon) \triangleq \max\\{i\,|\,\sigma_i > \epsilon\\}\end{equation}  
然而，“大”与“小”的概念应该是相对的，1相对于0.01大，但相对于100就小了，所以看起来除以$\sigma_1$来标准化一下更科学：  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M},\epsilon) \triangleq \max\big\\{i\,\big|\,\sigma_i/\sigma_1 > \epsilon\big\\}\end{equation}

除了直接对接近于零的奇异值进行截断外，我们还可以从低秩近似的角度来考虑这个问题。在[《低秩近似之路（二）：SVD》](/archives/10407)我们证明了，一个矩阵的最优$r$秩近似就是只保留最大的$r$个奇异值的SVD结果。反过来，我们可以指定一个相对误差$\epsilon$，将有效秩定义为可以达到这个相对误差的最小秩：  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M},\epsilon) \triangleq \min\left\\{ i\,\,\left|\,\,\sqrt{\left(\sum_{i=1}^r \sigma_i^2\right)\left/\left(\sum_{i=1}^n \sigma_i^2\right.\right)} \geq 1-\epsilon\right.\right\\}\end{equation}  
这个定义的数值意义更清晰，但它只考虑了总体误差，所以有些例子反而不大优雅。比如$n\times n$的单位阵，我们期望它的有效秩总是$n$，因为每个奇异值都是等同的1，不应有任何截断行为。但采用上述定义的话，当$n$足够大（$1/n < \epsilon$）时，有效秩就会小于$n$。

## 范数之比 #

上一节定义的有效秩虽然直观，但都依赖于一个超参数$\epsilon$，终究是不够简洁。现在我们的基本认知是有效秩的概念只能依赖于相对大小，所以问题等价为从$1\geq \sigma_2/\sigma_1\geq\cdots\geq\sigma_n/\sigma_1\geq 0$中构建有效秩。由于都在$[0,1]$内的特点，一个巧妙的想法是直接将它们求和  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \sum_{i=1}^n\frac{\sigma_i}{\sigma_1}\end{equation}  
从[《低秩近似之路（二）：SVD》](/archives/10407)我们知道，最大的奇异值$\sigma_1$是矩阵的[谱范数（Spectral Norm）](https://en.wikipedia.org/wiki/Matrix_norm#Spectral_norm_\(p_=_2\))，记为$\Vert\boldsymbol{M}\Vert_2$，而所有奇异值之和实际上也是一个矩阵范数，称为“[核范数（Nuclear Norm）](https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms)”，通常记为$\Vert\boldsymbol{M}\Vert_*$，于是上式也可以简记为  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \frac{\Vert\boldsymbol{M}\Vert_*}{\Vert\boldsymbol{M}\Vert_2}\label{eq:n-2}\end{equation}  
这最早的出处可能是[《An Introduction to Matrix Concentration Inequalities》](https://papers.cool/arxiv/1501.01571)，在里边被称为“Intrinsic Dimension”，但相关性质在[《Guaranteed Minimum-Rank Solutions of Linear Matrix Equations via Nuclear Norm Minimization》](https://papers.cool/arxiv/0706.4138)已经被探讨过了。

类似地，我们也可以通过平方求和来构建有效秩：  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \sum_{i=1}^n\frac{\sigma_i^2}{\sigma_1^2} = \frac{\Vert\boldsymbol{M}\Vert_F^2}{\Vert\boldsymbol{M}\Vert_2^2}\label{eq:f-2}\end{equation}  
这里的$\Vert\cdot\Vert_F$是$F$范数。该定义的出自应该是[《Sampling from large matrices: an approach through geometric functional analysis》](https://arxiv.org/abs/math/0503442)，当时被称为“Numerical Rank”，如今更多称为“Stable Rank”，是比较流行的有效秩概念之一。

从计算复杂度来看，式$\eqref{eq:f-2}$比式$\eqref{eq:n-2}$更低。因为计算核范数需要全体奇异值，这意味着需要完整的SVD；而$\Vert\boldsymbol{M}\Vert_F^2$又等于矩阵所有元素的平方和，所以式$\eqref{eq:f-2}$主要的计算量是最大奇异值，这比计算全体奇异值成本更低。如果愿意，我们还可以将式$\eqref{eq:n-2}$、式$\eqref{eq:f-2}$推广到$k$次方求和，结果实际上是更一般的[Schatten范数](https://en.wikipedia.org/wiki/Schatten_norm)与谱范数之比。

## 分布与熵 #

读者如果直接搜索“Effective Rank”，大概率会搜到论文[《The Effective Rank: a Measure of Effective Dimensionality》](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf)，这也是比较早探讨有效秩的文献，里边提出基于熵来定义有效秩。

首先，由于奇异值总是非负的，我们可以将它归一化得到一个概率分布：  
\begin{equation}p_i = \frac{\sigma_i^{\gamma}}{\sum_{j=1}^n \sigma_j^{\gamma}}\end{equation}  
其中$\gamma > 0$，从文献看$\gamma=1$和$\gamma=2$都经常有人用（我们在[Moonlight](/archives/10739)中用了$\gamma=2$），下面我们都以$\gamma=1$为例。有了概率分布，那么就可以计算信息熵（香农熵）：  
\begin{equation}H = -\sum_{i=1}^n p_i \log p_i\end{equation}  
回忆一下，熵的值域是$[0, \log n]$，取指数后则有$e^H \in [1, n]$，当分布是One Hot时$e^H=1$（只有一个非零奇异值），当分布均匀时$e^H=n$（全体奇异值相等），这正好是标准秩的两个特殊例子，这启发我们可以将有效秩定义为  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq e^H = \exp\left(-\sum_{i=1}^n p_i \log p_i\right)\label{eq:h-erank}\end{equation}  
代入$p_i$的定义，我们发现它可以进一步变换成  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) = \exp\left(\log\sum_{i=1}^n \sigma_i -\frac{\sum_{i=1}^n \sigma_i\log\sigma_i}{\sum_{i=1}^n \sigma_i}\right)\end{equation}  
很明显括号内的第一项$\exp$后就是$\Vert\boldsymbol{M}\Vert_*$；第二项是$\log\sigma_i$的加权平均，权重是$\sigma_i$，此时$\log\sigma_i$会约等于最大的$\log\sigma_1$，取指数后即$\sigma_1=\Vert\boldsymbol{M}\Vert_2$。所以，上式总的结果将会近似于式$\eqref{eq:n-2}$，这表明了基于熵来定义有效秩看似是一条截然不同的路径，但实际上跟上一节的范数之比有异曲同工之妙。

我们知道，标准秩满足三角不等式$\mathop{\text{rank}}(\boldsymbol{A}+\boldsymbol{B})\leq \mathop{\text{rank}}(\boldsymbol{A}) + \mathop{\text{rank}}(\boldsymbol{B})$，而原论文证明了，对于（半）正定对称矩阵$\boldsymbol{A},\boldsymbol{B}$，式$\eqref{eq:h-erank}$所定义的有效秩满足$\mathop{\text{erank}}(\boldsymbol{A}+\boldsymbol{B})\leq \mathop{\text{erank}}(\boldsymbol{A}) + \mathop{\text{erank}}(\boldsymbol{B})$，尚不清楚这个不等式是否可以推广到一般矩阵。目前看来，证明有效秩能否保持标准秩的一些不等式，是一件并不容易的事情。

## 稀疏指标 #

从上述一系列有效秩的定义中，尤其是从奇异值到分布再到熵的转变，相信已经有读者隐约意识到，有效秩跟稀疏性有明显的共性，实际上有效秩可以理解为奇异值向量的一个稀疏性度量，跟一般的稀疏性指标不同的是，我们会将值域对齐到$1\leq \mathop{\text{erank}}(\boldsymbol{M}) \leq \mathop{\text{rank}}(\boldsymbol{M}) \leq n$，使其跟秩的概念对齐，从而更直观地感知稀疏程度。

关于稀疏性度量，我们曾在[《如何度量数据的稀疏程度？》](/archives/9595)有过比较系统的讨论，理论上那里边的结果都可以用来构造有效秩。事实上我们也是这样做的，“范数之比”一节中我们基于Schatten范数与谱范数之比来构建有效值，其实只相当于那篇文章的公式$(1)$，我们还可以其他公式如$(16)$，它相当于将有效秩定义为核范数和$F$范数之比的平方：  
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \frac{\Vert\boldsymbol{M}\Vert_*^2}{\Vert\boldsymbol{M}\Vert_F^2} = \frac{(\sum_{i=1}^n\sigma_i)^2}{\sum_{i=1}^n\sigma_i^2}\end{equation}  
这同样满足$1\leq \mathop{\text{erank}}(\boldsymbol{M}) \leq \mathop{\text{rank}}(\boldsymbol{M})$，是一个可用的有效秩定义。

非常神奇，我们关于有效秩和稀疏性的认知，在无形之中达成了闭环。不得不说这是一种很美妙的体验：笔者开始学习稀疏性度量时对有效秩是一无所知的，而这几天在学习有效秩时，慢慢意识到它跟稀疏性本质是相通的，冥冥之中似乎有一种神秘力量，将我们在不同领域、不同学科中积累的知识悄然串联起来，最终汇聚在同一个正确的方向上。

## 文章小结 #

本文探讨了矩阵的有效秩（Effective Rank）概念，它是线性代数中矩阵的秩（Rank）概念在数值计算方面的延伸，能够更有效地度量矩阵的本质维度。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10847>_

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

苏剑林. (Apr. 10, 2025). 《矩阵的有效秩（Effective Rank） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10847>

@online{kexuefm-10847,  
title={矩阵的有效秩（Effective Rank）},  
author={苏剑林},  
year={2025},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/10847}},  
} 


---

## 公式推导与注释

在本节中，我们将对有效秩的各种定义进行深入的数学推导，探讨它们之间的内在联系，以及在信息论、矩阵理论和机器学习中的应用。

### 一、有效秩的基础定义与性质

#### 1.1 奇异值分解的回顾

对于矩阵 $\boldsymbol{M}\in\mathbb{R}^{n\times m}$，设 $n\leq m$，其奇异值分解（SVD）为：
$$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T = \sum_{i=1}^{r} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^T$$

其中：
- $r = \mathop{\text{rank}}(\boldsymbol{M})$ 是矩阵的代数秩
- $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 是非零奇异值
- $\boldsymbol{u}_i \in \mathbb{R}^n$ 和 $\boldsymbol{v}_i \in \mathbb{R}^m$ 分别是左、右奇异向量

代数秩的定义为：
$$\mathop{\text{rank}}(\boldsymbol{M}) = |\{i : \sigma_i > 0\}| = r$$

然而，在数值计算中，判断 $\sigma_i = 0$ 是不稳定的。例如，考虑奇异值序列 $\{\sigma_i\} = \{1, 10^{-8}, 10^{-16}\}$，虽然代数秩为3，但后两个奇异值在数值上可忽略。

#### 1.2 范数的定义与关系

在探讨有效秩之前，我们先回顾几个重要的矩阵范数：

**谱范数（Spectral Norm）**：
$$\|\boldsymbol{M}\|_2 = \sigma_1 = \max_{\|\boldsymbol{x}\|=1} \|\boldsymbol{M}\boldsymbol{x}\|_2$$

**Frobenius范数**：
$$\|\boldsymbol{M}\|_F = \sqrt{\sum_{i=1}^{r} \sigma_i^2} = \sqrt{\mathop{\text{tr}}(\boldsymbol{M}^T\boldsymbol{M})}$$

**核范数（Nuclear Norm）**：
$$\|\boldsymbol{M}\|_* = \sum_{i=1}^{r} \sigma_i = \mathop{\text{tr}}(\sqrt{\boldsymbol{M}^T\boldsymbol{M}})$$

**Schatten $p$-范数**（一般形式）：
$$\|\boldsymbol{M}\|_p = \left(\sum_{i=1}^{r} \sigma_i^p\right)^{1/p}, \quad p \geq 1$$

显然，谱范数、Frobenius范数和核范数分别是 Schatten $\infty$-范数、2-范数和1-范数。

这些范数满足如下不等式链：
$$\|\boldsymbol{M}\|_2 \leq \|\boldsymbol{M}\|_F \leq \|\boldsymbol{M}\|_* \leq \sqrt{r} \|\boldsymbol{M}\|_F \leq r \|\boldsymbol{M}\|_2$$

**推导**：
- $\|\boldsymbol{M}\|_2 = \sigma_1 \leq \sqrt{\sum_{i=1}^r \sigma_i^2} = \|\boldsymbol{M}\|_F$（显然）
- $\|\boldsymbol{M}\|_F^2 = \sum_{i=1}^r \sigma_i^2 \leq \left(\sum_{i=1}^r \sigma_i\right)^2 = \|\boldsymbol{M}\|_*^2$（当且仅当 $r=1$ 时等号成立）
- $\|\boldsymbol{M}\|_* = \sum_{i=1}^r \sigma_i \leq \sqrt{r} \sqrt{\sum_{i=1}^r \sigma_i^2} = \sqrt{r} \|\boldsymbol{M}\|_F$（Cauchy-Schwarz不等式）
- $\|\boldsymbol{M}\|_* = \sum_{i=1}^r \sigma_i \leq r \sigma_1 = r \|\boldsymbol{M}\|_2$

### 二、基于范数比的有效秩

#### 2.1 稳定秩（Stable Rank）

**定义**：
$$\mathop{\text{srank}}(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_F^2}{\|\boldsymbol{M}\|_2^2} = \frac{\sum_{i=1}^r \sigma_i^2}{\sigma_1^2}$$

**性质分析**：

(1) **值域**：$1 \leq \mathop{\text{srank}}(\boldsymbol{M}) \leq r$

**证明**：
- 下界：$\mathop{\text{srank}}(\boldsymbol{M}) = \frac{\sum_{i=1}^r \sigma_i^2}{\sigma_1^2} \geq \frac{\sigma_1^2}{\sigma_1^2} = 1$
- 上界：$\mathop{\text{srank}}(\boldsymbol{M}) = \frac{\sum_{i=1}^r \sigma_i^2}{\sigma_1^2} \leq \frac{r\sigma_1^2}{\sigma_1^2} = r$（当所有奇异值相等时取等号）

(2) **极端情况**：
- 当 $\boldsymbol{M}$ 秩1时（$\sigma_2 = \cdots = \sigma_r = 0$）：$\mathop{\text{srank}}(\boldsymbol{M}) = 1$
- 当所有非零奇异值相等时（$\sigma_1 = \sigma_2 = \cdots = \sigma_r$）：$\mathop{\text{srank}}(\boldsymbol{M}) = r$

(3) **归一化不变性**：对于任意 $c > 0$，
$$\mathop{\text{srank}}(c\boldsymbol{M}) = \frac{\|c\boldsymbol{M}\|_F^2}{\|c\boldsymbol{M}\|_2^2} = \frac{c^2\|\boldsymbol{M}\|_F^2}{c^2\|\boldsymbol{M}\|_2^2} = \mathop{\text{srank}}(\boldsymbol{M})$$

#### 2.2 本征维度（Intrinsic Dimension）

**定义**：
$$\mathop{\text{irank}}(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_*}{\|\boldsymbol{M}\|_2} = \frac{\sum_{i=1}^r \sigma_i}{\sigma_1}$$

**性质分析**：

(1) **值域**：$1 \leq \mathop{\text{irank}}(\boldsymbol{M}) \leq r$

**证明**：类似稳定秩，利用：
$$\sum_{i=1}^r \sigma_i \geq \sigma_1 \quad \text{且} \quad \sum_{i=1}^r \sigma_i \leq r\sigma_1$$

(2) **与稳定秩的关系**：

由Cauchy-Schwarz不等式：
$$\left(\sum_{i=1}^r \sigma_i\right)^2 \leq r \sum_{i=1}^r \sigma_i^2$$

因此：
$$\mathop{\text{irank}}(\boldsymbol{M})^2 = \frac{(\sum_{i=1}^r \sigma_i)^2}{\sigma_1^2} \leq r \cdot \frac{\sum_{i=1}^r \sigma_i^2}{\sigma_1^2} = r \cdot \mathop{\text{srank}}(\boldsymbol{M})$$

这说明本征维度通常小于或等于稳定秩（归一化后）。

#### 2.3 一般Schatten比

更一般地，我们可以定义：
$$\mathop{\text{erank}}_p(\boldsymbol{M}) = \left(\frac{\|\boldsymbol{M}\|_p}{\|\boldsymbol{M}\|_2}\right)^p = \frac{\sum_{i=1}^r \sigma_i^p}{\sigma_1^p}$$

特例：
- $p=1$：本征维度 $\mathop{\text{irank}}(\boldsymbol{M})$
- $p=2$：稳定秩 $\mathop{\text{srank}}(\boldsymbol{M})$
- $p\to\infty$：趋向于1

### 三、基于信息熵的有效秩

#### 3.1 归一化奇异值分布

为了应用信息论的工具，我们首先将奇异值归一化为概率分布。考虑参数 $\gamma > 0$，定义：
$$p_i = \frac{\sigma_i^\gamma}{\sum_{j=1}^r \sigma_j^\gamma} = \frac{\sigma_i^\gamma}{Z_\gamma}$$

其中配分函数：
$$Z_\gamma = \sum_{j=1}^r \sigma_j^\gamma = \|\boldsymbol{M}\|_\gamma^\gamma$$

显然 $p_i \geq 0$ 且 $\sum_{i=1}^r p_i = 1$，构成了一个离散概率分布。

**参数 $\gamma$ 的选择**：
- $\gamma = 1$：线性归一化，对应核范数
- $\gamma = 2$：二次归一化，对应Frobenius范数
- $\gamma > 1$：增强大奇异值的权重
- $\gamma < 1$：增强小奇异值的权重

#### 3.2 Shannon熵及其有效秩

**Shannon熵**定义为：
$$H(\boldsymbol{M}) = -\sum_{i=1}^r p_i \log p_i$$

其中对数底通常取 $e$ 或 2（本文统一使用自然对数）。

**熵的值域**：$0 \leq H(\boldsymbol{M}) \leq \log r$

- 最小值0：当某个 $p_i = 1$，其余为0（秩1矩阵）
- 最大值 $\log r$：当 $p_1 = p_2 = \cdots = p_r = 1/r$（均匀分布）

**基于熵的有效秩**定义为：
$$\mathop{\text{erank}}_H(\boldsymbol{M}) = e^{H(\boldsymbol{M})} = \exp\left(-\sum_{i=1}^r p_i \log p_i\right)$$

这样定义的有效秩满足 $1 \leq \mathop{\text{erank}}_H(\boldsymbol{M}) \leq r$。

#### 3.3 熵与范数的联系

现在我们证明熵定义的有效秩与范数比定义的有效秩之间的深刻联系。取 $\gamma = 1$，则：
$$p_i = \frac{\sigma_i}{\sum_{j=1}^r \sigma_j}$$

代入Shannon熵：
$$H = -\sum_{i=1}^r \frac{\sigma_i}{\sum_{j=1}^r \sigma_j} \log \frac{\sigma_i}{\sum_{j=1}^r \sigma_j}$$

展开对数：
$$H = -\sum_{i=1}^r \frac{\sigma_i}{\sum_{j=1}^r \sigma_j} \left(\log \sigma_i - \log \sum_{j=1}^r \sigma_j\right)$$

分离两项：
$$H = -\sum_{i=1}^r \frac{\sigma_i}{\sum_{j=1}^r \sigma_j} \log \sigma_i + \log \sum_{j=1}^r \sigma_j$$

记 $S = \sum_{j=1}^r \sigma_j = \|\boldsymbol{M}\|_*$，则：
$$H = \log S - \frac{1}{S}\sum_{i=1}^r \sigma_i \log \sigma_i$$

因此：
$$e^H = S \cdot \exp\left(-\frac{1}{S}\sum_{i=1}^r \sigma_i \log \sigma_i\right) = \|\boldsymbol{M}\|_* \cdot \exp\left(-\frac{1}{\|\boldsymbol{M}\|_*}\sum_{i=1}^r \sigma_i \log \sigma_i\right)$$

**关键观察**：第二项是 $\log \sigma_i$ 的加权几何平均的指数。设：
$$\bar{\sigma}_{\text{geom}} = \exp\left(\frac{\sum_{i=1}^r \sigma_i \log \sigma_i}{\sum_{i=1}^r \sigma_i}\right)$$

这是以 $\sigma_i$ 为权重的几何平均。那么：
$$\mathop{\text{erank}}_H(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_*}{\bar{\sigma}_{\text{geom}}}$$

**与本征维度的近似**：

当奇异值衰减较快时，主导奇异值 $\sigma_1$ 远大于其他奇异值，此时：
$$\bar{\sigma}_{\text{geom}} \approx \sigma_1 = \|\boldsymbol{M}\|_2$$

因此：
$$\mathop{\text{erank}}_H(\boldsymbol{M}) \approx \frac{\|\boldsymbol{M}\|_*}{\|\boldsymbol{M}\|_2} = \mathop{\text{irank}}(\boldsymbol{M})$$

这揭示了**熵定义和范数比定义的内在统一性**。

#### 3.4 详细的近似分析

为了更精确地理解这个近似，我们进行Taylor展开分析。设：
$$\sigma_i = \sigma_1 \cdot t_i, \quad 1 = t_1 \geq t_2 \geq \cdots \geq t_r \geq 0$$

则：
$$\log \sigma_i = \log \sigma_1 + \log t_i$$

代入加权几何平均：
$$\bar{\sigma}_{\text{geom}} = \exp\left(\frac{\sum_{i=1}^r \sigma_i (\log \sigma_1 + \log t_i)}{\sum_{i=1}^r \sigma_i}\right) = \sigma_1 \exp\left(\frac{\sum_{i=1}^r \sigma_i \log t_i}{\sum_{i=1}^r \sigma_i}\right)$$

记修正因子：
$$\delta = \exp\left(\frac{\sum_{i=1}^r \sigma_i \log t_i}{\sum_{i=1}^r \sigma_i}\right) = \exp\left(\frac{\sigma_1 \sum_{i=1}^r t_i \log t_i}{\sigma_1 \sum_{i=1}^r t_i}\right) = \exp\left(\frac{\sum_{i=1}^r t_i \log t_i}{\sum_{i=1}^r t_i}\right)$$

则 $\bar{\sigma}_{\text{geom}} = \delta \cdot \sigma_1$，所以：
$$\mathop{\text{erank}}_H(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_*}{\delta \cdot \|\boldsymbol{M}\|_2} = \frac{1}{\delta} \mathop{\text{irank}}(\boldsymbol{M})$$

**修正因子的性质**：
- 当所有 $t_i = 1$ 时（均匀分布），$\delta = 1$
- 当 $t_2, \ldots, t_r \ll 1$ 时（快速衰减），$\delta < 1$
- 总有 $0 < \delta \leq 1$（因为 $\log t_i \leq 0$）

因此 $\mathop{\text{erank}}_H(\boldsymbol{M}) \geq \mathop{\text{irank}}(\boldsymbol{M})$，熵定义的有效秩通常略大于本征维度。

### 四、Rényi熵与Tsallis熵的推广

#### 4.1 Rényi熵

Rényi熵是Shannon熵的参数化推广，定义为：
$$H_\alpha(\boldsymbol{M}) = \frac{1}{1-\alpha} \log \sum_{i=1}^r p_i^\alpha, \quad \alpha > 0, \alpha \neq 1$$

当 $\alpha \to 1$ 时，Rényi熵收敛到Shannon熵：
$$\lim_{\alpha \to 1} H_\alpha(\boldsymbol{M}) = H(\boldsymbol{M})$$

**证明**（L'Hôpital法则）：
$$\lim_{\alpha \to 1} H_\alpha = \lim_{\alpha \to 1} \frac{\log \sum_{i=1}^r p_i^\alpha}{1-\alpha}$$

分子分母求导：
$$= \lim_{\alpha \to 1} \frac{\frac{d}{d\alpha}\log \sum_{i=1}^r p_i^\alpha}{\frac{d}{d\alpha}(1-\alpha)} = \lim_{\alpha \to 1} \frac{\sum_{i=1}^r p_i^\alpha \log p_i}{\sum_{i=1}^r p_i^\alpha} \cdot (-1)$$

当 $\alpha = 1$：
$$= -\sum_{i=1}^r p_i \log p_i = H(\boldsymbol{M})$$

**基于Rényi熵的有效秩**：
$$\mathop{\text{erank}}_{\alpha}(\boldsymbol{M}) = e^{H_\alpha(\boldsymbol{M})} = \left(\sum_{i=1}^r p_i^\alpha\right)^{\frac{1}{1-\alpha}}$$

取 $\gamma = 1$，$p_i = \sigma_i / \|\boldsymbol{M}\|_*$：
$$\mathop{\text{erank}}_{\alpha}(\boldsymbol{M}) = \left(\sum_{i=1}^r \left(\frac{\sigma_i}{\|\boldsymbol{M}\|_*}\right)^\alpha\right)^{\frac{1}{1-\alpha}} = \frac{\|\boldsymbol{M}\|_*}{\left(\sum_{i=1}^r \sigma_i^\alpha\right)^{\frac{1}{1-\alpha}}} \cdot \left(\|\boldsymbol{M}\|_*\right)^{\frac{\alpha-1}{1-\alpha}}$$

简化：
$$\mathop{\text{erank}}_{\alpha}(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_*^{1/(\alpha)}}{(\sum_{i=1}^r \sigma_i^\alpha)^{1/\alpha}}$$

等等，让我重新计算。设 $p_i = \sigma_i / S$，其中 $S = \|\boldsymbol{M}\|_*$，则：
$$\sum_{i=1}^r p_i^\alpha = \sum_{i=1}^r \frac{\sigma_i^\alpha}{S^\alpha} = \frac{1}{S^\alpha} \sum_{i=1}^r \sigma_i^\alpha$$

因此：
$$\mathop{\text{erank}}_{\alpha}(\boldsymbol{M}) = \left(\frac{1}{S^\alpha} \sum_{i=1}^r \sigma_i^\alpha\right)^{\frac{1}{1-\alpha}} = S^{\frac{\alpha}{1-\alpha}} \left(\sum_{i=1}^r \sigma_i^\alpha\right)^{\frac{1}{1-\alpha}}$$

**特殊情况**：

- $\alpha = 0$：$H_0 = \log r$，$\mathop{\text{erank}}_0 = r$（代数秩）
- $\alpha = 2$：
$$\mathop{\text{erank}}_2(\boldsymbol{M}) = \left(\sum_{i=1}^r p_i^2\right)^{-1} = \frac{1}{\sum_{i=1}^r \left(\frac{\sigma_i}{\|\boldsymbol{M}\|_*}\right)^2} = \frac{\|\boldsymbol{M}\|_*^2}{\sum_{i=1}^r \sigma_i^2} = \frac{\|\boldsymbol{M}\|_*^2}{\|\boldsymbol{M}\|_F^2}$$

这正是我们在"稀疏指标"一节中提到的另一种有效秩定义！

- $\alpha \to \infty$：$H_\infty = -\log \max_i p_i = -\log \frac{\sigma_1}{\|\boldsymbol{M}\|_*}$，
$$\mathop{\text{erank}}_\infty(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_*}{\sigma_1} = \mathop{\text{irank}}(\boldsymbol{M})$$

#### 4.2 Tsallis熵

Tsallis熵是另一种熵的推广，定义为：
$$S_q(\boldsymbol{M}) = \frac{1}{q-1} \left(1 - \sum_{i=1}^r p_i^q\right), \quad q > 0, q \neq 1$$

当 $q \to 1$ 时，Tsallis熵也收敛到Shannon熵。

**基于Tsallis熵的有效秩**：类似地可以定义为：
$$\mathop{\text{erank}}_q(\boldsymbol{M}) = \left(\sum_{i=1}^r p_i^q\right)^{\frac{1}{1-q}}$$

注意到这与Rényi熵的有效秩形式完全相同（只是参数记号不同）。

### 五、信息论解释

#### 5.1 有效秩作为等效均匀分布的大小

Shannon熵的一个重要性质是，它度量了分布的"等效均匀程度"。具体地，若 $H(p) = \log k$，则分布 $p$ 携带的信息量等价于一个大小为 $k$ 的均匀分布。

对于有效秩 $\mathop{\text{erank}}_H(\boldsymbol{M}) = e^H$，我们可以理解为：
> 矩阵 $\boldsymbol{M}$ 的奇异值分布所携带的信息，等价于一个大小为 $\mathop{\text{erank}}_H(\boldsymbol{M})$ 的均匀分布。

换句话说，虽然矩阵的代数秩可能是 $r$，但由于奇异值分布的不均匀性，其"有效维度"只有 $e^H \leq r$。

#### 5.2 与主成分分析（PCA）的联系

在PCA中，我们通常保留前 $k$ 个主成分使得：
$$\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^r \lambda_i} \geq 1 - \epsilon$$

其中 $\lambda_i = \sigma_i^2$ 是特征值。这等价于：
$$\frac{\sum_{i=1}^k \sigma_i^2}{\|\boldsymbol{M}\|_F^2} \geq 1 - \epsilon$$

而稳定秩 $\mathop{\text{srank}}(\boldsymbol{M}) = \|\boldsymbol{M}\|_F^2 / \sigma_1^2$ 给出了一个"自动"的截断指标。

**解释**：如果稳定秩为 $k$，则意味着矩阵的能量主要集中在前 $k$ 个奇异值上。

#### 5.3 最小描述长度（MDL）原理

从信息编码的角度，Shannon熵 $H$ 给出了对奇异值分布进行最优编码所需的平均比特数（以 $\log_2$ 为底）。有效秩 $e^H$ 则对应于"等效编码空间大小"。

若我们需要传输矩阵 $\boldsymbol{M}$ 的低秩近似，选择秩 $k = \lceil \mathop{\text{erank}}_H(\boldsymbol{M}) \rceil$ 可以在信息损失和压缩率之间取得平衡。

### 六、矩阵稀疏性的度量

#### 6.1 奇异值的Gini系数

Gini系数是经济学中度量不平等性的指标，也可用于度量奇异值的集中程度。定义：
$$G(\boldsymbol{M}) = \frac{\sum_{i=1}^r \sum_{j=1}^r |\sigma_i - \sigma_j|}{2r \sum_{i=1}^r \sigma_i} = \frac{\sum_{i=1}^r (2i - r - 1) \sigma_i}{r \sum_{i=1}^r \sigma_i}$$

其中第二个等式利用了奇异值的降序排列。

**性质**：
- $0 \leq G(\boldsymbol{M}) \leq 1 - 1/r$
- $G = 0$：所有奇异值相等（最"均匀"）
- $G \to 1$：奇异值高度集中（最"稀疏"）

**与有效秩的关系**：

高Gini系数对应低有效秩（奇异值集中在少数几个上），低Gini系数对应高有效秩（奇异值分布均匀）。可以证明：
$$\mathop{\text{erank}}(\boldsymbol{M}) \approx r(1 - G(\boldsymbol{M}))$$

#### 6.2 稀疏性度量的统一框架

一般地，奇异值的稀疏性可以通过以下形式度量：
$$\mathop{\text{sparsity}}(\boldsymbol{M}) = 1 - \frac{\mathop{\text{erank}}(\boldsymbol{M})}{r}$$

这给出了一个归一化的稀疏度指标，值域为 $[0, 1-1/r]$。

### 七、与条件数的关系

#### 7.1 条件数的定义

矩阵的条件数定义为：
$$\kappa(\boldsymbol{M}) = \frac{\sigma_1}{\sigma_r} = \frac{\|\boldsymbol{M}\|_2}{\|\boldsymbol{M}^{-1}\|_2^{-1}}$$

（对于非满秩矩阵，$\sigma_r$ 取最小的非零奇异值）

条件数度量了矩阵的"病态程度"：
- $\kappa = 1$：正交矩阵（或其倍数），最"健康"
- $\kappa \gg 1$：病态矩阵，数值不稳定

#### 7.2 条件数与有效秩的权衡

考虑一个简化模型：设矩阵有 $k$ 个奇异值等于 $\sigma$，其余 $r-k$ 个等于 $\epsilon \sigma$（$\epsilon \ll 1$）。则：
$$\mathop{\text{srank}}(\boldsymbol{M}) = \frac{k\sigma^2 + (r-k)\epsilon^2\sigma^2}{\sigma^2} = k + (r-k)\epsilon^2 \approx k$$

$$\kappa(\boldsymbol{M}) = \frac{\sigma}{\epsilon\sigma} = \frac{1}{\epsilon}$$

可见：
- 高有效秩（$k \approx r$）+ 低条件数（$\epsilon \approx 1$）：理想情况
- 低有效秩（$k \ll r$）+ 高条件数（$\epsilon \ll 1$）：典型的低秩+病态矩阵

**一般关系**：

对于稳定秩，我们有：
$$\mathop{\text{srank}}(\boldsymbol{M}) \cdot \kappa(\boldsymbol{M})^{-2} = \frac{\sum_{i=1}^r \sigma_i^2}{\sigma_1^2} \cdot \frac{\sigma_r^2}{\sigma_1^2} = \frac{\sigma_r^2 \sum_{i=1}^r \sigma_i^2}{\sigma_1^4}$$

这表明稳定秩和条件数的倒数平方成正比关系（在固定其他奇异值时）。

#### 7.3 数值稳定性分析

在求解线性系统 $\boldsymbol{M}\boldsymbol{x} = \boldsymbol{b}$ 时，相对误差界为：
$$\frac{\|\Delta \boldsymbol{x}\|}{\|\boldsymbol{x}\|} \leq \kappa(\boldsymbol{M}) \frac{\|\Delta \boldsymbol{b}\|}{\|\boldsymbol{b}\|}$$

如果 $\mathop{\text{srank}}(\boldsymbol{M}) \ll r$，我们可以用稳定秩对应的低秩近似 $\tilde{\boldsymbol{M}}$ 来替代 $\boldsymbol{M}$，从而：
- 减少计算复杂度：从 $O(r^3)$ 到 $O(k^3)$，其中 $k \approx \mathop{\text{srank}}(\boldsymbol{M})$
- 改善数值稳定性：$\kappa(\tilde{\boldsymbol{M}}) < \kappa(\boldsymbol{M})$（因为去除了小奇异值）

### 八、在低秩近似中的应用

#### 8.1 最优低秩近似的误差界

Eckart-Young-Mirsky定理指出，矩阵 $\boldsymbol{M}$ 的最优秩 $k$ 近似（在Frobenius范数下）为：
$$\boldsymbol{M}_k = \sum_{i=1}^k \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^T$$

其误差为：
$$\|\boldsymbol{M} - \boldsymbol{M}_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$

**相对误差**：
$$\frac{\|\boldsymbol{M} - \boldsymbol{M}_k\|_F^2}{\|\boldsymbol{M}\|_F^2} = \frac{\sum_{i=k+1}^r \sigma_i^2}{\sum_{i=1}^r \sigma_i^2}$$

若取 $k = \lfloor \mathop{\text{srank}}(\boldsymbol{M}) \rfloor$，则：
$$\sum_{i=1}^k \sigma_i^2 \approx \mathop{\text{srank}}(\boldsymbol{M}) \cdot \sigma_1^2$$

**推导**：设 $k = \mathop{\text{srank}}(\boldsymbol{M})$，即 $\sum_{i=1}^r \sigma_i^2 = k \sigma_1^2$。假设前 $k$ 个奇异值相对均匀，则：
$$\sum_{i=1}^k \sigma_i^2 \approx k \cdot \bar{\sigma}^2, \quad \sum_{i=k+1}^r \sigma_i^2 \approx (k - k) \sigma_1^2 = 0$$

因此相对误差较小。

#### 8.2 随机化低秩近似

在大规模矩阵计算中，完整的SVD代价高昂。随机化算法利用有效秩来指导采样：

**算法框架**：
1. 估计有效秩 $k \approx \mathop{\text{erank}}(\boldsymbol{M})$（通过快速估计 $\|\boldsymbol{M}\|_F$ 和 $\|\boldsymbol{M}\|_2$）
2. 生成随机矩阵 $\boldsymbol{\Omega} \in \mathbb{R}^{m \times (k+p)}$（$p$ 是过采样参数）
3. 计算 $\boldsymbol{Y} = \boldsymbol{M}\boldsymbol{\Omega}$
4. 正交化得到 $\boldsymbol{Q}$，使得 $\boldsymbol{M} \approx \boldsymbol{Q}\boldsymbol{Q}^T\boldsymbol{M}$

**误差分析**：

期望误差界为：
$$\mathbb{E}\left[\|\boldsymbol{M} - \boldsymbol{Q}\boldsymbol{Q}^T\boldsymbol{M}\|_F\right] \leq \left(1 + \frac{k}{p-1}\right)^{1/2} \sigma_{k+1}$$

若 $k = \mathop{\text{srank}}(\boldsymbol{M})$，则 $\sigma_{k+1}$ 通常很小，保证了好的近似质量。

### 九、在神经网络中的解释性

#### 9.1 权重矩阵的有效秩

在深度神经网络中，权重矩阵 $\boldsymbol{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 的有效秩反映了其"表达容量"。

**观察**：
- 训练初期：$\mathop{\text{erank}}(\boldsymbol{W}) \approx \min(d_{\text{out}}, d_{\text{in}})$（接近满秩）
- 训练收敛：$\mathop{\text{erank}}(\boldsymbol{W}) \ll \min(d_{\text{out}}, d_{\text{in}})$（显著降秩）

这种"隐式正则化"现象表明，梯度下降倾向于找到低有效秩的解。

#### 9.2 注意力矩阵的有效秩

在Transformer中，注意力矩阵 $\boldsymbol{A} = \text{softmax}(\boldsymbol{Q}\boldsymbol{K}^T/\sqrt{d})$ 的有效秩度量了"注意力多样性"：

- 高有效秩：注意力分散，模型考虑了更多的上下文信息
- 低有效秩：注意力集中，模型只关注少数关键位置

**计算示例**：

设序列长度为 $n$，若注意力均匀分布（$A_{ij} = 1/n$），则：
$$\mathop{\text{srank}}(\boldsymbol{A}) = \frac{\sum_{i,j} A_{ij}^2}{\max_{i,j} A_{ij}^2} = \frac{n \cdot n \cdot (1/n)^2}{(1/n)^2} = n$$

若注意力只集中在一个位置（$A_{i,j_0} = 1$，其余为0），则：
$$\mathop{\text{srank}}(\boldsymbol{A}) = 1$$

**实践意义**：监控注意力矩阵的有效秩可以帮助诊断模型是否过拟合或欠拟合。

#### 9.3 梯度协方差矩阵

在优化过程中，梯度的协方差矩阵 $\boldsymbol{C} = \mathbb{E}[\nabla L \nabla L^T]$ 的有效秩反映了"有效梯度维度"：

$$\mathop{\text{erank}}(\boldsymbol{C}) \ll d \implies \text{梯度位于低维子空间}$$

这启发了诸如低秩自适应（LoRA）等参数高效微调方法。

### 十、有效秩的梯度与优化

#### 10.1 稳定秩的梯度

考虑将稳定秩作为正则化项：
$$\mathcal{R}(\boldsymbol{M}) = \mathop{\text{srank}}(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_F^2}{\|\boldsymbol{M}\|_2^2}$$

要计算其梯度，我们需要分别对分子和分母求导。

**Frobenius范数的梯度**：
$$\frac{\partial \|\boldsymbol{M}\|_F^2}{\partial \boldsymbol{M}} = 2\boldsymbol{M}$$

**谱范数的梯度**：

设 $\sigma_1$ 对应的左右奇异向量为 $\boldsymbol{u}_1, \boldsymbol{v}_1$，则：
$$\frac{\partial \|\boldsymbol{M}\|_2}{\partial \boldsymbol{M}} = \boldsymbol{u}_1 \boldsymbol{v}_1^T$$

因此：
$$\frac{\partial \|\boldsymbol{M}\|_2^2}{\partial \boldsymbol{M}} = 2\sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^T$$

**稳定秩的梯度**（商法则）：
$$\frac{\partial \mathcal{R}}{\partial \boldsymbol{M}} = \frac{2\boldsymbol{M} \cdot \|\boldsymbol{M}\|_2^2 - \|\boldsymbol{M}\|_F^2 \cdot 2\sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^T}{\|\boldsymbol{M}\|_2^4}$$

简化：
$$\frac{\partial \mathcal{R}}{\partial \boldsymbol{M}} = \frac{2}{\|\boldsymbol{M}\|_2^2} \left(\boldsymbol{M} - \mathop{\text{srank}}(\boldsymbol{M}) \cdot \sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^T\right)$$

**解释**：梯度包含两项：
1. $\boldsymbol{M}$：增加所有奇异值
2. $-\mathop{\text{srank}}(\boldsymbol{M}) \cdot \sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^T$：减少最大奇异值

这促使奇异值分布更均匀，从而增大有效秩。

#### 10.2 核范数的梯度（参考）

核范数常用于低秩正则化，其梯度为：
$$\frac{\partial \|\boldsymbol{M}\|_*}{\partial \boldsymbol{M}} = \boldsymbol{U}\boldsymbol{V}^T$$

其中 $\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T$ 是SVD分解。

**证明**：利用 $\|\boldsymbol{M}\|_* = \mathop{\text{tr}}(\sqrt{\boldsymbol{M}^T\boldsymbol{M}})$，通过矩阵微分计算得到。

#### 10.3 最大化有效秩的优化问题

在某些应用中（如正则化、多样性增强），我们希望最大化有效秩：
$$\max_{\boldsymbol{M}} \mathop{\text{srank}}(\boldsymbol{M}) \quad \text{s.t.} \quad \|\boldsymbol{M}\|_F^2 = C$$

其中 $C$ 是常数约束。

**解**：由于 $\mathop{\text{srank}}(\boldsymbol{M}) = \|\boldsymbol{M}\|_F^2 / \sigma_1^2$，在 $\|\boldsymbol{M}\|_F^2$ 固定时，最大化有效秩等价于最小化 $\sigma_1^2$，即：
$$\min_{\boldsymbol{M}} \sigma_1^2 \quad \text{s.t.} \quad \sum_{i=1}^r \sigma_i^2 = C$$

**最优解**：所有奇异值相等，$\sigma_1 = \sigma_2 = \cdots = \sigma_r = \sqrt{C/r}$，此时 $\mathop{\text{srank}}(\boldsymbol{M}) = r$（最大值）。

#### 10.4 有效秩作为正则化项

在机器学习中，我们可以在损失函数中添加有效秩的惩罚：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \cdot f(\mathop{\text{erank}}(\boldsymbol{W}))$$

其中 $f$ 是单调函数：
- $f(x) = -x$：鼓励高有效秩（多样性）
- $f(x) = x$：鼓励低有效秩（稀疏性）

**梯度下降更新**：
$$\boldsymbol{W}^{(t+1)} = \boldsymbol{W}^{(t)} - \eta \left(\frac{\partial \mathcal{L}_{\text{task}}}{\partial \boldsymbol{W}} + \lambda f'(\mathop{\text{erank}}(\boldsymbol{W}^{(t)})) \frac{\partial \mathop{\text{erank}}}{\partial \boldsymbol{W}}\right)$$

#### 10.5 Trace-Norm正则化与有效秩

核范数（Trace Norm）正则化：
$$\min_{\boldsymbol{M}} \mathcal{L}(\boldsymbol{M}) + \mu \|\boldsymbol{M}\|_*$$

等价于对所有奇异值施加 $L_1$ 惩罚，促使矩阵低秩（小有效秩）。

**与本征维度的关系**：

核范数正则化趋于减小 $\|\boldsymbol{M}\|_*$，同时在许多情况下 $\|\boldsymbol{M}\|_2$ 相对稳定，因此：
$$\mathop{\text{irank}}(\boldsymbol{M}) = \frac{\|\boldsymbol{M}\|_*}{\|\boldsymbol{M}\|_2} \downarrow$$

这解释了为何核范数正则化能有效降低模型复杂度。

### 十一、总结与进一步的推广

#### 11.1 各种有效秩定义的比较

我们总结几种主要的有效秩定义：

| 定义 | 公式 | 计算复杂度 | 性质 |
|------|------|-----------|------|
| 稳定秩 | $\|\boldsymbol{M}\|_F^2 / \|\boldsymbol{M}\|_2^2$ | 低（只需最大奇异值） | 常用，可导 |
| 本征维度 | $\|\boldsymbol{M}\|_* / \|\boldsymbol{M}\|_2$ | 高（需全部奇异值） | 理论性质好 |
| Shannon熵 | $\exp(-\sum p_i \log p_i)$ | 高（需全部奇异值） | 信息论意义 |
| Rényi熵 | $(\sum p_i^\alpha)^{1/(1-\alpha)}$ | 高 | 可调参数 |
| 核/F范数比 | $\|\boldsymbol{M}\|_*^2 / \|\boldsymbol{M}\|_F^2$ | 高 | 稀疏性度量 |

#### 11.2 开放问题

尽管有效秩已有广泛应用，仍有许多理论问题待解决：

1. **三角不等式的推广**：哪些有效秩定义满足 $\mathop{\text{erank}}(\boldsymbol{A}+\boldsymbol{B}) \leq \mathop{\text{erank}}(\boldsymbol{A}) + \mathop{\text{erank}}(\boldsymbol{B})$？

2. **乘积的有效秩**：$\mathop{\text{erank}}(\boldsymbol{AB})$ 与 $\mathop{\text{erank}}(\boldsymbol{A}), \mathop{\text{erank}}(\boldsymbol{B})$ 的关系？

3. **张量的有效秩**：如何推广到高阶张量？

4. **计算优化**：能否不计算全部奇异值而准确估计熵定义的有效秩？

#### 11.3 应用展望

有效秩在现代机器学习中有诸多潜在应用：

- **模型压缩**：基于有效秩的自适应剪枝
- **联邦学习**：通讯效率与有效秩的权衡
- **生成模型**：扩散模型中的秩坍缩问题
- **可解释性**：通过有效秩理解模型的"内在维度"

有效秩作为连接线性代数、信息论和机器学习的桥梁，将在理论和实践中继续发挥重要作用。

