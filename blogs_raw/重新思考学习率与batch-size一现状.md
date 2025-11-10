---
title: 重新思考学习率与Batch Size（一）：现状
slug: 重新思考学习率与batch-size一现状
date: 2025-09-01
tags: 梯度, 学习率, 优化器, 尺度定律, 生成模型
status: pending
---

# 重新思考学习率与Batch Size（一）：现状

**原文链接**: [https://spaces.ac.cn/archives/11260](https://spaces.ac.cn/archives/11260)

**发布日期**: 

---

在之前的文章[《当Batch Size增大时，学习率该如何随之变化？》](/archives/10542)和[《Adam的epsilon如何影响学习率的Scaling Law？》](/archives/10563)中，我们从理论上讨论了学习率随Batch Size的变化规律，其中比较经典的部分是由OpenAI提出的展开到二阶的分析。然而，当我们要处理非SGD优化器时，这套分析方法的计算过程往往会相当复杂，有种无从下手的感觉。

接下来的几篇文章，笔者将重新整理和思考上述文章中的相关细节，尝试简化其中的一些推导步骤，给出一条更通用、更轻盈的推导路径，并且探讨推广到Muon优化器的可能性。

## 方法大意 #

首先回顾一下之前的分析方法。在[《当Batch Size增大时，学习率该如何随之变化？》](/archives/10542)中，我们介绍了多种分析学习率与Batch Size规律的思路，其中OpenAI在[《An Empirical Model of Large-Batch Training》](https://papers.cool/arxiv/1812.06162)提出的二阶近似分析占了主要篇幅，本文也是沿用同样的思路。

接着需要引入一些记号。设损失函数为$\mathcal{L}(\boldsymbol{w})$，$\boldsymbol{w}\in\mathbb{R}^N$是参数向量，$\boldsymbol{g}$是它的梯度。注意理想的损失函数是在全体训练样本上算的期望，但实际我们只能采样一个Batch来算，这导致梯度也带有随机性，我们将单个样本的梯度记为$\tilde{\boldsymbol{g}}$，它的均值就是$\boldsymbol{g}$，而协方差矩阵记为$\boldsymbol{\Sigma}$；当Batch Size为$B$时，梯度记为$\tilde{\boldsymbol{g}}_B$，它的均值还是$\boldsymbol{g}$，但协方差矩阵变为$\boldsymbol{\Sigma}/B$。

进一步地，设当前学习率为$\eta$，更新向量为$\tilde{\boldsymbol{\varphi}}_B$，那么更新后的损失函数将是  
\begin{equation}\begin{aligned}  
\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{\varphi}}_B) \approx&\, \mathcal{L}(\boldsymbol{w}) - \eta \tilde{\boldsymbol{\varphi}}_B^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2\tilde{\boldsymbol{\varphi}}_B^{\top}\boldsymbol{H}\tilde{\boldsymbol{\varphi}}_B \\\  
=&\, \mathcal{L}(\boldsymbol{w}) - \eta \tilde{\boldsymbol{\varphi}}_B^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2\newcommand{tr}{\mathop{\text{tr}}}\tr(\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}\boldsymbol{H})  
\end{aligned}\end{equation}  
右侧我们泰勒展开到了二阶，$\boldsymbol{H}$是Hessian矩阵，$\tr$是矩阵的迹，第二个等号用到了$\tr(\boldsymbol{A}\boldsymbol{B})=\tr(\boldsymbol{B}\boldsymbol{A})$这个恒等式。为了得到一个确定性的结果，我们对两边求期望：  
\begin{equation}\mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{\varphi}}_B)] \approx \mathcal{L}(\boldsymbol{w}) - \eta\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \tr(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})\end{equation}  
我们把右端看成是关于$\eta$的二次函数，并假设二次项系数是正的（更强的假设是$\boldsymbol{H}$矩阵是正定的），那么可以得到最小值点  
\begin{equation}\eta^* \approx \frac{\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g}}{\tr(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})}\end{equation}  
这便是 _平均来说_ 让损失函数下降最快的学习率，是学习率的理论最优解。我们要做的事情，就是针对具体的$\tilde{\boldsymbol{\varphi}}_B$算出$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$和$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$，然后从上式析出它与Batch Size（即$B$）的关系。

## 热身练习 #

作为第一个例子，我们自然是考虑最简单的SGD，此时有$\tilde{\boldsymbol{\varphi}}_B=\tilde{\boldsymbol{g}}_B$，那么简单可得$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]=\boldsymbol{g}$以及$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]=\boldsymbol{g}\boldsymbol{g}^{\top} + \boldsymbol{\Sigma}/B$，于是有  
\begin{equation}\eta^* \approx \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\tr((\boldsymbol{g}\boldsymbol{g}^{\top} + \boldsymbol{\Sigma}/B)\boldsymbol{H})} = \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g} + \tr(\boldsymbol{\Sigma}\boldsymbol{H})/B} = \frac{\eta_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}\label{eq:eta-sgd}\end{equation}  
其中  
\begin{equation}\eta_{\max} = \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}},\qquad\mathcal{B}_{\text{noise}} = \frac{\tr(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\end{equation}

对于结果$\eqref{eq:eta-sgd}$，我们可以有多种解读方式。首先，它是一个单调递增但有上界的函数，上界为$\eta_{\max}$，这表明学习率不能无限增加，相比简单的线性律或者平方根律，它更符合我们的直觉认知；当$B \ll \mathcal{B}_{\text{noise}}$时，我们有  
\begin{equation}\eta^* \approx \frac{\eta_{\max}}{1 + \mathcal{B}_{\text{noise}}/B} \approx \frac{\eta_{\max}}{\mathcal{B}_{\text{noise}}/B} = \eta_{\max} B / \mathcal{B}_{\text{noise}}\end{equation}  
这表明在Batch Size比较小时，SGD的学习率与Batch Size确实呈线性关系，同时也暗示了$\mathcal{B}_{\text{noise}}$是一个关键统计量。不过$\mathcal{B}_{\text{noise}}$的定义依赖于Hessian矩阵$\boldsymbol{H}$，这在LLM中是几乎不可能精确计算的，所以实践中我们通常假设它是单位阵（的若干倍），得到一个简化的形式  
\begin{equation}\mathcal{B}_{\text{simple}} = \frac{\tr(\boldsymbol{\Sigma})}{\boldsymbol{g}^{\top}\boldsymbol{g}}\end{equation}  
该结果具有噪音强度（$\tr(\boldsymbol{\Sigma})$)除以信号强度（$\boldsymbol{g}^{\top}\boldsymbol{g}$）的形式，它其实就是信噪比的倒数，它表明信噪比越小，那么就需要更大的Batch Size才能用上相同的$\eta_{\max}$，这也跟我们的直觉认知相符。$\tr(\boldsymbol{\Sigma})$只依赖于$\boldsymbol{\Sigma}$的对角线元素，这表明我们只需要将每个参数独立地估计均值和方差，这在实践上是可行的。

## 数据效率 #

除了学习率与Batch Size的直接关系外，笔者认为由此衍生出来的关于训练数据量和训练步数的渐近关系，也是必须要学习的精彩部分。特别地，这个结论似乎比学习率的关系式$\eqref{eq:eta-sgd}$更为通用，因为后面我们将会看到，SignSGD也能得到同样形式的结论，但它的学习率规律并不是式$\eqref{eq:eta-sgd}$。

原论文对这部分的讨论比较复杂，下面的推导是经过笔者简化的。具体来说，我们将$\eta^*$代回到$\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{g}}_B)$，将得到  
\begin{equation}\overline{\Delta\mathcal{L}} = \mathcal{L}(\boldsymbol{w}) - \mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta^*\tilde{\boldsymbol{g}}_B)] \approx \frac{\Delta\mathcal{L}_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}\end{equation}  
其中$\Delta\mathcal{L}_{\max} = \frac{(\boldsymbol{g}^{\top}\boldsymbol{g})^2}{2\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}$。怎么理解这个结果呢？首先，它是关于$B$的单调递增函数，当$B\to\infty$时等于$\Delta\mathcal{L}_{\max}$，换言之如果我们能开无穷大的Batch Size，那么每一步的损失下降量是$\Delta\mathcal{L}_{\max}$，此时所需的训练步数最少，记为$S_{\min}$。

如果Batch Size是有限值，那么每一步的损失下降量平均来说只有$\overline{\Delta\mathcal{L}}$，这意味着平均而言我们要花$1 + \mathcal{B}_{\text{noise}}/B$步，才能达到无穷大Batch Size时1步的下降量，于是为了达到相同的损失，我们要训练$S = (1 + \mathcal{B}_{\text{noise}}/B)S_{\min}$步。

由于Batch Size为$B$，所以很容易得出训练消耗的数据总量为$E = BS = (B + \mathcal{B}_{\text{noise}})S_{\min}$。从这个结果可以看出，增大Batch Size后，想要达到相同的效果，我们还需要适当增加数据量$E$；当$B\to 0$时，所需要的数据量最少，为$E_{\min} = \mathcal{B}_{\text{noise}}S_{\min}$。利用这些记号，我们可以写出  
\begin{equation}\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1\end{equation}  
这便是训练数据量和训练步数的经典关系式，它有两个参数$S_{\min},E_{\min}$，我们也可以通过实验搜索多个$(S,E)$来拟合上式，从而估计$S_{\min},E_{\min}$，进而可以估算$\mathcal{B}_{\text{noise}} = E_{\min} / S_{\min}$。更多分析细节请看回之前的文章[《当Batch Size增大时，学习率该如何随之变化？》](/archives/10542)或OpenAI的原论文[《An Empirical Model of Large-Batch Training》](https://papers.cool/arxiv/1812.06162)。

## 困难分析 #

前面写了那么多，都还停留在SGD中。从计算角度看，SGD是平凡的，真正复杂的是$\tilde{\boldsymbol{\varphi}}_B$非线性地依赖于$\tilde{\boldsymbol{g}}_B$的情形，比如SignSGD对应于$\newcommand{sign}{\mathop{\text{sign}}}\tilde{\boldsymbol{\varphi}}_B=\sign(\tilde{\boldsymbol{g}}_B)$，在理论分析中它经常用作Adam的近似，更准确的近似则是考虑了$\epsilon$的SoftSignSGD，我们在[《Adam的epsilon如何影响学习率的Scaling Law？》](/archives/10563)尝试过分析它。

在这些非线性场景下，$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$和$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$的计算往往是相当困难的，即便我们将$\tilde{\boldsymbol{g}}_B$的分布假设为简单的正态分布也是如此（注意，在SGD的分析中，我们并不需要对它的分布形式做正态假设）。比如，在之前的文章中，对于$\tilde{\boldsymbol{\varphi}}_B=\sign(\tilde{\boldsymbol{g}}_B)$的SignSGD，为了计算$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$，我们经历了如下步骤：

> 1、假设$\tilde{\boldsymbol{g}}_B$的分量相互独立，问题简化为单个分量$\tilde{\varphi}_B=\sign(\tilde{g}_B)$（没有加粗）的期望；
> 
> 2、假设$\tilde{g}_B$（此时是一个标量）服从正态分布，那么就可以算出$\mathbb{E}[\tilde{\varphi}_B]$，答案要用$\newcommand{erf}{\mathop{\text{erf}}}\erf$函数来表示；
> 
> 3、将$\erf$函数用$x/\sqrt{x^2+c}$形式的函数近似，简化结果。

也就是说，我们要经过一堆弯弯绕绕的步骤，才勉强算出一个可以分析下去的近似结果（这个过程首次出现在Tencent的论文[《Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling》](https://papers.cool/arxiv/2405.14578)），而且这已经算是简单的了，因为如果是SoftSignSGD，则更加复杂：

> 1、假设$\tilde{\boldsymbol{g}}_B$的分量相互独立，问题简化为单个分量$\tilde{\varphi}_B=\newcommand{softsign}{\mathop{\text{softsign}}}\softsign(\tilde{g}_B, \epsilon)$的期望；
> 
> 2、将$\softsign$函数用分段线性函数近似，这样才能算出下面的积分；
> 
> 3、假设$\tilde{g}_B$服从正态分布，结合第2步的近似，可以算出$\mathbb{E}[\tilde{\varphi}_B]$，答案是包含$\erf$的复杂函数；
> 
> 4、将复杂函数用$x/\sqrt{x^2+c}$形式的函数近似，简化结果。

事情还没完。费那么大劲，加那么多假设，我们才堪堪算出$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$，接着还要算$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$，这往往更加复杂（SignSGD是个例外，因为$\sign(x)^2$一定是1，所以反而简单了）。然而，计算的复杂性还是次要的，主要是这些步骤看上去没有任何能推广的规律，似乎只能具体问题具体分析的样子，这就让人觉得非常心累。

## 未完待续 #

为了避免文章过长，本文就先到这里了，主要先简单回顾一下现有的分析结果和计算困难。在下一篇文章中，笔者将会介绍自己为了降低推导过程中的心智负担所做的一些尝试。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11260>_

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

苏剑林. (Sep. 01, 2025). 《重新思考学习率与Batch Size（一）：现状 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11260>

@online{kexuefm-11260,  
title={重新思考学习率与Batch Size（一）：现状},  
author={苏剑林},  
year={2025},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/11260}},  
} 


---

## 公式推导与注释

### 1. 学习率的基本定义与作用机制

学习率$\eta$是梯度下降算法中最关键的超参数之一。在最基本的梯度下降更新规则中：

$$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \nabla \mathcal{L}(\boldsymbol{w}_t)$$

学习率控制着参数更新的步长。从优化理论的角度，我们可以通过泰勒展开来理解学习率的作用：

$$\mathcal{L}(\boldsymbol{w}_t - \eta \boldsymbol{g}_t) \approx \mathcal{L}(\boldsymbol{w}_t) - \eta \boldsymbol{g}_t^{\top} \boldsymbol{g}_t + \frac{1}{2}\eta^2 \boldsymbol{g}_t^{\top} \boldsymbol{H}_t \boldsymbol{g}_t$$

其中$\boldsymbol{H}_t$是Hessian矩阵。对$\eta$求导并令导数为零，可得理论最优学习率：

$$\eta_{\text{optimal}} = \frac{\boldsymbol{g}_t^{\top} \boldsymbol{g}_t}{\boldsymbol{g}_t^{\top} \boldsymbol{H}_t \boldsymbol{g}_t}$$

这个结果表明，学习率应该与梯度范数的平方成正比，与梯度在Hessian矩阵方向上的二次型成反比。在实践中，Hessian矩阵难以计算，因此我们通常使用固定学习率或基于经验的调度策略。

**物理解释**：学习率可以理解为在损失函数landscape上的"步长"。太小的学习率导致收敛缓慢，太大的学习率可能导致震荡甚至发散。最优学习率应该平衡下降速度和稳定性。

### 2. Batch Size对梯度估计的影响

在实际训练中，我们无法计算全数据集上的真实梯度$\boldsymbol{g} = \mathbb{E}[\tilde{\boldsymbol{g}}]$，而是使用mini-batch来估计。设Batch Size为$B$，batch内的样本梯度为$\{\tilde{\boldsymbol{g}}_1, \tilde{\boldsymbol{g}}_2, ..., \tilde{\boldsymbol{g}}_B\}$，则batch梯度为：

$$\tilde{\boldsymbol{g}}_B = \frac{1}{B}\sum_{i=1}^{B} \tilde{\boldsymbol{g}}_i$$

**均值分析**：显然有

$$\mathbb{E}[\tilde{\boldsymbol{g}}_B] = \mathbb{E}\left[\frac{1}{B}\sum_{i=1}^{B} \tilde{\boldsymbol{g}}_i\right] = \frac{1}{B}\sum_{i=1}^{B} \mathbb{E}[\tilde{\boldsymbol{g}}_i] = \boldsymbol{g}$$

这表明batch梯度是真实梯度的无偏估计。

**方差分析**：假设样本之间独立同分布，单样本梯度的协方差矩阵为$\boldsymbol{\Sigma} = \mathbb{E}[(\tilde{\boldsymbol{g}} - \boldsymbol{g})(\tilde{\boldsymbol{g}} - \boldsymbol{g})^{\top}]$，则：

$$\text{Cov}[\tilde{\boldsymbol{g}}_B] = \text{Cov}\left[\frac{1}{B}\sum_{i=1}^{B} \tilde{\boldsymbol{g}}_i\right] = \frac{1}{B^2}\sum_{i=1}^{B} \text{Cov}[\tilde{\boldsymbol{g}}_i] = \frac{\boldsymbol{\Sigma}}{B}$$

这是统计学中的关键结论：**梯度估计的方差与Batch Size成反比**。

**噪声强度的刻画**：我们可以用迹来度量梯度噪声的总体强度：

$$\sigma_{\text{noise}}^2 = \text{tr}(\boldsymbol{\Sigma}) = \sum_{i=1}^{N} \text{Var}[\tilde{g}_i]$$

对于Batch Size为$B$的情况，梯度估计的总噪声为：

$$\sigma_{\text{noise}, B}^2 = \text{tr}(\boldsymbol{\Sigma}/B) = \frac{\sigma_{\text{noise}}^2}{B}$$

这个$1/\sqrt{B}$的衰减规律是后续所有分析的基础。

### 3. 线性缩放规则（Linear Scaling Rule）的理论推导

线性缩放规则最早由Goyal等人在2017年的论文中提出，其核心思想是：**当Batch Size增大$k$倍时，学习率也应该增大$k$倍**。

**直观推导**：考虑两种训练方案：
- 方案A：Batch Size为$B$，学习率为$\eta$，训练1步
- 方案B：Batch Size为$1$，学习率为$\eta/B$，训练$B$步

在方案A中，参数更新为：

$$\Delta \boldsymbol{w}_A = -\eta \tilde{\boldsymbol{g}}_B = -\eta \cdot \frac{1}{B}\sum_{i=1}^{B} \tilde{\boldsymbol{g}}_i$$

在方案B中，总的参数更新为：

$$\Delta \boldsymbol{w}_B = -\sum_{i=1}^{B} \frac{\eta}{B} \tilde{\boldsymbol{g}}_i = -\frac{\eta}{B}\sum_{i=1}^{B} \tilde{\boldsymbol{g}}_i$$

如果$B$步中遇到的样本恰好就是方案A的batch中的样本，则$\Delta \boldsymbol{w}_A = \Delta \boldsymbol{w}_B$。这说明线性缩放规则能保证**期望意义下**的参数更新一致性。

**严格推导**：从损失下降的角度，考虑一阶近似：

$$\mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta \tilde{\boldsymbol{g}}_B)] \approx \mathcal{L}(\boldsymbol{w}) - \eta \mathbb{E}[\tilde{\boldsymbol{g}}_B]^{\top} \boldsymbol{g} = \mathcal{L}(\boldsymbol{w}) - \eta \|\boldsymbol{g}\|^2$$

这表明在一阶近似下，损失下降量只与学习率和真实梯度有关，与Batch Size无关。因此，理论上可以任意缩放$(B, \eta)$而不改变训练效果。

**适用条件**：线性缩放规则成立需要以下假设：
1. 梯度噪声相对较小：$\text{tr}(\boldsymbol{\Sigma}) \ll \|\boldsymbol{g}\|^2$
2. 学习率足够小，使得二阶项可忽略
3. 训练处于稳定阶段，梯度方向基本确定

当这些条件不满足时（如训练初期、大学习率、强噪声环境），线性缩放规则会失效。

### 4. 平方根缩放规则的理论基础

当考虑梯度噪声的影响时，线性缩放规则不再成立。从式$\eqref{eq:eta-sgd}$出发：

$$\eta^* = \frac{\eta_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}$$

在两个极端情况下：

**情况1：小Batch区域** ($B \ll \mathcal{B}_{\text{noise}}$)

使用泰勒展开$\frac{1}{1+x} \approx 1 - x$（当$x \gg 1$时近似为$1/x$）：

$$\eta^* \approx \frac{\eta_{\max}}{\mathcal{B}_{\text{noise}}/B} = \frac{\eta_{\max} B}{\mathcal{B}_{\text{noise}}} \propto B$$

这恢复了**线性缩放规则**。

**情况2：大Batch区域** ($B \gg \mathcal{B}_{\text{noise}}$)

此时：

$$\eta^* \approx \eta_{\max} \cdot \text{const}$$

学习率趋于常数，不再随Batch Size增长。

**中间区域的平方根规则**：有些文献提出在中等Batch Size下使用$\eta \propto \sqrt{B}$。这可以通过噪声-信号比分析得到：

梯度的信噪比定义为：

$$\text{SNR} = \frac{\|\mathbb{E}[\tilde{\boldsymbol{g}}_B]\|^2}{\text{tr}(\text{Cov}[\tilde{\boldsymbol{g}}_B])} = \frac{\|\boldsymbol{g}\|^2}{\text{tr}(\boldsymbol{\Sigma}/B)} = \frac{B \|\boldsymbol{g}\|^2}{\text{tr}(\boldsymbol{\Sigma})}$$

要保持信噪比不变，如果我们希望在不同的Batch Size下获得相同的梯度估计质量，可以设置：

$$\eta \propto \frac{1}{\sqrt{\text{SNR}}} \propto \frac{1}{\sqrt{B}}$$

然后通过增加训练步数来补偿：步数$S \propto B$，使得总的有效更新量$\eta \cdot S \propto \sqrt{B} \cdot B = B^{3/2}$保持一定规律。

实际上，平方根规则更适合于**固定计算预算**下的分析：在总计算量固定时，如何分配Batch Size和训练步数。

### 5. 梯度噪声的统计深入分析

**噪声的来源**：梯度噪声主要来自三个方面：
1. **采样噪声**：mini-batch采样的随机性
2. **标签噪声**：训练数据本身的标注错误
3. **本征噪声**：数据分布的固有随机性

**协方差矩阵的结构**：定义单样本梯度的协方差矩阵：

$$\boldsymbol{\Sigma} = \mathbb{E}_{\xi \sim \mathcal{D}}[(\nabla \ell(\boldsymbol{w}; \xi) - \boldsymbol{g})(\nabla \ell(\boldsymbol{w}; \xi) - \boldsymbol{g})^{\top}]$$

其中$\xi$是从数据分布$\mathcal{D}$中采样的单个样本，$\ell(\boldsymbol{w}; \xi)$是该样本的损失。

协方差矩阵的迹可以分解为各参数维度上的方差之和：

$$\text{tr}(\boldsymbol{\Sigma}) = \sum_{i=1}^{N} \sigma_i^2, \quad \sigma_i^2 = \mathbb{E}[(g_i(\xi) - g_i)^2]$$

**简化的噪声模型**：在实践中，我们常假设：

$$\boldsymbol{\Sigma} \approx \sigma^2 \boldsymbol{I}$$

即各维度的噪声独立且同分布。此时：

$$\text{tr}(\boldsymbol{\Sigma}) = N\sigma^2$$

虽然这个假设很强，但它极大简化了分析，且在很多情况下能给出合理的预测。

**噪声尺度的实际估计**：在训练过程中，我们可以通过以下方式估计$\mathcal{B}_{\text{noise}}$：

1. 在同一位置采样多个不同的batch
2. 计算这些batch的梯度方差
3. 使用公式$\mathcal{B}_{\text{simple}} = \text{tr}(\boldsymbol{\Sigma}) / \|\boldsymbol{g}\|^2$

OpenAI的实验表明，$\mathcal{B}_{\text{noise}}$在训练过程中会变化：
- 训练初期：损失高，梯度大，噪声相对小，$\mathcal{B}_{\text{noise}}$较小
- 训练后期：损失低，梯度小，噪声相对大，$\mathcal{B}_{\text{noise}}$增大

这解释了为什么训练后期需要更大的Batch Size才能维持稳定性。

**噪声与Hessian的耦合**：更精确的噪声度量应该考虑Hessian矩阵：

$$\mathcal{B}_{\text{noise}} = \frac{\text{tr}(\boldsymbol{\Sigma} \boldsymbol{H})}{\boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}}$$

这个定义的物理意义是：噪声在损失函数曲率方向上的投影。在梯度方向曲率较大的区域，即使噪声绝对值小，其影响也会被放大。

### 6. 大Batch训练的泛化差距（Generalization Gap）

**泛化差距的定义**：定义训练损失为$\mathcal{L}_{\text{train}}$，测试损失为$\mathcal{L}_{\text{test}}$，泛化差距为：

$$\Delta_{\text{gen}} = \mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}}$$

实验观察到：在达到相同的训练损失时，使用大Batch Size训练的模型往往有更大的$\Delta_{\text{gen}}$。

**Sharp Minima vs. Flat Minima理论**：

一种解释基于损失landscape的几何性质。定义Hessian矩阵的最大特征值为锐度指标：

$$S(\boldsymbol{w}) = \lambda_{\max}(\boldsymbol{H}(\boldsymbol{w}))$$

Sharp minima（锐的极小值）对应高锐度，Flat minima（平坦的极小值）对应低锐度。

**泰勒展开分析**：考虑测试数据上的损失，在训练解$\boldsymbol{w}^*$附近展开：

$$\mathcal{L}_{\text{test}}(\boldsymbol{w}^* + \delta \boldsymbol{w}) \approx \mathcal{L}_{\text{test}}(\boldsymbol{w}^*) + \frac{1}{2} \delta \boldsymbol{w}^{\top} \boldsymbol{H}_{\text{test}} \delta \boldsymbol{w}$$

其中$\delta \boldsymbol{w}$表示训练数据和测试数据分布差异导致的最优解偏移。如果$\boldsymbol{H}_{\text{test}}$的特征值较大（sharp），则小的$\delta \boldsymbol{w}$会导致大的损失增加。

**大Batch如何导致Sharp Minima**：

小Batch SGD引入的噪声可以看作一种隐式正则化：

$$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \tilde{\boldsymbol{g}}_B = \boldsymbol{w}_t - \eta \boldsymbol{g} - \eta \boldsymbol{\epsilon}_B$$

其中$\boldsymbol{\epsilon}_B \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}/B)$是噪声项。这个噪声项类似于给参数添加了随机扰动，帮助模型逃离sharp minima。

当$B$增大时，噪声减小，模型更容易困在sharp minima中。可以证明，稳态分布近似为：

$$p(\boldsymbol{w}) \propto \exp\left(-\frac{2B}{\eta} \mathcal{L}(\boldsymbol{w})\right)$$

"有效温度"为$T = \eta/(2B)$。大Batch（小温度）导致分布更集中于sharp minima，小Batch（高温度）倾向于flat minima。

**数据效率与泛化的权衡**：

虽然大Batch提高了数据效率（每步看更多数据），但牺牲了泛化能力。总体效果取决于：

$$\text{Total Cost} = \alpha \cdot E + \beta \cdot \Delta_{\text{gen}}$$

其中$E$是数据消耗，$\alpha, \beta$是权重系数。最优的Batch Size需要在这两者之间平衡。

### 7. 临界Batch Size的定义与分析

**临界Batch Size的物理定义**：

从式$\eqref{eq:eta-sgd}$可见，当$B = \mathcal{B}_{\text{noise}}$时：

$$\eta^* = \frac{\eta_{\max}}{1 + 1} = \frac{\eta_{\max}}{2}$$

此时学习率已经是最大值的一半，继续增大$B$的收益递减。我们定义：

$$\mathcal{B}_{\text{crit}} = \mathcal{B}_{\text{noise}} = \frac{\text{tr}(\boldsymbol{\Sigma} \boldsymbol{H})}{\boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}}$$

为**临界Batch Size**。

**从数据效率角度理解**：

回顾$E = (B + \mathcal{B}_{\text{noise}})S_{\min}$，定义数据效率为：

$$\epsilon(B) = \frac{E_{\min}}{E} = \frac{\mathcal{B}_{\text{noise}}}{B + \mathcal{B}_{\text{noise}}}$$

对$B$求导：

$$\frac{d\epsilon}{dB} = -\frac{\mathcal{B}_{\text{noise}}}{(B + \mathcal{B}_{\text{noise}})^2}$$

在$B = \mathcal{B}_{\text{noise}}$处：

$$\left|\frac{d\epsilon}{dB}\right|_{B=\mathcal{B}_{\text{noise}}} = \frac{\mathcal{B}_{\text{noise}}}{4\mathcal{B}_{\text{noise}}^2} = \frac{1}{4\mathcal{B}_{\text{noise}}}$$

效率对Batch Size的敏感度在临界点附近达到一个特征尺度。

**临界Batch Size的实验确定**：

实践中可以通过以下方式确定$\mathcal{B}_{\text{crit}}$：

1. 固定其他超参数，扫描不同的$(B, \eta)$组合
2. 记录达到目标损失所需的步数$S$和数据量$E$
3. 拟合双曲线关系$(S/S_{\min} - 1)(E/E_{\min} - 1) = 1$
4. 提取$\mathcal{B}_{\text{crit}} = E_{\min}/S_{\min}$

**临界Batch Size的动态性**：

$\mathcal{B}_{\text{crit}}$不是常数，而是随训练过程变化：

$$\mathcal{B}_{\text{crit}}(t) = \frac{\text{tr}(\boldsymbol{\Sigma}(t) \boldsymbol{H}(t))}{\boldsymbol{g}(t)^{\top} \boldsymbol{H}(t) \boldsymbol{g}(t)}$$

一般规律：
- 训练初期：梯度大，$\mathcal{B}_{\text{crit}}$小，适合小Batch
- 训练中期：$\mathcal{B}_{\text{crit}}$增大，可以增大Batch Size
- 训练后期：梯度小，噪声相对大，$\mathcal{B}_{\text{crit}}$很大，需要大Batch

这启发了**动态Batch Size**策略：随训练进行逐步增大$B$。

### 8. 学习率预热（Warmup）的数学原理

**预热的必要性**：

在训练初期，如果直接使用大学习率，可能导致：
1. 参数剧烈变化，破坏预训练权重（如有）
2. 梯度爆炸或消失
3. 优化器的二阶动量（如Adam的$v$）尚未稳定

**线性预热的定义**：

在前$T_{\text{warmup}}$步内，学习率从$\eta_0$线性增加到目标值$\eta_{\text{target}}$：

$$\eta(t) = \eta_0 + \frac{t}{T_{\text{warmup}}}(\eta_{\text{target}} - \eta_0), \quad t \leq T_{\text{warmup}}$$

通常取$\eta_0 = 0$或很小的值。

**理论解释1：二阶动量的积累**

对于Adam等自适应优化器，其更新规则为：

$$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t} + \epsilon}$$

其中$\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) \boldsymbol{g}_t^2$是梯度平方的指数移动平均。

在初期几步，$\boldsymbol{v}_t$还未充分积累，其值偏小，导致有效学习率$\eta/\sqrt{\boldsymbol{v}_t}$偏大。预热相当于给$\boldsymbol{v}_t$一个建立的时间窗口。

定量分析：假设梯度保持常数$\boldsymbol{g}$，则：

$$v_t = (1-\beta_2^t) g^2$$

在$t \ll 1/(1-\beta_2)$时，$v_t \approx (1-\beta_2) t g^2$，有效学习率为：

$$\eta_{\text{eff}} = \frac{\eta}{\sqrt{v_t}} \approx \frac{\eta}{\sqrt{(1-\beta_2) t} |g|} \propto \frac{1}{\sqrt{t}}$$

为了抵消这种$1/\sqrt{t}$的效应，可以设置$\eta(t) \propto \sqrt{t}$，这接近线性预热（在对数尺度上）。

**理论解释2：曲率的估计**

在训练初期，损失landscape的局部曲率（Hessian）可能与稳态时差异很大。理论最优学习率：

$$\eta^* = \frac{\|\boldsymbol{g}\|^2}{\boldsymbol{g}^{\top} \boldsymbol{H} \boldsymbol{g}}$$

如果初期的$\boldsymbol{H}$特征值异常大（sharp），则$\eta^*$应该很小。预热相当于逐步探测和适应局部曲率。

**理论解释3：梯度噪声的初始阶段**

初期的梯度噪声可能异常大（因为参数远离最优解）。根据：

$$\eta^* = \frac{\eta_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}$$

如果初期$\mathcal{B}_{\text{noise}}$很大，则需要较小的$\eta$。随着训练进行，$\mathcal{B}_{\text{noise}}$减小，可以增大$\eta$。

**预热步数的选择**：

经验上，$T_{\text{warmup}}$常取总训练步数的1%-10%。理论上应该取：

$$T_{\text{warmup}} \sim \frac{1}{1 - \beta_2}$$

对于$\beta_2 = 0.999$（Adam默认值），这给出$T_{\text{warmup}} \sim 1000$步。

**与Batch Size的关系**：

当使用大Batch Size时，每步看到的数据更多，优化器的统计量（$\boldsymbol{m}, \boldsymbol{v}$）积累更快。因此，预热步数可以相应减少：

$$T_{\text{warmup}}(B) = \frac{T_{\text{warmup}}(B_0)}{B/B_0}$$

或者保持数据量恒定：

$$T_{\text{warmup}}(B) \cdot B = \text{const}$$

### 9. 学习率衰减策略的理论对比

**指数衰减（Exponential Decay）**：

$$\eta(t) = \eta_0 \gamma^{t/T}$$

其中$\gamma < 1$是衰减率，$T$是衰减周期。

理论依据：假设损失随时间指数衰减$\mathcal{L}(t) \sim e^{-\alpha t}$，则梯度也指数衰减$\|\boldsymbol{g}(t)\| \sim e^{-\alpha t}$。根据$\eta^* \propto \|\boldsymbol{g}\|^2$，学习率应该按$e^{-2\alpha t}$衰减，这正是指数衰减的形式。

**多项式衰减（Polynomial Decay）**：

$$\eta(t) = \eta_0 \left(1 - \frac{t}{T_{\text{max}}}\right)^p$$

常用$p=1$（线性衰减）或$p=2$（二次衰减）。

理论依据：在凸优化中，可以证明使用$\eta(t) = \eta_0/\sqrt{t}$可以达到$O(1/\sqrt{t})$的收敛率。多项式衰减是这种思想的离散化版本。

**余弦衰减（Cosine Annealing）**：

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)$$

理论依据：余弦函数提供了平滑的、非单调的衰减曲线。在$t=0$和$t=T_{\text{max}}$附近斜率接近0，避免了突变。

**步长衰减（Step Decay）**：

$$\eta(t) = \eta_0 \gamma^{\lfloor t/T \rfloor}$$

每隔$T$步将学习率乘以$\gamma$。

理论依据：简单实用，但缺乏理论支持。可以看作指数衰减的分段常数近似。

**从最优学习率角度的统一理解**：

所有衰减策略本质上都在近似：

$$\eta(t) \approx \eta^*(t) = \frac{\|\boldsymbol{g}(t)\|^2}{\boldsymbol{g}(t)^{\top} \boldsymbol{H}(t) \boldsymbol{g}(t)}$$

在训练过程中：
- $\|\boldsymbol{g}(t)\|$通常单调递减（接近最优解）
- $\boldsymbol{H}(t)$的主导特征值变化较慢

因此$\eta^*(t)$应该递减，但具体形式依赖于任务。

**衰减策略与Batch Size的交互**：

当增大Batch Size时，相当于减小了梯度噪声，可以使用更稳定的学习率。一种策略是：

- 小Batch：需要激进的衰减，如指数衰减
- 大Batch：可以使用温和的衰减，如余弦衰减

数学上，考虑到$\mathcal{B}_{\text{noise}}$的变化：

$$\eta^*(t, B) = \frac{\eta_{\max}(t)}{1 + \mathcal{B}_{\text{noise}}(t)/B}$$

当$B$很大时，$\eta^* \approx \eta_{\max}(t)$，只需要调度$\eta_{\max}$即可；当$B$很小时，$\eta^*$对$\mathcal{B}_{\text{noise}}$敏感，需要更精细的调度。

### 10. 实验现象的理论解释

**现象1：线性缩放在小Batch区域成立，在大Batch区域失效**

理论解释：从$\eta^* = \eta_{\max}/(1 + \mathcal{B}_{\text{noise}}/B)$可见：
- 当$B \ll \mathcal{B}_{\text{noise}}$时，$\eta^* \approx \eta_{\max} B / \mathcal{B}_{\text{noise}} \propto B$（线性）
- 当$B \gg \mathcal{B}_{\text{noise}}$时，$\eta^* \approx \eta_{\max}$（常数）

临界点在$B \sim \mathcal{B}_{\text{noise}}$。

**现象2：大Batch训练需要更长的Warmup**

理论解释：大Batch降低了梯度噪声，使得优化轨迹更加确定性。这意味着：
- 初始不良的方向会被持续强化
- 缺乏小Batch的"探索"效应来纠正

因此需要更长的Warmup来小心翼翼地寻找好的初始方向。

从动量积累的角度：虽然大Batch时每步的统计量积累更快，但实际需要的是"信息多样性"而非"数据量"。Warmup步数应该按**样本数**而非**batch数**来计算。

**现象3：训练后期增大Batch Size有助于稳定性**

理论解释：训练后期$\|\boldsymbol{g}\|$减小，但$\text{tr}(\boldsymbol{\Sigma})$下降较慢（因为数据噪声是固有的），导致$\mathcal{B}_{\text{noise}} = \text{tr}(\boldsymbol{\Sigma})/\|\boldsymbol{g}\|^2$增大。

要维持$\eta^*$在合理范围，需要增大$B$以匹配增大的$\mathcal{B}_{\text{noise}}$。

**现象4：Adam对Batch Size不如SGD敏感**

理论解释：Adam的自适应学习率相当于隐式地归一化了梯度：

$$\tilde{\boldsymbol{\varphi}}_B^{(\text{Adam})} = \frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t} + \epsilon}$$

即使Batch Size改变导致$\boldsymbol{m}_t$的尺度改变，$\boldsymbol{v}_t$也会相应改变，两者的比值保持相对稳定。

从噪声的角度：Adam的分母$\sqrt{\boldsymbol{v}_t}$包含了梯度的二阶信息，部分捕捉了$\boldsymbol{\Sigma}$的对角线。这相当于自适应地调整了不同维度的有效Batch Size。

**现象5：最优Batch Size随模型和数据集变化**

理论解释：$\mathcal{B}_{\text{noise}} = \text{tr}(\boldsymbol{\Sigma}\boldsymbol{H})/(\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g})$依赖于：
- 数据集的固有噪声（影响$\boldsymbol{\Sigma}$）
- 模型的容量和架构（影响$\boldsymbol{H}$）
- 优化的阶段（影响$\boldsymbol{g}$）

不同任务的这些量可以相差几个数量级，因此最优Batch Size也会相差很大。

实验经验：
- 图像分类：$B \sim 256 - 1024$
- 语言模型预训练：$B \sim 2048 - 8192$
- 强化学习：$B \sim 32 - 256$（高噪声环境）

**总结**：上述所有推导和解释都基于二阶泰勒展开的框架：

$$\mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{\varphi}}_B)] \approx \mathcal{L}(\boldsymbol{w}) - \eta\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \text{tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})$$

关键是计算$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$和$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$，它们完全由优化器的具体形式决定。SGD的情况最简单，因为$\tilde{\boldsymbol{\varphi}}_B = \tilde{\boldsymbol{g}}_B$是线性的；而SignSGD、Adam等需要更复杂的分析，这正是下篇文章将要探讨的内容。

