---
title: 当Batch Size增大时，学习率该如何随之变化？
slug: 当batch-size增大时学习率该如何随之变化
date: 2024-11-14
tags: 详细推导, 梯度, 学习率, 优化器, 尺度定律, 生成模型
status: pending
---
# 当Batch Size增大时，学习率该如何随之变化？

**原文链接**: [https://spaces.ac.cn/archives/10542](https://spaces.ac.cn/archives/10542)

**发布日期**: 

---

随着算力的飞速进步，有越多越多的场景希望能够实现“算力换时间”，即通过堆砌算力来缩短模型训练时间。理想情况下，我们希望投入$n$倍的算力，那么达到同样效果的时间则缩短为$1/n$，此时总的算力成本是一致的。这个“希望”看上去很合理和自然，但实际上并不平凡，即便我们不考虑通信之类的瓶颈，当算力超过一定规模或者模型小于一定规模时，增加算力往往只能增大Batch Size。然而，增大Batch Size一定可以缩短训练时间并保持效果不变吗？

这就是接下来我们要讨论的话题：当Batch Size增大时，各种超参数尤其是学习率该如何调整，才能保持原本的训练效果并最大化训练效率？我们也可以称之为Batch Size与学习率之间的Scaling Law。

## 方差视角 #

直觉上，当Batch Size增大时，每个Batch的梯度将会更准，所以步子就可以迈大一点，也就是增大学习率，以求更快达到终点，缩短训练时间，这一点大体上都能想到。问题就是，增大多少才是最合适的呢？

### 二次方根 #

这个问题最早的答案可能是平方根缩放，即Batch Size扩大到$n$倍，则学习率扩大到$\sqrt{n}$倍，出自2014年的[《One weird trick for parallelizing convolutional neural networks》](https://papers.cool/arxiv/1404.5997)，推导原理是让SGD增量的方差保持不变。具体来说，我们将随机采样一个样本的梯度记为$\tilde{\boldsymbol{g}}$，其均值和协方差分别记为$\boldsymbol{g}$和$\boldsymbol{\Sigma}$，这里的$\boldsymbol{g}$就是全体样本的梯度。当我们将采样数目增加到$B$个时，有  
\begin{equation}\tilde{\boldsymbol{g}}_B \triangleq \frac{1}{B}\sum_{i=1}^B \tilde{\boldsymbol{g}}^{(i)},\quad \mathbb{E}[\tilde{\boldsymbol{g}}_B] = \boldsymbol{g},\quad \mathbb{E}[(\tilde{\boldsymbol{g}}_B-\boldsymbol{g})(\tilde{\boldsymbol{g}}_B-\boldsymbol{g})^{\top}]=\frac{\boldsymbol{\Sigma}}{B}\end{equation}  
即增加采样数目不改变均值，而协方差则缩小到$1/B$。对于SGD优化器来说，增量为$-\eta \tilde{\boldsymbol{g}}_B$，其协方差正比于$\eta^2/B$，而我们认为优化过程中适量的（不多不少的）噪声是有必要的，所以当Batch Size $B$变化时，我们通过调整学习率$\eta$让增量的噪声强度即协方差矩阵保持不变，从得出  
\begin{equation}\frac{\eta^2}{B} = \text{常数}\quad\Rightarrow\quad \eta\propto \sqrt{B}\end{equation}  
这就得到了学习率与Batch Size的平方根缩放定律，后来的[《Train longer, generalize better: closing the generalization gap in large batch training of neural networks》](https://papers.cool/arxiv/1705.08741)也认同这个选择。

### 线性缩放 #

有意思的是，线性缩放即$\eta\propto B$在实践中的表现往往更好，甚至刚才说的最早提出平方根缩放的[《One weird trick for parallelizing convolutional neural networks》](https://papers.cool/arxiv/1404.5997)作者也在论文中指出了这一点，并表示他也无法给出合理的解释。

某种程度上来说，线性缩放更符合我们的直观认知，尤其是像[《Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour》](https://papers.cool/arxiv/1706.02677)那样，假设连续的$n$个Batch的梯度方向变化不大的话，那么线性缩放几乎是显然成立的。不过，这个假设显然过强，放宽这个假设则需要将SGD跟SDE（随机微分方程）联系起来，这由[《Stochastic Modified Equations and Dynamics of Stochastic Gradient Algorithms I: Mathematical Foundations》](https://papers.cool/arxiv/1811.01558)完成，但首先用于指出学习率与Batch Size的缩放关系的论文应该是[《On the Generalization Benefit of Noise in Stochastic Gradient Descent》](https://papers.cool/arxiv/2006.15081)。

事后来看，这个联系的建立其实并不难理解，设模型参数为$\boldsymbol{w}$，那么SGD的更新规则可以改写成  
\begin{equation}\boldsymbol{w}_{t+1} =\boldsymbol{w}_t - \eta \tilde{\boldsymbol{g}}_{B,t} =\boldsymbol{w}_t - \eta \boldsymbol{g}_t - \eta (\tilde{\boldsymbol{g}}_{B,t} - \boldsymbol{g}_t)\end{equation}  
其中$\tilde{\boldsymbol{g}}_{B,t} - \boldsymbol{g}_t$即为梯度的噪声，到目前为止，我们还没有对这个噪声的分布做任何假设，只知道它的均值为$\boldsymbol{0}$，协方差为$\boldsymbol{\Sigma}_t/B$。接下来我们假设这个噪声的分布是正态分布$\mathcal{N}(\boldsymbol{0},\boldsymbol{\Sigma}_t/B)$，那么上述迭代可以进一步改写成  
\begin{equation}\begin{aligned}  
\boldsymbol{w}_{t+1} =&\, \boldsymbol{w}_t - \eta \boldsymbol{g}_t - \eta (\tilde{\boldsymbol{g}}_{B,t} - \boldsymbol{g}_t) \\\\[5pt]  
=&\, \boldsymbol{w}_t - \eta \boldsymbol{g}_t - \eta \sqrt{\frac{\boldsymbol{\Sigma}_t}{B}}\boldsymbol{z},\quad \boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}) \\\\[5pt]  
=&\, \boldsymbol{w}_t - \eta \boldsymbol{g}_t - \sqrt{\eta} \sqrt{\frac{\eta\boldsymbol{\Sigma}_t}{B}}\boldsymbol{z},\quad \boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}) \end{aligned}\end{equation}  
这就意味着SGD的迭代格式$\boldsymbol{w}_{t+1} =\boldsymbol{w}_t - \eta \tilde{\boldsymbol{g}}_{B,t}$实际上在近似地求解SDE：  
\begin{equation}d\boldsymbol{w} = - \boldsymbol{g}_t dt - \sqrt{\frac{\eta\boldsymbol{\Sigma}_t}{B}}d\boldsymbol{z},\quad d\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},dt\boldsymbol{I}) \end{equation}  
因此，要想在$B$发生变化时，运行结果不产生明显变化，上述SDE的形式应该不变，这就得到了线性缩放$\eta\propto B$。这个过程中最关键的一步是， _SDE的噪声项步长是非噪声项的平方根_ ，从而分离出一项$\sqrt{\eta}$来。这一点我们在[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)也有过评析，简单来说就是零均值的高斯噪声长期会有一定的抵消作用，所以必须增大步长才能将噪声效应体现出来。

以上结论都是基于SGD优化器得出的，论文[《On the SDEs and Scaling Rules for Adaptive Gradient Algorithms》](https://papers.cool/arxiv/2205.10287)将它推广到了RMSProp、Adam等优化器上，结果是 _平方根缩放_ 。无独有偶，稍早一点的[《Large Batch Optimization for Deep Learning: Training BERT in 76 minutes》](https://papers.cool/arxiv/1904.00962)在测试Adam及其变体LAMB时，也应用了平方根缩放。更多内容还可以参考博客[《How to Scale Hyperparameters as Batch Size Increases》](https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/)。

## 直面损失 #

可以肯定的是，不管是平方根缩放还是线性缩放，它们都只能在局部范围内近似成立，因为它们都包含了“只要Batch Size足够大，那么学习率就可以任意大”的结论，这显然是不可能的。此外，前面两节的工作都围绕着方差做文章，但我们的根本任务是降低损失函数，因此以损失函数为导向或许更为本质。

### 单调有界 #

这个视角下的经典工作是OpenAI的[《An Empirical Model of Large-Batch Training》](https://papers.cool/arxiv/1812.06162)，它通过损失函数的二阶近似来分析SGD的最优学习率，得出“学习率随着Batch Size的增加而单调递增但有上界”的结论。同样的思路在稍早的[《Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients》](https://papers.cool/arxiv/1705.07774)也出现了，不过那篇论文并不是用来分析Batch Size的作用。

整个推导过程最关键的思想是将学习率也视作优化参数：设损失函数是$\mathcal{L}(\boldsymbol{w})$，当前Batch的梯度是$\tilde{\boldsymbol{g}}_B$，那么SGD后的损失函数则是$\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{g}}_B)$，我们将最优学习率的求解视为优化问题  
\begin{equation}\eta^* = \mathop{\text{argmin}}_{\eta} \mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{g}}_B)]\end{equation}  
这个目标显然很直观，就是选择学习率使得 _平均而言_ 训练效率最高（损失函数下降得最快）。为了求解这个问题，我们将损失函数近似地展开到二阶：  
\begin{equation}\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{g}}_B) \approx \mathcal{L}(\boldsymbol{w}) - \eta\tilde{\boldsymbol{g}}_B^{\top}\underbrace{\frac{\partial \mathcal{L}(\boldsymbol{w})}{\partial\boldsymbol{w}}}_{\text{就是}\boldsymbol{g}} + \frac{1}{2}\eta^2 \tilde{\boldsymbol{g}}_B^{\top}\underbrace{\frac{\partial^2 \mathcal{L}(\boldsymbol{w})}{\partial\boldsymbol{w}^2}}_{\text{记为}\boldsymbol{H}}\tilde{\boldsymbol{g}}_B = \mathcal{L}(\boldsymbol{w}) - \eta\tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{H}\tilde{\boldsymbol{g}}_B\end{equation}  
这里的$\boldsymbol{H}$就是Hessian矩阵，而$\frac{\partial \mathcal{L}(\boldsymbol{w})}{\partial\boldsymbol{w}}$是损失函数的梯度，理想的目标函数是基于全量样本来求的，这也就是为什么它的梯度就是$\tilde{\boldsymbol{g}}_B$的均值$\boldsymbol{g}$。接着求期望，我们得到  
\begin{equation}\mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{g}}_B)] \approx \mathbb{E}[\mathcal{L}(\boldsymbol{w}) - \eta\tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{H}\tilde{\boldsymbol{g}}_B] = \mathcal{L}(\boldsymbol{w}) - \eta\boldsymbol{g}^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \mathbb{E}[\tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{H}\tilde{\boldsymbol{g}}_B]\end{equation}  
最后一项有少许技巧：  
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\begin{aligned}  
\mathbb{E}[\tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{H}\tilde{\boldsymbol{g}}_B] =&\, \mathbb{E}[\tr(\tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{H}\tilde{\boldsymbol{g}}_B)]= \mathbb{E}[\tr(\tilde{\boldsymbol{g}}_B\tilde{\boldsymbol{g}}_B^{\top}\boldsymbol{H})] = \tr(\mathbb{E}[\tilde{\boldsymbol{g}}_B\tilde{\boldsymbol{g}}_B^{\top}]\boldsymbol{H})\\\\[5pt]  
=&\, \tr((\boldsymbol{g}\boldsymbol{g}^{\top} + \boldsymbol{\Sigma}/B)\boldsymbol{H}) = \boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g} + \tr(\boldsymbol{\Sigma}\boldsymbol{H})/B  
\end{aligned}\end{equation}  
变换过程主要利用到了$\tr(\boldsymbol{A}\boldsymbol{B}) = \tr(\boldsymbol{B}\boldsymbol{A})$。现在只要假定$\boldsymbol{H}$的正定性，那么问题就变成了二次函数的最小值，容易解得  
\begin{equation}\eta^* \approx \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g} + \tr(\boldsymbol{\Sigma}\boldsymbol{H})/B} = \frac{\eta_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}\label{eq:eta-opt}\end{equation}  
这就得出了“随着$B$单调递增有上界“的结果，其中  
\begin{equation}\eta_{\max} = \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}},\qquad\mathcal{B}_{\text{noise}} = \frac{\tr(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\end{equation}

### 实践分析 #

当$B \ll \mathcal{B}_{\text{noise}}$时，$1 + \mathcal{B}_{\text{noise}}/B\approx \mathcal{B}_{\text{noise}}/B$，所以$\eta^* \approx \eta_{\max}B/\mathcal{B}_{\text{noise}}\propto B$，即线性缩放，这再次体现了线性缩放只是小Batch Size时的局部近似；当$B > \mathcal{B}_{\text{noise}}$时，$\eta^*$逐渐趋于饱和值$\eta_{\max}$，这意味着训练成本的增加远大于训练效率的提升。所以，$\mathcal{B}_{\text{noise}}$相当于一个分水岭，当Batch Size超过这个数值时，就没必要继续投入算力去增大Batch Size了。

对于实践来说，最关键的问题无疑就是如何估计$\eta_{\max}$和$\mathcal{B}_{\text{noise}}$了，尤其是$\mathcal{B}_{\text{noise}}$直接关系到学习率的缩放规律和训练效率的饱和问题，二者的直接计算涉及到Hessian矩阵$\boldsymbol{H}$，其计算量正比于参数量的平方，在数亿参数量都算小模型的今天，计算Hessian矩阵显然是不现实的事情，所以必须寻找更有效的计算方式。

我们先来看$\mathcal{B}_{\text{noise}}$，它的式子是$\frac{\tr(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}$，分子分母都有一个$\boldsymbol{H}$，这无疑有一种让我们将它们“约掉”的冲动。事实上简化的思路也是如此，假设$\boldsymbol{H}$近似于单位阵的若干倍，那么得到  
\begin{equation}\mathcal{B}_{\text{noise}} = \frac{\tr(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\approx \frac{\tr(\boldsymbol{\Sigma})}{\boldsymbol{g}^{\top}\boldsymbol{g}}\triangleq \mathcal{B}_{\text{simple}}\end{equation}  
$\mathcal{B}_{\text{simple}}$在计算上更为可行，并且实验发现它通常是$\mathcal{B}_{\text{noise}}$的一个良好近似，因此我们选择估计$\mathcal{B}_{\text{simple}}$而不是$\mathcal{B}_{\text{noise}}$。注意$\tr(\boldsymbol{\Sigma})$只需要对角线上的元素，因此不用算出完整的协方差矩阵，只需要将每个梯度分量单独算方差然后求和。在数据并行场景，可以直接利用每个设备上算出来梯度来估计梯度方差。

需要指出的是，式$\eqref{eq:eta-opt}$等结果实际上是动态的，也就是说理论上每一步训练的$\eta_{\max}$、$\mathcal{B}_{\text{noise}}$、$\mathcal{B}_{\text{simple}}$都是不同的，所以如果我们希望得到一个静态的规律，需要持续训练一段时间，等到模型的训练进入“正轨”后计算的$\mathcal{B}_{\text{simple}}$才可靠的，或者也可以在训练过程中持续监控$\mathcal{B}_{\text{simple}}$，以便判断当前设置与最优的差距。

至于$\eta_{\max}$，其实就没必要根据公式来估计了，直接在某个小Batch Size下对学习率进行网格搜索，搜出一个近似的$\eta^*$，然后结合估计的$\mathcal{B}_{\text{simple}}$就可以反推出$\eta_{\max}$了。

### 数据效率 #

从上述结果出发，我们还可以推导关于训练数据量和训练步数的一个渐近关系。推导过程也很简单，将$\eqref{eq:eta-opt}$代入到损失函数中可以算得，在最优学习率下每一步迭代带来的损失函数减少量是：  
\begin{equation}\overline{\Delta\mathcal{L}} = \mathcal{L}(\boldsymbol{w}) - \mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta^*\tilde{\boldsymbol{g}}_B)] \approx \frac{\Delta\mathcal{L}_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}\label{eq:Delta-L-sgd}\end{equation}  
其中$\Delta\mathcal{L}_{\max} = \frac{(\boldsymbol{g}^{\top}\boldsymbol{g})^2}{2\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}$。接下来的重点是对这个结果的解读。

当$B\to\infty$也就是全量SGD时，每一步损失函数减少量达到了最大的$\Delta\mathcal{L}_{\max}$，这时候可以用最少的训练步数（记为$S_{\min}$）达到目标点。当$B$有限时，每一步的损失下降量平均只有$\overline{\Delta\mathcal{L}}$，这意味我们需要$1 + \mathcal{B}_{\text{noise}}/B$步才能达到全量SGD单步的下降量，所以训练总步数大致上是$S = (1 + \mathcal{B}_{\text{noise}}/B)S_{\min}$。

由于Batch Size为$B$，所以训练过程消耗的样本总数则是$E = BS = (B + \mathcal{B}_{\text{noise}})S_{\min}$，这是$B$的增函数，且当$B\to 0$时，$E_{\min} = \mathcal{B}_{\text{noise}}S_{\min}$，这表明只要我们使用足够小的Batch Size去训练模型，那么所需要的总训练样本数$E$也会相应地减少，代价是训练步数$S$非常多。进一步地，利用这些记号我们可以写出它们之间的关系是：  
\begin{equation}\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1\label{eq:E-S}\end{equation}  
这就是训练数据量和训练步数之间的缩放规律，表明数据量越小，那么应该缩小Batch Size，让训练步数更多，才能更有机会达到更优的解。这里的推导是经过笔者简化的，假设了$\mathcal{B}_{\text{noise}}$和$\Delta\mathcal{L}_{\max}$在整个训练过程的不变性，如果有必要也可以按照原论文附录用积分更精细地处理动态变化的情形（但需要引入假设$B = \sqrt{r\mathcal{B}_{\text{noise}}}$），这里就不展开了。

此外，由于$\mathcal{B}_{\text{noise}} = E_{\min}/S_{\min}$，所以上式也提供了估计$\mathcal{B}_{\text{noise}}$的另一个方案：通过多次实验加网格搜索得到多个$(S,E)$对，然后拟合上式就可以估计出$E_{\min},S_{\min}$，继而计算$\mathcal{B}_{\text{noise}}$。

## 自适应版 #

不得不说，OpenAI不愧为各种Scaling Law的先驱之一，前述分析可谓相当精彩，并且结果也相当丰富，更难得的是，整个推导过程并不复杂，给人一种大道至简的本质感。不过，目前的结论都是基于SGD来推的，对于Adam等自适应学习率优化器的适用性还不明朗，这部分内容由[《Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling》](https://papers.cool/arxiv/2405.14578)完成。

### 符号近似 #

分析Adam的思路跟SGD一样，都是基于二阶展开，不同的是方向向量由$\tilde{\boldsymbol{g}}_B$换成了一般的向量$\tilde{\boldsymbol{\varphi}}_B$，此时我们有  
\begin{equation}\mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{\varphi}}_B)] \approx \mathcal{L}(\boldsymbol{w}) - \eta\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \tr(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})\end{equation}  
现在需要确定$\tilde{\boldsymbol{\varphi}}_B$以及计算相应的$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$和$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$。由于只需要一个渐近关系，所以跟[《配置不同的学习率，LoRA还能再涨一点？》](/archives/10001)一样，我们选择SignSGD即$\newcommand{sign}{\mathop{\text{sign}}}\tilde{\boldsymbol{\varphi}}_B = \sign(\tilde{\boldsymbol{g}}_B)$作为Adam的近似，这个做法最早的出处可能是[《Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients》](https://papers.cool/arxiv/1705.07774)。这个近似的合理性体现在两点：

> 1、无论$\beta_1,\beta_2$取何值，Adam第一步的更新向量都是$\sign(\tilde{\boldsymbol{g}}_B)$；
> 
> 2、当$\beta_1=\beta_2=0$时，Adam的更新向量始终为$\sign(\tilde{\boldsymbol{g}}_B)$。

为了计算$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$和$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$，我们还需要跟“线性缩放”一节一样，假设$\tilde{\boldsymbol{g}}_B$服从分布$\mathcal{N}(\boldsymbol{g},\boldsymbol{\Sigma}/B)$，而为了简化计算，我们还要进一步假设$\boldsymbol{\Sigma}$是对角阵$\text{diag}(\sigma_1^2,\sigma_2^2,\sigma_3^2,\cdots)$，即假设分量之间是相互独立的，这样一来我们可以独立地处理每一个分量。根据重参数技巧，我们知道$\tilde{g}_B\sim \mathcal{N}(g, \sigma^2/B)$等价于$\tilde{g}_B=g + \sigma z/\sqrt{B},z\sim\mathcal{N}(0,1)$，因此  
\begin{equation}\begin{aligned}  
\mathbb{E}[\tilde{\varphi}_B] =&\, \mathbb{E}[\sign(g + \sigma z/\sqrt{B})] = \mathbb{E}[\sign(g\sqrt{B}/\sigma + z)] \\\\[5pt]  
=&\,\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \sign(g\sqrt{B}/\sigma + z) e^{-z^2/2}dz \\\\[5pt]  
=&\,\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{-g\sqrt{B}/\sigma} (-1)\times e^{-z^2/2}dz + \frac{1}{\sqrt{2\pi}}\int_{-g\sqrt{B}/\sigma}^{\infty} 1\times e^{-z^2/2}dz \\\\[5pt]  
=&\,\text{erf}\left(\frac{g}{\sigma}\sqrt{\frac{B}{2}}\right)  
\end{aligned}\end{equation}  
这里的$\text{erf}$是[误差函数](https://en.wikipedia.org/wiki/Error_function)，它是跟$\tanh$类似的值域为$(-1,1)$的S型函数，可以作为$\sign$的光滑近似。但$\text{erf}$本身没有初等函数表达式，所以我们最好找一个初等函数近似，才能更直观地观察变化规律，之前我们在[《GELU的两个初等函数近似是怎么来的》](/archives/7309)就讨论过这个话题，不过那里的近似还是太复杂了（都涉及到指数运算），这里我们整个简单点的：  
\begin{equation}\text{erf}(x)\approx \sign(x) = \frac{x}{|x|} = \frac{x}{\sqrt{x^2}}\approx \frac{x}{\sqrt{x^2+c}}\end{equation}  
我们选择$c=\pi/4$，使得这个近似在$x=0$处的一阶近似跟$\text{erf}$的一阶近似相等。当然，都做了这么多重近似了，这个$c$的值其实已经不大重要，我们只需要知道存在这么个$c > 0$就行了。基于这个近似，我们得到  
\begin{equation}\mathbb{E}[\tilde{\varphi}_B] \approx \frac{g/\sigma}{\sqrt{\pi/2B+(g/\sigma)^2}}\quad\Rightarrow\quad\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]_i \approx \frac{g_i/\sigma_i}{\sqrt{\pi/2B+(g_i/\sigma_i)^2}}\triangleq \mu_i\end{equation}  
可以发现，Adam跟SGD的一个明显区别是$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$这一步就已经跟$B$相关了。不过好在，此时的二阶矩更简单了，因为$\sign(x)$的平方必然是1，所以  
\begin{equation}\mathbb{E}[\tilde{\varphi}_B^2] = 1\quad\Rightarrow\quad\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]_{i,j} \to\left\\{\begin{aligned}&=1, & i = j \\\  
&\approx\mu_i \mu_j,&\,i\neq j\end{aligned}\right.\end{equation}  
利用这些结果，我们就可以求得  
\begin{gather}\eta^* \approx \frac{\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g}}{\tr(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})} \approx \frac{\sum_i \mu_i g_i}{\sum_i H_{i,i} + \sum_{i\neq j} \mu_i \mu_j H_{i,j}}\label{eq:eta-opt-sign} \\\\[5pt]  
\overline{\Delta\mathcal{L}} = \mathcal{L}(\boldsymbol{w}) - \mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta^*\tilde{\boldsymbol{\varphi}}_B)] \approx \frac{1}{2}\frac{(\sum_i \mu_i g_i)^2}{\sum_i H_{i,i} + \sum_{i\neq j} \mu_i \mu_j H_{i,j}}\label{eq:Delta-L-sign}\end{gather}

### 两个特例 #

相比SGD的式$\eqref{eq:eta-opt}$ ，Adam的式$\eqref{eq:eta-opt-sign}$更为复杂，以至于无法直观看出它对$B$的依赖规律，所以我们从几个特殊例子入手。

首先考虑$B\to\infty$，此时$\mu_i = \sign(g_i)$，所以  
\begin{equation}\eta^* \approx \frac{\sum_i |g_i|}{\sum_i H_{i,i} + \sum_{i\neq j} \sign(g_i g_j) H_{i,j}}\end{equation}  
它跟SGD的$\eta_{\max}$的区别是它关于梯度并不是齐次的，而是正比于梯度的scale。

接着我们考虑$\boldsymbol{H}$是对角阵的例子，即$i\neq j$时$H_{i,j}=0$，此时  
\begin{equation}\eta^* \approx \frac{\sum_i \mu_i g_i}{\sum_i H_{i,i}}=\frac{1}{\sum_i H_{i,i}}\sum_i \frac{g_i^2/\sigma_i}{\sqrt{\pi/2B+(g_i/\sigma_i)^2}}\end{equation}  
这里求和的每一项关于$B$都是单调递增有上界的，所以总的结果也是如此。为了捕捉最本质的规律，我们可以考虑进一步简化$\mu_i$（这里开始跟原论文不一样）：  
\begin{equation}\mu_i = \frac{g_i/\sigma_i}{\sqrt{\pi/2B+(g_i/\sigma_i)^2}} = \frac{\sign(g_i)}{\sqrt{1 + \pi(\sigma_i/g_i)^2/2B}} \approx \frac{\sign(g_i)}{\sqrt{1 + \pi\kappa^2/2B}}\label{eq:mu-approx}\end{equation}  
这里的假设是存在某个跟$i$无关的常数$\kappa^2$【比如可以考虑取全体$(\sigma_i/g_i)^2$的某种均值，其实这里的$\kappa^2$类似前面的$\mathcal{B}_{\text{simple}}$，按照$\mathcal{B}_{\text{simple}}$的定义来估计也可以】，使得对任意$i$来说把$(\sigma_i/g_i)^2$换成$\kappa^2$都是一个良好近似，于是  
\begin{equation}\eta^* \approx \frac{\sum_i \mu_i g_i}{\sum_i H_{i,i}}\approx \frac{\sum_i |g_i|}{\sum_i H_{i,i}}\frac{1}{\sqrt{1 + \pi\kappa^2/2B}}\label{eq:eta-opt-sign-diag}\end{equation}  
当$\pi\kappa^2\gg 2B$即$B \ll \pi\kappa^2/2$时，可以进一步写出近似  
\begin{equation}\eta^* \approx \frac{\sum_i |g_i|}{\kappa\sum_i H_{i,i}}\sqrt{\frac{2B}{\pi}} \propto \sqrt{B}\end{equation}  
这表明在Batch Size本身较小时，Adam确实适用于平方根缩放定律。

### 涌现行为 #

如果我们将近似$\eqref{eq:mu-approx}$应用到原始的式$\eqref{eq:eta-opt-sign}$，会发现它存在一些全新的特性，具体来说我们有  
\begin{equation}\eta^* \approx \frac{\sum_i \mu_i g_i}{\sum_i H_{i,i} + \sum_{i\neq j} \mu_i \mu_j H_{i,j}} \approx \frac{\eta_{\max}}{\frac{1}{2}\left(\frac{\beta_{\text{noise}}}{\beta} + \frac{\beta}{\beta_{\text{noise}}}\right)}\label{eq:eta-opt-beta}\end{equation}  
其中$\beta = (1 + \pi\kappa^2/2B)^{-1/2}$，以及  
\begin{equation}\beta_{\text{noise}} = \sqrt{\frac{\sum_i H_{i,i}}{\sum_{i\neq j}\sign(g_i g_j) H_{i,j}}},\quad \eta_{\max} = \frac{\sum_i |g_i|}{2\sqrt{\left(\sum_i H_{i,i}\right)\left(\sum_{i\neq j} \sign(g_i g_j) H_{i,j}\right)}}\end{equation}  
注意$\beta$是$B$的单调递增函数，但式$\eqref{eq:eta-opt-beta}$最后的近似并不是$\beta$的单调递增函数，它是先增后减的，最大值在$\beta=\beta_{\text{noise}}$取到。这意味着存在一个相应的$\mathcal{B}_{\text{noise}}$，当Batch Size超过这个$\mathcal{B}_{\text{noise}}$后，最佳学习率不应该增大反而要减小！这便是原论文标题所说的“Surge现象”。（当然这里还有一个限制，$\beta$是始终小于$1$的，如果$\beta_{\text{noise}} \geq 1$，那么最优学习率与Batch Size的关系依旧是单调递增的。）

关于Adam的$\eta^*$，其实OpenAI在论文附录中曾不加证明地“猜测”Adam的最优学习率应该是  
\begin{equation}\eta^* \approx \frac{\eta_{\max}}{(1 + \mathcal{B}_{\text{noise}}/B)^{\alpha}}\label{eq:openai-adam}\end{equation}  
其中$0.5 < \alpha < 1$。现在看来，这个形式只是Hessian矩阵对角线元素占主导时的近似结果，当非对角线元素的作用不可忽略时，则有可能涌现出“Batch Size足够大时学习率反而应该减小”的Surge现象。

如何直观地理解Surge现象呢？笔者认为，这本质上是自适应学习率策略的**次优性** 的体现。仍以近似$\tilde{\boldsymbol{\varphi}}_B = \sign(\tilde{\boldsymbol{g}}_B)$为例，$B$越大$\tilde{\boldsymbol{g}}_B$就越准，$B\to \infty$则是$\sign(\boldsymbol{g})$，然而$\sign(\boldsymbol{g})$是最科学的更新方向吗？不一定，尤其是训练后期这种自适应策略可能还有负面作用。因此，当$B$取适当值时，$\sign(\tilde{\boldsymbol{g}}_B)$的噪声反而可能修正这种次优性，而$B$继续增大时噪声减少，反而减少了修正的机会，从而需要更谨慎地降低学习率。

### 效率关系 #

同SGD的分析一样，最后我们还可以考虑$\overline{\Delta\mathcal{L}}$，将式$\eqref{eq:eta-opt-beta}$代入式$\eqref{eq:Delta-L-sign}$，恢复记号$B$然后化简（化简过程不需要任何近似）得到  
\begin{equation}\overline{\Delta\mathcal{L}} \approx \frac{\Delta\mathcal{L}_{\max}}{1 + \mathcal{B}_{\text{noise-2}}/B}\label{eq:Delta-L-sign-2}\end{equation}  
其中  
\begin{equation}\Delta\mathcal{L}_{\max} = \frac{\beta_{\text{noise}}\eta_{\max}\sum_i|g_i|}{1 + \beta_{\text{noise}}^2},\quad \mathcal{B}_{\text{noise-2}} = \frac{\pi\kappa^2\beta_{\text{noise}}^2}{2(1 + \beta_{\text{noise}}^2)}\label{eq:beta-B-noise}\end{equation}  
注意这里$\mathcal{B}_{\text{noise-2}}$是一个新的记号，它不是$\mathcal{B}_{\text{noise}}$，后者是由$\beta=\beta_{\text{noise}}$反解出来的理论最优Batch Size，结果是  
\begin{equation}\mathcal{B}_{\text{noise}} = \frac{\pi\kappa^2\beta_{\text{noise}}^2}{2(1 - \beta_{\text{noise}}^2)}\end{equation}  
它们之间的关系是  
\begin{equation}\frac{1}{\mathcal{B}_{\text{noise-2}}} - \frac{1}{\mathcal{B}_{\text{noise}}} = \frac{4}{\pi\kappa^2}\quad\Rightarrow\quad \mathcal{B}_{\text{noise}} = \left(\frac{1}{\mathcal{B}_{\text{noise-2}}} - \frac{4}{\pi\kappa^2}\right)^{-1}\label{eq:B-1-2}\end{equation}  
由于式$\eqref{eq:Delta-L-sign-2}$形式上跟SGD的式$\eqref{eq:Delta-L-sgd}$是一样的，所以那一节的分析同样适用，因此同样可以导出式$\eqref{eq:E-S}$：  
\begin{equation}\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1\end{equation}  
只不过现在$E_{\min}/S_{\min} = \mathcal{B}_{\text{noise-2}}$。这样一来，我们就有得到一种估计$\beta_{\text{noise}}$和$\mathcal{B}_{\text{noise}}$的方案：通过多次实验得到多个$(S,E)$对，实验过程中还可以顺便估计$\kappa^2$，然后拟合上式得到$E_{\min},S_{\min}$，继而估计$\mathcal{B}_{\text{noise-2}}$，最后由$\eqref{eq:beta-B-noise}$式解出$\beta_{\text{noise}}$。

如果$\beta_{\text{noise}} \geq 1$，那么不存在最优的$\mathcal{B}_{\text{noise}}$，如果$\beta_{\text{noise}} \gg 1$则说明Hessian矩阵对角线元素占主导，此时适用于缩放规律$\eqref{eq:eta-opt-sign-diag}$，增大Batch Size总可以适当增大学习率；当$\beta_{\text{noise}} < 1$时，可以由$\eqref{eq:B-1-2}$解出最优的$\mathcal{B}_{\text{noise}}$，Batch Size超出这个值学习率反而应该下降。

### 补充说明 #

需要指出的是，上面几节分析的出发点和最终结论，其实跟原论文[《Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling》](https://papers.cool/arxiv/2405.14578)大同小异，但中间过程的近似处理有所不同。

原论文得到的大部分结论，都是在$B \ll \pi(\sigma_i/g_i)^2/2$假设下的近似结果，所以得到Surge现象几乎总会出现的结论，这其实是不大科学的。最明显的是$B \ll \pi(\sigma_i/g_i)^2/2$这个假设的形式本身就有点问题，它右端是跟$i$相关的，我们总不能给每个分量都配一个单独的Batch Size，所以为了得到一个全局的结果就只能是$B \ll \min_i \pi(\sigma_i/g_i)^2/2$，但这未免有点苛刻了。

本文的做法则是引入近似$\eqref{eq:mu-approx}$，这可以看成是平均场近似，直觉上比逐点的假设$B \ll \pi(\sigma_i/g_i)^2/2$更为合理一些，所以原则上结论会更为精准，比如可以得到“即使Hessian矩阵的非对角线元素不可忽略，Surge现象也不一定会出现”的结论（取决于$\beta_{\text{noise}}$）。特别地，这种精准性并没有牺牲简洁性，比如式$\eqref{eq:eta-opt-beta}$同样很简明清晰，式$\eqref{eq:Delta-L-sign-2}$形式也跟原论文一致，并且不需要额外的近似假设，等等。

最后，稍微感慨一下，OpenAI对SGD的分析其实已经是2018年的工作了，而Surge现象这篇论文则是今年中才发布的，从SGD到Adam居然花了6年时间，这是让人比较意外的，大体是OpenAI的“威望”以及猜测$\eqref{eq:openai-adam}$，让大家觉得Adam已经没什么好做了，没想到Adam可能会有一些新的特性。当然，$\tilde{\boldsymbol{\varphi}}_B = \sign(\tilde{\boldsymbol{g}}_B)$作为Adam的近似究竟有多合理、能多大程度上代表实际情况等问题，笔者认为还值得进一步思考。

## 文章小结 #

本文从多个视角讨论了“Batch Size与学习率之间的Scaling Law”这一经典炼丹问题，其中着重介绍了OpenAI基于损失函数的二阶近似的推导和结论，以及后续利用同样的思想来分析Adam优化器的工作。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10542>_

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

苏剑林. (Nov. 14, 2024). 《当Batch Size增大时，学习率该如何随之变化？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10542>

@online{kexuefm-10542,  
title={当Batch Size增大时，学习率该如何随之变化？},  
author={苏剑林},  
year={2024},  
month={Nov},  
url={\url{https://spaces.ac.cn/archives/10542}},  
} 


---

## 公式推导与注释

本节提供极其详细的数学推导，深入探讨Batch Size与学习率之间的Scaling Law，涵盖从方差分析到实际应用的完整理论框架。

### 1. 梯度估计的基础理论

**推导1：单样本梯度的定义**

假设损失函数为 $\mathcal{L}(\boldsymbol{w})$，第 $i$ 个样本的损失为 $\ell_i(\boldsymbol{w})$，则：

$$
\mathcal{L}(\boldsymbol{w}) = \frac{1}{N}\sum_{i=1}^N \ell_i(\boldsymbol{w})
$$

其中 $N$ 是总样本数。真实梯度为：

$$
\boldsymbol{g}(\boldsymbol{w}) = \nabla_{\boldsymbol{w}}\mathcal{L}(\boldsymbol{w}) = \frac{1}{N}\sum_{i=1}^N \nabla_{\boldsymbol{w}}\ell_i(\boldsymbol{w})
$$

**推导2：随机梯度的期望和方差**

定义单个样本的随机梯度 $\tilde{\boldsymbol{g}}^{(i)} = \nabla_{\boldsymbol{w}}\ell_i(\boldsymbol{w})$。若样本是从总体中均匀随机抽取的，则：

$$
\mathbb{E}[\tilde{\boldsymbol{g}}^{(i)}] = \boldsymbol{g}(\boldsymbol{w})
$$

定义梯度的协方差矩阵：

$$
\boldsymbol{\Sigma} = \mathbb{E}[(\tilde{\boldsymbol{g}}^{(i)} - \boldsymbol{g})(\tilde{\boldsymbol{g}}^{(i)} - \boldsymbol{g})^T]
$$

**注释1**：协方差矩阵 $\boldsymbol{\Sigma}$ 衡量了单个样本梯度的不确定性。对角元素 $\Sigma_{ii}$ 表示第 $i$ 个参数的梯度方差。

**推导3：Batch梯度的期望和方差**

使用Batch Size为 $B$ 的小批量，梯度估计为：

$$
\tilde{\boldsymbol{g}}_B = \frac{1}{B}\sum_{i=1}^B \tilde{\boldsymbol{g}}^{(i)}
$$

由于样本独立同分布：

$$
\begin{aligned}
\mathbb{E}[\tilde{\boldsymbol{g}}_B] &= \mathbb{E}\left[\frac{1}{B}\sum_{i=1}^B \tilde{\boldsymbol{g}}^{(i)}\right] = \frac{1}{B}\sum_{i=1}^B \mathbb{E}[\tilde{\boldsymbol{g}}^{(i)}] = \boldsymbol{g} \\
\text{Cov}[\tilde{\boldsymbol{g}}_B] &= \text{Cov}\left[\frac{1}{B}\sum_{i=1}^B \tilde{\boldsymbol{g}}^{(i)}\right] = \frac{1}{B^2}\sum_{i=1}^B \text{Cov}[\tilde{\boldsymbol{g}}^{(i)}] = \frac{\boldsymbol{\Sigma}}{B}
\end{aligned}
$$

**注释2**：这个关键结果表明，增大Batch Size使梯度估计的方差按 $1/B$ 缩小。

### 2. 平方根缩放规律的推导

**推导4：SGD增量的方差分析**

SGD的参数更新为：

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\tilde{\boldsymbol{g}}_B
$$

更新量的期望和方差为：

$$
\begin{aligned}
\mathbb{E}[\Delta\boldsymbol{w}] &= -\eta\boldsymbol{g} \\
\text{Var}[\Delta\boldsymbol{w}] &= \eta^2\text{Var}[\tilde{\boldsymbol{g}}_B] = \frac{\eta^2\boldsymbol{\Sigma}}{B}
\end{aligned}
$$

**推导5：等方差原则**

假设我们希望在改变Batch Size时，保持更新量的方差不变：

$$
\frac{\eta_1^2}{B_1} = \frac{\eta_2^2}{B_2}
$$

整理得：

$$
\frac{\eta_2}{\eta_1} = \sqrt{\frac{B_2}{B_1}}
$$

**注释3**：这就是平方根缩放规律的理论基础——通过调整学习率，保持参数更新的噪声强度恒定。

**推导6：信噪比分析**

定义信噪比（SNR）：

$$
\text{SNR} = \frac{\|\mathbb{E}[\Delta\boldsymbol{w}]\|}{\sqrt{\text{Tr}(\text{Var}[\Delta\boldsymbol{w}])}} = \frac{\eta\|\boldsymbol{g}\|}{\eta\sqrt{\text{Tr}(\boldsymbol{\Sigma})/B}} = \sqrt{B}\frac{\|\boldsymbol{g}\|}{\sqrt{\text{Tr}(\boldsymbol{\Sigma})}}
$$

**注释4**：信噪比与 $\sqrt{B}$ 成正比。如果保持 $\eta$ 不变增大 $B$，SNR会增加，可能导致训练过于确定性，损失泛化能力。

### 3. 线性缩放规律的SDE推导

**推导7：从离散到连续**

将SGD迭代改写为：

$$
\boldsymbol{w}_{t+1} - \boldsymbol{w}_t = -\eta\boldsymbol{g}_t - \eta(\tilde{\boldsymbol{g}}_{B,t} - \boldsymbol{g}_t)
$$

假设梯度噪声服从高斯分布 $\tilde{\boldsymbol{g}}_{B,t} - \boldsymbol{g}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_t/B)$。

**推导8：重参数化技巧**

利用重参数化技巧，设 $\boldsymbol{z}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$：

$$
\tilde{\boldsymbol{g}}_{B,t} - \boldsymbol{g}_t = \frac{1}{\sqrt{B}}\boldsymbol{\Sigma}_t^{1/2}\boldsymbol{z}_t
$$

其中 $\boldsymbol{\Sigma}_t^{1/2}$ 是协方差矩阵的平方根。代入更新公式：

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\boldsymbol{g}_t - \frac{\eta}{\sqrt{B}}\boldsymbol{\Sigma}_t^{1/2}\boldsymbol{z}_t
$$

**推导9：分离学习率的幂次**

关键步骤：将 $\eta$ 重写为 $\sqrt{\eta}\cdot\sqrt{\eta}$：

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\boldsymbol{g}_t - \sqrt{\eta}\cdot\frac{\sqrt{\eta}}{\sqrt{B}}\boldsymbol{\Sigma}_t^{1/2}\boldsymbol{z}_t
$$

**推导10：连续极限（SDE）**

当 $\eta \to 0$ 时，离散迭代趋向于随机微分方程：

$$
d\boldsymbol{w}_t = -\boldsymbol{g}_t dt - \sqrt{\frac{\eta}{B}}\boldsymbol{\Sigma}_t^{1/2}d\boldsymbol{W}_t
$$

其中 $d\boldsymbol{W}_t$ 是Wiener过程（布朗运动），满足 $d\boldsymbol{W}_t \sim \mathcal{N}(\boldsymbol{0}, dt\boldsymbol{I})$。

**注释5**：SDE的噪声项系数是 $\sqrt{\eta/B}$ 而非 $\eta/B$，这是因为布朗运动的方差与时间成正比。

**推导11：不变性条件**

要使SDE在改变 $B$ 时保持不变，需要：

$$
\frac{\eta_1}{B_1} = \frac{\eta_2}{B_2}
$$

因此：

$$
\frac{\eta_2}{\eta_1} = \frac{B_2}{B_1}
$$

**注释6**：这就是线性缩放规律！与平方根缩放的区别在于，SDE考虑了噪声累积的长期效应。

### 4. 损失函数的二阶近似分析

**推导12：泰勒展开**

设当前参数为 $\boldsymbol{w}$，更新后为 $\boldsymbol{w}' = \boldsymbol{w} - \eta\tilde{\boldsymbol{g}}_B$。损失函数的二阶泰勒展开：

$$
\mathcal{L}(\boldsymbol{w}') = \mathcal{L}(\boldsymbol{w}) + \nabla\mathcal{L}(\boldsymbol{w})^T(\boldsymbol{w}'-\boldsymbol{w}) + \frac{1}{2}(\boldsymbol{w}'-\boldsymbol{w})^T\boldsymbol{H}(\boldsymbol{w}'-\boldsymbol{w}) + O(\|\boldsymbol{w}'-\boldsymbol{w}\|^3)
$$

其中 $\boldsymbol{H} = \nabla^2\mathcal{L}(\boldsymbol{w})$ 是Hessian矩阵。

**推导13：代入更新公式**

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{w}') &= \mathcal{L}(\boldsymbol{w}) + \boldsymbol{g}^T(-\eta\tilde{\boldsymbol{g}}_B) + \frac{1}{2}(-\eta\tilde{\boldsymbol{g}}_B)^T\boldsymbol{H}(-\eta\tilde{\boldsymbol{g}}_B) \\
&= \mathcal{L}(\boldsymbol{w}) - \eta\boldsymbol{g}^T\tilde{\boldsymbol{g}}_B + \frac{\eta^2}{2}\tilde{\boldsymbol{g}}_B^T\boldsymbol{H}\tilde{\boldsymbol{g}}_B
\end{aligned}
$$

**推导14：取期望**

$$
\mathbb{E}[\mathcal{L}(\boldsymbol{w}')] = \mathcal{L}(\boldsymbol{w}) - \eta\boldsymbol{g}^T\mathbb{E}[\tilde{\boldsymbol{g}}_B] + \frac{\eta^2}{2}\mathbb{E}[\tilde{\boldsymbol{g}}_B^T\boldsymbol{H}\tilde{\boldsymbol{g}}_B]
$$

**推导15：二阶矩的展开**

利用 $\mathbb{E}[\tilde{\boldsymbol{g}}_B] = \boldsymbol{g}$ 和 $\mathbb{E}[\tilde{\boldsymbol{g}}_B\tilde{\boldsymbol{g}}_B^T] = \boldsymbol{g}\boldsymbol{g}^T + \boldsymbol{\Sigma}/B$：

$$
\begin{aligned}
\mathbb{E}[\tilde{\boldsymbol{g}}_B^T\boldsymbol{H}\tilde{\boldsymbol{g}}_B] &= \mathbb{E}[\text{Tr}(\tilde{\boldsymbol{g}}_B^T\boldsymbol{H}\tilde{\boldsymbol{g}}_B)] = \text{Tr}(\boldsymbol{H}\mathbb{E}[\tilde{\boldsymbol{g}}_B\tilde{\boldsymbol{g}}_B^T]) \\
&= \text{Tr}(\boldsymbol{H}(\boldsymbol{g}\boldsymbol{g}^T + \boldsymbol{\Sigma}/B)) \\
&= \boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g} + \frac{1}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})
\end{aligned}
$$

**推导16：最终的期望损失**

$$
\mathbb{E}[\mathcal{L}(\boldsymbol{w}')] = \mathcal{L}(\boldsymbol{w}) - \eta\|\boldsymbol{g}\|^2 + \frac{\eta^2}{2}\left(\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g} + \frac{1}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})\right)
$$

**注释7**：第一项是损失减少（我们希望最大化），第二项是二阶修正（通常为正，限制了学习率）。

### 5. 最优学习率的精确推导

**推导17：一阶最优性条件**

对 $\eta$ 求导并令其为零：

$$
\frac{d}{d\eta}\mathbb{E}[\mathcal{L}(\boldsymbol{w}')] = -\|\boldsymbol{g}\|^2 + \eta\left(\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g} + \frac{1}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})\right) = 0
$$

解得最优学习率：

$$
\eta^* = \frac{\|\boldsymbol{g}\|^2}{\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g} + \frac{1}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})}
$$

**推导18：引入噪声批量大小**

定义两个关键量：

$$
\begin{aligned}
\eta_{\max} &= \frac{\|\boldsymbol{g}\|^2}{\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g}} \\
\mathcal{B}_{\text{noise}} &= \frac{\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})}{\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g}}
\end{aligned}
$$

**注释8**：$\eta_{\max}$ 是全量梯度下降（$B\to\infty$）的最优学习率；$\mathcal{B}_{\text{noise}}$ 是一个特征Batch Size。

**推导19：统一形式**

将最优学习率重写为：

$$
\eta^* = \frac{\eta_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}
$$

**推导20：小Batch极限**

当 $B \ll \mathcal{B}_{\text{noise}}$ 时：

$$
\eta^* \approx \frac{\eta_{\max}B}{\mathcal{B}_{\text{noise}}} \propto B
$$

这就是线性缩放规律的适用范围。

**推导21：大Batch极限**

当 $B \gg \mathcal{B}_{\text{noise}}$ 时：

$$
\eta^* \approx \eta_{\max}
$$

此时学习率趋于饱和，继续增大Batch Size不能再提高学习率。

### 6. 噪声尺度理论

**推导22：噪声对训练的影响**

定义噪声尺度（noise scale）：

$$
g_{\text{noise}} = \frac{\eta}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})
$$

这个量衡量了噪声对损失函数曲率的影响。

**推导23：简化假设下的噪声尺度**

假设 $\boldsymbol{H} \approx \lambda\boldsymbol{I}$（各向同性近似），则：

$$
g_{\text{noise}} = \frac{\eta\lambda}{B}\text{Tr}(\boldsymbol{\Sigma}) = \frac{\eta\lambda}{B}\sum_i \sigma_i^2
$$

**推导24：简化的噪声批量大小**

定义简化版本：

$$
\mathcal{B}_{\text{simple}} = \frac{\text{Tr}(\boldsymbol{\Sigma})}{\|\boldsymbol{g}\|^2}
$$

与完整版的关系：

$$
\mathcal{B}_{\text{noise}} = \frac{\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})}{\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g}} \approx \mathcal{B}_{\text{simple}}
$$

当Hessian矩阵接近各向同性时成立。

**注释9**：$\mathcal{B}_{\text{simple}}$ 的优势是只需要计算梯度的方差，不需要Hessian矩阵。

### 7. 有效批量大小的概念

**推导25：训练步数与样本数的关系**

设完成训练需要 $S$ 步，每步使用 $B$ 个样本，则总样本数：

$$
E = B \cdot S
$$

**推导26：损失下降速率**

每步的期望损失下降：

$$
\Delta\mathcal{L} = \mathcal{L}(\boldsymbol{w}) - \mathbb{E}[\mathcal{L}(\boldsymbol{w}')] = \eta\|\boldsymbol{g}\|^2 - \frac{\eta^2}{2}\left(\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g} + \frac{1}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma})\right)
$$

在最优学习率下：

$$
\Delta\mathcal{L}^* = \frac{(\|\boldsymbol{g}\|^2)^2}{2(\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g} + \frac{1}{B}\text{Tr}(\boldsymbol{H}\boldsymbol{\Sigma}))} = \frac{\Delta\mathcal{L}_{\max}}{1 + \mathcal{B}_{\text{noise}}/B}
$$

其中 $\Delta\mathcal{L}_{\max} = \frac{(\|\boldsymbol{g}\|^2)^2}{2\boldsymbol{g}^T\boldsymbol{H}\boldsymbol{g}}$。

**推导27：达到目标的最少步数**

当 $B\to\infty$（全量梯度）时，每步下降最大，需要最少步数 $S_{\min}$。有限 $B$ 时需要 $S$ 步，满足：

$$
S \cdot \Delta\mathcal{L}^* = S_{\min} \cdot \Delta\mathcal{L}_{\max}
$$

因此：

$$
S = S_{\min}(1 + \mathcal{B}_{\text{noise}}/B)
$$

**推导28：总样本数**

$$
E = BS = S_{\min}(B + \mathcal{B}_{\text{noise}})
$$

当 $B \to 0$ 时，$E_{\min} = S_{\min}\mathcal{B}_{\text{noise}}$。

**推导29：数据效率与计算效率的权衡**

定义归一化量：

$$
\tilde{S} = \frac{S}{S_{\min}}, \quad \tilde{E} = \frac{E}{E_{\min}}
$$

则：

$$
(\tilde{S} - 1)(\tilde{E} - 1) = 1
$$

**注释10**：这个双曲线关系表明，数据效率和计算效率不可兼得——要减少总样本数就要增加训练步数。

### 8. 泛化差距的理论分析

**推导30：大Batch的泛化问题**

定义训练损失和测试损失的差距：

$$
\text{Gap} = \mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}}
$$

**推导31：噪声正则化效应**

SGD的噪声可以看作隐式正则化。噪声协方差矩阵：

$$
\boldsymbol{C}_{\text{noise}} = \frac{\eta^2\boldsymbol{\Sigma}}{B}
$$

噪声强度：

$$
\|\boldsymbol{C}_{\text{noise}}\|_F = \frac{\eta^2}{B}\|\boldsymbol{\Sigma}\|_F
$$

**推导32：平坦最小值假说**

大噪声倾向于找到平坦（Hessian特征值小）的最小值，而平坦最小值通常泛化性能更好。用Hessian的迹衡量"尖锐度"：

$$
\text{Sharpness} = \text{Tr}(\boldsymbol{H})
$$

噪声帮助逃离高sharpness区域。

**推导33：SDE中的有效温度**

在SDE框架下，可以定义"温度"：

$$
T_{\text{eff}} = \frac{\eta}{B}
$$

高温（大 $\eta/B$）导致更广泛的探索，可能改善泛化。

### 9. 预热机制的数学原理

**推导34：初始阶段的不稳定性**

训练开始时，梯度通常很大且不稳定。如果立即使用大学习率：

$$
\|\boldsymbol{w}_1 - \boldsymbol{w}_0\| = \eta\|\tilde{\boldsymbol{g}}_0\|
$$

可能非常大，超出二阶近似的有效范围。

**推导35：线性预热策略**

定义预热步数 $T_{\text{warmup}}$，在前 $T_{\text{warmup}}$ 步内学习率线性增长：

$$
\eta_t = \frac{t}{T_{\text{warmup}}}\eta_{\text{target}}, \quad t \leq T_{\text{warmup}}
$$

**推导36：预热的稳定性分析**

小学习率阶段的更新：

$$
\boldsymbol{w}_t = \boldsymbol{w}_0 - \sum_{s=0}^{t-1}\frac{s}{T_{\text{warmup}}}\eta_{\text{target}}\tilde{\boldsymbol{g}}_s
$$

总位移：

$$
\|\boldsymbol{w}_t - \boldsymbol{w}_0\| \lesssim \frac{t^2}{2T_{\text{warmup}}}\eta_{\text{target}}\mathbb{E}[\|\tilde{\boldsymbol{g}}\|]
$$

增长是二次的而非线性的，更加平稳。

**推导37：最优预热步数**

一种启发式选择：让预热期间的总位移与一步大学习率更新的位移相当：

$$
\frac{T_{\text{warmup}}^2}{2T_{\text{warmup}}}\eta_{\text{target}}\|\boldsymbol{g}\| \approx \eta_{\text{target}}\|\boldsymbol{g}\|
$$

解得 $T_{\text{warmup}} \approx 2$，但实践中通常取几百到几千步。

### 10. Adam优化器的Batch Size缩放

**推导38：Adam的更新方向**

Adam的更新可以写为：

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\boldsymbol{\varphi}_t, \quad \boldsymbol{\varphi}_t = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}
$$

**推导39：符号梯度近似**

在 $\epsilon$ 较小时，Adam近似于符号梯度：

$$
\varphi_{t,i} \approx \text{sign}(m_{t,i})
$$

**推导40：符号梯度的期望（回顾）**

$$
\mathbb{E}[\text{sign}(\tilde{g}_B)] = \text{erf}\left(\frac{g}{\sigma}\sqrt{\frac{B}{2}}\right)
$$

**推导41：Adam的平方根缩放**

对于Adam，最优学习率的形式为：

$$
\eta^*_{\text{Adam}} \approx \frac{\eta_{\max,\text{Adam}}}{(1 + \mathcal{B}_{\text{noise}}/B)^{\alpha}}
$$

其中 $0.5 < \alpha < 1$。OpenAI的经验建议 $\alpha \approx 0.5$，即平方根缩放：

$$
\eta^*_{\text{Adam}} \propto \sqrt{B} \quad \text{when } B \ll \mathcal{B}_{\text{noise}}
$$

**推导42：Adam与SGD的对比**

- SGD：$\eta \propto B$（线性）
- Adam：$\eta \propto \sqrt{B}$（平方根）

原因是Adam的自适应缩放改变了噪声的累积方式。

### 11. 实际训练的复杂性

**推导43：动态的Batch Size调整**

在训练过程中，$\mathcal{B}_{\text{noise}}$ 是动态变化的：

$$
\mathcal{B}_{\text{noise}}(t) = \frac{\text{Tr}(\boldsymbol{H}_t\boldsymbol{\Sigma}_t)}{\boldsymbol{g}_t^T\boldsymbol{H}_t\boldsymbol{g}_t}
$$

通常在训练后期，$\|\boldsymbol{g}_t\|$ 减小，$\mathcal{B}_{\text{noise}}(t)$ 可能增大。

**推导44：自适应Batch Size策略**

理想情况下，可以根据估计的 $\mathcal{B}_{\text{noise}}(t)$ 动态调整 $B(t)$：

$$
B(t) = \min\left(B_{\max}, \alpha \mathcal{B}_{\text{noise}}(t)\right)
$$

其中 $\alpha$ 是一个常数（如0.1-1）。

### 12. 分布式训练的通信开销

**推导45：同步时间模型**

在数据并行训练中，每步的时间包括：

$$
T_{\text{step}} = T_{\text{compute}} + T_{\text{comm}}
$$

计算时间与Batch Size无关（假设每个worker的local batch固定）：

$$
T_{\text{compute}} = c_1
$$

通信时间（All-Reduce）依赖于模型大小 $P$ 和worker数量 $W$：

$$
T_{\text{comm}} = c_2 P \log W
$$

**推导46：总训练时间**

总时间：

$$
T_{\text{total}} = S(B) \cdot (T_{\text{compute}} + T_{\text{comm}})
$$

其中 $S(B) = S_{\min}(1 + \mathcal{B}_{\text{noise}}/B)$。

**推导47：最优Batch Size（考虑通信）**

最小化总时间不再等价于最大化Batch Size。定义效率：

$$
\text{Efficiency} = \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{comm}}}
$$

当效率低于某个阈值（如0.9）时，增大Batch Size的收益有限。

### 13. 临界Batch Size的估计

**推导48：实验拟合方法**

进行多组实验，记录不同 $(B, S, E)$ 的组合。根据关系：

$$
S = S_{\min}\left(1 + \frac{\mathcal{B}_{\text{noise}}}{B}\right)
$$

拟合得到 $S_{\min}$ 和 $\mathcal{B}_{\text{noise}}$。

**推导49：梯度方差估计**

在数据并行中，每个worker计算local batch的梯度 $\boldsymbol{g}_i$。在All-Reduce前，可以计算：

$$
\bar{\boldsymbol{g}} = \frac{1}{W}\sum_{i=1}^W \boldsymbol{g}_i, \quad \hat{\boldsymbol{\Sigma}} = \frac{1}{W-1}\sum_{i=1}^W (\boldsymbol{g}_i - \bar{\boldsymbol{g}})(\boldsymbol{g}_i - \bar{\boldsymbol{g}})^T
$$

然后估计：

$$
\mathcal{B}_{\text{simple}} \approx \frac{\text{Tr}(\hat{\boldsymbol{\Sigma}})}{\|\bar{\boldsymbol{g}}\|^2}
$$

**注释11**：这个估计只需要额外的一点通信和计算，可以在线监控。

### 14. 学习率调度策略

**推导50：余弦退火的数学形式**

常用的余弦退火：

$$
\eta_t = \eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2}\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

**推导51：退火与Batch Size的耦合**

在训练后期，梯度变小，$\mathcal{B}_{\text{noise}}$ 可能增大。如果同时进行学习率退火：

$$
\eta_t = \frac{\eta_{\max,t}}{1 + \mathcal{B}_{\text{noise}}(t)/B}
$$

其中 $\eta_{\max,t}$ 本身在减小。

**推导52：渐进式增大Batch Size**

另一种策略是固定学习率，逐渐增大Batch Size：

$$
B_t = B_0 \cdot r^t, \quad r > 1
$$

这在某些情况下可以替代学习率退火。

### 15. 高级主题：二阶优化方法

**推导53：牛顿法的Batch Size依赖**

牛顿法的更新：

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\boldsymbol{H}^{-1}\tilde{\boldsymbol{g}}_B
$$

更新量的方差：

$$
\text{Var}[\Delta\boldsymbol{w}] = \eta^2\boldsymbol{H}^{-1}\frac{\boldsymbol{\Sigma}}{B}\boldsymbol{H}^{-1}
$$

**推导54：自然梯度的缩放**

自然梯度使用Fisher信息矩阵 $\boldsymbol{F}$：

$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta\boldsymbol{F}^{-1}\tilde{\boldsymbol{g}}_B
$$

Fisher矩阵本身依赖于Batch Size的估计，使得缩放关系更加复杂。

### 16. 理论极限与实践差距

**推导55：理论最优与实际最优**

理论分析基于：
1. 二阶近似有效
2. Hessian矩阵稳定
3. 梯度噪声高斯分布

实际训练中这些假设可能不成立，导致：

$$
\eta_{\text{practice}}^* \neq \eta_{\text{theory}}^*
$$

**推导56：安全系数**

实践中通常使用理论值的一个比例：

$$
\eta_{\text{practice}} = \alpha \eta_{\text{theory}}, \quad 0.5 < \alpha < 1
$$

### 总结

本节系统推导了Batch Size与学习率之间的缩放关系，主要结论：

1. **平方根缩放**：从方差不变性导出 $\eta \propto \sqrt{B}$
2. **线性缩放**：从SDE连续极限导出 $\eta \propto B$（小Batch Size下）
3. **饱和效应**：存在临界Batch Size $\mathcal{B}_{\text{noise}}$，超过后学习率不应继续增大
4. **Adam的特殊性**：遵循平方根缩放而非线性缩放
5. **数据-计算权衡**：$(\tilde{S}-1)(\tilde{E}-1)=1$ 描述了效率的基本限制

这些理论为大规模分布式训练提供了重要指导。

