---
title: Muon续集：为什么我们选择尝试Muon？
slug: muon续集为什么我们选择尝试muon
date: 2025-02-27
tags: 矩阵, 梯度, 优化器, 谱范数, muon
status: pending
---

# Muon续集：为什么我们选择尝试Muon？

**原文链接**: [https://spaces.ac.cn/archives/10739](https://spaces.ac.cn/archives/10739)

**发布日期**: 

---

本文解读一下我们最新的技术报告[《Muon is Scalable for LLM Training》](https://papers.cool/arxiv/2502.16982)，里边分享了我们之前在[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)介绍过的Muon优化器的一次较大规模的实践，并开源了相应的模型（我们称之为“[Moonlight](https://github.com/MoonshotAI/Moonlight)”，目前是一个3B/16B的MoE模型）。我们发现了一个比较惊人的结论：在我们的实验设置下，Muon相比Adam能够达到将近2倍的训练效率。

[![Muon的Scaling Law及Moonlight的MMLU表现](/usr/uploads/2025/02/1300601661.png)](/usr/uploads/2025/02/1300601661.png "点击查看原图")

Muon的Scaling Law及Moonlight的MMLU表现

优化器的工作说多不多，但说少也不少，为什么我们会选择Muon来作为新的尝试方向呢？已经调好超参的Adam优化器，怎么快速切换到Muon上进行尝试呢？模型Scale上去之后，Muon与Adam的性能效果差异如何？接下来将分享我们的思考过程。

## 优化原理 #

关于优化器，其实笔者之前在[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)时就有过浅评，多数优化器改进实际上都只是一些小补丁，不能说毫无价值，但终究没能给人一种深刻和惊艳的感觉。

我们需要从更贴近本质的原理出发，来思考什么才是好的优化器。直观来想，理想的优化器应当有两个特性：**稳** 和**快** 。具体来说，理想的优化器每一步的更新应该满足两点：1、对模型扰动尽可能小；2、对Loss贡献尽可能大。说更直接点，就是我们不希望大改模型（稳），但希望大降Loss（快），典型的“既要...又要...”。

怎么将这两个特性转化为数学语言呢？**稳** 我们可以理解为对更新量的一个约束，而**快** 则可以理解为寻找让损失函数下降最快的更新量，所以这可以转化为一个约束优化问题。用回前文的记号，对于矩阵参数$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，其梯度为$\boldsymbol{G}\in\mathbb{R}^{n\times m}$，当参数由$\boldsymbol{W}$变成$\boldsymbol{W}+\Delta\boldsymbol{W}$时，损失函数的变化量为  
\begin{equation}\text{Tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})\end{equation}  
那么在**稳** 的前提下寻找**快** 的更新量，那么就可以表示为  
\begin{equation}\mathop{\text{argmin}}_{\Delta\boldsymbol{W}}\text{Tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})\quad\text{s.t.}\quad \rho(\Delta\boldsymbol{W})\leq \eta\label{eq:least-action}\end{equation}  
这里$\rho(\Delta\boldsymbol{W})\geq 0$是**稳** 的某个指标，越小表示越稳，而$\eta$是某个小于1的常数，表示我们对**稳** 的要求，后面我们会看到它实际上就是优化器的学习率。如果读者不介意，我们可以模仿理论物理的概念，称上述原理为优化器的“**最小作用量原理（Least Action Principle）** ”。

## 矩阵范数 #

式$\eqref{eq:least-action}$唯一不确定的就是**稳** 的度量$\rho(\Delta\boldsymbol{W})$，选定$\rho(\Delta\boldsymbol{W})$后就可以把$\Delta\boldsymbol{W}$明确求解出来（至少理论上没问题）。某种程度上来说，我们可以认为不同优化器的本质差异就是它们对**稳** 的定义不一样。

很多读者在刚学习SGD时想必都看到过类似“梯度反方向是函数值局部下降最快的方向”的说法，放到这里的框架来看，它其实就是把**稳** 的度量选择为矩阵的$F$范数$\Vert\Delta\boldsymbol{W}\Vert_F$，也就是说，“下降最快的方向”并不是一成不变的，选定度量后才能把它确定下来，换一个范数就不一定是梯度反方向了。

接下来的问题自然是什么范数才能最恰当地度量**稳** ？如果我们加了强约束，那么稳则稳矣，但优化器举步维艰，只能收敛到次优解；相反如果减弱约束，那么优化器放飞自我，那么训练进程将会极度不可控。所以，最理想的情况就能找到**稳** 的最精准指标。考虑到神经网络以矩阵乘法为主，我们以$\boldsymbol{y}=\boldsymbol{x}\boldsymbol{W}$为例，有  
\begin{equation}\Vert\Delta \boldsymbol{y}\Vert = \Vert\boldsymbol{x}(\boldsymbol{W} + \Delta\boldsymbol{W}) - \boldsymbol{x}\boldsymbol{W}\Vert = \Vert\boldsymbol{x} \Delta\boldsymbol{W}\Vert\leq \rho(\Delta\boldsymbol{W}) \Vert\boldsymbol{x}\Vert\end{equation}  
上式的意思是，当参数由$\boldsymbol{W}$变成$\boldsymbol{W}+\Delta\boldsymbol{W}$时，模型输出变化量为$\Delta\boldsymbol{y}$，我们寄望于这个变化量的模长能够被$\Vert\boldsymbol{x}\Vert$以及$\Delta\boldsymbol{W}$相关的一个函数$\rho(\Delta\boldsymbol{W})$控制，我们就用这个函数作为**稳** 的指标。从线性代数我们知道，$\rho(\Delta\boldsymbol{W})$的最准确值就是$\Delta\boldsymbol{W}$的谱范数$\Vert\Delta\boldsymbol{W}\Vert_2$，代入式$\eqref{eq:least-action}$得到  
\begin{equation}\mathop{\text{argmin}}_{\Delta\boldsymbol{W}}\text{Tr}(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})\quad\text{s.t.}\quad \Vert\Delta\boldsymbol{W}\Vert_2\leq \eta\end{equation}  
这个优化问题求解出来就是$\beta=0$的Muon：  
\begin{equation}\Delta\boldsymbol{W} = -\eta\, \text{msign}(\boldsymbol{G}) = -\eta\,\boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}, \quad \boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}^{\top} = \mathop{\text{SVD}}(\boldsymbol{G})\end{equation}  
当$\beta > 0$时，$\boldsymbol{G}$换成动量$\boldsymbol{M}$，$\boldsymbol{M}$可以看作是对梯度更平滑的估计，所以依然可以理解为上式的结论，因此我们可以得出“Muon就是谱范数下的最速下降”的说法，至于Newton-schulz迭代之类的，则是计算上的近似，这里就不细说。详细推导我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)已经给出，也不再重复。

## 权重衰减 #

至此，我们可以回答第一个问题：为什么选择尝试Muon？因为跟SGD一样，Muon给出的同样是下降最快的方向，但它的谱范数约束比SGD的$F$范数更为精准，所以有更佳的潜力。另一方面，从“为不同的参数选择最恰当的约束”角度来改进优化器，看起来也比各种补丁式修改更为本质。

当然，潜力不意味着实力，在更大尺寸的模型上验证Muon存在一些“陷阱”。首先登场的是Weight Decay问题，尽管我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)介绍Muon的时候带上了Weight Decay，但实际上作者提出Muon时是没有的，而我们一开始也按照官方版本来实现，结果发现Muon前期收敛是快，但很快就被Adam追上了，而且各种“内科”还有崩溃的苗头。

我们很快意识到这可能是Weight Decay的问题，于是补上Weight Decay：  
\begin{equation}\Delta\boldsymbol{W} = -\eta\, [\text{msign}(\boldsymbol{M})+ \lambda \boldsymbol{W}]\end{equation}  
继续实验，果不其然，这时候Muon一直保持领先于Adam，如论文Figure 2所示：  


[![有无Weight Decay的效果比较](/usr/uploads/2025/02/3675708776.png)](/usr/uploads/2025/02/3675708776.png "点击查看原图")

有无Weight Decay的效果比较

Weight Decay起到什么作用呢？事后分析来看，可能比较关键的地方是能够让参数范数保持有界：  
\begin{equation}\begin{aligned}  
\Vert\boldsymbol{W}_t\Vert =&\, \Vert\boldsymbol{W}_{t-1} - \eta_t (\boldsymbol{\Phi}_t + \lambda \boldsymbol{W}_{t-1})\Vert \\\\[5pt]  
=&\, \Vert(1 - \eta_t \lambda)\boldsymbol{W}_{t-1} - \eta_t \lambda (\boldsymbol{\Phi}_t/\lambda)\Vert \\\\[5pt]  
\leq &\,(1 - \eta_t \lambda)\Vert\boldsymbol{W}_{t-1}\Vert + \eta_t \lambda \Vert\boldsymbol{\Phi}_t/\lambda\Vert \\\\[5pt]  
\leq &\,\max(\Vert\boldsymbol{W}_{t-1}\Vert,\Vert\boldsymbol{\Phi}_t/\lambda\Vert) \\\\[5pt]  
\end{aligned}\end{equation}  
这里的$\Vert\cdot\Vert$是任意一种矩阵范数，即上述不等式对于任意矩阵范数都是成立的，$\boldsymbol{\Phi}_t$是优化器给出的更新向量，对Muon来说是$\text{msign}(\boldsymbol{M})$，当我们取谱范数时，有$\Vert\text{msign}(\boldsymbol{M})\Vert_2 = 1$，所以对于Muon来说有  
\begin{equation}  
\Vert\boldsymbol{W}_t\Vert_2 \leq \max(\Vert\boldsymbol{W}_{t-1}\Vert_2,1/\lambda)\leq\cdots \leq \max(\Vert\boldsymbol{W}_0\Vert_2,1/\lambda)\end{equation}  
这保证了模型“内科”的健康，因为$\Vert\boldsymbol{x}\boldsymbol{W}\Vert\leq \Vert\boldsymbol{x}\Vert\Vert\boldsymbol{W}\Vert_2$，$\Vert\boldsymbol{W}\Vert_2$被控制住了，意味着$\Vert\boldsymbol{x}\boldsymbol{W}\Vert$也被控制住了，就不会有爆炸的风险，这对于Attention Logits爆炸等问题也尤其重要。当然这个上界在多数情况下还是相当宽松的，实际中参数的谱范数多数会明显小于这个上界，这个不等关系只是简单显示Weight Decay有控制范数的性质存在。

## RMS对齐 #

当我们决定去尝试新的优化器时，一个比较头疼的问题是如何快速找到接近最优的超参数，比如Muon至少有学习率$\eta_t$和衰减率$\lambda$两个超参数。网格搜索自然是可以，但比较费时费力，这里我们提出Update RMS对齐的超参迁移思路，可以将Adam调好的超参数用到其他优化器上。

首先，对于一个矩阵$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，它的RMS（Root Mean Square）定义为  
\begin{equation}\text{RMS}(\boldsymbol{W}) = \frac{\Vert \boldsymbol{W}\Vert_F}{\sqrt{nm}} = \sqrt{\frac{1}{nm}\sum_{i=1}^n\sum_{j=1}^m W_{i,j}^2}\end{equation}  
简单来说，RMS度量了矩阵每个元素的平均大小。我们观察到Adam更新量的RMS是比较稳定的，通常在0.2～0.4之间，这也是为什么[理论分析](/archives/10542)常用SignSGD作为Adam近似。基于此，我们建议通过RMS Norm将新优化器的Update RMS对齐到0.2：  
\begin{gather}  
\boldsymbol{W}_t =\boldsymbol{W}_{t-1} - \eta_t (\boldsymbol{\Phi}_t + \lambda \boldsymbol{W}_{t-1}) \\\\[6pt]  
\downarrow \notag\\\\[6pt]  
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t (0.2\, \boldsymbol{\Phi}_t/\text{RMS}(\boldsymbol{\Phi}_t) + \lambda \boldsymbol{W}_{t-1})  
\end{gather}  
这样一来，我们就可以复用Adam的$\eta_t$和$\lambda$，以达到每步对参数的更新幅度大致相同的效果。实践表明，通过这个简单策略从Adam迁移到Muon，就能训出明显优于Adam的效果，接近进一步对Muon超参进行精搜索的结果。特别地，Muon的$\text{RMS}(\boldsymbol{\Phi}_t)=\text{RMS}(\boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top})$还可以解析地算出来：  
\begin{equation}nm\,\text{RMS}(\boldsymbol{\Phi}_t)^2 = \sum_{i=1}^n\sum_{j=1}^m \sum_{k=1}^r U_{i,k}^2V_{k,j}^2 = \sum_{k=1}^r\left(\sum_{i=1}^n U_{i,k}^2\right)\left(\sum_{j=1}^m V_{k,j}^2\right) = \sum_{k=1}^r 1 = r\end{equation}  
即$\text{RMS}(\boldsymbol{\Phi}_t) = \sqrt{r/nm}$，实践中一个矩阵严格低秩的概率比较小，因此可以认为$r = \min(n,m)$，从而有$\text{RMS}(\boldsymbol{\Phi}_t) = \sqrt{1/\max(n,m)}$。所以我们最终没有用RMS Norm而是用等价的解析版本：  
\begin{equation}\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t (0.2\, \boldsymbol{\Phi}_t\,\sqrt{\max(n,m)} + \lambda \boldsymbol{W}_{t-1})\end{equation}  
最后的这个式子，表明了在Muon中不适宜所有参数使用同一个学习率。比如Moonlight是一个MoE模型，有不少矩阵参数的形状都偏离方阵，$\max(n,m)$跨度比较大，如果用单一学习率，必然会导致某些参数学习过快/过慢的不同步问题，从而影响最终效果。

## 实验分析 #

我们在2.4B/16B这个尺寸的MoE上，对Adam和Muon做了比较充分的对比，发现Muon在收敛速度和最终效果上都有明显的优势。详细的比较结果建议大家去看原论文，这里仅截取部分分享。

> **Github:<https://github.com/MoonshotAI/Moonlight>**

首先是一个相对客观的对照表格，包括我们自己控制变量训练的Muon和Adam的对比，以及与外界（DeepSeek）用Adam训练的同样架构的模型对比（为了便于对比，Moonlight的架构跟DSV3-Small完全一致），显示出Muon的独特优势：  


[![Muon（Moonlight） vs Adam（Moonlight-A 和DSV3-small）的比较](/usr/uploads/2025/02/2528279021.png)](/usr/uploads/2025/02/2528279021.png "点击查看原图")

Muon（Moonlight） vs Adam（Moonlight-A 和DSV3-small）的比较

Muon训练出来的模型有什么不同呢？既然前面我们说Muon是谱范数下的最速下降，谱范数是最大的奇异值，所以我们想到了监控和分析奇异值。果然，我们发现了一些有趣的信号，Muon训出来的参数，奇异值分布相对更均匀一些，我们使用奇异值熵来定量描述这个现象：  
\begin{equation}H(\boldsymbol{\sigma}) = -\frac{1}{\log n}\sum_{i=1}^n \frac{\sigma_i^2}{\sum_{j=1}^n\sigma_j^2}\log \frac{\sigma_i^2}{\sum_{j=1}^n\sigma_j^2}\end{equation}  
这里$\boldsymbol{\sigma}=(\sigma_1,\sigma_2,\cdots,\sigma_n)$是某个参数的全体奇异值。Muon训出来的参数熵更大，即奇异值分布更均匀，意味着这个参数越不容易压缩，这说明Muon更充分发挥了参数的潜能：  


[![Muon训练出来的权重奇异值熵更高](/usr/uploads/2025/02/3782823216.png)](/usr/uploads/2025/02/3782823216.png "点击查看原图")

Muon训练出来的权重奇异值熵更高

还有一个有趣的发现是当我们将Muon用于微调（SFT）时，可能会因为预训练没用Muon而得到次优解。具体来说，如果预训练和微调都用Muon，那么表现是最好的，但如果是另外三种组合（Adam+Muon、Muon+Adam、Adam+Adam），其效果优劣没有呈现于明显的规律。  


[![预训练/微调分别用Muon/Adam的组合测试](/usr/uploads/2025/02/2697758072.png)](/usr/uploads/2025/02/2697758072.png "点击查看原图")

预训练/微调分别用Muon/Adam的组合测试

[![在开源模型上用Muon/Adam微调的尝试](/usr/uploads/2025/02/3565677851.png)](/usr/uploads/2025/02/3565677851.png "点击查看原图")

在开源模型上用Muon/Adam微调的尝试

这个现象表明存在一些特殊的初始化对Muon不利，当然反过来也可能存在一些初始化对Muon更有利，更底层的原理我们还在探索中。

## 拓展思考 #

总的来说，在我们的实验里，Muon的表现跟Adam相比显得非常有竞争力。作为一个形式上跟Adam差异较大的新优化器，Muon的这个表现其实不单单是“可圈可点”了，还表明它可能捕捉到了一些本质的特性。

此前，社区流传着一个观点：Adam之所以表现好，是因为主流的模型架构改进都在“过拟合”Adam。这个观点最早应该出自[《Neural Networks (Maybe) Evolved to Make Adam The Best Optimizer》](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/)，看上去有点荒谬，但实际上意蕴深长。试想一下，当我们尝试改进模型后，就会拿Adam训一遍看效果，效果好就保留，否则放弃。可这个效果好，究竟是因为它本质更佳，还是因为它更匹配Adam了呢？

这就有点耐人寻味了。当然不说全部，肯定至少有一部份工作，是因为它跟Adam更配而表现出更好的效果，所以久而久之，模型架构就会朝着一个有利于Adam的方向演进。在这个背景下，一个跟Adam显著不同的优化器还能“出圈”，就尤其值得关注和思考了。注意笔者和所在公司都不属Muon提出者，所以这番言论纯属“肺腑之言”，并不存在自卖自夸的意思。

接下来Muon还有什么工作可做呢？其实应该还有不少。比如上面提到的“Adam预训练+Muon微调”效果不佳问题，进一步分析还是有必要和有价值的，毕竟现在大家开源的模型权重基本都是Adam训的，如果Muon微调不行，必然也影响它的普及。当然了，我们还可以借着这个契机进一步深化对Muon理解（面向Bug学习）。

还有一个推广的思考，就是Muon基于谱范数，谱范数是最大奇异值，事实上基于奇异值我们还可以构造一系列范数，如[Schatten范数](https://en.wikipedia.org/wiki/Schatten_norm)，将它推广到这种更广义的范数然后进行调参，理论上还有机会取得更好的效果。此外，Moonlight发布后，还有一些读者问到Muon下[µP（maximal update parametrization）](https://papers.cool/arxiv/2203.03466)如何设计，这也是一个亟待解决的问题。

## 文章小结 #

本文介绍了我们在Muon优化器上的一次较大规模实践（Moonlight），并分享了我们对Muon优化器的最新思考。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10739>_

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

苏剑林. (Feb. 27, 2025). 《Muon续集：为什么我们选择尝试Muon？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10739>

@online{kexuefm-10739,  
title={Muon续集：为什么我们选择尝试Muon？},  
author={苏剑林},  
year={2025},  
month={Feb},  
url={\url{https://spaces.ac.cn/archives/10739}},  
} 


---

## 公式推导与注释

本节提供文章中核心概念的详细数学推导，从优化理论、线性代数和实验分析三个角度深入理解Muon优化器的设计原理。

### 1. Adam优化器的尺度不一致性分析

#### 1.1 Adam的更新规则回顾

Adam优化器对参数$\boldsymbol{W} \in \mathbb{R}^{n \times m}$的更新规则为：

$$
\begin{equation}
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1) \boldsymbol{G}_t \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) \boldsymbol{G}_t \odot \boldsymbol{G}_t \\
\hat{\boldsymbol{m}}_t &= \frac{\boldsymbol{m}_t}{1-\beta_1^t} \\
\hat{\boldsymbol{v}}_t &= \frac{\boldsymbol{v}_t}{1-\beta_2^t} \\
\boldsymbol{W}_{t+1} &= \boldsymbol{W}_t - \alpha_t \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}
\end{aligned}
\end{equation}
$$

其中$\boldsymbol{G}_t$是梯度，$\odot$表示逐元素乘法（Hadamard积），$\sqrt{\cdot}$和除法也都是逐元素操作。

#### 1.2 参数维度的尺度问题

考虑两个不同形状的权重矩阵：
- $\boldsymbol{W}_1 \in \mathbb{R}^{1024 \times 1024}$（隐藏层）
- $\boldsymbol{W}_2 \in \mathbb{R}^{1024 \times 32000}$（输出层）

对于Adam，每个参数元素的更新量约为：

$$
\Delta W_{ij} \approx \alpha \frac{m_{ij}}{\sqrt{v_{ij}} + \epsilon} \approx \alpha \cdot \text{sign}(G_{ij})
$$

当梯度稳定后，$\sqrt{v_{ij}}$近似为$|m_{ij}|$，因此Adam的更新近似于SignSGD。这意味着每个元素的更新幅度相似，但对整个矩阵的影响却不同。

定义矩阵更新的有效尺度为Frobenius范数：

$$
\Vert \Delta \boldsymbol{W}_1 \Vert_F \approx \alpha \sqrt{\sum_{i,j} 1} = \alpha \sqrt{1024 \times 1024} = 1024\alpha
$$

$$
\Vert \Delta \boldsymbol{W}_2 \Vert_F \approx \alpha \sqrt{1024 \times 32000} \approx 5702\alpha
$$

可见，尽管使用相同的学习率$\alpha$，但$\boldsymbol{W}_2$的更新幅度是$\boldsymbol{W}_1$的约5.6倍！

#### 1.3 输出扰动的不一致性

更严重的问题在于对输出的影响。考虑前向传播$\boldsymbol{y} = \boldsymbol{x} \boldsymbol{W}$，当$\boldsymbol{W} \to \boldsymbol{W} + \Delta \boldsymbol{W}$时：

$$
\Vert \Delta \boldsymbol{y} \Vert = \Vert \boldsymbol{x} \Delta \boldsymbol{W} \Vert
$$

对于Adam的更新，假设$\Vert \boldsymbol{x} \Vert = 1$（经过Layer Norm后的典型情况）：

$$
\Vert \Delta \boldsymbol{y} \Vert \leq \Vert \boldsymbol{x} \Vert \cdot \Vert \Delta \boldsymbol{W} \Vert_2
$$

这里关键是谱范数$\Vert \Delta \boldsymbol{W} \Vert_2$（矩阵的最大奇异值）。对于Adam的逐元素更新：

$$
\Vert \Delta \boldsymbol{W} \Vert_2 \leq \Vert \Delta \boldsymbol{W} \Vert_F = O(\alpha \sqrt{nm})
$$

但这个上界非常松弛！实际上，当$\Delta \boldsymbol{W}$的元素是独立同分布的随机变量时，由随机矩阵理论：

$$
\Vert \Delta \boldsymbol{W} \Vert_2 \approx \alpha \sqrt{\max(n,m)}
$$

这比$F$范数的尺度$\alpha\sqrt{nm}$要小，但仍然依赖于矩阵维度。更重要的是，Adam无法直接控制这个量，导致不同形状矩阵的输出扰动不可控。

### 2. 梯度的矩阵结构深度分析

#### 2.1 奇异值分解的意义

对梯度矩阵$\boldsymbol{G} \in \mathbb{R}^{n \times m}$进行奇异值分解（SVD）：

$$
\boldsymbol{G} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}
$$

其中：
- $\boldsymbol{U} = [\boldsymbol{u}_1, \ldots, \boldsymbol{u}_n] \in \mathbb{R}^{n \times n}$，$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}$
- $\boldsymbol{V} = [\boldsymbol{v}_1, \ldots, \boldsymbol{v}_m] \in \mathbb{R}^{m \times m}$，$\boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}$
- $\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \ldots, \sigma_r)$，$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $r = \text{rank}(\boldsymbol{G}) \leq \min(n,m)$

这个分解的几何意义：
1. $\boldsymbol{u}_i$是输入空间的正交基
2. $\boldsymbol{v}_i$是输出空间的正交基
3. $\sigma_i$是第$i$个主方向的重要性权重

#### 2.2 梯度的秩结构

在深度学习中，梯度矩阵通常具有低秩或近似低秩结构。考虑反向传播：

$$
\boldsymbol{G} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = \boldsymbol{x}^{\top} \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}
$$

对于单个样本，$\boldsymbol{x} \in \mathbb{R}^{n \times 1}$和$\frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \in \mathbb{R}^{m \times 1}$都是向量，因此：

$$
\text{rank}(\boldsymbol{G}) = 1
$$

对于批量大小为$B$的mini-batch：

$$
\boldsymbol{G} = \frac{1}{B} \sum_{i=1}^B \boldsymbol{x}_i^{\top} \frac{\partial \mathcal{L}_i}{\partial \boldsymbol{y}_i}
$$

理论上$\text{rank}(\boldsymbol{G}) \leq \min(B, n, m)$，实际中由于样本相关性，有效秩往往远小于批量大小。

#### 2.3 奇异值的衰减规律

实验观察表明，神经网络梯度的奇异值通常呈现快速衰减：

$$
\sigma_i \approx \sigma_1 \cdot i^{-\alpha}, \quad \alpha \in [1, 2]
$$

这意味着前几个奇异值包含了梯度的主要信息。定义有效秩：

$$
r_{\text{eff}} = \frac{(\sum_{i=1}^r \sigma_i)^2}{\sum_{i=1}^r \sigma_i^2}
$$

通常$r_{\text{eff}} \ll r = \min(n,m)$，这为低秩近似提供了理论基础。

### 3. 谱范数约束的最优性理论

#### 3.1 从扰动分析到谱范数

我们要找到一个范数$\rho(\cdot)$，使得：

$$
\Vert \boldsymbol{x} \boldsymbol{W} \Vert \leq \rho(\boldsymbol{W}) \Vert \boldsymbol{x} \Vert, \quad \forall \boldsymbol{x}
$$

且这个界是紧的（tight）。根据诱导范数的定义：

$$
\Vert \boldsymbol{W} \Vert_2 = \sup_{\boldsymbol{x} \neq 0} \frac{\Vert \boldsymbol{W} \boldsymbol{x} \Vert_2}{\Vert \boldsymbol{x} \Vert_2}
$$

由矩阵理论，谱范数等于最大奇异值：

$$
\Vert \boldsymbol{W} \Vert_2 = \sigma_1(\boldsymbol{W}) = \sigma_{\max}(\boldsymbol{W})
$$

**定理1（谱范数的紧性）**：对任意矩阵$\boldsymbol{W} \in \mathbb{R}^{n \times m}$和向量$\boldsymbol{x} \in \mathbb{R}^{n}$：

$$
\Vert \boldsymbol{W} \boldsymbol{x} \Vert_2 \leq \Vert \boldsymbol{W} \Vert_2 \Vert \boldsymbol{x} \Vert_2
$$

且等号成立当且仅当$\boldsymbol{x} \propto \boldsymbol{v}_1$（最大奇异值对应的右奇异向量）。

**证明**：设$\boldsymbol{W} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$，则：

$$
\begin{aligned}
\Vert \boldsymbol{W} \boldsymbol{x} \Vert_2^2 &= \left\Vert \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} \boldsymbol{x} \right\Vert_2^2 \\
&= \sum_{i=1}^r \sigma_i^2 (\boldsymbol{v}_i^{\top} \boldsymbol{x})^2 \quad \text{(因为}\boldsymbol{u}_i\text{正交)} \\
&\leq \sigma_1^2 \sum_{i=1}^r (\boldsymbol{v}_i^{\top} \boldsymbol{x})^2 \\
&\leq \sigma_1^2 \sum_{i=1}^m (\boldsymbol{v}_i^{\top} \boldsymbol{x})^2 \\
&= \sigma_1^2 \Vert \boldsymbol{x} \Vert_2^2 \quad \text{(Parseval恒等式)}
\end{aligned}
$$

因此$\Vert \boldsymbol{W} \boldsymbol{x} \Vert_2 \leq \sigma_1 \Vert \boldsymbol{x} \Vert_2 = \Vert \boldsymbol{W} \Vert_2 \Vert \boldsymbol{x} \Vert_2$。□

#### 3.2 与Frobenius范数的比较

Frobenius范数定义为：

$$
\Vert \boldsymbol{W} \Vert_F = \sqrt{\sum_{i=1}^n \sum_{j=1}^m W_{ij}^2} = \sqrt{\sum_{i=1}^r \sigma_i^2}
$$

容易验证：

$$
\Vert \boldsymbol{W} \Vert_2 = \sigma_1 \leq \sqrt{\sum_{i=1}^r \sigma_i^2} = \Vert \boldsymbol{W} \Vert_F
$$

等号成立当且仅当$\boldsymbol{W}$是秩1矩阵。一般情况下：

$$
\Vert \boldsymbol{W} \Vert_2 \leq \Vert \boldsymbol{W} \Vert_F \leq \sqrt{r} \Vert \boldsymbol{W} \Vert_2
$$

这说明$F$范数可能高估矩阵对向量的作用达$\sqrt{r}$倍！对于满秩矩阵，$r = \min(n,m)$可能非常大。

**例子**：考虑单位矩阵$\boldsymbol{I}_n \in \mathbb{R}^{n \times n}$：
- $\Vert \boldsymbol{I}_n \Vert_2 = 1$
- $\Vert \boldsymbol{I}_n \Vert_F = \sqrt{n}$

对任意$\Vert \boldsymbol{x} \Vert_2 = 1$，有$\Vert \boldsymbol{I}_n \boldsymbol{x} \Vert_2 = 1$，与谱范数一致，但$F$范数给出了$\sqrt{n}$的松弛界。

#### 3.3 谱范数约束下的最优更新方向

**定理2（最速下降方向）**：考虑优化问题：

$$
\min_{\Delta \boldsymbol{W}} \text{Tr}(\boldsymbol{G}^{\top} \Delta \boldsymbol{W}) \quad \text{s.t.} \quad \Vert \Delta \boldsymbol{W} \Vert_2 \leq \eta
$$

其最优解为：

$$
\Delta \boldsymbol{W}^* = -\eta \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}
$$

其中$\boldsymbol{u}_1, \boldsymbol{v}_1$是$\boldsymbol{G}$的最大奇异值对应的左右奇异向量。

**证明**：使用拉格朗日乘子法，构造拉格朗日函数：

$$
\mathcal{L}(\Delta \boldsymbol{W}, \lambda) = \text{Tr}(\boldsymbol{G}^{\top} \Delta \boldsymbol{W}) + \lambda(\Vert \Delta \boldsymbol{W} \Vert_2 - \eta)
$$

约束条件$\Vert \Delta \boldsymbol{W} \Vert_2 \leq \eta$在最优解处必然取等号（否则可以增大$\Vert \Delta \boldsymbol{W} \Vert$进一步减小目标），因此等价于：

$$
\max_{\Vert \Delta \boldsymbol{W} \Vert_2 = \eta} -\text{Tr}(\boldsymbol{G}^{\top} \Delta \boldsymbol{W})
$$

利用von Neumann迹不等式：对任意矩阵$\boldsymbol{A}, \boldsymbol{B}$，

$$
\text{Tr}(\boldsymbol{A}^{\top} \boldsymbol{B}) \leq \sum_{i=1}^r \sigma_i(\boldsymbol{A}) \sigma_i(\boldsymbol{B})
$$

其中$\sigma_i$按降序排列。等号成立当且仅当$\boldsymbol{A}$和$\boldsymbol{B}$共享相同的奇异向量。

对于$\Vert \Delta \boldsymbol{W} \Vert_2 = \eta$，即$\sigma_1(\Delta \boldsymbol{W}) = \eta$，$\sigma_i(\Delta \boldsymbol{W}) = 0$ for $i \geq 2$（秩1矩阵），因此：

$$
\text{Tr}(\boldsymbol{G}^{\top} \Delta \boldsymbol{W}) \leq \eta \sigma_1(\boldsymbol{G})
$$

等号成立当$\Delta \boldsymbol{W} = -\eta \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$。□

### 4. Muon优化器的完整推导

#### 4.1 带动量的谱范数约束优化

实际中我们使用动量$\boldsymbol{M}_t$而非直接的梯度$\boldsymbol{G}_t$：

$$
\boldsymbol{M}_t = \beta \boldsymbol{M}_{t-1} + (1-\beta) \boldsymbol{G}_t
$$

动量可以平滑梯度估计，减少噪声影响。优化问题变为：

$$
\min_{\Delta \boldsymbol{W}} \text{Tr}(\boldsymbol{M}_t^{\top} \Delta \boldsymbol{W}) \quad \text{s.t.} \quad \Vert \Delta \boldsymbol{W} \Vert_2 \leq \eta
$$

根据定理2，解为：

$$
\Delta \boldsymbol{W}_t = -\eta \boldsymbol{u}_1^{(t)} (\boldsymbol{v}_1^{(t)})^{\top}
$$

其中$\boldsymbol{u}_1^{(t)}, \boldsymbol{v}_1^{(t)}$是$\boldsymbol{M}_t$的主奇异向量。

定义矩阵符号函数（matrix sign）：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{U}_{:r} \boldsymbol{V}_{:r}^{\top}
$$

其中$\boldsymbol{U}_{:r}$取$\boldsymbol{U}$的前$r$列，$\boldsymbol{V}_{:r}$取$\boldsymbol{V}$的前$r$列。当$r=\text{rank}(\boldsymbol{M})$时，这给出秩保持的符号；实际中取$r=1$给出主方向。

因此Muon的更新为：

$$
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta_t \text{msign}(\boldsymbol{M}_t)
$$

#### 4.2 加入权重衰减的理论

权重衰减（Weight Decay）可以从$L2$正则化导出。考虑正则化损失：

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \Vert \boldsymbol{W} \Vert_F^2
$$

梯度变为：

$$
\nabla_{\boldsymbol{W}} \mathcal{L}_{\text{reg}} = \boldsymbol{G} + \lambda \boldsymbol{W}
$$

因此优化问题变为：

$$
\min_{\Delta \boldsymbol{W}} \text{Tr}((\boldsymbol{M}_t + \lambda \boldsymbol{W}_t)^{\top} \Delta \boldsymbol{W}) \quad \text{s.t.} \quad \Vert \Delta \boldsymbol{W} \Vert_2 \leq \eta
$$

但是$\boldsymbol{M}_t + \lambda \boldsymbol{W}_t$的奇异值分解复杂度高，Muon采用解耦权重衰减（decoupled weight decay）：

$$
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta_t [\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_t]
$$

这等价于：

$$
\boldsymbol{W}_{t+1} = (1 - \eta_t \lambda) \boldsymbol{W}_t - \eta_t \text{msign}(\boldsymbol{M}_t)
$$

#### 4.3 权重衰减的谱范数控制

**定理3（谱范数有界性）**：在Muon更新规则下，权重的谱范数满足：

$$
\Vert \boldsymbol{W}_t \Vert_2 \leq \max\left(\Vert \boldsymbol{W}_0 \Vert_2, \frac{1}{\lambda}\right)
$$

**证明**：由于$\Vert \text{msign}(\boldsymbol{M}_t) \Vert_2 = 1$（秩1单位矩阵），我们有：

$$
\begin{aligned}
\Vert \boldsymbol{W}_{t+1} \Vert_2 &= \Vert (1-\eta_t\lambda)\boldsymbol{W}_t - \eta_t \text{msign}(\boldsymbol{M}_t) \Vert_2 \\
&\leq (1-\eta_t\lambda) \Vert \boldsymbol{W}_t \Vert_2 + \eta_t \\
&= (1-\eta_t\lambda) \Vert \boldsymbol{W}_t \Vert_2 + \eta_t\lambda \cdot \frac{1}{\lambda}
\end{aligned}
$$

令$a_t = \Vert \boldsymbol{W}_t \Vert_2$，$c = 1/\lambda$，则：

$$
a_{t+1} \leq (1-\eta_t\lambda) a_t + \eta_t\lambda c
$$

这是一个加权平均，因此：

$$
a_{t+1} \leq \max(a_t, c) \leq \max(a_0, c)
$$

归纳可得结论。□

这个界说明权重不会无限增长，保证了训练的稳定性。

### 5. 优化器的定量比较

#### 5.1 各优化器的更新公式统一表示

为了公平比较，我们将各优化器表示为统一形式：

$$
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta_t \boldsymbol{\Phi}_t(\boldsymbol{G}_t, \boldsymbol{W}_t)
$$

其中$\boldsymbol{\Phi}_t$是优化器特定的更新算子。

| 优化器 | $\boldsymbol{\Phi}_t$ | 预条件器 |
|--------|----------------------|----------|
| SGD | $\boldsymbol{G}_t$ | $\boldsymbol{I}$ |
| Momentum SGD | $\boldsymbol{M}_t = \beta \boldsymbol{M}_{t-1} + (1-\beta)\boldsymbol{G}_t$ | $\boldsymbol{I}$ |
| Adam | $\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}$ | $\text{diag}(\frac{1}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon})$ |
| AdamW | $\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} + \lambda \boldsymbol{W}_t$ | $\text{diag}(\frac{1}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon})$ |
| Lion | $\text{sign}(\boldsymbol{m}_t)$ | $\boldsymbol{I}$ |
| Muon | $\text{msign}(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_t$ | 谱投影 |

#### 5.2 更新方向的有效性度量

定义更新方向与梯度的对齐度：

$$
\text{alignment}_t = \frac{\text{Tr}(\boldsymbol{G}_t^{\top} \boldsymbol{\Phi}_t)}{\Vert \boldsymbol{G}_t \Vert_F \Vert \boldsymbol{\Phi}_t \Vert_F}
$$

这是矩阵的余弦相似度，取值范围$[-1, 1]$。

**SGD**：$\text{alignment}_t = 1$（完美对齐）

**Adam**：由于逐元素归一化，$\text{alignment}_t \approx \frac{\Vert \text{sign}(\boldsymbol{G}_t) \Vert_F}{\Vert \boldsymbol{G}_t \Vert_F \cdot \sqrt{nm}}$

当梯度元素符号随机时，$\text{alignment}_t \approx \frac{1}{\sqrt{nm}}$（很小）

**Muon**：

$$
\begin{aligned}
\text{alignment}_t &= \frac{\text{Tr}(\boldsymbol{G}_t^{\top} \boldsymbol{u}_1 \boldsymbol{v}_1^{\top})}{\Vert \boldsymbol{G}_t \Vert_F} \\
&= \frac{\boldsymbol{v}_1^{\top} \boldsymbol{G}_t^{\top} \boldsymbol{u}_1}{\Vert \boldsymbol{G}_t \Vert_F} \\
&= \frac{\sigma_1(\boldsymbol{G}_t)}{\sqrt{\sum_i \sigma_i^2(\boldsymbol{G}_t)}}
\end{aligned}
$$

当$\sigma_1 \gg \sigma_{2,3,\ldots}$时（梯度低秩），$\text{alignment}_t \approx 1$（高对齐）。

#### 5.3 收敛速率的理论分析

考虑强凸函数$f(\boldsymbol{W})$，满足$\mu \boldsymbol{I} \preceq \nabla^2 f \preceq L \boldsymbol{I}$，条件数$\kappa = L/\mu$。

**SGD**：收敛速率为$O(\kappa \log(1/\epsilon))$次迭代达到$\epsilon$精度。

**预条件梯度下降**：若预条件器$\boldsymbol{P}$使得$\boldsymbol{P}^{-1/2} \nabla^2 f \boldsymbol{P}^{-1/2} \approx \boldsymbol{I}$，则收敛速率为$O(\log(1/\epsilon))$（条件数接近1）。

**Muon的隐式预条件**：谱范数约束相当于在每步将更新投影到单位球。对于二次函数：

$$
f(\boldsymbol{W}) = \frac{1}{2} \text{Tr}(\boldsymbol{W}^{\top} \boldsymbol{H} \boldsymbol{W}) - \text{Tr}(\boldsymbol{B}^{\top} \boldsymbol{W})
$$

梯度为$\boldsymbol{G} = \boldsymbol{H} \boldsymbol{W} - \boldsymbol{B}$。Muon更新：

$$
\Delta \boldsymbol{W} = -\eta \boldsymbol{u}_1(\boldsymbol{G}) \boldsymbol{v}_1(\boldsymbol{G})^{\top}
$$

这自动选择了Hessian $\boldsymbol{H}$的主方向，类似于power iteration，因此对主特征值方向收敛很快。

### 6. 参数空间的黎曼几何视角

#### 6.1 自然梯度与Fisher信息矩阵

在统计流形上，参数的真实距离由Fisher信息矩阵$\boldsymbol{F}$度量：

$$
ds^2 = d\boldsymbol{\theta}^{\top} \boldsymbol{F} d\boldsymbol{\theta}
$$

自然梯度定义为：

$$
\tilde{\nabla} f = \boldsymbol{F}^{-1} \nabla f
$$

它给出了在参数流形上的最速下降方向。

#### 6.2 Muon与自然梯度的联系

对于矩阵参数，Fisher信息矩阵具有Kronecker积结构：

$$
\boldsymbol{F} = \boldsymbol{A} \otimes \boldsymbol{B}
$$

其中$\boldsymbol{A} \in \mathbb{R}^{n \times n}$，$\boldsymbol{B} \in \mathbb{R}^{m \times m}$。完整计算$\boldsymbol{F}^{-1}$需要$O((nm)^3)$，不可行。

K-FAC（Kronecker-Factored Approximate Curvature）近似：

$$
\boldsymbol{F}^{-1} \approx \boldsymbol{A}^{-1} \otimes \boldsymbol{B}^{-1}
$$

自然梯度为：

$$
\tilde{\boldsymbol{G}} = \boldsymbol{A}^{-1} \boldsymbol{G} \boldsymbol{B}^{-1}
$$

Muon的msign可以看作另一种几何近似：不估计完整的度量张量，而是**在谱范数诱导的几何下优化**。

#### 6.3 矩阵流形上的测地线

考虑固定秩$r$的矩阵流形$\mathcal{M}_r = \{\boldsymbol{W} : \text{rank}(\boldsymbol{W}) = r\}$。这是一个非凸流形，维度为$r(n+m-r)$。

在$\mathcal{M}_r$上的切空间为：

$$
T_{\boldsymbol{W}} \mathcal{M}_r = \{\boldsymbol{U} \boldsymbol{X}^{\top} + \boldsymbol{Y} \boldsymbol{V}^{\top} : \boldsymbol{X} \in \mathbb{R}^{m \times r}, \boldsymbol{Y} \in \mathbb{R}^{n \times r}\}
$$

其中$\boldsymbol{W} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top}$是SVD分解。

Muon的更新$\boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$正是秩1流形$\mathcal{M}_1$上的一个元素，这可以看作在低维流形上优化的策略。

### 7. 预条件器的谱理论分析

#### 7.1 Adam的预条件器

Adam的预条件器为对角矩阵：

$$
\boldsymbol{P}_{\text{Adam}} = \text{diag}\left(\frac{1}{\sqrt{v_{11}}+\epsilon}, \ldots, \frac{1}{\sqrt{v_{nm}}+\epsilon}\right)
$$

其条件数为：

$$
\kappa(\boldsymbol{P}_{\text{Adam}}) = \frac{\max_i (\sqrt{v_i}+\epsilon)}{\min_i (\sqrt{v_i}+\epsilon)}
$$

当不同参数的梯度方差差异很大时，$\kappa$可能很大。

#### 7.2 Muon的隐式预条件

Muon通过谱投影实现了一种非线性预条件。考虑梯度$\boldsymbol{G}$的条件数：

$$
\kappa(\boldsymbol{G}) = \frac{\sigma_1(\boldsymbol{G})}{\sigma_r(\boldsymbol{G})}
$$

msign操作后：

$$
\kappa(\text{msign}(\boldsymbol{G})) = \frac{1}{1} = 1
$$

完美条件！这是因为msign将所有奇异值归一化为1。

#### 7.3 Hessian谱与收敛性

对于目标函数$f(\boldsymbol{W})$，Hessian矩阵的谱分布影响收敛速度。设Hessian的特征值为$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d > 0$。

**定理4（预条件效果）**：若预条件器$\boldsymbol{P}$使得$\boldsymbol{P} \boldsymbol{H}$的谱聚集在$[a, b]$，则预条件梯度下降的收敛速率依赖于$\kappa' = b/a$而非$\kappa = \lambda_1/\lambda_d$。

Adam试图通过逐元素归一化实现这一点，但只利用了对角信息。Muon通过谱投影，隐式地考虑了梯度的主模式，可能更好地预条件了Hessian的主特征空间。

### 8. 大模型训练的尺度定律

#### 8.1 Scaling Law的基本形式

根据Kaplan等人的研究，模型性能（如交叉熵损失$L$）与模型参数量$N$、数据量$D$、计算量$C$的关系为：

$$
L(N, D, C) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

其中$N_c, D_c, C_c$是特征尺度，$\alpha_N, \alpha_D, \alpha_C$是幂律指数（实验测得约为0.076, 0.095, 0.050）。

#### 8.2 优化器效率与等效计算量

不同优化器达到相同损失所需的步数不同。定义优化器效率因子$\gamma$：

$$
C_{\text{Muon}} = \gamma \cdot C_{\text{Adam}}
$$

根据文章中的实验，Muon达到相同性能约需Adam的50%步数，即$\gamma \approx 0.5$。

代入Scaling Law：

$$
L_{\text{Muon}}(C) \approx L_{\text{Adam}}(2C)
$$

这意味着在相同计算预算下，Muon能达到训练更大模型所对应的性能。

#### 8.3 参数效率的理论解释

Muon训练的模型奇异值熵更高，意味着参数利用更充分。定义参数有效性：

$$
\text{Efficiency} = \frac{\text{Model Capacity}}{\text{Parameter Count}}
$$

用奇异值熵衡量容量：

$$
\text{Capacity} = \exp\left(\frac{1}{L}\sum_{\ell=1}^L H(\boldsymbol{\sigma}^{(\ell)})\right)
$$

其中$L$是层数，$H(\boldsymbol{\sigma}^{(\ell)})$是第$\ell$层的奇异值熵。

Muon通过谱约束鼓励参数空间的均匀利用，提高了这个效率指标。

### 9. 计算复杂度与内存效率分析

#### 9.1 各优化器的计算复杂度

对于矩阵$\boldsymbol{W} \in \mathbb{R}^{n \times m}$：

| 优化器 | 时间复杂度 | 空间复杂度（状态） |
|--------|-----------|------------------|
| SGD | $O(1)$ | $O(nm)$（参数） |
| Momentum | $O(nm)$ | $2 \times O(nm)$ |
| Adam | $O(nm)$ | $3 \times O(nm)$（$m, v, W$） |
| Muon（完整SVD） | $O(nm\min(n,m))$ | $2 \times O(nm)$ |
| Muon（Newton-Schulz） | $O(nm)$ | $2 \times O(nm)$ |

#### 9.2 Newton-Schulz迭代的推导

完整SVD计算$\text{msign}(\boldsymbol{M})$需要$O(nm\min(n,m))$，不可接受。Newton-Schulz迭代提供了$O(nm)$的近似方法。

设$\boldsymbol{M} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top}$，我们要计算：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{U} \boldsymbol{V}^{\top} = \boldsymbol{M} (\boldsymbol{M}^{\top} \boldsymbol{M})^{-1/2}
$$

定义$\boldsymbol{A} = \boldsymbol{M}^{\top} \boldsymbol{M} / \Vert \boldsymbol{M} \Vert_F^2$，则$\boldsymbol{A}$的特征值在$(0, 1]$内。

Newton-Schulz迭代计算$\boldsymbol{A}^{-1/2}$：

$$
\boldsymbol{X}_0 = \boldsymbol{I}, \quad \boldsymbol{X}_{k+1} = \frac{1}{2} \boldsymbol{X}_k (3\boldsymbol{I} - \boldsymbol{A} \boldsymbol{X}_k^2)
$$

**定理5（Newton-Schulz收敛性）**：若$\boldsymbol{A}$的特征值在$(0, 2)$内，则$\boldsymbol{X}_k \to \boldsymbol{A}^{-1/2}$，且收敛速度为三次：

$$
\Vert \boldsymbol{X}_k \boldsymbol{A}^{1/2} - \boldsymbol{I} \Vert \leq C \cdot \delta^{3^k}
$$

其中$\delta = \Vert \boldsymbol{X}_0 \boldsymbol{A}^{1/2} - \boldsymbol{I} \Vert < 1$。

通常5-7次迭代即可达到机器精度。每次迭代需要两次矩阵乘法，复杂度$O(m^3)$（对$\boldsymbol{A} \in \mathbb{R}^{m \times m}$）。

但我们不需要显式计算$\boldsymbol{A}^{-1/2}$，而是直接迭代：

$$
\boldsymbol{Z}_0 = \boldsymbol{M} / \Vert \boldsymbol{M} \Vert_F, \quad \boldsymbol{Z}_{k+1} = \frac{3}{2}\boldsymbol{Z}_k - \frac{1}{2}\boldsymbol{Z}_k \boldsymbol{Z}_k^{\top} \boldsymbol{Z}_k
$$

此时每次迭代是$O(nm^2)$或$O(n^2 m)$（取决于$n, m$大小），仍比完整SVD快。

#### 9.3 内存优化技巧

在实现中，可以避免存储完整的动量$\boldsymbol{M}$，而是存储其低秩分解：

$$
\boldsymbol{M}_t \approx \boldsymbol{U}_k \boldsymbol{\Sigma}_k \boldsymbol{V}_k^{\top}
$$

只保留前$k$个奇异值（如$k=100$），内存从$O(nm)$降至$O(k(n+m))$。

对于大型矩阵（如$10000 \times 10000$），这可以节省数百倍内存。

### 10. 实验现象的深度理论解释

#### 10.1 Muon收敛更快的原因

实验显示Muon约2倍快于Adam。从优化轨迹分析：

**损失下降率**：第$t$步的损失下降为：

$$
\Delta L_t = L_t - L_{t+1} \approx -\text{Tr}(\boldsymbol{G}_t^{\top} \Delta \boldsymbol{W}_t)
$$

**Adam**：

$$
\Delta L_t^{\text{Adam}} \approx \eta \sum_{ij} \frac{G_{ij}^2}{\sqrt{v_{ij}}} \approx \eta \cdot O(\sqrt{nm})
$$

（假设$G_{ij}$和$v_{ij}$同尺度）

**Muon**：

$$
\Delta L_t^{\text{Muon}} = \eta \text{Tr}(\boldsymbol{G}_t^{\top} \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}) = \eta \sigma_1(\boldsymbol{G}_t)
$$

当梯度低秩（$\sigma_1 \gg \sigma_{2,\ldots}$）时，$\sigma_1 \approx \Vert \boldsymbol{G} \Vert_F$，因此：

$$
\frac{\Delta L_t^{\text{Muon}}}{\Delta L_t^{\text{Adam}}} \approx \frac{\Vert \boldsymbol{G} \Vert_F}{\sqrt{nm}} = \text{RMS}(\boldsymbol{G}) \sqrt{nm}
$$

对于典型值$\text{RMS}(\boldsymbol{G}) \sim 10^{-3}$，$nm \sim 10^6$，比值约为1-10，与实验观察一致！

#### 10.2 奇异值熵增加的意义

文章观察到Muon训练的模型$H(\boldsymbol{\sigma})$更高。这意味着什么？

**信息论视角**：奇异值熵衡量矩阵的"信息容量"。设$p_i = \sigma_i^2 / \sum_j \sigma_j^2$，则：

$$
H = -\sum_i p_i \log p_i
$$

最大熵（$H=\log r$）当所有$p_i$相等，即所有奇异值相等。此时矩阵在所有方向上"平等"地传播信息。

**低秩vs满秩**：
- 低秩矩阵（少数大奇异值）：$H$小，信息集中在少数模式
- 满秩矩阵（奇异值均匀）：$H$大，信息分散在所有模式

Muon鼓励$H$增大，意味着模型学会了利用更多的参数自由度，而非仅依赖少数主模式。

**泛化能力**：根据学习理论，更高的$H$对应更低的"有效参数数量"（effective parameter）的估计：

$$
N_{\text{eff}} = \exp(H) \cdot r
$$

这与隐式正则化相关，可能解释Muon的泛化优势。

#### 10.3 预训练-微调兼容性问题

实验发现Adam预训练+Muon微调效果不佳。可能的解释：

**Hessian谱的适配**：不同优化器会导致不同的Hessian谱结构。Adam倾向于平滑各向异性（因为逐元素归一化），而Muon保留主方向的各向异性。

设Adam训练后的Hessian在某个局部最优点$\boldsymbol{W}^*$附近：

$$
\boldsymbol{H}_{\text{Adam}} \approx \text{diag}(\lambda_1, \ldots, \lambda_{nm})
$$

（近似对角）

而Muon期望的Hessian：

$$
\boldsymbol{H}_{\text{Muon}} \approx \sum_{i=1}^k \lambda_i \boldsymbol{u}_i \boldsymbol{u}_i^{\top}
$$

（低秩+扰动）

两者结构不匹配，导致Muon在Adam的最优点附近难以找到好的下降方向。

**解决方案**：可能需要一个"过渡期"，在微调开始时使用较小的学习率让Muon适应当前参数空间的几何。

#### 10.4 RMS对齐的理论基础

文章提出的RMS对齐策略：

$$
\boldsymbol{\Phi}_{\text{norm}} = 0.2 \cdot \frac{\boldsymbol{\Phi}}{\text{RMS}(\boldsymbol{\Phi})}
$$

为什么选择0.2？

**经验观察**：Adam的更新RMS通常在0.2-0.4。这可以从以下角度理解：

假设梯度元素$G_{ij} \sim \mathcal{N}(0, \sigma^2)$，Adam更新：

$$
\Delta W_{ij} \approx \alpha \text{sign}(G_{ij})
$$

则：

$$
\text{RMS}(\Delta \boldsymbol{W}) = \alpha
$$

典型学习率$\alpha \in [10^{-4}, 10^{-3}]$看起来不对……实际上要考虑RMSProp的缩放：

$$
\text{RMS}(\Delta \boldsymbol{W}) = \alpha \cdot \text{RMS}\left(\frac{1}{\sqrt{v}}\right) \approx \frac{\alpha}{\text{RMS}(\sqrt{v})}
$$

当$v \sim \sigma^2$时，这个比例约为$10^{-4} / 10^{-3} \sim 0.1$...

实际上0.2是经验值，可能依赖于具体的网络结构和初始化。

#### 10.5 不同矩阵形状的学习率调整

文章指出：

$$
\text{RMS}(\text{msign}(\boldsymbol{M})) = \sqrt{\frac{\min(n,m)}{nm}} = \frac{1}{\sqrt{\max(n,m)}}
$$

因此有效学习率应该是：

$$
\eta_{\text{eff}} = \eta \cdot \sqrt{\max(n,m)}
$$

**几何解释**：对于$n \ll m$的"瘦高"矩阵，$\text{msign}(\boldsymbol{M})$的元素RMS很小，需要放大学习率以保持与方阵相同的更新幅度。

**例子**：
- $\boldsymbol{W}_1 \in \mathbb{R}^{1024 \times 1024}$：$\eta_{\text{eff}} = \eta \cdot 32$
- $\boldsymbol{W}_2 \in \mathbb{R}^{1024 \times 32000}$：$\eta_{\text{eff}} = \eta \cdot 179$

两者相差约5.6倍，正是$\sqrt{32000/1024} \approx 5.6$。

### 11. 扩展理论与未来方向

#### 11.1 Schatten范数族

Schatten-$p$范数定义为：

$$
\Vert \boldsymbol{W} \Vert_p = \left(\sum_{i=1}^r \sigma_i^p\right)^{1/p}
$$

特例：
- $p=1$：核范数（nuclear norm）
- $p=2$：Frobenius范数
- $p=\infty$：谱范数

对应的优化问题：

$$
\min_{\Delta \boldsymbol{W}} \text{Tr}(\boldsymbol{G}^{\top} \Delta \boldsymbol{W}) \quad \text{s.t.} \quad \Vert \Delta \boldsymbol{W} \Vert_p \leq \eta
$$

**$p=1$（核范数）**：解为低秩矩阵，相当于稀疏正则化（在奇异值上）

**$p=2$（Frobenius）**：解为$\Delta \boldsymbol{W} = -\eta \boldsymbol{G} / \Vert \boldsymbol{G} \Vert_F$（标准SGD）

**$p=\infty$（谱范数）**：解为$\Delta \boldsymbol{W} = -\eta \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$（Muon）

中间的$p \in (1, \infty)$可能提供更好的trade-off，值得探索！

#### 11.2 µP (Maximal Update Parametrization)与Muon

µP是一种参数化方案，使得超参数在不同模型尺度下可迁移。核心思想：

不同层的学习率应该按照其宽度缩放：

$$
\eta_{\ell} = \frac{\eta_{\text{base}}}{w_{\ell}}
$$

其中$w_{\ell}$是第$\ell$层的宽度。

对于Muon，由于已经有$\sqrt{\max(n,m)}$的自适应，µP的设计需要重新考虑：

$$
\eta_{\ell}^{\text{Muon}} = \eta_{\text{base}} \cdot \frac{\sqrt{\max(n_{\ell}, m_{\ell})}}{w_{\text{base}}}
$$

这是一个开放问题，需要实验验证。

#### 11.3 分布式训练的考虑

在数据并行中，梯度需要all-reduce：

$$
\boldsymbol{G} = \frac{1}{K} \sum_{k=1}^K \boldsymbol{G}_k
$$

对于Adam，all-reduce后直接应用逐元素操作，通信量$O(nm)$。

对于Muon，需要对all-reduce后的$\boldsymbol{G}$做SVD，通信量相同，但计算集中在主节点，可能成为瓶颈。

**优化方案**：每个节点独立计算$\boldsymbol{G}_k$的SVD，然后平均奇异向量：

$$
\boldsymbol{u} = \frac{1}{K} \sum_{k=1}^K \boldsymbol{u}_k, \quad \boldsymbol{v} = \frac{1}{K} \sum_{k=1}^K \boldsymbol{v}_k
$$

再正交化。这需要理论分析其近似误差。

### 12. 总结与洞察

本推导从优化理论的第一原理出发，完整地解释了Muon优化器的设计动机和理论基础：

1. **最小作用量原理**：优化器本质上是在"稳"（小扰动）和"快"（大降损失）之间权衡，不同的"稳"度量导致不同的优化器。

2. **谱范数的最优性**：对于矩阵参数，谱范数是控制输出扰动的最紧界，这是Muon相比SGD（$F$范数）的理论优势。

3. **msign的几何意义**：它是谱范数球上的最速下降方向，同时也是梯度的秩1最佳近似，体现了低秩结构的利用。

4. **与Adam的根本差异**：Adam是逐元素的自适应，忽略了参数矩阵的整体结构；Muon是整体的谱投影，充分利用了矩阵的几何。

5. **实验现象的理论解释**：
   - 收敛更快：因为更好地对齐了梯度的主方向
   - 奇异值熵更高：因为谱约束鼓励参数的均匀利用
   - 微调兼容性问题：因为Hessian谱结构的不匹配

6. **未来方向**：Schatten范数族、µP适配、分布式优化等都是值得探索的理论和实践问题。

这些推导不仅加深了对Muon的理解，也为设计新的优化器提供了理论指导：**从参数的内在几何结构出发，而非简单的逐元素操作**。

