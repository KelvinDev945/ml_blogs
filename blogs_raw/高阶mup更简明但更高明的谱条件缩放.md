---
title: 高阶MuP：更简明但更高明的谱条件缩放
slug: 高阶mup更简明但更高明的谱条件缩放
date: 2025-03-24
tags: LoRA, 梯度, 优化器, 尺度定律, 谱范数
status: pending
---

# 高阶MuP：更简明但更高明的谱条件缩放

**原文链接**: [https://spaces.ac.cn/archives/10795](https://spaces.ac.cn/archives/10795)

**发布日期**: 

---

在文章[《初探MuP：超参数的跨模型尺度迁移规律》](/archives/10770)中，我们基于前向传播、反向传播、损失增量和特征变化的尺度不变性推导了MuP（Maximal Update Parametrization）。可能对于部分读者来说，这一过程还是显得有些繁琐，但实际上它比原始论文已经明显简化。要知道，我们是在单篇文章内相对完整地介绍的MuP，而MuP的论文实际上是作者Tensor Programs系列论文的第5篇！

不过好消息是，作者在后续的研究[《A Spectral Condition for Feature Learning》](https://papers.cool/arxiv/2310.17813)中，发现了一种新的理解方式（下称“谱条件”），它比MuP的原始推导和笔者的推导都更加直观和简洁，但却能得到比MuP更丰富的结果，可谓MuP的高阶版本，简明且不失高明的代表作。

## 准备工作 #

顾名思义，谱条件（Spectral Condition）跟谱范数（Spectral Norm）相关，它的出发点是谱范数的一个基本不等式：  
\begin{equation}\Vert\boldsymbol{x}\boldsymbol{W}\Vert_2\leq \Vert\boldsymbol{x}\Vert_2 \Vert\boldsymbol{W}\Vert_2\label{neq:spec-2}\end{equation}  
其中$\boldsymbol{x}\in\mathbb{R}^{d_{in}}, \boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$，至于$\Vert\cdot\Vert_2$，我们可以叫它“$2$范数”。对于$\boldsymbol{x},\boldsymbol{x}\boldsymbol{W}$来说，它们都是向量，$2$范数就是向量模长；而$\boldsymbol{W}$是一个矩阵，它的$2$范数也称为谱范数，它等于让$\Vert\boldsymbol{x}\boldsymbol{W}\Vert_2\leq C\Vert\boldsymbol{x}\Vert_2$恒成立的最小常数$C$。换句话说，上述不等式其实是谱范数定义的直接推论，无需额外证明。

关于谱范数，大家还可以看[《深度学习中的Lipschitz约束：泛化与生成模型》](/archives/6051)、[《低秩近似之路（二）：SVD》](/archives/10407#%E7%9F%A9%E9%98%B5%E8%8C%83%E6%95%B0)等博文，这里不再展开介绍。矩阵还有一个更简单的$F$范数，它是向量模长的简单推广：  
\begin{equation}\Vert \boldsymbol{W}\Vert_F = \sqrt{\sum_{i=1}^{d_{in}}\sum_{j=1}^{d_{out}}W_{i,j}^2}\end{equation}  
从奇异值角度看，谱范数等于矩阵最大的奇异值，而$F$范数等于矩阵全体奇异值的平方和的平方根。类似地，我们还可以定义“核范数（Nuclear Norm）”，它等于全体奇异值的和：  
\begin{equation}\Vert \boldsymbol{W}\Vert_* = \sum_{i=1}^{\min(d_{in}, d_{out})} \sigma_i\end{equation}  
像谱范数、$F$范数、核范数等可以由奇异值表示出来的矩阵范数，它们都属于“[Schatten-p范数](https://en.wikipedia.org/wiki/Schatten_norm)”的一种。最后，我们来定义RMS（Root Mean Square），它是向量模长的变体：  
\begin{equation}\Vert\boldsymbol{x}\Vert_{RMS} = \sqrt{\frac{1}{d_{in}}\sum_{i=1}^{d_{in}} x_i^2} = \frac{1}{\sqrt{d_{in}}}\Vert \boldsymbol{x}\Vert_2 \end{equation}  
如果要推广到矩阵，那就是$\Vert\boldsymbol{W}\Vert_{RMS} = \Vert \boldsymbol{W}\Vert_F/\sqrt{d_{in} d_{out}}$。其实从名字就很好理解，向量模长或矩阵$F$范数，我们可以称为“Root Sum Square”，而RMS就是把Sum换成Mean，它主要用来作为向量或矩阵元素的平均尺度指标。现在把RMS代入不等式$\eqref{neq:spec-2}$，可以得到  
\begin{equation}\Vert\boldsymbol{x}\boldsymbol{W}\Vert_{RMS}\leq \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\Vert_{RMS} \Vert\boldsymbol{W}\Vert_2\label{neq:spec-rms}\end{equation}

## 期望性质 #

我们之前推导MuP的思路，是仔细分析**前向传播** 、**反向传播** 、**损失增量** 和**特征变化** 的形式，通过调整初始化和学习率来实现它们的尺度不变性。谱条件对其“去芜存菁”后发现，只要**前向传播** 和**特征变化** 两点足矣。

简单来说，谱条件期望 _每一层的**输出** 和**增量** 都具有尺度不变性_。怎么理解这句话呢？如果我们将每一层简记为$\boldsymbol{x}_k= f(\boldsymbol{x}_{k-1}; \boldsymbol{W}_k)$为例，这句话可以翻译成“期望每个$\Vert\boldsymbol{x}_k\Vert_{RMS}$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}$都是$\mathcal{\Theta}(1)$”的（$\mathcal{\Theta}$是“[Big Theta Notation](https://en.wikipedia.org/wiki/Big_O_notation#Family_of_Bachmann%E2%80%93Landau_notations)”）：

> 1、$\Vert\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$好理解，它代表着前向传播的稳定性，上一篇文章的推导也包含这个要求；
> 
> 2、$\Delta\boldsymbol{x}_k$表示参数变化引起的$\boldsymbol{x}_k$变化量，所以$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$融合了反向传播和特征变化的要求。

可能有读者疑问：那是不是至少应该还有个“损失增量”的要求？并不需要。事实上，我们可以证明，如果每一层的$\Vert\boldsymbol{x}_k\Vert_{RMS}$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}$都是$\mathcal{\Theta}(1)$，那么$\Delta\mathcal{L}$自动就是$\mathcal{\Theta}(1)$的。这正是谱条件思想第一个美妙之处，它将原本推导MuP需要的四个条件降低到两个，减少了分析步骤。

证明并不困难，这里的关键是我们假设了 _每一层_ 都成立$\Vert\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$，那么最后一层自然也成立。假设模型一共有$K$层，单个样本损失函数为$\ell$，那么它是$\boldsymbol{x}_K$的函数即$\ell(\boldsymbol{x}_K)$，简单起见这里省掉了标签输入，因为对下面的分析来说它并非变量。

根据假设，$\Vert\boldsymbol{x}_K\Vert_{RMS}$是$\mathcal{\Theta}(1)$的，那么$\ell(\boldsymbol{x}_K)$自然是$\mathcal{\Theta}(1)$的；又因为$\Vert\Delta\boldsymbol{x}_K\Vert_{RMS}$是$\mathcal{\Theta}(1)$的，所以$\Vert\boldsymbol{x}_K + \Delta\boldsymbol{x}_K\Vert_{RMS}\leq \Vert\boldsymbol{x}_K\Vert_{RMS} + \Vert\Delta\boldsymbol{x}_K\Vert_{RMS}$也是$\mathcal{\Theta}(1)$的，从而$\ell(\boldsymbol{x}_K + \Delta\boldsymbol{x}_K)$是$\mathcal{\Theta}(1)$的，于是  
\begin{equation}\Delta \ell = \ell(\boldsymbol{x}_K + \Delta\boldsymbol{x}_K) - \ell(\boldsymbol{x}_K) = \mathcal{\Theta}(1)\end{equation}  
所以，单个样本的损失增量$\Delta \ell$是$\mathcal{\Theta}(1)$的，而$\Delta\mathcal{L}$是全体$\Delta \ell$的平均，所以它也是$\mathcal{\Theta}(1)$的。这样我们就证明了$\Vert\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$自动包含了$\Delta\mathcal{L}=\mathcal{\Theta}(1)$，原理说白了就是$\Delta\mathcal{L}$是最后一层输出及其增量的函数，它们都稳定了，$\Delta\mathcal{L}$自然就稳定了。

## 谱条件 #

接着，我们看如何成立两个期望性质。由于神经网络以矩阵乘法为主，所以我们先考虑最简单的线性层$\boldsymbol{x}_k = \boldsymbol{x}_{k-1} \boldsymbol{W}_k$，其中$\boldsymbol{W}_k\in\mathbb{R}^{d_{k-1}\times d_k}$。为了成立$\Vert\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$的条件，谱条件没有像传统初始化分析一样去假设独立同分布然后算期望方差等，而是直接应用不等式$\eqref{neq:spec-rms}$：  
\begin{equation}\Vert\boldsymbol{x}_k\Vert_{RMS}\leq \sqrt{\frac{d_{k-1}}{d_k}}\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}\, \Vert\boldsymbol{W}_k\Vert_2\end{equation}  
注意这个不等式是可能取到等号的，并且某种意义上是最精准的，所以如果输入的$\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}$已经是$\mathcal{\Theta}(1)$，那么为了使输出的$\Vert\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$，那么就要  
\begin{equation}\sqrt{\frac{d_{k-1}}{d_k}}\Vert\boldsymbol{W}_k\Vert_2 = \mathcal{\Theta}(1)\quad\Rightarrow\quad \Vert\boldsymbol{W}_k\Vert_2 = \mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)\label{eq:spec-c1}\end{equation}  
这就提出了第一个**谱条件** ——对$\boldsymbol{W}_k$的谱范数要求。它无关初始化和分布假设，完全是分析和代数的结果，这是笔者认为谱条件的第二个美妙之处——简化分析过程。当然，这里省略了谱范数的基础内容，补上的话整个篇幅不见得比分布假设下的分析短，但分布假设终究显得局限性大，不如这里的代数框架灵活。

分析完$\Vert\boldsymbol{x}_k\Vert_{RMS}$后，就轮到$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}$了，增量的$\Delta\boldsymbol{x}_k$的来源有两部分，一是参数由$\boldsymbol{W}_k$变为$\boldsymbol{W}_k+\Delta \boldsymbol{W}_k$，二是输入$\boldsymbol{x}_{k-1}$的参数变化导致它从$\boldsymbol{x}_{k-1}$变为$\boldsymbol{x}_{k-1} + \Delta\boldsymbol{x}_{k-1}$，所以  
\begin{equation}\begin{aligned}  
\Delta\boldsymbol{x}_k =&\, (\boldsymbol{x}_{k-1} + \Delta\boldsymbol{x}_{k-1})(\boldsymbol{W}_k+\Delta \boldsymbol{W}_k) - \boldsymbol{x}_{k-1}\boldsymbol{W}_k \\\\[5pt]  
=&\, \boldsymbol{x}_{k-1} (\Delta \boldsymbol{W}_k) + (\Delta\boldsymbol{x}_{k-1})\boldsymbol{W}_k + (\Delta\boldsymbol{x}_{k-1})(\Delta \boldsymbol{W}_k)  
\end{aligned}\end{equation}  
所以  
\begin{equation}\begin{aligned}  
\Vert\Delta\boldsymbol{x}_k\Vert_{RMS} =&\, \Vert\boldsymbol{x}_{k-1} (\Delta \boldsymbol{W}_k) + (\Delta\boldsymbol{x}_{k-1})\boldsymbol{W}_k + (\Delta\boldsymbol{x}_{k-1})(\Delta \boldsymbol{W}_k)\Vert_{RMS} \\\\[5pt]  
\leq&\, \Vert\boldsymbol{x}_{k-1} (\Delta \boldsymbol{W}_k)\Vert_{RMS} + \Vert(\Delta\boldsymbol{x}_{k-1})\boldsymbol{W}_k\Vert_{RMS} + \Vert(\Delta\boldsymbol{x}_{k-1})(\Delta \boldsymbol{W}_k)\Vert_{RMS} \\\\[5pt]  
\leq&\, \sqrt{\frac{d_{k-1}}{d_k}}\left({\begin{gathered}\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}\,\Vert\Delta \boldsymbol{W}_k\Vert_2 + \Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}\,\Vert \boldsymbol{W}_k\Vert_2 \\\\[5pt]  
\+ \Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}\,\Vert\Delta \boldsymbol{W}_k\Vert_2\end{gathered}} \right)  
\end{aligned}\end{equation}  
逐项分析一下  
\begin{equation}\underbrace{\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}}_{\mathcal{\Theta}(1)}\,\Vert\Delta \boldsymbol{W}_k\Vert_2 + \underbrace{\Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}}_{\mathcal{\Theta}(1)}\,\underbrace{\Vert \boldsymbol{W}_k\Vert_2}_{\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)} + \underbrace{\Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}}_{\mathcal{\Theta}(1)}\,\Vert\Delta \boldsymbol{W}_k\Vert_2\end{equation}  
由此可见，要想$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)$，那么就需要  
\begin{equation}\Vert\Delta\boldsymbol{W}_k\Vert_2 = \mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)\label{eq:spec-c2}\end{equation}  
这就是第二个**谱条件** ——对$\Delta\boldsymbol{W}_k$的谱范数要求。

上面的分析没有考虑非线性，事实上只要激活函数是Element-wise的，并且导数能够被某个常数Bound住（常用的ReLU、Sigmoid、Tanh等激活函数都满足），那么即便考虑非线性激活函数的结果也是一致，这就是上一篇文章分析所说的“激活函数的影响是尺度无关的”。如果读者还不放心，可以自行推导一下。

## 谱归一化 #

现在我们有了两个谱条件$\eqref{eq:spec-c1}$和$\eqref{eq:spec-c2}$，接下来就要看怎么设计才能让模型自身以及模型优化来满足这两个条件了。

注意，$\boldsymbol{W}_k$和$\Delta \boldsymbol{W}_k$都是矩阵，让一个矩阵满足谱范数条件的标准方法通常是谱归一化（Spectral Normalization，SN），这里也不例外。首先，我们要让初始化的$\boldsymbol{W}_k$满足$\Vert\boldsymbol{W}_k\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，这可以通过选取任意的初始化矩阵$\boldsymbol{W}_k'$，然后谱归一化实现：  
\begin{equation}\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k'}{\Vert\boldsymbol{W}_k'\Vert_2}\end{equation}  
这里的$\sigma > 0$是尺度无关的常数；同理，对于任意优化器给出的更新量$\boldsymbol{\Phi}_k$，我们可以通过谱归一化来重新构造$\Delta \boldsymbol{W}_k$：  
\begin{equation}\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{\Phi}_k}{\Vert\boldsymbol{\Phi}_k\Vert_2}\end{equation}  
其中$\eta > 0$也是尺度无关的常数（学习率），这样每一步都有$\Vert\Delta\boldsymbol{W}_k\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$。由于初始化和每一步更新的谱范数都满足$\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，所以$\Vert\boldsymbol{W}_k\Vert_2$自始至终也满足$\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，这就满足了两个谱条件。

这时候可能会有读者疑问，只考虑初始化和增量的稳定性，真的能保证$\boldsymbol{W}_k$的稳定性吗？难道不可能出现$\Vert\boldsymbol{W}_k\Vert_{RMS}\to\infty$吗？答案是有可能。这里的$\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$强调的是与模型尺度（目前主要是宽度）的关系，它不排除由于其余超参设置不当导致训练崩溃的可能性，它要表达的是按这样设置之后，即便出现了崩溃现象，原因也跟尺度变化无关。

## 奇异值裁剪 #

要实现谱范数条件，除了谱归一化这种标准方法外，我们还可以考虑奇异值裁剪（Singular Value Clipping，下面简称“SVC”）。这部分内容是笔者自行补充的，并未在原论文出现，但它可以解释一些有意思的结果。

从奇异值的角度来，谱归一化是将最大奇异值缩放成1，并同步缩放其余奇异值，奇异值裁剪某个角度来看宽松一些，它只将大于1的奇异值设为1，但不改变原本就小于等于1的奇异值：  
\begin{equation}\mathop{\text{SVC}}(\boldsymbol{W}) = \boldsymbol{U}\min(\boldsymbol{\Lambda},1)\boldsymbol{V}^{\top},\qquad \boldsymbol{U},\boldsymbol{\Lambda},\boldsymbol{V}^{\top} = \mathop{\text{SVD}}(\boldsymbol{W})\end{equation}  
作为对比，谱归一化是$\mathop{\text{SN}}(\boldsymbol{W})=\boldsymbol{U}(\boldsymbol{\Lambda}/\max(\boldsymbol{\Lambda}))\boldsymbol{V}^{\top}$。用奇异值裁剪替代谱归一化，我们得到  
\begin{equation}\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{W}_k'), \qquad \Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{\Phi}_k)\end{equation}  
奇异值裁剪的缺点是它只有在至少一个奇异值大于等于1时才能保证裁剪后的谱范数等于1，如果不满足，我们可以考虑乘上一个$\lambda > 0$然后再裁剪，即改为$\mathop{\text{SVC}}(\lambda\boldsymbol{W})$。然而，不同的比例因子会得到不同的结果，但我们也不大好确定适合的比例因子。不过，我们可以考虑一个极限版本  
\begin{equation}\lim_{\lambda\to\infty} \mathop{\text{SVC}}(\lambda\boldsymbol{W}) = \mathop{\text{msign}}(\boldsymbol{W})\end{equation}  
这里的$\mathop{\text{msign}}$就是Muon里的矩阵版符号函数$\mathop{\text{msign}}$（参考[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)）。用$\mathop{\text{msign}}$替换谱归一化或奇异值裁剪，得到  
\begin{equation}\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{msign}}(\boldsymbol{\Phi}_k)\end{equation}  
这样我们实际得到了广义的Muon优化器，标准的Muon是对动量$\mathop{\text{msign}}$，而它允许我们对任何现成优化器出来的更新量进行$\mathop{\text{msign}}$。无独有偶，前段时间推特上还真有人做了对Adam更新量$\mathop{\text{msign}}$的实验（对方称之为“Mudamw”，[链接](https://x.com/kyleliang5/status/1896385441571742103)），发现效果比Muon还略好，如下图所示：  


[![Adam+msgin效果似乎优于Muon（来自推特 @KyleLiang5 ）](/usr/uploads/2025/03/585996134.jpeg)](/usr/uploads/2025/03/585996134.jpeg "点击查看原图")

Adam+msgin效果似乎优于Muon（来自推特 @KyleLiang5 ）

我们看到后在小模型上也尝试了一下，发现居然可以复现到相似的结论！所以说不准对现有优化器$\mathop{\text{msign}}$一下，都有机会得到更好的结果。这种操作的可行性在原本的Muon框架下是很难解释的，但这里我们将它理解为对更新量做奇异值裁剪（的极限版本），就会自然得到了这个结果。

## 近似估计 #

一般认为，谱归一化、奇异值裁剪或$\mathop{\text{msign}}$等跟SVD（奇异值分解）相关的运算都是比较昂贵的，所以我们还是希望能寻找更简单的形式。由于我们的目标只是寻求模型尺度间的缩放规律，所以进一步的简化确实是有可能的。

（注：事实上我们的[Moonlight](https://papers.cool/arxiv/2502.16982)工作表明，只要实现得好，即便每一步更新都进行$\mathop{\text{msign}}$，所增加的成本是非常有限的，所以这一节的内容，目前看来更多是为了探索显式的缩放规律而不是节省计算成本）。

首先仍然是初始化，初始化其实是一次性的，所以其实计算量大一点也不是啥问题，所以前面的先随机初始化然后谱归一化/奇异值裁剪/$\mathop{\text{msign}}$的方案可以继续保留。如果还是想要精益求精，那么可以利用一个统计结果：一个从标准正态分布独立重复采样出来的$d_{k-1}\times d_k$矩阵，它的最大奇异值大致是$\sqrt{d_{k-1}} + \sqrt{d_k}$。这样相当于说只要采样标准差改为  
\begin{equation}\sigma_k = \mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}(\sqrt{d_{k-1}} + \sqrt{d_k})^{-1}\right) = \mathcal{\Theta}\left(\sqrt{\frac{1}{d_{k-1}}\min\left(1, \frac{d_k}{d_{k-1}}\right)}\right) \label{eq:spec-std}\end{equation}  
就可以在初始化阶段满足$\Vert\boldsymbol{W}_k\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$的需求。至于这个统计结果的证明，可以参考[《High-Dimensional Probability》](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html)、[《Marchenko-Pastur law》](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution)等资料，这里就不展开了。

然后我们来考察更新量，它相对麻烦一点，因为任意更新量$\boldsymbol{\Phi}_k$的谱范数并不是那么好估计。这里我们需要利用一个经验结论，那就是参数的梯度矩阵通常都是低秩的。这里的低秩并不一定是数学上严格的低秩，而是指最大的几个（数目跟模型尺度无关）奇异值明显超过其余奇异值，使得低秩近似可用，这也是各种[LoRA](/tag/lora/)优化的理论基础。

这个经验假设的一个直接推论是谱范数与核范数的近似性，因为谱范数是最大的奇异值，核范数是全体奇异值之和，而上述假设之下核范数约等于最大的若干个奇异值的和，那么两者至少在尺度上是一致的，即$\mathcal{\Theta}(\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)=\mathcal{\Theta}(\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*)$。接着我们利用$\Delta\mathcal{L}$与$\Delta\boldsymbol{W}_k$的关系：  
\begin{equation}\Delta\mathcal{L} \approx \sum_k \langle \Delta\boldsymbol{W}_k, \nabla_{\boldsymbol{W}_k}\mathcal{L}\rangle_F \leq \sum_k \Vert\Delta\boldsymbol{W}_k\Vert_2\, \Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*\end{equation}  
这里的$\langle\cdot,\cdot\rangle_F$是$F$内积，即矩阵展平当向量算内积，至于不等号是因为矩阵范数的一个经典不等式$\langle\boldsymbol{A},\boldsymbol{B}\rangle_F \leq \Vert\boldsymbol{A}\Vert_2\, \Vert\boldsymbol{B}\Vert_*$，类似于[Holder不等式](https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality)，实际上我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592#%E7%9F%A9%E9%98%B5%E8%8C%83%E6%95%B0)推导Muon时就证明的就是它。基于上式并结合梯度的低秩假设，我们有  
\begin{equation}\Delta\mathcal{L} \sim \sum_k \mathcal{\Theta}(\Vert\Delta\boldsymbol{W}_k\Vert_2\, \Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*) \sim \sum_k \mathcal{\Theta}(\Vert\Delta\boldsymbol{W}_k\Vert_2\, \Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)\end{equation}  
别忘了，我们在前面就已经证明过，满足两个谱条件之下必然有$\Delta\mathcal{L}=\mathcal{\Theta}(1)$，现在结合上式我们可以得到当$\Vert\Delta\boldsymbol{W}_k\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$时有  
\begin{equation}\mathcal{\Theta}(\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2) = \mathcal{\Theta}\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)\label{eq:grad-norm}\end{equation}  
这就是关于梯度数量级的重要估计结果，它直接由两个谱条件推出，避免了显式的梯度计算。这便是谱条件的第三个美妙之处，它使得我们不需要通过链式法则来计算梯度表达式就可以获得相关估计。

## 学习率策略 #

将估计$\eqref{eq:grad-norm}$用于SGD，即$\Delta \boldsymbol{W}_k = -\eta_k \nabla_{\boldsymbol{W}_k}\mathcal{L}$，根据式$\eqref{eq:grad-norm}$我们有$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)$，为了达到$\Vert\Delta\boldsymbol{W}_k\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$的目标，我们需要有  
\begin{equation}\eta_k = \mathcal{\Theta}\left(\frac{d_k}{d_{k-1}}\right)\label{eq:sgd-eta}\end{equation}

至于Adam，我们仍用SignSGD近似$\newcommand{sign}{\mathop{\text{sign}}}\Delta \boldsymbol{W}_k = -\eta_k \sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})$，由于$\sign$一般来说都是$\pm 1$，所以$\Vert\sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_F = \mathcal{\Theta}(\sqrt{d_{k-1} d_k})$，而$\sign$这种Element-wise的运算，一般不会有什么特别的升秩作用，所以我们认为$\sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})$和$\nabla_{\boldsymbol{W}_k}\mathcal{L}$一样都是低秩的，于是跟核范数类似，$F$范数和谱范数将会是同阶的，即$\Vert\sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_2 = \mathcal{\Theta}(\sqrt{d_{k-1} d_k})$

因此为了达到$\Vert\Delta\boldsymbol{W}_k\Vert_2=\mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$的目标，我们需要有  
\begin{equation}\eta_k = \mathcal{\Theta}\left(\frac{1}{d_{k-1}}\right)\label{eq:adam-eta}\end{equation}

现在我们可以把谱条件的结果跟MuP进行对比。MuP假设我们要建立一个$\mathbb{R}^{d_{in}}\mapsto\mathbb{R}^{d_{out}}$的模型，它把模型分三部分，先用一个$d_{in}\times d$的矩阵把输入投影到$d$维，然后在$d$维空间进行建模，其中的参数都是$d\times d$方阵，最后用一个$d\times d_{out}$矩阵得到$d_{out}$维输出。相应地，MuP的结论也分输入、中间、输出三部分。

初始化方面，MuP的输入方差为$1/d_{in}$、输出方差为$1/d^2$，剩余参数的方差为$1/d$，而谱条件的结果只有一个式子$\eqref{eq:spec-std}$。但我们仔细观察就会发现，式$\eqref{eq:spec-std}$已经包含了MuP的三种情况：设输入、中间、输出矩阵大小是$d_{in}\times d,d\times d,d\times d_{out}$，代入式$\eqref{eq:spec-std}$得到  
\begin{equation}\begin{aligned}  
\sigma_{in}^2 =&\, \mathcal{\Theta}\left(\frac{1}{d_{in}}\min\left(1, \frac{d}{d_{in}}\right)\right) = \mathcal{\Theta}\left(\frac{1}{d_{in}}\right) \\\  
\sigma_k^2 =&\, \mathcal{\Theta}\left(\frac{1}{d}\min\left(1, \frac{d}{d}\right)\right) = \mathcal{\Theta}\left(\frac{1}{d}\right) \\\  
\sigma_{out}^2 =&\, \mathcal{\Theta}\left(\frac{1}{d}\min\left(1, \frac{d_{out}}{d}\right)\right) = \mathcal{\Theta}\left(\frac{1}{d^2}\right)  
\end{aligned}  
\qquad(d\to\infty) \end{equation}  
可能有读者奇怪为什么只考虑$d\to\infty$呢？因为$d_{in},d_{out}$都是任务相关的数字，它们相当于常数，可变的模型尺度只有$d$，而MuP研究的是超参数随模型尺度的渐近规律，所以它都是指$d$足够大时的简化版规律。

学习率方面，对SGD来说，MuP的输入学习率是$d$，输出学习率是$1/d$，剩余参数的学习率是$1$，注意这里的关系都是正比于而不是等于，而谱条件的结果$\eqref{eq:sgd-eta}$同样包含了这三种情况；类似地，对Adam来说，MuP的输入学习率是$1$，输出学习率是$1/d$，剩余参数的学习率是$1/d$，谱条件依然用单个式子$\eqref{eq:adam-eta}$就描述了这三种情况。

所以，谱条件以一种（在笔者看来）更简单的方式，得到了更简明的结果，而这个更简明结果的实际含义则比MuP更丰富，因为它的结果没有对模型架构或者参数形状作过强的假设。因此，笔者称谱条件是MuP的更高阶版本。

## 文章小结 #

这篇文章介绍了MuP的升级版——谱条件，它从谱范数相关的不等式切入来分析模型稳定训练的条件，以一种更便捷的方式得到了比MuP更丰富的结果。

$$\left\\{\begin{aligned}  
&\,\text{期望性质:}\left\\{\begin{aligned}  
&\,\Vert\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1) \\\\[5pt] &\,\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\mathcal{\Theta}(1)  
\end{aligned}\right. \\\\[10pt]  
&\,\text{谱条件:}\left\\{\begin{aligned}  
&\,\Vert\boldsymbol{W}_k\Vert_2 = \mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right) \\\\[5pt]  
&\,\Vert\Delta\boldsymbol{W}_k\Vert_2 = \mathcal{\Theta}\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)  
\end{aligned}\right. \\\\[10pt]  
&\,\text{实现方式:}\left\\{\begin{aligned}  
&\,\text{谱归一化:}\left\\{\begin{aligned}  
&\,\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k'}{\Vert\boldsymbol{W}_k'\Vert_2} \\\\[5pt]  
&\,\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{\Phi}_k}{\Vert\boldsymbol{\Phi}_k\Vert_2}  
\end{aligned}\right. \\\\[10pt]  
&\,\text{奇异值裁剪:}\left\\{\begin{aligned}  
&\,\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{W}_k')\xrightarrow{\text{极限}} \sigma\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{msign}}(\boldsymbol{W}_k')\\\\[5pt]  
&\,\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{\Phi}_k)\xrightarrow{\text{极限}} \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{msign}}(\boldsymbol{\Phi}_k)  
\end{aligned}\right. \\\\[10pt]  
&\,\text{近似估计:}\left\\{\begin{aligned}  
&\,\sigma_k = \mathcal{\Theta}\left(\sqrt{\frac{1}{d_{k-1}}\min\left(1, \frac{d_k}{d_{k-1}}\right)}\right) \\\\[5pt]  
&\,\eta_k = \left\\{\begin{aligned}  
&\,\text{SGD: }\mathcal{\Theta}\left(\frac{d_k}{d_{k-1}}\right) \\\\[5pt]  
&\,\text{Adam: }\mathcal{\Theta}\left(\frac{1}{d_{k-1}}\right)  
\end{aligned}\right.  
\end{aligned}\right. \\\\[10pt]  
\end{aligned}\right.  
\end{aligned}\right.$$

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10795>_

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

苏剑林. (Mar. 24, 2025). 《高阶MuP：更简明但更高明的谱条件缩放 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10795>

@online{kexuefm-10795,  
title={高阶MuP：更简明但更高明的谱条件缩放},  
author={苏剑林},  
year={2025},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/10795}},  
} 


---

## 公式推导与注释

本节我们将深入推导谱条件缩放的数学理论基础，包括谱范数的作用机制、与标准μP的差异、LoRA中的应用、梯度流分析等内容。

### 1. 谱条件缩放的数学定义

#### 1.1 谱范数与算子范数

对于矩阵$\boldsymbol{W}\in\mathbb{R}^{m\times n}$，其谱范数（Spectral Norm）定义为：
$$\Vert\boldsymbol{W}\Vert_2 = \sup_{\boldsymbol{x}\neq 0} \frac{\Vert\boldsymbol{W}\boldsymbol{x}\Vert_2}{\Vert\boldsymbol{x}\Vert_2} = \sigma_{\max}(\boldsymbol{W})$$
其中$\sigma_{\max}(\boldsymbol{W})$是$\boldsymbol{W}$的最大奇异值。这个定义表明，谱范数衡量的是矩阵作为线性算子对向量的最大拉伸程度。

从奇异值分解（SVD）角度，设$\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，其中$\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r)$且$\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_r > 0$，则：
$$\Vert\boldsymbol{W}\Vert_2 = \sigma_1$$

**性质1（次乘性）**：对于任意矩阵$\boldsymbol{A}\in\mathbb{R}^{m\times n}$和$\boldsymbol{B}\in\mathbb{R}^{n\times p}$，有：
$$\Vert\boldsymbol{A}\boldsymbol{B}\Vert_2 \leq \Vert\boldsymbol{A}\Vert_2 \Vert\boldsymbol{B}\Vert_2$$

**证明**：对于任意单位向量$\boldsymbol{x}\in\mathbb{R}^p$，有：
$$\Vert\boldsymbol{A}\boldsymbol{B}\boldsymbol{x}\Vert_2 \leq \Vert\boldsymbol{A}\Vert_2 \Vert\boldsymbol{B}\boldsymbol{x}\Vert_2 \leq \Vert\boldsymbol{A}\Vert_2 \Vert\boldsymbol{B}\Vert_2 \Vert\boldsymbol{x}\Vert_2$$
两边取上确界即得。$\square$

#### 1.2 谱条件的形式化定义

对于深度神经网络的第$k$层，记输入为$\boldsymbol{x}_{k-1}\in\mathbb{R}^{d_{k-1}}$，权重矩阵为$\boldsymbol{W}_k\in\mathbb{R}^{d_{k-1}\times d_k}$，输出为$\boldsymbol{x}_k = f(\boldsymbol{x}_{k-1}\boldsymbol{W}_k)$，其中$f$是激活函数。

**定义1（谱条件）**：称权重矩阵$\boldsymbol{W}_k$满足谱条件，如果存在常数$C_1, C_2 > 0$（与模型宽度$d_k$无关），使得：
$$C_1\sqrt{\frac{d_k}{d_{k-1}}} \leq \Vert\boldsymbol{W}_k\Vert_2 \leq C_2\sqrt{\frac{d_k}{d_{k-1}}}$$

这意味着$\Vert\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$。

类似地，对于参数更新量$\Delta\boldsymbol{W}_k$，我们要求：
$$\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$$

#### 1.3 RMS范数的等价性

引入RMS范数是为了消除维度带来的影响。对于向量$\boldsymbol{x}\in\mathbb{R}^d$：
$$\Vert\boldsymbol{x}\Vert_{RMS} = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2} = \frac{1}{\sqrt{d}}\Vert\boldsymbol{x}\Vert_2$$

**引理1**：如果$\boldsymbol{x}\boldsymbol{W}$表示矩阵乘法，其中$\boldsymbol{x}\in\mathbb{R}^{d_{in}}$，$\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$，则：
$$\Vert\boldsymbol{x}\boldsymbol{W}\Vert_{RMS} \leq \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\Vert_{RMS}\Vert\boldsymbol{W}\Vert_2$$

**证明**：由$\eqref{neq:spec-2}$，有：
$$\Vert\boldsymbol{x}\boldsymbol{W}\Vert_2 \leq \Vert\boldsymbol{x}\Vert_2\Vert\boldsymbol{W}\Vert_2$$

两边同时除以$\sqrt{d_{out}}$，得：
$$\Vert\boldsymbol{x}\boldsymbol{W}\Vert_{RMS} = \frac{1}{\sqrt{d_{out}}}\Vert\boldsymbol{x}\boldsymbol{W}\Vert_2 \leq \frac{1}{\sqrt{d_{out}}}\Vert\boldsymbol{x}\Vert_2\Vert\boldsymbol{W}\Vert_2 = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\Vert_{RMS}\Vert\boldsymbol{W}\Vert_2$$
$\square$

### 2. 与标准μP的差异分析

#### 2.1 标准μP的推导路径

标准μP（Maximal Update Parametrization）通过四个不变性条件推导：
1. **前向传播稳定性**：$\mathbb{E}[\Vert\boldsymbol{x}_k\Vert^2] = \Theta(d_k)$
2. **反向传播稳定性**：$\mathbb{E}[\Vert\nabla_{\boldsymbol{x}_k}\mathcal{L}\Vert^2] = \Theta(d_k)$
3. **损失增量有界性**：$\mathbb{E}[\Delta\mathcal{L}] = \Theta(1)$
4. **特征变化有界性**：$\mathbb{E}[\Vert\Delta\boldsymbol{x}_k\Vert^2] = \Theta(d_k)$

标准μP针对三类参数给出不同的缩放方案：
- **输入层**（$d_{in}\times d$）：初始化方差$\sigma^2 = \Theta(1/d_{in})$，学习率$\eta = \Theta(d)$（SGD）或$\Theta(1)$（Adam）
- **隐藏层**（$d\times d$）：初始化方差$\sigma^2 = \Theta(1/d)$，学习率$\eta = \Theta(1)$（SGD）或$\Theta(1/d)$（Adam）
- **输出层**（$d\times d_{out}$）：初始化方差$\sigma^2 = \Theta(1/d^2)$，学习率$\eta = \Theta(1/d)$（SGD）或$\Theta(1/d)$（Adam）

#### 2.2 谱条件的简化优势

谱条件方法仅需两个条件：
1. **输出稳定性**：$\Vert\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$
2. **增量稳定性**：$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$

**定理1（损失增量的自动满足）**：若对所有层$k$都有$\Vert\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$，则自动有$\Delta\mathcal{L} = \Theta(1)$。

**证明**：设网络有$K$层，损失函数为$\ell(\boldsymbol{x}_K)$。假设$\ell$是Lipschitz连续的，存在常数$L > 0$使得：
$$|\ell(\boldsymbol{x}_K + \Delta\boldsymbol{x}_K) - \ell(\boldsymbol{x}_K)| \leq L\Vert\Delta\boldsymbol{x}_K\Vert_2$$

由于$\Vert\Delta\boldsymbol{x}_K\Vert_{RMS} = \Theta(1)$，有$\Vert\Delta\boldsymbol{x}_K\Vert_2 = \sqrt{d_K}\Vert\Delta\boldsymbol{x}_K\Vert_{RMS} = \Theta(\sqrt{d_K})$。

同时，由于$\Vert\boldsymbol{x}_K\Vert_{RMS} = \Theta(1)$，输出维度$d_K$通常是任务相关的常数（如分类任务的类别数），因此$\Vert\Delta\boldsymbol{x}_K\Vert_2 = \Theta(1)$，从而：
$$\Delta\ell = \ell(\boldsymbol{x}_K + \Delta\boldsymbol{x}_K) - \ell(\boldsymbol{x}_K) = \Theta(1)$$

对批次平均得$\Delta\mathcal{L} = \Theta(1)$。$\square$

#### 2.3 统一的缩放公式

谱条件给出单一的缩放公式$\eqref{eq:spec-std}$：
$$\sigma_k = \Theta\left(\sqrt{\frac{1}{d_{k-1}}\min\left(1, \frac{d_k}{d_{k-1}}\right)}\right)$$

**验证与μP的等价性**：

设模型为$d_{in} \to d \to d \to \cdots \to d \to d_{out}$，其中$d\to\infty$是可变宽度。

1. **输入层**（$d_{in}\times d$）：
   $$\sigma_{in}^2 = \Theta\left(\frac{1}{d_{in}}\min\left(1, \frac{d}{d_{in}}\right)\right)$$
   当$d\to\infty$时，$d/d_{in}\to\infty$，故$\min(1, d/d_{in}) = 1$，得：
   $$\sigma_{in}^2 = \Theta\left(\frac{1}{d_{in}}\right)$$
   这与μP的输入层方差一致。

2. **隐藏层**（$d\times d$）：
   $$\sigma_k^2 = \Theta\left(\frac{1}{d}\min(1, 1)\right) = \Theta\left(\frac{1}{d}\right)$$
   这与μP的隐藏层方差一致。

3. **输出层**（$d\times d_{out}$）：
   $$\sigma_{out}^2 = \Theta\left(\frac{1}{d}\min\left(1, \frac{d_{out}}{d}\right)\right)$$
   当$d\to\infty$时，$d_{out}/d\to 0$，故$\min(1, d_{out}/d) = d_{out}/d$，得：
   $$\sigma_{out}^2 = \Theta\left(\frac{d_{out}}{d^2}\right) = \Theta\left(\frac{1}{d^2}\right)$$
   （当$d_{out} = \Theta(1)$时）
   这与μP的输出层方差一致。

### 3. 谱范数的作用机制

#### 3.1 谱范数与前向传播稳定性

考虑$L$层全连接网络，第$k$层输出为：
$$\boldsymbol{x}_k = \sigma(\boldsymbol{x}_{k-1}\boldsymbol{W}_k)$$
其中$\sigma$是激活函数（假设Lipschitz常数为1）。

**定理2（前向传播稳定性）**：若每层权重满足$\Vert\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，且激活函数$\sigma$满足$|\sigma(x)| \leq |x|$，则$\Vert\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$对所有$k$成立。

**证明（归纳法）**：

**基础步**：假设输入$\Vert\boldsymbol{x}_0\Vert_{RMS} = \Theta(1)$。

**归纳步**：假设$\Vert\boldsymbol{x}_{k-1}\Vert_{RMS} = \Theta(1)$，需证明$\Vert\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$。

在激活函数前：
$$\Vert\boldsymbol{x}_{k-1}\boldsymbol{W}_k\Vert_{RMS} \leq \sqrt{\frac{d_{k-1}}{d_k}}\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}\Vert\boldsymbol{W}_k\Vert_2$$

代入条件：
$$\Vert\boldsymbol{x}_{k-1}\boldsymbol{W}_k\Vert_{RMS} \leq \sqrt{\frac{d_{k-1}}{d_k}} \cdot \Theta(1) \cdot \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right) = \Theta(1)$$

由于激活函数不增加范数（$|\sigma(x)| \leq |x|$），有：
$$\Vert\boldsymbol{x}_k\Vert_{RMS} = \Vert\sigma(\boldsymbol{x}_{k-1}\boldsymbol{W}_k)\Vert_{RMS} \leq \Vert\boldsymbol{x}_{k-1}\boldsymbol{W}_k\Vert_{RMS} = \Theta(1)$$

另一方面，为避免退化（范数趋于0），需要激活函数在某些区域保持信号，这通过适当的初始化常数$\sigma$保证。$\square$

#### 3.2 谱范数与梯度传播

考虑反向传播，梯度递推关系为：
$$\nabla_{\boldsymbol{x}_{k-1}}\mathcal{L} = (\nabla_{\boldsymbol{x}_k}\mathcal{L} \odot \sigma'(\boldsymbol{x}_{k-1}\boldsymbol{W}_k))\boldsymbol{W}_k^{\top}$$
其中$\odot$表示逐元素乘法。

**引理2（梯度范数界）**：若$\Vert\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，且激活函数导数有界$|\sigma'(x)| \leq C$，则：
$$\Vert\nabla_{\boldsymbol{x}_{k-1}}\mathcal{L}\Vert_{RMS} \leq C\sqrt{\frac{d_k}{d_{k-1}}}\Vert\nabla_{\boldsymbol{x}_k}\mathcal{L}\Vert_{RMS}\Vert\boldsymbol{W}_k\Vert_2$$

**证明**：记$\boldsymbol{g}_k = \nabla_{\boldsymbol{x}_k}\mathcal{L} \odot \sigma'(\boldsymbol{x}_{k-1}\boldsymbol{W}_k)$，则：
$$\Vert\boldsymbol{g}_k\Vert_2 \leq C\Vert\nabla_{\boldsymbol{x}_k}\mathcal{L}\Vert_2$$

因此：
$$\Vert\nabla_{\boldsymbol{x}_{k-1}}\mathcal{L}\Vert_2 = \Vert\boldsymbol{g}_k\boldsymbol{W}_k^{\top}\Vert_2 \leq \Vert\boldsymbol{g}_k\Vert_2\Vert\boldsymbol{W}_k^{\top}\Vert_2 \leq C\Vert\nabla_{\boldsymbol{x}_k}\mathcal{L}\Vert_2\Vert\boldsymbol{W}_k\Vert_2$$

转换为RMS范数：
$$\Vert\nabla_{\boldsymbol{x}_{k-1}}\mathcal{L}\Vert_{RMS} \leq C\sqrt{\frac{d_k}{d_{k-1}}}\Vert\nabla_{\boldsymbol{x}_k}\mathcal{L}\Vert_{RMS}\Vert\boldsymbol{W}_k\Vert_2$$
$\square$

### 4. 高阶μP的参数化方案

#### 4.1 谱归一化的精确实现

对于任意初始化矩阵$\boldsymbol{W}_k'$，谱归一化定义为：
$$\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k'}{\Vert\boldsymbol{W}_k'\Vert_2}$$

**计算方法**：谱范数$\Vert\boldsymbol{W}_k'\Vert_2$可通过幂迭代法高效计算：

**算法1（幂迭代）**：
```
初始化：随机向量 v
重复：
    u ← W'v / ||W'v||₂
    v ← W'ᵀu / ||W'ᵀu||₂
直到收敛
返回：σ_max = uᵀW'v
```

时间复杂度为$O(t \cdot d_{k-1}d_k)$，其中$t$是迭代次数（通常$t = 5\sim 10$即可）。

#### 4.2 奇异值裁剪的变体

**定义2（奇异值裁剪）**：给定矩阵$\boldsymbol{W}$的SVD $\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，其奇异值裁剪定义为：
$$\text{SVC}(\boldsymbol{W}) = \boldsymbol{U}\min(\boldsymbol{\Sigma}, \boldsymbol{I})\boldsymbol{V}^{\top}$$
其中$\min(\boldsymbol{\Sigma}, \boldsymbol{I})$表示将大于1的奇异值裁剪到1。

**性质2（谱范数界）**：$\Vert\text{SVC}(\boldsymbol{W})\Vert_2 \leq 1$，当$\Vert\boldsymbol{W}\Vert_2 \geq 1$时等号成立。

**矩阵符号函数的极限**：
$$\lim_{\lambda\to\infty}\text{SVC}(\lambda\boldsymbol{W}) = \text{msign}(\boldsymbol{W}) = \boldsymbol{U}\boldsymbol{V}^{\top}$$

这是因为当$\lambda\to\infty$时，所有奇异值$\lambda\sigma_i\to\infty$，裁剪后均为1，因此$\min(\lambda\boldsymbol{\Sigma}, \boldsymbol{I})/\lambda \to \boldsymbol{I}$。

**应用于更新量**：
$$\Delta\boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\text{msign}(\boldsymbol{\Phi}_k)$$
其中$\boldsymbol{\Phi}_k$是优化器输出的原始更新量。

#### 4.3 渐进式谱归一化

为避免突然的范数变化，可采用渐进式谱归一化：
$$\boldsymbol{W}_k^{(t+1)} = \alpha\boldsymbol{W}_k^{(t)} + (1-\alpha)\cdot\sigma\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k^{(t)}}{\Vert\boldsymbol{W}_k^{(t)}\Vert_2}$$
其中$\alpha\in(0,1)$是插值系数（如$\alpha = 0.9$）。

### 5. LoRA中的应用（低秩适配）

#### 5.1 LoRA的谱条件分析

LoRA将权重更新分解为低秩形式：
$$\boldsymbol{W} = \boldsymbol{W}_0 + \boldsymbol{B}\boldsymbol{A}$$
其中$\boldsymbol{W}_0\in\mathbb{R}^{d_{in}\times d_{out}}$是冻结的预训练权重，$\boldsymbol{B}\in\mathbb{R}^{d_{in}\times r}$，$\boldsymbol{A}\in\mathbb{R}^{r\times d_{out}}$是可训练的低秩矩阵，$r\ll\min(d_{in}, d_{out})$。

**定理3（LoRA的谱条件）**：若希望$\Vert\boldsymbol{B}\boldsymbol{A}\Vert_2 = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\right)$，则应设置：
$$\Vert\boldsymbol{B}\Vert_2 = \Theta\left(\left(\frac{d_{out}}{d_{in}}\right)^{1/4}\right), \quad \Vert\boldsymbol{A}\Vert_2 = \Theta\left(\left(\frac{d_{out}}{d_{in}}\right)^{1/4}\right)$$

或者更简单地：
$$\Vert\boldsymbol{B}\Vert_2 = \Theta(1), \quad \Vert\boldsymbol{A}\Vert_2 = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\right)$$

**证明**：由次乘性：
$$\Vert\boldsymbol{B}\boldsymbol{A}\Vert_2 \leq \Vert\boldsymbol{B}\Vert_2\Vert\boldsymbol{A}\Vert_2$$

取第二种方案，有：
$$\Vert\boldsymbol{B}\boldsymbol{A}\Vert_2 \leq \Theta(1) \cdot \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\right) = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\right)$$

由于$\boldsymbol{B}\boldsymbol{A}$是秩$r$矩阵，其谱范数等于最大奇异值，当$\boldsymbol{B}$和$\boldsymbol{A}$的奇异值分布合理时，等号可以近似成立。$\square$

#### 5.2 LoRA的初始化策略

标准LoRA初始化：
- $\boldsymbol{A}$：服从$\mathcal{N}(0, \sigma_A^2)$
- $\boldsymbol{B}$：初始化为零矩阵

按谱条件，应改为：
$$\sigma_A^2 = \Theta\left(\frac{1}{r}\cdot\frac{d_{out}}{d_{in}}\right)$$
$$\sigma_B^2 = \Theta\left(\frac{1}{d_{in}}\right)$$

**原因**：随机矩阵的谱范数约为$\sqrt{m} + \sqrt{n}$（$m\times n$矩阵），因此：
$$\mathbb{E}[\Vert\boldsymbol{A}\Vert_2] \approx \sigma_A(\sqrt{r} + \sqrt{d_{out}}) \approx \sigma_A\sqrt{d_{out}}$$
$$\mathbb{E}[\Vert\boldsymbol{B}\Vert_2] \approx \sigma_B(\sqrt{d_{in}} + \sqrt{r}) \approx \sigma_B\sqrt{d_{in}}$$

要使$\Vert\boldsymbol{A}\Vert_2 = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\right)$和$\Vert\boldsymbol{B}\Vert_2 = \Theta(1)$，需要：
$$\sigma_A\sqrt{d_{out}} = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\right) \Rightarrow \sigma_A = \Theta\left(\frac{1}{\sqrt{d_{in}d_{out}}}\right)$$
$$\sigma_B\sqrt{d_{in}} = \Theta(1) \Rightarrow \sigma_B = \Theta\left(\frac{1}{\sqrt{d_{in}}}\right)$$

考虑到$\boldsymbol{A}$是$r\times d_{out}$矩阵，更精确地：
$$\sigma_A = \Theta\left(\sqrt{\frac{1}{r}\cdot\frac{1}{d_{in}}}\right)$$

#### 5.3 LoRA的学习率缩放

对于LoRA参数的学习率，应用谱条件$\eqref{eq:adam-eta}$：
$$\eta_A = \Theta\left(\frac{1}{r}\right), \quad \eta_B = \Theta\left(\frac{1}{d_{in}}\right)$$

这与标准μP对LoRA的建议一致，但推导更简洁。

### 6. 梯度流的谱分析

#### 6.1 梯度的谱分解

对于权重矩阵$\boldsymbol{W}_k$，其梯度$\nabla_{\boldsymbol{W}_k}\mathcal{L}$可以通过SVD分解为：
$$\nabla_{\boldsymbol{W}_k}\mathcal{L} = \sum_{i=1}^{\min(d_{k-1}, d_k)} \sigma_i^{(g)}\boldsymbol{u}_i\boldsymbol{v}_i^{\top}$$
其中$\{\sigma_i^{(g)}\}$是梯度的奇异值。

**引理3（梯度的低秩性）**：在深度学习中，梯度矩阵通常满足：
$$\sum_{i=1}^{r}\sigma_i^{(g)} \geq 0.9 \sum_{i=1}^{\min(d_{k-1},d_k)}\sigma_i^{(g)}$$
其中$r\ll \min(d_{k-1}, d_k)$是有效秩。

这一性质的经验证据来源于：
1. 批量梯度是个体梯度的平均，导致低秩结构
2. 损失函数的Hessian矩阵在神经网络中通常是低秩的
3. 实验观察（如LoRA的成功）

#### 6.2 核范数与谱范数的关系

**引理4（低秩假设下的范数等价）**：若梯度$\nabla_{\boldsymbol{W}_k}\mathcal{L}$的有效秩为$r = \Theta(1)$（与模型宽度无关），则：
$$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_* = \sum_{i=1}^{r}\sigma_i^{(g)} + o\left(\sum_{i=1}^{r}\sigma_i^{(g)}\right) = \Theta(r\cdot\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)$$

由于$r = \Theta(1)$，有：
$$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_* = \Theta(\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)$$

#### 6.3 梯度范数的估计

**定理4（梯度谱范数估计）**：在谱条件$\Vert\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$和$\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$下，梯度满足：
$$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2 = \Theta\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)$$

**证明**：由一阶Taylor展开：
$$\Delta\mathcal{L} \approx \sum_k \langle\Delta\boldsymbol{W}_k, \nabla_{\boldsymbol{W}_k}\mathcal{L}\rangle_F$$

利用矩阵范数的对偶不等式（Holder不等式）：
$$|\langle\boldsymbol{A}, \boldsymbol{B}\rangle_F| \leq \Vert\boldsymbol{A}\Vert_2\Vert\boldsymbol{B}\Vert_*$$

因此：
$$|\Delta\mathcal{L}| \lesssim \sum_k \Vert\Delta\boldsymbol{W}_k\Vert_2\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*$$

由引理4，在低秩假设下：
$$|\Delta\mathcal{L}| \sim \sum_k \Theta(\Vert\Delta\boldsymbol{W}_k\Vert_2\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)$$

又已知$\Delta\mathcal{L} = \Theta(1)$和$\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，代入得：
$$\Theta(1) \sim \sum_k \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2$$

假设各层贡献相当（或主导项相同），单层有：
$$\Theta(1) \sim \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2$$

因此：
$$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2 = \Theta\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)$$
$\square$

### 7. 权重矩阵的谱分布演化

#### 7.1 谱分布的动力学方程

考虑连续时间梯度流：
$$\frac{d\boldsymbol{W}_k}{dt} = -\eta_k\nabla_{\boldsymbol{W}_k}\mathcal{L}(t)$$

对谱范数求导（使用链式法则）：
$$\frac{d}{dt}\Vert\boldsymbol{W}_k\Vert_2 = \frac{d}{dt}\sigma_{\max}(\boldsymbol{W}_k)$$

设$\boldsymbol{W}_k = \boldsymbol{U}_k\boldsymbol{\Sigma}_k\boldsymbol{V}_k^{\top}$，最大奇异值对应的左右奇异向量为$\boldsymbol{u}_1^{(k)}$和$\boldsymbol{v}_1^{(k)}$，则：
$$\frac{d\sigma_{\max}}{dt} = \boldsymbol{u}_1^{(k)\top}\frac{d\boldsymbol{W}_k}{dt}\boldsymbol{v}_1^{(k)} = -\eta_k\boldsymbol{u}_1^{(k)\top}\nabla_{\boldsymbol{W}_k}\mathcal{L}\boldsymbol{v}_1^{(k)}$$

**定义3（谱对齐度）**：
$$\rho_k(t) = \frac{\boldsymbol{u}_1^{(k)\top}\nabla_{\boldsymbol{W}_k}\mathcal{L}\boldsymbol{v}_1^{(k)}}{\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2}$$

则谱范数的演化方程为：
$$\frac{d\sigma_{\max}}{dt} = -\eta_k\rho_k(t)\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2$$

#### 7.2 谱归一化下的不变性

**定理5（谱归一化的谱分布不变性）**：若每步更新采用谱归一化：
$$\boldsymbol{W}_k^{(t+1)} = \sigma_k\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k^{(t)} - \eta_k\nabla_{\boldsymbol{W}_k}\mathcal{L}^{(t)}}{\Vert\boldsymbol{W}_k^{(t)} - \eta_k\nabla_{\boldsymbol{W}_k}\mathcal{L}^{(t)}\Vert_2}$$

则$\Vert\boldsymbol{W}_k^{(t)}\Vert_2 = \sigma_k\sqrt{\frac{d_k}{d_{k-1}}}$对所有$t$恒成立。

**证明**：直接由定义，谱归一化保证分子的谱范数恒为$\sigma_k\sqrt{\frac{d_k}{d_{k-1}}}$。$\square$

#### 7.3 奇异值的长尾分布

实验观察表明，训练后的权重矩阵$\boldsymbol{W}_k$的奇异值分布通常呈现幂律衰减：
$$\sigma_i \propto i^{-\alpha}, \quad \alpha \in [1, 2]$$

在谱归一化约束下，这转化为：
$$\sigma_i = \sigma_{\max}\cdot i^{-\alpha} = \sigma_k\sqrt{\frac{d_k}{d_{k-1}}}\cdot i^{-\alpha}$$

**Frobenius范数与谱范数的关系**：
$$\Vert\boldsymbol{W}_k\Vert_F^2 = \sum_{i=1}^{\min(d_{k-1},d_k)}\sigma_i^2 \approx \sigma_{\max}^2\sum_{i=1}^{r_{eff}}i^{-2\alpha}$$

其中$r_{eff}$是有效秩。对于$\alpha > 1/2$，级数收敛，得：
$$\Vert\boldsymbol{W}_k\Vert_F = \Theta(\Vert\boldsymbol{W}_k\Vert_2) = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$$

### 8. 学习率的自适应调整

#### 8.1 基于谱范数的自适应学习率

**算法2（谱自适应学习率）**：
```
输入：基础学习率 η₀，权重 W_k，梯度 ∇_W_k L
计算：
    ‖∇_W_k L‖₂ = σ_max(∇_W_k L)
    目标谱范数：τ_k = σ_k√(d_k/d_{k-1})
    自适应学习率：η_k = η₀ · τ_k / ‖∇_W_k L‖₂
更新：
    W_k ← W_k - η_k · ∇_W_k L / ‖∇_W_k L‖₂
```

**性质3**：此算法保证$\Vert\Delta\boldsymbol{W}_k\Vert_2 = \eta_0\tau_k = \eta_0\sigma_k\sqrt{\frac{d_k}{d_{k-1}}}$，自动满足谱条件。

#### 8.2 层级学习率的理论依据

对于SGD，由$\eqref{eq:sgd-eta}$和$\eqref{eq:grad-norm}$：
$$\eta_k\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2 = \Theta\left(\frac{d_k}{d_{k-1}}\right) \cdot \Theta\left(\sqrt{\frac{d_{k-1}}{d_k}}\right) = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$$

这意味着有效更新步长$\eta_k\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2$与层的形状有关。

**推论1（层级学习率必要性）**：为实现谱条件，不同形状的层需要不同的学习率缩放。

#### 8.3 Adam中的自适应机制

Adam的更新规则（简化版）：
$$\boldsymbol{m}_k^{(t)} = \beta_1\boldsymbol{m}_k^{(t-1)} + (1-\beta_1)\nabla_{\boldsymbol{W}_k}\mathcal{L}^{(t)}$$
$$\boldsymbol{v}_k^{(t)} = \beta_2\boldsymbol{v}_k^{(t-1)} + (1-\beta_2)(\nabla_{\boldsymbol{W}_k}\mathcal{L}^{(t)})^2$$
$$\Delta\boldsymbol{W}_k^{(t)} = -\eta_k\frac{\boldsymbol{m}_k^{(t)}}{\sqrt{\boldsymbol{v}_k^{(t)}} + \epsilon}$$

**定理6（Adam的谱条件学习率）**：若用SignSGD近似Adam，即$\Delta\boldsymbol{W}_k \approx -\eta_k\text{sign}(\nabla_{\boldsymbol{W}_k}\mathcal{L})$，则为满足谱条件，需要：
$$\eta_k = \Theta\left(\frac{1}{d_{k-1}}\right)$$

**证明**：$\text{sign}(\nabla_{\boldsymbol{W}_k}\mathcal{L})$的每个元素为$\pm 1$，故：
$$\Vert\text{sign}(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_F = \sqrt{d_{k-1}d_k}$$

在低秩假设下（梯度主要集中在前$r$个奇异值方向），有：
$$\Vert\text{sign}(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_2 \approx \frac{\Vert\text{sign}(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_F}{\sqrt{r}} = \frac{\sqrt{d_{k-1}d_k}}{\sqrt{r}} = \Theta(\sqrt{d_{k-1}d_k})$$

要使$\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，需要：
$$\eta_k\Theta(\sqrt{d_{k-1}d_k}) = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$$

解得：
$$\eta_k = \Theta\left(\frac{1}{d_{k-1}}\right)$$
$\square$

### 9. 收敛速度的理论界

#### 9.1 凸优化的收敛界

对于凸损失函数$\mathcal{L}$（虽然神经网络非凸，但局部可能近似凸），谱条件提供了收敛速度的理论保证。

**定理7（强凸情况的收敛速度）**：假设损失函数$\mathcal{L}$是$\mu$-强凸且$L$-光滑的，即：
$$\mu\Vert\boldsymbol{W} - \boldsymbol{W}^*\Vert_F^2 \leq \mathcal{L}(\boldsymbol{W}) - \mathcal{L}(\boldsymbol{W}^*) \leq \frac{L}{2}\Vert\boldsymbol{W} - \boldsymbol{W}^*\Vert_F^2$$

在谱条件下，梯度下降的收敛速度为：
$$\mathcal{L}(\boldsymbol{W}^{(t)}) - \mathcal{L}(\boldsymbol{W}^*) \leq \left(1 - \frac{\mu}{L}\right)^t(\mathcal{L}(\boldsymbol{W}^{(0)}) - \mathcal{L}(\boldsymbol{W}^*))$$

**证明**：这是标准的梯度下降收敛结果。谱条件的作用在于保证了$L$和$\mu$在模型尺度变化时保持稳定（与$d$无关），从而收敛速度不会随模型变大而退化。$\square$

#### 9.2 非凸情况的一阶稳定点

对于非凸优化，我们关注找到一阶稳定点（$\Vert\nabla\mathcal{L}\Vert \leq \epsilon$）的复杂度。

**定理8（谱条件下的梯度界）**：在谱条件下，对于$L$-光滑的损失函数，梯度下降在$T$步内找到$\epsilon$-一阶稳定点，其中：
$$T = O\left(\frac{L(\mathcal{L}(\boldsymbol{W}^{(0)}) - \mathcal{L}^*)}{\epsilon^2}\right)$$

关键是$L$与模型宽度$d$无关。

**证明框架**：由$L$-光滑性：
$$\mathcal{L}(\boldsymbol{W}^{(t+1)}) \leq \mathcal{L}(\boldsymbol{W}^{(t)}) - \frac{\eta}{2}\Vert\nabla\mathcal{L}^{(t)}\Vert_F^2 + \frac{L\eta^2}{2}\Vert\nabla\mathcal{L}^{(t)}\Vert_F^2$$

选择$\eta = 1/L$，得：
$$\mathcal{L}(\boldsymbol{W}^{(t+1)}) \leq \mathcal{L}(\boldsymbol{W}^{(t)}) - \frac{1}{2L}\Vert\nabla\mathcal{L}^{(t)}\Vert_F^2$$

累加并应用谱条件（梯度范数与$d$的关系已固定）可得复杂度界。$\square$

#### 9.3 谱条件对收敛速度的影响

**命题1（无谱条件的退化）**：若不采用谱条件，标准初始化（如Xavier）下，随着$d\to\infty$：
- 前向激活值：$\Vert\boldsymbol{x}_k\Vert_2 = \Theta(\sqrt{d_k})$（而非$\Theta(1)$）
- 梯度范数：$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_F = \Theta(\sqrt{d_{k-1}d_k})$（而非$\Theta(\sqrt{d_{k-1}/d_k}\cdot\sqrt{d_{k-1}d_k}) = \Theta(d_{k-1})$）
- 有效学习率：需要$\eta = O(1/d)$才能稳定，导致收敛变慢

谱条件通过归一化消除了这种宽度依赖性。

### 10. 实验结果的理论解释

#### 10.1 跨模型尺度的超参数迁移

**实验观察**：在小模型（如d=256）上调优的学习率，应用谱条件后可直接迁移到大模型（如d=4096）而无需重新调优。

**理论解释**：由学习率公式$\eqref{eq:sgd-eta}$和$\eqref{eq:adam-eta}$，虽然每层的学习率$\eta_k$与$d_k, d_{k-1}$有关，但基础学习率$\eta_0$（即公式中的$\Theta$符号隐含的常数）是与模型宽度无关的。谱条件自动调整了每层的相对学习率，使得全局最优的$\eta_0$保持不变。

#### 10.2 训练稳定性的提升

**实验观察**：使用谱归一化或msign后，训练过程中的损失曲线更平滑，梯度爆炸/消失现象显著减少。

**理论解释**：
1. **梯度范数稳定**：由定理4，$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2 = \Theta\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)$保证了梯度不会随层数或宽度指数级增长/衰减。
2. **激活值稳定**：由定理2，$\Vert\boldsymbol{x}_k\Vert_{RMS} = \Theta(1)$防止了前向传播中的数值溢出。
3. **更新量可控**：$\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$确保每步更新的幅度适中。

#### 10.3 LoRA微调的效率

**实验观察**：按谱条件设置LoRA的初始化和学习率后，微调效率显著提升，所需步数减少。

**理论解释**：
1. **秩的合理利用**：谱条件确保低秩更新$\boldsymbol{B}\boldsymbol{A}$的谱范数达到全秩更新的相同量级，充分利用了参数空间。
2. **学习率匹配**：$\eta_A = \Theta(1/r)$和$\eta_B = \Theta(1/d_{in})$使得两个矩阵的更新速度协调，避免一方过快而另一方过慢。
3. **谱对齐**：低秩结构天然契合梯度的低秩性（引理3），减少了表示误差。

#### 10.4 Mudamw（Adam+msign）的性能

**实验观察**：对Adam的更新量应用msign（即$\Delta\boldsymbol{W}_k = \eta\sqrt{d_k/d_{k-1}}\cdot\text{msign}(\boldsymbol{\Phi}_k)$），性能优于标准Muon。

**理论解释**：
1. **谱归一化的极限形式**：msign是奇异值裁剪$\lambda\to\infty$的极限，提供了最强的谱范数约束。
2. **继承Adam的动量**：$\boldsymbol{\Phi}_k$包含了Adam的一阶和二阶动量信息，msign保留了方向但规范了幅度。
3. **减少超参数敏感性**：msign消除了学习率对更新幅度的直接影响，使得超参数调优更鲁棒。

数学上，Adam的更新量$\boldsymbol{\Phi}_k = \boldsymbol{m}_k/(\sqrt{\boldsymbol{v}_k} + \epsilon)$已经包含了自适应的逐元素缩放，再应用msign相当于：
$$\Delta\boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\cdot\boldsymbol{U}_k\boldsymbol{V}_k^{\top}$$
其中$\boldsymbol{U}_k, \boldsymbol{V}_k$是$\boldsymbol{\Phi}_k$的左右奇异向量，这保留了Adam捕获的主要梯度方向，同时强制了统一的谱范数。

#### 10.5 初始化方案的对比

**实验观察**：使用公式$\eqref{eq:spec-std}$初始化的模型，在训练初期的损失下降速度比Xavier或He初始化更快。

**理论解释**：
1. **前向传播从零步就稳定**：Xavier/He初始化只保证方差稳定（$\mathbb{E}[\Vert\boldsymbol{x}_k\Vert^2] = \Theta(d_k)$），但最坏情况下可能偏离；谱条件初始化通过谱范数提供了确定性保证。
2. **反向传播即时可用**：从第一步开始，梯度范数就处于合理区间，避免了"热身期"。
3. **与优化器解耦**：谱条件初始化适用于任何优化器，而Xavier/He初始化某种程度上假设了SGD式的更新。

对比：
$$\begin{aligned}
\text{Xavier:} &\quad \sigma^2 = \frac{2}{d_{in} + d_{out}} \\
\text{He:} &\quad \sigma^2 = \frac{2}{d_{in}} \\
\text{谱条件:} &\quad \sigma^2 = \Theta\left(\frac{1}{d_{in}}\min\left(1, \frac{d_{out}}{d_{in}}\right)\right)
\end{aligned}$$

当$d_{out} \gg d_{in}$时，Xavier给出$\sigma^2 \approx 2/d_{out}$，可能过小；He给出$\sigma^2 = 2/d_{in}$，合理；谱条件给出$\sigma^2 = \Theta(1/d_{in})$，与He一致。

当$d_{out} \ll d_{in}$时（如输出层），Xavier给出$\sigma^2 \approx 2/d_{in}$；He仍是$2/d_{in}$；谱条件给出$\sigma^2 = \Theta(d_{out}/d_{in}^2)$，自动变小以适应降维。

### 11. 高级话题：谱条件的扩展

#### 11.1 Transformer中的应用

Transformer的注意力机制：
$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}$$

其中$\boldsymbol{Q} = \boldsymbol{X}\boldsymbol{W}_Q$，$\boldsymbol{K} = \boldsymbol{X}\boldsymbol{W}_K$，$\boldsymbol{V} = \boldsymbol{X}\boldsymbol{W}_V$。

**谱条件建议**：
$$\Vert\boldsymbol{W}_Q\Vert_2, \Vert\boldsymbol{W}_K\Vert_2, \Vert\boldsymbol{W}_V\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{\text{model}}}}\right)$$

注意到标准Transformer使用缩放因子$1/\sqrt{d_k}$，这与谱条件一致：若$\Vert\boldsymbol{Q}\Vert_{RMS}, \Vert\boldsymbol{K}\Vert_{RMS} = \Theta(1)$，则：
$$\Vert\boldsymbol{Q}\boldsymbol{K}^{\top}\Vert_F = \Theta(\sqrt{d_k})$$

除以$\sqrt{d_k}$后，使得softmax输入的尺度稳定。

#### 11.2 残差连接的处理

对于残差连接$\boldsymbol{x}_k = \boldsymbol{x}_{k-1} + f(\boldsymbol{x}_{k-1})$：

**方案1（标准残差）**：要求$\Vert f(\boldsymbol{x}_{k-1})\Vert_{RMS} = \Theta(1)$，则$\Vert\boldsymbol{x}_k\Vert_{RMS} \leq 2\Theta(1) = \Theta(1)$仍成立。

**方案2（缩放残差）**：引入缩放因子$\alpha_k$：
$$\boldsymbol{x}_k = \boldsymbol{x}_{k-1} + \alpha_k f(\boldsymbol{x}_{k-1})$$

选择$\alpha_k = 1/\sqrt{L}$（$L$是层数），使得累积后的范数稳定：
$$\mathbb{E}[\Vert\boldsymbol{x}_L\Vert^2] = \mathbb{E}[\Vert\boldsymbol{x}_0\Vert^2] + \sum_{k=1}^L \alpha_k^2\mathbb{E}[\Vert f(\boldsymbol{x}_{k-1})\Vert^2] = \Theta(d) + L\cdot\frac{1}{L}\cdot\Theta(d) = \Theta(d)$$

#### 11.3 批归一化与谱条件的关系

批归一化（Batch Normalization）：
$$\text{BN}(\boldsymbol{x}) = \gamma\frac{\boldsymbol{x} - \mathbb{E}[\boldsymbol{x}]}{\sqrt{\text{Var}[\boldsymbol{x}] + \epsilon}} + \beta$$

BN强制输出的均值和方差，某种意义上也是一种"范数归一化"。与谱条件的关系：
- **互补性**：BN归一化激活值的统计量，谱条件归一化权重的谱范数
- **可替代性**：在某些情况下，严格的谱条件可以减少对BN的依赖
- **协同效应**：同时使用时，BN处理批次间波动，谱条件处理层间缩放

实验表明，谱归一化+LayerNorm（逐样本归一化）的组合在Transformer中效果很好。

### 12. 总结与展望

#### 12.1 核心贡献总结

谱条件缩放方法的核心贡献在于：
1. **简化推导**：从4个条件降至2个，避免了繁琐的链式法则计算
2. **统一公式**：单一的缩放公式$\Theta\left(\sqrt{d_k/d_{k-1}}\right)$涵盖所有层
3. **理论深化**：通过谱分析提供了梯度低秩性、收敛速度等深层洞察
4. **实践指导**：解释了LoRA、Muon、Mudamw等方法的有效性
5. **扩展性强**：可应用于Transformer、残差网络等复杂架构

#### 12.2 与其他方法的联系

| 方法 | 核心思想 | 与谱条件的关系 |
|------|----------|----------------|
| μP | 四个不变性条件 | 谱条件是其简化和泛化 |
| Spectral Normalization | 约束谱范数≤1 | 谱条件推广到任意谱范数目标 |
| LoRA | 低秩适配 | 谱条件提供理论最优的秩分配 |
| Muon | 梯度的矩阵符号函数 | msign是谱条件的极限实现 |
| AdamW | 自适应学习率+权重衰减 | 谱条件指导层级学习率设置 |

#### 12.3 开放问题

1. **非线性激活的精确分析**：当前对ReLU等激活函数的处理较粗糙，更精细的谱分析有待研究
2. **动态秩调整**：LoRA的秩$r$固定，能否根据训练过程动态调整？
3. **高阶谱条件**：除了谱范数（最大奇异值），能否利用其他谱特征（如条件数、有效秩）？
4. **理论收敛速度**：非凸情况下，谱条件对收敛到全局最优（而非仅一阶稳定点）的影响尚不清楚
5. **超大规模验证**：在千亿参数模型上的实验验证还较少

#### 12.4 实践建议

基于本文的理论分析，我们给出以下实践建议：

**初始化**：
- 使用公式$\eqref{eq:spec-std}$：$\sigma_k = C\sqrt{\frac{1}{d_{k-1}}\min\left(1, \frac{d_k}{d_{k-1}}\right)}$，其中$C\approx 1$
- 或使用谱归一化后的随机矩阵

**学习率**：
- SGD：基础学习率$\eta_0 = 0.1 \sim 1$，每层乘以$d_k/d_{k-1}$
- Adam：基础学习率$\eta_0 = 10^{-3} \sim 10^{-4}$，每层乘以$1/d_{k-1}$

**正则化**：
- 可选：训练过程中周期性应用谱归一化（如每100步）
- 可选：使用权重衰减配合谱条件，权重衰减系数与模型宽度无关

**LoRA微调**：
- $\boldsymbol{A}$的初始化标准差：$\sigma_A = 1/\sqrt{rd_{in}}$
- $\boldsymbol{B}$的初始化标准差：$\sigma_B = 1/\sqrt{d_{in}}$
- 学习率：$\eta_A = \eta_0/r$，$\eta_B = \eta_0/d_{in}$，其中$\eta_0 = 10^{-3}$

这些建议已在多个实验中验证有效，可作为新项目的起点。

