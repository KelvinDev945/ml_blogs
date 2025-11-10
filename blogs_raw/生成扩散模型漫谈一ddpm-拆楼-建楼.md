---
title: 生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼
slug: 生成扩散模型漫谈一ddpm-拆楼-建楼
date: 2022-06-13
tags: VAE, GAN, flow模型, 概率, 生成模型
status: pending
---

# 生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼

**原文链接**: [https://spaces.ac.cn/archives/9119](https://spaces.ac.cn/archives/9119)

**发布日期**: 

---

说到生成模型，[VAE](/tag/vae/)、[GAN](/tag/GAN/)可谓是“如雷贯耳”，本站也有过多次分享。此外，还有一些比较小众的选择，如[flow模型](/tag/flow/)、[VQ-VAE](/archives/6760)等，也颇有人气，尤其是VQ-VAE及其变体[VQ-GAN](https://papers.cool/arxiv/2012.09841)，近期已经逐渐发展到“图像的Tokenizer”的地位，用来直接调用NLP的各种预训练方法。除了这些之外，还有一个本来更小众的选择——扩散模型（Diffusion Models）——正在生成模型领域“异军突起”，当前最先进的两个文本生成图像——OpenAI的[DALL·E 2](https://papers.cool/arxiv/2204.06125)和Google的[Imagen](https://papers.cool/arxiv/2205.11487)，都是基于扩散模型来完成的。

[![Imagen“文本-图片”的部分例子](/usr/uploads/2022/06/2782509104.jpg)](/usr/uploads/2022/06/2782509104.jpg "点击查看原图")

Imagen“文本-图片”的部分例子

从本文开始，我们开一个新坑，逐渐介绍一下近两年关于生成扩散模型的一些进展。据说生成扩散模型以数学复杂闻名，似乎比VAE、GAN要难理解得多，是否真的如此？扩散模型真的做不到一个“大白话”的理解？让我们拭目以待。

## 新的起点 #

其实我们在之前的文章[《能量视角下的GAN模型（三）：生成模型=能量模型》](/archives/6612)、[《从去噪自编码器到生成模型》](/archives/7038)也简单介绍过扩散模型。说到扩散模型，一般的文章都会提到能量模型（Energy-based Models）、得分匹配（Score Matching）、朗之万方程（Langevin Equation）等等，简单来说，是通过得分匹配等技术来训练能量模型，然后通过郎之万方程来执行从能量模型的采样。

从理论上来讲，这是一套很成熟的方案，原则上可以实现任何连续型对象（语音、图像等）的生成和采样。但从实践角度来看，能量函数的训练是一件很艰难的事情，尤其是数据维度比较大（比如高分辨率图像）时，很难训练出完备能量函数来；另一方面，通过朗之万方程从能量模型的采样也有很大的不确定性，得到的往往是带有噪声的采样结果。所以很长时间以来，这种传统路径的扩散模型只是在比较低分辨率的图像上做实验。

如今生成扩散模型的大火，则是始于2020年所提出的[DDPM](https://papers.cool/arxiv/2006.11239)（Denoising Diffusion Probabilistic Model），虽然也用了“扩散模型”这个名字，但事实上除了采样过程的形式有一定的相似之外，DDPM与传统基于朗之万方程采样的扩散模型可以说完全不一样，这完全是一个新的起点、新的篇章。

准确来说，DDPM叫“渐变模型”更为准确一些，扩散模型这一名字反而容易造成理解上的误解，传统扩散模型的能量模型、得分匹配、朗之万方程等概念，其实跟DDPM及其后续变体都没什么关系。有意思的是，DDPM的数学框架其实在ICML2015的论文[《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》](https://papers.cool/arxiv/1503.03585)就已经完成了，但DDPM是首次将它在高分辨率图像生成上调试出来了，从而引导出了后面的火热。由此可见，一个模型的诞生和流行，往往还需要时间和机遇，

## 拆楼建楼 #

很多文章在介绍DDPM时，上来就引入转移分布，接着就是变分推断，一堆数学记号下来，先吓跑了一群人（当然，从这种介绍我们可以再次看出，DDPM实际上是VAE而不是扩散模型），再加之人们对传统扩散模型的固有印象，所以就形成了“需要很高深的数学知识”的错觉。事实上，DDPM也可以有一种很“大白话”的理解，它并不比有着“造假-鉴别”通俗类比的GAN更难。

首先，我们想要做一个像GAN那样的生成模型，它实际上是将一个随机噪声$\boldsymbol{z}$变换成一个数据样本$\boldsymbol{x}$的过程：  
\begin{equation}\require{AMScd}\begin{CD}  
\text{随机噪声}\boldsymbol{z}\quad @>\quad\text{变换}\quad>> \quad\text{样本数据}\boldsymbol{x}\\\  
@V \text{类比} VV @VV \text{类比} V\\\  
\text{砖瓦水泥}\quad @>\quad\text{建设}\quad>> \quad\text{高楼大厦}\\\  
\end{CD}\end{equation}

[![请叫我工程师](/usr/uploads/2022/06/403506617.jpeg)](/usr/uploads/2022/06/403506617.jpeg "点击查看原图")

请叫我工程师

我们可以将这个过程想象为“建设”，其中随机噪声$\boldsymbol{z}$是砖瓦水泥等原材料，样本数据$\boldsymbol{x}$是高楼大厦，所以生成模型就是一支用原材料建设高楼大厦的施工队。

这个过程肯定很难的，所以才有了那么多关于生成模型的研究。但俗话说“破坏容易建设难”，建楼你不会，拆楼你总会了吧？我们考虑将高楼大厦一步步地拆为砖瓦水泥的过程：设$\boldsymbol{x}_0$为建好的高楼大厦（数据样本），$\boldsymbol{x}_T$为拆好的砖瓦水泥（随机噪声），假设“拆楼”需要$T$步，整个过程可以表示为  
\begin{equation}\boldsymbol{x} = \boldsymbol{x}_0 \to \boldsymbol{x}_1 \to \boldsymbol{x}_2 \to \cdots \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_T = \boldsymbol{z}\end{equation}  
建高楼大厦的难度在于，从原材料$\boldsymbol{x}_T$到最终高楼大厦$\boldsymbol{x}_0$的跨度过大，普通人很难理解$\boldsymbol{x}_T$是怎么一下子变成$\boldsymbol{x}_0$的。但是，当我们有了“拆楼”的中间过程$\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_T$后，我们知道$\boldsymbol{x}_{t-1} \to \boldsymbol{x}_t$代表着拆楼的一步，那么反过来$\boldsymbol{x}_t\to \boldsymbol{x}_{t-1}$不就是建楼的一步？如果我们能学会两者之间的变换关系$\boldsymbol{x}_{t-1}=\boldsymbol{\mu}(\boldsymbol{x}_t)$，那么从$\boldsymbol{x}_T$出发，反复地执行$\boldsymbol{x}_{T-1}=\boldsymbol{\mu}(\boldsymbol{x}_T)$、$\boldsymbol{x}_{T-2}=\boldsymbol{\mu}(\boldsymbol{x}_{T-1})$、...，最终不就能造出高楼大厦$\boldsymbol{x}_0$出来？

## 该怎么拆 #

正所谓“饭要一口一口地吃”，楼也要一步一步地建，DDPM做生成模型的过程，其实跟上述“拆楼-建楼”的类比是完全一致的，它也是先反过来构建一个从数据样本渐变到随机噪声的过程，然后再考虑其逆变换，通过反复执行逆变换来完成数据样本的生成，所以本文前面才说DDPM这种做法其实应该更准确地称为“渐变模型”而不是“扩散模型”。

具体来说，DDPM将“拆楼”的过程建模为  
\begin{equation}\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t,\quad \boldsymbol{\varepsilon}_t\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:forward}\end{equation}  
其中有$\alpha_t,\beta_t > 0$且$\alpha_t^2 + \beta_t^2=1$，$\beta_t$通常很接近于0，代表着单步“拆楼”中对原来楼体的破坏程度，噪声$\boldsymbol{\varepsilon}_t$的引入代表着对原始信号的一种破坏，我们也可以将它理解为“原材料”，即每一步“拆楼”中我们都将$\boldsymbol{x}_{t-1}$拆解为“$\alpha_t \boldsymbol{x}_{t-1}$的楼体 + $\beta_t \boldsymbol{\varepsilon}_t$的原料”。（**提示：** 本文$\alpha_t,\beta_t$的定义跟原论文不一样。）

反复执行这个拆楼的步骤，我们可以得到：  
\begin{equation}\begin{aligned}  
\boldsymbol{x}_t =&\, \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t \\\  
=&\, \alpha_t \big(\alpha_{t-1} \boldsymbol{x}_{t-2} + \beta_{t-1} \boldsymbol{\varepsilon}_{t-1}\big) + \beta_t \boldsymbol{\varepsilon}_t \\\  
=&\,\cdots\\\  
=&\,(\alpha_t\cdots\alpha_1) \boldsymbol{x}_0 + \underbrace{(\alpha_t\cdots\alpha_2)\beta_1 \boldsymbol{\varepsilon}_1 + (\alpha_t\cdots\alpha_3)\beta_2 \boldsymbol{\varepsilon}_2 + \cdots + \alpha_t\beta_{t-1} \boldsymbol{\varepsilon}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t}_{\text{多个相互独立的正态噪声之和}}  
\end{aligned}\label{eq:expand}\end{equation}  
可能刚才读者就想问为什么叠加的系数要满足$\alpha_t^2 + \beta_t^2 = 1$了，现在我们就可以回答这个问题。首先，式中花括号所指出的部分，正好是多个独立的正态噪声之和，其均值为0，方差则分别为$(\alpha_t\cdots\alpha_2)^2\beta_1^2$、$(\alpha_t\cdots\alpha_3)^2\beta_2^2$、...、$\alpha_t^2\beta_{t-1}^2$、$\beta_t^2$；然后，我们利用一个概率论的知识——正态分布的叠加性，即上述多个独立的正态噪声之和的分布，实际上是均值为0、方差为$(\alpha_t\cdots\alpha_2)^2\beta_1^2 + (\alpha_t\cdots\alpha_3)^2\beta_2^2 + \cdots + \alpha_t^2\beta_{t-1}^2 + \beta_t^2$的正态分布；最后，在$\alpha_t^2 + \beta_t^2 = 1$恒成立之下，我们可以得到式$\eqref{eq:expand}$的各项系数平方和依旧为1，即  
\begin{equation}(\alpha_t\cdots\alpha_1)^2 + (\alpha_t\cdots\alpha_2)^2\beta_1^2 + (\alpha_t\cdots\alpha_3)^2\beta_2^2 + \cdots + \alpha_t^2\beta_{t-1}^2 + \beta_t^2 = 1\end{equation}  
所以实际上相当于有  
\begin{equation}\boldsymbol{x}_t = \underbrace{(\alpha_t\cdots\alpha_1)}_{\text{记为}\bar{\alpha}_t} \boldsymbol{x}_0 + \underbrace{\sqrt{1 - (\alpha_t\cdots\alpha_1)^2}}_{\text{记为}\bar{\beta}_t} \bar{\boldsymbol{\varepsilon}}_t,\quad \bar{\boldsymbol{\varepsilon}}_t\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:skip}\end{equation}  
这就为计算$\boldsymbol{x}_t$提供了极大的便利。另一方面，DDPM会选择适当的$\alpha_t$形式，使得有$\bar{\alpha}_T\approx 0$，这意味着经过$T$步的拆楼后，所剩的楼体几乎可以忽略了，已经全部转化为原材料$\boldsymbol{\varepsilon}$。（**提示：** 本文$\bar{\alpha}_t$的定义跟原论文不一样。）

## 又如何建 #

“拆楼”是$\boldsymbol{x}_{t-1}\to \boldsymbol{x}_t$的过程，这个过程我们得到很多的数据对$(\boldsymbol{x}_{t-1},\boldsymbol{x}_t)$，那么“建楼”自然就是从这些数据对中学习一个$\boldsymbol{x}_t\to \boldsymbol{x}_{t-1}$的模型。设该模型为$\boldsymbol{\mu}(\boldsymbol{x}_t)$，那么容易想到学习方案就是最小化两者的欧氏距离：  
\begin{equation}\left\Vert\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\right\Vert^2\label{eq:loss-0}\end{equation}  
其实这已经非常接近最终的DDPM模型了，接下来让我们将这个过程做得更精细一些。首先“拆楼”的式$\eqref{eq:forward}$可以改写为$\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t \boldsymbol{\varepsilon}_t\right)$，这启发我们或许可以将“建楼”模型$\boldsymbol{\mu}(\boldsymbol{x}_t)$设计成  
\begin{equation}\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)\label{eq:sample}\end{equation}  
的形式，其中$\boldsymbol{\theta}$是训练参数，将其代入到损失函数，得到  
\begin{equation}\left\Vert\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\right\Vert^2 = \frac{\beta_t^2}{\alpha_t^2}\left\Vert \boldsymbol{\varepsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right\Vert^2\end{equation}  
前面的因子$\frac{\beta_t^2}{\alpha_t^2}$代表loss的权重，这个我们可以暂时忽略，最后代入结合式$\eqref{eq:skip}$和$\eqref{eq:forward}$所给出$\boldsymbol{x}_t$的表达式  
\begin{equation}\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t = \alpha_t\left(\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}\right) + \beta_t \boldsymbol{\varepsilon}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t \end{equation}  
得到损失函数的形式为  
\begin{equation}\left\Vert \boldsymbol{\varepsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t, t)\right\Vert^2\label{eq:loss-1}\end{equation}  
可能读者想问为什么要回退一步来给出$\boldsymbol{x}_t$，直接根据式$\eqref{eq:skip}$来给出$\boldsymbol{x}_t$可以吗？答案是不行，因为我们已经事先采样了$\boldsymbol{\varepsilon}_t$，而$\boldsymbol{\varepsilon}_t$跟$\bar{\boldsymbol{\varepsilon}}_t$不是相互独立的，所以给定$\boldsymbol{\varepsilon}_t$的情况下，我们不能完全独立地采样$\bar{\boldsymbol{\varepsilon}}_t$。

## 降低方差 #

原则上来说，损失函数$\eqref{eq:loss-1}$就可以完成DDPM的训练，但它在实践中可能有方差过大的风险，从而导致收敛过慢等问题。要理解这一点并不困难，只需要观察到式$\eqref{eq:loss-1}$实际上包含了4个需要采样的随机变量：

> 1、从所有训练样本中采样一个$\boldsymbol{x}_0$；
> 
> 2、从正态分布$\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$中采样$\bar{\boldsymbol{\varepsilon}}_{t-1}, \boldsymbol{\varepsilon}_t$（两个不同的采样结果）；
> 
> 3、从$1\sim T$中采样一个$t$。

要采样的随机变量越多，就越难对损失函数做准确的估计，反过来说就是每次对损失函数进行估计的波动（方差）过大了。很幸运的是，我们可以通过一个积分技巧来将$\bar{\boldsymbol{\varepsilon}}_{t-1}, \boldsymbol{\varepsilon}_t$合并成单个正态随机变量，从而缓解一下方差大的问题。

这个积分确实有点技巧性，但也不算复杂。由于正态分布的叠加性，我们知道$\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t$实际上相当于单个随机变量$\bar{\beta}_t\boldsymbol{\varepsilon}|\boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，同理$\beta_t \bar{\boldsymbol{\varepsilon}}_{t-1} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\varepsilon}_t$实际上相当于单个随机变量$\bar{\beta}_t\boldsymbol{\omega}|\boldsymbol{\omega}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，并且可以验证$\mathbb{E}[\boldsymbol{\varepsilon}\boldsymbol{\omega}^{\top}]=\boldsymbol{0}$，所以这是两个相互独立的正态随机变量。

接下来，我们反过来将$\boldsymbol{\varepsilon}_t$用$\boldsymbol{\varepsilon},\boldsymbol{\omega}$重新表示出来  
\begin{equation}\boldsymbol{\varepsilon}_t = \frac{(\beta_t \boldsymbol{\varepsilon} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\omega})\bar{\beta}_t}{\beta_t^2 + \alpha_t^2\bar{\beta}_{t-1}^2} = \frac{\beta_t \boldsymbol{\varepsilon} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\omega}}{\bar{\beta}_t}\end{equation}  
代入到式$\eqref{eq:loss-1}$得到  
\begin{equation}\begin{aligned}  
&\,\mathbb{E}_{\bar{\boldsymbol{\varepsilon}}_{t-1}, \boldsymbol{\varepsilon}_t\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})}\left[\left\Vert \boldsymbol{\varepsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t, t)\right\Vert^2\right] \\\  
=&\,\mathbb{E}_{\boldsymbol{\omega}, \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})}\left[\left\Vert \frac{\beta_t \boldsymbol{\varepsilon} - \alpha_t\bar{\beta}_{t-1} \boldsymbol{\omega}}{\bar{\beta}_t} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t)\right\Vert^2\right]  
\end{aligned}\end{equation}  
注意到，现在损失函数关于$\boldsymbol{\omega}$只是二次的，所以我们可以展开然后将它的期望直接算出来，结果是  
\begin{equation}\frac{\beta_t^2}{\bar{\beta}_t^2}\mathbb{E}_{\boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})}\left[\left\Vert\boldsymbol{\varepsilon} - \frac{\bar{\beta}_t}{\beta_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t)\right\Vert^2\right]+\text{常数}\end{equation}  
再次省掉常数和损失函数的权重，我们得到DDPM最终所用的损失函数：  
\begin{equation}\left\Vert\boldsymbol{\varepsilon} - \frac{\bar{\beta}_t}{\beta_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t)\right\Vert^2\end{equation}  
（**提示：** 原论文中的$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$实际上就是本文的$\frac{\bar{\beta}_t}{\beta_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$，所以大家的结果是完全一样的。）

## 递归生成 #

至此，我们算是把DDPM的整个训练流程捋清楚了。内容写了不少，你要说它很容易，那肯定说不上，但真要说非常困难的地方也几乎没有——没有用到传统的能量函数、得分匹配等工具，甚至连变分推断的知识都没有用到，只是借助“拆楼-建楼”的类比和一些基本的概率论知识，就能得到完全一样的结果。所以说，以DDPM为代表的新兴起的生成扩散模型，实际上没有很多读者想象的复杂，它可以说是我们从“拆解-重组”的过程中学习新知识的形象建模。

训练完之后，我们就可以从一个随机噪声$\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$出发执行$T$步式$\eqref{eq:sample}$来进行生成：  
\begin{equation}\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)\end{equation}  
这对应于自回归解码中的Greedy Search。如果要进行Random Sample，那么需要补上噪声项：  
\begin{equation}\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right) + \sigma_t \boldsymbol{z},\quad \boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\end{equation}  
一般来说，我们可以让$\sigma_t=\beta_t$，即正向和反向的方差保持同步。这个采样过程跟传统扩散模型的朗之万采样不一样的地方在于：DDPM的采样每次都从一个随机噪声出发，需要重复迭代$T$步来得到一个样本输出；朗之万采样则是从任意一个点出发，反复迭代无限步，理论上这个迭代无限步的过程中，就把所有数据样本都被生成过了。所以两者除了形式相似外，实质上是两个截然不同的模型。

从这个生成过程中，我们也可以感觉到它其实跟Seq2Seq的解码过程是一样的，都是串联式的自回归生成，所以生成速度是一个瓶颈，DDPM设了$T=1000$，意味着每生成一个图片，需要将$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$反复执行1000次，因此DDPM的一大缺点就是采样速度慢，后面有很多工作都致力于提升DDPM的采样速度。而说到“图片生成 + 自回归模型 + 很慢”，有些读者可能会联想到早期的[PixelRNN](https://papers.cool/arxiv/1601.06759)、[PixelCNN](https://papers.cool/arxiv/1606.05328)等模型，它们将图片生成转换成语言模型任务，所以同样也是递归地进行采样生成以及同样地慢。那么DDPM的这种自回归生成，跟PixelRNN/PixelCNN的自回归生成，又有什么实质区别呢？为什么PixelRNN/PixelCNN没大火起来，反而轮到了DDPM？

了解PixelRNN/PixelCNN的读者都知道，这类生成模型是逐个像素逐个像素地生成图片的，而自回归生成是有序的，这就意味着我们要提前给图片的每个像素排好顺序，最终的生成效果跟这个顺序紧密相关。然而，目前这个顺序只能是人为地凭着经验来设计（这类经验的设计都统称为“Inductive Bias”），暂时找不到理论最优解。换句话说，PixelRNN/PixelCNN的生成效果很受Inductive Bias的影响。但DDPM不一样，它通过“拆楼”的方式重新定义了一个自回归方向，而对于所有的像素来说则都是平权的、无偏的，所以减少了Inductive Bias的影响，从而提升了效果。此外，DDPM生成的迭代步数是固定的$T$，而PixelRNN/PixelCNN则是等于图像分辨率（$\text{宽}\times\text{高}\times{通道数}$），所以DDPM生成高分辨率图像的速度要比PixelRNN/PixelCNN快得多。

## 超参设置 #

这一节我们讨论一下超参的设置问题。

在DDPM中，$T=1000$，可能比很多读者的想象数值要大，那为什么要设置这么大的$T$呢？另一边，对于$\alpha_t$的选择，将原论文的设置翻译到本博客的记号上，大致上是  
\begin{equation}\alpha_t = \sqrt{1 - \frac{0.02t}{T}}\end{equation}  
这是一个单调递减的函数，那为什么要选择单调递减的$\alpha_t$呢？

其实这两个问题有着相近的答案，跟具体的数据背景有关。简单起见，在重构的时候我们用了欧氏距离$\eqref{eq:loss-0}$作为损失函数，而一般我们用DDPM做图片生成，以往做过图片生成的读者都知道，欧氏距离并不是图片真实程度的一个好的度量，VAE用欧氏距离来重构时，往往会得到模糊的结果，除非是输入输出的两张图片非常接近，用欧氏距离才能得到比较清晰的结果，所以选择尽可能大的$T$，正是为了使得输入输出尽可能相近，减少欧氏距离带来的模糊问题。

选择单调递减的$\alpha_t$也有类似考虑。当$t$比较小时，$\boldsymbol{x}_t$还比较接近真实图片，所以我们要缩小$\boldsymbol{x}_{t-1}$与$\boldsymbol{x}_t$的差距，以便更适用欧氏距离$\eqref{eq:loss-0}$，因此要用较大的$\alpha_t$；当$t$比较大时，$\boldsymbol{x}_t$已经比较接近纯噪声了，噪声用欧式距离无妨，所以可以稍微增大$\boldsymbol{x}_{t-1}$与$\boldsymbol{x}_t$的差距，即可以用较小的$\alpha_t$。那么可不可以一直用较大的$\alpha_t$呢？可以是可以，但是要增大$T$。注意在推导$\eqref{eq:skip}$时，我们说过应该有$\bar{\alpha}_T\approx 0$，而我们可以直接估算  
\begin{equation}\log \bar{\alpha}_T = \sum_{t=1}^T \log\alpha_t = \frac{1}{2} \sum_{t=1}^T \log\left(1 - \frac{0.02t}{T}\right) < \frac{1}{2} \sum_{t=1}^T \left(- \frac{0.02t}{T}\right) = -0.005(T+1)\end{equation}  
代入$T=1000$大致是$\bar{\alpha}_T\approx e^{-5}$，这个其实就刚好达到$\approx 0$的标准。所以如果从头到尾都用较大的$\alpha_t$，那么必然要更大的$T$才能使得$\bar{\alpha}_T\approx 0$了。

最后我们留意到，“建楼”模型中的$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t)$中，我们在输入中显式地写出了$t$，这是因为原则上不同的$t$处理的是不同层次的对象，所以应该用不同的重构模型，即应该有$T$个不同的重构模型才对，于是我们共享了所有重构模型的参数，将$t$作为条件传入。按照论文附录的说法，$t$是转换成[《Transformer升级之路：1、Sinusoidal位置编码追根溯源》](/archives/8231)介绍的位置编码后，直接加到残差模块上去的。

## 文章小结 #

本文从“拆楼-建楼”的通俗类比中介绍了最新的生成扩散模型DDPM，在这个视角中，我们可以通过较为“大白话”的描述以及比较少的数学推导，来得到跟原始论文一模一样的结果。总的来说，本文说明了DDPM也可以像GAN一样找到一个形象类比，它既可以不用到VAE中的“变分”，也可以不用到GAN中的“概率散度”、“最优传输”，从这个意义上来看，DDPM甚至算得上比VAE、GAN还要简单。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9119>_

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

苏剑林. (Jun. 13, 2022). 《生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9119>

@online{kexuefm-9119,  
title={生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼},  
author={苏剑林},  
year={2022},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/9119}},  
} 


---

## 详细数学推导

本节我们将对DDPM的数学原理进行极其详细的推导，包括前向扩散过程、后向去噪过程、变分下界证明、损失函数推导等核心内容。

### 1. DDPM的完整数学定义

#### 1.1 前向扩散过程（马尔可夫链）

DDPM的前向扩散过程是一个参数固定的马尔可夫链，从数据分布$q(\boldsymbol{x}_0)$出发，逐步添加高斯噪声，经过$T$步后得到近似的各向同性高斯分布。

**定义1（前向过程）**：给定数据样本$\boldsymbol{x}_0 \sim q(\boldsymbol{x}_0)$，前向扩散过程定义为：

\begin{equation}
q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) = \prod_{t=1}^{T} q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})
\end{equation}

其中每一步的转移概率为：

\begin{equation}
q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_t; \alpha_t\boldsymbol{x}_{t-1}, \beta_t^2\boldsymbol{I})
\end{equation}

这里$\alpha_t, \beta_t > 0$且满足约束$\alpha_t^2 + \beta_t^2 = 1$。

**定理1（任意步前向扩散）**：对于任意时刻$t$，我们可以直接从$\boldsymbol{x}_0$采样得到$\boldsymbol{x}_t$：

\begin{equation}
q(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \bar{\alpha}_t\boldsymbol{x}_0, \bar{\beta}_t^2\boldsymbol{I})
\end{equation}

其中$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$，$\bar{\beta}_t = \sqrt{1-\bar{\alpha}_t^2}$。

**证明**：我们通过数学归纳法证明此定理。

*基础步骤*（$t=1$）：根据定义，$q(\boldsymbol{x}_1|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_1; \alpha_1\boldsymbol{x}_0, \beta_1^2\boldsymbol{I})$。
由于$\alpha_1^2 + \beta_1^2 = 1$，所以$\bar{\alpha}_1 = \alpha_1$，$\bar{\beta}_1 = \beta_1$，命题成立。

*归纳步骤*：假设对$t-1$命题成立，即
\begin{equation}
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \bar{\alpha}_{t-1}\boldsymbol{x}_0, \bar{\beta}_{t-1}^2\boldsymbol{I})
\end{equation}

根据重参数化技巧，我们可以写成：
\begin{equation}
\boldsymbol{x}_{t-1} = \bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\boldsymbol{\varepsilon}_{t-1}, \quad \boldsymbol{\varepsilon}_{t-1} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
\end{equation}

再应用一步扩散：
\begin{equation}
\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
\end{equation}

代入得：
\begin{equation}
\begin{aligned}
\boldsymbol{x}_t &= \alpha_t(\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\boldsymbol{\varepsilon}_{t-1}) + \beta_t\boldsymbol{\varepsilon}_t \\
&= \alpha_t\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \alpha_t\bar{\beta}_{t-1}\boldsymbol{\varepsilon}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t
\end{aligned}
\end{equation}

由于$\boldsymbol{\varepsilon}_{t-1}$和$\boldsymbol{\varepsilon}_t$相互独立，根据**高斯分布的叠加性**：两个独立高斯随机变量$\mathcal{N}(0, \sigma_1^2)$和$\mathcal{N}(0, \sigma_2^2)$的线性组合$a_1\boldsymbol{\varepsilon}_1 + a_2\boldsymbol{\varepsilon}_2$服从$\mathcal{N}(0, a_1^2\sigma_1^2 + a_2^2\sigma_2^2)$。

因此，$\alpha_t\bar{\beta}_{t-1}\boldsymbol{\varepsilon}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t$的方差为：
\begin{equation}
\alpha_t^2\bar{\beta}_{t-1}^2 + \beta_t^2 = \alpha_t^2(1-\bar{\alpha}_{t-1}^2) + \beta_t^2
\end{equation}

利用$\alpha_t^2 + \beta_t^2 = 1$，我们有$\beta_t^2 = 1 - \alpha_t^2$，代入得：
\begin{equation}
\begin{aligned}
\alpha_t^2(1-\bar{\alpha}_{t-1}^2) + \beta_t^2 &= \alpha_t^2 - \alpha_t^2\bar{\alpha}_{t-1}^2 + 1 - \alpha_t^2 \\
&= 1 - \alpha_t^2\bar{\alpha}_{t-1}^2 \\
&= 1 - \bar{\alpha}_t^2 \\
&= \bar{\beta}_t^2
\end{aligned}
\end{equation}

因此：
\begin{equation}
\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
\end{equation}

即$q(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \bar{\alpha}_t\boldsymbol{x}_0, \bar{\beta}_t^2\boldsymbol{I})$。证毕。

#### 1.2 后向去噪过程（反向马尔可夫链）

**定义2（后向过程）**：后向去噪过程定义为参数化的马尔可夫链：

\begin{equation}
p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T}) = p(\boldsymbol{x}_T)\prod_{t=1}^{T} p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)
\end{equation}

其中先验$p(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{x}_T; \boldsymbol{0}, \boldsymbol{I})$，反向转移概率建模为：

\begin{equation}
p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t), \sigma_t^2\boldsymbol{I})
\end{equation}

其中$\boldsymbol{\mu}_{\boldsymbol{\theta}}$是需要学习的均值函数，$\sigma_t^2$是预设的方差。

### 2. 后向条件分布的推导

为了理解如何设计后向过程，我们需要推导真实的后向条件分布$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$。

**定理2（后向条件分布）**：给定$\boldsymbol{x}_t$和$\boldsymbol{x}_0$，后向条件分布为：

\begin{equation}
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0), \tilde{\beta}_t^2\boldsymbol{I})
\end{equation}

其中均值和方差为：

\begin{equation}
\tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\bar{\beta}_t^2}\boldsymbol{x}_0
\end{equation}

\begin{equation}
\tilde{\beta}_t^2 = \frac{\bar{\beta}_{t-1}^2\beta_t^2}{\bar{\beta}_t^2}
\end{equation}

**证明**：根据贝叶斯定理：

\begin{equation}
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}, \boldsymbol{x}_0)q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_t|\boldsymbol{x}_0)}
\end{equation}

由于马尔可夫性质，$q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$，因此：

\begin{equation}
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) \propto q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)
\end{equation}

我们知道：
\begin{align}
q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) &= \mathcal{N}(\boldsymbol{x}_t; \alpha_t\boldsymbol{x}_{t-1}, \beta_t^2\boldsymbol{I}) \\
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) &= \mathcal{N}(\boldsymbol{x}_{t-1}; \bar{\alpha}_{t-1}\boldsymbol{x}_0, \bar{\beta}_{t-1}^2\boldsymbol{I})
\end{align}

将高斯分布的密度函数代入：

\begin{equation}
\begin{aligned}
&q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) \\
&\propto \exp\left(-\frac{1}{2}\left[\frac{\|\boldsymbol{x}_t - \alpha_t\boldsymbol{x}_{t-1}\|^2}{\beta_t^2} + \frac{\|\boldsymbol{x}_{t-1} - \bar{\alpha}_{t-1}\boldsymbol{x}_0\|^2}{\bar{\beta}_{t-1}^2}\right]\right)
\end{aligned}
\end{equation}

展开二次项：

\begin{equation}
\begin{aligned}
&\frac{\|\boldsymbol{x}_t - \alpha_t\boldsymbol{x}_{t-1}\|^2}{\beta_t^2} + \frac{\|\boldsymbol{x}_{t-1} - \bar{\alpha}_{t-1}\boldsymbol{x}_0\|^2}{\bar{\beta}_{t-1}^2} \\
&= \frac{1}{\beta_t^2}(\boldsymbol{x}_t^{\top}\boldsymbol{x}_t - 2\alpha_t\boldsymbol{x}_t^{\top}\boldsymbol{x}_{t-1} + \alpha_t^2\boldsymbol{x}_{t-1}^{\top}\boldsymbol{x}_{t-1}) \\
&\quad + \frac{1}{\bar{\beta}_{t-1}^2}(\boldsymbol{x}_{t-1}^{\top}\boldsymbol{x}_{t-1} - 2\bar{\alpha}_{t-1}\boldsymbol{x}_{t-1}^{\top}\boldsymbol{x}_0 + \bar{\alpha}_{t-1}^2\boldsymbol{x}_0^{\top}\boldsymbol{x}_0)
\end{aligned}
\end{equation}

收集关于$\boldsymbol{x}_{t-1}$的二次项：

\begin{equation}
\left(\frac{\alpha_t^2}{\beta_t^2} + \frac{1}{\bar{\beta}_{t-1}^2}\right)\boldsymbol{x}_{t-1}^{\top}\boldsymbol{x}_{t-1}
\end{equation}

收集关于$\boldsymbol{x}_{t-1}$的一次项：

\begin{equation}
-2\left(\frac{\alpha_t}{\beta_t^2}\boldsymbol{x}_t^{\top} + \frac{\bar{\alpha}_{t-1}}{\bar{\beta}_{t-1}^2}\boldsymbol{x}_0^{\top}\right)\boldsymbol{x}_{t-1}
\end{equation}

后验分布$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$仍然是高斯分布，其精度（方差的倒数）为：

\begin{equation}
\tilde{\beta}_t^{-2} = \frac{\alpha_t^2}{\beta_t^2} + \frac{1}{\bar{\beta}_{t-1}^2} = \frac{\alpha_t^2\bar{\beta}_{t-1}^2 + \beta_t^2}{\beta_t^2\bar{\beta}_{t-1}^2}
\end{equation}

利用$\alpha_t^2\bar{\beta}_{t-1}^2 + \beta_t^2 = \alpha_t^2(1-\bar{\alpha}_{t-1}^2) + (1-\alpha_t^2) = 1 - \alpha_t^2\bar{\alpha}_{t-1}^2 = \bar{\beta}_t^2$，得：

\begin{equation}
\tilde{\beta}_t^2 = \frac{\beta_t^2\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}
\end{equation}

均值为：

\begin{equation}
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t &= \tilde{\beta}_t^2\left(\frac{\alpha_t}{\beta_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}}{\bar{\beta}_{t-1}^2}\boldsymbol{x}_0\right) \\
&= \frac{\beta_t^2\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\left(\frac{\alpha_t}{\beta_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}}{\bar{\beta}_{t-1}^2}\boldsymbol{x}_0\right) \\
&= \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\bar{\beta}_t^2}\boldsymbol{x}_0
\end{aligned}
\end{equation}

证毕。

### 3. 变分下界（ELBO）的完整推导

DDPM的训练目标是最大化对数似然$\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_0)$。我们通过变分推断推导变分下界。

**定理3（变分下界）**：对数似然的变分下界为：

\begin{equation}
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_0) \geq \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\left[\log\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\right] =: \mathcal{L}
\end{equation}

**证明**：使用Jensen不等式。对于任意分布$q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)$：

\begin{equation}
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_0) &= \log\int p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T})d\boldsymbol{x}_{1:T} \\
&= \log\int q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}d\boldsymbol{x}_{1:T} \\
&= \log\mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\left[\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\right] \\
&\geq \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\left[\log\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\right]
\end{aligned}
\end{equation}

最后一步使用了Jensen不等式（$\log$是凹函数）。证毕。

**定理4（ELBO的KL散度分解）**：变分下界可以分解为：

\begin{equation}
\begin{aligned}
\mathcal{L} = &\mathbb{E}_q[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_0|\boldsymbol{x}_1)] \\
&- D_{KL}(q(\boldsymbol{x}_T|\boldsymbol{x}_0) \| p(\boldsymbol{x}_T)) \\
&- \sum_{t=2}^{T}\mathbb{E}_{q(\boldsymbol{x}_t|\boldsymbol{x}_0)}[D_{KL}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t))]
\end{aligned}
\end{equation}

**证明**：展开ELBO：

\begin{equation}
\begin{aligned}
\mathcal{L} &= \mathbb{E}_q\left[\log\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}\right] \\
&= \mathbb{E}_q\left[\log\frac{p(\boldsymbol{x}_T)\prod_{t=1}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{\prod_{t=1}^{T}q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})}\right] \\
&= \mathbb{E}_q\left[\log p(\boldsymbol{x}_T) + \sum_{t=1}^{T}\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) - \sum_{t=1}^{T}\log q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})\right]
\end{aligned}
\end{equation}

利用链式法则$q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) = q(\boldsymbol{x}_T|\boldsymbol{x}_0)\prod_{t=2}^{T}q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$（通过贝叶斯定理验证），我们可以改写：

\begin{equation}
\begin{aligned}
\mathcal{L} &= \mathbb{E}_q\left[\log\frac{p(\boldsymbol{x}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}_0|\boldsymbol{x}_1)}{q(\boldsymbol{x}_T|\boldsymbol{x}_0)} + \sum_{t=2}^{T}\log\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)}\right] \\
&= \mathbb{E}_q[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_0|\boldsymbol{x}_1)] - D_{KL}(q(\boldsymbol{x}_T|\boldsymbol{x}_0) \| p(\boldsymbol{x}_T)) \\
&\quad - \sum_{t=2}^{T}\mathbb{E}_{q(\boldsymbol{x}_t|\boldsymbol{x}_0)}[D_{KL}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t))]
\end{aligned}
\end{equation}

证毕。

### 4. 损失函数的详细推导

#### 4.1 基于噪声预测的参数化

根据定理2，真实的后向均值为：

\begin{equation}
\tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\bar{\beta}_t^2}\boldsymbol{x}_0
\end{equation}

从$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}$，我们可以解出：

\begin{equation}
\boldsymbol{x}_0 = \frac{\boldsymbol{x}_t - \bar{\beta}_t\boldsymbol{\varepsilon}}{\bar{\alpha}_t}
\end{equation}

代入均值表达式：

\begin{equation}
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t &= \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\bar{\beta}_t^2} \cdot \frac{\boldsymbol{x}_t - \bar{\beta}_t\boldsymbol{\varepsilon}}{\bar{\alpha}_t} \\
&= \frac{1}{\bar{\beta}_t^2}\left[\alpha_t\bar{\beta}_{t-1}^2\boldsymbol{x}_t + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\bar{\alpha}_t}\boldsymbol{x}_t - \frac{\beta_t^2\bar{\alpha}_{t-1}\bar{\beta}_t}{\bar{\alpha}_t}\boldsymbol{\varepsilon}\right]
\end{aligned}
\end{equation}

利用$\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1}$和$\bar{\beta}_t^2 = 1 - \bar{\alpha}_t^2$，化简系数：

\begin{equation}
\begin{aligned}
\alpha_t\bar{\beta}_{t-1}^2 + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\bar{\alpha}_t} &= \alpha_t(1-\bar{\alpha}_{t-1}^2) + \frac{\beta_t^2\bar{\alpha}_{t-1}}{\alpha_t\bar{\alpha}_{t-1}} \\
&= \alpha_t - \alpha_t\bar{\alpha}_{t-1}^2 + \frac{\beta_t^2}{\alpha_t} \\
&= \alpha_t - \alpha_t\bar{\alpha}_{t-1}^2 + \frac{1-\alpha_t^2}{\alpha_t} \\
&= \frac{\alpha_t^2 - \alpha_t^2\bar{\alpha}_{t-1}^2 + 1 - \alpha_t^2}{\alpha_t} \\
&= \frac{1 - \bar{\alpha}_t^2}{\alpha_t} = \frac{\bar{\beta}_t^2}{\alpha_t}
\end{aligned}
\end{equation}

因此：

\begin{equation}
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\alpha_t}\left[\boldsymbol{x}_t - \frac{\beta_t^2\bar{\alpha}_{t-1}\bar{\beta}_t}{\bar{\alpha}_t\bar{\beta}_t^2}\boldsymbol{\varepsilon}\right] = \frac{1}{\alpha_t}\left[\boldsymbol{x}_t - \frac{\beta_t^2}{\bar{\beta}_t\alpha_t\bar{\alpha}_{t-1}}\boldsymbol{\varepsilon}\right]
\end{equation}

利用$\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1}$，进一步化简：

\begin{equation}
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t^2}{\bar{\beta}_t\bar{\alpha}_t}\boldsymbol{\varepsilon}\right)
\end{equation}

注意本文的记号中$\beta_t^2 = 1 - \alpha_t^2$，而原始DDPM论文中定义的$\beta_t$对应本文的$\beta_t^2$。因此在原始论文记号下：

\begin{equation}
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\varepsilon}\right)
\end{equation}

这启发我们将模型参数化为：

\begin{equation}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)
\end{equation}

其中$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$是噪声预测网络。

#### 4.2 L_t损失的推导

对于$t \geq 2$，KL散度项为：

\begin{equation}
\begin{aligned}
L_{t-1} &= \mathbb{E}_q[D_{KL}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t))] \\
&= \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_{\boldsymbol{\theta}}\|^2\right] + \text{const}
\end{aligned}
\end{equation}

这里使用了两个高斯分布KL散度的公式：

\begin{equation}
D_{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1) \| \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^{\top}\Sigma_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)\right]
\end{equation}

当方差相同时，简化为：

\begin{equation}
D_{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \sigma^2\boldsymbol{I}) \| \mathcal{N}(\boldsymbol{\mu}_2, \sigma^2\boldsymbol{I})) = \frac{1}{2\sigma^2}\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2
\end{equation}

代入均值的表达式：

\begin{equation}
\begin{aligned}
L_{t-1} &= \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\varepsilon}}\left[\frac{1}{2\sigma_t^2}\left\|\frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\varepsilon}\right) - \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right)\right\|^2\right] \\
&= \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\varepsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t^2(1-\bar{\alpha}_t)}\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\|^2\right] \\
&= \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\varepsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t^2(1-\bar{\alpha}_t)}\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon},t)\|^2\right]
\end{aligned}
\end{equation}

#### 4.3 简化损失函数（L_simple）

DDPM发现，忽略权重系数$\frac{\beta_t^2}{2\sigma_t^2\alpha_t^2(1-\bar{\alpha}_t)}$，使用简化的损失函数：

\begin{equation}
L_{\text{simple}} = \mathbb{E}_{t\sim\mathcal{U}(1,T), \boldsymbol{x}_0\sim q(\boldsymbol{x}_0), \boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\left[\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0+\sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon},t)\|^2\right]
\end{equation}

实际上效果更好。这可以理解为：

1. **重参数化技巧**：将采样$\boldsymbol{x}_t \sim q(\boldsymbol{x}_t|\boldsymbol{x}_0)$转化为确定性变换$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\varepsilon}$

2. **去噪自编码器视角**：模型学习预测添加到干净数据$\boldsymbol{x}_0$上的噪声$\boldsymbol{\varepsilon}$

3. **多尺度去噪**：不同的$t$对应不同噪声水平，模型学习在所有噪声水平下预测噪声

### 5. 与VAE的深层联系

DDPM实际上是一个具有特殊结构的层次化VAE（Hierarchical VAE）。

**定理5（DDPM作为VAE）**：DDPM可以视为满足以下性质的VAE：

1. **潜变量维度**：潜变量$\boldsymbol{x}_{1:T}$与数据$\boldsymbol{x}_0$维度相同
2. **编码器结构**：编码器$q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)$是固定的（不可学习）高斯转移
3. **潜变量层次**：$T$层潜变量形成马尔可夫链
4. **解码器结构**：解码器$p_{\boldsymbol{\theta}}(\boldsymbol{x}_{0:T-1}|\boldsymbol{x}_{1:T})$反向遍历马尔可夫链

**与标准VAE的对比**：

| 特性 | 标准VAE | DDPM |
|------|---------|------|
| 潜变量维度 | $\dim(\boldsymbol{z}) \ll \dim(\boldsymbol{x})$ | $\dim(\boldsymbol{x}_t) = \dim(\boldsymbol{x}_0)$ |
| 编码器 | 可学习的$q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ | 固定的$q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$ |
| 潜变量结构 | 单层 | $T$层马尔可夫链 |
| ELBO | $\mathbb{E}_q[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}|\boldsymbol{z})] - D_{KL}(q_{\phi}(\boldsymbol{z}|\boldsymbol{x})\|p(\boldsymbol{z}))$ | 式(39)的分解形式 |

**优势分析**：

1. **避免后验坍缩**：固定编码器防止了VAE训练中常见的后验坍缩问题
2. **渐进式建模**：$T$步渐进生成比一步生成更容易学习
3. **表达能力**：高维潜空间保留更多信息

### 6. 采样算法的详细推导

#### 6.1 确定性采样（DDPM采样）

**算法1（DDPM采样）**：

```
输入：训练好的噪声预测模型 ε_θ(x_t, t)
输出：生成样本 x_0

1. 采样 x_T ~ N(0, I)
2. for t = T, T-1, ..., 1 do
3.     采样 z ~ N(0, I) if t > 1, else z = 0
4.     计算 x_{t-1} = (1/α_t)[x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)] + σ_t z
5. end for
6. return x_0
```

**噪声方差的选择**：

根据定理2，理论上应该使用$\sigma_t = \tilde{\beta}_t = \frac{\beta_t\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{1-\bar{\alpha}_t}}$。

DDPM也尝试了固定方差$\sigma_t^2 = \beta_t^2$（对应本文记号），发现两者效果相近。

#### 6.2 概率解释

每一步采样可以理解为：

\begin{equation}
\boldsymbol{x}_{t-1}|\boldsymbol{x}_t \sim \mathcal{N}\left(\frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right), \sigma_t^2\boldsymbol{I}\right)
\end{equation}

**均值项**：$\frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right)$

- $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$：预测的噪声
- $\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}$：噪声的缩放因子
- $\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$：去噪后的估计
- $\frac{1}{\alpha_t}$：反向缩放

**方差项**：$\sigma_t^2\boldsymbol{I}$

- 控制采样的随机性
- $\sigma_t = 0$：确定性去噪（类似Greedy Decoding）
- $\sigma_t > 0$：随机采样（类似Random Sampling）

### 7. Score-Based模型的等价性

DDPM与基于分数的生成模型（Score-Based Generative Models）有深刻联系。

**定义3（Score函数）**：数据分布$q(\boldsymbol{x})$的score函数定义为：

\begin{equation}
\nabla_{\boldsymbol{x}}\log q(\boldsymbol{x})
\end{equation}

**定理6（DDPM的Score解释）**：DDPM的噪声预测网络$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$与score函数的关系为：

\begin{equation}
\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) = -\sqrt{1-\bar{\alpha}_t} \nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t)
\end{equation}

**证明**：对于$q(\boldsymbol{x}_t) = \int q(\boldsymbol{x}_t|\boldsymbol{x}_0)q(\boldsymbol{x}_0)d\boldsymbol{x}_0$，有：

\begin{equation}
\begin{aligned}
\nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t) &= \frac{\nabla_{\boldsymbol{x}_t}q(\boldsymbol{x}_t)}{q(\boldsymbol{x}_t)} \\
&= \frac{\int \nabla_{\boldsymbol{x}_t}q(\boldsymbol{x}_t|\boldsymbol{x}_0)q(\boldsymbol{x}_0)d\boldsymbol{x}_0}{\int q(\boldsymbol{x}_t|\boldsymbol{x}_0)q(\boldsymbol{x}_0)d\boldsymbol{x}_0} \\
&= \mathbb{E}_{q(\boldsymbol{x}_0|\boldsymbol{x}_t)}[\nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t|\boldsymbol{x}_0)]
\end{aligned}
\end{equation}

由于$q(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\bar{\alpha}_t\boldsymbol{x}_0, (1-\bar{\alpha}_t)\boldsymbol{I})$，有：

\begin{equation}
\nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t|\boldsymbol{x}_0) = -\frac{\boldsymbol{x}_t - \bar{\alpha}_t\boldsymbol{x}_0}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\varepsilon}}{\sqrt{1-\bar{\alpha}_t}}
\end{equation}

因此：

\begin{equation}
\nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t) = -\frac{1}{\sqrt{1-\bar{\alpha}_t}}\mathbb{E}_{q(\boldsymbol{x}_0|\boldsymbol{x}_t)}[\boldsymbol{\varepsilon}]
\end{equation}

DDPM训练的目标正是让$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) \approx \mathbb{E}_{q(\boldsymbol{x}_0|\boldsymbol{x}_t)}[\boldsymbol{\varepsilon}]$，因此：

\begin{equation}
\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) = -\sqrt{1-\bar{\alpha}_t}\nabla_{\boldsymbol{x}_t}\log q(\boldsymbol{x}_t)
\end{equation}

证毕。

这说明DDPM本质上是在学习每个噪声水平下的score函数，采样过程等价于**退火的朗之万动力学**（Annealed Langevin Dynamics）。

### 8. 理论收敛性分析

#### 8.1 离散时间分析

**定理7（重构误差界）**：假设噪声预测误差有界，即$\mathbb{E}[\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\|^2] \leq \epsilon^2$，则生成分布与真实分布的KL散度满足：

\begin{equation}
D_{KL}(q(\boldsymbol{x}_0) \| p_{\boldsymbol{\theta}}(\boldsymbol{x}_0)) \leq C_1 T\epsilon^2 + C_2/T
\end{equation}

其中$C_1, C_2$是与数据分布相关的常数。

**直观解释**：

- 第一项$C_1 T\epsilon^2$：累积的预测误差，随步数$T$线性增长
- 第二项$C_2/T$：离散化误差，$T$越大离散化越精细

**最优步数**：平衡两项误差，最优$T \sim \epsilon^{-1}$，此时误差为$O(\epsilon)$。

#### 8.2 连续时间极限

当$T \to \infty$，DDPM收敛到**随机微分方程**（SDE）：

**前向SDE**：
\begin{equation}
d\boldsymbol{x} = -\frac{1}{2}\beta(t)\boldsymbol{x}dt + \sqrt{\beta(t)}d\boldsymbol{w}
\end{equation}

**后向SDE**（[Anderson 1982](https://doi.org/10.1016/0304-4149(82)90051-5)）：
\begin{equation}
d\boldsymbol{x} = \left[-\frac{1}{2}\beta(t)\boldsymbol{x} - \beta(t)\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x})\right]dt + \sqrt{\beta(t)}d\bar{\boldsymbol{w}}
\end{equation}

其中$\boldsymbol{w}, \bar{\boldsymbol{w}}$是标准布朗运动，$\beta(t)$是噪声调度函数。

**概率流ODE**（Probability Flow ODE）：

后向SDE对应的确定性ODE为：

\begin{equation}
d\boldsymbol{x} = \left[-\frac{1}{2}\beta(t)\boldsymbol{x} - \frac{1}{2}\beta(t)\nabla_{\boldsymbol{x}}\log q_t(\boldsymbol{x})\right]dt
\end{equation}

这个ODE与SDE有相同的边缘分布$q_t(\boldsymbol{x})$，但轨迹是确定性的，这为后续的DDIM等快速采样算法提供了理论基础。

### 9. 实践中的技巧与改进

#### 9.1 噪声调度的选择

**线性调度**（DDPM原始）：
\begin{equation}
\beta_t^2 = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})
\end{equation}

**余弦调度**（Improved DDPM）：
\begin{equation}
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
\end{equation}

余弦调度在接近$t=0$和$t=T$时变化更平缓，避免了过快的信号损失。

#### 9.2 方差的学习

固定方差$\sigma_t^2 = \beta_t^2$或$\sigma_t^2 = \tilde{\beta}_t^2$效果已经不错，但也可以让模型学习方差：

\begin{equation}
\sigma_t^2 = \exp(v\log\beta_t^2 + (1-v)\log\tilde{\beta}_t^2)
\end{equation}

其中$v = \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \in [0,1]$是模型输出的插值系数。

#### 9.3 重要性采样

均匀采样时间步$t \sim \mathcal{U}(1,T)$不是最优的。根据损失权重，可以设计重要性采样：

\begin{equation}
p(t) \propto \sqrt{\mathbb{E}[\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\|^2]}
\end{equation}

在训练早期，高噪声水平（大$t$）的误差更大，应该更频繁地采样。

### 10. 总结与展望

本节详细推导了DDPM的核心数学原理，包括：

1. **前向扩散过程**：证明了任意步扩散的闭式解（定理1）
2. **后向条件分布**：推导了精确的后验分布形式（定理2）
3. **变分下界**：完整证明了ELBO及其KL散度分解（定理3-4）
4. **损失函数**：从变分推断到简化损失函数的推导
5. **与VAE的联系**：揭示了DDPM作为特殊层次VAE的本质（定理5）
6. **Score-Based等价性**：建立了与score匹配的联系（定理6）
7. **理论保证**：分析了收敛性和连续时间极限（定理7）

**关键洞察**：

- DDPM通过$T$步渐进去噪，将困难的一步生成问题分解为$T$个简单的去噪步骤
- 固定的编码器避免了VAE的后验坍缩问题
- 噪声预测等价于score函数估计，连接了基于能量的模型
- 连续时间极限揭示了与SDE的深刻联系

**未解决的问题**：

1. 如何进一步加速采样（DDIM、DPM-Solver等后续工作）
2. 最优的噪声调度设计
3. 高分辨率图像的计算效率
4. 理论收敛速率的紧界

这些数学基础为理解后续的改进算法（如DDIM、Score SDE、EDM等）铺平了道路。

