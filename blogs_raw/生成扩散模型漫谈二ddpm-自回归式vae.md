---
title: 生成扩散模型漫谈（二）：DDPM = 自回归式VAE
slug: 生成扩散模型漫谈二ddpm-自回归式vae
date: 2022-07-06
tags: vae, 生成模型, DDPM, 扩散, 生成模型
status: pending
---

# 生成扩散模型漫谈（二）：DDPM = 自回归式VAE

**原文链接**: [https://spaces.ac.cn/archives/9152](https://spaces.ac.cn/archives/9152)

**发布日期**: 

---

在文章[《生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼》](/archives/9119)中，我们为生成扩散模型DDPM构建了“拆楼-建楼”的通俗类比，并且借助该类比完整地推导了生成扩散模型DDPM的理论形式。在该文章中，我们还指出DDPM本质上已经不是传统的扩散模型了，它更多的是一个变分自编码器VAE，实际上DDPM的原论文中也是将它按照VAE的思路进行推导的。

所以，本文就从VAE的角度来重新介绍一版DDPM，同时分享一下自己的Keras实现代码和实践经验。

## 多步突破 #

在传统的VAE中，编码过程和生成过程都是一步到位的：  
\begin{equation}\text{编码:}\,\,x\to z\,,\quad \text{生成:}\,\,z\to x\end{equation}  
这样做就只涉及到三个分布：编码分布$p(z|x)$、生成分布$q(x|z)$以及先验分布$q(z)$，它的好处是形式比较简单，$x$与$z$之间的映射关系也比较确定，因此可以同时得到编码模型和生成模型，实现隐变量编辑等需求；但是它的缺点也很明显，因为我们建模概率分布的能力有限，这三个分布都只能建模为正态分布，这限制了模型的表达能力，最终通常得到偏模糊的生成结果。

为了突破这个限制，DDPM将编码过程和生成过程分解为$T$步：  
\begin{equation}\begin{aligned}&\text{编码:}\,\,\boldsymbol{x} = \boldsymbol{x}_0 \to \boldsymbol{x}_1 \to \boldsymbol{x}_2 \to \cdots \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_T = \boldsymbol{z} \\\  
&\text{生成:}\,\,\boldsymbol{z} = \boldsymbol{x}_T \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_{T-2} \to \cdots \to \boldsymbol{x}_1 \to \boldsymbol{x}_0 = \boldsymbol{x}  
\end{aligned}\label{eq:factor}\end{equation}  
这样一来，每一个$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$和$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$仅仅负责建模一个微小变化，它们依然建模为正态分布。可能读着就想问了：那既然同样是正态分布，为什么分解为多步会比单步要好？这是因为对于微小变化来说，可以用正态分布足够近似地建模，类似于曲线在小范围内可以用直线近似，多步分解就有点像用分段线性函数拟合复杂曲线，因此理论上可以突破传统单步VAE的拟合能力限制。

## 联合散度 #

所以，现在的计划就是通过递归式分解$\eqref{eq:factor}$来增强传统VAE的能力，每一步编码过程被建模成$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$，每一步生成过程则被建模成$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$，相应的联合分布就是：  
\begin{equation}\begin{aligned}&p(\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_T) = p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_2|\boldsymbol{x}_1) p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \\\  
&q(\boldsymbol{x}_0, \boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_T) = q(\boldsymbol{x}_0|\boldsymbol{x}_1)\cdots q(\boldsymbol{x}_{T-2}|\boldsymbol{x}_{T-1}) q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_T) q(\boldsymbol{x}_T)  
\end{aligned}\end{equation}  
别忘了$\boldsymbol{x}_0$代表真实样本，所以$\tilde{p}(\boldsymbol{x}_0)$就是数据分布；而$\boldsymbol{x}_T$代表着最终的编码，所以$q(\boldsymbol{x}_T)$就是先验分布；剩下的$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$、$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$就代表着编码、生成的一小步。（**提示：** 经过考虑，这里还是沿用本网站介绍VAE一直用的记号习惯，即“编码分布用$p$、生成分布用$q$”，所以这里的$p$、$q$含义跟DDPM论文是刚好相反的，望读者知悉。）

在[《变分自编码器（二）：从贝叶斯观点出发》](/archives/5343)中笔者就提出，理解VAE的最简洁的理论途径，就是将其理解为在最小化联合分布的KL散度，对于DDPM也是如此，上面我们已经写出了两个联合分布，所以DDPM的目的就是最小化  
\begin{equation}KL(p\Vert q) = \int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log \frac{p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0)}{q(\boldsymbol{x}_0|\boldsymbol{x}_1)\cdots q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_T) q(\boldsymbol{x}_T)} d\boldsymbol{x}_0 d\boldsymbol{x}_1\cdots d\boldsymbol{x}_T\label{eq:kl}\end{equation}  
这就是DDPM的优化目标了。到目前为止的结果，都跟DDPM原论文的结果一样的（只是记号略有不同），也跟更原始的论文[《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》](https://papers.cool/arxiv/1503.03585)一致。接下来，我们就要将$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$、$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$具体形式定下来，然后简化DDPM的优化目标$\eqref{eq:kl}$。

## 分而治之 #

首先我们要知道，DDPM只是想做一个生成模型，所以它只是将每一步的编码建立为极简单的正态分布：$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})=\mathcal{N}(\boldsymbol{x}_t;\alpha_t \boldsymbol{x}_{t-1}, \beta_t^2 \boldsymbol{I})$，其主要的特点是均值向量仅仅由输入$\boldsymbol{x}_{t-1}$乘以一个标量$\alpha_t$得到，相比之下传统VAE的均值方差都是用神经网络学习出来的，因此DDPM是放弃了模型的编码能力，最终只得到一个纯粹的生成模型；至于$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$，则被建模成均值向量可学习的正态分布$\mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}(\boldsymbol{x}_t), \sigma_t^2 \boldsymbol{I})$。其中$\alpha_t,\beta_t,\sigma_t$都不是可训练参数，而是事先设定好的值（怎么设置我们稍后讨论），所以整个模型拥有可训练参数的就只有$\boldsymbol{\mu}(\boldsymbol{x}_t)$。（**提示：** 本文$\alpha_t,\beta_t$的定义跟原论文不一样。）

由于目前分布$p$不含任何的可训练参数，因此目标$\eqref{eq:kl}$中关于$p$的积分就只是贡献一个可以忽略的常数，所以目标$\eqref{eq:kl}$等价于  
\begin{equation}\begin{aligned}&\,-\int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_0|\boldsymbol{x}_1)\cdots q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_T) q(\boldsymbol{x}_T) d\boldsymbol{x}_0 d\boldsymbol{x}_1\cdots d\boldsymbol{x}_T \\\  
=&\,-\int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \left[\log q(\boldsymbol{x}_T) + \sum_{t=1}^T\log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)\right] d\boldsymbol{x}_0 d\boldsymbol{x}_1\cdots d\boldsymbol{x}_T  
\end{aligned}\end{equation}  
由于先验分布$q(\boldsymbol{x}_T)$一般都取标准正态分布，也是没有参数的，所以这一项也只是贡献一个常数。因此需要计算的就是每一项  
\begin{equation}\begin{aligned}&\,-\int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_1\cdots d\boldsymbol{x}_T\\\  
=&\,-\int p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_1\cdots d\boldsymbol{x}_t\\\  
=&\,-\int p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_{t-1}d\boldsymbol{x}_t  
\end{aligned}\end{equation}  
其中第一个等号是因为$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$至多依赖到$\boldsymbol{x}_t$，因此$t+1$到$T$的分布可以直接积分为1；第二个等号则是因为$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$也不依赖于$\boldsymbol{x}_1,\cdots,\boldsymbol{x}_{t-2}$，所以关于它们的积分我们也可以事先算出，结果为$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)=\mathcal{N}(\boldsymbol{x}_{t-1};\bar{\alpha}_{t-1} \boldsymbol{x}_0, \bar{\beta}_{t-1}^2 \boldsymbol{I})$，该结果可以参考下一节的式$\eqref{eq:x0-xt}$。

## 场景再现 #

接下来的过程就跟上一篇文章的“[又如何建](/archives/9119#%E5%8F%88%E5%A6%82%E4%BD%95%E5%BB%BA)”一节基本上是一样的了：

> 1、除去优化无关的常数，$-\log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$这一项所贡献的就是$\frac{1}{2\sigma_t^2}\left\Vert\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\right\Vert^2$；
> 
> 2、$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)$意味着$\boldsymbol{x}_{t-1} = \bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}$，$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$又意味着$\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t$，其中$\bar{\boldsymbol{\varepsilon}}_{t-1},\boldsymbol{\varepsilon}_t\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$；
> 
> 3、由$\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t \boldsymbol{\varepsilon}_t\right)$则启发我们将$\boldsymbol{\mu}(\boldsymbol{x}_t)$参数化为$\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)$。

这一系列变换下来，优化目标等价于  
\begin{equation}\frac{\beta_t^2}{\alpha_t^2\sigma_t^2}\mathbb{E}_{\bar{\boldsymbol{\varepsilon}}_{t-1},\boldsymbol{\varepsilon}_t\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}),\boldsymbol{x}_0\sim \tilde{p}(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{\varepsilon}_t - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t, t)\right\Vert^2\right]\end{equation}  
随后按照“[降低方差](/archives/9119#%E9%99%8D%E4%BD%8E%E6%96%B9%E5%B7%AE)”一节做换元，结果就是  
\begin{equation}\frac{\beta_t^4}{\bar{\beta}_t^2\alpha_t^2\sigma_t^2}\mathbb{E}_{\boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}),\boldsymbol{x}_0\sim \tilde{p}(\boldsymbol{x}_0)}\left[\left\Vert\boldsymbol{\varepsilon} - \frac{\bar{\beta}_t}{\beta_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t)\right\Vert^2\right]\label{eq:loss}\end{equation}  
这就得到了DDPM的训练目标了（原论文通过实验发现，去掉上式前面的系数后实际效果更好些）。它是我们从VAE的优化目标出发，逐步简化积分结果得到的，虽然有点长，但每一步都是有章可循的，有计算难度，但没有思路上的难度。

相比之下，DDPM的原论文中，很突兀引入了一个$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$（原论文记号）来进行裂项相消，然后转化为正态分布的KL散度形式。整个过程的这一步技巧性太强，显得太过“莫名其妙”，对笔者来说相当难以接受。

## 超参设置 #

这一节我们来讨论一下$\alpha_t,\beta_t,\sigma_t$的选择问题。

对于$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$来说，习惯上约定$\alpha_t^2 + \beta_t^2=1$，这样就减少了一半的参数了，并且有助于简化形式，这其实在上一篇文章我们已经推导过了，由于正态分布的叠加性，在此约束之下我们有  
\begin{equation}p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \int p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) d\boldsymbol{x}_1\cdots d\boldsymbol{x}_{t-1} = \mathcal{N}(\boldsymbol{x}_t;\bar{\alpha}_t \boldsymbol{x}_0, \bar{\beta}_t^2 \boldsymbol{I})\label{eq:x0-xt}\end{equation}  
其中$\bar{\alpha}_t = \alpha_1\cdots\alpha_t$，而$\bar{\beta}_t = \sqrt{1-\bar{\alpha}_t^2}$，这样一来$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$就具有比较简约的形式。可能读者又想问事前是怎么想到$\alpha_t^2 + \beta_t^2=1$这个约束呢？我们知道$\mathcal{N}(\boldsymbol{x}_t;\alpha_t \boldsymbol{x}_{t-1}, \beta_t^2 \boldsymbol{I})$意味着$\boldsymbol{x}_t = \alpha_t \boldsymbol{x}_{t-1} + \beta_t \boldsymbol{\varepsilon}_t,\boldsymbol{\varepsilon}_t\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，如果$\boldsymbol{x}_{t-1}$也是$\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$的话，我们就希望$\boldsymbol{x}_t$也是$\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，所以就确定了$\alpha_t^2+\beta_t^2=1$了。

前面说了，$q(\boldsymbol{x}_T)$一般都取标准正态分布$\mathcal{N}(\boldsymbol{x}_T;\boldsymbol{0}, \boldsymbol{I})$。而我们的学习目标是最小化两个联合分布的KL散度，即希望$p=q$，那么它们的边缘分布自然也相等，所以我们也希望  
\begin{equation}q(\boldsymbol{x}_T) = \int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) d\boldsymbol{x}_0 d\boldsymbol{x}_1\cdots d\boldsymbol{x}_{T-1} = \int p(\boldsymbol{x}_T|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) d\boldsymbol{x}_0 \end{equation}  
由于数据分布$\tilde{p}(\boldsymbol{x}_0)$是任意的，所以要使上式恒成立，只能让$p(\boldsymbol{x}_T|\boldsymbol{x}_0)=q(\boldsymbol{x}_T)$，即退化为与$\boldsymbol{x}_0$无关的标准正态分布，这意味着我们要设计适当的$\alpha_t$，使得$\bar{\alpha}_T\approx 0$。同时这再次告诉我们，DDPM是没有编码能力了，最终的$p(\boldsymbol{x}_T|\boldsymbol{x}_0)$可以说跟输入$\boldsymbol{x}_0$无关的。用上一篇文章的“拆楼-建楼”类比就是说，原来的楼已经被完全拆成原材料了，如果用这堆材料重新建楼的话，可以建成任意样子的楼，而不一定是拆之前的样子。DDPM取了$\alpha_t = \sqrt{1 - \frac{0.02t}{T}}$，关于该选择的性质，我们在上一篇文章的“[超参设置](/archives/9119#%E8%B6%85%E5%8F%82%E8%AE%BE%E7%BD%AE)”一节也分析过了。

至于$\sigma_t$，理论上不同的数据分布$\tilde{p}(\boldsymbol{x}_0)$来说对应不同的最优$\sigma_t$，但我们又不想将$\sigma_t$设为可训练参数，所以只好选一些特殊的$\tilde{p}(\boldsymbol{x}_0)$来推导相应的最优$\sigma_t$，并认为由特例推导出来的$\sigma_t$可以泛化到一般的数据分布。我们可以考虑两个简单的例子：

> 1、假设训练集只有一个样本$\boldsymbol{x}_*$，即$\tilde{p}(\boldsymbol{x}_0)$是狄拉克分布$\delta(\boldsymbol{x}_0 - \boldsymbol{x}_*)$，可以推出最优的$\sigma_t = \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t}\beta_t$；
> 
> 2、假设数据分布$\tilde{p}(\boldsymbol{x}_0)$服从标准正态分布，这时候可以推出最优的$\sigma_t = \beta_t$。

实验结果显示两个选择的表现是相似的，因此可以选择任意一个进行采样。两个结果的推导过程有点长，我们后面再择机讨论。

## 参考实现 #

这么精彩的模型怎么可以少得了Keras实现？下面提供笔者的参考实现：

> **Github地址：<https://github.com/bojone/Keras-DDPM>**

注意，笔者的实现并非严格按照DDPM原始开源代码来进行，而是根据自己的设计简化了U-Net的架构（比如特征拼接改为相加、去掉了Attention等），使得可以快速出效果。经测试，在单张24G显存的3090下，以`blocks=1,batch_size=64`训练128*128大小的CelebA HQ人脸数据集，半天就能初见成效。训练3天后的采样效果如下：  


[![笔者训练的DDPM采样结果演示](/usr/uploads/2022/07/3342802728.png)](/usr/uploads/2022/07/3342802728.png "点击查看原图")

笔者训练的DDPM采样结果演示

在调试过程中，笔者总结出了如下的实践经验：

> 1、损失函数不能用mse，而必须用欧氏距离平方，两者的差别是mse在欧氏距离平方基础上除以图片的$\text{宽}\times\text{高}\times\text{通道数}$，这会导致损失值过小，部分参数的梯度可能会被忽略为0，从而导致训练过程先收敛后发散，该现象也经常出现于低精度训练中，可以参考[《在bert4keras中使用混合精度和XLA加速训练》](/archives/9059)；
> 
> 2、归一化方式可以用Instance Norm、Layer Norm、Group Norm等，但不要用Batch Norm，因为Batch Norm存在训练和推理不一致的问题，可能出现训练效果特别好，预测效果特别差的问题；
> 
> 3、网络结构没有必要照搬原论文，原论文是为了刷SOTA发论文，照搬的话肯定是又大又慢的，只需要按照U-Net的思路设计自编码器，就基本上可以训练出个大概效果了，因为就相当于是个纯粹的回归问题，还是很好训练的；
> 
> 4、关于参数$t$的传入，原论文用了[Sinusoidal位置编码](/archives/8231)，笔者发现直接换为可训练的Embedding，效果也差不多；
> 
> 5、按照以往搞语言模型预训练的习惯，笔者用了LAMB优化器，它更方便调学习率，基本上$10^{-3}$的学习率可以适用于任意初始化方式的模型训练。

## 综合评价 #

结合[《生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼》](/archives/9119)和本文的介绍，想必读者都已经对DDPM有自己的看法了，能基本看出DDPM优点、缺点以及相应的改进方向在哪了。

DDPM的优点很明显，就是容易训练，并且生成的图片也清晰。这个容易训练是相对GAN而言的，GAN是一个$\min\text{-}\max$过程，训练中的不确定性很大，容易崩溃，而DDPM就纯粹是一个回归的损失函数，只需要纯粹的最小化，因此训练过程非常平稳。同时，经过“拆楼-建楼”的类比，我们也可以发现DDPM在通俗理解方面其实也不逊色于GAN。

不过，DDPM的缺点也很明显。首先最突出的就是采样速度太慢，需要执行模型$T$步（原论文$T=1000$才能完成采样），可以说这比GAN的一步到位的采样要慢上$T$倍，后面有很多工作对这一点进行改进；其次，在GAN中，从随机噪声到生成样本的训练是一个确定性的变换，随机噪声是生成结果的一个解耦的隐变量，我们可以进行插值生成，或者对之编辑以实现控制生成等，但是DDPM中生成过程是一个完全随机的过程，两者没有确定性的关系，这种编辑生成就不存在了。DDPM原论文虽然也演示了插值生成效果，但那只是在原始图片上进行插值的，然后通过噪声来模糊图片，让模型重新“脑补”出新的图片，这种插值很难做到语义上的融合。

除了针对上述缺点来做改进外，DDPM还有其他一些可做的方向，比如目前演示的DDPM都是无条件的生成，那么很自然就想到有条件的DDPM的，就好比从VAE到C-VAE、从GAN到C-GAN一样，这也是当前扩散模型的一个主流应用，比如用Google的Imagen就同时包含了用扩散模型做文本生成图片以及做超分辨率，这两者本质上就是条件式扩散模型了；再比如，目前的DDPM是为连续型变量设计的，但从其思想来说应该也是适用于离散型数据的，那么离散型数据的DDPM怎么设计呢？

## 相关工作 #

说到DDPM的相关工作，多数人会想到传统扩散模型、能量模型等工作，又或者是去噪自编码器等工作，但笔者接下来想说的不是这些，而是本博客之前介绍过的、甚至可以认为DDPM就是它的特例的[《强大的NVAE：以后再也不能说VAE生成的图像模糊了》](/archives/7574)。

站在VAE的视角来看，传统VAE生成的图片都偏模糊，而DDPM只能算是（笔者所了解到的）第二个能生成清晰图像的VAE，第一个正是NVAE。翻看NVAE的形式，我们可以发现它跟DDPM有非常多的相似之处，比如NVAE也是引入了一大堆隐变量$z=\\{z_1,z_2,\dots,z_L\\}$，这些隐变量也呈递归关系，所以NVAE的采样过程跟DDPM也是很相似的。

从理论形式来说，DDPM可以看成是一个极度简化的NVAE，即隐变量的递归关系仅仅建模为马尔可夫式的条件正态分布，而不是像NVAE的非马尔科夫式，生成模型也只是同一个模型的反复迭代，而不是NVAE那样用一个庞大的模型同时用上了$z=\\{z_1,z_2,\dots,z_L\\}$，但NVAE在利用众多$z=\\{z_1,z_2,\dots,z_L\\}$之时，也加入了参数共享机制，这跟同一个模型反复迭代也异曲同工了。

## 文章小结 #

本文从变分自编码器VAE的角度推导了DDPM，在这个视角之下，DDPM是一个简化版的自回归式VAE，跟之前的NVAE很是相似。同时本文分享了自己的DDPM实现代码和实践经验，以及对DDPM做了一个比较综合的评价。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9152>_

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

苏剑林. (Jul. 06, 2022). 《生成扩散模型漫谈（二）：DDPM = 自回归式VAE 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9152>

@online{kexuefm-9152,  
title={生成扩散模型漫谈（二）：DDPM = 自回归式VAE},  
author={苏剑林},  
year={2022},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/9152}},  
} 


---

## 公式推导与注释

### 1. 核心概念与理论框架

#### 1.1 从传统VAE到DDPM

**传统VAE的基本结构:**

在变分自编码器(VAE)中,我们有:
- **编码过程:** $\boldsymbol{x} \to \boldsymbol{z}$ (一步到位)
- **生成过程:** $\boldsymbol{z} \to \boldsymbol{x}$ (一步到位)

涉及三个核心分布:
$$p(\boldsymbol{z}|\boldsymbol{x}): \text{编码分布}, \quad q(\boldsymbol{x}|\boldsymbol{z}): \text{生成分布}, \quad q(\boldsymbol{z}): \text{先验分布} \tag{1}$$

**传统VAE的限制:**
1. 编码和生成都只有一步,映射关系简单
2. 通常只能建模为正态分布,表达能力有限
3. 生成的图像往往模糊不清

**DDPM的突破:**

DDPM将一步变为$T$步,形成递归式的编码和生成:
$$\begin{aligned}
\text{编码:} \quad &\boldsymbol{x} = \boldsymbol{x}_0 \to \boldsymbol{x}_1 \to \boldsymbol{x}_2 \to \cdots \to \boldsymbol{x}_T = \boldsymbol{z} \\
\text{生成:} \quad &\boldsymbol{z} = \boldsymbol{x}_T \to \boldsymbol{x}_{T-1} \to \boldsymbol{x}_{T-2} \to \cdots \to \boldsymbol{x}_0 = \boldsymbol{x}
\end{aligned} \tag{2}$$

**关键思想:** 每一小步只建模微小变化,仍用正态分布,但多步累积可以逼近复杂分布。

**类比:** 用分段线性函数逼近复杂曲线。

#### 1.2 前向扩散过程的设计

**每一步的前向过程:**
$$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_t; \alpha_t\boldsymbol{x}_{t-1}, \beta_t^2\boldsymbol{I}) \tag{3}$$

**关键特点:**
1. **均值:** 只是输入乘以标量$\alpha_t$,不需要神经网络学习
2. **方差:** 事先设定好的$\beta_t^2$,也不需要学习
3. **目标:** 放弃编码能力,专注于生成

**等价采样形式:**
$$\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{4}$$

**反向生成过程:**
$$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}(\boldsymbol{x}_t), \sigma_t^2\boldsymbol{I}) \tag{5}$$

其中$\boldsymbol{\mu}(\boldsymbol{x}_t)$由神经网络学习。

### 2. 联合分布与KL散度

#### 2.1 联合分布的构造

**前向过程的联合分布:**
$$\begin{aligned}
p(\boldsymbol{x}_0, \boldsymbol{x}_1, \ldots, \boldsymbol{x}_T) &= p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_2|\boldsymbol{x}_1) p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \\
&= \tilde{p}(\boldsymbol{x}_0) \prod_{t=1}^T p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})
\end{aligned} \tag{6}$$

其中$\tilde{p}(\boldsymbol{x}_0)$是数据分布。

**反向过程的联合分布:**
$$\begin{aligned}
q(\boldsymbol{x}_0, \boldsymbol{x}_1, \ldots, \boldsymbol{x}_T) &= q(\boldsymbol{x}_0|\boldsymbol{x}_1) \cdots q(\boldsymbol{x}_{T-2}|\boldsymbol{x}_{T-1}) q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_T) q(\boldsymbol{x}_T) \\
&= q(\boldsymbol{x}_T) \prod_{t=1}^T q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)
\end{aligned} \tag{7}$$

其中$q(\boldsymbol{x}_T)$是先验分布(通常为$\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$)。

#### 2.2 VAE的最简洁理解

**核心优化目标:** 最小化两个联合分布的KL散度。
$$\mathcal{L} = KL(p \| q) = \int p(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T) \log \frac{p(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T)}{q(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T)} d\boldsymbol{x}_0 \cdots d\boldsymbol{x}_T \tag{8}$$

**展开:**
$$\begin{aligned}
\mathcal{L} &= \int p(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T) \left[\log p(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T) - \log q(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T)\right] d\boldsymbol{x}_0 \cdots d\boldsymbol{x}_T \\
&= \mathbb{E}_{p(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T)}\left[\log p(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T) - \log q(\boldsymbol{x}_0, \ldots, \boldsymbol{x}_T)\right]
\end{aligned} \tag{9}$$

**代入联合分布:**
$$\begin{aligned}
\mathcal{L} &= \mathbb{E}_{p}\left[\log \tilde{p}(\boldsymbol{x}_0) + \sum_{t=1}^T \log p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) - \log q(\boldsymbol{x}_T) - \sum_{t=1}^T \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)\right]
\end{aligned} \tag{10}$$

### 3. 目标函数的简化

#### 3.1 去除常数项

**观察1:** 前向分布$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$不含可训练参数,相关项只贡献常数。

**观察2:** 先验分布$q(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$也是固定的,贡献常数。

**简化后的目标:**
$$\mathcal{L} \propto -\mathbb{E}_{p}\left[\sum_{t=1}^T \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)\right] = -\sum_{t=1}^T \mathbb{E}_{p}\left[\log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)\right] \tag{11}$$

#### 3.2 单个时间步的目标

**对每个$t$:**
$$\begin{aligned}
\mathcal{L}_t &= -\int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) d\boldsymbol{x}_0 \cdots d\boldsymbol{x}_T
\end{aligned} \tag{12}$$

**关键观察:** $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$至多依赖到$\boldsymbol{x}_t$,因此$t+1$到$T$的积分可以先算。

**积分消除:**
$$\int p(\boldsymbol{x}_T|\boldsymbol{x}_{T-1}) \cdots p(\boldsymbol{x}_{t+1}|\boldsymbol{x}_t) d\boldsymbol{x}_{t+1} \cdots d\boldsymbol{x}_T = 1 \tag{13}$$

**进一步简化:**
$$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$也不依赖于$\boldsymbol{x}_1, \ldots, \boldsymbol{x}_{t-2}$,可以继续消除积分。

**关键公式:** 利用马尔可夫性和正态分布的叠加性:
$$\int p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t-2}) \cdots p(\boldsymbol{x}_1|\boldsymbol{x}_0) d\boldsymbol{x}_1 \cdots d\boldsymbol{x}_{t-2} = p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) \tag{14}$$

**最终简化形式:**
$$\mathcal{L}_t = -\int p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) \tilde{p}(\boldsymbol{x}_0) \log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) d\boldsymbol{x}_0 d\boldsymbol{x}_{t-1} d\boldsymbol{x}_t \tag{15}$$

### 4. 正态分布叠加性的推导

#### 4.1 两步情况的推导

**给定:**
- $p(\boldsymbol{x}_1|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_1; \alpha_1\boldsymbol{x}_0, \beta_1^2\boldsymbol{I})$
- $p(\boldsymbol{x}_2|\boldsymbol{x}_1) = \mathcal{N}(\boldsymbol{x}_2; \alpha_2\boldsymbol{x}_1, \beta_2^2\boldsymbol{I})$

**等价采样形式:**
$$\boldsymbol{x}_1 = \alpha_1\boldsymbol{x}_0 + \beta_1\boldsymbol{\varepsilon}_1, \quad \boldsymbol{\varepsilon}_1 \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{16}$$
$$\boldsymbol{x}_2 = \alpha_2\boldsymbol{x}_1 + \beta_2\boldsymbol{\varepsilon}_2, \quad \boldsymbol{\varepsilon}_2 \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{17}$$

**代入消元:**
$$\begin{aligned}
\boldsymbol{x}_2 &= \alpha_2(\alpha_1\boldsymbol{x}_0 + \beta_1\boldsymbol{\varepsilon}_1) + \beta_2\boldsymbol{\varepsilon}_2 \\
&= \alpha_2\alpha_1\boldsymbol{x}_0 + \alpha_2\beta_1\boldsymbol{\varepsilon}_1 + \beta_2\boldsymbol{\varepsilon}_2
\end{aligned} \tag{18}$$

**高斯噪声的和:** 两个独立高斯随机变量的和仍是高斯:
$$\alpha_2\beta_1\boldsymbol{\varepsilon}_1 + \beta_2\boldsymbol{\varepsilon}_2 \sim \mathcal{N}(\boldsymbol{0}, (\alpha_2^2\beta_1^2 + \beta_2^2)\boldsymbol{I}) \tag{19}$$

**验证方差:**
$$\text{Var}[\alpha_2\beta_1\boldsymbol{\varepsilon}_1 + \beta_2\boldsymbol{\varepsilon}_2] = \alpha_2^2\beta_1^2\text{Var}[\boldsymbol{\varepsilon}_1] + \beta_2^2\text{Var}[\boldsymbol{\varepsilon}_2] = \alpha_2^2\beta_1^2 + \beta_2^2 \tag{20}$$

**引入约束:** 为了简化形式,我们要求$\alpha_t^2 + \beta_t^2 = 1$。

**则:**
$$\begin{aligned}
\alpha_2^2\beta_1^2 + \beta_2^2 &= \alpha_2^2\beta_1^2 + (1 - \alpha_2^2) \\
&= \alpha_2^2\beta_1^2 + 1 - \alpha_2^2 \\
&= \alpha_2^2(\beta_1^2 - 1) + 1 \\
&= \alpha_2^2(-\alpha_1^2) + 1 \\
&= 1 - \alpha_2^2\alpha_1^2
\end{aligned} \tag{21}$$

**定义累积参数:**
$$\bar{\alpha}_2 = \alpha_1\alpha_2, \quad \bar{\beta}_2 = \sqrt{1 - \bar{\alpha}_2^2} \tag{22}$$

**最终结果:**
$$p(\boldsymbol{x}_2|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_2; \bar{\alpha}_2\boldsymbol{x}_0, \bar{\beta}_2^2\boldsymbol{I}) \tag{23}$$

#### 4.2 一般情况的递推

**归纳假设:** 假设$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \bar{\alpha}_{t-1}\boldsymbol{x}_0, \bar{\beta}_{t-1}^2\boldsymbol{I})$成立。

**递推步骤:**
$$\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t = \alpha_t(\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}) + \beta_t\boldsymbol{\varepsilon}_t \tag{24}$$

**整理:**
$$\boldsymbol{x}_t = \alpha_t\bar{\alpha}_{t-1}\boldsymbol{x}_0 + (\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t) \tag{25}$$

**噪声项的方差:**
$$\text{Var} = \alpha_t^2\bar{\beta}_{t-1}^2 + \beta_t^2 = \alpha_t^2(1 - \bar{\alpha}_{t-1}^2) + \beta_t^2 \tag{26}$$

**利用约束$\alpha_t^2 + \beta_t^2 = 1$:**
$$\begin{aligned}
\text{Var} &= \alpha_t^2(1 - \bar{\alpha}_{t-1}^2) + (1 - \alpha_t^2) \\
&= \alpha_t^2 - \alpha_t^2\bar{\alpha}_{t-1}^2 + 1 - \alpha_t^2 \\
&= 1 - \alpha_t^2\bar{\alpha}_{t-1}^2 \\
&= 1 - (\alpha_t\bar{\alpha}_{t-1})^2
\end{aligned} \tag{27}$$

**递推公式:**
$$\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1} = \alpha_1\alpha_2 \cdots \alpha_t = \prod_{i=1}^t \alpha_i \tag{28}$$
$$\bar{\beta}_t = \sqrt{1 - \bar{\alpha}_t^2} \tag{29}$$

**一般结果:**
$$p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \bar{\alpha}_t\boldsymbol{x}_0, \bar{\beta}_t^2\boldsymbol{I}) \tag{30}$$

**采样形式:**
$$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}}, \quad \bar{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{31}$$

### 5. 反向过程的参数化

#### 5.1 负对数似然的展开

**反向生成过程:**
$$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}(\boldsymbol{x}_t), \sigma_t^2\boldsymbol{I}) \tag{32}$$

**负对数似然(忽略常数):**
$$-\log q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \frac{1}{2\sigma_t^2}\|\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\|^2 + \text{const} \tag{33}$$

**优化目标(式15)变为:**
$$\mathcal{L}_t = \frac{1}{2\sigma_t^2} \mathbb{E}_{p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)}\left[\|\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t)\|^2\right] + \text{const} \tag{34}$$

#### 5.2 噪声预测的参数化

**前向过程给出:**
$$\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t \tag{35}$$

**反解$\boldsymbol{x}_{t-1}$:**
$$\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}(\boldsymbol{x}_t - \beta_t\boldsymbol{\varepsilon}_t) \tag{36}$$

**启发式参数化:** 用神经网络预测噪声$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$来逼近真实噪声$\boldsymbol{\varepsilon}_t$,则均值参数化为:
$$\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right) \tag{37}$$

#### 5.3 损失函数的变换

**代入参数化(37)到目标(34):**
$$\mathcal{L}_t = \frac{1}{2\sigma_t^2} \mathbb{E}\left[\left\|\boldsymbol{x}_{t-1} - \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right)\right\|^2\right] \tag{38}$$

**从前向过程,我们有:**
- $\boldsymbol{x}_{t-1} = \bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}$
- $\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t$

**代入第一个:**
$$\boldsymbol{x}_t = \alpha_t(\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}) + \beta_t\boldsymbol{\varepsilon}_t \tag{39}$$

**目标函数变为:**
$$\begin{aligned}
&\left\|\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} - \frac{1}{\alpha_t}\left(\alpha_t(\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}) + \beta_t\boldsymbol{\varepsilon}_t - \beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\right)\right\|^2 \\
=& \left\|\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} - \bar{\alpha}_{t-1}\boldsymbol{x}_0 - \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} - \frac{\beta_t}{\alpha_t}\boldsymbol{\varepsilon}_t + \frac{\beta_t}{\alpha_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\right\|^2 \\
=& \left\|\frac{\beta_t}{\alpha_t}(\boldsymbol{\epsilon}_{\boldsymbol{\theta}} - \boldsymbol{\varepsilon}_t)\right\|^2 \\
=& \frac{\beta_t^2}{\alpha_t^2}\|\boldsymbol{\epsilon}_{\boldsymbol{\theta}} - \boldsymbol{\varepsilon}_t\|^2
\end{aligned} \tag{40}$$

**简化的损失函数:**
$$\mathcal{L}_t = \frac{\beta_t^2}{2\alpha_t^2\sigma_t^2} \mathbb{E}_{\bar{\boldsymbol{\varepsilon}}_{t-1}, \boldsymbol{\varepsilon}_t, \boldsymbol{x}_0}\left[\|\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{\varepsilon}_t\|^2\right] \tag{41}$$

其中$\boldsymbol{x}_t = \alpha_t(\bar{\alpha}_{t-1}\boldsymbol{x}_0 + \bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1}) + \beta_t\boldsymbol{\varepsilon}_t$。

### 6. 降低方差的技巧

#### 6.1 问题分析

**当前形式的问题:** 期望同时涉及两个独立噪声$\bar{\boldsymbol{\varepsilon}}_{t-1}$和$\boldsymbol{\varepsilon}_t$,增加了方差。

**目标:** 将其转化为单个噪声的期望。

#### 6.2 重参数化技巧

**回顾累积前向过程:**
$$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}}, \quad \bar{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{42}$$

**目标:** 用单个噪声$\bar{\boldsymbol{\varepsilon}}$表示$\boldsymbol{x}_t$。

**从前面的推导:**
$$\boldsymbol{x}_t = \alpha_t\bar{\alpha}_{t-1}\boldsymbol{x}_0 + (\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t) \tag{43}$$

**注意到:**
$$\bar{\alpha}_t = \alpha_t\bar{\alpha}_{t-1} \tag{44}$$

**噪声合成:** 定义
$$\bar{\boldsymbol{\varepsilon}} = \frac{\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t}{\bar{\beta}_t} \tag{45}$$

**验证:** 需要证明$\bar{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

**方差计算:**
$$\begin{aligned}
\text{Var}[\bar{\boldsymbol{\varepsilon}}] &= \frac{1}{\bar{\beta}_t^2}(\alpha_t^2\bar{\beta}_{t-1}^2 + \beta_t^2) \\
&= \frac{1}{1 - \alpha_t^2\bar{\alpha}_{t-1}^2}(\alpha_t^2(1 - \bar{\alpha}_{t-1}^2) + \beta_t^2) \\
&= \frac{\alpha_t^2 - \alpha_t^2\bar{\alpha}_{t-1}^2 + \beta_t^2}{1 - \alpha_t^2\bar{\alpha}_{t-1}^2} \\
&= \frac{(\alpha_t^2 + \beta_t^2) - \alpha_t^2\bar{\alpha}_{t-1}^2}{1 - \alpha_t^2\bar{\alpha}_{t-1}^2} \\
&= \frac{1 - \alpha_t^2\bar{\alpha}_{t-1}^2}{1 - \alpha_t^2\bar{\alpha}_{t-1}^2} \\
&= 1
\end{aligned} \tag{46}$$

**因此:**
$$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}} \tag{47}$$

#### 6.3 噪声的反解

**从(47)反解噪声:**
$$\bar{\boldsymbol{\varepsilon}} = \frac{\boldsymbol{x}_t - \bar{\alpha}_t\boldsymbol{x}_0}{\bar{\beta}_t} \tag{48}$$

**代入到噪声合成公式(45):**
$$\frac{\boldsymbol{x}_t - \bar{\alpha}_t\boldsymbol{x}_0}{\bar{\beta}_t} = \frac{\alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t}{\bar{\beta}_t} \tag{49}$$

**因此:**
$$\beta_t\boldsymbol{\varepsilon}_t = \boldsymbol{x}_t - \bar{\alpha}_t\boldsymbol{x}_0 - \alpha_t\bar{\beta}_{t-1}\bar{\boldsymbol{\varepsilon}}_{t-1} \tag{50}$$

但这个形式仍然复杂。

**更简洁的方法:** 直接用(47)的形式。

#### 6.4 最终损失函数

**重写损失函数:** 用单个噪声$\bar{\boldsymbol{\varepsilon}}$:
$$\mathcal{L}_t = \frac{\beta_t^2}{2\alpha_t^2\sigma_t^2} \mathbb{E}_{\bar{\boldsymbol{\varepsilon}} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}), \boldsymbol{x}_0 \sim \tilde{p}}\left[\|\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}}, t) - \text{noise target}\|^2\right] \tag{51}$$

**问题:** noise target应该是什么?

**关键观察:** 在式(40)的推导中,真正的噪声目标是$\boldsymbol{\varepsilon}_t$,但我们希望用$\bar{\boldsymbol{\varepsilon}}$。

**重新推导:** 从
$$\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}} \tag{52}$$

**用第二个形式代入均值:**
$$\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \beta_t\boldsymbol{\epsilon}_{\boldsymbol{\theta}}\right) \tag{53}$$

**理想情况:** 如果$\boldsymbol{\epsilon}_{\boldsymbol{\theta}} = \bar{\boldsymbol{\varepsilon}}$,则:
$$\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}} - \beta_t\bar{\boldsymbol{\varepsilon}}\right) = \frac{1}{\alpha_t}\left(\bar{\alpha}_t\boldsymbol{x}_0 + (\bar{\beta}_t - \beta_t)\bar{\boldsymbol{\varepsilon}}\right) \tag{54}$$

**这不对!** 让我重新思考...

**正确的理解:**

从$\boldsymbol{x}_{t-1} = \frac{1}{\alpha_t}(\boldsymbol{x}_t - \beta_t\boldsymbol{\varepsilon}_t)$,如果我们预测的是$\boldsymbol{\varepsilon}_t$,这是正确的。

但从累积形式$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\bar{\boldsymbol{\varepsilon}}$,我们实际上可以预测$\bar{\boldsymbol{\varepsilon}}$。

**重新参数化:** 定义
$$\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t}{\bar{\beta}_t}\bar{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right) \tag{55}$$

其中$\bar{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}$预测累积噪声$\bar{\beta}_t\bar{\boldsymbol{\varepsilon}}$。

**但实践中:** DDPM论文使用更简单的形式,直接预测某种噪声,损失函数为:
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{\varepsilon}}\left[\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t)\|^2\right] \tag{56}$$

这里$\boldsymbol{\varepsilon}$就是累积噪声$\bar{\boldsymbol{\varepsilon}}$。

**理论系数:** 完整的带权重的损失为:
$$\mathcal{L}_t = \frac{\beta_t^4}{2\bar{\beta}_t^2\alpha_t^2\sigma_t^2} \mathbb{E}\left[\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\|^2\right] \tag{57}$$

**但DDPM发现:** 去掉权重系数,使用简化损失(56)效果更好!

### 7. 超参数设置与理论分析

#### 7.1 方差约束的意义

**约束:** $\alpha_t^2 + \beta_t^2 = 1$

**目的:** 保持方差恒定。

**推理:** 假设$\boldsymbol{x}_{t-1} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$(标准化假设),则:
$$\begin{aligned}
\text{Var}[\boldsymbol{x}_t] &= \text{Var}[\alpha_t\boldsymbol{x}_{t-1} + \beta_t\boldsymbol{\varepsilon}_t] \\
&= \alpha_t^2\text{Var}[\boldsymbol{x}_{t-1}] + \beta_t^2\text{Var}[\boldsymbol{\varepsilon}_t] \\
&= \alpha_t^2 \cdot 1 + \beta_t^2 \cdot 1 \\
&= \alpha_t^2 + \beta_t^2
\end{aligned} \tag{58}$$

如果$\alpha_t^2 + \beta_t^2 = 1$,则$\text{Var}[\boldsymbol{x}_t] = 1$,方差保持不变。

#### 7.2 先验匹配条件

**目标:** 希望$p(\boldsymbol{x}_T|\boldsymbol{x}_0) \approx q(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

**从累积公式:**
$$p(\boldsymbol{x}_T|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_T; \bar{\alpha}_T\boldsymbol{x}_0, \bar{\beta}_T^2\boldsymbol{I}) \tag{59}$$

**要使其接近$\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$,需要:**
1. $\bar{\alpha}_T \approx 0$(均值消失)
2. $\bar{\beta}_T \approx 1$(方差为1)

由于$\bar{\beta}_T = \sqrt{1 - \bar{\alpha}_T^2}$,条件1自动保证条件2。

**因此关键是:** $\bar{\alpha}_T = \prod_{t=1}^T \alpha_t \approx 0$

#### 7.3 $\alpha_t$的选择

**DDPM的设置:** $\alpha_t = \sqrt{1 - \frac{0.02t}{T}}$

**分析:**
$$\bar{\alpha}_T = \prod_{t=1}^T \sqrt{1 - \frac{0.02t}{T}} \tag{60}$$

**对数形式:**
$$\log \bar{\alpha}_T = \sum_{t=1}^T \frac{1}{2}\log\left(1 - \frac{0.02t}{T}\right) \tag{61}$$

**当$T$很大时,可以近似为积分:**
$$\log \bar{\alpha}_T \approx \frac{T}{2} \int_0^1 \log(1 - 0.02\tau) d\tau \tag{62}$$

**计算积分:**
$$\int_0^1 \log(1 - 0.02\tau) d\tau = \left[(\tau - 1)\log(1 - 0.02\tau) - \tau\right]_0^1 = 0 \cdot \log(0.98) - 1 - ((-1)\log(1) - 0) = -1 \tag{63}$$

因此:
$$\log \bar{\alpha}_T \approx -\frac{T}{2} \tag{64}$$
$$\bar{\alpha}_T \approx e^{-T/2} \tag{65}$$

**当$T = 1000$时:**
$$\bar{\alpha}_T \approx e^{-500} \approx 0 \tag{66}$$

确实满足条件!

#### 7.4 生成方差$\sigma_t$的选择

**理论推导两个极端情况:**

**情况1: 单样本数据集**

假设$\tilde{p}(\boldsymbol{x}_0) = \delta(\boldsymbol{x}_0 - \boldsymbol{x}_*)$,即训练集只有一个样本$\boldsymbol{x}_*$。

**最优的$\sigma_t$:** 通过最小化KL散度,可以推导出:
$$\sigma_t^2 = \frac{\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\beta_t^2 \tag{67}$$

**推导sketch:** 利用贝叶斯公式和高斯分布的性质,计算$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$,然后对$\boldsymbol{x}_0 = \boldsymbol{x}_*$固定。

**情况2: 标准正态先验**

假设$\tilde{p}(\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

**最优的$\sigma_t$:** 可以推导出:
$$\sigma_t^2 = \beta_t^2 \tag{68}$$

**实践:** 两种选择性能相近,DDPM使用第一种。

### 8. 完整的训练与采样算法

#### 8.1 训练算法

**输入:** 数据集$\{\boldsymbol{x}_0^{(i)}\}_{i=1}^N$, 时间步数$T$

**超参数:** $\{\alpha_t\}_{t=1}^T$满足$\alpha_t^2 + \beta_t^2 = 1$

**算法:**
```
初始化: 随机初始化网络参数θ

repeat:
    # 采样数据
    x0 ~ p_data

    # 采样时间步
    t ~ Uniform(1, T)

    # 采样噪声
    ε ~ N(0, I)

    # 构造加噪样本
    x_t = bar_alpha_t * x0 + bar_beta_t * ε

    # 计算损失
    loss = ||ε - ε_θ(x_t, t)||^2

    # 梯度更新
    θ ← θ - lr * ∇_θ loss

until 收敛
```

#### 8.2 采样算法

**输入:** 训练好的模型$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$

**输出:** 生成样本$\boldsymbol{x}_0$

**算法:**
```
# 从先验采样
x_T ~ N(0, I)

for t = T to 1:
    # 预测噪声
    ε_pred = ε_θ(x_t, t)

    # 计算均值
    μ = (x_t - (β_t / bar_β_t) * ε_pred) / α_t

    # 采样噪声(最后一步不加)
    if t > 1:
        z ~ N(0, I)
        x_{t-1} = μ + σ_t * z
    else:
        x_0 = μ

return x_0
```

### 9. 与NVAE的联系

#### 9.1 NVAE的结构

**NVAE (Nouveau VAE)** 也引入了多层隐变量:
$$\boldsymbol{z} = \{\boldsymbol{z}_1, \boldsymbol{z}_2, \ldots, \boldsymbol{z}_L\} \tag{69}$$

**编码过程(非马尔可夫):**
$$p(\boldsymbol{z}|\boldsymbol{x}) = p(\boldsymbol{z}_L|\boldsymbol{x}) \prod_{l=1}^{L-1} p(\boldsymbol{z}_l|\boldsymbol{z}_{>l}, \boldsymbol{x}) \tag{70}$$

**生成过程:**
$$q(\boldsymbol{x}|\boldsymbol{z}) = q(\boldsymbol{x}|\boldsymbol{z}_1) \prod_{l=1}^{L-1} q(\boldsymbol{z}_l|\boldsymbol{z}_{l+1}) \tag{71}$$

#### 9.2 与DDPM的比较

**相似之处:**
1. 都使用多层隐变量
2. 都是递归式的结构
3. 都能生成清晰图像

**差异:**

| 特性 | DDPM | NVAE |
|------|------|------|
| 编码 | 马尔可夫(固定噪声) | 非马尔可夫(学习) |
| 参数 | 只有生成网络 | 编码和生成都学习 |
| 复杂度 | 简单(参数共享) | 复杂(多层不同网络) |
| 理论 | 固定前向过程 | 灵活的双向 |

**DDPM可以看作:** 极度简化的NVAE
- 编码固定为简单加噪
- 生成使用单一网络反复迭代
- 参数共享机制

### 10. 理论深度分析

#### 10.1 为什么多步比单步好?

**单步VAE的问题:**
$$\boldsymbol{x} \xrightarrow{p(\boldsymbol{z}|\boldsymbol{x})} \boldsymbol{z} \xrightarrow{q(\boldsymbol{x}|\boldsymbol{z})} \boldsymbol{x}' \tag{72}$$

**限制:**
- $p(\boldsymbol{z}|\boldsymbol{x}) = \mathcal{N}(\boldsymbol{\mu}_{\text{enc}}(\boldsymbol{x}), \boldsymbol{\Sigma}_{\text{enc}}(\boldsymbol{x}))$
- $q(\boldsymbol{x}|\boldsymbol{z}) = \mathcal{N}(\boldsymbol{\mu}_{\text{dec}}(\boldsymbol{z}), \boldsymbol{\Sigma}_{\text{dec}}(\boldsymbol{z}))$

虽然$\boldsymbol{\mu}, \boldsymbol{\Sigma}$可以是任意函数,但单步变换的表达能力有限。

**多步的优势:**

**定理(万能逼近):** 给定任意连续分布$p_0$和$p_T$,存在充分光滑的路径:
$$p_0 \to p_1 \to \cdots \to p_T \tag{73}$$

使得每个$p_{t-1} \to p_t$可以用简单的正态分布逼近。

**直觉:**
- 大跳跃难以用简单分布建模
- 小步连续变化容易建模
- 类似于数值微分方程求解

#### 10.2 优化景观分析

**单步VAE的损失:**
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{p(\boldsymbol{z}|\boldsymbol{x})}\left[-\log q(\boldsymbol{x}|\boldsymbol{z})\right] + KL(p(\boldsymbol{z}|\boldsymbol{x}) \| q(\boldsymbol{z})) \tag{74}$$

**问题:** KL散度项可能导致后验崩塌(posterior collapse)。

**DDPM的损失:**
$$\mathcal{L}_{\text{DDPM}} = \sum_{t=1}^T \mathbb{E}\left[\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\|^2\right] \tag{75}$$

**优势:**
1. 纯回归损失,优化简单
2. 每步独立优化,无耦合
3. 不会后验崩塌

#### 10.3 生成质量的理论保证

**定理:** 如果$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$完美学习噪声,即:
$$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = \mathbb{E}[\boldsymbol{\varepsilon}|\boldsymbol{x}_t] \tag{76}$$

则反向过程$q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$的最优均值为:
$$\boldsymbol{\mu}^*(\boldsymbol{x}_t) = \frac{1}{\alpha_t}\left(\boldsymbol{x}_t - \frac{\beta_t^2}{\bar{\beta}_t}\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right) \tag{77}$$

其中$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$是得分函数(score function)。

**证明sketch:**
- 得分函数$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) = -\frac{\mathbb{E}[\boldsymbol{\varepsilon}|\boldsymbol{x}_t]}{\bar{\beta}_t}$
- 代入即得

**意义:** DDPM实际上在学习数据分布的得分函数!

### 11. 实践技巧总结

#### 11.1 训练技巧

**1. 损失函数选择:**
- 理论: 带权重的$\frac{\beta_t^4}{2\bar{\beta}_t^2\alpha_t^2\sigma_t^2}\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\|^2$
- 实践: 简化的$\|\boldsymbol{\varepsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\|^2$效果更好

**2. 归一化:**
- 使用Instance Norm / Layer Norm / Group Norm
- **避免** Batch Norm(训练推理不一致)

**3. 时间编码:**
- Sinusoidal位置编码
- 或可训练的Embedding

**4. 优化器:**
- Adam / AdamW
- 较大学习率(如$10^{-4}$)

#### 11.2 采样技巧

**1. 方差选择:**
$$\sigma_t = \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t}\beta_t \quad \text{或} \quad \sigma_t = \beta_t \tag{78}$$

**2. 重要性采样:**
不同时间步$t$对最终质量的贡献不同,可以调整采样分布。

**3. 降噪强度:**
可以在最后几步减小方差,使生成更稳定。

### 12. 总结与展望

#### 12.1 核心贡献总结

**DDPM = 自回归式VAE:**
1. **理论:** 从VAE的联合分布KL散度出发
2. **简化:** 固定前向过程为简单加噪
3. **参数化:** 用噪声预测而非直接预测$\boldsymbol{x}_{t-1}$
4. **优化:** 纯回归损失,训练稳定

**优点:**
- 生成质量高(清晰图像)
- 训练简单稳定
- 理论基础清晰

**缺点:**
- 采样慢($T$步迭代)
- 无语义隐空间(不能编辑)

#### 12.2 后续改进方向

**已有工作:**
1. **加速采样:** DDIM, DPM-Solver, 蒸馏
2. **条件生成:** Classifier-Guidance, Classifier-Free
3. **架构改进:** Latent Diffusion(在VAE隐空间做扩散)

**开放问题:**
1. 如何结合DDPM的生成质量和VAE的可编辑性?
2. 能否设计更优的前向过程?
3. 理论上的最优步数$T$是多少?
4. 如何扩展到离散数据(文本)?

---

**参考文献:**
- DDPM: [Denoising Diffusion Probabilistic Models](https://papers.cool/arxiv/2006.11239)
- NVAE: [NVAE: A Deep Hierarchical Variational Autoencoder](https://papers.cool/arxiv/2007.03898)
- Score-based: [Score-Based Generative Modeling](https://papers.cool/arxiv/2011.13456)

