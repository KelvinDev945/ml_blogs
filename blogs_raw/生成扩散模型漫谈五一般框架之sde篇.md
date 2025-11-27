---
title: 生成扩散模型漫谈（五）：一般框架之SDE篇
slug: 生成扩散模型漫谈五一般框架之sde篇
date: 2022-08-03
tags: 详细推导, 微分方程, 生成模型, DDPM, 扩散, 生成模型
status: completed
---
# 生成扩散模型漫谈（五）：一般框架之SDE篇

**原文链接**: [https://spaces.ac.cn/archives/9209](https://spaces.ac.cn/archives/9209)

**发布日期**: 

---

在写[生成扩散模型](/search/%E7%94%9F%E6%88%90%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B/)的第一篇文章时，就有读者在评论区推荐了宋飏博士的论文[《Score-Based Generative Modeling through Stochastic Differential Equations》](https://papers.cool/arxiv/2011.13456)，可以说该论文构建了一个相当一般化的生成扩散模型理论框架，将DDPM、SDE、ODE等诸多结果联系了起来。诚然，这是一篇好论文，但并不是一篇适合初学者的论文，里边直接用到了随机微分方程（SDE）、Fokker-Planck方程、得分匹配等大量结果，上手难度还是颇大的。

不过，在经过了前四篇文章的积累后，现在我们可以尝试去学习一下这篇论文了。在接下来的文章中，笔者将尝试从尽可能少的理论基础出发，尽量复现原论文中的推导结果。

## 随机微分 #

在DDPM中，扩散过程被划分为了固定的$T$步，还是用[《生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼》](/archives/9119)的类比来说，就是“拆楼”和“建楼”都被事先划分为了$T$步，这个划分有着相当大的人为性。事实上，真实的“拆”、“建”过程应该是没有刻意划分的步骤的，我们可以将它们理解为一个在时间上连续的变换过程，可以用随机微分方程（Stochastic Differential Equation，SDE）来描述。

为此，我们用下述SDE描述前向过程（“拆楼”）：  
\begin{equation}d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}\label{eq:sde-forward}\end{equation}  
相信很多读者都对SDE很陌生，笔者也只是在硕士阶段刚好接触过一段时间，略懂皮毛。不过不懂不要紧，我们只需要将它看成是下述离散形式在$\Delta t\to 0$时的极限：  
\begin{equation}\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:sde-discrete}\end{equation}  
再直白一点，如果假设拆楼需要$1$天，那么拆楼就是$\boldsymbol{x}$从$t=0$到$t=1$的变化过程，每一小步的变化我们可以用上述方程描述。至于时间间隔$\Delta t$，我们并没有做特殊限制，只是越小的$\Delta t$意味着是对原始SDE越好的近似，如果取$\Delta t=0.001$，那就对应于原来的$T=1000$，如果是$\Delta t = 0.01$则对应于$T=100$，等等。也就是说，在连续时间的SDE视角之下，不同的$T$是SDE不同的离散化程度的体现，它们会自动地导致相似的结果，我们不需要事先指定$T$，而是根据实际情况下的精确度来取适当的$T$进行数值计算。

所以，引入SDE形式来描述扩散模型的本质好处是“将理论分析和代码实现分离开来”，我们可以借助连续性SDE的数学工具对它做分析，而实践的时候，则只需要用任意适当的离散化方案对SDE进行数值计算。

对于式$\eqref{eq:sde-discrete}$，读者可能比较有疑惑的是为什么右端第一项是$\mathcal{O}(\Delta t)$的，而第二项是$\mathcal{O}(\sqrt{\Delta t})$的？也就是说为什么随机项的阶要比确定项的阶要高？这个还真不是那么容易解释，也是SDE比较让人迷惑的地方之一。简单来说，就是$\boldsymbol{\varepsilon}$一直服从标准正态分布，如果随机项的权重也是$\mathcal{O}(\Delta t)$，那么由于标准正态分布的均值为$\boldsymbol{0}$、协方差为$ \boldsymbol{I}$，临近的随机效应会相互抵消掉，要放大到$\mathcal{O}(\sqrt{\Delta t})$才能在长期结果中体现出随机效应的作用。

## 逆向方程 #

用概率的语言，式$\eqref{eq:sde-discrete}$意味着条件概率为  
\begin{equation}\begin{aligned}  
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) =&\, \mathcal{N}\left(\boldsymbol{x}_{t+\Delta t};\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t, g_t^2\Delta t \,\boldsymbol{I}\right)\\\  
\propto&\, \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\Vert^2}{2 g_t^2\Delta t}\right)  
\end{aligned}\label{eq:sde-proba}\end{equation}  
简单起见，这里没有写出无关紧要的归一化因子。按照DDPM的思想，我们最终是想要从“拆楼”的过程中学会“建楼”，即得到$p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t})$，为此，我们像[《生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪》](/archives/9164)一样，用贝叶斯定理：  
\begin{equation}\begin{aligned}  
p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) =&\, \frac{p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t)p(\boldsymbol{x}_t)}{p(\boldsymbol{x}_{t+\Delta t})} = p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) \exp\left(\log p(\boldsymbol{x}_t) - \log p(\boldsymbol{x}_{t+\Delta t})\right)\\\  
\propto&\, \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\Vert^2}{2 g_t^2\Delta t} + \log p(\boldsymbol{x}_t) - \log p(\boldsymbol{x}_{t+\Delta t})\right)  
\end{aligned}\label{eq:bayes-dt}\end{equation}  
不难发现，当$\Delta t$足够小时，只有当$\boldsymbol{x}_{t+\Delta t}$与$\boldsymbol{x}_t$足够接近时，$p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t)$才会明显不等于0，反过来也只有这种情况下$p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t})$才会明显不等于0。因此，我们只需要对$\boldsymbol{x}_{t+\Delta t}$与$\boldsymbol{x}_t$足够接近时的情形做近似分析，为此，我们可以用泰勒展开：  
\begin{equation}\log p(\boldsymbol{x}_{t+\Delta t})\approx \log p(\boldsymbol{x}_t) + (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) + \Delta t \frac{\partial}{\partial t}\log p(\boldsymbol{x}_t)\end{equation}  
注意不要忽略了$\frac{\partial}{\partial t}$项，因为$p(\boldsymbol{x}_t)$实际上是“$t$时刻随机变量等于$\boldsymbol{x}_t$的概率密度”，而$p(\boldsymbol{x}_{t+\Delta t})$实际上是“$t+\Delta t$时刻随机变量等于$\boldsymbol{x}_{t+\Delta t}$的概率密度”，也就是说$p(\boldsymbol{x}_t)$实际上同时是$t$和$\boldsymbol{x}_t$的函数，所以要多一项$t$的偏导数。代入到式$\eqref{eq:bayes-dt}$后，配方得到  
\begin{equation}p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) \propto \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) \right]\Delta t\Vert^2}{2 g_t^2\Delta t} + \mathcal{O}(\Delta t)\right)\end{equation}  
当$\Delta t\to 0$时，$\mathcal{O}(\Delta t)\to 0$不起作用，因此  
\begin{equation}\begin{aligned}  
p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) \propto&\, \exp\left(-\frac{\Vert\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) \right]\Delta t\Vert^2}{2 g_t^2\Delta t}\right) \\\  
\approx&\,\exp\left(-\frac{\Vert \boldsymbol{x}_t - \boldsymbol{x}_{t+\Delta t} + \left[\boldsymbol{f}_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - g_{t+\Delta t}^2\nabla_{\boldsymbol{x}_{t+\Delta t}}\log p(\boldsymbol{x}_{t+\Delta t}) \right]\Delta t\Vert^2}{2 g_{t+\Delta t}^2\Delta t}\right)  
\end{aligned}\end{equation}  
即$p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t})$近似一个均值为$\boldsymbol{x}_{t+\Delta t} - \left[\boldsymbol{f}_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - g_{t+\Delta t}^2\nabla_{\boldsymbol{x}_{t+\Delta t}}\log p(\boldsymbol{x}_{t+\Delta t}) \right]\Delta t$、协方差为$g_{t+\Delta t}^2\Delta t\,\boldsymbol{I}$的正态分布，取$\Delta t\to 0$的极限，那么对应于SDE：  
\begin{equation}d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) \right] dt + g_t d\boldsymbol{w}\label{eq:reverse-sde}\end{equation}  
这就是反向过程对应的SDE，最早出现在[《Reverse-Time Diffusion Equation Models》](https://www.sciencedirect.com/science/article/pii/0304414982900515)中。这里我们特意在$p$处标注了下标$t$，以突出这是$t$时刻的分布。

## 得分匹配 #

现在我们已经得到了逆向的SDE为$\eqref{eq:reverse-sde}$，如果进一步知道$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$，那么就可以通过离散化格式  
\begin{equation}\boldsymbol{x}_t - \boldsymbol{x}_{t+\Delta t} = - \left[\boldsymbol{f}_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - g_{t+\Delta t}^2\nabla_{\boldsymbol{x}_{t+\Delta t}}\log p(\boldsymbol{x}_{t+\Delta t}) \right]\Delta t - g_{t+\Delta t} \sqrt{\Delta t}\boldsymbol{\varepsilon}\label{eq:reverse-sde-discrete}\end{equation}  
来逐步完成“建楼”的生成过程【其中$\boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$】，从而完成一个生成扩散模型的构建。

那么如何得到$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$呢？$t$时刻的$p_t(\boldsymbol{x})$就是前面的$p(\boldsymbol{x}_t)$，它的含义就是$t$时刻的边缘分布。在实际使用时，我们一般会设计能找到$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$解析解的模型，这意味着  
\begin{equation}\small p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \lim_{\Delta t\to 0}\int\cdots\iint p(\boldsymbol{x}_t|\boldsymbol{x}_{t-\Delta t})p(\boldsymbol{x}_{t-\Delta t}|\boldsymbol{x}_{t-2\Delta t})\cdots p(\boldsymbol{x}_{\Delta t}|\boldsymbol{x}_0) d\boldsymbol{x}_{t-\Delta t} d\boldsymbol{x}_{t-2\Delta t}\cdots d\boldsymbol{x}_{\Delta t}\end{equation}  
是可以直接求出的，比如当$\boldsymbol{f}_t(\boldsymbol{x})$是关于$\boldsymbol{x}$的线性函数时，$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$就可以解析求解。在此前提下，有  
\begin{equation}p(\boldsymbol{x}_t) = \int p(\boldsymbol{x}_t|\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)d\boldsymbol{x}_0=\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]\end{equation}  
于是  
\begin{equation}\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) = \frac{\mathbb{E}_{\boldsymbol{x}_0}\left[\nabla_{\boldsymbol{x}_t} p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]}{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]} = \frac{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]}{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]}\end{equation}  
可以看到最后的式子具有“$\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)$的加权平均”的形式，由于假设了$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$有解析解，因此上式实际上是能够直接估算的，然而它涉及到对全体训练样本$\boldsymbol{x}_0$的平均，一来计算量大，二来泛化能力也不够好。因此，我们希望用神经网络学一个函数$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$，使得它能够直接计算$\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t)$。

很多读者应该对如下结果并不陌生（或者推导一遍也不困难）：  
\begin{equation}\mathbb{E}[\boldsymbol{x}] = \mathop{\text{argmin}}_{\boldsymbol{\mu}}\mathbb{E}_{\boldsymbol{x}}\left[\Vert \boldsymbol{\mu} - \boldsymbol{x}\Vert^2\right]\end{equation}  
即要让$\boldsymbol{\mu}$等于$\boldsymbol{x}$的均值，只需要最小化$\Vert \boldsymbol{\mu} - \boldsymbol{x}\Vert^2$的均值。同理，要让$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$等于$\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)$的加权平均【即$\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t)$】，则只需要最小化$\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2$的加权平均，即  
\begin{equation} \frac{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right]}{\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]}\end{equation}  
分母的$\mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]$只是起到调节Loss权重的作用，简单起见我们可以直接去掉它，这不会影响最优解的结果。最后我们再对$\boldsymbol{x}_t$积分（相当于对于每一个$\boldsymbol{x}_t$都要最小化上述损失），得到最终的损失函数  
\begin{equation}\begin{aligned}&\,\int \mathbb{E}_{\boldsymbol{x}_0}\left[p(\boldsymbol{x}_t|\boldsymbol{x}_0)\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right] d\boldsymbol{x}_t \\\  
=&\, \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t \sim p(\boldsymbol{x}_t|\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right]  
\end{aligned}\label{eq:score-match}\end{equation}  
这就是“（条件）得分匹配”的损失函数，之前我们在[《从去噪自编码器到生成模型》](/archives/7038)推导的去噪自编码器的解析解，也是它的一个特例。得分匹配的最早出处可以追溯到2005年的论文[《Estimation of Non-Normalized Statistical Models by Score Matching》](https://www.jmlr.org/papers/v6/hyvarinen05a.html)，至于条件得分匹配的最早出处，笔者追溯到的是2011年的论文[《A Connection Between Score Matching and Denoising Autoencoders》](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)。

不过，虽然该结果跟得分匹配是一样的，但其实在这一节的推导中，我们已经抛开了“得分”的概念了，纯粹是由目标自然地引导出来的答案，笔者认为这样的处理过程更有启发性，希望这一推导能降低大家对得分匹配的理解难度。

## 结果倒推 #

至此，我们构建了生成扩散模型的一般流程：

> 1、通过随机微分方程$\eqref{eq:sde-forward}$定义“拆楼”（前向过程）；
> 
> 2、求$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$的表达式；
> 
> 3、通过损失函数$\eqref{eq:score-match}$训练$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$（得分匹配）；
> 
> 4、用$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$替换式$\eqref{eq:reverse-sde}$的$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$，完成“建楼”（反向过程）。

可能大家看到SDE、微分方程等字眼，天然就觉得“恐慌”，但本质上来说，SDE只是个“幌子”，实际上将对SDE的理解转换到式$\eqref{eq:sde-discrete}$和式$\eqref{eq:sde-proba}$上后，完全就可以抛开SDE的概念了，因此概念上其实是没有太大难度的。

不难发现，定义一个随机微分方程$\eqref{eq:sde-forward}$是很容易的，但是从$\eqref{eq:sde-forward}$求解$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$却是不容易的。原论文的剩余篇幅，主要是对两个有实用性的例子推导和实验。然而，既然求解$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$不容易，那么按照笔者的看法，与其先定义$\eqref{eq:sde-forward}$再求解$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$，倒不如像[DDIM](/archives/9181)一样，先定义$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$，然后再来反推对应的SDE？

例如，我们先定义  
\begin{equation} p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_t; \bar{\alpha}_t \boldsymbol{x}_0,\bar{\beta}_t^2 \boldsymbol{I})\end{equation}  
并且不失一般性假设起点是$t=0$，终点是$t=1$，那么$\bar{\alpha}_t,\bar{\beta}_t$要满足的边界就是  
\begin{equation} \bar{\alpha}_0 = 1,\quad \bar{\alpha}_1 = 0,\quad \bar{\beta}_0 = 0,\quad \bar{\beta}_1 = 1\end{equation}  
当然，上述边界条件理论上足够近似就行，也不一定非要精确相等，比如上一篇文章我们分析过DDPM相当于选择了$\bar{\alpha}_t = e^{-5t^2}$，当$t=1$时结果为$e^{-5}\approx 0$。

有了$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$，我们去反推$\eqref{eq:sde-forward}$，本质上就是要求解$p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t)$，它要满足  
\begin{equation} p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_0) = \int p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) p(\boldsymbol{x}_t|\boldsymbol{x}_0) d\boldsymbol{x}_t\end{equation}  
我们考虑线性的解，即  
\begin{equation}d\boldsymbol{x} = f_t\boldsymbol{x} dt + g_t d\boldsymbol{w}\end{equation}  
跟[《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](/archives/9181#%E5%BE%85%E5%AE%9A%E7%B3%BB%E6%95%B0)一样，我们写出  
\begin{array}{c|c|c}  
\hline  
\text{记号} & \text{含义} & \text{采样}\\\  
\hline  
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_t;\bar{\alpha}_{t+\Delta t} \boldsymbol{x}_0,\bar{\beta}_{t+\Delta t}^2 \boldsymbol{I}) & \boldsymbol{x}_{t+\Delta t} = \bar{\alpha}_{t+\Delta t} \boldsymbol{x}_0 + \bar{\beta}_{t+\Delta t} \boldsymbol{\varepsilon} \\\  
\hline  
p(\boldsymbol{x}_t|\boldsymbol{x}_0) & \mathcal{N}(\boldsymbol{x}_t;\bar{\alpha}_t \boldsymbol{x}_0,\bar{\beta}_t^2 \boldsymbol{I}) & \boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}_1 \\\  
\hline  
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) & \mathcal{N}(\boldsymbol{x}_{t+\Delta t}; (1 + f_t\Delta t) \boldsymbol{x}_t, g_t^2 \Delta t\, \boldsymbol{I}) & \boldsymbol{x}_{t+\Delta t} = (1 + f_t\Delta t) \boldsymbol{x}_t + g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}_2 \\\  
\hline  
{\begin{array}{c}\int p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) \\\  
p(\boldsymbol{x}_t|\boldsymbol{x}_0) d\boldsymbol{x}_t\end{array}} & & {\begin{aligned}&\,\boldsymbol{x}_{t+\Delta t} \\\  
=&\, (1 + f_t\Delta t) \boldsymbol{x}_t + g_t\sqrt{\Delta t} \boldsymbol{\varepsilon}_2 \\\  
=&\, (1 + f_t\Delta t) (\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}_1) + g_t\sqrt{\Delta t} \boldsymbol{\varepsilon}_2 \\\  
=&\, (1 + f_t\Delta t) \bar{\alpha}_t \boldsymbol{x}_0 + ((1 + f_t\Delta t)\bar{\beta}_t \boldsymbol{\varepsilon}_1 + g_t\sqrt{\Delta t} \boldsymbol{\varepsilon}_2) \\\  
\end{aligned}} \\\  
\hline  
\end{array}  
由此可得  
\begin{equation}\begin{aligned}  
\bar{\alpha}_{t+\Delta t} =&\, (1 + f_t\Delta t) \bar{\alpha}_t \\\  
\bar{\beta}_{t+\Delta t}^2 =&\, (1 + f_t\Delta t)^2\bar{\beta}_t^2 + g_t^2\Delta t  
\end{aligned}\end{equation}  
令$\Delta t\to 0$，分别解得  
\begin{equation}  
f_t = \frac{d}{dt} \left(\ln \bar{\alpha}_t\right) = \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt}, \quad g_t^2 = \bar{\alpha}_t^2 \frac{d}{dt}\left(\frac{\bar{\beta}_t^2}{\bar{\alpha}_t^2}\right) = 2\bar{\alpha}_t \bar{\beta}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\end{equation}  
取$\bar{\alpha}_t\equiv 1$时，结果就是论文中的VE-SDE（Variance Exploding SDE）；而如果取$\bar{\alpha}_t^2 + \bar{\beta}_t^2=1$时，结果就是原论文中的VP-SDE（Variance Preserving SDE）。

至于损失函数，此时我们可以算得  
\begin{equation}\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0) = -\frac{\boldsymbol{x}_t - \bar{\alpha}_t\boldsymbol{x}_0}{\bar{\beta}_t^2}=-\frac{\boldsymbol{\varepsilon}}{\bar{\beta}_t}\end{equation}  
第二个等号是因为$\boldsymbol{x}_t = \bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}$，为了跟以往的结果对齐，我们设$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\bar{\beta}_t}$，此时式$\eqref{eq:score-match}$为  
\begin{equation}\frac{1}{\bar{\beta}_t^2}\mathbb{E}_{\boldsymbol{x}_0\sim \tilde{p}(\boldsymbol{x}_0),\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\left[\left\Vert \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t) - \boldsymbol{\varepsilon}\right\Vert^2\right]\end{equation}  
忽略系数后就是DDPM的损失函数，而用$-\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t+\Delta t}, t+\Delta t)}{\bar{\beta}_{t+\Delta t}}$替换掉式$\eqref{eq:reverse-sde-discrete}$的$\nabla_{\boldsymbol{x}_{t+\Delta t}}\log p(\boldsymbol{x}_{t+\Delta t})$后，结果与DDPM的采样过程具有相同的一阶近似（意味着$\Delta t\to 0$时两者等价）。

## 文章小结 #

本文主要介绍了宋飏博士建立的利用SDE理解扩散模型的一般框架，其中包括以尽可能直观的语言推导了反向SDE、得分匹配等结果，并对方程的求解给出了自己的想法。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9209>_

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

苏剑林. (Aug. 03, 2022). 《生成扩散模型漫谈（五）：一般框架之SDE篇 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9209>

@online{kexuefm-9209,  
title={生成扩散模型漫谈（五）：一般框架之SDE篇},  
author={苏剑林},  
year={2022},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/9209}},  
} 


---

## 公式推导与注释

本节提供文章中涉及的随机微分方程（SDE）框架的极详细数学推导，包括Itô积分、Itô引理、前向SDE、逆向SDE、Anderson定理、Fokker-Planck方程、概率流ODE等核心内容。

### 1. 随机微分方程基础理论

#### 1.1 Brown运动与随机过程

**定义（标准Brown运动）**：随机过程 $\{\boldsymbol{w}_t\}_{t\geq 0}$ 称为标准Brown运动（或Wiener过程），如果满足：

1. $\boldsymbol{w}_0 = \boldsymbol{0}$（几乎处处）
2. 对于任意 $0 \leq s < t$，增量 $\boldsymbol{w}_t - \boldsymbol{w}_s \sim \mathcal{N}(\boldsymbol{0}, (t-s)\boldsymbol{I})$
3. 对于任意不相交的时间区间，增量相互独立
4. $\boldsymbol{w}_t$ 关于 $t$ 几乎处处连续

**性质1（二次变差）**：Brown运动的重要性质是其二次变差。对于分割 $0 = t_0 < t_1 < \cdots < t_n = t$，当 $\max_i(t_{i+1} - t_i) \to 0$ 时，我们有：

$$
\sum_{i=0}^{n-1} (\boldsymbol{w}_{t_{i+1}} - \boldsymbol{w}_{t_i})(\boldsymbol{w}_{t_{i+1}} - \boldsymbol{w}_{t_i})^T \to t\boldsymbol{I} \quad \text{（依概率）}
$$

这意味着在形式上，我们可以写 $(d\boldsymbol{w})^2 = dt \cdot \boldsymbol{I}$，这是Itô积分的核心性质。

**性质2（无处可微）**：虽然Brown运动轨道连续，但它几乎处处不可微。这可以从二次变差看出：如果 $\boldsymbol{w}_t$ 可微，那么二次变差应该趋于0，但实际上它趋于 $t$。

#### 1.2 Itô积分的定义

对于Brown运动 $\boldsymbol{w}_t$，我们希望定义形如 $\int_0^t \boldsymbol{h}_s d\boldsymbol{w}_s$ 的积分。由于 $\boldsymbol{w}_t$ 不可微，这不能用常规的Riemann-Stieltjes积分定义。

**构造（Itô积分）**：对于简单过程 $\boldsymbol{h}_t = \sum_{i=0}^{n-1} \boldsymbol{\xi}_i \mathbb{1}_{[t_i, t_{i+1})}(t)$，其中 $\boldsymbol{\xi}_i$ 是关于 $\mathcal{F}_{t_i}$ 可测的（$\mathcal{F}_t$ 是由 $\{\boldsymbol{w}_s\}_{s\leq t}$ 生成的 $\sigma$-代数），定义：

$$
\int_0^t \boldsymbol{h}_s d\boldsymbol{w}_s := \sum_{i=0}^{n-1} \boldsymbol{\xi}_i (\boldsymbol{w}_{t_{i+1}\wedge t} - \boldsymbol{w}_{t_i \wedge t})
$$

其中 $a \wedge b = \min(a,b)$。

然后通过 $L^2$ 极限将定义扩展到所有满足 $\mathbb{E}[\int_0^T \|\boldsymbol{h}_s\|^2 ds] < \infty$ 的适应过程。

**Itô等距性质**：对于满足条件的适应过程 $\boldsymbol{h}_t$，有：

$$
\mathbb{E}\left[\left\|\int_0^t \boldsymbol{h}_s d\boldsymbol{w}_s\right\|^2\right] = \mathbb{E}\left[\int_0^t \|\boldsymbol{h}_s\|^2 ds\right]
$$

这是Itô积分的核心性质，说明Itô积分是一个鞅（martingale）。

#### 1.3 Itô引理（Itô公式）

**定理（一维Itô引理）**：设 $x_t$ 满足SDE：

$$
dx_t = \mu_t dt + \sigma_t dw_t
$$

且 $f(x,t)$ 是 $C^{1,2}$ 函数（对 $t$ 一阶连续可微，对 $x$ 二阶连续可微），则 $y_t = f(x_t, t)$ 满足：

$$
dy_t = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right)dt + \sigma_t \frac{\partial f}{\partial x} dw_t
$$

**详细推导**：

1. 对 $f(x_t, t)$ 做Taylor展开（到二阶）：

$$
\begin{aligned}
df &= f(x_{t+dt}, t+dt) - f(x_t, t) \\
&= \frac{\partial f}{\partial t}dt + \frac{\partial f}{\partial x}dx + \frac{1}{2}\frac{\partial^2 f}{\partial x^2}(dx)^2 + \frac{\partial^2 f}{\partial x \partial t}dx\,dt + \frac{1}{2}\frac{\partial^2 f}{\partial t^2}(dt)^2 + O(dt^{3/2})
\end{aligned}
$$

2. 计算 $(dx)^2$：

$$
\begin{aligned}
(dx)^2 &= (\mu_t dt + \sigma_t dw_t)^2 \\
&= \mu_t^2 (dt)^2 + 2\mu_t \sigma_t dt\,dw_t + \sigma_t^2 (dw_t)^2
\end{aligned}
$$

3. 应用随机微积分规则：
   - $(dt)^2 = 0$（高阶无穷小）
   - $dt \cdot dw_t = 0$（高阶无穷小）
   - $(dw_t)^2 = dt$（Brown运动的二次变差性质）

4. 因此：

$$
(dx)^2 = \sigma_t^2 dt
$$

5. 代入得：

$$
df = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right)dt + \sigma_t \frac{\partial f}{\partial x} dw_t
$$

**多维Itô引理**：对于 $d$-维过程 $\boldsymbol{x}_t$ 满足：

$$
d\boldsymbol{x}_t = \boldsymbol{\mu}_t dt + \boldsymbol{\Sigma}_t d\boldsymbol{w}_t
$$

其中 $\boldsymbol{\mu}_t \in \mathbb{R}^d$，$\boldsymbol{\Sigma}_t \in \mathbb{R}^{d \times d}$，$\boldsymbol{w}_t$ 是 $d$-维Brown运动。若 $f: \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}$ 是 $C^{1,2}$ 函数，则：

$$
df(\boldsymbol{x}_t, t) = \left(\frac{\partial f}{\partial t} + \sum_{i=1}^d \mu_{t,i} \frac{\partial f}{\partial x_i} + \frac{1}{2}\sum_{i,j=1}^d (\boldsymbol{\Sigma}_t \boldsymbol{\Sigma}_t^T)_{ij} \frac{\partial^2 f}{\partial x_i \partial x_j}\right)dt + \sum_{i=1}^d \sum_{k=1}^d (\boldsymbol{\Sigma}_t)_{ik} \frac{\partial f}{\partial x_i} dw_{t,k}
$$

### 2. 前向SDE的详细分析

#### 2.1 前向扩散过程

文章中定义的前向SDE为：

$$
d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}
$$

这里：
- $\boldsymbol{f}_t(\boldsymbol{x})$：漂移系数（drift coefficient），描述确定性变化
- $g_t$：扩散系数（diffusion coefficient），描述随机性强度
- $\boldsymbol{w}_t$：标准Brown运动

**离散化（Euler-Maruyama方法）**：最简单的数值方法是Euler-Maruyama离散化：

$$
\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t} \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})
$$

注意 $\sqrt{\Delta t}$ 来自于Brown运动增量的方差：$\boldsymbol{w}_{t+\Delta t} - \boldsymbol{w}_t \sim \mathcal{N}(\boldsymbol{0}, \Delta t \boldsymbol{I})$。

#### 2.2 转移概率密度

给定初始条件 $\boldsymbol{x}_t = \boldsymbol{x}$，在 $\Delta t$ 时间后的条件分布为：

$$
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t = \boldsymbol{x}) \approx \mathcal{N}(\boldsymbol{x} + \boldsymbol{f}_t(\boldsymbol{x})\Delta t, g_t^2 \Delta t \boldsymbol{I})
$$

概率密度函数：

$$
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) = \frac{1}{(2\pi g_t^2 \Delta t)^{d/2}} \exp\left(-\frac{\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\|^2}{2g_t^2 \Delta t}\right)
$$

其中 $d$ 是向量维度。

### 3. 逆向SDE的详细推导

#### 3.1 时间反转问题

给定前向SDE：

$$
d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}
$$

我们希望找到逆向过程的SDE，即从 $t=T$ 到 $t=0$ 的演化。

#### 3.2 Anderson定理

**定理（Anderson, 1982）**：如果前向过程满足：

$$
d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}
$$

那么逆向过程（从 $t=T$ 反向到 $t=0$）满足：

$$
d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})\right] dt + g_t d\bar{\boldsymbol{w}}
$$

其中 $\bar{\boldsymbol{w}}_t$ 是逆向时间的Brown运动，$p_t(\boldsymbol{x})$ 是边缘分布密度。

**详细推导**：

**步骤1**：建立贝叶斯关系

前向转移概率：

$$
p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) \propto \exp\left(-\frac{\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\|^2}{2g_t^2\Delta t}\right)
$$

逆向转移概率（贝叶斯定理）：

$$
p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) = \frac{p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) p(\boldsymbol{x}_t)}{p(\boldsymbol{x}_{t+\Delta t})}
$$

取对数：

$$
\log p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) = \log p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t) + \log p(\boldsymbol{x}_t) - \log p(\boldsymbol{x}_{t+\Delta t})
$$

**步骤2**：Taylor展开

对 $\log p(\boldsymbol{x}_{t+\Delta t})$ 在 $\boldsymbol{x}_t$ 附近展开：

$$
\begin{aligned}
\log p(\boldsymbol{x}_{t+\Delta t}) &\approx \log p(\boldsymbol{x}_t) + (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t) \\
&\quad + \frac{1}{2}(\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \nabla_{\boldsymbol{x}_t}^2 \log p(\boldsymbol{x}_t) (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t) \\
&\quad + \Delta t \frac{\partial}{\partial t}\log p(\boldsymbol{x}_t)
\end{aligned}
$$

注意这里包含时间导数项，因为 $p(\boldsymbol{x}_{t+\Delta t})$ 实际上是 $p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})$。

**步骤3**：简化（仅保留一阶项）

当 $\Delta t \to 0$ 时，$\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = O(\sqrt{\Delta t})$（因为随机项），因此：

- 线性项：$O(\sqrt{\Delta t})$
- 二次项：$O(\Delta t)$（与 $\Delta t$ 项同阶，但系数较小）
- 时间导数项：$O(\Delta t)$

主要的一阶近似：

$$
\log p(\boldsymbol{x}_{t+\Delta t}) \approx \log p(\boldsymbol{x}_t) + (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t) + \Delta t \frac{\partial}{\partial t}\log p(\boldsymbol{x}_t)
$$

**步骤4**：代入前向转移概率

$$
\begin{aligned}
&\log p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) \\
&= -\frac{\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\|^2}{2g_t^2\Delta t} + \log p(\boldsymbol{x}_t) - \log p(\boldsymbol{x}_{t+\Delta t}) + C \\
&\approx -\frac{\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\|^2}{2g_t^2\Delta t} - (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t) + C'
\end{aligned}
$$

**步骤5**：配方

展开二次项：

$$
\begin{aligned}
&-\frac{\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\|^2}{2g_t^2\Delta t} - (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t) \\
&= -\frac{1}{2g_t^2\Delta t}\left[\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t\|^2 - 2(\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t + \|\boldsymbol{f}_t(\boldsymbol{x}_t)\|^2 (\Delta t)^2\right] \\
&\quad - (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t) \\
&= -\frac{1}{2g_t^2\Delta t}\left[\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t\|^2 - 2(\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)^T (\boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t + g_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t))\right] + O(\Delta t)
\end{aligned}
$$

配方得：

$$
= -\frac{\|\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t - (\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t))\Delta t\|^2}{2g_t^2\Delta t} + O(\Delta t)
$$

**步骤6**：得到逆向SDE

因此，$p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t})$ 近似为正态分布：

$$
p(\boldsymbol{x}_t|\boldsymbol{x}_{t+\Delta t}) \approx \mathcal{N}\left(\boldsymbol{x}_{t+\Delta t} - [\boldsymbol{f}_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - g_{t+\Delta t}^2 \nabla_{\boldsymbol{x}} \log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})]\Delta t, g_{t+\Delta t}^2 \Delta t \boldsymbol{I}\right)
$$

对应的逆向SDE（时间反向）：

$$
d\boldsymbol{x} = [\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})] dt + g_t d\bar{\boldsymbol{w}}
$$

### 4. Score函数的定义和性质

#### 4.1 Score函数定义

**定义**：给定概率分布 $p(\boldsymbol{x})$，其score函数定义为：

$$
\boldsymbol{s}(\boldsymbol{x}) := \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x})
$$

Score函数指向概率密度增长最快的方向。

**性质1（零均值）**：对于任意概率分布，其score函数的期望为零：

$$
\mathbb{E}_{\boldsymbol{x} \sim p}[\boldsymbol{s}(\boldsymbol{x})] = \int \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}) \cdot p(\boldsymbol{x}) d\boldsymbol{x} = \int \nabla_{\boldsymbol{x}} p(\boldsymbol{x}) d\boldsymbol{x} = \nabla_{\boldsymbol{x}} \int p(\boldsymbol{x}) d\boldsymbol{x} = \nabla_{\boldsymbol{x}} 1 = \boldsymbol{0}
$$

（假设可以交换积分和微分的顺序）

**性质2（条件score）**：对于条件分布 $p(\boldsymbol{x}_t|\boldsymbol{x}_0)$，条件score为：

$$
\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)
$$

对于高斯分布 $p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{\mu}_t(\boldsymbol{x}_0), \boldsymbol{\Sigma}_t)$：

$$
\log p(\boldsymbol{x}_t|\boldsymbol{x}_0) = -\frac{1}{2}(\boldsymbol{x}_t - \boldsymbol{\mu}_t)^T \boldsymbol{\Sigma}_t^{-1} (\boldsymbol{x}_t - \boldsymbol{\mu}_t) + C
$$

因此：

$$
\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0) = -\boldsymbol{\Sigma}_t^{-1}(\boldsymbol{x}_t - \boldsymbol{\mu}_t)
$$

对于DDPM中的 $p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\bar{\alpha}_t \boldsymbol{x}_0, \bar{\beta}_t^2 \boldsymbol{I})$：

$$
\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0) = -\frac{\boldsymbol{x}_t - \bar{\alpha}_t \boldsymbol{x}_0}{\bar{\beta}_t^2} = -\frac{\boldsymbol{\varepsilon}}{\bar{\beta}_t}
$$

其中 $\boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}$，$\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

#### 4.2 边缘score

边缘分布的score：

$$
\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) = \nabla_{\boldsymbol{x}_t} \log \int p(\boldsymbol{x}_t|\boldsymbol{x}_0) p(\boldsymbol{x}_0) d\boldsymbol{x}_0
$$

利用 $\nabla \log f = \frac{\nabla f}{f}$：

$$
\begin{aligned}
\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) &= \frac{\int \nabla_{\boldsymbol{x}_t} p(\boldsymbol{x}_t|\boldsymbol{x}_0) p(\boldsymbol{x}_0) d\boldsymbol{x}_0}{\int p(\boldsymbol{x}_t|\boldsymbol{x}_0) p(\boldsymbol{x}_0) d\boldsymbol{x}_0} \\
&= \frac{\int p(\boldsymbol{x}_t|\boldsymbol{x}_0) \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0) p(\boldsymbol{x}_0) d\boldsymbol{x}_0}{\int p(\boldsymbol{x}_t|\boldsymbol{x}_0) p(\boldsymbol{x}_0) d\boldsymbol{x}_0} \\
&= \mathbb{E}_{\boldsymbol{x}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}[\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)]
\end{aligned}
$$

这表明边缘score是条件score关于后验 $p(\boldsymbol{x}_0|\boldsymbol{x}_t)$ 的期望。

### 5. Fokker-Planck方程

#### 5.1 前向Fokker-Planck方程

**定理**：如果 $\boldsymbol{x}_t$ 满足SDE：

$$
d\boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) dt + g_t d\boldsymbol{w}_t
$$

那么其概率密度 $p_t(\boldsymbol{x})$ 满足Fokker-Planck方程（也称为Kolmogorov前向方程）：

$$
\frac{\partial p_t(\boldsymbol{x})}{\partial t} = -\nabla \cdot (\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})) + \frac{g_t^2}{2} \Delta p_t(\boldsymbol{x})
$$

其中 $\nabla \cdot$ 是散度算子，$\Delta = \nabla^2$ 是拉普拉斯算子。

**详细推导**：

**步骤1**：Master方程

从Chapman-Kolmogorov方程出发：

$$
p_t(\boldsymbol{x}) = \int p(\boldsymbol{x}|\boldsymbol{y}) p_{t-\Delta t}(\boldsymbol{y}) d\boldsymbol{y}
$$

其中 $p(\boldsymbol{x}|\boldsymbol{y})$ 是从 $\boldsymbol{y}$ 在时间 $\Delta t$ 内转移到 $\boldsymbol{x}$ 的概率密度。

**步骤2**：Kramers-Moyal展开

对于SDE，转移概率在 $\Delta t \to 0$ 时可以展开：

$$
p(\boldsymbol{x}|\boldsymbol{y}) = \delta(\boldsymbol{x} - \boldsymbol{y} - \boldsymbol{f}_t(\boldsymbol{y})\Delta t) * \mathcal{N}(\boldsymbol{0}, g_t^2 \Delta t \boldsymbol{I})
$$

其中 $*$ 表示卷积。

**步骤3**：对 $p_t$ 做时间差分

$$
\frac{p_t(\boldsymbol{x}) - p_{t-\Delta t}(\boldsymbol{x})}{\Delta t} = \frac{1}{\Delta t}\int [p(\boldsymbol{x}|\boldsymbol{y}) - \delta(\boldsymbol{x} - \boldsymbol{y})] p_{t-\Delta t}(\boldsymbol{y}) d\boldsymbol{y}
$$

**步骤4**：Taylor展开

对 $p_{t-\Delta t}(\boldsymbol{y})$ 在 $\boldsymbol{y} = \boldsymbol{x}$ 附近展开：

$$
p_{t-\Delta t}(\boldsymbol{y}) = p_{t-\Delta t}(\boldsymbol{x}) - (\boldsymbol{y} - \boldsymbol{x})^T \nabla p_{t-\Delta t}(\boldsymbol{x}) + \frac{1}{2}(\boldsymbol{y} - \boldsymbol{x})^T \nabla^2 p_{t-\Delta t}(\boldsymbol{x}) (\boldsymbol{y} - \boldsymbol{x}) + \cdots
$$

**步骤5**：计算矩

对于转移概率，我们需要计算：

$$
\begin{aligned}
\text{一阶矩：} &\quad \mathbb{E}[\boldsymbol{x} - \boldsymbol{y}|\boldsymbol{y}] = \boldsymbol{f}_t(\boldsymbol{y}) \Delta t \\
\text{二阶矩：} &\quad \mathbb{E}[(\boldsymbol{x} - \boldsymbol{y})(\boldsymbol{x} - \boldsymbol{y})^T|\boldsymbol{y}] = g_t^2 \Delta t \boldsymbol{I} + O((\Delta t)^2)
\end{aligned}
$$

**步骤6**：代入并取极限

$$
\begin{aligned}
\frac{\partial p_t}{\partial t} &= -\nabla \cdot (\boldsymbol{f}_t p_t) + \frac{1}{2} \text{Tr}(g_t^2 \boldsymbol{I} \nabla^2 p_t) \\
&= -\nabla \cdot (\boldsymbol{f}_t p_t) + \frac{g_t^2}{2} \Delta p_t
\end{aligned}
$$

#### 5.2 逆向Fokker-Planck方程

对于逆向SDE：

$$
d\boldsymbol{x} = [\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})] dt + g_t d\bar{\boldsymbol{w}}
$$

其漂移项为 $\tilde{\boldsymbol{f}}_t(\boldsymbol{x}) = \boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})$，因此时间反向的Fokker-Planck方程为：

$$
-\frac{\partial p_t(\boldsymbol{x})}{\partial t} = -\nabla \cdot ([\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})] p_t(\boldsymbol{x})) + \frac{g_t^2}{2} \Delta p_t(\boldsymbol{x})
$$

展开右边：

$$
\begin{aligned}
&-\nabla \cdot (\boldsymbol{f}_t p_t) + \nabla \cdot (g_t^2 (\nabla p_t)) + \frac{g_t^2}{2} \Delta p_t \\
&= -\nabla \cdot (\boldsymbol{f}_t p_t) + g_t^2 \Delta p_t + \frac{g_t^2}{2} \Delta p_t \\
&= -\nabla \cdot (\boldsymbol{f}_t p_t) + \frac{g_t^2}{2} \Delta p_t
\end{aligned}
$$

这恰好抵消，说明逆向SDE确实使 $p_t$ 沿时间反向演化回初始分布。

### 6. 概率流ODE

#### 6.1 从SDE到ODE

**定理**：对于前向SDE：

$$
d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}
$$

存在一个确定性的ODE（常微分方程），称为概率流ODE：

$$
d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})\right] dt
$$

它具有与原SDE相同的边缘分布 $p_t(\boldsymbol{x})$。

**推导思路**：两个过程有相同边缘分布，意味着它们满足相同的Fokker-Planck方程。对于ODE：

$$
d\boldsymbol{x} = \boldsymbol{u}_t(\boldsymbol{x}) dt
$$

其Fokker-Planck方程为（无扩散项）：

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (\boldsymbol{u}_t p_t)
$$

要与原SDE的Fokker-Planck方程相同：

$$
-\nabla \cdot (\boldsymbol{u}_t p_t) = -\nabla \cdot (\boldsymbol{f}_t p_t) + \frac{g_t^2}{2} \Delta p_t
$$

展开右边的拉普拉斯项：

$$
\Delta p_t = \nabla \cdot (\nabla p_t) = \nabla \cdot (p_t \nabla \log p_t)
$$

因此：

$$
-\nabla \cdot (\boldsymbol{u}_t p_t) = -\nabla \cdot (\boldsymbol{f}_t p_t) + \frac{g_t^2}{2} \nabla \cdot (p_t \nabla \log p_t)
$$

消去散度算子（在适当条件下）：

$$
\boldsymbol{u}_t = \boldsymbol{f}_t - \frac{g_t^2}{2} \nabla \log p_t
$$

#### 6.2 逆向概率流ODE

类似地，逆向过程也可以写成ODE形式：

$$
d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}) + \frac{g_t^2}{2} \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})\right] dt = \left[\boldsymbol{f}_t(\boldsymbol{x}) - \frac{g_t^2}{2} \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})\right] dt
$$

这与前向概率流ODE（时间反向）是一致的。

**优势**：ODE采样的优势在于：
1. 确定性：给定初始噪声，生成结果唯一
2. 精确重构：可以从数据编码到潜空间再解码回来
3. 插值：可以在潜空间进行有意义的插值

### 7. 与离散DDPM的连续极限关系

#### 7.1 DDPM的离散形式

DDPM定义：

$$
\boldsymbol{x}_t = \sqrt{\alpha_t} \boldsymbol{x}_{t-1} + \sqrt{1 - \alpha_t} \boldsymbol{\varepsilon}_t
$$

累积形式：

$$
\boldsymbol{x}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\varepsilon}
$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

#### 7.2 连续极限

设总时长为1，步数为 $N$，则 $\Delta t = 1/N$，$t_k = k\Delta t$。

定义连续函数 $\bar{\alpha}(t)$ 使得 $\bar{\alpha}(t_k) = \bar{\alpha}_k$。

对数微分：

$$
\frac{d \log \bar{\alpha}(t)}{dt} = \lim_{\Delta t \to 0} \frac{\log \bar{\alpha}_{t+\Delta t} - \log \bar{\alpha}_t}{\Delta t} = \lim_{\Delta t \to 0} \frac{\log \alpha_{t+\Delta t}}{\Delta t}
$$

对于DDPM，$\alpha_t = 1 - \beta_t$，$\beta_t$ 很小，因此：

$$
\log \alpha_t = \log(1 - \beta_t) \approx -\beta_t
$$

定义 $\beta(t)$ 使得 $\beta(t_k) = \beta_k$，则：

$$
\frac{d \log \bar{\alpha}(t)}{dt} \approx -\frac{\beta(t)}{\Delta t} \cdot \Delta t = -\beta(t)
$$

因此：

$$
\bar{\alpha}(t) = \exp\left(-\int_0^t \beta(s) ds\right)
$$

#### 7.3 从离散到连续SDE

离散步骤：

$$
\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = (\sqrt{\alpha_{t+\Delta t}} - 1)\boldsymbol{x}_t + \sqrt{1 - \alpha_{t+\Delta t}} \boldsymbol{\varepsilon}
$$

连续极限：

$$
\sqrt{\alpha_{t+\Delta t}} = \sqrt{1 - \beta_{t+\Delta t}} \approx 1 - \frac{\beta_{t+\Delta t}}{2}
$$

因此：

$$
\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t \approx -\frac{\beta_t}{2}\boldsymbol{x}_t \Delta t + \sqrt{\beta_t \Delta t} \boldsymbol{\varepsilon}
$$

对应SDE：

$$
d\boldsymbol{x} = -\frac{1}{2}\beta_t \boldsymbol{x} dt + \sqrt{\beta_t} d\boldsymbol{w}
$$

这正是VP-SDE（Variance Preserving SDE）的形式，其中：
- $\boldsymbol{f}_t(\boldsymbol{x}) = -\frac{1}{2}\beta_t \boldsymbol{x}$
- $g_t = \sqrt{\beta_t}$

### 8. 采样算法

#### 8.1 Euler-Maruyama方法

对于逆向SDE：

$$
d\boldsymbol{x} = [\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})] dt + g_t d\bar{\boldsymbol{w}}
$$

Euler-Maruyama离散化：

$$
\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t - [\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)] \Delta t + g_t \sqrt{\Delta t} \boldsymbol{z}_t
$$

其中 $\boldsymbol{z}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，$\boldsymbol{s}_{\boldsymbol{\theta}}$ 是训练好的score网络。

**算法（逆向采样）**：

```
输入：score网络 s_θ, 初始噪声 x_T ~ N(0, I), 时间步长 Δt
输出：生成样本 x_0

for t = T, T-Δt, ..., Δt:
    z ~ N(0, I)  # 当 t > Δt 时
    如果 t = Δt: z = 0  # 最后一步不加噪声

    # 计算漂移和扩散
    drift = f_t(x_t) - g_t^2 * s_θ(x_t, t)
    diffusion = g_t * sqrt(Δt) * z

    # 更新
    x_{t-Δt} = x_t - drift * Δt + diffusion

返回 x_0
```

#### 8.2 预测器-校正器采样（Predictor-Corrector）

为了提高采样质量，可以结合预测步骤和Langevin动力学校正步骤：

**预测步骤（Euler-Maruyama）**：

$$
\tilde{\boldsymbol{x}}_{t-\Delta t} = \boldsymbol{x}_t - [\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)] \Delta t + g_t \sqrt{\Delta t} \boldsymbol{z}_t
$$

**校正步骤（Langevin MCMC）**：对于 $M$ 步：

$$
\boldsymbol{x}^{(m+1)} = \boldsymbol{x}^{(m)} + \epsilon \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}^{(m)}, t) + \sqrt{2\epsilon} \boldsymbol{z}^{(m)}
$$

其中 $\boldsymbol{x}^{(0)} = \tilde{\boldsymbol{x}}_{t-\Delta t}$，最终 $\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}^{(M)}$。

#### 8.3 ODE求解器

对于概率流ODE：

$$
d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})\right] dt
$$

可以使用高阶ODE求解器，如Runge-Kutta方法：

**RK4（四阶Runge-Kutta）**：

$$
\begin{aligned}
\boldsymbol{k}_1 &= \boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \\
\boldsymbol{k}_2 &= \boldsymbol{f}_{t-\Delta t/2}(\boldsymbol{x}_t - \frac{\Delta t}{2}\boldsymbol{k}_1) - \frac{1}{2}g_{t-\Delta t/2}^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t - \frac{\Delta t}{2}\boldsymbol{k}_1, t-\frac{\Delta t}{2}) \\
\boldsymbol{k}_3 &= \boldsymbol{f}_{t-\Delta t/2}(\boldsymbol{x}_t - \frac{\Delta t}{2}\boldsymbol{k}_2) - \frac{1}{2}g_{t-\Delta t/2}^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t - \frac{\Delta t}{2}\boldsymbol{k}_2, t-\frac{\Delta t}{2}) \\
\boldsymbol{k}_4 &= \boldsymbol{f}_{t-\Delta t}(\boldsymbol{x}_t - \Delta t \boldsymbol{k}_3) - \frac{1}{2}g_{t-\Delta t}^2 \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t - \Delta t \boldsymbol{k}_3, t-\Delta t) \\
\boldsymbol{x}_{t-\Delta t} &= \boldsymbol{x}_t - \frac{\Delta t}{6}(\boldsymbol{k}_1 + 2\boldsymbol{k}_2 + 2\boldsymbol{k}_3 + \boldsymbol{k}_4)
\end{aligned}
$$

ODE求解通常比SDE采样需要更少的步数即可获得高质量样本。

### 9. 训练过程的完整推导

#### 9.1 Score匹配目标

目标是学习 $\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \approx \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$。

完整的denoising score matching损失：

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{t \sim \mathcal{U}[0,T]} \mathbb{E}_{\boldsymbol{x}_0 \sim p_0} \mathbb{E}_{\boldsymbol{x}_t \sim p(\boldsymbol{x}_t|\boldsymbol{x}_0)} \left[\lambda(t) \|\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\|^2\right]
$$

其中 $\lambda(t)$ 是权重函数。

#### 9.2 对于线性SDE的简化

对于 $p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}(\bar{\alpha}_t \boldsymbol{x}_0, \bar{\beta}_t^2 \boldsymbol{I})$，采样 $\boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}$，$\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

条件score：

$$
\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0) = -\frac{\boldsymbol{x}_t - \bar{\alpha}_t \boldsymbol{x}_0}{\bar{\beta}_t^2} = -\frac{\boldsymbol{\varepsilon}}{\bar{\beta}_t}
$$

如果参数化为 $\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\bar{\beta}_t}$，则损失变为：

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{\varepsilon}} \left[\frac{\lambda(t)}{\bar{\beta}_t^2} \|\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, t) - \boldsymbol{\varepsilon}\|^2\right]
$$

选择 $\lambda(t) = \bar{\beta}_t^2$ 得到DDPM损失：

$$
\mathcal{L}_{\text{simple}}(\boldsymbol{\theta}) = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{\varepsilon}} \left[\|\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{\varepsilon}\|^2\right]
$$

### 10. 总结与统一视角

#### 10.1 三种等价表示

对于扩散模型，有三种等价的表示方式：

1. **噪声预测**：$\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \approx \boldsymbol{\varepsilon}$
2. **Score预测**：$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \approx \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$
3. **数据预测**：$\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) \approx \mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t]$

它们之间的关系：

$$
\boldsymbol{s}_{\boldsymbol{\theta}} = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}}{\bar{\beta}_t} = -\frac{\boldsymbol{x}_t - \bar{\alpha}_t \hat{\boldsymbol{x}}_{\boldsymbol{\theta}}}{\bar{\beta}_t^2}
$$

$$
\hat{\boldsymbol{x}}_{\boldsymbol{\theta}} = \frac{\boldsymbol{x}_t - \bar{\beta}_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}}{\bar{\alpha}_t}
$$

#### 10.2 连续时间框架的优势

SDE框架的主要优势：

1. **理论统一**：将DDPM、score matching、去噪等统一在同一框架下
2. **灵活采样**：可以选择任意步数，不受训练时步数限制
3. **多种求解器**：可以使用SDE求解器或ODE求解器
4. **精确似然计算**：通过ODE可以精确计算似然（使用连续归一化流）
5. **条件生成**：容易扩展到条件生成和引导生成

这个统一的SDE视角为扩散模型提供了坚实的数学基础，使得我们能够更深入地理解和改进这类生成模型。

