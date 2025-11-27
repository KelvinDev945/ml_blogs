---
title: 生成扩散模型漫谈（十五）：构建ODE的一般步骤（中）
slug: 生成扩散模型漫谈十五构建ode的一般步骤中
date: 2022-12-22
tags: 详细推导, 微分方程, 生成模型, 扩散, 格林函数, 生成模型
status: completed
---
# 生成扩散模型漫谈（十五）：构建ODE的一般步骤（中）

**原文链接**: [https://spaces.ac.cn/archives/9379](https://spaces.ac.cn/archives/9379)

**发布日期**: 

---

上周笔者写了[《生成扩散模型漫谈（十四）：构建ODE的一般步骤（上）》](/archives/9370)（当时还没有“上”这个后缀），本以为已经窥见了构建ODE扩散模型的一般规律，结果不久后评论区大神 [@gaohuazuo](/archives/9370#comment-20572) 就给出了一个构建格林函数更高效、更直观的方案，让笔者自愧不如。再联想起之前大神之前在[《生成扩散模型漫谈（十二）：“硬刚”扩散ODE》](/archives/9280#comment-19951)同样也给出了一个关于扩散ODE的精彩描述（间接启发了上一篇博客的结果），大神的洞察力不得不让人叹服。

经过讨论和思考，笔者发现大神的思路本质上就是一阶偏微分方程的特征线法，通过构造特定的向量场保证初值条件，然后通过求解微分方程保证终值条件，同时保证了初值和终值条件，真的非常巧妙！最后，笔者将自己的收获总结成此文，作为上一篇的后续。

## 前情回顾 #

简单回顾一下上一篇文章的结果。假设随机变量$\boldsymbol{x}_0\in\mathbb{R}^d$连续地变换成$\boldsymbol{x}_T$，其变化规律服从ODE  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt}=\boldsymbol{f}_t(\boldsymbol{x}_t)\label{eq-ode}\end{equation}  
那么对应的$t$时刻的分布$p_t(\boldsymbol{x}_t)$服从“连续性方程”：  
\begin{equation}\frac{\partial}{\partial t} p_t(\boldsymbol{x}_t) = - \nabla_{\boldsymbol{x}_t}\cdot\Big(\boldsymbol{f}_t(\boldsymbol{x}_t) p_t(\boldsymbol{x}_t)\Big)\label{eq:ode-f-eq-fp}\end{equation}  
记$\boldsymbol{u}(t, \boldsymbol{x}_t)=(p_t( \boldsymbol{x}_t), \boldsymbol{f}_t(\boldsymbol{x}_t) p_t(\boldsymbol{x}_t))\in\mathbb{R}^{d+1}$，那么连续性方程可以简写成  
\begin{equation}\left\\{\begin{aligned}  
&\nabla_{(t,\, \boldsymbol{x}_t)}\cdot\boldsymbol{u}(t, \boldsymbol{x}_t)=0 \\\  
&\boldsymbol{u}_1(0, \boldsymbol{x}_0) = p_0(\boldsymbol{x}_0),\int \boldsymbol{u}_1(t, \boldsymbol{x}_t) d\boldsymbol{x}_t = 1  
\end{aligned}\right.\label{eq:div-eq}\end{equation}  
为了求解这个方程，可以用格林函数的思想，即先求解  
\begin{equation}\left\\{\begin{aligned}  
&\nabla_{(t,\, \boldsymbol{x}_t)}\cdot\boldsymbol{G}(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0)=0\\\  
&\boldsymbol{G}_1(0, 0; \boldsymbol{x}_t, \boldsymbol{x}_0) = \delta(\boldsymbol{x}_t - \boldsymbol{x}_0),\int \boldsymbol{G}_1(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0) d\boldsymbol{x}_t = 1  
\end{aligned}\right.\label{eq:div-green}\end{equation}  
那么  
\begin{equation}\boldsymbol{u}(t, \boldsymbol{x}_t) = \int \boldsymbol{G}(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0)p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 = \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)}[\boldsymbol{G}(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0)]\label{eq:div-green-int}\end{equation}  
就是满足约束条件的解之一。

## 几何直观 #

所谓格林函数，其实思想很简单，它就是说我们先不要着急解决复杂数据生成，我们先假设要生成的数据只有一个点$\boldsymbol{x}_0$，先解决单个数据点的生成。有的读者想这不是很简单吗？直接$\boldsymbol{x}_T\times 0 + \boldsymbol{x}_0$就完事了？当然不是这么简单，我们需要的是连续的、渐变的生成，如下图所示，就是$t=T$上的任意一点$\boldsymbol{x}_T$，都沿着一条光滑轨迹运行到$t=0$的$\boldsymbol{x}_0$上：  


[![格林函数示意图。图中T=1，在t=1处的每个点，都沿着特定的轨迹运行到t=0处的一个点，除了公共点外，轨迹之间无重叠，这些轨迹就是格林函数的场线](/usr/uploads/2022/12/3260358826.svg)](/usr/uploads/2022/12/3260358826.svg "点击查看原图")

格林函数示意图。图中T=1，在t=1处的每个点，都沿着特定的轨迹运行到t=0处的一个点，除了公共点外，轨迹之间无重叠，这些轨迹就是格林函数的场线

而我们的目的，只是构造一个生成模型出来，所以我们原则上并不在乎轨迹的形状如何，只要它们都穿过$\boldsymbol{x}_0$，那么，我们可以人为地选择我们喜欢的、经过$\boldsymbol{x}_0$的一个轨迹簇，记为  
\begin{equation}\boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \boldsymbol{x}_T\label{eq:track}\end{equation}  
再次强调，这代表着以$\boldsymbol{x}_0$为起点、以$\boldsymbol{x}_T$为终点的一个轨迹簇，轨迹自变量、因变量分别为$t,\boldsymbol{x}_t$，起点$\boldsymbol{x}_0$是固定不变的，终点$\boldsymbol{x}_T$是可以任意变化的，轨迹的形状是无所谓的，我们可以选择直线、抛物线等等。

现在我们对式$\eqref{eq:track}$两边求导，由于$\boldsymbol{x}_T$是可以随意变化的，它相当于微分方程的积分常数，对它求导就等于$\boldsymbol{0}$，于是我们有  
\begin{equation}\frac{\partial \boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}{\partial \boldsymbol{x}_t}\frac{d\boldsymbol{x}_t}{dt} + \frac{\partial \boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}{\partial t} = \boldsymbol{0} \\\  
\Downarrow \\\  
\frac{d\boldsymbol{x}_t}{dt} = - \left(\frac{\partial \boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}{\partial \boldsymbol{x}_t}\right)^{-1} \frac{\partial \boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}{\partial t}\end{equation}  
对比式$\eqref{eq-ode}$，我们就得到  
\begin{equation}\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = - \left(\frac{\partial \boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}{\partial \boldsymbol{x}_t}\right)^{-1} \frac{\partial \boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}{\partial t}\label{eq:f-xt-x0}\end{equation}  
这里将原本的记号$\boldsymbol{f}_t(\boldsymbol{x}_t)$替换为了$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$，以标记轨线具有公共点$\boldsymbol{x}_0$。也就是说，这样构造出来的力场$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$所对应的ODE轨迹，必然是经过$\boldsymbol{x}_0$的，这就保证了格林函数的初值条件。

## 特征线法 #

既然初值条件有保证了，那么我们不妨要求更多一点：再保证一下终值条件。终值条件也就是希望$t=T$时$\boldsymbol{x}_T$的分布是跟$\boldsymbol{x}_0$无关的简单分布。上一篇文章的求解框架的主要缺点，就是无法直接保证终值分布的简单性，只能通过事后分析来研究。这篇文章的思路则是直接通过设计特定的$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$来保证初值条件，然后就有剩余空间来保证终值条件了。而且，同时保证了初、终值后，在满足连续性方程$\eqref{eq:ode-f-eq-fp}$的前提下，积分条件是自然满足的。

用数学的方式说，我们就是要在给定$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$和$p_T(\boldsymbol{x}_T)$的前提下，去求解方程$\eqref{eq:ode-f-eq-fp}$，这是一个一阶偏微分方程，可以通过“特征线法”求解，其理论介绍可以参考笔者之前写的[《一阶偏微分方程的特征线法》](/archives/4718)。首先，我们将方程$\eqref{eq:ode-f-eq-fp}$等价地改写成  
\begin{equation}\frac{\partial}{\partial t} p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) + \nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) \cdot \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = - p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) \nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\end{equation}  
同前面类似，由于接下来是在给定起点$\boldsymbol{x}_0$进行求解，所以上式将$p_t(\boldsymbol{x}_t)$替换为$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$，以标记这是起点为$\boldsymbol{x}_0$的解。

特征线法的思路，是先在某条特定的轨迹上考虑偏微分方程的解，这可以将偏微分转化为常微分，降低求解难度。具体来说，我们假设$\boldsymbol{x}_t$是$t$的函数，在方程$\eqref{eq-ode}$的轨线上求解。此时由于成立方程$\eqref{eq-ode}$，将上式左端的$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$替换为$\frac{d\boldsymbol{x}_t}{dt}$后，左端正好是$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$的全微分，所以此时有  
\begin{equation}\frac{d}{dt}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = - p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) \nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\end{equation}  
注意，此时所有的$\boldsymbol{x}_t$应当被替换为对应的$t$的函数，这理论上可以从轨迹方程$\eqref{eq:track}$解出。替换后，上式的$p$、$\boldsymbol{f}$都是纯粹$t$的函数，所以上式只是关于$p$的一个线性常微分方程，可以解得  
\begin{equation}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = C \exp\left(\int_t^T \nabla_{\boldsymbol{x}_s}\cdot \boldsymbol{f}_s(\boldsymbol{x}_s|\boldsymbol{x}_0) ds\right)\end{equation}  
代入终值条件$p_T(\boldsymbol{x}_T)$，得到$C=p_T(\boldsymbol{x}_T)$，即  
\begin{equation}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = p_T(\boldsymbol{x}_T) \exp\left(\int_t^T \nabla_{\boldsymbol{x}_s}\cdot \boldsymbol{f}_s(\boldsymbol{x}_s|\boldsymbol{x}_0) ds\right)\label{eq:pt-xt-x0}\end{equation}  
把轨迹方程$\eqref{eq:track}$的$\boldsymbol{x}_T$代入，就得到一个只含有$t,\boldsymbol{x}_t,\boldsymbol{x}_0$的函数，便是最终要求解的格林函数$\boldsymbol{G}_1(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0)$了，相应地有$\boldsymbol{G}_{> 1}(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0)=p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$。

## 训练目标 #

有了格林函数，我们就可以得到  
\begin{equation}\begin{aligned}  
\boldsymbol{u}_1(t, \boldsymbol{x}_t) =&\, \int p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 = p_t(\boldsymbol{x}_t)\\\  
\boldsymbol{u}_{> 1}(t, \boldsymbol{x}_t) =&\, \int \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0  
\end{aligned}\end{equation}  
于是  
\begin{equation}\begin{aligned}  
\boldsymbol{f}_t(\boldsymbol{x}_t)=&\,\frac{\boldsymbol{u}_{> 1}(t, \boldsymbol{x}_t)}{\boldsymbol{u}_1(t, \boldsymbol{x}_t)} \\\  
=&\,\int \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) \frac{p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) p_0(\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_t)} d\boldsymbol{x}_0 \\\  
=&\,\int \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) p_t(\boldsymbol{x}_0|\boldsymbol{x}_t) d\boldsymbol{x}_0 \\\  
=&\,\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]  
\end{aligned}\end{equation}  
根据[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209#%E5%BE%97%E5%88%86%E5%8C%B9%E9%85%8D)中构建得分匹配目标的方法，可以构建训练目标  
\begin{equation}\begin{aligned}&\,\mathbb{E}_{\boldsymbol{x}_t\sim p  
_t(\boldsymbol{x}_t)}\Big[\mathbb{E}_{\boldsymbol{x}_0\sim p  
_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right]\Big] d\boldsymbol{x}_t \\\  
=&\, \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t \sim p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)p  
_0(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right]  
\end{aligned}\label{eq:score-match}\end{equation}  
它跟[《Flow Matching for Generative Modeling》](https://papers.cool/arxiv/2210.02747)所给出的“Conditional Flow Matching”形式上是一致的，后面我们还会看到，该论文的结果都可以从本文的方法推出。训练完成后，就可以通过求解方程$\frac{d\boldsymbol{x}_t}{dt}=\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$来生成样本了。从这个训练目标也可以看出，我们对$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$的要求是易于采样就行了。

## 一些例子 #

可能前面的抽象结果对大家来说还是不大好理解，接下来我们来给出一些具体例子，以便加深大家对这个框架的直观理解。至于特征线法本身，笔者在[《一阶偏微分方程的特征线法》](/archives/4718)也说过，一开始笔者也觉得特征线法像是“变魔术”一样难以捉摸，按照步骤操作似乎不困难，但总把握不住关键之处，理解它需要一个反复斟酌的思考过程，无法进一步代劳了。

### 直线轨迹 #

作为最简单的例子，我们假设$\boldsymbol{x}_T$是沿着直线轨迹变为$\boldsymbol{x}_0$，简单起见我们还可以将$T$设为1，这不会损失一般性，那么$\boldsymbol{x}_t$的方程可以写为  
\begin{equation}\boldsymbol{x}_t = (\boldsymbol{x}_1 - \boldsymbol{x}_0)t + \boldsymbol{x}_0\quad\Rightarrow\quad \frac{\boldsymbol{x}_t - \boldsymbol{x}_0}{t} + \boldsymbol{x}_0 = \boldsymbol{x}_1\label{eq:simplest-x1}\end{equation}  
根据式$\eqref{eq:f-xt-x0}$，有  
\begin{equation}\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{\boldsymbol{x}_t - \boldsymbol{x}_0}{t}\end{equation}  
此时$\nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)=\frac{d}{t}$，根据式$\eqref{eq:pt-xt-x0}$就有  
\begin{equation}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{p_1(\boldsymbol{x}_1)}{t^d}\end{equation}  
代入式$\eqref{eq:simplest-x1}$中的$\boldsymbol{x}_1$，得到  
\begin{equation}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{p_1\left(\frac{\boldsymbol{x}_t - \boldsymbol{x}_0}{t} + \boldsymbol{x}_0\right)}{t^d}\end{equation}  
特别地，如果$p_1(\boldsymbol{x}_1)$是标准正态分布，那么上式实则意味着$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)=\mathcal{N}(\boldsymbol{x}_t;(1-t)\boldsymbol{x}_0,t^2\boldsymbol{I})$，这正好是常见的高斯扩散模型之一。这个框架的新结果，是允许我们选择更一般的先验分布$p_1(\boldsymbol{x}_1)$，比如均匀分布。另外在介绍得分匹配$\eqref{eq:score-match}$时也已经说了，对$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$我们只需要知道它的采样方式就行了，而上式告诉我们只需要先验分布易于采样就行，因为：  
\begin{equation}\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\quad\Leftrightarrow\quad \boldsymbol{x}_t=(1-t)\boldsymbol{x}_0 + t\boldsymbol{\varepsilon},\,\boldsymbol{\varepsilon}\sim p_1(\boldsymbol{\varepsilon})\end{equation}

### 效果演示 #

注意，我们假设从$\boldsymbol{x}_0$到$\boldsymbol{x}_1$的轨迹是一条直线，这仅仅是对于单点生成的，也就是格林函数解。当通过格林函数叠加出一般分布对应的的力场$\boldsymbol{f}_t(\boldsymbol{x}_t)$时，其生成轨迹就不再是直线了。

下图演示了先验分布为均匀分布时多点生成的轨线图：  


[![单点生成](/usr/uploads/2022/12/4276426527.svg)](/usr/uploads/2022/12/4276426527.svg "点击查看原图")

单点生成

[![两点生成](/usr/uploads/2022/12/2845102935.svg)](/usr/uploads/2022/12/2845102935.svg "点击查看原图")

两点生成

[![三点生成](/usr/uploads/2022/12/2249423194.svg)](/usr/uploads/2022/12/2249423194.svg "点击查看原图")

三点生成

参考作图代码：
    
    
    import numpy as np
    from scipy.integrate import odeint
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    
    prior = lambda x: 0.5 if 2 >= x >= 0 else 0
    p = lambda xt, x0, t: prior((xt - x0) / t + x0) / t
    f = lambda xt, x0, t: (xt - x0) / t
    
    def f_full(xt, t):
        x0s = [0.5, 0.5, 1.2, 1.7]  # 0.5出现两次，代表其频率是其余的两倍
        fs = np.array([f(xt, x0, t) for x0 in x0s]).reshape(-1)
        ps = np.array([p(xt, x0, t) for x0 in x0s]).reshape(-1)
        return (fs * ps).sum() / (ps.sum() + 1e-8)
    
    for x1 in np.arange(0.01, 1.99, 0.10999/2):
        ts = np.arange(1, 0, -0.001)
        xs = odeint(f_full, x1, ts).reshape(-1)[::-1]
        ts = ts[::-1]
        if abs(xs[0] - 0.5) < 0.1:
            _ = plt.plot(ts, xs, color='skyblue')
        elif abs(xs[0] - 1.2) < 0.1:
            _ = plt.plot(ts, xs, color='orange')
        else:
            _ = plt.plot(ts, xs, color='limegreen')
    
    plt.xlabel('$t$')
    plt.ylabel(r'$\boldsymbol{x}$')
    plt.show()

### 一般推广 #

其实上面的结果还可以一般地推广到  
\begin{equation}\boldsymbol{x}_t = \boldsymbol{\mu}_t(\boldsymbol{x}_0) + \sigma_t \boldsymbol{x}_1\quad\Rightarrow\quad \frac{\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0)}{\sigma_t }= \boldsymbol{x}_1\end{equation}  
这里的$\boldsymbol{\mu}_t(\boldsymbol{x}_0)$是任意满足$\boldsymbol{\mu}_0(\boldsymbol{x}_0)=\boldsymbol{x}_0, \boldsymbol{\mu}_1(\boldsymbol{x}_0)=\boldsymbol{0}$的$\mathbb{R}^d\mapsto\mathbb{R}^d$函数，$\sigma_t$是任意满足$\sigma_0=0,\sigma_1=1$的单调递增函数。根据式$\eqref{eq:f-xt-x0}$，有  
\begin{equation}\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \dot{\boldsymbol{\mu}}_t(\boldsymbol{x}_0) + \frac{\dot{\sigma}_t}{\sigma_t}(\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0))\end{equation}  
这也等价于[《Flow Matching for Generative Modeling》](https://papers.cool/arxiv/2210.02747)中的式$(15)$，此时$\nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)=\frac{d\dot{\sigma}_t}{\sigma_t}$，根据式$\eqref{eq:pt-xt-x0}$就有  
\begin{equation}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{p_1(\boldsymbol{x}_1)}{\sigma_t^d}\end{equation}  
代入$\boldsymbol{x}_1$，最终结果是  
\begin{equation}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{p_1\left(\frac{\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0)}{\sigma_t }\right)}{\sigma_t^d}\end{equation}  
这是关于线性ODE扩散的一般结果，包含高斯扩散，也允许使用非高斯的先验分布。

### 再复杂些？ #

前面的例子，都是通过$\boldsymbol{x}_0$（的某个变换）与$\boldsymbol{x}_1$的简单线性插值（插值权重纯粹是$t$的函数）来构建$\boldsymbol{x}_t$的变化轨迹。那么一个很自然的问题就是：可不可以考虑更复杂的轨迹呢？理论上可以，但是更高的复杂度意味着隐含了更多的假设，而我们通常很难检验目标数据是否支持这些假设，因此通常都不考虑更复杂的轨迹了。此外，对于更复杂的轨迹，解析求解的难度通常也更高，不管是理论还是实验，都难以操作下去。

更重要的一点的，我们目前所假设的轨迹，仅仅是单点生成的轨迹而已，前面已经演示了，即便假设为直线，多点生成依然会导致复杂的曲线。所以，如果单点生成的轨迹都假设得不必要的复杂，那么可以想像多点生成的轨迹复杂度将会奇高，模型可能会极度不稳定。

## 文章小结 #

接着上一篇文章的内容，本文再次讨论了ODE式扩散模型的构建思路。这一次我们从几何直观出发，通过构造特定的向量场保证结果满足初值分布条件，然后通过求解微分方程保证终值分布条件，得到一个同时满足初值和终值条件的格林函数。特别地，该方法允许我们使用任意简单分布作为先验分布，摆脱以往对高斯分布的依赖来构建扩散模型。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9379>_

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

苏剑林. (Dec. 22, 2022). 《生成扩散模型漫谈（十五）：构建ODE的一般步骤（中） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9379>

@online{kexuefm-9379,  
title={生成扩散模型漫谈（十五）：构建ODE的一般步骤（中）},  
author={苏剑林},  
year={2022},  
month={Dec},  
url={\url{https://spaces.ac.cn/archives/9379}},  
} 


---

## 公式推导与注释

本节将从多个角度深入探讨格林函数的数学理论，特别是在扩散过程中的应用。我们将涵盖泛函分析、偏微分方程理论以及扩散过程的概率论视角。

### 1. 格林函数的基本性质

#### 1.1 对称性与正定性

格林函数作为线性算子的逆核，具有重要的代数性质。对于自伴算子$L$，其格林函数满足对称性：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) = G(t, s; \boldsymbol{y}, \boldsymbol{x})
$$

**证明**：考虑自伴算子$L = L^*$，其格林函数定义满足：

$$
L_{\boldsymbol{x}} G(t, s; \boldsymbol{x}, \boldsymbol{y}) = \delta(\boldsymbol{x} - \boldsymbol{y})\delta(t - s)
$$

对任意测试函数$\phi(\boldsymbol{x}), \psi(\boldsymbol{y})$，有：

$$
\begin{aligned}
\int\int \phi(\boldsymbol{x}) L_{\boldsymbol{x}} G(t, s; \boldsymbol{x}, \boldsymbol{y}) \psi(\boldsymbol{y}) d\boldsymbol{x} d\boldsymbol{y} &= \int \phi(\boldsymbol{y}) \psi(\boldsymbol{y}) d\boldsymbol{y} \\
\int\int (L_{\boldsymbol{x}} \phi(\boldsymbol{x})) G(t, s; \boldsymbol{x}, \boldsymbol{y}) \psi(\boldsymbol{y}) d\boldsymbol{x} d\boldsymbol{y} &= \int \phi(\boldsymbol{y}) \psi(\boldsymbol{y}) d\boldsymbol{y}
\end{aligned}
$$

由自伴性和分部积分，交换$\phi, \psi$的角色，得到对称性。

对于扩散型算子，格林函数还满足**正定性**。设$L = -\nabla \cdot (D\nabla) + V(\boldsymbol{x})$，其中$D > 0$是扩散系数，$V \geq 0$是势函数，则：

$$
\int\int f(\boldsymbol{x}) G(t, t; \boldsymbol{x}, \boldsymbol{y}) f(\boldsymbol{y}) d\boldsymbol{x} d\boldsymbol{y} \geq 0
$$

这来自于算子的正定性：

$$
\langle f, Lf \rangle = \int D|\nabla f|^2 + V|f|^2 \geq 0
$$

#### 1.2 归一化条件与概率解释

对于时间依赖的格林函数$G(t, s; \boldsymbol{x}, \boldsymbol{y})$，当它表示扩散过程的转移概率密度时，必须满足归一化条件：

$$
\int G(t, s; \boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y} = 1, \quad \forall t > s, \boldsymbol{x}
$$

**推导**：从连续性方程出发，

$$
\frac{\partial}{\partial t} p(t, \boldsymbol{x}) = -\nabla \cdot (p\boldsymbol{f})
$$

两边对$\boldsymbol{x}$积分，利用散度定理和边界条件（假设$p\boldsymbol{f} \to 0$当$|\boldsymbol{x}| \to \infty$）：

$$
\frac{d}{dt} \int p(t, \boldsymbol{x}) d\boldsymbol{x} = -\int \nabla \cdot (p\boldsymbol{f}) d\boldsymbol{x} = 0
$$

因此总概率守恒。对于格林函数解$p(t, \boldsymbol{x}) = \int G(t, 0; \boldsymbol{x}, \boldsymbol{y}) p_0(\boldsymbol{y}) d\boldsymbol{y}$，要求：

$$
\int p(t, \boldsymbol{x}) d\boldsymbol{x} = \int p_0(\boldsymbol{y}) d\boldsymbol{y} = 1
$$

这要求格林函数满足归一化条件。

### 2. 谱表示理论

#### 2.1 特征值展开

对于紧域上的自伴算子$L$，可以利用谱分解理论展开格林函数。设$L$的特征值问题为：

$$
L \phi_n(\boldsymbol{x}) = \lambda_n \phi_n(\boldsymbol{x})
$$

其中$\{\phi_n\}$构成完备正交系：$\langle \phi_m, \phi_n \rangle = \delta_{mn}$。则格林函数可展开为：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) = \sum_{n=0}^{\infty} \frac{e^{-\lambda_n(t-s)}}{\lambda_n} \phi_n(\boldsymbol{x}) \phi_n^*(\boldsymbol{y})
$$

**推导**：利用$\delta$函数的展开：

$$
\delta(\boldsymbol{x} - \boldsymbol{y}) = \sum_{n=0}^{\infty} \phi_n(\boldsymbol{x}) \phi_n^*(\boldsymbol{y})
$$

对于时间演化算子$\partial_t - L$，其格林函数$G$满足：

$$
(\partial_t - L) G(t, s; \boldsymbol{x}, \boldsymbol{y}) = \delta(t-s)\delta(\boldsymbol{x} - \boldsymbol{y})
$$

设$G = \sum_{n} g_n(t, s) \phi_n(\boldsymbol{x}) \phi_n^*(\boldsymbol{y})$，代入得：

$$
\sum_n \left[\frac{dg_n}{dt} - \lambda_n g_n\right] \phi_n(\boldsymbol{x}) \phi_n^*(\boldsymbol{y}) = \delta(t-s) \sum_n \phi_n(\boldsymbol{x}) \phi_n^*(\boldsymbol{y})
$$

比较系数：

$$
\frac{dg_n}{dt} - \lambda_n g_n = \delta(t-s)
$$

对于$t > s$，解为：

$$
g_n(t, s) = \frac{e^{\lambda_n(t-s)}}{\lambda_n}
$$

（这里假设$\lambda_n < 0$以保证收敛，或者理解为$L$是负定的）。

#### 2.2 谱表示的收敛性

谱展开的收敛性依赖于特征值的增长速率。对于$d$维欧氏空间上的拉普拉斯算子$\Delta$，Weyl渐近公式给出：

$$
N(\lambda) \sim \frac{\omega_d}{(2\pi)^d} \text{Vol}(\Omega) \lambda^{d/2}, \quad \lambda \to \infty
$$

其中$N(\lambda)$是小于$\lambda$的特征值个数，$\omega_d$是单位球的体积。这意味着特征值增长如：

$$
\lambda_n \sim C n^{2/d}
$$

因此谱展开中的项衰减如：

$$
\left|\frac{e^{-\lambda_n t}}{\lambda_n} \phi_n(\boldsymbol{x}) \phi_n(\boldsymbol{y})\right| \lesssim \frac{e^{-C n^{2/d} t}}{n^{2/d}}
$$

对于$t > 0$，这个级数快速收敛。

### 3. 半群理论

#### 3.1 演化算子的半群性质

时间演化算子$U(t, s): f \mapsto \int G(t, s; \cdot, \boldsymbol{y}) f(\boldsymbol{y}) d\boldsymbol{y}$构成一个算子半群。半群的基本性质是：

$$
U(t, r) = U(t, s) \circ U(s, r), \quad t \geq s \geq r
$$

用格林函数表示，这就是**Chapman-Kolmogorov方程**：

$$
G(t, r; \boldsymbol{x}, \boldsymbol{z}) = \int G(t, s; \boldsymbol{x}, \boldsymbol{y}) G(s, r; \boldsymbol{y}, \boldsymbol{z}) d\boldsymbol{y}
$$

**证明**：设$p(r, \boldsymbol{z})$是初始分布，则：

$$
\begin{aligned}
p(t, \boldsymbol{x}) &= \int G(t, r; \boldsymbol{x}, \boldsymbol{z}) p(r, \boldsymbol{z}) d\boldsymbol{z} \\
&= \int G(t, s; \boldsymbol{x}, \boldsymbol{y}) p(s, \boldsymbol{y}) d\boldsymbol{y} \\
&= \int G(t, s; \boldsymbol{x}, \boldsymbol{y}) \left[\int G(s, r; \boldsymbol{y}, \boldsymbol{z}) p(r, \boldsymbol{z}) d\boldsymbol{z}\right] d\boldsymbol{y} \\
&= \int \left[\int G(t, s; \boldsymbol{x}, \boldsymbol{y}) G(s, r; \boldsymbol{y}, \boldsymbol{z}) d\boldsymbol{y}\right] p(r, \boldsymbol{z}) d\boldsymbol{z}
\end{aligned}
$$

由初始分布的任意性，得到C-K方程。

#### 3.2 无穷小生成元

半群的生成元定义为：

$$
\mathcal{L} f = \lim_{t \to 0^+} \frac{U(t, 0)f - f}{t}
$$

对于扩散过程，生成元具有形式：

$$
\mathcal{L} = \boldsymbol{f}(\boldsymbol{x}) \cdot \nabla + \frac{1}{2} \text{tr}(D(\boldsymbol{x}) \nabla^2)
$$

其中$\boldsymbol{f}$是漂移系数，$D$是扩散矩阵。这对应于随机微分方程：

$$
d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t) dt + \sqrt{D(\boldsymbol{X}_t)} d\boldsymbol{W}_t
$$

对于本文的ODE情况（无扩散项），生成元简化为：

$$
\mathcal{L} = \boldsymbol{f}(\boldsymbol{x}, t) \cdot \nabla
$$

这是一个一阶偏微分算子，对应于无噪声的确定性流。

#### 3.3 Hille-Yosida定理

半群理论的核心结果是Hille-Yosida定理，它刻画了什么样的算子可以作为半群的生成元。

**定理（Hille-Yosida）**：线性算子$\mathcal{L}$是压缩半群的生成元当且仅当：
1. $\mathcal{L}$稠密定义且闭
2. 对所有$\lambda > 0$，$(\lambda I - \mathcal{L})^{-1}$存在且满足$\|(\lambda I - \mathcal{L})^{-1}\| \leq 1/\lambda$

对于扩散过程，可以验证生成元$\mathcal{L} = \frac{1}{2}\Delta + \boldsymbol{b} \cdot \nabla$满足这些条件（在适当的Sobolev空间中）。

### 4. 热核理论

#### 4.1 热核的定义与性质

热核是热方程$\partial_t u = \Delta u$的基本解。在全空间$\mathbb{R}^d$上，热核为：

$$
K(t, \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{(4\pi t)^{d/2}} \exp\left(-\frac{|\boldsymbol{x} - \boldsymbol{y}|^2}{4t}\right)
$$

这是一个高斯函数，具有以下性质：

**性质1（归一化）**：
$$
\int K(t, \boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y} = 1
$$

**性质2（半群性）**：
$$
\int K(t, \boldsymbol{x}, \boldsymbol{z}) K(s, \boldsymbol{z}, \boldsymbol{y}) d\boldsymbol{z} = K(t+s, \boldsymbol{x}, \boldsymbol{y})
$$

**性质3（初值条件）**：
$$
\lim_{t \to 0^+} K(t, \boldsymbol{x}, \boldsymbol{y}) = \delta(\boldsymbol{x} - \boldsymbol{y})
$$

这里的极限理解为弱收敛，即对任意连续有界函数$f$：

$$
\lim_{t \to 0^+} \int K(t, \boldsymbol{x}, \boldsymbol{y}) f(\boldsymbol{y}) d\boldsymbol{y} = f(\boldsymbol{x})
$$

#### 4.2 热核的推导

我们从Fourier变换推导热核。设$u(t, \boldsymbol{x})$满足热方程$\partial_t u = \Delta u$，取Fourier变换：

$$
\hat{u}(t, \boldsymbol{k}) = \int e^{-i\boldsymbol{k} \cdot \boldsymbol{x}} u(t, \boldsymbol{x}) d\boldsymbol{x}
$$

则：

$$
\frac{\partial \hat{u}}{\partial t} = -|\boldsymbol{k}|^2 \hat{u}
$$

解得：

$$
\hat{u}(t, \boldsymbol{k}) = e^{-t|\boldsymbol{k}|^2} \hat{u}(0, \boldsymbol{k})
$$

对于初值$u(0, \boldsymbol{x}) = \delta(\boldsymbol{x} - \boldsymbol{y})$，有$\hat{u}(0, \boldsymbol{k}) = e^{-i\boldsymbol{k} \cdot \boldsymbol{y}}$，因此：

$$
\hat{K}(t, \boldsymbol{k}, \boldsymbol{y}) = e^{-t|\boldsymbol{k}|^2 - i\boldsymbol{k} \cdot \boldsymbol{y}}
$$

逆Fourier变换：

$$
\begin{aligned}
K(t, \boldsymbol{x}, \boldsymbol{y}) &= \frac{1}{(2\pi)^d} \int e^{i\boldsymbol{k} \cdot \boldsymbol{x} - t|\boldsymbol{k}|^2 - i\boldsymbol{k} \cdot \boldsymbol{y}} d\boldsymbol{k} \\
&= \frac{1}{(2\pi)^d} \int e^{i\boldsymbol{k} \cdot (\boldsymbol{x} - \boldsymbol{y}) - t|\boldsymbol{k}|^2} d\boldsymbol{k}
\end{aligned}
$$

完成平方：

$$
i\boldsymbol{k} \cdot (\boldsymbol{x} - \boldsymbol{y}) - t|\boldsymbol{k}|^2 = -t\left|\boldsymbol{k} - \frac{i(\boldsymbol{x} - \boldsymbol{y})}{2t}\right|^2 - \frac{|\boldsymbol{x} - \boldsymbol{y}|^2}{4t}
$$

利用高斯积分公式$\int e^{-a|z|^2} dz = (\pi/a)^{d/2}$：

$$
K(t, \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{(2\pi)^d} \left(\frac{\pi}{t}\right)^{d/2} e^{-\frac{|\boldsymbol{x} - \boldsymbol{y}|^2}{4t}} = \frac{1}{(4\pi t)^{d/2}} e^{-\frac{|\boldsymbol{x} - \boldsymbol{y}|^2}{4t}}
$$

#### 4.3 带漂移的热核

对于带漂移项的扩散方程：

$$
\partial_t u = \Delta u + \boldsymbol{b}(\boldsymbol{x}) \cdot \nabla u
$$

可以通过Girsanov变换或者Feynman-Kac公式得到格林函数的表达式。在常漂移$\boldsymbol{b}$的情况下，热核为：

$$
K(t, \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{(4\pi t)^{d/2}} \exp\left(-\frac{|\boldsymbol{x} - \boldsymbol{y} - \boldsymbol{b}t|^2}{4t}\right)
$$

这对应于均值为$\boldsymbol{y} + \boldsymbol{b}t$、方差为$2t\boldsymbol{I}$的高斯分布。

### 5. 扩散过程的格林函数

#### 5.1 Fokker-Planck方程

扩散过程由随机微分方程描述：

$$
d\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t, t) dt + \boldsymbol{\sigma}(\boldsymbol{X}_t, t) d\boldsymbol{W}_t
$$

其概率密度演化遵循Fokker-Planck方程（也称Kolmogorov前向方程）：

$$
\frac{\partial p}{\partial t} = -\nabla \cdot (p\boldsymbol{f}) + \frac{1}{2}\sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j}(D_{ij} p)
$$

其中扩散矩阵$D = \boldsymbol{\sigma} \boldsymbol{\sigma}^T$。格林函数$G(t, s; \boldsymbol{x}, \boldsymbol{y})$满足：

$$
\frac{\partial G}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot (G\boldsymbol{f}) + \frac{1}{2}\sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j}(D_{ij} G)
$$

配以初始条件$G(s, s; \boldsymbol{x}, \boldsymbol{y}) = \delta(\boldsymbol{x} - \boldsymbol{y})$。

#### 5.2 后向Kolmogorov方程

除了前向方程，格林函数还满足后向方程（关于初始变量$\boldsymbol{y}$和初始时间$s$）：

$$
-\frac{\partial G}{\partial s} = \boldsymbol{f}(\boldsymbol{y}, s) \cdot \nabla_{\boldsymbol{y}} G + \frac{1}{2}\sum_{i,j} D_{ij}(\boldsymbol{y}, s) \frac{\partial^2 G}{\partial y_i \partial y_j}
$$

这两个方程是对偶的，反映了扩散过程的时间可逆性（在适当的测度变换下）。

#### 5.3 路径积分表示

格林函数可以通过路径积分（Feynman-Kac公式）表示。对于扩散过程：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) = \mathbb{E}\left[\delta(\boldsymbol{X}_t - \boldsymbol{x}) \Big| \boldsymbol{X}_s = \boldsymbol{y}\right]
$$

形式上可以写为路径积分：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) = \int_{\gamma: \boldsymbol{y} \to \boldsymbol{x}} \mathcal{D}\gamma \exp\left(-\frac{1}{2}\int_s^t |\dot{\gamma}_r - \boldsymbol{f}(\gamma_r, r)|^2 dr\right)
$$

这里$\mathcal{D}\gamma$是路径测度。虽然路径积分在数学上需要严格定义，但它提供了有用的直观理解和计算方法（如弱噪声极限、鞍点近似等）。

### 6. 时间依赖的格林函数

#### 6.1 非齐次情况

当系统参数依赖于时间时，格林函数不再满足平移不变性$G(t, s) \neq G(t-s, 0)$。此时需要完整保留两个时间参数。

对于时变ODE $\frac{d\boldsymbol{x}}{dt} = \boldsymbol{f}_t(\boldsymbol{x})$，连续性方程为：

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \boldsymbol{f}_t) = 0
$$

格林函数满足：

$$
\frac{\partial G}{\partial t} + \nabla_{\boldsymbol{x}} \cdot (G \boldsymbol{f}_t(\boldsymbol{x})) = 0
$$

#### 6.2 Dyson级数

对于时变系统，演化算子可以展开为Dyson级数（时间有序指数）：

$$
U(t, s) = \mathcal{T}\exp\left(\int_s^t \mathcal{L}_r dr\right) = \sum_{n=0}^{\infty} \int_s^t dr_1 \int_s^{r_1} dr_2 \cdots \int_s^{r_{n-1}} dr_n \mathcal{L}_{r_1} \mathcal{L}_{r_2} \cdots \mathcal{L}_{r_n}
$$

其中$\mathcal{T}$表示时间排序算符。对于弱时间依赖，可以用微扰展开：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) = G_0(t-s; \boldsymbol{x}, \boldsymbol{y}) + \int_s^t dr \int G_0(t-r; \boldsymbol{x}, \boldsymbol{z}) V_r(\boldsymbol{z}) G_0(r-s; \boldsymbol{z}, \boldsymbol{y}) d\boldsymbol{z} + \cdots
$$

其中$G_0$是时间齐次的参考格林函数，$V_r$是时变微扰。

### 7. 初值响应分析

#### 7.1 线性响应理论

考虑初值的微小扰动$\boldsymbol{x}_0 \to \boldsymbol{x}_0 + \delta\boldsymbol{x}_0$，其对终值的影响由Jacobi矩阵刻画：

$$
\frac{\partial \boldsymbol{x}_t}{\partial \boldsymbol{x}_0} = J_t
$$

其演化方程为：

$$
\frac{dJ_t}{dt} = \frac{\partial \boldsymbol{f}_t}{\partial \boldsymbol{x}}(\boldsymbol{x}_t) J_t, \quad J_0 = I
$$

这是一个矩阵微分方程，形式解为：

$$
J_t = \mathcal{T}\exp\left(\int_0^t \frac{\partial \boldsymbol{f}_s}{\partial \boldsymbol{x}}(\boldsymbol{x}_s) ds\right)
$$

对于本文的格林函数，特征线$\boldsymbol{x}_t(\boldsymbol{x}_0)$的Jacobian正是概率密度变换的关键：

$$
p_t(\boldsymbol{x}_t | \boldsymbol{x}_0) = \delta(\boldsymbol{x}_t - \boldsymbol{\phi}_t(\boldsymbol{x}_0)) |\det J_t|^{-1}
$$

但由于我们采用了连续密度表示，这个关系蕴含在式$\eqref{eq:pt-xt-x0}$中。

#### 7.2 Liouville方程的观点

从辛几何的观点，相空间密度演化遵循Liouville方程：

$$
\frac{dp}{dt} + \{H, p\} = 0
$$

其中$\{,\}$是泊松括号。对于正则哈密顿系统，相空间体积守恒（Liouville定理）：

$$
\frac{d}{dt}\det J_t = \det J_t \cdot \text{tr}\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} = \det J_t \cdot \nabla \cdot \boldsymbol{f}
$$

因此：

$$
\det J_t = \exp\left(\int_0^t \nabla \cdot \boldsymbol{f}_s(\boldsymbol{x}_s) ds\right)
$$

这正好出现在式$\eqref{eq:pt-xt-x0}$中的指数因子里，体现了相空间体积变化对密度的影响。

### 8. 格林函数的渐近行为

#### 8.1 短时渐近

当$t \to s^+$时，格林函数趋向于$\delta$函数：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) \approx \frac{1}{(t-s)^{d/2}} \exp\left(-\frac{|\boldsymbol{x} - \boldsymbol{y}|^2}{4(t-s)}\right)
$$

这是热核的主导行为。更精确的展开包含梯度修正：

$$
G(t, s; \boldsymbol{x}, \boldsymbol{y}) = \frac{1}{(4\pi(t-s))^{d/2}} e^{-\frac{|\boldsymbol{x}-\boldsymbol{y}|^2}{4(t-s)}} \left[1 + O(t-s)\right]
$$

#### 8.2 长时渐近

对于$t \to \infty$，格林函数的行为依赖于系统是否有不变分布。如果存在唯一稳态$p_{\infty}(\boldsymbol{x})$，则：

$$
\lim_{t \to \infty} G(t, 0; \boldsymbol{x}, \boldsymbol{y}) = p_{\infty}(\boldsymbol{x})
$$

收敛速率由谱隙（spectral gap）$\lambda_1 - \lambda_0$决定：

$$
|G(t, 0; \boldsymbol{x}, \boldsymbol{y}) - p_{\infty}(\boldsymbol{x})| \lesssim e^{-(\lambda_1 - \lambda_0)t}
$$

对于扩散模型，$p_{\infty}$就是先验分布$p_T(\boldsymbol{x}_T)$。

#### 8.3 中间标度行为

在中间时间尺度，格林函数可能展现自相似行为或标度律。例如，对于幂律势场，可能有：

$$
G(t, 0; \boldsymbol{x}, \boldsymbol{y}) \sim t^{-\alpha} F\left(\frac{\boldsymbol{x}}{t^{\beta}}, \frac{\boldsymbol{y}}{t^{\beta}}\right)
$$

其中$\alpha, \beta$是标度指数，$F$是标度函数。

### 9. 与基本解的关系

#### 9.1 基本解的定义

偏微分方程$Lu = f$的基本解$E(\boldsymbol{x}, \boldsymbol{y})$满足：

$$
L_{\boldsymbol{x}} E(\boldsymbol{x}, \boldsymbol{y}) = \delta(\boldsymbol{x} - \boldsymbol{y})
$$

方程的解可表示为：

$$
u(\boldsymbol{x}) = \int E(\boldsymbol{x}, \boldsymbol{y}) f(\boldsymbol{y}) d\boldsymbol{y}
$$

#### 9.2 时间依赖方程的基本解

对于演化方程$(\partial_t - L)u = f$，基本解$E(t, s; \boldsymbol{x}, \boldsymbol{y})$满足：

$$
(\partial_t - L_{\boldsymbol{x}}) E(t, s; \boldsymbol{x}, \boldsymbol{y}) = \delta(t-s)\delta(\boldsymbol{x} - \boldsymbol{y})
$$

这正是格林函数$G(t, s; \boldsymbol{x}, \boldsymbol{y})$的定义方程。因此，格林函数就是时间依赖偏微分方程的基本解。

#### 9.3 Green公式与表示定理

利用Green第二恒等式，可以得到解的积分表示。对于热方程，有：

$$
u(t, \boldsymbol{x}) = \int G(t, 0; \boldsymbol{x}, \boldsymbol{y}) u(0, \boldsymbol{y}) d\boldsymbol{y} + \int_0^t \int G(t, s; \boldsymbol{x}, \boldsymbol{y}) f(s, \boldsymbol{y}) d\boldsymbol{y} ds
$$

第一项是初值的贡献，第二项是源项的贡献。对于齐次方程（$f=0$），只有初值项。

### 10. 实际计算方法

#### 10.1 直接数值积分

对于简单的格林函数（如高斯核），可以直接进行蒙特卡洛采样或数值积分。例如：

$$
p_t(\boldsymbol{x}_t) = \int G(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0) p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 \approx \frac{1}{N}\sum_{i=1}^N G(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0^{(i)})
$$

其中$\boldsymbol{x}_0^{(i)} \sim p_0$。

#### 10.2 特征线方法的数值实现

对于式$\eqref{eq:pt-xt-x0}$的计算，需要：
1. 从轨迹方程$\eqref{eq:track}$解出$\boldsymbol{x}_T(\boldsymbol{x}_t, \boldsymbol{x}_0)$
2. 计算$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$沿轨迹的散度积分
3. 代入先验分布$p_T$

**算法步骤**：
```
输入: t, x_t, x_0
1. 求解 φ_t(x_t|x_0) = x_T (代数或数值求解)
2. 计算 f_s(x_s|x_0) 对 s ∈ [t, T]
3. 数值积分 I = ∫_t^T ∇·f_s(x_s|x_0) ds
4. 计算 p_T(x_T)
5. 返回 p_t(x_t|x_0) = p_T(x_T) exp(I)
```

#### 10.3 谱方法

对于周期边界条件或紧支撑域，可以使用谱方法：
1. 展开$p_t = \sum_n c_n(t) \phi_n$
2. 将PDE转化为ODE系统：$\dot{c}_n = \sum_m M_{nm} c_m$
3. 求解ODE得到系数演化
4. 重构$p_t$

#### 10.4 有限元/有限差分

对于复杂几何或边界条件，使用空间离散化方法：
- 有限差分：将$\nabla, \Delta$离散化为差分算子
- 有限元：将解展开为分片多项式基函数
- 有限体积：保证守恒律的离散化

这些方法将PDE转化为大型稀疏线性系统，可用迭代法求解。

#### 10.5 神经网络参数化

现代扩散模型采用神经网络直接学习$\boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$或其期望$\boldsymbol{f}_t(\boldsymbol{x}_t)$。训练目标式$\eqref{eq:score-match}$可以通过随机梯度下降优化：

```
for epoch in epochs:
    采样 x_0 ~ p_0
    采样 t ~ Uniform[0, T]
    计算 x_t ~ p_t(·|x_0)
    计算损失 L = ||v_θ(x_t, t) - f_t(x_t|x_0)||²
    反向传播更新 θ
```

这避免了显式求解格林函数，而是通过数据驱动的方式学习速度场。

### 11. 特殊情况：线性轨迹的完整推导

#### 11.1 一般线性形式

回到式中的线性轨迹$\boldsymbol{x}_t = \boldsymbol{\mu}_t(\boldsymbol{x}_0) + \sigma_t \boldsymbol{x}_1$，我们详细推导所有中间步骤。

**步骤1**：计算速度场。轨迹方程为：

$$
\boldsymbol{\varphi}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0)}{\sigma_t}
$$

这应该等于$\boldsymbol{x}_1$（常数）。对$t$求全导数：

$$
\frac{\partial \boldsymbol{\varphi}_t}{\partial t} + \frac{\partial \boldsymbol{\varphi}_t}{\partial \boldsymbol{x}_t} \frac{d\boldsymbol{x}_t}{dt} = 0
$$

计算偏导：

$$
\frac{\partial \boldsymbol{\varphi}_t}{\partial t} = \frac{-\dot{\boldsymbol{\mu}}_t(\boldsymbol{x}_0)}{\sigma_t} - \frac{\dot{\sigma}_t}{\sigma_t^2}(\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0))
$$

$$
\frac{\partial \boldsymbol{\varphi}_t}{\partial \boldsymbol{x}_t} = \frac{1}{\sigma_t} I
$$

代入得：

$$
\frac{d\boldsymbol{x}_t}{dt} = -\sigma_t \frac{\partial \boldsymbol{\varphi}_t}{\partial t} = \dot{\boldsymbol{\mu}}_t(\boldsymbol{x}_0) + \frac{\dot{\sigma}_t}{\sigma_t}(\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0))
$$

**步骤2**：计算散度。

$$
\nabla_{\boldsymbol{x}_t} \cdot \boldsymbol{f}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \nabla_{\boldsymbol{x}_t} \cdot \left[\frac{\dot{\sigma}_t}{\sigma_t}(\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0))\right] = \frac{d\dot{\sigma}_t}{\sigma_t}
$$

**步骤3**：求解密度演化。根据式$\eqref{eq:pt-xt-x0}$：

$$
p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = p_T(\boldsymbol{x}_T) \exp\left(\int_t^T \frac{d\dot{\sigma}_s}{\sigma_s} ds\right) = p_T(\boldsymbol{x}_T) \exp\left(d\log\frac{\sigma_T}{\sigma_t}\right) = \frac{\sigma_T^d}{\sigma_t^d} p_T(\boldsymbol{x}_T)
$$

由于$\sigma_T = 1$：

$$
p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{p_T(\boldsymbol{x}_T)}{\sigma_t^d}
$$

**步骤4**：代入轨迹关系。从$\boldsymbol{x}_t = \boldsymbol{\mu}_t(\boldsymbol{x}_0) + \sigma_t \boldsymbol{x}_T$，得：

$$
\boldsymbol{x}_T = \frac{\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0)}{\sigma_t}
$$

代入上式：

$$
p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{1}{\sigma_t^d} p_T\left(\frac{\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0)}{\sigma_t}\right)
$$

这正是文中的结果。特别地，当$p_T = \mathcal{N}(0, I)$时：

$$
p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{1}{\sigma_t^d (2\pi)^{d/2}} \exp\left(-\frac{|\boldsymbol{x}_t - \boldsymbol{\mu}_t(\boldsymbol{x}_0)|^2}{2\sigma_t^2}\right) = \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{\mu}_t(\boldsymbol{x}_0), \sigma_t^2 I)
$$

#### 11.2 验证Chapman-Kolmogorov方程

作为完整性检查，我们验证格林函数满足C-K方程。对于线性情况：

$$
\begin{aligned}
&\int G(t, s; \boldsymbol{x}_t, \boldsymbol{x}_s) G(s, 0; \boldsymbol{x}_s, \boldsymbol{x}_0) d\boldsymbol{x}_s \\
=&\, \int \frac{1}{\sigma_{t|s}^d} p_T\left(\frac{\boldsymbol{x}_t - \boldsymbol{\mu}_{t|s}(\boldsymbol{x}_s)}{\sigma_{t|s}}\right) \frac{1}{\sigma_s^d} p_T\left(\frac{\boldsymbol{x}_s - \boldsymbol{\mu}_s(\boldsymbol{x}_0)}{\sigma_s}\right) d\boldsymbol{x}_s
\end{aligned}
$$

这里$\boldsymbol{\mu}_{t|s}, \sigma_{t|s}$表示从$s$到$t$的参数。对于线性轨迹，可以验证这等于$G(t, 0; \boldsymbol{x}_t, \boldsymbol{x}_0)$（两个高斯卷积仍是高斯）。

### 12. 总结与展望

通过以上详细推导，我们从多个角度理解了格林函数在扩散模型中的作用：

1. **泛函分析角度**：格林函数是线性微分算子的逆，具有对称性、正定性等代数性质，可以通过谱分解表示。

2. **偏微分方程角度**：格林函数是Fokker-Planck方程或连续性方程的基本解，满足Chapman-Kolmogorov方程，通过特征线法可以构造显式解。

3. **概率论角度**：格林函数表示扩散过程的转移概率密度，通过路径积分或Feynman-Kac公式可以得到其随机过程表示。

4. **半群理论角度**：格林函数定义了演化算子半群，其无穷小生成元刻画了系统的动力学。

5. **数值计算角度**：格林函数可以通过谱方法、有限元方法或神经网络学习等多种方式计算和近似。

这些不同的视角相互补充，为我们设计和分析扩散模型提供了坚实的数学基础。特别是本文的特征线法，通过巧妙地构造满足初值条件的轨迹族，再利用特征线法保证终值条件，得到了一个统一而优雅的框架，这对理解和推广扩散模型具有重要意义。

