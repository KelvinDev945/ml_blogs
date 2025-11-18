---
title: 生成扩散模型漫谈（二十）：从ReFlow到WGAN-GP
slug: 生成扩散模型漫谈二十从reflow到wgan-gp
date: 2023-06-28
tags: 优化, GAN, 梯度, 扩散, 生成模型
status: pending
---

# 生成扩散模型漫谈（二十）：从ReFlow到WGAN-GP

**原文链接**: [https://spaces.ac.cn/archives/9668](https://spaces.ac.cn/archives/9668)

**发布日期**: 

---

上一篇文章[《生成扩散模型漫谈（十九）：作为扩散ODE的GAN》](/archives/9662)中，我们介绍了如何将GAN理解为在另一个时间维度上的扩散ODE，简而言之，GAN实际上就是将扩散模型中样本的运动转化为生成器参数的运动！然而，该文章的推导过程依赖于Wasserstein梯度流等相对复杂和独立的内容，没法很好地跟扩散系列前面的文章连接起来，技术上显得有些“断层”。

在笔者看来，[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](/archives/9497)所介绍的ReFlow是理解扩散ODE的最直观方案，既然可以从扩散ODE的角度理解GAN，那么必定存在一个从ReFlow理解GAN的角度。经过一番尝试，笔者成功从ReFlow推出了类似WGAN-GP的结果。

## 理论回顾 #

之所以说“ReFlow是理解扩散ODE的最直观方案”，是因为它本身非常灵活，以及非常贴近实验代码——它能够通过ODE建立任意噪声分布到目标数据分布的映射，而且训练目标非常直观，不需要什么“弯弯绕绕”就可以直接跟实验代码对应起来。

具体来说，假设$\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)$是先验分布采样的随机噪声，$\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)$是目标分布采样的真实样本（注：前面的文章中，普遍都是$\boldsymbol{x}_T$是噪声、$\boldsymbol{x}_0$是目标样本，这里方便起见反过来了），ReFlow允许我们指定任意从$\boldsymbol{x}_0$到$\boldsymbol{x}_1$的运动轨迹。简单起见，ReFlow选择的是直线，即  
\begin{equation}\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1\label{eq:line}\end{equation}  
现在我们求出它满足的ODE：  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0\end{equation}  
这个ODE很简单，但是却不实用，因为我们想要的是通过ODE由$\boldsymbol{x}_0$生成$\boldsymbol{x}_1$，但上述ODE却将我们要生成的目标放在了方程里边，可谓是“因果倒置”了。为了弥补这个缺陷，ReFlow的思路很简单：学一个$\boldsymbol{x}_t$的函数去逼近$\boldsymbol{x}_1 - \boldsymbol{x}_0$，学完之后就用它来取代$\boldsymbol{x}_1 - \boldsymbol{x}_0$，即  
\begin{equation}\boldsymbol{\varphi}^* = \mathop{\text{argmin}}_{\boldsymbol{\varphi}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right]\label{eq:s-loss}\end{equation}  
以及  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0\quad\Rightarrow\quad\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{x}_t, t)\label{eq:ode-core}\end{equation}  
之前我们已经证明过，在$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t)$具有无限拟合能力的假设下，新的ODE确实能够实现从分布$p_0(\boldsymbol{x}_0)$到分布$p_1(\boldsymbol{x}_1)$的样本变换。

## 相对运动 #

ReFlow的重要特性之一，是它没有限制先验分布$p_0(\boldsymbol{x}_0)$的形式，这意味着我们可以将先验分布换成任意我们想要的分布，比如，由一个生成器$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$变换而来的分布：  
\begin{equation}\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)\quad\Leftrightarrow\quad \boldsymbol{x}_0 = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}),\,\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{equation}  
代入式$\eqref{eq:s-loss}$训练完成后，我们就可以利用式$\eqref{eq:ode-core}$，将任意$\boldsymbol{x}_0 = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$变换为真实样本$\boldsymbol{x}_1$了。

然而，我们并不满足于此。前面说过，GAN是将扩散模型中样本的运动转化为生成器参数的运动，这个ReFlow的框架中同样可以如此：假设生成器当前参数为$\boldsymbol{\theta}_{\tau}$，我们期望$\boldsymbol{\theta}_{\tau}\to \boldsymbol{\theta}_{\tau+1}$的变化能模拟式$\eqref{eq:ode-core}$前进一小步的效果  
\begin{equation}\boldsymbol{\theta}_{\tau+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\Big[\big\Vert \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}) - \epsilon\,\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0)\big\Vert^2\Big]\label{eq:g-loss}\end{equation}  
要注意，式$\eqref{eq:s-loss}$和式$\eqref{eq:ode-core}$中的$t$跟参数$\boldsymbol{\theta}_{\tau}$中的$\tau$不是同一含义，前者是ODE的时间参数，后者是训练进度，所以这里用了不同记号。此外，$\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z})$是作为ODE的$\boldsymbol{x}_0$出现的，所以往前推一小步时，得到的是$\boldsymbol{x}_{\epsilon}$，$\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{x}_t, t)$中要代入的时间$t$是$0$。

现在，我们有了新的$\boldsymbol{g}_{\boldsymbol{\theta}_{\tau+1}}(\boldsymbol{z})$，理论上它产生的分布更加接近真实分布一些（因为往前推了一小步），接着把它当作新的$\boldsymbol{x}_0$代入到式$\eqref{eq:s-loss}$训练，训练完成后又可以代入到式$\eqref{eq:g-loss}$优化生成器，以此类推，就是一个类似GAN的交替训练过程。

## WGAN-GP #

那么，能否将这个过程定量地跟已有的GAN联系起来呢？能！还是带梯度惩罚的[WGAN-GP](/archives/4439)。

首先我们来看损失函数$\eqref{eq:s-loss}$，将求期望的部分展开，结果是  
\begin{equation}\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t)\Vert^2 - \langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t),\boldsymbol{x}_1 - \boldsymbol{x}_0\rangle + \frac{1}{2}\Vert\boldsymbol{x}_1 - \boldsymbol{x}_0\Vert^2\end{equation}  
第三项跟参数$\boldsymbol{\varphi}$无关，去掉也不影响结果。现在我们假设$\boldsymbol{v}_{\boldsymbol{\varphi}}$有足够强的拟合能力，以至于我们不需要显式输入$t$，那么上式作为损失函数，等价于  
\begin{equation}\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t),\boldsymbol{x}_1 - \boldsymbol{x}_0\rangle = \frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \left\langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t),\frac{d\boldsymbol{x}_t}{dt}\right\rangle\end{equation}  
$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$是一个输入输出维度相同的向量函数，我们进一步假设它是某个标量函数$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$的梯度，即$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)=\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$，那么上式就是  
\begin{equation}\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \left\langle\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t),\frac{d\boldsymbol{x}_t}{dt}\right\rangle = \frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \frac{d D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)}{dt}\end{equation}  
假设$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$的变化比较平稳，那么$\frac{d D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)}{dt}$应该与它在$t=0,t=1$两点处的差分$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_1)-D_{\boldsymbol{\varphi}}(\boldsymbol{x}_0)$比较接近，于是上述损失函数近似于  
\begin{equation}\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - D_{\boldsymbol{\varphi}}(\boldsymbol{x}_1) + D_{\boldsymbol{\varphi}}(\boldsymbol{x}_0)\end{equation}  
熟悉GAN的读者应该会觉得很眼熟，它正是带梯度惩罚的WGAN的判别器损失函数！甚至连梯度惩罚项的$\boldsymbol{x}_t$的构造方式$\eqref{eq:line}$都一模一样（在真假样本之间线性插值）！唯一不同的是原始WGAN-GP的梯度惩罚是以1为中心，这里是以零为中心，但事实上[《WGAN-div：一个默默无闻的WGAN填坑者》](/archives/6139)、[《从动力学角度看优化算法（四）：GAN的第三个阶段》](/archives/6583)等文章已经表明以零为中心的梯度惩罚通常效果更好。

所以说，在特定的参数化和假设之下，损失函数$\eqref{eq:s-loss}$其实就等价于WGAN-GP的判别器损失。至于生成器损失，在上一篇文章[《生成扩散模型漫谈（十九）：作为扩散ODE的GAN》](/archives/9662)中我们已经证明了当$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)=\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$时，式$\eqref{eq:g-loss}$单步优化的梯度等价于  
\begin{equation}\boldsymbol{\theta}_{\tau+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}[-D(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))]\end{equation}  
的梯度，而这正好也是WGAN-GP的生成器损失。

## 文章小结 #

在这篇文章中，笔者尝试从ReFlow出发推导了WGAN-GP与扩散ODE之间的联系，这个角度相对来说更加简单直观，并且避免了Wasserstein梯度流等相对复杂的概念。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9668>_

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

苏剑林. (Jun. 28, 2023). 《生成扩散模型漫谈（二十）：从ReFlow到WGAN-GP 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9668>

@online{kexuefm-9668,  
title={生成扩散模型漫谈（二十）：从ReFlow到WGAN-GP},  
author={苏剑林},  
year={2023},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/9668}},  
} 


---

## 公式推导与注释

本节对文章中的核心公式进行详细推导和注释,帮助读者深入理解从ReFlow到WGAN-GP的理论联系。

### 一、ReFlow基础理论推导

#### 1.1 直线轨迹的设计

ReFlow的核心思想是在噪声分布$p_0(\boldsymbol{x}_0)$和目标分布$p_1(\boldsymbol{x}_1)$之间建立最直接的路径。

**公式 (1.1)** - 直线插值轨迹
\begin{equation}
\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1 \tag{1.1}
\end{equation}

**推导注释**: 这个公式定义了从$\boldsymbol{x}_0$到$\boldsymbol{x}_1$的线性插值路径,其中:
- 当$t=0$时,$\boldsymbol{x}_t = \boldsymbol{x}_0$(噪声)
- 当$t=1$时,$\boldsymbol{x}_t = \boldsymbol{x}_1$(目标样本)
- 中间任意$t\in(0,1)$都表示从噪声到目标的过渡状态

#### 1.2 ODE的推导

对公式(1.1)关于时间$t$求导:

\begin{equation}
\frac{d\boldsymbol{x}_t}{dt} = \frac{d}{dt}\left[(1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1\right] = -\boldsymbol{x}_0 + \boldsymbol{x}_1 = \boldsymbol{x}_1 - \boldsymbol{x}_0 \tag{1.2}
\end{equation}

**数学直觉**: 这个ODE表明,沿着直线轨迹运动的速度是恒定的,方向始终指向目标$\boldsymbol{x}_1$。

**问题所在**: 公式(1.2)将我们要生成的目标$\boldsymbol{x}_1$放在了方程右侧,这是"因果倒置"的——我们需要通过ODE生成$\boldsymbol{x}_1$,但右侧却已经包含了$\boldsymbol{x}_1$!

#### 1.3 速度场的学习

为解决因果倒置问题,ReFlow引入可学习的速度场$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t)$:

**公式 (1.3)** - 速度场学习目标
\begin{equation}
\boldsymbol{\varphi}^* = \mathop{\text{argmin}}_{\boldsymbol{\varphi}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0,\boldsymbol{x}_1\sim p_1}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right] \tag{1.3}
\end{equation}

**详细推导**:

(a) 损失函数展开:
\begin{align}
\mathcal{L}(\boldsymbol{\varphi}) &= \mathbb{E}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right] \tag{1.4}\\
&= \mathbb{E}\left[\frac{1}{2}\boldsymbol{v}_{\boldsymbol{\varphi}}^T\boldsymbol{v}_{\boldsymbol{\varphi}} - \boldsymbol{v}_{\boldsymbol{\varphi}}^T(\boldsymbol{x}_1 - \boldsymbol{x}_0) + \frac{1}{2}\Vert\boldsymbol{x}_1 - \boldsymbol{x}_0\Vert^2\right] \tag{1.5}
\end{align}

(b) 最后一项与$\boldsymbol{\varphi}$无关,优化时可忽略:
\begin{equation}
\boldsymbol{\varphi}^* = \mathop{\text{argmin}}_{\boldsymbol{\varphi}} \mathbb{E}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}\Vert^2 - \langle\boldsymbol{v}_{\boldsymbol{\varphi}}, \boldsymbol{x}_1 - \boldsymbol{x}_0\rangle\right] \tag{1.6}
\end{equation}

(c) 当模型容量足够时,最优解满足:
\begin{equation}
\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{x}_t, t) = \mathbb{E}[\boldsymbol{x}_1 - \boldsymbol{x}_0|\boldsymbol{x}_t] \tag{1.7}
\end{equation}

**理论意义**: 学习到的速度场$\boldsymbol{v}_{\boldsymbol{\varphi}^*}$给出了从任意中间状态$\boldsymbol{x}_t$到目标的期望方向。

### 二、生成器参数的运动

#### 2.1 参数化先验分布

引入生成器$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$来参数化先验分布:

\begin{equation}
\boldsymbol{x}_0 = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}), \quad \boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}) \tag{2.1}
\end{equation}

此时轨迹变为:
\begin{equation}
\boldsymbol{x}_t = (1-t)\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) + t \boldsymbol{x}_1 \tag{2.2}
\end{equation}

#### 2.2 参数更新的推导

**核心思想**: 将样本空间的运动转化为参数空间的运动。

给定当前参数$\boldsymbol{\theta}_{\tau}$,我们希望更新到$\boldsymbol{\theta}_{\tau+1}$使得:

\begin{equation}
\boldsymbol{g}_{\boldsymbol{\theta}_{\tau+1}}(\boldsymbol{z}) \approx \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}) + \epsilon\,\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0) \tag{2.3}
\end{equation}

**详细推导**:

(a) 最小二乘目标:
\begin{align}
\boldsymbol{\theta}_{\tau+1} &= \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}}\Big[\big\Vert \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}) - \epsilon\,\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0)\big\Vert^2\Big] \tag{2.4}
\end{align}

(b) 展开平方项:
\begin{align}
&\Vert \boldsymbol{g}_{\boldsymbol{\theta}} - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}} - \epsilon\boldsymbol{v}\Vert^2 \notag\\
=&\, \Vert \boldsymbol{g}_{\boldsymbol{\theta}} - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}\Vert^2 - 2\epsilon\langle\boldsymbol{g}_{\boldsymbol{\theta}} - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}, \boldsymbol{v}\rangle + \epsilon^2\Vert\boldsymbol{v}\Vert^2 \tag{2.5}
\end{align}

(c) 对$\boldsymbol{\theta}$求梯度(在$\boldsymbol{\theta}=\boldsymbol{\theta}_{\tau}$处):
\begin{align}
\nabla_{\boldsymbol{\theta}}\mathcal{L}\big|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{\tau}} &= \mathbb{E}_{\boldsymbol{z}}\left[-2\epsilon\nabla_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})\big|_{\boldsymbol{\theta}_{\tau}}^T \boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0)\right] \tag{2.6}\\
&= -2\epsilon\mathbb{E}_{\boldsymbol{z}}\left[\nabla_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z})^T \boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0)\right] \tag{2.7}
\end{align}

**关键观察**: 为什么在$t=0$处评估$\boldsymbol{v}$? 因为$\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z})$对应的是ODE的初始点$\boldsymbol{x}_0$,向前推进一小步$\epsilon$后到达$\boldsymbol{x}_{\epsilon}$。

### 三、从ReFlow到WGAN-GP

#### 3.1 损失函数的等价变换

原始损失函数(对应公式1.3):
\begin{equation}
\mathcal{L}_s = \mathbb{E}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t)\Vert^2 - \langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t),\boldsymbol{x}_1 - \boldsymbol{x}_0\rangle\right] \tag{3.1}
\end{equation}

**步骤1**: 利用$\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$,有:
\begin{equation}
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0 \tag{3.2}
\end{equation}

代入(3.1):
\begin{equation}
\mathcal{L}_s = \mathbb{E}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}\Vert^2 - \left\langle\boldsymbol{v}_{\boldsymbol{\varphi}},\frac{d\boldsymbol{x}_t}{dt}\right\rangle\right] \tag{3.3}
\end{equation}

#### 3.2 梯度场假设

**关键假设**: 假设$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$可以表示为某个标量势函数$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$的梯度:

\begin{equation}
\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t) = \nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t) \tag{3.4}
\end{equation}

**物理直觉**: 这类似于物理学中的保守力场,速度场可以由一个势能函数的梯度表示。

**步骤2**: 代入梯度场假设:
\begin{align}
\mathcal{L}_s &= \mathbb{E}\left[\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D\Vert^2 - \left\langle\nabla_{\boldsymbol{x}_t} D,\frac{d\boldsymbol{x}_t}{dt}\right\rangle\right] \tag{3.5}
\end{align}

#### 3.3 链式法则应用

利用链式法则:
\begin{equation}
\frac{d D(\boldsymbol{x}_t)}{dt} = \nabla_{\boldsymbol{x}_t} D \cdot \frac{d\boldsymbol{x}_t}{dt} = \left\langle\nabla_{\boldsymbol{x}_t} D, \frac{d\boldsymbol{x}_t}{dt}\right\rangle \tag{3.6}
\end{equation}

因此:
\begin{equation}
\mathcal{L}_s = \mathbb{E}\left[\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D\Vert^2 - \frac{d D(\boldsymbol{x}_t)}{dt}\right] \tag{3.7}
\end{equation}

#### 3.4 差分近似

**关键步骤**: 用差分近似导数:

当$D(\boldsymbol{x}_t)$沿$t$变化平稳时:
\begin{equation}
\frac{d D(\boldsymbol{x}_t)}{dt} \approx D(\boldsymbol{x}_1) - D(\boldsymbol{x}_0) \tag{3.8}
\end{equation}

**严格性分析**: 这个近似在以下条件下成立:
- $D(\boldsymbol{x}_t)$关于$t$是Lipschitz连续的
- 时间步长足够小
- $\boldsymbol{x}_t$的轨迹足够平滑

代入(3.7):
\begin{equation}
\mathcal{L}_s \approx \mathbb{E}\left[\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D\Vert^2 - D(\boldsymbol{x}_1) + D(\boldsymbol{x}_0)\right] \tag{3.9}
\end{equation}

#### 3.5 WGAN-GP形式

**最终结果**: 公式(3.9)正是WGAN-GP的判别器损失!

\begin{equation}
\mathcal{L}_{\text{WGAN-GP}} = \mathbb{E}_{\boldsymbol{x}_1\sim p_{\text{real}}}[D(\boldsymbol{x}_1)] - \mathbb{E}_{\boldsymbol{x}_0\sim p_{\text{fake}}}[D(\boldsymbol{x}_0)] + \lambda\mathbb{E}_{\boldsymbol{x}_t}\left[\Vert\nabla_{\boldsymbol{x}_t} D\Vert^2\right] \tag{3.10}
\end{equation}

**对比分析**:
1. 第一项$-D(\boldsymbol{x}_1)$: 最大化真样本的判别分数
2. 第二项$+D(\boldsymbol{x}_0)$: 最小化假样本的判别分数
3. 第三项$\frac{1}{2}\Vert\nabla D\Vert^2$: 梯度惩罚项

**关键差异**: 原始WGAN-GP的梯度惩罚是$(\Vert\nabla D\Vert - 1)^2$,而这里推导出的是$\Vert\nabla D\Vert^2$,以零为中心。

### 四、梯度惩罚的深入分析

#### 4.1 为什么梯度惩罚以零为中心更好?

**理论原因1 - 最优传输视角**:

在最优传输理论中,Wasserstein距离的Kantorovich-Rubinstein对偶形式为:
\begin{equation}
W(p_1, p_0) = \sup_{\Vert f\Vert_L \leq 1} \mathbb{E}_{\boldsymbol{x}_1}[f(\boldsymbol{x}_1)] - \mathbb{E}_{\boldsymbol{x}_0}[f(\boldsymbol{x}_0)] \tag{4.1}
\end{equation}

其中$\Vert f\Vert_L$表示Lipschitz常数。

**推导**:
- 1-Lipschitz条件: $|f(\boldsymbol{x}) - f(\boldsymbol{y})| \leq \Vert\boldsymbol{x} - \boldsymbol{y}\Vert$
- 微分形式: $\Vert\nabla f\Vert \leq 1$ (几乎处处成立)

**理论原因2 - ReFlow的直线轨迹**:

沿直线轨迹$\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$:
\begin{align}
\frac{d\boldsymbol{x}_t}{dt} &= \boldsymbol{x}_1 - \boldsymbol{x}_0 = \text{常数} \tag{4.2}\\
\nabla_{\boldsymbol{x}_t}\log p_t &= 0 \quad \text{(理想情况)} \tag{4.3}
\end{align}

因此最优速度场应该满足:
\begin{equation}
\boldsymbol{v}^* = \nabla D^* = 0 \tag{4.4}
\end{equation}

这自然导出以零为中心的惩罚$\Vert\nabla D\Vert^2$。

#### 4.2 实验验证

文献《WGAN-div》和《从动力学角度看优化算法(四)》的实验表明:
- 零中心梯度惩罚: 训练更稳定,生成质量更高
- 1中心梯度惩罚: 可能导致梯度消失,训练不稳定

### 五、生成器损失的等价性

#### 5.1 从MSE到判别器损失

回顾公式(2.4)的生成器更新:
\begin{equation}
\boldsymbol{\theta}_{\tau+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}\Big[\Vert \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}) - \epsilon\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0)\Vert^2\Big] \tag{5.1}
\end{equation}

**步骤1**: 对$\boldsymbol{\theta}$求梯度:
\begin{align}
\nabla_{\boldsymbol{\theta}}\mathcal{L}\big|_{\boldsymbol{\theta}_{\tau}} &= \mathbb{E}\Big[2(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}} - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}} - \epsilon\boldsymbol{v})\cdot\nabla_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}}\big|_{\boldsymbol{\theta}_{\tau}}\Big] \tag{5.2}\\
&= \mathbb{E}\Big[-2\epsilon\boldsymbol{v}_{\boldsymbol{\varphi}^*}\cdot\nabla_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}\Big] \tag{5.3}
\end{align}

**步骤2**: 代入$\boldsymbol{v}_{\boldsymbol{\varphi}^*} = \nabla D_{\boldsymbol{\varphi}^*}$:
\begin{align}
\nabla_{\boldsymbol{\theta}}\mathcal{L}\big|_{\boldsymbol{\theta}_{\tau}} &= \mathbb{E}\Big[-2\epsilon\nabla_{\boldsymbol{x}}D\cdot\nabla_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}\Big] \tag{5.4}\\
&= -2\epsilon\mathbb{E}\Big[\nabla_{\boldsymbol{\theta}}D(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}))\Big] \tag{5.5}
\end{align}

**步骤3**: 这等价于最小化:
\begin{equation}
\mathcal{L}_G = \mathbb{E}_{\boldsymbol{z}}[-D(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))] \tag{5.6}
\end{equation}

**结论**: 从扩散ODE的角度推导出的生成器损失,与WGAN的生成器损失完全一致!

### 六、理论统一的意义

#### 6.1 多角度理解

从三个视角看待同一个模型:

**视角1 - 扩散模型**:
\begin{equation}
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t) \tag{6.1}
\end{equation}

**视角2 - GAN**:
\begin{align}
\mathcal{L}_D &= \mathbb{E}[D(\boldsymbol{x}_{\text{real}})] - \mathbb{E}[D(\boldsymbol{x}_{\text{fake}})] + \lambda\mathbb{E}[\Vert\nabla D\Vert^2] \tag{6.2}\\
\mathcal{L}_G &= -\mathbb{E}[D(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))] \tag{6.3}
\end{align}

**视角3 - 最优传输**:
\begin{equation}
W(p_1, p_0) = \inf_{\gamma\in\Pi(p_1,p_0)} \mathbb{E}_{(\boldsymbol{x}_1,\boldsymbol{x}_0)\sim\gamma}[\Vert\boldsymbol{x}_1 - \boldsymbol{x}_0\Vert] \tag{6.4}
\end{equation}

#### 6.2 实践启示

**启示1**: GAN可以看作在参数空间的扩散过程
- 样本空间: $\boldsymbol{x}_t$的运动
- 参数空间: $\boldsymbol{\theta}_{\tau}$的运动
- 联系: 公式(2.4)建立了两者的对应关系

**启示2**: 梯度惩罚的本质是速度场的正则化
- 物理意义: 保证速度场平滑
- 数学意义: Lipschitz约束
- 几何意义: 最优传输路径

**启示3**: 交替训练的理论依据
- 判别器: 估计速度场$\boldsymbol{v} = \nabla D$
- 生成器: 沿速度场前进一步
- 收敛性: 理论上收敛到$p_{\text{fake}} = p_{\text{real}}$

### 七、高级扩展

#### 7.1 高阶ODE求解器

从公式(2.3)的一阶欧拉法:
\begin{equation}
\boldsymbol{\theta}_{\tau+1} = \boldsymbol{\theta}_{\tau} + \epsilon\nabla_{\boldsymbol{\theta}}\mathbb{E}[D(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}})] \tag{7.1}
\end{equation}

可以扩展到高阶方法,如Runge-Kutta:

**二阶RK方法**:
\begin{align}
\boldsymbol{k}_1 &= \nabla_{\boldsymbol{\theta}}\mathbb{E}[D(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}})] \tag{7.2}\\
\boldsymbol{k}_2 &= \nabla_{\boldsymbol{\theta}}\mathbb{E}[D(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau} + \frac{\epsilon}{2}\boldsymbol{k}_1})] \tag{7.3}\\
\boldsymbol{\theta}_{\tau+1} &= \boldsymbol{\theta}_{\tau} + \epsilon\boldsymbol{k}_2 \tag{7.4}
\end{align}

**优势**:
- 更高的数值精度
- 更少的迭代步数
- 更稳定的训练

#### 7.2 自适应步长

可以根据梯度范数自适应调整步长:
\begin{equation}
\epsilon_{\tau} = \frac{\epsilon_0}{\sqrt{1 + \Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}\Vert^2}} \tag{7.5}
\end{equation}

这类似于Adam等自适应优化器的思想。

### 八、实验验证要点

#### 8.1 超参数设置建议

基于理论推导,推荐设置:

1. **梯度惩罚系数**:
\begin{equation}
\lambda = \frac{1}{2} \tag{8.1}
\end{equation}
这对应公式(3.9)中的系数。

2. **步长**:
\begin{equation}
\epsilon \in [10^{-4}, 10^{-3}] \tag{8.2}
\end{equation}
太大会导致数值不稳定,太小会训练缓慢。

3. **判别器更新频率**:
建议每更新生成器1次,更新判别器5次,以充分估计速度场。

#### 8.2 收敛性监控

监控以下指标:

**指标1 - 速度场误差**:
\begin{equation}
\epsilon_v = \mathbb{E}\left[\Vert\boldsymbol{v}_{\boldsymbol{\varphi}} - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right] \tag{8.3}
\end{equation}

**指标2 - 梯度范数**:
\begin{equation}
\Vert\nabla D\Vert_{\text{avg}} = \mathbb{E}[\Vert\nabla_{\boldsymbol{x}_t} D(\boldsymbol{x}_t)\Vert] \tag{8.4}
\end{equation}
应该趋向于0。

**指标3 - Wasserstein距离估计**:
\begin{equation}
\hat{W} = \mathbb{E}[D(\boldsymbol{x}_{\text{real}})] - \mathbb{E}[D(\boldsymbol{x}_{\text{fake}})] \tag{8.5}
\end{equation}
应该逐渐减小。

### 九、总结与洞察

本文从ReFlow出发,通过一系列严格的数学推导,建立了扩散模型与WGAN-GP之间的理论联系:

**核心洞察**:
1. **统一视角**: GAN和扩散模型本质上都在求解最优传输问题
2. **梯度惩罚**: 以零为中心的梯度惩罚有更深刻的扩散理论依据
3. **参数运动**: 样本空间的扩散等价于参数空间的梯度流

**理论贡献**:
- 避免了Wasserstein梯度流等复杂概念
- 提供了更直观的ReFlow到WGAN-GP的路径
- 为理解GAN的训练动力学提供了新视角

**未来方向**:
- 将此框架推广到其他GAN变体
- 研究更高阶的ODE求解器在GAN训练中的应用
- 探索参数空间扩散的几何性质

