---
title: 生成扩散模型漫谈（六）：一般框架之ODE篇
slug: 生成扩散模型漫谈六一般框架之ode篇
date: 2022-08-08
tags: 详细推导, flow模型, 微分方程, 生成模型, DDPM, 扩散, ODE, 概率流, Fokker-Planck, 连续归一化流
status: completed
tags_reviewed: true
---
# 生成扩散模型漫谈（六）：一般框架之ODE篇

**原文链接**: [https://spaces.ac.cn/archives/9228](https://spaces.ac.cn/archives/9228)

**发布日期**: 

---

上一篇文章[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)中，我们对宋飏博士的论文[《Score-Based Generative Modeling through Stochastic Differential Equations》](https://papers.cool/arxiv/2011.13456)做了基本的介绍和推导。然而，顾名思义，上一篇文章主要涉及的是原论文中SDE相关的部分，而遗留了被称为“概率流ODE（Probability flow ODE）”的部分内容，所以本文对此做个补充分享。

事实上，遗留的这部分内容在原论文的正文中只占了一小节的篇幅，但我们需要新开一篇文章来介绍它，因为笔者想了很久后发现，该结果的推导还是没办法绕开Fokker-Planck方程，所以我们需要一定的篇幅来介绍Fokker-Planck方程，然后才能请主角ODE登场。

## 再次反思 #

我们来大致总结一下上一篇文章的内容：首先，我们通过SDE来定义了一个前向过程（“拆楼”）：  
\begin{equation}d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}\label{eq:sde-forward}\end{equation}  
然后，我们推导了相应的逆向过程的SDE（“建楼”）：  
\begin{equation}d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) \right] dt + g_t d\boldsymbol{w}\label{eq:sde-reverse}\end{equation}  
最后，我们推导了用神经网络$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$来估计$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$的损失函数（得分匹配）：  
\begin{equation}\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t \sim p(\boldsymbol{x}_t|\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right] \end{equation}  
至此，我们完成了扩散模型的训练、预测的一般框架，可以说，它是DDPM的非常一般化的推广了。但正如[《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](/archives/9181)中介绍的DDIM是DDPM的高观点反思结果，SDE作为DDPM的推广，有没有相应的“高观点反思结果”呢？有，其结果就是本文主题“概率流ODE”。

## Dirac函数 #

DDIM做了什么反思呢？很简单，DDIM发现DDPM的训练目标主要跟$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$有关，而跟$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$无关，所以它以$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$为出发点，去推导更一般的$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$和$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1},\boldsymbol{x}_0)$。概率流ODE做的反思是类似的，它想知道在SDE框架中，对于固定的$p(\boldsymbol{x}_t)$，能找出哪些不同的$p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t)$（或者说找到不同的前向过程SDE）。

我们先写出前向过程$\eqref{eq:sde-forward}$的离散形式  
\begin{equation}\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:sde-discrete}\end{equation}  
这个等式描述的是随机变量$\boldsymbol{x}_{t+\Delta t},\boldsymbol{x}_t,\boldsymbol{\varepsilon}$之间的关系，我们可以方便地对两边求期望，然而我们并非想求期望，而是想求分布$p(\boldsymbol{y}_t)$（所满足的关系式）。怎么将分布转换成期望形式呢？答案是[Dirac函数](/archives/1870)：  
\begin{equation}p(\boldsymbol{x}) = \int \delta(\boldsymbol{x} - \boldsymbol{y}) p(\boldsymbol{y}) d\boldsymbol{y} = \mathbb{E}_{\boldsymbol{y}}[\delta(\boldsymbol{x} - \boldsymbol{y})]\end{equation}  
Dirac函数严格定义是属于泛函分析的内容，但我们通常都是当它是普通函数来处理，一般都能得到正确的结果。由上式还可以得知，对于任意$f(\boldsymbol{x})$，成立  
\begin{equation}p(\boldsymbol{x})f(\boldsymbol{x}) = \int \delta(\boldsymbol{x} - \boldsymbol{y}) p(\boldsymbol{y})f(\boldsymbol{y}) d\boldsymbol{y} = \mathbb{E}_{\boldsymbol{y}}[\delta(\boldsymbol{x} - \boldsymbol{y}) f(\boldsymbol{y})]\end{equation}  
直接对上式两边求偏导数，得到  
\begin{equation}\nabla_{\boldsymbol{x}}[p(\boldsymbol{x}) f(\boldsymbol{x})] = \mathbb{E}_{\boldsymbol{y}}\left[\nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{y}) f(\boldsymbol{y})\right] = \mathbb{E}_{\boldsymbol{y}}\left[f(\boldsymbol{y})\nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{y})\right]\end{equation}  
这是后面要用到的性质之一，可以发现它本质上是狄拉克函数的导数能够通过积分转移到所乘函数上去。

## F-P方程 #

经过上述铺垫，现在我们根据式$\eqref{eq:sde-discrete}$写出  
\begin{equation}\begin{aligned}  
&\,\delta(\boldsymbol{x} - \boldsymbol{x}_{t+\Delta t}) \\\\[5pt]  
=&\, \delta(\boldsymbol{x} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t - g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}) \\\  
\approx&\, \delta(\boldsymbol{x} - \boldsymbol{x}_t) - \left(\boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}\right)\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t) + \frac{1}{2} \left(g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}\cdot \nabla_{\boldsymbol{x}}\right)^2\delta(\boldsymbol{x} - \boldsymbol{x}_t)  
\end{aligned}\end{equation}  
这里当$\delta(\cdot)$是普通函数那样做了泰勒展开，只保留了不超过$\mathcal{O}(\Delta t)$的项。现在我们两边求期望：  
\begin{equation}\begin{aligned}  
&\,p_{t+\Delta t}(\boldsymbol{x}) \\\\[6pt]  
=&\,\mathbb{E}_{\boldsymbol{x}_{t+\Delta t}}\left[\delta(\boldsymbol{x} - \boldsymbol{x}_{t+\Delta t})\right] \\\  
\approx&\, \mathbb{E}_{\boldsymbol{x}_t, \boldsymbol{\varepsilon}}\left[\delta(\boldsymbol{x} - \boldsymbol{x}_t) - \left(\boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}\right)\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t) + \frac{1}{2} \left(g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}\cdot \nabla_{\boldsymbol{x}}\right)^2\delta(\boldsymbol{x} - \boldsymbol{x}_t)\right] \\\  
=&\, \mathbb{E}_{\boldsymbol{x}_t}\left[\delta(\boldsymbol{x} - \boldsymbol{x}_t) - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t) + \frac{1}{2} g_t^2\Delta t \nabla_{\boldsymbol{x}}\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t)\right] \\\  
=&\,p_t(\boldsymbol{x}) - \nabla_{\boldsymbol{x}}\cdot\left[\boldsymbol{f}_t(\boldsymbol{x})\Delta t\, p_t(\boldsymbol{x})\right] + \frac{1}{2}g_t^2\Delta t \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})  
\end{aligned}\end{equation}  
两边除以$\Delta t$，并取$\Delta t\to 0$的极限，结果是  
\begin{equation}\frac{\partial}{\partial t} p_t(\boldsymbol{x}) = - \nabla_{\boldsymbol{x}}\cdot\left[\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})\right] + \frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\label{eq:fp}  
\end{equation}  
这就是式$\eqref{eq:sde-forward}$所对应的“F-P方程”（[Fokker-Planck方程](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)），它是描述边际分布的偏微分方程。

## 等价变换 #

大家看到偏微分方程不用担心，因为这里并没有打算去研究怎么求解偏微分方程，只是借助它来引导一个等价变换而已。对于任意满足$\sigma_t^2\leq g_t^2$的函数$\sigma_t$，F-P方程$\eqref{eq:fp}$完全等价于  
\begin{equation}\begin{aligned}  
\frac{\partial}{\partial t} p_t(\boldsymbol{x}) =&\, - \nabla_{\boldsymbol{x}}\cdot\left[\boldsymbol{f}_t(\boldsymbol{x})p_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\right] + \frac{1}{2}\sigma_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x}) \\\  
=&\,- \nabla_{\boldsymbol{x}}\cdot\left[\left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right)p_t(\boldsymbol{x})\right] + \frac{1}{2}\sigma_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})  
\end{aligned}\label{eq:fp-2}\end{equation}  
形式上该F-P方程又相当于原来的F-P的$\boldsymbol{f}_t(\boldsymbol{x})$换成了$\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$、$g_t$换成了$\sigma_t$，根据式$\eqref{eq:fp}$对应于式$\eqref{eq:sde-forward}$，上式则对应于  
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}\label{eq:sde-forward-2}\end{equation}  
但是别忘了式$\eqref{eq:fp}$跟式$\eqref{eq:fp-2}$是完全等价的，所以这意味着式$\eqref{eq:sde-forward}$和式$\eqref{eq:sde-forward-2}$这两个随机微分方程所对应的边际分布$p_t(\boldsymbol{x})$是完全等价的！这个结果告诉我们存在不同方差的前向过程，它们产生的边际分布是一样的。这个结果相当于DDIM的升级版，后面我们还会证明，当$\boldsymbol{f}_t(\boldsymbol{x})$是关于$\boldsymbol{x}$的线性函数时，它就完全等价于DDIM。

特别地，根据上一篇SDE的结果，我们可以写出式$\eqref{eq:sde-forward-2}$对应的反向SDE：  
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 + \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}\label{eq:sde-reverse-2}\end{equation}

## 神经ODE #

式$\eqref{eq:sde-forward-2}$允许我们改变采样过程的方差，这里我们特别考虑$\sigma_t = 0$的极端情形，此时SDE退化为ODE（常微分方程）：  
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt\label{eq:flow-ode}\end{equation}  
这个ODE称为“概率流ODE（Probability flow ODE）”，由于实践中的$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$需要用神经网络$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$近似，所以上式也对应一个“神经ODE”。

为什么要特别研究方差为0的情形呢？因为此时传播过程不带噪声，从$\boldsymbol{x}_0$到$\boldsymbol{x}_T$是一个确定性变换，所以我们直接反向求解ODE就能得到由$\boldsymbol{x}_T$变换为$\boldsymbol{x}_0$的逆变换，这也是一个确定性变换（直接在式$\eqref{eq:sde-reverse-2}$中代入$\sigma_t=0$也可以发现前向和反向的方程是一样的）。这个过程和[flow模型](/tag/flow/)是一致的（即通过一个可逆的变换将噪声变换成样本），所以概率流ODE允许我们将扩散模型的结果与flow模型相关结果对应起来，比如原论文提到概率流ODE允许我们做精确的似然计算、获得隐变量表征等，这些本质上都是flow模型的好处。由于flow模型的可逆性，它还允许我们在隐变量空间对原图做一些图片编辑等。

另一方面，从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的变换由一个ODE描述，这意味着我们可以通过各种高阶的ODE数值算法来加速从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的变换过程。当然，原则上SDE的求解也有一些加速方法，但SDE的加速研究远远不如ODE的容易和深入。总的来说，相比SDE，ODE在理论分析和实际求解中都显得更为简单直接。

## 回顾DDIM #

在[《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](/archives/9181)的最后，我们推导了DDIM的连续版本对应于ODE  
\begin{equation}\frac{d}{ds}\left(\frac{\boldsymbol{x}(s)}{\bar{\alpha}(s)}\right) = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}(s), t(s)\right)\frac{d}{ds}\left(\frac{\bar{\beta}(s)}{\bar{\alpha}(s)}\right)\label{eq:ddim-ode}\end{equation}  
接下来我们可以看到，该结果其实就是本文的式$\eqref{eq:flow-ode}$在$\boldsymbol{f}_t(\boldsymbol{x})$取线性函数$f_t \boldsymbol{x}$时的特例：在[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)的末尾，我们推导过对应的关系  
\begin{equation}\left\\{\begin{aligned}  
&f_t = \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt} \\\  
&g^2 (t) = 2\bar{\alpha}_t \bar{\beta}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right) \\\  
&\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)}{\bar{\beta}_t}  
\end{aligned}\right.\end{equation}  
将这些关系代入到式$\eqref{eq:flow-ode}$【$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$替换为$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$】后，整理得到  
\begin{equation}\frac{1}{\bar{\alpha}_t}\frac{d\boldsymbol{x}}{dt} - \frac{\boldsymbol{x}}{\bar{\alpha}_t^2}\frac{d\bar{\alpha}_t}{dt} = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\end{equation}  
左端可以进一步整理得到$\frac{d}{dt}\left(\frac{\boldsymbol{x}}{\bar{\alpha}_t}\right)$，因此上式跟式$\eqref{eq:ddim-ode}$完全等价。

## 文章小结 #

本文在SDE篇的基础上，借助F-P方程推导了更一般化的前向方程，继而推导出了“概率流ODE”，并证明了DDIM是它的一个特例。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9228>_

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

苏剑林. (Aug. 08, 2022). 《生成扩散模型漫谈（六）：一般框架之ODE篇 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9228>

@online{kexuefm-9228,  
title={生成扩散模型漫谈（六）：一般框架之ODE篇},  
author={苏剑林},  
year={2022},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/9228}},  
} 


---

## 公式推导与注释

本节将对概率流ODE进行极详细的数学推导，从多个角度（微分方程理论、动力系统、概率论）深入理解从SDE到ODE的转换过程、数值求解方法、以及与连续归一化流的关系。

### 一、从Fokker-Planck方程到概率流ODE的完整推导

#### 1.1 F-P方程的深入理解

Fokker-Planck方程描述了概率密度函数$p_t(\boldsymbol{x})$随时间的演化，它是从微观的随机动力学（SDE）推导出宏观的概率分布演化方程。对于一般的SDE：
$$d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}$$

对应的Fokker-Planck方程为：
$$\frac{\partial p_t(\boldsymbol{x})}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot [\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})] + \frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x})$$

这个方程由两部分组成：

**漂移项（Drift term）**：$-\nabla_{\boldsymbol{x}} \cdot [\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})]$，描述确定性流动导致的概率密度变化。从物理角度看，这是连续性方程的形式，表示"概率流"$\boldsymbol{J} = \boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})$的散度。

**扩散项（Diffusion term）**：$\frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x})$，描述随机扰动导致的概率密度扩散。这是一个热方程类型的项，使概率密度趋于平滑。

#### 1.2 等价SDE族的构造

关键观察：对于给定的边际分布$p_t(\boldsymbol{x})$，存在无穷多个不同的SDE都能产生相同的边际分布演化。这是通过在F-P方程中引入一个"自由参数"$\sigma_t$来实现的。

对于满足$0 \leq \sigma_t^2 \leq g_t^2$的任意函数$\sigma_t$，考虑修改后的F-P方程：
$$\frac{\partial p_t(\boldsymbol{x})}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot \left[\tilde{\boldsymbol{f}}_t(\boldsymbol{x}) p_t(\boldsymbol{x})\right] + \frac{1}{2}\sigma_t^2 \nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x})$$

其中修改后的漂移项为：
$$\tilde{\boldsymbol{f}}_t(\boldsymbol{x}) = \boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$$

**证明等价性**：我们需要证明这个修改后的方程与原F-P方程完全等价。展开修改后的漂移项：

$$\begin{aligned}
-\nabla_{\boldsymbol{x}} \cdot [\tilde{\boldsymbol{f}}_t(\boldsymbol{x}) p_t(\boldsymbol{x})] &= -\nabla_{\boldsymbol{x}} \cdot \left[\left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) p_t(\boldsymbol{x})\right] \\
&= -\nabla_{\boldsymbol{x}} \cdot [\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})] + \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}} \cdot [p_t(\boldsymbol{x})\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})] \\
&= -\nabla_{\boldsymbol{x}} \cdot [\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})] + \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x})
\end{aligned}$$

因此，完整的修改后F-P方程为：
$$\begin{aligned}
\frac{\partial p_t(\boldsymbol{x})}{\partial t} &= -\nabla_{\boldsymbol{x}} \cdot [\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})] + \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x}) + \frac{1}{2}\sigma_t^2 \nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x}) \\
&= -\nabla_{\boldsymbol{x}} \cdot [\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})] + \frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}} \cdot \nabla_{\boldsymbol{x}} p_t(\boldsymbol{x})
\end{aligned}$$

这正是原始的F-P方程！这个推导表明，尽管SDE的形式不同，但它们产生的边际分布演化是完全相同的。

#### 1.3 概率流ODE的导出

现在取极限情况$\sigma_t = 0$，此时SDE退化为ODE：
$$\frac{d\boldsymbol{x}}{dt} = \boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$$

这就是**概率流ODE（Probability Flow ODE）**。它具有以下重要性质：

1. **确定性传播**：给定初始点$\boldsymbol{x}_0$，ODE的解是唯一确定的，不含随机性。
2. **保持边际分布**：尽管是确定性的，概率流ODE产生的边际分布$p_t(\boldsymbol{x})$与原始SDE完全相同。
3. **可逆性**：ODE是时间可逆的，我们可以精确地从$\boldsymbol{x}_T$反推$\boldsymbol{x}_0$。

**物理直觉**：从动力系统的角度，概率流ODE定义了一个时变的向量场$\boldsymbol{v}_t(\boldsymbol{x}) = \boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$。这个向量场在每个时刻$t$推动概率密度沿着特定的轨迹演化，使得整体的概率分布按照F-P方程演化，但个体粒子的运动是完全确定的。

### 二、ODE求解的数值方法

概率流ODE的形式为：
$$\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}_t(\boldsymbol{x}), \quad \boldsymbol{v}_t(\boldsymbol{x}) = \boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$$

其中$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$是神经网络对$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$的近似。

#### 2.1 欧拉法（Euler Method）

最简单的数值求解方法是前向欧拉法：
$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \Delta t \cdot \boldsymbol{v}_t(\boldsymbol{x}_t)$$

**局部截断误差分析**：泰勒展开真实解：
$$\boldsymbol{x}(t+\Delta t) = \boldsymbol{x}(t) + \Delta t \boldsymbol{x}'(t) + \frac{(\Delta t)^2}{2}\boldsymbol{x}''(t) + O((\Delta t)^3)$$

而欧拉法给出：
$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \Delta t \boldsymbol{v}_t(\boldsymbol{x}_t)$$

局部截断误差（单步误差）为：
$$\boldsymbol{e}_{local} = \frac{(\Delta t)^2}{2}\boldsymbol{x}''(t) + O((\Delta t)^3) = O((\Delta t)^2)$$

**全局误差分析**：假设从$t=0$到$t=T$需要$N = T/\Delta t$步，全局误差累积为：
$$\boldsymbol{e}_{global} = N \cdot O((\Delta t)^2) = \frac{T}{\Delta t} \cdot O((\Delta t)^2) = O(\Delta t)$$

因此欧拉法是**一阶方法**。

#### 2.2 四阶龙格-库塔法（RK4）

RK4是最常用的高阶ODE求解器，其更新公式为：
$$\begin{aligned}
\boldsymbol{k}_1 &= \boldsymbol{v}_t(\boldsymbol{x}_t) \\
\boldsymbol{k}_2 &= \boldsymbol{v}_{t+\Delta t/2}(\boldsymbol{x}_t + \frac{\Delta t}{2}\boldsymbol{k}_1) \\
\boldsymbol{k}_3 &= \boldsymbol{v}_{t+\Delta t/2}(\boldsymbol{x}_t + \frac{\Delta t}{2}\boldsymbol{k}_2) \\
\boldsymbol{k}_4 &= \boldsymbol{v}_{t+\Delta t}(\boldsymbol{x}_t + \Delta t\boldsymbol{k}_3) \\
\boldsymbol{x}_{t+\Delta t} &= \boldsymbol{x}_t + \frac{\Delta t}{6}(\boldsymbol{k}_1 + 2\boldsymbol{k}_2 + 2\boldsymbol{k}_3 + \boldsymbol{k}_4)
\end{aligned}$$

**精度分析**：RK4通过在时间步内多次采样向量场，利用加权平均来逼近真实的积分曲线。其局部截断误差为$O((\Delta t)^5)$，全局误差为$O((\Delta t)^4)$，因此是**四阶方法**。

**扩散模型中的应用**：对于概率流ODE，RK4意味着：
- 每步需要评估神经网络4次
- 相比欧拉法，RK4可以用更大的步长达到相同精度
- 在实践中，RK4通常能减少50%以上的神经网络评估次数

#### 2.3 多步法（Multistep Methods）

多步法利用之前多个时间点的信息来提高精度。一个典型的例子是**Adams-Bashforth法**。

**二阶Adams-Bashforth**（AB2）：
$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \Delta t\left(\frac{3}{2}\boldsymbol{v}_t(\boldsymbol{x}_t) - \frac{1}{2}\boldsymbol{v}_{t-\Delta t}(\boldsymbol{x}_{t-\Delta t})\right)$$

这个方法使用当前和前一步的导数信息，通过外推来估计下一步的位置。

**高阶Adams-Bashforth**（AB4）：
$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \Delta t\left(\frac{55}{24}\boldsymbol{v}_t - \frac{59}{24}\boldsymbol{v}_{t-\Delta t} + \frac{37}{24}\boldsymbol{v}_{t-2\Delta t} - \frac{9}{24}\boldsymbol{v}_{t-3\Delta t}\right)$$

多步法的优势：
- 每步只需一次函数评估（相比RK4的4次）
- 可以达到高阶精度
- 需要额外存储历史信息

### 三、确定性采样的理论保证

#### 3.1 ODE解的存在唯一性

对于概率流ODE：
$$\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}_t(\boldsymbol{x}), \quad \boldsymbol{x}(T) = \boldsymbol{x}_T$$

**Picard-Lindelöf定理**：如果向量场$\boldsymbol{v}_t(\boldsymbol{x})$满足：
1. **连续性**：$\boldsymbol{v}_t(\boldsymbol{x})$关于$(t,\boldsymbol{x})$连续
2. **Lipschitz条件**：存在常数$L$使得$\|\boldsymbol{v}_t(\boldsymbol{x}) - \boldsymbol{v}_t(\boldsymbol{y})\| \leq L\|\boldsymbol{x} - \boldsymbol{y}\|$

则ODE的解在$[0,T]$上**存在且唯一**。

**在扩散模型中的应用**：由于$\boldsymbol{v}_t(\boldsymbol{x}) = \boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$，其中：
- $\boldsymbol{f}_t(\boldsymbol{x})$通常是线性或温和非线性的
- $\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$是训练好的神经网络，在有界区域内是Lipschitz连续的

因此在实践中，ODE解的存在唯一性得到保证。

#### 3.2 逆向过程的唯一性

概率流ODE的一个关键性质是**时间可逆性**。给定终点$\boldsymbol{x}_T$，反向求解ODE：
$$\frac{d\boldsymbol{x}}{dt} = -\boldsymbol{v}_t(\boldsymbol{x}), \quad t: T \to 0$$

由ODE的唯一性定理，这个逆向过程给出唯一的轨迹$\boldsymbol{x}_t$，使得$\boldsymbol{x}_0$是从$\boldsymbol{x}_T$"解码"出的唯一样本。

**与SDE的对比**：对于原始的SDE：
$$d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}$$

即使给定相同的$\boldsymbol{x}_T$，由于布朗运动$d\boldsymbol{w}$的随机性，每次采样都会得到不同的$\boldsymbol{x}_0$。而概率流ODE消除了这种随机性，实现了确定性的编码-解码过程。

#### 3.3 采样质量的理论分析

**定理（采样一致性）**：如果神经网络$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$完美估计了score function $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$，则通过概率流ODE采样得到的分布$\hat{p}_0(\boldsymbol{x})$与数据分布$p_{data}(\boldsymbol{x})$完全一致。

**证明思路**：
1. 完美的score估计 $\Rightarrow$ 准确的向量场$\boldsymbol{v}_t(\boldsymbol{x})$
2. 准确的向量场 $\Rightarrow$ 概率流ODE产生正确的边际分布$p_t(\boldsymbol{x})$
3. 正确的边际分布演化 $\Rightarrow$ $p_0(\boldsymbol{x}) = p_{data}(\boldsymbol{x})$

**实践中的近似误差**：实际应用中存在两类误差：
1. **Score估计误差**：$\|\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t) - \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\|$
2. **数值积分误差**：ODE求解器的离散化误差

总误差可以通过以下方式控制：
- 使用更强大的神经网络架构减少score估计误差
- 使用高阶ODE求解器（如RK4）减少数值误差
- 使用自适应步长控制保证精度

### 四、概率流ODE与DDIM的深入联系

#### 4.1 DDIM的ODE形式回顾

在DDIM中，我们有：
$$\boldsymbol{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{\boldsymbol{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的}\boldsymbol{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$$

其连续极限对应的ODE为：
$$\frac{d}{dt}\left(\frac{\boldsymbol{x}(t)}{\bar{\alpha}(t)}\right) = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}(t), t)\frac{d}{dt}\left(\frac{\bar{\beta}(t)}{\bar{\alpha}(t)}\right)$$

其中$\bar{\beta}(t) = \sqrt{1-\bar{\alpha}^2(t)}$。

#### 4.2 从概率流ODE推导DDIM

考虑线性SDE：$d\boldsymbol{x} = f_t \boldsymbol{x} dt + g_t d\boldsymbol{w}$，对应的概率流ODE为：
$$\frac{d\boldsymbol{x}}{dt} = f_t \boldsymbol{x} - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$$

在DDPM/DDIM的参数化中，我们有：
$$\begin{aligned}
f_t &= \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt} = -\frac{1}{2}\frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t^2}{dt} \\
g_t^2 &= 2\bar{\alpha}_t \bar{\beta}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right) \\
\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) &= -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)}{\bar{\beta}_t}
\end{aligned}$$

代入概率流ODE：
$$\begin{aligned}
\frac{d\boldsymbol{x}}{dt} &= f_t \boldsymbol{x} + \frac{1}{2}g_t^2\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)}{\bar{\beta}_t} \\
&= \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt}\boldsymbol{x} + \bar{\alpha}_t \bar{\beta}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)}{\bar{\beta}_t} \\
&= \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt}\boldsymbol{x} + \bar{\alpha}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)
\end{aligned}$$

左边可以写成：
$$\frac{d\boldsymbol{x}}{dt} = \frac{d\bar{\alpha}_t}{dt}\frac{\boldsymbol{x}}{\bar{\alpha}_t} + \bar{\alpha}_t \frac{d}{dt}\left(\frac{\boldsymbol{x}}{\bar{\alpha}_t}\right)$$

因此：
$$\bar{\alpha}_t \frac{d}{dt}\left(\frac{\boldsymbol{x}}{\bar{\alpha}_t}\right) + \frac{d\bar{\alpha}_t}{dt}\frac{\boldsymbol{x}}{\bar{\alpha}_t} = \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt}\boldsymbol{x} + \bar{\alpha}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$$

简化得：
$$\frac{d}{dt}\left(\frac{\boldsymbol{x}}{\bar{\alpha}_t}\right) = \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$$

这正是DDIM的ODE形式！这个推导表明**DDIM是概率流ODE在线性SDE情况下的特例**。

### 五、连续归一化流（Continuous Normalizing Flows）

#### 5.1 变量变换公式

概率流ODE定义了一个时变的可逆映射$\boldsymbol{x}_t = \Phi_t(\boldsymbol{x}_0)$。根据变量变换公式，概率密度的变化由雅可比行列式决定：
$$p_t(\boldsymbol{x}_t) = p_0(\boldsymbol{x}_0) \left|\det\frac{\partial \boldsymbol{x}_0}{\partial \boldsymbol{x}_t}\right|$$

或者等价地：
$$\log p_t(\boldsymbol{x}_t) = \log p_0(\boldsymbol{x}_0) + \log\left|\det\frac{\partial \boldsymbol{x}_0}{\partial \boldsymbol{x}_t}\right|$$

#### 5.2 瞬时变化率公式

**关键问题**：如何计算雅可比行列式的对数$\log\left|\det\frac{\partial \boldsymbol{x}_0}{\partial \boldsymbol{x}_t}\right|$？

直接计算雅可比矩阵的行列式需要$O(d^3)$的计算复杂度，其中$d$是维度。对于高维数据（如图像），这是不可行的。

**连续归一化流的解决方案**：利用瞬时变化率公式（instantaneous change of variables）。

对ODE $\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}_t(\boldsymbol{x})$，概率密度的对数随时间的变化率为：
$$\frac{d}{dt}\log p_t(\boldsymbol{x}_t) = -\text{tr}\left(\frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\right)_{\boldsymbol{x}=\boldsymbol{x}_t}$$

这里$\text{tr}\left(\frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\right)$是向量场的散度$\nabla \cdot \boldsymbol{v}_t$。

**推导**：从$\log p_t(\boldsymbol{x}_t) = \log p_0(\boldsymbol{x}_0) - \log\left|\det\frac{\partial \boldsymbol{x}_t}{\partial \boldsymbol{x}_0}\right|$对$t$求导：
$$\frac{d}{dt}\log p_t(\boldsymbol{x}_t) = -\frac{d}{dt}\log\left|\det\frac{\partial \boldsymbol{x}_t}{\partial \boldsymbol{x}_0}\right|$$

利用Jacobi公式：
$$\frac{d}{dt}\log\det \boldsymbol{J} = \text{tr}\left(\boldsymbol{J}^{-1}\frac{d\boldsymbol{J}}{dt}\right)$$

其中$\boldsymbol{J} = \frac{\partial \boldsymbol{x}_t}{\partial \boldsymbol{x}_0}$。进一步推导可得：
$$\frac{d\boldsymbol{J}}{dt} = \frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\boldsymbol{J}$$

因此：
$$\frac{d}{dt}\log\det \boldsymbol{J} = \text{tr}\left(\frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\right)$$

#### 5.3 似然计算

利用瞬时变化率公式，我们可以通过积分计算似然：
$$\log p_0(\boldsymbol{x}_0) = \log p_T(\boldsymbol{x}_T) + \int_0^T \text{tr}\left(\frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\right)_{\boldsymbol{x}=\boldsymbol{x}_t} dt$$

**算法流程**：
1. 从数据$\boldsymbol{x}_0$开始
2. 沿着概率流ODE积分到$\boldsymbol{x}_T$
3. 同时积分散度项$\int_0^T \nabla \cdot \boldsymbol{v}_t dt$
4. 计算$\log p_0(\boldsymbol{x}_0) = \log p_T(\boldsymbol{x}_T) + \text{divergence term}$

**与传统Flow模型的对比**：
- 传统Flow（如Glow, RealNVP）：需要精心设计可逆且雅可比行列式易于计算的架构
- 连续归一化Flow：可以使用任意神经网络作为向量场，灵活性更高

#### 5.4 Hutchinson's Trace Estimator

计算散度$\nabla \cdot \boldsymbol{v}_t = \text{tr}\left(\frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\right)$仍需要计算雅可比矩阵的迹。对于高维情况，可以使用Hutchinson估计器：
$$\text{tr}(\boldsymbol{A}) = \mathbb{E}_{\boldsymbol{\epsilon}}[\boldsymbol{\epsilon}^T \boldsymbol{A} \boldsymbol{\epsilon}]$$

其中$\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$或$\boldsymbol{\epsilon} \sim \text{Rademacher}(\pm 1)$。

应用到散度计算：
$$\nabla \cdot \boldsymbol{v}_t = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T \frac{\partial \boldsymbol{v}_t}{\partial \boldsymbol{x}}\boldsymbol{\epsilon}\right] = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T \nabla_{\boldsymbol{x}}(\boldsymbol{v}_t \cdot \boldsymbol{\epsilon})\right]$$

这只需要一次向量-雅可比积（VJP），可以通过自动微分高效计算，复杂度为$O(d)$而非$O(d^2)$。

### 六、Flow Matching的数学基础

#### 6.1 条件流与边际流

Flow Matching是一种训练连续归一化流的新方法，它不依赖于score matching，而是直接学习向量场。

**条件概率路径**：给定数据点$\boldsymbol{x}_1 \sim p_{data}$，定义一个从噪声$\boldsymbol{x}_0 \sim p_0$（如$\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$）到数据的路径。一个简单的选择是线性插值：
$$\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1, \quad t \in [0,1]$$

对应的条件向量场为：
$$\boldsymbol{u}_t(\boldsymbol{x}|\boldsymbol{x}_1) = \frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0$$

**边际向量场**：我们希望学习的是边际向量场：
$$\boldsymbol{v}_t(\boldsymbol{x}) = \mathbb{E}_{\boldsymbol{x}_1|\boldsymbol{x}_t=\boldsymbol{x}}[\boldsymbol{u}_t(\boldsymbol{x}|\boldsymbol{x}_1)]$$

#### 6.2 Flow Matching目标函数

**定理（Flow Matching Loss）**：最小化以下损失等价于学习边际向量场：
$$\mathcal{L}_{FM}(\boldsymbol{\theta}) = \mathbb{E}_{t,\boldsymbol{x}_1,\boldsymbol{x}_0}\left[\left\|\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{u}_t(\boldsymbol{x}_t|\boldsymbol{x}_1)\right\|^2\right]$$

其中期望关于：
- $t \sim \text{Uniform}[0,1]$
- $\boldsymbol{x}_1 \sim p_{data}$
- $\boldsymbol{x}_0 \sim p_0$
- $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$

**关键优势**：
1. **简单性**：条件向量场$\boldsymbol{u}_t(\boldsymbol{x}_t|\boldsymbol{x}_1)$是已知的（如线性插值）
2. **无需score function**：不需要估计$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$
3. **高效训练**：损失函数形式简单，易于优化

#### 6.3 与扩散模型的联系

**概率流ODE可以看作Flow Matching的一种特殊情况**。在扩散模型中：
$$\boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$$

对应的条件向量场可以通过Tweedie公式推导：
$$\boldsymbol{u}_t(\boldsymbol{x}_t|\boldsymbol{x}_0) = \frac{d\bar{\alpha}_t}{dt}\frac{\boldsymbol{x}_0 - \boldsymbol{x}_t/\bar{\alpha}_t}{\bar{\beta}_t/\bar{\alpha}_t} + \text{diffusion term}$$

这与概率流ODE的形式一致，表明扩散模型本质上也是在学习一个归一化流。

### 七、理论性质的综合分析

#### 7.1 概率流ODE的唯一性与稳定性

**定理（轨迹唯一性）**：对于任意初始点$\boldsymbol{x}_T$，概率流ODE
$$\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}_t(\boldsymbol{x})$$
定义了唯一的轨迹$\boldsymbol{x}_t$，$t \in [0,T]$。

**推论**：这意味着概率流ODE定义了一个双射$\Phi: \boldsymbol{x}_T \mapsto \boldsymbol{x}_0$，它的逆映射$\Phi^{-1}$对应于时间反转的ODE。

**稳定性分析**：考虑两条初始点略有不同的轨迹$\boldsymbol{x}_t^{(1)}$和$\boldsymbol{x}_t^{(2)}$，它们的距离随时间的变化由以下微分方程控制：
$$\frac{d}{dt}\|\boldsymbol{x}_t^{(1)} - \boldsymbol{x}_t^{(2)}\| \leq L\|\boldsymbol{x}_t^{(1)} - \boldsymbol{x}_t^{(2)}\|$$

其中$L$是向量场的Lipschitz常数。这给出指数界：
$$\|\boldsymbol{x}_t^{(1)} - \boldsymbol{x}_t^{(2)}\| \leq e^{Lt}\|\boldsymbol{x}_0^{(1)} - \boldsymbol{x}_0^{(2)}\|$$

#### 7.2 与动力系统理论的联系

概率流ODE定义了一个**非自治动力系统**（时变的向量场）。从动力系统的角度：

**相流（Phase flow）**：概率流ODE定义了相空间中的一族流形变换$\phi_t: \mathbb{R}^d \to \mathbb{R}^d$，满足：
$$\phi_0 = \text{id}, \quad \frac{\partial \phi_t}{\partial t} = \boldsymbol{v}_t \circ \phi_t$$

**Liouville方程**：概率密度的演化由Liouville方程描述：
$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \boldsymbol{v}_t) = 0$$

这正是不含扩散项（$g_t=0$）的Fokker-Planck方程。

#### 7.3 数值精度与采样质量的权衡

在实践中，我们需要在三个因素之间权衡：
1. **神经网络评估次数**（NFE）：决定采样速度
2. **ODE求解器阶数**：决定单步精度
3. **最终样本质量**：通常用FID等指标衡量

**实验观察**：
- 欧拉法：需要100-1000步达到好的样本质量
- RK4：可以减少到10-50步
- 高阶自适应方法：可能进一步减少到5-10步

**理论指导**：设总积分区间为$T$，步数为$N$，ODE求解器阶数为$p$，则全局误差为：
$$\text{Error} \approx C \cdot \left(\frac{T}{N}\right)^p$$

要达到误差$\epsilon$，所需步数为：
$$N \approx C^{1/p} \cdot T \cdot \epsilon^{-1/p}$$

高阶方法（大的$p$）对$\epsilon$的依赖更弱，因此在追求高精度时更有优势。

### 八、与其他生成模型的统一视角

概率流ODE为不同类型的生成模型提供了统一的理论框架：

#### 8.1 VAE、Flow、扩散模型的统一

|  | VAE | Normalizing Flow | 扩散模型（ODE） |
|--|-----|------------------|----------------|
| 隐变量到数据 | 随机解码器 | 确定性可逆变换 | 确定性ODE |
| 似然计算 | 近似（ELBO） | 精确（变量变换） | 精确（CNF公式） |
| 灵活性 | 高（任意解码器） | 低（需可逆架构） | 高（任意向量场） |

#### 8.2 连续时间视角的价值

将生成模型视为连续时间过程的好处：
1. **理论分析更清晰**：可以利用微分方程、动力系统、随机过程的丰富理论
2. **数值方法更成熟**：ODE/SDE求解器经过数十年发展，高度优化
3. **推广更自然**：容易推广到条件生成、插值、编辑等任务

---

通过以上详细的数学推导，我们从多个角度深入理解了概率流ODE：
- **从微分方程角度**：它是SDE在零扩散极限下的退化形式
- **从动力系统角度**：它定义了相空间中的确定性流
- **从概率论角度**：它保持了边际概率分布的演化
- **从数值分析角度**：它可以通过各种高效的ODE求解器实现
- **从生成模型角度**：它统一了Flow模型和扩散模型的视角

概率流ODE的理论不仅加深了我们对扩散模型的理解，也为设计新的生成模型算法提供了坚实的数学基础。

---

### 第1部分：核心理论、公理与历史基础

#### 1.1 理论起源与历史发展

**概率流ODE的理论根源**可追溯到多个数学和物理学领域的交叉：

<div class="theorem-box">

**多领域融合**：
- **常微分方程理论** (17世纪, 牛顿/莱布尼茨)：描述确定性动力系统
- **Fokker-Planck方程** (1914, Fokker & Planck)：描述概率分布在随机过程下的演化
- **Liouville方程** (1838)：描述哈密顿系统中概率密度的演化
- **随机微分方程** (1940s, 伊藤清)：严格定义布朗运动驱动的随机过程
- **连续归一化流** (2018, Chen et al., Neural ODE)：用ODE定义可逆变换

</div>

**关键里程碑**：

1. **1914 - Fokker-Planck方程**：建立SDE与概率密度演化的联系
2. **1949 - Kolmogorov前向方程**：将F-P方程推广到一般马尔可夫过程
3. **2018 - Neural ODE (Chen et al.)**：证明ODE可以用神经网络参数化并端到端训练
4. **2019 - FFJORD (Grathwohl et al.)**：将连续归一化流应用于生成建模
5. **2020 - Score SDE (Song et al.)**：首次提出概率流ODE统一扩散模型和Flow模型
6. **2022 - Flow Matching (Lipman et al.)**：提出不依赖score的ODE训练方法

#### 1.2 数学公理与基础假设

<div class="theorem-box">

### 公理1：随机过程的马尔可夫性

扩散过程是一个马尔可夫过程，未来状态只依赖于当前状态：
$$P(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t, \boldsymbol{x}_{<t}) = P(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t)$$

</div>

<div class="theorem-box">

### 公理2：局部高斯性（Infinitesimal Gaussian）

在无穷小时间间隔内，扩散过程的增量是高斯分布：
$$\boldsymbol{x}_{t+dt} - \boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) dt + g_t d\boldsymbol{w}$$
其中$d\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{0}, dt \boldsymbol{I})$

</div>

<div class="theorem-box">

### 公理3：边际分布的唯一性

给定初始分布$p_0(\boldsymbol{x})$和SDE参数$(\boldsymbol{f}_t, g_t)$，边际分布$p_t(\boldsymbol{x})$由Fokker-Planck方程唯一确定。

</div>

<div class="theorem-box">

### 公理4：ODE的确定性与可逆性

对于确定性ODE $\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}_t(\boldsymbol{x})$，在满足Lipschitz条件下：
- **存在唯一性**：给定初值，解唯一存在
- **时间可逆性**：可以从$\boldsymbol{x}_T$精确反推$\boldsymbol{x}_0$

</div>

#### 1.3 设计哲学

概率流ODE的核心设计哲学体现为**"随机性与确定性的统一"**：

**核心思想**：
- **边际分布等价**：虽然去除了随机性，但边际概率分布与SDE完全相同
- **确定性路径**：每个噪声$\boldsymbol{x}_T$对应唯一的数据点$\boldsymbol{x}_0$
- **可逆性**：既可以编码（$\boldsymbol{x}_0 \to \boldsymbol{x}_T$）也可以解码（$\boldsymbol{x}_T \to \boldsymbol{x}_0$）

**与SDE的本质区别**：

| 维度 | SDE | 概率流ODE |
|------|-----|----------|
| 随机性 | 有（布朗运动） | 无（确定性） |
| 轨迹 | 每次采样不同 | 完全确定 |
| 可逆性 | 近似可逆 | 精确可逆 |
| 似然计算 | 困难 | 精确（CNF公式） |
| 数值求解 | SDE求解器 | 高阶ODE求解器 |

**哲学意义**：
- 概率流ODE证明了"随机过程的本质不在于单个轨迹的随机性，而在于集合层面的概率分布"
- 它揭示了扩散模型与Flow模型的深层联系，二者都是在学习一个从噪声到数据的可逆映射

---

### 第3部分：数学直觉、多角度解释与类比

#### 3.1 生活化类比

<div class="intuition-box">

### 🧠 直觉理解1：河流与洋流的类比

**SDE（随机微分方程）**：
- 想象在河流中放入大量树叶（概率分布）
- 每片树叶受到水流（漂移项$\boldsymbol{f}_t$）和随机扰动（扩散项$g_t d\boldsymbol{w}$）的双重影响
- 每片树叶的轨迹都是随机的、不可预测的
- 但大量树叶的整体分布是可预测的（由F-P方程描述）

**概率流ODE**：
- 现在想象一个理想化的洋流系统（没有小规模涡流和扰动）
- 每个水分子沿着完全确定的流线运动（ODE轨迹）
- 虽然每个分子的路径是确定的，但大量分子的整体分布仍与之前的河流相同
- **关键**：去掉了微观的随机扰动，但保持了宏观的分布演化

**类比的深意**：
- 随机性（布朗运动）可以被"吸收"到漂移项中
- 修正后的漂移项$\boldsymbol{f}_t - \frac{1}{2}g_t^2\nabla\log p_t$包含了原本的随机性信息

</div>

<div class="intuition-box">

### 🧠 直觉理解2：GPS导航的确定性路线

**问题**：如何从家（$\boldsymbol{x}_0$）到机场（$\boldsymbol{x}_T$）？

**方案A（SDE）**：
- 每次开车都随机选择路线
- 有时走高速，有时走省道，甚至绕远路
- 但平均来说，大多数情况都能到达机场
- 每次路线不同，不可复现

**方案B（概率流ODE）**：
- GPS给出一条确定的最优路线
- 每次都走相同的路径
- 完全可复现：给定起点，终点唯一确定
- 反过来，从机场也能原路返回

**为什么方案B也能生成多样的样本？**
- 虽然单条路径是确定的，但不同的起点（不同的$\boldsymbol{x}_T \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$）会到达不同的终点（不同的样本）
- 多样性来自初始噪声的多样性，而非路径本身的随机性

</div>

<div class="intuition-box">

### 🧠 直觉理解3：音乐与乐谱

**SDE（演奏）**：
- 爵士乐即兴演奏：每次演奏都有随机的变化
- 相同的乐谱，每次演奏都略有不同
- 无法精确复现上一次的演奏

**概率流ODE（录音）**：
- 古典音乐的标准演奏：严格遵循乐谱
- 每次播放录音，声音完全相同
- 完全可复现

**哲学**：
- 两种方式都能生成美妙的音乐（样本）
- ODE像是"给SDE录音"，固定了原本随机的部分

</div>

#### 3.2 几何意义

**几何视角1：相空间中的流**

<div class="intuition-box">

在$d$维数据空间$\mathbb{R}^d$中：

**SDE的几何图景**：
- 概率密度$p_t(\boldsymbol{x})$像一团"云"在空间中演化
- 每个粒子沿着随机轨迹运动（布朗运动）
- 云的边界是模糊的（扩散效应）

**概率流ODE的几何图景**：
- 概率密度仍像一团"云"
- 但现在每个粒子沿着确定的流线运动
- 云的形状和位置演化完全相同，但内部粒子的运动是确定的
- 流线不相交（ODE解的唯一性）

**可视化**（2维情况）：
- 想象在$(x_1, x_2)$平面上画出向量场$\boldsymbol{v}_t(\boldsymbol{x})$
- 每个点都有一个箭头，表示该点处的流动方向
- ODE的解就是沿着箭头方向移动的曲线（积分曲线）
- 不同起点的曲线永不相交

</div>

**几何视角2：拉伸与压缩**

<div class="intuition-box">

ODE定义的流会**拉伸和压缩**空间：

- 在某些区域，流线发散（体积膨胀） → 概率密度降低
- 在某些区域，流线汇聚（体积收缩） → 概率密度升高

这由散度$\nabla \cdot \boldsymbol{v}_t$控制：
- $\nabla \cdot \boldsymbol{v}_t > 0$：膨胀区域，密度降低
- $\nabla \cdot \boldsymbol{v}_t < 0$：收缩区域，密度升高
- $\nabla \cdot \boldsymbol{v}_t = 0$：保体积流（如哈密顿流）

**类比**：
- 像是空间中的"潮汐"：某处涨潮（密度升高），某处退潮（密度降低）
- 总体积（概率总和）保持为1

</div>

#### 3.3 多角度理解

**📊 概率论视角**

<div class="intuition-box">

从概率论看，概率流ODE实现了：

**推送前向测度（Push-forward measure）**：
- 给定初始分布$p_0$和ODE流$\Phi_t$，任意时刻的分布为：
  $$p_t = (\Phi_t)_{\#} p_0$$
- 这是一个确定性的分布变换（vs SDE的随机变换）

**条件期望的几何实现**：
- ODE的向量场$\boldsymbol{v}_t(\boldsymbol{x})$可以理解为：
  $$\boldsymbol{v}_t(\boldsymbol{x}) = \mathbb{E}[\boldsymbol{u}_t|\boldsymbol{x}_t=\boldsymbol{x}]$$
  其中$\boldsymbol{u}_t$是条件向量场
- 这是对所有可能的随机轨迹求平均后的"期望轨迹"

</div>

**🔄 动力系统视角**

<div class="intuition-box">

从动力系统理论看：

**自治系统 vs 非自治系统**：
- 标准ODE：$\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}(\boldsymbol{x})$（向量场不随时间变化）
- 概率流ODE：$\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}_t(\boldsymbol{x})$（时变向量场）

**不变流形（Invariant Manifold）**：
- 数据流形$\mathcal{M}_{data}$是ODE的吸引子
- 随着时间从$T$到$0$，所有轨迹最终收敛到数据流形附近
- 这解释了为什么ODE能"去噪"：本质是向数据流形投影

**庞加莱截面（Poincaré Section）**：
- 在不同时刻$t$观察空间分布，相当于在动力系统中取截面
- 每个截面上的概率分布由F-P方程联系

</div>

**📐 测度论视角**

<div class="intuition-box">

从测度论角度：

**Radon-Nikodym导数**：
- ODE定义了概率测度之间的变换：$p_t = T_t^{\#} p_0$
- Radon-Nikodym导数（密度变化率）由雅可比行列式给出：
  $$\frac{dp_t}{dp_0}(\boldsymbol{x}) = \left|\det \frac{\partial \Phi_t^{-1}}{\partial \boldsymbol{x}}(\boldsymbol{x})\right|$$

**绝对连续性**：
- 如果$p_0$关于Lebesgue测度绝对连续，则对所有$t$，$p_t$也绝对连续
- 这保证了ODE不会产生奇异分布（如Dirac测度）

**测度保持性**：
- ODE流$\Phi_t$是测度保持的当且仅当$\nabla \cdot \boldsymbol{v}_t = 0$
- 对于一般的概率流ODE，$\nabla \cdot \boldsymbol{v}_t \neq 0$，所以测度会变化

</div>

**🔥 热力学视角**

<div class="intuition-box">

类比热力学：

**熵的演化**：
- 信息熵：$H[p_t] = -\int p_t(\boldsymbol{x}) \log p_t(\boldsymbol{x}) d\boldsymbol{x}$
- 对于扩散过程（SDE），熵单调增加（第二定律）
- 对于概率流ODE，熵可能增加或减少（可逆过程）

**自由能下降**：
- 从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的过程可以看作自由能的梯度流
- 向量场$\boldsymbol{v}_t$指向自由能下降的方向

**可逆性与不可逆性**：
- SDE：不可逆过程（有扩散项，熵增）
- 概率流ODE：可逆过程（确定性，无熵增）

</div>

---

### 第4部分：方法论变体、批判性比较与优化

#### 4.1 主流方法对比表

| 方法 | 核心思想 | 优点 | **缺陷** | **优化方向** |
|------|---------|------|---------|-------------|
| **SDE采样（原始扩散）** | 随机微分方程，每步加噪声 | ✅ 理论完备<br>✅ 样本多样性高<br>✅ 训练稳定 | ❌ **采样慢**（需1000步）<br>❌ 不可复现<br>❌ 似然计算困难 | ✅ 高阶SDE求解器<br>✅ 自适应步长<br>✅ 重要性采样 |
| **概率流ODE（本文）** | 常微分方程，确定性流 | ✅ **采样快**（可用10-50步）<br>✅ 完全可复现<br>✅ 精确似然计算<br>✅ 可逆编辑 | ❌ **多样性略降**<br>❌ 对初始噪声敏感<br>❌ 高阶求解器开销大 | ✅ 混合ODE-SDE<br>✅ 自适应求解器<br>✅ 更好的向量场参数化 |
| **DDIM** | 非马尔可夫链，跳步采样 | ✅ 加速明显<br>✅ 可插值编辑 | ❌ **步长选择启发式**<br>❌ 理论不如ODE完备<br>❌ 大步长时质量下降 | ✅ 自适应步长<br>✅ 理论化为ODE |
| **Flow Matching** | 直接学习向量场 | ✅ 训练简单<br>✅ 无需score matching | ❌ **路径选择影响大**<br>❌ 理论还不成熟 | ✅ 最优传输路径<br>✅ 多模态路径 |
| **DPM-Solver** | 高阶ODE求解器 | ✅ 10步高质量<br>✅ 理论保证 | ❌ **实现复杂**<br>❌ 对噪声schedule敏感 | ✅ 自适应阶数<br>✅ 误差估计 |

#### 4.2 概率流ODE - 批判性分析

<div class="analysis-box">

### **核心缺陷**

**缺陷1：样本多样性略有降低**

**问题描述**：
- 相比SDE采样，ODE采样的样本略微"保守"
- 在某些任务上，生成样本的多样性（diversity）和覆盖率（coverage）稍低

**根本原因**：
- SDE的随机性提供了"探索"机制，能发现多个局部最优
- ODE是确定性的，从噪声$\boldsymbol{x}_T$到数据$\boldsymbol{x}_0$的路径是唯一的
- 多样性完全依赖于初始噪声的分布

**定量影响**：
- 文献报告：在某些数据集上，ODE采样的FID略高于SDE（1-3分）
- Inception Score略低于SDE（~0.1-0.2分）
- 覆盖率（Coverage）降低5%-10%

**示例数据**（ImageNet 64×64）：
| 采样方法 | FID ↓ | Inception Score ↑ | NFE（函数评估次数） |
|---------|-------|-------------------|-------------------|
| SDE (1000步) | 2.92 | 195.8 | 1000 |
| 概率流ODE (RK45) | 3.21 | 194.3 | 42 |
| 概率流ODE (欧拉) | 4.15 | 190.2 | 100 |

---

**缺陷2：对score估计误差更敏感**

**问题描述**：
- 概率流ODE的向量场依赖于score function $\nabla\log p_t$
- 相比SDE，ODE对score的估计误差更敏感

**根本原因**：
- SDE中的布朗运动提供了"自我修正"机制（类似模拟退火）
- ODE是确定性的，误差会沿轨迹累积
- 数学上：ODE解对初值和向量场的连续依赖性（Lipschitz）

**定量影响**：
- 如果score估计误差为$\epsilon$，SDE采样的最终误差约为$O(\sqrt{\epsilon})$
- ODE采样的最终误差约为$O(\epsilon \cdot T)$（$T$是积分时间）

**数学分析**：
假设score估计误差有界：$\|\boldsymbol{s}_{\theta}(\boldsymbol{x}, t) - \nabla\log p_t(\boldsymbol{x})\| \leq \epsilon$

则ODE解的误差满足（Gr��nwall不等式）：
$$\|\boldsymbol{x}(t) - \boldsymbol{x}^*(t)\| \leq \epsilon \cdot T \cdot e^{LT}$$
其中$L$是向量场的Lipschitz常数。

---

**缺陷3：高阶求解器的计算开销**

**问题描述**：
- 为了达到高质量，需要使用高阶ODE求解器（如RK4、DPM-Solver）
- 每步需要多次神经网络评估

**根本原因**：
- 欧拉法只有$O(\Delta t)$精度，需要很多步
- RK4等高阶方法有$O((\Delta t)^4)$精度，但每步需要4次函数评估

**定量影响**：
- 欧拉法：需要100-200步达到FID < 5.0
- RK4：需要10-50步，但每步4次评估 → 总评估次数40-200次
- DPM-Solver：可以10-20步，但实现复杂

**计算开销对比**：
| 求解器 | 阶数 | 每步评估次数 | 总步数（FID<5） | 总NFE |
|--------|-----|------------|---------------|------|
| 欧拉法 | 1 | 1 | 200 | 200 |
| RK4 | 4 | 4 | 20 | 80 |
| DPM-Solver | 2-3 | 2-3 | 10-15 | 20-45 |

---

### **优化方向**

**优化1：混合采样（Hybrid ODE-SDE）**

**策略**：结合ODE的高效性和SDE的多样性

**公式**（可调方差的SDE）：
$$d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}$$

**实践策略**：
- 前期（$t$ large）：使用ODE（$\sigma_t = 0$）快速逼近
- 后期（$t$ small）：使用SDE（$\sigma_t > 0$）增加多样性

**效果**（Stable Diffusion实验）：
- FID从3.21降至2.95（混合方案）
- 采样步数从50降至30（混合可以更激进）
- Inception Score提升1.5分

---

**优化2：自适应ODE求解器**

**策略**：根据局部误差动态调整步长

**算法**（RK45 with adaptive step size）：
1. 用4阶和5阶RK方法分别估计$\boldsymbol{x}_{t+\Delta t}$
2. 计算两者差异作为误差估计：$\epsilon_{local} = \|\boldsymbol{x}^{(4)} - \boldsymbol{x}^{(5)}\|$
3. 如果$\epsilon_{local} > tol$，减小$\Delta t$；如果$\epsilon_{local} \ll tol$，增大$\Delta t$

**优势**：
- 在平滑区域（score变化慢）：大步长
- 在复杂区域（score变化快）：小步长
- 自动平衡精度和效率

**效果**：
- 相比固定步长，总NFE减少30%-50%
- 精度更稳定（自动控制误差在容忍范围内）

---

**优化3：改进的向量场参数化**

**策略1：速度预测 (Velocity Prediction)**
- 不直接预测score，而是预测速度$\boldsymbol{v}_t(\boldsymbol{x})$
- 公式：神经网络$\boldsymbol{v}_{\theta}(\boldsymbol{x}, t)$直接逼近ODE右端

**优势**：
- 减少从$\boldsymbol{\epsilon}_{\theta}$到$\boldsymbol{v}_t$的转换误差
- 训练更稳定（速度场比score更平滑）

**策略2：Flow Matching目标**
- 损失函数：
  $$\mathcal{L} = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{x}_1}\left[\|\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\|^2\right]$$
- 无需估计score，直接学习条件向量场

**效果**：
- 训练速度提升2-3倍
- 采样质量与score-based持平或略好

---

**优化4：轨迹蒸馏（Trajectory Distillation）**

**策略**：将多步ODE蒸馏到少步模型

**方法**（Progressive Distillation, Salimans & Ho 2022）：
1. 训练一个teacher模型（如50步ODE）
2. 训练一个student模型（25步）来匹配teacher的输出
3. 递归：25步 → 12步 → 6步 → 3步

**公式**：
$$\mathcal{L}_{distill} = \mathbb{E}_{\boldsymbol{x}_t}\left[\|\boldsymbol{x}_0^{(student)} - \boldsymbol{x}_0^{(teacher)}\|^2\right]$$

**效果**：
- 最终可以用4-8步达到50步ODE的质量
- 推理速度提升6-12倍
- FID损失<0.5分

---

**优化5：更优的噪声schedule**

**策略**：优化$\bar{\alpha}_t, \bar{\beta}_t$的时间演化曲线

**方法**：
- **Cosine schedule** (Nichol & Dhariwal 2021)：
  $$\bar{\alpha}_t^2 = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$
  其中$s$是小的偏移量（如0.008）

- **可学习的schedule**：
  将$\bar{\alpha}_t$参数化为神经网络，端到端优化

**效果**：
- Cosine schedule相比线性schedule，FID降低0.5-1.0分
- ODE采样步数可以减少20%-30%

</div>

#### 4.3 与其他方法的对比

<div class="analysis-box">

### DDIM vs 概率流ODE

**相同点**：
- 都是确定性采样
- 都可以实现快速采样（10-50步）
- 都支持可逆编辑

**不同点**：
| 方面 | DDIM | 概率流ODE |
|------|------|----------|
| 理论基础 | 非马尔可夫假设 | Fokker-Planck方程 |
| 推导方式 | 从离散过程出发 | 从连续SDE出发 |
| 通用性 | 仅适用于线性SDE | 适用于任意SDE |
| 似然计算 | 困难 | 精确（CNF公式） |

**结论**：概率流ODE是DDIM的理论升级版

---

### SDE vs 概率流ODE

**何时使用SDE**：
- 追求最高生成质量（FID、IS）
- 需要最大化样本多样性
- 计算资源充足（可以承受1000步）

**何时使用概率流ODE**：
- 需要快速采样（实时应用）
- 需要可复现性（如图像编辑、插值）
- 需要精确似然计算（如用于评估或压缩）
- 需要可逆性（编码-解码）

**混合方案**：
- 大多数情况下，使用混合ODE-SDE（$\sigma_t$可调）是最佳选择
- 灵活在速度和质量之间权衡

</div>

---

### 第5部分：学习路线图与未来展望

#### 5.1 学习路线图

**必备前置知识**

**数学基础**：
- 常微分方程（ODE）：解的存在唯一性、数值求解方法
- 偏微分方程（PDE）：Fokker-Planck方程、热方程
- 随机过程：布朗运动、伊藤积分、SDE基础
- 测度论：Radon-Nikodym导数、推送前向测度

**物理/工程背景**：
- 统计力学：F-P方程的物理意义
- 流体力学：Liouville方程、连续性方程
- 动力系统：相流、不变流形

**机器学习基础**：
- 扩散模型基础：DDPM、DDIM、Score-based SDE
- 生成模型：VAE、Normalizing Flow、GAN
- 神经ODE：连续深度学习、adjoint method

**推荐学习顺序**：

1. **理解SDE框架**（《生成扩散模型漫谈（五）》）
2. **学习Fokker-Planck方程**（本文第一部分）
3. **理解等价SDE族**（本文第二部分）
4. **推导概率流ODE**（本文核心）
5. **学习CNF和Flow Matching**（扩展阅读）
6. **实践数值求解方法**（代码实现）

---

**核心论文列表（按时间顺序）**

**理论奠基**：
1. Fokker (1914), Planck (1917) - "Fokker-Planck方程"
2. Kolmogorov (1931) - "前向方程与后向方程"
3. Itô (1944) - "随机微分方程理论"

**深度学习时代**：
4. Chen et al. (2018) - "Neural Ordinary Differential Equations" ⭐
5. Grathwohl et al. (2019) - "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"

**扩散模型核心**：
6. Song et al. (2020) - "Score-Based Generative Modeling through Stochastic Differential Equations" ⭐⭐⭐
7. Nichol & Dhariwal (2021) - "Improved DDPM"

**加速与改进**：
8. Lu et al. (2022) - "DPM-Solver: Fast ODE Solver for Diffusion Probabilistic Models" ⭐
9. Salimans & Ho (2022) - "Progressive Distillation for Fast Sampling"
10. Lipman et al. (2022) - "Flow Matching for Generative Modeling" ⭐

---

#### 5.2 研究空白与未来方向

#### **方向1：理论层面 - ODE轨迹的几何结构**

**研究空白**：
- 当前缺乏对概率流ODE轨迹几何性质的深入理解
- 不清楚什么样的向量场能产生"更好"的轨迹
- 轨迹的曲率、扭转与生成质量的关系未知

**具体研究问题**：

1. **问题**：概率流ODE的轨迹是否具有特定的几何结构（如测地线）？
   - **挑战**：高维空间中轨迹的几何分析困难
   - **潜在方法**：
     - 利用黎曼几何工具分析数据流形上的测地线
     - 研究轨迹的曲率张量与生成质量的关系
     - 探索是否存在"最短路径"采样
   - **潜在意义**：找到最优轨迹可能进一步减少采样步数

2. **问题**：向量场的正则性（平滑度）如何影响采样效率？
   - **已知**：更平滑的向量场允许更大的步长
   - **未知**：如何量化"平滑度"与数值误差的关系
   - **潜在方法**：
     - 定义向量场的Sobolev范数作为平滑度指标
     - 推导平滑度、步长、误差三者的理论界
     - 设计鼓励平滑向量场的训练目标

3. **问题**：能否设计"直线轨迹"的ODE？
   - **动机**：直线是最简单的路径，数值求解最高效
   - **探索方向**：
     - Flow Matching with Optimal Transport（最优传输）
     - Rectified Flow：通过蒸馏将弯曲轨迹拉直
     - 分析直线轨迹的可实现条件

**优化方向**：
- 引入几何惩罚项（如轨迹曲率）到训练损失中
- 使用Riemannian metrics指导ODE设计
- 借鉴最优控制理论设计能量最优的路径

**量化目标**：
- 推导轨迹曲率与数值误差的上界：$\text{Error} \leq C \cdot \kappa \cdot (\Delta t)^p$
- 设计"直线度"$> 0.9$的轨迹（用某种归一化度量）
- 实现采样步数减少到5步以内，FID < 5.0

---

#### **方向2：效率层面 - 极致加速与一步生成**

**研究空白**：
- 当前最好的方法仍需10-20步
- 一步生成方法（如Consistency Models）质量有明显差距
- 缺乏理论指导"最少需要多少步"

**具体研究问题**：

1. **问题**：概率流ODE采样的步数下界是多少？
   - **理论问题**：给定容忍误差$\epsilon$，最少需要多少步？
   - **潜在方法**：
     - 利用信息论建立采样复杂度下界
     - 分析ODE的Kolmogorov复杂度
     - 研究数据分布的内在维度与步数的关系
   - **实践价值**：知道理论极限，指导算法设计

2. **问题**：如何实现高质量的一步生成？
   - **现状**：Consistency Models一步生成FID约10-15（vs 多步的3-5）
   - **优化方向**：
     - **多阶段一步生成**：粗到细的级联模型
     - **自蒸馏**：teacher多步 → student一步 → 递归
     - **混合方法**：一步生成初稿 + 少量步骤精修
     - **更强的架构**：Transformer替代UNet作为向量场网络
   - **量化目标**：一步生成FID < 5.0

3. **问题**：并行采样是否可能？
   - **挑战**：ODE是自回归的，似乎无法并行
   - **可能的突破**：
     - Picard迭代的并行化
     - 使用隐式方法（如隐式欧拉、BDF）
     - 波形松弛（Waveform Relaxation）技术
   - **潜在价值**：在多GPU上并行，进一步加速

**优化方向**：
- 开发专用的神经网络架构（专门为ODE求解优化）
- 研究模型量化（INT8、FP16）对ODE采样的影响
- 探索神经架构搜索（NAS）自动发现高效架构

**量化目标**：
- 5步采样达到FID < 3.0（当前SOTA约10-20步）
- 一步生成FID < 5.0（当前约10-15）
- 在A100 GPU上实现512×512图像生成<0.1秒（当前约1-2秒）
- 移动端（如iPhone）实时生成256×256图像

---

#### **方向3：应用层面 - 可控生成与多模态**

**研究空白**：
- 概率流ODE在条件生成中的应用不够深入
- 多模态（图像+文本）的联合ODE设计不明确
- 可控性（如编辑、插值）的理论保证不足

**具体研究问题**：

1. **问题**：如何在ODE框架下实现精准的图像编辑？
   - **需求**：如"只改变人物表情，保持其他不变"
   - **优化方向**：
     - **语义空间编辑**：在隐空间$\boldsymbol{x}_T$进行语义操作，然后ODE解码
     - **条件ODE**：$\boldsymbol{v}_t(\boldsymbol{x}|\boldsymbol{c})$，$\boldsymbol{c}$是编辑指令
     - **轨迹干预**：在特定时刻$t^*$修改轨迹，引导到目标
   - **挑战**：如何保证非编辑区域不变？

2. **问题**：多模态数据的联合概率流ODE？
   - **场景**：图像+文本，音频+视频的联合生成
   - **优化方向**：
     - **模态特定的噪声schedule**：不同模态用不同的$\bar{\alpha}_t$
     - **跨模态耦合**：在向量场中加入跨模态注意力
     - **统一隐空间**：将多模态映射到共同的隐空间，在该空间中定义ODE
   - **挑战**：如何对齐不同模态的"时间"（如文本的离散性 vs 图像的连续性）？

3. **问题**：视频生成中的时空一致性？
   - **当前问题**：逐帧生成视频会闪烁（temporal inconsistency）
   - **优化方向**：
     - **4D ODE**：在$(x, y, z, t)$四维空间中定义ODE
     - **轨迹共享**：相邻帧的ODE轨迹部分共享
     - **因果约束**：确保ODE满足时间因果性
   - **量化目标**：视频FVD（Fréchet Video Distance）< 200

**优化方向**：
- 开发用户友好的编辑接口（如brush-based editing with ODE）
- 研究ODE的可解释性：可视化轨迹，理解每个时间步在做什么
- 探索ODE与其他生成模型的结合（如ODE + GAN）

**量化目标**：
- 图像编辑精准度 > 90%（用户评估）
- 多模态生成在各模态上的质量都达到单模态水平
- 视频生成FVD < 200（当前SOTA约300-500）

---

#### **方向4：鲁棒性层面 - 分布外数据与对抗性**

**研究空白**：
- 概率流ODE对分布外数据的行为未知
- 对对抗性扰动的鲁棒性未研究
- ODE的"安全性"（是否会生成有害内容）未分析

**具体研究问题**：

1. **问题**：分布外的初始噪声$\boldsymbol{x}_T$会导致什么？
   - **场景**：如果$\boldsymbol{x}_T$不是标准正态分布，而是有偏差
   - **现象**：可能生成低质量或异常的样本
   - **优化方向**：
     - **噪声净化**：在采样前对$\boldsymbol{x}_T$进行检测和修正
     - **鲁棒性训练**：在训练时加入分布外噪声
     - **异常检测**：实时监测ODE轨迹是否偏离正常区域
   - **理论分析**：推导ODE对初值扰动的敏感度界

2. **问题**：对抗攻击如何影响概率流ODE？
   - **攻击场景**：给$\boldsymbol{x}_T$加入小的对抗性扰动$\delta$
   - **目标**：生成特定的（可能有害的）内容
   - **防御方法**：
     - **Randomized smoothing**：在$\boldsymbol{x}_T$加入额外的噪声
     - **Certified defense**：提供provable的鲁棒性保证
     - **Adversarial training**：训练时加入对抗样本
   - **量化目标**：在$\ell_2$球$\|\delta\| \leq 0.1$内，攻击成功率<10%

3. **问题**：如何避免生成有害内容？
   - **问题**：无条件生成可能产生暴力、色情等内容
   - **优化方向**：
     - **Safe ODE设计**：在向量场中加入"安全引导"项
     - **轨迹监控**：实时检测ODE轨迹是否进入"危险区域"
     - **后处理过滤**：生成后用分类器过滤
   - **理论保证**：能否设计provably safe的ODE？

**优化方向**：
- 借鉴控制论中的鲁棒控制理论
- 使用Lyapunov函数分析ODE稳定性
- 开发形式化验证方法（formal verification）

**量化目标**：
- 对$\ell_2$半径0.1的对抗扰动，准确率下降<5%
- 有害内容生成率<0.1%（当前未控制的模型约1-5%）
- 分布外数据的FID增加<10%

---

#### **方向5：新型架构 - 物理启发的ODE设计**

**研究空白**：
- 当前ODE设计主要基于经验和score matching
- 缺乏物理原理指导的ODE架构
- 未充分利用物理守恒律

**具体研究问题**：

1. **问题**：能否设计满足物理守恒律的ODE？
   - **物理约束**：
     - **能量守恒**：哈密顿ODE，$H(\boldsymbol{x}, t) = \text{const}$
     - **辛结构**：保持相空间的辛形式
     - **体积保持**：$\nabla \cdot \boldsymbol{v}_t = 0$
   - **潜在优势**：
     - 更好的数值稳定性
     - 更准确的长时间积分
     - 物理可解释性
   - **挑战**：如何将这些约束融入神经网络？

2. **问题**：基于最优传输的ODE设计？
   - **思想**：从$p_0$到$p_1$的最优传输定义了一个自然的ODE
   - **Monge-Ampère方程**：
     $$\det(D^2 \phi) = \frac{p_0}{p_1 \circ \nabla \phi}$$
     其中$\phi$是传输势函数
   - **向量场**：$\boldsymbol{v}_t = \nabla \phi$
   - **优势**：最短路径（在Wasserstein度量下）
   - **挑战**：高维时Monge-Ampère方程难解

3. **问题**：量子启发的ODE？
   - **灵感**：薛定谔方程、路径积分
   - **可能的设计**：
     - 使用量子势（quantum potential）引导ODE
     - 借鉴量子隧穿效应处理多模态分布
     - 利用波函数坍缩类比理解采样过程

**优化方向**：
- 将物理约束编码为正则化项或网络架构约束
- 使用物理信息神经网络（PINNs）框架
- 借鉴计算物理中的数值方法（如保结构算法）

**量化目标**：
- 设计保辛ODE，长时间积分误差不增长
- 基于最优传输的ODE，采样步数减少到3-5步
- 物理约束的ODE在科学数据（如分子动力学）上取得SOTA

---

#### **潜在应用场景**

**科学计算**：
- **分子生成**：基于ODE的药物设计（保持化学约束）
- **蛋白质折叠**：在SE(3)流形上的等变ODE
- **PDE求解**：用扩散ODE作为PDE求解器的先验

**艺术创作**：
- **可控艺术生成**：精准控制风格、内容的ODE编辑
- **音乐生成**：连续时间的音乐ODE
- **3D场景生成**：神经辐射场（NeRF）+ ODE

**工业应用**：
- **医学影像**：去噪、超分辨率（确定性ODE保证可重复性）
- **自动驾驶**：场景生成用于仿真测试
- **数字人**：实时人脸生成（需要极快的ODE采样）

---

### 总结

概率流ODE是扩散模型理论的重要里程碑，它：

1. **统一了生成模型**：将扩散模型与Flow模型联系起来
2. **加速了采样**：从1000步减少到10-50步
3. **实现了可逆性**：支持精确编码-解码和编辑
4. **提供了理论工具**：精确似然计算、CNF框架

**核心要点**：
- Fokker-Planck方程是桥梁：连接SDE与概率密度演化
- 等价SDE族：不同方差的SDE产生相同的边际分布
- 概率流ODE：$\sigma_t = 0$的极端情况，确定性但保持分布
- 数值方法至关重要：高阶ODE求解器大幅提升效率

**未来最值得关注的方向**：
1. **5步以内的高质量采样**：接近理论极限
2. **一步生成的突破**：质量达到多步水平
3. **物理启发的ODE设计**：利用守恒律和最优传输
4. **多模态与可控性**：实用化的关键
5. **鲁棒性与安全性**：大规模部署的前提

