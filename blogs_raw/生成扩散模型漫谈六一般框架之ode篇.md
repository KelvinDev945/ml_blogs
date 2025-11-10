---
title: 生成扩散模型漫谈（六）：一般框架之ODE篇
slug: 生成扩散模型漫谈六一般框架之ode篇
date: 2022-08-08
tags: flow模型, 微分方程, 生成模型, DDPM, 扩散
status: pending
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

