---
title: 生成扩散模型漫谈（十二）：“硬刚”扩散ODE
slug: 生成扩散模型漫谈十二硬刚扩散ode
date: 2022-09-28
tags: 微分方程, 生成模型, 扩散, 生成模型, attention
status: pending
---

# 生成扩散模型漫谈（十二）：“硬刚”扩散ODE

**原文链接**: [https://spaces.ac.cn/archives/9280](https://spaces.ac.cn/archives/9280)

**发布日期**: 

---

在[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)中，我们从SDE的角度理解了生成扩散模型，然后在[《生成扩散模型漫谈（六）：一般框架之ODE篇》](/archives/9228)中，我们知道SDE对应的扩散模型中，实际上隐含了一个ODE模型。无独有偶，在[《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](/archives/9181)中我们也知道原本随机采样的DDPM模型中，也隐含了一个确定性的采样过程DDIM，它的连续极限也是一个ODE。

细想上述过程，可以发现不管是“DDPM→DDIM”还是“SDE→ODE”，都是从随机采样模型过渡到确定性模型，而如果我们一开始的目标就是ODE，那么该过程未免显得有点“迂回”了。在本文中，笔者尝试给出ODE扩散模型的直接推导，并揭示了它与雅可比行列式、热传导方程等内容的联系。

## 微分方程 #

像GAN这样的生成模型，它本质上是希望找到一个确定性变换，能将从简单分布（如标准正态分布）采样出来的随机变量，变换为特定数据分布的样本。flow模型也是生成模型之一，它的思路是反过来，先找到一个能将数据分布变换简单分布的可逆变换，再求解相应的逆变换来得到一个生成模型。

传统的flow模型是通过设计精巧的耦合层（参考“[细水长flow](/search/%E7%BB%86%E6%B0%B4%E9%95%BFflow)”系列）来实现这个可逆变换，但后来大家就意识到，其实通过微分方程也能实现这个变换，并且理论上还很优雅。基于“神经网络 + 微分方程”做生成模型等一系列研究，构成了被称为“神经ODE”的一个子领域。

考虑$\boldsymbol{x}_t\in\mathbb{R}^d$上的一阶（常）微分方程（组）  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt}=\boldsymbol{f}_t(\boldsymbol{x}_t)\label{eq:ode}\end{equation}  
假设$t\in[0, T]$，那么给定$\boldsymbol{x}_0$，（在比较容易实现的条件下）我们可以确定地求解出$\boldsymbol{x}_T$，也就是说该微分方程描述了从$\boldsymbol{x}_0$到$\boldsymbol{x}_T$的一个变换。特别地，该变换还是可逆的，即可以逆向求解该微分方程，得到从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的变换。所以说，微分方程本身就是构建可逆变换的一个理论优雅的方案。

## 雅可比行列式 #

跟之前的扩散模型一样，在这篇文章中，我们将$\boldsymbol{x}_0$视为一个数据样本，而将$\boldsymbol{x}_T$视为简单分布的样本，我们希望通过微分方程，来实现从数据分布到简单分布的变换。

首先，我们从离散化的角度来理解微分方程$\eqref{eq:ode}$：  
\begin{equation}\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\label{eq:ode-diff}\end{equation}  
由于是确定性变换，所以我们有  
\begin{equation}p_t(\boldsymbol{x}_t) d\boldsymbol{x}_t = p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) d\boldsymbol{x}_{t+\Delta t} = p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) \left| \frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t} \right| d\boldsymbol{x}_t\end{equation}  
这里的$\frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t}$表示变换的雅可比矩阵，$|\cdot|$代表行列式的绝对值。直接对式$\eqref{eq:ode-diff}$两边求偏导，我们就得到  
\begin{equation}\frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t} = \boldsymbol{I} + \frac{\partial \boldsymbol{f}_t(\boldsymbol{x}_t)}{\partial \boldsymbol{x}_t}\Delta t\end{equation}  
根据[《行列式的导数》](/archives/2383)一文，我们就有  
\begin{equation}\left|\frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t}\right| \approx 1 + \text{Tr}\,\frac{\partial \boldsymbol{f}_t(\boldsymbol{x}_t)}{\partial \boldsymbol{x}_t}\Delta t = 1 + \nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\approx e^{\nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t}\end{equation}  
于是我们可以写出  
\begin{equation}\log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - \log p_t(\boldsymbol{x}_t) \approx -\nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\label{eq:approx-ode}\end{equation}

## 泰勒近似 #

假设$p_t(\boldsymbol{x}_t)$是一簇随着参数$t$连续变化的分布的概率密度函数，其中$p_0(\boldsymbol{x}_0)$是数据分布，$p_T(\boldsymbol{x}_T)$则是简单分布，当$\Delta t$和$\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t$都较小时，我们有一阶泰勒近似  
\begin{equation}\log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - \log p_t(\boldsymbol{x}_t) \approx (\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) + \Delta t\frac{\partial}{\partial t}\log p_t(\boldsymbol{x}_t)\end{equation}  
代入式$\eqref{eq:ode-diff}$的$\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t$，然后对照式$\eqref{eq:approx-ode}$，可以得到$\boldsymbol{f}_t(\boldsymbol{x}_t)$所满足的方程  
\begin{equation}-\nabla_{\boldsymbol{x}_t}\cdot \boldsymbol{f}_t(\boldsymbol{x}_t) = \boldsymbol{f}_t(\boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) + \frac{\partial}{\partial t}\log p_t(\boldsymbol{x}_t)\label{eq:ode-f-eq}\end{equation}  
换句话说，满足该方程的任意$\boldsymbol{f}_t(\boldsymbol{x}_t)$，都可以用来构造一个微分方程$\eqref{eq:ode}$，通过求解它来实现数据分布和简单分布之间的变换。我们也可以将它整理得  
\begin{equation}\frac{\partial}{\partial t} p_t(\boldsymbol{x}_t) = - \nabla_{\boldsymbol{x}_t}\cdot\Big(\boldsymbol{f}_t(\boldsymbol{x}_t) p_t(\boldsymbol{x}_t)\Big)\label{eq:ode-f-eq-fp}\end{equation}  
它其实就是[《生成扩散模型漫谈（六）：一般框架之ODE篇》](/archives/9228#F-P%E6%96%B9%E7%A8%8B)介绍的“Fokker-Planck方程”在$g_t=0$时的特例。

## 热传导方程 #

我们考虑如下格式的解  
\begin{equation}\boldsymbol{f}_t(\boldsymbol{x}_t) = - \boldsymbol{D}_t(\boldsymbol{x}_t)\,\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\label{eq:ode-f-grad}\end{equation}  
其中$\boldsymbol{D}_t(\boldsymbol{x}_t)$可以是一个矩阵，也可能是一个标量，视具体考虑的复杂度而定。为什么要考虑这种形式的解？说实话，笔者一开始就是往DDIM格式去凑的，后来就是发现一般化后能跟下面的扩散方程联系起来，所以就直接设为式$\eqref{eq:ode-f-grad}$了。事后来看，如果假设$\boldsymbol{D}_t(\boldsymbol{x}_t)$是非负标量函数，那么将它代入式$\eqref{eq:ode-diff}$后，就会发现其格式跟梯度下降有点相似，即从$\boldsymbol{x}_0$到$\boldsymbol{x}_T$是逐渐寻找低概率区域，反之从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$就是逐渐寻找高概率区域，跟直觉相符，这也算是式$\eqref{eq:ode-f-grad}$的一个启发式引导吧。

将式$\eqref{eq:ode-f-grad}$代入方程$\eqref{eq:ode-f-eq-fp}$后，我们可以得到  
\begin{equation}\frac{\partial}{\partial t}p_t(\boldsymbol{x}_t) = \nabla_{\boldsymbol{x}_t}\cdot\Big(\boldsymbol{D}_t(\boldsymbol{x}_t)\,\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t)\Big)\end{equation}  
这就是偏微分方程中的“[扩散方程](https://en.wikipedia.org/wiki/Diffusion_equation)”。这里我们只考虑一个极简单的情形——$\boldsymbol{D}_t(\boldsymbol{x}_t)$是跟$\boldsymbol{x}_t$无关的标量函数$D_t$，此时扩散方程简化为  
\begin{equation}\frac{\partial}{\partial t}p_t(\boldsymbol{x}_t) = D_t \nabla_{\boldsymbol{x}_t}^2 p_t(\boldsymbol{x}_t)\label{eq:heat}\end{equation}  
这就是“[热传导方程](https://en.wikipedia.org/wiki/Heat_equation)”，是我们接下来要重点求解和分析的对象。

## 求解分布 #

利用傅里叶变换，可以将热传导方程转为常微分方程，继而完成分布$p_t(\boldsymbol{x}_t)$的求解，结果是：  
\begin{equation}\begin{aligned}  
p_t(\boldsymbol{x}_t) =&\, \int \frac{1}{(2\pi\sigma_t^2)^{d/2}}\exp\left(-\frac{\Vert \boldsymbol{x}_t - \boldsymbol{x}_0\Vert^2}{2\sigma_t^2}\right)p_0(\boldsymbol{x}_0) d \boldsymbol{x}_0 \\\  
=&\, \int \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})\, p_0(\boldsymbol{x}_0) d \boldsymbol{x}_0  
\end{aligned}\label{eq:heat-sol}\end{equation}  
其中$\sigma_t^2 = 2\int_0^t D_s ds$，或者$D_t = \dot{\sigma}_t \sigma_t$（其中$\sigma_0=0$）。可以看到，热传导方程的解正好是以$p_0(\boldsymbol{x}_0)$为初始分布的高斯混合模型。

> **过程：** 这里简单介绍一下热传导方程的求解思路。对于不关心求解过程的读者，或者已经熟悉热传导方程的读者，可以跳过这部分内容。
> 
> 用傅里叶变换求热传导方程$\eqref{eq:heat}$其实很简单，对两边的$\boldsymbol{x}_t$变量做傅里叶变换，根据$\nabla_{\boldsymbol{x}_t}\to i\boldsymbol{\omega}$的原则，结果是  
>  \begin{equation}\frac{\partial}{\partial t}\mathcal{F}_t(\boldsymbol{\omega}) = -D_t \boldsymbol{\omega}^2 \mathcal{F}_t(\boldsymbol{\boldsymbol{\omega}})\end{equation}  
>  这只是关于$t$的常微分方程，可以解得  
>  \begin{equation}\mathcal{F}_t(\boldsymbol{\omega}) = \mathcal{F}_0(\boldsymbol{\omega}) \exp\left(-\frac{1}{2}\sigma_t^2 \boldsymbol{\omega}^2\right)\end{equation}  
>  其中$\sigma_t^2 = 2\int_0^t D_s ds$，而$\mathcal{F}_0(\boldsymbol{\omega})$则是$p_0(\boldsymbol{x}_0)$的傅里叶变换。现在对两边做傅里叶逆变换，$\mathcal{F}_t(\boldsymbol{\omega})$自然变回$p_t(\boldsymbol{x}_t)$，$\mathcal{F}_0(\boldsymbol{\omega})$变回$p_0(\boldsymbol{x}_0)$，$\exp\left(-\frac{1}{2}\sigma_t^2 \boldsymbol{\omega}^2\right)$则对应正态分布$\mathcal{N}(\boldsymbol{x}_t; \boldsymbol{0}, \sigma_t^2 \boldsymbol{I})$，最后利用傅里叶变换的卷积性质，就得到解$\eqref{eq:heat-sol}$。

## 完成设计 #

现在我们汇总一下我们的结果：通过求解热传导方程，我们确定了  
\begin{equation}p_t(\boldsymbol{x}_t) = \int \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})\, p_0(\boldsymbol{x}_0) d \boldsymbol{x}_0 \label{eq:heat-sol-2}\end{equation}  
此时对应的微分方程  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt}=-\dot{\sigma}_t \sigma_t \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\end{equation}  
给出了从$p_0(\boldsymbol{x}_0)$到$p_T(\boldsymbol{x}_T)$的一个确定性变换。如果$p_T(\boldsymbol{x}_T)$易于采样，并且$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$已知，那么我们就可以随机采样$\boldsymbol{x}_T\sim p_T(\boldsymbol{x}_T)$，然后逆向求解该微分方程，来生成$\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)$的样本。

第一个问题，什么时候$p_T(\boldsymbol{x}_T)$是易于采样的？根据结果$\eqref{eq:heat-sol-2}$，我们知道  
\begin{equation}\boldsymbol{x}_T\sim p_T(\boldsymbol{x}_T) \quad\Leftrightarrow\quad \boldsymbol{x}_T = \boldsymbol{x}_0 + \sigma_T \boldsymbol{\varepsilon},\,\,\, \boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\,\boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{equation}  
当$\sigma_T$足够大时，$\boldsymbol{x}_0$对$\boldsymbol{x}_T$的影响就很微弱了，此时可以认为  
\begin{equation}\boldsymbol{x}_T\sim p_T(\boldsymbol{x}_T) \quad\Leftrightarrow\quad \boldsymbol{x}_T = \sigma_T \boldsymbol{\varepsilon},\,\,\,\boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{equation}  
这就实现了$p_T(\boldsymbol{x}_T)$易于采样的目的。因此，选择$\sigma_t$的一般要求是：满足$\sigma_0 = 0$和$\sigma_T \gg 1$的光滑单调递增函数。

第二个问题，就是如何计算$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$？这其实跟[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209#%E5%BE%97%E5%88%86%E5%8C%B9%E9%85%8D)中的“得分匹配”一节是一样的，我们用一个神经网络$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$去拟合它，训练目标是  
\begin{equation}\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t \sim \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})p_0(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2 \boldsymbol{I})\right\Vert^2\right]  
\end{equation}  
这叫做“条件得分匹配”，其推导我们在SDE篇已经给出了，这里就不重复了。

## 文章小结 #

在这篇文章中，我们对ODE式扩散模型做了一个“自上而下”的推导：首先从ODE出发，结合雅可比行列式得到了概率变化的一阶近似，然后对比直接泰勒展开的一阶近似，得到了ODE应该要满足的方程，继而转化为扩散方程、热传导方程来求解。相对来说，整个过程比较一步到位，不需要通过SDE、FP方程等结果来做过渡。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9280>_

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

苏剑林. (Sep. 28, 2022). 《生成扩散模型漫谈（十二）：“硬刚”扩散ODE 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9280>

@online{kexuefm-9280,  
title={生成扩散模型漫谈（十二）：“硬刚”扩散ODE},  
author={苏剑林},  
year={2022},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/9280}},  
} 


---

## 公式推导与注释

### 1. ODE与可逆变换的数学基础

#### 1.1 ODE定义的变换

考虑一阶ODE系统：

$$
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{x}_t), \quad t \in [0, T] \tag{1}
$$

**存在唯一性定理**：若$\boldsymbol{f}_t$满足Lipschitz条件：

$$
\|\boldsymbol{f}_t(\boldsymbol{x}) - \boldsymbol{f}_t(\boldsymbol{y})\| \leq L\|\boldsymbol{x} - \boldsymbol{y}\| \tag{2}
$$

则给定初值$\boldsymbol{x}_0$，存在唯一解$\boldsymbol{x}_t$。

**可逆性**：定义映射$\Phi_{0 \to T}: \boldsymbol{x}_0 \mapsto \boldsymbol{x}_T$。其逆映射$\Phi_{T \to 0}$由反向ODE定义：

$$
\frac{d\boldsymbol{x}_t}{dt} = -\boldsymbol{f}_t(\boldsymbol{x}_t) \tag{3}
$$

#### 1.2 离散化与概率流

**Euler离散化**：

$$
\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t \tag{4}
$$

这定义了一系列确定性变换$\boldsymbol{T}_{\Delta t}: \boldsymbol{x}_t \mapsto \boldsymbol{x}_{t+\Delta t}$。

### 2. 变量变换公式与雅可比行列式

#### 2.1 概率密度的变换

**定理2.1**（变量变换公式）：设$\boldsymbol{y} = \boldsymbol{g}(\boldsymbol{x})$是可逆变换，则概率密度满足：

$$
p_Y(\boldsymbol{y}) = p_X(\boldsymbol{g}^{-1}(\boldsymbol{y})) \left|\det \frac{\partial \boldsymbol{g}^{-1}}{\partial \boldsymbol{y}}\right| = p_X(\boldsymbol{x}) \left|\det \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}}\right|^{-1} \tag{5}
$$

**推导**：从测度守恒$p_X(\boldsymbol{x})d\boldsymbol{x} = p_Y(\boldsymbol{y})d\boldsymbol{y}$出发：

$$
p_X(\boldsymbol{x})d\boldsymbol{x} = p_Y(\boldsymbol{g}(\boldsymbol{x})) \left|\det \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{x}}\right| d\boldsymbol{x} \tag{6}
$$

#### 2.2 ODE诱导的雅可比

**计算**：对(4)求导，记$\mathbf{J}_t = \frac{\partial \boldsymbol{f}_t}{\partial \boldsymbol{x}_t}$：

$$
\frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t} = \boldsymbol{I} + \mathbf{J}_t \Delta t \tag{7}
$$

**行列式近似**（利用$\det(\boldsymbol{I} + \epsilon \mathbf{A}) \approx 1 + \epsilon \text{Tr}(\mathbf{A})$）：

$$
\det\left(\frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t}\right) = \det(\boldsymbol{I} + \mathbf{J}_t \Delta t) \approx 1 + \text{Tr}(\mathbf{J}_t) \Delta t = 1 + (\nabla \cdot \boldsymbol{f}_t) \Delta t \tag{8}
$$

其中$\nabla \cdot \boldsymbol{f}_t = \sum_{i=1}^d \frac{\partial f_{t,i}}{\partial x_{t,i}}$是散度。

**进一步近似**：

$$
1 + (\nabla \cdot \boldsymbol{f}_t) \Delta t \approx \exp\left((\nabla \cdot \boldsymbol{f}_t) \Delta t\right) \tag{9}
$$

（利用$e^x \approx 1 + x$）

#### 2.3 对数密度的变化

**结合(6)和(9)**：

$$
\log p_t(\boldsymbol{x}_t) = \log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - \log \left|\det \frac{\partial \boldsymbol{x}_{t+\Delta t}}{\partial \boldsymbol{x}_t}\right| \tag{10}
$$

$$
\log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - \log p_t(\boldsymbol{x}_t) \approx -(\nabla_{\boldsymbol{x}_t} \cdot \boldsymbol{f}_t) \Delta t \tag{11}
$$

### 3. 从泰勒展开到Fokker-Planck方程

#### 3.1 联合泰勒展开

假设$p_t(\boldsymbol{x}_t)$光滑，对$\log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})$在$(\boldsymbol{x}_t, t)$处泰勒展开：

$$
\begin{aligned}
\log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) &\approx \log p_t(\boldsymbol{x}_t) + \frac{\partial \log p_t}{\partial t} \Delta t \\
&\quad + \langle \nabla_{\boldsymbol{x}} \log p_t, \boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t \rangle + O(\Delta t^2)
\end{aligned} \tag{12}
$$

代入$\boldsymbol{x}_{t+\Delta t} - \boldsymbol{x}_t = \boldsymbol{f}_t \Delta t$：

$$
\log p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t}) - \log p_t(\boldsymbol{x}_t) \approx \frac{\partial \log p_t}{\partial t} \Delta t + \langle \nabla_{\boldsymbol{x}} \log p_t, \boldsymbol{f}_t \rangle \Delta t \tag{13}
$$

#### 3.2 匹配两种推导

**对比(11)和(13)**：

$$
-(\nabla \cdot \boldsymbol{f}_t) \Delta t = \left[\frac{\partial \log p_t}{\partial t} + \langle \nabla_{\boldsymbol{x}} \log p_t, \boldsymbol{f}_t \rangle\right] \Delta t \tag{14}
$$

消去$\Delta t$：

$$
\frac{\partial \log p_t}{\partial t} + \boldsymbol{f}_t \cdot \nabla_{\boldsymbol{x}} \log p_t + \nabla_{\boldsymbol{x}} \cdot \boldsymbol{f}_t = 0 \tag{15}
$$

**整理为连续性方程**：

$$
\frac{\partial p_t}{\partial t} + \nabla_{\boldsymbol{x}} \cdot (p_t \boldsymbol{f}_t) = 0 \tag{16}
$$

这正是**Fokker-Planck方程**（扩散项$g_t = 0$的情形）。

### 4. 梯度流形式与热传导方程

#### 4.1 Score函数参数化

**假设**：$\boldsymbol{f}_t$取梯度流形式：

$$
\boldsymbol{f}_t(\boldsymbol{x}_t) = -D_t \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) \tag{17}
$$

其中$D_t > 0$是扩散系数（标量）。

**动机**：
- 几何上，这是$-\log p_t$的梯度下降方向
- 从高概率区域向低概率区域流动（前向）
- 反向ODE则从低概率向高概率流动（生成）

#### 4.2 推导热传导方程

将(17)代入(16)：

$$
\frac{\partial p_t}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot \left(-D_t p_t \nabla_{\boldsymbol{x}} \log p_t\right) = \nabla_{\boldsymbol{x}} \cdot \left(D_t \nabla_{\boldsymbol{x}} p_t\right) \tag{18}
$$

假设$D_t$与$\boldsymbol{x}_t$无关：

$$
\frac{\partial p_t}{\partial t} = D_t \nabla_{\boldsymbol{x}}^2 p_t \tag{19}
$$

这是经典的**热传导方程**（或扩散方程）。

### 5. 热传导方程的求解

#### 5.1 傅里叶变换方法

**定义傅里叶变换**：

$$
\mathcal{F}_t(\boldsymbol{\omega}) = \int e^{-i\boldsymbol{\omega} \cdot \boldsymbol{x}_t} p_t(\boldsymbol{x}_t) d\boldsymbol{x}_t \tag{20}
$$

**变换性质**：$\nabla_{\boldsymbol{x}}^2 \leftrightarrow -\|\boldsymbol{\omega}\|^2$

**变换后的ODE**：对(19)两边做傅里叶变换：

$$
\frac{\partial \mathcal{F}_t}{\partial t} = -D_t \|\boldsymbol{\omega}\|^2 \mathcal{F}_t \tag{21}
$$

**求解**：这是关于$t$的常微分方程，通解为：

$$
\mathcal{F}_t(\boldsymbol{\omega}) = \mathcal{F}_0(\boldsymbol{\omega}) \exp\left(-\|\boldsymbol{\omega}\|^2 \int_0^t D_s ds\right) \tag{22}
$$

定义：

$$
\sigma_t^2 := 2\int_0^t D_s ds \quad \Leftrightarrow \quad D_t = \frac{1}{2}\frac{d\sigma_t^2}{dt} = \dot{\sigma}_t \sigma_t \tag{23}
$$

（假设$\sigma_0 = 0$）

则：

$$
\mathcal{F}_t(\boldsymbol{\omega}) = \mathcal{F}_0(\boldsymbol{\omega}) \exp\left(-\frac{\sigma_t^2}{2}\|\boldsymbol{\omega}\|^2\right) \tag{24}
$$

#### 5.2 逆变换与卷积

**关键观察**：$\exp(-\frac{\sigma_t^2}{2}\|\boldsymbol{\omega}\|^2)$是$\mathcal{N}(\mathbf{0}, \sigma_t^2\boldsymbol{I})$的傅里叶变换。

**卷积定理**：频域乘积对应空域卷积：

$$
p_t(\boldsymbol{x}_t) = \int \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2\boldsymbol{I}) p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 \tag{25}
$$

**等价表述**（随机变量形式）：

$$
\boldsymbol{x}_t = \boldsymbol{x}_0 + \sigma_t \boldsymbol{\varepsilon}, \quad \boldsymbol{x}_0 \sim p_0, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}) \tag{26}
$$

**验证**：这正是高斯扩散过程的形式！

### 6. 完整的生成模型

#### 6.1 前向过程与终态分布

**前向ODE**（数据→噪声）：

$$
\frac{d\boldsymbol{x}_t}{dt} = -D_t \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t), \quad t: 0 \to T \tag{27}
$$

**终态分析**：当$\sigma_T \to \infty$时，由(26)：

$$
p_T(\boldsymbol{x}_T) = \int \mathcal{N}(\boldsymbol{x}_T; \boldsymbol{x}_0, \sigma_T^2\boldsymbol{I}) p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 \approx \mathcal{N}(\mathbf{0}, \sigma_T^2\boldsymbol{I}) \tag{28}
$$

（当$\sigma_T^2 \gg \text{Var}[p_0]$时成立）

#### 6.2 反向过程与生成

**反向ODE**（噪声→数据）：

$$
\frac{d\boldsymbol{x}_t}{dt} = D_t \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t), \quad t: T \to 0 \tag{29}
$$

**采样算法**：
1. 采样$\boldsymbol{x}_T \sim \mathcal{N}(\mathbf{0}, \sigma_T^2\boldsymbol{I})$
2. 数值求解(29)，得到$\boldsymbol{x}_0$

**核心问题**：需要知道score function $\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$。

### 7. 得分匹配与模型训练

#### 7.1 条件得分匹配

**目标**：训练神经网络$\boldsymbol{s}_\theta(\boldsymbol{x}_t, t)$逼近$\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t)$。

**条件得分的精确形式**：由(25)的条件分布：

$$
p(\boldsymbol{x}_0|\boldsymbol{x}_t) = \frac{\mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2\boldsymbol{I}) p_0(\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_t)} \tag{30}
$$

计算score：

$$
\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) = \nabla_{\boldsymbol{x}_t} \log \int \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2\boldsymbol{I}) p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 \tag{31}
$$

**条件期望表示**：

$$
\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) = \mathbb{E}_{\boldsymbol{x}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\nabla_{\boldsymbol{x}_t} \log \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2\boldsymbol{I})\right] \tag{32}
$$

$$
= \mathbb{E}_{\boldsymbol{x}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[-\frac{\boldsymbol{x}_t - \boldsymbol{x}_0}{\sigma_t^2}\right] = -\frac{1}{\sigma_t^2}\mathbb{E}[\boldsymbol{x}_t - \boldsymbol{x}_0 | \boldsymbol{x}_t] \tag{33}
$$

#### 7.2 训练目标

**定理7.1**（去噪得分匹配）：最小化以下目标等价于得分匹配：

$$
\mathcal{L} = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{\varepsilon}}\left[\left\|\boldsymbol{s}_\theta(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2\boldsymbol{I})\right\|^2\right] \tag{34}
$$

其中$\boldsymbol{x}_t = \boldsymbol{x}_0 + \sigma_t \boldsymbol{\varepsilon}$，$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$。

**简化形式**（噪声预测）：

$$
\nabla_{\boldsymbol{x}_t} \log \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}_0, \sigma_t^2\boldsymbol{I}) = -\frac{\boldsymbol{x}_t - \boldsymbol{x}_0}{\sigma_t^2} = -\frac{\boldsymbol{\varepsilon}}{\sigma_t} \tag{35}
$$

因此等价于训练噪声预测网络$\boldsymbol{\epsilon}_\theta$：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \boldsymbol{x}_0, \boldsymbol{\varepsilon}}\left[\|\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_0 + \sigma_t\boldsymbol{\varepsilon}, t) - \boldsymbol{\varepsilon}\|^2\right] \tag{36}
$$

### 8. 总结与洞察

#### 8.1 核心公式链

$$
\text{ODE} \xrightarrow{\text{雅可比}} \text{连续性方程} \xrightarrow{\text{梯度流}} \text{热传导方程} \xrightarrow{\text{傅里叶}} \text{高斯卷积}
$$

#### 8.2 与DDIM的联系

**DDIM采样公式**回顾：

$$
\boldsymbol{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\frac{\boldsymbol{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{预测的}\boldsymbol{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta \tag{37}
$$

**本文方法**：通过ODE直接推导，得到：

$$
d\boldsymbol{x}_t = \dot{\sigma}_t \sigma_t \nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t) dt \approx \frac{\dot{\sigma}_t}{\sigma_t}(\boldsymbol{x}_t - \boldsymbol{x}_0) dt \tag{38}
$$

（利用$\nabla \log p_t \approx -\frac{\boldsymbol{x}_t - \mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t]}{\sigma_t^2}$）

**统一性**：选择$\sigma_t = \sqrt{1-\bar{\alpha}_t}$时，两者一致！

#### 8.3 优势

1. **直接性**：无需通过SDE→ODE的转换
2. **清晰性**：热传导方程有明确的物理解释
3. **灵活性**：可选择不同的$\sigma_t(t)$调度

---

**最终总结**："硬刚"ODE提供了扩散模型的第一性原理推导，从连续微分方程出发，通过雅可比行列式和热传导方程，直接得到高斯扩散过程，绕过了SDE的随机性，强调了确定性ODE的本质。

