---
title: 测试函数法推导连续性方程和Fokker-Planck方程
slug: 测试函数法推导连续性方程和fokker-planck方程
date: 2023-02-11
tags: 详细推导, 概率, 微分方程, 随机, 扩散, 生成模型
status: pending
---
# 测试函数法推导连续性方程和Fokker-Planck方程

**原文链接**: [https://spaces.ac.cn/archives/9461](https://spaces.ac.cn/archives/9461)

**发布日期**: 

---

在文章[《生成扩散模型漫谈（六）：一般框架之ODE篇》](/archives/9228)中，我们推导了SDE的Fokker-Planck方程；而在[《生成扩散模型漫谈（十二）：“硬刚”扩散ODE》](/archives/9280)中，我们单独推导了ODE的连续性方程。它们都是描述随机变量沿着SDE/ODE演化的分布变化方程，连续性方程是Fokker-Planck方程的特例。在推导Fokker-Planck方程时，我们将泰勒展开硬套到了狄拉克函数上，虽然结果是对的，但未免有点不伦不类；在推导连续性方程时，我们结合了雅可比行列式和泰勒展开，方法本身比较常规，但没法用来推广到Fokker-Planck方程。

这篇文章我们介绍“测试函数法”，它是推导连续性方程和Fokker-Planck方程的标准方法之一，其分析过程比较正规，并且适用场景也比较广。

## 分部积分 #

正式推导之前，这里先介绍后面推导会用到的关键结果——分部积分法的高维推广。

一般教程对分部积分的介绍仅限于一维情形，即  
\begin{equation}\int_a^b uv'dx = uv|_a^b - \int_a^b vu'dx\end{equation}  
这里$u,v$是$x$的函数，$'$表示求函数关于$x$的导数。下面我们需要它的一个高维版本，为此，我们先回顾一维分部积分的推导过程，它依赖于求导的乘法法则：  
\begin{equation}(uv)' = uv' + vu'\end{equation}  
然后两边对$x$积分并移项，就得到分部积分公式。对于高维情形，我们考虑类似的公式：  
\begin{equation}\nabla\cdot(u\boldsymbol{v}) = \boldsymbol{v}\cdot\nabla u + u\nabla \cdot\boldsymbol{v}\end{equation}  
其中$u$是$\boldsymbol{x}$的标量函数，$\boldsymbol{v}$是$\boldsymbol{x}$的向量函数（维度跟$\boldsymbol{v}$一致），$\nabla$表示求函数关于$\boldsymbol{x}$的梯度。现在我们对两端在区域$\Omega$积分：  
\begin{equation}\int_{\Omega}\nabla\cdot(u\boldsymbol{v})d\boldsymbol{x} = \int_{\Omega}\boldsymbol{v}\cdot\nabla u d\boldsymbol{x} + \int_{\Omega}u\nabla \cdot\boldsymbol{v} d\boldsymbol{x}\end{equation}  
根据[高斯散度定理](https://en.wikipedia.org/wiki/Divergence_theorem)，左侧等于$\int_{\partial\Omega}u\boldsymbol{v}\cdot\hat{\boldsymbol{n}}dS$，$\partial\Omega$是$\Omega$的边界，$\hat{\boldsymbol{n}}$是边界的外向单位法向量，$dS$是面积微元。所以，移项后有  
\begin{equation}\int_{\Omega}\boldsymbol{v}\cdot\nabla u d\boldsymbol{x} = \int_{\partial\Omega}u\boldsymbol{v}\cdot\hat{\boldsymbol{n}}dS - \int_{\Omega}u\nabla \cdot\boldsymbol{v} d\boldsymbol{x}\label{eq:int-by-parts}\end{equation}  
这就是我们要推导的高维空间分部积分公式。特别地，对于概率密度函数$p$，那么由于非负性和积分为1的限制，无穷远处必然有$p\to 0$和$\nabla p\to \boldsymbol{0}$，所以如果$\Omega$选为全空间（没有特别注明积分区域的，默认为全空间），那么分别将$u=p$和$\boldsymbol{v}=\nabla p$代入上式，得到  
\begin{align}\int\boldsymbol{v}\cdot\nabla p d\boldsymbol{x} =&\, - \int p\nabla \cdot\boldsymbol{v} d\boldsymbol{x}\label{eq:int-by-parts-p} \\\  
\int u\nabla \cdot\nabla p d\boldsymbol{x} = &\,-\int\nabla p\cdot\nabla u d\boldsymbol{x}\label{eq:int-by-parts-gp}\end{align}  
如果要进一步严格化上述结论，可以假设$p$具有紧的支撑集。不过这纯粹是数学上的严格化，事实上对于一般理解来说，直接默认在无穷远处成立$p\to 0$和$\nabla p\to \boldsymbol{0}$就够了。

## ODE演化 #

测试函数法的原理，是如果对于任意函数$\phi(\boldsymbol{x})$，都成立  
\begin{equation}\int f(\boldsymbol{x})\phi(\boldsymbol{x})d\boldsymbol{x} = \int g(\boldsymbol{x})\phi(\boldsymbol{x})d\boldsymbol{x}\end{equation}  
那么就成立$f(\boldsymbol{x})=g(\boldsymbol{x})$，其中$\phi(\boldsymbol{x})$就叫做测试函数。更严谨的定义需要声明$\phi(\boldsymbol{x})$的选取空间，以及等号的具体含义（如严格相等/几乎处处相等/依概率相等之类），这里我们就不引入这些细节了。

对于ODE  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt}=\boldsymbol{f}_t(\boldsymbol{x}_t)\label{eq:ode}\end{equation}  
我们将它离散化为  
\begin{equation}\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\label{eq:ode-diff}\end{equation}  
那么就有  
\begin{equation}\phi(\boldsymbol{x}_{t+\Delta t}) = \phi(\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t)\approx \phi(\boldsymbol{x}_t) + \Delta t\,\,\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)\end{equation}  
两边求期望，得到：  
\begin{equation}\int p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})\phi(\boldsymbol{x}_{t+\Delta t}) d\boldsymbol{x}_{t+\Delta t}\approx \int p_t(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \Delta t\int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\end{equation}  
由于积分的结果不依赖于被积自变量的记号，所以左边将$\boldsymbol{x}_{t+\Delta t}$换成$\boldsymbol{x}_t$也是等价的：  
\begin{equation}\int p_{t+\Delta t}(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t\approx \int p_t(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \Delta t\int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\label{eq:change-var}\end{equation}  
将右边第一项移到左边，然后取$\Delta t\to 0$的极限，得到：  
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\label{eq:dt-0}\end{equation}  
右端利用分部积分公式$\eqref{eq:int-by-parts-p}$得到  
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = -\int \Big[\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)\Big]\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\end{equation}  
根据测试函数法的相等原理，就有  
\begin{equation}\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} = -\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)\end{equation}  
这称为“[连续性方程](https://en.wikipedia.org/wiki/Continuity_equation)”。

## SDE演化 #

对于SDE  
\begin{equation}d\boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) dt + g_t d\boldsymbol{w}\label{eq:sde}\end{equation}  
我们离散化为  
\begin{equation}\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:sde-diff}\end{equation}  
那么  
\begin{equation}\begin{aligned}  
\phi(\boldsymbol{x}_{t+\Delta t}) =&\, \phi(\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}) \\\  
\approx&\, \phi(\boldsymbol{x}_t) + \left(\boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}\right)\cdot \nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) + \frac{1}{2} \left(g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}\cdot \nabla_{\boldsymbol{x}_t}\right)^2\phi(\boldsymbol{x}_t)  
\end{aligned}\end{equation}  
两边求期望，注意右边要同时对$\boldsymbol{x}_t$和$\boldsymbol{\varepsilon}$求期望，其中$\boldsymbol{\varepsilon}$的期望可以事先求出，结果是  
\begin{equation}\phi(\boldsymbol{x}_t) + \Delta t\,\,\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) + \frac{1}{2} \Delta t\,g_t^2\nabla_{\boldsymbol{x}_t}\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)  
\end{equation}  
于是  
\begin{equation}\begin{aligned}  
&\,\int p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})\phi(\boldsymbol{x}_{t+\Delta t}) d\boldsymbol{x}_{t+\Delta t}\\\  
\approx&\, \int p_t(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \Delta t\int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \int\frac{1}{2} \Delta t\,g_t^2 p_t(\boldsymbol{x}_t)\nabla_{\boldsymbol{x}_t}\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t  
\end{aligned}\end{equation}  
跟式$\eqref{eq:change-var}$、式$\eqref{eq:dt-0}$类似，取$\Delta\to 0$的极限，得到  
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \int\frac{1}{2} \,g_t^2 p_t(\boldsymbol{x}_t)\nabla_{\boldsymbol{x}_t}\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t\end{equation}  
对右边第一项应用式$\eqref{eq:int-by-parts-p}$、对右边第二项先应用式$\eqref{eq:int-by-parts-gp}$再应用式$\eqref{eq:int-by-parts-p}$，得到  
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int \left[-\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)+\frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\right]\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\end{equation}  
根据测试函数法的相等原理得  
\begin{equation}\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} = -\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)+\frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\end{equation}  
这就是“[Fokker-Planck方程](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation)”。

## 文章小结 #

本文介绍了用于推导某些概率方程的测试函数法，主要内容包括分部积分法的高维推广，以及ODE的连续性方程和SDE的Fokker-Planck方程的推导过程。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9461>_

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

苏剑林. (Feb. 11, 2023). 《测试函数法推导连续性方程和Fokker-Planck方程 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9461>

@online{kexuefm-9461,  
title={测试函数法推导连续性方程和Fokker-Planck方程},  
author={苏剑林},  
year={2023},  
month={Feb},  
url={\url{https://spaces.ac.cn/archives/9461}},  
} 


---

## 公式推导与注释

### 1. 测试函数的定义与性质

**1.1 测试函数空间**

测试函数（test function）是数学分析和偏微分方程理论中的核心概念，用于定义函数的弱形式（weak form）。测试函数$\phi(\boldsymbol{x})$通常需要满足以下条件：

1. **光滑性**：$\phi \in C^\infty(\mathbb{R}^n)$，即$\phi$是无穷次可微的
2. **紧支撑**：存在有界区域$K \subset \mathbb{R}^n$，使得当$\boldsymbol{x} \notin K$时，$\phi(\boldsymbol{x}) = 0$
3. **衰减性**：对于非紧支撑的情况，要求$\phi$及其所有导数在无穷远处快速衰减

满足这些条件的函数空间记为$\mathcal{D}(\mathbb{R}^n)$或$C_0^\infty(\mathbb{R}^n)$，称为测试函数空间。

**1.2 测试函数的基本性质**

测试函数有以下重要性质：

**性质1（线性性）**：如果$\phi_1, \phi_2 \in \mathcal{D}$，$\alpha, \beta \in \mathbb{R}$，则$\alpha\phi_1 + \beta\phi_2 \in \mathcal{D}$。

**性质2（导数封闭性）**：如果$\phi \in \mathcal{D}$，则对任意多重指标$\alpha = (\alpha_1, \ldots, \alpha_n)$，有
$$\frac{\partial^{|\alpha|} \phi}{\partial x_1^{\alpha_1} \cdots \partial x_n^{\alpha_n}} \in \mathcal{D}$$

**性质3（乘法封闭性）**：如果$\phi \in \mathcal{D}$，$g \in C^\infty$，则$g\phi \in \mathcal{D}$（当$g$增长不太快时）。

**性质4（积分有界性）**：由于紧支撑性，测试函数的任意次积分都是良定义的：
$$\int_{\mathbb{R}^n} \left|\frac{\partial^{|\alpha|} \phi}{\partial x_1^{\alpha_1} \cdots \partial x_n^{\alpha_n}}\right| d\boldsymbol{x} < \infty$$

**1.3 分布理论中的测试函数**

在分布理论（distribution theory）中，测试函数用于定义广义函数。一个分布$T$是测试函数空间$\mathcal{D}$上的连续线性泛函：
$$T: \mathcal{D}(\mathbb{R}^n) \to \mathbb{R}$$
满足线性性：
$$T(\alpha\phi_1 + \beta\phi_2) = \alpha T(\phi_1) + \beta T(\phi_2)$$

例如，狄拉克$\delta$函数可以通过其作用在测试函数上的结果来定义：
$$\langle \delta_{\boldsymbol{x}_0}, \phi \rangle = \phi(\boldsymbol{x}_0)$$

对于概率密度函数$p(\boldsymbol{x})$，它定义了一个分布：
$$\langle p, \phi \rangle = \int_{\mathbb{R}^n} p(\boldsymbol{x})\phi(\boldsymbol{x}) d\boldsymbol{x}$$

### 2. 弱形式与强形式的关系

**2.1 强形式与弱形式的定义**

考虑偏微分方程（PDE）：
$$\mathcal{L}[u] = f$$
其中$\mathcal{L}$是微分算子，$u$是未知函数，$f$是已知函数。

**强形式（Strong form）**：函数$u$在经典意义下处处满足方程$\mathcal{L}[u] = f$。这要求$u$具有足够的光滑性，使得所有导数都存在。

**弱形式（Weak form）**：对任意测试函数$\phi \in \mathcal{D}$，成立
$$\langle \mathcal{L}[u], \phi \rangle = \langle f, \phi \rangle$$
即
$$\int_{\mathbb{R}^n} \mathcal{L}[u](\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x} = \int_{\mathbb{R}^n} f(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x}$$

**2.2 从强形式到弱形式**

如果$u$是强解（即在经典意义下满足方程），则对任意测试函数$\phi$，有：
$$\int_{\mathbb{R}^n} \mathcal{L}[u](\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x} = \int_{\mathbb{R}^n} f(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x}$$
因此强解必然是弱解。

**2.3 从弱形式到强形式**

反过来，如果$u$是充分光滑的弱解，并且对**任意**测试函数$\phi$都成立弱形式，则由分布理论的基本引理（fundamental lemma of the calculus of variations）可得：
$$\mathcal{L}[u](\boldsymbol{x}) = f(\boldsymbol{x}), \quad \text{几乎处处成立}$$

**基本引理**：如果$g \in L^1_{loc}(\mathbb{R}^n)$满足
$$\int_{\mathbb{R}^n} g(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x} = 0, \quad \forall \phi \in \mathcal{D}(\mathbb{R}^n)$$
则$g(\boldsymbol{x}) = 0$几乎处处成立。

**证明思路**：假设存在测度不为零的集合$E$使得$g(\boldsymbol{x}) > 0$在$E$上成立。由于$g$的局部可积性和测试函数的任意性，可以构造一个支撑在$E$上的非负测试函数$\phi$，使得
$$\int_{\mathbb{R}^n} g(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x} > 0$$
这与假设矛盾，因此$g = 0$几乎处处成立。

**2.4 弱形式的优势**

使用弱形式有以下优势：

1. **放松光滑性要求**：弱解不需要所有导数都存在，只需积分意义下的"弱导数"存在
2. **包含间断解**：可以处理激波等间断解
3. **自然包含边界条件**：通过分部积分，边界条件自然嵌入到弱形式中
4. **数值方法的基础**：有限元方法等数值方法基于弱形式

### 3. 高维分部积分的详细推导

**3.1 梯度与散度的基本性质**

对于标量函数$u: \mathbb{R}^n \to \mathbb{R}$，梯度定义为：
$$\nabla u = \left(\frac{\partial u}{\partial x_1}, \frac{\partial u}{\partial x_2}, \ldots, \frac{\partial u}{\partial x_n}\right)$$

对于向量函数$\boldsymbol{v} = (v_1, v_2, \ldots, v_n): \mathbb{R}^n \to \mathbb{R}^n$，散度定义为：
$$\nabla \cdot \boldsymbol{v} = \sum_{i=1}^n \frac{\partial v_i}{\partial x_i}$$

**3.2 乘积的散度公式**

考虑标量函数$u$和向量函数$\boldsymbol{v}$的乘积$u\boldsymbol{v}$的散度：
$$\nabla \cdot (u\boldsymbol{v}) = \nabla \cdot (uv_1, uv_2, \ldots, uv_n)$$

根据散度的定义：
$$\nabla \cdot (u\boldsymbol{v}) = \sum_{i=1}^n \frac{\partial (uv_i)}{\partial x_i}$$

应用乘法法则$\frac{\partial (uv_i)}{\partial x_i} = v_i\frac{\partial u}{\partial x_i} + u\frac{\partial v_i}{\partial x_i}$：
$$\nabla \cdot (u\boldsymbol{v}) = \sum_{i=1}^n \left(v_i\frac{\partial u}{\partial x_i} + u\frac{\partial v_i}{\partial x_i}\right)$$

重新整理：
$$\nabla \cdot (u\boldsymbol{v}) = \sum_{i=1}^n v_i\frac{\partial u}{\partial x_i} + u\sum_{i=1}^n \frac{\partial v_i}{\partial x_i}$$

即：
$$\nabla \cdot (u\boldsymbol{v}) = \boldsymbol{v} \cdot \nabla u + u\nabla \cdot \boldsymbol{v}$$

这是高维空间中的乘积法则。

**3.3 高斯散度定理**

高斯散度定理（也称为散度定理或Gauss定理）陈述：对于向量场$\boldsymbol{F}: \Omega \to \mathbb{R}^n$，其中$\Omega \subset \mathbb{R}^n$是具有光滑边界$\partial\Omega$的有界区域，成立：
$$\int_\Omega \nabla \cdot \boldsymbol{F} \, d\boldsymbol{x} = \int_{\partial\Omega} \boldsymbol{F} \cdot \hat{\boldsymbol{n}} \, dS$$

其中$\hat{\boldsymbol{n}}$是边界$\partial\Omega$的外向单位法向量，$dS$是边界上的面积元。

**物理直观**：散度定理表明，向量场在区域内的散度的总和，等于向量场穿过区域边界的总通量。

**3.4 高维分部积分公式的推导**

现在我们推导高维分部积分公式。对$\nabla \cdot (u\boldsymbol{v}) = \boldsymbol{v} \cdot \nabla u + u\nabla \cdot \boldsymbol{v}$两边在区域$\Omega$上积分：
$$\int_\Omega \nabla \cdot (u\boldsymbol{v}) \, d\boldsymbol{x} = \int_\Omega \boldsymbol{v} \cdot \nabla u \, d\boldsymbol{x} + \int_\Omega u\nabla \cdot \boldsymbol{v} \, d\boldsymbol{x}$$

对左边应用高斯散度定理：
$$\int_{\partial\Omega} u\boldsymbol{v} \cdot \hat{\boldsymbol{n}} \, dS = \int_\Omega \boldsymbol{v} \cdot \nabla u \, d\boldsymbol{x} + \int_\Omega u\nabla \cdot \boldsymbol{v} \, d\boldsymbol{x}$$

移项得到高维分部积分公式：
$$\int_\Omega \boldsymbol{v} \cdot \nabla u \, d\boldsymbol{x} = \int_{\partial\Omega} u\boldsymbol{v} \cdot \hat{\boldsymbol{n}} \, dS - \int_\Omega u\nabla \cdot \boldsymbol{v} \, d\boldsymbol{x}$$

**3.5 概率密度函数的边界条件**

对于概率密度函数$p(\boldsymbol{x})$，由于其非负性和归一化条件：
$$p(\boldsymbol{x}) \geq 0, \quad \int_{\mathbb{R}^n} p(\boldsymbol{x}) d\boldsymbol{x} = 1$$

在无穷远处必然有：
$$\lim_{|\boldsymbol{x}| \to \infty} p(\boldsymbol{x}) = 0$$

更强地，由于概率质量必须是有限的，我们有：
$$\lim_{|\boldsymbol{x}| \to \infty} |\boldsymbol{x}|^n p(\boldsymbol{x}) = 0$$

这意味着$p$的衰减速度快于$|\boldsymbol{x}|^{-n}$。进一步，对于光滑的概率密度，其梯度也必须快速衰减：
$$\lim_{|\boldsymbol{x}| \to \infty} \nabla p(\boldsymbol{x}) = \boldsymbol{0}$$

因此，当$\Omega = \mathbb{R}^n$（全空间）时，边界项消失：
$$\int_{\partial\mathbb{R}^n} u\boldsymbol{v} \cdot \hat{\boldsymbol{n}} \, dS = 0$$

这给出简化的分部积分公式：
$$\int_{\mathbb{R}^n} \boldsymbol{v} \cdot \nabla p \, d\boldsymbol{x} = -\int_{\mathbb{R}^n} p\nabla \cdot \boldsymbol{v} \, d\boldsymbol{x}$$

**3.6 二阶导数的分部积分**

对于拉普拉斯算子$\Delta u = \nabla \cdot \nabla u$，我们可以应用两次分部积分。设$\boldsymbol{F} = \nabla p$，则：
$$\int_{\mathbb{R}^n} u \nabla \cdot \nabla p \, d\boldsymbol{x} = \int_{\mathbb{R}^n} u \nabla \cdot (\nabla p) \, d\boldsymbol{x}$$

第一次分部积分（取$\boldsymbol{v} = \nabla p$）：
$$\int_{\mathbb{R}^n} u \nabla \cdot (\nabla p) \, d\boldsymbol{x} = -\int_{\mathbb{R}^n} \nabla u \cdot \nabla p \, d\boldsymbol{x}$$

（边界项在无穷远处为零）

第二次分部积分（现在积分中$\nabla p$的角色变为向量场）：
$$-\int_{\mathbb{R}^n} \nabla u \cdot \nabla p \, d\boldsymbol{x} = -\int_{\mathbb{R}^n} \nabla p \cdot \nabla u \, d\boldsymbol{x}$$

这可以看作是以$p$为"权重"，$\nabla u$为"向量场"的分部积分，但更常见的用法是保持这个形式：
$$\int_{\mathbb{R}^n} u \Delta p \, d\boldsymbol{x} = -\int_{\mathbb{R}^n} \nabla u \cdot \nabla p \, d\boldsymbol{x}$$

对于对称的情况：
$$\int_{\mathbb{R}^n} u \Delta p \, d\boldsymbol{x} = \int_{\mathbb{R}^n} p \Delta u \, d\boldsymbol{x}$$

（当边界项消失时）

### 4. 连续性方程的完整推导

**4.1 ODE系统与流的概念**

考虑常微分方程（ODE）系统：
$$\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{x}_t)$$

其中$\boldsymbol{x}_t \in \mathbb{R}^n$是状态变量，$\boldsymbol{f}_t: \mathbb{R}^n \to \mathbb{R}^n$是速度场。

这个方程定义了一个**流**（flow）$\Phi_t: \mathbb{R}^n \to \mathbb{R}^n$，使得：
$$\Phi_t(\boldsymbol{x}_0) = \boldsymbol{x}_t$$
是初值为$\boldsymbol{x}_0$的解。

**4.2 概率密度的演化**

假设初始时刻$t=0$，随机变量$\boldsymbol{x}_0$的概率密度为$p_0(\boldsymbol{x})$。在时刻$t$，通过流映射$\Phi_t$，得到$\boldsymbol{x}_t = \Phi_t(\boldsymbol{x}_0)$。

概率守恒要求：
$$P(\boldsymbol{x}_t \in A) = P(\boldsymbol{x}_0 \in \Phi_t^{-1}(A))$$

对于任意可测集合$A$，这可以写成：
$$\int_A p_t(\boldsymbol{x}) d\boldsymbol{x} = \int_{\Phi_t^{-1}(A)} p_0(\boldsymbol{x}) d\boldsymbol{x}$$

**4.3 欧拉描述与拉格朗日描述**

在流体力学中，有两种描述方式：

- **拉格朗日描述**：跟踪每个粒子的轨迹$\boldsymbol{x}_t = \Phi_t(\boldsymbol{x}_0)$
- **欧拉描述**：在固定空间点观察流体性质的变化

连续性方程采用欧拉描述，描述固定空间点处密度$p_t(\boldsymbol{x})$如何随时间变化。

**4.4 离散化与泰勒展开**

将ODE离散化：
$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t + O(\Delta t^2)$$

对于任意测试函数$\phi(\boldsymbol{x})$，应用泰勒展开：
$$\phi(\boldsymbol{x}_{t+\Delta t}) = \phi(\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t)$$

一阶泰勒展开：
$$\phi(\boldsymbol{x}_t + \boldsymbol{\delta}) \approx \phi(\boldsymbol{x}_t) + \boldsymbol{\delta} \cdot \nabla_{\boldsymbol{x}_t} \phi(\boldsymbol{x}_t) + O(|\boldsymbol{\delta}|^2)$$

取$\boldsymbol{\delta} = \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t$：
$$\phi(\boldsymbol{x}_{t+\Delta t}) = \phi(\boldsymbol{x}_t) + \Delta t \, \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla_{\boldsymbol{x}_t} \phi(\boldsymbol{x}_t) + O(\Delta t^2)$$

**4.5 期望值的演化**

对$\phi(\boldsymbol{x}_{t+\Delta t})$在$\boldsymbol{x}_t$分布下取期望：
$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\Delta t})] = \int p_t(\boldsymbol{x}_t) \phi(\boldsymbol{x}_{t+\Delta t}) d\boldsymbol{x}_t$$

代入泰勒展开：
$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\Delta t})] = \int p_t(\boldsymbol{x}_t) \left[\phi(\boldsymbol{x}_t) + \Delta t \, \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla \phi(\boldsymbol{x}_t)\right] d\boldsymbol{x}_t + O(\Delta t^2)$$

分开积分：
$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\Delta t})] = \int p_t(\boldsymbol{x}_t) \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t + \Delta t \int p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t + O(\Delta t^2)$$

**4.6 变量替换与密度演化**

另一方面，期望值也可以用$t+\Delta t$时刻的密度表示：
$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\Delta t})] = \int p_{t+\Delta t}(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x}$$

注意这里积分变量记为$\boldsymbol{x}$（代表$t+\Delta t$时刻的状态）。由于积分变量只是哑变量，我们可以统一记为$\boldsymbol{x}_t$：
$$\int p_{t+\Delta t}(\boldsymbol{x}_t) \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t) \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t + \Delta t \int p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t + O(\Delta t^2)$$

**4.7 时间导数的提取**

移项并除以$\Delta t$：
$$\frac{1}{\Delta t}\int [p_{t+\Delta t}(\boldsymbol{x}_t) - p_t(\boldsymbol{x}_t)] \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t + O(\Delta t)$$

取$\Delta t \to 0$的极限：
$$\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t$$

这就是连续性方程的弱形式。

**4.8 应用分部积分得到强形式**

对右边应用分部积分公式：
$$\int p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t) \cdot \nabla \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = -\int \nabla \cdot [p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t)] \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t$$

（利用了$p$和$\boldsymbol{f}$的衰减性，边界项为零）

因此：
$$\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = -\int \nabla \cdot [p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t)] \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t$$

即：
$$\int \left[\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} + \nabla \cdot (p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t))\right] \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = 0$$

由于这对任意测试函数$\phi$成立，根据基本引理，得到连续性方程的强形式：
$$\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} + \nabla \cdot [p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t)] = 0$$

或写成：
$$\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} = -\nabla \cdot [p_t(\boldsymbol{x}_t) \boldsymbol{f}_t(\boldsymbol{x}_t)]$$

**4.9 连续性方程的展开形式**

应用乘积的散度公式：
$$\nabla \cdot (p\boldsymbol{f}) = \boldsymbol{f} \cdot \nabla p + p\nabla \cdot \boldsymbol{f}$$

连续性方程可以写成：
$$\frac{\partial p_t}{\partial t} = -\boldsymbol{f}_t \cdot \nabla p_t - p_t \nabla \cdot \boldsymbol{f}_t$$

或者用物质导数（material derivative）的记号：
$$\frac{\partial p_t}{\partial t} + \boldsymbol{f}_t \cdot \nabla p_t = -p_t \nabla \cdot \boldsymbol{f}_t$$

左边是沿着流的方向的导数，右边描述了流的压缩或膨胀效应。

### 5. Fokker-Planck方程的完整推导

**5.1 随机微分方程（SDE）**

考虑Itô型随机微分方程：
$$d\boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) dt + g_t d\boldsymbol{w}_t$$

其中$\boldsymbol{w}_t \in \mathbb{R}^n$是标准布朗运动，$\boldsymbol{f}_t$是漂移项，$g_t$是扩散系数。

更一般地，扩散项可以是矩阵形式：
$$d\boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) dt + G_t d\boldsymbol{w}_t$$

其中$G_t \in \mathbb{R}^{n \times n}$。为简化，我们考虑标量扩散$g_t$的情况。

**5.2 Itô公式的应用**

Itô公式是随机微积分的核心工具，对于光滑函数$\phi(\boldsymbol{x})$，有：
$$d\phi(\boldsymbol{x}_t) = \nabla \phi \cdot d\boldsymbol{x}_t + \frac{1}{2} \sum_{i,j} \frac{\partial^2 \phi}{\partial x_i \partial x_j} d\boldsymbol{x}_t^i d\boldsymbol{x}_t^j$$

其中$d\boldsymbol{x}_t^i$是$\boldsymbol{x}_t$的第$i$个分量，且满足Itô乘法规则：
$$dt \cdot dt = 0, \quad dt \cdot d\boldsymbol{w}_t = 0, \quad d\boldsymbol{w}_t^i \cdot d\boldsymbol{w}_t^j = \delta_{ij} dt$$

**5.3 Itô乘法表**

代入$d\boldsymbol{x}_t = \boldsymbol{f}_t dt + g_t d\boldsymbol{w}_t$：
$$d\boldsymbol{x}_t^i d\boldsymbol{x}_t^j = (\boldsymbol{f}_t^i dt + g_t d\boldsymbol{w}_t^i)(\boldsymbol{f}_t^j dt + g_t d\boldsymbol{w}_t^j)$$

展开：
$$d\boldsymbol{x}_t^i d\boldsymbol{x}_t^j = \boldsymbol{f}_t^i \boldsymbol{f}_t^j (dt)^2 + \boldsymbol{f}_t^i g_t dt \cdot d\boldsymbol{w}_t^j + g_t \boldsymbol{f}_t^j d\boldsymbol{w}_t^i \cdot dt + g_t^2 d\boldsymbol{w}_t^i d\boldsymbol{w}_t^j$$

应用Itô规则，保留$O(dt)$项：
$$d\boldsymbol{x}_t^i d\boldsymbol{x}_t^j = g_t^2 \delta_{ij} dt$$

**5.4 Itô公式的展开**

因此，Itô公式变为：
$$d\phi(\boldsymbol{x}_t) = \nabla \phi \cdot (\boldsymbol{f}_t dt + g_t d\boldsymbol{w}_t) + \frac{1}{2} \sum_{i,j} \frac{\partial^2 \phi}{\partial x_i \partial x_j} g_t^2 \delta_{ij} dt$$

简化：
$$d\phi(\boldsymbol{x}_t) = \nabla \phi \cdot \boldsymbol{f}_t dt + \nabla \phi \cdot g_t d\boldsymbol{w}_t + \frac{g_t^2}{2} \sum_i \frac{\partial^2 \phi}{\partial x_i^2} dt$$

即：
$$d\phi(\boldsymbol{x}_t) = \left[\boldsymbol{f}_t \cdot \nabla \phi + \frac{g_t^2}{2} \Delta \phi\right] dt + g_t \nabla \phi \cdot d\boldsymbol{w}_t$$

其中$\Delta \phi = \nabla \cdot \nabla \phi = \sum_i \frac{\partial^2 \phi}{\partial x_i^2}$是拉普拉斯算子。

**5.5 有限差分近似**

将SDE离散化：
$$\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t + g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}$$

其中$\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$是标准正态随机变量。

注意布朗运动增量的方差：$\mathbb{E}[(w_{t+\Delta t} - w_t)^2] = \Delta t$，因此$d\boldsymbol{w}_t$的标准差为$\sqrt{\Delta t}$。

**5.6 测试函数的泰勒展开**

对$\phi(\boldsymbol{x}_{t+\Delta t})$进行二阶泰勒展开：
$$\phi(\boldsymbol{x}_t + \boldsymbol{\delta}) = \phi(\boldsymbol{x}_t) + \boldsymbol{\delta} \cdot \nabla \phi + \frac{1}{2} \sum_{i,j} \delta_i \delta_j \frac{\partial^2 \phi}{\partial x_i \partial x_j} + O(|\boldsymbol{\delta}|^3)$$

其中$\boldsymbol{\delta} = \boldsymbol{f}_t\Delta t + g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}$。

**5.7 随机项的期望计算**

对随机变量$\boldsymbol{\varepsilon}$取期望。首先，关键的期望值：
$$\mathbb{E}[\varepsilon_i] = 0$$
$$\mathbb{E}[\varepsilon_i \varepsilon_j] = \delta_{ij}$$
$$\mathbb{E}[\varepsilon_i \varepsilon_j \varepsilon_k] = 0$$

计算$\boldsymbol{\delta}$的各阶矩：

**一阶项**：
$$\mathbb{E}_\varepsilon[\boldsymbol{\delta}] = \boldsymbol{f}_t\Delta t$$

**二阶项**：
$$\mathbb{E}_\varepsilon[\delta_i \delta_j] = \mathbb{E}_\varepsilon[(\boldsymbol{f}_t^i\Delta t + g_t\sqrt{\Delta t}\varepsilon_i)(\boldsymbol{f}_t^j\Delta t + g_t\sqrt{\Delta t}\varepsilon_j)]$$

展开：
$$= \boldsymbol{f}_t^i\boldsymbol{f}_t^j(\Delta t)^2 + g_t^2\Delta t \mathbb{E}[\varepsilon_i\varepsilon_j] + g_t\sqrt{\Delta t}\Delta t(\boldsymbol{f}_t^i\mathbb{E}[\varepsilon_j] + \boldsymbol{f}_t^j\mathbb{E}[\varepsilon_i])$$

$$= \boldsymbol{f}_t^i\boldsymbol{f}_t^j(\Delta t)^2 + g_t^2\Delta t \delta_{ij}$$

在$\Delta t \to 0$的极限下，保留主导项（$O(\Delta t)$项）：
$$\mathbb{E}_\varepsilon[\delta_i \delta_j] = g_t^2 \delta_{ij} \Delta t + O(\Delta t^2)$$

**5.8 期望值的计算**

对泰勒展开式取期望：
$$\mathbb{E}_\varepsilon[\phi(\boldsymbol{x}_{t+\Delta t})] = \phi(\boldsymbol{x}_t) + \mathbb{E}_\varepsilon[\boldsymbol{\delta}] \cdot \nabla \phi + \frac{1}{2} \sum_{i,j} \mathbb{E}_\varepsilon[\delta_i \delta_j] \frac{\partial^2 \phi}{\partial x_i \partial x_j} + O(\Delta t^{3/2})$$

代入上面计算的期望值：
$$\mathbb{E}_\varepsilon[\phi(\boldsymbol{x}_{t+\Delta t})] = \phi(\boldsymbol{x}_t) + \boldsymbol{f}_t\Delta t \cdot \nabla \phi + \frac{1}{2} \sum_i g_t^2\Delta t \frac{\partial^2 \phi}{\partial x_i^2} + O(\Delta t^{3/2})$$

即：
$$\mathbb{E}_\varepsilon[\phi(\boldsymbol{x}_{t+\Delta t})] = \phi(\boldsymbol{x}_t) + \Delta t\left[\boldsymbol{f}_t \cdot \nabla \phi + \frac{g_t^2}{2} \Delta \phi\right] + O(\Delta t^{3/2})$$

**5.9 对初始分布取期望**

现在对$\boldsymbol{x}_t$的分布$p_t$取期望：
$$\int p_{t+\Delta t}(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x} = \int p_t(\boldsymbol{x}_t) \mathbb{E}_\varepsilon[\phi(\boldsymbol{x}_{t+\Delta t})] d\boldsymbol{x}_t$$

代入上式：
$$\int p_{t+\Delta t}(\boldsymbol{x}) \phi(\boldsymbol{x}) d\boldsymbol{x} = \int p_t(\boldsymbol{x}_t) \phi(\boldsymbol{x}_t) d\boldsymbol{x}_t + \Delta t \int p_t(\boldsymbol{x}_t) \left[\boldsymbol{f}_t \cdot \nabla \phi + \frac{g_t^2}{2} \Delta \phi\right] d\boldsymbol{x}_t + O(\Delta t^{3/2})$$

**5.10 弱形式的推导**

移项并取$\Delta t \to 0$：
$$\int \frac{\partial p_t}{\partial t} \phi \, d\boldsymbol{x} = \int p_t \boldsymbol{f}_t \cdot \nabla \phi \, d\boldsymbol{x} + \int p_t \frac{g_t^2}{2} \Delta \phi \, d\boldsymbol{x}$$

这是Fokker-Planck方程的弱形式。

**5.11 第一项的分部积分**

对第一项应用分部积分：
$$\int p_t \boldsymbol{f}_t \cdot \nabla \phi \, d\boldsymbol{x} = -\int \nabla \cdot (p_t \boldsymbol{f}_t) \phi \, d\boldsymbol{x}$$

**5.12 第二项的分部积分**

对第二项，需要应用两次分部积分。首先：
$$\int p_t \Delta \phi \, d\boldsymbol{x} = \int p_t \nabla \cdot (\nabla \phi) \, d\boldsymbol{x}$$

第一次分部积分（$u = p_t$，$\boldsymbol{v} = \nabla \phi$）：
$$\int p_t \nabla \cdot (\nabla \phi) \, d\boldsymbol{x} = -\int \nabla p_t \cdot \nabla \phi \, d\boldsymbol{x}$$

第二次分部积分（现在$\nabla \phi$看作"测试向量场"）：
$$-\int \nabla p_t \cdot \nabla \phi \, d\boldsymbol{x} = \int \nabla \cdot (\nabla p_t) \phi \, d\boldsymbol{x} = \int \Delta p_t \, \phi \, d\boldsymbol{x}$$

因此：
$$\int p_t \frac{g_t^2}{2} \Delta \phi \, d\boldsymbol{x} = \int \frac{g_t^2}{2} \Delta p_t \, \phi \, d\boldsymbol{x}$$

**5.13 Fokker-Planck方程的强形式**

综合两项：
$$\int \frac{\partial p_t}{\partial t} \phi \, d\boldsymbol{x} = \int \left[-\nabla \cdot (p_t \boldsymbol{f}_t) + \frac{g_t^2}{2} \Delta p_t\right] \phi \, d\boldsymbol{x}$$

由于对任意$\phi$成立，得到Fokker-Planck方程：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t \boldsymbol{f}_t) + \frac{g_t^2}{2} \Delta p_t$$

或写成：
$$\frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t \boldsymbol{f}_t) + \frac{1}{2}\nabla \cdot (g_t^2 \nabla p_t)$$

后一种形式更清楚地显示了方程的对流-扩散结构。

**5.14 Fokker-Planck方程的展开形式**

展开散度项：
$$\nabla \cdot (p_t \boldsymbol{f}_t) = \boldsymbol{f}_t \cdot \nabla p_t + p_t \nabla \cdot \boldsymbol{f}_t$$

$$\frac{g_t^2}{2}\Delta p_t = \frac{g_t^2}{2} \nabla \cdot \nabla p_t$$

完整方程：
$$\frac{\partial p_t}{\partial t} = -\boldsymbol{f}_t \cdot \nabla p_t - p_t \nabla \cdot \boldsymbol{f}_t + \frac{g_t^2}{2} \Delta p_t$$

这可以写成：
$$\frac{\partial p_t}{\partial t} + \boldsymbol{f}_t \cdot \nabla p_t = -p_t \nabla \cdot \boldsymbol{f}_t + \frac{g_t^2}{2} \Delta p_t$$

左边是对流项（沿着流的导数），右边第一项是压缩/膨胀项，第二项是扩散项。

### 6. 边界条件与稳态解

**6.1 边界条件的类型**

对于Fokker-Planck方程，常见的边界条件包括：

**（1）无穷远边界条件**：
$$\lim_{|\boldsymbol{x}| \to \infty} p_t(\boldsymbol{x}) = 0, \quad \lim_{|\boldsymbol{x}| \to \infty} \nabla p_t(\boldsymbol{x}) = \boldsymbol{0}$$

**（2）反射边界条件**（reflecting boundary）：在边界$\partial\Omega$上，法向通量为零：
$$(p_t \boldsymbol{f}_t - \frac{g_t^2}{2}\nabla p_t) \cdot \hat{\boldsymbol{n}} = 0$$

**（3）吸收边界条件**（absorbing boundary）：在边界上，密度为零：
$$p_t(\boldsymbol{x}) = 0, \quad \boldsymbol{x} \in \partial\Omega$$

**6.2 稳态解**

稳态解满足$\frac{\partial p}{\partial t} = 0$，即：
$$-\nabla \cdot (p \boldsymbol{f}) + \frac{g^2}{2} \Delta p = 0$$

或写成：
$$\nabla \cdot \left(p \boldsymbol{f} - \frac{g^2}{2}\nabla p\right) = 0$$

这意味着概率流密度（probability current）：
$$\boldsymbol{J} = p \boldsymbol{f} - \frac{g^2}{2}\nabla p$$
是无散的。

**6.3 详细平衡条件**

如果稳态解满足详细平衡（detailed balance）：
$$\boldsymbol{J} = p \boldsymbol{f} - \frac{g^2}{2}\nabla p = \boldsymbol{0}$$

则有：
$$p \boldsymbol{f} = \frac{g^2}{2}\nabla p$$

即：
$$\boldsymbol{f} = \frac{g^2}{2p}\nabla p = \frac{g^2}{2}\nabla \ln p$$

**6.4 势函数与Boltzmann分布**

如果漂移项可以写成势函数的梯度：
$$\boldsymbol{f}(\boldsymbol{x}) = -\nabla V(\boldsymbol{x})$$

则详细平衡条件变为：
$$-\nabla V = \frac{g^2}{2}\nabla \ln p$$

即：
$$\nabla \ln p = -\frac{2}{g^2}\nabla V$$

积分得：
$$\ln p = -\frac{2V}{g^2} + C$$

因此稳态分布为Boltzmann分布：
$$p(\boldsymbol{x}) \propto \exp\left(-\frac{2V(\boldsymbol{x})}{g^2}\right)$$

如果定义"温度"$T = \frac{g^2}{2}$（在物理单位中），则：
$$p(\boldsymbol{x}) \propto \exp\left(-\frac{V(\boldsymbol{x})}{T}\right)$$

这是统计物理中的经典结果。

**6.5 稳态解的唯一性**

在适当条件下（如$\boldsymbol{f}$满足某种增长条件），稳态解是唯一的。这可以通过能量方法或者Lyapunov函数来证明。

考虑相对熵（Kullback-Leibler散度）：
$$H(p_t | p_\infty) = \int p_t(\boldsymbol{x}) \ln\frac{p_t(\boldsymbol{x})}{p_\infty(\boldsymbol{x})} d\boldsymbol{x}$$

其中$p_\infty$是稳态解。可以证明：
$$\frac{dH}{dt} \leq 0$$

且$\frac{dH}{dt} = 0$当且仅当$p_t = p_\infty$。这证明了系统最终会收敛到稳态。

### 7. 与Langevin动力学的联系

**7.1 Langevin方程**

Langevin方程描述受到摩擦和随机力作用的粒子运动：
$$m\frac{d^2 \boldsymbol{x}}{dt^2} = -\gamma \frac{d\boldsymbol{x}}{dt} - \nabla V(\boldsymbol{x}) + \boldsymbol{\xi}(t)$$

其中$\gamma$是摩擦系数，$V$是势能，$\boldsymbol{\xi}(t)$是随机力，满足：
$$\langle \xi_i(t) \rangle = 0, \quad \langle \xi_i(t)\xi_j(t') \rangle = 2\gamma k_B T \delta_{ij}\delta(t-t')$$

**7.2 过阻尼极限**

在过阻尼（overdamped）极限下，$m \to 0$或$\gamma \to \infty$，惯性项可以忽略：
$$\gamma \frac{d\boldsymbol{x}}{dt} = -\nabla V(\boldsymbol{x}) + \boldsymbol{\xi}(t)$$

即：
$$d\boldsymbol{x} = -\frac{1}{\gamma}\nabla V(\boldsymbol{x}) dt + \frac{1}{\gamma}\boldsymbol{\xi}(t)dt$$

将随机力写成布朗运动形式，$\boldsymbol{\xi}(t)dt = \sqrt{2\gamma k_B T} d\boldsymbol{w}_t$：
$$d\boldsymbol{x} = -\frac{1}{\gamma}\nabla V(\boldsymbol{x}) dt + \sqrt{\frac{2k_B T}{\gamma}} d\boldsymbol{w}_t$$

**7.3 对应的Fokker-Planck方程**

根据前面的推导，这个SDE对应的Fokker-Planck方程为：
$$\frac{\partial p}{\partial t} = \nabla \cdot \left[\frac{1}{\gamma}\nabla V \, p + \frac{k_B T}{\gamma}\nabla p\right]$$

即：
$$\frac{\partial p}{\partial t} = \frac{1}{\gamma}\nabla \cdot \left[\nabla V \, p + k_B T\nabla p\right]$$

这称为**Smoluchowski方程**，是Fokker-Planck方程在过阻尼Langevin动力学中的具体形式。

**7.4 平衡态验证**

稳态解满足：
$$\nabla V \, p_\infty + k_B T\nabla p_\infty = \boldsymbol{0}$$

即：
$$\nabla V \, p_\infty = -k_B T\nabla p_\infty$$

$$\nabla V = -k_B T\nabla \ln p_\infty$$

积分：
$$p_\infty(\boldsymbol{x}) = \frac{1}{Z}\exp\left(-\frac{V(\boldsymbol{x})}{k_B T}\right)$$

其中$Z = \int \exp(-V/k_B T)d\boldsymbol{x}$是配分函数。这正是经典统计力学的正则分布！

### 8. 数值求解方法

**8.1 有限差分方法**

对于一维Fokker-Planck方程：
$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}(f(x)p) + \frac{g^2}{2}\frac{\partial^2 p}{\partial x^2}$$

采用前向Euler时间离散和中心差分空间离散：
$$\frac{p_i^{n+1} - p_i^n}{\Delta t} = -\frac{(fp)_{i+1/2}^n - (fp)_{i-1/2}^n}{\Delta x} + \frac{g^2}{2}\frac{p_{i+1}^n - 2p_i^n + p_{i-1}^n}{(\Delta x)^2}$$

其中$p_i^n \approx p(x_i, t_n)$。

**8.2 上风格式**

对流项需要特别处理以保证数值稳定性。采用上风（upwind）格式：
$$(fp)_{i+1/2} = \begin{cases}
f_{i+1/2}p_i & \text{if } f_{i+1/2} \geq 0 \\
f_{i+1/2}p_{i+1} & \text{if } f_{i+1/2} < 0
\end{cases}$$

这保证了数值格式的单调性和稳定性。

**8.3 CFL条件**

数值稳定性要求时间步长满足CFL（Courant-Friedrichs-Lewy）条件：
$$\Delta t \leq \min\left\{\frac{\Delta x}{|f|_{\max}}, \frac{(\Delta x)^2}{g^2}\right\}$$

第一项来自对流项，第二项来自扩散项。

**8.4 蒙特卡洛方法**

另一种方法是直接模拟SDE：
$$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \boldsymbol{f}(\boldsymbol{x}_n)\Delta t + g\sqrt{\Delta t}\boldsymbol{\varepsilon}_n$$

其中$\boldsymbol{\varepsilon}_n \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。

通过大量粒子的统计，可以估计概率密度：
$$p_t(\boldsymbol{x}) \approx \frac{1}{N}\sum_{i=1}^N K_h(\boldsymbol{x} - \boldsymbol{x}_t^{(i)})$$

其中$K_h$是带宽为$h$的核函数（如高斯核）。

**8.5 谱方法**

对于周期边界条件，可以使用傅里叶谱方法。将$p(\boldsymbol{x}, t)$展开为傅里叶级数：
$$p(\boldsymbol{x}, t) = \sum_{\boldsymbol{k}} \hat{p}_{\boldsymbol{k}}(t) e^{i\boldsymbol{k}\cdot\boldsymbol{x}}$$

Fokker-Planck方程在傅里叶空间变为ODE：
$$\frac{d\hat{p}_{\boldsymbol{k}}}{dt} = -i\boldsymbol{k} \cdot \widehat{(fp)}_{\boldsymbol{k}} - \frac{g^2}{2}|\boldsymbol{k}|^2 \hat{p}_{\boldsymbol{k}}$$

这可以高精度地求解，但卷积项$\widehat{(fp)}_{\boldsymbol{k}}$的计算需要FFT。

**8.6 有限元方法**

将$p$在测试函数空间中展开：
$$p(\boldsymbol{x}, t) = \sum_j p_j(t) \varphi_j(\boldsymbol{x})$$

其中$\{\varphi_j\}$是有限元基函数。将此代入弱形式：
$$\int \frac{\partial p}{\partial t} \phi \, d\boldsymbol{x} = \int \left[-\nabla \cdot (p\boldsymbol{f}) + \frac{g^2}{2}\Delta p\right] \phi \, d\boldsymbol{x}$$

取$\phi = \varphi_i$，得到ODE系统：
$$M\frac{d\boldsymbol{p}}{dt} = A\boldsymbol{p}$$

其中$M_{ij} = \int \varphi_i\varphi_j d\boldsymbol{x}$是质量矩阵，$A_{ij}$包含对流和扩散项的贡献。

### 9. 物理意义与多角度理解

**9.1 概率论视角**

从概率论角度，Fokker-Planck方程描述了随机过程$\boldsymbol{x}_t$的概率密度$p_t(\boldsymbol{x})$如何演化。它是Chapman-Kolmogorov方程在连续时间、连续状态空间中的微分形式。

**转移概率密度**：定义$\mathcal{P}(t, \boldsymbol{x} | s, \boldsymbol{y})$为从$(s, \boldsymbol{y})$转移到$(t, \boldsymbol{x})$的条件概率密度，则：
$$p_t(\boldsymbol{x}) = \int \mathcal{P}(t, \boldsymbol{x} | 0, \boldsymbol{y}) p_0(\boldsymbol{y}) d\boldsymbol{y}$$

**9.2 偏微分方程视角**

从PDE角度，Fokker-Planck方程是一个**抛物型方程**，结合了：
- **对流项**：$-\nabla \cdot (p\boldsymbol{f})$，描述确定性漂移
- **扩散项**：$\frac{g^2}{2}\Delta p$，描述随机扩散

这类似于对流-扩散方程：
$$\frac{\partial u}{\partial t} + \boldsymbol{v} \cdot \nabla u = D\Delta u$$

但Fokker-Planck方程的对流项有特殊形式（守恒形式）。

**9.3 统计力学视角**

在统计力学中，Fokker-Planck方程描述了相空间中的概率流。对于Hamiltonian系统加上耗散和噪声：
$$\dot{\boldsymbol{q}} = \frac{\partial H}{\partial \boldsymbol{p}}, \quad \dot{\boldsymbol{p}} = -\frac{\partial H}{\partial \boldsymbol{q}} - \gamma\boldsymbol{p} + \boldsymbol{\xi}(t)$$

其相空间分布满足相应的Fokker-Planck方程。

**9.4 信息论视角**

从信息论角度，Fokker-Planck方程描述了系统的Shannon熵：
$$S(t) = -\int p_t(\boldsymbol{x}) \ln p_t(\boldsymbol{x}) d\boldsymbol{x}$$

的演化。可以证明，对于满足详细平衡的系统：
$$\frac{dS}{dt} \geq 0$$

这是热力学第二定律的微观体现（$H$定理）。

**9.5 机器学习视角**

在扩散模型（diffusion models）中：
- **前向过程**：向数据添加噪声，对应SDE：
  $$d\boldsymbol{x} = -\frac{1}{2}\boldsymbol{x}dt + d\boldsymbol{w}$$

- **逆向过程**：从噪声恢复数据，对应逆时间SDE：
  $$d\boldsymbol{x} = \left[-\frac{1}{2}\boldsymbol{x} + \nabla \ln p_t(\boldsymbol{x})\right]dt + d\bar{\boldsymbol{w}}$$

Fokker-Planck方程用于分析分布$p_t$的演化。

### 10. 总结与推广

**10.1 方法总结**

测试函数法的核心步骤：
1. 将动力学方程（ODE/SDE）离散化
2. 对测试函数进行泰勒展开
3. 对随机性取期望（如有）
4. 对初始分布取期望，建立弱形式
5. 应用分部积分转移导数
6. 由测试函数的任意性得到强形式

**10.2 推广方向**

该方法可以推广到：
- **空间相关的扩散**：$G_t(\boldsymbol{x})$是矩阵函数
- **跳跃过程**：包含Poisson跳跃的过程
- **分数阶导数**：非局部算子
- **非线性方程**：如Burgers方程
- **无穷维情况**：随机偏微分方程（SPDE）

**10.3 理论意义**

测试函数法是现代PDE理论的基础，连接了：
- 经典的强解理论
- 现代的弱解和分布理论
- 数值分析中的变分方法
- 物理中的守恒律

它为理解随机过程、偏微分方程和概率分布的演化提供了统一的框架。

