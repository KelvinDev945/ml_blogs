---
title: 流形上的最速下降：1.  SGD + 超球面
slug: 流形上的最速下降1-sgd-超球面
date: 2025-08-01
tags: 不等式, 优化器, 约束, 最速下降, 生成模型
status: pending
---

# 流形上的最速下降：1.  SGD + 超球面

**原文链接**: [https://spaces.ac.cn/archives/11196](https://spaces.ac.cn/archives/11196)

**发布日期**: 

---

类似“梯度的反方向是下降最快的方向”的描述，经常用于介绍梯度下降（SGD）的原理。然而，这句话是有条件的，比如“方向”在数学上是单位向量，它依赖于“范数（模长）”的定义，不同范数的结论也不同，[Muon](/archives/10592)实际上就是给矩阵参数换了个谱范数，从而得到了新的下降方向。又比如，当我们从无约束优化转移到约束优化时，下降最快的方向也未必是梯度的反方向。

为此，在这篇文章中，我们将新开一个系列，以“约束”为主线，重新审视“最速下降”这一命题，探查不同条件下的“下降最快的方向”指向何方。

## 优化原理 #

作为第一篇文章，我们先从SGD出发，理解“梯度的反方向是下降最快的方向”这句话背后的数学意义，然后应用于超球面上的优化。不过在此之前，笔者还想带大家重温一下[《Muon续集：为什么我们选择尝试Muon？》](/archives/10739)所提的关于优化器的“**最小作用量原理（Least Action Principle）** ”。

这个原理尝试回答“什么才是好的优化器”。首先，我们肯定是希望模型收敛速度越快越好，但由于神经网络本身的复杂性，如果步子迈得太大，那么反而容易训崩。所以，一个好的优化器应该是又**稳** 又**快** ，最好是不用大改模型，但却可以明显降低损失，写成数学形式是  
\begin{equation}\min_{\Delta \boldsymbol{w}} \mathcal{L}(\boldsymbol{w} +\Delta\boldsymbol{w}) \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{w})\leq \eta\end{equation}  
其实$\mathcal{L}$是损失函数，$\boldsymbol{w}\in\mathbb{R}^n$是参数向量，$\Delta \boldsymbol{w}$是更新量，$\rho(\Delta\boldsymbol{w})$是更新量$\Delta\boldsymbol{w}$大小的某种度量。上述目标很直观，就是在“步子”不超过$\eta$（**稳** ）的前提下，寻找让损失函数下降最多（**快** ）的更新量，这便是“最小作用量原理”的数学含义，也是“最速下降”的数学含义。

## 目标转化 #

假设$\eta$足够小，那么$\Delta\boldsymbol{w}$也足够小，以至于一阶近似足够准确，那么我们就可以将$\mathcal{L}(\boldsymbol{w} +\Delta\boldsymbol{w})$替换为$\mathcal{L}(\boldsymbol{w}) + \langle\boldsymbol{g},\Delta\boldsymbol{w}\rangle$，其中$\boldsymbol{g} = \nabla_{\boldsymbol{w}}\mathcal{L}(\boldsymbol{w})$，得到等效目标  
\begin{equation}\min_{\Delta \boldsymbol{w}} \langle\boldsymbol{g},\Delta\boldsymbol{w}\rangle \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{w})\leq \eta\end{equation}  
这就将优化目标简化成$\Delta \boldsymbol{w}$的线性函数，降低了求解难度。进一步地，我们设$\Delta \boldsymbol{w} = -\kappa \boldsymbol{\varphi}$，其中$\rho(\boldsymbol{\varphi})=1$，那么上述目标等价于  
\begin{equation}\max_{\kappa,\boldsymbol{\varphi}} \kappa\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \rho(\boldsymbol{\varphi}) = 1, \,\,\kappa\in[0, \eta]\end{equation}  
假设我们至少能找到一个满足条件的$\boldsymbol{\varphi}$使得$\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle\geq 0$，那么有$\max\limits_{\kappa\in[0,\eta]} \kappa\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \eta\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle$，也就是$\kappa$的优化可以事先求出来，结果是$\kappa=\eta$，最终等效于只剩下$\boldsymbol{\varphi}$的优化  
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \rho(\boldsymbol{\varphi}) = 1\label{eq:core}\end{equation}  
这里的$\boldsymbol{\varphi}$满足某种“模长”$\rho(\boldsymbol{\varphi})$等于1的条件，所以它代表了某种“方向向量”的定义，最大化它与梯度$\boldsymbol{g}$的内积，就意味着寻找让损失下降最快的方向（即$\boldsymbol{\varphi}$的反方向）。

## 梯度下降 #

从式$\eqref{eq:core}$可以看出，对于“下降最快的方向”，唯一不确定的是度量$\rho$，这是优化器里边很本质的一个先验（Inductive Bias），不同的度量将会得到不同的最速下降方向。比较简单的就是L2范数或者说欧几里得范数$\rho(\boldsymbol{\varphi})=\Vert \boldsymbol{\varphi} \Vert_2$，也就是我们通常意义下的模长，这时候我们有柯西不等式：  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g}\Vert_2 \Vert\boldsymbol{\varphi}\Vert_2 = \Vert\boldsymbol{g}\Vert_2\end{equation}  
等号成立的条件是$\boldsymbol{\varphi} \propto\boldsymbol{g}$，加上模长为1的条件，我们得到$\boldsymbol{\varphi}=\boldsymbol{g}/\Vert\boldsymbol{g}\Vert_2$，这正是梯度的方向。所以说，“梯度的反方向是下降最快的方向”前提是所选取的度量是欧几里得范数。更一般地，我们考虑$p$范数  
\begin{equation}\rho(\boldsymbol{\varphi}) = \Vert\boldsymbol{\varphi}\Vert_p = \sqrt[\uproot{10}p]{\sum_{i=1}^n |\varphi_i|^p}\end{equation}  
柯西不等式可以推广成[Hölder不等式](https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality)：  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g}\Vert_q \Vert\boldsymbol{\varphi}\Vert_p = \Vert\boldsymbol{g}\Vert_q,\qquad 1/p + 1/q=1\end{equation}  
等号成立的条件$\boldsymbol{\varphi}^{[p]} \propto\boldsymbol{g}^{[q]}$，所以解得  
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}}\boldsymbol{\varphi} = \frac{\boldsymbol{g}^{[q/p]}}{\Vert\boldsymbol{g}^{[q/p]}\Vert_p},\qquad \boldsymbol{g}^{[\alpha]} \triangleq \big[\sign(g_1) |g_1|^{\alpha},\sign(g_2) |g_2|^{\alpha},\cdots,\sign(g_n) |g_n|^{\alpha}\big]\end{equation}  
以它为方向向量的优化器叫做pbSGD，出自[《pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non-Convex Optimization》](https://www.ijcai.org/proceedings/2020/451)。它有两个特例，一是$p=q=2$是退化为SGD，二是$p\to\infty$时$q\to 1$，此时$|g_i|^{q/p}\to 1$，更新方向为梯度的符号函数，即SignSGD。

## 超球面上 #

前面的讨论中，我们只是对参数的增量$\Delta\boldsymbol{w}$施加了约束，接下来我们希望的是给参数$\boldsymbol{w}$也添加约束。具体来说，我们假设参数$\boldsymbol{w}$位于单位球面上，我们希望更新后的参数$\boldsymbol{w}+\Delta\boldsymbol{w}$依然位于单位球面上（参考[《Hypersphere》](https://docs.modula.systems/algorithms/manifold/hypersphere/)）。从目标$\eqref{eq:core}$出发，我们可以将新目标写成  
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \Vert\boldsymbol{\varphi}\Vert_2 = 1,\,\,\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2 = 1,\,\,\Vert\boldsymbol{w}\Vert_2=1\end{equation}  
我们依然贯彻“$\eta$足够小，一阶近似够用”的原则，得到  
\begin{equation}1 = \Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2^2 = \Vert\boldsymbol{w}\Vert_2^2 - 2\eta\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle + \eta^2 \Vert\boldsymbol{\varphi}\Vert_2^2\approx 1 - 2\eta\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle\end{equation}  
所以这相当于将约束转化为线性形式$\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0$。为了求解新的目标，我们引入待定系数$\lambda$，写出  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle + \lambda\langle\boldsymbol{w},\boldsymbol{\varphi}\rangle =\langle \boldsymbol{g} + \lambda\boldsymbol{w},\boldsymbol{\varphi}\rangle\leq \Vert\boldsymbol{g} + \lambda\boldsymbol{w}\Vert_2 \Vert\boldsymbol{\varphi}\Vert_2 = \Vert\boldsymbol{g} + \lambda\boldsymbol{w}\Vert_2\end{equation}  
等号成立的条件是$\boldsymbol{\varphi}\propto \boldsymbol{g} + \lambda\boldsymbol{w}$，再加上$\Vert\boldsymbol{\varphi}\Vert_2=1,\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0,\Vert\boldsymbol{w}\Vert_2=1$的条件，可以解得  
\begin{equation}\boldsymbol{\varphi} = \frac{\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}}{\Vert\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}\Vert_2}\end{equation}  
注意现在有$\Vert\boldsymbol{w}\Vert_2=1,\Vert\boldsymbol{\varphi}\Vert_2=1$，并且$\boldsymbol{w}$和$\boldsymbol{\varphi}$是正交的，那么$\boldsymbol{w} - \eta\boldsymbol{\varphi}$的模长是并不是精确地等于1，而是$\sqrt{1 + \eta^2}=1 + \eta^2/2 + \cdots$，精确到$\mathcal{O}(\eta^2)$，这跟我们前面的假设“$\eta$的一阶项够用”吻合。如果想更新后的参数模长严格等于1，那么可以在更新规则上多加一步缩回操作：  
\begin{equation}\boldsymbol{w}\quad\leftarrow\quad \frac{\boldsymbol{w} - \eta\boldsymbol{\varphi}}{\sqrt{1 + \eta^2}}\end{equation}

## 几何意义 #

刚才我们通过“一阶近似够用”原则，将非线性约束$\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2 = 1$简化为线性约束$\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0$，后者的几何意义是“与$\boldsymbol{w}$垂直”，这还有个更专业的说法，叫做$\Vert\boldsymbol{w}\Vert_2=1$的“切空间”，而$\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}$这一运算，正对应于把梯度$\boldsymbol{g}$投影到切空间的投影运算。

所以很幸运，球面上的SGD有非常清晰的几何意义，如下图所示：  


[![球面上的最速下降-几何意义](/usr/uploads/2025/07/1221229703.png)](/usr/uploads/2025/07/1221229703.png "点击查看原图")

球面上的最速下降-几何意义

相信很多读者都喜欢这种几何视角，它确实让人赏心悦目。但这是还是要提醒大家一下，应当优先认真理解代数求解过程，因为清晰的几何意义很多时候都只是一种奢望，属于可遇而不可求的，大多数情况下复杂的代数过程才是本质。

## 一般结果 #

接下来是不是有读者想要将它推广到一般的$p$范数？让我们一起来尝试下，看看会遇到什么困难。这时候问题是：  
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \Vert\boldsymbol{\varphi}\Vert_p = 1,\,\,\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_p = 1,\,\,\Vert\boldsymbol{w}\Vert_p=1\end{equation}  
一阶近似将$\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_p = 1$转换成$\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = 0$，然后引入待定系数$\lambda$：  
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle + \lambda\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = \langle \boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]}\Vert_q \Vert\boldsymbol{\varphi}\Vert_p = \Vert\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]}\Vert_q  
\end{equation}  
取等号的条件是  
\begin{equation}\boldsymbol{\varphi} = \frac{(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})^{[q/p]}}{\Vert(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})^{[q/p]}\Vert_p}\end{equation}  
到目前为止，都没有实质困难。然而，接下来我们需要寻找$\lambda$，使得$\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = 0$，当$p \neq 2$时这是一个复杂的非线性方程，并没有很好的求解办法（当然，一旦求解出来，我们就肯定能得到最优解，这是Hölder不等式保证的）。所以，一般$p$的求解我们只能止步于此，等遇到$p\neq 2$的实例时我们再具体探寻数值求解方法。

不过除了$p=2$，我们还可以尝试求解$p\to\infty$，此时$\boldsymbol{\varphi}=\sign(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})$，条件$\Vert\boldsymbol{w}\Vert_p=1$给出$|w_1|,|w_2|,\cdots,|w_n|$的最大值等于1。如果进一步假设最大值只有一个，那么$\boldsymbol{w}^{[p-1]}$是一个one hot向量，绝对值最大值的位置为$\pm 1$，其余是零，这时候就可以解出$\lambda$，结果是把最大值位置的梯度裁剪成零。

## 文章小结 #

这篇文章新开一个系列，主要围绕“等式约束”来讨论优化问题，试图给一些常见的约束条件寻找“下降最快的方向”。作为第一篇文章，本文讨论了“超球面”约束下的SGD变体。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11196>_

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

苏剑林. (Aug. 01, 2025). 《流形上的最速下降：1. SGD + 超球面 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11196>

@online{kexuefm-11196,  
title={流形上的最速下降：1. SGD + 超球面},  
author={苏剑林},  
year={2025},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/11196}},  
} 


---

## 公式推导与注释

本节将从微分几何、黎曼优化和深度学习三个角度，为超球面上的最速下降提供极详细的数学推导。

### 1. 超球面的微分几何基础

#### 1.1 超球面的定义

**定义 1.1（超球面流形）**：$n$维单位超球面定义为：
$$
\mathcal{S}^{n-1} = \{\boldsymbol{w} \in \mathbb{R}^n : \|\boldsymbol{w}\|_2 = 1\} = \{\boldsymbol{w} \in \mathbb{R}^n : \langle\boldsymbol{w}, \boldsymbol{w}\rangle = 1\}
$$

这是一个$(n-1)$维的嵌入子流形，嵌入在$n$维欧氏空间$\mathbb{R}^n$中。

**性质**：
- $\mathcal{S}^{n-1}$是紧致的（有界且闭合）
- $\mathcal{S}^{n-1}$是光滑的（无处不可微）
- $\mathcal{S}^{n-1}$具有恒定的正曲率$K = 1$

**例子**：
- $\mathcal{S}^1$：单位圆（1维流形在$\mathbb{R}^2$中）
- $\mathcal{S}^2$：单位球面（2维流形在$\mathbb{R}^3$中）
- 神经网络中：权重归一化层的参数空间

#### 1.2 切空间（Tangent Space）

**定义 1.2（切空间）**：在点$\boldsymbol{w} \in \mathcal{S}^{n-1}$处的切空间$T_{\boldsymbol{w}}\mathcal{S}^{n-1}$定义为：
$$
T_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\boldsymbol{v} \in \mathbb{R}^n : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}
$$

**推导**：设$\gamma(t)$是$\mathcal{S}^{n-1}$上经过$\boldsymbol{w}$的任意光滑曲线，满足$\gamma(0) = \boldsymbol{w}$。由于曲线始终在球面上，有：
$$
\|\gamma(t)\|_2^2 = \langle\gamma(t), \gamma(t)\rangle = 1, \quad \forall t
$$

对$t$求导：
$$
\frac{d}{dt}\langle\gamma(t), \gamma(t)\rangle = 2\langle\gamma'(t), \gamma(t)\rangle = 0
$$

在$t=0$处取值：
$$
\langle\gamma'(0), \gamma(0)\rangle = \langle\gamma'(0), \boldsymbol{w}\rangle = 0
$$

这表明曲线的切向量$\gamma'(0)$与位置向量$\boldsymbol{w}$正交。所有这样的切向量构成切空间。

**几何意义**：切空间是与球面在$\boldsymbol{w}$点相切的$(n-1)$维超平面。

**维度验证**：
$$
\dim(T_{\boldsymbol{w}}\mathcal{S}^{n-1}) = \dim(\{\boldsymbol{v} : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}) = n - 1
$$
这与流形的维度一致。

#### 1.3 法空间（Normal Space）

**定义 1.3（法空间）**：在点$\boldsymbol{w} \in \mathcal{S}^{n-1}$处的法空间$N_{\boldsymbol{w}}\mathcal{S}^{n-1}$定义为：
$$
N_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\alpha\boldsymbol{w} : \alpha \in \mathbb{R}\} = \text{span}\{\boldsymbol{w}\}
$$

**正交分解定理**：环境空间$\mathbb{R}^n$可以正交分解为切空间和法空间：
$$
\mathbb{R}^n = T_{\boldsymbol{w}}\mathcal{S}^{n-1} \oplus N_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

任意向量$\boldsymbol{v} \in \mathbb{R}^n$可唯一分解为：
$$
\boldsymbol{v} = \boldsymbol{v}_{\text{tan}} + \boldsymbol{v}_{\text{nor}}
$$
其中：
- 切向分量：$\boldsymbol{v}_{\text{tan}} = \boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$
- 法向分量：$\boldsymbol{v}_{\text{nor}} = \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} \in N_{\boldsymbol{w}}\mathcal{S}^{n-1}$

**验证正交性**：
$$
\langle\boldsymbol{v}_{\text{tan}}, \boldsymbol{v}_{\text{nor}}\rangle = \langle\boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}, \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}\rangle = \langle\boldsymbol{v}, \boldsymbol{w}\rangle\langle\boldsymbol{v}, \boldsymbol{w}\rangle - \langle\boldsymbol{v}, \boldsymbol{w}\rangle^2\langle\boldsymbol{w}, \boldsymbol{w}\rangle = 0
$$

#### 1.4 投影算子（Projection Operator）

**定义 1.4（切空间投影）**：投影到切空间的算子$\mathcal{P}_{\boldsymbol{w}}: \mathbb{R}^n \to T_{\boldsymbol{w}}\mathcal{S}^{n-1}$定义为：
$$
\mathcal{P}_{\boldsymbol{w}}(\boldsymbol{v}) = \boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w} = (\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top})\boldsymbol{v}
$$

其中$\boldsymbol{I}$是$n \times n$单位矩阵，$\boldsymbol{w}\boldsymbol{w}^{\top}$是秩1投影矩阵。

**投影矩阵性质**：设$\boldsymbol{P}_{\boldsymbol{w}} = \boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top}$，则：

1. **幂等性**（投影两次等于投影一次）：
$$
\boldsymbol{P}_{\boldsymbol{w}}^2 = (\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top})^2 = \boldsymbol{I} - 2\boldsymbol{w}\boldsymbol{w}^{\top} + \boldsymbol{w}\boldsymbol{w}^{\top}\boldsymbol{w}\boldsymbol{w}^{\top} = \boldsymbol{I} - 2\boldsymbol{w}\boldsymbol{w}^{\top} + \boldsymbol{w}\boldsymbol{w}^{\top} = \boldsymbol{P}_{\boldsymbol{w}}
$$

2. **对称性**：
$$
\boldsymbol{P}_{\boldsymbol{w}}^{\top} = (\boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top})^{\top} = \boldsymbol{I} - \boldsymbol{w}\boldsymbol{w}^{\top} = \boldsymbol{P}_{\boldsymbol{w}}
$$

3. **正交投影**：
$$
\langle\mathcal{P}_{\boldsymbol{w}}(\boldsymbol{v}), \boldsymbol{w}\rangle = \langle\boldsymbol{v} - \langle\boldsymbol{v}, \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{w}\rangle = \langle\boldsymbol{v}, \boldsymbol{w}\rangle - \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0
$$

### 2. 黎曼梯度与投影公式

#### 2.1 约束优化问题的设定

考虑超球面上的约束优化问题：
$$
\min_{\boldsymbol{w} \in \mathcal{S}^{n-1}} f(\boldsymbol{w})
$$

其中$f: \mathbb{R}^n \to \mathbb{R}$是定义在环境空间上的光滑目标函数。

**拉格朗日乘数法推导**：引入约束$g(\boldsymbol{w}) = \|\boldsymbol{w}\|^2 - 1 = 0$，构造拉格朗日函数：
$$
\mathcal{L}(\boldsymbol{w}, \lambda) = f(\boldsymbol{w}) + \lambda(\|\boldsymbol{w}\|^2 - 1)
$$

最优性条件（KKT条件）：
$$
\nabla_{\boldsymbol{w}}\mathcal{L} = \nabla f(\boldsymbol{w}) + 2\lambda\boldsymbol{w} = \boldsymbol{0}
$$

这表明欧氏梯度$\nabla f(\boldsymbol{w})$与位置向量$\boldsymbol{w}$平行，即梯度在法空间中。但我们需要的是切空间中的梯度方向。

#### 2.2 黎曼梯度的定义

**定义 2.1（黎曼梯度）**：在黎曼流形$(\mathcal{M}, g)$上，函数$f$在点$\boldsymbol{w}$处的黎曼梯度$\text{grad} f(\boldsymbol{w})$定义为切空间中唯一满足以下条件的向量：
$$
g_{\boldsymbol{w}}(\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}) = Df(\boldsymbol{w})[\boldsymbol{\xi}], \quad \forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{M}
$$

其中$Df(\boldsymbol{w})[\boldsymbol{\xi}] = \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle$是$f$在$\boldsymbol{w}$处沿$\boldsymbol{\xi}$方向的方向导数。

**对于超球面**：超球面继承了欧氏空间的黎曼度量，即$g_{\boldsymbol{w}}(\boldsymbol{\xi}, \boldsymbol{\eta}) = \langle\boldsymbol{\xi}, \boldsymbol{\eta}\rangle$。因此：
$$
\langle\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}\rangle = \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle, \quad \forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

#### 2.3 黎曼梯度的投影公式推导

**定理 2.1**：超球面上的黎曼梯度为欧氏梯度在切空间上的投影：
$$
\text{grad} f(\boldsymbol{w}) = \mathcal{P}_{\boldsymbol{w}}(\nabla f(\boldsymbol{w})) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
$$

**证明**：

**步骤1**：利用正交分解，将欧氏梯度分解为：
$$
\nabla f(\boldsymbol{w}) = \nabla f(\boldsymbol{w})_{\text{tan}} + \nabla f(\boldsymbol{w})_{\text{nor}}
$$
其中：
$$
\nabla f(\boldsymbol{w})_{\text{tan}} = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$
$$
\nabla f(\boldsymbol{w})_{\text{nor}} = \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w} \in N_{\boldsymbol{w}}\mathcal{S}^{n-1}
$$

**步骤2**：对于任意$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$，计算内积：
$$
\langle\nabla f(\boldsymbol{w})_{\text{tan}}, \boldsymbol{\xi}\rangle = \langle\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{\xi}\rangle
$$
$$
= \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle
$$

由于$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$，有$\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle = 0$，因此：
$$
\langle\nabla f(\boldsymbol{w})_{\text{tan}}, \boldsymbol{\xi}\rangle = \langle\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle = Df(\boldsymbol{w})[\boldsymbol{\xi}]
$$

**步骤3**：由黎曼梯度的定义和唯一性，得：
$$
\text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w})_{\text{tan}} = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
$$

**几何解释**：法向分量$\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$对应于约束力（离开或靠近球心的力），不影响沿球面的运动。只有切向分量才对球面上的优化有贡献。

**计算复杂度分析**：
- 计算$\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle$：$O(n)$次乘加
- 计算$\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$：$O(n)$次标量乘法
- 计算最终投影：$O(n)$次减法
- **总复杂度**：$O(n)$，与计算梯度本身的复杂度相同

### 3. 超球面上的最速下降方向

#### 3.1 最速下降问题的精确表述

**问题 3.1**：给定当前点$\boldsymbol{w} \in \mathcal{S}^{n-1}$，寻找切向量$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$使得：
$$
\boldsymbol{\xi}^* = \arg\min_{\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}, \|\boldsymbol{\xi}\|_2 = 1} Df(\boldsymbol{w})[\boldsymbol{\xi}]
$$

等价于最大化问题：
$$
\boldsymbol{\xi}^* = \arg\max_{\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}, \|\boldsymbol{\xi}\|_2 = 1} -Df(\boldsymbol{w})[\boldsymbol{\xi}] = \arg\max_{\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}, \|\boldsymbol{\xi}\|_2 = 1} \langle-\nabla f(\boldsymbol{w}), \boldsymbol{\xi}\rangle
$$

这正是文中式$\eqref{eq:core}$在超球面约束下的形式。

#### 3.2 最速下降方向的求解

**定理 3.1**：超球面上的最速下降方向为：
$$
\boldsymbol{\xi}^* = -\frac{\text{grad} f(\boldsymbol{w})}{\|\text{grad} f(\boldsymbol{w})\|_2} = -\frac{\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}}{\|\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}\|_2}
$$

**证明**：

设$\boldsymbol{g} = \nabla f(\boldsymbol{w})$，$\boldsymbol{g}_{\text{tan}} = \text{grad} f(\boldsymbol{w}) = \boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}$。

要最大化$\langle-\boldsymbol{g}, \boldsymbol{\xi}\rangle$，约束条件为$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$且$\|\boldsymbol{\xi}\|_2 = 1$。

由于$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$，有$\langle\boldsymbol{\xi}, \boldsymbol{w}\rangle = 0$，因此：
$$
\langle-\boldsymbol{g}, \boldsymbol{\xi}\rangle = \langle-\boldsymbol{g} + \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{\xi}\rangle = \langle-\boldsymbol{g}_{\text{tan}}, \boldsymbol{\xi}\rangle
$$

应用柯西-施瓦茨不等式：
$$
\langle-\boldsymbol{g}_{\text{tan}}, \boldsymbol{\xi}\rangle \leq \|\boldsymbol{g}_{\text{tan}}\|_2 \|\boldsymbol{\xi}\|_2 = \|\boldsymbol{g}_{\text{tan}}\|_2
$$

等号成立当且仅当$\boldsymbol{\xi} = -\boldsymbol{g}_{\text{tan}}/\|\boldsymbol{g}_{\text{tan}}\|_2$。

**归一化验证**：
$$
\left\|-\frac{\boldsymbol{g}_{\text{tan}}}{\|\boldsymbol{g}_{\text{tan}}\|_2}\right\|_2 = 1 \quad \checkmark
$$

**切空间验证**：
$$
\left\langle-\frac{\boldsymbol{g}_{\text{tan}}}{\|\boldsymbol{g}_{\text{tan}}\|_2}, \boldsymbol{w}\right\rangle = -\frac{1}{\|\boldsymbol{g}_{\text{tan}}\|_2}\langle\boldsymbol{g}_{\text{tan}}, \boldsymbol{w}\rangle = -\frac{1}{\|\boldsymbol{g}_{\text{tan}}\|_2}\langle\boldsymbol{g} - \langle\boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}, \boldsymbol{w}\rangle = 0 \quad \checkmark
$$

#### 3.3 退化情况分析

**退化情况**：当$\text{grad} f(\boldsymbol{w}) = \boldsymbol{0}$时，即$\nabla f(\boldsymbol{w}) = \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$时，黎曼梯度消失。

**几何意义**：此时欧氏梯度完全在法空间中，沿球面的任意切方向函数值都不变（一阶近似），$\boldsymbol{w}$是约束优化问题的临界点。

**判定条件**：
$$
\|\boldsymbol{g}_{\text{tan}}\|_2 = 0 \Leftrightarrow \nabla f(\boldsymbol{w}) \parallel \boldsymbol{w}
$$

### 4. 测地线与指数映射

#### 4.1 测地线的定义与性质

**定义 4.1（测地线）**：黎曼流形上的测地线$\gamma(t)$是满足以下微分方程的曲线：
$$
\nabla_{\dot{\gamma}(t)}\dot{\gamma}(t) = \boldsymbol{0}
$$

其中$\nabla$是黎曼联络，$\dot{\gamma}(t) = d\gamma/dt$是切向量。

**物理意义**：测地线是流形上的"直线"，具有局部最短路径性质。

**超球面上的测地线**：从$\boldsymbol{w} \in \mathcal{S}^{n-1}$出发，沿初始方向$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$的测地线为大圆弧：
$$
\gamma(t) = \cos(t\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(t\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
$$

**推导**：

**步骤1**：测地线满足二阶微分方程，设$\gamma(t) \in \mathcal{S}^{n-1}$，则$\|\gamma(t)\|_2 = 1$。

对时间求二阶导数：
$$
\frac{d^2}{dt^2}\langle\gamma(t), \gamma(t)\rangle = 2\langle\ddot{\gamma}(t), \gamma(t)\rangle + 2\|\dot{\gamma}(t)\|_2^2 = 0
$$

因此：
$$
\langle\ddot{\gamma}(t), \gamma(t)\rangle = -\|\dot{\gamma}(t)\|_2^2
$$

**步骤2**：测地线方程要求加速度垂直于流形，即$\ddot{\gamma}(t)$在切空间的投影为0：
$$
\mathcal{P}_{\gamma(t)}(\ddot{\gamma}(t)) = \ddot{\gamma}(t) - \langle\ddot{\gamma}(t), \gamma(t)\rangle\gamma(t) = \boldsymbol{0}
$$

因此：
$$
\ddot{\gamma}(t) = \langle\ddot{\gamma}(t), \gamma(t)\rangle\gamma(t) = -\|\dot{\gamma}(t)\|_2^2 \gamma(t)
$$

**步骤3**：这是一个二阶常系数微分方程。设初始条件$\gamma(0) = \boldsymbol{w}$，$\dot{\gamma}(0) = \boldsymbol{\xi}$，且$\langle\boldsymbol{\xi}, \boldsymbol{w}\rangle = 0$，$\|\boldsymbol{w}\|_2 = 1$。

令$s = \|\boldsymbol{\xi}\|_2$，$\boldsymbol{v} = \boldsymbol{\xi}/s$（单位切向量），则解为：
$$
\gamma(t) = \cos(st)\boldsymbol{w} + \sin(st)\boldsymbol{v}
$$

**验证**：
$$
\|\gamma(t)\|_2^2 = \cos^2(st) + \sin^2(st) = 1 \quad \checkmark
$$
$$
\dot{\gamma}(t) = -s\sin(st)\boldsymbol{w} + s\cos(st)\boldsymbol{v}
$$
$$
\|\dot{\gamma}(t)\|_2 = s \quad \text{(常数速度)}
$$

#### 4.2 指数映射（Exponential Map）

**定义 4.2（指数映射）**：指数映射$\text{Exp}_{\boldsymbol{w}}: T_{\boldsymbol{w}}\mathcal{S}^{n-1} \to \mathcal{S}^{n-1}$定义为：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \gamma_{\boldsymbol{\xi}}(1)
$$

其中$\gamma_{\boldsymbol{\xi}}(t)$是从$\boldsymbol{w}$出发、初速度为$\boldsymbol{\xi}$的测地线。

**超球面的指数映射公式**：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \cos(\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
$$

**性质**：
1. $\text{Exp}_{\boldsymbol{w}}(\boldsymbol{0}) = \boldsymbol{w}$
2. $\frac{d}{dt}\text{Exp}_{\boldsymbol{w}}(t\boldsymbol{\xi})\Big|_{t=0} = \boldsymbol{\xi}$
3. 指数映射是局部微分同胚（在小邻域内一一对应）

**小步长近似**：当$\|\boldsymbol{\xi}\|_2 \ll 1$时，利用泰勒展开：
$$
\cos(\|\boldsymbol{\xi}\|_2) \approx 1 - \frac{\|\boldsymbol{\xi}\|_2^2}{2} + O(\|\boldsymbol{\xi}\|_2^4)
$$
$$
\sin(\|\boldsymbol{\xi}\|_2) \approx \|\boldsymbol{\xi}\|_2 - \frac{\|\boldsymbol{\xi}\|_2^3}{6} + O(\|\boldsymbol{\xi}\|_2^5)
$$

因此：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) \approx \left(1 - \frac{\|\boldsymbol{\xi}\|_2^2}{2}\right)\boldsymbol{w} + \boldsymbol{\xi} + O(\|\boldsymbol{\xi}\|_2^3)
$$

归一化到单位长度：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) \approx \frac{\boldsymbol{w} + \boldsymbol{\xi}}{\|\boldsymbol{w} + \boldsymbol{\xi}\|_2} + O(\|\boldsymbol{\xi}\|_2^3)
$$

### 5. Retraction算子

#### 5.1 Retraction的定义

**定义 5.1（Retraction）**：从点$\boldsymbol{w} \in \mathcal{M}$处的retraction是映射$R_{\boldsymbol{w}}: T_{\boldsymbol{w}}\mathcal{M} \to \mathcal{M}$，满足：
1. $R_{\boldsymbol{w}}(\boldsymbol{0}) = \boldsymbol{w}$
2. $\frac{d}{dt}R_{\boldsymbol{w}}(t\boldsymbol{\xi})\Big|_{t=0} = \boldsymbol{\xi}$，$\forall \boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{M}$
3. 局部刚性：$R_{\boldsymbol{w}}$是局部微分同胚

**作用**：Retraction提供了一种从切空间"回到"流形的方法，是指数映射的计算友好替代。

#### 5.2 超球面的常用Retraction

**1. 归一化Retraction（最常用）**：
$$
R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \frac{\boldsymbol{w} + \boldsymbol{\xi}}{\|\boldsymbol{w} + \boldsymbol{\xi}\|_2}
$$

**验证条件**：
- 条件1：$R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{0}) = \boldsymbol{w}/\|\boldsymbol{w}\|_2 = \boldsymbol{w}$ ✓
- 条件2：设$\phi(t) = R_{\boldsymbol{w}}^{\text{norm}}(t\boldsymbol{\xi})$，则
$$
\phi(t) = \frac{\boldsymbol{w} + t\boldsymbol{\xi}}{\|\boldsymbol{w} + t\boldsymbol{\xi}\|_2}
$$
$$
\frac{d\phi}{dt}\Big|_{t=0} = \frac{\boldsymbol{\xi}\|\boldsymbol{w}\|_2 - \boldsymbol{w}\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle/\|\boldsymbol{w}\|_2}{\|\boldsymbol{w}\|_2^2} = \boldsymbol{\xi} - \langle\boldsymbol{w}, \boldsymbol{\xi}\rangle\boldsymbol{w} = \boldsymbol{\xi}
$$
（最后一步利用了$\langle\boldsymbol{w}, \boldsymbol{\xi}\rangle = 0$）✓

**计算成本**：
- 向量加法：$O(n)$
- 范数计算：$O(n)$
- 标量除法：$O(n)$
- **总成本**：$O(n)$

**2. 指数映射Retraction（精确但昂贵）**：
$$
R_{\boldsymbol{w}}^{\text{exp}}(\boldsymbol{\xi}) = \text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \cos(\|\boldsymbol{\xi}\|_2)\boldsymbol{w} + \sin(\|\boldsymbol{\xi}\|_2)\frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}
$$

**计算成本**：需要计算三角函数，约$O(n) + O(\log(1/\epsilon))$（$\epsilon$是精度）。

#### 5.3 归一化Retraction的误差分析

**定理 5.1**：归一化retraction与指数映射的误差为$O(\|\boldsymbol{\xi}\|_2^3)$：
$$
\left\|R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) - \text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi})\right\|_2 = O(\|\boldsymbol{\xi}\|_2^3)
$$

**证明**：

设$s = \|\boldsymbol{\xi}\|_2$，$\boldsymbol{v} = \boldsymbol{\xi}/s$（单位切向量）。

**指数映射**：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) = \cos(s)\boldsymbol{w} + \sin(s)\boldsymbol{v}
$$

**归一化retraction**：
$$
R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \frac{\boldsymbol{w} + s\boldsymbol{v}}{\|\boldsymbol{w} + s\boldsymbol{v}\|_2}
$$

计算分母：
$$
\|\boldsymbol{w} + s\boldsymbol{v}\|_2^2 = 1 + s^2 = 1 + s^2
$$
$$
\|\boldsymbol{w} + s\boldsymbol{v}\|_2 = \sqrt{1 + s^2} = 1 + \frac{s^2}{2} - \frac{s^4}{8} + O(s^6)
$$

因此：
$$
R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \frac{\boldsymbol{w} + s\boldsymbol{v}}{1 + s^2/2 + O(s^4)} = (\boldsymbol{w} + s\boldsymbol{v})(1 - s^2/2 + O(s^4))
$$
$$
= \boldsymbol{w} + s\boldsymbol{v} - \frac{s^2}{2}\boldsymbol{w} + O(s^3) = (1 - s^2/2)\boldsymbol{w} + s\boldsymbol{v} + O(s^3)
$$

对比泰勒展开：
$$
\cos(s) = 1 - \frac{s^2}{2} + \frac{s^4}{24} + O(s^6)
$$
$$
\sin(s) = s - \frac{s^3}{6} + O(s^5)
$$

误差为：
$$
\text{Exp}_{\boldsymbol{w}}(\boldsymbol{\xi}) - R_{\boldsymbol{w}}^{\text{norm}}(\boldsymbol{\xi}) = \left(\frac{s^4}{24}\boldsymbol{w} - \frac{s^3}{6}\boldsymbol{v}\right) + O(s^5) = O(s^3)
$$

**结论**：对于小步长优化（$s \approx \eta \ll 1$），归一化retraction是足够精确且高效的选择。

### 6. 超球面上的SGD更新规则

#### 6.1 完整的更新算法

**算法 6.1（超球面上的随机梯度下降）**：

**输入**：初始点$\boldsymbol{w}_0 \in \mathcal{S}^{n-1}$，学习率$\eta_t$，目标函数$f$

**For** $t = 0, 1, 2, \ldots$ **do**:

1. **计算随机梯度**：$\boldsymbol{g}_t = \nabla f(\boldsymbol{w}_t; \mathcal{B}_t)$（$\mathcal{B}_t$是小批量数据）

2. **投影到切空间**（计算黎曼梯度）：
   $$
   \boldsymbol{g}_t^{\text{Riem}} = \boldsymbol{g}_t - \langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle\boldsymbol{w}_t
   $$

3. **确定下降方向**：
   $$
   \boldsymbol{\xi}_t = -\eta_t \frac{\boldsymbol{g}_t^{\text{Riem}}}{\|\boldsymbol{g}_t^{\text{Riem}}\|_2}
   $$
   或者（不归一化版本）：
   $$
   \boldsymbol{\xi}_t = -\eta_t \boldsymbol{g}_t^{\text{Riem}}
   $$

4. **Retraction回流形**：
   $$
   \boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t + \boldsymbol{\xi}_t}{\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2}
   $$

**Output**：$\boldsymbol{w}_T$

#### 6.2 更新规则的简化形式

当使用非归一化的黎曼梯度时，更新可以写成：
$$
\boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta_t(\boldsymbol{g}_t - \langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle\boldsymbol{w}_t)}{\|\boldsymbol{w}_t - \eta_t(\boldsymbol{g}_t - \langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle\boldsymbol{w}_t)\|_2}
$$

进一步简化（假设$\|\boldsymbol{w}_t\|_2 = 1$）：
$$
\boldsymbol{w}_{t+1} = \frac{(1 + \eta_t\langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle)\boldsymbol{w}_t - \eta_t\boldsymbol{g}_t}{\|(1 + \eta_t\langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle)\boldsymbol{w}_t - \eta_t\boldsymbol{g}_t\|_2}
$$

#### 6.3 小步长近似

当$\eta_t \ll 1$时，$\boldsymbol{\xi}_t$很小，利用一阶近似：
$$
\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2 \approx \sqrt{1 + \eta_t^2\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2} \approx 1 + \frac{\eta_t^2\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2}{2}
$$

因此：
$$
\boldsymbol{w}_{t+1} \approx \left(1 - \frac{\eta_t^2\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2}{2}\right)(\boldsymbol{w}_t - \eta_t\boldsymbol{g}_t^{\text{Riem}}) + O(\eta_t^3)
$$

**验证约束保持**：
$$
\|\boldsymbol{w}_{t+1}\|_2^2 = \frac{\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2^2}{\|\boldsymbol{w}_t + \boldsymbol{\xi}_t\|_2^2} = 1 \quad \checkmark
$$

### 7. 收敛性分析

#### 7.1 下降引理（Descent Lemma）

**引理 7.1（黎曼流形上的下降引理）**：假设$f$是$L$-光滑的（黎曼Hessian有界），即：
$$
\|\nabla^2 f(\boldsymbol{w})\| \leq L, \quad \forall \boldsymbol{w} \in \mathcal{S}^{n-1}
$$

则对于任意$\boldsymbol{w}, \boldsymbol{w}' \in \mathcal{S}^{n-1}$，有：
$$
f(\boldsymbol{w}') \leq f(\boldsymbol{w}) + \langle\text{grad} f(\boldsymbol{w}), \boldsymbol{\xi}\rangle + \frac{L}{2}\|\boldsymbol{\xi}\|_2^2
$$

其中$\boldsymbol{w}' = R_{\boldsymbol{w}}(\boldsymbol{\xi})$，$\boldsymbol{\xi} \in T_{\boldsymbol{w}}\mathcal{S}^{n-1}$。

#### 7.2 单步下降量估计

**定理 7.1（单步下降）**：使用学习率$\eta_t \leq 1/L$，超球面SGD满足：
$$
\mathbb{E}[f(\boldsymbol{w}_{t+1})] \leq f(\boldsymbol{w}_t) - \frac{\eta_t}{2}\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \frac{L\eta_t^2}{2}\mathbb{E}[\|\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t)\|_2^2]
$$

**证明**：

应用下降引理，设$\boldsymbol{\xi}_t = -\eta_t\boldsymbol{g}_t^{\text{Riem}}$：
$$
f(\boldsymbol{w}_{t+1}) \leq f(\boldsymbol{w}_t) + \langle\text{grad} f(\boldsymbol{w}_t), -\eta_t\boldsymbol{g}_t^{\text{Riem}}\rangle + \frac{L\eta_t^2}{2}\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2
$$

第二项：
$$
\langle\text{grad} f(\boldsymbol{w}_t), -\eta_t\boldsymbol{g}_t^{\text{Riem}}\rangle = -\eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t)\rangle
$$

由于$\text{grad} f(\boldsymbol{w}_t) \in T_{\boldsymbol{w}_t}\mathcal{S}^{n-1}$，投影不改变：
$$
= -\eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\nabla f(\boldsymbol{w}_t))\rangle - \eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t))\rangle
$$
$$
= -\eta_t\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 - \eta_t\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t))\rangle
$$

取期望（$\mathbb{E}[\boldsymbol{g}_t | \boldsymbol{w}_t] = \nabla f(\boldsymbol{w}_t)$）：
$$
\mathbb{E}\langle\text{grad} f(\boldsymbol{w}_t), \mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t))\rangle = 0
$$

对第三项，注意$\|\mathcal{P}_{\boldsymbol{w}_t}(\boldsymbol{v})\|_2 \leq \|\boldsymbol{v}\|_2$（投影缩短长度），因此：
$$
\mathbb{E}[\|\boldsymbol{g}_t^{\text{Riem}}\|_2^2] \leq \mathbb{E}[\|\boldsymbol{g}_t\|_2^2] = \|\nabla f(\boldsymbol{w}_t)\|_2^2 + \mathbb{E}[\|\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t)\|_2^2]
$$

综合，当$\eta_t \leq 1/L$时：
$$
\mathbb{E}[f(\boldsymbol{w}_{t+1})] \leq f(\boldsymbol{w}_t) - \eta_t\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \frac{L\eta_t^2}{2}(\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \sigma^2)
$$
$$
\leq f(\boldsymbol{w}_t) - \frac{\eta_t}{2}\|\text{grad} f(\boldsymbol{w}_t)\|_2^2 + \frac{L\eta_t^2\sigma^2}{2}
$$

其中$\sigma^2 = \mathbb{E}[\|\boldsymbol{g}_t - \nabla f(\boldsymbol{w}_t)\|_2^2]$是梯度估计的方差。

#### 7.3 收敛速率

**定理 7.2（收敛速率）**：对于非凸目标函数，使用常学习率$\eta_t = \eta = \min\{1/L, \sqrt{T}/\sigma\}$，经过$T$次迭代：
$$
\min_{t=0,\ldots,T-1}\mathbb{E}[\|\text{grad} f(\boldsymbol{w}_t)\|_2^2] \leq \frac{2(f(\boldsymbol{w}_0) - f^*)}{\eta T} + L\eta\sigma^2 = O\left(\frac{1}{\sqrt{T}}\right)
$$

其中$f^* = \inf_{\boldsymbol{w} \in \mathcal{S}^{n-1}} f(\boldsymbol{w})$。

**证明**：累加单步下降不等式：
$$
\sum_{t=0}^{T-1}\mathbb{E}[\|\text{grad} f(\boldsymbol{w}_t)\|_2^2] \leq \frac{2}{\eta}(f(\boldsymbol{w}_0) - f^*) + L\eta T\sigma^2
$$

取$\eta = \sqrt{(f(\boldsymbol{w}_0) - f^*)/(LT\sigma^2)}$即得。

**与欧氏SGD的对比**：收敛率相同，均为$O(1/\sqrt{T})$，表明约束不影响渐近收敛速度。

### 8. 与欧氏空间SGD的对比

#### 8.1 更新规则对比

| 特性 | 欧氏SGD | 超球面SGD |
|------|---------|-----------|
| 参数空间 | $\mathbb{R}^n$ | $\mathcal{S}^{n-1}$ |
| 梯度 | $\nabla f(\boldsymbol{w})$ | $\text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}$ |
| 更新方向 | $-\nabla f(\boldsymbol{w})$ | $-\text{grad} f(\boldsymbol{w})$ |
| 更新规则 | $\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta_t\nabla f(\boldsymbol{w}_t)$ | $\boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta_t\text{grad} f(\boldsymbol{w}_t)}{\|\cdots\|_2}$ |
| 约束保持 | 无 | $\|\boldsymbol{w}_t\|_2 = 1$ 自动满足 |
| 计算复杂度 | $O(n)$ | $O(n)$ （额外一次内积和归一化） |

#### 8.2 几何对比

**欧氏SGD**：
- 沿着负梯度方向在平坦的欧氏空间中前进
- 步长$\eta_t$直接控制参数改变量的欧氏距离
- 可能偏离约束集（需要投影）

**超球面SGD**：
- 沿着球面上的测地线（大圆弧）前进
- 步长$\eta_t$控制在切空间中的移动量
- 自动保持在球面上（通过retraction）

#### 8.3 有效维度分析

**欧氏空间**：$n$个自由度

**超球面**：$n-1$个自由度（一个约束$\|\boldsymbol{w}\|_2 = 1$）

**影响**：
- 超球面SGD实际上在$(n-1)$维空间中优化
- 约束消除了"径向"自由度，只保留"切向"自由度
- 这可以看作是一种正则化，防止参数范数无限增长

#### 8.4 梯度范数对比

**引理 8.1**：超球面黎曼梯度范数不超过欧氏梯度范数：
$$
\|\text{grad} f(\boldsymbol{w})\|_2 \leq \|\nabla f(\boldsymbol{w})\|_2
$$

**证明**：
$$
\|\text{grad} f(\boldsymbol{w})\|_2^2 = \|\nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}\|_2^2
$$
$$
= \|\nabla f(\boldsymbol{w})\|_2^2 - 2\langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle^2 + \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle^2\|\boldsymbol{w}\|_2^2
$$
$$
= \|\nabla f(\boldsymbol{w})\|_2^2 - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle^2 \leq \|\nabla f(\boldsymbol{w})\|_2^2
$$

**几何意义**：投影到切空间会丢失法向分量，因此黎曼梯度更小或相等。

### 9. 在神经网络中的应用

#### 9.1 权重归一化（Weight Normalization）

**背景**：权重归一化是一种重参数化技巧，将权重分解为方向和尺度：
$$
\boldsymbol{w} = g\frac{\boldsymbol{v}}{\|\boldsymbol{v}\|_2}
$$

其中$g \in \mathbb{R}$是标量增益，$\boldsymbol{v}/\|\boldsymbol{v}\|_2 \in \mathcal{S}^{n-1}$是方向向量。

**优化问题**：给定损失$L(g, \boldsymbol{v})$，分别对$g$和$\boldsymbol{v}$优化：
$$
\frac{\partial L}{\partial g} = \frac{\partial L}{\partial \boldsymbol{w}}\frac{\partial \boldsymbol{w}}{\partial g} = \frac{\partial L}{\partial \boldsymbol{w}}\frac{\boldsymbol{v}}{\|\boldsymbol{v}\|_2}
$$

对$\boldsymbol{v}$的梯度：
$$
\frac{\partial L}{\partial \boldsymbol{v}} = \frac{\partial L}{\partial \boldsymbol{w}}\frac{\partial \boldsymbol{w}}{\partial \boldsymbol{v}} = g\frac{\partial L}{\partial \boldsymbol{w}}\left(\frac{\boldsymbol{I}}{\|\boldsymbol{v}\|_2} - \frac{\boldsymbol{v}\boldsymbol{v}^{\top}}{\|\boldsymbol{v}\|_2^3}\right)
$$

简化：
$$
\frac{\partial L}{\partial \boldsymbol{v}} = \frac{g}{\|\boldsymbol{v}\|_2}\left(\frac{\partial L}{\partial \boldsymbol{w}} - \frac{\langle\partial L/\partial \boldsymbol{w}, \boldsymbol{v}\rangle}{\|\boldsymbol{v}\|_2^2}\boldsymbol{v}\right)
$$

设$\boldsymbol{u} = \boldsymbol{v}/\|\boldsymbol{v}\|_2 \in \mathcal{S}^{n-1}$（单位化），$\partial L/\partial \boldsymbol{w} = \boldsymbol{g}_w$：
$$
\frac{\partial L}{\partial \boldsymbol{u}} = g\left(\boldsymbol{g}_w - \langle\boldsymbol{g}_w, \boldsymbol{u}\rangle\boldsymbol{u}\right) = g \cdot \text{grad}_{\boldsymbol{u}} L
$$

**更新规则**：
$$
g_{t+1} = g_t - \eta_g\frac{\partial L}{\partial g}
$$
$$
\boldsymbol{u}_{t+1} = \frac{\boldsymbol{u}_t - \eta_u \cdot \text{grad}_{\boldsymbol{u}} L}{\|\boldsymbol{u}_t - \eta_u \cdot \text{grad}_{\boldsymbol{u}} L\|_2}
$$

**好处**：
- 解耦方向和尺度的学习
- 稳定训练（方向变化被约束在球面上）
- 加速收敛（特别是对RNN和GAN）

#### 9.2 谱归一化（Spectral Normalization）

**应用**：GAN判别器的Lipschitz约束。

**目标**：将权重矩阵$\boldsymbol{W}$的最大奇异值限制为1：
$$
\boldsymbol{W}_{\text{SN}} = \frac{\boldsymbol{W}}{\sigma_1(\boldsymbol{W})}
$$

其中$\sigma_1(\boldsymbol{W}) = \max_{\|\boldsymbol{v}\|_2=1} \|\boldsymbol{W}\boldsymbol{v}\|_2$。

**计算**：使用幂迭代法，迭代求解：
$$
\boldsymbol{u}_{t+1} = \frac{\boldsymbol{W}\boldsymbol{v}_t}{\|\boldsymbol{W}\boldsymbol{v}_t\|_2}, \quad \boldsymbol{v}_{t+1} = \frac{\boldsymbol{W}^{\top}\boldsymbol{u}_{t+1}}{\|\boldsymbol{W}^{\top}\boldsymbol{u}_{t+1}\|_2}
$$

这正是在两个超球面$\mathcal{S}^{m-1}$和$\mathcal{S}^{n-1}$上交替优化！

**梯度计算**：
$$
\frac{\partial L}{\partial \boldsymbol{W}} = \frac{1}{\sigma_1}\left(\frac{\partial L}{\partial \boldsymbol{W}_{\text{SN}}} - \left\langle\frac{\partial L}{\partial \boldsymbol{W}_{\text{SN}}}, \boldsymbol{u}\boldsymbol{v}^{\top}\right\rangle\boldsymbol{u}\boldsymbol{v}^{\top}\right)
$$

#### 9.3 球面卷积神经网络

**应用**：处理球面数据（地球表面、全景图像、分子构象）。

**挑战**：标准卷积在平面网格定义，不适用于球面。

**解决方案**：
1. 使用球谐函数（spherical harmonics）作为特征
2. 在切空间定义局部卷积
3. 使用指数映射在不同切空间之间传输

**超球面SGD的角色**：优化定义在球面上的神经网络参数，自然保持几何结构。

#### 9.4 度量学习（Metric Learning）

**目标**：学习嵌入$\phi: \mathcal{X} \to \mathcal{S}^{n-1}$，使得相似样本在球面上接近。

**损失函数**：球面softmax
$$
L = -\log\frac{\exp(s\cos\theta_{y_i})}{\sum_{j=1}^C\exp(s\cos\theta_j)}
$$

其中$\theta_j = \arccos(\langle\boldsymbol{w}_j, \boldsymbol{x}_i\rangle)$是特征$\boldsymbol{x}_i$与类中心$\boldsymbol{w}_j$的夹角，$\boldsymbol{w}_j \in \mathcal{S}^{n-1}$。

**优化**：
- 特征归一化：$\boldsymbol{x}_i \leftarrow \boldsymbol{x}_i/\|\boldsymbol{x}_i\|_2$
- 类中心$\boldsymbol{w}_j$使用超球面SGD更新

**优势**：
- 几何解释清晰（角度距离）
- 避免范数膨胀
- 改善类别间可分性

### 10. 数值实验与实现细节

#### 10.1 算法伪代码

```python
def spherical_sgd(f, w0, T, eta, batch_size):
    """
    超球面上的随机梯度下降

    参数:
        f: 目标函数（返回损失和梯度）
        w0: 初始参数（应满足 ||w0||_2 = 1）
        T: 迭代次数
        eta: 学习率
        batch_size: 批量大小
    """
    w = w0 / np.linalg.norm(w0)  # 确保初始化在球面上

    for t in range(T):
        # 1. 采样小批量并计算梯度
        batch = sample_batch(batch_size)
        loss, grad = f(w, batch)

        # 2. 投影到切空间（黎曼梯度）
        grad_riem = grad - np.dot(grad, w) * w

        # 3. 切空间中的更新
        xi = -eta * grad_riem

        # 4. Retraction（归一化）
        w_new = w + xi
        w = w_new / np.linalg.norm(w_new)

    return w
```

#### 10.2 数值稳定性技巧

**1. 避免零除**：当$\|\boldsymbol{g}_t^{\text{Riem}}\|_2 \approx 0$时：
$$
\boldsymbol{w}_{t+1} = \boldsymbol{w}_t \quad \text{(跳过更新)}
$$

**2. 梯度裁剪**：
$$
\boldsymbol{g}_t^{\text{Riem}} \leftarrow \min\left(1, \frac{\tau}{\|\boldsymbol{g}_t^{\text{Riem}}\|_2}\right)\boldsymbol{g}_t^{\text{Riem}}
$$

**3. 周期性重归一化**（防止数值漂移）：
$$
\boldsymbol{w}_t \leftarrow \frac{\boldsymbol{w}_t}{\|\boldsymbol{w}_t\|_2} \quad \text{每}K\text{步}
$$

#### 10.3 复杂度总结

**每步时间复杂度**：
- 梯度计算：$O(C_f)$（依赖于$f$）
- 内积$\langle\boldsymbol{g}_t, \boldsymbol{w}_t\rangle$：$O(n)$
- 投影：$O(n)$
- 归一化：$O(n)$（计算范数+标量除法）
- **总计**：$O(C_f + n)$

**空间复杂度**：$O(n)$（存储$\boldsymbol{w}_t, \boldsymbol{g}_t$）

**与欧氏SGD的额外开销**：约2-3倍的向量操作（内积+投影+归一化），在现代硬件上可忽略。

### 11. 理论深化：曲率的影响

#### 11.1 黎曼曲率张量

超球面的曲率张量为：
$$
R(X, Y)Z = \langle Y, Z\rangle X - \langle X, Z\rangle Y
$$

**截面曲率**：恒为$K = 1$（正曲率）。

**影响**：
- 正曲率导致测地线相互汇聚
- 优化轨迹趋向于收敛到局部区域
- 与负曲率流形（如双曲空间）形成对比

#### 11.2 Jacobi场与测地线稳定性

**Jacobi方程**：
$$
\nabla_t^2 J + R(\dot{\gamma}, J)\dot{\gamma} = 0
$$

对于超球面：
$$
\nabla_t^2 J + J = 0
$$

**解**：$J(t) = A\cos(t) + B\sin(t)$（周期性振荡）

**几何意义**：从对跖点出发的测地线会在距离$\pi$处再次相交，导致指数映射在大步长时失去单射性。

**优化启示**：学习率不应过大，否则可能"越过"最优点。

### 12. 小结与展望

#### 12.1 核心要点回顾

1. **超球面是$(n-1)$维黎曼流形**，嵌入在$\mathbb{R}^n$中，约束为$\|\boldsymbol{w}\|_2 = 1$。

2. **切空间$T_{\boldsymbol{w}}\mathcal{S}^{n-1} = \{\boldsymbol{v} : \langle\boldsymbol{v}, \boldsymbol{w}\rangle = 0\}$**，由与$\boldsymbol{w}$正交的向量组成。

3. **黎曼梯度**是欧氏梯度在切空间的投影：
   $$
   \text{grad} f(\boldsymbol{w}) = \nabla f(\boldsymbol{w}) - \langle\nabla f(\boldsymbol{w}), \boldsymbol{w}\rangle\boldsymbol{w}
   $$

4. **最速下降方向**是负黎曼梯度方向：
   $$
   \boldsymbol{\xi}^* = -\frac{\text{grad} f(\boldsymbol{w})}{\|\text{grad} f(\boldsymbol{w})\|_2}
   $$

5. **Retraction（归一化）**将切空间更新映射回流形：
   $$
   \boldsymbol{w}_{t+1} = \frac{\boldsymbol{w}_t - \eta_t\text{grad} f(\boldsymbol{w}_t)}{\|\boldsymbol{w}_t - \eta_t\text{grad} f(\boldsymbol{w}_t)\|_2}
   $$

6. **收敛速率$O(1/\sqrt{T})$**，与无约束SGD相同。

7. **应用广泛**：权重归一化、谱归一化、度量学习、球面神经网络。

#### 12.2 与原文的联系

本推导详细展开了原文第5节"超球面上"的数学细节：
- **方程(9)**的几何意义：最速下降方向的优化问题
- **方程(10)**的推导：球面约束的一阶近似
- **方程(11-12)**的求解：拉格朗日乘数法
- **方程(13)**的归一化：精确保持约束

同时补充了原文未涉及的：
- 微分几何基础（切空间、法空间、投影）
- 黎曼梯度的严格定义
- 测地线与指数映射
- 收敛性理论分析
- 神经网络应用实例

#### 12.3 进一步阅读

- **黎曼优化**：Absil et al., "Optimization Algorithms on Matrix Manifolds" (2008)
- **流形学习**：Boumal, "An Introduction to Optimization on Smooth Manifolds" (2020)
- **神经网络应用**：Salimans & Kingma, "Weight Normalization" (2016)
- **谱归一化**：Miyato et al., "Spectral Normalization for GANs" (2018)

