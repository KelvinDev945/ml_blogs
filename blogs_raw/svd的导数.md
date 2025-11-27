---
title: SVD的导数
slug: svd的导数
date: 2025-04-26
tags: 详细推导, 微积分, 分析, 矩阵, SVD, 梯度
status: completed
---
# SVD的导数

**原文链接**: [https://spaces.ac.cn/archives/10878](https://spaces.ac.cn/archives/10878)

**发布日期**: 

---

SVD（Singular Value Decomposition，奇异值分解）是常见的矩阵分解算法，相信很多读者都已经对它有所了解，此前我们在[《低秩近似之路（二）：SVD》](/archives/10407)也专门介绍过它。然而，读者是否想到，SVD竟然还可以求导呢？笔者刚了解到这一结论时也颇感意外，因为直觉上“分解”往往都是不可导的。但事实是，SVD在一般情况下确实可导，这意味着理论上我们可以将SVD嵌入到模型中，并用基于梯度的优化器来端到端训练。

问题来了，既然SVD可导，那么它的导函数长什么样呢？接下来，我们将参考文献[《Differentiating the Singular Value Decomposition》](https://j-towns.github.io/papers/svd-derivative.pdf)，逐步推导SVD的求导公式。

## 推导基础 #

假设$\boldsymbol{W}$是满秩的$n\times n$矩阵，且全体奇异值两两不等，这是比较容易讨论的情形，后面我们也会讨论哪些条件可以放宽一点。接着，我们设$\boldsymbol{W}$的SVD为：  
\begin{equation}\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\end{equation}  
所谓SVD求导，实际上就是设法分别求出$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}$关于$\boldsymbol{W}$的梯度或微分。为此，我们先对上式两边求微分  
\begin{equation}d\boldsymbol{W} = (d\boldsymbol{U})\boldsymbol{\Sigma}\boldsymbol{V}^{\top} + \boldsymbol{U}(d\boldsymbol{\Sigma})\boldsymbol{V}^{\top} + \boldsymbol{U}\boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\end{equation}  
左乘$\boldsymbol{U}^{\top}$、右乘$\boldsymbol{V}$，并利用$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}, \boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}$得到  
\begin{equation}\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V} = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}\label{eq:core}\end{equation}  
这便是后面推导的基础。注意对恒等式$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}, \boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}$两端求微分，我们还可以得到  
\begin{equation}(d\boldsymbol{U})^{\top}\boldsymbol{U} + \boldsymbol{U}^{\top}(d\boldsymbol{U}) = \boldsymbol{0},\quad (d\boldsymbol{V})^{\top}\boldsymbol{V} + \boldsymbol{V}^{\top}(d\boldsymbol{V}) = \boldsymbol{0}\end{equation}  
这表明$\boldsymbol{U}^{\top}(d\boldsymbol{U})$和$(d\boldsymbol{V})^{\top}\boldsymbol{V}$都是反对称矩阵。

## 奇异值 #

反对称矩阵的特点是对角线元素都是零，而$\boldsymbol{\Sigma}$是对角阵，非对角线元素都是零，这启发我们可能需要分别处理处理对角线和非对角线元素。

首先，我们定义矩阵$\boldsymbol{I}$和$\bar{\boldsymbol{I}}$：$\boldsymbol{I}$就是单位阵，即对角线元素全是1，非对角线元素全是0；$\bar{\boldsymbol{I}}$则是单位阵的互补阵，即对角线元素全是0，而非对角线元素全是1。利用$\boldsymbol{I}$、$\bar{\boldsymbol{I}}$和Hadamard积$\otimes$，我们可以分别提取出式$\eqref{eq:core}$的对角线和非对角线部分：  
\begin{align}  
\boldsymbol{I}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) =&\, \boldsymbol{I}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}) = d\boldsymbol{\Sigma} \label{eq:value} \\\\[8pt]  
\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) =&\, \bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}) = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}\label{eq:vector}  
\end{align}  
现在我们先来看式$\eqref{eq:value}$，它可以等价地写成  
\begin{equation}d\sigma_i = \boldsymbol{u}_i^{\top}(d\boldsymbol{W})\boldsymbol{v}_i\label{eq:d-sigma}\end{equation}  
这就是第$i$个奇异值$\sigma_i$的微分，其中$\boldsymbol{u}_i,\boldsymbol{v}_i$分别是$\boldsymbol{U},\boldsymbol{V}$的第$i$列。[《从谱范数梯度到新式权重衰减的思考》](/archives/10648)中讨论的谱范数梯度，实际上就只是这里$i=1$时的特例。

## 奇异向量 #

然后我们再来看式$\eqref{eq:vector}$：  
\begin{equation}\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}\label{eq:vector-1}\end{equation}  
转置一下  
\begin{equation}\begin{aligned}  
\bar{\boldsymbol{I}}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}) =&\, \boldsymbol{\Sigma}(d\boldsymbol{U})^{\top}\boldsymbol{U} + \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} \\\\[6pt]  
=&\, -\boldsymbol{\Sigma}\boldsymbol{U}^{\top}(d\boldsymbol{U}) - (d\boldsymbol{V})^{\top}\boldsymbol{V}\boldsymbol{\Sigma}  
\end{aligned}\label{eq:vector-2}\end{equation}  
第二个等号利用了“$\boldsymbol{U}^{\top}(d\boldsymbol{U})$和$(d\boldsymbol{V})^{\top}\boldsymbol{V}$都是反对称矩阵”这一事实。式$\eqref{eq:vector-1}$和式$\eqref{eq:vector-2}$就是关于$d\boldsymbol{U},d\boldsymbol{V}$的线性方程组，我们要从中解出$d\boldsymbol{U},d\boldsymbol{V}$。

求解思路就是普通的消元法。首先，由$\eqref{eq:vector-1}\times\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\times\eqref{eq:vector-2}$得到  
\begin{equation}\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}) = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma}^2 - \boldsymbol{\Sigma}^2\boldsymbol{U}^{\top}(d\boldsymbol{U})\end{equation}  
这里利用了对角阵$\boldsymbol{\Sigma}$满足$\boldsymbol{\Sigma}(\bar{\boldsymbol{I}}\otimes \boldsymbol{M}) = \bar{\boldsymbol{I}}\otimes (\boldsymbol{\Sigma}\boldsymbol{M})$以及$(\bar{\boldsymbol{I}}\otimes \boldsymbol{M})\boldsymbol{\Sigma} = \bar{\boldsymbol{I}}\otimes (\boldsymbol{M}\boldsymbol{\Sigma})$的事实。我们知道，左（右）乘一个对角阵，等于矩阵的每一行（列）都乘以对角阵上相应的元素，所以如果我们定义矩阵$\boldsymbol{E}$，其中$\boldsymbol{E}_{i,j} = \sigma_j^2 - \sigma_i^2$，那么$\boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma}^2 - \boldsymbol{\Sigma}^2\boldsymbol{U}^{\top}(d\boldsymbol{U}) = \boldsymbol{E}\otimes (\boldsymbol{U}^{\top}(d\boldsymbol{U}))$，于是上式可以写成  
\begin{equation}\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}) = \boldsymbol{E}\otimes (\boldsymbol{U}^{\top}(d\boldsymbol{U}))\label{eq:dU-0}\end{equation}  
继而可以解得  
\begin{equation}d\boldsymbol{U} = \boldsymbol{U}(\boldsymbol{F}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}))\label{eq:dU}\end{equation}  
类似地，由$\boldsymbol{\Sigma}\times \eqref{eq:vector-1} + \eqref{eq:vector-2}\times \boldsymbol{\Sigma}$解得：  
\begin{equation}d\boldsymbol{V} = \boldsymbol{V}(\boldsymbol{F}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}))\label{eq:dV}\end{equation}  
式$\eqref{eq:dU},\eqref{eq:dV}$便是特征向量的微分。其中  
\begin{equation}\boldsymbol{F}_{i,j} = \left\\{\begin{aligned}  
&\, 1/(\sigma_j^2 - \sigma_i^2), &\, i\neq j \\\  
&\, 0, &\, i = j  
\end{aligned}\right.\end{equation}

## 梯度（一） #

微分有了，怎么把梯度求出来呢？这还真有点麻烦。倒不是技术上的麻烦，而是表示上的麻烦，比如$\boldsymbol{W},\boldsymbol{U}$是一个$n\times n$矩阵，那么$\boldsymbol{U}$关于$\boldsymbol{W}$的完整梯度就是一个$n\times n\times n\times n$的4阶张量，而高阶张量是大多数人包括笔者都不熟悉的内容。

为了绕开高阶张量的麻烦，我们有两个方案。首先，从编程角度看，我们根本没必要求出梯度的形式，而是根据微分的结果写出等价的前向形式，然后交给框架的自动求导就行。比如，从式$\eqref{eq:d-sigma}$我们可以断定$\sigma_i$的梯度等于$\newcommand{\sg}[1]{\color{skyblue}{#1}} \sg{\boldsymbol{u}_i}^{\top} \boldsymbol{W} \sg{\boldsymbol{v}_i}$的梯度，即  
\begin{equation}\nabla_{\boldsymbol{W}} \sigma_i = \nabla_{\boldsymbol{W}} (\sg{\boldsymbol{u}_i}^{\top} \boldsymbol{W} \sg{\boldsymbol{v}_i})\end{equation}  
这里将符号颜色改为$\sg{\blacksquare}$色代表stop_gradient算子，避免公式过于臃肿。刚好$\sigma_i$又等于$\boldsymbol{u}_i^{\top}\boldsymbol{W}\boldsymbol{v}_i$，所以我们只需要把代码中出现$\sigma_i$的地方，都替换成$\sg{\boldsymbol{u}_i}^{\top} \boldsymbol{W} \sg{\boldsymbol{v}_i}$，那么就可以自动获得正确的梯度。一般地，我们有  
\begin{equation}\nabla_{\boldsymbol{W}} \boldsymbol{\Sigma} = \nabla_{\boldsymbol{W}} (\boldsymbol{I}\otimes(\sg{\boldsymbol{U}}^{\top} \boldsymbol{W} \sg{\boldsymbol{V}}))\end{equation}  
即所有$\boldsymbol{\Sigma}$换成$\boldsymbol{I}\otimes(\sg{\boldsymbol{U}}^{\top} \boldsymbol{W} \sg{\boldsymbol{V}})$即可。

同理，从式$\eqref{eq:dU}$我们知道  
\begin{equation}\nabla_{\boldsymbol{W}}\boldsymbol{U} = \nabla_{\boldsymbol{W}}(\sg{\boldsymbol{U}}(\sg{\boldsymbol{F}}\otimes(\sg{\boldsymbol{U}}^{\top}\boldsymbol{W}\sg{\boldsymbol{V}\boldsymbol{\Sigma}} + \sg{\boldsymbol{\Sigma}\boldsymbol{V}}^{\top}\boldsymbol{W}^{\top}\sg{\boldsymbol{U}})))\end{equation}  
可以验证$\boldsymbol{U}(\boldsymbol{F}\otimes(\boldsymbol{U}^{\top}\boldsymbol{W}\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}\boldsymbol{W}^{\top}\boldsymbol{U}))$刚好是零矩阵，所以我们只需要将代码中所有出现$\boldsymbol{U}$的地方，都替换成  
\begin{equation}\boldsymbol{U} \quad \to \quad \sg{\boldsymbol{U}} + \sg{\boldsymbol{U}}(\sg{\boldsymbol{F}}\otimes(\sg{\boldsymbol{U}}^{\top}\boldsymbol{W}\sg{\boldsymbol{V}\boldsymbol{\Sigma}} + \sg{\boldsymbol{\Sigma}\boldsymbol{V}}^{\top}\boldsymbol{W}^{\top}\sg{\boldsymbol{U}}))\end{equation}  
那就能保持正确的前向结果，同时获得正确的梯度。基于同样的原理，$\boldsymbol{V}$的替换格式是  
\begin{equation}\boldsymbol{V} \quad \to \quad \sg{\boldsymbol{V}} + \sg{\boldsymbol{V}}(\sg{\boldsymbol{F}}\otimes(\sg{\boldsymbol{V}}^{\top}\boldsymbol{W}^{\top}\sg{\boldsymbol{U}\boldsymbol{\Sigma}} + \sg{\boldsymbol{\Sigma}\boldsymbol{U}}^{\top}\boldsymbol{W}\sg{\boldsymbol{V}}))\end{equation}

## 梯度（二） #

第二个方案是直接求出损失函数关于$\boldsymbol{W}$的梯度。具体来说，假设损失函数是$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}$的函数，记为$\mathcal{L}(\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V})$，我们直接求$\nabla_{\boldsymbol{W}}\mathcal{L}$，它是一个矩阵，可以用$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V},\nabla_{\boldsymbol{U}}\mathcal{L},\nabla_{\boldsymbol{\Sigma}}\mathcal{L},\nabla_{\boldsymbol{V}}\mathcal{L}$表示出来，这些量也都只是矩阵，所以不用涉及到高阶张量。

在上一节中，我们已经求出了具有相同梯度的$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}$的等效函数，除开被stop_gradient的部分外，这些等效函数关于$\boldsymbol{W}$都是线性的，所以此时问题本质上已经变成了线性复合函数的梯度。我们之前在[《低秩近似之路（一）：伪逆》](/archives/10366#%E7%9F%A9%E9%98%B5%E6%B1%82%E5%AF%BC)的“矩阵求导”一节便已经讨论过相关方法。具体来说，我们有  
\begin{align}  
\boldsymbol{X} = \boldsymbol{A}\boldsymbol{B}\boldsymbol{C} &\,\quad\Rightarrow\quad \nabla_{\boldsymbol{B}}f(\boldsymbol{X}) = \boldsymbol{A}^{\top}(\nabla_{\boldsymbol{X}}f(\boldsymbol{X}))\boldsymbol{C}^{\top} \\\\[8pt]  
\boldsymbol{X} = \boldsymbol{A}\boldsymbol{B}^{\top}\boldsymbol{C} &\,\quad\Rightarrow\quad \nabla_{\boldsymbol{B}}f(\boldsymbol{X}) = \boldsymbol{C}(\nabla_{\boldsymbol{X}}f(\boldsymbol{X}))^{\top}\boldsymbol{A} \\\\[8pt]  
\boldsymbol{X} = \boldsymbol{A}\otimes\boldsymbol{B} &\,\quad\Rightarrow\quad \nabla_{\boldsymbol{B}}f(\boldsymbol{X}) = \boldsymbol{A}\otimes \nabla_{\boldsymbol{X}}f(\boldsymbol{X})  
\end{align}  
利用这些基本公式，以及复合函数求导的链式法则，我们可以写出  
\begin{equation}\begin{aligned}  
\nabla_{\boldsymbol{W}}\mathcal{L} \quad = \qquad &\,\boldsymbol{U}(\boldsymbol{F}\otimes(\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{U}}\mathcal{L}) - (\nabla_{\boldsymbol{U}}\mathcal{L})^{\top}\boldsymbol{U}))\boldsymbol{\Sigma}\boldsymbol{V}^{\top} \\\\[6pt]  
\+ &\,\boldsymbol{U}(\boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L}))\boldsymbol{V}^{\top} \\\\[6pt]  
\+ &\,\boldsymbol{U}\boldsymbol{\Sigma}(\boldsymbol{F}\otimes(\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L}) - (\nabla_{\boldsymbol{V}}\mathcal{L})^{\top}\boldsymbol{V})))\boldsymbol{V}^{\top}  
\end{aligned}\end{equation}  
整个过程就是反复地利用基本公式和链式法则，以及$\boldsymbol{F}^{\top} = -\boldsymbol{F}$，原则上没有难度，就是需要谨慎地集中注意力，建议读者亲自动手完成这个推导过程，这是一道相当实用的矩阵求导练习题。最后引入两个记号  
\begin{equation}\newcommand{\sym}[1]{\color{red}{[}#1\color{red}{]_{sym}}} \newcommand{\skew}[1]{\color{red}{[}#1\color{red}{]_{skew}}} \sym{\boldsymbol{X}} = \frac{1}{2}(\boldsymbol{X} + \boldsymbol{X}^{\top}),\qquad \skew{\boldsymbol{X}} = \frac{1}{2}(\boldsymbol{X} - \boldsymbol{X}^{\top})\end{equation}  
我们可以将梯度结果简写成  
\begin{equation}\nabla_{\boldsymbol{W}}\mathcal{L} = \boldsymbol{U}\Big(2(\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{U}}\mathcal{L})})\boldsymbol{\Sigma} + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L}) + 2\boldsymbol{\Sigma}(\boldsymbol{F}\otimes\skew{\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L})})\Big)\boldsymbol{V}^{\top}  
\label{eq:w-grad-l}\end{equation}

## 梯度（三） #

现在可以牛刀小试一下，求$\boldsymbol{O}=\mathop{\text{msign}}(\boldsymbol{W})=\boldsymbol{U}\boldsymbol{V}^{\top}$的梯度，其中$\mathop{\text{msign}}$的概念我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)介绍Muon时已经讨论过。

根据$\mathop{\text{msign}}$的定义，我们有  
\begin{equation}\nabla_{\boldsymbol{U}}\mathcal{L} = (\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V},\qquad \nabla_{\boldsymbol{V}}\mathcal{L} = (\nabla_{\boldsymbol{O}}\mathcal{L})^{\top} \boldsymbol{U}\end{equation}  
代入式$\eqref{eq:w-grad-l}$得到  
\begin{equation}\begin{aligned}  
\nabla_{\boldsymbol{W}}\mathcal{L} =&\, 2\boldsymbol{U}\Big((\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}})\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(\boldsymbol{F}\otimes\skew{\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})^{\top}\boldsymbol{U}})\Big)\boldsymbol{V}^{\top} \\\\[6pt]  
=&\, 2\boldsymbol{U}\Big((\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}})\boldsymbol{\Sigma} - \boldsymbol{\Sigma}(\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}})\Big)\boldsymbol{V}^{\top} \\\\[6pt]  
=&\, 2\boldsymbol{U}\big(\boldsymbol{G}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}}\big)\boldsymbol{V}^{\top}  
\end{aligned}\end{equation}  
其中$\boldsymbol{G}_{i,j} = 1/(\sigma_i + \sigma_j)$。

## 梯度（四） #

最后让我们来考虑一个常用的特殊例子，当$\boldsymbol{W}$还是正定对称矩阵时，它的SVD具有$\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$的形式，即$\boldsymbol{U}=\boldsymbol{V}$。我们重复前面的推导，先对两边求微分  
\begin{equation}d\boldsymbol{W} = (d\boldsymbol{V})\boldsymbol{\Sigma}\boldsymbol{V}^{\top} + \boldsymbol{V}(d\boldsymbol{\Sigma})\boldsymbol{V}^{\top} + \boldsymbol{V}\boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\end{equation}  
然后左乘$\boldsymbol{V}^{\top}$、右乘$\boldsymbol{V}$：  
\begin{equation}\begin{aligned}  
\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V} =&\, \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V} \\\\[6pt]  
=&\, \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} - \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{V})  
\end{aligned}\end{equation}  
由此可见  
\begin{gather}  
d\boldsymbol{\Sigma} = \boldsymbol{I}\otimes(\boldsymbol{V}(d\boldsymbol{W})\boldsymbol{V}^{\top}) \\\\[8pt]  
\boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} - \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{V}) = \bar{\boldsymbol{I}}\otimes(\boldsymbol{V}(d\boldsymbol{W})\boldsymbol{V}^{\top})  
\end{gather}  
由第二式可以进一步解得  
\begin{equation}d\boldsymbol{V} = \boldsymbol{V}(\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V}))\end{equation}  
其中  
\begin{equation}\boldsymbol{K}_{i,j} = \left\\{\begin{aligned}  
&\, 1/(\sigma_i - \sigma_j), &\, i\neq j \\\  
&\, 0, &\, i = j  
\end{aligned}\right.\end{equation}  
根据这个结果，我们有  
\begin{equation}\nabla_{\boldsymbol{W}}\mathcal{L} = \boldsymbol{V}(\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L})) + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L}))\boldsymbol{V}^{\top} \end{equation}  
注意陷阱！上式其实是错误的，它是链式法则的结果，但链式法则只适用于无约束求导，而这里带有约束$\boldsymbol{W}=\boldsymbol{W}^{\top}$，正确的梯度应该还要包含对称化：  
\begin{equation}\begin{aligned}  
\nabla_{\boldsymbol{W}}\mathcal{L} =&\, \boldsymbol{V}\Big(\sym{\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L}))} + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L})\Big)\boldsymbol{V}^{\top} \\\\[6pt]  
=&\, \boldsymbol{V}\Big(\boldsymbol{K}^{\top}\otimes\skew{\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L})} + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L})\Big)\boldsymbol{V}^{\top}\end{aligned}\end{equation}  
另一个陷阱是直接将$\boldsymbol{U}=\boldsymbol{V}$代入式$\eqref{eq:w-grad-l}$来推导，这将会导致$\boldsymbol{K}$项翻倍，原因是式$\eqref{eq:w-grad-l}$区分了$\nabla_{\boldsymbol{U}}\mathcal{L}$和$\nabla_{\boldsymbol{V}}\mathcal{L}$，而在正定对称假设下，$\boldsymbol{U},\boldsymbol{V}$是相同的，对$\boldsymbol{V}$求梯度实际上无形中求了原本的$\nabla_{\boldsymbol{U}}\mathcal{L},\nabla_{\boldsymbol{V}}\mathcal{L}$之和，所以会导致重复计算。相关文献还可以参考[《Matrix Backpropagation for Deep Networks With Structured Layers》](https://openaccess.thecvf.com/content_iccv_2015/html/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.html)。

## 数值问题 #

可能有些读者疑问：我只能保证初始化矩阵的奇异值两两不等，怎么保证经过训练后的矩阵仍然满足这个条件呢？答案是梯度自然会帮我们保证。从式$\eqref{eq:dU}$和式$\eqref{eq:dV}$可以看出，它的梯度包含了$\boldsymbol{F}$，而由$\boldsymbol{F}_{i,j} = \frac{1}{\sigma_j^2 - \sigma_i^2}$可知，一旦两个奇异值接近，那么梯度将会非常大，所以优化器会自动把它们推开。

不过，这个特性也给实际训练带来了数值不稳定性，主要体现在$\sigma_i,\sigma_j$接近时的梯度爆炸。对此，论文[《Backpropagation-Friendly Eigendecomposition》](https://papers.cool/arxiv/1906.09023)提出用“幂迭代（Power Iteration）”替代精确的SVD。后来，论文[《Robust Differentiable SVD》](https://papers.cool/arxiv/2104.03821)证明了它理论上等于对$\boldsymbol{F}_{i,j} = -\frac{1}{（\sigma_i + \sigma_j)(\sigma_i - \sigma_j）}$中的$\frac{1}{\sigma_i - \sigma_j}$做泰勒近似（假设$\sigma_j < \sigma_i$）：  
\begin{equation}\frac{1}{\sigma_i - \sigma_j} = \frac{1}{\sigma_i}\frac{1}{1-(\sigma_j/\sigma_i)}\approx \frac{1}{\sigma_i}\left(1 + \left(\frac{\sigma_j}{\sigma_i}\right) + \left(\frac{\sigma_j}{\sigma_i}\right)^2 + \cdots + \left(\frac{\sigma_j}{\sigma_i}\right)^N \right)\end{equation}  
使用泰勒近似后，至少对于$\sigma_j \to \sigma_i$的情况不会出现梯度爆炸了。再后来，作者在[《Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?》](https://papers.cool/arxiv/2105.02498)将其推广到更一般的Padé近似。这一系列工作笔者并不是太熟悉，因此就不过多展开了。

只是笔者有一个疑问，如果只是单纯为了避免数值爆炸，似乎没必要上这些工具，直接给$\sigma_j/\sigma_i$加个截断不就好了？比如  
\begin{equation}\frac{1}{1-(\sigma_j/\sigma_i)}\approx \frac{1}{1-\min(\sigma_j/\sigma_i, 0.99)}\end{equation}  
这样不就简单避免了它趋于无穷？还是笔者有什么认识不到位的地方？如果读者了解相关背景，敬请在评论区指正，谢谢。

## 一般结果 #

到目前为止，上面出现的所有矩阵都是$n\times n$矩阵，因为这是在文章开头的假设“$\boldsymbol{W}$是满秩的$n\times n$矩阵，且全体奇异值两两不等”下进行的推导。这一节我们来讨论该条件可以放宽到什么程度。

简单来说，SVD可导的条件是“全体非零奇异值两两不等”，换言之，方阵可以去掉，满秩也可以去掉，但是非零奇异值仍须互不相同。因为一旦有相等的奇异值，那么SVD就不唯一，这从根本上破坏了可导性。当然，如果只要部分导数，那我们还可以放宽条件，比如只想要谱范数即$\sigma_1$的导数，那么只需要$\sigma_1 > \sigma_2$。

那么，放宽条件后，微分结果怎样变化呢？我们不妨一般地设$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，秩为$r$，SVD为  
\begin{equation}\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top},\quad\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m},\boldsymbol{V}\in\mathbb{R}^{m\times m}\end{equation}  
之所以允许零奇异值，是因为此时我们只关心$d\boldsymbol{U}_{[:, :r]}$和$d\boldsymbol{V}_{[:, :r]}$，在非零奇异值两两不等的假设下，它们是可以唯一确定的。我们从式$\eqref{eq:dU-0}$出发，直到式$\eqref{eq:dU-0}$的推导都是通用的，从式$\eqref{eq:dU-0}$也可以看出为什么要拒绝相同奇异值，因为$\sigma_i = \sigma_j$时$\sigma_j - \sigma_i$就无法求逆了。在式$\eqref{eq:dU-0}$中只保留跟$d\boldsymbol{U}_{[:, :r]}$相关的部分，我们得到  
\begin{equation}\bar{\boldsymbol{I}}_{[:, :r]}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma}_{[:, :r]} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}) = \boldsymbol{E}_{[:, :r]}\otimes (\boldsymbol{U}^{\top}(d\boldsymbol{U}_{[:, :r]}))\end{equation}  
这里$\boldsymbol{E}$的定义依然是$\boldsymbol{E}_{i,j} = \sigma_j^2 - \sigma_i^2$，但$\boldsymbol{E}_{[:, :r]}$正好排除了所有0，所以可以顺利求逆，结果是  
\begin{equation}d\boldsymbol{U}_{[:, :r]} = \boldsymbol{U}(\boldsymbol{F}_{[:, :r]}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma}_{[:, :r]} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}))\end{equation}  
进一步地，做如下分块  
\begin{equation}\boldsymbol{U} = \begin{pmatrix}\boldsymbol{U}_{[:,:r]} & \boldsymbol{U}_{[:,r:]}\end{pmatrix},\quad \boldsymbol{F}_{[:, :r]} = \begin{pmatrix} \boldsymbol{F}_{[:r, :r]} \\\ \boldsymbol{F}_{[r:, :r]}\end{pmatrix}\end{equation}  
再加上$\boldsymbol{V}\boldsymbol{\Sigma}_{[:, :r]} = \boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}$，可以得到  
\begin{equation}\begin{aligned}  
d\boldsymbol{U}_{[:, :r]} =&\, \boldsymbol{U}_{[:, :r]}(\boldsymbol{F}_{[:r, :r]}\otimes(\boldsymbol{U}_{[:, :r]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]} + \boldsymbol{\Sigma}_{[:r, :r]}\boldsymbol{V}_{[:, :r]}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]})) \\\\[6pt]  
&\,\qquad + \boldsymbol{U}_{[:, r:]}(\boldsymbol{F}_{[r:, :r]}\otimes(\boldsymbol{U}_{[:, r:]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}))  
\end{aligned}\label{eq:dU-r}\end{equation}  
根据假设，当$i > r$时$\sigma_i=0$，所以$\boldsymbol{F}_{[r:, :r]}$的每一行都是$(\sigma_1^{-2},\sigma_2^{-2},\cdots,\sigma_r^{-2})$，于是$\boldsymbol{F}_{[r:, :r]}\otimes$等于右乘$\boldsymbol{\Sigma}_{[:r, :r]}^{-2}$，因此  
\begin{equation}\boldsymbol{F}_{[r:, :r]}\otimes(\boldsymbol{U}_{[:, r:]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}) = \boldsymbol{U}_{[:, r:]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}^{-1}\end{equation}  
最后利用$\boldsymbol{U}\boldsymbol{U}^{\top} = \boldsymbol{I}$和$\boldsymbol{U} = (\boldsymbol{U}_{[:,:r]},\boldsymbol{U}_{[:,r:]})$，得到$\boldsymbol{U}_{[:, r:]}\boldsymbol{U}_{[:, r:]}^{\top} = \boldsymbol{I} - \boldsymbol{U}_{[:, :r]}\boldsymbol{U}_{[:, :r]}^{\top}$，代入式$\eqref{eq:dU-r}$得  
\begin{equation}\begin{aligned}  
d\boldsymbol{U}_{[:, :r]} =&\, \boldsymbol{U}_{[:, :r]}(\boldsymbol{F}_{[:r, :r]}\otimes(\boldsymbol{U}_{[:, :r]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]} + \boldsymbol{\Sigma}_{[:r, :r]}\boldsymbol{V}_{[:, :r]}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]})) \\\\[6pt]  
&\,\qquad \color{orange}{+ (\boldsymbol{I} - \boldsymbol{U}_{[:, :r]}\boldsymbol{U}_{[:, :r]}^{\top})(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}^{-1}}  
\end{aligned}\end{equation}  
这就将$d\boldsymbol{U}_{[:, r:]}$表示成了$\boldsymbol{U}_{[:, :r]},\boldsymbol{\Sigma}_{[:r, :r]},\boldsymbol{V}_{[:, :r]}$的函数，在非零奇异值两两不等的假设下，三者是唯一确定的，所以$d\boldsymbol{U}_{[:, :r]}$是唯一确定的，相比式$\eqref{eq:dU}$多出了橙色这一项。类似地  
\begin{equation}\begin{aligned}  
d\boldsymbol{V}_{[:, :r]} =&\, \boldsymbol{V}_{[:, :r]}(\boldsymbol{F}_{[:r, :r]}\otimes(\boldsymbol{V}_{[:, :r]}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]} + \boldsymbol{\Sigma}_{[:r, :r]}\boldsymbol{U}_{[:, :r]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]})) \\\\[6pt]  
&\,\qquad \color{orange}{+ (\boldsymbol{I} - \boldsymbol{V}_{[:, :r]}\boldsymbol{V}_{[:, :r]}^{\top})(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}^{-1}}  
\end{aligned}\end{equation}

## 文章小结 #

本文较为详细地推导了SVD的求导公式。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10878>_

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

苏剑林. (Apr. 26, 2025). 《SVD的导数 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10878>

@online{kexuefm-10878,  
title={SVD的导数},  
author={苏剑林},  
year={2025},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/10878}},  
} 


---

## 公式推导与注释

本节将对SVD导数进行极详细的数学推导，从多个角度深入理解这一重要结果。我们将涵盖微积分、矩阵分析、梯度计算等多个视角，并讨论退化情况、扰动理论、数值稳定性等关键问题。

### 一、SVD分解的微分理论基础

#### 1.1 矩阵微分的基本概念

对于矩阵函数$\boldsymbol{W}(t)$，其微分定义为：
$$d\boldsymbol{W} = \lim_{\epsilon\to 0}\frac{\boldsymbol{W}(t+\epsilon) - \boldsymbol{W}(t)}{\epsilon}dt$$

这个定义可以推广到矩阵值函数。关键性质包括：

**性质1（乘积法则）**：对于矩阵乘积$\boldsymbol{W} = \boldsymbol{A}\boldsymbol{B}\boldsymbol{C}$，有：
$$d\boldsymbol{W} = (d\boldsymbol{A})\boldsymbol{B}\boldsymbol{C} + \boldsymbol{A}(d\boldsymbol{B})\boldsymbol{C} + \boldsymbol{A}\boldsymbol{B}(d\boldsymbol{C})$$

**性质2（转置法则）**：$(d\boldsymbol{A})^{\top} = d(\boldsymbol{A}^{\top})$

**性质3（正交约束）**：若$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}$，对两边求微分：
$$\boldsymbol{0} = d(\boldsymbol{U}^{\top}\boldsymbol{U}) = (d\boldsymbol{U})^{\top}\boldsymbol{U} + \boldsymbol{U}^{\top}(d\boldsymbol{U})$$

这表明$\boldsymbol{U}^{\top}(d\boldsymbol{U})$是反对称矩阵（skew-symmetric），即：
$$[\boldsymbol{U}^{\top}(d\boldsymbol{U})]_{ij} = -[\boldsymbol{U}^{\top}(d\boldsymbol{U})]_{ji}$$

特别地，对角线元素必为零：$[\boldsymbol{U}^{\top}(d\boldsymbol{U})]_{ii} = 0$。

#### 1.2 从SVD分解到微分方程

给定SVD分解$\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，对两边求微分：
$$d\boldsymbol{W} = (d\boldsymbol{U})\boldsymbol{\Sigma}\boldsymbol{V}^{\top} + \boldsymbol{U}(d\boldsymbol{\Sigma})\boldsymbol{V}^{\top} + \boldsymbol{U}\boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}$$

**关键变换**：左乘$\boldsymbol{U}^{\top}$，右乘$\boldsymbol{V}$，利用正交性$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}$和$\boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}$：
$$\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V} = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}$$

记$\boldsymbol{\Omega}_U = \boldsymbol{U}^{\top}(d\boldsymbol{U})$和$\boldsymbol{\Omega}_V = (d\boldsymbol{V})^{\top}\boldsymbol{V}$，它们都是反对称矩阵。上式变为：
$$\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V} = \boldsymbol{\Omega}_U\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{\Omega}_V$$

这是关于$\boldsymbol{\Omega}_U$、$d\boldsymbol{\Sigma}$、$\boldsymbol{\Omega}_V$的Sylvester型方程。

#### 1.3 对角与非对角分离技术

利用Hadamard积（element-wise product）$\otimes$，我们可以分离方程的对角和非对角部分。

定义：
- $\boldsymbol{I}$：单位矩阵（对角元素为1，非对角元素为0）
- $\bar{\boldsymbol{I}}$：单位矩阵的补（对角元素为0，非对角元素为1）

显然$\boldsymbol{I} + \bar{\boldsymbol{I}} = \boldsymbol{J}$（全1矩阵）。

**对角部分**：由于$\boldsymbol{\Omega}_U$和$\boldsymbol{\Omega}_V$的对角元素为零，对角阵$\boldsymbol{\Sigma}$的非对角元素为零：
$$\boldsymbol{I}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) = \boldsymbol{I}\otimes(d\boldsymbol{\Sigma}) = d\boldsymbol{\Sigma}$$

写成分量形式：
$$d\sigma_i = [\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ii} = \boldsymbol{u}_i^{\top}(d\boldsymbol{W})\boldsymbol{v}_i$$

这就是**奇异值的微分公式**。

**非对角部分**：
$$\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) = \bar{\boldsymbol{I}}\otimes(\boldsymbol{\Omega}_U\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{\Omega}_V)$$

展开为分量形式（$i\neq j$）：
$$[\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij} = [\boldsymbol{\Omega}_U]_{ij}\sigma_j + \sigma_i[\boldsymbol{\Omega}_V]_{ij}$$

### 二、奇异值梯度的深入分析

#### 2.1 奇异值梯度的几何意义

从$d\sigma_i = \boldsymbol{u}_i^{\top}(d\boldsymbol{W})\boldsymbol{v}_i$可以立即得出：
$$\frac{\partial \sigma_i}{\partial W_{jk}} = u_{ji}v_{ki}$$

以矩阵形式：
$$\nabla_{\boldsymbol{W}}\sigma_i = \boldsymbol{u}_i\boldsymbol{v}_i^{\top}$$

**几何解释**：这是一个秩1矩阵，其方向由左右奇异向量张成。物理意义是：沿着$\boldsymbol{u}_i\boldsymbol{v}_i^{\top}$方向扰动矩阵$\boldsymbol{W}$，会最大程度地增加第$i$个奇异值。

#### 2.2 谱范数的梯度（最大奇异值）

谱范数定义为$\|\boldsymbol{W}\|_2 = \sigma_1 = \max_i \sigma_i$。其梯度为：
$$\nabla_{\boldsymbol{W}}\|\boldsymbol{W}\|_2 = \boldsymbol{u}_1\boldsymbol{v}_1^{\top}$$

这在深度学习中非常重要，特别是在**谱归一化（Spectral Normalization）**中。

**应用：谱归一化层**

在GAN等模型中，为了Lipschitz约束，需要对权重矩阵进行谱归一化：
$$\bar{\boldsymbol{W}} = \frac{\boldsymbol{W}}{\|\boldsymbol{W}\|_2}$$

其梯度为（使用商法则）：
$$\nabla_{\boldsymbol{W}}\mathcal{L}(\bar{\boldsymbol{W}}) = \frac{1}{\sigma_1}(\nabla_{\bar{\boldsymbol{W}}}\mathcal{L}) - \frac{1}{\sigma_1^2}(\nabla_{\bar{\boldsymbol{W}}}\mathcal{L}:\boldsymbol{W})\boldsymbol{u}_1\boldsymbol{v}_1^{\top}$$

其中$\boldsymbol{A}:\boldsymbol{B} = \text{tr}(\boldsymbol{A}^{\top}\boldsymbol{B})$是Frobenius内积。

#### 2.3 核范数的梯度（所有奇异值之和）

核范数（nuclear norm）定义为$\|\boldsymbol{W}\|_* = \sum_i \sigma_i$。其梯度为：
$$\nabla_{\boldsymbol{W}}\|\boldsymbol{W}\|_* = \sum_i \boldsymbol{u}_i\boldsymbol{v}_i^{\top} = \boldsymbol{U}\boldsymbol{V}^{\top}$$

这恰好是矩阵的"符号函数"$\text{msign}(\boldsymbol{W})$，在低秩优化中扮演重要角色。

### 三、奇异向量梯度的求解

#### 3.1 Sylvester方程的求解策略

回到非对角方程（$i\neq j$）：
$$[\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij} = [\boldsymbol{\Omega}_U]_{ij}\sigma_j + \sigma_i[\boldsymbol{\Omega}_V]_{ij}$$

**目标**：求解$\boldsymbol{\Omega}_U$和$\boldsymbol{\Omega}_V$。

**方法1：直接消元**

利用转置关系。对原方程转置：
$$[\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}]_{ij} = [\boldsymbol{\Omega}_V^{\top}]_{ij}\sigma_j + \sigma_i[\boldsymbol{\Omega}_U^{\top}]_{ij}$$

由反对称性$\boldsymbol{\Omega}_U^{\top} = -\boldsymbol{\Omega}_U$和$\boldsymbol{\Omega}_V^{\top} = -\boldsymbol{\Omega}_V$：
$$[\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}]_{ij} = -[\boldsymbol{\Omega}_V]_{ij}\sigma_j - \sigma_i[\boldsymbol{\Omega}_U]_{ij}$$

现在我们有两个方程消除一个未知数。

**消去$\boldsymbol{\Omega}_V$**：原方程乘以$\sigma_i$加上转置方程乘以$\sigma_j$：
$$\sigma_i[\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij} + \sigma_j[\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}]_{ij} = [\boldsymbol{\Omega}_U]_{ij}(\sigma_i\sigma_j - \sigma_i\sigma_j) + \sigma_i^2[\boldsymbol{\Omega}_V]_{ij} - \sigma_j^2[\boldsymbol{\Omega}_V]_{ij}$$

简化为：
$$\sigma_i[\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij} + \sigma_j[\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}]_{ij} = (\sigma_i^2 - \sigma_j^2)[\boldsymbol{\Omega}_V]_{ij}$$

因此：
$$[\boldsymbol{\Omega}_V]_{ij} = \frac{\sigma_i[\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij} + \sigma_j[\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}]_{ij}}{\sigma_i^2 - \sigma_j^2}$$

以矩阵形式（$i\neq j$）：
$$\boldsymbol{\Omega}_V = \boldsymbol{F}\otimes(\boldsymbol{\Sigma}\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V} + \boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}\boldsymbol{\Sigma})$$

其中$\boldsymbol{F}_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2}$（$i\neq j$），$\boldsymbol{F}_{ii} = 0$。

**恢复$d\boldsymbol{V}$**：由$\boldsymbol{\Omega}_V = (d\boldsymbol{V})^{\top}\boldsymbol{V}$，两边右乘$\boldsymbol{V}^{\top}$：
$$(d\boldsymbol{V})^{\top} = \boldsymbol{\Omega}_V\boldsymbol{V}^{\top}$$

转置得：
$$d\boldsymbol{V} = \boldsymbol{V}\boldsymbol{\Omega}_V^{\top}$$

类似地可得$d\boldsymbol{U} = \boldsymbol{U}\boldsymbol{\Omega}_U^{\top}$。

#### 3.2 矩阵乘法的简化表示

注意到：
$$\boldsymbol{\Omega}_U\boldsymbol{\Sigma} - \boldsymbol{\Sigma}\boldsymbol{\Omega}_U = \boldsymbol{E}\otimes\boldsymbol{\Omega}_U$$

其中$\boldsymbol{E}_{ij} = \sigma_j - \sigma_i$。这是因为：
$$[\boldsymbol{\Omega}_U\boldsymbol{\Sigma}]_{ij} = [\boldsymbol{\Omega}_U]_{ij}\sigma_j$$
$$[\boldsymbol{\Sigma}\boldsymbol{\Omega}_U]_{ij} = \sigma_i[\boldsymbol{\Omega}_U]_{ij}$$
$$[\boldsymbol{\Omega}_U\boldsymbol{\Sigma} - \boldsymbol{\Sigma}\boldsymbol{\Omega}_U]_{ij} = (\sigma_j - \sigma_i)[\boldsymbol{\Omega}_U]_{ij}$$

类似地：
$$\boldsymbol{\Omega}_U\boldsymbol{\Sigma}^2 - \boldsymbol{\Sigma}^2\boldsymbol{\Omega}_U = (\sigma_j^2 - \sigma_i^2)\boldsymbol{\Omega}_U = \boldsymbol{E}_2\otimes\boldsymbol{\Omega}_U$$

其中$\boldsymbol{E}_2 = \boldsymbol{E}\odot\boldsymbol{E}$（这里$\odot$表示element-wise乘法，实际上就是平方）。

### 四、退化情况：重复奇异值的处理

#### 4.1 重复奇异值的不可导性

**定理**：若$\boldsymbol{W}$存在重复奇异值（即$\sigma_i = \sigma_j$对某些$i\neq j$），则SVD分解不唯一，导数不存在。

**证明**：假设$\sigma_i = \sigma_j$，对应的奇异向量为$\boldsymbol{u}_i, \boldsymbol{u}_j$和$\boldsymbol{v}_i, \boldsymbol{v}_j$。那么对任意正交矩阵$\boldsymbol{Q}\in\mathbb{R}^{2\times 2}$：
$$\boldsymbol{W} = \cdots + \sigma_i\boldsymbol{u}_i\boldsymbol{v}_i^{\top} + \sigma_j\boldsymbol{u}_j\boldsymbol{v}_j^{\top} + \cdots$$

可以替换为：
$$\boldsymbol{W} = \cdots + \sigma_i(\boldsymbol{u}_i, \boldsymbol{u}_j)\boldsymbol{Q}\begin{pmatrix}\boldsymbol{v}_i^{\top} \\ \boldsymbol{v}_j^{\top}\end{pmatrix} + \cdots$$

这给出无穷多个等价的SVD分解。由于分解不唯一，导数无法定义。

从公式角度，$\boldsymbol{F}_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2}$在$\sigma_i = \sigma_j$时出现$0/0$型不定式，无法计算。

#### 4.2 扰动视角：Weyl不等式

**Weyl扰动定理**：设$\boldsymbol{W}$和$\boldsymbol{W} + \boldsymbol{E}$的奇异值分别为$\sigma_1\geq\cdots\geq\sigma_n$和$\tilde{\sigma}_1\geq\cdots\geq\tilde{\sigma}_n$，则：
$$|\sigma_i - \tilde{\sigma}_i| \leq \|\boldsymbol{E}\|_2$$

更精确的Weyl不等式：
$$|\sigma_i(\boldsymbol{W}) - \sigma_i(\boldsymbol{W}+\boldsymbol{E})| \leq \sigma_1(\boldsymbol{E})$$

**推论**：若$\sigma_i > \sigma_{i+1}$且间隙$\delta = \sigma_i - \sigma_{i+1} > 0$，那么当$\|\boldsymbol{E}\|_2 < \delta/2$时，扰动后的第$i$个奇异值仍然与其他奇异值分离，保持良定性。

#### 4.3 部分导数的存在性

即使存在重复奇异值，某些部分导数仍然存在：

**命题**：若仅要求$\nabla_{\boldsymbol{W}}\sigma_1$（最大奇异值的梯度），则只需$\sigma_1 > \sigma_2$，而不需要所有奇异值两两不等。

**证明**：$d\sigma_1 = \boldsymbol{u}_1^{\top}(d\boldsymbol{W})\boldsymbol{v}_1$的推导只依赖于对角元素，不涉及$\boldsymbol{F}_{ij}$的计算。即使$\sigma_2 = \sigma_3$，也不影响$\sigma_1$的梯度。

类似地，前$k$个奇异值$\sigma_1,\ldots,\sigma_k$的梯度存在，只需这$k$个奇异值互异，而不管$\sigma_{k+1},\ldots,\sigma_n$是否重复。

### 五、隐函数定理的应用

#### 5.1 SVD作为隐函数

将SVD视为隐函数关系：
$$\boldsymbol{G}(\boldsymbol{W}, \boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V}) = \boldsymbol{W} - \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \boldsymbol{0}$$

加上正交约束：
$$\boldsymbol{H}_U(\boldsymbol{U}) = \boldsymbol{U}^{\top}\boldsymbol{U} - \boldsymbol{I} = \boldsymbol{0}$$
$$\boldsymbol{H}_V(\boldsymbol{V}) = \boldsymbol{V}^{\top}\boldsymbol{V} - \boldsymbol{I} = \boldsymbol{0}$$

**隐函数定理**：若Jacobian矩阵$\frac{\partial(\boldsymbol{G}, \boldsymbol{H}_U, \boldsymbol{H}_V)}{\partial(\boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V})}$可逆，则存在局部微分映射：
$$\boldsymbol{U} = \boldsymbol{U}(\boldsymbol{W}),\quad \boldsymbol{\Sigma} = \boldsymbol{\Sigma}(\boldsymbol{W}),\quad \boldsymbol{V} = \boldsymbol{V}(\boldsymbol{W})$$

#### 5.2 可逆性条件

Jacobian可逆等价于奇异值两两不等。这可以从Sylvester方程的可解性看出：

考虑线性方程$\boldsymbol{A}\boldsymbol{X} - \boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}$，其解存在且唯一当且仅当$\boldsymbol{A}$和$\boldsymbol{B}$无公共特征值。

在我们的情况下，$\boldsymbol{A} = \text{diag}(\sigma_1^2,\ldots,\sigma_n^2)$，$\boldsymbol{B} = \text{diag}(\sigma_1^2,\ldots,\sigma_n^2)$，公共特征值意味着$\sigma_i^2 = \sigma_j^2$，即$\sigma_i = \sigma_j$（因奇异值非负）。

因此，**奇异值两两不等**是SVD可微的充要条件。

### 六、变分法推导

#### 6.1 优化视角

从优化角度理解SVD：最大奇异值可以写成优化问题：
$$\sigma_1 = \max_{\|\boldsymbol{u}\|=1, \|\boldsymbol{v}\|=1} \boldsymbol{u}^{\top}\boldsymbol{W}\boldsymbol{v}$$

**变分法**：对优化问题求导，设Lagrangian：
$$\mathcal{L}(\boldsymbol{u}, \boldsymbol{v}, \lambda, \mu) = \boldsymbol{u}^{\top}\boldsymbol{W}\boldsymbol{v} - \frac{\lambda}{2}(\boldsymbol{u}^{\top}\boldsymbol{u} - 1) - \frac{\mu}{2}(\boldsymbol{v}^{\top}\boldsymbol{v} - 1)$$

KKT条件：
$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{u}} = \boldsymbol{W}\boldsymbol{v} - \lambda\boldsymbol{u} = \boldsymbol{0} \quad\Rightarrow\quad \boldsymbol{W}\boldsymbol{v} = \lambda\boldsymbol{u}$$
$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{v}} = \boldsymbol{W}^{\top}\boldsymbol{u} - \mu\boldsymbol{v} = \boldsymbol{0} \quad\Rightarrow\quad \boldsymbol{W}^{\top}\boldsymbol{u} = \mu\boldsymbol{v}$$

由$\boldsymbol{W}\boldsymbol{v} = \lambda\boldsymbol{u}$得$\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{v} = \lambda\boldsymbol{W}^{\top}\boldsymbol{u} = \lambda\mu\boldsymbol{v}$，故$\lambda\mu = \sigma_1^2$，$\lambda = \mu = \sigma_1$。

#### 6.2 对$\boldsymbol{W}$的灵敏度分析

对KKT条件关于$\boldsymbol{W}$求微分：
$$d\boldsymbol{W}\cdot\boldsymbol{v} + \boldsymbol{W}\cdot d\boldsymbol{v} = (d\sigma_1)\boldsymbol{u} + \sigma_1(d\boldsymbol{u})$$
$$(d\boldsymbol{W})^{\top}\boldsymbol{u} + \boldsymbol{W}^{\top}(d\boldsymbol{u}) = (d\sigma_1)\boldsymbol{v} + \sigma_1(d\boldsymbol{v})$$

左乘第一式以$\boldsymbol{u}^{\top}$：
$$\boldsymbol{u}^{\top}(d\boldsymbol{W})\boldsymbol{v} + \boldsymbol{u}^{\top}\boldsymbol{W}(d\boldsymbol{v}) = d\sigma_1 + \sigma_1\boldsymbol{u}^{\top}(d\boldsymbol{u})$$

由正交约束$\boldsymbol{u}^{\top}(d\boldsymbol{u}) = 0$和$\boldsymbol{u}^{\top}\boldsymbol{W} = \sigma_1\boldsymbol{v}^{\top}$：
$$\boldsymbol{u}^{\top}(d\boldsymbol{W})\boldsymbol{v} + \sigma_1\boldsymbol{v}^{\top}(d\boldsymbol{v}) = d\sigma_1$$

又$\boldsymbol{v}^{\top}(d\boldsymbol{v}) = 0$，故：
$$d\sigma_1 = \boldsymbol{u}^{\top}(d\boldsymbol{W})\boldsymbol{v}$$

这再次验证了奇异值微分公式！

### 七、数值稳定性与实现算法

#### 7.1 病态问题：奇异值接近的情况

当$\sigma_i\approx\sigma_j$时，$\boldsymbol{F}_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2}$会变得极大，导致数值不稳定。

**问题根源**：$\boldsymbol{F}_{ij}$的大小决定了梯度的magnitude。定义条件数：
$$\kappa_{ij} = \frac{\max(\sigma_i, \sigma_j)}{\|\sigma_i - \sigma_j\|}$$

当$\kappa_{ij}\to\infty$时，问题变为病态。

#### 7.2 Taylor近似方法

对于$\sigma_i > \sigma_j$，可以进行Taylor展开：
$$\frac{1}{\sigma_i^2 - \sigma_j^2} = \frac{1}{\sigma_i^2}\cdot\frac{1}{1 - (\sigma_j/\sigma_i)^2}$$

设$r = \sigma_j/\sigma_i < 1$，则：
$$\frac{1}{1 - r^2} = 1 + r^2 + r^4 + \cdots = \sum_{k=0}^{\infty}r^{2k}$$

截断至$N$阶：
$$\frac{1}{\sigma_i^2 - \sigma_j^2}\approx\frac{1}{\sigma_i^2}\sum_{k=0}^{N}\left(\frac{\sigma_j}{\sigma_i}\right)^{2k}$$

**优点**：当$r$接近1时，级数每项大小相近，不会出现单个巨大的项。

#### 7.3 Padé近似

Padé近似是有理函数逼近：
$$\frac{1}{1-r^2}\approx\frac{P_m(r^2)}{Q_n(r^2)}$$

其中$P_m$和$Q_n$是多项式。例如$(m, n) = (1, 1)$时：
$$\frac{1}{1-r^2}\approx\frac{1 + \alpha r^2}{1 + \beta r^2}$$

通过匹配Taylor系数确定$\alpha, \beta$。Padé近似在整个区间上比Taylor多项式更精确。

#### 7.4 截断与正则化

最简单的方法是截断$\boldsymbol{F}_{ij}$：
$$\tilde{\boldsymbol{F}}_{ij} = \begin{cases}
\boldsymbol{F}_{ij}, & |\sigma_i^2 - \sigma_j^2| > \epsilon \\
0, & \text{otherwise}
\end{cases}$$

或使用软截断（Huber型）：
$$\tilde{\boldsymbol{F}}_{ij} = \frac{\sigma_j^2 - \sigma_i^2}{(\sigma_j^2 - \sigma_i^2)^2 + \epsilon^2}$$

这在$\sigma_i\approx\sigma_j$时提供平滑的过渡。

#### 7.5 幂迭代方法

对于最大奇异值$\sigma_1$及其奇异向量$\boldsymbol{u}_1, \boldsymbol{v}_1$，可以用幂迭代避免完整SVD：

**算法**：
1. 初始化：随机$\boldsymbol{v}^{(0)}$，归一化
2. 迭代：
   $$\boldsymbol{u}^{(k)} = \frac{\boldsymbol{W}\boldsymbol{v}^{(k-1)}}{\|\boldsymbol{W}\boldsymbol{v}^{(k-1)}\|}$$
   $$\boldsymbol{v}^{(k)} = \frac{\boldsymbol{W}^{\top}\boldsymbol{u}^{(k)}}{\|\boldsymbol{W}^{\top}\boldsymbol{u}^{(k)}\|}$$
3. 收敛后：$\sigma_1 = \|\boldsymbol{W}\boldsymbol{v}^{(\infty)}\|$

**梯度计算**：将幂迭代嵌入计算图，自动微分自然处理梯度传播。这避免了显式计算$\boldsymbol{F}_{ij}$。

**优点**：
- 只计算需要的奇异值/向量
- 数值稳定（避免近似奇异值的除法）
- 可微且可嵌入端到端训练

### 八、深度学习中的应用

#### 8.1 谱归一化（Spectral Normalization）

在GAN中，判别器需要满足Lipschitz连续性以稳定训练。对权重矩阵$\boldsymbol{W}$：
$$\bar{\boldsymbol{W}} = \frac{\boldsymbol{W}}{\sigma_1(\boldsymbol{W})}$$

则$\|\bar{\boldsymbol{W}}\|_2 = 1$，保证$\|f(\boldsymbol{x}) - f(\boldsymbol{y})\| \leq \|\boldsymbol{x} - \boldsymbol{y}\|$。

**反向传播**：
$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{W}} = \frac{1}{\sigma_1}\frac{\partial\mathcal{L}}{\partial\bar{\boldsymbol{W}}} - \frac{1}{\sigma_1^2}\left(\frac{\partial\mathcal{L}}{\partial\bar{\boldsymbol{W}}}:\boldsymbol{W}\right)\boldsymbol{u}_1\boldsymbol{v}_1^{\top}$$

其中第二项是对归一化因子的修正。

**实践技巧**：
- 使用幂迭代估计$\sigma_1$（通常1-2次迭代足够）
- 在训练过程中复用上一步的$\boldsymbol{u}_1, \boldsymbol{v}_1$作为初始值
- 仅在判别器中应用，生成器不需要

#### 8.2 低秩正则化

在矩阵分解、推荐系统中，常用核范数正则化：
$$\min_{\boldsymbol{W}} \mathcal{L}(\boldsymbol{W}) + \lambda\|\boldsymbol{W}\|_*$$

梯度为：
$$\nabla_{\boldsymbol{W}}\mathcal{L} + \lambda\boldsymbol{U}\boldsymbol{V}^{\top}$$

这促进$\boldsymbol{W}$的低秩结构。

#### 8.3 正交化层（Orthogonal Layers）

在某些架构中（如Transformer的attention），希望保持权重正交性以避免梯度消失/爆炸。

可以参数化为：
$$\boldsymbol{W} = \boldsymbol{U}\boldsymbol{V}^{\top}$$

其中$\boldsymbol{U}, \boldsymbol{V}$是正交矩阵。通过SVD投影实现：
$$\boldsymbol{W}_{\text{proj}} = \boldsymbol{U}\boldsymbol{V}^{\top},\quad \text{where } \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \text{SVD}(\boldsymbol{W}_{\text{raw}})$$

梯度计算需要用到本文推导的公式。

### 九、与特征值分解导数的对比

#### 9.1 对称矩阵的特殊情况

对于对称正定矩阵$\boldsymbol{W} = \boldsymbol{W}^{\top}$，SVD退化为特征值分解：
$$\boldsymbol{W} = \boldsymbol{V}\boldsymbol{\Lambda}\boldsymbol{V}^{\top}$$

其中$\boldsymbol{\Lambda} = \text{diag}(\lambda_1,\ldots,\lambda_n)$是特征值对角阵。

**微分方程**：
$$\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V} = \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Lambda} + d\boldsymbol{\Lambda} + \boldsymbol{\Lambda}(d\boldsymbol{V})^{\top}\boldsymbol{V}$$

利用反对称性：
$$\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V} = \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Lambda} + d\boldsymbol{\Lambda} - \boldsymbol{\Lambda}\boldsymbol{V}^{\top}(d\boldsymbol{V})$$

**对角部分**：
$$d\lambda_i = \boldsymbol{v}_i^{\top}(d\boldsymbol{W})\boldsymbol{v}_i$$

**非对角部分**（$i\neq j$）：
$$[\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij} = [\boldsymbol{V}^{\top}(d\boldsymbol{V})]_{ij}(\lambda_j - \lambda_i)$$

解得：
$$[\boldsymbol{V}^{\top}(d\boldsymbol{V})]_{ij} = \frac{[\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V}]_{ij}}{\lambda_j - \lambda_i}$$

定义$\boldsymbol{K}_{ij} = \frac{1}{\lambda_j - \lambda_i}$（$i\neq j$），$\boldsymbol{K}_{ii} = 0$，则：
$$d\boldsymbol{V} = \boldsymbol{V}(\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V}))$$

#### 9.2 关键区别

| 特征值分解（对称矩阵） | SVD（一般矩阵） |
|---|---|
| $\boldsymbol{W} = \boldsymbol{V}\boldsymbol{\Lambda}\boldsymbol{V}^{\top}$ | $\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$ |
| 单个正交矩阵$\boldsymbol{V}$ | 两个正交矩阵$\boldsymbol{U}, \boldsymbol{V}$ |
| $\boldsymbol{K}_{ij} = \frac{1}{\lambda_j - \lambda_i}$ | $\boldsymbol{F}_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2}$ |
| 要求$\lambda_i\neq\lambda_j$ | 要求$\sigma_i\neq\sigma_j$ |
| 受约束$\boldsymbol{W} = \boldsymbol{W}^{\top}$ | 无约束 |

**注意**：在对称情况下，将$\boldsymbol{U} = \boldsymbol{V}$, $\sigma_i = \lambda_i$直接代入SVD公式会**导致错误**，因为：
1. 对$\boldsymbol{V}$求导时，实际上$\boldsymbol{U}$和$\boldsymbol{V}$都在变，导致重复计算
2. 对称约束$\boldsymbol{W} = \boldsymbol{W}^{\top}$需要额外的对称化操作

正确做法是从头推导对称情况，或在最终梯度上应用对称化$\text{sym}(\cdot) = \frac{1}{2}(\cdot + \cdot^{\top})$。

### 十、理论深化：扰动展开与高阶导数

#### 10.1 一阶扰动理论

考虑扰动$\boldsymbol{W}(\epsilon) = \boldsymbol{W}_0 + \epsilon\boldsymbol{E}$，其SVD为$\boldsymbol{U}(\epsilon)\boldsymbol{\Sigma}(\epsilon)\boldsymbol{V}(\epsilon)^{\top}$。

**Taylor展开**：
$$\sigma_i(\epsilon) = \sigma_i(0) + \epsilon\sigma_i'(0) + O(\epsilon^2)$$
$$\boldsymbol{u}_i(\epsilon) = \boldsymbol{u}_i(0) + \epsilon\boldsymbol{u}_i'(0) + O(\epsilon^2)$$
$$\boldsymbol{v}_i(\epsilon) = \boldsymbol{v}_i(0) + \epsilon\boldsymbol{v}_i'(0) + O(\epsilon^2)$$

从$\boldsymbol{W}(\epsilon) = \boldsymbol{U}(\epsilon)\boldsymbol{\Sigma}(\epsilon)\boldsymbol{V}(\epsilon)^{\top}$：
$$\boldsymbol{W}_0 + \epsilon\boldsymbol{E} = (\boldsymbol{U}_0 + \epsilon\boldsymbol{U}_0')(\boldsymbol{\Sigma}_0 + \epsilon\boldsymbol{\Sigma}_0')(\boldsymbol{V}_0 + \epsilon\boldsymbol{V}_0')^{\top} + O(\epsilon^2)$$

展开并比较$\epsilon$的系数：
$$\boldsymbol{E} = \boldsymbol{U}_0'\boldsymbol{\Sigma}_0\boldsymbol{V}_0^{\top} + \boldsymbol{U}_0\boldsymbol{\Sigma}_0'\boldsymbol{V}_0^{\top} + \boldsymbol{U}_0\boldsymbol{\Sigma}_0(\boldsymbol{V}_0')^{\top}$$

这与微分公式$d\boldsymbol{W} = (d\boldsymbol{U})\boldsymbol{\Sigma}\boldsymbol{V}^{\top} + \cdots$完全一致（识别$d\boldsymbol{U} = \epsilon\boldsymbol{U}_0'$等）。

#### 10.2 二阶导数

二阶导数$\frac{\partial^2\sigma_i}{\partial W_{jk}\partial W_{lm}}$涉及奇异向量的导数，计算复杂。

从$\frac{\partial\sigma_i}{\partial W_{jk}} = u_{ji}v_{ki}$求导：
$$\frac{\partial^2\sigma_i}{\partial W_{lm}\partial W_{jk}} = \frac{\partial u_{ji}}{\partial W_{lm}}v_{ki} + u_{ji}\frac{\partial v_{ki}}{\partial W_{lm}}$$

利用本文推导的$\frac{\partial\boldsymbol{u}_i}{\partial\boldsymbol{W}}$和$\frac{\partial\boldsymbol{v}_i}{\partial\boldsymbol{W}}$（以4阶张量形式），可以计算出二阶导数。

**应用**：二阶优化方法（如Newton法）需要Hessian矩阵。对于包含SVD的目标函数，可以用二阶导数构造精确的Hessian。

### 十一、数值验证与测试

#### 11.1 有限差分检验

验证梯度公式的标准方法是有限差分：
$$\frac{\partial f}{\partial W_{ij}}\approx\frac{f(\boldsymbol{W} + \epsilon\boldsymbol{E}_{ij}) - f(\boldsymbol{W} - \epsilon\boldsymbol{E}_{ij})}{2\epsilon}$$

其中$\boldsymbol{E}_{ij}$是第$(i,j)$位置为1其余为0的矩阵。

**测试案例**：
```python
import numpy as np

def test_svd_gradient():
    np.random.seed(42)
    n = 5
    W = np.random.randn(n, n)
    U, S, Vt = np.linalg.svd(W)

    # 解析梯度：最大奇异值
    grad_analytical = np.outer(U[:, 0], Vt[0, :])

    # 数值梯度
    epsilon = 1e-7
    grad_numerical = np.zeros_like(W)
    for i in range(n):
        for j in range(n):
            E = np.zeros_like(W)
            E[i, j] = 1
            s1_plus = np.linalg.svd(W + epsilon * E, compute_uv=False)[0]
            s1_minus = np.linalg.svd(W - epsilon * E, compute_uv=False)[0]
            grad_numerical[i, j] = (s1_plus - s1_minus) / (2 * epsilon)

    # 比较
    error = np.linalg.norm(grad_analytical - grad_numerical)
    print(f"Gradient error: {error}")
    assert error < 1e-5
```

#### 11.2 自动微分对比

现代深度学习框架（PyTorch, JAX）提供自动微分。可以对比手动实现与框架自动计算的梯度：

```python
import torch

def svd_spectral_norm_manual(W):
    """手动实现谱归一化的梯度"""
    U, S, V = torch.svd(W)
    sigma1 = S[0]
    u1 = U[:, 0:1]
    v1 = V[:, 0:1]
    W_normalized = W / sigma1
    return W_normalized, u1, v1, sigma1

# 对比自动微分
W = torch.randn(5, 5, requires_grad=True)
W_norm_manual, u1, v1, s1 = svd_spectral_norm_manual(W)
loss_manual = W_norm_manual.sum()
loss_manual.backward()
grad_manual = W.grad.clone()

# 使用自动微分
W.grad.zero_()
W_norm_auto = W / torch.norm(W, p=2)
loss_auto = W_norm_auto.sum()
loss_auto.backward()
grad_auto = W.grad

print("Gradient difference:", torch.norm(grad_manual - grad_auto))
```

### 十二、总结与展望

#### 12.1 核心要点回顾

1. **可导条件**：SVD可导当且仅当所有非零奇异值两两不等
2. **奇异值梯度**：$\nabla_{\boldsymbol{W}}\sigma_i = \boldsymbol{u}_i\boldsymbol{v}_i^{\top}$
3. **奇异向量梯度**：涉及矩阵$\boldsymbol{F}_{ij} = \frac{1}{\sigma_j^2 - \sigma_i^2}$的Hadamard积
4. **数值稳定性**：接近奇异值导致梯度爆炸，需用Taylor/Padé近似或幂迭代
5. **深度学习应用**：谱归一化、低秩正则化、正交化层

#### 12.2 理论意义

SVD求导展示了**微分几何**与**数值线性代数**的深刻联系：
- SVD定义了矩阵流形上的坐标系
- 奇异值/向量的变化受Riemann度量约束
- 梯度公式反映了流形的几何结构

#### 12.3 开放问题

1. **自适应正则化**：如何根据奇异值间隙自动调整$\boldsymbol{F}_{ij}$的正则化强度？
2. **高阶优化**：如何高效计算SVD的Hessian矩阵用于二阶优化？
3. **随机SVD的梯度**：对于大规模矩阵，随机SVD算法的梯度如何计算？
4. **非凸优化**：含SVD项的非凸优化问题的全局收敛性如何保证？

#### 12.4 实践建议

- **优先使用框架自动微分**：除非对性能有极致要求，否则让PyTorch/JAX处理SVD梯度
- **监控条件数**：训练过程中跟踪$\min_{i\neq j}|\sigma_i - \sigma_j|$，过小时调整正则化
- **幂迭代足够用**：大多数情况下，1-2次幂迭代即可获得足够精确的$\sigma_1, \boldsymbol{u}_1, \boldsymbol{v}_1$
- **避免完整SVD**：如果只需要少数奇异值，使用Lanczos或Arnoldi迭代而非完整分解

通过本节的详细推导，我们从多个角度理解了SVD导数的本质，掌握了理论基础、数值算法和实际应用。这些知识不仅在深度学习中重要，在优化、信号处理、控制理论等领域也有广泛应用。

