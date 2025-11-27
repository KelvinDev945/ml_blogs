---
title: 低秩近似之路（三）：CR
slug: 低秩近似之路三cr
date: 2024-10-11
tags: 近似, 最优, 矩阵, 低秩, 生成模型
status: completed
---

# 低秩近似之路（三）：CR

**原文链接**: [https://spaces.ac.cn/archives/10427](https://spaces.ac.cn/archives/10427)

**发布日期**: 

---

在[《低秩近似之路（二）：SVD》](/archives/10407)中，我们证明了SVD可以给出任意矩阵的最优低秩近似。那里的最优近似是无约束的，也就是说SVD给出的结果只管误差上的最小，不在乎矩阵的具体结构，而在很多应用场景中，出于可解释性或者非线性处理等需求，我们往往希望得到具有某些特殊结构的近似分解。

因此，从这篇文章开始，我们将探究一些具有特定结构的低秩近似，而本文将聚焦于其中的CR近似（Column-Row Approximation），它提供了加速矩阵乘法运算的一种简单方案。

## 问题背景 #

矩阵的最优$r$秩近似的一般提法是  
\begin{equation}\mathop{\text{argmin}}_{\text{rank}(\tilde{\boldsymbol{M}})\leq r}\Vert \tilde{\boldsymbol{M}} - \boldsymbol{M}\Vert_F^2\label{eq:loss-m2}\end{equation}  
其中$\boldsymbol{M},\tilde{\boldsymbol{M}}\in\mathbb{R}^{n\times m},r < \min(n,m)$。在前两篇文章中，我们已经讨论了两种情况：

> 1、如果$\tilde{\boldsymbol{M}}$不再有其他约束，那么$\tilde{\boldsymbol{M}}$的最优解就是$\boldsymbol{U}_{[:,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$，其中$\boldsymbol{M}=\boldsymbol{U}\boldsymbol{\Sigma} \boldsymbol{V}^{\top}$是$\boldsymbol{M}$的奇异值分解（SVD）；
> 
> 2、如果约定$\tilde{\boldsymbol{M}}=\boldsymbol{A}\boldsymbol{B}$（$\boldsymbol{A}\in\mathbb{R}^{n\times r},\boldsymbol{B}\in\mathbb{R}^{r\times m}$），且$\boldsymbol{A}$（或$\boldsymbol{B}$）已经给定，那么$\tilde{\boldsymbol{M}}$的最优解是$\boldsymbol{A} \boldsymbol{A}^{\dagger} \boldsymbol{M}$（或$\boldsymbol{M} \boldsymbol{B}^{\dagger} \boldsymbol{B}$），这里的${}^\dagger$是“[伪逆](/archives/10366)”。

这两个结果都有很广泛的应用，但它们都没有显式地引入$\tilde{\boldsymbol{M}}$与$\boldsymbol{M}$结构上的关联，这就导致了很难直观地看到$\tilde{\boldsymbol{M}}$与$\boldsymbol{M}$的关联，换言之$\tilde{\boldsymbol{M}}$的可解释性不强。

此外，如果目标中包含非线性运算如$\phi(\boldsymbol{X}\boldsymbol{W})$，通常也不允许我们使用任意实投影矩阵来降维，而是要求使用“选择矩阵（Selective Matrix）”，比如$\phi(\boldsymbol{X}\boldsymbol{W})\boldsymbol{S} = \phi(\boldsymbol{X}\boldsymbol{W}\boldsymbol{S})$对于任意矩阵$\boldsymbol{S}$不是恒成立的，但对于选择矩阵$\boldsymbol{S}$是恒成立的。

所以，接下来我们关注选择矩阵约束下的低秩近似。具体来说，我们有$\boldsymbol{X}\in\mathbb{R}^{n\times l},\boldsymbol{Y}\in\mathbb{R}^{l\times m}$，然后选定$\boldsymbol{M}=\boldsymbol{X}\boldsymbol{Y}$，我们的任务是从$\boldsymbol{X}$中选出$r$列、从$\boldsymbol{Y}$中选出相应的$r$行来构建$\tilde{\boldsymbol{M}}$，即  
\begin{equation}\mathop{\text{argmin}}_S\Vert \underbrace{\boldsymbol{X}_{[:,S]}}_{\boldsymbol{C}}\underbrace{\boldsymbol{Y}_{[S,:]}}_{\boldsymbol{R}} - \boldsymbol{X}\boldsymbol{Y}\Vert_F^2\quad\text{s.t.}\quad S\subset \\{0,1,\cdots,l-1\\}, |S|=r\end{equation}  
这里的$S$可以理解为slice，即按照Python的切片规则来理解，我们称$\boldsymbol{X}_{[:,S]}\boldsymbol{Y}_{[S,:]}$为$\boldsymbol{X}\boldsymbol{Y}$的“CR近似”。注意这种切片结果也可以用选择矩阵来等价描述，假设$\boldsymbol{X}_{[:,S]}$的第$1,2,\cdots,r$列分别为$\boldsymbol{X}$的第$s_1,s_2,\cdots,s_r$列，那么可以定义选择矩阵$\boldsymbol{S}\in\\{0,1\\}^{l\times r}$：  
\begin{equation}S_{i,j}=\left\\{\begin{aligned}&1, &i = s_j \\\ &0, &i\neq s_j\end{aligned}\right.\end{equation}  
即$\boldsymbol{S}$的第$j$列的第$s_j$个元素为1，其余都为0，这样一来就有$\boldsymbol{X}_{[:,S]}=\boldsymbol{X}\boldsymbol{S}$以及$\boldsymbol{Y}_{[S,:]}=\boldsymbol{S}^{\top} \boldsymbol{Y}$。

## 初步近似 #

如果我们将$\boldsymbol{X},\boldsymbol{Y}$分别表示成  
\begin{equation}\boldsymbol{X} = (\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_l),\quad \boldsymbol{Y}=\begin{pmatrix}\boldsymbol{y}_1^{\top} \\\ \boldsymbol{y}_2^{\top} \\\ \vdots \\\ \boldsymbol{y}_l^{\top}\end{pmatrix}\end{equation}  
其中$\boldsymbol{x}_i\in\mathbb{R}^{n\times 1},\boldsymbol{y}_i\in\mathbb{R}^{m\times 1}$都是列向量，那么$\boldsymbol{X}\boldsymbol{Y}$可以写成  
\begin{equation}\boldsymbol{X}\boldsymbol{Y} = \sum_{i=1}^l \boldsymbol{x}_i\boldsymbol{y}_i^{\top}\end{equation}  
而找$\boldsymbol{X}\boldsymbol{Y}$的最优CR近似则可以等价地写成  
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\\{0,1\\}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{x}_i\boldsymbol{y}_i^{\top} - \sum_{i=1}^l\boldsymbol{x}_i\boldsymbol{y}_i^{\top}\right\Vert_F^2\quad\text{s.t.}\quad \sum_{i=1}^l \lambda_i = r\label{eq:xy-l-k}\end{equation}  
我们知道，矩阵的$F$范数实际上就是将矩阵展平成向量来算模长，所以这个优化问题本质上就相当于给定$l$个向量$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l\in\mathbb{R}^d$，求  
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\\{0,1\\}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{v}_i - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^l \lambda_i = r\label{eq:v-l-k}\end{equation}  
其中$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i \boldsymbol{y}_i^{\top})$，$d=nm$。记$\gamma_i = 1 - \lambda_i$，那么可以进一步简化成  
\begin{equation}\mathop{\text{argmin}}_{\gamma_1,\gamma_2,\cdots,\gamma_l\in\\{0,1\\}}\left\Vert\sum_{i=1}^l \gamma_i \boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^l \gamma_i = l-r\label{eq:v-l-k-0}\end{equation}  
如果笔者没有理解错，这个优化问题的精确求解是NP-Hard的，所以一般情况下只能寻求近似算法。一个可精确求解的简单例子是$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l$两两垂直，此时  
\begin{equation}\left\Vert\sum_{i=1}^l \gamma_i \boldsymbol{v}_i\right\Vert^2 = \sum_{i=1}^l \gamma_i^2 \Vert\boldsymbol{v}_i\Vert^2\end{equation}  
所以它的最小值就是最小的$l-r$个$\Vert\boldsymbol{v}_i\Vert^2$之和，即让模长最小的$l-r$个$\boldsymbol{v}_i$的$\gamma_i$等于1，剩下的$\gamma_i$则等于0。当两两正交的条件不严格成立时，我们依然可以将选择模长最小的$l-r$个$\boldsymbol{v}_i$作为一个近似解。回到原始的CR近似问题上，我们有$\Vert\boldsymbol{x}_i\boldsymbol{y}_i^{\top}\Vert_F = \Vert\boldsymbol{x}_i\Vert \Vert \boldsymbol{y}_i\Vert$，所以$\boldsymbol{X}\boldsymbol{Y}$的最优CR近似的一个baseline，就是保留$\boldsymbol{X}$的列向量与$\boldsymbol{Y}$对应的行向量模长乘积最大的$r$个列/行向量。

## 采样视角 #

有一些场景允许我们将式$\eqref{eq:xy-l-k}$放宽为  
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\mathbb{R}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{x}_i\boldsymbol{y}_i^{\top} - \sum_{i=1}^l\boldsymbol{x}_i\boldsymbol{y}_i^{\top}\right\Vert_F^2\quad\text{s.t.}\quad \sum_{i=1}^l \\#[\lambda_i\neq 0] = r\end{equation}  
其中$\\#[\lambda_i\neq 0]$表示$\lambda_i\neq 0$时输出1，否则输出0。这个宽松版本其实就是将CR近似的形式从$\boldsymbol{C}\boldsymbol{R}$拓展成$\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$，其中$\boldsymbol{\Lambda}\in\mathbb{R}^{r\times r}$是对角阵，即允许我们调整对角阵$\boldsymbol{\Lambda}\in\mathbb{R}^{r\times r}$来达到更高的精度。相应地，式$\eqref{eq:v-l-k}$变为  
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\mathbb{R}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{v}_i - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^l \\#[\lambda_i\neq 0] = r\end{equation}

这样放宽之后，我们可以从采样视角来看待这个问题。首先我们引入任意$l$元分布$\boldsymbol{p}=(p_1,p_2,\cdots,p_l)$，然后我们就可以写出  
\begin{equation}\sum_{i=1}^l\boldsymbol{v}_i = \sum_{i=1}^l p_i\times\frac{\boldsymbol{v}_i}{p_i} = \mathbb{E}_{i\sim \boldsymbol{p}} \left[\frac{\boldsymbol{v}_i}{p_i}\right] \end{equation}  
也就是说，$\boldsymbol{v}_i/p_i$的数学期望正好是我们要逼近的目标，所以我们可以通过从$\boldsymbol{p}$分布 _**独立重复采样**_ 来构建近似：  
\begin{equation}\sum_{i=1}^l\boldsymbol{v}_i = \mathbb{E}_{i\sim \boldsymbol{p}} \left[\frac{\boldsymbol{v}_i}{p_i}\right] \approx \frac{1}{r}\sum_{j=1}^r \frac{\boldsymbol{v}_{s_j}}{p_{s_j}},\quad s_1,s_2,\cdots,s_r\sim \boldsymbol{p}\end{equation}  
这意味着当$i$是$s_1,s_2,\cdots,s_r$之一时有$\lambda_i = (r p_i)^{-1}$，否则$\lambda_i=0$。可能有读者疑问为什么要独立重复采样，而不是更符合逼近需求的不放回采样呢？无他，纯粹是因为独立重复采样使得后面的分析更简单。

到目前为止，我们的理论结果跟分布$\boldsymbol{p}$的选择无关，也就是说对于任意$\boldsymbol{p}$都是成立的，这给我们提供了选择最优$\boldsymbol{p}$的可能性。那如何衡量$\boldsymbol{p}$的优劣呢？很显然我们希望每次采样估计的误差越小越好，因此可以用采样估计的误差  
\begin{equation}\mathbb{E}_{i\sim \boldsymbol{p}} \left[\left\Vert\frac{\boldsymbol{v}_i}{p_i} - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\right] = \left(\sum_{i=1}^l \frac{\Vert\boldsymbol{v}_i\Vert^2}{p_i}\right) - \left\Vert\sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2 \end{equation}  
来比较不同的$\boldsymbol{p}$之间的优劣。接着利用均值不等式有  
\begin{equation}\sum_{i=1}^l \frac{\Vert\boldsymbol{v}_i\Vert^2}{p_i} = \left(\sum_{i=1}^l \frac{\Vert\boldsymbol{v}_i\Vert^2}{p_i} + p_i Z^2\right) - Z^2\geq \left(\sum_{i=1}^l 2\Vert\boldsymbol{v}_i\Vert Z\right) - Z^2\end{equation}  
等号在$\Vert\boldsymbol{v}_i\Vert^2 / p_i = p_i Z^2$时取到，由此可得最优的$\boldsymbol{p}$为  
\begin{equation}p_i^* = \frac{\Vert\boldsymbol{v}_i\Vert}{Z},\quad Z = \sum\limits_{i=1}^l \Vert\boldsymbol{v}_i\Vert\end{equation}  
对应的误差为  
\begin{equation}\mathbb{E}_{i\sim \boldsymbol{p}} \left[\left\Vert\frac{\boldsymbol{v}_i}{p_i} - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\right] = \left(\sum_{i=1}^l \Vert\boldsymbol{v}_i\Vert\right)^2 - \left\Vert\sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2 \end{equation}  
最优的$p_i$正好正比于$\Vert\boldsymbol{v}_i\Vert$，所以概率最大的$r$个$\boldsymbol{v}_i$也正是模长最大的$r$个$\boldsymbol{v}_i$，这就跟上一节的近似联系起来了。该结果出自2006年的论文[《Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication》](https://www.stat.berkeley.edu/~mmahoney/pubs/matrix1_SICOMP.pdf)，初衷是加速矩阵乘法，它表明只要按照$p_i\propto \Vert \boldsymbol{x}_i\boldsymbol{y}_i^{\top}\Vert_F = \Vert \boldsymbol{x}_i\Vert \Vert\boldsymbol{y}_i\Vert$来采样$\boldsymbol{X},\boldsymbol{Y}$对应的列/行，并乘以$(r p_i)^{-1/2}$，就可以得到$\boldsymbol{X}\boldsymbol{Y}$的一个CR近似，从而将乘法复杂度从$\mathcal{O}(lmn)$降低到$\mathcal{O}(rmn)$。

## 延伸讨论 #

不管是按模长排序还是按$p_i\propto \Vert\boldsymbol{v}_i\Vert$随机采样，它们都允许我们在线性复杂度【即$\mathcal{O}(l)$】内构建一个CR近似，这对于实时计算来说当然是很理想的，但由于排序或采样都只依赖于$\Vert\boldsymbol{v}_i\Vert$，所以精度只能说一般。如果我们可以接受更高的复杂度，那么如何提高CR近似的精度呢？

我们可以尝试将排序的单位改为$k$元组。简单起见，假设$k \leq l-r$是$l-r$的一个因数，$l$个向量$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l$选$k$个的组合数为$C_l^k$，对于每个组合$\\{s_1,s_2,\cdots,s_k\\}$我们都可以算出向量和的模长$\Vert \boldsymbol{v}_{s_1} + \boldsymbol{v}_{s_2} + \cdots + \boldsymbol{v}_{s_k}\Vert$。有了这些数据，我们就可以贪婪地构建$\eqref{eq:v-l-k-0}$的近似解：

> 初始化$\Omega = \\{1,2,\cdots,l\\},\Theta=\\{\\}$
> 
> 对于$t=1,2,\cdots,(l-r)/k$，执行：  
>  $\Theta = \Theta\,\cup\,\mathop{\text{argmin}}\limits_{\\{s_1,s_2,\cdots,s_k\\}\subset \Omega}\Vert \boldsymbol{v}_{s_1} + \boldsymbol{v}_{s_2} + \cdots + \boldsymbol{v}_{s_k}\Vert$；  
>  $\Omega = \Omega\,\backslash\,\Theta$；
> 
> 返回$\Theta$。

说白了，就是每次都从剩下的向量中挑选和模长最小的$k$个向量，重复挑选$(l-r)/k$次即得到$l-r$个向量，它是按照单个向量模长来排序的自然推广，其复杂度为$\mathcal{O}(C_l^k)$，当$k > 1$且$l$比较大时可能难以承受，这也侧面体现了原问题精确求解的复杂性。

另一个值得思考的问题是如果允许CR近似放宽为$\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$，那么$\boldsymbol{\Lambda}$的最优解是什么呢？如果不限定$\boldsymbol{\Lambda}$的结构，那么答案可以由伪逆给出  
\begin{equation}\boldsymbol{\Lambda}^* = \mathop{\text{argmin}}_{\boldsymbol{\Lambda}}\Vert \boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R} - \boldsymbol{X}\boldsymbol{Y}\Vert_F^2 = \boldsymbol{C}^{\dagger}\boldsymbol{X}\boldsymbol{Y}\boldsymbol{R}^{\dagger}\end{equation}  
如果$\boldsymbol{\Lambda}$必须是对角阵呢？那可以先将问题重述为给定$\\{\boldsymbol{u}_1,\boldsymbol{u}_2,\cdots,\boldsymbol{u}_r\\}\subset\\{\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l\\}$，求  
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_r}\left\Vert\sum_{i=1}^r \lambda_i \boldsymbol{u}_i - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\end{equation}  
我们记$\boldsymbol{U} = (\boldsymbol{u}_1,\boldsymbol{u}_2,\cdots,\boldsymbol{u}_r), \boldsymbol{V} = (\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l), \boldsymbol{\lambda}=(\lambda_1,\lambda_2,\cdots,\lambda_r)^{\top}$，那么优化目标可以写成  
\begin{equation}\mathop{\text{argmin}}_{\boldsymbol{\lambda}}\left\Vert\boldsymbol{U}\boldsymbol{\lambda} - \boldsymbol{V}\boldsymbol{1}_{l\times 1}\right\Vert^2\end{equation}  
这同样可以通过伪逆写出最优解  
\begin{equation}\boldsymbol{\lambda}^* = \boldsymbol{U}^{\dagger}\boldsymbol{V}\boldsymbol{1}_{l\times 1} = (\boldsymbol{U}^{\top}\boldsymbol{U})^{-1}\boldsymbol{U}^{\top}\boldsymbol{V}\boldsymbol{1}_{l\times 1} \end{equation}  
最后一个等号假设了$\boldsymbol{U}^{\top}\boldsymbol{U}$可逆，这通常能满足，如果不满足的话$(\boldsymbol{U}^{\top}\boldsymbol{U})^{-1}$改$(\boldsymbol{U}^{\top}\boldsymbol{U})^{\dagger}$就行。

现在的问题是直接套用上式的话对原始问题来说计算量太大，因为$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i \boldsymbol{y}_i^{\top})$，即$\boldsymbol{v}_i$是$mn$维向量，所以$\boldsymbol{V}$大小为$mn\times l$、$\boldsymbol{U}$大小为$mn\times r$，这在$m,n$较大时比较难受。利用$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i \boldsymbol{y}_i^{\top})$能帮我们进一步化简上式。不妨设$\boldsymbol{u}_i = \text{vec}(\boldsymbol{c}_i \boldsymbol{r}_i^{\top})$，那么  
\begin{equation}\begin{aligned}(\boldsymbol{U}^{\top}\boldsymbol{V})_{i,j} =&\, \langle \boldsymbol{c}_i \boldsymbol{r}_i^{\top}, \boldsymbol{x}_j \boldsymbol{y}_j^{\top}\rangle_F = \text{Tr}(\boldsymbol{r}_i \boldsymbol{c}_i^{\top}\boldsymbol{x}_j \boldsymbol{y}_j^{\top}) = (\boldsymbol{c}_i^{\top}\boldsymbol{x}_j)(\boldsymbol{r}_i^{\top} \boldsymbol{y}_j) \\\\[5pt]  
=&\, [(\boldsymbol{C}^{\top}\boldsymbol{X})\otimes (\boldsymbol{R}\boldsymbol{Y}^{\top})]_{i,j}  
\end{aligned}\end{equation}  
即$\boldsymbol{U}^{\top}\boldsymbol{V}=(\boldsymbol{C}^{\top}\boldsymbol{X})\otimes (\boldsymbol{R}\boldsymbol{Y}^{\top}),\boldsymbol{U}^{\top}\boldsymbol{U}=(\boldsymbol{C}^{\top}\boldsymbol{C})\otimes (\boldsymbol{R}\boldsymbol{R}^{\top})$，这里的$\otimes$是[Hadamard积](https://en.wikipedia.org/wiki/Hadamard_product_\(matrices\))，这样恒等变换之后$\boldsymbol{U}^{\top}\boldsymbol{V}$和$\boldsymbol{U}^{\top}\boldsymbol{U}$的计算量就降低了。

## 文章小结 #

本文介绍了矩阵乘法的CR近似，这是一种具有特定行列结构的低秩近似，相比由SVD给出的最优低秩近似，CR近似具有更直观的物理意义以及更好的可解释性。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10427>_

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

苏剑林. (Oct. 11, 2024). 《低秩近似之路（三）：CR 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10427>

@online{kexuefm-10427,  
title={低秩近似之路（三）：CR},  
author={苏剑林},  
year={2024},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/10427}},  
} 


---

## 推导

### 1. CR分解的形式化定义

CR分解（Column-Row decomposition）是一种特殊的矩阵分解方法，它通过选择原矩阵的部分列和对应行来构造低秩近似。对于矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，我们可以将其写成$\boldsymbol{M}=\boldsymbol{X}\boldsymbol{Y}$的形式，其中$\boldsymbol{X}\in\mathbb{R}^{n\times l}$，$\boldsymbol{Y}\in\mathbb{R}^{l\times m}$。

**定义1**（CR分解）：给定矩阵$\boldsymbol{M}=\boldsymbol{X}\boldsymbol{Y}$和秩参数$r < l$，CR分解寻找索引集$S\subset\{1,2,\cdots,l\}$满足$|S|=r$，使得
$$\boldsymbol{M}\approx \boldsymbol{C}\boldsymbol{R} = \boldsymbol{X}_{[:,S]}\boldsymbol{Y}_{[S,:]}$$
其中$\boldsymbol{C}\in\mathbb{R}^{n\times r}$由$\boldsymbol{X}$的$r$列组成，$\boldsymbol{R}\in\mathbb{R}^{r\times m}$由$\boldsymbol{Y}$的对应$r$行组成。

更一般地，我们可以引入权重矩阵$\boldsymbol{\Lambda}\in\mathbb{R}^{r\times r}$，得到加权CR分解：
$$\boldsymbol{M}\approx \boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$$

**命题1**（选择矩阵表示）：令$\boldsymbol{S}\in\{0,1\}^{l\times r}$为选择矩阵，其中$S_{i,j}=1$当且仅当$i=s_j$（第$j$个被选中的索引），则有
$$\boldsymbol{C} = \boldsymbol{X}\boldsymbol{S},\quad \boldsymbol{R} = \boldsymbol{S}^{\top}\boldsymbol{Y}$$

证明：选择矩阵$\boldsymbol{S}$的第$j$列是一个单位向量，其第$s_j$个位置为1，其余为0。因此$\boldsymbol{X}\boldsymbol{S}$选择了$\boldsymbol{X}$的第$s_1,s_2,\cdots,s_r$列，$\boldsymbol{S}^{\top}\boldsymbol{Y}$选择了$\boldsymbol{Y}$的对应行。

### 2. 列子集选择问题（Column Subset Selection Problem）

**定义2**（CSSP）：列子集选择问题（Column Subset Selection Problem, CSSP）定义为：给定$\boldsymbol{A}\in\mathbb{R}^{n\times m}$和整数$r$，寻找$r$个列的索引集$S$，使得
$$\min_{S:|S|=r}\min_{\boldsymbol{Z}\in\mathbb{R}^{r\times m}}\|\boldsymbol{A} - \boldsymbol{A}_{[:,S]}\boldsymbol{Z}\|_F$$

这个问题在CR分解中有直接应用。当我们固定了列选择$\boldsymbol{C}$后，最优的$\boldsymbol{Z}$可以通过伪逆得到：
$$\boldsymbol{Z}^* = \boldsymbol{C}^{\dagger}\boldsymbol{A} = (\boldsymbol{C}^{\top}\boldsymbol{C})^{-1}\boldsymbol{C}^{\top}\boldsymbol{A}$$

**定理1**（CSSP的复杂度）：CSSP问题是NP-hard的。

证明思路：可以归约到组合优化问题。从$l$个列中选择$r$个的组合数为$\binom{l}{r}$，穷举搜索的复杂度为指数级。即使贪婪算法也无法保证常数近似比。

因此，我们需要随机化算法和近似算法来处理CSSP。

### 3. Leverage Score采样理论

leverage score是一种重要的列重要性度量，它源于统计回归和数值线性代数。

**定义3**（Leverage Scores）：给定矩阵$\boldsymbol{A}\in\mathbb{R}^{n\times m}$，设其秩-$r$ SVD为$\boldsymbol{A}\approx\boldsymbol{U}_r\boldsymbol{\Sigma}_r\boldsymbol{V}_r^{\top}$，则第$i$列的leverage score定义为：
$$\tau_i = \|\boldsymbol{U}_r^{\top}\boldsymbol{a}_i\|^2 = \sum_{j=1}^r U_{j,i}^2$$
其中$\boldsymbol{a}_i$是$\boldsymbol{A}$的第$i$列，$\boldsymbol{U}_r$是左奇异向量矩阵的前$r$列。

**性质**：leverage scores满足：
$$\sum_{i=1}^m \tau_i = \text{Tr}(\boldsymbol{U}_r^{\top}\boldsymbol{U}_r) = r$$

**几何解释**：$\tau_i$度量了第$i$列在前$r$个左奇异向量张成的子空间中的"影响力"。leverage score大的列对子空间贡献更多。

对于CR分解，我们可以定义联合leverage score。设$\boldsymbol{M}=\boldsymbol{X}\boldsymbol{Y}$，令
$$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i\boldsymbol{y}_i^{\top})\in\mathbb{R}^{nm}$$
堆叠所有$\boldsymbol{v}_i$得到矩阵$\boldsymbol{V}=(\boldsymbol{v}_1,\cdots,\boldsymbol{v}_l)\in\mathbb{R}^{nm\times l}$。

**定义4**（CR的Leverage Scores）：对于CR分解，第$i$个列-行对的leverage score定义为：
$$\ell_i = \|\boldsymbol{P}_r\boldsymbol{v}_i\|^2 / \|\boldsymbol{v}_i\|^2$$
其中$\boldsymbol{P}_r$是$\boldsymbol{V}$的前$r$个主成分子空间的投影矩阵。

然而，直接计算leverage scores需要SVD，代价为$\mathcal{O}(\min(nm^2l, n^2ml))$，这太昂贵。因此我们使用更简单的采样分布。

### 4. 基于模长的采样策略

回顾正文中的结果，最优采样分布为：
$$p_i = \frac{\|\boldsymbol{v}_i\|}{\sum_{j=1}^l\|\boldsymbol{v}_j\|} = \frac{\|\boldsymbol{x}_i\|\cdot\|\boldsymbol{y}_i\|}{\sum_{j=1}^l\|\boldsymbol{x}_j\|\cdot\|\boldsymbol{y}_j\|}$$

现在我们严格推导这个结果并给出误差界。

**定理2**（无偏估计）：定义采样估计
$$\hat{\boldsymbol{M}} = \frac{1}{r}\sum_{j=1}^r \frac{\boldsymbol{x}_{s_j}\boldsymbol{y}_{s_j}^{\top}}{p_{s_j}}$$
其中$s_1,\cdots,s_r$独立同分布地从分布$\boldsymbol{p}=(p_1,\cdots,p_l)$中采样，则
$$\mathbb{E}[\hat{\boldsymbol{M}}] = \boldsymbol{M}$$

证明：
\begin{align}
\mathbb{E}[\hat{\boldsymbol{M}}] &= \frac{1}{r}\sum_{j=1}^r \mathbb{E}\left[\frac{\boldsymbol{x}_{s_j}\boldsymbol{y}_{s_j}^{\top}}{p_{s_j}}\right] \\
&= \frac{1}{r}\sum_{j=1}^r \sum_{i=1}^l p_i \cdot \frac{\boldsymbol{x}_i\boldsymbol{y}_i^{\top}}{p_i} \\
&= \frac{1}{r}\sum_{j=1}^r \sum_{i=1}^l \boldsymbol{x}_i\boldsymbol{y}_i^{\top} \\
&= \sum_{i=1}^l \boldsymbol{x}_i\boldsymbol{y}_i^{\top} = \boldsymbol{M}
\end{align}

**定理3**（方差界）：采样估计的方差为：
$$\mathbb{E}\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F^2 = \frac{1}{r}\left(\sum_{i=1}^l \frac{\|\boldsymbol{v}_i\|^2}{p_i} - \|\boldsymbol{M}\|_F^2\right)$$

证明：由于$s_1,\cdots,s_r$独立同分布，
\begin{align}
\mathbb{E}\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F^2 &= \mathbb{E}\left\|\frac{1}{r}\sum_{j=1}^r\left(\frac{\boldsymbol{v}_{s_j}}{p_{s_j}} - \boldsymbol{M}\right)\right\|^2 \\
&= \frac{1}{r^2}\sum_{j=1}^r \mathbb{E}\left\|\frac{\boldsymbol{v}_{s_j}}{p_{s_j}} - \boldsymbol{M}\right\|^2 \\
&= \frac{1}{r}\mathbb{E}\left\|\frac{\boldsymbol{v}_{s_1}}{p_{s_1}} - \boldsymbol{M}\right\|^2
\end{align}
其中第二行利用了独立性（交叉项期望为0），第三行利用了同分布性。

继续计算：
\begin{align}
\mathbb{E}\left\|\frac{\boldsymbol{v}_{s_1}}{p_{s_1}} - \boldsymbol{M}\right\|^2 &= \mathbb{E}\left\|\frac{\boldsymbol{v}_{s_1}}{p_{s_1}}\right\|^2 - \|\boldsymbol{M}\|^2 \\
&= \sum_{i=1}^l p_i \cdot \frac{\|\boldsymbol{v}_i\|^2}{p_i^2} - \|\boldsymbol{M}\|^2 \\
&= \sum_{i=1}^l \frac{\|\boldsymbol{v}_i\|^2}{p_i} - \|\boldsymbol{M}\|_F^2
\end{align}

**推论1**（最优分布）：使方差最小的分布为
$$p_i^* = \frac{\|\boldsymbol{v}_i\|}{\sum_{j=1}^l\|\boldsymbol{v}_j\|}$$
对应的最小方差为：
$$\mathbb{E}\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F^2 = \frac{1}{r}\left[\left(\sum_{i=1}^l\|\boldsymbol{v}_i\|\right)^2 - \|\boldsymbol{M}\|_F^2\right]$$

证明：这是带约束$\sum_i p_i=1$的凸优化问题。构造拉格朗日函数：
$$\mathcal{L} = \sum_{i=1}^l \frac{\|\boldsymbol{v}_i\|^2}{p_i} + \lambda\left(\sum_{i=1}^l p_i - 1\right)$$

求导并令其为零：
$$\frac{\partial\mathcal{L}}{\partial p_i} = -\frac{\|\boldsymbol{v}_i\|^2}{p_i^2} + \lambda = 0$$

得到$p_i = \|\boldsymbol{v}_i\|/\sqrt{\lambda}$。代入约束$\sum_i p_i=1$得到$\sqrt{\lambda}=\sum_j\|\boldsymbol{v}_j\|$，因此
$$p_i^* = \frac{\|\boldsymbol{v}_i\|}{\sum_{j=1}^l\|\boldsymbol{v}_j\|}$$

### 5. 概率误差界的推导

现在我们推导更精细的概率界，而不仅仅是期望误差。

**定理4**（Frobenius范数的概率界）：设$\hat{\boldsymbol{M}}$是按$p_i^*$采样$r$次得到的估计，则对任意$\delta\in(0,1)$，以至少$1-\delta$的概率有：
$$\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F \leq \frac{1}{\sqrt{r}}\left(\sum_{i=1}^l\|\boldsymbol{v}_i\|\right)\sqrt{\frac{2\ln(2/\delta)}{1}}$$

证明：定义独立随机变量
$$\boldsymbol{Z}_j = \frac{\boldsymbol{v}_{s_j}}{p_{s_j}} - \boldsymbol{M}$$

则$\mathbb{E}[\boldsymbol{Z}_j]=0$且$\|\boldsymbol{Z}_j\|$有界。我们有
$$\hat{\boldsymbol{M}} - \boldsymbol{M} = \frac{1}{r}\sum_{j=1}^r \boldsymbol{Z}_j$$

注意到
$$\|\boldsymbol{Z}_j\| \leq \frac{\|\boldsymbol{v}_{s_j}\|}{p_{s_j}} + \|\boldsymbol{M}\| \leq \frac{\sum_i\|\boldsymbol{v}_i\|}{\min_i p_i} + \|\boldsymbol{M}\|$$

由于$p_i^* = \|\boldsymbol{v}_i\|/Z$（其中$Z=\sum_i\|\boldsymbol{v}_i\|$），我们有$\min_i p_i \geq \|\boldsymbol{v}_{\min}\|/Z$。

应用矩阵Bernstein不等式（见参考文献[Tropp, 2012]），可得上述概率界。具体细节涉及复杂的集中不等式理论。

**定理5**（谱范数界）：以高概率，采样估计还满足谱范数界：
$$\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_2 \leq \mathcal{O}\left(\frac{\sigma_{\max}(\boldsymbol{M})\sqrt{l\log l}}{\sqrt{r}}\right)$$
其中$\sigma_{\max}(\boldsymbol{M})$是$\boldsymbol{M}$的最大奇异值。

这个界说明：随着采样数$r$增加，误差以$1/\sqrt{r}$的速度下降。

### 6. 与截断SVD的误差对比

回顾截断SVD给出的最优秩-$r$近似：
$$\boldsymbol{M}_r^{\text{SVD}} = \boldsymbol{U}_r\boldsymbol{\Sigma}_r\boldsymbol{V}_r^{\top}$$

其误差为：
$$\|\boldsymbol{M} - \boldsymbol{M}_r^{\text{SVD}}\|_F = \sqrt{\sum_{i=r+1}^{\min(n,m)}\sigma_i^2}$$

这是所有秩-$r$矩阵中的最优误差（Eckart-Young定理）。

**定理6**（CR vs SVD误差比较）：对于CR采样估计$\hat{\boldsymbol{M}}$，期望误差满足
$$\mathbb{E}\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F^2 \geq \|\boldsymbol{M} - \boldsymbol{M}_r^{\text{SVD}}\|_F^2$$

这是因为SVD提供了全局最优解，而CR受到列选择的约束。

**定量比较**：设$\boldsymbol{M}$的奇异值为$\sigma_1\geq\sigma_2\geq\cdots$，则
- SVD误差：$\|\boldsymbol{M} - \boldsymbol{M}_r^{\text{SVD}}\|_F^2 = \sum_{i>r}\sigma_i^2$
- CR期望误差：$\mathbb{E}\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F^2 = \frac{1}{r}\left[(\sum_i\|\boldsymbol{v}_i\|)^2 - \|\boldsymbol{M}\|_F^2\right]$

**例子**：假设$\boldsymbol{M}$的奇异值呈快速衰减，即$\sigma_i \approx \sigma_1 e^{-\alpha i}$，则：
- SVD能捕获主要能量，误差小
- CR的性能依赖于列之间的相关性：如果$\boldsymbol{v}_i$之间正交性强，CR表现接近SVD；如果相关性强，CR误差可能显著大于SVD

**近似比**：在最坏情况下，可以证明存在常数$c$使得
$$\mathbb{E}\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F^2 \leq c\cdot r\cdot \|\boldsymbol{M} - \boldsymbol{M}_r^{\text{SVD}}\|_F^2$$

但在实际应用中，通过leverage score采样可以获得更好的常数$c$。

### 7. Determinantal Point Process (DPP) 采样

DPP是一种优雅的采样方法，它能自然地捕获多样性，在CR分解中表现出色。

**定义5**（DPP）：一个离散DPP由核矩阵$\boldsymbol{L}\in\mathbb{R}^{l\times l}$（半正定）定义。子集$S\subseteq\{1,\cdots,l\}$的采样概率为：
$$\mathbb{P}(S) \propto \det(\boldsymbol{L}_{S,S})$$
其中$\boldsymbol{L}_{S,S}$是$\boldsymbol{L}$的子矩阵，索引为$S$。

归一化常数为：
$$\sum_{S\subseteq\{1,\cdots,l\}}\det(\boldsymbol{L}_{S,S}) = \det(\boldsymbol{I} + \boldsymbol{L})$$

**DPP用于CR分解**：定义核矩阵
$$\boldsymbol{L}_{ij} = \langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle = \text{Tr}(\boldsymbol{y}_i\boldsymbol{x}_i^{\top}\boldsymbol{x}_j\boldsymbol{y}_j^{\top})$$

即$\boldsymbol{L} = \boldsymbol{V}^{\top}\boldsymbol{V}$，其中$\boldsymbol{V}=(\boldsymbol{v}_1,\cdots,\boldsymbol{v}_l)$。

**k-DPP**：为了精确采样大小为$r$的子集，我们使用$k$-DPP，它条件化在$|S|=r$：
$$\mathbb{P}(S \mid |S|=r) \propto \det(\boldsymbol{L}_{S,S})$$

**定理7**（DPP的期望投影）：如果$S$从$k$-DPP采样（$k=r$），定义投影
$$\boldsymbol{P}_S = \boldsymbol{V}_S(\boldsymbol{V}_S^{\top}\boldsymbol{V}_S)^{-1}\boldsymbol{V}_S^{\top}$$
其中$\boldsymbol{V}_S = (\boldsymbol{v}_{s_1},\cdots,\boldsymbol{v}_{s_r})$，则
$$\mathbb{E}[\boldsymbol{P}_S] = \boldsymbol{U}_r\boldsymbol{U}_r^{\top}$$
其中$\boldsymbol{U}_r$是$\boldsymbol{V}$的前$r$个左奇异向量。

这表明DPP自动"瞄准"主要子空间！

**DPP采样算法**：
1. 计算$\boldsymbol{L} = \boldsymbol{V}^{\top}\boldsymbol{V}$的特征分解：$\boldsymbol{L} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$
2. 对每个特征值$\lambda_i$，以概率$\lambda_i/(\lambda_i+1)$选入集合$J$
3. 从$J$中采样基向量，构造子集$S$

复杂度为$\mathcal{O}(l^3)$（特征分解）$+$ $\mathcal{O}(lr^2)$（采样）。

**定理8**（DPP的误差界）：使用DPP采样的CR近似满足：
$$\mathbb{E}\|\boldsymbol{M} - \boldsymbol{C}\boldsymbol{C}^{\dagger}\boldsymbol{M}\|_F^2 \leq \frac{r}{r-k+1}\|\boldsymbol{M} - \boldsymbol{M}_k^{\text{SVD}}\|_F^2$$
对于$k\leq r$。

这给出了与最优SVD近似的相对误差保证。

**DPP的优势**：
1. 多样性：自动选择"分散"的列，避免冗余
2. 理论保证：有严格的误差界
3. 无需调参：不需要手工设计采样分布

**DPP的劣势**：
1. 计算代价：需要$\mathcal{O}(l^3)$时间
2. 对于超大规模问题，仍需近似DPP方法

### 8. QR分解的列主元策略

列主元QR（Column-Pivoted QR, CPQR）是一种经典的确定性列选择方法。

**定义6**（列主元QR）：给定$\boldsymbol{A}\in\mathbb{R}^{n\times m}$，CPQR寻找置换矩阵$\boldsymbol{\Pi}$和正交矩阵$\boldsymbol{Q}$、上三角矩阵$\boldsymbol{R}$使得
$$\boldsymbol{A}\boldsymbol{\Pi} = \boldsymbol{Q}\boldsymbol{R}$$
其中$\boldsymbol{\Pi}$的选择使得$\boldsymbol{R}$的对角元素递减：$|R_{11}|\geq |R_{22}|\geq\cdots\geq |R_{mm}|$。

**贪婪列主元算法**：
```
初始化：剩余矩阵 A_res = A
for k = 1 to r:
    选择列 j = argmax_i ||A_res[:,i]||
    交换 A[:, k] 和 A[:, j]
    Householder变换消去 A[k:, k] 下方元素
    更新 A_res
```

复杂度：$\mathcal{O}(nmr)$。

**定理9**（CPQR的误差界）：设$\boldsymbol{R}_{11}\in\mathbb{R}^{r\times r}$是前$r$步的子矩阵，则
$$\sigma_{\min}(\boldsymbol{R}_{11}) \geq \sigma_r(\boldsymbol{A})/\sqrt{1 + r(m-r)}$$

这保证了选出的列具有良好的数值性质。

**秩揭示性质**：CPQR能有效识别矩阵的数值秩。如果$\sigma_{r+1}(\boldsymbol{A}) \ll \sigma_r(\boldsymbol{A})$，则$|R_{r+1,r+1}|$会显著小于$|R_{rr}|$。

### 9. 强秩揭示QR分解（RRQR）

标准CPQR的秩揭示能力有限。强秩揭示QR（Rank-Revealing QR, RRQR）通过后处理改进。

**定义7**（RRQR条件）：QR分解$\boldsymbol{A}\boldsymbol{\Pi}=\boldsymbol{Q}\boldsymbol{R}$是$(f,g)$-RRQR，如果存在常数$f,g$使得
$$\sigma_i(\boldsymbol{R}_{11}) \leq f\sigma_i(\boldsymbol{A}), \quad i=1,\cdots,r$$
$$\sigma_i(\boldsymbol{R}_{22}) \geq g\sigma_{r+i}(\boldsymbol{A}), \quad i=1,\cdots,m-r$$
其中$\boldsymbol{R} = \begin{pmatrix}\boldsymbol{R}_{11} & \boldsymbol{R}_{12} \\ 0 & \boldsymbol{R}_{22}\end{pmatrix}$。

**Gu-Eisenstat RRQR算法**：
1. 执行标准CPQR得到初始$\boldsymbol{Q},\boldsymbol{R},\boldsymbol{\Pi}$
2. while 未收敛:
   - 检查$\boldsymbol{R}_{12}$中的大元素
   - 如果存在$(i,j)$使得$|R_{ij}| > \tau\cdot|R_{ii}|$（$\tau$是阈值），交换列$i$和$r+j$
   - 更新QR分解
3. 返回$\boldsymbol{Q},\boldsymbol{R},\boldsymbol{\Pi}$

**定理10**（RRQR的近似性能）：使用RRQR选出的前$r$列构造的近似满足
$$\|\boldsymbol{A} - \boldsymbol{A}_{[:,S]}\boldsymbol{A}_{[:,S]}^{\dagger}\boldsymbol{A}\|_F^2 \leq (1+f^2r(m-r))\|\boldsymbol{A} - \boldsymbol{A}_r^{\text{SVD}}\|_F^2$$

对于$f=2$，这给出了可控的近似比。

**复杂度**：RRQR的复杂度为$\mathcal{O}(nmr)$ + 迭代次数$\times$ $\mathcal{O}(nmr)$。实际中迭代次数很少（通常<5）。

### 10. 计算复杂度优势分析

现在我们系统地比较不同方法的计算复杂度。

**方法对比表**：

| 方法 | 列选择复杂度 | 近似构造复杂度 | 总复杂度 | 误差保证 |
|------|------------|--------------|---------|---------|
| 截断SVD | $\mathcal{O}(\min(nm^2,n^2m))$ | $\mathcal{O}(nmr)$ | $\mathcal{O}(\min(nm^2,n^2m))$ | 最优 |
| 模长排序 | $\mathcal{O}(l)$ | $\mathcal{O}(rmn)$ | $\mathcal{O}(l+rmn)$ | 无保证 |
| 随机采样 | $\mathcal{O}(l)$ | $\mathcal{O}(rmn)$ | $\mathcal{O}(l+rmn)$ | 期望界 |
| DPP采样 | $\mathcal{O}(l^3)$ | $\mathcal{O}(rmn)$ | $\mathcal{O}(l^3+rmn)$ | 相对误差 |
| CPQR | $\mathcal{O}(lmn)$ | 已包含 | $\mathcal{O}(lmn)$ | 弱保证 |
| RRQR | $\mathcal{O}(lmn)$ | 已包含 | $\mathcal{O}(lmn)$ | 强保证 |

**详细分析**：

1. **截断SVD**：
   - 对于矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$（$n\geq m$），标准SVD复杂度为$\mathcal{O}(nm^2)$
   - 如果只需前$r$个奇异值/向量，可用Lanczos或随机化SVD降到$\mathcal{O}(nmr)$
   - 优点：最优误差
   - 缺点：无CR结构、不可解释、对非线性映射不友好

2. **模长排序/采样**：
   - 计算$\|\boldsymbol{x}_i\|\|\boldsymbol{y}_i\|$需要$\mathcal{O}(n+m)$每列，总计$\mathcal{O}(l(n+m))$
   - 排序/采样$\mathcal{O}(l\log l)$或$\mathcal{O}(l)$
   - 构造$\boldsymbol{C}\boldsymbol{R}$：$\mathcal{O}(rmn)$（矩阵乘法）
   - 优点：极快、易实现
   - 缺点：误差无理论保证、可能选到冗余列

3. **DPP采样**：
   - 计算核$\boldsymbol{L}$：需要$\boldsymbol{V}^{\top}\boldsymbol{V}$，若直接做是$\mathcal{O}(l^2nm)$
   - 可以通过$\boldsymbol{L}_{ij} = \boldsymbol{x}_i^{\top}\boldsymbol{x}_j\cdot\boldsymbol{y}_i^{\top}\boldsymbol{y}_j$降到$\mathcal{O}(l^2(n+m))$
   - 特征分解$\boldsymbol{L}$：$\mathcal{O}(l^3)$
   - DPP采样：$\mathcal{O}(lr^2)$
   - 优点：理论保证强、自动多样性
   - 缺点：$l$很大时$\mathcal{O}(l^3)$不可承受

4. **CPQR/RRQR**：
   - 对$\boldsymbol{X}\boldsymbol{Y}$做QR：需要先形成乘积，$\mathcal{O}(lmn)$
   - QR分解：$\mathcal{O}(m^2n)$或$\mathcal{O}(n^2m)$（取决于形状）
   - 对于CR分解，可以在不显式形成$\boldsymbol{X}\boldsymbol{Y}$的情况下做CPQR，这需要隐式矩阵-向量乘法
   - 优点：确定性、数值稳定
   - 缺点：比简单采样慢

**大规模场景（$l,m,n$都很大）**：
- 如果$l$不太大（$l<10^4$）：优先DPP或RRQR
- 如果$l$很大（$l>10^6$）：只能用简单采样
- 如果需要在线更新：随机采样是唯一选择
- 如果$r\ll l$且需要高精度：随机化SVD + RRQR后处理

### 11. 在大规模矩阵中的应用

**应用1：加速矩阵乘法**
给定$\boldsymbol{X}\in\mathbb{R}^{n\times l}$，$\boldsymbol{Y}\in\mathbb{R}^{l\times m}$，直接计算$\boldsymbol{X}\boldsymbol{Y}$需要$\mathcal{O}(lmn)$。

使用CR近似：
$$\boldsymbol{X}\boldsymbol{Y} \approx \boldsymbol{X}_{[:,S]}\boldsymbol{\Lambda}\boldsymbol{Y}_{[S,:]}$$

计算步骤：
1. 采样列/行（$\mathcal{O}(l)$）
2. 计算$\boldsymbol{C}=\boldsymbol{X}_{[:,S]}$（已有）
3. 计算$\boldsymbol{R}=\boldsymbol{Y}_{[S,:]}$（已有）
4. 计算$\boldsymbol{\Lambda}$（$\mathcal{O}(r^2(n+m))$）
5. 计算$\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$（$\mathcal{O}(rmn + r^2n + r^2m)$）

总复杂度：$\mathcal{O}(rmn)$，当$r\ll l$时显著加速。

**误差分析**：设真实乘积为$\boldsymbol{M}=\boldsymbol{X}\boldsymbol{Y}$，近似为$\hat{\boldsymbol{M}}$，则相对误差
$$\epsilon = \frac{\|\hat{\boldsymbol{M}} - \boldsymbol{M}\|_F}{\|\boldsymbol{M}\|_F}$$

根据定理4，期望相对误差满足
$$\mathbb{E}[\epsilon^2] = \frac{1}{r}\left[\frac{(\sum_i\|\boldsymbol{v}_i\|)^2}{\|\boldsymbol{M}\|_F^2} - 1\right]$$

定义"相干性"指标
$$\mu = \frac{(\sum_i\|\boldsymbol{v}_i\|)^2}{l\|\boldsymbol{M}\|_F^2}$$

则$\mathbb{E}[\epsilon^2] = \frac{\mu l - 1}{r}$。

当$\mu\approx 1$（列-行对近似正交）时，只需$r=\mathcal{O}(l/\epsilon^2)$即可达到误差$\epsilon$。

**应用2：推荐系统中的矩阵补全**
用户-物品评分矩阵$\boldsymbol{R}\in\mathbb{R}^{n\times m}$（$n$个用户，$m$个物品），通常是低秩的。

使用CR分解$\boldsymbol{R}\approx\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}_{\text{rows}}$：
- $\boldsymbol{C}$：选出的代表性用户组
- $\boldsymbol{R}_{\text{rows}}$：选出的代表性物品组
- $\boldsymbol{\Lambda}$：用户组-物品组的交互矩阵

优点：
1. 可解释性：可以识别"原型用户"和"原型物品"
2. 冷启动：新用户只需与代表性物品比较
3. 实时推荐：只需计算$r$维向量内积

**应用3：神经网络压缩**
全连接层的权重矩阵$\boldsymbol{W}\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$可以用CR近似：
$$\boldsymbol{W} \approx \boldsymbol{W}_{[:,S]}\boldsymbol{\Lambda}\boldsymbol{W}_{[S,:]}$$

这相当于在输入-输出之间插入一个$r$维的"瓶颈层"，且瓶颈层的基是从原始权重中选出的。

前向传播：
$$\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x} \approx \boldsymbol{W}_{[:,S]}(\boldsymbol{\Lambda}(\boldsymbol{W}_{[S,:]}\boldsymbol{x}))$$

计算量从$\mathcal{O}(d_{\text{in}}d_{\text{out}})$降到$\mathcal{O}(r(d_{\text{in}}+d_{\text{out}}))$。

与Tucker分解等方法相比，CR的优势是保留了原始神经元的语义。

**应用4：大规模核方法**
核矩阵$\boldsymbol{K}\in\mathbb{R}^{n\times n}$，$K_{ij}=k(\boldsymbol{x}_i,\boldsymbol{x}_j)$，通常$n$很大（$>10^6$）。

Nyström方法：选择$r$个"地标点"$S$，近似
$$\boldsymbol{K} \approx \boldsymbol{K}_{[:,S]}\boldsymbol{K}_{S,S}^{-1}\boldsymbol{K}_{[S,:]}$$

这正是CR分解的一个特例（$\boldsymbol{\Lambda}=\boldsymbol{K}_{S,S}^{-1}$）。

使用leverage score采样选择地标点可以保证
$$\|\boldsymbol{K} - \boldsymbol{K}_{[:,S]}\boldsymbol{K}_{S,S}^{-1}\boldsymbol{K}_{[S,:]} \|_F \leq \epsilon\|\boldsymbol{K}\|_F$$
只需$r=\mathcal{O}(k/\epsilon^2)$个地标点（$k$是有效秩）。

复杂度：
- 构造近似：$\mathcal{O}(nr^2 + r^3)$
- 矩阵向量乘法：从$\mathcal{O}(n^2)$降到$\mathcal{O}(nr)$

**应用5：张量分解的初始化**
高阶张量$\mathcal{T}\in\mathbb{R}^{n_1\times n_2\times n_3}$的CP分解需要良好初始化。

将张量展开为矩阵$\boldsymbol{T}^{(1)}\in\mathbb{R}^{n_1\times n_2n_3}$，应用CR分解选择代表性"纤维"，然后在低维空间中初始化CP分解。

这比随机初始化收敛更快，且避免局部极小值。

### 12. 加权CR分解的最优权重

回到加权CR分解$\boldsymbol{M}\approx\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$，我们已经知道当$\boldsymbol{C},\boldsymbol{R}$固定时，
$$\boldsymbol{\Lambda}^* = \boldsymbol{C}^{\dagger}\boldsymbol{M}\boldsymbol{R}^{\dagger}$$

现在推导对角$\boldsymbol{\Lambda}$的情况。

**定理11**（对角权重的最优解）：设$\boldsymbol{C},\boldsymbol{R}$已固定，求解
$$\min_{\boldsymbol{\Lambda}=\text{diag}(\lambda_1,\cdots,\lambda_r)}\|\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R} - \boldsymbol{M}\|_F^2$$

令$\boldsymbol{c}_i$为$\boldsymbol{C}$的第$i$列，$\boldsymbol{r}_i$为$\boldsymbol{R}$的第$i$行，则最优解满足：
$$\lambda_i^* = \frac{\langle\boldsymbol{c}_i\boldsymbol{r}_i, \boldsymbol{M}\rangle_F}{\|\boldsymbol{c}_i\|^2\|\boldsymbol{r}_i\|^2}$$

证明：目标函数可以写成
\begin{align}
\|\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R} - \boldsymbol{M}\|_F^2 &= \left\|\sum_{i=1}^r \lambda_i\boldsymbol{c}_i\boldsymbol{r}_i - \boldsymbol{M}\right\|_F^2 \\
&= \sum_{i=1}^r\lambda_i^2\|\boldsymbol{c}_i\|^2\|\boldsymbol{r}_i\|^2 - 2\sum_{i=1}^r\lambda_i\langle\boldsymbol{c}_i\boldsymbol{r}_i,\boldsymbol{M}\rangle_F + \|\boldsymbol{M}\|_F^2
\end{align}

这是关于$\lambda_i$的二次函数（假设$\boldsymbol{c}_i\boldsymbol{r}_i$相互正交，否则需要联合优化）。对$\lambda_i$求导：
$$\frac{\partial}{\partial\lambda_i}\|\cdots\|_F^2 = 2\lambda_i\|\boldsymbol{c}_i\|^2\|\boldsymbol{r}_i\|^2 - 2\langle\boldsymbol{c}_i\boldsymbol{r}_i,\boldsymbol{M}\rangle_F$$

令其为零得到$\lambda_i^*$。

**推论2**：对于采样方法，$\lambda_i=(rp_i)^{-1}$对应的权重正是使得估计无偏的选择。

### 13. 实用算法总结

基于以上理论，我们总结实用的CR分解算法：

**算法1：快速CR近似（基于模长）**
```
输入：X ∈ R^{n×l}, Y ∈ R^{l×m}, 秩 r
输出：C, Λ, R

1. 计算 w_i = ||x_i|| · ||y_i|| for i=1,...,l
2. 选择 S = top-r indices by w
3. C = X[:, S], R = Y[S, :]
4. 计算 Λ = diag(λ_1,...,λ_r) 其中
   λ_i = <c_i r_i, XY>_F / (||c_i||^2 ||r_i||^2)
5. 返回 C, Λ, R
```
复杂度：$\mathcal{O}(l(n+m) + lmn + r(nm))$

**算法2：随机CR近似（leverage score采样）**
```
输入：X, Y, r, 采样数 s > r
输出：C, Λ, R

1. 计算 p_i = ||x_i|| · ||y_i|| / Σ_j ||x_j|| · ||y_j||
2. 采样 S = {s_1,...,s_s} ~ p (有放回)
3. C = X[:, S] · diag(1/√(s·p_{s_i}))
4. R = diag(1/√(s·p_{s_i})) · Y[S, :]
5. (可选) 正交化: C ← orth(C)
6. 计算 Λ = C^† · XY · R^†
7. 返回 C, Λ, R
```
复杂度：$\mathcal{O}(l(n+m) + s(nm))$

**算法3：RRQR-based CR近似**
```
输入：X, Y, r
输出：C, R, Π

1. M = XY  (或使用隐式矩阵-向量乘法)
2. [Q, R, Π] = rrqr(M, r)  (RRQR分解)
3. C = M[:, Π[:r]]
4. R = C^† · M
5. 返回 C, R
```
复杂度：$\mathcal{O}(lmn + nmr)$

这三个算法代表了速度-精度的权衡：
- 算法1最快但精度最低
- 算法2中等速度、有理论保证
- 算法3较慢但精度最高、数值最稳定

### 14. 数值稳定性考虑

在实际实现中，数值稳定性至关重要。

**问题1：伪逆计算**
计算$\boldsymbol{C}^{\dagger}$时，如果$\boldsymbol{C}$接近秩亏，直接用$(\boldsymbol{C}^{\top}\boldsymbol{C})^{-1}\boldsymbol{C}^{\top}$会数值不稳定。

**解决方案**：使用SVD计算伪逆：
$$\boldsymbol{C} = \boldsymbol{U}_C\boldsymbol{\Sigma}_C\boldsymbol{V}_C^{\top}$$
$$\boldsymbol{C}^{\dagger} = \boldsymbol{V}_C\boldsymbol{\Sigma}_C^{-1}\boldsymbol{U}_C^{\top}$$
其中$\boldsymbol{\Sigma}_C^{-1}$中对于$\sigma_i < \epsilon$的奇异值，设$1/\sigma_i=0$。

**问题2：重复采样**
独立重复采样可能选到同一列多次，导致$\boldsymbol{C}$秩亏。

**解决方案1**：使用无放回采样（但理论分析变复杂）
**解决方案2**：过采样，即采样$s=\omega(r)$个列，然后用RRQR选出其中最好的$r$个
**解决方案3**：采样后正交化

**问题3：权重尺度**
当采样概率$p_i$很小时，权重$1/p_i$很大，导致数值不稳定。

**解决方案**：归一化权重，或使用对数空间计算。

**推荐实现**：
```python
import numpy as np
from scipy.linalg import qr, svd

def stable_pinv(C, tol=1e-10):
    U, s, Vt = svd(C, full_matrices=False)
    s_inv = np.where(s > tol, 1/s, 0)
    return Vt.T @ np.diag(s_inv) @ U.T

def cr_decomp_stable(X, Y, r):
    # 计算采样概率
    w = np.linalg.norm(X, axis=0) * np.linalg.norm(Y, axis=1)
    p = w / w.sum()

    # 过采样
    s = min(3 * r, len(p))
    S = np.random.choice(len(p), s, replace=False, p=p)

    # 构造加权矩阵
    C = X[:, S] / np.sqrt(s * p[S])
    R = Y[S, :] / np.sqrt(s * p[S])[:, None]

    # 正交化并降秩
    Q, _ = qr(C)
    C = Q[:, :r]

    # 稳定计算权重
    M = X @ Y
    Lambda = stable_pinv(C) @ M @ stable_pinv(R.T).T

    return C, Lambda, R[:r, :]
```

### 15. 理论与实践的差距

最后，我们讨论理论结果与实际性能的差距。

**理论假设 vs 实际情况**：

1. **独立采样 vs 无放回采样**
   - 理论：独立重复采样便于分析
   - 实际：无放回采样性能更好（无重复、更多样性）
   - 差距：实际误差通常小于理论上界

2. **最坏情况 vs 平均情况**
   - 理论界：针对最坏情况矩阵
   - 实际矩阵：通常有更好的结构（如快速衰减的奇异值）
   - 差距：实际中$r$远小于理论要求

3. **Frobenius范数 vs 谱范数**
   - 理论：Frobenius范数界更紧
   - 实际：有时更关心谱范数或元素级误差
   - 差距：不同范数下的性能可能差异很大

4. **固定秩 vs 自适应秩**
   - 理论：假设$r$已知
   - 实际：需要自适应选择$r$（如通过交叉验证）
   - 方法：绘制误差-秩曲线，选择拐点

**改进方向**：

1. **混合方法**：结合多种采样策略（如先leverage score粗选，再DPP精选）
2. **迭代refinement**：初始CR分解后，迭代更新$\boldsymbol{C},\boldsymbol{R},\boldsymbol{\Lambda}$
3. **自适应采样**：根据当前残差调整采样分布
4. **分层采样**：对不同奇异值区间使用不同策略

**实验建议**：

对于新应用，建议按以下顺序尝试：
1. 模长排序（baseline，最快）
2. 随机leverage score采样（理论保证，较快）
3. RRQR（确定性，高精度）
4. DPP采样（如果$l$不太大且需要多样性）
5. 混合/迭代方法（如果上述方法不够好）

同时跟踪多个指标：
- 计算时间
- 近似误差（Frobenius范数、谱范数、相对误差）
- 列的多样性（如条件数、最小奇异值）
- 任务性能（如果用于下游任务）

通过这些详细推导，我们建立了CR分解从理论到实践的完整图景，涵盖了算法设计、误差分析、复杂度优化和数值稳定性等多个方面。CR分解作为一种兼顾效率与可解释性的低秩近似方法，在大规模数据处理中有广阔的应用前景。

