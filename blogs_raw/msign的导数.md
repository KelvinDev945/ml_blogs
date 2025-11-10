---
title: msign的导数
slug: msign的导数
date: 2025-06-13
tags: 微积分, 矩阵, 梯度, muon, 生成模型
status: pending
---

# msign的导数

**原文链接**: [https://spaces.ac.cn/archives/11025](https://spaces.ac.cn/archives/11025)

**发布日期**: 

---

这篇文章我们来推导$\newcommand{msign}{\mathop{\text{msign}}}\msign$算子的求导公式。如果读者想要像[《Test-Time Training Done Right》](https://papers.cool/arxiv/2505.23884)一样，将[TTT](https://papers.cool/arxiv/2407.04620)和[Muon](/archives/10592)结合起来，那么本文可能会对你有帮助。

## 两种定义 #

本文依然假设大家已经对$\msign$有所了解，如果还没有，可以先移步阅读[《Muon优化器赏析：从向量到矩阵的本质跨越》](/archives/10592)和[《msign算子的Newton-Schulz迭代（上）》](/archives/10922)。现设有矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，那么  
\begin{equation}\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}^{\top} = \text{SVD}(\boldsymbol{M}) \quad\Rightarrow\quad \msign(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}\end{equation}  
其中$\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m},\boldsymbol{V}\in\mathbb{R}^{m\times m}$，$r$是$\boldsymbol{M}$的秩。简单来说，$\msign$就是把矩阵的所有非零奇异值都变成1后所得的新矩阵。基于SVD，我们还可以证明  
\begin{equation}\msign(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}\end{equation}  
这里的$^{-1/2}$的矩阵的$1/2$次幂的逆，由于$\boldsymbol{M}\boldsymbol{M}^{\top}$和$\boldsymbol{M}^{\top}\boldsymbol{M}$是（半）正定对称的，所以$1/2$次幂总是可求，但逆未必，不可逆的时候我们可以用“[伪逆](/archives/10366)”。$\msign$这个名字，源于上式与实数符号函数$\newcommand{sign}{\mathop{\text{sign}}}\sign(x) = x/\sqrt{x^2}$的相似性。然而，我们之前也提到过，符号函数还有[另一个矩阵版](https://en.wikipedia.org/wiki/Matrix_sign_function)，这里称为$\newcommand{mcsgn}{\mathop{\text{mcsgn}}}\newcommand{csgn}{\mathop{\text{csgn}}}\mcsgn$：  
\begin{equation}\mcsgn(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}\end{equation}  
即$\msign$的$\boldsymbol{M}^{\top}\boldsymbol{M}$换成了$\boldsymbol{M}^2$。由于只有方阵才能算平方，所以这种定义只适用于方阵。在一篇文章内引入两种相似但不同的定义是容易引起混淆的，但很不幸，后面的计算中两种定义都需要用到，所以不得不同时出现。

$\mcsgn$具备相似不变性，如果$\boldsymbol{M}=\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，那么$\mcsgn(\boldsymbol{M})=\boldsymbol{P}\mcsgn(\boldsymbol{\Lambda})\boldsymbol{P}^{-1}$。进一步地，如果$\boldsymbol{\Lambda}$是对角阵（在复数域内几乎总是可以做到），那么有  
\begin{equation}\mcsgn(\boldsymbol{M}) = \boldsymbol{P}\csgn(\boldsymbol{\Lambda})\boldsymbol{P}^{-1}\end{equation}  
$\csgn(\boldsymbol{\Lambda})$表示对角线的元素都取$\csgn$，其中$\csgn(z) = z/\sqrt{z^2}$是符号函数的复数版，如果$z$的实部非零那么它等于$\sign(\mathop{\text{Re}}[z])$。这样看来，$\msign$和$\mcsgn$的区别在于，前者是在奇异值分解基础上对奇异值取符号函数，后者是在特征值分解基础上对特征值取符号函数。当$\boldsymbol{M}$是对称矩阵时，它们是相等的。

## 同一计算 #

目前而言，$\msign$的数值计算主要靠如下格式的“Newton-Schulz迭代”：  
\begin{equation}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\Vert\boldsymbol{M}\Vert_F},\qquad \boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c_{t+1}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2\end{equation}  
至于系数的选择，这我们在[《msign算子的Newton-Schulz迭代（上）》](/archives/10922)和[《msign算子的Newton-Schulz迭代（下）》](/archives/10996)已经详细探讨过了，其中出自下篇的比较新的结果是：  
$$\begin{array}{c|ccc}  
\hline  
t & a\times 1.01 & b\times 1.01^3 & c\times 1.01^5 \\\  
\hline  
\quad 1\quad & 8.28721 & -23.5959 & 17.3004 \\\  
2 & 4.10706 & -2.94785 & 0.544843 \\\  
3 & 3.94869 & -2.9089 & 0.551819 \\\  
4 & 3.31842 & -2.48849 & 0.510049 \\\  
5 & 2.30065 & -1.6689 & 0.418807 \\\  
6 & 1.8913 & 1.268 & 0.376804 \\\  
7 & 1.875 & -1.25 & 0.375 \\\  
8 & 1.875 & -1.25 & 0.375 \\\  
\hline  
\end{array}$$  
这个结果的好处是可以任意截断和叠加，比如只保留前5行它就是最佳的5步迭代，保留前6行就是最佳6步迭代，并且近似程度是有保证地优于5步迭代，依此类推。

至于$\mcsgn$，它只是把$\msign$的$\boldsymbol{M}^{\top}\boldsymbol{M}$换成了$\boldsymbol{M}^2$，所以理论上也可以用Newton-Schulz迭代，但由于特征值可以是复数，因此一般的收敛会困难得多。不过，如果我们可以实现确认矩阵$\boldsymbol{M}$的特征值都是实数（比如本文后面要应用$\mcsgn$的分块三角矩阵），那么就可以复用$\msign$的迭代和系数：  
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\sqrt{\tr(\boldsymbol{M}^2)}},\qquad \boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t^3 + c_{t+1}\boldsymbol{X}_t^5\end{equation}

## 推导过程 #

下面正式进入主题——求$\boldsymbol{O}=\msign(\boldsymbol{M})$的导数。如果读者只是将Muon当成一个普通优化器用，那么本文多半跟你无关了。当我们需要参考TTT，用Muon优化器来构建RNN模型时，才需要$\msign$的导数，此时$\msign$在模型表现为前向传播，而要对整个模型反向传播，自然就涉及到了$\msign$的导数。

由于$\msign$是通过Newton-Schulz迭代计算的，实际上它可以直接反向传播，所以$\msign$的数值求导本身不是问题，但基于迭代反传意味着有很多中间状态要存，显存往往要爆炸，所以希望能得到导数的解析解来简化。另一方面，在[《SVD的导数》](/archives/10878)中我们其实也求过$\msign$的导数，但那是基于SVD的表达式，而SVD并不是GPU高效的算法。

所以，我们的目的是寻求一个不依赖于SVD的、能够高效计算的结果。我们从恒等式  
\begin{equation}\boldsymbol{M} = \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{O}\end{equation}  
出发（由$\msign$的定义即可证明），两边微分得到  
\begin{equation}d\boldsymbol{M} = (d\boldsymbol{O})\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}(d\boldsymbol{M}^{\top})\boldsymbol{O} + \boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})\label{eq:dm-do}\end{equation}  
这个结果的难度在于无法简单地分离出$d\boldsymbol{M}=f(d\boldsymbol{O})$或$d\boldsymbol{O}=f(d\boldsymbol{M})$的形式，因此不大好看出$\nabla_{\boldsymbol{O}}\mathcal{L}$与$\nabla_{\boldsymbol{W}}\mathcal{L}$的关系（$\mathcal{L}$是损失函数）。这种情况下最好的办法是回归到矩阵求导的根本思路——“迹技巧”：

> **迹技巧（trace trick）** 如果我们能找到跟$\boldsymbol{M}$同形状的矩阵$\boldsymbol{G}$满足 \begin{equation}d\mathcal{L}=\langle \boldsymbol{G}, d\boldsymbol{M}\rangle_F = \tr(\boldsymbol{G}^{\top} (d\boldsymbol{M}))\end{equation} 那么$\boldsymbol{G} = \nabla_{\boldsymbol{M}}\mathcal{L}$。

迹技巧的要义是化矩阵/向量为标量，然后化标量为迹，继而可以利用迹的恒等式：  
\begin{equation}\tr(\boldsymbol{A}\boldsymbol{B}) = \tr(\boldsymbol{B}\boldsymbol{A}) = \tr(\boldsymbol{A}^{\top}\boldsymbol{B}^{\top}) = \tr(\boldsymbol{B}^{\top}\boldsymbol{A}^{\top})\end{equation}  
现在设$\boldsymbol{X}$是任意跟$\boldsymbol{M}$同形状矩阵，给式$\eqref{eq:dm-do}$两边乘$\boldsymbol{X}^{\top}$，然后求迹  
\begin{equation}\begin{aligned}  
\tr(\boldsymbol{X}^{\top}(d\boldsymbol{M})) =&\, \tr(\boldsymbol{X}^{\top}(d\boldsymbol{O})\boldsymbol{M}^{\top}\boldsymbol{O}) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}(d\boldsymbol{M}^{\top})\boldsymbol{O}) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})) \\\\[7pt]  
=&\, \tr(\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top}(d\boldsymbol{O})) + \tr(\boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O}(d\boldsymbol{M}^{\top})) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})) \\\\[7pt]  
=&\, \tr(\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top}(d\boldsymbol{O})) + \tr(\boldsymbol{O}^{\top}\boldsymbol{X}\boldsymbol{O}^{\top}(d\boldsymbol{M})) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})) \\\\[7pt]  
\end{aligned}\end{equation}  
由此可得  
\begin{equation}\tr((\boldsymbol{X}^{\top} - \boldsymbol{O}^{\top}\boldsymbol{X}\boldsymbol{O}^{\top})(d\boldsymbol{M})) = \tr((\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top} + \boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top})(d\boldsymbol{O}))\end{equation}  
如果我们让$\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top} + \boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}=(\nabla_{\boldsymbol{O}}\mathcal{L})^{\top}$，那么上式便具有$d\mathcal{L}$的含义，那么根据迹技巧就有$\boldsymbol{X}^{\top} - \boldsymbol{O}^{\top}\boldsymbol{X}\boldsymbol{O}^{\top}=(\nabla_{\boldsymbol{M}}\mathcal{L})^{\top}$，这表明$\nabla_{\boldsymbol{M}}\mathcal{L}$和$\nabla_{\boldsymbol{O}}\mathcal{L}$的关系，由下述方程组描述  
\begin{gather}\boldsymbol{X} - \boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O} = \nabla_{\boldsymbol{M}}\mathcal{L} \label{eq:g-m}\\\\[7pt]  
\boldsymbol{X}\boldsymbol{O}^{\top}\boldsymbol{M} + \boldsymbol{M}\boldsymbol{O}^{\top}\boldsymbol{X} = \nabla_{\boldsymbol{O}}\mathcal{L}\label{eq:g-o}\end{gather}

## 理论形式 #

所以，现在问题变成了，从式$\eqref{eq:g-o}$中解出$\boldsymbol{X}$，然后代入式$\eqref{eq:g-m}$得到$\nabla_{\boldsymbol{M}}\mathcal{L}$，即将$\nabla_{\boldsymbol{M}}\mathcal{L}$表示为$\nabla_{\boldsymbol{O}}\mathcal{L}$的函数，避免直接求$\nabla_{\boldsymbol{M}}\boldsymbol{O}$。很明显，唯一的难度就是方程$\eqref{eq:g-o}$的求解。

这一节我们先基于SVD求一个不那么实用的理论解，它可以帮助我们了解方程$\eqref{eq:g-o}$的性质，并且跟之前的结果对齐。设$\boldsymbol{X}=\boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top}$，然后我们还有$\boldsymbol{O}^{\top}\boldsymbol{M} = (\boldsymbol{M}^{\top}\boldsymbol{M})^{1/2} = \boldsymbol{V}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})^{1/2}\boldsymbol{V}^{\top}$和$\boldsymbol{M}\boldsymbol{O}^{\top}=(\boldsymbol{M}\boldsymbol{M}^{\top})^{1/2} = \boldsymbol{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})^{1/2}\boldsymbol{U}^{\top}$，将这些等式代入方程$\eqref{eq:g-o}$得到  
\begin{equation}\boldsymbol{U}\boldsymbol{Y}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})^{1/2}\boldsymbol{V}^{\top} + \boldsymbol{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})^{1/2}\boldsymbol{Y}\boldsymbol{V}^{\top} = \nabla_{\boldsymbol{O}}\mathcal{L}\end{equation}  
即  
\begin{equation}\boldsymbol{Y}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})^{1/2} + (\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})^{1/2}\boldsymbol{Y} = \boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}\label{eq:g-o-2}\end{equation}  
上式左端如果写成分量形式是$\boldsymbol{Y}_{i,j}\sigma_j + \sigma_i \boldsymbol{Y}_{i,j} = (\sigma_i + \sigma_j)\boldsymbol{Y}_{i,j}$，其中$\sigma_1,\sigma_2,\cdots,\sigma_r$是$\boldsymbol{M}$的非零奇异值，而$0=\sigma_{r+1}=\sigma_{r+2}=\cdots$。很明显，如果当$\boldsymbol{M}$是满秩方阵时，可以解得  
\begin{equation}\boldsymbol{Y} = (\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}) \oslash \boldsymbol{S}\end{equation}  
其中$\boldsymbol{S}_{i,j} = \sigma_i+\sigma_j$，$\oslash$是Hadamard除（逐位相除）。这时候我们将$\boldsymbol{X}=\boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top}$代入式$\eqref{eq:g-m}$，就得到跟[《SVD的导数》](/archives/10878)里边一致的结果。这个殊途同归也增强了我们的信心，看来至少到目前为止我们的推导都还是正确的。

若$\boldsymbol{M}$不满秩或不是方阵呢？此时如果右端的$\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}$“不配合”，那么方程$\eqref{eq:g-o-2}$无解。但方程$\eqref{eq:g-o-2}$是从实际问题得到的，所以它肯定有解，那么右端“必须配合”！怎样才是配合呢？如果$\boldsymbol{M}$的秩为$r$，那么矩阵$\boldsymbol{S}$只有$\boldsymbol{S}_{[:r,:r]}$是非零的，为了使得方程$\eqref{eq:g-o-2}$有解，$(\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V})_{[:r,:r]}$以外的部分只能是零。在这个条件下，我们可以写出  
\begin{equation}\boldsymbol{Y} = \lim_{\epsilon\to 0}\,\, (\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}) \oslash (\boldsymbol{S} + \epsilon) \end{equation}  
这相当于说，我们可以给奇异值加些扰动，转化为全体奇异值非零的情况，计算完成后再让扰动趋于零，从而得到正确的结果。

## 高效求解 #

上一节的SVD解往往只有理论价值，为了在GPU中高效计算，我们还需要寻求其他形式的解。引入记号$\boldsymbol{M}\boldsymbol{O}^{\top}=\boldsymbol{A},\boldsymbol{O}^{\top}\boldsymbol{M}=\boldsymbol{B},\nabla_{\boldsymbol{O}}\mathcal{L}=\boldsymbol{C}$，那么式$\eqref{eq:g-o}$实际上是一个[Sylvester方程](https://en.wikipedia.org/wiki/Sylvester_equation)：  
\begin{equation}\boldsymbol{A}\boldsymbol{X}+\boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}\end{equation}  
求解Sylvester方程的方法有很多，其中最精妙、对GPU最高效的，可能是基于$\mcsgn$（不是$\msign$）的求解方案（这里参考了[《Fast Differentiable Matrix Square Root》](https://papers.cool/arxiv/2201.08663)）。首先，从上述方程出发，我们可以验证下式成立  
\begin{equation}\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix} = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{A} & \boldsymbol{0} \\\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1}  
\end{equation}  
两边取$\mcsgn$，根据$\mcsgn$的性质，我们有  
\begin{equation}\mcsgn\left(\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\right) = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \mcsgn(\boldsymbol{A}) & \boldsymbol{0} \\\ \boldsymbol{0} & -\mcsgn(\boldsymbol{B})\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1}  
\end{equation}  
注意$\boldsymbol{A}=\boldsymbol{M}\boldsymbol{O}^{\top}=(\boldsymbol{M}\boldsymbol{M}^{\top})^{1/2}, \boldsymbol{B}=\boldsymbol{O}^{\top}\boldsymbol{M}=(\boldsymbol{M}^{\top}\boldsymbol{M})^{1/2}$，假设$\boldsymbol{M}$是满秩方阵，那么$\boldsymbol{A},\boldsymbol{B}$都是正定对称的，正定对称矩阵的$\mcsgn$都是方阵，所以  
\begin{equation}\mcsgn\left(\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\right) = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{0} \\\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1} = \begin{bmatrix} \boldsymbol{I} & -2\boldsymbol{X} \\\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}  
\end{equation}  
最后的化简用到了等式$\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1}=\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}$。从该结果可以看出，我们只需要对分块矩阵$\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}$算$\mcsgn$，然后就可以从结果的右上角读出$\boldsymbol{X}$。$\mcsgn$可以通过Newton-Schulz迭代高效地计算，因此该方案是GPU友好的。

当$\boldsymbol{M}$不满秩或非方阵时，$\boldsymbol{A},\boldsymbol{B}$只是半正定的，这时候它们就$\mcsgn$就不是$\boldsymbol{I}$。不过，上一节的经验告诉我们，由于$\nabla_{\boldsymbol{O}}\mathcal{L}$“必须配合”，所以只需要给$\boldsymbol{\Sigma}$加点扰动，让它变成正定的情况即可解。这里给$\boldsymbol{\Sigma}$加扰动，相当于给$\boldsymbol{A},\boldsymbol{B}$加$\epsilon \boldsymbol{I}$，所以  
\begin{equation}\boldsymbol{X} = -\frac{1}{2} \left(\lim_{\epsilon\to 0}\,\, \mcsgn\left(\begin{bmatrix} \boldsymbol{A} + \epsilon \boldsymbol{I} & -\boldsymbol{C} \\\ \boldsymbol{0} & -\boldsymbol{B} - \epsilon \boldsymbol{I}\end{bmatrix}\right)\right)_{[:n,n:]}  
\end{equation}  
实际计算时，就只能选择一个比较小的$\epsilon > 0$来近似计算了，可以考虑$\epsilon=10^{-3}$，它在我们之前寻找Newton-Schulz迭代的下界范围内。

## 详细数学推导 #

在本节中，我们将从多个角度深入探讨$\msign$算子的导数理论，包括Fréchet导数的严格定义、基于Lyapunov方程的推导、积分表示、数值计算方法以及在深度学习中的实际应用。

### Fréchet导数的定义与性质 ###

首先，我们需要严格定义矩阵函数的导数。对于从$\mathbb{R}^{n\times m}$到$\mathbb{R}^{p\times q}$的映射$F$，其**Fréchet导数**（或方向导数）定义如下：

**定义1（Fréchet导数）**：设$F: \mathbb{R}^{n\times m} \to \mathbb{R}^{p\times q}$是矩阵函数，如果存在线性算子$\mathcal{D}F(\boldsymbol{M}): \mathbb{R}^{n\times m} \to \mathbb{R}^{p\times q}$使得
\begin{equation}
F(\boldsymbol{M} + \boldsymbol{H}) = F(\boldsymbol{M}) + \mathcal{D}F(\boldsymbol{M})[\boldsymbol{H}] + o(\Vert\boldsymbol{H}\Vert_F)
\end{equation}
其中$\lim_{\Vert\boldsymbol{H}\Vert_F\to 0} \frac{\Vert o(\Vert\boldsymbol{H}\Vert_F)\Vert_F}{\Vert\boldsymbol{H}\Vert_F} = 0$，则称$\mathcal{D}F(\boldsymbol{M})$为$F$在$\boldsymbol{M}$处的Fréchet导数。

对于我们的情况，$F(\boldsymbol{M}) = \msign(\boldsymbol{M})$，我们需要找到$\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]$，即当$\boldsymbol{M}$有一个微小扰动$\boldsymbol{H}$时，$\msign(\boldsymbol{M})$的变化。

**性质1（链式法则）**：如果$F = G \circ H$，则
\begin{equation}
\mathcal{D}F(\boldsymbol{M})[\boldsymbol{H}] = \mathcal{D}G(H(\boldsymbol{M}))[\mathcal{D}H(\boldsymbol{M})[\boldsymbol{H}]]
\end{equation}

**性质2（线性性）**：Fréchet导数关于扰动$\boldsymbol{H}$是线性的，即
\begin{equation}
\mathcal{D}F(\boldsymbol{M})[\alpha\boldsymbol{H}_1 + \beta\boldsymbol{H}_2] = \alpha\mathcal{D}F(\boldsymbol{M})[\boldsymbol{H}_1] + \beta\mathcal{D}F(\boldsymbol{M})[\boldsymbol{H}_2]
\end{equation}

**性质3（乘积法则）**：如果$F(\boldsymbol{M}) = \boldsymbol{A}(\boldsymbol{M})\boldsymbol{B}(\boldsymbol{M})$，则
\begin{equation}
\mathcal{D}F(\boldsymbol{M})[\boldsymbol{H}] = \mathcal{D}\boldsymbol{A}(\boldsymbol{M})[\boldsymbol{H}]\cdot\boldsymbol{B}(\boldsymbol{M}) + \boldsymbol{A}(\boldsymbol{M})\cdot\mathcal{D}\boldsymbol{B}(\boldsymbol{M})[\boldsymbol{H}]
\end{equation}

### msign的Fréchet导数显式公式 ###

现在我们推导$\msign$的Fréchet导数。记$\boldsymbol{O} = \msign(\boldsymbol{M})$，从恒等式
\begin{equation}
\boldsymbol{M} = \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{O}
\end{equation}
出发，对两边取Fréchet导数（使用乘积法则）：
\begin{equation}
\boldsymbol{H} = \mathcal{D}\boldsymbol{O}[\boldsymbol{H}]\cdot\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}\cdot\boldsymbol{H}^{\top}\cdot\boldsymbol{O} + \boldsymbol{O}\cdot\boldsymbol{M}^{\top}\cdot\mathcal{D}\boldsymbol{O}[\boldsymbol{H}]
\end{equation}

记$d\boldsymbol{O} = \mathcal{D}\boldsymbol{O}[\boldsymbol{H}]$为$\boldsymbol{O}$在方向$\boldsymbol{H}$上的导数，我们有
\begin{equation}
\boldsymbol{H} = (d\boldsymbol{O})\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})
\end{equation}

定义算子$\mathcal{L}: \mathbb{R}^{n\times m} \to \mathbb{R}^{n\times m}$为
\begin{equation}
\mathcal{L}(\boldsymbol{X}) = \boldsymbol{X}\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{X}
\end{equation}

则我们的导数方程可以写成算子方程的形式：
\begin{equation}
\mathcal{L}(d\boldsymbol{O}) = \boldsymbol{H} - \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O}
\end{equation}

因此，$\msign$的Fréchet导数的**显式公式**为：
\begin{equation}
\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}] = \mathcal{L}^{-1}(\boldsymbol{H} - \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O})
\end{equation}

其中$\mathcal{L}^{-1}$是算子$\mathcal{L}$的逆。

### 基于Lyapunov方程的推导 ###

注意到算子$\mathcal{L}$定义的方程实际上是一个**连续时间Lyapunov方程**的特殊形式。标准的Lyapunov方程为：
\begin{equation}
\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}
\end{equation}

在我们的情况下，令$\boldsymbol{A} = \boldsymbol{M}\boldsymbol{O}^{\top}, \boldsymbol{B} = \boldsymbol{O}^{\top}\boldsymbol{M}$（注意这里是转置），$\boldsymbol{X} = d\boldsymbol{O}$，$\boldsymbol{C} = \boldsymbol{H} - \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O}$，我们可以将导数方程改写为：
\begin{equation}
\boldsymbol{M}\boldsymbol{O}^{\top}(d\boldsymbol{O}) + (d\boldsymbol{O})\boldsymbol{O}^{\top}\boldsymbol{M} = \boldsymbol{H} - \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O}
\end{equation}

这正是Sylvester方程的形式。我们知道$\boldsymbol{A} = \boldsymbol{M}\boldsymbol{O}^{\top} = (\boldsymbol{M}\boldsymbol{M}^{\top})^{1/2}$和$\boldsymbol{B} = \boldsymbol{O}^{\top}\boldsymbol{M} = (\boldsymbol{M}^{\top}\boldsymbol{M})^{1/2}$都是半正定矩阵。

**定理1（Lyapunov方程可解性）**：如果$\boldsymbol{A}$和$\boldsymbol{B}$的特征值满足$\lambda_i(\boldsymbol{A}) + \lambda_j(\boldsymbol{B}) \neq 0$对所有$i,j$成立，则Lyapunov方程$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}$有唯一解。

在我们的情况下，$\boldsymbol{A}$和$\boldsymbol{B}$的特征值都是非负的（它们是平方根矩阵的特征值），因此$\lambda_i(\boldsymbol{A}) + \lambda_j(\boldsymbol{B}) \geq 0$。只有当$\boldsymbol{M}$的某些奇异值为零时，和才可能为零。

**引理1**：当$\boldsymbol{M}$满秩时，Lyapunov方程有唯一解。当$\boldsymbol{M}$不满秩时，方程有解当且仅当$\boldsymbol{C}$满足相容性条件。

相容性条件可以通过SVD分析得到。设$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，秩为$r$，则
\begin{equation}
\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}^{1/2}_{ext}\boldsymbol{U}^{\top}, \quad \boldsymbol{B} = \boldsymbol{V}\boldsymbol{\Sigma}^{1/2}_{ext}\boldsymbol{V}^{\top}
\end{equation}
其中$\boldsymbol{\Sigma}_{ext}$是扩展的对角矩阵，保持前$r$个奇异值，其余为零。

将$d\boldsymbol{O} = \boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top}$代入Lyapunov方程，得到
\begin{equation}
\boldsymbol{U}\boldsymbol{\Sigma}^{1/2}_{ext}\boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top} + \boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top}\boldsymbol{V}\boldsymbol{\Sigma}^{1/2}_{ext}\boldsymbol{V}^{\top} = \boldsymbol{C}
\end{equation}

简化后得到
\begin{equation}
\boldsymbol{\Sigma}^{1/2}_{ext}\boldsymbol{Y} + \boldsymbol{Y}\boldsymbol{\Sigma}^{1/2}_{ext} = \boldsymbol{U}^{\top}\boldsymbol{C}\boldsymbol{V}
\end{equation}

写成分量形式：
\begin{equation}
(\sigma_i^{1/2} + \sigma_j^{1/2})\boldsymbol{Y}_{ij} = [\boldsymbol{U}^{\top}\boldsymbol{C}\boldsymbol{V}]_{ij}
\end{equation}

因此，解为
\begin{equation}
\boldsymbol{Y}_{ij} = \begin{cases}
\frac{[\boldsymbol{U}^{\top}\boldsymbol{C}\boldsymbol{V}]_{ij}}{\sigma_i^{1/2} + \sigma_j^{1/2}} & \text{if } i,j \leq r \\
0 & \text{otherwise}
\end{cases}
\end{equation}

相容性条件就是：当$i > r$或$j > r$时，$[\boldsymbol{U}^{\top}\boldsymbol{C}\boldsymbol{V}]_{ij} = 0$。

### 基于积分表示的推导 ###

除了代数方法，我们还可以从积分表示的角度理解$\msign$的导数。考虑矩阵符号函数的**Cauchy积分表示**：

对于可对角化的矩阵$\boldsymbol{M}$（特征值不在负实轴上），符号函数可以表示为
\begin{equation}
\text{sign}(\boldsymbol{M}) = \frac{1}{2\pi i}\oint_{\Gamma} \text{sign}(z)(\boldsymbol{M} - z\boldsymbol{I})^{-1}dz
\end{equation}

其中$\Gamma$是包围$\boldsymbol{M}$所有特征值的闭合路径。

对于$\msign$，我们有类似的表示。设$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，定义
\begin{equation}
\msign(\boldsymbol{M}) = \boldsymbol{U}\text{sign}(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}
\end{equation}

其中$\text{sign}(\boldsymbol{\Sigma})$对对角元素逐个应用符号函数（正数变1，负数变-1，零保持为零）。

更一般地，我们可以使用**谱分解**来表示$\msign$：
\begin{equation}
\msign(\boldsymbol{M}) = \sum_{i=1}^{r} \boldsymbol{u}_i\boldsymbol{v}_i^{\top}
\end{equation}

其中$\boldsymbol{u}_i, \boldsymbol{v}_i$是对应非零奇异值的左右奇异向量。

这个表示的导数可以通过对每一项求导得到：
\begin{equation}
\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}] = \sum_{i=1}^{r} \left(\mathcal{D}\boldsymbol{u}_i[\boldsymbol{H}]\boldsymbol{v}_i^{\top} + \boldsymbol{u}_i\mathcal{D}\boldsymbol{v}_i[\boldsymbol{H}]^{\top}\right)
\end{equation}

奇异向量的导数可以通过扰动理论得到。对于$\boldsymbol{M}\boldsymbol{v}_i = \sigma_i\boldsymbol{u}_i$，扰动后我们有
\begin{equation}
(\boldsymbol{M} + \boldsymbol{H})(\boldsymbol{v}_i + \delta\boldsymbol{v}_i) = (\sigma_i + \delta\sigma_i)(\boldsymbol{u}_i + \delta\boldsymbol{u}_i)
\end{equation}

忽略高阶项，得到
\begin{equation}
\boldsymbol{M}\delta\boldsymbol{v}_i + \boldsymbol{H}\boldsymbol{v}_i = \sigma_i\delta\boldsymbol{u}_i + \delta\sigma_i\boldsymbol{u}_i
\end{equation}

左乘$\boldsymbol{u}_i^{\top}$，利用$\boldsymbol{u}_i^{\top}\boldsymbol{M}\delta\boldsymbol{v}_i = \sigma_i\boldsymbol{u}_i^{\top}\delta\boldsymbol{v}_i$和正交性条件$\boldsymbol{u}_i^{\top}\delta\boldsymbol{u}_i = 0$，得到
\begin{equation}
\delta\sigma_i = \boldsymbol{u}_i^{\top}\boldsymbol{H}\boldsymbol{v}_i
\end{equation}

这给出了奇异值的一阶扰动公式。

对于奇异向量的扰动，我们有
\begin{equation}
\delta\boldsymbol{u}_i = \sum_{j\neq i}\frac{\boldsymbol{u}_j^{\top}\boldsymbol{H}\boldsymbol{v}_i}{\sigma_i - \sigma_j}\boldsymbol{u}_j
\end{equation}
\begin{equation}
\delta\boldsymbol{v}_i = \sum_{j\neq i}\frac{\boldsymbol{u}_i^{\top}\boldsymbol{H}\boldsymbol{v}_j}{\sigma_i - \sigma_j}\boldsymbol{v}_j
\end{equation}

但要注意，当奇异值重复时，这个公式会出现奇异性。对于$\msign$，由于所有非零奇异值都被映射到1，这种奇异性是不可避免的。

### 反向传播中的梯度计算 ###

在深度学习的反向传播中，我们通常不直接计算Fréchet导数，而是计算标量损失函数关于参数的梯度。设$\mathcal{L}$是标量损失函数，我们已知$\nabla_{\boldsymbol{O}}\mathcal{L}$（即损失对$\boldsymbol{O} = \msign(\boldsymbol{M})$的梯度），需要计算$\nabla_{\boldsymbol{M}}\mathcal{L}$。

根据链式法则，我们有
\begin{equation}
\langle \nabla_{\boldsymbol{M}}\mathcal{L}, \boldsymbol{H} \rangle = \langle \nabla_{\boldsymbol{O}}\mathcal{L}, \mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}] \rangle
\end{equation}

其中$\langle \cdot, \cdot \rangle$表示Frobenius内积。这可以改写为
\begin{equation}
\tr((\nabla_{\boldsymbol{M}}\mathcal{L})^{\top}\boldsymbol{H}) = \tr((\nabla_{\boldsymbol{O}}\mathcal{L})^{\top}\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}])
\end{equation}

从前面的推导，我们知道$\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]$满足
\begin{equation}
\boldsymbol{H} = \mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]\cdot\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{M}^{\top}\cdot\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]
\end{equation}

记$d\boldsymbol{O} = \mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]$，$\boldsymbol{G} = \nabla_{\boldsymbol{O}}\mathcal{L}$，我们需要求$\nabla_{\boldsymbol{M}}\mathcal{L}$使得
\begin{equation}
\tr((\nabla_{\boldsymbol{M}}\mathcal{L})^{\top}\boldsymbol{H}) = \tr(\boldsymbol{G}^{\top}d\boldsymbol{O})
\end{equation}

引入辅助矩阵$\boldsymbol{X}$满足
\begin{equation}
\boldsymbol{X}\boldsymbol{O}^{\top}\boldsymbol{M} + \boldsymbol{M}\boldsymbol{O}^{\top}\boldsymbol{X} = \boldsymbol{G}
\end{equation}

这也是一个Sylvester方程，与前向传播的方程形式相同。

将$\boldsymbol{H} = (d\boldsymbol{O})\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{H}^{\top}\boldsymbol{O} + \boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})$代入梯度计算，经过迹的循环性质变换，可以得到
\begin{equation}
\nabla_{\boldsymbol{M}}\mathcal{L} = \boldsymbol{X} - \boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O}
\end{equation}

**算法1（msign反向传播）**：
1. 输入：$\boldsymbol{M}, \boldsymbol{O} = \msign(\boldsymbol{M}), \boldsymbol{G} = \nabla_{\boldsymbol{O}}\mathcal{L}$
2. 计算$\boldsymbol{A} = \boldsymbol{M}\boldsymbol{O}^{\top}, \boldsymbol{B} = \boldsymbol{O}^{\top}\boldsymbol{M}$
3. 求解Sylvester方程：$\boldsymbol{X}\boldsymbol{B} + \boldsymbol{A}\boldsymbol{X} = \boldsymbol{G}$
4. 计算$\nabla_{\boldsymbol{M}}\mathcal{L} = \boldsymbol{X} - \boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O}$
5. 输出：$\nabla_{\boldsymbol{M}}\mathcal{L}$

### 基于mcsgn的高效求解算法 ###

前文提到，Sylvester方程可以通过$\mcsgn$高效求解。我们详细展开这个方法。

构造增广矩阵
\begin{equation}
\boldsymbol{T} = \begin{bmatrix} \boldsymbol{A} & -\boldsymbol{G} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}
\end{equation}

其中$\boldsymbol{A} = \boldsymbol{M}\boldsymbol{O}^{\top}, \boldsymbol{B} = \boldsymbol{O}^{\top}\boldsymbol{M}$。

**引理2**：如果$\boldsymbol{X}$满足$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} = \boldsymbol{G}$，则
\begin{equation}
\boldsymbol{T} = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{A} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}
\end{equation}

证明：展开右边
\begin{align}
&\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{A} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\\
&= \begin{bmatrix} \boldsymbol{A} & -\boldsymbol{X}\boldsymbol{B} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\\
&= \begin{bmatrix} \boldsymbol{A} & -\boldsymbol{A}\boldsymbol{X} - \boldsymbol{X}\boldsymbol{B} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}
\end{align}

如果$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} = \boldsymbol{G}$，则右上角正好是$-\boldsymbol{G}$。$\square$

应用$\mcsgn$的相似不变性，我们有
\begin{equation}
\mcsgn(\boldsymbol{T}) = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \mcsgn(\boldsymbol{A}) & \boldsymbol{0} \\ \boldsymbol{0} & \mcsgn(-\boldsymbol{B})\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}
\end{equation}

当$\boldsymbol{M}$满秩时，$\boldsymbol{A}, \boldsymbol{B}$都是正定的，因此$\mcsgn(\boldsymbol{A}) = \boldsymbol{I}, \mcsgn(-\boldsymbol{B}) = -\boldsymbol{I}$，从而
\begin{equation}
\mcsgn(\boldsymbol{T}) = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix} = \begin{bmatrix} \boldsymbol{I} & -2\boldsymbol{X} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}
\end{equation}

因此
\begin{equation}
\boldsymbol{X} = -\frac{1}{2}[\mcsgn(\boldsymbol{T})]_{[:n,n:]}
\end{equation}

其中$[\cdot]_{[:n,n:]}$表示取右上角的$n\times m$块。

**算法2（基于mcsgn的反向传播）**：
1. 输入：$\boldsymbol{M}, \boldsymbol{O} = \msign(\boldsymbol{M}), \boldsymbol{G} = \nabla_{\boldsymbol{O}}\mathcal{L}$
2. 计算$\boldsymbol{A} = \boldsymbol{M}\boldsymbol{O}^{\top}, \boldsymbol{B} = \boldsymbol{O}^{\top}\boldsymbol{M}$
3. 构造$\boldsymbol{T} = \begin{bmatrix} \boldsymbol{A} + \epsilon\boldsymbol{I} & -\boldsymbol{G} \\ \boldsymbol{0} & -\boldsymbol{B} - \epsilon\boldsymbol{I}\end{bmatrix}$（$\epsilon$用于数值稳定）
4. 使用Newton-Schulz迭代计算$\mcsgn(\boldsymbol{T})$
5. 提取$\boldsymbol{X} = -\frac{1}{2}[\mcsgn(\boldsymbol{T})]_{[:n,n:]}$
6. 计算$\nabla_{\boldsymbol{M}}\mathcal{L} = \boldsymbol{X} - \boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O}$
7. 输出：$\nabla_{\boldsymbol{M}}\mathcal{L}$

### Newton-Schulz迭代的导数 ###

前面提到，$\msign$可以通过Newton-Schulz迭代计算。迭代格式为
\begin{equation}
\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\Vert\boldsymbol{M}\Vert_F}, \quad \boldsymbol{X}_{t+1} = a_t\boldsymbol{X}_t + b_t\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c_t\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2
\end{equation}

这个迭代过程本身是可微的，可以直接反向传播。设$\boldsymbol{X}_T \approx \msign(\boldsymbol{M})$，我们需要反向传播梯度。

对于第$t$步的更新，设$\boldsymbol{G}_t = \nabla_{\boldsymbol{X}_t}\mathcal{L}$，我们需要计算$\boldsymbol{G}_{t-1} = \nabla_{\boldsymbol{X}_{t-1}}\mathcal{L}$。

记$\boldsymbol{P}_t = \boldsymbol{X}_t^{\top}\boldsymbol{X}_t$，则
\begin{equation}
\boldsymbol{X}_t = a_{t-1}\boldsymbol{X}_{t-1} + b_{t-1}\boldsymbol{X}_{t-1}\boldsymbol{P}_{t-1} + c_{t-1}\boldsymbol{X}_{t-1}\boldsymbol{P}_{t-1}^2
\end{equation}

对$\boldsymbol{X}_{t-1}$求导（这里需要使用矩阵微积分的乘积法则和链式法则）：
\begin{align}
\boldsymbol{G}_{t-1} &= a_{t-1}\boldsymbol{G}_t + b_{t-1}\boldsymbol{G}_t\boldsymbol{P}_{t-1} + c_{t-1}\boldsymbol{G}_t\boldsymbol{P}_{t-1}^2\\
&\quad + b_{t-1}\nabla_{\boldsymbol{X}_{t-1}}(\boldsymbol{X}_{t-1}\boldsymbol{P}_{t-1})^{\top}\boldsymbol{G}_t\\
&\quad + c_{t-1}\nabla_{\boldsymbol{X}_{t-1}}(\boldsymbol{X}_{t-1}\boldsymbol{P}_{t-1}^2)^{\top}\boldsymbol{G}_t
\end{align}

对于$\boldsymbol{P}_t = \boldsymbol{X}_t^{\top}\boldsymbol{X}_t$的导数，我们有
\begin{equation}
\nabla_{\boldsymbol{X}_t}\tr(\boldsymbol{G}_P^{\top}\boldsymbol{P}_t) = \nabla_{\boldsymbol{X}_t}\tr(\boldsymbol{G}_P^{\top}\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) = 2\boldsymbol{X}_t\boldsymbol{G}_P
\end{equation}

因此，从$\nabla_{\boldsymbol{X}_t\boldsymbol{P}_t}\mathcal{L}$反向传播到$\boldsymbol{X}_t$时，需要考虑$\boldsymbol{P}_t$对$\boldsymbol{X}_t$的依赖。

完整的反向传播公式为：
\begin{align}
\boldsymbol{G}_{P,t-1} &= b_{t-1}\boldsymbol{X}_{t-1}^{\top}\boldsymbol{G}_t + 2c_{t-1}\boldsymbol{P}_{t-1}\boldsymbol{X}_{t-1}^{\top}\boldsymbol{G}_t\\
\boldsymbol{G}_{t-1} &= a_{t-1}\boldsymbol{G}_t + b_{t-1}\boldsymbol{G}_t\boldsymbol{P}_{t-1} + c_{t-1}\boldsymbol{G}_t\boldsymbol{P}_{t-1}^2 + 2\boldsymbol{X}_{t-1}\boldsymbol{G}_{P,t-1}
\end{align}

这个递推关系使得我们可以从$\boldsymbol{G}_T$逐步反向传播到$\boldsymbol{G}_0$。

### 自动微分的实现 ###

在现代深度学习框架（如PyTorch, JAX）中，我们可以利用自动微分（Automatic Differentiation, AD）来自动计算梯度。这里讨论两种实现方式：

**方式1：直接反向传播Newton-Schulz迭代**

最简单的方式是将Newton-Schulz迭代作为一系列可微操作，让自动微分框架自动处理反向传播。

```python
def msign_autograd(M, num_iters=8):
    """使用自动微分计算msign的梯度"""
    # 初始化
    X = M / torch.norm(M, 'fro')

    # Newton-Schulz迭代（使用预定义系数）
    coeffs = [
        (8.28721 * 1.01, -23.5959 * 1.01**3, 17.3004 * 1.01**5),
        (4.10706 * 1.01, -2.94785 * 1.01**3, 0.544843 * 1.01**5),
        # ... 更多系数
    ]

    for i in range(num_iters):
        a, b, c = coeffs[i]
        XTX = X.T @ X
        X = a * X + b * X @ XTX + c * X @ (XTX @ XTX)

    return X
```

这种方式的优点是实现简单，缺点是需要存储所有中间状态$\boldsymbol{X}_0, \boldsymbol{X}_1, \ldots, \boldsymbol{X}_T$，显存开销大。

**方式2：自定义反向传播函数**

更高效的方式是自定义反向传播，使用基于Sylvester方程的解析解。

```python
class MSignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, epsilon=1e-3):
        """前向传播：计算O = msign(M)"""
        O = compute_msign_newton_schulz(M)
        ctx.save_for_backward(M, O)
        ctx.epsilon = epsilon
        return O

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算grad_M"""
        M, O = ctx.saved_tensors
        G = grad_output
        epsilon = ctx.epsilon

        # 构造增广矩阵
        A = M @ O.T
        B = O.T @ M
        n, m = M.shape

        # 添加正则化以处理奇异情况
        A_reg = A + epsilon * torch.eye(n, device=M.device)
        B_reg = B + epsilon * torch.eye(m, device=M.device)

        # 构造T矩阵
        T = torch.cat([
            torch.cat([A_reg, -G], dim=1),
            torch.cat([torch.zeros(m, n, device=M.device), -B_reg], dim=1)
        ], dim=0)

        # 计算mcsgn(T)
        T_sgn = compute_mcsgn_newton_schulz(T)

        # 提取X
        X = -0.5 * T_sgn[:n, n:]

        # 计算最终梯度
        grad_M = X - O @ X.T @ O

        return grad_M, None
```

**方式3：隐函数定理**

另一种思路是利用隐函数定理（Implicit Function Theorem）。由于$\boldsymbol{O}$满足$\boldsymbol{M} = \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{O}$，可以将这个约束视为隐式定义$\boldsymbol{O}$的方程。

设$F(\boldsymbol{O}, \boldsymbol{M}) = \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{O} - \boldsymbol{M} = \boldsymbol{0}$，根据隐函数定理
\begin{equation}
\frac{\partial \boldsymbol{O}}{\partial \boldsymbol{M}} = -\left(\frac{\partial F}{\partial \boldsymbol{O}}\right)^{-1}\frac{\partial F}{\partial \boldsymbol{M}}
\end{equation}

这需要求解一个线性系统，但这个线性系统的结构正是我们前面讨论的Sylvester方程。

### 与SVD导数的关系 ###

前文提到，$\msign$可以通过SVD定义：$\msign(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$。因此，$\msign$的导数与SVD的导数密切相关。

SVD的导数公式已在文献中充分研究。对于$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，我们有
\begin{align}
d\boldsymbol{U} &= (\boldsymbol{I} - \boldsymbol{U}\boldsymbol{U}^{\top})(d\boldsymbol{M})\boldsymbol{V}\boldsymbol{\Sigma}^{-1} + \boldsymbol{U}\boldsymbol{\Omega}_U\\
d\boldsymbol{V} &= (\boldsymbol{I} - \boldsymbol{V}\boldsymbol{V}^{\top})(d\boldsymbol{M})^{\top}\boldsymbol{U}\boldsymbol{\Sigma}^{-1} + \boldsymbol{V}\boldsymbol{\Omega}_V\\
d\boldsymbol{\Sigma} &= \boldsymbol{U}^{\top}(d\boldsymbol{M})\boldsymbol{V}
\end{align}

其中$\boldsymbol{\Omega}_U, \boldsymbol{\Omega}_V$是反对称矩阵，满足
\begin{equation}
[\boldsymbol{\Omega}_U]_{ij} = \frac{\boldsymbol{u}_i^{\top}(d\boldsymbol{M})\boldsymbol{v}_j\sigma_j - \boldsymbol{u}_j^{\top}(d\boldsymbol{M})\boldsymbol{v}_i\sigma_i}{\sigma_i^2 - \sigma_j^2}
\end{equation}

对于$\msign(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$，其导数为
\begin{equation}
d[\msign(\boldsymbol{M})] = (d\boldsymbol{U}_{[:,:r]})\boldsymbol{V}_{[:,:r]}^{\top} + \boldsymbol{U}_{[:,:r]}(d\boldsymbol{V}_{[:,:r]})^{\top}
\end{equation}

注意到$\msign$不依赖于$\boldsymbol{\Sigma}$（因为非零奇异值都变为1），所以$\msign$对奇异值的变化不敏感，只对奇异向量的变化敏感。

将SVD导数公式代入，经过复杂的代数运算，可以验证这与我们前面基于恒等式$\boldsymbol{M} = \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{O}$得到的结果是一致的。这为我们的推导提供了另一个验证。

**定理2（SVD导数与Sylvester方程的等价性）**：基于SVD的$\msign$导数公式与基于Sylvester方程的解在数值上是等价的（当$\boldsymbol{M}$满秩时）。

证明思路：将SVD表达式代入Sylvester方程，利用奇异向量的正交性和特殊结构，可以证明两者给出相同的结果。详细证明较为冗长，这里省略。

### 链式法则的应用 ###

在深度学习中，$\msign$通常不是单独出现，而是作为某个复合函数的一部分。例如，在Muon优化器中，参数更新涉及到
\begin{equation}
\boldsymbol{W}_{t+1} = \boldsymbol{W}_t - \eta\msign(\boldsymbol{G}_t)
\end{equation}

其中$\boldsymbol{G}_t$是梯度或动量。

如果我们要分析优化器的二阶性质（例如计算Hessian或进行超参数优化），就需要对优化器本身求导，这涉及到嵌套的链式法则。

**例1：嵌套msign**

考虑$f(\boldsymbol{M}) = \msign(\msign(\boldsymbol{M}))$。根据链式法则
\begin{equation}
\mathcal{D}f(\boldsymbol{M})[\boldsymbol{H}] = \mathcal{D}[\msign](\msign(\boldsymbol{M}))[\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]]
\end{equation}

注意到$\msign(\msign(\boldsymbol{M})) = \msign(\boldsymbol{M})$（因为$\msign$是幂等的），所以外层导数实际上是在$\boldsymbol{O} = \msign(\boldsymbol{M})$处计算的。

**例2：msign的线性组合**

考虑$f(\boldsymbol{M}) = \boldsymbol{A}\msign(\boldsymbol{M})\boldsymbol{B}$，其中$\boldsymbol{A}, \boldsymbol{B}$是常数矩阵。导数为
\begin{equation}
\mathcal{D}f(\boldsymbol{M})[\boldsymbol{H}] = \boldsymbol{A}\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}]\boldsymbol{B}
\end{equation}

在反向传播中，如果$\nabla_f\mathcal{L} = \boldsymbol{G}$，则
\begin{equation}
\nabla_{\msign(\boldsymbol{M})}\mathcal{L} = \boldsymbol{A}^{\top}\boldsymbol{G}\boldsymbol{B}^{\top}
\end{equation}

然后再用我们的Sylvester方程方法计算$\nabla_{\boldsymbol{M}}\mathcal{L}$。

**例3：迹函数**

考虑$f(\boldsymbol{M}) = \tr(\boldsymbol{C}^{\top}\msign(\boldsymbol{M}))$，这在某些正则化项中出现。

\begin{equation}
\frac{\partial f}{\partial \boldsymbol{M}} = \mathcal{D}[\msign](\boldsymbol{M})^*[\boldsymbol{C}]
\end{equation}

其中$\mathcal{D}[\msign](\boldsymbol{M})^*$是伴随算子（adjoint operator），满足
\begin{equation}
\langle \mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}], \boldsymbol{C} \rangle = \langle \boldsymbol{H}, \mathcal{D}[\msign](\boldsymbol{M})^*[\boldsymbol{C}] \rangle
\end{equation}

这正是我们反向传播算法计算的量。

### 在RNN和TTT中的应用 ###

Test-Time Training (TTT) 结合 Muon 优化器需要将优化步骤嵌入到模型的前向传播中。具体来说，RNN的隐藏状态更新可能涉及
\begin{equation}
\boldsymbol{h}_{t+1} = f(\boldsymbol{h}_t, \boldsymbol{x}_t; \boldsymbol{\theta}_t)
\end{equation}

其中$\boldsymbol{\theta}_t$通过Muon优化器在测试时更新：
\begin{equation}
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\msign(\boldsymbol{g}_t)
\end{equation}

$\boldsymbol{g}_t$是某种梯度或动量估计。

为了训练整个TTT-RNN系统，我们需要通过整个展开的计算图反向传播，这需要$\msign$的导数。

**梯度流分析**：

设最终损失为$\mathcal{L}(\boldsymbol{h}_T)$，我们需要计算$\frac{\partial\mathcal{L}}{\partial\boldsymbol{\theta}_0}$。根据链式法则
\begin{equation}
\frac{\partial\mathcal{L}}{\partial\boldsymbol{\theta}_0} = \sum_{t=0}^{T-1}\frac{\partial\mathcal{L}}{\partial\boldsymbol{\theta}_t}\frac{\partial\boldsymbol{\theta}_t}{\partial\boldsymbol{\theta}_0}
\end{equation}

其中
\begin{equation}
\frac{\partial\boldsymbol{\theta}_{t+1}}{\partial\boldsymbol{\theta}_t} = \boldsymbol{I} - \eta\frac{\partial\msign(\boldsymbol{g}_t)}{\partial\boldsymbol{\theta}_t}
\end{equation}

如果$\boldsymbol{g}_t$本身依赖于$\boldsymbol{\theta}_t$（例如$\boldsymbol{g}_t = \nabla_{\boldsymbol{\theta}_t}\mathcal{L}_{\text{aux}}$），则需要进一步展开链式法则。

这种嵌套的微分结构使得TTT+Muon的训练在计算上非常昂贵，但我们的高效Sylvester求解方法可以显著减少计算成本。

### 数值稳定性考虑 ###

在实际实现中，数值稳定性至关重要。以下是几个关键的稳定性问题和解决方案：

**问题1：奇异值接近零**

当$\boldsymbol{M}$接近秩亏时，$\boldsymbol{A} = (\boldsymbol{M}\boldsymbol{M}^{\top})^{1/2}$和$\boldsymbol{B} = (\boldsymbol{M}^{\top}\boldsymbol{M})^{1/2}$会有接近零的特征值，导致Sylvester方程病态。

**解决方案**：添加正则化项
\begin{equation}
\boldsymbol{A}_{\text{reg}} = \boldsymbol{A} + \epsilon\boldsymbol{I}, \quad \boldsymbol{B}_{\text{reg}} = \boldsymbol{B} + \epsilon\boldsymbol{I}
\end{equation}

其中$\epsilon \sim 10^{-3}$。这相当于假设$\boldsymbol{M}$的最小奇异值至少为$\sqrt{2\epsilon}$。

**问题2：Newton-Schulz迭代的数值误差累积**

Newton-Schulz迭代是高阶方法，但每步都会引入舍入误差。多步迭代后，误差可能累积。

**解决方案**：
1. 使用混合精度计算（FP32用于累积，FP16用于矩阵乘法）
2. 限制迭代次数（通常5-8步足够）
3. 监控迭代收敛性，提前停止

**问题3：梯度爆炸/消失**

在反向传播中，梯度可能因为多次矩阵乘法而爆炸或消失。

**解决方案**：
1. 梯度裁剪（gradient clipping）
2. 使用归一化的梯度（如$\frac{\nabla_{\boldsymbol{M}}\mathcal{L}}{\Vert\nabla_{\boldsymbol{M}}\mathcal{L}\Vert_F}$）
3. 自适应学习率

**问题4：mcsgn的特征值要求**

$\mcsgn$要求矩阵的特征值不在负实轴上。对于增广矩阵$\boldsymbol{T}$，这通常是满足的，但在极端情况下可能失败。

**解决方案**：
1. 验证$\boldsymbol{M}\boldsymbol{O}^{\top}$和$\boldsymbol{O}^{\top}\boldsymbol{M}$的正定性
2. 如果检测到负特征值，增加$\epsilon$
3. 回退到直接SVD方法（作为备用方案）

**问题5：大矩阵的显存限制**

对于大规模矩阵（如$1000 \times 1000$），构造增广矩阵$\boldsymbol{T}$（大小为$2000 \times 2000$）可能超出显存。

**解决方案**：
1. 分块处理（如果矩阵有稀疏或分块结构）
2. 使用低秩近似（Nyström方法）
3. 迭代求解Sylvester方程（不显式构造$\boldsymbol{T}$）

### 计算复杂度分析 ###

让我们分析各种方法的计算复杂度：

**方法1：直接反向传播Newton-Schulz迭代**
- 时间复杂度：$O(Tn^2m)$，其中$T$是迭代次数
- 空间复杂度：$O(Tnm)$（存储所有中间状态）

**方法2：基于SVD的解析解**
- 时间复杂度：$O(n^2m + nm^2)$（SVD计算）
- 空间复杂度：$O(nm)$

**方法3：基于Sylvester方程和mcsgn**
- 时间复杂度：$O(T'(n+m)^3)$，其中$T'$是mcsgn的迭代次数
- 空间复杂度：$O((n+m)^2)$（增广矩阵）

对于$n \approx m$的方阵，方法3的时间复杂度约为$O(T'n^3)$。由于$T' \sim 5$-$8$通常远小于直接方法的操作数，且GPU上矩阵乘法高度优化，方法3在实践中往往最快。

**优化技巧**：
1. **缓存复用**：在前向传播中计算的$\boldsymbol{O}$在反向传播中复用
2. **融合算子**：将多个矩阵操作融合为单个CUDA kernel
3. **异步计算**：利用GPU的流（streams）并行计算不同部分
4. **量化**：使用低精度（如FP16）加速，关键步骤用FP32

### 实验验证：梯度检验 ###

实现$\msign$的自定义梯度后，务必进行**梯度检验**（gradient check）以验证正确性。

**有限差分法**：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial M_{ij}} \approx \frac{\mathcal{L}(\boldsymbol{M} + \epsilon\boldsymbol{E}_{ij}) - \mathcal{L}(\boldsymbol{M} - \epsilon\boldsymbol{E}_{ij})}{2\epsilon}
\end{equation}

其中$\boldsymbol{E}_{ij}$是第$(i,j)$位置为1、其余为0的矩阵。

**检验代码示例**：
```python
def gradient_check(M, loss_fn, epsilon=1e-5):
    """梯度检验"""
    # 自动微分计算的梯度
    M_tensor = torch.tensor(M, requires_grad=True)
    loss = loss_fn(msign(M_tensor))
    loss.backward()
    grad_auto = M_tensor.grad.numpy()

    # 有限差分计算的梯度
    grad_fd = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M_plus = M.copy()
            M_plus[i, j] += epsilon
            M_minus = M.copy()
            M_minus[i, j] -= epsilon

            loss_plus = loss_fn(msign(torch.tensor(M_plus))).item()
            loss_minus = loss_fn(msign(torch.tensor(M_minus))).item()

            grad_fd[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    # 比较
    relative_error = np.linalg.norm(grad_auto - grad_fd) / np.linalg.norm(grad_fd)
    print(f"Relative error: {relative_error}")
    return relative_error < 1e-4  # 通过阈值
```

如果相对误差小于$10^{-4}$，则梯度实现基本正确。

### 高阶导数 ###

在某些高级应用中（如元学习、超参数优化），可能需要二阶导数（Hessian）。

$\msign$的二阶导数可以通过对一阶导数再次应用Fréchet导数得到。设
\begin{equation}
\mathcal{D}^2[\msign](\boldsymbol{M})[\boldsymbol{H}_1, \boldsymbol{H}_2] = \mathcal{D}\left(\mathcal{D}[\msign](\boldsymbol{M})[\boldsymbol{H}_1]\right)[\boldsymbol{H}_2]
\end{equation}

从一阶导数的Sylvester方程
\begin{equation}
\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}
\end{equation}

对$\boldsymbol{M}$再次求导，得到
\begin{equation}
(d\boldsymbol{A})\boldsymbol{X} + \boldsymbol{A}(d\boldsymbol{X}) + (d\boldsymbol{X})\boldsymbol{B} + \boldsymbol{X}(d\boldsymbol{B}) = d\boldsymbol{C}
\end{equation}

其中$d\boldsymbol{A}, d\boldsymbol{B}, d\boldsymbol{C}$可以从一阶导数计算得出，而$d\boldsymbol{X}$就是我们要求的二阶导数，它满足另一个Sylvester方程。

这个过程可以递归地进行，理论上可以计算任意阶导数，但实际中很少需要三阶及以上。

### 小结 ###

在本节中，我们从多个角度深入探讨了$\msign$算子的导数理论：

1. **Fréchet导数框架**：建立了严格的数学基础，定义了矩阵函数的导数
2. **Lyapunov方程方法**：将导数计算转化为标准的线性代数问题
3. **积分表示和谱分解**：提供了另一个理论视角，揭示了与奇异值/特征值的关系
4. **反向传播算法**：给出了实用的、GPU高效的梯度计算方法
5. **与SVD导数的联系**：验证了不同方法的等价性
6. **链式法则应用**：展示了如何在复杂的深度学习模型中使用$\msign$
7. **数值稳定性**：讨论了实际实现中的各种数值问题和解决方案
8. **计算复杂度**：分析了不同方法的效率，指出了最优选择

这些理论和技术为在TTT+Muon等先进架构中使用$\msign$提供了坚实的基础。

## 文章小结 #

本文讨论了$\msign$算子的导数计算，如果你关心"TTT + Muon"的组合，那么本文也许对你有帮助。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11025>_

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

苏剑林. (Jun. 13, 2025). 《msign的导数 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11025>

@online{kexuefm-11025,
title={msign的导数},
author={苏剑林},
year={2025},
month={Jun},
url={\url{https://spaces.ac.cn/archives/11025}},
}