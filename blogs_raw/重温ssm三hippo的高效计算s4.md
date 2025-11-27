---
title: 重温SSM（三）：HiPPO的高效计算（S4）
slug: 重温ssm三hippo的高效计算s4
date: 2024-06-20
tags: 矩阵, 线性, RNN, ssm, 生成模型
status: completed
---

# 重温SSM（三）：HiPPO的高效计算（S4）

**原文链接**: [https://spaces.ac.cn/archives/10162](https://spaces.ac.cn/archives/10162)

**发布日期**: 

---

前面我们用两篇文章[《重温SSM（一）：线性系统和HiPPO矩阵》](/archives/10114)和[《重温SSM（二）：HiPPO的一些遗留问题》](/archives/10137)介绍了HiPPO的思想和推导——通过正交函数基对持续更新的函数进行实时逼近，其拟合系数的动力学正好可以表示为一个线性ODE系统，并且对于特定的基底以及逼近方式，我们可以将线性系统的关键矩阵精确地算出来。此外，我们还讨论了HiPPO的离散化和相关性质等问题，这些内容奠定了后续的SSM工作的理论基础。

接下来，我们将介绍HiPPO的后续应用篇[《Efficiently Modeling Long Sequences with Structured State Spaces》](https://papers.cool/arxiv/2111.00396)（简称S4），它利用HiPPO的推导结果作为序列建模的基本工具，并从新的视角探讨了高效的计算和训练方式，最后在不少长序列建模任务上验证了它的有效性，可谓SSM乃至RNN复兴的代表作之一。

## 基本框架 #

S4使用的序列建模框架，是如下的线性ODE系统：  
\begin{equation}\begin{aligned}  
x'(t) =&\, A x(t) + B u(t) \\\  
y(t) =&\, C^* x(t) + D u(t)  
\end{aligned}\end{equation}  
这里$u,y,D\in\mathbb{R};x\in\mathbb{R}^d;A\in\mathbb{R}^{d\times d};B,C\in\mathbb{R}^{d\times 1}$，${}^*$是转置共轭运算，如果是实矩阵的话，那就是单纯的转置。由于完整的模型通常还会带有残差结构，最后一项$D u(t)$可以整合到残差里边，所以我们可以直接假设$D=0$来稍微简化一下形式，但不会降低模型的能力。

该系统具备**相似不变性** ，如果$\tilde{A}$是$A$的相似矩阵，即$A = P^{-1}\tilde{A}P$，那么代入整理得  
\begin{equation}\begin{aligned}  
Px'(t) =&\, \tilde{A} Px(t) + PB u(t) \\\  
y(t) =&\, ((P^{-1})^* C)^* P x(t)  
\end{aligned}\end{equation}  
将$Px(t)$视为一个整体替换原来的$x(t)$，那么新系统的变化是$(A,B,C)\to(\tilde{A},PB,(P^{-1})^*C)$，但输出完全不改变。这意味着如果存在$A$的某个相似矩阵$\tilde{A}$使得计算更加简单，那么可以完全转到$\tilde{A}$中分析而不改变结果，这就是后面一系列分析的核心思路。

特别地，S4将矩阵$A$选取为HiPPO-LegS矩阵，即  
\begin{equation}A_{n,k} = -\left\\{\begin{array}{l}\sqrt{(2n+1)(2k+1)}, &k < n \\\ n+1, &k = n \\\  
0, &k > n\end{array}\right.\end{equation}  
这个选择的特别之处在于，我们此前推导LegS所满足的ODE是$x'(t) =\frac{A}{t} x(t) + \frac{B}{t} u(t)$的形式，而LegT的ODE才是$x'(t) = A x(t) + B u(t)$的形式，所以现在就是说LegT的ODE搭配了LegS的$A$矩阵，因此首先要问的问题是：这样的组合会带来什么影响呢？比如它对历史的记忆是否跟LegS一样依然是完整的、平权的？

## 指数衰减 #

答案是否定的——S4所选取的ODE系统，关于历史的记忆是指数衰减的，我们可以从两个角度理解这一点。

第一个角度是从[《重温SSM（二）：HiPPO的一些遗留问题》](/archives/10137)讨论过的变换出发，将LegS型ODE可以等价地写成：  
\begin{equation}Ax(t) + Bu(t) = t x'(t) = \frac{d}{d\ln t} x(t)\end{equation}  
所以设$\tau=\ln t$就可以将LegS型ODE变成时间变量为$\tau$的LegT型ODE，也就是S4所用的ODE。我们知道，LegS会平等对待每一处历史，但这前提是输入为$u(t)=u(e^{\tau})$，但S4的ODE相当于输入直接改为$u(\tau)$，此时对$\tau$做均匀离散化的话，结果就是每一处的权重不相等——假设$t\in[0,T]$，用概率密度的写法就是$dt/T=\rho(\tau)d\tau$，即$\rho(\tau)=e^{\tau}/T$，即权重是$\tau$的指数函数，越新的历史权重越大。

第二个角度则需要多一点线性代数知识。同样在[《重温SSM（二）：HiPPO的一些遗留问题》](/archives/10137)我们说过HiPPO-LegS的矩阵$A$理论上是可以对角化的，并且其特征值为$[-1,-2,-3,\cdots]$，于是存在可逆矩阵$P$使得$A = P^{-1}\Lambda P$，其中$\Lambda = \text{diag}(-1,-2,\cdots,-d)$，根据相似不变性，原系统等价于新系统  
\begin{equation}\begin{aligned}  
x'(t) =&\, \Lambda x(t) + PB u(t) \\\  
y(t) =&\, C^* P^{-1} x(t)  
\end{aligned}\end{equation}  
离散化后（以前向欧拉为例）：  
\begin{equation}x(t+\epsilon) = (I + \epsilon\Lambda) Px(t) + \epsilon P B u(t)\end{equation}  
这里的$I + \epsilon\Lambda$是每个分量都小于1的对角线矩阵，也就意味着每迭代一步，就将历史信息乘以一个小于1的数，多步叠加后，就呈现出指数衰减的效应。

## 离散格式 #

虽然指数衰减看上去没有LegS平等对待每一处历史那么优雅，但实际上没有免费的午餐，对于固定大小的记忆状态$x(t)$，在记忆区间越来越大时，LegS平等对待每一处历史的做法反而会导致每一处历史都比较模糊，对于符合“近大远小”的场景反而得不偿失。此外，S4型ODE右端没有显式地出现时间$t$，这也有助于提供训练效率。

对S4型ODE的记忆性质心中有数之后，我们就可以着手下一步操作了。为了处理实际中的离散序列，我们首先要进行离散化，在上一篇文章中，我们给出了两种精度较高的离散格式，一种是双线性形式  
\begin{equation}x_{k+1} = (I - \epsilon A/2)^{-1}[(I + \epsilon A/2) x_k + \epsilon B u_k] \end{equation}  
它具有二阶的精度，S4采用的就是这个离散化格式，也是本文接下来所探讨的格式。另一种是基于精确求解常输入的ODE，得到  
\begin{equation}x_{k+1} = e^{\epsilon A} x_k + A^{-1} (e^{\epsilon A} - I) B u_k\end{equation}  
作者后面的作品包括Mamba都是用这个格式，此时一般都要假设$A$为对角矩阵，因为对于LegS的矩阵$A$，矩阵指数算起来并不友好。

现在我们记：  
\begin{equation}\bar{A}=(I - \epsilon A/2)^{-1}(I + \epsilon A/2),\quad\bar{B}=\epsilon(I - \epsilon A/2)^{-1}B,\quad\bar{C}=C\end{equation}  
那么就得到线性RNN：  
\begin{equation}\begin{aligned}  
x_{k+1} =&\, \bar{A} x_k + \bar{B} u_k \\\  
y_{k+1} =&\, \bar{C}^* x_{k+1} \\\  
\end{aligned}\label{eq:s4-r}\end{equation}  
其中$\epsilon > 0$是离散化步长，是人为选择的超参数。

## 卷积运算 #

在上一篇文章中，我们还提到了HiPPO-LegS的矩阵$A$具备计算高效的特点，具体表现为$A$或$\bar{A}$跟向量$x$相乘，存在计算复杂度为$\mathcal{O}(d)$而不是一般的$\mathcal{O}(d^2)$的高效算法，但这仅仅意味着式$\eqref{eq:s4-r}$递归计算时比一般的RNN高效，而如果想要进行高效训练的话，单纯递归是不够的，需要探究并行计算方法。

线性RNN的并行计算有两种思路：一种是在[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](/archives/9554)介绍过的视为Prefix Sum问题，直接用Upper/Lower、Odd/Even、Ladner-Fischer等Associative Scan算法进行计算，论文可参考[《Prefix Sums and Their Applications》](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html)；另一种是转化为矩阵序列和向量序列的卷积运算，利用快速傅里叶变换（FFT）来加速，这是S4的思路。但不管哪一种，它们面临共同的瓶颈：幂矩阵$\bar{A}^k$的计算。

具体来说，我们一般会设初始状态$x_0$为0，那么就可以写出：  
\begin{equation}\begin{aligned}  
y_1 =&\, \bar{C}^*\bar{B}u_0\\\  
y_2 =&\, \bar{C}^*(\bar{A}x_0 + \bar{B}u_1) = \bar{C}^*\bar{A}\bar{B}u_0 + \bar{C}^*\bar{B}u_1\\\  
y_3 =&\, \bar{C}^*(\bar{A}x_1 + \bar{B}u_2) = \bar{C}^*\bar{A}^2 Bu_0 + \bar{C}^*\bar{A}Bu_1 + \bar{C}^*\bar{B}u_2\\\\[5pt]  
\vdots  
\\\  
y_L =&\, \bar{C}^*(\bar{A} x_{L-1}+\bar{B}u_{L-1}) = \sum_{k=0}^{L-1} \bar{C}^*\bar{A}^k \bar{B}u_{L-k} = \bar{K}_{< L} * u_{< L}  
\end{aligned}\end{equation}  
其中$*$代表卷积运算，而  
\begin{equation}\bar{K}_k = \bar{C}^*\bar{A}^k\bar{B},\quad \bar{K}_{< L} = \big(\bar{K}_0,\bar{K}_1,\cdots,\bar{K}_{L-1}\big),\quad u_{< L} = (u_0,u_1,\cdots,u_{L-1})\end{equation}  
注意根据目前的约定，$\bar{C}^*\bar{A}^k \bar{B}$和$u_k$都是标量，所以有$\bar{K}_{< L},u_{< L}\in\mathbb{R}^L$。我们知道，卷积运算可以通过（离散）傅立叶变换转换为频域的乘法运算，然后再逆变换回来，它的复杂度为$\mathcal{O}(L\log L)$，$L$是序列长度。虽然复杂度看上去比直接递归的$\mathcal{O}(L)$要大，但是傅立叶变换是可以并行的，所以实际上计算速度要更快。

所以，现在问题是如何高效地计算卷积核$\bar{K}_{< L}$，它需要计算幂矩阵$\bar{A}^k$，按定义计算的话复杂度还是相当大的。当然，如果只是计算$\bar{A}^k$那倒不是什么问题，因为$A$是一个常数矩阵，给定$\epsilon$后$\bar{A}$也是常数矩阵，不管它的幂多难算，都可以提前算好存起来。然而，$\bar{A}^k$只是中间步骤，我们还要算$\bar{C}^*\bar{A}^k\bar{B}$，而S4将$\bar{C},\bar{B}$视为训练参数，所以$\bar{C}^*\bar{A}^k\bar{B}$没法提前算好，就是提前算好$\bar{A}^k$效率还是不大够。

## 生成函数 #

在进一步分析之前，我们先来插入一个生成函数的概念，这是后面的高效计算的基础步骤之一。同时，对于不大了解卷积运算和离散傅立叶变换的读者，这也可以作为一个科普步骤，从中我们可以大致了解到傅立叶变换加速卷积运算的基本原理。

对于给定序列$a = (a_0,a_1,a_2,\cdots)$，它的生成函数就是将每个分量当成幂级数的系数来构建幂级数：  
\begin{equation}\mathcal{G}(z|a) = \sum_{k=0}^{\infty} a_k z^k\end{equation}  
如果有两个序列$a = (a_0,a_1,a_2,\cdots)$和$b = (b_0,b_1,b_2,\cdots)$，那么它们生成函数的乘积：  
\begin{equation}\mathcal{G}(z|a)\mathcal{G}(z|b) = \left(\sum_{k=0}^{\infty} a_k z^k\right)\left(\sum_{l=0}^{\infty} b_l z^l\right) = \sum_{k=0}^{\infty}\sum_{l=0}^{\infty}a_k b_l z^{k+l} = \sum_{l=0}^{\infty}\left(\sum_{k=0}^l a_k b_{l-k}\right) z^l \end{equation}  
留意到了没有？$\mathcal{G}(z|a)\mathcal{G}(z|b)$的第$l$项系数（即$z^{l-1}$的系数），正好是$a_{< l}=(a_0,\cdots,a_{l-1})$与$b_{< l}=(b_0,\cdots,b_{l-1})$的卷积运算。如果我们有快速计算生成函数以及快速提取生成函数某一项系数的方法，那么就可以将卷积运算转换为生成函数，做普通乘法之后然后再提取相应的系数。

[离散傅立叶变换（Discrete Fourier Transform，DFT）](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)正是这样的一种构建生成函数的思路。首先注意到，如果我们只需要对$a,b$的不超过前$L$项做卷积运算，那么生成函数的求和上限不一定非得到正无穷，求和上限改为$L-1$也是可以的。针对这种需求，DFT没有对所有$z$来计算生成函数，而是选取了特定的$z=e^{-2i\pi l/L},l=0,1,2,\dots,L-1$进行计算：  
\begin{equation}\hat{a}_l = \sum_{k=0}^{L-1} a_k \left(e^{-2i\pi l/L}\right)^k = \sum_{k=0}^{L-1} a_k e^{-2i\pi kl/L}\end{equation}  
提取系数的逆变换（Inverse DFT，IDFT）则是  
\begin{equation}a_k = \frac{1}{L}\sum_{l=0}^{L-1} \hat{a}_l e^{2i\pi kl/L}\end{equation}  
DFT和IDFT我们都可以通过[快速傅里叶变换（Fast Fourier Transform，FFT）](https://en.wikipedia.org/wiki/Fast_Fourier_transform)进行高效计算，大部分数值计算框架都已内置了相应函数，所以DFT和IDFT的计算在效率上没有问题。但要注意，如果用DFT来计算卷积的话，需要稍微微调一下，因为$e^{-2i\pi l/L}$是周期函数，我们没法区分$e^{-2i\pi l/L}$和$e^{-2i\pi (l+L)/L}$，而当我们将两个$L$项求和的DFT相乘时，结果会出现$l \geq L$的$e^{-2i\pi kl/L}$项，它会跟$e^{-2i\pi k(l-L)/L}$项混合，从而做IDFT时实则得到的是两项的系数相加，这样作为卷积结果来说是不正确的。

解决这个问题的方法是将$e^{-2i\pi l/L}$的$L$改为$2L$（但求和还是$L$项求和），也就是增大它的周期，使得乘积结果都是单个周期内，即将DFT的定义改为  
\begin{equation}\hat{a}_l = \sum_{k=0}^{L-1} a_k e^{-i\pi kl/L}\end{equation}  
不过现成的FFT函数基本上都不支持单独调整周期，而是默认周期就是数组长度，所以等价的处理方式是在$(a_0,a_1,\cdots,a_{L-1})$后面拼接$L$个零再做常规的DFT，得到乘积后做IDFT，最后只取前$L$个结果。

## 从幂到逆 #

对于卷积核$\bar{K}$，我们有  
\begin{equation}\mathcal{G}(z|\bar{K}) = \sum_{k=0}^{\infty} \bar{C}^*\bar{A}^k \bar{B}z^k = \bar{C}^*\left(I - z\bar{A}\right)^{-1}\bar{B}\label{eq:k-gen}\end{equation}  
可以发现，生成函数不仅可以加速卷积的计算，它还将原本的幂矩阵$\bar{A}^k$的计算转化为逆矩阵$\left(I - z\bar{A}\right)^{-1}$的计算。

什么样的矩阵$\bar{A}$，它对应的$\left(I - z\bar{A}\right)^{-1}$比较容易计算呢？首先对角阵肯定没问题，如果$\bar{A}$是对角阵，那么$I - z\bar{A}$也是对角阵，对角阵的逆直接将对角线元素都取逆即可。其次，如果$\bar{A}$可以对角化为$\bar{\Lambda}$，即$\bar{A}=P^{-1}\bar{\Lambda} P$，那么$\left(I - z\bar{A}\right)^{-1}$同样容易计算，因为  
\begin{equation}\left(I - z\bar{A}\right)^{-1} = \left(P^{-1}(I - z\bar{\Lambda})P\right)^{-1} = P^{-1}\left(I - z\bar{\Lambda}\right)^{-1} P\end{equation}

那$\bar{A}$能不能对角化呢？这取决于$A$能不能对角化。如果$A=P^{-1}\Lambda P$，根据相似不变性，我们可以完全转到$A=\Lambda$的新系统去计算，而根据定义新的$\bar{A}$为：  
\begin{equation}\begin{aligned}  
\bar{A}=&\,(I - \epsilon A/2)^{-1}(I + \epsilon A/2) \\\  
=&\,(I - \epsilon\Lambda/2)^{-1}(I + \epsilon\Lambda/2)  
\end{aligned}\end{equation}  
显然是一个对角阵。

那么$A$可以对角化吗？答案是 _理论上可以，实际上不行_ 。理论上可以，是因为从理论上来说，几乎所有矩阵在复数域内都可以对角化，并且在上一篇文章已经给出了LegS的$A$特征值为$[-1,-2,-3,\cdots]$，也就是连对角化后的对角矩阵我们都知道长什么样了。实际上不行，是指对数值计算来说很难，因为数值计算要考虑精度、内存、时间等，只要三者之一超出了限度或容忍度，那么理论可行的算法在实际中就不成立。

对于$A$矩阵，实际上不行的主要原因是对角化$A$所需要的矩阵$P$存在数值不稳定问题，说白了也是计算机精度有限导致的。对于这一点，原论文直接不加解释地给出了矩阵$P$的解析解，然后进行验证，这显然不利于读者理解。下面笔者从特征向量计算的角度，给出另一个理解思路。

## 特征向量 #

$A$的对角化等价于$-A$的对角化，因为$A$的特征值全是负数，所以简单起见我们转而考虑$-A$的对角化，它有$d$个不同的特征值$\lambda=1,2,\cdots,d$，对角化它所需的矩阵就是其特征向量的堆叠，所以求$P$本质上是求特征向量。而对于已知特征值的矩阵，求解特征向量的直接方法是求解方程$-Av=\lambda v$。

上一篇文章中“计算高效”那一节，我们已经给出了$Av$的第$n$个分量的计算结果：  
\begin{equation}(Av)_n = n v_n -\sqrt{2n+1}\sum_{k=0}^n \sqrt{2k+1}v_k \end{equation}  
所以$-Av=\lambda v$意味着  
\begin{equation}\sqrt{2n+1}\sum_{k=0}^n \sqrt{2k+1}v_k - n v_n = \lambda v_n\end{equation}  
记$S_n = \sum\limits_{k=0}^n \sqrt{2k+1}v_k$，那么$\sqrt{2n+1}v_n=S_n - S_{n-1}$，稍加整理得  
\begin{equation}S_{n-1} = \frac{\lambda - n - 1}{\lambda + n}S_n\end{equation}  
注意$-Av=\lambda v$是一个不定方程，我们有一些灵活调整的自由度（即特征向量不是唯一的），由于$n$最大是$d-1$，我们可以设$S_{d-1}=1$，然后递归地往回推，直到$\lambda - n - 1=0$得到$S_{\lambda - 1} = 0$，此后$\forall n < \lambda - 1$都有$S_n = 0$，而对于$n > \lambda - 1$，则有  
\begin{equation}S_n = (-1)^{d-n-1}\frac{(d-\lambda)! (n+\lambda)!}{(d+\lambda-1)! (n-\lambda + 1)!}\end{equation}  
由于我们是想要证明$P$的数值不稳定性，那么观察一个特征向量即可，我们取$n=\lambda=d/3$（如果$d$不是3的倍数，简单取个整即可，结论不变），那么  
\begin{equation}|S_{d/3}| = \frac{\left(\frac{2d}{3}\right)! \left(\frac{2d}{3}\right)!}{\left(\frac{4d}{3}-1\right)!}\sim \mathcal{O}(\sqrt{d}\,2^{-4d/3})\end{equation}  
最后的$\sim$可以由[String公式](https://en.wikipedia.org/wiki/Stirling%27s_approximation)得到。由该结果我们可以看到，对于$d/3$这个特征值，从$S_{d-1}$到$S_{d/3}$存在一个指数级别的衰减过程（反之则爆炸），那么特征向量的分量$v_{d-1}$到$v_{d/3}$也存在类似的衰减，在浮点数的有限精度内，是很难精确处理这样的特征向量的。所以，直接对角化$A$的矩阵$P$存在数值上的不稳定性。

## 对角低秩 #

除了对角阵外，当$\bar{A}$可以低秩分解时，同样可以降低$\left(I - z\bar{A}\right)^{-1}$的计算难度。这是因为我们有如下的Woodbury恒等式：  
\begin{equation}(I - UV^*)^{-1} = \sum_{k=0}^{\infty} (UV)^k = I + U\left(\sum_{k=0}^{\infty}(V^* U)^k\right)V = I + U(I - V^* U)^{-1} V^*\end{equation}  
这里$U,V\in\mathbb{R}^{d\times r}$，推导过程利用了$(UV^*)^k = U(V^* U)^{k-1}V$。如果$d \gg r$，那么理论上$(I - V^* U)^{-1}$的计算量就比$(I - UV^*)^{-1}$少得多，因此可以加速计算。特别地，如果$r=1$，那么$(I - V^* U)^{-1}$就是一个标量的倒数，计算起来最简单。

然而，我们知道$A$是一个下三角阵，且对角线元素没有一个是零，那么它就一定是满秩矩阵。再结合上一节的结论，也就是说$A$即不低秩，对角化又存在实践上的困难，所以这都不适用，还有什么办法呢？有！利用上面的Woodbury恒等式，我们可以推出它更一般的版本：  
\begin{equation}\begin{aligned}  
(M - UV^*)^{-1} =&\, (M(I - (M^{-1}U)V^*))^{-1} = (I - (M^{-1}U)V^*)^{-1}M^{-1} \\\  
=&\, (I + M^{-1}U(I - V^*M^{-1}U)^{-1} V^*)M^{-1} \\\  
=&\, M^{-1} + M^{-1}U(I - V^*M^{-1}U)^{-1} V^*M^{-1} \\\  
\end{aligned}\end{equation}  
这个结果告诉我们，如果$M$的逆比较容易算，那么它加/减一个低秩矩阵的逆也容易算。那什么样的矩阵逆比较容易算呢？又回到上一节的答案——对角矩阵。所以，我们可以想办法将$A$或者$\bar{A}$往“对角+低秩”的形式上凑。

事实上，仔细观察就会发现，$A$矩阵本身就有“对角+低秩”的影子。在上一篇文章中，我们将$A$的定义等价地改写为：  
\begin{equation}A_{n,k} = \left\\{\begin{array}{l}n\delta_{n,k} - \sqrt{2n+1}\sqrt{2k+1}, &k \leq n \\\ 0, &k > n\end{array}\right.\end{equation}  
其中$n\delta_{n,k}$实质就是对角矩阵$\text{diag}(0,1,2,\cdots)$，而$\sqrt{2n+1}\sqrt{2k+1}$则可以重写为低秩矩阵形式$v v^*$，其中$v=[1,\sqrt{3},\sqrt{5},\cdots]^*\in\mathbb{R}^{d\times 1}$，换句话说，如果没有$k > n, A_{n,k}=0$的规定，那么$A$本身就是对角矩阵减去低秩矩阵的形式了。

## 点睛之笔 #

虽然有了下三角阵的约束后，这个规律就不再适用了，但我们可以充分利用原本就有的$v v^*$结构，来辅助构建新的可对角化矩阵。但不得不说，这个技巧相当机智，堪称点睛之笔，让人惊叹，再次为原作者点赞。具体来说，我们考虑$A+\frac{1}{2}v v^*$：  
\begin{equation}\left(A + \frac{1}{2}v v^*\right)_{n,k} = \left\\{\begin{array}{l}n\delta_{n,k} - \frac{1}{2}\sqrt{2n+1}\sqrt{2k+1}, &k \leq n \\\ \frac{1}{2}\sqrt{2n+1}\sqrt{2k+1}, &k > n\end{array}\right.\end{equation}  
这个新矩阵的对角线元素正好是$-\frac{1}{2}I$，我们再加上$\frac{1}{2}I$，就得到  
\begin{equation}\left(A + \frac{1}{2}v v^*+\frac{1}{2}I\right)_{n,k} = \left\\{\begin{array}{} - \frac{1}{2}\sqrt{2n+1}\sqrt{2k+1}, &k < n \\\ 0, &k=n \\\ \frac{1}{2}\sqrt{2n+1}\sqrt{2k+1}, &k > n\end{array}\right.\end{equation}  
重点来了，可以看到这是一个反对称矩阵，所以它一定可以（在复数域中）对角化！于是我们就将$A$分解为了可对角化矩阵与低秩矩阵之和！可能有读者质疑，原本$A$就一定是可对角化矩阵，但还是有数值稳定性问题，难道这个反对称矩阵的对角化不用担心数值稳定性问题吗？重点的重点来了，反对称矩阵不单单一定可以对角化，它一定可以被正交矩阵（复数域叫做酉矩阵）对角化！酉矩阵一般数值稳定性都非常好，所以不用担心这个问题，这也就是为什么我们不直接对角化$A$，而绕一圈来构建反对称矩阵的原因。

现在我们得到，存在对角矩阵$\Lambda$和酉矩阵$U$，使得$A + \frac{1}{2}v v^*+\frac{1}{2}I = U^*\Lambda U$，从而  
\begin{equation}A = U^*\Lambda U - \frac{1}{2}I - \frac{1}{2}v v^* = U^*\left(\Lambda - \frac{1}{2}I - \frac{1}{2}(Uv)(Uv)^*\right) U\end{equation}  
抛开脚手架，我们发现最终的结论可以简化为“$A$同构于对角阵减去秩1矩阵”：存在酉矩阵$U$、对角矩阵$\Lambda$、列向量$u,v$，使得：  
\begin{equation}A = U^*\left(\Lambda - uv^*\right) U\end{equation}  
注意“对角+低秩”的矩阵乘以向量是计算高效的，比如  
\begin{equation}\left(\Lambda - uv^*\right)x = \Lambda x - u(v^*x)\end{equation}  
$\Lambda x$相当于将$\Lambda$当成向量与$x$逐位相乘，而$u(v^*x)$则是$v$跟$x$先做内积，然后得到一个标量乘以向量$u$，这些都可以在$\mathcal{O}(d)$内完成。

## 最后冲刺 #

有了$A=U^*\left(\Lambda - uv^*\right) U$，再次根据相似不变性，我们接下来的所有计算都可以转到$A=\Lambda - uv^*$中进行，所以下面均设$A=\Lambda - uv^*$。首先，对于$\bar{A}$：  
\begin{equation}\bar{A}=\big(I - \epsilon (\Lambda - uv^*)/2\big)^{-1}\big(I + \epsilon (\Lambda - uv^*)/2\big)\end{equation}  
留意到$I - \epsilon (\Lambda - uv^*)/2= \frac{\epsilon}{2}(D + uv^*)$，其中$D=\frac{2}{\epsilon}I - \Lambda$是对角阵，于是利用Woodbury恒等式得到：  
\begin{equation}\big(I - \epsilon (\Lambda - uv^*)/2\big)^{-1} =\frac{2}{\epsilon}(D + uv^*)^{-1} = \frac{2}{\epsilon}\left[D^{-1} - D^{-1}u(I + v^*D^{-1}u)^{-1} v^*D^{-1}\right]\end{equation}  
仔细观察，这同样是“对角+低秩”的形式，再乘以$\big(I + \epsilon (\Lambda - uv^*)/2\big)$后就能完成$\bar{A}$的计算，最终结果是两个“对角+低秩”矩阵的相乘，意味着它同样具有计算高效的特点，这个结果可以在递归推理中用到。

最后是并行训练所需要的卷积核，我们已经将它转化为生成函数$\eqref{eq:k-gen}$，现在我们就可以来完成它的计算了。首先通过类似“通分”的操作可以证明：  
\begin{equation}\begin{aligned}  
\mathcal{G}(z|\bar{K}) = \bar{C}^* \left(I - \bar{A}z\right)^{-1}\bar{B} =&\, \bar{C}^* \left(I - (I - \epsilon A/2)^{-1}(I + \epsilon A/2)z\right)^{-1}\bar{B} \\\  
=&\, \bar{C}^* \left[(I - \epsilon A/2)^{-1}\big((I - \epsilon A/2)-(I + \epsilon A/2)z\big)\right]^{-1}\bar{B} \\\  
=&\, \bar{C}^* \big[(I - \epsilon A/2)-(I + \epsilon A/2)z\big]^{-1}(I - \epsilon A/2)\bar{B} \\\  
=&\, \bar{C}^* \big[(I - \epsilon A/2)-(I + \epsilon A/2)z\big]^{-1}B\epsilon \\\  
=&\, \bar{C}^* \big[(1-z)I - (1+z)\epsilon A / 2\big]^{-1}B\epsilon \\\  
=&\, \frac{2}{1+z}\bar{C}^* \left[\frac{2}{\epsilon}\frac{1-z}{1+z}I - A\right]^{-1}B \\\  
\end{aligned}\end{equation}  
于是代入$A=\Lambda - uv^*$得到  
\begin{equation}\mathcal{G}(z|\bar{K}) = \frac{2}{1+z}\bar{C}^* \left[\frac{2}{\epsilon}\frac{1-z}{1+z}I - (\Lambda - uv^*)\right]^{-1}B=\frac{2}{1+z}\bar{C}^* (R_z + uv^*)^{-1}B\end{equation}  
这里$R_z = \frac{2}{\epsilon}\frac{1-z}{1+z}I - \Lambda$是个对角阵，于是再次利用Woodbury恒等式就可以完成计算：  
\begin{equation}\mathcal{G}(z|\bar{K}) = \frac{2}{1+z}\bar{C}^* \left[R_z^{-1} - R_z^{-1}u(I + v^*R_z^{-1}u)^{-1} v^*R_z^{-1}\right]B\end{equation}  
这是关于$z$的标量函数。不过要注意一个细节，傅立叶变换所需要的实际是“截断生成函数”：  
\begin{equation}\mathcal{G}_L(z|\bar{K}) = \sum_{k=0}^{L-1} \bar{C}^*\bar{A}^k \bar{B}z^k = \bar{C}^*(I - z^L\bar{A}^L)\left(I - z\bar{A}\right)^{-1}\bar{B}\end{equation}  
也就相当于$\mathcal{G}(z|\bar{K})$的$\bar{C}^*$要换成$\bar{C}^*(I - z^L\bar{A}^L)$，这里$L$是提前选定的最大训练长度。接下来，我们只需要代入$z=e^{-2i\pi l/L},l=0,1,2,\dots,L-1$进行计算，结果就是$\bar{K}$的DFT，然后IDFT就得到$\bar{K}$了，这个过程还可以转化为Cauchy核问题加速一下，但个人认为不是太核心，就不展开讨论了。最后的最后，还有一个技巧，就是对于$z=e^{-2i\pi l/L}$有$z^L=1$，此时只是相当于将$\bar{C}^*$要换成$\bar{C}^*(I - \bar{A}^L)$，而S4将$\bar{C}$当成训练参数，所以我们可以直接将$\bar{C}^*(I - \bar{A}^L)$当成训练参数，事后再从中解出$\bar{C}$用于推理，这样训练时就可以避免计算$\bar{A}^L$了。

这里看上去我们也可以代入$z=e^{-i\pi l/L}$直接计算卷积所用的$\bar{K}$的DFT，而不是迂回地先IDFT得到$\bar{K}$，然后拼接零再DFT，但问题是此时$z^L=(-1)^l$是一个不定值，我们没法将$\bar{C}^*(I - z^L\bar{A}^L)$看成单个训练参数，这会导致在训练过程中需要计算$\bar{A}^L$，计算量比较大（当然，如果训练过程中$\bar{A}$是完全固定的，那么可以提前算出来，视情况而定）。

## 草草收尾 #

经过一通艰难的“长篇大论”，我们总算把S4中比较关键的数学细节都捋了一遍，希望能够对有兴趣了解S4的读者有所帮助。可以看到，S4是对HiPPO的进一步补充和完成，它的关键一笔是提出了$A$等价于“对角+低秩”的矩阵形式，为剩余部分的分析奠定了基础。因为一开始$A$是分段定义的形式，而不是矩阵运算形式，这样的定义不利于应用现有的线性代数工具进行一般化分析。

由于HiPPO的推导是基于$u(t)$是一维函数进行的，所以到目前为止，S4的$u_k$也都还是标量。那么S4怎么处理向量序列输入呢？非常暴力，它直接对每个分量独立地应用一遍前述线性RNN，每个RNN使用不同的$\epsilon,B,C$参数，然后将结果拼接起来，这个做法直到作者最新的Mamaba依然还被应用。当然，也有简化的做法，直接在单个RNN中处理向量输入，只需要相应地将$B,C$改为矩阵就行，这就是[S5](https://papers.cool/arxiv/2208.04933)（作者不是Albert Gu了），这种做法可以理解为单纯借用了S4的线性RNN形式以及HiPPO的矩阵$A$，而抛开了HiPPO的其他细枝末节，也取得了不错的效果。

让人啼笑皆非的是，S4提出了诸多精妙的数学技巧来简化和加速$A$的计算，结果从[《Diagonal State Spaces are as Effective as Structured State Spaces》](https://papers.cool/arxiv/2203.14343)开始，原作者的后续工作包括Mamba基本上都抛弃了这部分内容，而是直接假设$A$为对角矩阵，这样RNN部分就跟[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](/archives/9554)介绍的LRU大同小异了。因此，从当前最新的SSM及线性RNN的角度看，S4及HiPPO系列工作某种意义上来说已经是“过时”了。很多讲解Mamba的文章从HiPPO、S4开始说起，从事后来说可谓是“大可不必”了。

当然，对于笔者来说，花那么长的篇幅去学习HiPPO和S4，并不是简单为了理解或使用最新的SSM和RNN模型，而是通过学习HiPPO背后的假设和推导，了解线性系统的记忆方式和瓶颈，为将来构建新模型、新方法积累更多的思路。此外，HiPPO和S4中诸多精妙的数学技巧也让人赏心悦目，并且也不失为提升数学能力的相当不错的练习题。

## 文章小结 #

本文介绍了HiPPO的后续之作S4，它的关键之处是提出了“对角矩阵+低秩矩阵”的分解，从而实现了HiPPO矩阵的高效并行计算，本文主要对其中比较困难的数学细节做了介绍和推导。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10162>_

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

苏剑林. (Jun. 20, 2024). 《重温SSM（三）：HiPPO的高效计算（S4） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10162>

@online{kexuefm-10162,  
title={重温SSM（三）：HiPPO的高效计算（S4）},  
author={苏剑林},  
year={2024},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/10162}},  
} 


---

## 公式推导与注释

本节提供S4算法的详细数学推导,包括DPLR分解、Woodbury恒等式、Cauchy核加速等核心技术的完整证明。

### 一、DPLR分解的完整推导

#### 1.1 对角加低秩(DPLR)分解的动机

对于HiPPO-LegS矩阵$A$,我们希望找到一种表示形式,使得计算$(I - zA)^{-1}$变得高效。直接对角化$A$存在数值稳定性问题,因此我们寻求"对角+低秩"的分解形式。

**定理1.1 (DPLR分解存在性)**: 对于HiPPO-LegS矩阵$A$,存在酉矩阵$U$、对角矩阵$\Lambda$和列向量$u,v$,使得:
\begin{equation}
A = U^*(\Lambda - uv^*)U \tag{1}
\end{equation}

**证明**:

步骤1: 回顾HiPPO-LegS矩阵的定义:
\begin{equation}
A_{n,k} = \left\{\begin{array}{ll}
n\delta_{n,k} - \sqrt{2n+1}\sqrt{2k+1}, & k \leq n \\
0, & k > n
\end{array}\right. \tag{2}
\end{equation}

步骤2: 构造辅助矩阵$M = A + \frac{1}{2}vv^* + \frac{1}{2}I$,其中$v = [\sqrt{1}, \sqrt{3}, \sqrt{5}, \cdots]^T$:
\begin{equation}
M_{n,k} = \left\{\begin{array}{ll}
-\frac{1}{2}\sqrt{2n+1}\sqrt{2k+1}, & k < n \\
0, & k = n \\
\frac{1}{2}\sqrt{2n+1}\sqrt{2k+1}, & k > n
\end{array}\right. \tag{3}
\end{equation}

步骤3: 观察到$M$是反对称矩阵,即$M^T = -M$。反对称矩阵的关键性质:
- 所有特征值都是纯虚数或零
- 一定可以被酉矩阵对角化
- 对角化后的特征值形式为$\pm i\lambda_k$

步骤4: 由谱定理,存在酉矩阵$U$和对角矩阵$\tilde{\Lambda}$使得:
\begin{equation}
M = U^* \tilde{\Lambda} U \tag{4}
\end{equation}

步骤5: 回代得到$A$的表示:
\begin{align}
A &= M - \frac{1}{2}I - \frac{1}{2}vv^* \tag{5} \\
&= U^* \tilde{\Lambda} U - \frac{1}{2}I - \frac{1}{2}vv^* \tag{6} \\
&= U^*\left(\tilde{\Lambda} - \frac{1}{2}I - \frac{1}{2}(Uv)(Uv)^*\right)U \tag{7} \\
&= U^*(\Lambda - uv^*)U \tag{8}
\end{align}
其中$\Lambda = \tilde{\Lambda} - \frac{1}{2}I$是对角矩阵,$u = Uv$。$\square$

#### 1.2 DPLR分解的数值稳定性

**引理1.2 (酉变换的数值稳定性)**: 酉矩阵$U$满足$U^*U = I$,其条件数$\kappa(U) = 1$,因此酉变换在数值计算中具有最优稳定性。

**证明**: 对于酉矩阵$U$,有:
\begin{equation}
\kappa(U) = \|U\| \cdot \|U^{-1}\| = \|U\| \cdot \|U^*\| = 1 \tag{9}
\end{equation}
最后一个等号利用了酉矩阵的性质$\|U\| = \|U^*\| = 1$。$\square$

**定理1.3 (DPLR与直接对角化的稳定性比较)**: 设$P$为对角化$A$的矩阵(即$A = P^{-1}\Lambda P$),则:
\begin{equation}
\kappa(P) \geq \mathcal{O}(2^{d/3}\sqrt{d}) \tag{10}
\end{equation}
而DPLR分解中的酉矩阵$U$满足$\kappa(U) = 1$。

**证明**: 从特征向量的计算可知,矩阵$P$的列向量包含指数级别的数值衰减。具体地,对于$\lambda = d/3$的特征值,其对应的系数$S_n$满足:
\begin{equation}
|S_{d/3}| \sim \mathcal{O}(\sqrt{d} \cdot 2^{-4d/3}) \tag{11}
\end{equation}

这意味着$P$的不同列之间存在指数级别的尺度差异,导致:
\begin{equation}
\kappa(P) = \frac{\sigma_{\max}(P)}{\sigma_{\min}(P)} \geq \mathcal{O}(2^{d/3}\sqrt{d}) \tag{12}
\end{equation}
$\square$

#### 1.3 DPLR分解的计算复杂度

**定理1.4 (DPLR分解的向量乘法复杂度)**: 对于$A = U^*(\Lambda - uv^*)U$形式的矩阵,计算$Ax$的复杂度为$\mathcal{O}(d)$。

**证明**: 分解计算过程:
\begin{align}
Ax &= U^*(\Lambda - uv^*)Ux \tag{13} \\
&= U^*(\Lambda(Ux) - u(v^*(Ux))) \tag{14}
\end{align}

计算步骤和复杂度:
1. 计算$y_1 = Ux$: 虽然$U$是$d \times d$矩阵,但由于FFT等快速算法,可在$\mathcal{O}(d\log d)$完成
2. 计算$s = v^*y_1$: $\mathcal{O}(d)$(内积)
3. 计算$y_2 = \Lambda y_1$: $\mathcal{O}(d)$(对角矩阵乘法)
4. 计算$y_3 = us$: $\mathcal{O}(d)$(标量乘向量)
5. 计算$y_4 = y_2 - y_3$: $\mathcal{O}(d)$
6. 计算$Ax = U^*y_4$: $\mathcal{O}(d\log d)$

总复杂度: $\mathcal{O}(d\log d)$,在实际中通常视为$\mathcal{O}(d)$。$\square$

### 二、Woodbury恒等式的详细证明

#### 2.1 基本Woodbury恒等式

**定理2.1 (Woodbury矩阵恒等式)**: 对于$U \in \mathbb{R}^{d \times r}$,$V \in \mathbb{R}^{d \times r}$,有:
\begin{equation}
(I - UV^*)^{-1} = I + U(I - V^*U)^{-1}V^* \tag{15}
\end{equation}

**证明**:

方法1 (直接验证): 我们验证$(I - UV^*) \cdot [I + U(I - V^*U)^{-1}V^*] = I$:
\begin{align}
&(I - UV^*)[I + U(I - V^*U)^{-1}V^*] \tag{16} \\
&= I + U(I - V^*U)^{-1}V^* - UV^* - UV^*U(I - V^*U)^{-1}V^* \tag{17} \\
&= I + U(I - V^*U)^{-1}V^* - UV^* - U(V^*U)(I - V^*U)^{-1}V^* \tag{18} \\
&= I + U[(I - V^*U)^{-1} - (V^*U)(I - V^*U)^{-1}]V^* - UV^* \tag{19} \\
&= I + U[(I - V^*U)(I - V^*U)^{-1}]V^* - UV^* \tag{20} \\
&= I + UV^* - UV^* = I \tag{21}
\end{align}

方法2 (级数展开): 当$\|UV^*\| < 1$时:
\begin{align}
(I - UV^*)^{-1} &= \sum_{k=0}^{\infty}(UV^*)^k \tag{22} \\
&= I + \sum_{k=1}^{\infty}(UV^*)^k \tag{23} \\
&= I + U\left(\sum_{k=0}^{\infty}(V^*U)^k\right)V^* \tag{24} \\
&= I + U(I - V^*U)^{-1}V^* \tag{25}
\end{align}
其中第(24)式利用了$(UV^*)^k = U(V^*U)^{k-1}V^*$。$\square$

#### 2.2 广义Woodbury恒等式

**定理2.2 (广义Woodbury恒等式)**: 对于可逆矩阵$M$和矩阵$U,V$,有:
\begin{equation}
(M - UV^*)^{-1} = M^{-1} + M^{-1}U(I - V^*M^{-1}U)^{-1}V^*M^{-1} \tag{26}
\end{equation}

**证明**: 分解计算:
\begin{align}
(M - UV^*)^{-1} &= (M(I - M^{-1}UV^*))^{-1} \tag{27} \\
&= (I - M^{-1}UV^*)^{-1}M^{-1} \tag{28} \\
&= [I + M^{-1}U(I - V^*M^{-1}U)^{-1}V^*]M^{-1} \tag{29} \\
&= M^{-1} + M^{-1}U(I - V^*M^{-1}U)^{-1}V^*M^{-1} \tag{30}
\end{align}
第(29)式应用了基本Woodbury恒等式(15)。$\square$

#### 2.3 在S4中的应用

**定理2.3 (S4中的Woodbury应用)**: 对于$A = \Lambda - uv^*$,离散化矩阵$\bar{A}$满足:
\begin{equation}
\bar{A} = (I - \epsilon(\Lambda - uv^*)/2)^{-1}(I + \epsilon(\Lambda - uv^*)/2) \tag{31}
\end{equation}
可以通过Woodbury恒等式高效计算。

**详细推导**:

步骤1: 计算$(I - \epsilon(\Lambda - uv^*)/2)^{-1}$。令$D = \frac{2}{\epsilon}I - \Lambda$,则:
\begin{align}
I - \epsilon(\Lambda - uv^*)/2 &= \frac{\epsilon}{2}(D + uv^*) \tag{32}
\end{align}

步骤2: 应用Woodbury恒等式:
\begin{align}
(D + uv^*)^{-1} &= D^{-1} - D^{-1}u(I + v^*D^{-1}u)^{-1}v^*D^{-1} \tag{33}
\end{align}

步骤3: 注意$D$是对角矩阵,$D^{-1}$的计算为$\mathcal{O}(d)$。设:
\begin{align}
\alpha &= v^*D^{-1}u \quad (\text{标量,复杂度}\mathcal{O}(d)) \tag{34} \\
\beta &= (1 + \alpha)^{-1} \quad (\text{标量倒数}) \tag{35} \\
w &= D^{-1}u \quad (\text{向量,复杂度}\mathcal{O}(d)) \tag{36} \\
z &= D^{-1}v \quad (\text{向量,复杂度}\mathcal{O}(d)) \tag{37}
\end{align}

步骤4: 则$(D + uv^*)^{-1}$可表示为:
\begin{equation}
(D + uv^*)^{-1} = D^{-1} - \beta w z^* \tag{38}
\end{equation}
这是"对角+秩1"的形式,总计算复杂度$\mathcal{O}(d)$。

步骤5: 完整的$\bar{A}$计算:
\begin{align}
\bar{A} &= \frac{2}{\epsilon}(D + uv^*)^{-1}(I + \epsilon(\Lambda - uv^*)/2) \tag{39} \\
&= \frac{2}{\epsilon}(D^{-1} - \beta wz^*)\left(\frac{\epsilon}{2}(D + uv^*) + \epsilon(\Lambda - uv^*)/2\right) \tag{40}
\end{align}

经过整理,这仍然保持"对角+低秩"形式,总复杂度$\mathcal{O}(d)$。$\square$

### 三、生成函数与卷积加速

#### 3.1 生成函数的基本理论

**定义3.1 (序列生成函数)**: 对于序列$\{a_k\}_{k=0}^{\infty}$,其生成函数定义为:
\begin{equation}
\mathcal{G}(z|a) = \sum_{k=0}^{\infty}a_k z^k \tag{41}
\end{equation}

**定理3.2 (卷积定理)**: 若$c_n = \sum_{k=0}^{n}a_k b_{n-k}$是$a$和$b$的卷积,则:
\begin{equation}
\mathcal{G}(z|c) = \mathcal{G}(z|a) \cdot \mathcal{G}(z|b) \tag{42}
\end{equation}

**证明**:
\begin{align}
\mathcal{G}(z|a)\mathcal{G}(z|b) &= \left(\sum_{k=0}^{\infty}a_k z^k\right)\left(\sum_{l=0}^{\infty}b_l z^l\right) \tag{43} \\
&= \sum_{k=0}^{\infty}\sum_{l=0}^{\infty}a_k b_l z^{k+l} \tag{44} \\
&= \sum_{n=0}^{\infty}\left(\sum_{k=0}^{n}a_k b_{n-k}\right)z^n \tag{45} \\
&= \sum_{n=0}^{\infty}c_n z^n = \mathcal{G}(z|c) \tag{46}
\end{align}
$\square$

#### 3.2 SSM卷积核的生成函数

**定理3.3 (SSM卷积核生成函数)**: 对于卷积核$\bar{K}_k = \bar{C}^*\bar{A}^k\bar{B}$,其生成函数为:
\begin{equation}
\mathcal{G}(z|\bar{K}) = \sum_{k=0}^{\infty}\bar{C}^*\bar{A}^k\bar{B}z^k = \bar{C}^*(I - z\bar{A})^{-1}\bar{B} \tag{47}
\end{equation}

**证明**: 利用几何级数:
\begin{align}
\mathcal{G}(z|\bar{K}) &= \sum_{k=0}^{\infty}\bar{C}^*\bar{A}^k\bar{B}z^k \tag{48} \\
&= \bar{C}^*\left(\sum_{k=0}^{\infty}(z\bar{A})^k\right)\bar{B} \tag{49} \\
&= \bar{C}^*(I - z\bar{A})^{-1}\bar{B} \tag{50}
\end{align}
其中第(49)到(50)利用了几何级数公式$\sum_{k=0}^{\infty}M^k = (I - M)^{-1}$(当$\|M\| < 1$时)。$\square$

#### 3.3 截断生成函数

**定义3.4 (截断生成函数)**: 对于长度为$L$的序列,截断生成函数为:
\begin{equation}
\mathcal{G}_L(z|\bar{K}) = \sum_{k=0}^{L-1}\bar{K}_k z^k \tag{51}
\end{equation}

**定理3.5 (截断生成函数的闭形式)**:
\begin{equation}
\mathcal{G}_L(z|\bar{K}) = \bar{C}^*(I - z^L\bar{A}^L)(I - z\bar{A})^{-1}\bar{B} \tag{52}
\end{equation}

**证明**: 利用几何级数的有限形式:
\begin{align}
\sum_{k=0}^{L-1}M^k &= (I - M^L)(I - M)^{-1} \tag{53}
\end{align}

代入$M = z\bar{A}$:
\begin{align}
\mathcal{G}_L(z|\bar{K}) &= \sum_{k=0}^{L-1}\bar{C}^*\bar{A}^k\bar{B}z^k \tag{54} \\
&= \bar{C}^*\left(\sum_{k=0}^{L-1}(z\bar{A})^k\right)\bar{B} \tag{55} \\
&= \bar{C}^*(I - z^L\bar{A}^L)(I - z\bar{A})^{-1}\bar{B} \tag{56}
\end{align}
$\square$

### 四、离散傅里叶变换(DFT)加速

#### 4.1 DFT的数学基础

**定义4.1 (离散傅里叶变换)**: 对于长度为$L$的序列$\{a_k\}_{k=0}^{L-1}$,其DFT定义为:
\begin{equation}
\hat{a}_l = \sum_{k=0}^{L-1}a_k e^{-2\pi i kl/L}, \quad l = 0,1,\cdots,L-1 \tag{57}
\end{equation}

**定义4.2 (逆DFT, IDFT)**:
\begin{equation}
a_k = \frac{1}{L}\sum_{l=0}^{L-1}\hat{a}_l e^{2\pi i kl/L}, \quad k = 0,1,\cdots,L-1 \tag{58}
\end{equation}

**定理4.3 (DFT的卷积定理)**: 若$c = a * b$是卷积,则:
\begin{equation}
\widehat{c} = \widehat{a} \odot \widehat{b} \tag{59}
\end{equation}
其中$\odot$表示逐点乘法。

**证明**: 设$c_n = \sum_{k=0}^{n}a_k b_{n-k}$(循环卷积):
\begin{align}
\hat{c}_l &= \sum_{n=0}^{L-1}c_n e^{-2\pi inl/L} \tag{60} \\
&= \sum_{n=0}^{L-1}\left(\sum_{k=0}^{L-1}a_k b_{(n-k)\bmod L}\right)e^{-2\pi inl/L} \tag{61} \\
&= \sum_{k=0}^{L-1}a_k e^{-2\pi ikl/L}\sum_{n=0}^{L-1}b_{(n-k)\bmod L}e^{-2\pi i(n-k)l/L} \tag{62} \\
&= \hat{a}_l \cdot \hat{b}_l \tag{63}
\end{align}
$\square$

#### 4.2 DFT计算生成函数

**定理4.4 (生成函数与DFT的关系)**: 对于$z = e^{-2\pi il/L}$,生成函数值就是DFT:
\begin{equation}
\mathcal{G}(e^{-2\pi il/L}|a) = \sum_{k=0}^{L-1}a_k e^{-2\pi ikl/L} = \hat{a}_l \tag{64}
\end{equation}

**推论4.5 (通过DFT计算卷积核)**:
\begin{align}
\text{DFT}(\bar{K}) &= \{\mathcal{G}_L(e^{-2\pi il/L}|\bar{K})\}_{l=0}^{L-1} \tag{65} \\
&= \left\{\bar{C}^*(I - e^{-2\pi il/L}\bar{A})^{-1}\bar{B}\right\}_{l=0}^{L-1} \tag{66}
\end{align}

其中第二个等号利用了$e^{-2\pi iL/L} = 1$,使得$(I - z^L\bar{A}^L) = (I - \bar{A}^L)$可以吸收进$\bar{C}$。

#### 4.3 零填充(Zero-Padding)技术

**问题**: 直接DFT会导致循环卷积,而我们需要线性卷积。

**解决方案**: 零填充——将长度$L$的序列扩展为长度$2L$:
\begin{align}
\bar{a} &\to (\bar{a}, 0, 0, \cdots, 0) \tag{67} \\
\bar{b} &\to (\bar{b}, 0, 0, \cdots, 0) \tag{68}
\end{align}

**定理4.6 (零填充的正确性)**: 对长度为$2L$的零填充序列进行DFT,然后逐点相乘,再IDFT,前$L$个结果正好是线性卷积。

**证明**: 零填充使得循环卷积的"环绕"部分落在后$L$个位置,不影响前$L$个结果。$\square$

### 五、Cauchy核加速算法

#### 5.1 Cauchy核问题的定义

在S4中,我们需要计算:
\begin{equation}
\mathcal{G}_L(e^{-2\pi il/L}|\bar{K}) = \bar{C}^*(I - e^{-2\pi il/L}\bar{A})^{-1}\bar{B}, \quad l = 0,1,\cdots,L-1 \tag{69}
\end{equation}

对于$A = \Lambda - uv^*$(对角+秩1),应用Woodbury恒等式后,核心问题变为计算Cauchy矩阵:

**定义5.1 (Cauchy矩阵)**: 矩阵$C \in \mathbb{C}^{L \times d}$定义为:
\begin{equation}
C_{lk} = \frac{1}{\omega_l - \lambda_k} \tag{70}
\end{equation}
其中$\omega_l = e^{-2\pi il/L}$,$\lambda_k$是$\Lambda$的对角元素。

#### 5.2 快速多极算法(FMM)加速

**定理5.2 (Cauchy矩阵-向量乘法复杂度)**: 使用快速多极算法,Cauchy矩阵与向量的乘法可在$\mathcal{O}((L+d)\log^2(L+d))$完成。

标准算法复杂度为$\mathcal{O}(Ld)$,FMM通过分层结构将其降低到近线性复杂度。

**算法框架**:
1. 构建$\omega$和$\lambda$的层次化分组
2. 对每组使用多极展开近似
3. 递归计算各层贡献
4. 合并结果

#### 5.3 与FFT复杂度的对比

| 方法 | 复杂度 | 依赖性 |
|------|--------|--------|
| 朴素计算 | $\mathcal{O}(Ld)$ | 依赖$L$和$d$ |
| S4+Cauchy | $\mathcal{O}((L+d)\log^2(L+d))$ | 依赖$L$和$d$ |
| RFT+DFT | $\mathcal{O}(L\log L)$ | 仅依赖$L$ |

### 六、数值稳定性分析

#### 6.1 条件数与误差传播

**定义6.1 (条件数)**: 矩阵$M$的条件数定义为:
\begin{equation}
\kappa(M) = \|M\| \cdot \|M^{-1}\| = \frac{\sigma_{\max}(M)}{\sigma_{\min}(M)} \tag{71}
\end{equation}

**定理6.3 (误差放大估计)**: 若计算$y = Mx$时输入误差为$\delta x$,则输出误差满足:
\begin{equation}
\frac{\|\delta y\|}{\|y\|} \leq \kappa(M)\frac{\|\delta x\|}{\|x\|} + \mathcal{O}(\epsilon_{\text{machine}}) \tag{72}
\end{equation}

#### 6.2 DPLR分解的稳定性优势

**定理6.4 (DPLR的数值稳定性)**: DPLR分解$A = U^*(\Lambda - uv^*)U$中:
\begin{align}
\kappa(U) &= 1 \tag{73} \\
\kappa(\Lambda) &= \frac{|\lambda_{\max}|}{|\lambda_{\min}|} \tag{74}
\end{align}

对于HiPPO-LegS,$\lambda_k \in [-1,-2,\cdots,-d]$,故$\kappa(\Lambda) = d$,远小于直接对角化的$\kappa(P) \sim 2^{d/3}$。

#### 6.3 双线性变换的数值特性

**定理6.5 (双线性变换的A-稳定性)**: 双线性离散化:
\begin{equation}
\bar{A} = (I - \epsilon A/2)^{-1}(I + \epsilon A/2) \tag{75}
\end{equation}
是A-稳定的,即对于$\text{Re}(\lambda(A)) < 0$的所有特征值,都有$|\lambda(\bar{A})| < 1$。

**证明**: 设$\lambda$是$A$的特征值,$\text{Re}(\lambda) < 0$。对应的$\bar{A}$的特征值为:
\begin{align}
\bar{\lambda} &= \frac{1 + \epsilon\lambda/2}{1 - \epsilon\lambda/2} \tag{76}
\end{align}

计算模长:
\begin{align}
|\bar{\lambda}|^2 &= \frac{|1 + \epsilon\lambda/2|^2}{|1 - \epsilon\lambda/2|^2} \tag{77} \\
&= \frac{(1 + \epsilon\text{Re}(\lambda)/2)^2 + (\epsilon\text{Im}(\lambda)/2)^2}{(1 - \epsilon\text{Re}(\lambda)/2)^2 + (\epsilon\text{Im}(\lambda)/2)^2} \tag{78}
\end{align}

当$\text{Re}(\lambda) < 0$时,分子$< $分母,故$|\bar{\lambda}| < 1$。$\square$

### 七、完整算法流程

#### 7.1 训练阶段算法

**算法7.1 (S4训练)**:
```
输入: 序列u₀, u₁, ..., u_{L-1}
参数: A (HiPPO-LegS), B, C, ε (步长)

1. 预计算:
   a. 对角化辅助矩阵: M = A + vv*/2 + I/2 → U, Λ̃
   b. 构造 Λ = Λ̃ - I/2, u = Uv
   c. 计算 Ā = (I - εA/2)⁻¹(I + εA/2) (通过Woodbury)
   d. 计算 B̄ = ε(I - εA/2)⁻¹B
   e. 计算 C̃ = C*(I - Ā^L)

2. 生成函数计算:
   对 l = 0, 1, ..., L-1:
     z_l = exp(-2πil/L)
     R_z = (2/ε)(1-z_l)/(1+z_l)I - Λ
     应用Woodbury: G_l = (2/(1+z_l))C̃*(R_z⁻¹ - ...)B̄

3. FFT加速:
   K̂ = {G_l}_{l=0}^{L-1}
   K = IDFT(K̂)

4. 卷积计算:
   û = DFT([u₀,...,u_{L-1}, 0,...,0])  // 零填充
   K̂_pad = DFT([K₀,...,K_{L-1}, 0,...,0])
   ŷ = K̂_pad ⊙ û
   y = IDFT(ŷ)[0:L]  // 取前L个元素

输出: y₀, y₁, ..., y_{L-1}
```

时间复杂度: $\mathcal{O}(L\log L + (L+d)\log^2(L+d))$

#### 7.2 推理阶段算法

**算法7.2 (S4推理)**:
```
输入: 当前输入u_k, 当前状态x_k
参数: Ā, B̄, C̄

1. 状态更新:
   x_{k+1} = Āx_k + B̄u_k

2. 输出计算:
   y_{k+1} = C̄*x_{k+1}

输出: y_{k+1}, x_{k+1}
```

时间复杂度: $\mathcal{O}(d\log d)$(利用DPLR结构)

### 八、实践考虑

#### 8.1 超参数选择

1. **步长$\epsilon$**:
   - 理论上$\epsilon \sim 1/L$较优
   - 实践中可设为可学习参数
   - 初始值: $\epsilon = 0.001 \sim 0.1$

2. **状态维度$d$**:
   - 较大的$d$提供更强表达能力
   - 需权衡计算成本
   - 典型值: $d = 64, 128, 256$

3. **序列长度$L$**:
   - S4支持长序列($L > 10000$)
   - 零填充至$2L$用于FFT

#### 8.2 数值技巧

1. **矩阵指数计算**: 使用Padé近似或特征值方法
2. **逆矩阵计算**: 利用Woodbury避免显式求逆
3. **复数运算**: 注意精度损失,必要时使用高精度库

#### 8.3 常见问题

**问题1**: $\bar{A}^L$计算成本高
**解决**: 训练时用$\tilde{C} = C(I - \bar{A}^L)$作为参数,推理时再恢复$C$

**问题2**: 大$d$时Cauchy核计算慢
**解决**: 使用RFT方法,完全消除对$d$的依赖

**问题3**: 梯度消失/爆炸
**解决**: 使用双线性离散化,确保A-稳定性

### 九、与其他方法的对比

#### 9.1 S4 vs RNN

| 特性 | S4 | 传统RNN |
|------|-----|---------|
| 记忆衰减 | 多项式 | 指数 |
| 并行训练 | 是(FFT) | 否 |
| 长序列 | 优秀($L>10^4$) | 困难 |
| 参数量 | $\mathcal{O}(d^2)$ | $\mathcal{O}(d^2)$ |

#### 9.2 S4 vs Transformer

| 特性 | S4 | Transformer |
|------|-----|-------------|
| 复杂度(训练) | $\mathcal{O}(L\log L)$ | $\mathcal{O}(L^2d)$ |
| 复杂度(推理) | $\mathcal{O}(d)$ | $\mathcal{O}(L)$ |
| 归纳偏置 | 线性系统 | 注意力 |
| 长序列 | 优秀 | 受限 |

### 十、总结与展望

#### 10.1 核心贡献

S4的关键创新:
1. DPLR分解 — 解决HiPPO矩阵的数值稳定性
2. Woodbury恒等式 — 实现高效逆矩阵计算
3. Cauchy核加速 — 降低计算复杂度
4. 理论保证 — A-稳定性和多项式记忆衰减

#### 10.2 局限性

1. DPLR分解仍需计算$d$维矩阵
2. Cauchy核加速实现复杂
3. 超参数调优需要经验

#### 10.3 后续发展

- **S5**: 简化为单个SSM处理多通道
- **Mamba**: 简化$A$为对角矩阵,引入选择性机制
- **RFT**: 完全在有理函数空间参数化

S4为SSM的发展奠定了坚实的数学基础,其核心技术在后续工作中得到继承和发展。

