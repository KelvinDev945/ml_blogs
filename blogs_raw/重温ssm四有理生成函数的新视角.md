---
title: 重温SSM（四）：有理生成函数的新视角
slug: 重温ssm四有理生成函数的新视角
date: 2024-06-27
tags: 生成函数, 线性, RNN, ssm, 生成模型
status: completed
---

# 重温SSM（四）：有理生成函数的新视角

**原文链接**: [https://spaces.ac.cn/archives/10180](https://spaces.ac.cn/archives/10180)

**发布日期**: 

---

在前三篇文章中，我们较为详细地讨论了HiPPO和S4的大部分数学细节。那么，对于接下来的第四篇文章，大家预期我们会讨论什么工作呢？S5、Mamba乃至Mamba2？都不是。本系列文章主要关心SSM的数学基础，旨在了解SSM的同时也补充自己的数学能力。而在上一篇文章我们简单提过S5和Mamba，S5是S4的简化版，相比S4基本上没有引入新的数学技巧，而Mamba系列虽然表现优异，但它已经将$A$简化为对角矩阵，所用到的数学技巧就更少了，它更多的是体现了工程方面的能力。

这篇文章我们来学习一篇暂时还声名不显的新工作[《State-Free Inference of State-Space Models: The Transfer Function Approach》](https://papers.cool/arxiv/2405.06147)（简称RFT），它提出了一个新方案，将SSM的训练、推理乃至参数化，都彻底转到了生成函数空间中，为SSM的理解和应用开辟了新的视角

## 基础回顾 #

首先我们简单回顾一下上一篇文章关于S4的探讨结果。S4基于如下线性RNN  
\begin{equation}\begin{aligned}  
x_{k+1} =&\, \bar{A} x_k + \bar{B} u_k \\\  
y_{k+1} =&\, \bar{C}^* x_{k+1} \\\  
\end{aligned}\label{eq:linear}\end{equation}  
其中$u,y\in\mathbb{R},x\in\mathbb{R}^d,\bar{A}\in\mathbb{R}^{d\times d},\bar{B},\bar{C}\in\mathbb{R}^{d\times 1}$，这里旨在做一般化的讨论，所以我们绕过了$\bar{A}$与$A$的联系，假设$\bar{A}$为一般矩阵。设初始状态为零，那么直接迭代可以写出：  
\begin{equation}y_L = \sum_{k=0}^L \bar{C}^*\bar{A}^k \bar{B}u_{L-k} = \bar{K}_{< L} * u_{< L} \end{equation}

其中$*$是卷积运算，而  
\begin{equation}\bar{K}_k = \bar{C}^*\bar{A}^k\bar{B},\quad \bar{K}_{< L} = \big(\bar{K}_0,\bar{K}_1,\cdots,\bar{K}_{L-1}\big),\quad u_{< L} = (u_0,u_1,\cdots,u_{L-1})\end{equation}  
由于卷积可以通过离散傅里叶变换（DFT）高效计算，所以剩下的问题是如何高效地将$\bar{K}$算出来，这就是S4的核心贡献。为此，S4引入了生成函数  
\begin{align}\mathcal{G}(z|\bar{K}) =&\, \sum_{k=0}^{\infty} \bar{C}^*\bar{A}^k \bar{B}z^k = \bar{C}^*\left(I - z\bar{A}\right)^{-1}\bar{B} \\\  
\mathcal{G}_L(z|\bar{K}) =&\, \sum_{k=0}^{L-1} \bar{C}^*\bar{A}^k \bar{B}z^k = \bar{C}^*(I - z^L\bar{A}^L)\left(I - z\bar{A}\right)^{-1}\bar{B} \end{align}  
如果能够高效地计算$\mathcal{G}_L(z|\bar{K})$，那么就可以代入$z=e^{-2i\pi l/L},l=0,1,2,\dots,L-1$，其结果就是$\bar{K}$的DFT，于是进一步逆变换（IDFT）后就可以得到$\bar{K}$，由于此时的$z$总满足$z^L=1$，所以我们也可以设$\tilde{C}^* = \bar{C}^*(I - \bar{A}^L)$，那么$\mathcal{G}_L(z|\bar{K})$的形式就跟$\mathcal{G}(z|\bar{K})$一致了：  
\begin{equation}\mathcal{G}_L(z|\bar{K}) = \tilde{C}^*\left(I - z\bar{A}\right)^{-1}\bar{B}\end{equation}

那怎么高效计算$\mathcal{G}(z|\bar{K})$或者$\mathcal{G}_L(z|\bar{K})$呢？S4将$\bar{A}$分解为“对角+低秩”的形式，然后通过Woodbury恒等式进行计算，其最终结果为  
\begin{equation}\mathcal{G}(z|\bar{K}) = \frac{2}{1+z}\bar{C}^* \left[R_z^{-1} - R_z^{-1}u(I + v^*R_z^{-1}u)^{-1} v^*R_z^{-1}\right]B\end{equation}  
其中$R_z$是$d\times d$的对角阵，$u,v,B,\bar{C}$都是$d\times 1$的列向量，这意味着给定$z$计算$\mathcal{G}(z|\bar{K})$的复杂度为$\mathcal{O}(d)$，而如果要对$z=e^{-2i\pi l/L},l=0,1,2,\dots,L-1$进行计算，那么朴素实现的计算量是$\mathcal{O}(Ld)$。S4提出可以将其转化为Cauchy核问题进行计算，复杂度进一步降低到$\mathcal{O}((L+d)\log^2(L+d))$。

不管哪一种，我们可以发现其复杂度不仅依赖于$L$，还依赖于$d$（state size），而RFT则提出了一种新方法，将复杂度直接降低到了最理想的$\mathcal{O}(L\log L)$，不依赖state size，而且推导过程相比S4还明显简化，同时也不依赖于$\bar{A}$是对角阵或者“对角+低秩”的假设。

## 有理函数 #

RFT是Rational Transfer Function的缩写，它的重点是Rational Function，即我们所说的有理函数（两个多项式相除）。它跟生成函数有什么关系呢？RFT的作者们非常高明地观察到，$\mathcal{G}_L(z|\bar{K})$实际上是一个有理函数！具体来说，我们有  
\begin{equation}\mathcal{G}_L(z|\bar{K}) = \tilde{C}^*\left(I - z\bar{A}\right)^{-1}\bar{B} = \frac{b_1 + b_2 z + b_3 z^2 + \cdots + b_dz^{d-1}}{1 + a_1 z + a_2 z^2 + \cdots + a_d z^d}\label{eq:gz-rf}\end{equation}  
其中$a_1,a_2,\cdots,a_d,b_1,b_2,\cdots,b_d$都是标量，如果$\bar{A},\bar{B},\tilde{C}$都是实矩阵，那么它们都是实数。如果单纯想要意识到存在这么个相等的形式，我们只需要利用矩阵求逆的一个经典公式：  
\begin{equation}M^{-1} = \frac{\text{adj}(M)}{\det(M)}\end{equation}  
其中$\det(M)$是$M$的行列式，而$\text{adj}(M)$是$M$的[伴随矩阵](https://en.wikipedia.org/wiki/Adjugate_matrix)，由于伴随矩阵涉及到大量行列式计算，所以这个求逆公式在实际计算中通常没什么价值，但在理论分析时通常能起到奇效。比如，我们将它代入到$\mathcal{G}_L(z|\bar{K})$中，就得到  
\begin{equation}\mathcal{G}_L(z|\bar{K}) = \tilde{C}^*\left(I - z\bar{A}\right)^{-1}\bar{B} = \frac{\tilde{C}^*\text{adj}(I - z\bar{A})\bar{B}}{\det(I - z\bar{A})}\end{equation}  
我们知道，$d$阶行列式多项$d$个元素相乘的求和，所以$\det(I - z\bar{A})$是关于$z$的$d$次多项式；接着根据伴随矩阵的定义，它的每个元素都是$d-1$阶行列式，也就是$d-1$次多项式，左乘$\tilde{C}^*$和右乘$\bar{B}$只不过是将这些元素加权求和，所以结果还是$d-1$次多项式。因此，$\mathcal{G}_L(z|\bar{K})$是$z$的$d-1$次行列式除以$z$的$d$次行列式，再将分母的常数项系数标准化为$1$，就得到了式$\eqref{eq:gz-rf}$。

## 对应关系 #

进一步，我们可以利用一个行列式恒等式来确定系数$a=(a_1,a_2,\cdots,a_d),b=(b_1,b_2,\cdots,b_d)$与$\bar{A},\bar{B},\tilde{C}$的关系。这个恒等式是  
\begin{equation}\det(I + UV) = \det(I + VU)\label{eq:det-iuv}\end{equation}  
直接证明这个行列式不难，算是一道普通的考研题，只需要注意到  
\begin{equation}\begin{pmatrix}I & U \\\ -V & I\end{pmatrix} = \begin{pmatrix}I + UV & U \\\ 0 & I\end{pmatrix}\begin{pmatrix}I & 0 \\\ -V & I\end{pmatrix} = \begin{pmatrix}I & 0 \\\ -V & I + VU\end{pmatrix}\begin{pmatrix}I & U \\\ 0 & I\end{pmatrix}\end{equation}  
根据行列式的定义和形式，中间部份的行列式就是$\det(I + UV)$，最右边的行列式就是$\det(I + VU)$，它们同一个矩阵的行列式，所以结果相等。这个结果还可以进一步推广到（当$A,D$都可逆时）  
\begin{equation}\det\begin{pmatrix}A & B \\\ C & D\end{pmatrix} = \det(A)\det(D-CA^{-1}B) = \det(D)\det(A-BD^{-1}C)\end{equation}  
更远一点的话，它还可以一般化为我们在[《两个多元正态分布的KL散度、巴氏距离和W距离》](/archives/8512)提过的“[舒尔补（Schur complement）](https://en.wikipedia.org/wiki/Schur_complement)”理论。

回到正题，注意在式$\eqref{eq:det-iuv}$及其推导中，我们不需要假设$U,V$都是方阵，所以实际上式$\eqref{eq:det-iuv}$对于非方阵也是成立的，只要单位阵$I$自动匹配$UV$和$VU$的大小就行。特别地，如果$U,V$分别是列、行向量，那么$VU$就是一个标量，对应的$I$就是1，其行列式就是自身，即$\det(I + UV) = 1 + VU$。利用这个特例，我们有  
\begin{equation}\begin{aligned}  
\mathcal{G}_L(z|\bar{K}) =&\, z^{-1}\left[1+\tilde{C}^*\left(z^{-1}I - \bar{A}\right)^{-1}\bar{B} - 1 \right]\\\  
=&\, z^{-1}\left[\det\left(I + \left(z^{-1} I - \bar{A}\right)^{-1}\bar{B}\tilde{C}^*\right) - 1\right]\\\  
=&\, z^{-1}\left\\{\det\left[\left(z^{-1} I - \bar{A}\right)^{-1}\left(z^{-1} I - \bar{A} + \bar{B}\tilde{C}^*\right)\right] - 1\right\\} \\\  
=&\, z^{-1}\left[\frac{\det(z^{-1} I - \bar{A} + \bar{B}\tilde{C}^*)}{\det(z^{-1} I - \bar{A})} - 1\right] \\\  
=&\, \frac{z^{d-1}\left[\det(z^{-1} I - \bar{A} + \bar{B}\tilde{C}^*)-\det(z^{-1} I - \bar{A})\right]}{z^d\det(z^{-1} I - \bar{A})} \\\  
\end{aligned}\end{equation}  
分母中的$\det(z^{-1} I - \bar{A})$，是以$\lambda=z^{-1}$为变量的矩阵$\bar{A}$的特征多项式，它是$z^{-1}$的$d$次首一多项式，乘以$z^d$后变成了$z$的常数项为1的$d$次多项式；同理，分子中的$\det(z^{-1} I - \bar{A} + \bar{B}\tilde{C}^*)$是$\bar{A} - \bar{B}\tilde{C}^*$的特征多项式（$z^{-1}$的$d$次首一多项式），减去$\det(z^{-1} I - \bar{A})$后正好得到$z^{-1}$的$d-1$次多项式，乘以$z^{d-1}$变成$z$的$d-1$次多项式。所以，$a$向量正好是多项式$\det(\lambda I - \bar{A})$除最高次项外的各系数，而$b$向量则是多项式$\det(\lambda I - \bar{A} + \bar{B}\tilde{C}^*)-\det(\lambda I - \bar{A})$的各系数（次数从高到低排序）。

## 惊喜突现 #

现在我们先缓一缓，思考一下我们做了什么，要往哪里去。

我们的出发点是线性系统$\eqref{eq:linear}$，为了让它可以并行训练，我们将其转化为了$\bar{K}_{< L}$与$u_{< L}$的卷积，就可以通过先DFT后相乘再IDFT来高效计算，因此这一步的效率不成问题。现在$u_{< L}$是现成的，但$\bar{K}_{< L}$未知，所以问题变成了如何高效计算卷积核$\bar{K}_{< L}=\\{\tilde{C}^*\bar{A}^k\bar{B}\\}_{k=0}^{L-1}$，为此我们进一步引入了生成函数$\mathcal{G}_L(z|\bar{K})$，只要能够高效计算$\mathcal{G}_L(z|\bar{K})$，那么就有  
\begin{equation}DFT(\bar{K}_{< L}) = \Big\\{\mathcal{G}_L(z|\bar{K})\Big\\}_{z=e^{-2i\pi l/L},l=0,1,2,\dots,L-1}\end{equation}  
然后IDFT就可以恢复原本的$\bar{K}_{< L}$。对于$z=e^{-2i\pi l/L}$，我们有$z^L=1$，于是  
\begin{equation}\mathcal{G}_L(z|\bar{K}) = \tilde{C}^*(I - z^L\bar{A}^L)\left(I - z\bar{A}\right)^{-1}\bar{B} = \underbrace{\bar{C}^*(I - \bar{A}^L)}_{\tilde{C}^*}\left(I - z\bar{A}\right)^{-1}\bar{B} \end{equation}  
也就是我们可以先将整个$\bar{C}^*(I - \bar{A}^L)$视为训练参数$\tilde{C}^*$，事后再解出对应的$\bar{C}$用于推理。

S4通过“对角+低秩”的分解来计算$\mathcal{G}_L(z|\bar{K})$，而这篇文章则指出$\mathcal{G}_L(z|\bar{K})$实际上是一个有理函数，即式$\eqref{eq:gz-rf}$。如果我们此时代入$z=e^{-2i\pi l/L}$，就会发现一些让人惊喜的结果，比如分母  
\begin{equation}1 + a_1 z + a_2 z^2 + \cdots + a_d z^d = \sum_{k=0}^L a_k z^k = \sum_{k=0}^L a_k e^{-2i\pi kl/L} = DFT(\bar{a}_{< L})\end{equation}  
其中$\bar{a}_{< L} = (a_0,a_1,a_2,\cdots,a_{L-1}) = (1, a_1,a_2, \cdots, a_d, 0, \cdots, 0)$，也就是说，根据定义分母就是将$a$左边拼一个1、后边拼若干个0、凑成$L$个数后的DFT！同理，定义$\bar{b}_{< L} = (b_1,b_2,\cdots,b_d,0,0,\cdots,0)$，那么分子就是$DFT(\bar{b}_{< L})$，于是我们可以简单地写出  
\begin{equation}DFT(\bar{K}_{< L}) = \frac{DFT(\bar{b}_{< L})}{DFT(\bar{a}_{< L})} = \frac{DFT(b_1,b_2,\cdots,b_d,0,0,\cdots,0)}{DFT(1, a_1,a_2, \cdots, a_d, 0, \cdots, 0)}\end{equation}  
然后IDFT就可以得到$\bar{K}_{< L}$，其中DFT和IDFT的计算复杂度都是$\mathcal{O}(L\log L)$，跟$d$无关（只需要$d < L$）！这就是RTF的复杂度与state size大小$d$无关的核心思想。

## 另起炉灶 #

按照上面的引入顺序，我们的计算过程应该是先给定$\bar{A},\bar{B},\tilde{C}$，然后计算$\bar{A}$和$\bar{A}-\bar{B}\tilde{C}^*$的特征多项式系数，进而得到$a_1,a_2, \cdots, a_d$和$b_1,b_2,b_3,\cdots,b_d$，最后计算DFT、相除然后IDFT来得到$\bar{K}_{< L}$。如果是单纯的计算，那么这个过程没啥问题，但我们面对的是训练场景，$\bar{A},\bar{B},\tilde{C}$可能带有训练参数，这时候计算$\bar{A}$和$\bar{A}-\bar{B}\tilde{C}^*$的特征多项式这一步就不那么容易传播梯度了。

对于这个问题，更加干脆的方案是“另起炉灶”——直接以RTF形式的式$\eqref{eq:gz-rf}$为出发点，将$a=(a_1,a_2, \cdots, a_d)$和$b=(b_1,b_2,b_3,\cdots,b_d)$设为可训练参数，那么我们连特征多项式的计算都省了，直接就可以DFT和IDFT去算$\bar{K}_{\leq L}$。不仅如此，原本$\bar{A},\bar{B},\tilde{C}$共有$d^2+2d$个参数，现在$a,b$两个向量一共就$2d$个参数，大大节省了参数量。而因为任意的$\bar{A},\bar{B},\tilde{C}$都可以算出对应的$a,b$，所以RFT的理论能力是不差于原始的RNN形式的。

当然，RTF只是提供了一种直接以$a,b$为参数的高效训练的方式，如果要做step by step推理，那么还是要转回RNN形式，这意味着给定训练好的$a,b$，我们要找出一组$\bar{A},\bar{B},\tilde{C}$，然后代入式$\eqref{eq:linear}$来推理。注意$a,b\to\bar{A},\bar{B},\tilde{C}$是$2d$个参数到$d^2+2d$个参数的映射，肯定有无穷多组解，而我们只需要找出尽可能简单的一组解就行了。

## 友之矩阵 #

怎么求这组解呢？前面我们已经证明了，$a$向量正好是多项式$\det(z I - \bar{A})$除最高次项外的各系数，所以给定$a$求$\bar{A}$，就是已知特征多项式的情况下求对应的矩阵，最简单的解是对角矩阵，假设$\lambda_1,\lambda_2,\cdots,\lambda_d$为$\lambda^d + a_1 \lambda^{d-1} + a_2 \lambda^{d-2} + \cdots + a_d =0$的$d$个根，那么让$\bar{A}=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_d)$即可。不过，这样可能会出现虚数根，某种程度上可能不够简洁，同时这种纯粹的形式解也无法直接观察$\bar{A}$与$a$之间的联系。

事实上，求一个实矩阵使其特征多项式为给定的实系数多项式，这个问题早有研究，其答案有一个有趣的名字，叫做“[友矩阵（Companion matrix）](https://en.wikipedia.org/wiki/Companion_matrix)”，其形式为（为了对齐原论文的结果，这里相比维基百科的格式多了个翻转）：  
\begin{equation}\bar{A} = \begin{pmatrix}-a_1 & \- a_2 & \cdots & -a_{d-1} & -a_d \\\  
1 & 0 & \cdots & 0 & 0 \\\  
0 & 1 & \cdots & 0 & 0 \\\  
\vdots & \vdots & \ddots & \vdots & \vdots \\\  
0 & 0 & \cdots & 1 & 0 \\\  
\end{pmatrix}\label{eq:bar-A}\end{equation}  
事后去证明该矩阵满足  
\begin{equation}\det(\lambda I-\bar{A})=\lambda^d + a_1 \lambda^{d-1} + a_2 \lambda^{d-2} + \cdots + a_d\end{equation}  
并不难，直接根据行列式的定义对$\det(\lambda I-\bar{A})$的第一行展开即可。更深刻的问题是如何想到这个构造，这里笔者提供自己的想法。根据特征多项式来构造矩阵，本质上就是逐渐将多项式变换为一个$\lambda$只出现在对角线上的行列式，比如$d=2$时我们有  
\begin{equation}\lambda^2 + a_1 \lambda + a_2 = (\lambda + a_1)\lambda - (-1) \times a_2  
) = \det\begin{pmatrix} \lambda + a_1 & a_2 \\\ -1 & \lambda\end{pmatrix}\end{equation}  
这就可以抽出对应的$\bar{A}$。对于一般的$d$，我们有  
\begin{equation}\lambda^d + a_1 \lambda^{d-1} + a_2 \lambda^{d-2} + \cdots + a_d = \det\begin{pmatrix} \lambda^{d-1} + a_1 \lambda^{d-2} + \cdots + a_{d-1} & a_d \\\ -1 & \lambda\end{pmatrix}\end{equation}  
这当然还不是最终答案，但这成功将多项式的次数减少了一，这启示我们或许可以考虑递归地构建，即左上角再以$\lambda^{d-1} + a_1 \lambda^{d-2} + \cdots + a_{d-1}$为特征多项式构造原矩阵，然后微调一下右上和左下的行列，形成分块矩阵。细心多尝试一下，就有机会自己构造出式$\eqref{eq:bar-A}$的结果。

有了$\bar{A}$，构造$\bar{B},\tilde{C}$就容易多了。还是根据前面的结论，我们有  
\begin{equation}\begin{gathered}  
\det(\lambda I - \bar{A} + \bar{B}\tilde{C}^*)-\det(\lambda I - \bar{A}) = b_1 \lambda^{d-1} + b_2 \lambda^{d-2} + \cdots + b_d \\\  
\Downarrow \\\  
\det(\lambda I - \bar{A} + \bar{B}\tilde{C}^*)= \lambda^d + (a_1 + b_1) \lambda^{d-1} + (a_2+b_2) \lambda^{d-2} + \cdots + (a_d + b_d)  
\end{gathered}\end{equation}  
也就是$\bar{A} - \bar{B}\tilde{C}^*$的特征多项式为上式，那么根据$\bar{A}$的构造方式，我们得到$\bar{A} - \bar{B}\tilde{C}^*$的一个解是  
\begin{equation}\bar{A} - \bar{B}\tilde{C}^* = \begin{pmatrix}-a_1 - b_1 & \- a_2 - b_2 & \cdots & -a_{d-1} - b_{d-1} & -a_d - b_d\\\  
1 & 0 & \cdots & 0 & 0 \\\  
0 & 1 & \cdots & 0 & 0 \\\  
\vdots & \vdots & \ddots & \vdots & \vdots \\\  
0 & 0 & \cdots & 1 & 0 \\\  
\end{pmatrix}\end{equation}  
于是  
\begin{equation}\bar{B}\tilde{C}^* = \begin{pmatrix}b_1 & b_2 & \cdots & b_{d-1} & b_d\\\  
0 & 0 & \cdots & 0 & 0 \\\  
0 & 0 & \cdots & 0 & 0 \\\  
\vdots & \vdots & \ddots & \vdots & \vdots \\\  
0 & 0 & \cdots & 0 & 0 \\\  
\end{pmatrix} = \begin{pmatrix}1 \\\ 0 \\\ \vdots \\\ 0 \\\ 0\end{pmatrix}\begin{pmatrix}b_1 & b_2 & \cdots & b_{d-1} & b_d\end{pmatrix}\end{equation}  
这意味着我们可以找到一组解$\bar{B} = [1, 0, \cdots, 0, 0], \tilde{C}^* = [b_1 , b_2 , \cdots , b_{d-1} , b_d]$，然后进一步解得$\bar{C}^* = \tilde{C}^*(I - \bar{A}^L)^{-1}$。

## 初始方式 #

我们来完整地$x_k$的递归形式：  
\begin{equation}  
x_{k+1} = \begin{pmatrix}-a_1 & \- a_2 & \cdots & -a_{d-1} & -a_d \\\  
1 & 0 & \cdots & 0 & 0 \\\  
0 & 1 & \cdots & 0 & 0 \\\  
\vdots & \vdots & \ddots & \vdots & \vdots \\\  
0 & 0 & \cdots & 1 & 0 \\\  
\end{pmatrix} x_k + \begin{pmatrix}1 \\\ 0 \\\ \vdots \\\ 0 \\\ 0\end{pmatrix} u_k = \begin{pmatrix} u_k - \langle a, x_k\rangle \\\ x_{k,(0)} \\\ x_{k,(1)} \\\ \vdots \\\ x_{k,(d-3)} \\\ x_{k,(d-2)}\end{pmatrix}  
\end{equation}  
由于$\bar{A}$极其稀疏的特点，每一步递归可以在$\mathcal{O}(d)$而不是$\mathcal{O}(d^2)$完成。特别地，当$a_1=a_2=\cdots=a_d=0$时，我们可以得到：  
\begin{equation}\begin{aligned}  
x_1 =&\, [u_0,0,\cdots,0] \\\  
x_2 =&\, [u_1,u_0,0,\cdots,0] \\\  
&\vdots \\\  
x_d =&\, [u_{d-1},\cdots,u_1,u_0] \\\  
x_{d+1} =&\, [u_d, u_{d-1},\cdots,u_1] \\\  
&\vdots \\\  
\end{aligned}\end{equation}  
也就是说，模型一直在滚动储存最近的$d$个$u_k$，如果没有任何其他先验知识，那么这很明显是一个很合理的初始解，所以原论文在初始化阶段将$a_1,a_2,\cdots,a_d$设为零。

原论文对这个初始化还有一个增强数值稳定性、防止梯度爆炸的解释。从上一篇文章我们知道，线性系统$\eqref{eq:linear}$具备相似不变性，这意味着它的动力学跟将$\bar{A}$对角化后的动力学在数学上是一致的，而$\bar{A}$的对角化矩阵，就是它的特征多项式的所有零点组成的对角矩阵，如果某个零点$\lambda_k$的模大于1，那么经过多步递归后就可能发生数值/梯度爆炸。

换句话说，我们最好可以约束$a_1,a_2,\cdots,a_d$，使得多项式$\lambda^d + a_1 \lambda^{d-1} + a_2 \lambda^{d-2} + \cdots + a_d$的所有零点的模都不大于1，以获得更好的数值稳定性同时避免梯度爆炸。然而，保证多项式的零点都在单位圆内的充要条件依然不得而知，但有一个相对简单的充分条件是$|a_1| + |a_2| + \cdots + |a_d| < 1$。

> **结论：** 当$|a_1| + |a_2| + \cdots + |a_d| < 1$时，多项式$\lambda^d + a_1 \lambda^{d-1} + a_2 \lambda^{d-2} + \cdots + a_d$的所有零点模长都不超过1。
> 
> **证明：** 用反证法。假设该多项式有一个模大于1的零点$\lambda_0$，那么$|\lambda_0^{-1}| < 1$，于是  
>  \begin{equation}\begin{aligned}  
>  1 =&\, -a_1\lambda_0^{-1}-a_2\lambda_0^{-2}-\cdots-a_d \lambda_0^{-d} \\\  
>  \leq &\, |a_1\lambda_0^{-1}+a_2\lambda_0^{-2}+\cdots+a_d \lambda_0^{-d}| \\\  
>  \leq &\, |a_1\lambda_0^{-1}|+|a_2\lambda_0^{-2}|+\cdots+|a_d \lambda_0^{-d}| \\\  
>  \leq &\, |a_1|+|a_2|+\cdots+|a_d| \\\  
>  < &\, 1  
>  \end{aligned}\end{equation}  
>  这就出现了$1 < 1$的矛盾，因此假设不成立，多项式的所有零点模长都不大于1。

然而，RTF指出，如果直接约束$a_1,a_2,\cdots,a_d$满足$|a_1| + |a_2| + \cdots + |a_d| < 1$，会大大削弱模型的表达能力，弊大于利；RTF进一步发现，只需要在初始化阶段尽可能满足该条件，然后让模型自己慢慢学就行了。最满足这个条件的取值自然是$a_1=a_2=\cdots=a_d=0$，所以RTF选取了全零初始化。

## 实验效果 #

关于实验部份，下面两张图表就可以看出RTF的显著特点：  


[![RTF的复杂度基本上跟state size无关](/usr/uploads/2024/06/1648668254.png)](/usr/uploads/2024/06/1648668254.png "点击查看原图")

RTF的复杂度基本上跟state size无关

[![RTF可以通过增大state size来提高效果](/usr/uploads/2024/06/3604252695.png)](/usr/uploads/2024/06/3604252695.png "点击查看原图")

RTF可以通过增大state size来提高效果

第一张图显示了RTF的计算复杂度（时间、空间）跟state size没有明显关系，而正因为如此，我们可以通过增大RTF的state size来改善RTF的效果（反正不增加复杂度），也就是第二张图表所显示出来的效果。其他实验结果读者自行翻阅原论文即可。

## 文章小结 #

本文介绍了SSM模型的一个新工作RTF，它观察到线性RNN的卷积核的生成函数实际上可以表示为一个有理函数（分式多项式），利用这个特点，我们可以将SSM的参数化全部转移到生成函数空间上去，并利用离散傅立叶变换来加速，这使得整个计算流程显著简化。跟S4的“对角+低秩”分解相比，RTF也显得更为简明直观。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10180>_

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

苏剑林. (Jun. 27, 2024). 《重温SSM（四）：有理生成函数的新视角 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10180>

@online{kexuefm-10180,  
title={重温SSM（四）：有理生成函数的新视角},  
author={苏剑林},  
year={2024},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/10180}},  
} 


---

## 公式推导与注释

本节提供有理生成函数方法的详细数学推导,包括友矩阵构造、Z变换理论、传递函数分析、稳定性理论等核心内容的完整证明。

### 一、有理函数的基本理论

#### 1.1 有理函数的定义与性质

**定义1.1 (有理函数)**: 有理函数是两个多项式之比:
\begin{equation}
H(z) = \frac{P(z)}{Q(z)} = \frac{b_0 + b_1z + \cdots + b_mz^m}{a_0 + a_1z + \cdots + a_nz^n} \tag{1}
\end{equation}

**定理1.2 (有理函数的标准形式)**: 任何有理函数可以标准化为:
\begin{equation}
H(z) = \frac{b_1 + b_2z + \cdots + b_dz^{d-1}}{1 + a_1z + \cdots + a_dz^d} \tag{2}
\end{equation}
其中$d = \max(m,n)$,分母首项系数归一化为1。

**证明**: 两边同时除以$a_0$(假设$a_0 \neq 0$),必要时补零使分子分母次数匹配。$\square$

**定理1.3 (部分分式展开)**: 若$H(z)$的极点$\lambda_1,\ldots,\lambda_d$互不相同,则:
\begin{equation}
H(z) = \sum_{k=1}^{d}\frac{r_k}{1 - \lambda_k z} \tag{3}
\end{equation}
其中$r_k$为留数。

**推论1.4**: 部分分式展开对应的时域信号为:
\begin{equation}
h_n = \sum_{k=1}^{d}r_k \lambda_k^n \tag{4}
\end{equation}

#### 1.2 从SSM到有理函数

**定理1.5 (SSM生成函数的有理性)**: 对于线性SSM:
\begin{align}
x_{k+1} &= \bar{A}x_k + \bar{B}u_k \tag{5} \\
y_k &= \bar{C}^*x_k \tag{6}
\end{align}
其卷积核的生成函数为:
\begin{equation}
\mathcal{G}(z) = \bar{C}^*(I - z\bar{A})^{-1}\bar{B} \tag{7}
\end{equation}
是$z$的有理函数。

**详细推导**:

步骤1: 设初始状态$x_0 = 0$,递归展开:
\begin{align}
y_1 &= \bar{C}^*\bar{B}u_0 \tag{8} \\
y_2 &= \bar{C}^*\bar{A}\bar{B}u_0 + \bar{C}^*\bar{B}u_1 \tag{9} \\
&\vdots \tag{10} \\
y_n &= \sum_{k=0}^{n-1}\bar{C}^*\bar{A}^k\bar{B}u_{n-1-k} \tag{11}
\end{align}

步骤2: 定义卷积核$K_k = \bar{C}^*\bar{A}^k\bar{B}$,构造生成函数:
\begin{align}
\mathcal{G}(z) &= \sum_{k=0}^{\infty}K_k z^k \tag{12} \\
&= \sum_{k=0}^{\infty}\bar{C}^*\bar{A}^k\bar{B}z^k \tag{13} \\
&= \bar{C}^*\left(\sum_{k=0}^{\infty}(z\bar{A})^k\right)\bar{B} \tag{14} \\
&= \bar{C}^*(I - z\bar{A})^{-1}\bar{B} \tag{15}
\end{align}

步骤3: 使用$M^{-1} = \frac{\text{adj}(M)}{\det(M)}$:
\begin{equation}
\mathcal{G}(z) = \frac{\bar{C}^*\text{adj}(I - z\bar{A})\bar{B}}{\det(I - z\bar{A})} \tag{16}
\end{equation}

步骤4: 分析:
- 分母$\det(I - z\bar{A})$是$z$的$d$次多项式
- 分子$\bar{C}^*\text{adj}(I - z\bar{A})\bar{B}$是$z$的至多$d-1$次多项式
- 因此$\mathcal{G}(z)$是有理函数$\square$

#### 1.3 有理函数的唯一性

**定理1.6 (参数化的非唯一性)**: 给定有理函数$H(z) = \frac{P(z)}{Q(z)}$,存在无穷多组$(\bar{A},\bar{B},\bar{C})$使得:
\begin{equation}
H(z) = \bar{C}^*(I - z\bar{A})^{-1}\bar{B} \tag{17}
\end{equation}

**证明**: 对于任意可逆矩阵$T$,令:
\begin{equation}
(\bar{A}',\bar{B}',\bar{C}') = (T\bar{A}T^{-1}, T\bar{B}, (T^{-1})^*\bar{C}) \tag{18}
\end{equation}

则:
\begin{align}
\bar{C}'^*(I - z\bar{A}')^{-1}\bar{B}' &= \bar{C}^*T^{-*}(I - zT\bar{A}T^{-1})^{-1}T\bar{B} \tag{19} \\
&= \bar{C}^*T^{-*}T(I - z\bar{A})^{-1}T^{-1}T\bar{B} \tag{20} \\
&= \bar{C}^*(I - z\bar{A})^{-1}\bar{B} \tag{21}
\end{align}
$\square$

**推论1.7**: 不同的状态空间实现(realization)对应相同的输入-输出行为。

### 二、友矩阵(Companion Matrix)的构造

#### 2.1 友矩阵的定义

**定义2.1 (友矩阵)**: 给定首一多项式$p(\lambda) = \lambda^d + a_1\lambda^{d-1} + \cdots + a_d$,其友矩阵定义为:
\begin{equation}
C_p = \begin{pmatrix}
-a_1 & -a_2 & -a_3 & \cdots & -a_{d-1} & -a_d \\
1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 1 & 0 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & 1 & 0
\end{pmatrix} \tag{22}
\end{equation}

**定理2.2 (友矩阵的特征多项式)**: 友矩阵$C_p$的特征多项式恰好是$p(\lambda)$。

**详细证明**:

步骤1: 计算$\det(\lambda I - C_p)$:
\begin{equation}
\det(\lambda I - C_p) = \det\begin{pmatrix}
\lambda + a_1 & a_2 & a_3 & \cdots & a_{d-1} & a_d \\
-1 & \lambda & 0 & \cdots & 0 & 0 \\
0 & -1 & \lambda & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & \cdots & -1 & \lambda
\end{pmatrix} \tag{23}
\end{equation}

步骤2: 按第一行展开行列式。对第一列,系数为$(\lambda + a_1)$,子式为:
\begin{equation}
M_1 = \det\begin{pmatrix}
\lambda & 0 & 0 & \cdots & 0 \\
-1 & \lambda & 0 & \cdots & 0 \\
0 & -1 & \lambda & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \lambda
\end{pmatrix} = \lambda^{d-1} \tag{24}
\end{equation}

步骤3: 对第$j$列($j \geq 2$),系数为$(-1)^{1+j}a_j$,子式为:
\begin{equation}
M_j = (-1)^{j-1}\lambda^{d-j} \tag{25}
\end{equation}

步骤4: 综合得到:
\begin{align}
\det(\lambda I - C_p) &= (\lambda + a_1)\lambda^{d-1} + \sum_{j=2}^{d}(-1)^{1+j}a_j \cdot (-1)^{j-1}\lambda^{d-j} \tag{26} \\
&= \lambda^d + a_1\lambda^{d-1} + a_2\lambda^{d-2} + \cdots + a_d \tag{27} \\
&= p(\lambda) \tag{28}
\end{align}
$\square$

#### 2.2 可控标准型

**定理2.3 (可控标准型实现)**: 给定传递函数:
\begin{equation}
H(z) = \frac{b_1 + b_2z + \cdots + b_dz^{d-1}}{1 + a_1z + \cdots + a_dz^d} \tag{29}
\end{equation}

可控标准型实现为:
\begin{align}
\bar{A} &= \begin{pmatrix}
-a_1 & -a_2 & \cdots & -a_d \\
1 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{pmatrix} \tag{30} \\
\bar{B} &= \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix} \tag{31} \\
\bar{C} &= \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_d \end{pmatrix} \tag{32}
\end{align}

**详细验证**:

步骤1: 计算$(I - z\bar{A})^{-1}$。注意到:
\begin{equation}
I - z\bar{A} = \begin{pmatrix}
1 + za_1 & za_2 & \cdots & za_d \\
-z & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{pmatrix} \tag{33}
\end{equation}

步骤2: 计算行列式:
\begin{equation}
\det(I - z\bar{A}) = 1 + a_1z + a_2z^2 + \cdots + a_dz^d \tag{34}
\end{equation}

步骤3: 利用Cramer法则,$(I - z\bar{A})^{-1}\bar{B}$的第$k$个分量为:
\begin{equation}
[(I - z\bar{A})^{-1}\bar{B}]_k = \frac{z^{k-1}}{\det(I - z\bar{A})} \tag{35}
\end{equation}

步骤4: 计算$\bar{C}^*(I - z\bar{A})^{-1}\bar{B}$:
\begin{align}
\bar{C}^*(I - z\bar{A})^{-1}\bar{B} &= \sum_{k=1}^{d}b_k \cdot \frac{z^{k-1}}{1 + a_1z + \cdots + a_dz^d} \tag{36} \\
&= \frac{b_1 + b_2z + \cdots + b_dz^{d-1}}{1 + a_1z + \cdots + a_dz^d} \tag{37} \\
&= H(z) \tag{38}
\end{align}
$\square$

#### 2.3 可观标准型

**定理2.4 (可观标准型)**: 同样的传递函数,可观标准型实现为:
\begin{align}
\bar{A} &= \begin{pmatrix}
-a_1 & 1 & 0 & \cdots & 0 \\
-a_2 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
-a_d & 0 & 0 & \cdots & 0
\end{pmatrix} \tag{39} \\
\bar{B} &= \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_d \end{pmatrix} \tag{40} \\
\bar{C} &= \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix} \tag{41}
\end{align}

这是可控标准型的转置形式。

### 三、Z变换理论

#### 3.1 Z变换的定义

**定义3.1 (Z变换)**: 离散时间信号$\{x_n\}_{n=0}^{\infty}$的Z变换定义为:
\begin{equation}
X(z) = \mathcal{Z}\{x_n\} = \sum_{n=0}^{\infty}x_n z^{-n} \tag{42}
\end{equation}

**注**: 与生成函数$\sum_{n=0}^{\infty}x_n z^n$相比,Z变换使用$z^{-n}$。两者通过变量替换$z \leftrightarrow z^{-1}$关联。

**定理3.2 (Z变换的收敛域)**: $X(z)$在$|z| > R$的区域收敛,其中$R = \limsup_{n \to \infty}|x_n|^{1/n}$。

#### 3.2 Z变换的基本性质

**定理3.3 (线性性)**:
\begin{equation}
\mathcal{Z}\{ax_n + by_n\} = aX(z) + bY(z) \tag{43}
\end{equation}

**定理3.4 (时移性质)**:
\begin{equation}
\mathcal{Z}\{x_{n-k}\} = z^{-k}X(z) \tag{44}
\end{equation}
(假设$x_n = 0, n < 0$)

**定理3.5 (卷积定理)**:
\begin{equation}
\mathcal{Z}\{x_n * y_n\} = X(z)Y(z) \tag{45}
\end{equation}

**证明**: 设$w_n = \sum_{k=0}^{n}x_k y_{n-k}$,则:
\begin{align}
W(z) &= \sum_{n=0}^{\infty}w_n z^{-n} \tag{46} \\
&= \sum_{n=0}^{\infty}\left(\sum_{k=0}^{n}x_k y_{n-k}\right)z^{-n} \tag{47} \\
&= \sum_{k=0}^{\infty}\sum_{m=0}^{\infty}x_k y_m z^{-(k+m)} \tag{48} \\
&= \left(\sum_{k=0}^{\infty}x_k z^{-k}\right)\left(\sum_{m=0}^{\infty}y_m z^{-m}\right) \tag{49} \\
&= X(z)Y(z) \tag{50}
\end{align}
$\square$

#### 3.3 传递函数的Z域表示

**定义3.6 (传递函数)**: 系统的传递函数定义为输出与输入的Z变换之比:
\begin{equation}
H(z) = \frac{Y(z)}{U(z)} \tag{51}
\end{equation}

**定理3.7 (差分方程的Z变换)**: 差分方程:
\begin{equation}
y_n + a_1y_{n-1} + \cdots + a_dy_{n-d} = b_0u_n + b_1u_{n-1} + \cdots + b_du_{n-d} \tag{52}
\end{equation}

对应的传递函数为:
\begin{equation}
H(z) = \frac{b_0 + b_1z^{-1} + \cdots + b_dz^{-d}}{1 + a_1z^{-1} + \cdots + a_dz^{-d}} \tag{53}
\end{equation}

**证明**: 对(52)两边取Z变换,利用时移性质:
\begin{align}
Y(z) + a_1z^{-1}Y(z) + \cdots + a_dz^{-d}Y(z) &= b_0U(z) + \cdots + b_dz^{-d}U(z) \tag{54} \\
(1 + a_1z^{-1} + \cdots + a_dz^{-d})Y(z) &= (b_0 + \cdots + b_dz^{-d})U(z) \tag{55} \\
H(z) = \frac{Y(z)}{U(z)} &= \frac{b_0 + b_1z^{-1} + \cdots + b_dz^{-d}}{1 + a_1z^{-1} + \cdots + a_dz^{-d}} \tag{56}
\end{align}
$\square$

### 四、极点-零点分析

#### 4.1 极点与零点的定义

**定义4.1**: 对于传递函数$H(z) = \frac{P(z)}{Q(z)}$:
- **零点**: $P(z) = 0$的根,记为$\{z_k\}$
- **极点**: $Q(z) = 0$的根,记为$\{\lambda_k\}$

**定理4.2 (极点与状态矩阵)**: 极点恰好是$\bar{A}$的特征值:
\begin{equation}
\det(I - z\bar{A}) = 0 \Leftrightarrow \det(\lambda I - \bar{A}) = 0, \quad \lambda = z^{-1} \tag{57}
\end{equation}

#### 4.2 稳定性分析

**定理4.3 (BIBO稳定性)**: 系统BIBO(有界输入有界输出)稳定当且仅当所有极点的模严格小于1:
\begin{equation}
|\lambda_k| < 1, \quad \forall k \tag{58}
\end{equation}

**详细证明**:

步骤1: 部分分式展开:
\begin{equation}
H(z) = \sum_{k=1}^{d}\frac{r_k}{1 - \lambda_k z} \tag{59}
\end{equation}

步骤2: 逆Z变换得到脉冲响应:
\begin{equation}
h_n = \sum_{k=1}^{d}r_k \lambda_k^n \tag{60}
\end{equation}

步骤3: BIBO稳定性要求$\sum_{n=0}^{\infty}|h_n| < \infty$:
\begin{align}
\sum_{n=0}^{\infty}|h_n| &\leq \sum_{n=0}^{\infty}\sum_{k=1}^{d}|r_k||\lambda_k|^n \tag{61} \\
&= \sum_{k=1}^{d}|r_k|\sum_{n=0}^{\infty}|\lambda_k|^n \tag{62}
\end{align}

步骤4: 几何级数收敛条件:
\begin{equation}
\sum_{n=0}^{\infty}|\lambda_k|^n < \infty \Leftrightarrow |\lambda_k| < 1 \tag{63}
\end{equation}
$\square$

**推论4.4 (边界情况)**:
- 若$|\lambda_k| = 1$,系统临界稳定(marginally stable)
- 若$|\lambda_k| > 1$,系统不稳定

#### 4.3 零点的作用

**定理4.5 (零点的影响)**: 零点不影响稳定性,但影响:
1. **瞬态响应**: 零点位置影响系统的超调和振荡
2. **频率响应**: 零点造成频率响应的局部极大
3. **可逆性**: 若零点在单位圆内,系统可因果稳定地逆转

**示例4.6**:
\begin{equation}
H_1(z) = \frac{1}{1 - 0.5z} \quad \text{vs} \quad H_2(z) = \frac{1 - 0.9z}{1 - 0.5z} \tag{64}
\end{equation}

两者极点相同(都稳定),但$H_2$的零点$z = 1/0.9 \approx 1.11$在单位圆外,导致阶跃响应有不同特性。

### 五、频率响应分析

#### 5.1 频率响应的定义

**定义5.1 (频率响应)**: 将单位圆上的点$z = e^{j\omega}$代入传递函数:
\begin{equation}
H(e^{j\omega}) = |H(e^{j\omega})|e^{j\angle H(e^{j\omega})} \tag{65}
\end{equation}
- $|H(e^{j\omega})|$: 幅度响应(magnitude response)
- $\angle H(e^{j\omega})$: 相位响应(phase response)

**定理5.2 (稳态正弦响应)**: 若输入$u_n = A\cos(\omega n)$,稳态输出为:
\begin{equation}
y_n^{ss} = A|H(e^{j\omega})|\cos(\omega n + \angle H(e^{j\omega})) \tag{66}
\end{equation}

**证明**:
步骤1: 复数形式输入$u_n = Ae^{j\omega n}$的Z变换:
\begin{equation}
U(z) = \frac{A}{1 - e^{j\omega}z^{-1}} \tag{67}
\end{equation}

步骤2: 输出的Z变换:
\begin{equation}
Y(z) = H(z)U(z) = H(z)\frac{A}{1 - e^{j\omega}z^{-1}} \tag{68}
\end{equation}

步骤3: 部分分式展开,主导项(来自$z = e^{j\omega}$的极点):
\begin{equation}
y_n^{ss} = AH(e^{j\omega})e^{j\omega n} \tag{69}
\end{equation}

步骤4: 取实部得到(66)式。$\square$

#### 5.2 Bode图

**定义5.3 (Bode图)**: 频率响应的对数图:
- **幅度图**: $20\log_{10}|H(e^{j\omega})|$ vs $\omega$
- **相位图**: $\angle H(e^{j\omega})$ vs $\omega$

**定理5.4 (极点-零点对频率响应的贡献)**:
\begin{align}
|H(e^{j\omega})| &= K\frac{\prod_k|e^{j\omega} - z_k|}{\prod_l|e^{j\omega} - \lambda_l|} \tag{70} \\
\angle H(e^{j\omega}) &= \sum_k\angle(e^{j\omega} - z_k) - \sum_l\angle(e^{j\omega} - \lambda_l) \tag{71}
\end{align}

**几何解释**:
- 零点$z_k$: 从$e^{j\omega}$到$z_k$的向量
- 极点$\lambda_l$: 从$e^{j\omega}$到$\lambda_l$的向量
- 幅度响应 = (到零点距离之积) / (到极点距离之积)

#### 5.3 特殊频率点

**定理5.5 (Nyquist频率)**: 对于离散系统,有意义的频率范围为$\omega \in [0, \pi]$,因为:
\begin{equation}
H(e^{j(\omega + 2\pi)}) = H(e^{j\omega}) \tag{72}
\end{equation}

**定义5.6**:
- **DC增益**: $H(1) = H(e^{j0})$
- **Nyquist增益**: $H(-1) = H(e^{j\pi})$

### 六、IIR滤波器设计

#### 6.1 IIR滤波器的基本概念

**定义6.1 (IIR滤波器)**: 无限脉冲响应(IIR)滤波器的差分方程:
\begin{equation}
y_n = \sum_{k=1}^{d}a_k y_{n-k} + \sum_{k=0}^{d}b_k u_{n-k} \tag{73}
\end{equation}

与FIR(有限脉冲响应)滤波器的区别: IIR包含输出的反馈项。

**优点**: 相同性能下,IIR需要更少的系数
**缺点**: 可能不稳定,相位响应非线性

#### 6.2 Butterworth滤波器

**定理6.2 (Butterworth低通滤波器)**: $N$阶Butterworth滤波器的频率响应:
\begin{equation}
|H(e^{j\omega})|^2 = \frac{1}{1 + (\omega/\omega_c)^{2N}} \tag{74}
\end{equation}
其中$\omega_c$是截止频率。

**特性**:
- 通带内最大平坦(maximally flat)
- 无波纹
- 单调衰减

**极点配置**: Butterworth极点均匀分布在半径为$\omega_c$的圆上:
\begin{equation}
\lambda_k = \omega_c e^{j\pi(2k+N-1)/(2N)}, \quad k = 0,1,\ldots,N-1 \tag{75}
\end{equation}

#### 6.3 Chebyshev滤波器

**定理6.3 (Chebyshev Type I)**:
\begin{equation}
|H(e^{j\omega})|^2 = \frac{1}{1 + \epsilon^2 T_N^2(\omega/\omega_c)} \tag{76}
\end{equation}
其中$T_N$是$N$阶Chebyshev多项式,$\epsilon$控制通带波纹。

**特性**:
- 通带有等波纹
- 过渡带比Butterworth陡峭
- 阻带单调衰减

#### 6.4 双线性变换设计法

**定理6.4 (双线性变换)**: 从模拟滤波器$H_a(s)$到数字滤波器$H(z)$:
\begin{equation}
s = \frac{2}{T}\frac{1 - z^{-1}}{1 + z^{-1}} \tag{77}
\end{equation}

**性质**:
1. 将$s$平面的虚轴映射到$z$平面的单位圆
2. 保持稳定性: 若$H_a(s)$稳定,则$H(z)$稳定
3. 产生频率扭曲(warping):
\begin{equation}
\omega = 2\arctan(\Omega T/2) \tag{78}
\end{equation}

**预扭曲(Pre-warping)**: 设计时调整模拟频率以补偿扭曲:
\begin{equation}
\Omega_c = \frac{2}{T}\tan(\omega_c/2) \tag{79}
\end{equation}

### 七、递归实现与并行实现

#### 7.1 直接形式I

**算法7.1 (Direct Form I)**:
```
对每个时间步n:
  # 计算FIR部分
  v_n = Σ(k=0 to d) b_k · u_{n-k}

  # 计算IIR部分
  y_n = v_n - Σ(k=1 to d) a_k · y_{n-k}
```

**复杂度**: $\mathcal{O}(d)$每时间步
**存储**: 需要$2d$个延迟单元

#### 7.2 直接形式II(标准型)

**算法7.2 (Direct Form II)**:
```
对每个时间步n:
  # 合并延迟线
  w_n = u_n - Σ(k=1 to d) a_k · w_{n-k}
  y_n = Σ(k=0 to d) b_k · w_{n-k}
```

**优点**: 只需$d$个延迟单元(最小化)
**缺点**: 对有限字长敏感

#### 7.3 级联形式

**定理7.3 (级联分解)**: 将高阶传递函数分解为二阶节(biquad)级联:
\begin{equation}
H(z) = K\prod_{k=1}^{\lceil d/2 \rceil}H_k(z) \tag{80}
\end{equation}
其中每个$H_k(z)$是二阶:
\begin{equation}
H_k(z) = \frac{b_{k0} + b_{k1}z + b_{k2}z^2}{1 + a_{k1}z + a_{k2}z^2} \tag{81}
\end{equation}

**优点**:
- 数值稳定性好
- 易于调整
- 可独立设计每个二阶节

#### 7.4 并行形式

**定理7.4 (并行分解)**: 部分分式展开:
\begin{equation}
H(z) = C + \sum_{k=1}^{d}\frac{r_k}{1 - \lambda_k z} \tag{82}
\end{equation}

**实现**: 每个一阶节并行计算,最后求和:
```python
y_n = C·u_n + Σ_k r_k·x_k[n]
其中 x_k[n+1] = λ_k·x_k[n] + u_n
```

**优点**: 完全并行,适合硬件实现
**缺点**: 对极点位置敏感

### 八、RFT方法的深入分析

#### 8.1 从矩阵到多项式

**定理8.1 (特征多项式的计算)**: 对于$d \times d$矩阵$\bar{A}$,其特征多项式:
\begin{equation}
p(\lambda) = \det(\lambda I - \bar{A}) = \lambda^d + c_1\lambda^{d-1} + \cdots + c_d \tag{83}
\end{equation}
可通过以下方法计算:

**方法1 (Faddeev-LeVerrier算法)**:
\begin{align}
M_0 &= I \tag{84} \\
c_k &= -\frac{1}{k}\text{tr}(\bar{A}M_{k-1}) \tag{85} \\
M_k &= \bar{A}M_{k-1} + c_k I \tag{86}
\end{align}

复杂度: $\mathcal{O}(d^4)$

**方法2 (通过QR分解)**:
1. QR分解: $\bar{A} = QR$
2. 转为Hessenberg形式
3. 特征多项式由Hessenberg矩阵快速计算

复杂度: $\mathcal{O}(d^3)$

#### 8.2 RFT参数化的优势

**定理8.2 (参数化的简化)**: RFT参数化:
\begin{align}
\text{矩阵参数化:} &\quad (\bar{A},\bar{B},\bar{C}) \in \mathbb{R}^{d^2 + 2d} \tag{87} \\
\text{RFT参数化:} &\quad (a,b) \in \mathbb{R}^{2d} \tag{88}
\end{align}

参数减少: $d^2 + 2d \to 2d$,节约$\mathcal{O}(d^2)$。

**定理8.3 (训练复杂度)**:
- **矩阵形式**: 需计算$\bar{A}^k$,$k = 0,1,\ldots,L-1$,复杂度$\mathcal{O}(Ld^3)$
- **RFT形式**: 直接DFT,复杂度$\mathcal{O}(L\log L)$

### 九、稳定性约束

#### 9.1 Schur-Cohn判据

**定理9.1 (Schur-Cohn判据)**: 多项式$p(z) = a_0 + a_1z + \cdots + a_nz^n$的所有根在单位圆内当且仅当Schur-Cohn矩阵正定。

**Schur-Cohn矩阵**:
\begin{equation}
S = \begin{pmatrix}
a_0 & 0 & \cdots & 0 & \bar{a}_n \\
a_1 & a_0 & \cdots & 0 & \bar{a}_{n-1} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_n & a_{n-1} & \cdots & a_0 & \bar{a}_0
\end{pmatrix} \tag{89}
\end{equation}

#### 9.2 充分条件

**定理9.2 (简化充分条件)**: 若:
\begin{equation}
\sum_{k=1}^{d}|a_k| < 1 \tag{90}
\end{equation}
则多项式$1 + a_1z + \cdots + a_dz^d$的所有根的模都不超过1。

**证明**: 使用反证法和Rouché定理(详见文件2的证明)。$\square$

**实践应用**: RFT初始化策略:
\begin{align}
a_1,\ldots,a_d &\sim \mathcal{N}(0, \sigma^2), \quad \sigma \ll 1 \tag{91} \\
b_1,\ldots,b_d &\sim \mathcal{N}(0, 1/\sqrt{d}) \tag{92}
\end{align}

确保初始时系统接近稳定。

### 十、数值示例与可视化

#### 10.1 简单一阶系统

**示例10.1**: 考虑$H(z) = \frac{0.5}{1 - 0.8z}$

**参数**: $a_1 = -0.8$, $b_1 = 0.5$

**极点**: $\lambda = 0.8$(在单位圆内,稳定)

**脉冲响应**: $h_n = 0.5 \times 0.8^n$

**频率响应**:
\begin{equation}
|H(e^{j\omega})| = \frac{0.5}{|1 - 0.8e^{j\omega}|} = \frac{0.5}{\sqrt{1.64 - 1.6\cos\omega}} \tag{93}
\end{equation}

在$\omega = 0$时最大: $|H(1)| = 0.5/0.2 = 2.5$

#### 10.2 二阶谐振系统

**示例10.2**: $H(z) = \frac{0.1}{1 + 1.5z + 0.9z^2}$

**极点**:
\begin{equation}
\lambda_{1,2} = \frac{-1.5 \pm \sqrt{1.5^2 - 4 \times 0.9}}{2} = -0.75 \pm j0.66 \tag{94}
\end{equation}

$|\lambda_{1,2}| = \sqrt{0.75^2 + 0.66^2} \approx 1.0$(临界稳定)

**谐振频率**: $\omega_r = \arctan(0.66/0.75) \approx 0.73$ rad

**行为**: 系统在$\omega \approx 0.73$处有强烈谐振。

### 十一、与其他方法的联系

#### 11.1 RFT vs S4

| 特性 | S4 | RFT |
|------|-----|-----|
| 参数化 | $(\bar{A},\bar{B},\bar{C})$ | $(a,b)$ |
| 参数量 | $\mathcal{O}(d^2)$ | $\mathcal{O}(d)$ |
| 训练复杂度 | $\mathcal{O}((L+d)\log^2(L+d))$ | $\mathcal{O}(L\log L)$ |
| 依赖state size | 是 | 否 |
| $\bar{A}$约束 | DPLR分解 | 无(隐式通过$a$) |

**核心区别**: RFT直接在传递函数空间工作,避免了显式的矩阵运算。

#### 11.2 RFT vs LTI系统理论

RFT本质上是经典LTI(线性时不变)系统理论在深度学习中的应用:
- **极点配置 $\leftrightarrow$ 动力学特性**
- **零点配置 $\leftrightarrow$ 瞬态响应**
- **有理函数 $\leftrightarrow$ SSM等价性**

**创新点**: 将$(a,b)$作为可学习参数,通过反向传播优化。

### 十二、实践指南

#### 12.1 初始化策略

**策略12.1**:
```python
# 保守初始化(接近积分器)
a = np.zeros(d)
a[0] = -0.9  # 单极点在z=0.9
b = np.random.randn(d) / np.sqrt(d)

# 随机初始化(探索性)
a = np.random.randn(d) * 0.1
b = np.random.randn(d) / np.sqrt(d)
确保: np.sum(np.abs(a)) < 0.9  # 稳定性
```

#### 12.2 训练技巧

**技巧12.2**:
1. **梯度裁剪**: 防止极点跳出单位圆
\begin{equation}
a_k \leftarrow \text{clip}(a_k, -1, 1) \tag{95}
\end{equation}

2. **稳定性投影**: 每$N$步投影到稳定区域
```python
if np.sum(np.abs(a)) >= 1:
    a = a * 0.9 / np.sum(np.abs(a))
```

3. **学习率调度**: 极点参数用较小学习率

#### 12.3 调试检查清单

**检查12.3**:
- [ ] 极点都在单位圆内?
- [ ] DFT/IDFT配对正确?
- [ ] 零填充长度正确($2L$)?
- [ ] 数值稳定(无NaN/Inf)?
- [ ] 梯度正常传播?

### 十三、总结

#### 13.1 核心思想

RFT的本质: **将SSM参数化为有理函数的系数,利用Z变换和DFT实现高效计算**。

关键步骤:
1. **建模**: $H(z) = \frac{b_1 + \cdots + b_dz^{d-1}}{1 + a_1z + \cdots + a_dz^d}$
2. **参数化**: $(a_1,\ldots,a_d,b_1,\ldots,b_d)$作为可学习参数
3. **训练**: 通过DFT计算卷积,复杂度$\mathcal{O}(L\log L)$
4. **推理**: 通过友矩阵恢复状态空间实现,复杂度$\mathcal{O}(d)$

#### 13.2 理论意义

- 连接了深度学习与经典控制理论
- 提供了SSM的极简参数化
- 证明了有理函数的表达充分性

#### 13.3 实践价值

- 大幅降低计算复杂度(与state size无关)
- 简化参数调优(参数更少,更直观)
- 易于分析和解释(极点-零点视角)

**展望**: RFT开启了"学习传递函数"的新范式,为序列建模提供了优雅而高效的方案。

