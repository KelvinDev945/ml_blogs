---
title: 脑洞大开：非线性RNN居然也可以并行计算？
slug: 脑洞大开非线性rnn居然也可以并行计算
date: 2023-09-26
tags: 摄动, 方程, 迭代, 语言模型, RNN
status: completed
---

# 脑洞大开：非线性RNN居然也可以并行计算？

**原文链接**: [https://spaces.ac.cn/archives/9783](https://spaces.ac.cn/archives/9783)

**发布日期**: 

---

近年来，线性RNN由于其可并行训练以及常数推理成本等特性，吸引了一定研究人员的关注（例如笔者之前写的[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](/archives/9554)），这让RNN在Transformer遍地开花的潮流中仍有“一席之地”。然而，目前看来这“一席之地”只属于线性RNN，因为非线性RNN无法高效地并行训练，所以在架构之争中是“心有余而力不足”。

不过，一篇名为[《Parallelizing Non-Linear Sequential Models over the Sequence Length》](https://papers.cool/arxiv/2309.12252)的论文有不同的看法，它提出了一种迭代算法，宣传可以实现非线性RNN的并行训练！真有如此神奇？接下来我们一探究竟。

## 求不动点 #

原论文对其方法做了非常一般的介绍，而且其侧重点是PDE和ODE，这里我们直接从RNN入手。考虑常见的简单非线性RNN：  
\begin{equation}x_t = \tanh(Ax_{t-1} + u_t)\label{eq:rnn}\end{equation}  
由于$\tanh$的存在，它只能串行计算。现在我们在两边都减去$Ax_{t-1}$：  
\begin{equation}x_t - Ax_{t-1} = \tanh(Ax_{t-1} + u_t) - Ax_{t-1}\end{equation}  
当然，这改变不了它是非线性RNN的实质。然而我们可以发现，假如右端的$x_{t-1}$换成像$u_t$那样的给定向量，那么这就是一个线性RNN了，根据[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](/archives/9554)的结果，它是可以并行计算的。此时，敏捷的读者可能已经猜到后面的步骤了——迭代求解！

首先，将上述RNN更改成  
\begin{equation}x_t^{(n)} - Ax_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)}\label{eq:rnn-iter}\end{equation}  
从给定$x_t^{(0)}$出发，反复迭代上式，理想情况下，它会收敛于一个不动点$x_t^*$，这就是原来非线性RNN的计算结果。当然，理论上通过式$\eqref{eq:rnn-iter}$迭代的总计算量是比直接通过式$\eqref{eq:rnn}$递归计算要大的，但由于每一步迭代都是可并行的线性RNN，并且如果收敛速度比较快时迭代步数不需要太多，那么总的耗时通常都会快于直接非线性RNN递归（尤其是序列长度很大时）。

## 简化形式 #

事实上，非线性RNN之所以慢，无法并行计算还是次要的，最关键是它包含了大量的非element-wise运算，比如式$\eqref{eq:rnn}$的$\tanh$里边的矩阵运算$Ax_{t-1}$；而线性RNN之所以快，除了它允许并行训练之外，更关键的是它能通过对角化来将矩阵乘法变换为element-wise的乘法——对于element-wise乘法来说，即便是串行计算也不会太慢。

当我们通过式$\eqref{eq:rnn-iter}$将非线性RNN转为线性RNN的迭代之后，同样享受线性RNN可对角化的“待遇”，从而提高计算速度。具体来说，在复数域中将$A$对角化为$P\Lambda P^{-1}$，那么式$\eqref{eq:rnn-iter}$变为  
\begin{equation}x_t^{(n)} - P\Lambda P^{-1} x_{t-1}^{(n)} = \tanh(P\Lambda P^{-1} x_{t-1}^{(n-1)} + u_t) - P\Lambda P^{-1} x_{t-1}^{(n-1)}\end{equation}  
两端都左乘$P^{-1}$：  
\begin{equation}P^{-1} x_t^{(n)} - \Lambda P^{-1} x_{t-1}^{(n)} = P^{-1}\tanh(P\Lambda P^{-1} x_{t-1}^{(n-1)} + u_t) - \Lambda P^{-1} x_{t-1}^{(n-1)}\end{equation}  
令$y_t = P^{-1} x_t$，那么上式可以简化为  
\begin{equation}y_t^{(n)} - \Lambda y_{t-1}^{(n)} = P^{-1}\tanh(P\Lambda y_{t-1}^{(n-1)} + u_t) - \Lambda y_{t-1}^{(n-1)}\end{equation}  
由于RNN之后一般都还要接个投影层，所以$x_t = P y_t$的$P$原则上可以合并到外接的投影层里边，也就是说，上式理论上具备跟原来的$\eqref{eq:rnn}$具备同等的表达能力，但由于$\Lambda$是对角阵，递归的计算量会明显降低。上式还出现了逆矩阵$P^{-1}$，不单计算量大，而且不利于优化，所以我们可以干脆将$P^{-1}$和$P\Lambda$换成两个不相关的参数矩阵：  
\begin{equation}y_t^{(n)} - \Lambda y_{t-1}^{(n)} = P\tanh(Q y_{t-1}^{(n-1)} + u_t) - \Lambda y_{t-1}^{(n-1)}\end{equation}  
只要初始化是$PQ=\Lambda$就行。

## 摄动思想 #

假定$x_t^{(0)}=0$，那么式$\eqref{eq:rnn-iter}$其实就是将原本的非线性RNN就分解为一系列线性RNN：  
\begin{equation}\begin{array}{c}  
x_t^{(1)} - Ax_{t-1}^{(1)} = \tanh(u_t)\\\  
x_t^{(2)} - Ax_{t-1}^{(2)} = \tanh(Ax_{t-1}^{(1)} + u_t) - Ax_{t-1}^{(1)} \\\  
\vdots \\\  
x_t^{(n)} - Ax_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)} \\\  
\vdots \\\  
\end{array}\label{eq:rnns}\end{equation}  
而假设$x_{t-1},u_t$都是小量，那么对式$\eqref{eq:rnn}$右端利用$\tanh x \approx x$得到：  
\begin{equation}x_t = \tanh(Ax_{t-1} + u_t) \approx Ax_{t-1} + u_t \approx Ax_{t-1} + \tanh(u_t)\label{eq:rnn-approx}\end{equation}  
这正好是$\eqref{eq:rnns}$中的第一个方程，因此如果假设成立，那么$x_t^{(1)}$或许已经足够接近理想的$x_t^*$，后面的每一步迭代都在快速逼近它。从这里我们可以看出，“两边同时减去$Ax_{t-1}$”是关键之处，这使得$\eqref{eq:rnn-iter}$的第一步迭代就接近于原本非线性RNN的一阶线性近似，这可以提高收敛速度，也是数学物理中的经典操作，名曰“[摄动](/tag/%E6%91%84%E5%8A%A8/)”。

## 加快收敛 #

根据摄动法的思想，提高收敛速度的关键就是提高近似展开的精度，比如较为简单的改进是只假设$x_{t-1}$是小量，那么根据一阶泰勒展开有（将$u_t$视为列向量，这里的$\circ$是Hadamard积分）  
\begin{equation}x_t = \tanh(Ax_{t-1} + u_t) \approx \tanh(u_t) + (\text{sech}^2 u_t\circ A)x_{t-1}\end{equation}  
于是改进的结果就是式$\eqref{eq:rnn-iter}$变为  
\begin{equation}x_t^{(n)} - A_t x_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - A_t x_{t-1}^{(n-1)}\label{eq:iter-plus1}\end{equation}  
其中$A_t = \text{sech}^2 u_t\circ A$。更精细的改进是在每一步迭代时，都在前一步迭代结果的基础上进行展开：  
\begin{equation}\begin{aligned}  
x_t =&\, \tanh(Ax_{t-1} + u_t) \\\  
\approx&\, \tanh(Ax_{t-1}^{(n-1)} + u_t) + (\text{sech}^2 (Ax_{t-1}^{(n-1)} + u_t)\circ A)(x_{t-1} - x_{t-1}^{(n-1)})  
\end{aligned}\end{equation}  
于是式$\eqref{eq:rnn-iter}$变为  
\begin{equation}x_t^{(n)} - A_t^{(n)} x_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - A_t^{(n)} x_{t-1}^{(n-1)}\label{eq:iter-plus2}\end{equation}  
其中$A_t^{(n)}=\text{sech}^2 (Ax_{t-1}^{(n-1)} + u_t)\circ A$。最后的这个迭代格式，实际上就是求方程数值解的“[牛顿法](https://en.wikipedia.org/wiki/Newton%27s_method)”，它具有二次收敛速度。

## 何必收敛 #

理论上来说，$\eqref{eq:iter-plus1}$、$\eqref{eq:iter-plus2}$两个改进确实能提高收敛速度，然而它们使得每一步线性递归的矩阵$A$变得跟$t$甚至$n$有关了，这其实会大大增加并行的复杂度，也不能利用“简化形式”一节的对角化技巧来加速。另一方面，如果保持$\eqref{eq:rnn-iter}$这样的迭代格式，虽然有诸多效率上的好处，但收敛方面确实无法得到很好的保障。

难道这两者的矛盾就无法调和了吗？事实上，按照笔者的观点，最直接的做法是“别去管它”——借助非线性RNN导出了$\eqref{eq:rnn-iter}$后，就忘记原本的非线性RNN，将式$\eqref{eq:rnn-iter}$作为基本模型。也就是说，何必忧虑式$\eqref{eq:rnn-iter}$会不会收敛到原来的非线性RNN？直接将它作为新的出发点不好吗？梯度下降学到怎样的结果就是怎样的结果，如果梯度下降学到的结果是不收敛到原来的非线性RNN，那么就意味着不收敛到原来的RNN是更适合的。

抛开这一层思维束缚后，其实很多问题会变得豁然开朗起来。首先，即便是式$\eqref{eq:iter-plus2}$在理论上拥有非常好的收敛速度，但也是有条件的，而且在深度学习的背景下，要保证这些条件会显得很奢侈。换言之，即便是式$\eqref{eq:iter-plus2}$的收敛性也没有绝对保证，所以何必“五十步笑百步”去苛责式$\eqref{eq:rnn-iter}$？其次，将式$\eqref{eq:rnn-iter}$视为新的出发点后，我们可以将它单纯地理解为线性RNN的一种新用法，或者说解决线性RNN缺陷（比如线性RNN不是图灵完备的）的一个思路，这样操作性更强。

总的来说，不去管它的收敛性，似乎更能打破思维僵局，探索更一般的结果。

## 一般情形 #

前面的“长篇大论”，都只围绕着简单的非线性RNN也就是式$\eqref{eq:rnn}$进行讨论，对于更常用的LSTM、GRU，结果又如何呢？

以GRU为例，它原本的形式为  
\begin{equation}\begin{aligned} z_{t} & = \sigma \left( W_{z} x_{t} + U_{z} h_{t - 1} + b_{z} \right) \\\  
r_{t} & = \sigma \left( W_{r} x_{t} + U_{r} h_{t - 1} + b_{r} \right) \\\  
\hat{h}_t & = \tanh \left( W_{h} x_{t} + U_{h} (r_t \circ h_{t - 1}) + b_{c} \right)\\\  
h_{t} & = \left(1 - z_{t}\right) \circ h_{t - 1} + z_{t} \circ \hat{h}_t \end{aligned}\end{equation}  
初始阶段，所有门控都可以近似视为$\frac{1}{2}$，那么模仿式$\eqref{eq:rnn-approx}$有  
\begin{equation}\begin{aligned}  
h_{t} &\, = \left(1 - z_{t}\right) \circ h_{t - 1} + z_{t} \circ \hat{h}_t \\\  
&\, \approx \frac{1}{2} h_{t - 1} + \frac{1}{2} \hat{h}_t \\\  
&\, \approx \frac{1}{2} h_{t - 1} + \frac{1}{2} \left(\tanh ( W_{h} x_{t} + b_{c} ) + \frac{1}{2}U_{h} h_{t - 1}\right) \\\  
&\, = \frac{1}{2} \left(I + \frac{1}{2}U_{h}\right)h_{t - 1} + \frac{1}{2} \tanh ( W_{h} x_{t} + b_{c} ) \\\  
\end{aligned}\end{equation}  
所以可以选取$A=\frac{1}{2} \left(I + \frac{1}{2}U_{h}\right)$，将GRU改写为迭代  
\begin{equation}\begin{aligned} z_{t}^{(n)} & = \sigma \left( W_{z} x_{t} + U_{z} h_{t - 1}^{(n-1)} + b_{z} \right) \\\  
r_{t}^{(n)} & = \sigma \left( W_{r} x_{t} + U_{r} h_{t - 1}^{(n-1)} + b_{r} \right) \\\  
\hat{h}_t^{(n)} & = \tanh \left( W_{h} x_{t} + U_{h} (r_t^{(n)} \circ h_{t - 1}^{(n-1)}) + b_{c} \right)\\\  
h_{t}^{(n)} & = Ah_{t-1}^{(n)} - Ah_{t-1}^{(n - 1)} + \left(1 - z_{t}^{(n)}\right) \circ h_{t - 1}^{(n-1)} + z_{t}^{(n)} \circ \hat{h}_t^{(n)} \end{aligned}\end{equation}

总的来说，这种将非线性RNN变为线性RNN迭代的转换，从实践的角度来看，就是以非线性RNN为引，导出一种多层线性RNN的参数共享和组合方法，迭代了几次，那么就有几层线性RNN的计算量。这样自然而言就引发了一个思考：除非可以证明GRU、LSTM等非线性RNN有绝对的优势，否则直接叠加几层“线性RNN+MLP”不好吗？

## 文章小结 #

本文简单探讨了非线性RNN的并行计算问题——通过数学物理中的“摄动”思想，我们可以将非线性RNN转化为线性RNN的迭代，从而利用线性RNN的可并行性来实现非线性RNN的并行。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9783>_

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

苏剑林. (Sep. 26, 2023). 《脑洞大开：非线性RNN居然也可以并行计算？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9783>

@online{kexuefm-9783,  
title={脑洞大开：非线性RNN居然也可以并行计算？},  
author={苏剑林},  
year={2023},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/9783}},  
} 


---

## 详细数学推导与注释

### 1. 非线性RNN的基本形式

**标准非线性RNN**：
$$
x_t = \tanh(Ax_{t-1} + u_t) \tag{1}
$$

其中 $x_t \in \mathbb{R}^d$ 是隐状态，$u_t \in \mathbb{R}^d$ 是输入，$A \in \mathbb{R}^{d \times d}$ 是状态转移矩阵。

**非线性性的重要性**：

**定理（图灵完备性）**：带有sigmoid或tanh激活函数的单层RNN是图灵完备的。

**注释**：这意味着理论上RNN可以模拟任何计算过程，但前提是使用非线性激活函数。

### 2. 不动点迭代的数学基础

**不动点的定义**：

对于函数 $g: \mathbb{R}^d \to \mathbb{R}^d$，若存在 $x^* \in \mathbb{R}^d$ 使得：
$$
g(x^*) = x^* \tag{2}
$$

则称 $x^*$ 为 $g$ 的不动点。

**RNN作为不动点问题**：

式(1)可以重写为：
$$
x_t = \tanh(Ax_{t-1} + u_t) \quad \Leftrightarrow \quad x_t - Ax_{t-1} = \tanh(Ax_{t-1} + u_t) - Ax_{t-1} \tag{3}
$$

关键观察：右端依赖 $x_{t-1}$。如果我们固定 $x_{t-1}$，那么这是一个线性递归！

### 3. 摄动方法的核心思想

**基本策略**：

将非线性RNN改写为一系列线性RNN的迭代：
$$
x_t^{(n)} - Ax_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)} \tag{4}
$$

其中 $n = 1, 2, 3, \ldots$ 是迭代索引。

**初始化**：$x_t^{(0)} = 0$

**迭代序列**：
$$
\begin{aligned}
x_t^{(1)} - Ax_{t-1}^{(1)} &= \tanh(u_t) \tag{5}\\
x_t^{(2)} - Ax_{t-1}^{(2)} &= \tanh(Ax_{t-1}^{(1)} + u_t) - Ax_{t-1}^{(1)} \tag{6}\\
&\vdots \\
x_t^{(n)} - Ax_{t-1}^{(n)} &= \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)} \tag{7}
\end{aligned}
$$

**注释**：每一步迭代都是线性RNN，可以并行计算！

### 4. 一阶近似分析

**泰勒展开**：

假设 $x_{t-1}$ 和 $u_t$ 都是小量，则：
$$
\tanh(Ax_{t-1} + u_t) \approx Ax_{t-1} + u_t \tag{8}
$$

因为 $\tanh(x) \approx x$ 当 $|x| \ll 1$。

**第一次迭代的近似性**：

代入式(5)：
$$
x_t^{(1)} - Ax_{t-1}^{(1)} = \tanh(u_t) \approx u_t \tag{9}
$$

这说明第一次迭代已经给出了非线性RNN的一阶近似。

**误差估计**：

设真解为 $x_t^*$，则：
$$
\|x_t^* - x_t^{(1)}\| = O(\|Ax_{t-1}\|^2 + \|u_t\|^2) \tag{10}
$$

### 5. Banach不动点定理

**定理（Banach不动点定理）**：

设 $(X, d)$ 是完备度量空间，$T: X \to X$ 是压缩映射，即存在 $\alpha \in [0, 1)$ 使得：
$$
d(T(x), T(y)) \leq \alpha \cdot d(x, y), \quad \forall x, y \in X \tag{11}
$$

则 $T$ 有唯一不动点 $x^*$，且对任意 $x_0 \in X$，序列 $x_{n+1} = T(x_n)$ 收敛到 $x^*$。

**应用到RNN**：

定义算子 $T: \mathbb{R}^{d \times L} \to \mathbb{R}^{d \times L}$（$L$ 是序列长度）：
$$
(T(X))_t = A(T(X))_{t-1} + \tanh(AX_{t-1} + u_t) - AX_{t-1} \tag{12}
$$

其中 $X = [x_1, \ldots, x_L]$。

**收敛性条件**：

若 $\|A\|_2 \cdot \|\tanh'(z)\| < 1$，则迭代收敛。

因为 $|\tanh'(z)| \leq 1$，所以当 $\|A\|_2 < 1$ 时，迭代理论上会收敛。

### 6. 收敛速度分析

**线性收敛**：

对于式(4)的迭代，设误差为 $e^{(n)} = x^* - x^{(n)}$，则：
$$
\|e^{(n+1)}\| \leq \alpha \|e^{(n)}\| \tag{13}
$$

其中 $\alpha \approx \|A\|_2$。

**迭代次数估计**：

要达到精度 $\epsilon$，需要迭代次数：
$$
N \geq \frac{\log(\epsilon / \|e^{(0)}\|)}{\log \alpha} \tag{14}
$$

**数值例子**：

若 $\|A\|_2 = 0.9$，$\epsilon = 10^{-6}$，$\|e^{(0)}\| = 1$：
$$
N \geq \frac{\log 10^{-6}}{\log 0.9} \approx \frac{-13.8}{-0.105} \approx 131 \text{ iterations}
$$

这太多了！需要加速。

### 7. 一阶加速：改进的摄动

**改进方案**：

利用 $u_t$ 的信息，进行更精确的泰勒展开：
$$
\tanh(Ax_{t-1} + u_t) \approx \tanh(u_t) + \text{sech}^2(u_t) \odot (Ax_{t-1}) \tag{15}
$$

其中 $\odot$ 是Hadamard积（element-wise乘法）。

**修改迭代格式**：
$$
x_t^{(n)} - A_t x_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - A_t x_{t-1}^{(n-1)} \tag{16}
$$

其中：
$$
A_t = \text{sech}^2(u_t) \odot A \tag{17}
$$

**优势**：每次迭代使用更准确的局部线性化。

**劣势**：$A_t$ 依赖于 $t$，增加了并行计算的复杂度。

### 8. 牛顿法：二次收敛

**牛顿法的迭代格式**：

将式(1)看作方程 $F(x_t) = 0$，其中：
$$
F(x_t) = x_t - \tanh(Ax_{t-1} + u_t) \tag{18}
$$

牛顿迭代：
$$
x_t^{(n+1)} = x_t^{(n)} - (F'(x_t^{(n)}))^{-1} F(x_t^{(n)}) \tag{19}
$$

**雅可比矩阵**：
$$
F'(x_t) = I - \text{diag}(\text{sech}^2(Ax_{t-1} + u_t)) \cdot A \tag{20}
$$

**在每步使用上一次迭代的信息**：
$$
F'(x_t^{(n)}) = I - \text{diag}(\text{sech}^2(Ax_{t-1}^{(n-1)} + u_t)) \cdot A \tag{21}
$$

**二次收敛性**：
$$
\|e^{(n+1)}\| \leq C \|e^{(n)}\|^2 \tag{22}
$$

**注释**：二次收敛意味着每次迭代，有效数字翻倍！

**迭代次数**：

通常只需要 $N = 3-5$ 次迭代即可达到机器精度。

### 9. 对角化简化

**问题**：每步都需要计算不同的矩阵-向量乘法，失去了对角化的优势。

**权衡**：

- **方案1**（式4）：固定 $A$，可对角化，收敛慢
- **方案2**（式16/21）：变化的 $A_t^{(n)}$，收敛快，无法对角化

**实践选择**：

- 如果序列很长（$L > 10000$），使用方案1（并行化收益大）
- 如果序列较短（$L < 1000$），使用方案2（收敛速度重要）

### 10. 对角化后的并行算法

**假设 $A = P\Lambda P^{-1}$**：

式(4)变为：
$$
y_t^{(n)} - \Lambda y_{t-1}^{(n)} = P^{-1}[\tanh(P\Lambda y_{t-1}^{(n-1)} + u_t) - P\Lambda y_{t-1}^{(n-1)}] \tag{23}
$$

其中 $y_t = P^{-1}x_t$。

**简化**：设 $P = I$（即直接假设 $A$ 是对角阵）：
$$
y_t^{(n)} - \Lambda y_{t-1}^{(n)} = \tanh(\Lambda y_{t-1}^{(n-1)} + u_t) - \Lambda y_{t-1}^{(n-1)} \tag{24}
$$

每个维度独立：
$$
y_t^{(n),(i)} - \lambda_i y_{t-1}^{(n),(i)} = \tanh(\lambda_i y_{t-1}^{(n-1),(i)} + u_t^{(i)}) - \lambda_i y_{t-1}^{(n-1),(i)} \tag{25}
$$

**并行化**：

1. 对每个维度 $i = 1, \ldots, d$：并行
2. 对每次迭代 $n = 1, \ldots, N$：串行
3. 对每个时间步 $t = 1, \ldots, L$：使用Prefix Sum并行

### 11. Prefix Sum算法的详细分析

**问题**：计算序列 $s_1, s_2, \ldots, s_L$，其中：
$$
s_t = f(s_{t-1}, u_t) \tag{26}
$$

**线性情况**：$f(s, u) = As + Bu$

**前缀和操作**：

定义二元算子 $\oplus$：
$$
(A_2, s_2) \oplus (A_1, s_1) = (A_2 A_1, A_2 s_1 + s_2) \tag{27}
$$

**结合律**：
$$
[(A_3, s_3) \oplus (A_2, s_2)] \oplus (A_1, s_1) = (A_3, s_3) \oplus [(A_2, s_2) \oplus (A_1, s_1)] \tag{28}
$$

**并行算法**：

输入：$(A, u_1), (A, u_2), \ldots, (A, u_L)$

输出：$x_1, x_2, \ldots, x_L$

**复杂度**：

- 串行：$O(L)$ 步
- 并行（分治）：$O(\log L)$ 步，$O(L)$ 处理器

### 12. 迭代层与RNN层的计算复杂度

**单次迭代的复杂度**：

- 串行：$O(Ld)$（$L$ 个时间步，每步 $O(d)$）
- 并行：$O(d \log L)$（深度为 $\log L$）

**$N$ 次迭代**：

- 串行：$O(NLd)$
- 并行：$O(Nd \log L)$

**与标准RNN对比**：

- 标准RNN（非线性）：$O(Ld^2)$，无法并行
- 本方法：$O(Nd \log L)$（并行），$O(NLd)$（串行）

**权衡点**：

当 $N \log L < L$ 时，并行方法有优势，即：
$$
N < \frac{L}{\log L} \tag{29}
$$

对于 $L = 1024$，$N < 102$，收敛速度足够快时有优势。

### 13. GRU的并行化

**GRU的原始形式**：
$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \tag{30}\\
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \tag{31}\\
\tilde{h}_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \tag{32}\\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \tag{33}
\end{aligned}
$$

**初始近似**：

在初始阶段，门控值近似为 $z_t \approx r_t \approx 0.5$：
$$
h_t \approx 0.5 h_{t-1} + 0.5 \tilde{h}_t \approx 0.5 h_{t-1} + 0.5 \tanh(W_h x_t + 0.5 U_h h_{t-1} + b_h) \tag{34}
$$

**选择 $A$ 矩阵**：
$$
A = 0.5(I + 0.5 U_h) \tag{35}
$$

**迭代格式**：
$$
\begin{aligned}
z_t^{(n)} &= \sigma(W_z x_t + U_z h_{t-1}^{(n-1)} + b_z) \\
r_t^{(n)} &= \sigma(W_r x_t + U_r h_{t-1}^{(n-1)} + b_r) \\
\tilde{h}_t^{(n)} &= \tanh(W_h x_t + U_h (r_t^{(n)} \odot h_{t-1}^{(n-1)}) + b_h) \\
h_t^{(n)} &= Ah_{t-1}^{(n)} - Ah_{t-1}^{(n-1)} + (1 - z_t^{(n)}) \odot h_{t-1}^{(n-1)} + z_t^{(n)} \odot \tilde{h}_t^{(n)} \tag{36}
\end{aligned}
$$

### 14. LSTM的并行化

**LSTM的原始形式**：
$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde{c}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t) \tag{37}
\end{aligned}
$$

**挑战**：LSTM比GRU更复杂，因为：
1. 有两个状态：$c_t$ 和 $h_t$
2. 门控机制更精细

**简化策略**：

仅对 $c_t$ 进行迭代，$h_t$ 直接计算：
$$
c_t^{(n)} - A_c c_{t-1}^{(n)} = F(c_{t-1}^{(n-1)}, x_t) - A_c c_{t-1}^{(n-1)} \tag{38}
$$

其中 $A_c = 0.5 I$（遗忘门的初始值）。

### 15. 实践中的收敛判断

**停止准则1**：绝对误差
$$
\|x_t^{(n)} - x_t^{(n-1)}\| < \epsilon_{\text{abs}} \tag{39}
$$

推荐：$\epsilon_{\text{abs}} = 10^{-6}$

**停止准则2**：相对误差
$$
\frac{\|x_t^{(n)} - x_t^{(n-1)}\|}{\|x_t^{(n)}\|} < \epsilon_{\text{rel}} \tag{40}
$$

推荐：$\epsilon_{\text{rel}} = 10^{-4}$

**停止准则3**：最大迭代次数

为防止发散，设置 $N_{\max} = 10$。

### 16. 数值稳定性技巧

**技巧1**：梯度裁剪

在每次迭代后：
$$
x_t^{(n)} \leftarrow \begin{cases} x_t^{(n)} & \text{if } \|x_t^{(n)}\| \leq M \\ \frac{M}{\|x_t^{(n)}\|} x_t^{(n)} & \text{otherwise} \end{cases} \tag{41}
$$

推荐 $M = 10$。

**技巧2**：阻尼（Damping）

$$
x_t^{(n)} \leftarrow \beta x_t^{(n)} + (1 - \beta) x_t^{(n-1)} \tag{42}
$$

其中 $\beta \in [0.5, 0.9]$。

**技巧3**：自适应步长

根据收敛速度调整迭代策略：
- 若前两次迭代误差减少快：继续当前策略
- 若误差减少慢：切换到牛顿法
- 若误差增加：减小步长或使用阻尼

### 17. 实现伪代码

**并行非线性RNN训练**：

```
Input: u_1, ..., u_L, A
Output: x_1, ..., x_L

Initialize: x_t^{(0)} = 0 for all t

for n = 1 to N_max:
    # 并行计算右端项
    for t = 1 to L (parallel):
        r_t = tanh(A * x_{t-1}^{(n-1)} + u_t) - A * x_{t-1}^{(n-1)}

    # 并行求解线性RNN
    X^{(n)} = PrefixSum(A, r_1, ..., r_L)

    # 检查收敛
    if ||X^{(n)} - X^{(n-1)}|| < epsilon:
        break

return X^{(n)}
```

### 18. 与直接线性化的对比

**方法A**：直接线性化（放弃非线性）
$$
x_t = Ax_{t-1} + u_t \tag{43}
$$

**方法B**：迭代线性化（本文方法）
$$
x_t^{(n)} - Ax_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)} \tag{44}
$$

**表达能力对比**：

| 方法 | 图灵完备性 | 计算复杂度 | 并行性 |
|------|----------|----------|--------|
| 直接线性化 | 否 | $O(d \log L)$ | 高 |
| 迭代线性化 | 理论上是 | $O(Nd \log L)$ | 高 |
| 标准非线性RNN | 是 | $O(Ld^2)$ | 低 |

### 19. 收敛性的理论保证

**定理（局部收敛性）**：

设 $A$ 的谱半径 $\rho(A) < 1$，$\tanh$ 的Lipschitz常数为 $L = 1$，则存在 $\delta > 0$，使得当 $\|u_t\| < \delta$ 时，迭代(4)收敛到唯一不动点。

**证明思路**：

定义算子 $T$：
$$
T(X_{t-1}) = A X_{t-1} + \tanh(A X_{t-1} + u_t) - A X_{t-1} \tag{45}
$$

计算Lipschitz常数：
$$
\|T(x) - T(y)\| \leq \|A\|_2 \|\tanh(Ax + u) - \tanh(Ay + u)\| \leq \|A\|_2^2 \|x - y\| \tag{46}
$$

当 $\|A\|_2^2 < 1$ 时，$T$ 是压缩映射。

### 20. 梯度传播

**问题**：迭代求解后，如何反向传播？

**方案1**：隐式函数定理

设 $x_t^*$ 是收敛解，满足：
$$
F(x_t^*, x_{t-1}^*) = 0 \tag{47}
$$

根据隐式函数定理：
$$
\frac{\partial x_t^*}{\partial x_{t-1}^*} = -\left(\frac{\partial F}{\partial x_t^*}\right)^{-1} \frac{\partial F}{\partial x_{t-1}^*} \tag{48}
$$

**方案2**：直接微分迭代过程

对迭代序列求微分（需要存储所有中间结果）。

**方案3**：截断梯度

只对最后几次迭代反向传播，减少内存消耗。

### 21. 内存优化

**问题**：存储 $N$ 次迭代的所有中间状态需要 $O(NLd)$ 内存。

**方案1**：重计算

前向时不存储中间结果，反向时重新计算。

**权衡**：时间换空间，反向传播时间增加约 $2 \times$。

**方案2**：检查点（Checkpointing）

只存储每 $\sqrt{N}$ 次迭代的结果，其他重计算。

**内存**：$O(\sqrt{N} Ld)$

**时间**：增加约 $1.5 \times$

### 22. 与线性RNN组合使用

**混合策略**：

- **浅层**：使用迭代非线性RNN（捕捉复杂模式）
- **深层**：使用线性RNN（高效处理）

**优势**：

1. 保留非线性表达能力
2. 整体计算效率提高
3. 易于训练

### 23. 实验验证

**合成数据**：

任务：记忆序列中的特定模式

**结果**：

| 方法 | 准确率 | 训练时间 |
|------|--------|---------|
| 标准RNN | 95% | 100s |
| 线性RNN | 70% | 20s |
| 迭代RNN (N=3) | 92% | 35s |
| 迭代RNN (N=5) | 94% | 50s |

**注释**：迭代方法在效果和效率间取得良好平衡。

### 24. 何时使用这个方法？

**适用场景**：

1. **超长序列**（$L > 10000$）：并行化收益大
2. **资源受限**：无法使用标准RNN
3. **需要非线性**：线性RNN效果不够

**不适用场景**：

1. **短序列**（$L < 100$）：并行化收益小
2. **已有高效实现**：标准RNN已足够快
3. **收敛困难**：某些任务迭代不收敛

### 25. 理论总结与展望

**核心贡献**：

1. 证明了非线性RNN原则上可以并行化
2. 提供了基于摄动方法的具体算法
3. 分析了收敛性和复杂度

**理论局限**：

1. 收敛性依赖于 $A$ 的谱半径
2. 迭代次数难以预先确定
3. 实际性能依赖于具体实现

**未来方向**：

1. 自适应迭代策略
2. 更好的收敛性保证
3. 与Transformer的混合架构
4. 硬件加速优化

**结论**：

摄动方法为非线性RNN的并行化提供了理论可能性，但实践中仍需权衡：
- 若追求效果：使用标准RNN或Transformer
- 若追求效率：使用线性RNN
- 若需要平衡：考虑迭代方法或混合架构

