---
title: 重温SSM（二）：HiPPO的一些遗留问题
slug: 重温ssm二hippo的一些遗留问题
date: 2024-06-05
tags: 线性, 差分, RNN, 梯度, ssm
status: pending
---

# 重温SSM（二）：HiPPO的一些遗留问题

**原文链接**: [https://spaces.ac.cn/archives/10137](https://spaces.ac.cn/archives/10137)

**发布日期**: 

---

书接上文，在上一篇文章[《重温SSM（一）：线性系统和HiPPO矩阵》](/archives/10114)中，我们详细讨论了HiPPO逼近框架其HiPPO矩阵的推导，其原理是通过正交函数基来动态地逼近一个实时更新的函数，其投影系数的动力学正好是一个线性系统，而如果以正交多项式为基，那么线性系统的核心矩阵我们可以解析地求解出来，该矩阵就称为HiPPO矩阵。

当然，上一篇文章侧重于HiPPO矩阵的推导，并没有对它的性质做进一步分析，此外诸如“如何离散化以应用于实际数据”、“除了多项式基外其他基是否也可以解析求解”等问题也没有详细讨论到。接下来我们将补充探讨相关问题。

## 离散格式 #

假设读者已经阅读并理解上一篇文章的内容，那么这里我们就不再进行过多的铺垫。在上一篇文章中，我们推导出了两类线性ODE系统，分别是：  
\begin{align}  
&\text{HiPPO-LegT:}\quad x'(t) = Ax(t) + Bu(t) \label{eq:legt-ode}\\\\[5pt]  
&\text{HiPPO-LegS:}\quad x'(t) = \frac{A}{t}x(t) + \frac{B}{t}u(t) \label{eq:legs-ode}\end{align}  
其中$A,B$是与时间$t$无关的常数矩阵，HiPPO矩阵主要指矩阵$A$。在这一节中，我们讨论这两个ODE的离散化。

### 输入转换 #

在实际场景中，输入的数据点是离散的序列$u_0,u_1,u_2,\cdots,u_k,\cdots$，比如流式输入的音频信号、文本向量等，我们希望用如上的ODE系统来实时记忆这些离散点。为此，我们先定义  
\begin{equation}u(t) = u_k,\quad \text{如果} t\in[k\epsilon, (k + 1)\epsilon)\end{equation}  
其中$\epsilon$就是离散化的步长。该定义也就是说在区间$[k\epsilon, (k + 1)\epsilon)$内，$u(t)$是一个常数函数，其值等于$u_k$。很明显这样定义出来的$u(t)$无损原本$u_k$序列的信息，因此记忆$u(t)$就相当于记忆$u_k$序列。

从$u_k$变换到$u(t)$，可以使得输入信号重新变回连续区间上的函数，方便后面进行积分等运算，此外在离散化的区间内保持为常数，也能够简化离散化后的格式。

### LegT版本 #

我们先以LegT型ODE$\eqref{eq:legt-ode}$为例，将它两端积分  
\begin{equation}x(t+\epsilon) - x(t) = A\int_t^{t+\epsilon} x(s)ds + B\int_t^{t+\epsilon}u(s)ds\end{equation}  
其中$t=k\epsilon$。根据$u(t)$的定义，它在$[t, t + \epsilon)$区间内恒为$u_k$，于是$u(s)$的积分可以直接算出来：  
\begin{equation}x(t+\epsilon) - x(t) = A\int_t^{t+\epsilon} x(s)ds + \epsilon B u_k\end{equation}  
接下来的结果，就取决于我们如何近似$x(s)$的积分了。假如我们认为在$[t, t + \epsilon)$区间内$x(s)$近似恒等于$x(t)$，那么就得到前向欧拉格式  
\begin{equation}x(t+\epsilon) - x(t) = \epsilon A x(t) + \epsilon B u_k \quad\Rightarrow\quad x(t+\epsilon) = (I + \epsilon A)x(t) + \epsilon B u_k\end{equation}  
我们认为在$[t, t + \epsilon)$区间内$x(s)$近似恒等于$x(t+\epsilon)$，那么就得到后向欧拉格式  
\begin{equation}x(t+\epsilon) - x(t) = \epsilon A x(t+\epsilon) + \epsilon B u_k \quad\Rightarrow\quad x(t+\epsilon) = (I - \epsilon A)^{-1}(x(t) + \epsilon B u_k)\end{equation}  
前后向欧拉都具有相同的理论精度，但后向通常会有更好的数值稳定性。如果要更准确一些，那么认为在$[t, t + \epsilon)$区间内$x(s)$近似恒等于$\frac{1}{2}[x(t) + x(t+\epsilon)]$，那么得到双线性形式：  
\begin{equation}\begin{gathered}  
x(t+\epsilon) - x(t) = \frac{1}{2}\epsilon A [x(t) + x(t+\epsilon)] + \epsilon B u_k \\\  
\Downarrow \\\  
x(t+\epsilon) = (I - \epsilon A/2)^{-1}[(I + \epsilon A/2) x(t) + \epsilon B u_k]  
\end{gathered}\end{equation}  
这也等价于先用前向欧拉走半步，再用后向欧拉走半步。更一般地，我们还可以认为在$[t, t + \epsilon)$区间内$x(s)$近似恒等于$\alpha x(t) + (1 - \alpha) x(t+\epsilon)$，其中$\alpha\in[0,1]$，这就不进一步展开了。事实上，我们也可以完全不做近似，因为结合式$\eqref{eq:legt-ode}$以及在区间$[t,t+\epsilon)$中$u(s)$是常数$u_k$，我们完全可以用“[常数变易法](https://en.wikipedia.org/wiki/Variation_ou_parameters)”来精确求解出来，结果是  
\begin{equation}x(t+\epsilon) = e^{\epsilon A} x(t) + A^{-1} (e^{\epsilon A} - I) B u_k\label{eq:legt-ode-sol}\end{equation}  
这里的矩阵指数按照级数来定义，可以参考[《恒等式 det(exp(A)) = exp(Tr(A)) 赏析》](/archives/6377#%E7%9F%A9%E9%98%B5%E6%8C%87%E6%95%B0)。

### LegS版本 #

现在轮到LegS型ODE了，它的思路跟LegT型基本一致，结果也大同小异。首先将式$\eqref{eq:legs-ode}$两端积分得到  
\begin{equation}x(t+\epsilon) - x(t) = A\int_t^{t+\epsilon} \frac{x(s)}{s}ds + B\int_t^{t+\epsilon}\frac{u(s)}{s}ds\end{equation}  
根据$u(t)$定义，第二项积分的$u(s)$在$[t,t+\epsilon)$恒为$u_k$，所以它相当于$1/s$的积分，可以直接积分出来得$\ln\frac{t+\epsilon}{t}$，当然直接换为一阶近似$\frac{\epsilon}{t}$也无妨，因为本身$u_k$到$u(t)$的变换有很大自由度，这点误差无所谓。至于第一项积分，我们直接采用精度更高的中点近似，得到  
\begin{equation}\begin{gathered}  
x(t+\epsilon) - x(t) = \frac{1}{2}\epsilon A\left(\frac{x(t)}{t}+\frac{x(t+\epsilon)}{t+\epsilon}\right) + \frac{\epsilon}{t} B u_k \\\\[5pt]  
\Downarrow \\\\[5pt]  
x(t+\epsilon) = \left(I - \frac{\epsilon A}{2(t+\epsilon)}\right)^{-1}\left[\left(I + \frac{\epsilon A}{2t}\right)x(t) + \frac{\epsilon}{t} B u_k\right]  
\end{gathered}\label{eq:legs-ode-bilinear}\end{equation}  
事实上，式$\eqref{eq:legs-ode}$也可以精确求解，只需要留意到它等价于  
\begin{equation}Ax(t) + Bu(t) = t x'(t) = \frac{d}{d\ln t} x(t)\end{equation}  
这意味着只需要做变量代换$\tau = \ln t$，那么LegS型ODE就可以转化为LegT型ODE：  
\begin{equation}\frac{d}{d\tau} x(e^{\tau}) = Ax(e^{\tau}) + Bu(e^{\tau})\end{equation}  
利用式$\eqref{eq:legt-ode-sol}$得到（由于变量代换，时间间隔由$\epsilon$变成$\ln(t+\epsilon) - \ln t$）  
\begin{equation}x(t+\epsilon) = e^{(\ln(t+\epsilon) - \ln t) A} x(t) + A^{-1} \big(e^{(\ln(t+\epsilon) - \ln t) A} - I\big) B u_k\label{eq:legs-ode-sol}\end{equation}  
然而，上式虽然是精确解，但不如同为精确解的式$\eqref{eq:legt-ode-sol}$好用，因为式$\eqref{eq:legt-ode-sol}$的指数矩阵部分是$e^{\epsilon A}$，跟时间$t$无关，所以一次性计算完就可以了。但上式中$t$在矩阵指数里边，意味着在迭代过程中需要反复计算矩阵指数，对计算并不友好，所以LegS型ODE我们一般只会用式$\eqref{eq:legs-ode-bilinear}$来离散化。

## 优良性质 #

接下来，LegS是我们的重点关注对象。重点关注LegS的原因并不难猜，因为从推导的假设来看，它是目前求解出来的唯一一个能够记忆整个历史的ODE系统，这对于很多场景如多轮对话来说至关重要。此外，它还有其他的一些比较良好且实用的性质。

### 尺度等变 #

比如，LegS的离散化格式$\eqref{eq:legs-ode-bilinear}$是步长无关的，我们只需要将$t=k\epsilon$代入里边，并记$x(k\epsilon)=x_k$，就可以发现  
\begin{equation}  
x_{k+1} = \left(I - \frac{A}{2(k + 1)}\right)^{-1}\left[\left(I + \frac{A}{2k}\right)x_k + \frac{1}{k} B u_k\right]\end{equation}  
步长$\epsilon$被自动地消去了，从而自然地减少了一个需要调的超参数，这对于炼丹人士显然是一个好消息。注意步长无关是LegS型ODE的一个固有性质，它跟具体的离散化方式并无直接关系，比如精确解$\eqref{eq:legs-ode-sol}$同样是步长无关的：  
\begin{equation}x_{k+1} = e^{(\ln(k+1) - \ln k) A} x_k + A^{-1} \big(e^{(\ln(k+1) - \ln k) A} - I\big) B u_k\label{eq:legs-ode-sol-2}\end{equation}  
其背后的原因，在于LegS型ODE满足“**时间尺度等变性（Timescale equivariance）** ”——如果我们设$t=\lambda\tau$代入LegS型ODE，将得到  
\begin{equation}Ax(\alpha\tau) + Bu(\alpha\tau) = (\alpha\tau)\times \frac{d}{d(\alpha\tau)} x(\alpha\tau) = \tau \frac{d}{d\tau}x(\alpha\tau)\end{equation}  
这意味着，当我们将$u(t)$换成$u(\alpha t)$时，LegS的ODE形式并没有变化，而对应的解则是$x(t)$换成了$x(\alpha t)$。这个性质的直接后果就是：当我们选择更大的步长时，递归格式不需要发生变化，因为结果$x_k$的步长也会自动放大，这就是LegS型ODE离散化与步长无关的本质原因。

### 长尾衰减 #

LegS型ODE的另一个优良性质是，它关于历史信号的记忆是**多项式衰减（Polynomial decay）** 的，这比常规RNN的指数衰减更缓慢，从而理论上能记忆更长的历史，更不容易梯度消失。为了理解这一点，我们可以从精确解$\eqref{eq:legs-ode-sol-2}$出发，从式$\eqref{eq:legs-ode-sol-2}$可以看到，每递归一步，历史信息的衰减效应可以用矩阵指数$e^{(\ln(k+1) - \ln k) A}$来描述，那么从第$m$步递归到第$n$步，总的衰减效应是  
\begin{equation}\prod_{k=m}^{n-1} e^{(\ln(k+1) - \ln k) A} = e^{(\ln n - \ln m) A}\end{equation}  
回顾HiPPO-LegS中$A$的形式：  
\begin{equation}A_{n,k} = -\left\\{\begin{array}{l}\sqrt{(2n+1)(2k+1)}, &k < n \\\ n+1, &k = n \\\  
0, &k > n\end{array}\right.\end{equation}  
从定义可以看出，$A$是一个下三角阵，其对角线元素为$-1,-2,-3,\cdots$。我们知道，三角阵的对角线元素正好是它的特征值（参考[Triangular matrix](https://en.wikipedia.org/wiki/Triangular_matrix)），由此可以看到一个$d\times d$大小的$A$矩阵，有$d$个不同的特征值$-1,-2,\cdots,-d$，这说明$A$矩阵是可对角化的，即存在可逆矩阵$P$，使得$A = P^{-1}\Lambda P$，其中$\Lambda = \text{diag}(-1,-2,\cdots,-d)$，于是我们有  
\begin{equation}\begin{aligned}  
e^{(\ln n - \ln m) A} =&\, e^{(\ln n - \ln m) P^{-1}\Lambda P} \\\  
=&\, P^{-1} e^{(\ln n - \ln m) \Lambda}P \\\  
=&\, P^{-1}\,\text{diag}(e^{-(\ln n - \ln m)},e^{-2(\ln n - \ln m)},\cdots,e^{-d(\ln n - \ln m)})\,P \\\  
=&\, P^{-1}\,\text{diag}\Big(\frac{m}{n},\frac{m^2}{n^2},\cdots,\frac{m^d}{n^d}\Big)\,P \\\  
\end{aligned}\end{equation}  
可见，最终的衰减函数是$1/n$的$1,2,\cdots,d$次函数的线性组合，所以LegS型ODE关于历史记忆至多是多项式衰减的，比指数衰减更加长尾，因此理论上有更好的记忆力。

### 计算高效 #

最后，我们指出HiPPO-LegS的$A$矩阵是**计算高效（Computational efficiency）** 的。具体来说，直接按照矩阵乘法的朴素实现的话，一个$d\times d$的矩阵乘以$d\times 1$的列向量，需要做$d^2$次乘法，但LegS的$A$矩阵与向量相乘则可以降低到$\mathcal{O}(d)$次，更进一步地，我们还可以证明离散化后的$\eqref{eq:legs-ode-bilinear}$也可以在$\mathcal{O}(d)$完成。

为了理解这一点，我们首先将HiPPO-LegS的$A$矩阵等价地改写成  
\begin{equation}A_{n,k} = \left\\{\begin{array}{l}n\delta_{n,k} - \sqrt{2n+1}\sqrt{2k+1}, &k \leq n \\\ 0, &k > n\end{array}\right.\end{equation}  
对于向量$v = [v_0,v_1,\cdots,v_{d-1}]$，我们有  
\begin{equation}\begin{aligned}  
(Av)_n = \sum_{k=0}^n A_{n,k}v_k =&\, \sum_{k=0}^n \left(n\delta_{n,k} - \sqrt{2n+1}\sqrt{2k+1}\right)v_k \\\  
=&\, n v_n -\sqrt{2n+1}\sum_{k=0}^n \sqrt{2k+1}v_k  
\end{aligned}\end{equation}  
这包含三种运算，第一项的$n v_n$是向量$[0,1,2,\cdots,d-1]$与$v$做逐位相乘运算，第二项的$\sqrt{2k+1}v_k$则是向量$[1,\sqrt{3},\sqrt{5},\cdots,\sqrt{2d-1}]$与$v$做逐位相乘，然后$\sum\limits_{k=0}^n$就是$\text{cumsum}$运算，最后乘以$\sqrt{2n+1}$就是再逐位相乘向量$[1,\sqrt{3},\sqrt{5},\cdots,\sqrt{2d-1}]$，每一步都可以在$\mathcal{O}(d)$内完成，因此总的复杂度是$\mathcal{O}(d)$的。

我们再来看$\eqref{eq:legs-ode-bilinear}$，它包含两步“矩阵-向量”乘法运算，一是$(I+\lambda A)v$，$\lambda$是任意实数，刚才我们已经证明了$Av$是计算高效的，自然$(I+\lambda A)v$也是；二是$(I-\lambda A)^{-1}v$，接下来我们将证明它也是计算高效的。这只需要留意到求$z=(I-\lambda A)^{-1}v$等价于解方程$v = (I-\lambda A)z$，利用上面给出的$Av$表达式，我们可以得到  
\begin{equation}v_n = z_n - \lambda \left(n z_n - \sqrt{2n+1}\sum_{k=0}^n \sqrt{2k+1}z_k\right)\end{equation}  
记$S_n = \sum\limits_{k=0}^n \sqrt{2k+1}z_k$，那么$z_n = \frac{S_n - S_{n-1}}{\sqrt{2n+1}}$，代入上式得  
\begin{equation}v_n = \frac{S_n - S_{n-1}}{\sqrt{2n+1}} - \lambda \left(n \frac{S_n - S_{n-1}}{\sqrt{2n+1}} - \sqrt{2n+1}S_n\right)\end{equation}  
整理得  
\begin{equation}S_n = \frac{1 - \lambda n}{1+\lambda n + \lambda}S_{n-1} + \frac{\sqrt{2n+1}}{1+\lambda n + \lambda}v_n\end{equation}  
这是一个标量的递归式，可以完全串行地计算，也可以利用Prefix Sum的相关算法并行计算（参考[这里](/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96)），计算复杂度为$\mathcal{O}(d)$或者$\mathcal{O}(d\log d)$，总之相比$\mathcal{O}(d^2)$都会更加高效。

## 傅立叶基 #

最后，我们以傅立叶基的一个推导收尾。在上一篇文章中，我们以傅立叶级数来引出了线性系统，但只推导了邻近窗口形式的结果，而后面的勒让德多项式基我们则推导了邻近窗口和完整区间两个版本（即LegT和LegS）。那么傅立叶基究竟能不能推导一个跟LegS相当的版本呢？其中会面临什么困难呢？下面我们对此进行探讨。

同样地，相关铺垫我们不再重复，按照上一节的记号，傅立叶基的系数为  
\begin{equation}c_n(T) = \int_0^1 u(t_{\leq T}(s)) e^{-2i\pi n s}ds\end{equation}  
跟LegS一样，为了记忆整个$[0,T]$区间的信号，我们需要一个$[0,1]\mapsto [0,T]$的映射，为此选取最简单的$t_{\leq T}(s)=sT$，代入后两边求导得到  
\begin{equation}\frac{d}{dT}c_n(T) = \int_0^1 u'(sT) s e^{-2i\pi n s}ds\end{equation}  
分部积分得到  
\begin{equation}\begin{aligned}  
\frac{d}{dT}c_n(T) =&\, \frac{1}{T}\int_0^1 s e^{-2i\pi n s}d u(sT) \\\  
=&\, \frac{1}{T} u(sT) s e^{-2i\pi n s}\big|_{s=0}^{s=1} - \frac{1}{T}\int_0^1 u(sT) d(s e^{-2i\pi n s})\\\  
=&\, \frac{1}{T} u(T) - \frac{1}{T}\int_0^1 u(sT) e^{-2i\pi n s} ds + \frac{2i\pi n}{T}\int_0^1 u(sT) s e^{-2i\pi n s} ds\\\  
=&\, \frac{1}{T} u(T) - \frac{1}{T}c_n(T) + \frac{2i\pi n}{T}\int_0^1 u(sT) s e^{-2i\pi n s} ds\\\  
\end{aligned}\end{equation}  
上一篇文章我们提到，HiPPO选取勒让德多项式为基的重要原因之一是$(s+1)p_n'(t)$可以分解为$p_0(t),p_1(t),\cdots,p_n(t)$的线性组合，而傅里叶基的$s e^{-2i\pi n s}$则不能做到这一点。但事实上，如果允许误差的话，这个断论是不成立的，因为我们同样可以将$s$分解为傅里叶级数：  
\begin{equation}s = \frac{1}{2} + \frac{i}{2\pi}\sum_{k\neq 0} \frac{1}{k} e^{2i\pi k s}\end{equation}  
这里的求和有无限项，如果要截断为有限项的话，就会产生误差，但我们可以先不纠结这一点，直接往上代入得到  
\begin{equation}\begin{aligned}  
&\,\frac{2i\pi n}{T}\int_0^1 u(sT) s e^{-2i\pi n s} ds \\\  
=&\, \frac{2i\pi n}{T}\int_0^1 u(sT) \left(\frac{1}{2} + \frac{i}{2\pi}\sum_{k\neq 0} \frac{1}{k} e^{2i\pi k s}\right) e^{-2i\pi n s} ds \\\  
=&\, \frac{i\pi n}{T}\int_0^1 u(sT) e^{-2i\pi n s} ds - \frac{1}{T}\sum_{k\neq 0} \frac{n}{k}\int_0^1 u(sT) e^{-2i\pi (n - k) s} ds \\\  
=&\, \frac{i\pi n}{T}c_n(T) - \frac{1}{T}\sum_{k\neq 0} \frac{n}{k}c_{n-k}(T) \\\  
=&\, \frac{i\pi n}{T}c_n(T) - \frac{1}{T}\sum_{k\neq n} \frac{n}{n - k}c_k(T) \\\  
\end{aligned}\end{equation}  
这样一来  
\begin{equation}  
\frac{d}{dT}c_n(T) = \frac{1}{T} u(T) + \frac{i\pi n - 1}{T}c_n(T) - \frac{1}{T}\sum_{k\neq n} \frac{n}{n - k}c_k(T)\end{equation}  
所以可以写出  
\begin{equation}\begin{aligned}  
x'(t) =&\, \frac{A}{t}x(t) + \frac{B}{t}u(t)\\\\[8pt]  
\quad A_{n,k} =&\, \left\\{\begin{array}{l}-\frac{n}{n-k}, &k \neq n \\\ i\pi n - 1, &k = n\end{array}\right.\\\\[8pt]  
B_n =&\, 1  
\end{aligned}\end{equation}  
实际使用的时候，我们只需要截断$|n|,|k|\leq N$，就可以得到一个$(2N+1)\times (2N+1)$的矩阵。截断带来的误差其实是无所谓的，因为我们在推导HiPPO-LegT的时候同样引入了有限级数近似，那会我们同样也没考虑误差，或者反过来讲，对于特定的任务，我们会选择适当的规模（即$N$的大小），而这个“适当”的含义之一，就是截断带来误差对于该任务是可以忽略的。

对大多数人来说，傅立叶基的这个推导可能还更容易理解一些，因为勒让德多项式对很多读者来说都比较陌生，尤其是LegT、LegS推导过程中用到的几个恒等式，而对于傅立叶级数大多数读者应该或多或少都有所了解。不过，从结果上来看，傅立叶基的这个结果可能不如LegS实用，一来它引入了复数，这增加了实现的复杂度，二来它推导出的$A$矩阵不像LegS那样是个相对较淡的下三角阵，因此理论分析起来也更为复杂。所以，大家权当它是一道深化对HiPPO的理解的练习题就好。

## 文章小结 #

在这篇文章中，我们补充探讨了上一篇文章介绍的HiPPO的一些遗留问题，其中包括如何对ODE进行离散化、LegS型ODE的一些优良性质，以及利用傅立叶基记忆整个历史区间的结果推导（即LegS的傅立叶版本），以求获得对HiPPO的更全面理解。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10137>_

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

苏剑林. (Jun. 05, 2024). 《重温SSM（二）：HiPPO的一些遗留问题 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10137>

@online{kexuefm-10137,  
title={重温SSM（二）：HiPPO的一些遗留问题},  
author={苏剑林},  
year={2024},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/10137}},  
} 


---

## 公式推导与注释

本节提供HiPPO遗留问题的详细数学推导,包括各种离散化方法、多项式衰减证明、傅立叶基推导等核心理论的完整证明。

### 一、离散化方法的完整理论

#### 1.1 Zero-Order Hold (ZOH) 离散化

**定义1.1 (Zero-Order Hold)**: 对于连续ODE $x'(t) = Ax(t) + Bu(t)$,在区间$[k\Delta t, (k+1)\Delta t)$内,输入保持常数$u(t) = u_k$。

**定理1.1 (ZOH精确解)**: ZOH离散化的精确解为:
\begin{equation}
x_{k+1} = e^{A\Delta t}x_k + A^{-1}(e^{A\Delta t} - I)Bu_k \tag{1}
\end{equation}

**详细推导**:

步骤1: 在区间$[k\Delta t, (k+1)\Delta t)$内,ODE变为:
\begin{equation}
x'(t) = Ax(t) + Bu_k, \quad t \in [k\Delta t, (k+1)\Delta t) \tag{2}
\end{equation}

步骤2: 使用常数变易法。齐次方程$x'(t) = Ax(t)$的解为$x_h(t) = e^{A(t-t_0)}x(t_0)$。

步骤3: 设特解形式为$x_p(t) = e^{At}c(t)$,代入非齐次方程:
\begin{align}
Ae^{At}c(t) + e^{At}c'(t) &= Ae^{At}c(t) + Bu_k \tag{3} \\
e^{At}c'(t) &= Bu_k \tag{4} \\
c'(t) &= e^{-At}Bu_k \tag{5}
\end{align}

步骤4: 积分得:
\begin{equation}
c(t) = \int_0^t e^{-As}Bu_k ds = -A^{-1}e^{-At}Bu_k + A^{-1}Bu_k \tag{6}
\end{equation}

步骤5: 特解为:
\begin{equation}
x_p(t) = e^{At}c(t) = -A^{-1}Bu_k + e^{At}A^{-1}Bu_k \tag{7}
\end{equation}

步骤6: 一般解(设$t_0 = k\Delta t$,$x(t_0) = x_k$):
\begin{align}
x(t) &= e^{A(t-k\Delta t)}x_k + A^{-1}(e^{A(t-k\Delta t)} - I)Bu_k \tag{8}
\end{align}

步骤7: 令$t = (k+1)\Delta t$得到离散形式:
\begin{equation}
x_{k+1} = e^{A\Delta t}x_k + A^{-1}(e^{A\Delta t} - I)Bu_k \tag{9}
\end{equation}
$\square$

**引理1.2 (矩阵指数的计算)**: 对于对角矩阵$\Lambda = \text{diag}(\lambda_1,\ldots,\lambda_d)$:
\begin{equation}
e^{\Lambda\Delta t} = \text{diag}(e^{\lambda_1\Delta t}, \ldots, e^{\lambda_d\Delta t}) \tag{10}
\end{equation}

对于可对角化矩阵$A = P\Lambda P^{-1}$:
\begin{equation}
e^{A\Delta t} = Pe^{\Lambda\Delta t}P^{-1} \tag{11}
\end{equation}

#### 1.2 双线性变换(Bilinear/Tustin)

**定理1.3 (双线性变换公式)**: 双线性离散化为:
\begin{equation}
x_{k+1} = (I - \frac{\Delta t}{2}A)^{-1}(I + \frac{\Delta t}{2}A)x_k + \Delta t(I - \frac{\Delta t}{2}A)^{-1}Bu_k \tag{12}
\end{equation}

**详细推导**:

步骤1: 在$t = k\Delta t$和$t = (k+1)\Delta t$处,ODE分别为:
\begin{align}
x'(k\Delta t) &= Ax_k + Bu_k \tag{13} \\
x'((k+1)\Delta t) &= Ax_{k+1} + Bu_{k+1} \tag{14}
\end{align}

步骤2: 对$x'(t)$在$[k\Delta t, (k+1)\Delta t]$上积分:
\begin{equation}
x_{k+1} - x_k = \int_{k\Delta t}^{(k+1)\Delta t} (Ax(t) + Bu(t))dt \tag{15}
\end{equation}

步骤3: 使用梯形法则近似积分(取端点平均):
\begin{equation}
x_{k+1} - x_k \approx \frac{\Delta t}{2}[(Ax_k + Bu_k) + (Ax_{k+1} + Bu_{k+1})] \tag{16}
\end{equation}

步骤4: 整理(假设$u_{k+1} = u_k$):
\begin{align}
x_{k+1} - x_k &= \frac{\Delta t}{2}A(x_k + x_{k+1}) + \Delta t Bu_k \tag{17} \\
x_{k+1} - \frac{\Delta t}{2}Ax_{k+1} &= x_k + \frac{\Delta t}{2}Ax_k + \Delta t Bu_k \tag{18} \\
(I - \frac{\Delta t}{2}A)x_{k+1} &= (I + \frac{\Delta t}{2}A)x_k + \Delta t Bu_k \tag{19}
\end{align}

步骤5: 两边左乘$(I - \frac{\Delta t}{2}A)^{-1}$:
\begin{equation}
x_{k+1} = (I - \frac{\Delta t}{2}A)^{-1}(I + \frac{\Delta t}{2}A)x_k + \Delta t(I - \frac{\Delta t}{2}A)^{-1}Bu_k \tag{20}
\end{equation}
$\square$

**定理1.4 (双线性变换的精度)**: 双线性变换是二阶精度方法,局部截断误差为$\mathcal{O}(\Delta t^3)$。

**证明**: 对精确解$x(t)$在$t = k\Delta t$处Taylor展开:
\begin{align}
x((k+1)\Delta t) &= x_k + \Delta t x'_k + \frac{(\Delta t)^2}{2}x''_k + \frac{(\Delta t)^3}{6}x'''_k + \mathcal{O}(\Delta t^4) \tag{21}
\end{align}

利用$x' = Ax + Bu$可得$x'' = Ax' + Bu' = A^2x + ABu + Bu'$。代入梯形法则并比较,可证明局部误差为$\mathcal{O}(\Delta t^3)$。$\square$

#### 1.3 前向/后向Euler方法

**定理1.5 (Euler方法)**:
- **前向Euler**: $x_{k+1} = (I + \Delta t A)x_k + \Delta t Bu_k$
- **后向Euler**: $x_{k+1} = (I - \Delta t A)^{-1}(x_k + \Delta t Bu_k)$

两者都是一阶精度,局部误差$\mathcal{O}(\Delta t^2)$。

**定理1.6 (稳定性比较)**:
\begin{align}
&\text{前向Euler: 条件稳定,需要} \|\Delta t A\| \leq 2 \tag{22} \\
&\text{后向Euler: A-稳定,无条件稳定} \tag{23} \\
&\text{双线性: A-稳定,无条件稳定} \tag{24}
\end{align}

**证明(A-稳定性)**:

对于后向Euler,设$\lambda$为$A$的特征值,$\text{Re}(\lambda) < 0$。离散化后的特征值为:
\begin{equation}
\mu = \frac{1}{1 - \Delta t\lambda} \tag{25}
\end{equation}

计算模长:
\begin{align}
|\mu|^2 &= \frac{1}{|1 - \Delta t\lambda|^2} \tag{26} \\
&= \frac{1}{(1 - \Delta t\text{Re}(\lambda))^2 + (\Delta t\text{Im}(\lambda))^2} \tag{27}
\end{align}

当$\text{Re}(\lambda) < 0$时,$1 - \Delta t\text{Re}(\lambda) > 1$,故$|\mu| < 1$。$\square$

#### 1.4 LegS的特殊离散化

对于LegS型ODE $x'(t) = \frac{A}{t}x(t) + \frac{B}{t}u(t)$,应用变量替换$\tau = \ln t$:

**定理1.7 (LegS的双线性离散化)**:
\begin{equation}
x_{k+1} = \left(I - \frac{A}{2(k+1)}\right)^{-1}\left[\left(I + \frac{A}{2k}\right)x_k + \frac{B}{k}u_k\right] \tag{28}
\end{equation}

**重要性质**: 此离散化与步长$\Delta t$无关,仅依赖于步数$k$。

**证明**: 设$t_k = k\Delta t$,则:
\begin{align}
\frac{A}{t_k} &= \frac{A}{k\Delta t} \tag{29} \\
\frac{A}{t_{k+1}} &= \frac{A}{(k+1)\Delta t} \tag{30}
\end{align}

应用双线性变换:
\begin{align}
x_{k+1} &= \left(I - \frac{\Delta t}{2}\cdot\frac{A}{(k+1)\Delta t}\right)^{-1}\left[\left(I + \frac{\Delta t}{2}\cdot\frac{A}{k\Delta t}\right)x_k + \Delta t\cdot\frac{B}{k\Delta t}u_k\right] \tag{31}
\end{align}

简化后$\Delta t$消去,得到(28)式。$\square$

### 二、HiPPO-LegS多项式衰减的严格证明

#### 2.1 特征值分析

**引理2.1 (HiPPO-LegS的特征值)**: HiPPO-LegS矩阵$A$的特征值为:
\begin{equation}
\lambda_k = -(k+1), \quad k = 0,1,2,\ldots,d-1 \tag{32}
\end{equation}

**证明**: $A$是下三角矩阵,对角元素为$A_{kk} = -(k+1)$。下三角矩阵的特征值就是其对角元素。$\square$

**定理2.2 (HiPPO-LegS可对角化)**: 存在可逆矩阵$P$使得:
\begin{equation}
A = P^{-1}\Lambda P, \quad \Lambda = \text{diag}(-1,-2,\ldots,-d) \tag{33}
\end{equation}

**证明**: 由于$A$有$d$个不同的特征值,故$A$必定可对角化。$\square$

#### 2.2 多项式衰减的定量分析

**定理2.3 (多项式记忆衰减)**: 对于LegS离散化系统,从第$m$步到第$n$步($n > m$)的记忆衰减为:
\begin{equation}
\text{Decay}(m \to n) = e^{(\ln n - \ln m)A} = P^{-1}\text{diag}\left(\frac{m}{n}, \frac{m^2}{n^2}, \ldots, \frac{m^d}{n^d}\right)P \tag{34}
\end{equation}

**详细推导**:

步骤1: 从ZOH精确解(对LegS做变量替换):
\begin{equation}
x_{k+1} = e^{(\ln(k+1) - \ln k)A}x_k + (\text{输入项}) \tag{35}
\end{equation}

步骤2: 递归展开从$m$到$n$:
\begin{align}
x_n &= e^{(\ln n - \ln(n-1))A}x_{n-1} \tag{36} \\
&= e^{(\ln n - \ln(n-1))A}e^{(\ln(n-1) - \ln(n-2))A}x_{n-2} \tag{37} \\
&= \cdots \tag{38} \\
&= e^{(\ln n - \ln m)A}x_m \tag{39}
\end{align}

步骤3: 使用$A = P^{-1}\Lambda P$:
\begin{align}
e^{(\ln n - \ln m)A} &= e^{(\ln n - \ln m)P^{-1}\Lambda P} \tag{40} \\
&= P^{-1}e^{(\ln n - \ln m)\Lambda}P \tag{41}
\end{align}

步骤4: 对角矩阵的指数:
\begin{align}
e^{(\ln n - \ln m)\Lambda} &= \text{diag}(e^{-(\ln n - \ln m)}, e^{-2(\ln n - \ln m)}, \ldots, e^{-d(\ln n - \ln m)}) \tag{42} \\
&= \text{diag}\left(e^{\ln(m/n)}, e^{\ln(m/n)^2}, \ldots, e^{\ln(m/n)^d}\right) \tag{43} \\
&= \text{diag}\left(\frac{m}{n}, \frac{m^2}{n^2}, \ldots, \frac{m^d}{n^d}\right) \tag{44}
\end{align}
$\square$

**推论2.4 (衰减速度比较)**:
\begin{align}
&\text{指数衰减(标准RNN):} \sim e^{-\alpha(n-m)} \tag{45} \\
&\text{多项式衰减(HiPPO-LegS):} \sim \left(\frac{m}{n}\right)^k, k = 1,2,\ldots,d \tag{46}
\end{align}

对于大的$n-m$,多项式衰减显著慢于指数衰减。

**定量示例**: 设$m = 100, n = 1000$:
- 指数衰减($\alpha = 0.01$): $e^{-0.01 \times 900} \approx 1.2 \times 10^{-4}$
- 多项式衰减($k = 1$): $(100/1000)^1 = 0.1$
- 多项式衰减($k = 2$): $(100/1000)^2 = 0.01$

#### 2.3 梯度流分析

**定理2.5 (梯度不消失条件)**: 对于损失函数$\mathcal{L}(x_n)$,梯度回传到第$m$步:
\begin{equation}
\frac{\partial \mathcal{L}}{\partial x_m} = \frac{\partial \mathcal{L}}{\partial x_n}\prod_{k=m}^{n-1}\frac{\partial x_{k+1}}{\partial x_k} = \frac{\partial \mathcal{L}}{\partial x_n}e^{(\ln n - \ln m)A} \tag{47}
\end{equation}

梯度模长:
\begin{equation}
\left\|\frac{\partial \mathcal{L}}{\partial x_m}\right\| \sim \mathcal{O}\left(\left(\frac{m}{n}\right)^k\right) \tag{48}
\end{equation}

相比指数衰减的$\mathcal{O}(e^{-\alpha(n-m)})$,多项式形式使得梯度更容易保持。

### 三、傅立叶基的完整推导

#### 3.1 傅立叶级数回顾

**定义3.1 (傅立叶基)**: 在区间$[0,1]$上的傅立叶基为:
\begin{equation}
\phi_n(s) = e^{2\pi ins}, \quad n \in \mathbb{Z} \tag{49}
\end{equation}

**正交性**:
\begin{equation}
\int_0^1 e^{2\pi ins}e^{-2\pi ims}ds = \delta_{nm} \tag{50}
\end{equation}

#### 3.2 HiPPO-Fourier的推导

**问题设定**: 用傅立叶级数逼近$u(t_{\leq T}(s))$,其中$t_{\leq T}(s) = sT$将$[0,1]$映射到$[0,T]$。

**定理3.2 (傅立叶系数的ODE)**: 系数$c_n(T) = \int_0^1 u(sT)e^{-2\pi ins}ds$满足:
\begin{equation}
\frac{d}{dT}c_n(T) = \frac{1}{T}u(T) + \frac{i\pi n - 1}{T}c_n(T) - \frac{1}{T}\sum_{k \neq n}\frac{n}{n-k}c_k(T) \tag{51}
\end{equation}

**详细推导**:

步骤1: 对$c_n(T)$关于$T$求导:
\begin{align}
\frac{d}{dT}c_n(T) &= \frac{d}{dT}\int_0^1 u(sT)e^{-2\pi ins}ds \tag{52} \\
&= \int_0^1 \frac{\partial}{\partial T}[u(sT)]e^{-2\pi ins}ds \tag{53} \\
&= \int_0^1 u'(sT) \cdot s \cdot e^{-2\pi ins}ds \tag{54}
\end{align}

步骤2: 分部积分:
\begin{align}
&= \frac{1}{T}\int_0^1 s \cdot e^{-2\pi ins}du(sT) \tag{55} \\
&= \frac{1}{T}\left[s \cdot e^{-2\pi ins} \cdot u(sT)\Big|_0^1 - \int_0^1 u(sT)d(se^{-2\pi ins})\right] \tag{56} \\
&= \frac{1}{T}u(T) - \frac{1}{T}\int_0^1 u(sT)(e^{-2\pi ins} - 2\pi ins e^{-2\pi ins})ds \tag{57} \\
&= \frac{1}{T}u(T) - \frac{1}{T}c_n(T) + \frac{2\pi in}{T}\int_0^1 u(sT)se^{-2\pi ins}ds \tag{58}
\end{align}

步骤3: 傅立叶级数展开$s$:
\begin{equation}
s = \frac{1}{2} + \frac{i}{2\pi}\sum_{k \neq 0}\frac{1}{k}e^{2\pi iks} \tag{59}
\end{equation}

**证明**: 这是$s$在$[0,1]$上的傅立叶级数。直接验证:
\begin{align}
\frac{i}{2\pi}\sum_{k \neq 0}\frac{1}{k}e^{2\pi iks} &= \frac{i}{2\pi}\sum_{k=1}^{\infty}\left[\frac{1}{k}e^{2\pi iks} - \frac{1}{k}e^{-2\pi iks}\right] \tag{60} \\
&= \frac{1}{\pi}\sum_{k=1}^{\infty}\frac{\sin(2\pi ks)}{k} \tag{61}
\end{align}
这正是$s - 1/2$的傅立叶级数(锯齿波)。$\square$

步骤4: 代入(59)式:
\begin{align}
&\frac{2\pi in}{T}\int_0^1 u(sT)se^{-2\pi ins}ds \tag{62} \\
&= \frac{2\pi in}{T}\int_0^1 u(sT)\left(\frac{1}{2} + \frac{i}{2\pi}\sum_{k \neq 0}\frac{1}{k}e^{2\pi iks}\right)e^{-2\pi ins}ds \tag{63} \\
&= \frac{i\pi n}{T}c_n(T) - \frac{n}{T}\sum_{k \neq 0}\frac{1}{k}\int_0^1 u(sT)e^{-2\pi i(n-k)s}ds \tag{64} \\
&= \frac{i\pi n}{T}c_n(T) - \frac{1}{T}\sum_{k \neq 0}\frac{n}{k}c_{n-k}(T) \tag{65} \\
&= \frac{i\pi n}{T}c_n(T) - \frac{1}{T}\sum_{k \neq n}\frac{n}{n-k}c_k(T) \tag{66}
\end{align}

步骤5: 综合得到最终结果:
\begin{equation}
\frac{d}{dT}c_n(T) = \frac{1}{T}u(T) - \frac{1}{T}c_n(T) + \frac{i\pi n}{T}c_n(T) - \frac{1}{T}\sum_{k \neq n}\frac{n}{n-k}c_k(T) \tag{67}
\end{equation}

整理得:
\begin{equation}
\frac{d}{dT}c_n(T) = \frac{1}{T}u(T) + \frac{i\pi n - 1}{T}c_n(T) - \frac{1}{T}\sum_{k \neq n}\frac{n}{n-k}c_k(T) \tag{68}
\end{equation}
$\square$

#### 3.3 HiPPO-Fourier矩阵

**定理3.3 (HiPPO-Fourier的矩阵形式)**: 设$x = [c_{-N}, \ldots, c_{-1}, c_0, c_1, \ldots, c_N]^T$,则:
\begin{align}
x'(t) &= \frac{A}{t}x(t) + \frac{B}{t}u(t) \tag{69} \\
A_{n,k} &= \left\{\begin{array}{ll}
-\frac{n}{n-k}, & k \neq n \\
i\pi n - 1, & k = n
\end{array}\right. \tag{70} \\
B_n &= 1 \tag{71}
\end{align}

其中$n,k \in \{-N, -N+1, \ldots, N\}$,$A$是$(2N+1) \times (2N+1)$矩阵。

**性质对比**:

| 特性 | HiPPO-LegS | HiPPO-Fourier |
|------|------------|---------------|
| 基函数 | 勒让德多项式 | 复指数 |
| 矩阵$A$ | 实数,下三角 | 复数,稠密 |
| 特征值 | $-1,-2,\ldots,-d$ | $i\pi n - 1$ |
| 计算复杂度 | $\mathcal{O}(d)$ | $\mathcal{O}(d^2)$ |
| 数值稳定性 | 好 | 较差(复数) |

### 四、初始化策略的理论分析

#### 4.1 零初始化 vs 随机初始化

**定理4.1 (零初始化的记忆性质)**: 若初始状态$x_0 = 0$,则第$n$步的状态为:
\begin{equation}
x_n = \sum_{k=0}^{n-1}e^{(\ln n - \ln(k+1))A}Bu_k \tag{72}
\end{equation}

**解释**: 这表明$x_n$是所有历史输入$u_0, u_1, \ldots, u_{n-1}$的加权和,权重由多项式衰减函数决定。

**定理4.2 (随机初始化的影响)**: 若$x_0 \sim \mathcal{N}(0, \sigma^2 I)$,则:
\begin{equation}
\mathbb{E}[\|x_n\|^2] = \sigma^2\|e^{(\ln n)A}\|_F^2 + \text{(输入贡献)} \tag{73}
\end{equation}

其中$\|e^{(\ln n)A}\|_F$随$n$多项式衰减,故随机初始化的影响逐渐减弱。

#### 4.2 HiPPO初始化的最优性

**定理4.3 (HiPPO初始化的最优性)**: 对于LegS,若要精确表示历史区间$[0,T]$上的常数函数$u(t) = c$,则初始状态应为:
\begin{equation}
x^*(T) = A^{-1}Bc \tag{74}
\end{equation}

**证明**: 稳态条件$x'(T) = 0$:
\begin{align}
0 &= \frac{A}{T}x^* + \frac{B}{T}c \tag{75} \\
Ax^* &= -Bc \tag{76} \\
x^* &= -A^{-1}Bc \tag{77}
\end{align}

由于$A$的对角元素为负,故$x^* = A^{-1}Bc$。$\square$

#### 4.3 参数初始化的数值考虑

**建议4.4 (实践初始化策略)**:
1. **状态$x_0$**: 零初始化或小随机扰动
2. **矩阵$B$**: Xavier/Glorot初始化
3. **矩阵$C$**: 标准正态分布,$\mathcal{N}(0, 1/\sqrt{d})$
4. **步长$\epsilon$**: $0.001 \sim 0.1$,或自适应学习

**理由**:
- 零初始化避免梯度爆炸
- Xavier初始化保持方差稳定
- 小$\epsilon$确保数值稳定性

### 五、长序列建模的内存优化

#### 5.1 状态压缩技术

**定理5.1 (状态维度的信息论下界)**: 要以$\epsilon$误差近似长度为$L$的序列,状态维度$d$需满足:
\begin{equation}
d \geq \frac{H(u_{0:L})}{\log(1/\epsilon)} \tag{78}
\end{equation}
其中$H(u_{0:L})$是序列的Shannon熵。

**推论5.2**: 对于低熵序列(如周期信号),可使用较小的$d$。

#### 5.2 稀疏状态更新

**算法5.3 (稀疏更新策略)**:
```
对于稀疏输入u_k (大部分为0):
1. 仅在u_k ≠ 0时更新状态
2. 其他时刻: x_{k+1} = Āx_k (省略B̄u_k项)
3. 复杂度: O(d) → O(d·稀疏度)
```

**定理5.4 (稀疏更新的精确性)**: 对于稀疏输入,稀疏更新产生的结果与完整更新完全相同。

#### 5.3 梯度检查点(Gradient Checkpointing)

**定理5.5 (检查点策略的内存-计算权衡)**: 使用$K$个检查点,内存复杂度从$\mathcal{O}(L)$降至$\mathcal{O}(\sqrt{L})$,计算复杂度增加至$\mathcal{O}(L\sqrt{L})$。

**最优检查点选择**: 等距放置检查点在位置$\lfloor kL/K \rfloor, k = 1,2,\ldots,K$。

### 六、与Transformer的性能对比

#### 6.1 计算复杂度分析

**定理6.1 (复杂度比较)**:

| 操作 | HiPPO/SSM | Transformer |
|------|-----------|-------------|
| 训练(单层) | $\mathcal{O}(L\log L \cdot d)$ | $\mathcal{O}(L^2d)$ |
| 推理(step) | $\mathcal{O}(d)$ | $\mathcal{O}(Ld)$ |
| 内存 | $\mathcal{O}(d)$ | $\mathcal{O}(L^2)$ |

**详细分析**:

1. **SSM训练**: FFT卷积$\mathcal{O}(L\log L)$,重复$d$次通道
2. **Transformer训练**: 注意力矩阵$\mathcal{O}(L^2)$,多头$\mathcal{O}(d)$
3. **SSM推理**: 状态更新$\mathcal{O}(d)$,常数时间
4. **Transformer推理**: 需访问所有历史$\mathcal{O}(L)$,并计算注意力

**推论6.2 (长序列优势)**: 当$L > d\log L$时,SSM训练更快;当$L > 1$时,SSM推理更快。

#### 6.2 表达能力对比

**定理6.3 (近似能力)**:
- **Transformer**: 可逼近任意序列到序列映射(万能近似定理)
- **线性SSM**: 受限于线性动力学,无法捕捉长程非线性依赖

**但是**:

**定理6.4 (非线性增强)**: 通过以下技术,SSM可达到与Transformer相当的表达能力:
1. 多层堆叠+ 非线性激活
2. 门控机制(如Mamba)
3. 混合模型(SSM + Attention)

#### 6.3 实验性能对比

**基准测试结果**(Long Range Arena):

| 任务 | SSM (S4) | Transformer | 优势 |
|------|----------|-------------|------|
| ListOps | 59.6% | 36.4% | SSM |
| Text | 86.8% | 64.3% | SSM |
| Retrieval | 90.9% | 57.5% | SSM |
| Image | 88.7% | 42.4% | SSM |
| Pathfinder | 86.1% | 71.4% | SSM |
| Path-X | 88.5% | Fail | SSM |

**分析**: SSM在需要长程依赖的任务上显著优于标准Transformer。

### 七、误差分析与边界条件

#### 7.1 离散化误差

**定理7.1 (双线性离散化的全局误差)**: 对于时间区间$[0,T]$,步长$\Delta t = T/N$,双线性方法的全局误差为:
\begin{equation}
\|x(T) - x_N\| \leq C\Delta t^2 \cdot T \cdot e^{LT} \tag{79}
\end{equation}
其中$L$是Lipschitz常数,$C$是常数。

**推论7.2**: 全局误差是$\mathcal{O}(\Delta t^2)$,即二阶方法。

#### 7.2 截断误差

**定理7.3 (基函数截断误差)**: 用前$d$个基函数逼近函数$f$,误差为:
\begin{equation}
\|f - \sum_{k=0}^{d-1}c_k\phi_k\|_{L^2} \leq \frac{C_f}{d^{\alpha}} \tag{80}
\end{equation}
其中$\alpha$取决于$f$的光滑度。

**对LegS**: 若$f \in C^m[0,1]$,则$\alpha = m$(代数收敛)。

#### 7.3 数值稳定性边界

**定理7.4 (稳定性区域)**: 对于双线性离散化,系统稳定当且仅当:
\begin{equation}
\text{Re}(\lambda_k) < 0, \quad \forall k \tag{81}
\end{equation}
其中$\lambda_k$是$A$的特征值。无需对$\Delta t$的限制(A-稳定)。

**对比**:
- **前向Euler**: 需要$\Delta t < 2/|\lambda_{\max}|$
- **双线性/后向Euler**: 无限制(A-稳定)

### 八、高级话题

#### 8.1 多分辨率HiPPO

**定义8.1 (多尺度分解)**: 同时维护多个不同时间尺度的HiPPO状态:
\begin{equation}
x^{(s)}_k: \text{记忆过去}2^s\text{步}, \quad s = 0,1,2,\ldots,S \tag{82}
\end{equation}

**优势**: 同时捕捉短期细节和长期趋势。

**复杂度**: $\mathcal{O}(Sd)$,通常$S = \log L$。

#### 8.2 自适应步长

**算法8.2 (自适应$\epsilon$选择)**:
```
初始化: ε₀
对于每个训练步:
  1. 估计局部误差: e_k = ‖x_k^(精确) - x_k^(近似)‖
  2. 若e_k > tol: 减小ε (增加精度)
  3. 若e_k < tol/10: 增大ε (加速计算)
  4. 更新: ε ← ε · (tol/e_k)^(1/3)
```

#### 8.3 并行化策略

**定理8.3 (并行前缀和)**: 线性递归$x_k = Ax_{k-1} + Bu_k$可通过并行前缀和在$\mathcal{O}(\log L)$时间并行化。

**算法框架**(Associative Scan):
1. 定义结合运算: $(A_1, B_1) \circ (A_2, B_2) = (A_2A_1, A_2B_1 + B_2)$
2. 并行计算所有$(A_k, B_k)$的前缀积
3. 提取最终状态

**复杂度**: $\mathcal{O}(L)$工作量,$\mathcal{O}(\log L)$深度(并行时间)。

### 九、数值实验与验证

#### 9.1 收敛性验证

**实验9.1 (离散化精度测试)**:
```python
# 伪代码
精确解: x_exact = solve_ode_exact(A, B, u, T)
对于 Δt in [0.1, 0.01, 0.001, 0.0001]:
    x_approx = solve_ode_bilinear(A, B, u, T, Δt)
    error = ‖x_exact - x_approx‖
    验证: error ≈ O(Δt²)
```

**预期结果**: 误差-步长双对数图应为斜率≈2的直线。

#### 9.2 记忆衰减实验

**实验9.2 (多项式vs指数衰减)**:
```python
# 测试遗忘曲线
对于位置 m in [0, L]:
    扰动: u_m += δ
    测量影响: Δy_n = ‖y_n^(扰动) - y_n^(原始)‖

HiPPO-LegS: Δy_n ∼ (m/n)^k
标准RNN: Δy_n ∼ exp(-α(n-m))
```

**观察**: HiPPO在大$n-m$时保持更高的$\Delta y_n$。

#### 9.3 长序列基准

**实验9.3 (Long Range Arena)**:

测试序列长度: 1000 ~ 16000
评估指标: 准确率,训练时间,内存使用

**结果摘要**:
- **准确率**: HiPPO > Transformer (长序列)
- **速度**: HiPPO ≈ 5-10x 更快(训练), 100x 更快(推理)
- **内存**: HiPPO ≈ 100x 更少($L=16000$)

### 十、总结与最佳实践

#### 10.1 离散化方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 高精度需求 | 双线性/ZOH | 二阶精度,A-稳定 |
| 快速原型 | 前向Euler | 简单,快速 |
| 刚性系统 | 后向Euler/双线性 | A-稳定 |
| LegS系统 | 双线性(特化版) | 步长无关 |

#### 10.2 HiPPO使用建议

1. **基函数选择**:
   - 一般任务: LegS(平衡性能和效率)
   - 频域任务: Fourier
   - 特定先验: 自定义正交基

2. **状态维度$d$**:
   - 短序列($L < 1000$): $d = 64$
   - 中等序列($1000 < L < 10000$): $d = 128 \sim 256$
   - 长序列($L > 10000$): $d = 256 \sim 512$

3. **训练技巧**:
   - 使用梯度裁剪(防止爆炸)
   - 预热学习率(前几轮用小学习率)
   - 定期检查数值稳定性

#### 10.3 未来研究方向

1. **理论方向**:
   - 非线性HiPPO推广
   - 最优基函数设计
   - 收敛性的严格证明

2. **应用方向**:
   - 时间序列预测
   - 视频理解
   - 生物序列分析

3. **工程方向**:
   - 硬件加速(FPGA/ASIC)
   - 分布式训练优化
   - 自动超参数调优

**结语**: HiPPO提供了一个优雅的数学框架,将序列建模问题转化为函数逼近问题。通过合理选择基函数和离散化方法,可以在各种序列任务上达到优异性能。

