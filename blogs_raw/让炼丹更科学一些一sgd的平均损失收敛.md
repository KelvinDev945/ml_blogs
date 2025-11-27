---
title: 让炼丹更科学一些（一）：SGD的平均损失收敛
slug: 让炼丹更科学一些一sgd的平均损失收敛
date: 2023-12-19
tags: 优化器, 不等式, 优化器, sgd, 炼丹
status: completed
---

# 让炼丹更科学一些（一）：SGD的平均损失收敛

**原文链接**: [https://spaces.ac.cn/archives/9902](https://spaces.ac.cn/archives/9902)

**发布日期**: 

---

很多时候我们将深度学习模型的训练过程戏称为“炼丹”，因为整个过程跟古代的炼丹术一样，看上去有一定的科学依据，但整体却给人一种“玄之又玄”的感觉。尽管本站之前也关注过一些[优化器](/tag/%E4%BC%98%E5%8C%96%E5%99%A8/)相关的工作，甚至也写过[《从动力学角度看优化算法》](/search/%E4%BB%8E%E5%8A%A8%E5%8A%9B%E5%AD%A6%E8%A7%92%E5%BA%A6%E7%9C%8B%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/)系列，但都是比较表面的介绍，并没有涉及到更深入的理论。为了让以后的炼丹更科学一些，笔者决定去补习一些优化相关的理论结果，争取让炼丹之路多点理论支撑。

在本文中，我们将学习随机梯度下降（SGD）的一个非常基础的收敛结论。虽然现在看来，该结论显得很粗糙且不实用，但它是优化器收敛性证明的一次非常重要的尝试，特别是它考虑了我们实际使用的是 _随机_ 梯度下降（SGD）而不是 _全量_ 梯度下降（GD）这一特性，使得结论更加具有参考意义。

## 问题设置 #

设损失函数是$L(\boldsymbol{x},\boldsymbol{\theta})$，其实$\boldsymbol{x}$是训练集，而$\boldsymbol{\theta}\in\mathbb{R}^d$是训练参数。受限于算力，我们通常只能执行随机梯度下降（SGD），即每步只能采样一个训练子集来计算损失函数并更新参数，假设采样是独立同分布的，第$t$步采样到的子集为$\boldsymbol{x}_t$，那么我们可以合理地认为实际优化的最终目标是  
\begin{equation}L(\boldsymbol{\theta}) = \lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta})\label{eq:loss}\end{equation}  
实际情况下，我们也只能训练有限步，所以我们假设$T$是一个足够大的正整常数。我们的目标是寻找$L(\boldsymbol{\theta})$的最小值点，即希望找到$\boldsymbol{\theta}^*$：  
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}\in\mathbb{R}^d} L(\boldsymbol{\theta})\label{eq:argmin}\end{equation}  
现在，我们考虑如下SGD迭代：  
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\label{eq:sgd}\end{equation}  
其中$\eta_t > 0$是学习率，其中$\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta})\triangleq \nabla_{\boldsymbol{\theta}}L(\boldsymbol{x}_t,\boldsymbol{\theta})$是$L(\boldsymbol{x}_t,\boldsymbol{\theta})$关于$\boldsymbol{\theta}$的梯度。我们的任务就是分析如此迭代下去，$\boldsymbol{\theta}_t$是否能够收敛到到目标点$\boldsymbol{\theta}^*$。

## 结论初探 #

首先，我们给出最终要证明的不等式：在适当的假设之下，有  
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{D^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T\eta_t\label{neq:core}\end{equation}

其中$D,G$是跟优化过程无关的常数。后面我们会逐一介绍“适当的假设”具体是什么，在此之前我们先来观察一下不等式$\eqref{neq:core}$所表达的具体含义：

> 1、左端第一项，是优化过程中每一步的损失函数的平均结果；
> 
> 2、左端第二项，根据式$\eqref{eq:loss}$，当$T$足够大时可以认为它就是$L(\boldsymbol{\theta}^*)$；
> 
> 3、左端合并起来，就是优化过程中损失函数的平均与损失函数的理论最小值之差；
> 
> 4、右端是一个只与学习率策略$\\{\eta_t\\}$有关的式子。

综合1、2、3、4点，不等式$\eqref{neq:core}$就是说：在适当的假设之下，SGD的平均损失与我们要寻找的理想目标的差距，可以被一个只与学习率策略有关的式子控制，如果我们可以选择适当的学习率让该式趋于零，那么意味着SGD的平均损失一定能收敛到理论最优点。（当然，从理论上来说，该结论只能保证找到损失函数的最小值$L(\boldsymbol{\theta}^*)$，但无法保证找到具体的最小值点$\boldsymbol{\theta}^*$。）

说白了，这就是关于SGD在什么情况下会收敛的一个理论结果。对了，不等式$\eqref{neq:core}$左端有一个特别的名字，叫做“遗憾”（Regret，有些教程也直接翻译为“悔”）。

## 两个例子 #

例如，假设学习率是常数$\eta$，那么我们有不等式$\eqref{neq:core}$右端有  
\begin{equation}\frac{D^2}{2T\eta} + \frac{G^2}{2T}\sum_{t=1}^T\eta = \frac{D^2}{2T\eta} + \frac{G^2\eta}{2}\geq \frac{DG}{\sqrt{T}}\end{equation}  
等号成立时$\eta=\frac{D}{G\sqrt{T}}$，也就是说学习率取常数$\frac{D}{G\sqrt{T}}$，那么就有  
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{DG}{\sqrt{T}}\label{neq:case-1}\end{equation}  
当$T\to\infty$时，右端趋于零，这意味着当训练步数$T$足够大时，将学习率设为常数$\frac{D}{G\sqrt{T}}$，就可以让SGD迭代的平均与理论最优点的差距任意小。

另一个例子是考虑衰减策略$\eta_t = \frac{\alpha}{\sqrt{t}}$，利用  
\begin{equation}\sum_{t=1}^T \frac{1}{\sqrt{t}} = 1+\sum_{t=2}^T \frac{1}{\sqrt{t}}\leq 1+\sum_{t=2}^T \frac{2}{\sqrt{t-1} + \sqrt{t}}=1+\sum_{t=2}^T 2(\sqrt{t}-\sqrt{t-1})=2\sqrt{T}-1 < 2\sqrt{T}\end{equation}  
代入式$\eqref{neq:core}$得到  
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*) < \frac{D^2}{2\alpha\sqrt{T}} + \frac{G^2\alpha}{\sqrt{T}}\label{neq:case-2}\end{equation}  
式$\eqref{neq:case-2}$和式$\eqref{neq:case-1}$关于$T$都是$\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$的，因此理论上它们都能收敛。跟式$\eqref{neq:case-1}$相比，式$\eqref{neq:case-2}$的常数更大，这意味着$\eta_t\equiv\frac{D}{G\sqrt{T}}$的收敛速度很可能比$\eta_t = \frac{\alpha}{\sqrt{t}}$快。然而，在实际中我们更愿意用后者，因为前者的需要提前确定训练总步数$T$，训练完就结束了并且精度也固定了，后者并没有这些限制，甚至$\alpha$也不需要调，直接$\eta_t = \frac{1}{\sqrt{t}}$就可以持续训练下去，并且理论上平均损失与理论最小值的差距会越来越小。

可即便如此，$\eta_t = \frac{1}{\sqrt{t}}$这种学习率策略，不管在量级或者变化规律上都依然与我们平时训练所用的相距甚远，因此不难猜测里边必然加了不少很强的假设。事不宜迟，我们马上来展开证明过程，逐一展开其中的假设。

## 证明过程 #

证明的开始，我们假设对于任意$\boldsymbol{x}$，$L(\boldsymbol{x},\boldsymbol{\theta})$都是关于$\boldsymbol{\theta}$的凸函数。这是一个非常强且通常非常不符合训练事实的假设，但没办法，理论分析通常都只能做一些很强的假设，然后将这些假设之下的结论启发性地用到实际场景。

凸函数有很多不同的定义方式，这里直接采用如下定义：  
\begin{equation}L(\boldsymbol{x}_t,\boldsymbol{\theta}_2) - L(\boldsymbol{x}_t,\boldsymbol{\theta}_1) \geq (\boldsymbol{\theta}_2-\boldsymbol{\theta}_1)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_1),\quad \forall \boldsymbol{\theta}_1,\boldsymbol{\theta}_2\label{eq:convex}\end{equation}  
其中$\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_1)\triangleq \nabla_{\boldsymbol{\theta}}L(\boldsymbol{x}_t,\boldsymbol{\theta})$是$L(\boldsymbol{x}_t,\boldsymbol{\theta})$关于$\boldsymbol{\theta}$的梯度，$\cdot$是向量内积，上述定义的几何意义就是凸函数的图像总在其切线（面）的上方。

证明的要点，是考虑$\boldsymbol{\theta}_{t+1}$与$\boldsymbol{\theta}^*$的距离：  
\begin{equation}\begin{aligned}  
\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2=&\, \Vert\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)- \boldsymbol{\theta}^*\Vert^2 \\\  
=&\, \Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2 - 2\eta_t (\boldsymbol{\theta}_t- \boldsymbol{\theta}^*)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) + \eta_t^2\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2  
\end{aligned}\end{equation}  
将它改写成  
\begin{equation}(\boldsymbol{\theta}_t- \boldsymbol{\theta}^*)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) = \frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} + \frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\end{equation}  
根据式$\eqref{eq:convex}$，我们有$L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq (\boldsymbol{\theta}_t- \boldsymbol{\theta}^*)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)$，代入上式有  
\begin{equation}L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} + \frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\end{equation}  
两端对$t=1,2,\cdots,T$求和：  
\begin{equation}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \sum_{t=1}^TL(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \sum_{t=1}^T\left(\frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t}\right) + \sum_{t=1}^T\frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\label{neq:base}\end{equation}  
再次引入一个新的假设——$\eta_t$是关于$t$的单调递减函数（即$\eta_t\geq \eta_{t+1}$），并且记$D = \max_t \Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert$，那么就有  
\begin{equation}\begin{aligned}  
&\,\sum_{t=1}^T\left(\frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t}\right)\\\  
=&\,\frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\eta_1} - \frac{\Vert\boldsymbol{\theta}_{T+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_T} + \sum_{t=2}^T\left(\frac{1}{2\eta_t} - \frac{1}{2\eta_{t-1}}\right)\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2\\\  
\leq&\,\frac{D^2}{2\eta_1} + \sum_{t=2}^T\left(\frac{1}{2\eta_t} - \frac{1}{2\eta_{t-1}}\right)D^2\\\  
=&\, \frac{D^2}{2\eta_T}  
\end{aligned}\end{equation}  
最后我们记$G = \max_t \Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert$，然后将上式代入式$\eqref{neq:base}$得到  
\begin{equation}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{D^2}{2\eta_T} + \sum_{t=1}^T\frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\leq \frac{D^2}{2\eta_T} + \frac{G^2}{2}\sum_{t=1}^T\eta_t\end{equation}  
最后两端除以$T$即得不等式$\eqref{neq:core}$。

注意现在的常数$D,G$是优化相关的，即先要确定学习率策略$\\{\eta_t\\}$，然后完成优化过程才能得到$D,G$。要想$D,G$成为优化无关的常数，我们需要假设对于任意的$\\{\eta_t\\}$，$\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert$和$\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert$都分别不超过某个常数$D,G$，这样才能使得不等式$\eqref{neq:core}$右端只与学习率策略有关。

## 域内投影 #

然而，最后的假设“对于任意的$\\{\eta_t\\}$，$\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert$和$\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert$都分别不超过某个常数$D,G$”总有种“本末倒置”的感觉：看上去需要先完成优化才能确定$D,G$，而我们的目的却是要通过所证明的结果来改进优化。为了去掉这种奇怪的感觉，我们干脆将这两个假设改为：

> 1、式$\eqref{eq:argmin}$中的$\boldsymbol{\theta}\in\mathbb{R}^d$改为$\boldsymbol{\theta}\in\boldsymbol{\Theta}$，其中$\boldsymbol{\Theta}\subseteq \mathbb{R}^d$是一个有界凸集；
> 
> 1.1）**有界** ：$D=\max\limits_{\boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}}\Vert\boldsymbol{\theta}_1-\boldsymbol{\theta}_2\Vert < \infty$；
> 
> 1.2）**凸集** ：$\forall \boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}$以及$\forall\lambda\in[0,1]$，都有$\lambda \boldsymbol{\theta}_1 + (1-\lambda)\boldsymbol{\theta}_2 \in \boldsymbol{\Theta}$。
> 
> 2、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，都有$\Vert\boldsymbol{g}(\boldsymbol{x},\boldsymbol{\theta})\Vert\leq G$

第2点可能更容易接受，无非是给损失函数$L(\boldsymbol{x},\boldsymbol{\theta})$再加了个假设而已，就好比“债多不愁”，凸函数这么强的假设都加了，再多加点也无妨。但第1点假设似乎不那么容易理解：凸集是因为凸函数本身只能定义在凸集上，这个也能接受，但有界如何保证呢？即如何保证迭代$\eqref{eq:sgd}$的输出一定有界？

答案是“多加一步投影”。我们定义投影运算：  
\begin{equation}\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}) = \mathop{\text{argmin}}_{\boldsymbol{\theta}\in\boldsymbol{\Theta}}\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert\end{equation}  
即在$\boldsymbol{\Theta}$中找到与$\boldsymbol{\varphi}$最相近的向量，于是我们可以将式$\eqref{eq:sgd}$改为  
\begin{equation}\boldsymbol{\theta}_{t+1} = \Pi_{\boldsymbol{\Theta}}\big(\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\big)\in \boldsymbol{\Theta}\label{eq:sgd-p}\end{equation}  
这就能保证迭代结果一定在集合$\boldsymbol{\Theta}$中。

然而，这样修改之后，上一节的证明和前面的结论（主要是不等式$\eqref{neq:core}$）还成立吗？很幸运，还成立，我们只需要证明对于式$\eqref{eq:sgd-p}$所定义的投影SGD有  
\begin{equation}\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert \leq \Vert\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \boldsymbol{\theta}^*\Vert\end{equation}  
那么“证明过程”一节的推导依然可以进行下去，只不过部分等号变成$\leq$而已。为此，我们只需要证明$\forall \boldsymbol{\varphi}\in\mathbb{R}^d, \boldsymbol{\theta}\in \boldsymbol{\Theta}$，都有  
\begin{equation}\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}) - \boldsymbol{\theta}\Vert \leq \Vert \boldsymbol{\varphi} - \boldsymbol{\theta}\Vert\end{equation}

> **证明** ：证明的关键是将凸集和$\Pi_{\boldsymbol{\Theta}}$的定义结合起来。首先，根据凸集的定义，我们知道$\forall \lambda\in(0,1)$都有$\lambda\boldsymbol{\theta} + (1-\lambda)\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\in \boldsymbol{\Theta}$，于是根据$\Pi_{\boldsymbol{\Theta}}$的定义，恒成立  
>  \begin{equation}\Vert\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert\leq \Vert\boldsymbol{\varphi} - \lambda\boldsymbol{\theta} - (1-\lambda)\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert = \Vert(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})) + \lambda(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\Vert\end{equation}  
>  两端平方然后相减，得到  
>  \begin{equation}\lambda^2\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + 2\lambda(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\cdot(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}))\geq 0\end{equation}  
>  注意我们刚限制了$\lambda\in(0,1)$，所以两端可以除以$\lambda$：  
>  \begin{equation}\lambda\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + 2(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\cdot(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}))\geq 0\end{equation}  
>  这是个恒成立的式子，那么对于$\lambda\to 0^+$依然是恒成立的，于是  
>  \begin{equation}(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\cdot(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}))\geq 0\end{equation}  
>  两端加上$\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + \Vert\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert^2$，左端正好是$\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert^2$，所以有  
>  \begin{equation}\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert^2\geq \Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + \Vert\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert^2 \geq \Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2\end{equation}

## 假设分析 #

至此，我们已经完整了证明过程。以上结果出自[《Online Convex Programming and Generalized Infinitesimal Gradient Ascent》](https://www.cs.cmu.edu/~maz/publications/techconvex.pdf)，是2003年的论文。特别地，优化相关的文献非常多，笔者作为一个初学者，在后面的文献溯源方面大概率会有错漏，请了解的读者不吝指正。

现在我们可以来“盘点”一下完整的证明过程用到的所有假设：

> 1、$\boldsymbol{\Theta}$是一个有界凸集，$D=\max\limits_{\boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}}\Vert\boldsymbol{\theta}_1-\boldsymbol{\theta}_2\Vert < \infty$；
> 
> 2、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，$L(\boldsymbol{x},\boldsymbol{\theta})$都是关于$\boldsymbol{\theta}$的凸函数；
> 
> 3、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，都有$\Vert\nabla_{\boldsymbol{\theta}}L(\boldsymbol{x},\boldsymbol{\theta})\Vert\leq G < \infty$；
> 
> 4、学习率$\eta_t$是关于$t$的单调递减函数（即$\eta_t\geq \eta_{t+1}$）；

在这些假设之下，投影SGD即式$\eqref{eq:sgd-p}$成立不等式$\eqref{neq:core}$。

其中，第1、4点假设都无可厚非，甚至可以说非常合理，比如第1点，对于实际计算来说，一个充分大的球体跟$\mathbb{R}^d$并没有实质区别了，而第4点递减的学习率更是符合已有认知；最强且最不符合事实的是第2点凸函数假设，但没办法，多看几篇优化相关的文献就释然了，因为几乎所有优化理论都是基于凸函数假设进行的，我们只能寄望于优化进入一定区域后损失函数能部分符合凸函数的性质；第3点本质上也是很强的假设，但实际运算中如果初始化做得好，并且学习率也设置得适当，基本上能将梯度模长控制在一定范围内，因此也通常能接受。

## 文章小结 #

在这篇文章中，我们重温了一篇凸优化的旧论文，介绍了SGD的一个非常基础的收敛性证明：在适当（实际上非常强）的假设下，SGD的收敛性可以得到保证。尽管这些假设在实际应用中可能并不总是成立，例如凸函数假设和梯度模长的限制，但这些理论结果仍能为我们提供了关于SGD收敛性的重要见解。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9902>_

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

苏剑林. (Dec. 19, 2023). 《让炼丹更科学一些（一）：SGD的平均损失收敛 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9902>

@online{kexuefm-9902,  
title={让炼丹更科学一些（一）：SGD的平均损失收敛},  
author={苏剑林},  
year={2023},  
month={Dec},  
url={\url{https://spaces.ac.cn/archives/9902}},  
} 


---

## 公式推导与注释

### 1. 基础SGD收敛性理论

让我们从最基础的SGD算法开始,系统地推导其收敛性质。

#### 1.1 凸函数情况下的收敛性

对于凸函数$L(\boldsymbol{\theta})$,我们有凸性定义:
\begin{equation}L(\boldsymbol{\theta}_2) \geq L(\boldsymbol{\theta}_1) + \nabla L(\boldsymbol{\theta}_1)^{\top}(\boldsymbol{\theta}_2 - \boldsymbol{\theta}_1)\tag{1}\end{equation}

**数学直觉**:这个不等式说明,凸函数总是位于其任意点的切平面上方。这是凸优化的基石。

取$\boldsymbol{\theta}_1 = \boldsymbol{\theta}_t$,$\boldsymbol{\theta}_2 = \boldsymbol{\theta}^*$,我们得到:
\begin{equation}L(\boldsymbol{\theta}^*) \geq L(\boldsymbol{\theta}_t) + \nabla L(\boldsymbol{\theta}_t)^{\top}(\boldsymbol{\theta}^* - \boldsymbol{\theta}_t)\tag{2}\end{equation}

改写为:
\begin{equation}L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*) \leq \nabla L(\boldsymbol{\theta}_t)^{\top}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)\tag{3}\end{equation}

这个不等式给出了当前损失与最优损失之间的差距上界。

#### 1.2 距离递推关系

考虑SGD更新:$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t$,其中$\boldsymbol{g}_t$是梯度的无偏估计。

计算$\boldsymbol{\theta}_{t+1}$到最优点$\boldsymbol{\theta}^*$的距离平方:
\begin{equation}\begin{aligned}
\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2 &= \|\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t - \boldsymbol{\theta}^*\|^2\\
&= \|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 - 2\eta_t \boldsymbol{g}_t^{\top}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + \eta_t^2\|\boldsymbol{g}_t\|^2
\end{aligned}\tag{4}\end{equation}

**数学直觉**:这个展开式包含三项:
- 第一项:当前距离的平方
- 第二项:梯度方向对距离减少的贡献(负项,使距离减小)
- 第三项:由于步长引入的噪声(正项,使距离可能增大)

#### 1.3 期望分析

对式(4)两边取期望:
\begin{equation}\mathbb{E}[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2] = \mathbb{E}[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2] - 2\eta_t \mathbb{E}[\boldsymbol{g}_t^{\top}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)] + \eta_t^2\mathbb{E}[\|\boldsymbol{g}_t\|^2]\tag{5}\end{equation}

由于$\mathbb{E}[\boldsymbol{g}_t|\boldsymbol{\theta}_t] = \nabla L(\boldsymbol{\theta}_t)$,我们有:
\begin{equation}\mathbb{E}[\boldsymbol{g}_t^{\top}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)] = \mathbb{E}[\nabla L(\boldsymbol{\theta}_t)^{\top}(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)] \geq \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]\tag{6}\end{equation}

其中不等号由凸性(式3)得到。

#### 1.4 梯度方差界

定义梯度的方差:
\begin{equation}\sigma_t^2 = \mathbb{E}[\|\boldsymbol{g}_t - \nabla L(\boldsymbol{\theta}_t)\|^2]\tag{7}\end{equation}

利用$\mathbb{E}[\|\boldsymbol{g}_t\|^2] = \|\nabla L(\boldsymbol{\theta}_t)\|^2 + \sigma_t^2$,我们有:
\begin{equation}\mathbb{E}[\|\boldsymbol{g}_t\|^2] \leq G^2 + \sigma_t^2 \leq G^2 + \sigma^2\tag{8}\end{equation}

其中$G = \sup_{\boldsymbol{\theta}}\|\nabla L(\boldsymbol{\theta})\|$是梯度的上界,$\sigma^2 = \sup_t \sigma_t^2$是方差的上界。

#### 1.5 主要不等式推导

结合式(5)、(6)、(8),我们得到:
\begin{equation}\mathbb{E}[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2] \leq \mathbb{E}[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2] - 2\eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] + \eta_t^2(G^2 + \sigma^2)\tag{9}\end{equation}

改写为:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{\mathbb{E}[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2] - \mathbb{E}[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2]}{2\eta_t} + \frac{\eta_t(G^2 + \sigma^2)}{2}\tag{10}\end{equation}

### 2. 不同凸性条件下的收敛率

#### 2.1 一般凸函数

对式(10)在$t=1,\ldots,T$上求和:
\begin{equation}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{\|\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\|^2}{2\eta_1} + \frac{G^2 + \sigma^2}{2}\sum_{t=1}^T \eta_t\tag{11}\end{equation}

**定理1**(一般凸函数收敛): 若$\eta_t = \frac{D}{G\sqrt{T}}$,其中$D = \|\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\|$,则:
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{DG}{\sqrt{T}} + \frac{D\sigma}{2G\sqrt{T}} = \mathcal{O}\left(\frac{1}{\sqrt{T}}\right)\tag{12}\end{equation}

**数学直觉**: $\mathcal{O}(1/\sqrt{T})$收敛率意味着要达到$\epsilon$精度,需要$\mathcal{O}(1/\epsilon^2)$步迭代。这在凸优化中是次线性收敛。

#### 2.2 强凸函数

若$L(\boldsymbol{\theta})$是$\mu$-强凸的,即:
\begin{equation}L(\boldsymbol{\theta}_2) \geq L(\boldsymbol{\theta}_1) + \nabla L(\boldsymbol{\theta}_1)^{\top}(\boldsymbol{\theta}_2 - \boldsymbol{\theta}_1) + \frac{\mu}{2}\|\boldsymbol{\theta}_2 - \boldsymbol{\theta}_1\|^2\tag{13}\end{equation}

取$\boldsymbol{\theta}_1 = \boldsymbol{\theta}^*$(注意$\nabla L(\boldsymbol{\theta}^*) = 0$):
\begin{equation}L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*) \geq \frac{\mu}{2}\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2\tag{14}\end{equation}

结合式(9),选择固定学习率$\eta_t = \eta = \frac{1}{\mu}$:
\begin{equation}\begin{aligned}
\mathbb{E}[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2] &\leq \mathbb{E}[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2] - \frac{2}{\mu}\mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] + \frac{G^2 + \sigma^2}{\mu^2}\\
&\leq \mathbb{E}[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2] - \|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2 + \frac{G^2 + \sigma^2}{\mu^2}\\
&= \frac{G^2 + \sigma^2}{\mu^2}
\end{aligned}\tag{15}\end{equation}

**定理2**(强凸函数收敛): 若$\eta = \frac{1}{\mu}$,则:
\begin{equation}\mathbb{E}[\|\boldsymbol{\theta}_T - \boldsymbol{\theta}^*\|^2] \leq \frac{G^2 + \sigma^2}{\mu^2}\tag{16}\end{equation}

更精细的分析(使用递减学习率$\eta_t = \frac{2}{\mu(t+1)}$)可以得到:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{2(G^2 + \sigma^2)}{\mu T} = \mathcal{O}\left(\frac{1}{T}\right)\tag{17}\end{equation}

**数学直觉**: $\mathcal{O}(1/T)$收敛率比一般凸函数快,达到$\epsilon$精度只需$\mathcal{O}(1/\epsilon)$步。

#### 2.3 光滑凸函数

若$L(\boldsymbol{\theta})$是$L$-光滑的,即梯度满足Lipschitz条件:
\begin{equation}\|\nabla L(\boldsymbol{\theta}_1) - \nabla L(\boldsymbol{\theta}_2)\| \leq L\|\boldsymbol{\theta}_1 - \boldsymbol{\theta}_2\|\tag{18}\end{equation}

等价地,有二次上界:
\begin{equation}L(\boldsymbol{\theta}_2) \leq L(\boldsymbol{\theta}_1) + \nabla L(\boldsymbol{\theta}_1)^{\top}(\boldsymbol{\theta}_2 - \boldsymbol{\theta}_1) + \frac{L}{2}\|\boldsymbol{\theta}_2 - \boldsymbol{\theta}_1\|^2\tag{19}\end{equation}

对于光滑凸函数,使用固定学习率$\eta = \frac{1}{L}$:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_{t+1})] \leq \mathbb{E}[L(\boldsymbol{\theta}_t)] - \frac{1}{2L}\|\nabla L(\boldsymbol{\theta}_t)\|^2 + \frac{\sigma^2}{2L}\tag{20}\end{equation}

**定理3**(光滑凸函数收敛): 若$\eta = \frac{1}{L}$,则:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{LD^2}{2T} + \frac{\sigma^2}{2L} = \mathcal{O}\left(\frac{1}{T}\right) + \mathcal{O}(\sigma^2)\tag{21}\end{equation}

**数学直觉**: 光滑性允许我们使用固定学习率,并获得线性收敛率。但随机梯度的方差$\sigma^2$会产生一个恒定的误差下界。

#### 2.4 强凸且光滑函数

若$L(\boldsymbol{\theta})$同时是$\mu$-强凸和$L$-光滑的,定义条件数$\kappa = L/\mu$。

使用固定学习率$\eta = \frac{2}{L + \mu}$:
\begin{equation}\mathbb{E}[\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\|^2] \leq \left(1 - \frac{2\mu}{L+\mu}\right)\mathbb{E}[\|\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\|^2] + \frac{2\sigma^2}{(L+\mu)^2}\tag{22}\end{equation}

递推得到:
\begin{equation}\mathbb{E}[\|\boldsymbol{\theta}_T - \boldsymbol{\theta}^*\|^2] \leq \left(1 - \frac{2}{\kappa+1}\right)^T D^2 + \frac{2\kappa\sigma^2}{L^2(\kappa+1)}\tag{23}\end{equation}

**定理4**(强凸光滑函数收敛): 达到$\epsilon$精度需要:
\begin{equation}T = \mathcal{O}\left(\kappa\log\frac{1}{\epsilon}\right)\tag{24}\end{equation}

**数学直觉**: 这是线性收敛率,远快于次线性收敛。条件数$\kappa$越小,收敛越快。

### 3. 非凸函数情况

对于非凸函数,我们无法保证收敛到全局最优,但可以分析收敛到驻点的性质。

#### 3.1 光滑非凸函数

假设$L(\boldsymbol{\theta})$是$L$-光滑的(满足式19),但不一定是凸的。

从式(19)出发,取$\boldsymbol{\theta}_2 = \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t$:
\begin{equation}\begin{aligned}
L(\boldsymbol{\theta}_{t+1}) &\leq L(\boldsymbol{\theta}_t) - \eta_t \nabla L(\boldsymbol{\theta}_t)^{\top}\boldsymbol{g}_t + \frac{L\eta_t^2}{2}\|\boldsymbol{g}_t\|^2
\end{aligned}\tag{25}\end{equation}

取期望:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_{t+1})] \leq \mathbb{E}[L(\boldsymbol{\theta}_t)] - \eta_t\mathbb{E}[\|\nabla L(\boldsymbol{\theta}_t)\|^2] + \frac{L\eta_t^2}{2}(G^2 + \sigma^2)\tag{26}\end{equation}

对$t=1,\ldots,T$求和:
\begin{equation}\sum_{t=1}^T \eta_t\mathbb{E}[\|\nabla L(\boldsymbol{\theta}_t)\|^2] \leq L(\boldsymbol{\theta}_1) - \mathbb{E}[L(\boldsymbol{\theta}_T)] + \frac{L(G^2 + \sigma^2)}{2}\sum_{t=1}^T \eta_t^2\tag{27}\end{equation}

假设$L(\boldsymbol{\theta})$有下界$L^* = \inf_{\boldsymbol{\theta}} L(\boldsymbol{\theta})$,则:
\begin{equation}\sum_{t=1}^T \eta_t\mathbb{E}[\|\nabla L(\boldsymbol{\theta}_t)\|^2] \leq \Delta + \frac{L(G^2 + \sigma^2)}{2}\sum_{t=1}^T \eta_t^2\tag{28}\end{equation}

其中$\Delta = L(\boldsymbol{\theta}_1) - L^*$。

**定理5**(非凸函数一阶驻点): 若$\eta_t = \eta = \frac{1}{L}$,则:
\begin{equation}\min_{1\leq t\leq T}\mathbb{E}[\|\nabla L(\boldsymbol{\theta}_t)\|^2] \leq \frac{\sum_{t=1}^T \eta_t\mathbb{E}[\|\nabla L(\boldsymbol{\theta}_t)\|^2]}{\sum_{t=1}^T \eta_t} \leq \frac{L\Delta}{T} + \frac{G^2 + \sigma^2}{2} = \mathcal{O}\left(\frac{1}{T}\right)\tag{29}\end{equation}

**数学直觉**: 对于非凸函数,我们保证能找到梯度范数小的点(一阶驻点),但不保证是局部或全局最优。

#### 3.2 Polyak-Łojasiewicz(PL)条件

PL条件是一个比强凸更弱但比一般非凸更强的条件:
\begin{equation}\frac{1}{2}\|\nabla L(\boldsymbol{\theta})\|^2 \geq \mu(L(\boldsymbol{\theta}) - L^*)\tag{30}\end{equation}

满足PL条件的函数可能是非凸的,但仍能保证全局收敛。

**定理6**(PL条件下的收敛): 若$L(\boldsymbol{\theta})$满足PL条件且$L$-光滑,使用$\eta = \frac{1}{L}$:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L^*] \leq \left(1 - \frac{\mu}{L}\right)^T(L(\boldsymbol{\theta}_1) - L^*) + \frac{\sigma^2}{2\mu}\tag{31}\end{equation}

**数学直觉**: PL条件保证线性收敛,即使函数非凸。许多神经网络在过参数化时满足PL条件。

### 4. 方差减少技术

随机梯度的方差$\sigma^2$是影响SGD收敛的关键因素。方差减少技术旨在降低这个方差。

#### 4.1 SVRG算法

**随机方差减少梯度**(SVRG)的核心思想是周期性计算全梯度作为参考点。

**算法框架**:
- 外循环:每$m$步计算一次全梯度$\tilde{\boldsymbol{g}} = \nabla L(\tilde{\boldsymbol{\theta}})$
- 内循环:使用方差减少的梯度估计
\begin{equation}\boldsymbol{v}_t = \nabla L(\boldsymbol{x}_t, \boldsymbol{\theta}_t) - \nabla L(\boldsymbol{x}_t, \tilde{\boldsymbol{\theta}}) + \tilde{\boldsymbol{g}}\tag{32}\end{equation}

**方差分析**:
\begin{equation}\mathbb{E}[\boldsymbol{v}_t] = \nabla L(\boldsymbol{\theta}_t)\tag{33}\end{equation}
\begin{equation}\mathbb{E}[\|\boldsymbol{v}_t - \nabla L(\boldsymbol{\theta}_t)\|^2] \leq 2L^2\|\boldsymbol{\theta}_t - \tilde{\boldsymbol{\theta}}\|^2\tag{34}\end{equation}

**数学直觉**: $\boldsymbol{v}_t$是无偏的,且方差随着$\boldsymbol{\theta}_t$接近$\tilde{\boldsymbol{\theta}}$而减小。

**定理7**(SVRG收敛率): 对于$\mu$-强凸$L$-光滑函数:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L^*] \leq \left(1 - \min\left\{\frac{\mu}{3L}, \frac{1}{3m}\right\}\right)^T(L(\boldsymbol{\theta}_0) - L^*)\tag{35}\end{equation}

**数学直觉**: SVRG达到线性收敛率,且收敛速度不依赖于数据集大小$n$,只依赖于条件数$\kappa$。

#### 4.2 SAGA算法

SAGA维护所有样本梯度的表格,避免了SVRG的内外循环结构。

**更新规则**:
\begin{equation}\boldsymbol{v}_t = \nabla L(\boldsymbol{x}_t, \boldsymbol{\theta}_t) - \boldsymbol{g}_t^{old}(\boldsymbol{x}_t) + \frac{1}{n}\sum_{i=1}^n \boldsymbol{g}_t^{old}(\boldsymbol{x}_i)\tag{36}\end{equation}

其中$\boldsymbol{g}_t^{old}(\boldsymbol{x}_i)$是存储的第$i$个样本的旧梯度。

**定理8**(SAGA收敛率): 对于$\mu$-强凸$L$-光滑函数:
\begin{equation}\mathbb{E}[\|\boldsymbol{\theta}_T - \boldsymbol{\theta}^*\|^2] \leq \left(1 - \min\left\{\frac{1}{3\kappa}, \frac{1}{3n}\right\}\right)^T\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2\tag{37}\end{equation}

#### 4.3 方差减少的几何解释

考虑梯度空间中的分解:
\begin{equation}\nabla L(\boldsymbol{x}_t, \boldsymbol{\theta}_t) = \nabla L(\boldsymbol{\theta}_t) + [\nabla L(\boldsymbol{x}_t, \boldsymbol{\theta}_t) - \nabla L(\boldsymbol{\theta}_t)]\tag{38}\end{equation}

第二项是噪声。SVRG/SAGA通过减去相关噪声$\nabla L(\boldsymbol{x}_t, \tilde{\boldsymbol{\theta}})$来减少方差:
\begin{equation}\nabla L(\boldsymbol{x}_t, \boldsymbol{\theta}_t) - \nabla L(\boldsymbol{x}_t, \tilde{\boldsymbol{\theta}}) \approx \nabla^2 L(\tilde{\boldsymbol{\theta}})(\boldsymbol{\theta}_t - \tilde{\boldsymbol{\theta}})\tag{39}\end{equation}

这可以看作是对噪声的线性近似和补偿。

### 5. 学习率调度策略

学习率的选择对SGD收敛至关重要。

#### 5.1 常数学习率

**优点**: 简单,不需要调参
**缺点**: 存在收敛误差下界

对于强凸函数,固定$\eta = \frac{1}{L}$:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L^*] \leq \frac{\sigma^2}{2\mu}\tag{40}\end{equation}

无法收敛到精确最优,误差与方差成正比。

#### 5.2 多项式衰减

常用的衰减策略:$\eta_t = \frac{\eta_0}{(1 + t)^{\alpha}}$,其中$\alpha \in (0.5, 1]$。

**Robbins-Monro条件**: 为保证收敛,学习率需满足:
\begin{equation}\sum_{t=1}^{\infty} \eta_t = \infty, \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty\tag{41}\end{equation}

**数学直觉**: 第一个条件保证能够到达任意远的点,第二个条件保证累积噪声有界。

对于$\eta_t = \frac{\eta_0}{\sqrt{t}}$:
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L^*] = \mathcal{O}\left(\frac{\log T}{\sqrt{T}}\right)\tag{42}\end{equation}

#### 5.3 指数衰减

指数衰减:$\eta_t = \eta_0 \gamma^t$,其中$\gamma \in (0, 1)$。

**优点**: 快速降低学习率,适合强凸函数
**缺点**: 不满足Robbins-Monro条件,可能过早停止

实践中常用分段指数衰减:
\begin{equation}\eta_t = \eta_0 \gamma^{\lfloor t/T_0 \rfloor}\tag{43}\end{equation}

每$T_0$步衰减一次。

#### 5.4 Cosine Annealing

余弦退火:
\begin{equation}\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)\tag{44}\end{equation}

**优点**: 平滑衰减,在训练后期有小幅增加,有助于跳出局部最优
**数学直觉**: 模拟物理退火过程,逐渐降温但偶尔升温以探索。

#### 5.5 Warm Restart

循环余弦退火(SGDR):
\begin{equation}\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)\tag{45}\end{equation}

其中$T_{cur}$是当前周期内的步数,$T_i$是第$i$个周期的长度。

周期长度通常递增:$T_{i+1} = T_i \times T_{mult}$(如$T_{mult} = 2$)。

**数学直觉**: 周期性重启允许模型多次探索参数空间,有助于找到更好的局部最优。

### 6. Mini-Batch大小的影响

#### 6.1 方差与批量大小的关系

对于批量大小$B$,梯度估计的方差为:
\begin{equation}\text{Var}[\boldsymbol{g}_B] = \frac{\sigma^2}{B}\tag{46}\end{equation}

**数学直觉**: 增大批量可以线性地减少方差,但计算成本也线性增加。

#### 6.2 线性缩放规则

当批量大小从$B_1$增加到$B_2 = kB_1$时,为保持相同的收敛行为,学习率应相应增加:
\begin{equation}\eta_2 = k\eta_1\tag{47}\end{equation}

**理论基础**: 从式(12)可以看出,增大$B$减少了方差项的影响,允许使用更大的学习率。

**限制**: 线性缩放规则在$B$较大时会失效,因为会超过最优学习率的上界。

#### 6.3 临界批量大小

定义临界批量大小$B_c$,超过这个值后增大批量不再显著提升收敛速度:
\begin{equation}B_c \sim \frac{\sigma^2}{\|\nabla L(\boldsymbol{\theta})\|^2}\tag{48}\end{equation}

**数学直觉**: 当批量足够大时,梯度噪声已经很小,进一步增大批量的边际收益递减。

**实践建议**:
- 训练初期:梯度大,$B_c$大,可以使用大批量
- 训练后期:梯度小,$B_c$小,应该减小批量或学习率

### 7. 数值示例

#### 7.1 二次函数优化

考虑简单的二次函数:
\begin{equation}L(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^{\top}Q\boldsymbol{\theta} - \boldsymbol{b}^{\top}\boldsymbol{\theta}\tag{49}\end{equation}

其中$Q$是正定矩阵。最优解为$\boldsymbol{\theta}^* = Q^{-1}\boldsymbol{b}$。

**强凸性**: $\mu = \lambda_{min}(Q)$
**光滑性**: $L = \lambda_{max}(Q)$
**条件数**: $\kappa = \frac{\lambda_{max}(Q)}{\lambda_{min}(Q)}$

对于$Q = \text{diag}(1, 10)$,$\boldsymbol{b} = [1, 1]^{\top}$:
- $\mu = 1, L = 10, \kappa = 10$
- 理论最优学习率: $\eta^* = \frac{2}{L + \mu} = \frac{2}{11} \approx 0.18$

**数值验证**:
- $\eta = 0.1$: 收敛需要约50步
- $\eta = 0.18$: 收敛需要约30步(最优)
- $\eta = 0.3$: 振荡,收敛需要约100步
- $\eta = 0.5$: 发散

#### 7.2 Logistic回归

考虑二分类问题,损失函数:
\begin{equation}L(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \log(1 + \exp(-y_i\boldsymbol{x}_i^{\top}\boldsymbol{\theta}))\tag{50}\end{equation}

**性质**:
- 凸函数(Hessian半正定)
- 光滑性: $L = \frac{1}{4}\max_i \|\boldsymbol{x}_i\|^2$

**实验设置**: $n=1000$,$d=20$,生成随机数据。

**学习率比较**:
| 学习率策略 | 收敛步数 | 最终误差 |
|----------|---------|----------|
| $\eta = 0.01$(常数) | 1000+ | $10^{-3}$ |
| $\eta_t = 0.1/\sqrt{t}$ | ~500 | $10^{-4}$ |
| $\eta_t = 1/t$ | ~300 | $10^{-5}$ |
| $\eta_t = 0.1/(1+0.01t)$ | ~400 | $10^{-4}$ |

**数学直觉**: 衰减学习率显著改善收敛精度,但收敛速度可能较慢。

#### 7.3 深度神经网络

对于深度网络,损失函数高度非凸。考虑两层神经网络:
\begin{equation}f(\boldsymbol{x};\boldsymbol{\theta}) = \boldsymbol{W}_2\sigma(\boldsymbol{W}_1\boldsymbol{x})\tag{51}\end{equation}

**实验观察**:
1. **训练初期**: 梯度大且不稳定,需要小学习率或warmup
2. **训练中期**: 梯度稳定,可以使用较大学习率
3. **训练后期**: 梯度小,需要更小学习率来精细调整

**Warmup策略**:
\begin{equation}\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}}, & t \leq T_{warmup}\\
\eta_{max} \cdot \text{schedule}(t - T_{warmup}), & t > T_{warmup}
\end{cases}\tag{52}\end{equation}

典型设置:$T_{warmup} = 0.1T$(训练总步数的10%)。

### 8. 实践建议与总结

#### 8.1 学习率选择指南

**步骤1**: 学习率范围测试
- 从极小值(如$10^{-6}$)开始,指数增长
- 记录每个学习率下的损失变化
- 选择损失下降最快且稳定的学习率

**步骤2**: 精细调整
- 在选定范围内进行网格搜索
- 考虑使用学习率调度器

**步骤3**: 监控训练
- 观察损失曲线:过大的学习率导致振荡,过小的学习率导致收敛慢
- 观察梯度范数:梯度爆炸或消失都提示需要调整

**经验法则**:
\begin{equation}\eta \in \left[\frac{1}{10L}, \frac{2}{L}\right]\tag{53}\end{equation}

其中$L$可以通过Hessian的最大特征值或Lipschitz常数估计。

#### 8.2 批量大小选择

**小批量**(B = 32-128):
- 优点:更好的泛化性能,训练更稳定
- 缺点:训练慢,GPU利用率低

**大批量**(B = 512-4096):
- 优点:训练快,GPU利用率高
- 缺点:泛化性能可能下降,需要更大学习率

**折中方案**: 渐进增大批量
\begin{equation}B_t = \min(B_{max}, B_0 \cdot 2^{\lfloor t/T_0 \rfloor})\tag{54}\end{equation}

#### 8.3 收敛性诊断

**判断是否收敛**:
1. 损失不再显著下降(相对变化 < 0.1%)
2. 梯度范数足够小($\|\nabla L(\boldsymbol{\theta})\| < \epsilon$)
3. 参数变化很小($\|\boldsymbol{\theta}_t - \boldsymbol{\theta}_{t-1}\| < \delta$)

**未收敛的可能原因**:
- 学习率过大:降低学习率或增加批量大小
- 学习率过小:增大学习率或使用自适应优化器
- 梯度消失:检查网络初始化,使用BatchNorm或ResNet
- 鞍点:增加随机性(减小批量)或使用动量

#### 8.4 理论与实践的差距

**理论假设 vs 实践现实**:
1. **凸性假设**: 深度网络高度非凸,但局部凸性可能存在
2. **有界梯度**: 实践中使用梯度裁剪来强制满足
3. **独立采样**: 数据增强可能引入相关性
4. **光滑性**: 激活函数(如ReLU)不处处光滑

**实践中的经验方法**:
- **Adam/AdamW**: 自适应学习率,对超参数不敏感
- **SGD+Momentum**: 需要仔细调参,但泛化性能通常更好
- **学习率Warmup**: 缓解训练初期的不稳定
- **梯度裁剪**: 防止梯度爆炸

#### 8.5 总结

本文系统推导了SGD在不同条件下的收敛性理论:

**主要结论**:
1. 一般凸函数: $\mathcal{O}(1/\sqrt{T})$收敛率
2. 强凸函数: $\mathcal{O}(1/T)$收敛率
3. 非凸函数: 收敛到一阶驻点
4. 方差减少技术(SVRG/SAGA)可以达到线性收敛率

**关键洞察**:
- 学习率衰减是处理随机梯度方差的关键
- 批量大小影响方差,需要与学习率协调
- 理论分析提供指导,但实践需要灵活调整

**未来方向**:
- 非凸优化的更精细理论
- 自适应优化器的收敛性分析
- 分布式SGD的理论保证

