---
title: 生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配
slug: 生成扩散模型漫谈十六w距离-得分匹配
date: 2023-02-14
tags: 详细推导, 微分方程, GAN, 生成模型, 扩散, 生成模型
status: pending
---
# 生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配

**原文链接**: [https://spaces.ac.cn/archives/9467](https://spaces.ac.cn/archives/9467)

**发布日期**: 

---

Wasserstein距离（下面简称“W距离”），是基于最优传输思想来度量两个概率分布差异程度的距离函数，笔者之前在[《从Wasserstein距离、对偶理论到WGAN》](/archives/6280)等博文中也做过介绍。对于很多读者来说，第一次听说W距离，是因为2017年出世的[WGAN](https://papers.cool/arxiv/1701.07875)，它开创了从最优传输视角来理解GAN的新分支，也提高了最优传输理论在机器学习中的地位。很长一段时间以来，[GAN](https://papers.cool/arxiv/1406.2661)都是生成模型领域的“主力军”，直到最近这两年扩散模型异军突起，GAN的风头才有所下降，但其本身仍不失为一个强大的生成模型。

从形式上来看，扩散模型和GAN差异很明显，所以其研究一直都相对独立。不过，去年底的一篇论文[《Score-based Generative Modeling Secretly Minimizes the Wasserstein Distance》](https://papers.cool/arxiv/2212.06359)打破了这个隔阂：它证明了扩散模型的得分匹配损失可以写成W距离的上界形式。这意味着在某种程度上，最小化扩散模型的损失函数，实则跟WGAN一样，都是在最小化两个分布的W距离。

## 结论分析 #

具体来说，原论文的结果，是针对[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)中介绍的SDE式扩散模型的，其核心结论是不等式（其中$I_t$是$t$的非负函数，具体含义我们后来再详细介绍）  
\begin{equation}\mathcal{W}_2[p_0,q_0]\leq \int_0^T g_t^2 I_t \left(\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right]\right)^{1/2}dt + I_T \mathcal{W}_2[p_T,q_T]\label{eq:w-neq}\end{equation}  
那么怎样理解这个不等式呢？首先，扩散模型可以理解为SDE从$t=T$到$t=0$的一个运动过程，最右边的$p_T,q_T$是$T$时刻的随机采样分布，$p_T$通常就是标准正态分布，而实际应用中一般都有$q_T = p_T$，所以$\mathcal{W}_2[p_T,q_T]=0$，原论文之所以显式写出它，只是为了从理论上给出最一般的结果。

接着，左边的$p_0$，是从$p_T$采样的随机点出发，经反向SDE  
\begin{equation}d\boldsymbol{x}_t = \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) \right] dt + g_t d\boldsymbol{w}\label{eq:reverse-sde}\end{equation}  
求解得到的$t=0$时刻的值的分布，它实际上就是要生成的数据分布；而$q_0$，则是从$q_T$采样的随机点出发，经过SDE  
\begin{equation}d\boldsymbol{x}_t = \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) \right] dt + g_t d\boldsymbol{w}\end{equation}  
求解得到的$t=0$时刻的值的分布，其中$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$是$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$的神经网络近似，所以$q_0$实际就是扩散模型生成的数据分布。因此，$\mathcal{W}_2[p_0,q_0]$的含义就是数据分布与生成分布的W距离。

最后，剩下的积分项，其关键部分是  
\begin{equation}\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right]\label{eq:sm}\end{equation}  
这也正好是扩散模型的“得分匹配”损失。所以，当我们用得分匹配损失去训练扩散模型的时候，其实也间接地最小化了数据分布与生成分布的W距离。跟WGAN不同的是，WGAN优化的W距离是$\mathcal{W}_1[p_0,q_0]$而这里是$\mathcal{W}_2[p_0,q_0]$。

> **注：** 准确来说，式$\eqref{eq:sm}$还不是扩散模型的损失函数，扩散模型的损失函数应该是“条件得分匹配”，它跟得分匹配的关系是：  
>  \begin{equation}\begin{aligned}  
>  &\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\\  
>  =&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right] - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\\  
>  \leq &\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\\  
>  = &\,\mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\\  
>  \end{aligned}\end{equation}  
>  最后的结果才是扩散模型的损失函数“条件得分匹配”。第一个等号是因为恒等式$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)=\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]$，第二个不等号则是因为平方平均不等式的推广或者詹森不等式，第三个等号则是贝叶斯公式了。也就是说，条件得分匹配是得分匹配的上界，所以也是W距离的上界。

从式$\eqref{eq:w-neq}$中我们也可以简单理解为什么扩散模型的目标函数要舍去模长前面的系数了，因为W距离是概率分布的良好度量，而式$\eqref{eq:w-neq}$右端的$g_t^2 I_t$是关于$t$的单调递增函数，这意味着我们要适当加大当$t$较小时的得分匹配损失。而在[《生成扩散模型漫谈（五）：一般框架之SDE篇》](/archives/9209)我们推导过得到匹配的最终形式为：  
\begin{equation}\frac{1}{\bar{\beta}_t^2}\mathbb{E}_{\boldsymbol{x}_0\sim \tilde{p}(\boldsymbol{x}_0),\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\left[\left\Vert \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\bar{\alpha}_t\boldsymbol{x}_0 + \bar{\beta}_t\boldsymbol{\varepsilon}, t) - \boldsymbol{\varepsilon}\right\Vert^2\right]\end{equation}  
舍去系数$\frac{1}{\bar{\beta}_t^2}$等价于乘以$\bar{\beta}_t^2$，而$\bar{\beta}_t^2$也是$t$的单调递增函数。也就是说，可以简单地认为舍去系数是让训练目标更加接近两个分布的W距离。

## 准备工作 #

尽管原论文给出了不等式$\eqref{eq:w-neq}$的证明过程，但涉及到较多的最优传输相关知识，如连续性方程、梯度流等，特别是它不加证明引用的一个定理，还是放在一本梯度流专著的第8章或另一本最优传输专著的第5章，这对笔者来说阅读难度实在太大。经过一段时间的尝试，笔者终于在上周笔者完成了自己关于不等式$\eqref{eq:w-neq}$的（一部分）证明，其中只需要用到W距离的定义、微分方程基础以及柯西不等式，相比原论文的证明理解难度应该是明显降低了。经过几天的修改完善，给出如下的证明过程。

在开始证明之前，我们先做一下准备，先整理一下接下来会用到的一些基本概念和结论。首先是W距离，它定义为  
\begin{equation}\mathcal{W}_{\rho}[p,q]=\left(\inf_{\gamma\in \Pi[p,q]} \iint \gamma(\boldsymbol{x},\boldsymbol{y}) \Vert\boldsymbol{x} - \boldsymbol{y}\Vert^{\rho} d\boldsymbol{x}d\boldsymbol{y}\right)^{1/\rho}\end{equation}  
其中$\Pi[p,q]$是指所有以$p,q$为边缘分布的联合概率密度函数，它描述了具体的传输方案。本文只考虑$\rho=2$，因为只有这种情形方便后续推导。注意到W距离的定义包含了下确界运算$\inf$，这就意味着对于任意我们能写出的$\gamma\in \Pi[p,q]$，都有  
\begin{equation}\mathcal{W}_2[p,q]\leq\left(\iint \gamma(\boldsymbol{x},\boldsymbol{y}) \Vert\boldsymbol{x} - \boldsymbol{y}\Vert^{2} d\boldsymbol{x}d\boldsymbol{y}\right)^{1/2}\label{eq:core-neq}\end{equation}  
这是笔者所给证明的核心思想。证明过程的放缩，主要用到柯西不等式：  
\begin{equation}\begin{aligned}  
&\text{向量版：}\quad\boldsymbol{x}\cdot\boldsymbol{y}\leq \Vert \boldsymbol{x}\Vert \Vert\boldsymbol{y}\Vert\\\  
&\text{期望版：}\quad\mathbb{E}_{\boldsymbol{x}}\left[f(\boldsymbol{x})g(\boldsymbol{x})\right]\leq \left(\mathbb{E}_{\boldsymbol{x}}\left[f^2(\boldsymbol{x})\right]\right)^{1/2}\left(\mathbb{E}_{\boldsymbol{x}}\left[g^2(\boldsymbol{x})\right]\right)^{1/2}  
\end{aligned}\end{equation}  
证明过程中我们会假设函数$\boldsymbol{g}_t(\boldsymbol{x})$满足“单侧Lipschitz约束”，其定义为  
\begin{equation}(\boldsymbol{g}_t(\boldsymbol{x}) - \boldsymbol{g}_t(\boldsymbol{y}))\cdot(\boldsymbol{x} - \boldsymbol{y}) \leq L_t \Vert \boldsymbol{x} - \boldsymbol{y}\Vert^2\label{eq:assum}\end{equation}  
可以证明它比常见的Lipschitz约束（参考[《深度学习中的Lipschitz约束：泛化与生成模型》](/archives/6051)）更弱，即如果函数$\boldsymbol{g}_t(\boldsymbol{x})$满足Lipschitz约束，那么它一定满足单侧Lipschitz约束。

## 牛刀小试 #

不等式$\eqref{eq:w-neq}$过于一般了，一上来就试图分析一般化的结果并不利于我们的思考和理解。所以，我们先将问题简化一下，看能不能先证明一个稍弱一些的结果。怎么简化呢？首先，不等式$\eqref{eq:w-neq}$考虑了初始分布（提示，扩散模型是$t=T$到$t=0$的演化过程，所以$t=T$是初始时刻，$t=0$是终止时刻）的差异，而这里我们先考虑相同初始分布；此外，原本的反向方程$\eqref{eq:reverse-sde}$是一个SDE，这里先考虑确定性的ODE。

具体来说，我们考虑从同一个分布$q(\boldsymbol{z})$出发采样$\boldsymbol{z}$作为$T$时刻的初始值，然后分别沿着两个不同的ODE  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{x}_t),\quad \frac{d\boldsymbol{y}_t}{dt} = \boldsymbol{g}_t(\boldsymbol{y}_t)\end{equation}  
进行演化，设$t$时刻$\boldsymbol{x}_t$的分布为$p_t$、$\boldsymbol{y}_t$的分布为$q_t$，我们尝试去估计$\mathcal{W}_2[p_0,q_0]$的一个上界。

我们知道，$\boldsymbol{x}_t,\boldsymbol{y}_t$都是以$\boldsymbol{z}$为初始值通过各自的ODE演化而来，所以它们其实都是$\boldsymbol{z}$的确定性函数，更准确的记号应该是$\boldsymbol{x}_t(\boldsymbol{z}),\boldsymbol{y}_t(\boldsymbol{z})$，简单起见我们才略去了$\boldsymbol{z}$。这就意味着对应于同一个$\boldsymbol{x}$的$\boldsymbol{x}_t\leftrightarrow \boldsymbol{y}_t$构成了$p_t,q_t$的样本之间的一个对应关系（传输方案），如下图（这个图不大好画，就随便手画了一下）：  


[![近似最优传输方案示意图](/usr/uploads/2023/02/1814162448.png)](/usr/uploads/2023/02/1814162448.png "点击查看原图")

近似最优传输方案示意图

  
于是根据式$\eqref{eq:core-neq}$，我们可以写出  
\begin{equation}\mathcal{W}_2^2[p_t,q_t]\leq \mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{x}_t - \boldsymbol{y}_t\Vert^{2} \right]\triangleq \tilde{\mathcal{W}}_2^2[p_t,q_t]\label{eq:core-neq-2}\end{equation}  
下面我们对$\tilde{\mathcal{W}}_2^2[p_t,q_t]$进行放缩。为了将它跟$\boldsymbol{f}_t(\boldsymbol{x}_t),\boldsymbol{g}_t(\boldsymbol{y}_t)$联系起来，我们对它求导：

\begin{equation}\begin{aligned}  
\pm\frac{d\left(\tilde{\mathcal{W}}_2^2[p_t,q_t]\right)}{dt}=&\, \pm2\mathbb{E}_{\boldsymbol{z}}\left[(\boldsymbol{x}_t - \boldsymbol{y}_t)\cdot \left(\frac{d\boldsymbol{x}_t}{dt} - \frac{d\boldsymbol{y}_t}{dt}\right)\right] \\\\[5pt]  
=&\, \pm 2\mathbb{E}_{\boldsymbol{z}}\left[(\boldsymbol{x}_t - \boldsymbol{y}_t)\cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))\right] \\\\[5pt]  
=&\, \pm 2\mathbb{E}_{\boldsymbol{z}}\left[(\boldsymbol{x}_t - \boldsymbol{y}_t)\cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t))\right] \pm 2\mathbb{E}_{\boldsymbol{z}}\left[(\boldsymbol{x}_t - \boldsymbol{y}_t)\cdot (\boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))\right] \\\\[5pt]  
\leq&\, 2\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{x}_t - \boldsymbol{y}_t\Vert \Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert\right] + 2\mathbb{E}_{\boldsymbol{z}}\left[L_t\Vert\boldsymbol{x}_t - \boldsymbol{y}_t\Vert^2\right] \\\\[5pt]  
\leq&\, 2\left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{x}_t - \boldsymbol{y}_t\Vert^2\right]\right)^{1/2} \left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} + 2L_t\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{x}_t - \boldsymbol{y}_t\Vert^2\right] \\\\[5pt]  
=&\, 2 \tilde{\mathcal{W}}_2[p_t,q_t] \left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} + 2L_t\tilde{\mathcal{W}}_2^2[p_t,q_t] \\\\[5pt]  
\end{aligned}\label{eq:der-neq-0}\end{equation}

其中第一个不等号用到了柯西不等式的向量版，以及单侧Lipschitz约束假设$\eqref{eq:assum}$，第二个不等号则用到了柯西不等式的期望版，$\pm$的意思是最终得到的不等关系，不管取$+$还是$-$都是成立的，下面的推导只用到了$-$这一侧。结合$\left(w^2\right)'=2ww'$，我们得到  
\begin{equation}-\frac{d\tilde{\mathcal{W}}_2[p_t,q_t]}{dt} \leq \left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} + L_t\tilde{\mathcal{W}}_2[p_t,q_t] \label{eq:der-neq-1}\end{equation}  
用常数变易法，设$\tilde{\mathcal{W}}_2[p_t,q_t]=C_t \exp\left(\int_t^T L_s ds\right)$，代入上式得到  
\begin{equation}-\frac{dC_t}{dt} \leq \exp\left(-\int_t^T L_s ds\right)\left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2}\label{eq:der-neq-2}\end{equation}  
两边在$[0,T]$积分，并结合$C_T=0$（初始时刻两个分布相等，距离为0），得到  
\begin{equation}C_0 \leq \int_0^T \exp\left(-\int_t^T L_s ds\right)\left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} dt\end{equation}  
于是  
\begin{equation}\tilde{\mathcal{W}}_2[p_0,q_0] \leq C_0 \exp\left(\int_0^T L_s ds\right) =\int_0^T I_t\left(\mathbb{E}_{\boldsymbol{z}}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} dt\end{equation}  
其中$I_t = \exp\left(\int_0^t L_s ds\right)$。根据式$\eqref{eq:core-neq-2}$，这也是$\mathcal{W}_2[p_0,q_0]$的上界。最后，由于求期望的式子只是$\boldsymbol{x}_t$的函数，$\boldsymbol{x}_t$又是$\boldsymbol{z}$的确定性函数，对于它关于$\boldsymbol{z}$的期望等价于直接关于$\boldsymbol{x}_t$的期望，于是：  
\begin{equation}\mathcal{W}_2[p_0,q_0] \leq\int_0^T I_t\left(\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} dt\label{eq:w-neq-0}\end{equation}

## 一鼓作气 #

实际上，简化版的不等式$\eqref{eq:w-neq-0}$已经和更一般的$\eqref{eq:w-neq}$没有本质区别了，它的推导过程已经包含了导出完整结果的一般思路，下面我们来完成剩余的推导过程。

首先，我们将式$\eqref{eq:w-neq-0}$推广到不同初始分布的场景，假设两个初始分布为$p_T(\boldsymbol{z}_1),q_T(\boldsymbol{z}_2)$，从$p_T(\boldsymbol{z}_1)$采样初始值演化$\boldsymbol{x}_t$，从$q_T(\boldsymbol{z}_2)$采样初始值演化$\boldsymbol{y}_t$，所以此时$\boldsymbol{x}_t,\boldsymbol{y}_t$分别是$\boldsymbol{z}_1,\boldsymbol{z}_2$的函数，而不是像之前那样是同一个$\boldsymbol{z}$的函数，所以无法直接构造一个传输方案。所以，我们还需要$\boldsymbol{z}_1,\boldsymbol{z}_2$之间的一个对应关系（传输方案），我们将它选择为$p_T(\boldsymbol{z}_1),q_T(\boldsymbol{z}_2)$之间的一个最优传输方案$\gamma^*(\boldsymbol{z}_1,\boldsymbol{z}_2)$。于是，我们可以写出类似式$\eqref{eq:core-neq-2}$的结果：  
\begin{equation}\mathcal{W}_2^2[p_t,q_t]\leq \mathbb{E}_{\boldsymbol{z}_1,\boldsymbol{z}_2\sim \gamma^*(\boldsymbol{z}_1,\boldsymbol{z}_2)}\left[\Vert\boldsymbol{x}_t - \boldsymbol{y}_t\Vert^{2} \right]\triangleq \tilde{\mathcal{W}}_2^2[p_t,q_t]\label{eq:core-neq-3}\end{equation}  
由于定义的一致性，那么放缩过程$\eqref{eq:der-neq-0}$同样是成立的，只不过期望$\mathbb{E}_{\boldsymbol{z}}$换成了$\mathbb{E}_{\boldsymbol{z}_1,\boldsymbol{z}_2}$，所以不等式$\eqref{eq:der-neq-1}$、$\eqref{eq:der-neq-2}$也是成立的。不同的是在对$\eqref{eq:der-neq-2}$两端在$[0,T]$积分时，不再有$C_T = 0$，而是根据定义有$C_T=\tilde{\mathcal{W}}_2[p_T,q_T]=\mathcal{W}_2[p_T,q_T]$。所以，最终的结果是  
\begin{equation}\mathcal{W}_2[p_0,q_0] \leq\int_0^T I_t\left(\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\Vert\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\Vert^2\right]\right)^{1/2} dt + I_T \mathcal{W}_2[p_T,q_T]\label{eq:w-neq-1}\end{equation}

最后，我们回到扩散模型。在[《生成扩散模型漫谈（六）：一般框架之ODE篇》](/archives/9228)我们已经推导过，同一个前向扩散过程，实际上对应一簇反向过程：  
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 + \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}\label{eq:sde-reverse-2}\end{equation}  
其中$\sigma_t$是可以自由选择的标准差函数，当$\sigma_t=g_t$时，那么就是方程$\eqref{eq:reverse-sde}$。由于我们上面分析的是ODE，所以我们先考虑$\sigma_t=0$的情形，此时结果$\eqref{eq:w-neq-1}$依然可用，只不过将$\boldsymbol{f}_t(\boldsymbol{x}_t)$换成$\boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$、将$\boldsymbol{g}_t(\boldsymbol{x}_t)$换成$\boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$，代入式$\eqref{eq:w-neq-1}$后就得到文章开头的结论$\eqref{eq:w-neq}$了。当然别忘了我们推导过程中对$\boldsymbol{g}_t(\boldsymbol{x}_t)$所做的单侧Lipschitz约束假设$\eqref{eq:assum}$，现在可以分别对$\boldsymbol{f}_t(\boldsymbol{x}_t)$、$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$做出假设，这些细节就不展开了。

## 艰难收尾 #

按照流程，接下来我们应该再接再厉，完成$\sigma_t\neq 0$的收尾证明。不过很遗憾，本文的思路不能完全证明SDE的情形，下面给出笔者的分析过程。事实上，对于大部分读者来说，了解到上一节的ODE例子就可以窥见式$\eqref{eq:w-neq-1}$的精髓了，完整的细节也不是太重要。

简单起见，下面我们以$\eqref{eq:reverse-sde}$为例，更一般的$\eqref{eq:sde-reverse-2}$也可以类似地分析。我们需要估算的是如下两个SDE的演化轨迹分布差异：  
\begin{equation}\left\\{\begin{aligned}  
d\boldsymbol{x}_t =&\, \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) \right] dt + g_t d\boldsymbol{w}\\\\[5pt]  
d\boldsymbol{y}_t =&\, \left[\boldsymbol{f}_t(\boldsymbol{y}_t) - g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t,t) \right] dt + g_t d\boldsymbol{w}  
\end{aligned}\right.\end{equation}  
也就是将准确的$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$换成近似的$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t,t)$，对最终分布的影响有多大。笔者的证明思路同样是将它转化为ODE，继而用回前面的证明过程。首先，根据式$\eqref{eq:sde-reverse-2}$，我们知道第一个SDE对应的ODE为：  
\begin{equation}  
d\boldsymbol{x}_t = \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) \right] dt + g_t d\boldsymbol{w}\\\  
\Downarrow \\\  
d\boldsymbol{x}_t = \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) \right] dt  
\end{equation}  
至于第二个SDE对应的ODE的推导有些技巧，需要先变为$-g_t^2\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t)$的形式，然后再利用式$\eqref{eq:sde-reverse-2}$：  
\begin{equation}  
d\boldsymbol{y}_t = \left[\boldsymbol{f}_t(\boldsymbol{y}_t) - g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t,t) \right] dt + g_t d\boldsymbol{w} \\\  
\Downarrow \\\  
d\boldsymbol{y}_t = \Big[\underbrace{\boldsymbol{f}_t(\boldsymbol{y}_t) - g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t,t) + g_t^2\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t)}_{\text{看成整体}} - g_t^2\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t) \Big] dt + g_t d\boldsymbol{w} \\\  
\Downarrow \\\  
d\boldsymbol{y}_t = \left[\boldsymbol{f}_t(\boldsymbol{y}_t) - g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t,t) + g_t^2\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t) \right] dt \\\  
\Downarrow \\\  
d\boldsymbol{y}_t = \left[\boldsymbol{f}_t(\boldsymbol{y}_t) - g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t,t) + \frac{1}{2}g_t^2\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t)\right] dt  
\end{equation}  
对这两个ODE重复放缩过程$\eqref{eq:der-neq-0}$（$\pm$取负号），那么主要的区别是多出来一项  
\begin{equation}-\frac{1}{2}g_t^2\mathbb{E}_{\boldsymbol{z}}\left[(\boldsymbol{x}_t - \boldsymbol{y}_t)\cdot (\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)-\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t))\right]\end{equation}  
如果这一项小于等于0，那么放缩过程$\eqref{eq:der-neq-0}$依然成立，后面的所有结果同样也成立，最终结论的形式跟式$\eqref{eq:w-neq-1}$一致。

所以，现在剩下的问题就是能否证明  
\begin{equation}\mathbb{E}_{\boldsymbol{z}}\left[(\boldsymbol{x}_t - \boldsymbol{y}_t)\cdot (\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)-\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t))\right]\geq 0\end{equation}  
很遗憾，可以举出反例表明它一般是不成立的。原论文的证明过程也出现了类似的一项，不过求期望的分布不是$\boldsymbol{z}$，而是$\boldsymbol{x}_t,\boldsymbol{y}_t$的最优传输分布，在此前提之下，原论文直接抛出两篇文献的结论作为引理，寥寥几行便完成了证明。不得不说原论文作者们真的很熟悉最优传输相关内容，各种文献结论“信手拈来”，就是苦了笔者这样的新手读者，想要彻底理解却难以下手，只能到此为止了。

特别注意的是，我们不能对$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$或$\nabla_{\boldsymbol{y}_t}\log q_t(\boldsymbol{y}_t)$做单侧Lipschitz约束假设，因为很容易举出其对数梯度不满足单侧Lipschitz约束的分布，因此，要证明这个不等式，只能参考原论文的思路通过分布本身的性质来进行，不能强加额外的假设。

## 文章小结 #

本文介绍了一个新的理论结果，显示扩散模型的得分匹配损失可以写成W距离的上界形式，并给出了自己的部分证明。这个结果意味着，在某种程度上扩散模型和WGAN都有着相同的优化目标，扩散模型也在偷偷优化W距离！

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9467>_

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

苏剑林. (Feb. 14, 2023). 《生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9467>

@online{kexuefm-9467,  
title={生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配},  
author={苏剑林},  
year={2023},  
month={Feb},  
url={\url{https://spaces.ac.cn/archives/9467}},  
} 


---

## 公式推导与注释

本节提供文章中涉及的所有核心数学公式的详细推导过程，包括Wasserstein距离的理论基础、最优传输理论、以及W距离与得分匹配之间的深层联系。

### 1. Wasserstein距离的数学定义与性质

#### 1.1 基本定义

Wasserstein距离（也称为Earth Mover's Distance）是测度论中用于衡量两个概率分布之间差异的重要工具。

**定义1.1（Wasserstein-p距离）**：设 $p, q$ 是 $\mathbb{R}^d$ 上的两个概率测度，则 $p$-Wasserstein距离定义为：

$$
\mathcal{W}_{\rho}[p,q] = \left(\inf_{\gamma \in \Pi[p,q]} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|\boldsymbol{x} - \boldsymbol{y}\|^{\rho} d\gamma(\boldsymbol{x}, \boldsymbol{y})\right)^{1/\rho}
$$

其中 $\Pi[p,q]$ 表示所有以 $p$ 和 $q$ 为边缘分布的联合概率测度的集合，即：

$$
\Pi[p,q] = \left\{\gamma \in \mathcal{P}(\mathbb{R}^d \times \mathbb{R}^d) : \int_{\mathbb{R}^d} d\gamma(\boldsymbol{x}, \boldsymbol{y}) = q(\boldsymbol{y}), \int_{\mathbb{R}^d} d\gamma(\boldsymbol{x}, \boldsymbol{y}) = p(\boldsymbol{x})\right\}
$$

**注释**：这个定义的直观理解是，我们想要将分布 $p$ 转换为分布 $q$，而 $\gamma(\boldsymbol{x}, \boldsymbol{y})$ 表示从位置 $\boldsymbol{x}$ 运输到位置 $\boldsymbol{y}$ 的"质量"。运输成本与距离的 $\rho$ 次方成正比，Wasserstein距离就是最小的总运输成本的 $\rho$ 次根。

#### 1.2 边缘分布约束的详细解释

对于任意 $\gamma \in \Pi[p,q]$，边缘分布约束意味着：

$$
\int_{\mathbb{R}^d} \gamma(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y} = p(\boldsymbol{x}), \quad \forall \boldsymbol{x} \in \mathbb{R}^d
$$

$$
\int_{\mathbb{R}^d} \gamma(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x} = q(\boldsymbol{y}), \quad \forall \boldsymbol{y} \in \mathbb{R}^d
$$

**推导1.1**：验证边缘分布约束的物理意义。

设 $\gamma$ 为一个运输方案，我们对所有目标位置 $\boldsymbol{y}$ 积分：

$$
\int_{\mathbb{R}^d} \gamma(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y}
$$

这表示从源位置 $\boldsymbol{x}$ 运出的总质量。根据质量守恒，这应该等于 $p(\boldsymbol{x})$，即源分布在 $\boldsymbol{x}$ 处的概率密度。

类似地，对所有源位置 $\boldsymbol{x}$ 积分：

$$
\int_{\mathbb{R}^d} \gamma(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x}
$$

这表示运输到目标位置 $\boldsymbol{y}$ 的总质量，应该等于 $q(\boldsymbol{y})$。

#### 1.3 W2距离的特殊性质

本文重点关注 $\rho = 2$ 的情况，即 $\mathcal{W}_2$ 距离：

$$
\mathcal{W}_2[p,q] = \left(\inf_{\gamma \in \Pi[p,q]} \int_{\mathbb{R}^d \times \mathbb{R}^d} \|\boldsymbol{x} - \boldsymbol{y}\|^2 d\gamma(\boldsymbol{x}, \boldsymbol{y})\right)^{1/2}
$$

**定理1.1（W2距离是度量）**：$\mathcal{W}_2$ 满足度量的三个基本性质。

**证明**：

(1) 非负性：$\mathcal{W}_2[p,q] \geq 0$，且 $\mathcal{W}_2[p,p] = 0$。

这是显然的，因为距离平方 $\|\boldsymbol{x} - \boldsymbol{y}\|^2 \geq 0$，且当 $p = q$ 时，可以取 $\gamma(\boldsymbol{x}, \boldsymbol{y}) = p(\boldsymbol{x})\delta(\boldsymbol{x} - \boldsymbol{y})$（对角线上的测度），此时：

$$
\int_{\mathbb{R}^d \times \mathbb{R}^d} \|\boldsymbol{x} - \boldsymbol{y}\|^2 \delta(\boldsymbol{x} - \boldsymbol{y}) p(\boldsymbol{x}) d\boldsymbol{x} d\boldsymbol{y} = \int_{\mathbb{R}^d} \|\boldsymbol{x} - \boldsymbol{x}\|^2 p(\boldsymbol{x}) d\boldsymbol{x} = 0
$$

(2) 对称性：$\mathcal{W}_2[p,q] = \mathcal{W}_2[q,p]$。

对于任何 $\gamma \in \Pi[p,q]$，定义 $\tilde{\gamma}(\boldsymbol{x}, \boldsymbol{y}) = \gamma(\boldsymbol{y}, \boldsymbol{x})$，则 $\tilde{\gamma} \in \Pi[q,p]$，且：

$$
\int \|\boldsymbol{x} - \boldsymbol{y}\|^2 d\gamma(\boldsymbol{x}, \boldsymbol{y}) = \int \|\boldsymbol{y} - \boldsymbol{x}\|^2 d\tilde{\gamma}(\boldsymbol{x}, \boldsymbol{y})
$$

因此两个下确界相等。

(3) 三角不等式：$\mathcal{W}_2[p,r] \leq \mathcal{W}_2[p,q] + \mathcal{W}_2[q,r]$。

这个证明较为复杂，需要使用Gluing引理来构造合适的联合测度。

### 2. Kantorovich对偶理论

#### 2.1 Kantorovich对偶公式

**定理2.1（Kantorovich对偶）**：对于 $\mathcal{W}_1$ 距离，存在对偶表示：

$$
\mathcal{W}_1[p,q] = \sup_{\|f\|_L \leq 1} \left(\int f(\boldsymbol{x}) dp(\boldsymbol{x}) - \int f(\boldsymbol{y}) dq(\boldsymbol{y})\right)
$$

其中 $\|f\|_L \leq 1$ 表示 $f$ 是1-Lipschitz函数，即 $|f(\boldsymbol{x}) - f(\boldsymbol{y})| \leq \|\boldsymbol{x} - \boldsymbol{y}\|$。

**推导2.1**：从原始问题到对偶问题的转换。

原始问题（Primal Problem）：

$$
\min_{\gamma \in \Pi[p,q]} \int_{\mathbb{R}^d \times \mathbb{R}^d} c(\boldsymbol{x}, \boldsymbol{y}) d\gamma(\boldsymbol{x}, \boldsymbol{y})
$$

其中 $c(\boldsymbol{x}, \boldsymbol{y}) = \|\boldsymbol{x} - \boldsymbol{y}\|$ 是成本函数。

引入Lagrange乘子 $f(\boldsymbol{x})$ 和 $g(\boldsymbol{y})$ 对应边缘分布约束，对偶问题（Dual Problem）为：

$$
\max_{f,g} \left(\int f(\boldsymbol{x}) dp(\boldsymbol{x}) + \int g(\boldsymbol{y}) dq(\boldsymbol{y})\right)
$$

约束条件为：$f(\boldsymbol{x}) + g(\boldsymbol{y}) \leq c(\boldsymbol{x}, \boldsymbol{y})$。

通过变量替换 $g(\boldsymbol{y}) = -f(\boldsymbol{y})$，可以得到：

$$
f(\boldsymbol{x}) - f(\boldsymbol{y}) \leq \|\boldsymbol{x} - \boldsymbol{y}\|
$$

这正是1-Lipschitz条件。

#### 2.2 对偶间隙与强对偶性

**定理2.2（强对偶性）**：在适当的紧性条件下，Kantorovich问题满足强对偶性，即原始问题的最优值等于对偶问题的最优值：

$$
\inf_{\gamma \in \Pi[p,q]} \int c(\boldsymbol{x}, \boldsymbol{y}) d\gamma = \sup_{f \oplus g \leq c} \left(\int f dp + \int g dq\right)
$$

**证明思路**：

利用Fenchel-Rockafellar对偶定理。定义算子 $A: \mathcal{M}(\mathbb{R}^d \times \mathbb{R}^d) \to \mathcal{M}(\mathbb{R}^d) \times \mathcal{M}(\mathbb{R}^d)$：

$$
A(\gamma) = \left(\int \gamma(\cdot, \boldsymbol{y}) d\boldsymbol{y}, \int \gamma(\boldsymbol{x}, \cdot) d\boldsymbol{x}\right)
$$

则原始问题可以写为：

$$
\min_{\gamma \geq 0, A\gamma = (p,q)} \int c d\gamma
$$

应用Fenchel-Rockafellar定理，在适当的内点条件下，对偶间隙为零。

#### 2.3 最优运输映射的存在性

**定理2.3（Brenier定理）**：当 $\rho = 2$ 且源分布 $p$ 绝对连续时，存在唯一的最优运输映射 $T: \mathbb{R}^d \to \mathbb{R}^d$ 使得：

$$
T_{\#}p = q
$$

且 $T$ 是某个凸函数的梯度，即存在凸函数 $\phi$ 使得 $T = \nabla \phi$。

**推导2.2**：最优运输映射与凸函数的关系。

设 $T$ 是最优运输映射，定义 $\phi$ 为：

$$
\phi(\boldsymbol{x}) = \sup_{\boldsymbol{y}} \left(\boldsymbol{x} \cdot \boldsymbol{y} - \frac{1}{2}\|\boldsymbol{y}\|^2\right)
$$

这是一个凸函数（作为线性函数的上确界）。

对于最优运输，必须满足c-单调性条件：对于任意 $(x_i, y_i)$，$i = 1, 2$ 在最优运输计划的支撑集中，有：

$$
\|x_1 - y_1\|^2 + \|x_2 - y_2\|^2 \leq \|x_1 - y_2\|^2 + \|x_2 - y_1\|^2
$$

展开这个不等式：

$$
x_1 \cdot y_1 + x_2 \cdot y_2 \geq x_1 \cdot y_2 + x_2 \cdot y_1
$$

这正是凸函数梯度的单调性条件。

### 3. Benamou-Brenier动态公式

#### 3.1 动态最优传输问题

**定理3.1（Benamou-Brenier公式）**：Wasserstein-2距离可以表示为：

$$
\mathcal{W}_2^2[p_0, p_1] = \inf_{(\rho_t, \boldsymbol{v}_t)} \int_0^1 \int_{\mathbb{R}^d} \rho_t(\boldsymbol{x}) \|\boldsymbol{v}_t(\boldsymbol{x})\|^2 d\boldsymbol{x} dt
$$

约束条件为连续性方程：

$$
\frac{\partial \rho_t}{\partial t} + \nabla \cdot (\rho_t \boldsymbol{v}_t) = 0
$$

边界条件为：$\rho_0 = p_0$，$\rho_1 = p_1$。

**推导3.1**：从静态到动态的转换。

考虑一个从 $p_0$ 到 $p_1$ 的连续路径 $\{\rho_t\}_{t \in [0,1]}$，伴随速度场 $\boldsymbol{v}_t$。

路径上任意粒子的轨迹满足：

$$
\frac{d\boldsymbol{X}_t}{dt} = \boldsymbol{v}_t(\boldsymbol{X}_t)
$$

运输总成本可以表示为：

$$
\text{Cost} = \int_0^1 \int_{\mathbb{R}^d} \rho_t(\boldsymbol{x}) \|\boldsymbol{v}_t(\boldsymbol{x})\|^2 d\boldsymbol{x} dt
$$

这里 $\rho_t(\boldsymbol{x}) \|\boldsymbol{v}_t(\boldsymbol{x})\|^2$ 可以理解为位置 $\boldsymbol{x}$ 处的动能密度。

质量守恒要求 $\rho_t$ 满足连续性方程：

$$
\frac{\partial \rho_t}{\partial t} + \nabla \cdot (\rho_t \boldsymbol{v}_t) = 0
$$

#### 3.2 连续性方程的推导

**推导3.2**：连续性方程的物理意义。

考虑一个固定区域 $\Omega \subset \mathbb{R}^d$，区域内的总质量为：

$$
M(t) = \int_{\Omega} \rho_t(\boldsymbol{x}) d\boldsymbol{x}
$$

质量的变化率为：

$$
\frac{dM(t)}{dt} = \int_{\Omega} \frac{\partial \rho_t(\boldsymbol{x})}{\partial t} d\boldsymbol{x}
$$

另一方面，质量变化也等于通过边界 $\partial\Omega$ 流出的通量：

$$
\frac{dM(t)}{dt} = -\int_{\partial\Omega} \rho_t \boldsymbol{v}_t \cdot \boldsymbol{n} dS
$$

其中 $\boldsymbol{n}$ 是外法向量。应用散度定理（Divergence Theorem）：

$$
\int_{\partial\Omega} \rho_t \boldsymbol{v}_t \cdot \boldsymbol{n} dS = \int_{\Omega} \nabla \cdot (\rho_t \boldsymbol{v}_t) d\boldsymbol{x}
$$

因此：

$$
\int_{\Omega} \frac{\partial \rho_t}{\partial t} d\boldsymbol{x} = -\int_{\Omega} \nabla \cdot (\rho_t \boldsymbol{v}_t) d\boldsymbol{x}
$$

由于 $\Omega$ 是任意的，我们得到连续性方程：

$$
\frac{\partial \rho_t}{\partial t} + \nabla \cdot (\rho_t \boldsymbol{v}_t) = 0
$$

#### 3.3 动态公式与静态公式的等价性

**推导3.3**：证明动态公式确实给出 $\mathcal{W}_2$ 距离。

设 $T: \mathbb{R}^d \to \mathbb{R}^d$ 是从 $p_0$ 到 $p_1$ 的最优运输映射。定义测地线路径：

$$
\boldsymbol{X}_t = (1-t)\boldsymbol{x} + t T(\boldsymbol{x}), \quad t \in [0,1]
$$

其中 $\boldsymbol{x} \sim p_0$。则速度场为：

$$
\boldsymbol{v}_t = \frac{d\boldsymbol{X}_t}{dt} = T(\boldsymbol{x}) - \boldsymbol{x}
$$

密度 $\rho_t$ 由推前测度定义：$\rho_t = (\boldsymbol{X}_t)_{\#}p_0$。

计算动力学成本：

$$
\begin{aligned}
\int_0^1 \int_{\mathbb{R}^d} \rho_t(\boldsymbol{y}) \|\boldsymbol{v}_t(\boldsymbol{y})\|^2 d\boldsymbol{y} dt
&= \int_0^1 \int_{\mathbb{R}^d} \|T(\boldsymbol{x}) - \boldsymbol{x}\|^2 p_0(\boldsymbol{x}) d\boldsymbol{x} dt \\
&= \int_{\mathbb{R}^d} \|T(\boldsymbol{x}) - \boldsymbol{x}\|^2 p_0(\boldsymbol{x}) d\boldsymbol{x}
\end{aligned}
$$

最后这个积分正是 $\mathcal{W}_2^2[p_0, p_1]$ 的定义（当 $T$ 是最优运输映射时）。

### 4. 单侧Lipschitz条件

#### 4.1 单侧Lipschitz条件的定义

**定义4.1（单侧Lipschitz）**：函数 $\boldsymbol{g}: \mathbb{R}^d \to \mathbb{R}^d$ 满足单侧Lipschitz条件，如果存在常数 $L$ 使得：

$$
(\boldsymbol{g}(\boldsymbol{x}) - \boldsymbol{g}(\boldsymbol{y})) \cdot (\boldsymbol{x} - \boldsymbol{y}) \leq L \|\boldsymbol{x} - \boldsymbol{y}\|^2
$$

对所有 $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^d$ 成立。

**注释**：单侧Lipschitz条件比标准Lipschitz条件（$\|\boldsymbol{g}(\boldsymbol{x}) - \boldsymbol{g}(\boldsymbol{y})\| \leq L \|\boldsymbol{x} - \boldsymbol{y}\|$）更弱，它只约束了函数变化在连接两点方向上的投影。

#### 4.2 单侧Lipschitz条件的性质

**命题4.1**：如果 $\boldsymbol{g}$ 满足标准Lipschitz条件（常数为 $L$），则它满足单侧Lipschitz条件（常数也为 $L$）。

**证明**：

由Cauchy-Schwarz不等式：

$$
\begin{aligned}
(\boldsymbol{g}(\boldsymbol{x}) - \boldsymbol{g}(\boldsymbol{y})) \cdot (\boldsymbol{x} - \boldsymbol{y})
&\leq \|\boldsymbol{g}(\boldsymbol{x}) - \boldsymbol{g}(\boldsymbol{y})\| \cdot \|\boldsymbol{x} - \boldsymbol{y}\| \\
&\leq L \|\boldsymbol{x} - \boldsymbol{y}\| \cdot \|\boldsymbol{x} - \boldsymbol{y}\| \\
&= L \|\boldsymbol{x} - \boldsymbol{y}\|^2
\end{aligned}
$$

**推导4.1**：单侧Lipschitz条件与梯度的关系。

假设 $\boldsymbol{g}$ 可微，对于任意 $\boldsymbol{x}, \boldsymbol{y}$，考虑函数：

$$
h(t) = \boldsymbol{g}(t\boldsymbol{x} + (1-t)\boldsymbol{y}) \cdot (\boldsymbol{x} - \boldsymbol{y})
$$

则：

$$
h'(t) = \nabla\boldsymbol{g}(t\boldsymbol{x} + (1-t)\boldsymbol{y})(\boldsymbol{x} - \boldsymbol{y}) \cdot (\boldsymbol{x} - \boldsymbol{y})
$$

根据微积分基本定理：

$$
\begin{aligned}
&\,(\boldsymbol{g}(\boldsymbol{x}) - \boldsymbol{g}(\boldsymbol{y})) \cdot (\boldsymbol{x} - \boldsymbol{y}) \\
=&\, h(1) - h(0) \\
=&\, \int_0^1 h'(t) dt \\
=&\, \int_0^1 (\boldsymbol{x} - \boldsymbol{y})^T \nabla\boldsymbol{g}(t\boldsymbol{x} + (1-t)\boldsymbol{y}) (\boldsymbol{x} - \boldsymbol{y}) dt
\end{aligned}
$$

如果 Jacobian 矩阵 $\nabla\boldsymbol{g}$ 的对称部分的最大特征值有界，即：

$$
\lambda_{\max}\left(\frac{\nabla\boldsymbol{g} + \nabla\boldsymbol{g}^T}{2}\right) \leq L
$$

则单侧Lipschitz条件成立。

#### 4.3 单侧Lipschitz条件在ODE中的应用

**定理4.1（Gronwall不等式的应用）**：考虑两个ODE：

$$
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{x}_t), \quad \frac{d\boldsymbol{y}_t}{dt} = \boldsymbol{g}_t(\boldsymbol{y}_t)
$$

如果 $\boldsymbol{g}_t$ 满足单侧Lipschitz条件（常数 $L_t$），则：

$$
\frac{d}{dt}\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2 \leq 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)) + 2L_t\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2
$$

**证明**：

$$
\begin{aligned}
\frac{d}{dt}\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2
&= 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot \left(\frac{d\boldsymbol{x}_t}{dt} - \frac{d\boldsymbol{y}_t}{dt}\right) \\
&= 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t)) \\
&= 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)) \\
&\quad + 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t)) \\
&\leq 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)) + 2L_t\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2
\end{aligned}
$$

最后一步使用了单侧Lipschitz条件。

### 5. 核心不等式的详细推导

#### 5.1 传输方案的构造

**推导5.1**：从ODE轨迹构造传输方案。

考虑从同一初始分布 $q(\boldsymbol{z})$ 出发的两条ODE轨迹：

$$
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{x}_t), \quad \boldsymbol{x}_T = \boldsymbol{z}
$$

$$
\frac{d\boldsymbol{y}_t}{dt} = \boldsymbol{g}_t(\boldsymbol{y}_t), \quad \boldsymbol{y}_T = \boldsymbol{z}
$$

由于 $\boldsymbol{x}_t$ 和 $\boldsymbol{y}_t$ 都由 $\boldsymbol{z}$ 唯一确定，我们可以写作 $\boldsymbol{x}_t(\boldsymbol{z})$ 和 $\boldsymbol{y}_t(\boldsymbol{z})$。

定义联合分布：

$$
\gamma_t(\boldsymbol{x}, \boldsymbol{y}) = \int q(\boldsymbol{z}) \delta(\boldsymbol{x} - \boldsymbol{x}_t(\boldsymbol{z})) \delta(\boldsymbol{y} - \boldsymbol{y}_t(\boldsymbol{z})) d\boldsymbol{z}
$$

验证边缘分布：

$$
\begin{aligned}
\int \gamma_t(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{y}
&= \int q(\boldsymbol{z}) \delta(\boldsymbol{x} - \boldsymbol{x}_t(\boldsymbol{z})) d\boldsymbol{z} \\
&= p_t(\boldsymbol{x})
\end{aligned}
$$

其中使用了推前测度的定义：$p_t = (\boldsymbol{x}_t)_{\#}q$。

类似地，$\int \gamma_t(\boldsymbol{x}, \boldsymbol{y}) d\boldsymbol{x} = q_t(\boldsymbol{y})$。

因此 $\gamma_t \in \Pi[p_t, q_t]$。

#### 5.2 上界的建立

**推导5.2**：从传输方案到Wasserstein距离的上界。

根据Wasserstein距离的定义：

$$
\mathcal{W}_2^2[p_t, q_t] = \inf_{\gamma \in \Pi[p_t, q_t]} \int \|\boldsymbol{x} - \boldsymbol{y}\|^2 d\gamma(\boldsymbol{x}, \boldsymbol{y})
$$

由于 $\gamma_t$ 是一个可行的传输方案，我们有：

$$
\begin{aligned}
\mathcal{W}_2^2[p_t, q_t]
&\leq \int \|\boldsymbol{x} - \boldsymbol{y}\|^2 d\gamma_t(\boldsymbol{x}, \boldsymbol{y}) \\
&= \int \|\boldsymbol{x}_t(\boldsymbol{z}) - \boldsymbol{y}_t(\boldsymbol{z})\|^2 q(\boldsymbol{z}) d\boldsymbol{z} \\
&= \mathbb{E}_{\boldsymbol{z} \sim q}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2] \\
&\triangleq \tilde{\mathcal{W}}_2^2[p_t, q_t]
\end{aligned}
$$

我们定义 $\tilde{\mathcal{W}}_2^2[p_t, q_t]$ 为这个上界，下面的目标是对它进行放缩。

#### 5.3 时间导数的计算

**推导5.3**：计算 $\tilde{\mathcal{W}}_2^2[p_t, q_t]$ 对时间的导数。

$$
\begin{aligned}
\frac{d}{dt}\tilde{\mathcal{W}}_2^2[p_t, q_t]
&= \frac{d}{dt}\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2] \\
&= \mathbb{E}_{\boldsymbol{z}}\left[\frac{d}{dt}\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2\right]
\end{aligned}
$$

使用链式法则：

$$
\frac{d}{dt}\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2 = \frac{d}{dt}[(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{x}_t - \boldsymbol{y}_t)]
$$

$$
= 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot \left(\frac{d\boldsymbol{x}_t}{dt} - \frac{d\boldsymbol{y}_t}{dt}\right)
$$

代入ODE：

$$
= 2(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))
$$

因此：

$$
\frac{d}{dt}\tilde{\mathcal{W}}_2^2[p_t, q_t] = 2\mathbb{E}_{\boldsymbol{z}}[(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))]
$$

**注释**：这里我们可以交换求导和期望的顺序，前提是满足适当的可积性条件（Dominated Convergence Theorem）。

#### 5.4 巧妙的拆分技巧

**推导5.4**：将差项拆分为两部分。

关键技巧是引入中间项 $\boldsymbol{g}_t(\boldsymbol{x}_t)$：

$$
\begin{aligned}
&\,(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t)) \\
=&\, (\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t) + \boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t)) \\
=&\, (\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)) \\
&\, + (\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))
\end{aligned}
$$

第一项衡量了两个向量场 $\boldsymbol{f}_t$ 和 $\boldsymbol{g}_t$ 在同一点的差异。

第二项衡量了向量场 $\boldsymbol{g}_t$ 在不同点之间的变化。

### 6. Cauchy-Schwarz不等式的应用

#### 6.1 向量形式的Cauchy-Schwarz不等式

**引理6.1（Cauchy-Schwarz不等式）**：对于任意向量 $\boldsymbol{u}, \boldsymbol{v} \in \mathbb{R}^d$：

$$
\boldsymbol{u} \cdot \boldsymbol{v} \leq |\boldsymbol{u} \cdot \boldsymbol{v}| \leq \|\boldsymbol{u}\| \|\boldsymbol{v}\|
$$

等号成立当且仅当 $\boldsymbol{u}$ 和 $\boldsymbol{v}$ 共线。

**证明**：

考虑函数 $f(\lambda) = \|\boldsymbol{u} - \lambda\boldsymbol{v}\|^2 \geq 0$，展开：

$$
f(\lambda) = \|\boldsymbol{u}\|^2 - 2\lambda(\boldsymbol{u} \cdot \boldsymbol{v}) + \lambda^2\|\boldsymbol{v}\|^2
$$

这是关于 $\lambda$ 的二次函数，其判别式必须非正：

$$
\Delta = 4(\boldsymbol{u} \cdot \boldsymbol{v})^2 - 4\|\boldsymbol{u}\|^2\|\boldsymbol{v}\|^2 \leq 0
$$

因此：

$$
(\boldsymbol{u} \cdot \boldsymbol{v})^2 \leq \|\boldsymbol{u}\|^2\|\boldsymbol{v}\|^2
$$

取平方根即得结论。

#### 6.2 应用到第一项

**推导6.1**：对第一项应用Cauchy-Schwarz不等式。

$$
\begin{aligned}
&\,\mathbb{E}_{\boldsymbol{z}}[(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t))] \\
\leq&\, \mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\| \cdot \|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|]
\end{aligned}
$$

这是向量Cauchy-Schwarz不等式的直接应用。注意这里两边都可能是负数，但取绝对值后不等式依然成立。

#### 6.3 期望形式的Cauchy-Schwarz不等式

**引理6.2（期望版Cauchy-Schwarz）**：对于随机变量 $X, Y$：

$$
|\mathbb{E}[XY]| \leq \mathbb{E}[|XY|] \leq \sqrt{\mathbb{E}[X^2]} \sqrt{\mathbb{E}[Y^2]}
$$

**证明思路**：

与向量版本类似，考虑 $\mathbb{E}[(X - \lambda Y)^2] \geq 0$：

$$
\mathbb{E}[X^2] - 2\lambda\mathbb{E}[XY] + \lambda^2\mathbb{E}[Y^2] \geq 0
$$

判别式非正：

$$
4(\mathbb{E}[XY])^2 - 4\mathbb{E}[X^2]\mathbb{E}[Y^2] \leq 0
$$

#### 6.4 应用到放缩过程

**推导6.2**：继续对第一项进行放缩。

$$
\begin{aligned}
&\,\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\| \cdot \|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|] \\
\leq&\, \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2]} \cdot \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} \\
=&\, \tilde{\mathcal{W}}_2[p_t, q_t] \cdot \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]}
\end{aligned}
$$

这里利用了 $\tilde{\mathcal{W}}_2^2[p_t, q_t] = \mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2]$。

#### 6.5 应用到第二项

**推导6.3**：对第二项应用单侧Lipschitz条件。

根据单侧Lipschitz条件（式 $\eqref{eq:assum}$）：

$$
(\boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t)) \cdot (\boldsymbol{x}_t - \boldsymbol{y}_t) \leq L_t\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2
$$

取期望：

$$
\mathbb{E}_{\boldsymbol{z}}[(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))] \leq L_t\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2] = L_t\tilde{\mathcal{W}}_2^2[p_t, q_t]
$$

### 7. 微分不等式的求解

#### 7.1 综合两项得到微分不等式

**推导7.1**：合并所有不等式。

将前面的结果代回原导数表达式：

$$
\begin{aligned}
\frac{d}{dt}\tilde{\mathcal{W}}_2^2[p_t, q_t]
&= 2\mathbb{E}_{\boldsymbol{z}}[(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t))] \\
&\quad + 2\mathbb{E}_{\boldsymbol{z}}[(\boldsymbol{x}_t - \boldsymbol{y}_t) \cdot (\boldsymbol{g}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{y}_t))] \\
&\leq 2\tilde{\mathcal{W}}_2[p_t, q_t] \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} + 2L_t\tilde{\mathcal{W}}_2^2[p_t, q_t]
\end{aligned}
$$

#### 7.2 转换为一阶微分不等式

**推导7.2**：从二阶到一阶。

设 $W_t = \tilde{\mathcal{W}}_2[p_t, q_t]$，则 $W_t^2 = \tilde{\mathcal{W}}_2^2[p_t, q_t]$。

求导：

$$
\frac{d(W_t^2)}{dt} = 2W_t\frac{dW_t}{dt}
$$

因此：

$$
2W_t\frac{dW_t}{dt} \leq 2W_t\sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} + 2L_tW_t^2
$$

除以 $2W_t$（假设 $W_t > 0$）：

$$
\frac{dW_t}{dt} \leq \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} + L_tW_t
$$

为了处理反向时间，我们考虑 $-\frac{dW_t}{dt}$：

$$
-\frac{dW_t}{dt} \leq \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} + L_tW_t
$$

#### 7.3 常数变易法（Variation of Constants）

**推导7.3**：求解线性非齐次微分不等式。

标准形式的微分不等式：

$$
-\frac{dW_t}{dt} \leq F_t + L_tW_t
$$

其中 $F_t = \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]}$。

这是一个一阶线性微分不等式。使用常数变易法，设：

$$
W_t = C_t e^{\int_t^T L_s ds}
$$

代入不等式：

$$
-\frac{dC_t}{dt}e^{\int_t^T L_s ds} + C_t e^{\int_t^T L_s ds} \cdot L_t \leq F_t + L_t C_t e^{\int_t^T L_s ds}
$$

简化：

$$
-\frac{dC_t}{dt}e^{\int_t^T L_s ds} \leq F_t
$$

$$
-\frac{dC_t}{dt} \leq F_t e^{-\int_t^T L_s ds}
$$

#### 7.4 积分求解

**推导7.4**：对时间积分。

对上式从 $0$ 到 $T$ 积分：

$$
\int_0^T \left(-\frac{dC_t}{dt}\right) dt \leq \int_0^T F_t e^{-\int_t^T L_s ds} dt
$$

左边：

$$
\int_0^T \left(-\frac{dC_t}{dt}\right) dt = C_0 - C_T
$$

初始条件：$W_T = 0$（因为两条轨迹从同一点出发），所以：

$$
C_T = W_T e^{-\int_T^T L_s ds} = 0
$$

因此：

$$
C_0 \leq \int_0^T F_t e^{-\int_t^T L_s ds} dt
$$

#### 7.5 得到最终估计

**推导7.5**：恢复原变量。

由 $W_0 = C_0 e^{\int_0^T L_s ds}$：

$$
W_0 = C_0 e^{\int_0^T L_s ds} \leq e^{\int_0^T L_s ds} \int_0^T F_t e^{-\int_t^T L_s ds} dt
$$

改写指数部分：

$$
e^{\int_0^T L_s ds} \cdot e^{-\int_t^T L_s ds} = e^{\int_0^t L_s ds}
$$

定义 $I_t = e^{\int_0^t L_s ds}$，得到：

$$
W_0 \leq \int_0^T I_t F_t dt
$$

即：

$$
\tilde{\mathcal{W}}_2[p_0, q_0] \leq \int_0^T I_t \sqrt{\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} dt
$$

### 8. 从轨迹期望到分布期望

#### 8.1 测度变换

**推导8.1**：期望的等价表达。

注意到 $\boldsymbol{x}_t$ 是 $\boldsymbol{z}$ 的确定性函数，即 $\boldsymbol{x}_t = \boldsymbol{x}_t(\boldsymbol{z})$。

对于任意函数 $h(\boldsymbol{x}_t)$：

$$
\mathbb{E}_{\boldsymbol{z} \sim q}[h(\boldsymbol{x}_t(\boldsymbol{z}))] = \int h(\boldsymbol{x}_t(\boldsymbol{z})) q(\boldsymbol{z}) d\boldsymbol{z}
$$

使用推前测度的定义，$p_t = (\boldsymbol{x}_t)_{\#}q$ 意味着：

$$
\int h(\boldsymbol{x}) p_t(\boldsymbol{x}) d\boldsymbol{x} = \int h(\boldsymbol{x}_t(\boldsymbol{z})) q(\boldsymbol{z}) d\boldsymbol{z}
$$

因此：

$$
\mathbb{E}_{\boldsymbol{z} \sim q}[h(\boldsymbol{x}_t(\boldsymbol{z}))] = \mathbb{E}_{\boldsymbol{x}_t \sim p_t}[h(\boldsymbol{x}_t)]
$$

#### 8.2 应用到不等式

**推导8.2**：改写期望的下标。

应用上述测度变换：

$$
\mathbb{E}_{\boldsymbol{z}}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2] = \mathbb{E}_{\boldsymbol{x}_t \sim p_t}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]
$$

代入之前的不等式：

$$
\tilde{\mathcal{W}}_2[p_0, q_0] \leq \int_0^T I_t \sqrt{\mathbb{E}_{\boldsymbol{x}_t \sim p_t}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]} dt
$$

结合 $\mathcal{W}_2[p_0, q_0] \leq \tilde{\mathcal{W}}_2[p_0, q_0]$，得到：

$$
\mathcal{W}_2[p_0, q_0] \leq \int_0^T I_t \left(\mathbb{E}_{\boldsymbol{x}_t \sim p_t}[\|\boldsymbol{f}_t(\boldsymbol{x}_t) - \boldsymbol{g}_t(\boldsymbol{x}_t)\|^2]\right)^{1/2} dt
$$

这就是式 $\eqref{eq:w-neq-0}$ 的完整推导。

### 9. 推广到不同初始分布

#### 9.1 最优传输耦合

**推导9.1**：构造跨分布的传输方案。

现在考虑两个不同的初始分布 $p_T(\boldsymbol{z}_1)$ 和 $q_T(\boldsymbol{z}_2)$。

设 $\gamma^*(\boldsymbol{z}_1, \boldsymbol{z}_2)$ 是 $p_T$ 和 $q_T$ 之间的最优传输计划，即：

$$
\mathcal{W}_2^2[p_T, q_T] = \int \|\boldsymbol{z}_1 - \boldsymbol{z}_2\|^2 d\gamma^*(\boldsymbol{z}_1, \boldsymbol{z}_2)
$$

从 $\boldsymbol{z}_1 \sim p_T$ 出发运行第一个ODE得到 $\boldsymbol{x}_t(\boldsymbol{z}_1)$。

从 $\boldsymbol{z}_2 \sim q_T$ 出发运行第二个ODE得到 $\boldsymbol{y}_t(\boldsymbol{z}_2)$。

定义时刻 $t$ 的联合分布：

$$
\gamma_t(\boldsymbol{x}, \boldsymbol{y}) = \int \gamma^*(\boldsymbol{z}_1, \boldsymbol{z}_2) \delta(\boldsymbol{x} - \boldsymbol{x}_t(\boldsymbol{z}_1)) \delta(\boldsymbol{y} - \boldsymbol{y}_t(\boldsymbol{z}_2)) d\boldsymbol{z}_1 d\boldsymbol{z}_2
$$

#### 9.2 上界的构造

**推导9.2**：建立新的上界。

类似之前的分析：

$$
\begin{aligned}
\mathcal{W}_2^2[p_t, q_t]
&\leq \int \|\boldsymbol{x} - \boldsymbol{y}\|^2 d\gamma_t(\boldsymbol{x}, \boldsymbol{y}) \\
&= \iint \|\boldsymbol{x}_t(\boldsymbol{z}_1) - \boldsymbol{y}_t(\boldsymbol{z}_2)\|^2 d\gamma^*(\boldsymbol{z}_1, \boldsymbol{z}_2) \\
&= \mathbb{E}_{(\boldsymbol{z}_1, \boldsymbol{z}_2) \sim \gamma^*}[\|\boldsymbol{x}_t - \boldsymbol{y}_t\|^2] \\
&\triangleq \tilde{\mathcal{W}}_2^2[p_t, q_t]
\end{aligned}
$$

#### 9.3 边界条件的变化

**推导9.3**：初始时刻的距离。

当 $t = T$ 时：

$$
\tilde{\mathcal{W}}_2^2[p_T, q_T] = \mathbb{E}_{(\boldsymbol{z}_1, \boldsymbol{z}_2) \sim \gamma^*}[\|\boldsymbol{x}_T - \boldsymbol{y}_T\|^2] = \mathbb{E}_{(\boldsymbol{z}_1, \boldsymbol{z}_2) \sim \gamma^*}[\|\boldsymbol{z}_1 - \boldsymbol{z}_2\|^2]
$$

由于 $\gamma^*$ 是最优传输计划：

$$
\tilde{\mathcal{W}}_2^2[p_T, q_T] = \mathcal{W}_2^2[p_T, q_T]
$$

因此边界条件变为：

$$
C_T = W_T = \mathcal{W}_2[p_T, q_T]
$$

而不是之前的 $C_T = 0$。

#### 9.4 积分时的修正

**推导9.4**：修正积分结果。

回到积分步骤：

$$
C_0 - C_T \leq \int_0^T F_t e^{-\int_t^T L_s ds} dt
$$

现在 $C_T = \mathcal{W}_2[p_T, q_T]$：

$$
C_0 \leq \int_0^T F_t e^{-\int_t^T L_s ds} dt + \mathcal{W}_2[p_T, q_T]
$$

乘以 $e^{\int_0^T L_s ds}$：

$$
W_0 \leq \int_0^T I_t F_t dt + I_T \mathcal{W}_2[p_T, q_T]
$$

其中 $I_T = e^{\int_0^T L_s ds}$。

### 10. 应用到扩散模型

#### 10.1 反向SDE的ODE形式

**推导10.1**：从SDE到ODE。

扩散模型的一般反向SDE（式 $\eqref{eq:sde-reverse-2}$）：

$$
d\boldsymbol{x}_t = \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}(g_t^2 + \sigma_t^2)\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right]dt + \sigma_t d\boldsymbol{w}
$$

当 $\sigma_t = 0$ 时，退化为ODE：

$$
\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)
$$

这是概率流ODE（Probability Flow ODE）。

#### 10.2 得分函数的近似

**推导10.2**：神经网络近似的ODE。

用神经网络 $\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$ 近似 $\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$，得到近似ODE：

$$
\frac{d\boldsymbol{y}_t}{dt} = \boldsymbol{f}_t(\boldsymbol{y}_t) - \frac{1}{2}g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{y}_t, t)
$$

两个ODE的差异为：

$$
\begin{aligned}
&\,\boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \left[\boldsymbol{f}_t(\boldsymbol{x}_t) - \frac{1}{2}g_t^2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right] \\
=&\, -\frac{1}{2}g_t^2\left[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right]
\end{aligned}
$$

#### 10.3 代入不等式

**推导10.3**：得分匹配损失的出现。

将向量场差异代入之前的不等式：

$$
\begin{aligned}
&\,\mathcal{W}_2[p_0, q_0] \\
\leq&\, \int_0^T I_t \left(\mathbb{E}_{\boldsymbol{x}_t \sim p_t}\left[\left\|\frac{1}{2}g_t^2\left[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right]\right\|^2\right]\right)^{1/2} dt \\
&\, + I_T\mathcal{W}_2[p_T, q_T] \\
=&\, \int_0^T I_t \frac{g_t^2}{2} \left(\mathbb{E}_{\boldsymbol{x}_t \sim p_t}\left[\left\|\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right\|^2\right]\right)^{1/2} dt \\
&\, + I_T\mathcal{W}_2[p_T, q_T]
\end{aligned}
$$

这正是文章开头的核心不等式 $\eqref{eq:w-neq}$（取 $g_t^2 I_t/2$ 为新的权重函数）。

### 11. 得分函数的Lipschitz性质

#### 11.1 对数密度梯度的性质

**命题11.1**：对于光滑的概率密度 $p(\boldsymbol{x})$，得分函数 $\nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$ 的Lipschitz性质与密度的Hessian矩阵有关。

**推导11.1**：计算得分函数的Jacobian。

设 $\boldsymbol{s}(\boldsymbol{x}) = \nabla_{\boldsymbol{x}}\log p(\boldsymbol{x})$，计算其Jacobian矩阵：

$$
\begin{aligned}
J_{ij} = \frac{\partial s_i}{\partial x_j}
&= \frac{\partial}{\partial x_j}\left(\frac{\partial \log p}{\partial x_i}\right) \\
&= \frac{\partial}{\partial x_j}\left(\frac{1}{p}\frac{\partial p}{\partial x_i}\right) \\
&= -\frac{1}{p^2}\frac{\partial p}{\partial x_j}\frac{\partial p}{\partial x_i} + \frac{1}{p}\frac{\partial^2 p}{\partial x_i \partial x_j}
\end{aligned}
$$

用得分函数表示：

$$
J_{ij} = -s_i(\boldsymbol{x})s_j(\boldsymbol{x}) + \frac{\partial s_i}{\partial x_j}
$$

这是一个非线性表达式，说明得分函数的Lipschitz性质依赖于密度本身。

#### 11.2 强log-凹分布的情况

**定义11.1（强log-凹）**：如果密度 $p(\boldsymbol{x})$ 满足 $-\log p(\boldsymbol{x})$ 是强凸的，即：

$$
-\nabla^2 \log p(\boldsymbol{x}) \succeq m I
$$

对某个 $m > 0$，则称 $p$ 是 $m$-强log-凹的。

**推导11.2**：强log-凹分布的单侧Lipschitz性。

对于强log-凹分布：

$$
\nabla^2 \log p(\boldsymbol{x}) \preceq -m I
$$

得分函数的Jacobian：

$$
\nabla \boldsymbol{s}(\boldsymbol{x}) = \nabla^2 \log p(\boldsymbol{x}) \preceq -m I
$$

因此，对于任意 $\boldsymbol{x}, \boldsymbol{y}$：

$$
\begin{aligned}
&\,(\boldsymbol{s}(\boldsymbol{x}) - \boldsymbol{s}(\boldsymbol{y})) \cdot (\boldsymbol{x} - \boldsymbol{y}) \\
=&\, \int_0^1 (\boldsymbol{x} - \boldsymbol{y})^T \nabla\boldsymbol{s}(t\boldsymbol{x} + (1-t)\boldsymbol{y}) (\boldsymbol{x} - \boldsymbol{y}) dt \\
\leq&\, -m\|\boldsymbol{x} - \boldsymbol{y}\|^2
\end{aligned}
$$

这表明强log-凹分布的得分函数满足单侧Lipschitz条件（常数为 $-m < 0$）。

#### 11.3 高斯分布的例子

**例11.1（高斯分布）**：对于高斯分布 $p(\boldsymbol{x}) = \mathcal{N}(\boldsymbol{\mu}, \Sigma)$：

$$
\log p(\boldsymbol{x}) = -\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu}) + \text{const}
$$

得分函数：

$$
\boldsymbol{s}(\boldsymbol{x}) = \nabla_{\boldsymbol{x}}\log p(\boldsymbol{x}) = -\Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu})
$$

这是一个线性函数，其Jacobian为：

$$
\nabla \boldsymbol{s}(\boldsymbol{x}) = -\Sigma^{-1}
$$

对于单位协方差 $\Sigma = I$：

$$
\nabla \boldsymbol{s}(\boldsymbol{x}) = -I
$$

因此满足单侧Lipschitz条件（常数 $L = -1$）。

### 12. 条件得分匹配

#### 12.1 条件期望的性质

**推导12.1**：从无条件得分到条件得分。

利用条件期望，得分函数可以写为：

$$
\begin{aligned}
\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)
&= \nabla_{\boldsymbol{x}_t}\log \int p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0 \\
&= \frac{\int \nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)} \\
&= \frac{\int p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)p_0(\boldsymbol{x}_0) d\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)}
\end{aligned}
$$

#### 12.2 后验分布

**推导12.2**：利用贝叶斯公式。

由贝叶斯公式：

$$
p_t(\boldsymbol{x}_0|\boldsymbol{x}_t) = \frac{p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)p_0(\boldsymbol{x}_0)}{p_t(\boldsymbol{x}_t)}
$$

因此：

$$
\begin{aligned}
\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)
&= \int \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) p_t(\boldsymbol{x}_0|\boldsymbol{x}_t) d\boldsymbol{x}_0 \\
&= \mathbb{E}_{\boldsymbol{x}_0 \sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)]
\end{aligned}
$$

这就是条件期望的形式。

#### 12.3 Jensen不等式的应用

**推导12.3**：从条件得分匹配到无条件得分匹配。

考虑损失函数：

$$
\mathcal{L}_1 = \mathbb{E}_{\boldsymbol{x}_t}[\|\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\|^2]
$$

利用条件期望：

$$
\begin{aligned}
\mathcal{L}_1
&= \mathbb{E}_{\boldsymbol{x}_t}\left[\left\|\mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)] - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right\|^2\right]
\end{aligned}
$$

条件得分匹配损失：

$$
\mathcal{L}_2 = \mathbb{E}_{\boldsymbol{x}_t}\mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\|\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\|^2]
$$

由Jensen不等式（$\mathbb{E}[X]$ 的平方小于等于 $\mathbb{E}[X^2]$）：

$$
\mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\|\boldsymbol{g}(\boldsymbol{x}_0) - \boldsymbol{c}\|^2] \geq \|\mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\boldsymbol{g}(\boldsymbol{x}_0)] - \boldsymbol{c}\|^2
$$

取 $\boldsymbol{g}(\boldsymbol{x}_0) = \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$，$\boldsymbol{c} = \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$：

$$
\mathcal{L}_1 \leq \mathcal{L}_2
$$

因此，优化条件得分匹配也会优化无条件得分匹配。

### 13. 总结与展望

通过以上详尽的推导，我们完成了从Wasserstein距离的基本定义到扩散模型得分匹配损失上界的完整证明链条。主要技术要点包括：

1. **Wasserstein距离的下确界性质**：允许我们用任意可行的传输方案构造上界。

2. **ODE轨迹的确定性耦合**：提供了自然的传输方案。

3. **Cauchy-Schwarz不等式**：关键的放缩工具。

4. **单侧Lipschitz条件**：比标准Lipschitz更弱但足够的条件。

5. **常数变易法**：求解微分不等式的经典方法。

6. **测度论的等价变换**：连接轨迹期望与分布期望。

这些技术的综合运用，最终建立了得分匹配损失作为Wasserstein距离上界的理论基础，为理解扩散模型的优化目标提供了新的视角。

