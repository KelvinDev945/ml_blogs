---
title: 生成扩散模型漫谈（十九）：作为扩散ODE的GAN
slug: 生成扩散模型漫谈十九作为扩散ode的gan
date: 2023-06-24
tags: 优化, GAN, 扩散, 生成模型, attention
status: pending
---

# 生成扩散模型漫谈（十九）：作为扩散ODE的GAN

**原文链接**: [https://spaces.ac.cn/archives/9662](https://spaces.ac.cn/archives/9662)

**发布日期**: 

---

在文章[《生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配》](/archives/9467)中，我们推导了Wasserstein距离与扩散模型得分匹配损失之间的一个不等式，表明扩散模型的优化目标与WGAN的优化目标在某种程度上具有相似性。而在本文，我们将探讨[《MonoFlow: Rethinking Divergence GANs via the Perspective of Wasserstein Gradient Flows》](https://papers.cool/arxiv/2302.01075)中的研究成果，它进一步展示了GAN与扩散模型之间的联系：GAN实际上可以被视为在另一个时间维度上的扩散ODE！

这些发现表明，尽管GAN和扩散模型表面上是两种截然不同的生成式模型，但它们实际上存在许多相似之处，并在许多方面可以相互借鉴和参考。

## 思路简介 #

我们知道，GAN所训练的生成器是从噪声$\boldsymbol{z}$到真实样本的一个直接的确定性变换$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$，而扩散模型的显著特点是“渐进式生成”，它的生成过程对应于从一系列渐变的分布$p_0(\boldsymbol{x}_0),p_1(\boldsymbol{x}_1),\cdots,p_T(\boldsymbol{x}_T)$中采样（注：在前面十几篇文章中，$\boldsymbol{x}_T$是噪声，$\boldsymbol{x}_0$是目标样本，采样过程是$\boldsymbol{x}_T\to \boldsymbol{x}_0$，但为了便于下面的表述，这里反过来改为$\boldsymbol{x}_0\to \boldsymbol{x}_T$）。看上去确实找不到多少相同之处，那怎么才能将两者联系起来呢？

很明显，如果想要从扩散模型的视角理解GAN，那么就要想办法构造出一系列渐变的分布出来。生成器$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$本身就是一个一步到位的变换，不存在渐变，然而我们知道模型的优化是渐变的，可否用参数$\boldsymbol{\theta}$的历史轨迹$\boldsymbol{\theta}_t$来构建这一系列渐变分布呢？具体来说，假设生成器初始化为$\boldsymbol{\theta}_0$，经过$T$步对抗训练后得到最优参数$\boldsymbol{\theta}_T$，训练过程的中间参数为$\boldsymbol{\theta}_1,\boldsymbol{\theta}_2,\cdots,\boldsymbol{\theta}_{T-1}$，那么我们定义$\boldsymbol{x}_t = \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})$，不就定义了一系列渐变的$\boldsymbol{x}_0,\boldsymbol{x}_1,\cdots,\boldsymbol{x}_T$，从而也就定义了渐变的分布$p_0(\boldsymbol{x}_0),p_1(\boldsymbol{x}_1),\cdots,p_T(\boldsymbol{x}_T)$了？

如果这个思路可行的话，那么GAN就可以诠释为梯度下降的（虚拟）时间维度上的扩散模型！下面我们就沿着这个思路进行探索。

## 梯度之流 #

首先，我们需要重温上一篇文章[《梯度流：探索通向最小值之路》](/archives/9660)关于Wasserstein梯度流的结果：它指出方程  
\begin{equation}\frac{\partial q_t(\boldsymbol{x})}{\partial t} = - \nabla_{\boldsymbol{x}}\cdot\big(q_t(\boldsymbol{x})\nabla_{\boldsymbol{x}}\log r_t(\boldsymbol{x})\big)\label{eq:w-flow}\end{equation}  
在最小化$p(\boldsymbol{x})$和$q_t(\boldsymbol{x})$的KL散度，即$\lim\limits_{t\to\infty} q_t(\boldsymbol{x}) = p(\boldsymbol{x})$，这里$r_t(\boldsymbol{x})=\frac{p(\boldsymbol{x})}{q_t(\boldsymbol{x})}$。如果$p(\boldsymbol{x})$代表真实样本的分布，那么如果能实现从$q_t(\boldsymbol{x})$采样的话，那么逐渐推到$t\to\infty$时，就可以实现从$p(\boldsymbol{x})$采样了。根据[《测试函数法推导连续性方程和Fokker-Planck方程》](/archives/9461)，从$q_t(\boldsymbol{x})$采样可以通过下述ODE实现：  
\begin{equation}\frac{d\boldsymbol{x}}{dt} = \nabla_{\boldsymbol{x}}\log r_t(\boldsymbol{x})\label{eq:ode-core}\end{equation}  
然而，上式中的$r_t(\boldsymbol{x})$是未知的，所以我们还无法通过上式进行采样，需要先想办法估算$r_t(\boldsymbol{x})$。

## 判别估计 #

这时候登场的是GAN的判别器。以最早的Vanilla GAN为例，它的训练目标是  
\begin{equation}\max_D\, \mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[\log \sigma(D(\boldsymbol{x}))] + \mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}[\log (1 - \sigma(D(\boldsymbol{x})))]\label{eq:gan-d}\end{equation}  
这里的$D$是判别器，$\sigma(t)=1/(1+e^{-t})$是sigmoid函数，$p(\boldsymbol{x})$是真样本的分布，$q(\boldsymbol{x})$是假样本的分布。可以证明（不清楚的读者可以参考[《RSGAN：对抗模型中的“图灵测试”思想》](/archives/6110)中的“补充证明”一节），上式中判别器$D$的理论最优解是  
\begin{equation}D(\boldsymbol{x}) = \log \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\end{equation}  
更一般化的f-GAN（参考[《f-GAN简介：GAN模型的生产车间》](/archives/6016)、[《Designing GANs：又一个GAN生产车间》](/archives/7210)）结果会稍有不同，但可以证明的是它们判别器的理论最优解都是$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$的函数。也就是说，只要我们可以实现从$p(\boldsymbol{x})$和$q_t(\boldsymbol{x})$中采样，那么通过GAN的判别器训练$\eqref{eq:gan-d}$就可以估算出$r_t(\boldsymbol{x})=\frac{p(\boldsymbol{x})}{q_t(\boldsymbol{x})}$出来。

## 向前一步 #

这时候可能有读者疑惑：这不就进入“鸡生蛋、蛋生鸡”的循环论证了吗？我们估算$r_t(\boldsymbol{x})$不就是为了利用式$\eqref{eq:ode-core}$实现从$q_t(\boldsymbol{x})$中采样吗？现在你又假设能从$q_t(\boldsymbol{x})$采样才来估算$r_t(\boldsymbol{x})$？不着急，经典的一笔就要来了。

假设我们有生成器$\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})$，它的采样生成结果就等于从$q_t(\boldsymbol{x})$采样的结果，即  
\begin{equation}\big\\{\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})\big|\,\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\big\\}\quad = \quad\big\\{\boldsymbol{x}_t\big|\,\boldsymbol{x}_t\sim q_t(\boldsymbol{x})\big\\}\end{equation}  
那么现在我们就可以利用它和式$\eqref{eq:gan-d}$来估算$r_t(\boldsymbol{x})$。注意这只是$t$时刻的$r_t(\boldsymbol{x})$，其他时刻的$r_t(\boldsymbol{x})$我们并不知道，所以无法直接通过式$\eqref{eq:ode-core}$完成最终的采样过程，但是我们可以往前推一小步：  
\begin{equation}\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \epsilon \nabla_{\boldsymbol{x}_t}\log r_t(\boldsymbol{x}_t) = \boldsymbol{x}_t + \epsilon \nabla_{\boldsymbol{x}_t} D(\boldsymbol{x}_t)\label{eq:forward}\end{equation}  
这里的$\epsilon$是一个很小的正数，代表步长。那么，现在我们就有了下一步采样的结果，我们希望它继续能等价于下一步的生成器的采样结果，即  
\begin{equation}\begin{aligned}  
\big\\{\boldsymbol{g}_{\boldsymbol{\theta}_{t+1}}(\boldsymbol{z})\big|\,\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\big\\}\quad =& \quad\big\\{\boldsymbol{x}_{t+1}\big|\,\boldsymbol{x}_{t+1}\sim q_{t+1}(\boldsymbol{x})\big\\} \\\\[5pt]  
\quad =& \quad\big\\{\boldsymbol{x}_{t+1}\big|\,\boldsymbol{x}_t + \epsilon \nabla_{\boldsymbol{x}_t} D(\boldsymbol{x}_t),\boldsymbol{x}_t\sim q_t(\boldsymbol{x})\big\\}  
\end{aligned}\end{equation}  
换句话说，我们想要**将扩散模型中样本的运动转化为生成器参数的运动** ！为了达到这个目标，我们通过如下损失去求$\boldsymbol{\theta}_{t+1}$：  
\begin{equation}\boldsymbol{\theta}_{t+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\Big[\big\Vert \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) - \epsilon \nabla_{\boldsymbol{g}}D(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}))\big\Vert^2\Big]\label{eq:gan-g0}\end{equation}  
也就是说，拿$\boldsymbol{x}_t = \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})$往前迭代一步得到$\boldsymbol{x}_{t+1}$，然后希望新的$\boldsymbol{g}_{\boldsymbol{\theta}_{t+1}}(\boldsymbol{z})$能尽量等于$\boldsymbol{x}_{t+1}$。完成这一轮后，再用$\boldsymbol{\theta}_{t+1}$替代原本的$\boldsymbol{\theta}_t$开始新一轮的迭代，也就是式$\eqref{eq:gan-d}$和式$\eqref{eq:gan-g0}$交替执行，是不是就有GAN的味道了？

## 点睛之笔 #

如果这还不够，我们还可以继续完善一下，将它变得跟GAN更加一致。注意到式$\eqref{eq:gan-g0}$的被期望函数的梯度是：  
\begin{equation}\begin{aligned}  
&\,\nabla_{\boldsymbol{\theta}}\Vert \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) - \epsilon \nabla_{\boldsymbol{g}}D(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}))\Vert^2 \\\  
=&\,2\big\langle\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) - \epsilon \nabla_{\boldsymbol{g}}D(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})), \nabla_{\boldsymbol{\theta}}\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) \big\rangle \\\  
\end{aligned}\end{equation}  
代入当前值$\boldsymbol{\theta}=\boldsymbol{\theta}_t$，那么结果是  
\begin{equation}-2\epsilon\big\langle \nabla_{\boldsymbol{g}}D(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})), \nabla_{\boldsymbol{\theta}_t}\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) \big\rangle = -2\epsilon\nabla_{\boldsymbol{\theta}_t}D(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}))\end{equation}  
也就是说，如果用基于梯度的优化器只优化一步的话，那么以式$\eqref{eq:gan-g0}$为损失函数，跟以下式为损失函数，结果是等价的（因为梯度只差一个常数倍）：  
\begin{equation}\boldsymbol{\theta}_{t+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}[-D(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))]\label{eq:gan-g}\end{equation}  
这便是常见的生成器损失之一。式$\eqref{eq:gan-d}$和式$\eqref{eq:gan-g}$交替训练，就是一个常见的GAN变体。

特别地，原论文还证明了生成器的损失函数可以一般化为  
\begin{equation}\boldsymbol{\theta}_{t+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}[-h(D(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})))]\end{equation}  
其中$h(\cdot)$是任意单调递增函数，它也对应于Wasserstein梯度流$\eqref{eq:w-flow}$中的$\log r_t(\boldsymbol{x})$可以换成$h(\log r_t(\boldsymbol{x}))$，这应该也是MonoFlow一词的来源（Monotonically increasing function + Wasserstein flow）。这个证明过程就不展开了，大家自行看原论文就好。

## 意义思考 #

总的来说，将GAN理解为扩散模型的思路是  
$$\require{AMScd}\begin{CD}  
\cdots @>\quad>> \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) @> 式\eqref{eq:gan-d} >> r_t(\boldsymbol{x}) @> 式\eqref{eq:forward}>> \boldsymbol{x}_{t+1} @> 式\eqref{eq:gan-g0}>> \boldsymbol{g}_{\boldsymbol{\theta}_{t+1}}(\boldsymbol{z})@>\quad>>\cdots  
\end{CD}$$  
其中，核心的式子是$\eqref{eq:forward}$，它源于Wasserstein梯度流的式$\eqref{eq:w-flow}$和式$\eqref{eq:ode-core}$，这部分我们在上一篇文章[《梯度流：探索通向最小值之路》](/archives/9660)讨论过了。

可能有读者想问：这个视角看上去并没有得到比GAN更多的东西，为什么要费这番大功夫去重新理解GAN呢？首先，在笔者看来，从扩散模型角度理解GAN，或者说将扩散模型和GAN统一起来，它本身就是一件很有趣、很好玩的事情，并不一定需要发挥什么实际作用，有趣、好玩就是它最大的意义。

其次，如同作者在原论文所说，已有的GAN的推导过程跟它实际的训练过程是不一致的，而本文所讨论的扩散视角，则是跟训练过程是一致的。也就是说，以训练过程为标准的话，GAN已有的推导过程是错的，本文的扩散视角才是对的。怎么理解这一点呢？以前面提到的GAN为例，判别器和生成器的目标分别是：  
\begin{gather}\max_D\, \mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[\log \sigma(D(\boldsymbol{x}))] + \mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}[\log (1 - \sigma(D(\boldsymbol{x})))] \\\  
\min_q\mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}[-D(\boldsymbol{x})]  
\end{gather}  
通常的证明方式是，证明$D$的最优解是$\log\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$，然后代入生成器的损失函数，发现它在最小化$q(\boldsymbol{x}),p(\boldsymbol{x})$的KL散度，所以最优解是$q(\boldsymbol{x})=p(\boldsymbol{x})$。但是，这样的证明对应的训练方式应该是先针对任意的$q(\boldsymbol{x})$，将$\max\limits_D$这一步都求解出来（求出的$D$应该是$q(x)$的函数，或者说应该是生成器参数$\boldsymbol{\theta}$的函数），然后再去执行$\min\limits_q$这一步，而不是实际上用的交替训练。而基于扩散模型的理解，它在设计上就是交替的，所以它跟训练过程更加一致。

总的来说，从扩散模型的角度来理解GAN，不单单是理解GAN的一种新视角，而且还是一种更贴近训练过程的视角。比如，我们可以解释为什么GAN的生成器不能训练太多步，因为只有单步优化时，式$\eqref{eq:gan-g}$和式$\eqref{eq:gan-g0}$才等价，如果GAN要进行更多步的优化，那么应该使用式$\eqref{eq:gan-g0}$为损失函数。事实上，式$\eqref{eq:gan-g0}$就相当于笔者之前在[《用变分推断统一理解生成模型（VAE、GAN、AAE、ALI）》](/archives/5716)所提出的$KL\left(q(x)\Vert q^{o}(x)\right)$项，它保证了生成器的“传承”而不仅仅是“创新”。

## 文章小结 #

本文介绍了MonoFlow，它展示了我们可以将GAN理解为在另一个时间维度上的扩散ODE，从而建立了一种基于扩散模型理解GAN的新视角。特别地，这是一种比GAN的常规推导更加贴近训练过程的视角。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9662>_

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

苏剑林. (Jun. 24, 2023). 《生成扩散模型漫谈（十九）：作为扩散ODE的GAN 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9662>

@online{kexuefm-9662,  
title={生成扩散模型漫谈（十九）：作为扩散ODE的GAN},  
author={苏剑林},  
year={2023},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/9662}},  
} 


---

## 公式推导与注释

### 1. Wasserstein梯度流与GAN的连接

#### 1.1 Wasserstein梯度流回顾

**定理1.1**（Wasserstein梯度流）：考虑概率密度$q_t(\boldsymbol{x})$随时间$t$的演化。为了最小化KL散度$D_{KL}(q_t\|p)$，最速下降流（steepest descent flow）由连续性方程描述：

$$
\frac{\partial q_t(\boldsymbol{x})}{\partial t} = -\nabla_{\boldsymbol{x}} \cdot \left(q_t(\boldsymbol{x}) \nabla_{\boldsymbol{x}} \log r_t(\boldsymbol{x})\right) \tag{1}
$$

其中$r_t(\boldsymbol{x}) = \frac{p(\boldsymbol{x})}{q_t(\boldsymbol{x})}$是密度比。

**推导**：KL散度的泛函导数为：

$$
\frac{\delta}{\delta q} D_{KL}(q\|p) = \log q(\boldsymbol{x}) - \log p(\boldsymbol{x}) + 1 = -\log r(\boldsymbol{x}) + 1 \tag{2}
$$

Wasserstein空间中的梯度流对应于选择速度场$\boldsymbol{v}(\boldsymbol{x})$使得：

$$
\boldsymbol{v}(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}} \frac{\delta D_{KL}}{\delta q} = \nabla_{\boldsymbol{x}} \log r(\boldsymbol{x}) \tag{3}
$$

代入连续性方程$\frac{\partial q}{\partial t} + \nabla \cdot (q\boldsymbol{v}) = 0$即得(1)。

#### 1.2 从连续性方程到ODE

**引理1.2**（特征线方法）：方程(1)等价于以下ODE的推送测度（pushforward measure）：

$$
\frac{d\boldsymbol{x}}{dt} = \nabla_{\boldsymbol{x}} \log r_t(\boldsymbol{x}) \tag{4}
$$

**证明**：设$\boldsymbol{x}_t$满足(4)，初始分布$\boldsymbol{x}_0 \sim q_0$。则$q_t$是$\boldsymbol{x}_t$的边缘分布。利用Itô引理的确定性版本（传输公式）：

$$
\frac{d}{dt}\phi(\boldsymbol{x}_t) = \langle \nabla \phi, \frac{d\boldsymbol{x}_t}{dt} \rangle = \langle \nabla \phi, \nabla \log r_t \rangle \tag{5}
$$

对任意测试函数$\phi$，积分得：

$$
\frac{d}{dt}\int \phi(\boldsymbol{x}) q_t(\boldsymbol{x}) d\boldsymbol{x} = \int \langle \nabla \phi, \nabla \log r_t \rangle q_t d\boldsymbol{x} \tag{6}
$$

分部积分（假设边界项消失）：

$$
= -\int \phi(\boldsymbol{x}) \nabla \cdot (q_t \nabla \log r_t) d\boldsymbol{x} \tag{7}
$$

这正是(1)的弱形式。□

### 2. GAN判别器与密度比估计

#### 2.1 Vanilla GAN的最优判别器

**GAN的判别器目标**：

$$
\max_D \mathbb{E}_{\boldsymbol{x} \sim p}\left[\log \sigma(D(\boldsymbol{x}))\right] + \mathbb{E}_{\boldsymbol{x} \sim q}\left[\log(1 - \sigma(D(\boldsymbol{x})))\right] \tag{8}
$$

其中$\sigma(z) = \frac{1}{1 + e^{-z}}$是sigmoid函数。

**定理2.1**（最优判别器）：(8)的最优解为：

$$
D^*(\boldsymbol{x}) = \log \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})} = \log r(\boldsymbol{x}) \tag{9}
$$

**证明**：对每个$\boldsymbol{x}$，目标函数关于$D(\boldsymbol{x})$的泛函为：

$$
L(D(\boldsymbol{x})) = p(\boldsymbol{x}) \log \sigma(D(\boldsymbol{x})) + q(\boldsymbol{x}) \log(1 - \sigma(D(\boldsymbol{x}))) \tag{10}
$$

求导并令其为零：

$$
\frac{\partial L}{\partial D} = p(\boldsymbol{x}) \sigma'(D) / \sigma(D) - q(\boldsymbol{x}) \sigma'(D) / (1 - \sigma(D)) = 0 \tag{11}
$$

利用$\sigma'(z) = \sigma(z)(1-\sigma(z))$：

$$
p(\boldsymbol{x})(1 - \sigma(D^*)) = q(\boldsymbol{x}) \sigma(D^*) \tag{12}
$$

解得：

$$
\sigma(D^*(\boldsymbol{x})) = \frac{p(\boldsymbol{x})}{p(\boldsymbol{x}) + q(\boldsymbol{x})} \tag{13}
$$

应用$\sigma(z) = \frac{1}{1 + e^{-z}}$的逆函数：

$$
D^*(\boldsymbol{x}) = \log \frac{\sigma(D^*)}{1 - \sigma(D^*)} = \log \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})} \tag{14}
$$

□

#### 2.2 f-GAN的一般化

**f-divergence**：对于凸函数$f$，定义：

$$
D_f(p\|q) = \mathbb{E}_{\boldsymbol{x} \sim q}\left[f\left(\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right)\right] \tag{15}
$$

**f-GAN目标**：

$$
\max_D \mathbb{E}_{\boldsymbol{x} \sim p}[g_f(D(\boldsymbol{x}))] + \mathbb{E}_{\boldsymbol{x} \sim q}[h_f(D(\boldsymbol{x}))] \tag{16}
$$

其中$g_f$, $h_f$是与$f$共轭的函数。

**定理2.2**：f-GAN的最优判别器满足：

$$
D^*(\boldsymbol{x}) = f'\left(\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right) \tag{17}
$$

即$D^*$是密度比$r(\boldsymbol{x})$的单调变换。

### 3. 从样本空间到参数空间的映射

#### 3.1 生成器参数化的分布族

**设定**：生成器$\boldsymbol{g}_{\boldsymbol{\theta}}: \mathbb{R}^d \to \mathbb{R}^n$将噪声$\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$映射到样本空间。定义诱导分布：

$$
q_{\boldsymbol{\theta}}(\boldsymbol{x}) = \int \delta(\boldsymbol{x} - \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})) \mathcal{N}(\boldsymbol{z}; \mathbf{0}, \boldsymbol{I}) d\boldsymbol{z} \tag{18}
$$

或等价地：

$$
\boldsymbol{x} = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}), \quad \boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I}) \Rightarrow \boldsymbol{x} \sim q_{\boldsymbol{\theta}} \tag{19}
$$

#### 3.2 参数空间的时间演化

**核心思想**：将训练过程的参数轨迹$\boldsymbol{\theta}_t$视为时间$t$的函数，对应分布族$\{q_t\}_{t \geq 0}$，其中：

$$
q_t(\boldsymbol{x}) := q_{\boldsymbol{\theta}_t}(\boldsymbol{x}) \tag{20}
$$

**样本空间的演化**：对于固定的$\boldsymbol{z}$，样本轨迹为：

$$
\boldsymbol{x}_t = \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) \tag{21}
$$

求时间导数：

$$
\frac{d\boldsymbol{x}_t}{dt} = \frac{\partial \boldsymbol{g}_{\boldsymbol{\theta}}}{\partial \boldsymbol{\theta}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t} \cdot \frac{d\boldsymbol{\theta}_t}{dt} = \mathbf{J}_{\boldsymbol{g}}(\boldsymbol{z}, \boldsymbol{\theta}_t) \dot{\boldsymbol{\theta}}_t \tag{22}
$$

其中$\mathbf{J}_{\boldsymbol{g}} \in \mathbb{R}^{n \times m}$（$m$是参数维度）是Jacobian矩阵。

#### 3.3 匹配Wasserstein流

**目标**：希望样本空间的演化(22)匹配Wasserstein流(4)：

$$
\mathbf{J}_{\boldsymbol{g}}(\boldsymbol{z}, \boldsymbol{\theta}_t) \dot{\boldsymbol{\theta}}_t = \nabla_{\boldsymbol{x}_t} \log r_t(\boldsymbol{x}_t) \tag{23}
$$

**离散化**：用欧拉法离散化，步长$\epsilon$：

$$
\boldsymbol{x}_{t+1} \approx \boldsymbol{x}_t + \epsilon \nabla_{\boldsymbol{x}_t} D_t(\boldsymbol{x}_t) \tag{24}
$$

其中$D_t$是在时刻$t$训练的判别器（近似$\log r_t$）。

**参数更新目标**：希望新参数$\boldsymbol{\theta}_{t+1}$满足：

$$
\boldsymbol{g}_{\boldsymbol{\theta}_{t+1}}(\boldsymbol{z}) \approx \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) + \epsilon \nabla_{\boldsymbol{g}} D_t(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})) \tag{25}
$$

这导致以下优化问题：

$$
\boldsymbol{\theta}_{t+1} = \arg\min_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})}\left[\left\|\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) - \epsilon \nabla_{\boldsymbol{g}} D_t(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}))\right\|^2\right] \tag{26}
$$

### 4. 从优化视角的等价性

#### 4.1 单步梯度下降的等价性

**引理4.1**：如果用梯度下降法单步优化(26)，等价于优化：

$$
\min_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{z} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})}\left[-D_t(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))\right] \tag{27}
$$

**证明**：计算(26)的梯度，在$\boldsymbol{\theta} = \boldsymbol{\theta}_t$处：

$$
\begin{aligned}
&\nabla_{\boldsymbol{\theta}}\left\|\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) - \epsilon \nabla_{\boldsymbol{g}} D_t(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}))\right\|^2\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t} \\
&= 2(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}) - \epsilon \nabla_{\boldsymbol{g}} D_t)^T \mathbf{J}_{\boldsymbol{g}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t} \\
&= -2\epsilon (\nabla_{\boldsymbol{g}} D_t)^T \mathbf{J}_{\boldsymbol{g}}(\boldsymbol{z}, \boldsymbol{\theta}_t) \\
&= -2\epsilon \nabla_{\boldsymbol{\theta}} D_t(\boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z}))
\end{aligned} \tag{28}
$$

因此，(26)和(27)的梯度只差常数倍$-2\epsilon$，单步梯度下降等价。□

**推论**：标准GAN生成器损失$\min_G \mathbb{E}[-D(G(\boldsymbol{z}))]$是MonoFlow离散化的一阶近似。

#### 4.2 MonoFlow算法框架

**算法1**（MonoFlow交替训练）：

```
初始化：生成器参数 θ_0，判别器 D_0
对 t = 0, 1, 2, ... :
    # 判别器步骤
    采样 {x_i ~ p}_i=1^n (真实数据)
    采样 {z_i ~ N(0,I)}_i=1^n, 计算 {g_θ_t(z_i)}_i=1^n
    更新判别器：
        D_{t+1} = argmax_D [ E_p[log σ(D(x))] + E_{g_θ_t(z)}[log(1-σ(D(x)))] ]

    # 生成器步骤（MonoFlow形式）
    采样 {z_i ~ N(0,I)}_i=1^m
    更新生成器：
        θ_{t+1} = argmin_θ E_z[ ||g_θ(z) - g_θ_t(z) - ε∇_g D_{t+1}(g_θ_t(z))||^2 ]

    # 或等价地（标准GAN形式，单步优化时）
    θ_{t+1} = argmin_θ E_z[ -D_{t+1}(g_θ(z)) ]
```

### 5. 单调变换的一般化

#### 5.1 MonoFlow名称的由来

**定理5.1**（单调函数变换）：Wasserstein梯度流(1)可推广为：

$$
\frac{\partial q_t}{\partial t} = -\nabla \cdot \left(q_t \nabla h(\log r_t)\right) \tag{29}
$$

其中$h: \mathbb{R} \to \mathbb{R}$是任意单调递增函数。

**证明思路**：梯度流本质是沿着KL散度减小最快的方向。引入重参数化$\tilde{r}_t = h(r_t)$，由于$h$单调，$D_{KL}(q_t\|p)$减小等价于$D_{\tilde{KL}}(q_t\|p)$减小（基于$\tilde{r}_t$）。

**对应的生成器损失**：

$$
\min_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{z}}\left[-h(D_t(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})))\right] \tag{30}
$$

**常见选择**：
- $h(z) = z$：标准GAN，$-D(G(\boldsymbol{z}))$
- $h(z) = -\log(1 + e^{-z})$：Non-saturating GAN，$-\log \sigma(D(G(\boldsymbol{z})))$
- $h(z) = -e^{-z}$：某些f-GAN变体

### 6. 理论意义与实践洞察

#### 6.1 训练一致性

**经典GAN理论的不足**：通常证明路径是：

$$
\text{固定}\ q \to \text{求最优}\ D^* = \log r \to \text{代入生成器损失} \to \min_q D_{KL}(q\|p)
$$

这隐含假设每步都求解$\max_D$到最优，但实际训练是交替单步优化。

**MonoFlow的优势**：

$$
\text{当前}\ q_t \to \text{训练}\ D_t \approx \log r_t \to \text{Wasserstein流一步} \to q_{t+1}
$$

这与实际训练过程一致：判别器不需要完全最优，只需近似当前密度比。

#### 6.2 生成器多步优化的问题

**定理6.1**：若对固定$D_t$，生成器优化$k > 1$步，则偏离Wasserstein流。

**证明**：记$\boldsymbol{\theta}_t^{(0)} = \boldsymbol{\theta}_t$，进行$k$步梯度下降：

$$
\boldsymbol{\theta}_t^{(j+1)} = \boldsymbol{\theta}_t^{(j)} - \eta \nabla_{\boldsymbol{\theta}} \mathbb{E}[-D_t(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))]\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}_t^{(j)}} \tag{31}
$$

当$j > 0$时，$D_t$已经不再是$q_{\boldsymbol{\theta}_t^{(j)}}$的准确密度比估计，因此(31)不对应Wasserstein流。

**实践建议**：
- 判别器多步优化：可以，因为每次更新都改进密度比估计
- 生成器单步优化：推荐，保持与理论一致
- 若要多步优化生成器：使用MonoFlow损失(26)而非标准GAN损失(27)

#### 6.3 与传承和创新的联系

**MonoFlow损失(26)的分解**：

$$
\begin{aligned}
&\min_{\boldsymbol{\theta}} \mathbb{E}\left[\|\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_t}(\boldsymbol{z})\|^2\right] \\
&\quad + \min_{\boldsymbol{\theta}} \mathbb{E}\left[\|\epsilon \nabla_{\boldsymbol{g}} D_t\|^2\right] - 2\epsilon \mathbb{E}\left[\langle \boldsymbol{g}_{\boldsymbol{\theta}} - \boldsymbol{g}_{\boldsymbol{\theta}_t}, \nabla_{\boldsymbol{g}} D_t \rangle\right]
\end{aligned} \tag{32}
$$

第一项：$D_{KL}(q_{\boldsymbol{\theta}}\|q_{\boldsymbol{\theta}_t})$，**传承**——保持与前一步的相似性
第三项：$-\mathbb{E}[D_t(\boldsymbol{g}_{\boldsymbol{\theta}})]$，**创新**——提升判别器评分

这与VAE/AAE中的正则化项类似（参考苏剑林博客）。

### 7. 总结与开放问题

#### 7.1 核心结论

1. **GAN = 参数空间扩散ODE**：通过$\boldsymbol{\theta}_t$的演化诱导$q_t$的Wasserstein流
2. **判别器 = 密度比估计器**：$D_t(\boldsymbol{x}) \approx \log r_t(\boldsymbol{x})$
3. **交替训练的合理性**：单步优化保证与Wasserstein流一致

#### 7.2 理论贡献

| 视角 | GAN经典理论 | MonoFlow理论 |
|------|------------|--------------|
| 判别器角色 | 完全最优（理想） | 当前时刻近似（现实） |
| 生成器更新 | 最小化KL散度（全局） | Wasserstein流一步（局部） |
| 训练一致性 | ✗（理论假设与实践不符） | ✓（严格对应交替训练） |

#### 7.3 未来方向

1. **高阶数值方法**：能否用Runge-Kutta等高阶方法改进采样质量？
2. **自适应步长**：$\epsilon_t$应该如何随训练动态调整？
3. **稳定性分析**：什么条件下保证$\lim_{t\to\infty} q_t = p$？
4. **扩展到条件生成**：如何在条件GAN中应用MonoFlow框架？

---

**最终总结**：MonoFlow为GAN提供了基于扩散ODE的严格数学基础，澄清了判别器和生成器交替训练的理论正当性。这一视角不仅统一了GAN和扩散模型，还为改进GAN训练算法提供了新的设计原则。

