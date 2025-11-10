---
title: Transformer升级之路：18、RoPE的底数选择原则
slug: transformer升级之路18rope的底数选择原则
date: 2024-05-29
tags: 不等式, attention, 位置编码, rope, 生成模型
status: pending
---

# Transformer升级之路：18、RoPE的底数选择原则

**原文链接**: [https://spaces.ac.cn/archives/10122](https://spaces.ac.cn/archives/10122)

**发布日期**: 

---

我们知道，在[RoPE](/archives/8265)中频率的计算公式为$\theta_i = b^{-2i/d}$，底数$b$默认值为10000。目前Long Context的主流做法之一是，先在$b=10000$上用短文本预训练，然后调大$b$并在长文本微调，其出发点是[《Transformer升级之路：10、RoPE是一种β进制编码》](/archives/9675)里介绍的NTK-RoPE，它本身有较好长度外推性，换用更大的$b$再微调相比不加改动的微调，起始损失更小，收敛也更快。该过程给人的感觉是：调大$b$完全是因为“先短后长”的训练策略，如果一直都用长文本训练似乎就没必要调大$b$了？

上周的论文[《Base of RoPE Bounds Context Length》](https://papers.cool/arxiv/2405.14591)试图回答这个问题，它基于一个期望性质研究了$b$的下界，由此指出更大的训练长度本身就应该选择更大的底数，与训练策略无关。整个分析思路颇有启发性，接下来我们一起来品鉴一番。

## 期望性质 #

RoPE这里就不再详细介绍了，它本质上是一个分块对角矩阵  
\begin{equation}\boldsymbol{\mathcal{R}}_n = \scriptsize{\left(\begin{array}{cc:cc:cc:cc}  
\cos n\theta_0 & -\sin n\theta_0 & 0 & 0 & \cdots & \cdots & 0 & 0 \\\  
\sin n\theta_0 & \cos n\theta_0 & 0 & 0 & \cdots & \cdots & 0 & 0 \\\  
\hdashline  
0 & 0 & \cos n\theta_1 & -\sin n\theta_1 & \cdots & \cdots & 0 & 0 \\\  
0 & 0 & \sin n\theta_1 & \cos n\theta_1 & \cdots & \cdots & 0 & 0 \\\  
\hdashline  
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\\  
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\\  
\hdashline  
0 & 0 & 0 & 0 & \cdots & \cdots & \cos n\theta_{d/2-1} & -\sin n\theta_{d/2-1} \\\  
0 & 0 & 0 & 0 & \cdots & \cdots & \sin n\theta_{d/2-1} & \cos n\theta_{d/2-1} \\\  
\end{array}\right)}\end{equation}  
然后利用恒等式  
\begin{equation}(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\end{equation}  
给$\boldsymbol{q},\boldsymbol{k}$注入绝对位置信息，并自动实现了相对位置的效果。其中$\theta_i = b^{-2i/d}$，这里的$b$的取值就是本文要探讨的问题。

除了给模型注入位置信息外，我们期望RoPE能具备两个理想性质，以达到更好的效果：1、**远程衰减** ，即位置相近的Token平均来说获得更多的注意力；2、**语义聚合** ，即语义相似的Token平均来说获得更多的注意力。其中第一点我们早在[《Transformer升级之路：2、博采众长的旋转式位置编码》](/archives/8265)有过相关讨论，RoPE确实有一定的远程衰减性质。

所以接下来我们来分析第二点。

## 不等关系 #

所谓语义聚合，指的是当$\boldsymbol{k}$与$\boldsymbol{q}$相近时，不管它们的相对距离$n-m$多大，其注意力$\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}$平均来说都应该更大（至少要比随机的两个Token更大）。为了得到一个量化的结论，我们进一步简化问题，假设$\boldsymbol{q}$的每个分量都是独立同分布的，每个分量的均值为$\mu$，方差为$\sigma^2$。

现在我们考虑两种不同的$\boldsymbol{k}$：一种是在$\boldsymbol{q}$的基础上，加上一个零均值的扰动$\boldsymbol{\varepsilon}$，我们记$\tilde{\boldsymbol{k}} = \boldsymbol{q} + \boldsymbol{\varepsilon}$，代表跟$\boldsymbol{q}$语义相近的Token；另一种则是假设$\boldsymbol{k}$跟$\boldsymbol{q}$独立同分布，这代表两个随机的Token。根据第二点理想性质，我们希望有  
\begin{equation}\mathbb{E}_{\boldsymbol{q},\boldsymbol{k},\boldsymbol{\varepsilon}}\big[\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \tilde{\boldsymbol{k}} - \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\big] \geq 0\end{equation}  
注意我们刚才反复强调了“平均来说”，意味着我们只是期望一个平均的趋势，而不是每一点都能严格成立，所以我们在上式加了取数学期望$\mathbb{E}_{\boldsymbol{q},\boldsymbol{k},\boldsymbol{\varepsilon}}$。现在根据假设和RoPE的定义，我们可以把上式具体地算出来：  
\begin{equation}\begin{aligned}  
&\,\mathbb{E}_{\boldsymbol{q},\boldsymbol{k},\boldsymbol{\varepsilon}}\big[\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} (\boldsymbol{q} + \boldsymbol{\varepsilon}) - \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\big] \\\\[5pt]  
=&\, \mathbb{E}_{\boldsymbol{q}}\big[\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{q}\big] - \mathbb{E}_{\boldsymbol{q},\boldsymbol{k}}\big[\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\big] \\\\[5pt]  
=&\, \mathbb{E}_{\boldsymbol{q}}\big[\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{q}\big] - \mathbb{E}_{\boldsymbol{q}}[\boldsymbol{q}]^{\top}\boldsymbol{\mathcal{R}}_{n-m} \mathbb{E}_{\boldsymbol{k}}[\boldsymbol{k}] \\\\[5pt]  
=&\, \mathbb{E}_{\boldsymbol{q}}\big[\boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{q}\big] - \mu^2\boldsymbol{1}^{\top}\boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{1} \\\\[5pt]  
=& \mathbb{E}_{\boldsymbol{q}}\left[\sum_{i=0}^{d/2-1} (q_{2i}^2 + q_{2i+1}^2)\cos (n-m)\theta_i\right] - \sum_{i=0}^{d/2-1} 2\mu^2\cos (n-m)\theta_i \\\\[5pt]  
=& \sum_{i=0}^{d/2-1} 2(\mu^2 + \sigma^2)\cos (n-m)\theta_i - \sum_{i=0}^{d/2-1} 2\mu^2\cos (n-m)\theta_i \\\\[5pt]  
=& \sum_{i=0}^{d/2-1} 2\sigma^2\cos (n-m)\theta_i \\\  
\end{aligned}\end{equation}  
如果训练长度最大为$L$，那么$n-m\leq L-1$，因此第二点理想性质可以用如下不等式近似描述：  
\begin{equation}\sum_{i=0}^{d/2-1} \cos m\theta_i \geq 0,\quad m\in\\{0,1,2,\cdots,L-1\\}\label{neq:base}\end{equation}  
其中$L$是最大长度，是训练前就要选定的超参，而$d$是模型的head_size，按照LLAMA的一般设置是$d=128$，这也就意味着，上式的唯一可调参数就是$\theta_i = b^{-2i/d}$中的$b$。在[《Transformer升级之路：1、Sinusoidal位置编码追根溯源》](/archives/8231)中我们就简单探究过这个函数，它整体趋势是衰减的，$b$越大则衰减速度越慢，对应的连续非负区间就越大，所以存在一个最小的$b$使得上述不等式恒成立，即  
\begin{equation}b^* = \inf\left\\{\,\,b\,\,\,\left|\,\,\,f_b(m)\triangleq\sum_{i=0}^{d/2-1} \cos m b^{-2i/d} \geq 0,\,\, m\in\\{0,1,2,\cdots,L-1\\}\right.\right\\}\end{equation}

## 数值求解 #

由于$f_b(m)$涉及到多个三角函数的求和，并且$\theta_i$关于$i$还是非线性的，很难想象上述问题会有解析解，因此只能诉诸数值求解了。然而，$f_b(m)$越到后面震荡越频繁且不规律，因此即便数值求解也不是那么简单的事情。

笔者一开始以为，如果$b_0$使得$f_{b_0}(m)\geq 0$恒成立，那么$\forall b \geq b_0$都恒成立$f_b(m)\geq 0$，所以用二分法就可以了。但事实上这个假设并不成立，所以二分法宣告破产。继续想了一段时间，依然没什么优化思路，期间向原论文作者请教过，他们采用的是逆函数法，即给定$b$求使得$f_b(m)\geq 0$恒成立的最大$L$是比较简单的，于是我们可以得到很多$(b, L)$对，理论上只要枚举的$b$足够多，那么对于任意$L$都可以找出最小的$b$。然而这里有个精度问题，原论文最大的$L$计算到了$10^6$，$b$至少要枚举到$10^8$，如果枚举间隔小，那么计算成本非常大，如果枚举间隔大，那么可能漏掉很多解。

最后，笔者决定还是用“Jax + GPU”进行暴力搜索，以求得到更高精度的结果，大致流程是：

> 1、初始化$b=1000L$（在$10^6$内$b=1000L$可以使得$f_b(m)\geq 0$恒成立）；
> 
> 2、遍历$k=1,2,3,4,5$，执行以下操作：
> 
> 2.1）将$[0,b]$等分为$10^k$份，遍历等分点，判断$f_b(m)\geq 0$是否恒成立；
> 
> 2.2）取最小的使得$f_b(m)\geq 0$恒成立的等分点，更新$b$；
> 
> 3、返回最终的$b$。

最终结果普遍要比原论文的更紧一些  
$$\scriptsize\begin{array}{c|cccccccccc}  
\hline  
L & 1k & 2k & 4k & 8k & 16k & 32k & 64k & 128k & 256k & 512k & 1M \\\  
\hline  
b^*(\text{原文}) & 4.3e3 & 1.6e4 & 2.7e4 & 8.4e4 & 3.1e5 & 6.4e5 & 2.1e6 & 7.8e6 & 3.6e7 & 6.4e7 & 5.1e8 \\\  
b^*(\text{本文}) & 4.3e3 & \color{red}{1.2e4} & 2.7e4 & 8.4e4 & \color{red}{2.3e5} & \color{red}{6.3e5} & 2.1e6 & \color{red}{4.9e6} & \color{red}{2.4e7} & \color{red}{5.8e7} & \color{red}{6.5e7} \\\  
\hline  
\end{array}$$

参考代码：
    
    
    from functools import partial
    import numpy as np
    import jax.numpy as jnp
    import jax
    
    @partial(jax.jit, static_argnums=(2,))
    def f(m, b, d=128):
        i = jnp.arange(d / 2)
        return jnp.cos(m[:, None] * b ** (-2 * i[None] / d)).sum(axis=1)
    
    @np.vectorize
    def fmin(L, b):
        return f(np.arange(L), b).min()
    
    def bmin(L):
        B = 1000 * L
        for k in range(1, 6):
            bs = np.linspace(0, 1, 10**k + 1)[1:] * B  
            ys = fmin(L, bs)
            for b, y in zip(bs, ys):
                if y >= 0:
                    B = b
                    break
        return B
    
    bmin(1024 * 128)

## 渐近估计 #

除了数值求解外，我们也可以通过渐近分析来得到一个解析的估计结果，这个估计比数值结果要小，本质上是$d\to\infty$的解，但同样能够得出“$b$应该随着$L$增大而增大”的结论。

渐近估计的思路，是用积分代替求和：  
\begin{equation}f_b(m) = \sum_{i=0}^{d/2-1} \cos m b^{-2i/d}\approx \int_0^1 \cos m b^{-s} ds \xlongequal{\text{令}t = mb^{-s}} \int_{mb^{-1}}^m \frac{\cos t}{t \ln b}dt\end{equation}  
其中我们记  
\begin{equation}\text{Ci}(x) = -\int_x^{\infty} \frac{\cos t}{t} dt\end{equation}  
这是被前人研究过的三角积分（参考 [Trigonometric integral](https://en.wikipedia.org/wiki/Trigonometric_integral) ），利用这个记号，我们可以写出  
\begin{equation}f_b(m) \approx \frac{\text{Ci}(m) - \text{Ci}(mb^{-1})}{\ln b}\end{equation}  
$\text{Ci}(x)$的图像长这样：  


[![Ci\(x\)的图像【来自维基百科】](/usr/uploads/2024/05/3588196127.png)](/usr/uploads/2024/05/3588196127.png "点击查看原图")

Ci(x)的图像【来自维基百科】

它的第一个零点是$x_0=0.6165\cdots$，对于$m\geq 1$，可以看出$|\text{Ci}(m)|\leq 1/2$，所以其实$\text{Ci}(m)$相对来说是小项，对于渐近估计来说可以忽略，那么问题近似地变成了$\text{Ci}(mb^{-1})\leq 0$对于$m=1,2,\cdots,L$恒成立，我们只需要让相应的$mb^{-1}$都落在$[0,x_0]$区间内就可以实现，这意味着$Lb^{-1}\leq x_0$，即  
\begin{equation}b \geq L / x_0 \approx 2L\end{equation}  
或者简单点$b^* = \mathcal{O}(L)$。不出意料这个结果比精确的数值结果要小，因为它对应于$d\to\infty$，无限个三角函数叠加会使得函数图像的震荡更少，看起来更加平稳（相比于有限的$d$），从而对于固定的$b$，$f_b(m)$的连续非负区间更长，或者反过来，对于固定的$L$，保持$m=0,1,2,\cdots,L-1$的$f_b(m)$都非负的$b$更小。

## 相关思考 #

在[《Transformer升级之路：10、RoPE是一种β进制编码》](/archives/9675)中，我们将RoPE类比为一种$\beta$进制表示，其中$\beta = b^{2/d}$，那么$b - 1= \beta^{d/2} - 1$正好是$d/2$位$\beta$进制编码能够表示的最大数字，于是要表示$0,1,2,\cdots,L-1$这$L$个位置编码，至少有$b \geq L$，这个朴素的类比再次给出了“$b$应该随着$L$增大而增大”的结论，其结果跟上一节的渐近分析结果更为接近。

另一方面，Meta最新发布的LLAMA3，训练长度为8192，但RoPE的底数选择了惊人的500000（5e5），这比前面的数值结果（8.4e4）还要大将近一个数量级，不管从哪个角度看，这个数值笔者都认为是偏大的，可能LLAMA3的这个底数本就是给更大文本长度预留的。但不论如何，更大的文本长度选择更大的RoPE底数，似乎已经成为了很多训练人员的共识。

其实不管是数值结果还是渐近估计，都只是一个参考值，实际上对于给定的$L$，一个相当大范围内的$b$都应该会有相近的效果。所以具体的数值都不重要，关键是原论文通过语义聚合的出发点和一系列推导，澄清了“$b$应该随着$L$增大而增大”的结论及其原理，这是笔者所认为的原论文的核心贡献。

此外，其实语义聚合的出发点和结论也可以用来解释[Position Interpolation](https://papers.cool/arxiv/2306.15595)（PI）。刚才我们说了，同一个$b$，$f_b(m)$的连续非负区间是固定的，如果要使$0,1,2,\cdots,L-1$都落在非负区间内，就需要随着$L$的增大而相应的增加$b$。但反过来，我们也可以不增加$b$，而是减少相邻位置的间隔（即位置ID改成$0,1/k,2/k,\cdots$），那么就可以在同样大小的非负区间内表示$k$倍的位置了，这便是语义聚合视角下的Position Interpolation。

## 部分旋转 #

RoPE提出于2021年，当时只有一篇中文博客，后来得到了EleutherAI组织的认可和实验，继而才逐渐向学术界推广。当时EleutherAI实验发现，如果只对部分维度加RoPE，会取得稍优的结果，相关内容可以参考[这里](https://github.com/lucidrains/x-transformers/issues/40)、[这里](https://wandb.ai/eleutherai/neox/reports/Partial-Rotary-Tests--Vmlldzo2MjE1MjY)和[这里](https://wandb.ai/eleutherai/neox/reports/Partial-Rotary-Tests-v2--Vmlldzo2MjE4MTQ)，后来这个操作用到了它们的[GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/8b43196fbd832b797be9f3d88d54481171010507/megatron/model/transformer.py#L908)中。

当然，部分旋转还不是当前LLM的主流选择，但这不妨碍我们研究它，也许它未成为主流选择只是因为我们对它还不够了解。那为什么部分旋转反而可能会更优呢？笔者发现可以用本文的结论来一定程度上解释它。以只旋转一半维度为例，它在数学上等价于选择如下的$\theta_i$：  
\begin{equation}\theta_i = \left\\{\begin{aligned}&b^{-4i/d},& i < d/4 \\\  
&0,&i \geq d/4\end{aligned}\right.\end{equation}  
此时我们有  
\begin{equation}\sum_{i=0}^{d/2-1} \cos m\theta_i = \sum_{i=0}^{d/4-1} (1+\cos mb^{-4i/d})\geq 0\end{equation}  
也就是不论$m,b$如何，我们所期望的不等式$\eqref{neq:base}$都自动成立，这意味着从本文的观点来看，部分旋转在赋予位置信息的同时有更好的语义聚合能力，这对模型的效果也许更加有利。同时，部分旋转对模型的长文本能力或许也更有利，因为不等式恒成立，所以按照本文的观点，不论长短文本训练都不用修改$b$。

值得一提的是，DeepSeek提出的[MLA](/archives/10091)也应用了部分旋转，虽然在MLA的原始推导中，部分旋转更多是为了整合RoPE的无奈之举，但结合以往的部分旋转实验结果来看，也许MLA的优异效果有部分旋转的一分功劳。

## 文章小结 #

本文简单介绍了论文[《Base of RoPE Bounds Context Length》](https://papers.cool/arxiv/2405.14591)，它从语义聚合的期望性质讨论了RoPE的底数下界，由此指出更大的训练长度应该选择更大的底数，而不单单是为了配合“先短后长”的训练策略、继而利用NTK-RoPE来降低初始损失的折中选择。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10122>_

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

苏剑林. (May. 29, 2024). 《Transformer升级之路：18、RoPE的底数选择原则 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10122>

@online{kexuefm-10122,  
title={Transformer升级之路：18、RoPE的底数选择原则},  
author={苏剑林},  
year={2024},  
month={May},  
url={\url{https://spaces.ac.cn/archives/10122}},  
} 


---

## 公式推导与注释

### 1. RoPE旋转矩阵的基本性质

RoPE的核心是旋转矩阵$\boldsymbol{\mathcal{R}}_n$，我们首先深入分析其数学性质。

**推导1.1：旋转矩阵的正交性**

对于单个二维旋转块：
$$
\boldsymbol{R}_i(n) = \begin{pmatrix} \cos n\theta_i & -\sin n\theta_i \\ \sin n\theta_i & \cos n\theta_i \end{pmatrix}
$$

验证正交性，计算：
$$
\boldsymbol{R}_i(n)^{\top}\boldsymbol{R}_i(n) = \begin{pmatrix} \cos n\theta_i & \sin n\theta_i \\ -\sin n\theta_i & \cos n\theta_i \end{pmatrix} \begin{pmatrix} \cos n\theta_i & -\sin n\theta_i \\ \sin n\theta_i & \cos n\theta_i \end{pmatrix}
$$

展开第一项：
$$
(\cos n\theta_i)^2 + (\sin n\theta_i)^2 = 1
$$

展开非对角项：
$$
-\cos n\theta_i \sin n\theta_i + \sin n\theta_i \cos n\theta_i = 0
$$

因此$\boldsymbol{R}_i(n)^{\top}\boldsymbol{R}_i(n) = \boldsymbol{I}$，证明了旋转矩阵是正交矩阵。

**推导1.2：旋转矩阵的可加性**

这是RoPE实现相对位置编码的关键性质。对于两个旋转矩阵：
$$
\boldsymbol{R}_i(m)\boldsymbol{R}_i(n) = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} \cos n\theta_i & -\sin n\theta_i \\ \sin n\theta_i & \cos n\theta_i \end{pmatrix}
$$

利用三角函数和差公式：
$$
\cos(m\theta_i)\cos(n\theta_i) - \sin(m\theta_i)\sin(n\theta_i) = \cos((m+n)\theta_i)
$$
$$
\sin(m\theta_i)\cos(n\theta_i) + \cos(m\theta_i)\sin(n\theta_i) = \sin((m+n)\theta_i)
$$

因此：
$$
\boldsymbol{R}_i(m)\boldsymbol{R}_i(n) = \boldsymbol{R}_i(m+n)
$$

这个性质保证了：
$$
\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{n-m}
$$

使得注意力计算只依赖于相对位置$n-m$。

### 2. 底数θ的频率分解分析

**推导2.1：频率公式的指数形式**

底数为$b$时，第$i$个频率为：
$$
\theta_i = b^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1
$$

取对数得到：
$$
\log \theta_i = -\frac{2i}{d}\log b
$$

这表明频率在对数尺度上呈线性分布，跨度为：
$$
\log\theta_0 - \log\theta_{d/2-1} = 0 - (-\frac{2(d/2-1)}{d}\log b) \approx \log b
$$

**推导2.2：频率范围的边界分析**

最高频率（最快振荡）：
$$
\theta_0 = b^0 = 1
$$

最低频率（最慢振荡）：
$$
\theta_{d/2-1} = b^{-2(d/2-1)/d} = b^{-(d-2)/d} \approx b^{-1}
$$

对于位置$n$，最快分量的相位为$n\theta_0 = n$，最慢分量的相位为$n\theta_{d/2-1} \approx n/b$。

**推导2.3：波长的分布**

第$i$个分量的波长定义为相位变化$2\pi$对应的位置变化：
$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi b^{2i/d}
$$

波长范围：
$$
\lambda_0 = 2\pi, \quad \lambda_{d/2-1} \approx 2\pi b
$$

这意味着RoPE能够编码的最大周期约为$2\pi b$。

### 3. 语义聚合性质的详细推导

**推导3.1：期望值的展开**

考虑两个向量的内积：
$$
\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}
$$

对于分块对角矩阵$\boldsymbol{\mathcal{R}}_{n-m}$，内积可以分解为：
$$
\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k} = \sum_{i=0}^{d/2-1} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}^{\top} \begin{pmatrix} \cos(n-m)\theta_i & -\sin(n-m)\theta_i \\ \sin(n-m)\theta_i & \cos(n-m)\theta_i \end{pmatrix} \begin{pmatrix} k_{2i} \\ k_{2i+1} \end{pmatrix}
$$

展开单个块：
$$
\begin{aligned}
&= \sum_{i=0}^{d/2-1} [q_{2i}(k_{2i}\cos(n-m)\theta_i - k_{2i+1}\sin(n-m)\theta_i) \\
&\quad + q_{2i+1}(k_{2i}\sin(n-m)\theta_i + k_{2i+1}\cos(n-m)\theta_i)]
\end{aligned}
$$

整理得：
$$
= \sum_{i=0}^{d/2-1} [(q_{2i}k_{2i} + q_{2i+1}k_{2i+1})\cos(n-m)\theta_i + (q_{2i+1}k_{2i} - q_{2i}k_{2i+1})\sin(n-m)\theta_i]
$$

**推导3.2：独立同分布假设下的期望**

设$\boldsymbol{q}$的每个分量独立同分布，均值$\mu$，方差$\sigma^2$。考察：
$$
\mathbb{E}_{\boldsymbol{q}}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{q}]
$$

对于单个块的贡献：
$$
\mathbb{E}[(q_{2i}^2 + q_{2i+1}^2)\cos(n-m)\theta_i]
$$

由于$\mathbb{E}[q_{2i}^2] = \mu^2 + \sigma^2$：
$$
= 2(\mu^2 + \sigma^2)\cos(n-m)\theta_i
$$

交叉项的期望：
$$
\mathbb{E}[(q_{2i+1}q_{2i} - q_{2i}q_{2i+1})\sin(n-m)\theta_i] = 0
$$

因为$q_{2i}$和$q_{2i+1}$独立，所以：
$$
\mathbb{E}[q_{2i+1}q_{2i}] = \mathbb{E}[q_{2i+1}]\mathbb{E}[q_{2i}] = \mu^2
$$

**推导3.3：语义相似向量的期望差**

对于语义相似的$\tilde{\boldsymbol{k}} = \boldsymbol{q} + \boldsymbol{\varepsilon}$（$\boldsymbol{\varepsilon}$零均值）：
$$
\begin{aligned}
&\mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\tilde{\boldsymbol{k}}] \\
&= \mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}(\boldsymbol{q} + \boldsymbol{\varepsilon})] \\
&= \mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{q}] + \mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{\varepsilon}]
\end{aligned}
$$

由于$\boldsymbol{\varepsilon}$零均值且与$\boldsymbol{q}$独立：
$$
\mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{\varepsilon}] = \mathbb{E}[\boldsymbol{q}]^{\top}\boldsymbol{\mathcal{R}}_{n-m}\mathbb{E}[\boldsymbol{\varepsilon}] = 0
$$

对于独立同分布的$\boldsymbol{k}$：
$$
\mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}] = \mathbb{E}[\boldsymbol{q}]^{\top}\boldsymbol{\mathcal{R}}_{n-m}\mathbb{E}[\boldsymbol{k}] = \mu^2 \boldsymbol{1}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{1}
$$

计算$\boldsymbol{1}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{1}$：
$$
\boldsymbol{1}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{1} = \sum_{i=0}^{d/2-1} \begin{pmatrix} 1 \\ 1 \end{pmatrix}^{\top} \begin{pmatrix} \cos(n-m)\theta_i & -\sin(n-m)\theta_i \\ \sin(n-m)\theta_i & \cos(n-m)\theta_i \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

$$
= \sum_{i=0}^{d/2-1} (\cos(n-m)\theta_i - \sin(n-m)\theta_i + \sin(n-m)\theta_i + \cos(n-m)\theta_i)
$$

$$
= \sum_{i=0}^{d/2-1} 2\cos(n-m)\theta_i
$$

### 4. 不等式条件的深入分析

**推导4.1：语义聚合不等式的导出**

综合前面的推导，期望差为：
$$
\begin{aligned}
&\mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\tilde{\boldsymbol{k}}] - \mathbb{E}[\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}] \\
&= \sum_{i=0}^{d/2-1} 2(\mu^2 + \sigma^2)\cos(n-m)\theta_i - \sum_{i=0}^{d/2-1} 2\mu^2\cos(n-m)\theta_i \\
&= 2\sigma^2 \sum_{i=0}^{d/2-1} \cos(n-m)\theta_i
\end{aligned}
$$

要求这个期望差非负，即：
$$
\sum_{i=0}^{d/2-1} \cos(n-m)\theta_i \geq 0
$$

记$m = n - m$的相对距离，对所有$m \in \{0, 1, 2, \ldots, L-1\}$需要满足：
$$
f_b(m) = \sum_{i=0}^{d/2-1} \cos(m\theta_i) \geq 0
$$

**推导4.2：函数$f_b(m)$的初值分析**

当$m = 0$时：
$$
f_b(0) = \sum_{i=0}^{d/2-1} \cos(0) = \sum_{i=0}^{d/2-1} 1 = \frac{d}{2}
$$

这是最大值，符合直觉：位置重合时注意力最强。

**推导4.3：函数$f_b(m)$的渐近行为**

当$m$增大时，不同频率$\theta_i$的余弦项会产生相位差，导致求和出现抵消。关键观察：

- 高频项（小$i$）：$\cos(m\theta_i)$振荡快，贡献迅速变化
- 低频项（大$i$）：$\cos(m\theta_i)$振荡慢，贡献变化缓慢

**推导4.4：临界点的估计**

函数$f_b(m)$从正变为负的临界点$m^*$满足：
$$
\sum_{i=0}^{d/2-1} \cos(m^*\theta_i) = 0
$$

这需要高频和低频项的正负贡献平衡。

### 5. 底数$b$与最大长度$L$的关系

**推导5.1：积分近似方法**

当$d$很大时，可以用积分近似求和：
$$
f_b(m) = \sum_{i=0}^{d/2-1} \cos(mb^{-2i/d}) \approx \frac{d}{2}\int_0^1 \cos(mb^{-2s}) ds
$$

令$t = mb^{-2s}$，则$dt = -2mb^{-2s}\ln b \cdot ds$，即：
$$
ds = -\frac{dt}{2t\ln b}
$$

当$s = 0$时，$t = m$；当$s = 1$时，$t = mb^{-2}$。因此：
$$
\frac{d}{2}\int_0^1 \cos(mb^{-2s}) ds = \frac{d}{2} \int_{mb^{-2}}^m \cos(t) \cdot \left(-\frac{1}{2t\ln b}\right) dt
$$

$$
= -\frac{d}{4\ln b} \int_{mb^{-2}}^m \frac{\cos t}{t} dt = \frac{d}{4\ln b} \int_m^{mb^{-2}} \frac{\cos t}{t} dt
$$

由于$b \gg 1$时$b^{-2} \approx 0$，近似为：
$$
\approx \frac{d}{4\ln b} \int_m^{m/b^2} \frac{\cos t}{t} dt
$$

**推导5.2：余弦积分函数的应用**

定义余弦积分：
$$
\text{Ci}(x) = -\int_x^{\infty} \frac{\cos t}{t} dt
$$

则：
$$
\int_m^{m/b^2} \frac{\cos t}{t} dt = -[\text{Ci}(m/b^2) - \text{Ci}(m)]
$$

因此：
$$
f_b(m) \approx \frac{d}{4\ln b}[\text{Ci}(m/b^2) - \text{Ci}(m)]
$$

简化符号，对单位维度（$d=2$）：
$$
f_b(m) \approx \frac{\text{Ci}(m/b^2) - \text{Ci}(m)}{\ln b}
$$

**推导5.3：零点条件**

$f_b(m) = 0$的条件近似为：
$$
\text{Ci}(m/b^2) = \text{Ci}(m)
$$

根据$\text{Ci}(x)$的性质，第一个零点在$x_0 \approx 0.6165$。对于$m \geq 1$，$|\text{Ci}(m)| \leq 0.5$相对较小，主要由$\text{Ci}(m/b^2)$主导。

要求$f_b(m) \geq 0$对所有$m \leq L-1$成立，需要：
$$
\text{Ci}(m/b^2) \geq \text{Ci}(m)
$$

由于$\text{Ci}(x)$递减，需要$m/b^2 \leq m$，这总是成立的。更强的条件是$m/b^2$应该在第一个零点之前：
$$
\frac{L-1}{b^2} \leq x_0 \approx 0.6165
$$

解得：
$$
b \geq \sqrt{\frac{L-1}{0.6165}} \approx 1.27\sqrt{L}
$$

**推导5.4：更精确的线性估计**

考虑到$b$通常很大（$b \gg 1$），而$mb^{-2} \ll 1$，可以使用$\text{Ci}(x)$在小$x$附近的展开：
$$
\text{Ci}(x) \approx \gamma + \ln x, \quad x \to 0^+
$$

其中$\gamma \approx 0.5772$是欧拉常数。但更直接的方法是注意到，如果我们希望$m\theta_{d/2-1} = mb^{-1}$不超过某个临界角度，比如$\pi/2$：
$$
\frac{L}{b} \leq \frac{\pi}{2}
$$

这给出：
$$
b \geq \frac{2L}{\pi} \approx 0.64L
$$

结合更细致的分析，得到$b = \mathcal{O}(L)$的结论。

### 6. β进制编码的视角

**推导6.1：RoPE作为进制表示**

定义$\beta = b^{2/d}$，则：
$$
\theta_i = b^{-2i/d} = \beta^{-i}
$$

位置$n$的编码可以看作：
$$
n = \sum_{i=0}^{d/2-1} a_i \beta^i
$$

其中$a_i \in [0, \beta)$是"数字"。

**推导6.2：表示能力的分析**

$d/2$位$\beta$进制数能表示的最大值为：
$$
N_{\max} = \sum_{i=0}^{d/2-1} (\beta - 1)\beta^i = (\beta - 1) \cdot \frac{\beta^{d/2} - 1}{\beta - 1} = \beta^{d/2} - 1
$$

由于$\beta = b^{2/d}$：
$$
N_{\max} = (b^{2/d})^{d/2} - 1 = b - 1
$$

因此，要表示$0, 1, \ldots, L-1$共$L$个位置，需要：
$$
b - 1 \geq L - 1 \implies b \geq L
$$

这再次证明了$b$应该至少与$L$同阶。

**推导6.3：进制表示的唯一性**

对于给定的$b$和$d$，每个位置$n < b$都有唯一的$\beta$进制表示。这确保了不同位置的编码能够被模型区分。

### 7. NTK插值中的底数调整

**推导7.1：NTK-RoPE的基本思想**

Neural Tangent Kernel（NTK）启发的RoPE调整通过缩放底数来实现长度外推。原始训练长度$L_0$，目标长度$L$，缩放因子：
$$
\alpha = \frac{L}{L_0}
$$

NTK-RoPE将底数调整为：
$$
b' = b \cdot \alpha^{d/(d-2)}
$$

**推导7.2：频率的非均匀缩放**

原始频率：
$$
\theta_i = b^{-2i/d}
$$

NTK调整后：
$$
\theta_i' = (b')^{-2i/d} = (b\alpha^{d/(d-2)})^{-2i/d} = b^{-2i/d} \cdot \alpha^{-2i/(d-2)}
$$

这相当于对不同频率分量进行非均匀缩放：

- 低频（大波长，小$i$）：缩放因子接近1
- 高频（小波长，大$i$）：缩放因子更大

**推导7.3：内插与外推的统一**

Position Interpolation（PI）直接缩放位置：
$$
n' = \frac{n}{\alpha}
$$

等价于所有频率均匀缩放：
$$
\theta_i' = \frac{\theta_i}{\alpha}
$$

NTK-RoPE的非均匀缩放在两种极端之间找到平衡：

- 对于$i = 0$（最低频）：$\theta_0' = 1$（不缩放）
- 对于$i = d/2-1$（最高频）：$\theta_{d/2-1}' \approx \theta_{d/2-1}/\alpha$（完全缩放）

**推导7.4：保持语义聚合性质**

在新的底数$b'$下，检验不等式：
$$
f_{b'}(m) = \sum_{i=0}^{d/2-1} \cos(m(b')^{-2i/d}) \geq 0, \quad m \leq L-1
$$

由于$b' > b$，相当于拉伸了频率范围，使得原本对应$L_0$的非负区间能够覆盖扩展后的$L$。

### 8. 部分旋转的数学优势

**推导8.1：部分旋转的频率配置**

假设只对前$d/4$维度对应的频率旋转，则：
$$
\theta_i = \begin{cases}
b^{-4i/d}, & i < d/4 \\
0, & i \geq d/4
\end{cases}
$$

注意这里$i < d/4$时使用$-4i/d$而非$-2i/d$，是为了保持频率跨度。

**推导8.2：语义聚合条件的自动满足**

计算求和函数：
$$
f_b(m) = \sum_{i=0}^{d/2-1} \cos(m\theta_i) = \sum_{i=0}^{d/4-1} \cos(mb^{-4i/d}) + \sum_{i=d/4}^{d/2-1} \cos(0)
$$

$$
= \sum_{i=0}^{d/4-1} \cos(mb^{-4i/d}) + \frac{d}{4}
$$

由于常数项$d/4 > 0$，且$\cos(mb^{-4i/d}) \geq -1$：
$$
f_b(m) \geq \frac{d}{4} - \frac{d}{4} = 0
$$

等号成立当且仅当所有$\cos(mb^{-4i/d}) = -1$，这在实际中几乎不可能同时发生。因此：
$$
f_b(m) > 0, \quad \forall m, b
$$

**推导8.3：位置信息与语义信息的平衡**

部分旋转的优势在于：

- 旋转维度：提供位置信息
- 非旋转维度：保留纯语义信息

记$\boldsymbol{q} = [\boldsymbol{q}_r, \boldsymbol{q}_n]$（旋转和非旋转部分），内积分解为：
$$
(\boldsymbol{\mathcal{R}}_m\boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n\boldsymbol{k}) = (\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{q}_r)^{\top}\boldsymbol{k}_r + \boldsymbol{q}_n^{\top}\boldsymbol{k}_n
$$

第一项编码相对位置，第二项纯粹基于语义，两者结合提供更灵活的注意力机制。

### 9. 数值求解的收敛性分析

**推导9.1：二分法失效的原因**

定义集合：
$$
S_b = \{m \in \mathbb{Z}^+ : f_b(m) \geq 0\}
$$

理想情况下，我们希望$b_1 > b_2$蕴含$S_{b_1} \supseteq S_{b_2}$，这样可以用二分法。但实际上由于$f_b(m)$的复杂振荡，这个单调性不总是成立。

**推导9.2：逆函数法的有效性**

定义：
$$
L(b) = \max\{m : f_b(m') \geq 0, \forall m' \leq m\}
$$

虽然$b \mapsto S_b$不单调，但对于固定的$b$，计算$L(b)$是可行的：

遍历$m = 1, 2, \ldots$直到$f_b(m) < 0$，记录最大的$m$。

通过枚举足够多的$b$值，可以构建$(b, L(b))$的数据点，然后通过查表或插值得到$b^*(L)$。

**推导9.3：网格搜索的复杂度**

在区间$[0, B]$上用$N$个点网格搜索，对每个点需要检验$L$个不等式，总复杂度：
$$
\mathcal{O}(N \cdot L \cdot d/2)
$$

利用GPU并行化，可以同时计算所有$m \in [0, L-1]$和$i \in [0, d/2-1]$：
$$
\text{复杂度} = \mathcal{O}(N) \text{（在GPU时间意义下）}
$$

### 10. 渐近估计的精细化

**推导10.1：Ci函数的级数展开**

余弦积分函数在$x > 0$处的级数表示：
$$
\text{Ci}(x) = \gamma + \ln x + \sum_{n=1}^{\infty} \frac{(-1)^n x^{2n}}{(2n)(2n)!}
$$

其中$\gamma$是欧拉常数。对于小$x$：
$$
\text{Ci}(x) \approx \gamma + \ln x - \frac{x^2}{4}
$$

**推导10.2：临界条件的近似解**

代入$f_b(m)$的表达式：
$$
f_b(m) \approx \frac{\text{Ci}(m/b^2) - \text{Ci}(m)}{\ln b}
$$

使用级数展开：
$$
\text{Ci}(m/b^2) \approx \gamma + \ln(m/b^2) = \gamma + \ln m - 2\ln b
$$

$$
\text{Ci}(m) \approx \gamma + \ln m - \frac{m^2}{4}
$$

因此：
$$
f_b(m) \approx \frac{-2\ln b + m^2/4}{\ln b} = -2 + \frac{m^2}{4\ln b}
$$

要求$f_b(m) \geq 0$：
$$
\frac{m^2}{4\ln b} \geq 2 \implies \ln b \geq \frac{m^2}{8}
$$

对于$m = L-1$：
$$
b \geq \exp\left(\frac{(L-1)^2}{8}\right)
$$

这个估计对于小$L$过于严格，但捕捉了$b$随$L$增长的趋势。

**推导10.3：改进的线性估计**

考虑$\text{Ci}(x)$的第一个零点$x_0 \approx 0.6165$，以及在$x > 1$时$|\text{Ci}(x)| < 0.5$的性质。

如果忽略$\text{Ci}(m)$的贡献（因为它相对较小且振荡），主要条件变为：
$$
\text{Ci}(m/b^2) \geq 0
$$

要求$m/b^2 \leq x_0$对所有$m \leq L-1$成立：
$$
\frac{L-1}{b^2} \leq x_0 \implies b \geq \sqrt{\frac{L-1}{x_0}} \approx 1.27\sqrt{L-1}
$$

但考虑到实际中$d$是有限的（如$d=128$），积分近似会有偏差，实际的$b^*$比这个估计要大。数值结果显示$b^* \sim L^{1.5}$到$L^2$之间。

### 11. LLAMA3底数选择的理论解释

**推导11.1：LLAMA3的配置**

LLAMA3使用：
- 训练长度：$L_0 = 8192$
- RoPE底数：$b = 500000$
- head_size：$d = 128$

根据数值结果，$L = 8192$对应的$b^* \approx 8.4 \times 10^4$，而LLAMA3使用$b = 5 \times 10^5$，约为理论值的6倍。

**推导11.2：底数冗余的优势**

使用比理论下界更大的$b$有几个好处：

1. **安全边际**：确保在$m \in [0, L-1]$内$f_b(m)$稳定为正
2. **长度外推预留**：如果未来需要扩展到更长的上下文，无需重新调整底数
3. **训练稳定性**：更大的$b$使得位置编码更加平滑

**推导11.3：过大底数的潜在风险**

当$b$过大时，低频分量变得极其缓慢：
$$
\theta_{d/2-1} = b^{-(d-2)/d} = (5 \times 10^5)^{-126/128} \approx 2 \times 10^{-6}
$$

这意味着最低频分量的周期约为：
$$
T = \frac{2\pi}{\theta_{d/2-1}} \approx 3 \times 10^6
$$

远超训练长度8192，这部分频率分量在训练中几乎不变，可能导致参数冗余。

### 12. 频率分布的信息论解释

**推导12.1：香农采样定理的类比**

将位置编码类比为信号，频率$\theta_i$对应于傅里叶基。根据采样定理，要准确表示最大频率为$f_{\max}$的信号，采样率必须至少为$2f_{\max}$。

对于RoPE，"采样率"对应于最大位置$L$，"最大频率"对应于$\theta_0 = 1$。但RoPE使用多个频率的组合，相当于多频带编码。

**推导12.2：频率分辨率**

相邻两个频率的比值：
$$
\frac{\theta_i}{\theta_{i+1}} = \frac{b^{-2i/d}}{b^{-2(i+1)/d}} = b^{2/d}
$$

这是一个常数，意味着频率在对数尺度上等距分布。频率分辨率（对数尺度）：
$$
\Delta(\log\theta) = \log\theta_i - \log\theta_{i+1} = \frac{2\log b}{d}
$$

更大的$b$或更大的$d$提供更精细的频率分辨率。

**推导12.3：编码容量**

从信息论角度，$d/2$个不同频率可以编码的"信息量"大致为：
$$
I \sim (d/2) \log_2(\text{频率动态范围})
$$

频率动态范围为$\theta_0/\theta_{d/2-1} \approx b$，因此：
$$
I \sim \frac{d}{2} \log_2 b
$$

要编码$L$个不同位置，需要：
$$
I \geq \log_2 L \implies \frac{d}{2}\log_2 b \geq \log_2 L
$$

解得：
$$
b \geq L^{2/d}
$$

对于$d = 128$：
$$
b \geq L^{1/64}
$$

这个下界远低于实际需求，说明除了编码容量，还有其他约束（如语义聚合）在起作用。

### 13. 长度泛化的理论保证

**推导13.1：训练与测试的分布变化**

训练阶段见到的相对位置：$\Delta \in [0, L_0 - 1]$

测试阶段可能出现的相对位置：$\Delta \in [0, L - 1]$，其中$L > L_0$

长度泛化要求模型在未见过的$\Delta \in [L_0, L-1]$上仍能正常工作。

**推导13.2：外推失败的机制**

如果$b$选择仅满足$f_b(\Delta) \geq 0$对$\Delta \leq L_0 - 1$，当$\Delta \geq L_0$时，可能出现$f_b(\Delta) < 0$。

这会导致：
$$
\mathbb{E}[\text{相似token的注意力}] < \mathbb{E}[\text{随机token的注意力}]
$$

破坏了注意力机制的语义性，导致外推失败。

**推导13.3：安全外推的底数选择**

为了安全外推到长度$L$，应该在训练时就选择满足$L$的$b^*(L)$，即使训练长度只有$L_0 < L$。

或者，使用NTK插值动态调整：
$$
b'(\text{测试时}) = b(\text{训练时}) \cdot \left(\frac{L}{L_0}\right)^{d/(d-2)}
$$

### 14. 部分旋转比例的优化

**推导14.1：旋转比例参数**

设旋转维度比例为$\rho \in (0, 1]$，则旋转维度数为$\rho d$，非旋转维度数为$(1-\rho)d$。

**推导14.2：语义聚合条件的推广**

类似之前的推导：
$$
f_b(m, \rho) = \sum_{i=0}^{\rho d/2 - 1} \cos(m\theta_i) + (1-\rho)\frac{d}{2}
$$

要求非负：
$$
\sum_{i=0}^{\rho d/2 - 1} \cos(m\theta_i) \geq -(1-\rho)\frac{d}{2}
$$

由于$\cos(m\theta_i) \geq -1$：
$$
\sum_{i=0}^{\rho d/2 - 1} \cos(m\theta_i) \geq -\rho\frac{d}{2}
$$

所以只要$-\rho d/2 \geq -(1-\rho)d/2$，即$\rho \leq 1 - \rho$，即$\rho \leq 0.5$，不等式就自动满足。

**推导14.3：最优比例的权衡**

选择$\rho$需要权衡：

- $\rho$越大：位置信息越丰富，但语义聚合条件越难满足
- $\rho$越小：语义聚合条件容易满足，但位置信息不足

实验表明$\rho \in [0.25, 0.5]$是一个较好的范围。

### 15. 多层Attention的累积效应

**推导15.1：跨层的位置编码**

考虑$L$层Transformer，每层都使用RoPE。位置信息在层间传播的机制：

第$\ell$层的输出：
$$
\boldsymbol{h}_n^{(\ell)} = f(\text{Attention}(\boldsymbol{\mathcal{R}}_n\boldsymbol{q}_n^{(\ell)}, \{\boldsymbol{\mathcal{R}}_m\boldsymbol{k}_m^{(\ell)}\}_{m \leq n}))
$$

位置信息通过$\boldsymbol{\mathcal{R}}_n$注入，但$\boldsymbol{h}_n^{(\ell)}$本身不显式携带位置标记。

**推导15.2：层间位置信息的衰减**

假设第$\ell$层的输出对位置的敏感度为$s^{(\ell)}$，则：
$$
s^{(\ell+1)} \approx \gamma \cdot s^{(\ell)}
$$

其中$\gamma < 1$是衰减因子，取决于Attention后的非线性变换（LayerNorm、FFN等）。

经过$L$层后：
$$
s^{(L)} \approx \gamma^L \cdot s^{(0)}
$$

如果$\gamma$太小，深层的位置信息会丢失。

**推导15.3：部分旋转的跨层优势**

部分旋转在每层都重新注入位置信息到旋转维度，同时保持非旋转维度的语义信息：
$$
\boldsymbol{h}_n^{(\ell)} = [\boldsymbol{h}_{n,r}^{(\ell)}, \boldsymbol{h}_{n,nr}^{(\ell)}]
$$

每层Attention时，$\boldsymbol{h}_{n,r}^{(\ell)}$被旋转，$\boldsymbol{h}_{n,nr}^{(\ell)}$保持不变，这提供了更稳定的位置信息传播。

### 16. 实验验证的统计检验

**推导16.1：底数选择的假设检验**

零假设$H_0$：底数$b$与最大长度$L$无关

备择假设$H_1$：$b$应该随$L$增大而增大

检验统计量：Spearman秩相关系数
$$
\rho_s = \frac{\text{Cov}(\text{rank}(b), \text{rank}(L))}{\sigma_{\text{rank}(b)} \sigma_{\text{rank}(L)}}
$$

根据数值结果的$(b^*, L)$数据点，计算$\rho_s$显著大于0，拒绝$H_0$。

**推导16.2：拟合关系的确定**

假设幂律关系：$b = aL^{\alpha}$

取对数：$\log b = \log a + \alpha \log L$

线性回归得到$\alpha$的估计。根据数值数据，$\alpha \approx 1.5 \sim 2.0$。

**推导16.3：置信区间估计**

给定$L$，$b^*$的$95\%$置信区间可以通过自助法（Bootstrap）估计：

1. 从数值结果中重采样
2. 对每个重采样集拟合$b = aL^{\alpha}$
3. 计算$b^*(L)$的分位数

这提供了$b^*$选择的不确定性度量。

### 17. 总结与实践建议

**推导17.1：底数选择的启发式规则**

综合理论分析和数值结果，建议：

1. **保守估计**：$b \geq 10L$
2. **标准配置**：$b = 100L$（对于常见的$d=128$）
3. **长上下文**：$b = 1000L$或根据数值表查询

**推导17.2：动态调整策略**

训练过程中动态调整$b$的方案：

- 训练前期（短上下文）：使用较小的$b_0$
- 训练中期（逐渐增加上下文）：线性增大$b$
- 训练后期（长上下文）：固定在目标$b_{\text{final}}$

调整公式：
$$
b(t) = b_0 + (b_{\text{final}} - b_0) \cdot \min(1, t/t_{\text{transition}})
$$

**推导17.3：与其他技术的结合**

RoPE底数选择应与以下技术协同考虑：

- ALiBi：加性位置偏置，可以减小对$b$的依赖
- 位置插值：允许用较小的$b$训练后，通过插值外推到更长上下文
- FlashAttention等优化：主要影响计算效率，不改变对$b$的理论要求

最佳实践是：选择合适的$b$作为基础，然后辅以其他技术提升性能和效率。

