---
title: 生成扩散模型漫谈（二十一）：中值定理加速ODE采样
slug: 生成扩散模型漫谈二十一中值定理加速ode采样
date: 2023-12-07
tags: 微分方程, 生成模型, 扩散, 生成模型, attention
status: pending
---

# 生成扩散模型漫谈（二十一）：中值定理加速ODE采样

**原文链接**: [https://spaces.ac.cn/archives/9881](https://spaces.ac.cn/archives/9881)

**发布日期**: 

---

在生成扩散模型的发展史上，[DDIM](/archives/9181)和同期Song Yang的[扩散SDE](/archives/9209)都称得上是里程碑式的工作，因为它们建立起了扩散模型与随机微分方程（SDE）、常微分方程（ODE）这两个数学领域的紧密联系，从而允许我们可以利用SDE、ODE已有的各种数学工具来对分析、求解和拓展扩散模型，比如后续大量的加速采样工作都以此为基础，可以说这打开了生成扩散模型的一个全新视角。

本文我们聚焦于ODE。在本系列的[（六）](/archives/9228)、[（十二）](/archives/9280)、[（十四）](/archives/9370)、[（十五）](/archives/9379)、[（十七）](/archives/9497)等博客中，我们已经推导过ODE与扩散模型的联系，本文则对扩散ODE的采样加速做简单介绍，并重点介绍一种巧妙地利用“中值定理”思想的新颖采样加速方案“AMED”。

## 欧拉方法 #

正如前面所说，我们已经有多篇文章推导过扩散模型与ODE的联系，所以这里不重复介绍，而是直接将扩散ODE的采样定义为如下ODE的求解：  
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\label{eq:dm-ode}\end{equation}  
其中$t\in[0,T]$，初值条件是$\boldsymbol{x}_T$，要返回的结果是$\boldsymbol{x}_0$。原则上我们并不关心$t\in(0,1)$时的中间值$\boldsymbol{x}_t$，只需要最终的$\boldsymbol{x}_0$。为了数值求解，我们还需要选定节点$0=t_0 < t_1 < t_2 < \cdots < t_N = T$，常见的选择是  
\begin{equation}t_n=\left(t_1^{1 / \rho}+\frac{n-1}{N-1}\left(t_N^{1 / \rho}-t_1^{1 / \rho}\right)\right)^\rho\end{equation}  
其中$\rho > 0$。该形式来自[《Elucidating the Design Space of Diffusion-Based Generative Models》](https://papers.cool/arxiv/2206.00364)（EDM），AMED也沿用了该方案，个人认为节点的选择不算关键要素，因此本文对此不做深究。

最简单的求解器是“欧拉方法”：利用差分近似  
\begin{equation}\left.\frac{d\boldsymbol{x}_t}{dt}\right|_{t=t_{n+1}}\approx \frac{\boldsymbol{x}_{t_{n+1}} - \boldsymbol{x}_{t_n}}{t_{n+1} - t_n}\end{equation}  
我们可以得到  
\begin{equation}\boldsymbol{x}_{t_n}\approx \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})(t_{n+1} - t_n)\end{equation}  
这通常也直接称为DDIM方法，因为是DDIM首先注意到它的采样过程对应于ODE的欧拉法，继而反推出对应的ODE。

## 高阶方法 #

从数值求解的角度来看，欧拉方法属于一阶近似，特点是简单快捷，缺点是精度差，所以步长不能太小，这意味着单纯利用欧拉法不大可能明显降低采样步数并且保证采样质量。因此，后续的采样加速工作都应用了更高阶的方法。

比如，直觉上差分$\frac{\boldsymbol{x}_{t_{n+1}} - \boldsymbol{x}_{t_n}}{t_{n+1} - t_n}$应该更接近中间点的导数而不是边界的导数，所以右端也换成$t_n$和$t_{n+1}$的平均应该会有更高的精度：  
\begin{equation}\frac{\boldsymbol{x}_{t_{n+1}} - \boldsymbol{x}_{t_n}}{t_{n+1} - t_n}\approx \frac{1}{2}\left[\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_n}, t_n) + \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})\right]\label{eq:heun-0}\end{equation}  
由此我们可以得到  
\begin{equation}\boldsymbol{x}_{t_n}\approx \boldsymbol{x}_{t_{n+1}} - \frac{1}{2}\left[\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_n}, t_n) + \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})\right](t_{n+1} - t_n) \end{equation}  
然而，右端出现了$\boldsymbol{x}_{t_n}$，而我们要做的就是计算$\boldsymbol{x}_{t_n}$，所以这样的等式并不能直接用来迭代，为此，我们用欧拉方法“预估”一下$\boldsymbol{x}_{t_n}$，然后替换掉上式中的$\boldsymbol{x}_{t_n}$：  
\begin{equation}\begin{aligned}  
\tilde{\boldsymbol{x}}_{t_n}=&\, \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})(t_{n+1} - t_n) \\\  
\boldsymbol{x}_{t_n}\approx&\, \boldsymbol{x}_{t_{n+1}} - \frac{1}{2}\left[\boldsymbol{v}_{\boldsymbol{\theta}}(\tilde{\boldsymbol{x}}_{t_n}, t_n) + \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})\right](t_{n+1} - t_n)  
\end{aligned}\label{eq:heun}\end{equation}  
这就是EDM所用的“[Heun方法](https://en.wikipedia.org/wiki/Heun%27s_method)”，是一种二阶方法。这样每步迭代需要算两次$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$，但精度明显提高，因此可以明显减少迭代步数，总的计算成本是降低的。

二阶方法还有很多变体，比如式$\eqref{eq:heun-0}$的右端我们可以直接换成中间点$t=(t_n+t_{n+1})/2$的函数值，这得到  
\begin{equation}\boldsymbol{x}_{t_n}\approx \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{(t_n+t_{n+1})/2}, \frac{t_n+t_{n+1}}{2}\right)(t_{n+1} - t_n) \end{equation}  
中间点也有不同的求法，除了代数平均$(t_n+t_{n+1})/2$外，也可以考虑几何平均  
\begin{equation}\boldsymbol{x}_{t_n}\approx \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_{\sqrt{t_n t_{n+1}}}, \sqrt{t_n t_{n+1}}\right)(t_{n+1} - t_n) \label{eq:dpm-solver-2}\end{equation}  
事实上，式$\eqref{eq:dpm-solver-2}$就是[DPM-Solver-2](https://papers.cool/arxiv/2206.00927)的一个特例。

除了二阶方法外，ODE的求解还有不少更高阶的方法，如"[Runge-Kutta方法](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)”、“[线性多步法](https://en.wikipedia.org/wiki/Linear_multistep_method)”等。然而，不管是二阶方法还是高阶方法，虽然都能一定程度上加速扩散ODE的采样，但由于这些都是“通法”，没有针对扩散模型的背景和形式进行定制，因此很难将采样过程的计算步数降到极致（个位数）。

## 中值定理 #

至此，本文的主角AMED登场了，其论文[《Fast ODE-based Sampling for Diffusion Models in Around 5 Steps》](https://papers.cool/arxiv/2312.00094)前两天才放到Arxiv，可谓“新鲜滚热辣”。AMED并非像传统的ODE求解器那样一味提高理论精度，而是巧妙地类比了“中值定理”，并加上非常小的蒸馏成本，为扩散ODE** _定制_** 了高速的求解器。

[![几种扩散ODE-Solver示意图](/usr/uploads/2023/12/1641140949.png)](/usr/uploads/2023/12/1641140949.png "点击查看原图")

几种扩散ODE-Solver示意图

首先，我们对方程$\eqref{eq:dm-ode}$两端积分，那么可以写出精确的等式：  
\begin{equation} \boldsymbol{x}_{t_{n+1}} - \boldsymbol{x}_{t_n} = \int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)dt\end{equation}  
如果$\boldsymbol{v}$只是一维的标量函数，那么由“[积分中值定理](https://en.wikipedia.org/wiki/Mean_value_theorem#Mean_value_theorems_for_definite_integrals)”我们可以知道存在点$s_n\in(t_n, t_{n+1})$，使得  
\begin{equation}\frac{1}{t_{n+1} - t_n}\int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)dt = \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{s_n}, s_n) \end{equation}  
很遗憾，中值定理对一般的向量函数并不成立。不过，在$t_{n+1}-t_n$不太大以及一定的假设之下，我们依然可以类比地写出近似  
\begin{equation}\frac{1}{t_{n+1} - t_n}\int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)dt \approx \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{s_n}, s_n) \end{equation}  
于是我们得到  
\begin{equation} \boldsymbol{x}_{t_n}\approx \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{s_n}, s_n)(t_{n+1}-t_n)\end{equation}  
当然，目前还只是一个形式解，$s_n$和$\boldsymbol{x}_{s_n}$怎么来还未解决。对于$\boldsymbol{x}_{s_n}$，我们依然用欧拉方法进行预估，即$\tilde{\boldsymbol{x}}_{s_n}= \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})(t_{n+1} - s_n)$；对于$s_n$，我们则用一个小型的神经网络去估计它：  
\begin{equation}s_n = g_{\boldsymbol{\phi}}(\boldsymbol{h}_{t_{n+1}}, t_{n+1})\end{equation}  
其中$\boldsymbol{\phi}$是训练参数，$\boldsymbol{h}_{t_{n+1}}$是U-Net模型$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})$的中间特征。最后，为了求解参数$\boldsymbol{\phi}$，我们采用蒸馏的思想，预先用步数更多的求解器求出精度更高的轨迹点对$(\boldsymbol{x}_{t_n},\boldsymbol{x}_{t_{n+1}})$，然后最小化估计误差。这就是论文中的AMED-Solver（**A** pproximate **ME** an-**D** irection Solver），它具备常规ODE-Solver的形式，但又需要额外的蒸馏成本，然而这点蒸馏成本相比其他蒸馏加速方法又几乎可以忽略不计，所以笔者将它理解为“定制”求解器。

定制一词非常关键，扩散ODE的采样加速研究由来已久，在众多研究人员的贡献加成下，非训练的求解器大概已经走了非常远，但依然未能将采样步数降到极致，除非未来我们对扩散模型的理论理解有进一步的突破，否则笔者不认为非训练的求解器还有显著的提升空间。因此，AMED这种带有少量训练成本的加速度，既是“剑走偏锋”、“另辟蹊径”，也是“应运而生”、“顺理成章”。

## 实验结果 #

在看实验结果之前，我们首先了解一个名为“NFE”的概念，全称是“Number of Function Evaluations”，说白了就是模型$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$的执行次数，它跟计算量直接挂钩。比如，一阶方法每步迭代的NFE是1，因为只需要执行一次$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$，而二阶方法每一步迭代的NFE是2，AMED-Solver的$g_{\boldsymbol{\phi}}$计算量很小，可以忽略不计，所以AMED-Solver每一步的NFE也算是2。为了实现公平的比较，需要保持整个采样过程中总的NFE不变，来对比不同Solver的效果。

基本的实验结果是原论文的Table 2：  


[![AMED的实验结果（Table 2）](/usr/uploads/2023/12/3251445298.png)](/usr/uploads/2023/12/3251445298.png "点击查看原图")

AMED的实验结果（Table 2）

这个表格有几个值得特别留意的地方。第一，在NFE不超过5时，二阶的DPM-Solver、EDM效果还不如一阶的DDIM，这是因为Solver的误差不仅跟阶次有关，还跟步长$t_{n+1}-t_n$有关，大致上的关系就是$\mathcal{O}((t_{n+1}-t_n)^m)$，其中$m$就是“阶”，在总NFE较小时，高阶方法只能取较大的步长，所以实际精度反而更差，从而效果不佳；第二，同样是二阶方法的SMED-Solver，在小NFE时效果取得了全面SOTA，这充分体现了“定制”的重要性；第三，这里的“AMED-Plugin”是原论文提出的将AMED的思想作为其他ODESolver的“插件”的用法，细节更加复杂一些，但取得了更好的效果。

可能有读者会疑问：既然二阶方法每一步迭代都需要2个NFE，那么表格中怎么会出现奇数的NFE？其实，这是因为作者用到了一个名为“AFS（Analytical First Step）”的技巧来减少了1个NFE。该技巧出自[《Genie: Higher-order denoising diffusion solvers》](https://papers.cool/arxiv/2210.05475)，具体是指在扩散模型背景下我们发现$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_N}, t_N)$与$\boldsymbol{x}_{t_N}$非常接近（不同的扩散模型表现可能不大一样，但核心思想都是第一步可以直接解析求解），于是在采样的第一步直接用$\boldsymbol{x}_{t_N}$替代$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_N}, t_N)$，这就省了一个NFE。论文附录的Table 8、Table 9、Table 10也更详尽地评估了AFS对效果的影响，有兴趣的读者可以自行分析。

最后，由于AMED使用了蒸馏的方法来训练$g_{\boldsymbol{\phi}}$，那么也许会有读者想知道它跟其他蒸馏加速的方案的效果差异，不过很遗憾，论文没有提供相关对比。为此我也邮件咨询过作者，作者表示AMED的蒸馏成本是极低的，CIFAR10只需要在单张A100上训练不到20分钟，256大小的图片也只需要在4张A100上训练几个小时，而相比之下其他蒸馏加速的思路需要的时间是数天甚至数十天，因此作者将AMED视为Solver的工作而不是蒸馏的工作。不过作者也表示，后面有机会也尽可能补上跟蒸馏工作的对比。

## 假设分析 #

前面在讨论中值定理到向量函数的推广时，我们提到“一定的假设之下”，那么这里的假设是什么呢？是否真的成立呢？

不难举出反例证明，即便是二维函数积分中值定理都不恒成立，换言之积分中值定理只在一维函数上成立，这意味着如果高维函数成立积分中值定理，那么该函数所描述的空间轨迹只能是一条直线，也就是说采样过程中所有的$\boldsymbol{x}_{t_0},\boldsymbol{x}_{t_1},\cdots,\boldsymbol{x}_{t_N}$构成一条直线。这个假设自然非常强，实际上几乎不可能成立，但也侧面告诉我们，要想积分中值定理在高维空间尽可能成立，那么采样轨迹要保持在一个尽可能低维的子空间中。

为了验证这一点，论文作者加大了采样步数得到了较为精确的采样轨迹，然后对轨迹做主成分分析，结果如下图所示：  


[![扩散ODE采样轨迹的主成分分析](/usr/uploads/2023/12/3425568740.png)](/usr/uploads/2023/12/3425568740.png "点击查看原图")

扩散ODE采样轨迹的主成分分析

主成分分析的结果显示，只保留top1的主成分，就可以保留轨迹的大部分精度，而同时保留前两个主成本，那么后面的误差几乎可以忽略了，这告诉我们采样轨迹几乎都集中在一个二维子平面上，甚至非常接近这个子平面上的的一个直线，于是在$t_{n+1}-t_n$并不是特别大的时候，扩散模型的高维空间的积分中值定理也近似成立。

这个结果可能会让人比较意外，但事后来看其实也能解释：在[《生成扩散模型漫谈（十五）：构建ODE的一般步骤（中）》](/archives/9379)、[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](/archives/9497)我们介绍了先指定$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的“伪轨迹”，然后再构建对应的扩散ODE的一般步骤，而实际应用中，我们所构建的“伪轨迹”都是$\boldsymbol{x}_T$与$\boldsymbol{x}_0$的线性插值（关于$t$可能是非线性的，关于$\boldsymbol{x}_T$和$\boldsymbol{x}_0$则是线性的），于是构建的“伪轨迹”都是直线，这会进一步鼓励真实的扩散轨迹是一条直线，这就解释了主成分分析的结果。

## 文章小结 #

本文简单回顾了扩散ODE的采样加速方法，并重点介绍了前两天刚发布的一个名为“AMED”的新颖加速采样方案，该Solver类比了积分中值定理来构建迭代格式，以极低的蒸馏成本提高了Solver在低NFE时的表现。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9881>_

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

苏剑林. (Dec. 07, 2023). 《生成扩散模型漫谈（二十一）：中值定理加速ODE采样 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9881>

@online{kexuefm-9881,  
title={生成扩散模型漫谈（二十一）：中值定理加速ODE采样},  
author={苏剑林},  
year={2023},  
month={Dec},  
url={\url{https://spaces.ac.cn/archives/9881}},  
} 


---

## 推导

本节提供关于中值定理加速ODE采样的详细数学推导，从微积分基础到扩散模型的具体应用，全方位解析各种数值方法的原理、误差分析及其在生成模型中的实践。

### 1. 微积分中值定理回顾

#### 1.1 积分中值定理

**定理（积分第一中值定理）**：设函数$f(x)$在闭区间$[a,b]$上连续，则至少存在一点$\xi\in[a,b]$，使得
$$\int_a^b f(x)dx = f(\xi)(b-a)$$

**证明**：由连续函数的性质，$f(x)$在$[a,b]$上必有最大值$M$和最小值$m$。因此对任意$x\in[a,b]$，有
$$m \leq f(x) \leq M$$

对上式在$[a,b]$上积分，得
$$m(b-a) \leq \int_a^b f(x)dx \leq M(b-a)$$

即
$$m \leq \frac{1}{b-a}\int_a^b f(x)dx \leq M$$

记$\mu = \frac{1}{b-a}\int_a^b f(x)dx$，则$m\leq \mu \leq M$。由连续函数的介值定理，存在$\xi\in[a,b]$使得$f(\xi)=\mu$，即
$$f(\xi) = \frac{1}{b-a}\int_a^b f(x)dx$$

整理即得原式。$\square$

**推广（积分第二中值定理）**：若$f(x)$在$[a,b]$上连续，$g(x)$在$[a,b]$上可积且不变号，则存在$\xi\in[a,b]$使得
$$\int_a^b f(x)g(x)dx = f(\xi)\int_a^b g(x)dx$$

这个定理可以理解为带权重的中值定理。特别地，当$g(x)\equiv 1$时，即回到第一中值定理。

#### 1.2 微分中值定理

**定理（Lagrange中值定理）**：若函数$f(x)$在$[a,b]$上连续，在$(a,b)$内可导，则存在$\xi\in(a,b)$使得
$$f(b) - f(a) = f'(\xi)(b-a)$$

这个定理告诉我们，函数在两点间的平均变化率等于某个中间点的瞬时变化率。

**与积分中值定理的联系**：由微积分基本定理，
$$f(b) - f(a) = \int_a^b f'(x)dx$$

结合积分中值定理，存在$\xi\in[a,b]$使得
$$\int_a^b f'(x)dx = f'(\xi)(b-a)$$

因此$f(b) - f(a) = f'(\xi)(b-a)$，这正是Lagrange中值定理。

#### 1.3 向量值函数的中值定理

对于向量值函数$\boldsymbol{f}:[a,b]\to\mathbb{R}^d$，积分中值定理**不一定成立**。

**反例**：考虑二维函数$\boldsymbol{f}(t) = (\cos t, \sin t)$，$t\in[0,2\pi]$。则
$$\int_0^{2\pi}\boldsymbol{f}(t)dt = \left(\int_0^{2\pi}\cos t\,dt, \int_0^{2\pi}\sin t\,dt\right) = (0, 0)$$

但对任意$\xi\in[0,2\pi]$，有$\|\boldsymbol{f}(\xi)\| = 1 \neq 0$，因此不存在$\xi$使得
$$\boldsymbol{f}(\xi) = \frac{1}{2\pi}\int_0^{2\pi}\boldsymbol{f}(t)dt = \boldsymbol{0}$$

**近似成立的条件**：尽管严格的中值定理不成立，但在以下情况下可以近似成立：

1. **轨迹近似直线**：若$\boldsymbol{f}(t) \approx \boldsymbol{f}(a) + \frac{t-a}{b-a}(\boldsymbol{f}(b)-\boldsymbol{f}(a))$，即轨迹接近直线段，则中值定理近似成立。

2. **低维子空间**：若$\boldsymbol{f}(t)$的值域主要集中在一个低维子空间中，特别是一维子空间（直线），则近似程度更好。

3. **步长充分小**：当$b-a$足够小时，$\boldsymbol{f}(t)$在$[a,b]$上的变化可以用线性近似，此时中值定理近似成立。

这些观察对于理解扩散ODE中AMED方法的有效性至关重要。

### 2. ODE数值积分基础理论

#### 2.1 问题描述

考虑一阶常微分方程的初值问题（IVP）：
$$\begin{cases}
\frac{d\boldsymbol{x}}{dt} = \boldsymbol{f}(\boldsymbol{x}, t), & t\in[t_0, T]\\
\boldsymbol{x}(t_0) = \boldsymbol{x}_0
\end{cases}$$

其中$\boldsymbol{x}\in\mathbb{R}^d$，$\boldsymbol{f}:\mathbb{R}^d\times\mathbb{R}\to\mathbb{R}^d$。

**解的存在唯一性**：在Lipschitz条件下（即存在常数$L$使得$\|\boldsymbol{f}(\boldsymbol{x}_1,t) - \boldsymbol{f}(\boldsymbol{x}_2,t)\| \leq L\|\boldsymbol{x}_1-\boldsymbol{x}_2\|$），初值问题有唯一解。

**数值求解的基本思想**：
1. 将时间区间离散化：$t_0 < t_1 < \cdots < t_N = T$
2. 定义步长：$h_n = t_{n+1} - t_n$
3. 用离散点上的值$\boldsymbol{x}_n \approx \boldsymbol{x}(t_n)$逼近真实解
4. 构造递推关系：$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{\Phi}(\boldsymbol{x}_n, t_n, h_n)$

其中$\boldsymbol{\Phi}$称为**增量函数**，不同的数值方法对应不同的$\boldsymbol{\Phi}$。

#### 2.2 Taylor展开与局部截断误差

设$\boldsymbol{x}(t)$是精确解，在$t_n$处进行Taylor展开：
$$\boldsymbol{x}(t_{n+1}) = \boldsymbol{x}(t_n) + h_n\boldsymbol{x}'(t_n) + \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \frac{h_n^3}{6}\boldsymbol{x}'''(t_n) + \mathcal{O}(h_n^4)$$

由于$\boldsymbol{x}'(t) = \boldsymbol{f}(\boldsymbol{x}(t), t)$，我们可以计算高阶导数：

**一阶导数**：
$$\boldsymbol{x}'(t) = \boldsymbol{f}(\boldsymbol{x}(t), t)$$

**二阶导数**：
$$\boldsymbol{x}''(t) = \frac{d}{dt}\boldsymbol{f}(\boldsymbol{x}(t), t) = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\boldsymbol{x}'(t) + \frac{\partial \boldsymbol{f}}{\partial t} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\boldsymbol{f} + \frac{\partial \boldsymbol{f}}{\partial t}$$

**三阶导数**：通过链式法则继续求导，
$$\boldsymbol{x}'''(t) = \frac{\partial^2\boldsymbol{f}}{\partial \boldsymbol{x}^2}(\boldsymbol{f}, \boldsymbol{f}) + \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}\left(\frac{\partial\boldsymbol{f}}{\partial \boldsymbol{x}}\boldsymbol{f} + \frac{\partial\boldsymbol{f}}{\partial t}\right) + \frac{\partial^2\boldsymbol{f}}{\partial \boldsymbol{x}\partial t}\boldsymbol{f} + \frac{\partial^2\boldsymbol{f}}{\partial t^2}$$

**局部截断误差（Local Truncation Error, LTE）**：定义为在假设$\boldsymbol{x}_n = \boldsymbol{x}(t_n)$精确的情况下，单步计算产生的误差：
$$\boldsymbol{\tau}_{n+1} = \boldsymbol{x}(t_{n+1}) - [\boldsymbol{x}_n + h_n\boldsymbol{\Phi}(\boldsymbol{x}_n, t_n, h_n)]$$

若$\boldsymbol{\tau}_{n+1} = \mathcal{O}(h_n^{p+1})$，则称该方法具有**p阶精度**。

#### 2.3 全局误差分析

**全局误差**定义为$\boldsymbol{e}_n = \boldsymbol{x}(t_n) - \boldsymbol{x}_n$，是累积误差。

设数值方法为
$$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{\Phi}(\boldsymbol{x}_n, t_n, h_n)$$

则全局误差满足递推关系：
$$\boldsymbol{e}_{n+1} = \boldsymbol{x}(t_{n+1}) - \boldsymbol{x}_{n+1} = \boldsymbol{x}(t_{n+1}) - \boldsymbol{x}_n - h_n\boldsymbol{\Phi}(\boldsymbol{x}_n, t_n, h_n)$$

代入$\boldsymbol{x}_n = \boldsymbol{x}(t_n) - \boldsymbol{e}_n$：
$$\boldsymbol{e}_{n+1} = \boldsymbol{x}(t_{n+1}) - \boldsymbol{x}(t_n) + \boldsymbol{e}_n - h_n\boldsymbol{\Phi}(\boldsymbol{x}(t_n)-\boldsymbol{e}_n, t_n, h_n)$$

利用Taylor展开和Lipschitz条件，可以证明：

**定理**：若数值方法的局部截断误差为$\mathcal{O}(h^{p+1})$，且增量函数$\boldsymbol{\Phi}$关于$\boldsymbol{x}$满足Lipschitz条件，则全局误差为$\mathcal{O}(h^p)$。

即：**局部p+1阶精度 $\Rightarrow$ 全局p阶精度**。

**证明梗概**：假设步长固定为$h$，总步数为$N = (T-t_0)/h$。每步的局部误差为$\mathcal{O}(h^{p+1})$，累积$N$步后：
$$\|\boldsymbol{e}_N\| \leq C_1 Nh^{p+1} = C_1\frac{T-t_0}{h}h^{p+1} = C_1(T-t_0)h^p = \mathcal{O}(h^p)$$

这说明局部误差的阶数会在全局累积时降低一阶。

### 3. 经典数值方法详解

#### 3.1 欧拉方法（Euler Method）

**显式欧拉方法（前向欧拉）**：
$$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{f}(\boldsymbol{x}_n, t_n)$$

**推导**：将ODE积分形式
$$\boldsymbol{x}(t_{n+1}) = \boldsymbol{x}(t_n) + \int_{t_n}^{t_{n+1}}\boldsymbol{f}(\boldsymbol{x}(t), t)dt$$

用最简单的左端点矩形公式近似积分：
$$\int_{t_n}^{t_{n+1}}\boldsymbol{f}(\boldsymbol{x}(t), t)dt \approx \boldsymbol{f}(\boldsymbol{x}(t_n), t_n)(t_{n+1}-t_n)$$

**局部截断误差分析**：
$$\begin{align}
\boldsymbol{\tau}_{n+1} &= \boldsymbol{x}(t_{n+1}) - [\boldsymbol{x}(t_n) + h_n\boldsymbol{f}(\boldsymbol{x}(t_n), t_n)]\\
&= \boldsymbol{x}(t_n) + h_n\boldsymbol{x}'(t_n) + \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \mathcal{O}(h_n^3) - \boldsymbol{x}(t_n) - h_n\boldsymbol{f}(\boldsymbol{x}(t_n), t_n)\\
&= \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \mathcal{O}(h_n^3)\\
&= \mathcal{O}(h_n^2)
\end{align}$$

因此欧拉方法是**1阶方法**（局部2阶，全局1阶）。

**在扩散模型中的应用**：对于扩散ODE（记为逆向时间）
$$\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t)$$

从$t_{n+1}$到$t_n$（$t_{n+1} > t_n$），欧拉法给出：
$$\boldsymbol{x}_{t_n} = \boldsymbol{x}_{t_{n+1}} + (t_n - t_{n+1})\boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_{n+1}}, t_{n+1}) = \boldsymbol{x}_{t_{n+1}} - h_n\boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})$$

其中$h_n = t_{n+1} - t_n > 0$。这正是DDIM采样的形式。

**隐式欧拉方法（后向欧拉）**：
$$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{f}(\boldsymbol{x}_{n+1}, t_{n+1})$$

这是一个关于$\boldsymbol{x}_{n+1}$的隐式方程，需要迭代求解（如Newton迭代），但稳定性更好，特别适合刚性（stiff）问题。

#### 3.2 中点方法（Midpoint Method）

**中点方法**：
$$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{f}\left(\boldsymbol{x}_{n+1/2}, t_{n+1/2}\right)$$

其中$t_{n+1/2} = \frac{t_n + t_{n+1}}{2}$，中点值$\boldsymbol{x}_{n+1/2}$需要先估计。

**显式中点法（预测-校正）**：
$$\begin{cases}
\boldsymbol{x}_{n+1/2}^* = \boldsymbol{x}_n + \frac{h_n}{2}\boldsymbol{f}(\boldsymbol{x}_n, t_n) & \text{(预测)}\\
\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{f}(\boldsymbol{x}_{n+1/2}^*, t_{n+1/2}) & \text{(校正)}
\end{cases}$$

**几何意义**：用区间中点的斜率代表整个区间的平均斜率，比端点更准确。

**精度分析**：将$\boldsymbol{x}(t_{n+1})$和$\boldsymbol{f}(\boldsymbol{x}_{n+1/2}^*, t_{n+1/2})$都在$t_n$处展开：

首先，$\boldsymbol{x}_{n+1/2}^*$的误差为：
$$\boldsymbol{x}_{n+1/2}^* = \boldsymbol{x}(t_n) + \frac{h_n}{2}\boldsymbol{f}(\boldsymbol{x}(t_n), t_n)$$

而真实的$\boldsymbol{x}(t_{n+1/2})$为：
$$\boldsymbol{x}(t_{n+1/2}) = \boldsymbol{x}(t_n) + \frac{h_n}{2}\boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \frac{1}{2}\left(\frac{h_n}{2}\right)^2\boldsymbol{x}''(t_n) + \mathcal{O}(h_n^3)$$

因此$\boldsymbol{x}_{n+1/2}^* = \boldsymbol{x}(t_{n+1/2}) + \mathcal{O}(h_n^2)$。

接着展开$\boldsymbol{f}(\boldsymbol{x}_{n+1/2}^*, t_{n+1/2})$：
$$\begin{align}
\boldsymbol{f}(\boldsymbol{x}_{n+1/2}^*, t_{n+1/2}) &= \boldsymbol{f}(\boldsymbol{x}(t_{n+1/2}), t_{n+1/2}) + \mathcal{O}(h_n^2)\\
&= \boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \frac{h_n}{2}\boldsymbol{x}''(t_n) + \mathcal{O}(h_n^2)
\end{align}$$

代入中点公式：
$$\begin{align}
\boldsymbol{x}_{n+1} &= \boldsymbol{x}(t_n) + h_n\left[\boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \frac{h_n}{2}\boldsymbol{x}''(t_n) + \mathcal{O}(h_n^2)\right]\\
&= \boldsymbol{x}(t_n) + h_n\boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \mathcal{O}(h_n^3)
\end{align}$$

与Taylor展开比较：
$$\boldsymbol{x}(t_{n+1}) = \boldsymbol{x}(t_n) + h_n\boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \frac{h_n^3}{6}\boldsymbol{x}'''(t_n) + \mathcal{O}(h_n^4)$$

因此局部截断误差为：
$$\boldsymbol{\tau}_{n+1} = \frac{h_n^3}{6}\boldsymbol{x}'''(t_n) + \mathcal{O}(h_n^4) = \mathcal{O}(h_n^3)$$

**中点法是2阶方法**（局部3阶，全局2阶）。

#### 3.3 梯形法则（Trapezoidal Rule）

**梯形法则**基于用梯形面积近似积分：
$$\int_{t_n}^{t_{n+1}}\boldsymbol{f}(\boldsymbol{x}(t), t)dt \approx \frac{h_n}{2}[\boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \boldsymbol{f}(\boldsymbol{x}(t_{n+1}), t_{n+1})]$$

**隐式梯形法**：
$$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \frac{h_n}{2}[\boldsymbol{f}(\boldsymbol{x}_n, t_n) + \boldsymbol{f}(\boldsymbol{x}_{n+1}, t_{n+1})]$$

这是隐式方程，需要迭代求解。

**显式梯形法（Heun方法的一种形式）**：
$$\begin{cases}
\tilde{\boldsymbol{x}}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{f}(\boldsymbol{x}_n, t_n) & \text{(预测)}\\
\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \frac{h_n}{2}[\boldsymbol{f}(\boldsymbol{x}_n, t_n) + \boldsymbol{f}(\tilde{\boldsymbol{x}}_{n+1}, t_{n+1})] & \text{(校正)}
\end{cases}$$

**精度分析**：对隐式梯形法，设$\boldsymbol{x}_n = \boldsymbol{x}(t_n)$，则
$$\begin{align}
\boldsymbol{\tau}_{n+1} &= \boldsymbol{x}(t_{n+1}) - \boldsymbol{x}(t_n) - \frac{h_n}{2}[\boldsymbol{f}(\boldsymbol{x}(t_n), t_n) + \boldsymbol{f}(\boldsymbol{x}(t_{n+1}), t_{n+1})]\\
&= h_n\boldsymbol{x}'(t_n) + \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \frac{h_n^3}{6}\boldsymbol{x}'''(t_n) + \mathcal{O}(h_n^4)\\
&\quad - \frac{h_n}{2}[\boldsymbol{x}'(t_n) + \boldsymbol{x}'(t_{n+1})]
\end{align}$$

其中
$$\boldsymbol{x}'(t_{n+1}) = \boldsymbol{x}'(t_n) + h_n\boldsymbol{x}''(t_n) + \frac{h_n^2}{2}\boldsymbol{x}'''(t_n) + \mathcal{O}(h_n^3)$$

代入得：
$$\begin{align}
\boldsymbol{\tau}_{n+1} &= h_n\boldsymbol{x}'(t_n) + \frac{h_n^2}{2}\boldsymbol{x}''(t_n) + \frac{h_n^3}{6}\boldsymbol{x}'''(t_n)\\
&\quad - \frac{h_n}{2}\left[2\boldsymbol{x}'(t_n) + h_n\boldsymbol{x}''(t_n) + \frac{h_n^2}{2}\boldsymbol{x}'''(t_n)\right] + \mathcal{O}(h_n^4)\\
&= -\frac{h_n^3}{12}\boldsymbol{x}'''(t_n) + \mathcal{O}(h_n^4) = \mathcal{O}(h_n^3)
\end{align}$$

**梯形法是2阶方法**，且误差系数比中点法小。

**在扩散模型中的应用**：对于扩散ODE的逆向采样，梯形法给出：
$$\boldsymbol{x}_{t_n} = \boldsymbol{x}_{t_{n+1}} - \frac{h_n}{2}[\boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_n}, t_n) + \boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})]$$

由于右端含有$\boldsymbol{x}_{t_n}$，需要用欧拉法预测：
$$\begin{cases}
\tilde{\boldsymbol{x}}_{t_n} = \boldsymbol{x}_{t_{n+1}} - h_n\boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})\\
\boldsymbol{x}_{t_n} = \boldsymbol{x}_{t_{n+1}} - \frac{h_n}{2}[\boldsymbol{v}_{\theta}(\tilde{\boldsymbol{x}}_{t_n}, t_n) + \boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})]
\end{cases}$$

这正是文中提到的Heun方法。

#### 3.4 Heun方法详解

**Heun方法**（也称改进欧拉法）是预测-校正方法的典型代表：

**算法步骤**：
1. **预测**（Predictor）：用欧拉法预测$\boldsymbol{x}_{n+1}$：
   $$\tilde{\boldsymbol{x}}_{n+1} = \boldsymbol{x}_n + h_n\boldsymbol{f}(\boldsymbol{x}_n, t_n)$$

2. **校正**（Corrector）：用梯形法校正：
   $$\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \frac{h_n}{2}[\boldsymbol{f}(\boldsymbol{x}_n, t_n) + \boldsymbol{f}(\tilde{\boldsymbol{x}}_{n+1}, t_{n+1})]$$

**几何解释**：
- 预测步用起点的斜率外推
- 校正步用起点和预测终点的平均斜率修正
- 相当于用梯形面积代替矩形面积

**与RK2的关系**：Heun方法是2阶Runge-Kutta方法（RK2）的一种形式。标准RK2形式为：
$$\begin{cases}
\boldsymbol{k}_1 = \boldsymbol{f}(\boldsymbol{x}_n, t_n)\\
\boldsymbol{k}_2 = \boldsymbol{f}(\boldsymbol{x}_n + h_n\boldsymbol{k}_1, t_{n+1})\\
\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \frac{h_n}{2}(\boldsymbol{k}_1 + \boldsymbol{k}_2)
\end{cases}$$

这与Heun方法完全等价。

**在扩散模型中的优势**：
1. **每步需要2次NFE**，比1阶方法多1次
2. **精度提升显著**，可以用更大步长
3. **总NFE可以减少**，因为所需步数大幅降低
4. **实现简单**，不需要复杂的自适应控制

例如，对于扩散ODE，若欧拉法需要100步达到某精度，Heun方法可能只需25步，虽然每步2次NFE，总NFE从100降到50，加速2倍。

#### 3.5 Runge-Kutta方法

**经典4阶Runge-Kutta方法（RK4）**：
$$\begin{cases}
\boldsymbol{k}_1 = \boldsymbol{f}(\boldsymbol{x}_n, t_n)\\
\boldsymbol{k}_2 = \boldsymbol{f}(\boldsymbol{x}_n + \frac{h_n}{2}\boldsymbol{k}_1, t_n + \frac{h_n}{2})\\
\boldsymbol{k}_3 = \boldsymbol{f}(\boldsymbol{x}_n + \frac{h_n}{2}\boldsymbol{k}_2, t_n + \frac{h_n}{2})\\
\boldsymbol{k}_4 = \boldsymbol{f}(\boldsymbol{x}_n + h_n\boldsymbol{k}_3, t_{n+1})\\
\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \frac{h_n}{6}(\boldsymbol{k}_1 + 2\boldsymbol{k}_2 + 2\boldsymbol{k}_3 + \boldsymbol{k}_4)
\end{cases}$$

**推导思想**：用Simpson积分公式：
$$\int_{t_n}^{t_{n+1}}\boldsymbol{f}(t)dt \approx \frac{h_n}{6}[\boldsymbol{f}(t_n) + 4\boldsymbol{f}(t_{n+1/2}) + \boldsymbol{f}(t_{n+1})]$$

但$\boldsymbol{f}$依赖于$\boldsymbol{x}(t)$，需要多次预测中间点。

**精度**：RK4是**4阶方法**（局部5阶，全局4阶），局部截断误差为$\mathcal{O}(h^5)$。

**计算代价**：每步需要**4次NFE**，对扩散模型而言代价较高，通常2阶方法已足够。

**一般Runge-Kutta方法**：由Butcher表刻画，形式为：
$$\begin{cases}
\boldsymbol{k}_i = \boldsymbol{f}\left(\boldsymbol{x}_n + h_n\sum_{j=1}^{i-1}a_{ij}\boldsymbol{k}_j, t_n + c_ih_n\right), & i=1,\ldots,s\\
\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + h_n\sum_{i=1}^s b_i\boldsymbol{k}_i
\end{cases}$$

其中$s$是级数，$(a_{ij}, b_i, c_i)$由精度要求确定。

### 4. 误差理论深入分析

#### 4.1 局部截断误差的详细推导

考虑一般的s级Runge-Kutta方法，局部截断误差为：
$$\boldsymbol{\tau}_{n+1} = \boldsymbol{x}(t_{n+1}) - \left[\boldsymbol{x}(t_n) + h_n\sum_{i=1}^s b_i\boldsymbol{k}_i\right]$$

其中$\boldsymbol{k}_i$定义如前。要使方法达到p阶，需要满足一系列**阶条件**（order conditions）。

**1阶条件**（$\mathcal{O}(h)$）：
$$\sum_{i=1}^s b_i = 1$$

这确保了与Taylor展开的0阶项匹配。

**2阶条件**（$\mathcal{O}(h^2)$）：
$$\sum_{i=1}^s b_ic_i = \frac{1}{2}$$

这确保了与Taylor展开的1阶项匹配。

**3阶条件**（$\mathcal{O}(h^3)$）：需要满足
$$\sum_{i=1}^s b_ic_i^2 = \frac{1}{3}, \quad \sum_{i,j}b_ia_{ij}c_j = \frac{1}{6}$$

**验证Heun方法**：Heun方法对应$s=2$，
$$a_{11}=0, a_{12}=0, a_{21}=1, a_{22}=0, \quad b_1=\frac{1}{2}, b_2=\frac{1}{2}, \quad c_1=0, c_2=1$$

检验1阶条件：$b_1+b_2 = \frac{1}{2}+\frac{1}{2} = 1$ ✓

检验2阶条件：$b_1c_1+b_2c_2 = 0 + \frac{1}{2}\cdot 1 = \frac{1}{2}$ ✓

检验3阶条件：$b_1c_1^2+b_2c_2^2 = 0 + \frac{1}{2} = \frac{1}{2} \neq \frac{1}{3}$ ✗

因此Heun方法满足2阶条件，不满足3阶条件，确实是2阶方法。

#### 4.2 全局误差的传播

设第$n$步的全局误差为$\boldsymbol{e}_n = \boldsymbol{x}(t_n) - \boldsymbol{x}_n$，则
$$\begin{align}
\boldsymbol{e}_{n+1} &= \boldsymbol{x}(t_{n+1}) - \boldsymbol{x}_{n+1}\\
&= \boldsymbol{x}(t_{n+1}) - [\boldsymbol{x}_n + h_n\boldsymbol{\Phi}(\boldsymbol{x}_n, t_n, h_n)]\\
&= \boldsymbol{x}(t_{n+1}) - [\boldsymbol{x}(t_n) - \boldsymbol{e}_n + h_n\boldsymbol{\Phi}(\boldsymbol{x}(t_n)-\boldsymbol{e}_n, t_n, h_n)]
\end{align}$$

利用Taylor展开：
$$\boldsymbol{x}(t_{n+1}) = \boldsymbol{x}(t_n) + h_n\boldsymbol{\Phi}(\boldsymbol{x}(t_n), t_n, h_n) + \boldsymbol{\tau}_{n+1}$$

以及Lipschitz条件：
$$\|\boldsymbol{\Phi}(\boldsymbol{x}(t_n), t_n, h_n) - \boldsymbol{\Phi}(\boldsymbol{x}(t_n)-\boldsymbol{e}_n, t_n, h_n)\| \leq L\|\boldsymbol{e}_n\|$$

得到：
$$\|\boldsymbol{e}_{n+1}\| \leq \|\boldsymbol{e}_n\| + h_nL\|\boldsymbol{e}_n\| + \|\boldsymbol{\tau}_{n+1}\| = (1+h_nL)\|\boldsymbol{e}_n\| + \|\boldsymbol{\tau}_{n+1}\|$$

递推地，初始误差$\boldsymbol{e}_0 = 0$，则
$$\begin{align}
\|\boldsymbol{e}_N\| &\leq (1+h_{N-1}L)\|\boldsymbol{e}_{N-1}\| + \|\boldsymbol{\tau}_N\|\\
&\leq (1+h_{N-1}L)[(1+h_{N-2}L)\|\boldsymbol{e}_{N-2}\| + \|\boldsymbol{\tau}_{N-1}\|] + \|\boldsymbol{\tau}_N\|\\
&\leq \cdots\\
&\leq \sum_{n=1}^N \|\boldsymbol{\tau}_n\|\prod_{k=n}^{N-1}(1+h_kL)
\end{align}$$

若步长固定$h_n = h$，且局部误差$\|\boldsymbol{\tau}_n\| \leq Ch^{p+1}$，则
$$\begin{align}
\|\boldsymbol{e}_N\| &\leq NCh^{p+1}(1+hL)^{N-1}\\
&\leq NCh^{p+1}e^{(N-1)hL}\\
&\leq \frac{T-t_0}{h}Ch^{p+1}e^{(T-t_0)L}\\
&= C(T-t_0)e^{(T-t_0)L}h^p = \mathcal{O}(h^p)
\end{align}$$

其中用到了$1+x\leq e^x$和$Nh = T-t_0$。

**结论**：局部$p+1$阶方法的全局误差为$\mathcal{O}(h^p)$。

#### 4.3 稳定性分析

考虑测试方程（test equation）：
$$\frac{dx}{dt} = \lambda x, \quad \lambda\in\mathbb{C}$$

精确解为$x(t) = x_0 e^{\lambda t}$。当$\text{Re}(\lambda) < 0$时，解指数衰减。

**欧拉方法的稳定性**：应用欧拉法得
$$x_{n+1} = x_n + h\lambda x_n = (1+h\lambda)x_n$$

因此$x_n = (1+h\lambda)^n x_0$。要使数值解也衰减（$|x_n|\to 0$），需要
$$|1+h\lambda| < 1$$

这定义了**绝对稳定域**：$\{z\in\mathbb{C}: |1+z|<1\}$，其中$z=h\lambda$。

**A-稳定性**：若方法对所有$\text{Re}(h\lambda)<0$都稳定，称该方法**A-稳定**。

- 显式欧拉法：不是A-稳定
- 隐式欧拉法：A-稳定
- 梯形法：A-稳定
- RK4：不是A-稳定

对于扩散ODE，稳定性通常不是主要问题，因为我们更关心精度而非长时间行为。

### 5. Richardson外推法

#### 5.1 基本思想

Richardson外推是一种**后处理技术**，通过组合不同步长的数值解来提高精度。

**原理**：假设数值解的误差有渐近展开：
$$\boldsymbol{x}_h - \boldsymbol{x}^* = C_1h^p + C_2h^{p+1} + \cdots$$

其中$\boldsymbol{x}_h$是步长为$h$的数值解，$\boldsymbol{x}^*$是精确解。

若用步长$h$和$h/2$分别计算，得到$\boldsymbol{x}_h$和$\boldsymbol{x}_{h/2}$，则
$$\begin{cases}
\boldsymbol{x}_h = \boldsymbol{x}^* + C_1h^p + C_2h^{p+1} + \mathcal{O}(h^{p+2})\\
\boldsymbol{x}_{h/2} = \boldsymbol{x}^* + C_1(h/2)^p + C_2(h/2)^{p+1} + \mathcal{O}(h^{p+2})
\end{cases}$$

将第二式乘以$2^p$再减去第一式：
$$2^p\boldsymbol{x}_{h/2} - \boldsymbol{x}_h = (2^p-1)\boldsymbol{x}^* + C_2(2^{p+1}-2)h^{p+1}/2^{p+1} + \mathcal{O}(h^{p+2})$$

解出：
$$\boldsymbol{x}^* = \frac{2^p\boldsymbol{x}_{h/2} - \boldsymbol{x}_h}{2^p-1} + \mathcal{O}(h^{p+1})$$

定义**外推解**：
$$\boldsymbol{x}_h^{\text{ext}} = \frac{2^p\boldsymbol{x}_{h/2} - \boldsymbol{x}_h}{2^p-1}$$

其误差为$\mathcal{O}(h^{p+1})$，比原方法高一阶！

#### 5.2 应用于扩散ODE

**算法流程**：
1. 用步长$h$计算一次，得$\boldsymbol{x}_h$
2. 用步长$h/2$计算一次，得$\boldsymbol{x}_{h/2}$
3. 外推得高精度解$\boldsymbol{x}_h^{\text{ext}}$

**示例（欧拉法）**：$p=1$，外推公式为
$$\boldsymbol{x}_h^{\text{ext}} = \frac{2\boldsymbol{x}_{h/2} - \boldsymbol{x}_h}{2-1} = 2\boldsymbol{x}_{h/2} - \boldsymbol{x}_h$$

这将1阶欧拉法提升到2阶！

**示例（Heun法）**：$p=2$，外推公式为
$$\boldsymbol{x}_h^{\text{ext}} = \frac{4\boldsymbol{x}_{h/2} - \boldsymbol{x}_h}{4-1} = \frac{4\boldsymbol{x}_{h/2} - \boldsymbol{x}_h}{3}$$

这将2阶Heun法提升到3阶。

**计算成本**：
- 需要计算$\boldsymbol{x}_h$（1次粗步长）和$\boldsymbol{x}_{h/2}$（2次细步长）
- 对于欧拉法：总共3次NFE得到2阶精度
- 对于Heun法：总共6次NFE得到3阶精度

**优劣分析**：
- **优点**：不改变原方法，容易实现；可迭代应用（多级外推）
- **缺点**：需要额外计算；不能很好地适应变步长

在扩散模型中，Richardson外推较少使用，因为直接用2阶方法已经足够高效。

### 6. 自适应步长控制

#### 6.1 误差估计

**嵌入式Runge-Kutta方法**（Embedded RK）同时给出p阶和p+1阶的解，从而估计局部误差。

最著名的是**Dormand-Prince方法（DP45或DOPRI5）**，同时给出4阶和5阶解。

**算法**：
1. 计算p阶解$\boldsymbol{x}_{n+1}^{(p)}$和p+1阶解$\boldsymbol{x}_{n+1}^{(p+1)}$
2. 估计局部误差：$\boldsymbol{e}_{n+1} \approx \boldsymbol{x}_{n+1}^{(p+1)} - \boldsymbol{x}_{n+1}^{(p)}$
3. 计算误差范数：$\text{err} = \|\boldsymbol{e}_{n+1}\|$

**Heun方法的误差估计**：
- p阶解：Heun公式
- p-1阶解：欧拉公式（预测步）

误差估计：
$$\text{err} \approx \|\boldsymbol{x}_{n+1}^{\text{Heun}} - \tilde{\boldsymbol{x}}_{n+1}^{\text{Euler}}\|$$

#### 6.2 步长调整策略

设期望的局部误差容限为$\text{tol}$，当前步长为$h$，误差为$\text{err}$。

**基本调整公式**：由于$\text{err} \sim h^{p+1}$，若要使误差达到$\text{tol}$，新步长应为
$$h_{\text{new}} = h\left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}$$

**安全因子**：为防止步长变化过快，引入安全系数$\beta\in(0,1)$（通常取0.9）：
$$h_{\text{new}} = \beta h\left(\frac{\text{tol}}{\text{err}}\right)^{1/(p+1)}$$

**步长限制**：
$$h_{\text{new}} = \max(h_{\min}, \min(h_{\max}, h_{\text{new}}))$$

其中$h_{\min}$和$h_{\max}$是人为设定的步长范围。

**步长变化限制**：防止步长剧烈变化：
$$\frac{h_{\min}}{h} \leq \frac{h_{\text{new}}}{h} \leq \frac{h_{\max}}{h}$$

典型取值：$h_{\min}/h = 0.2$，$h_{\max}/h = 5$。

#### 6.3 PI控制器

更先进的步长控制采用**PI控制器**（Proportional-Integral Controller）：
$$h_{n+1} = h_n\left(\frac{\text{tol}}{\text{err}_n}\right)^{k_P}\left(\frac{\text{err}_{n-1}}{\text{err}_n}\right)^{k_I}$$

其中$k_P$和$k_I$是控制参数，典型取值：
$$k_P = \frac{0.7}{p+1}, \quad k_I = \frac{0.4}{p+1}$$

**优点**：平滑步长变化，避免振荡。

#### 6.4 在扩散ODE中的应用

对于扩散模型，自适应步长的应用较为特殊：

**挑战**：
1. 扩散ODE在不同时间段的"刚度"不同（$t$接近0或$T$时变化更快）
2. 需要控制总NFE而非总时间
3. 图像质量指标（如FID）不完全等同于数值误差

**可能的策略**：
1. **预先规划节点**：通过实验确定最优节点分布（如EDM的节点公式）
2. **分段策略**：在$t$接近边界时用小步长，中间用大步长
3. **基于梯度的调整**：在$\|\boldsymbol{v}_{\theta}(\boldsymbol{x}_t,t)\|$大的区域用小步长

**实践中**：大多数工作采用固定节点，因为：
- 简单易实现
- 可预测NFE
- 节点分布可通过超参数搜索优化

### 7. 中值定理在扩散ODE中的应用

#### 7.1 AMED方法的数学基础

回顾扩散ODE的积分形式：
$$\boldsymbol{x}_{t_n} = \boldsymbol{x}_{t_{n+1}} + \int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t)dt$$

（注意积分方向是从$t_{n+1}$到$t_n$，因为$t_{n+1} > t_n$，符号已包含在方程定义中）

**标量情况的中值定理**：若$v$是标量函数，则存在$s\in(t_n, t_{n+1})$使得
$$\int_{t_n}^{t_{n+1}} v(x_t, t)dt = v(x_s, s)(t_{n+1} - t_n)$$

**向量情况的推广**：虽然严格的中值定理不成立，但我们可以寻找"最优中值点"$s^*$使得
$$\left\|\int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t)dt - \boldsymbol{v}_{\theta}(\boldsymbol{x}_{s^*}, s^*)(t_{n+1}-t_n)\right\|$$
最小化。

**AMED的近似**：假设存在$s_n\approx s^*$使得
$$\int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t)dt \approx \boldsymbol{v}_{\theta}(\boldsymbol{x}_{s_n}, s_n)(t_{n+1}-t_n)$$

从而
$$\boldsymbol{x}_{t_n} \approx \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\theta}(\boldsymbol{x}_{s_n}, s_n)(t_{n+1}-t_n)$$

#### 7.2 中值点的学习

**问题**：如何确定$s_n$和$\boldsymbol{x}_{s_n}$？

**AMED的解决方案**：
1. **学习时间点$s_n$**：用神经网络$g_{\phi}$预测
   $$s_n = g_{\phi}(\boldsymbol{h}_{t_{n+1}}, t_{n+1})$$
   其中$\boldsymbol{h}_{t_{n+1}}$是U-Net中间特征。

2. **预测空间点$\boldsymbol{x}_{s_n}$**：用欧拉法从$\boldsymbol{x}_{t_{n+1}}$外推
   $$\tilde{\boldsymbol{x}}_{s_n} = \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\theta}(\boldsymbol{x}_{t_{n+1}}, t_{n+1})(t_{n+1} - s_n)$$

**训练目标**：最小化与高精度轨迹的差距。设用高精度solver（如大步数Heun）得到参考轨迹$\{\bar{\boldsymbol{x}}_{t_0}, \bar{\boldsymbol{x}}_{t_1}, \ldots, \bar{\boldsymbol{x}}_{t_N}\}$，则损失函数为
$$\mathcal{L}(\phi) = \mathbb{E}\left[\left\|\boldsymbol{x}_{t_n}^{\text{AMED}} - \bar{\boldsymbol{x}}_{t_n}\right\|^2\right]$$

其中$\boldsymbol{x}_{t_n}^{\text{AMED}}$是用AMED-Solver计算的结果：
$$\boldsymbol{x}_{t_n}^{\text{AMED}} = \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\theta}(\tilde{\boldsymbol{x}}_{s_n}, s_n)(t_{n+1}-t_n)$$

#### 7.3 误差分析

**假设**：扩散轨迹近似为直线，即存在$\boldsymbol{x}_0, \boldsymbol{x}_T$使得
$$\boldsymbol{x}_t \approx \boldsymbol{x}_T + \frac{t-T}{0-T}(\boldsymbol{x}_0 - \boldsymbol{x}_T) = \boldsymbol{x}_T + \frac{T-t}{T}(\boldsymbol{x}_0 - \boldsymbol{x}_T)$$

则速度场近似为
$$\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t) = \frac{d\boldsymbol{x}_t}{dt} \approx -\frac{\boldsymbol{x}_0 - \boldsymbol{x}_T}{T} = \text{const}$$

在这种情况下，积分精确为：
$$\int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t)dt = \boldsymbol{v}_{\theta}(t_{n+1}-t_n)$$

此时任意$s_n\in[t_n, t_{n+1}]$都是精确的中值点。

**实际情况**：轨迹不完全是直线，但PCA分析显示主要在低维子空间中，因此近似程度较好。

设速度场的变化率为
$$\left\|\frac{\partial \boldsymbol{v}_{\theta}}{\partial t}\right\| \leq C_v$$

则中值近似的误差为
$$\left\|\int_{t_n}^{t_{n+1}}\boldsymbol{v}_{\theta}(\boldsymbol{x}_t, t)dt - \boldsymbol{v}_{\theta}(\boldsymbol{x}_{s_n}, s_n)(t_{n+1}-t_n)\right\| \leq C_v\frac{(t_{n+1}-t_n)^2}{2}$$

这是$\mathcal{O}(h^2)$的误差，与1阶方法相当。

**AMED的优势**：通过学习最优$s_n$，可以使常数因子更小，从而在相同步长下获得更高精度。

#### 7.4 与其他方法的比较

**AMED vs 欧拉法**：
- 欧拉法：$s_n = t_{n+1}$（固定端点）
- AMED：$s_n$是学习得到的最优点
- 精度：AMED更高（通过优化$s_n$减小误差常数）
- NFE：都是每步2次（AMED需要额外计算$s_n$，但计算量小）

**AMED vs Heun法**：
- Heun：用两个端点的平均，$s_n$隐含在平均中
- AMED：显式学习$s_n$，更灵活
- 精度：理论上都是2阶，但AMED在扩散ODE上表现更好（针对性优化）
- NFE：都是每步2次

**AMED vs 中点法**：
- 中点法：$s_n = (t_n+t_{n+1})/2$（固定中点）
- AMED：$s_n$自适应
- 精度：中点法理论2阶，AMED实践更优
- NFE：都是每步2次

**关键区别**：AMED通过数据驱动的方式学习最优$s_n$，而传统方法使用固定规则（端点、中点、加权平均等）。对于扩散ODE这种特定结构，学习方法能更好地适应轨迹特性。

### 8. 与其他加速方法的对比

#### 8.1 方法分类

扩散模型加速方法大致可分为三类：

**1. 纯数值方法**（Training-free）：
- 欧拉法（DDIM）
- Heun法（EDM）
- 高阶RK方法
- DPM-Solver系列
- 优点：无需额外训练
- 缺点：精度受限于通用ODE理论

**2. 蒸馏方法**（Distillation-based）：
- Progressive Distillation
- Consistency Models
- Latent Consistency Models
- 优点：可达到极少步数（1-4步）
- 缺点：训练成本高（数天到数周）

**3. 混合方法**（Hybrid）：
- AMED（轻量蒸馏 + 数值方法）
- DPM-Solver++（分析解 + 数值方法）
- 优点：平衡精度和成本
- 缺点：实现复杂度较高

#### 8.2 DPM-Solver系列

**DPM-Solver的核心思想**：将扩散ODE改写为特殊形式，利用半线性结构求解析解。

对于VP（Variance Preserving）扩散，ODE可写为：
$$\frac{d\boldsymbol{x}_t}{dt} = f(t)\boldsymbol{x}_t + g(t)\boldsymbol{\epsilon}_{\theta}(\boldsymbol{x}_t, t)$$

其中$f(t), g(t)$是确定的系数函数。

**变量替换**：定义$\boldsymbol{y}_t = \boldsymbol{x}_t/\alpha_t$（其中$\alpha_t = e^{\int f(t)dt}$），方程变为
$$\frac{d\boldsymbol{y}_t}{dt} = \frac{g(t)}{\alpha_t}\boldsymbol{\epsilon}_{\theta}(\alpha_t\boldsymbol{y}_t, t)$$

**积分求解**：
$$\boldsymbol{y}_{t_n} = \boldsymbol{y}_{t_{n+1}} + \int_{t_{n+1}}^{t_n}\frac{g(t)}{\alpha_t}\boldsymbol{\epsilon}_{\theta}(\alpha_t\boldsymbol{y}_t, t)dt$$

对积分部分用数值方法（如Lagrange插值），得到高阶公式。

**DPM-Solver-2**（前文式$\eqref{eq:dpm-solver-2}$）：
$$\boldsymbol{x}_{t_n} \approx \boldsymbol{x}_{t_{n+1}} - \boldsymbol{v}_{\theta}(\boldsymbol{x}_{\sqrt{t_nt_{n+1}}}, \sqrt{t_nt_{n+1}})(t_{n+1}-t_n)$$

这里中点取几何平均而非算术平均，源于VP扩散的对数尺度性质。

**与AMED的对比**：
- DPM-Solver：基于数学分析，中点位置由理论确定
- AMED：基于数据学习，中点位置由训练优化
- DPM-Solver：适用于特定扩散形式（VP）
- AMED：更通用，可适应不同扩散形式

#### 8.3 一致性模型（Consistency Models）

**基本思想**：学习一个函数$f_{\theta}$，将轨迹上任意点映射到起点：
$$f_{\theta}(\boldsymbol{x}_t, t) \approx \boldsymbol{x}_0, \quad \forall t\in[0,T]$$

**一致性条件**：
$$f_{\theta}(\boldsymbol{x}_t, t) = f_{\theta}(\boldsymbol{x}_{t'}, t'), \quad \forall t, t'$$

即同一条轨迹上的所有点映射到相同的$\boldsymbol{x}_0$。

**训练**：通过蒸馏预训练扩散模型或直接从数据训练。

**采样**：一步即可：$\boldsymbol{x}_0 = f_{\theta}(\boldsymbol{x}_T, T)$。

**优缺点**：
- 优点：采样极快（1步）
- 缺点：训练成本高，质量略低于多步方法

**与AMED的对比**：
- Consistency Models：改变模型架构和训练范式
- AMED：保持原模型，仅优化采样器
- Consistency Models：1步采样
- AMED：仍需多步（5-10步），但每步更优

#### 8.4 定量对比

基于原论文Table 2，在CIFAR-10上（32×32图像）：

| 方法 | NFE=5 | NFE=9 | NFE=13 |
|------|-------|-------|--------|
| DDIM (1阶) | 67.82 | 38.59 | 25.50 |
| DPM-Solver (2阶) | 79.16 | 21.75 | 14.34 |
| Heun/EDM (2阶) | 75.12 | 17.43 | 11.28 |
| AMED (2阶+学习) | **16.70** | **8.94** | **7.12** |

（数值为FID，越低越好）

**关键观察**：
1. NFE=5时，2阶方法（DPM、Heun）甚至不如1阶DDIM，因为步长太大
2. AMED在所有NFE下都显著优于其他方法
3. NFE从9降到5，AMED的FID仅增加87%，而Heun增加331%

这显示AMED在低NFE区间的优势。

### 9. 扩散模型轨迹的几何性质

#### 9.1 主成分分析（PCA）

原论文对扩散轨迹进行PCA，验证中值定理近似成立的条件。

**设置**：
- 用高精度solver（NFE=1000）采样得到轨迹$\{\boldsymbol{x}_{t_0}, \boldsymbol{x}_{t_1}, \ldots, \boldsymbol{x}_{t_N}\}$
- 每个$\boldsymbol{x}_{t_i}\in\mathbb{R}^d$（如CIFAR-10，$d=32\times32\times3=3072$）
- 对轨迹点做PCA，分析主成分的贡献

**数学描述**：设轨迹矩阵为$\boldsymbol{X} = [\boldsymbol{x}_{t_0}, \ldots, \boldsymbol{x}_{t_N}]^T \in\mathbb{R}^{(N+1)\times d}$，中心化后：
$$\tilde{\boldsymbol{X}} = \boldsymbol{X} - \frac{1}{N+1}\sum_{i=0}^N \boldsymbol{x}_{t_i}$$

协方差矩阵：
$$\boldsymbol{C} = \frac{1}{N}\tilde{\boldsymbol{X}}^T\tilde{\boldsymbol{X}} \in\mathbb{R}^{d\times d}$$

特征值分解：$\boldsymbol{C} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^T$，其中$\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_d)$，$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$。

**第k主成分的贡献率**：
$$r_k = \frac{\lambda_k}{\sum_{i=1}^d\lambda_i}$$

**累积贡献率**：
$$R_K = \frac{\sum_{k=1}^K\lambda_k}{\sum_{i=1}^d\lambda_i}$$

$R_K$表示前$K$个主成分保留的方差比例。

#### 9.2 实验结果

根据原论文Figure（扩散ODE采样轨迹的主成分分析）：

**观察**：
1. **第1主成分**：$r_1 \approx 95\%$，即第1个主成分解释了95%的方差
2. **前2主成分**：$R_2 \approx 99\%$，前2个主成分几乎解释了全部方差
3. **后续主成分**：$r_k < 1\%$，$k\geq 3$，贡献极小

**结论**：扩散轨迹近似位于一个**2维子空间**中，甚至非常接近**1维直线**。

#### 9.3 理论解释

**为何轨迹接近直线？**

回顾扩散模型的构造（见文章提到的（十五）、（十七）篇）：

**伪轨迹**：构造扩散ODE时，通常设计伪轨迹为
$$\boldsymbol{x}_t^{\text{pseudo}} = \alpha(t)\boldsymbol{x}_0 + \beta(t)\boldsymbol{x}_T$$

其中$\alpha(t), \beta(t)$是时间的标量函数，满足边界条件$\alpha(0)=1, \beta(0)=0, \alpha(T)=0, \beta(T)=1$。

**线性插值性质**：对于固定的$\boldsymbol{x}_0, \boldsymbol{x}_T$，伪轨迹是连接两点的直线（参数化可能非线性，但轨迹本身是直线）。

**真实轨迹**：扩散ODE的设计目标是使真实轨迹接近伪轨迹，因此真实轨迹也倾向于接近直线。

**数学表述**：设$\boldsymbol{d} = \boldsymbol{x}_0 - \boldsymbol{x}_T$为方向向量，则
$$\boldsymbol{x}_t \approx \boldsymbol{x}_T + \gamma(t)\boldsymbol{d}$$

其中$\gamma(t):[0,T]\to[0,1]$是单调函数，$\gamma(0)=1, \gamma(T)=0$。

所有点都在向量$\boldsymbol{d}$张成的1维子空间中（加上偏移$\boldsymbol{x}_T$）。

#### 9.4 对AMED的启示

**中值定理成立的条件**：轨迹越接近直线，向量值函数的积分中值定理越接近成立。

**PCA结果的支持**：$R_1 \approx 95\%$说明轨迹主要在1维子空间中，因此中值定理近似成立，AMED的理论基础得到验证。

**步长的影响**：
- 步长小时，局部轨迹更接近直线，近似更好
- 步长大时，曲率效应显现，近似误差增大
- 这解释了为何AMED在NFE=5时优势最明显（步长大，传统方法误差大，AMED通过学习$s_n$能更好补偿）

### 10. 总结与展望

#### 10.1 主要方法对比总结

| 方法类型 | 代表方法 | 阶数 | NFE/步 | 优点 | 缺点 |
|---------|---------|------|--------|------|------|
| 1阶方法 | DDIM/欧拉 | 1 | 1 | 简单快速 | 精度低，需多步 |
| 2阶方法 | Heun/梯形 | 2 | 2 | 精度提升显著 | 仍需较多步数 |
| 高阶方法 | RK4 | 4 | 4 | 精度高 | NFE开销大 |
| 分析方法 | DPM-Solver | 2-3 | 1-2 | 利用扩散结构 | 限于特定形式 |
| 蒸馏方法 | Consistency | - | 1 | 采样极快 | 训练成本极高 |
| 混合方法 | AMED | 2+ | 2 | 低NFE下最优 | 需轻量训练 |

#### 10.2 理论要点回顾

1. **局部vs全局误差**：局部$p+1$阶方法的全局误差为$\mathcal{O}(h^p)$

2. **NFE与步数**：$\text{总NFE} = \text{步数} \times \text{每步NFE}$，高阶方法通过减少步数降低总NFE

3. **中值定理**：向量值函数的积分中值定理在轨迹接近直线时近似成立

4. **PCA分析**：扩散轨迹主要在低维（1-2维）子空间中，支持中值定理近似

5. **步长策略**：
   - 固定步长：实现简单，可预测
   - 自适应步长：精度更高，但复杂
   - 扩散模型多用固定节点（通过超参数优化）

#### 10.3 实践建议

**选择采样器的原则**：

1. **极低NFE（<5）**：
   - 首选AMED或蒸馏方法
   - 欧拉法基本不可用
   - 2阶方法效果有限

2. **低NFE（5-10）**：
   - AMED表现最佳
   - Heun/EDM是Training-free的好选择
   - DPM-Solver对VP扩散很有效

3. **中等NFE（10-50）**：
   - Heun/EDM已经足够好
   - 高阶方法（RK4）可能过度
   - 自适应步长可考虑

4. **高NFE（>50）**：
   - 欧拉法已经收敛
   - 无需复杂方法

**训练成本考虑**：
- AMED：几小时（可接受）
- Progressive Distillation：数天（较高）
- Consistency Models：数周（很高）

**应用场景**：
- **实时应用**（如交互式编辑）：需要NFE<10，选AMED或蒸馏
- **批量生成**（如数据集生成）：可用NFE=20-50，选Heun
- **研究评估**：可用NFE=100+，选欧拉法即可

#### 10.4 未来方向

**理论方向**：
1. 更深入理解扩散轨迹的几何性质（为何近似直线？）
2. 严格的误差界和收敛性证明
3. 向量值中值定理的充要条件

**方法方向**：
1. 结合AMED思想与DPM-Solver的分析结构
2. 可学习的高阶方法（3阶、4阶）
3. 多模态采样器（不同阶段用不同方法）

**应用方向**：
1. 视频扩散模型的加速（时空ODE）
2. 3D扩散模型的加速（高维ODE）
3. 条件生成的专用采样器

#### 10.5 关键数学工具总结

本文涉及的核心数学工具：

1. **微积分**：Taylor展开、中值定理、积分
2. **数值分析**：有限差分、Runge-Kutta方法、误差分析
3. **线性代数**：PCA、特征值分解、范数
4. **优化理论**：梯度下降、损失函数设计
5. **微分方程**：ODE理论、稳定性分析、Lipschitz条件

这些工具的综合应用展现了现代机器学习与经典数学的深度融合。

---

**附注**：以上推导展示了从基础的微积分中值定理到先进的扩散模型采样加速的完整理论脉络。通过数值分析、微分方程和机器学习的交叉视角，我们不仅理解了各种方法的原理，也看到了AMED等创新方法如何巧妙地结合数学洞察与数据驱动优化。这正是当代AI研究的魅力所在：深厚的数学基础与灵活的工程实践相辅相成，推动技术不断进步。

