---
title: Adam的epsilon如何影响学习率的Scaling Law？
slug: adam的epsilon如何影响学习率的scaling-law
date: 2024-11-18
tags: 详细推导, 梯度, 学习率, 优化器, 尺度定律, 生成模型
status: completed
---
# Adam的epsilon如何影响学习率的Scaling Law？

**原文链接**: [https://spaces.ac.cn/archives/10563](https://spaces.ac.cn/archives/10563)

**发布日期**: 

---

上一篇文章[《当Batch Size增大时，学习率该如何随之变化？》](/archives/10542)我们从多个角度讨论了学习率与Batch Size之间的缩放规律，其中对于Adam优化器我们采用了SignSGD近似，这是分析Adam优化器常用的手段。那么一个很自然的问题就是：用SignSGD来近似Adam究竟有多科学呢？

我们知道，Adam优化器的更新量分母会带有一个$\epsilon$，初衷是预防除零错误，所以其值通常很接近于零，以至于我们做理论分析的时候通常选择忽略掉它。然而，当前LLM的训练尤其是低精度训练，我们往往会选择偏大的$\epsilon$，这导致在训练的中、后期$\epsilon$往往已经超过梯度平方大小，所以$\epsilon$的存在事实上已经不可忽略。

因此，这篇文章我们试图探索$\epsilon$如何影响Adam的学习率与Batch Size的Scaling Law，为相关问题提供一个参考的计算方案。

## SoftSign #

由于是接着上一篇文章介绍，所以就不再重复相关背景了。为了探究$\epsilon$的作用，我们从SignSGD换到SoftSignSGD，即$\newcommand{sign}{\mathop{\text{sign}}}\tilde{\boldsymbol{\varphi}}_B = \sign(\tilde{\boldsymbol{g}}_B)$变成$\tilde{\boldsymbol{\varphi}}_B = \text{softsign}(\tilde{\boldsymbol{g}}_B)$，其中  
\begin{equation}\sign(x)=\frac{x}{\sqrt{x^2}}\quad\to\quad\text{softsign}(x, \epsilon)=\frac{x}{\sqrt{x^2+\epsilon^2}}\end{equation}  
这个形式无疑更贴近更贴近Adam。但在此之前，我们需要确认$\epsilon$是否真的不可忽略，才能确定是否有进一步研究的价值。

在Keras的Adam实现中，$\epsilon$的默认值是$10^{-7}$，在Torch中则是$10^{-8}$，这时候梯度绝对值小于$\epsilon$几率还不算大；但在LLM中，$\epsilon$的普遍取值是$10^{-5}$（比如[LLAMA2](https://papers.cool/arxiv/2307.09288)），当训练进入“正轨”后，梯度绝对值小于$\epsilon$的分量将会很普遍了，所以$\epsilon$的影响是显著的。

这个跟LLM的参数量也有一定关系。一个能稳定训练的模型，不管参数量多大，它的梯度 _模长大小_ 大致都在同一数量级，这是反向传播的稳定性决定的（参考[《训练1000层的Transformer究竟有什么困难？》](/archives/8978)）。因此，参数量越大的模型，平均下来每个参数的梯度绝度值就相对变小了，从而$\epsilon$的作用就更突出了。

值得指出的是，$\epsilon$的引入实际上提供了Adam与SGD之间的一个插值，这是因为当$x\neq 0$时  
\begin{equation}\lim_{\epsilon\to \infty}\epsilon\,\text{softsign}(x, \epsilon)=\lim_{\epsilon\to \infty}\frac{x \epsilon}{\sqrt{x^2+\epsilon^2}} = x\end{equation}  
所以，$\epsilon$越大，Adam表现越接近SGD。

## S型近似 #

确认了引入$\epsilon$必要性后，我们着手开始分析。在分析过程中，我们将会反复遇到S型函数，所以还有一个准备工作是探究S型函数的简单近似。

S型函数相比大家已经见怪不怪，上一节引入的$\text{softsign}$函数本身就是之一，上一篇文章分析过程中的$\text{erf}$函数也是一例，此外还有$\tanh$、$\text{sigmoid}$等。接下来我们处理的是满足如下特性的S型函数$S(x)$：

> 1、全局光滑且单调递增；
> 
> 2、奇函数，值域是$[-1,1]$；
> 
> 3、在原点处斜率为$k > 0$。

对于这类函数，我们考虑两种近似。第一种近似跟$\text{softsign}$类似：  
\begin{equation}S(x)\approx \frac{x}{\sqrt{x^2 + 1/k^2}}\end{equation}  
它大概是保留$S(x)$如上3点性质的最简单函数了；第二种近似是基于$\text{clip}$函数：  
\begin{equation}S(x)\approx \text{clip}(kx, -1, 1) \triangleq \left\\{\begin{aligned}&1, &kx\geq 1 \\\  
&kx, &-1 < kx < 1\\\  
&-1, &kx \leq -1\end{aligned}\right.\end{equation}  
这本质上是一个分段线性函数，放弃了全局光滑的性质，但分段线性会使得积分算起来更容易，我们很快就会看到这一点。

[![Erf函数与它的两种近似](/usr/uploads/2024/11/103719578.png)](/usr/uploads/2024/11/103719578.png "点击查看原图")

Erf函数与它的两种近似

## 均值估计 #

事不宜迟，沿着上一篇文章的方法，出发点还是  
\begin{equation}\mathbb{E}[\mathcal{L}(\boldsymbol{w} - \eta\tilde{\boldsymbol{\varphi}}_B)] \approx \mathcal{L}(\boldsymbol{w}) - \eta\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g} + \frac{1}{2}\eta^2 \text{Tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})\end{equation}  
我们需要做的事情就是估计$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$和$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$。

这一节我们算的是$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]$，为此我们需要用$\text{clip}$函数去近似$\text{softsign}$函数：  
\begin{equation}\text{softsign}(x, \epsilon)\approx \text{clip}(x/\epsilon, -1, 1) = \left\\{\begin{aligned}&1, & x/\epsilon \geq 1 \\\  
& x / \epsilon, & -1 < x/\epsilon < 1 \\\  
&-1, & x/\epsilon \leq -1 \\\  
\end{aligned}\right.\end{equation}  
然后我们有  
\begin{equation}\begin{aligned}  
\mathbb{E}[\tilde{\varphi}_B] =&\, \mathbb{E}[\text{softsign}(g + \sigma z/\sqrt{B}, \epsilon)] \approx \mathbb{E}[\text{clip}(g/\epsilon + \sigma z/\epsilon\sqrt{B},-1, 1)] \\\\[5pt]  
=&\,\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \text{clip}(g/\epsilon + \sigma z/\epsilon\sqrt{B},-1, 1) e^{-z^2/2}dz \\\\[5pt]  
=&\,\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{-(g+\epsilon)\sqrt{B}/\sigma} (-1)\times e^{-z^2/2}dz + \frac{1}{\sqrt{2\pi}}\int_{-(g-\epsilon)\sqrt{B}/\sigma}^{\infty} 1\times e^{-z^2/2}dz \\\\[5pt]  
&\,\qquad\qquad + \frac{1}{\sqrt{2\pi}}\int_{-(g+\epsilon)\sqrt{B}/\sigma}^{-(g-\epsilon)\sqrt{B}/\sigma} (g/\epsilon + \sigma z/\epsilon\sqrt{B})\times e^{-z^2/2}dz  
\end{aligned}\end{equation}  
积分形式很复杂，但用Mathematica算并不难，结果可以用$\text{erf}$函数表达出来：  
\begin{equation}\frac{1}{2}\left[\text{erf}\left(\frac{a+b}{\sqrt{2}}\right)+\text{erf}\left(\frac{a-b}{\sqrt{2}}\right)\right]+\frac{a}{2b}\left[\text{erf}\left(\frac{a+b}{\sqrt{2}}\right)-\text{erf}\left(\frac{a-b}{\sqrt{2}}\right)\right]+\frac{e^{-(a+b)^2/2} - e^{-(a-b)^2/2}}{b\sqrt{2\pi}}\end{equation}  
其中$a = g\sqrt{B}/\sigma, b=\epsilon \sqrt{B}/\sigma$。这个函数看起来比较复杂，但它刚好是$a$的S型函数，值域为$(-1,1)$且在$a=0$处的斜率是$\text{erf}(b/\sqrt{2})/b$，所以利用第一种近似形式  
\begin{equation}\mathbb{E}[\tilde{\varphi}_B]\approx\frac{a}{\sqrt{a^2 + b^2 / \text{erf}(b/\sqrt{2})^2}}\approx \frac{a}{\sqrt{a^2 + b^2 + \pi / 2}}=\frac{g/\sigma}{\sqrt{(g^2+\epsilon^2)/\sigma^2 + \pi / 2B}}\label{eq:E-u-approx}\end{equation}  
第二个约等号是利用近似$\text{erf}(x)\approx x / \sqrt{x^2 + \pi / 4}$来处理分母中的$\text{erf}(b/\sqrt{2})$。可以说相当幸运，最终的形式并没有太复杂。接着我们有  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]_i \approx \frac{g_i/\sigma_i}{\sqrt{(g_i^2+\epsilon^2)/\sigma_i^2 + \pi / 2B}} = \frac{\text{softsign}(g_i, \epsilon)}{\sqrt{1 + \pi \sigma_i^2 /(g_i^2+\epsilon^2)/2B}}\approx \frac{\text{softsign}(g_i, \epsilon)}{\sqrt{1 + \pi \kappa^2/2B}} = \nu_i \beta\end{equation}  
跟上一篇文章一样，最后一个约等号使用了平均场近似，$\kappa^2$是全体$\sigma_i^2 /(g_i^2+\epsilon^2)$的某种平均，而$\nu_i = \text{softsign}(g_i, \epsilon)$以及$\beta = (1 + \pi\kappa^2 / 2B)^{-1/2}$。

## 方差估计 #

均值也就是一阶矩解决了，现在轮到二阶矩了  
\begin{equation}\begin{aligned}  
\mathbb{E}[\tilde{\varphi}_B^2] =&\, \mathbb{E}[\text{softsign}(g + \sigma z/\sqrt{B}, \epsilon)^2] \approx \mathbb{E}[\text{clip}(g/\epsilon + \sigma z/\epsilon\sqrt{B},-1, 1)^2] \\\\[5pt]  
=&\,\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} \text{clip}(g/\epsilon + \sigma z/\epsilon\sqrt{B},-1, 1)^2 e^{-z^2/2}dz \\\\[5pt]  
=&\,\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{-(g+\epsilon)\sqrt{B}/\sigma} (-1)^2\times e^{-z^2/2}dz + \frac{1}{\sqrt{2\pi}}\int_{-(g-\epsilon)\sqrt{B}/\sigma}^{\infty} 1^2\times e^{-z^2/2}dz \\\\[5pt]  
&\,\qquad\qquad + \frac{1}{\sqrt{2\pi}}\int_{-(g+\epsilon)\sqrt{B}/\sigma}^{-(g-\epsilon)\sqrt{B}/\sigma} (g/\epsilon + \sigma z/\epsilon\sqrt{B})^2\times e^{-z^2/2}dz  
\end{aligned}\end{equation}  
结果同样可以用$\text{erf}$函数表示，但更加冗长，这里就不写出来了，还是那句话，对Mathematica来说这都不是事。视为$a$的函数时，可以发现结果是一条倒钟形的曲线，关于$y$轴对称，上界是$1$，最小值是则在$(0,1)$内。参考$\mathbb{E}[\tilde{\varphi}_B]$的近似式$\eqref{eq:E-u-approx}$，我们选择如下近似  
\begin{equation}\mathbb{E}[\tilde{\varphi}_B^2] \approx 1 - \frac{b^2}{a^2 + b^2 + \pi / 2} = 1 - \frac{\epsilon^2/(g^2+\epsilon^2)}{1 + \pi\sigma^2/(g^2+\epsilon^2) / 2B}\end{equation}  
有一说一，这个近似的精度并不高，主要是为了计算的方便，但它已经保留了倒钟形、$y$轴对称、上界为1、$b=0$时结果为1、$b\to\infty$结果则为0等关键特性。接下来继续应用平均场近似：  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]_{i,i} \approx 1 - \frac{\epsilon^2/(g_i^2+\epsilon^2)}{1 + \pi\sigma_i^2/(g_i^2+\epsilon^2) / 2B}\approx 1 - \frac{\epsilon^2/(g_i^2+\epsilon^2)}{1 + \pi\kappa^2 / 2B} = \nu_i^2 \beta^2 + (1 - \beta^2)\end{equation}  
所以$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]_{i,j}\approx \nu_i \nu_j \beta^2 + \delta_{i,j}(1-\beta^2)$。其中$\delta_{i,j}(1-\beta^2)$这一项就代表了$\tilde{\boldsymbol{\varphi}}$的协方差矩阵$(1-\beta^2)\boldsymbol{I}$，它是一个对角阵，这是可以预料的，因为我们的假设之一是$\tilde{\boldsymbol{\varphi}}$各分量之间的独立性，所以协方差矩阵必然是对角阵。

## 结果初探 #

由此我们得到  
\begin{equation}\eta^* \approx \frac{\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g}}{\text{Tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})} \approx \frac{\beta\sum_i \nu_i g_i}{\beta^2\sum_{i,j} \nu_i \nu_j H_{i,j} + (1-\beta^2)\sum_i H_{i,i} }\end{equation}  
注意，除了$\beta$外，剩余的其他符号都不依赖于$B$，所以上式已经给出$\eta^*$与$B$的依赖关系。注意为了保证极小值的存在性，我们都会假设$\boldsymbol{H}$矩阵的正定性，而在此假设之下必然有$\sum_{i,j} \nu_i \nu_j H_{i,j} > 0$和$\sum_i H_{i,i} > 0$。

上一篇文章我们说Adam最重要的特性是可能会出现“Surge现象”，即$\eta^*$关于$B$不再是全局的单调递增函数。接下来我们将会证明，$\epsilon > 0$的引入会降低Surge现象出现的几率，并且$\epsilon \to \infty$时完全消失。这个证明并不难，很明显Surge现象出现的必要条件是  
\begin{equation}\sum_{i,j} \nu_i \nu_j H_{i,j} - \sum_i H_{i,i} > 0\end{equation}  
若否，整个$\eta^*$关于$\beta$便是单调递增的，而$\beta$关于$B$是单调递增的，所以整个$\eta^*$关于$B$单调递增，不存在Surge现象。别忘了$\nu_i = \text{softsign}(g_i, \epsilon)$是关于$\epsilon$的单调递减函数，所以当$\epsilon$增大时$\sum_{i,j} \nu_i \nu_j H_{i,j}$会更小，从而上述不等式成立的可能性更低，并且$\epsilon\to \infty$时$\nu_i$为零，上述不等式不可能再成立，因此Surge现象消失。

进一步，我们可以证明$\epsilon\to\infty$时，结果跟SGD的一致，这只需要留意到  
\begin{equation}\frac{\eta^*}{\epsilon} \approx \frac{\beta\sum_i (\epsilon \nu_i) g_i}{\beta^2\sum_{i,j} (\epsilon \nu_i)(\epsilon \nu_j) H_{i,j} + \epsilon^2(1-\beta^2)\sum_i H_{i,i} }\end{equation}  
我们有极限  
\begin{equation}\lim_{\epsilon\to\infty} \beta = 1,\quad\lim_{\epsilon\to\infty} \epsilon \nu_i = g_i, \quad \lim_{\epsilon\to\infty} \epsilon^2(1-\beta^2) = \pi \sigma^2 / 2B\end{equation}  
这里$\sigma^2$是全体$\sigma_i^2$的某种平均。于是我们得到当$\epsilon$足够大时有近似  
\begin{equation}\frac{\eta^*}{\epsilon} \approx \frac{\sum_i g_i^2}{\sum_{i,j} g_i g_j H_{i,j} + \left(\pi \sigma^2\sum_i H_{i,i}\right)/2B }\end{equation}  
右端就是假设梯度协方差矩阵为$(\pi\sigma^2/2B)\boldsymbol{I}$时的SGD结果。

## 文章小结 #

本文延续了上一篇文章的方法，尝试分析了Adam的$\epsilon$对学习率与Batch Size之间的Scaling Law的影响，结果是一个介乎SGD与SignSGD之间的形式，当$\epsilon$越大，结果越接近SGD，“Surge现象”出现的概率就越低。总的来说，计算结果没有特别让人意外之处，但可以作为分析$\epsilon$作用的一个参考过程。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10563>_

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

苏剑林. (Nov. 18, 2024). 《Adam的epsilon如何影响学习率的Scaling Law？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10563>

@online{kexuefm-10563,  
title={Adam的epsilon如何影响学习率的Scaling Law？},  
author={苏剑林},  
year={2024},  
month={Nov},  
url={\url{https://spaces.ac.cn/archives/10563}},  
} 


---

## 公式推导与注释

本节提供极其详细的数学推导，深入探讨Adam优化器中epsilon参数的作用机制及其对学习率Scaling Law的影响。

### 1. Adam优化器的完整数学定义

**基本更新规则**

Adam优化器的完整更新公式可以写成：

$$
\begin{aligned}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^2 \\
\hat{\boldsymbol{m}}_t &= \frac{\boldsymbol{m}_t}{1-\beta_1^t} \\
\hat{\boldsymbol{v}}_t &= \frac{\boldsymbol{v}_t}{1-\beta_2^t} \\
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t - \eta\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}
\end{aligned}
$$

**注释1**：$\boldsymbol{m}_t$ 是梯度的一阶矩（动量）的指数移动平均。$\beta_1$ 通常取0.9，意味着历史梯度的影响以指数衰减的方式保留。

**注释2**：$\boldsymbol{v}_t$ 是梯度平方的二阶矩估计。$\beta_2$ 通常取0.999，相比一阶矩，二阶矩需要更长的历史窗口才能稳定。

**注释3**：偏差修正项 $1-\beta_1^t$ 和 $1-\beta_2^t$ 用于修正初始化为零向量导致的偏差。在训练初期（$t$ 较小时）这些修正非常重要。

**推导1：偏差修正的必要性**

假设初始化 $\boldsymbol{m}_0 = \boldsymbol{0}$，真实梯度为常数 $\boldsymbol{g}$，则：

$$
\begin{aligned}
\boldsymbol{m}_1 &= \beta_1 \cdot \boldsymbol{0} + (1-\beta_1)\boldsymbol{g} = (1-\beta_1)\boldsymbol{g} \\
\boldsymbol{m}_2 &= \beta_1(1-\beta_1)\boldsymbol{g} + (1-\beta_1)\boldsymbol{g} = (1-\beta_1)(1+\beta_1)\boldsymbol{g} \\
\boldsymbol{m}_t &= (1-\beta_1)\sum_{i=0}^{t-1}\beta_1^i \boldsymbol{g} = (1-\beta_1)\frac{1-\beta_1^t}{1-\beta_1}\boldsymbol{g} = (1-\beta_1^t)\boldsymbol{g}
\end{aligned}
$$

因此 $\mathbb{E}[\boldsymbol{m}_t] = (1-\beta_1^t)\boldsymbol{g}$ 而非 $\boldsymbol{g}$，需要除以 $1-\beta_1^t$ 来修正。

### 2. Epsilon参数的数值稳定性作用

**推导2：除零保护**

考虑分母 $\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon$，如果某个参数的梯度始终很小，可能导致 $\hat{v}_{t,i} \approx 0$。此时：

$$
\frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}}+\epsilon} \approx \frac{\hat{m}_{t,i}}{\epsilon}
$$

**注释4**：$\epsilon$ 为小正数（如 $10^{-7}$ 或 $10^{-8}$）防止除零错误，同时保证数值稳定性。

**推导3：有效学习率的尺度分析**

对于第 $i$ 个参数，实际更新步长为：

$$
\Delta w_{t,i} = -\eta\frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}}+\epsilon}
$$

定义有效学习率：

$$
\eta_{\text{eff},i} = \frac{\eta}{\sqrt{\hat{v}_{t,i}}+\epsilon}
$$

当梯度尺度较大时（$\hat{v}_{t,i} \gg \epsilon^2$）：

$$
\eta_{\text{eff},i} \approx \frac{\eta}{\sqrt{\hat{v}_{t,i}}} = \frac{\eta}{\text{RMS}(g_{t,i})}
$$

这是标准的自适应学习率。

当梯度尺度较小时（$\hat{v}_{t,i} \ll \epsilon^2$）：

$$
\eta_{\text{eff},i} \approx \frac{\eta}{\epsilon}
$$

此时有效学习率由 $\epsilon$ 主导，不再自适应。

### 3. SoftSign函数的详细分析

**定义与性质**

SoftSign函数定义为：

$$
\text{softsign}(x, \epsilon) = \frac{x}{\sqrt{x^2+\epsilon^2}}
$$

**推导4：与Sign函数的关系**

$$
\lim_{\epsilon\to 0}\text{softsign}(x, \epsilon) = \lim_{\epsilon\to 0}\frac{x}{\sqrt{x^2+\epsilon^2}} = \frac{x}{|x|} = \text{sign}(x)
$$

**推导5：导数计算**

$$
\begin{aligned}
\frac{\partial}{\partial x}\text{softsign}(x, \epsilon) &= \frac{\partial}{\partial x}\left[x(x^2+\epsilon^2)^{-1/2}\right] \\
&= (x^2+\epsilon^2)^{-1/2} + x \cdot \left(-\frac{1}{2}\right)(x^2+\epsilon^2)^{-3/2}\cdot 2x \\
&= \frac{1}{\sqrt{x^2+\epsilon^2}} - \frac{x^2}{(x^2+\epsilon^2)^{3/2}} \\
&= \frac{x^2+\epsilon^2-x^2}{(x^2+\epsilon^2)^{3/2}} \\
&= \frac{\epsilon^2}{(x^2+\epsilon^2)^{3/2}}
\end{aligned}
$$

**注释5**：导数在 $x=0$ 处达到最大值 $1/\epsilon$，且导数恒正，说明softsign是单调递增的光滑函数。

**推导6：渐近行为**

当 $|x| \to \infty$ 时：

$$
\text{softsign}(x, \epsilon) = \frac{x}{\sqrt{x^2+\epsilon^2}} = \frac{x}{|x|\sqrt{1+\epsilon^2/x^2}} \to \text{sign}(x)
$$

当 $x \to 0$ 时，使用泰勒展开：

$$
\text{softsign}(x, \epsilon) = \frac{x}{\epsilon\sqrt{1+x^2/\epsilon^2}} \approx \frac{x}{\epsilon}\left(1-\frac{x^2}{2\epsilon^2}\right) = \frac{x}{\epsilon} - \frac{x^3}{2\epsilon^3}
$$

### 4. Adam向SGD的插值性质

**推导7：大epsilon极限**

考虑更新方向：

$$
\boldsymbol{\varphi} = \frac{\boldsymbol{m}}{\sqrt{\boldsymbol{v}}+\epsilon}
$$

当 $\epsilon \to \infty$ 时，对第 $i$ 个分量：

$$
\varphi_i = \frac{m_i}{\sqrt{v_i}+\epsilon} = \frac{m_i}{\epsilon\sqrt{v_i/\epsilon^2+1}} \approx \frac{m_i}{\epsilon}\left(1-\frac{v_i}{2\epsilon^2}\right)
$$

因此整个更新量：

$$
\eta\boldsymbol{\varphi} = \eta\frac{\boldsymbol{m}}{\epsilon} - \eta\frac{\boldsymbol{m}\odot\boldsymbol{v}}{2\epsilon^3} \approx \frac{\eta}{\epsilon}\boldsymbol{m}
$$

定义 $\tilde{\eta} = \eta/\epsilon$，则：

$$
\lim_{\epsilon\to\infty}\eta\frac{\boldsymbol{m}}{\sqrt{\boldsymbol{v}}+\epsilon} = \tilde{\eta}\boldsymbol{m}
$$

**注释6**：这表明当 $\epsilon$ 足够大时，Adam的行为接近于动量SGD，只是学习率需要相应缩放。

### 5. S型函数的近似理论

**推导8：Clip函数近似softsign**

考虑softsign函数在原点处的斜率：

$$
\left.\frac{d}{dx}\text{softsign}(x,\epsilon)\right|_{x=0} = \frac{\epsilon^2}{\epsilon^3} = \frac{1}{\epsilon}
$$

因此线性区域的近似为：

$$
\text{softsign}(x,\epsilon) \approx \frac{x}{\epsilon}, \quad |x| \ll \epsilon
$$

当 $|x/\epsilon| \geq 1$ 时，$\text{softsign}(x,\epsilon) \approx \text{sign}(x)$。综合得到clip近似：

$$
\text{softsign}(x,\epsilon) \approx \text{clip}(x/\epsilon, -1, 1)
$$

**推导9：误差函数erf的近似**

误差函数定义为：

$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}dt
$$

在原点处的泰勒展开：

$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}}\left(x - \frac{x^3}{3} + \frac{x^5}{10} - \cdots\right)
$$

原点处的斜率为 $2/\sqrt{\pi}$。使用第一种近似形式，要求在 $x=0$ 处斜率匹配：

$$
\frac{d}{dx}\frac{x}{\sqrt{x^2+c}}\bigg|_{x=0} = \frac{1}{\sqrt{c}} = \frac{2}{\sqrt{\pi}}
$$

解得 $c = \pi/4$。

### 6. 梯度期望值的详细推导

**推导10：单个分量的期望计算**

假设梯度的单个分量服从 $\tilde{g}_B \sim \mathcal{N}(g, \sigma^2/B)$，我们要计算：

$$
\mathbb{E}[\text{softsign}(\tilde{g}_B, \epsilon)]
$$

使用clip近似：

$$
\mathbb{E}[\text{clip}(\tilde{g}_B/\epsilon, -1, 1)] = \int_{-\infty}^{\infty} \text{clip}(z/\epsilon, -1, 1) \cdot \frac{B}{2\pi\sigma^2}\exp\left(-\frac{B(z-g)^2}{2\sigma^2}\right)dz
$$

**推导11：分段积分**

将积分分为三个区域：

区域1：$z < -\epsilon$，此时clip值为-1
区域2：$-\epsilon \leq z \leq \epsilon$，此时clip值为 $z/\epsilon$
区域3：$z > \epsilon$，此时clip值为1

$$
\begin{aligned}
\mathbb{E}[\text{clip}] &= \int_{-\infty}^{-\epsilon}(-1)\cdot p(z)dz + \int_{-\epsilon}^{\epsilon}\frac{z}{\epsilon}p(z)dz + \int_{\epsilon}^{\infty}1\cdot p(z)dz \\
&= -P(Z<-\epsilon) + \frac{1}{\epsilon}\mathbb{E}[Z\mathbb{1}_{|Z|\leq\epsilon}] + P(Z>\epsilon)
\end{aligned}
$$

其中 $Z \sim \mathcal{N}(g, \sigma^2/B)$。

**推导12：标准化变换**

令 $U = \frac{\sqrt{B}(Z-g)}{\sigma} \sim \mathcal{N}(0,1)$，则 $Z = g + \frac{\sigma U}{\sqrt{B}}$：

$$
\begin{aligned}
P(Z < -\epsilon) &= P\left(U < -\frac{\sqrt{B}(g+\epsilon)}{\sigma}\right) = \Phi\left(-\frac{(g+\epsilon)\sqrt{B}}{\sigma}\right) \\
P(Z > \epsilon) &= P\left(U > \frac{\sqrt{B}(\epsilon-g)}{\sigma}\right) = 1-\Phi\left(\frac{(\epsilon-g)\sqrt{B}}{\sigma}\right)
\end{aligned}
$$

这里 $\Phi$ 是标准正态分布的累积分布函数。

**推导13：用erf表示**

利用 $\Phi(x) = \frac{1}{2}\left[1+\text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$：

$$
\begin{aligned}
-P(Z<-\epsilon) + P(Z>\epsilon) &= -\frac{1}{2}\left[1+\text{erf}\left(-\frac{(g+\epsilon)\sqrt{B}}{\sigma\sqrt{2}}\right)\right] + \frac{1}{2}\left[1-\text{erf}\left(\frac{(\epsilon-g)\sqrt{B}}{\sigma\sqrt{2}}\right)\right] \\
&= \frac{1}{2}\left[\text{erf}\left(\frac{(g+\epsilon)\sqrt{B}}{\sigma\sqrt{2}}\right) - \text{erf}\left(\frac{(\epsilon-g)\sqrt{B}}{\sigma\sqrt{2}}\right)\right]
\end{aligned}
$$

**推导14：中间项的计算**

对于中间项：

$$
\frac{1}{\epsilon}\mathbb{E}[Z\mathbb{1}_{|Z|\leq\epsilon}] = \frac{1}{\epsilon}\int_{-\epsilon}^{\epsilon}z\cdot\frac{1}{\sqrt{2\pi\sigma^2/B}}\exp\left(-\frac{B(z-g)^2}{2\sigma^2}\right)dz
$$

这个积分比较复杂，最终结果可以用高斯分布的性质表示为：

$$
\frac{1}{\epsilon}\mathbb{E}[Z\mathbb{1}_{|Z|\leq\epsilon}] = \frac{g}{2\epsilon}\left[\text{erf}\left(\frac{(g+\epsilon)\sqrt{B}}{\sigma\sqrt{2}}\right)-\text{erf}\left(\frac{(g-\epsilon)\sqrt{B}}{\sigma\sqrt{2}}\right)\right] + \frac{\sigma}{\epsilon\sqrt{2\pi B}}\left[e^{-\frac{B(g-\epsilon)^2}{2\sigma^2}} - e^{-\frac{B(g+\epsilon)^2}{2\sigma^2}}\right]
$$

### 7. 简化为softsign形式

**推导15：引入无量纲参数**

定义：
- $a = g\sqrt{B}/\sigma$：信号强度参数
- $b = \epsilon\sqrt{B}/\sigma$：噪声尺度参数

则前面的期望可以重写为关于 $a$ 和 $b$ 的函数。

**推导16：大B极限**

当 $B \to \infty$ 时，$a,b \to \infty$，此时：

$$
\mathbb{E}[\text{clip}(\tilde{g}_B/\epsilon, -1, 1)] \to \text{sign}(g)
$$

这是因为噪声 $\sigma/\sqrt{B} \to 0$，梯度估计变得准确。

**推导17：erf函数的softsign近似**

根据前面的分析，复杂的erf表达式可以近似为：

$$
\mathbb{E}[\tilde{\varphi}_B] \approx \frac{a}{\sqrt{a^2+b^2+\pi/2}} = \frac{g/\sigma}{\sqrt{(g^2+\epsilon^2)/\sigma^2 + \pi/2B}}
$$

进一步整理：

$$
\mathbb{E}[\tilde{\varphi}_B] = \frac{g}{\sqrt{g^2+\epsilon^2+\pi\sigma^2/2B}}
$$

### 8. 平均场近似

**推导18：引入方差比参数**

定义方差比：

$$
\kappa_i^2 = \frac{\sigma_i^2}{g_i^2+\epsilon^2}
$$

这个量衡量了噪声相对于信号的强度。

**推导19：分解更新方向**

可以将期望值重写为：

$$
\mathbb{E}[\tilde{\varphi}_B]_i = \frac{g_i}{\sqrt{g_i^2+\epsilon^2}}\cdot\frac{1}{\sqrt{1+\pi\kappa_i^2/2B}} = \nu_i \beta_i
$$

其中：
- $\nu_i = \text{softsign}(g_i, \epsilon)$ 是确定性部分
- $\beta_i = (1+\pi\kappa_i^2/2B)^{-1/2}$ 是随机性修正

**推导20：平均场假设**

假设存在平均的 $\kappa^2$，使得：

$$
\kappa^2 \approx \frac{1}{n}\sum_{i=1}^n \kappa_i^2 = \frac{1}{n}\sum_{i=1}^n \frac{\sigma_i^2}{g_i^2+\epsilon^2}
$$

则所有分量的修正因子近似相同：

$$
\beta_i \approx \beta = \frac{1}{\sqrt{1+\pi\kappa^2/2B}}
$$

**注释7**：这个近似假设所有参数的"信噪比"相近，在实际训练中后期通常成立。

### 9. 二阶矩的详细计算

**推导21：sign函数的平方**

对于 $\tilde{\varphi}_B = \text{sign}(\tilde{g}_B)$：

$$
\tilde{\varphi}_B^2 = \text{sign}(\tilde{g}_B)^2 = 1
$$

因此：

$$
\mathbb{E}[\tilde{\varphi}_B^2] = 1
$$

**推导22：softsign平方的期望**

对于softsign：

$$
\mathbb{E}[\text{softsign}(\tilde{g}_B, \epsilon)^2] = \mathbb{E}\left[\frac{\tilde{g}_B^2}{\tilde{g}_B^2+\epsilon^2}\right]
$$

这个期望的精确计算很复杂，我们使用如下近似。当 $|\tilde{g}_B| \gg \epsilon$ 时，softsign$^2 \approx 1$；当 $|\tilde{g}_B| \ll \epsilon$ 时，softsign$^2 \approx \tilde{g}_B^2/\epsilon^2$。

**推导23：倒钟形近似**

基于前面的分析，我们采用如下近似形式：

$$
\mathbb{E}[\text{softsign}(\tilde{g}_B, \epsilon)^2] \approx 1 - \frac{\epsilon^2/(g^2+\epsilon^2)}{1+\pi\sigma^2/(g^2+\epsilon^2)/2B}
$$

**注释8**：这个公式的直观解释是：当真实梯度 $g$ 很大时，期望接近1；当 $g$ 接近0时，期望接近 $1-1/(1+\text{const})$，小于1。

**推导24：协方差矩阵的结构**

对于二阶矩矩阵：

$$
\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^T]_{ij} = \begin{cases}
\mathbb{E}[\tilde{\varphi}_{B,i}^2] \approx 1-\frac{\epsilon^2/(g_i^2+\epsilon^2)}{1+\pi\kappa^2/2B}, & i=j \\
\mathbb{E}[\tilde{\varphi}_{B,i}]\mathbb{E}[\tilde{\varphi}_{B,j}] \approx \nu_i\nu_j\beta^2, & i\neq j
\end{cases}
$$

**推导25：对角项的进一步处理**

对角项可以分解为：

$$
\begin{aligned}
\mathbb{E}[\tilde{\varphi}_{B,i}^2] &\approx 1 - \frac{\epsilon^2/(g_i^2+\epsilon^2)}{1+\pi\kappa^2/2B} \\
&= \frac{1+\pi\kappa^2/2B - \epsilon^2/(g_i^2+\epsilon^2)}{1+\pi\kappa^2/2B} \\
&= \frac{(g_i^2+\epsilon^2+\pi\kappa^2\sigma_i^2/2B)-\epsilon^2}{(g_i^2+\epsilon^2)(1+\pi\kappa^2/2B)} \\
&\approx \nu_i^2\beta^2 + (1-\beta^2)
\end{aligned}
$$

其中最后一步使用了 $\nu_i^2 = g_i^2/(g_i^2+\epsilon^2)$。

### 10. 最优学习率的推导

**推导26：损失函数的二阶近似**

从基本的泰勒展开开始：

$$
\mathcal{L}(\boldsymbol{w}-\eta\tilde{\boldsymbol{\varphi}}_B) \approx \mathcal{L}(\boldsymbol{w}) - \eta\tilde{\boldsymbol{\varphi}}_B^T\boldsymbol{g} + \frac{\eta^2}{2}\tilde{\boldsymbol{\varphi}}_B^T\boldsymbol{H}\tilde{\boldsymbol{\varphi}}_B
$$

取期望：

$$
\mathbb{E}[\mathcal{L}(\boldsymbol{w}-\eta\tilde{\boldsymbol{\varphi}}_B)] \approx \mathcal{L}(\boldsymbol{w}) - \eta\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^T\boldsymbol{g} + \frac{\eta^2}{2}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^T\boldsymbol{H}\tilde{\boldsymbol{\varphi}}_B]
$$

**推导27：迹的技巧**

利用标量的迹等于自身：

$$
\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^T\boldsymbol{H}\tilde{\boldsymbol{\varphi}}_B] = \mathbb{E}[\text{Tr}(\tilde{\boldsymbol{\varphi}}_B^T\boldsymbol{H}\tilde{\boldsymbol{\varphi}}_B)] = \text{Tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^T]\boldsymbol{H})
$$

**推导28：展开迹**

$$
\begin{aligned}
\text{Tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^T]\boldsymbol{H}) &= \sum_i\mathbb{E}[\tilde{\varphi}_{B,i}^2]H_{ii} + \sum_{i\neq j}\mathbb{E}[\tilde{\varphi}_{B,i}\tilde{\varphi}_{B,j}]H_{ij} \\
&\approx \sum_i(\nu_i^2\beta^2+(1-\beta^2))H_{ii} + \sum_{i\neq j}\nu_i\nu_j\beta^2 H_{ij} \\
&= \beta^2\sum_{i,j}\nu_i\nu_j H_{ij} + (1-\beta^2)\sum_i H_{ii}
\end{aligned}
$$

**推导29：最优化问题**

要最小化期望损失，对 $\eta$ 求导并令其为零：

$$
\frac{d}{d\eta}\mathbb{E}[\mathcal{L}] = -\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^T\boldsymbol{g} + \eta\text{Tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^T]\boldsymbol{H}) = 0
$$

解得：

$$
\eta^* = \frac{\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^T\boldsymbol{g}}{\text{Tr}(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^T]\boldsymbol{H})}
$$

**推导30：代入前面的结果**

$$
\eta^* \approx \frac{\beta\sum_i\nu_i g_i}{\beta^2\sum_{i,j}\nu_i\nu_j H_{ij} + (1-\beta^2)\sum_i H_{ii}}
$$

### 11. Surge现象的理论分析

**推导31：单调性条件**

$\eta^*$ 关于 $\beta$ 的导数为：

$$
\frac{d\eta^*}{d\beta} = \frac{d}{d\beta}\left[\frac{\beta\sum_i\nu_i g_i}{\beta^2\sum_{i,j}\nu_i\nu_j H_{ij} + (1-\beta^2)\sum_i H_{ii}}\right]
$$

记 $A = \sum_i\nu_i g_i$，$B = \sum_{i,j}\nu_i\nu_j H_{ij}$，$C = \sum_i H_{ii}$，则：

$$
\eta^* = \frac{A\beta}{\beta^2 B + (1-\beta^2)C} = \frac{A\beta}{\beta^2(B-C) + C}
$$

**推导32：求导**

$$
\frac{d\eta^*}{d\beta} = \frac{A[\beta^2(B-C)+C] - A\beta\cdot 2\beta(B-C)}{[\beta^2(B-C)+C]^2} = \frac{A[C-\beta^2(B-C)]}{[\beta^2(B-C)+C]^2}
$$

**推导33：单调性判定**

当 $B > C$ 时，存在临界点 $\beta_c = \sqrt{C/(B-C)}$。当 $\beta < \beta_c$ 时，导数为正（递增）；当 $\beta > \beta_c$ 时，导数为负（递减）。

**注释9**：这就是Surge现象的数学根源——当Hessian矩阵的非对角元素足够大时（$B > C$），最优学习率先增后减。

### 12. Epsilon趋于无穷的极限行为

**推导34**：当 $\epsilon \to \infty$ 时，

$$
\nu_i = \frac{g_i}{\sqrt{g_i^2+\epsilon^2}} = \frac{g_i/\epsilon}{\sqrt{g_i^2/\epsilon^2+1}} \to 0
$$

但是：

$$
\epsilon\nu_i = \frac{g_i}{\sqrt{1+g_i^2/\epsilon^2}} \to g_i
$$

**推导35**：对于 $\beta$：

$$
\beta = \frac{1}{\sqrt{1+\pi\kappa^2/2B}} = \frac{1}{\sqrt{1+\frac{\pi\sigma^2}{2B(g_i^2+\epsilon^2)}}}
$$

当 $\epsilon \to \infty$ 时：

$$
1-\beta^2 = \frac{\pi\sigma^2/2B}{g_i^2+\epsilon^2} \approx \frac{\pi\sigma^2}{2B\epsilon^2}
$$

因此：

$$
\epsilon^2(1-\beta^2) \to \frac{\pi\sigma^2}{2B}
$$

**推导36**：代入 $\eta^*/\epsilon$：

$$
\frac{\eta^*}{\epsilon} = \frac{\beta\sum_i(\epsilon\nu_i)g_i}{\beta^2\sum_{i,j}(\epsilon\nu_i)(\epsilon\nu_j)H_{ij} + \epsilon^2(1-\beta^2)\sum_i H_{ii}}
$$

取极限：

$$
\lim_{\epsilon\to\infty}\frac{\eta^*}{\epsilon} = \frac{\sum_i g_i^2}{\sum_{i,j}g_i g_j H_{ij} + \frac{\pi\sigma^2}{2B}\sum_i H_{ii}}
$$

这正是SGD的形式！

### 13. 不同Batch Size下的行为

**推导37**：小Batch Size极限（$B \ll \pi\kappa^2/2$）

$$
\beta \approx \sqrt{\frac{2B}{\pi\kappa^2}}, \quad \beta^2 \approx \frac{2B}{\pi\kappa^2}
$$

因此：

$$
\eta^* \approx \frac{\sqrt{2B/\pi\kappa^2}\sum_i\nu_i g_i}{\frac{2B}{\pi\kappa^2}\sum_{i,j}\nu_i\nu_j H_{ij} + \sum_i H_{ii}} \approx \frac{\sqrt{2B/\pi\kappa^2}\sum_i\nu_i g_i}{\sum_i H_{ii}} \propto \sqrt{B}
$$

**注释10**：这证明了小Batch Size下的平方根缩放规律。

**推导38**：大Batch Size极限（$B \gg \pi\kappa^2/2$）

$$
\beta \approx 1 - \frac{\pi\kappa^2}{4B}, \quad 1-\beta^2 \approx \frac{\pi\kappa^2}{2B}
$$

此时：

$$
\eta^* \approx \frac{\sum_i\nu_i g_i}{\sum_{i,j}\nu_i\nu_j H_{ij} + \frac{\pi\kappa^2}{2B}\sum_i H_{ii}}
$$

当 $B$ 继续增大，$\eta^*$ 趋于饱和。

### 14. Epsilon对不同梯度尺度的影响

**推导39**：定义梯度尺度分类

对于不同的参数，根据 $|g_i|$ 与 $\epsilon$ 的大小关系分类：

- 大梯度参数：$|g_i| \gg \epsilon$，此时 $\nu_i \approx \text{sign}(g_i)$
- 小梯度参数：$|g_i| \ll \epsilon$，此时 $\nu_i \approx g_i/\epsilon$

**推导40**：混合行为

对于大梯度参数，Adam表现为符号更新（接近AdaSign）；对于小梯度参数，Adam表现为缩放的梯度更新（接近SGD）。整体行为是这两种模式的混合。

### 15. 实际影响的定量分析

**推导41**：有效学习率的分布

考虑参数的有效学习率分布：

$$
\eta_{\text{eff},i} = \eta\frac{\nu_i\beta}{v_i^{1/2}+\epsilon}
$$

其中 $v_i$ 是第 $i$ 个参数的二阶矩估计。

当 $\epsilon$ 增大时：
1. 分子中的 $\nu_i$ 减小
2. 分母中的 $\epsilon$ 增大

两者共同作用使得有效学习率减小。

**注释11**：这解释了为什么LLM训练中使用较大的 $\epsilon$（如 $10^{-5}$）需要相应调整基础学习率。

### 总结

本节详细推导了Adam优化器中epsilon参数的数学机制。关键发现包括：

1. **Epsilon的双重作用**：既提供数值稳定性，又控制Adam到SGD的插值程度
2. **平方根缩放规律**：在小Batch Size下，Adam遵循 $\eta \propto \sqrt{B}$ 的规律
3. **Surge现象的抑制**：增大epsilon可以降低Surge现象出现的概率
4. **自适应性的权衡**：Epsilon越大，自适应程度越弱，但训练稳定性可能提高

这些理论分析为实际训练中epsilon的选择提供了数学依据。

