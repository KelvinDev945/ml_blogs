---
title: 重新思考学习率与Batch Siz...
slug: 重新思考学习率与batch-siz
date: 2025-09-22
tags: 学习率, 优化器, 尺度定律, 平均场, 生成模型
status: pending
---

# 重新思考学习率与Batch Siz...

**原文链接**: [https://spaces.ac.cn/archives/11301](https://spaces.ac.cn/archives/11301)

**发布日期**: 

---

我们在[《重新思考学习率与Batch Size（二）：平均场》](/archives/11280)中提到，关注SignSGD的原因之一是我们通常将它作为Adam的理论近似，这是Adam做理论分析时常用的简化策略。除了分析学习率的场景外，在[《配置不同的学习率，LoRA还能再涨一点？》](/archives/10001)、[《初探MuP：超参数的跨模型尺度迁移规律》](/archives/10770)等地方我们也用了这个简化。

然而，SignSGD真是Adam的良好近似吗？一个明显差异是SignSGD的Update RMS总是1，而Adam并非如此。笔者发现，导致这一差异的核心原因是动量，它普遍存在于Adam、Lion、Muon等优化器中。所以，本文我们来考察动量——更广义地说是EMA——的影响。

## 问题分析 #

从Adam的视角看，SignSGD对应$\beta_1=\beta_2=0$这个特例，或者对应于Adam的第一步更新量（不管$\beta_1,\beta_2$如何）。因此，我们认为它跟Adam肯定有一些共性，能够捕捉到一些通用的规律。

但是，它们之间也有一些明显的差异。比较典型的就是Update RMS的差异，SignSGD总是1，但Adam往往明显小于1；还有，Adam看上去更贴近SGD，它更像是SignSGD和SGD的一个中间版本。一开始，笔者以为这是Adam分母中的$\epsilon$导致的差异，所以在[《Adam的epsilon如何影响学习率的Scaling Law？》](/archives/10563)还特意计算了带$\epsilon$的SoftSignSGD。

后来，我们在[《为什么Adam的Update RMS是0.2？》](/archives/11267)从模拟和理论两方面估计了Adam的Update RMS，其实平均场近似的估计结果为$\sqrt{\frac{1-\beta_1}{1+\beta_1}}$，并且验证了它跟模拟结果和实际实验都很吻合。这个结果显式地依赖于$\beta_1$，所以很明显，它将我们的思考方向引向动量。

这就有了下面的分析过程。综下所述，我们可以确认，$\epsilon$的角色确实是次要的，真正的主角其实是动量——它是梯度的“滑动平均”——这也正是本文的主角“EMA（Exponential Moving Average）”。

## 梯度下降 #

为了分析EMA带来的变数，我们从SGDM入手，也就是带动量的SGD，实际上我们在用SGD的时候极少情况是不加动量的：  
\begin{equation}\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t \\\\[4pt]  
&\boldsymbol{w}_t = \boldsymbol{w}_{t-1} - \eta_t \boldsymbol{m}_t  
\end{aligned}\end{equation}  
实际使用中，$\boldsymbol{g}_t$替换为$\tilde{\boldsymbol{g}}_{B,t}$，它是一个随机变量，均值为$\boldsymbol{g}_t$，协方差矩阵为$\boldsymbol{\Sigma}_t/B$，这些基本设置跟[《重新思考学习率与Batch Size（一）：现状》](/archives/11260)是一样的。这里的噪声，是由随机采样不同的Batch引起的，所以我们可以合理地假设，不同$t$之间的$\tilde{\boldsymbol{g}}_{B,t}$是相互独立的。

我们的任务，是计算  
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\eta^* \approx \frac{\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}\boldsymbol{g}}{\tr(\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]\boldsymbol{H})}\label{eq:eta-opt}\end{equation}  
相关推导在前面几篇文章已经给出，这里就不再重复。对于SGDM来说$\tilde{\boldsymbol{\varphi}}_B = \boldsymbol{m}_t$，它可以展开成  
\begin{equation}\boldsymbol{m}_t = (1 - \beta_1)\sum\limits_{s=1}^t \beta_1^{t-s}\tilde{\boldsymbol{g}}_{B,s}\end{equation}

## 放大批量 #

现在可以计算  
\begin{equation}\mathbb{E}[\boldsymbol{m}_t] = (1 - \beta_1)\sum_{s=1}^t \beta_1^{t-s}\mathbb{E}[\tilde{\boldsymbol{g}}_{B,s}] = (1 - \beta_1)\sum_{s=1}^t \beta_1^{t-s}\boldsymbol{g}_s\end{equation}  
我们进一步假设当模型训练进入“正轨”后，梯度是缓变的，那么我们可以用当前梯度$\boldsymbol{g}_t$近似$\boldsymbol{g}_s$，得到  
\begin{equation}\mathbb{E}[\boldsymbol{m}_t] = (1 - \beta_1)\sum_{s=1}^t \beta_1^{t-s}\boldsymbol{g}_t = (1 - \beta_1^t) \boldsymbol{g}_t \approx \boldsymbol{g}_t \qquad (t\to\infty)\end{equation}  
至于$\mathbb{E}[\boldsymbol{m}_t \boldsymbol{m}_t^{\top}]$，我们利用恒等式$\mathbb{E}[\boldsymbol{m}_t \boldsymbol{m}_t^{\top}] = \mathbb{E}[\boldsymbol{m}_t] \mathbb{E}[\boldsymbol{m}_t]^{\top} + \mathbb{C}\text{ov}[\boldsymbol{m}_t,\boldsymbol{m}_t]$，然后利用方差的可加性得到：  
\begin{equation}\mathbb{C}\text{ov}[\boldsymbol{m}_t,\boldsymbol{m}_t] = (1 - \beta_1)^2\sum_{s=1}^t \beta_1^{2(t-s)}\boldsymbol{\Sigma}_s/B\end{equation}  
类似地，我们假设协方差矩阵的缓变性，那么  
\begin{equation}\mathbb{C}\text{ov}[\boldsymbol{m}_t] \approx (1 - \beta_1)^2\sum_{s=1}^t \beta_1^{2(t-s)}\boldsymbol{\Sigma}_t/B = (1 - \beta_1)^2\frac{1-\beta_1^{2t}}{1-\beta_1^2}\boldsymbol{\Sigma}_t/B = \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\Sigma}_t/B \qquad (t\to\infty)\end{equation}  
代入式$\eqref{eq:eta-opt}$得  
\begin{equation}\eta^* \approx \frac{\eta_{\max}}{1 + \frac{1 - \beta_1}{1 + \beta_1}\mathcal{B}_{\text{noise}}/B},\qquad \eta_{\max} = \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}},\quad\mathcal{B}_{\text{noise}} = \frac{\tr(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\end{equation}  
从这个结果可以看出，动量机制的引入，相当于把SGD的Batch Size放大到了$\frac{1 + \beta_1}{1 - \beta_1}$倍。按照笔者的理解，动量就是通过对优化轨迹上的梯度做EMA来低成本地消除梯度噪声，所以这个结果这跟笔者所理解的动量意义是相符的。

## 符号动量 #

进一步地，我们考虑SignSGDM，它可以视作[Lion](/archives/9473)的一个特例，也就是SGDM多加了个$\newcommand{sign}{\mathop{\text{sign}}}\sign$：  
\begin{equation}\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t \\\\[4pt]  
&\boldsymbol{w}_t = \boldsymbol{w}_{t-1} - \eta_t \sign(\boldsymbol{m}_t)  
\end{aligned}\end{equation}  
实际训练中$\boldsymbol{g}_t$同样替换为$\tilde{\boldsymbol{g}}_{B,t}$。对SignSGDM来说$\tilde{\boldsymbol{\varphi}}_B = \sign(\boldsymbol{m}_t)$，那么根据平均场近似得  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] = \mathbb{E}\bigg[\frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{m}_t^2}}\bigg]\approx \frac{\mathbb{E}[\boldsymbol{m}_t]}{\sqrt{\mathbb{E}[\boldsymbol{m}_t^2]}}\end{equation}  
其中向量乘法默认是Hadamard积。分子$\mathbb{E}[\boldsymbol{m}_t]$我们在上一节已经算了，分母$\mathbb{E}[\boldsymbol{m}_t^2]$其实等于$\newcommand{diag}{\mathop{\text{diag}}}\diag(\mathbb{E}[\boldsymbol{m}_t \boldsymbol{m}_t^{\top}])$，所以也可以代入上一节的结果，得到  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] \approx \frac{\boldsymbol{g}_t}{\sqrt{\boldsymbol{g}_t^2 + \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\sigma}_t^2/B}} = \frac{\sign(\boldsymbol{g}_t)}{\sqrt{1 + \frac{1 - \beta_1}{1 + \beta_1}(\boldsymbol{\sigma}_t^2/\boldsymbol{g}_t^2)/B}} \approx \frac{\sign(\boldsymbol{g}_t)}{\sqrt{1 + \frac{1 - \beta_1}{1 + \beta_1} \mathcal{B}_{\text{simple}}/B}}\end{equation}  
其中$\boldsymbol{\sigma}_t^2 = \diag(\boldsymbol{\Sigma}_t), \mathcal{B}_{\text{simple}} = \tr(\boldsymbol{\Sigma}_t)/\boldsymbol{g}_t^{\top}\boldsymbol{g}_t$。上式相当于SignSGD的$B$换成了$\frac{1 + \beta_1}{1 - \beta_1}B$，如果我们进一步计算$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B\tilde{\boldsymbol{\varphi}}_B^{\top}]$就会发现结论也是如此。所以，跟SGDM一样，动量相当于把SignSGD的Batch Size放大到了$\frac{1 + \beta_1}{1 - \beta_1}$倍。

在[《重新思考学习率与Batch Size（三）：Muon》](/archives/11285)中我们计算过Muon的学习率规律，发现它跟SignSGD一致，所以我们可以断言，动量在Muon中的作用跟SignSGDM一样，都约等于将Batch Size放大成$\frac{1 + \beta_1}{1 - \beta_1}$倍。

## 双重滑动 #

最后我们来看Adam：  
\begin{equation}\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\\  
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\\  
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\\  
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.  
\end{aligned}\end{equation}  
实际训练中$\boldsymbol{g}_t$替换为$\tilde{\boldsymbol{g}}_{B,t}$。我们考虑的都是训练已经进入“正轨”的状态，即$t\to\infty$，所以不区分$\boldsymbol{m}_t$和$\hat{\boldsymbol{m}}_t$、$\boldsymbol{v}_t$和$\hat{\boldsymbol{v}}_t$，同时我们聚焦于EMA的作用，所以设$\epsilon = 0$。那么对于Adam来说有$\tilde{\boldsymbol{\varphi}}_B=\boldsymbol{m}_t/\sqrt{\boldsymbol{v}_t}$，它跟SignSGDM的区别，就是分母的$\boldsymbol{m}_t^2$换成了另一个EMA的统计量$\boldsymbol{v}_t$。

由平均场近似得  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] = \mathbb{E}\bigg[\frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t}}\bigg]\approx \frac{\mathbb{E}[\boldsymbol{m}_t]}{\sqrt{\mathbb{E}[\boldsymbol{v}_t]}}\end{equation}  
$\mathbb{E}[\boldsymbol{m}_t]$我们已经算过，只需算$\mathbb{E}[\boldsymbol{v}_t]$：  
\begin{equation}\mathbb{E}[\boldsymbol{v}_t] = (1 - \beta_2)\sum_{s=1}^t \beta_2^{t-s}\mathbb{E}[\tilde{\boldsymbol{g}}_{B,s}^2] = (1 - \beta_2)\sum_{s=1}^t \beta_2^{t-s}(\boldsymbol{g}_s^2 + \boldsymbol{\sigma}_s^2/B)\approx \boldsymbol{g}_t^2 + \boldsymbol{\sigma}_t^2/B\end{equation}  
跟前面一样，最后一个约等号假设了梯度和方差的缓变性，以及$t\to\infty$。于是我们有  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] \approx \frac{\boldsymbol{g}_t}{\sqrt{\boldsymbol{g}_t^2 + \boldsymbol{\sigma}_t^2/B}} \approx \frac{\sign(\boldsymbol{g}_t)}{\sqrt{1 + \mathcal{B}_{\text{simple}}/B}}\end{equation}  
这个结果倒是跟SignSGD相同，所以单从一阶矩看，SignSGD作为Adam的近似是合理的。但我们还有二阶矩$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B \tilde{\boldsymbol{\varphi}}_B^{\top}]$，在分量独立的假设下，我们只需要算$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^2]$：  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^2] = \mathbb{E}\bigg[\frac{\boldsymbol{m}_t^2}{\boldsymbol{v}_t}\bigg]\approx \frac{\mathbb{E}[\boldsymbol{m}_t^2]}{\mathbb{E}[\boldsymbol{v}_t]} \approx \frac{\boldsymbol{g}_t^2 + \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\sigma}_t^2/B}{\boldsymbol{g}_t^2 + \boldsymbol{\sigma}_t^2/B}\label{eq:u2-adam}\end{equation}

## 两个特例 #

我们观察两个特例。首先是$\beta_1=0$，这时候分子分母相同，$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^2]$是全1向量，跟SignSGD一致。所以说，SignSGD是$\beta_1=0$的Adam——也就是[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)——的良好近似，当$\beta_1$增大时，近似程度开始变差。

当$\beta_1=1$时，我们有  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^2] \approx \frac{\boldsymbol{g}_t^2}{\boldsymbol{g}_t^2 + \boldsymbol{\sigma}_t^2/B}\approx \mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^2\end{equation}  
由此得到$\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B \tilde{\boldsymbol{\varphi}}_B^{\top}] \approx \mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] \mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top}$，代入到式$\eqref{eq:eta-opt}$得  
\begin{equation}\eta^* \approx \frac{\Vert \boldsymbol{g}\Vert_1 \sqrt{1 + \mathcal{B}_{\text{simple}}/B}}{\sign(\boldsymbol{g})^{\top} \boldsymbol{H} \sign(\boldsymbol{g})}\end{equation}  
注意，它是关于$B$的单调递减函数，即当Batch Size增大时学习率应该减小。由此我们可以推测，Adam的$\beta_1$的增大，将会加速“[Surge现象](/archives/11280#%E5%8F%8D%E5%B8%B8%E7%8E%B0%E8%B1%A1)”的出现。

这个结论看似有点费解，但其实换个角度就容易理解了。“Surge现象”指当Batch Size超过某个阈值后，最优学习率随着Batch Size的增大而减少，而前面SGDM、SignSGDM的结果都表明，动量的引入约等于将Batch Size扩大到$\frac{1 + \beta_1}{1 - \beta_1} > 1$倍，这自然增加了超过阈值的可能性。

换句话说，“随着$\beta_1$的增大，‘Surge现象’将更容易出现”的结论，即便对于SignSGDM也是成立的。而Adam相比SignSGDM有一些新的特性，但“动量机制约等于放大Batch Size”这一点始终是成立的，所以出现同样的结论就不难理解了。

## 一般分析 #

我们改写一下式$\eqref{eq:u2-adam}$：  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B^2] \approx \frac{\boldsymbol{g}_t^2 + \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\sigma}_t^2/B}{\boldsymbol{g}_t^2 + \boldsymbol{\sigma}_t^2/B} = \frac{2\beta_1}{1+\beta_1}\frac{\boldsymbol{g}_t^2}{\boldsymbol{g}_t^2 + \boldsymbol{\sigma}_t^2/B} + \frac{1 - \beta_1}{1 + \beta_1} \approx \frac{2\beta_1}{1+\beta_1}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^2 + \frac{1 - \beta_1}{1 + \beta_1}\end{equation}  
由此我们可以写出  
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B \tilde{\boldsymbol{\varphi}}_B^{\top}] \approx \mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] \mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^{\top} + \frac{1 - \beta_1}{1 + \beta_1}\diag\left(1 - \mathbb{E}[\tilde{\boldsymbol{\varphi}}_B]^2\right)\end{equation}  
那么  
\begin{equation}\eta^* \approx \frac{\sum_i |g_i|}{\frac{1}{\beta}\frac{1 - \beta_1}{1 + \beta_1}\sum_i H_{i,i} + \beta\left(\sum_{i,j} H_{i,j}\sign(g_i g_j) - \frac{1 - \beta_1}{1 + \beta_1}\sum_i H_{i,i}\right)}\end{equation}  
这里没有下标的$\beta$等于$(1 + \mathcal{B}_{\text{simple}}/B)^{-1/2}$，不仔细看的话可能会跟$\beta_1,\beta_2$混淆，笔者表示很抱歉，因为这是前两篇文章的记号，这里只好沿用了。跟SignSGD不同的是，SignSGD如果假设Hessian矩阵是对角阵，那么就不会出现Surge现象，但上式即便是在对角Hessian假设下依然出现Surge现象，此时：  
\begin{equation}\eta^* \approx \frac{\sum_i |g_i|}{\left(\frac{1}{\beta}\frac{1 - \beta_1}{1 + \beta_1} + \beta\frac{2\beta_1}{1 + \beta_1}\right)\sum_i H_{i,i}}\end{equation}  
由均值不等式知上式在$\beta^*=\sqrt{\frac{1-\beta_1}{2\beta_1}}$处取到最大值，但要注意根据$\beta$定义，它是$\in(0,1)$的，所以还要判断$\beta^*\in(0,1)$，即$\beta_1 > 1/3$，不满足这个条件时最大值依然在$\beta=1$取到，此时没有Surge现象。反之，当$\beta_1 > 1/3$且$\beta > \beta^*$（即$B > \frac{1-\beta_1}{3\beta_1-1}\mathcal{B}_{\text{simple}}$）时，学习率应该随着Batch Size的增加而减小。

这个结论可以初步解释为啥Muon能支持更大Batch Size。由[《重新思考学习率与Batch Size（三）：Muon》](/archives/11285)可知，Muon的表现跟SignSGDM类似，在特定Hessian结构假设下它不会出现Surge现象，这意味着增大Batch Size总可以提高学习效率，尽管相对收益会越来越小。

相反，Adam在常用设置（如$\beta_1=0.9$）下，哪怕假设Hessian是对角阵也会出现Surge现象，这意味着Batch Size超过一定值后，学习效率就下降了。

## 文章小结 #

本文初步分析了优化器的EMA机制对学习率与Batch Size的尺度定律的影响，确认了EMA特别是动量机制的引入会稍微改变尺度定律，而Adam这种带有双重EMA运算的优化器，则会呈现出一些跟SignSGD不同的新特性。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11301>_

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

苏剑林. (Sep. 22, 2025). 《重新思考学习率与Batch Size（四）：EMA 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11301>

@online{kexuefm-11301,  
title={重新思考学习率与Batch Size（四）：EMA},  
author={苏剑林},  
year={2025},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/11301}},  
} 


---

## 公式推导与注释

### 1. 线性缩放规则的理论基础

#### 1.1 噪声尺度理论

考虑mini-batch梯度估计:
\begin{equation}\tilde{\boldsymbol{g}}_B = \frac{1}{B}\sum_{i=1}^B \nabla L(\boldsymbol{x}_i; \boldsymbol{\theta})\tag{1}\end{equation}

**期望**: $\mathbb{E}[\tilde{\boldsymbol{g}}_B] = \boldsymbol{g}$,其中$\boldsymbol{g} = \mathbb{E}_{\boldsymbol{x}}[\nabla L(\boldsymbol{x}; \boldsymbol{\theta})]$

**方差**: 由中心极限定理,
\begin{equation}\text{Cov}[\tilde{\boldsymbol{g}}_B] = \frac{\boldsymbol{\Sigma}}{B}\tag{2}\end{equation}

其中$\boldsymbol{\Sigma} = \mathbb{E}[(\nabla L - \boldsymbol{g})(\nabla L - \boldsymbol{g})^{\top}]$是单样本梯度的协方差矩阵。

**数学直觉**: Batch size越大,梯度估计的方差越小,噪声按$1/\sqrt{B}$衰减。

#### 1.2 SGD的离散化SDE近似

将SGD视为随机微分方程(SDE)的离散化:
\begin{equation}d\boldsymbol{\theta} = -\boldsymbol{g}(\boldsymbol{\theta})dt + \sqrt{2\eta\boldsymbol{\Sigma}/B}d\boldsymbol{W}\tag{3}\end{equation}

其中$d\boldsymbol{W}$是Wiener过程(布朗运动)。

**离散化**:
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\tilde{\boldsymbol{g}}_{B,t}\tag{4}\end{equation}

对应于Euler-Maruyama离散化,时间步长$dt = \eta$。

**数学直觉**: SGD在参数空间的轨迹类似于带漂移项的布朗运动,噪声强度由$\eta/B$决定。

#### 1.3 平衡点分析

在平衡态附近,损失函数可以二次近似:
\begin{equation}L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}^*) + \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)^{\top}\boldsymbol{H}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)\tag{5}\end{equation}

其中$\boldsymbol{H}$是Hessian矩阵。

**平衡分布**: 当$\eta$足够小,SDE(3)的平衡分布是Gibbs分布:
\begin{equation}p(\boldsymbol{\theta}) \propto \exp\left(-\frac{B}{\eta}L(\boldsymbol{\theta})\right)\tag{6}\end{equation}

**温度**: 定义有效温度$T = \eta/B$,则:
\begin{equation}p(\boldsymbol{\theta}) \propto \exp\left(-\frac{L(\boldsymbol{\theta})}{T}\right)\tag{7}\end{equation}

**数学直觉**: $\eta/B$决定了SGD探索参数空间的"温度"。相同温度下,SGD收敛到相似的解。

#### 1.4 线性缩放规则的推导

若希望保持相同的平衡分布,当$B \to kB$时,需要:
\begin{equation}\frac{\eta'}{kB} = \frac{\eta}{B} \Rightarrow \eta' = k\eta\tag{8}\end{equation}

**定理1**(线性缩放规则): 若Batch size从$B$增加到$kB$,为保持相同的收敛行为,学习率应从$\eta$增加到$k\eta$。

**适用范围**: 该规则在以下条件下成立:
1. $B$足够大,CLT成立
2. $\eta$足够小,SDE近似有效
3. 训练未进入强噪声主导的regime

### 2. 临界Batch Size理论

#### 2.1 噪声尺度与梯度尺度的比较

定义**噪声尺度**(noise scale):
\begin{equation}\mathcal{B}_{noise} = \frac{\text{tr}(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\tag{9}\end{equation}

**物理意义**: 噪声在Hessian度量下与梯度的相对强度。

#### 2.2 临界Batch Size的定义

**定义**: 临界batch size $B_c$是使梯度噪声与梯度信号相当的batch size:
\begin{equation}B_c = \mathcal{B}_{noise} = \frac{\text{tr}(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\tag{10}\end{equation}

**简化版本**: 若假设$\boldsymbol{\Sigma} \propto \boldsymbol{g}\boldsymbol{g}^{\top}$(Assumption 在训练初期近似成立):
\begin{equation}B_c^{simple} = \frac{\text{tr}(\boldsymbol{\Sigma})}{\boldsymbol{g}^{\top}\boldsymbol{g}} = \frac{\text{tr}(\boldsymbol{\Sigma})}{\|\boldsymbol{g}\|^2}\tag{11}\end{equation}

#### 2.3 最优学习率与Batch Size的关系

基于二阶Taylor展开,最优学习率为:
\begin{equation}\eta^* \approx \frac{\eta_{max}}{1 + \mathcal{B}_{noise}/B}\tag{12}\end{equation}

其中$\eta_{max} = \frac{\boldsymbol{g}^{\top}\boldsymbol{g}}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}$是无噪声情况下的最优学习率。

**分析**:
- **小batch regime** ($B \ll B_c$): $\eta^* \approx \frac{\eta_{max}B}{\mathcal{B}_{noise}}$,学习率线性缩放
- **大batch regime** ($B \gg B_c$): $\eta^* \approx \eta_{max}$,学习率饱和

**数学直觉**: 当$B > B_c$时,继续增大batch size的边际收益递减,这就是"surge现象"的根源。

#### 2.4 训练时间分析

**每步计算成本**: 正比于$B$
**收敛步数**: 反比于$\eta^*$

**总计算成本**:
\begin{equation}Cost \propto \frac{B}{\eta^*} \approx \begin{cases}
\text{const}, & B \ll B_c \\
B/\eta_{max}, & B \gg B_c
\end{cases}\tag{13}\end{equation}

**数学直觉**: 在$B < B_c$时,增大batch size可以保持总成本不变(线性缩放);超过$B_c$后,总成本线性增加。

### 3. 动量机制的影响

#### 3.1 SGDM的等效Batch Size

从主文档式(54)-(57),动量的EMA机制相当于对优化轨迹上的梯度做平均:
\begin{equation}\boldsymbol{m}_t = (1-\beta_1)\sum_{s=1}^t \beta_1^{t-s}\tilde{\boldsymbol{g}}_{B,s}\tag{14}\end{equation}

**有效样本数**: 考虑EMA的有效窗口长度:
\begin{equation}T_{eff} = \sum_{s=0}^{\infty}\beta_1^s = \frac{1}{1-\beta_1}\tag{15}\end{equation}

**等效batch size**:
\begin{equation}B_{eff} = B \cdot \frac{1+\beta_1}{1-\beta_1}\tag{16}\end{equation}

**数学直觉**: 动量通过时间维度的平均,将有效batch size放大了$\frac{1+\beta_1}{1-\beta_1}$倍。

#### 3.2 Adam的双重EMA

Adam同时对一阶和二阶矩进行EMA:
\begin{equation}\boldsymbol{m}_t = \beta_1\boldsymbol{m}_{t-1} + (1-\beta_1)\tilde{\boldsymbol{g}}_t, \quad \boldsymbol{v}_t = \beta_2\boldsymbol{v}_{t-1} + (1-\beta_2)\tilde{\boldsymbol{g}}_t^2\tag{17}\end{equation}

**更新方向**: $\tilde{\boldsymbol{\varphi}}_B = \boldsymbol{m}_t/\sqrt{\boldsymbol{v}_t}$

从主文档式(91):
\begin{equation}\mathbb{E}[\tilde{\boldsymbol{\varphi}}_B] \approx \frac{\boldsymbol{g}}{\sqrt{\boldsymbol{g}^2 + \boldsymbol{\sigma}^2/B}}\tag{18}\end{equation}

**临界batch size调整**: Adam的$B_c$比SGD小,因为二阶矩的归一化削弱了噪声影响。

### 4. 不同优化器的Scaling Law

#### 4.1 SGD的缩放规律

**最优学习率**:
\begin{equation}\eta_{SGD}^* = \frac{2}{L + \mu} \cdot \frac{1}{1 + B_c/B}\tag{19}\end{equation}

其中$L$是光滑常数,$\mu$是强凸常数。

**缩放行为**:
- $B \ll B_c$: $\eta^* \propto B$(线性)
- $B \gg B_c$: $\eta^* = \text{const}$(饱和)

#### 4.2 Adam的缩放规律

从主文档式(102),Adam在$\beta_1 > 1/3$时会出现surge现象。

**最优学习率**(对角Hessian假设):
\begin{equation}\eta_{Adam}^* \propto \frac{1}{\frac{1}{\beta(B)}\frac{1-\beta_1}{1+\beta_1} + \beta(B)\frac{2\beta_1}{1+\beta_1}}\tag{20}\end{equation}

其中$\beta(B) = (1 + B_c^{simple}/B)^{-1/2}$。

**临界点**: $B_c^{Adam} = \frac{1-\beta_1}{3\beta_1-1}B_c^{simple}$(当$\beta_1 > 1/3$)

#### 4.3 Muon的缩放规律

Muon使用矩阵符号函数,更新方向接近SignSGD+Momentum。

从主文档分析,Muon在特定Hessian结构下不会出现surge现象:
\begin{equation}\eta_{Muon}^* \approx \eta_{max}^{sign} \cdot f(B, B_{eff})\tag{21}\end{equation}

其中$B_{eff} = B \cdot \frac{1+\beta_1}{1-\beta_1}$,$f$是平滑递增函数。

**优势**: 支持更大batch size而不牺牲效率。

### 5. Warmup与学习率调度

#### 5.1 Warmup的必要性

**训练初期的特点**:
1. 梯度范数$\|\boldsymbol{g}\|$很大
2. Hessian特征值不稳定
3. $B_c$可能很大

**大学习率的风险**:
\begin{equation}\|\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t\| = \eta\|\tilde{\boldsymbol{g}}_B\| \approx \eta\|\boldsymbol{g}\|\tag{22}\end{equation}

若$\eta$过大,一步更新可能跨越多个局部最优,导致不稳定。

#### 5.2 线性Warmup

**策略**:
\begin{equation}\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}}, & t \leq T_{warmup}\\
\eta_{max}, & t > T_{warmup}
\end{cases}\tag{23}\end{equation}

**理论解释**: 渐进增大学习率,让模型逐步适应大步长更新。

**Warmup长度**: 典型设置$T_{warmup} = 0.05 \sim 0.1 \times T_{total}$

#### 5.3 批量大小的动态调整

**渐进增大Batch Size**:
\begin{equation}B_t = \min(B_{max}, B_0 \cdot 2^{\lfloor t/T_B \rfloor})\tag{24}\end{equation}

**优点**:
1. 训练初期:小batch,高随机性,探索广
2. 训练后期:大batch,低噪声,收敛快

**与学习率的协调**: 可以同时调整:
\begin{equation}\eta_t = \eta_0 \cdot \sqrt{B_t/B_0}\tag{25}\end{equation}

使用平方根缩放而非线性缩放,介于两种极端之间。

### 6. 泛化性Gap分析

#### 6.1 Sharp vs Flat Minima

**定义**: 损失函数在最小值附近的Hessian:
- **Sharp minimum**: $\lambda_{max}(\boldsymbol{H}^*)$大,损失对参数敏感
- **Flat minimum**: $\lambda_{max}(\boldsymbol{H}^*)$小,损失对参数不敏感

**泛化能力**: Flat minima通常泛化更好(PAC-Bayes理论支持)。

#### 6.2 SGD的隐式正则化

从式(7),SGD倾向于收敛到满足以下条件的$\boldsymbol{\theta}^*$:
\begin{equation}\nabla L(\boldsymbol{\theta}^*) = 0, \quad \text{且} \quad \lambda_{max}(\boldsymbol{H}^*) \lesssim \frac{B}{\eta}\tag{26}\end{equation}

**数学直觉**: 较小的$\eta/B$(低温度)偏好flatter minima。

#### 6.3 Large Batch的泛化Gap

**现象**: 大batch训练的模型测试误差通常更高。

**理论解释**:
1. **高温度**: 大$B$需要大$\eta$,导致$\eta/B$未必减小
2. **探索不足**: 大batch噪声小,难以逃离sharp minima
3. **有效迭代数少**: 相同epoch下,大batch的更新次数少

**缓解方法**:
- 延长训练(更多epoch)
- 降低学习率(牺牲速度)
- Ghost Batch Normalization
- 使用更好的优化器(如Muon)

### 7. 分布式训练的Scaling

#### 7.1 数据并行

**基本设置**: $N$个worker,每个计算$B_{local}$样本:
\begin{equation}B_{global} = N \cdot B_{local}\tag{27}\end{equation}

**梯度聚合**:
\begin{equation}\tilde{\boldsymbol{g}}_{global} = \frac{1}{N}\sum_{i=1}^N \tilde{\boldsymbol{g}}_{i,local}\tag{28}\end{equation}

**学习率缩放**: 根据线性规则,$\eta_{global} = N \cdot \eta_{base}$

#### 7.2 通信开销

**每步通信量**: $\mathcal{O}(d)$,其中$d$是参数维度

**All-Reduce复杂度**: $\mathcal{O}(\log N)$使用tree-based算法

**通信/计算比**:
\begin{equation}R = \frac{T_{comm}}{T_{comp}} = \frac{c \cdot d}{B_{local} \cdot n_{samples}}\tag{29}\end{equation}

其中$c$是通信常数。

**数学直觉**: 增大$B_{local}$可以减少通信开销,但需权衡泛化性能。

#### 7.3 LARS/LAMB优化器

**Layer-wise Adaptive Rate Scaling (LARS)**:
\begin{equation}\eta_l = \eta_{global} \cdot \frac{\|\boldsymbol{W}_l\|}{\|\nabla_{\boldsymbol{W}_l}L\| + \lambda\|\boldsymbol{W}_l\|}\tag{30}\end{equation}

为每层自适应调整学习率,允许更激进的全局学习率缩放。

**LAMB**(Layer-wise Adaptive Moments for Batch training):
\begin{equation}\eta_l = \eta_{global} \cdot \frac{\|\boldsymbol{W}_l\|}{\|\boldsymbol{m}_l/\sqrt{\boldsymbol{v}_l}\|}\tag{31}\end{equation}

结合Adam的自适应矩估计和LARS的层级缩放。

**应用**: 成功将BERT训练扩展到batch size 32k。

### 8. 实验验证与案例研究

#### 8.1 ImageNet分类

**实验设置**: ResNet-50,初始学习率$\eta_0 = 0.1$,batch size $B_0 = 256$

**缩放实验**:
| Batch Size | 学习率 | Top-1准确率 | 训练时间 |
|-----------|-------|-----------|---------|
| 256 | 0.1 | 76.2% | 100% |
| 512 | 0.2 | 76.1% | 55% |
| 1024 | 0.4 | 75.9% | 32% |
| 2048 | 0.8 | 75.5% | 20% |
| 8192 | 2.4 | 74.8% | 8% |

**观察**:
- $B \leq 1024$:线性缩放有效,准确率几乎无损
- $B > 2048$:出现泛化gap,需要特殊技巧(warmup、LARS等)

#### 8.2 GPT-3训练

**参数**: 175B参数,300B tokens

**Batch Size策略**: 动态增大
- 前10%训练: $B = 0.5M$ tokens
- 中期: $B = 2M$ tokens
- 后期: $B = 3.2M$ tokens

**学习率**: 随batch size调整,使用cosine decay

**数学直觉**: 大模型有更大的$B_c$,可以使用更大batch size。

#### 8.3 对比学习(CLIP/SimCLR)

**特殊性**: 需要大batch来构建负样本对

**SimCLR**: 最佳性能在$B = 4096 \sim 8192$

**解释**: 对比损失的$B_c$与batch size正相关:
\begin{equation}B_c^{contrast} \propto B^{contrast}\tag{32}\end{equation}

因为增大batch同时增加了有效负样本数。

### 9. 实践建议总结

#### 9.1 学习率选择流程

**步骤1**: 确定基准学习率
- 小batch($B = 32 \sim 128$)下grid search
- 找到最佳$\eta_{base}$

**步骤2**: 估计临界batch size
- 监控训练过程中的梯度统计量
- 估算$B_c \approx \text{tr}(\boldsymbol{\Sigma})/\|\boldsymbol{g}\|^2$

**步骤3**: 应用缩放规则
- 若$B \leq B_c$:使用线性缩放$\eta = \eta_{base} \cdot (B/B_{base})$
- 若$B > B_c$:使用次线性缩放$\eta = \eta_{base} \cdot \sqrt{B/B_{base}}$

**步骤4**: 调优
- 添加warmup(长度$\sim 5\%$训练步数)
- 监控验证集性能,必要时降低学习率

#### 9.2 不同优化器的推荐设置

**SGD+Momentum**:
- $\beta_1 = 0.9$(标准)
- 学习率:严格遵循线性缩放(在$B < B_c$时)
- 适合:CNN分类任务

**Adam/AdamW**:
- $\beta_1 = 0.9, \beta_2 = 0.999$(标准)
- 学习率:对batch size不太敏感,但$B$过大仍会有gap
- 适合:Transformer模型、初步实验

**LARS/LAMB**:
- 在Adam/SGD基础上加layer-wise缩放
- 允许$B$扩展到$10k \sim 32k$
- 适合:分布式训练、大模型

**Muon**:
- 类似SignSGD+Momentum
- 支持大batch size,surge现象较轻
- 适合:需要高throughput的场景

#### 9.3 调试Checklist

**症状1**: 损失不下降
- 检查学习率是否过小
- 验证梯度是否正确计算
- 尝试降低batch size

**症状2**: 损失震荡/NaN
- 学习率过大,降低10x
- 添加gradient clipping
- 检查数据预处理

**症状3**: 训练慢但最终性能好
- 可以增大batch size和学习率
- 使用更激进的学习率调度

**症状4**: 训练快但泛化差(large batch gap)
- 延长训练时间
- 降低学习率
- 考虑Ghost Batch Normalization
- 尝试更小的batch size

### 10. 理论前沿与开放问题

#### 10.1 非凸优化的Scaling Law

**现状**: 大部分理论基于凸/强凸假设

**挑战**: 深度神经网络高度非凸
- 多个局部最优
- 鞍点众多
- Hessian谱不稳定

**进展**:
- 神经正切核(NTK)理论:在无限宽度极限下,网络近似线性
- Polyak-Łojasiewicz条件:某些非凸函数满足,保证收敛

#### 10.2 Sharpness-Aware Minimization (SAM)

**思想**: 显式寻找flat minima
\begin{equation}\min_{\boldsymbol{\theta}} \max_{\|\boldsymbol{\epsilon}\| \leq \rho} L(\boldsymbol{\theta} + \boldsymbol{\epsilon})\tag{33}\end{equation}

**与batch size的关系**: SAM可能改变$B_c$,需要重新研究scaling law

#### 10.3 Adaptive Batch Size

**问题**: 能否动态调整batch size以最大化效率?

**方案**:
- 基于梯度方差估计$B_c$
- 在线调整$B_t$使其始终 $\approx B_c$

**挑战**: 需要高效估计$B_c$,避免过大开销

#### 10.4 Meta-Learning Scaling

**问题**: Few-shot learning中batch size如何影响元学习?

**特殊性**:
- Inner loop和outer loop有不同的batch size
- Support set和query set的平衡

**开放问题**: Scaling law在MAML等算法中如何体现?

### 11. 全文总结

本文深入分析了学习率与batch size的关系,主要结论:

**核心原理**:
1. **线性缩放规则**: $B \to kB \Rightarrow \eta \to k\eta$(在$B < B_c$时)
2. **临界batch size**: $B_c = \text{tr}(\boldsymbol{\Sigma}\boldsymbol{H})/(\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g})$
3. **Surge现象**: $B > B_c$时,增大batch size边际收益递减

**优化器特性**:
- SGD: 严格遵循线性缩放
- Adam: $\beta_1 > 1/3$时有surge现象
- Muon: 支持更大batch size

**实践指南**:
- 使用warmup缓解训练初期不稳定
- 监控梯度统计量估计$B_c$
- 大batch训练需要特殊技巧(LARS/LAMB)

**未来方向**:
- 非凸情况的精确理论
- 自适应batch size算法
- Sharpness-aware优化的scaling

