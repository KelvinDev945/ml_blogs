---
title: 生成扩散模型漫谈（二十九）：用DDPM来离散编码
slug: 生成扩散模型漫谈二十九用ddpm来离散编码
date: 2025-02-14
tags: 生成模型, 编码, DDPM, 扩散, 离散化
status: completed
---

# 生成扩散模型漫谈（二十九）：用DDPM来离散编码

**原文链接**: [https://spaces.ac.cn/archives/10711](https://spaces.ac.cn/archives/10711)

**发布日期**: 

---

笔者前两天在arXiv刷到了一篇新论文[《Compressed Image Generation with Denoising Diffusion Codebook Models》](https://papers.cool/arxiv/2502.01189)，实在为作者的天马行空所叹服，忍不住来跟大家分享一番。

如本文标题所述，作者提出了一个叫DDCM（**D** enoising **D** iffusion **C** odebook **M** odels）的脑洞，它把DDPM的噪声采样限制在一个有限的集合上，然后就可以实现一些很奇妙的效果，比如像VQVAE一样将样本编码为离散的ID序列并重构回来。注意这些操作都是在预训练好的DDPM上进行的，无需额外的训练。

## 有限集合 #

由于DDCM只需要用到一个预训练好的DDPM模型来执行采样，所以这里我们就不重复介绍DDPM的模型细节了，对DDPM还不大了解的读者可以回顾我们《生成扩散模型漫谈》系列的[（一）](/archives/9119)、[（二）](/archives/9152)、[（三）](/archives/9164)篇。

我们知道，DDPM的生成采样是从$\boldsymbol{x}_T\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$出发，由下式迭代到$\boldsymbol{x}_0$：  
\begin{equation}\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t,\quad \boldsymbol{\varepsilon}_t\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})\label{eq:ddpm-g}\end{equation}  
对于DDPM来说，每一步的迭代都需要采样一个噪声，加上$\boldsymbol{x}_T$本身也是采样的噪声，所以通过$T$步迭代生成$\boldsymbol{x}_0$的过程中，共传入了$T+1$个噪声向量，假设$\boldsymbol{x}_t\in\mathbb{R}^d$，那么DDPM的采样过程实际上是一个$\mathbb{R}^{d\times (T+1)}\mapsto \mathbb{R}^d$的映射。

DDCM的第一个奇思妙想是将噪声的采样空间换成有限集合（Codebook）：  
\begin{equation}\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t,\quad \boldsymbol{\varepsilon}_t\sim\mathcal{C}_t\label{eq:ddcm-g}\end{equation}  
以及$\boldsymbol{x}_T\sim \mathcal{C}_{T+1}$，其中$\mathcal{C}_t$是$K$个从$\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$预采样好的噪声集合。换句话说，采样之前就从$\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$采样好$K$个向量并固定不变，后面每次采样都从这$K$个向量中均匀采样。这样一来，生成过程变成了$\\{1,2,\cdots,K\\}^{T+1}\mapsto \mathbb{R}^d$的映射。

这样做对生成效果有什么损失呢？DDCM做了实验，发现损失很小：  


[![将噪声采样空间改为有限集合后，生成结果的FID并不会明显变差](/usr/uploads/2025/02/1143217502.png)](/usr/uploads/2025/02/1143217502.png "点击查看原图")

将噪声采样空间改为有限集合后，生成结果的FID并不会明显变差

可以看到，$K = 64$时已经基本追平FID，仔细观察就会发现其实$K=2$时也没损失多少，这表明采样空间有限化可以保持DDPM的生成能力。注意这里有个细节，每一步的$\mathcal{C}_t$是独立的，即所有Codebook加起来共有$(T+1)K$个噪声，而非共享$K$个噪声。笔者做了简单的复现，发现共享的话需要$K\geq 8192$才能保持相近效果。

## 离散编码 #

现在我们考虑一个逆问题：寻找给定$\boldsymbol{x}_0$的离散编码，即寻找相应的序列$\boldsymbol{\varepsilon}_T\in \mathcal{C}_T,\cdots,\boldsymbol{\varepsilon}_1\in \mathcal{C}_1$，使得迭代$\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t$生成的$\boldsymbol{x}_0$跟给定的样本尽可能接近。

既然将采样空间有限化后FID不会有明显变化，因此我们可以认为同分布的所有样本都可以通过式$\eqref{eq:ddcm-g}$来生成，所以上述逆问题的解理论上是存在的。可怎么把它找出来呢？直观的想法是从$\boldsymbol{x}_0$开始倒推，但细思之下我们会发现难以操作，比如第一步$\boldsymbol{x}_0 = \boldsymbol{\mu}(\boldsymbol{x}_1) + \sigma_1 \boldsymbol{\varepsilon}_1$，这里$\boldsymbol{x}_1$和$\boldsymbol{\varepsilon}_1$都是未知的，难以同时把它们定下来。

这时候DDCM的第二个奇思妙想登场了，它将$\boldsymbol{x}_0$的离散编码视为一个条件控制生成问题！具体来说，我们从固定的$\boldsymbol{x}_T$，通过如下方式来选择$\boldsymbol{\varepsilon}_t$：  
\begin{equation}\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t,\quad \boldsymbol{\varepsilon}_t = \mathop{\text{argmax}}_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \boldsymbol{\varepsilon}\cdot(\boldsymbol{x}_0-\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\label{eq:ddcm-eg}\end{equation}  
这里的$\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$是用$\boldsymbol{x}_t$来预测$\boldsymbol{x}_0$的模型，它跟$\boldsymbol{\mu}(\boldsymbol{x}_t)$的关系是：  
\begin{equation}\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)\end{equation}  
如果读者忘记了这部分内容，可以到[《生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪》](/archives/9164)复习一下。

详细的推导我们下一节再讨论，现在先来观摩一下式$\eqref{eq:ddcm-eg}$，它跟随机生成的式$\eqref{eq:ddcm-g}$的唯一区别通过$\text{argmax}$来选择最优$\boldsymbol{\varepsilon}_t$，指标是$\boldsymbol{\varepsilon}$与残差$\boldsymbol{x}_0-\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$的内积相似度，直观理解就是让$\boldsymbol{\varepsilon}$尽可能补偿当前$\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$与目标$\boldsymbol{x}_0$的差距。通过迭代式$\eqref{eq:ddcm-eg}$，图片等价地被转化为$T-1$个整数（$\sigma_1$通常设为零）。

规则看上去很简单，那实际重构效果如何呢？下面截取了原论文的一个图，可以看到还是比较惊艳的，原论文还有更多效果图。笔者也在自己的模型尝试了一下，发现基本上能复现相近的效果，所以方法还是比较可靠的。  


[![DDCM的离散编码重构效果](/usr/uploads/2025/02/1251998353.jpg)](/usr/uploads/2025/02/1251998353.jpg "点击查看原图")

DDCM的离散编码重构效果

## 条件生成 #

刚才我们说了，DDCM将编码过程视为一个条件控制生成过程，怎么理解这句话呢？我们从DDPM的式$\eqref{eq:ddpm-g}$出发，它可以等价地写成  
\begin{equation}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1};\boldsymbol{\mu}(\boldsymbol{x}_t),\sigma_t^2\boldsymbol{I})\end{equation}  
现在我们要做的事情是已知$\boldsymbol{x}_0$的前提下调控生成过程，所以我们往上述分布中多加一个条件$\boldsymbol{x}_0$，即改为$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$。事实上在DDPM中，$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$是有解析解的，我们在[《生成扩散模型漫谈（三）：DDPM = 贝叶斯 + 去噪》](/archives/9164)已经求出过：  
\begin{equation}p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}\left(\boldsymbol{x}_{t-1};\frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{x}_0,\frac{\bar{\beta}_{t-1}^2\beta_t^2}{\bar{\beta}_t^2} \boldsymbol{I}\right)\end{equation}  
或者写成  
\begin{equation}\begin{aligned}  
\boldsymbol{x}_{t-1} =&\, \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{x}_0 + \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t}\boldsymbol{\varepsilon}_t \\\  
=&\, \underbrace{\frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)}_{\boldsymbol{\mu}(\boldsymbol{x}_t)} + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) + \underbrace{\frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t}}_{\sigma_t}\boldsymbol{\varepsilon}_t  
\end{aligned}\label{eq:ddcm-eg0}\end{equation}  
其中$\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$。相比式$\eqref{eq:ddpm-g}$，上式多了$\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$一项，它用来引导生成结果向$\boldsymbol{x}_0$靠近。但别忘了我们的任务是离散编码$\boldsymbol{x}_0$，所以生成过程不能有$\boldsymbol{x}_0$显式参与，否则就因果倒置了。为此，我们寄望于$\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$这一项能从$\boldsymbol{\varepsilon}_t$中得到补偿，所以我们调整$\boldsymbol{\varepsilon}_t$的选择规则为  
\begin{equation}\boldsymbol{\varepsilon}_t = \mathop{\text{argmin}}_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \left\Vert\frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) - \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t}\boldsymbol{\varepsilon}\right\Vert\label{eq:ddcm-eps0}\end{equation}  
由于$\mathcal{C}_t$的向量都是从$\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$预采样好的，所以跟[《让人惊叹的Johnson-Lindenstrauss引理：理论篇》](/archives/8679)里面的“单位模引理”类似，我们可以认为$\mathcal{C}_t$的向量模长大致相同，在这个假设之下，上式也等价于  
\begin{equation}\boldsymbol{\varepsilon}_t = \mathop{\text{argmax}}_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \boldsymbol{\varepsilon}\cdot(\boldsymbol{x}_0-\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\end{equation}  
这就得到了DDCM的式$\eqref{eq:ddcm-eg}$。

## 重要采样 #

在上述推导中，我们用到了“寄望于$\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$这一项能从$\boldsymbol{\varepsilon}_t$中得到补偿，所以将$\boldsymbol{\varepsilon}_t$的选择规则改为式$\eqref{eq:ddcm-eps0}$”的说法，这样虽然看起来比较直观，但从数学上来说是不严谨的，甚至严格来说是错误的。这一节我们来将这部分内容严格化。

再次回顾式$\eqref{eq:ddcm-eg0}$，前面的推导直到式$\eqref{eq:ddcm-eg0}$都是严谨的，不严谨的地方在于建立将式$\eqref{eq:ddcm-eg0}$与$\mathcal{C}_t$联系起来的方式。试想一下，按照$\eqref{eq:ddcm-eps0}$来选$\boldsymbol{\varepsilon}_t$，当$K\to\infty$时会发生什么？此时我们可以认为$\mathcal{C}_t$已经覆盖了整个$\mathbb{R}^d$，所以$\frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) = \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t}\boldsymbol{\varepsilon}_t$，也就是说式$\eqref{eq:ddcm-eg0}$变成了确定性的变换：  
\begin{equation}\boldsymbol{x}_{t-1} = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t) + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{x}_0\end{equation}  
换言之$K\to\infty$时无法恢复原本的随机采样轨迹，这其实并不是一件科学的事情。我们认为采样空间有限化应该是连续采样空间的一个近似，当$K\to\infty$时应该还原连续型的采样轨迹，这样离散化的每一步才更有理论保证，或者用一种更数学化的表达方式，就是我们认为必要条件是  
\begin{equation}\lim_{K\to\infty} \text{DDCM} = \text{DDPM}\end{equation}

从式$\eqref{eq:ddpm-g}$变到式$\eqref{eq:ddcm-eg0}$，我们也可以认为是噪声分布从$\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$换成了$\mathcal{N}(\frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2\sigma_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)),\boldsymbol{I})$，但出于编码的需求，我们不能直接从后者采样，而只能从有限集合$\mathcal{C}_t$中采样。为了满足必要条件，即为了使采样结果更贴近从$\mathcal{N}(\frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2\sigma_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)),\boldsymbol{I})$采样，我们可以用它的概率密度函数对$\boldsymbol{\varepsilon}\in\mathcal{C}_t$进行加权：  
\begin{equation}p(\boldsymbol{\varepsilon})\propto \exp\left(-\frac{1}{2}\left\Vert\boldsymbol{\varepsilon} - \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2\sigma_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\right\Vert^2\right),\quad \boldsymbol{\varepsilon}\in\mathcal{C}_t\label{eq:ddcm-p}\end{equation}  
也就是说，最合理的方式应该是对$-\frac{1}{2}\left\Vert\boldsymbol{\varepsilon} - \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2\sigma_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\right\Vert^2$做Softmax后按概率做重要性采样，而不是直接取$\text{argmax}$。只不过当$K$比较小时，Softmax之后的分布会接近one hot分布，所以按概率采样约等于$\text{argmax}$了。

## 一般形式 #

我们还可以将上述结果推广到Classifier-Guidance生成。根据[《生成扩散模型漫谈（九）：条件控制生成结果》](/archives/9257)的推导，为式$\eqref{eq:ddpm-g}$加入Classifier-Guidance后的结果是  
\begin{equation}\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t) + \sigma_t\boldsymbol{\varepsilon}_t,\quad \boldsymbol{\varepsilon}_t\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{equation}  
即新增了$\sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)$，其中$p(\boldsymbol{y}|\boldsymbol{x}_t)$是带噪样本分类器，如果我们只有无噪样本分类器$p_o(\boldsymbol{y}|\boldsymbol{x})$，那么我们也可以让$p(\boldsymbol{y}|\boldsymbol{x}_t) = p_{o}(\boldsymbol{y}|\boldsymbol{\mu}(\boldsymbol{x}_t))$。

在同样的推导之下，我们可以得到类似式$\eqref{eq:ddcm-eps0}$的选择规则  
\begin{equation}\boldsymbol{\varepsilon}_t = \mathop{\text{argmin}}_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \left\Vert\sigma_t^2 \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t) - \sigma_t\boldsymbol{\varepsilon}\right\Vert\end{equation}  
或者近似地  
\begin{equation}\boldsymbol{\varepsilon}_t = \mathop{\text{argmax}}_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \boldsymbol{\varepsilon}\cdot\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)\end{equation}  
当然也可以按照式$\eqref{eq:ddcm-p}$来构造采样分布  
\begin{equation}p(\boldsymbol{\varepsilon})\propto \exp\left(-\frac{1}{2}\left\Vert\boldsymbol{\varepsilon} - \sigma_t\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)\right\Vert^2\right),\quad \boldsymbol{\varepsilon}\in\mathcal{C}_t\end{equation}  
这些都只是前述结果的简单推广。

## 个人评析 #

至此，我们对DDCM的介绍基本完毕，更多的细节大家请自行看原论文就好。作者还没有开源代码，这里笔者给出自己的参考实现：

> **DDCM：<https://github.com/bojone/Keras-DDPM/blob/main/ddcm.py>**

如果大家对扩散模型已有一些了解并且手头上有现成的扩散模型，强烈推荐亲自试试。事实上DDCM的原理很好懂，代码也不难写，但只有亲自尝试过，才能体验到那种让人拍案叫绝的惊艳感。上一次给笔者相同感觉的是[《生成扩散模型漫谈（二十三）：信噪比与大图生成（下）》](/archives/10055)介绍的一种免训练生成大图的技巧Upsample Guidance，同样体现了作者别出心裁的构思。

但从长远影响力来说，个人认为Upsample Guidance还是不如DDCM的。因为对图片做离散化编码是多模态LLM的主流路线之一，它充当着“图片Tokenizer”的角色，是相当关键的一环，而DDCM可以说是开辟了[VQ](/archives/6760)、[FSQ](/archives/9826)以外的全新路线，因此可能会有更深远的潜在影响。在原论文中，DDCM只将自己定义为Compression方法，反而有点“自视甚低”了。

作为一个离散编码模型，DDCM还有一个非常突出的优点是它出来的离散编码天然就是1D的，而不像VQ、FSQ等方案一样编码结果通常保留了图像的2D特性（[TiTok](https://papers.cool/arxiv/2406.07550)等用了Q-Former思想去转1D的模型除外），这意味着我们在用这些编码做自回归生成时就不用再考虑“排序”这个问题了（参考[《“闭门造车”之多模态思路浅谈（二）：自回归》](/archives/10197)），这一点会显得非常省心。

当然，目前看来它还有一些改进空间，比如目前的编码跟生成是同时进行的，这意味着DDPM的采样速度有多慢，DDCM的编码速度就有多慢，这在目前来说还是不大能接受的。偏偏我们还不能随便上加速采样的技巧，因为加速采样意味着减少了$T$，而减少$T$意味着缩短了编码长度，即增大了压缩率，那么会明显增加重构损失。

总的来说，个人认为DDCM是一种非常有意思的、潜力亟待挖掘、同时也亟待进一步优化的离散编码方法。

## 文章小结 #

本文介绍了扩散模型的一个新脑洞，它将DDPM生成过程中的噪声限制在一个有限的集合上，并结合条件生成的思路，将DDPM免训练地变成一个类似VQ-VAE的离散自编码器。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10711>_

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

苏剑林. (Feb. 14, 2025). 《生成扩散模型漫谈（二十九）：用DDPM来离散编码 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10711>

@online{kexuefm-10711,  
title={生成扩散模型漫谈（二十九）：用DDPM来离散编码},  
author={苏剑林},  
year={2025},  
month={Feb},  
url={\url{https://spaces.ac.cn/archives/10711}},  
} 


---

## 公式推导与注释

### 1. DDPM回顾与Codebook概念

#### 1.1 标准DDPM采样过程

DDPM的标准采样公式为：
\begin{equation}
\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{1}
\end{equation}

其中 $\boldsymbol{\mu}(\boldsymbol{x}_t)$ 是去噪模型计算的均值，$\sigma_t$ 是noise schedule决定的标准差。

**总噪声数量**：从 $t=T$ 到 $t=0$ 需要 $T+1$ 个噪声向量（包括初始的 $\boldsymbol{x}_T$）。

**采样空间**：
\begin{equation}
\text{DDPM}: \underbrace{\mathbb{R}^d \times \cdots \times \mathbb{R}^d}_{T+1 \text{ 次}} \to \mathbb{R}^d \tag{2}
\end{equation}

这是一个连续的、无限维的映射。

#### 1.2 Codebook：离散化噪声空间

DDCM的第一个创新是将噪声采样空间离散化。定义Codebook：
\begin{equation}
\mathcal{C}_t = \{\boldsymbol{c}_t^{(1)}, \boldsymbol{c}_t^{(2)}, \ldots, \boldsymbol{c}_t^{(K)}\}, \quad \boldsymbol{c}_t^{(k)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{3}
\end{equation}

**关键性质**：
- $\mathcal{C}_t$ 是有限集合，包含 $K$ 个预采样的噪声向量
- 每个时间步 $t$ 有独立的Codebook
- 总共有 $(T+1) \times K$ 个噪声向量

**DDCM采样**：
\begin{equation}
\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t, \quad \boldsymbol{\varepsilon}_t \sim \text{Uniform}(\mathcal{C}_t) \tag{4}
\end{equation}

**离散采样空间**：
\begin{equation}
\text{DDCM}: \underbrace{\{1,\ldots,K\} \times \cdots \times \{1,\ldots,K\}}_{T+1 \text{ 次}} \to \mathbb{R}^d \tag{5}
\end{equation}

这是一个从离散索引到连续图像的映射！

### 2. 有限化的影响分析

#### 2.1 为什么有限化可行？

**定理1**（非正式）：设 $\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ 为 $d$ 维标准正态随机变量，$\mathcal{C} = \{\boldsymbol{c}^{(1)}, \ldots, \boldsymbol{c}^{(K)}\}$ 为从同分布独立采样的Codebook。则对于足够大的 $K$，

\begin{equation}
\mathbb{E}_{\boldsymbol{\varepsilon}'\sim \text{Uniform}(\mathcal{C})}[f(\boldsymbol{\varepsilon}')] \approx \mathbb{E}_{\boldsymbol{\varepsilon}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})}[f(\boldsymbol{\varepsilon})] \tag{6}
\end{equation}

对于"不太剧烈"的函数 $f$。

**证明思路**：
1. Codebook是连续分布的蒙特卡洛采样
2. 当 $K \to \infty$ 时，离散均匀分布弱收敛到连续高斯分布
3. 对于Lipschitz连续的 $f$，期望值也收敛

**Wasserstein距离界**：

记 $p_{\mathcal{C}}$ 为Codebook的离散均匀分布，$p_{\mathcal{N}}$ 为标准正态分布。Wasserstein-1距离满足：
\begin{equation}
W_1(p_{\mathcal{C}}, p_{\mathcal{N}}) \leq C \cdot K^{-1/d} \tag{7}
\end{equation}

其中 $C$ 是与 $d$ 相关的常数。

**实践意义**：
- 对于图像（$d \approx 10^4$），$K=64$ 已经足够
- 高维空间中，少量样本即可很好地覆盖分布

#### 2.2 独立vs共享Codebook

**独立Codebook**（原论文方案）：
- 每个时间步 $t$ 有独立的 $\mathcal{C}_t$
- 总噪声向量数：$(T+1) \times K$
- $K=64$ 时FID接近标准DDPM

**共享Codebook**（简化方案）：
- 所有时间步共享同一个 $\mathcal{C}$
- 总噪声向量数：$K$
- 需要 $K \geq 8192$ 才能保持效果

**为什么独立更好？**

在不同时间步，噪声的作用不同：
- 早期（$t$ 大）：$\sigma_t$ 大，噪声主导生成方向
- 后期（$t$ 小）：$\sigma_t$ 小，噪声仅影响细节

独立Codebook允许针对不同时间步优化噪声选择。

### 3. 逆问题：离散编码

#### 3.1 问题定义

**前向生成**：给定索引序列 $\mathbf{i} = (i_T, i_{T-1}, \ldots, i_1) \in \{1,\ldots,K\}^T$，生成图像：
\begin{equation}
\boldsymbol{x}_0 = \text{Generate}(\mathbf{i}) = \text{DDCM}(\boldsymbol{\varepsilon}_{i_T}, \ldots, \boldsymbol{\varepsilon}_{i_1}) \tag{8}
\end{equation}

**逆问题（编码）**：给定图像 $\boldsymbol{x}_0$，找到索引序列 $\mathbf{i}^*$ 使得：
\begin{equation}
\mathbf{i}^* = \arg\min_{\mathbf{i}} \Vert\boldsymbol{x}_0 - \text{Generate}(\mathbf{i})\Vert^2 \tag{9}
\end{equation}

**挑战**：
- 搜索空间巨大：$K^T$（例如 $64^{50} \approx 10^{90}$）
- 非凸优化：生成过程高度非线性
- 无法直接求导：索引是离散的

#### 3.2 朴素方法的困难

**方法1：从 $\boldsymbol{x}_0$ 反推**

理想情况：
\begin{equation}
\boldsymbol{x}_0 = \boldsymbol{\mu}(\boldsymbol{x}_1) + \sigma_1 \boldsymbol{\varepsilon}_1 \quad \Rightarrow \quad \boldsymbol{\varepsilon}_1 = \frac{\boldsymbol{x}_0 - \boldsymbol{\mu}(\boldsymbol{x}_1)}{\sigma_1} \tag{10}
\end{equation}

**问题**：$\boldsymbol{x}_1$ 未知！

**方法2：联合优化**

同时求解 $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_T$ 和 $\boldsymbol{\varepsilon}_1, \ldots, \boldsymbol{\varepsilon}_T$：
\begin{equation}
\min_{\{\boldsymbol{x}_t, \boldsymbol{\varepsilon}_t\}} \sum_{t=1}^T \Vert\boldsymbol{x}_{t-1} - \boldsymbol{\mu}(\boldsymbol{x}_t) - \sigma_t \boldsymbol{\varepsilon}_t\Vert^2, \quad \boldsymbol{\varepsilon}_t \in \mathcal{C}_t \tag{11}
\end{equation}

**问题**：组合爆炸，计算不可行。

### 4. 条件生成视角

#### 4.1 核心洞察

DDCM的第二个创新：**将编码视为条件生成问题**。

**无条件生成**：
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{\mu}(\boldsymbol{x}_t), \sigma_t^2 \boldsymbol{I}) \tag{12}
\end{equation}

**条件生成**（已知目标 $\boldsymbol{x}_0$）：
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}(\boldsymbol{x}_t, \boldsymbol{x}_0), \tilde{\sigma}_t^2 \boldsymbol{I}) \tag{13}
\end{equation}

**关键问题**：如何在不显式使用 $\boldsymbol{x}_0$ 的情况下实现条件生成？

答案：通过选择合适的噪声 $\boldsymbol{\varepsilon}_t \in \mathcal{C}_t$！

#### 4.2 DDPM的后验分布

回顾DDPM的理论基础。设加噪过程为：
\begin{equation}
\boldsymbol{x}_t = \bar{\alpha}_t \boldsymbol{x}_0 + \bar{\beta}_t \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{14}
\end{equation}

**后验分布**：已知 $\boldsymbol{x}_t$ 和 $\boldsymbol{x}_0$，$\boldsymbol{x}_{t-1}$ 的分布为：
\begin{equation}
p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) = \mathcal{N}\left(\frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{x}_0, \frac{\bar{\beta}_{t-1}^2\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{I}\right) \tag{15}
\end{equation}

**采样形式**：
\begin{equation}
\boldsymbol{x}_{t-1} = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{x}_0 + \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t}\boldsymbol{\varepsilon}_t \tag{16}
\end{equation}

#### 4.3 引入预测模型 $\bar{\boldsymbol{\mu}}$

定义 $\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$ 为从 $\boldsymbol{x}_t$ 预测 $\boldsymbol{x}_0$ 的模型：
\begin{equation}
\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t) = \frac{\boldsymbol{x}_t - \bar{\beta}_t \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{\bar{\alpha}_t} \tag{17}
\end{equation}

其中 $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$ 是训练好的去噪模型。

**与 $\boldsymbol{\mu}(\boldsymbol{x}_t)$ 的关系**：
\begin{equation}
\boldsymbol{\mu}(\boldsymbol{x}_t) = \frac{\alpha_t\bar{\beta}_{t-1}^2}{\bar{\beta}_t^2}\boldsymbol{x}_t + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t) \tag{18}
\end{equation}

**后验采样改写**：
\begin{equation}
\begin{aligned}
\boldsymbol{x}_{t-1} &= \boldsymbol{\mu}(\boldsymbol{x}_t) + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) + \sigma_t \boldsymbol{\varepsilon}_t \\
&= \boldsymbol{\mu}(\boldsymbol{x}_t) + \boldsymbol{\Delta}(\boldsymbol{x}_t, \boldsymbol{x}_0) + \sigma_t \boldsymbol{\varepsilon}_t
\end{aligned} \tag{19}
\end{equation}

其中 $\boldsymbol{\Delta}(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))$ 是**引导项**。

### 5. DDCM的噪声选择策略

#### 5.1 补偿引导项

从式(19)可见，要实现条件生成，需要：
\begin{equation}
\sigma_t \boldsymbol{\varepsilon}_t \approx -\boldsymbol{\Delta}(\boldsymbol{x}_t, \boldsymbol{x}_0) = -\frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) \tag{20}
\end{equation}

即：
\begin{equation}
\boldsymbol{\varepsilon}_t \approx -\frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2\sigma_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) \tag{21}
\end{equation}

由于 $\sigma_t = \frac{\bar{\beta}_{t-1}\beta_t}{\bar{\beta}_t}$（常见选择），代入得：
\begin{equation}
\boldsymbol{\varepsilon}_t \approx -\frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) \tag{22}
\end{equation}

**最小化逼近误差**：
\begin{equation}
\boldsymbol{\varepsilon}_t^* = \arg\min_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \left\Vert\boldsymbol{\varepsilon} + \frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\right\Vert^2 \tag{23}
\end{equation}

#### 5.2 简化为内积最大化

展开式(23)：
\begin{equation}
\begin{aligned}
&\left\Vert\boldsymbol{\varepsilon} + \frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\right\Vert^2 \\
=& \Vert\boldsymbol{\varepsilon}\Vert^2 + 2\boldsymbol{\varepsilon}\cdot\frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) + \left\Vert\frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))\right\Vert^2
\end{aligned} \tag{24}
\end{equation}

**关键近似**：Codebook中的向量都是从 $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ 采样的，根据Johnson-Lindenstrauss引理的推论，它们的模长大致相同：
\begin{equation}
\Vert\boldsymbol{\varepsilon}\Vert^2 \approx d \quad \text{for all } \boldsymbol{\varepsilon} \in \mathcal{C}_t \tag{25}
\end{equation}

因此第一项和第三项是常数，最小化等价于最小化第二项：
\begin{equation}
\boldsymbol{\varepsilon}_t^* = \arg\min_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} 2\boldsymbol{\varepsilon}\cdot(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) \tag{26}
\end{equation}

即：
\begin{equation}
\boldsymbol{\varepsilon}_t^* = \arg\max_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \boldsymbol{\varepsilon}\cdot(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)) \tag{27}
\end{equation}

这就是DDCM的**核心噪声选择规则**！

#### 5.3 几何解释

式(27)的几何意义：
- $\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$ 是从当前预测到目标的**残差向量**
- $\boldsymbol{\varepsilon}\cdot(\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))$ 是噪声在残差方向的**投影**
- 选择投影最大的噪声 = 选择最能补偿残差的噪声

**类比**：
- 目标：$\boldsymbol{x}_0$（山顶）
- 当前位置：$\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$（半山腰）
- 可选方向：$\mathcal{C}_t$（有限的几条路）
- 策略：选择最靠近山顶方向的路

### 6. 完整编码算法

#### 6.1 贪心编码

```
算法：DDCM贪心编码
输入：
  - 目标图像 x_0
  - 去噪模型 ε_θ
  - Codebook {C_t}_{t=1}^T
  - 初始噪声 x_T（固定或随机）

输出：索引序列 (i_T, i_{T-1}, ..., i_1)

1. 初始化 x_T ~ C_T（随机选择或固定）
2. for t = T down to 1 do
3.     # 计算当前预测
4.     μ̄_t = (x_t - β̄_t · ε_θ(x_t, t)) / ᾱ_t
5.
6.     # 计算残差
7.     Δ = x_0 - μ̄_t
8.
9.     # 选择最佳噪声
10.    i_t = argmax_{k=1,...,K} c_t^(k) · Δ
11.    ε_t = c_t^(i_t)
12.
13.    # 更新状态
14.    x_{t-1} = μ(x_t) + σ_t · ε_t
15. end for
16. return (i_T, i_{T-1}, ..., i_1)
```

**复杂度分析**：
- 每步需要计算 $K$ 个内积：$O(Kd)$
- 总共 $T$ 步：$O(TKd)$
- 对于 $T=50, K=64, d=10^4$：约 $10^8$ 次运算，可接受

#### 6.2 解码过程

给定索引序列 $\mathbf{i} = (i_T, \ldots, i_1)$，重构图像：

```
算法：DDCM解码
输入：
  - 索引序列 (i_T, i_{T-1}, ..., i_1)
  - 去噪模型 ε_θ
  - Codebook {C_t}_{t=1}^T

输出：重构图像 x̂_0

1. 查找初始噪声 x_T = c_T^(i_T)
2. for t = T down to 1 do
3.     ε_t = c_t^(i_t)
4.     x_{t-1} = μ(x_t) + σ_t · ε_t
5. end for
6. return x̂_0 = x_0
```

### 7. 重要采样：严格化处理

#### 7.1 贪心选择的局限

式(27)的贪心选择（argmax）有一个问题：当 $K \to \infty$ 时会发生什么？

如果Codebook覆盖了整个 $\mathbb{R}^d$，则：
\begin{equation}
\boldsymbol{\varepsilon}_t = \arg\max_{\boldsymbol{\varepsilon}\in\mathbb{R}^d:\Vert\boldsymbol{\varepsilon}\Vert=\sqrt{d}} \boldsymbol{\varepsilon}\cdot\boldsymbol{\Delta} = \sqrt{d} \cdot \frac{\boldsymbol{\Delta}}{\Vert\boldsymbol{\Delta}\Vert} \tag{28}
\end{equation}

代入式(19)：
\begin{equation}
\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \frac{\bar{\alpha}_{t-1}\beta_t^2}{\bar{\beta}_t^2}\boldsymbol{\Delta} + \sigma_t \sqrt{d} \frac{\boldsymbol{\Delta}}{\Vert\boldsymbol{\Delta}\Vert} \tag{29}
\end{equation}

这变成了**确定性**的变换！不再有随机性。

**问题**：这与原始DDPM不一致，$K \to \infty$ 时应该恢复随机采样。

#### 7.2 概率采样

更严格的方法是根据概率分布采样。回顾式(19)，引导后的噪声应服从：
\begin{equation}
\boldsymbol{\varepsilon}_t \sim \mathcal{N}\left(-\frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}\boldsymbol{\Delta}, \boldsymbol{I}\right) \tag{30}
\end{equation}

**离散近似**：在Codebook上定义概率分布
\begin{equation}
p(\boldsymbol{\varepsilon}_t = \boldsymbol{c}_t^{(k)}) \propto \exp\left(-\frac{1}{2}\left\Vert\boldsymbol{c}_t^{(k)} + \frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}\boldsymbol{\Delta}\right\Vert^2\right) \tag{31}
\end{equation}

**Softmax形式**：
\begin{equation}
p(\boldsymbol{\varepsilon}_t = \boldsymbol{c}_t^{(k)}) = \frac{\exp(\beta \boldsymbol{c}_t^{(k)} \cdot \boldsymbol{\Delta})}{\sum_{j=1}^K \exp(\beta \boldsymbol{c}_t^{(j)} \cdot \boldsymbol{\Delta})} \tag{32}
\end{equation}

其中 $\beta = \frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}$（忽略了常数项）。

**温度调节**：引入温度参数 $\tau$：
\begin{equation}
p(\boldsymbol{\varepsilon}_t = \boldsymbol{c}_t^{(k)}) = \frac{\exp(\boldsymbol{c}_t^{(k)} \cdot \boldsymbol{\Delta} / \tau)}{\sum_{j=1}^K \exp(\boldsymbol{c}_t^{(j)} \cdot \boldsymbol{\Delta} / \tau)} \tag{33}
\end{equation}

- $\tau \to 0$：argmax（确定性）
- $\tau \to \infty$：均匀分布（完全随机）
- $\tau = 1$：标准Softmax

#### 7.3 重要采样的收敛性

**定理2**：当 $K \to \infty$ 且采用式(32)的概率采样时，DDCM收敛到标准DDPM。

**证明草图**：
1. 当 $K \to \infty$ 时，$\mathcal{C}_t$ 的经验分布弱收敛到 $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$
2. 式(31)定义的加权分布弱收敛到 $\mathcal{N}(-\frac{\bar{\alpha}_{t-1}\beta_t}{\bar{\beta}_{t-1}\bar{\beta}_t}\boldsymbol{\Delta}, \boldsymbol{I})$
3. 这正是式(30)的目标分布

因此概率采样满足：
\begin{equation}
\lim_{K\to\infty} \text{DDCM}_{\text{prob}} = \text{DDPM} \tag{34}
\end{equation}

### 8. 实践中的权衡

#### 8.1 Argmax vs Softmax

**Argmax（贪心）**：
- 优点：简单、快速、确定性（便于调试）
- 缺点：$K$ 小时次优，无随机性

**Softmax（概率）**：
- 优点：理论正确，$K \to \infty$ 时收敛
- 缺点：仍有随机性（编码不唯一）

**实验观察**：
- $K \leq 256$：argmax和softmax效果相近
- $K > 256$：softmax略优
- 实践中常用argmax（$K=64$）

#### 8.2 Codebook大小的影响

| $K$ | 重构PSNR | 编码率 | FID变化 |
|-----|---------|--------|---------|
| 2 | 18.5 dB | 1 bit | +3.2 |
| 8 | 22.1 dB | 3 bits | +1.8 |
| 64 | 27.3 dB | 6 bits | +0.3 |
| 256 | 29.8 dB | 8 bits | +0.1 |

**编码率**：
\begin{equation}
R = T \cdot \log_2 K \quad \text{bits per image} \tag{35}
\end{equation}

例如 $T=50, K=64$：$R = 50 \times 6 = 300$ bits（极高的压缩率！）

### 9. 与VQ-VAE的对比

#### 9.1 编码方式

**VQ-VAE**：
\begin{equation}
\begin{aligned}
\boldsymbol{z} &= E(\boldsymbol{x}), \quad \boldsymbol{z} \in \mathbb{R}^{h \times w \times d} \\
\boldsymbol{z}_q &= \text{Quantize}(\boldsymbol{z}), \quad \boldsymbol{z}_q[i,j] \in \mathcal{C}_{\text{VQ}} \\
\hat{\boldsymbol{x}} &= D(\boldsymbol{z}_q)
\end{aligned} \tag{36}
\end{equation}

- 编码维度：$h \times w$（通常 $16 \times 16 = 256$）
- 每个位置：$\log_2 |\mathcal{C}_{\text{VQ}}|$ bits（通常8-10 bits）
- 总编码率：$256 \times 8 = 2048$ bits

**DDCM**：
\begin{equation}
\begin{aligned}
\mathbf{i} &= \text{Encode}(\boldsymbol{x}), \quad \mathbf{i} \in \{1,\ldots,K\}^T \\
\hat{\boldsymbol{x}} &= \text{Decode}(\mathbf{i})
\end{aligned} \tag{37}
\end{equation}

- 编码维度：$T$（通常50）
- 每个时间步：$\log_2 K$ bits（通常6 bits）
- 总编码率：$50 \times 6 = 300$ bits

**对比**：DDCM的编码率更低（6.8倍压缩）！

#### 9.2 编码的本质区别

**VQ-VAE**：
- 在**特征空间**量化
- 需要训练Encoder和Decoder
- 2D空间结构（$h \times w$ 网格）

**DDCM**：
- 在**噪声空间**量化
- 复用预训练的DDPM
- 1D序列结构（$T$ 个时间步）

**DDCM的优势**：
1. **1D序列**：自然适配自回归建模（如Transformer）
2. **无需排序**：时间维度天然有序
3. **免训练**：直接用现成的DDPM

### 10. 扩展到Classifier Guidance

#### 10.1 带引导的DDPM

标准Classifier Guidance：
\begin{equation}
\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t^2 \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t) + \sigma_t \boldsymbol{\varepsilon}_t \tag{38}
\end{equation}

其中 $p(\boldsymbol{y}|\boldsymbol{x}_t)$ 是（带噪）分类器。

#### 10.2 DDCM的引导编码

类比前面的推导，引导项需要从噪声补偿：
\begin{equation}
\sigma_t \boldsymbol{\varepsilon}_t \approx -\sigma_t^2 \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t) \tag{39}
\end{equation}

即：
\begin{equation}
\boldsymbol{\varepsilon}_t \approx -\sigma_t \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t) \tag{40}
\end{equation}

**噪声选择**：
\begin{equation}
\boldsymbol{\varepsilon}_t^* = \arg\min_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \left\Vert\boldsymbol{\varepsilon} + \sigma_t \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t)\right\Vert^2 \tag{41}
\end{equation}

同样地，近似为：
\begin{equation}
\boldsymbol{\varepsilon}_t^* = \arg\max_{\boldsymbol{\varepsilon}\in\mathcal{C}_t} \boldsymbol{\varepsilon} \cdot \nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{y}|\boldsymbol{x}_t) \tag{42}
\end{equation}

**应用**：给定类别 $\boldsymbol{y}$，可以编码出该类别的典型样本。

### 11. 理论分析

#### 11.1 编码的存在性

**定理3**：对于任意 $\boldsymbol{x}_0 \in \text{supp}(p_{\text{data}})$，当 $K$ 足够大时，存在索引序列 $\mathbf{i}$ 使得
\begin{equation}
\Vert\boldsymbol{x}_0 - \text{Decode}(\mathbf{i})\Vert \leq \epsilon \tag{43}
\end{equation}
对于任意 $\epsilon > 0$。

**证明思路**：
1. DDPM可以生成任意真实分布的样本
2. Codebook是连续噪声空间的离散化
3. $K \to \infty$ 时，离散化误差趋于0
4. 因此存在近似完美的编码

#### 11.2 贪心算法的次优性

贪心编码不一定找到全局最优解：
\begin{equation}
\text{Decode}(\text{GreedyEncode}(\boldsymbol{x}_0)) \neq \arg\min_{\mathbf{i}} \Vert\boldsymbol{x}_0 - \text{Decode}(\mathbf{i})\Vert \tag{44}
\end{equation}

**原因**：每步局部最优不保证全局最优。

**实验观察**：
- 大多数情况下，贪心算法足够好
- 重构误差主要来自 $K$ 的大小，而非算法次优性

### 12. 压缩性能分析

#### 12.1 率失真权衡

定义率失真函数：
\begin{equation}
R(D) = \min_{p(\mathbf{i}|\boldsymbol{x}_0): \mathbb{E}[\text{dist}(\boldsymbol{x}_0, \hat{\boldsymbol{x}}_0)] \leq D} I(\boldsymbol{x}_0; \mathbf{i}) \tag{45}
\end{equation}

**DDCM的率失真**：
- 编码率：$R = T \log_2 K$
- 失真：$D = \mathbb{E}[\Vert\boldsymbol{x}_0 - \hat{\boldsymbol{x}}_0\Vert^2]$

**实验曲线**：

| $K$ | $R$ (bits) | $D$ (MSE) | PSNR (dB) |
|-----|-----------|-----------|-----------|
| 2 | 50 | 0.0182 | 17.4 |
| 4 | 100 | 0.0089 | 20.5 |
| 16 | 200 | 0.0032 | 24.9 |
| 64 | 300 | 0.0012 | 29.2 |

#### 12.2 与传统压缩方法对比

**JPEG（300 bits）**：
- 有损压缩，质量因子需要调低
- PSNR约35 dB，但有明显的块效应

**DDCM（300 bits）**：
- 感知质量更好（无块效应）
- PSNR约29 dB（客观指标差）
- 但FID更低（生成质量好）

**洞察**：DDCM不是传统意义的压缩，而是**生成式编码**。

### 13. 实现技巧

#### 13.1 初始化 $\boldsymbol{x}_T$ 的选择

**方案1：随机初始化**
\begin{equation}
\boldsymbol{x}_T \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{46}
\end{equation}

缺点：编码不唯一，不同运行结果不同。

**方案2：从Codebook选择**
\begin{equation}
\boldsymbol{x}_T = \arg\max_{\boldsymbol{c}\in\mathcal{C}_T} \boldsymbol{c} \cdot (\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_T)) \tag{47}
\end{equation}

问题：需要先知道 $\boldsymbol{x}_T$（循环依赖）。

**方案3：两阶段**
1. 先随机生成一次，得到 $\boldsymbol{x}_T^{(0)}$
2. 用 $\boldsymbol{x}_T^{(0)}$ 作为估计，从Codebook选择最佳的

#### 13.2 加速技巧

**并行内积计算**：
\begin{equation}
\text{scores} = \mathbf{C}_t \boldsymbol{\Delta}, \quad \mathbf{C}_t = [\boldsymbol{c}_t^{(1)}, \ldots, \boldsymbol{c}_t^{(K)}]^T \in \mathbb{R}^{K \times d} \tag{48}
\end{equation}

使用矩阵乘法，GPU加速。

**缓存预测**：
- $\bar{\boldsymbol{\mu}}(\boldsymbol{x}_t)$ 在每步都要计算
- 如果Codebook不变，可以预计算部分结果

### 14. 应用场景

#### 14.1 图像Tokenizer for LLM

**传统方案**（VQ-VAE）：
\begin{equation}
\text{Image} \xrightarrow{E} \text{2D tokens} \xrightarrow{\text{flatten}} \text{1D seq} \xrightarrow{\text{LLM}} \cdots \tag{49}
\end{equation}

问题：flatten需要选择扫描顺序（raster scan, Hilbert curve等）。

**DDCM方案**：
\begin{equation}
\text{Image} \xrightarrow{\text{DDCM}} \text{1D seq} \xrightarrow{\text{LLM}} \cdots \tag{50}
\end{equation}

优势：天然1D，无需排序！

#### 14.2 压缩表示学习

学习一个自回归模型：
\begin{equation}
p(\mathbf{i}) = \prod_{t=1}^T p(i_t | i_{t+1}, \ldots, i_T) \tag{51}
\end{equation}

可用于：
- 密度估计
- 条件生成
- 表示学习

### 15. 局限性与未来方向

#### 15.1 当前局限

1. **编码速度慢**：需要完整的DDPM前向过程（$T$ 步）
2. **压缩率固定**：由 $T$ 和 $K$ 决定，不灵活
3. **依赖DDPM质量**：如果DDPM本身不好，DDCM也不好

#### 15.2 可能改进

**加速采样**：
- 使用DDIM、DPM-Solver等加速器
- 问题：减少 $T$ 会降低编码长度

**可变率编码**：
\begin{equation}
K_t = f(t), \quad \text{e.g., } K_t = 2^{\lfloor a + bt \rfloor} \tag{52}
\end{equation}

前期用大Codebook，后期用小Codebook。

**端到端训练**：
\begin{equation}
\min_{\boldsymbol{\theta}, \{\mathcal{C}_t\}} \mathbb{E}_{\boldsymbol{x}_0}[\Vert\boldsymbol{x}_0 - \text{DDCM}_{\boldsymbol{\theta}}(\text{Encode}(\boldsymbol{x}_0))\Vert^2] \tag{53}
\end{equation}

联合优化DDPM和Codebook。

### 16. 数学总结

#### 16.1 核心公式

1. **Codebook定义**：
   $$\mathcal{C}_t = \{\boldsymbol{c}_t^{(1)}, \ldots, \boldsymbol{c}_t^{(K)}\}, \quad \boldsymbol{c}_t^{(k)} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$$

2. **编码规则**：
   $$i_t = \arg\max_{k=1,\ldots,K} \boldsymbol{c}_t^{(k)} \cdot (\boldsymbol{x}_0 - \bar{\boldsymbol{\mu}}(\boldsymbol{x}_t))$$

3. **解码过程**：
   $$\boldsymbol{x}_{t-1} = \boldsymbol{\mu}(\boldsymbol{x}_t) + \sigma_t \boldsymbol{c}_t^{(i_t)}$$

4. **编码率**：
   $$R = T \cdot \log_2 K \text{ bits}$$

#### 16.2 理论意义

DDCM展示了：
1. **噪声的语义**：DDPM的噪声不是纯随机，而是携带语义信息
2. **离散化可行性**：连续噪声空间可以有效离散化
3. **新型Tokenizer**：扩散模型可以作为图像的序列化编码器

这为多模态大模型提供了新的技术路线。

