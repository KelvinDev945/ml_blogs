---
title: Transformer升级之路：7、长度外推性与局部注意力
slug: transformer升级之路7长度外推性与局部注意力
date: 2023-01-12
tags: 详细推导, 语言模型, attention, 位置编码, 外推, 生成模型
status: completed
---
# Transformer升级之路：7、长度外推性与局部注意力

**原文链接**: [https://spaces.ac.cn/archives/9431](https://spaces.ac.cn/archives/9431)

**发布日期**: 

---

对于Transformer模型来说，其长度的外推性是我们一直在追求的良好性质，它是指我们在短序列上训练的模型，能否不用微调地用到长序列上并依然保持不错的效果。之所以追求长度外推性，一方面是理论的完备性，觉得这是一个理想模型应当具备的性质，另一方面也是训练的实用性，允许我们以较低成本（在较短序列上）训练出一个长序列可用的模型。

下面我们来分析一下加强Transformer长度外推性的关键思路，并由此给出一个“超强基线”方案，然后我们带着这个“超强基线”来分析一些相关的研究工作。

## 思维误区 #

第一篇明确研究Transformer长度外推性的工作应该是[ALIBI](https://papers.cool/arxiv/2108.12409)，出自2021年中期，距今也不算太久。为什么这么晚（相比Transformer首次发表的2017年）才有人专门做这个课题呢？估计是因为我们长期以来，都想当然地认为Transformer的长度外推性是位置编码的问题，找到更好的位置编码就行了。

事实上，通过对比现有的一些位置编码的外推效果，确实能找到支撑该观点的一些论据。比如后面分享的多篇实验效果显示，相对位置编码的长度外推性，平均好于绝对位置编码的；像[RoPE](/archives/8265)这样的函数式相对位置编码，又会比训练式相对位置编码的外推效果好些。所以看上去，似乎只要我们不断优化位置编码形式，最终就能给Transformer提供更好的长度外推性，从而解决这个问题。然而，情况没有那么乐观，像RoPE算是外推能力较好的位置编码，也只能外推10%到20%左右的长度而保持效果不变差，再长效果就会骤降。这个比例与预期差太远了，设想中好歹能外推个几倍长度才算是有价值的外推，所以不难想象，单靠改进位置编码改进Transformer的长度外推性，就不知道要等多久才能实现更长的效果了。

在直觉上，相信很多读者觉得像[Sinusoidal](/archives/8231)或[RoPE](/archives/8265)之类的函数式位置编码，它们没有训练参数，长度外推性应该很好才对，但事实上并非如此，这类位置编码并没有在长度外推方面表现出什么优势。为什么会这样呢？其实是大家在假设函数式位置编码的外推性时，忘了它的基本前提——“光滑性”。

其实，外推性就是局部推断整体，对此我们应该并不陌生，泰勒级数近似就是经典的例子，它只需要知道函数某点处若干阶导数的值，就可以对一个邻域内的值做有效估计，它依赖的就是给定函数的高阶光滑性（高阶导数存在且有界）。但是[Sinusoidal](/archives/8231)或[RoPE](/archives/8265)是这种函数吗？并不是。它们是一系列正余弦函数的组合，其相位函数是$k/10000^{2i/d}$，当$2i/d\approx 0$时，函数近似就是$\sin k, \cos k$，这算是关于位置编码$k$的高频振荡函数了，而不是直线或者渐近趋于直线之类的函数，所以基于它的模型往往外推行为难以预估。能否设计不振荡的位置编码？很难，位置编码函数如果不振荡，那么往往缺乏足够的容量去编码足够多的位置信息，也就是某种意义上来说，位置编码函数的复杂性本身也是编码位置的要求。

## 超强基线 #

事实上，更准确的定位应该是：

> 长度外推性是一个训练和预测的长度不一致的问题。

具体来说，不一致的地方有两点：

> 1、预测的时候用到了没训练过的位置编码（不管绝对还是相对）；
> 
> 2、预测的时候注意力机制所处理的token数量远超训练时的数量。

第1点可能大家都容易理解，没训练过的就没法保证能处理好，这是DL中很现实的现象，哪怕是[Sinusoidal](/archives/8231)或[RoPE](/archives/8265)这种函数式位置编码也是如此。关于第2点，可能读者会有些迷惑，Attention理论上不就是可以处理任意长度的序列吗？训练和预测长度不一致影响什么呢？答案是熵，我们在[《从熵不变性看Attention的Scale操作》](/archives/8823)也已经分析过这个问题，越多的token去平均注意力，意味着最后的分布相对来说越“均匀”（熵更大），即注意力越分散；而训练长度短，则意味着注意力的熵更低，注意力越集中，这也是一种训练和预测的差异性，也会影响效果。

事实上，对于相对位置编码的Transformer模型，通过一个非常简单的Attention Mask，就可以一次性解决以上两个问题，并且取得接近SOTA的效果：  


[![超强基线模型（双向注意力版）](/usr/uploads/2023/01/1015423166.svg)](/usr/uploads/2023/01/1015423166.svg "点击查看原图")

超强基线模型（双向注意力版）

[![超强基线模型（单向注意力版）](/usr/uploads/2023/01/2044543966.svg)](/usr/uploads/2023/01/2044543966.svg "点击查看原图")

超强基线模型（单向注意力版）

不难理解，这就是将预测时的Attention变为一个局部Attention，每个token只能看到训练长度个token。这样一来，每个token可以看到的token数跟训练时一致，这就解决了第2个问题，同时由于是相对位置编码，位置的计数以当前token为原点，因此这样的局部Attention也不会比训练时使用更多的未知编码，这就解决了第1个问题。所以，就这个简单的Attention Mask一次性解决了长度外推的2个难点，还不用重新训练模型，更令人惊叹的是，各种实验结果显示，如果以它为baseline，那么各种同类工作的相对提升就弱得可怜了，也就是它本身已经很接近SOTA了，可谓是又快又好的“超强基线”。

## 论文学习 #

自ALIBI起，确实已经有不少工作投入到了Transformer长度外推性的研究中。在这一节中，笔者学习并整理了一下其中的一些代表工作，从中我们也可以发现，它们基本都跟基线模型有着诸多相通之处，甚至可以说它们某种程度上都是基线模型的变体，由此我们将进一步体会到长度外推性与注意力的局部性之间的深刻联系。

### ALIBI #

作为“开山之作”，ALIBI是绕不过去的，它出自论文[《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》](https://papers.cool/arxiv/2108.12409)。事后来看，ALIBI所做的改动非常简单，只是在Softmax之前，将Attention的计算从$\boldsymbol{q}_m^{\top}\boldsymbol{k}_n$改为  
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{k}_n - \lambda|m - n|\label{eq:alibi}\end{equation}  
其中$\lambda > 0$是超参数，每个head设置不同的值。从这个定义就可以看出ALIBI跟基线模型的相似之处了，两者都是在Softmax之前减去一个非负矩阵，只不过被减去的非负矩阵有所不同，ALIBI可以看成是基线模型的“光滑版”：  


[![基线模型所减去的矩阵](/usr/uploads/2023/01/3899853922.svg)](/usr/uploads/2023/01/3899853922.svg "点击查看原图")

基线模型所减去的矩阵

[![ALIBI所减去的矩阵](/usr/uploads/2023/01/1507343310.svg)](/usr/uploads/2023/01/1507343310.svg "点击查看原图")

ALIBI所减去的矩阵

ALIBI是一个很朴素（当然也很有效）的光滑局部注意力技巧，但如果将它理解为“位置编码”，那么应该是不大妥当的，如果就按照式$\eqref{eq:alibi}$来推广到双向注意力，那么由于$|m - n|=|n - m|$，按理说模型就无法区分“左”和“右”，只能识别相对距离的远近，显然是无法准确识别位置信息的。至于它在单向语言模型上效果好，是因为单向语言模型即便不加任何位置编码也能取得非平凡的效果（明显高于随机），ALIBI所施加的局部注意力，起到的是增强局域性的作用，贴合语言模型任务本身的特性。

### KERPLE #

KERPLE出自论文[《KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation》](https://papers.cool/arxiv/2205.09921)，它实质上就是ALIBI的简单推广，它引入了两个训练参数$r_1,r_2$来一般化式$\eqref{eq:alibi}$：  
\begin{equation}\left\\{\begin{aligned}&\boldsymbol{q}_m^{\top}\boldsymbol{k}_n - r_1|m - n|^{r_2} ,\qquad\qquad r_1 >0, 0 < r_2 \leq 2\\\  
&\boldsymbol{q}_m^{\top}\boldsymbol{k}_n - r_1\log(1+r_2|m - n|),\qquad\qquad r_1, r_2 > 0  
\end{aligned}\right.\label{eq:kerple}\end{equation}  
又是一般化，又有可训练参数，KERPLE能取得比ALIBI更好的效果并不让人意外。不过这里要严重批评一下KERPLE论文的故弄玄虚，按照排版，原论文第三节是论文的理论支撑，但很明显它只是为了强行提高文章的数学深度而引入的无关篇幅，实质上对理解KERPLE毫无帮助，甚至会降低读者的阅读兴致（说白了，为审稿人服务，不是为读者服务）。

### Sandwich #

Sandwich也是KERPLE的作者们搞的，出自[《Receptive Field Alignment Enables Transformer Length Extrapolation》](https://papers.cool/arxiv/2212.10356)，上个月才放到Arxiv上的，它将式$\eqref{eq:alibi}$替换为  
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{k}_n + \lambda\boldsymbol{p}_m^{\top}\boldsymbol{p}_n\label{eq:sandwich}\end{equation}  
其中$\boldsymbol{p}_m,\boldsymbol{p}_n$是[Sinusoidal](/archives/8231)位置编码，$\lambda > 0$是超参数。从[《Transformer升级之路：1、Sinusoidal位置编码追根溯源》](/archives/8231)我们知道，$\boldsymbol{p}_m^{\top}\boldsymbol{p}_n$是$m-n$的标量函数，并且 _平均而言_ 是$|m-n|$的单调递增函数，所以它的作用也跟$-\lambda|m-n|$相似。之所以强调“平均而言”，是因为$\boldsymbol{p}_m^{\top}\boldsymbol{p}_n$整体并非严格的单调，而是振荡下降，如图所示：  


[![dot\(p_m, p_n\) 的函数图像（减去了d/2）](/usr/uploads/2023/01/3615292479.png)](/usr/uploads/2023/01/3615292479.png "点击查看原图")

dot(p_m, p_n) 的函数图像（减去了d/2）

如果有必要，我们也可以将Sandwich转换成[RoPE](/archives/8265)那样“绝对位置编码实现相对位置编码”的形式，这只需要留意到  
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{k}_n + \lambda\boldsymbol{p}_m^{\top}\boldsymbol{p}_n = \left[\boldsymbol{q}_m, \sqrt{\lambda}\boldsymbol{p}_m\right]^{\top}\left[\boldsymbol{k}_n, \sqrt{\lambda}\boldsymbol{p}_n\right]\end{equation}  
也就是说Sandwich通过拼接的方式补充绝对位置信息，其Attention结果则相当于相对位置编码。不过目前看来这个转换也就只有理论价值，因为拼接增加了向量维度，反而会进一步增加Attention的计算量。

### XPOS #

XPOS出自论文[《A Length-Extrapolatable Transformer》](https://papers.cool/arxiv/2212.10554)，跟Sandwich同一天出现在Arxiv上，它是[RoPE](/archives/8265)的一个一脉相承的推广。我们知道，RoPE的基本解是：  
\begin{equation}\boldsymbol{q}_m\to \boldsymbol{\mathcal{R}}_m\boldsymbol{q}_m,\quad \boldsymbol{k}_n\to \boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n\end{equation}  
其中$\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\\ \sin n\theta & \cos n\theta\end{pmatrix}$。笔者当时推导RoPE的时候，假设了“$Q$、$K$做同样的变换”，事实上单从“绝对位置实现相对位置”角度来看，并没有必要限制两者的变换格式一致，比如XPOS考虑的是  
\begin{equation}\boldsymbol{q}_m\to \boldsymbol{\mathcal{R}}_m\boldsymbol{q}_m \xi^m,\quad \boldsymbol{k}_n\to \boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \xi^{-n}\end{equation}  
其中$\xi$是一个标量超参，这样以来  
\begin{equation}(\boldsymbol{\mathcal{R}}_m\boldsymbol{q}_m \xi^m)^{\top}(\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \xi^{-n}) = \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}_n \xi^{m-n}\end{equation}  
总的结果依然是只依赖于相对位置$m-n$的。然而，现在问题是指数部分是$m-n$而不是$|m-n|$，只要$\xi\neq 1$，那么总有一边是发散的。XPOS的机智之处是它选择了跟很多相关工作一样选定了场景——只关注单向语言模型——这样一来我们只会用到$m\geq n$部分的注意力！此时只需要选择$\xi\in(0,1)$，就可以实现随着相对距离衰减的效果。

事实上，额外引入的指数衰减项$\xi^m$并非XPOS的首创，笔者所阅读的文献中，同样的项最早出现在[PermuteFormer](https://papers.cool/arxiv/2109.02377)中，只不过PermuteFormer主要关心的是线性Attention场景。细节上，XPOS给每个分块都分配了不同的$\xi$，但在跟作者私下交流的时候，作者补做了共享同一个$\xi$的实验，发现设置不同$\xi$带来的提升几乎可以忽略。另外，我们要适当控制$\xi$的值，以防止$\xi^{-n}$在$n$较大的时候溢出。

值得指出的是，这里的随着相对距离衰减，是直接乘在Softmax之前的Attention Score的，结果是相对距离较远的Score变得很接近于0，而不是像前面几种设计那样趋向负无穷。$e^0$并没有趋于零的效果，所以说，这样的设计并非是局部Attention的变体，因此它的效果没有达到SOTA。为弥补这部分差距，XPOS设计了一个特殊的局部Attention（称为Blockwise Causal Attention，简写BCA），加上去之后就能够补足这个差距了。在交流时作者表示使用BCA是因为在实现上有优势，实际上基线模型的局部Attention效果更好，所以，要外推性还是要看局部注意力啊。

原论文的实验还是很丰富很值得参考的，建议大家细读～

## 文章小结 #

本文总结了增强Transformer的长度外推能力的相关工作，其中包含了一个简单但强大的基线方案，以及若干篇聚焦于长度外推性的相关工作，从中我们可以发现，这些工作本质上都是基线方案——局部注意力的变体，局部注意力是长度外推的关键环节之一。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9431>_

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

苏剑林. (Jan. 12, 2023). 《Transformer升级之路：7、长度外推性与局部注意力 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9431>

@online{kexuefm-9431,  
title={Transformer升级之路：7、长度外推性与局部注意力},  
author={苏剑林},  
year={2023},  
month={Jan},  
url={\url{https://spaces.ac.cn/archives/9431}},  
} 


---

## 公式推导与注释

### 1. 局部注意力的窗口函数表示

#### 1.1 全局注意力的数学定义

标准的全局自注意力机制定义为：

$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}$$

对于序列长度为 $L$ 的输入，注意力矩阵 $\boldsymbol{A} \in \mathbb{R}^{L \times L}$ 的第 $(m, n)$ 个元素为：

$$A_{m,n} = \frac{\exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_n}{\sqrt{d_k}}\right)}{\sum_{j=1}^{L} \exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_j}{\sqrt{d_k}}\right)}$$

这意味着每个位置 $m$ 可以关注到所有 $L$ 个位置。

#### 1.2 窗口函数的定义

**定义1（窗口掩码函数）**：对于窗口大小 $w$，定义窗口掩码函数 $\mathcal{M}_w: \mathbb{Z} \times \mathbb{Z} \to \{0, 1\}$：

$$\mathcal{M}_w(m, n) = \begin{cases}
1, & \text{if } |m - n| \leq w \\
0, & \text{otherwise}
\end{cases}$$

对于单向注意力（因果掩码），窗口函数修改为：

$$\mathcal{M}_w^{\text{causal}}(m, n) = \begin{cases}
1, & \text{if } 0 \leq m - n \leq w \\
0, & \text{otherwise}
\end{cases}$$

这确保了位置 $m$ 只能看到其前面的 $w$ 个位置（包括自己）。

#### 1.3 局部注意力的显式表示

应用窗口掩码后的局部注意力为：

$$A_{m,n}^{\text{local}} = \begin{cases}
\frac{\exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_n}{\sqrt{d_k}}\right)}{\sum_{j: \mathcal{M}_w(m,j)=1} \exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_j}{\sqrt{d_k}}\right)}, & \text{if } \mathcal{M}_w(m,n) = 1 \\
0, & \text{otherwise}
\end{cases}$$

也可以写成掩码加法的形式：

$$A_{m,n}^{\text{local}} = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}} + \boldsymbol{M}_w\right)$$

其中掩码矩阵 $\boldsymbol{M}_w \in \mathbb{R}^{L \times L}$：

$$(\boldsymbol{M}_w)_{m,n} = \begin{cases}
0, & \text{if } \mathcal{M}_w(m,n) = 1 \\
-\infty, & \text{otherwise}
\end{cases}$$

#### 1.4 滑动窗口的数学性质

**性质1（局部性）**：对于窗口大小 $w$，每个token最多关注 $2w+1$ 个位置（双向）或 $w+1$ 个位置（单向）。

**性质2（稀疏性）**：注意力矩阵的非零元素比例为：
$$\rho = \frac{\text{非零元素数}}{\text{总元素数}} = \frac{O(Lw)}{L^2} = O\left(\frac{w}{L}\right)$$

当 $w \ll L$ 时，$\rho \to 0$，注意力矩阵变得高度稀疏。

**性质3（带状结构）**：局部注意力矩阵具有带状（band）结构，非零元素集中在主对角线附近宽度为 $2w+1$ 的带内。

### 2. 稀疏注意力模式的数学定义

#### 2.1 稀疏注意力的一般框架

**定义2（稀疏注意力模式）**：稀疏注意力模式由一个索引集合函数 $\mathcal{S}: \{1, \ldots, L\} \to 2^{\{1, \ldots, L\}}$ 定义，其中 $\mathcal{S}(m)$ 表示位置 $m$ 可以关注的位置集合。

稀疏注意力为：
$$A_{m,n}^{\text{sparse}} = \begin{cases}
\frac{\exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_n}{\sqrt{d_k}}\right)}{\sum_{j \in \mathcal{S}(m)} \exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_j}{\sqrt{d_k}}\right)}, & \text{if } n \in \mathcal{S}(m) \\
0, & \text{otherwise}
\end{cases}$$

#### 2.2 常见的稀疏模式

**局部窗口模式**：
$$\mathcal{S}_{\text{local}}(m) = \{n : |m - n| \leq w\}$$

**跨步模式（Strided Pattern）**：
$$\mathcal{S}_{\text{stride}}(m) = \{n : n \equiv m \pmod{s}\}$$

其中 $s$ 是步长。

**固定模式（Fixed Pattern）**：
$$\mathcal{S}_{\text{fixed}}(m) = \{n : n \in \{0, s, 2s, \ldots\}\} \cup \{n : |m - n| \leq w\}$$

结合了全局稀疏采样和局部窗口。

**随机模式（Random Pattern）**：
$$\mathcal{S}_{\text{random}}(m) = \{n : n \text{ 从 } \{1, \ldots, L\} \text{ 中随机采样 } k \text{ 个}\}$$

#### 2.3 ALIBI的数学表示

ALIBI通过在注意力得分上添加线性偏置实现软局部化：

$$\text{score}(m, n) = \frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_n}{\sqrt{d_k}} - \lambda|m - n|$$

其中 $\lambda > 0$ 是衰减系数。

经过softmax后：
$$A_{m,n}^{\text{ALIBI}} = \frac{\exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_n}{\sqrt{d_k}} - \lambda|m - n|\right)}{\sum_{j=1}^{L} \exp\left(\frac{\boldsymbol{q}_m^{\top}\boldsymbol{k}_j}{\sqrt{d_k}} - \lambda|j - n|\right)}$$

**性质4（指数衰减）**：ALIBI的注意力权重随距离指数衰减：
$$A_{m,n}^{\text{ALIBI}} \propto \exp(-\lambda|m - n|)$$

当 $|m - n|$ 很大时，$A_{m,n}^{\text{ALIBI}} \approx 0$，实现了软局部化。

#### 2.4 稀疏度的量化

**定义3（注意力稀疏度）**：对于稀疏模式 $\mathcal{S}$，其稀疏度定义为：

$$\text{Sparsity}(\mathcal{S}) = 1 - \frac{1}{L^2}\sum_{m=1}^{L} |\mathcal{S}(m)|$$

其中 $|\mathcal{S}(m)|$ 是位置 $m$ 关注的位置数量。

对于局部窗口（窗口大小 $w$）：
$$\text{Sparsity}(\mathcal{S}_{\text{local}}) = 1 - \frac{2w+1}{L} \approx 1 - \frac{2w}{L}$$

当 $L \to \infty$ 且 $w$ 固定时，稀疏度趋向于1。

### 3. 长度外推的理论界限

#### 3.1 外推问题的形式化

**定义4（长度外推）**：设模型在长度 $L_{\text{train}}$ 的序列上训练，要在长度 $L_{\text{test}} > L_{\text{train}}$ 的序列上测试，定义外推比率：

$$r = \frac{L_{\text{test}}}{L_{\text{train}}}$$

**问题**：当 $r > 1$ 时，模型性能如何变化？

#### 3.2 位置编码的外推界限

**定理1（位置编码外推界）**：对于周期性位置编码（如Sinusoidal和RoPE），设最大周期为 $T_{\max}$，则外推到长度 $L$ 时，至少有 $\lfloor L/T_{\max} \rfloor$ 个周期重复。

对于RoPE，最大周期为：
$$T_{\max} = 2\pi \cdot \theta_{\text{base}}^{(d-2)/d} = 2\pi \cdot 10000^{(d-2)/d}$$

当 $d = 128$ 时，$T_{\max} \approx 2\pi \cdot 10000^{126/128} \approx 62700$。

**推论**：当 $L_{\text{test}} > T_{\max}$ 时，位置编码开始重复，模型难以区分相差 $T_{\max}$ 倍数的位置。

#### 3.3 注意力熵的变化

**定义5（注意力熵）**：对于位置 $m$ 的注意力分布 $\{A_{m,n}\}_{n=1}^{L}$，定义其熵为：

$$H_m = -\sum_{n=1}^{L} A_{m,n} \log A_{m,n}$$

**定理2（熵与序列长度的关系）**：在均匀注意力假设下（$A_{m,n} \approx 1/L$），注意力熵随序列长度对数增长：

$$H_m \approx \log L$$

证明：当 $A_{m,n} = 1/L$ 时，
$$H_m = -\sum_{n=1}^{L} \frac{1}{L} \log \frac{1}{L} = -L \cdot \frac{1}{L} \log \frac{1}{L} = \log L$$

**推论**：当 $L_{\text{test}} > L_{\text{train}}$ 时，注意力熵增加：
$$\Delta H = H_{\text{test}} - H_{\text{train}} \approx \log \frac{L_{\text{test}}}{L_{\text{train}}} = \log r$$

这导致注意力更加分散，与训练时的集中注意力不一致。

#### 3.4 局部注意力的熵不变性

对于窗口大小为 $w$ 的局部注意力，无论序列长度 $L$ 如何增加，注意力熵保持有界：

$$H_m^{\text{local}} \leq \log(2w+1)$$

**定理3（局部注意力的熵稳定性）**：局部注意力的熵不随序列长度变化，从而保持训练与测试的一致性。

证明：由于每个位置最多关注 $2w+1$ 个位置，在均匀分布假设下：
$$H_m^{\text{local}} = -\sum_{n \in \mathcal{S}_{\text{local}}(m)} \frac{1}{|\mathcal{S}_{\text{local}}(m)|} \log \frac{1}{|\mathcal{S}_{\text{local}}(m)|} = \log |\mathcal{S}_{\text{local}}(m)| \leq \log(2w+1)$$

这与序列长度 $L$ 无关。

### 4. 滑动窗口与全局注意力的权衡

#### 4.1 感受野分析

**定义6（有效感受野）**：在 $\ell$ 层Transformer中，位置 $m$ 的有效感受野是所有能够影响其表示的位置集合。

对于全局注意力，单层的感受野为整个序列：
$$\text{RF}_1^{\text{global}}(m) = \{1, 2, \ldots, L\}$$

对于局部注意力（窗口大小 $w$），单层的感受野为：
$$\text{RF}_1^{\text{local}}(m) = \{n : |m - n| \leq w\}$$

**定理4（多层感受野的增长）**：在 $\ell$ 层局部注意力中，感受野线性增长：

$$|\text{RF}_\ell^{\text{local}}(m)| = \min(2\ell w + 1, L)$$

证明（归纳法）：
- 基础情况（$\ell = 1$）：$|\text{RF}_1^{\text{local}}(m)| = 2w + 1$
- 归纳步骤：假设 $|\text{RF}_{\ell-1}^{\text{local}}(m)| = 2(\ell-1)w + 1$，则第 $\ell$ 层中，每个位置可以关注到前一层感受野的 $\pm w$ 范围，因此：
  $$|\text{RF}_\ell^{\text{local}}(m)| = 2(\ell-1)w + 1 + 2w = 2\ell w + 1$$

**推论**：要覆盖整个序列 $L$，需要的层数为：
$$\ell_{\text{min}} = \left\lceil \frac{L - 1}{2w} \right\rceil$$

#### 4.2 信息传播速度

**定义7（信息传播速度）**：从位置 $i$ 传播信息到位置 $j$ 所需的最小层数。

对于全局注意力，信息传播是即时的（1层）：
$$\text{Hops}_{\text{global}}(i, j) = 1, \quad \forall i, j$$

对于局部注意力：
$$\text{Hops}_{\text{local}}(i, j) = \left\lceil \frac{|i - j|}{w} \right\rceil$$

**定理5（信息瓶颈）**：局部注意力中，远距离信息传播需要多层，可能导致信息损失。

#### 4.3 表达能力的权衡

**定理6（全局vs局部的Pareto前沿）**：记 $C$ 为计算成本，$E$ 为表达能力，则：

- 全局注意力：$C_{\text{global}} = O(L^2 d)$，$E_{\text{global}} = $ 最大
- 局部注意力：$C_{\text{local}} = O(Lwd)$，$E_{\text{local}} = $ 受限于感受野

不存在同时最小化 $C$ 和最大化 $E$ 的方案，形成Pareto前沿。

#### 4.4 最优窗口大小的选择

**优化问题**：给定计算预算 $C_{\max}$ 和目标序列长度 $L$，选择最优窗口大小 $w^*$ 和层数 $\ell^*$：

$$\begin{aligned}
\max_{w, \ell} \quad & \text{Performance}(w, \ell) \\
\text{s.t.} \quad & \ell \cdot L \cdot w \cdot d \leq C_{\max} \\
& 2\ell w \geq L \quad \text{（覆盖整个序列）}
\end{aligned}$$

**启发式解**：在实践中，常选择 $w = O(\sqrt{L})$，使得 $\ell = O(\sqrt{L})$ 也能覆盖整个序列，计算复杂度为 $O(L^{1.5}d)$。

### 5. 计算复杂度分析

#### 5.1 时间复杂度

**全局注意力**：
- 计算 $\boldsymbol{Q}\boldsymbol{K}^{\top}$：$O(L^2 d)$
- Softmax：$O(L^2)$
- 乘以 $\boldsymbol{V}$：$O(L^2 d)$
- **总计**：$O(L^2 d)$

**局部注意力（窗口大小 $w$）**：
- 对每个位置，计算 $w$ 个点积：$O(Lwd)$
- Softmax（每个位置 $w$ 个元素）：$O(Lw)$
- 加权求和：$O(Lwd)$
- **总计**：$O(Lwd)$

**复杂度比**：
$$\frac{C_{\text{local}}}{C_{\text{global}}} = \frac{Lwd}{L^2d} = \frac{w}{L}$$

当 $w \ll L$ 时，局部注意力显著更快。

#### 5.2 空间复杂度

**全局注意力**：
- 注意力矩阵：$O(L^2)$
- QKV矩阵：$O(Ld)$
- **总计**：$O(L^2 + Ld) = O(L^2)$（当 $L > d$ 时）

**局部注意力**：
- 稀疏注意力矩阵（只存储非零元素）：$O(Lw)$
- QKV矩阵：$O(Ld)$
- **总计**：$O(Lw + Ld) = O(L(w+d))$

**空间节省比**：
$$\frac{S_{\text{local}}}{S_{\text{global}}} = \frac{Lw}{L^2} = \frac{w}{L}$$

#### 5.3 不同稀疏模式的复杂度

| 模式 | 时间复杂度 | 空间复杂度 | 感受野（$\ell$ 层） |
|------|-----------|-----------|------------------|
| 全局注意力 | $O(L^2d)$ | $O(L^2)$ | $L$ |
| 局部窗口 | $O(Lwd)$ | $O(Lw)$ | $2\ell w$ |
| 跨步 | $O(L^2d/s)$ | $O(L^2/s)$ | $L$（稀疏） |
| 固定+局部 | $O(L(w+k)d)$ | $O(L(w+k))$ | $L$ |
| Longformer | $O(L(w+g)d)$ | $O(L(w+g))$ | $L$ |

其中 $k$ 是固定采样数，$g$ 是全局token数。

#### 5.4 实际加速比分析

考虑现代GPU的并行特性，实际加速比可能低于理论值：

**定理7（Amdahl定律的应用）**：设并行部分比例为 $p$，串行部分比例为 $1-p$，则实际加速比为：

$$\text{Speedup} = \frac{1}{(1-p) + \frac{p}{N}}$$

其中 $N$ 是理论加速倍数。

对于局部注意力，$N = L/w$，但由于存在串行部分（如Softmax归一化），实际加速比小于 $L/w$。

### 6. 外推误差的定量估计

#### 6.1 误差来源分解

外推误差可以分解为两部分：

$$\mathcal{E}_{\text{total}} = \mathcal{E}_{\text{position}} + \mathcal{E}_{\text{attention}}$$

其中：
- $\mathcal{E}_{\text{position}}$：未见过位置编码的误差
- $\mathcal{E}_{\text{attention}}$：注意力分布变化的误差

#### 6.2 位置编码误差

**定义8（位置编码外推误差）**：对于位置 $m > L_{\text{train}}$，位置编码的误差为：

$$\mathcal{E}_{\text{position}}(m) = \|\text{PE}(m) - \text{PE}_{\text{interp}}(m)\|$$

其中 $\text{PE}_{\text{interp}}(m)$ 是基于训练集位置编码的插值或外推。

对于函数式位置编码（如RoPE），假设模型学到的是位置编码之间的关系而非绝对值，则：

**定理8（RoPE的外推误差界）**：对于RoPE，外推误差主要来自高频分量的振荡：

$$\mathcal{E}_{\text{position}}^{\text{RoPE}}(m) \leq C \sum_{i=0}^{d/2-1} |\sin(m\theta_i) - \mathbb{E}[\sin(m'\theta_i)]|$$

其中期望是对训练集中的 $m' \in [1, L_{\text{train}}]$ 求的。

对于 $m > L_{\text{train}}$，高频项（$\theta_i$ 大）的振荡更剧烈，误差更大。

#### 6.3 注意力分布误差

**定义9（注意力分布的KL散度）**：测量训练与测试时注意力分布的差异：

$$\mathcal{E}_{\text{attention}}(m) = \text{KL}\left(A_m^{\text{train}} \| A_m^{\text{test}}\right)$$

其中 $A_m^{\text{train}}$ 和 $A_m^{\text{test}}$ 分别是训练和测试时位置 $m$ 的注意力分布。

**定理9（全局注意力的KL散度界）**：在均匀注意力假设下：

$$\text{KL}\left(\text{Uniform}(L_{\text{train}}) \| \text{Uniform}(L_{\text{test}})\right) = \log \frac{L_{\text{test}}}{L_{\text{train}}} = \log r$$

这与前面的熵增加一致。

**定理10（局部注意力的KL散度界）**：对于窗口大小 $w$ 的局部注意力，当 $w < \min(L_{\text{train}}, L_{\text{test}})$ 时：

$$\text{KL}\left(A_m^{\text{train}} \| A_m^{\text{test}}\right) \approx 0$$

因为两者的注意力范围相同（都是 $2w+1$ 个位置）。

#### 6.4 误差的上界估计

**定理11（总外推误差的上界）**：对于局部注意力，总外推误差有界：

$$\mathcal{E}_{\text{total}} \leq \mathcal{E}_{\text{position}} + O\left(\frac{w}{L_{\text{train}}}\right)$$

证明：注意力分布误差主要来自边界效应（序列开头和结尾的窗口不完整），其影响为 $O(w/L)$。

对于全局注意力：
$$\mathcal{E}_{\text{total}} \leq \mathcal{E}_{\text{position}} + C \log r$$

其中 $r = L_{\text{test}}/L_{\text{train}}$ 是外推比率。

#### 6.5 实验验证的理论预测

基于上述分析，我们可以预测：

**预测1**：局部注意力的外推性能随外推比率 $r$ 缓慢下降：
$$\text{Performance}_{\text{local}}(r) \approx \text{Performance}_{\text{local}}(1) - O(\epsilon_{\text{pos}})$$

其中 $\epsilon_{\text{pos}}$ 是位置编码误差，与 $r$ 的关系较弱。

**预测2**：全局注意力的外推性能随 $r$ 快速下降：
$$\text{Performance}_{\text{global}}(r) \approx \text{Performance}_{\text{global}}(1) - C \log r$$

**预测3**：最优窗口大小应接近训练长度：
$$w^* \approx \alpha L_{\text{train}}, \quad \alpha \in [0.5, 1]$$

这样既保持了训练时的注意力模式，又允许一定的外推能力。

### 7. log(n)缩放的理论分析

#### 7.1 熵不变性原理

**定理12（熵不变缩放）**：将注意力得分乘以 $\log_m n$（$m$ 是训练长度，$n$ 是测试长度）可以近似保持注意力熵不变。

修改后的注意力为：
$$\text{Attention}_{\text{log}}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\log_m n}{\sqrt{d}}\boldsymbol{Q}\boldsymbol{K}^{\top}\right)\boldsymbol{V}$$

**证明思路**：在均匀注意力近似下，原始注意力熵为 $H = \log n$。缩放后，softmax的温度变为 $\sqrt{d}/\log_m n$，等效注意力熵为：
$$H' \approx \log n - \log(\log_m n) = \log n - \log \frac{\log n}{\log m}$$

当 $n \approx m$ 时，$H' \approx \log m = H_{\text{train}}$。

#### 7.2 温度缩放的效果

温度参数 $\tau$ 控制softmax的"锐度"：
$$\text{softmax}_{\tau}(\boldsymbol{z}) = \frac{\exp(\boldsymbol{z}/\tau)}{\sum_j \exp(z_j/\tau)}$$

**性质5（温度与熵的关系）**：
- $\tau \to 0$：分布趋向one-hot（熵 $\to 0$）
- $\tau \to \infty$：分布趋向均匀（熵 $\to \log L$）

通过设置 $\tau = \sqrt{d}/\log_m n$，可以补偿序列长度变化对熵的影响。

#### 7.3 与局部注意力的结合

结合局部注意力和log缩放：
$$A_{m,n}^{\text{combined}} = \text{softmax}\left(\frac{\log_m n}{\sqrt{d}}(\boldsymbol{Q}\boldsymbol{K}^{\top} + \boldsymbol{M}_w)\right)$$

**定理13（组合优势）**：局部注意力 + log缩放可以同时解决：
1. 位置编码外推问题（通过局部窗口）
2. 注意力熵变化问题（通过log缩放）

从而达到最佳外推效果。

### 8. ALIBI与KERPLE的数学分析

#### 8.1 ALIBI的衰减特性

ALIBI的注意力权重为：
$$A_{m,n}^{\text{ALIBI}} = \frac{\exp\left(s_{m,n} - \lambda|m-n|\right)}{\sum_j \exp\left(s_{m,j} - \lambda|m-j|\right)}$$

其中 $s_{m,n} = \boldsymbol{q}_m^{\top}\boldsymbol{k}_n/\sqrt{d}$。

**渐近行为**：当 $|m-n| \to \infty$：
$$A_{m,n}^{\text{ALIBI}} \sim \exp(-\lambda|m-n|) \to 0$$

等效窗口大小可定义为：
$$w_{\text{eff}} = \frac{1}{\lambda} \log \frac{1}{\epsilon}$$

其中 $\epsilon$ 是注意力权重的阈值（如 $\epsilon = 0.01$）。

#### 8.2 KERPLE的幂律和对数形式

KERPLE提供两种形式：

**幂律形式**：
$$\text{score}(m,n) = s_{m,n} - r_1|m-n|^{r_2}$$

其中 $0 < r_2 \leq 2$。

- $r_2 = 1$：线性衰减（ALIBI）
- $r_2 = 2$：二次衰减（更快局部化）
- $r_2 < 1$：次线性衰减（更缓慢局部化）

**对数形式**：
$$\text{score}(m,n) = s_{m,n} - r_1 \log(1 + r_2|m-n|)$$

对数形式的衰减更慢，适合需要较大感受野的任务。

#### 8.3 衰减函数的比较

定义有效注意力范围为使得 $A_{m,n} > \epsilon$ 的最大 $|m-n|$：

| 方法 | 衰减函数 $f(d)$ | 有效范围 $d_{\text{eff}}$ |
|------|----------------|------------------------|
| 硬窗口 | $\begin{cases}0, & d \leq w \\ -\infty, & d > w\end{cases}$ | $w$ |
| ALIBI | $-\lambda d$ | $O(\log(1/\epsilon)/\lambda)$ |
| KERPLE幂律 | $-r_1 d^{r_2}$ | $O((1/r_1)^{1/r_2})$ |
| KERPLE对数 | $-r_1\log(1+r_2 d)$ | $O(\exp(1/r_1))$ |

### 9. XPOS的理论分析

#### 9.1 指数衰减项的引入

XPOS在RoPE基础上添加指数衰减：
$$\boldsymbol{q}_m \to \xi^m \boldsymbol{\mathcal{R}}_m \boldsymbol{q}_m, \quad \boldsymbol{k}_n \to \xi^{-n} \boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n$$

注意力得分为：
$$\text{score}(m,n) = \xi^{m-n} \boldsymbol{q}_m^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}_n$$

#### 9.2 单向性的保证

对于因果注意力（$m \geq n$），相对位置 $m - n \geq 0$，选择 $\xi < 1$ 使得：
$$\xi^{m-n} \leq 1, \quad \forall m \geq n$$

且随 $m - n$ 增大而衰减。

**定理14（XPOS的衰减率）**：XPOS的注意力权重满足：
$$A_{m,n}^{\text{XPOS}} \propto \xi^{m-n}$$

衰减时间常数为：
$$\tau = \frac{1}{-\log \xi}$$

#### 9.3 与ALIBI的关系

XPOS和ALIBI都实现了距离衰减，但方式不同：
- ALIBI：加法偏置，$\exp(-\lambda d)$
- XPOS：乘法因子，$\xi^d$

两者等价当 $\xi = e^{-\lambda}$。

### 10. 总结与理论启示

#### 10.1 主要理论结果

1. **局部性原理**：局部注意力通过限制感受野，保持了训练与测试的注意力熵一致性
2. **外推误差界**：局部注意力的外推误差主要来自位置编码，与序列长度关系较弱
3. **计算效率**：局部注意力将复杂度从 $O(L^2d)$ 降至 $O(Lwd)$，$w \ll L$
4. **感受野增长**：多层局部注意力的感受野线性增长，需要 $O(L/w)$ 层覆盖全序列

#### 10.2 设计指导原则

1. **窗口大小选择**：$w \approx L_{\text{train}}$ 以匹配训练时的注意力模式
2. **层数设计**：$\ell \geq L_{\text{test}}/(2w)$ 以保证足够的感受野
3. **混合策略**：结合局部和全局注意力，平衡效率与表达能力
4. **缩放调整**：使用log缩放补偿序列长度变化

#### 10.3 开放问题

1. 最优窗口大小的理论下界？
2. 不同任务对感受野的最小需求？
3. 稀疏模式的自适应学习方法？
4. 长度外推的泛化界是否可证明？

本文通过严格的数学分析，阐明了局部注意力在长度外推中的关键作用，为Transformer的长序列建模提供了理论基础。

