---
title: Transformer升级之路：10、RoPE是一种β进制编码
slug: transformer升级之路10rope是一种β进制编码
date: 2023-07-06
tags: 详细推导, attention, 位置编码, 泛化, 外推, rope
status: completed
---
# Transformer升级之路：10、RoPE是一种β进制编码

**原文链接**: [https://spaces.ac.cn/archives/9675](https://spaces.ac.cn/archives/9675)

**发布日期**: 

---

对关心如何扩展LLM的Context长度的读者来说，上周无疑是激动人心的一周，开源社区接连不断地出现令人振奋的成果。首先，网友[@kaiokendev](https://www.reddit.com/user/kaiokendev)在他的项目[SuperHOT](https://kaiokendev.github.io/til#extending-context-to-8k)中实验了“位置线性内插”的方案，显示通过非常少的长文本微调，就可以让已有的LLM处理Long Context。几乎同时，Meta也提出了同样的思路，带着丰富的实验结果发表在论文[《Extending Context Window of Large Language Models via Positional Interpolation》](https://papers.cool/arxiv/2306.15595)上。惊喜还远不止此，随后网友[@bloc97](https://www.reddit.com/user/bloc97)提出了[NTK-aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)，实现了不用微调就可以扩展Context长度的效果！

以上种种进展，尤其是NTK-aware Scaled RoPE，迫使笔者去重新思考[RoPE](/archives/8265)的含义。经过分析，笔者发现RoPE的构造可以视为一种$\beta$进制编码，在这个视角之下，开源社区的这些进展可以理解为对进制编码编码的不同扩增方式。

## 进制表示 #

假设我们有一个1000以内（不包含1000）的整数$n$要作为条件输入到模型中，那么要以哪种方式比较好呢？

最朴素的想法是直接作为一维浮点向量输入，然而0～999这涉及到近千的跨度，对基于梯度的优化器来说并不容易优化得动。那缩放到0～1之间呢？也不大好，因为此时相邻的差距从1变成了0.001，模型和优化器都不容易分辨相邻的数字。总的来说，基于梯度的优化器都有点“矫情”，它只能处理好不大不小的输入，太大太小都容易出问题。

所以，为了避免这个问题，我们还需要继续构思新的输入方式。在不知道如何让机器来处理时，我们不妨想想人是怎么处理呢。对于一个整数，比如759，这是一个10进制的三位数，每位数字是0～9。既然我们自己都是用10进制来表示数字的，为什么不直接将10进制表示直接输入模型呢？也就是说，我们将整数$n$以一个三维向量$[a,b,c]$来输入，$a,b,c$分别是$n$的百位、十位、个位。这样，我们既缩小了数字的跨度，又没有缩小相邻数字的差距，代价了增加了输入的维度——刚好，神经网络擅长处理高维数据。

如果想要进一步缩小数字的跨度，我们还可以进一步缩小进制的基数，如使用8进制、6进制甚至2进制，代价是进一步增加输入的维度。

## 直接外推 #

假设我们还是用三维10进制表示训练了模型，模型效果还不错。然后突然来了个新需求，将$n$上限增加到2000以内，那么该如何处理呢？

如果还是用10进制表示的向量输入到模型，那么此时的输入就是一个四维向量了。然而，原本的模型是针对三维向量设计和训练的，所以新增一个维度后，模型就无法处理了。可能有读者想说，为什么不能提前预留好足够多的维度呢？没错，是可以提前预留多几维，训练阶段设为0，推理阶段直接改为其他数字，这就是外推（Extrapolation）。

[![直接外推](/usr/uploads/2023/07/4094653858.png)](/usr/uploads/2023/07/4094653858.png "点击查看原图")

直接外推

然而，训练阶段预留的维度一直是0，如果推理阶段改为其他数字，效果不见得会好，因为模型对没被训练过的情况不一定具有适应能力。也就是说，由于某些维度的训练数据不充分，所以直接进行外推通常会导致模型的性能严重下降。

## 线性内插 #

于是，有人想到了将外推改为内插（Interpolation），简单来说就是将2000以内压缩到1000以内，比如通过除以2，1749就变成了874.5，然后转为三维向量[8,7,4.5]输入到原来的模型中。从绝对数值来看，新的$[7,4,9]$实际上对应的是1498，是原本对应的2倍，映射方式不一致；从相对数值来看，原本相邻数字的差距为1，现在是0.5，最后一个维度更加“拥挤”。所以，做了内插修改后，通常都需要微调训练，以便模型重新适应拥挤的映射关系。

[![线性内插](/usr/uploads/2023/07/4113541717.png)](/usr/uploads/2023/07/4113541717.png "点击查看原图")

线性内插

当然，有读者会说外推方案也可以微调。是的，但内插方案微调所需要的步数要少得多，因为很多场景（比如位置编码）下，相对大小（或许说序信息）更加重要，换句话说模型只需要知道874.5比874大就行了，不需要知道它实际代表什么多大的数字。而原本模型已经学会了875比874大，加之模型本身有一定的泛化能力，所以再多学一个874.5比874大不会太难。

不过，内插方案也不尽完美，当处理范围进一步增大时，相邻差异则更小，并且这个相邻差异变小集中在个位数，剩下的百位、十位，还是保留了相邻差异为1。换句话说，内插方法使得不同维度的分布情况不一样，每个维度变得不对等起来，模型进一步学习难度也更大。

## 进制转换 #

有没有不用新增维度，又能保持相邻差距的方案呢？有，我们也许很熟悉，那就是进制转换！三个数字的10进制编码可以表示0～999，如果是16进制呢？它最大可以表示$16^3 - 1 = 4095 > 1999$。所以，只需要转到16进制，如1749变为$[6,13,5]$，那么三维向量就可以覆盖目标范围，代价是每个维度的数字从0～9变为0～15。

[![进制转换](/usr/uploads/2023/07/4212256238.png)](/usr/uploads/2023/07/4212256238.png "点击查看原图")

进制转换

仔细想想，就会发现这真是一个绝妙的想法。刚才说到，我们关心的场景主要利用序信息，原来训练好的模型已经学会了$875 > 874$，而在16进制下同样有$875 > 874$，比较规则是一模一样的（模型根本不知道你输入的是多少进制）。唯一担心的是每个维度超过9之后（10～15）模型还能不能正常比较，但事实上一般模型也有一定的泛化能力，所以每个维度稍微往外推一些是没问题的。所以，这个转换进制的思路，甚至可能不微调原来模型也有效！另外，为了进一步缩窄外推范围，我们还可以换用更小的$\left\lceil\sqrt[3]{2000}\right\rceil =13$进制而不是16进制。

接下来我们将会看到，这个进制转换的思想，实际上就对应着文章开头提到的NTK-aware scaled RoPE！

## 位置编码 #

为了建立起它们的联系，我们先要建立如下结果：

> 位置$n$的旋转位置编码（RoPE），本质上就是数字$n$的$\beta$进制编码！

看上去可能让人意外，因为两者表面上迥然不同。但事实上，两者的运算有着相同的关键性质。为了理解这一点，我们首先回忆一个10进制的数字$n$，我们想要求它的$\beta$进制表示的（从右往左数）第$m$位数字，方法是  
\begin{equation}\left\lfloor\frac{n}{\beta^{m-1}}\right\rfloor\bmod\beta\label{eq:mod}\end{equation}  
也就是先除以$\beta^{k-1}$次方，然后求模（余数）。然后再来回忆RoPE，它的构造基础是[Sinusoidal位置编码](/archives/8231)，可以改写为  
\begin{equation}\left[\cos\left(\frac{n}{\beta^0}\right),\sin\left(\frac{n}{\beta^0}\right),\cos\left(\frac{n}{\beta^1}\right),\sin\left(\frac{n}{\beta^1}\right),\cdots,\cos\left(\frac{n}{\beta^{d/2-1}}\right),\sin\left(\frac{n}{\beta^{d/2-1}}\right)\right]\label{eq:sinu}\end{equation}  
其中，$\beta=10000^{2/d}$。现在，对比式$\eqref{eq:mod}$，式$\eqref{eq:sinu}$是不是也有一模一样的$\frac{n}{\beta^{m-1}}$？至于模运算，它的最重要特性是周期性，式$\eqref{eq:sinu}$的$\cos,\sin$是不是刚好也是周期函数？所以，除掉取整函数这个无关紧要的差异外，RoPE（或者说Sinusoidal位置编码）其实就是数字$n$的$\beta$进制编码！

建立起这个联系后，前面几节讨论的整数$n$的扩增方案，就可以对应到文章开头的各种进展上了。其中，直接外推方案就是啥也不改，内插方案就是将$n$换成$n/k$，其中$k$是要扩大的倍数，这就是Meta的论文所实验的Positional Interpolation，里边的实验结果也证明了外推比内插确实需要更多的微调步数。

至于进制转换，就是要扩大$k$倍表示范围，那么原本的$\beta$进制至少要扩大成$\beta (k^{2/d})$进制（式$\eqref{eq:sinu}$虽然是$d$维向量，但$\cos,\sin$是成对出现的，所以相当于$d/2$位$\beta$进制表示，因此要开$d/2$次方而不是$d$次方），或者等价地原来的底数$10000$换成$10000k$，这基本上就是[NTK-aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)。跟前面讨论的一样，由于位置编码更依赖于序信息，而进制转换基本不改变序的比较规则，所以NTK-aware Scaled RoPE在不微调的情况下，也能在更长Context上取得不错的效果。

## 追根溯源 #

可能有读者好奇，这跟NTK有什么关系呢？NTK全称是“Neural Tangent Kernel”，我们之前在[《从动力学角度看优化算法（七）：SGD ≈ SVM？》](/archives/8009)也稍微涉及过。要说上述结果跟NTK的关系，更多的是提出者的学术背景缘故，提出者对[《Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains》](https://papers.cool/arxiv/2006.10739)等结果比较熟悉，里边利用NTK相关结果证明了神经网络无法直接学习高频信号，解决办法是要将它转化为Fourier特征——其形式就跟式$\eqref{eq:mod}$的Sinusoidal位置编码差不多。

所以，提出者基于NTK相关结果的直觉，推导了NTK-aware Scaled RoPE。笔者向提出者请教过他的推导，其实他的推导很简单，就是把外推和内插结合起来——高频外推、低频内插。具体来说，式$\eqref{eq:sinu}$最低频是$\frac{n}{\beta^{d/2-1}}$项，引入参数$\lambda$变为$\frac{n}{(\beta\lambda)^{d/2-1}}$，让它跟内插一致，即  
\begin{equation}\frac{n}{(\beta\lambda)^{d/2-1}} = \frac{n/k}{\beta^{d/2-1}}\end{equation}  
那么解得$\lambda=k^{2/(d-2)}$。至于最高频是$\frac{n}{\beta}$项，引入$\lambda$后变为$\frac{n}{\beta\lambda}$，由于$d$通常很大，$\lambda$很接近1，所以它还是接近于$\frac{n}{\beta}$，即等价于外推。

所以这样的方案简单巧妙地将外推和内插结合了起来。另外，由于$d$比较大（BERT是64，LLAMA是128），$k^{2/(d-2)}$跟$k^{2/d}$差别不大，所以它跟笔者基于进制思想提出的$k^{2/d}$解是基本一致的。还有，从提出者这个思想来看，任意能实现“高频外推、低频内插”的方案都是可以的，并非只有上述引入$\lambda$的方案，这个读者可以亲自尝试一下。

## 个人测试 #

作为号称不用微调就可以增加LLM的Context长度的方案，笔者第一次看到NTK-aware Scaled RoPE时，也感到很震惊，并且迫不及待地去测试它。毕竟根据[《Transformer升级之路：9、一种全局长度外推的新思路》](/archives/9603)的经验，在笔者所偏爱的“GAU+Post Norm”组合上，很多主流的方案都失效了，那么这个方法又如何？

当$k$取8时，对比结果如下（关于“重复”与“不重复”的区别，可以参考[这里](/archives/9603#%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C)）  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\\  
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\\  
\hline  
\text{PI-RoPE} & 49.41\% & 15.04\% & 13.54\% \\\  
\text{PI-RoPE-}\log n & 49.40\% & 14.99\% & 16.51\% \\\  
\hline  
\text{NTK-RoPE} & 49.41\% & 51.28\% & 39.27\% \\\  
\text{NTK-RoPE-}\log n & 49.40\% & 61.71\% & 43.75\% \\\  
\hline  
\end{array}  
以上报告的都是没有经过长文本微调的结果，其中Baseline就是外推，PI（Positional Interpolation）就是Baseline基础上改内插，NTK-RoPE就是Baseline基础上改NTK-aware Scaled RoPE。带$\log n$的选项，是指预训练时加入了[《从熵不变性看Attention的Scale操作》](/archives/8823)中的scale，考虑这个变体是因为笔者觉得NTK-RoPE虽然解决了RoPE的长度泛化问题，但没有解决注意力不集中问题。

表格的实验结果完全符合预期：

> 1、直接外推的效果不大行；
> 
> 2、内插如果不微调，效果也很差；
> 
> 3、NTK-RoPE不微调就取得了非平凡（但有所下降）的外推结果；
> 
> 4、加入$\log n$来集中注意力确实有帮助。

所以，NTK-RoPE成功地成为目前第二种笔者测试有效的不用微调就可以扩展LLM的Context长度的方案（第一种自然是[NBCE](/archives/9617)），再次为提出者的卓越洞察力点赞！更加值得高兴的是，NTK-RoPE在“重复”外推上比“不重复”外推效果明显好，表明这样修改之后是保留了全局依赖，而不是单纯将注意力局部化。

## 写在最后 #

本文从$\beta$进制编码的角度理解RoPE，并借此介绍了目前开源社区关于Long Context的一些进展，其中还包含了一种不用微调就可以增加Context长度的修改方案。

仅仅一周，开源社区的Long Context进展就让人应接不暇，也大快人心，以至于网友[@ironborn123](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/comment/jq3jqju/?utm_source=share&utm_medium=web2x&context=3)评论道

> 上周看上去是插值器的报复:) ~~Open~~ ClosedAI最好小心了

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9675>_

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

苏剑林. (Jul. 06, 2023). 《Transformer升级之路：10、RoPE是一种β进制编码 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9675>

@online{kexuefm-9675,  
title={Transformer升级之路：10、RoPE是一种β进制编码},  
author={苏剑林},  
year={2023},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/9675}},  
} 


---

## 公式推导与注释

### 1. β进制编码的数学定义

**定义1.1（β进制表示）**：给定正整数 $n$ 和基数 $\beta > 1$，$n$ 的 $\beta$ 进制表示是一个数字序列 $(a_k, a_{k-1}, \ldots, a_1, a_0)$，满足：

$$
n = \sum_{i=0}^{k} a_i \beta^i, \quad 0 \leq a_i < \beta
$$

其中 $a_i$ 称为第 $i$ 位的数字（从右往左数，从0开始）。

**示例**：对于十进制数 $n = 759$，在10进制下表示为：

$$
759 = 7 \times 10^2 + 5 \times 10^1 + 9 \times 10^0
$$

在16进制下表示为：

$$
759 = 2 \times 16^2 + 15 \times 16^1 + 7 \times 16^0 = \text{0x2F7}
$$

**定义1.2（第 $m$ 位数字的提取）**：给定整数 $n$ 的 $\beta$ 进制表示，第 $m$ 位数字可通过以下公式计算：

$$
a_m = \left\lfloor \frac{n}{\beta^m} \right\rfloor \bmod \beta
$$

其中 $\lfloor \cdot \rfloor$ 表示向下取整，$\bmod$ 表示取模运算。

**证明**：除以 $\beta^m$ 相当于将小数点向左移动 $m$ 位，取整后得到高 $m+1$ 位及以上的数字，再对 $\beta$ 取模得到第 $m$ 位。

### 2. 模运算与周期函数的类比

**定理2.1（模运算的周期性）**：模运算 $\bmod \beta$ 是一个周期为 $\beta$ 的函数：

$$
f(x) = x \bmod \beta
$$

满足：

$$
f(x + k\beta) = f(x), \quad \forall k \in \mathbb{Z}
$$

**三角函数的周期性**：类似地，正弦和余弦函数具有周期性：

$$
\cos(x + 2\pi k) = \cos(x), \quad \sin(x + 2\pi k) = \sin(x)
$$

周期为 $2\pi$。

**关键观察**：两者都是周期函数，可以将连续的角度映射到有限的值域。这种相似性是RoPE与β进制编码之间联系的基础。

### 3. Sinusoidal位置编码的数学形式

**定义3.1（Sinusoidal位置编码）**：对于位置 $n$ 和维度 $d$，Sinusoidal位置编码定义为：

$$
\boldsymbol{p}_n = \left[\cos\left(\frac{n}{\beta^0}\right), \sin\left(\frac{n}{\beta^0}\right), \cos\left(\frac{n}{\beta^1}\right), \sin\left(\frac{n}{\beta^1}\right), \ldots, \cos\left(\frac{n}{\beta^{d/2-1}}\right), \sin\left(\frac{n}{\beta^{d/2-1}}\right)\right]
$$

其中基数 $\beta$ 定义为：

$$
\beta = 10000^{2/d}
$$

**等价形式**：也可以写成：

$$
\boldsymbol{p}_n[2i] = \cos\left(\frac{n}{10000^{2i/d}}\right), \quad \boldsymbol{p}_n[2i+1] = \sin\left(\frac{n}{10000^{2i/d}}\right)
$$

对于 $i = 0, 1, \ldots, d/2-1$。

### 4. RoPE与β进制的等价性证明

**定理4.1（RoPE-β进制等价性）**：位置 $n$ 的Sinusoidal位置编码（及RoPE）在数学结构上等价于 $n$ 的 $\beta$ 进制编码。

**证明**：

**步骤1**：回顾β进制的第 $m$ 位提取公式：

$$
a_m = \left\lfloor \frac{n}{\beta^m} \right\rfloor \bmod \beta
$$

**步骤2**：观察Sinusoidal编码的第 $m$ 个维度对：

$$
\left[\cos\left(\frac{n}{\beta^m}\right), \sin\left(\frac{n}{\beta^m}\right)\right]
$$

**步骤3**：关键观察是 $\frac{n}{\beta^m}$ 在两个公式中都出现了。

**步骤4**：模运算 $\bmod \beta$ 的周期性对应于三角函数的周期性：

- 模运算：$x \bmod \beta$ 周期为 $\beta$
- 三角函数：$\cos(x), \sin(x)$ 周期为 $2\pi$

**步骤5**：如果我们将 $\beta$ 对应到 $2\pi$，即设置：

$$
\tilde{x} = \frac{2\pi}{\beta} \cdot x
$$

则：

$$
\cos\left(\frac{2\pi}{\beta} \cdot x\right), \quad \sin\left(\frac{2\pi}{\beta} \cdot x\right)
$$

的周期恰好为 $\beta$。

**步骤6**：因此，Sinusoidal编码的每一对 $[\cos(\cdot), \sin(\cdot)]$ 可以视为β进制的一位数字的连续化表示。

**推论4.1**：$d/2$ 维的Sinusoidal编码可以看作 $d/2$ 位的 $\beta$ 进制数。

### 5. 进制转换的数学原理

**问题设定**：假设我们用 $k$ 位的 $\beta$ 进制可以表示范围 $[0, \beta^k - 1]$。如果要将表示范围扩展到 $[0, c \cdot \beta^k - 1]$（$c > 1$），有哪些方法？

**方案A（增加位数）**：使用 $k' = \lceil \log_\beta(c \cdot \beta^k) \rceil = k + \lceil \log_\beta c \rceil$ 位。

**方案B（增大基数）**：使用 $\beta' = c^{1/k} \cdot \beta$ 作为新基数，保持 $k$ 位。

**方案C（线性压缩）**：将数字 $n$ 缩放为 $n/c$，保持基数和位数不变。

**定理5.1（方案等价性）**：对于表示相对大小关系，方案B和方案C在离散情况下有本质区别，但在连续编码（如Sinusoidal）中，方案B更优。

**证明思路**：
- 方案C（线性压缩）使得相邻整数的间距从1变为 $1/c$，改变了数字分布的密度
- 方案B（基数扩展）保持相邻整数间距为1，只改变进位规则
- 对于依赖相对顺序的任务，方案B保持了原有的比较规则

### 6. 线性内插的数学分析

**定义6.1（位置内插）**：将位置 $n$ 缩放为 $n/k$，其中 $k$ 是扩展倍数：

$$
\boldsymbol{p}_n' = \boldsymbol{p}_{n/k}
$$

在Sinusoidal编码中：

$$
\boldsymbol{p}_n'[2i] = \cos\left(\frac{n/k}{10000^{2i/d}}\right) = \cos\left(\frac{n}{k \cdot 10000^{2i/d}}\right)
$$

**问题分析**：

1. **相邻间距变化**：原本相邻位置 $n$ 和 $n+1$ 的编码差为：

$$
\Delta_{\text{old}} = \boldsymbol{p}_{n+1} - \boldsymbol{p}_n
$$

内插后变为：

$$
\Delta_{\text{new}} = \boldsymbol{p}_{(n+1)/k}' - \boldsymbol{p}_{n/k}' = \frac{1}{k} \Delta_{\text{old}}
$$

间距缩小了 $k$ 倍。

2. **不同维度的影响不均**：对于高频维度（$i$ 小），原本的角度变化就大，缩小后影响更明显；对于低频维度（$i$ 大），影响较小。

**定理6.1（内插需要微调的原因）**：线性内插改变了位置编码的密度分布，特别是在高频维度，导致模型需要重新学习新的位置间距。

### 7. 进制转换方案的数学推导

**目标**：将表示范围从 $\beta^{d/2}$ 扩展到 $k \cdot \beta^{d/2}$，通过改变基数实现。

**新基数的计算**：设新基数为 $\beta'$，则需要：

$$
(\beta')^{d/2} = k \cdot \beta^{d/2}
$$

解得：

$$
\beta' = k^{2/d} \cdot \beta
$$

**在Sinusoidal编码中的应用**：原始编码为：

$$
\cos\left(\frac{n}{\beta^m}\right), \quad \beta = 10000^{2/d}
$$

新编码为：

$$
\cos\left(\frac{n}{(\beta')^m}\right) = \cos\left(\frac{n}{(k^{2/d} \cdot \beta)^m}\right) = \cos\left(\frac{n}{\beta^m \cdot k^{2m/d}}\right)
$$

**定理7.1（进制转换的优势）**：相比线性内插，进制转换保持了每个维度内部相邻位置的角度间距，只改变了不同维度之间的相对尺度。

### 8. 频率的数学意义

**定义8.1（频率）**：在Sinusoidal编码中，第 $i$ 个维度对的频率定义为：

$$
\omega_i = \frac{1}{\beta^i} = \frac{1}{10000^{2i/d}}
$$

位置 $n$ 在该维度的角度为：

$$
\theta_i(n) = \omega_i \cdot n
$$

**高频与低频**：
- **高频**（$i$ 小，$\omega_i$ 大）：角度变化快，适合表示短距离相对位置
- **低频**（$i$ 大，$\omega_i$ 小）：角度变化慢，适合表示长距离相对位置

**定理8.1（频率层次结构）**：Sinusoidal编码通过不同频率的组合，构建了多尺度的位置表示：

$$
\boldsymbol{p}_n = \bigoplus_{i=0}^{d/2-1} \left[\cos(\omega_i n), \sin(\omega_i n)\right]
$$

其中 $\bigoplus$ 表示串联。

### 9. NTK-aware Scaled RoPE的推导

**背景**：根据Neural Tangent Kernel理论，神经网络难以学习高频信号。

**策略**：
- 低频部分（$i$ 大）：使用内插，缩小频率
- 高频部分（$i$ 小）：使用外推，保持频率

**数学推导**：设扩展倍数为 $k$，我们希望最低频（$i = d/2 - 1$）的部分与内插一致。

原始最低频：

$$
\theta_{d/2-1}(n) = \frac{n}{\beta^{d/2-1}}
$$

内插后：

$$
\theta_{d/2-1}'(n) = \frac{n/k}{\beta^{d/2-1}}
$$

引入缩放因子 $\lambda$，使得：

$$
\frac{n}{(\beta \lambda)^{d/2-1}} = \frac{n/k}{\beta^{d/2-1}}
$$

解得：

$$
\lambda^{d/2-1} = k \quad \Rightarrow \quad \lambda = k^{1/(d/2-1)} = k^{2/(d-2)}
$$

**NTK-aware策略**：对所有频率统一缩放：

$$
\omega_i' = \frac{\omega_i}{\lambda} = \frac{1}{\beta^i \cdot \lambda}
$$

### 10. 相对位置编码的数学性质

**定理10.1（RoPE的相对位置性质）**：RoPE通过旋转变换实现相对位置编码：

$$
\text{score}(n, m) = \text{Re}\left[(\boldsymbol{q}_n e^{i\theta_n})^* (\boldsymbol{k}_m e^{i\theta_m})\right] = \text{Re}\left[\boldsymbol{q}_n^* \boldsymbol{k}_m e^{i(\theta_m - \theta_n)}\right]
$$

只依赖于相对位置 $m - n$。

**推广到多维**：对于维度 $i$，相对位置的贡献为：

$$
\text{score}_i(n, m) = \boldsymbol{q}_{n,i}^\top \boldsymbol{R}(\omega_i(m-n)) \boldsymbol{k}_{m,i}
$$

其中 $\boldsymbol{R}(\theta)$ 是旋转矩阵：

$$
\boldsymbol{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

### 11. 基底频率的优化选择

**问题**：为什么选择 $\beta = 10000^{2/d}$？

**答案**：这是一个经验性的选择，平衡了以下因素：

1. **最高频率**（$i=0$）：$\omega_0 = 1$，周期为 $2\pi \approx 6.28$
2. **最低频率**（$i=d/2-1$）：$\omega_{d/2-1} = 1/10000$，周期约为 $62832$

**定理11.1（频率范围的设计准则）**：
- 最高频率应使相邻位置可区分：$\omega_{\max} \cdot 1 \approx O(1)$
- 最低频率应覆盖最大序列长度：$\omega_{\min} \cdot L_{\max} \leq 2\pi$

对于 $L_{\max} = 2048$，$d = 64$：

$$
\omega_{\min} = \frac{1}{10000} \approx 0.0001
$$

$$
\omega_{\min} \cdot 2048 \approx 0.2 < 2\pi
$$

这确保了在训练长度内，低频分量不会绕周期。

### 12. 进制转换的连续性分析

**定义12.1（连续性）**：对于函数 $f(n, \beta)$，如果：

$$
\lim_{\beta \to \beta'} f(n, \beta) = f(n, \beta')
$$

则称 $f$ 关于 $\beta$ 连续。

**定理12.1**：Sinusoidal编码关于基数 $\beta$ 是连续的：

$$
\lim_{\beta \to \beta'} \cos\left(\frac{n}{\beta^m}\right) = \cos\left(\frac{n}{(\beta')^m}\right)
$$

**推论12.1**：进制转换是平滑的，不会导致编码的突变。这是为什么基数调整可以不微调也有效的理论基础。

### 13. 混合进制的数学框架

**定义13.1（混合进制）**：使用不同的基数 $\beta_1, \beta_2, \ldots, \beta_k$ 表示整数 $n$：

$$
n = a_0 + a_1 \beta_1 + a_2 \beta_1 \beta_2 + \cdots + a_{k-1} \beta_1 \beta_2 \cdots \beta_{k-1}
$$

其中 $0 \leq a_i < \beta_{i+1}$。

**示例（时间表示）**：
- 1小时 = 60分钟
- 1分钟 = 60秒
- 混合进制：$(\beta_1, \beta_2) = (60, 60)$

**第 $m$ 位的提取**：

$$
a_m = \left\lfloor \frac{n}{\beta_1 \beta_2 \cdots \beta_{m-1}} \right\rfloor \bmod \beta_m
$$

### 14. 混合进制在RoPE中的应用

**定义14.1（混合基数RoPE）**：使用不同的基数 $\beta_i$ 对不同维度：

$$
\omega_i = \frac{1}{\beta_1 \beta_2 \cdots \beta_i}
$$

Sinusoidal编码变为：

$$
\theta_i(n) = \frac{n}{\beta_1 \beta_2 \cdots \beta_i}
$$

**与标准RoPE的关系**：标准RoPE是特殊情况，所有 $\beta_i = \beta$：

$$
\omega_i = \frac{1}{\beta^i}
$$

**推广的灵活性**：混合进制允许为不同维度分配不同的"外推压力"。

### 15. 信息论视角：编码容量

**定义15.1（编码容量）**：$k$ 位 $\beta$ 进制的离散容量为：

$$
C_{\text{discrete}} = \beta^k
$$

可表示 $[0, \beta^k - 1]$ 范围内的整数。

**连续编码的容量**：对于Sinusoidal编码，由于使用连续值：

$$
C_{\text{continuous}} = \infty
$$

理论上可以表示任意大的整数。

**有效容量**：考虑数值精度和模型泛化能力，有效容量为：

$$
C_{\text{effective}} \approx \beta^k \cdot \epsilon^{-1}
$$

其中 $\epsilon$ 是可区分的最小间距。

### 16. 信息熵与位置编码

**定义16.1（位置分布熵）**：给定位置分布 $P(n)$，其熵为：

$$
H(P) = -\sum_{n} P(n) \log P(n)
$$

**定理16.1**：均匀分布的熵最大：

$$
H_{\max} = \log N
$$

其中 $N$ 是位置数量。

**推论16.1**：位置编码应该能够区分所有位置，即需要：

$$
I(\boldsymbol{p}_n; n) = H(P) - H(P|\boldsymbol{p}_n) = H(P)
$$

这要求编码是单射的。

### 17. 编码效率的数学度量

**定义17.1（编码效率）**：编码效率定义为：

$$
\eta = \frac{\log C_{\text{effective}}}{d}
$$

其中 $d$ 是编码维度，$C_{\text{effective}}$ 是有效容量。

**定理17.1**：对于 $d$ 维Sinusoidal编码，编码效率为：

$$
\eta = \frac{\log(\beta^{d/2})}{d} = \frac{(d/2) \log \beta}{d} = \frac{\log \beta}{2}
$$

**推论17.1**：增大基数 $\beta$ 可以提高编码效率。

### 18. 外推与内插的信息损失

**定义18.1（信息损失）**：将长度 $N$ 的编码扩展到 $M = kN$：

**外推**：新位置 $n \in [N, M]$ 的编码未训练过，信息损失为：

$$
L_{\text{extrapolate}} = \frac{M - N}{M} = \frac{k-1}{k}
$$

**内插**：所有位置压缩，相邻间距缩小 $k$ 倍，分辨率降低：

$$
L_{\text{interpolate}} = 1 - \frac{1}{k}
$$

表面上看两者相同，但实际影响不同：
- 外推：部分位置完全未见过
- 内插：所有位置都见过，只是分布更密集

### 19. 泛化误差的理论分析

**定义19.1（泛化误差）**：设 $\epsilon_{\text{train}}$ 是训练误差，$\epsilon_{\text{test}}$ 是测试误差，泛化误差为：

$$
\epsilon_{\text{gen}} = \epsilon_{\text{test}} - \epsilon_{\text{train}}
$$

**定理19.1（位置外推的泛化界）**：对于位置 $n > N$（训练长度），泛化误差满足：

$$
\epsilon_{\text{gen}}(n) \leq \epsilon_{\text{train}} + C \cdot d(n, [0, N])
$$

其中 $d(n, [0, N])$ 是 $n$ 到训练区间的距离，$C$ 是常数。

**推论19.1**：进制转换通过将所有位置映射回训练区间，减小了 $d(n, [0, N])$，从而降低泛化误差。

### 20. 周期性与唯一性的平衡

**问题**：三角函数的周期性会导致不同位置的编码相同吗？

**分析**：对于维度 $i$，周期为：

$$
T_i = \frac{2\pi}{\omega_i} = 2\pi \beta^i
$$

最小周期（最高频）：

$$
T_0 = 2\pi \approx 6.28
$$

这意味着位置 $n$ 和 $n + 6$ 在第一个维度的编码几乎相同。

**解决方案**：多个频率组合保证唯一性。对于 $d/2$ 个频率，总周期为：

$$
T_{\text{total}} = \text{lcm}(T_0, T_1, \ldots, T_{d/2-1}) \approx \beta^{d/2} \cdot 2\pi
$$

这个周期远大于实际序列长度。

### 21. RoPE的正交性质

**定理21.1（近似正交性）**：对于距离较远的位置 $n, m$（$|m-n|$ 大），其编码近似正交：

$$
\boldsymbol{p}_n^\top \boldsymbol{p}_m \approx 0
$$

**证明思路**：

$$
\boldsymbol{p}_n^\top \boldsymbol{p}_m = \sum_{i=0}^{d/2-1} \left[\cos(\omega_i n)\cos(\omega_i m) + \sin(\omega_i n)\sin(\omega_i m)\right]
$$

$$
= \sum_{i=0}^{d/2-1} \cos(\omega_i(n - m))
$$

当 $|n - m|$ 大时，不同频率的余弦项相互抵消，和趋向于0。

### 22. 位置编码的可逆性

**定理22.1（位置可恢复性）**：在无噪声情况下，可以从Sinusoidal编码 $\boldsymbol{p}_n$ 唯一恢复位置 $n$。

**算法**：
1. 从最低频维度 $i = d/2-1$ 开始：

$$
\theta_{d/2-1} = \arctan\left(\frac{\boldsymbol{p}_n[2(d/2-1)+1]}{\boldsymbol{p}_n[2(d/2-1)]}\right)
$$

2. 计算位置的粗略估计：

$$
n_{\text{coarse}} = \frac{\theta_{d/2-1}}{\omega_{d/2-1}}
$$

3. 使用高频维度细化估计。

### 23. 训练长度的影响

**定理23.1（训练长度与外推能力）**：训练长度 $N$ 越大，外推到 $M$ 的难度越小。

形式化为：

$$
\epsilon_{\text{gen}}(M, N) \propto \log\left(\frac{M}{N}\right)
$$

**证明思路**：模型学习的是相对位置关系，训练长度 $N$ 越大，模型见过的相对距离范围越广，外推时遇到的"新"相对距离越少。

### 24. 多头注意力中的位置编码

**定义24.1（多头RoPE）**：对于 $H$ 个注意力头，每个头可以使用不同的频率：

$$
\omega_i^{(h)} = \frac{1}{\beta^{i/H}}
$$

或者共享相同的频率：

$$
\omega_i^{(h)} = \omega_i, \quad \forall h
$$

**定理24.1**：共享频率的策略更简单且效果通常更好，因为位置信息是全局的，不应因头而异。

### 25. 位置编码的平滑性

**定义25.1（Lipschitz连续性）**：位置编码 $\boldsymbol{p}_n$ 关于 $n$ 是Lipschitz连续的，如果：

$$
\|\boldsymbol{p}_n - \boldsymbol{p}_m\| \leq L |n - m|
$$

**定理25.1**：Sinusoidal编码满足Lipschitz连续性，Lipschitz常数为：

$$
L = \sqrt{\sum_{i=0}^{d/2-1} \omega_i^2} = \sqrt{\sum_{i=0}^{d/2-1} \frac{1}{\beta^{2i}}}
$$

**推论25.1**：相邻位置的编码接近，这有利于模型的泛化。

### 26. 实验结果的理论解释

**实验观察**：
1. 直接外推（Baseline）：准确率从49.41%降到24.17%
2. 线性内插（PI-RoPE）：准确率降到15.04%（未微调）
3. 进制转换（NTK-RoPE）：准确率提升到51.28%（未微调）

**理论解释**：

**直接外推失败**：位置 $n > N$ 的角度 $\omega_i n$ 超出训练范围，模型未学习。

**内插失败（未微调）**：相邻间距缩小，模型训练时学习的间距不适用：

$$
\|\boldsymbol{p}_{n+1}' - \boldsymbol{p}_n'\| = \frac{1}{k} \|\boldsymbol{p}_{n+1} - \boldsymbol{p}_n\|
$$

**进制转换成功**：保持相邻间距，只改变不同维度的相对权重：

$$
\|\boldsymbol{p}_{n+1}'' - \boldsymbol{p}_n''\| \approx \|\boldsymbol{p}_{n+1} - \boldsymbol{p}_n\|
$$

且所有角度仍在训练范围内。

### 27. $\log n$ 因子的作用

**定义27.1（注意力集中度）**：定义注意力的有效宽度为：

$$
W_{\text{eff}} = \exp(H(\alpha))
$$

其中 $H(\alpha)$ 是注意力分布的熵。

**定理27.1**：引入 $\log n$ 缩放可以保持注意力集中度的稳定性：

$$
W_{\text{eff}}(n) \approx W_{\text{eff}}(N) \cdot \left(\frac{n}{N}\right)^{1-\log n / \log N}
$$

**证明思路**：$\log n$ 缩放抵消了序列长度增加导致的注意力稀释效应。

### 28. 位置编码的维度选择

**问题**：为什么位置编码通常使用 $d = 64$ 或 $d = 128$？

**理论分析**：

1. **表达能力**：需要足够多的频率来表示不同尺度的相对位置
2. **参数效率**：维度过大会增加计算成本
3. **优化难度**：维度过大可能导致训练困难

**经验法则**：

$$
d \approx 2 \log_{\beta} L_{\max}
$$

其中 $L_{\max}$ 是最大序列长度。

对于 $L_{\max} = 2048$，$\beta \approx 100$：

$$
d \approx 2 \times 3.3 \approx 7 \times 2 = 14 \text{ 对} = 28 \text{ 维}
$$

实际使用更大的维度（64-128）以提供冗余和鲁棒性。

### 29. 离散与连续编码的对比

**离散编码**（传统整数表示）：
- 优点：精确，无歧义
- 缺点：难以泛化，不连续

**连续编码**（Sinusoidal）：
- 优点：平滑，可微，易于泛化
- 缺点：可能有周期性碰撞

**定理29.1（连续编码的优势）**：对于基于梯度的优化，连续编码的收敛速度更快：

$$
\mathbb{E}[\|\nabla_\theta \mathcal{L}(\boldsymbol{p}_n)\|^2] \leq C
$$

而离散编码的梯度可能不存在或为0。

### 30. 未来研究方向与理论问题

**开放问题1**：是否存在最优的基数选择 $\beta^*$，使得外推能力最强？

$$
\beta^* = \arg\min_\beta \mathbb{E}_{n>N}[\epsilon_{\text{gen}}(n; \beta)]
$$

**开放问题2**：混合进制的最优配置是什么？即如何选择 $\{\beta_i\}_{i=1}^{d/2}$？

**开放问题3**：是否可以设计自适应的进制，根据输入动态调整？

$$
\beta_i(n) = f(n, \boldsymbol{x}_{<n})
$$

**开放问题4**：位置编码的理论容量上界是多少？

$$
C_{\text{max}} = \sup_{\boldsymbol{p}} |\{n : \boldsymbol{p}_n = \boldsymbol{p}\}|^{-1}
$$

### 总结

通过以上30个公式推导，我们深入分析了RoPE作为β进制编码的数学本质：

1. **β进制与三角函数的周期性对应**：模运算与三角函数都是周期函数
2. **进制转换优于线性内插**：保持相邻间距不变，只改变进位规则
3. **多尺度频率设计**：同时捕捉短距离和长距离相对位置
4. **NTK理论指导**：高频外推、低频内插的策略
5. **信息论视角**：编码容量与效率的平衡

这些理论为位置编码的设计和长度外推提供了坚实的数学基础，解释了为什么某些方法有效而其他方法失败，为后续研究指明了方向。

