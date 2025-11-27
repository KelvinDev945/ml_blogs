---
title: Bias项的神奇作用：RoPE + Bias = 更好的长度外推性
slug: bias项的神奇作用rope-bias-更好的长度外推性
date: 2023-04-03
tags: 语言模型, attention, 位置编码, 外推, rope
status: completed
---

# Bias项的神奇作用：RoPE + Bias = 更好的长度外推性

**原文链接**: [https://spaces.ac.cn/archives/9577](https://spaces.ac.cn/archives/9577)

**发布日期**: 

---

万万没想到，Bias项能跟Transformer的长度外推性联系在一起！

长度外推性是我们希望Transformer具有的一个理想性质，笔者曾在[《Transformer升级之路：7、长度外推性与局部注意力》](/archives/9431)、[《Transformer升级之路：8、长度外推性与位置鲁棒性》](/archives/9444)系统地介绍过这一问题。至于Bias项（偏置项），目前的主流观点是当模型足够大时，Bias项不会有什么特别的作用，所以很多模型选择去掉Bias项，其中代表是Google的[T5](/archives/7867)和[PaLM](https://papers.cool/arxiv/2204.02311)，我们后面做的[RoFormerV2](/archives/8998)和[GAU-α](/archives/9052)也沿用了这个做法。

那么，这两个看上去“风牛马不相及”的东西，究竟是怎么联系起来的呢？Bias项真的可以增强Transformer的长度外推性？且听笔者慢慢道来。

## 隐藏彩蛋 #

首先，为什么会想到考察Bias项和长度外推性的联系呢？这是因为笔者前几天在重温GAU的论文[《Transformer Quality in Linear Time》](https://papers.cool/arxiv/2202.10447)时，发现了之前没有在意的一个“隐藏彩蛋”——加性相对位置编码，其伪代码为  


[![GAU的加性相对位置编码的伪代码](/usr/uploads/2023/04/1959476500.png)](/usr/uploads/2023/04/1959476500.png "点击查看原图")

GAU的加性相对位置编码的伪代码

这里我们主要看$n\geq 512$的部分，如果写成公式，大致是  
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \quad\to\quad \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n+ \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b}\label{eq:rel-bias}\end{equation}  
其中$\boldsymbol{\mathcal{R}}_m,\boldsymbol{\mathcal{R}}_n$是RoPE的旋转矩阵，$\boldsymbol{a},\boldsymbol{b}$是两个可学习参数。

这个加性相对位置编码其实之前也留意到了，但当时的评价只是“不理解为什么同时用几种位置编码”，而最近笔者一直在思考长度外推性问题，所以对这个形式就比较敏感了。可以证明，当$\boldsymbol{a}=\boldsymbol{b}=[\sqrt{\lambda},0,\sqrt{\lambda},0,\cdots,\sqrt{\lambda},0]^{\top}$时，结果正好是[《Transformer升级之路：7、长度外推性与局部注意力》](/archives/9431)介绍的能改善长度外推性的Sandwich ，其原理就是$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b}$呈现出关于$|m-n|$递减的趋势，加到注意力矩阵上后，能够起到局部化注意力的作用，而根据[《Transformer升级之路：7、长度外推性与局部注意力》](/archives/9431)，注意力局部化是语言模型外推性的关键。

所以笔者不禁猜测，难道原论文中的这个加性相对位置编码，就是用来增强长度外推性的？GAU的作者竟然如此有先见之明，早在Sandwich之前就提出了类似的想法来解决长度外推性问题？

## 换成偏置 #

不过，对于笔者来说，这种往Attention矩阵上额外加上一项来增强长度外推性的方案都显得不够优雅，所以不管原作者意图如何以及实际效果如何，笔者都不倾向这样做。有什么类似的但几乎“无感”的方案呢？笔者考虑到，如果$\boldsymbol{a}$、$\boldsymbol{b}$分别是$\boldsymbol{q}_m,\boldsymbol{k}_n$的Bias项，或许可以起到类似的效果，即考虑  
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \quad\to\quad (\boldsymbol{q}_m + \boldsymbol{a})^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n(\boldsymbol{k}_n + \boldsymbol{b})\end{equation}  
很明显，单纯增加一个Bias项，不管从形式上还是计算量上看都几乎是“无感”的，如果这样就能增强长度外推性，无疑是一个很漂亮的方案。是否可行呢？我们先来看展开后的结果：  
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n + \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n + \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b} + \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b} \label{eq:bias}\end{equation}  
其中第一项和第四项正好对应公式$\eqref{eq:rel-bias}$，它们都是我们想要的，所以我们想看看第二项和第三项起到什么作用，如果它们不会有什么明显的效应，那么直接加上Bias项的做法，至少是“有希望”能够取得跟式$\eqref{eq:rel-bias}$或者Sandwich相似的外推效果。

笔者是这样想的：作为Attention的Query和Key，$\boldsymbol{q}_m$、$\boldsymbol{k}_n$应该是比较“各向同性”的，即它们的方向比较均匀，接近球面上均匀采样，而$\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}$只是一个正交变换，它不改变$\boldsymbol{q}_m$、$\boldsymbol{k}_n$的各向同性性质，那么$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n $、$\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b}$这两项，就相当于从各向同性分布采样出来的向量，跟一个固定向量的内积，根据我们在[《n维空间下两个随机向量的夹角分布》](/archives/7076)中的讨论，这样的两个向量夹角应该是很接近90度的，换言之这个内积的期望应该是0，所以第二项和第三项的效应理论上没有剩余两项那么强。

当然，这仅仅是猜测，实际它会训练成怎样，只能通过实验来确定。所以事不宜迟，笔者立刻进行了实验。

## 实验结果 #

这次笔者选了语言模型任务进行实验，模型架构还是之前的[GAU-α](/archives/9052)，训练长度和batch_size都是512，优化器是[Tiger](/archives/9512)，两个模型的唯一差别就是Q、K的Bias是否开启（其他Bias仍被去掉）。

外推效果上的对比：  
$$\begin{array}{c}  
\text{不同测试长度下的LM准确率} \\\  
{\begin{array}{c|cccc}  
\hline  
& 512 & 1024 & 2048 & 4096 \\\  
\hline  
\text{w/o Bias} & 52.37\% & 33.15\% & 22.85\% & 17.87\% \\\  
\text{w/ Bias} & 52.75\% & 50.99\% & 45.25\% & 39.55\% \\\  
\hline  
\end{array}}  
\end{array}$$  
可以看到，Bias项确实不怎么影响训练效果（512长度），但却在长度外推性上面明显拉开了差距，看似毫无存在感的Bias项居然有此神奇作用！当然，要是重跑几次实验，外推性的结果可能会有明显的波动，毕竟长度外推性属于“赠送功能”，并不是我们主动触发的。

为了验证剩下生效机制是否如我们猜测，笔者可视化了式$\eqref{eq:bias}$的四项在某个样本某一层的变化规律：  


[![加上Bias后四项内积对比](/usr/uploads/2023/04/83521782.svg)](/usr/uploads/2023/04/83521782.svg "点击查看原图")

加上Bias后四项内积对比

可以看到，第4项确确实实呈现衰减趋势，并且其大小占据了主导地位，将这四项叠加起来，与没有加Bias的模型对比如下：  


[![有无Bias的Attention矩阵对比](/usr/uploads/2023/04/2535762443.svg)](/usr/uploads/2023/04/2535762443.svg "点击查看原图")

有无Bias的Attention矩阵对比

没有Bias的模型（蓝色），Attention在训练长度（512）范围内确实也呈现出衰减趋势，但长度增加之后就上升了，没有明显的局部性，这就是它外推性不够好的原因；相反，跟前面的猜测一致，带有Bias项的模型（橙色）的注意力矩阵呈现更明显的衰减趋势，换言之它的局部化效应更加强，从而有更好的外推性能。需要指出的是，加上Bias的模型并不是每一层的Attention都有这么明显的衰减趋势，总体来说前面的层衰减趋势更明显些，后面的层衰减趋势更弱些，说明越靠近输入的层越关注局部信息，这跟[《The Devil in Linear Transformer》](https://papers.cool/arxiv/2210.10340)的结论一致。

**【注：后来经过反复测试发现，发现此篇文章的长度外推结果可复现性比较不稳定（可能跟模型结构、超参数等紧密相关），请自行斟酌使用。】**

## 延伸思考 #

这时候问题就来了：之前做长度外推性的工作不是都验证了RoPE的外推性不大好了吗？难道它们都没加Bias？为此，笔者特意去考证了一下，果然”不出所料”：“开山之作”ALIBI和最近的XPOS都是没有加Bias项的，而KERPLE和Sandwich则是加了Bias项的。之前笔者在读论文的时候，就一直感觉KERPLE和Sandwich中的RoPE外推效果似乎比ALIBI和XPOS中的好，现在可以肯定这应该不是错觉了，既然KERPLE和Sandwich都加了Bias，那么根据本文的结论，RoPE是可能呈现出更好的长度外推性的。

可能有读者想起，之前不是说Attention的Key的Bias可以去掉吗？难道这里也可以去掉？关于这个问题，可以参考知乎的提问[《为什么有的 Vision Transformer 中的 key 不需要 bias ？》](https://www.zhihu.com/question/506218961)，事实上，“可以去掉Key的Bias”这个结论，是针对没有RoPE的Attention的，由于Softmax的存在，加上的bias可以约掉：  
\begin{equation}\frac{e^{\boldsymbol{q}\cdot(\boldsymbol{k}_n + \boldsymbol{b})}}{\sum\limits_n e^{\boldsymbol{q}\cdot(\boldsymbol{k}_n + \boldsymbol{b})}} = \frac{e^{\boldsymbol{q}\cdot\boldsymbol{k}_n}e^{\boldsymbol{q}\cdot\boldsymbol{b}}}{\sum\limits_n e^{\boldsymbol{q}\cdot\boldsymbol{k}_n} e^{\boldsymbol{q}\cdot\boldsymbol{b}}}= \frac{e^{\boldsymbol{q}\cdot\boldsymbol{k}_n}}{\sum\limits_n e^{\boldsymbol{q}\cdot\boldsymbol{k}_n}}\end{equation}  
然而，这个“可以约掉”依赖于$\boldsymbol{b}$跟$n$无关，但从式$\eqref{eq:bias}$我们就知道，经过RoPE后，$\boldsymbol{b}$也算是$m,n$的函数了，实际上是无法约掉的，因此对于加了RoPE的模型，Bias项去掉前后会有不一样的效果。

还有一个问题，就是为什么要费力探索长度外推性呢？直接在更长的样本下微调模型不行吗？事实上，即便是对于抱有这样想法的读者，长度外推性也是有好处的。抛开算力不说，更好的长度外推性意味着在微调的时候与预训练差距更小，于是微调更不容易发生灾难性遗忘，这对于当前的LLM更为重要了。当然，还可以发散一下，最理想的结果是：在短文本学习的模型，能够切换到长文本场景而无损效果甚至效果更优。

## 文章小结 #

本文分享了笔者发现的一个“万万没想到”的有趣结论：Bias项能增强RoPE模型的长度外推性！看上去毫无存在感的Bias项，居然能跟Transformer的长度外推性联系在一起，让人不得不感叹细节的重要性——细枝末节有时候也能发挥关键作用。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9577>_

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

苏剑林. (Apr. 03, 2023). 《Bias项的神奇作用：RoPE + Bias = 更好的长度外推性 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9577>

@online{kexuefm-9577,  
title={Bias项的神奇作用：RoPE + Bias = 更好的长度外推性},  
author={苏剑林},  
year={2023},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9577}},  
} 


---

## 公式推导与注释

### 1. Attention机制的基础数学原理

**Scaled Dot-Product Attention的标准形式**：

对于查询矩阵$\boldsymbol{Q} \in \mathbb{R}^{n \times d}$、键矩阵$\boldsymbol{K} \in \mathbb{R}^{n \times d}$和值矩阵$\boldsymbol{V} \in \mathbb{R}^{n \times d_v}$，Attention机制定义为：

\begin{equation}
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d}}\right)\boldsymbol{V} \tag{1}
\end{equation}

**注释**：这里$n$是序列长度，$d$是隐藏维度，$\sqrt{d}$是缩放因子，用于控制点积的方差。

**逐元素展开**：

对于位置$m$的查询和位置$n$的键，它们的注意力得分为：

\begin{equation}
s_{mn} = \frac{\boldsymbol{q}_m \cdot \boldsymbol{k}_n}{\sqrt{d}} = \frac{1}{\sqrt{d}}\sum_{i=1}^d q_{m,i} k_{n,i} \tag{2}
\end{equation}

**Softmax归一化**：

\begin{equation}
\alpha_{mn} = \frac{\exp(s_{mn})}{\sum_{j=1}^N \exp(s_{mj})} \tag{3}
\end{equation}

**注释**：Softmax确保对每个查询位置$m$，所有注意力权重$\alpha_{mn}$的和为1，即$\sum_{j=1}^N \alpha_{mj} = 1$。

**输出计算**：

\begin{equation}
\boldsymbol{o}_m = \sum_{n=1}^N \alpha_{mn} \boldsymbol{v}_n \tag{4}
\end{equation}

### 2. 绝对位置编码的数学表示

**正弦位置编码（Sinusoidal Position Encoding）**：

对于位置$m$和维度$i$，位置编码定义为：

\begin{equation}
\text{PE}(m, 2i) = \sin\left(\frac{m}{10000^{2i/d}}\right) \tag{5}
\end{equation}

\begin{equation}
\text{PE}(m, 2i+1) = \cos\left(\frac{m}{10000^{2i/d}}\right) \tag{6}
\end{equation}

**注释**：这种编码使用不同频率的正弦和余弦函数，使模型能够学习相对位置关系。

**加性位置编码**：

\begin{equation}
\boldsymbol{x}_m' = \boldsymbol{x}_m + \text{PE}(m) \tag{7}
\end{equation}

其中$\boldsymbol{x}_m$是位置$m$的原始嵌入向量。

### 3. 旋转位置编码（RoPE）的核心理论

**旋转矩阵的定义**：

对于二维空间中的旋转，旋转矩阵定义为：

\begin{equation}
\boldsymbol{R}(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix} \tag{8}
\end{equation}

**旋转矩阵的性质**：

1. **正交性**：$\boldsymbol{R}^{\top}\boldsymbol{R} = \boldsymbol{I}$
2. **行列式**：$\det(\boldsymbol{R}) = 1$
3. **内积保持**：$(\boldsymbol{R}\boldsymbol{a})^{\top}(\boldsymbol{R}\boldsymbol{b}) = \boldsymbol{a}^{\top}\boldsymbol{b}$

**证明内积保持性**：

\begin{equation}
(\boldsymbol{R}\boldsymbol{a})^{\top}(\boldsymbol{R}\boldsymbol{b}) = \boldsymbol{a}^{\top}\boldsymbol{R}^{\top}\boldsymbol{R}\boldsymbol{b} = \boldsymbol{a}^{\top}\boldsymbol{I}\boldsymbol{b} = \boldsymbol{a}^{\top}\boldsymbol{b} \tag{9}
\end{equation}

**注释**：内积保持性是RoPE的关键性质，它保证旋转不改变向量的范数和夹角。

**高维旋转矩阵的分块构造**：

对于$d$维向量（$d$为偶数），将其分为$d/2$个二维子空间，每个子空间使用不同的旋转角度：

\begin{equation}
\boldsymbol{\mathcal{R}}_m = \begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix} \tag{10}
\end{equation}

其中$\theta_i = 10000^{-2i/d}$是第$i$个子空间的基础旋转频率。

### 4. RoPE的相对位置性质

**相对位置编码的推导**：

应用RoPE后，位置$m$的查询和位置$n$的键的内积为：

\begin{equation}
\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n = \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}_n \tag{11}
\end{equation}

**证明**：利用旋转矩阵的群性质：

\begin{equation}
\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{-m}\boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{n-m} \tag{12}
\end{equation}

**注释**：这个性质表明，RoPE使得Attention得分只依赖于相对位置$n-m$，而不依赖于绝对位置$m$和$n$。

**具体计算示例（二维情况）**：

设$\boldsymbol{q}_m = [q_1, q_2]^{\top}$，$\boldsymbol{k}_n = [k_1, k_2]^{\top}$，旋转角度$\theta = 1$。

位置$m=1$的旋转矩阵：
\begin{equation}
\boldsymbol{\mathcal{R}}_1 = \begin{pmatrix}
\cos(1) & -\sin(1) \\
\sin(1) & \cos(1)
\end{pmatrix} \approx \begin{pmatrix}
0.540 & -0.841 \\
0.841 & 0.540
\end{pmatrix} \tag{13}
\end{equation}

位置$m=3$的旋转矩阵：
\begin{equation}
\boldsymbol{\mathcal{R}}_3 = \begin{pmatrix}
\cos(3) & -\sin(3) \\
\sin(3) & \cos(3)
\end{pmatrix} \approx \begin{pmatrix}
-0.990 & -0.141 \\
0.141 & -0.990
\end{pmatrix} \tag{14}
\end{equation}

相对旋转矩阵：
\begin{equation}
\boldsymbol{\mathcal{R}}_1^{\top}\boldsymbol{\mathcal{R}}_3 = \boldsymbol{\mathcal{R}}_2 = \begin{pmatrix}
\cos(2) & -\sin(2) \\
\sin(2) & \cos(2)
\end{pmatrix} \tag{15}
\end{equation}

### 5. Bias项在标准Attention中的作用

**不带位置编码的Attention与Bias**：

标准的Query和Key计算包含Bias项：

\begin{equation}
\boldsymbol{q}_m = \boldsymbol{x}_m\boldsymbol{W}_q + \boldsymbol{b}_q \tag{16}
\end{equation}

\begin{equation}
\boldsymbol{k}_n = \boldsymbol{x}_n\boldsymbol{W}_k + \boldsymbol{b}_k \tag{17}
\end{equation}

**内积展开**：

\begin{equation}
\boldsymbol{q}_m \cdot \boldsymbol{k}_n = (\boldsymbol{x}_m\boldsymbol{W}_q + \boldsymbol{b}_q)^{\top}(\boldsymbol{x}_n\boldsymbol{W}_k + \boldsymbol{b}_k) \tag{18}
\end{equation}

\begin{equation}
= \boldsymbol{x}_m\boldsymbol{W}_q\boldsymbol{W}_k^{\top}\boldsymbol{x}_n^{\top} + \boldsymbol{x}_m\boldsymbol{W}_q\boldsymbol{b}_k^{\top} + \boldsymbol{b}_q^{\top}\boldsymbol{W}_k^{\top}\boldsymbol{x}_n^{\top} + \boldsymbol{b}_q^{\top}\boldsymbol{b}_k \tag{19}
\end{equation}

**Softmax的常数消除性质**：

对于最后一项常数$\boldsymbol{b}_q^{\top}\boldsymbol{b}_k$，在Softmax中会被消除：

\begin{equation}
\frac{e^{s_n + c}}{\sum_j e^{s_j + c}} = \frac{e^c \cdot e^{s_n}}{e^c \sum_j e^{s_j}} = \frac{e^{s_n}}{\sum_j e^{s_j}} \tag{20}
\end{equation}

**注释**：因此，在标准Attention中，Key的Bias可以被省略，因为它只会在所有位置上加一个常数。

### 6. RoPE与Bias的结合

**带RoPE的Query和Key**：

\begin{equation}
\tilde{\boldsymbol{q}}_m = \boldsymbol{\mathcal{R}}_m(\boldsymbol{x}_m\boldsymbol{W}_q + \boldsymbol{b}_q) = \boldsymbol{\mathcal{R}}_m\boldsymbol{q}_m \tag{21}
\end{equation}

\begin{equation}
\tilde{\boldsymbol{k}}_n = \boldsymbol{\mathcal{R}}_n(\boldsymbol{x}_n\boldsymbol{W}_k + \boldsymbol{b}_k) = \boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \tag{22}
\end{equation}

**内积展开（关键推导）**：

\begin{equation}
\tilde{\boldsymbol{q}}_m^{\top}\tilde{\boldsymbol{k}}_n = (\boldsymbol{q}_m + \boldsymbol{a})^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n(\boldsymbol{k}_n + \boldsymbol{b}) \tag{23}
\end{equation}

其中$\boldsymbol{a} = \boldsymbol{b}_q$，$\boldsymbol{b} = \boldsymbol{b}_k$。

**四项分解**：

\begin{equation}
= \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n + \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n + \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b} + \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b} \tag{24}
\end{equation}

**项的解释**：

1. **第一项**：标准的RoPE项，依赖于相对位置$n-m$
2. **第二项**：Bias与Key的交互，引入新的位置依赖
3. **第三项**：Query与Bias的交互，引入新的位置依赖
4. **第四项**：纯Bias项，产生衰减的相对位置模式

### 7. 第四项的衰减性质分析

**相对旋转的向量表示**：

对于二维子空间$(i, i+1)$，旋转后的Bias向量为：

\begin{equation}
\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{b}_{[i:i+1]} = \begin{pmatrix}
\cos((n-m)\theta_i) & -\sin((n-m)\theta_i) \\
\sin((n-m)\theta_i) & \cos((n-m)\theta_i)
\end{pmatrix}\begin{pmatrix}
b_i \\
b_{i+1}
\end{pmatrix} \tag{25}
\end{equation}

**Bias向量的特殊选择**：

如果选择$\boldsymbol{a} = \boldsymbol{b} = [\sqrt{\lambda}, 0, \sqrt{\lambda}, 0, \ldots, \sqrt{\lambda}, 0]^{\top}$，则：

\begin{equation}
\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{b} = \sum_{i=1}^{d/2} \lambda \cos((n-m)\theta_i) \tag{26}
\end{equation}

**衰减性证明**：

当$|n-m|$增大时，由于$\theta_i$的不同频率，余弦项会产生相位抵消，导致和值衰减。

**具体示例（$d=4$的情况）**：

设$\theta_1 = 0.1$，$\theta_2 = 0.01$，$\lambda = 1$：

- $n-m=0$：$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_0\boldsymbol{b} = 1 \cdot \cos(0) + 1 \cdot \cos(0) = 2$
- $n-m=10$：$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{10}\boldsymbol{b} = \cos(1) + \cos(0.1) \approx 1.535$
- $n-m=50$：$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{50}\boldsymbol{b} = \cos(5) + \cos(0.5) \approx 0.595$
- $n-m=100$：$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{100}\boldsymbol{b} = \cos(10) + \cos(1) \approx -0.304$

**注释**：可以看到，随着相对距离增大，内积值呈现衰减振荡的趋势。

### 8. 各向同性假设与第二、三项的分析

**各向同性分布的定义**：

如果向量$\boldsymbol{v}$的方向在单位球面上均匀分布，则称$\boldsymbol{v}$服从各向同性分布。

**数学表示**：

对于归一化向量$\boldsymbol{v} = \frac{\boldsymbol{x}}{\|\boldsymbol{x}\|}$，其中$\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$，则$\boldsymbol{v}$在单位球面上均匀分布。

**两个各向同性向量的内积期望**：

对于独立的各向同性向量$\boldsymbol{u}$和$\boldsymbol{v}$：

\begin{equation}
\mathbb{E}[\boldsymbol{u}^{\top}\boldsymbol{v}] = 0 \tag{27}
\end{equation}

**证明**：由对称性，$\boldsymbol{u}^{\top}\boldsymbol{v}$的分布关于0对称，因此期望为0。

**第二项的期望分析**：

\begin{equation}
\mathbb{E}[\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n] = \mathbb{E}[\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}_n] \tag{28}
\end{equation}

由于$\boldsymbol{\mathcal{R}}_{n-m}$是正交变换，它不改变$\boldsymbol{k}_n$的各向同性性质，因此：

\begin{equation}
\mathbb{E}[\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}_n] = \mathbb{E}[\boldsymbol{a}^{\top}\boldsymbol{u}] = 0 \tag{29}
\end{equation}

其中$\boldsymbol{u} = \boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}_n$也是各向同性的。

**注释**：第二项和第三项的期望为0，意味着它们不会产生系统性的偏置，主要贡献来自第一项和第四项。

### 9. 高维球面上的几何理解

**$d$维单位球面的体积元**：

$d$维单位球面$S^{d-1}$的表面积为：

\begin{equation}
A_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)} \tag{30}
\end{equation}

**两个随机向量夹角的概率密度**：

在$d$维空间中，两个各向同性向量的夹角$\theta$的概率密度为：

\begin{equation}
p(\theta) = \frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)} (\sin\theta)^{d-2}, \quad \theta \in [0, \pi] \tag{31}
\end{equation}

**期望夹角**：

\begin{equation}
\mathbb{E}[\theta] = \int_0^{\pi} \theta \cdot p(\theta) d\theta \tag{32}
\end{equation}

当$d$很大时，$\mathbb{E}[\theta] \approx \pi/2$，即高维空间中的随机向量几乎正交。

**内积期望的计算**：

\begin{equation}
\mathbb{E}[\boldsymbol{u}^{\top}\boldsymbol{v}] = \|\boldsymbol{u}\|\|\boldsymbol{v}\|\mathbb{E}[\cos\theta] \tag{33}
\end{equation}

对于归一化向量（$\|\boldsymbol{u}\| = \|\boldsymbol{v}\| = 1$）：

\begin{equation}
\mathbb{E}[\cos\theta] = \int_0^{\pi} \cos\theta \cdot p(\theta) d\theta = 0 \tag{34}
\end{equation}

### 10. 长度外推性的理论基础

**外推问题的定义**：

模型在训练长度$N_{\text{train}}$上训练，但需要在测试长度$N_{\text{test}} > N_{\text{train}}$上推理。

**外推失败的原因**：

1. **位置编码的分布偏移**：测试时的位置超出训练范围
2. **Attention模式的崩溃**：未见过的相对距离导致Attention权重异常
3. **数值不稳定**：大的位置索引导致数值溢出或下溢

**RoPE的外推性质**：

RoPE的相对位置性质理论上支持外推：

\begin{equation}
\text{score}(m, n) = f(n-m) \tag{35}
\end{equation}

只要相对距离$n-m$在合理范围内，Attention得分就应该合理。

**实际外推困难**：

尽管有相对位置性质，实际中RoPE的外推仍然困难，原因包括：

1. **旋转频率的不匹配**：高频分量在长距离上振荡剧烈
2. **Attention分布的偏移**：训练时学到的局部性模式在长序列上不适用
3. **层间累积效应**：误差在多层中累积

### 11. Bias项增强外推性的机制

**局部化Attention的必要性**：

外推性好的模型应该具有局部化的Attention模式，即：

\begin{equation}
\alpha_{mn} \text{ 应随 } |n-m| \text{ 增大而衰减} \tag{36}
\end{equation}

**Bias项产生的衰减效应**：

第四项$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{b}$提供了一个与$|n-m|$相关的衰减项：

\begin{equation}
s_{mn} \approx \underbrace{\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}_n}_{\text{内容相关}} + \underbrace{\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{b}}_{\text{距离偏置}} \tag{37}
\end{equation}

**Softmax后的效果**：

加入距离偏置后，Softmax变为：

\begin{equation}
\alpha_{mn} = \frac{\exp(s_{mn}^{\text{content}} + s_{mn}^{\text{bias}})}{\sum_j \exp(s_{mj}^{\text{content}} + s_{mj}^{\text{bias}})} \tag{38}
\end{equation}

由于$s_{mn}^{\text{bias}}$随$|n-m|$衰减，远距离的$\alpha_{mn}$会被抑制。

**数值示例**：

假设内容得分均匀：$s_{mn}^{\text{content}} = 1$，距离偏置：$s_{mn}^{\text{bias}} = -0.1|n-m|$

不带Bias（$m=5$，$N=10$）：
\begin{equation}
\alpha_{5n} = \frac{e^1}{\sum_{j=1}^{10} e^1} = \frac{1}{10} = 0.1 \tag{39}
\end{equation}

带Bias（$m=5$，$N=10$）：
\begin{equation}
\alpha_{55} = \frac{e^{1+0}}{e^1 + 2e^{0.9} + 2e^{0.8} + 2e^{0.7} + 2e^{0.6}} \approx 0.135 \tag{40}
\end{equation}

\begin{equation}
\alpha_{51} = \frac{e^{1-0.4}}{...} \approx 0.073 \tag{41}
\end{equation}

**注释**：可以看到，Bias项使得对角线附近的注意力权重增大，远距离的权重减小，产生了局部化效应。

### 12. 熵与注意力集中度

**信息熵的定义**：

对于概率分布$\{\alpha_1, \alpha_2, \ldots, \alpha_N\}$，其香农熵为：

\begin{equation}
H = -\sum_{n=1}^N \alpha_n \log \alpha_n \tag{42}
\end{equation}

**熵的范围**：

- 最小熵：$H_{\min} = 0$（当某个$\alpha_i = 1$，其余为0时）
- 最大熵：$H_{\max} = \log N$（当所有$\alpha_i = 1/N$时）

**归一化熵**：

\begin{equation}
\tilde{H} = \frac{H}{\log N} \in [0, 1] \tag{43}
\end{equation}

**注释**：归一化熵接近0表示注意力高度集中，接近1表示注意力均匀分布。

**Bias对熵的影响**：

不带Bias时，如果Attention得分相似，熵接近$\log N$。加入衰减的Bias后，Attention更集中于局部，熵减小。

**外推时的熵变化**：

- 训练长度$N=512$：$H_{\text{train}} \approx \log(512) \approx 6.2$
- 测试长度$N=2048$（无Bias）：$H_{\text{test}} \approx \log(2048) \approx 7.6$
- 测试长度$N=2048$（有Bias）：$H_{\text{test}} \approx 6.5$（接近训练时的熵）

**注释**：Bias项通过维持熵的相对稳定性，增强了外推能力。

### 13. 计算复杂度分析

**标准Attention的复杂度**：

时间复杂度：
\begin{equation}
O(N^2 d + N^2 d_v) = O(N^2 d) \tag{44}
\end{equation}

空间复杂度：
\begin{equation}
O(N^2) \text{（存储Attention矩阵）} \tag{45}
\end{equation}

**RoPE的额外开销**：

旋转变换：
\begin{equation}
O(Nd) \text{（对每个位置应用旋转）} \tag{46}
\end{equation}

**Bias项的额外开销**：

Bias加法：
\begin{equation}
O(Nd) \text{（可忽略不计）} \tag{47}
\end{equation}

**总复杂度**：

\begin{equation}
O(N^2 d + Nd) = O(N^2 d) \tag{48}
\end{equation}

**注释**：Bias项几乎不增加计算开销，是一种"免费"的改进。

### 14. 数值稳定性考虑

**RoPE中的三角函数计算**：

对于大的位置索引$m$，需要计算：

\begin{equation}
\cos(m\theta), \quad \sin(m\theta) \tag{49}
\end{equation}

**潜在问题**：

当$m$很大时，浮点数精度可能导致$m\theta$的计算误差，影响三角函数值。

**解决方案**：

使用周期性归约：

\begin{equation}
m\theta \mod 2\pi \tag{50}
\end{equation}

**Softmax的数值稳定实现**：

标准技巧：减去最大值

\begin{equation}
\alpha_{mn} = \frac{\exp(s_{mn} - \max_j s_{mj})}{\sum_j \exp(s_{mj} - \max_j s_{mj})} \tag{51}
\end{equation}

**Bias项对数值稳定性的影响**：

Bias项的衰减特性可以缓解极端值问题，减小$\max_j s_{mj} - \min_j s_{mj}$的范围。

### 15. 多头注意力中的Bias

**多头Attention的定义**：

\begin{equation}
\text{MultiHead}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\boldsymbol{W}^O \tag{52}
\end{equation}

其中：

\begin{equation}
\text{head}_i = \text{Attention}(\boldsymbol{Q}\boldsymbol{W}_i^Q, \boldsymbol{K}\boldsymbol{W}_i^K, \boldsymbol{V}\boldsymbol{W}_i^V) \tag{53}
\end{equation}

**每个头的独立Bias**：

每个头可以有独立的Bias向量$\boldsymbol{a}_i, \boldsymbol{b}_i$：

\begin{equation}
\tilde{\boldsymbol{q}}_m^{(i)} = \boldsymbol{\mathcal{R}}_m(\boldsymbol{q}_m^{(i)} + \boldsymbol{a}_i) \tag{54}
\end{equation}

\begin{equation}
\tilde{\boldsymbol{k}}_n^{(i)} = \boldsymbol{\mathcal{R}}_n(\boldsymbol{k}_n^{(i)} + \boldsymbol{b}_i) \tag{55}
\end{equation}

**不同头的Bias学习不同模式**：

不同的$\boldsymbol{a}_i, \boldsymbol{b}_i$可以学习不同的距离衰减模式，增强模型的表达能力。

### 16. 训练与推理的实现细节

**前向传播伪代码**：

```
def rope_with_bias(Q, K, V, positions, bias_q, bias_k):
    # 应用Bias
    Q = Q + bias_q  # shape: (batch, seq_len, d)
    K = K + bias_k

    # 计算旋转角度
    angles = positions[:, None] * thetas[None, :]  # thetas: (d/2,)

    # 应用旋转
    Q_rot = apply_rotation(Q, angles)
    K_rot = apply_rotation(K, angles)

    # 计算Attention
    scores = (Q_rot @ K_rot.T) / sqrt(d)
    attn_weights = softmax(scores)
    output = attn_weights @ V

    return output
```

**旋转变换的高效实现**：

将向量分为两部分进行旋转：

\begin{equation}
\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{pmatrix} \tag{56}
\end{equation}

**注释**：通过向量化操作，可以高效地并行计算所有位置的旋转。

### 17. 实验验证的理论解释

**实验设置回顾**：

- 训练长度：512
- 测试长度：512, 1024, 2048, 4096
- 模型：GAU-α
- 任务：语言模型（MLM准确率）

**结果分析**：

训练长度（512）：
- 无Bias：52.37%
- 有Bias：52.75%

**注释**：在训练长度上，两者性能相当，说明Bias不影响正常性能。

外推到1024：
- 无Bias：33.15%（下降19.22个百分点）
- 有Bias：50.99%（仅下降1.76个百分点）

**注释**：Bias显著提升了外推性能。

**理论解释**：

1. **局部化效应**：Bias产生的距离衰减使得Attention保持局部化
2. **熵稳定性**：Attention的熵在不同长度下保持相对稳定
3. **相对位置鲁棒性**：Bias加强了相对位置的作用，减少了绝对位置的影响

### 18. 与其他位置编码方法的比较

**ALiBi（Attention with Linear Biases）**：

直接在Attention得分上加线性偏置：

\begin{equation}
s_{mn} = \boldsymbol{q}_m^{\top}\boldsymbol{k}_n - \lambda |m-n| \tag{57}
\end{equation}

**优点**：简单，外推性好

**缺点**：需要额外的超参数$\lambda$，且线性衰减可能不是最优的

**T5的相对位置编码**：

使用可学习的相对位置嵌入：

\begin{equation}
s_{mn} = \boldsymbol{q}_m^{\top}\boldsymbol{k}_n + \boldsymbol{r}_{m-n} \tag{58}
\end{equation}

其中$\boldsymbol{r}$是可学习的相对位置向量。

**优点**：灵活，可以学习复杂的相对位置模式

**缺点**：外推性差（训练时未见过的相对距离没有对应的嵌入）

**RoPE + Bias的优势**：

1. **兼具相对位置编码和距离偏置的优点**
2. **无需额外超参数**（Bias在训练中学习）
3. **外推性好**（Bias的衰减模式自然延伸到长序列）

### 19. Bias向量的初始化策略

**标准初始化**：

通常使用零初始化或小随机值：

\begin{equation}
\boldsymbol{b} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2\boldsymbol{I}), \quad \sigma = 0.01 \tag{59}
\end{equation}

**Sandwich风格初始化**：

基于Sandwich的设计，初始化为：

\begin{equation}
\boldsymbol{b} = [\sqrt{\lambda}, 0, \sqrt{\lambda}, 0, \ldots]^{\top}, \quad \lambda = 0.1 \tag{60}
\end{equation}

**优点**：直接编码了期望的衰减模式

**缺点**：可能限制模型的学习能力

**自适应初始化**：

根据训练长度$N_{\text{train}}$调整：

\begin{equation}
\lambda = \frac{1}{\log N_{\text{train}}} \tag{61}
\end{equation}

**注释**：这使得Bias的影响范围与训练长度匹配。

### 20. 梯度流与训练动力学

**Bias的梯度**：

对于损失函数$\mathcal{L}$，Bias的梯度为：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} = \sum_{m,n} \frac{\partial \mathcal{L}}{\partial s_{mn}} \frac{\partial s_{mn}}{\partial \boldsymbol{b}} \tag{62}
\end{equation}

其中：

\begin{equation}
\frac{\partial s_{mn}}{\partial \boldsymbol{b}} = \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n(\boldsymbol{a} + \boldsymbol{q}_m) \tag{63}
\end{equation}

**梯度的相对位置依赖**：

由于旋转矩阵的相对位置性质，梯度主要依赖于$\boldsymbol{\mathcal{R}}_{n-m}$，使得Bias能够学习与相对距离相关的模式。

**训练稳定性**：

Bias的梯度与Query、Key的梯度在同一量级，不会导致训练不稳定。

### 21. 理论局限性与未来方向

**当前理论的局限**：

1. **各向同性假设不总是成立**：实际训练中，Query和Key可能不是完全各向同性的
2. **衰减模式的最优性未证明**：目前只是经验观察，缺乏理论最优性保证
3. **多层交互的复杂性**：Bias在多层Transformer中的累积效应尚未充分研究

**改进方向**：

1. **自适应Bias**：根据层深度和位置动态调整Bias
2. **可学习的衰减函数**：不限于余弦形式，学习更复杂的距离函数
3. **Bias的正则化**：通过正则化鼓励Bias学习特定的模式（如单调衰减）

### 22. 实践建议总结

**何时使用Bias**：

1. **需要长度外推的场景**（如长文本生成）
2. **使用RoPE作为位置编码的模型**
3. **模型在长序列上性能下降明显**

**实现检查清单**：

1. ✓ 确保Query和Key的Bias都启用
2. ✓ 使用合适的初始化（零初始化或小随机值）
3. ✓ 在多头Attention中，每个头有独立的Bias
4. ✓ 监控训练过程中Bias的范数和模式
5. ✓ 在不同测试长度下评估性能

**超参数调优**：

- Bias初始化的标准差：$\sigma \in [0.001, 0.1]$
- 学习率：与其他参数相同或稍小
- 是否共享Query和Key的Bias：通常不共享性能更好

**注释**：这些建议基于实验经验，具体应用中可能需要根据任务调整。

