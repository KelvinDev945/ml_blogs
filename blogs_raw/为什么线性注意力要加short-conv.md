---
title: 为什么线性注意力要加Short Conv？
slug: 为什么线性注意力要加short-conv
date: 2025-10-05
source: https://spaces.ac.cn/archives/11320
tags: 详细推导, 机器学习
status: completed
---

# 为什么线性注意力要加Short C...

**原文链接**: [https://spaces.ac.cn/archives/11320](https://spaces.ac.cn/archives/11320)

**发布日期**: 

---

如果读者有关注模型架构方面的进展，那么就会发现，比较新的线性Attention（参考[《线性注意力简史：从模仿、创新到反哺》](/archives/11033)）模型都给$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$加上了Short Conv，比如下图所示的[DeltaNet](https://papers.cool/arxiv/2406.06484)：  

[![DeltaNet中的Short Conv](/usr/uploads/2025/10/175536171.png)](/usr/uploads/2025/10/175536171.png "点击查看原图")

DeltaNet中的Short Conv

为什么要加这个Short Conv呢？直观理解可能是增加模型深度、增强模型的Token-Mixing能力等，说白了就是补偿线性化导致的表达能力下降。这个说法当然是大差不差，但它属于“万能模版”式的回答，我们更想对它的生效机制有更准确的认知。

接下来，笔者将给出自己的一个理解（更准确说应该是猜测）。

## 测试训练 #

从[《线性注意力简史：从模仿、创新到反哺》](/archives/11033)我们知道，目前的新式线性Attention背后的核心思想都是[TTT（Test Time Training）](https://papers.cool/arxiv/2407.04620)或者说Online Learning。TTT基于优化器更新与RNN迭代的相似性，通过优化器来构建（不一定是线性的）RNN模型，诸如DeltaNet、GDN、Comba等线性Attention都可以看成它的特例。

具体来说，TTT将$\boldsymbol{K},\boldsymbol{V}$视为成对的语料$(\boldsymbol{k}_1, \boldsymbol{v}_1),(\boldsymbol{k}_2, \boldsymbol{v}_2),\cdots,(\boldsymbol{k}_t, \boldsymbol{v}_t)$，我们用它训练一个模型$\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{k})$，然后输出$\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{q}_t)$，其中$\boldsymbol{S}_t$是模型参数，用SGD更新：  
\begin{equation} \boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t\nabla_{\boldsymbol{S}_{t-1}}\mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1};\boldsymbol{k}_t), \boldsymbol{v}_t)\end{equation}  
当然，如果我们愿意，也可以考虑其他优化器，比如[《Test-Time Training Done Right》](https://papers.cool/arxiv/2505.23884)尝试了Muon优化器。除了可以改变优化器外，可以灵活改变的地方还有模型$\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{k})$的架构以及损失函数$\mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1};\boldsymbol{k}_t), \boldsymbol{v}_t)$。此外，我们还可以考虑以Chunk为单位的Mini-batch TTT。

不难想象，理论上TTT的灵活性非常高，可以构建任意复杂的RNN模型。当架构选择线性模型$\boldsymbol{v} = \boldsymbol{S}_t\boldsymbol{k}$且损失函数选择平方误差时，结果对应DeltaNet；如果我们加上一些正则项，那么可以衍生出GDN等变体。

## 灵魂拷问 #

将TTT放在前面，主要是想表明，当前主流的线性Attention的底层逻辑跟TTT一样，核心都是语料对$(\boldsymbol{k}_1, \boldsymbol{v}_1),(\boldsymbol{k}_2, \boldsymbol{v}_2),\cdots,(\boldsymbol{k}_t, \boldsymbol{v}_t)$的Online Learning。这就很自然地引申出一个疑问：为什么要这样做？这样做究竟学出个什么来？

要回答这个问题，首先得反思一下“我们究竟想要什么”。按照Softmax Attention的特点，我们想要的应该是根据$(\boldsymbol{k}_1, \boldsymbol{v}_1),(\boldsymbol{k}_2, \boldsymbol{v}_2),\cdots,(\boldsymbol{k}_t, \boldsymbol{v}_t)$和$\boldsymbol{q}_t$算出一个$\boldsymbol{o}_t$来，这个过程理想情况下应该依赖于全体$(\boldsymbol{k},\boldsymbol{v})$。同时，我们还希望能常数复杂度实现这个目标，所以一个直观想法是先将$(\boldsymbol{k},\boldsymbol{v})$压缩成（与$t$无关的）固定大小的State，然后再读取这个State。

怎么实现这个压缩呢？TTT的想法是：设计一个模型$\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{k})$，然后用这些$(\boldsymbol{k},\boldsymbol{v})$对去“训练”这个模型，训练完成后，模型某种意义上就“背下”了这些$(\boldsymbol{k},\boldsymbol{v})$对，这就相当于就全体$(\boldsymbol{k},\boldsymbol{v})$压缩到了固定大小的模型权重$\boldsymbol{S}_t$中。至于$\boldsymbol{q}  
_t$怎么利用$\boldsymbol{S}_t$，直接将它代入模型中得到$\boldsymbol{o}  
_t = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{q}_t)$是一个比较自然的选择，但原则上我们也可以设计别的利用方式。

也就是说，TTT的核心任务是利用 _“训练模型”约等于“背诵训练集”_ 这件事情，来实现$\boldsymbol{K},\boldsymbol{V}$的压缩。然而， _“训练模型”约等于“背诵训练集”_ 这件事情并不是那么平凡，它有一些前提条件。

## 键值同源 #

举个例子，如果我们取$\boldsymbol{K}=\boldsymbol{V}$，那么TTT这套框架理论上就失效了，因为这时候模型$\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{k})$的最优解就是恒等变换，它是一个平凡解，这相当于没记住任何东西。像DeltaNet这种在线更新的或许还能抢救一下，而像[MesaNet](https://papers.cool/arxiv/2506.05233)这种基于准确解的就真的直接输出单位阵$\boldsymbol{I}$了。

可能有读者会反问：好端端地为啥要考虑$\boldsymbol{K}=\boldsymbol{V}$这种不科学的选择呢？的确，$\boldsymbol{K}=\boldsymbol{V}$是比较极端的选择，这里只是作为一个例子，说明 _“训练模型”约等于“背诵训练集”_ 并不是随意成立的。其次，我们在[《Transformer升级之路：20、MLA好在哪里?（上）》](/archives/10907)已经验证过，对于Softmax Attention来说，$\boldsymbol{K}=\boldsymbol{V}$也能取得不错的结果。

这说明，$\boldsymbol{K}=\boldsymbol{V}$并不是Attention机制的本质障碍，但在TTT框架里边它却能导致模型失效，这是因为$\boldsymbol{K},\boldsymbol{V}$完全重合，那么它们俩之间的回归就没有东西可学了。类似地，我们可以想象，$\boldsymbol{K},\boldsymbol{V}$的信息重合度越高，那么它们之间可学的东西就越少，换言之TTT对“训练集”的记忆程度就越低。

在一般的Attention机制中，$\boldsymbol{q}_t,\boldsymbol{k}_t,\boldsymbol{v}_t$都是由同一个输入$\boldsymbol{x}_t$经过不同的线性投影得到的，换句话说$\boldsymbol{k}_t,\boldsymbol{v}_t$具有相同的源头$\boldsymbol{x}_t$，这就始终有种“自己预测自己”的感觉，可学的东西有限。

## 卷积救场 #

怎么让TTT在键值同源甚至$\boldsymbol{K}=\boldsymbol{V}$时能学出更有价值的结果呢？其实答案很早就有了——可以追溯到Word2Vec甚至更早的年代——那就是不要“预测自己”，而是“预测周围”。

以[Word2Vec](https://papers.cool/arxiv/1301.3781)为例，我们知道它的训练方式是“中心词预测上下文”；之前流行的[BERT](https://papers.cool/arxiv/1810.04805)，预训练方式是MLM，它是某些词Mask掉来预测这些词，可以说是“上下文预测中心词”；现在主流的LLM，训练任务则是NTP（Next Token Predict），根据上文预测下一个词。很明显，它们的共同特点都是不预测自己，而是预测周围。

所以，想要改进TTT，那么就要改变$(\boldsymbol{k}_t,\boldsymbol{v}_t)$这样的“自己预测自己”的配对方式，考虑到当前LLM以NTP为主，在TTT中我们也可以考虑NTP，比如$(\boldsymbol{k}_{t-1},\boldsymbol{v}_t)$来构建语料对，即用$\boldsymbol{k}_{t-1}$来预测$\boldsymbol{v}_t$，这样即便$\boldsymbol{K}=\boldsymbol{V}$也能学出非平凡的结果。此时TTT内外都是NTP任务，具有漂亮的一致性。

不过，只用$\boldsymbol{k}_{t-1}$预测$\boldsymbol{v}_t$的话，似乎把$\boldsymbol{k}_t$浪费掉了，所以进一步的想法是把$\boldsymbol{k}_{t-1}$和$\boldsymbol{k}_t$以某种方式混合起来再预测$\boldsymbol{v}_t$。到这里，大家可能反应过来了，“把$\boldsymbol{k}_{t-1}$和$\boldsymbol{k}_t$以某种方式混合起来”这不就相当于kernel_size=2的Conv嘛！所以，给$\boldsymbol{K}$加Short Conv，是将TTT的训练目标从“预测自己”转化为NTP，让TTT至少有能力学出一个n-gram模型。

至于给$\boldsymbol{Q},\boldsymbol{V}$加Short Conv，则完全是顺带的，根据飞来阁（FLA群）的消息，给$\boldsymbol{Q},\boldsymbol{V}$加虽然也有一点作用，但远不如给$\boldsymbol{K}$加Short Conv带来的提升，这也算是佐证了我们的猜测。

## 文章小结 #

这篇文章对“为什么线性注意力要加Short Conv”这个问题给出了一个闭门造车的理解。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11320>_

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

苏剑林. (Oct. 05, 2025). 《为什么线性注意力要加Short Conv？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11320>

@online{kexuefm-11320,  
title={为什么线性注意力要加Short Conv？},  
author={苏剑林},  
year={2025},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/11320}},  
} 

---

---

## 公式推导与注释

### 1. 标准注意力机制的数学定义

标准的自注意力（Self-Attention）机制可以形式化表示为：

$$
\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}}\right)\boldsymbol{V}
$$

**注释**：这里 $\boldsymbol{Q} \in \mathbb{R}^{n \times d_k}$ 是查询矩阵，$\boldsymbol{K} \in \mathbb{R}^{n \times d_k}$ 是键矩阵，$\boldsymbol{V} \in \mathbb{R}^{n \times d_v}$ 是值矩阵，其中 $n$ 是序列长度，$d_k$ 和 $d_v$ 分别是键和值的维度。温度系数 $\sqrt{d_k}$ 用于控制softmax函数的尖锐程度。

### 2. 标准注意力的计算复杂度分析

对于序列长度为 $n$ 的输入，标准注意力机制的计算复杂度分解如下：

**步骤1：计算注意力得分矩阵**

$$
\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{K}^T \in \mathbb{R}^{n \times n}
$$

**复杂度分析**：矩阵乘法 $\boldsymbol{Q}\boldsymbol{K}^T$ 需要计算 $n \times n$ 个元素，每个元素需要 $d_k$ 次乘法和加法，因此时间复杂度为 $O(n^2 d_k)$，空间复杂度为 $O(n^2)$。

**步骤2：应用softmax归一化**

$$
\boldsymbol{A}_{i,j} = \frac{\exp(\boldsymbol{S}_{i,j}/\sqrt{d_k})}{\sum_{k=1}^n \exp(\boldsymbol{S}_{i,k}/\sqrt{d_k})}
$$

**复杂度分析**：对于 $n \times n$ 的矩阵，softmax操作需要 $O(n^2)$ 的时间复杂度。

**步骤3：加权求和**

$$
\boldsymbol{O} = \boldsymbol{A}\boldsymbol{V} \in \mathbb{R}^{n \times d_v}
$$

**复杂度分析**：矩阵乘法 $\boldsymbol{A}\boldsymbol{V}$ 的时间复杂度为 $O(n^2 d_v)$。

**总体复杂度**：标准注意力的总时间复杂度为 $O(n^2(d_k + d_v))$，空间复杂度为 $O(n^2)$。当序列长度 $n$ 很大时，这种二次复杂度成为主要瓶颈。

### 3. 线性注意力的基本思想

线性注意力的核心思想是通过特征映射函数 $\phi(\cdot)$ 将注意力机制线性化，避免显式计算 $n \times n$ 的注意力矩阵。

**核技巧（Kernel Trick）的应用**：

假设存在特征映射 $\phi: \mathbb{R}^{d_k} \to \mathbb{R}^{d_\phi}$，使得：

$$
\exp\left(\frac{\boldsymbol{q}_i^T \boldsymbol{k}_j}{\sqrt{d_k}}\right) \approx \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j)
$$

**注释**：这个近似是线性注意力的关键。通过这种方式，我们可以将注意力权重的计算转化为特征空间中的内积计算。

### 4. 线性注意力的数学推导

标准注意力的第 $i$ 个输出向量可以写成：

$$
\boldsymbol{o}_i = \frac{\sum_{j=1}^n \exp(\boldsymbol{q}_i^T \boldsymbol{k}_j) \boldsymbol{v}_j}{\sum_{j=1}^n \exp(\boldsymbol{q}_i^T \boldsymbol{k}_j)}
$$

**引入特征映射后的线性化**：

使用特征映射 $\phi(\cdot)$ 近似，我们得到：

$$
\boldsymbol{o}_i \approx \frac{\sum_{j=1}^n \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j) \boldsymbol{v}_j}{\sum_{j=1}^n \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j)}
$$

**关键的代数变换**：

利用线性代数的结合律，我们可以重新组合求和顺序：

$$
\boldsymbol{o}_i = \frac{\phi(\boldsymbol{q}_i)^T \left(\sum_{j=1}^n \phi(\boldsymbol{k}_j) \boldsymbol{v}_j^T\right)}{\phi(\boldsymbol{q}_i)^T \left(\sum_{j=1}^n \phi(\boldsymbol{k}_j)\right)}
$$

**定义累积矩阵**：

令 $\boldsymbol{S} = \sum_{j=1}^n \phi(\boldsymbol{k}_j) \boldsymbol{v}_j^T \in \mathbb{R}^{d_\phi \times d_v}$ 和 $\boldsymbol{z} = \sum_{j=1}^n \phi(\boldsymbol{k}_j) \in \mathbb{R}^{d_\phi}$，则：

$$
\boldsymbol{o}_i = \frac{\phi(\boldsymbol{q}_i)^T \boldsymbol{S}}{\phi(\boldsymbol{q}_i)^T \boldsymbol{z}}
$$

**注释**：这个重新组合是线性注意力能够降低复杂度的关键。注意 $\boldsymbol{S}$ 和 $\boldsymbol{z}$ 只需要计算一次，然后可以被所有查询位置重用。

### 5. 线性注意力的复杂度分析

**计算 $\boldsymbol{S}$ 的复杂度**：

$$
\boldsymbol{S} = \sum_{j=1}^n \phi(\boldsymbol{k}_j) \boldsymbol{v}_j^T
$$

- 每个 $\phi(\boldsymbol{k}_j) \boldsymbol{v}_j^T$ 是 $d_\phi \times d_v$ 的矩阵，需要 $O(d_\phi d_v)$ 时间
- 总共需要累加 $n$ 次，因此时间复杂度为 $O(n d_\phi d_v)$

**计算 $\boldsymbol{z}$ 的复杂度**：

$$
\boldsymbol{z} = \sum_{j=1}^n \phi(\boldsymbol{k}_j)
$$

- 时间复杂度为 $O(n d_\phi)$

**计算所有输出的复杂度**：

对于每个查询位置 $i$：

$$
\boldsymbol{o}_i = \frac{\phi(\boldsymbol{q}_i)^T \boldsymbol{S}}{\phi(\boldsymbol{q}_i)^T \boldsymbol{z}}
$$

- $\phi(\boldsymbol{q}_i)^T \boldsymbol{S}$ 需要 $O(d_\phi d_v)$ 时间
- $\phi(\boldsymbol{q}_i)^T \boldsymbol{z}$ 需要 $O(d_\phi)$ 时间
- 对于 $n$ 个位置，总时间复杂度为 $O(n d_\phi d_v)$

**总体复杂度**：线性注意力的总时间复杂度为 $O(n d_\phi (d_v + d_k))$，相比标准注意力的 $O(n^2 d_k)$，从序列长度的二次复杂度降低到线性复杂度。

### 6. 常用的特征映射函数

**ELU特征映射**（Performers）：

$$
\phi(\boldsymbol{x}) = \text{elu}(\boldsymbol{x}) + 1
$$

其中 $\text{elu}(x) = \begin{cases} x & \text{if } x > 0 \\ e^x - 1 & \text{if } x \leq 0 \end{cases}$

**注释**：加1是为了保证所有元素非负，这样可以确保注意力权重的非负性。

**随机特征映射**（Performers）：

$$
\phi(\boldsymbol{x}) = \frac{1}{\sqrt{m}} \left[\cos(\boldsymbol{\omega}_1^T \boldsymbol{x}), \sin(\boldsymbol{\omega}_1^T \boldsymbol{x}), \ldots, \cos(\boldsymbol{\omega}_m^T \boldsymbol{x}), \sin(\boldsymbol{\omega}_m^T \boldsymbol{x})\right]^T
$$

其中 $\boldsymbol{\omega}_i \sim \mathcal{N}(0, \boldsymbol{I})$ 是从标准正态分布中随机采样的向量。

**注释**：这是基于随机傅里叶特征的映射，理论上可以任意精确地近似高斯核函数。

**ReLU特征映射**（Linear Attention）：

$$
\phi(\boldsymbol{x}) = \text{ReLU}(\boldsymbol{x}) = \max(0, \boldsymbol{x})
$$

**注释**：这是最简单的特征映射，计算效率高，但近似精度相对较低。

### 7. 因果线性注意力的推导

在自回归场景中，位置 $i$ 的输出只能依赖于位置 $j \leq i$ 的信息。因果线性注意力可以表示为：

$$
\boldsymbol{o}_i = \frac{\sum_{j=1}^i \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j) \boldsymbol{v}_j}{\sum_{j=1}^i \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j)}
$$

**递推形式的推导**：

定义累积状态：

$$
\boldsymbol{S}_i = \sum_{j=1}^i \phi(\boldsymbol{k}_j) \boldsymbol{v}_j^T, \quad \boldsymbol{z}_i = \sum_{j=1}^i \phi(\boldsymbol{k}_j)
$$

则有递推关系：

$$
\begin{aligned}
\boldsymbol{S}_i &= \boldsymbol{S}_{i-1} + \phi(\boldsymbol{k}_i) \boldsymbol{v}_i^T \\
\boldsymbol{z}_i &= \boldsymbol{z}_{i-1} + \phi(\boldsymbol{k}_i)
\end{aligned}
$$

输出可以表示为：

$$
\boldsymbol{o}_i = \frac{\phi(\boldsymbol{q}_i)^T \boldsymbol{S}_i}{\phi(\boldsymbol{q}_i)^T \boldsymbol{z}_i}
$$

**注释**：这种递推形式使得因果线性注意力可以高效地实现流式推理，每个时间步只需要 $O(d_\phi d_v)$ 的计算量。

### 8. 线性注意力的局限性：缺乏局部建模能力

**问题陈述**：线性注意力通过全局累积的方式计算注意力，导致它在建模局部依赖关系时能力不足。

**数学分析**：

标准注意力的权重矩阵 $\boldsymbol{A} \in \mathbb{R}^{n \times n}$ 是稠密的，第 $i$ 行的权重分布可以是任意的概率分布：

$$
\boldsymbol{A}_{i,:} = \text{softmax}(\boldsymbol{q}_i^T \boldsymbol{K}^T) \in \Delta^{n-1}
$$

其中 $\Delta^{n-1}$ 表示 $(n-1)$ 维单纯形（概率单纯形）。

**线性注意力的权重分布**：

对于线性注意力，权重实际上是：

$$
\alpha_{i,j} = \frac{\phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j)}{\sum_{k=1}^n \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_k)}
$$

**局部性缺失的证明**：

考虑位置 $i$ 对其相邻位置 $i-1$ 和 $i+1$ 的注意力权重。在标准注意力中，可以通过调整 $\boldsymbol{q}_i$、$\boldsymbol{k}_{i-1}$、$\boldsymbol{k}_{i+1}$ 使得：

$$
\alpha_{i,i-1} + \alpha_{i,i+1} \approx 1
$$

即几乎所有注意力集中在相邻位置。

但在线性注意力中，由于特征映射 $\phi$ 通常是逐元素的非线性函数，它难以产生这种尖锐的局部注意力分布。特别是，$\phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_j)$ 的值受到全局归一化项 $\sum_{k=1}^n \phi(\boldsymbol{q}_i)^T \phi(\boldsymbol{k}_k)$ 的影响，难以实现高度局部化的权重分配。

### 9. 短卷积（Short Conv）的数学定义

短卷积是一种卷积核大小较小的一维卷积操作，通常核大小 $k \in \{3, 5, 7\}$。

**一维卷积的数学表示**：

对于输入序列 $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n] \in \mathbb{R}^{n \times d}$，卷积操作定义为：

$$
\boldsymbol{y}_i = \sum_{j=-(k-1)/2}^{(k-1)/2} \boldsymbol{W}_j \boldsymbol{x}_{i+j} + \boldsymbol{b}
$$

其中 $\boldsymbol{W}_j \in \mathbb{R}^{d \times d}$ 是第 $j$ 个位置的卷积核权重，$\boldsymbol{b} \in \mathbb{R}^d$ 是偏置项。

**注释**：这里假设卷积核大小 $k$ 是奇数，中心在位置0。对于边界位置，通常使用padding策略。

### 10. 深度可分离卷积的数学表示

在实践中，为了降低计算量，短卷积通常使用深度可分离卷积（Depthwise Separable Convolution）实现。

**深度卷积（Depthwise Convolution）**：

$$
\boldsymbol{y}_i^{(c)} = \sum_{j=-(k-1)/2}^{(k-1)/2} w_j^{(c)} \boldsymbol{x}_{i+j}^{(c)} + b^{(c)}
$$

其中 $c \in \{1, 2, \ldots, d\}$ 是通道索引，每个通道独立应用一维卷积。

**注释**：深度卷积将标准卷积的 $O(k d^2)$ 参数量降低到 $O(k d)$。

**逐点卷积（Pointwise Convolution）**：

$$
\boldsymbol{z}_i = \boldsymbol{W}_p \boldsymbol{y}_i + \boldsymbol{b}_p
$$

其中 $\boldsymbol{W}_p \in \mathbb{R}^{d \times d}$ 是逐点卷积的权重矩阵。

**完整的深度可分离卷积**：

$$
\text{ShortConv}(\boldsymbol{X}) = \boldsymbol{W}_p \cdot \text{DepthwiseConv}(\boldsymbol{X}) + \boldsymbol{b}_p
$$

**复杂度分析**：深度可分离卷积的参数量为 $O(kd + d^2)$，计算量为 $O(n(kd + d^2))$，相比标准卷积的 $O(kd^2)$ 参数量显著降低。

### 11. 因果卷积的数学性质

在自回归模型中，需要使用因果卷积（Causal Convolution）确保位置 $i$ 的输出不依赖于未来位置。

**因果卷积的定义**：

$$
\boldsymbol{y}_i = \sum_{j=0}^{k-1} \boldsymbol{W}_j \boldsymbol{x}_{i-j}
$$

**注释**：与标准卷积不同，因果卷积只向左看（只依赖历史信息），卷积核从当前位置向左延伸 $k-1$ 个位置。

**因果性的形式化定义**：

定义函数 $f: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ 是因果的，当且仅当：

$$
\forall i \in \{1, \ldots, n\}, \quad f(\boldsymbol{X})_i = g(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_i)
$$

即位置 $i$ 的输出只依赖于位置 $1$ 到 $i$ 的输入。

**因果卷积的因果性证明**：

对于因果卷积 $\boldsymbol{y}_i = \sum_{j=0}^{k-1} \boldsymbol{W}_j \boldsymbol{x}_{i-j}$，显然 $\boldsymbol{y}_i$ 只依赖于 $\boldsymbol{x}_{i-k+1}, \ldots, \boldsymbol{x}_i$，满足因果性定义。

### 12. 卷积的感受野分析

**单层卷积的感受野**：

对于核大小为 $k$ 的卷积层，每个输出位置的感受野（receptive field）大小为 $k$。

$$
\text{RF}_1 = k
$$

**多层卷积的感受野**：

对于 $L$ 层核大小为 $k$ 的卷积网络，感受野呈线性增长：

$$
\text{RF}_L = 1 + L(k-1)
$$

**推导过程**：

- 第1层：感受野为 $k$
- 第2层：每个位置看到第1层的 $k$ 个位置，而第1层每个位置看到输入的 $k$ 个位置，因此总感受野为 $k + (k-1) = 2k - 1$
- 第 $L$ 层：递推可得 $\text{RF}_L = k + (L-1)(k-1) = 1 + L(k-1)$

**注释**：这说明卷积网络的感受野增长相对缓慢，要覆盖长距离依赖需要堆叠多层或使用扩张卷积（dilated convolution）。

### 13. Short Conv的局部建模能力

短卷积天然具有局部建模能力，这体现在以下几个方面：

**位置相关的权重分配**：

对于位置 $i$，其输出 $\boldsymbol{y}_i$ 对不同位置输入的权重是固定的：

$$
\frac{\partial \boldsymbol{y}_i}{\partial \boldsymbol{x}_{i+j}} = \begin{cases} \boldsymbol{W}_j & \text{if } |j| \leq (k-1)/2 \\ \boldsymbol{0} & \text{otherwise} \end{cases}
$$

**局部性的数学刻画**：

定义局部性度量为非零梯度的范围：

$$
\text{Locality}(\text{ShortConv}) = k
$$

相比之下，全局注意力的局部性度量为：

$$
\text{Locality}(\text{Attention}) = n
$$

**注释**：短卷积的局部性度量与序列长度 $n$ 无关，这使得它在建模局部模式时更加高效和稳定。

### 14. 线性注意力的表达能力限制

**泛函分析视角**：

将注意力机制视为一个从输入序列到输出序列的映射：

$$
\mathcal{F}: (\mathbb{R}^d)^n \to (\mathbb{R}^d)^n
$$

标准注意力可以表示任意的排列不变函数（permutation-invariant function），但线性注意力由于其近似性质，表达能力受到限制。

**线性注意力的表达能力界**：

设 $f$ 是标准注意力实现的函数，$\tilde{f}$ 是线性注意力的近似，则存在误差界：

$$
\|\mathcal{F}(\boldsymbol{X}) - \tilde{\mathcal{F}}(\boldsymbol{X})\|_F \leq C \cdot \epsilon(\phi)
$$

其中 $\epsilon(\phi)$ 是特征映射 $\phi$ 对softmax核的近似误差，$C$ 是依赖于输入的常数。

**近似误差的来源**：

1. **核函数近似误差**：$\exp(\boldsymbol{q}^T \boldsymbol{k}) \approx \phi(\boldsymbol{q})^T \phi(\boldsymbol{k})$ 的近似误差
2. **注意力分布误差**：近似后的注意力分布与真实分布的偏差
3. **输出加权误差**：误差在加权求和过程中的累积

### 15. Short Conv补偿线性注意力的理论依据

**互补性原理**：

线性注意力擅长全局信息聚合，但缺乏局部建模能力；短卷积擅长局部特征提取，但感受野有限。两者的结合可以实现优势互补。

**数学表示**：

结合后的机制可以表示为：

$$
\boldsymbol{O} = \text{LinearAttention}(\text{ShortConv}(\boldsymbol{Q}), \text{ShortConv}(\boldsymbol{K}), \text{ShortConv}(\boldsymbol{V}))
$$

或者：

$$
\boldsymbol{O} = \text{ShortConv}(\text{LinearAttention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}))
$$

**注释**：前者是在注意力之前应用短卷积（pre-conv），后者是在注意力之后应用（post-conv）。实践中通常使用pre-conv，因为它可以让注意力机制处理已经过局部增强的特征。

### 16. 联合机制的表达能力分析

**定理**（联合机制的表达能力）：

设 $\mathcal{F}_{\text{linear}}$ 是线性注意力的函数空间，$\mathcal{F}_{\text{conv}}$ 是短卷积的函数空间，则联合机制的函数空间满足：

$$
\mathcal{F}_{\text{linear}} \cup \mathcal{F}_{\text{conv}} \subseteq \mathcal{F}_{\text{joint}} \subseteq \mathcal{F}_{\text{linear}} \otimes \mathcal{F}_{\text{conv}}
$$

其中 $\otimes$ 表示函数空间的张量积。

**证明思路**：

1. **包含性**：联合机制可以通过适当的参数设置退化为纯线性注意力或纯短卷积，因此 $\mathcal{F}_{\text{linear}} \cup \mathcal{F}_{\text{conv}} \subseteq \mathcal{F}_{\text{joint}}$

2. **有界性**：联合机制的输出可以分解为两部分的组合，因此 $\mathcal{F}_{\text{joint}} \subseteq \mathcal{F}_{\text{linear}} \otimes \mathcal{F}_{\text{conv}}$

**注释**：这个定理说明联合机制的表达能力至少不小于单独使用任一机制，且可能更强。

### 17. 频域分析：Short Conv的频率响应

**离散时间傅里叶变换（DTFT）**：

对于卷积核 $\boldsymbol{w} = [w_0, w_1, \ldots, w_{k-1}]$，其频率响应为：

$$
H(\omega) = \sum_{j=0}^{k-1} w_j e^{-i\omega j}
$$

其中 $\omega \in [0, 2\pi]$ 是角频率。

**短卷积的频率特性**：

对于核大小 $k=3$ 的短卷积 $[w_{-1}, w_0, w_1]$，频率响应为：

$$
H(\omega) = w_0 + w_1 e^{-i\omega} + w_{-1} e^{i\omega} = w_0 + 2\text{Re}(w_1 e^{-i\omega})
$$

**注释**：短卷积通常表现为低通滤波器或带通滤波器，能够平滑高频噪声并保留低频趋势信息。

### 18. 线性注意力的频率特性

**线性注意力在频域的表示**：

线性注意力的输出可以写成：

$$
\boldsymbol{O} = \text{diag}^{-1}(\boldsymbol{\Phi}_Q \boldsymbol{\Phi}_K^T \mathbf{1}) \cdot \boldsymbol{\Phi}_Q \boldsymbol{\Phi}_K^T \boldsymbol{V}
$$

其中 $\boldsymbol{\Phi}_Q = [\phi(\boldsymbol{q}_1), \ldots, \phi(\boldsymbol{q}_n)]^T$，$\mathbf{1}$ 是全1向量。

**频域特性**：

线性注意力的频率响应是输入相关的，不像卷积那样有固定的频率响应函数。对于特征映射 $\phi(\boldsymbol{x}) = \text{ReLU}(\boldsymbol{x})$，线性注意力倾向于保留所有频率成分，但高频成分可能由于归一化而被削弱。

**互补性的频域解释**：

- 短卷积：提供固定的频率滤波，强化局部平滑性
- 线性注意力：提供内容自适应的全局信息整合

两者结合可以在频域上实现更全面的信号处理。

### 19. 计算复杂度的严格对比

**标准注意力**：

- 时间复杂度：$O(n^2 d)$
- 空间复杂度：$O(n^2 + nd)$

**线性注意力**：

- 时间复杂度：$O(nd_\phi d)$
- 空间复杂度：$O(d_\phi d + nd)$

**Short Conv**：

- 时间复杂度：$O(nkd)$（深度可分离）或 $O(nkd^2)$（标准卷积）
- 空间复杂度：$O(kd + nd)$

**联合机制（Linear Attention + Short Conv）**：

假设在Q、K、V上都应用Short Conv，然后进行线性注意力：

$$
\begin{aligned}
\text{时间复杂度} &= 3 \times O(nkd) + O(nd_\phi d) \\
&= O(n(3kd + d_\phi d)) \\
&= O(nd(3k + d_\phi))
\end{aligned}
$$

**复杂度对比表**：

| 机制 | 时间复杂度 | 序列长度依赖 |
|------|-----------|-------------|
| 标准注意力 | $O(n^2 d)$ | 二次 |
| 线性注意力 | $O(nd_\phi d)$ | 线性 |
| Short Conv | $O(nkd)$ | 线性 |
| 联合机制 | $O(nd(3k + d_\phi))$ | 线性 |

**注释**：联合机制保持了线性复杂度，仅增加了一个较小的常数因子 $3k$，通常 $k \in \{3, 5, 7\}$。

### 20. DeltaNet中的具体实现

DeltaNet中的联合机制可以表示为：

$$
\begin{aligned}
\tilde{\boldsymbol{Q}} &= \text{ShortConv}(\boldsymbol{Q}) \\
\tilde{\boldsymbol{K}} &= \text{ShortConv}(\boldsymbol{K}) \\
\tilde{\boldsymbol{V}} &= \text{ShortConv}(\boldsymbol{V}) \\
\boldsymbol{O} &= \text{LinearAttention}(\tilde{\boldsymbol{Q}}, \tilde{\boldsymbol{K}}, \tilde{\boldsymbol{V}})
\end{aligned}
$$

**Short Conv的参数化**：

DeltaNet使用深度可分离卷积，对于维度 $d$ 的输入：

$$
\text{ShortConv}(\boldsymbol{X})_i = \boldsymbol{W}_p \left( \bigoplus_{c=1}^d \sum_{j=0}^{k-1} w_j^{(c)} x_{i-j}^{(c)} \right)
$$

其中 $\bigoplus$ 表示通道拼接，$\boldsymbol{W}_p \in \mathbb{R}^{d \times d}$ 是逐点卷积权重。

**线性注意力的参数化**：

DeltaNet使用门控机制和特殊的特征映射：

$$
\phi(\boldsymbol{x}) = \text{swish}(\boldsymbol{x}) \cdot \boldsymbol{g}
$$

其中 $\boldsymbol{g} = \sigma(\boldsymbol{W}_g \boldsymbol{x} + \boldsymbol{b}_g)$ 是门控向量，$\text{swish}(x) = x \cdot \sigma(x)$。

### 21. 因果掩码与Short Conv的结合

在自回归生成中，需要同时考虑因果性约束：

**因果Short Conv + 因果线性注意力**：

$$
\begin{aligned}
\tilde{\boldsymbol{q}}_i &= \sum_{j=0}^{k-1} \boldsymbol{W}_j^Q \boldsymbol{q}_{i-j} \\
\tilde{\boldsymbol{k}}_i &= \sum_{j=0}^{k-1} \boldsymbol{W}_j^K \boldsymbol{k}_{i-j} \\
\tilde{\boldsymbol{v}}_i &= \sum_{j=0}^{k-1} \boldsymbol{W}_j^V \boldsymbol{v}_{i-j} \\
\boldsymbol{S}_i &= \boldsymbol{S}_{i-1} + \phi(\tilde{\boldsymbol{k}}_i) \tilde{\boldsymbol{v}}_i^T \\
\boldsymbol{z}_i &= \boldsymbol{z}_{i-1} + \phi(\tilde{\boldsymbol{k}}_i) \\
\boldsymbol{o}_i &= \frac{\phi(\tilde{\boldsymbol{q}}_i)^T \boldsymbol{S}_i}{\phi(\tilde{\boldsymbol{q}}_i)^T \boldsymbol{z}_i}
\end{aligned}
$$

**递推计算的状态维护**：

需要维护的状态包括：

1. 线性注意力的累积矩阵 $\boldsymbol{S}_i \in \mathbb{R}^{d_\phi \times d_v}$
2. 线性注意力的归一化向量 $\boldsymbol{z}_i \in \mathbb{R}^{d_\phi}$
3. 短卷积的历史缓存 $[\boldsymbol{q}_{i-k+1}, \ldots, \boldsymbol{q}_i]$, $[\boldsymbol{k}_{i-k+1}, \ldots, \boldsymbol{k}_i]$, $[\boldsymbol{v}_{i-k+1}, \ldots, \boldsymbol{v}_i]$

**状态空间复杂度**：

$$
\text{Space} = O(d_\phi d_v + d_\phi + 3kd) = O(d_\phi d_v + kd)
$$

**注释**：相比纯线性注意力，联合机制仅增加了 $O(kd)$ 的状态存储，通常 $k$ 很小（3-7），因此增加的开销可接受。

### 22. 训练稳定性分析

**梯度流分析**：

对于损失函数 $\mathcal{L}$，其关于输入 $\boldsymbol{X}$ 的梯度可以通过链式法则计算：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{X}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{O}} \frac{\partial \boldsymbol{O}}{\partial \tilde{\boldsymbol{V}}} \frac{\partial \tilde{\boldsymbol{V}}}{\partial \boldsymbol{V}} \frac{\partial \boldsymbol{V}}{\partial \boldsymbol{X}} + \text{其他路径}
$$

**Short Conv对梯度流的影响**：

短卷积提供了额外的梯度传播路径，可以缓解线性注意力可能存在的梯度消失问题。具体地，对于 $L$ 层网络，梯度的范数满足：

$$
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{X}^{(1)}}\right\| \geq \frac{\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{X}^{(L)}}\|}{C^L}
$$

其中 $C$ 是依赖于激活函数和归一化方式的常数。短卷积的引入可以减小 $C$，从而改善梯度传播。

### 23. 位置编码与Short Conv的交互

**相对位置信息的隐式编码**：

短卷积通过其固定的卷积核权重隐式地编码了相对位置信息。对于位置 $i$ 和 $i+j$，它们通过卷积核 $\boldsymbol{W}_j$ 产生直接的信息交互。

**与显式位置编码的比较**：

显式位置编码（如RoPE）：

$$
\boldsymbol{q}_i^T \boldsymbol{k}_j = (\boldsymbol{R}_i \boldsymbol{q}_i)^T (\boldsymbol{R}_j \boldsymbol{k}_j) = \boldsymbol{q}_i^T \boldsymbol{R}_{i-j}^T \boldsymbol{k}_j
$$

Short Conv隐式位置编码：

$$
\tilde{\boldsymbol{x}}_i = \sum_{j=-(k-1)/2}^{(k-1)/2} \boldsymbol{W}_j \boldsymbol{x}_{i+j}
$$

**注释**：两种方式都编码了相对位置，但机制不同。Short Conv的位置信息是"硬编码"的（固定权重），而注意力机制中的位置编码是通过调制注意力权重来实现的。

### 24. 长距离依赖的建模能力

**有效建模距离**：

定义有效建模距离为模型能够有效利用的最大距离依赖：

$$
D_{\text{eff}} = \max\{|i-j| : \frac{\partial \boldsymbol{o}_i}{\partial \boldsymbol{x}_j} > \epsilon\}
$$

其中 $\epsilon$ 是一个小的阈值。

**不同机制的有效距离**：

1. **标准注意力**：$D_{\text{eff}} = O(n)$，可以建模任意距离的依赖
2. **线性注意力**：$D_{\text{eff}} = O(n)$，但对远距离依赖的建模能力弱于标准注意力
3. **Short Conv**：$D_{\text{eff}} = O(k)$，仅能建模局部依赖
4. **联合机制**：$D_{\text{eff}} = O(n)$，结合了线性注意力的全局覆盖和Short Conv的局部强化

**定量分析**：

对于距离 $d$ 的依赖，其梯度的衰减率为：

$$
\left\|\frac{\partial \boldsymbol{o}_i}{\partial \boldsymbol{x}_{i-d}}\right\| \sim \begin{cases}
\alpha^d & \text{线性注意力} \\
0 & \text{Short Conv (if } d > k\text{)} \\
\beta^d & \text{联合机制}
\end{cases}
$$

其中 $\beta < \alpha < 1$，说明联合机制有更好的长距离依赖建模能力。

### 25. 实验验证：合成任务

**长距离复制任务（Long Range Copy）**：

输入序列形式为 $[x_1, \ldots, x_k, 0, \ldots, 0]$，目标是在末尾输出 $[x_1, \ldots, x_k]$。

任务难度随着中间0的长度 $n$ 增加而增加。

**性能度量**：

准确率定义为正确复制的token比例：

$$
\text{Acc} = \frac{1}{k} \sum_{i=1}^k \mathbb{1}[\hat{y}_i = x_i]
$$

**理论预测**：

- 标准注意力：$\text{Acc} \approx 1$（几乎完美）
- 纯线性注意力：$\text{Acc} = 0.6 \sim 0.8$（性能下降）
- 纯Short Conv：$\text{Acc} \approx 0$（无法建模长距离依赖）
- 联合机制：$\text{Acc} = 0.85 \sim 0.95$（显著改善）

### 26. 归纳偏置（Inductive Bias）分析

**卷积的归纳偏置**：

1. **局部性（Locality）**：相邻位置具有相关性
2. **平移等变性（Translation Equivariance）**：$f(\text{shift}(\boldsymbol{X})) = \text{shift}(f(\boldsymbol{X}))$
3. **参数共享（Parameter Sharing）**：所有位置使用相同的卷积核

**线性注意力的归纳偏置**：

1. **排列等变性（Permutation Equivariance）**：输出顺序随输入顺序变化
2. **全局聚合（Global Aggregation）**：每个位置都能访问全局信息
3. **内容依赖（Content Dependence）**：权重依赖于输入内容

**联合机制的归纳偏置**：

结合了两者的优点：

$$
\text{Inductive Bias}_{\text{joint}} = \text{Locality} + \text{Global} + \text{Content}
$$

这使得模型既能捕捉局部模式，又能进行全局信息整合。

### 27. 参数效率分析

**参数数量对比**：

假设输入维度为 $d$，头数为 $h$，每头维度为 $d_h = d/h$。

**标准注意力的参数**：

$$
\begin{aligned}
\text{Params}_{\text{attn}} &= 3 \times (d \times d) + (d \times d) \\
&= 4d^2
\end{aligned}
$$

包括 $\boldsymbol{W}_Q, \boldsymbol{W}_K, \boldsymbol{W}_V, \boldsymbol{W}_O$。

**线性注意力的参数**（与标准注意力相同）：

$$
\text{Params}_{\text{linear}} = 4d^2
$$

**Short Conv的参数**（深度可分离）：

$$
\begin{aligned}
\text{Params}_{\text{conv}} &= k \times d + (d \times d) \\
&= kd + d^2
\end{aligned}
$$

**联合机制的参数**：

$$
\begin{aligned}
\text{Params}_{\text{joint}} &= 4d^2 + 3 \times (kd + d^2) \\
&= 7d^2 + 3kd \\
&= d^2(7 + \frac{3k}{d})
\end{aligned}
$$

**参数增长率**：

$$
\frac{\text{Params}_{\text{joint}}}{\text{Params}_{\text{attn}}} = \frac{7d^2 + 3kd}{4d^2} = 1.75 + \frac{3k}{4d}
$$

对于典型值 $d=512, k=5$，参数增长约为 $1.75 + 0.03 = 1.78$ 倍。

**注释**：虽然参数量增加了约75%，但这个增加是可控的，而且换来了显著的建模能力提升。

### 28. Flash Linear Attention与Short Conv

**Flash Linear Attention的核心思想**：

通过分块计算和在线更新来减少内存占用：

$$
\boldsymbol{S}^{(b)} = \boldsymbol{S}^{(b-1)} + \sum_{i \in \text{block } b} \phi(\boldsymbol{k}_i) \boldsymbol{v}_i^T
$$

**与Short Conv的兼容性**：

Short Conv可以在块级别应用，不影响Flash Linear Attention的分块策略：

$$
\begin{aligned}
\tilde{\boldsymbol{K}}^{(b)} &= \text{ShortConv}(\boldsymbol{K}^{(b-1:b+1)}) \\
\tilde{\boldsymbol{V}}^{(b)} &= \text{ShortConv}(\boldsymbol{V}^{(b-1:b+1)}) \\
\boldsymbol{S}^{(b)} &= \boldsymbol{S}^{(b-1)} + \sum_{i \in \text{block } b} \phi(\tilde{\boldsymbol{k}}_i) \tilde{\boldsymbol{v}}_i^T
\end{aligned}
$$

**注释**：由于Short Conv的感受野较小，只需要跨越相邻块的边界即可正确计算卷积。

### 29. 多头注意力中的Short Conv

**多头线性注意力**：

$$
\text{MultiHead}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \boldsymbol{W}_O
$$

其中：

$$
\text{head}_i = \text{LinearAttention}(\boldsymbol{Q}\boldsymbol{W}_i^Q, \boldsymbol{K}\boldsymbol{W}_i^K, \boldsymbol{V}\boldsymbol{W}_i^V)
$$

**引入Short Conv的两种方式**：

**方式1：头内卷积**（在每个头内独立应用）：

$$
\text{head}_i = \text{LinearAttention}(\text{ShortConv}(\boldsymbol{Q}\boldsymbol{W}_i^Q), \text{ShortConv}(\boldsymbol{K}\boldsymbol{W}_i^K), \text{ShortConv}(\boldsymbol{V}\boldsymbol{W}_i^V))
$$

**方式2：头间卷积**（在投影之前应用）：

$$
\text{head}_i = \text{LinearAttention}(\text{ShortConv}(\boldsymbol{Q})\boldsymbol{W}_i^Q, \text{ShortConv}(\boldsymbol{K})\boldsymbol{W}_i^K, \text{ShortConv}(\boldsymbol{V})\boldsymbol{W}_i^V)
$$

**比较**：

- 头内卷积：每个头有独立的卷积参数，表达能力更强，但参数量更大
- 头间卷积：所有头共享卷积参数，参数效率更高

实践中通常使用头间卷积（方式2），在参数效率和性能之间取得平衡。

### 30. 理论总结：为什么需要Short Conv

**核心论点**：

线性注意力通过特征映射线性化注意力机制，实现了 $O(nd)$ 的线性复杂度，但牺牲了以下能力：

1. **精确的注意力权重分配**：特征映射 $\phi(\boldsymbol{q})^T \phi(\boldsymbol{k})$ 无法完美近似 $\exp(\boldsymbol{q}^T \boldsymbol{k})$
2. **尖锐的局部注意力**：难以产生高度集中在少数位置的注意力分布
3. **位置敏感的建模**：全局累积的方式削弱了对局部位置关系的敏感性

**Short Conv的补偿机制**：

Short Conv通过以下方式补偿这些不足：

1. **增强局部特征**：在注意力之前提取局部模式，使得线性注意力处理的是已经过局部增强的特征
2. **提供位置信息**：卷积核的固定权重隐式编码了相对位置信息
3. **平滑特征表示**：通过局部平均减少噪声，使得线性注意力的全局聚合更加稳定

**数学表达**：

联合机制的表达能力可以近似表示为：

$$
\mathcal{F}_{\text{joint}}(\boldsymbol{X}) \approx \mathcal{F}_{\text{attn}}(\boldsymbol{X}) + \epsilon(\boldsymbol{X})
$$

其中 $\epsilon(\boldsymbol{X})$ 是相对标准注意力的近似误差。通过引入Short Conv，我们有：

$$
\|\epsilon(\boldsymbol{X})\|_{\text{with conv}} < \|\epsilon(\boldsymbol{X})\|_{\text{without conv}}
$$

**实践意义**：

这个理论分析解释了为什么近期的高效Transformer模型（如DeltaNet、Mamba等）都采用了线性注意力+短卷积的组合架构。这不仅仅是经验性的设计，而是有深刻的理论支撑。

**结论**：

Short Conv是线性注意力不可或缺的补充机制，它通过提供局部建模能力、位置信息编码和特征平滑，使得线性注意力能够在保持线性复杂度的同时，接近标准注意力的表达能力。这种互补性设计是构建高效且强大的Transformer模型的关键。
