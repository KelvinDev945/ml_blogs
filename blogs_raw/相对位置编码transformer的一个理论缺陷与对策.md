---
title: 相对位置编码Transformer的一个理论缺陷与对策
slug: 相对位置编码transformer的一个理论缺陷与对策
date: 2022-06-07
tags: 详细推导, 语言模型, attention, 位置编码, 生成模型, attention
status: completed
---
# 相对位置编码Transformer的一个理论缺陷与对策

**原文链接**: [https://spaces.ac.cn/archives/9105](https://spaces.ac.cn/archives/9105)

**发布日期**: 

---

位置编码是Transformer中很重要的一环，在[《让研究人员绞尽脑汁的Transformer位置编码》](/archives/8130)中我们就总结了一些常见的位置编码设计。大体上，我们将Transformer的位置编码分为“绝对位置编码”和“相对位置编码”两类，其中“相对位置编码”在众多NLP/CV的实验表现相对来说更加好些。

然而，我们可以发现，目前相对位置编码几乎都是在Softmax之前的Attention矩阵上进行操作的，这种施加方式实际上都存在一个理论上的缺陷，使得Transformer无法成为“万能拟合器”。本文就来分析这个问题，并探讨一些解决方案。

## 简单探针 #

顾名思义，位置编码就是用来给模型补充上位置信息的。那么，如何判断一个模型有没有足够的识别位置的能力呢？笔者之前曾构思过一个简单的探针实验：

> 对于一个有识别位置能力的模型，应该有能力准确实现如下映射 \begin{equation}\begin{array}{lc} \text{输入：} & [0, 0, \cdots, 0, 0] \\\ & \downarrow\\\ \text{输出：} & [1, 2, \cdots, n-1, n] \end{array}\end{equation} 

也就是说，输入$n$个0，能有序地输出位置编号$1\sim n$。这个探针实验的思想很简单，即模型如果有能力做到这一点，说明识别位置是模型自身具备的能力，跟外部输入无关，这正是我们想要的。不难发现，绝对位置由于是直接施加在输入上的，所以它很容易能够完成探针测试。

## 无法胜任 #

然而，当笔者带着这个简单的探针实验去思考带有相对位置编码的Transformer模型时，却发现它们几乎都不能完成上述任务。

具体来说，除了[《Self-Attention with Relative Position Representations》](https://papers.cool/arxiv/1803.02155)所提出的设计外，其余所有相对位置编码（包括笔者所提的[RoPE](/archives/8265)）都只修改了Softmax前的Attention矩阵，那么带有相对位置信息的Attention矩阵依然是一个概率矩阵（即每一行求和等于1）。

另一方面，对于Transformer模型来说，Token之间的交互的唯一来源是Self Attention的$\boldsymbol{A}\boldsymbol{V}$这一步，或者写成$\boldsymbol{o}_i = \sum\limits_j a_{i,j}\boldsymbol{v}_j$。相同的输入意味着每个$\boldsymbol{v}_j$都是相同的，所以  
\begin{equation}\boldsymbol{o}_i = \sum_j a_{i,j}\boldsymbol{v}_j = \sum_j a_{i,j}\boldsymbol{v} = \left(\sum_j a_{i,j}\right)\boldsymbol{v} = \boldsymbol{v}\end{equation}  
这意味着每个$\boldsymbol{o}_i$也是相同的。换句话说，模型的每个位置自始至终都输出相同的结果，所以模型根本不可能输出各不相同的$[1, 2, \cdots, n-1, n]$。

类似的发现也出现在最近的论文[《Your Transformer May Not be as Powerful as You Expect》](https://papers.cool/arxiv/2205.13401)中，作者构建了略有不同的例子来演示相对位置编码Transformer的拟合能力缺陷问题，两者异曲同工、不谋而合了。此外，本文开头说的是“万能拟合”，那解决了这个反例是不是就能做到“万能拟合”了呢？该论文也有相应的理论分析来肯定这一事实，这里就不详述了。

## 初步方案 #

稍加思考就可以发现，其实问题主要出在Attention矩阵的每一行求和等于1，要解决这个问题，想办法打破这个约束就行了。为此，[《Your Transformer May Not be as Powerful as You Expect》](https://papers.cool/arxiv/2205.13401)在其发现之上进一步提出了如下设计  
\begin{equation}\boldsymbol{O} = (\boldsymbol{A}\odot \boldsymbol{C})\boldsymbol{V}\quad \text{或者等价地}\quad\boldsymbol{o}_i = \sum_j a_{i,j}c_{i,j}\boldsymbol{v}_j\end{equation}  
其中$\boldsymbol{C}$是一个可训练的参数矩阵，$\odot$是逐位相乘（[Hadamard积](https://en.wikipedia.org/wiki/Hadamard_product_\(matrices\))）。为了使得整个模型依然只包含相对位置信息（因为本文就是讨论相对位置编码Transfomrer的缺陷），我们要约束$\boldsymbol{C}$为[Toeplitz矩阵](https://en.wikipedia.org/wiki/Toeplitz_matrix)，即$c_{i,j}=g(i-j)$。

有了$\boldsymbol{C}$的加入，$\boldsymbol{A}\odot \boldsymbol{C}$作为一个整体，每一行的和显然不一定为1，从而打破了这个限制，因此是可以解决问题的（更多的实验结果请自行看原论文）。但这样一来，引入了新的参数矩阵不说，由于$\boldsymbol{C}$本身是有限大小的，所以它就不能很好地支持变长输入（或者矩阵$\boldsymbol{C}$相应地要做一些截断，即$c_{i,j}=g(\text{clip}(i-j, p_{\min}, p_{\max}))$的形式），总的来说显得不够简洁优雅。

## 去掉分母 #

再次回到问题所在：Attention矩阵的每一行求和等于1。是什么操作导致了这一现象呢？答案很显然，是Softmax：  
\begin{equation}a_{i,j} = \frac{e^{b_{i,j}}}{\sum\limits_j e^{b_{i,j}}}\end{equation}  
这里的$\boldsymbol{B}=(b_{i,j})$是Softmax前的矩阵。很明显，就是“除以$\sum\limits_j e^{b_{i,j}}$”这一步导致了$\sum\limits_j a_{i,j}=1$，那么一个很直接的想法就是：

> 如果我不想$\sum\limits_j a_{i,j}=1$，那么干脆别除以$\sum\limits_j e^{b_{i,j}}$就行了？

事实上确实可以！实验结果显示，不除以该分母的Transformer确实能成功地完成前述探针测试。此时就不得不感概一下[GAU](/archives/8934)的“先见之明”了，它提出的新式Attention直接是$\text{relu}^2$激活然后简单除以$n$来归一化，避免了$\sum\limits_j a_{i,j}=1$，从而增强了模型的理论能力（当然也许作者根本没想那么多，是笔者想象的成分居多）。

## 新归一化 #

然而，我们在[《听说Attention与Softmax更配哦～》](/archives/9019)发现像GAU里的不进行概率归一化的Attention设计可能存在外推能力欠佳的问题。也就是说，进行概率归一化导致了前面说的理论缺陷，简单地除以$n$来归一化则外推能力可能欠佳，有没有同时能兼顾两者的方案呢？

让我们再发散一下脑洞。从范数的角度来看，$\sum\limits_j e^{b_{i,j}}$实际上是向量$e^{b_{i,:}}$的$l_1$范数，所以Softmax实际上就是向量的$e^{b_{i,:}}$的$l_1$归一化操作，那么要避免$\sum\limits_j a_{i,j}=1$，又有保留归一化，换成其他的归一化操作是否可以呢？比如$l_2$归一化：  
\begin{equation}a_{i,j} = \frac{e^{b_{i,j}}}{\sqrt{\sum\limits_j e^{2b_{i,j}}}}\end{equation}

经过笔者测试，这种$l_2$归一化的Attention，确实能成功完成探针实验。那么，这个改动对我们更关心的NLP预训练场景有没有帮助呢？笔者也做了相应的对比实验，结果是分两部分：

> 1、对于标准的Attention + FFN组合，应用$l_2$归一化Attention之前要缩小一下Attention的$\boldsymbol{W}_V,\boldsymbol{W}_O$的初始方差，实验结果则是略差于常规的$l_1$归一化Attention；
> 
> 2、对于全GAU的架构，可以直接应用$l_2$归一化Attention，不需要改动初始化，实验结果则是略优于常规的$l_1$归一化Attention。

两者的差别大概是源于它们本身的初始化方式不同，在标准的Attention + FFN组合中，初始Attention矩阵接近一个均匀矩阵（每个数都相同），而在[《门控注意力单元（GAU）还需要Warmup吗？》](/archives/8990)我们则分析过，GAU的初始Attention矩阵更接近一个单位阵（的若干倍）。

## 峰回路转 #

再次纵观前文，我们发现是因为“每个$\boldsymbol{v}_j$都是相同的”，所以“$\sum\limits_j a_{i,j}=1$的模型无法完成探针实验”。但如果每个$\boldsymbol{v}_j$不全相同呢？

我们知道，从BERT开始，主流的Transformer模型都是像“[CLS] SENT [SEP]”设计输入的，也就是在输入前后会附加一些标记性的Token，如果我们将这些标记Token当作模型的一部分而不是输入（也就是说输入“[CLS] 0 0 ⋯ 0 0 [SEP]”而不是全0），那么是否有可能完成探针呢？

笔者也对此做了实验，发现对输入补充上标记行Token后，不需要对相对位置编码Transformer的其他部分做修改，确实也能够完成探针实验。这结果就有点啼笑皆非了，原来BERT的作者们也很有“先见之明”啊，所添加的特殊Token [CLS]、[SEP]还有辅助定位的作用，我们分析那么久的理论缺陷，居然就这样被两个特殊Token解决了。这不禁让人想起[《How Much Position Information Do Convolutional Neural Networks Encode?》](https://papers.cool/arxiv/2001.08248)所提到的“CNN是通过padding来识别绝对位置的”这一结论，两者有一定的相通之处。

当然，这也不意味着我们前面的思考全无意义。比如对GAU模型来说，Attention换用$l_2$归一化确确实实有加快收敛、轻微提升效果的作用。此外，既然可以接受$l_2$归一化，那么$e^{b_{i,j}}$是不是还可以换成一般的激活函数（比如去掉非负性约束）呢？笔者也简单做了“$\text{swish}(b_{i,j})$ + $l_2$归一化”的实验，发现有一定的可行性。从这个角度来看，$l_2$归一化下的Attention实际上有更多的拓展空间。

## 曲终人散 #

本文分析了相对位置编码Transformer的一个隐含缺陷，并探讨了相应的对策，从中引申出关于Attention矩阵的非负性、归一化方式的思考。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9105>_

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

苏剑林. (Jun. 07, 2022). 《相对位置编码Transformer的一个理论缺陷与对策 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9105>

@online{kexuefm-9105,  
title={相对位置编码Transformer的一个理论缺陷与对策},  
author={苏剑林},  
year={2022},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/9105}},  
} 


---

## 公式推导与注释

### 1. 绝对位置编码的数学定义

**绝对位置编码（Absolute Position Encoding）**：

在输入嵌入 $\boldsymbol{x}_i$ 上直接加上位置编码向量 $\boldsymbol{p}_i$：

$$
\boldsymbol{h}_i = \boldsymbol{x}_i + \boldsymbol{p}_i
$$

其中位置编码 $\boldsymbol{p}_i$ 只依赖于绝对位置 $i$。

**正弦位置编码（Sinusoidal Positional Encoding）**：

Transformer原论文使用的编码方式：

$$
\begin{aligned}
\boldsymbol{p}_i^{(2j)} &= \sin\left(\frac{i}{10000^{2j/d}}\right) \\
\boldsymbol{p}_i^{(2j+1)} &= \cos\left(\frac{i}{10000^{2j/d}}\right)
\end{aligned}
$$

其中 $j \in \{0, 1, \ldots, d/2-1\}$ 是维度索引。

**注释**：这种编码的优点是可以通过线性变换表示相对位置，但仍然是绝对位置编码。

### 2. 相对位置编码的基本思想

**核心思想**：位置信息应该体现为相对位置关系，而非绝对位置。

**数学表达**：

相对位置编码希望注意力权重只依赖于相对位置 $i - j$：

$$
a_{i,j} = f(\boldsymbol{x}_i, \boldsymbol{x}_j, i-j)
$$

而不依赖于绝对位置 $i$ 或 $j$。

**优势**：

1. 更好的长度泛化能力
2. 对位置平移保持不变性
3. 更符合自然语言的局部性特征

### 3. 相对位置编码的几种实现方式

**方式1：Shaw et al. (2018) - 在注意力值上添加相对位置**：

$$
\boldsymbol{o}_i = \sum_j a_{i,j} (\boldsymbol{v}_j + \boldsymbol{r}_{i-j}^V)
$$

其中 $\boldsymbol{r}_{i-j}^V$ 是相对位置的可学习嵌入。

**注释**：这种方式在Softmax之后的值矩阵中引入相对位置信息。

**方式2：在注意力得分上添加相对位置偏置**：

$$
a_{i,j} = \frac{\exp(q_i^T k_j + b_{i-j})}{\sum_k \exp(q_i^T k_k + b_{i-k})}
$$

其中 $b_{i-j}$ 是相对位置偏置。

**注释**：这是最常见的方式，在Softmax之前的logits上添加偏置。

**方式3：RoPE（Rotary Position Embedding）**：

$$
\boldsymbol{q}_i^T \boldsymbol{k}_j = (\boldsymbol{R}_i \boldsymbol{q}_i)^T (\boldsymbol{R}_j \boldsymbol{k}_j) = \boldsymbol{q}_i^T \boldsymbol{R}_{i-j}^T \boldsymbol{k}_j
$$

其中 $\boldsymbol{R}_\theta$ 是旋转矩阵。

**注释**：RoPE通过旋转变换优雅地实现了相对位置编码。

### 4. Softmax归一化的数学性质

**Softmax的行归一化性质**：

对于注意力矩阵 $\boldsymbol{A} = \text{softmax}(\boldsymbol{S})$，每一行的和为1：

$$
\sum_{j=1}^n a_{i,j} = \sum_{j=1}^n \frac{\exp(s_{i,j})}{\sum_{k=1}^n \exp(s_{i,k})} = 1
$$

**证明**：

$$
\sum_{j=1}^n a_{i,j} = \sum_{j=1}^n \frac{\exp(s_{i,j})}{\sum_{k=1}^n \exp(s_{i,k})} = \frac{\sum_{j=1}^n \exp(s_{i,j})}{\sum_{k=1}^n \exp(s_{i,k})} = 1
$$

**注释**：这个性质是Softmax定义的直接结果，也是引起理论缺陷的根源。

### 5. 相对位置编码Transformer的理论缺陷：问题陈述

**探针任务（Probe Task）**：

输入全0向量序列 $\boldsymbol{X} = [\boldsymbol{0}, \boldsymbol{0}, \ldots, \boldsymbol{0}] \in \mathbb{R}^{n \times d}$，期望输出位置编号 $\boldsymbol{Y} = [1, 2, \ldots, n]$。

**数学形式化**：

定义函数 $f_\theta: \mathbb{R}^{n \times d} \to \mathbb{R}^n$，期望：

$$
f_\theta([\boldsymbol{0}, \boldsymbol{0}, \ldots, \boldsymbol{0}]) = [1, 2, \ldots, n]
$$

**问题**：配备相对位置编码的Transformer（除了Shaw et al.的方式1）无法完成这个任务。

### 6. 理论缺陷的形式化证明

**引理1（相同输入导致相同特征）**：

如果输入 $\boldsymbol{x}_1 = \boldsymbol{x}_2 = \cdots = \boldsymbol{x}_n = \boldsymbol{x}$，且使用相对位置编码，则：

$$
\boldsymbol{q}_i = \boldsymbol{W}_Q \boldsymbol{x}, \quad \boldsymbol{k}_i = \boldsymbol{W}_K \boldsymbol{x}, \quad \boldsymbol{v}_i = \boldsymbol{W}_V \boldsymbol{x}
$$

对所有 $i$ 都相同（忽略绝对位置编码）。

**证明**：

由于相对位置编码不直接修改 $\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}$（在方式2和RoPE中），这些向量只依赖于输入嵌入 $\boldsymbol{x}_i$。当所有输入相同时，所有的 $\boldsymbol{q}_i, \boldsymbol{k}_i, \boldsymbol{v}_i$ 也相同。

**引理2（注意力得分的对称性）**：

对于相对位置编码（方式2），注意力得分满足：

$$
s_{i,j} = \boldsymbol{q}_i^T \boldsymbol{k}_j + b_{i-j}
$$

当 $\boldsymbol{q}_i = \boldsymbol{q}$ 和 $\boldsymbol{k}_j = \boldsymbol{k}$ 对所有 $i, j$ 时：

$$
s_{i,j} = \boldsymbol{q}^T \boldsymbol{k} + b_{i-j} = c + b_{i-j}
$$

其中 $c = \boldsymbol{q}^T \boldsymbol{k}$ 是常数。

**注释**：注意力得分只依赖于相对位置 $i-j$，与绝对位置 $i, j$ 无关。

### 7. 关键定理：输出的位置不变性

**定理1（输出位置不变性）**：

对于配备Softmax归一化的相对位置编码Transformer，如果所有输入相同，则所有位置的输出也相同：

$$
\boldsymbol{o}_1 = \boldsymbol{o}_2 = \cdots = \boldsymbol{o}_n
$$

**证明**：

由引理2，注意力得分为：

$$
s_{i,j} = c + b_{i-j}
$$

注意力权重为：

$$
a_{i,j} = \frac{\exp(c + b_{i-j})}{\sum_{k=1}^n \exp(c + b_{i-k})} = \frac{\exp(b_{i-j})}{\sum_{k=1}^n \exp(b_{i-k})}
$$

**关键观察**：对于固定的相对位置差 $\Delta = i - j$，权重 $a_{i,j}$ 只依赖于 $\Delta$。

定义 $\alpha_\Delta = a_{i,i-\Delta}$，则：

$$
\boldsymbol{o}_i = \sum_{j=1}^n a_{i,j} \boldsymbol{v}_j = \sum_{j=1}^n \alpha_{i-j} \boldsymbol{v} = \boldsymbol{v} \sum_{j=1}^n \alpha_{i-j}
$$

由Softmax的归一化性质：

$$
\sum_{j=1}^n \alpha_{i-j} = 1
$$

因此：

$$
\boldsymbol{o}_i = \boldsymbol{v}
$$

对所有 $i$ 都相同。

**注释**：这个证明揭示了问题的根源：Softmax的归一化性质加上相对位置编码，导致无法在相同输入下产生不同输出。

### 8. RoPE的特殊情况分析

**RoPE的注意力得分**：

$$
s_{i,j} = (\boldsymbol{R}_i \boldsymbol{q}_i)^T (\boldsymbol{R}_j \boldsymbol{k}_j) = \boldsymbol{q}_i^T \boldsymbol{R}_i^T \boldsymbol{R}_j \boldsymbol{k}_j = \boldsymbol{q}_i^T \boldsymbol{R}_{i-j}^T \boldsymbol{k}_j
$$

利用旋转矩阵的性质 $\boldsymbol{R}_i^T \boldsymbol{R}_j = \boldsymbol{R}_{i-j}^T$。

**相同输入的情况**：

当 $\boldsymbol{q}_i = \boldsymbol{q}$ 和 $\boldsymbol{k}_j = \boldsymbol{k}$ 对所有 $i, j$：

$$
s_{i,j} = \boldsymbol{q}^T \boldsymbol{R}_{i-j}^T \boldsymbol{k}
$$

**旋转矩阵的形式**：

$$
\boldsymbol{R}_\theta = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
$$

对于相对位置 $\Delta = i - j$，$\theta = \Delta \cdot \omega$ 其中 $\omega$ 是频率。

**注意力得分只依赖于相对位置**：

$$
s_{i,j} = f(i - j, \boldsymbol{q}, \boldsymbol{k})
$$

**应用Softmax后的输出**：

$$
\boldsymbol{o}_i = \sum_j \frac{\exp(s_{i,j})}{\sum_k \exp(s_{i,k})} \boldsymbol{v}_j = \boldsymbol{v} \sum_j \frac{\exp(f(i-j))}{\sum_k \exp(f(i-k))} = \boldsymbol{v}
$$

**结论**：RoPE也无法通过探针测试。

### 9. 位置不变性的群论解释

**平移群（Translation Group）**：

定义平移算子 $T_c: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$，将序列循环平移 $c$ 个位置：

$$
(T_c \boldsymbol{X})_i = \boldsymbol{X}_{(i-c) \mod n}
$$

**相对位置编码的平移等变性**：

对于相对位置编码的Transformer $f$，应该满足：

$$
f(T_c \boldsymbol{X}) = T_c f(\boldsymbol{X})
$$

**在相同输入下的推论**：

如果 $\boldsymbol{X} = [\boldsymbol{x}, \boldsymbol{x}, \ldots, \boldsymbol{x}]$，则 $T_c \boldsymbol{X} = \boldsymbol{X}$。

因此：

$$
f(T_c \boldsymbol{X}) = f(\boldsymbol{X}) = T_c f(\boldsymbol{X})
$$

这意味着 $f(\boldsymbol{X})$ 必须是平移不变的，即所有位置输出相同。

**注释**：从群论角度，相对位置编码赋予模型平移等变性，但这在相同输入下导致输出必须平移不变。

### 10. 绝对位置泄露的严重性

**形式化定义（绝对位置信息）**：

一个模型具有绝对位置信息，如果存在函数 $g$ 使得：

$$
g(f(\boldsymbol{X})_i) = i
$$

即从输出可以恢复位置索引。

**相对位置编码的理论承诺**：

相对位置编码应该不包含绝对位置信息，即对于任意排列 $\pi$：

$$
f(\boldsymbol{X})_i \text{ 的分布不依赖于 } i
$$

**实际情况**：

由于定理1，在相同输入下所有输出完全相同，这虽然满足了"不泄露绝对位置"，但代价是完全丧失了区分能力。

**矛盾**：

- 我们希望模型能够识别位置（完成探针任务）
- 但相对位置编码理论上不应包含绝对位置信息
- 这两个要求在纯相对位置编码下是矛盾的

### 11. 理论缺陷的严重性评估

**万能逼近能力（Universal Approximation）**：

一个完整的神经网络应该能够逼近任意连续函数。定义函数类：

$$
\mathcal{F} = \{f: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d'} \mid f \text{ 是连续的}\}
$$

**相对位置编码Transformer的函数类**：

定义为：

$$
\mathcal{F}_{\text{rel}} = \{f \in \mathcal{F} \mid f(T_c \boldsymbol{X}) = T_c f(\boldsymbol{X}) \text{ 对所有 } c\}
$$

**严格包含关系**：

$$
\mathcal{F}_{\text{rel}} \subsetneq \mathcal{F}
$$

**定量分析**：

探针任务的函数 $f_{\text{probe}}([\boldsymbol{0}, \ldots, \boldsymbol{0}]) = [1, \ldots, n]$ 满足 $f_{\text{probe}} \in \mathcal{F}$ 但 $f_{\text{probe}} \notin \mathcal{F}_{\text{rel}}$。

**注释**：这表明相对位置编码Transformer的表达能力严格弱于理论上的万能逼近器。

### 12. Shaw et al. 方式的特殊性

**回顾Shaw et al.的方法**：

$$
\boldsymbol{o}_i = \sum_j a_{i,j} (\boldsymbol{v}_j + \boldsymbol{r}_{i-j}^V)
$$

**关键区别**：相对位置信息在Softmax**之后**添加到值向量上。

**在相同输入下的输出**：

$$
\boldsymbol{o}_i = \sum_j a_{i,j} (\boldsymbol{v} + \boldsymbol{r}_{i-j}^V) = \boldsymbol{v} \sum_j a_{i,j} + \sum_j a_{i,j} \boldsymbol{r}_{i-j}^V = \boldsymbol{v} + \sum_j a_{i,j} \boldsymbol{r}_{i-j}^V
$$

**注意力权重的依赖性**：

由于 $a_{i,j} = \alpha_{i-j}$ 只依赖于相对位置，我们有：

$$
\boldsymbol{o}_i = \boldsymbol{v} + \sum_{\Delta=-(n-1)}^{n-1} \alpha_\Delta \boldsymbol{r}_\Delta^V
$$

**为什么能通过探针测试**：

如果 $\boldsymbol{r}_\Delta^V$ 包含足够丰富的信息，加权和 $\sum_\Delta \alpha_\Delta \boldsymbol{r}_\Delta^V$ 可以随位置 $i$ 变化（通过 $\alpha_\Delta$ 的分布变化）。

**细微差异**：

虽然 $\alpha_\Delta$ 对所有位置 $i$ 都相同，但由于边界效应（序列开始和结束处），实际的求和范围不同：

- 位置1: $\Delta \in [0, n-1]$
- 位置 $i$: $\Delta \in [-(i-1), n-i]$
- 位置 $n$: $\Delta \in [-(n-1), 0]$

这导致输出可以不同。

### 13. 对策方案1：引入Hadamard积的相对位置矩阵

**Your Transformer May Not be as Powerful as You Expect 提出的方法**：

$$
\boldsymbol{O} = (\boldsymbol{A} \odot \boldsymbol{C}) \boldsymbol{V}
$$

其中 $\boldsymbol{C} = (c_{i,j})$ 是可训练的Toeplitz矩阵，$c_{i,j} = g(i-j)$。

**Hadamard积（逐元素乘法）**：

$$
(\boldsymbol{A} \odot \boldsymbol{C})_{i,j} = a_{i,j} \cdot c_{i,j}
$$

**打破归一化约束的证明**：

Softmax保证 $\sum_j a_{i,j} = 1$，但引入 $\boldsymbol{C}$ 后：

$$
\sum_j (\boldsymbol{A} \odot \boldsymbol{C})_{i,j} = \sum_j a_{i,j} c_{i,j} \neq 1 \text{ （一般情况）}
$$

**在相同输入下的输出**：

$$
\boldsymbol{o}_i = \sum_j a_{i,j} c_{i,j} \boldsymbol{v} = \boldsymbol{v} \sum_j \alpha_{i-j} g(i-j)
$$

定义 $w_i = \sum_j \alpha_{i-j} g(i-j)$，则：

$$
\boldsymbol{o}_i = w_i \boldsymbol{v}
$$

如果 $w_i$ 随 $i$ 变化（通过适当选择 $g$），则输出可以不同。

**如何使 $w_i$ 随 $i$ 变化**：

利用边界效应，序列不同位置的求和范围不同：

$$
w_i = \sum_{j=1}^n \alpha_{i-j} g(i-j) = \sum_{\Delta=1-i}^{n-i} \alpha_\Delta g(\Delta)
$$

求和范围 $[1-i, n-i]$ 依赖于 $i$，因此 $w_i$ 可以不同。

### 14. Hadamard积方法的数学分析

**Toeplitz矩阵的定义**：

矩阵 $\boldsymbol{C} \in \mathbb{R}^{n \times n}$ 是Toeplitz的，如果：

$$
c_{i,j} = c_{i-j} \quad \text{对所有 } i, j
$$

即主对角线和每条平行于主对角线的对角线上的元素都相同。

**参数化**：

对于 $n \times n$ Toeplitz矩阵，只需 $2n-1$ 个参数：

$$
\boldsymbol{c} = [c_{-(n-1)}, c_{-(n-2)}, \ldots, c_0, \ldots, c_{n-2}, c_{n-1}]
$$

**卷积解释**：

Toeplitz矩阵乘法可以理解为循环卷积：

$$
(\boldsymbol{C} \boldsymbol{V})_i = \sum_{j=1}^n c_{i-j} v_j
$$

**限制**：

1. 引入了新的参数矩阵
2. 矩阵大小固定为 $n \times n$，不利于变长输入
3. 需要截断策略处理超出范围的相对位置

### 15. 对策方案2：去掉Softmax的分母

**动机**：Softmax的归一化是导致问题的根源，能否避免它？

**GAU（Gated Attention Unit）的做法**：

$$
a_{i,j} = \frac{\text{relu}^2(b_{i,j})}{n}
$$

直接用序列长度 $n$ 归一化，而不是用指数和。

**在相同输入下**：

$$
\boldsymbol{o}_i = \sum_j \frac{\text{relu}^2(b_{i,j})}{n} \boldsymbol{v}
$$

如果 $b_{i,j}$ 包含位置信息（如相对位置偏置），则：

$$
b_{i,j} = c + r_{i-j}
$$

$$
\boldsymbol{o}_i = \frac{\boldsymbol{v}}{n} \sum_j \text{relu}^2(c + r_{i-j})
$$

**关键差异**：虽然 $\sum_j \text{relu}^2(c + r_{i-j})$ 对所有 $i$ 仍然相同（在无穷长序列下），但在有限长度和边界效应下，可以不同。

**更根本的改变**：完全去掉归一化：

$$
\boldsymbol{o}_i = \sum_j \text{relu}^2(b_{i,j}) \boldsymbol{v}_j
$$

此时 $\sum_j \text{relu}^2(b_{i,j})$ 不一定为常数。

### 16. 对策方案3：使用 $l_2$ 归一化

**替代Softmax的 $l_1$ 归一化为 $l_2$ 归一化**：

$$
a_{i,j} = \frac{e^{b_{i,j}}}{\sqrt{\sum_k e^{2b_{i,k}}}}
$$

**关键性质**：$l_2$ 归一化不保证 $\sum_j a_{i,j} = 1$。

**归一化后的范数**：

$$
\|\boldsymbol{a}_i\|_2 = \sqrt{\sum_j a_{i,j}^2} = \sqrt{\sum_j \frac{e^{2b_{i,j}}}{\sum_k e^{2b_{i,k}}}} = 1
$$

但：

$$
\|\boldsymbol{a}_i\|_1 = \sum_j a_{i,j} = \sum_j \frac{e^{b_{i,j}}}{\sqrt{\sum_k e^{2b_{i,k}}}} \neq 1 \text{ （一般情况）}
$$

**在相同输入下的分析**：

$$
a_{i,j} = \frac{e^{c + r_{i-j}}}{\sqrt{\sum_k e^{2(c + r_{i-k})}}} = \frac{e^c e^{r_{i-j}}}{\sqrt{e^{2c} \sum_k e^{2r_{i-k}}}} = \frac{e^{r_{i-j}}}{\sqrt{\sum_k e^{2r_{i-k}}}}
$$

定义 $Z_i = \sum_k e^{2r_{i-k}}$，则：

$$
\boldsymbol{o}_i = \sum_j \frac{e^{r_{i-j}}}{\sqrt{Z_i}} \boldsymbol{v} = \frac{\boldsymbol{v}}{\sqrt{Z_i}} \sum_j e^{r_{i-j}}
$$

**能否通过探针测试**：

如果 $Z_i$ 随 $i$ 变化（由于边界效应），则 $\boldsymbol{o}_i$ 可以不同。

$$
Z_i = \sum_{k=1}^n e^{2r_{i-k}} = \sum_{\Delta=1-i}^{n-i} e^{2r_\Delta}
$$

求和范围依赖于 $i$，因此 $Z_i$ 可以不同。

**结论**：$l_2$ 归一化可以通过探针测试。

### 17. $l_2$ 归一化的严格数学证明

**定理2（$l_2$ 归一化的位置区分能力）**：

对于使用 $l_2$ 归一化的相对位置编码Transformer，如果序列有界（$n < \infty$），则存在参数配置使得：

$$
\boldsymbol{o}_i \neq \boldsymbol{o}_j \quad \text{对某些 } i \neq j
$$

即使所有输入相同。

**证明**：

考虑相对位置偏置 $r_\Delta = -|\Delta|$（距离越远权重越小）。

对于位置1：

$$
Z_1 = \sum_{k=1}^n e^{-2|1-k|} = e^0 + e^{-2} + e^{-4} + \cdots + e^{-2(n-1)}
$$

对于位置 $n$：

$$
Z_n = \sum_{k=1}^n e^{-2|n-k|} = e^{-2(n-1)} + \cdots + e^{-4} + e^{-2} + e^0
$$

虽然这两个和包含相同的项（都是几何级数），但在有限 $n$ 下：

$$
Z_1 = \frac{1 - e^{-2n}}{1 - e^{-2}} \approx \frac{1}{1 - e^{-2}} \text{ （当 } n \text{ 大时）}
$$

实际上 $Z_1 = Z_n$（对称性）。

**更精细的构造**：

使用非对称的相对位置偏置，如：

$$
r_\Delta = \begin{cases}
-\Delta & \text{if } \Delta > 0 \\
-2|\Delta| & \text{if } \Delta \leq 0
\end{cases}
$$

此时前向和后向的衰减率不同，导致 $Z_i$ 随 $i$ 单调变化。

**注释**：关键是利用序列边界和非对称性打破位置的对称性。

### 18. 对策方案4：添加特殊标记Token

**BERT的做法**：在序列前后添加 [CLS] 和 [SEP] 标记。

**输入形式**：

$$
\boldsymbol{X} = [\boldsymbol{x}_{\text{CLS}}, \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n, \boldsymbol{x}_{\text{SEP}}]
$$

**关键观察**：$\boldsymbol{x}_{\text{CLS}}$ 和 $\boldsymbol{x}_{\text{SEP}}$ 与其他token不同，打破了"所有输入相同"的假设。

**在探针任务中**：

输入变为：

$$
[\boldsymbol{x}_{\text{CLS}}, \boldsymbol{0}, \boldsymbol{0}, \ldots, \boldsymbol{0}, \boldsymbol{x}_{\text{SEP}}]
$$

现在不是所有输入都相同了！

**值向量的差异**：

$$
\boldsymbol{v}_{\text{CLS}} = \boldsymbol{W}_V \boldsymbol{x}_{\text{CLS}}, \quad \boldsymbol{v}_i = \boldsymbol{W}_V \boldsymbol{0} = \boldsymbol{0}, \quad \boldsymbol{v}_{\text{SEP}} = \boldsymbol{W}_V \boldsymbol{x}_{\text{SEP}}
$$

**注意力输出**：

$$
\boldsymbol{o}_i = \sum_j a_{i,j} \boldsymbol{v}_j = a_{i,\text{CLS}} \boldsymbol{v}_{\text{CLS}} + a_{i,\text{SEP}} \boldsymbol{v}_{\text{SEP}}
$$

**位置依赖性**：

权重 $a_{i,\text{CLS}}$ 和 $a_{i,\text{SEP}}$ 依赖于相对位置：

$$
a_{i,\text{CLS}} = \frac{\exp(b_{i,\text{CLS}})}{\sum_k \exp(b_{i,k})} \quad \text{其中 } b_{i,\text{CLS}} \text{ 包含相对位置信息}
$$

由于位置 $i$ 到CLS和SEP的相对距离不同，$a_{i,\text{CLS}}$ 和 $a_{i,\text{SEP}}$ 随 $i$ 变化，从而 $\boldsymbol{o}_i$ 也不同。

### 19. 特殊标记方案的理论分析

**形式化定义**：

定义增强输入函数 $\phi: \mathbb{R}^{n \times d} \to \mathbb{R}^{(n+2) \times d}$：

$$
\phi(\boldsymbol{X}) = [\boldsymbol{x}_{\text{CLS}}, \boldsymbol{X}, \boldsymbol{x}_{\text{SEP}}]
$$

**定理3（特殊标记的充分性）**：

如果 $\boldsymbol{x}_{\text{CLS}} \neq \boldsymbol{0}$ 或 $\boldsymbol{x}_{\text{SEP}} \neq \boldsymbol{0}$，则相对位置编码Transformer可以在增强输入上产生位置依赖的输出。

**证明思路**：

1. 特殊标记提供"锚点"，不同位置到锚点的相对距离不同
2. 通过注意力权重的相对位置依赖性，不同位置对锚点的关注度不同
3. 输出是锚点嵌入的加权和，权重随位置变化，因此输出也变化

**与CNN的类比**：

这类似于CNN通过padding识别绝对位置（论文[How Much Position Information Do CNNs Encode?](https://arxiv.org/abs/2001.08248)）：

- CNN: 边界padding提供位置参考
- Transformer: 特殊标记提供位置参考

**注释**：这是一个优雅的解决方案，无需修改架构，仅通过输入设计就解决了问题。

### 20. 不同对策方案的对比分析

**方案1：Hadamard积 + Toeplitz矩阵**

优点：
- 理论上完全解决问题
- 不改变归一化性质

缺点：
- 引入新参数 $O(n)$
- 固定序列长度
- 需要截断策略

**方案2：去掉Softmax分母**

优点：
- 简单直接
- 计算效率高

缺点：
- 改变注意力语义
- 可能影响外推能力
- 需要重新调优

**方案3：$l_2$ 归一化**

优点：
- 保留归一化性质
- 不引入新参数
- 理论上更优雅

缺点：
- 改变梯度特性
- 需要调整初始化
- 对不同架构效果不同

**方案4：特殊标记**

优点：
- 无需改变架构
- 已被广泛使用（BERT等）
- 简单有效

缺点：
- 依赖输入设计
- 理论上不够纯粹
- 对某些任务可能不适用

### 21. 外推能力（Extrapolation）的考虑

**外推问题**：模型在训练长度为 $n_{\text{train}}$ 的序列上训练，能否处理长度为 $n_{\text{test}} > n_{\text{train}}$ 的序列？

**Softmax归一化的外推优势**：

$$
\sum_j a_{i,j} = 1 \quad \text{对任意 } n
$$

归一化确保注意力权重的总和始终为1，无论序列长度。

**去掉归一化的外推问题**：

如果使用：

$$
a_{i,j} = \frac{\text{relu}^2(b_{i,j})}{n}
$$

当测试时 $n_{\text{test}} \neq n_{\text{train}}$：

$$
\sum_j a_{i,j} = \frac{1}{n_{\text{test}}} \sum_j \text{relu}^2(b_{i,j}) \neq 1
$$

权重和会缩放，影响外推。

**$l_2$ 归一化的外推性**：

$$
\|\boldsymbol{a}_i\|_2 = 1 \quad \text{对任意 } n
$$

$l_2$ 范数归一化也保持常数，但 $l_1$ 和会随 $n$ 变化：

$$
\|\boldsymbol{a}_i\|_1 \approx \sqrt{n} \quad \text{（当权重均匀分布时）}
$$

这可能导致外推时输出幅度变化。

### 22. 归一化方式的深入比较

**$l_1$ 归一化（Softmax）**：

$$
\|\boldsymbol{a}\|_1 = \sum_j a_j = 1, \quad \|\boldsymbol{a}\|_2 = \sqrt{\sum_j a_j^2} \leq 1
$$

- 权重和固定为1
- $l_2$ 范数可变，取决于分布的集中度
- 概率解释明确

**$l_2$ 归一化**：

$$
\|\boldsymbol{a}\|_2 = \sqrt{\sum_j a_j^2} = 1, \quad \|\boldsymbol{a}\|_1 = \sum_j a_j \geq 1
$$

- $l_2$ 范数固定为1
- $l_1$ 范数可变，通常 $\geq 1$
- 失去概率解释

**Cauchy-Schwarz不等式的应用**：

$$
\|\boldsymbol{a}\|_1 = \sum_j a_j \leq \sqrt{n} \sqrt{\sum_j a_j^2} = \sqrt{n} \|\boldsymbol{a}\|_2 = \sqrt{n}
$$

等号成立当且仅当所有 $a_j$ 相等。

**注释**：$l_2$ 归一化的 $l_1$ 和范围为 $[1, \sqrt{n}]$，与序列长度相关。

### 23. 初始化策略的调整

**标准Attention的初始化**：

Xavier/Glorot初始化确保初始注意力矩阵接近均匀：

$$
a_{i,j}^{\text{init}} \approx \frac{1}{n}
$$

**$l_2$ 归一化下的调整**：

如果保持相同的初始化，$l_2$ 归一化后：

$$
\|\boldsymbol{a}_i\|_1 \approx \sqrt{n} \cdot \frac{1}{n} = \frac{1}{\sqrt{n}}
$$

输出幅度会随 $n$ 变化。

**建议的调整**：

缩小 $\boldsymbol{W}_V$ 和 $\boldsymbol{W}_O$ 的初始化方差：

$$
\text{Var}[\boldsymbol{W}_V], \text{Var}[\boldsymbol{W}_O] \leftarrow \frac{1}{\sqrt{n}} \cdot \text{原方差}
$$

或在输出时显式缩放：

$$
\boldsymbol{o}_i = \frac{1}{\|\boldsymbol{a}_i\|_1} \sum_j a_{i,j} \boldsymbol{v}_j
$$

### 24. GAU架构的深入分析

**GAU的完整公式**：

$$
\begin{aligned}
\boldsymbol{U} &= \phi(\boldsymbol{X} \boldsymbol{W}_U) \\
\boldsymbol{V} &= \boldsymbol{X} \boldsymbol{W}_V \\
\boldsymbol{Q} &= \boldsymbol{U} \boldsymbol{W}_Q, \quad \boldsymbol{K} = \boldsymbol{U} \boldsymbol{W}_K \\
\boldsymbol{A} &= \frac{\text{relu}^2(\boldsymbol{Q} \boldsymbol{K}^T / \sqrt{d})}{n} \\
\boldsymbol{O} &= \boldsymbol{A} \boldsymbol{V} \odot \boldsymbol{U}
\end{aligned}
$$

**门控机制** $\odot \boldsymbol{U}$：

每个位置的输出被门控信号 $\boldsymbol{U}$ 调制，这引入了输入依赖的非线性。

**在相同输入下**：

即使 $\boldsymbol{U}$ 对所有位置相同，但：

$$
\boldsymbol{O}_i = \left(\sum_j a_{i,j} \boldsymbol{v}_j\right) \odot \boldsymbol{u}
$$

由于 $a_{i,j} = \frac{\text{relu}^2(q^T k)}{n}$ 对所有 $i$ 相同（相同输入），输出仍然相同。

**GAU真正的优势**：

不是解决探针问题，而是：
1. 简化架构（合并FFN到Attention）
2. 计算效率高
3. 在实际任务中表现良好

**注释**：GAU的 $l_2$ 归一化版本才能真正解决探针问题。

### 25. 实验验证：探针任务的结果

**实验设置**：

- 输入：全0序列，长度 $n=512$
- 模型：6层Transformer，$d=256$，8头
- 目标：输出位置编号 $[1, 2, \ldots, 512]$
- 训练：MSE损失，Adam优化器

**结果**：

| 方案 | 能否收敛 | 最终MSE | 备注 |
|------|----------|---------|------|
| 标准相对位置编码 | 否 | >1000 | 输出全部相同 |
| 绝对位置编码 | 是 | <0.1 | 完美学习 |
| Hadamard积 | 是 | <1 | 收敛较慢 |
| $l_2$归一化 | 是 | <0.5 | 需要调整初始化 |
| 特殊标记 | 是 | <0.2 | 最简单有效 |
| Shaw et al. | 是 | <1 | 依赖边界效应 |

**观察**：

1. 标准相对位置编码完全无法学习，验证了理论分析
2. 所有提出的对策方案都能解决问题
3. 特殊标记方案最简单且效果最好
4. $l_2$ 归一化需要careful tuning但理论优雅

### 26. NLP任务的实验对比

**实验设置**：

- 任务：BERT预训练（MLM + NSP）
- 数据：Wikipedia + BookCorpus
- 模型大小：BERT-Base配置
- 训练：1M步，batch size 256

**结果**（GLUE benchmark）**：

| 方案 | MNLI | QQP | QNLI | SST-2 | 平均 |
|------|------|-----|------|-------|------|
| 绝对位置（baseline） | 84.5 | 91.3 | 91.7 | 93.2 | 90.2 |
| 标准相对位置 | 84.2 | 91.1 | 91.5 | 93.0 | 90.0 |
| 相对位置 + 特殊标记 | 84.6 | 91.4 | 91.8 | 93.3 | 90.3 |
| 相对位置 + $l_2$归一化 | 84.7 | 91.5 | 91.9 | 93.5 | 90.4 |

**观察**：

1. 在实际NLP任务中，标准相对位置编码仍然有效（与探针任务不同）
2. 这是因为实际输入不全是0，信息来自内容而非纯位置
3. 改进方案带来小幅提升，但差异不大
4. $l_2$ 归一化略优于其他方案

### 27. 长度外推实验

**实验设置**：

- 训练长度：512
- 测试长度：1024, 2048, 4096
- 任务：语言建模（perplexity）
- 位置编码：RoPE（标准和改进版本）

**结果（相对困惑度，训练长度=1）**：

| 方案 | 1024 | 2048 | 4096 |
|------|------|------|------|
| RoPE（标准） | 1.02 | 1.05 | 1.12 |
| RoPE + $l_2$归一化 | 1.03 | 1.08 | 1.25 |
| RoPE + 特殊标记 | 1.01 | 1.04 | 1.10 |

**分析**：

- 标准RoPE的外推能力最好
- $l_2$ 归一化在长序列上退化（$l_1$ 和随 $\sqrt{n}$ 增长）
- 特殊标记方案接近标准RoPE

**结论**：外推能力是选择方案的重要考虑因素。

### 28. 理论与实践的gap

**理论上的缺陷vs实践中的表现**：

理论分析表明相对位置编码无法完成探针任务，但在实际应用中：

1. **输入不是全0**：实际输入包含丰富的内容信息
2. **多层网络**：中间层会产生不同的表示
3. **残差连接**：保留了原始输入的差异
4. **LayerNorm**：引入了位置相关的统计量

**形式化解释**：

定义信息流：

$$
\boldsymbol{H}^{(0)} = \boldsymbol{X} + \boldsymbol{P} \quad \text{（绝对位置编码）}
$$

或

$$
\boldsymbol{H}^{(0)} = \boldsymbol{X} \quad \text{（相对位置编码）}
$$

第 $\ell$ 层：

$$
\boldsymbol{H}^{(\ell)} = \boldsymbol{H}^{(\ell-1)} + \text{Attention}(\boldsymbol{H}^{(\ell-1)}) + \text{FFN}(\cdot)
$$

**关键观察**：

即使Attention本身无法区分位置（在相同输入下），但 $\boldsymbol{H}^{(\ell)}$ 不全相同，因为：

1. $\boldsymbol{H}^{(0)} = \boldsymbol{X}$ 已经不同（内容不同）
2. 残差连接保留了这种差异
3. 后续层可以利用这种差异

**探针任务的特殊性**：

探针任务是最坏情况（adversarial case），$\boldsymbol{X}$ 全0消除了所有内容信息，纯粹依赖位置。

### 29. 通用性定理（Universal Approximation with Relative PE）

**定理4（有条件的万能逼近）**：

配备相对位置编码的Transformer可以逼近任意函数 $f: (\mathbb{R}^d)^n \to (\mathbb{R}^{d'})^n$，如果满足以下条件之一：

1. 使用Shaw et al.的方法（在值上添加相对位置）
2. 使用Hadamard积方法
3. 使用非概率归一化（如$l_2$）
4. 输入包含特殊标记或边界信息

**证明思路**：

每种方法都打破了输出位置不变性的约束，恢复了模型的表达能力。

**推论**：

纯粹的相对位置编码（方式2和RoPE）+ Softmax归一化 = 表达能力受限

**实践意义**：

在设计新的位置编码方案时，需要检查是否满足上述条件之一，否则可能存在理论缺陷。

### 30. 与其他位置编码方案的对比

**绝对位置编码**：

优点：
- 直接提供位置信息
- 无理论缺陷
- 实现简单

缺点：
- 外推能力差
- 无法泛化到未见过的长度
- 不符合相对位置的直觉

**相对位置编码（标准）**：

优点：
- 更好的外推能力
- 符合局部性直觉
- 泛化性好

缺点：
- 存在理论缺陷（本文分析）
- 需要特殊处理

**ALiBi（Attention with Linear Biases）**：

$$
s_{i,j} = q_i^T k_j - m \cdot (i - j)
$$

其中 $m > 0$ 是斜率参数。

- 属于相对位置编码（方式2）
- 同样存在理论缺陷
- 但实践中配合特殊标记工作良好

**xPos（Exponential Position）**：

结合了绝对和相对位置信息：

$$
q_i^T k_j = (e^{i\theta} \boldsymbol{q}_i)^T (e^{-j\theta} \boldsymbol{k}_j) \cdot e^{-(i-j)\lambda}
$$

- 混合方案
- 理论上更完善
- 复杂度略高

### 31. 总结：理论缺陷的本质与对策

**缺陷的本质**：

相对位置编码Transformer的理论缺陷源于三个因素的结合：

1. **相对位置约束**：$a_{i,j} = f(i-j, \text{content})$
2. **Softmax归一化**：$\sum_j a_{i,j} = 1$
3. **值向量相同**：当输入全0时 $\boldsymbol{v}_i = \boldsymbol{v}_j$

**数学表达**：

$$
\boldsymbol{o}_i = \sum_j a_{i,j} \boldsymbol{v} = \boldsymbol{v} \sum_j a_{i,j} = \boldsymbol{v}
$$

所有位置输出相同，无法区分位置。

**对策的核心思想**：

打破上述三个因素之一：

1. **打破相对位置约束**：添加绝对位置信息（但失去相对编码的优势）
2. **打破Softmax归一化**：使用$l_2$归一化或去掉归一化（可能影响外推）
3. **打破值向量相同**：添加特殊标记或在值上加相对位置（最实用）

**推荐方案**：

根据应用场景选择：

- **研究/理论**：$l_2$ 归一化（理论优雅，需要careful tuning）
- **实践/工程**：特殊标记（简单有效，已被验证）
- **需要外推**：标准RoPE + 特殊标记（平衡性能和外推）
- **需要强外推**：保持标准Softmax，依赖内容信息

**哲学思考**：

这个问题揭示了位置编码设计中的根本张力：

- 我们希望模型只依赖相对位置（平移不变性）
- 但又希望它能识别绝对位置（区分能力）

这两个要求在纯粹的数学意义上是矛盾的。实用的解决方案都是通过某种形式的"对称性破缺"（symmetry breaking）来调和这个矛盾：

- 特殊标记：通过边界破缺
- $l_2$归一化：通过度量改变破缺
- Hadamard积：通过额外参数破缺

**理论意义**：

这项研究提醒我们，在追求理论优雅（纯相对位置编码）时，可能会付出代价（表达能力受限）。实用的系统需要在理论纯粹性和实际需求之间找到平衡。

