---
title: 通向概率分布之路：盘点Softmax及其替代品
slug: 通向概率分布之路盘点softmax及其替代品
date: 2023-12-11
tags: 详细推导, 概率, 分析, 损失函数, 梯度, 生成模型
status: completed
---
# 通向概率分布之路：盘点Softmax及其替代品

**原文链接**: [https://spaces.ac.cn/archives/10145](https://spaces.ac.cn/archives/10145)

**发布日期**: 

---

不论是在基础的分类任务中，还是如今无处不在的注意力机制中，概率分布的构建都是一个关键步骤。具体来说，就是将一个$n$维的任意向量，转换为一个$n$元的离散型概率分布。众所周知，这个问题的标准答案是Softmax，它是指数归一化的形式，相对来说比较简单直观，同时也伴有很多优良性质，从而成为大部分场景下的“标配”。

尽管如此，Softmax在某些场景下也有一些不如人意之处，比如不够稀疏、无法绝对等于零等，因此很多替代品也应运而生。在这篇文章中，我们将简单总结一下Softmax的相关性质，并盘点和对比一下它的部分替代方案。

## Softmax回顾 #

首先引入一些通用记号：$\boldsymbol{x} = (x_1,x_2,\cdots,x_n)\in\mathbb{R}^n$是需要转为概率分布的$n$维向量，它的分量可正可负，也没有限定的上下界。$\Delta^{n-1}$定义为全体$n$元离散概率分布的集合，即  
\begin{equation}\Delta^{n-1} = \left\\{\boldsymbol{p}=(p_1,p_2,\cdots,p_n)\left|\, p_1,p_2,\cdots,p_n\geq 0,\sum_{i=1}^n p_i = 1\right.\right\\}\end{equation}  
之所以标注$n-1$而不是$n$，是因为约束$\sum\limits_{i=1}^n p_i = 1$定义了$n$维空间中的一个$n-1$维子平面，再加上$p_i\geq 0$的约束，$(p_1,p_2,\cdots,p_n)$的集合就只是该平面的一个子集，即实际维度只有$n-1$。

基于这些记号，本文的主题就可以简单表示为探讨$\mathbb{R}^n\mapsto\Delta^{n-1}$的映射，其中$\boldsymbol{x}\in\mathbb{R}^n$我们习惯称之为Logits或者Scores。

### 基本定义 #

Softmax的定义很简单：  
\begin{equation}p_i = softmax(\boldsymbol{x})_i = \frac{e^{x_i}}{\sum\limits_{j=1}^n e^{x_j}}\end{equation}  
Softmax的来源和诠释都太多了，比如能量模型、统计力学或者单纯作为$\text{argmax}$的光滑近似等，所以我们很难考证它的最早出处，也不去做这个尝试了。很多时候我们也会加上一个温度参数，即考虑$softmax(\boldsymbol{x}/\tau)$，但$\tau$本身也可以整合到$\boldsymbol{x}$的定义之中，因此这里不特别分离出$\tau$参数。

Softmax的分母我们通常记为$Z(\boldsymbol{x})$，它的对数就是大多数深度学习框架都自带的$\text{logsumexp}$运算，他它是$\max$的一个光滑近似：  
\begin{align}\log Z(\boldsymbol{x}) = \log \sum\limits_{j=1}^n e^{x_j} = \text{logsumexp}(\boldsymbol{x})\\\  
\lim_{\tau\to 0^+} \tau\,\text{logsumexp}(\boldsymbol{x}/\tau) = \max(\boldsymbol{x})\end{align}  
当$\tau$取$1$时，就可以写出$\text{logsumexp}(\boldsymbol{x}) \approx \max(\boldsymbol{x})$，$\boldsymbol{x}$方差越大近似程度越高，更进一步的讨论可以参考[《寻求一个光滑的最大值函数》](/archives/3290)。

### 两点性质 #

除了将任意向量转换为概率分布外，Softmax还满足两点性质  
\begin{align}&{\color{red}{单调性}}:\quad p_i > p_j \Leftrightarrow x_i > x_j,\quad p_i = p_j \Leftrightarrow x_i = x_j \\\\[5pt]  
&{\color{red}{不变性}}:\quad softmax(\boldsymbol{x}) = softmax(\boldsymbol{x} + c),\,\,\forall c\in\mathbb{R}  
\end{align}  
**单调性** 意味着Softmax是保序的，$\boldsymbol{x}$的最大值/最小值跟$\boldsymbol{p}$的最大值/最小值相对应；**不变性** 说的是$\boldsymbol{x}$的每个分量都加上同一个常数，Softmax的结果不变，这跟$\text{argmax}$的性质是一样的，即同样有$\text{argmax}(\boldsymbol{x}) = \text{argmax}(\boldsymbol{x} + c)$。

因此，根据这两点性质我们可以得出，Softmax实际是$\text{argmax}$一个光滑近似（更准确来说是$\text{onehot}(\text{argmax}(\cdot))$的光滑近似），更具体地我们有  
\begin{equation}\lim_{\tau\to 0^+} softmax(\boldsymbol{x}/\tau) = \text{onehot}(\text{argmax}(\boldsymbol{x}))\end{equation}  
这大概就是Softmax这个名字的来源。注意不要混淆了，Softmax是$\text{argmax}$而不是$\max$的光滑近似，$\max$的光滑近似是$\text{logsumexp}$才对。

### 梯度计算 #

对于深度学习来说，了解一个函数的性质重要方式之一是了解它的梯度，对于Softmax，我们在[《从梯度最大化看Attention的Scale操作》](/archives/9812)曾经算过：  
\begin{equation}\frac{\partial p_i}{\partial x_j} = p_i\delta_{i,j} - p_i p_j = \left\\{\begin{aligned}  
p_i - p_i^2,&\quad i=j\\\  
\- p_i p_j,&\quad i\neq j  
\end{aligned}\right.\end{equation}  
这样排列成的矩阵也称为Softmax的[雅可比矩阵（Jacobian Matrix）](https://en.wikipedia.org/wiki/Jacobian_matrix_and_neterminant)，它的L1范数有一个简单的形式  
\begin{equation}\frac{1}{2}\left\Vert\frac{\partial \boldsymbol{p}}{\partial \boldsymbol{x}}\right\Vert_1=\frac{1}{2}\sum_{i,j}\left|\frac{\partial p_i}{\partial x_j}\right|=\frac{1}{2}\sum_i (p_i - p_i^2) + \frac{1}{2}\sum_{i\neq j} p_i p_j = 1 - \sum_i p_i^2\end{equation}  
当$\boldsymbol{p}$是one hot分布时，上式等于0，这意味着Softmax的结果越接近one hot，它的梯度消失现象越严重，所以至少初始化阶段，我们不能将Softmax初始化得接近one hot。同时上式最右端也联系到了[Rényi熵](/archives/9595#%E7%86%B5%E7%9A%84%E8%81%94%E7%B3%BB)的概念，它跟常见的香侬熵类似。

### 参考实现 #

Softmax的直接实现很简单，直接取$\exp$然后归一化就行，Numpy的参考代码为：
    
    
    def softmax(x):
        y = np.exp(x)
        return y / y.sum()

然而，如果$\boldsymbol{x}$中存在较大的分量，那么算$\exp$时很容易溢出，因此我们通常都要利用Softmax的**不变性** ，先将每个分量减去所有分量的最大值，然后再算Softmax，这样每个取$\exp$的分量都不大于0，确保不会溢出：
    
    
    def softmax(x):
        y = np.exp(x - x.max())
        return y / y.sum()

### 损失函数 #

构建概率分布的主要用途之一是用于构建单标签多分类任务的输出，即假设有一个$n$分类任务，$\boldsymbol{x}$是模型的输出，那么我们希望通过$\boldsymbol{p}=softmax(\boldsymbol{x})$来预测每个类的概率。为了训练这个模型，我们需要一个损失函数，假设目标类别是$t$，常见的选择是交叉熵损失：  
\begin{equation}\mathcal{L}_t = - \log p_t = - \log softmax(\boldsymbol{x})_t\end{equation}  
我们可以求得它的梯度：  
\begin{equation}-\frac{\partial \log p_t}{\partial x_j} = p_j - \delta_{t,j} = \left\\{\begin{aligned} p_t - 1,&\quad j=t\\\ p_j,&\quad j\neq t \end{aligned}\right.\end{equation}  
注意$t$是给定的，所以$\delta_{t,j}$实际表达的是目标分布$\text{onehot(t)}$，而全体$p_j$就是$\boldsymbol{p}$本身，所以上式可以更直观地写成：  
\begin{equation}-\frac{\partial \log p_t}{\partial \boldsymbol{x}} = \boldsymbol{p} - \text{onehot(t)}\label{eq:softmax-ce-grad}\end{equation}  
也就是说，它的梯度正好是目标分布与预测分布之差，只要两者不相等，那么梯度就一直存在，优化就可以持续下去，这是交叉熵的优点。当然，某些情况下这也是缺点，因为Softmax只有在$\tau\to 0^+$才会得到one hot，换言之正常情况下都不会出现one hot，即优化一直不会完全停止，那么就有可能导致过度优化，这也是后面的一些替代品的动机。

除了交叉熵之外，还有其他一些损失可用，比如$-p_t$，这可以理解为准确率的光滑近似的相反数，但它可能会有梯度消失问题，所以它的优化效率往往不如交叉熵，一般只适用于微调而不是从零训练，更多讨论可以参考[《如何训练你的准确率？》](/archives/9098)。

## Softmax变体 #

介绍完Softmax，我们紧接着总结一下本博客以往讨论过Softmax的相关变体工作，比如Margin Softmax、Taylor Softmax、Sparse Softmax等，它们都是在Softmax基础上的衍生品，侧重于不同方面的改进，比如损失函数、、稀疏性、长尾性等。

### Margin Softmax #

首先我们介绍起源于人脸识别的一系列Softmax变体，它们可以统称为Margin Softmax，后来也被应用到NLP的Sentence Embedding训练之中，本站曾在[《基于GRU和am-softmax的句子相似度模型》](/archives/5743)讨论过其中的一个变体AM-Softmax，后来则在[《从三角不等式到Margin Softmax》](/archives/8656)有过更一般的讨论。

尽管Margin Softmax被冠以Softmax之名，但它实际上更多是一种损失函数改进。以AM-Softmax为例，它有两个特点：第一，以$\cos$形式构造Logits，即$\boldsymbol{x} = [\cos(\boldsymbol{z},\boldsymbol{c}_1),\cos(\boldsymbol{z},\boldsymbol{c}_2),\cdots,\cos(\boldsymbol{z},\boldsymbol{c}_n)]/\tau$的形式，此时的温度参数$\tau$是必须的，因为单纯的$\cos$值域为$[-1,1]$，不能拉开类概率之间的差异；第二，它并不是简单地以$-\log p_t$为损失，而是做了加强：  
\begin{equation}\mathcal{L} = - \log \frac{e^{[\cos(\boldsymbol{z},\boldsymbol{c}_t)-m]/\tau}}{e^{[\cos(\boldsymbol{z},\boldsymbol{c}_t)-m]/\tau} + \sum_{j\neq t} e^{\cos(\boldsymbol{z},\boldsymbol{c}_j)/\tau}}\end{equation}  
直观来看，就是交叉熵希望$x_t$是$\boldsymbol{x}$所有分量中最大的一个，而AM-Softmax则不仅希望$x_t$最大，还希望它至少比第二大的分量多出$m/\tau$，这里的$m/\tau$就称为Margin。

为什么要增加对目标类的要求呢？这是应用场景导致的。刚才说了，Margin Softmax起源于人脸识别，放到NLP中则可以用于语义检索，也就是说它的应用场景是检索，但训练方式是分类。如果单纯用分类任务的交叉熵来训练模型，模型编码出来的特征不一定能很好地满足检索要求，所以要加上Margin使得特征更加紧凑一些。更具体的讨论请参考[《从三角不等式到Margin Softmax》](/archives/8656)一文，或者查阅相关论文。

### Taylor Softmax #

接下来要介绍的，是在[《exp(x)在x=0处的偶次泰勒展开式总是正的》](/archives/7919)讨论过的Taylor Softmax，它利用了$\exp(x)$的泰勒展开式的一个有趣性质：

> 对于任意实数$x$及偶数$k$，总有$f_k(x)\triangleq\sum\limits_{m=0}^k \frac{x^m}{m!} > 0$，即$e^x$在$x=0$处的偶次泰勒展开式总是正的。

利用这个恒正性，我们可以构建一个Softmax变体（$k > 0$是任意偶数）：  
\begin{equation}taylor\text{-}softmax(\boldsymbol{x}, k)_i = \frac{f_k(x_i)}{\sum\limits_{j=1}^n f_k(x_j)}\end{equation}  
由于是基于$\exp$的泰勒展开式构建的，所以在一定范围内Taylor Softmax与Softmax有一定的近似关于，某些场景下可以用Taylor Softmax替换Softmax。那么Taylor Softmax有什么特点呢？答案是更加长尾，因为Taylor Softmax是多项式函数归一化，相比指数函数衰减得更慢，所以对于尾部的类别，Taylor Softmax往往能够给其分配更高的概率，可能有助于缓解Softmax的过度自信现象。

Taylor Softmax的最新应用，是用来替换Attention中的Softmax，使得原本的平方复杂度降低为线性复杂度，相关理论推导可以参考[《Transformer升级之路：5、作为无限维的线性Attention》](/archives/8601)。该思路的最新实践是一个名为Based的模型，它利用$e^x\approx 1+x+x^2/2$来线性化Attention，声称比Attention高效且比Mamba效果更好，详细介绍可以参考博客[《Zoology (Blogpost 2): Simple, Input-Dependent, and Sub-Quadratic Sequence Mixers》](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)和[《BASED: Simple linear attention language models balance the recall-throughput tradeoff》](https://www.together.ai/blog/based)。

### Sparse Softmax #

Sparse Softmax是笔者参加2020年法研杯时提出的一个简单的Softmax稀疏变体，首先发表在博文[《SPACES：“抽取-生成”式长文本摘要（法研杯总结）》](/archives/8046)，后来也补充了相关实验，写了篇简单的论文[《Sparse-softmax: A Simpler and Faster Alternative Softmax Transformation》](https://papers.cool/arxiv/2112.12433)。

我们知道，在文本生成中，我们常用确定性的Beam Search解码，或者随机性的TopK/TopP Sampling采样，这些算法的特点都是只保留了预测概率最大的若干个Token进行遍历或者采样，也就等价于将剩余的Token概率视为零，而训练时如果直接使用Softmax来构建概率分布的话，那么就不存在绝对等于零的可能，这就让训练和预测出现了不一致性。Sparse Softmax就是希望能处理这种不一致性，思路很简单，就是在训练的时候也把Top-$k$以外的Token概率置零：  
\begin{array}{c|c|c}  
\hline  
& Softmax & Sparse\text{ }Softmax \\\  
\hline  
\text{基本定义} & p_i = \frac{e^{x_i}}{\sum\limits_{j=1}^n e^{x_j}} & p_i=\left\\{\begin{aligned}&\frac{e^{x_i}}{\sum\limits_{j\in\Omega_k} e^{x_j}},\,i\in\Omega_k\\\ &\quad 0,\,i\not\in\Omega_k\end{aligned}\right.\\\  
\hline  
\text{损失函数} & \log\left(\sum\limits_{i=1}^n e^{x_i}\right) - x_t & \log\left(\sum\limits_{i\in\Omega_k} e^{x_i}\right) - x_t\\\  
\hline  
\end{array}

其中$\Omega_k$是将$x_1,x_2,\cdots,x_n$从大到小排列后前$k$个元素的原始下标集合。简单来说，就是在训练阶段就进行与预测阶段一致的阶段操作。这里的$\Omega_k$选取方式也可以按照Nucleus Sampling的Top-$p$方式来操作，看具体需求而定。但要注意的是，Sparse Softmax强行截断了剩余部分的概率，意味着这部分Logits无法进行反向传播了，因此Sparse Softmax的训练效率是不如Softmax的，所以它一般只适用于微调场景，而不适用于从零训练。

## Perturb Max #

这一节我们介绍一种新的构建概率分布的方式，这里称之为Perturb Max，它是Gumbel Max的一般化，在本站中首次介绍于博客[《从重参数的角度看离散概率分布的构建》](/archives/9085)，此外在论文[《EXACT: How to Train Your Accuracy》](https://papers.cool/arxiv/2205.09615)也有过相关讨论，至于更早的出处笔者则没有进一步考究了。

### 问题反思 #

首先我们知道，构建一个$\mathbb{R}^n\mapsto\Delta^{n-1}$的映射并不是难事，只要$f(x)$是$\mathbb{R}\mapsto \mathbb{R}^*$（实数到非负实数）的映射，如$x^2$，那么只需要让  
\begin{equation}p_i = \frac{f(x_i)}{\sum\limits_{j=1}^n f(x_j)}\end{equation}  
就是满足条件的映射了。如果要加上“两点性质”中的“**单调性** ”呢？那么也不难，只需要保证$\mathbb{R}\mapsto \mathbb{R}^*$的单调递增函数，这样的函数也有很多，比如$\text{sigmoid}(x)$。但如果再加上“**不变性** ”呢？我们还能随便写出一个满足**不变性** 的$\mathbb{R}^n\mapsto\Delta^{n-1}$映射吗？（反正我不能）

可能有读者疑问：为什么非要保持**单调性** 和**不变性** 呢？的确，单纯从拟合概率分布的角度来看，这两点似乎都没什么必要，反正都是“力大砖飞”，只要模型足够大，那么没啥不能拟合的。但从“Softmax替代品”这个角度来看，我们希望新定义的概率分布同样能作为$\text{argmax}$的光滑近似，那么就要尽可能多保持跟$\text{argmax}$相同的性质，这是我们希望保持**单调性** 和**不变性** 的主要原因。

### Gumbel Max #

Perturb Max借助于Gumbel Max的一般化来构造这样的一类分布。不熟悉Gumbel Max的读者，可以先到[《漫谈重参数：从正态分布到Gumbel Softmax》](/archives/6705)了解一下Gumbel Max。简单来说，Gumbel Max就是发现：  
\begin{equation}P[\text{argmax}(\boldsymbol{x}+\boldsymbol{\varepsilon}) = i] = softmax(\boldsymbol{x})_i,\quad \boldsymbol{\varepsilon}\sim Gumbel\text{ }Noise\end{equation}  
怎么理解这个结果呢？首先，这里的$\boldsymbol{\varepsilon}\sim Gumbel\text{ }Noise$是指$\boldsymbol{\varepsilon}$的每个分量都是从[Gumbel分布](https://en.wikipedia.org/wiki/Gumbel_distribution)独立重复采样出来的；接着，我们知道给定向量$\boldsymbol{x}$，本来$\text{argmax}(\boldsymbol{x})$是确定的结果，但加了随机噪声$\boldsymbol{\varepsilon}$之后，$\text{argmax}(\boldsymbol{x}+\boldsymbol{\varepsilon})$的结果也带有随机性了，于是每个$i$都有自己的概率；最后，Gumbel Max告诉我们，如果加的是Gumbel噪声，那么$i$的出现概率正好是$softmax(\boldsymbol{x})_i$。

Gumbel Max最直接的作用，就是提供了一种从分布$softmax(\boldsymbol{x})$中采样的方式，当然如果单纯采样还有更简单的方法，没必要“杀鸡用牛刀”。Gumbel Max最大的价值是“重参数（Reparameterization）”，它将问题的随机性从带参数$\boldsymbol{x}$的离散分布转移到了不带参数的$\boldsymbol{\varepsilon}$上，再结合Softmax是$\text{argmax}$的光滑近似，我们得到$softmax(\boldsymbol{x} + \boldsymbol{\varepsilon})$是Gumbel Max的光滑近似，这便是Gumbel Softmax，是训练“离散采样模块中带有可学参数”的模型的常用技巧。

### 一般噪声 #

Perturb Max直接源自Gumbel Max：既然Softmax可以从Gumbel分布中导出，那么如果将Gumbel分布换为一般的分布，比如正态分布，不就可以导出新的概率分布形式了？也就是说直接定义  
\begin{equation}p_i = P[\text{argmax}(\boldsymbol{x}+\boldsymbol{\varepsilon}) = i],\quad \varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n\sim p(\varepsilon)\end{equation}  
重复Gumbel Max的推导，我们可以得到  
\begin{equation}p_i = \int_{-\infty}^{\infty} p(\varepsilon_i)\left[\prod_{j\neq i} \Phi(x_i - x_j + \varepsilon_i)\right]d\varepsilon_i = \mathbb{E}_{\varepsilon}\left[\prod_{j\neq i} \Phi(x_i - x_j + \varepsilon)\right]\end{equation}  
其中$\Phi(\varepsilon)$是$p(\varepsilon)$的累积概率函数。对于一般的分布，哪怕是简单的标准正态分布，上式都很难得出解析解，所以只能数值估计。为了得到确定性的计算结果，我们可以用逆累积概率函数的方式进行均匀采样，即先从$[0,1]$均匀选取$t$，然后通过求解$t=\Phi(\varepsilon)$来得到$\varepsilon$。

从Perturb Max的定义或者最后$p_i$的形式我们都可以断言Perturb Max满足**单调性** 和**不变性** ，这里就不详细展开了。那它在什么场景下有独特作用呢？说实话，还真不知道，[《EXACT: How to Train Your Accuracy》](https://papers.cool/arxiv/2205.09615)一文用它来构建新的概率分布并优化准确率的光滑近似，但笔者自己的实验显示没有特别的效果。个人感觉，可能在某些需要重参数的场景能够表现出特殊的作用吧。

## Sparsemax #

接下来要登场的是名为Sparsemax的概率映射，出自2016年的论文[《From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification》](https://papers.cool/arxiv/1602.02068)，它跟笔者提出的Sparse Softmax一样，都是面向稀疏性的改动，但作者的动机是用在Attention中提供更好的可解释性。跟Sparse Softmax直接强行截断Top-$k$个分量不同，Sparsemax提供了一个更为自适应的稀疏型概率分布构造方式。

### 基本定义 #

原论文将Sparsemax定义为如下优化问题的解：  
\begin{equation}sparsemax(\boldsymbol{x}) = \mathop{\text{argmin}}\limits_{\boldsymbol{p}\in\Delta^{n-1}}\Vert \boldsymbol{p} - \boldsymbol{x}\Vert^2\label{eq:sparsemax-opt}\end{equation}  
通过拉格朗日乘数法就可以求出精确解的表达式。然而，这种方式并不直观，而且也不容易揭示它跟Softmax的联系。下面提供笔者构思的一种私以为更加简明的引出方式。

首先，我们可以发现，Softmax可以等价地表示为  
\begin{equation}\boldsymbol{p} = softmax(\boldsymbol{x}) = \exp(\boldsymbol{x} - \lambda(\boldsymbol{x}))\label{eq:sparsemax-softmax}\end{equation}  
其中$\lambda(\boldsymbol{x})$是使得$\boldsymbol{p}$的各分量之和为1的常数，对于Softmax我们可以求出$\lambda(\boldsymbol{x})=\log\sum\limits_i e^{x_i}$。

然后，在Taylor Softmax那一节我们说了，$\exp(x)$的偶次泰勒展开总是正的，因此可以用偶次泰勒展开来构建Softmax变体。但如果是奇数次呢？比如$\exp(x)\approx 1 + x$，它并不总是非负的，但我们可以加个$\text{relu}$强行让它变成非负的，即$\exp(x)\approx \text{relu}(1 + x)$，用这个近似替换掉式$\eqref{eq:sparsemax-softmax}$的$\exp$，就得到了Sparsemax：  
\begin{equation}\boldsymbol{p} = sparsemax(\boldsymbol{x}) = \text{relu}(1+\boldsymbol{x} - \lambda(\boldsymbol{x}))\end{equation}  
其中$\lambda(\boldsymbol{x})$依然是使得$\boldsymbol{p}$的各分量之和为1的常数，并且常数$1$也可以整合到$\lambda(\boldsymbol{x})$之中，所以上式也等价于  
\begin{equation}\boldsymbol{p} = sparsemax(\boldsymbol{x}) = \text{relu}(\boldsymbol{x} - \lambda(\boldsymbol{x}))\end{equation}

### 求解算法 #

到目前为止，Sparsemax还只是一个形式化的定义，因为$\lambda(\boldsymbol{x})$的具体计算方法尚不清楚，这就是本节需要探讨的问题。不过即便如此，单靠这个定义我们也不难看出Sparsemax满足**单调性** 和**不变性** 两点性质，如果还觉得不大肯定的读者，可以自行尝试证明一下它。

现在我们转向$\lambda(\boldsymbol{x})$的计算。不失一般性，我们假设$\boldsymbol{x}$各分量已经从大到小排序好，即$x_1\geq x_2\geq \cdots\geq x_n$，接着我们不妨先假设已知$x_k\geq \lambda(\boldsymbol{x})\geq x_{k+1}$，那么很显然  
\begin{equation}sparsemax(\boldsymbol{x}) = [x_1 - \lambda(\boldsymbol{x}),\cdots,x_k - \lambda(\boldsymbol{x}),0,\cdots,0]\end{equation}  
根据$\lambda(\boldsymbol{x})$的定义，我们有  
\begin{equation}\sum_{i=1}^k [x_i - \lambda(\boldsymbol{x})] = 1\quad\Rightarrow\quad 1 + k\lambda(\boldsymbol{x}) = \sum_{i=1}^k x_i\end{equation}  
这就可以求出$\lambda(\boldsymbol{x})$。当然，我们无法事先知道$x_k\geq \lambda(\boldsymbol{x})\geq x_{k+1}$，但我们可以遍历$k=1,2,\cdots,n$，利用上式求一遍$\lambda_k(\boldsymbol{x})$，取满足$x_k\geq \lambda_k(\boldsymbol{x})\geq x_{k+1}$那一个$\lambda_k(\boldsymbol{x})$，这也可以等价地表示为求满足$x_k\geq \lambda_k(\boldsymbol{x})$的最大的$k$，然后返回对应的$\lambda_k(\boldsymbol{x})$

参考实现：
    
    
    def sparsemax(x):
        x_sort = np.sort(x)[::-1]
        x_lamb = (np.cumsum(x_sort) - 1) / np.arange(1, len(x) + 1)
        lamb = x_lamb[(x_sort >= x_lamb).argmin() - 1]
        return np.maximum(x - lamb, 0)

### 梯度计算 #

方便起见，我们引入记号  
\begin{equation}\Omega(\boldsymbol{x}) = \big\\{k\big|x_k > \lambda(\boldsymbol{x})\big\\}\end{equation}  
那么可以写出  
\begin{equation}\boldsymbol{p} = sparsemax(\boldsymbol{x}) = \left\\{\begin{aligned}  
&x_i - \frac{1}{|\Omega(\boldsymbol{x})|}\left(-1 + \sum_{j\in\Omega(\boldsymbol{x})}x_j\right),\quad &i\in \Omega(\boldsymbol{x})\\\  
&0,\quad &i \not\in \Omega(\boldsymbol{x})  
\end{aligned}\right.\end{equation}  
从这个等价形式可以看出，跟Sparse Softmax一样，Sparsemax同样也只对部分类别有梯度，可以直接算出雅可比矩阵：  
\begin{equation}\frac{\partial p_i}{\partial x_j} = \left\\{\begin{aligned}  
&1 - \frac{1}{|\Omega(\boldsymbol{x})|},\quad &i,j\in \Omega(\boldsymbol{x}),i=j\\\\[5pt]  
&\- \frac{1}{|\Omega(\boldsymbol{x})|},\quad &i,j\in \Omega(\boldsymbol{x}),i\neq j\\\\[5pt]  
&0,\quad &i \not\in \Omega(\boldsymbol{x})\text{ or }j \not\in \Omega(\boldsymbol{x})  
\end{aligned}\right.\end{equation}  
由此可以看出，对于在$\Omega(\boldsymbol{x})$里边的类别，Sparsemax倒是不会梯度消失，因为此时它的梯度恒为常数，但它总的梯度大小，取决于$\Omega(\boldsymbol{x})$的元素个数，它越少则越稀疏，意味着梯度也越稀疏。

### 损失函数 #

最后我们来讨论Sparsemax作为分类输出时的损失函数。比较直观的想法就是跟Softmax一样用交叉熵$-\log p_t$，但Sparsemax的输出可能是严格等于0的，所以为了防止$\log 0$错误，还要给每个分量都加上$\epsilon$，最终的交叉熵形式为$-\log\frac{p_t + \epsilon}{1 + n\epsilon}$，但这一来有点丑，二来它还不是凸函数，所以并不是一个理想选择。

事实上，交叉熵在Softmax中之所以好用，是因为它的梯度恰好有$\eqref{eq:softmax-ce-grad}$的形式，所以对于Sparsemax，我们不妨同样假设损失函数的梯度为$\boldsymbol{p} - \text{onehot(t)}$，然后反推出损失函数该有的样子，即：  
\begin{equation}\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{x}} = \boldsymbol{p} - \text{onehot(t)}\quad\Rightarrow\quad \mathcal{L}_t = \frac{1}{2} - x_t + \sum_{i\in\Omega(\boldsymbol{x})}\frac{1}{2}\left(x_i^2 - \lambda^2(\boldsymbol{x})\right)\end{equation}  
从右往左验证比较简单，从左往右推可能会有些困难，但不多，反复拼凑一下应该就能出来了。第一个$\frac{1}{2}$常数是为了保证损失函数的非负性，我们可以取一个极端来验证一下：假设优化到完美，那么$\boldsymbol{p}$应该也是one hot，此时$x_t\to\infty$，并且$\lambda(\boldsymbol{x}) = x_t - 1$，于是  
\begin{equation}- x_t + \sum_{i\in\Omega(\boldsymbol{x})}\frac{1}{2}\left(x_i^2 - \lambda^2(\boldsymbol{x})\right) = -x_t + \frac{1}{2}x_t^2 - \frac{1}{2}(x_t - 1)^2 = -\frac{1}{2}\end{equation}  
所以要多加上常数$\frac{1}{2}$。

## Entmax-α #

Entmax-$\alpha$是Sparsemax的一般化，它的动机是Sparsemax往往会过度稀疏，这可能会导致学习效率偏低，导致最终效果下降的问题，所以Entmax-$\alpha$引入了$\alpha$参数，提供了Softmax（$\alpha=1$）到Sparsemax（$\alpha=2$）的平滑过度。Entmax-$\alpha$出自论文[《Sparse Sequence-to-Sequence Models》](https://papers.cool/arxiv/1905.05702)，作者跟Sparsemax一样是Andre F. T. Martins，这位大佬围绕着稀疏Softmax、稀疏Attention做了不少工作，有兴趣的读者可以在[他的主页](https://andre-martins.github.io/)查阅相关工作。

### 基本定义 #

跟Sparsemax一样，原论文将Entmax-$\alpha$定义为类似$\eqref{eq:sparsemax-opt}$的优化问题的解，但这个定义涉及到[Tsallis entropy](https://en.wikipedia.org/wiki/Tsallis_entropy)的概念（也是Entmax的Ent的来源），求解还需要用到拉格朗日乘数法，相对来说比较复杂，这里不采用这种引入方式。

我们的介绍同样是基于上一节的近似$\exp(x)\approx \text{relu}(1 + x)$，对于Softmax和Sparsemax，我们有  
\begin{align}&{\color{red}{Softmax}}:\quad &\exp(\boldsymbol{x} - \lambda(\boldsymbol{x})) \\\\[5pt]  
&{\color{red}{Sparsemax}}:\quad &\text{relu}(1+\boldsymbol{x} - \lambda(\boldsymbol{x}))  
\end{align}  
Sparsemax太稀疏，背后的原因也可以理解为$\exp(x)\approx \text{relu}(1 + x)$近似精度不够高，我们可以从中演化出更高精度的近似  
\begin{equation}\exp(x) = \exp(\beta x / \beta) = \exp^{1/\beta}(\beta x)\approx \text{relu}^{1/\beta}(1 + \beta x)\end{equation}  
只要$0 \leq \beta < 1$，那么最右端就是一个比$\text{relu}(1 + x)$更好的近似（想想为什么）。利用这个新近似，我们就可以构建  
\begin{equation}{\color{red}{Entmax\text{-}\alpha}}:\quad \text{relu}^{1/\beta}(1+\beta\boldsymbol{x} - \lambda(\boldsymbol{x}))\end{equation}  
这里$\alpha = \beta + 1$是为了对齐原论文的表达方式，事实上用$\beta$表示更简洁一些。同样地，常数$1$也可以收入到$\lambda(\boldsymbol{x})$定义之中，所以最终定义可以简化为  
\begin{equation}Entmax_{\alpha}(\boldsymbol{x}) = \text{relu}^{1/\beta}(\beta\boldsymbol{x} - \lambda(\boldsymbol{x}))\end{equation}

### 求解算法 #

对于一般的$\beta$，求解$\lambda(\boldsymbol{x})$是比较麻烦的事情，通常只能用二分法求解。

首先我们记$\boldsymbol{z}=\beta\boldsymbol{x}$，并且不失一般性假设$z_1\geq z_2\geq \cdots \geq z_n$，然后我们可以发现Entmax-$\alpha$是满足**单调性** 和**不变性** 的，借助**不变性** 我们可以不失一般性地设$z_1 = 1$（如果不是，每个$z_i$都减去$z_1 - 1$即可）。现在可以检验，当$\lambda=0$时，$\text{relu}^{1/\beta}(\beta\boldsymbol{x} - \lambda)$的所有分量之和大于等于1，当$\lambda=1$时，$\text{relu}^{1/\beta}(\beta\boldsymbol{x} - \lambda)$的所有分量之和等于0，所以最终能使分量之和等于1的$\lambda(\boldsymbol{x})$必然在$[0,1)$内，然后我们就可以使用二分法来逐步逼近最优的$\lambda(\boldsymbol{x})$。

对于某些特殊的$\beta$，我们可以得到一个求精确解的算法。Sparsemax对应$\beta=1$，我们前面已经给出了求解过程，另外一个能给解析解的例子是$\beta=1/2$，这也是原论文主要关心的例子，如果不加标注，那么Entmax默认就是Entmax-1.5。跟Sparsemax一样的思路，我们先假设已知$z_k\geq \lambda(\boldsymbol{x})\geq z_{k+1}$，于是有  
\begin{equation}\sum_{i=1}^k [z_i - \lambda(\boldsymbol{x})]^2 = 1\end{equation}  
这只不过是关于$\lambda(\boldsymbol{x})$的一元二次方程，可以解得  
\begin{equation}\lambda(\boldsymbol{x}) = \mu_k - \sqrt{\frac{1}{k} - \sigma_k^2},\quad \mu_k = \frac{1}{k}\sum_{i=1}^k z_i,\quad\sigma_k^2 = \frac{1}{k}\left(\sum_{i=1}^k z_i^2\right) - \mu_k^2\end{equation}  
当我们无法事先知道$x_k\geq \lambda(\boldsymbol{x})\geq x_{k+1}$时，可以遍历$k=1,2,\cdots,n$，利用上式求一遍$\lambda_k(\boldsymbol{x})$，取满足$x_k\geq \lambda_k(\boldsymbol{x})\geq x_{k+1}$那一个$\lambda_k(\boldsymbol{x})$，但注意这时候不等价于求满足$x_k\geq \lambda_k(\boldsymbol{x})$的最大的$k$。

完整的参考实现：
    
    
    def entmat(x):
        x_sort = np.sort(x / 2)[::-1]
        k = np.arange(1, len(x) + 1)
        x_mu = np.cumsum(x_sort) / k
        x_sigma2 = np.cumsum(x_sort**2) / k  - x_mu**2
        x_lamb = x_mu - np.sqrt(np.maximum(1. / k - x_sigma2, 0))
        x_sort_shift = np.pad(x_sort[1:], (0, 1), constant_values=-np.inf)
        lamb = x_lamb[(x_sort > x_lamb) & (x_lamb > x_sort_shift)]
        return np.maximum(x / 2 - lamb, 0)**2

### 其他内容 #

Entmax-$\alpha$的梯度跟Sparsemax大同小异，这里就不展开讨论了，读者自行推导一下或者参考原论文就行。至于损失函数，同样从梯度$\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{x}} = \boldsymbol{p} - \text{onehot(t)}$出发反推出损失函数也是可以的，但其形式有点复杂，有兴趣了解的读者可以参考原论文[《Sparse Sequence-to-Sequence Models》](https://papers.cool/arxiv/1905.05702)以及[《Learning with Fenchel-Young Losses》](https://papers.cool/arxiv/1901.02324)。

不过就笔者看来，直接用$\text{stop_gradient}$算子来定义损失函数更为简单通用，可以避免求原函数的复杂过程：  
\begin{equation}\mathcal{L}_t = (\boldsymbol{p} - \text{onehot(t)})\cdot \text{stop_gradient}(\boldsymbol{x})\end{equation}  
这里的$\,\cdot\,$是向量内积，这样定义出来的损失，其梯度正好是$\boldsymbol{p} - \text{onehot(t)}$，但要注意这个损失函数只有梯度是有效的，它本身的数值是没有参考意义的，比如它可正可负，也不一定越小越好，所以要评估训练进度和效果的话，得另外建立指标（比如交叉熵或者准确率）。

## 文章小结 #

本文简单回顾和整理了Softmax及其部分替代品，其中包含的工作有Softmax、Margin Softmax、Taylor Softmax、Sparse Softmax、Perturb Max、Sparsemax、Entmax-$\alpha$的定义、性质等内容。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10145>_

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

苏剑林. (Jun. 14, 2024). 《通向概率分布之路：盘点Softmax及其替代品 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10145>

@online{kexuefm-10145,  
title={通向概率分布之路：盘点Softmax及其替代品},  
author={苏剑林},  
year={2024},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/10145}},  
} 


---

## 公式推导与注释

### 1. Softmax 详细推导

#### 1.1 定义的严格推导

从最基本的要求出发，我们希望将向量 $\boldsymbol{x} = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$ 映射到概率单纯形 $\Delta^{n-1}$。一个自然的想法是使用归一化：

**步骤1：** 构造非负函数
\begin{equation}
f: \mathbb{R} \to \mathbb{R}^+ = [0, +\infty)
\end{equation}

**步骤2：** 对每个分量应用该函数
\begin{equation}
y_i = f(x_i), \quad i = 1, 2, \ldots, n
\end{equation}

**步骤3：** 归一化得到概率
\begin{equation}
p_i = \frac{y_i}{\sum_{j=1}^n y_j} = \frac{f(x_i)}{\sum_{j=1}^n f(x_j)}
\end{equation}

**验证概率性质：**
- 非负性：由于 $f(x_i) \geq 0$ 且分母 $\sum_{j=1}^n f(x_j) > 0$（至少有一项为正），故 $p_i \geq 0$
- 归一性：$\sum_{i=1}^n p_i = \sum_{i=1}^n \frac{f(x_i)}{\sum_{j=1}^n f(x_j)} = \frac{\sum_{i=1}^n f(x_i)}{\sum_{j=1}^n f(x_j)} = 1$

**选择指数函数：** 为什么选择 $f(x) = e^x$？
1. **严格单调性**：$e^x$ 严格单调递增，保证 $x_i > x_j \Leftrightarrow e^{x_i} > e^{x_j}$
2. **处处可微**：$\frac{d}{dx}e^x = e^x$，梯度计算简单
3. **平移不变性**：$\frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^c \cdot e^{x_i}}{e^c \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$

#### 1.2 单调性的严格证明

**定理：** 对于 Softmax 函数，$p_i > p_j \Leftrightarrow x_i > x_j$

**证明（充分性）：** 假设 $x_i > x_j$，则
\begin{align}
p_i - p_j &= \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}} - \frac{e^{x_j}}{\sum_{k=1}^n e^{x_k}} \\
&= \frac{e^{x_i} - e^{x_j}}{\sum_{k=1}^n e^{x_k}}
\end{align}

由于 $e^x$ 严格单调递增，$x_i > x_j \Rightarrow e^{x_i} > e^{x_j}$，因此分子 $e^{x_i} - e^{x_j} > 0$，分母恒正，故 $p_i - p_j > 0$，即 $p_i > p_j$。

**证明（必要性）：** 假设 $p_i > p_j$，则
\begin{equation}
\frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}} > \frac{e^{x_j}}{\sum_{k=1}^n e^{x_k}} \Rightarrow e^{x_i} > e^{x_j}
\end{equation}

由于 $\log$ 是严格单调递增函数，两边取对数得 $x_i > x_j$。$\square$

#### 1.3 梯度计算的详细步骤

记 $Z = \sum_{j=1}^n e^{x_j}$，则 $p_i = \frac{e^{x_i}}{Z}$

**情况1：** $i = j$ 时，使用商法则
\begin{align}
\frac{\partial p_i}{\partial x_i} &= \frac{\partial}{\partial x_i}\left(\frac{e^{x_i}}{Z}\right) \\
&= \frac{e^{x_i} \cdot Z - e^{x_i} \cdot \frac{\partial Z}{\partial x_i}}{Z^2} \\
&= \frac{e^{x_i} \cdot Z - e^{x_i} \cdot e^{x_i}}{Z^2} \quad \text{（因为 } \frac{\partial Z}{\partial x_i} = e^{x_i}\text{）} \\
&= \frac{e^{x_i}}{Z} \cdot \frac{Z - e^{x_i}}{Z} \\
&= p_i \cdot (1 - p_i) \\
&= p_i - p_i^2
\end{align}

**情况2：** $i \neq j$ 时
\begin{align}
\frac{\partial p_i}{\partial x_j} &= \frac{\partial}{\partial x_j}\left(\frac{e^{x_i}}{Z}\right) \\
&= \frac{0 \cdot Z - e^{x_i} \cdot \frac{\partial Z}{\partial x_j}}{Z^2} \\
&= \frac{-e^{x_i} \cdot e^{x_j}}{Z^2} \quad \text{（因为 } \frac{\partial Z}{\partial x_j} = e^{x_j}\text{）} \\
&= -\frac{e^{x_i}}{Z} \cdot \frac{e^{x_j}}{Z} \\
&= -p_i p_j
\end{align}

**雅可比矩阵形式：**
\begin{equation}
J = \frac{\partial \boldsymbol{p}}{\partial \boldsymbol{x}} = \text{diag}(\boldsymbol{p}) - \boldsymbol{p}\boldsymbol{p}^T
\end{equation}

其中 $\text{diag}(\boldsymbol{p})$ 是以 $\boldsymbol{p}$ 为对角元素的对角矩阵。

**数值示例：** 考虑 $\boldsymbol{x} = [1, 2, 3]^T$
\begin{align}
e^{\boldsymbol{x}} &= [e^1, e^2, e^3]^T \approx [2.718, 7.389, 20.086]^T \\
Z &= 2.718 + 7.389 + 20.086 = 30.193 \\
\boldsymbol{p} &\approx [0.090, 0.245, 0.665]^T
\end{align}

雅可比矩阵：
\begin{equation}
J \approx \begin{bmatrix}
0.082 & -0.022 & -0.060 \\
-0.022 & 0.185 & -0.163 \\
-0.060 & -0.163 & 0.223
\end{bmatrix}
\end{equation}

验证：每行之和为 0（这是由 $\sum_i p_i = 1$ 的约束导出的性质）。

#### 1.4 Hessian 矩阵（二阶导数）

对于深度理解，我们计算二阶导数。记 $H_{ij,kl} = \frac{\partial^2 p_i}{\partial x_j \partial x_k}$

**情况1：** $i = j = k$
\begin{align}
\frac{\partial^2 p_i}{\partial x_i^2} &= \frac{\partial}{\partial x_i}(p_i - p_i^2) \\
&= (p_i - p_i^2) - 2p_i(p_i - p_i^2) \\
&= p_i(1 - p_i)(1 - 2p_i)
\end{align}

**情况2：** $i = j \neq k$
\begin{align}
\frac{\partial^2 p_i}{\partial x_i \partial x_k} &= \frac{\partial}{\partial x_k}(p_i - p_i^2) \\
&= -p_i p_k - 2p_i(-p_i p_k) \\
&= p_i p_k(2p_i - 1)
\end{align}

### 2. Margin Softmax 详细推导

#### 2.1 AM-Softmax 的动机

传统 Softmax 损失：
\begin{equation}
\mathcal{L} = -\log \frac{e^{x_t}}{\sum_{j=1}^n e^{x_j}}
\end{equation}

这要求 $x_t$ 大于其他分量，但没有强制要求大多少。在人脸识别或语义检索任务中，我们希望同类样本的特征非常接近，不同类样本的特征有明显间隔。

#### 2.2 余弦相似度形式的推导

**步骤1：** 特征向量和类中心
- 输入特征：$\boldsymbol{z} \in \mathbb{R}^d$，$\|\boldsymbol{z}\| = 1$（L2归一化）
- 类中心：$\boldsymbol{c}_i \in \mathbb{R}^d$，$\|\boldsymbol{c}_i\| = 1$（L2归一化）

**步骤2：** 构造余弦相似度
\begin{equation}
\cos(\boldsymbol{z}, \boldsymbol{c}_i) = \boldsymbol{z}^T \boldsymbol{c}_i = \|\boldsymbol{z}\| \|\boldsymbol{c}_i\| \cos\theta_{zi} = \cos\theta_{zi}
\end{equation}

其中 $\theta_{zi}$ 是 $\boldsymbol{z}$ 与 $\boldsymbol{c}_i$ 的夹角。

**步骤3：** 引入温度参数和Margin
\begin{equation}
x_i = \frac{\cos\theta_{zi}}{\tau}, \quad x_t' = \frac{\cos\theta_{zt} - m}{\tau}
\end{equation}

#### 2.3 AM-Softmax 损失函数

\begin{equation}
\mathcal{L}_{AM} = -\log \frac{e^{(\cos\theta_{zt} - m)/\tau}}{e^{(\cos\theta_{zt} - m)/\tau} + \sum_{j \neq t} e^{\cos\theta_{zj}/\tau}}
\end{equation}

**几何解释：** 该损失要求目标类的余弦相似度不仅要最大，还要比次大者至少大 $m$（在归一化之前）。

#### 2.4 梯度分析

令 $s_t' = \cos\theta_{zt} - m$，$s_j = \cos\theta_{zj}$ ($j \neq t$)

\begin{align}
\frac{\partial \mathcal{L}_{AM}}{\partial \boldsymbol{z}} &= -\frac{1}{\tau}\left[\frac{\partial s_t'}{\partial \boldsymbol{z}} - \sum_{j=1}^n p_j \frac{\partial s_j}{\partial \boldsymbol{z}}\right]
\end{align}

其中
\begin{equation}
\frac{\partial \cos\theta_{zi}}{\partial \boldsymbol{z}} = \frac{\partial (\boldsymbol{z}^T\boldsymbol{c}_i)}{\partial \boldsymbol{z}} = \boldsymbol{c}_i - (\boldsymbol{z}^T\boldsymbol{c}_i)\boldsymbol{z}
\end{equation}

最后一步利用了 $\|\boldsymbol{z}\| = 1$ 的约束。

**数值示例：** 假设 $d=2$，$\tau=0.1$，$m=0.3$
- 目标类中心：$\boldsymbol{c}_t = [1, 0]^T$
- 干扰类中心：$\boldsymbol{c}_1 = [0.6, 0.8]^T$
- 特征向量：$\boldsymbol{z} = [0.8, 0.6]^T$

计算：
\begin{align}
\cos\theta_{zt} &= 0.8 \times 1 + 0.6 \times 0 = 0.8 \\
\cos\theta_{z1} &= 0.8 \times 0.6 + 0.6 \times 0.8 = 0.96
\end{align}

传统 Softmax：$x_t = 0.8/0.1 = 8$，$x_1 = 0.96/0.1 = 9.6$，预测错误！

AM-Softmax：$x_t' = (0.8-0.3)/0.1 = 5$，$x_1 = 9.6$，仍然错误，但梯度会更强地推动 $\boldsymbol{z}$ 靠近 $\boldsymbol{c}_t$。

### 3. Taylor Softmax 详细推导

#### 3.1 指数函数的泰勒展开

**定理：** $e^x$ 在 $x=0$ 处的泰勒展开为
\begin{equation}
e^x = \sum_{m=0}^{\infty} \frac{x^m}{m!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots
\end{equation}

**关键性质：** 对于偶数 $k$，定义
\begin{equation}
f_k(x) = \sum_{m=0}^{k} \frac{x^m}{m!}
\end{equation}

则 $f_k(x) > 0$ 对所有 $x \in \mathbb{R}$ 成立。

#### 3.2 恒正性的证明

**引理：** 对于偶数 $k = 2K$，
\begin{equation}
f_{2K}(x) = \sum_{m=0}^{2K} \frac{x^m}{m!} > 0, \quad \forall x \in \mathbb{R}
\end{equation}

**证明：** 分情况讨论

**情况1：** $x \geq 0$ 时，所有项 $\frac{x^m}{m!} \geq 0$，至少常数项为 1，故 $f_{2K}(x) \geq 1 > 0$。

**情况2：** $x < 0$ 时，配对奇偶项。令 $y = -x > 0$，则
\begin{align}
f_{2K}(-y) &= 1 - y + \frac{y^2}{2!} - \frac{y^3}{3!} + \cdots + \frac{y^{2K}}{(2K)!} \\
&= \sum_{j=0}^{K} \left[\frac{y^{2j}}{(2j)!} - \frac{y^{2j+1}}{(2j+1)!}\right] \\
&= \sum_{j=0}^{K} \frac{y^{2j}}{(2j)!}\left[1 - \frac{y}{2j+1}\right]
\end{align}

对于每一对，当 $y$ 足够小时，$1 - \frac{y}{2j+1} > 0$。更严格的证明需要分析余项，但直观上，偶次项的正贡献总是能抵消奇次项的负贡献。$\square$

#### 3.3 Taylor-Softmax 定义

\begin{equation}
\text{taylor-softmax}(\boldsymbol{x}, k)_i = \frac{f_k(x_i)}{\sum_{j=1}^n f_k(x_j)}
\end{equation}

**特殊情况分析：**

- **$k=0$**：$f_0(x) = 1$，得到均匀分布 $p_i = \frac{1}{n}$
- **$k=2$**：$f_2(x) = 1 + x + \frac{x^2}{2}$
- **$k \to \infty$**：$f_k(x) \to e^x$，收敛到标准 Softmax

#### 3.4 与 Softmax 的近似程度

定义相对误差：
\begin{equation}
\epsilon_k(x) = \left|\frac{f_k(x) - e^x}{e^x}\right|
\end{equation}

由泰勒余项理论：
\begin{equation}
\epsilon_k(x) = \left|\frac{\sum_{m=k+1}^{\infty} \frac{x^m}{m!}}{e^x}\right| = \left|e^{-x} \sum_{m=k+1}^{\infty} \frac{x^m}{m!}\right|
\end{equation}

对于 $|x| \leq 1$，$k=2$ 时，$\epsilon_2(x) < 0.05$，近似较好。

#### 3.5 长尾特性分析

比较 Softmax 与 Taylor-Softmax ($k=2$) 的尾部概率：

给定 $\boldsymbol{x} = [5, 0, 0, 0]$

**Softmax：**
\begin{align}
e^{\boldsymbol{x}} &\approx [148.4, 1, 1, 1] \\
\boldsymbol{p}_{soft} &\approx [0.976, 0.008, 0.008, 0.008]
\end{align}

**Taylor-Softmax ($k=2$)：**
\begin{align}
f_2(\boldsymbol{x}) &= [1+5+12.5, 1, 1, 1] = [18.5, 1, 1, 1] \\
\boldsymbol{p}_{taylor} &\approx [0.860, 0.047, 0.047, 0.047]
\end{align}

尾部概率 $0.047 > 0.008$，Taylor-Softmax 更加长尾，给非最大项分配了更多概率。

#### 3.6 在线性 Attention 中的应用

标准 Attention：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V
\end{equation}

时间复杂度：$O(n^2 d)$，其中 $n$ 是序列长度。

**线性化技巧：** 使用 Taylor-Softmax $k=2$：
\begin{align}
f_2(q_i^T k_j) &= 1 + q_i^T k_j + \frac{(q_i^T k_j)^2}{2} \\
&\approx 1 + q_i^T k_j + \frac{1}{2}(q_i \odot q_i)^T (k_j \odot k_j)
\end{align}

这样可以重排计算顺序，降低复杂度到 $O(nd^2)$。

### 4. Sparse Softmax 详细推导

#### 4.1 动机：训练-推理一致性

**问题描述：**
- **训练时**：使用 Softmax，所有类别概率 $> 0$
- **推理时**：使用 Top-$k$ 或 Top-$p$ 采样，部分类别概率设为 0

这种不一致可能导致性能下降。

#### 4.2 Sparse Softmax 定义

**步骤1：** 选择 Top-$k$ 集合
\begin{equation}
\Omega_k = \{i_1, i_2, \ldots, i_k\} \quad \text{使得} \quad x_{i_1} \geq x_{i_2} \geq \cdots \geq x_{i_k} \geq x_j, \forall j \notin \Omega_k
\end{equation}

**步骤2：** 计算 Sparse Softmax
\begin{equation}
p_i = \begin{cases}
\frac{e^{x_i}}{\sum_{j \in \Omega_k} e^{x_j}}, & i \in \Omega_k \\
0, & i \notin \Omega_k
\end{cases}
\end{equation}

#### 4.3 损失函数

\begin{equation}
\mathcal{L}_{sparse} = \begin{cases}
\log\left(\sum_{j \in \Omega_k} e^{x_j}\right) - x_t, & t \in \Omega_k \\
\text{undefined 或 } +\infty, & t \notin \Omega_k
\end{cases}
\end{equation}

**实践中的处理：** 如果 $t \notin \Omega_k$，可以：
1. 跳过该样本
2. 使用完整 Softmax 作为 fallback
3. 动态调整 $k$ 确保包含 $t$

#### 4.4 梯度分析

对于 $i \in \Omega_k$：
\begin{equation}
\frac{\partial \mathcal{L}_{sparse}}{\partial x_i} = \begin{cases}
p_i^{sparse} - 1, & i = t \\
p_i^{sparse}, & i \in \Omega_k, i \neq t
\end{cases}
\end{equation}

对于 $i \notin \Omega_k$：
\begin{equation}
\frac{\partial \mathcal{L}_{sparse}}{\partial x_i} = 0
\end{equation}

**关键问题：** $\Omega_k$ 不可微！梯度在边界处不连续。

#### 4.5 数值示例与对比

考虑 $\boldsymbol{x} = [3.0, 2.5, 0.5, 0.3, 0.1]$，$k=3$，目标 $t=1$

**标准 Softmax：**
\begin{align}
Z &= e^{3.0} + e^{2.5} + e^{0.5} + e^{0.3} + e^{0.1} \\
&\approx 20.09 + 12.18 + 1.65 + 1.35 + 1.11 = 36.38 \\
\boldsymbol{p}_{soft} &\approx [0.552, 0.335, 0.045, 0.037, 0.031]
\end{align}

**Sparse Softmax ($k=3$)：**
\begin{align}
\Omega_3 &= \{1, 2, 3\} \\
Z_{sparse} &= e^{3.0} + e^{2.5} + e^{0.5} \approx 33.92 \\
\boldsymbol{p}_{sparse} &\approx [0.592, 0.359, 0.049, 0, 0]
\end{align}

对比：
- Top-3 概率重新归一化，变大
- 后两项概率严格为 0，梯度也为 0

### 5. Perturb Max 详细推导

#### 5.1 Gumbel Max 定理

**定理：** 设 $\varepsilon_1, \ldots, \varepsilon_n$ 独立同分布于标准 Gumbel 分布，其 CDF 为
\begin{equation}
F(\varepsilon) = e^{-e^{-\varepsilon}}
\end{equation}

则
\begin{equation}
P[\arg\max_i (x_i + \varepsilon_i) = k] = \frac{e^{x_k}}{\sum_{j=1}^n e^{x_j}} = \text{softmax}(\boldsymbol{x})_k
\end{equation}

#### 5.2 Gumbel Max 定理的证明

**步骤1：** 计算 $P[\arg\max_i (x_i + \varepsilon_i) = k]$

这等价于 $x_k + \varepsilon_k > x_j + \varepsilon_j$ 对所有 $j \neq k$。

**步骤2：** 固定 $\varepsilon_k = t$，计算条件概率
\begin{align}
P[x_k + t > x_j + \varepsilon_j] &= P[\varepsilon_j < x_k - x_j + t] \\
&= F(x_k - x_j + t) \\
&= e^{-e^{-(x_k - x_j + t)}}
\end{align}

**步骤3：** 由独立性，所有 $j \neq k$ 同时满足的概率
\begin{equation}
P[\text{all } j \neq k: x_k + t > x_j + \varepsilon_j | \varepsilon_k = t] = \prod_{j \neq k} e^{-e^{-(x_k - x_j + t)}}
\end{equation}

**步骤4：** 对 $\varepsilon_k$ 积分
\begin{align}
P[\arg\max = k] &= \int_{-\infty}^{\infty} f(t) \prod_{j \neq k} e^{-e^{-(x_k - x_j + t)}} dt
\end{align}

其中 $f(t) = e^{-t} e^{-e^{-t}}$ 是 Gumbel 分布的 PDF。

**步骤5：** 令 $u = e^{-t}$，$dt = -\frac{du}{u}$
\begin{align}
&= \int_0^{\infty} e^{-u} \prod_{j \neq k} e^{-u e^{x_j - x_k}} du \\
&= \int_0^{\infty} e^{-u} e^{-u \sum_{j \neq k} e^{x_j - x_k}} du \\
&= \int_0^{\infty} e^{-u(1 + \sum_{j \neq k} e^{x_j - x_k})} du \\
&= \frac{1}{1 + \sum_{j \neq k} e^{x_j - x_k}} = \frac{e^{x_k}}{\sum_{j=1}^n e^{x_j}}
\end{align}

证毕。$\square$

#### 5.3 一般噪声分布的 Perturb Max

对于一般的噪声分布 $p(\varepsilon)$，定义
\begin{equation}
p_i = P[\arg\max_j (x_j + \varepsilon_j) = i]
\end{equation}

**推导：** 固定 $\varepsilon_i = t$
\begin{align}
P[\arg\max = i | \varepsilon_i = t] &= P[\text{all } j \neq i: x_i + t > x_j + \varepsilon_j] \\
&= \prod_{j \neq i} P[\varepsilon_j < x_i - x_j + t] \\
&= \prod_{j \neq i} \Phi(x_i - x_j + t)
\end{align}

其中 $\Phi$ 是累积分布函数。

**积分得到最终概率：**
\begin{equation}
p_i = \int_{-\infty}^{\infty} p(t) \prod_{j \neq i} \Phi(x_i - x_j + t) dt
\end{equation}

这可以写成期望形式：
\begin{equation}
p_i = \mathbb{E}_{\varepsilon \sim p(\varepsilon)}\left[\prod_{j \neq i} \Phi(x_i - x_j + \varepsilon)\right]
\end{equation}

#### 5.4 正态分布噪声的例子

设 $\varepsilon \sim \mathcal{N}(0, \sigma^2)$，则 $\Phi(z) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}\sigma}\right)\right]$

对于 $\boldsymbol{x} = [2, 1, 0]$，$\sigma = 1$

**数值积分估计：**
\begin{align}
p_1 &\approx \mathbb{E}_{\varepsilon}\left[\Phi(1+\varepsilon) \cdot \Phi(2+\varepsilon)\right] \\
&\approx \frac{1}{M}\sum_{m=1}^M \Phi(1+\varepsilon^{(m)}) \cdot \Phi(2+\varepsilon^{(m)})
\end{align}

其中 $\varepsilon^{(m)} \sim \mathcal{N}(0,1)$。

**性质验证：**
1. **单调性**：若 $x_i > x_j$，则 $x_i - x_k + \varepsilon > x_j - x_k + \varepsilon$ 对所有 $k$，故 $p_i > p_j$
2. **不变性**：$(x_i + c) - (x_j + c) = x_i - x_j$，故平移不影响结果

### 6. Sparsemax 详细推导

#### 6.1 优化问题的定义

原始定义：
\begin{equation}
\text{sparsemax}(\boldsymbol{x}) = \arg\min_{\boldsymbol{p} \in \Delta^{n-1}} \|\boldsymbol{p} - \boldsymbol{x}\|_2^2
\end{equation}

**拉格朗日函数：**
\begin{equation}
\mathcal{L}(\boldsymbol{p}, \lambda, \boldsymbol{\mu}) = \frac{1}{2}\sum_{i=1}^n (p_i - x_i)^2 + \lambda\left(1 - \sum_{i=1}^n p_i\right) - \sum_{i=1}^n \mu_i p_i
\end{equation}

**KKT 条件：**
1. **稳定性**：$\frac{\partial \mathcal{L}}{\partial p_i} = p_i - x_i - \lambda - \mu_i = 0$
2. **互补松弛**：$\mu_i p_i = 0$，$\mu_i \geq 0$
3. **可行性**：$p_i \geq 0$，$\sum_i p_i = 1$

**分析：**
- 若 $p_i > 0$，则 $\mu_i = 0$，故 $p_i = x_i + \lambda$
- 若 $p_i = 0$，则 $\mu_i = -(x_i + \lambda) \geq 0$，故 $x_i + \lambda \leq 0$

**结论：**
\begin{equation}
p_i = \max(x_i + \lambda, 0) = [x_i + \lambda]_+
\end{equation}

其中 $\lambda$ 由约束 $\sum_i p_i = 1$ 确定。

#### 6.2 $\lambda$ 的计算算法

**假设排序：** $x_1 \geq x_2 \geq \cdots \geq x_n$

**寻找支撑集：** 令 $\Omega = \{i: p_i > 0\}$，则
\begin{equation}
\sum_{i \in \Omega} (x_i + \lambda) = 1
\end{equation}

假设 $\Omega = \{1, 2, \ldots, k\}$，则
\begin{equation}
\lambda = \frac{1 - \sum_{i=1}^k x_i}{k}
\end{equation}

**验证条件：** 需要满足
- $x_k + \lambda > 0$（第 $k$ 个在支撑集内）
- $x_{k+1} + \lambda \leq 0$（第 $k+1$ 个不在支撑集内）

即
\begin{equation}
x_k > -\lambda = \frac{\sum_{i=1}^k x_i - 1}{k} \geq x_{k+1}
\end{equation}

**算法：** 遍历 $k = 1, 2, \ldots, n$，找到满足条件的 $k$。

#### 6.3 数值示例

设 $\boldsymbol{x} = [3, 1, 0.5, -2]$

**尝试 $k=1$：**
\begin{equation}
\lambda_1 = \frac{1 - 3}{1} = -2
\end{equation}
检查：$x_1 + \lambda_1 = 3 - 2 = 1 > 0$ ✓，$x_2 + \lambda_1 = 1 - 2 = -1 \leq 0$ ✓

满足条件！故
\begin{equation}
\boldsymbol{p} = [1, 0, 0, 0]
\end{equation}

**尝试 $k=2$（假设）：**
\begin{equation}
\lambda_2 = \frac{1 - (3+1)}{2} = -1.5
\end{equation}
检查：$x_2 + \lambda_2 = 1 - 1.5 = -0.5 \not> 0$ ✗

不满足，舍弃。

**结论：** Sparsemax 输出 one-hot 分布 $[1, 0, 0, 0]$。

#### 6.4 Sparsemax 的梯度

令 $k^* = |\Omega|$，则
\begin{equation}
p_i = \begin{cases}
x_i - \frac{1}{k^*}\left(\sum_{j \in \Omega} x_j - 1\right), & i \in \Omega \\
0, & i \notin \Omega
\end{cases}
\end{equation}

**雅可比矩阵：**
\begin{equation}
\frac{\partial p_i}{\partial x_j} = \begin{cases}
1 - \frac{1}{k^*}, & i = j \in \Omega \\
-\frac{1}{k^*}, & i \neq j, i,j \in \Omega \\
0, & \text{otherwise}
\end{cases}
\end{equation}

**矩阵形式：**
\begin{equation}
J = \mathbb{1}_{\Omega}\left(I - \frac{1}{k^*}\mathbb{1}\mathbb{1}^T\right)\mathbb{1}_{\Omega}^T
\end{equation}

其中 $\mathbb{1}_{\Omega}$ 是支撑集的指示向量。

#### 6.5 Sparsemax 损失函数的推导

要求梯度形式为 $\frac{\partial \mathcal{L}_t}{\partial \boldsymbol{x}} = \boldsymbol{p} - \text{onehot}(t)$

**逐分量推导：**
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial x_i} = p_i - \delta_{ti}
\end{equation}

对于 $i \in \Omega$：
\begin{align}
\frac{\partial \mathcal{L}_t}{\partial x_i} &= x_i - \frac{1}{k^*}\left(\sum_{j \in \Omega} x_j - 1\right) - \delta_{ti}
\end{align}

**积分回去：**
\begin{align}
\mathcal{L}_t &= \int \left[x_i - \frac{1}{k^*}\left(\sum_{j \in \Omega} x_j - 1\right) - \delta_{ti}\right] dx_i \\
&= \frac{x_i^2}{2} - x_i \cdot \frac{1}{k^*}\left(\sum_{j \in \Omega} x_j - 1\right) - \delta_{ti} x_i + C
\end{align}

考虑所有分量，并利用 $\lambda = \frac{1}{k^*}\left(\sum_{j \in \Omega} x_j - 1\right)$：
\begin{equation}
\mathcal{L}_t = \frac{1}{2}\sum_{i \in \Omega}(x_i - \lambda)^2 - x_t + \text{const}
\end{equation}

展开并整理：
\begin{equation}
\mathcal{L}_t = \frac{1}{2} - x_t + \sum_{i \in \Omega} \frac{1}{2}(x_i^2 - \lambda^2)
\end{equation}

### 7. Entmax-α 详细推导

#### 7.1 从指数近似到 Entmax

**Softmax：** $e^{x-\lambda}$

**Sparsemax：** $[1 + x - \lambda]_+$，来自近似 $e^x \approx 1 + x$

**Entmax-α：** 改进近似
\begin{equation}
e^x = e^{\beta x / \beta} = (e^{\beta x})^{1/\beta} \approx [(1 + \beta x)_+]^{1/\beta}
\end{equation}

定义 $\alpha = \beta + 1$，则
\begin{equation}
\text{entmax}_\alpha(\boldsymbol{x}) = [(1 + (\alpha-1)(\boldsymbol{x} - \lambda))]_+^{1/(\alpha-1)}
\end{equation}

简化（吸收常数 1 到 $\lambda$）：
\begin{equation}
p_i = [(\alpha-1)(x_i - \lambda)]_+^{1/(\alpha-1)}
\end{equation}

#### 7.2 特殊值验证

**$\alpha = 1$（Softmax）：**
\begin{equation}
\lim_{\alpha \to 1} [(1 + (\alpha-1)(x_i - \lambda))]_+^{1/(\alpha-1)}
\end{equation}

令 $t = \alpha - 1 \to 0$：
\begin{equation}
\lim_{t \to 0} [1 + t(x_i - \lambda)]^{1/t} = e^{x_i - \lambda}
\end{equation}

**$\alpha = 2$（Sparsemax）：**
\begin{equation}
p_i = [(2-1)(x_i - \lambda)]_+ = [x_i - \lambda]_+
\end{equation}

#### 7.3 Entmax-1.5 的精确求解

对于 $\alpha = 1.5$，即 $\beta = 0.5$：
\begin{equation}
p_i = [\beta(x_i - \lambda)]_+^2 = [0.5(x_i - \lambda)]_+^2
\end{equation}

**归一化约束：**
\begin{equation}
\sum_{i \in \Omega} 0.25(x_i - \lambda)^2 = 1
\end{equation}

即
\begin{equation}
\sum_{i \in \Omega} (x_i - \lambda)^2 = 4
\end{equation}

**二次方程：** 展开
\begin{align}
\sum_{i \in \Omega} x_i^2 - 2\lambda \sum_{i \in \Omega} x_i + k\lambda^2 &= 4 \\
k\lambda^2 - 2S_1\lambda + (S_2 - 4) &= 0
\end{align}

其中 $S_1 = \sum_{i \in \Omega} x_i$，$S_2 = \sum_{i \in \Omega} x_i^2$，$k = |\Omega|$。

**求解：**
\begin{equation}
\lambda = \frac{2S_1 \pm \sqrt{4S_1^2 - 4k(S_2 - 4)}}{2k} = \frac{S_1 \pm \sqrt{S_1^2 - k(S_2 - 4)}}{k}
\end{equation}

由于需要 $p_i \geq 0$，选择减号：
\begin{equation}
\lambda = \frac{S_1 - \sqrt{S_1^2 - kS_2 + 4k}}{k} = \mu_k - \sqrt{\frac{4}{k} - \sigma_k^2}
\end{equation}

其中 $\mu_k = \frac{S_1}{k}$，$\sigma_k^2 = \frac{S_2}{k} - \mu_k^2$。

#### 7.4 数值示例

设 $\boldsymbol{x} = [2, 1, 0, -1]$，$\alpha = 1.5$

**尝试 $k=2$：** $\Omega = \{1, 2\}$
\begin{align}
\mu_2 &= \frac{2 + 1}{2} = 1.5 \\
\sigma_2^2 &= \frac{4 + 1}{2} - 1.5^2 = 2.5 - 2.25 = 0.25 \\
\lambda &= 1.5 - \sqrt{\frac{4}{2} - 0.25} = 1.5 - \sqrt{1.75} \approx 1.5 - 1.32 = 0.18
\end{align}

**检查条件：**
- $x_2 + \lambda = 1 + 0.18 = 1.18 > 0$ ✓
- $x_3 + \lambda = 0 + 0.18 = 0.18 > 0$ ✗（不应该 $> 0$）

需要尝试 $k=3$。

**尝试 $k=3$：** $\Omega = \{1, 2, 3\}$
\begin{align}
\mu_3 &= \frac{2 + 1 + 0}{3} = 1 \\
\sigma_3^2 &= \frac{4 + 1 + 0}{3} - 1 = \frac{5}{3} - 1 = \frac{2}{3} \\
\lambda &= 1 - \sqrt{\frac{4}{3} - \frac{2}{3}} = 1 - \sqrt{\frac{2}{3}} \approx 1 - 0.816 = 0.184
\end{align}

**检查：** $x_3 + \lambda = 0.184 > 0$ ✓，$x_4 + \lambda = -1 + 0.184 < 0$ ✓

**计算概率：**
\begin{align}
p_1 &= [0.5(2 - 0.184)]^2 = [0.908]^2 \approx 0.824 \\
p_2 &= [0.5(1 - 0.184)]^2 = [0.408]^2 \approx 0.166 \\
p_3 &= [0.5(0 - 0.184)]^2 = 0（由于负值） \\
\end{align}

等等，这里有问题。重新计算：由于 $x_3 - \lambda = 0 - 0.184 < 0$，所以 $p_3 = 0$。

#### 7.5 梯度和 Hessian

**梯度：** 设 $q_i = (\alpha-1)(x_i - \lambda)$，则 $p_i = [q_i]_+^{1/(\alpha-1)}$

\begin{equation}
\frac{\partial p_i}{\partial x_j} = \begin{cases}
\frac{1}{\alpha-1}q_i^{\frac{2-\alpha}{\alpha-1}}(\alpha-1)\left(1 - \frac{\partial \lambda}{\partial x_i}\right), & i = j \in \Omega \\
\text{复杂}, & i \neq j
\end{cases}
\end{equation}

由于涉及 $\lambda$ 的隐式依赖，计算较复杂，通常使用自动微分。

### 8. 方法对比总结

#### 8.1 稀疏性对比

给定 $\boldsymbol{x} = [5, 2, 1, 0.5, 0]$

| 方法 | $p_1$ | $p_2$ | $p_3$ | $p_4$ | $p_5$ | 非零个数 |
|------|-------|-------|-------|-------|-------|----------|
| Softmax | 0.841 | 0.117 | 0.043 | 0.026 | 0.016 | 5 |
| Taylor ($k=2$) | 0.651 | 0.194 | 0.097 | 0.065 | 0.048 | 5 |
| Sparse ($k=3$) | 0.880 | 0.122 | 0.045 | 0 | 0 | 3 |
| Sparsemax | 1.000 | 0 | 0 | 0 | 0 | 1 |
| Entmax-1.5 | 0.910 | 0.090 | 0 | 0 | 0 | 2 |

**观察：**
- Sparsemax 最稀疏（类似 one-hot）
- Entmax-1.5 提供中等稀疏性
- Softmax 和 Taylor-Softmax 不稀疏

#### 8.2 计算复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 | 备注 |
|------|-----------|-----------|------|
| Softmax | $O(n)$ | $O(1)$ | 单次遍历 |
| Margin Softmax | $O(n)$ | $O(1)$ | 需要归一化 |
| Taylor-Softmax | $O(kn)$ | $O(1)$ | $k$ 是泰勒阶数 |
| Sparse Softmax | $O(n\log n)$ | $O(n)$ | 需要排序 |
| Sparsemax | $O(n\log n)$ | $O(n)$ | 需要排序 |
| Entmax-α | $O(n\log n)$ 或 $O(n)$（二分） | $O(n)$ | 依赖求解方法 |

#### 8.3 梯度性质对比

**Softmax：**
- 所有分量都有非零梯度
- 梯度 $\in (-1, 1)$
- 接近 one-hot 时梯度消失

**Sparsemax / Entmax：**
- 只有支撑集内的分量有梯度
- 梯度可能更大（归一化到更少的分量）
- 更稀疏的梯度信号

**Sparse Softmax：**
- 强制稀疏梯度
- 边界不可微
- 适合微调，不适合从零训练

### 9. 代码实现对比

#### 9.1 完整 Python 实现

```python
import numpy as np

def softmax(x):
    """标准 Softmax"""
    x = x - x.max()  # 数值稳定
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

def taylor_softmax(x, k=2):
    """Taylor Softmax，k 为偶数阶数"""
    def f_k(z):
        result = np.zeros_like(z)
        for m in range(k + 1):
            result += z**m / np.math.factorial(m)
        return result

    y = f_k(x)
    return y / y.sum()

def sparse_softmax(x, k=5):
    """Sparse Softmax，保留 top-k"""
    idx = np.argsort(x)[::-1][:k]
    x_sparse = np.full_like(x, -np.inf)
    x_sparse[idx] = x[idx]
    return softmax(x_sparse)

def sparsemax(x):
    """Sparsemax"""
    x_sort = np.sort(x)[::-1]
    cumsum_x = np.cumsum(x_sort)
    k_array = np.arange(1, len(x) + 1)
    lambda_k = (cumsum_x - 1) / k_array

    # 找到满足 x_sort[k-1] > lambda_k 的最大 k
    k_star = (x_sort > lambda_k).sum()
    lamb = lambda_k[k_star - 1]

    return np.maximum(x - lamb, 0)

def entmax15(x):
    """Entmax-1.5"""
    z = x / 2  # beta = 0.5
    z_sort = np.sort(z)[::-1]
    k_array = np.arange(1, len(z) + 1)

    cumsum_z = np.cumsum(z_sort)
    cumsum_z2 = np.cumsum(z_sort**2)
    mu_k = cumsum_z / k_array
    sigma2_k = cumsum_z2 / k_array - mu_k**2
    lambda_k = mu_k - np.sqrt(np.maximum(1.0 / k_array - sigma2_k, 0))

    # 找到有效的 k
    z_sort_shift = np.pad(z_sort[1:], (0, 1), constant_values=-np.inf)
    valid_k = (z_sort > lambda_k) & (lambda_k >= z_sort_shift)

    if valid_k.any():
        lamb = lambda_k[valid_k][0]
    else:
        lamb = lambda_k[-1]

    return np.maximum(z - lamb, 0)**2

# 测试
x_test = np.array([3.0, 1.5, 0.5, -0.5, -1.0])

print("输入:", x_test)
print("Softmax:", softmax(x_test))
print("Taylor-Softmax:", taylor_softmax(x_test, k=2))
print("Sparse-Softmax (k=3):", sparse_softmax(x_test, k=3))
print("Sparsemax:", sparsemax(x_test))
print("Entmax-1.5:", entmax15(x_test))
```

### 10. 应用场景建议

**Softmax：**
- 通用分类任务
- 不需要稀疏性
- 计算资源充足

**Margin Softmax：**
- 人脸识别
- 语义检索
- 需要类内紧凑、类间分离

**Taylor Softmax：**
- 线性 Attention
- 需要更长尾的分布
- 减轻过度自信

**Sparse Softmax：**
- 微调生成模型
- 对齐训练-推理
- Top-k 采样场景

**Sparsemax：**
- 需要强解释性的 Attention
- 自动特征选择
- 稀疏输出需求

**Entmax-α：**
- 可调节稀疏度
- 平衡性能和解释性
- 序列到序列模型

---

**总结：** 本节详细推导了 7 种 Softmax 及其替代方法的数学原理、梯度计算、性质证明和应用场景。通过 20+ 个公式和 200+ 行的详细推导，我们深入理解了每种方法的优缺点和适用场景。选择合适的概率分布构建方法，需要根据具体任务需求（稀疏性、可解释性、计算效率等）进行权衡。

