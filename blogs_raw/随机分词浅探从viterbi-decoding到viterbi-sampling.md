---
title: 随机分词浅探：从Viterbi Decoding到Viterbi Sampling
slug: 随机分词浅探从viterbi-decoding到viterbi-sampling
date: 2023-09-16
tags: 概率, 随机, 分词, 新词发现, 生成模型
status: pending
---

# 随机分词浅探：从Viterbi Decoding到Viterbi Sampling

**原文链接**: [https://spaces.ac.cn/archives/9768](https://spaces.ac.cn/archives/9768)

**发布日期**: 

---

上一篇文章[《大词表语言模型在续写任务上的一个问题及对策》](/archives/9762)发布后，很快就有读者指出可以在训练阶段引入带有随机性的分词结果来解决同样的问题，并且已经有论文和实现。经过进一步查阅学习，笔者发现这是一个名为[Subword Regularization](https://papers.cool/arxiv/1804.10959)的技巧，最早应用在NMT（机器翻译）中，目前SentencePiece也有相应的实现。看起来这个技巧确实能缓解前述问题，甚至有助于增强语言模型的容错能力，所以就有了将它加进去[BytePiece](/archives/9752)的想法。

那么问题来了，如何将确定性分词改为随机性分词呢？BytePiece是基于Unigram模型的，它通过Viterbi算法找最大概率的分词方案，既然有概率，是否就可以自然地导出随机采样？本文来讨论这个问题，并分享自己的解决方案。

## 要点分析 #

现阶段，Unigram分词是直接输出最大概率的切分方案，通常这是一个确定性的输出。具体来说，假设$\boldsymbol{w}=(w_1,w_2,\cdots,w_k)$代表一个切分方案，对应的打分为$P(\boldsymbol{w})=p(w_1)p(w_2)\cdots p(w_k)$，$\Omega(S)$代表句子$S$所有可能的切分方案的集合，那么分词算法可以描述为  
\begin{equation}\boldsymbol{w}^* = \mathop{\text{argmax}}_{\boldsymbol{w}\in \Omega(S)}P(\boldsymbol{w})\end{equation}  
这可以通过Viterbi算法在线性时间内来完成，所以这个过程我们也称之为“Viterbi Decoding”。看起来，Unigram模型天然带有概率，所以似乎并不难将它改为依概率采样的形式，但细想之下才发现这并非一个平凡的问题，有很多细节上的困难需要克服。

笔者设想是模仿自回归语言模型设计一个递归采样流程，但这里最困难的地方是如何尽量保持原来的候选切分方案的排序不变，或者就算不能保持所有的排序不变，也至少满足最大概率不变，即Viterbi解码的最大概率路径$\boldsymbol{w}^*$应该对应所设计的递归采样算法的最大概率采样结果。由于所有切分方案$\Omega(S)$构成一个有向无环图（DAG，Directed Acyclic Graph），笔者一开始以为直接在有向无环图上随机游走是一个可行方案，但再思考后发现很难设计适当的转移概率来保证最大概率路径不变（因为同一起点的不同边不是平权的，不能简单按照边的频率为权重做采样）。

## 已有方案 #

由于一时半会没有新想法，所以笔者决定去翻翻“参考答案”——看看Subword Regularization的原始论文[《Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates》](https://papers.cool/arxiv/1804.10959)是怎么做的。

然而，这个“标准答案”却让笔者有点哭笑不得。原来Subword Regularization的思路非常简单直接：先搜索出$P(\boldsymbol{w})$最大的$n$个分词方案$\boldsymbol{w}^*_1,\boldsymbol{w}^*_2,\cdots,\boldsymbol{w}^*_n$（$n$-best segmentations），然后构建如下分布  
\begin{equation}p_i = \frac{P(\boldsymbol{w}^*_i)^{\alpha}}{\sum\limits_{j=1}^n P(\boldsymbol{w}^*_j)^{\alpha}}\end{equation}  
对这$n$个方案进行依概率采样，其中$\alpha > 0$是一个超参数。该算法已经集成在SentencePiece中，读者可以自行测试（使用方法参考[这里](https://github.com/google/sentencepiece/tree/master#subword-regularization-and-bpe-dropout)）。

问题是，“简单直接”不代表着“高效”，尽管搜索top-$n$个分词方案最优方案的复杂度也是线性的（有兴趣的读者可以自行找找N-best Viterbi的资料)，但明显比只找top1的Viterbi Decoding要大很多（理论上是$n$倍复杂度），所以直接的后果是开启了随机采样后，会比确定性的分词要慢很多，所以这并非是笔者心中的理想采样方法。

## 个人思路 #

思路再度陷入了僵局。一筹莫展之际，笔者决定把思路再捋一捋：我们的目标是想要找到复杂度类似Viterbi Decoding的随机采样算法，既然如此，Viterbi Decoding本身应该是一个不错的突破点。于是，笔者再次翻开了分词代码，[当时的分词函数](https://github.com/bojone/bytepiece/blob/b65716b76938b3ac4124661a3367fc1c270373fa/bytepiece/faster.pyx)长这个样：
    
    
    def _tokenize(self, bytes text):
        cdef int e, k, s
        cdef double v, score
        cdef list routes = [(0, None)] + [(-INFINITY, None) for _ in text]
        cdef list tokens = []
        for e, (k, v) in self._automaton.iter(text):
            s, e = e - k + 1, e + 1
            score = routes[s][0] + v
            if score > routes[e][0]:
                routes[e] = score, s
        while text:
            s = routes[e][1]
            tokens.append(text[s:e])
            text, e = text[:s], s
        return tokens[::-1]

反复读了几遍，总算有了灵感：Viterbi Decoding的关键是`if score > routes[e][0]:`这一句，它代表保留截止到当前位置的最优切分方案，其中`score`是新切分方案分数（概率对数），`routes[e][0]`是历史最优分数，如果新方案更优则覆盖。这让笔者联想到了[MCMC算法](/archives/8084)的接受率设计，如果在这里引入随机采样，不就可以将分词结果随机化了？

我们用$r\in \\{1, 0\\}$表示接受/拒绝新方案，由于这一步只是一个二元选择，所以将它概率化也非常简单：  
\begin{equation}  
r_i = \left\\{\begin{aligned}&\,1\,, \,\, s_i > s_{i-1} \\\  
&\,0\,, \,\, \text{else}\end{aligned}\right.\qquad\longrightarrow\qquad  
r_i = \left\\{\begin{aligned}&\,1\,, \,\, \varepsilon < \sigma(\alpha(s_i - s_{i-1})) \\\  
&\,0\,, \,\, \text{else}\end{aligned}\right.  
\end{equation}  
这里$\varepsilon\sim U[0,1]$是均匀随机数，$\alpha > 0$是超参数，$\sigma(t)=1/(1+e^{-t})$是Sigmoid函数，$s_i,s_{i-1}$分别是新旧方案的得分（概率对数）。不难发现，左端的确定性采样对应$\alpha\to\infty$的随机性采样。

这样，在Viterbi解码的基础上我们得到了一个非常自然、非常轻量级的随机采样算法，这里称之为“Viterbi Sampling”，实现它只需要将`if score > routes[e][0]:`这一判据换成带随机数的版本。由于Sigmoid函数的单调性，当$s_i > s_{i-1}$时，它自然会给新方案分配更大的概率，所以很明显原来的的最大概率切分在Viterbi Sampling之下也是最大概率结果，并且当$s_i - s_{i-1}$越大，$\sigma(\alpha(s_i - s_{i-1}))$也越大，这意味着原本得分越大的方案被采样到的概率也越高，一定程度上保持了切分方案的排序不变（尽管还没有证明一定严格保序，但从应用角度看，近似保序就够了）。

## 简单测试 #

从0.4.0版本开始，Viterbi Sampling就内置在BytePiece的分词函数中，只需要在`tokenizer.tokenize`或者`tokenizer.encode`时加入大于0的alpha参数，结果就是随机的：
    
    
    import bytepiece
    assert bytepiece.__version__ >= '0.4.0'
    
    tokenizer = bytepiece.Tokenizer('bytepiece_160k.model')
    text = '今天天气不错'
    print(tokenizer.tokenize(text))  # alpha默认值为-1，alpha≤0 都代表确定性分词
    for i in range(5):
        print(tokenizer.tokenize(text, alpha=0.1))
    
    # [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
    # [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6', b'\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
    # [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
    # [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8', b'\x8d', b'\xe9\x94', b'\x99']
    # [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9', b'\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
    # [b'\xe4\xbb', b'\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9', b'\xe6\xb0\x94\xe4\xb8\x8d', b'\xe9\x94', b'\x99']
    

下面对比一下SentencePiece的Subword Regularization和BytePiece的Viterbi Sampling的速度（随机性分词时都设$\alpha=0.1$）：  
\begin{array}{c|cc}  
\hline  
& \text{确定性分词} & \text{随机性分词} & \\\  
\hline  
\text{SP-BPE} & \text{1.36M bytes/sec} & \text{1.25M bytes/sec} \\\  
\text{SP-Unigram} & \text{5.65M bytes/sec} & \text{1.28M bytes/sec} \\\  
\text{BytePiece} & \text{1.95M bytes/sec} & \text{1.36M bytes/sec}\\\  
\hline  
\end{array}  
可以看到，Subword Regularization（“SP-Unigram”这一行）开启之后，分词速度不到原来的1/4，这表明Subword Regularization的采样算法是相当低效的。相比之下，本文提出的Viterbi Sampling只下降了30%左右，效率显然更高，下降的部分在于随机数的生成和Sigmoid函数的计算，如果能进一步优化这两部分，速度还能进一步提升。至于BPE模型，它的随机分词叫做[BPE Dropout](https://papers.cool/arxiv/1910.13267)，这是专属于BPE模型的方法，有兴趣的读者自行了解，这里就不介绍了。

## 文章小结 #

本文主要探讨了将Unigram分词模型的确定性分词改为随机性分词的策略。尽管已有名为“Subword Regularization”的方法可以实现这一目标，但其效率相对较低。为此，笔者提出了一种更高效的采样算法Viterbi Sampling，它仅需对确定性的Viterbi Decoding进行简单的修改，从而基本保持了原有的效率。实验证明，新的算法采样速度明显超越了Subword Regularization。相应的实现已经内置在BytePiece最新版中。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9768>_

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

苏剑林. (Sep. 16, 2023). 《随机分词浅探：从Viterbi Decoding到Viterbi Sampling 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9768>

@online{kexuefm-9768,  
title={随机分词浅探：从Viterbi Decoding到Viterbi Sampling},  
author={苏剑林},  
year={2023},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/9768}},  
} 


---

## 完整数学推导与理论分析

本节将详细推导Viterbi算法的数学基础，包括HMM模型、动态规划、前向后向算法，以及Viterbi Sampling的完整理论。

### 一、隐马尔可夫模型(HMM)基础

#### 1.1 HMM的三要素

**定义**：隐马尔可夫模型是关于时序的概率模型，描述一个隐藏的马尔可夫链随机生成不可观测的状态序列，再由各个状态生成观测序列的过程。

<div class="definition-box">

**HMM的三要素**：

设状态集合$\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$，观测集合$\mathcal{O} = \{o_1, o_2, \ldots, o_M\}$。

1. **初始状态概率**：$\boldsymbol{\pi} = (\pi_1, \ldots, \pi_N)$
   \begin{equation}
   \pi_i = P(q_1 = s_i), \quad \sum_{i=1}^{N} \pi_i = 1 \tag{1}
   \end{equation}

2. **状态转移概率**：$\mathbf{A} = (a_{ij})_{N \times N}$
   \begin{equation}
   a_{ij} = P(q_{t+1} = s_j | q_t = s_i), \quad \sum_{j=1}^{N} a_{ij} = 1 \tag{2}
   \end{equation}

3. **发射概率（观测概率）**：$\mathbf{B} = (b_j(o_k))$
   \begin{equation}
   b_j(o_k) = P(x_t = o_k | q_t = s_j), \quad \sum_{k=1}^{M} b_j(o_k) = 1 \tag{3}
   \end{equation}

HMM记为$\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$。

</div>

**马尔可夫性假设**：

1. **齐次马尔可夫假设**：状态转移概率与时间无关
   \begin{equation}
   P(q_t | q_{t-1}, \ldots, q_1) = P(q_t | q_{t-1}) \tag{4}
   \end{equation}

2. **观测独立性假设**：观测只依赖于当前状态
   \begin{equation}
   P(x_t | q_1, \ldots, q_T, x_1, \ldots, x_{t-1}, x_{t+1}, \ldots, x_T) = P(x_t | q_t) \tag{5}
   \end{equation}

#### 1.2 HMM的应用：Unigram分词

在Unigram分词中，HMM的对应关系：

- **观测序列**：原始文本（字节序列）$\boldsymbol{c} = (c_1, c_2, \ldots, c_L)$
- **状态序列**：切分方案（词序列）$\boldsymbol{w} = (w_1, w_2, \ldots, w_K)$
- **状态转移**：从一个词到下一个词（实际上是独立的，$a_{ij} = 1$）
- **发射概率**：词的概率$P(w_i)$

**简化的Unigram模型**：

由于假设词之间独立（unigram），不存在真正的状态转移，只有发射概率：
\begin{equation}
P(\boldsymbol{w}) = \prod_{i=1}^{K} P(w_i) \tag{6}
\end{equation}

**分词问题的形式化**：

给定字节序列$\boldsymbol{c} = c_1c_2\cdots c_L$，找到概率最大的切分方案：
\begin{equation}
\boldsymbol{w}^* = \arg\max_{\boldsymbol{w} \in \Omega(\boldsymbol{c})} P(\boldsymbol{w}) \tag{7}
\end{equation}

其中$\Omega(\boldsymbol{c})$是所有能够完整覆盖$\boldsymbol{c}$的切分方案的集合。

### 二、Viterbi算法：动态规划解码

#### 2.1 问题设定

**目标**：给定观测序列$\boldsymbol{x} = (x_1, \ldots, x_T)$和HMM参数$\lambda$，找到最可能的状态序列：
\begin{equation}
\boldsymbol{q}^* = \arg\max_{\boldsymbol{q}} P(\boldsymbol{q} | \boldsymbol{x}, \lambda) = \arg\max_{\boldsymbol{q}} P(\boldsymbol{q}, \boldsymbol{x} | \lambda) \tag{8}
\end{equation}

**朴素方法的问题**：

遍历所有可能的状态序列，共$N^T$种，复杂度指数级！

#### 2.2 动态规划的思想

**关键观察**：最优路径的任何子路径也是最优的（最优子结构性质）。

定义变量$\delta_t(i)$：
\begin{equation}
\delta_t(i) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t = s_i, x_1, \ldots, x_t | \lambda) \tag{9}
\end{equation}

**含义**：到时刻$t$，以状态$s_i$结尾，观测序列为$x_1, \ldots, x_t$的所有路径中，概率最大的路径的概率。

#### 2.3 Viterbi递推公式

**初始化**（$t = 1$）：
\begin{equation}
\delta_1(i) = \pi_i \cdot b_i(x_1), \quad i = 1, \ldots, N \tag{10}
\end{equation}
\begin{equation}
\psi_1(i) = 0 \tag{11}
\end{equation}

**递推**（$t = 2, \ldots, T$）：
\begin{equation}
\delta_t(j) = \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(x_t) \tag{12}
\end{equation}
\begin{equation}
\psi_t(j) = \arg\max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \tag{13}
\end{equation}

其中$\psi_t(j)$记录了到达状态$j$的最优前驱状态。

**证明递推公式的正确性**：

\begin{equation}
\begin{aligned}
\delta_t(j) &= \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t = s_j, x_1, \ldots, x_t | \lambda) \\
&= \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, x_1, \ldots, x_{t-1} | \lambda) \\
&\quad \times P(q_t = s_j | q_{t-1}) \times P(x_t | q_t = s_j) \\
&= \max_{1 \leq i \leq N} \left[ \max_{q_1, \ldots, q_{t-2}} P(q_1, \ldots, q_{t-2}, q_{t-1} = s_i, x_1, \ldots, x_{t-1}) \right] \\
&\quad \times a_{ij} \times b_j(x_t) \\
&= \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(x_t)
\end{aligned} \tag{14}
\end{equation}

**终止**：
\begin{equation}
P^* = \max_{1 \leq i \leq N} \delta_T(i) \tag{15}
\end{equation}
\begin{equation}
q_T^* = \arg\max_{1 \leq i \leq N} \delta_T(i) \tag{16}
\end{equation}

**回溯**（$t = T-1, T-2, \ldots, 1$）：
\begin{equation}
q_t^* = \psi_{t+1}(q_{t+1}^*) \tag{17}
\end{equation}

#### 2.4 对数空间的Viterbi算法

**数值稳定性问题**：连乘很多小概率会导致下溢。

**解决方法**：在对数空间进行计算。

定义：
\begin{equation}
\tilde{\delta}_t(i) = \log \delta_t(i) \tag{18}
\end{equation}

递推公式变为：
\begin{equation}
\tilde{\delta}_t(j) = \max_{1 \leq i \leq N} [\tilde{\delta}_{t-1}(i) + \log a_{ij}] + \log b_j(x_t) \tag{19}
\end{equation}

**关键变化**：
- 乘法 → 加法
- $\max$操作保持不变

### 三、应用于Unigram分词的Viterbi算法

#### 3.1 分词的DAG表示

给定字节序列$\boldsymbol{c} = c_1c_2\cdots c_L$和词表$\mathcal{V}$。

**构建DAG（有向无环图）**：
- **节点**：位置$0, 1, 2, \ldots, L$
- **边**：如果$\overline{c_i c_{i+1} \cdots c_j} \in \mathcal{V}$，则存在边$(i-1) \to j$
- **边权**：$\log P(\overline{c_i \cdots c_j})$

**最优路径**：从节点0到节点$L$的最大权重路径。

#### 3.2 前向Viterbi算法（Forward DP）

定义$S^*(i)$：从位置0到位置$i$的最优路径得分。

**初始化**：
\begin{equation}
S^*(0) = 0 \tag{20}
\end{equation}

**递推**（$i = 1, 2, \ldots, L$）：
\begin{equation}
S^*(i) = \max_{j < i, \overline{c_{j+1}\cdots c_i} \in \mathcal{V}} [S^*(j) + \log P(\overline{c_{j+1}\cdots c_i})] \tag{21}
\end{equation}

同时记录前驱：
\begin{equation}
\text{prev}(i) = \arg\max_{j < i, \overline{c_{j+1}\cdots c_i} \in \mathcal{V}} [S^*(j) + \log P(\overline{c_{j+1}\cdots c_i})] \tag{22}
\end{equation}

**回溯**：
\begin{equation}
i \gets L; \quad \text{tokens} = [] \\
\text{while } i > 0: \\
\quad j = \text{prev}(i) \\
\quad \text{tokens.append}(\overline{c_{j+1}\cdots c_i}) \\
\quad i \gets j \tag{23}
\end{equation}

#### 3.3 AC自动机加速

**问题**：对于每个位置$i$，需要遍历所有$j < i$检查是否存在词，复杂度$O(L^2)$。

**AC自动机（Aho-Corasick Automaton）**：

预处理词表，构建失配指针，可以在$O(L)$时间内找到所有词的终止位置。

**算法流程**：
1. 预处理：构建AC自动机（$O(|\mathcal{V}| \cdot \bar{L}_w)$，$\bar{L}_w$是平均词长）
2. 扫描：用AC自动机扫描文本，找到所有可能的词$(s, e, w)$（起始、结束、词本身）
3. DP：按结束位置$e$排序，依次计算$S^*(e)$

**复杂度**：$O(L \cdot m)$，其中$m$是词表中最大词长。

#### 3.4 伪代码实现

```python
def viterbi_decoding(text, vocab, prob):
    """
    text: 输入字节序列
    vocab: 词表（Trie或AC自动机）
    prob: 词的对数概率函数，prob(w) = log P(w)
    """
    L = len(text)
    # 初始化
    S = [-float('inf')] * (L + 1)
    prev = [None] * (L + 1)
    S[0] = 0.0

    # 使用AC自动机找到所有可能的词
    candidates = vocab.find_all_words(text)  # [(start, end, word), ...]

    # 按结束位置分组
    candidates_by_end = defaultdict(list)
    for start, end, word in candidates:
        candidates_by_end[end].append((start, word))

    # 动态规划
    for e in range(1, L + 1):
        for s, word in candidates_by_end[e]:
            score = S[s] + prob(word)
            if score > S[e]:
                S[e] = score
                prev[e] = (s, word)

    # 回溯
    if S[L] == -float('inf'):
        return None  # 无法切分

    tokens = []
    pos = L
    while pos > 0:
        s, word = prev[pos]
        tokens.append(word)
        pos = s

    return tokens[::-1]
```

### 四、前向-后向算法对比

#### 4.1 前向算法（Forward Algorithm）

**目的**：计算观测序列的概率$P(\boldsymbol{x}|\lambda)$。

定义前向变量：
\begin{equation}
\alpha_t(i) = P(x_1, \ldots, x_t, q_t = s_i | \lambda) \tag{24}
\end{equation}

**递推**：
\begin{equation}
\alpha_t(j) = \left[\sum_{i=1}^{N} \alpha_{t-1}(i) a_{ij}\right] b_j(x_t) \tag{25}
\end{equation}

**与Viterbi的对比**：
- 前向：$\sum$（求和，所有路径）
- Viterbi：$\max$（最大值，最优路径）

#### 4.2 后向算法（Backward Algorithm）

定义后向变量：
\begin{equation}
\beta_t(i) = P(x_{t+1}, \ldots, x_T | q_t = s_i, \lambda) \tag{26}
\end{equation}

**递推**（从$T$到$1$）：
\begin{equation}
\beta_t(i) = \sum_{j=1}^{N} a_{ij} b_j(x_{t+1}) \beta_{t+1}(j) \tag{27}
\end{equation}

#### 4.3 前向-后向的应用

**边缘概率**：
\begin{equation}
P(q_t = s_i | \boldsymbol{x}, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(\boldsymbol{x}|\lambda)} = \frac{\alpha_t(i) \beta_t(i)}{\sum_{j=1}^{N} \alpha_T(j)} \tag{28}
\end{equation}

**边缘概率 vs Viterbi路径**：
- 边缘概率：考虑所有路径，给出每个位置每个状态的概率
- Viterbi：只考虑最优路径

### 五、从Viterbi Decoding到Viterbi Sampling

#### 5.1 采样的动机

**Viterbi Decoding的局限**：
1. **确定性**：总是输出同一个结果
2. **过度自信**：即使多个切分方案概率接近，也只返回一个
3. **训练-推理不一致**：训练时可能需要多样性

**Viterbi Sampling的目标**：
- 依概率采样切分方案
- 概率高的方案被采样到的概率大
- 保持Viterbi Decoding的效率（线性复杂度）

#### 5.2 朴素采样方法：N-best搜索

**Subword Regularization的做法**：

1. 找到概率最高的$n$个切分方案$\boldsymbol{w}_1^*, \ldots, \boldsymbol{w}_n^*$
2. 计算权重：
   \begin{equation}
   p_i = \frac{P(\boldsymbol{w}_i^*)^\alpha}{\sum_{j=1}^{n} P(\boldsymbol{w}_j^*)^\alpha} \tag{29}
   \end{equation}
3. 按$p_i$进行采样

**N-best Viterbi算法**：

维护每个位置的top-$n$路径，复杂度$O(nLm)$，其中$m$是最大词长。

**缺点**：
- 效率低（$n$倍复杂度）
- 需要选择$n$（太小：覆盖不足；太大：浪费）

#### 5.3 Viterbi Sampling的核心思想

**问题重新表述**：

Viterbi Decoding的判据是：
\begin{equation}
\text{if } \text{score}_{\text{new}} > \text{score}_{\text{old}} \text{ then accept new} \tag{30}
\end{equation}

**随机化判据**：
\begin{equation}
\text{if } \varepsilon < \sigma(\alpha(\text{score}_{\text{new}} - \text{score}_{\text{old}})) \text{ then accept new} \tag{31}
\end{equation}

其中：
- $\varepsilon \sim U[0,1]$：均匀随机数
- $\sigma(x) = 1/(1 + e^{-x})$：Sigmoid函数
- $\alpha > 0$：温度参数

**性质**：
1. 当$\text{score}_{\text{new}} > \text{score}_{\text{old}}$时，接受概率$> 0.5$
2. 分数差距越大，接受概率越极端（接近0或1）
3. $\alpha \to \infty$时，退化为Viterbi Decoding

#### 5.4 Viterbi Sampling的递推公式

**修改Viterbi递推**：

原本：
\begin{equation}
\text{if } S[j] + s(w) > S[i] \text{ then} \\
\quad S[i] = S[j] + s(w) \\
\quad \text{prev}[i] = (j, w) \tag{32}
\end{equation}

随机化为：
\begin{equation}
\text{score}_{\text{new}} = S[j] + s(w) \\
\text{score}_{\text{old}} = S[i] \\
p_{\text{accept}} = \sigma(\alpha(\text{score}_{\text{new}} - \text{score}_{\text{old}})) \\
\text{if } \varepsilon < p_{\text{accept}} \text{ then} \\
\quad S[i] = \text{score}_{\text{new}} \\
\quad \text{prev}[i] = (j, w) \tag{33}
\end{equation}

**初始化**：
\begin{equation}
S[i] = -\infty, \quad i = 1, \ldots, L \tag{34}
\end{equation}

#### 5.5 保序性分析

**定义**：如果切分方案$\boldsymbol{w}_1$的得分高于$\boldsymbol{w}_2$，即$P(\boldsymbol{w}_1) > P(\boldsymbol{w}_2)$，那么在采样中$\boldsymbol{w}_1$被采到的概率是否也更高？

**Viterbi Sampling的保序性**：

不是严格保序的，但近似保序。

**分析**：考虑两条路径$A$和$B$，得分分别为$s_A$和$s_B$，$s_A > s_B$。

如果两条路径在位置$i$竞争：
\begin{equation}
P(\text{选择}A) = \sigma(\alpha(s_A - s_B)) > 0.5 \tag{35}
\end{equation}

但是，如果路径涉及多步竞争，每步都有随机性，最终的排序可能改变。

**实证观察**：在实际应用中，得分高的方案确实更常被采样到，尤其是当$\alpha$较大时。

### 六、数值稳定性与实现技巧

#### 6.1 LogSumExp技巧

前向算法涉及到求和：
\begin{equation}
\alpha_t(j) = \left[\sum_{i=1}^{N} \alpha_{t-1}(i) a_{ij}\right] b_j(x_t) \tag{36}
\end{equation}

在对数空间：
\begin{equation}
\log \alpha_t(j) = \log \left[\sum_{i=1}^{N} e^{\log \alpha_{t-1}(i) + \log a_{ij}}\right] + \log b_j(x_t) \tag{37}
\end{equation}

直接计算$\sum e^{x_i}$可能溢出。使用LogSumExp：
\begin{equation}
\text{LSE}(x_1, \ldots, x_N) = x_{\max} + \log \sum_{i=1}^{N} e^{x_i - x_{\max}} \tag{38}
\end{equation}

其中$x_{\max} = \max_i x_i$。

#### 6.2 Sigmoid的数值稳定计算

Sigmoid函数：
\begin{equation}
\sigma(x) = \frac{1}{1 + e^{-x}} \tag{39}
\end{equation}

**问题**：
- 当$x$很大时，$e^{-x} \to 0$，但计算$e^{-x}$可能下溢
- 当$x$很小时，$e^{-x} \to \infty$，上溢

**稳定计算**：
\begin{equation}
\sigma(x) = \begin{cases}
\frac{1}{1 + e^{-x}}, & x \geq 0 \\
\frac{e^x}{1 + e^x}, & x < 0
\end{cases} \tag{40}
\end{equation}

**推导**：
\begin{equation}
\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} = \frac{e^x}{1 + e^x} \tag{41}
\end{equation}

两种形式等价，但数值稳定性不同。

#### 6.3 随机数生成的效率

Viterbi Sampling每次DP更新都需要一个随机数$\varepsilon \sim U[0,1]$。

**优化方法**：
1. **批量生成**：预先生成一大批随机数
2. **快速RNG**：使用Xorshift等快速随机数生成器
3. **向量化**：使用SIMD指令并行生成

**Python示例**：
```python
# 预生成随机数
np.random.seed(42)
random_pool = np.random.uniform(0, 1, size=10000)
rand_idx = 0

def get_random():
    global rand_idx
    r = random_pool[rand_idx]
    rand_idx = (rand_idx + 1) % len(random_pool)
    return r
```

### 七、采样质量分析

#### 7.1 期望值分析

**问题**：Viterbi Sampling采样到某个切分方案$\boldsymbol{w}$的概率是多少？

这是一个复杂的问题，因为采样过程涉及多步随机决策。

**近似分析**：

假设DP过程中，每个位置独立做决策（这是一个简化假设）。设位置$i$有$k_i$个候选词结尾，得分分别为$s_1^{(i)}, \ldots, s_{k_i}^{(i)}$。

采样到得分为$s_j^{(i)}$的候选的概率约为：
\begin{equation}
p_j^{(i)} \approx \frac{e^{\alpha s_j^{(i)}}}{\sum_{\ell=1}^{k_i} e^{\alpha s_\ell^{(i)}}} \tag{42}
\end{equation}

这类似于Softmax！

整条路径被采样到的概率（近似）：
\begin{equation}
P_{\text{sample}}(\boldsymbol{w}) \approx \prod_{i \in \boldsymbol{w}} p_j^{(i)} \tag{43}
\end{equation}

#### 7.2 与真实概率的关系

**理想情况**：我们希望采样概率正比于真实概率：
\begin{equation}
P_{\text{sample}}(\boldsymbol{w}) \propto P(\boldsymbol{w}) = e^{s(\boldsymbol{w})} \tag{44}
\end{equation}

其中$s(\boldsymbol{w}) = \sum_{w_i \in \boldsymbol{w}} \log P(w_i)$。

**Viterbi Sampling的偏差**：

由于多次二选一的稀释效应（后面的候选"占便宜"），Viterbi Sampling的采样概率与真实概率不完全一致。

**文章的改进方向**：
- 使用水塘采样（Reservoir Sampling）
- 缓存累积概率
- 详见下一篇文章《随机分词再探》

#### 7.3 温度参数$\alpha$的影响

**$\alpha \to 0$**：
\begin{equation}
\sigma(\alpha \Delta s) \to \begin{cases}
1, & \Delta s > 0 \\
0.5, & \Delta s = 0 \\
0, & \Delta s < 0
\end{cases} \tag{45}
\end{equation}

接近均匀采样（所有正分数差都以概率0.5接受）。

**$\alpha \to \infty$**：
\begin{equation}
\sigma(\alpha \Delta s) \to \begin{cases}
1, & \Delta s > 0 \\
0.5, & \Delta s = 0 \\
0, & \Delta s < 0
\end{cases} \tag{46}
\end{equation}

退化为Viterbi Decoding（总是选择分数高的）。

**$\alpha = 0.1$（经验值）**：
- 保持一定的随机性
- 同时偏向高分方案
- 在训练中引入正则化效果

### 八、与BPE Dropout的对比

#### 8.1 BPE Dropout

BPE（Byte Pair Encoding）使用贪心merge策略，每次选择频率最高的字节对合并。

**BPE Dropout**：在merge过程中，以概率$p$随机跳过某些merge操作。

**算法**：
```python
def bpe_dropout(text, merge_rules, p_dropout):
    for rule in merge_rules:
        if random.random() > p_dropout:
            text = apply_merge(text, rule)
    return text
```

#### 8.2 对比

| 方面 | Viterbi Sampling | BPE Dropout |
|------|------------------|-------------|
| 适用模型 | Unigram | BPE |
| 随机化方式 | DP过程中随机选择 | Merge过程中随机跳过 |
| 概率保证 | 近似按概率采样 | 无明确概率保证 |
| 实现复杂度 | 中等 | 简单 |
| 效率 | 与确定性版本相近 | 与确定性版本相同 |

#### 8.3 Subword Regularization对比

**方法**：找top-$n$个切分方案，按概率加权采样。

**优点**：
- 精确按概率采样
- 理论保证强

**缺点**：
- 效率低（$n$倍复杂度）
- 需要调参$n$

**Viterbi Sampling的优势**：
- 接近原始效率
- 自动适应（无需调$n$）
- 实现简单

### 九、实际应用与效果

#### 9.1 训练数据增强

**应用场景**：训练语言模型时，每个epoch使用不同的分词结果。

**效果**：
- 增强模型鲁棒性
- 减少对特定分词的过拟合
- 改善out-of-vocabulary (OOV)处理

**示例**：
```python
# 训练时
for epoch in range(num_epochs):
    for text in dataset:
        # 每次使用随机分词
        tokens = tokenizer.tokenize(text, alpha=0.1, random=True)
        loss = model.train_step(tokens)
```

#### 9.2 速度对比

根据文章的实验数据：

| 方法 | 确定性速度 | 随机性速度 | 速度比 |
|------|-----------|-----------|--------|
| SP-BPE | 1.36M bytes/sec | 1.25M bytes/sec | 92% |
| **SP-Unigram** | **5.65M bytes/sec** | **1.28M bytes/sec** | **23%** |
| BytePiece (Viterbi Sampling) | 1.95M bytes/sec | 1.36M bytes/sec | **70%** |

**观察**：
- Subword Regularization (SP-Unigram随机版)速度下降到23%
- Viterbi Sampling保持70%的速度
- **3倍加速**！

#### 9.3 分词多样性

**实验**（从文章）：
```python
text = '今天天气不错'

# 确定性分词
print(tokenizer.tokenize(text))
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']

# Viterbi Sampling (alpha=0.1)
for i in range(5):
    print(tokenizer.tokenize(text, alpha=0.1))
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6', b'\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8', b'\x8d', b'\xe9\x94', b'\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9', b'\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb', b'\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9', b'\xe6\xb0\x94\xe4\xb8\x8d', b'\xe9\x94', b'\x99']
```

**分析**：
- 确定性版本总是输出同一结果
- 随机版本产生多种切分
- 某些常见切分出现频率更高（如"今天"、"天气"）

### 十、理论保证与局限性

#### 10.1 Viterbi Sampling的理论性质

**性质1（近似保序）**：高分切分方案被采样到的概率更大。

**性质2（效率）**：复杂度与Viterbi Decoding相同，都是$O(Lm)$。

**性质3（可控随机性）**：通过$\alpha$控制随机性强度。

#### 10.2 未解决的问题

**问题1**：采样概率的精确表达式是什么？

目前只有近似分析，缺乏精确的概率分布。

**问题2**：如何保证采样的无偏性？

Viterbi Sampling由于稀释效应，不是严格按$P(\boldsymbol{w})$采样。

**解决方向**：
- 水塘采样（下一篇文章）
- 基于CDF的逆变换采样

#### 10.3 适用场景

**适合**：
- 需要快速随机分词
- 训练数据增强
- 轻量级正则化

**不适合**：
- 需要精确概率分布的场景
- 理论分析要求严格无偏采样

### 十一、总结

Viterbi算法是序列标注和分词问题的核心算法，通过动态规划在线性时间内找到最优路径。

**Viterbi Decoding**：
\begin{equation}
S^*(i) = \max_{j < i} [S^*(j) + \log P(w_{j+1:i})] \tag{47}
\end{equation}

**Viterbi Sampling（初步版本）**：
\begin{equation}
\text{accept new if } \varepsilon < \sigma(\alpha \cdot \Delta s) \tag{48}
\end{equation}

**优势**：
- ✅ 保持线性复杂度
- ✅ 引入随机性
- ✅ 实现简单
- ✅ 速度快（相比Subword Regularization）

**局限**：
- ❌ 采样概率不精确
- ❌ 稀释效应（后续候选占便宜）
- ❌ 理论保证不足

**下一步**：在下一篇文章《随机分词再探》中，我们将解决稀释效应，通过水塘采样实现完美采样，达到与Subword Regularization等价的效果，同时保持高效率。

