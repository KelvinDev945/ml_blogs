---
title: NBCE：使用朴素贝叶斯扩展LLM的Context处理长度
slug: nbce使用朴素贝叶斯扩展llm的context处理长度
date: 2023-05-23
tags: 语言模型, 外推, LLM, 贝叶斯, 生成模型
status: pending
---

# NBCE：使用朴素贝叶斯扩展LLM的Context处理长度

**原文链接**: [https://spaces.ac.cn/archives/9617](https://spaces.ac.cn/archives/9617)

**发布日期**: 

---

> 在LLM时代还玩朴素贝叶斯（Naive Bayes）？

这可能是许多读者在看到标题后的首个想法。确实如此，当古老的朴素贝叶斯与前沿的LLM相遇时，产生了令人惊讶的效果——我们可以直接扩展现有LLM模型的Context处理长度，无需对模型进行微调，也不依赖于模型架构，具有线性效率，而且效果看起来还不错——这就是本文所提出的NBCE（**N** aive **B** ayes-based **C** ontext **E** xtension）方法。

## 摸石过河 #

假设$T$为要生成的token序列，$S_1,S_2,\cdots,S_n$是给定的若干个相对独立的Context集合（比如$n$个不同的段落，至少不是一个句子被分割为两个片段那种），假设它们的总长度已经超过了训练长度，而单个$S_k$加$T$还在训练长度内。我们需要根据$S_1,S_2,\cdots,S_n$生成$T$，即估计$p(T|S_1, S_2,\cdots,S_n)$。

简单来说，朴素贝叶斯就是“贝叶斯公式+独立假设”。根据贝叶斯公式：  
\begin{equation}p(T|S_1, S_2,\cdots,S_n) \propto p(S_1, S_2,\cdots,S_n|T)p(T)\end{equation}  
这里的$\propto$，是省去了与$T$无关的常数因子。根据（条件）独立假设：  
\begin{equation}p(S_1, S_2,\cdots,S_n|T) = \prod_{k=1}^n p(S_k|T)\end{equation}  
所以有  
\begin{equation}p(T|S_1, S_2,\cdots,S_n) \propto p(T)\prod_{k=1}^n p(S_k|T)\end{equation}  
再次根据贝叶斯公式$p(S_k|T) \propto \frac{p(T|S_k)}{p(T)}$，得到  
\begin{equation}p(T|S_1, S_2,\cdots,S_n) \propto \frac{1}{p^{n-1}(T)}\prod_{k=1}^n p(T|S_k)\end{equation}  
或者  
\begin{equation}\log p(T|S_1, S_2,\cdots,S_n) = \color{red}{\sum_{k=1}^n \log p(T|S_k)} - \color{green}{(n-1)\log p(T)} + \color{skyblue}{\text{常数}}\label{eq:nbce-1}\end{equation}

这里的$\color{red}{p(T|S_k)}$和$\color{green}{p(T)}$都可以直接用现有的LLM进行计算，而且只要是语言模型都行，跟架构无关，也不需要用长文本微调。其中，$\color{red}{p(T|S_k)}$是单个Context所预测的概率，$\color{green}{p(T)}$则无Context（或者Context为空）的概率，并且多个Context可以放在同一个batch中并行计算，计算量随着Context数的增加是线性增长的。

## 抽丝剥茧 #

当然，朴素贝叶斯依赖于独立假设，这会限制它的实际效果。为了“青出于蓝而胜于蓝”，我们不妨将式$\eqref{eq:nbce-1}$进一步“抽丝剥茧”、“去芜存菁”，以达到更好的效果。

首先我们记$\log p(T|S) = [\log p(T|S_1),\cdots,\log p(T|S_n)]$，以及  
\begin{equation}\overline{\log p(T|S)} = \frac{1}{n}\sum_{k=1}^n \log p(T|S_k)\end{equation}  
并设$\beta = n - 1$，那么式$\eqref{eq:nbce-1}$可以重写为  
\begin{equation}\log p(T|S_1, S_2,\cdots,S_n) = \color{red}{(\beta + 1)\overline{\log p(T|S)}} - \color{green}{\beta\log p(T)} + \color{skyblue}{\text{常数}}\label{eq:nbce-2}\end{equation}

重写为上述形式后，自然而言地引出了两个问题：

> 1、如果将$\beta$作为超参数来调，是否可能取得更好的效果？
> 
> 2、$\overline{\log p(T|S)}$就是$\log p(T|S)$的Average Pooling，那么换成其他Pooling方法（简记为$\mathcal{P}$）是否有更好的效果？即  
>  \begin{equation}\log p(T|S_1, S_2,\cdots,S_n) = \color{red}{(\beta + 1)\mathcal{P}[\log p(T|S)]} - \color{green}{\beta\log p(T)} + \color{skyblue}{\text{常数}}\label{eq:nbce-3}\end{equation}

于是笔者在7B模型上围绕这两个问题进行调试，得到的初步结论是：在阅读理解场景中Max Pooling配合$\beta=0.25$，用Greedy Search总体表现比较好，然而Random Sample出来的结果基本不可读。

## 最终方案 #

为什么会出现Greedy Search好而Random Sample差的情况呢？我们知道，Random Sample是“按照分布采样”，它的效果差说明Max Pooling的结果不是一个合理的分布；而Greedy Search只关心最大概率者，而不关心分布的合理性，它的效果好告诉我们概率最大的token正确性较高。

概率越大说明不确定性越低，所以为了改善Random Sample的效果，我们将Pooling方式改为直接输出不确定性最低的那个分布：  
\begin{equation}\begin{aligned}  
&\mathcal{P}[\log p(T|S)] = \log p(T|S_{\color{red}{k}}) \\\\[5pt]  
&\color{red}{k} = \mathop{\text{argmin}} \big\\{H_1,H_2,\cdots,H_n\big\\} \\\\[5pt]  
&H_i = -\sum_T p(T|S_i)\log p(T|S_i)  
\end{aligned}\end{equation}  
代入到式$\eqref{eq:nbce-3}$，就是最终的NBCE（**N** aive **B** ayes-based **C** ontext **E** xtension）。

值得指出的是，虽然我们的出发点是朴素贝叶斯，但一般化后的式$\eqref{eq:nbce-3}$已经超出了常规的朴素贝叶斯的范畴，同时保留了朴素贝叶斯的可解释性。不难看出，式$\eqref{eq:nbce-3}$的形式很是直观：

> 1、不同Context的预测结果通过方法$\mathcal{P}$聚合（或者说投票）在一起（权重为$\beta+1$），并减去无Context的预测结果（权重为$\beta$）；
> 
> 2、之所以要减去无Context预测结果，是为了让模型更加倾向于结合Context而不是纯粹根据自身知识储备来回答（注：3天后出现在Arxiv的论文[《Trusting Your Evidence: Hallucinate Less with Context-aware Decoding》](https://papers.cool/arxiv/2305.14739)也提出了相同的技巧用来减少幻觉）；
> 
> 3、不同场景可以选择不同的$\beta$，比如需要结合Context做阅读理解的，可以考虑较大的$\beta$，如果偏向于自由创作，则选择较小的$\beta$，笔者认为$\beta\geq -1$都是合理的。

## 参考实现 #

下面给出NBCE的参考实现：

> **Github:<https://github.com/bojone/NBCE>**

从演示代码可以看出，NBCE的实现很简单，只需要修改一下解码函数中的logits构建方式，跟解码算法的选择并不冲突。

[![Naive Bayes-based Context Extension（NBCE）示意图](/usr/uploads/2023/05/2311579094.svg)](/usr/uploads/2023/05/2311579094.svg "点击查看原图")

Naive Bayes-based Context Extension（NBCE）示意图

所给的Demo包含12段不同的Context，总长度为9000多字，连同8个问题一次性输入到模型中（模型训练长度为2048，参数量为7B，可以在[OpenBuddy](https://openbuddy.ai/)下载），模型能够逐一根据所给Context正确回答这8个问题。值得指出的是，所有的Context、问题和答案加起来，超过了1万字！另外，有朋友简单尝试了简历匹配和作文打分应用，效果也尚可，非常建议大家亲自调试一下。

## 相关工作 #

扩展LLM的Context长度其实已有不少，但多数是通过结合检索或者摘要的方式来缩短样本的长Context，如[Unlimiformer](https://papers.cool/arxiv/2305.01625)。由于不是直接处理长Context，因此通常无法做精细的阅读理解，而且这些方案往往需要在训练阶段就考虑进去，而不是事后即插即用到已有的LLM模型中。

在NBCE之前，能够不微调地扩展Context长度的方案是Parallel Context Window（下面简称PCW），出自论文[《Parallel Context Windows for Large Language Models》](https://papers.cool/arxiv/2212.10947)和[《Structured Prompting: Scaling In-Context Learning to 1,000 Examples》](https://papers.cool/arxiv/2212.06713)，两篇论文是同一时期不同作者的工作，但所提的方法只有细微的差别，因此这里都将它们叫做PCW。

PCW适用于Self Attention模型，主要修改包括Position Encoding和Attention Mask，如下图所示：  


[![Parallel Context Window](/usr/uploads/2023/05/2761768682.svg)](/usr/uploads/2023/05/2761768682.svg "点击查看原图")

Parallel Context Window

首先确定Context的最大长度$L$（图中为6），然后每个Context的最后一个位置编码为$L-1$，倒数第二个位置编码为$L-2$，...，依此类推，这种编码方式我们称为“右对齐”（或者“左缩进”）；另一边，对于Task Tokens部分（Prompt+生成内容），我们的位置编码是$L,L+1,L+2,\cdots$。每个Context单独编码，所以对应的Attention Mask是分块对角矩阵，而因为是LM，所以是分块对角下三角阵；至于Task Tokens部分需要结合所有的Context，所以它需要Attention到所有Context（以及它自身）。这样一来，如果将每个Context单独拿出来，和Task Tokens拼在一起，其Attention模式就跟原本的LM一致了。

或许有读者看出，其实NBCE跟PCW有着很相似的特性，比如对于Context都是无序的、平权的。事实上，如果将NBCE应用到单层单头注意力模型中，那么结果大致上就是PCW。为了显示这一点，我们写出单层单头注意力的语言模型为  
\begin{equation}p(x_t|x_{< t}) = softmax\left(\sum_{i=1}^t a_{t,i}v_i W\right)\end{equation}  
所以大致上有$\log p(x_t|x_{< t}) \sim \sum\limits_{i=1}^t a_{t,i}v_i W$，接着代入到式$\eqref{eq:nbce-2}$并取$\beta=0$，得到  
\begin{equation}\log p(T|S_1, S_2,\cdots,S_n) \sim \frac{1}{n}\sum_{k=1}^n\left(\sum_{i\in S_k} a_{T,i}v_i\right) W = \left(\sum_{i\in S_1\oplus\cdots\oplus S_n} \frac{a_{T,i}}{n}v_i\right) W \end{equation}  
这里假设的是$T$是单个token，但其实已经不失一般性了，$\oplus$是拼接的意思。在上式中，$S_k\oplus T$是作为一个连续片段来推理的（NBCE的设定），所以它们的位置编码相邻，而$a_{T,i}/n$构成了$T$与所有$S_i$的一个整体Attention（求和同样是1），这些特性跟PCW其实是一致的，PCW只不过是以Attention Mask的方式更优雅地整合到每一层中。

因此，PCW大致上就是Average Pooling版的NBCE，我们实测也发现它跟Average Pooling版的NBCE有着相似的缺点——当Context数据增加时，输出的结果开始不够准确，具体表现为主题相关，但是作为问题的答案来说是错误的。

## 延伸思考 #

NBCE的一大缺点是无序性，即无法识别Context的输入顺序，这在续写故事等场景可能表现欠佳。为了缓解这一点，可以考虑在每一个Context前面加个能指示序信息的prefix，就好比小说中的“第一章”、“第二章”那样。

总的来说，目前笔者关于NBCE的测试都限于“阅读理解”场景，即“理解”长文本，能否用此方法来“生成”长文本，还是个未知数，期待大家的测试结果。

此外，还有一个有意思的问题是：

> 既然朴素贝叶斯都能在LLM领域能派上用场，那么其他传统概率模型（比如HMM）是否也能在LLM领域有它们的一席之地呢？

## 文章小结 #

本文提出了NBCE（Naive Bayes-based Context Extension），它基于朴素贝叶斯思想来扩展LLM的Context处理长度，有着即插即用、模型无关、无须微调、线性效率、实现简单等优点，并且看上去效果还不错，欢迎大家测试。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9617>_

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

苏剑林. (May. 23, 2023). 《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9617>

@online{kexuefm-9617,  
title={NBCE：使用朴素贝叶斯扩展LLM的Context处理长度},  
author={苏剑林},  
year={2023},  
month={May},  
url={\url{https://spaces.ac.cn/archives/9617}},  
} 


---

## 完整数学推导与理论分析

本节将对NBCE方法进行深入的数学推导和理论分析，包括朴素贝叶斯基础、条件独立性假设、与其他方法的对比等。

### 一、朴素贝叶斯的概率论基础

#### 1.1 贝叶斯公式与后验概率

**贝叶斯定理**是概率论中的基本定理，它描述了在获得新证据后如何更新我们对某个假设的信念。

<div class="formula-box">

**贝叶斯定理（单个条件）**：
\begin{equation}
P(A|B) = \frac{P(B|A)P(A)}{P(B)} \tag{1}
\end{equation}

其中：
- $P(A|B)$：后验概率（posterior），在观察到$B$后$A$的概率
- $P(B|A)$：似然（likelihood），在$A$条件下观察到$B$的概率
- $P(A)$：先验概率（prior），观察前对$A$的信念
- $P(B)$：边缘概率（evidence），$B$的总概率

</div>

**推导过程**：

从条件概率的定义出发：
\begin{equation}
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B|A) = \frac{P(A \cap B)}{P(A)} \tag{2}
\end{equation}

由于$P(A \cap B) = P(B \cap A)$（交换律），我们有：
\begin{equation}
P(A \cap B) = P(A|B)P(B) = P(B|A)P(A) \tag{3}
\end{equation}

因此：
\begin{equation}
P(A|B) = \frac{P(B|A)P(A)}{P(B)} \tag{4}
\end{equation}

**扩展到多个条件**：

对于多个观测变量$S_1, S_2, \ldots, S_n$和目标变量$T$，贝叶斯公式变为：
\begin{equation}
P(T|S_1, S_2, \ldots, S_n) = \frac{P(S_1, S_2, \ldots, S_n|T)P(T)}{P(S_1, S_2, \ldots, S_n)} \tag{5}
\end{equation}

由于分母$P(S_1, S_2, \ldots, S_n)$与$T$无关，在比较不同$T$的概率时可以视为归一化常数，因此：
\begin{equation}
P(T|S_1, S_2, \ldots, S_n) \propto P(S_1, S_2, \ldots, S_n|T)P(T) \tag{6}
\end{equation}

这里的$\propto$表示"正比于"。

#### 1.2 条件独立性假设

朴素贝叶斯的核心假设是**条件独立性**（conditional independence）。

<div class="definition-box">

**条件独立性定义**：

给定$T$，如果$S_1, S_2, \ldots, S_n$相互独立，则：
\begin{equation}
P(S_1, S_2, \ldots, S_n|T) = \prod_{k=1}^{n} P(S_k|T) \tag{7}
\end{equation}

更正式地，对于任意$i \neq j$：
\begin{equation}
P(S_i|S_j, T) = P(S_i|T) \tag{8}
\end{equation}

即：在给定$T$的条件下，知道$S_j$不会提供关于$S_i$的额外信息。

</div>

**为什么称为"朴素"**？

这个假设在现实中往往不成立。例如，对于文档分类：
- $S_1$：文档包含词"机器"
- $S_2$：文档包含词"学习"
- $T$：文档类别是"AI"

显然，$P(S_2|S_1, T) \neq P(S_2|T)$，因为"机器"和"学习"经常一起出现。

但即使假设不严格成立，朴素贝叶斯在实践中往往仍有很好的效果，这被称为朴素贝叶斯的"鲁棒性悖论"。

**应用到NBCE**：

在NBCE中，条件独立性假设为：
\begin{equation}
P(S_1, S_2, \ldots, S_n|T) = \prod_{k=1}^{n} P(S_k|T) \tag{9}
\end{equation}

其中$S_k$是第$k$个Context段落。这个假设的合理性取决于：
1. Context段落的划分方式（段落间应尽可能独立）
2. 任务特性（生成的$T$与各Context的关系）

#### 1.3 从贝叶斯公式到NBCE核心公式

将公式(6)和公式(9)结合：
\begin{equation}
P(T|S_1, \ldots, S_n) \propto P(T) \prod_{k=1}^{n} P(S_k|T) \tag{10}
\end{equation}

**关键转换**：我们想要的是$P(T|S_1, \ldots, S_n)$，但LLM直接给出的是$P(T|S_k)$和$P(T)$，如何建立联系？

再次应用贝叶斯公式到$P(S_k|T)$：
\begin{equation}
P(S_k|T) = \frac{P(T|S_k)P(S_k)}{P(T)} \tag{11}
\end{equation}

代入公式(10)：
\begin{equation}
\begin{aligned}
P(T|S_1, \ldots, S_n) &\propto P(T) \prod_{k=1}^{n} \frac{P(T|S_k)P(S_k)}{P(T)} \\
&= P(T) \cdot \frac{\prod_{k=1}^{n} P(T|S_k) \prod_{k=1}^{n} P(S_k)}{P(T)^n} \\
&= \frac{1}{P(T)^{n-1}} \prod_{k=1}^{n} P(T|S_k) \cdot \prod_{k=1}^{n} P(S_k)
\end{aligned} \tag{12}
\end{equation}

由于$\prod_{k=1}^{n} P(S_k)$与$T$无关（Context是给定的），可以并入比例常数：
\begin{equation}
P(T|S_1, \ldots, S_n) \propto \frac{1}{P(T)^{n-1}} \prod_{k=1}^{n} P(T|S_k) \tag{13}
\end{equation}

**取对数得到加法形式**：
\begin{equation}
\log P(T|S_1, \ldots, S_n) = \sum_{k=1}^{n} \log P(T|S_k) - (n-1)\log P(T) + \text{const} \tag{14}
\end{equation}

这正是文章主体部分的核心公式！

### 二、序列生成的自回归分解

在语言模型中，$T$不是单个token而是一个序列$T = (t_1, t_2, \ldots, t_m)$。

#### 2.1 自回归分解

根据概率链式法则：
\begin{equation}
P(T) = P(t_1, t_2, \ldots, t_m) = \prod_{j=1}^{m} P(t_j | t_{<j}) \tag{15}
\end{equation}

其中$t_{<j} = (t_1, \ldots, t_{j-1})$表示$t_j$之前的所有token。

对于条件概率$P(T|S_k)$：
\begin{equation}
P(T|S_k) = \prod_{j=1}^{m} P(t_j | t_{<j}, S_k) \tag{16}
\end{equation}

#### 2.2 Token级别的NBCE

对于每个位置$j$，应用NBCE公式(14)：
\begin{equation}
\log P(t_j | t_{<j}, S_1, \ldots, S_n) = \sum_{k=1}^{n} \log P(t_j|t_{<j}, S_k) - (n-1)\log P(t_j|t_{<j}) + \text{const} \tag{17}
\end{equation}

**实现细节**：

在生成第$j$个token时：
1. 对每个Context $S_k$，计算$\log P(t_j|t_{<j}, S_k)$（$n$次前向传播）
2. 计算无Context的$\log P(t_j|t_{<j})$（1次前向传播）
3. 按公式(17)组合得到最终的logits
4. 应用softmax或采样策略选择$t_j$

总计算量：$(n+1)$次前向传播/token，可以batch化为1次（$n+1$个样本的batch）。

### 三、参数$\beta$的一般化与解释

#### 3.1 从$n-1$到$\beta$

文章将$n-1$替换为可调参数$\beta$：
\begin{equation}
\log P(T|S_1, \ldots, S_n) = \sum_{k=1}^{n} \log P(T|S_k) - \beta \log P(T) + \text{const} \tag{18}
\end{equation}

改写为：
\begin{equation}
\log P(T|S_1, \ldots, S_n) = (\beta+1)\overline{\log P(T|S)} - \beta \log P(T) + \text{const} \tag{19}
\end{equation}

其中$\overline{\log P(T|S)} = \frac{1}{n}\sum_{k=1}^{n} \log P(T|S_k)$。

#### 3.2 $\beta$的物理意义

将公式(19)重新整理：
\begin{equation}
\begin{aligned}
\log P(T|S_1, \ldots, S_n) &= \overline{\log P(T|S)} + \beta[\overline{\log P(T|S)} - \log P(T)] + \text{const} \\
&= \overline{\log P(T|S)} + \beta \cdot \Delta + \text{const}
\end{aligned} \tag{20}
\end{equation}

其中$\Delta = \overline{\log P(T|S)} - \log P(T)$衡量"Context的贡献"。

**解释**：
- $\Delta > 0$：Context支持生成$T$（比无Context更可能）
- $\Delta < 0$：Context不支持生成$T$
- $\beta > 0$：放大Context的作用
- $\beta = 0$：完全依赖Context平均，忽略先验
- $\beta < 0$：减弱Context作用（甚至反向）

**不同任务的$\beta$选择**：

| 任务类型 | 推荐$\beta$ | 原因 |
|---------|------------|------|
| 严格QA（答案必须在Context中） | $\beta \geq 1$ | 强制模型依赖Context |
| 摘要生成 | $\beta \in [0.5, 1]$ | 平衡Context和语言流畅性 |
| 创意写作 | $\beta \in [-0.5, 0.5]$ | 允许模型发挥 |
| 知识融合 | $\beta \in [0, 0.5]$ | 结合Context和模型知识 |

#### 3.3 极限情况分析

**情况1：$\beta \to \infty$**

\begin{equation}
\frac{\log P(T|S_1, \ldots, S_n)}{\beta} \to \overline{\log P(T|S)} - \log P(T) \tag{21}
\end{equation}

此时只有$\Delta$的符号重要：
- 如果$\overline{\log P(T|S)} > \log P(T)$，则$P(T|S_1, \ldots, S_n) \to 1$
- 否则$P(T|S_1, \ldots, S_n) \to 0$

**情况2：$\beta = 0$**

\begin{equation}
\log P(T|S_1, \ldots, S_n) = \overline{\log P(T|S)} + \text{const} \tag{22}
\end{equation}

等价于对$n$个Context的预测做几何平均。

**情况3：$\beta = -1$**

\begin{equation}
\log P(T|S_1, \ldots, S_n) = \text{const} \tag{23}
\end{equation}

均匀分布！所有$T$等概率（这显然没有意义）。

### 四、Pooling策略的数学分析

#### 4.1 从Average Pooling到一般Pooling

公式(19)使用了Average Pooling：
\begin{equation}
\mathcal{P}_{\text{avg}}[\log P(T|S)] = \frac{1}{n}\sum_{k=1}^{n} \log P(T|S_k) \tag{24}
\end{equation}

**问题**：不同Context的质量可能不同，平均对待可能不是最优的。

#### 4.2 Max Pooling

\begin{equation}
\mathcal{P}_{\text{max}}[\log P(T|S)] = \max_{k=1,\ldots,n} \log P(T|S_k) \tag{25}
\end{equation}

**优点**：
- 选择最支持当前token的Context
- 天然稀疏（只用一个Context）
- 适合QA任务（答案通常只在一个段落中）

**缺点**：
- 忽略了其他Context的信息
- $\max$不可微（需要straight-through estimator）

#### 4.3 基于熵的Pooling（NBCE最终方案）

定义每个Context预测的熵：
\begin{equation}
H_k = -\sum_{t} P(t|t_{<j}, S_k) \log P(t|t_{<j}, S_k) \tag{26}
\end{equation}

选择熵最小的（最确定的）Context：
\begin{equation}
k^* = \arg\min_{k=1,\ldots,n} H_k \tag{27}
\end{equation}

\begin{equation}
\mathcal{P}_{\text{min-entropy}}[\log P(T|S)] = \log P(T|S_{k^*}) \tag{28}
\end{equation}

**优点**：
1. **置信度选择**：熵低意味着模型对该Context的预测更有信心
2. **动态适应**：不同位置可能选择不同的Context
3. **采样友好**：保持了完整的概率分布，支持采样解码

**与Max Pooling的区别**：
- Max Pooling：选择当前token概率最大的Context
- Min-Entropy Pooling：选择整个分布最确定的Context

这两者不一定相同！例如：
- Context A：$P(\text{"cat"}) = 0.6, P(\text{"dog"}) = 0.4$，$H_A \approx 0.67$
- Context B：$P(\text{"cat"}) = 0.5, P(\text{"dog"}) = 0.05, P(\text{"bird"}) = 0.45$，$H_B \approx 1.05$

如果当前考虑的token是"cat"：
- Max Pooling选A（0.6 > 0.5）
- Min-Entropy Pooling也选A（$H_A < H_B$）

但如果考虑的token是"mouse"，而：
- Context A：$P(\text{"mouse"}) = 0.001$
- Context B：$P(\text{"mouse"}) = 0.002$

Max Pooling会选B，但Min-Entropy Pooling仍选A（因为A的整体分布更确定）。

#### 4.4 熵的快速计算

在实现中，不需要完整遍历词表来计算熵。常用近似：

**Top-K熵**：
\begin{equation}
H_k^{\text{top-K}} = -\sum_{t \in \text{Top-K}} P(t|S_k) \log P(t|S_k) \tag{29}
\end{equation}

**基于Gini系数**（更快）：
\begin{equation}
\text{Gini}_k = 1 - \sum_{t \in \text{Top-K}} P(t|S_k)^2 \tag{30}
\end{equation}

Gini系数与熵高度相关，但计算更快（不需要log）。

### 五、数值稳定性：LogSumExp技巧

#### 5.1 数值上溢问题

直接计算$\log P(T)$涉及到：
\begin{equation}
P(T) = \sum_{t} e^{x_t} \tag{31}
\end{equation}

如果某些$x_t$很大（如50），$e^{50} \approx 10^{21}$会导致上溢。

#### 5.2 LogSumExp变换

定义LogSumExp函数：
\begin{equation}
\text{LSE}(\boldsymbol{x}) = \log \sum_{i} e^{x_i} \tag{32}
\end{equation}

**稳定计算方法**：
\begin{equation}
\text{LSE}(\boldsymbol{x}) = x_{\max} + \log \sum_{i} e^{x_i - x_{\max}} \tag{33}
\end{equation}

其中$x_{\max} = \max_i x_i$。

**证明**：
\begin{equation}
\begin{aligned}
\text{LSE}(\boldsymbol{x}) &= \log \sum_{i} e^{x_i} \\
&= \log \left(e^{x_{\max}} \sum_{i} e^{x_i - x_{\max}}\right) \\
&= x_{\max} + \log \sum_{i} e^{x_i - x_{\max}}
\end{aligned} \tag{34}
\end{equation}

现在$x_i - x_{\max} \leq 0$，所以$e^{x_i - x_{\max}} \in (0, 1]$，不会上溢！

#### 5.3 NBCE中的应用

在计算公式(19)时：
\begin{equation}
\begin{aligned}
&\log P(T|S_1, \ldots, S_n) \\
&= (\beta+1) \cdot \frac{1}{n}\sum_{k=1}^{n} \log P(T|S_k) - \beta \log P(T) + \text{const} \\
&= (\beta+1) \cdot \frac{1}{n}\sum_{k=1}^{n} [x_k - \text{LSE}(\boldsymbol{x}_k)] - \beta [\bar{x} - \text{LSE}(\bar{\boldsymbol{x}})] + \text{const}
\end{aligned} \tag{35}
\end{equation}

其中$x_k$是第$k$个Context在当前token上的logit，$\boldsymbol{x}_k$是整个词表的logit向量。

**伪代码**：
```python
def nbce_logits(context_logits, no_context_logits, beta=0.25):
    """
    context_logits: [n_contexts, vocab_size]
    no_context_logits: [vocab_size]
    """
    # 每个Context的log概率
    log_probs = context_logits - logsumexp(context_logits, dim=-1, keepdim=True)

    # 平均log概率
    avg_log_prob = log_probs.mean(dim=0)

    # 无Context的log概率
    no_context_log_prob = no_context_logits - logsumexp(no_context_logits)

    # NBCE组合
    final_logits = (beta + 1) * avg_log_prob - beta * no_context_log_prob

    return final_logits
```

### 六、与Parallel Context Windows (PCW)的详细对比

#### 6.1 PCW的基本原理

PCW的核心是修改Attention mask和Position encoding，使得：
1. 不同Context独立编码（分块对角Attention）
2. Position编码"右对齐"
3. 生成的Token能attend到所有Context

**数学表示**：

设Attention权重为：
\begin{equation}
\alpha_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{\ell} \exp(q_i^\top k_\ell / \sqrt{d})} \tag{36}
\end{equation}

在标准Transformer中，$j$遍历所有之前的位置。在PCW中：
- 对于Context内的token $i$：$j$只遍历同一Context内的位置
- 对于生成的token $i$：$j$遍历所有Context和之前生成的token

#### 6.2 数学等价性分析

文章声称"PCW大致上就是Average Pooling版的NBCE"。我们来验证这一点。

**单层单头Attention的输出**：
\begin{equation}
o_i = \sum_{j} \alpha_{ij} v_j W \tag{37}
\end{equation}

对于Softmax输出：
\begin{equation}
P(x_t|x_{<t}) = \text{softmax}(o_t W_{\text{out}}) \tag{38}
\end{equation}

粗略地：
\begin{equation}
\log P(x_t|x_{<t}) \approx o_t W_{\text{out}} = \sum_{j} \alpha_{tj} v_j W W_{\text{out}} \tag{39}
\end{equation}

**PCW的Attention模式**：

对于生成的token $t$，它attend到$n$个Context：
\begin{equation}
o_t = \sum_{k=1}^{n} \sum_{j \in S_k} \alpha_{tj} v_j W \tag{40}
\end{equation}

如果不同Context的权重相近（即$\sum_{j \in S_k} \alpha_{tj} \approx 1/n$），则：
\begin{equation}
o_t \approx \frac{1}{n} \sum_{k=1}^{n} \sum_{j \in S_k} v_j W = \frac{1}{n} \sum_{k=1}^{n} o_t^{(k)} \tag{41}
\end{equation}

其中$o_t^{(k)}$是只attend到Context $k$时的输出。

因此：
\begin{equation}
\log P(x_t|x_{<t}) \approx \frac{1}{n} \sum_{k=1}^{n} \log P(x_t|S_k, x_{<t}) \tag{42}
\end{equation}

这正是NBCE的$\beta=0$情况（Average Pooling，无先验惩罚）！

#### 6.3 差异分析

尽管数学上近似，PCW和NBCE有几个关键差异：

**1. 架构依赖性**
- PCW：仅适用于Transformer架构（依赖Attention机制）
- NBCE：架构无关，适用于任何语言模型

**2. 可调性**
- PCW：Attention权重由模型学习，不可直接调节
- NBCE：通过$\beta$和Pooling策略灵活控制

**3. 训练需求**
- PCW：需要在长Context数据上训练/微调
- NBCE：零样本应用到现有模型

**4. Context数量**
- PCW：受限于模型的最大序列长度
- NBCE：理论上无限（实践中受batch size限制）

**5. 计算效率**
- PCW：需要一次大规模Attention计算（复杂度$O(L^2)$，$L$为总长度）
- NBCE：需要$n+1$次独立前向传播，但可并行

#### 6.4 实验对比

原文提到两个关键观察：

**观察1**：PCW和Average Pooling版NBCE有相似的问题
- 当Context数量增加时，输出准确性下降
- 原因：相关信号被稀释

**观察2**：Min-Entropy Pooling显著改善
- 通过动态选择最相关的Context，避免稀释
- 这是NBCE相对PCW的主要优势

### 七、复杂度分析

#### 7.1 时间复杂度

设：
- $n$：Context数量
- $L$：每个Context长度
- $m$：生成序列长度
- $d$：模型隐藏维度
- $V$：词表大小

**NBCE（每个token）**：
1. $n$次前向传播（带Context）：$O(n \cdot L \cdot d^2)$
2. 1次前向传播（无Context）：$O(m \cdot d^2)$
3. Pooling和组合：$O(n \cdot V)$

总计：$O(n \cdot L \cdot d^2 + n \cdot V)$ per token

由于可以batch化：$O((n+1) \cdot L \cdot d^2 / B + n \cdot V)$，其中$B$是batch size。

**PCW（整个序列）**：
1. Position encoding：$O((nL + m) \cdot d)$
2. Attention：$O((nL + m)^2 \cdot d)$
3. 其余前向传播：$O((nL + m) \cdot d^2)$

对于大$n$，Attention项主导：$O(n^2 L^2 \cdot d)$

**对比**：
- NBCE：线性于$n$（$O(n)$）
- PCW：二次于$n$（$O(n^2)$，通过序列长度）

因此NBCE在$n$较大时更高效。

#### 7.2 空间复杂度

**NBCE**：
- 需要缓存$n+1$个模型的KV cache（如果使用）
- 空间：$O((n+1) \cdot m \cdot d)$

**PCW**：
- 单个模型，但序列更长
- 空间：$O((nL + m) \cdot d)$

对于$m \ll nL$，PCW空间更优；但如果$m$很大，NBCE可能更优。

### 八、实践建议与tricks

#### 8.1 Context分割策略

**原则**：满足条件独立性假设

**推荐方法**：
1. **段落级分割**：自然段落
2. **语义分割**：基于主题转换
3. **固定长度**：每$L$ tokens一段（简单但可能破坏语义）

**避免**：
- 句子中间截断
- 相互强依赖的段落分开

#### 8.2 $\beta$调参指南

**网格搜索**：$\beta \in \{-0.5, 0, 0.25, 0.5, 1.0, 2.0\}$

**根据任务调整**：
```python
if task == "extractive_qa":
    beta = 1.0  # 强制依赖Context
elif task == "abstractive_summarization":
    beta = 0.5  # 平衡Context和流畅性
elif task == "creative_writing":
    beta = 0.0  # 更多自由度
```

#### 8.3 动态$\beta$

可以根据生成过程动态调整：
\begin{equation}
\beta(j) = \beta_0 \cdot \exp(-\lambda j) \tag{43}
\end{equation}

- 开始时$\beta$大：紧跟Context
- 后期$\beta$减小：允许更多创造性

#### 8.4 混合Pooling

结合多种Pooling：
\begin{equation}
\mathcal{P}_{\text{hybrid}} = \alpha \mathcal{P}_{\text{min-H}} + (1-\alpha) \mathcal{P}_{\text{avg}} \tag{44}
\end{equation}

- $\alpha \to 1$：更稀疏，更依赖单一Context
- $\alpha \to 0$：更平滑，综合多个Context

#### 8.5 温度缩放

NBCE后添加温度：
\begin{equation}
P_{\text{final}}(t) = \text{softmax}\left(\frac{\log P_{\text{NBCE}}(t)}{\tau}\right) \tag{45}
\end{equation}

- $\tau < 1$：更确定性
- $\tau > 1$：更多样性

### 九、理论保证与局限性

#### 9.1 何时NBCE表现良好

**定理（非正式）**：如果：
1. Context $S_1, \ldots, S_n$关于$T$条件独立
2. 每个$S_k$包含生成$T$的部分信息
3. 先验$P(T)$不过度自信

则NBCE的后验$P(T|S_1, \ldots, S_n)$接近真实后验。

**证明思路**：
在条件独立性下，NBCE公式(14)是贝叶斯公式的精确解（忽略归一化）。

#### 9.2 失败情况

**情况1：强相关Context**

如果$S_1$和$S_2$高度相关（如重复），朴素贝叶斯会"双重计数"：
\begin{equation}
P(S_1, S_2|T) \neq P(S_1|T)P(S_2|T) \tag{46}
\end{equation}

实际上$P(S_1, S_2|T) \approx P(S_1|T)$（因为$S_2$没有新信息），但NBCE当作两个独立证据。

**缓解**：去重或权重衰减

**情况2：顺序依赖**

对于叙事性文本（如故事），Context顺序很重要，但NBCE无序处理。

**缓解**：添加位置前缀，如"第一段："、"第二段："

**情况3：长程依赖**

如果$T$需要综合多个Context的信息（如比较、因果），NBCE可能不如PCW。

**原因**：NBCE每次只看一个Context（Min-Entropy Pooling），缺少跨Context的信息整合。

**缓解**：使用Average Pooling或加权组合

### 十、扩展与未来方向

#### 10.1 层次化NBCE

对于更复杂的场景，可以构建两层结构：
1. **第一层**：每个Context内部用标准LLM
2. **第二层**：Context间用NBCE聚合

\begin{equation}
\log P(T|S_1, \ldots, S_n) = \mathcal{P}\left[\log P(T|S_1, \text{其他}), \ldots, \log P(T|S_n, \text{其他})\right] - \beta \log P(T) \tag{47}
\end{equation}

其中"其他"可以是其他Context的摘要。

#### 10.2 学习Pooling策略

用元学习学习最优Pooling：
\begin{equation}
\mathcal{P}_{\theta}[\log P(T|S)] = \sum_{k=1}^{n} w_k(\boldsymbol{S}, T; \theta) \log P(T|S_k) \tag{48}
\end{equation}

其中$w_k$是可学习的注意力权重。

#### 10.3 贝叶斯超参数

将$\beta$视为隐变量，进行贝叶斯推断：
\begin{equation}
P(T|S) = \int P(T|S, \beta) P(\beta|S) d\beta \tag{49}
\end{equation}

$P(\beta|S)$可以根据Context的特征（如长度、多样性）动态确定。

### 十一、总结

NBCE是一个简洁而强大的方法，通过朴素贝叶斯假设将长Context问题分解为多个短Context问题。其核心公式：

\begin{equation}
\log P(T|S_1, \ldots, S_n) = \color{red}{(\beta+1)\mathcal{P}[\log P(T|S)]} - \color{blue}{\beta \log P(T)} + \text{const} \tag{50}
\end{equation}

- **红色项**：Context证据的聚合
- **蓝色项**：先验知识的惩罚
- $\beta$：两者的权衡参数

**关键优势**：
1. ✅ 架构无关
2. ✅ 零样本应用
3. ✅ 线性复杂度
4. ✅ 灵活可调

**主要局限**：
1. ❌ 条件独立性假设可能不满足
2. ❌ 无法处理Context间的复杂交互
3. ❌ 无序处理可能丢失叙事结构

尽管有局限性，NBCE为超长Context处理提供了一个实用的解决方案，特别是在阅读理解、检索增强生成等场景中表现出色。

