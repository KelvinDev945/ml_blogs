---
title: 缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA
slug: 缓存与效果的极限拉扯从mhamqagqa到mla
date: 2024-05-13
tags: 优化, 语言模型, 生成模型, attention, 生成模型
status: completed
---

# 缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA

**原文链接**: [https://spaces.ac.cn/archives/10091](https://spaces.ac.cn/archives/10091)

**发布日期**: 

---

前几天，幻方发布的[DeepSeek-V2](https://papers.cool/arxiv/2405.04434)引起了大家的热烈讨论。首先，最让人哗然的是1块钱100万token的价格，普遍比现有的各种竞品API便宜了两个数量级，以至于有人调侃“这个价格哪怕它输出乱码，我也会认为这个乱码是一种艺术”；其次，从模型的技术报告看，如此便宜的价格背后的关键技术之一是它新提出的MLA（**M** ulti-head **L** atent **A** ttention），这是对GQA的改进，据说能比GQA更省更好，也引起了读者的广泛关注。

接下来，本文将跟大家一起梳理一下从MHA、MQA、GQA到MLA的演变历程，并着重介绍一下MLA的设计思路。

## MHA #

MHA（**M** ulti-**H** ead **A** ttention），也就是多头注意力，是开山之作[《Attention is all you need》](/archives/4765)所提出的一种Attention形式，可以说它是当前主流LLM的基础工作。在数学上，多头注意力MHA等价于多个独立的单头注意力的拼接，假设输入的（行）向量序列为$\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_l$，其中$\boldsymbol{x}_i\in\mathbb{R}^d$，那么MHA可以形式地记为  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\\  
\boldsymbol{k}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d\times d_k} \\\  
\boldsymbol{v}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d\times d_v}  
\end{gathered}  
\end{equation}  
简单起见，这里省略了Attention矩阵的缩放因子。实践上，常见的设置是$d_k = d_v = d / h$，对于LLAMA2-7b有$d=4096, h=32, d_k = d_v = 128$，LLAMA2-70b则是$d=8192,h=64, d_k = d_v = 128$

由于这里只考虑了主流的自回归LLM所用的Causal Attention，因此在token by token递归生成时，新预测出来的第$t+1$个token，并不会影响到已经算好的$\boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}$，因此这部分结果我们可以缓存下来供后续生成调用，避免不必要的重复计算，这就是所谓的KV Cache。

而后面的MQA、GQA、MLA，都是围绕“如何减少KV Cache同时尽可能地保证效果”这个主题发展而来的产物。

## 瓶颈 #

一个自然的问题是：为什么降低KV Cache的大小如此重要？

众所周知，一般情况下LLM的推理都是在GPU上进行，单张GPU的显存是有限的，一部分我们要用来存放模型的参数和前向计算的激活值，这部分依赖于模型的体量，选定模型后它就是个常数；另外一部分我们要用来存放模型的KV Cache，这部分不仅依赖于模型的体量，还依赖于模型的输入长度，也就是在推理过程中是动态增长的，当Context长度足够长时，它的大小就会占主导地位，可能超出一张卡甚至一台机（8张卡）的总显存量。

在GPU上部署模型的原则是：能一张卡部署的，就不要跨多张卡；能一台机部署的，就不要跨多台机。这是因为“卡内通信带宽 > 卡间通信带宽 > 机间通信带宽”，由于“木桶效应”，模型部署时跨的设备越多，受设备间通信带宽的的“拖累”就越大，事实上即便是单卡H100内SRAM与HBM的带宽已经达到了3TB/s，但对于Short Context来说这个速度依然还是推理的瓶颈，更不用说更慢的卡间、机间通信了。

所以，减少KV Cache的目的就是要实现在更少的设备上推理更长的Context，或者在相同的Context长度下让推理的batch size更大，从而实现更快的推理速度或者更大的吞吐总量。当然，最终目的都是为了实现更低的推理成本。

要想更详细地了解这个问题，读者可以进一步阅读[《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》](https://papers.cool/arxiv/2205.14135)、[《A guide to LLM inference and performance》](https://www.baseten.co/blog/llm-transformer-inference-guide/)、[《LLM inference speed of light》](https://zeux.io/2024/03/15/llm-inference-sol/)等文章，这里就不继续展开了（主要是笔者水平也有限，唯恐说多错多）。

## MQA #

MQA，即“**M** ulti-**Q** uery **A** ttention”，是减少KV Cache的一次非常朴素的尝试，首次提出自[《Fast Transformer Decoding: One Write-Head is All You Need》](https://papers.cool/arxiv/1911.02150)，这已经是2019年的论文了，这也意味着早在LLM火热之前，减少KV Cache就已经是研究人员非常关注的一个课题了。

MQA的思路很简单，直接让所有Attention Head共享同一个K、V，用公式来说，就是取消MHA所有的$\boldsymbol{k},\boldsymbol{v}$的上标${}^{(s)}$：  
\begin{equation}\require{cancel}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{\color{#ccc}{\smash{\bcancel{(s)}}}} ,\boldsymbol{v}_{\leq t}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\\  
\boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{x}_i\boldsymbol{W}_k^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_k} \\\  
\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{x}_i\boldsymbol{W}_v^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_v}  
\end{gathered}  
\end{equation}  
使用MQA的模型包括[PaLM](https://arxiv.org/pdf/2204.02311)、[StarCoder](https://papers.cool/arxiv/2305.06161)、[Gemini](https://papers.cool/arxiv/2312.11805)等。很明显，MQA直接将KV Cache减少到了原来的$1/h$，这是非常可观的，单从节省显存角度看已经是天花板了。

效果方面，目前看来大部分任务的损失都比较有限，且MQA的支持者相信这部分损失可以通过进一步训练来弥补回。此外，注意到MQA由于共享了K、V，将会导致Attention的参数量减少了将近一半，而为了模型总参数量的不变，通常会相应地增大FFN/GLU的规模，这也能弥补一部分效果损失。

## GQA #

然而，也有人担心MQA对KV Cache的压缩太严重，以至于会影响模型的学习效率以及最终效果。为此，一个MHA与MQA之间的过渡版本GQA（**G** rouped-**Q** uery **A** ttention）应运而生，出自论文[《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》](https://papers.cool/arxiv/2305.13245)，是去年的工作。

事后看来，GQA的思想也很朴素，它就是将所有Head分为$g$个组（$g$可以整除$h$），每组共享同一对K、V，用数学公式表示为  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{\color{red}{(\lceil sg/h\rceil)}} ,\boldsymbol{v}_{\leq t}^{\color{red}{(\lceil sg/h\rceil)}}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{red}{(\lceil sg/h\rceil)}}{}^{\top}\right)\boldsymbol{v}_i^{\color{red}{(\lceil sg/h\rceil)}}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{red}{(\lceil sg/h\rceil)}}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\\  
\boldsymbol{k}_i^{\color{red}{(\lceil sg/h\rceil)}} = \boldsymbol{x}_i\boldsymbol{W}_k^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d\times d_k} \\\  
\boldsymbol{v}_i^{\color{red}{(\lceil sg/h\rceil)}} = \boldsymbol{x}_i\boldsymbol{W}_v^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{\color{red}{(\lceil sg/h\rceil)}}\in\mathbb{R}^{d\times d_v}  
\end{gathered}  
\end{equation}  
这里的$\lceil\cdot\rceil$是上取整符号。GQA提供了MHA到MQA的自然过渡，当$g=h$时就是MHA，$g=1$时就是MQA，当$1 < g < h$时，它只将KV Cache压缩到$g/h$，压缩率不如MQA，但同时也提供了更大的自由度，效果上更有保证。GQA最知名的使用者，大概是Meta开源的[LLAMA2-70B](https://llama.meta.com/llama2/)，以及[LLAMA3](https://llama.meta.com/llama3/)全系列，此外使用GQA的模型还有[TigerBot](https://papers.cool/arxiv/2312.08688)、[DeepSeek-V1](https://papers.cool/arxiv/2401.02954)、[StarCoder2](https://papers.cool/arxiv/2402.19173)、[Yi](https://papers.cool/arxiv/2403.04652)、[ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)、[ChatGLM3](https://github.com/THUDM/ChatGLM3)等，相比使用MQA的模型更多（ChatGLM虽然在它的介绍中说自己是MQA，但实际是$g=2$的GQA）。

在llama2/3-70B中，GQA的$g=8$，其他用了GQA的同体量模型基本上也保持了这个设置，这并非偶然，而是同样出于推理效率的考虑。我们知道，70B这个体量的模型，如果不进行极端的量化，那么不可能部署到单卡（A100/H100 80G）上。单卡不行，那么就能单机了，一般情况下一台机可以装8张卡，刚才我们说了，Attention的每个Head实际上是独立运算然后拼接起来的，当$g=8$时，正好可以每张卡负责计算一组K、V对应的Attention Head，这样可以在尽可能保证K、V多样性的同时最大程度上减少卡间通信。

## MLA #

有了MHA、MQA、GQA的铺垫，我们理解MLA（**M** ulti-head **L** atent **A** ttention）就相对容易一些了。DeepSeek-V2的技术报告里是从低秩投影的角度引入MLA的，以至于有部分读者提出“为什么LoRA提出这么久了，直到MLA才提出对KV Cache低秩分解的做法”之类的疑问。

然而，笔者认为低秩投影这个角度并不贴近本质，因为要说低秩投影的话，事实上只要我们将GQA的所有K、V叠在一起，就会发现GQA也相当于在做低秩投影：  
\begin{equation}\underbrace{\left[\boldsymbol{k}_i^{(1)},\cdots,\boldsymbol{k}_i^{(g)},\boldsymbol{v}_i^{(1)},\cdots,\boldsymbol{v}_i^{(g)}\right]}_{\boldsymbol{c}_i\in\mathbb{R}^{g(d_k+d_v)}} = \boldsymbol{x}_i \underbrace{\left[\boldsymbol{W}_k^{(1)},\cdots,\boldsymbol{W}_k^{(g)},\boldsymbol{W}_v^{(1)},\cdots,\boldsymbol{W}_v^{(g)}\right]}_{\boldsymbol{W}_c\in\mathbb{R}^{d\times g(d_k+d_v)}}\end{equation}  
这里我们将所有$\boldsymbol{k}_i^{(s)},\boldsymbol{v}_i^{(s)}$拼在一起记为$\boldsymbol{c}_i$，相应的投影矩阵也拼在一起记为$\boldsymbol{W}_c$，注意到一般都有$d_c = g(d_k+d_v) < d$，所以$\boldsymbol{x}_i$到$\boldsymbol{c}_i$的变换就是一个低秩投影。所以，MLA的本质改进不是低秩投影，而是低秩投影之后的工作。

### Part 1 #

GQA在投影之后做了什么呢？首先它将向量对半分为两份分别作为K、V，然后每一份又均分为$g$份，每一份复制$h/g$次，以此来“凑”够$h$个Attention Head所需要的K、V。我们知道分割、复制都是简单的线性变换，所以MLA的第一个想法是将这些简单的线性变换换成一般的线性变换，以增强模型的能力：  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\\  
\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_c\times d_k} \\\  
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\\\[10pt]  
\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c}  
\end{gathered}  
\end{equation}  
然而，理论上这样是能增加模型能力，但别忘了GQA的主要目的是减少KV Cache，出于节省计算和通信成本的考虑，我们一般会缓存的是投影后的$\boldsymbol{k}_i, \boldsymbol{v}_i$而不是投影前的$\boldsymbol{c}_i$或$\boldsymbol{x}_i$，而MLA的这个做法，通过不同的投影矩阵再次让所有的K、V Head都变得各不相同，那么KV Cache的大小就恢复成跟MHA一样大了，违背了GQA的初衷。

对此，MLA发现，我们可以结合Dot-Attention的具体形式，通过一个简单但不失巧妙的恒等变换来规避这个问题。首先，在训练阶段还是照常进行，此时优化空间不大；然后，在推理阶段，我们利用  
\begin{equation}\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top} = \left(\boldsymbol{x}_t\boldsymbol{W}_q^{(s)}\right) \left(\boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\right){}^{\top} = \boldsymbol{x}_t\left(\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}\right)\boldsymbol{c}_i^{\top} \end{equation}  
这意味着推理阶段，我们可以将$\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并起来作为Q的投影矩阵，那么$\boldsymbol{c}_i$则取代了原本的$\boldsymbol{k}_i$，同理，在$\boldsymbol{o}_t$后面我们还有一个投影矩阵，于是$\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}$的$\boldsymbol{W}_v^{(s)}$也可以吸收到后面的投影矩阵中去，于是等效地$\boldsymbol{v}_i$也可以用$\boldsymbol{c}_i$代替，也就是说此时KV Cache只需要存下所有的$\boldsymbol{c}_i$就行，而不至于存下所有的$\boldsymbol{k}_i^{(s)}$、$\boldsymbol{v}_i^{(s)}$。注意到$\boldsymbol{c}_i$跟${}^{(s)}$无关，也就是说是所有头共享的，即MLA在推理阶段它可以恒等变换为一个MQA。

再次强调，本文的主题是一直都是减少KV Cache，那到目前为止，MLA做到了什么呢？答案是通过不同的投影矩阵来增强了GQA的能力，并且推理时可以保持同样大小的KV Cache。那么反过来，如果我们只需要跟GQA相近的能力，那么是不是就可以再次减少KV Cache了？换言之，$d_c$没必要取$g(d_k+d_v)$，而是取更小的值（DeepSeek-V2取了512），从而进一步压缩KV Cache，这就是MLA的核心思想。

> **补充说明：**
> 
> 1、$\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并成一个矩阵的恒等变换，理论上只有在无限精度下才成立，实际上如果我们使用单精度尤其是BF16的话，经过变换后的精度损失往往还是挺明显的，经过多层累积后可能放大到比较可观的程度；
> 
> 2、实际上我们一般不按照$\boldsymbol{x}_t\left(\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}\right)$来计算Q，而是按照$\left(\boldsymbol{x}_t\boldsymbol{W}_q^{(s)}\right)\boldsymbol{W}_k^{(s)}{}^{\top}$来计算，这样虽然是串行的，但在低秩假设下计算量更少，并且理论精度的损失也更少，不过在文章中，我们仍按照$\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并成一个矩阵来介绍。

### Part 2 #

一切似乎都很完美，看上去一个又好又省的理想设计就要出炉了。不过别急，当我们再深入思考一下就会发现，到目前为止的MLA有一个难以绕开的缺陷——不兼容[RoPE（旋转位置编码）](/archives/8265)。

刚才我们说了，MLA之所以能保持跟GQA一样大小的KV Cache，其关键一步是“将$\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$合并成一个（跟位置无关的）矩阵作为Q的投影矩阵”，但如果加了RoPE的话，这一步就无法实现了。这是因为RoPE是一个跟位置相关的、$d_k\times d_k$的分块对角矩阵$\boldsymbol{\mathcal{R}}_m$，满足$\boldsymbol{\mathcal{R}}_m\boldsymbol{\mathcal{R}}_n^{\top}=\boldsymbol{\mathcal{R}}_{m-n}$，MLA加入RoPE之后会让$\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}$之间多插入了一项$\boldsymbol{\mathcal{R}}_{t-i}$：  
\begin{equation}  
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\quad,\quad\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i} \\\  
\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top} = \left(\boldsymbol{x}_t\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_t}\right) \left(\boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right){}^{\top} = \boldsymbol{x}_t\left(\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_{t-i}}\boldsymbol{W}_k^{(s)}{}^{\top}\right)\boldsymbol{c}_i^{\top} \end{equation}  
这里的$\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_{t-i}}\boldsymbol{W}_k^{(s)}{}^{\top}$就无法合并为一个固定的投影矩阵了（跟位置差$t-i$相关），从而MLA的想法无法结合RoPE实现。

前段时间，笔者也很荣幸跟DeepSeek团队讨论过这个问题，但这个问题可以说非常本质，所以当时笔者实际上也没能提出什么有效的建议。最简单的方式是放弃RoPE，换用其他基于Attention Bias的位置编码，如[ALIBI](/archives/9431#ALIBI)，但DeepSeek的实验显示它明显不如RoPE（注意，MLA不是不能加RoPE，而是加了RoPE之后无法用恒等变换技巧来减少KV Cache），笔者也提议过换[Sandwich](/archives/9431#Sandwich)，它不像ALIBI单调衰减到负无穷，估计效果会好些，但感觉是治标不治本。还有一个折中的办法是将$\boldsymbol{q}_i$的输入也改为$\boldsymbol{c}_i$，然后RoPE加在$\boldsymbol{c}_i$之后，即  
\begin{equation}\boldsymbol{q}_i^{(s)} = \boldsymbol{c}_i\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\boldsymbol{W}_q^{(s)},\quad\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\boldsymbol{W}_k^{(s)}\end{equation}  
这样$\boldsymbol{\mathcal{R}}_i$就可以吸收到$\boldsymbol{c}_i$中去，但这样就没有$\boldsymbol{\mathcal{R}}_m\boldsymbol{\mathcal{R}}_n^{\top}=\boldsymbol{\mathcal{R}}_{m-n}$的运算了，此时的RoPE不再是通过绝对位置实现相对位置，而单纯是在Q、K上加绝对位置，让模型自己想办法提炼相对位置信息。

最后发布的MLA，采取了一种混合的方法——每个Attention Head的Q、K新增$d_r$个维度用来添加RoPE，其中K新增的维度每个Head共享：  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r},\quad \boldsymbol{W}_{qc}^{(s)}\in\mathbb{R}^{d\times d_k},\boldsymbol{W}_{qr}^{(s)}\in\mathbb{R}^{d\times d_r}\\\  
\boldsymbol{k}_i^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k+d_r},\quad \boldsymbol{W}_{kc}^{(s)}\in\mathbb{R}^{d_c\times d_k}, \boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_r} \\\  
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\\\[10pt]  
\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c}  
\end{gathered}  
\end{equation}  
这样一来，没有RoPE的维度就可以重复“Part 1”的操作，在推理时KV Cache只需要存$\boldsymbol{c}_i$，新增的带RoPE的维度就可以用来补充位置信息，并且由于所有Head共享，所以也就只有在K Cache这里增加了$d_r$个维度，原论文取了$d_r = d_k / 2 = 64$，相比原本的$d_c=512$，增加的幅度不大。

### Part 3 #

最后有一个细节，就是MLA的最终版本，还将Q的输入也改为了低秩投影形式，这与减少KV Cache无关，主要是为了减少训练期间参数量和相应的梯度（原论文说的是激活值，个人表示不大理解）所占的显存：  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{c}_i'\boldsymbol{W}_{qc}^{(s)}, \boldsymbol{c}_i'\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r},\quad \boldsymbol{W}_{qc}^{(s)}\in\mathbb{R}^{d_c'\times d_k},\boldsymbol{W}_{qr}^{(s)}\in\mathbb{R}^{d_c'\times d_r}\\\  
\boldsymbol{k}_i^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k+d_r},\quad \boldsymbol{W}_{kc}^{(s)}\in\mathbb{R}^{d_c\times d_k}, \boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_r} \\\  
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\\\[10pt]  
\boldsymbol{c}_i' = \boldsymbol{x}_i \boldsymbol{W}_c'\in\mathbb{R}^{d_c'},\quad \boldsymbol{W}_c'\in\mathbb{R}^{d\times d_c'} \\\  
\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c} \\\  
\end{gathered}  
\label{eq:mla-mha}\end{equation}  
注意$\boldsymbol{k}_i^{(s)}$中的第二项，带RoPE的部分，其输入还是$\boldsymbol{x}_i$而不是$\boldsymbol{c}_i$，这里保持了原论文的设置，不是笔误，$d_c'$原论文的取值是1536，跟$d_c=512$不同。同时，我们把带RoPE的MHA放在下面，方便大家对比：  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{(s)} ,\boldsymbol{v}_{\leq t}^{(s)}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\\  
\boldsymbol{k}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_k^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d\times d_k} \\\  
\boldsymbol{v}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d\times d_v}  
\end{gathered}  
\end{equation}  
可以发现，其实在训练阶段，除了多了一步低秩投影以及只在部分维度加RoPE外，MLA与Q、K的Head Size由$d_k$换成$d_k + d_r$的MHA基本无异。

解码阶段的MLA则改为MQA形式  
\begin{equation}  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}\boldsymbol{W}_v^{(1)}, \boldsymbol{o}_t^{(2)}\boldsymbol{W}_v^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\boldsymbol{W}_v^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = Attention\left(\boldsymbol{q}_t^{(s)}, \boldsymbol{k}_{\leq t}^{\color{#ccc}{\smash{\bcancel{(s)}}}} ,\boldsymbol{c}_{\leq t}\right)\triangleq\frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{c}_i}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{c}_i'\boldsymbol{W}_{qc}^{(s)}\boldsymbol{W}_{kc}^{(s)}{}^{\top}, \boldsymbol{c}_i'\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c + d_r}\\\  
\boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \left[\boldsymbol{c}_i, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c+d_r}\\\  
\boldsymbol{W}_{qc}^{(s)}\in\mathbb{R}^{d_c'\times d_k},\boldsymbol{W}_{kc}^{(s)}\in\mathbb{R}^{d_c\times d_k},\boldsymbol{W}_{qr}^{(s)}\in\mathbb{R}^{d_c'\times d_r},\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\in\mathbb{R}^{d\times d_r} \\\\[10pt]  
\boldsymbol{c}_i' = \boldsymbol{x}_i \boldsymbol{W}_c'\in\mathbb{R}^{d_c'},\quad \boldsymbol{W}_c'\in\mathbb{R}^{d\times d_c'} \\\  
\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c} \\\  
\end{gathered}  
\label{eq:mla-mqa}\end{equation}  
此时Q、K的Head Size变成了$d_c + d_r$，V的Head Size 则变成了$d_c$，按照原论文的设置，这是$d_k$、$d_v$的4倍。所以实际上MLA在解码阶段做的这个转换，虽然能有效减少KV Cache，但其解码的计算量是增加的。

那为什么还能提高推理效率呢？这又回到“瓶颈”一节所讨论的问题了，我们可以将LLM的推理分两部分：第一个Token的生成（Prefill）和后续每个Token的生成（Generation），Prefill阶段涉及到对输入所有Token的并行计算，然后把对应的KV Cache存下来，这部分对于计算、带宽和显存都是瓶颈，我们可以用MLA的MHA形式$\eqref{eq:mla-mha}$来算；但是Generation阶段由于每步只计算一个Token，实际上它更多的是带宽瓶颈和显存瓶颈，此时我们可以用MLA的MQA形式$\eqref{eq:mla-mqa}$来算，从而明显提高Generation的速度。

还有一个细节充分体现了这个特性。一般的LLM架构参数满足$h \times d_k = d$，即num_heads * head_size = hidden_size，但DeepSeek-V2不一样，它$d_k=128,d=5120$，但$h=128$，是一般设置的3倍！这是因为MLA的KV Cache大小跟$h$无关，增大$h$只会增加计算量和提升模型能力，但不会增加KV Cache，所以不会带来速度瓶颈。

## 小结 #

本文简单概述了多头注意力的演变历程，特别是从MHA向MQA、GQA，最终到MLA的变化理念，最后详细展开了对MLA的介绍。在本文中，MLA被视为GQA的一般化，它用投影矩阵的方式替代了GQA的分割、重复，并引入了一个恒等变换技巧来可以进一步压缩KV Cache，同时采用了一种混合方法来兼容RoPE。总的来说，MLA称得上是一种非常实用的注意力变体。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10091>_

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

苏剑林. (May. 13, 2024). 《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10091>

@online{kexuefm-10091,  
title={缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA},  
author={苏剑林},  
year={2024},  
month={May},  
url={\url{https://spaces.ac.cn/archives/10091}},  
} 


---

## 公式推导与注释

### 一、多头注意力(MHA)的完整数学推导

#### 1.1 标准MHA的数学定义

对于输入序列 $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_l] \in \mathbb{R}^{l \times d}$，多头注意力机制可以形式化为：

\begin{equation}
\text{MHA}(\boldsymbol{X}) = \text{Concat}(\boldsymbol{O}^{(1)}, \boldsymbol{O}^{(2)}, \ldots, \boldsymbol{O}^{(h)}) \boldsymbol{W}_O
\tag{1}
\end{equation}

其中 $\boldsymbol{O}^{(s)} \in \mathbb{R}^{l \times d_v}$ 是第 $s$ 个头的输出，$\boldsymbol{W}_O \in \mathbb{R}^{hd_v \times d}$ 是输出投影矩阵。

**数学直觉**: MHA通过多个独立的注意力头并行处理信息，每个头关注不同的表示子空间，最后拼接融合。这类似于卷积神经网络中的多通道机制。

#### 1.2 单头注意力的详细推导

对于第 $s$ 个注意力头，计算过程为：

\begin{equation}
\boldsymbol{O}^{(s)} = \text{Attention}(\boldsymbol{Q}^{(s)}, \boldsymbol{K}^{(s)}, \boldsymbol{V}^{(s)})
\tag{2}
\end{equation}

展开注意力计算：

\begin{equation}
\text{Attention}(\boldsymbol{Q}^{(s)}, \boldsymbol{K}^{(s)}, \boldsymbol{V}^{(s)}) = \text{softmax}\left(\frac{\boldsymbol{Q}^{(s)} {\boldsymbol{K}^{(s)}}^\top}{\sqrt{d_k}}\right) \boldsymbol{V}^{(s)}
\tag{3}
\end{equation}

其中：
- $\boldsymbol{Q}^{(s)} = \boldsymbol{X}\boldsymbol{W}_q^{(s)} \in \mathbb{R}^{l \times d_k}$ (查询矩阵)
- $\boldsymbol{K}^{(s)} = \boldsymbol{X}\boldsymbol{W}_k^{(s)} \in \mathbb{R}^{l \times d_k}$ (键矩阵)
- $\boldsymbol{V}^{(s)} = \boldsymbol{X}\boldsymbol{W}_v^{(s)} \in \mathbb{R}^{l \times d_v}$ (值矩阵)

**缩放因子的数学原理**: $1/\sqrt{d_k}$ 的缩放确保了点积的方差为1。假设 $\boldsymbol{q}$ 和 $\boldsymbol{k}$ 的每个元素独立同分布，均值为0，方差为1，则点积 $\boldsymbol{q}^\top \boldsymbol{k} = \sum_{i=1}^{d_k} q_i k_i$ 的方差为：

\begin{equation}
\text{Var}(\boldsymbol{q}^\top \boldsymbol{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
\tag{4}
\end{equation}

因此除以 $\sqrt{d_k}$ 可以将方差归一化为1，防止softmax饱和。

#### 1.3 自回归生成中的因果注意力

对于自回归语言模型，第 $t$ 个token只能关注到位置 $\leq t$ 的token：

\begin{equation}
\boldsymbol{o}_t^{(s)} = \sum_{i=1}^{t} \alpha_{t,i}^{(s)} \boldsymbol{v}_i^{(s)}
\tag{5}
\end{equation}

其中注意力权重为：

\begin{equation}
\alpha_{t,i}^{(s)} = \frac{\exp\left(\boldsymbol{q}_t^{(s)} {\boldsymbol{k}_i^{(s)}}^\top / \sqrt{d_k}\right)}{\sum_{j=1}^{t} \exp\left(\boldsymbol{q}_t^{(s)} {\boldsymbol{k}_j^{(s)}}^\top / \sqrt{d_k}\right)}
\tag{6}
\end{equation}

**数学性质**: 注意力权重满足 $\sum_{i=1}^{t} \alpha_{t,i}^{(s)} = 1$ 且 $\alpha_{t,i}^{(s)} \geq 0$，因此这是一个凸组合。

#### 1.4 MHA的计算复杂度分析

**时间复杂度**:
1. QKV投影: $\mathcal{O}(3hld \cdot d_k) = \mathcal{O}(hld^2)$ (因为通常 $hd_k = d$)
2. 注意力矩阵计算: $\mathcal{O}(hl^2d_k)$
3. 注意力加权求和: $\mathcal{O}(hl^2d_v)$
4. 输出投影: $\mathcal{O}(ld \cdot hd_v) = \mathcal{O}(ld^2)$

总时间复杂度：

\begin{equation}
T_{\text{MHA}} = \mathcal{O}(ld^2 + hl^2d) = \mathcal{O}(l^2d)
\tag{7}
\end{equation}

(当 $l \gg d$ 时，$l^2d$ 项占主导)

**空间复杂度**:
1. QKV矩阵: $\mathcal{O}(3hld_k) = \mathcal{O}(ld)$
2. 注意力矩阵: $\mathcal{O}(hl^2)$
3. 输出矩阵: $\mathcal{O}(ld)$

总空间复杂度：

\begin{equation}
S_{\text{MHA}} = \mathcal{O}(ld + hl^2)
\tag{8}
\end{equation}

### 二、KV Cache的数学原理与内存分析

#### 2.1 KV Cache的必要性推导

在自回归生成中，生成第 $t+1$ 个token时，需要计算：

\begin{equation}
\boldsymbol{o}_{t+1}^{(s)} = \sum_{i=1}^{t+1} \alpha_{t+1,i}^{(s)} \boldsymbol{v}_i^{(s)}
\tag{9}
\end{equation}

注意到对于 $i \leq t$，$\boldsymbol{k}_i^{(s)}$ 和 $\boldsymbol{v}_i^{(s)}$ 在计算 $\boldsymbol{o}_t^{(s)}$ 时已经算过，无需重新计算。

**重复计算量分析**: 如果不使用KV Cache，生成长度为 $L$ 的序列，总计算量为：

\begin{equation}
\text{Total FLOPs} = \sum_{t=1}^{L} \mathcal{O}(t \cdot hd_k) = \mathcal{O}(L^2hd_k)
\tag{10}
\end{equation}

使用KV Cache后，每步只需计算新token的KV：

\begin{equation}
\text{Total FLOPs (cached)} = \sum_{t=1}^{L} \mathcal{O}(hd_k) = \mathcal{O}(Lhd_k)
\tag{11}
\end{equation}

**加速比**:

\begin{equation}
\text{Speedup} = \frac{\mathcal{O}(L^2hd_k)}{\mathcal{O}(Lhd_k)} = \mathcal{O}(L)
\tag{12}
\end{equation}

#### 2.2 MHA的KV Cache内存占用

对于MHA，每层需要缓存 $h$ 个头的K和V矩阵。对于长度为 $L$ 的序列：

单层KV Cache大小：

\begin{equation}
\text{Cache}_{\text{MHA}} = 2 \times h \times L \times d_k \times \text{sizeof(dtype)}
\tag{13}
\end{equation}

对于LLAMA2-7B ($d=4096, h=32, d_k=128$)，使用FP16 (2字节)：

\begin{equation}
\text{Cache}_{\text{MHA}}^{\text{per layer}} = 2 \times 32 \times L \times 128 \times 2 = 16384L \text{ bytes} = 16L \text{ KB}
\tag{14}
\end{equation}

LLAMA2-7B有32层，总KV Cache：

\begin{equation}
\text{Total Cache} = 32 \times 16L = 512L \text{ KB} = 0.5L \text{ MB}
\tag{15}
\end{equation}

**示例**: 对于 $L=4096$ 的上下文：
- 单层: $16 \times 4096 = 65536$ KB = 64 MB
- 32层总计: $32 \times 64 = 2048$ MB = 2 GB

对于 $L=32768$ (32K上下文):
- 总KV Cache: $0.5 \times 32768 = 16384$ MB = 16 GB

#### 2.3 KV Cache的瓶颈分析

**内存带宽瓶颈**: 假设GPU有带宽 $B$ (例如A100的HBM带宽为1.5 TB/s)，每次生成需要读取的KV Cache大小为 $S_{\text{cache}}$，则生成一个token的最小时间为：

\begin{equation}
t_{\text{min}} = \frac{S_{\text{cache}}}{B}
\tag{16}
\end{equation}

对于LLAMA2-7B，$L=4096$时 $S_{\text{cache}} = 2$ GB：

\begin{equation}
t_{\text{min}} = \frac{2 \text{ GB}}{1.5 \text{ TB/s}} = \frac{2}{1500} \text{ s} \approx 1.33 \text{ ms}
\tag{17}
\end{equation}

**吞吐量限制**: 假设batch size为 $B_s$，每个请求的平均序列长度为 $\bar{L}$，GPU显存为 $M$，则最大batch size受限于：

\begin{equation}
B_s \leq \frac{M - M_{\text{model}}}{N_{\text{layer}} \times \text{Cache}_{\text{MHA}}(\bar{L})}
\tag{18}
\end{equation}

其中 $M_{\text{model}}$ 是模型参数占用的显存。

**数值示例**: A100 80GB，LLAMA2-7B (FP16约14GB)，$\bar{L}=2048$：

\begin{equation}
B_s \leq \frac{80 - 14}{32 \times 0.5 \times 2048 / 1024} = \frac{66}{32} \approx 2
\tag{19}
\end{equation}

### 三、MQA (Multi-Query Attention) 的数学推导

#### 3.1 MQA的动机与数学定义

MQA的核心思想是让所有头共享同一组K和V：

\begin{equation}
\boldsymbol{o}_t^{(s)} = \sum_{i=1}^{t} \frac{\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^\top / \sqrt{d_k}\right)}{\sum_{j=1}^{t} \exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_j^\top / \sqrt{d_k}\right)} \boldsymbol{v}_i
\tag{20}
\end{equation}

注意这里 $\boldsymbol{k}_i$ 和 $\boldsymbol{v}_i$ 没有上标 $(s)$，即所有头共享。

**参数对比**:
- MHA: $h$ 个 $\boldsymbol{W}_q^{(s)}$，$h$ 个 $\boldsymbol{W}_k^{(s)}$，$h$ 个 $\boldsymbol{W}_v^{(s)}$
- MQA: $h$ 个 $\boldsymbol{W}_q^{(s)}$，1 个 $\boldsymbol{W}_k$，1 个 $\boldsymbol{W}_v$

#### 3.2 MQA的KV Cache压缩比

单层KV Cache：

\begin{equation}
\text{Cache}_{\text{MQA}} = 2 \times 1 \times L \times d_k \times \text{sizeof(dtype)}
\tag{21}
\end{equation}

压缩比：

\begin{equation}
\text{Compression Ratio} = \frac{\text{Cache}_{\text{MHA}}}{\text{Cache}_{\text{MQA}}} = \frac{2hLd_k}{2Ld_k} = h
\tag{22}
\end{equation}

**数值示例**: 对于LLAMA2-7B ($h=32$)，KV Cache减少到原来的 $1/32$：
- $L=4096$: MHA 2GB → MQA 64MB
- $L=32768$: MHA 16GB → MQA 512MB

#### 3.3 MQA的表达能力分析

**秩的角度**: MHA的输出可以写成：

\begin{equation}
\boldsymbol{O}_{\text{MHA}} = \sum_{s=1}^{h} \boldsymbol{A}^{(s)} \boldsymbol{V}^{(s)}
\tag{23}
\end{equation}

其中 $\boldsymbol{A}^{(s)} \in \mathbb{R}^{l \times l}$ 是注意力矩阵。理论秩最高为 $\min(hd_v, l)$。

MQA的输出：

\begin{equation}
\boldsymbol{O}_{\text{MQA}} = \left(\sum_{s=1}^{h} \boldsymbol{A}^{(s)}\right) \boldsymbol{V}
\tag{24}
\end{equation}

理论秩最高为 $\min(d_v, l)$，降低了 $h$ 倍。

**容量损失**: MQA相当于强制了一个低秩约束，可能降低模型的表达能力。实验表明对于大多数任务损失有限。

#### 3.4 MQA的参数量变化

MHA参数量：

\begin{equation}
P_{\text{MHA}} = h(d \times d_k + d \times d_k + d \times d_v) + hd_v \times d
\tag{25}
\end{equation}

当 $d_k = d_v = d/h$ 时：

\begin{equation}
P_{\text{MHA}} = 3d^2 + d^2 = 4d^2
\tag{26}
\end{equation}

MQA参数量：

\begin{equation}
P_{\text{MQA}} = hd \times d_k + d \times d_k + d \times d_v + hd_v \times d
\tag{27}
\end{equation}

\begin{equation}
P_{\text{MQA}} = d^2 + 2d \times \frac{d}{h} + d^2 = 2d^2 + \frac{2d^2}{h}
\tag{28}
\end{equation}

参数减少量：

\begin{equation}
\Delta P = 4d^2 - (2d^2 + \frac{2d^2}{h}) = 2d^2(1 - \frac{1}{h}) \approx 2d^2
\tag{29}
\end{equation}

**补偿策略**: 为保持总参数量不变，通常增大FFN的隐藏层维度。

### 四、GQA (Grouped-Query Attention) 的数学推导

#### 4.1 GQA的分组机制

GQA将 $h$ 个头分成 $g$ 组，每组共享一组K和V。设每组有 $h/g$ 个头：

\begin{equation}
\boldsymbol{o}_t^{(s)} = \sum_{i=1}^{t} \alpha_{t,i}^{(s)} \boldsymbol{v}_i^{(\lceil sg/h \rceil)}, \quad s = 1, 2, \ldots, h
\tag{30}
\end{equation}

其中 $\lceil sg/h \rceil$ 是第 $s$ 个头所属的组索引。

**示例**: $h=8, g=2$:
- 组1: 头1,2,3,4 共享 $\boldsymbol{k}^{(1)}, \boldsymbol{v}^{(1)}$
- 组2: 头5,6,7,8 共享 $\boldsymbol{k}^{(2)}, \boldsymbol{v}^{(2)}$

#### 4.2 GQA的KV Cache分析

单层KV Cache：

\begin{equation}
\text{Cache}_{\text{GQA}} = 2 \times g \times L \times d_k \times \text{sizeof(dtype)}
\tag{31}
\end{equation}

压缩比：

\begin{equation}
\text{Compression Ratio} = \frac{\text{Cache}_{\text{MHA}}}{\text{Cache}_{\text{GQA}}} = \frac{h}{g}
\tag{32}
\end{equation}

**参数化**:
- $g=h$: GQA退化为MHA (无压缩)
- $g=1$: GQA退化为MQA (最大压缩)
- $1 < g < h$: 中间状态，平衡效果与效率

#### 4.3 LLAMA2-70B的GQA配置分析

LLAMA2-70B使用 $g=8$ 的GQA配置：
- $h=64$ (64个头)
- $g=8$ (8组)
- 每组 $64/8 = 8$ 个头

**内存节省**:

\begin{equation}
\text{Saving} = 1 - \frac{g}{h} = 1 - \frac{8}{64} = 87.5\%
\tag{33}
\end{equation}

**分布式推理优势**: 8组KV正好对应8张GPU卡，每张卡负责一组：

\begin{equation}
\text{Per-GPU Cache} = \frac{\text{Cache}_{\text{GQA}}}{8} = \frac{2gLd_k}{8} = \frac{Ld_k}{4} \times \text{sizeof(dtype)}
\tag{34}
\end{equation}

这避免了跨卡通信KV Cache的开销。

#### 4.4 GQA的低秩分解视角

将GQA的所有K和V拼接：

\begin{equation}
\boldsymbol{C} = [\boldsymbol{k}^{(1)}, \ldots, \boldsymbol{k}^{(g)}, \boldsymbol{v}^{(1)}, \ldots, \boldsymbol{v}^{(g)}] \in \mathbb{R}^{l \times g(d_k + d_v)}
\tag{35}
\end{equation}

可以写成低秩投影：

\begin{equation}
\boldsymbol{C} = \boldsymbol{X} \boldsymbol{W}_c
\tag{36}
\end{equation}

其中 $\boldsymbol{W}_c \in \mathbb{R}^{d \times g(d_k + d_v)}$。

**秩约束**: 由于 $g(d_k + d_v) < h(d_k + d_v) = d$ (通常情况)，这是一个秩为 $g(d_k + d_v)$ 的低秩矩阵。

### 五、MLA (Multi-head Latent Attention) 的深入推导

#### 5.1 MLA的核心思想：低秩投影

MLA的关键洞察是：GQA的分割和复制操作可以用更一般的线性变换替代。

**GQA的操作**:
1. 投影到低维: $\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c \in \mathbb{R}^{d_c}$
2. 分割: 将 $\boldsymbol{c}_i$ 分成 $g$ 份
3. 复制: 每份复制 $h/g$ 次
4. 线性变换得到K和V

**MLA的改进**: 用可学习的线性变换替代固定的分割和复制：

\begin{equation}
\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i \boldsymbol{W}_k^{(s)}, \quad \boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i \boldsymbol{W}_v^{(s)}
\tag{37}
\end{equation}

其中 $\boldsymbol{W}_k^{(s)} \in \mathbb{R}^{d_c \times d_k}$, $\boldsymbol{W}_v^{(s)} \in \mathbb{R}^{d_c \times d_v}$。

#### 5.2 MLA的恒等变换技巧

**训练阶段**: 正常计算注意力

\begin{equation}
\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)\top} = (\boldsymbol{x}_t \boldsymbol{W}_q^{(s)}) (\boldsymbol{c}_i \boldsymbol{W}_k^{(s)})^\top
\tag{38}
\end{equation}

**推理阶段的恒等变换**:

\begin{equation}
\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)\top} = \boldsymbol{x}_t (\boldsymbol{W}_q^{(s)} \boldsymbol{W}_k^{(s)\top}) \boldsymbol{c}_i^\top
\tag{39}
\end{equation}

定义合并矩阵：

\begin{equation}
\boldsymbol{W}_{qk}^{(s)} = \boldsymbol{W}_q^{(s)} \boldsymbol{W}_k^{(s)\top} \in \mathbb{R}^{d \times d_c}
\tag{40}
\end{equation}

则：

\begin{equation}
\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)\top} = (\boldsymbol{x}_t \boldsymbol{W}_{qk}^{(s)}) \boldsymbol{c}_i^\top
\tag{41}
\end{equation}

**关键洞察**: KV Cache只需要存 $\boldsymbol{c}_i \in \mathbb{R}^{d_c}$，而不是 $h$ 个 $\boldsymbol{k}_i^{(s)} \in \mathbb{R}^{d_k}$ 和 $\boldsymbol{v}_i^{(s)} \in \mathbb{R}^{d_v}$！

#### 5.3 MLA的KV Cache大小

单层KV Cache (不含RoPE部分):

\begin{equation}
\text{Cache}_{\text{MLA}}^{\text{base}} = L \times d_c \times \text{sizeof(dtype)}
\tag{42}
\end{equation}

DeepSeek-V2设置 $d_c = 512$，相比MHA的 $h \times d_k = 32 \times 128 = 4096$，压缩比为：

\begin{equation}
\text{Compression} = \frac{4096}{512} = 8\times
\tag{43}
\end{equation}

#### 5.4 MLA与RoPE的兼容性问题

**矛盾**: RoPE需要在K和Q上应用位置相关的旋转矩阵：

\begin{equation}
\boldsymbol{q}_t^{(s)} = \boldsymbol{x}_t \boldsymbol{W}_q^{(s)} \boldsymbol{\mathcal{R}}_t, \quad \boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i \boldsymbol{W}_k^{(s)} \boldsymbol{\mathcal{R}}_i
\tag{44}
\end{equation}

此时：

\begin{equation}
\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)\top} = \boldsymbol{x}_t \boldsymbol{W}_q^{(s)} \boldsymbol{\mathcal{R}}_t \boldsymbol{\mathcal{R}}_i^\top \boldsymbol{W}_k^{(s)\top} \boldsymbol{c}_i^\top = \boldsymbol{x}_t \boldsymbol{W}_q^{(s)} \boldsymbol{\mathcal{R}}_{t-i} \boldsymbol{W}_k^{(s)\top} \boldsymbol{c}_i^\top
\tag{45}
\end{equation}

由于 $\boldsymbol{\mathcal{R}}_{t-i}$ 依赖于位置差，无法预先合并到投影矩阵中！

#### 5.5 MLA的混合解决方案

**方案**: 将Q和K分为两部分：

1. **低秩部分** (不加RoPE): 维度 $d_k$
\begin{equation}
\boldsymbol{q}_{t,c}^{(s)} = \boldsymbol{x}_t \boldsymbol{W}_{qc}^{(s)}, \quad \boldsymbol{k}_{i,c}^{(s)} = \boldsymbol{c}_i \boldsymbol{W}_{kc}^{(s)}
\tag{46}
\end{equation}

2. **RoPE部分** (共享): 维度 $d_r$
\begin{equation}
\boldsymbol{q}_{t,r}^{(s)} = \boldsymbol{x}_t \boldsymbol{W}_{qr}^{(s)} \boldsymbol{\mathcal{R}}_t, \quad \boldsymbol{k}_{i,r} = \boldsymbol{x}_i \boldsymbol{W}_{kr} \boldsymbol{\mathcal{R}}_i
\tag{47}
\end{equation}

注意 $\boldsymbol{k}_{i,r}$ 所有头共享 (无上标$(s)$)。

**最终的Q和K**:

\begin{equation}
\boldsymbol{q}_t^{(s)} = [\boldsymbol{q}_{t,c}^{(s)}, \boldsymbol{q}_{t,r}^{(s)}] \in \mathbb{R}^{d_k + d_r}
\tag{48}
\end{equation}

\begin{equation}
\boldsymbol{k}_i^{(s)} = [\boldsymbol{k}_{i,c}^{(s)}, \boldsymbol{k}_{i,r}] \in \mathbb{R}^{d_k + d_r}
\tag{49}
\end{equation}

#### 5.6 MLA的最终KV Cache

需要缓存：
1. $\boldsymbol{c}_i \in \mathbb{R}^{d_c}$ (低秩部分)
2. $\boldsymbol{k}_{i,r} \in \mathbb{R}^{d_r}$ (RoPE部分，所有头共享)

单层总KV Cache：

\begin{equation}
\text{Cache}_{\text{MLA}} = L \times (d_c + d_r) \times \text{sizeof(dtype)}
\tag{50}
\end{equation}

DeepSeek-V2: $d_c = 512, d_r = 64$:

\begin{equation}
\text{Cache}_{\text{MLA}} = L \times 576 \times 2 = 1152L \text{ bytes}
\tag{51}
\end{equation}

相比MHA的 $2 \times 32 \times 128 \times L \times 2 = 16384L$ 字节:

\begin{equation}
\text{Compression} = \frac{16384}{1152} \approx 14.2\times
\tag{52}
\end{equation}

#### 5.7 MLA的低秩投影Q

为减少训练时的激活内存，MLA也对Q使用低秩投影：

\begin{equation}
\boldsymbol{c}_i' = \boldsymbol{x}_i \boldsymbol{W}_c' \in \mathbb{R}^{d_c'}
\tag{53}
\end{equation}

\begin{equation}
\boldsymbol{q}_i^{(s)} = [\boldsymbol{c}_i' \boldsymbol{W}_{qc}^{(s)}, \boldsymbol{c}_i' \boldsymbol{W}_{qr}^{(s)} \boldsymbol{\mathcal{R}}_i]
\tag{54}
\end{equation}

DeepSeek-V2: $d_c' = 1536$，相比 $d = 5120$ 节省约 70% 的Q投影参数。

#### 5.8 MLA推理时的计算量分析

**Prefill阶段** (训练模式):
- QKV投影: $\mathcal{O}(Ld \cdot d_c') + \mathcal{O}(Lhd_c' \cdot d_k) + \mathcal{O}(Lhd_c \cdot d_v)$
- 注意力计算: $\mathcal{O}(L^2h(d_k + d_r))$

**Generation阶段** (推理模式):
- Q计算: $\mathcal{O}(h \cdot d_c \cdot d_k)$ (通过 $\boldsymbol{W}_{qk}^{(s)} = \boldsymbol{W}_{qc}^{(s)} \boldsymbol{W}_{kc}^{(s)\top}$)
- 注意力计算: $\mathcal{O}(Lh(d_c + d_r))$

**相比MHA的变化**:
- 计算量增加: head size从 $d_k$ 变为 $d_c + d_r = 512 + 64 = 576$ (4.5倍)
- 但内存带宽减少: KV Cache减少14倍

由于Generation是内存带宽瓶颈，MLA仍然获得显著加速。

### 六、性能-效率权衡的理论分析

#### 6.1 注意力机制的表达能力

**定理**: 对于序列长度为 $l$ 的输入，MHA的输出空间维度最多为 $\min(l, hd_v)$。

**证明**: MHA的输出为：

\begin{equation}
\boldsymbol{O} = \text{Concat}\left(\sum_{i=1}^{l} \alpha_i^{(1)} \boldsymbol{v}_i^{(1)}, \ldots, \sum_{i=1}^{l} \alpha_i^{(h)} \boldsymbol{v}_i^{(h)}\right)
\tag{55}
\end{equation}

每个头的输出是 $l$ 个向量的凸组合，因此最多有 $l$ 个线性独立的输出。总共 $h$ 个头，每个头维度 $d_v$，因此最多 $\min(l, hd_v)$ 维。□

**推论**: MQA的输出空间维度最多为 $\min(l, d_v)$，降低了 $h$ 倍。

#### 6.2 KV Cache压缩与容量的trade-off

定义**有效容量** $C$ 为模型能够建模的独立模式数量：

\begin{equation}
C_{\text{MHA}} \propto hd_v, \quad C_{\text{MQA}} \propto d_v, \quad C_{\text{GQA}} \propto gd_v
\tag{56}
\end{equation}

定义**效率** $E$ 为单位内存的吞吐量：

\begin{equation}
E \propto \frac{1}{\text{Cache Size}}
\tag{57}
\end{equation}

**Pareto前沿**:

\begin{equation}
C \times E = \text{constant}
\tag{58}
\end{equation}

GQA通过参数 $g$ 在Pareto前沿上滑动，选择最优的容量-效率平衡点。

#### 6.3 数值示例：不同方案的对比

假设 $d=4096, h=32, d_k=d_v=128, L=4096$:

| 方案 | KV Cache (MB) | 有效容量 | 推理吞吐 (相对) |
|------|---------------|----------|-----------------|
| MHA  | 2048          | $32 \times 128 = 4096$ | 1.0× |
| GQA-8 | 256          | $8 \times 128 = 1024$ | 8.0× |
| MQA  | 64            | $128$ | 32× |
| MLA  | 144           | $\approx 1536$ (有效) | 14.2× |

**结论**: MLA在几乎不损失容量的情况下，获得了接近MQA的效率提升。

### 七、实践建议与数值验证

#### 7.1 选择合适的注意力机制

**决策树**:

1. **内存充足** ($L < 2048$): 使用MHA，最大化模型容量
2. **中等序列** ($2048 \leq L \leq 8192$): 使用GQA ($g=8$)，平衡效果与效率
3. **长序列** ($L > 8192$): 使用MLA或MQA，优先节省内存
4. **资源受限**: 使用MQA，最大化吞吐

#### 7.2 GQA的组数选择

经验公式：

\begin{equation}
g = \min\left(\frac{M_{\text{avail}}}{M_{\text{target}}}, h\right)
\tag{59}
\end{equation}

其中 $M_{\text{avail}}$ 是可用内存，$M_{\text{target}}$ 是目标KV Cache大小。

**示例**: 希望KV Cache不超过1GB，$L=8192$:

\begin{equation}
2gLd_k \times 2 \leq 1024 \text{ MB} \Rightarrow g \leq \frac{1024 \times 10^6}{2 \times 8192 \times 128 \times 2} \approx 12
\tag{60}
\end{equation}

选择 $g=8$ (最接近的2的幂次)。

#### 7.3 MLA的参数配置

**低秩维度选择**: DeepSeek-V2的经验是 $d_c \approx d/10$:

\begin{equation}
d_c = \max\left(\frac{d}{10}, 512\right)
\tag{61}
\end{equation}

**RoPE维度**: 通常取 $d_r = d_k / 2$:

\begin{equation}
d_r = \frac{d_k}{2} = \frac{d/h}{2}
\tag{62}
\end{equation}

**Q的低秩维度**: 约为K的3倍:

\begin{equation}
d_c' \approx 3d_c
\tag{63}
\end{equation}

#### 7.4 推理优化实践

**量化**: FP16 → INT8可进一步减少2倍KV Cache:

\begin{equation}
\text{Cache}_{\text{INT8}} = \frac{\text{Cache}_{\text{FP16}}}{2}
\tag{64}
\end{equation}

**分页注意力**: 将KV Cache分块存储，减少内存碎片：

\begin{equation}
\text{Efficiency} = \frac{L \times \text{Block Size}}{\lceil L / \text{Block Size} \rceil \times \text{Block Size}} \geq 90\%
\tag{65}
\end{equation}

推荐Block Size = 64 或 128。

### 八、总结

本文详细推导了从MHA到MLA的演变过程，关键公式总结：

**KV Cache大小**:
\begin{equation}
\begin{cases}
\text{MHA}: & 2hLd_k \\
\text{MQA}: & 2Ld_k \\
\text{GQA}: & 2gLd_k \\
\text{MLA}: & L(d_c + d_r)
\end{cases}
\tag{66}
\end{equation}

**压缩比**:
\begin{equation}
\text{Ratio}_{\text{MLA}} = \frac{2hd_k}{d_c + d_r} = \frac{2 \times 32 \times 128}{512 + 64} \approx 14.2\times
\tag{67}
\end{equation}

**推理加速**:
\begin{equation}
\text{Speedup} \approx \text{Compression Ratio} \times \text{Bandwidth Factor}
\tag{68}
\end{equation}

MLA通过巧妙的数学变换，在几乎不损失模型能力的前提下，实现了显著的内存节省和推理加速，是Transformer推理优化的重要里程碑。

