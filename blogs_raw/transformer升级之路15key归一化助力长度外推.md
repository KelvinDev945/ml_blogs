---
title: Transformer升级之路：15、Key归一化助力长度外推
slug: transformer升级之路15key归一化助力长度外推
date: 2023-11-20
tags: 详细推导, attention, 位置编码, 泛化, 外推, 生成模型
status: pending
---
# Transformer升级之路：15、Key归一化助力长度外推

**原文链接**: [https://spaces.ac.cn/archives/9859](https://spaces.ac.cn/archives/9859)

**发布日期**: 

---

大体上，我们可以将目前Transformer的长度外推技术分为两类：一类是事后修改，比如[NTK-RoPE](/archives/9675)、[YaRN](https://papers.cool/arxiv/2309.00071)、[ReRoPE](/archives/9708)等，这类方法的特点是直接修改推理模型，无需微调就能达到一定的长度外推效果，但缺点是它们都无法保持模型在训练长度内的恒等性；另一类自然是事前修改，如[ALIBI](/archives/9431#ALIBI)、[KERPLE](/archives/9431#KERPLE)、[XPOS](/archives/9431#XPOS)以及[HWFA](/archives/9603)等，它们可以不加改动地实现一定的长度外推，但相应的改动需要在训练之前就引入，因此无法不微调地用于现成模型，并且这类方法是否能够Scale Up还没得到广泛认可。

在这篇文章中，笔者将介绍一种意外发现的长度外推方案——“KeyNorm”——对Attention的Key序列做L2 Normalization，很明显它属于事前修改一类，但对Attention机制的修改非常小，因此看上去非常有希望能够Scale Up。

## 最初动机 #

之所以说“意外发现”，是因为该改动的原始动机并不是长度外推，而是尝试替换Scaled Dot-Product Attention中的Scale方式。我们知道，Attention的标准定义是（本文主要考虑Causal场景）  
\begin{equation}\boldsymbol{o}_i = \frac{\sum_{j = 1}^i\exp\left(\frac{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}{\sqrt{d}}\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\frac{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}{\sqrt{d}}\right)},\quad \boldsymbol{q}_i,\boldsymbol{k}_j\in\mathbb{R}^d\label{eq:sdpa}\end{equation}  
其中，Scale因子$\frac{1}{\sqrt{d}}$我们已经多次进行过解释甚至推广，比如[《浅谈Transformer的初始化、参数化与标准化》](/archives/8620#NTK%E5%8F%82%E6%95%B0%E5%8C%96)、[《从熵不变性看Attention的Scale操作》](/archives/8823)、[《从梯度最大化看Attention的Scale操作》](/archives/9812)等。标准的推导是在“$\boldsymbol{q}_i,\boldsymbol{k}_j$均独立地采样自“均值为0、方差为1”的分布”的假设下进行的，而在该假设之下，我们还有  
\begin{equation}\Vert\boldsymbol{q}_i\Vert\approx \sqrt{d},\quad \Vert\boldsymbol{k}_j\Vert\approx \sqrt{d}\end{equation}  
这是因为  
\begin{equation}\Vert\boldsymbol{x}\Vert^2 = \sum_{i=1}^d x_i^2 = d\times\frac{1}{d}\sum_{i=1}^d x_i^2\approx d\,\mathbb{E}_{x\sim\mathcal{N}(0,1)}[x^2] = d\end{equation}  
相关推广还可以参考[《让人惊叹的Johnson-Lindenstrauss引理：理论篇》](/archives/8679#%E5%BC%95%E7%90%86%E7%9A%84%E5%BC%95%E7%90%86)。这个近似式意味着，在Attention的初始阶段式$\eqref{eq:sdpa}$与下面两个变体有着相同的效果：  
\begin{align}\color{red}{\text{Q}}\text{uery}\color{red}{\text{N}}\text{orm:}\quad\boldsymbol{o}_i =&\, \frac{\sum_{j = 1}^i\exp\left(\tilde{\boldsymbol{q}}_i\cdot \boldsymbol{k}_j\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\tilde{\boldsymbol{q}}_i\cdot \boldsymbol{k}_j\right)},\qquad \tilde{\boldsymbol{q}}_i = \frac{\boldsymbol{q}_i}{\Vert\boldsymbol{q}_i\Vert} \\\\[5pt]  
\color{red}{\text{K}}\text{ey}\color{red}{\text{N}}\text{orm:}\quad\boldsymbol{o}_i =&\, \frac{\sum_{j = 1}^i\exp\left(\boldsymbol{q}_i\cdot \tilde{\boldsymbol{k}}_j\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\boldsymbol{q}_i\cdot \tilde{\boldsymbol{k}}_j\right)},\qquad \tilde{\boldsymbol{k}}_j = \frac{\boldsymbol{k}_j}{\Vert\boldsymbol{k}_j\Vert}  
\end{align}  
因此，就有了验证这两个变体与标准的式$\eqref{eq:sdpa}$哪个更优的想法。为了描述的方便，我们可以相应地称为“Query/Key-Normalized Dot-Product Attention”，分别简称为“QNA”和“KNA”。

此外，既然可以QueryNorm和KeyNorm，那么自然也可以考虑两者都Norm一下，所以我们将如下“Scaled Cosine Attention（CosA）”也一并进行实验：  
\begin{equation}\boldsymbol{o}_i = \frac{\sum_{j = 1}^i\exp\left(\lambda\,\tilde{\boldsymbol{q}}_i\cdot \tilde{\boldsymbol{k}}_j\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\lambda\,\tilde{\boldsymbol{q}}_i\cdot \tilde{\boldsymbol{k}}_j\right)} = \frac{\sum_{j = 1}^i\exp\left(\lambda\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\lambda\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)\right)}  
\end{equation}  
其中$\lambda$采用[《从梯度最大化看Attention的Scale操作》](/archives/9812)中的结果，即$\lambda = 4\log n$（原文是3.5，但下面训练长度比较小，改为4更精准一些），其中$n$固定为训练长度的一半，或者动态取位置id加1。

## 先看结果 #

沿着[之前](/archives/9731#%E5%AE%9E%E9%AA%8C)做长度外推的实验设置，都是1亿参数的小模型，[GAU](/archives/9052)架构，训练相同的步数（时间有限，这个步数下其实模型还没训练充分），训练长度512，并考虑外推到4096长度，实验结果如下表。其中Baseline就是式$\eqref{eq:sdpa}$，$\text{-}\log n$就是加入[《从熵不变性看Attention的Scale操作》](/archives/8823)介绍的长度相关的缩放因子。评价指标是语言模型的逐token准确率，越大越好。  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复}) \\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\\  
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\\  
\hline  
\text{QNA} & 49.55\% & 22.45\% & 22.18\% \\\  
\text{QNA-}\log n & 49.42\% & 19.55\% & 18.74\% \\\  
\text{KNA} & 49.60\% & 61.08\% & 47.69\% \\\  
\text{KNA-}\log n & 49.58\% & 63.17\% & 46.40\%\\\  
\text{CosA} & 49.73\% & 58.90\% & 46.98\% \\\  
\text{CosA-}\log n & 49.67\% & 64.74\% & 48.95\% \\\  
\hline  
\end{array}  
从表格中我们可以看出：1、不管是QueryNorm还是KeyNorm，都在训练长度上取得了更好的效果，虽然这个优势非常微弱，大概率随着训练的进一步推进可以忽略不计，但这个优势非常稳定，暗示着让训练更加平稳的可能性；2、**KeyNorm对长度外推的提升非常明显** ，这就是实验结果中的“意外之喜”！

注意，跟NTK-RoPE、YaRN需要修改推理阶段的模型不同，这里的KNA和CosA的长度外推在推理阶段是完全不做改动的。因此，可能有读者想知道，既然KNA和CosA推理时不加改动外推效果都这么好了，如果配合NTK-RoPE、YaRN等外推技巧，效果会不会“更上一层楼”？对此，笔者也进行了测试，结果如下表：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复}) \\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\\  
\text{Baseline-NTK} & 49.41\% & 60.57\% & 42.20\% \\\  
\text{Baseline-YaRN} & 49.41\% & 80.10\% & 47.45\% \\\  
\text{Baseline-ReRoPE} & 49.41\% & 76.11\% & 47.82\% \\\  
\hline  
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\\  
\text{Baseline-}\log n\text{-NTK} & 49.40\% & 75.86\% & 47.06\% \\\  
\text{Baseline-}\log n\text{-YaRN} & 49.40\% & 82.57\% & 46.52\% \\\  
\text{Baseline-}\log n\text{-ReRoPE} & 49.40\% & 85.47\% & 48.87\% \\\  
\hline  
\text{QNA} & 49.55\% & 22.45\% & 22.18\% \\\  
\text{QNA-NTK} & 49.55\% & 52.28\% & 39.88\% \\\  
\text{QNA-YaRN} & 49.55\% & 82.53\% & 47.50\% \\\  
\text{QNA-ReRoPE} & 49.55\% & 78.22\% & 47.72\% \\\  
\hline  
\text{QNA-}\log n & 49.42\% & 19.55\% & 18.74\% \\\  
\text{QNA-}\log n\text{-NTK} & 49.42\% & 57.44\% & 41.56\% \\\  
\text{QNA-}\log n\text{-YaRN} & 49.42\% & 80.08\% & 45.16\% \\\  
\text{QNA-}\log n\text{-ReRoPE} & 49.42\% & 84.71\% & 48.31\% \\\  
\hline  
\text{KNA} & 49.60\% & 61.08\% & 47.69\% \\\  
\text{KNA-NTK} & 49.60\% & 64.44\% & 43.02\% \\\  
\text{KNA-YaRN} & 49.60\% & 84.19\% & 47.44\% \\\  
\text{KNA-ReRoPE} & 49.60\% & 77.76\% & 47.73\% \\\  
\hline  
\text{KNA-}\log n & 49.58\% & 63.17\% & 46.40\%\\\  
\text{KNA-}\log n\text{-NTK} & 49.58\% & 79.05\% & 47.43\%\\\  
\text{KNA-}\log n\text{-YaRN} & 49.58\% & 83.95\% & 47.16\%\\\  
\text{KNA-}\log n\text{-ReRoPE} & 49.58\% & 85.48\% & 48.78\%\\\  
\hline  
\text{CosA} & 49.73\% & 58.90\% & 46.98\% \\\  
\text{CosA-NTK} & 49.73\% & 62.50\% & 42.77\% \\\  
\text{CosA-YaRN} & 49.73\% & 83.40\% & 47.80\% \\\  
\text{CosA-ReRoPE} & 49.73\% & 77.82\% & 47.80\% \\\  
\hline  
\text{CosA-}\log n & 49.67\% & 64.74\% & 48.39\% \\\  
\text{CosA-}\log n\text{-NTK} & 49.67\% & 78.97\% & 47.46\% \\\  
\text{CosA-}\log n\text{-YaRN} & 49.67\% & 82.28\% & 45.72\% \\\  
\text{CosA-}\log n\text{-ReRoPE} & 49.67\% & 85.67\% & 48.39\% \\\  
\hline  
\end{array}  
这个表比较啰嗦，主要是为了让大家对主流长度外推技巧的效果差异有一个全面的感知，大家选择自己感兴趣的维度比较就好，但要注意如果看长度外推效果的话，应该以“不重复”一列为主，“重复”一列为辅。从上表看，结果着实有点让人意外，KeyNorm似乎“免疫”了已有的RoPE外推技巧，NTK、YaRN等技巧叠加上去并没有明显提升，甚至可能会下降，不过总体来看“重复”一列还是有显著提升的，不显著的是“不重复”一列。这些结果表明，KeyNorm依然有着无法有效识别超出训练长度的位置（所以“重复”的结果不高）的问题，但有效地避免了PPL爆炸问题（所以“不重复”的结果还不错）。

这对做Long Context的同学来说可能是个好消息：一方面，KeyNorm不像ALIBI、KERPLE等，它的长度外推不用加Local约束，训练完成后也不做任何修改，纯属是“免费的午餐”，甚至看上去加了KeyNorm后训练效果都变好了；另一方面，也因为它是非Local的，所以可以更长文本继续训练，并且继续训练时再也不用纠结是选[PI](https://papers.cool/arxiv/2306.15595)还是[ABF](https://papers.cool/arxiv/2309.16039)了，对于KeyNorm来说，啥也不改就行。

## 原理分析 #

尽管这是个意外发现，但我们仍需要尝试去解释它，不然它就一直只是个意外。所以这一节我们尝试来思考，为什么KeyNorm会有助于长度外推。

让我们重新回到式$\eqref{eq:sdpa}$，第$i$个token与第$j$个token的相关性打分由内积完成：  
\begin{equation}s(j|i) = \boldsymbol{q}_i\cdot \boldsymbol{k}_j = \Vert\boldsymbol{q}_i\Vert \Vert\boldsymbol{k}_j\Vert \cos(\boldsymbol{q}_i,\boldsymbol{k}_j),\quad p(j|i) = \frac{\exp\left(\frac{s(j|i)}{\sqrt{d}}\right)}{\sum_{j=1}^i \exp\left(\frac{s(j|i)}{\sqrt{d}}\right)}\end{equation}  
第二个等号，我们从几何意义出发，将它分解为了各自模长与夹角余弦的乘积。注意力$p(j|i)$是一个条件概率，$\Vert\boldsymbol{q}_i\Vert$只跟当前位置$i$有关，它不改变注意力的相对大小，而只改变[稀疏程度](/archives/9595)；$\Vert\boldsymbol{k}_j\Vert$则有能力改变$p(j|i)$的相对大小，但它不涉及到$i,j$的交互，可以用来表达一些绝对信号，比如[Scissorhands](https://papers.cool/arxiv/2305.17118)表明某些绝对位置的token的注意力一直都会很高，这就有可能用$\Vert\boldsymbol{k}_j\Vert$来表达；剩下的$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$就是用来表达$i,j$的交互，它是自由度最大的一项。

很明显，为了提高某个位置$j$的相对重要性，模型有两个选择：1、增大模长$\Vert\boldsymbol{k}_j\Vert$；2、增大$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$，即缩小$\boldsymbol{q}_i,\boldsymbol{k}_j$的夹角大小。然而，由于“[维度灾难](/archives/7076)”的存在，在高维空间中显著地改变夹角大小相对来说没有那么容易，所以如果能靠增大模长$\Vert\boldsymbol{k}_j\Vert$完成的，模型会优先选择通过增大模长$\Vert\boldsymbol{k}_j\Vert$来完成，这导致的直接后果是：$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练可能并不充分。

这里笔者作出一个断言（猜测）：

> $\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练不充分是Attention无法长度外推的主要原因。

$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练不充分，是指被训练过的$\boldsymbol{q}_i,\boldsymbol{k}_j$的夹角只是一个有限的集合，而进行长度外推时，它要面对一个更大的集合，从而无法进行正确的预测。仔细思考[YaRN](https://papers.cool/arxiv/2309.00071)一文的推导就会发现，NTK、YaRN之所以有效，是因为修改了推理阶段RoPE的实现，使得$\boldsymbol{q}_i,\boldsymbol{k}_j$的夹角落到原本训练阶段的有限集合中，避免面对没见过的更大的集合，转外推为内插；ReRoPE则更加干脆，直接截断Window以外的相对位置，这使得推理阶段的位置编码都不会“面生”。这些技巧一定程度上都间接地验证了这个断言。

从这个断言出发，KeyNorm的长度外推起因就变得简单了。不论是只进行KeyNorm的KNA，还是QueryNorm、KeyNorm都做的CosA，它们都将$\Vert\boldsymbol{k}_j\Vert$从Attention的定义中排除掉了，于是为了改变$j$的相对重要性，模型就只有“调整$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$”这一个选择，这将会使得模型更加充分地训练和利用$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$，从而间接促进了长度外推性。此外，笔者也实验过“KeyNorm + NoPE”的组合，但并没有发现长度外推性，这说明RoPE也在KeyNorm的长度外推中担任重要角色。事实上这也不难理解，RoPE对$\boldsymbol{q}_i,\boldsymbol{k}_j$进行旋转，更有助于扩大训练期间$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的范围，从而使得$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练更为充分。

有没有工作已经尝试过QueryNorm和KeyNorm了呢？有。2020年的论文[《Query-Key Normalization for Transformers》](https://papers.cool/arxiv/2010.04245)曾实验过CosA，论文还提出了一个类似的长度对数的Scale因子，但没有讨论到长度外推问题。此外，今年初Google的论文[《Scaling Vision Transformers to 22 Billion Parameters》](https://papers.cool/arxiv/2302.05442)也在Query和Key加了Norm，但加的是LayerNorm，LayerNorm或者RMSNorm都带有可学的gamma参数，这使得Norm之后的向量模长未必为常数，因此并不好说是否能达到本文一样的长度外推效果。

## 文章小结 #

本文介绍了笔者意外发现的一种长度外推方案“KeyNorm”——对Attention的Key序列进行L2归一化，在训练长度上取得了更好的效果，并在长度外推方面表现出显著的提升。它属于“事前修改”方案，跟其他事前修改方案如ALIBI、KERPLE等相比，它没有Local约束，因此更有希望能够Scale Up；相比于NTK-RoPE、YaRN等“事后修改”方案，它在外推的时候则不会损失训练长度内的性能。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9859>_

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

苏剑林. (Nov. 20, 2023). 《Transformer升级之路：15、Key归一化助力长度外推 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9859>

@online{kexuefm-9859,  
title={Transformer升级之路：15、Key归一化助力长度外推},  
author={苏剑林},  
year={2023},  
month={Nov},  
url={\url{https://spaces.ac.cn/archives/9859}},  
} 


---

## 公式推导与注释

### 1. Key归一化的数学基础

**定义1.1：标准Scaled Dot-Product Attention**

标准的注意力机制可以表示为：
$$
\boldsymbol{o}_i = \frac{\sum_{j=1}^i \exp\left(\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\sqrt{d}}\right) \boldsymbol{v}_j}{\sum_{j=1}^i \exp\left(\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\sqrt{d}}\right)}
$$

其中：
- $\boldsymbol{q}_i, \boldsymbol{k}_j \in \mathbb{R}^d$ 分别是Query和Key向量
- $d$ 是向量维度
- $\boldsymbol{v}_j \in \mathbb{R}^d$ 是Value向量
- $i$ 表示当前位置（Causal Mask）

**推导1.1：Scale因子$\frac{1}{\sqrt{d}}$的来源**

假设$\boldsymbol{q}_i, \boldsymbol{k}_j$的每个分量独立同分布于$\mathcal{N}(0,1)$，则：
$$
\boldsymbol{q}_i \cdot \boldsymbol{k}_j = \sum_{\ell=1}^d q_{i,\ell} k_{j,\ell}
$$

由于$q_{i,\ell} \sim \mathcal{N}(0,1)$且$k_{j,\ell} \sim \mathcal{N}(0,1)$是独立的，我们有：
$$
\mathbb{E}[q_{i,\ell} k_{j,\ell}] = \mathbb{E}[q_{i,\ell}] \mathbb{E}[k_{j,\ell}] = 0 \cdot 0 = 0
$$

方差为：
$$
\text{Var}[q_{i,\ell} k_{j,\ell}] = \mathbb{E}[(q_{i,\ell} k_{j,\ell})^2] - (\mathbb{E}[q_{i,\ell} k_{j,\ell}])^2 = \mathbb{E}[q_{i,\ell}^2]\mathbb{E}[k_{j,\ell}^2] = 1 \cdot 1 = 1
$$

因此内积的方差为：
$$
\text{Var}\left[\sum_{\ell=1}^d q_{i,\ell} k_{j,\ell}\right] = \sum_{\ell=1}^d \text{Var}[q_{i,\ell} k_{j,\ell}] = d
$$

故：
$$
\boldsymbol{q}_i \cdot \boldsymbol{k}_j \sim \mathcal{N}(0, d)
$$

为了归一化方差到1，我们除以$\sqrt{d}$：
$$
\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\sqrt{d}} \sim \mathcal{N}(0, 1)
$$

### 2. 向量模长的统计性质

**定理2.1：高维随机向量的模长**

对于$\boldsymbol{x} \in \mathbb{R}^d$，其中$x_i \stackrel{i.i.d.}{\sim} \mathcal{N}(0,1)$，有：
$$
\|\boldsymbol{x}\|^2 = \sum_{i=1}^d x_i^2
$$

由于$x_i^2 \sim \chi^2(1)$（卡方分布），所以：
$$
\|\boldsymbol{x}\|^2 \sim \chi^2(d)
$$

**推导2.1：模长的期望**

卡方分布$\chi^2(d)$的期望和方差分别为$d$和$2d$，因此：
$$
\mathbb{E}[\|\boldsymbol{x}\|^2] = d
$$

利用Jensen不等式（由于平方根是凹函数）：
$$
\mathbb{E}[\|\boldsymbol{x}\|] \leq \sqrt{\mathbb{E}[\|\boldsymbol{x}\|^2]} = \sqrt{d}
$$

实际上，对于卡方分布，我们有更精确的结果：
$$
\mathbb{E}[\|\boldsymbol{x}\|] = \sqrt{2} \frac{\Gamma((d+1)/2)}{\Gamma(d/2)} \approx \sqrt{d} - \frac{1}{4\sqrt{d}} + O(d^{-3/2})
$$

当$d$较大时，我们有近似：
$$
\|\boldsymbol{x}\| \approx \sqrt{d}
$$

**推导2.2：集中不等式**

利用卡方分布的集中性，对于任意$\epsilon > 0$：
$$
\mathbb{P}\left(\left|\|\boldsymbol{x}\|^2 - d\right| > \epsilon d\right) \leq 2\exp\left(-\frac{d\epsilon^2}{8}\right)
$$

这说明当维度$d$较大时，$\|\boldsymbol{x}\|^2$高概率集中在$d$附近，因此：
$$
\mathbb{P}\left(\left|\|\boldsymbol{x}\| - \sqrt{d}\right| > \frac{\epsilon\sqrt{d}}{2}\right) \to 0 \quad \text{当} \quad d \to \infty
$$

### 3. Key Normalization的数学定义

**定义3.1：Key Normalized Attention (KNA)**

Key归一化注意力定义为：
$$
\boldsymbol{o}_i = \frac{\sum_{j=1}^i \exp\left(\boldsymbol{q}_i \cdot \tilde{\boldsymbol{k}}_j\right) \boldsymbol{v}_j}{\sum_{j=1}^i \exp\left(\boldsymbol{q}_i \cdot \tilde{\boldsymbol{k}}_j\right)}
$$

其中归一化的Key向量为：
$$
\tilde{\boldsymbol{k}}_j = \frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|}
$$

满足$\|\tilde{\boldsymbol{k}}_j\| = 1$。

**定理3.1：KNA与标准Attention的初始等价性**

在训练初期，当$\boldsymbol{q}_i, \boldsymbol{k}_j$的分量独立同分布于$\mathcal{N}(0,1)$时，KNA近似等价于标准的Scaled Attention。

**证明：**

由前述分析，当$\boldsymbol{k}_j$的分量独立同分布于$\mathcal{N}(0,1)$时：
$$
\|\boldsymbol{k}_j\| \approx \sqrt{d}
$$

因此：
$$
\tilde{\boldsymbol{k}}_j = \frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|} \approx \frac{\boldsymbol{k}_j}{\sqrt{d}}
$$

代入KNA的定义：
$$
\boldsymbol{q}_i \cdot \tilde{\boldsymbol{k}}_j \approx \boldsymbol{q}_i \cdot \frac{\boldsymbol{k}_j}{\sqrt{d}} = \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\sqrt{d}}
$$

这恰好是标准Attention中的形式。$\square$

### 4. Query-Key内积的几何分解

**定理4.1：内积的几何分解**

任意两个向量$\boldsymbol{q}_i, \boldsymbol{k}_j \in \mathbb{R}^d$的内积可以分解为：
$$
\boldsymbol{q}_i \cdot \boldsymbol{k}_j = \|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)
$$

其中$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$是两向量夹角的余弦值：
$$
\cos(\boldsymbol{q}_i, \boldsymbol{k}_j) = \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\|}
$$

**推导4.1：三个因子的作用分析**

在标准Attention中，注意力分数为：
$$
s(j|i) = \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\sqrt{d}} = \frac{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)}{\sqrt{d}}
$$

三个因子的作用：

1. **Query模长$\|\boldsymbol{q}_i\|$**：只依赖于位置$i$，不改变注意力的相对大小，只影响注意力分布的稀疏程度（温度）。

2. **Key模长$\|\boldsymbol{k}_j\|$**：能够改变位置$j$的绝对重要性，不涉及$i,j$的交互。

3. **余弦$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$**：表达$i,j$之间的真正交互，取值范围$[-1,1]$，是自由度最大的一项。

**推导4.2：注意力权重的详细展开**

注意力权重为：
$$
\alpha_{ij} = \frac{\exp\left(\frac{s(j|i)}{\sqrt{d}}\right)}{\sum_{k=1}^i \exp\left(\frac{s(k|i)}{\sqrt{d}}\right)}
$$

代入几何分解：
$$
\alpha_{ij} = \frac{\exp\left(\frac{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)}{\sqrt{d}}\right)}{\sum_{k=1}^i \exp\left(\frac{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_k\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_k)}{\sqrt{d}}\right)}
$$

由于$\|\boldsymbol{q}_i\|$在分子分母中都有，可以提取出来：
$$
\alpha_{ij} = \frac{\exp\left(\frac{\|\boldsymbol{k}_j\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)}{\sqrt{d}/\|\boldsymbol{q}_i\|}\right)}{\sum_{k=1}^i \exp\left(\frac{\|\boldsymbol{k}_k\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_k)}{\sqrt{d}/\|\boldsymbol{q}_i\|}\right)}
$$

这表明$\|\boldsymbol{q}_i\|$起到温度参数的作用。

### 5. Key归一化对注意力分布的影响

**定理5.1：KNA消除了Key模长的影响**

在Key归一化后，注意力分数变为：
$$
s_{KNA}(j|i) = \boldsymbol{q}_i \cdot \tilde{\boldsymbol{k}}_j = \|\boldsymbol{q}_i\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)
$$

**推导5.1：KNA的注意力权重**

$$
\alpha_{ij}^{KNA} = \frac{\exp\left(\|\boldsymbol{q}_i\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)\right)}{\sum_{k=1}^i \exp\left(\|\boldsymbol{q}_i\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_k)\right)}
$$

对比标准Attention，KNA有以下特点：

1. **消除了Key模长的影响**：所有Key的模长都被归一化为1，模型无法通过增大$\|\boldsymbol{k}_j\|$来提高位置$j$的重要性。

2. **强制模型优化余弦相似度**：模型只能通过调整$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$（即向量方向）来改变注意力分布。

3. **保留了Query模长的温度效应**：$\|\boldsymbol{q}_i\|$仍然可以控制注意力分布的尖锐程度。

**定理5.2：余弦相似度的有界性**

由于$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j) \in [-1, 1]$，KNA的注意力分数满足：
$$
-\|\boldsymbol{q}_i\| \leq s_{KNA}(j|i) \leq \|\boldsymbol{q}_i\|
$$

这个有界性保证了注意力分数不会过大，有助于训练稳定性。

### 6. 维度灾难与角度调整的困难

**定理6.1：高维空间中的角度分布**

在$d$维空间中，随机向量之间的夹角余弦值趋向于集中在0附近。

**推导6.1：随机向量夹角的分布**

设$\boldsymbol{u}, \boldsymbol{v} \in \mathbb{R}^d$为两个独立的随机向量，分量独立同分布于$\mathcal{N}(0,1)$。定义：
$$
\rho = \cos(\boldsymbol{u}, \boldsymbol{v}) = \frac{\boldsymbol{u} \cdot \boldsymbol{v}}{\|\boldsymbol{u}\| \|\boldsymbol{v}\|}
$$

由于$\boldsymbol{u} \cdot \boldsymbol{v} = \sum_{i=1}^d u_i v_i$，其中$u_i v_i$独立，我们有：
$$
\mathbb{E}[\boldsymbol{u} \cdot \boldsymbol{v}] = \sum_{i=1}^d \mathbb{E}[u_i v_i] = 0
$$
$$
\text{Var}[\boldsymbol{u} \cdot \boldsymbol{v}] = \sum_{i=1}^d \text{Var}[u_i v_i] = d
$$

因此$\boldsymbol{u} \cdot \boldsymbol{v} \sim \mathcal{N}(0, d)$。同时：
$$
\|\boldsymbol{u}\|^2 \sim \chi^2(d), \quad \|\boldsymbol{v}\|^2 \sim \chi^2(d)
$$

由大数定律：
$$
\frac{\|\boldsymbol{u}\|^2}{d} \to 1, \quad \frac{\|\boldsymbol{v}\|^2}{d} \to 1 \quad \text{当} \quad d \to \infty
$$

因此：
$$
\rho = \frac{\boldsymbol{u} \cdot \boldsymbol{v}}{\|\boldsymbol{u}\| \|\boldsymbol{v}\|} \approx \frac{\boldsymbol{u} \cdot \boldsymbol{v}}{d}
$$

标准化后：
$$
\sqrt{d} \cdot \rho \approx \frac{\boldsymbol{u} \cdot \boldsymbol{v}}{\sqrt{d}} \sim \mathcal{N}(0, 1)
$$

即：
$$
\rho \sim \mathcal{N}\left(0, \frac{1}{d}\right)
$$

这说明随着维度$d$增大，余弦相似度集中在0附近，方差为$O(1/d)$。

**推导6.2：角度调整的梯度分析**

要改变两个向量的余弦相似度，需要调整向量的方向。设我们要通过梯度下降优化$\boldsymbol{k}_j$使得$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$增大。损失函数为：
$$
\mathcal{L} = -\cos(\boldsymbol{q}_i, \boldsymbol{k}_j) = -\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\|}
$$

梯度为：
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{k}_j} = -\frac{\boldsymbol{q}_i}{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\|} + \frac{(\boldsymbol{q}_i \cdot \boldsymbol{k}_j) \boldsymbol{k}_j}{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\|^3}
$$

简化为：
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{k}_j} = -\frac{1}{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\|}\left(\boldsymbol{q}_i - \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\|\boldsymbol{k}_j\|^2} \boldsymbol{k}_j\right)
$$

这是$\boldsymbol{q}_i$在垂直于$\boldsymbol{k}_j$方向上的分量（归一化后）。

另一方面，如果直接优化模长：
$$
\mathcal{L}' = -\|\boldsymbol{k}_j\|
$$

梯度为：
$$
\frac{\partial \mathcal{L}'}{\partial \boldsymbol{k}_j} = -\frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|}
$$

在高维空间中，改变向量方向（调整余弦）比改变向量模长需要更精细的协调，因此模型倾向于优先调整模长。

### 7. 长度外推与位置编码的训练充分性

**定义7.1：位置编码的训练集**

设训练长度为$L_{train}$，则训练过程中出现的相对位置集合为：
$$
\mathcal{P}_{train} = \{m - n : 0 \leq m, n \leq L_{train}, m \geq n\} = \{0, 1, 2, \ldots, L_{train}\}
$$

在长度外推到$L_{test} > L_{train}$时，需要预测的相对位置集合为：
$$
\mathcal{P}_{test} = \{0, 1, 2, \ldots, L_{test}\}
$$

显然$\mathcal{P}_{train} \subset \mathcal{P}_{test}$，且$\mathcal{P}_{test} \setminus \mathcal{P}_{train} \neq \emptyset$。

**假设7.1：余弦训练充分性假设**

模型能够成功进行长度外推，当且仅当$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$在训练阶段得到充分训练，使得其学到的模式能够泛化到新的相对位置。

**推导7.1：标准Attention中余弦训练不足的原因**

在标准Attention中，为了提高位置$j$相对于位置$i$的注意力权重，模型有两个选择：

1. **增大$\|\boldsymbol{k}_j\|$**：这不需要改变向量方向，相对简单。
2. **增大$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$**：这需要精细调整向量方向，在高维空间中较困难。

由于路径1更容易，模型会优先选择调整$\|\boldsymbol{k}_j\|$，导致：
- $\|\boldsymbol{k}_j\|$充分训练，但它编码的是绝对位置信息
- $\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$训练不足，但它才是真正编码相对位置的关键

**推导7.2：KNA强制余弦充分训练**

在KNA中，由于$\|\tilde{\boldsymbol{k}}_j\| = 1$恒定，模型无法通过调整Key模长来改变注意力。因此：
$$
\text{唯一的优化路径：} \quad \max_{\boldsymbol{k}_j} \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)
$$

这强制模型充分训练余弦相似度，使得位置信息更多地编码在向量方向而非模长中。

### 8. RoPE与Key归一化的协同作用

**定理8.1：RoPE的旋转性质**

RoPE通过旋转矩阵$\boldsymbol{\mathcal{R}}_m$对Query和Key进行变换：
$$
\tilde{\boldsymbol{q}}_m = \boldsymbol{\mathcal{R}}_m \boldsymbol{q}_m, \quad \tilde{\boldsymbol{k}}_n = \boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n
$$

其中旋转矩阵保持向量模长不变：
$$
\|\boldsymbol{\mathcal{R}}_m \boldsymbol{q}_m\| = \|\boldsymbol{q}_m\|, \quad \|\boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n\| = \|\boldsymbol{k}_n\|
$$

且满足相对位置性质：
$$
\tilde{\boldsymbol{q}}_m \cdot \tilde{\boldsymbol{k}}_n = \boldsymbol{q}_m^T \boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n = \boldsymbol{q}_m^T \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}_n
$$

**推导8.1：RoPE增加角度多样性**

对于二维旋转矩阵：
$$
\boldsymbol{\mathcal{R}}_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

相对旋转为：
$$
\boldsymbol{\mathcal{R}}_{n-m} = \begin{pmatrix} \cos((n-m)\theta) & -\sin((n-m)\theta) \\ \sin((n-m)\theta) & \cos((n-m)\theta) \end{pmatrix}
$$

这使得$\boldsymbol{q}_m$和$\boldsymbol{k}_n$之间的夹角变为：
$$
\cos(\tilde{\boldsymbol{q}}_m, \tilde{\boldsymbol{k}}_n) = \cos(\boldsymbol{q}_m, \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}_n)
$$

通过不同的相对位置$n-m$，旋转角度$(n-m)\theta$会遍历$[0, 2\pi)$的多个值（当$\theta$选择合适时），这显著增加了不同位置对之间角度的多样性。

**推导8.2：角度覆盖率分析**

设训练长度为$L_{train}$，RoPE的基频率为$\theta$。在训练过程中，相对位置$\Delta = n - m \in [0, L_{train}]$对应的旋转角度为：
$$
\{\Delta \theta : \Delta = 0, 1, 2, \ldots, L_{train}\}
$$

这些角度在单位圆上的分布密度为：
$$
\rho = \frac{L_{train} \theta}{2\pi}
$$

当$\rho \gg 1$时，即：
$$
\theta > \frac{2\pi}{L_{train}}
$$

单位圆被密集覆盖，训练到的角度样本充分，有利于泛化到更长的序列。

**定理8.2：KNA + RoPE的协同效应**

KNA强制模型优化余弦相似度，RoPE增加角度的多样性。两者结合：

1. **KNA作用**：$\|\tilde{\boldsymbol{k}}_j\| = 1$，消除模长干扰
2. **RoPE作用**：通过旋转增加不同相对位置的角度多样性
3. **协同效应**：充分训练的角度模式 + 丰富的角度样本 = 更好的长度外推

### 9. Cosine Attention的数学表示

**定义9.1：Scaled Cosine Attention (CosA)**

Cosine Attention同时对Query和Key进行归一化：
$$
\boldsymbol{o}_i = \frac{\sum_{j=1}^i \exp\left(\lambda \tilde{\boldsymbol{q}}_i \cdot \tilde{\boldsymbol{k}}_j\right) \boldsymbol{v}_j}{\sum_{j=1}^i \exp\left(\lambda \tilde{\boldsymbol{q}}_i \cdot \tilde{\boldsymbol{k}}_j\right)}
$$

其中：
$$
\tilde{\boldsymbol{q}}_i = \frac{\boldsymbol{q}_i}{\|\boldsymbol{q}_i\|}, \quad \tilde{\boldsymbol{k}}_j = \frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|}
$$

$\lambda$是可学习的温度参数。

**推导9.1：CosA注意力分数的性质**

注意力分数为：
$$
s_{CosA}(j|i) = \lambda \tilde{\boldsymbol{q}}_i \cdot \tilde{\boldsymbol{k}}_j = \lambda \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)
$$

由于$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j) \in [-1, 1]$，有：
$$
s_{CosA}(j|i) \in [-\lambda, \lambda]
$$

这提供了严格的分数界限，有助于训练稳定性。

**定理9.1：温度参数的理论值**

根据熵不变性原理，温度参数应该设置为：
$$
\lambda = c \log n
$$

其中$n$是序列长度，$c$是常数（通常取4左右）。

**推导9.2：温度参数的梯度最大化推导**

假设我们希望注意力分布的最大梯度保持稳定。注意力对输入的梯度为：
$$
\frac{\partial \alpha_{ij}}{\partial s(j|i)} = \alpha_{ij}(1 - \alpha_{ij})
$$

梯度最大值出现在$\alpha_{ij} = 0.5$时，此时：
$$
\frac{\partial \alpha_{ij}}{\partial s(j|i)} = 0.25
$$

对于均匀分布（$n$个位置），有$\alpha_{ij} = 1/n$。softmax的性质告诉我们，当分数差异为：
$$
\Delta s = \log n
$$

时，能够产生显著的概率差异。因此温度参数应该随$\log n$缩放。

### 10. 温度参数的自适应调整

**定义10.1：动态温度参数**

对于变长序列，温度参数应该根据实际序列长度动态调整：
$$
\lambda(n) = \max(1, c \log n)
$$

其中$c \in [3.5, 4]$是常数，$n$是当前位置的索引（或序列长度的一半）。

**推导10.1：熵不变性条件**

设注意力分布为$\{\alpha_{ij}\}_{j=1}^i$，其熵为：
$$
H_i = -\sum_{j=1}^i \alpha_{ij} \log \alpha_{ij}
$$

在标准Attention中，由于softmax的性质，当分数尺度固定时：
$$
H_i \approx \log i - \text{const}
$$

即熵随序列长度对数增长。

为了保持熵相对稳定（相对于最大熵$\log i$的比例），我们需要温度参数随$\log i$调整：
$$
s'(j|i) = \frac{s(j|i)}{\lambda(i)}, \quad \lambda(i) \propto \log i
$$

**推导10.2：温度参数对注意力分布的影响**

注意力权重对温度的导数：
$$
\frac{\partial \alpha_{ij}}{\partial \lambda} = \frac{\partial}{\partial \lambda} \frac{\exp(\lambda s_{ij})}{\sum_k \exp(\lambda s_{ik})}
$$

利用softmax的导数性质：
$$
\frac{\partial \alpha_{ij}}{\partial \lambda} = \alpha_{ij} \left(s_{ij} - \sum_k \alpha_{ik} s_{ik}\right)
$$

这表明：
- 当$s_{ij}$高于平均分数时，增大$\lambda$会增大$\alpha_{ij}$（富者更富）
- 温度参数控制分布的尖锐程度

### 11. LayerNorm与L2 Normalization的对比

**定义11.1：LayerNorm**

LayerNorm对向量进行如下变换：
$$
\text{LayerNorm}(\boldsymbol{x}) = \gamma \odot \frac{\boldsymbol{x} - \mu}{\sigma} + \beta
$$

其中：
$$
\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}
$$

$\gamma, \beta \in \mathbb{R}^d$是可学习参数。

**定义11.2：L2 Normalization**

L2归一化仅调整模长：
$$
\text{L2Norm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\|\boldsymbol{x}\|}
$$

不改变向量方向，无可学习参数。

**对比11.1：中心化**

- **LayerNorm**：先减去均值$\mu$，进行中心化
- **L2Norm**：不进行中心化，保留原始方向

**对比11.2：缩放**

- **LayerNorm**：除以标准差$\sigma$后，再乘以可学习的$\gamma$，输出模长不固定
- **L2Norm**：固定输出模长为1，$\|\text{L2Norm}(\boldsymbol{x})\| = 1$

**对比11.3：可学习参数**

- **LayerNorm**：有$2d$个可学习参数（$\gamma, \beta$）
- **L2Norm**：无可学习参数

**定理11.1：LayerNorm可能破坏Key归一化的效果**

如果对Key应用LayerNorm而非L2归一化：
$$
\tilde{\boldsymbol{k}}_j = \gamma \odot \frac{\boldsymbol{k}_j - \mu_j}{\sigma_j} + \beta
$$

由于$\gamma$是可学习的，模型可以通过调整$\gamma$来改变$\|\tilde{\boldsymbol{k}}_j\|$：
$$
\|\tilde{\boldsymbol{k}}_j\| = \left\|\gamma \odot \frac{\boldsymbol{k}_j - \mu_j}{\sigma_j} + \beta\right\|
$$

这个模长不再固定为1，因此LayerNorm无法保证消除Key模长的影响。

### 12. 长度外推的稳定性分析

**定义12.1：PPL爆炸**

在长度外推时，如果测试集的困惑度(Perplexity)显著高于训练长度下的困惑度，我们称发生了PPL爆炸：
$$
\text{PPL}_{test}(L_{test}) \gg \text{PPL}_{train}(L_{train}) \quad \text{当} \quad L_{test} \gg L_{train}
$$

**定理12.1：KNA的数值稳定性**

在Key归一化后，注意力分数有严格的上界：
$$
|s_{KNA}(j|i)| = |\|\boldsymbol{q}_i\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)| \leq \|\boldsymbol{q}_i\|
$$

这防止了注意力分数过大导致的softmax溢出。

**推导12.1：标准Attention的不稳定性来源**

在标准Attention中，如果某个Key的模长异常大：
$$
\|\boldsymbol{k}_j\| \gg \sqrt{d}
$$

则对应的注意力分数：
$$
s(j|i) = \frac{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)}{\sqrt{d}}
$$

可能非常大，导致：
$$
\exp(s(j|i)) \gg \sum_{k \neq j} \exp(s(k|i))
$$

注意力几乎完全集中在位置$j$，损失了对其他位置的感知。

**推导12.2：KNA保持注意力分布的多样性**

由于$s_{KNA}(j|i)$有界，softmax不会出现极端集中：
$$
\alpha_{ij}^{KNA} = \frac{\exp(\|\boldsymbol{q}_i\| \cos_j)}{\sum_{k=1}^i \exp(\|\boldsymbol{q}_i\| \cos_k)}
$$

即使$\cos_j = 1$（最大值），只要存在其他$\cos_k$接近1的位置，注意力仍会分散。

### 13. 相对位置的外推误差界

**定义13.1：位置编码函数**

设位置编码为函数$f: \mathbb{N} \to \mathbb{R}^d$，满足相对位置性质：
$$
\boldsymbol{q}_m^T \boldsymbol{k}_n = g(f(m), f(n)) = h(m - n)
$$

其中$h: \mathbb{Z} \to \mathbb{R}$仅依赖于相对位置。

**定理13.1：训练位置集的Lipschitz性质**

假设$h$在训练集$[0, L_{train}]$上是Lipschitz连续的，常数为$L$：
$$
|h(\Delta_1) - h(\Delta_2)| \leq L |\Delta_1 - \Delta_2|, \quad \forall \Delta_1, \Delta_2 \in [0, L_{train}]
$$

**推导13.1：外推误差的上界**

当外推到$L_{test} > L_{train}$时，对于新的相对位置$\Delta \in (L_{train}, L_{test}]$，我们需要估计$h(\Delta)$。

最近邻插值：
$$
\hat{h}(\Delta) = h(L_{train})
$$

误差为：
$$
|h(\Delta) - \hat{h}(\Delta)| = |h(\Delta) - h(L_{train})|
$$

如果$h$满足全局Lipschitz条件（外推假设）：
$$
|h(\Delta) - h(L_{train})| \leq L (\Delta - L_{train}) \leq L (L_{test} - L_{train})
$$

误差随外推长度线性增长。

**推导13.2：KNA减小Lipschitz常数**

在标准Attention中：
$$
h(\Delta) = \|\boldsymbol{q}\| \|\boldsymbol{k}\| \cos(\Delta\theta) / \sqrt{d}
$$

其导数：
$$
|h'(\Delta)| = \frac{\|\boldsymbol{q}\| \|\boldsymbol{k}\| \theta |\sin(\Delta\theta)|}{\sqrt{d}} \leq \frac{\|\boldsymbol{q}\| \|\boldsymbol{k}\| \theta}{\sqrt{d}}
$$

在KNA中：
$$
h_{KNA}(\Delta) = \|\boldsymbol{q}\| \cos(\Delta\theta)
$$

导数：
$$
|h'_{KNA}(\Delta)| = \|\boldsymbol{q}\| \theta |\sin(\Delta\theta)| \leq \|\boldsymbol{q}\| \theta
$$

如果$\|\boldsymbol{k}\| > \sqrt{d}$（训练后期常见），则：
$$
|h'_{KNA}| < |h'_{std}|
$$

即KNA的Lipschitz常数更小，外推误差更小。

### 14. 注意力熵与长度外推的关系

**定义14.1：注意力熵**

位置$i$的注意力分布的熵定义为：
$$
H_i = -\sum_{j=1}^i \alpha_{ij} \log \alpha_{ij}
$$

熵越大，注意力越分散；熵越小，注意力越集中。

**定理14.1：长度外推需要适度的注意力熵**

成功的长度外推要求：
$$
\frac{H_i}{\log i} \approx \text{const}, \quad \forall i \in [1, L_{test}]
$$

即归一化熵保持稳定。

**推导14.1：熵崩塌导致PPL爆炸**

如果在长序列中$H_i \to 0$（熵崩塌），意味着注意力几乎完全集中在少数位置：
$$
\exists j^* : \alpha_{ij^*} \approx 1, \quad \alpha_{ij} \approx 0, \forall j \neq j^*
$$

这导致输出仅依赖于位置$j^*$的信息，丢失了其他位置的信息，预测质量下降。

**推导14.2：KNA维持注意力熵**

在KNA中，由于所有Key归一化，不存在某个位置通过极大的$\|\boldsymbol{k}_j\|$"垄断"注意力的情况。注意力分数：
$$
s_{KNA}(j|i) = \|\boldsymbol{q}_i\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)
$$

所有位置"平等竞争"，$\cos$值在$[-1,1]$范围内，差异相对温和，从而维持了适度的注意力熵。

### 15. Query Normalization的对比实验

**定义15.1：Query Normalized Attention (QNA)**

$$
\boldsymbol{o}_i = \frac{\sum_{j=1}^i \exp\left(\tilde{\boldsymbol{q}}_i \cdot \boldsymbol{k}_j\right) \boldsymbol{v}_j}{\sum_{j=1}^i \exp\left(\tilde{\boldsymbol{q}}_i \cdot \boldsymbol{k}_j\right)}, \quad \tilde{\boldsymbol{q}}_i = \frac{\boldsymbol{q}_i}{\|\boldsymbol{q}_i\|}
$$

**定理15.1：QNA无法实现长度外推**

实验表明，QNA在训练长度上效果与标准Attention相当，但在长度外推上表现糟糕，甚至劣于Baseline。

**推导15.1：QNA失效的原因**

在QNA中：
$$
s_{QNA}(j|i) = \tilde{\boldsymbol{q}}_i \cdot \boldsymbol{k}_j = \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\|\boldsymbol{q}_i\|} = \|\boldsymbol{k}_j\| \cos(\boldsymbol{q}_i, \boldsymbol{k}_j)
$$

Key模长$\|\boldsymbol{k}_j\|$仍然存在，模型仍可通过调整$\|\boldsymbol{k}_j\|$来改变注意力分布，因此$\cos$的训练不充分问题未解决。

同时，Query归一化消除了$\|\boldsymbol{q}_i\|$的温度效应：
$$
\alpha_{ij}^{QNA} = \frac{\exp(\|\boldsymbol{k}_j\| \cos_j)}{\sum_k \exp(\|\boldsymbol{k}_k\| \cos_k)}
$$

失去了通过$\|\boldsymbol{q}_i\|$调节注意力尖锐度的能力，反而增加了训练难度。

### 16. 多头注意力中的Key归一化

**定义16.1：多头Key归一化**

对于$H$个注意力头，每个头独立进行Key归一化：
$$
\tilde{\boldsymbol{k}}_{j}^{(h)} = \frac{\boldsymbol{k}_j^{(h)}}{\|\boldsymbol{k}_j^{(h)}\|}, \quad h = 1, 2, \ldots, H
$$

**推导16.1：不同头学习不同的角度模式**

由于每个头的Query和Key投影不同：
$$
\boldsymbol{q}_i^{(h)} = \boldsymbol{W}_Q^{(h)} \boldsymbol{x}_i, \quad \boldsymbol{k}_j^{(h)} = \boldsymbol{W}_K^{(h)} \boldsymbol{x}_j
$$

归一化后：
$$
\cos^{(h)}(\boldsymbol{q}_i, \boldsymbol{k}_j) = \frac{\boldsymbol{q}_i^{(h)} \cdot \boldsymbol{k}_j^{(h)}}{\|\boldsymbol{q}_i^{(h)}\| \|\boldsymbol{k}_j^{(h)}\|}
$$

不同的$\boldsymbol{W}_Q^{(h)}, \boldsymbol{W}_K^{(h)}$导致不同的余弦模式，实现多样化的位置感知。

**定理16.1：多头KNA增强表达能力**

多头KNA的输出为：
$$
\boldsymbol{o}_i = \boldsymbol{W}_O \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_H)
$$

其中每个head学习不同的相对位置模式，组合后能够表达复杂的位置依赖关系。

### 17. 训练稳定性的理论分析

**定义17.1：梯度范数**

训练稳定性可以通过梯度范数来衡量：
$$
\|\nabla_\theta \mathcal{L}\|
$$

如果梯度范数在训练过程中保持适度范围，训练稳定；如果出现梯度爆炸或消失，训练不稳定。

**定理17.1：KNA的梯度有界性**

由于KNA的注意力分数有界：
$$
|s_{KNA}(j|i)| \leq \|\boldsymbol{q}_i\|
$$

反向传播时梯度也相应有界，不会出现极端的梯度爆炸。

**推导17.1：注意力分数对参数的梯度**

$$
\frac{\partial s_{KNA}(j|i)}{\partial \theta} = \frac{\partial}{\partial \theta} \left(\boldsymbol{q}_i \cdot \frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|}\right)
$$

利用链式法则：
$$
\frac{\partial s_{KNA}}{\partial \theta} = \frac{\boldsymbol{q}_i}{\|\boldsymbol{k}_j\|} \cdot \frac{\partial \boldsymbol{k}_j}{\partial \theta} - \frac{(\boldsymbol{q}_i \cdot \boldsymbol{k}_j) \boldsymbol{k}_j}{\|\boldsymbol{k}_j\|^3} \cdot \frac{\partial \boldsymbol{k}_j}{\partial \theta}
$$

由于$\|\tilde{\boldsymbol{k}}_j\| = 1$，第二项提供了自动的梯度缩放，防止梯度过大。

### 18. 长度外推的实证分析

**实验设置**：
- 模型：GAU架构，1亿参数
- 训练长度：512
- 测试长度：4096
- 评价指标：逐token准确率

**结果18.1：KNA显著提升外推效果**

从实验表格可见：
- Baseline在4096（不重复）：23.16%
- KNA在4096（不重复）：47.69%（提升超过2倍）
- CosA在4096（不重复）：46.98%（接近KNA）

**分析18.1：为何"不重复"比"重复"更重要**

"重复"测试集：将512长度的文本重复8次拼接成4096
"不重复"测试集：真实的4096长度文本

"不重复"更能反映真实的长度外推能力，因为：
1. 真实场景中不会有大量重复文本
2. "重复"可能被模型识别出周期性模式
3. "不重复"需要模型真正理解长距离依赖

**分析18.2：KNA与外推技巧的叠加效果**

实验显示：
- Baseline + YaRN：47.45%
- KNA（无修改）：47.69%
- KNA + YaRN：47.44%（几乎无提升）

结论：KNA已经隐式地解决了长度外推问题，进一步叠加YaRN等技巧无显著增益。

### 19. 计算复杂度分析

**定理19.1：KNA的额外计算成本**

Key归一化的操作为：
$$
\tilde{\boldsymbol{k}}_j = \frac{\boldsymbol{k}_j}{\|\boldsymbol{k}_j\|} = \frac{\boldsymbol{k}_j}{\sqrt{\sum_{\ell=1}^d k_{j,\ell}^2}}
$$

**计算步骤**：
1. 计算$\|\boldsymbol{k}_j\|^2 = \sum_{\ell=1}^d k_{j,\ell}^2$：$O(d)$
2. 计算$\|\boldsymbol{k}_j\| = \sqrt{\|\boldsymbol{k}_j\|^2}$：$O(1)$
3. 逐元素除法：$O(d)$

总计：$O(d)$

对于序列长度$n$，总复杂度为$O(nd)$。

**对比19.1：相对于Attention的计算量**

标准Attention的计算复杂度：
- Query-Key点积：$O(n^2 d)$
- Softmax：$O(n^2)$
- Attention-Value加权：$O(n^2 d)$

总计：$O(n^2 d)$

Key归一化的$O(nd)$相对于$O(n^2 d)$可忽略不计（当$n \gg 1$时）。

### 20. 与其他位置编码方案的比较

**对比20.1：ALIBI**

ALIBI通过在Attention分数上加入线性衰减bias：
$$
s_{ALIBI}(j|i) = \boldsymbol{q}_i \cdot \boldsymbol{k}_j - m \cdot (i - j)
$$

其中$m$是衰减率。

缺点：
- 显式的局部注意力偏好，可能损失远程依赖
- 需要为每个头设置不同的$m$
- 不适用于Encoder

**对比20.2：NTK-RoPE**

NTK-RoPE修改RoPE的base：
$$
\theta_i = (10000 \cdot \kappa)^{-2i/d}, \quad \kappa = \left(\frac{L_{test}}{L_{train}}\right)^{d/(d-2)}
$$

缺点：
- 需要修改推理阶段的模型
- 无法保持训练长度内的恒等性
- 外推效果有限（约$L_{test}/2$）

**对比20.3：YaRN**

YaRN对不同频率进行分段处理：
$$
\theta_i^{new} = \left[\gamma_i + (1-\gamma_i)\frac{L_{train}}{L_{test}}\right] \theta_i
$$

优点：
- 效果好于NTK-RoPE
- 理论基础较好（转圈视角）

缺点：
- 仍需修改推理模型
- 实现相对复杂

**对比20.4：KNA的优势总结**

| 方案 | 训练修改 | 推理修改 | 保持恒等性 | 远程依赖 | 实现复杂度 |
|------|----------|----------|------------|----------|------------|
| ALIBI | 是 | 否 | 否 | 损失 | 低 |
| NTK-RoPE | 否 | 是 | 否 | 保留 | 低 |
| YaRN | 否 | 是 | 否 | 保留 | 中 |
| KNA | 是 | 否 | 是 | 保留 | 低 |

KNA的独特优势：
1. 推理时完全不修改（保持恒等性）
2. 不引入局部偏好（保留远程依赖）
3. 实现简单（单个归一化操作）
4. 训练效果还略有提升

### 21. Key归一化的几何解释

**几何视角21.1：单位球面上的点**

Key归一化将所有Key向量投影到单位球面：
$$
\mathcal{S}^{d-1} = \{\boldsymbol{k} \in \mathbb{R}^d : \|\boldsymbol{k}\| = 1\}
$$

注意力分数变为：
$$
s_{KNA}(j|i) = \|\boldsymbol{q}_i\| \cdot (\hat{\boldsymbol{q}}_i \cdot \hat{\boldsymbol{k}}_j)
$$

其中$\hat{\boldsymbol{q}}_i = \boldsymbol{q}_i / \|\boldsymbol{q}_i\|$也在单位球面上。

**几何意义**：注意力分数正比于两个单位向量的内积，即它们在球面上的距离（测地线距离）。

**推导21.1：球面距离与余弦相似度**

单位球面上两点$\hat{\boldsymbol{q}}, \hat{\boldsymbol{k}}$的测地线距离为：
$$
d(\hat{\boldsymbol{q}}, \hat{\boldsymbol{k}}) = \arccos(\hat{\boldsymbol{q}} \cdot \hat{\boldsymbol{k}})
$$

Taylor展开：
$$
\hat{\boldsymbol{q}} \cdot \hat{\boldsymbol{k}} = \cos d = 1 - \frac{d^2}{2} + O(d^4)
$$

当两向量接近时：
$$
\hat{\boldsymbol{q}} \cdot \hat{\boldsymbol{k}} \approx 1 - \frac{d^2}{2}
$$

因此余弦相似度近似反映了球面距离的平方。

**几何视角21.2：RoPE的旋转群作用**

RoPE在球面上施加旋转群$SO(d)$的作用：
$$
\hat{\boldsymbol{k}}_n \mapsto \boldsymbol{\mathcal{R}}_n \hat{\boldsymbol{k}}_n
$$

由于旋转保持内积：
$$
(\boldsymbol{\mathcal{R}}_m \hat{\boldsymbol{q}}) \cdot (\boldsymbol{\mathcal{R}}_n \hat{\boldsymbol{k}}) = \hat{\boldsymbol{q}} \cdot (\boldsymbol{\mathcal{R}}_{n-m} \hat{\boldsymbol{k}})
$$

这在球面上实现了相对位置编码。

### 22. 信息论视角下的Key归一化

**定义22.1：注意力的信息容量**

位置$i$的注意力分布$\{\alpha_{ij}\}$的信息容量（负熵）为：
$$
C_i = \max_{p} I(Y; X) = H(Y) - H(Y|X) = H(Y)
$$

其中$Y$是选择的位置，$X$是给定的Query。

**定理22.1：KNA最大化位置信息**

由于KNA消除了Key模长的影响，所有位置"平等竞争"，位置选择的熵接近最大：
$$
H_i^{KNA} \approx \log i
$$

相比之下，标准Attention可能出现某些位置因模长过大而"垄断"注意力，降低熵。

**推导22.1：互信息的分解**

注意力分布与位置的互信息：
$$
I(\alpha; \Delta) = H(\alpha) - H(\alpha | \Delta)
$$

其中$\Delta = n - m$是相对位置。

KNA强制$\|\boldsymbol{k}_j\| = 1$，使得$H(\alpha | \Delta)$最小化（给定相对位置，注意力确定性最强），从而最大化互信息$I(\alpha; \Delta)$。

这意味着KNA让注意力分布与相对位置的关系最紧密，有助于学习位置模式。

### 23. 训练动态与收敛性

**定义23.1：训练损失的Lipschitz平滑性**

如果损失函数$\mathcal{L}(\theta)$满足：
$$
\|\nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2)\| \leq L \|\theta_1 - \theta_2\|
$$

则称$\mathcal{L}$是$L$-Lipschitz平滑的。

**定理23.1：KNA提升训练平滑性**

由于KNA限制了注意力分数的范围，损失函数的Lipschitz常数$L$减小，训练更平滑。

**推导23.1：注意力分数对参数的二阶导数**

$$
\frac{\partial^2 s_{KNA}}{\partial \theta^2} = \frac{\partial}{\partial \theta} \left(\frac{\partial s_{KNA}}{\partial \theta}\right)
$$

由于$s_{KNA}$有界，其二阶导数也有界，保证了损失函数的平滑性。

**定理23.2：KNA的收敛保证**

在平滑损失下，梯度下降的收敛率为：
$$
\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \leq \frac{L \|\theta_0 - \theta^*\|^2}{2T}
$$

其中$T$是迭代次数，$\theta^*$是最优解。

KNA的较小Lipschitz常数$L$意味着更快的收敛速度。

### 24. 泛化理论

**定义24.1：泛化误差**

$$
\mathcal{E}_{gen} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{test}}[\mathcal{L}(f(x), y)] - \mathbb{E}_{(x,y) \sim \mathcal{D}_{train}}[\mathcal{L}(f(x), y)]
$$

**定理24.1：KNA减小Rademacher复杂度**

模型的Rademacher复杂度为：
$$
\mathcal{R}(\mathcal{F}) = \mathbb{E}_{\sigma} \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \sigma_i f(x_i)
$$

KNA通过限制Key的范数，减小了函数类$\mathcal{F}$的复杂度，从而减小$\mathcal{R}(\mathcal{F})$。

**推导24.1：泛化界**

根据Rademacher复杂度理论，泛化误差有界：
$$
\mathcal{E}_{gen} \leq 2\mathcal{R}(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}
$$

其中$n$是训练样本数，$\delta$是置信度。

KNA的较小$\mathcal{R}(\mathcal{F})$导致更小的泛化界。

### 25. 位置编码的傅里叶分析

**定理25.1：RoPE的频谱特性**

RoPE可以看作傅里叶变换的一种形式：
$$
e^{i n \theta_k} = \cos(n\theta_k) + i\sin(n\theta_k)
$$

不同的$\theta_k = 10000^{-2k/d}$对应不同的频率。

**推导25.1：高频与低频的位置分辨率**

对于$\theta_k = 10000^{-2k/d}$：
- 当$k \to 0$（高频）：$\theta_k \to 1$，周期$T_k = 2\pi / \theta_k \approx 6.28$
- 当$k \to d/2$（低频）：$\theta_k \to 10000^{-1}$，周期$T_k \approx 62800$

高频成分能够分辨短距离的相对位置，低频成分能够分辨长距离的相对位置。

**推导25.2：KNA与频谱的交互**

KNA不改变RoPE的频谱结构，只是消除了Key模长的影响。因此：
$$
s_{KNA}(n-m) = \|\boldsymbol{q}_m\| \text{Re}\left[\sum_{k=0}^{d/2-1} q_k^* k_k e^{i(n-m)\theta_k}\right]
$$

其中$q_k, k_k$是Query和Key的频域系数（归一化后）。

这保留了RoPE的多尺度位置感知能力。

### 26. 实践建议与超参数设置

**建议26.1：Key归一化的实现**

```python
# 伪代码
k_norm = k / torch.norm(k, dim=-1, keepdim=True)
scores = torch.matmul(q, k_norm.transpose(-2, -1))
```

注意：
1. 归一化在最后一维（head_dim）进行
2. 使用`keepdim=True`保持维度一致
3. 数值稳定性：可以加上小常数$\epsilon$防止除零

**建议26.2：是否同时归一化Query**

实验表明：
- 只归一化Key（KNA）：最佳长度外推效果
- 同时归一化Query和Key（CosA）：效果接近KNA，但需要调整温度参数
- 只归一化Query（QNA）：无长度外推效果

推荐：
- 如果不想调温度参数，使用KNA
- 如果能精细调参，CosA可能略优

**建议26.3：与RoPE的结合**

KNA必须与RoPE结合使用才能体现长度外推效果：
- KNA + RoPE：显著外推效果
- KNA + NoPE：无外推效果
- KNA + ALIBI：未充分测试，理论上有冲突

### 27. 注意力矩阵的秩分析

**定义27.1：注意力矩阵**

$$
\boldsymbol{A} \in \mathbb{R}^{n \times n}, \quad A_{ij} = \alpha_{ij}
$$

**定理27.1：KNA保持注意力矩阵的秩**

标准Attention中，如果某些Key的模长过大，可能导致注意力矩阵退化为低秩：
$$
\text{rank}(\boldsymbol{A}) \ll n
$$

KNA通过归一化保持各位置的竞争力，维持较高的秩：
$$
\text{rank}(\boldsymbol{A}_{KNA}) \approx n
$$

**推导27.1：低秩退化的机制**

如果$\|\boldsymbol{k}_{j^*}\| \gg \|\boldsymbol{k}_j\|$ for $j \neq j^*$，则：
$$
\alpha_{i j^*} \approx 1, \quad \alpha_{ij} \approx 0, \quad \forall i
$$

注意力矩阵近似为：
$$
\boldsymbol{A} \approx \boldsymbol{1} \boldsymbol{e}_{j^*}^T
$$

这是秩1矩阵。

KNA消除了这种退化的可能性。

### 28. 长度外推的必要条件

**定理28.1：成功长度外推的三要素**

1. **相对位置编码**：必须使用RoPE等相对位置编码，绝对位置编码无法外推
2. **充分的角度训练**：$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$必须在训练中充分优化
3. **数值稳定性**：注意力分数不能过大，避免softmax饱和

**推导28.1：为何绝对位置编码无法外推**

绝对位置编码：
$$
\boldsymbol{q}_m = \boldsymbol{W}_Q (\boldsymbol{x}_m + \boldsymbol{p}_m)
$$

其中$\boldsymbol{p}_m$是位置$m$的嵌入。

当$m > L_{train}$时，$\boldsymbol{p}_m$从未见过，模型无法泛化。

相对位置编码：
$$
\boldsymbol{q}_m^T \boldsymbol{k}_n = f(m - n)
$$

只要相对位置$m - n$的模式在训练中学到，就能外推。

**推导28.2：KNA满足所有三要素**

1. ✓ 与RoPE配合，保留相对位置编码
2. ✓ 强制模型充分训练$\cos$
3. ✓ 归一化保证数值稳定性

因此KNA能够成功实现长度外推。

### 29. 与Transformer架构变体的兼容性

**兼容性29.1：GAU (Gated Attention Unit)**

GAU使用单头注意力：
$$
\boldsymbol{o} = (\boldsymbol{Z} \odot \text{Attention}(\boldsymbol{U})) \boldsymbol{V}
$$

KNA可以直接应用于GAU的注意力部分，实验表明效果显著。

**兼容性29.2：Multi-Query Attention (MQA)**

MQA中所有Query头共享同一组Key和Value：
$$
\boldsymbol{K} = \boldsymbol{W}_K \boldsymbol{X}, \quad \text{所有头共享}
$$

Key归一化仍然有效：
$$
\tilde{\boldsymbol{K}} = \boldsymbol{K} / \|\boldsymbol{K}\|_{\text{row}}
$$

**兼容性29.3：Grouped-Query Attention (GQA)**

GQA是MQA的泛化，多个Query头共享一组Key-Value。KNA同样适用。

### 30. 总结与展望

**总结30.1：Key归一化的核心洞察**

Key归一化通过一个简单的操作——将Key向量归一化到单位球面——实现了显著的长度外推效果。其核心机制是：

1. **消除捷径**：去除Key模长这一"简单路径"，强制模型优化更本质的角度信息
2. **充分训练**：使$\cos(\boldsymbol{q}_i, \boldsymbol{k}_j)$得到充分训练，增强泛化能力
3. **数值稳定**：有界的注意力分数保证训练和推理的稳定性
4. **保持恒等**：推理时无需修改，在训练长度内保持原始性能

**展望30.2：未来研究方向**

1. **理论完善**：进一步建立Key归一化与长度外推的严格理论联系
2. **大规模验证**：在更大规模的LLM上验证效果（数十亿、千亿参数）
3. **多模态扩展**：将KNA应用于视觉、音频等多模态场景
4. **优化算法**：设计针对KNA的专用优化算法
5. **硬件加速**：开发针对归一化操作的高效硬件实现

**理论意义**：Key归一化揭示了Attention机制中模长与方向的不同作用，为理解和改进Transformer提供了新视角。

**实践价值**：作为一个简单、有效、易于实现的方法，KNA为长文本建模提供了新的工具，特别适合资源有限但需要长度外推的场景。

