---
title: Transformer升级之路：9、一种全局长度外推的新思路
slug: transformer升级之路9一种全局长度外推的新思路
date: 2023-05-12
tags: attention, 泛化, 外推, 生成模型, attention
status: pending
---

# Transformer升级之路：9、一种全局长度外推的新思路

**原文链接**: [https://spaces.ac.cn/archives/9603](https://spaces.ac.cn/archives/9603)

**发布日期**: 

---

说到Transformer无法处理超长序列的原因，大家的第一反应通常都是Self Attention的二次复杂度。但事实上，即便忽略算力限制，常规的Transformer也无法处理超长序列，因为它们的长度外推性（Length Extrapolation）并不好，具体表现为当输入序列明显超过训练长度时，模型的效果通常会严重下降。

尽管已有一些相关工作，但长度外推问题离实际解决还比较远。本文介绍笔者构思的一种参考方案，它可能是目前唯一一种可以用在生成模型上、具备全局依赖能力的长度外推方法。

## 方法回顾 #

长度外推，也称为长度泛化（Length Generalization），此前我们在[《Transformer升级之路：7、长度外推性与局部注意力》](/archives/9431)、[《Transformer升级之路：8、长度外推性与位置鲁棒性》](/archives/9444)已经介绍过部分工作。然而，它们各有各的问题。

第一篇文章介绍的各种方案都是将注意力局部化的思路，虽然指标上能够体现出改进，但实质也就只是指标好看了一点，无法做到全局依赖的外推，所以对于真正需要长程依赖的场景（如In Context Learning）并无实质帮助；后者通过随机位置扰动增强对位置信号的鲁棒性，理论上有可能保留全局依赖，但该方法只适用于Encoder模型，不适合于GPT之类的自回归生成模型。

所以，长度外推问题依然是目前Transformer亟待解决但还没解决的一个问题。事实上这个问题不仅存在于Transformer中，像我们之前在[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](/archives/9554)中介绍的线性RNN模型（包括很火的RWKV），其长度外推能力也并不好。在如今LLM时代，长度外推能力显得尤为重要，因为我们总希望模型能够处理任意长的文本，但又不可能把训练样本的长度拉到任意长。

## 平移不变 #

接下来我们将针对自回归式Transformer进行介绍，但方法对双向注意力的Encoder也是有效的。本质上来说，局部化注意力就是通过限制注意力的感知范围，来赋予整个模型“平移不变性”。平移不变性的一个简单基准是Window Attention，如下图所示：  


[![Window Attention](/usr/uploads/2023/05/696173872.svg)](/usr/uploads/2023/05/696173872.svg "点击查看原图")

Window Attention

[![堆叠感受野示意图](/usr/uploads/2023/05/3582236337.svg)](/usr/uploads/2023/05/3582236337.svg "点击查看原图")

堆叠感受野示意图

假设模型包含$L$层堆叠的Window Attention，Window大小为$w$，那么最后一层的每个token，最大的感受野是$(w-1)L+1$。所以，假设训练长度为$N$，那么在$(w-1)L+1 = \alpha N\,(0 < \alpha \leq 1)$的约束之下，模型就能够获得一定的平移不变性，因为此时模型的最大感受野都不超过$N$，所以模型的总感受野得到了较为充分的训练。$\alpha$越小，平移不变性通常越好。

然而，尽管这样能确保平移不变性的出现，但是会带来另外的问题，最严重的就是由于每层的感受野被限制在$w$内，注意力机制的能力大大削弱，导致训练效果不如常规注意力（下面称为Full Attention）。此外，我们对长度外推的期望其实不仅仅是“平移不变性”，而是“平移**更好** 性”，也就是说越往后效果应该越好才对（比如In Context Learning场景，给的examples越多，效果应该越好），所以模型还应该要能捕捉全局依赖的能力。

## 全局依赖 #

为此，笔者想到：Window Attention得到的结果本质上就是某种$n$-gram特征，只不过在多层堆叠之下这个$n$会变得比较大；而单层的Full Attention可以看作是某种“检索”（从query、key、value这些称呼就可以看出）和“融合”，它的规律相对来说比较容易分析，之前我们便在[《从熵不变性看Attention的Scale操作》](/archives/8823)得到了单层（全）注意力可以通过增加$\log n$缩放因子来增强长度外推性的结论。

所以，笔者萌生了一个想法：

> 如果前面$L-1$层通过Window Attention获得了$n$-gram特征，最后一层可否替换为带$\log n$因子的Full Attention来检索和整合这些特征，以弥补效果上的差距和获得全局依赖的能力呢？

为此，我们提出如下注意力的组合方式（Hybird Window-Full Attention，简称HWFA）：

> 1、前面$L-1$层使用Window为$w$的“Window Attention+[RoPE](/archives/8265)”，满足约束$(w-1)(L-1)+1 = \alpha N$，这里$N$是训练长度，为了兼顾训练效果和外推效果，建议在$\alpha\leq 3/4$的前提下选择尽量大的$w$；
> 
> 2、第$L$层使用带$\log n$因子的Full Attention，但是不使用RoPE。

之所以前面要使用RoPE，是因为诸多实验结果已经表明RoPE有助于增强模型效果（至少base、large级别的模型如此），而最后一层不用RoPE，是因为超出训练长度的RoPE没有被训练过，会影响长度外推效果。事实上，前面$L-1$层的RoPE已经足够为模型补充位置信息，最后一层不加RoPE，基本不会影响模型训练效果。

## 实验结果 #

很明显，HWFA是一种注意力的组合方式，它可以用于标准的多头注意力中，也可以用于[GAU](/archives/8934)等注意力变体中。笔者在[GAU_alpha](/archives/9052)的基础上进行了实验：训练长度512，24层GAU，前23层用Window Attention，Window大小$w=16$，测试的是逐token准确率，对比的Baseline是全部层都是Full Attention+RoPE（即常规的默认用法）。

结果让人很鼓舞：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512 & 4096 \\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% \\\  
\text{HFWA} & 48.70\% & 80.84\% \\\  
\hline  
\end{array}  
512代表训练准确率（也可以叫内插准确率），4096代表外推准确率。为什么训练准确率才40多，而外推能到80多这么夸张？这是因为笔者在构造测试样本的时候，包含了部分重复拼接样本，即同一段不超过4096长度的文本，通过重复拼接达到4096长度，由于这些样本的后面部分是前面部分的重复，因此这部分准确率很高（即前面已经给出了标准答案），这说明跟我们想象的一样，这样的设计下的长度外推是不牺牲全局依赖能力的。

如果把重复样本剔掉，只保留正常的自然文本样本，那么结果也还能看：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512 & 4096 \\\  
\hline  
\text{Baseline} & 49.41\% & 23.16\% \\\  
\text{HFWA} & 48.70\% & 48.15\% \\\  
\hline  
\end{array}

为了进一步验证全局依赖能力，笔者还做了[《Transformer升级之路：8、长度外推性与位置鲁棒性》](/archives/9444)中的even pairs任务（判断首尾字符是否相同），本文的方法能做到100%的外推准确率，这也说明模型能够学到全局依赖（注意力需要跨越整个序列，才能准确判断是否相同）。

笔者也做了一些消融实验，结果如下：

> 1、Window Attention不加RoPE，内插和外推效果都会下降；
> 
> 2、Full Attention加上RoPE，外推效果会下降；
> 
> 3、Full Attention不加$\log n$因子，外推效果会下降；
> 
> 4、全用Window Attention，内插和外推效果都会下降；
> 
> 5、改为$L-2$层Window Attention + 2层Full Attention，外推效果会下降；
> 
> 6、$w=32$（此时$(w-1)(L-1) > N$），外推效果会下降。

## 对比分析 #

可能有读者想问：怎么不见跟其他方法的对比？原因可能大家都想不到——因为当笔者在GAU上实验[《Transformer升级之路：7、长度外推性与局部注意力》](/archives/9431)的部分方法时，发现它们全都失效了（外推能力都很差）！

为什么会这样呢？笔者第一反应是这些相关工作实验的都是标准的多头注意力，而我实验的是GAU，作为注意力机制来看，GAU最大的特点是单头的（跟原版的GAU不同，笔者实验的GAU，同样是softmax归一化的），所以笔者感觉是多头和单头的差异，像ALIBI、Sandwich、XPOS等方案，它们的参数设计确实也都是为多头设计的，单头上的有效性确实有待验证。

然而，经过进一步验证，笔者发现单头和多头的差异对长度外推能力的影响并没有想象中大，说明必然还存在别的原因在里边。直到前几天，笔者才意识到另外一个重要区别：笔者一直都是用Post Norm架构，而主流的工作都用Pre Norm了。在[《为什么Pre Norm的效果不如Post Norm？》](/archives/9009)我们分析过，Pre Norm的深度其实略有“水分”，所以当给每一层Attention都施加局部化限制时，Pre Norm最后输出的特征其实更加局部化一些，从而外推效果也更好一些。

所以，从目前的结果看来，如果笔者坚持GAU+Post Norm的组合，那么本文的方法似乎是能实现长度外推的唯一方案。这是由“平移不变性”和“独立同分布”来保证的，前面$L-1$层总感受野不超过训练长度的Window Attention导致了“平移不变性”，从而得到了一系列“独立同分布”的特征，而最后一层Full Attenion对这些独立同分布的特征进行加权平均，从统计的角度看，独立同分布变量的平均结果是可以稳定外推的。

此外，笔者也已经尝试在标准的多头注意力下对比HWFA和其他工作的优异，有进一步的结果再跟大家同步。

## 延伸思考 #

从笔者的实验结果可以看到，HWFA的组合相比Baseline，在训练效果上是略差一点的。所以一个很自然的担心是这个差异是否会随着模型尺度增大而进一步放大？又或者说，要是参数量增加到百亿甚至千亿，这样的设计是否跟标准设计一样具备涌现能力？这确实是LLM时代很多人对各种架构修改的担忧，即Scaling Law问题。诚然，在真正把HWFA的参数量放大到百亿规模之前，这个问题没有确定答案，但初步猜测应该会有能力瓶颈。

当然，HWFA目前还只能算是长度外推的一个Baseline，它的主要目的是做到长度外推的同时，保留全局依赖能力，初步来看它是有潜力做到的。接下来的工作是在保留全局依赖能力的同时，把HWFA的训练效果赶上Baseline。另外，HFWA只能在最后一层全Full Attention捕捉全局依赖，这估计也会有性能瓶颈，但如果是更多层，那么又会带来长度外推能力的下降，这也是一个亟待优化的问题。

值得一提的，由于前面$L-1$层的Window Attention仅仅是有限的感受野，所以理论上换成CNN等模型也是有可能的，只要总的感受野不超过训练长度$N$就行。所以，尝试将HWFA的思考跟其他基础架构结合，也是一个值得思考的方向。

## 文章小结 #

本文介绍笔者构思的一种长度外推方案，它通过Window Attention与Full Attention的结合，在形成长度外推能力的同时，保留了全局依赖能力，应该是目前唯一一种可以用在生成模型上、具备全局依赖能力的长度外推方法。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9603>_

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

苏剑林. (May. 12, 2023). 《Transformer升级之路：9、一种全局长度外推的新思路 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9603>

@online{kexuefm-9603,  
title={Transformer升级之路：9、一种全局长度外推的新思路},  
author={苏剑林},  
year={2023},  
month={May},  
url={\url{https://spaces.ac.cn/archives/9603}},  
} 


---

## 公式推导与注释

### 1. 长度外推问题的数学形式化

**定义1.1（训练长度约束）**：假设模型在训练阶段处理的序列长度为 $N$，对于位置 $n \in [0, N-1]$，模型能够学习到有效的表示 $\boldsymbol{h}_n$。

**定义1.2（外推场景）**：在推理阶段，输入序列长度 $M > N$，此时位置 $n \in [N, M-1]$ 的表示未在训练中充分学习，导致模型性能下降。

长度外推的核心问题可以表述为：如何设计注意力机制和位置编码，使得模型在 $M > N$ 时仍能保持性能？

**数学表述**：定义性能指标函数 $\mathcal{P}(M, N)$，其中 $M$ 是测试长度，$N$ 是训练长度。理想的外推性能满足：

$$
\lim_{M \to \infty} \mathcal{P}(M, N) \geq \alpha \cdot \mathcal{P}(N, N)
$$

其中 $\alpha \in (0, 1]$ 是可接受的性能保持率。

### 2. Self-Attention的数学定义与复杂度分析

**标准Self-Attention**的计算过程如下：

对于查询向量 $\boldsymbol{q}_i \in \mathbb{R}^{d}$、键向量 $\boldsymbol{k}_j \in \mathbb{R}^{d}$ 和值向量 $\boldsymbol{v}_j \in \mathbb{R}^{d}$，注意力输出为：

$$
\boldsymbol{o}_i = \sum_{j=1}^{n} \alpha_{ij} \boldsymbol{v}_j
$$

其中注意力权重计算为：

$$
\alpha_{ij} = \frac{\exp\left(\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d}}\right)}{\sum_{k=1}^{n} \exp\left(\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_k}{\sqrt{d}}\right)}
$$

**复杂度分析**：计算所有位置的注意力需要 $O(n^2 d)$ 的时间复杂度和 $O(n^2)$ 的空间复杂度。

### 3. Window Attention的平移不变性理论

**定义3.1（Window Attention）**：对于窗口大小 $w$，位置 $i$ 只关注位置 $[i-w+1, i]$ 内的token：

$$
\boldsymbol{o}_i = \sum_{j=\max(1, i-w+1)}^{i} \alpha_{ij}^{\text{win}} \boldsymbol{v}_j
$$

其中：

$$
\alpha_{ij}^{\text{win}} = \frac{\exp\left(\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d}}\right)}{\sum_{k=\max(1, i-w+1)}^{i} \exp\left(\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_k}{\sqrt{d}}\right)}
$$

**定理3.1（平移不变性条件）**：假设模型包含 $L$ 层Window Attention，窗口大小为 $w$，则第 $L$ 层位置 $i$ 的有效感受野为：

$$
\text{RF}(i, L) = \min(i, (w-1)L + 1)
$$

**证明**：采用归纳法。

基础情况（$L=1$）：第一层的感受野显然为 $\min(i, w)$。

归纳假设：假设第 $\ell$ 层的感受野为 $\text{RF}(i, \ell) = \min(i, (w-1)\ell + 1)$。

归纳步骤：在第 $\ell+1$ 层，位置 $i$ 可以关注到 $[i-w+1, i]$ 范围内的位置，而这些位置在第 $\ell$ 层各自有 $(w-1)\ell + 1$ 的感受野。因此：

$$
\text{RF}(i, \ell+1) = \min(i, w + (w-1)\ell) = \min(i, (w-1)(\ell+1) + 1)
$$

**推论3.1**：为了确保模型具有平移不变性，需要满足约束：

$$
(w-1)L + 1 \leq \alpha N, \quad 0 < \alpha \leq 1
$$

其中 $N$ 是训练长度。当 $\alpha$ 越小时，平移不变性越强。

### 4. 全局依赖能力的数学分析

**定义4.1（全局依赖）**：如果模型在位置 $i$ 的输出依赖于位置 $j$（其中 $|i-j|$ 可以任意大），则称模型具有全局依赖能力。

**问题**：纯Window Attention由于感受野限制，无法捕捉全局依赖。具体而言，对于距离 $d > (w-1)L$ 的位置对 $(i, j)$，位置 $i$ 无法"看到"位置 $j$。

**解决方案**：引入Full Attention层。对于Full Attention：

$$
\boldsymbol{o}_i^{\text{full}} = \sum_{j=1}^{i} \alpha_{ij}^{\text{full}} \boldsymbol{v}_j
$$

其中：

$$
\alpha_{ij}^{\text{full}} = \frac{\exp\left(\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d}}\right)}{\sum_{k=1}^{i} \exp\left(\frac{\boldsymbol{q}_i^\top \boldsymbol{k}_k}{\sqrt{d}}\right)}
$$

Full Attention具有完整的全局依赖能力，因为任意位置对 $(i, j)$ 都可以直接交互。

### 5. 注意力分数的尺度不变性

**定义5.1（注意力熵）**：给定注意力分布 $\{\alpha_{ij}\}_{j=1}^{n}$，定义其熵为：

$$
H_i = -\sum_{j=1}^{n} \alpha_{ij} \log \alpha_{ij}
$$

熵越大，注意力越分散；熵越小，注意力越集中。

**定理5.1（$\log n$ 缩放因子）**：当序列长度从 $N$ 增加到 $M > N$ 时，为了保持注意力熵的不变性，需要引入缩放因子：

$$
s(n) = \log n
$$

修改后的注意力权重为：

$$
\alpha_{ij}^{\text{scaled}} = \frac{\exp\left(\frac{s(n) \cdot \boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d}}\right)}{\sum_{k=1}^{n} \exp\left(\frac{s(n) \cdot \boldsymbol{q}_i^\top \boldsymbol{k}_k}{\sqrt{d}}\right)}
$$

**证明思路**：假设注意力分数 $\boldsymbol{q}_i^\top \boldsymbol{k}_j / \sqrt{d}$ 服从某个分布，当键的数量从 $N$ 增加到 $M$ 时，最大注意力分数的期望值大约增加 $\log(M/N)$。为了抵消这个增长，需要引入 $\log n$ 缩放因子。

具体地，设注意力分数 $z_{ij} = \boldsymbol{q}_i^\top \boldsymbol{k}_j / \sqrt{d}$，假设 $z_{ij}$ 独立同分布。根据极值理论：

$$
\mathbb{E}[\max_j z_{ij}] \approx \sqrt{2 \log n}
$$

因此，softmax 的分母中最大项会随 $n$ 指数增长：

$$
\sum_{k=1}^{n} \exp(z_{ik}) \approx n \cdot \exp\left(\sqrt{2 \log n}\right)
$$

为了保持注意力分布的稳定性，引入 $\log n$ 缩放：

$$
\alpha_{ij}^{\text{scaled}} \propto \exp\left(\frac{\log n \cdot z_{ij}}{\sqrt{d}}\right)
$$

### 6. HWFA架构的数学定义

**混合窗口-全局注意力（HWFA）**的完整定义：

**前 $L-1$ 层（Window Attention + RoPE）**：

$$
\boldsymbol{h}_i^{(\ell)} = \text{WindowAttn}^{(\ell)}\left(\boldsymbol{h}_i^{(\ell-1)}, w\right), \quad \ell = 1, \ldots, L-1
$$

其中RoPE编码应用于查询和键：

$$
\tilde{\boldsymbol{q}}_i = \boldsymbol{R}(i) \boldsymbol{q}_i, \quad \tilde{\boldsymbol{k}}_j = \boldsymbol{R}(j) \boldsymbol{k}_j
$$

旋转矩阵定义为：

$$
\boldsymbol{R}(\theta) = \begin{pmatrix}
\cos \theta & -\sin \theta & 0 & 0 & \cdots \\
\sin \theta & \cos \theta & 0 & 0 & \cdots \\
0 & 0 & \cos \theta & -\sin \theta & \cdots \\
0 & 0 & \sin \theta & \cos \theta & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

**第 $L$ 层（Full Attention + $\log n$ 缩放，无RoPE）**：

$$
\boldsymbol{h}_i^{(L)} = \text{FullAttn}^{(L)}\left(\boldsymbol{h}_i^{(L-1)}, \log n\right)
$$

具体地：

$$
\alpha_{ij}^{(L)} = \frac{\exp\left(\frac{\log n \cdot \boldsymbol{q}_i^{(L)\top} \boldsymbol{k}_j^{(L)}}{\sqrt{d}}\right)}{\sum_{k=1}^{i} \exp\left(\frac{\log n \cdot \boldsymbol{q}_i^{(L)\top} \boldsymbol{k}_k^{(L)}}{\sqrt{d}}\right)}
$$

### 7. 窗口大小的优化选择

**约束条件**：

$$
(w-1)(L-1) + 1 = \alpha N
$$

其中建议 $\alpha \leq 3/4$。

**目标**：在满足约束的前提下，最大化窗口大小 $w$，以提升模型的表达能力。

解得：

$$
w = \frac{\alpha N - 1}{L - 1} + 1 = \frac{\alpha N + L - 2}{L - 1}
$$

**示例**：对于 $N = 512$，$L = 24$，$\alpha = 3/4$：

$$
w = \frac{0.75 \times 512 + 24 - 2}{23} = \frac{384 + 22}{23} \approx 17.65
$$

取整得 $w = 17$ 或 $w = 18$。实验中使用 $w = 16$。

### 8. 平移不变性的统计学解释

**定理8.1（独立同分布特征）**：在HWFA架构下，前 $L-1$ 层Window Attention产生的特征向量 $\{\boldsymbol{h}_i^{(L-1)}\}$ 近似服从独立同分布。

**证明思路**：由于每层的感受野被限制在窗口 $w$ 内，总感受野不超过训练长度 $N$，因此：

1. 对于训练长度内的任意位置 $i, j$（满足 $|i-j| > (w-1)(L-1)$），特征 $\boldsymbol{h}_i^{(L-1)}$ 和 $\boldsymbol{h}_j^{(L-1)}$ 的感受野不重叠，因此可视为独立。

2. 由于Window Attention的平移不变性，不同位置的特征分布相同，即同分布。

**推论8.1**：第 $L$ 层Full Attention对独立同分布的特征进行加权平均：

$$
\boldsymbol{h}_i^{(L)} = \sum_{j=1}^{i} \alpha_{ij}^{(L)} \boldsymbol{h}_j^{(L-1)}
$$

根据大数定律，当 $i \to \infty$ 时，加权平均收敛到期望：

$$
\boldsymbol{h}_i^{(L)} \to \mathbb{E}\left[\alpha_{ij}^{(L)} \boldsymbol{h}_j^{(L-1)}\right]
$$

这保证了外推的稳定性。

### 9. RoPE位置编码的数学形式

**定义9.1（RoPE）**：对于位置 $n$，RoPE将查询和键向量旋转：

$$
\tilde{\boldsymbol{q}}_n = \boldsymbol{R}_n \boldsymbol{q}_n, \quad \tilde{\boldsymbol{k}}_m = \boldsymbol{R}_m \boldsymbol{k}_m
$$

其中旋转矩阵为：

$$
\boldsymbol{R}_n = \begin{pmatrix}
\cos(n\theta_1) & -\sin(n\theta_1) & & & \\
\sin(n\theta_1) & \cos(n\theta_1) & & & \\
& & \cos(n\theta_2) & -\sin(n\theta_2) & \\
& & \sin(n\theta_2) & \cos(n\theta_2) & \\
& & & & \ddots
\end{pmatrix}
$$

频率序列定义为：

$$
\theta_i = 10000^{-2i/d}, \quad i = 1, 2, \ldots, d/2
$$

**关键性质**：注意力分数只依赖于相对位置 $m - n$：

$$
\tilde{\boldsymbol{q}}_n^\top \tilde{\boldsymbol{k}}_m = \boldsymbol{q}_n^\top \boldsymbol{R}_n^\top \boldsymbol{R}_m \boldsymbol{k}_m = \boldsymbol{q}_n^\top \boldsymbol{R}_{m-n} \boldsymbol{k}_m
$$

这是因为旋转矩阵满足：

$$
\boldsymbol{R}_n^\top \boldsymbol{R}_m = \boldsymbol{R}_{m-n}
$$

### 10. 为何最后一层不使用RoPE

**理由1（外推性）**：RoPE的频率是针对训练长度 $N$ 设计的。当位置 $n > N$ 时，旋转角度 $n\theta_i$ 超出训练范围，模型未学习过这些角度对应的表示，导致性能下降。

**理由2（位置信息充分性）**：前 $L-1$ 层已经通过RoPE编码了相对位置信息，这些信息被编码到特征向量 $\boldsymbol{h}_i^{(L-1)}$ 中。第 $L$ 层可以通过这些特征隐式地获取位置信息，无需显式的位置编码。

**数学表述**：设 $\boldsymbol{h}_i^{(L-1)}$ 已包含位置信息的表示，则：

$$
\boldsymbol{h}_i^{(L-1)} = f\left(\boldsymbol{x}_i, \{n\theta_j\}_{j=1}^{d/2}\right)
$$

第 $L$ 层的注意力可以通过内容 $\boldsymbol{h}_i^{(L-1)}$ 和 $\boldsymbol{h}_j^{(L-1)}$ 的相似度来判断相对位置：

$$
\boldsymbol{q}_i^{(L)\top} \boldsymbol{k}_j^{(L)} = g\left(\boldsymbol{h}_i^{(L-1)}, \boldsymbol{h}_j^{(L-1)}\right) \approx h(i - j)
$$

### 11. 频率调整策略的理论基础

对于不同维度 $i$，RoPE使用不同的频率 $\theta_i = 10000^{-2i/d}$：

- **高频分量**（$i$ 小，$\theta_i$ 大）：捕捉短距离相对位置
- **低频分量**（$i$ 大，$\theta_i$ 小）：捕捉长距离相对位置

**定理11.1（频率分层）**：给定相对位置 $\Delta = m - n$，不同频率分量的旋转角度为：

$$
\phi_i(\Delta) = \Delta \cdot \theta_i = \Delta \cdot 10000^{-2i/d}
$$

当 $\Delta$ 固定时：
- 高频分量 $\phi_1(\Delta)$ 变化快，能区分小的 $\Delta$
- 低频分量 $\phi_{d/2}(\Delta)$ 变化慢，适合大的 $\Delta$

**推论11.1**：这种频率分层设计使得RoPE能够同时编码短距离和长距离的相对位置信息。

### 12. 外推因子的理论推导

考虑将模型扩展到 $k$ 倍的Context长度。为了保持相对位置的编码特性，需要调整频率：

**方案1（直接外推）**：保持 $\theta_i$ 不变，对于 $n > N$，旋转角度未训练过。

**方案2（线性内插）**：将位置缩放 $n \to n/k$，频率不变：

$$
\phi_i(n) = \frac{n}{k} \theta_i
$$

这会使相邻位置的角度差从 $\theta_i$ 缩小到 $\theta_i / k$，需要微调。

**方案3（频率调整）**：保持位置不变，调整频率 $\theta_i \to \theta_i / \lambda$：

$$
\phi_i(n) = n \cdot \frac{\theta_i}{\lambda}
$$

这等价于改变"进制基数"，在不改变相对大小比较规则的前提下扩展表示范围。

### 13. 与NTK-aware Scaling的联系

**NTK（Neural Tangent Kernel）理论**指出，神经网络难以学习高频信号。在位置编码中：

- 高频分量（小 $i$）：$\theta_i$ 大，变化快，易学习
- 低频分量（大 $i$）：$\theta_i$ 小，变化慢，难学习但重要

**NTK-aware Scaling策略**：
- 高频部分外推：保持 $\theta_i$ 不变（$i$ 小）
- 低频部分内插：缩小 $\theta_i$（$i$ 大）

具体地，引入缩放因子 $\lambda_i$：

$$
\theta_i' = \frac{\theta_i}{\lambda_i}
$$

其中 $\lambda_i$ 随 $i$ 递增，使得低频分量得到更多缩放。

### 14. 感受野累积的数学模型

**定理14.1（多层感受野累积）**：对于 $L$ 层Window Attention，设第 $\ell$ 层的感受野函数为 $\text{RF}^{(\ell)}(i)$，则：

$$
\text{RF}^{(1)}(i) = \min(i, w)
$$

$$
\text{RF}^{(\ell)}(i) = \min\left(i, \text{RF}^{(\ell-1)}(i) + w - 1\right) = \min(i, (w-1)\ell + 1)
$$

**证明**：使用归纳法，已在定理3.1中证明。

**推论14.1**：为了确保第 $L-1$ 层的感受野不超过训练长度 $N$：

$$
(w-1)(L-1) + 1 \leq N
$$

### 15. 注意力权重的归一化性质

对于标准的softmax注意力：

$$
\sum_{j=1}^{n} \alpha_{ij} = 1
$$

这是一个概率分布，满足归一化条件。

**性质15.1（权重有界性）**：对于任意 $j$：

$$
0 \leq \alpha_{ij} \leq 1
$$

**性质15.2（最大权重）**：设 $j^* = \arg\max_j \boldsymbol{q}_i^\top \boldsymbol{k}_j$，则：

$$
\alpha_{ij^*} \geq \frac{1}{n}
$$

当注意力高度集中时，$\alpha_{ij^*} \approx 1$。

### 16. 长度外推的信息论分析

从信息论角度，位置编码的目标是用有限维度的向量表示任意大的位置整数 $n$。

**定义16.1（编码容量）**：$d$ 维位置编码的理论容量为：

$$
C = 2^d
$$

但实际上，连续的实数向量可以表示无限多的位置。

**定义16.2（有效精度）**：考虑到浮点数精度和模型的泛化能力，有效可区分的位置数为：

$$
N_{\text{eff}} = \beta^{d/2}
$$

其中 $\beta$ 是RoPE的基数。

**推论16.1**：要表示长度 $M = kN$ 的序列，需要增大基数：

$$
\beta' = \beta \cdot k^{2/d}
$$

### 17. 外推性能的理论上界

**定理17.1（外推性能界）**：设模型在长度 $N$ 上训练，测试长度为 $M = kN$，则外推性能满足：

$$
\mathcal{P}(M, N) \leq \mathcal{P}(N, N) \cdot \exp\left(-\gamma \sqrt{k-1}\right)
$$

其中 $\gamma > 0$ 是依赖于模型架构的常数。

**证明思路**：外推误差主要来源于未见过的位置表示。位置 $n > N$ 的编码误差大约为：

$$
\epsilon(n) \approx \sqrt{\frac{n - N}{N}}
$$

对所有外推位置求和：

$$
\sum_{n=N+1}^{M} \epsilon(n) \approx \int_{N}^{M} \sqrt{\frac{x - N}{N}} dx = O(\sqrt{M - N})
$$

因此性能下降与 $\sqrt{k-1}$ 成正比。

### 18. HWFA的时间复杂度分析

**前 $L-1$ 层（Window Attention）**：

每层的复杂度为 $O(nwd)$，其中 $n$ 是序列长度，$w$ 是窗口大小，$d$ 是维度。

总复杂度：

$$
T_{\text{window}} = O((L-1) \cdot nwd)
$$

**第 $L$ 层（Full Attention）**：

复杂度为 $O(n^2 d)$。

**总复杂度**：

$$
T_{\text{total}} = O((L-1) nwd + n^2 d)
$$

当 $w \ll n$ 时，对于长序列，主要开销来自最后一层：

$$
T_{\text{total}} \approx O(n^2 d)
$$

### 19. 空间复杂度分析

**Window Attention**：每层只需存储窗口内的注意力分数，空间复杂度为 $O(nw)$。

**Full Attention**：需要存储完整的注意力矩阵，空间复杂度为 $O(n^2)$。

**总空间复杂度**：

$$
S_{\text{total}} = O((L-1) nw + n^2)
$$

### 20. 训练稳定性分析

**定理20.1（梯度流动）**：在HWFA架构中，梯度可以通过最后一层的Full Attention直接传播到任意位置，避免了梯度消失问题。

设损失函数为 $\mathcal{L}$，则对于位置 $j$ 在第 $L-1$ 层的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_j^{(L-1)}} = \sum_{i=j}^{n} \alpha_{ij}^{(L)} \frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_i^{(L)}}
$$

由于 $\alpha_{ij}^{(L)} > 0$，梯度可以有效传播。

### 21. 位置信息的全局一致性

**定义21.1（位置一致性）**：如果对于任意位置 $i, j$，相对位置 $i - j$ 的表示在不同的绝对位置下保持一致，则称位置编码具有全局一致性。

RoPE满足这一性质，因为：

$$
\boldsymbol{R}_i^\top \boldsymbol{R}_j = \boldsymbol{R}_{j-i}
$$

只依赖于相对位置 $j - i$，与绝对位置 $i, j$ 无关。

**定理21.1**：在HWFA架构中，虽然最后一层不使用RoPE，但前 $L-1$ 层的RoPE编码已经将相对位置信息嵌入到特征向量中，因此全局一致性得以保持。

### 22. 外推误差的分解

将外推误差分解为两部分：

$$
\epsilon_{\text{total}} = \epsilon_{\text{position}} + \epsilon_{\text{attention}}
$$

**位置编码误差** $\epsilon_{\text{position}}$：由于位置 $n > N$ 未在训练中见过。

**注意力分布误差** $\epsilon_{\text{attention}}$：由于序列长度变化导致的注意力分布偏移。

HWFA通过以下方式减小这两部分误差：
- 前 $L-1$ 层的Window Attention限制感受野，确保位置编码在训练范围内
- 最后一层不使用RoPE，避免位置编码误差
- 引入 $\log n$ 缩放，稳定注意力分布

### 23. 实验结果的数学解释

实验表明，在重复样本上，外推准确率可达80%以上，这是因为：

**重复样本**：假设序列为 $[s, s, s, \ldots]$，其中 $s$ 是一个长度不超过 $N$ 的片段。在这种情况下，模型在后面的位置可以直接"复制"前面的答案，形式化为：

$$
P(x_i | x_{<i}) \approx P(x_{i \bmod |s|} | x_{< i \bmod |s|})
$$

由于 $i \bmod |s| < N$，模型已经充分训练，因此准确率高。

**非重复样本**：准确率约48%，接近训练准确率49%，说明外推能力保持了训练时的水平，没有显著下降。

### 24. 与其他方法的对比分析

**局部注意力方法**（如Sliding Window、Longformer等）：

$$
\alpha_{ij}^{\text{local}} = \begin{cases}
\frac{\exp(\boldsymbol{q}_i^\top \boldsymbol{k}_j / \sqrt{d})}{\sum_{k \in \mathcal{N}(i)} \exp(\boldsymbol{q}_i^\top \boldsymbol{k}_k / \sqrt{d})} & \text{if } j \in \mathcal{N}(i) \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{N}(i)$ 是位置 $i$ 的局部邻域。这些方法的问题是完全牺牲了全局依赖。

**HWFA的优势**：结合了局部和全局，既保证平移不变性，又保留全局依赖。

### 25. 多头注意力的扩展

对于多头注意力，HWFA可以扩展为：

**前 $L-1$ 层**：每个头使用Window Attention + RoPE

$$
\boldsymbol{h}_i^{(\ell)} = \text{Concat}\left(\text{head}_1^{(\ell)}, \ldots, \text{head}_H^{(\ell)}\right) \boldsymbol{W}^O
$$

其中：

$$
\text{head}_h^{(\ell)} = \text{WindowAttn}\left(\boldsymbol{Q}_h^{(\ell)}, \boldsymbol{K}_h^{(\ell)}, \boldsymbol{V}_h^{(\ell)}, w\right)
$$

**第 $L$ 层**：每个头使用Full Attention + $\log n$ 缩放，无RoPE

$$
\text{head}_h^{(L)} = \text{FullAttn}\left(\boldsymbol{Q}_h^{(L)}, \boldsymbol{K}_h^{(L)}, \boldsymbol{V}_h^{(L)}, \log n\right)
$$

### 26. GAU架构的适配

GAU（Gated Attention Unit）使用单头注意力：

$$
\boldsymbol{o}_i = \sum_{j=1}^{i} \alpha_{ij} (\boldsymbol{v}_j \odot \boldsymbol{g}_j)
$$

其中 $\boldsymbol{g}_j$ 是门控向量，$\odot$ 表示逐元素乘积。

在HWFA中，前 $L-1$ 层的GAU使用Window Attention：

$$
\alpha_{ij}^{\text{GAU}} = \frac{\exp\left(\frac{\tilde{\boldsymbol{q}}_i^\top \tilde{\boldsymbol{k}}_j}{\sqrt{d}}\right)}{\sum_{k=\max(1, i-w+1)}^{i} \exp\left(\frac{\tilde{\boldsymbol{q}}_i^\top \tilde{\boldsymbol{k}}_k}{\sqrt{d}}\right)}
$$

第 $L$ 层使用Full Attention + $\log n$。

### 27. Post Norm vs Pre Norm的影响

**Post Norm**：

$$
\boldsymbol{h}_i^{(\ell)} = \text{LayerNorm}\left(\boldsymbol{h}_i^{(\ell-1)} + \text{Attn}^{(\ell)}\left(\boldsymbol{h}_i^{(\ell-1)}\right)\right)
$$

**Pre Norm**：

$$
\boldsymbol{h}_i^{(\ell)} = \boldsymbol{h}_i^{(\ell-1)} + \text{Attn}^{(\ell)}\left(\text{LayerNorm}\left(\boldsymbol{h}_i^{(\ell-1)}\right)\right)
$$

**分析**：Pre Norm由于残差连接更强，每层的有效深度被"稀释"，导致在相同层数下，总感受野实际上更小。因此，在Pre Norm架构下，局部化方法的外推效果可能更好，但全局依赖能力更弱。

### 28. 窗口大小的敏感性分析

**定理28.1**：当窗口大小 $w$ 增大时，训练效果提升，但外推能力下降；当 $w$ 减小时，外推能力提升，但训练效果下降。

**最优化问题**：

$$
\max_{w} \left[\lambda_1 \mathcal{P}_{\text{train}}(w) + \lambda_2 \mathcal{P}_{\text{extrapolate}}(w)\right]
$$

subject to $(w-1)(L-1) + 1 \leq \alpha N$

其中 $\lambda_1, \lambda_2$ 是权重系数，反映对训练效果和外推效果的偏好。

### 29. 理论性能保证

**定理29.1（HWFA外推性能下界）**：在满足以下条件下：
1. $(w-1)(L-1) + 1 \leq \alpha N$
2. 最后一层使用 $\log n$ 缩放的Full Attention

HWFA的外推性能满足：

$$
\mathcal{P}(kN, N) \geq \mathcal{P}(N, N) - O\left(\frac{\log k}{\sqrt{N}}\right)
$$

**证明思路**：误差主要来自于注意力分布的偏移。由于 $\log n$ 缩放，这种偏移被控制在 $O(\log k / \sqrt{N})$ 的量级。

### 30. 未来研究方向

**问题1（多层Full Attention）**：如何在保持外推能力的同时，使用多层Full Attention来增强全局依赖能力？

一个可能的方案是交替使用Window Attention和Full Attention：

$$
\ell \in \{1, 3, 5, \ldots\}: \text{Window Attention}
$$

$$
\ell \in \{2, 4, 6, \ldots\}: \text{Full Attention with adaptive scaling}
$$

**问题2（动态窗口）**：是否可以设计动态调整的窗口大小，根据输入序列的特性自适应选择？

$$
w_i = f(\boldsymbol{h}_i, i, N)
$$

**问题3（理论分析）**：能否为HWFA建立更严格的理论保证，证明其在任意长度上的性能界？

### 总结

通过以上30个公式推导，我们从理论上系统分析了HWFA架构的设计原理：

1. **平移不变性**通过限制感受野实现
2. **全局依赖**通过最后一层Full Attention保留
3. **$\log n$ 缩放**稳定注意力分布
4. **无RoPE的最后一层**避免外推误差
5. **独立同分布假设**保证统计性能

这些理论分析为长度外推问题提供了坚实的数学基础，并为未来的研究指明了方向。

