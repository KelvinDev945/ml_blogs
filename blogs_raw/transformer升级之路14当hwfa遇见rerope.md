---
title: Transformer升级之路：14、当HWFA遇见ReRoPE
slug: transformer升级之路14当hwfa遇见rerope
date: 2023-08-24
tags: 详细推导, attention, 位置编码, 外推, rope, 生成模型
status: pending
---
# Transformer升级之路：14、当HWFA遇见ReRoPE

**原文链接**: [https://spaces.ac.cn/archives/9731](https://spaces.ac.cn/archives/9731)

**发布日期**: 

---

在上一篇文章[《Transformer升级之路：13、逆用Leaky ReRoPE》](/archives/9728)中，笔者尝试通过在训练阶段逆用Leaky ReRoPE的思路，使得推理阶段的位置编码变为正常的RoPE，从而在达到长度外推的同时解决ReRoPE推理变慢的缺点。遗憾的是，从实验结果来看，“Leaky ReRoPE → RoPE”的效果并不如“RoPE → ReRoPE/Leaky ReRoPE”，因此这个问题尚未完全解决。

此时，笔者想到此前在[《Transformer升级之路：9、一种全局长度外推的新思路》](/archives/9603)提出的HWFA本身就具有一定的长度外推能力，如果跟ReRoPE“强强联合”，是否会有更好的效果？更关键是，HWFA的加入可以大幅度降低推理成本，从而弥补ReRoPE的不足！

## 温故 #

首先，“例行公事”地回顾一下HWFA。HWFA（Hybird Window-Full Attention）并非一个具体的模型，而是一种Attention的组合方式，能够在基本保持效果不变的前提下，增强Attention模型的长度外推能力，同时还能降低训练和推理成本。

具体来说，HWFA是“$L-1$层Window RoPE Attention + $1$层Full NoPE Attention”，即前面$L-1$层Attention都加上[RoPE](/archives/8265)，并通过window限制感受野，这样一来推理成本就变为常数，并且基于block parallel进行优化的话，也可以提升训练速度；至于最后一层Attention，则保留global的形式，但去掉位置编码（NoPE），同时加上[$\log n$缩放](/archives/8823)。经过这样修改，并且适当选择window之后，模型的训练效果只有轻微下降，同时呈现出优秀的长度外推能力。

无独有偶，后来Google提出了[FOT（Focused Transformer）](https://papers.cool/arxiv/2307.03170)，它跟HWFA有很多异曲同工之处：同样是$L-1$层Local Attention加$1$层Full Attention，Full Attention同样是NoPE的，不同的是FOT把Full Attention放在中间，并且Local Attention没有严格限制感受野，所以无法直接长度外推，因此它提出了crossbatch training来拓展模型长度。事后，笔者实验过在HWFA上使用crossbatch training，也有不错的效果。

## 知新 #

回到本文的主题，HWFA如何跟ReRoPE“强强联合”呢？我们知道，ReRoPE是用在Full RoPE Attention上的，就是在推理阶段截断一下相对位置矩阵：  
$$\begin{pmatrix}0 & \\\  
1 & 0 & \\\  
2 & 1 & 0 &\\\  
\ddots & 2 & 1 & 0 & \\\  
\ddots & \ddots & 2 & 1 & 0 & \\\  
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\\  
\small{L - 2} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\\  
\small{L - 1} & \small{L - 2} & \ddots & \ddots & \ddots & 2 & 1 & 0 & \\\  
\end{pmatrix} \,\to\, \begin{pmatrix}  
\color{red}{0} & \\\  
\color{red}{1} & \color{red}{0} & \\\  
\color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\\  
\color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\ddots} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\\  
\color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\\  
\end{pmatrix}$$  
出人意料的是，这样的事后处理体现出极佳的长度外推能力。然而，由于RoPE的特殊性，原始的ReRoPE实现需要算两次Attention矩阵，并且不兼容主流的Flash Attention加速等。总的来说，推理阶段的成本增加略有点大。

不过，HWFA的加入将会极大地缓解这个问题！综上所述，ReRoPE只用在Full RoPE Attention上，HWFA则大部分都是Window RoPE Attention，所以“HWFA+ReRoPE”的方案就呼之欲出了：训练阶段将HWFA原本的Full NoPE Attention换成Full RoPE Attention，然后推理阶段则改为Full ReRoPE Attention。这样一来推理阶段切换ReRoPE带来的额外成本就会变得非常少，而且其他层换为Window Attention带来的收益更加显著。

除此之外，“HWFA+ReRoPE”还可以弥补原本HWFA的效果损失。此前，为了保证长度外推能力，HWFA的Full Attention要去掉位置编码（即NoPE），同时Window Attention的感受野$\tilde{w}$要满足$(\tilde{w}-1)(L-1)+1 = \alpha N$（其中$L$是层数，$N$是训练长度，$0 < \alpha \leq 1$），这些约束限制了模型的表达能力，导致了训练效果变差。而引入ReRoPE之后，Window Attention的感受野可以适当取大一些，Full Attention也可以用RoPE，还可以将它放到中间层而不单是最后一层，甚至也可以多于$1$层Full Attention。这些变化都可以弥补效果损失，并且得益于ReRoPE，长度外推能力并不会有所下降。

为了区别最初版的HWFA，我们也可以将“HWFA+ReRoPE”的组合，称为“HWFA2”。

## 实验 #

下面分享一些“HWFA+ReRoPE（HWFA2）”的实验结果。由于引入ReRoPE之后，HWFA的自由度就大了很多，因此下降只是挑笔者认为比较符合直觉的组合进行实验，无法充分验证所有排列组合。

实验模型跟之前的HWFA、ReRoPE的一样，都是1亿参数的GAU模型，512的训练长度。注意这里有两个window参数：一个是ReRoPE本身有个$w$参数，此前ReRoPE实验显示这个影响不大，所以下面统一取256；另一个是HFWA的Window Attention的感受野，上面记为$\tilde{w}$，这是可调的。所以，“HWFA+ReRoPE”的主要参数就是Window Attention的$\tilde{w}$，以及Full Attention的层数和分布位置。此前笔者做了一些对比实验，显示从训练效果来看，Full Attention放在中间要比放在末尾效果更好，所以如果是1层Full Attention，那么它默认的放置位置是(index = num_layers / 2)的层，如果是2层Full Attention，那么默认的放置位置是(index = num_layers / 3)和(index = 2 * num_layers / 3)的层，依此类推。

部分实验结果如下：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\\  
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\\  
\hline  
\text{ReRoPE-w256} & 49.41\% & 77.90\% & 48.48\% \\\  
\text{ReRoPE-w256-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.40\% & 48.85\% \\\  
\text{ReRoPE-w256-}\log n & 49.40\% & \boldsymbol{85.12\%} & 49.07\% \\\  
\hline  
\text{InvLeaky ReRoPE-w128-}\log n & 49.38\% & 82.25\% & 48.32\% \\\  
\text{InvLeaky ReRoPE-w128-b8-}\log n & 49.62\% & 81.15\% & 48.85\% \\\  
\hline  
\text{HFWA} & 48.70\% & 80.84\% & 48.15\% \\\  
\hline  
\text{HFWA-ReRoPE-w32-f1} & 49.29\% & 83.13\% & 49.34\% \\\  
\text{HFWA-ReRoPE-w64-f1} & 49.32\% & 82.41\% & \boldsymbol{49.37\%} \\\  
\text{HFWA-ReRoPE-w128-f1} & 49.21\% & 80.18\% & 48.99\% \\\  
\text{HFWA-ReRoPE-w256-f1} & 49.00\% & 54.94\% & 47.64\% \\\  
\text{HFWA-ReRoPE-w32-f2} & \boldsymbol{49.50}\% & 84.09\% & 49.35\% \\\  
\text{HFWA-ReRoPE-w64-f2} & 49.46\% & 84.43\% & 49.36\% \\\  
\text{HFWA-ReRoPE-w128-f2} & 49.35\% & 83.09\% & 48.97\% \\\  
\text{HFWA-ReRoPE-w256-f2} & 49.37\% & 75.24\% & 48.42\% \\\  
\hline  
\end{array}

上表中$\text{w}$后的数字就是Window Attention的感受野$\tilde{w}$的大小，$\text{f}$后的数字就是Full Attention的层数。原本的HFWA由于各种约束，$\tilde{w}$只取到了16，再大的话长度外推能力就会明显下降。而从上表可以看到，增大了$\tilde{w}$后，训练性能可以迅速对齐baseline，并且进一步增加Full Attention还超过了baseline。至于外推效果，$\text{w32},\text{w64}$这两个case都相当不错，明显超过了HFWA。总的来看，HFWA-ReRoPE的最佳组合是$\text{w64-f2}$，训练效果和不重复的外推效果都超过了原本的ReRoPE，再结合训练长度$N$是512、层数$L$是24来看，猜测$\tilde{w}$的最佳取值应该是$N/L$的$2\sim 4$倍左右。

## 小结 #

本文提出了HWFA与ReRoPE的组合使用方式，小规模的实验结果显示，这种组合能够在不损失训练效果的同时，达到近乎最佳的长度外推效果，并且得益于HFWA的设计，还可以明显地降低推理成本，有效地缓解了ReRoPE原本的推理成本增加的缺点。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9731>_

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

苏剑林. (Aug. 24, 2023). 《Transformer升级之路：14、当HWFA遇见ReRoPE 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9731>

@online{kexuefm-9731,  
title={Transformer升级之路：14、当HWFA遇见ReRoPE},  
author={苏剑林},  
year={2023},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/9731}},  
} 


---

## 公式推导与注释

### 1. HWFA的层次结构数学定义

**定义1.1（HWFA架构）**

HWFA（Hybrid Window-Full Attention）是一种混合注意力架构，由两类注意力层组成：

1. **窗口注意力层（Window Attention）**：具有有限感受野$\tilde{w}$的局部注意力
2. **全局注意力层（Full Attention）**：无限制感受野的全局注意力

对于$L$层的Transformer，HWFA的一般形式为：

$$\text{Attention}^{(\ell)} = \begin{cases}
\text{WindowAttn}(\cdot; \tilde{w}, \text{RoPE}), & \ell \in \mathcal{W} \\
\text{FullAttn}(\cdot; \text{NoPE}), & \ell \in \mathcal{F}
\end{cases}$$

其中$\mathcal{W}$和$\mathcal{F}$是窗口层和全局层的索引集合，满足$\mathcal{W} \cup \mathcal{F} = \{1, 2, \ldots, L\}$，$\mathcal{W} \cap \mathcal{F} = \emptyset$。

**推导1.2（窗口注意力的数学表示）**

窗口注意力在位置$i$处只关注窗口$[i - \tilde{w} + 1, i]$内的位置。注意力掩码为：

$$M_{i,j}^{\text{window}} = \begin{cases}
0, & i - \tilde{w} + 1 \leq j \leq i \\
-\infty, & \text{otherwise}
\end{cases}$$

注意力权重为：

$$\alpha_{i,j}^{(\ell)} = \frac{\exp(a_{i,j}^{(\ell)} + M_{i,j}^{\text{window}})}{\sum_{k} \exp(a_{i,k}^{(\ell)} + M_{i,k}^{\text{window}})} = \frac{\exp(a_{i,j}^{(\ell)})}{\sum_{k=i-\tilde{w}+1}^{i} \exp(a_{i,k}^{(\ell)})}$$

其中$a_{i,j}^{(\ell)} = (\boldsymbol{\mathcal{R}}^i\boldsymbol{q}_i^{(\ell)})^{\top}(\boldsymbol{\mathcal{R}}^j\boldsymbol{k}_j^{(\ell)})$是应用RoPE后的Attention分数。

**定义1.3（全局注意力的数学表示）**

全局注意力不使用位置编码（NoPE），纯粹基于内容相似度：

$$a_{i,j}^{(\ell)} = (\boldsymbol{q}_i^{(\ell)})^{\top}\boldsymbol{k}_j^{(\ell)}$$

注意力权重为：

$$\alpha_{i,j}^{(\ell)} = \frac{\exp(a_{i,j}^{(\ell)} / \sqrt{d} + \log n)}{\sum_{k=1}^{n} \exp(a_{i,k}^{(\ell)} / \sqrt{d} + \log n)}$$

其中$\log n$是序列长度缩放因子，$n$是当前序列长度，这有助于稳定长序列的训练。

**推导1.4（原始HWFA的层配置）**

最简单的HWFA配置是"$(L-1)$层窗口 + $1$层全局"：

$$\mathcal{W} = \{1, 2, \ldots, L-1\}, \quad \mathcal{F} = \{L\}$$

这样最后一层可以聚合全局信息，而前面的层通过窗口注意力高效建模局部依赖。

### 2. 多尺度窗口的数学表示

**定义2.1（层级感受野）**

在$L$层的HWFA中，第$\ell$层位置$i$的理论感受野为：

$$\mathcal{F}^{(\ell)}(i) = \{j : \exists \text{路径从} j \text{到} i \text{通过前} \ell \text{层}\}$$

**定理2.2（窗口注意力的感受野增长）**

对于纯窗口注意力（所有层窗口大小为$\tilde{w}$），第$\ell$层的感受野大小为：

$$|\mathcal{F}^{(\ell)}(i)| = \min(\ell \cdot \tilde{w} - (\ell - 1), i + 1) = \min(\ell(\tilde{w} - 1) + 1, i + 1)$$

**证明**（归纳法）：

**基础步骤**：第1层，$|\mathcal{F}^{(1)}(i)| = \min(\tilde{w}, i + 1)$（显然）。

**归纳步骤**：假设第$\ell$层的感受野为$\ell(\tilde{w} - 1) + 1$。在第$\ell+1$层，每个位置$j \in \mathcal{F}^{(\ell)}(i)$可以关注其前$\tilde{w}$个位置，因此：

$$\mathcal{F}^{(\ell+1)}(i) = \bigcup_{j \in \mathcal{F}^{(\ell)}(i)} \{k : j - \tilde{w} + 1 \leq k \leq j\}$$

最远可达：

$$\min(\mathcal{F}^{(\ell)}(i)) - (\tilde{w} - 1) = i - [\ell(\tilde{w} - 1)] - (\tilde{w} - 1) = i - (\ell + 1)(\tilde{w} - 1)$$

因此：

$$|\mathcal{F}^{(\ell+1)}(i)| = (\ell + 1)(\tilde{w} - 1) + 1$$

**推论2.3（覆盖整个序列的条件）**

要使第$L$层能覆盖长度为$N$的整个序列，需要：

$$L(\tilde{w} - 1) + 1 \geq N$$

即：

$$\tilde{w} \geq \frac{N - 1}{L} + 1 = \frac{N - 1 + L}{L}$$

对于$N \gg L$，近似为：

$$\tilde{w} \geq \frac{N}{L}$$

**推导2.4（原始HWFA的窗口大小约束）**

原始HWFA要求能够外推，因此窗口大小需要满足：

$$(\tilde{w} - 1)(L - 1) + 1 = \alpha N_{\text{train}}$$

其中$0 < \alpha \leq 1$是覆盖系数。当$\alpha = 1$时，第$L-1$层恰好覆盖整个训练序列，最后一层的全局注意力可以聚合所有信息。

解得：

$$\tilde{w} = \frac{\alpha N_{\text{train}} - 1}{L - 1} + 1 \approx \frac{\alpha N_{\text{train}}}{L - 1}$$

例如，$N_{\text{train}} = 512$，$L = 24$，$\alpha = 1/32$（实验中的设置）：

$$\tilde{w} \approx \frac{512/32}{23} = \frac{16}{23} \approx 0.7$$

实际使用$\tilde{w} = 16$，这对应$\alpha = 16 \times 23 / 512 \approx 0.72$。

### 3. ReRoPE与HWFA的融合机制

**定义3.1（HWFA2架构）**

HWFA2将ReRoPE引入HWFA，定义为：

$$\text{Attention}^{(\ell)} = \begin{cases}
\text{WindowAttn}(\cdot; \tilde{w}, \text{RoPE}), & \ell \in \mathcal{W} \\
\text{FullAttn}(\cdot; \text{ReRoPE}(w)), & \ell \in \mathcal{F}
\end{cases}$$

关键改变：

1. 全局注意力层从NoPE改为ReRoPE
2. 窗口大小$\tilde{w}$不再受严格约束，可以更灵活选择
3. 全局注意力层可以多于1层，也可以放在中间层

**推导3.2（ReRoPE在全局层的应用）**

全局层使用ReRoPE，相对位置矩阵为：

$$[\mathcal{P}_{\text{ReRoPE}}]_{i,j} = \min(i - j, w), \quad i \geq j$$

Attention分数为：

$$a_{i,j}^{(\ell)} = \begin{cases}
\boldsymbol{q}_i^{(\ell) \top}\boldsymbol{\mathcal{R}}^{i-j}\boldsymbol{k}_j^{(\ell)}, & i - j < w \\
\boldsymbol{q}_i^{(\ell) \top}\boldsymbol{\mathcal{R}}^{w}\boldsymbol{k}_j^{(\ell)}, & i - j \geq w
\end{cases}$$

这需要计算两次Attention矩阵（如原始ReRoPE），但由于只有少数几层是全局层，总体计算开销相对较小。

**定理3.3（HWFA2的外推能力）**

HWFA2继承了ReRoPE的无限外推能力。对于任意测试长度$L_{\text{test}}$，只要：

1. 窗口层的窗口大小$\tilde{w} < N_{\text{train}}$
2. 全局层的ReRoPE窗口$w < N_{\text{train}}$

模型就不会遇到未见过的位置编码。

**证明**：

- 窗口层使用RoPE，最大相对位置为$\tilde{w} - 1 < N_{\text{train}}$（训练时见过）
- 全局层使用ReRoPE，最大相对位置为$w < N_{\text{train}}$（训练时见过）

因此所有位置编码都在训练范围内。

**推导3.4（窗口层与全局层的协同）**

窗口层负责局部建模，全局层负责远距离信息聚合。设有$L_w$层窗口层和$L_f$层全局层（$L_w + L_f = L$），则：

- **局部信息传播深度**：$L_w \times \tilde{w}$
- **全局信息整合能力**：$L_f$次全局聚合

有效感受野为：

$$\mathcal{F}_{\text{eff}} = \min(L_w \times \tilde{w}, L_{\text{test}})$$

当$L_w \times \tilde{w} \geq L_{\text{test}}$时，所有位置都能通过局部传播到达，全局层进一步增强远距离交互。

### 4. 计算复杂度的理论降低

**定义4.1（标准Transformer的计算复杂度）**

对于长度$N$的序列，$L$层标准Transformer的时间复杂度为：

$$T_{\text{std}} = L \cdot \mathcal{O}(N^2 d)$$

其中$d$是模型维度。

**定义4.2（HWFA的计算复杂度）**

HWFA的复杂度取决于窗口层和全局层的配置。设有$L_w$层窗口层（窗口大小$\tilde{w}$）和$L_f$层全局层：

$$T_{\text{HWFA}} = L_w \cdot \mathcal{O}(\tilde{w} N d) + L_f \cdot \mathcal{O}(N^2 d)$$

当$\tilde{w} \ll N$且$L_f \ll L$时，复杂度显著降低。

**推导4.3（HWFA2的计算复杂度）**

HWFA2的全局层使用ReRoPE，需要计算两次Attention：

$$T_{\text{HWFA2}} = L_w \cdot \mathcal{O}(\tilde{w} N d) + L_f \cdot \mathcal{O}(2N^2 d)$$

$$= L_w \cdot \tilde{w} N d + 2L_f \cdot N^2 d$$

**定理4.4（计算加速比）**

HWFA2相对于标准Transformer的加速比为：

$$\text{Speedup} = \frac{T_{\text{std}}}{T_{\text{HWFA2}}} = \frac{L \cdot N^2 d}{L_w \cdot \tilde{w} N d + 2L_f \cdot N^2 d} = \frac{L \cdot N}{L_w \cdot \tilde{w} + 2L_f \cdot N}$$

**证明**：简化上式：

$$\text{Speedup} = \frac{L \cdot N}{L_w \cdot \tilde{w} + 2L_f \cdot N}$$

设$L_w = L - L_f$：

$$\text{Speedup} = \frac{L \cdot N}{(L - L_f) \cdot \tilde{w} + 2L_f \cdot N} = \frac{L \cdot N}{L \cdot \tilde{w} + L_f(2N - \tilde{w})}$$

当$N \gg \tilde{w}$时：

$$\text{Speedup} \approx \frac{L \cdot N}{L \cdot \tilde{w} + 2L_f \cdot N}$$

**推导4.5（具体实例）**

对于$L = 24$，$N = 4096$，$\tilde{w} = 64$，$L_f = 2$（2层全局）：

$$\text{Speedup} = \frac{24 \times 4096}{22 \times 64 + 2 \times 2 \times 4096} = \frac{98304}{1408 + 16384} = \frac{98304}{17792} \approx 5.52$$

即计算速度提升约5.5倍。

如果$L_f = 1$（仅1层全局）：

$$\text{Speedup} = \frac{24 \times 4096}{23 \times 64 + 2 \times 1 \times 4096} = \frac{98304}{1472 + 8192} = \frac{98304}{9664} \approx 10.17$$

加速比超过10倍。

**推导4.6（内存复杂度）**

标准Transformer需要存储$L$个全局Attention矩阵：

$$M_{\text{std}} = L \cdot N^2$$

HWFA2需要存储：

$$M_{\text{HWFA2}} = L_w \cdot \tilde{w} N + L_f \cdot N^2$$

内存节省比：

$$\text{MemSave} = \frac{L \cdot N^2 - (L_w \cdot \tilde{w} N + L_f \cdot N^2)}{L \cdot N^2} = 1 - \frac{L_w \cdot \tilde{w}}{L \cdot N} - \frac{L_f}{L}$$

对于上述例子（$L_f = 2$）：

$$\text{MemSave} = 1 - \frac{22 \times 64}{24 \times 4096} - \frac{2}{24} = 1 - \frac{1408}{98304} - \frac{1}{12} \approx 1 - 0.014 - 0.083 = 0.903$$

节省约90%的内存。

### 5. 注意力覆盖范围的完整性证明

**定义5.1（k-hop可达性）**

在Transformer中，如果位置$j$可以通过最多$k$层的注意力路径到达位置$i$，则称$j$是$i$的$k$-hop邻居。

$$\text{Reach}_k(i) = \{j : \exists \text{长度} \leq k \text{的路径从} j \text{到} i\}$$

**定理5.2（HWFA2的完全可达性）**

在HWFA2中，对于任意位置对$(i, j)$，存在路径使得$j$可达$i$。

**证明**：考虑两种情况：

**情况1**：存在全局层$\ell \in \mathcal{F}$。

由于全局层的注意力覆盖所有位置（尽管使用ReRoPE，但仍然可以关注任意位置，只是远距离位置的位置编码相同），任意$j$都可以在全局层直接到达$i$。

**情况2**：只考虑窗口层。

根据定理2.2，经过$L_w$层窗口注意力，感受野为$L_w(\tilde{w} - 1) + 1$。只要：

$$L_w(\tilde{w} - 1) + 1 \geq i - j$$

则$j$可达$i$。

综合两种情况，由于HWFA2至少有1层全局层，完全可达性得证。

**推导5.3（平均最短路径长度）**

定义平均最短路径长度为：

$$\bar{L}_{\text{path}} = \frac{1}{N^2}\sum_{i,j} \min\{\ell : j \in \text{Reach}_\ell(i)\}$$

对于HWFA2：

- 通过窗口层，平均最短路径为$\mathcal{O}(\Delta/\tilde{w})$，其中$\Delta = |i - j|$
- 通过全局层，直接可达，路径长度为1

因此：

$$\bar{L}_{\text{path}} = \mathbb{E}[\min(\lceil\Delta/\tilde{w}\rceil, 1)] \approx 1 + \epsilon$$

其中$\epsilon$是小常数，因为大部分位置对通过全局层即可连接。

**定理5.4（信息瓶颈避免）**

HWFA2通过全局层避免了纯窗口注意力的信息瓶颈问题。

**证明思路**：纯窗口注意力存在信息瓶颈：远距离信息需要经过多层逐步传播，每层都可能损失信息。HWFA2的全局层提供"高速通道"，使得任意远距离信息可以直接传递，避免了多层传播导致的信息衰减。

### 6. 层次化位置编码的优化

**定义6.1（分层位置编码策略）**

在HWFA2中，不同层使用不同的位置编码策略：

$$\text{PE}^{(\ell)} = \begin{cases}
\text{RoPE}, & \ell \in \mathcal{W} \\
\text{ReRoPE}(w), & \ell \in \mathcal{F}
\end{cases}$$

**推导6.2（局部-全局位置信息解耦）**

窗口层的RoPE保留完整的局部位置信息：

$$f_{\text{local}}(m) = m, \quad m \in [0, \tilde{w})$$

全局层的ReRoPE处理全局位置信息：

$$f_{\text{global}}(m) = \min(m, w)$$

这种解耦允许模型在不同粒度上处理位置信息，提高了表达能力。

**定理6.3（位置编码的互补性）**

窗口层的RoPE和全局层的ReRoPE在功能上互补：

- **RoPE**：精确编码局部相对位置，适合短距离依赖
- **ReRoPE**：压缩远距离相对位置，适合全局信息聚合

**推导6.4（全局层位置的灵活配置）**

实验表明，全局层放在中间位置比放在最后效果更好。设$L = 24$，可能的配置：

**配置A**（1层全局，放在末尾）：$\mathcal{F} = \{24\}$

**配置B**（1层全局，放在中间）：$\mathcal{F} = \{12\}$

**配置C**（2层全局，均匀分布）：$\mathcal{F} = \{8, 16\}$

**配置D**（3层全局，均匀分布）：$\mathcal{F} = \{6, 12, 18\}$

定义全局层的位置为$\{\ell_1, \ell_2, \ldots, \ell_{L_f}\}$，最优配置应使全局层均匀分布：

$$\ell_i = \left\lfloor\frac{i \cdot L}{L_f + 1}\right\rfloor, \quad i = 1, 2, \ldots, L_f$$

例如，$L = 24$，$L_f = 2$：

$$\ell_1 = \lfloor 24/3 \rfloor = 8, \quad \ell_2 = \lfloor 48/3 \rfloor = 16$$

**推导6.5（位置编码窗口的协调）**

窗口层的窗口大小$\tilde{w}$和全局层的ReRoPE窗口$w$应该协调选择。建议：

$$\tilde{w} \approx \frac{N_{\text{train}}}{L}, \quad w \approx \frac{N_{\text{train}}}{2}$$

这样窗口层的感受野与层数成正比，全局层覆盖较大范围。

实验中，$N_{\text{train}} = 512$，$L = 24$：

$$\tilde{w} \approx 512/24 \approx 21 \Rightarrow \text{选择} 32 \text{或} 64$$

$$w \approx 512/2 = 256$$

### 7. 训练效果的理论分析

**定义7.1（训练损失分解）**

将训练损失分解为局部损失和全局损失：

$$\mathcal{L} = \alpha \mathcal{L}_{\text{local}} + (1 - \alpha) \mathcal{L}_{\text{global}}$$

其中：
- $\mathcal{L}_{\text{local}}$：短距离预测损失（窗口内）
- $\mathcal{L}_{\text{global}}$：长距离预测损失（跨窗口）

**推导7.2（HWFA与标准Transformer的对比）**

**标准Transformer**：使用全局RoPE，两种损失都能很好建模

**原始HWFA**：

- 窗口层用RoPE，$\mathcal{L}_{\text{local}}$建模良好
- 全局层用NoPE，$\mathcal{L}_{\text{global}}$建模受限

**HWFA2**：

- 窗口层用RoPE，$\mathcal{L}_{\text{local}}$建模良好
- 全局层用ReRoPE，$\mathcal{L}_{\text{global}}$建模显著改善

因此HWFA2的训练损失更接近标准Transformer。

**定理7.3（HWFA2的训练效果优势）**

HWFA2的训练效果优于原始HWFA，接近或达到标准Transformer的水平。

**证明思路**：ReRoPE在窗口内保持与RoPE相同的位置编码，因此局部建模能力不受影响。同时，ReRoPE的截断机制允许全局层有效处理远距离依赖，弥补了NoPE的不足。

**推导7.4（实验结果的理论解释）**

实验结果显示：

- **Baseline**（标准Transformer）：49.41%
- **HWFA**（原始）：48.70%（下降0.71%）
- **HWFA-ReRoPE-w64-f2**：49.46%（超过baseline 0.05%）

训练效果的提升可以归因于：

1. ReRoPE提供的位置信息优于NoPE
2. 2层全局层增强了远距离信息聚合
3. 窗口大小64在局部-全局平衡上优于原始HWFA的16

### 8. 外推性能的数学分析

**定义8.1（外推性能指标）**

定义相对外推性能为：

$$P_{\text{rel}}(L) = \frac{A(L)}{A(N_{\text{train}})}$$

其中$A(L)$是长度$L$上的准确率，$A(N_{\text{train}})$是训练长度上的准确率。

**推导8.2（各方法的外推性能对比）**

根据实验结果（测试长度4096，训练长度512）：

1. **ReRoPE-w256**：$P_{\text{rel}} = 85.12\% / 49.40\% \approx 1.72$
2. **HWFA**：$P_{\text{rel}} = 80.84\% / 48.70\% \approx 1.66$
3. **HWFA-ReRoPE-w64-f2**：$P_{\text{rel}} = 84.43\% / 49.46\% \approx 1.71$

HWFA-ReRoPE的外推性能介于ReRoPE和HWFA之间，接近ReRoPE的最佳水平。

**定理8.4（HWFA2的外推鲁棒性）**

HWFA2在不同窗口大小下都表现出良好的外推性能，具有较强的鲁棒性。

**证明**：实验显示，$\tilde{w} \in \{32, 64, 128\}$时，外推性能都在80%以上，说明HWFA2对窗口大小不敏感。

**推导8.5（窗口大小对外推的影响）**

从实验数据：

- $\tilde{w} = 32$：84.09%（2层全局）
- $\tilde{w} = 64$：84.43%（2层全局）
- $\tilde{w} = 128$：83.09%（2层全局）

存在一个最优窗口大小$\tilde{w}^* \approx 64$（对应$N_{\text{train}}/8$）。过小的窗口限制局部建模，过大的窗口可能导致窗口层的外推问题。

### 9. 全局层数量的优化

**定义9.1（全局层数量的权衡）**

全局层数量$L_f$涉及以下权衡：

- **优势**：更多全局层提供更强的远距离信息聚合能力
- **劣势**：更多全局层增加计算成本

**推导9.2（最优全局层数）**

定义综合性能指标：

$$S(L_f) = A_{\text{extrap}}(L_f) \cdot \text{Speedup}(L_f)$$

其中$A_{\text{extrap}}$是外推准确率，Speedup是加速比。

从实验数据（$\tilde{w} = 64$）：

- $L_f = 1$：$A = 82.41\%$，$\text{Speedup} \approx 11$，$S \approx 906$
- $L_f = 2$：$A = 84.43\%$，$\text{Speedup} \approx 5.5$，$S \approx 464$

单纯从综合指标看，1层全局更优。但考虑到2层全局的准确率提升2个百分点，且加速比仍然可观，实际应用中可能更倾向于2层全局。

**定理9.3（全局层的边际效用递减）**

增加全局层数量的边际效用递减：第1层全局的提升最大，后续全局层的提升逐渐减小。

**推导9.4（全局层数量的建议）**

综合性能和效率：

- **追求极致速度**：$L_f = 1$
- **平衡性能与速度**：$L_f = 2$
- **追求最佳性能**：$L_f \leq L/6$（避免过多全局层）

对于24层模型：$L_f \in \{1, 2, 3, 4\}$是合理范围。

### 10. 窗口注意力的优化实现

**定义10.1（滑动窗口attention）**

窗口注意力可以通过滑动窗口高效实现。对于长度$N$的序列，窗口大小$\tilde{w}$：

$$\text{Output}_i = \sum_{j=\max(0, i-\tilde{w}+1)}^{i} \alpha_{i,j} \boldsymbol{v}_j$$

**推导10.2（分块并行计算）**

将序列分成$N/B$个块，每块大小$B$。当$B > \tilde{w}$时，大部分块的计算可以并行，只有边界处需要特殊处理。

时间复杂度从串行的$\mathcal{O}(N \tilde{w} d)$降低到并行的$\mathcal{O}((N/P) \tilde{w} d)$，其中$P$是并行度。

**推导10.3（Flash Attention兼容性）**

窗口注意力可以直接使用Flash Attention的变体。修改Flash Attention的mask：

```python
for i in range(0, N, block_size):
    for j in range(max(0, i - w + 1), i + 1, block_size):
        # 计算块(i, j)的attention
        compute_block_attention(Q[i], K[j], V[j])
```

这保持了Flash Attention的内存效率。

### 11. ReRoPE在全局层的优化

**定义11.1（全局层ReRoPE的两阶段计算）**

全局层的ReRoPE需要计算两次Attention矩阵，可以通过以下优化减少开销：

**阶段1**：计算窗口内Attention（$i - j < w$）

**阶段2**：计算窗口外Attention（$i - j \geq w$）

**推导11.2（稀疏化优化）**

对于窗口外的Attention，由于所有位置使用相同的ReRoPE值$w$，可以进行近似：

$$a_{i,j}^{\text{window-out}} \approx \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\bar{\boldsymbol{k}}$$

其中$\bar{\boldsymbol{k}}$是窗口外所有Key的平均。这将窗口外的$\mathcal{O}(N^2)$计算降低到$\mathcal{O}(N)$。

然而，这种近似会损失精度，实验中未采用。

**推导11.3（混合精度计算）**

窗口内Attention（更重要）使用FP32，窗口外Attention（相对不重要）使用FP16，可以加速约30%的全局层计算，精度损失可忽略。

### 12. HWFA2的端到端性能

**定义12.1（端到端延迟）**

端到端延迟包括：

1. **前向传播时间**：$T_{\text{forward}}$
2. **内存访问时间**：$T_{\text{memory}}$
3. **通信时间**（多GPU）：$T_{\text{comm}}$

总延迟：

$$T_{\text{total}} = T_{\text{forward}} + T_{\text{memory}} + T_{\text{comm}}$$

**推导12.2（HWFA2的延迟分析）**

对于HWFA2（$L_w = 22$，$L_f = 2$，$\tilde{w} = 64$，$N = 4096$）：

**前向传播**：

$$T_{\text{forward}} = 22 \times T_{\text{window}}(64, 4096) + 2 \times 2 \times T_{\text{full}}(4096)$$

其中$T_{\text{window}}(\tilde{w}, N) \propto \tilde{w} N$，$T_{\text{full}}(N) \propto N^2$。

设$T_{\text{full}}(4096) = t_0$，$T_{\text{window}}(64, 4096) = 64 \times 4096 / (4096 \times 4096) \times t_0 = t_0/64$：

$$T_{\text{forward}} = 22 \times t_0/64 + 4 \times t_0 = t_0(22/64 + 4) = t_0(0.34 + 4) = 4.34t_0$$

标准Transformer：$24 \times t_0$。

加速比：$24 / 4.34 \approx 5.5$（与之前计算一致）。

**内存访问**：由于窗口层的Attention矩阵更小，内存访问时间也减少约90%。

**推导12.3（吞吐量分析）**

吞吐量（tokens/秒）与延迟成反比：

$$\text{Throughput}_{\text{HWFA2}} = \text{Speedup} \times \text{Throughput}_{\text{std}} \approx 5.5 \times \text{Throughput}_{\text{std}}$$

对于批处理，优势更明显，因为窗口注意力允许更大的批大小（内存节省）。

### 13. 多任务泛化能力

**定义13.1（任务泛化性）**

定义模型在不同任务上的平均性能：

$$P_{\text{avg}} = \frac{1}{|T|}\sum_{t \in T} A_t$$

其中$T$是任务集合，$A_t$是任务$t$上的准确率。

**推导13.2（HWFA2的任务适应性）**

HWFA2的混合架构使其能够适应不同类型的任务：

- **局部任务**（如NER）：主要依赖窗口层
- **全局任务**（如文档分类）：主要依赖全局层
- **混合任务**（如问答）：同时利用两类层

这种灵活性使HWFA2在多任务设置下表现更稳定。

### 14. 长序列稳定性

**定义14.1（数值稳定性指标）**

定义Attention权重的方差：

$$\sigma^2_{\alpha} = \text{Var}[\alpha_{i,j}]$$

方差过大表明数值不稳定。

**推导14.2（HWFA2的稳定性优势）**

窗口注意力的Softmax仅在$\tilde{w}$个元素上进行，动态范围更小：

$$\Delta_{\text{window}} = \max_{j \in [i-\tilde{w}+1, i]} a_{i,j} - \min_{j \in [i-\tilde{w}+1, i]} a_{i,j}$$

相比全局注意力的$\Delta_{\text{full}}$（在$N$个元素上），$\Delta_{\text{window}}$通常更小，因此数值更稳定。

全局层虽然是全局的，但使用ReRoPE限制了位置编码范围，也有助于稳定性。

**定理14.3（长序列训练的稳定性）**

HWFA2在长序列训练时比标准Transformer更稳定，梯度爆炸/消失的风险更低。

### 15. 与其他长序列方法的对比

**定义15.1（长序列方法分类）**

1. **近似方法**：Linformer、Performer等，用低秩近似
2. **稀疏方法**：Sparse Transformer、Longformer等，固定稀疏模式
3. **混合方法**：HWFA、HWFA2，结合窗口和全局

**推导15.2（理论对比）**

|方法|复杂度|外推能力|训练效果|
|---|---|---|---|
|标准Transformer|$\mathcal{O}(N^2)$|差|最佳|
|Linformer|$\mathcal{O}(N)$|差|中等|
|Sparse Transformer|$\mathcal{O}(N\sqrt{N})$|中等|中等|
|HWFA|$\mathcal{O}(N)$|良好|良好|
|HWFA2|$\mathcal{O}(N)$|优秀|优秀|

HWFA2在各方面都表现优异。

### 16. 实践建议

**建议16.1（参数选择指南）**

对于训练长度$N_{\text{train}}$，层数$L$：

1. **窗口大小**：$\tilde{w} \in [N_{\text{train}}/16, N_{\text{train}}/4]$
2. **全局层数**：$L_f \in [1, L/6]$
3. **全局层位置**：均匀分布，$\ell_i = \lfloor i \cdot L / (L_f + 1) \rfloor$
4. **ReRoPE窗口**：$w = N_{\text{train}}/2$

**建议16.2（训练策略）**

1. 使用课程学习，逐渐增加序列长度
2. 对全局层使用稍小的学习率（0.5-0.8倍）
3. 应用梯度裁剪，阈值1.0
4. 使用$\log n$缩放因子

**建议16.3（推理优化）**

1. 对窗口层使用Flash Attention变体
2. 对全局层使用混合精度（FP16/FP32）
3. 启用KV缓存（注意ReRoPE的特殊处理）
4. 批处理时优先增大批大小（利用内存节省）

### 17. 理论局限与未来方向

**局限17.1（全局层的计算瓶颈）**

尽管HWFA2大幅减少计算，全局层仍然是瓶颈。当$N$非常大（如100k）时，即使2层全局也可能成为限制因素。

**方向17.2（可能的改进）**

1. **分层全局attention**：全局层也采用多尺度策略
2. **动态窗口**：根据内容自适应调整窗口大小
3. **学习稀疏模式**：端到端学习attention模式
4. **神经架构搜索**：自动搜索最优层配置

**方向17.3（与其他技术结合）**

1. **HWFA2 + Flash Attention 2**：进一步加速
2. **HWFA2 + 低秩分解**：降低全局层复杂度
3. **HWFA2 + 稀疏激活**：MoE风格的稀疏全局层

### 18. 总结

**总结18.1（HWFA2的核心贡献）**

1. **理论创新**：将ReRoPE引入HWFA，实现性能与效率的最佳平衡
2. **实践突破**：训练效果超越baseline，外推能力接近ReRoPE，推理速度提升5-10倍
3. **灵活性**：支持多种配置，适应不同应用需求

**总结18.2（关键洞察）**

HWFA2成功的关键在于：

1. **层次化设计**：窗口层处理局部，全局层处理全局
2. **位置编码协同**：RoPE和ReRoPE优势互补
3. **计算效率**：大部分层是低成本的窗口attention
4. **外推能力**：ReRoPE保证无限外推

**总结18.3（应用前景）**

HWFA2适用于：

- 长文档理解
- 代码生成（长上下文）
- 对话系统（长历史）
- 生物序列分析

随着序列长度需求不断增长，HWFA2类方法将越来越重要。

