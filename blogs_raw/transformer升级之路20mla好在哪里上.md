---
title: Transformer升级之路：20、MLA好在哪里?（上）
slug: transformer升级之路20mla好在哪里上
date: 2025-05-04
tags: 详细推导, 优化, 语言模型, 生成模型, attention, 生成模型
status: completed
---
# Transformer升级之路：20、MLA好在哪里?（上）

**原文链接**: [https://spaces.ac.cn/archives/10907](https://spaces.ac.cn/archives/10907)

**发布日期**: 

---

自从DeepSeek爆火后，它所提的Attention变体MLA（**M** ulti-head **L** atent **A** ttention）也愈发受到关注。MLA通过巧妙的设计实现了MHA与MQA的自由切换，使得模型可以根据训练和推理的不同特性（Compute-Bound or Memory-Bound）选择最佳的形式，尽可能地达到效率最大化。

诚然，MLA很有效，但也有观点认为它不够优雅，所以寻找MLA替代品的努力一直存在，包括我们也有在尝试。然而，经过一段时间的实验，我们发现很多KV Cache相同甚至更大的Attention变体，最终效果都不如MLA。这不得不让我们开始反思：MLA的出色表现背后的关键原因究竟是什么？

接下来，本文将详细介绍笔者围绕这一问题的思考过程以及相关实验结果。

## 观察 #

MLA提出自[DeepSeek-V2](https://papers.cool/arxiv/2405.04434)，本文假设读者已经熟悉MLA，至少了解之前的博客[《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》](/archives/10091)所介绍的内容，因此MLA自身的细节将不会过多展开。

MLA的主要特点如下：

> 1、MLA在训练阶段是一个qk_head_dims=(128+64)、v_head_dims=128的MHA；
> 
> 2、MLA在解码阶段是一个qk_head_dims=(512+64)、v_head_dims=512、KV-Shared的MQA；
> 
> 3、MLA的[qc, qr]、[kc, kr]拼接，可以理解为一种[Partial RoPE](/archives/10122#%E9%83%A8%E5%88%86%E6%97%8B%E8%BD%AC)。

## 猜测 #

MHA、GQA常用的head_dims是128，而对于MLA来说，不管是从训练看的128+64，还是从推理看的512+64，都要大于128，再结合[《突破瓶颈，打造更强大的Transformer》](/archives/7325)的经验，我们有：

> **猜测1** ： 增大head_dims是MLA好的关键之一。

另外，KV-Shared这个特性，可以在同等KV Cache大小下，增大GQA的head_dims或者num_groups，所以有：

> **猜测2** ： KV-Shared是MLA好的关键之一。

最后，此前有一些理论和实验显示Partial RoPE可能会对效果有正面帮助（参考[《Transformer升级之路：18、RoPE的底数选择原则》](/archives/10122#%E9%83%A8%E5%88%86%E6%97%8B%E8%BD%AC)），所以有

> **猜测3** ： Partial RoPE是MLA好的关键之一。

## 实验 #

现在我们通过实验逐一检验以上猜测。

### 设置 #

所有实验公共部分的超参数如下：

> 1、类似LLAMA3的Dense模型；
> 
> 2、hidden_size=2048，num_layers=12，num_heads=16；
> 
> 3、优化器是[Muon](/archives/10592)，Attention部分per head更新；
> 
> 4、训练长度为4096，总tokens数为16B，总训练步数为16k；
> 
> 5、所有实验都是只改变Attention，所以参数量不会严格对齐。

### Part I #

MLA的KV Cache大小是512+64，约等于GQA2-128（第一个数字是num_groups，第二个数字是head_dims），所以对比的baseline为GQA2-128和GQA1-256。为了验证Partial RoPE，我们增加了GQA1-256-PR，具体做法是将Q、K的256 dims分成192+64两部分，在64上加RoPE，192不加。

结果如下：  
$$\begin{array}{c|ccc}  
\hline  
& \text{Params} & \text{Loss} & \text{Cache} \\\  
\hline  
\text{MLA} & 894M & 2.721 & 576 \\\  
\text{GQA2-128} & 842M & 2.75 & 512 \\\  
\text{GQA1-256} & 943M & 2.72 & 512 \\\  
\text{GQA1-256-PR} & 943M & 2.711 & 512 \\\  
\hline  
\end{array}$$

即  
$$\text{GQA2-128} < \text{MLA} \lesssim \text{GQA1-256} < \text{GQA1-256-PR}$$  
初步验证了增大head_dims和Partial RoPE的作用。这样看来，MLA的设计中，RoPE和NoPE拼接这部分看似无奈的设计，极有可能是它效果优异的关键原因！原论文声称MLA甚至优于MHA，大概率也是因为所对比的MHA的head_dims只有128。

### Part II #

为了进一步验证增大head_dims的作用，我们另外跑了MHA、GQA2-192、MLA-256三个实验，MHA是head_dims=128的常规MHA，GQA2-192是直接增大GQA2的head_dims到192，MLA-256是将MLA的128+64提升到192+64，对照如下

$$\begin{array}{c|ccc}  
\hline  
& \text{Params} & \text{Loss} & \text{Cache} \\\  
\hline  
\text{MHA} & 931M & 2.721 & 4096 \\\  
\text{MLA} & 894M & 2.721 & 576 \\\  
\text{MLA-256} & 989M & 2.705 & 576 \\\  
\text{GQA2-128} & 842M & 2.75 & 512 \\\  
\text{GQA2-192} & 899M & 2.729 & 768 \\\  
\text{GQA1-256} & 943M & 2.72 & 512 \\\  
\text{GQA1-256-PR} & 943M & 2.711 & 512 \\\  
\hline  
\end{array}$$

可以看到，MHA总参数量更多，KV Cache更是7倍于MLA，但Loss才堪堪追平MLA，这跟DeepSeek-V2里边的结论接近。此外，GQA2-192优于GQA2-128，但不如GQA1-256；MLA的head_dims升到(192+64)后，相比(128+64)也还能进一步提升效果。这些现象都表明，增加head_dims远比增加num_groups更有效。

### Part III #

接下来我们验证KV-Shared，即K、V共享全部或大部分dims。这里我们主要考虑的替代品是head_dims不超过256的GQA，并且控制KV Cache的总大小跟MLA接近，所以当KV-Shared时，我们可以至多可以考虑GQA2-256。

由于KV-Shared跟RoPE不完全兼容，参考MLA的做法，我们将256分成192+64两部分，其中

> 1、192部分不加RoPE，在K、V间共享；
> 
> 2、64部分加RoPE，只用于K；
> 
> 3、V另外再投影64 dims，concat到共享的192 dims上去。

这样一来，K、V的head_dims都是256，KV Cache总大小是(192+64+64)*2=640，略大于MLA的512+64=576，这个版本我们简记为“GQA2-(192+64)-S1”，其实“S1”是“Shared-1”的缩写。

### Part IV #

另外一种KV-Shared的方案是：

> 1、192部分不加RoPE，在K、V间共享；
> 
> 2、64部分加RoPE，同样在K、V间共享；
> 
> 3、做Attention，由于V带RoPE，此时是绝对位置编码效果；
> 
> 4、为了保证相对位置编码，将输出分成192+64两部分，64部分再加一次逆向RoPE。

这种做法是K、V完全共享，KV Cache大小是(192+64)*2=512，略小于MLA。这个版本我们称为“GQA2-(192+64)-S2”，“S2”是“Shared-2”的缩写，背后的原理是笔者新提出的VO-RoPE，参考[《Transformer升级之路：19、第二类旋转位置编码》](/archives/10862)。

### Part V #

另外，根据同样思路补了几个GQA4和GQA1的实验。所有实验结果汇总如下：  
$$\begin{array}{c|ccc|c}  
\hline  
& \text{Params} & \text{Loss} & \text{Cache} & \text{备注} \\\  
\hline  
\text{MLA} & 894M & 2.721 & 576 & \\\  
\text{MLA-256} & 989M & 2.705 & 576 & \\\  
\text{GQA2-(192+64)-S1} & 946M & 2.714 & 640 & \\\  
\text{GQA2-(192+64)-S2} & 943M & 2.708 & 512 & \text{引入VO-RoPE} \\\  
\text{GQA4-(64+64)-S2} & 842M & 2.738 & 512 & \\\  
\text{GQA4-(128+64)-S2} & 899M & 2.713 & 768 & \text{KV Cache最大} \\\  
\text{GQA1-(512+64)-S3} & 1171M & 2.677 & 576 & \text{head_dims最大} \\\  
\hline  
\end{array}$$

这里“GQA1-(512+64)-S3”是按照MLA的推理形式实现的MQA，形式介乎S1与S2之间，它的主要特点是head_dims大。

结果解读：

> 1、KV-Shared的GQA自带Partial RoPE；
> 
> 2、KV-Shared的GQA2-256，也能超过MLA；
> 
> 3、VO-RoPE的引入，似乎有利于效果（S1 ≲ S2）；
> 
> 4、同等KV Cache下，head_dims越大越好；
> 
> 5、GQA2-(192+64)-S2 略微超过 GQA1-256-PR；
> 
> 6、GQA4-(128+64)-S2 的KV Cache最大，但效果不是最优，再次表明head_dims更关键。

关于KV-Shared，还有两点观察：

> 1、训练过程中，GQA1-256-PR前期是明显领先GQA2-(192+64)-S2，但后期被追平甚至略微反先，猜测GQA1-256-PR可能有后劲不足的嫌疑；
> 
> 2、如果没有KV-Shared，GQA顶多是GQA1-256，也就是说head_dims顶天了256，但有KV-Shared的话，GQA可以做到GQA1-512-S，单纯从head_dims看，KV-Shared天花板更高。

### Part VI #

由于没有严格对齐参数量，可能读者会有“到底是增加参数量还是增加head_dims更本质”的疑虑，所以这里补充几个对齐参数量的实验。

这里考虑的对齐参数量的方式有三种：

> 1、**double-heads** ：以“GQA2-128 vs GQA1-256”为例，将GQA2-128的num_heads翻倍，可以让GQA2-128的参数量跟GQA1-256相同；
> 
> 2、**缩减MLP** ：缩小MLP（SwiGLU）的intermediate_size，也可以使得GQA1-256的参数量跟GQA2-128大致相同；
> 
> 3、**Q &O LoRA**：GQA的主要参数量来自Query和Output的投影矩阵，对这两个矩阵改用LoRA，也可以降低GQA1-256的参数量。

实验结果如下：  
$$\begin{array}{c|ccc|ccc}  
\hline  
& \text{Params} & \text{Loss} & \text{Cache} & \text{num_heads} & \text{intermediate_size} & \text{qo_lora} \\\  
\hline  
\text{MLA} & 894M & 2.721 & 576 & 16 & 5456 & \text{No}\\\  
\hline  
\text{GQA2-128} & 842M & 2.75 & 512 & 16 & 5456 & \text{No}\\\  
\text{GQA1-256} & 943M & 2.72 & 512 & 16 & 5456 & \text{No}\\\  
\hline  
\text{GQA2-128} & 943M & 2.723 & 512 & \color{red}{32} & 5456 & \text{No} \\\  
\text{GQA1-256} & 843M & 2.747 & 512 & 16 & \color{red}{4096} & \text{No} \\\  
\text{GQA1-256} & 842M & 2.726 & 512 & 16 & 5456 & \color{red}{\text{Yes}} \\\  
\hline  
\text{GQA4-(64+64)-S2} & 842M & 2.738 & 512 & 16 & 5456 & \text{No} \\\  
\text{GQA2-(192+64)-S2} & 943M & 2.708 & 512 & 16 & 5456 & \text{No} \\\  
\hline  
\text{GQA4-(64+64)-S2} & 943M & 2.711 & 512 & \color{red}{32} & 5456 & \text{No} \\\  
\text{GQA2-(192+64)-S2} & 843M & 2.733 & 512 & 16 & \color{red}{4096} & \text{No} \\\  
\text{GQA2-(192+64)-S2} & 842M & 2.708 & 512 & 16 & 5456 & \color{red}{\text{Yes}} \\\  
\hline  
\end{array}$$

结果主要分三块：

> 1、heads翻倍相比head_dims翻倍，loss稳定差0.003左右；
> 
> 2、缩小MLP比head_dims减半，loss稳定优0.004左右；
> 
> 3、Q&O LoRA性能损失最小，可以实现head_dims翻倍但参数量不增，且loss明显降。

结论：如果从增加参数量角度看，增大head_dims可能是效果增益较大的方向，配合Q&O LoRA可以实现参数量几乎不增，但收益仍相当。

## 小结 #

初步结论是：

> 1、增大head_dims收益最大；
> 
> 2、Partial RoPE对Loss也有一定帮助；
> 
> 3、KV-Shared应该也有一定作用。

这样看来，此前我们一直在head_dims=128下找MLA的替代品，感觉是起点就先天不足了，难怪一直比不上MLA。要想追平MLA，head_dims应该要192起步了，并辅以Partial RoPE。至于KV-Shared，也可能有用，但应该还需要更大规模的验证。

## 意义 #

其实这里边的意义，就看我们换掉MLA的决心有多强。

假设 GQA2-(192+64)-S2 可以替代MLA，但MLA也可以升到256，目前看来 GQA2-(192+64)-S2 比不上 MLA-256 。那么换掉MLA的唯二好处是：

> 1、结构更简单，可以方便加QK-Norm；
> 
> 2、解码阶段的head_dims由512+64变成了256，同时num_groups变为2，可以TP。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10907>_

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

苏剑林. (May. 04, 2025). 《Transformer升级之路：20、MLA好在哪里?（上） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10907>

@online{kexuefm-10907,  
title={Transformer升级之路：20、MLA好在哪里?（上）},  
author={苏剑林},  
year={2025},  
month={May},  
url={\url{https://spaces.ac.cn/archives/10907}},  
} 


---

## 公式推导与注释

### 1. MLA的数学定义与基本结构

**推导1.1：标准Multi-Head Attention（MHA）的数学表述**

对于标准的MHA，给定输入$\boldsymbol{x}_i \in \mathbb{R}^d$，每个head $s \in \{1, 2, \ldots, h\}$计算：
$$
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i \boldsymbol{W}_q^{(s)}, \quad \boldsymbol{W}_q^{(s)} \in \mathbb{R}^{d \times d_k}
$$
$$
\boldsymbol{k}_i^{(s)} = \boldsymbol{x}_i \boldsymbol{W}_k^{(s)}, \quad \boldsymbol{W}_k^{(s)} \in \mathbb{R}^{d \times d_k}
$$
$$
\boldsymbol{v}_i^{(s)} = \boldsymbol{x}_i \boldsymbol{W}_v^{(s)}, \quad \boldsymbol{W}_v^{(s)} \in \mathbb{R}^{d \times d_v}
$$

注意力输出：
$$
\boldsymbol{o}_i^{(s)} = \sum_{j \leq i} \frac{\exp(\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{k}_j^{(s)})}{\sum_{j' \leq i} \exp(\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{k}_{j'}^{(s)})} \boldsymbol{v}_j^{(s)}
$$

最终拼接：
$$
\boldsymbol{o}_i = [\boldsymbol{o}_i^{(1)}, \boldsymbol{o}_i^{(2)}, \ldots, \boldsymbol{o}_i^{(h)}] \boldsymbol{W}_o
$$

**推导1.2：MLA的双重投影结构**

MLA引入中间表示$\boldsymbol{c}_i \in \mathbb{R}^{d_c}$（latent compression）：
$$
\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c, \quad \boldsymbol{W}_c \in \mathbb{R}^{d \times d_c}
$$

然后从$\boldsymbol{c}_i$生成每个head的K和V：
$$
\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i \boldsymbol{W}_k^{(s)}, \quad \boldsymbol{W}_k^{(s)} \in \mathbb{R}^{d_c \times d_k}
$$
$$
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i \boldsymbol{W}_v^{(s)}, \quad \boldsymbol{W}_v^{(s)} \in \mathbb{R}^{d_c \times d_v}
$$

而Q仍然直接从$\boldsymbol{x}_i$生成：
$$
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i \boldsymbol{W}_q^{(s)}, \quad \boldsymbol{W}_q^{(s)} \in \mathbb{R}^{d \times d_k}
$$

**推导1.3：MLA的参数量分析**

MHA的参数量：
$$
P_{\text{MHA}} = h \cdot d \cdot (d_k + d_k + d_v) + h \cdot d_v \cdot d = h \cdot d \cdot (2d_k + d_v + d_v)
$$

简化为（假设$d_k = d_v = d_h$）：
$$
P_{\text{MHA}} = 4h \cdot d \cdot d_h
$$

MLA的参数量：
$$
P_{\text{MLA}} = d \cdot d_c + h \cdot d_c \cdot (d_k + d_v) + h \cdot d \cdot d_k + h \cdot d_v \cdot d
$$

整理得：
$$
P_{\text{MLA}} = d \cdot d_c + h \cdot d_c \cdot (d_k + d_v) + h \cdot d \cdot (d_k + d_v)
$$

### 2. 低秩分解的理论基础

**推导2.1：K和V矩阵的低秩表示**

将所有位置的K向量堆叠成矩阵$\boldsymbol{K}^{(s)} \in \mathbb{R}^{n \times d_k}$：
$$
\boldsymbol{K}^{(s)} = \begin{bmatrix} \boldsymbol{k}_1^{(s)} \\ \boldsymbol{k}_2^{(s)} \\ \vdots \\ \boldsymbol{k}_n^{(s)} \end{bmatrix} = \begin{bmatrix} \boldsymbol{x}_1 \boldsymbol{W}_c \boldsymbol{W}_k^{(s)} \\ \boldsymbol{x}_2 \boldsymbol{W}_c \boldsymbol{W}_k^{(s)} \\ \vdots \\ \boldsymbol{x}_n \boldsymbol{W}_c \boldsymbol{W}_k^{(s)} \end{bmatrix}
$$

可以分解为：
$$
\boldsymbol{K}^{(s)} = \boldsymbol{C} \boldsymbol{W}_k^{(s)}
$$

其中$\boldsymbol{C} = [\boldsymbol{c}_1, \boldsymbol{c}_2, \ldots, \boldsymbol{c}_n]^{\top} \in \mathbb{R}^{n \times d_c}$。

**推导2.2：秩的上界分析**

矩阵$\boldsymbol{K}^{(s)}$的秩满足：
$$
\text{rank}(\boldsymbol{K}^{(s)}) = \text{rank}(\boldsymbol{C} \boldsymbol{W}_k^{(s)}) \leq \min(\text{rank}(\boldsymbol{C}), \text{rank}(\boldsymbol{W}_k^{(s)}))
$$

由于$\boldsymbol{C} \in \mathbb{R}^{n \times d_c}$和$\boldsymbol{W}_k^{(s)} \in \mathbb{R}^{d_c \times d_k}$：
$$
\text{rank}(\boldsymbol{K}^{(s)}) \leq \min(n, d_c, d_k)
$$

对于长序列，$n \gg d_c$，所以：
$$
\text{rank}(\boldsymbol{K}^{(s)}) \leq d_c
$$

这意味着MLA强制K矩阵是低秩的，秩不超过$d_c$。

**推导2.3：低秩分解的误差界**

假设MHA的K矩阵$\boldsymbol{K}_{\text{full}}^{(s)}$可以用SVD分解：
$$
\boldsymbol{K}_{\text{full}}^{(s)} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}
$$

其中$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$是奇异值。

MLA使用秩$d_c$的近似：
$$
\boldsymbol{K}_{\text{MLA}}^{(s)} \approx \sum_{i=1}^{d_c} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}
$$

Frobenius范数意义下的近似误差：
$$
\|\boldsymbol{K}_{\text{full}}^{(s)} - \boldsymbol{K}_{\text{MLA}}^{(s)}\|_F = \sqrt{\sum_{i=d_c+1}^r \sigma_i^2}
$$

如果奇异值快速衰减（$\sigma_i \sim i^{-\alpha}$，$\alpha > 1$），则低秩近似误差很小。

### 3. KV Cache压缩的数学原理

**推导3.1：标准MHA的KV Cache大小**

对于序列长度$n$，MHA需要缓存每个head的K和V：
$$
\text{Cache}_{\text{MHA}} = n \cdot h \cdot (d_k + d_v)
$$

对于$h=16$，$d_k = d_v = 128$：
$$
\text{Cache}_{\text{MHA}} = n \cdot 16 \cdot 256 = 4096n
$$

**推导3.2：MLA的KV Cache大小**

MLA只需要缓存压缩表示$\boldsymbol{c}_i$和RoPE部分$\boldsymbol{k}_r^{(s)}$。

设$\boldsymbol{c}_i \in \mathbb{R}^{d_{c,k}}$（K的压缩维度）和$\boldsymbol{c}_i^v \in \mathbb{R}^{d_{c,v}}$（V的压缩维度），以及RoPE维度$d_r$：
$$
\text{Cache}_{\text{MLA}} = n \cdot (d_{c,k} + d_{c,v} + h \cdot d_r)
$$

对于DeepSeek-V2的配置（$d_{c,k} = 512$，$d_{c,v} = 512$，$d_r = 64$，$h = 16$）：
$$
\text{Cache}_{\text{MLA}} = n \cdot (512 + 512 + 16 \times 64) = n \cdot 2048
$$

但实际上K和V可以共享大部分：
$$
\text{Cache}_{\text{MLA}} = n \cdot (512 + 64) = 576n
$$

**推导3.3：压缩比的计算**

压缩比定义为：
$$
\rho = \frac{\text{Cache}_{\text{MLA}}}{\text{Cache}_{\text{MHA}}} = \frac{576n}{4096n} = \frac{576}{4096} \approx 0.14
$$

即MLA将KV Cache压缩到原来的14%，约7倍的压缩。

### 4. 注意力矩阵的秩分析

**推导4.1：注意力矩阵的定义**

对于第$s$个head，注意力矩阵$\boldsymbol{A}^{(s)} \in \mathbb{R}^{n \times n}$定义为：
$$
A_{ij}^{(s)} = \frac{\exp(\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{k}_j^{(s)})}{\sum_{j' \leq i} \exp(\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{k}_{j'}^{(s)})}
$$

未归一化的注意力分数矩阵：
$$
\boldsymbol{S}^{(s)} = \boldsymbol{Q}^{(s)} (\boldsymbol{K}^{(s)})^{\top}
$$

其中$\boldsymbol{Q}^{(s)}, \boldsymbol{K}^{(s)} \in \mathbb{R}^{n \times d_k}$。

**推导4.2：MLA注意力矩阵的秩界**

对于MLA，$\boldsymbol{K}^{(s)} = \boldsymbol{C} \boldsymbol{W}_k^{(s)}$，因此：
$$
\boldsymbol{S}^{(s)} = \boldsymbol{Q}^{(s)} (\boldsymbol{C} \boldsymbol{W}_k^{(s)})^{\top} = \boldsymbol{Q}^{(s)} (\boldsymbol{W}_k^{(s)})^{\top} \boldsymbol{C}^{\top}
$$

注意力分数矩阵的秩：
$$
\text{rank}(\boldsymbol{S}^{(s)}) \leq \min(\text{rank}(\boldsymbol{Q}^{(s)}), \text{rank}(\boldsymbol{C}), d_k)
$$

由于$\boldsymbol{C} \in \mathbb{R}^{n \times d_c}$：
$$
\text{rank}(\boldsymbol{S}^{(s)}) \leq \min(n, d_c, d_k)
$$

当$n \gg d_c$时：
$$
\text{rank}(\boldsymbol{S}^{(s)}) \leq d_c
$$

**推导4.3：低秩注意力的表达能力**

虽然注意力矩阵是低秩的，但这不一定限制表达能力。考虑输出：
$$
\boldsymbol{O}^{(s)} = \boldsymbol{A}^{(s)} \boldsymbol{V}^{(s)}
$$

如果$\boldsymbol{V}^{(s)} = \boldsymbol{C} \boldsymbol{W}_v^{(s)}$也是低秩的：
$$
\boldsymbol{O}^{(s)} = \boldsymbol{A}^{(s)} \boldsymbol{C} \boldsymbol{W}_v^{(s)}
$$

即使$\boldsymbol{A}^{(s)}$是低秩的，$\boldsymbol{A}^{(s)} \boldsymbol{C}$可以恢复丰富的表示，因为$\boldsymbol{C}$包含了所有位置的压缩信息。

### 5. 参数效率与表达能力的权衡

**推导5.1：有效参数量的分析**

定义有效参数量为影响前向传播的独立参数数量。

对于MHA：
- Q投影：$h \cdot d \cdot d_k$
- K投影：$h \cdot d \cdot d_k$
- V投影：$h \cdot d \cdot d_v$
- O投影：$h \cdot d_v \cdot d$

总计：$P_{\text{MHA}} = h \cdot d \cdot (2d_k + d_v) + h \cdot d_v \cdot d$

对于MLA：
- 压缩投影：$d \cdot d_c$
- Q投影：$h \cdot d \cdot d_k$
- K投影（从压缩）：$h \cdot d_c \cdot d_k$
- V投影（从压缩）：$h \cdot d_c \cdot d_v$
- O投影：$h \cdot d_v \cdot d$

总计：$P_{\text{MLA}} = d \cdot d_c + h \cdot d \cdot d_k + h \cdot d_c \cdot (d_k + d_v) + h \cdot d_v \cdot d$

**推导5.2：参数效率比**

假设$d_k = d_v = d_h$，比较MLA和MHA在相同KV Cache下的参数量。

设MHA使用GQA，$g$个groups，则：
$$
P_{\text{GQA}} = h \cdot d \cdot d_h + g \cdot d \cdot 2d_h + h \cdot d_h \cdot d
$$

对于$g=2$，$d_h=128$：
$$
P_{\text{GQA}} = 16 \cdot d \cdot 128 + 2 \cdot d \cdot 256 + 16 \cdot 128 \cdot d = (2048 + 512 + 2048)d = 4608d
$$

对于MLA（$d_c = 512$，$d_h = 128 + 64 = 192$）：
$$
P_{\text{MLA}} = d \cdot 512 + 16 \cdot d \cdot 128 + 16 \cdot 512 \cdot 192 + 16 \cdot 192 \cdot d
$$
$$
= 512d + 2048d + 1,572,864 + 3072d = 5632d + 1,572,864
$$

当$d = 2048$时：
$$
P_{\text{MLA}} \approx 11.5M + 1.57M = 13.07M
$$
$$
P_{\text{GQA}} = 4608 \times 2048 = 9.44M
$$

所以MLA参数量稍多，但提供了更大的head_dims。

**推导5.3：表达能力的量化**

定义表达能力为模型能够表示的函数空间的维度。对于线性变换：

MHA的表达能力受限于：
$$
\mathcal{C}_{\text{MHA}} \sim \mathcal{O}(h \cdot d_h \cdot d)
$$

MLA的表达能力：
$$
\mathcal{C}_{\text{MLA}} \sim \mathcal{O}(d_c \cdot d + h \cdot d_c \cdot d_h)
$$

当$d_c$足够大时，MLA可以达到接近MHA的表达能力，同时享受压缩的KV Cache。

### 6. 计算复杂度的严格分析

**推导6.1：训练阶段的计算复杂度**

对于序列长度$n$，MHA的计算分为几个步骤：

1. **QKV投影**：
$$
\mathcal{O}_{\text{QKV}} = n \cdot h \cdot d \cdot (d_k + d_k + d_v)
$$

2. **注意力分数计算**：
$$
\mathcal{O}_{\text{score}} = n^2 \cdot h \cdot d_k
$$

3. **注意力加权求和**：
$$
\mathcal{O}_{\text{weighted}} = n^2 \cdot h \cdot d_v
$$

4. **输出投影**：
$$
\mathcal{O}_{\text{output}} = n \cdot h \cdot d_v \cdot d
$$

总计（忽略常数）：
$$
\mathcal{O}_{\text{MHA}} = n \cdot h \cdot d \cdot d_h + n^2 \cdot h \cdot d_h
$$

**推导6.2：MLA训练阶段的计算复杂度**

1. **压缩投影**：
$$
\mathcal{O}_{\text{compress}} = n \cdot d \cdot d_c
$$

2. **Q投影**：
$$
\mathcal{O}_{\text{Q}} = n \cdot h \cdot d \cdot d_k
$$

3. **KV投影（从压缩）**：
$$
\mathcal{O}_{\text{KV}} = n \cdot h \cdot d_c \cdot (d_k + d_v)
$$

4. **注意力计算**（与MHA相同）：
$$
\mathcal{O}_{\text{attn}} = n^2 \cdot h \cdot d_k + n^2 \cdot h \cdot d_v
$$

5. **输出投影**：
$$
\mathcal{O}_{\text{output}} = n \cdot h \cdot d_v \cdot d
$$

总计：
$$
\mathcal{O}_{\text{MLA}} = n \cdot d \cdot d_c + n \cdot h \cdot d \cdot d_k + n \cdot h \cdot d_c \cdot (d_k + d_v) + n^2 \cdot h \cdot (d_k + d_v)
$$

**推导6.3：Decoding阶段的计算复杂度**

Decoding时，每步只生成一个token，利用KV Cache。

MHA的单步计算：
$$
\mathcal{O}_{\text{MHA,decode}} = h \cdot d \cdot d_k + n \cdot h \cdot d_k + n \cdot h \cdot d_v + h \cdot d_v \cdot d
$$

简化为：
$$
\mathcal{O}_{\text{MHA,decode}} = h \cdot d \cdot d_h + n \cdot h \cdot d_h
$$

MLA的单步计算：
$$
\mathcal{O}_{\text{MLA,decode}} = d \cdot d_c + h \cdot d \cdot d_k + n \cdot h \cdot d_k + n \cdot h \cdot d_v + h \cdot d_v \cdot d
$$

由于MLA的KV Cache更小（$d_c$维而非$h \cdot d_h$维），访存更少，实际速度更快。

### 7. head_dims的影响分析

**推导7.1：head_dims与模型容量**

对于固定的总维度$d = h \cdot d_h$，可以选择：
- 更多的heads（大$h$，小$d_h$）
- 更大的head_dims（小$h$，大$d_h$）

模型的表达能力与参数量相关：
$$
P = h \cdot d \cdot d_h = d \cdot d
$$

参数量相同，但分配不同。

**推导7.2：head_dims与注意力多样性**

每个head学习不同的注意力模式。设head $s$的注意力熵为：
$$
H^{(s)} = -\sum_{j=1}^n A_{ij}^{(s)} \log A_{ij}^{(s)}
$$

总的注意力多样性：
$$
H_{\text{total}} = \sum_{s=1}^h H^{(s)}
$$

更多的heads（大$h$）可以提供更多样的注意力模式，但每个head的容量（$d_h$）更小。

**推导7.3：head_dims与梯度流**

在反向传播中，梯度通过每个head传播。设损失$\mathcal{L}$对输出$\boldsymbol{o}_i$的梯度为$\frac{\partial \mathcal{L}}{\partial \boldsymbol{o}_i}$。

对于MHA：
$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{q}_i^{(s)}} = \sum_{j} \frac{\partial \mathcal{L}}{\partial A_{ij}^{(s)}} \frac{\partial A_{ij}^{(s)}}{\partial \boldsymbol{q}_i^{(s)}}
$$

梯度的方差与$d_h$相关。更大的$d_h$提供更稳定的梯度，有助于训练。

### 8. Partial RoPE的数学机制

**推导8.1：Partial RoPE的分解**

将Q和K分为两部分：
$$
\boldsymbol{q}_i^{(s)} = [\boldsymbol{q}_{i,c}^{(s)}, \boldsymbol{q}_{i,r}^{(s)}]
$$
$$
\boldsymbol{k}_j^{(s)} = [\boldsymbol{k}_{j,c}^{(s)}, \boldsymbol{k}_{j,r}^{(s)}]
$$

其中$\boldsymbol{q}_{i,c}^{(s)} \in \mathbb{R}^{d_{c}}$（content，不加RoPE），$\boldsymbol{q}_{i,r}^{(s)} \in \mathbb{R}^{d_r}$（rotary，加RoPE）。

**推导8.2：注意力分数的分解**

注意力分数：
$$
\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{k}_j^{(s)} = \boldsymbol{q}_{i,c}^{(s)} \cdot \boldsymbol{k}_{j,c}^{(s)} + (\boldsymbol{\mathcal{R}}_{i-j} \boldsymbol{q}_{i,r}^{(s)}) \cdot \boldsymbol{k}_{j,r}^{(s)}
$$

第一项是纯内容相似度，第二项包含相对位置信息。

**推导8.3：Partial RoPE的优势**

从期望的角度：
$$
\mathbb{E}[\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{k}_j^{(s)}] = \mathbb{E}[\boldsymbol{q}_{i,c}^{(s)} \cdot \boldsymbol{k}_{j,c}^{(s)}] + \mathbb{E}[(\boldsymbol{\mathcal{R}}_{i-j} \boldsymbol{q}_{i,r}^{(s)}) \cdot \boldsymbol{k}_{j,r}^{(s)}]
$$

第一项不受位置影响，完全基于语义；第二项受位置调制。这种分离使得模型可以灵活平衡语义和位置信息。

### 9. KV Shared的理论分析

**推导9.1：KV共享的数学表示**

完全KV共享意味着：
$$
\boldsymbol{k}_i = \boldsymbol{v}_i = \boldsymbol{c}_i
$$

注意力计算变为：
$$
\boldsymbol{o}_i^{(s)} = \sum_{j \leq i} \frac{\exp(\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{c}_j)}{\sum_{j' \leq i} \exp(\boldsymbol{q}_i^{(s)} \cdot \boldsymbol{c}_{j'})} \boldsymbol{c}_j
$$

**推导9.2：KV共享的秩分析**

当K和V共享时，它们的秩相同：
$$
\text{rank}(\boldsymbol{K}) = \text{rank}(\boldsymbol{V}) = \text{rank}(\boldsymbol{C}) \leq d_c
$$

这进一步压缩了表示，但也限制了灵活性。

**推导9.3：部分KV共享**

MLA采用部分共享：content部分共享，RoPE部分独立：
$$
\boldsymbol{k}_i = [\boldsymbol{c}_i, \boldsymbol{k}_{i,r}], \quad \boldsymbol{v}_i = [\boldsymbol{c}_i, \boldsymbol{v}_{i,r}]
$$

KV Cache大小：
$$
\text{Cache} = n \cdot (d_c + d_{k,r} + d_{v,r})
$$

这在压缩和灵活性之间取得平衡。

### 10. GQA与MLA的理论对比

**推导10.1：GQA的数学形式**

Grouped Query Attention将$h$个heads分成$g$个groups，每组共享K和V：
$$
\boldsymbol{k}_i^{(g)} = \boldsymbol{x}_i \boldsymbol{W}_k^{(g)}, \quad g = 1, 2, \ldots, g
$$

Head $s$属于group $g(s)$，使用$\boldsymbol{k}_i^{(g(s))}$。

**推导10.2：GQA的KV Cache**

KV Cache大小：
$$
\text{Cache}_{\text{GQA}} = n \cdot g \cdot (d_k + d_v)
$$

对于$g=2$，$d_k = d_v = 128$：
$$
\text{Cache}_{\text{GQA}} = n \cdot 2 \cdot 256 = 512n
$$

这与MLA的Cache大小接近。

**推导10.3：GQA vs MLA的表达能力**

GQA的表达能力受限于group内共享。每个group学习一个K/V表示，所有该组的heads使用相同的K/V。

MLA虽然也共享（通过$\boldsymbol{c}_i$），但每个head从$\boldsymbol{c}_i$独立投影得到$\boldsymbol{k}_i^{(s)}$和$\boldsymbol{v}_i^{(s)}$，提供了更大的灵活性。

形式上，GQA可以看作MLA的特例：
$$
\boldsymbol{W}_k^{(s_1)} = \boldsymbol{W}_k^{(s_2)}, \quad \forall s_1, s_2 \in \text{same group}
$$

但MLA不强制这种约束，允许每个head学习不同的投影。

### 11. 实验结果的理论解释

**推导11.1：增大head_dims的效果**

从实验看，GQA1-256优于GQA2-128。理论解释：

更大的$d_h$意味着每个head有更大的表示空间。对于复杂的注意力模式，需要足够的维度来捕捉。

信息瓶颈理论：head的信息容量为$\mathcal{I} \sim d_h$。更大的$d_h$减少信息损失。

**推导11.2：Partial RoPE的提升**

实验显示GQA1-256-PR优于GQA1-256。从推导8.2，Partial RoPE提供了content和position的解耦：
$$
\text{score} = \text{content\_sim} + \text{position\_bias}
$$

这种解耦使得模型可以独立优化两者，提升了学习效率。

**推导11.3：KV Shared的作用**

实验中GQA2-(192+64)-S2超过GQA1-256-PR，说明KV共享在某些情况下有益。

理论上，KV共享相当于施加了额外的正则化：
$$
\boldsymbol{k}_i \approx \boldsymbol{v}_i
$$

这减少了参数冗余，可能提升泛化能力。

### 12. Q&O LoRA的数学原理

**推导12.1：LoRA的低秩分解**

对于权重矩阵$\boldsymbol{W} \in \mathbb{R}^{d_1 \times d_2}$，LoRA用低秩分解替代：
$$
\boldsymbol{W} = \boldsymbol{W}_0 + \boldsymbol{A} \boldsymbol{B}
$$

其中$\boldsymbol{A} \in \mathbb{R}^{d_1 \times r}$，$\boldsymbol{B} \in \mathbb{R}^{r \times d_2}$，$r \ll \min(d_1, d_2)$。

**推导12.2：参数量的减少**

原始参数量：$P_0 = d_1 \cdot d_2$

LoRA参数量：$P_{\text{LoRA}} = d_1 \cdot r + r \cdot d_2 = r(d_1 + d_2)$

减少量：
$$
\Delta P = d_1 \cdot d_2 - r(d_1 + d_2)
$$

对于$d_1 = d_2 = d$，$r = d/8$：
$$
\Delta P = d^2 - \frac{d}{8}(d + d) = d^2 - \frac{d^2}{4} = \frac{3d^2}{4}
$$

参数量减少75%。

**推导12.3：表达能力的保持**

虽然参数量减少，但如果原始矩阵本身接近低秩，LoRA可以很好地近似：
$$
\|\boldsymbol{W} - \boldsymbol{A}\boldsymbol{B}\|_F \approx \sum_{i=r+1}^{\min(d_1,d_2)} \sigma_i
$$

如果$\sigma_{r+1}, \sigma_{r+2}, \ldots$很小，近似误差可忽略。

### 13. head_dims与num_heads的权衡

**推导13.1：固定总维度下的分配**

给定总维度$D = h \cdot d_h$，可以选择不同的$(h, d_h)$组合。

例如$D = 2048$：
- $(h=16, d_h=128)$：标准配置
- $(h=8, d_h=256)$：更大head_dims
- $(h=32, d_h=64)$：更多heads

**推导13.2：参数量分析**

Q投影参数量：$P_Q = h \cdot d \cdot d_h = d \cdot D$（与分配无关）

但K、V投影参数量受group数影响：
$$
P_{KV} = g \cdot d \cdot (d_k + d_v)
$$

对于MQA（$g=1$）：$P_{KV} = d \cdot 2d_h$

对于GQA（$g=h/2$）：$P_{KV} = \frac{h}{2} \cdot d \cdot 2d_h = h \cdot d \cdot d_h$

**推导13.3：实验验证**

实验结果：
- GQA2-128（32 heads）：loss = 2.723
- GQA2-128（16 heads）：loss = 2.75

更多heads反而略差，支持了"head_dims更重要"的结论。

### 14. MLP缩减的影响

**推导14.1：MLP的参数量**

标准Transformer的MLP：
$$
\boldsymbol{h}_{\text{MLP}} = \text{SwiGLU}(\boldsymbol{h} \boldsymbol{W}_1) \boldsymbol{W}_2
$$

参数量：
$$
P_{\text{MLP}} = d \cdot d_{\text{inter}} + d_{\text{inter}} \cdot d = 2d \cdot d_{\text{inter}}
$$

（SwiGLU需要两个$d \times d_{\text{inter}}$矩阵，合并计算）

**推导14.2：缩减MLP的影响**

将$d_{\text{inter}}$从5456降到4096：
$$
\Delta P = 2d \cdot (5456 - 4096) = 2d \cdot 1360 = 2720d
$$

对于$d=2048$：
$$
\Delta P = 2720 \times 2048 \approx 5.57M
$$

这与GQA1-256和GQA2-128的参数量差接近。

**推导14.3：缩减MLP vs 缩减head_dims**

实验显示：
- GQA1-256（缩小MLP）：loss = 2.747
- GQA2-128（标准MLP）：loss = 2.75

缩小MLP比缩小head_dims更伤效果，说明MLP对容量更关键。

### 15. 训练阶段vs推理阶段的复杂度对比

**推导15.1：训练阶段的瓶颈**

训练时，主要计算量在：
$$
\mathcal{O}_{\text{train}} = n^2 \cdot h \cdot d_h
$$

这是注意力矩阵计算的复杂度，与序列长度$n$的平方成正比。

对于$n=4096$，$h=16$，$d_h=128$：
$$
\mathcal{O}_{\text{train}} \sim 4096^2 \times 16 \times 128 \approx 3.4 \times 10^{10}
$$

**推导15.2：推理阶段的瓶颈**

推理时使用KV Cache，每步计算：
$$
\mathcal{O}_{\text{decode}} = n \cdot h \cdot d_h
$$

但主要瓶颈是内存带宽，需要加载Cache：
$$
\text{Memory\_access} = n \cdot h \cdot (d_k + d_v)
$$

对于$n=4096$，MHA的内存访问：
$$
4096 \times 16 \times 256 \approx 16.8M \text{ 参数}
$$

MLA的内存访问：
$$
4096 \times 576 \approx 2.4M \text{ 参数}
$$

节省约7倍。

**推导15.3：MLA的双重优化**

MLA通过双重投影：
- 训练时：保持$d_h$较小（128），控制$\mathcal{O}_{\text{train}}$
- 推理时：KV Cache压缩到576维，优化内存访问

实现了训练和推理的双重优化。

### 16. 长序列建模的挑战

**推导16.1：长序列的复杂度爆炸**

对于序列长度$n=32K$：
$$
\mathcal{O}_{\text{train}} = (32K)^2 \times h \times d_h \approx 1.07 \times 10^9 \times h \times d_h
$$

相比$n=4K$，计算量增加64倍。

**推导16.2：KV Cache的内存压力**

对于$n=32K$，MHA的KV Cache：
$$
32K \times 16 \times 256 \approx 134M \text{ 参数}
$$

以float16存储，约268MB每个样本。在批处理和多层情况下，内存需求巨大。

**推导16.3：MLA的长序列优势**

MLA的KV Cache：
$$
32K \times 576 \approx 18.4M \text{ 参数}
$$

约37MB每个样本，是MHA的$\frac{18.4}{134} \approx 14\%$。

这使得MLA可以在相同内存下处理更长的序列或更大的批次。

### 17. 总结与理论启示

**推导17.1：MLA的核心数学思想**

MLA本质上是一个低秩分解的Attention机制：
$$
\boldsymbol{K} = \boldsymbol{C} \boldsymbol{W}_k, \quad \boldsymbol{V} = \boldsymbol{C} \boldsymbol{W}_v
$$

通过共享压缩表示$\boldsymbol{C}$，实现了参数和Cache的压缩，同时保持了表达能力。

**推导17.2：关键设计选择的理论基础**

1. **增大head_dims**：提供足够的表示空间，避免信息瓶颈
2. **Partial RoPE**：解耦content和position，提升学习效率
3. **KV Shared**：施加正则化约束，减少冗余，提升泛化

这些设计不是孤立的，而是相互配合，共同实现了MLA的优异性能。

**推导17.3：未来研究方向**

从数学角度，MLA还有改进空间：

1. **动态秩分配**：根据不同层或不同位置，动态调整$d_c$
2. **非线性压缩**：用非线性函数替代线性投影$\boldsymbol{W}_c$
3. **自适应head_dims**：不同heads使用不同的$d_h$

这些方向值得进一步探索。

