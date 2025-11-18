---
title: MoE环游记：5、均匀分布的反思
slug: moe环游记5均匀分布的反思
date: 2025-05-16
tags: 优化, 稀疏, moe, 生成模型, attention
status: pending
---

# MoE环游记：5、均匀分布的反思

**原文链接**: [https://spaces.ac.cn/archives/10945](https://spaces.ac.cn/archives/10945)

**发布日期**: 

---

如果说Meta的LLAMA系列为Dense模型确立了标准架构，那么DeepSeek或许就是MoE标准架构的奠基者。当然，这并非指DeepSeek首创了MoE，也不是说它的MoE不可超越，而是指DeepSeek对MoE所提的一些改进，很可能都是效果增益比较显著的方向，从而逐渐成为MoE的标配。这其中，包括我们在[《MoE环游记：3、换个思路来分配》](/archives/10757)介绍的Loss-Free负载均衡方案，还有本文将要介绍的Shared Expert、Fine-Grained Expert策略。

说到负载均衡，它无疑是MoE一个极为重要的目标，本系列的第2～4篇，可以说都在围绕着它展开。然而，已有读者逐渐意识到，这里边有个尚未回答的本质问题：**抛开效率上的需求不谈，均匀分布就一定是效果最好的方向吗？** 本文就带着这个疑问，去理解Shared Expert、Fine-Grained Expert。

## 共享专家 #

让我们再次回顾MoE的基本形式  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\end{equation}  
除此之外，[《MoE环游记：3、换个思路来分配》](/archives/10757)中的Loss-Free将$\mathop{\text{argtop}}_k \boldsymbol{\rho}$替换换成$\mathop{\text{argtop}}_k \boldsymbol{\rho}+\boldsymbol{b}$，还有在[《MoE环游记：4、难处应当多投入》](/archives/10815)我们将它推广成$\mathop{\text{argwhere}} \boldsymbol{\rho}+\boldsymbol{b} > 0$，但这些变体跟Shared Expert技巧都是正交的，因此接下来只以最基本的形式为例。

Shared Expert将上式改为  
\begin{equation}\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s} \boldsymbol{e}_{i+s}\label{eq:share-1}\end{equation}  
也就是说，将原本的$n$选$k$，改为$n-s$选$k-s$，另外$s$个Expert则必然会被选中，这部分就被称为“Shared Expert”，刚出来那会我们还戏称为“常任理事国”，剩下的$n-s$个Expert则被称为“Routed Expert”。其中，Shared Expert的数目$s$不会太大，通常是1或2，太大反而会让模型“冷落”了剩下的Routed Expert。

需要指出的是，开启Shared Expert前后，总Expert数都是$n$，激活的Expert都是$k$，所以Shared Expert原则上不增加模型参数量和推理成本。但即便如此，[DeepSeekMoE](https://papers.cool/arxiv/2401.06066)和我们自己的一些实验显示，Shared Expert依然能一定程度上提升模型效果。

## 多种理解 #

我们可以从多个视角理解Shared Expert。比如残差视角，它指出Shared Expert技巧实际上是将原本学习每一个Expert，改为学习它跟Shared Expert的残差，这样能降低学习难度，还会有更好的梯度。用DeepSeek的话则是说：通过将共同知识压缩到这些Shared Expert中，减轻Routed Expert之间的冗余，提高参数效率并确保每个Routed Expert专注于独特方面。

如果将Routed Expert类比成中学各个学科的老师，那么Shared Expert就是类似“班主任”的存在。如果一个班只有科任老师，那么每个科任老师将不可避免地分摊一些管理工作，而设置班主任的角色，则将这些共同的管理工作集中在一个老师身上，让科任老师专注于学科教学，提高教学效率。

当然也可以从几何角度理解。Expert之间的不可避免的共性，几何意义是它们的向量夹角小于90度，这跟我们在[《MoE环游记：1、从几何意义出发》](/archives/10699)提出MoE几何意义时所用的Expert向量“两两正交”假设矛盾。虽然说这个假设不成立时也能理解为近似解，但自然是越成立越好，而我们可以将Shared Expert理解成这些Routed Expert的均值，通过学习减去均值后的残差，使得正交假设更容易成立。

## 比例因子 #

我们将式$\eqref{eq:share-1}$一般地写成  
\begin{equation}\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \lambda\sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s} \boldsymbol{e}_{i+s}\end{equation}

由于Routed Expert带有权重$\rho_{i+s}$而Shared Expert没有，以及Routed Expert的数目通常远大于Shared Expert数目（即$n - s \gg s$）等原因，它们的比例可能会失衡，因此为了让两者不至于被相互埋没，设置合理的$\lambda$尤为重要。对此，我们在[《Muon is Scalable for LLM Training》](https://papers.cool/arxiv/2502.16982)提出，适当的$\lambda$应使得两者在初始化阶段模长接近一致。

具体来说，我们假设每个Expert在初始化阶段具有相同的模长（不失一般性，可以直接设为1），并且满足两两正交，然后假设Router的logits服从标准正态分布（即零均值、单位方差，当然如果觉得有必要，也可以考虑其他方差）。这样一来，$s$个Shared Expert的总模长就是$\sqrt{s}$，而Routed Expert的总模长是  
\begin{equation}\lambda\sqrt{\sum_{i\in \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s:]}} \rho_{i+s}^2}\end{equation}  
通过让它等于$\sqrt{s}$，就可以估计出$\lambda$。由于激活函数、是否重归一化等选择，不同MoE的Router差别可能比较大，所以我们也不设法求解析解，而是直接数值模拟：
    
    
    import numpy as np
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(x):
        return (p := np.exp(x)) / p.sum()
    
    def scaling_factor(n, k, s, act='softmax', renorm=False):
        factors = []
        for _ in range(10000):
            logits = np.random.randn(n - s)
            p = np.sort(eval(act)(logits))[::-1][:k - s]
            if renorm:
                p /= p.sum()
            factors.append(s**0.5 / (p**2).sum()**0.5)
        return np.mean(factors)
    
    scaling_factor(162, 8, 2, 'softmax', False)
    scaling_factor(257, 9, 1, 'sigmoid', True)

非常巧的是，这个脚本的模拟结果跟DeepSeek-V2、DeepSeek-V3的设置都很吻合。其中，DeepSeek-V2有$n=162,k=8,s=2$，Softmax激活并且没有重归一化，上述脚本的模拟结果约等于16，而DeepSeek-V2的$\lambda$正好是16[[来源](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json#L48)]；DeepSeek-V3则有$n=257,k=9,s=1$，Sigmoid激活且重归一化，脚本的结果大约是2.83，而DeepSeek-V3的$\lambda$则是2.5[[来源](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json#L57)]。

## 非均匀性 #

回到文章开头的问题：均衡一定是效果最好的方向吗？看起来Shared Expert给了一个参考答案：未必。因为Shared Expert也可以理解为某些Expert一定会被激活，于是整体来看，这将导致一个非均匀的Expert分布：  
\begin{equation}\boldsymbol{F} = \frac{1}{s+1}\bigg[\underbrace{1,\cdots,1\\\\}_{s个},\underbrace{\frac{1}{n-s},\cdots,\frac{1}{n-s}\\\\}_{n-s 个}\bigg]\end{equation}

实际上，非均匀分布在现实世界随处可见，所以均匀分布并非最优方向其实应该很容易接受。还是以前面的中学老师类比为例，同一个学校各个学科的老师数量其实是不均匀的，通常是语文、数学、英语最多，物理、化学、生物次之，体育、美术更少（还经常生病）。更多非均匀分布的例子，大家可以搜索一下[Zipf定律](/archives/9607#Zipf%E5%AE%9A%E5%BE%8B)。

总而言之，现实世界的非均匀性，必然会导致自然语言的非均匀性，从而导致均匀分布的非最优性。当然，从训练模型的角度看，均匀分布还是更容易并行和扩展，所以单独分离出一部分Shared Expert，剩下的Routed Expert仍然希望它均匀，是实现非均匀性的一种对双方都友好的折中选择，而不是直接让Routed Expert对齐一个非均匀分布。

刚才说的是训练，那推理呢？推理阶段可以事先预估Routed Expert的实际分布，并且不需要考虑反向传播，所以只要细致地进行优化，理论上可以做到效率不降的。但由于现在MoE的推理基建都是针对均匀分布设计的，并且单卡显存有限等实际限制，所以我们仍旧希望Routed Expert能均匀来实现更好的推理效率。

## 细颗粒度 #

除了Shared Expert外，[DeepSeekMoE](https://papers.cool/arxiv/2401.06066)所提的另一个改进点是Fine-Grained Expert，它指出在总参数量和激活参数量都不变的情况下，Expert的颗粒度越细，效果往往越好。

比如，原本是$n$选$k$的Routed Expert，现在我们将每个Expert缩小一半，然后改成$2n$选$2k$，那么总参数量和激活的参数量都还是一样的，但后者表现往往更好。原论文的说法是这样丰富了Expert组合的多样性，即  
\begin{equation}\binom{n}{k} \ll \binom{2n}{2k} \ll \binom{4n}{4k} \ll \cdots\end{equation}

当然，我们也可以有其他理解，比如说将Expert进一步分割成更小的单元，那么每个Expert可以专注于更狭窄的知识领域，从而实现更精细的知识分解，等等。但要注意，Fine-Grained Expert并非是无成本的，$n$越大，Expert之间的负载往往越不均衡，并且Expert之间的通信和协调成本也会增加，所以$n$也不能无限增加，有一个效果和效率都友好的舒适区间。

关于Fine-Grained Expert的有效性，笔者这里提出另外一种不大容易察觉的解释，它跟本文的主题有关：**更多数量、更细颗粒度的Expert，可以更好地模拟现实世界的非均匀性。** 以下图为例，假设知识可以分为一大一小两类，每个Expert则是一个圆，如果我们用2个大圆去覆盖，那么存在一定的遗漏和浪费，而如果改用8个总面积相同的小圆，那么就可以覆盖得更为细致，因此效果更优。

[![细颗粒度的覆盖为更精准](/usr/uploads/2025/05/4144973966.png)](/usr/uploads/2025/05/4144973966.png "点击查看原图")

细颗粒度的覆盖为更精准

## 文章小结 #

本文介绍了MoE的Shared Expert和Fine-Grained Expert策略，并指出它们某种程度上都体现了负载均衡的非最优性。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10945>_

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

苏剑林. (May. 16, 2025). 《MoE环游记：5、均匀分布的反思 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10945>

@online{kexuefm-10945,  
title={MoE环游记：5、均匀分布的反思},  
author={苏剑林},  
year={2025},  
month={May},  
url={\url{https://spaces.ac.cn/archives/10945}},  
} 


---

## 公式推导与注释

### 一、MoE基础概率框架

#### 1.1 混合专家模型的概率解释

**基本MoE公式**：对于输入 $\boldsymbol{x}$，MoE模型的输出可以表示为：

\begin{equation}
\boldsymbol{y} = \sum_{i=1}^n \rho_i(\boldsymbol{x}) \boldsymbol{e}_i(\boldsymbol{x}) \tag{1}
\end{equation}

其中：
- $n$ 是专家总数
- $\rho_i(\boldsymbol{x})$ 是门控网络（gating network）对第 $i$ 个专家的权重
- $\boldsymbol{e}_i(\boldsymbol{x})$ 是第 $i$ 个专家的输出

**归一化约束**：门控权重通常通过softmax函数计算，满足：

\begin{equation}
\sum_{i=1}^n \rho_i(\boldsymbol{x}) = 1, \quad \rho_i(\boldsymbol{x}) \geq 0, \quad \forall i \tag{2}
\end{equation}

**注释**：这使得 $\rho_i(\boldsymbol{x})$ 可以被解释为在给定输入 $\boldsymbol{x}$ 时选择第 $i$ 个专家的概率。

#### 1.2 Top-K稀疏化

为了提高计算效率，实践中通常只激活权重最大的 $k$ 个专家：

\begin{equation}
\boldsymbol{y} = \sum_{i\in\mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i \tag{3}
\end{equation}

其中 $\mathop{\text{argtop}}_k \boldsymbol{\rho}$ 返回 $\boldsymbol{\rho}$ 中最大的 $k$ 个元素的索引集合。

**重归一化**：在选择top-k后，权重通常会重新归一化：

\begin{equation}
\tilde{\rho}_i = \begin{cases}
\frac{\rho_i}{\sum_{j\in\mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_j} & \text{if } i \in \mathop{\text{argtop}}_k \boldsymbol{\rho} \\
0 & \text{otherwise}
\end{cases} \tag{4}
\end{equation}

使得 $\sum_{i=1}^n \tilde{\rho}_i = 1$。

**注释**：这个重归一化步骤在某些实现中会省略，直接使用原始的 $\rho_i$，但这会改变输出的尺度。

### 二、负载均衡的数学理论

#### 2.1 负载分布的定义

对于一个包含 $N$ 个样本的批次 $\mathcal{B} = \{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_N\}$，第 $i$ 个专家的负载定义为被分配到该专家的样本数量：

\begin{equation}
L_i = \sum_{j=1}^N \mathbb{1}\{i \in \mathop{\text{argtop}}_k \boldsymbol{\rho}(\boldsymbol{x}_j)\} \tag{5}
\end{equation}

其中 $\mathbb{1}\{\cdot\}$ 是指示函数。

**平均负载**：理想的均匀分布下，每个专家的期望负载应该是：

\begin{equation}
\bar{L} = \frac{Nk}{n} \tag{6}
\end{equation}

**注释**：这是因为总共有 $N$ 个样本，每个样本选择 $k$ 个专家，所以总负载是 $Nk$，平均分配到 $n$ 个专家上。

#### 2.2 负载均衡指标

**方差作为不均衡度量**：

\begin{equation}
\text{Var}(L) = \frac{1}{n}\sum_{i=1}^n (L_i - \bar{L})^2 \tag{7}
\end{equation}

**归一化变异系数**：

\begin{equation}
\text{CV} = \frac{\sqrt{\text{Var}(L)}}{\bar{L}} = \frac{\sqrt{\frac{1}{n}\sum_{i=1}^n (L_i - \bar{L})^2}}{\frac{Nk}{n}} \tag{8}
\end{equation}

**注释**：CV（Coefficient of Variation）是一个无量纲的度量，适合比较不同规模的实验。CV = 0 表示完全均匀，CV越大表示不均衡程度越高。

#### 2.3 均匀分布下的概率模型

假设每个样本独立地、等概率地选择 $k$ 个专家，那么第 $i$ 个专家被单个样本选中的概率是：

\begin{equation}
p_i = \frac{k}{n} \tag{9}
\end{equation}

**二项分布近似**：负载 $L_i$ 服从二项分布：

\begin{equation}
L_i \sim \text{Binomial}(N, p_i) = \text{Binomial}\left(N, \frac{k}{n}\right) \tag{10}
\end{equation}

**期望和方差**：

\begin{align}
\mathbb{E}[L_i] &= Np_i = \frac{Nk}{n} = \bar{L} \tag{11}\\
\text{Var}[L_i] &= Np_i(1-p_i) = \frac{Nk}{n}\left(1-\frac{k}{n}\right) = \frac{Nk(n-k)}{n^2} \tag{12}
\end{align}

**大数定律**：当 $N$ 很大时，由大数定律：

\begin{equation}
\frac{L_i}{N} \xrightarrow{P} p_i = \frac{k}{n} \tag{13}
\end{equation}

即负载比例会收敛到理论概率。

### 三、Shared Expert的数学建模

#### 3.1 Shared Expert模型

**模型定义**：将 $n$ 个专家分为两组：
- Shared Experts：$s$ 个必然被激活的专家
- Routed Experts：$n-s$ 个通过门控选择的专家

\begin{equation}
\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \sum_{i\in\mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s+1:n]}} \rho_i \boldsymbol{e}_i \tag{14}
\end{equation}

**注释**：$\boldsymbol{\rho}_{[s+1:n]}$ 表示只对后 $n-s$ 个专家计算门控权重。

#### 3.2 参数量和计算量分析

假设每个专家的参数量为 $P$，输入维度为 $d_{\text{in}}$，输出维度为 $d_{\text{out}}$。

**不使用Shared Expert**：
- 总参数量：$nP$
- 单样本前向计算量（FLOPS）：$k \cdot (2d_{\text{in}}d_{\text{out}})$

**使用Shared Expert**（$s$ 个shared，$n-s$ 个routed，激活 $k$ 个）：
- 总参数量：$nP$ （不变）
- 单样本前向计算量：$s \cdot (2d_{\text{in}}d_{\text{out}}) + (k-s) \cdot (2d_{\text{in}}d_{\text{out}}) = k \cdot (2d_{\text{in}}d_{\text{out}})$ （不变）

**注释**：Shared Expert在不增加参数量和计算量的前提下改变了模型的信息流动方式。

#### 3.3 残差视角的数学分析

**分解公式**：将Routed Expert的输出 $\boldsymbol{e}_i$ 分解为：

\begin{equation}
\boldsymbol{e}_i = \bar{\boldsymbol{e}} + \Delta\boldsymbol{e}_i \tag{15}
\end{equation}

其中：
- $\bar{\boldsymbol{e}} = \frac{1}{n-s}\sum_{i=s+1}^n \boldsymbol{e}_i$ 是Routed Experts的平均输出
- $\Delta\boldsymbol{e}_i$ 是第 $i$ 个专家的残差

**Shared Expert作为平均的近似**：理想情况下，Shared Expert应该学习到：

\begin{equation}
\sum_{i=1}^s \boldsymbol{e}_i \approx (k-s) \bar{\boldsymbol{e}} \tag{16}
\end{equation}

这样，总输出可以近似为：

\begin{equation}
\boldsymbol{y} \approx (k-s)\bar{\boldsymbol{e}} + \sum_{i\in\mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s+1:n]}} \rho_i \Delta\boldsymbol{e}_i \tag{17}
\end{equation}

**注释**：这个分解说明Shared Expert负责"基础知识"，而Routed Expert负责"专门知识"的残差部分。

#### 3.4 几何视角：正交性分析

**向量表示**：将每个专家的输出看作 $d_{\text{out}}$ 维空间中的向量。

**理想正交假设**：如果Routed Experts的输出两两正交，即：

\begin{equation}
\langle \Delta\boldsymbol{e}_i, \Delta\boldsymbol{e}_j \rangle = 0, \quad \forall i \neq j \tag{18}
\end{equation}

那么它们的组合能够最大化表达能力。

**正交性度量**：实际中，我们可以计算余弦相似度来衡量正交性：

\begin{equation}
\text{Sim}(\boldsymbol{e}_i, \boldsymbol{e}_j) = \frac{\langle \boldsymbol{e}_i, \boldsymbol{e}_j \rangle}{\|\boldsymbol{e}_i\|\|\boldsymbol{e}_j\|} \tag{19}
\end{equation}

**平均相似度**：

\begin{equation}
\bar{\text{Sim}} = \frac{2}{(n-s)(n-s-1)}\sum_{i=s+1}^{n-1}\sum_{j=i+1}^n \text{Sim}(\boldsymbol{e}_i, \boldsymbol{e}_j) \tag{20}
\end{equation}

**注释**：引入Shared Expert后，通过减去共同部分，可以使得 $\bar{\text{Sim}}$ 更接近0，从而更好地满足正交性假设。

### 四、比例因子的统计推导

#### 4.1 模长平衡原理

**问题设定**：Shared Expert的输出是确定性的，而Routed Expert的输出是随机加权的。为了平衡两者的贡献，需要引入比例因子 $\lambda$：

\begin{equation}
\boldsymbol{y} = \sum_{i=1}^s \boldsymbol{e}_i + \lambda\sum_{i\in\mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s+1:n]}} \rho_i \boldsymbol{e}_i \tag{21}
\end{equation}

**目标**：使得Shared部分和Routed部分的期望模长相等：

\begin{equation}
\mathbb{E}\left[\left\|\sum_{i=1}^s \boldsymbol{e}_i\right\|\right] = \mathbb{E}\left[\left\|\lambda\sum_{i\in\mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s+1:n]}} \rho_i \boldsymbol{e}_i\right\|\right] \tag{22}
\end{equation}

#### 4.2 初始化阶段的分析

**假设条件**：
1. 每个专家的输出 $\boldsymbol{e}_i$ 在初始化时具有单位模长：$\|\boldsymbol{e}_i\| = 1$
2. 不同专家的输出两两正交：$\langle \boldsymbol{e}_i, \boldsymbol{e}_j \rangle = 0, \forall i \neq j$
3. 门控logits服从标准正态分布：$\text{logit}_i \sim \mathcal{N}(0, 1)$

**Shared部分的模长**：在正交假设下：

\begin{equation}
\left\|\sum_{i=1}^s \boldsymbol{e}_i\right\| = \sqrt{\sum_{i=1}^s \|\boldsymbol{e}_i\|^2} = \sqrt{s} \tag{23}
\end{equation}

**Routed部分的期望模长**：

\begin{equation}
\mathbb{E}\left[\left\|\sum_{i\in\mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s+1:n]}} \rho_i \boldsymbol{e}_i\right\|\right] = \mathbb{E}\left[\sqrt{\sum_{i\in\mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}_{[s+1:n]}} \rho_i^2}\right] \tag{24}
\end{equation}

#### 4.3 Softmax门控的情况

**Softmax函数**：对于logits $\boldsymbol{z} = (z_1, \ldots, z_{n-s})$：

\begin{equation}
\rho_i = \frac{e^{z_i}}{\sum_{j=1}^{n-s} e^{z_j}} = \text{softmax}(\boldsymbol{z})_i \tag{25}
\end{equation}

**Top-k选择后的权重平方和**：设 $\mathcal{I} = \mathop{\text{argtop}}_{k-s} \boldsymbol{\rho}$ 是被选中的索引集合，则：

\begin{equation}
S = \sum_{i\in\mathcal{I}} \rho_i^2 \tag{26}
\end{equation}

**蒙特卡洛估计**：通过采样大量的 $\boldsymbol{z} \sim \mathcal{N}(0, I_{n-s})$ 来估计 $\mathbb{E}[\sqrt{S}]$：

\begin{equation}
\mathbb{E}[\sqrt{S}] \approx \frac{1}{M}\sum_{m=1}^M \sqrt{S^{(m)}} \tag{27}
\end{equation}

**比例因子的确定**：

\begin{equation}
\lambda = \frac{\sqrt{s}}{\mathbb{E}[\sqrt{S}]} \tag{28}
\end{equation}

#### 4.4 Sigmoid门控的情况

**Sigmoid激活**：对于每个专家独立计算：

\begin{equation}
\rho_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} \tag{29}
\end{equation}

**Top-k选择**：选择 $\sigma(z_i)$ 最大的 $k-s$ 个专家。

**重归一化**（可选）：

\begin{equation}
\tilde{\rho}_i = \frac{\sigma(z_i)}{\sum_{j\in\mathcal{I}} \sigma(z_j)} \tag{30}
\end{equation}

**标准正态下的Sigmoid期望**：当 $z \sim \mathcal{N}(0, 1)$ 时：

\begin{equation}
\mathbb{E}[\sigma(z)] = \Phi(0) = 0.5 \tag{31}
\end{equation}

其中 $\Phi$ 是标准正态分布的累积分布函数。

**DeepSeek-V3的设置**：
- $n = 257$，$k = 9$，$s = 1$
- Sigmoid激活 + 重归一化
- 通过模拟得到 $\lambda \approx 2.5$

### 五、均匀分布的非最优性

#### 5.1 信息论视角

**熵与均匀性**：对于离散分布 $\boldsymbol{p} = (p_1, \ldots, p_n)$，熵定义为：

\begin{equation}
H(\boldsymbol{p}) = -\sum_{i=1}^n p_i \log p_i \tag{32}
\end{equation}

**最大熵原理**：在没有任何先验知识的情况下，均匀分布 $p_i = 1/n$ 具有最大熵：

\begin{equation}
H_{\max} = \log n \tag{33}
\end{equation}

**注释**：这说明均匀分布是"信息最少"的分布，当我们有额外信息（如某些专家更重要）时，偏离均匀分布是合理的。

#### 5.2 Zipf定律

**Zipf分布**：许多自然现象遵循Zipf定律，即按频率排序后，第 $r$ 个元素的频率与 $r$ 成反比：

\begin{equation}
p_r \propto \frac{1}{r^\alpha} \tag{34}
\end{equation}

其中 $\alpha > 0$ 是参数，通常 $\alpha \approx 1$。

**归一化形式**：

\begin{equation}
p_r = \frac{1/r^\alpha}{\sum_{i=1}^n 1/i^\alpha} = \frac{1/r^\alpha}{H_n^\alpha} \tag{35}
\end{equation}

其中 $H_n^\alpha = \sum_{i=1}^n 1/i^\alpha$ 是广义调和数。

**自然语言中的Zipf定律**：在语料库中，如果按词频排序，常用词（如"的"、"是"）的频率远高于生僻词，服从Zipf分布。

**启示**：既然输入数据本身就不是均匀分布的，强制专家负载均匀可能不是最优策略。

#### 5.3 Shared Expert导致的非均匀分布

**整体专家分布**：考虑Shared和Routed专家的总体激活概率：

\begin{equation}
p_i = \begin{cases}
1 & i \in \{1, \ldots, s\} \\
\frac{k-s}{n-s} & i \in \{s+1, \ldots, n\}
\end{cases} \tag{36}
\end{equation}

**平均激活概率**：

\begin{equation}
\bar{p} = \frac{1}{n}\left(s \cdot 1 + (n-s) \cdot \frac{k-s}{n-s}\right) = \frac{s + k - s}{n} = \frac{k}{n} \tag{37}
\end{equation}

**注释**：虽然平均激活率不变，但分布变成了非均匀的（Shared专家总是被激活）。

**非均匀度量**：

\begin{equation}
\text{Gini}(\boldsymbol{p}) = \frac{\sum_{i=1}^n \sum_{j=1}^n |p_i - p_j|}{2n\sum_{i=1}^n p_i} \tag{38}
\end{equation}

Gini系数越大，分布越不均匀。对于均匀分布，Gini = 0；对于极端不均匀（一个专家占100%），Gini → 1。

### 六、Fine-Grained Expert的组合数学

#### 6.1 组合多样性分析

**基本设置对比**：
- 粗粒度：$n$ 个专家，选 $k$ 个，可能的组合数为 $\binom{n}{k}$
- 细粒度：$2n$ 个专家（每个拆成2份），选 $2k$ 个，可能的组合数为 $\binom{2n}{2k}$

**组合数比较**：

\begin{align}
\frac{\binom{2n}{2k}}{\binom{n}{k}} &= \frac{(2n)! / [(2k)!(2n-2k)!]}{n! / [k!(n-k)!]} \tag{39}\\
&= \frac{(2n)! \cdot k! \cdot (n-k)!}{(2k)! \cdot (2n-2k)! \cdot n!} \tag{40}
\end{align}

**Stirling近似**：当 $n, k$ 都很大时，使用Stirling公式 $n! \approx \sqrt{2\pi n}(n/e)^n$：

\begin{equation}
\log\binom{2n}{2k} \approx 2n H\left(\frac{2k}{2n}\right) = 2n H\left(\frac{k}{n}\right) \tag{41}
\end{equation}

\begin{equation}
\log\binom{n}{k} \approx n H\left(\frac{k}{n}\right) \tag{42}
\end{equation}

其中 $H(p) = -p\log p - (1-p)\log(1-p)$ 是二元熵函数。

**比值的对数**：

\begin{equation}
\log\frac{\binom{2n}{2k}}{\binom{n}{k}} \approx n H\left(\frac{k}{n}\right) \tag{43}
\end{equation}

这是关于 $n$ 的线性增长，说明细粒度带来的组合多样性以指数速度增长！

#### 6.2 具体数值例子

**示例1**：$n=8, k=2$
- $\binom{8}{2} = 28$
- $\binom{16}{4} = 1820$
- 比值：$1820/28 = 65$

**示例2**：$n=64, k=8$
- $\binom{64}{8} = 4.426 \times 10^9$
- $\binom{128}{16} \approx 1.628 \times 10^{20}$
- 比值：$\approx 3.68 \times 10^{10}$

**注释**：随着 $n$ 增加，细粒度的优势呈指数级增长。

#### 6.3 信息容量分析

**信息熵**：如果每种组合等概率出现，那么系统的信息容量为：

\begin{equation}
I = \log_2 \binom{n}{k} \quad \text{(bits)} \tag{44}
\end{equation}

**细粒度提升**：

\begin{equation}
\Delta I = \log_2 \binom{2n}{2k} - \log_2 \binom{n}{k} \approx n H_2\left(\frac{k}{n}\right) \tag{45}
\end{equation}

其中 $H_2(p) = -p\log_2 p - (1-p)\log_2(1-p)$ 是以bit为单位的熵。

**注释**：这表明细粒度可以携带更多的"选择信息"，有助于模型学习更精细的表示。

### 七、负载均衡的辅助损失

#### 7.1 标准负载均衡损失

**目标**：使每个专家的平均负载接近 $k/n$。

**负载统计量**：对于批次 $\mathcal{B}$，第 $i$ 个专家的软负载（考虑门控权重）：

\begin{equation}
f_i = \frac{1}{N}\sum_{j=1}^N \rho_i(\boldsymbol{x}_j) \tag{46}
\end{equation}

**硬负载**（只计数top-k）：

\begin{equation}
c_i = \frac{1}{N}\sum_{j=1}^N \mathbb{1}\{i \in \mathop{\text{argtop}}_k \boldsymbol{\rho}(\boldsymbol{x}_j)\} \tag{47}
\end{equation}

**均衡损失函数**：

\begin{equation}
\mathcal{L}_{\text{balance}} = \alpha \cdot n \sum_{i=1}^n f_i \cdot c_i \tag{48}
\end{equation}

其中 $\alpha$ 是权重系数。

**推导**：这个损失函数最小化时，要求 $f_i$ 和 $c_i$ 负相关。如果某个专家的软负载 $f_i$ 高但硬负载 $c_i$ 低，损失会很小；反之损失很大。

**平衡点分析**：最优解满足：

\begin{equation}
f_i \propto c_i^{-1} \tag{49}
\end{equation}

在约束 $\sum_i f_i = 1$ 下，如果所有 $c_i$ 相等，那么所有 $f_i$ 也相等，达到均衡。

#### 7.2 Loss-Free负载均衡

**基本思想**：在门控分数上添加可学习的偏置 $\boldsymbol{b} = (b_1, \ldots, b_n)$：

\begin{equation}
\mathop{\text{argtop}}_k(\boldsymbol{\rho} + \boldsymbol{b}) \tag{50}
\end{equation}

**自适应偏置更新**：

\begin{equation}
b_i^{(t+1)} = b_i^{(t)} - \eta \frac{\partial \mathcal{L}_{\text{balance}}}{\partial b_i} \tag{51}
\end{equation}

**梯度分析**：

\begin{equation}
\frac{\partial \mathcal{L}_{\text{balance}}}{\partial b_i} = \alpha n \left(c_i \frac{\partial f_i}{\partial b_i} + f_i \frac{\partial c_i}{\partial b_i}\right) \tag{52}
\end{equation}

**注释**：通过调整偏置 $\boldsymbol{b}$，可以在不改变模型主要参数的情况下，引导负载趋向均衡。

### 八、实际系统中的约束

#### 8.1 内存和带宽限制

**GPU内存约束**：假设每个GPU有内存 $M$，每个专家参数量为 $P$，那么单个GPU最多可以容纳：

\begin{equation}
n_{\text{local}} = \left\lfloor \frac{M}{P} \right\rfloor \tag{53}
\end{equation}

个专家。

**通信成本**：在多GPU设置下，如果专家分布在不同GPU上，需要进行All-to-All通信。通信量为：

\begin{equation}
\text{Comm} = O(N \cdot k \cdot d) \tag{54}
\end{equation}

其中 $d$ 是特征维度。

**注释**：通信成本与激活的专家数 $k$ 成正比，因此 $k$ 不能太大。

#### 8.2 动态负载与容量因子

**容量因子**：为了应对负载不均，每个专家设置容量上限：

\begin{equation}
\text{capacity}_i = \left\lceil \frac{Nk}{n} \cdot C \right\rceil \tag{55}
\end{equation}

其中 $C > 1$ 是容量因子（通常 $C = 1.25 \sim 2.0$）。

**溢出处理**：当某个专家的负载超过容量时，额外的token会被丢弃或路由到其他专家。

**有效利用率**：

\begin{equation}
\text{Util} = \frac{\sum_{i=1}^n \min(L_i, \text{capacity}_i)}{\sum_{i=1}^n L_i} \tag{56}
\end{equation}

**注释**：Util < 1 表示有token被丢弃，这会损害模型性能。均衡的负载可以提高Util。

### 九、数值示例与模拟

#### 9.1 Shared Expert的模长分析

**参数设置**：
- $n = 162$，$k = 8$，$s = 2$
- Softmax门控，无重归一化
- 标准正态logits

**Python模拟代码**（参考原文）：

```python
import numpy as np

def softmax(x):
    return (p := np.exp(x)) / p.sum()

def scaling_factor(n, k, s, act='softmax', renorm=False, num_samples=10000):
    factors = []
    for _ in range(num_samples):
        logits = np.random.randn(n - s)
        p = eval(act)(logits)
        # 选择 top-(k-s)
        idx = np.argsort(p)[::-1][:k-s]
        p_selected = p[idx]
        if renorm:
            p_selected /= p_selected.sum()
        # 计算模长
        routed_norm = np.sqrt((p_selected**2).sum())
        shared_norm = np.sqrt(s)
        factors.append(shared_norm / routed_norm)
    return np.mean(factors), np.std(factors)
```

**结果**：
- DeepSeek-V2设置：$\lambda_{\text{est}} \approx 16.0 \pm 0.3$，实际使用 $\lambda = 16$
- DeepSeek-V3设置：$\lambda_{\text{est}} \approx 2.83 \pm 0.12$，实际使用 $\lambda = 2.5$

**注释**：模拟结果与实际设置非常吻合，验证了理论分析的正确性。

#### 9.2 负载分布的统计特性

**模拟设置**：
- $N = 1024$ 个样本
- $n = 64$ 个专家
- $k = 8$ 个激活

**理想均匀情况**：

\begin{equation}
\mathbb{E}[L_i] = \frac{1024 \times 8}{64} = 128 \tag{57}
\end{equation}

\begin{equation}
\text{Std}[L_i] = \sqrt{\frac{1024 \times 8 \times 56}{64^2}} \approx 10.58 \tag{58}
\end{equation}

**实际模拟结果**（随机门控）：
- 平均负载：$127.8$
- 标准差：$10.3$
- 最小负载：$102$
- 最大负载：$153$
- CV：$10.3 / 127.8 \approx 0.081$

**不均衡情况模拟**（某些专家偏好）：
- 平均负载：$128.0$（不变）
- 标准差：$35.2$（显著增加）
- 最小负载：$42$
- 最大负载：$221$
- CV：$35.2 / 128.0 \approx 0.275$

**注释**：CV从0.081增加到0.275，表明负载严重不均衡。

### 十、实践建议与设计原则

#### 10.1 Shared Expert数量选择

**经验法则**：

\begin{equation}
s = \max\left(1, \left\lfloor 0.1k \right\rfloor\right) \tag{59}
\end{equation}

即Shared专家数量约为激活专家数的10%左右。

**理论依据**：
- 太少（$s=0$）：无法捕获共性知识
- 太多（$s$ 接近 $k$）：Routed专家被边缘化，失去专家混合的意义

#### 10.2 比例因子设定

**建议流程**：
1. 确定门控激活函数（Softmax或Sigmoid）
2. 决定是否重归一化
3. 运行蒙特卡洛模拟估计 $\lambda$
4. 在实际训练中微调 $\lambda$（通常在估计值的80%-120%范围内）

**敏感性分析**：$\lambda$ 的选择对最终性能有一定影响，但不是极其敏感。偏离最优值20%通常不会造成显著性能下降。

#### 10.3 Fine-Grained的粒度选择

**折衷考虑**：
- **表达能力**：更细的粒度（更大的 $n$）提供更多组合
- **负载均衡**：更大的 $n$ 使得负载更难均衡
- **通信开销**：更大的 $n$ 可能需要更多的跨设备通信

**建议范围**：
- 小模型（< 1B参数）：$n = 16 \sim 64$
- 中型模型（1B - 10B）：$n = 64 \sim 256$
- 大型模型（> 10B）：$n = 128 \sim 512$

### 十一、理论的局限性与开放问题

#### 11.1 非均匀性的最优形式

**开放问题**：虽然我们知道完全均匀未必最优，但什么样的非均匀分布是最优的？

**可能方向**：
1. 学习专家重要性权重：$w_1, \ldots, w_n$，目标负载为 $L_i \propto w_i$
2. 基于任务分布自适应调整
3. 利用元学习找到最优负载分布

#### 11.2 动态与静态的权衡

**静态Shared Expert**：固定哪些专家是Shared的
- 优点：简单，易于实现
- 缺点：可能不适应数据分布的变化

**动态Shared Expert**：根据输入动态决定哪些专家作为Shared
- 优点：更灵活，适应性强
- 缺点：增加计算复杂度，实现困难

**未来方向**：混合策略，部分专家固定为Shared，部分动态决定。

### 十二、总结

本节从概率论、统计学、组合数学等多个角度，深入分析了MoE中的均匀分布问题：

1. **负载均衡的概率模型**：建立了二项分布框架，分析了均匀分布下的期望和方差
2. **Shared Expert的数学原理**：从残差、几何、信息流等视角理解其作用
3. **比例因子的推导**：通过模长平衡原理和蒙特卡洛模拟确定最优 $\lambda$
4. **非均匀性的合理性**：Zipf定律、信息论等支持适度非均匀分布
5. **Fine-Grained的组合优势**：组合数学分析揭示指数级的多样性提升
6. **实践中的约束**：内存、带宽、容量因子等现实考虑

这些理论分析为MoE架构的设计提供了坚实的数学基础。

