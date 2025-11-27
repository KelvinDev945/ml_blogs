---
title: 预训练一下，Transformer的长序列成绩还能涨不少！
slug: 预训练一下transformer的长序列成绩还能涨不少
date: 2023-10-08
tags: 语言模型, attention, 生成模型, attention, 优化
status: completed
---

# 预训练一下，Transformer的长序列成绩还能涨不少！

**原文链接**: [https://spaces.ac.cn/archives/9787](https://spaces.ac.cn/archives/9787)

**发布日期**: 

---

作为LLM的主流模型架构，Transformer在各类任务上的总体表现都出色，大多数情况下，Transformer的槽点只是它的平方复杂度，而不是效果——除了一个名为Long Range Arena（下面简称LRA）的Benchmark。一直以来，LRA一直是线性RNN类模型的“主场”，与之相比Transformer在上面有明显的差距，以至于让人怀疑这是否就是Transformer的固有缺陷。

不过，近日论文[《Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors》](https://papers.cool/arxiv/2310.02980)将这“缺失的一环”给补齐了。论文指出，缺乏预训练是Transformer在LRA上效果较差的主要原因，而所有架构都可以通过预训练获得一定的提升，Transformer的提升则更为明显。

## 旧背景 #

Long Range Arena（LRA）是长序列建模的一个Benchmark，提出自论文[《Long Range Arena: A Benchmark for Efficient Transformers》](https://papers.cool/arxiv/2011.04006)，从论文标题就可以看出，LRA是为了测试各种Efficient版的Transformer而构建的，里边包含了多种类型的数据，序列长度从1k到16k不等，此前不少Efficient Transformer的工作也都在LRA进行了测试。虽然在代表性方面有些争议，但LRA依然不失为一个测试Efficient Transformer的长序列能力的经典Benchmark。

[![MEGA论文中的LRA结果](/usr/uploads/2023/10/3692059662.png)](/usr/uploads/2023/10/3692059662.png "点击查看原图")

MEGA论文中的LRA结果

可能会让部分读者意外的是，标准的Transformer（XFM）在这个Benchmark上的成绩并不出色，明显落后于一系列线性RNN类模型，比如经典的SSM（[S4](https://papers.cool/arxiv/2111.00396)、[S4D](https://papers.cool/arxiv/2203.14343)、[S5](https://papers.cool/arxiv/2208.04933)）或者此前我们介绍过的[LRU](/archives/9554)，甚至于此前的SOTA模型[MEGA](https://papers.cool/arxiv/2209.10655)，也需要在[GAU](/archives/8934)的基础上装备线性RNN模块（论文里边称为EMA）。总而言之，此前LRA上的模型排行情况，强烈地透露着“Attention可以有，但RNN必不可少”的信号。

**（注：LRA的完整成绩排行可以在<https://paperswithcode.com/sota/long-range-modeling-on-lra> 查阅。）**

## 新结论 #

很明显，[《Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors》](https://papers.cool/arxiv/2310.02980)的出现打破了这一印象，它指出用训练集预训练就可以大大缩小两者的差距，并进一步提出“无预训练，不公平”的观点。

[![“Transformer+预训练”相比于Transformer及各种Effective版的提升](/usr/uploads/2023/10/1358414937.png)](/usr/uploads/2023/10/1358414937.png "点击查看原图")

“Transformer+预训练”相比于Transformer及各种Effective版的提升

预训练的做法很简单，任务选择MLM或者GPT都可以，数据集则还是原本的训练集，这样一来除了增加了算力消耗外，并没有引入额外的知识来源，所以比较是公平的。事实上，不管是Transformer还是RNN，经过预训练之后都能获得明显的提升，只不过Transformer的提升更加明显：  


[![“Transformer+预训练”与“S4+预训练”](/usr/uploads/2023/10/546916399.png)](/usr/uploads/2023/10/546916399.png "点击查看原图")

“Transformer+预训练”与“S4+预训练”

[![与SOTA模型的对比](/usr/uploads/2023/10/476588374.png)](/usr/uploads/2023/10/476588374.png "点击查看原图")

与SOTA模型的对比

事后来看，论文的结论并不让人意外，甚至有点“显然成立”的感觉，但此前大家似乎都没往这个方向去想（或者是想到了但觉得不是关键？），所以作者们首先意识到并证明预训练在LRA的重要性，依然是非常值得称赞的。

预训练的重要性实际上表明了Inductive Bias在LRA上的重要性，因为LRA为了使得序列足够Long，它的token颗粒度是非常细的，比如文本任务是以字母为token的，图像任务是以像素为token并直接将二维图像展平为一维序列的，很明显这些任务既需要远程依赖，又有明显的局域性，线性RNN正好非常贴合它的特性。而Transformer相对来说没有那么明显的Inductive Bias，它还需要额外加位置编码才有位置信息，而即便加了也没有显著的局域性，因此更需要预训练来适应数据特性，或者说，通过预训练来补充Inductive Bias。

## 全剧终 #

本文跟大家快速分享了一个较新的实验结论，即预训练能有效提高各种模型在LRA上的成绩，尤其是Transformer经过预训练之后，效果基本上也能接近SOTA梯队，这打破了笔者一直以来LRA必须要加线性RNN的印象。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9787>_

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

苏剑林. (Oct. 08, 2023). 《预训练一下，Transformer的长序列成绩还能涨不少！ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9787>

@online{kexuefm-9787,  
title={预训练一下，Transformer的长序列成绩还能涨不少！},  
author={苏剑林},  
year={2023},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/9787}},  
} 


---

## 公式推导与注释

### 一、Transformer长序列建模的数学基础

#### 1.1 长序列的复杂度问题

标准Transformer的自注意力机制的时间和空间复杂度为：

\begin{equation}
\text{Time}(\text{Attention}) = \mathcal{O}(L^2 d)
\tag{1}
\end{equation}

\begin{equation}
\text{Space}(\text{Attention}) = \mathcal{O}(L^2)
\tag{2}
\end{equation}

其中 $L$ 是序列长度，$d$ 是模型维度。

**瓶颈分析**: 当 $L$ 从2K增加到16K时：

\begin{equation}
\frac{\text{Cost}_{L=16K}}{\text{Cost}_{L=2K}} = \left(\frac{16K}{2K}\right)^2 = 64
\tag{3}
\end{equation}

计算和内存开销增加64倍！

#### 1.2 长序列的位置编码挑战

**问题1: 外推失败** - 训练长度 $L_{\text{train}}$，测试长度 $L_{\text{test}} > L_{\text{train}}$：

对于绝对位置编码：

\begin{equation}
\boldsymbol{x}_i' = \boldsymbol{x}_i + \boldsymbol{PE}(i)
\tag{4}
\end{equation}

当 $i > L_{\text{train}}$ 时，$\boldsymbol{PE}(i)$ 是模型从未见过的，导致性能急剧下降。

**问题2: 远程依赖衰减** - Softmax的归一化效应：

\begin{equation}
\alpha_{i,j} = \frac{\exp(s_{i,j})}{\sum_{k=1}^{L} \exp(s_{i,k})}
\tag{5}
\end{equation}

当 $L$ 很大时，每个位置分配到的注意力权重被稀释：

\begin{equation}
\mathbb{E}[\alpha_{i,j}] = \frac{1}{L}
\tag{6}
\end{equation}

**数值示例**: $L=2048$ vs $L=16384$:

\begin{equation}
\frac{\mathbb{E}[\alpha_{i,j}]_{L=16K}}{\mathbb{E}[\alpha_{i,j}]_{L=2K}} = \frac{1/16384}{1/2048} = \frac{1}{8}
\tag{7}
\end{equation}

每个token的注意力降低到原来的1/8。

### 二、预训练对长序列的数学作用

#### 2.1 Inductive Bias的数学定义

**定义**: Inductive Bias是模型假设空间的先验约束。

对于函数空间 $\mathcal{F}$，Inductive Bias $\mathcal{B}$ 限制了可学习的函数：

\begin{equation}
f \in \mathcal{F}_{\mathcal{B}} \subset \mathcal{F}
\tag{8}
\end{equation}

#### 2.2 预训练作为Inductive Bias注入

**无预训练**: 模型从随机初始化开始，$\mathcal{F}_{\text{random}}$ 很大。

**有预训练**: 模型从预训练权重开始，$\mathcal{F}_{\text{pretrain}}$ 更聚焦。

\begin{equation}
\mathcal{F}_{\text{pretrain}} \subset \mathcal{F}_{\text{random}}
\tag{9}
\end{equation}

**数学直觉**: 预训练在数据的特定流形上"雕刻"出了模型，减少了优化难度。

#### 2.3 长序列预训练的优化视角

考虑损失函数 $\mathcal{L}(\boldsymbol{\theta}; L)$，其中 $L$ 是序列长度。

**泛化界**: 根据PAC学习理论，泛化误差满足：

\begin{equation}
\mathcal{L}_{\text{test}} \leq \mathcal{L}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{d \log L}{n}}\right)
\tag{10}
\end{equation}

其中 $n$ 是训练样本数，$d$ 是模型复杂度。

**关键洞察**: 在长序列 $L$ 上预训练，可以直接减小第二项（泛化gap）！

#### 2.4 位置编码的分布适应

**原始分布**: 训练集位置编码分布 $P_{\text{train}}(i)$，$i \in [1, L_{\text{train}}]$

**目标分布**: 测试集位置编码分布 $P_{\text{test}}(i)$，$i \in [1, L_{\text{test}}]$

预训练使得：

\begin{equation}
D_{\text{KL}}(P_{\text{test}} \| P_{\text{model}}) < D_{\text{KL}}(P_{\text{test}} \| P_{\text{random}})
\tag{11}
\end{equation}

其中 $D_{\text{KL}}$ 是KL散度。

### 三、Long Range Arena (LRA) 基准的数学分析

#### 3.1 LRA任务的特征

LRA包含多个长序列任务：
- ListOps: 序列长度 ~2K
- Text: 字符级文本，长度 ~4K
- Retrieval: 文档检索，长度 ~4K
- Image: 像素级图像，长度 ~1K (32×32)
- PathFinder: 视觉路径，长度 ~1K
- Path-X: 扩展路径，长度 ~16K

**共同特点**: Token粒度细（字符级、像素级），需要捕捉长程依赖。

#### 3.2 局域性的数学刻画

**定义**: 局域性指相邻位置的相关性强于远距离位置。

\begin{equation}
\text{Corr}(\boldsymbol{x}_i, \boldsymbol{x}_j) = f(|i - j|)
\tag{12}
\end{equation}

其中 $f$ 是单调递减函数。

对于字符级文本：

\begin{equation}
f(d) \approx \exp(-\lambda d), \quad \lambda > 0
\tag{13}
\end{equation}

**数值示例**: 相邻字符相关性约0.8，间隔10个字符降到0.1。

#### 3.3 线性RNN的局域性优势

线性RNN (如SSM) 的更新公式：

\begin{equation}
\boldsymbol{h}_t = \boldsymbol{A} \boldsymbol{h}_{t-1} + \boldsymbol{B} \boldsymbol{x}_t
\tag{14}
\end{equation}

**隐式局域性**: 当 $\boldsymbol{A}$ 的谱半径 $\rho(\boldsymbol{A}) < 1$ 时：

\begin{equation}
\|\boldsymbol{h}_t\| \leq \|\boldsymbol{A}\|^t \|\boldsymbol{h}_0\| + \sum_{k=0}^{t-1} \|\boldsymbol{A}\|^k \|\boldsymbol{B}\| \|\boldsymbol{x}_{t-k}\|
\tag{15}
\end{equation}

远距离的 $\boldsymbol{x}_{t-k}$ 贡献按 $\|\boldsymbol{A}\|^k$ 指数衰减！

#### 3.4 Transformer的全局注意力困境

Transformer的注意力是全局的，没有先验的局域性偏置：

\begin{equation}
\boldsymbol{o}_i = \sum_{j=1}^{L} \alpha_{i,j} \boldsymbol{v}_j
\tag{16}
\end{equation}

$\alpha_{i,j}$ 在初始化时对所有 $j$ 几乎均匀分布。

**需要学习**: 模型需要从数据中学习局域性模式，而预训练正是提供这种学习的机会！

### 四、位置插值的深入数学推导

#### 4.1 朴素外推的失败分析

对于RoPE，位置 $n$ 的编码：

\begin{equation}
\boldsymbol{\mathcal{R}}_n = \text{diag}(\boldsymbol{R}(n\theta_0), \ldots, \boldsymbol{R}(n\theta_{d/2-1}))
\tag{17}
\end{equation}

**训练范围**: $n \in [0, L_{\text{train}}-1]$，角度范围 $\theta_i n \in [0, (L_{\text{train}}-1)\theta_i]$

**测试范围**: $n \in [0, L_{\text{test}}-1]$，角度范围扩大到 $[0, (L_{\text{test}}-1)\theta_i]$

**问题**: 对于小的 $\theta_i$ (高频)，角度超出训练范围导致外推失败。

**数值示例**: $L_{\text{train}}=2048, L_{\text{test}}=8192, \theta_0=1$:

\begin{equation}
\theta_0 \cdot 8191 = 8191 \gg \theta_0 \cdot 2047 = 2047
\tag{18}
\end{equation}

角度增加了4倍，完全超出训练范围！

#### 4.2 位置插值 (PI) 的数学原理

**核心思想**: 将测试位置缩放到训练范围内。

\begin{equation}
n_{\text{scaled}} = n \cdot \frac{L_{\text{train}}}{L_{\text{test}}}
\tag{19}
\end{equation}

**相对距离保持**: 对于位置 $m, n$:

\begin{equation}
\frac{n_{\text{scaled}} - m_{\text{scaled}}}{L_{\text{test}}} = \frac{n - m}{L_{\text{train}}}
\tag{20}
\end{equation}

相对距离的比例不变！

#### 4.3 PI的频率分析

PI等效于降低所有频率：

\begin{equation}
\theta_i \cdot n_{\text{scaled}} = \theta_i \cdot n \cdot \frac{L_{\text{train}}}{L_{\text{test}}} = \left(\theta_i \cdot \frac{L_{\text{train}}}{L_{\text{test}}}\right) \cdot n
\tag{21}
\end{equation}

等价于频率从 $\theta_i$ 变为 $\theta_i' = \theta_i \cdot s$，其中 $s = L_{\text{train}} / L_{\text{test}} < 1$。

**信息损失**: 所有频率统一降低，高频信息被压缩。

#### 4.4 PI的理论保证

**定理**: 如果训练时模型学习到了相对位置模式，PI保持这些模式。

**证明**: RoPE的内积形式：

\begin{equation}
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^\top (\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^\top \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
\tag{22}
\end{equation}

PI后：

\begin{equation}
(\boldsymbol{\mathcal{R}}_{ms} \boldsymbol{q})^\top (\boldsymbol{\mathcal{R}}_{ns} \boldsymbol{k}) = \boldsymbol{q}^\top \boldsymbol{\mathcal{R}}_{(n-m)s} \boldsymbol{k}
\tag{23}
\end{equation}

相对位置 $(n-m)$ 缩放了相同因子 $s$，模式保持！□

### 五、注意力稀疏化的数学策略

#### 5.1 滑动窗口注意力

**定义**: 每个位置只关注局部窗口内的token。

\begin{equation}
\alpha_{i,j} = \begin{cases}
\frac{\exp(s_{i,j})}{\sum_{k \in W_i} \exp(s_{i,k})}, & j \in W_i \\
0, & \text{otherwise}
\end{cases}
\tag{24}
\end{equation}

其中窗口 $W_i = \{j : |i - j| \leq w\}$，$w$ 是窗口大小。

**复杂度降低**:

\begin{equation}
\text{Time} = \mathcal{O}(Lwd) = \mathcal{O}(Ld) \quad (w \ll L)
\tag{25}
\end{equation}

从 $\mathcal{O}(L^2d)$ 降为 $\mathcal{O}(Ld)$，线性复杂度！

#### 5.2 Longformer的膨胀窗口

**组合策略**: 局部窗口 + 全局token + 膨胀注意力

\begin{equation}
\text{Attention Pattern} = W_{\text{local}} \cup G_{\text{global}} \cup W_{\text{dilated}}
\tag{26}
\end{equation}

其中：
- $W_{\text{local}}$: 滑动窗口，$|W_{\text{local}}| = 2w$
- $G_{\text{global}}$: 全局token (如[CLS])，$|G_{\text{global}}| = g$
- $W_{\text{dilated}}$: 膨胀窗口，间隔 $r$ 采样

**有效感受野**: 经过 $L_{\text{layer}}$ 层后：

\begin{equation}
\text{Receptive Field} = w \times L_{\text{layer}} + g \times L + r \times \lfloor w/r \rfloor
\tag{27}
\end{equation}

#### 5.3 BigBird的随机注意力

**三种注意力模式**:
1. 全局注意力: $g$ 个随机选择的token对所有位置
2. 窗口注意力: 窗口大小 $w$
3. 随机注意力: 每个位置随机关注 $r$ 个token

**复杂度**:

\begin{equation}
\text{Time} = \mathcal{O}((g + w + r) \cdot L \cdot d)
\tag{28}
\end{equation}

**理论保证**: 随机图的连通性理论保证，$r = \mathcal{O}(\log L)$ 即可保证图连通，实现全局信息传播。

### 六、预训练策略的数学设计

#### 6.1 掩码语言模型 (MLM) 的长序列扩展

标准MLM：

\begin{equation}
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(\boldsymbol{x}_i | \boldsymbol{x}_{\setminus \mathcal{M}})
\tag{29}
\end{equation}

其中 $\mathcal{M}$ 是被掩码的位置集合。

**长序列挑战**: $|\boldsymbol{x}|$ 很大时，上下文过长可能导致：
1. 过拟合局部模式
2. 全局信息被稀释

**改进: 跨度掩码**:

\begin{equation}
\mathcal{M} = \bigcup_{k=1}^{K} [s_k, s_k + l_k)
\tag{30}
\end{equation}

连续掩码多个token，迫使模型学习更长程的依赖。

#### 6.2 自回归预训练 (GPT-style)

\begin{equation}
\mathcal{L}_{\text{AR}} = -\sum_{i=1}^{L} \log P(\boldsymbol{x}_i | \boldsymbol{x}_{<i})
\tag{31}
\end{equation}

**长序列优势**: 每个token都作为预测目标，充分利用长序列数据。

**困难**: 早期位置的预测依赖的上下文少，晚期位置的上下文过长。

**解决: 文档级预训练**: 将多个文档拼接，学习跨文档的长程模式。

#### 6.3 对比学习的长序列视角

**SimCLR for Sequences**:

\begin{equation}
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\boldsymbol{h}_i, \boldsymbol{h}_i^+) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\boldsymbol{h}_i, \boldsymbol{h}_j) / \tau)}
\tag{32}
\end{equation}

其中 $\boldsymbol{h}_i^+$ 是正样本（同一序列的不同视角），$\boldsymbol{h}_j$ 是负样本。

**长序列的增强策略**:
1. 随机裁剪不同长度片段
2. 时间扭曲 (temporal warping)
3. 位置打乱

### 七、长序列的计算优化

#### 7.1 FlashAttention的数学原理

**核心思想**: 减少HBM (高带宽内存) 访问，利用SRAM (片上内存)。

**标准Attention的内存访问**:
1. 从HBM读取Q, K, V: $\mathcal{O}(Ld)$
2. 计算 $\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{K}^\top$，写入HBM: $\mathcal{O}(L^2)$
3. 计算 $\boldsymbol{P} = \text{softmax}(\boldsymbol{S})$，写入HBM: $\mathcal{O}(L^2)$
4. 计算 $\boldsymbol{O} = \boldsymbol{P}\boldsymbol{V}$: $\mathcal{O}(L^2d)$

总HBM访问: $\mathcal{O}(L^2 + Ld)$

**FlashAttention的分块策略**:

将Q, K, V分成大小为 $B$ 的块，在SRAM中完成计算：

\begin{equation}
\boldsymbol{O}_{[i]} = \sum_{j} \text{softmax}\left(\frac{\boldsymbol{Q}_{[i]} \boldsymbol{K}_{[j]}^\top}{\sqrt{d}}\right) \boldsymbol{V}_{[j]}
\tag{33}
\end{equation}

**HBM访问降低**: $\mathcal{O}(L^2 / B + Ld)$，当 $B \sim \sqrt{L}$ 时接近线性！

#### 7.2 Flash-Decoupled Attention

**分离QK和V的计算**:

\begin{equation}
\boldsymbol{A} = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}}\right)
\tag{34}
\end{equation}

\begin{equation}
\boldsymbol{O} = \boldsymbol{A} \boldsymbol{V}
\tag{35}
\end{equation}

**优化**: 可以先计算稀疏的 $\boldsymbol{A}$ (只保留top-k)，再乘 $\boldsymbol{V}$。

\begin{equation}
\text{Sparse}(\boldsymbol{A})_{i,j} = \begin{cases}
A_{i,j}, & j \in \text{top-k}(A_{i,:}) \\
0, & \text{otherwise}
\end{cases}
\tag{36}
\end{equation}

**复杂度**: $\mathcal{O}(Lkd)$ vs $\mathcal{O}(L^2d)$，当 $k \ll L$ 时显著降低。

#### 7.3 梯度检查点 (Gradient Checkpointing)

**标准反向传播**: 需要存储所有中间激活，内存 $\mathcal{O}(N \times L \times d)$

**梯度检查点**: 只存储部分激活，需要时重新计算。

分 $C$ 个检查点：

\begin{equation}
\text{Memory} = \mathcal{O}(C \times L \times d)
\tag{37}
\end{equation}

\begin{equation}
\text{Computation} = \mathcal{O}(N \times L \times d) \times \left(1 + \frac{N}{C}\right)
\tag{38}
\end{equation}

**最优策略**: $C = \sqrt{N}$，内存降低 $\sqrt{N}$ 倍，计算增加常数倍。

### 八、实验结果的数学解释

#### 8.1 LRA性能提升的分解

预训练后的性能提升可以分解为：

\begin{equation}
\Delta_{\text{Acc}} = \Delta_{\text{pos}} + \Delta_{\text{local}} + \Delta_{\text{global}}
\tag{39}
\end{equation}

其中：
- $\Delta_{\text{pos}}$: 位置编码适应
- $\Delta_{\text{local}}$: 局部模式学习
- $\Delta_{\text{global}}$: 全局依赖建模

**Transformer的优势**: $\Delta_{\text{global}}$ 更大，因为全局注意力机制。

**数值示例**: 在ListOps任务上：
- 无预训练: 36.4%
- 有预训练: 38.6%
- $\Delta_{\text{pos}} \approx 1.0\%$
- $\Delta_{\text{local}} \approx 0.5\%$
- $\Delta_{\text{global}} \approx 0.7\%$

#### 8.2 Transformer vs SSM的对比

**SSM的优势**: 天然的局域性偏置

\begin{equation}
\text{Score}_{\text{SSM}} = \alpha \cdot \text{Local} + \beta \cdot \text{Global}
\tag{40}
\end{equation}

其中 $\alpha > \beta$（局部权重更大）。

**Transformer + 预训练**: 学会了局域性，同时保留全局能力

\begin{equation}
\text{Score}_{\text{TF+PT}} = \alpha' \cdot \text{Local} + \beta' \cdot \text{Global}
\tag{41}
\end{equation}

其中 $\alpha' \approx \alpha, \beta' > \beta$。

**关键**: 预训练让Transformer学会了SSM的优势，同时保持了自己的优势！

### 九、理论分析与实践建议

#### 9.1 预训练长度的选择

**目标长度** $L_{\text{target}}$，**预训练长度** $L_{\text{pretrain}}$ 的选择：

**经验公式**:

\begin{equation}
L_{\text{pretrain}} = \min(L_{\text{target}}, \alpha \cdot L_{\text{train}})
\tag{42}
\end{equation}

其中 $L_{\text{train}}$ 是原训练长度，$\alpha \in [2, 4]$。

**数值示例**: 原训练2K，目标16K:
- 直接微调: 失败
- 预训练4K → 微调16K: 成功
- 预训练8K → 微调16K: 更好

#### 9.2 预训练数据量的估计

**PAC界**: 需要的样本数与序列长度相关：

\begin{equation}
n \geq \frac{c}{\epsilon^2} (d \log L + \log(1/\delta))
\tag{43}
\end{equation}

其中 $\epsilon$ 是目标误差，$\delta$ 是失败概率。

**关键**: 数据需求随 $\log L$ 增长，而非线性增长！

**数值示例**: $L$ 从2K增加到16K ($\times 8$)，数据需求增加 $\log_2 8 = 3$ 倍。

#### 9.3 位置编码方案的选择

| 序列长度 | 推荐方案 | 理由 |
|----------|----------|------|
| ≤ 2K | 标准RoPE | 无需特殊处理 |
| 2K-8K | PI | 简单有效 |
| 8K-32K | NTK-Aware PI | 保留高频信息 |
| > 32K | ALiBi或预训练 | 外推性更好 |

#### 9.4 注意力模式的配置

**推荐配置**: Longformer-style

```
窗口大小 w = min(512, L/4)
全局token数 g = ceil(log_2(L))
膨胀率 r = 2 (偶数层) / 4 (奇数层)
```

**复杂度**:

\begin{equation}
\text{Total} = \mathcal{O}((w + g + w/r) \times L \times d) \approx \mathcal{O}(wLd)
\tag{44}
\end{equation}

### 十、总结

本节详细推导了长序列Transformer的数学原理：

**核心发现**:
1. **预训练的作用**: 注入Inductive Bias，适应位置分布
2. **位置插值**: 保持相对位置模式，实现长度外推
3. **稀疏注意力**: 降低复杂度，保留关键依赖

**关键公式**:

**位置插值**:
\begin{equation}
n_{\text{scaled}} = n \cdot \frac{L_{\text{train}}}{L_{\text{test}}}
\tag{45}
\end{equation}

**稀疏注意力复杂度**:
\begin{equation}
\mathcal{O}(L^2d) \to \mathcal{O}((w + g + r) Ld)
\tag{46}
\end{equation}

**预训练样本需求**:
\begin{equation}
n \geq \mathcal{O}(d \log L)
\tag{47}
\end{equation}

**实践启示**:
1. 长序列任务必须预训练（尤其是细粒度token）
2. 位置编码选择需要考虑外推性
3. 稀疏注意力可以有效降低复杂度
4. Transformer的全局能力在预训练后得以充分发挥

通过预训练，Transformer在长序列建模上不再逊色于线性RNN，甚至在某些任务上表现更优，这为长序列处理提供了新的范式。

