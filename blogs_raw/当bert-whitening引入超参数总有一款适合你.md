---
title: 当BERT-whitening引入超参数：总有一款适合你
slug: 当bert-whitening引入超参数总有一款适合你
date: 2022-05-18
tags: 语言模型, 语义, 语义相似度, 生成模型, attention
status: completed
---

# 当BERT-whitening引入超参数：总有一款适合你

**原文链接**: [https://spaces.ac.cn/archives/9079](https://spaces.ac.cn/archives/9079)

**发布日期**: 

---

在[《你可能不需要BERT-flow：一个线性变换媲美BERT-flow》](/archives/8069)中，笔者提出了BERT-whitening，验证了一个线性变换就能媲美当时的SOTA方法BERT-flow。此外，BERT-whitening还可以对句向量进行降维，带来更低的内存占用和更快的检索速度。然而，在[《无监督语义相似度哪家强？我们做了个比较全面的评测》](/archives/8321)中我们也发现，whitening操作并非总能带来提升，有些模型本身就很贴合任务（如经过有监督训练的SimBERT），那么额外的whitening操作往往会降低效果。

为了弥补这个不足，本文提出往BERT-whitening中引入了两个超参数，通过调节这两个超参数，我们几乎可以总是获得“降维不掉点”的结果。换句话说，即便是原来加上whitening后效果会下降的任务，如今也有机会在降维的同时获得相近甚至更好的效果了。

## 方法概要 #

目前BERT-whitening的流程是：  
\begin{equation}\begin{aligned}  
\tilde{\boldsymbol{x}}_i =&\, (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-1/2} \\\  
\boldsymbol{\mu} =&\, \frac{1}{N}\sum\limits_{i=1}^N \boldsymbol{x}_i \\\  
\boldsymbol{\Sigma} =&\, \frac{1}{N}\sum\limits_{i=1}^N (\boldsymbol{x}_i - \boldsymbol{\mu})^{\top}(\boldsymbol{x}_i - \boldsymbol{\mu}) = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top} \,\,(\text{SVD分解})  
\end{aligned}\end{equation}  
其中$\boldsymbol{x}_i$是给定的句向量（如无说明，向量默认为行向量），$\tilde{\boldsymbol{x}}_i$是变换后的向量，SVD分解的结果中，$\boldsymbol{U}$是正交矩阵，$\boldsymbol{\Lambda}$是对角矩阵，并且对角线的元素非负且从大到小排列。可以看到，目前的流程是完全固定的，即没有任何可调的超参数。

为了增加一定的调节空间，我们可以往里边引入两个超参数$\beta,\gamma$（标量），使其变为  
\begin{equation}\begin{aligned}  
\tilde{\boldsymbol{x}}_i =&\, (\boldsymbol{x}_i - {\color{red}\beta}\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-{\color{red}\gamma}/2} \\\  
\boldsymbol{\mu} =&\, \frac{1}{N}\sum\limits_{i=1}^N \boldsymbol{x}_i \\\  
\boldsymbol{\Sigma} =&\, \frac{1}{N}\sum\limits_{i=1}^N (\boldsymbol{x}_i - {\color{red}\beta}\boldsymbol{\mu})^{\top}(\boldsymbol{x}_i - {\color{red}\beta}\boldsymbol{\mu}) = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top} \,\,(\text{SVD分解})  
\end{aligned}\end{equation}

## 思路分析 #

可以看到，当$\beta=\gamma=1$时，就是原来的BERT-whitening；而当$\beta=\gamma=0$时，净变换就是  
\begin{equation}\tilde{\boldsymbol{x}}_i =\boldsymbol{x}_i \boldsymbol{U}\end{equation}  
由于$\boldsymbol{U}$是正交矩阵，所以不改变内积结果，即$\tilde{\boldsymbol{x}}_i\tilde{\boldsymbol{x}}_i^{\top} = \boldsymbol{x}_i \boldsymbol{U} (\boldsymbol{x}_i \boldsymbol{U})^{\top} = \boldsymbol{x}_i\boldsymbol{x}_i^{\top}$，所以当我们用余弦相似度作为相似度量时，它不会改变原有结果。换句话说，引入这组超参数后，它提供了“不逊色于变换前的效果”的可能性，那么当我们精调这组参数时，就有可能取得比变换前更好的效果。这也是这两个超参数的设计思路。

此外，在这样的改动之下，原来的降维能力还是得以保留的。我们可以将变换拆开为两部分看：  
\begin{equation}\tilde{\boldsymbol{x}}_i = \color{red}{\underbrace{(\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}}_{\text{part 1}}}\color{skyblue}{\underbrace{\boldsymbol{\Lambda}^{-\gamma/2}}_{\text{part 2}}}\end{equation}  
第一部分主要是正交变换$\boldsymbol{U}$，$\boldsymbol{U}$是$\boldsymbol{\Sigma}$矩阵SVD分解之后的结果，它能将向量$\boldsymbol{x}_i - \beta\boldsymbol{\mu}$变换成每个分量尽量独立的新向量，并且新向量的每个分量与0的平均波动正好是由$\boldsymbol{\Lambda}^{1/2}$的对角线元素来衡量，如果对应的波动很接近于0，那么我们就可以认为它实际就是0，舍去这个分量也不会影响余弦值的计算结果，这就是降维的原理。而由于SVD分解的结果已经提前将$\boldsymbol{\Lambda}$从大到小排好了顺序，因此我们可以直接通过保留前$k$维的操作$\tilde{\boldsymbol{x}}_i\text{[:}k\text{]}$就可以实现降到$k$维了。

至于第二部分$\boldsymbol{\Lambda}^{-\gamma/2}$，我们可以理解为当前任务对各向同性的依赖程度，如果$\gamma=1$，那么相当于每个分量都是各平权的，这可以作为一个无监督的先验结果，但未必对所有任务都是最优的，所以我们可以通过调节$\gamma$来更好地适应当前任务。

## 实验结果 #

文章[《无监督语义相似度哪家强？我们做了个比较全面的评测》](/archives/8321)已经显示，在ATEC、BQ、LCQMC三个任务上，SimBERT加上默认的whitening操作（即$\beta=\gamma=1$）都会导致效果下降，而如果我们取$\beta=\gamma=0$，那么结果就不一样了（随便演示了两个组合，其他组合结果相似）：  
$$\small{\begin{array}{c}  
\text{BERT-P4效果表} \\\  
{\begin{array}{l|ccccc}  
\hline  
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{STS-B} \\\  
\hline  
\beta=\gamma=1 & 24.51 / \color{green}{27.00} / \color{green}{27.91} & 38.81 / \color{red}{32.29} / \color{red}{37.67} & 64.75 / \color{green}{64.75} / \color{green}{65.65} & 15.12 / \color{green}{17.80} / \color{green}{15.34} & 61.66 / \color{green}{69.45} / \color{green}{69.37}  
\\\  
\beta=\gamma=0 & 24.51 / 24.51 / \color{green}{24.59} & 38.81 / 38.81 / \color{green}{38.99} & 64.75 / 64.75 / \color{red}{63.45} & 15.12 / 15.12 / \color{red}{14.59} & 61.66 / 61.66 / \color{green}{62.30} \\\  
\hline  
\end{array}} \\\  
\\\  
\text{SimBERT-P1效果表} \\\  
{\begin{array}{l|ccccc}  
\hline  
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{STS-B} \\\  
\hline  
\beta=\gamma=1 & 38.50 / \color{red}{23.64} / \color{red}{30.79} & 48.54 / \color{red}{31.78} / \color{red}{40.01} & 76.23 / \color{red}{75.05} / \color{red}{74.50} & 15.10 / \color{green}{18.49} / \color{green}{15.64} & 74.14 / \color{red}{73.37} / \color{green}{75.29} \\\  
\beta=\gamma=0 & 38.50 / 38.50 / \color{green}{38.81} & 48.54 / 48.54 / \color{green}{48.66} & 76.23 / 76.23 / \color{red}{76.22} & 15.10 / 15.10 / \color{red}{14.88} & 74.14 / 74.14 / \color{green}{74.46} \\\  
\hline  
\end{array}}  
\end{array}}$$

跟之前的文章一样，表格中的每个元素是$a / b / c$的形式，代表该任务在该模型下“不加whitening”的得分为$a$、“加whitening”的得分为$b$、“加whitening并降到256维”的得分为$c$；如果$b > a$，那么$b$显示为绿色，小于则显示为红色；如果$c > a$，那么$c$显示为绿色，小于则显示为红色。前面说了，如果不降维的话，$\beta=\gamma=0$的净变换就是$\boldsymbol{U}$，不改变余弦值结果，因此$\beta=\gamma=0$时的$a,b$都是相等的。

在这个表格中，我们主要看$a/b/c$中的第三个结果$c$，它是将向量从768维降低到256维的结果，可以看到当$\beta=\gamma=0$时，不管是无监督的BERT还是有监督的SimBERT，该结果基本都很接近原始向量的结果（即$a$），部分结果甚至还有提升。这就意味着，$\beta=\gamma=0,k=256$这个组合几乎可以算是“免费的午餐”，几乎无损效果，并且实现了降维。

笔者也试过精调$\beta,\gamma$，在一些任务上确实能取得比上述两个组合更好的效果，但精调需要标签数据，争议性可能会比较大，这里就不演示了。如果原来的句向量模型本就是有监督训练得到的，用BERT-whitening仅仅是奔着降维去的，那么就可以用验证集来精调一下$\beta,\gamma$和$k$了，这种场景下就是无争议的了。

## 文章小结 #

本文通过引入两个超参数的方式来赋予BERT-whitening一定的调参空间，使其具备“不逊色于变换前的效果”的可能性，并且保留了降维的能力。换言之，即便是之前已经训练好的句向量模型，我们也可以用新的BERT-whitening将它降维，并且保持效果基本不变，有时候甚至还更优～

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9079>_

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

苏剑林. (May. 18, 2022). 《当BERT-whitening引入超参数：总有一款适合你 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9079>

@online{kexuefm-9079,  
title={当BERT-whitening引入超参数：总有一款适合你},  
author={苏剑林},  
year={2022},  
month={May},  
url={\url{https://spaces.ac.cn/archives/9079}},  
} 


---

## 公式推导与注释

### 1. PCA与协方差矩阵的数学基础

#### 1.1 协方差矩阵的定义与性质

对于句向量集合 $\{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_N\}$，其中每个 $\boldsymbol{x}_i \in \mathbb{R}^d$，我们首先计算均值向量：

\begin{equation}
\boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \boldsymbol{x}_i
\tag{1}
\end{equation}

**数学直觉**：均值向量代表所有句向量的"中心位置"，去中心化操作将数据平移到原点附近。

中心化后的向量为：

\begin{equation}
\tilde{\boldsymbol{x}}_i = \boldsymbol{x}_i - \boldsymbol{\mu}
\tag{2}
\end{equation}

协方差矩阵定义为：

\begin{equation}
\boldsymbol{\Sigma} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{x}_i - \boldsymbol{\mu})(\boldsymbol{x}_i - \boldsymbol{\mu})^{\top} = \frac{1}{N}\sum_{i=1}^N \tilde{\boldsymbol{x}}_i \tilde{\boldsymbol{x}}_i^{\top}
\tag{3}
\end{equation}

**协方差矩阵的性质**：

\begin{equation}
\boldsymbol{\Sigma}_{jk} = \frac{1}{N}\sum_{i=1}^N (x_{ij} - \mu_j)(x_{ik} - \mu_k)
\tag{4}
\end{equation}

其中 $\boldsymbol{\Sigma}_{jk}$ 表示第 $j$ 维和第 $k$ 维特征的协方差。

**重要性质**：
1. 对称性：$\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^{\top}$
2. 半正定性：$\boldsymbol{v}^{\top}\boldsymbol{\Sigma}\boldsymbol{v} \geq 0$ 对所有 $\boldsymbol{v} \in \mathbb{R}^d$
3. 实对称矩阵可对角化

#### 1.2 特征值分解（SVD分解）

由于协方差矩阵是实对称矩阵，可以进行特征值分解（SVD）：

\begin{equation}
\boldsymbol{\Sigma} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}
\tag{5}
\end{equation}

其中：
- $\boldsymbol{U} = [\boldsymbol{u}_1, \boldsymbol{u}_2, \ldots, \boldsymbol{u}_d] \in \mathbb{R}^{d \times d}$ 是正交矩阵（特征向量）
- $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$ 是对角矩阵（特征值）
- $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$（按降序排列）

**正交性**：$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{U}\boldsymbol{U}^{\top} = \boldsymbol{I}$

**特征方程**：

\begin{equation}
\boldsymbol{\Sigma}\boldsymbol{u}_i = \lambda_i \boldsymbol{u}_i, \quad i = 1, 2, \ldots, d
\tag{6}
\end{equation}

**数学直觉**：第 $i$ 个特征向量 $\boldsymbol{u}_i$ 指向数据方差最大的第 $i$ 个方向，对应的特征值 $\lambda_i$ 表示该方向上的方差大小。

#### 1.3 主成分分析（PCA）的几何意义

将中心化向量投影到特征向量构成的新坐标系：

\begin{equation}
\boldsymbol{y}_i = \boldsymbol{U}^{\top}(\boldsymbol{x}_i - \boldsymbol{\mu})
\tag{7}
\end{equation}

投影后的协方差矩阵：

\begin{equation}
\begin{aligned}
\text{Cov}(\boldsymbol{y}) &= \frac{1}{N}\sum_{i=1}^N \boldsymbol{y}_i\boldsymbol{y}_i^{\top} \\
&= \frac{1}{N}\sum_{i=1}^N \boldsymbol{U}^{\top}(\boldsymbol{x}_i - \boldsymbol{\mu})(\boldsymbol{x}_i - \boldsymbol{\mu})^{\top}\boldsymbol{U} \\
&= \boldsymbol{U}^{\top}\boldsymbol{\Sigma}\boldsymbol{U} \\
&= \boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}\boldsymbol{U} \\
&= \boldsymbol{\Lambda}
\end{aligned}
\tag{8}
\end{equation}

**关键结论**：PCA变换将数据转换到一个新的坐标系，使得各维度之间不相关（协方差为0），且按方差大小排序。

### 2. Whitening（白化）变换的完整推导

#### 2.1 白化的目标

白化变换的目标是找到一个线性变换 $\boldsymbol{W}$，使得变换后的数据满足：

\begin{equation}
\mathbb{E}[\tilde{\boldsymbol{x}}] = \boldsymbol{0}, \quad \text{Cov}(\tilde{\boldsymbol{x}}) = \boldsymbol{I}
\tag{9}
\end{equation}

即：均值为零，协方差矩阵为单位矩阵（各维度独立且方差为1）。

#### 2.2 ZCA白化（Zero-phase Component Analysis）

ZCA白化的变换形式为：

\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{\Sigma}^{-1/2}
\tag{10}
\end{equation}

其中 $\boldsymbol{\Sigma}^{-1/2}$ 定义为：

\begin{equation}
\boldsymbol{\Sigma}^{-1/2} = \boldsymbol{U}\boldsymbol{\Lambda}^{-1/2}\boldsymbol{U}^{\top}
\tag{11}
\end{equation}

**验证白化性质**：

\begin{equation}
\begin{aligned}
\text{Cov}(\tilde{\boldsymbol{x}}) &= \frac{1}{N}\sum_{i=1}^N \tilde{\boldsymbol{x}}_i\tilde{\boldsymbol{x}}_i^{\top} \\
&= \frac{1}{N}\sum_{i=1}^N (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{\Sigma}^{-1/2}(\boldsymbol{\Sigma}^{-1/2})^{\top}(\boldsymbol{x}_i - \boldsymbol{\mu})^{\top} \\
&= \boldsymbol{\Sigma}^{-1/2}\boldsymbol{\Sigma}(\boldsymbol{\Sigma}^{-1/2})^{\top} \\
&= \boldsymbol{\Sigma}^{-1/2}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1/2} \\
&= \boldsymbol{I}
\end{aligned}
\tag{12}
\end{equation}

#### 2.3 PCA白化（降维白化）

PCA白化不经过 $\boldsymbol{U}^{\top}$ 再回转，而是直接在主成分空间进行白化：

\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-1/2}
\tag{13}
\end{equation}

**分解理解**：

\begin{equation}
\tilde{\boldsymbol{x}}_i = \underbrace{(\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{U}}_{\text{旋转到主成分空间}} \cdot \underbrace{\boldsymbol{\Lambda}^{-1/2}}_{\text{方差归一化}}
\tag{14}
\end{equation}

第一步：旋转变换
\begin{equation}
\boldsymbol{z}_i = (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{U}
\tag{15}
\end{equation}

此时 $\text{Cov}(\boldsymbol{z}) = \boldsymbol{\Lambda}$（对角矩阵，各维度独立但方差不等）

第二步：方差归一化
\begin{equation}
\tilde{\boldsymbol{x}}_i = \boldsymbol{z}_i\boldsymbol{\Lambda}^{-1/2}
\tag{16}
\end{equation}

此时 $\text{Cov}(\tilde{\boldsymbol{x}}) = \boldsymbol{I}$（单位矩阵，各维度独立且方差为1）

**数学直觉**：PCA白化首先通过旋转消除相关性，然后通过缩放统一方差。

### 3. BERT-Whitening中的超参数设计

#### 3.1 参数化白化变换

引入超参数 $\beta$ 和 $\gamma$ 后的白化变换：

\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-\gamma/2}
\tag{17}
\end{equation}

相应的协方差矩阵计算也需要调整：

\begin{equation}
\boldsymbol{\Sigma}_{\beta} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{x}_i - \beta\boldsymbol{\mu})(\boldsymbol{x}_i - \beta\boldsymbol{\mu})^{\top} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}
\tag{18}
\end{equation}

#### 3.2 参数β的数学意义（中心化程度）

**情形1**：$\beta = 1$（完全中心化）

\begin{equation}
\boldsymbol{x}_i - \boldsymbol{\mu} = \boldsymbol{x}_i - \frac{1}{N}\sum_{j=1}^N \boldsymbol{x}_j
\tag{19}
\end{equation}

此时数据严格中心化，均值为零。

**情形2**：$\beta = 0$（不中心化）

\begin{equation}
\boldsymbol{x}_i - 0 \cdot \boldsymbol{\mu} = \boldsymbol{x}_i
\tag{20}
\end{equation}

保持原始数据分布。

**中间值**：$0 < \beta < 1$

\begin{equation}
\boldsymbol{x}_i - \beta\boldsymbol{\mu} = (1-\beta)\boldsymbol{x}_i + \beta(\boldsymbol{x}_i - \boldsymbol{\mu})
\tag{21}
\end{equation}

这是原始向量和中心化向量的凸组合。

**协方差矩阵的变化**：

\begin{equation}
\begin{aligned}
\boldsymbol{\Sigma}_{\beta} &= \frac{1}{N}\sum_{i=1}^N (\boldsymbol{x}_i - \beta\boldsymbol{\mu})(\boldsymbol{x}_i - \beta\boldsymbol{\mu})^{\top} \\
&= \frac{1}{N}\sum_{i=1}^N [(\boldsymbol{x}_i - \boldsymbol{\mu}) + (1-\beta)\boldsymbol{\mu}][(\boldsymbol{x}_i - \boldsymbol{\mu}) + (1-\beta)\boldsymbol{\mu}]^{\top} \\
&= \boldsymbol{\Sigma}_1 + (1-\beta)^2\boldsymbol{\mu}\boldsymbol{\mu}^{\top}
\end{aligned}
\tag{22}
\end{equation}

其中 $\boldsymbol{\Sigma}_1$ 是完全中心化时的协方差矩阵。

#### 3.3 参数γ的数学意义（各向同性程度）

**情形1**：$\gamma = 1$（完全白化）

\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-1/2}
\tag{23}
\end{equation}

此时各维度方差完全相等：

\begin{equation}
\text{Var}(\tilde{x}_{ij}) = 1, \quad \forall j
\tag{24}
\end{equation}

**情形2**：$\gamma = 0$（不归一化）

\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^0 = (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}
\tag{25}
\end{equation}

由于 $\boldsymbol{U}$ 是正交矩阵，内积保持不变：

\begin{equation}
\begin{aligned}
\tilde{\boldsymbol{x}}_i \cdot \tilde{\boldsymbol{x}}_j &= [(\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}] \cdot [(\boldsymbol{x}_j - \beta\boldsymbol{\mu})\boldsymbol{U}] \\
&= (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{U}^{\top}(\boldsymbol{x}_j - \beta\boldsymbol{\mu})^{\top} \\
&= (\boldsymbol{x}_i - \beta\boldsymbol{\mu})(\boldsymbol{x}_j - \beta\boldsymbol{\mu})^{\top}
\end{aligned}
\tag{26}
\end{equation}

特别地，当 $\beta = 0$ 时：

\begin{equation}
\tilde{\boldsymbol{x}}_i \cdot \tilde{\boldsymbol{x}}_j = \boldsymbol{x}_i \cdot \boldsymbol{x}_j
\tag{27}
\end{equation}

余弦相似度完全保持：

\begin{equation}
\cos(\tilde{\boldsymbol{x}}_i, \tilde{\boldsymbol{x}}_j) = \frac{\tilde{\boldsymbol{x}}_i \cdot \tilde{\boldsymbol{x}}_j}{\|\tilde{\boldsymbol{x}}_i\|\|\tilde{\boldsymbol{x}}_j\|} = \frac{\boldsymbol{x}_i \cdot \boldsymbol{x}_j}{\|\boldsymbol{x}_i\|\|\boldsymbol{x}_j\|} = \cos(\boldsymbol{x}_i, \boldsymbol{x}_j)
\tag{28}
\end{equation}

**中间值**：$0 < \gamma < 1$

\begin{equation}
\boldsymbol{\Lambda}^{-\gamma/2} = \text{diag}(\lambda_1^{-\gamma/2}, \lambda_2^{-\gamma/2}, \ldots, \lambda_d^{-\gamma/2})
\tag{29}
\end{equation}

变换后的方差：

\begin{equation}
\text{Var}(\tilde{x}_{ij}) = \lambda_j^{1-\gamma}
\tag{30}
\end{equation}

当 $\gamma < 1$ 时，保留部分原始方差信息：
- 大特征值对应的维度：$\lambda_j$ 大，$\lambda_j^{1-\gamma}$ 也大
- 小特征值对应的维度：$\lambda_j$ 小，$\lambda_j^{1-\gamma}$ 也小

### 4. 降维原理的数学分析

#### 4.1 主成分选择

保留前 $k$ 个主成分（$k < d$）：

\begin{equation}
\tilde{\boldsymbol{x}}_i^{(k)} = [(\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-\gamma/2}]_{:k}
\tag{31}
\end{equation}

其中 $[\cdot]_{:k}$ 表示取前 $k$ 维。

等价于：

\begin{equation}
\tilde{\boldsymbol{x}}_i^{(k)} = (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}_{:k}\boldsymbol{\Lambda}_{k}^{-\gamma/2}
\tag{32}
\end{equation}

其中 $\boldsymbol{U}_{:k} \in \mathbb{R}^{d \times k}$ 是前 $k$ 个特征向量，$\boldsymbol{\Lambda}_k = \text{diag}(\lambda_1, \ldots, \lambda_k)$。

#### 4.2 信息保留率

定义累积方差贡献率：

\begin{equation}
\eta_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}
\tag{33}
\end{equation}

**重构误差**（当 $\gamma = 1$，$\beta = 1$ 时）：

\begin{equation}
\begin{aligned}
E_k &= \frac{1}{N}\sum_{i=1}^N \|\boldsymbol{x}_i - \hat{\boldsymbol{x}}_i^{(k)}\|^2 \\
&= \sum_{j=k+1}^d \lambda_j
\end{aligned}
\tag{34}
\end{equation}

其中 $\hat{\boldsymbol{x}}_i^{(k)}$ 是从低维重构回的向量。

**相对误差**：

\begin{equation}
\frac{E_k}{\sum_{i=1}^d \lambda_i} = 1 - \eta_k
\tag{35}
\end{equation}

#### 4.3 余弦距离的保持性

对于降维后的向量，余弦相似度的变化：

\begin{equation}
\begin{aligned}
\cos(\tilde{\boldsymbol{x}}_i^{(k)}, \tilde{\boldsymbol{x}}_j^{(k)}) &= \frac{(\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}_{:k}\boldsymbol{\Lambda}_k^{-\gamma}(\boldsymbol{x}_j - \beta\boldsymbol{\mu})^{\top}}{\sqrt{[(\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}_{:k}\boldsymbol{\Lambda}_k^{-\gamma}\boldsymbol{U}_{:k}^{\top}(\boldsymbol{x}_i - \beta\boldsymbol{\mu})^{\top}]} \cdot \sqrt{\cdots}}
\end{aligned}
\tag{36}
\end{equation}

当 $\gamma = 1$（完全白化）时，在高维空间中所有维度平权，降维后的余弦相似度近似为：

\begin{equation}
\cos(\tilde{\boldsymbol{x}}_i^{(k)}, \tilde{\boldsymbol{x}}_j^{(k)}) \approx \frac{k}{d} \cos(\tilde{\boldsymbol{x}}_i, \tilde{\boldsymbol{x}}_j)
\tag{37}
\end{equation}

当特征值快速衰减（前 $k$ 个特征值占主导）时，余弦相似度能较好保持。

### 5. 各向异性问题的数学分析

#### 5.1 各向异性的定义

BERT等预训练模型的句向量常存在各向异性问题，表现为：

\begin{equation}
\lambda_1 \gg \lambda_2 \gg \cdots \gg \lambda_d
\tag{38}
\end{equation}

即少数几个主成分占据了绝大部分方差。

**各向异性度量**（有效维度数）：

\begin{equation}
D_{\text{eff}} = \exp\left(-\sum_{i=1}^d p_i \log p_i\right), \quad p_i = \frac{\lambda_i}{\sum_j \lambda_j}
\tag{39}
\end{equation}

当 $D_{\text{eff}} \ll d$ 时，表示各向异性严重。

#### 5.2 各向异性的影响

假设 $\lambda_1 = \alpha \sum_{i=2}^d \lambda_i$，其中 $\alpha \gg 1$。

对于两个向量的余弦相似度：

\begin{equation}
\cos(\boldsymbol{x}_i, \boldsymbol{x}_j) = \frac{\boldsymbol{x}_i \cdot \boldsymbol{x}_j}{\|\boldsymbol{x}_i\|\|\boldsymbol{x}_j\|}
\tag{40}
\end{equation}

如果两个向量在第一主成分方向上的投影都很大，即使它们语义不相关，余弦相似度也会很高：

\begin{equation}
\boldsymbol{x}_i \cdot \boldsymbol{x}_j \approx (\boldsymbol{x}_i \cdot \boldsymbol{u}_1)(\boldsymbol{x}_j \cdot \boldsymbol{u}_1) + \text{小量}
\tag{41}
\end{equation}

#### 5.3 白化如何缓解各向异性

白化通过 $\boldsymbol{\Lambda}^{-1/2}$ 将各方向的方差归一化：

\begin{equation}
\text{Var}(\tilde{x}_{ij}) = \lambda_j \cdot \lambda_j^{-1} = 1, \quad \forall j
\tag{42}
\end{equation}

变换后的协方差矩阵：

\begin{equation}
\text{Cov}(\tilde{\boldsymbol{x}}) = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top} \cdot \boldsymbol{U}\boldsymbol{\Lambda}^{-1}\boldsymbol{U}^{\top} = \boldsymbol{I}
\tag{43}
\end{equation}

此时数据在所有方向上的方差相等，各向同性。

### 6. 数值稳定性分析

#### 6.1 小特征值问题

当 $\lambda_i$ 很小时，$\lambda_i^{-1/2}$ 会非常大，导致数值不稳定。

**正则化方案**：

\begin{equation}
\boldsymbol{\Lambda}_{\epsilon}^{-1/2} = \text{diag}\left(\frac{1}{\sqrt{\lambda_1 + \epsilon}}, \ldots, \frac{1}{\sqrt{\lambda_d + \epsilon}}\right)
\tag{44}
\end{equation}

其中 $\epsilon > 0$ 是小的正则化常数（如 $10^{-5}$）。

#### 6.2 矩阵求逆的数值稳定性

协方差矩阵的条件数：

\begin{equation}
\kappa(\boldsymbol{\Sigma}) = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{\lambda_1}{\lambda_d}
\tag{45}
\end{equation}

当 $\kappa(\boldsymbol{\Sigma}) \gg 1$ 时，矩阵接近奇异，求逆不稳定。

**使用SVD的优势**：SVD方法数值稳定性好，即使矩阵接近奇异也能得到可靠结果。

### 7. SimBERT实验的数学解释

#### 7.1 有监督模型的特点

SimBERT是有监督训练的句向量模型，其目标函数通常包含对比学习损失：

\begin{equation}
\mathcal{L} = -\log \frac{\exp(\text{sim}(\boldsymbol{x}_i, \boldsymbol{x}_i^+)/\tau)}{\sum_j \exp(\text{sim}(\boldsymbol{x}_i, \boldsymbol{x}_j)/\tau)}
\tag{46}
\end{equation}

其中 $\boldsymbol{x}_i^+$ 是正样本，$\tau$ 是温度参数。

有监督训练倾向于学习更加各向同性的表示：

\begin{equation}
\lambda_1 / \lambda_d \approx \mathcal{O}(1)
\tag{47}
\end{equation>

#### 7.2 为什么有监督模型不适合β=γ=1

当模型本身已经各向同性时，强制白化（$\gamma = 1$）可能破坏有用的方差信息：

\begin{equation}
\boldsymbol{\Lambda}^{-1/2} \approx \boldsymbol{I} \cdot c
\tag{48}
\end{equation}

此时白化变换近似于均匀缩放，但中心化（$\beta = 1$）可能移除有用的位置信息。

#### 7.3 β=γ=0的优势

对于有监督模型，设置 $\beta = \gamma = 0$：

\begin{equation}
\tilde{\boldsymbol{x}}_i = \boldsymbol{x}_i\boldsymbol{U}
\tag{49}
\end{equation}

这仅是一个正交旋转，保持所有几何性质：

\begin{equation}
\|\tilde{\boldsymbol{x}}_i\| = \|\boldsymbol{x}_i\|, \quad \cos(\tilde{\boldsymbol{x}}_i, \tilde{\boldsymbol{x}}_j) = \cos(\boldsymbol{x}_i, \boldsymbol{x}_j)
\tag{50}
\end{equation}

但可以通过降维减少存储：

\begin{equation}
\tilde{\boldsymbol{x}}_i^{(k)} = \boldsymbol{x}_i\boldsymbol{U}_{:k}, \quad k = 256 \ll d = 768
\tag{51}
\end{equation}

### 8. 超参数调节的理论指导

#### 8.1 参数空间的几何结构

$(\beta, \gamma)$ 参数空间的四个顶点：

1. $(0, 0)$：正交旋转，保持所有性质
2. $(1, 0)$：中心化+旋转，移除均值偏移
3. $(0, 1)$：不中心化白化（较少使用）
4. $(1, 1)$：标准BERT-whitening

**插值性质**：中间值 $(0 < \beta < 1, 0 < \gamma < 1)$ 在这些极端情况之间平滑过渡。

#### 8.2 性能曲面的直觉

对于不同任务，最优 $(\beta, \gamma)$ 不同：

- 无监督BERT：倾向于 $(\beta, \gamma) \to (1, 1)$（需要强去相关）
- 有监督SimBERT：倾向于 $(\beta, \gamma) \to (0, 0)$（保持原有结构）
- 中等监督模型：可能需要中间值

#### 8.3 降维维度k的选择

根据累积方差贡献率选择 $k$：

\begin{equation}
k = \min\left\{j : \frac{\sum_{i=1}^j \lambda_i}{\sum_{i=1}^d \lambda_i} \geq \theta\right\}
\tag{52}
\end{equation}

常用阈值：$\theta = 0.95$ 或 $0.99$。

对于BERT（768维），通常 $k = 256$ 或 $k = 384$ 即可保留大部分信息。

### 9. 完整算法流程

**输入**：句向量集合 $\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_N\}$，超参数 $\beta, \gamma, k$

**步骤1**：计算均值
\begin{equation}
\boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \boldsymbol{x}_i
\tag{53}
\end{equation}

**步骤2**：计算协方差矩阵
\begin{equation}
\boldsymbol{\Sigma} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{x}_i - \beta\boldsymbol{\mu})(\boldsymbol{x}_i - \beta\boldsymbol{\mu})^{\top}
\tag{54}
\end{equation}

**步骤3**：SVD分解
\begin{equation}
\boldsymbol{\Sigma} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}
\tag{55}
\end{equation}

**步骤4**：白化变换
\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}_{:k}\boldsymbol{\Lambda}_k^{-\gamma/2}
\tag{56}
\end{equation}

**输出**：降维白化后的向量 $\{\tilde{\boldsymbol{x}}_1, \ldots, \tilde{\boldsymbol{x}}_N\}$

### 10. 数值计算示例

假设 $d = 4$，$N = 3$，句向量为：

\begin{equation}
\boldsymbol{X} = \begin{bmatrix}
1 & 2 & 3 & 4 \\
2 & 3 & 4 & 5 \\
3 & 4 & 5 & 6
\end{bmatrix}
\tag{57}
\end{equation}

**步骤1**：均值
\begin{equation}
\boldsymbol{\mu} = \frac{1}{3}[6, 9, 12, 15] = [2, 3, 4, 5]
\tag{58}
\end{equation}

**步骤2**：中心化（$\beta = 1$）
\begin{equation}
\boldsymbol{X}_c = \begin{bmatrix}
-1 & -1 & -1 & -1 \\
0 & 0 & 0 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
\tag{59}
\end{equation}

**步骤3**：协方差矩阵
\begin{equation}
\boldsymbol{\Sigma} = \frac{1}{3}\boldsymbol{X}_c^{\top}\boldsymbol{X}_c = \frac{2}{3}\begin{bmatrix}
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1
\end{bmatrix}
\tag{60}
\end{equation}

**步骤4**：特征值分解
\begin{equation}
\lambda_1 = \frac{8}{3}, \quad \lambda_2 = \lambda_3 = \lambda_4 = 0
\tag{61}
\end{equation}

这个例子中数据是秩1的（所有向量共线），只有一个非零特征值。

**步骤5**：白化（保留 $k=1$ 维）
\begin{equation}
\tilde{\boldsymbol{x}}_i = (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{u}_1 \cdot \lambda_1^{-1/2} = (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{u}_1 \cdot \sqrt{\frac{3}{8}}
\tag{62}
\end{equation}

其中 $\boldsymbol{u}_1 = \frac{1}{2}[1, 1, 1, 1]^{\top}$。

### 11. 理论总结与实践建议

**理论要点**：
1. BERT-whitening通过PCA+白化解决各向异性问题
2. 超参数 $\beta$ 控制中心化程度，$\gamma$ 控制方差归一化程度
3. $(\beta, \gamma) = (0, 0)$ 提供"免费午餐"式的降维，保持原有性质
4. 降维通过保留主成分实现，信息损失可控

**实践建议**：
1. **无监督模型**（如BERT）：使用 $\beta = \gamma = 1$，强去相关
2. **有监督模型**（如SimBERT）：使用 $\beta = \gamma = 0$，仅降维不改变分布
3. **中等监督模型**：在验证集上网格搜索 $\beta, \gamma \in \{0, 0.5, 1\}$
4. **降维维度**：$k = 256$ 对于768维BERT通常足够（保留约90%方差）
5. **数值稳定**：添加正则化 $\epsilon \approx 10^{-5}$ 避免小特征值问题

**复杂度分析**：
- 时间复杂度：$O(Nd^2 + d^3)$（协方差计算+SVD）
- 空间复杂度：$O(Nd + d^2)$（存储数据+协方差矩阵）
- 推理复杂度：$O(dk)$（每个向量的变换）

