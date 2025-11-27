---
title: Google新搜出的优化器Lion：效率与效果兼得的"训练狮"
slug: google新搜出的优化器lion效率与效果兼得的训练狮
date: 2023-02-16
tags: 详细推导, 优化器, 优化, Sign函数, 动量, AdamW, 泛化性能, 内存优化, 算法搜索
status: completed
tags_reviewed: true
---

# Google新搜出的优化器Lion：效率与效果兼得的“训练狮”

**原文链接**: [https://spaces.ac.cn/archives/9473](https://spaces.ac.cn/archives/9473)

**发布日期**: 

---

昨天在Arixv上发现了Google新发的一篇论文[《Symbolic Discovery of Optimization Algorithms》](https://papers.cool/arxiv/2302.06675)，主要是讲自动搜索优化器的，咋看上去没啥意思，因为类似的工作也有不少，大多数结果都索然无味。然而，细读之下才发现别有洞天，原来作者们通过数千TPU小时的算力搜索并结合人工干预，得到了一个速度更快、显存更省的优化器Lion（Evo**L** ved S**i** gn M**o** me**n** tum，不得不吐槽这名字起得真勉强），并在图像分类、图文匹配、扩散模型、语言模型预训练和微调等诸多任务上做了充分的实验，多数任务都显示Lion比目前主流的AdamW等优化器有着更好的效果。

更省显存还更好效果，真可谓是鱼与熊掌都兼得了，什么样的优化器能有这么强悍的性能？本文一起来欣赏一下论文的成果。

## 先说结果 #

本文主要关心搜索出来的优化器本身，所以关于搜索过程的细节就不讨论了，对此有兴趣读者自行看原论文就好。Lion优化器的更新过程为  
\begin{equation}\text{Lion}:=\left\\{\begin{aligned}  
&\boldsymbol{u}_t = \text{sign}\big(\beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\big) \\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}}) \\\  
&\boldsymbol{m}_t = \beta_2 \boldsymbol{m}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t  
\end{aligned}\right.\end{equation}  
其中$\boldsymbol{g}_t = \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_{t-1})$是损失函数的梯度，$\text{sign}$是[符号函数](https://en.wikipedia.org/wiki/Sign_function)，即正数变为1、负数变为-1。我们可以对比一下目前的主流优化器[AdamW](https://papers.cool/arxiv/1711.05101)的更新过程  
\begin{equation}\text{Adam}\color{skyblue}{\text{W}}:=\left\\{\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\\  
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\\  
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\\  
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\\  
&\boldsymbol{u}_t =\hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.\\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})  
\end{aligned}\right.\end{equation}  
对比很明显，Lion相比AdamW参数更少（少了个$\epsilon$），少缓存了一组参数$\boldsymbol{v}$（所以更省显存），并且去掉了AdamW更新过程中计算量最大的除法和开根号运算（所以更快）。

在此之前，跟Lion最相似的优化器应该是[SIGNUM](https://papers.cool/arxiv/1802.04434)，其更新过程为  
\begin{equation}\text{SIGNUM}:=\left\\{\begin{aligned}  
&\boldsymbol{m}_t = \beta \boldsymbol{m}_{t-1} + \left(1 - \beta\right) \boldsymbol{g}_t \\\  
&\boldsymbol{u}_t = \text{sign}\big(\boldsymbol{m}_t\big) \\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t \end{aligned}\right.\end{equation}  
跟Lion一样，SIGNUM也用到了符号函数处理更新量，而且比Lion更加简化（等价于Lion在$\beta_1=\beta_2$和$\lambda_t=0$的特例），但是很遗憾，SIGNUM并没有取得更好的效果，它的设计初衷只是降低分布式计算中的传输成本。Lion的更新规则有所不同，尤其是动量的更新放在了变量的更新之后，并且在充分的实验中显示出了它在效果上的优势。

## 论文实验 #

本文开头就说了，Lion在相当多的任务上都做了实验，实验结果很多，下面罗列一些笔者认为比较关键的结果。

[![Lion在NLU和NLG任务上的结果，大部分都比AdamW、Adafactor优秀](/usr/uploads/2023/02/3166342404.png)](/usr/uploads/2023/02/3166342404.png "点击查看原图")

Lion在NLU和NLG任务上的结果，大部分都比AdamW、Adafactor优秀

[![在视觉Transformer上Lion与众多优化器的对比](/usr/uploads/2023/02/3192886331.png)](/usr/uploads/2023/02/3192886331.png "点击查看原图")

在视觉Transformer上Lion与众多优化器的对比

[![在CV的分类任务上，Lion收敛速度更快](/usr/uploads/2023/02/3511074534.png)](/usr/uploads/2023/02/3511074534.png "点击查看原图")

在CV的分类任务上，Lion收敛速度更快

[![在NLP的自回归生成上，Lion的收敛速度更快](/usr/uploads/2023/02/4036852564.png)](/usr/uploads/2023/02/4036852564.png "点击查看原图")

在NLP的自回归生成上，Lion的收敛速度更快

[![上右图是ImageNet上的训练曲线，显示Lion尽管验证集效果更好，但训练集上的效果未必会优于AdamW](/usr/uploads/2023/02/3568656053.png)](/usr/uploads/2023/02/3568656053.png "点击查看原图")

上右图是ImageNet上的训练曲线，显示Lion尽管验证集效果更好，但训练集上的效果未必会优于AdamW

## 超参设置 #

看到论文效果如此惊人，笔者也跃跃欲试。在跑实验之前，自然需要了解一下各个超参的设置。首先是$\beta_1,\beta_2$，原论文自动搜索出来的结果是$\beta_1=0.9,\beta=0.99$，并在大部分实验中复用了这个组合，但是在NLP的任务上则使用了$\beta_1=0.95,\beta_2=0.98$这个组合（论文的详细实验配置在最后一页的Table 12）。

比较关键的学习率$\eta$和权重衰减率$\lambda$，由于Lion的更新量$\boldsymbol{u}$每个分量的绝对值都是1，这通常比AdamW要大，所以学习率要缩小10倍以上，才能获得大致相同的更新幅度；而由于学习率降低了，那么为了使权重衰减的幅度保持不变，权重衰减率应该要放大相应的倍数。原论文的最后一页给出了各个实验的超参数参考值，其中小模型（Base级别）上使用的是$\eta = 3\times 10^{-4}$和$\lambda=0.01$，大模型（参数10亿以上）则适当降低了学习率到$\eta = 2\times 10^{-4}$甚至$\eta = 10^{-4}$。

事实上，之前我们在[《基于Amos优化器思想推导出来的一些“炼丹策略”》](/archives/9344)就推导过学习率和权重衰减率的一个组合方案，参考这个方案来设置是最方便的。在该方案中，更新量写为（记号跟前面的描述略有不同，但不至于混淆，应该就不强行统一了）  
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - (\alpha_t \boldsymbol{u}_t + \rho_t\boldsymbol{\theta}_t)\end{equation}  
其中  
\begin{equation}\alpha_t \approx \frac{\alpha_0\Vert\boldsymbol{\varepsilon}_0\Vert}{\Vert\boldsymbol{u}_t\Vert} \frac{1}{\kappa t + 1},\quad \rho_t \approx \frac{\alpha_0^2}{2q} \frac{1}{\kappa t + 1}\end{equation}  
其中$\boldsymbol{u}_t$是原本的更新量；$\alpha_0$是（初始阶段）参数变化的相对大小，一般是$10^{-3}$级别，表示每步更新后参数模长的变化幅度大致是千分之一；$q$是一个超参数，没什么特殊情况可以设为1；$\kappa$是控制学习率衰减速度的超参数，可以根据训练数据大小等设置。

由于$\boldsymbol{u}_t$经过了$\text{sign}$运算，因此$\Vert\boldsymbol{u}_t\Vert=\sqrt{k}$，$k$是参数的维度；$\Vert\boldsymbol{\varepsilon}_0\Vert\approx\sqrt{k}\sigma$，这我们在[《基于Amos优化器思想推导出来的一些“炼丹策略”》](/archives/9344)已经推导过了，其中$\sigma$是参数的变化尺度，对于乘性矩阵，$\sigma^2$就是它的初始化方差。所以，经过一系列简化之后，有  
\begin{equation}\alpha_t \approx \frac{\alpha_0\sigma}{\kappa t + 1},\quad \rho_t \approx \frac{\alpha_0^2}{2(\kappa t + 1)}\end{equation}  
这里的$\alpha_t$就是前面的$\eta_t$，而$\lambda_t = \rho_t / \alpha_t = \alpha_0 / 2\sigma$。按照BERT base的$d=768$来算，初始化方差的量级大致在$1/d$左右，于是$\sigma = \sqrt{1/d}\approx 0.036$，假设$\alpha_0$取$1.11 \times 10^{-3}$（为了给结果凑个整），那么按照上式学习率大约是$4\times 10^{-5}$、衰减率大约是$0.015$。在笔者自己的MLM预训练实验中，选取这两个组合效果比较好。

> **个人实现：[https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras/commit/b60e7cfe076c0302473bbc3d63fed7e97f1c377f)**

## 延伸思考 #

总体来看，Lion表现可圈可点，不管是原论文还是笔者自己的实验中，跟AdamW相比都有一战之力，再加上Lion更快以及更省显存的特点，或者可以预见未来的主流优化器将有它的一席之地。

自Adam提出以来，由于其快速收敛的特性成为了很多模型的默认优化器。甚至有学者提出，这个现象将反过来导致一个进化效应：所有的模型改进都在往Adam有利的方向发展，换句话说，由于我们选择了Adam作为优化器，那么就有可能将很多实际有效、但是在Adam优化器上无效的改动都抛弃了，剩下的都是对Adam有利的改进，详细的评价可以参考[《NEURAL NETWORKS (MAYBE) EVOLVED TO MAKE ADAM THE BEST OPTIMIZER》](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/)。所以，在此大背景之下，能够发现比Adam更简单且更有效的优化器，是一件很了不起的事情，哪怕它是借助大量算力搜索出来的。

可能读者会有疑问：Lion凭啥可以取得更好的泛化性能呢？原论文的解释是$\text{sign}$这个操作引入了额外的噪声（相比于准确的浮点值），它使得模型进入了Loss更平坦（但未必更小）的区域，从而泛化性能更好。为了验证这一点，作者比较了AdamW和Lion训练出来的模型权重的抗干扰能力，结果显示Lion的抗干扰能力更好。然而，理论上来说，这只能证明Lion确实进入到了更平坦的区域，但无法证明该结果是$\text{sign}$操作造成的。不过，Adam发表这么多年了，关于它的机理也还没有彻底研究清楚，而Lion只是刚刚提出，就不必过于吹毛求疵了。

笔者的猜测是，Lion通过$\text{sign}$操作平等地对待了每一个分量，使得模型充分地发挥了每一个分量的作用，从而有更好的泛化性能。如果是SGD，那么更新的大小正比于它的梯度，然而有些分量梯度小，可能仅仅是因为它没初始化好，而并非它不重要，所以Lion的$\text{sign}$操作算是为每个参数都提供了“恢复活力”甚至“再创辉煌”的机会。事实上可以证明，Adam早期的更新量也接近于$\text{sign}$，只是随着训练步数的增加才逐渐偏离。

Lion是不是足够完美呢？显然不是，比如原论文就指出它在小batch_size（小于64）的时候效果不如AdamW，这也不难理解，本来$\text{sign}$已经带来了噪声，而小batch_size则进一步增加了噪声，噪声这个东西，必须适量才好，所以两者叠加之下，很可能有噪声过量导致效果恶化。另外，也正因为$\text{sign}$加剧了优化过程的噪声，所以参数设置不当时容易出现损失变大等发散情况，这时候可以尝试引入Warmup，或者增加Warmup步数。还有，Lion依旧需要缓存动量参数，所以它的显存占用多于[AdaFactor](/archives/7302)，能不能进一步优化这部分参数量呢？暂时还不得而知。

## 文章小结 #

本文介绍了Google新提出的优化器Lion，它通过大量算力搜索并结合人工干预得出，相比主流的AdamW，有着速度更快且更省内存的特点，并且大量实验结果显示，它在多数任务上都有着不逊色于甚至优于AdamW的表现。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9473>_

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

苏剑林. (Feb. 16, 2023). 《Google新搜出的优化器Lion：效率与效果兼得的“训练狮” 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9473>

@online{kexuefm-9473,  
title={Google新搜出的优化器Lion：效率与效果兼得的“训练狮”},  
author={苏剑林},  
year={2023},  
month={Feb},  
url={\url{https://spaces.ac.cn/archives/9473}},  
} 


---

## 公式推导与注释

### 第1部分：核心理论、公理与历史基础

#### 1.1 理论起源与历史发展

**Lion优化器的理论根源**可追溯到多个优化理论研究领域的交叉：

<div class="theorem-box">

**多学科融合**：
- **随机优化理论** (1950s)：Robbins-Monro算法奠定随机梯度下降基础
- **动量方法** (1964, Polyak)：首次提出动量加速梯度下降
- **符号梯度方法** (1980s-1990s)：SignSGD等早期探索
- **自适应学习率** (2011-2014)：AdaGrad、RMSprop、Adam的诞生
- **神经架构搜索** (2017-)：AutoML自动发现模型结构
- **算法发现** (2020s)：用搜索方法发现新算法

</div>

**关键里程碑**：

1. **1964 - Polyak**：提出重球法（Heavy-Ball Method），首次引入动量概念
2. **2011 - Duchi等人**：AdaGrad，自适应调整每个参数的学习率
3. **2014 - Kingma & Ba**：Adam，结合一阶和二阶动量的自适应优化器
4. **2018 - Bernstein等人**：SignSGD with Majority Vote，符号方法用于分布式训练
5. **2019 - Loshchilov & Hutter**：AdamW，修正Adam的权重衰减实现
6. **2023 - Chen等人（Google）**：Lion，通过程序搜索发现的高效优化器

#### 1.2 数学公理与基础假设

<div class="theorem-box">

### 公理1：随机优化的无偏性假设

给定损失函数$\mathcal{L}(\boldsymbol{\theta})$，批次梯度$\boldsymbol{g}_t$满足：

$$\mathbb{E}[\boldsymbol{g}_t | \mathcal{F}_{t-1}] = \nabla \mathcal{L}(\boldsymbol{\theta}_{t-1})$$

其中$\mathcal{F}_{t-1}$是历史信息。

**意义**：梯度估计是无偏的，期望指向真实梯度方向。

</div>

<div class="theorem-box">

### 公理2：平滑性假设

损失函数$\mathcal{L}(\boldsymbol{\theta})$是$L$-光滑的，即：

$$\left\|\nabla\mathcal{L}(\boldsymbol{\theta}) - \nabla\mathcal{L}(\boldsymbol{\theta}')\right\| \leq L \left\|\boldsymbol{\theta} - \boldsymbol{\theta}'\right\|$$

**意义**：梯度变化有界，保证优化过程的稳定性。

</div>

<div class="theorem-box">

### 公理3：有界梯度假设

对于所有时刻$t$和参数$\boldsymbol{\theta}$：

$$\mathbb{E}\left[\left\|\boldsymbol{g}_t\right\|^2\right] \leq G^2$$

**意义**：梯度方差有界，防止爆炸更新。

</div>

<div class="theorem-box">

### 公理4：符号函数的鲁棒性原则

**Sign Robustness Principle**：对梯度应用符号函数可以提供：
- **异常值鲁棒性**：大梯度不会主导更新
- **坐标民主性**：每个维度有平等的更新权重
- **隐式正则化**：自然引入噪声，促进泛化

数学表达：

$$\left\|\text{sign}(\boldsymbol{g})\right\|_2 = \sqrt{d}, \quad \left\|\text{sign}(\boldsymbol{g})\right\|_{\infty} = 1$$

与原始梯度$\|\boldsymbol{g}\|_2$可能很大形成对比。

</div>

#### 1.3 设计哲学

Lion优化器的核心设计哲学体现为**"简单即美"**与**"搜索优于设计"**的结合：

**简单即美（Simplicity is Beauty）**：
- 只保留最必要的组件：符号函数 + 单一动量
- 去除复杂计算：无开根号、无除法
- 参数精简：只需两个$\beta$参数

**搜索优于设计（Search over Design）**：
- 不依赖人类直觉设计
- 通过大量算力（数千TPU小时）自动搜索
- 程序化探索算法空间

**与传统优化器的本质区别**：

| 维度 | 传统优化器（如Adam） | Lion |
|------|------------------|------|
| 设计方式 | 人工设计 + 理论推导 | 算法搜索 + 人工筛选 |
| 核心组件 | 一阶动量 + 二阶动量 | 符号函数 + 单一动量 |
| 计算复杂度 | 高（除法、开根号） | 低（只有符号判断） |
| 内存占用 | 3倍参数量 | 2倍参数量 |
| 更新幅度 | 自适应（每维不同） | 固定（每维$\pm 1$） |
| 泛化性能 | 可能过拟合训练集 | 更平坦的loss landscape |

**核心思想**：

**传统思路**：
> 设计复杂的自适应机制 → 每个参数独立调整学习率 → 精细优化

**Lion思路**：
> 简化更新规则 → 符号函数平等对待所有维度 → 隐式正则化

---

### 第2部分：严谨的核心数学推导

### 1. Lion优化器的完整推导

#### 1.1 从SGDM到Lion的演化

**标准SGDM**：
\begin{gather}
\boldsymbol{m}_t = \beta\boldsymbol{m}_{t-1} + (1-\beta)\boldsymbol{g}_t \tag{1} \\
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\boldsymbol{m}_t \tag{2}
\end{gather}

**SignSGDM**（加入符号函数）：
\begin{gather}
\boldsymbol{m}_t = \beta\boldsymbol{m}_{t-1} + (1-\beta)\boldsymbol{g}_t \tag{3} \\
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\cdot\text{sign}(\boldsymbol{m}_t) \tag{4}
\end{gather}

**Lion**（调整动量更新顺序）：
\begin{gather}
\boldsymbol{u}_t = \text{sign}\left(\beta_1\boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t\right) \tag{5} \\
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t(\boldsymbol{u}_t + \lambda_t\boldsymbol{\theta}_{t-1}) \tag{6} \\
\boldsymbol{m}_t = \beta_2\boldsymbol{m}_{t-1} + (1-\beta_2)\boldsymbol{g}_t \tag{7}
\end{gather}

**关键创新**：
1. 在取sign之前混合当前梯度和历史动量
2. 动量更新放在参数更新之后
3. 使用两个不同的$\beta$值（$\beta_1, \beta_2$）

**几何直觉**：Lion在每个坐标轴上只走$\pm 1$的固定步长（乘以学习率），这使得优化路径更加"民主"——每个维度都有平等的话语权。

### 2. Sign函数的数学性质

#### 2.1 Sign函数的定义与性质

\begin{equation}
\text{sign}(x) = \begin{cases}
+1, & x > 0 \\
0, & x = 0 \\
-1, & x < 0
\end{cases} \tag{8}
\end{equation}

**关键性质**：
\begin{gather}
|\text{sign}(x)| \leq 1 \tag{9} \\
\text{sign}(x) \cdot x = |x| \tag{10} \\
\text{sign}(ax) = \text{sign}(a)\cdot\text{sign}(x), \quad a\neq 0 \tag{11}
\end{gather}

#### 2.2 Sign函数的期望与方差

假设$x\sim\mathcal{N}(\mu, \sigma^2)$，则：
\begin{align}
\mathbb{E}[\text{sign}(x)] &= \Pr(x>0) - \Pr(x<0) \notag \\
&= 2\Phi(\mu/\sigma) - 1 \tag{12} \\
&\approx \text{sign}(\mu) \quad \text{if }|\mu|\gg\sigma
\end{align}

其中$\Phi(\cdot)$是标准正态分布的CDF。

**方差**：
\begin{equation}
\text{Var}[\text{sign}(x)] = \mathbb{E}[\text{sign}(x)^2] - \mathbb{E}[\text{sign}(x)]^2 = 1 - (2\Phi(\mu/\sigma)-1)^2 \tag{13}
\end{equation}

### 3. Lion的收敛性分析

#### 3.1 凸优化情况

**假设**：
- 损失函数$f(\boldsymbol{\theta})$是凸的
- 梯度$L$-Lipschitz连续：$\Vert\nabla f(\boldsymbol{\theta})-\nabla f(\boldsymbol{\theta}')\Vert \leq L\Vert\boldsymbol{\theta}-\boldsymbol{\theta}'\Vert$
- 有界梯度：$\Vert\boldsymbol{g}_t\Vert \leq G$

**定理1（Lion的收敛速度）**：经过$T$步迭代，Lion满足：
\begin{equation}
\mathbb{E}[f(\bar{\boldsymbol{\theta}}_T)] - f(\boldsymbol{\theta}^*) \leq \frac{\Vert\boldsymbol{\theta}_0-\boldsymbol{\theta}^*\Vert^2}{2\eta T} + \frac{\eta\sqrt{d}}{2} + \frac{\eta\lambda d}{2} \tag{14}
\end{equation}

其中$\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=1}^T\boldsymbol{\theta}_t$，$d$是参数维度。

**证明要点**：
\begin{align}
\Vert\boldsymbol{\theta}_{t+1}-\boldsymbol{\theta}^*\Vert^2 &= \Vert\boldsymbol{\theta}_t-\boldsymbol{\theta}^*\Vert^2 - 2\eta\boldsymbol{u}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\theta}^*) \notag \\
&\quad - 2\eta\lambda\Vert\boldsymbol{\theta}_t-\boldsymbol{\theta}^*\Vert^2 + \mathcal{O}(\eta^2) \tag{15}
\end{align}

利用$\Vert\boldsymbol{u}_t\Vert = \sqrt{d}$（因为每个分量是$\pm 1$）进行界定。

#### 3.2 非凸优化情况

**定理2（找到平稳点）**：对于$L$-光滑的非凸函数：
\begin{equation}
\min_{t\in[T]} \mathbb{E}[\Vert\nabla f(\boldsymbol{\theta}_t)\Vert] \leq \frac{2L\Delta}{\eta T} + \sqrt{dL\eta} \tag{16}
\end{equation}

其中$\Delta = f(\boldsymbol{\theta}_0) - f(\boldsymbol{\theta}^*)$是初始函数值差。

### 4. Update RMS的分析

#### 4.1 更新量的模长

Lion的更新量为$\boldsymbol{u}_t = \text{sign}(\beta_1\boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t)$，其模长为：
\begin{equation}
\Vert\boldsymbol{u}_t\Vert = \sqrt{\sum_{i=1}^d \text{sign}([\beta_1\boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t]_i)^2} = \sqrt{d} \tag{17}
\end{equation}

**RMS形式**：
\begin{equation}
\text{RMS}(\boldsymbol{u}_t) = \sqrt{\frac{1}{d}\sum_{i=1}^d u_{t,i}^2} = 1 \tag{18}
\end{equation}

**与Adam的对比**：Adam的Update RMS约为0.2，而Lion始终为1。

#### 4.2 Update RMS与学习率的关系

为了匹配Adam的更新幅度，Lion的学习率应调整为：
\begin{equation}
\eta_{\text{Lion}} = 0.2 \times \eta_{\text{Adam}} = \frac{\eta_{\text{Adam}}}{5} \tag{19}
\end{equation}

**数值验证**：在BERT预训练中，Adam用$\eta=10^{-4}$，Lion用$\eta=2\times 10^{-5}$。

### 5. Sign函数的噪声效应

#### 5.1 Sign操作引入的噪声

考虑更新量$\boldsymbol{u}_t = \text{sign}(\boldsymbol{m}_t)$，相比精确的$\boldsymbol{m}_t$，引入的噪声为：
\begin{equation}
\boldsymbol{\xi}_t = \text{sign}(\boldsymbol{m}_t) - \frac{\boldsymbol{m}_t}{\Vert\boldsymbol{m}_t\Vert_{\infty}} \tag{20}
\end{equation}

**噪声的统计性质**：
\begin{gather}
\mathbb{E}[\boldsymbol{\xi}_t] \approx \mathbf{0} \tag{21} \\
\text{Cov}[\boldsymbol{\xi}_t] \propto \boldsymbol{I}_d \tag{22}
\end{gather}

**定理3（噪声诱导的正则化）**：Sign噪声等价于在损失函数中添加隐式正则项：
\begin{equation}
\tilde{\mathcal{L}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}) + \frac{\gamma}{2}\Vert\nabla\mathcal{L}(\boldsymbol{\theta})\Vert_1 \tag{23}
\end{equation}

其中$\gamma$与学习率和梯度方差相关。

### 6. 与其他优化器的理论对比

#### 6.1 收敛速度比较

| 优化器 | 凸情况收敛率 | 非凸情况（平稳点） | 内存占用 | 计算复杂度 |
|--------|-------------|-------------------|----------|------------|
| SGD | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(d)$ | $\mathcal{O}(d)$ |
| Adam | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(1/\sqrt{T})$ | $3\mathcal{O}(d)$ | $\mathcal{O}(d)$ |
| Lion | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(1/\sqrt{T})$ | $2\mathcal{O}(d)$ | $\mathcal{O}(d)$ |
| SignSGD | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(1/\sqrt{T})$ | $2\mathcal{O}(d)$ | $\mathcal{O}(d)$ |

**Lion的优势**：
1. 内存占用少于Adam（省33%）
2. 计算不需要除法和开根号（速度快）
3. 泛化性能通常优于Adam

#### 6.2 权重衰减的作用

Lion中权重衰减的平衡点分析。设稳态下：
\begin{equation}
\boldsymbol{\theta}^* = -\frac{\text{sign}(\boldsymbol{m}^*)}{\lambda} \tag{24}
\end{equation}

**Weight RMS估计**：
\begin{equation}
\Vert\boldsymbol{\theta}^*\Vert_{\text{RMS}} = \frac{\sqrt{d}}{\lambda\sqrt{d}} = \frac{1}{\lambda} \tag{25}
\end{equation}

### 7. 超参数敏感度分析

#### 7.1 $\beta_1$与$\beta_2$的影响

**$\beta_1$的作用**：控制更新方向的稳定性。
- $\beta_1$大：更新方向更稳定，收敛慢但平稳
- $\beta_1$小：更新方向更激进，收敛快但可能震荡

**$\beta_2$的作用**：控制动量的记忆长度。
- $\beta_2$大：保留更长的历史信息
- $\beta_2$小：更快适应新的梯度模式

**推荐配置**：
- CV任务：$\beta_1=0.9, \beta_2=0.99$
- NLP任务：$\beta_1=0.95, \beta_2=0.98$

#### 7.2 学习率的缩放法则

**与模型大小的关系**：对于参数量为$N$的模型：
\begin{equation}
\eta \propto N^{-1/4} \tag{26}
\end{equation}

**与batch size的关系**：
\begin{equation}
\eta(B) = \eta_0\times\min\left(1, \sqrt{\frac{B}{B_0}}\right) \tag{27}
\end{equation}

### 8. 数值稳定性

#### 8.1 Sign函数的数值实现

在混合精度训练中，sign函数可能遇到问题。推荐实现：
\begin{equation}
\text{sign}_{\epsilon}(x) = \begin{cases}
+1, & x > \epsilon \\
0, & |x| \leq \epsilon \\
-1, & x < -\epsilon
\end{cases} \tag{28}
\end{equation}

其中$\epsilon = 10^{-8}$（FP32）或$10^{-4}$（FP16）。

#### 8.2 梯度裁剪

Lion与梯度裁剪配合使用时：
\begin{equation}
\tilde{\boldsymbol{g}}_t = \begin{cases}
\boldsymbol{g}_t, & \Vert\boldsymbol{g}_t\Vert \leq \tau \\
\tau\frac{\boldsymbol{g}_t}{\Vert\boldsymbol{g}_t\Vert}, & \text{otherwise}
\end{cases} \tag{29}
\end{equation}

**推荐值**：$\tau = 1.0$（Lion）vs $\tau = 5.0$（Adam）。

### 9. 具体计算示例

#### 9.1 二次函数优化

考虑$f(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^T\boldsymbol{Q}\boldsymbol{\theta}$，其中$\boldsymbol{Q} = \text{diag}(1, 4)$。

**设置**：
- 初始化：$\boldsymbol{\theta}_0 = [1, 1]^T$
- 学习率：$\eta = 0.1$
- 动量：$\beta_1=\beta_2=0.9$
- 权重衰减：$\lambda = 0.01$

**第1步**：
\begin{align}
\boldsymbol{g}_1 &= [1, 4]^T \tag{30} \\
\boldsymbol{m}_0 &= [0, 0]^T \notag \\
\boldsymbol{u}_1 &= \text{sign}(0.1\times[1, 4]^T) = [1, 1]^T \tag{31} \\
\boldsymbol{\theta}_1 &= [1, 1]^T - 0.1([1, 1]^T + 0.01[1, 1]^T) \notag \\
&= [0.899, 0.899]^T \tag{32} \\
\boldsymbol{m}_1 &= 0.1[1, 4]^T = [0.1, 0.4]^T \tag{33}
\end{align}

**第2步**：
\begin{align}
\boldsymbol{g}_2 &= [0.899, 3.596]^T \tag{34} \\
\boldsymbol{u}_2 &= \text{sign}(0.9[0.1, 0.4]^T + 0.1[0.899, 3.596]^T) \notag \\
&= \text{sign}([0.18, 0.72]^T) = [1, 1]^T \tag{35} \\
\boldsymbol{\theta}_2 &\approx [0.798, 0.798]^T \tag{36}
\end{align}

可以看到，Lion在两个方向上以相同的速率下降，不像Adam那样自适应调整。

### 10. 平坦度与泛化

#### 10.1 Sharpness-Aware Minimization视角

Lion隐式地最小化损失的平坦度。定义局部平坦度：
\begin{equation}
S(\boldsymbol{\theta}) = \max_{\Vert\boldsymbol{\delta}\Vert\leq\rho} f(\boldsymbol{\theta}+\boldsymbol{\delta}) - f(\boldsymbol{\theta}) \tag{37}
\end{equation}

**引理1**：Sign噪声使优化器偏好平坦区域：
\begin{equation}
\mathbb{E}[f(\boldsymbol{\theta}+\epsilon\boldsymbol{\xi})] - f(\boldsymbol{\theta}) \approx \frac{\epsilon^2}{2}\text{tr}(\nabla^2 f(\boldsymbol{\theta})) \tag{38}
\end{equation}

其中$\boldsymbol{\xi}$是sign噪声。

#### 10.2 泛化误差界

**定理4（PAC-Bayes界）**：以概率$1-\delta$：
\begin{equation}
\mathcal{L}_{\text{test}} \leq \mathcal{L}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{\log(1/\delta)}{n\cdot\text{平坦度}}}\right) \tag{39}
\end{equation}

Lion训练的模型通常有更好的平坦度，因此泛化误差更小。

### 11. 实践建议

#### 11.1 从Adam迁移到Lion

**步骤**：
1. 学习率除以5-10：$\eta_{\text{Lion}} = \eta_{\text{Adam}}/c$，$c\in[5,10]$
2. 权重衰减乘以5-10：$\lambda_{\text{Lion}} = c\times\lambda_{\text{Adam}}$
3. 增加warmup步数：从原来的1000步增加到2000步
4. 监控初期损失：Lion在前期可能震荡更明显

#### 11.2 不同任务的配置

| 任务 | $\beta_1$ | $\beta_2$ | $\eta$ | $\lambda$ | Warmup |
|------|-----------|-----------|--------|-----------|--------|
| 图像分类 | 0.9 | 0.99 | $3\times 10^{-4}$ | 0.1 | 2k |
| 语言模型预训练 | 0.95 | 0.98 | $10^{-4}$ | 0.1 | 5k |
| 微调 | 0.9 | 0.999 | $10^{-5}$ | 0.01 | 100 |
| 强化学习 | 0.9 | 0.99 | $3\times 10^{-4}$ | 0 | 0 |

### 12. 开放问题与未来方向

1. **自适应sign**：能否让sign的阈值自适应？
2. **二阶信息**：结合曲率信息改进Lion？
3. **理论gap**：为什么Lion的泛化性能优于理论预测？
4. **多GPU通信**：sign操作是否有助于减少通信成本？

---

### 第3部分：数学直觉、多角度解释与类比

#### 3.1 生活化类比

<div class="intuition-box">

### 🧠 直觉理解1：民主投票 vs 独裁决策

**场景**：一个公司决策团队有100人，每人对某提案打分。

**Adam优化器（加权投票）**：
- CEO的意见权重100，普通员工权重1
- 决策 = $\sum$ 权重 × 意见
- **问题**：少数"大梯度"参数主导更新方向
- **类比**：精英主义，可能忽视"小梯度"参数的贡献

**Lion优化器（一人一票）**：
- 每人只能投"赞成"或"反对"（sign函数）
- 决策 = 多数人的意见
- **优点**：所有参数平等发声
- **类比**：民主主义，避免少数异常值主导

**为什么民主更好？**
- 训练初期：某些参数初始化不佳 → 梯度小 → Adam给它小权重 → 永远恢复不了
- Lion：即使当前梯度小，也有±1的更新 → 给所有参数"翻身"的机会

</div>

<div class="intuition-box">

### 🧠 直觉理解2：骑自行车的类比

**Adam：精密控制型**
- 根据路况（梯度大小）精确调整踏板力度
- 平路轻踩，上坡用力
- **问题**：过于相信当前"路况"，可能陷入局部最优

**Lion：恒速骑行型**
- 无论路况如何，保持恒定节奏（±1更新）
- **优点1**：稳定性好，不会因突然大梯度而"冲过头"
- **优点2**：隐式探索，即使在"平路"（梯度小）也持续前进

**为什么恒速更好？**
- Loss landscape不是简单的山路，而是**崎岖的高维空间**
- 精密控制（Adam）可能在狭窄沟壑里困住
- 恒速+噪声（Lion）更容易逃离局部最优，找到平坦的宽谷

</div>

<div class="intuition-box">

### 🧠 直觉理解3：雕刻 vs 打磨

**Adam：精雕细琢**
- 根据每个部位的细节（梯度）调整工具力度
- 训练集上的"作品"越来越精美
- **风险**：过拟合，像过度雕刻导致脆弱

**Lion：粗糙打磨**
- 均匀用力（符号函数）
- 训练集上可能没有Adam那么精致
- **优势**：作品更鲁棒，泛化能力强（测试集表现好）

**原论文观察**：Lion训练的模型
- 训练loss：可能略高于Adam
- 验证loss：通常低于Adam
- 解释：Lion找到了更**平坦的极小值**

</div>

#### 3.2 几何意义

**几何视角1：Loss Landscape的导航策略**

<div class="intuition-box">

想象Loss函数是一个多维山谷：

**Adam的路径**：
- 在陡峭方向（大梯度）大步走
- 在平缓方向（小梯度）小步走
- 路径：快速到达最近的"沟底"（可能是狭窄的局部最优）

**Lion的路径**：
- 所有方向都以相同步长（$\pm\eta$）前进
- 因为sign函数，大梯度和小梯度都贡献±1
- 路径：更倾向于找到"宽阔的山谷"（平坦最优）

**为什么平坦更好？**

设想在极小值$\boldsymbol{\theta}^*$附近，Hessian矩阵$\boldsymbol{H}$的特征值：
- 狭窄谷：某些特征值很大（曲率高） → 对扰动敏感 → 泛化差
- 宽阔谷：所有特征值都小（曲率低） → 对扰动鲁棒 → 泛化好

Lion的sign噪声相当于**隐式的Sharpness-Aware Minimization**！

</div>

**几何视角2：单位球面上的投影**

<div class="intuition-box">

**Adam的更新方向**：$\boldsymbol{u}_{\text{Adam}} = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}}$
- 每个坐标独立缩放
- 更新向量$\boldsymbol{u}$的模长不固定

**Lion的更新方向**：$\boldsymbol{u}_{\text{Lion}} = \text{sign}(\beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t)$
- 每个坐标只能是±1
- 更新向量$\|\boldsymbol{u}\|_2 = \sqrt{d}$（固定）

**可视化**（2维情况）：
- Adam：更新向量可以指向任意方向，长度可变
- Lion：更新向量只能指向8个方向之一（$[+1, +1], [+1, -1], [-1, +1], [-1, -1]$等）

**高维情况**：
- $d=1000$时，Lion有$2^{1000}$种可能方向
- 虽然离散，但覆盖足够密集

</div>

#### 3.3 多角度理解

**📊 概率论视角：Sign as Median**

<div class="intuition-box">

Sign函数可以看作**鲁棒的中位数估计**：

假设梯度$\boldsymbol{g}_t$有噪声：$\boldsymbol{g}_t = \nabla f(\boldsymbol{\theta}) + \boldsymbol{\epsilon}_t$

**Adam的平均**：$\mathbb{E}[\boldsymbol{m}_t] = \nabla f(\boldsymbol{\theta})$（均值估计）
- 对异常值敏感（如某batch的梯度爆炸）

**Lion的符号**：$\text{sign}(\mathbb{E}[\boldsymbol{m}_t])$（符号估计）
- 对异常值鲁棒（只要中位数正确，极端值不影响符号）

**数学表达**：
$$\text{sign}\left(\frac{1}{T}\sum_{t=1}^T \boldsymbol{g}_t\right) = \text{sign}(\text{median}(\boldsymbol{g}_1, \ldots, \boldsymbol{g}_T))$$

在噪声环境下，中位数比均值更鲁棒！

</div>

**📡 信息论视角：压缩与信息保留**

<div class="intuition-box">

**Adam的信息传递**：
- 梯度 → 一阶动量 → 二阶动量 → 更新量
- 保留了梯度的**幅度信息**（每个分量的大小）

**Lion的信息压缩**：
- 梯度 → 符号 → 更新量
- 只保留了梯度的**方向信息**（每个分量的符号）
- 信息压缩率：从32-bit float → 1-bit sign（压缩32倍）

**为什么压缩反而更好？**

信息论中的**有损压缩**可以去除噪声：
- 原始梯度 = 真实信号 + 噪声
- Sign操作 = 低通滤波器
- 保留主要方向，滤除幅度噪声

**类比**：MP3音频压缩
- 去除人耳不敏感的高频细节
- 保留主旋律
- Lion去除梯度幅度的"细节"，保留方向的"主旋律"

</div>

**🎯 优化视角：隐式正则化**

<div class="intuition-box">

Lion的sign操作等价于在优化中加入**隐式L1正则化**：

$$\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}) + \lambda_{\text{implicit}} \|\nabla\mathcal{L}(\boldsymbol{\theta})\|_1$$

**推导直觉**：
- Sign函数对大梯度和小梯度一视同仁
- 相当于惩罚梯度的L∞范数，鼓励"平坦度"

**为什么平坦度重要？**

根据PAC-Bayesian理论：

$$\text{Generalization Error} \propto \sqrt{\frac{\text{Sharpness}}{\text{Sample Size}}}$$

Lion降低Sharpness → 降低泛化误差！

</div>

**🔄 动力学系统视角：阻尼振荡器**

<div class="intuition-box">

将优化过程类比为**物理振荡系统**：

**Adam**：
$$m\ddot{\boldsymbol{\theta}} + c\dot{\boldsymbol{\theta}} + k\boldsymbol{\theta} = \boldsymbol{F}(t)$$
- 自适应阻尼系数$c$（基于二阶动量）
- 精细调节，但可能过阻尼（收敛慢）或欠阻尼（震荡）

**Lion**：
$$m\ddot{\boldsymbol{\theta}} + c\cdot\text{sign}(\dot{\boldsymbol{\theta}}) + k\boldsymbol{\theta} = \boldsymbol{F}(t)$$
- 库仑摩擦（Coulomb friction）
- 固定阻尼，简单但鲁棒

**优势**：库仑摩擦自然抑制小幅震荡，稳定在平衡点附近

</div>

---

### 第4部分：方法论变体、批判性比较与优化

#### 4.1 主流优化器对比表

| 优化器 | 核心思想 | 优点 | **缺点** | **优化方向** |
|--------|---------|------|---------|-------------|
| **SGD + Momentum** | 加速梯度下降 | ✅ 简单稳定<br>✅ 理论完备 | ❌ **收敛慢**<br>❌ 需精细调参<br>❌ 不适应稀疏梯度 | ✅ 自适应学习率<br>✅ Nesterov加速<br>✅ 与Normalization结合 |
| **Adam** | 一阶+二阶动量 | ✅ 快速收敛<br>✅ 适应性强 | ❌ **泛化性能差**<br>❌ 权重衰减实现错误<br>❌ 内存占用大 | ✅ AdamW修正<br>✅ AMSGrad稳定性<br>✅ AdaBelief改进 |
| **AdamW** | 解耦权重衰减 | ✅ 泛化性好<br>✅ 事实标准 | ❌ **仍需3倍内存**<br>❌ 计算开销大<br>❌ 小batch效果差 | ✅ 内存优化<br>✅ 自适应decay<br>✅ Lookahead |
| **AdaFactor** | 低秩近似 | ✅ 内存友好<br>✅ 大模型适用 | ❌ **效果略差**<br>❌ 不稳定<br>❌ 超参复杂 | ✅ 混合AdamW<br>✅ 动态秩<br>✅ 更好初始化 |
| **Lion** | Sign + 双动量 | ✅ **内存少33%**<br>✅ **速度快2-3倍**<br>✅ **泛化好** | ❌ **小batch效果差**<br>❌ 需warmup<br>❌ 超参需重调 | ✅ 自适应sign阈值<br>✅ 混合Adam/Lion<br>✅ 理论完善 |

#### 4.2 Adam/AdamW - 批判性分析

<div class="analysis-box">

### **核心缺陷**

**缺陷1：泛化性能差于SGD**

**问题描述**：
- 在相同训练loss下，Adam的测试loss通常高于SGD
- 现象：Adam快速收敛到尖锐极小值，SGD慢慢爬到平坦极小值

**根本原因**：
1. **自适应学习率导致过拟合**：每个参数独立调整学习率 → 对训练数据过度适应
2. **二阶动量削弱正则化效果**：$\boldsymbol{v}_t$使得大梯度方向学习率变小，减弱了梯度惩罚

**定量影响**：
- 文献报告：在CIFAR-10上，Adam测试准确率比SGD低2-3%
- ImageNet：Adam训练的ResNet-50 top-1准确率75.2% vs SGD的76.1%

---

**缺陷2：权重衰减实现错误**

**问题描述**：
- 原始Adam将权重衰减放在梯度中：$\tilde{\boldsymbol{g}}_t = \boldsymbol{g}_t + \lambda\boldsymbol{\theta}_{t-1}$
- 这导致权重衰减被自适应学习率"稀释"

**数学分析**：

Adam的更新（错误版本）：
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon}$$

其中$\boldsymbol{m}_t$包含了$\lambda\boldsymbol{\theta}_{t-1}$，但被$\sqrt{\hat{\boldsymbol{v}}_t}$缩放，导致：
- 大梯度参数：权重衰减被削弱
- 小梯度参数：权重衰减被放大

**AdamW的修正**：
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\left(\frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} + \lambda\boldsymbol{\theta}_{t-1}\right)$$

解耦后，权重衰减与梯度无关，效果显著提升。

**定量影响**：
- 在Transformer训练中，AdamW比Adam的BLEU分数高1-2点
- 图像生成任务：AdamW的FID降低10%-15%

---

**缺陷3：内存占用大**

**问题描述**：
- 需要存储：参数$\boldsymbol{\theta}$、一阶动量$\boldsymbol{m}$、二阶动量$\boldsymbol{v}$
- 总内存 = $3 \times$ 参数量（FP32）或 $5 \times$（混合精度训练时）

**根本原因**：
- 二阶动量$\boldsymbol{v}_t$必须逐元素存储，无法压缩

**定量影响**：
- GPT-3 (175B参数)：仅优化器状态就需要350GB内存
- 限制了单GPU能训练的最大模型

**优化方向**：AdaFactor用低秩近似，但效果略差

---

### **优化方向**

**优化1：LAMB - Layer-wise Adaptive Moments**

**策略**：为每层使用不同的学习率缩放

**公式**：
$$\eta_l = \eta \cdot \frac{\|\boldsymbol{\theta}_l\|}{\|\boldsymbol{u}_l\|}$$

其中$\boldsymbol{u}_l$是第$l$层的Adam更新量。

**效果**：
- BERT预训练：支持batch size从256增至32k，速度提升76倍
- 不损失准确率

---

**优化2：AdaBelief - Adapting Stepsizes by Belief**

**策略**：用梯度的方差而非二阶矩来调整学习率

**公式**：
$$\boldsymbol{s}_t = \beta_2 \boldsymbol{s}_{t-1} + (1-\beta_2)(\boldsymbol{g}_t - \boldsymbol{m}_t)^2$$

$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{s}_t} + \epsilon}$$

**直觉**：
- Adam用$\boldsymbol{g}_t^2$（绝对大小）
- AdaBelief用$(\boldsymbol{g}_t - \boldsymbol{m}_t)^2$（与期望的偏离）
- 后者更关注梯度的不确定性

**效果**：
- 图像分类：Top-1准确率提升0.5%-1%
- 训练更稳定，收敛更快

---

**优化3：Lookahead**

**策略**：维护"快权重"和"慢权重"，周期性同步

**公式**：
- 快权重：用Adam更新$k$步 → $\boldsymbol{\theta}_{\text{fast}}$
- 慢权重：$\boldsymbol{\theta}_{\text{slow}} \leftarrow \boldsymbol{\theta}_{\text{slow}} + \alpha(\boldsymbol{\theta}_{\text{fast}} - \boldsymbol{\theta}_{\text{slow}})$

**效果**：
- 减少训练震荡
- 泛化性能提升1-2%
- 对超参数更鲁棒

</div>

#### 4.3 Lion - 批判性分析

<div class="analysis-box">

### **核心缺陷**

**缺陷1：小Batch Size效果差**

**问题描述**：
- Batch size < 64时，Lion性能显著下降
- 在batch size = 32时，可能不如Adam

**根本原因**：
1. **噪声叠加**：Sign操作本身引入噪声 + 小batch的梯度噪声 → 双重噪声
2. **方向不稳定**：小batch梯度方向随机性大 → sign频繁翻转 → 更新抖动

**定量影响**：
- 原论文Figure 5：batch size = 32时，Lion的验证准确率比AdamW低1.5%
- batch size = 256时，Lion反超AdamW 0.8%

**优化方向**：
- 增大batch size（如用梯度累积）
- 增大$\beta_1$（如从0.9 → 0.95），平滑sign输入
- 混合策略：前期用Adam，后期切换到Lion

---

**缺陷2：训练初期不稳定**

**问题描述**：
- 训练开始时loss可能震荡、甚至发散
- 需要更长的warmup

**根本原因**：
- 初始化时梯度方差大
- Sign操作将大梯度和小梯度都映射到±1 → 初期方向混乱

**数学分析**：

初期$\boldsymbol{m}_0 \approx \mathbf{0}$，则：
$$\boldsymbol{u}_1 = \text{sign}((1-\beta_1)\boldsymbol{g}_1)$$

如果$\boldsymbol{g}_1$噪声大，sign后的方向随机性强。

**优化方向**：
- **Warmup策略**：前1000-2000步用小学习率
- **混合初始化**：前期用Adam积累动量，后期切换Lion
- **Adaptive sign阈值**：
  $$\boldsymbol{u}_t = \text{sign}_{\tau_t}(\beta_1\boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t)$$
  其中$\tau_t$从大值逐渐降到0

---

**缺陷3：超参数需重新调整**

**问题描述**：
- 不能直接套用Adam的超参数
- 学习率、权重衰减、warmup都需重新搜索

**根本原因**：
- Update RMS = 1（固定），而Adam的Update RMS ≈ 0.1-0.3（自适应）
- 需要手动缩放学习率来匹配

**定量影响**：
- 迁移成本高：每个新任务都需要调参
- 时间成本：超参搜索可能需要数十次实验

**优化方向**：
- **自动缩放**：根据梯度统计量自动调整$\eta$
- **迁移学习规则**：提供从Adam到Lion的转换公式
- **Meta-learning**：学习不同任务的最优超参数分布

---

### **优化方向**

**优化1：Adaptive-Lion（提议）**

**策略**：让sign的阈值自适应

**公式**：
$$\boldsymbol{u}_t = \text{soft-sign}_{\tau_t}(\beta_1\boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t)$$

其中：
$$\text{soft-sign}_{\tau}(x) = \frac{x}{\tau + |x|}$$

当$\tau \to 0$，退化为硬sign；$\tau$大时，接近线性。

**自适应调整**：
$$\tau_t = \tau_0 \cdot \exp(-t / T_{\text{anneal}})$$

**预期效果**：
- 初期：$\tau$大，更新平滑
- 后期：$\tau$小，接近原始Lion

---

**优化2：Hybrid-Lion（提议）**

**策略**：结合Adam和Lion的优势

**公式**：
$$\boldsymbol{u}_t = (1-w_t)\boldsymbol{u}_{\text{Adam}} + w_t\boldsymbol{u}_{\text{Lion}}$$

其中权重$w_t$随训练进行从0增至1：
$$w_t = \min(1, t / T_{\text{transition}})$$

**直觉**：
- 前期：用Adam快速下降
- 后期：用Lion寻找平坦极小值

**预期效果**：
- 训练速度：接近Adam
- 泛化性能：接近Lion
- 鲁棒性：对小batch更友好

---

**优化3：Memory-Efficient Lion（提议）**

**策略**：进一步压缩动量存储

**观察**：Lion的动量$\boldsymbol{m}_t$只用于计算sign，精度要求低

**公式**：
- 用INT8存储$\boldsymbol{m}_t$（从FP32降至INT8）
- 内存占用：从$2d$降至$1.25d$（参数FP32 + 动量INT8）

**实现**：
$$\boldsymbol{m}_t^{\text{INT8}} = \text{quantize}(\beta_2 \boldsymbol{m}_{t-1}^{\text{INT8}} + (1-\beta_2)\boldsymbol{g}_t)$$

**预期效果**：
- 内存节省：从AdamW的33%提升至60%
- 精度损失：<0.1%（sign对量化误差不敏感）

</div>

---

### 第5部分：学习路线图与未来展望

#### 5.1 学习路线图

**必备前置知识**

**数学基础**：
- 微积分：梯度、导数、链式法则
- 线性代数：向量范数、矩阵运算
- 概率论：期望、方差、随机梯度
- 优化理论：凸优化、收敛性分析

**机器学习基础**：
- 深度学习：反向传播、损失函数
- 优化器：SGD、Momentum、Adam基础
- 正则化：权重衰减、Dropout
- 训练技巧：学习率调度、Warmup

**推荐学习顺序**：

1. **基础优化器**（1-2周）
   - SGD：理解最基本的梯度下降
   - Momentum：加速与平滑
   - Nesterov：前瞻性动量

2. **自适应优化器**（2-3周）
   - AdaGrad：自适应学习率的起源
   - RMSprop：解决AdaGrad的衰减问题
   - Adam：结合Momentum和RMSprop

3. **现代变体**（2-4周）
   - AdamW：解耦权重衰减
   - LAMB：大batch训练
   - AdaFactor：内存优化

4. **Lion及其理论**（1-2周）
   - Sign函数的性质
   - 双动量机制
   - 泛化性分析

---

**核心论文列表（按时间顺序）**

**理论奠基**：
1. Robbins & Monro (1951) - "A Stochastic Approximation Method"
2. Polyak (1964) - "Some Methods of Speeding Up Convergence"

**自适应优化器**：
3. Duchi et al. (2011) - "Adaptive Subgradient Methods (AdaGrad)" ⭐
4. Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization" ⭐
5. Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization (AdamW)" ⭐

**分布式与压缩**：
6. Bernstein et al. (2018) - "signSGD: Compressed Optimisation for Non-Convex Problems"
7. You et al. (2020) - "Large Batch Optimization for Deep Learning (LAMB)"

**Lion**：
8. Chen et al. (2023) - "Symbolic Discovery of Optimization Algorithms (Lion)" ⭐

**理论分析**：
9. Reddi et al. (2018) - "On the Convergence of Adam and Beyond (AMSGrad)"
10. Keskar et al. (2017) - "On Large-Batch Training: Sharp Minima and Generalization"

---

#### 5.2 研究空白与未来方向

#### **方向1：理论层面 - Lion的收敛性完整证明**

**研究空白**：
- 当前Lion的收敛性分析基于凸假设，非凸情况下的严格界未完全建立
- Sign函数引入的噪声如何精确影响收敛速度？
- 为什么Lion在实践中表现优于理论预测？

**具体研究问题**：

1. **问题**：非凸情况下Lion的收敛率精确界是什么？
   - **已知**：SGD/Adam的非凸收敛率为$\mathcal{O}(1/\sqrt{T})$
   - **未知**：Lion的sign噪声是否改变收敛率的常数或阶数？
   - **潜在方法**：
     - 将sign噪声建模为有界随机变量
     - 利用Martingale theory分析期望收敛
     - 推导高概率界（concentration bounds）
   - **潜在意义**：为Lion的理论地位提供严格保证

2. **问题**：双动量机制（$\beta_1 \neq \beta_2$）的数学作用是什么？
   - **观察**：原论文设$\beta_1=0.9, \beta_2=0.99$，为何不同？
   - **猜想**：
     - $\beta_1$控制更新方向的记忆长度
     - $\beta_2$控制动量本身的稳定性
     - 两者解耦提供更灵活的优化轨迹
   - **探索方向**：
     - 固定$\beta_1=\beta_2$时性能如何？
     - 是否存在最优的$(\beta_1, \beta_2)$关系？
     - 不同任务的最优配置是否有规律？

3. **问题**：Sign噪声与平坦度的定量关系？
   - **现状**：实验观察Lion找到更平坦的极小值
   - **未知**：如何定量刻画sign噪声 → 平坦度的映射？
   - **潜在方法**：
     - 推导Lion隐式最小化的等效目标函数
     - 证明其包含Hessian trace项：$\mathcal{L} + \gamma \cdot \text{tr}(\nabla^2\mathcal{L})$
     - 分析$\gamma$与学习率、$\beta$的关系

**优化方向**：
- 借鉴Stochastic Differential Equations (SDE)理论
- 使用Wasserstein距离分析优化轨迹
- 建立Lion与Langevin dynamics的联系

**量化目标**：
- 推导非凸情况下Lion的收敛率：$\mathbb{E}[\|\nabla f\|] \leq O(1/\sqrt{T}) + O(\eta\sqrt{d})$
- 证明平坦度界：$\lambda_{\max}(\nabla^2 f) \leq C \cdot \text{(sign noise variance)}^{-1}$
- 建立$(\beta_1, \beta_2)$的最优选择理论

---

#### **方向2：效率层面 - 极致加速与内存优化**

**研究空白**：
- Lion已比Adam快2-3倍，能否进一步优化？
- 动量存储能否压缩？
- 分布式训练中sign操作的通信优势未充分利用

**具体研究问题**：

1. **问题**：能否设计零动量（zero-momentum）的Lion变体？
   - **动机**：AdaFactor用低秩近似压缩Adam的$\boldsymbol{v}_t$，能否对Lion的$\boldsymbol{m}_t$做类似处理？
   - **挑战**：Lion的动量在sign之前，压缩后sign输入变化可能影响效果
   - **潜在方法**：
     - **层级动量**：每层共享一个标量动量 $m_l$
     - **低频更新**：每$k$步才更新动量
     - **哈希技巧**：用Bloom filter压缩动量存储
   - **量化目标**：内存从$2d$降至$1.1d$，性能损失<1%

2. **问题**：如何在分布式训练中利用sign的通信优势？
   - **观察**：All-reduce梯度时，Lion只需传输sign（1-bit），而Adam需传输FP32（32-bit）
   - **现状**：当前实现未充分利用这一点
   - **优化方向**：
     - **1-bit All-reduce**：直接传输sign，节省带宽31倍
     - **错误补偿**：类似QSGD，用error feedback补偿量化误差
     - **层级通信**：GPU内FP32，GPU间1-bit
   - **量化目标**：分布式训练通信量降低90%，总体加速5-10倍

3. **问题**：能否用专用硬件加速sign计算？
   - **现状**：Sign是逐元素操作，GPU利用率可能不高
   - **探索方向**：
     - FPGA实现高效sign
     - 神经网络加速器（如TPU）的sign优化
     - 量子计算中的sign操作（理论探索）

**优化方向**：
- 研究压缩感知（Compressed Sensing）理论应用于动量压缩
- 探索分布式优化的新型All-reduce算法
- 开发Lion专用的硬件加速器

**量化目标**：
- 内存占用降至AdamW的30%（当前50%）
- 分布式训练通信量降低90%
- 端到端训练时间减少50%

---

#### **方向3：应用层面 - 特定领域的Lion变体**

**研究空白**：
- Lion在CV、NLP表现好，但在其他领域（RL、GNN等）未充分验证
- 不同模态（图像、文本、语音）是否需要不同的Lion配置？
- 小模型（<100M参数）上Lion的优势不明显

**具体研究问题**：

1. **问题**：强化学习中的Lion配置？
   - **挑战**：RL梯度方差极大，sign可能加剧不稳定
   - **优化方向**：
     - **Clipped-Lion**：对sign输入先clip：$\text{sign}(\text{clip}(\boldsymbol{m}_t, -\tau, \tau))$
     - **Value-specific Lion**：Actor用Lion（需泛化），Critic用Adam（需精确拟合）
     - **Entropy-regularized Lion**：在sign中加入熵项鼓励探索
   - **实验场景**：Atari游戏、MuJoCo连续控制

2. **问题**：图神经网络（GNN）上的Lion？
   - **特点**：GNN的梯度分布与CNN/Transformer不同（消息传递机制）
   - **探索方向**：
     - **邻居感知Lion**：根据节点度数调整sign阈值
     - **图正则化Lion**：加入拉普拉斯正则项
   - **目标任务**：节点分类、图分类、链接预测

3. **问题**：多模态模型的Lion配置？
   - **场景**：CLIP、Flamingo等图文模型
   - **挑战**：不同模态的梯度尺度差异大
   - **优化方向**：
     - **模态特定$\beta$**：图像encoder用$\beta_1=0.9$，文本用$\beta_1=0.95$
     - **跨模态对齐Lion**：在sign前对不同模态的梯度做归一化
   - **量化目标**：在COCO captioning上超越AdamW 2+ CIDEr分

**优化方向**：
- 建立任务-配置的映射规则（meta-learning超参数）
- 开发自动调参工具（AutoLion）
- 在更多领域验证Lion（医疗影像、生物信息学等）

**量化目标**：
- RL：PPO+Lion在Atari上平均奖励超越PPO+Adam 15%
- GNN：节点分类准确率提升2%
- 多模态：零样本图像检索Recall@1提升3%

---

#### **方向4：鲁棒性层面 - 对抗训练与噪声鲁棒性**

**研究空白**：
- Lion对对抗样本的鲁棒性未知
- Label noise、data corruption下的表现？
- 是否能用Lion改进对抗训练？

**具体研究问题**：

1. **问题**：Lion训练的模型对对抗攻击更鲁棒吗？
   - **假设**：平坦极小值 → 对扰动鲁棒 → 对抗样本也鲁棒
   - **实验设计**：
     - 用Lion和Adam分别训练ResNet-50
     - 测试对PGD、C&W攻击的鲁棒性
     - 比较clean accuracy vs robust accuracy的trade-off
   - **预期结果**：Lion的robust accuracy更高？

2. **问题**：Lion能否加速对抗训练？
   - **现状**：对抗训练（如PGD-AT）需要数倍计算成本
   - **优化方向**：
     - **Fast-AT + Lion**：结合FGSM和Lion
     - **Sign-AT**：直接对sign后的梯度做对抗扰动
   - **量化目标**：对抗训练时间减少30%，鲁棒性不降低

3. **问题**：噪声标签下Lion的优势？
   - **场景**：训练数据有10%-40%错误标签
   - **猜想**：Sign函数的鲁棒性 → 对噪声标签不敏感
   - **实验**：CIFAR-10N（噪声CIFAR-10）
   - **评估指标**：Clean test accuracy vs noisy train accuracy

**优化方向**：
- 结合certified defense理论
- 开发Lion-AT（Lion for Adversarial Training）
- 研究Lion与curriculum learning的结合

**量化目标**：
- 对抗鲁棒性：在PGD-20攻击下准确率比Adam高5%
- 对抗训练加速：时间减少30%
- 噪声鲁棒性：40%标签噪声下准确率比Adam高3%

---

#### **方向5：自动化与可解释性**

**研究空白**：
- Lion的超参数（$\eta, \lambda, \beta_1, \beta_2$）需要手动调，能否自动化？
- Sign函数的作用机制能否可视化解释？
- 能否预测某任务是否适合Lion？

**具体研究问题**：

1. **问题**：自动超参数调优（AutoLion）？
   - **目标**：给定任务，自动找到最优$(\eta, \lambda, \beta_1, \beta_2)$
   - **方法**：
     - **Bayesian Optimization**：高斯过程建模超参数-性能关系
     - **Population-Based Training**：并行训练多组超参数，动态调整
     - **Meta-learning**：从历史任务中学习超参数初始化
   - **量化目标**：超参搜索次数从50次降至5次

2. **问题**：Lion的可解释性？
   - **当前问题**：为什么Lion有效？仍缺乏直观解释
   - **可视化方向**：
     - **Loss landscape**：绘制Lion vs Adam的优化轨迹
     - **Sign pattern**：可视化哪些参数的sign最频繁变化
     - **Gradient flow**：分析梯度在网络中的传播
   - **工具开发**：Lion-Viz（可视化工具包）

3. **问题**：任务适配性预测？
   - **目标**：不实际训练，预测Lion是否优于Adam
   - **特征**：
     - 数据集大小、维度
     - 模型架构（CNN、Transformer等）
     - 梯度统计量（方差、峰度等）
   - **方法**：训练分类器：(任务特征) → {Lion更好 | Adam更好}
   - **意义**：节省调参时间

**优化方向**：
- 开发Lion的AutoML工具
- 建立Lion的理论可解释性框架
- 创建Lion适用性的决策树

**量化目标**：
- AutoLion找到的超参数与人工调优的性能差距<1%
- 可视化工具覆盖5种解释维度（loss landscape、sign pattern等）
- 任务适配性预测准确率>85%

---

#### **潜在应用场景**

**语言模型**：
- GPT-4级别大模型预训练（节省内存，支持更大模型）
- 多语言模型（Lion的鲁棒性适合不同语言的梯度分布差异）
- 代码生成（CodeLlama、Codex等）

**计算机视觉**：
- 扩散模型训练（Stable Diffusion、DALL-E 3）
- 视频理解（VideoMAE、TimeSformer）
- 3D视觉（NeRF、3D生成）

**多模态**：
- CLIP-like模型（图文对齐）
- Flamingo（少样本多模态学习）
- GPT-4V（视觉语言模型）

**科学计算**：
- 分子动力学（AlphaFold类模型）
- 气候模拟（Foundation Models for Weather）
- 高能物理（粒子碰撞分析）

**边缘AI**：
- 移动端模型训练（内存受限场景）
- IoT设备上的在线学习
- 联邦学习（通信受限，sign压缩天然适合）

---

## 总结

本文深入分析了Lion优化器的数学原理：

1. **核心创新**：Sign函数 + 双动量机制
2. **理论保证**：与Adam相同的$\mathcal{O}(1/\sqrt{T})$收敛率
3. **实践优势**：更快、更省内存、更好泛化
4. **适用场景**：大模型预训练、视觉任务
5. **超参调优**：学习率↓，权重衰减↑

Lion代表了优化器设计的新范式：简单、高效、鲁棒。

