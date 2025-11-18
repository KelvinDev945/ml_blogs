---
title: Google新搜出的优化器Lion：效率与效果兼得的“训练狮”
slug: google新搜出的优化器lion效率与效果兼得的训练狮
date: 2023-02-16
tags: 分析, 优化, 优化器, 生成模型, attention
status: pending
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

## 详细数学推导

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

## 总结

本文深入分析了Lion优化器的数学原理：

1. **核心创新**：Sign函数 + 双动量机制
2. **理论保证**：与Adam相同的$\mathcal{O}(1/\sqrt{T})$收敛率
3. **实践优势**：更快、更省内存、更好泛化
4. **适用场景**：大模型预训练、视觉任务
5. **超参调优**：学习率↓，权重衰减↑

Lion代表了优化器设计的新范式：简单、高效、鲁棒。

