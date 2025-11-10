---
title: 对齐全量微调！这是我看过最精彩的LoRA改进（一）
slug: 对齐全量微调这是我看过最精彩的lora改进一
date: 2024-07-12
tags: 梯度, 优化器, 低秩, lora, 生成模型
status: pending
---

# 对齐全量微调！这是我看过最精彩的LoRA改进（一）

**原文链接**: [https://spaces.ac.cn/archives/10226](https://spaces.ac.cn/archives/10226)

**发布日期**: 

---

众所周知，LoRA是一种常见的参数高效的微调方法，我们在[《梯度视角下的LoRA：简介、分析、猜测及推广》](/archives/9590)做过简单介绍。LoRA利用低秩分解来降低微调参数量，节省微调显存，同时训练好的权重可以合并到原始权重上，推理架构不需要作出改变，是一种训练和推理都比较友好的微调方案。此外，我们在[《配置不同的学习率，LoRA还能再涨一点？》](/archives/10001)还讨论过LoRA的不对称性，指出给$A,B$设置不同的学习率能取得更好的效果，该结论被称为“LoRA+”。

为了进一步提升效果，研究人员还提出了不少其他LoRA变体，如[AdaLoRA](https://papers.cool/arxiv/2303.10512)、[rsLoRA](https://papers.cool/arxiv/2312.03732)、[DoRA](https://papers.cool/arxiv/2402.09353)、[PiSSA](https://papers.cool/arxiv/2404.02948)等，这些改动都有一定道理，但没有特别让人深刻的地方觉。然而，前两天的[《LoRA-GA: Low-Rank Adaptation with Gradient Approximation》](https://papers.cool/arxiv/2407.05000)，却让笔者眼前一亮，仅扫了摘要就有种必然有效的感觉，仔细阅读后更觉得它是至今最精彩的LoRA改进。

究竟怎么个精彩法？LoRA-GA的实际含金量如何？我们一起来学习一下。

## 基础回顾 #

首先我们再来温习一下LoRA。假设预训练参数为$W_0 \in \mathbb{R}^{n\times m}$，那么全量微调时的更新量自然也是一个$n\times m$矩阵，LoRA将更新量约束为低秩矩阵来降低训练时的参数量，即设$W=W_0 + AB$，其中$A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m}$以及$r\ll \min(n,m)$，用新的$W$替换模型原参数，并固定$W_0$不变，只训练$A,B$，如下图所示：  
$$\style{display: inline-block; width: 24ex; padding: 10ex 0; border: 1px solid #6C8EBF; background-color: #DAE8FC}{W_0\in\mathbb{R}^{n\times m}} \quad + \quad \style{display: inline-block; width: 8ex; padding: 10ex 0; border: 1px solid #D79B00; background-color: #FFE6CC}{A\in\mathbb{R}^{n\times r}}\quad\times\quad \style{display: inline-block; width: 24ex; padding: 3ex 0; border: 1px solid #D79B00; background-color: #FFE6CC}{B\in\mathbb{R}^{r\times m}}$$

为了使得LoRA的初始状态跟预训练模型一致，我们通常会将$A,B$之一全零初始化，这样可以得到$A_0 B_0=0$，那么初始的$W$就是$W_0$。但这并不是必须的，如果$A,B$都是非全零初始化，那么我们只需要将$W$设置为  
\begin{equation}W = (W_0 - A_0 B_0) + AB\end{equation}  
也就是说将固定不变的权重从$W_0$换为$W_0 - A_0 B_0$，同样可以满足 _**初始$W$等于$W_0$**_ 这一条件。

需要指出的是，LoRA往往只是显存不足的无奈之选，因为一般情况下全量微调的效果都会优于LoRA，所以如果算力足够并且要追求效果最佳时，请优先选择全量微调。这也是LoRA-GA的假设之一，因为它的改进方向就是向全量微调对齐。使用LoRA的另一个场景是有大量的微型定制化需求，我们要存下非常多的微调结果，此时使用LoRA能减少储存成本。

## 对齐全量 #

LoRA-GA提出了一个非常深刻的优化点：通过$W=(W_0 - A_0 B_0) + AB$我们可以保证$W$的初始值等于$W_0$，即初始状态的LoRA与全量微调是等价的，那么我们是否还可以调整$A_0$和$B_0$，使得LoRA和全量微调在后续训练中也尽可能近似？比如最简单地，让经过第一步优化后的$W_1$尽可能相等？

越仔细回味，我们会越发现这个优化点是如此“直击本质”——LoRA的目标不就是“以小搏大”，希望能接近全量微调的效果吗？既然如此，尽可能对齐全量微调的后续更新结果，不就是最正确的改进方向？从逼近的角度来看，“$W$的初始值等于$W_0$”相当于全量微调的零阶近似，保持后面的$W_1,W_2,\cdots$接近，则相当于是更高阶的近似，是合情合理的选择，所以笔者看完摘要后就有种“就是它了”的强烈感觉。

具体来说，假设我们的优化器是SGD，那么对于全量微调，我们有  
\begin{equation} W_1 = W_0 - \eta \frac{\partial \mathcal{L}}{\partial W_0}\end{equation}  
其中$\mathcal{L}$是损失函数，$\eta$是学习率。如果是LoRA的话，那么有  
\begin{equation}\begin{gathered}  
A_1 = A_0 - \eta \frac{\partial \mathcal{L}}{\partial A_0} = A_0 - \eta \frac{\partial \mathcal{L}}{\partial W_0} B_0^{\top},\quad B_1 = B_0 - \eta \frac{\partial \mathcal{L}}{\partial B_0} = B_0 - \eta A_0^{\top}\frac{\partial \mathcal{L}}{\partial W_0} \\\\[8pt]  
W_1 = W_0 - A_0 B_0 + A_1 B_1 \approx W_0 - \eta\left(A_0 A_0^{\top}\frac{\partial \mathcal{L}}{\partial W_0} + \frac{\partial \mathcal{L}}{\partial W_0}B_0^{\top} B_0\right)  
\end{gathered}\end{equation}  
最后的近似省略了$\eta$的二阶项。现在两个$W_1$具有相似的形式，为了让它们尽可能近似，我们可以考虑最小化  
\begin{equation}\mathop{\text{argmin}}_{A_0,B_0}\left\Vert A_0 A_0^{\top}\frac{\partial \mathcal{L}}{\partial W_0} + \frac{\partial \mathcal{L}}{\partial W_0}B_0^{\top} B_0 - \frac{\partial \mathcal{L}}{\partial W_0}\right\Vert_F^2 \label{eq:loss-0}\end{equation}  
其中$\Vert\cdot\Vert_F^2$是矩阵的[Frobenius范数](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)的平方，即矩阵每个元素的平方和。

## 求解过程 #

简单起见，我们记$G_0=\frac{\partial \mathcal{L}}{\partial W_0}$，那么目标$\eqref{eq:loss-0}$可以简写成  
\begin{equation}\mathop{\text{argmin}}_{A_0,B_0}\left\Vert A_0 A_0^{\top}G_0 + G_0 B_0^{\top} B_0 - G_0\right\Vert_F^2 \label{eq:loss-1}\end{equation}  
注意$A_0 A_0^{\top}G_0$、$G_0 B_0^{\top} B_0$的秩顶多为$r$，它们相加后的秩顶多为$2r$，我们假设$2r < \min(n,m)$，所以上述目标相当于寻找$G_0$的一个秩不超过$2r$的最优近似。

我们先考虑$G_0$是非负对角阵的情形，并且对角线元素已经按照从大到小的顺序排列。这个例子很简单，它的秩不超过$2r$的最优近似就是只保留对角线前$2r$个元素的新对角矩阵，这个结论叫做“[Eckart-Young-Mirsky定理](https://en.wikipedia.org/wiki/Low-rank_approximation)”，而能让$A_0 A_0^{\top}G_0 + G_0 B_0^{\top} B_0$只保留$G_0$的前$2r$个对角线元素的$A_0,B_0$可以是（分块矩阵）：  
\begin{equation}A_0 = (I_n)_{[:, :r]}, \quad B_0 = (I_m)_{[r:2r, :]}\end{equation}  
其中$I_n,I_m$分别是$n,m$阶单位阵，${}_{[:, :r]}$和${}_{[r:2r, :]}$就是像Python切片那样，取前$r$列和第$r+1\sim 2r$行。注意我们说的是“可以是”，也就是说解并不唯一，说白了就是要把$G_0$的前$2r$个对角线元素挑出来，$A_0 A_0^{\top}G_0$和 $G_0 B_0^{\top} B_0$各挑一半，至于怎么分配就无所谓了。上面给出的解，对应的是$A_0 A_0^{\top}G_0$挑出前$r$个，$G_0 B_0^{\top} B_0$挑出第$r+1\sim 2r$个。

当$G_0$不是对角阵时，我们将它SVD为$U\Sigma V$，其中$U\in\mathbb{R}^{n\times n},V\in\mathbb{R}^{m\times m}$为正交矩阵，$\Sigma\in\mathbb{R}^{n\times m}$为对角矩阵，对角线元素非负且从大到小排列。代入式$\eqref{eq:loss-1}$后得到  
\begin{equation}\begin{aligned}  
&\,\left\Vert A_0 A_0^{\top}G_0 + G_0 B_0^{\top} B_0 - G_0\right\Vert_F^2 \\\  
=&\, \left\Vert A_0 A_0^{\top}U\Sigma V + U\Sigma V B_0^{\top} B_0 - U\Sigma V\right\Vert_F^2 \\\  
=&\, \left\Vert U\left[(U^{\top}A_0) (U^{\top}A_0)^{\top}\Sigma + \Sigma (B_0 V^{\top})^{\top} (B_0 V^{\top}) - \Sigma \right]V\right\Vert_F^2 \\\  
=&\, \left\Vert (U^{\top}A_0) (U^{\top}A_0)^{\top}\Sigma + \Sigma (B_0 V^{\top})^{\top} (B_0 V^{\top}) - \Sigma\right\Vert_F^2 \\\  
\end{aligned}\end{equation}  
前两个等号都是简单的代换，第三个等号是因为正交变换不改变Frobenius范数（请读者自行证明一下）。经过这样的转换，我们发现逼近的对象重新转变为对角阵$\Sigma$，自变量则变成了$U^{\top}A_0$、$B_0 V^{\top}$，那么按照$G_0$是对角矩阵时所给出的解，我们得到  
\begin{equation}A_0 = U(I_n)_{[:, :r]} = U_{[:, :r]},\quad B_0 = (I_m)_{[r:2r, :]} V = V_{[r:2r, :]}\end{equation}

## 一般结果 #

现在我们就得到了LoRA的一种初始化方法：

> **LoRA-GA** 选取一批样本，计算初始梯度$G_0 = \nabla_{W_0}\mathcal{L}$，对梯度SVD为$G_0 = U\Sigma V$，取$U$的前$r$列初始化$A$，取$V$的第$r+1\sim 2r$行初始化$B$。

这样LoRA + SGD得到的$W_1$就跟全量微调的$W_1$尽可能相近了。此外，梯度最重要的是方向，其模长不大重要，所以初始化结果我们还可以乘以个scale，LoRA本身也可以乘以个scale，即$W = (W_0 - \lambda A_0 B_0) + \lambda AB$，这些都是LoRA常见的超参数，这里就不展开讨论了。顺便提一下，形式上跟LoRA-GA比较相似的是[PiSSA](https://papers.cool/arxiv/2404.02948)，它是对$W_0$做SVD来初始化$A,B$，这在理论支持上就不如LoRA-GA了，是一个纯粹的经验选择。

当然，可能有读者会发现目前的推导都是基于SGD优化器的假设，那么对于我们更常用的Adam优化器，结论是否要做出改变呢？理论上是要的。我们在[《配置不同的学习率，LoRA还能再涨一点？》](/archives/10001)讨论过，对于Adam来说，第一步优化结果是$W_1 = W_0 - \eta\, \text{sign}(G_0)$而不是$W_1 = W_0 - \eta G_0$，这样重复前面的推导，我们可以得到优化目标为  
\begin{equation}\mathop{\text{argmin}}_{A_0,B_0}\left\Vert A_0 \text{sign}(A_0^{\top}G_0) + \text{sign}(G_0 B_0^{\top}) B_0 - \text{sign}(G_0)\right\Vert_F^2 \label{eq:loss-adam}\end{equation}  
由于符号函数$\text{sign}$的存在，我们没法求出它的解析解，所以针对Adam的理论分析就只能止步于此了。

在这个背景下，对于Adam优化器，我们有三个选择：

> 1、**信仰** ：直接引用SGD的结果，相信它也可以在Adam中发挥同样的效果；
> 
> 2、**硬刚** ：用优化器直接去最小化目标$\eqref{eq:loss-adam}$，由于目标比较简单，计算量尚能接受；
> 
> 3、**投机** ：直觉上将$G_0$换成$\text{sign}(G_0)$，然后代入SGD的结论，可能更贴合Adam。

看起来原论文选择的是第1个方案，论文的实验结果确实也支持这一选择。

## 实验效果 #

论文的实验结果还是比较惊艳的，尤其是在GLUE上取得了最接近全量微调的效果：  


[![LoRA-GA + T5-Base 在GLUE上的表现](/usr/uploads/2024/07/4024630098.png)](/usr/uploads/2024/07/4024630098.png "点击查看原图")

LoRA-GA + T5-Base 在GLUE上的表现

平均来说，训练数据量越少，相对提升的幅度越大，这表明LoRA-GA对齐全量微调的策略，不仅有助于提高最终效果，还能提高训练效率，即可以用更少的训练步数就能达到更优的效果。

在LLAMA2-7b上的表现也可圈可点：  


[![LoRA-GA + LLAMA2-7b 在几个Benchmark的表现](/usr/uploads/2024/07/1845520815.png)](/usr/uploads/2024/07/1845520815.png "点击查看原图")

LoRA-GA + LLAMA2-7b 在几个Benchmark的表现

注意使用LoRA的主要场景是显存不足，但LoRA的初始化需要求出所有训练参数的完整梯度，这可能会由于显存不足而无法实现。为此，原论文提出的技巧是我们可以一个个参数串行地求梯度，而不是同时求所有训练参数的梯度，这样就可以把单步计算的显存降下来。串行求梯度虽然会降低效率，但初始化本身是一次性工作，因此稍慢点也无妨。至于怎么实现这个操作，不同框架有不同方法，这里也不展开讨论了。

## 文章小结 #

本文介绍了LoRA的一个新改进LoRA-GA。虽然LoRA的各种变体并不鲜见，但LoRA-GA以非常直观的理论指导折服了笔者，其改进思路给人一种“确认过眼神，它就是对的论文”的感觉，再配上可圈可点的实验结果，整个过程如行云流水，让人赏心悦目。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10226>_

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

苏剑林. (Jul. 12, 2024). 《对齐全量微调！这是我看过最精彩的LoRA改进（一） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10226>

@online{kexuefm-10226,  
title={对齐全量微调！这是我看过最精彩的LoRA改进（一）},  
author={苏剑林},  
year={2024},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/10226}},  
} 


---

## 公式推导与注释

### 1. 全量微调的数学定义与基础理论

**推导1.1：全量微调的梯度下降更新**

对于预训练权重 $W_0 \in \mathbb{R}^{n\times m}$，全量微调在第 $t$ 步的更新可以写为：

$$W_{t+1} = W_t - \eta \nabla_W \mathcal{L}(W_t)$$

其中 $\mathcal{L}$ 是损失函数，$\eta$ 是学习率。展开到多步更新：

$$W_t = W_0 - \eta \sum_{i=0}^{t-1} \nabla_W \mathcal{L}(W_i)$$

这个累积梯度项 $\sum_{i=0}^{t-1} \nabla_W \mathcal{L}(W_i)$ 是一个 $n \times m$ 的满秩矩阵（在一般情况下）。

**注释**：全量微调的参数空间是 $\mathbb{R}^{nm}$，这意味着我们有 $nm$ 个自由度来优化损失函数。

**推导1.2：全量微调的参数量分析**

全量微调需要存储和更新的参数总数为：

$$\text{Params}_{\text{full}} = nm$$

对于优化器状态（如Adam需要存储一阶和二阶动量）：

$$\text{Memory}_{\text{full}} = nm \times (1 + 2) = 3nm$$

第一项是参数本身，后两项是Adam的 $m_t$ 和 $v_t$ 状态。

**注释**：对于一个7B参数的大模型，假设使用FP32，全量微调需要约 $7 \times 10^9 \times 3 \times 4 = 84$ GB 的显存，这对于单卡是难以承受的。

### 2. LoRA的数学框架与约束

**推导2.1：LoRA的低秩分解**

LoRA将权重更新限制为低秩形式：

$$W = W_0 + \Delta W, \quad \Delta W = AB$$

其中 $A \in \mathbb{R}^{n \times r}$，$B \in \mathbb{R}^{r \times m}$，$r \ll \min(n,m)$。

秩的约束：

$$\text{rank}(\Delta W) = \text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B)) \leq r$$

**注释**：这个秩约束是LoRA的核心，它将参数空间从 $\mathbb{R}^{nm}$ 压缩到一个 $r$ 维子空间。

**推导2.2：LoRA的参数量与压缩比**

LoRA需要存储和更新的参数总数为：

$$\text{Params}_{\text{LoRA}} = nr + rm = r(n+m)$$

压缩比为：

$$\text{Compression Ratio} = \frac{\text{Params}_{\text{full}}}{\text{Params}_{\text{LoRA}}} = \frac{nm}{r(n+m)}$$

当 $n = m$ 时：

$$\text{Compression Ratio} = \frac{n^2}{2rn} = \frac{n}{2r}$$

**注释**：对于 $n=4096, r=8$，压缩比为 $4096/(2 \times 8) = 256$，即参数量减少到原来的 1/256。

**推导2.3：LoRA的梯度计算**

对于损失函数 $\mathcal{L}$，LoRA的梯度通过链式法则计算：

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial W} \frac{\partial W}{\partial A} = \frac{\partial \mathcal{L}}{\partial W} \cdot B^{\top}$$

$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial A} \frac{\partial A}{\partial B} = A^{\top} \cdot \frac{\partial \mathcal{L}}{\partial W}$$

其中我们使用了矩阵微分的乘积法则。

**注释**：注意 $\frac{\partial \mathcal{L}}{\partial A}$ 的形状是 $n \times r$，$\frac{\partial \mathcal{L}}{\partial B}$ 的形状是 $r \times m$，这与 $A, B$ 的形状一致。

### 3. LoRA与全量微调的对齐理论

**推导3.1：第一步更新的对比**

对于全量微调，第一步更新后：

$$W_1^{\text{full}} = W_0 - \eta G_0$$

其中 $G_0 = \frac{\partial \mathcal{L}}{\partial W_0}$。

对于LoRA（假设 $A_0 B_0 = 0$，即初始化满足恒等性）：

$$A_1 = A_0 - \eta \frac{\partial \mathcal{L}}{\partial A_0} = A_0 - \eta G_0 B_0^{\top}$$

$$B_1 = B_0 - \eta \frac{\partial \mathcal{L}}{\partial B_0} = B_0 - \eta A_0^{\top} G_0$$

**注释**：这里我们使用了前面推导的梯度公式。

**推导3.2：LoRA第一步更新的展开**

LoRA更新后的权重为：

$$W_1^{\text{LoRA}} = W_0 + A_1 B_1$$

代入 $A_1, B_1$ 的表达式：

$$W_1^{\text{LoRA}} = W_0 + (A_0 - \eta G_0 B_0^{\top})(B_0 - \eta A_0^{\top} G_0)$$

展开：

$$W_1^{\text{LoRA}} = W_0 + A_0 B_0 - \eta A_0 A_0^{\top} G_0 - \eta G_0 B_0^{\top} B_0 + \eta^2 G_0 B_0^{\top} A_0^{\top} G_0$$

假设 $A_0 B_0 = 0$（初始化条件），并忽略 $\eta^2$ 的二阶小量：

$$W_1^{\text{LoRA}} \approx W_0 - \eta (A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0)$$

**注释**：忽略二阶项是合理的，因为学习率 $\eta$ 通常很小（如 $10^{-3}$ 或 $10^{-4}$），二阶项可以忽略不计。

**推导3.3：更新差异的量化**

全量微调和LoRA的第一步更新差异为：

$$\Delta W_1 = W_1^{\text{full}} - W_1^{\text{LoRA}} \approx -\eta G_0 + \eta (A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0)$$

$$= \eta [(A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0) - G_0]$$

$$= \eta [(A_0 A_0^{\top} - I_n) G_0 + G_0 (B_0^{\top} B_0 - I_m)]$$

其中 $I_n, I_m$ 分别是 $n \times n$ 和 $m \times m$ 的单位矩阵。

**注释**：如果 $A_0 A_0^{\top} = I_n$ 且 $B_0^{\top} B_0 = I_m$，则 $\Delta W_1 = 0$，LoRA完全对齐全量微调。但这在 $r < \min(n,m)$ 时是不可能的，因为 $A_0 A_0^{\top}$ 的秩最多为 $r$。

### 4. 梯度空间的投影分析

**推导4.1：投影算子的定义**

定义左投影算子：

$$P_L = A_0 A_0^{\top}$$

这是一个将向量投影到 $A_0$ 列空间的投影矩阵。

验证投影性质：

$$(P_L)^2 = (A_0 A_0^{\top})(A_0 A_0^{\top}) = A_0 (A_0^{\top} A_0) A_0^{\top} = A_0 A_0^{\top} = P_L$$

**注释**：$P_L$ 是幂等的，这是投影算子的特征性质。但注意 $P_L$ 不一定是对称的，除非 $A_0^{\top} A_0 = I_r$。

**推导4.2：右投影算子**

类似地，定义右投影算子：

$$P_R = B_0^{\top} B_0$$

验证：

$$(P_R)^2 = (B_0^{\top} B_0)(B_0^{\top} B_0) = B_0^{\top} (B_0 B_0^{\top}) B_0 = B_0^{\top} B_0 = P_R$$

**注释**：$P_R$ 是一个 $m \times m$ 的投影矩阵。

**推导4.3：LoRA的梯度更新作为投影**

LoRA的第一步更新可以写为：

$$W_1^{\text{LoRA}} - W_0 \approx -\eta (P_L G_0 + G_0 P_R)$$

这表明LoRA的更新是梯度 $G_0$ 的左投影和右投影的组合。

全量微调的更新是：

$$W_1^{\text{full}} - W_0 = -\eta G_0$$

**注释**：LoRA只能在 $A_0$ 的列空间和 $B_0$ 的行空间所张成的子空间中更新权重，这是一个严格的限制。

**推导4.4：投影误差的范数界**

定义投影误差：

$$E = (P_L G_0 + G_0 P_R) - G_0 = (P_L - I_n) G_0 + G_0 (P_R - I_m)$$

其Frobenius范数的上界：

$$\|E\|_F \leq \|(P_L - I_n) G_0\|_F + \|G_0 (P_R - I_m)\|_F$$

使用次可乘性：

$$\|E\|_F \leq \|P_L - I_n\|_F \|G_0\|_F + \|G_0\|_F \|P_R - I_m\|_F$$

$$= \|G_0\|_F (\|P_L - I_n\|_F + \|P_R - I_m\|_F)$$

**注释**：投影误差与梯度的范数成正比，也与投影算子距离单位矩阵的距离成正比。

### 5. 秩不足的补偿机制

**推导5.1：秩约束的影响**

$P_L G_0$ 的秩满足：

$$\text{rank}(P_L G_0) = \text{rank}(A_0 A_0^{\top} G_0) \leq \min(\text{rank}(A_0 A_0^{\top}), \text{rank}(G_0)) \leq r$$

类似地：

$$\text{rank}(G_0 P_R) \leq r$$

因此：

$$\text{rank}(P_L G_0 + G_0 P_R) \leq \text{rank}(P_L G_0) + \text{rank}(G_0 P_R) \leq 2r$$

**注释**：LoRA的梯度更新最多是秩 $2r$ 的矩阵，而全量微调的梯度 $G_0$ 可能是满秩的（秩为 $\min(n,m)$）。

**推导5.2：最优低秩逼近（Eckart-Young-Mirsky定理）**

对于任意矩阵 $M \in \mathbb{R}^{n \times m}$，其秩不超过 $k$ 的最优逼近（在Frobenius范数意义下）为：

$$M_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{\top}$$

其中 $M = \sum_{i=1}^{\text{rank}(M)} \sigma_i u_i v_i^{\top}$ 是SVD分解，$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ 是奇异值。

逼近误差为：

$$\min_{\text{rank}(X) \leq k} \|M - X\|_F = \sqrt{\sum_{i=k+1}^{\text{rank}(M)} \sigma_i^2}$$

**注释**：这个定理告诉我们，低秩逼近的最优解是通过SVD得到的，且误差由被截断的奇异值决定。

**推导5.3：LoRA-GA的优化目标推导**

我们希望找到 $A_0, B_0$ 使得：

$$\min_{A_0, B_0} \|P_L G_0 + G_0 P_R - G_0\|_F^2$$

展开：

$$= \min_{A_0, B_0} \|(A_0 A_0^{\top} - I_n) G_0 + G_0 (B_0^{\top} B_0 - I_m)\|_F^2$$

$$= \min_{A_0, B_0} \|A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0 - G_0\|_F^2$$

**注释**：这个优化问题的目标是找到最佳的投影方向，使得投影后的梯度尽可能接近原始梯度。

### 6. 对齐损失的数学推导

**推导6.1：对齐损失的展开**

定义对齐损失：

$$\mathcal{L}_{\text{align}} = \|A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0 - G_0\|_F^2$$

展开Frobenius范数的平方：

$$\mathcal{L}_{\text{align}} = \text{Tr}[(A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0 - G_0)^{\top} (A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0 - G_0)]$$

其中 $\text{Tr}(\cdot)$ 是矩阵的迹。

**注释**：Frobenius范数的平方等于矩阵所有元素平方和，也等于 $AA^{\top}$ 的迹。

**推导6.2：对齐损失的三项分解**

展开上式：

$$\mathcal{L}_{\text{align}} = \underbrace{\text{Tr}[G_0^{\top} A_0 A_0^{\top} A_0 A_0^{\top} G_0]}_{\text{Term 1}} + \underbrace{\text{Tr}[B_0^{\top} B_0 G_0^{\top} G_0 B_0^{\top} B_0]}_{\text{Term 2}}$$

$$+ \underbrace{2\text{Tr}[G_0^{\top} A_0 A_0^{\top} G_0 B_0^{\top} B_0]}_{\text{Cross term}} - \underbrace{2\text{Tr}[G_0^{\top} A_0 A_0^{\top} G_0]}_{\text{Term 3}} - \underbrace{2\text{Tr}[G_0^{\top} G_0 B_0^{\top} B_0]}_{\text{Term 4}} + \underbrace{\text{Tr}[G_0^{\top} G_0]}_{\text{Constant}}$$

**注释**：最后一项是常数，对优化无影响。前面几项涉及 $A_0, B_0$ 的二次型和四次型。

**推导6.3：SVD分解下的简化**

设 $G_0 = U \Sigma V^{\top}$ 是SVD分解，其中：
- $U \in \mathbb{R}^{n \times n}$ 是左奇异向量矩阵（正交）
- $\Sigma \in \mathbb{R}^{n \times m}$ 是对角奇异值矩阵
- $V \in \mathbb{R}^{m \times m}$ 是右奇异向量矩阵（正交）

代入对齐损失：

$$\mathcal{L}_{\text{align}} = \|A_0 A_0^{\top} U \Sigma V^{\top} + U \Sigma V^{\top} B_0^{\top} B_0 - U \Sigma V^{\top}\|_F^2$$

利用正交变换不改变Frobenius范数：

$$= \|U^{\top} A_0 A_0^{\top} U \Sigma + \Sigma V B_0^{\top} B_0 V^{\top} - \Sigma\|_F^2$$

令 $\tilde{A}_0 = U^{\top} A_0$，$\tilde{B}_0 = B_0 V^{\top}$：

$$= \|\tilde{A}_0 \tilde{A}_0^{\top} \Sigma + \Sigma \tilde{B}_0^{\top} \tilde{B}_0 - \Sigma\|_F^2$$

**注释**：通过SVD变换，问题被转换到奇异值空间，大大简化了分析。

**推导6.4：对角矩阵情况的解析解**

当 $\Sigma$ 是方阵且对角时，设 $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{\min(n,m)})$，其中 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$。

目标变为逼近对角矩阵 $\Sigma$ 的每个对角元素：

$$\min \sum_{i,j} [(\tilde{A}_0 \tilde{A}_0^{\top})_{ij} \sigma_j + \sigma_i (\tilde{B}_0^{\top} \tilde{B}_0)_{ij} - \sigma_i \delta_{ij}]^2$$

最优策略是让 $\tilde{A}_0 \tilde{A}_0^{\top}$ 和 $\tilde{B}_0^{\top} \tilde{B}_0$ 在对角线上分配最大的奇异值。

**注释**：由于秩的限制，我们只能保留前 $2r$ 个最大的奇异值。

**推导6.5：LoRA-GA的最优解**

根据上述分析，最优的 $A_0, B_0$ 为：

$$A_0 = U_{[:, :r]} = [u_1, u_2, \ldots, u_r]$$

$$B_0 = V_{[r:2r, :]}^{\top} = [v_{r+1}, v_{r+2}, \ldots, v_{2r}]^{\top}$$

其中 $u_i$ 是 $G_0$ 的第 $i$ 个左奇异向量，$v_i$ 是第 $i$ 个右奇异向量。

**注释**：这个初始化策略保留了梯度 $G_0$ 的前 $2r$ 个主要方向，最大化了与全量微调的对齐程度。

### 7. 理论保证与收敛性分析

**推导7.1：最优对齐误差的界**

使用LoRA-GA的初始化，对齐误差为：

$$\mathcal{L}_{\text{align}}^* = \|G_0 - (P_L^* G_0 + G_0 P_R^*)\|_F^2$$

根据Eckart-Young定理：

$$\mathcal{L}_{\text{align}}^* = \sum_{i=2r+1}^{\min(n,m)} \sigma_i^2$$

相对误差：

$$\frac{\mathcal{L}_{\text{align}}^*}{\|G_0\|_F^2} = \frac{\sum_{i=2r+1}^{\min(n,m)} \sigma_i^2}{\sum_{i=1}^{\min(n,m)} \sigma_i^2}$$

**注释**：如果梯度 $G_0$ 的奇异值快速衰减（这在深度学习中很常见），则相对误差会很小，LoRA-GA能很好地对齐全量微调。

**推导7.2：收敛速度的比较**

对于凸优化问题，梯度下降的收敛速度与条件数有关。设损失函数的Hessian矩阵的最大和最小特征值为 $L$ 和 $\mu$，则：

全量微调的收敛率：

$$\|W_t - W^*\|^2 \leq (1 - \frac{\mu}{L})^t \|W_0 - W^*\|^2$$

LoRA的收敛率受限于投影误差：

$$\|W_t^{\text{LoRA}} - W^*\| \leq \|(I - P_{\text{LoRA}}) (W^* - W_0)\| + \mathcal{O}((1 - \frac{\mu}{L})^t)$$

其中 $P_{\text{LoRA}}$ 是LoRA所能表示的子空间的投影。

**注释**：即使LoRA收敛到其子空间中的最优点，仍可能与全量微调的全局最优有一个固定的gap，这个gap由投影误差决定。

**推导7.3：多步更新的误差累积**

考虑 $T$ 步更新后的累积误差：

$$W_T^{\text{full}} - W_T^{\text{LoRA}} = \sum_{t=0}^{T-1} \eta [(P_L^{(t)} G_t + G_t P_R^{(t)}) - G_t]$$

其中 $P_L^{(t)}, P_R^{(t)}$ 是第 $t$ 步的投影算子（因为 $A_t, B_t$ 在变化）。

Frobenius范数：

$$\|W_T^{\text{full}} - W_T^{\text{LoRA}}\|_F \leq \eta \sum_{t=0}^{T-1} \|(P_L^{(t)} G_t + G_t P_R^{(t)}) - G_t\|_F$$

**注释**：如果每步的投影误差都很小（通过良好的初始化实现），则累积误差也会被控制在合理范围内。

**推导7.4：泛化误差分析**

根据统计学习理论，模型的泛化误差可以分解为：

$$\mathbb{E}[\mathcal{L}(W)] - \min_W \mathcal{L}(W) = \underbrace{[\mathbb{E}[\mathcal{L}(\hat{W})] - \min_W \mathcal{L}(W)]}_{\text{Estimation error}} + \underbrace{[\mathcal{L}(\hat{W}) - \mathbb{E}[\mathcal{L}(\hat{W})]]}_{\text{Optimization error}}$$

LoRA由于参数量较少，estimation error可能更小（过拟合风险低），但optimization error可能更大（受秩约束）。

**注释**：在数据量有限的情况下，LoRA的正则化效应可能带来更好的泛化，这也是为什么有时LoRA在小数据集上表现不错的原因。

### 8. Adam优化器的扩展分析

**推导8.1：Adam的一阶更新近似**

Adam优化器的更新规则为：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

其中 $\hat{m}_t = m_t/(1-\beta_1^t)$，$\hat{v}_t = v_t/(1-\beta_2^t)$ 是偏差校正后的动量。

在第一步（$t=1$），假设 $m_0 = v_0 = 0$：

$$m_1 = (1-\beta_1) g_0$$
$$v_1 = (1-\beta_2) g_0^2$$
$$\hat{m}_1 = g_0, \quad \hat{v}_1 = g_0^2$$

因此第一步更新近似为：

$$\theta_1 \approx \theta_0 - \eta \frac{g_0}{|g_0| + \epsilon} \approx \theta_0 - \eta \cdot \text{sign}(g_0)$$

**注释**：当梯度的绝对值远大于 $\epsilon$ 时，Adam的第一步更新主要由梯度的符号决定。

**推导8.2：LoRA-GA在Adam下的对齐目标**

对于Adam，LoRA的第一步更新为：

$$W_1^{\text{LoRA}} \approx W_0 - \eta [A_0 \text{sign}(A_0^{\top} G_0) + \text{sign}(G_0 B_0^{\top}) B_0]$$

全量微调为：

$$W_1^{\text{full}} \approx W_0 - \eta \text{sign}(G_0)$$

对齐目标变为：

$$\min_{A_0, B_0} \|A_0 \text{sign}(A_0^{\top} G_0) + \text{sign}(G_0 B_0^{\top}) B_0 - \text{sign}(G_0)\|_F^2$$

**注释**：由于符号函数的非线性，这个优化问题没有解析解，但我们可以相信SGD的解（通过SVD得到）在Adam中也是一个好的近似。

**推导8.3：符号梯度的统计性质**

假设梯度 $G_0$ 的元素服从某个分布，$\text{sign}(G_0)$ 保留了梯度的主要方向信息。

对于高斯分布的梯度，$\text{sign}(G_0)$ 与 $G_0$ 的相关性为：

$$\text{Corr}(\text{sign}(G_{ij}), G_{ij}) = \sqrt{\frac{2}{\pi}} \approx 0.798$$

这说明符号函数保留了约80%的方向信息。

**注释**：因此，使用SGD推导的初始化策略在Adam下仍然是合理的。

### 9. 实验结果的理论解释

**推导9.1：GLUE数据集上的性能提升**

GLUE数据集包含多个小规模任务，数据量有限。设训练样本数为 $N$，LoRA的有效参数数为 $d = r(n+m)$。

根据学习理论，泛化误差的界为：

$$\mathcal{L}_{\text{test}} - \mathcal{L}_{\text{train}} \leq \mathcal{O}\left(\sqrt{\frac{d \log(N/d)}{N}}\right)$$

LoRA-GA通过更好的初始化，减少了训练所需的步数 $T$，从而降低了优化误差：

$$\text{Optimization error} \propto (1 - \frac{\mu}{L})^T$$

更少的 $T$ 意味着更快达到相同的训练损失。

**注释**：在小数据集上，快速收敛尤为重要，因为过度训练容易导致过拟合。

**推导9.2：相对提升幅度与数据量的关系**

定义相对提升为：

$$\text{Relative Gain} = \frac{\text{Acc}_{\text{LoRA-GA}} - \text{Acc}_{\text{LoRA}}}{\text{Acc}_{\text{Full}} - \text{Acc}_{\text{LoRA}}}$$

理论上，这个提升应该与 LoRA-GA 对齐误差的减少成正比：

$$\text{Relative Gain} \propto \frac{\mathcal{L}_{\text{align}}^{\text{random}} - \mathcal{L}_{\text{align}}^{\text{GA}}}{\mathcal{L}_{\text{align}}^{\text{random}}}$$

在数据量较小时，梯度估计的方差较大，但主要方向仍然稳定，因此SVD初始化的优势更明显。

**注释**：实验显示，在训练数据量越少的任务上，LoRA-GA的相对提升越大，这与理论预测一致。

**推导9.3：训练步数的减少**

设LoRA达到目标损失 $\mathcal{L}^*$ 需要 $T_{\text{LoRA}}$ 步，LoRA-GA需要 $T_{\text{GA}}$ 步。

由于更好的初始化，初始损失更低：

$$\mathcal{L}_0^{\text{GA}} < \mathcal{L}_0^{\text{LoRA}}$$

收敛速度相同的情况下：

$$\mathcal{L}_0^{\text{GA}} (1 - \alpha)^{T_{\text{GA}}} = \mathcal{L}^* = \mathcal{L}_0^{\text{LoRA}} (1 - \alpha)^{T_{\text{LoRA}}}$$

解得：

$$T_{\text{GA}} = T_{\text{LoRA}} - \frac{\log(\mathcal{L}_0^{\text{GA}} / \mathcal{L}_0^{\text{LoRA}})}{\log(1 - \alpha)}$$

**注释**：更好的初始化可以显著减少训练时间，这在大模型微调中非常有价值。

### 10. 与其他LoRA变体的对比

**推导10.1：PiSSA的初始化策略**

PiSSA对预训练权重 $W_0$ 进行SVD：

$$W_0 = U \Sigma V^{\top}$$

然后设置：

$$A_0 = U_{[:, :r]} \Sigma_{[:r, :r]}^{1/2}, \quad B_0 = \Sigma_{[:r, :r]}^{1/2} V_{[:, :r]}^{\top}$$

使得 $A_0 B_0 \approx W_0$（秩-$r$ 近似）。

PiSSA的目标是初始化时就接近预训练权重，但这与梯度方向无关。

**注释**：LoRA-GA关注梯度方向，PiSSA关注权重本身，两者的出发点不同。

**推导10.2：AdaLoRA的动态秩调整**

AdaLoRA引入重要性分数来动态调整每个参数的秩：

$$S_i = \frac{|\sigma_i|}{\\sum_j |\sigma_j|}$$

其中 $\sigma_i$ 是某种奇异值的估计。

AdaLoRA会剪枝掉重要性低的维度，但这需要额外的计算开销。

**注释**：LoRA-GA是一次性初始化，AdaLoRA需要持续监控和调整，计算成本更高。

**推导10.3：DoRA的方向与幅度分解**

DoRA将权重更新分解为方向和幅度：

$$W = \|W\| \cdot \frac{W}{\|W\|}$$

分别优化方向和幅度分量。

这与LoRA-GA的投影视角有本质不同，DoRA更关注权重的几何结构。

**注释**：DoRA的优势在于更细粒度的控制，但理论基础不如LoRA-GA清晰。

### 11. 计算复杂度分析

**推导11.1：SVD的计算复杂度**

对 $G_0 \in \mathbb{R}^{n \times m}$ 进行完整SVD的复杂度为：

$$\mathcal{O}(\min(nm^2, n^2 m))$$

但LoRA-GA只需要前 $2r$ 个奇异值和奇异向量，可以使用截断SVD，复杂度降为：

$$\mathcal{O}(nmr)$$

**注释**：对于 $n=m=4096, r=8$，截断SVD需要约 $4096^2 \times 8 \approx 1.3 \times 10^8$ 次浮点运算，在现代GPU上可以在秒级完成。

**推导11.2：梯度计算的显存需求**

计算初始梯度 $G_0$ 需要一次完整的前向和反向传播，显存需求为：

$$\text{Memory} = nm \times \text{sizeof}(\text{float})$$

对于 FP32 和 $n=m=4096$：

$$\text{Memory} = 4096^2 \times 4 \text{ bytes} = 64 \text{ MB}$$

这对于现代GPU是完全可接受的。

**注释**：论文提到可以串行计算每层的梯度，进一步降低峰值显存。

**推导11.3：LoRA-GA的一次性成本摊销**

设LoRA-GA的初始化需要额外时间 $T_{\text{init}}$，而每个epoch的训练时间为 $T_{\text{epoch}}$。

如果LoRA-GA能减少 $k$ 个epoch达到相同效果，则总时间节省为：

$$\Delta T = k \cdot T_{\text{epoch}} - T_{\text{init}}$$

只要 $k > T_{\text{init}} / T_{\text{epoch}}$，LoRA-GA就是值得的。

实验中通常 $T_{\text{init}} \ll T_{\text{epoch}}$，因此即使只减少1个epoch，也是划算的。

**注释**：初始化是一次性成本，可以通过减少训练步数来快速回本。

### 12. 扩展到多层和多模块

**推导12.1：多层LoRA的独立初始化**

对于Transformer中的多个LoRA层，每层独立计算梯度并进行SVD初始化：

$$A_0^{(l)} = U^{(l)}_{[:, :r]}, \quad B_0^{(l)} = V^{(l)}_{[r:2r, :]}$$

其中 $G_0^{(l)} = U^{(l)} \Sigma^{(l)} (V^{(l)})^{\top}$ 是第 $l$ 层的梯度SVD。

**注释**：不同层的梯度分布可能差异很大，因此独立初始化是必要的。

**推导12.2：注意力层的特殊处理**

对于多头注意力，每个头可能需要不同的秩：

$$r_h \propto \sigma_1^{(h)} / \sum_h \sigma_1^{(h)}$$

这样可以根据每个头的重要性动态分配秩。

**注释**：这是一个潜在的改进方向，但会增加实现复杂度。

### 13. 理论局限性与未来方向

**推导13.1：非凸优化的挑战**

深度学习的损失函数通常是非凸的，梯度 $G_t$ 在训练过程中会不断变化。

LoRA-GA只对齐第一步，后续步骤的梯度方向可能偏离初始方向：

$$\langle G_t, G_0 \rangle / (\|G_t\| \|G_0\|) \to \text{decrease as } t \to \infty$$

**注释**：这是LoRA-GA的一个局限，可能需要周期性地重新初始化或调整。

**推导13.2：自适应秩调整的可能性**

可以设计一个指标来衡量当前 $A_t, B_t$ 的对齐程度：

$$\text{Alignment}(t) = \frac{\|A_t A_t^{\top} G_t + G_t B_t^{\top} B_t\|_F}{\|G_t\|_F}$$

当这个比值下降时，可以考虑增加秩或重新初始化。

**注释**：这需要在线监控梯度，可能带来额外开销。

**推导13.3：理论与实践的gap**

理论分析基于一阶近似和SGD优化器，但实践中使用Adam和复杂的学习率调度。

理论预测的最优秩可能与实践中的最优秩有偏差：

$$r_{\text{theory}} = \arg\min_r [\text{Approximation error} + \lambda \cdot r]$$

$$r_{\text{practice}} = \arg\min_r [\text{Validation loss}]$$

两者不一定相等，需要实验验证。

**注释**：理论提供方向性指导，但超参数仍需根据具体任务调整。

### 总结

本节详细推导了LoRA-GA的数学基础，包括：

1. **全量微调与LoRA的差异**：量化了秩约束带来的表达能力损失
2. **对齐理论**：通过投影分析揭示了LoRA与全量微调的本质差距
3. **SVD初始化**：证明了基于梯度SVD的初始化是最优的低秩逼近
4. **收敛性保证**：分析了LoRA-GA如何加速收敛并减少优化误差
5. **实验验证**：从理论角度解释了实验中观察到的性能提升

LoRA-GA是一个理论与实践完美结合的范例，其核心思想简单而深刻：**通过对齐梯度方向来对齐优化轨迹**。这为LoRA的进一步改进提供了坚实的理论基础。
