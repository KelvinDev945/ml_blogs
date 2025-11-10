---
title: 梯度视角下的LoRA：简介、分析、猜测及推广
slug: 梯度视角下的lora简介分析猜测及推广
date: 2023-04-17
tags: 梯度, 优化器, 低秩, lora, 生成模型
status: pending
---

# 梯度视角下的LoRA：简介、分析、猜测及推广

**原文链接**: [https://spaces.ac.cn/archives/9590](https://spaces.ac.cn/archives/9590)

**发布日期**: 

---

随着ChatGPT及其平替的火热，各种参数高效（Parameter-Efficient）的微调方法也“水涨船高”，其中最流行的方案之一就是本文的主角**LoRA** 了，它出自论文[《LoRA: Low-Rank Adaptation of Large Language Models》](https://papers.cool/arxiv/2106.09685)。LoRA方法上比较简单直接，而且也有不少现成实现，不管是理解还是使用都很容易上手，所以本身也没太多值得细写的地方了。

然而，直接实现LoRA需要修改网络结构，这略微麻烦了些，同时LoRA给笔者的感觉是很像之前的优化器[AdaFactor](/archives/7302)，所以笔者的问题是：**能否从优化器角度来分析和实现LoRA呢？** 本文就围绕此主题展开讨论。

## 方法简介 #

以往的一些结果（比如[《Exploring Aniversal Intrinsic Task Subspace via Prompt Tuning》](https://papers.cool/arxiv/2110.07867)）显示，尽管预训练模型的参数量很大，但每个下游任务对应的本征维度（Intrinsic Dimension）并不大，换句话说，理论上我们可以微调非常小的参数量，就能在下游任务取得不错的效果。

LoRA借鉴了上述结果，提出对于预训练的参数矩阵$W_0\in\mathbb{R}^{n\times m}$，我们不去直接微调$W_0$，而是对增量做低秩分解假设：  
\begin{equation}W = W_0 + A B,\qquad A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m}\end{equation}  
其中$A,B$之一用全零初始化，$W_0$固定不变，优化器只优化$A,B$。由于本征维度很小的结论，所以$r$我们可以取得很小，常见的是$r=8$，极端情况下我们甚至可以取$1$。所以说，LoRA是一种参数高效的微调方法，至少被优化的参数量大大降低了。

用MathJax直接画了个示意图：  
$$\style{display: inline-block; width: 24ex; padding: 10ex 0; border: 1px solid #6C8EBF; background-color: #DAE8FC}{W_0\in\mathbb{R}^{n\times m}} \quad + \quad \style{display: inline-block; width: 8ex; padding: 10ex 0; border: 1px solid #D79B00; background-color: #FFE6CC}{A\in\mathbb{R}^{n\times r}}\quad\times\quad \style{display: inline-block; width: 24ex; padding: 3ex 0; border: 1px solid #D79B00; background-color: #FFE6CC}{B\in\mathbb{R}^{r\times m}}$$

## 梯度分析 #

正如[《Ladder Side-Tuning：预训练模型的“过墙梯”》](/archives/9138)所提到的，很多参数高效的微调实际上只是降低了显存需求，并没有降低计算量。那么LoRA是否例外呢？它在显存和计算量方面的效率如何呢？下面我们来分析一下。

首先，我们知道训练模型所消耗的显存来源包括**模型参数** 、**模型梯度** 、**模型激活值** 、**优化器状态** 四部份，LoRA通过低秩分解降低了模型参数量，那么梯度和优化器状态也会随之降低，因此节省的显存是很明显的。那它能否节省计算量呢？

这取决于LoRA的实现方式，不同的实现方式计算梯度的复杂度不一样。LoRA的两种等效实现如下：  
\begin{align}Y =&\, XW = X(W_0 + AB) \label{eq:lora-1}\\\\[5pt]  
Y =&\, XW_0 + XAB = XW_0 + ZB \label{eq:lora-2}\end{align}  
其中$X\in\mathbb{R}^{b\times n}$是模型输入，$Z=XA\in\mathbb{R}^{b\times r}$是中间输出。针对实现$\eqref{eq:lora-1}$，我们有  
\begin{equation}\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial W} B^{\top} = \left(X^{\top}\frac{\partial \mathcal{L}}{\partial Y}\right) B^{\top},\quad \frac{\partial \mathcal{L}}{\partial B} = A^{\top}\frac{\partial \mathcal{L}}{\partial W} = A^{\top}\left(X^{\top}\frac{\partial \mathcal{L}}{\partial Y}\right)\label{eq:grad-1}\end{equation}  
$\mathcal{L}$是损失函数。很明显，这种实现导致的后果是需要算完整梯度$\frac{\partial \mathcal{L}}{\partial W}\in\mathbb{R}^{n\times m}$，然后才能算$A,B$的梯度，这意味着它比不LoRA还慢，也费显存。对于实现$\eqref{eq:lora-2}$，我们则有  
\begin{equation}\frac{\partial \mathcal{L}}{\partial A} = X^{\top}\frac{\partial \mathcal{L}}{\partial Z} = X^{\top}\left(\frac{\partial \mathcal{L}}{\partial Y} B^{\top}\right),\quad \frac{\partial \mathcal{L}}{\partial B} = Z^{\top}\frac{\partial \mathcal{L}}{\partial Y} = (XA)^{\top}\frac{\partial \mathcal{L}}{\partial Y}\label{eq:grad-2}\end{equation}  
此时的$Z,\frac{\partial \mathcal{L}}{\partial Z}\in\mathbb{R}^{b\times r}$，相比完整的梯度显然省了不少，计算复杂度也明显降低。所以，LoRA想要节省显存和计算最大化，关键是按照$\eqref{eq:lora-2}$而不是$\eqref{eq:lora-1}$来实现。

（注：关于矩阵计算梯度，我们可以根据链式法则和输出形状来“凑”，比如$\frac{\partial \mathcal{L}}{\partial A}$，根据链式法则我们知道它必然是$\frac{\partial \mathcal{L}}{\partial W}$和$B$以某种方式相乘，我们约定$\frac{\partial \mathcal{L}}{\partial A}$的形状跟$A$一致，即$n\times r$，想要用$\frac{\partial \mathcal{L}}{\partial W}$和$B$凑出一个$n\times r$的结果来，那就只有$\frac{\partial \mathcal{L}}{\partial W} B^{\top}$了。）

## 其他原因 #

除了低秩分解带来的好处外，如下几点也是LoRA能节省显存和提速的原因：

> 1、只更新了部分参数：比如LoRA原论文就选择只更新Self Attention的参数，实际使用时我们还可以选择只更新部分层的参数；
> 
> 2、减少了通信时间：由于更新的参数量变少了，所以（尤其是多卡训练时）要传输的数据量也变少了，从而减少了传输时间；
> 
> 3、采用了各种低精度加速技术，如FP16、FP8或者INT8量化等。

当然，这三部分原因确实能加快训练速度，但它们并不是LoRA所独有的，事实上几乎都有参数高效方法都具有这些特点。LoRA的突出优点是它的低秩分解很直观，在不少场景下跟全量微调的效果一致，以及在预测阶段可以直接把$W_0,A,B$合并成单个矩阵从而不增加推理成本。

## 优化视角 #

梯度$\eqref{eq:grad-1}$还告诉了我们如何从优化器角度来实现LoRA。优化器可以直接获取到全量梯度$\frac{\partial \mathcal{L}}{\partial W}$，然后我们只需要按照公式$\eqref{eq:grad-1}$对梯度进行投影，就得到$A,B$的梯度，接着就可以按照常规的优化器实现$A,B$的更新了。

假如优化器是SGD，那么就是  
\begin{equation}\begin{aligned}  
A_{t+1} =&\, A_t - \eta\frac{\partial \mathcal{L}}{\partial W_t} B_t^{\top},\quad B_{t+1} = B_t - \eta A_t^{\top}\frac{\partial \mathcal{L}}{\partial W_t}\\\\[5pt]  
W_{t+1} =&\, W_0 + A_{t+1} B_{t+1} = W_t + (A_{t+1} B_{t+1} - A_t B_t)  
\end{aligned}\end{equation}  
如果是Adam之类的带滑动变量的优化器，则只需要滑动投影后的梯度，因此是降低了优化器的参数量，节省了一定的显存。模型越大，这部分参数所占的显存比例也就越大。

LoRA约定$A$或$B$之一使用全零初始化，这是为了保证初始状态模型跟预训练一致，但同时也带来了不对称问题（一个全零，一个非全零）。事实上，$A,B$都使用非全零初始化也是可以的，只需要事先将预训练权重减去$A_0 B_0$就行了，或者等价地说，将$W$参数化为  
\begin{equation}W = W_0 - A_0 B_0 + A B\end{equation}  
这样同时保持了初始状态一致，同时允许$A,B$都用非全零初始化，增强了对称性。

## 随机投影 #

如果我们将SGD场景下的更新量$A_{t+1} B_{t+1} - A_t B_t$展开，结果将是  
\begin{equation}- \eta\left(\frac{\partial \mathcal{L}}{\partial W_t} B_t^{\top} B_t + A_t A_t^{\top}\frac{\partial \mathcal{L}}{\partial W_t}\right) + \eta^2 \frac{\partial \mathcal{L}}{\partial W_t} B_t^{\top} A_t^{\top}\frac{\partial \mathcal{L}}{\partial W_t}\end{equation}  
假设$\eta^2$项是可以忽略的高阶项，那么就剩下  
\begin{equation}- \eta\left(\frac{\partial \mathcal{L}}{\partial W_t} B_t^{\top} B_t + A_t A_t^{\top}\frac{\partial \mathcal{L}}{\partial W_t}\right)\end{equation}  
从这个角度来看，相比全量微调的SGD，LoRA就是用括号中的结果替代了全量的梯度$\frac{\partial \mathcal{L}}{\partial W_t}$。

简单起见，接下来我们只关心$r=1$的情形，留意到在上式中，$t$时刻的投影向量$A_t,B_t$是依赖于$t$的，如果我们将它们换成不依赖于$t$的随机向量（每步训练都重新随机生成），那么会发生什么呢？我们考虑$u,v\sim\mathcal{N}(0,1)$，其中$u\in\mathbb{R}^{m\times 1}, v\in\mathbb{R}^{1\times n}$，那么更新量就变为  
\begin{equation}- \eta\left(\frac{\partial \mathcal{L}}{\partial W_t} v^{\top} v + u u^{\top}\frac{\partial \mathcal{L}}{\partial W_t}\right)\end{equation}  
可以证明的是  
\begin{equation}\mathbb{E}_{u\sim \mathcal{N}(0,1)}[u u^{\top}] = I_{n\times n},\quad \mathbb{E}_{v\sim \mathcal{N}(0,1)}[v^{\top} v] = I_{m\times m}\end{equation}  
这里的$I_{n\times n},I_{m\times m}$分别指$n\times n,m\times m$的单位矩阵。因此，跟“[零阶梯度](/archives/7737#%E9%9B%B6%E9%98%B6%E6%A2%AF%E5%BA%A6)”类似，在平均意义下，这种每步都重新初始化的LoRA事实上等价于满秩的SGD。然而，真要按照这个方式实现的话，其速度甚至可能比满秩的SGD都要慢，所以它的目的不是提速，而是希望能缓解灾难遗忘问题——通过对单个（batch）样本使用低秩矩阵（而不是满秩）更新量的方式，减少对整个模型权重的影响。当然，这只是猜测，实际效果如何，笔者还没有实验过。

## 一个变体 #

同样还是先只考虑$r=1$的情形，LoRA相当于假设了$\Delta w_{i,j} = u_i v_j$，我们能不能做其他低秩分解假设呢？比如$\Delta w_{i,j} = u_i + v_j$？写成矩阵形式就是  
\begin{equation}W = W_0 + A \mathbb{1}_{1\times m} + \mathbb{1}_{n\times 1} B,\qquad A\in\mathbb{R}^{n\times 1},B\in\mathbb{R}^{1\times m}\end{equation}  
其中$\mathbb{1}_{1\times m},\mathbb{1}_{n\times 1}$分别指$1\times m,n\times 1$的全1矩阵。容易求出它的梯度是：  
\begin{equation}\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial W} \mathbb{1}_{m\times 1},\quad \frac{\partial \mathcal{L}}{\partial B} = \mathbb{1}_{1\times n}\frac{\partial \mathcal{L}}{\partial W}\end{equation}  
其实就是原本梯度的行求和与列求和。相比原版LoRA，这个加性分解有两个优点：1、加比乘计算量更低，梯度形式也更简单；2、$AB$的秩一定是1，但是$A \mathbb{1}_{1\times m} + \mathbb{1}_{n\times 1} B$的秩可能是2，如果秩代表了模型能力的话，那也就是说同样的参数量，加性的表达能力可能还更强。至于具体效果如何，后面笔者用到LoRA的时候，再做对比实验吧。

那么，加性分解能不能推广到$r > 1$的情形呢？自然是可以的，但稍微有些技巧。这里约定$m,n$都能被$r$整除，那么我们只需要将参数化方式改为  
\begin{equation}W = W_0 + A I_{r(1\times m/r)} + I_{r(n/r\times 1)} B,\qquad A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m}\end{equation}  
这里的$I_{r(1\times m/r)}$、$I_{r(n/r\times 1)}$分别指$1\times m/r$、$n/r\times 1$的分块矩阵，每一块则是$r\times r$的单位阵。这个形式说白了，就是分别将$A$、$B$看成是$n/r\times 1$、$1\times m/r$的分块矩阵，然后套用$r=1$的思路来操作。

## 文章小结 #

本文介绍了从梯度角度来理解LoRA，除了基本的介绍外，还包含了笔者的一些猜测和推广，供读者参考。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9590>_

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

苏剑林. (Apr. 17, 2023). 《梯度视角下的LoRA：简介、分析、猜测及推广 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9590>

@online{kexuefm-9590,  
title={梯度视角下的LoRA：简介、分析、猜测及推广},  
author={苏剑林},  
year={2023},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9590}},  
} 


---

## 公式推导与注释

### 1. LoRA的低秩分解数学定义

**定义1.1（预训练权重矩阵）**：设预训练模型中某一层的权重矩阵为 $W_0 \in \mathbb{R}^{n \times m}$，其中 $n$ 是输出维度，$m$ 是输入维度。

**定义1.2（LoRA的低秩分解）**：LoRA假设权重的更新可以用低秩矩阵的乘积表示：

$$
W = W_0 + \Delta W = W_0 + AB
$$

其中：
- $A \in \mathbb{R}^{n \times r}$ 是降维矩阵
- $B \in \mathbb{R}^{r \times m}$ 是升维矩阵
- $r \ll \min(n, m)$ 是秩（rank），通常 $r \in \{1, 2, 4, 8, 16\}$

**性质1.1（秩约束）**：更新矩阵 $\Delta W = AB$ 的秩满足：

$$
\text{rank}(\Delta W) = \text{rank}(AB) \leq \min(r, n, m) = r
$$

因为 $A$ 的列空间维度最多为 $r$，$B$ 的行空间维度最多为 $r$。

**定理1.1（参数数量对比）**：

全量微调的参数数量：$nm$

LoRA的参数数量：$nr + rm = r(n+m)$

参数压缩比：
$$
\rho = \frac{r(n+m)}{nm} = \frac{r}{nm}(n+m)
$$

**示例**：设 $n = m = 4096, r = 8$，则：

$$
\rho = \frac{8 \times (4096 + 4096)}{4096 \times 4096} = \frac{8 \times 8192}{16777216} \approx 0.0039
$$

即LoRA只需要约0.39%的参数。

### 2. LoRA的初始化策略

**定义2.1（零初始化策略）**：为了保证初始状态与预训练模型一致，LoRA采用：

$$
A \sim \mathcal{N}(0, \sigma^2), \quad B = 0
$$

或者：

$$
A = 0, \quad B \sim \mathcal{N}(0, \sigma^2)
$$

**性质2.1（初始等价性）**：当 $B = 0$ 或 $A = 0$ 时：

$$
W|_{t=0} = W_0 + AB = W_0 + A \cdot 0 = W_0
$$

因此初始模型输出与预训练模型完全一致。

**定理2.1（Kaiming初始化）**：对于非零的矩阵（如 $A$），通常使用Kaiming初始化：

$$
A_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

其中 $n_{in}$ 是输入维度。这保证了前向传播时激活值的方差稳定。

### 3. LoRA的前向传播

**定义3.1（标准前向传播）**：给定输入 $x \in \mathbb{R}^{b \times m}$（$b$ 是batch size），输出为：

$$
y = xW = x(W_0 + AB)
$$

**定义3.2（分解形式的前向传播）**：为了计算效率，实际实现为：

$$
y = xW_0 + xAB = xW_0 + (xA)B
$$

令 $h = xA \in \mathbb{R}^{b \times r}$，则：

$$
y = xW_0 + hB
$$

**定理3.1（计算复杂度对比）**：

直接计算 $x(W_0 + AB)$ 的复杂度：
1. 计算 $AB$：$O(nrm)$
2. 计算 $x(W_0 + AB)$：$O(bnm)$
3. 总计：$O(nrm + bnm)$

分解计算 $xW_0 + (xA)B$ 的复杂度：
1. 计算 $xW_0$：$O(bnm)$
2. 计算 $xA$：$O(bnr)$
3. 计算 $(xA)B$：$O(brm)$
4. 总计：$O(bnm + bnr + brm) = O(bnm + br(n+m))$

当 $b$ 较小且 $r \ll \min(n, m)$ 时，分解形式更高效。

### 4. LoRA的梯度推导

**定理4.1（损失对A的梯度）**：设损失函数为 $\mathcal{L}$，则：

$$
\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial h} \frac{\partial h}{\partial A} = \left(\frac{\partial \mathcal{L}}{\partial y} B^T\right)^T x = x^T \frac{\partial \mathcal{L}}{\partial y} B^T
$$

**详细推导**：

从 $y = xW_0 + hB$ 和 $h = xA$，我们有：

$$
\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial h} \frac{\partial h}{\partial A}
$$

首先计算 $\frac{\partial \mathcal{L}}{\partial h}$：

$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial h} = \frac{\partial \mathcal{L}}{\partial y} B^T
$$

因为 $y = xW_0 + hB$，对 $h$ 求导得 $B^T$（注意矩阵维度）。

然后计算 $\frac{\partial h}{\partial A}$：

由 $h = xA$，我们有：

$$
\frac{\partial \mathcal{L}}{\partial A} = x^T \frac{\partial \mathcal{L}}{\partial h} = x^T \left(\frac{\partial \mathcal{L}}{\partial y} B^T\right)
$$

整理得：

$$
\frac{\partial \mathcal{L}}{\partial A} = x^T \frac{\partial \mathcal{L}}{\partial y} B^T \in \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{n \times r} \quad \text{（转置后）}
$$

实际上：

$$
\frac{\partial \mathcal{L}}{\partial A} = \left(x^T \frac{\partial \mathcal{L}}{\partial y}\right) B^T \in \mathbb{R}^{n \times r}
$$

**定理4.2（损失对B的梯度）**：

$$
\frac{\partial \mathcal{L}}{\partial B} = h^T \frac{\partial \mathcal{L}}{\partial y} = (xA)^T \frac{\partial \mathcal{L}}{\partial y} = A^T x^T \frac{\partial \mathcal{L}}{\partial y}
$$

**详细推导**：

从 $y = xW_0 + hB$，对 $B$ 求导：

$$
\frac{\partial y}{\partial B} = h^T \text{（根据矩阵微积分）}
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial B} = h^T \frac{\partial \mathcal{L}}{\partial y} = (xA)^T \frac{\partial \mathcal{L}}{\partial y} \in \mathbb{R}^{r \times m}
$$

### 5. 梯度的低秩结构分析

**定义5.1（完整权重的梯度）**：如果我们对整个 $W = W_0 + AB$ 进行全量微调，梯度为：

$$
\frac{\partial \mathcal{L}}{\partial W} = x^T \frac{\partial \mathcal{L}}{\partial y} \in \mathbb{R}^{m \times n} \rightarrow \mathbb{R}^{n \times m}
$$

（转置后维度正确）

**定理5.1（LoRA梯度的低秩分解）**：LoRA实际更新的等效梯度可以表示为：

$$
\Delta W_{LoRA} = AB
$$

其梯度更新等价于：

$$
\frac{\partial \mathcal{L}}{\partial W}_{LoRA} = \frac{\partial \mathcal{L}}{\partial A} \frac{\partial A}{\partial W} + \frac{\partial \mathcal{L}}{\partial B} \frac{\partial B}{\partial W}
$$

但这里的关键洞察是：LoRA隐式地假设 $\frac{\partial \mathcal{L}}{\partial W}$ 具有低秩结构。

**定理5.2（梯度的秩上界）**：LoRA的有效梯度 $G_{eff}$ 可以表示为：

$$
G_{eff} = \frac{\partial \mathcal{L}}{\partial A} B + A \frac{\partial \mathcal{L}}{\partial B}
$$

其秩满足：

$$
\text{rank}(G_{eff}) \leq 2r
$$

**证明**：

第一项：$\frac{\partial \mathcal{L}}{\partial A} B$ 的秩 $\leq r$（因为 $B$ 的秩 $\leq r$）

第二项：$A \frac{\partial \mathcal{L}}{\partial B}$ 的秩 $\leq r$（因为 $A$ 的秩 $\leq r$）

根据秩的次可加性：

$$
\text{rank}(G_{eff}) \leq \text{rank}\left(\frac{\partial \mathcal{L}}{\partial A} B\right) + \text{rank}\left(A \frac{\partial \mathcal{L}}{\partial B}\right) \leq 2r
$$

这说明LoRA的梯度更新被限制在一个低秩子空间中。$\square$

### 6. 参数效率的理论证明

**定义6.1（参数效率）**：参数效率定义为达到相同性能所需的参数量之比。

**定理6.1（显存占用对比）**：

全量微调的显存占用（主要部分）：
- 模型参数：$nm$ 个float
- 梯度：$nm$ 个float
- 优化器状态（Adam）：$2nm$ 个float（一阶动量 + 二阶动量）
- 总计：$4nm$ 个float

LoRA微调的显存占用：
- 预训练参数（冻结）：$nm$ 个float（不需要梯度）
- LoRA参数 $A, B$：$r(n+m)$ 个float
- 梯度：$r(n+m)$ 个float
- 优化器状态：$2r(n+m)$ 个float
- 总计：$nm + 3r(n+m)$ 个float

**显存节省比**：

$$
\text{Saving} = 1 - \frac{nm + 3r(n+m)}{4nm} = 1 - \frac{1}{4} - \frac{3r(n+m)}{4nm}
$$

当 $r \ll \frac{nm}{n+m}$ 时，节省接近75%。

**示例**：$n = m = 4096, r = 8$

$$
\text{Saving} = 1 - \frac{4096^2 + 3 \times 8 \times 8192}{4 \times 4096^2} \approx 1 - 0.25 - 0.003 = 74.7\%
$$

### 7. 梯度更新的等价性分析

**定理7.1（单步更新的形式）**：使用SGD优化器，LoRA的一步更新为：

$$
A_{t+1} = A_t - \eta \frac{\partial \mathcal{L}}{\partial A_t}
$$

$$
B_{t+1} = B_t - \eta \frac{\partial \mathcal{L}}{\partial B_t}
$$

等价的权重更新为：

$$
W_{t+1} = W_0 + A_{t+1}B_{t+1}
$$

**定理7.2（更新的非线性）**：LoRA的更新是非线性的：

$$
W_{t+1} - W_t = A_{t+1}B_{t+1} - A_t B_t
$$

展开：

$$
= (A_t - \eta \nabla_A)(B_t - \eta \nabla_B) - A_t B_t
$$

$$
= A_t B_t - \eta A_t \nabla_B - \eta \nabla_A B_t + \eta^2 \nabla_A \nabla_B - A_t B_t
$$

$$
= -\eta(A_t \nabla_B + \nabla_A B_t) + \eta^2 \nabla_A \nabla_B
$$

其中 $\nabla_A = \frac{\partial \mathcal{L}}{\partial A_t}, \nabla_B = \frac{\partial \mathcal{L}}{\partial B_t}$。

**近似7.1（忽略二阶项）**：当学习率 $\eta$ 较小时，忽略 $\eta^2$ 项：

$$
W_{t+1} - W_t \approx -\eta(A_t \nabla_B + \nabla_A B_t)
$$

这是LoRA的一阶近似更新。

### 8. 与全参数微调的关系

**定义8.1（全参数微调）**：全参数微调直接更新所有权重：

$$
W_{t+1} = W_t - \eta \frac{\partial \mathcal{L}}{\partial W_t}
$$

**定理8.1（LoRA作为约束优化）**：LoRA可以看作是在低秩约束下的优化：

$$
\min_{W} \mathcal{L}(W) \quad \text{s.t.} \quad \text{rank}(W - W_0) \leq r
$$

**证明**：LoRA的参数化 $W = W_0 + AB$ 自动满足：

$$
\text{rank}(W - W_0) = \text{rank}(AB) \leq r
$$

因此LoRA在参数空间的一个低秩流形上进行优化。$\square$

**定理8.2（表达能力的限制）**：LoRA无法表示任意的权重更新。

设 $\Delta W$ 是任意的 $n \times m$ 矩阵，$\text{rank}(\Delta W) = k$。

- 如果 $k \leq r$，则存在 $A, B$ 使得 $\Delta W = AB$
- 如果 $k > r$，则不存在秩为 $r$ 的分解

**推论8.1**：当真实的最优更新 $\Delta W^*$ 的秩大于 $r$ 时，LoRA只能找到次优解。

然而，实验表明在许多任务中，$r$ 很小（如8）就足够了，暗示梯度确实具有低秩结构。

### 9. 不同秩r的表达能力

**定义9.1（秩r的自由度）**：秩为 $r$ 的 $n \times m$ 矩阵的自由度为：

$$
\text{DoF}(r) = r(n + m - r)
$$

**推导**：秩$r$矩阵可以表示为 $UV^T$，其中 $U \in \mathbb{R}^{n \times r}, V \in \mathbb{R}^{m \times r}$。

但这个表示不唯一，对于任意可逆矩阵 $R \in \mathbb{R}^{r \times r}$：

$$
UV^T = (UR)(R^{-1}V^T)
$$

因此实际自由度为：

$$
nr + mr - r^2 = r(n + m - r)
$$

**定理9.1（秩与表达能力的关系）**：

- $r = 1$：自由度 $= n + m - 1$，可以表示所有秩1矩阵
- $r = \min(n,m)$：自由度 $= nm$（满秩），可以表示任意矩阵
- $r \in (1, \min(n,m))$：部分表达能力

**定理9.2（秩增长与性能）**：实验观察表明，性能随 $r$ 增长的曲线通常为：

$$
\text{Performance}(r) = P_{\max} - (P_{\max} - P_0) e^{-\lambda r}
$$

其中 $P_{\max}$ 是全量微调性能，$P_0$ 是零样本性能，$\lambda$ 是任务相关常数。

这表明：
- 小 $r$ 时，性能快速提升
- 大 $r$ 时，边际收益递减
- 存在一个"充分秩" $r^*$，超过它性能不再显著提升

### 10. LoRA的梯度低秩假设的理论依据

**假设10.1（本征维度假设）**：许多下游任务的本征维度远低于参数空间维度。

形式化地，存在一个低维子空间 $\mathcal{S} \subset \mathbb{R}^{nm}$，维度为 $d \ll nm$，使得在 $\mathcal{S}$ 内优化就能达到接近全局最优的性能。

**定理10.1（梯度的主成分）**：如果梯度 $\nabla_W \mathcal{L}$ 可以被低秩分解良好近似，即：

$$
\nabla_W \mathcal{L} \approx \sum_{i=1}^r \sigma_i u_i v_i^T
$$

其中 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r$ 是主要奇异值，且 $\sum_{i>r} \sigma_i^2 \ll \sum_{i=1}^r \sigma_i^2$，则LoRA能够捕获梯度的主要方向。

**定理10.2（预训练的作用）**：预训练模型已经学习了数据的通用表示，因此在下游任务的微调中，只需要在低维子空间中调整即可。

**证明思路**：设预训练模型的权重为 $W_0$，最优微调权重为 $W^*$。如果 $W_0$ 已经接近最优，则：

$$
W^* = W_0 + \Delta W, \quad \|\Delta W\| \ll \|W_0\|
$$

此时 $\Delta W$ 可以看作是在 $W_0$ 处的局部调整，很可能具有低秩结构。$\square$

### 11. LoRA的优化轨迹分析

**定义11.1（优化轨迹）**：LoRA的优化轨迹在参数空间中为：

$$
W(t) = W_0 + A(t)B(t)
$$

其中 $A(t), B(t)$ 随训练时间 $t$ 演化。

**定理11.1（轨迹的流形约束）**：LoRA的优化被约束在一个 $r(n+m)$ 维的流形上：

$$
\mathcal{M}_r = \{W_0 + AB : A \in \mathbb{R}^{n \times r}, B \in \mathbb{R}^{r \times m}\}
$$

**性质11.1（流形的维度）**：$\mathcal{M}_r$ 是 $\mathbb{R}^{n \times m}$ 中的一个子流形，维度为 $r(n+m) - r^2$（考虑旋转不变性）。

**定理11.2（投影梯度下降）**：LoRA可以看作是在流形 $\mathcal{M}_r$ 上的黎曼梯度下降。

在每一步，全量梯度 $\nabla_W \mathcal{L}$ 被投影到 $\mathcal{M}_r$ 的切空间上：

$$
\nabla_{LoRA} = \text{Proj}_{\mathcal{M}_r}(\nabla_W \mathcal{L})
$$

然后沿着投影梯度更新参数。

### 12. LoRA与其他低秩方法的对比

**方法12.1（Adapters）**：Adapters在每层插入小型MLP瓶颈：

$$
h' = h + f(h), \quad f(h) = W_{up} \sigma(W_{down} h)
$$

其中 $W_{down} \in \mathbb{R}^{r \times d}, W_{up} \in \mathbb{R}^{d \times r}$。

**对比**：
- LoRA：直接修改权重矩阵，推理无额外开销
- Adapters：引入额外的前向计算，推理有开销

**方法12.2（Prompt Tuning）**：只优化输入的prompt向量：

$$
x' = [p_1, p_2, \ldots, p_k, x_1, x_2, \ldots, x_n]
$$

其中 $p_i$ 是可学习的prompt向量。

**对比**：
- LoRA：参数量 $O(r \cdot \text{model size})$
- Prompt Tuning：参数量 $O(k \cdot d)$，与层数无关

**方法12.3（BitFit）**：只微调bias参数。

**对比**：
- LoRA：修改权重矩阵，表达能力强
- BitFit：只修改bias，参数量最少但表达能力弱

### 13. LoRA的缩放因子分析

**定义13.1（LoRA的缩放）**：实际实现中，LoRA的输出会乘以一个缩放因子：

$$
y = xW_0 + \frac{\alpha}{r} xAB
$$

其中 $\alpha$ 是可调节的超参数（通常设为 $r$）。

**定理13.1（缩放因子的作用）**：缩放因子 $\frac{\alpha}{r}$ 有两个作用：

1. **归一化**：当改变 $r$ 时，保持 $\Delta W = AB$ 的尺度大致恒定
2. **控制影响**：调节LoRA相对于预训练权重的影响程度

**定理13.2（最优缩放）**：最优缩放因子取决于任务的难度和预训练模型的质量。

如果预训练模型已经很接近最优，应使用较小的 $\alpha$；反之，应使用较大的 $\alpha$。

**经验规律**：通常设 $\alpha = r$，这样 $\frac{\alpha}{r} = 1$，使得LoRA的初始影响与矩阵维度无关。

### 14. LoRA的训练动力学

**定理14.1（初始阶段的快速适应）**：在训练初期，LoRA通常比全量微调收敛更快。

**直觉**：低秩约束起到了正则化的作用，减少了过拟合的风险，允许使用更大的学习率。

**定理14.2（渐近性能）**：在足够的训练后，LoRA的性能会接近但略低于全量微调（如果秩不够大）。

形式化地，设 $L_{LoRA}^*$ 和 $L_{full}^*$ 分别是LoRA和全量微调的最优损失，则：

$$
L_{LoRA}^* \geq L_{full}^*
$$

等号成立当且仅当最优更新 $\Delta W^*$ 的秩 $\leq r$。

**定理14.3（正则化效应）**：LoRA的低秩约束起到了隐式正则化的作用：

$$
\min_{W} \mathcal{L}(W) + \lambda \text{rank}(W - W_0)
$$

这有助于防止过拟合，特别是在小样本场景下。

### 15. LoRA的组合与分解

**定理15.1（LoRA模块的可加性）**：多个LoRA模块可以通过简单相加组合：

设有两个任务对应的LoRA：$\Delta W_1 = A_1 B_1$ 和 $\Delta W_2 = A_2 B_2$，则：

$$
W_{combined} = W_0 + \alpha_1 A_1 B_1 + \alpha_2 A_2 B_2
$$

其中 $\alpha_1, \alpha_2$ 是混合权重。

**应用**：这允许多任务学习和任务插值。

**定理15.2（LoRA的合并）**：推理时，LoRA可以合并到原始权重中：

$$
W_{merged} = W_0 + AB
$$

这样推理时只需要执行 $y = xW_{merged}$，没有额外开销。

**定理15.3（动态秩调整）**：训练过程中，可以动态调整秩：

从较小的 $r_1$ 开始，逐渐增加到 $r_2$。实现方法是填充零：

$$
A' = [A, 0_{n \times (r_2 - r_1)}], \quad B' = \begin{bmatrix} B \\ 0_{(r_2-r_1) \times m} \end{bmatrix}
$$

### 16. LoRA的梯度流分析

**定理16.1（梯度的反向传播）**：在反向传播中，梯度流经两条路径：

1. 通过预训练权重 $W_0$（但 $W_0$ 被冻结，不更新）
2. 通过LoRA路径 $A$ 和 $B$

**详细分析**：

设输出层损失为 $\mathcal{L}$，则：

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} (W_0 + AB)^T
$$

这个梯度包含了 $W_0$ 和 $AB$ 的信息，因此即使 $W_0$ 被冻结，它仍然参与梯度的反向传播。

**定理16.2（梯度消失/爆炸的缓解）**：LoRA可以缓解深层网络中的梯度消失/爆炸问题。

**推理**：由于 $AB$ 是从零初始化的，早期训练时 $W \approx W_0$，梯度主要通过预训练的 $W_0$ 传播。随着训练进行，$AB$ 逐渐贡献更多的梯度。

这种渐进式的调整比突然改变所有权重更稳定。

### 17. LoRA的泛化性能分析

**定理17.1（泛化界）**：基于Rademacher复杂度，LoRA的泛化误差上界为：

$$
\mathcal{R}(h_{LoRA}) \leq \hat{\mathcal{R}}(h_{LoRA}) + O\left(\sqrt{\frac{r(n+m)}{N}}\right)
$$

其中 $\mathcal{R}$ 是泛化误差，$\hat{\mathcal{R}}$ 是训练误差，$N$ 是样本数。

对比全量微调：

$$
\mathcal{R}(h_{full}) \leq \hat{\mathcal{R}}(h_{full}) + O\left(\sqrt{\frac{nm}{N}}\right)
$$

**推论17.1**：当 $r(n+m) \ll nm$ 时，LoRA的泛化界更紧，意味着更好的泛化性能。

**定理17.2（VC维）**：LoRA的VC维约为 $O(r(n+m))$，远小于全量微调的 $O(nm)$。

较低的VC维意味着更强的泛化能力，特别是在样本有限时。

### 18. LoRA的变体：加性分解

**定义18.1（加性分解）**：如文章所提，一个替代方案是使用加性分解：

$$
W = W_0 + A \mathbf{1}_{1 \times m} + \mathbf{1}_{n \times 1} B
$$

其中 $A \in \mathbb{R}^{n \times 1}, B \in \mathbb{R}^{1 \times m}$，$\mathbf{1}$ 表示全1向量。

**定理18.1（加性分解的秩）**：加性分解的秩可以达到2（而乘性的秩为1当$r=1$时）：

$$
\text{rank}(A \mathbf{1}^T + \mathbf{1} B^T) \leq 2
$$

**梯度18.1（加性分解的梯度）**：

$$
\frac{\partial \mathcal{L}}{\partial A} = \left(\frac{\partial \mathcal{L}}{\partial W}\right) \mathbf{1}_{m \times 1}
$$

$$
\frac{\partial \mathcal{L}}{\partial B} = \mathbf{1}_{1 \times n} \frac{\partial \mathcal{L}}{\partial W}
$$

即原始梯度的行和与列和。

**定理18.2（加性分解的优势）**：

1. **计算简单**：加法比乘法更快
2. **秩更高**：相同参数量下秩可以更高
3. **可解释性**：$A$ 可以理解为对每个输出维度的偏置，$B$ 为对每个输入维度的缩放

**定理18.3（加性分解的扩展到高秩）**：对于 $r > 1$，可以分块处理：

将 $W$ 分为 $r \times r$ 个块，每个块使用加性分解：

$$
W = W_0 + \sum_{i=1}^r \sum_{j=1}^r A_i \mathbf{1}^T + \mathbf{1} B_j^T
$$

但这需要精心设计块的划分。

### 19. LoRA的数值稳定性

**定理19.1（条件数的影响）**：LoRA的数值稳定性与矩阵 $A$ 和 $B$ 的条件数相关。

设 $\kappa(A) = \frac{\sigma_{max}(A)}{\sigma_{min}(A)}$ 是条件数，则当 $\kappa(A)$ 或 $\kappa(B)$ 很大时，训练可能不稳定。

**解决方案19.1（正交初始化）**：可以使用正交初始化来保证 $\kappa(A) = \kappa(B) = 1$：

$$
A^T A = I_{r \times r}, \quad BB^T = I_{r \times r}
$$

这样 $A$ 和 $B$ 都是正交矩阵（的一部分），条件数为1。

**定理19.2（梯度裁剪）**：为了防止梯度爆炸，建议使用梯度裁剪：

$$
\nabla \leftarrow \begin{cases}
\nabla & \text{if } \|\nabla\| \leq \theta \\
\theta \frac{\nabla}{\|\nabla\|} & \text{if } \|\nabla\| > \theta
\end{cases}
$$

其中 $\theta$ 是裁剪阈值（如1.0）。

### 20. LoRA的可视化理解

**定理20.1（秩1的几何解释）**：当 $r = 1$ 时，$\Delta W = ab^T$ 是一个秩1矩阵，其几何意义为：

$$
\Delta W = ab^T = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_m \\ a_2 b_1 & a_2 b_2 & \cdots & a_2 b_m \\ \vdots & \vdots & \ddots & \vdots \\ a_n b_1 & a_n b_2 & \cdots & a_n b_m \end{bmatrix}
$$

每一列都是 $a$ 的标量倍数，每一行都是 $b$ 的标量倍数。

**可视化**：$\Delta W$ 的所有列向量都在同一条直线上（由 $a$ 定义），所有行向量也在同一条直线上（由 $b$ 定义）。

**定理20.2（秩r的几何解释）**：当 $r > 1$ 时：

$$
\Delta W = AB = \sum_{i=1}^r a_i b_i^T
$$

是 $r$ 个秩1矩阵的和，即：

$$
\Delta W = a_1 b_1^T + a_2 b_2^T + \cdots + a_r b_r^T
$$

**可视化**：$\Delta W$ 的列空间是 $r$ 维的，由 $\{a_1, \ldots, a_r\}$ 张成；行空间也是 $r$ 维的，由 $\{b_1, \ldots, b_r\}$ 张成。

### 21. LoRA在不同架构中的应用

**应用21.1（Transformer中的LoRA）**：在Transformer中，LoRA通常应用于：

1. Query矩阵：$W_Q = W_Q^0 + A_Q B_Q$
2. Key矩阵：$W_K = W_K^0 + A_K B_K$
3. Value矩阵：$W_V = W_V^0 + A_V B_V$
4. Output矩阵：$W_O = W_O^0 + A_O B_O$

**定理21.1（注意力中的低秩结构）**：注意力权重矩阵 $\text{softmax}(QK^T/\sqrt{d})$ 通常是低秩的。

**证明思路**：如果输入序列中有相似的token，则 $Q$ 和 $K$ 的行向量会相似，导致 $QK^T$ 有许多相近的行/列，从而是低秩的。$\square$

**应用21.2（CNN中的LoRA）**：在卷积层中，可以将卷积核reshape为矩阵后应用LoRA：

卷积核 $W \in \mathbb{R}^{c_{out} \times c_{in} \times k \times k}$ 可以reshape为 $\mathbb{R}^{c_{out} \times (c_{in} \cdot k^2)}$，然后应用LoRA。

**应用21.3（RNN中的LoRA）**：在RNN中，可以对隐藏状态转移矩阵应用LoRA：

$$
h_t = \tanh((W_0 + AB) h_{t-1} + U x_t)
$$

### 22. LoRA的理论局限

**局限22.1（秩不足的问题）**：如果真实的最优更新矩阵的秩大于 $r$，LoRA无法找到真正的最优解。

**定理22.1（次优性）**：存在任务使得LoRA的最优解严格差于全量微调：

$$
\min_{A,B} \mathcal{L}(W_0 + AB) > \min_W \mathcal{L}(W)
$$

**例子**：如果目标是学习一个满秩矩阵，而 $r < \text{rank}(\Delta W^*)$，则LoRA必然次优。

**局限22.2（任务特异性）**：最优秩 $r$ 是任务相关的，没有通用的选择准则。

某些任务需要 $r = 64$ 或更高，而其他任务 $r = 4$ 就足够了。

**局限22.3（推理开销）**：虽然可以合并权重，但在需要动态切换多个LoRA的场景下，仍有额外开销。

### 23. LoRA的扩展与变体

**变体23.1（AdaLoRA）**：自适应地为不同层和不同模块分配不同的秩。

核心思想：重要的层使用更大的 $r$，不重要的层使用更小的 $r$。

**变体23.2（QLoRA）**：结合量化的LoRA。

将预训练权重 $W_0$ 量化为低精度（如4-bit），只有 $A, B$ 使用高精度。

**公式**：
$$
W = \text{Dequant}(W_0^{quant}) + AB
$$

**优势**：进一步减少显存占用。

**变体23.3（LoRA+）**：在LoRA的基础上增加额外的正则化：

$$
\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda_1 \|A\|_F^2 + \lambda_2 \|B\|_F^2 + \lambda_3 \|AB\|_*
$$

其中 $\|\cdot\|_*$ 是核范数（所有奇异值之和），鼓励低秩。

### 24. LoRA的信息论视角

**定义24.1（互信息）**：定义输入 $x$ 和输出 $y$ 的互信息：

$$
I(X; Y) = H(Y) - H(Y|X)
$$

**定理24.1（信息瓶颈）**：LoRA通过低秩约束实现了一种信息瓶颈：

$$
\max_{A,B} I(X; Y) \quad \text{s.t.} \quad I(X; \{A,B\}) \leq r(n+m) \log 2
$$

（假设参数用固定精度编码）

**推论24.1**：LoRA强制模型在有限的"信息预算"下学习任务相关的特征，这自然地起到了正则化作用。

**定理24.2（率失真理论）**：从率失真理论的角度，LoRA在给定失真（性能损失）下最小化了参数量（率）。

### 25. LoRA的未来方向与理论展望

**方向25.1（自动秩选择）**：开发理论指导的自动秩选择方法。

**提议**：基于梯度矩阵的奇异值衰减速度自动选择 $r$：

$$
r^* = \arg\min_r \left\{ \frac{\sum_{i>r} \sigma_i^2}{\sum_{i=1}^n \sigma_i^2} < \epsilon \right\}
$$

其中 $\sigma_i$ 是梯度矩阵的奇异值，$\epsilon$ 是容忍的误差。

**方向25.2（非线性低秩分解）**：探索非线性的低秩分解：

$$
W = W_0 + f(A) g(B)
$$

其中 $f, g$ 是非线性函数（如神经网络）。

**方向25.3（动态LoRA）**：根据输入动态调整 $A$ 和 $B$：

$$
A(x) = A_0 + h_A(x), \quad B(x) = B_0 + h_B(x)
$$

其中 $h_A, h_B$ 是轻量级的动态调整网络。

**定理25.1（LoRA的普适性）**：LoRA的核心思想（低秩近似）可以应用于：

1. **模型压缩**：近似大模型的权重矩阵
2. **知识蒸馏**：教师模型到学生模型的低秩映射
3. **持续学习**：每个新任务使用一个LoRA模块
4. **元学习**：学习一个好的初始化 $W_0$，使得各任务的 $AB$ 都很小

**最终定理**：LoRA揭示了深度学习中的一个基本原理：**预训练模型的适应是低维的**。

形式化地：

$$
\mathcal{H}_{adapt} \subset \mathcal{H}_{full}, \quad \dim(\mathcal{H}_{adapt}) \ll \dim(\mathcal{H}_{full})
$$

其中 $\mathcal{H}_{adapt}$ 是适应子空间，$\mathcal{H}_{full}$ 是完整参数空间。

这一原理不仅适用于LoRA，也适用于其他参数高效微调方法，是理解和设计高效微调方法的理论基础。

### 总结

通过以上25个小节的详细推导，我们完整地分析了LoRA的：

1. **数学基础**：低秩分解、矩阵理论、梯度计算
2. **理论优势**：参数效率、计算效率、泛化性能
3. **梯度视角**：梯度的低秩结构、优化轨迹、流形约束
4. **与全参数微调的关系**：约束优化、表达能力、权衡分析
5. **变体与扩展**：加性分解、AdaLoRA、QLoRA等
6. **理论局限**：秩不足、任务特异性、次优性
7. **未来方向**：自动秩选择、非线性分解、动态LoRA

LoRA的核心贡献是证明了"预训练模型的微调可以在低维子空间中进行"这一假设的实用性，为参数高效微调提供了理论基础和实践指导。

