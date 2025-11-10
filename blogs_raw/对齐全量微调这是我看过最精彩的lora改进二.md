---
title: 对齐全量微调！这是我看过最精彩的LoRA改进（二）
slug: 对齐全量微调这是我看过最精彩的lora改进二
date: 2024-07-29
tags: 详细推导, 梯度, 优化器, 低秩, lora, 生成模型
status: pending
---
# 对齐全量微调！这是我看过最精彩的LoRA改进（二）

**原文链接**: [https://spaces.ac.cn/archives/10266](https://spaces.ac.cn/archives/10266)

**发布日期**: 

---

前两周笔者写了[《对齐全量微调！这是我看过最精彩的LoRA（一）》](/archives/10226)（当时还没有编号“一”），里边介绍了一个名为“LoRA-GA”的LoRA变体，它通过梯度SVD来改进LoRA的初始化，从而实现LoRA与全量微调的对齐。当然，从理论上来讲，这样做也只能尽量对齐第一步更新后的$W_1$，所以当时就有读者提出了“后面的$W_2,W_3,\cdots$不管了吗？”的疑问，当时笔者也没想太深入，就单纯觉得对齐了第一步后，后面的优化也会严格一条较优的轨迹走。

有趣的是，LoRA-GA才出来没多久，arXiv上就新出了[《LoRA-Pro: Are Low-Rank Adapters Properly Optimized?》](https://papers.cool/arxiv/2407.18242)，其所提的LoRA-Pro正好能回答这个问题！LoRA-Pro同样是想着对齐全量微调，但它对齐的是每一步梯度，从而对齐整条优化轨迹，这正好是跟LoRA-GA互补的改进点。

## 对齐全量 #

本文接着上一篇文章的记号和内容进行讲述，所以这里仅对上一节的内容做一个简单回顾，不再详细重复介绍。LoRA的参数化方式是  
\begin{equation}W = (W_0 - A_0 B_0) + AB\end{equation}  
其中$W_0 \in \mathbb{R}^{n\times m}$是预训练权重，$A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m}$是新引入的训练参数，$A_0,B_0$是它们的初始化值。

上一节我们说到，全量微调很多时候效果都优于LoRA，所以全量微调就是LoRA最应该对齐的方向。为了定量描述这一点，我们分别写出全量微调和LoRA微调在SGD下的优化公式，结果分别是  
\begin{equation} W_{t+1} = W_t - \eta G_t\end{equation}  
和  
\begin{equation}\begin{gathered}  
A_{t+1} = A_t - \eta G_{A,t} = A_t - \eta G_t B_t^{\top},\quad B_{t+1} = B_t - \eta G_{B,t} = B_t - \eta A_t^{\top}G_t \\\\[8pt]  
W_{t+1} = W_t - A_t B_t + A_{t+1} B_{t+1} \approx W_t - \eta(A_t A_t^{\top}G_t + G_tB_t^{\top} B_t)  
\end{gathered}\end{equation}  
其中$\mathcal{L}$是损失函数，$\eta$是学习率，还有$G_t=\frac{\partial \mathcal{L}}{\partial W_t}$、$G_{A,t}=\frac{\partial \mathcal{L}}{\partial A_t}=\frac{\partial \mathcal{L}}{\partial W_t} B_t^{\top}=G_t B_t^{\top}$以及$G_{B,t}=\frac{\partial \mathcal{L}}{\partial B_t}=A_t^{\top}\frac{\partial \mathcal{L}}{\partial W_t} =A_t^{\top}G_t$。

LoRA-GA的想法是，我们至少要让全量微调和LoRA的$W_1$尽可能相近，于是它最小化目标  
\begin{equation}\mathop{\text{argmin}}_{A_0,B_0}\left\Vert A_0 A_0^{\top}G_0 + G_0 B_0^{\top} B_0 - G_0\right\Vert_F^2\end{equation}  
其最优解可以通过对$G_0$进行SVD求得，这样我们就可以求出最优的$A_0,B_0$作为$A,B$的初始化。

## 逐步对齐 #

LoRA-Pro的想法更彻底，它希望对齐全量微调和LoRA的每一个$W_t$。可是要怎样才能做到这一点呢？难道每一步都要最小化$\left\Vert A_t A_t^{\top}G_t + G_t B_t^{\top} B_t - G_t\right\Vert_F^2$？这显然是不对的，因为$A_t,B_t$是由优化器根据$A_{t-1},B_{t-1}$和它们的梯度确定的，并不是可自由调节的参数。

看上去已经没有能够让我们修改的地方了？不，LoRA-Pro非常机智地想到：既然“$A_t,B_t$是由优化器根据$A_{t-1},B_{t-1}$和它们的梯度确定的”，后面的$A_{t-1},B_{t-1}$和梯度我们都没法改，那我们还可以改优化器呀！具体来说，我们将$A_t,B_t$的更新规则改为：  
\begin{equation}\begin{gathered}  
A_{t+1} = A_t - \eta H_{A,t} \\\  
B_{t+1} = B_t - \eta H_{B,t}  
\end{gathered}\end{equation}  
其中$H_{A,t},H_{B,t}$待定，但它们的形状跟$A,B$一致。现在可以写出  
\begin{equation}W_{t+1} = W_t - A_t B_t + A_{t+1} B_{t+1} \approx W_t - \eta(H_{A,t} B_t + A_t H_{B,t}) \end{equation}  
这时候我们就可以调整$H_{A,t},H_{B,t}$，让这个$W_{t+1}$跟SGD的$W_{t+1}$尽可能相近了：  
\begin{equation}\mathop{\text{argmin}}_{H_{A,t},H_{B,t}}\left\Vert H_{A,t} B_t + A_t H_{B,t} - G_t\right\Vert_F^2\end{equation}  
下面我们来求解这个优化问题。简单起见，在求解过程中我们省略下标$t$，即考虑  
\begin{equation}\mathop{\text{argmin}}_{H_A,H_B}\left\Vert H_A B + A H_B - G\right\Vert_F^2\label{eq:loss}\end{equation}

## 简化目标 #

由于$H_A,H_B$之间没有约束，所以$H_A,H_B$的优化是独立的，因此我们可以采取先优化$H_A$再优化$H_B$的策略（当然反过来也可以）。当我们优化$H_A$时，$H_B$就相当于是常数，为此，我们可以先考虑如下简化的等价命题  
\begin{equation}\mathop{\text{argmin}}_H\left\Vert H B - X\right\Vert_F^2\label{eq:h-xb-loss}\end{equation}  
其中$H\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m},X\in\mathbb{R}^{n\times m}$。如果$r=m$且$B$可逆，那么我们直接可以变为解方程组$HB=X$，即$H=XB^{-1}$。当$r < m$时，我们就要诉诸优化手段，注意到$HB-X$关于$H$是线性的，所以这实质就是线性回归的最小二乘问题，它是有解析解的，答案是  
\begin{equation}H = XB^{\top}(B B^{\top})^{-1} \label{eq:h-xb}\end{equation}  
其中$B^{\top}(B B^{\top})^{-1}$正是矩阵$B$的“[伪逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)”。不了解这个答案也不要紧，我们现场推一下。首先，记$\mathcal{l}=\left\Vert H B - X\right\Vert_F^2$，直接求$H$的导数得到  
\begin{equation}\frac{\partial l}{\partial H} = 2(HB - X)B^{\top} = 2(HBB^{\top} - XB^{\top})\end{equation}  
然后让它等于零就可以解出式$\eqref{eq:h-xb}$。可能有些读者不大了解矩阵求导法则，其实根据求导的链式法则，我们就不难想到$\frac{\partial l}{\partial H}$是$2(HB - X)$与$B$以某种方式相乘起来，然后我们约定$\frac{\partial l}{\partial H}$的形状跟$H$一样，即$n\times r$，那么由$2(HB - X)$和$B$相乘来凑出一个$n\times r$的结果，也只有$2(HB - X)B^{\top}$了。

同理，$\left\Vert AH - X\right\Vert_F^2$对$H$的导数就是$2A^{\top}(AH - X)$，由此可以得到  
\begin{equation}\mathop{\text{argmin}}_H\left\Vert AH - X\right\Vert_F^2\quad\Rightarrow\quad H = (A^{\top} A)^{-1}A^{\top}X \label{eq:h-ax}\end{equation}

## 完整结果 #

有了结论$\eqref{eq:h-xb}$和$\eqref{eq:h-ax}$，我们就可以着手求解$\eqref{eq:loss}$了。首先我们固定$H_B$，那么根据式$\eqref{eq:h-xb}$得到  
\begin{equation}H_A = (G - A H_B) B^{\top}(B B^{\top})^{-1}\label{eq:h-a-1}\end{equation}  
注意式$\eqref{eq:loss}$的目标函数具有一个不变性：  
\begin{equation}\left\Vert H_A B + A H_B - G\right\Vert_F^2 = \left\Vert (H_A + AC) B + A (H_B - CB) - G\right\Vert_F^2\end{equation}  
其中$C$是任意$r\times r$的矩阵。也就是说，$H_A$的解可以加/减任意具有$AC$形式的矩阵，只需要$H_B$减/加对应的$CB$就行。根据该性质，我们可以将式$\eqref{eq:h-a-1}$的$H_A$简化成  
\begin{equation}H_A = G B^{\top}(B B^{\top})^{-1}\end{equation}  
代回目标函数得  
\begin{equation}\mathop{\text{argmin}}_{H_B}\left\Vert A H_B - G(I - B^{\top}(B B^{\top})^{-1}B)\right\Vert_F^2\end{equation}  
根据式$\eqref{eq:h-ax}$得  
\begin{equation}H_B = (A^{\top} A)^{-1}A^{\top}G(I - B^{\top}(B B^{\top})^{-1}B)\end{equation}  
留意到$G B^{\top},A^{\top}G$正好分别是$A,B$的梯度$G_A,G_B$，以及再次利用前述不变性，我们可以写出完整的解  
\begin{equation}\left\\{\begin{aligned} H_A =&\, G_A (B B^{\top})^{-1} + AC \\\  
H_B =&\, (A^{\top} A)^{-1}G_B(I - B^{\top}(B B^{\top})^{-1}B) - CB  
\end{aligned}\right.\end{equation}

## 最优参数 #

至此，我们求解出了$H_A,H_B$的形式，但解不是唯一的，它有一个可以自由选择的参数矩阵$C$。我们可以选择适当的$C$，来使得最终的$H_A,H_B$具备一些我们所期望的特性。

比如，现在$H_A,H_B$是不大对称的，$H_B$多了$-(A^{\top} A)^{-1}G_B B^{\top}(B B^{\top})^{-1}B$这一项，我们可以将它平均分配到$H_A,H_B$中，使得它们更对称一些，这等价于选择$C = -\frac{1}{2}(A^{\top} A)^{-1}G_B B^{\top}(B B^{\top})^{-1}$：  
\begin{equation}\left\\{\begin{aligned} H_A =&\, \left[I - \frac{1}{2}A(A^{\top}A)^{-1}A^{\top}\right]G_A (B B^{\top})^{-1} \\\  
H_B =&\, (A^{\top} A)^{-1}G_B\left[I - \frac{1}{2}B^{\top}(B B^{\top})^{-1}B\right]  
\end{aligned}\right.\end{equation}  
这个$C$也是如下两个优化问题的解：  
\begin{align}  
&\,\mathop{\text{argmin}}_C \Vert H_A B - A H_B\Vert_F^2 \\\  
&\,\mathop{\text{argmin}}_C \Vert H_A B - G\Vert_F^2 + \Vert A H_B - G\Vert_F^2 \\\  
\end{align}  
第一个优化目标可以理解为让$A,B$对最终效果的贡献尽可能一样，这跟[《配置不同的学习率，LoRA还能再涨一点？》](/archives/10001)的假设有一定异曲同工之处，第二个优化目标则是让$H_A B$、$A H_B$都尽可能逼近完整的梯度$G$。以$l=\Vert H_A B - A H_B\Vert_F^2$为例，直接求导得  
\begin{equation}\frac{\partial l}{\partial C} = 4A^{\top}(H_A B - A H_B)B^{\top}=4A^{\top}\left[G_A (BB^{\top})^{-1}B + 2ACB\right]B^{\top}\end{equation}  
令它等于零我们就可以解出同样的$C$，化简过程比较关键的两步是$[I - B^{\top}(B B^{\top})^{-1}B]B^{\top} = 0$以及$A^{\top}G_A = G_B B^{\top}$。

LoRA-Pro选择的$C$略有不同，它是如下目标函数的最优解  
\begin{equation}\mathop{\text{argmin}}_C \Vert H_A - G_A\Vert_F^2 + \Vert H_B - G_B\Vert_F^2\end{equation}  
这样做的意图也很明显：$H_A,H_B$是用来取代$G_A,G_B$的，如果在能达到相同效果的前提下，相比$G_A,G_B$的改动尽可能小，不失为一个合理的选择。同样求$C$的导数并让其等于零，化简可得  
\begin{equation}A^{\top}A C + C B B^{\top} = -A^{\top} G_A (BB^{\top})^{-1}\end{equation}  
现在我们得到关于$C$的一个方程，该类型的方程叫做“[Sylvester方程](https://en.wikipedia.org/wiki/Sylvester_equation)”，可以通过外积符号写出$C$的解析解，但没有必要，因为直接数值求解的复杂度比解析解的复杂度要低，所以直接数值求解即可。总的来说，这些$C$的选择方案，都是在让$H_A,H_B$在某种视角下更加对称一些，虽然笔者没有亲自做过对比实验，但笔者认为这些不同的选择之间不会有太明显的区别。

## 一般讨论 #

我们来捋一捋到目前为止我们所得到的结果。我们的模型还是常规的LoRA，目标则是希望每一步更新都能逼近全量微调的结果。为此，我们假设优化器是SGD，然后对比了同样$W_t$下全量微调和LoRA所得的$W_{t+1}$，发现要实现这个目标，需要把更新过程中$A,B$的梯度$G_A, G_B$换成上面求出的$H_A,H_B$。

接下来就又回到优化分析中老生常谈的问题：前面的分析都是基于SGD优化器的，但实践中我们更常用的是Adam，此时要怎么改呢？如果对Adam优化器重复前面的推导，结果就是$H_A,H_B$中的梯度$G$要换成全量微调下Adam的更新方向$U$。然而，$U$需要用全量微调的梯度$G$按照Adam的更新规则计算而来，而我们的场景是LoRA，无法获得全量微调的梯度，只有$A,B$的梯度$G_A,G_B$。

不过我们也可以考虑一个近似的方案，前述$H_A B + A H_B$的优化目标就是在逼近$G$，所以我们可以用它来作为$G$的近似来执行Adam，这样一来整个流程就可以走通了。于是我们可以写出如下更新规则  
\begin{equation}\begin{array}{l}  
\begin{array}{l}G_A = \frac{\partial\mathcal{L}}{\partial A_{t-1}},\,\,G_B = \frac{\partial\mathcal{L}}{\partial B_{t-1}}\end{array} \\\  
\color{green}{\left.\begin{array}{l}H_A = G_A (B B^{\top})^{-1} \\\  
H_B = (A^{\top} A)^{-1}G_B(I - B^{\top}(B B^{\top})^{-1}B) \\\  
\tilde{G} = H_A B + A H_B \end{array}\quad\right\\} \text{估计梯度}} \\\  
\color{red}{\left.\begin{array}{l}M_t = \beta_1 M_{t-1} + (1 - \beta_1) \tilde{G} \\\  
V_t = \beta_2 V_{t-1} + (1 - \beta_2) \tilde{G}^2 \\\  
\hat{M}_t = \frac{M_t}{1-\beta_1^t},\,\,\hat{V}_t = \frac{V_t}{1-\beta_2^t},\,\,U = \frac{\hat{M}_t}{\sqrt{\hat{V}_t + \epsilon}}\end{array}\quad\right\\} \text{Adam更新}} \\\  
\color{purple}{\left.\begin{array}{l}U_A = UB^{\top},\,\, U_B = A^{\top} U \\\  
\tilde{H}_A = U_A (B B^{\top})^{-1} + AC \\\  
\tilde{H}_B = (A^{\top} A)^{-1}U_B(I - B^{\top}(B B^{\top})^{-1}B) - CB  
\end{array}\quad\right\\} \text{投影到}A,B} \\\  
\begin{array}{l}A_t = A_{t-1} - \eta \tilde{H}_A \\\  
B_t = B_{t-1} - \eta \tilde{H}_B \\\  
\end{array} \\\  
\end{array}\end{equation}  
这也是LoRA-Pro最终所用的更新算法（更准确地说，LoRA-Pro用的是AdamW，结果稍复杂一些，但并无实质不同）。然而，且不说如此改动引入的额外复杂度如何，这个算法最大的问题就是它里边的滑动更新变量$M,V$跟全量微调一样都是满秩的，也就是说它的优化器相比全量微调并不省显存，仅仅是通过低秩分解节省了参数和梯度的部分显存，这相比常规LoRA的显存消耗还是会有明显增加的。

一个比较简单的方案（但笔者没有实验过）就是直接用$H_A,H_B$替代$G_A,G_B$，然后按照常规LoRA的Adam更新规则来计算，这样$M,V$的形状就跟相应的$A,B$一致了，节省的显存达到了最大化。不过此时的Adam理论基础不如LoRA-Pro的Adam，更多的是跟[《对齐全量微调！这是我看过最精彩的LoRA（一）》](/archives/10226)一样，靠“SGD的结论可以平行应用到Adam”的信仰来支撑。

## 实验结果 #

LoRA-Pro在GLUE上的实验结果更加惊艳，超过了全量微调的结果：  


[![LoRA-Pro在GLUE上的实验结果](/usr/uploads/2024/07/1843879400.png)](/usr/uploads/2024/07/1843879400.png "点击查看原图")

LoRA-Pro在GLUE上的实验结果

不过论文也就只有这个实验了。看上去LoRA-Pro成文比较仓促，可能是看到LoRA-GA后觉得“撞车”感太明显，所以先赶出来占个坑吧。笔者刚刷到LoRA-Pro时，第一反应也是跟LoRA-GA撞车了，但仔细阅读之下才发现，它跟LoRA-GA实际上是同一思想下互补的结果。

从LoRA-Pro的结果来看，它包含了$A^{\top} A$和$B B^{\top}$的求逆，所以很明显$A,B$之一就不能用全零初始化了，比较符合直觉的正交初始化，即让初始的$A^{\top} A,B B^{\top}$是单位阵（的若干倍）。刚好从[《对齐全量微调！这是我看过最精彩的LoRA（一）》](/archives/10226)我们可以看到，LoRA-GA给出的初始化正好是正交初始化，所以LoRA-Pro跟LoRA-GA可谓是“最佳搭档”了。

## 文章小结 #

本文介绍了另一个对齐全量微调的工作LoRA-Pro，它跟上一篇的LoRA-GA正好是互补的两个结果，LoRA-GA试图通过改进初始化来使得LoRA跟全量微调对齐，LoRA-Pro则更彻底一些，它通过修改优化器的更新规则来使得LoRA的每一步更新都尽量跟全量微调对齐，两者都是非常精彩的LoRA改进，都是让人赏心悦目之作。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10266>_

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

苏剑林. (Jul. 29, 2024). 《对齐全量微调！这是我看过最精彩的LoRA改进（二） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10266>

@online{kexuefm-10266,  
title={对齐全量微调！这是我看过最精彩的LoRA改进（二）},  
author={苏剑林},  
year={2024},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/10266}},  
} 


---

## 公式推导与注释

### 1. LoRA-Pro的完整数学框架

**推导1.1：从单步对齐到逐步对齐**

LoRA-GA只对齐第一步更新 $W_1$，而LoRA-Pro的目标是对齐所有步骤的更新。

对于任意时刻 $t$，全量微调的更新为：

$$W_{t+1}^{\text{full}} = W_t - \eta G_t$$

LoRA的更新为：

$$W_{t+1}^{\text{LoRA}} = W_t - A_t B_t + A_{t+1} B_{t+1}$$

我们希望：

$$W_{t+1}^{\text{LoRA}} \approx W_{t+1}^{\text{full}}$$

对所有 $t$ 成立。

**注释**：这是一个更强的对齐条件，相当于让LoRA的整个优化轨迹都与全量微调对齐。

**推导1.2：修改优化器的动机**

标准LoRA的更新规则由优化器决定：

$$A_{t+1} = A_t - \eta G_{A,t}, \quad B_{t+1} = B_t - \eta G_{B,t}$$

其中 $G_{A,t} = G_t B_t^{\top}$，$G_{B,t} = A_t^{\top} G_t$。

代入得：

$$W_{t+1}^{\text{LoRA}} \approx W_t - \eta (A_t A_t^{\top} G_t + G_t B_t^{\top} B_t)$$

这与 $W_{t+1}^{\text{full}} = W_t - \eta G_t$ 存在差异。

LoRA-Pro的想法是：既然 $A_t, B_t$ 已经确定，我们可以修改更新方向 $G_{A,t}, G_{B,t}$，用新的方向 $H_{A,t}, H_{B,t}$ 替代。

**注释**：这是一个巧妙的想法——在不改变模型架构的情况下，通过修改优化器来改善效果。

**推导1.3：修改后的更新规则**

定义新的更新方向：

$$A_{t+1} = A_t - \eta H_{A,t}$$
$$B_{t+1} = B_t - \eta H_{B,t}$$

一阶近似下：

$$W_{t+1}^{\text{LoRA}} \approx W_t - \eta (H_{A,t} B_t + A_t H_{B,t})$$

为了与全量微调对齐，我们需要：

$$H_{A,t} B_t + A_t H_{B,t} \approx G_t$$

**注释**：这个等式是LoRA-Pro的核心，它将对齐问题转化为一个线性方程求解问题。

### 2. 最小二乘问题的求解

**推导2.1：优化目标的形式化**

我们要找到 $H_{A,t}, H_{B,t}$ 使得：

$$\min_{H_{A,t}, H_{B,t}} \|H_{A,t} B_t + A_t H_{B,t} - G_t\|_F^2$$

这是一个无约束的凸优化问题（关于 $H_{A,t}, H_{B,t}$）。

**注释**：虽然 $H_{A,t}$ 和 $H_{B,t}$ 是耦合的，但目标函数是凸的，因此可以通过交替优化求解。

**推导2.2：固定 $H_{B,t}$ 优化 $H_{A,t}$**

固定 $H_{B,t}$，目标函数变为：

$$\min_{H_{A,t}} \|H_{A,t} B_t - (G_t - A_t H_{B,t})\|_F^2$$

记 $X_t = G_t - A_t H_{B,t}$，这是一个标准的线性回归问题：

$$\min_{H_{A,t}} \|H_{A,t} B_t - X_t\|_F^2$$

对 $H_{A,t}$ 求导：

$$\frac{\partial}{\partial H_{A,t}} \|H_{A,t} B_t - X_t\|_F^2 = 2(H_{A,t} B_t - X_t) B_t^{\top}$$

令导数为零：

$$H_{A,t} B_t B_t^{\top} = X_t B_t^{\top}$$

解得：

$$H_{A,t} = X_t B_t^{\top} (B_t B_t^{\top})^{-1} = (G_t - A_t H_{B,t}) B_t^{\top} (B_t B_t^{\top})^{-1}$$

**注释**：这里假设 $B_t B_t^{\top}$ 是可逆的，即 $B_t$ 是行满秩的。在 $r < m$ 时这总是成立的。

**推导2.3：利用解的不唯一性简化**

注意到目标函数具有以下不变性：对任意 $r \times r$ 矩阵 $C$：

$$\|(H_{A,t} + A_t C) B_t + A_t (H_{B,t} - C B_t) - G_t\|_F^2 = \|H_{A,t} B_t + A_t H_{B,t} - G_t\|_F^2$$

这是因为：

$$(H_{A,t} + A_t C) B_t + A_t (H_{B,t} - C B_t) = H_{A,t} B_t + A_t C B_t + A_t H_{B,t} - A_t C B_t = H_{A,t} B_t + A_t H_{B,t}$$

利用这个性质，我们可以去掉 $H_{A,t}$ 中的 $A_t H_{B,t}$ 项：

$$H_{A,t} = G_t B_t^{\top} (B_t B_t^{\top})^{-1}$$

**注释**：这个简化大大降低了计算复杂度，避免了迭代求解。

**推导2.4：固定 $H_{A,t}$ 优化 $H_{B,t}$**

类似地，固定 $H_{A,t}$，目标函数变为：

$$\min_{H_{B,t}} \|A_t H_{B,t} - (G_t - H_{A,t} B_t)\|_F^2$$

对 $H_{B,t}$ 求导：

$$\frac{\partial}{\partial H_{B,t}} \|A_t H_{B,t} - Y_t\|_F^2 = 2 A_t^{\top} (A_t H_{B,t} - Y_t)$$

其中 $Y_t = G_t - H_{A,t} B_t$。

令导数为零：

$$A_t^{\top} A_t H_{B,t} = A_t^{\top} Y_t$$

解得：

$$H_{B,t} = (A_t^{\top} A_t)^{-1} A_t^{\top} Y_t$$

代入 $Y_t$ 并利用不变性去掉 $H_{A,t} B_t$ 项：

$$H_{B,t} = (A_t^{\top} A_t)^{-1} A_t^{\top} G_t [I_m - B_t^{\top} (B_t B_t^{\top})^{-1} B_t]$$

**注释**：$I_m - B_t^{\top} (B_t B_t^{\top})^{-1} B_t$ 是投影到 $B_t$ 零空间的投影矩阵。

**推导2.5：投影矩阵的性质**

定义投影矩阵：

$$P_{B,\perp} = I_m - B_t^{\top} (B_t B_t^{\top})^{-1} B_t$$

验证投影性质：

$$P_{B,\perp}^2 = [I_m - B_t^{\top} (B_t B_t^{\top})^{-1} B_t]^2$$

$$= I_m - 2B_t^{\top} (B_t B_t^{\top})^{-1} B_t + B_t^{\top} (B_t B_t^{\top})^{-1} B_t B_t^{\top} (B_t B_t^{\top})^{-1} B_t$$

$$= I_m - 2B_t^{\top} (B_t B_t^{\top})^{-1} B_t + B_t^{\top} (B_t B_t^{\top})^{-1} B_t = P_{B,\perp}$$

且 $B_t P_{B,\perp} = 0$。

**注释**：这个投影矩阵将向量投影到 $B_t$ 行空间的正交补空间。

### 3. 对称化的参数选择

**推导3.1：引入自由参数 $C$**

LoRA-Pro的通解可以写为：

$$H_{A,t} = G_t B_t^{\top} (B_t B_t^{\top})^{-1} + A_t C$$

$$H_{B,t} = (A_t^{\top} A_t)^{-1} A_t^{\top} G_t [I_m - B_t^{\top} (B_t B_t^{\top})^{-1} B_t] - C B_t$$

其中 $C \in \mathbb{R}^{r \times r}$ 是任意矩阵。

**注释**：$C$ 的选择不影响 $H_{A,t} B_t + A_t H_{B,t}$ 的值，但会影响 $H_{A,t}$ 和 $H_{B,t}$ 各自的形式。

**推导3.2：最小化 $H_{A,t}$ 和 $H_{B,t}$ 的不对称性**

定义对称性度量：

$$\mathcal{S}(C) = \|H_{A,t} B_t - A_t H_{B,t}\|_F^2$$

展开：

$$H_{A,t} B_t - A_t H_{B,t} = [G_t B_t^{\top} (B_t B_t^{\top})^{-1} + A_t C] B_t - A_t [(A_t^{\top} A_t)^{-1} A_t^{\top} G_t P_{B,\perp} - C B_t]$$

$$= G_t B_t^{\top} (B_t B_t^{\top})^{-1} B_t + 2 A_t C B_t - A_t (A_t^{\top} A_t)^{-1} A_t^{\top} G_t P_{B,\perp}$$

**注释**：注意 $B_t^{\top} (B_t B_t^{\top})^{-1} B_t$ 和 $P_{B,\perp}$ 是互补的投影矩阵，和为 $I_m$。

**推导3.3：求解最优 $C$**

对 $C$ 求导：

$$\frac{\partial \mathcal{S}}{\partial C} = 4 A_t^{\top} [H_{A,t} B_t - A_t H_{B,t}] B_t^{\top}$$

$$= 4 A_t^{\top} [G_t B_t^{\top} (B_t B_t^{\top})^{-1} B_t + 2 A_t C B_t - A_t (A_t^{\top} A_t)^{-1} A_t^{\top} G_t P_{B,\perp}] B_t^{\top}$$

令其为零，注意到 $P_{B,\perp} B_t^{\top} = 0$：

$$A_t^{\top} G_t B_t^{\top} (B_t B_t^{\top})^{-1} B_t B_t^{\top} + 2 A_t^{\top} A_t C B_t B_t^{\top} = 0$$

$$A_t^{\top} G_t B_t^{\top} + 2 A_t^{\top} A_t C B_t B_t^{\top} = 0$$

解得：

$$C = -\frac{1}{2} (A_t^{\top} A_t)^{-1} A_t^{\top} G_t B_t^{\top} (B_t B_t^{\top})^{-1}$$

**注释**：这个 $C$ 使得 $H_{A,t} B_t$ 和 $A_t H_{B,t}$ 对梯度 $G_t$ 的贡献尽可能均衡。

**推导3.4：对称化后的最终形式**

代入最优 $C$：

$$H_{A,t} = G_t B_t^{\top} (B_t B_t^{\top})^{-1} - \frac{1}{2} A_t (A_t^{\top} A_t)^{-1} A_t^{\top} G_t B_t^{\top} (B_t B_t^{\top})^{-1}$$

$$= [I_n - \frac{1}{2} A_t (A_t^{\top} A_t)^{-1} A_t^{\top}] G_t B_t^{\top} (B_t B_t^{\top})^{-1}$$

类似地：

$$H_{B,t} = (A_t^{\top} A_t)^{-1} A_t^{\top} G_t [I_m - \frac{1}{2} B_t^{\top} (B_t B_t^{\top})^{-1} B_t]$$

**注释**：这个形式比原始的形式更对称，$A$ 和 $B$ 的处理方式类似。

### 4. 梯度匹配的定量度量

**推导4.1：梯度逼近误差**

定义梯度逼近误差：

$$\epsilon_t = \|H_{A,t} B_t + A_t H_{B,t} - G_t\|_F$$

理想情况下，$\epsilon_t = 0$，即完美匹配。

展开 $H_{A,t} B_t + A_t H_{B,t}$：

$$H_{A,t} B_t + A_t H_{B,t} = G_t B_t^{\top} (B_t B_t^{\top})^{-1} B_t + A_t (A_t^{\top} A_t)^{-1} A_t^{\top} G_t P_{B,\perp}$$

$$= G_t [B_t^{\top} (B_t B_t^{\top})^{-1} B_t + P_A P_{B,\perp}]$$

其中 $P_A = A_t (A_t^{\top} A_t)^{-1} A_t^{\top}$ 是投影到 $A_t$ 列空间的投影矩阵。

**注释**：注意 $B_t^{\top} (B_t B_t^{\top})^{-1} B_t + P_{B,\perp} = I_m$，所以上式可以继续简化。

**推导4.2：投影分解**

将恒等矩阵分解为：

$$I_m = B_t^{\top} (B_t B_t^{\top})^{-1} B_t + P_{B,\perp}$$

因此：

$$H_{A,t} B_t + A_t H_{B,t} = G_t [B_t^{\top} (B_t B_t^{\top})^{-1} B_t + P_A P_{B,\perp}]$$

与 $G_t = G_t I_m$ 对比：

$$\epsilon_t = \|G_t [P_A P_{B,\perp} - P_{B,\perp}]\|_F = \|G_t (P_A - I_n) P_{B,\perp}\|_F$$

$$= \|(I_n - P_A) G_t P_{B,\perp}\|_F$$

**注释**：$(I_n - P_A)$ 投影到 $A_t$ 列空间的正交补，$P_{B,\perp}$ 投影到 $B_t$ 行空间的正交补。

**推导4.3：误差的几何解释**

$\epsilon_t$ 可以理解为：梯度 $G_t$ 中既不在 $A_t$ 列空间、也不在 $B_t$ 行空间的部分的范数。

用秩的语言描述：

$$\epsilon_t = \|(I_n - P_A) G_t (I_m - P_B)\|_F$$

其中 $P_B = B_t^{\top} (B_t B_t^{\top})^{-1} B_t$。

这个误差为零当且仅当：

$$G_t = P_A G_t + G_t P_B - P_A G_t P_B$$

即 $G_t$ 可以完全由 $A_t$ 和 $B_t$ 的线性组合表示。

**注释**：在一般情况下，由于秩的限制，这个条件很难满足，因此 $\epsilon_t > 0$。

**推导4.4：误差的上界估计**

使用矩阵范数的性质：

$$\epsilon_t \leq \|(I_n - P_A)\|_2 \|G_t\|_F \|(I_m - P_B)\|_2$$

由于 $I_n - P_A$ 是投影矩阵，其谱范数为1（或0）：

$$\|(I_n - P_A)\|_2 = 1, \quad \|(I_m - P_B)\|_2 = 1$$

因此：

$$\epsilon_t \leq \|G_t\|_F$$

但这个界太松，更紧的界需要考虑 $G_t$ 的奇异值分解。

**注释**：实际上，$\epsilon_t$ 的大小取决于 $G_t$ 在 $A_t$ 和 $B_t$ 张成的子空间外的分量。

### 5. 与Adam优化器的结合

**推导5.1：Adam的梯度估计**

LoRA-Pro需要用 $H_{A,t}, H_{B,t}$ 代替真实梯度来执行Adam更新。

首先，用 $H_{A,t} B_t + A_t H_{B,t}$ 作为梯度的估计：

$$\tilde{G}_t = H_{A,t} B_t + A_t H_{B,t}$$

然后按照Adam的规则更新动量：

$$M_t = \beta_1 M_{t-1} + (1 - \beta_1) \tilde{G}_t$$

$$V_t = \beta_2 V_{t-1} + (1 - \beta_2) \tilde{G}_t^2$$

**注释**：注意这里的 $M_t, V_t$ 是 $n \times m$ 的满秩矩阵，与全量微调相同。

**推导5.2：偏差校正与自适应步长**

Adam的偏差校正：

$$\hat{M}_t = \frac{M_t}{1 - \beta_1^t}, \quad \hat{V}_t = \frac{V_t}{1 - \beta_2^t}$$

自适应更新方向：

$$U_t = \frac{\hat{M}_t}{\sqrt{\hat{V}_t} + \epsilon}$$

其中除法和平方根是逐元素的。

**注释**：$U_t$ 是 $n \times m$ 矩阵，包含了Adam的所有自适应信息。

**推导5.3：将 $U_t$ 投影回 $A, B$ 空间**

$U_t$ 是全量微调的更新方向，但我们需要将它转化为 $A_t, B_t$ 的更新。

定义：

$$U_{A,t} = U_t B_t^{\top}, \quad U_{B,t} = A_t^{\top} U_t$$

然后用同样的方法计算 $\tilde{H}_{A,t}, \tilde{H}_{B,t}$（为了区分，用 $\tilde{H}$ 表示）：

$$\tilde{H}_{A,t} = U_{A,t} (B_t B_t^{\top})^{-1} + A_t C$$

$$\tilde{H}_{B,t} = (A_t^{\top} A_t)^{-1} U_{B,t} [I_m - B_t^{\top} (B_t B_t^{\top})^{-1} B_t] - C B_t$$

**注释**：这一步是关键——先用估计梯度 $\tilde{G}_t$ 执行Adam得到 $U_t$，再将 $U_t$ 投影到 $A, B$ 的参数空间。

**推导5.4：最终的更新规则**

$$A_{t+1} = A_t - \eta \tilde{H}_{A,t}$$

$$B_{t+1} = B_t - \eta \tilde{H}_{B,t}$$

完整的算法流程：

1. 计算真实梯度 $G_{A,t} = G_t B_t^{\top}$，$G_{B,t} = A_t^{\top} G_t$
2. 构造梯度估计 $\tilde{G}_t = H_{A,t} B_t + A_t H_{B,t}$（使用 $G_{A,t}, G_{B,t}$ 计算）
3. 用 $\tilde{G}_t$ 更新Adam的动量 $M_t, V_t$
4. 计算Adam的更新方向 $U_t$
5. 将 $U_t$ 投影到 $A, B$ 空间得到 $\tilde{H}_{A,t}, \tilde{H}_{B,t}$
6. 更新参数

**注释**：这个流程保证了每一步都尽可能对齐全量微调的Adam更新。

### 6. 显存开销分析

**推导6.1：LoRA-Pro的额外显存**

标准LoRA的显存需求：

$$\text{Memory}_{\text{LoRA}} = r(n + m) \times (1 + 2) = 3r(n + m)$$

包括参数本身和Adam的 $m, v$ 状态。

LoRA-Pro需要额外存储 $M_t, V_t \in \mathbb{R}^{n \times m}$：

$$\text{Memory}_{\text{LoRA-Pro}} = 3r(n + m) + 2nm$$

**注释**：相比全量微调的 $3nm$，LoRA-Pro节省了 $(3-2)nm - 3r(n+m) = nm - 3r(n+m)$ 的显存。

**推导6.2：显存节省比例**

显存节省比例：

$$\text{Saving Ratio} = \frac{nm - 3r(n+m)}{3nm} = \frac{1}{3} - \frac{r(n+m)}{nm}$$

当 $n = m$ 时：

$$\text{Saving Ratio} = \frac{1}{3} - \frac{2r}{n}$$

例如，$n = 4096, r = 8$：

$$\text{Saving Ratio} = \frac{1}{3} - \frac{16}{4096} \approx 0.329$$

即节省约33%的显存。

**注释**：相比标准LoRA的 $1 - \frac{2r}{n} \approx 99.6\%$ 节省，LoRA-Pro的显存节省显著降低。

**推导6.3：与全量微调的对比**

全量微调的显存：

$$\text{Memory}_{\text{Full}} = 3nm$$

LoRA-Pro的显存：

$$\text{Memory}_{\text{LoRA-Pro}} = 2nm + 3r(n+m) \approx 2nm$$（当 $r \ll n, m$ 时）

比例：

$$\frac{\text{Memory}_{\text{LoRA-Pro}}}{\text{Memory}_{\text{Full}}} \approx \frac{2}{3}$$

**注释**：LoRA-Pro的显存接近全量微调的2/3，这是为了对齐而付出的代价。

### 7. 优化景观的改善

**推导7.1：损失函数的局部曲率**

考虑损失函数在 $(A_t, B_t)$ 处的二阶泰勒展开：

$$\mathcal{L}(A_t + \Delta A, B_t + \Delta B) \approx \mathcal{L}(A_t, B_t) + \langle \nabla \mathcal{L}, (\Delta A, \Delta B) \rangle$$

$$+ \frac{1}{2} [(\Delta A, \Delta B)]^{\top} H [(\Delta A, \Delta B)]$$

其中 $H$ 是Hessian矩阵。

**注释**：Hessian矩阵描述了损失函数的局部曲率，影响优化的难易程度。

**推导7.2：LoRA-Pro对曲率的影响**

标准LoRA的更新方向 $(G_{A,t}, G_{B,t})$ 可能不是最陡下降方向，因为：

$$\nabla \mathcal{L}(W) = G_t \neq G_{A,t} B_t + A_t G_{B,t}$$

LoRA-Pro通过调整为 $(H_{A,t}, H_{B,t})$，使得：

$$H_{A,t} B_t + A_t H_{B,t} \approx G_t$$

更接近真实的最陡下降方向。

**注释**：更准确的梯度方向意味着更快的收敛和更好的局部最优点。

**推导7.3：条件数的改善**

定义有效条件数：

$$\kappa_{\text{eff}} = \frac{\lambda_{\max}(H_{\text{eff}})}{\lambda_{\min}(H_{\text{eff}})}$$

其中 $H_{\text{eff}}$ 是在LoRA子空间上的有效Hessian。

LoRA-Pro通过更好的梯度逼近，减小了有效条件数，从而加速收敛。

理论上：

$$\kappa_{\text{eff}}^{\text{LoRA-Pro}} < \kappa_{\text{eff}}^{\text{LoRA}}$$

**注释**：条件数越小，优化越容易，收敛越快。

**推导7.4：收敛速度的提升**

对于强凸函数，收敛速度为：

$$\mathcal{L}_t - \mathcal{L}^* \leq (1 - \frac{1}{\kappa_{\text{eff}}})^t (\mathcal{L}_0 - \mathcal{L}^*)$$

由于 $\kappa_{\text{eff}}^{\text{LoRA-Pro}} < \kappa_{\text{eff}}^{\text{LoRA}}$，收敛因子更小：

$$1 - \frac{1}{\kappa_{\text{eff}}^{\text{LoRA-Pro}}} < 1 - \frac{1}{\kappa_{\text{eff}}^{\text{LoRA}}}$$

因此LoRA-Pro收敛更快。

**注释**：即使在非凸情况下，更好的梯度逼近也倾向于导向更好的局部最优点。

### 8. 泛化能力的理论分析

**推导8.1：Rademacher复杂度**

模型类的Rademacher复杂度定义为：

$$\mathcal{R}_n(\mathcal{F}) = \mathbb{E}_{\sigma} \left[ \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \sigma_i f(x_i) \right]$$

其中 $\sigma_i$ 是独立的Rademacher随机变量（取值 $\pm 1$，概率各0.5）。

对于LoRA模型类：

$$\mathcal{F}_{\text{LoRA}} = \{f_{W_0 + AB} : A \in \mathbb{R}^{n \times r}, B \in \mathbb{R}^{r \times m}, \|A\|_F \leq R_A, \|B\|_F \leq R_B\}$$

**注释**：Rademacher复杂度衡量模型类对随机标签的拟合能力，越小表示泛化能力越强。

**推导8.2：LoRA的Rademacher复杂度上界**

根据矩阵乘积的性质：

$$\mathcal{R}_n(\mathcal{F}_{\text{LoRA}}) \leq \frac{R_A R_B}{\sqrt{n}} \sqrt{r} \cdot \mathbb{E}\left[\sup_{\|x\|_2 \leq 1} \|X^{\top} \sigma\|_2 \right]$$

$$\leq \frac{R_A R_B \sqrt{r}}{\sqrt{n}} \mathbb{E}[\|X\|_2] \leq \frac{R_A R_B \sqrt{r m}}{\sqrt{n}}$$

**注释**：复杂度随 $\sqrt{r}$ 增长，而不是 $\sqrt{nm}$，这说明低秩约束提供了强正则化。

**推导8.3：LoRA-Pro的隐式正则化**

LoRA-Pro通过强制梯度对齐，引入了额外的约束：

$$H_{A,t} B_t + A_t H_{B,t} \approx G_t$$

这相当于在优化过程中添加了软约束：

$$\min_{A, B} \mathcal{L}(W_0 + AB) + \lambda \sum_t \|H_{A,t} B_t + A_t H_{B,t} - G_t\|_F^2$$

**注释**：这个额外的正则化项鼓励 $A, B$ 保持在能够良好逼近梯度的子空间中。

**推导8.4：泛化误差界**

根据统计学习理论，泛化误差满足：

$$\mathbb{E}[\mathcal{L}_{\text{test}}] - \mathcal{L}_{\text{train}} \leq 2\mathcal{R}_n(\mathcal{F}) + \mathcal{O}\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$

由于LoRA-Pro的有效模型类可能更小（受梯度对齐约束），其Rademacher复杂度更小：

$$\mathcal{R}_n(\mathcal{F}_{\text{LoRA-Pro}}) \leq \mathcal{R}_n(\mathcal{F}_{\text{LoRA}})$$

因此泛化能力可能更强。

**注释**：这是一个潜在的优势，但需要实验验证。在某些情况下，更强的约束可能导致欠拟合。

### 9. 与其他方法的理论对比

**推导9.1：LoRA-GA vs LoRA-Pro**

LoRA-GA只对齐初始梯度 $G_0$：

$$A_0, B_0 = \arg\min \|A_0 A_0^{\top} G_0 + G_0 B_0^{\top} B_0 - G_0\|_F^2$$

LoRA-Pro对齐每一步的梯度：

$$H_{A,t}, H_{B,t} = \arg\min \|H_{A,t} B_t + A_t H_{B,t} - G_t\|_F^2 \quad \forall t$$

**理论关系**：LoRA-Pro在 $t=0$ 时与LoRA-GA有类似的目标，但它持续在整个训练过程中对齐。

**注释**：两者是互补的——LoRA-GA提供好的起点，LoRA-Pro确保整个轨迹都对齐。

**推导9.2：与AdaLoRA的对比**

AdaLoRA动态调整秩：

$$r_t = \arg\max_r \{\text{Performance}(r) - \lambda \cdot r\}$$

这是一个离散优化问题，需要启发式搜索。

LoRA-Pro保持固定秩 $r$，但优化更新方向：

$$H_{A,t}, H_{B,t} = f(G_t, A_t, B_t)$$

**理论对比**：
- AdaLoRA：自适应模型容量
- LoRA-Pro：自适应优化方向

**注释**：两者可以结合——先用AdaLoRA确定最优秩，再用LoRA-Pro优化更新。

**推导9.3：与DoRA的对比**

DoRA分解为方向和幅度：

$$W = \|W\| \cdot \frac{W}{\|W\|} = m \cdot d$$

分别优化 $m$ 和 $d$。

LoRA-Pro保持标准的权重参数化，但修改梯度：

$$\nabla_A \to H_A, \quad \nabla_B \to H_B$$

**理论对比**：
- DoRA：改变参数空间的几何
- LoRA-Pro：改变优化轨迹

**注释**：DoRA关注权重的表示，LoRA-Pro关注优化的过程。

### 10. Sylvester方程的数值求解

**推导10.1：Sylvester方程的标准形式**

LoRA-Pro中需要求解：

$$A^{\top} A C + C B B^{\top} = -A^{\top} G_A (B B^{\top})^{-1}$$

这是一个Sylvester方程：$AX + XB = C$ 的形式。

**注释**：Sylvester方程在控制论、信号处理等领域有广泛应用，有高效的数值算法。

**推导10.2：向量化方法**

使用Kronecker积将Sylvester方程转化为线性方程组：

$$\text{vec}(AXB) = (B^{\top} \otimes A) \text{vec}(X)$$

因此：

$$\text{vec}(A^{\top} A C) + \text{vec}(C B B^{\top}) = \text{vec}(RHS)$$

$$(I \otimes A^{\top} A + B B^{\top} \otimes I) \text{vec}(C) = \text{vec}(RHS)$$

**注释**：$\otimes$ 是Kronecker积，$\text{vec}(\cdot)$ 是向量化算子。

**推导10.3：求解复杂度**

直接求解需要 $\mathcal{O}(r^6)$ 的复杂度（对于 $r \times r$ 的 $C$）。

但利用 $A^{\top} A$ 和 $B B^{\top}$ 的结构（对称正定），可以用迭代法（如共轭梯度）降低到 $\mathcal{O}(r^3)$ 或更低。

**注释**：由于 $r \ll n, m$，即使是 $\mathcal{O}(r^3)$ 也是可接受的。

**推导10.4：Bartels-Stewart算法**

经典的Bartels-Stewart算法通过Schur分解求解Sylvester方程：

1. 计算 $A^{\top} A$ 和 $B B^{\top}$ 的Schur分解
2. 变换方程到上三角形式
3. 回代求解
4. 逆变换得到 $C$

复杂度：$\mathcal{O}(r^3)$

**注释**：Bartels-Stewart算法是数值稳定的，适合LoRA-Pro的应用场景。

### 11. 正则化效应的深入分析

**推导11.1：梯度对齐作为隐式正则化**

LoRA-Pro的更新可以看作是带约束的优化：

$$\min_{A, B} \mathcal{L}(A, B) \quad \text{s.t.} \quad H_A B + A H_B \approx G$$

使用拉格朗日乘数法：

$$\mathcal{L}_{\text{aug}}(A, B, \lambda) = \mathcal{L}(A, B) + \lambda \|H_A B + A H_B - G\|_F^2$$

**注释**：梯度对齐约束相当于引入了一个软约束正则化项。

**推导11.2：与权重衰减的关系**

标准的权重衰减：

$$\mathcal{L}_{\text{reg}}(A, B) = \mathcal{L}(A, B) + \frac{\lambda}{2} (\|A\|_F^2 + \|B\|_F^2)$$

LoRA-Pro的隐式正则化更复杂，它不仅惩罚权重的范数，还惩罚梯度逼近误差。

**关系**：两者可以结合使用，获得更强的正则化效果。

**注释**：隐式正则化往往比显式正则化更有效，因为它自适应于优化过程。

**推导11.3：路径长度正则化**

定义优化路径长度：

$$L_{\text{path}} = \sum_{t=0}^{T-1} \|\Delta W_t\|_F = \sum_{t=0}^{T-1} \|A_{t+1} B_{t+1} - A_t B_t\|_F$$

LoRA-Pro通过更精确的梯度方向，可能减少了路径长度：

$$L_{\text{path}}^{\text{LoRA-Pro}} \leq L_{\text{path}}^{\text{LoRA}}$$

更短的路径通常对应更好的泛化。

**注释**：这是基于"flat minima"理论——平坦的最优点通常泛化更好，而更直接的路径更容易找到平坦的最优点。

### 12. 实验结果的理论解释

**推导12.1：超越全量微调的可能性**

实验显示LoRA-Pro在GLUE上超过了全量微调，这似乎违反直觉。理论解释：

1. **正则化效应**：LoRA-Pro的秩约束提供了强正则化，在小数据集上减少过拟合
2. **优化路径**：LoRA-Pro的更新路径可能避开了全量微调陷入的局部最优
3. **隐式偏置**：梯度对齐约束可能引入了有益的归纳偏置

**注释**：在深度学习中，约束并非总是坏事，适当的约束可以改善泛化。

**推导12.2：数据效率的提升**

定义样本复杂度：

$$N_{\epsilon}(\mathcal{F}) = \min\{N : \mathbb{E}[\mathcal{L}_{\text{test}}] - \mathcal{L}^* \leq \epsilon\}$$

LoRA-Pro的样本复杂度可能更低：

$$N_{\epsilon}(\mathcal{F}_{\text{LoRA-Pro}}) \leq N_{\epsilon}(\mathcal{F}_{\text{LoRA}})$$

这意味着达到相同性能需要更少的训练样本。

**注释**：这在数据稀缺的场景下尤为重要，如小语种NLP、专业领域任务等。

**推导12.3：与LoRA-GA的协同效应**

LoRA-GA + LoRA-Pro的组合效果：

$$\text{Performance}_{\text{GA+Pro}} > \text{Performance}_{\text{GA}} + \text{Performance}_{\text{Pro}} - \text{Baseline}$$

即存在超线性的增益。

理论解释：
- LoRA-GA提供好的初始子空间
- LoRA-Pro确保在该子空间中沿最优方向移动
- 两者相辅相成

**注释**：这个协同效应在实验中得到了验证，两者结合的效果最好。

### 13. 扩展与变体

**推导13.1：多秩LoRA-Pro**

考虑不同层使用不同的秩 $r_l$：

$$H_{A,t}^{(l)} = G_t^{(l)} (B_t^{(l)})^{\top} [(B_t^{(l)} (B_t^{(l)})^{\top}]^{-1}$$

每层独立优化，但可以共享Adam的全局状态（如学习率调度）。

**注释**：这允许模型根据每层的重要性自适应调整参数量。

**推导13.2：稀疏LoRA-Pro**

引入稀疏性约束：

$$\min_{H_A, H_B} \|H_A B + A H_B - G\|_F^2 + \lambda (\|H_A\|_1 + \|H_B\|_1)$$

使用近端梯度法求解：

$$H_A^{(k+1)} = \text{prox}_{\lambda \|\cdot\|_1} (H_A^{(k)} - \alpha \nabla_{H_A} \mathcal{L})$$

**注释**：稀疏性可以进一步减少计算量和显存占用。

**推导13.3：分块LoRA-Pro**

将大矩阵分块处理：

$$A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}, \quad B = [B_1, B_2]$$

每块独立计算 $H_A, H_B$，减少计算复杂度：

$$\mathcal{O}(nmr) \to \mathcal{O}(\frac{nm}{k^2} r)$$

其中 $k$ 是分块数。

**注释**：这在超大模型中尤为有用，可以进一步降低显存峰值。

### 总结

本节详细推导了LoRA-Pro的数学框架，包括：

1. **逐步对齐理论**：从单步对齐（LoRA-GA）扩展到逐步对齐（LoRA-Pro）
2. **最小二乘求解**：推导了 $H_A, H_B$ 的解析解，并分析了解的不唯一性
3. **对称化策略**：通过优化自由参数 $C$ 实现 $A, B$ 的对称处理
4. **梯度匹配度量**：量化了LoRA-Pro对全量微调梯度的逼近程度
5. **Adam集成**：详细推导了如何将LoRA-Pro与Adam优化器结合
6. **显存分析**：分析了LoRA-Pro的显存开销，约为全量微调的2/3
7. **优化景观改善**：解释了为什么LoRA-Pro收敛更快、效果更好
8. **泛化理论**：从Rademacher复杂度角度分析了泛化能力
9. **方法对比**：理论对比了LoRA-Pro与其他LoRA变体的异同

LoRA-Pro的核心洞察是：**通过修改优化器来持续对齐全量微调的梯度方向，从而在整个训练过程中保持与全量微调的一致性**。这是一个比初始化更深层次的改进，代表了LoRA优化的新范式。
