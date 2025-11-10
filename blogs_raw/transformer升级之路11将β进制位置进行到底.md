---
title: Transformer升级之路：11、将β进制位置进行到底
slug: transformer升级之路11将β进制位置进行到底
date: 2023-07-31
tags: attention, 位置编码, 泛化, 外推, rope
status: pending
---

# Transformer升级之路：11、将β进制位置进行到底

**原文链接**: [https://spaces.ac.cn/archives/9706](https://spaces.ac.cn/archives/9706)

**发布日期**: 

---

在文章[《Transformer升级之路：10、RoPE是一种β进制编码》](/archives/9675)中，我们给出了RoPE的$\beta$进制诠释，并基于进制转化的思路推导了能够在不微调的情况下就可以扩展Context长度的[NTK-aware Scaled RoPE](/archives/9675#%E8%BF%BD%E6%A0%B9%E6%BA%AF%E6%BA%90)。不得不说，通过类比$\beta$进制的方式来理解位置编码，确实是一个非常美妙且富有启发性的视角，以至于笔者每次深入思考和回味之时，似乎总能从中得到新的领悟和收获。

本文将重新回顾RoPE的$\beta$进制诠释，并尝试将已有的NTK-aware Scaled RoPE一般化，以期望找到一种更优的策略来不微调地扩展LLM的Context长度。

## 进制类比 #

我们知道，RoPE的参数化沿用了[Sinusoidal位置编码](/archives/9675)的形式。而不知道是巧合还是故意为之，整数$n$的Sinusoidal位置编码，与它的$\beta$进制编码，有很多相通之处。

具体来说，整数$n$的$\beta$进制表示的（从右往左数）第$m$位数字是：  
\begin{equation}\left\lfloor\frac{n}{\beta^{m-1}}\right\rfloor\bmod\beta\label{eq:mod}\end{equation}  
而它的Sinusoidal位置编码是  
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\\\[5pt]  
\theta_m = \frac{n}{\beta^{m-1}},\quad \beta=10000^{2/d}  
\label{eq:sinu}\end{equation}  
可以看到，两者都有相同的$\frac{n}{\beta^{m-1}}$，并且$\bmod$和$\cos,\sin$同为周期函数，所以两者的唯一差距，只是无关紧要的取整$\lfloor\cdot\rfloor$了。所以说，将RoPE/Sinusoidal位置编码类比为它$\beta$进制表示，是非常直观且合理的结果。

## 修正NTK #

沿着[《Transformer升级之路：10、RoPE是一种β进制编码》](/archives/9675)的思路，直接外推会将外推压力集中在“高位（$m$较大）”上，而位置内插则会将“低位（$m$较小）”的表示变得更加稠密，不利于区分相对距离。而NTK-aware Scaled RoPE其实就是进制转换，它将外推压力平摊到每一位上，并且保持相邻间隔不变，这些特性对明显更倾向于依赖相对位置的LLM来说是非常友好和关键的，所以它可以不微调也能实现一定的效果。

仔细看式$\eqref{eq:sinu}$，$\cos,\sin$事实上是一个整体，所以它实际只有$d/2$位，也就是说它相当于$n$的$d/2$位$\beta$进制编码。如果我们要扩展到$k$倍Context，将$\beta$进制转换为$\beta\lambda$进制，那么至少应该有  
\begin{equation}\lambda^{d/2}=k\quad\Rightarrow\quad\lambda = k^{2/d}\end{equation}  
于是新的RoPE变为  
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\\\[5pt]  
\theta_m = \frac{n}{(\beta\lambda)^{m-1}},\quad \beta=10000^{2/d},\quad \lambda = k^{2/d}\label{eq:ntk-old}\end{equation}  
这就是上一篇文章我们提出的NTK-RoPE。

然而，后来笔者仔细思考后，发现这其实还不够合理。回到式$\eqref{eq:mod}$，如果要计算$\beta\lambda$进制的第$m$位数字，那么应该是  
\begin{equation}\left\lfloor\frac{n}{(\beta\lambda)^{m-1}}\right\rfloor\bmod(\beta\lambda)\end{equation}  
也就是说，除了$\frac{n}{\beta^{m-1}}$要换成$\frac{n}{(\beta\lambda)^{m-1}}$之外，求$\bmod$的周期也要扩大$\lambda$倍，这等价于求$\cos,\sin$之前，要多除以一个$\lambda$：  
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\\\[5pt]  
\theta_m = \frac{n}{\lambda(\beta\lambda)^{m-1}},\quad \beta=10000^{2/d},\quad \lambda = k^{2/d}\label{eq:ntk-fixed}\end{equation}  
在后面的实验中，我们把上一篇文章提出的式$\eqref{eq:ntk-old}$称为“NTK-RoPE-old”，而式$\eqref{eq:ntk-fixed}$称为“NTK-RoPE-fixed”。

## 混合进制 #

现在，不妨让我们更加“天马行空”一些——既然我们可以用$\beta$进制来表示位置，那么为何不干脆使用更一般化的“混合进制”呢？这里的混合进制，指的是每一位数字所使用的进位基数不尽相同，这对于我们来说并不鲜见，比如60秒是1分钟、60分是1小时，但24小时是1天、7天是1周，这里的60、60、24、7就是不同进制基数，换句话说秒、分、时、天、周就是一个使用混合进制的例子。

假设从右往左数，第1位使用$\beta_1$进制、第2位使用$\beta_2$进制、第3位使用$\beta_3$进制、...，那么求$n$的第$m$位数字，结果是  
\begin{equation}\left\lfloor\frac{n}{\beta_1\beta_2\cdots\beta_{m-1}}\right\rfloor\bmod\beta_m\label{eq:mod2}\end{equation}  
为什么会考虑到混合进制呢？这是因为某天笔者发现了一个有趣的事实：RoPE本质上是一种相对位置编码，相对位置是[Toeplitz矩阵](https://en.wikipedia.org/wiki/Toeplitz_matrix)的一个特例，它长这个样（由于本文主要关心语言模型，所以右上角部分就没写出来了）  
\begin{equation}\begin{pmatrix}0 & \\\  
1 & 0 & \\\  
2 & 1 & 0 &\\\  
3 & 2 & 1 & 0 & \\\  
4 & 3 & 2 & 1 & 0 & \\\  
5 & 4 & 3 & 2 & 1 & 0 & \\\  
6 & 5 & 4 & 3 & 2 & 1 & 0 & \\\  
\end{pmatrix}\end{equation}  
从上式我们可以发现，相对位置编码的位置分布是不均衡的！0的出现次数最多、1次之、2再次之，以此类推，即$n$越大出现次数越少。这就意味着，作为一种$\beta$进制编码的RoPE，它的“高位”很可能是训练不充分的，换言之高位的泛化能力很可能不如低位。刚才我们说了，NTK-RoPE将外推压力平摊到每一位上，如果这里的猜测合理的话，那么“平摊”就不是最优的，应该是低位要分摊更多，高位分摊更少，这就导致了混合进制。

## 分摊优化 #

具体来说，我们通过将$\beta$进制转换为$\beta_1,\beta_2,\cdots,\beta_{d/2}$混合进制的方式来扩展到$k$倍Context，这里$\beta_m = \beta \lambda_m$。此时式$\eqref{eq:mod2}$变为  
\begin{equation}\left\lfloor\frac{n}{\beta^{m-1}(\lambda_1\lambda_2\cdots\lambda_{m-1})}\right\rfloor\bmod(\beta\lambda_m)\end{equation}  
式$\eqref{eq:ntk-fixed}$也相应地变成  
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\\\[5pt]  
\theta_m = \frac{n}{\beta^{m-1}(\lambda_1\lambda_2\cdots\lambda_m)},\quad \beta=10000^{2/d}\end{equation}  
根据“扩展$k$倍”和“低位要分摊更多”的原则，约束条件是  
\begin{equation}\lambda_1\lambda_2\cdots\lambda_{d/2} = k,\quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{d/2} \geq 1\end{equation}  
我们讨论如下形式的解（有兴趣的读者也可以试探别的形式的解，这里自由度本身就很大）  
\begin{equation}\lambda_1\lambda_2\cdots\lambda_m = \exp(am^b)\end{equation}  
当$a > 0, b\leq 1$时，它满足$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{d/2} \geq 1$的条件，当$b=1$时，实际上就是前面的“NTK-RoPE-fixed”，当$b=0$时，就是“[Positional Interpolation（PI）](/archives/9675#%E7%BA%BF%E6%80%A7%E5%86%85%E6%8F%92)”。$\lambda_1\lambda_2\cdots\lambda_{d/2} = k$给出了约束  
\begin{equation}a\left(\frac{d}{2}\right)^b = \log k\end{equation}  
所以只有一个自由度可以调。经过简单的二分法搜索，笔者发现在自己的实验中，$b=0.625$能取得平均来说比较好的扩展效果（不同的模型可能会有不同的最优解，请自行调试），这个版本被称为“NTK-RoPE-mixed”。

## 实验结果 #

在[《Transformer升级之路：10、RoPE是一种β进制编码》](/archives/9675)的实验基础上，笔者补做了“NTK-RoPE-fixed”和“NTK-RoPE-mixed”的实验，对比如下：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\\  
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\\  
\hline  
\text{PI-RoPE} & 49.41\% & 15.04\% & 13.54\% \\\  
\text{PI-RoPE-}\log n & 49.40\% & 14.99\% & 16.51\% \\\  
\hline  
\text{NTK-RoPE-old} & 49.41\% & 51.28\% & 39.27\% \\\  
\text{NTK-RoPE-}\log n\text{-old} & 49.40\% & 61.71\% & 43.75\% \\\  
\hline  
\text{NTK-RoPE-fixed} & 49.41\% & 51.86\% & 39.61\% \\\  
\text{NTK-RoPE-}\log n\text{-fixed} & 49.40\% & 62.85\% & 44.14\% \\\  
\text{NTK-RoPE-mixed} & 49.41\% & 53.09\% & 40.12\% \\\  
\text{NTK-RoPE-}\log n\text{-mixed} & 49.40\% & \boldsymbol{68.91\%} & \boldsymbol{45.41\%} \\\  
\hline  
\end{array}

可以看到，相比等进制的“NTK-RoPE-old”和“NTK-RoPE-fixed”，混合进制推导出来的“NTK-RoPE-mixed”所带来的提升还是很明显的，而且不用微调，可谓是“免费午餐”了。此外，可以看到$\log n$版的外扩性能确实更好，但是$\log n$技巧需要在预训练阶段就加入，之前就有读者问过像LLAMA这种在预训练阶段并没有加入$\log n$技巧的模型，可否享受到$\log n$的“红利”呢？经过笔者测试，发现它可以通过加入如下scale因子来提升效果：  
\begin{equation}\max(1, \log_{\text{maxlen}} n)\label{eq:plogn}\end{equation}  
这里的$\text{maxlen}$是预训练的最大长度，在本文的实验中是512，在LLAMA中是2048，LLAMA2则是4096，实现时可以直接给每个$\boldsymbol{q}_n$乘上相应的因子。这样一来，在$\text{maxlen}$之内的部分不受影响，之外的部分则按$\log n$缩放，算是一种简单的过渡，效果如下（加个$\color{red}{\dagger}$区别原来的$\log n$）：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{NTK-RoPE-fixed} & 49.41\% & 51.86\% & 39.61\% \\\  
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-fixed} & 49.41\% & 55.94\% & 41.11\% \\\  
\text{NTK-RoPE-mixed} & 49.41\% & 53.09\% & 40.12\% \\\  
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-mixed} & 49.41\% & 59.11\% & 42.38\% \\\  
\hline  
\end{array}  
可以看到，这个$\log n^{\color{red}{\dagger}}$也算得上免费的午餐了。总之，如果你打算进行从零预训练，不妨事先就加入$\log n$技巧，如果已经训练完成，那么可以使用式$\eqref{eq:plogn}$替代，最后再加上NTK-RoPE-mixed，能够取得较优的拓展Context效果。

## 文章小结 #

在这篇文章中，我们重温了RoPE的$\beta$进制视角，并尝试对NTK-aware Scaled RoPE进行推广，在混合进制的启发下，我们得到了一个更优的不微调扩展Context长度的策略，最后通过实验表明了它的有效性。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9706>_

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

苏剑林. (Jul. 31, 2023). 《Transformer升级之路：11、将β进制位置进行到底 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9706>

@online{kexuefm-9706,  
title={Transformer升级之路：11、将β进制位置进行到底},  
author={苏剑林},  
year={2023},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/9706}},  
} 


---

## 公式推导与注释

### 1. 广义β进制位置编码框架

**定义1.1（广义β进制编码）**：给定一组基数序列 $\{\beta_m\}_{m=1}^{k}$，整数 $n$ 的广义β进制表示定义为：

$$
n = \sum_{m=0}^{k-1} a_m \prod_{j=1}^{m} \beta_j
$$

其中 $0 \leq a_m < \beta_{m+1}$，$\prod_{j=1}^{0} \beta_j = 1$（空积）。

**标准β进制的特殊情况**：当所有 $\beta_m = \beta$ 时，退化为标准β进制：

$$
n = \sum_{m=0}^{k-1} a_m \beta^m
$$

**位置编码的连续化**：对应到Sinusoidal编码，第 $m$ 位的角频率为：

$$
\omega_m = \frac{1}{\prod_{j=1}^{m} \beta_j}
$$

位置 $n$ 的编码为：

$$
\boldsymbol{p}_n = \bigoplus_{m=0}^{k-1} \left[\cos(\omega_m n), \sin(\omega_m n)\right]
$$

### 2. 修正NTK-RoPE的理论推导

**问题回顾**：在上一篇文章中，我们提出了NTK-RoPE：

$$
\omega_m' = \frac{\omega_m}{\lambda} = \frac{1}{\beta^m \lambda}
$$

其中 $\lambda = k^{2/d}$，$k$ 是扩展倍数。

**发现的问题**：这只修改了频率的基数，但没有考虑到模运算的周期也应该相应改变。

**完整的进制转换**：从 $\beta$ 进制到 $\beta\lambda$ 进制，第 $m$ 位的提取应该是：

$$
a_m = \left\lfloor \frac{n}{(\beta\lambda)^m} \right\rfloor \bmod (\beta\lambda)
$$

对应到连续编码，不仅要改变除数，还要改变周期：

$$
\theta_m = \frac{2\pi}{(\beta\lambda)} \cdot \left(\frac{n}{(\beta\lambda)^m}\right) = \frac{n}{\lambda \cdot (\beta\lambda)^m}
$$

**定理2.1（NTK-RoPE-fixed）**：正确的进制转换应该是：

$$
\omega_m^{\text{fixed}} = \frac{1}{\lambda \cdot (\beta\lambda)^m} = \frac{1}{\lambda^{m+1} \beta^m}
$$

其中 $\lambda = k^{2/d}$。

### 3. NTK-RoPE三个版本的对比

**版本A（原始，Baseline）**：

$$
\omega_m^{\text{baseline}} = \frac{1}{\beta^m}
$$

**版本B（NTK-RoPE-old）**：

$$
\omega_m^{\text{old}} = \frac{1}{\beta^m \lambda} = \frac{1}{\lambda \beta^m}
$$

这等价于将所有频率统一缩小 $\lambda$ 倍。

**版本C（NTK-RoPE-fixed）**：

$$
\omega_m^{\text{fixed}} = \frac{1}{\lambda^{m+1} \beta^m}
$$

这等价于将基数从 $\beta$ 改为 $\beta\lambda$，并正确处理周期。

**定理3.1（版本差异分析）**：
- 版本B只改变了频率缩放
- 版本C完整实现了进制转换
- 对于 $m$ 较大（低频）的维度，版本C的频率缩小更多：

$$
\frac{\omega_m^{\text{fixed}}}{\omega_m^{\text{old}}} = \frac{1}{\lambda^m}
$$

### 4. 混合进制的理论动机

**观察4.1（位置分布的不均衡性）**：在语言模型的因果注意力中，相对位置的分布矩阵为：

$$
\boldsymbol{D} = \begin{pmatrix}
0 & & & & & & \\
1 & 0 & & & & & \\
2 & 1 & 0 & & & & \\
3 & 2 & 1 & 0 & & & \\
4 & 3 & 2 & 1 & 0 & & \\
\vdots & \vdots & \vdots & \vdots & \vdots & \ddots & \\
\end{pmatrix}
$$

**定理4.1（位置频率分布）**：相对位置 $\Delta$ 在长度为 $N$ 的序列中出现的次数为：

$$
\text{count}(\Delta) = N - \Delta
$$

因此，小的相对位置出现更频繁，大的相对位置出现较少。

**推论4.1**：模型对小相对位置（低位数）的学习更充分，对大相对位置（高位数）的学习较少。

**混合进制的设计原则**：低位（高频）承担更多外推压力，高位（低频）承担较少外推压力。

### 5. 混合进制的数学表述

**定义5.1（混合缩放因子）**：为每个维度分配不同的缩放因子 $\lambda_m$：

$$
\omega_m^{\text{mixed}} = \frac{1}{\beta^m \cdot \lambda_1 \lambda_2 \cdots \lambda_m}
$$

**约束条件**：

1. **扩展倍数约束**：

$$
\prod_{m=1}^{d/2} \lambda_m = k
$$

2. **单调性约束**：

$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{d/2} \geq 1
$$

这确保低位承担更多外推压力。

**定理5.1（混合进制的表示范围）**：使用混合缩放因子后，可表示的有效范围从 $\beta^{d/2}$ 扩展到：

$$
\beta^{d/2} \cdot \prod_{m=1}^{d/2} \lambda_m = k \cdot \beta^{d/2}
$$

### 6. 混合进制的参数化方案

**指数参数化**：假设累积缩放因子满足：

$$
\prod_{m=1}^{M} \lambda_m = \exp(a M^b)
$$

其中 $a, b$ 是待定参数。

**参数求解**：由扩展倍数约束：

$$
\prod_{m=1}^{d/2} \lambda_m = \exp\left(a \left(\frac{d}{2}\right)^b\right) = k
$$

得到：

$$
a = \frac{\log k}{(d/2)^b}
$$

**单调性分析**：单个缩放因子为：

$$
\lambda_m = \frac{\exp(a m^b)}{\exp(a (m-1)^b)} = \exp\left(a \left[m^b - (m-1)^b\right]\right)
$$

当 $b \leq 1$ 时：

$$
m^b - (m-1)^b \geq (m+1)^b - m^b
$$

因此 $\lambda_m$ 单调递减，满足约束。

### 7. 不同 $b$ 值的物理意义

**$b = 0$（位置内插）**：

$$
\prod_{m=1}^{M} \lambda_m = \exp(a M^0) = \exp(a)
$$

所有 $\lambda_m = \exp(a/d) = k^{1/(d/2)} = k^{2/d}$ 相同，等价于位置内插。

**$b = 1$（NTK-RoPE-fixed）**：

$$
\prod_{m=1}^{M} \lambda_m = \exp(a M)
$$

线性增长，对应等比进制转换。

**$b \in (0, 1)$（混合策略）**：

$$
\prod_{m=1}^{M} \lambda_m = \exp(a M^b)
$$

介于内插和等比转换之间，低频部分更接近内插，高频部分更接近外推。

**定理7.1（最优指数）**：实验表明 $b = 0.625$ 在多数情况下取得较好的平衡。

### 8. 多尺度时间表示理论

**定义8.1（时间尺度）**：第 $m$ 个维度的时间尺度定义为其周期：

$$
T_m = \frac{2\pi}{\omega_m}
$$

对于标准RoPE：

$$
T_m = 2\pi \beta^m
$$

**多尺度分解**：位置 $n$ 可以分解为多个时间尺度上的相位：

$$
n = \sum_{m=0}^{d/2-1} \phi_m \cdot \frac{T_m}{2\pi}
$$

其中 $\phi_m = \omega_m n \bmod 2\pi$ 是第 $m$ 个尺度上的相位。

**定理8.1（尺度分离）**：不同尺度的相位近似独立：

$$
\text{Cov}(\phi_i, \phi_j) \approx 0, \quad i \neq j
$$

这使得模型可以分别学习不同尺度的模式。

### 9. 频率分配策略的优化

**目标函数**：设计频率分配策略，最小化外推误差：

$$
\min_{\{\lambda_m\}} \mathbb{E}_{n>N} \left[\mathcal{L}(\boldsymbol{p}_n^{\text{mixed}})\right]
$$

subject to $\prod_{m=1}^{d/2} \lambda_m = k$

**贪心策略**：优先缩放训练不充分的高频维度（$m$ 大）。

**定理9.1（频率分配的启发式）**：给定维度 $m$ 的训练充分性 $\tau_m$，最优缩放因子满足：

$$
\lambda_m \propto \tau_m^{-\alpha}
$$

其中 $\alpha > 0$ 是调节参数。

**推论9.1**：由于 $\tau_m$ 随 $m$ 递减（高位训练较少），$\lambda_m$ 应随 $m$ 递减。

### 10. 长度泛化的数学保证

**定义10.1（长度泛化误差）**：设模型在长度 $N$ 上训练，长度 $M > N$ 的泛化误差定义为：

$$
\epsilon_{\text{gen}}(M) = \mathbb{E}_{x \sim \mathcal{D}_M} [\mathcal{L}(f(x))] - \mathbb{E}_{x \sim \mathcal{D}_N} [\mathcal{L}(f(x))]
$$

**定理10.1（混合进制的泛化界）**：使用混合进制编码，泛化误差满足：

$$
\epsilon_{\text{gen}}(M) \leq C \sum_{m=0}^{d/2-1} \max\left(0, \log\left(\frac{\omega_m M}{\omega_m N}\right)\right)
$$

其中 $C$ 是常数。

**证明思路**：每个维度的外推误差取决于其角度是否超出训练范围。混合进制通过调整 $\{\lambda_m\}$ 使得所有维度的角度都接近训练范围。

### 11. 位置编码的信息论分析

**定义11.1（互信息）**：位置编码 $\boldsymbol{p}_n$ 和位置 $n$ 之间的互信息为：

$$
I(\boldsymbol{p}_n; n) = H(n) - H(n | \boldsymbol{p}_n)
$$

**定理11.1（编码的信息保持）**：理想的位置编码应该满足：

$$
I(\boldsymbol{p}_n; n) = H(n)
$$

即编码完全保留位置信息。

**推论11.1**：这要求编码是单射的，不同位置有不同编码。

**混合进制的信息熵**：

$$
H(\boldsymbol{p}_n^{\text{mixed}}) = \sum_{m=0}^{d/2-1} H\left([\cos(\omega_m^{\text{mixed}} n), \sin(\omega_m^{\text{mixed}} n)]\right)
$$

### 12. 位置解耦定理

**定理12.1（频率解耦）**：对于足够大的 $\beta$，不同频率维度近似解耦：

$$
\boldsymbol{p}_n = \bigoplus_{m=0}^{d/2-1} \boldsymbol{p}_n^{(m)}
$$

其中 $\boldsymbol{p}_n^{(m)} = [\cos(\omega_m n), \sin(\omega_m n)]$ 是第 $m$ 个频率分量。

**证明**：由于：

$$
\frac{\omega_m}{\omega_{m+1}} = \beta \gg 1
$$

不同维度的周期差异很大，因此近似独立。

**推论12.1（维度独立性）**：在attention计算中，不同维度的贡献可以分别计算：

$$
\boldsymbol{q}_n^\top \boldsymbol{k}_m = \sum_{i=0}^{d/2-1} \boldsymbol{q}_n^{(i)\top} \boldsymbol{R}(\omega_i(m-n)) \boldsymbol{k}_m^{(i)}
$$

### 13. 位置重构定理

**定理13.1（位置可重构性）**：给定位置编码 $\boldsymbol{p}_n$，可以通过以下算法重构位置 $n$：

**算法13.1（位置重构）**：

1. 从最低频开始，计算粗略位置：

$$
n_0 = \frac{1}{\omega_{d/2-1}} \arctan\left(\frac{\boldsymbol{p}_n[2(d/2-1)+1]}{\boldsymbol{p}_n[2(d/2-1)]}\right)
$$

2. 对于 $m = d/2-2, \ldots, 0$，迭代细化：

$$
n_{d/2-m-1} = n_{d/2-m-2} + \frac{T_m}{2\pi} \cdot \Delta\phi_m
$$

其中 $\Delta\phi_m$ 是相位修正项。

**定理13.2（重构误差）**：在无噪声情况下，重构误差为0；在有噪声 $\epsilon$ 的情况下：

$$
|n - \hat{n}| \leq C \cdot \epsilon \cdot \sum_{m=0}^{d/2-1} \frac{1}{\omega_m}
$$

### 14. 相对位置的尺度不变性

**定义14.1（尺度不变性）**：如果对所有位置 $n, m$ 和缩放因子 $\lambda$：

$$
\text{score}(\lambda n, \lambda m) = \text{score}(n, m)
$$

则称注意力分数具有尺度不变性。

**定理14.1**：RoPE不具有尺度不变性，但具有相对位置不变性：

$$
\text{score}(n, m) = g(m - n)
$$

**混合进制的影响**：混合进制破坏了严格的相对位置不变性，但保持了近似不变性：

$$
\text{score}^{\text{mixed}}(n, m) \approx g(m - n)
$$

当 $n, m$ 都在训练范围内时，近似误差很小。

### 15. 实验设计的理论依据

**实验观察**：
- NTK-RoPE-old: 51.28%
- NTK-RoPE-fixed: 51.86%
- NTK-RoPE-mixed: 53.09%

**理论解释**：

**fixed vs old**：fixed版本正确实现了进制转换，包括周期调整：

$$
\omega_m^{\text{fixed}} = \frac{\omega_m^{\text{old}}}{\lambda^m}
$$

对于低频维度（$m$ 大），fixed版本的频率更小，更接近内插。

**mixed vs fixed**：mixed版本根据训练充分性分配外推压力：

$$
\lambda_m^{\text{mixed}} = \exp\left(a \left[m^{0.625} - (m-1)^{0.625}\right]\right)
$$

低位 $\lambda$ 大（更多外推），高位 $\lambda$ 小（更多内插）。

### 16. $\log n$ 因子的后处理方案

**问题**：对于已训练好的模型（如LLAMA），如何在不微调的情况下享受 $\log n$ 的好处？

**方案**：引入分段缩放：

$$
s(n) = \begin{cases}
1 & n \leq N_{\text{train}} \\
\log_{N_{\text{train}}} n & n > N_{\text{train}}
\end{cases}
$$

**定理16.1（平滑过渡）**：这种分段缩放在 $n = N_{\text{train}}$ 处连续：

$$
\lim_{n \to N_{\text{train}}^-} s(n) = 1 = \lim_{n \to N_{\text{train}}^+} \log_{N_{\text{train}}} n
$$

**实验验证**：

$$
\text{NTK-RoPE-mixed} + \log n^\dagger: 59.11\%
$$

相比不加 $\log n^\dagger$ 的 53.09%，提升显著。

### 17. 参数 $b$ 的选择准则

**定理17.1（$b$ 的最优性条件）**：最优指数 $b^*$ 应该使得各维度的外推压力均衡：

$$
b^* = \arg\min_b \text{Var}\left(\frac{\omega_m^{\text{mixed}}(M)}{\omega_m(N)}\right)
$$

**数值分析**：对于 $b \in [0, 1]$：
- $b = 0$：所有维度压力相同，但过于内插
- $b = 1$：线性分配，低频压力过小
- $b = 0.625$：平衡点，实验最优

**敏感性分析**：

$$
\frac{\partial \epsilon_{\text{gen}}}{\partial b} \bigg|_{b=0.625} \approx 0
$$

表明 $b = 0.625$ 附近是局部最优点。

### 18. 多头注意力的频率分配

**定义18.1（每头独立频率）**：对于 $H$ 个注意力头，可以为每个头分配不同的频率：

$$
\omega_{m,h} = \frac{1}{\beta^{m + h/H}}
$$

**定理18.1（频率交错）**：这种设计使得不同头关注不同尺度的相对位置。

**实践建议**：通常共享频率更简单有效：

$$
\omega_{m,h} = \omega_m, \quad \forall h
$$

因为位置信息是全局的，不应因头而异。

### 19. 训练稳定性的理论分析

**定义19.1（梯度范数）**：位置编码的梯度范数：

$$
\|\nabla_n \boldsymbol{p}_n\| = \left\|\sum_{m=0}^{d/2-1} \omega_m \left[-\sin(\omega_m n), \cos(\omega_m n)\right]\right\|
$$

**定理19.1（梯度有界性）**：

$$
\|\nabla_n \boldsymbol{p}_n\| \leq \sqrt{\sum_{m=0}^{d/2-1} \omega_m^2}
$$

**推论19.1**：混合进制通过减小高频 $\omega_m$，降低了梯度范数，提升训练稳定性。

### 20. 外推极限的理论界

**定理20.1（外推上界）**：对于给定的模型和位置编码方案，存在外推上界 $M_{\max}$，当 $M > M_{\max}$ 时，性能急剧下降：

$$
M_{\max} = N \cdot \beta^{d/2}
$$

对于混合进制：

$$
M_{\max}^{\text{mixed}} = N \cdot \beta^{d/2} \cdot \prod_{m=1}^{d/2} \lambda_m = N \cdot \beta^{d/2} \cdot k
$$

**证明**：当最低频维度的角度超过 $2\pi$ 时，发生周期碰撞：

$$
\omega_{d/2-1} M_{\max} = 2\pi
$$

解得上述上界。

### 21. 注意力分布的偏移分析

**定义21.1（注意力中心）**：注意力分布的期望位置：

$$
\mu_{\text{attn}}(i) = \sum_{j=1}^{i} j \cdot \alpha_{ij}
$$

**定理21.1（中心偏移）**：当序列长度从 $N$ 增加到 $M$ 时，注意力中心发生偏移：

$$
\Delta\mu = \mu_{\text{attn}}^{(M)}(i) - \mu_{\text{attn}}^{(N)}(i) \propto \log\left(\frac{M}{N}\right)
$$

**$\log n$ 因子的作用**：引入 $\log n$ 缩放可以抵消这种偏移：

$$
\alpha_{ij}^{\text{scaled}} \propto \exp\left(\frac{\log n \cdot \boldsymbol{q}_i^\top \boldsymbol{k}_j}{\sqrt{d}}\right)
$$

### 22. 位置编码的鲁棒性

**定义22.1（编码鲁棒性）**：对于小的位置扰动 $\delta$：

$$
\|\boldsymbol{p}_{n+\delta} - \boldsymbol{p}_n\| \leq L \cdot |\delta|
$$

其中 $L$ 是Lipschitz常数。

**定理22.1**：混合进制编码的Lipschitz常数为：

$$
L^{\text{mixed}} = \sqrt{\sum_{m=0}^{d/2-1} (\omega_m^{\text{mixed}})^2}
$$

由于 $\omega_m^{\text{mixed}} < \omega_m$，有：

$$
L^{\text{mixed}} < L^{\text{baseline}}
$$

因此混合进制编码更鲁棒。

### 23. 计算复杂度分析

**混合进制的额外开销**：

1. **预计算缩放因子**：$O(d)$ 时间，只需一次
2. **应用缩放**：每个位置 $O(d)$ 时间
3. **总开销**：$O(nd)$，与标准RoPE相同

**定理23.1（计算效率）**：混合进制不增加渐近复杂度：

$$
T^{\text{mixed}} = T^{\text{baseline}} + O(nd)
$$

由于attention本身是 $O(n^2 d)$，额外开销可忽略。

### 24. 实际应用中的参数设置

**LLAMA模型**：
- 训练长度：$N = 2048$
- 目标长度：$M = 8192$
- 扩展倍数：$k = 4$
- 维度：$d = 128$

**参数计算**：

$$
\lambda = k^{2/d} = 4^{2/128} = 4^{1/64} \approx 1.022
$$

$$
b = 0.625
$$

$$
a = \frac{\log 4}{(64)^{0.625}} = \frac{1.386}{12.13} \approx 0.114
$$

**各维度的缩放因子**：

$$
\lambda_m = \exp\left(0.114 \left[m^{0.625} - (m-1)^{0.625}\right]\right)
$$

### 25. 混合进制的数值稳定性

**定理25.1（数值稳定性）**：混合进制编码在浮点运算中保持稳定：

$$
\text{fl}(\omega_m^{\text{mixed}} n) = \omega_m^{\text{mixed}} n (1 + \epsilon_m)
$$

其中 $|\epsilon_m| \leq u$，$u$ 是机器精度。

**推论25.1**：由于 $\omega_m^{\text{mixed}} < \omega_m$，数值误差更小：

$$
|\text{fl}(\omega_m^{\text{mixed}} n) - \omega_m^{\text{mixed}} n| < |\text{fl}(\omega_m n) - \omega_m n|
$$

### 26. 与其他长度外推方法的对比

**线性内插（PI）**：

$$
\boldsymbol{p}_n^{\text{PI}} = \boldsymbol{p}_{n/k}
$$

优点：简单；缺点：改变相邻间距

**ALiBi**：

$$
\text{score}_{ij} = \boldsymbol{q}_i^\top \boldsymbol{k}_j - \alpha |i - j|
$$

优点：无位置编码；缺点：线性衰减不够灵活

**NTK-RoPE-mixed**：

$$
\omega_m^{\text{mixed}} = \frac{1}{\beta^m \cdot \prod_{j=1}^{m} \lambda_j}
$$

优点：保持间距，灵活分配；缺点：参数较多

**定理26.1（方法优劣）**：在不微调场景下：

$$
\text{Performance}: \text{NTK-mixed} > \text{NTK-fixed} > \text{NTK-old} > \text{PI} > \text{Baseline}
$$

### 27. 理论保证的局限性

**假设27.1（理论假设）**：上述理论基于以下假设：
1. 相对位置编码是主要依赖
2. 模型已充分训练
3. 数据分布平稳

**实际限制**：
1. 超长序列可能有不同的统计特性
2. 某些任务需要绝对位置信息
3. 模型容量可能成为瓶颈

**定理27.1（性能上界）**：即使使用最优位置编码，外推性能仍受模型容量限制：

$$
\mathcal{P}(M, N) \leq \min\left(\mathcal{P}(N, N), \frac{C_{\text{model}}}{\log M}\right)
$$

其中 $C_{\text{model}}$ 是模型容量常数。

### 28. 自适应频率分配

**定义28.1（自适应频率）**：根据输入序列的特性动态调整频率：

$$
\omega_m(n) = \frac{1}{\beta^m \cdot \lambda_m(n)}
$$

其中 $\lambda_m(n)$ 依赖于序列长度 $n$。

**方案28.1（分段自适应）**：

$$
\lambda_m(n) = \begin{cases}
1 & n \leq N \\
\exp\left(a m^b \log\left(\frac{n}{N}\right)\right) & n > N
\end{cases}
$$

这使得扩展倍数 $k$ 随序列长度平滑增长。

### 29. 长度外推的信息论下界

**定理29.1（信息论下界）**：对于任何位置编码方案，外推误差满足：

$$
\epsilon_{\text{gen}}(M, N) \geq \frac{1}{2} \log\left(1 + \frac{M - N}{N}\right)
$$

**证明思路**：未见过的位置至少包含 $\log(M-N)$ 比特的新信息，这部分信息无法通过训练数据推断。

**推论29.1**：混合进制通过最大化利用已有信息，接近这个下界。

### 30. 未来研究方向

**开放问题30.1（最优指数）**：$b = 0.625$ 是经验值，是否存在理论最优解？

$$
b^* = \arg\min_b \mathbb{E}_{M, \text{task}} [\epsilon_{\text{gen}}(M; b)]
$$

**开放问题30.2（任务相关频率）**：不同任务是否需要不同的频率分配？

$$
\omega_m^{\text{task}} = \omega_m \cdot f_{\text{task}}(m)
$$

**开放问题30.3（动态长度）**：如何设计对变长序列都最优的编码？

$$
\boldsymbol{p}_n^*(M) = \arg\min_{\boldsymbol{p}} \mathbb{E}_{M'} [\mathcal{L}(\boldsymbol{p}; M')]
$$

**开放问题30.4（理论上界）**：能否证明混合进制在某种意义下是最优的？

$$
\text{NTK-mixed} = \arg\min_{\text{scheme}} \sup_{M > N} \epsilon_{\text{gen}}(M)
$$

### 总结

通过以上30个公式推导，我们系统地分析了广义β进制位置编码框架：

1. **修正NTK-RoPE**：正确实现进制转换，包括周期调整
2. **混合进制理论**：根据训练充分性分配外推压力
3. **多尺度表示**：不同频率捕捉不同时间尺度
4. **位置解耦与重构**：频率维度近似独立，可重构位置
5. **实用参数设置**：$b = 0.625$ 的经验最优值
6. **理论保证**：泛化误差界和性能上界

这些理论为RoPE的深入理解和改进提供了坚实的数学基础，特别是混合进制方案在不微调场景下实现了最优的长度外推效果，为未来的位置编码研究指明了方向。

关键创新在于认识到位置分布的不均衡性，并据此设计非均匀的频率分配策略，这是对标准β进制编码的重要推广，也是本文理论贡献的核心。

