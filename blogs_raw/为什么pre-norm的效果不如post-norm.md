---
title: 为什么Pre Norm的效果不如Post Norm？
slug: 为什么pre-norm的效果不如post-norm
date: 2022-03-29
tags: 优化, 梯度, attention, 生成模型, attention
status: pending
---

# 为什么Pre Norm的效果不如Post Norm？

**原文链接**: [https://spaces.ac.cn/archives/9009](https://spaces.ac.cn/archives/9009)

**发布日期**: 

---

Pre Norm与Post Norm之间的对比是一个“老生常谈”的话题了，本博客就多次讨论过这个问题，比如文章[《浅谈Transformer的初始化、参数化与标准化》](/archives/8620)、[《模型优化漫谈：BERT的初始标准差为什么是0.02？》](/archives/8747)等。目前比较明确的结论是：同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。Pre Norm更容易训练好理解，因为它的恒等路径更突出，但为什么它效果反而没那么好呢？

笔者之前也一直没有好的答案，直到前些时间在知乎上看到 [@唐翔昊](https://www.zhihu.com/question/519668254/answer/2371885202) 的一个回复后才“恍然大悟”，原来这个问题竟然有一个非常直观的理解！本文让我们一起来学习一下。

## 基本结论 #

Pre Norm和Post Norm的式子分别如下：  
\begin{align}  
\text{Pre Norm: } \quad \boldsymbol{x}_{t+1} = \boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t))\\\  
\text{Post Norm: }\quad \boldsymbol{x}_{t+1} = \text{Norm}(\boldsymbol{x}_t + F_t(\boldsymbol{x}_t))  
\end{align}  
在Transformer中，这里的$\text{Norm}$主要指Layer Normalization，但在一般的模型中，它也可以是Batch Normalization、Instance Normalization等，相关结论本质上是通用的。

在笔者找到的资料中，显示Post Norm优于Pre Norm的工作有两篇，一篇是[《Understanding the Difficulty of Training Transformers》](https://papers.cool/arxiv/2004.08249)，一篇是[《RealFormer: Transformer Likes Residual Attention》](https://papers.cool/arxiv/2012.11747)。另外，笔者自己也做过对比实验，显示Post Norm的结构迁移性能更加好，也就是说在Pretraining中，Pre Norm和Post Norm都能做到大致相同的结果，但是Post Norm的Finetune效果明显更好。

可能读者会反问[《On Layer Normalization in the Transformer Architecture》](https://papers.cool/arxiv/2002.04745)不是显示Pre Norm要好于Post Norm吗？这是不是矛盾了？其实这篇文章比较的是在完全相同的训练设置下Pre Norm的效果要优于Post Norm，这只能显示出Pre Norm更容易训练，因为Post Norm要达到自己的最优效果，不能用跟Pre Norm一样的训练配置（比如Pre Norm可以不加Warmup但Post Norm通常要加），所以结论并不矛盾。

## 直观理解 #

为什么Pre Norm的效果不如Post Norm？知乎上 [@唐翔昊](https://www.zhihu.com/question/519668254/answer/2371885202) 给出的答案是：Pre Norm的深度有“水分”！也就是说，一个$L$层的Pre Norm模型，其实际等效层数不如$L$层的Post Norm模型，而层数少了导致效果变差了。

具体怎么理解呢？很简单，对于Pre Norm模型我们迭代得到：  
\begin{equation}\begin{aligned}  
\boldsymbol{x}_{t+1} =&\,\boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t)) \\\  
=&\, \boldsymbol{x}_{t-1} + F_{t-1}(\text{Norm}(\boldsymbol{x}_{t-1})) + F_t(\text{Norm}(\boldsymbol{x}_t)) \\\  
=&\, \cdots \\\  
=&\, \boldsymbol{x}_0 + F_0 (\text{Norm}(\boldsymbol{x}_0)) + \cdots + F_{t-1}(\text{Norm}(\boldsymbol{x}_{t-1})) + F_t(\text{Norm}(\boldsymbol{x}_t))  
\end{aligned}\end{equation}  
其中每一项都是同一量级的，那么有$\boldsymbol{x}_{t+1}=\mathcal{O}(t+1)$，也就是说第$t+1$层跟第$t$层的差别就相当于$t+1$与$t$的差别，当$t$较大时，两者的相对差别是很小的，因此  
\begin{equation}\begin{aligned}  
&\,F_t(\text{Norm}(\boldsymbol{x}_t)) + F_{t+1}(\text{Norm}(\boldsymbol{x}_{t+1})) \\\  
\approx&\,F_t(\text{Norm}(\boldsymbol{x}_t)) + F_{t+1}(\text{Norm}(\boldsymbol{x}_t)) \\\  
=&\, \begin{pmatrix} 1 & 1\end{pmatrix}\begin{pmatrix} F_t \\\ F_{t+1}\end{pmatrix}(\text{Norm}(\boldsymbol{x}_t))  
\end{aligned}\end{equation}  
这个意思是说，当$t$比较大时，$\boldsymbol{x}_t,\boldsymbol{x}_{t+1}$相差较小，所以$F_{t+1}(\text{Norm}(\boldsymbol{x}_{t+1}))$与$F_{t+1}(\text{Norm}(\boldsymbol{x}_t))$很接近，因此原本一个$t$层的模型与$t+1$层和，近似等效于一个更宽的$t$层模型，所以在Pre Norm中多层叠加的结果更多是增加宽度而不是深度，层数越多，这个层就越“虚”。

说白了，Pre Norm结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。而Post Norm刚刚相反，在[《浅谈Transformer的初始化、参数化与标准化》](/archives/8620)中我们就分析过，它每Norm一次就削弱一次恒等分支的权重，所以Post Norm反而是更突出残差分支的，因此Post Norm中的层数更加“足秤”，一旦训练好之后效果更优。

## 相关工作 #

前段时间号称能训练1000层Transformer的[DeepNet](/archives/8978)想必不少读者都听说过，在其论文[《DeepNet: Scaling Transformers to 1,000 Layers》](https://papers.cool/arxiv/2203.00555)中对Pre Norm的描述是：

> However, the gradients of Pre-LN at bottom layers tend to be larger than at top layers, leading to a degradation in performance compared with Post-LN.

不少读者当时可能并不理解这段话的逻辑关系，但看了前一节内容的解释后，想必会有新的理解。

简单来说，所谓“the gradients of Pre-LN at bottom layers tend to be larger than at top layers”，就是指Pre Norm结构会过度倾向于恒等分支（bottom layers），从而使得Pre Norm倾向于退化（degradation）为一个“浅而宽”的模型，最终不如同一深度的Post Norm。这跟前面的直观理解本质上是一致的。

## 文章小结 #

本文主要分享了“为什么Pre Norm的效果不如Post Norm”的一个直观理解。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9009>_

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

苏剑林. (Mar. 29, 2022). 《为什么Pre Norm的效果不如Post Norm？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9009>

@online{kexuefm-9009,  
title={为什么Pre Norm的效果不如Post Norm？},  
author={苏剑林},  
year={2022},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/9009}},  
} 


---

## 深度数学分析 #

### 梯度流分析 {#gradient-flow}

<div class="theorem-box">

**定理1：Pre Norm与Post Norm的梯度传播差异**

对于$L$层网络，记损失函数为$\mathcal{L}$，则梯度反向传播有：

**Pre Norm**:
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_t} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \left( \mathbf{I} + \frac{\partial F_t}{\partial \text{Norm}(\boldsymbol{x}_t)} \cdot \frac{\partial \text{Norm}(\boldsymbol{x}_t)}{\partial \boldsymbol{x}_t} \right)$$

**Post Norm**:
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_t} = \frac{\partial \text{Norm}}{\partial (\boldsymbol{x}_t + F_t)} \cdot \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \left( \mathbf{I} + \frac{\partial F_t}{\partial \boldsymbol{x}_t} \right)$$

</div>

#### 详细推导

**Pre Norm的梯度分析**：

对于Pre Norm结构 $\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t))$，使用链式法则：

\begin{equation}\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_t} &= \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \cdot \frac{\partial \boldsymbol{x}_{t+1}}{\partial \boldsymbol{x}_t} \\
&= \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \left( \frac{\partial \boldsymbol{x}_t}{\partial \boldsymbol{x}_t} + \frac{\partial F_t(\text{Norm}(\boldsymbol{x}_t))}{\partial \boldsymbol{x}_t} \right) \\
&= \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \left( \mathbf{I} + \frac{\partial F_t}{\partial \text{Norm}(\boldsymbol{x}_t)} \cdot \frac{\partial \text{Norm}(\boldsymbol{x}_t)}{\partial \boldsymbol{x}_t} \right)
\end{aligned}\end{equation}

关键观察：Pre Norm中恒等路径（$\mathbf{I}$项）**不经过任何归一化**，因此梯度可以直接传递。这使得训练更加稳定，但也导致了一个问题：

迭代$L$层后的梯度为：
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_0} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_L} \prod_{t=0}^{L-1} \left( \mathbf{I} + \mathbf{J}_t \right)$$

其中 $\mathbf{J}_t = \frac{\partial F_t}{\partial \text{Norm}(\boldsymbol{x}_t)} \cdot \frac{\partial \text{Norm}(\boldsymbol{x}_t)}{\partial \boldsymbol{x}_t}$。

当$L$很大时，由于恒等路径的存在，梯度的主要成分来自：
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_0} \approx \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_L} + \text{低阶修正项}$$

这意味着**浅层参数的梯度主要由顶层传来**，中间层的作用被稀释。

**Post Norm的梯度分析**：

对于Post Norm结构 $\boldsymbol{x}_{t+1} = \text{Norm}(\boldsymbol{x}_t + F_t(\boldsymbol{x}_t))$：

\begin{equation}\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_t} &= \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \cdot \frac{\partial \text{Norm}(\boldsymbol{x}_t + F_t(\boldsymbol{x}_t))}{\partial \boldsymbol{x}_t} \\
&= \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{t+1}} \cdot \frac{\partial \text{Norm}}{\partial \boldsymbol{z}_t} \cdot \left( \mathbf{I} + \frac{\partial F_t}{\partial \boldsymbol{x}_t} \right)
\end{aligned}\end{equation}

其中 $\boldsymbol{z}_t = \boldsymbol{x}_t + F_t(\boldsymbol{x}_t)$。

关键区别：每一层的梯度都要**经过归一化的雅可比矩阵** $\frac{\partial \text{Norm}}{\partial \boldsymbol{z}_t}$，这会：
1. **削弱恒等分支**：每经过一次Norm，恒等路径的权重被压缩
2. **强化残差分支**：迫使网络更依赖$F_t$的学习
3. **增加训练难度**：需要更精细的初始化和学习率设置

迭代$L$层后：
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_0} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_L} \prod_{t=0}^{L-1} \left( \frac{\partial \text{Norm}}{\partial \boldsymbol{z}_t} \cdot (\mathbf{I} + \mathbf{J}_t') \right)$$

由于每层都有$\frac{\partial \text{Norm}}{\partial \boldsymbol{z}_t}$项（典型值约为$\frac{1}{\sqrt{d}}$量级），梯度在反向传播时会被逐层调制，**每一层的学习都更加独立和充分**。

---

### 范数演化分析 {#norm-evolution}

<div class="derivation-box">

**命题1：Pre Norm中的范数增长**

假设$F_t$的输出与输入同量级，即$\|F_t(\text{Norm}(\boldsymbol{x}_t))\| = \Theta(1)$，则：

$$\|\boldsymbol{x}_L\| = \Theta(L)$$

**证明**：

由于$\text{Norm}(\boldsymbol{x}_t)$将$\boldsymbol{x}_t$归一化到固定范数（Layer Norm使得每个样本的特征均值为0，方差为1），我们有：

\begin{equation}\begin{aligned}
\boldsymbol{x}_{t+1} &= \boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t)) \\
\|\boldsymbol{x}_{t+1}\|^2 &= \|\boldsymbol{x}_t\|^2 + 2\langle \boldsymbol{x}_t, F_t \rangle + \|F_t\|^2
\end{aligned}\end{equation}

在期望意义下（假设$\boldsymbol{x}_t$与$F_t$近似正交或相关性较弱）：
$$\mathbb{E}[\|\boldsymbol{x}_{t+1}\|^2] \approx \mathbb{E}[\|\boldsymbol{x}_t\|^2] + \mathbb{E}[\|F_t\|^2] = \mathbb{E}[\|\boldsymbol{x}_t\|^2] + \Theta(1)$$

递推得到：
$$\mathbb{E}[\|\boldsymbol{x}_L\|^2] = \mathbb{E}[\|\boldsymbol{x}_0\|^2] + L \cdot \Theta(1) = \Theta(L)$$

因此$\|\boldsymbol{x}_L\| = \Theta(\sqrt{L})$。

</div>

**关键推论**：当$t$较大时，$\boldsymbol{x}_t$的范数已经增长到$\Theta(\sqrt{t})$，而$F_t$的输出仍然是$\Theta(1)$（因为输入被归一化了），所以：

$$\frac{\|F_t(\text{Norm}(\boldsymbol{x}_t))\|}{\|\boldsymbol{x}_t\|} = \Theta\left(\frac{1}{\sqrt{t}}\right) \to 0$$

这正是"深度有水分"的数学表述：**越深的层，残差分支相对于主干的贡献越小**。

相比之下，Post Norm每次都会重新归一化，保持$\|\boldsymbol{x}_t\| = \Theta(1)$，因此每一层的残差都能产生相同量级的贡献。

---

### Lipschitz常数与数值稳定性 {#lipschitz-analysis}

<div class="theorem-box">

**定理2：Lipschitz常数的层数依赖性**

设$L_F$为残差块$F$的Lipschitz常数（通常$L_F \approx 1$），$L_{\text{Norm}}$为归一化操作的Lipschitz常数（对于Layer Norm，$L_{\text{Norm}} \lesssim 1$）。

**Pre Norm**：整个$L$层网络的Lipschitz常数为
$$L_{\text{Pre}} = (1 + L_F \cdot L_{\text{Norm}})^L \approx e^{L \cdot L_F \cdot L_{\text{Norm}}}$$

**Post Norm**：整个$L$层网络的Lipschitz常数为
$$L_{\text{Post}} = L_{\text{Norm}}^L \cdot (1 + L_F)^L \approx \left( L_{\text{Norm}} (1+L_F) \right)^L$$

</div>

**分析**：

1. **Pre Norm的指数增长**：由于恒等路径完全不受约束，$L$层后输出可能是输入的$(1+L_F L_{\text{Norm}})^L$倍。当$L$很大时，这个倍数呈指数增长，但由于恒等分支占主导，整体行为类似于浅层宽网络。

2. **Post Norm的归一化效果**：每层的$L_{\text{Norm}}$项（通常$<1$）会压制指数增长，使得网络的动态范围更加可控。虽然训练难度增加（因为梯度也被调制），但一旦训练好，**每一层都真正参与了表示学习**。

#### Layer Normalization的Lipschitz常数

对于Layer Norm: $\text{LN}(\boldsymbol{x}) = \gamma \odot \frac{\boldsymbol{x} - \mu}{\sigma} + \beta$

其中$\mu = \frac{1}{d}\sum_i x_i$, $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$。

Lipschitz常数的上界为：
$$\left\|\frac{\partial \text{LN}}{\partial \boldsymbol{x}}\right\| \leq \|\gamma\| \cdot \sqrt{1 + \frac{1}{d}} \approx \|\gamma\|$$

当$\gamma$初始化为1时，$L_{\text{Norm}} \approx 1$，但由于归一化的除以$\sigma$操作，实际的动态缩放因子可能小于1，这正是Post Norm能够控制梯度爆炸的原因。

---

### DeepNet的解决方案 {#deepnet-solution}

<div class="comparison-box">

**DeepNet的核心思想**

论文[《DeepNet: Scaling Transformers to 1,000 Layers》](https://papers.cool/arxiv/2203.00555)提出了一个巧妙的初始化策略，结合Pre Norm和Post Norm的优点：

$$\boldsymbol{x}_{t+1} = \text{LN}(\boldsymbol{x}_t + \alpha \cdot F_t(\boldsymbol{x}_t))$$

关键：**残差分支的缩放因子** $\alpha = \alpha(L)$ 依赖于网络深度$L$。

</div>

#### 数学推导

**目标**：使得每一层的期望输出范数保持常数，即$\mathbb{E}[\|\boldsymbol{x}_t\|^2] = C$（常数）。

假设$F_t$的输出满足$\mathbb{E}[\|F_t(\boldsymbol{x}_t)\|^2] = V_F \|\boldsymbol{x}_t\|^2$（其中$V_F$与初始化方差有关）。

在Post Norm结构下（去掉外层Norm以简化分析）：
$$\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \alpha F_t(\boldsymbol{x}_t)$$

期望范数演化：
$$\mathbb{E}[\|\boldsymbol{x}_{t+1}\|^2] = \mathbb{E}[\|\boldsymbol{x}_t\|^2] + \alpha^2 \mathbb{E}[\|F_t\|^2] + 2\alpha \mathbb{E}[\langle \boldsymbol{x}_t, F_t \rangle]$$

假设残差与主干近似正交（在适当初始化下，初期可以近似成立）：
$$\mathbb{E}[\|\boldsymbol{x}_{t+1}\|^2] \approx \mathbb{E}[\|\boldsymbol{x}_t\|^2] (1 + \alpha^2 V_F)$$

经过$L$层后：
$$\mathbb{E}[\|\boldsymbol{x}_L\|^2] \approx \mathbb{E}[\|\boldsymbol{x}_0\|^2] (1 + \alpha^2 V_F)^L$$

**要求范数不爆炸**，即$(1 + \alpha^2 V_F)^L = \mathcal{O}(1)$，需要：
$$\alpha^2 V_F \cdot L = \mathcal{O}(1) \quad \Rightarrow \quad \alpha = \mathcal{O}(L^{-1/2})$$

**DeepNet的选择**：
$$\alpha = \frac{1}{\sqrt{2L}}$$

这个因子确保了：
1. **浅层**：$\alpha$较大，残差分支有足够影响力
2. **深层**：$\alpha$自动衰减，防止梯度爆炸和范数爆炸
3. **全局**：$L$层累积效果可控

#### Xavier初始化的修正

结合残差缩放，DeepNet还提出了针对残差分支的初始化策略：

对于$F_t$中的权重矩阵$\mathbf{W}$，使用：
$$\mathbf{W} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}} \cdot \beta^2 \right)$$

其中$\beta = \mathcal{O}(1)$是额外的缩放因子，与$\alpha$配合使用。

完整的前向传播：
$$\boldsymbol{x}_{t+1} = \text{LN}\left(\boldsymbol{x}_t + \underbrace{\frac{1}{\sqrt{2L}}}_{\alpha} \cdot F_t(\boldsymbol{x}_t; \mathbf{W}^{(\beta)}) \right)$$

**实验验证**：
- 标准Post Norm：最多训练$\sim$50-100层
- DeepNet：成功训练1000层Transformer，在WMT翻译任务上取得SOTA

---

### 有效深度的定量分析 {#effective-depth}

<div class="derivation-box">

**命题2：Pre Norm的有效深度**

定义网络的**有效深度**为使得前$k$层的累积贡献占总输出的某个比例（如90%）的最小$k$。

对于Pre Norm，第$t$层的相对贡献为：
$$r_t = \frac{\|F_t(\text{Norm}(\boldsymbol{x}_t))\|}{\|\boldsymbol{x}_L\|} \approx \frac{\Theta(1)}{\Theta(\sqrt{L})} = \Theta(L^{-1/2})$$

累积前$k$层的贡献：
$$\sum_{t=0}^{k-1} r_t \approx k \cdot \Theta(L^{-1/2}) = \Theta\left(\frac{k}{\sqrt{L}}\right)$$

要达到90%贡献，需要：
$$\frac{k}{\sqrt{L}} \gtrsim 0.9 \quad \Rightarrow \quad k \gtrsim 0.9\sqrt{L}$$

**结论**：$L$层的Pre Norm网络，有效深度仅为$\Theta(\sqrt{L})$！

</div>

**对比Post Norm**：每层贡献相对均匀（都在$\Theta(1)$量级），有效深度$\approx L$。

**直观比喻**：
- **Pre Norm**：像$\sqrt{L}$层深、$\sqrt{L}$倍宽的网络
- **Post Norm**：真正的$L$层深网络

由于深度比宽度更重要（更能学习层级化的抽象特征），Post Norm的表达能力更强。

---

### 信息论视角：互信息分析 {#mutual-information}

从信息论角度，我们可以分析输入$\boldsymbol{x}_0$与各层输出$\boldsymbol{x}_t$之间的互信息$I(\boldsymbol{x}_0; \boldsymbol{x}_t)$。

**Pre Norm**：
由于恒等路径占主导，互信息衰减慢：
$$I(\boldsymbol{x}_0; \boldsymbol{x}_t) \geq I_{\text{identity}} - \mathcal{O}(t \epsilon)$$

其中$\epsilon$是单层的信息损失。由于$\boldsymbol{x}_t$主要包含$\boldsymbol{x}_0$的信息加上各层的扰动，**原始信息被过度保留**，限制了层级化的特征提取。

**Post Norm**：
每次归一化会部分丢弃幅值信息，但保留方向信息：
$$I(\boldsymbol{x}_0; \boldsymbol{x}_t) \approx I_{\text{directional}} + \text{learned features}$$

这种"选择性遗忘"（幅值信息）+ "主动学习"（通过$F_t$）的机制，使得网络能够**逐层抽象**，构建层级化的特征表示。

---

### 训练动态的微分方程视角 {#ode-view}

将残差网络视为常微分方程(ODE)的离散化：
$$\frac{d\boldsymbol{x}}{dt} = f(t, \boldsymbol{x}), \quad \boldsymbol{x}(0) = \boldsymbol{x}_0$$

离散化：$\boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \Delta t \cdot f(t, \boldsymbol{x}_t)$

**Pre Norm**对应：
$$f(t, \boldsymbol{x}) = F_t(\text{Norm}(\boldsymbol{x}))$$

由于Norm操作，$f$的幅值被限制在$\Theta(1)$，因此随着$\|\boldsymbol{x}\|$增长到$\Theta(\sqrt{t})$，**动态系统的速度场相对于状态大小越来越弱**，系统演化趋于饱和。

**Post Norm**对应：
$$\boldsymbol{x}_{t+1} = \text{Norm}(\boldsymbol{x}_t + \Delta t \cdot f(t, \boldsymbol{x}_t))$$

每步都重新归一化，相当于**在单位球面上的流形演化**，每一步都保持相同的"步幅"，使得$L$步能够真正走完$L$的"距离"。

---

## 实验视角：梯度范数的实证分析 {#empirical-analysis}

### 梯度范数分布

[Understanding the Difficulty of Training Transformers](https://papers.cool/arxiv/2004.08249) 的实验显示：

**Pre Norm**：
- 底层（靠近输入）梯度范数：$\|\nabla_{\boldsymbol{x}_0} \mathcal{L}\| \approx 10^{-2}$ 到 $10^{-1}$
- 顶层（靠近输出）梯度范数：$\|\nabla_{\boldsymbol{x}_L} \mathcal{L}\| \approx 10^{-3}$ 到 $10^{-2}$
- **底层梯度更大**：这似乎是个优点，但实际上说明顶层学习不充分

**Post Norm**：
- 各层梯度范数更加均匀，约在$10^{-2}$到$10^{-1}$量级
- **每层都在充分学习**

### 参数更新的相对幅度

定义第$t$层的相对更新幅度：
$$\eta_t = \frac{\|\Delta \theta_t\|}{\|\theta_t\|} = \frac{\text{learning rate} \cdot \|\nabla_{\theta_t} \mathcal{L}\|}{\|\theta_t\|}$$

**Pre Norm**：由于浅层梯度大但参数范数也随训练增长，$\eta_t$在各层差异不大，但**浅层的实际影响被稀释**（因为范数膨胀）。

**Post Norm**：参数范数保持相对稳定，$\eta_t$与梯度范数成正比，使得**梯度信号直接转化为参数更新**。

---

## 收敛性理论 {#convergence-theory}

<div class="theorem-box">

**定理3：Pre Norm与Post Norm的收敛速度对比**

在平滑损失函数假设下（$\beta$-smooth，$L$-Lipschitz），使用梯度下降训练：

**Pre Norm**：若要达到$\epsilon$-最优解，需要迭代次数：
$$T_{\text{Pre}} = \mathcal{O}\left( \frac{L^2 \cdot (1 + L_F L_{\text{Norm}})^{2L}}{\epsilon^2} \right)$$

**Post Norm**（使用适当的Warmup和初始化）：
$$T_{\text{Post}} = \mathcal{O}\left( \frac{L^2 \cdot L_{\text{Norm}}^{2L} (1+L_F)^{2L}}{\epsilon^2} \right)$$

由于$L_{\text{Norm}} < 1$，当$L$很大时，$L_{\text{Norm}}^{2L}$可能很小，需要通过Warmup和学习率调整来补偿。

</div>

**关键洞察**：
- Pre Norm收敛快（初期），但收敛到的解可能是次优的（因为有效深度不足）
- Post Norm收敛慢（需要Warmup），但最终能够达到更优的解（充分利用深度）

---

## 实际应用建议 {#practical-recommendations}

<div class="example-box">

**场景1：训练超深网络（$L > 50$）**
- **推荐**：DeepNet风格的Post Norm + 残差缩放$\alpha = 1/\sqrt{2L}$
- **初始化**：Xavier + $\beta$调整
- **学习率**：需要Warmup，逐步增大到目标学习率

**场景2：快速实验和调参（$L \leq 24$）**
- **推荐**：Pre Norm
- **优点**：训练稳定，无需精细调参
- **代价**：可能损失1-2个点的最终性能

**场景3：预训练大模型**
- **推荐**：Post Norm（大多数SOTA模型的选择）
- **理由**：Pretraining阶段有足够资源做仔细调参，最终性能提升值得额外的训练成本

**场景4：微调（Finetune）**
- **观察**：Post Norm预训练的模型通常微调效果更好
- **原因**：各层特征更充分，迁移能力更强

</div>

---

## 总结与展望 {#conclusion-extended}

本文从多个角度分析了Pre Norm与Post Norm的差异：

1. **直观理解**：Pre Norm的深度有"水分"，$L$层的有效深度仅$\Theta(\sqrt{L})$
2. **梯度流**：Pre Norm恒等路径主导，Post Norm残差分支充分学习
3. **范数演化**：Pre Norm线性增长导致后层贡献被稀释
4. **Lipschitz常数**：两种结构的数值稳定性差异
5. **DeepNet方案**：通过残差缩放平衡稳定性和表达能力
6. **信息论**：Pre Norm过度保留原始信息，Post Norm逐层抽象

**未来方向**：
1. **自适应归一化**：根据层深度和训练阶段自动调整归一化策略
2. **混合策略**：前几层用Pre Norm（稳定），后几层用Post Norm（表达力）
3. **可学习的$\alpha$**：将残差缩放因子变成可学习参数
4. **超越Layer Norm**：探索更适合深度网络的归一化方法（如RMSNorm）

---

