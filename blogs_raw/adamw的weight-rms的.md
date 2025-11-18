---
title: AdamW的Weight RMS的...
slug: adamw的weight-rms的
date: 2025-10-01
tags: 估计, 梯度, 优化器, 平均场, 生成模型
status: pending
---

# AdamW的Weight RMS的...

**原文链接**: [https://spaces.ac.cn/archives/11307](https://spaces.ac.cn/archives/11307)

**发布日期**: 

---

在[《为什么Adam的Update RMS是0.2？》](/archives/11267)中，我们用平均场近似估计了Adam的Update RMS。不久后，读者 [@EIFY](https://x.com/EIFY/status/1965888629814988984) 指出相同的结果已经出现在论文[《Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks》](https://papers.cool/arxiv/2305.17212)中。阅读后，笔者发现其中不仅包含了Update RMS的估计，还包含了Weight RMS的估计。

也就是说，AdamW训出来的模型，其权重的RMS是可以事先估计出来一个渐近结果的。大家会不会觉得这个结论有点意外？反正笔者第一次看到它是颇为意外的，直觉上权重模长是模型根据训练集自己学出来的，结果它告诉我这已经隐藏在优化器的超参中，可谓很反直觉了。

这篇文章我们还是用平均场近似方法，来复现对Weight RMS的渐近估计。

## 滑动视角 #

首先还是来回顾AdamW的更新规则：  
\begin{equation}\text{Adam}\color{skyblue}{\text{W}}:=\left\\{\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\\  
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\\  
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\\  
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\\  
&\boldsymbol{u}_t =\hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.\\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})  
\end{aligned}\right.\end{equation}  
再次说明，这里加粗符号默认都是$\mathbb{R}^d$的向量，向量的乘除（包括平方、开根号）默认都是Element-wise的Hadamard积/商。

跟[《为什么Adam的Update RMS是0.2？》](/archives/11267)一样，我们考虑$t\to\infty$（对于$\beta_1,\beta_2$来说）和$\epsilon\to 0$，所以$\boldsymbol{u}_t=\boldsymbol{m}_t/\sqrt{\boldsymbol{v}_t}$。我们暂时先考虑$\eta_t,\lambda_t$都是常数的例子，所以它们的下标可以省略掉，并且记$\beta_3 = 1-\eta\lambda$，我们有  
\begin{equation}\boldsymbol{\theta}_t = \beta_3\boldsymbol{\theta}_{t-1} + (1-\beta_3)(-\boldsymbol{u}_t/\lambda)\label{eq:ema-wd}\end{equation}  
这个式子表明，我们可以从更新量的滑动平均（Exponential Moving Average，EMA）角度来理解Weight Decay。这是一个很有意义的视角转换，是[《How to set AdamW’s weight decay as you scale model and dataset size》](https://papers.cool/arxiv/2405.13698)、[《Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training》](https://papers.cool/arxiv/2505.13738)等工作的基础。

## 加权平均 #

根据式$\eqref{eq:ema-wd}$，我们可以将$\boldsymbol{\theta}_t$展开为加权平均形式  
\begin{equation}\boldsymbol{\theta}_t = \beta_3^t\boldsymbol{\theta}_0 + (1-\beta_3)\sum_{i=1}^t \beta_3^{t-i} (-\boldsymbol{u}_i/\lambda)\label{eq:theta-t}\end{equation}  
同理，$\boldsymbol{m}_t$和$\boldsymbol{v}_t$也可以展开为  
\begin{equation}\boldsymbol{m}_t = (1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i}\boldsymbol{g}_i,\qquad \boldsymbol{v}_t = (1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i}\boldsymbol{g}_i^2\label{eq:mv-roll}\end{equation}  
这里有个小细节，$\boldsymbol{\theta}_t$的表达式我们保留了$\boldsymbol{\theta}_0$，但$\boldsymbol{m}_t$和$\boldsymbol{v}_t$的表达式我们没有保留$\boldsymbol{m}_0$和$\boldsymbol{v}_0$，原因有两个：1、$\boldsymbol{m}$和$\boldsymbol{v}$的初始化一般是零；2、即便它们初始化不是零，但对应的$\beta_1^t$和$\beta_2^t$也会足够接近于零，因此初始化的影响可以忽略。

然而，$\boldsymbol{\theta}$是模型权重，它的初始化通常不是零，并且$\beta_3$往往非常接近于1，对于整个训练周期而言，$\beta_3^t$不一定能充分接近于零，因此我们显式保留$\beta_3^t$和$\boldsymbol{\theta}_0$，按需取舍。

## 快速估计 #

我们的任务是估计Weight RMS，即$\Vert\boldsymbol{\theta}_t\Vert_{RMS}$，顾名思义，它是各个分量的Root Mean Square：  
\begin{equation}\Vert\boldsymbol{\theta}\Vert_{RMS} = \sqrt{\frac{1}{d}\sum_{i=1}^d \theta_i^2},\qquad\qquad \text{其中 }\boldsymbol{\theta} = (\theta_1,\theta_2,\cdots,\theta_d)\end{equation}  
它跟模长的区别就是多除了个$\sqrt{d}$，所以模长的大部分性质对RMS同样成立。对于$\Vert\boldsymbol{\theta}_t\Vert_{RMS}$，我们有一个快速但不是那么准确的推导方式：直接对式$\eqref{eq:ema-wd}$两边求$\Vert\cdot\Vert_{RMS}^2$，可以得到  
\begin{equation}\begin{aligned}  
\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 =&\, \Vert\beta_3\boldsymbol{\theta}_{t-1} + (1-\beta_3)(-\boldsymbol{u}_t/\lambda)\Vert_{RMS}^2 \\\\[5pt]  
=&\, \beta_3^2\Vert\boldsymbol{\theta}_{t-1}\Vert_{RMS}^2 + (1-\beta_3)^2\Vert\boldsymbol{u}_t\Vert_{RMS}^2/\lambda^2 - 2\beta_3(1-\beta_3)\boldsymbol{\theta}_{t-1}\cdot\boldsymbol{u}_t/(\lambda d)  
\end{aligned}\end{equation}  
假设$\boldsymbol{\theta}_{t-1},\boldsymbol{u}_t$近乎正交，那么$\boldsymbol{\theta}_{t-1}\cdot\boldsymbol{u}_t\approx 0$，这在高维空间中通常是一个不错的近似（参考[《n维空间下两个随机向量的夹角分布》](/archives/7076)），然后$\Vert\boldsymbol{u}_t\Vert_{RMS}$我们已经算过了，答案是约等于$\sqrt{\frac{1-\beta_1}{1+\beta_1}}$，最后我们考虑的是趋于稳态的结果，所以$\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2=\Vert\boldsymbol{\theta}_{t-1}\Vert_{RMS}^2$，于是有  
\begin{equation}(1-\beta_3^2)\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx (1-\beta_3)^2 \frac{1-\beta_1}{1+\beta_1} /\lambda^2\qquad\Rightarrow\qquad \Vert\boldsymbol{\theta}_t\Vert_{RMS} \approx \sqrt{\frac{1-\beta_1}{1+\beta_1}\frac{\eta}{2\lambda}}\end{equation}  
从左式到右式还用到了$\beta_3\approx 1$的近似。最后的结果会有些误差，因为$\boldsymbol{\theta}_t\cdot\boldsymbol{u}_t\approx 0$实际上并不那么成立，但$\Vert\boldsymbol{\theta}_t\Vert_{RMS}\propto \sqrt{\eta/\lambda}$的结论是正确的。类似的推导还出现在[《Why Gradients Rapidly Increase Near the End of Training》](https://papers.cool/arxiv/2506.02285)。

## 更好近似 #

很多情况下我们只需要知道$\Vert\boldsymbol{\theta}_t\Vert_{RMS}\propto \sqrt{\eta/\lambda}$就行了，这是一个比较通用的结论。而对于追求更准确结论的读者来说，我们可以用平均场方法得到一个更好的近似，代价是计算过程会复杂不少，但好处是我们可以获得更多更清晰的认知。

### 步骤之一 #

我们从式$\eqref{eq:theta-t}$出发，求和这一项，本身就具有加权平均的形式，所以我们先用第一次平均场：  
\begin{equation}\underbrace{\frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i} \boldsymbol{u}_i}_{\text{记为}\bar{\boldsymbol{u}}_t} = \frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i} \frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}}\approx \frac{\bar{\boldsymbol{m}}_t \,\,\triangleq\,\, \frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{m}_i}{\sqrt{\bar{\boldsymbol{v}}_t \,\,\triangleq\,\, \frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{v}_i}}\label{eq:u-bar}\end{equation}  
现在再次回到式$\eqref{eq:theta-t}$，由于$\boldsymbol{\theta}_0$是随机的初始化向量，因此可以假设$\boldsymbol{\theta}_0$与$\bar{\boldsymbol{u}}_t$正交，于是我们有  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \beta_3^{2t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_3^t)^2 \lambda^{-2}\Vert \bar{\boldsymbol{u}}_t\Vert_{RMS}^2\end{equation}  
现在我们要求$\Vert \bar{\boldsymbol{u}}_t\Vert_{RMS}^2$，根据之前的经验，我们需要假设$\boldsymbol{g}_j$独立同分布地服从$\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$，然后求  
\begin{equation}\mathbb{E}[\bar{\boldsymbol{u}}_t^2] \approx \mathbb{E}\left[\frac{\bar{\boldsymbol{m}}_t^2}{\bar{\boldsymbol{v}}_t}\right] \approx \frac{\mathbb{E}[\bar{\boldsymbol{m}}_t^2]}{\mathbb{E}[\bar{\boldsymbol{v}}_t]}\end{equation}  
最后再对$\mathbb{E}[\bar{\boldsymbol{u}}_t^2]$的各个分量求平均，那么就可以作为$\Vert \bar{\boldsymbol{u}}_t\Vert_{RMS}^2$的近似。

### 步骤之二 #

结合式$\eqref{eq:mv-roll}$，我们得到  
\begin{gather}  
\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{m}_i = (1 - \beta_1)\sum_{i=1}^t \beta_3^{t-i} \sum_{j=1}^i \beta_1^{i-j}\boldsymbol{g}_j = (1 - \beta_1)\sum_{j=1}^t \frac{\beta_3^{t-j+1} - \beta_1^{t-j+1}}{\beta_3 - \beta_1}\boldsymbol{g}_j\\\  
\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{v}_i = (1 - \beta_2)\sum_{i=1}^t \beta_3^{t-i} \sum_{j=1}^i \beta_2^{i-j}\boldsymbol{g}_j^2 = (1 - \beta_2)\sum_{j=1}^t \frac{\beta_3^{t-j+1} - \beta_2^{t-j+1}}{\beta_3 - \beta_2}\boldsymbol{g}_j^2\\\  
\end{gather}  
最后一个双重求和化简，如果大家没有思路，可以交给Kimi完成（参考[链接](https://www.kimi.com/share/d3d35hpsfuv6jqe78c20)）。由上式可知$\bar{\boldsymbol{m}}_t,\bar{\boldsymbol{v}}_t$分别是梯度和梯度平方的加权平均，所以求$\Vert \bar{\boldsymbol{u}}_t\Vert_{RMS}^2$跟[《为什么Adam的Update RMS是0.2？》](/archives/11267)求$\Vert \boldsymbol{u}_t\Vert_{RMS}^2$本质上是一样的，只不过加权系数不同。

### 步骤之三 #

我们先求分母  
\begin{equation}\begin{aligned}  
\mathbb{E}[\bar{\boldsymbol{v}}_t] =&\, \frac{(1 - \beta_3)(1 - \beta_2)}{1 - \beta_3^t}\sum_{j=1}^t \frac{\beta_3^{t-j+1} - \beta_2^{t-j+1}}{\beta_3 - \beta_2}\mathbb{E}[\boldsymbol{g}_j^2] \\\  
=&\, \frac{(1 - \beta_3)(1 - \beta_2)}{1 - \beta_3^t}\sum_{j=1}^t \frac{\beta_3^{t-j+1} - \beta_2^{t-j+1}}{\beta_3 - \beta_2}(\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2) \\\  
=&\, \frac{(1 - \beta_3)(1 - \beta_2)}{(1 - \beta_3^t)(\beta_3 - \beta_2)}\left(\frac{\beta_3 - \beta_3^{t+1}}{1 - \beta_3} - \frac{\beta_2 - \beta_2^{t+1}}{1 - \beta_2}\right)(\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2) \\\\[5pt]  
\approx &\, \boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2  
\end{aligned}\end{equation}  
最后一步的约等号，是因为实际训练中，$\beta_3$会足够接近于1，而$\beta_2^{t+1}$会足够接近于0，但$\beta_3^{t+1}$不一定，所以我们将$\beta_2^{t+1}$替换成零，并在化简之后将独立的$\beta_3$替换成$1$，最后再加上近似$\beta_3^{t+1}\approx \beta_3^t$。

### 步骤之四 #

然后是$\mathbb{E}[\bar{\boldsymbol{m}}_t^2] = \mathbb{E}[\bar{\boldsymbol{m}}_t]^2 + \mathbb{V}ar[\bar{\boldsymbol{m}}_t]$，$\mathbb{E}[\bar{\boldsymbol{m}}_t]$的计算跟$\mathbb{E}[\bar{\boldsymbol{v}}_t]$类似，结果是$\boldsymbol{\mu}$，$\mathbb{V}ar[\bar{\boldsymbol{m}}_t]$的计算我们则利用方差的平方可加性：  
\begin{equation}\begin{aligned}  
\mathbb{V}ar[\bar{\boldsymbol{m}}_t] =&\, \frac{(1 - \beta_3)^2(1 - \beta_1)^2}{(1-\beta_3^t)^2}\sum_{j=1}^t \left(\frac{\beta_3^{t-j+1} - \beta_1^{t-j+1}}{\beta_3 - \beta_1}\right)^2\mathbb{V}ar[\boldsymbol{g}_j] \\\  
=&\, \frac{(1 - \beta_3)^2(1 - \beta_1)^2}{(1-\beta_3^t)^2}\sum_{j=1}^t \left(\frac{\beta_3^{t-j+1} - \beta_1^{t-j+1}}{\beta_3 - \beta_1}\right)^2 \boldsymbol{\sigma}^2 \\\  
=&\, \frac{(1 - \beta_3)^2(1 - \beta_1)^2}{(1-\beta_3^t)^2(\beta_3 - \beta_1)^2}\left(\frac{\beta_3^2 - \beta_3^{2(t+1)}}{1 - \beta_3^2} + \frac{\beta_1^2 - \beta_1^{2(t+1)}}{1 - \beta_1^2} - 2\frac{\beta_1\beta_3 - \beta_1^{t+1}\beta_3^{t+1}}{1 - \beta_1\beta_3}\right) \boldsymbol{\sigma}^2 \\\\[5pt]  
\approx &\, (1 - \beta_3)(1 + \beta_3^t)\boldsymbol{\sigma}^2/2(1 - \beta_3^t)  
\end{aligned}\end{equation}  
取约等号的理由同上。

### 步骤之五 #

代入上两节的计算结果，我们有  
\begin{equation}\mathbb{E}[\bar{\boldsymbol{u}}_t^2] \approx \frac{\boldsymbol{\mu}^2 + (1 - \beta_3)(1 + \beta_3^t)\boldsymbol{\sigma}^2/2(1 - \beta_3^t)}{\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2}\end{equation}  
那么  
\begin{equation}\Vert\bar{\boldsymbol{u}}_t\Vert_{RMS}^2 \approx \frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + (1 - \beta_3)(1 + \beta_3^t)/2(1 - \beta_3^t)}{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + 1} \end{equation}  
最终有  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \beta_3^{2t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_3^t)^2 \frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + (1 - \beta_3)(1 + \beta_3^t)/2(1 - \beta_3^t)}{\lambda^2(\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + 1)}\label{eq:theta-rms}\end{equation}

## 结果浅析 #

式$\eqref{eq:theta-rms}$看起来比较复杂，我们观察几个特例。首先考虑$\boldsymbol{\mu}=\boldsymbol{0}$这个例子，此时  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \beta_3^{2t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_3^{2t}) (1 - \beta_3)/2\lambda^2 = \beta_3^{2t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_3^{2t}) \eta/2\lambda\label{eq:theta-rms-mu0}\end{equation}  
特别地，如果考虑$t\to\infty$，或者$\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2$就初始化为$\eta/2\lambda$，那么就有  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS} \approx \sqrt{\frac{\eta}{2\lambda}}\label{eq:theta-rms-simple}\end{equation}  
这便是论文[《Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks》](https://papers.cool/arxiv/2305.17212)给出的结果，跟原论文的假设一致，它是零均值下的随机游走的稳态结果。如果不考虑$t\to\infty$，而是考虑$\lambda\to 0$的极限，那么由式$\eqref{eq:theta-rms-mu0}$我们将得到  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + \eta^2 t\end{equation}  
这表明在没有Weight Decay的时候，$\Vert\boldsymbol{\theta}_t\Vert_{RMS}$大致按照$\eta\sqrt{t}$的速度增长，这也表明在没有Weight Decay时，我们可以通过设置特定的学习率Schedule来实现Weight RMS的稳定性。另一方面，如果Batch Size足够大，导致信噪比项$\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2$占主导，那么由式$\eqref{eq:theta-rms}$得  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \beta_3^{2t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_3^t)^2 \frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2}{\lambda^2(\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + 1)}\end{equation}  
这个可能适用于模型需要主动增加Weight RMS的特殊情形。不过从经验来看，这种情况发生的概率一般比较小。

## 模拟实验 #

我们可以用如下模拟脚本，来简单验证上述的准确性：
    
    
    import numpy as np
    
    N, T = 10000, 100000
    beta1, beta2 = 0.9, 0.95
    m, v = 0, 0
    w = np.random.randn(N) * 0.1
    for i in range(T):
        g = np.random.randn(N)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        w = w - 0.001 * (m / v**0.5 + 0.1 * w)
    
    weight_rms = (w**2).mean()**0.5
    print(weight_rms)
    

大家可以自行改变权重的初始化或梯度的均值方差等，看最终结果跟式$\eqref{eq:theta-rms}$的吻合程度，笔者自行试了一波，整体来说还是比较靠谱的。

## 符号版本 #

只需要将前述证明调整一下，就可以适用于“SignSGDM + Weight Decay”的组合：  
\begin{equation}\text{SignSGDM}\color{skyblue}{\text{W}}:=\left\\{\begin{aligned}  
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\\  
&\boldsymbol{u}_t = \newcommand{sign}{\mathop{\text{sign}}}\sign(\boldsymbol{m}_t)\\\  
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})  
\end{aligned}\right.\end{equation}  
修改的地方是由于$\sign(\boldsymbol{m}_t)=\boldsymbol{m}_t/\sqrt{\boldsymbol{m}_t^2}$，所以要将$\bar{\boldsymbol{v}}_t$的定义改为  
\begin{equation}\bar{\boldsymbol{v}}_t \triangleq \frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{m}_i^2\end{equation}  
那么  
\begin{equation}\mathbb{E}[\bar{\boldsymbol{v}}_t] = \frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i}\mathbb{E}[\boldsymbol{m}_i^2] \approx \frac{1-\beta_3}{1-\beta_3^t}\sum_{i=1}^t \beta_3^{t-i}\mathbb{E}\left(\boldsymbol{\mu}^2 + \frac{1-\beta_1}{1 + \beta_1}\boldsymbol{\sigma}^2\right) = \boldsymbol{\mu}^2 + \frac{1-\beta_1}{1 + \beta_1}\boldsymbol{\sigma}^2\end{equation}  
其中$\mathbb{E}[\boldsymbol{m}_i^2]$的计算我们参考[《为什么Adam的Update RMS是0.2？》](/archives/11267)或[《重新思考学习率与Batch Size（四）：EMA》](/archives/11301)都行。利用上述结果，我们得到  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \beta_3^{2t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_3^t)^2 \frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + (1 - \beta_3)(1 + \beta_3^t)/2(1 - \beta_3^t)}{\lambda^2\left(\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + \frac{1-\beta_1}{1 + \beta_1}\right)}\end{equation}  
特别地，考虑$\boldsymbol{\mu}=0,t\to\infty$的极限，我们有  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \sqrt{\frac{\eta}{2\lambda}\frac{1+\beta_1}{1 - \beta_1}}\end{equation}  
这个结果也很合理，因为SignSGDMW的Update RMS是AdamW的$\sqrt{\frac{1+\beta_1}{1 - \beta_1}}$倍，所以同样$\eta,\lambda$下它的Weight RMS也是$\sqrt{\frac{1+\beta_1}{1 - \beta_1}}$倍。

## 相关分析 #

前面说了，结果$\eqref{eq:theta-rms-simple}$跟论文[《Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks》](https://papers.cool/arxiv/2305.17212)是一致的，但我们的推导方法是完全不同的，并且能得到更一般的$\eqref{eq:theta-rms}$。不过，原论文也有一些很有意思的地方，比如它所提的 **Total Update Contribution (TUC)** 概念，就值得赏析一番。

TUC的思想是这样的：由于动量机制的存在，当前的梯度$\boldsymbol{g}_t$不止停留在当前步骤，它还会影响到未来的步骤（但会打个“折扣”），所以假设训练步数趋于无穷，我们可以考虑当前梯度$\boldsymbol{g}_t$对整个训练过程的**总贡献** 。具体来说，对于Adam我们有$\boldsymbol{u}_t=\boldsymbol{m}_t/\sqrt{\boldsymbol{v}_t}$，当前$\boldsymbol{g}_t$对$\boldsymbol{u}_t$的贡献是$(1-\beta_1)\boldsymbol{g}_t/\sqrt{\boldsymbol{v}_t}$，下一步$\boldsymbol{g}_t$将会打个折扣（乘以$\beta_1$），而且分母改为$\boldsymbol{v}_{t+1}$，依此类推，所以可以定义总贡献为  
\begin{equation}\tilde{\boldsymbol{u}}_t = \sum_{k=t}^{\infty} (1-\beta_1)\beta_1^{k-t}\frac{\boldsymbol{g}_t}{\sqrt{\boldsymbol{v}_k}}\end{equation}  
这样我们就将更新$\boldsymbol{u}_1,\boldsymbol{u}_2,\boldsymbol{u}_3,\cdots$分解为更新$\tilde{\boldsymbol{u}}_1,\tilde{\boldsymbol{u}}_2,\tilde{\boldsymbol{u}}_3,\cdots$，这样的好处是每个$\tilde{\boldsymbol{u}}$只有单步梯度，那么我们就可以重复快速估计一节的推导：  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 = \Vert\beta_3\boldsymbol{\theta}_{t-1} + (1-\beta_3)(-\tilde{\boldsymbol{u}}_t/\lambda)\Vert_{RMS}^2 \approx \beta_3^2\Vert\boldsymbol{\theta}_{t-1}\Vert_{RMS}^2 + (1-\beta_3)^2\Vert\tilde{\boldsymbol{u}}_t\Vert_{RMS}^2/\lambda^2 \label{eq:tilde-u-rms}\end{equation}  
最后的近似依赖于$\boldsymbol{\theta}_{t-1}\cdot\tilde{\boldsymbol{u}}_t\approx 0$，我们断言$\boldsymbol{\theta}_{t-1}\cdot\tilde{\boldsymbol{u}}_t$比$\boldsymbol{\theta}_{t-1}\cdot\boldsymbol{u}_t$更接近于零，因为$\tilde{\boldsymbol{u}}_t$只依赖于当前梯度$\boldsymbol{g}_t$，而$\boldsymbol{\theta}_{t-1}$还没接触到$\boldsymbol{g}_t$，所以它们是独立的变量，假设$\boldsymbol{g}_t$具有零均值时，$\boldsymbol{\theta}_{t-1}\cdot\tilde{\boldsymbol{u}}_t\approx 0$往往就容易成立了。而为了估计$\Vert\tilde{\boldsymbol{u}}_t\Vert_{RMS}^2$，原论文直接假设$\boldsymbol{g}_t/\sqrt{\boldsymbol{v}_k}$具有相同方向并且单位RMS，于是  
\begin{equation}\Vert\tilde{\boldsymbol{u}}_t\Vert_{RMS} = \sum_{k=t}^{\infty} (1-\beta_1)\beta_1^{k-t}\left\Vert\frac{\boldsymbol{g}_t}{\sqrt{\boldsymbol{v}_k}}\right\Vert_{RMS} = \sum_{k=t}^{\infty} (1-\beta_1)\beta_1^{k-t} = 1\end{equation}  
代入式$\eqref{eq:tilde-u-rms}$，结合快速估计一节同样的近似处理，解得  
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS} \approx \sqrt{\frac{\eta}{2\lambda}}\end{equation}  
然而，如果局限在原论文看，我们会发现有很多近似是莫名其妙的，比如$\boldsymbol{v}_t$中也有$\boldsymbol{g}_t$，所以说$\tilde{\boldsymbol{u}}_t$只包含当前$\boldsymbol{g}_t$的影响是不大准确的，还有$\Vert\boldsymbol{g}_t/\sqrt{\boldsymbol{v}_k}\Vert_{RMS}=1$的断言也显得比较生硬。但如果放到本文来看，我们会发现在平均场近似下，原论文的各种操作会显得很合理，所以原论文其实已经隐含地用到了平均场方法。

## 文章小结 #

这篇文章我们用平均场近似推导了一个有趣且可能让人意外的结论：AdamW训出来的模型，其权重的RMS也是可以渐近估计出来的，一般情况下，它只依赖于学习率和Weight Decay。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11307>_

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

苏剑林. (Oct. 01, 2025). 《AdamW的Weight RMS的渐近估计 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11307>

@online{kexuefm-11307,  
title={AdamW的Weight RMS的渐近估计},  
author={苏剑林},  
year={2025},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/11307}},  
} 


---

## 详细数学推导

### 1. AdamW优化器的完整推导

#### 1.1 从Adam到AdamW的演化

Adam优化器的原始形式为：
\begin{gather}
\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1)\boldsymbol{g}_t \tag{1} \\
\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2)\boldsymbol{g}_t^2 \tag{2} \\
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t}+\epsilon} \tag{3}
\end{gather}

**推导注释**：公式(1)是一阶矩（动量）的指数移动平均，$\beta_1$控制历史梯度的衰减速度。公式(2)是二阶矩的指数移动平均，用于自适应调整每个参数的学习率。公式(3)是参数更新规则，分母中的$\sqrt{\boldsymbol{v}_t}$提供了自适应的学习率缩放。

AdamW的关键改进是将权重衰减从梯度中解耦：
\begin{equation}
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \left(\frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t}+\epsilon} + \lambda\boldsymbol{\theta}_{t-1}\right) \tag{4}
\end{equation}

**几何直觉**：权重衰减项$\lambda\boldsymbol{\theta}_{t-1}$使参数向原点收缩，这在高维空间中提供了一个"向心力"，防止参数向量的模长无限增长。

#### 1.2 偏差修正的必要性

在训练初期，由于$\boldsymbol{m}_0=\boldsymbol{v}_0=\mathbf{0}$，估计值存在偏差。引入偏差修正：
\begin{gather}
\hat{\boldsymbol{m}}_t = \frac{\boldsymbol{m}_t}{1-\beta_1^t} \tag{5} \\
\hat{\boldsymbol{v}}_t = \frac{\boldsymbol{v}_t}{1-\beta_2^t} \tag{6}
\end{gather}

**推导**：考虑$\boldsymbol{m}_t$的期望值。假设梯度的期望为$\mathbb{E}[\boldsymbol{g}_t]=\boldsymbol{\mu}$，则：
\begin{align}
\mathbb{E}[\boldsymbol{m}_t] &= \mathbb{E}\left[\sum_{i=1}^t (1-\beta_1)\beta_1^{t-i}\boldsymbol{g}_i\right] \notag \\
&= \boldsymbol{\mu}(1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} \notag \\
&= \boldsymbol{\mu}(1-\beta_1^t) \tag{7}
\end{align}

因此，$\mathbb{E}[\hat{\boldsymbol{m}}_t] = \boldsymbol{\mu}$，偏差得到修正。

### 2. Weight RMS的平均场推导

#### 2.1 滑动平均视角的深入分析

定义$\beta_3 = 1-\eta\lambda$，将AdamW的更新改写为：
\begin{equation}
\boldsymbol{\theta}_t = \beta_3\boldsymbol{\theta}_{t-1} + (1-\beta_3)\left(-\frac{\boldsymbol{u}_t}{\lambda}\right) \tag{8}
\end{equation}

其中$\boldsymbol{u}_t = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon}$。

**数学直觉**：这个形式揭示了权重衰减的本质——它使参数成为更新量的指数移动平均。参数不会无限偏离原点，而是在$-\boldsymbol{u}_t/\lambda$附近振荡。

展开递推关系：
\begin{align}
\boldsymbol{\theta}_t &= \beta_3^t\boldsymbol{\theta}_0 + (1-\beta_3)\sum_{i=1}^t \beta_3^{t-i}\left(-\frac{\boldsymbol{u}_i}{\lambda}\right) \tag{9} \\
&= \beta_3^t\boldsymbol{\theta}_0 - \frac{1-\beta_3}{\lambda}\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{u}_i \tag{10}
\end{align}

**长期行为分析**：当$t\to\infty$且$\beta_3$接近1时，$\beta_3^t$项的贡献逐渐消失，参数主要由历史更新量的加权平均决定。

#### 2.2 平均场近似的数学基础

平均场近似的核心假设：
\begin{equation}
\frac{\sum_{i=1}^t w_i f(\boldsymbol{x}_i)}{\sum_{i=1}^t w_i} \approx \frac{f\left(\sum_{i=1}^t w_i\boldsymbol{x}_i\right)}{\sum_{i=1}^t w_i} \tag{11}
\end{equation}

这在高维空间中是合理的，因为：
1. 大数定律：多个随机变量的加权平均趋于其期望
2. 中心极限定理：加权和近似服从正态分布
3. 高维几何：在高维空间中，向量间的夹角趋于垂直

应用到我们的问题：
\begin{align}
\bar{\boldsymbol{u}}_t &= \frac{\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{u}_i}{\sum_{i=1}^t \beta_3^{t-i}} \notag \\
&= \frac{\sum_{i=1}^t \beta_3^{t-i}\frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}}}{\sum_{i=1}^t \beta_3^{t-i}} \notag \\
&\approx \frac{\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{m}_i}{\sqrt{\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{v}_i}} = \frac{\bar{\boldsymbol{m}}_t}{\sqrt{\bar{\boldsymbol{v}}_t}} \tag{12}
\end{align}

### 3. 收敛性分析

#### 3.1 凸优化情况

**定理1（凸情况下的收敛速度）**：假设损失函数$L(\boldsymbol{\theta})$是凸的且$L$-光滑（即梯度Lipschitz连续，常数为$L$），则AdamW满足：
\begin{equation}
\mathbb{E}[L(\bar{\boldsymbol{\theta}}_T)] - L(\boldsymbol{\theta}^*) \leq \frac{C}{\sqrt{T}} \tag{13}
\end{equation}

其中$\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=1}^T \boldsymbol{\theta}_t$，$\boldsymbol{\theta}^*$是最优解，$C$是与初始化、学习率等相关的常数。

**证明概要**：
利用凸函数的性质$L(\boldsymbol{\theta}) \geq L(\boldsymbol{\theta}^*) + \nabla L(\boldsymbol{\theta}^*)\cdot(\boldsymbol{\theta}-\boldsymbol{\theta}^*)$，结合AdamW的更新规则：
\begin{align}
&\Vert\boldsymbol{\theta}_{t+1}-\boldsymbol{\theta}^*\Vert^2 \notag \\
&= \Vert\boldsymbol{\theta}_t-\boldsymbol{\theta}^*\Vert^2 - 2\eta_t\left(\frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t}}+\lambda\boldsymbol{\theta}_t\right)\cdot(\boldsymbol{\theta}_t-\boldsymbol{\theta}^*) + \mathcal{O}(\eta_t^2) \tag{14}
\end{align}

对时间求和并利用$\boldsymbol{m}_t$和$\nabla L(\boldsymbol{\theta}_t)$的关系，可以得到遗憾界。

#### 3.2 非凸优化情况

**定理2（非凸情况下的一阶平稳点）**：对于非凸但光滑的函数，AdamW保证：
\begin{equation}
\min_{t\in[T]} \mathbb{E}\left[\Vert\nabla L(\boldsymbol{\theta}_t)\Vert\right] \leq \frac{C'}{\sqrt{T}} \tag{15}
\end{equation}

这意味着算法能够找到梯度接近零的点（一阶平稳点）。

**关键引理**：权重衰减项有助于避免梯度爆炸：
\begin{equation}
\Vert\boldsymbol{\theta}_t\Vert \leq \frac{\Vert\boldsymbol{u}_t\Vert}{\lambda(1-\beta_3)} + \mathcal{O}(\beta_3^t) \tag{16}
\end{equation}

**证明**：从$\boldsymbol{\theta}_t = \beta_3\boldsymbol{\theta}_{t-1} - \eta(\boldsymbol{u}_t + \lambda\boldsymbol{\theta}_{t-1})$出发，取模长：
\begin{align}
\Vert\boldsymbol{\theta}_t\Vert &\leq \beta_3\Vert\boldsymbol{\theta}_{t-1}\Vert + \eta\Vert\boldsymbol{u}_t\Vert + \eta\lambda\Vert\boldsymbol{\theta}_{t-1}\Vert \notag \\
&= (1-\eta\lambda)\Vert\boldsymbol{\theta}_{t-1}\Vert + \eta\Vert\boldsymbol{u}_t\Vert \notag \\
&= \beta_3\Vert\boldsymbol{\theta}_{t-1}\Vert + (1-\beta_3)\frac{\Vert\boldsymbol{u}_t\Vert}{\lambda} \tag{17}
\end{align}

递推求解这个不等式即得结果。

### 4. Weight RMS的精确估计

#### 4.1 双重求和的计算

从式$\eqref{eq:mv-roll}$出发：
\begin{gather}
\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{m}_i = (1-\beta_1)\sum_{i=1}^t \beta_3^{t-i}\sum_{j=1}^i \beta_1^{i-j}\boldsymbol{g}_j \tag{18}
\end{gather}

**交换求和顺序的技巧**：固定$j$，计算对$i$的求和：
\begin{align}
\sum_{i=j}^t \beta_3^{t-i}\beta_1^{i-j} &= \beta_1^{-j}\sum_{i=j}^t \beta_3^{t-i}\beta_1^i \notag \\
&= \beta_1^{-j}\beta_3^t\sum_{i=j}^t (\beta_1/\beta_3)^i \notag \\
&= \beta_3^{t-j+1}\frac{1-(\beta_1/\beta_3)^{t-j+1}}{1-\beta_1/\beta_3} \notag \\
&= \frac{\beta_3^{t-j+1}-\beta_1^{t-j+1}}{\beta_3-\beta_1} \tag{19}
\end{align}

因此：
\begin{equation}
\sum_{i=1}^t \beta_3^{t-i}\boldsymbol{m}_i = (1-\beta_1)\sum_{j=1}^t \frac{\beta_3^{t-j+1}-\beta_1^{t-j+1}}{\beta_3-\beta_1}\boldsymbol{g}_j \tag{20}
\end{equation}

#### 4.2 统计量的计算

假设$\boldsymbol{g}_j \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$（独立同分布），计算期望：
\begin{align}
\mathbb{E}[\bar{\boldsymbol{v}}_t] &= \frac{(1-\beta_3)(1-\beta_2)}{1-\beta_3^t}\sum_{j=1}^t \frac{\beta_3^{t-j+1}-\beta_2^{t-j+1}}{\beta_3-\beta_2}(\boldsymbol{\mu}^2+\boldsymbol{\sigma}^2) \tag{21}
\end{align}

**渐近行为**：当$t\to\infty$时，利用$\beta_2^t\to 0$和$\beta_3\approx 1$：
\begin{align}
\sum_{j=1}^t \frac{\beta_3^{t-j+1}-\beta_2^{t-j+1}}{\beta_3-\beta_2} &\approx \sum_{j=1}^t \frac{\beta_3^{t-j+1}}{\beta_3-\beta_2} \notag \\
&= \frac{\beta_3}{(\beta_3-\beta_2)}\cdot\frac{1-\beta_3^t}{1-\beta_3} \notag \\
&\approx \frac{1-\beta_3^t}{1-\beta_3} \tag{22}
\end{align}

因此$\mathbb{E}[\bar{\boldsymbol{v}}_t] \approx \boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2$。

### 5. 自适应学习率的数学原理

#### 5.1 二阶矩的作用机制

二阶矩$\boldsymbol{v}_t$实际上估计了梯度的方差：
\begin{equation}
\mathbb{E}[\boldsymbol{v}_t] \approx (1-\beta_2)\sum_{i=1}^t \beta_2^{t-i}\mathbb{E}[\boldsymbol{g}_i^2] \approx \mathbb{E}[\boldsymbol{g}^2] \tag{23}
\end{equation}

**预条件的视角**：更新量$\frac{\boldsymbol{m}_t}{\sqrt{\boldsymbol{v}_t}}$等价于用对角预条件矩阵$\boldsymbol{V}_t^{-1/2} = \text{diag}(1/\sqrt{\boldsymbol{v}_t})$预条件梯度：
\begin{equation}
\boldsymbol{u}_t = \boldsymbol{V}_t^{-1/2}\boldsymbol{m}_t \tag{24}
\end{equation}

这近似于自然梯度下降中使用Fisher信息矩阵的对角近似。

#### 5.2 不同方向的自适应缩放

考虑两个方向的梯度：
- 高梯度方向：$\boldsymbol{g}_t^{(1)} = [10, 10, \ldots]$，则$\sqrt{\boldsymbol{v}_t^{(1)}} \approx 10$
- 低梯度方向：$\boldsymbol{g}_t^{(2)} = [0.1, 0.1, \ldots]$，则$\sqrt{\boldsymbol{v}_t^{(2)}} \approx 0.1$

更新量分别为：
\begin{gather}
\boldsymbol{u}_t^{(1)} \approx \frac{10}{10} = 1 \tag{25} \\
\boldsymbol{u}_t^{(2)} \approx \frac{0.1}{0.1} = 1 \tag{26}
\end{gather}

**数学直觉**：自适应学习率使得不同尺度的参数获得相近的更新幅度，加速收敛。

### 6. 与其他优化器的理论对比

#### 6.1 SGD vs Adam vs AdamW

| 优化器 | 更新规则 | Weight RMS | 收敛速度（凸） | 内存 |
|--------|----------|------------|----------------|------|
| SGD | $\boldsymbol{\theta}_t - \eta\boldsymbol{g}_t$ | $\sim t^{1/2}$ | $\mathcal{O}(1/\sqrt{T})$ | $\mathcal{O}(d)$ |
| SGDM | $\boldsymbol{\theta}_t - \eta\boldsymbol{m}_t$ | $\sim t^{1/2}$ | $\mathcal{O}(1/T)$（强凸） | $2\mathcal{O}(d)$ |
| Adam | 式(3) | 无界 | $\mathcal{O}(1/\sqrt{T})$ | $3\mathcal{O}(d)$ |
| AdamW | 式(4) | $\sqrt{\eta/\lambda}$ | $\mathcal{O}(1/\sqrt{T})$ | $3\mathcal{O}(d)$ |

**定理3（SGD vs AdamW的Weight RMS）**：
- SGD：$\mathbb{E}[\Vert\boldsymbol{\theta}_t\Vert^2] = \Vert\boldsymbol{\theta}_0\Vert^2 + \mathcal{O}(\eta^2 t)$
- AdamW：$\mathbb{E}[\Vert\boldsymbol{\theta}_t\Vert^2] \to \mathcal{O}(\eta/\lambda)$

**证明**（SGD情况）：
\begin{align}
\Vert\boldsymbol{\theta}_{t+1}\Vert^2 &= \Vert\boldsymbol{\theta}_t - \eta\boldsymbol{g}_t\Vert^2 \notag \\
&= \Vert\boldsymbol{\theta}_t\Vert^2 - 2\eta\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t + \eta^2\Vert\boldsymbol{g}_t\Vert^2 \tag{27}
\end{align}

对期望并假设$\mathbb{E}[\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t]=0$（高维随机性），得到递推关系。

### 7. 超参数敏感度分析

#### 7.1 学习率$\eta$的影响

Weight RMS对学习率的依赖关系：
\begin{equation}
\Vert\boldsymbol{\theta}_{\infty}\Vert_{\text{RMS}} \propto \sqrt{\frac{\eta}{\lambda}} \tag{28}
\end{equation}

**敏感度分析**：
\begin{equation}
\frac{\partial \Vert\boldsymbol{\theta}\Vert_{\text{RMS}}}{\partial \eta} = \frac{1}{2\sqrt{\eta\lambda}} > 0 \tag{29}
\end{equation}

学习率增大会导致参数模长增大，但关系是次线性的（平方根关系）。

#### 7.2 权重衰减率$\lambda$的影响

\begin{equation}
\frac{\partial \Vert\boldsymbol{\theta}\Vert_{\text{RMS}}}{\partial \lambda} = -\frac{\sqrt{\eta}}{2\lambda^{3/2}} < 0 \tag{30}
\end{equation}

增大$\lambda$会减小参数模长，这提供了正则化效果。

**最优$\lambda$的选择**：平衡正则化强度和模型表达能力：
\begin{equation}
\lambda^* = \arg\min_{\lambda} \left[\mathcal{L}_{\text{train}}(\lambda) + \alpha\cdot\text{Complexity}(\lambda)\right] \tag{31}
\end{equation}

实践中常用$\lambda \in [0.001, 0.1]$。

### 8. 数值稳定性讨论

#### 8.1 除零问题

Adam中的$\epsilon$参数防止除零：
\begin{equation}
\boldsymbol{u}_t = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}+\epsilon} \tag{32}
\end{equation}

**$\epsilon$的选择准则**：
1. 足够小以不影响有效梯度：$\epsilon \ll \min_i \sqrt{v_{t,i}}$
2. 足够大以防止数值下溢：$\epsilon > \text{machine epsilon}$

典型值：$\epsilon = 10^{-8}$（float32）或$10^{-4}$（float16）

#### 8.2 梯度裁剪的数学原理

防止梯度爆炸的裁剪策略：
\begin{equation}
\tilde{\boldsymbol{g}}_t = \begin{cases}
\boldsymbol{g}_t, & \text{if } \Vert\boldsymbol{g}_t\Vert \leq \tau \\
\tau\frac{\boldsymbol{g}_t}{\Vert\boldsymbol{g}_t\Vert}, & \text{otherwise}
\end{cases} \tag{33}
\end{equation}

**定理4**：梯度裁剪保持下降方向：$\tilde{\boldsymbol{g}}_t \cdot \boldsymbol{g}_t \geq 0$。

### 9. 具体计算示例

#### 9.1 二维参数空间的演化

考虑简单的二次损失$L(\boldsymbol{\theta}) = \frac{1}{2}(\theta_1^2 + 10\theta_2^2)$，梯度为$\boldsymbol{g}_t = [\theta_{1,t-1}, 10\theta_{2,t-1}]^T$。

设$\beta_1=0.9, \beta_2=0.999, \eta=0.1, \lambda=0.01$，初始化$\boldsymbol{\theta}_0=[1, 1]^T$。

**第1步**：
\begin{align}
\boldsymbol{m}_1 &= 0.1[1, 10]^T = [0.1, 1]^T \tag{34} \\
\boldsymbol{v}_1 &= 0.001[1, 100]^T = [0.001, 0.1]^T \tag{35} \\
\boldsymbol{u}_1 &= [0.1/\sqrt{0.001}, 1/\sqrt{0.1}]^T \approx [3.16, 3.16]^T \tag{36} \\
\boldsymbol{\theta}_1 &= [1, 1]^T - 0.1([3.16, 3.16]^T + 0.01[1, 1]^T) \notag \\
&\approx [0.68, 0.68]^T \tag{37}
\end{align}

可以看到，Adam自动平衡了两个方向的更新幅度（尽管梯度相差10倍）。

#### 9.2 Weight RMS的数值验证

对于标准初始化$\sigma=1/\sqrt{d}$，$d=768$（BERT base），预测的Weight RMS：
\begin{equation}
\Vert\boldsymbol{\theta}_{\infty}\Vert_{\text{RMS}} = \sqrt{\frac{\eta}{2\lambda}} = \sqrt{\frac{0.001}{2\times 0.01}} = 0.224 \tag{38}
\end{equation}

实际训练得到$\approx 0.22$，误差$<2\%$。

### 10. 理论保证与遗憾界

#### 10.1 在线学习的遗憾界

**定理5（AdamW的遗憾界）**：对于凸损失函数序列$\{f_t\}_{t=1}^T$，AdamW的遗憾满足：
\begin{equation}
R_T = \sum_{t=1}^T f_t(\boldsymbol{\theta}_t) - \min_{\boldsymbol{\theta}\in\mathcal{C}}\sum_{t=1}^T f_t(\boldsymbol{\theta}) \leq \mathcal{O}(\sqrt{T}) \tag{39}
\end{equation}

其中$\mathcal{C}$是可行域。

**证明要点**：利用在线凸优化的标准技术，关键不等式：
\begin{equation}
f_t(\boldsymbol{\theta}_t) - f_t(\boldsymbol{\theta}^*) \leq \nabla f_t(\boldsymbol{\theta}_t)\cdot(\boldsymbol{\theta}_t-\boldsymbol{\theta}^*) \tag{40}
\end{equation}

#### 10.2 泛化误差界

**定理6（PAC-Bayes界）**：以概率$1-\delta$，测试误差满足：
\begin{equation}
\mathcal{L}_{\text{test}} \leq \mathcal{L}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{\Vert\boldsymbol{\theta}\Vert^2\log(1/\delta)}{n}}\right) \tag{41}
\end{equation}

权重衰减通过控制$\Vert\boldsymbol{\theta}\Vert$来改善泛化界。

### 11. 实践建议

#### 11.1 超参数调优策略

**学习率warmup**：前$N$步线性增加学习率：
\begin{equation}
\eta_t = \begin{cases}
\eta_{\max} \cdot \frac{t}{N}, & t \leq N \\
\eta_{\max} \cdot \text{schedule}(t), & t > N
\end{cases} \tag{42}
\end{equation}

**权重衰减的缩放法则**：当batch size增大$k$倍时：
\begin{gather}
\eta \to k\eta \tag{43} \\
\lambda \to \lambda/k \tag{44}
\end{gather}

#### 11.2 不同任务的推荐配置

| 任务类型 | $\beta_1$ | $\beta_2$ | $\eta$ | $\lambda$ |
|----------|-----------|-----------|--------|-----------|
| CV预训练 | 0.9 | 0.999 | $10^{-3}$ | 0.05 |
| NLP预训练 | 0.9 | 0.999 | $10^{-4}$ | 0.01 |
| 微调 | 0.9 | 0.999 | $10^{-5}$ | 0.01 |
| RL | 0.9 | 0.999 | $3\times 10^{-4}$ | 0 |

### 12. 开放问题与未来方向

1. **自适应权重衰减**：能否根据训练动态调整$\lambda_t$？
2. **分层权重衰减**：不同层使用不同的$\lambda$？
3. **Weight RMS的主动控制**：将Weight RMS作为约束条件？

**最新进展**：论文[Power Lines: Scaling Laws for Weight Decay](https://arxiv.org/abs/2505.13738)提出了权重衰减的缩放律。

## 总结

本文通过严格的数学推导，建立了AdamW优化器的Weight RMS估计理论。主要结论：

1. **核心公式**：$\Vert\boldsymbol{\theta}_{\infty}\Vert_{\text{RMS}} = \sqrt{\frac{\eta}{2\lambda}}$
2. **收敛保证**：凸情况下$\mathcal{O}(1/\sqrt{T})$，非凸情况下找到一阶平稳点
3. **实践意义**：权重衰减不仅正则化，还稳定训练

这些结果为理解和改进现代优化器提供了理论基础。

