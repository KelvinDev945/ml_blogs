---
title: Softmax后传：寻找Top-K的光滑近似
slug: softmax后传寻找top-k的光滑近似
date: 2024-09-19
tags: 概率, 近似, 梯度, 光滑, 生成模型
status: pending
---

# Softmax后传：寻找Top-K的光滑近似

**原文链接**: [https://spaces.ac.cn/archives/10373](https://spaces.ac.cn/archives/10373)

**发布日期**: 

---

Softmax，顾名思义是“soft的max”，是$\max$算子（准确来说是$\text{argmax}$）的光滑近似，它通过指数归一化将任意向量$\boldsymbol{x}\in\mathbb{R}^n$转化为分量非负且和为1的新向量，并允许我们通过温度参数来调节它与$\text{argmax}$（的one hot形式）的近似程度。除了指数归一化外，我们此前在[《通向概率分布之路：盘点Softmax及其替代品》](/archives/10145)也介绍过其他一些能实现相同效果的方案。

我们知道，最大值通常又称Top-1，它的光滑近似方案看起来已经相当成熟，那读者有没有思考过，一般的Top-$k$的光滑近似又是怎么样的呢？下面让我们一起来探讨一下这个问题。

## 问题描述 #

设向量$\boldsymbol{x}=(x_1,x_2,\cdots,x_n)\in\mathbb{R}^n$，简单起见我们假设它们两两不相等，即$i\neq j \Leftrightarrow x_i\neq x_j$。记$\Omega_k(\boldsymbol{x})$为$\boldsymbol{x}$最大的$k$个分量的下标集合，即$|\Omega_k(\boldsymbol{x})|=k$以及$\forall i\in \Omega_k(\boldsymbol{x}), j \not\in \Omega_k(\boldsymbol{x})\Rightarrow x_i > x_j$。我们定义Top-$k$算子$\mathcal{T}_k$为$\mathbb{R}^n\mapsto\\{0,1\\}^n$的映射：  
\begin{equation}  
[\mathcal{T}_k(\boldsymbol{x})]_i = \left\\{\begin{aligned}1,\,\, i\in \Omega_k(\boldsymbol{x}) \\\ 0,\,\, i \not\in \Omega_k(\boldsymbol{x})\end{aligned}\right.  
\end{equation}  
说白了，如果$x_i$属于最大的$k$个元素之一，那么对应的位置变成1，否则变成0，最终结果是一个Multi-Hot向量，比如$\mathcal{T}_2([3,2,1,4]) = [1,0,0,1]$。

从$\boldsymbol{x}$到$\mathcal{T}_k(\boldsymbol{x})$实际上是一种硬指派（Hard Assignment）运算，它本质上是不连续的，没有保留关于$\boldsymbol{x}$的有效梯度，因此无法集成到模型中进行端到端训练。为了解决这个问题，我们就需要构造一个能够提供有效梯度信息的$\mathcal{T}_k(\boldsymbol{x})$的光滑近似——在有些文献中也称为“可微Top-$k$算子（Differentiable Top-$k$ Operator）”。

具体来说，我们定义集合  
\begin{equation}\Delta_k^{n-1} = \left\\{\boldsymbol{p}=(p_1,p_2,\cdots,p_n)\left|\, p_1,p_2,\cdots,p_n\in[0,1],\sum_{i=1}^n p_i = k\right.\right\\}\end{equation}  
那么我们要做的事情就是构建$\mathbb{R}^n\mapsto \Delta_k^{n-1}$的一个映射$\mathcal{ST}_k(\boldsymbol{x})$，并尽量满足如下性质  
\begin{align}&{\color{red}{单调性}}:\quad [\mathcal{ST}_k(\boldsymbol{x})]_i \geq [\mathcal{ST}_k(\boldsymbol{x})]_j \,\,\Leftrightarrow\,\, x_i \geq x_j \\\\[8pt]  
&{\color{red}{不变性}}:\quad \mathcal{ST}_k(\boldsymbol{x}) = \mathcal{ST}_k(\boldsymbol{x} + c),\,\,\forall c\in\mathbb{R} \\\\[8pt]  
&{\color{red}{趋近性}}:\quad \lim_{\tau\to 0^+}\mathcal{ST}_k(\boldsymbol{x}/\tau) = \mathcal{T}_k(\boldsymbol{x}) \\\  
\end{align}  
可以验证作为$\mathcal{ST}_1(\boldsymbol{x})$的Softmax是满足如上性质的，所以提出上述性质实际就是希望构建出来的$\mathcal{ST}_k(\boldsymbol{x})$能够成为Softmax的自然推广。当然，构建Top-$k$的光滑近似本身就比Top-1的要难，所以如果遇到困难的话，不必要严格遵循以上性质，只要能表明所构建的映射确实具备$\mathcal{T}_k(\boldsymbol{x})$的光滑近似的特性就行。

## 迭代构造 #

事实上，笔者很早之前就关注过该问题，首次讨论于2019年的文章[《函数光滑化杂谈：不可导函数的可导逼近》](/archives/6620)中，当时称之为$\text{soft-}k\text{-max}$，并给出过一个迭代构造方案：

> 输入为$\boldsymbol{x}$，初始化$\boldsymbol{p}^{(0)}$为全0向量；  
>  执行$\boldsymbol{x} = \boldsymbol{x} - \min(\boldsymbol{x})$（保证所有元素非负）
> 
> 对于$i=1,2,\dots,k$，执行：  
>  $\boldsymbol{y} = (1 - \boldsymbol{p}^{(i-1)})\otimes\boldsymbol{x}$;  
>  $\boldsymbol{p}^{(i)} = \boldsymbol{p}^{(i-1)} + \text{softmax}(\boldsymbol{y})$
> 
> 返回$\boldsymbol{p}^{(k)}$。

其实这个迭代的构造思路很简单，我们可以先将$\text{softmax}(\boldsymbol{y})$替换为$\mathcal{T}_1(\boldsymbol{y})$来理解，此时算法流程就是先保证分量非负，然后识别出Top-1，再将Top-1置零（最大值变成了最小值），接着识别出剩下的Top-1，依此类推，最终的$\boldsymbol{p}_k$正好就是$\mathcal{T}_k(\boldsymbol{x})$。而$\text{softmax}(\boldsymbol{y})$作为$\mathcal{T}_1(\boldsymbol{y})$的光滑近似，在迭代过程中使用$\text{softmax}(\boldsymbol{y})$自然也就得到$\mathcal{T}_k(\boldsymbol{x})$的光滑近似了。

无独有偶，笔者发现在Stack Exchange上的提问[《Is there something like softmax but for top k values?》](https://stats.stackexchange.com/a/454788)中有回复者提出过一个相同思路的方案，它先定义了加权Softmax：  
\begin{equation}[\text{softmax}(\boldsymbol{x};\boldsymbol{w})]_i = \frac{w_i e^{x_i}}{\sum\limits_{i=1}^n w_i e^{x_i}}\end{equation}  
然后构建的迭代过程为

> 输入为$\boldsymbol{x}$，初始化$\boldsymbol{p}^{(0)}$为全0向量；
> 
> 对于$i=1,2,\dots,k$，执行：  
>  $\boldsymbol{p}^{(i)} = \boldsymbol{p}^{(i-1)} + \text{softmax}(\boldsymbol{x}; 1 - \boldsymbol{p}^{(i-1)})$
> 
> 返回$\boldsymbol{p}^{(k)}$。

这跟笔者所提的迭代过程在思路上是完全一样的，只不过笔者把$1 - \boldsymbol{p}_{i-1}$乘到了$\boldsymbol{x}$上而它则乘到了$e^{\boldsymbol{x}}$上，借助$e^{\boldsymbol{x}}$本身的非负性简化了流程。然而，这个迭代实际上是错误的，它 _不符合“趋近性”_ ，比如当$k=2$时，代入$\boldsymbol{x}/\tau$取$\tau\to 0^+$的极限并不是Multi-Hot向量，而是最大值变为1.5、次大值变为0.5、其余都变为0的向量，这是因为$1-p_{\max}$跟$e^{-x_{\max}}$大致上是同阶的，$1-p_{\max}$乘到$e^{x_{\max}}$上并不能完全消除最大值。

## 作为梯度 #

迭代构造全凭经验，可能会隐藏一些难以发现的问题，比如看起来更简单的加权Softmax迭代实际上不合理。由于没有更贴近本质的原理指导，这些方案往往也难以理论分析，比如笔者构造的迭代，虽然测起来没有问题，但我们很难证明$\boldsymbol{p}_k$分量都在$[0,1]$内，也很难判断它是否满足单调性。

所以，我们希望得到一个更高观点的原理去指导和设计这个光滑近似。就在前几天，笔者突然意识到一个关键的事实  
\begin{equation}\mathcal{T}_k(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \sum_{i\in\Omega_k(\boldsymbol{x})} x_i\end{equation}  
也就是说，最大的$k$个分量的和，它的梯度正好是$\mathcal{T}_k(\boldsymbol{x})$。所以我们似乎可以改为找$\sum\limits_{i\in\Omega_k(\boldsymbol{x})} x_i$的光滑近似，然后求梯度就得到$\mathcal{T}_k(\boldsymbol{x})$的光滑近似了，前者是一个标量，它的光滑近似找起来更容易一些，比如利用恒等式  
\begin{equation}\sum_{i\in\Omega_k(\boldsymbol{x})} x_i = \max_{i_1 < \cdots < i_k} (x_{i_1} + \cdots + x_{i_k})\end{equation}  
即遍历所有$k$个分量之和取最大值，这样一来，问题变成了找$\max$的光滑近似，这个我们早已解决（参考[《寻求一个光滑的最大值函数》](/archives/3290)），答案是$\text{logsumexp}$：  
\begin{equation}\max_{i_1 < \cdots < i_k} (x_{i_1} + \cdots + x_{i_k})\approx \log\sum_{i_1 < \cdots < i_k} e^{x_{i_1} + \cdots + x_{i_k}}\triangleq \log Z_k\end{equation}  
对它求梯度，我们就得到$\mathcal{ST}_k(\boldsymbol{x})$的一个形式：  
\begin{equation}[\mathcal{ST}_k(\boldsymbol{x})]_i = \frac{\sum\limits_{i_2 < \cdots < i_k} e^{x_i+x_{i_2} + \cdots + x_{i_k}}}{\sum\limits_{i_1 < \cdots < i_k} e^{x_{i_1} +x_{i_2}+ \cdots + x_{i_k}}}\triangleq \frac{Z_{k,i}}{Z_k}\label{eq:k-max-grad}\end{equation}  
分母是所有$k$分量和的指数和，分子则是所有包含$x_i$的$k$分量和的指数和。根据这个形式，我们可以轻松证明  
\begin{equation}0 < [\mathcal{ST}_k(\boldsymbol{x})]_i < 1,\quad \sum_{i=1}^n [\mathcal{ST}_k(\boldsymbol{x})]_i = k\end{equation}  
所以这样定义的$\mathcal{ST}_k(\boldsymbol{x})$确实是属于$\Delta_k^{n-1}$的。事实上，我们还可以证明它同时满足单调性、不变性和趋近性，并且$\mathcal{ST}_1(\boldsymbol{x})$它就是Softmax，这些特性显示它确实是Softmax对于Top-$k$算子的自然推广，我们暂且称之为“**GradTopK** （Gradient-guided Soft Top-k operator）”。

不过还没到庆祝时刻，因为式$\eqref{eq:k-max-grad}$的数值计算问题还没解决。如果直接按照式$\eqref{eq:k-max-grad}$来计算的话，分母就涉及到$C_n^k$项指数求和，计算量非常可观，所以必须找到一个高效的计算方法。我们已经分别记了分子、分母为$Z_{k,i},Z_k$，可以观察到分子$Z_{k,i}$满足递归式  
\begin{equation}Z_{k,i} = e^{x_i}(Z_{k-1} - Z_{k-1,i})\end{equation}  
结合$Z_{k,i}$对$i$求和等于$kZ_k$的事实，我们可以构建一个递归计算过程：  
\begin{equation}\begin{aligned}  
\log Z_{k,i} =&\, x_i + \log(e^{\log Z_{k-1}} - e^{\log Z_{k-1,i}}) \\\  
\log Z_k =&\, \left(\log\sum_{i=1}^n e^{\log Z_{k,i}}\right) - \log k \\\  
\end{aligned}\end{equation}  
其中$\log Z_{1,i} = x_i$，为了减少溢出风险，我们对两端都取了对数。现在只需要迭代$k$步就可以完成$\mathcal{ST}_k(\boldsymbol{x})$的计算，效率上是可以接受的。然而，即便做了对数化处理，上述递归也只能对小方差的$\boldsymbol{x}$或者比较小的$k$算一下，反之$\log Z_{k-1}$与最大的$\log Z_{k-1,i}$就会相当接近，当数值上无法区分时就会出现$\log 0$的Bug，个人认为这是这种递归转化的根本困难。

一个非常简陋的参考实现：
    
    
    import numpy as np
    
    def GradTopK(x, k):
        for i in range(1, k + 1):
            logZs = x if i == 1 else x + logZ + np.log(1 - np.exp(logZs - logZ))
            logZ = np.logaddexp.reduce(logZs) - np.log(i)
        return np.exp(logZs - logZ)
    
    k, x = 10, np.random.randn(100)
    GradTopK(x, k)

## 待定常数 #

上一节通过梯度来构建Top-$k$光滑近似的思路，确实能给人一种高屋建瓴的美感，但可能也会有读者觉得它过于抽象，缺乏一种由表及里的直观感，同时对于大方差的$\boldsymbol{x}$或者比较大的$k$的数值不稳定性，也让我们难以完全满意现有结果。所以，接下来我们将探究一种自下而上的构建思路。

### 方法大意 #

该思路来源于Stack Exchange上的另一个帖子[《Differentiable top-k function》](https://math.stackexchange.com/a/4506773)的回复。设$f(x)$是任意$\mathbb{R}\mapsto [0,1]$的、光滑的、单调递增的函数，并且满足$\lim\limits_{x\to\infty}f(x) = 1,\lim\limits_{x\to-\infty}f(x) = 0$。看上去条件很多，但实际上这种函数构造起来没有什么难度，比如经典的Sigmoid函数$\sigma(x)=1/(1+e^{-x})$，还有$\text{clip}(x,0,1)$、$\min(1, e^x)$等。接着我们考虑  
\begin{equation}f(\boldsymbol{x}) = [f(x_1),f(x_2),\cdots,f(x_n)]\end{equation}  
$f(\boldsymbol{x})$跟我们想要的$\mathcal{ST}_k(\boldsymbol{x})$差多远呢？每个分量都在$[0,1]$内肯定是满足了，但是分量之和等于$k$无法保证，所以我们引入一个跟$\boldsymbol{x}$相关的待定常数$\lambda(x)$来保证这一点：  
\begin{equation}\mathcal{ST}_k(\boldsymbol{x}) \triangleq f(\boldsymbol{x} - \lambda(\boldsymbol{x})),\quad \sum_{i=1}^n f(x_i - \lambda(\boldsymbol{x})) = k\end{equation}  
也就是通过分量之和为$k$来反解出$\lambda(\boldsymbol{x})$，我们可以称之为“**ThreTopK** （Threshold-adjusted Soft Top-k operator）”，如果读者已经阅读过[《通向概率分布之路：盘点Softmax及其替代品》](/archives/10145)，就会发现这个做法跟Sparsemax、Entmax-$\alpha$是一样的。

ThreTopK会是我们理想的$\mathcal{ST}_k(\boldsymbol{x})$吗？还真是！首先我们假设了$f$的单调性，所以单调性是满足的，其次$f(\boldsymbol{x} - \lambda(\boldsymbol{x}))=f(\boldsymbol{x}+c - (c+\lambda(\boldsymbol{x})))$，也就是常数可以收纳到$\lambda(\boldsymbol{x})$里边，所以不变性也是满足的。最后，当$\tau\to 0^+$时，我们可以找到一个适当的阈值$\lambda(\boldsymbol{x}/\tau)$，使得$\boldsymbol{x}/\tau-\lambda(\boldsymbol{x}/\tau)$最大的$k$个分量趋于$\infty$，剩下的分量趋于$-\infty$，从而$f(\boldsymbol{x}/\tau-\lambda(\boldsymbol{x}/\tau))$就等于$\mathcal{T}_k(\boldsymbol{x})$，也就是说满足趋近性。

### 解析求解 #

既然已经证明了ThreTopK的理论优越性，那么接下来要解决的就是$\lambda(\boldsymbol{x})$的计算了，这个大部份情况下都只能诉诸数值计算方法，不过对于$f(x)=\min(1, e^x)$，我们可以求得一个解析解。

求解的思路跟之前的Sparsemax是一样的。不失一般性，假设$\boldsymbol{x}$的分量已经从大到小排好顺序，即$x_1 > x_2 > \cdots > x_n$，接着假设我们已经知道$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$，那么此时  
\begin{equation}k = \sum_{i=1}^n \min(1, e^{x_i - \lambda(\boldsymbol{x})}) = m + \sum_{i=m+1}^n e^{x_i - \lambda(\boldsymbol{x})}\end{equation}  
由此解得  
\begin{equation}\lambda(\boldsymbol{x})=\log\left(\sum_{i=m+1}^n e^{x_i}\right) - \log(k-m)\end{equation}  
由此我们可以看出，当$k=1$时，$m$只能取$0$，此时可以发现ThreTopK正好就是Softmax；当$k > 1$时，我们无法事先确定$m$的值，所以只能遍历$m=0,1,\cdots,k-1$，根据上式计算$\lambda(\boldsymbol{x})$，寻找满足$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$的$\lambda(\boldsymbol{x})$。下面同样是一个非常简陋的参考实现：
    
    
    import numpy as np
    
    def ThreTopK(x, k):
        x_sort = np.sort(x)
        x_lamb = np.logaddexp.accumulate(x_sort)[-k:] - np.log(np.arange(k) + 1)
        x_sort_shift = np.pad(x_sort[-k:][1:], (0, 1), constant_values=np.inf)
        lamb = x_lamb[(x_lamb <= x_sort_shift) & (x_lamb >= x_sort[-k:])]
        return np.clip(np.exp(x - lamb), 0, 1)
    
    k, x = 10, np.random.randn(100)
    ThreTopK(x, k)

### 通用结果 #

从原理和代码都可以看出，$f(x)=\min(1, e^x)$时的ThreTopK几乎不会出现数值稳定性问题，并且$k=1$时能退化为Softmax，这些都是它的优势。然而，$\min(1, e^x)$实际上也算不上完全光滑（除了当$k=1$时$\min$不起作用），它在$x=0$处是不可导的。如果介意这一点，那么我们就需要选择处处可导的$f(x)$，比如$\sigma(x)$。

下面我们以$f(x)=\sigma(x)$为例，此时我们无法求出$\lambda(\boldsymbol{x})$的解析解，不过由于$\sigma(x)$的单调递增性，所以函数  
\begin{equation}F(\lambda)\triangleq \sum_{i=1}^n \sigma(x_i - \lambda)\end{equation}  
关于$\lambda$是单调递减的，因此$F(\lambda(\boldsymbol{x}))=k$的数值求解并不困难，二分法、牛顿法均可。以二分法为例，不难看出$\lambda(\boldsymbol{x})\in[x_{\min} - \sigma^{-1}(k/n), x_{\max} - \sigma^{-1}(k/n)]$，这里$\sigma^{-1}$是$\sigma$的逆函数，从这个初始区间出发，逐步二分到指定精度即可：
    
    
    import numpy as np
    
    def sigmoid(x):
        y = np.exp(-np.abs(x))
        return np.where(x >= 0, 1, y) / (1 + y)
    
    def sigmoid_inv(x):
        return np.log(x / (1 - x))
    
    def ThreTopK(x, k, epsilon=1e-4):
        low = x.min() - sigmoid_inv(k / len(x))
        high = x.max() - sigmoid_inv(k / len(x))
        while high - low > epsilon:
            lamb = (low + high) / 2
            Z = sigmoid(x - lamb).sum()
            low, high = (low, lamb) if Z < k else (lamb, high)
        return sigmoid(x - lamb)
    
    k, x = 10, np.random.randn(100)
    ThreTopK(x, k)

因此，$\lambda(\boldsymbol{x})$的数值计算并没有太大困难，真正的困难是当我们用数值方法去计算$\lambda(\boldsymbol{x})$时，往往会丢失$\lambda(\boldsymbol{x})$关于$\boldsymbol{x}$的梯度，从而影响端到端的训练。针对这个问题，我们可以手动把$\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})$算出来，然后自定义反向传播过程。具体来说，我们对  
\begin{equation}\sum_{i=1}^n \sigma(x_i - \lambda(\boldsymbol{x})) = k\end{equation}  
两边求某个$x_j$的偏导数，得到  
\begin{equation}\sigma'(x_j - \lambda(\boldsymbol{x}))-\sum_{i=1}^n \sigma'(x_i - \lambda(\boldsymbol{x}))\frac{\partial\lambda(\boldsymbol{x})}{\partial x_j} = 0\end{equation}  
那么  
\begin{equation}\frac{\partial\lambda(\boldsymbol{x})}{\partial x_j} = \frac{\sigma'(x_j - \lambda(\boldsymbol{x}))}{\sum\limits_{i=1}^n \sigma'(x_i - \lambda(\boldsymbol{x}))}\end{equation}  
其中$\sigma'$是$\sigma$的导函数。现在我们有了$\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})$的表达式，它的每一项都是可计算的（$\lambda(\boldsymbol{x})$也已经由数值方法求出），我们可以直接指定它作为反向传播的结果。一个比较简单且通用的实现方法是利用$\text{stop_gradient}$（下面简称$\text{sg}$）技巧，即在实现模型时将$\lambda(\boldsymbol{x})$替换为  
\begin{equation}\boldsymbol{x}\cdot\text{sg}[\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})] + \text{sg}[\lambda(\boldsymbol{x}) - \boldsymbol{x}\cdot\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})]\end{equation}  
其中$\cdot$是向量的内积。这样一来，前向传播时等价于$\text{sg}$不存在，所以结果是$\lambda(\boldsymbol{x})$，反向传播时被$\text{sg}$的部份梯度都是零，所以梯度就是给定的$\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})$，这样我们就自定义了$\lambda(\boldsymbol{x})$的梯度，跟$\lambda(\boldsymbol{x})$如何计算得到的无关。

### 二者兼之 #

现在我们看到，$f(x)=\min(1,e^x)$有解析解，但它并非全局光滑；$f(x)=\sigma(x)$倒是足够光滑了，但它求解起来比较复杂。有没有兼顾两者优点的选择呢？还真有，笔者发现下述的$f(x)$是全局光滑并且$\lambda(\boldsymbol{x})$可以解析求解的：  
\begin{equation}f(x) = \left\\{\begin{aligned}1 - e^{-x}/2,\quad x\geq 0 \\\ e^x / 2,\quad x < 0\end{aligned}\right.\end{equation}  
也可以写成$f(x) = (1 - e^{-|x|})\text{sign}(x)/2+1/2$。可以验证$f(x)$其实也是一个S型函数，虽然它是分段函数，但它本身及其导函数在$x=0$处都是连续的，因此足够光滑了。

求解思路跟之前一样，不失一般性设$x_1 > x_2 > \cdots > x_n$，假设已经知道$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$，那么  
\begin{equation}\begin{aligned}  
k =&\, \sum_{i=1}^m (1 - e^{-(x_i - \lambda(\boldsymbol{x}))}/2) + \sum_{i=m+1}^n e^{x_i - \lambda(\boldsymbol{x})}/2 \\\  
=&\, m - \frac{1}{2}e^{\lambda(\boldsymbol{x})}\sum_{i=1}^m e^{-x_i} + \frac{1}{2}e^{-\lambda(\boldsymbol{x})}\sum_{i=m+1}^n e^{x_i}  
\end{aligned}\end{equation}  
由此解得  
\begin{equation}\lambda(\boldsymbol{x})=\log\sum_{i=m+1}^n e^{x_i} - \log\left(\sqrt{(k-m)^2 + \left(\sum_{i=1}^m e^{-x_i}\right)\left(\sum_{i=m+1}^n e^{x_i}\right)}+(k-m)\right)\end{equation}  
然后遍历$m=0,1,\cdots,n-1$，寻找满足$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$的$\lambda(\boldsymbol{x})$即可。读者还可以尝试证明一下，当$k=1$时此$f(x)$下的ThreTopK也正好退化为Softmax。

参考实现：
    
    
    import numpy as np
    
    def ThreTopK(x, k):
        x_sort = np.sort(x)
        lse1 = np.logaddexp.accumulate(x_sort)
        lse2 = np.pad(np.logaddexp.accumulate(-x_sort[::-1])[::-1], (0, 1), constant_values=-np.inf)[1:]
        m = np.arange(len(x) - 1, -1, -1)
        x_lamb = lse1 - np.log(np.sqrt((k - m)**2 + np.exp(lse1 + lse2)) + (k - m))
        x_sort_shift = np.pad(x_sort[1:], (0, 1), constant_values=np.inf)
        lamb = x_lamb[(x_lamb <= x_sort_shift) & (x_lamb >= x_sort)]
        return (1 - np.exp(-np.abs(x - lamb))) * np.sign(x - lamb) * 0.5 + 0.5
    
    k, x = 10, np.random.randn(100)
    ThreTopK(x, k)

## 文章小结 #

本文探讨了Top-k算子的光滑近似问题，它是Softmax等Top1的光滑近似的一般推广，提出了迭代构造、梯度指引、待定常数三种构造思路，并分析了它们的优缺点。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10373>_

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

苏剑林. (Sep. 19, 2024). 《Softmax后传：寻找Top-K的光滑近似 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10373>

@online{kexuefm-10373,  
title={Softmax后传：寻找Top-K的光滑近似},  
author={苏剑林},  
year={2024},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/10373}},  
} 


---

## 公式推导与注释

本节将深入推导Top-K操作的光滑近似理论，包括Top-K的严格数学定义、不可微性问题的完整分析、Softmax光滑近似的理论基础、α-entmax的详细推导、sparsemax的完整证明、光滑性与稀疏性的权衡分析、梯度计算与反向传播，以及近似误差的定量分析。

### 1. Top-K操作的数学定义与性质

#### 1.1 Top-K算子的形式化定义

**推导1：Top-K选择集合**

给定向量$\boldsymbol{x} = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$，Top-K选择集合定义为：

$$
\Omega_k(\boldsymbol{x}) = \{i_1, i_2, \ldots, i_k\} \subset \{1, 2, \ldots, n\}
$$

满足：

$$
x_{i_1} \geq x_{i_2} \geq \cdots \geq x_{i_k} > x_j, \quad \forall j \notin \Omega_k(\boldsymbol{x})
$$

即$\Omega_k(\boldsymbol{x})$包含$\boldsymbol{x}$中最大的$k$个分量的下标。

**推导2：Top-K指示向量**

定义Top-K算子$\mathcal{T}_k: \mathbb{R}^n \to \{0, 1\}^n$：

$$
[\mathcal{T}_k(\boldsymbol{x})]_i = \begin{cases}
1, & i \in \Omega_k(\boldsymbol{x}) \\
0, & i \notin \Omega_k(\boldsymbol{x})
\end{cases}
$$

验证性质：$\sum_{i=1}^{n} [\mathcal{T}_k(\boldsymbol{x})]_i = k$（恰有$k$个1）。

**推导3：Top-K作为投影算子**

Top-K可以视为向量到Multi-Hot集合的投影：

$$
\mathcal{T}_k(\boldsymbol{x}) = \mathop{\arg\min}_{\boldsymbol{z} \in \{0,1\}^n, \|\boldsymbol{z}\|_1 = k} \|\boldsymbol{z} - \boldsymbol{x}\|^2
$$

这个优化问题的解正是保留最大的$k$个分量。

证明：目标函数展开为

$$
\|\boldsymbol{z} - \boldsymbol{x}\|^2 = \sum_{i=1}^{n} (z_i - x_i)^2
$$

在约束$z_i \in \{0, 1\}$和$\sum_i z_i = k$下，要最小化上式，应该让$z_i = 1$对应最大的$x_i$。

#### 1.2 Top-K的组合性质

**推导4：Top-K的排列不变性**

对于任意置换$\pi: \{1, \ldots, n\} \to \{1, \ldots, n\}$：

$$
\mathcal{T}_k(\boldsymbol{x}_\pi) = (\mathcal{T}_k(\boldsymbol{x}))_\pi
$$

其中$\boldsymbol{x}_\pi = (x_{\pi(1)}, \ldots, x_{\pi(n)})$。

这表明Top-K只依赖于值的大小关系，不依赖于下标。

**推导5：Top-K的单调性**

如果$x_i > x_j$，那么：

$$
[\mathcal{T}_k(\boldsymbol{x})]_i \geq [\mathcal{T}_k(\boldsymbol{x})]_j
$$

即较大的分量更可能被选中。

**推导6：Top-K的平移不变性**

对于任意常数$c \in \mathbb{R}$：

$$
\mathcal{T}_k(\boldsymbol{x} + c \boldsymbol{1}) = \mathcal{T}_k(\boldsymbol{x})
$$

其中$\boldsymbol{1} = (1, 1, \ldots, 1)$。Top-K只关心相对大小，不受整体平移影响。

### 2. 不可微性问题的深入分析

#### 2.1 Top-K的不连续性

**推导7：Top-K的跳变点**

考虑一维情况，$n=2, k=1$，$\boldsymbol{x} = (x_1, x_2)$：

$$
\mathcal{T}_1(\boldsymbol{x}) = \begin{cases}
(1, 0), & x_1 > x_2 \\
(0, 1), & x_1 < x_2
\end{cases}
$$

在$x_1 = x_2$处，$\mathcal{T}_1$不连续（从$(1, 0)$跳变到$(0, 1)$）。

**推导8：Top-K的梯度不存在性**

在跳变点，偏导数不存在。考虑$x_1 = x_2 = t$：

$$
\lim_{\epsilon \to 0^+} \frac{[\mathcal{T}_1(t+\epsilon, t)]_1 - [\mathcal{T}_1(t, t)]_1}{\epsilon}
$$

这个极限不存在，因为：
- 当$\epsilon > 0$时，$[\mathcal{T}_1(t+\epsilon, t)]_1 = 1$
- 当$\epsilon < 0$时，$[\mathcal{T}_1(t+\epsilon, t)]_1 = 0$

左右导数不相等。

**推导9：测度零集上的可微性**

Top-K在$\mathbb{R}^n$的几乎处处可微（除了测度零的边界集），但在边界处梯度为零：

$$
\nabla \mathcal{T}_k(\boldsymbol{x}) = \boldsymbol{0}, \quad \text{a.e.}
$$

这对于基于梯度的优化毫无帮助。

#### 2.2 反向传播的困难

**推导10：链式法则的失效**

假设损失函数$\mathcal{L}(\mathcal{T}_k(\boldsymbol{x}))$，链式法则要求：

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \mathcal{T}_k} \cdot \frac{\partial \mathcal{T}_k}{\partial x_i}
$$

但由于$\frac{\partial \mathcal{T}_k}{\partial x_i} = 0$（几乎处处），梯度信息无法回传。

**推导11：次梯度的局限性**

虽然可以定义次梯度（subgradient），但对于Top-K：

$$
\partial \mathcal{T}_k(\boldsymbol{x}) = \text{convex hull of gradients near } \boldsymbol{x}
$$

这个集合通常包含零向量，不提供有用的优化方向。

### 3. Softmax光滑近似的理论基础

#### 3.1 Softmax作为Top-1的近似

**推导12：Softmax的定义**

Softmax函数定义为：

$$
[\text{softmax}(\boldsymbol{x})]_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

关键性质：
- $\sum_{i=1}^{n} [\text{softmax}(\boldsymbol{x})]_i = 1$（归一化）
- $[\text{softmax}(\boldsymbol{x})]_i > 0$（严格正）
- $x_i > x_j \Rightarrow [\text{softmax}(\boldsymbol{x})]_i > [\text{softmax}(\boldsymbol{x})]_j$（单调性）

**推导13：温度参数的作用**

引入温度$\tau > 0$：

$$
[\text{softmax}(\boldsymbol{x}/\tau)]_i = \frac{e^{x_i/\tau}}{\sum_{j=1}^{n} e^{x_j/\tau}}
$$

当$\tau \to 0^+$时的极限行为：

设$x_{\max} = \max_i x_i$，$\Omega_1 = \{i : x_i = x_{\max}\}$（最大值的下标集合，可能有多个）。

$$
\lim_{\tau \to 0^+} [\text{softmax}(\boldsymbol{x}/\tau)]_i = \begin{cases}
\frac{1}{|\Omega_1|}, & i \in \Omega_1 \\
0, & i \notin \Omega_1
\end{cases}
$$

证明：对于$i \in \Omega_1$，$x_i = x_{\max}$：

$$
[\text{softmax}(\boldsymbol{x}/\tau)]_i = \frac{e^{x_{\max}/\tau}}{\sum_{j \in \Omega_1} e^{x_{\max}/\tau} + \sum_{j \notin \Omega_1} e^{x_j/\tau}}
$$

当$\tau \to 0^+$时，$e^{(x_j - x_{\max})/\tau} \to 0$对所有$j \notin \Omega_1$（因为$x_j < x_{\max}$），所以：

$$
[\text{softmax}(\boldsymbol{x}/\tau)]_i \to \frac{e^{x_{\max}/\tau}}{|\Omega_1| e^{x_{\max}/\tau}} = \frac{1}{|\Omega_1|}
$$

**推导14：Softmax作为LogSumExp的梯度**

定义LogSumExp（LSE）：

$$
\text{LSE}(\boldsymbol{x}) = \log \sum_{i=1}^{n} e^{x_i}
$$

计算其梯度：

$$
\frac{\partial \text{LSE}(\boldsymbol{x})}{\partial x_i} = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} = [\text{softmax}(\boldsymbol{x})]_i
$$

这个联系很重要：Softmax可以视为某个凸函数的梯度，因此继承了良好的优化性质。

#### 3.2 Softmax的优化性质

**推导15：Softmax的光滑性**

Softmax是无穷次可微的（$C^\infty$）。对于任意$i, j$：

$$
\frac{\partial^2 \text{softmax}_i}{\partial x_j \partial x_k} \text{ 存在且连续}
$$

Jacobian矩阵：

$$
J_{ij} = \frac{\partial \text{softmax}_i(\boldsymbol{x})}{\partial x_j} = \begin{cases}
\text{softmax}_i(1 - \text{softmax}_i), & i = j \\
-\text{softmax}_i \cdot \text{softmax}_j, & i \neq j
\end{cases}
$$

这是对角占优矩阵，数值稳定。

**推导16：Softmax的Lipschitz连续性**

存在常数$L$使得：

$$
\|\text{softmax}(\boldsymbol{x}_1) - \text{softmax}(\boldsymbol{x}_2)\| \leq L \|\boldsymbol{x}_1 - \boldsymbol{x}_2\|
$$

具体地，对于$\ell_2$范数，$L = 1$：

$$
\|\text{softmax}(\boldsymbol{x}_1) - \text{softmax}(\boldsymbol{x}_2)\|_2 \leq \|\boldsymbol{x}_1 - \boldsymbol{x}_2\|_2
$$

这保证了Softmax不会放大输入的扰动。

### 4. 推广到Top-K：LogSumExp方法

#### 4.1 K-sum的LogSumExp

**推导17：K-sum的定义**

定义K-sum为最大的$k$个分量之和：

$$
S_k(\boldsymbol{x}) = \sum_{i \in \Omega_k(\boldsymbol{x})} x_i
$$

这可以重写为：

$$
S_k(\boldsymbol{x}) = \max_{|I| = k} \sum_{i \in I} x_i
$$

其中最大化遍历所有大小为$k$的下标集合$I$。

**推导18：K-sum的光滑近似**

用LogSumExp近似最大值：

$$
\tilde{S}_k(\boldsymbol{x}) = \log \sum_{|I|=k} \exp\left(\sum_{i \in I} x_i\right)
$$

定义配分函数：

$$
Z_k = \sum_{|I|=k} \exp\left(\sum_{i \in I} x_i\right) = \sum_{i_1 < i_2 < \cdots < i_k} e^{x_{i_1} + x_{i_2} + \cdots + x_{i_k}}
$$

则$\tilde{S}_k(\boldsymbol{x}) = \log Z_k$。

**推导19：梯度推导（GradTopK）**

计算$\tilde{S}_k$关于$x_i$的梯度：

$$
\frac{\partial \tilde{S}_k}{\partial x_i} = \frac{1}{Z_k} \frac{\partial Z_k}{\partial x_i}
$$

注意到$Z_k$中包含$x_i$的项是所有包含$i$的$k$元子集：

$$
\frac{\partial Z_k}{\partial x_i} = \sum_{\substack{|I|=k \\ i \in I}} e^{\sum_{j \in I} x_j}
$$

定义：

$$
Z_{k,i} = \sum_{\substack{|I|=k \\ i \in I}} e^{\sum_{j \in I} x_j}
$$

则：

$$
[\text{GradTopK}(\boldsymbol{x})]_i = \frac{Z_{k,i}}{Z_k}
$$

**推导20：GradTopK的性质验证**

验证归一化：

$$
\sum_{i=1}^{n} \frac{Z_{k,i}}{Z_k} = \frac{1}{Z_k} \sum_{i=1}^{n} Z_{k,i}
$$

注意到每个$k$元子集$I$在$\sum_{i} Z_{k,i}$中被计数$k$次（$I$的每个元素贡献一次），所以：

$$
\sum_{i=1}^{n} Z_{k,i} = k \cdot Z_k
$$

因此：

$$
\sum_{i=1}^{n} [\text{GradTopK}(\boldsymbol{x})]_i = k
$$

验证极限行为：设$\boldsymbol{x}/\tau$，当$\tau \to 0^+$时，$Z_k$由最大的$k$个分量主导：

$$
Z_k \approx \exp\left(\frac{1}{\tau}\sum_{i \in \Omega_k} x_i\right)
$$

类似地，$Z_{k,i} \approx \exp\left(\frac{1}{\tau}\sum_{j \in \Omega_k} x_j\right)$如果$i \in \Omega_k$，否则指数项更小。

因此：

$$
\lim_{\tau \to 0^+} [\text{GradTopK}(\boldsymbol{x}/\tau)]_i = \begin{cases}
1, & i \in \Omega_k(\boldsymbol{x}) \\
0, & i \notin \Omega_k(\boldsymbol{x})
\end{cases}
$$

这正是$\mathcal{T}_k(\boldsymbol{x})$！

#### 4.2 递归计算的数值稳定性

**推导21：递归关系**

注意到：

$$
Z_{k,i} = e^{x_i} \cdot (Z_{k-1} - Z_{k-1,i})
$$

解释：包含$i$的$k$元子集 = $\{i\}$ + 不包含$i$的$(k-1)$元子集。

但这有个问题：$Z_{k-1,i}$也计数了包含$i$的子集，需要排除。

正确的递归是：

$$
Z_{k,i} = e^{x_i} \cdot Z_{k-1}^{(-i)}
$$

其中$Z_{k-1}^{(-i)}$是从$\boldsymbol{x}_{-i} = (x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)$计算的配分函数。

**推导22：对数域计算**

为了数值稳定，在对数域计算：

$$
\log Z_k = \text{LSE}\left(\left\{\sum_{i \in I} x_i : |I| = k\right\}\right)
$$

使用动态规划：定义$\ell_k^{(m)}$为从前$m$个元素中选$k$个的log配分函数。

递归：

$$
\ell_k^{(m)} = \text{LSE}(\ell_k^{(m-1)}, x_m + \ell_{k-1}^{(m-1)})
$$

边界：$\ell_0^{(m)} = 0$，$\ell_k^{(k)} = \sum_{i=1}^{k} x_i$。

**推导23：数值不稳定性分析**

即使在对数域，当$k$较大或$\boldsymbol{x}$方差大时，$\log Z_{k-1}$和$\max_i \log Z_{k-1,i}$可能非常接近，导致：

$$
\log(e^{\log Z_{k-1}} - e^{\log Z_{k-1,i}}) \approx \log(e^{A} - e^{A-\epsilon})
$$

当$\epsilon$很小时，精度损失严重。这是LogSumExp方法的根本限制。

### 5. 阈值方法：ThreTopK

#### 5.1 通用阈值化框架

**推导24：阈值化构造**

定义光滑的S型函数$f: \mathbb{R} \to [0, 1]$，满足：
- $\lim_{x \to \infty} f(x) = 1$
- $\lim_{x \to -\infty} f(x) = 0$
- $f$单调递增且光滑

构造：

$$
\mathcal{ST}_k(\boldsymbol{x}) = f(\boldsymbol{x} - \lambda(\boldsymbol{x}))
$$

其中$\lambda(\boldsymbol{x})$满足：

$$
\sum_{i=1}^{n} f(x_i - \lambda(\boldsymbol{x})) = k
$$

这保证了归一化。

**推导25：阈值的存在唯一性**

定义函数：

$$
g(\lambda) = \sum_{i=1}^{n} f(x_i - \lambda)
$$

由于$f$单调递增，$g(\lambda)$单调递减：

$$
\lambda_1 < \lambda_2 \Rightarrow g(\lambda_1) > g(\lambda_2)
$$

边界条件：
- $\lim_{\lambda \to -\infty} g(\lambda) = n$
- $\lim_{\lambda \to \infty} g(\lambda) = 0$

由中间值定理，存在唯一的$\lambda$使得$g(\lambda) = k$（假设$0 < k < n$）。

**推导26：ThreTopK的单调性**

如果$x_i > x_j$，则：

$$
[\mathcal{ST}_k(\boldsymbol{x})]_i = f(x_i - \lambda) > f(x_j - \lambda) = [\mathcal{ST}_k(\boldsymbol{x})]_j
$$

因为$f$单调且$x_i - \lambda > x_j - \lambda$。

**推导27：ThreTopK的极限行为**

对于$\boldsymbol{x}/\tau$，当$\tau \to 0^+$时，$\lambda(\boldsymbol{x}/\tau)$的行为：

假设$\boldsymbol{x}$的第$k$大和第$(k+1)$大分别为$x_{(k)}$和$x_{(k+1)}$，$x_{(k)} > x_{(k+1)}$。

选择$\lambda \in (x_{(k+1)}, x_{(k)})$（在$\tau \to 0$的尺度下），则：
- 对于$i \in \Omega_k$：$x_i/\tau - \lambda/\tau \to +\infty$，所以$f(x_i/\tau - \lambda/\tau) \to 1$
- 对于$i \notin \Omega_k$：$x_i/\tau - \lambda/\tau \to -\infty$，所以$f(x_i/\tau - \lambda/\tau) \to 0$

因此$\mathcal{ST}_k(\boldsymbol{x}/\tau) \to \mathcal{T}_k(\boldsymbol{x})$。

#### 5.2 具体实例：指数型阈值

**推导28：$f(x) = \min(1, e^x)$的解析解**

对于这个$f$，约束变为：

$$
\sum_{i=1}^{n} \min(1, e^{x_i - \lambda}) = k
$$

假设$\boldsymbol{x}$已降序排列：$x_1 \geq x_2 \geq \cdots \geq x_n$。

猜测$\lambda \in [x_{m+1}, x_m]$，则：
- 对于$i \leq m$：$x_i \geq \lambda$，所以$\min(1, e^{x_i - \lambda}) = 1$
- 对于$i > m$：$x_i < \lambda$，所以$\min(1, e^{x_i - \lambda}) = e^{x_i - \lambda}$

约束变为：

$$
m + \sum_{i=m+1}^{n} e^{x_i - \lambda} = k
$$

解得：

$$
e^{-\lambda} = \frac{k - m}{\sum_{i=m+1}^{n} e^{x_i}}
$$

即：

$$
\lambda = \log\left(\sum_{i=m+1}^{n} e^{x_i}\right) - \log(k - m)
$$

需要验证$\lambda \in [x_{m+1}, x_m]$。遍历$m = 0, 1, \ldots, k-1$找到满足条件的$m$。

**推导29：退化到Softmax**

当$k=1$时，$m$只能为0，约束为：

$$
\sum_{i=1}^{n} e^{x_i - \lambda} = 1
$$

即：

$$
e^{-\lambda} = \frac{1}{\sum_{i=1}^{n} e^{x_i}}
$$

所以：

$$
[\mathcal{ST}_1(\boldsymbol{x})]_i = e^{x_i - \lambda} = \frac{e^{x_i}}{\sum_j e^{x_j}} = [\text{softmax}(\boldsymbol{x})]_i
$$

完美退化！

#### 5.3 Sigmoid型阈值

**推导30：$f(x) = \sigma(x)$的数值求解**

对于Sigmoid函数$\sigma(x) = \frac{1}{1 + e^{-x}}$，约束为：

$$
\sum_{i=1}^{n} \sigma(x_i - \lambda) = k
$$

定义目标函数：

$$
F(\lambda) = \sum_{i=1}^{n} \sigma(x_i - \lambda) - k
$$

求解$F(\lambda) = 0$。使用二分法或牛顿法。

**推导31：二分法的界**

由于$\sigma(x) \in (0, 1)$，我们有：

$$
0 < F(\lambda) < n
$$

当$\lambda \to -\infty$时，$\sigma(x_i - \lambda) \to 1$，$F(\lambda) \to n - k > 0$（如果$k < n$）。
当$\lambda \to +\infty$时，$\sigma(x_i - \lambda) \to 0$，$F(\lambda) \to -k < 0$。

所以$F$在某个区间$[\lambda_{\min}, \lambda_{\max}]$内有零点。更精确的界：

$$
\lambda_{\min} = x_{\min} - \sigma^{-1}(k/n), \quad \lambda_{\max} = x_{\max} - \sigma^{-1}(k/n)
$$

其中$\sigma^{-1}(p) = \log\frac{p}{1-p}$（logit函数）。

**推导32：牛顿法迭代**

牛顿法更新：

$$
\lambda_{t+1} = \lambda_t - \frac{F(\lambda_t)}{F'(\lambda_t)}
$$

计算$F'(\lambda)$：

$$
F'(\lambda) = -\sum_{i=1}^{n} \sigma'(x_i - \lambda) = -\sum_{i=1}^{n} \sigma(x_i - \lambda)(1 - \sigma(x_i - \lambda))
$$

注意$F'(\lambda) < 0$（单调递减），保证收敛。

### 6. 梯度计算与反向传播

#### 6.1 隐函数定理求梯度

**推导33：阈值$\lambda$关于$\boldsymbol{x}$的梯度**

约束方程：

$$
G(\boldsymbol{x}, \lambda) = \sum_{i=1}^{n} f(x_i - \lambda(\boldsymbol{x})) - k = 0
$$

对$x_j$求偏导：

$$
\frac{\partial G}{\partial x_j} + \frac{\partial G}{\partial \lambda} \frac{\partial \lambda}{\partial x_j} = 0
$$

其中：

$$
\frac{\partial G}{\partial x_j} = f'(x_j - \lambda), \quad \frac{\partial G}{\partial \lambda} = -\sum_{i=1}^{n} f'(x_i - \lambda)
$$

解得：

$$
\frac{\partial \lambda}{\partial x_j} = \frac{f'(x_j - \lambda)}{\sum_{i=1}^{n} f'(x_i - \lambda)}
$$

**推导34：ThreTopK的Jacobian**

对于$y_i = f(x_i - \lambda(\boldsymbol{x}))$：

$$
\frac{\partial y_i}{\partial x_j} = f'(x_i - \lambda) \left(\delta_{ij} - \frac{\partial \lambda}{\partial x_j}\right)
$$

代入$\frac{\partial \lambda}{\partial x_j}$：

$$
\frac{\partial y_i}{\partial x_j} = f'(x_i - \lambda) \left(\delta_{ij} - \frac{f'(x_j - \lambda)}{\sum_k f'(x_k - \lambda)}\right)
$$

矩阵形式（记$w_i = f'(x_i - \lambda)$，$W = \sum_k w_k$）：

$$
J = \text{diag}(\boldsymbol{w}) - \frac{1}{W} \boldsymbol{w} \boldsymbol{w}^T
$$

这是秩1修正的对角矩阵。

**推导35：反向传播公式**

损失$\mathcal{L}(\boldsymbol{y})$关于$\boldsymbol{x}$的梯度：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}} = J^T \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}
$$

展开：

$$
\frac{\partial \mathcal{L}}{\partial x_i} = w_i \frac{\partial \mathcal{L}}{\partial y_i} - \frac{w_i}{W} \sum_{j=1}^{n} w_j \frac{\partial \mathcal{L}}{\partial y_j}
$$

可以高效计算：先算$\sum_{j} w_j \frac{\partial \mathcal{L}}{\partial y_j}$，然后对每个$i$计算。

#### 6.2 Stop Gradient技巧

**推导36：显式梯度替换**

定义：

$$
\lambda_{\text{sg}} = \boldsymbol{x} \cdot \text{sg}[\nabla_{\boldsymbol{x}} \lambda] + \text{sg}[\lambda - \boldsymbol{x} \cdot \nabla_{\boldsymbol{x}} \lambda]
$$

其中$\text{sg}[\cdot]$是stop gradient算子。

前向传播：$\lambda_{\text{sg}} = \lambda$（因为$\text{sg}[x] = x$在前向）
反向传播：

$$
\nabla_{\boldsymbol{x}} \lambda_{\text{sg}} = \text{sg}[\nabla_{\boldsymbol{x}} \lambda] + \text{sg}[0] = \nabla_{\boldsymbol{x}} \lambda
$$

这样我们手动指定了$\lambda$的梯度。

**推导37：计算图分离**

在实现中，可以：
1. 用数值方法求$\lambda$（如二分法），得到标量值
2. 计算$\nabla_{\boldsymbol{x}} \lambda$的解析表达式
3. 用stop gradient技巧将两者结合

伪代码：
```
lambda_val = binary_search(x, k)  # 数值求解，不建计算图
grad_lambda = f'(x - lambda_val) / sum(f'(x - lambda_val))  # 解析梯度
lambda_with_grad = x @ sg(grad_lambda) + sg(lambda_val - x @ grad_lambda)
y = f(x - lambda_with_grad)
```

### 7. α-entmax的完整推导

#### 7.1 Tsallis熵与α-entmax

**推导38：Tsallis熵**

普通Shannon熵：

$$
H(\boldsymbol{p}) = -\sum_{i} p_i \log p_i
$$

Tsallis熵（$\alpha > 0$，$\alpha \neq 1$）：

$$
H_\alpha(\boldsymbol{p}) = \frac{1}{\alpha - 1} \left(1 - \sum_{i} p_i^\alpha\right)
$$

当$\alpha \to 1$时，Tsallis熵退化为Shannon熵。

**推导39：α-entmax的优化问题**

α-entmax定义为如下优化问题的解：

$$
\mathcal{E}_\alpha(\boldsymbol{x}) = \mathop{\arg\max}_{\boldsymbol{p} \in \Delta^{n-1}} \boldsymbol{p} \cdot \boldsymbol{x} + H_\alpha(\boldsymbol{p})
$$

其中$\Delta^{n-1} = \{\boldsymbol{p} : p_i \geq 0, \sum_i p_i = 1\}$是概率单纯形。

展开目标函数：

$$
\mathcal{L}_\alpha(\boldsymbol{p}) = \sum_{i} p_i x_i + \frac{1}{\alpha - 1} \left(1 - \sum_{i} p_i^\alpha\right)
$$

**推导40：KKT条件**

拉格朗日函数：

$$
\mathcal{L} = \sum_{i} p_i x_i + \frac{1}{\alpha - 1} (1 - \sum_{i} p_i^\alpha) - \lambda (\sum_{i} p_i - 1) - \sum_{i} \mu_i p_i
$$

KKT条件：
1. $\frac{\partial \mathcal{L}}{\partial p_i} = x_i - \frac{\alpha}{\alpha - 1} p_i^{\alpha - 1} - \lambda - \mu_i = 0$
2. $\mu_i \geq 0, p_i \geq 0, \mu_i p_i = 0$（互补松弛）
3. $\sum_i p_i = 1$

**推导41：最优解的形式**

对于活跃的$p_i > 0$（$\mu_i = 0$）：

$$
x_i - \frac{\alpha}{\alpha - 1} p_i^{\alpha - 1} = \lambda
$$

解得：

$$
p_i = \left(\frac{\alpha - 1}{\alpha}(x_i - \lambda)\right)^{1/(\alpha - 1)}
$$

定义投影函数：

$$
[t]_+ = \max(0, t)
$$

则：

$$
p_i = \left[\frac{\alpha - 1}{\alpha}(x_i - \lambda)\right]_+^{1/(\alpha - 1)}
$$

其中$\lambda$由归一化条件$\sum_i p_i = 1$确定。

**推导42：特殊情况**

- $\alpha = 1$：退化为Softmax

  极限$\alpha \to 1$：使用L'Hôpital法则，可以证明解为$p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$

- $\alpha = 2$：Sparsemax

  $$
  p_i = [x_i - \lambda]_+
  $$

  这正是Sparsemax！

#### 7.2 Sparsemax的详细推导

**推导43：Sparsemax作为欧几里得投影**

Sparsemax是向量$\boldsymbol{x}$到概率单纯形$\Delta^{n-1}$的欧几里得投影：

$$
\text{sparsemax}(\boldsymbol{x}) = \mathop{\arg\min}_{\boldsymbol{p} \in \Delta^{n-1}} \|\boldsymbol{p} - \boldsymbol{x}\|^2
$$

拉格朗日函数：

$$
\mathcal{L} = \frac{1}{2} \sum_{i} (p_i - x_i)^2 - \lambda(\sum_{i} p_i - 1) - \sum_{i} \mu_i p_i
$$

KKT条件：$p_i - x_i - \lambda - \mu_i = 0$

对于$p_i > 0$：$p_i = x_i + \lambda$
对于$p_i = 0$：$x_i + \lambda \leq 0$

所以：

$$
p_i = [x_i + \lambda]_+
$$

由于Sparsemax是最小化（而非最大化），$\lambda$通常是负数，改写为$\lambda = -\tau$：

$$
p_i = [x_i - \tau]_+
$$

**推导44：阈值$\tau$的计算**

设$\boldsymbol{x}$降序排列，猜测支撑集$S = \{1, 2, \ldots, k\}$（前$k$个非零）。

归一化条件：

$$
\sum_{i=1}^{k} (x_i - \tau) = 1
$$

解得：

$$
\tau = \frac{\sum_{i=1}^{k} x_i - 1}{k}
$$

需要验证：
- 对于$i \leq k$：$x_i \geq \tau$（非零）
- 对于$i > k$：$x_i < \tau$（为零）

遍历$k$找到满足条件的那个。

**推导45：Sparsemax梯度**

使用隐函数定理（与之前ThreTopK类似）：

$$
\frac{\partial \tau}{\partial x_j} = \frac{1_{j \in S}}{|S|}
$$

其中$S$是支撑集，$1_{j \in S}$是指示函数。

Jacobian：

$$
\frac{\partial p_i}{\partial x_j} = 1_{i \in S} \left(\delta_{ij} - \frac{1}{|S|}\right)
$$

矩阵形式（限制到支撑集$S$）：

$$
J_S = I_S - \frac{1}{|S|} \boldsymbol{1}_S \boldsymbol{1}_S^T
$$

这是中心化矩阵。

### 8. 光滑性与稀疏性的权衡

#### 8.1 定量分析

**推导46：稀疏性度量**

定义稀疏性为零分量的比例：

$$
\text{Sparsity}(\boldsymbol{p}) = \frac{|\{i : p_i = 0\}|}{n}
$$

对于不同方法：
- Softmax：$\text{Sparsity} = 0$（无稀疏性）
- Sparsemax：$\text{Sparsity} \approx 1 - \frac{k}{n}$（高稀疏性）
- α-entmax（$1 < \alpha < 2$）：介于两者之间

**推导47：光滑性度量**

用Lipschitz常数衡量光滑性：

$$
L = \sup_{\boldsymbol{x}_1 \neq \boldsymbol{x}_2} \frac{\|\mathcal{ST}_k(\boldsymbol{x}_1) - \mathcal{ST}_k(\boldsymbol{x}_2)\|}{\|\boldsymbol{x}_1 - \boldsymbol{x}_2\|}
$$

对于Softmax：$L = 1$（最光滑）
对于Sparsemax：$L$可能较大（在边界处有跳变）

**推导48：熵与稀疏性的关系**

对于α-entmax，熵$H_\alpha$随$\alpha$变化：
- $\alpha \to 1$：熵大，分布平滑，稀疏性低
- $\alpha \to 2$：熵小，分布集中，稀疏性高

权衡曲线：

$$
\alpha \in [1, 2] \Rightarrow \text{从光滑到稀疏的连续过渡}
$$

#### 8.2 近似误差分析

**推导49：$L^2$误差界**

定义近似误差：

$$
E(\boldsymbol{x}, \tau) = \|\mathcal{ST}_k(\boldsymbol{x}/\tau) - \mathcal{T}_k(\boldsymbol{x})\|^2
$$

当$\tau \to 0$时，$E \to 0$。量化收敛速率：

对于Softmax类方法（指数型），可以证明：

$$
E(\boldsymbol{x}, \tau) = O(e^{-\Delta/\tau})
$$

其中$\Delta = x_{(k)} - x_{(k+1)}$是第$k$大和第$(k+1)$大的间隔（gap）。

证明梗概：主要误差来自$i = k$和$i = k+1$附近，其他分量指数衰减。

**推导50：最大误差界**

$$
\|[\mathcal{ST}_k(\boldsymbol{x}/\tau)]_i - [\mathcal{T}_k(\boldsymbol{x})]_i\|_\infty = O(\tau)
$$

对于线性型方法（如Sparsemax），误差随$\tau$线性下降。

**推导51：梯度误差**

近似的梯度$\nabla \mathcal{ST}_k$与真实梯度（次梯度）$\partial \mathcal{T}_k$的差异：

$$
\|\nabla \mathcal{ST}_k(\boldsymbol{x}/\tau) - \partial \mathcal{T}_k(\boldsymbol{x})\| = O(1/\tau)
$$

当$\tau$很小时，梯度可能很大（数值不稳定）。实践中需要选择适中的$\tau$。

### 9. 综合对比与应用建议

**推导52：计算复杂度对比**

| 方法 | 前向复杂度 | 反向复杂度 | 数值稳定性 |
|------|-----------|-----------|-----------|
| GradTopK | $O(n C_n^k)$ | $O(n C_n^k)$ | 中等（大$k$时差） |
| ThreTopK (二分) | $O(n \log(1/\epsilon))$ | $O(n)$ | 好 |
| ThreTopK (解析) | $O(n \log n)$ | $O(n)$ | 很好 |
| Sparsemax | $O(n \log n)$ | $O(n)$ | 很好 |

**推导53：应用场景推荐**

1. **需要高度稀疏**：Sparsemax或$\alpha$-entmax（$\alpha$接近2）
   - 例如：注意力机制中只关注少数Token

2. **需要平滑梯度**：Softmax或$\alpha$-entmax（$\alpha$接近1）
   - 例如：训练早期，避免梯度消失

3. **计算效率优先**：ThreTopK with $f(x) = \min(1, e^x)$
   - 解析解，无需迭代

4. **$k$较大**：避免GradTopK（组合爆炸）
   - 使用ThreTopK或Sparsemax

**推导54：理论性质总结表**

| 性质 | Softmax | GradTopK | ThreTopK | Sparsemax |
|-----|---------|----------|----------|-----------|
| 归一化 | $\sum p_i = 1$ | $\sum p_i = k$ | $\sum p_i = k$ | $\sum p_i = 1$ |
| 稀疏性 | 无 | 无 | 取决于$f$ | 有 |
| $C^\infty$光滑 | ✓ | ✓ | 取决于$f$ | ✗（$C^1$） |
| $k=1$退化 | 自身 | Softmax | 取决于$f$ | Softmax |
| 单调性 | ✓ | ✓ | ✓ | ✓ |
| 平移不变性 | ✓ | ✓ | ✓ | ✓ |

### 10. 总结

通过以上54个详细推导，我们全面分析了Top-K光滑近似问题：

1. **数学定义**：严格定义了Top-K算子及其组合性质
2. **不可微性**：深入分析了Top-K的跳变点和梯度不存在性
3. **Softmax理论**：建立了Softmax作为LogSumExp梯度的理论基础
4. **GradTopK**：推导了基于K-sum LogSumExp的梯度方法及其数值问题
5. **ThreTopK**：提出了通用阈值化框架，给出解析解和数值解
6. **梯度计算**：使用隐函数定理和STE技巧推导了完整的反向传播
7. **α-entmax**：从Tsallis熵推导了α-entmax，统一了Softmax和Sparsemax
8. **权衡分析**：定量分析了光滑性与稀疏性的权衡，给出误差界
9. **综合对比**：比较了各方法的计算复杂度和适用场景

这些理论为理解和应用Top-K光滑近似提供了坚实的数学基础，并指导实际选择合适的方法。

