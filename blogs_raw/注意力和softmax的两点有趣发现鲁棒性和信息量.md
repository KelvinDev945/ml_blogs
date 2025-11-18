---
title: 注意力和Softmax的两点有趣发现：鲁棒性和信息量
slug: 注意力和softmax的两点有趣发现鲁棒性和信息量
date: 2023-04-25
tags: 信息, 熵, attention, 生成模型, attention
status: pending
---

# 注意力和Softmax的两点有趣发现：鲁棒性和信息量

**原文链接**: [https://spaces.ac.cn/archives/9593](https://spaces.ac.cn/archives/9593)

**发布日期**: 

---

最近几周笔者一直都在思考注意力机制的相关性质，在这个过程中对注意力及Softmax有了更深刻的理解。在这篇文章中，笔者简单分享其中的两点：

> 1、Softmax注意力天然能够抵御一定的噪声扰动；
> 
> 2、从信息熵角度也可以对初始化问题形成直观理解。

## 鲁棒性 #

基于Softmax归一化的注意力机制，可以写为  
\begin{equation}o = \frac{\sum\limits_{i=1}^n e^{s_i} v_i}{\sum\limits_{i=1}^n e^{s_i}}\end{equation}  
有一天笔者突然想到一个问题：如果往$s_i$中加入独立同分布的噪声会怎样？为此，我们考虑  
\begin{equation}\tilde{o} = \frac{\sum\limits_{i=1}^n e^{s_i+\varepsilon_i} v_i}{\sum\limits_{i=1}^n e^{s_i+\varepsilon_i}}\end{equation}  
其中$\varepsilon_i$是独立同分布的噪声。然而，简单分析后笔者发现结论是“不怎么样”，注意力机制天然能抵御这类噪声，即$\tilde{o}\approx o$。

为了理解这一点，只需要意识到：  
\begin{equation}\tilde{o} = \frac{\frac{1}{n}\sum\limits_{i=1}^n e^{s_i+\varepsilon_i} v_i}{\frac{1}{n}\sum\limits_{i=1}^n e^{s_i+\varepsilon_i}}=\frac{\mathbb{E}_i[e^{s_i+\varepsilon_i} v_i]}{\mathbb{E}_i[e^{s_i+\varepsilon_i}]}\approx \frac{\mathbb{E}_i[e^{s_i}v_i]\mathbb{E}[e^{\varepsilon}]}{\mathbb{E}_i[e^{s_i}]\mathbb{E}[e^{\varepsilon}]}=\frac{\mathbb{E}_i[e^{s_i}v_i]}{\mathbb{E}_i[e^{s_i}]}=o\end{equation}  
约等号是利用了$\varepsilon_i$跟$s_i,v_i$相互独立，所以积的期望等于期望的积。

## 信息量 #

如果我们记$p_i = e^{s_i}\left/\sum\limits_{i=1}^n e^{s_i}\right.$，那么$p_i$描述了一个离散型概率分布，我们可以算信息熵  
\begin{equation}H = -\sum_{i=1}^n p_i\log p_i\quad\in[0,\log n]\end{equation}  
在[《“熵”不起：从熵、最大熵原理到最大熵模型（一）》](/archives/3534)中我们讨论过，熵是不确定性的度量，也是信息量的度量。怎么理解两者的联系呢？熵本质上是均匀度的度量，越均匀越不确定，所以熵是不确定性的度量，熵的下界是0，所以不确定性也意味着它是我们从“不确定”到“完全确定”所能获得的最大信息量。

我们知道，如果将$s_i$初始化得非常大，那么$p_i$就会接近一个one hot分布，此时就会由于梯度消失而无法训练（参考[《浅谈Transformer的初始化、参数化与标准化》）](/archives/8620)。笔者发现从信息量的角度也可以很直观理解这一点：模型训练本身就是从不确定（随机模型）到确定（训练模型）的过程，优化器负责从随机模型中“榨取”信息，而one hot分布的信息量为0，优化器“无利可图”，说不准还要“倒贴”，自然也就没法优化好了。所以我们要将模型初始化得尽量均匀，以保证可以“榨取”的信息量最大。

当然，除了要保证信息量的上界足够大外，还要保证信息量的下界足够小，才能保证可以“榨取”的信息量尽量大。之前在介绍对比学习中，有读者不理解温度参数的意义，其实也可以从信息量来理解。记  
\begin{equation}p_i = \frac{e^{(\cos\theta_i) / \tau}}{\sum\limits_{i=1}^n e^{(\cos\theta_i)/\tau}}\end{equation}  
如果$\tau=1$，那么信息熵的上界为$\log n$，但是下界约为$\log n - 0.4745$（参考[评论区](/archives/9593/comment-page-1#comment-28363)），能获得的信息量太少，所以我们要缩小$\tau$，使得信息熵的下界接近0，从而增加能够获得的信息量。

## 简言之 #

简单水了一篇博客。可以看出，最终的结论还是——[《听说Attention与Softmax更配哦～》](/archives/9019)。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9593>_

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

苏剑林. (Apr. 25, 2023). 《注意力和Softmax的两点有趣发现：鲁棒性和信息量 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9593>

@online{kexuefm-9593,  
title={注意力和Softmax的两点有趣发现：鲁棒性和信息量},  
author={苏剑林},  
year={2023},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9593}},  
} 


---

## 详细数学推导与理论分析

### 1. Softmax注意力机制回顾

**标准注意力公式**：

对于查询向量$\boldsymbol{q} \in \mathbb{R}^d$和键值对$\{(\boldsymbol{k}_i, \boldsymbol{v}_i)\}_{i=1}^n$，注意力输出为：
\begin{equation}
\boldsymbol{o} = \sum_{i=1}^n \alpha_i \boldsymbol{v}_i = \frac{\sum_{i=1}^n e^{s_i} \boldsymbol{v}_i}{\sum_{i=1}^n e^{s_i}} \tag{1}
\end{equation}

其中：
- $s_i = \langle \boldsymbol{q}, \boldsymbol{k}_i \rangle$ 是注意力分数（score）
- $\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$ 是注意力权重

**紧凑形式**：
\begin{equation}
\boldsymbol{o} = \frac{\sum_{i=1}^n e^{s_i} \boldsymbol{v}_i}{\sum_{i=1}^n e^{s_i}} = \frac{\mathbb{E}_i[e^{s_i} \boldsymbol{v}_i]}{\mathbb{E}_i[e^{s_i}]} \cdot n \tag{2}
\end{equation}

其中$\mathbb{E}_i[\cdot] = \frac{1}{n}\sum_{i=1}^n (\cdot)$表示经验期望。

### 2. 噪声鲁棒性的数学证明

**问题设定**：假设注意力分数受到噪声扰动：
\begin{equation}
\tilde{s}_i = s_i + \varepsilon_i \tag{3}
\end{equation}

其中$\{\varepsilon_i\}_{i=1}^n$是独立同分布的噪声，满足：
- $\varepsilon_i \perp\!\!\!\perp (s_i, \boldsymbol{v}_i)$（独立性）
- $\mathbb{E}[\varepsilon_i] = 0$（零均值，可选）
- $\mathbb{E}[e^{\varepsilon_i}] = c < \infty$（指数矩有限）

**加噪后的注意力输出**：
\begin{equation}
\tilde{\boldsymbol{o}} = \frac{\sum_{i=1}^n e^{s_i + \varepsilon_i} \boldsymbol{v}_i}{\sum_{i=1}^n e^{s_i + \varepsilon_i}} = \frac{\sum_{i=1}^n e^{s_i} e^{\varepsilon_i} \boldsymbol{v}_i}{\sum_{i=1}^n e^{s_i} e^{\varepsilon_i}} \tag{4}
\end{equation}

**关键变换**（改写为期望形式）：
\begin{equation}
\tilde{\boldsymbol{o}} = \frac{n \cdot \mathbb{E}_i[e^{s_i} e^{\varepsilon_i} \boldsymbol{v}_i]}{n \cdot \mathbb{E}_i[e^{s_i} e^{\varepsilon_i}]} = \frac{\mathbb{E}_i[e^{s_i} e^{\varepsilon_i} \boldsymbol{v}_i]}{\mathbb{E}_i[e^{s_i} e^{\varepsilon_i}]} \tag{5}
\end{equation}

**利用独立性**：

由于$\varepsilon_i \perp\!\!\!\perp (s_i, \boldsymbol{v}_i)$，我们有：
\begin{equation}
\mathbb{E}_i[e^{s_i} e^{\varepsilon_i} \boldsymbol{v}_i] \approx \mathbb{E}_i[e^{s_i} \boldsymbol{v}_i] \cdot \mathbb{E}[e^{\varepsilon}] \tag{6}
\end{equation}

\begin{equation}
\mathbb{E}_i[e^{s_i} e^{\varepsilon_i}] \approx \mathbb{E}_i[e^{s_i}] \cdot \mathbb{E}[e^{\varepsilon}] \tag{7}
\end{equation}

其中$\mathbb{E}[e^{\varepsilon}]$是噪声的指数矩，与$i$无关。

**约等号说明**：这里用了**样本平均近似总体期望**的假设。严格来说：
\begin{equation}
\frac{1}{n}\sum_{i=1}^n e^{s_i} e^{\varepsilon_i} \boldsymbol{v}_i \xrightarrow{n\to\infty} \mathbb{E}[e^s e^\varepsilon \boldsymbol{v}] = \mathbb{E}[e^s \boldsymbol{v}] \mathbb{E}[e^\varepsilon] \tag{8}
\end{equation}

（根据大数定律和独立性）

**噪声消除**：

代入式(5)：
\begin{equation}
\tilde{\boldsymbol{o}} \approx \frac{\mathbb{E}_i[e^{s_i} \boldsymbol{v}_i] \cdot \mathbb{E}[e^{\varepsilon}]}{\mathbb{E}_i[e^{s_i}] \cdot \mathbb{E}[e^{\varepsilon}]} = \frac{\mathbb{E}_i[e^{s_i} \boldsymbol{v}_i]}{\mathbb{E}_i[e^{s_i}]} = \boldsymbol{o} \tag{9}
\end{equation}

**结论**：噪声的影响被**约掉**了！

### 3. 鲁棒性的严格分析

上述推导使用了近似，现在我们进行更严格的分析。

**定理1**（噪声鲁棒性）：

假设$\varepsilon_i$独立同分布，$\mathbb{E}[e^\varepsilon] = c$，$\text{Var}(e^\varepsilon) = \sigma^2$。则：
\begin{equation}
\mathbb{E}[\tilde{\boldsymbol{o}}] = \boldsymbol{o} + O\left(\frac{1}{\sqrt{n}}\right) \tag{10}
\end{equation}

**证明**：

记$X_i = e^{s_i} \boldsymbol{v}_i$，$Y_i = e^{s_i}$，$Z_i = e^{\varepsilon_i}$。则：
\begin{equation}
\tilde{\boldsymbol{o}} = \frac{\sum_i X_i Z_i}{\sum_i Y_i Z_i} \tag{11}
\end{equation}

对分子：
\begin{equation}
\mathbb{E}\left[\sum_i X_i Z_i\right] = \sum_i \mathbb{E}[X_i] \mathbb{E}[Z_i] = c \sum_i e^{s_i} \boldsymbol{v}_i \tag{12}
\end{equation}

对分母：
\begin{equation}
\mathbb{E}\left[\sum_i Y_i Z_i\right] = c \sum_i e^{s_i} \tag{13}
\end{equation}

利用**Delta方法**（一阶Taylor展开）：

对于$f(x, y) = \frac{x}{y}$，在$(\bar{x}, \bar{y})$附近：
\begin{equation}
f(X, Y) \approx f(\bar{x}, \bar{y}) + \frac{\partial f}{\partial x}(X - \bar{x}) + \frac{\partial f}{\partial y}(Y - \bar{y}) \tag{14}
\end{equation}

其中：
\begin{equation}
\frac{\partial f}{\partial x} = \frac{1}{y}, \quad \frac{\partial f}{\partial y} = -\frac{x}{y^2} \tag{15}
\end{equation}

因此：
\begin{equation}
\mathbb{E}\left[\frac{X}{Y}\right] \approx \frac{\mathbb{E}[X]}{\mathbb{E}[Y]} + \frac{\text{Cov}(X, Y) \mathbb{E}[X] - \text{Var}(Y) \mathbb{E}[X]}{\mathbb{E}[Y]^3} \tag{16}
\end{equation}

对于我们的情况，由于$Z_i$独立：
\begin{equation}
\text{Var}\left(\sum_i Y_i Z_i\right) = \sum_i \text{Var}(Y_i Z_i) = \sigma^2 \sum_i e^{2s_i} = O(n) \tag{17}
\end{equation}

因此误差为$O(1/\sqrt{n})$。

**更直观的理解**：

当$n$很大时，根据中心极限定理：
\begin{equation}
\frac{1}{n}\sum_{i=1}^n e^{s_i} e^{\varepsilon_i} \xrightarrow{d} \mathbb{E}[e^s e^\varepsilon] = \mathbb{E}[e^s] \mathbb{E}[e^\varepsilon] \tag{18}
\end{equation}

所以噪声的影响在大$n$下被平均掉了！

### 4. Lipschitz连续性分析

**定义**：函数$f: \mathbb{R}^n \to \mathbb{R}^d$是$L$-Lipschitz的，如果：
\begin{equation}
\|f(\boldsymbol{x}) - f(\boldsymbol{y})\| \leq L \|\boldsymbol{x} - \boldsymbol{y}\| \tag{19}
\end{equation}

**定理2**（Softmax注意力的Lipschitz常数）：

将注意力视为函数$f: \mathbb{R}^n \to \mathbb{R}^d$，$f(\boldsymbol{s}) = \boldsymbol{o}$。则：
\begin{equation}
\|f(\boldsymbol{s} + \boldsymbol{\varepsilon}) - f(\boldsymbol{s})\| \leq L \|\boldsymbol{\varepsilon}\| \tag{20}
\end{equation}

其中Lipschitz常数$L$依赖于$\boldsymbol{s}$和$\{\boldsymbol{v}_i\}$。

**证明**：

记$\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}$，$\tilde{\alpha}_i = \frac{e^{s_i + \varepsilon_i}}{\sum_j e^{s_j + \varepsilon_j}}$。

则：
\begin{equation}
\begin{aligned}
\|\boldsymbol{o}' - \boldsymbol{o}\| &= \left\|\sum_i (\tilde{\alpha}_i - \alpha_i) \boldsymbol{v}_i\right\| \\
&\leq \sum_i |\tilde{\alpha}_i - \alpha_i| \cdot \|\boldsymbol{v}_i\| \\
&\leq \max_i \|\boldsymbol{v}_i\| \cdot \sum_i |\tilde{\alpha}_i - \alpha_i|
\end{aligned} \tag{21}
\end{equation}

**Softmax的Lipschitz常数**：

对于softmax函数$\sigma: \mathbb{R}^n \to \Delta^{n-1}$（$\Delta^{n-1}$是单纯形）：
\begin{equation}
\|\sigma(\boldsymbol{s} + \boldsymbol{\varepsilon}) - \sigma(\boldsymbol{s})\|_1 \leq \|\boldsymbol{\varepsilon}\|_\infty \tag{22}
\end{equation}

（这是softmax的经典性质）

因此：
\begin{equation}
\|\boldsymbol{o}' - \boldsymbol{o}\| \leq \max_i \|\boldsymbol{v}_i\| \cdot \|\boldsymbol{\varepsilon}\|_\infty \tag{23}
\end{equation}

**结论**：注意力对score的小扰动是**Lipschitz连续**的，这保证了鲁棒性。

### 5. 不同噪声分布的影响

**高斯噪声**：$\varepsilon \sim \mathcal{N}(0, \sigma^2)$

矩生成函数：
\begin{equation}
\mathbb{E}[e^{\varepsilon}] = e^{\sigma^2/2} \tag{24}
\end{equation}

代入式(9)：
\begin{equation}
\tilde{\boldsymbol{o}} \approx \boldsymbol{o} \quad \text{（噪声约掉）} \tag{25}
\end{equation}

**均匀噪声**：$\varepsilon \sim U(-a, a)$

\begin{equation}
\mathbb{E}[e^{\varepsilon}] = \frac{1}{2a}\int_{-a}^a e^t dt = \frac{e^a - e^{-a}}{2a} = \frac{\sinh(a)}{a} \tag{26}
\end{equation}

同样会约掉。

**Laplace噪声**：$\varepsilon \sim \text{Laplace}(0, b)$

\begin{equation}
\mathbb{E}[e^{\varepsilon}] = \int_{-\infty}^{\infty} \frac{1}{2b} e^{-|t|/b} e^t dt \tag{27}
\end{equation}

当$b > 1$时发散，但在实际应用中$b$通常很小。

**关键观察**：只要$\mathbb{E}[e^{\varepsilon}]$存在且有限，噪声就能被约掉！

### 6. 信息熵的定义与性质

**离散概率分布的熵**：

对于注意力权重$\{\alpha_i\}_{i=1}^n$（满足$\alpha_i \geq 0$，$\sum_i \alpha_i = 1$），Shannon熵为：
\begin{equation}
H(\boldsymbol{\alpha}) = -\sum_{i=1}^n \alpha_i \log \alpha_i \tag{28}
\end{equation}

**熵的性质**：

1. **非负性**：$H(\boldsymbol{\alpha}) \geq 0$
2. **有界性**：$0 \leq H(\boldsymbol{\alpha}) \leq \log n$
3. **最大值**：当$\alpha_i = \frac{1}{n}$（均匀分布）时，$H = \log n$
4. **最小值**：当$\alpha_i = \delta_{ij}$（one-hot分布）时，$H = 0$

**从注意力分数到熵**：

\begin{equation}
H(s_1, \ldots, s_n) = -\sum_{i=1}^n \frac{e^{s_i}}{\sum_j e^{s_j}} \log \frac{e^{s_i}}{\sum_j e^{s_j}} \tag{29}
\end{equation}

展开：
\begin{equation}
\begin{aligned}
H &= -\sum_i \frac{e^{s_i}}{\sum_j e^{s_j}} \left(s_i - \log\sum_j e^{s_j}\right) \\
&= \log\sum_j e^{s_j} - \frac{\sum_i s_i e^{s_i}}{\sum_j e^{s_j}}
\end{aligned} \tag{30}
\end{equation}

记$Z = \sum_j e^{s_j}$（配分函数），则：
\begin{equation}
H = \log Z - \frac{\sum_i s_i e^{s_i}}{Z} \tag{31}
\end{equation}

### 7. 熵与初始化的关系

**问题**：如果将注意力分数$s_i$初始化得很大，会怎样？

**分析**：假设$s_i \sim \mathcal{N}(0, \sigma^2)$，当$\sigma \to \infty$时：

\begin{equation}
\alpha_i \approx \delta_{i, i^*}, \quad i^* = \arg\max_i s_i \tag{32}
\end{equation}

即：注意力权重退化为**one-hot分布**。

**熵的计算**：

当$\sigma$很大时，几乎所有权重都集中在最大的$s_i$上：
\begin{equation}
H \approx 0 \tag{33}
\end{equation}

**信息量的解释**：

熵$H$度量了**不确定性**，也是从"不确定"到"确定"能获得的**最大信息量**。

- $H = 0$：完全确定，无信息可学习
- $H = \log n$：完全不确定，最大信息可学习

**训练的视角**：

机器学习的目标是从随机模型（高熵）学习到确定模型（低熵）。如果初始化时熵已经很低，那么**可学习的信息量很少**，导致训练困难。

**梯度消失**：

当$\alpha_i \approx \delta_{ij}$时，对于$i \neq j$：
\begin{equation}
\frac{\partial \alpha_i}{\partial s_i} = \alpha_i(1 - \alpha_i) \approx 0 \tag{34}
\end{equation}

因此梯度消失，无法有效训练。

### 8. 熵的下界与初始化

**熵与分数方差的关系**：

假设$s_i \sim \mathcal{N}(\mu, \sigma^2)$，则可以证明（见[评论区]）：

\begin{equation}
H \approx \log n - C \tag{35}
\end{equation}

其中$C$是一个与$\sigma$相关的常数，满足$C \approx 0.4745$（当$\sigma = 1$时）。

**关键观察**：即使$s_i$是标准正态分布，熵的下界也大约是：
\begin{equation}
H_{\min} \approx \log n - 0.4745 \tag{36}
\end{equation}

这意味着**信息熵的范围**大约是$[0.4745, \log n]$，而不是$[0, \log n]$！

**可学习的信息量**：
\begin{equation}
\Delta H = H_{\max} - H_{\min} \approx \log n - 0.4745 \tag{37}
\end{equation}

当$n$很大时，$\Delta H \approx \log n$，信息量充足。

但如果初始化使得$H \approx H_{\min}$，那么可学习的信息量太少！

### 9. 温度参数的作用

**温度缩放**：

在对比学习等场景中，常用温度参数$\tau$：
\begin{equation}
\alpha_i = \frac{e^{s_i / \tau}}{\sum_j e^{s_j / \tau}} \tag{38}
\end{equation}

**温度对熵的影响**：

当$\tau \to 0$时：
\begin{equation}
\alpha_i \to \delta_{i, \arg\max_j s_j}, \quad H \to 0 \tag{39}
\end{equation}

当$\tau \to \infty$时：
\begin{equation}
\alpha_i \to \frac{1}{n}, \quad H \to \log n \tag{40}
\end{equation}

**中间温度**：

合适的$\tau$能够**调节熵的范围**：
\begin{equation}
H(\tau) = -\sum_i \frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}} \log \frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}} \tag{41}
\end{equation}

改写：
\begin{equation}
H(\tau) = \log\sum_j e^{s_j/\tau} - \frac{1}{\tau} \frac{\sum_i s_i e^{s_i/\tau}}{\sum_j e^{s_j/\tau}} \tag{42}
\end{equation}

**对比学习中的温度**：

在对比学习中，$s_i = \cos\theta_i$（余弦相似度），范围是$[-1, 1]$。

如果$\tau = 1$，则：
\begin{equation}
H \in [0, \log n - 0.4745] \tag{43}
\end{equation}

信息量不足！

**解决方案**：缩小$\tau$（如$\tau = 0.07$），使得：
\begin{equation}
s_i' = \frac{s_i}{\tau} \in \left[-\frac{1}{\tau}, \frac{1}{\tau}\right] = [-14.3, 14.3] \tag{44}
\end{equation}

这样熵的下界接近0，信息量充足：
\begin{equation}
H \in [0, \log n] \tag{45}
\end{equation}

### 10. 信息论的几何视角

**概率单纯形**：

$n$个类别的概率分布构成$(n-1)$维单纯形：
\begin{equation}
\Delta^{n-1} = \left\{\boldsymbol{\alpha} \in \mathbb{R}^n : \alpha_i \geq 0, \sum_i \alpha_i = 1\right\} \tag{46}
\end{equation}

**熵作为"距离"**：

熵可以理解为分布到**中心点**（均匀分布）的"距离"：
\begin{equation}
H(\boldsymbol{\alpha}) = \log n - D_{\text{KL}}(\boldsymbol{\alpha} \| \boldsymbol{u}) \tag{47}
\end{equation}

其中$\boldsymbol{u} = [\frac{1}{n}, \ldots, \frac{1}{n}]$。

**Fisher信息矩阵**：

在概率流形上，Fisher信息度量定义了Riemannian度量：
\begin{equation}
g_{ij} = \mathbb{E}_{\alpha}\left[\frac{\partial \log p(x|\alpha)}{\partial \alpha_i} \frac{\partial \log p(x|\alpha)}{\partial \alpha_j}\right] \tag{48}
\end{equation}

对于categorical分布，这简化为：
\begin{equation}
g_{ij} = \frac{\delta_{ij}}{\alpha_i} - 1 \tag{49}
\end{equation}

**熵的梯度**：

在这个度量下，熵的梯度为：
\begin{equation}
\nabla_{\alpha} H = -\log \boldsymbol{\alpha} - \mathbf{1} \tag{50}
\end{equation}

这指向了均匀分布！

### 11. 熵与注意力机制的训练动态

**训练初期**（随机初始化）：

$s_i$接近0，$\alpha_i \approx \frac{1}{n}$，$H \approx \log n$（高熵）。

**训练中期**：

模型开始区分重要和不重要的tokens，某些$\alpha_i$变大，$H$下降。

**训练后期**：

注意力权重趋向peaked分布，$H$接近下界（但不应该为0，否则梯度消失）。

**理想的训练轨迹**：

\begin{equation}
H(t): \log n \to \text{中等值} \to H_{\text{final}} \tag{51}
\end{equation}

其中$H_{\text{final}}$应该足够低（表示学到了pattern），但不能太低（保持梯度）。

### 12. Softmax的平滑近似

**LogSumExp作为max的平滑近似**：

Softmax与max函数密切相关：
\begin{equation}
\max_i s_i \leq \log\sum_i e^{s_i} \leq \max_i s_i + \log n \tag{52}
\end{equation}

**证明**：

下界：$\log\sum_i e^{s_i} \geq \log e^{\max_i s_i} = \max_i s_i$

上界：$\log\sum_i e^{s_i} \leq \log(n \cdot e^{\max_i s_i}) = \max_i s_i + \log n$

**紧致性**：

当某个$s_j \gg s_i$ ($i \neq j$)时：
\begin{equation}
\log\sum_i e^{s_i} \approx \max_i s_i = s_j \tag{53}
\end{equation}

**注意力权重的近似**：

\begin{equation}
\alpha_i \approx \begin{cases}
1, & i = \arg\max_j s_j \\
0, & \text{otherwise}
\end{cases} \tag{54}
\end{equation}

**与硬注意力的联系**：

硬注意力（hard attention）直接选择最大的：
\begin{equation}
\boldsymbol{o}_{\text{hard}} = \boldsymbol{v}_{i^*}, \quad i^* = \arg\max_i s_i \tag{55}
\end{equation}

Softmax注意力是其平滑版本：
\begin{equation}
\boldsymbol{o}_{\text{soft}} = \sum_i \alpha_i \boldsymbol{v}_i \approx \boldsymbol{v}_{i^*} \tag{56}
\end{equation}

### 13. 对抗样本的影响分析

**对抗扰动**：

假设对分数进行精心设计的扰动：
\begin{equation}
\tilde{s}_i = s_i + \delta_i, \quad \|\boldsymbol{\delta}\| \leq \epsilon \tag{57}
\end{equation}

目标是最大化输出变化$\|\tilde{\boldsymbol{o}} - \boldsymbol{o}\|$。

**一阶近似**：

对注意力输出进行Taylor展开：
\begin{equation}
\tilde{\boldsymbol{o}} \approx \boldsymbol{o} + \sum_i \frac{\partial \boldsymbol{o}}{\partial s_i} \delta_i \tag{58}
\end{equation}

**梯度计算**：

\begin{equation}
\frac{\partial \boldsymbol{o}}{\partial s_i} = \frac{\partial}{\partial s_i}\left(\sum_j \alpha_j \boldsymbol{v}_j\right) = \sum_j \frac{\partial \alpha_j}{\partial s_i} \boldsymbol{v}_j \tag{59}
\end{equation}

利用softmax的性质：
\begin{equation}
\frac{\partial \alpha_j}{\partial s_i} = \begin{cases}
\alpha_i(1 - \alpha_i), & i = j \\
-\alpha_i \alpha_j, & i \neq j
\end{cases} \tag{60}
\end{equation}

代入：
\begin{equation}
\frac{\partial \boldsymbol{o}}{\partial s_i} = \alpha_i(1-\alpha_i)\boldsymbol{v}_i - \alpha_i\sum_{j\neq i}\alpha_j\boldsymbol{v}_j = \alpha_i(\boldsymbol{v}_i - \boldsymbol{o}) \tag{61}
\end{equation}

**对抗方向**：

最大化$\|\tilde{\boldsymbol{o}} - \boldsymbol{o}\|$的扰动为：
\begin{equation}
\delta_i^* = \epsilon \cdot \text{sign}\left(\left\langle \frac{\partial \boldsymbol{o}}{\partial s_i}, \boldsymbol{v} \right\rangle\right) \tag{62}
\end{equation}

其中$\boldsymbol{v}$是某个目标方向。

**鲁棒性分析**：

注意到式(61)中的因子$\alpha_i(1-\alpha_i)$：
- 当$\alpha_i \approx 0$或$\alpha_i \approx 1$时，梯度很小（鲁棒）
- 当$\alpha_i \approx 0.5$时，梯度最大（易受攻击）

这与熵有关：均匀分布（$\alpha_i = \frac{1}{n}$）时熵最大，对抗扰动的影响也最大。

### 14. 缩放因子的理论分析

**标准Scaled Dot-Product Attention**：

\begin{equation}
\boldsymbol{o} = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}}\right) \boldsymbol{V} \tag{63}
\end{equation}

其中$d$是维度，$\sqrt{d}$是缩放因子。

**为什么需要缩放？**

假设$\boldsymbol{q}, \boldsymbol{k} \in \mathbb{R}^d$的元素独立同分布，$\mathbb{E}[q_i] = 0$，$\text{Var}(q_i) = \sigma^2$。

则：
\begin{equation}
s = \langle \boldsymbol{q}, \boldsymbol{k} \rangle = \sum_{i=1}^d q_i k_i \tag{64}
\end{equation}

期望：$\mathbb{E}[s] = 0$

方差：
\begin{equation}
\text{Var}(s) = \sum_{i=1}^d \text{Var}(q_i k_i) = d \cdot \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = d\sigma^4 \tag{65}
\end{equation}

**问题**：当$d$很大时，$s$的方差变得很大，导致$e^s$数值不稳定。

**解决方案**：除以$\sqrt{d}$：
\begin{equation}
s' = \frac{s}{\sqrt{d}} \Rightarrow \text{Var}(s') = \sigma^4 \tag{66}
\end{equation}

这样方差不随$d$变化！

**与熵的联系**：

如果不缩放，当$d \to \infty$时，$s_i$的方差趋向无穷，导致：
\begin{equation}
\alpha_{i^*} \to 1, \quad H \to 0 \tag{67}
\end{equation}

缩放后，熵保持在合理范围。

### 15. 信息瓶颈理论

**信息瓶颈（Information Bottleneck）**：

在深度学习中，隐藏层$Z$应该：
1. 最大化与标签$Y$的互信息：$I(Z; Y)$
2. 最小化与输入$X$的互信息：$I(Z; X)$

平衡两者：
\begin{equation}
\mathcal{L}_{\text{IB}} = I(Z; X) - \beta I(Z; Y) \tag{68}
\end{equation}

**注意力的视角**：

注意力机制相当于从$\{\boldsymbol{v}_i\}$中选择信息压缩到$\boldsymbol{o}$。

- 高熵$H$：保留更多信息（$I(Z; X)$大）
- 低熵$H$：压缩信息（$I(Z; X)$小）

**最优熵**：

存在一个最优熵$H^*$，平衡信息保留和压缩。初始化时应该接近$H^*$。

### 16. 多头注意力与熵

**多头注意力**：

\begin{equation}
\text{MultiHead}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \boldsymbol{W}^O \tag{69}
\end{equation}

其中：
\begin{equation}
\text{head}_i = \text{Attention}(\boldsymbol{Q}\boldsymbol{W}_i^Q, \boldsymbol{K}\boldsymbol{W}_i^K, \boldsymbol{V}\boldsymbol{W}_i^V) \tag{70}
\end{equation}

**每个头的熵**：

第$i$个头的熵为$H_i$。总的信息量大约是：
\begin{equation}
H_{\text{total}} \approx \sum_{i=1}^h H_i \tag{71}
\end{equation}

（近似，假设各头独立）

**多样性**：

理想情况下，不同的头应该关注不同的方面，即熵分布$\{H_i\}$应该有多样性。

### 17. 自注意力的谱分析

**自注意力矩阵**：

对于自注意力，$\boldsymbol{A} = \text{softmax}(\boldsymbol{Q}\boldsymbol{K}^\top / \sqrt{d})$是一个$n \times n$随机矩阵（每行和为1）。

**谱性质**：

1. 最大特征值$\lambda_1 = 1$（对应特征向量$\mathbf{1}$）
2. 其他特征值$|\lambda_i| < 1$

**与熵的关系**：

可以证明，熵与特征值分布有关：
\begin{equation}
H \approx -\sum_i \lambda_i \log \lambda_i \tag{72}
\end{equation}

（这是von Neumann熵的类比）

**Rank与信息**：

注意力矩阵的有效秩（effective rank）：
\begin{equation}
\text{rank}_{\text{eff}}(\boldsymbol{A}) = \exp(H) \tag{73}
\end{equation}

这度量了注意力分布的"有效宽度"。

### 18. 位置编码的影响

**绝对位置编码**：

添加位置编码$\boldsymbol{p}_i$后：
\begin{equation}
s_{ij} = \langle \boldsymbol{q}_i + \boldsymbol{p}_i, \boldsymbol{k}_j + \boldsymbol{p}_j \rangle \tag{74}
\end{equation}

**对熵的影响**：

位置编码增加了$s_{ij}$的多样性，通常会：
- 增加熵$H$（如果位置编码是随机的）
- 减少熵$H$（如果位置编码诱导了结构，如局部性）

**相对位置编码**：

\begin{equation}
s_{ij} = \langle \boldsymbol{q}_i, \boldsymbol{k}_j \rangle + b_{i-j} \tag{75}
\end{equation}

其中$b_k$是相对位置偏置。

这倾向于使注意力集中在附近的tokens，**降低熵**。

### 19. 注意力熵的实验观察

**Transformer训练中的熵变化**（经验观察）：

1. **初始阶段**：$H \approx \log n$（接近均匀）
2. **快速下降**：$H$迅速下降，模型学习到基本patterns
3. **震荡稳定**：$H$在某个值附近震荡
4. **过拟合**：如果继续训练，$H$可能继续下降（过拟合迹象）

**层间差异**：

- **浅层**：$H$较高，保留更多全局信息
- **深层**：$H$较低，关注特定的重要tokens

**任务依赖**：

- **语言建模**：$H$逐层下降
- **机器翻译**：encoder的$H$较高，decoder的$H$较低

### 20. 正则化与熵约束

**熵正则化**：

在损失函数中添加熵项：
\begin{equation}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda H(\boldsymbol{\alpha}) \tag{76}
\end{equation}

- $\lambda > 0$：鼓励高熵（平滑注意力）
- $\lambda < 0$：鼓励低熵（peaked注意力）

**效果**：

- 高熵正则：防止过拟合，增强鲁棒性
- 低熵正则：加速收敛，但可能过拟合

**与Label Smoothing的类比**：

Label smoothing也是一种熵正则化，鼓励输出分布有更高的熵。

### 21. 稀疏注意力与熵

**稀疏注意力**：

只计算部分attention scores：
\begin{equation}
\alpha_i = \begin{cases}
\frac{e^{s_i}}{\sum_{j \in \mathcal{N}(i)} e^{s_j}}, & j \in \mathcal{N}(i) \\
0, & \text{otherwise}
\end{cases} \tag{77}
\end{equation}

其中$\mathcal{N}(i)$是$i$的邻域。

**熵的变化**：

稀疏化通常降低有效序列长度$n$，因此最大熵变为：
\begin{equation}
H_{\max} = \log |\mathcal{N}(i)| < \log n \tag{78}
\end{equation}

**信息损失**：

稀疏化会损失一些信息，但换来了计算效率。

### 22. 连续vs.离散注意力

**离散注意力**（如Pointer Network）：

选择单个index：
\begin{equation}
i^* = \arg\max_i s_i \tag{79}
\end{equation}

熵为0（确定性）。

**Gumbel-Softmax**：

从softmax分布中采样，同时保持可微：
\begin{equation}
\alpha_i = \frac{e^{(s_i + g_i) / \tau}}{\sum_j e^{(s_j + g_j) / \tau}} \tag{80}
\end{equation}

其中$g_i \sim \text{Gumbel}(0, 1)$。

温度$\tau$控制熵：
- $\tau \to 0$：接近离散（低熵）
- $\tau$大：接近连续（高熵）

### 23. 注意力可视化与熵

**热图可视化**：

绘制注意力矩阵$\boldsymbol{A} = [\alpha_{ij}]$，颜色深浅表示权重大小。

**熵的视觉含义**：

- **高熵**：热图较均匀，颜色浅
- **低熵**：热图有明显的亮点，颜色深

**通过可视化诊断**：

1. **过于均匀**（$H \approx \log n$）：模型未学到有用信息
2. **过于集中**（$H \approx 0$）：可能过拟合或梯度消失
3. **适中熵**：理想状态

### 24. 理论与实践的Gap

**理论假设**：

- 噪声独立同分布
- 大$n$极限
- 线性近似

**实际情况**：

- 噪声可能相关（如系统性偏差）
- $n$有限（通常$n < 1000$）
- 非线性效应

**Gap的影响**：

理论预测的完美鲁棒性在实践中可能不成立，但**定性结论**仍然有用：
- Softmax注意力对独立噪声有一定鲁棒性
- 熵是训练动态的重要指标

### 25. 开放问题与未来方向

**问题1**：能否设计更鲁棒的注意力机制？

例如，使用其他归一化方法（如L1归一化）？

**问题2**：最优熵的理论表征

对于给定任务和数据，最优的注意力熵是多少？

**问题3**：熵的动态调整

能否自适应地调整温度参数，使熵在训练过程中自动达到最优值？

**问题4**：多模态注意力的熵

在视觉-语言模型中，cross-modal attention的熵有何特点？

**未来方向**：
1. 自适应温度学习
2. 熵感知的正则化
3. 鲁棒性的理论保证
4. 高效的熵计算方法

---

## 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Pereyra et al., "Regularizing Neural Networks by Penalizing Confident Output Distributions", ICLR 2017
3. Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017
4. Cordonnier et al., "On the Relationship between Self-Attention and Convolutional Layers", ICLR 2020
5. Kobayashi et al., "Attention is Not Only a Weight: Analyzing Transformers with Vector Norms", EMNLP 2020

## 总结

本文深入分析了Softmax注意力的两个关键性质：

1. **鲁棒性**：
   - 对独立同分布噪声有天然鲁棒性
   - 噪声的指数矩被约掉
   - Lipschitz连续性保证稳定性

2. **信息量**：
   - 熵是不确定性和信息量的度量
   - 初始化应保证足够的熵（可学习信息）
   - 温度参数调节熵的范围

**关键洞察**：
- Softmax结构隐含了平均化效应，天然具有降噪能力
- 熵提供了理解训练动态的信息论视角
- 合适的初始化和温度选择对训练至关重要

这些理论分析为Transformer等attention-based模型的设计和调试提供了坚实的理论基础。

