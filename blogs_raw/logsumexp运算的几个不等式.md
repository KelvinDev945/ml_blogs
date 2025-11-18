---
title: logsumexp运算的几个不等式
slug: logsumexp运算的几个不等式
date: 2022-05-10
tags: 不等式, 函数, 生成模型, attention, 优化
status: pending
---

# logsumexp运算的几个不等式

**原文链接**: [https://spaces.ac.cn/archives/9070](https://spaces.ac.cn/archives/9070)

**发布日期**: 

---

$\text{logsumexp}$是机器学习经常遇到的运算，尤其是交叉熵的相关实现和推导中都会经常出现，同时它还是$\max$的光滑近似（参考[《寻求一个光滑的最大值函数》](/archives/3290)）。设$x=(x_1,x_2,\cdots,x_n)$，$\text{logsumexp}$定义为  
\begin{equation}\text{logsumexp}(x)=\log\sum_{i=1}^n e^{x_i}\end{equation}  
本文来介绍$\text{logsumexp}$的几个在理论推导中可能用得到的不等式。

## 基本界 #

记$x_{\max} = \max(x_1,x_2,\cdots,x_n)$，那么显然有  
\begin{equation}e^{x_{\max}} < \sum_{i=1}^n e^{x_i} \leq \sum_{i=1}^n e^{x_{\max}} = ne^{x_{\max}}\end{equation}  
各端取对数即得  
\begin{equation}x_{\max} < \text{logsumexp}(x) \leq x_{\max} + \log n\end{equation}  
这是关于$\text{logsumexp}$上下界的最基本结果，它表明$\text{logsumexp}$对$\max$的近似误差不超过$\log n$。注意这个误差跟$x$本身无关，于是我们有  
\begin{equation}x_{\max}/\tau < \text{logsumexp}(x/\tau) \leq x_{\max}/\tau + \log n\end{equation}  
各端乘以$\tau$得到  
\begin{equation}x_{\max} < \tau\text{logsumexp}(x/\tau) \leq x_{\max} + \tau\log n\end{equation}  
当$\tau\to 0$时，误差就趋于0了，这告诉我们可以通过降低温度参数来提高对$\max$的近似程度。

## 平均界 #

我们知道$e^x$是凸函数，满足[詹森不等式](https://en.wikipedia.org/wiki/Jensen%27s_inequality)$\mathbb{E}[e^{x}]\geq e^{\mathbb{E}[x]}$，因此  
\begin{equation}\frac{1}{n}\sum_{i=1}^n e^{x_i}\geq e^{\bar{x}}\end{equation}  
这里$\bar{x}=\frac{1}{n}\sum\limits_{i=1}^n x_i$，两边乘以$n$后取对数得  
\begin{equation}\text{logsumexp}(x)\geq \bar{x} + \log n\end{equation}  
这是关于$\text{logsumexp}$下界的另一个结果。该结果可以进一步推广到加权平均的情形：设有$p_1,p_2,\cdots,p_n\geq 0$且$\sum\limits_{i=1}^n p_i = 1$，由柯西不等式得  
\begin{equation}\left[\sum_{i=1}^n (e^{x_i/2})^2\right]\left[\sum_{i=1}^n p_i^2\right]\geq \left[\sum_{i=1}^n p_i e^{x_i/2}\right]^2\end{equation}  
对右端方括号内的式子应用詹森不等式得到  
\begin{equation}\left[\sum_{i=1}^n p_i e^{x_i/2}\right]^2\geq \left[e^{\left(\sum\limits_{i=1}^n p_i x_i/2\right)}\right]^2 = e^{\left(\sum\limits_{i=1}^n p_i x_i\right)}\end{equation}  
各式两端取对数，整理得到  
\begin{equation}\text{logsumexp}(x)\geq \sum_{i=1}^n p_i x_i - \log\sum_{i=1}^n p_i^2\end{equation}  
如果开始不用柯西不等式而是用更一般的[Hölder不等式](https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality)，那么还可以得到  
\begin{equation}\text{logsumexp}(x)\geq \sum_{i=1}^n p_i x_i - \frac{1}{t-1}\log\sum_{i=1}^n p_i^t,\quad \forall t > 1\end{equation}  
特别地，取$t\to 1$的极限，我们可以得到  
\begin{equation}\text{logsumexp}(x)\geq \sum_{i=1}^n p_i x_i - \sum_{i=1}^n p_i \log p_i\end{equation}  
它可以等价地改写为$\sum\limits_{i=1}^n p_i \log \frac{p_i}{e^{x_i}/Z} \geq 0$，其中$Z=e^{\text{logsumexp}(x)}$是归一化因子，所以它实际就是两个分布的$KL$散度。

## L约束 #

在无穷范数下，$\text{logsumexp}$还满足Lipschitz约束，即  
\begin{equation}|\text{logsumexp}(x) - \text{logsumexp}(y)| \leq |x - y|_{\infty}\end{equation}  
这里的$|x-y|_{\infty} = \max\limits_i |x_i - y_i|$（其实记为$|x - y|_{\max}$还更直观一些）。证明也不算困难，定义  
\begin{equation}f(t) = \text{logsumexp}(tx + (1-t)y),\quad t\in[0, 1]\end{equation}  
将它视为关于$t$的一元函数，由[中值定理](https://en.wikipedia.org/wiki/Mean_value_theorem)知存在$\varepsilon\in(0, 1)$，使得  
\begin{equation}f'(\varepsilon) = \frac{f(1) - f(0)}{1 - 0} = \text{logsumexp}(x) - \text{logsumexp}(y) \end{equation}  
不难求出  
\begin{equation}f'(\varepsilon) = \frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}(x_i - y_i)}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}} \end{equation}  
所以  
\begin{equation}\begin{aligned}&\,|\text{logsumexp}(x) - \text{logsumexp}(y)| = \left|\frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}(x_i - y_i)}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}}\right| \\\  
\leq &\, \frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i} |x_i - y_i|}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}} \leq \frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i} |x - y|_{\infty}}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}} = |x - y|_{\infty}  
\end{aligned}\end{equation}

## 凸函数 #

最后是一个很强的结论：$\text{logsumexp}$还是一个凸函数！这意味着凸函数相关的所有不等式都适用于$\text{logsumexp}$，比如最基本的詹森不等式：  
\begin{equation} \mathbb{E}[\text{logsumexp}(x)] \geq \text{logsumexp}(\mathbb{E}[x])\end{equation}

要证明$\text{logsumexp}$是凸函数，就是要证明对于$\forall t\in[0, 1]$，都成立  
\begin{equation} t\text{logsumexp}(x) + (1-t)\text{logsumexp}(y)\geq \text{logsumexp}(tx + (1-t)y)\end{equation}  
证明过程其实就是[Hölder不等式](https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality)的基本应用。具体来说，我们有  
\begin{equation}t\text{logsumexp}(x) + (1-t)\text{logsumexp}(y) = \log\left(\sum_{i=1}^n e^{x_i}\right)^t \left(\sum_{i=1}^n e^{y_i}\right)^{(1-t)}\end{equation}  
现在直接应用Hölder不等式就可以得到  
\begin{equation}\log\left(\sum_{i=1}^n e^{x_i}\right)^t \left(\sum_{i=1}^n e^{y_i}\right)^{(1-t)}\geq \log\sum_{i=1}^n e^{tx_i + (1-t)y_i} = \text{logsumexp}(tx + (1-t)y)\end{equation}  
这就证明了$\text{logsumexp}$是凸函数。

## 文末结 #

主要总结了$\text{logsumexp}$运算的相关不等式，以备不时之需。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9070>_

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

苏剑林. (May. 10, 2022). 《logsumexp运算的几个不等式 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9070>

@online{kexuefm-9070,  
title={logsumexp运算的几个不等式},  
author={苏剑林},  
year={2022},  
month={May},  
url={\url{https://spaces.ac.cn/archives/9070}},  
} 


---

## 详细数学推导与注释

本节提供logsumexp运算的完整数学推导，包括基本性质、不等式证明、数值稳定性分析和实践应用。

### 1. 基本定义与性质

#### 1.1 定义与基本形式

logsumexp函数定义为：
\begin{equation}\text{logsumexp}(x) = \log\sum_{i=1}^n e^{x_i}\tag{1}\end{equation}

**数学直觉**：该函数将指数求和的结果取对数，可以视为在对数空间中的"软最大值"运算。

**基本性质**：
\begin{equation}\text{logsumexp}(x + c) = \text{logsumexp}(x) + c,\quad \forall c\in\mathbb{R}\tag{2}\end{equation}

**证明**：
\begin{align}
\text{logsumexp}(x + c) &= \log\sum_{i=1}^n e^{x_i+c}\tag{3}\\
&= \log\left(e^c\sum_{i=1}^n e^{x_i}\right)\tag{4}\\
&= \log e^c + \log\sum_{i=1}^n e^{x_i}\tag{5}\\
&= c + \text{logsumexp}(x)\tag{6}
\end{align}

这个性质在数值稳定性计算中至关重要。

#### 1.2 与最大值的关系

记$x_{\max} = \max(x_1,x_2,\cdots,x_n)$，则：
\begin{equation}x_{\max} \leq \text{logsumexp}(x) \leq x_{\max} + \log n\tag{7}\end{equation}

**详细证明**：

**下界**：因为$e^{x_i} > 0$对所有$i$成立，所以：
\begin{equation}\sum_{i=1}^n e^{x_i} > e^{x_{\max}}\tag{8}\end{equation}

两边取对数得：
\begin{equation}\text{logsumexp}(x) > x_{\max}\tag{9}\end{equation}

注意这里是严格不等号，因为至少有两个不同的$x_i$时求和严格大于最大项。当所有$x_i$相等时取等号。

**上界**：因为$e^{x_i} \leq e^{x_{\max}}$对所有$i$成立，所以：
\begin{equation}\sum_{i=1}^n e^{x_i} \leq \sum_{i=1}^n e^{x_{\max}} = ne^{x_{\max}}\tag{10}\end{equation}

两边取对数得：
\begin{equation}\text{logsumexp}(x) \leq x_{\max} + \log n\tag{11}\end{equation}

**误差分析**：定义近似误差为：
\begin{equation}\varepsilon(x) = \text{logsumexp}(x) - x_{\max}\tag{12}\end{equation}

则$0 < \varepsilon(x) \leq \log n$，且：
- 当$n=1$时，$\varepsilon(x) = 0$
- 当所有$x_i$相等时，$\varepsilon(x) = \log n$（达到上界）
- 当其他$x_i \ll x_{\max}$时，$\varepsilon(x) \approx 0$

### 2. 基本界的详细分析

#### 2.1 温度参数的影响

引入温度参数$\tau > 0$，定义：
\begin{equation}\text{logsumexp}_{\tau}(x) = \tau\log\sum_{i=1}^n e^{x_i/\tau}\tag{13}\end{equation}

**性质**：
\begin{equation}x_{\max} \leq \text{logsumexp}_{\tau}(x) \leq x_{\max} + \tau\log n\tag{14}\end{equation}

**证明**：令$y_i = x_i/\tau$，则：
\begin{align}
\text{logsumexp}_{\tau}(x) &= \tau\text{logsumexp}(y)\tag{15}\\
&\leq \tau(\max_i y_i + \log n)\tag{16}\\
&= \tau\cdot\frac{x_{\max}}{\tau} + \tau\log n\tag{17}\\
&= x_{\max} + \tau\log n\tag{18}
\end{align}

下界同理可证。

**温度参数的作用**：
- $\tau \to 0$时：$\text{logsumexp}_{\tau}(x) \to x_{\max}$（硬最大值）
- $\tau \to \infty$时：误差$\tau\log n$增大，近似变差
- $\tau = 1$时：标准logsumexp

**实践应用**：在softmax中，较小的$\tau$产生更"尖锐"的分布，较大的$\tau$产生更"平滑"的分布。

#### 2.2 紧界估计

对于特殊情况，我们可以得到更紧的界。设$x_1 \geq x_2 \geq \cdots \geq x_n$（已排序），定义：
\begin{equation}\delta_i = x_1 - x_i,\quad i=2,\ldots,n\tag{19}\end{equation}

则：
\begin{equation}\text{logsumexp}(x) = x_1 + \log\left(1 + \sum_{i=2}^n e^{-\delta_i}\right)\tag{20}\end{equation}

**推导**：
\begin{align}
\text{logsumexp}(x) &= \log\sum_{i=1}^n e^{x_i}\tag{21}\\
&= \log\left(e^{x_1}\sum_{i=1}^n e^{x_i-x_1}\right)\tag{22}\\
&= x_1 + \log\left(1 + \sum_{i=2}^n e^{x_i-x_1}\right)\tag{23}\\
&= x_1 + \log\left(1 + \sum_{i=2}^n e^{-\delta_i}\right)\tag{24}
\end{align}

**误差估计**：
\begin{equation}0 < \log\left(1 + \sum_{i=2}^n e^{-\delta_i}\right) \leq \log n\tag{25}\end{equation}

如果$\delta_2$（第二大值与最大值的差）比较大，则误差接近0。

### 3. 平均界的深入探讨

#### 3.1 詹森不等式应用

设$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$为算术平均，由于$e^x$是凸函数，根据詹森不等式：
\begin{equation}\mathbb{E}[e^x] \geq e^{\mathbb{E}[x]}\tag{26}\end{equation}

应用到均匀分布上：
\begin{equation}\frac{1}{n}\sum_{i=1}^n e^{x_i} \geq e^{\bar{x}}\tag{27}\end{equation}

两边乘以$n$后取对数：
\begin{equation}\text{logsumexp}(x) \geq \bar{x} + \log n\tag{28}\end{equation}

**等号成立条件**：当且仅当所有$x_i$相等时取等号。

**几何意义**：logsumexp总是大于等于算术平均加上$\log n$的修正项。

#### 3.2 加权詹森不等式

对于权重$p_1,\ldots,p_n \geq 0$且$\sum_{i=1}^n p_i = 1$，定义加权平均：
\begin{equation}\bar{x}_p = \sum_{i=1}^n p_i x_i\tag{29}\end{equation}

**基本不等式**：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - \log\sum_{i=1}^n p_i^2\tag{30}\end{equation}

**详细证明**：

**步骤1**：柯西不等式
\begin{equation}\left[\sum_{i=1}^n a_i^2\right]\left[\sum_{i=1}^n b_i^2\right] \geq \left[\sum_{i=1}^n a_ib_i\right]^2\tag{31}\end{equation}

令$a_i = e^{x_i/2}$，$b_i = p_i$：
\begin{equation}\left[\sum_{i=1}^n e^{x_i}\right]\left[\sum_{i=1}^n p_i^2\right] \geq \left[\sum_{i=1}^n p_i e^{x_i/2}\right]^2\tag{32}\end{equation}

**步骤2**：对右侧应用詹森不等式
\begin{align}
\sum_{i=1}^n p_i e^{x_i/2} &\geq e^{\sum_{i=1}^n p_i \cdot x_i/2}\tag{33}\\
&= e^{\bar{x}_p/2}\tag{34}
\end{align}

**步骤3**：代入并整理
\begin{equation}\left[\sum_{i=1}^n e^{x_i}\right]\left[\sum_{i=1}^n p_i^2\right] \geq e^{\bar{x}_p}\tag{35}\end{equation}

两边取对数：
\begin{equation}\text{logsumexp}(x) + \log\sum_{i=1}^n p_i^2 \geq \bar{x}_p\tag{36}\end{equation}

移项得：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - \log\sum_{i=1}^n p_i^2\tag{37}\end{equation}

#### 3.3 Hölder不等式推广

Hölder不等式指出，对于$t > 1$和$s = \frac{t}{t-1}$（共轭指数），有：
\begin{equation}\sum_{i=1}^n |a_ib_i| \leq \left(\sum_{i=1}^n |a_i|^t\right)^{1/t}\left(\sum_{i=1}^n |b_i|^s\right)^{1/s}\tag{38}\end{equation}

令$a_i = e^{x_i/t}$，$b_i = p_i$：
\begin{equation}\sum_{i=1}^n p_i e^{x_i/t} \leq \left(\sum_{i=1}^n e^{x_i}\right)^{1/t}\left(\sum_{i=1}^n p_i^s\right)^{1/s}\tag{39}\end{equation}

**步骤1**：应用詹森不等式到左侧
\begin{equation}\sum_{i=1}^n p_i e^{x_i/t} \geq e^{\bar{x}_p/t}\tag{40}\end{equation}

**步骤2**：结合Hölder不等式
\begin{equation}e^{\bar{x}_p/t} \leq \left(\sum_{i=1}^n e^{x_i}\right)^{1/t}\left(\sum_{i=1}^n p_i^s\right)^{1/s}\tag{41}\end{equation}

两边取$t$次方：
\begin{equation}e^{\bar{x}_p} \leq \left(\sum_{i=1}^n e^{x_i}\right)\left(\sum_{i=1}^n p_i^s\right)^{t/s}\tag{42}\end{equation}

注意到$t/s = t-1$，所以：
\begin{equation}e^{\bar{x}_p} \leq \left(\sum_{i=1}^n e^{x_i}\right)\left(\sum_{i=1}^n p_i^s\right)^{t-1}\tag{43}\end{equation}

两边取对数：
\begin{equation}\bar{x}_p \leq \text{logsumexp}(x) + (t-1)\log\sum_{i=1}^n p_i^s\tag{44}\end{equation}

其中$s = \frac{t}{t-1}$，移项得：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - (t-1)\log\sum_{i=1}^n p_i^{t/(t-1)}\tag{45}\end{equation}

改写为：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - \frac{1}{t-1}\log\sum_{i=1}^n p_i^{t/(t-1)}\tag{46}\end{equation}

令$t' = \frac{t}{t-1}$，当$t > 1$时$t' > 1$，可得：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - \frac{1}{t'-1}\log\sum_{i=1}^n p_i^{t'},\quad \forall t' > 1\tag{47}\end{equation}

#### 3.4 极限情况：KL散度

当$t \to 1$时，使用洛必达法则：
\begin{align}
\lim_{t\to 1}\frac{\log\sum_{i=1}^n p_i^t}{t-1} &= \lim_{t\to 1}\frac{d}{dt}\log\sum_{i=1}^n p_i^t\tag{48}\\
&= \lim_{t\to 1}\frac{\sum_{i=1}^n p_i^t\log p_i}{\sum_{i=1}^n p_i^t}\tag{49}\\
&= \sum_{i=1}^n p_i\log p_i\tag{50}
\end{align}

因此：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - \sum_{i=1}^n p_i\log p_i\tag{51}\end{equation}

**KL散度形式**：定义$q_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$为归一化后的分布，则：
\begin{equation}\sum_{i=1}^n p_i\log p_i - \sum_{i=1}^n p_i\log q_i = \text{KL}(p\|q)\tag{52}\end{equation}

展开第二项：
\begin{align}
\sum_{i=1}^n p_i\log q_i &= \sum_{i=1}^n p_i\log\frac{e^{x_i}}{\sum_j e^{x_j}}\tag{53}\\
&= \sum_{i=1}^n p_i x_i - \log\sum_j e^{x_j}\tag{54}\\
&= \bar{x}_p - \text{logsumexp}(x)\tag{55}
\end{align}

因此：
\begin{equation}\text{KL}(p\|q) = \sum_{i=1}^n p_i\log p_i - \bar{x}_p + \text{logsumexp}(x)\tag{56}\end{equation}

移项得：
\begin{equation}\text{logsumexp}(x) = \bar{x}_p - \sum_{i=1}^n p_i\log p_i + \text{KL}(p\|q)\tag{57}\end{equation}

由于$\text{KL}(p\|q) \geq 0$，我们再次得到：
\begin{equation}\text{logsumexp}(x) \geq \bar{x}_p - \sum_{i=1}^n p_i\log p_i\tag{58}\end{equation}

等号成立当且仅当$p_i = q_i$，即$p_i \propto e^{x_i}$。

### 4. Lipschitz连续性

#### 4.1 Lipschitz常数

**定理**：logsumexp函数关于无穷范数是1-Lipschitz连续的：
\begin{equation}|\text{logsumexp}(x) - \text{logsumexp}(y)| \leq \|x - y\|_{\infty}\tag{59}\end{equation}

其中$\|x - y\|_{\infty} = \max_i |x_i - y_i|$。

**证明准备**：定义辅助函数
\begin{equation}f(t) = \text{logsumexp}(tx + (1-t)y),\quad t\in[0,1]\tag{60}\end{equation}

**步骤1**：计算导数

对$t$求导：
\begin{align}
f'(t) &= \frac{d}{dt}\log\sum_{i=1}^n e^{tx_i + (1-t)y_i}\tag{61}\\
&= \frac{\sum_{i=1}^n e^{tx_i + (1-t)y_i}\cdot(x_i - y_i)}{\sum_{i=1}^n e^{tx_i + (1-t)y_i}}\tag{62}\\
&= \sum_{i=1}^n w_i(t)(x_i - y_i)\tag{63}
\end{align}

其中权重为：
\begin{equation}w_i(t) = \frac{e^{tx_i + (1-t)y_i}}{\sum_j e^{tx_j + (1-t)y_j}}\tag{64}\end{equation}

注意到$w_i(t) \geq 0$且$\sum_i w_i(t) = 1$，所以$f'(t)$是$x_i - y_i$的凸组合。

**步骤2**：应用中值定理

由中值定理，存在$\varepsilon \in (0,1)$使得：
\begin{equation}f(1) - f(0) = f'(\varepsilon)\tag{65}\end{equation}

即：
\begin{equation}\text{logsumexp}(x) - \text{logsumexp}(y) = \sum_{i=1}^n w_i(\varepsilon)(x_i - y_i)\tag{66}\end{equation}

**步骤3**：估计上界

取绝对值：
\begin{align}
|\text{logsumexp}(x) - \text{logsumexp}(y)| &= \left|\sum_{i=1}^n w_i(\varepsilon)(x_i - y_i)\right|\tag{67}\\
&\leq \sum_{i=1}^n w_i(\varepsilon)|x_i - y_i|\tag{68}\\
&\leq \sum_{i=1}^n w_i(\varepsilon)\cdot\|x - y\|_{\infty}\tag{69}\\
&= \|x - y\|_{\infty}\tag{70}
\end{align}

这就证明了1-Lipschitz性质。

#### 4.2 梯度的性质

logsumexp的梯度为：
\begin{equation}\nabla_x\text{logsumexp}(x) = \left[\frac{e^{x_1}}{\sum_j e^{x_j}},\ldots,\frac{e^{x_n}}{\sum_j e^{x_j}}\right]^{\top}\tag{71}\end{equation}

这正是softmax函数！记为：
\begin{equation}\nabla_x\text{logsumexp}(x) = \text{softmax}(x)\tag{72}\end{equation}

**性质**：
1. $\sum_i [\nabla_x\text{logsumexp}(x)]_i = 1$（概率分布）
2. $[\nabla_x\text{logsumexp}(x)]_i \in (0,1)$
3. $\|\nabla_x\text{logsumexp}(x)\|_1 = 1$
4. $\|\nabla_x\text{logsumexp}(x)\|_{\infty} \leq 1$

这些性质确保了Lipschitz常数为1。

### 5. 凸性分析

#### 5.1 凸函数的证明

**定理**：logsumexp是凸函数。

**证明方法1**：Hessian矩阵

计算二阶导数，Hessian矩阵的$(i,j)$元素为：
\begin{equation}H_{ij} = \frac{\partial^2}{\partial x_i\partial x_j}\text{logsumexp}(x)\tag{73}\end{equation}

令$p_i = \frac{e^{x_i}}{\sum_k e^{x_k}}$（softmax），可以证明：
\begin{equation}H_{ij} = \begin{cases}
p_i(1-p_i) & \text{if } i=j\\
-p_ip_j & \text{if } i\neq j
\end{cases}\tag{74}\end{equation}

**验证半正定性**：对任意向量$v$，
\begin{align}
v^{\top}Hv &= \sum_{i=1}^n v_i^2 p_i(1-p_i) - \sum_{i\neq j}v_iv_jp_ip_j\tag{75}\\
&= \sum_{i=1}^n v_i^2p_i - \sum_{i=1}^n v_i^2p_i^2 - \sum_{i\neq j}v_iv_jp_ip_j\tag{76}\\
&= \sum_{i=1}^n v_i^2p_i - \sum_{i,j}v_iv_jp_ip_j\tag{77}\\
&= \sum_{i=1}^n v_i^2p_i - \left(\sum_{i}v_ip_i\right)^2\tag{78}
\end{align}

由柯西-施瓦茨不等式：
\begin{equation}\left(\sum_{i}v_ip_i\right)^2 \leq \left(\sum_i v_i^2p_i\right)\left(\sum_i p_i\right) = \sum_i v_i^2p_i\tag{79}\end{equation}

因此$v^{\top}Hv \geq 0$，Hessian半正定，logsumexp是凸函数。

**证明方法2**：定义验证

对于$\lambda\in[0,1]$和向量$x,y$，需要证明：
\begin{equation}\text{logsumexp}(\lambda x + (1-\lambda)y) \leq \lambda\text{logsumexp}(x) + (1-\lambda)\text{logsumexp}(y)\tag{80}\end{equation}

**步骤1**：展开左侧
\begin{align}
\text{logsumexp}(\lambda x + (1-\lambda)y) &= \log\sum_{i=1}^n e^{\lambda x_i + (1-\lambda)y_i}\tag{81}\\
&= \log\sum_{i=1}^n (e^{x_i})^{\lambda}(e^{y_i})^{1-\lambda}\tag{82}
\end{align}

**步骤2**：应用Hölder不等式

对于$p = 1/\lambda$，$q = 1/(1-\lambda)$（满足$1/p + 1/q = 1$）：
\begin{equation}\sum_{i=1}^n (e^{x_i})^{\lambda}(e^{y_i})^{1-\lambda} \leq \left(\sum_{i=1}^n e^{x_i}\right)^{\lambda}\left(\sum_{i=1}^n e^{y_i}\right)^{1-\lambda}\tag{83}\end{equation}

**步骤3**：取对数
\begin{align}
\log\sum_{i=1}^n (e^{x_i})^{\lambda}(e^{y_i})^{1-\lambda} &\leq \log\left[\left(\sum_{i=1}^n e^{x_i}\right)^{\lambda}\left(\sum_{i=1}^n e^{y_i}\right)^{1-\lambda}\right]\tag{84}\\
&= \lambda\log\sum_{i=1}^n e^{x_i} + (1-\lambda)\log\sum_{i=1}^n e^{y_i}\tag{85}\\
&= \lambda\text{logsumexp}(x) + (1-\lambda)\text{logsumexp}(y)\tag{86}
\end{align}

这就证明了凸性。

#### 5.2 凸性的应用

**詹森不等式**：对于随机变量$X$，
\begin{equation}\mathbb{E}[\text{logsumexp}(X)] \geq \text{logsumexp}(\mathbb{E}[X])\tag{87}\end{equation}

**最优化**：在凸优化问题中，logsumexp可以作为光滑的约束或目标函数。

**次梯度**：虽然logsumexp处处可导，但其作为凸函数的次梯度集合为：
\begin{equation}\partial\text{logsumexp}(x) = \{\text{softmax}(x)\}\tag{88}\end{equation}

### 6. 数值稳定性

#### 6.1 直接计算的问题

直接计算$\log\sum_{i=1}^n e^{x_i}$可能导致：
- **上溢**：当某些$x_i$很大时，$e^{x_i}$超出浮点数表示范围
- **下溢**：当所有$x_i$都很小（负数）时，$e^{x_i}$接近0，求和后再取对数损失精度

**示例**：假设$x = [1000, 1001, 1002]$，直接计算：
\begin{equation}e^{1000} \approx 10^{434}\tag{89}\end{equation}
这远超出double精度浮点数的表示范围（约$10^{308}$）。

#### 6.2 数值稳定的计算方法

**Log-Sum-Exp技巧**：利用性质(2)，令$c = x_{\max}$：
\begin{align}
\text{logsumexp}(x) &= \text{logsumexp}(x - x_{\max}) + x_{\max}\tag{90}\\
&= \log\sum_{i=1}^n e^{x_i - x_{\max}} + x_{\max}\tag{91}\\
&= x_{\max} + \log\sum_{i=1}^n e^{x_i - x_{\max}}\tag{92}
\end{align}

**优势**：
1. 所有$x_i - x_{\max} \leq 0$，因此$e^{x_i - x_{\max}} \in (0,1]$，不会上溢
2. 至少有一项$e^{0} = 1$，求和结果至少为1，取对数不会有问题
3. 最终加回$x_{\max}$得到正确结果

**Python实现**：
```python
import numpy as np

def logsumexp_stable(x):
    """数值稳定的logsumexp实现"""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))
```

**示例**：对于$x = [1000, 1001, 1002]$：
\begin{align}
\text{logsumexp}(x) &= 1002 + \log(e^{-2} + e^{-1} + e^0)\tag{93}\\
&= 1002 + \log(0.135 + 0.368 + 1)\tag{94}\\
&= 1002 + \log(1.503)\tag{95}\\
&= 1002 + 0.407\tag{96}\\
&\approx 1002.407\tag{97}
\end{align}

所有中间计算都在安全范围内。

#### 6.3 向量化与批处理

对于矩阵$X\in\mathbb{R}^{m\times n}$，沿某个维度计算logsumexp：

**按行计算**（对每行的$n$个元素）：
```python
def logsumexp_rows(X):
    """对每行计算logsumexp"""
    x_max = np.max(X, axis=1, keepdims=True)
    return x_max.squeeze() + np.log(np.sum(np.exp(X - x_max), axis=1))
```

**按列计算**（对每列的$m$个元素）：
```python
def logsumexp_cols(X):
    """对每列计算logsumexp"""
    x_max = np.max(X, axis=0, keepdims=True)
    return x_max.squeeze() + np.log(np.sum(np.exp(X - x_max), axis=0))
```

### 7. 计算示例

#### 7.1 基本示例

**示例1**：均匀情况
\begin{equation}x = [0, 0, 0, 0]\quad(n=4)\tag{98}\end{equation}
\begin{align}
\text{logsumexp}(x) &= \log(e^0 + e^0 + e^0 + e^0)\tag{99}\\
&= \log(4)\tag{100}\\
&\approx 1.386\tag{101}
\end{align}

验证上下界：$x_{\max} = 0$，$\bar{x} = 0$
- 上界：$0 + \log 4 = 1.386$ ✓
- 下界：$0 + \log 4 = 1.386$ ✓（等号成立）

**示例2**：递增序列
\begin{equation}x = [0, 1, 2, 3]\tag{102}\end{equation}
\begin{align}
\text{logsumexp}(x) &= 3 + \log(e^{-3} + e^{-2} + e^{-1} + e^0)\tag{103}\\
&= 3 + \log(0.050 + 0.135 + 0.368 + 1.000)\tag{104}\\
&= 3 + \log(1.553)\tag{105}\\
&\approx 3.440\tag{106}
\end{align}

验证：
- $x_{\max} = 3$，误差$= 0.440 < \log 4 = 1.386$ ✓
- $\bar{x} = 1.5$，$\bar{x} + \log 4 = 2.886 < 3.440$ ✓

#### 7.2 温度参数示例

考虑$x = [0, 1, 2]$，不同温度下的结果：

**$\tau = 0.1$**（低温）：
\begin{align}
\text{logsumexp}_{0.1}(x) &= 0.1\log(e^0 + e^{10} + e^{20})\tag{107}\\
&\approx 0.1 \times 20\tag{108}\\
&= 2\tag{109}
\end{align}
接近$\max(x) = 2$。

**$\tau = 1$**（标准）：
\begin{equation}\text{logsumexp}_1(x) = \log(1 + e + e^2) \approx 2.407\tag{110}\end{equation}

**$\tau = 10$**（高温）：
\begin{align}
\text{logsumexp}_{10}(x) &= 10\log(e^0 + e^{0.1} + e^{0.2})\tag{111}\\
&= 10\log(3.315)\tag{112}\\
&\approx 11.99\tag{113}
\end{align}
接近$\bar{x}\cdot n = 1 \times 3 = 3$加上较大的修正。

### 8. 实践应用

#### 8.1 Softmax计算

Softmax函数定义为：
\begin{equation}\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}\tag{114}\end{equation}

使用logsumexp可以写成：
\begin{equation}\log\text{softmax}(x)_i = x_i - \text{logsumexp}(x)\tag{115}\end{equation}

**数值稳定实现**：
```python
def log_softmax_stable(x):
    """数值稳定的log-softmax"""
    return x - logsumexp_stable(x)

def softmax_stable(x):
    """数值稳定的softmax"""
    return np.exp(log_softmax_stable(x))
```

#### 8.2 交叉熵损失

分类问题的交叉熵损失：
\begin{equation}\mathcal{L} = -\sum_{i=1}^n y_i\log\text{softmax}(x)_i\tag{116}\end{equation}

其中$y$是one-hot标签。设$y_k = 1$（其他为0），则：
\begin{align}
\mathcal{L} &= -\log\text{softmax}(x)_k\tag{117}\\
&= -x_k + \text{logsumexp}(x)\tag{118}
\end{align}

这避免了显式计算softmax，提高数值稳定性。

#### 8.3 注意力机制

Scaled Dot-Product Attention中：
\begin{equation}\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V\tag{119}\end{equation}

计算$\frac{QK^{\top}}{\sqrt{d_k}}$后，对每行应用softmax。使用logsumexp技巧确保大模型训练的稳定性。

### 9. 理论应用拓展

#### 9.1 最大值的光滑近似

logsumexp作为max的光滑近似，满足：
\begin{equation}\lim_{\beta\to\infty}\frac{1}{\beta}\text{logsumexp}(\beta x) = \max(x)\tag{120}\end{equation}

**证明**：
\begin{align}
\frac{1}{\beta}\text{logsumexp}(\beta x) &= \frac{1}{\beta}\log\sum_{i=1}^n e^{\beta x_i}\tag{121}\\
&= \frac{1}{\beta}\log\left(e^{\beta x_{\max}}\sum_{i=1}^n e^{\beta(x_i - x_{\max})}\right)\tag{122}\\
&= x_{\max} + \frac{1}{\beta}\log\sum_{i=1}^n e^{\beta(x_i - x_{\max})}\tag{123}
\end{align}

当$\beta\to\infty$时，只有$x_i = x_{\max}$的项贡献$e^0 = 1$，其他项趋于0：
\begin{equation}\lim_{\beta\to\infty}\frac{1}{\beta}\log\sum_{i=1}^n e^{\beta(x_i - x_{\max})} = 0\tag{124}\end{equation}

因此：
\begin{equation}\lim_{\beta\to\infty}\frac{1}{\beta}\text{logsumexp}(\beta x) = x_{\max}\tag{125}\end{equation}

#### 9.2 凸优化中的应用

在凸优化中，用logsumexp替代max可以将非光滑问题转化为光滑问题：
\begin{equation}\min_x \max_i f_i(x) \approx \min_x \frac{1}{\beta}\text{logsumexp}(\beta f(x))\tag{126}\end{equation}

这使得可以使用梯度下降等光滑优化方法。

### 10. 总结与实践建议

**关键不等式汇总**：
1. 基本界：$x_{\max} \leq \text{logsumexp}(x) \leq x_{\max} + \log n$
2. 平均界：$\text{logsumexp}(x) \geq \bar{x} + \log n$
3. 加权界：$\text{logsumexp}(x) \geq \bar{x}_p - \sum_i p_i\log p_i$
4. Lipschitz：$|\text{logsumexp}(x) - \text{logsumexp}(y)| \leq \|x-y\|_{\infty}$

**数值计算建议**：
1. 始终使用$x_{\max}$技巧避免上溢
2. 向量化操作提高效率
3. 注意维度保持（keepdims=True）
4. 批处理时注意内存占用

**理论分析工具**：
1. 凸性用于优化理论
2. Lipschitz性用于收敛性分析
3. 不等式用于界的估计
4. 温度参数用于调节近似程度

这些性质使得logsumexp成为机器学习中不可或缺的基础运算。

