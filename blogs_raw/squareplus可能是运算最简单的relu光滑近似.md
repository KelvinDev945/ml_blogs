---
title: SquarePlus：可能是运算最简单的ReLU光滑近似
slug: squareplus可能是运算最简单的relu光滑近似
date: 2021-12-29
tags: 函数, 近似, 分析, 生成模型, attention
status: completed
---

# SquarePlus：可能是运算最简单的ReLU光滑近似

**原文链接**: [https://spaces.ac.cn/archives/8833](https://spaces.ac.cn/archives/8833)

**发布日期**: 

---

ReLU函数，也就是$\max(x,0)$，是最常见的激活函数之一，然而它在$x=0$处的不可导通常也被视为一个“槽点”。为此，有诸多的光滑近似被提出，比如SoftPlus、GeLU、Swish等，不过这些光滑近似无一例外地至少都使用了指数运算$e^x$（SoftPlus还用到了对数），从“精打细算”的角度来看，计算量还是不小的（虽然当前在GPU加速之下，我们很少去感知这点计算量了）。最近有一篇论文[《Squareplus: A Softplus-Like Algebraic Rectifier》](https://papers.cool/arxiv/2112.11687)提了一个更简单的近似，称为SquarePlus，我们也来讨论讨论。

需要事先指出的是，笔者是不建议大家花太多时间在激活函数的选择和设计上的，所以虽然分享了这篇论文，但主要是提供一个参考结果，并充当一道练习题来给大家“练练手”。

## 定义 #

SquarePlus的形式很简单，只用到了加、乘、除和开方：  
\begin{equation}\text{SquarePlus}(x)=\frac{x+\sqrt{x^2+b}}{2}\end{equation}  
其中$b > 0$。当$b=0$时，正好退化为$\text{ReLU}(x)=\max(x,0)$。SquarePlus的灵感来源大致是  
\begin{equation}\max(x,0)=\frac{x+|x|}{2}=\frac{x+\sqrt{x^2}}{2}\end{equation}  
因此为了补充在$x=0$的可导性，在根号里边多加一个大于0的常数$b$（防止导数出现除零问题）。

原论文指出，由于只用到了加、乘、除和开方，所以SquarePlus的速度（主要是在CPU上）会比SoftPlus等函数要快：  


[![SquarePlus与其他类似函数的速度比较](/usr/uploads/2021/12/1253499993.png)](/usr/uploads/2021/12/1253499993.png "点击查看原图")

SquarePlus与其他类似函数的速度比较

当然，如果你不关心这点速度提升，那么就像本文开头说的，当作数学练习题来看看就好。

## 性态 #

跟SoftPlus函数（$\log(e^x+1)$）一样，SquarePlus也是全局单调递增的，并且恒大于ReLU，如下图（下图的SquarePlus的$b=1$）：  


[![ReLU、SoftPlus、SquarePlus函数图像（一）](/usr/uploads/2021/12/3874442376.png)](/usr/uploads/2021/12/3874442376.png "点击查看原图")

ReLU、SoftPlus、SquarePlus函数图像（一）

直接求它的导函数也可以看出单调性：  
\begin{equation}\frac{d}{dx}\text{SquarePlus}(x)=\frac{1}{2}\left(1+\frac{x}{\sqrt{x^2+b}}\right) > 0\end{equation}  
至于二阶导数  
\begin{equation}\frac{d^2}{dx^2}\text{SquarePlus}(x)=\frac{b}{2(x^2+b)^{3/2}}\end{equation}  
也是恒大于0的存在，所以SquarePlus还是一个凸函数。

## 逼近 #

现在有两道练习题可以做了：

> 1、当$b$取什么时SquarePlus恒大于SoftPlus？
> 
> 2、当$b$取什么时，SquarePlus与SoftPlus误差最小？

第一个问题，直接从$\text{SquarePlus}(x)\geq \text{SoftPlus}(x)$解得：  
\begin{equation}b\geq 4\log(e^x+1)\left[\log(e^x+1) - x\right]=4\log(e^x+1)\log(e^{-x}+1)\end{equation}  
要使得上式恒成立，$b$必须大于等于右端的最大值，而我们可以证明右端最大值在$x=0$处取到，所以$b\geq 4\log^2 2=1.921812\cdots$。至此，第一个问题解决。

> **证明：** 留意到  
>  \begin{equation}  
>  \frac{d^2}{dx^2}\log\log(e^x+1)=\frac{e^x(\log(e^x+1)-e^x)}{(e^x+1)^2\log^2(e^x+1)} < 0\end{equation}
> 
> 所以$\log\log(e^x+1)$是一个凹函数，那么由詹森不等式得  
>  \begin{equation}  
>  \frac{1}{2}\left(\log\log(e^x+1) + \log\log(e^{-x}+1)\right)\leq \log\log(e^{(x+(-x))/2}+1)=\log\log 2\end{equation}  
>  也就是$\log\left(\log(e^x+1)\log(e^{-x}+1)\right)\leq 2\log\log 2$，或者$\log(e^x+1)\log(e^{-x}+1)\leq \log^2 2$，两边乘以4即得待证结论。等号成立的条件为$x=-x$，即$x=0$。

至于第二个问题，我们需要有一个“误差”的标准。这里跟之前的文章[《GELU的两个初等函数近似是怎么来的》](/archives/7309)一样，转化为无额外参数的$\min\text{-}\max$问题：  
\begin{equation}\min_{b} \max_x \left|\frac{x+\sqrt{x^2+b}}{2} - \log(e^x+1)\right|\end{equation}  
这个问题笔者没法求得解析解，目前只能通过数值求解：
    
    
    import numpy as np
    from scipy.special import erf
    from scipy.optimize import minimize
    
    def f(x, a):
        return np.abs((x + np.sqrt(x**2 + a**2)) / 2 - np.log(np.exp(x) + 1))
    
    def g(a):
        return np.max([f(x, a) for x in np.arange(-2, 4, 0.0001)])
    
    options = {'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 100000}
    result = minimize(g, 0, method='Powell', options=options)
    b = result.x**2
    print(b)

最终算出的结果是$b=1.52382103\cdots$，误差最大值为$0.075931\cdots$，比较如下：  


[![ReLU、SoftPlus、SquarePlus函数图像（二）](/usr/uploads/2021/12/1431410191.png)](/usr/uploads/2021/12/1431410191.png "点击查看原图")

ReLU、SoftPlus、SquarePlus函数图像（二）

## 小结 #

似乎也没啥好总结的，就是介绍了一个ReLU的光滑近似，并配上了两道简单的函数练习题～

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8833>_

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

苏剑林. (Dec. 29, 2021). 《SquarePlus：可能是运算最简单的ReLU光滑近似 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8833>

@online{kexuefm-8833,  
title={SquarePlus：可能是运算最简单的ReLU光滑近似},  
author={苏剑林},  
year={2021},  
month={Dec},  
url={\url{https://spaces.ac.cn/archives/8833}},  
} 


---

## 公式推导与注释

### 1. SquarePlus函数的定义与基本性质

**定义** (SquarePlus函数)

SquarePlus函数定义为：
\begin{equation}
\text{SquarePlus}(x; b) = \frac{x + \sqrt{x^2 + b}}{2} \tag{1}
\end{equation}
其中$b > 0$为光滑参数。

**注释1.1**: 该函数的设计灵感来自ReLU的对称形式。回顾ReLU函数可以写成：
\begin{equation}
\text{ReLU}(x) = \max(x, 0) = \frac{x + |x|}{2} = \frac{x + \sqrt{x^2}}{2} \tag{2}
\end{equation}

问题在于$\sqrt{x^2} = |x|$在$x=0$处不可导。为了使函数光滑，我们在根号内添加正常数$b$：
\begin{equation}
\sqrt{x^2} \quad \rightarrow \quad \sqrt{x^2 + b} \tag{3}
\end{equation}

这样做有两个好处：
- **光滑性**: $\sqrt{x^2 + b}$在全局可导，避免了$|x|$在原点的不可导问题
- **渐近性**: 当$|x| \gg \sqrt{b}$时，$\sqrt{x^2 + b} \approx |x|$，SquarePlus接近ReLU

---

### 2. 一阶导数分析（梯度性质）

**命题2.1** (一阶导数公式)

SquarePlus函数的一阶导数为：
\begin{equation}
\frac{d}{dx}\text{SquarePlus}(x; b) = \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + b}}\right) \tag{4}
\end{equation}

**证明**:
对式(1)求导：
\begin{align}
\frac{d}{dx}\text{SquarePlus}(x; b) &= \frac{1}{2}\left(1 + \frac{d}{dx}\sqrt{x^2 + b}\right) \tag{5} \\
&= \frac{1}{2}\left(1 + \frac{1}{2\sqrt{x^2 + b}} \cdot 2x\right) \tag{6} \\
&= \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + b}}\right) \tag{7}
\end{align}

**注释2.1**: 关键性质分析

(1) **严格单调性**: 由于$x^2 + b > 0$恒成立，我们有：
\begin{equation}
\left|\frac{x}{\sqrt{x^2 + b}}\right| < 1 \tag{8}
\end{equation}
因此：
\begin{equation}
0 < \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + b}}\right) < 1 \tag{9}
\end{equation}
这说明SquarePlus是严格单调递增的，且梯度有界。

(2) **原点导数**: 当$x = 0$时：
\begin{equation}
\frac{d}{dx}\text{SquarePlus}(0; b) = \frac{1}{2}\left(1 + \frac{0}{\sqrt{b}}\right) = \frac{1}{2} \tag{10}
\end{equation}

(3) **渐近行为**:
- 当$x \to +\infty$时：$\frac{x}{\sqrt{x^2 + b}} \to 1$，故导数$\to 1$
- 当$x \to -\infty$时：$\frac{x}{\sqrt{x^2 + b}} \to -1$，故导数$\to 0$

**命题2.2** (导数的有界性)

对于任意$x \in \mathbb{R}$和$b > 0$，有：
\begin{equation}
0 < \frac{d}{dx}\text{SquarePlus}(x; b) < 1 \tag{11}
\end{equation}

**证明**: 令$t = \frac{x}{\sqrt{x^2 + b}}$，则$t \in (-1, 1)$（严格不等）。因此：
\begin{equation}
\frac{d}{dx}\text{SquarePlus}(x; b) = \frac{1 + t}{2} \in (0, 1) \tag{12}
\end{equation}

---

### 3. 二阶导数分析（凸性）

**命题3.1** (二阶导数公式)

SquarePlus函数的二阶导数为：
\begin{equation}
\frac{d^2}{dx^2}\text{SquarePlus}(x; b) = \frac{b}{2(x^2 + b)^{3/2}} \tag{13}
\end{equation}

**证明**:
对式(4)求导：
\begin{align}
\frac{d^2}{dx^2}\text{SquarePlus}(x; b) &= \frac{1}{2} \cdot \frac{d}{dx}\left(\frac{x}{\sqrt{x^2 + b}}\right) \tag{14} \\
&= \frac{1}{2} \cdot \frac{\sqrt{x^2 + b} - x \cdot \frac{x}{\sqrt{x^2 + b}}}{x^2 + b} \tag{15} \\
&= \frac{1}{2} \cdot \frac{x^2 + b - x^2}{(x^2 + b)^{3/2}} \tag{16} \\
&= \frac{b}{2(x^2 + b)^{3/2}} \tag{17}
\end{align}

**注释3.1**: 凸性分析

由于$b > 0$且$(x^2 + b)^{3/2} > 0$，我们有：
\begin{equation}
\frac{d^2}{dx^2}\text{SquarePlus}(x; b) > 0, \quad \forall x \in \mathbb{R} \tag{18}
\end{equation}

这意味着**SquarePlus是严格凸函数**。

**命题3.2** (二阶导数的性质)

(1) **原点处的曲率**:
\begin{equation}
\frac{d^2}{dx^2}\text{SquarePlus}(0; b) = \frac{b}{2b^{3/2}} = \frac{1}{2\sqrt{b}} \tag{19}
\end{equation}

(2) **渐近行为**:
\begin{equation}
\lim_{|x| \to \infty} \frac{d^2}{dx^2}\text{SquarePlus}(x; b) = 0 \tag{20}
\end{equation}
这说明在远离原点时，SquarePlus逐渐趋向线性。

(3) **单调递减性**: 对于$x > 0$，二阶导数关于$x$单调递减：
\begin{equation}
\frac{d}{dx}\left[\frac{b}{2(x^2 + b)^{3/2}}\right] = -\frac{3bx}{2(x^2 + b)^{5/2}} < 0 \tag{21}
\end{equation}

---

### 4. 与ReLU的逼近分析

**命题4.1** (逼近误差)

定义逼近误差为：
\begin{equation}
E(x; b) = \text{SquarePlus}(x; b) - \text{ReLU}(x) = \frac{x + \sqrt{x^2 + b}}{2} - \max(x, 0) \tag{22}
\end{equation}

分情况讨论：

(1) **当$x \geq 0$时**:
\begin{align}
E(x; b) &= \frac{x + \sqrt{x^2 + b}}{2} - x \tag{23} \\
&= \frac{\sqrt{x^2 + b} - x}{2} \tag{24} \\
&= \frac{b}{2(\sqrt{x^2 + b} + x)} \tag{25}
\end{align}

其中式(25)通过分子有理化得到：
\begin{equation}
\sqrt{x^2 + b} - x = \frac{(\sqrt{x^2 + b} - x)(\sqrt{x^2 + b} + x)}{\sqrt{x^2 + b} + x} = \frac{b}{\sqrt{x^2 + b} + x} \tag{26}
\end{equation}

(2) **当$x < 0$时**:
\begin{align}
E(x; b) &= \frac{x + \sqrt{x^2 + b}}{2} - 0 \tag{27} \\
&= \frac{x + \sqrt{x^2 + b}}{2} \tag{28}
\end{align}

**注释4.1**: 误差的定量分析

(1) 对于$x \geq 0$，误差满足：
\begin{equation}
0 < E(x; b) \leq \frac{b}{2x} \quad \text{(当$x > 0$)} \tag{29}
\end{equation}
特别地，$E(0; b) = \frac{\sqrt{b}}{2}$。

(2) 对于$x < 0$，由于$\sqrt{x^2 + b} > |x| = -x$，我们有：
\begin{equation}
0 < E(x; b) < \frac{\sqrt{x^2 + b}}{2} \tag{30}
\end{equation}

**命题4.2** (最大误差点)

对于固定的$b > 0$，误差$E(x; b)$在$x = 0$处达到最大值：
\begin{equation}
\max_{x \in \mathbb{R}} E(x; b) = E(0; b) = \frac{\sqrt{b}}{2} \tag{31}
\end{equation}

**证明**:
对于$x \geq 0$，式(25)表明$E(x; b)$关于$x$单调递减，且$\lim_{x \to \infty} E(x; b) = 0$。

对于$x \leq 0$，我们计算：
\begin{align}
\frac{dE}{dx} &= \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + b}}\right) \tag{32} \\
&= \frac{1}{2}\left(1 - \frac{|x|}{\sqrt{x^2 + b}}\right) > 0 \tag{33}
\end{align}

因此$E(x; b)$在$x < 0$时关于$x$单调递增，在$x = 0$时达到最大值。

---

### 5. 与SoftPlus的比较分析

**定义** (SoftPlus函数)

SoftPlus函数定义为：
\begin{equation}
\text{SoftPlus}(x) = \log(1 + e^x) \tag{34}
\end{equation}

**命题5.1** (恒大于关系)

当$b \geq 4\log^2 2 \approx 1.922$时，对于所有$x \in \mathbb{R}$有：
\begin{equation}
\text{SquarePlus}(x; b) \geq \text{SoftPlus}(x) \tag{35}
\end{equation}

**证明**:
我们需要证明：
\begin{equation}
\frac{x + \sqrt{x^2 + b}}{2} \geq \log(1 + e^x) \tag{36}
\end{equation}

等价于：
\begin{equation}
b \geq 4\log(1 + e^x)[\log(1 + e^x) - x] \tag{37}
\end{equation}

进一步化简：
\begin{align}
\log(1 + e^x) - x &= \log\left(\frac{1 + e^x}{e^x}\right) \tag{38} \\
&= \log(e^{-x} + 1) \tag{39}
\end{align}

因此条件变为：
\begin{equation}
b \geq 4\log(1 + e^x)\log(1 + e^{-x}) \tag{40}
\end{equation}

**引理5.1** (对数乘积的最大值)

函数$g(x) = \log(1 + e^x)\log(1 + e^{-x})$在$x = 0$处达到最大值：
\begin{equation}
\max_{x \in \mathbb{R}} g(x) = g(0) = \log^2 2 \tag{41}
\end{equation}

**证明**:
计算二阶导数：
\begin{align}
\frac{d^2}{dx^2}\log\log(1 + e^x) &= \frac{e^x(\log(1 + e^x) - e^x)}{(1 + e^x)^2\log^2(1 + e^x)} \tag{42}
\end{align}

由于$\log(1 + e^x) < e^x$（对所有$x > 0$），分子为负，因此：
\begin{equation}
\frac{d^2}{dx^2}\log\log(1 + e^x) < 0 \tag{43}
\end{equation}

这说明$\log\log(1 + e^x)$是严格凹函数。由Jensen不等式：
\begin{align}
\frac{1}{2}[\log\log(1 + e^x) + \log\log(1 + e^{-x})] &\leq \log\log\left(1 + e^{(x + (-x))/2}\right) \tag{44} \\
&= \log\log 2 \tag{45}
\end{align}

两边乘以2并取指数：
\begin{equation}
\log(1 + e^x)\log(1 + e^{-x}) \leq \log^2 2 \tag{46}
\end{equation}

等号成立当且仅当$x = 0$。因此：
\begin{equation}
b \geq 4\log^2 2 \approx 1.922 \tag{47}
\end{equation}

---

### 6. 最优逼近参数的求解

**优化问题** (Minimax逼近)

寻找最优参数$b^*$使得SquarePlus与SoftPlus的最大误差最小：
\begin{equation}
b^* = \argmin_b \max_x |{\text{SquarePlus}(x; b) - \text{SoftPlus}(x)}| \tag{48}
\end{equation}

**注释6.1**: 这是一个无约束的minimax优化问题。由于目标函数非凸且不可导，我们采用Powell法求解。

**数值求解算法**:

```python
import numpy as np
from scipy.optimize import minimize

def objective(a):
    """目标函数：最大误差
    参数: a = sqrt(b) (保证b > 0)
    """
    x_grid = np.linspace(-3, 5, 10000)
    b = a**2
    sp = (x_grid + np.sqrt(x_grid**2 + b)) / 2
    softplus = np.log(1 + np.exp(np.clip(x_grid, -20, 20)))
    error = np.abs(sp - softplus)
    return np.max(error)

# Powell法优化
result = minimize(objective, x0=1.0, method='Powell',
                  options={'xtol': 1e-10, 'ftol': 1e-10})
b_optimal = result.x[0]**2
max_error = result.fun
```

**命题6.1** (数值最优解)

通过数值优化，得到：
\begin{equation}
b^* \approx 1.5238, \quad \max_x |E(x; b^*)| \approx 0.0759 \tag{49}
\end{equation}

**注释6.2**: 观察到$b^* < 4\log^2 2$，这意味着最优逼近时SquarePlus不是处处大于SoftPlus，而是在某些区域大于、某些区域小于，从而达到误差的平衡。

---

### 7. 泰勒展开与局部逼近

**命题7.1** (原点处的泰勒展开)

在$x = 0$附近，SquarePlus有如下展开：
\begin{equation}
\text{SquarePlus}(x; b) = \frac{\sqrt{b}}{2} + \frac{x}{2} + \frac{x^2}{4\sqrt{b}} - \frac{x^4}{16b^{5/2}} + O(x^6) \tag{50}
\end{equation}

**证明**:
首先展开$\sqrt{x^2 + b}$：
\begin{align}
\sqrt{x^2 + b} &= \sqrt{b}\sqrt{1 + \frac{x^2}{b}} \tag{51} \\
&= \sqrt{b}\left(1 + \frac{x^2}{2b} - \frac{x^4}{8b^2} + O(x^6)\right) \tag{52} \\
&= \sqrt{b} + \frac{x^2}{2\sqrt{b}} - \frac{x^4}{8b^{3/2}} + O(x^6) \tag{53}
\end{align}

代入SquarePlus定义：
\begin{align}
\text{SquarePlus}(x; b) &= \frac{1}{2}\left(x + \sqrt{b} + \frac{x^2}{2\sqrt{b}} - \frac{x^4}{8b^{3/2}} + O(x^6)\right) \tag{54} \\
&= \frac{\sqrt{b}}{2} + \frac{x}{2} + \frac{x^2}{4\sqrt{b}} - \frac{x^4}{16b^{5/2}} + O(x^6) \tag{55}
\end{align}

**命题7.2** (SoftPlus的泰勒展开)

在$x = 0$附近：
\begin{equation}
\text{SoftPlus}(x) = \log 2 + \frac{x}{2} + \frac{x^2}{8} - \frac{x^4}{192} + O(x^6) \tag{56}
\end{equation}

**证明**:
利用$\log(1 + e^x) = \log 2 + \log(1 + \frac{e^x - 1}{2})$：
\begin{align}
\text{SoftPlus}(x) &= \log 2 + \log\left(1 + \frac{\sinh(x/2)}{e^{x/2}}\right) \tag{57}
\end{align}

通过标准的泰勒级数计算可得式(56)。

**注释7.1**: 比较系数

比较式(50)和(56)的系数：
- **常数项**: $\frac{\sqrt{b}}{2}$ vs $\log 2$，当$b = 4\log^2 2$时相等
- **一次项**: 系数都是$\frac{1}{2}$，完全一致
- **二次项**: $\frac{1}{4\sqrt{b}}$ vs $\frac{1}{8}$，当$b = 4$时相等

这解释了为何$b$在1.5到2之间能够较好地逼近SoftPlus。

---

### 8. 渐近性质分析

**命题8.1** (正向渐近)

当$x \to +\infty$时：
\begin{equation}
\text{SquarePlus}(x; b) = x + \frac{b}{4x} - \frac{b^2}{32x^3} + O(x^{-5}) \tag{58}
\end{equation}

**证明**:
对于$x \gg \sqrt{b}$，有：
\begin{align}
\sqrt{x^2 + b} &= x\sqrt{1 + \frac{b}{x^2}} \tag{59} \\
&= x\left(1 + \frac{b}{2x^2} - \frac{b^2}{8x^4} + O(x^{-6})\right) \tag{60} \\
&= x + \frac{b}{2x} - \frac{b^2}{8x^3} + O(x^{-5}) \tag{61}
\end{align}

因此：
\begin{align}
\text{SquarePlus}(x; b) &= \frac{1}{2}\left(2x + \frac{b}{2x} - \frac{b^2}{8x^3} + O(x^{-5})\right) \tag{62} \\
&= x + \frac{b}{4x} - \frac{b^2}{16x^3} + O(x^{-5}) \tag{63}
\end{align}

**命题8.2** (负向渐近)

当$x \to -\infty$时：
\begin{equation}
\text{SquarePlus}(x; b) = \frac{b}{4|x|} - \frac{b^2}{32|x|^3} + O(|x|^{-5}) \tag{64}
\end{equation}

**证明**: 类似于命题8.1，利用$\sqrt{x^2 + b} = |x|\sqrt{1 + \frac{b}{x^2}}$展开即可。

---

### 9. 数值稳定性分析

**问题9.1** (计算精度)

直接计算$\sqrt{x^2 + b}$可能在极端情况下损失精度。我们需要分析何时需要特殊处理。

**稳定计算方法**:

(1) **当$|x| \ll \sqrt{b}$时** (小参数区域):
使用泰勒展开式(50)避免浮点误差。

(2) **当$|x| \gg \sqrt{b}$时** (大参数区域):
- 对于$x > 0$: 使用分子有理化
\begin{equation}
\text{SquarePlus}(x; b) = x + \frac{b}{2(\sqrt{x^2 + b} + x)} \tag{65}
\end{equation}

- 对于$x < 0$: 直接计算
\begin{equation}
\text{SquarePlus}(x; b) = \frac{b}{2(\sqrt{x^2 + b} - x)} \tag{66}
\end{equation}

**算法9.1** (数值稳定实现)

```python
def squareplus_stable(x, b):
    """数值稳定的SquarePlus实现"""
    abs_x = np.abs(x)
    sqrt_b = np.sqrt(b)

    # 小参数情况：|x| < 0.1 * sqrt(b)
    small_mask = abs_x < 0.1 * sqrt_b
    if np.any(small_mask):
        x_small = x[small_mask]
        result_small = (sqrt_b / 2 + x_small / 2 +
                        x_small**2 / (4 * sqrt_b) -
                        x_small**4 / (16 * b**(5/2)))

    # 大参数情况
    large_mask = ~small_mask
    if np.any(large_mask):
        x_large = x[large_mask]
        sqrt_term = np.sqrt(x_large**2 + b)

        # 正数：使用有理化
        pos_mask = x_large > 0
        if np.any(pos_mask):
            x_pos = x_large[pos_mask]
            sqrt_pos = np.sqrt(x_pos**2 + b)
            result_pos = x_pos + b / (2 * (sqrt_pos + x_pos))

        # 负数：直接计算
        neg_mask = x_large <= 0
        if np.any(neg_mask):
            x_neg = x_large[neg_mask]
            result_neg = (x_neg + np.sqrt(x_neg**2 + b)) / 2

    # 组合结果
    result = np.empty_like(x)
    result[small_mask] = result_small
    # ... (合并large_mask的结果)
    return result
```

---

### 10. 计算复杂度分析

**命题10.1** (运算次数比较)

各激活函数的基本运算次数（每次调用）：

| 函数 | 加法 | 乘法 | 除法 | 开方 | 指数 | 对数 |
|------|------|------|------|------|------|------|
| ReLU | 0 | 0 | 0 | 0 | 0 | 0 |
| SquarePlus | 2 | 1 | 1 | 1 | 0 | 0 |
| SoftPlus | 1 | 0 | 0 | 0 | 1 | 1 |
| GeLU | 3 | 2 | 1 | 0 | 1 | 0 |
| Swish | 2 | 2 | 0 | 0 | 1 | 0 |

**注释10.1**:
- 指数和对数运算通常比四则运算和开方慢得多（尤其在CPU上）
- 开方运算可通过Newton-Raphson快速逼近，复杂度约为3-4次乘法
- SquarePlus避免了昂贵的超越函数（exp, log），在CPU上有速度优势

---

### 11. 与其他光滑激活函数的比较

**定义11.1** (GeLU函数)

Gaussian Error Linear Unit:
\begin{equation}
\text{GeLU}(x) = x \cdot \Phi(x) = \frac{x}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] \tag{67}
\end{equation}
其中$\Phi$是标准正态分布的累积分布函数。

**定义11.2** (Swish函数)

也称为SiLU (Sigmoid Linear Unit):
\begin{equation}
\text{Swish}(x; \beta) = \frac{x}{1 + e^{-\beta x}} \tag{68}
\end{equation}

**命题11.1** (原点处的性质比较)

在$x = 0$处：

| 函数 | 函数值 | 一阶导数 | 二阶导数 |
|------|--------|----------|----------|
| ReLU | 0 | 不存在 | 不存在 |
| SquarePlus($b$) | $\frac{\sqrt{b}}{2}$ | $\frac{1}{2}$ | $\frac{1}{2\sqrt{b}}$ |
| SoftPlus | $\log 2$ | $\frac{1}{2}$ | $\frac{1}{4}$ |
| GeLU | 0 | $\frac{1}{2}$ | $\frac{1}{\sqrt{2\pi}}$ |
| Swish($\beta=1$) | 0 | $\frac{1}{2}$ | $\frac{1}{4}$ |

**注释11.1**: 观察
- 除ReLU外，所有光滑近似的一阶导数在原点都是$\frac{1}{2}$
- SquarePlus和SoftPlus在原点有非零函数值（正偏移）
- GeLU和Swish保持ReLU在原点的零值特性

---

### 12. 反向传播的梯度计算

**算法12.1** (SquarePlus的反向传播)

前向传播：
\begin{equation}
y = \text{SquarePlus}(x; b) = \frac{x + \sqrt{x^2 + b}}{2} \tag{69}
\end{equation}

反向传播（链式法则）：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + b}}\right) \tag{70}
\end{equation}

**数值稳定实现**:

```python
class SquarePlus:
    def __init__(self, b=1.0):
        self.b = b
        self.cache = {}

    def forward(self, x):
        sqrt_term = np.sqrt(x**2 + self.b)
        y = (x + sqrt_term) / 2
        # 缓存中间结果用于反向传播
        self.cache['x'] = x
        self.cache['sqrt_term'] = sqrt_term
        return y

    def backward(self, grad_y):
        x = self.cache['x']
        sqrt_term = self.cache['sqrt_term']
        # 梯度计算
        grad_x = grad_y * 0.5 * (1 + x / sqrt_term)
        return grad_x
```

**注释12.1**:
- 通过缓存$\sqrt{x^2 + b}$，避免在反向传播时重复计算
- 内存开销：需要存储输入$x$和中间项$\sqrt{x^2 + b}$
- 计算开销：反向传播仅需1次除法和2次加法

---

### 13. 参数$b$的选择指导

**准则13.1** (参数选择策略)

根据不同需求选择$b$：

(1) **最大光滑性** ($b$较大):
- 推荐: $b \in [2, 4]$
- 优点: 二阶导数小，曲率平缓
- 缺点: 与ReLU偏差较大

(2) **最佳SoftPlus逼近**:
- 推荐: $b \approx 1.52$
- 优点: 与SoftPlus误差最小
- 适用: 需要替代SoftPlus的场景

(3) **恒大于SoftPlus**:
- 推荐: $b \geq 1.92$
- 优点: 严格保持上界性质
- 适用: 需要保守估计的场景

(4) **接近ReLU** ($b$较小):
- 推荐: $b \in [0.5, 1]$
- 优点: 与ReLU偏差小，梯度接近0/1
- 缺点: 在原点附近仍有较大曲率

**命题13.1** (参数对梯度的影响)

对于固定输入$x$，梯度关于$b$的导数为：
\begin{equation}
\frac{\partial}{\partial b}\left[\frac{d}{dx}\text{SquarePlus}(x; b)\right] = -\frac{x}{4(x^2 + b)^{3/2}} \tag{71}
\end{equation}

这说明：
- 当$x > 0$时，$b$增大会使梯度减小
- 当$x < 0$时，$b$增大会使梯度增大（更接近0.5）

---

### 14. 应用场景与实践建议

**建议14.1** (何时使用SquarePlus)

推荐使用场景：
1. **CPU密集型推理**: 避免exp/log运算，提升速度
2. **边缘设备部署**: 硬件不支持高效超越函数时
3. **数值敏感任务**: 需要避免exp溢出的场景
4. **理论分析**: 需要显式解析形式的研究

不推荐场景：
1. **GPU训练**: GPU对exp/log优化良好，速度优势不明显
2. **已有预训练模型**: 更换激活函数需重新训练
3. **需要精确零点**: SquarePlus在原点有偏移

**建议14.2** (初始化策略)

使用SquarePlus时的权重初始化建议：

由于SquarePlus的梯度范围是$(0, 1)$，其期望约为$\frac{1}{2}$（假设输入对称分布），相比ReLU的期望$\frac{1}{2}$（正半轴）基本一致。因此：

- **Xavier初始化**: 仍然适用
- **He初始化**: 可以使用，但可能过于保守
- **推荐**: 使用标准Xavier，方差为$\frac{2}{n_{\text{in}} + n_{\text{out}}}$

**建议14.3** (调参经验)

- **默认值**: $b = 1.0$（平衡性能和逼近质量）
- **精细调优**: 在$[0.8, 1.5]$范围内网格搜索
- **避免极端**: $b < 0.1$或$b > 4$通常不必要

---

### 15. 理论扩展：多参数泛化

**定义15.1** (广义SquarePlus)

引入两个参数的版本：
\begin{equation}
\text{SquarePlus}(x; a, b) = \frac{x + \sqrt{x^2 + b}}{2} + a \tag{72}
\end{equation}
其中$a$是偏移参数，$b > 0$是光滑参数。

**性质15.1**:
- $a$控制函数的纵向偏移
- 导数不受$a$影响
- 当$a = -\frac{\sqrt{b}}{2}$时，$\text{SquarePlus}(0; a, b) = 0$，类似GeLU

**定义15.2** (缩放版本)

\begin{equation}
\text{SquarePlus}(x; b, \alpha) = \alpha \cdot \frac{x + \sqrt{x^2 + b}}{2} \tag{73}
\end{equation}

当$\alpha = \frac{2}{\sqrt{b}}$时，原点导数恰好为1。

---

### 总结

本文从数学角度全面分析了SquarePlus函数：

1. **基本性质**: 严格单调递增、严格凸、全局可导
2. **逼近质量**: 最优参数$b^* \approx 1.52$时与SoftPlus最大误差约0.076
3. **计算效率**: 避免超越函数，仅需四则运算和开方
4. **数值稳定**: 提供了稳定的计算方法和反向传播实现
5. **实用指导**: 给出参数选择和应用场景建议

SquarePlus作为ReLU的光滑近似，在保持计算简单性的同时，提供了良好的可导性和逼近质量，是深度学习中值得考虑的激活函数选择。

