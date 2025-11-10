---
title: 圆内随机n点在同一个圆心角为θ的扇形的概率
slug: 圆内随机n点在同一个圆心角为θ的扇形的概率
date: 2022-10-25
tags: 概率, 竞赛, 随机, 生成模型, attention
status: pending
---

# 圆内随机n点在同一个圆心角为θ的扇形的概率

**原文链接**: [https://spaces.ac.cn/archives/9324](https://spaces.ac.cn/archives/9324)

**发布日期**: 

---

这几天网上热传了一道“四鸭共半圆”题目：  


[![四鸭共半圆问题](/usr/uploads/2022/10/3296253276.png)](/usr/uploads/2022/10/3296253276.png "点击查看原图")

四鸭共半圆问题

可能有不少读者看到后也尝试做过，就连李永乐老师也专门开了一节课讲这道题（参考[《圆形水池四只鸭子在同一个半圆里，概率有多大？》](https://www.bilibili.com/read/cv19251490)）。就这道题目本身而言，答案并不算困难，可以有很多方法算出来。稍微有难度的是它的推广版本，也就是本文标题所描述的，将鸭子的数目一般化为$n$只，将半圆一般化为圆心角为$\theta$的扇形。更有趣的是，当$\theta \leq \pi$时，依然有比较初等的解法，但是当$\theta > \pi$后，复杂度开始“剧增”...

## 题目转换 #

首先要说明的是，这里我们是将鸭子抽象为圆内均匀分布随机抽取的点来处理的，那些诸如“鸭子的占地面积”等推广这里我们就不考虑了。就“四鸭共半圆”而言，我们很容易去将它一般化，比如推广到高维空间中：

> $d$维超球内均匀分布的$n$个点，它们位于同一个$d$维超半球的概率是多少？

这个推广的答案其实在1962年发表的论文[《A Problem in Geometric Probability》](https://www.mscand.dk/article/view/10655/8676)就给出了。另一个推广就是本文的标题：

> 圆内均匀分布的$n$个点，它们位于同一个圆心角为$\theta$的扇形的概率是多少？

本文主要就是讨论这个问题，它还可以等价地转换成（怎么转换请读者自行思考）：

> 圆周上均匀随机地选取$n$个点，将圆周分为$n$段圆弧，最长的圆弧所对的圆心角大于$2\pi - \theta$的概率是多少？

进一步地，可以等价地转换为

> 单位长线段上均匀随机地选取$n-1$个点，将线段分为$n$段，最长的线段长度大于$1 - \frac{\theta}{2\pi}$的概率是多少？

## 分布求解 #

其实，最后一个等价表述所涉及的分布，已经被很好地讨论过了，比如知乎上的[《在 (0, 1) 内随机取点将区间分成 n 段，最长段的长度期望是多少？》](https://www.zhihu.com/question/50198685/answer/1509793840)。这里为了方便大家理解，笔者重新组织语言介绍一遍，将会分为几个步骤，整个过程有点长，但这是求通解（适用于$\theta > \pi$）必须的。

顺便指出的是，多个随机变量的最值，属于“[顺序统计量（Order Statistic）](https://en.wikipedia.org/wiki/Order_statistic)”之一，在机器学习中也颇为常见，比如[《EAE：自编码器 + BN + 最大熵 = 生成模型》](/archives/7343#%E7%86%B5%E7%9A%84%E9%87%87%E6%A0%B7%E9%82%BB%E8%BF%91%E4%BC%B0%E8%AE%A1)介绍的熵的最邻近估计、[《从重参数的角度看离散概率分布的构建》](/archives/9085)介绍的离散分布的一般构建，都可归结为此类。

### 联合分布 #

从最后一个等价表述出发，设单位长线段被随机的$n-1$个点分为的$n$段，从左到右的长度依次为$x_1,x_2,\cdots,x_n$（注：只是定了个方向，并没有按大小排列），记它们的联合分布的概率密度函数为$p_n(x_1,x_2,\cdots,x_n)$。知乎链接中的答案是直接基于“$p_n(x_1,x_2,\cdots,x_n)$是均匀分布”这一事实来进行后续求解的，但不知道是不是笔者漏了什么细节，还是哪里没绕过弯，笔者觉得“$p_n(x_1,x_2,\cdots,x_n)$是均匀分布”并不是一件显然成立的事情，所以这里对它进行推导一遍。

首先，留意到两个事实：

> 1、$x_1,x_2,\cdots,x_n$带有约束$x_1 + x_2 + \cdots + x_n = 1$，所以它只有$n-1$个自由度，我们取前$n-1$个变量为自由变量；
> 
> 2、$p_n(x_1,x_2,\cdots,x_n)$是概率密度而非概率，$p_n(x_1,x_2,\cdots,x_n)dx_1 dx_2 \cdots dx_{n-1}$才是概率。

为了理解“$p_n(x_1,x_2,\cdots,x_n)$是均匀分布”这一事实，我们从$n=2$出发，此时就是在$(0, 1)$中均匀随机取一个点将线段分为两部分，由于取点是均匀分布的（概率密度是1），取点跟$(x_1, x_2)$一一对应，所以也有$p_2(x_1, x_2)=1$。

接着考虑$n=3$，对应的概率是$p_3(x_1, x_2, x_3)dx_1 dx_2$，它有两种可能：

> 1、先采样一个点，将线段分为$(x_1, x_2 + x_3)$两段，这部分概率为$p_2(x_1, x_2 + x_3) dx_1$，然后再采样一个点，将$x_2 + x_3$这一段分为$(x_2, x_3)$两段，这部分概率为$dx_2$，所以乘起来是$p_2(x_1, x_2 + x_3) dx_1 dx_2$；
> 
> 2、先采样一个点，将线段分为$(x_1 + x_2, x_3)$两段，这部分概率为$p_2(x_1 + x_2, x_3) dx_2$，然后再采样一个点，将$x_1 + x_2$这一段分为$(x_1, x_2)$两段，这部分概率为$dx_1$，所以乘起来是$p_2(x_1 + x_2, x_3) dx_1 dx_2$；

两者加起来是  
\begin{equation}p_3(x_1, x_2, x_3)dx_1 dx_2 = p_2(x_1, x_2 + x_3) dx_1 dx_2 + p_2(x_1 + x_2, x_3) dx_1 dx_2 = 2dx_1 dx_2\end{equation}  
即$p_3(x_1, x_2, x_3)=2$也是一个均匀分布。类似地递推，可以得到  
\begin{equation}p_n(x_1,x_2,\cdots,x_n) = (n - 1) p_{n-1}(x_1,x_2,\cdots,x_{n-1}) = \cdots = (n - 1)!\end{equation}

### 边缘分布 #

有了联合分布，那么我们就可以求出任选$k$个变量的边缘分布了，这是为下一节使用“容斥原理”作准备的。由于联合分布就是一个均匀分布，所以各个变量是全对称的，因此不失一般性，我们求前$k$个变量的边缘分布即可。

根据定义  
\begin{equation}p_n(x_1,x_2,\cdots,x_k) = \int \cdots \int p_n(x_1,x_2,\cdots,x_n) dx_{k+1}\cdots dx_{n-1}\end{equation}  
要注意的是积分上下限，下限自然是$0$，因为约束$x_1 + x_2 + \cdots + x_n = 1$的存在，上限对于每个变量都是不一样的，给定$x_1,x_2,\cdots,x_i$后，$x_{i+1}$的取值就只能是$0\sim 1 - (x_1 + x_2 + \cdots + x_i)$，所以准确形式是  
\begin{equation}\int_0^{1 - (x_1 + x_2 + \cdots + x_k)} \cdots \left[\int_0^{1 - (x_1 + x_2 + \cdots + x_{n-2})} p_n(x_1,x_2,\cdots,x_n) dx_{n-1}\right] \cdots dx_{k+1}\end{equation}  
由于$p_n(x_1,x_2,\cdots,x_n)=(n-1)!$只是一个常数，所以上式是可以逐次积分出来的，最终结果是  
\begin{equation}p_n(x_1,x_2,\cdots,x_k) = \frac{(n-1)!}{(n - k - 1)!}[1 - (x_1 + x_2 + \cdots + x_k)]^{n-k-1}\end{equation}  
这时候我们就可以求出$x_1,x_2,\cdots,x_k$分别大于给定阈值$c_1, c_2, \cdots, c_k$（成立$c_1 + c_2 + \cdots + c_k \leq 1$）的概率：  
\begin{equation}\begin{aligned}  
&\, \qquad P_n(x_1 > c_1,x_2 > c_2,\cdots,x_k > c_k) = \int_{c_1}^{1 - (c_2 + c_2 + \cdots + c_k)} \cdots  
\\\  
&\, \left[\int_{c_{k-1}}^{1 - (x_1 + x_2 + \cdots + x_{k-2} + c_k)}\left[\int_{c_k}^{1 - (x_1 + x_2 + \cdots + x_{k-1})} p_n(x_1,x_2,\cdots,x_k) dx_k\right]dx_{k-1}\right] \cdots dx_1  
\end{aligned}\end{equation}  
积分上限跟前面类似，都是在约束$x_1 + x_2 + \cdots + x_n = 1$下进行推导的。跟$p_n(x_1,x_2,\cdots,x_n)$一样，上式也是可以逐次积分出来的，最终结果反而很简单  
\begin{equation}P_n(x_1 > c_1,x_2 > c_2,\cdots,x_k > c_k) = [1 - (c_1 + c_2 + \cdots + c_k)]^{n-1}\end{equation}

### 容斥原理 #

现在一切准备就绪，轮到“容斥原理”来完成“最后一击”了。别忘了我们的目标是求出最长段长度的概率，即对于某个阈值$x$，计算出$P_n(\max(x_1,x_2,\cdots,x_n) > x)$，而$\max(x_1,x_2,\cdots,x_n) > x$这件事本身是多个事件的并集：  
\begin{equation}\max(x_1,x_2,\cdots,x_n) > x \quad\Leftrightarrow\quad x_1 > x \,\color{red}{\text{或}}\, x_2 > x \,\color{red}{\text{或}}\, \cdots \,\color{red}{\text{或}}\, x_n > x \end{equation}  
关键是，当$x < \frac{1}{2}$时，各个事件之间并非不相交的，所以不能直接将每一部分的概率简单相加，而需要用到“[容斥原理](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle)”：  
\begin{equation}\begin{aligned}  
\text{存在一点大于}x\text{的概率} = &\,\text{任一点大于}x\text{的概率之和} \\\  
&\,\quad \color{red}{-} \text{任两点都大于}x\text{的概率之和} \\\  
&\, \quad\quad \color{red}{+} \text{任三点都大于}x\text{的概率之和} \\\  
&\, \quad\quad\quad \color{red}{-} \text{任四点都大于}x\text{的概率之和} \\\  
&\, \quad\quad\quad\quad \color{red}{+} \cdots  
\end{aligned}\end{equation}  
根据上一节的结果，任选$k$点，每一点都大于$x$的概率为$(1-kx)^{n-1}$，而任选$k$点的组合数为$C_n^k$，所以代入上式得  
\begin{equation}\begin{aligned}  
P_n(\max(x_1,x_2,\cdots,x_n) > x) =&\, \sum_{k=1, 1 - kx > 0}^n (-1)^{k-1} C_n^k (1-kx)^{n-1} \\\  
=&\, C_n^1 (1 - x)^{n-1} - C_n^2 (1 - 2x)^{n-1} + \cdots  
\end{aligned}\end{equation}

## 答案分析 #

对于本文开始的题目来说，即$x = 1 - \frac{\theta}{2\pi}$，当$x > \frac{1}{2}$（即$\theta < \pi$）时，后面的$1 - k x$都小于0了，即实际上不存在，因此答案是最简单的：  
\begin{equation}C_n^1 (1 - x)^{n-1} = n \left(\frac{\theta}{2\pi}\right)^{n-1}\end{equation}  
当$x < \frac{1}{2}$时，看$x$的具体大小来增减项，$x$越小，项数相对来说越多，这就是李永乐老师文章说的“当$\theta$大于180度时，情况将变得非常复杂”了。

我们也可以求它的期望，这知乎上的提问所讨论的问题。还有一个有意思的情况，显然当$x < \frac{1}{n}$时，恒有$1 - kx > 0$，即所有项都需要用上了：  
\begin{equation}P_n(\max(x_1,x_2,\cdots,x_n) > x) = \sum_{k=1}^n (-1)^{k-1} C_n^k (1-kx)^{n-1}\end{equation}  
然而根据“抽屉原理”我们可以得知，将单位长线段分为$n$段后，最长的一段的长度必然是不小于$\frac{1}{n}$的，这意味着$P_n(\max(x_1,x_2,\cdots,x_n) > \frac{1}{n}) = 1$，因此当$x < \frac{1}{n}$时，必然成立  
\begin{equation}\sum_{k=1}^n (-1)^{k-1} C_n^k (1-kx)^{n-1} = 1\end{equation}  
又因为左端仅仅是简单的代数多项式，因此它必然是解析的，所以上式对于所有的$x$都恒成立。感兴趣的读者，不妨试试直接从代数角度证明它。

## 文章小结 #

本文讨论了“四鸭共半圆”的推广问题的一般解法，其中主要思想是容斥原理。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9324>_

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

苏剑林. (Oct. 25, 2022). 《圆内随机n点在同一个圆心角为θ的扇形的概率 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9324>

@online{kexuefm-9324,  
title={圆内随机n点在同一个圆心角为θ的扇形的概率},  
author={苏剑林},  
year={2022},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/9324}},  
} 


---

## 公式推导与注释

本文探讨了一个经典的几何概率问题：圆内随机n点在同一个扇形内的概率。这个问题涉及顺序统计量、容斥原理等重要概率论概念，下面我们进行极详细的数学推导。

### 1. 问题背景与动机

#### 1.1 "四鸭共半圆"问题

**原问题**：圆形水池中有4只鸭子随机分布，求它们都在某个半圆内的概率。

**数学形式化**：
- 圆内均匀分布的4个点
- 事件：存在某个半圆包含所有4个点
- 求：该事件的概率 $P_4(\theta = \pi)$

#### 1.2 问题的推广

**推广1（维度推广）**：
$$d\text{维超球内均匀分布的}n\text{个点位于同一}d\text{维超半球的概率}$$

这已在1962年的论文《A Problem in Geometric Probability》中解决。

**推广2（角度推广，本文重点）**：
$$\text{圆内均匀分布的}n\text{个点位于同一圆心角为}\theta\text{的扇形的概率}$$

### 2. 问题的等价转换

#### 2.1 从圆内点到圆周点

**定理 1**：以下两个问题等价：

**问题 A**：圆内均匀分布的 $n$ 个点位于某个圆心角为 $\theta$ 的扇形内。

**问题 B**：圆周上均匀分布的 $n$ 个点，最长圆弧对应的圆心角 $> 2\pi - \theta$。

**证明**：

对于圆内的点，其位置由极坐标 $(r_i, \phi_i)$ 确定，其中：
- $r_i$ 的分布：$p(r) = 2r$（圆内均匀分布）
- $\phi_i$ 的分布：$p(\phi) = \frac{1}{2\pi}$（角度均匀）

**关键观察**：是否在同一扇形内只依赖于角度 $\phi_i$，与半径 $r_i$ 无关。

因此，我们可以将每个点"投影"到圆周上（保持角度），问题变为圆周上的问题。

**扇形覆盖的等价性**：

$n$ 个点在圆周上的角度为 $\phi_1, \ldots, \phi_n$（不失一般性设 $0 \leq \phi_1 < \phi_2 < \cdots < \phi_n < 2\pi$）。

存在圆心角为 $\theta$ 的扇形包含所有点，等价于：

存在某个起始角 $\alpha$ 使得所有 $\phi_i \in [\alpha, \alpha + \theta] \pmod{2\pi}$。

这等价于：相邻点之间的最大间隔角度 $\leq 2\pi - \theta$。

**间隔角度的定义**：

$$\Delta_1 = \phi_2 - \phi_1, \quad \Delta_2 = \phi_3 - \phi_2, \quad \ldots, \quad \Delta_{n-1} = \phi_n - \phi_{n-1}, \quad \Delta_n = 2\pi + \phi_1 - \phi_n$$

注意 $\sum_{i=1}^n \Delta_i = 2\pi$。

**等价性**：

$$\text{存在扇形覆盖所有点} \Leftrightarrow \max_{i} \Delta_i \leq 2\pi - \theta \Leftrightarrow \max_i \Delta_i > 2\pi - \theta \text{ 不成立}$$

反过来：

$$P(\text{覆盖}) = 1 - P(\max_i \Delta_i > 2\pi - \theta)$$

但我们要求的正是 $P(\max_i \Delta_i > 2\pi - \theta)$（对于最长弧）。✅

#### 2.2 从圆周到线段（归一化）

**定理 2**：问题B等价于：

**问题 C**：单位线段 $[0, 1]$ 上均匀随机放置 $n-1$ 个点，将线段分成 $n$ 段，最长段的长度 $> 1 - \frac{\theta}{2\pi}$。

**证明**：

定义归一化长度：

$$x_i = \frac{\Delta_i}{2\pi}, \quad i = 1, \ldots, n$$

则：
- $x_i \geq 0$
- $\sum_{i=1}^n x_i = 1$
- $x_i$ 在 $[0, 1]$ 的单纯形上均匀分布

**等价性**：

$$\max_i \Delta_i > 2\pi - \theta \Leftrightarrow \max_i x_i > \frac{2\pi - \theta}{2\pi} = 1 - \frac{\theta}{2\pi}$$

记 $c = 1 - \frac{\theta}{2\pi}$，则问题变为求 $P(\max_i x_i > c)$。✅

### 3. 联合分布的严格推导

#### 3.1 问题设定

在 $[0, 1]$ 上均匀随机放置 $n-1$ 个点 $u_1, \ldots, u_{n-1}$，排序后得到 $0 < u_{(1)} < u_{(2)} < \cdots < u_{(n-1)} < 1$。

定义间隔：

$$x_1 = u_{(1)}, \quad x_2 = u_{(2)} - u_{(1)}, \quad \ldots, \quad x_{n-1} = u_{(n-1)} - u_{(n-2)}, \quad x_n = 1 - u_{(n-1)}$$

约束：$x_i > 0$ 且 $\sum_{i=1}^n x_i = 1$。

#### 3.2 联合分布的推导

**定理 3**：$(x_1, \ldots, x_n)$ 的联合概率密度函数为：

$$p_n(x_1, \ldots, x_n) = (n-1)! \quad \text{在单纯形 } \Delta^{n-1} = \{x_i \geq 0, \sum x_i = 1\} \text{ 上}$$

**证明（归纳法）**：

**基础情况（$n=2$）**：

一个点 $u_1 \sim \text{Uniform}(0, 1)$，分成两段：
- $x_1 = u_1$
- $x_2 = 1 - u_1$

雅可比变换：

$$\frac{\partial(x_1, x_2)}{\partial u_1} = \begin{vmatrix} 1 \\ -1 \end{vmatrix}$$

但 $x_1, x_2$ 有约束 $x_1 + x_2 = 1$，只有1个自由度。

在约束下，$p_2(x_1, x_2) = p_{u_1}(u_1) = 1$（因为 $u_1 = x_1$ 是均匀分布）。✅

**归纳步骤（$n-1 \to n$）**：

假设 $p_{n-1}(x_1, \ldots, x_{n-1}) = (n-2)!$。

现在添加第 $n-1$ 个点。考虑两种情况（以 $n=3$ 为例说明逻辑）：

设原来有1个点，分成 $(y_1, y_2)$ 两段，$p_2(y_1, y_2) = 1$。

现在加入第2个点，可能落在：
1. **第1段** $y_1$ 内（概率 $y_1$），将其分成 $(x_1, x_2)$，剩余 $x_3 = y_2$
2. **第2段** $y_2$ 内（概率 $y_2$），将其分成 $(x_2, x_3)$，剩余 $x_1 = y_1$

**情况1的密度贡献**：

点落在 $y_1$ 内的位置 $\sim \text{Uniform}(0, y_1)$，设该位置为 $v$，则：
- $x_1 = v$，$x_2 = y_1 - v$，$x_3 = y_2$
- 雅可比：$\frac{1}{y_1}$（归一化到 $[0, y_1]$）
- 贡献：$p_2(y_1, y_2) \cdot y_1 \cdot \frac{1}{y_1} = 1$

但要考虑约束。实际上，通过更仔细的计算：

$$p_3(x_1, x_2, x_3) dx_1 dx_2 = p_2(x_1, x_2+x_3) dx_1 \cdot \frac{dx_2}{x_2+x_3} \cdot (x_2+x_3) + p_2(x_1+x_2, x_3) \cdot \frac{dx_1}{x_1+x_2} \cdot (x_1+x_2) dx_2$$

$$= p_2(x_1, x_2+x_3) dx_1 dx_2 + p_2(x_1+x_2, x_3) dx_1 dx_2$$

$$= 1 \cdot dx_1 dx_2 + 1 \cdot dx_1 dx_2 = 2 dx_1 dx_2$$

因此 $p_3(x_1, x_2, x_3) = 2 = 2!$。✅

**一般情况**：

$$p_n(x_1, \ldots, x_n) = \sum_{i=1}^{n-1} p_{n-1}(\text{合并第}i\text{段和第}(i+1)\text{段后的}(n-1)\text{段})$$

由于对称性，每项贡献相同，且有 $n-1$ 项：

$$p_n = (n-1) \cdot p_{n-1} = (n-1) \cdot (n-2)! = (n-1)!$$ ✅

#### 3.3 几何解释

**单纯形的体积**：

$n$ 维单纯形 $\Delta^{n-1} = \{x \in \mathbb{R}^n: x_i \geq 0, \sum x_i = 1\}$ 的 $(n-1)$ 维体积为：

$$\text{Vol}(\Delta^{n-1}) = \frac{1}{(n-1)!}$$

**均匀分布**：

如果 $(x_1, \ldots, x_n)$ 在 $\Delta^{n-1}$ 上均匀分布，其密度应为：

$$p(x_1, \ldots, x_n) = \frac{1}{\text{Vol}(\Delta^{n-1})} = (n-1)!$$

这与我们的结果一致！✅

### 4. 边缘分布的计算

#### 4.1 任意 $k$ 个变量的边缘分布

**定理 4**：对于 $1 \leq k \leq n-1$，前 $k$ 个变量的边缘分布为：

$$p_n(x_1, \ldots, x_k) = \frac{(n-1)!}{(n-k-1)!} \left(1 - \sum_{i=1}^k x_i\right)^{n-k-1}$$

**证明**：

$$p_n(x_1, \ldots, x_k) = \int_0^{\infty} \cdots \int_0^{\infty} p_n(x_1, \ldots, x_n) \, dx_{k+1} \cdots dx_{n-1} \cdot \delta\left(\sum_{i=1}^n x_i - 1\right)$$

由于 $x_n = 1 - \sum_{i=1}^{n-1} x_i$，实际上是 $(n-1)$ 维积分。

**逐次积分**：

记 $S_k = \sum_{i=1}^k x_i$。

首先对 $x_{n-1}$ 积分，约束为 $x_{n-1} \in [0, 1 - S_{n-2}]$：

$$\int_0^{1-S_{n-2}} (n-1)! \, dx_{n-1} = (n-1)! \cdot (1 - S_{n-2})$$

然后对 $x_{n-2}$ 积分，约束为 $x_{n-2} \in [0, 1 - S_{n-3}]$：

$$\int_0^{1-S_{n-3}} (n-1)! (1 - S_{n-2}) \, dx_{n-2} = (n-1)! \int_0^{1-S_{n-3}} (1 - S_{n-3} - x_{n-2}) \, dx_{n-2}$$

$$= (n-1)! \left[(1-S_{n-3}) x_{n-2} - \frac{x_{n-2}^2}{2}\right]_0^{1-S_{n-3}} = (n-1)! \cdot \frac{(1-S_{n-3})^2}{2}$$

**归纳模式**：

经过 $n-k-1$ 次积分后：

$$p_n(x_1, \ldots, x_k) = (n-1)! \cdot \frac{(1-S_k)^{n-k-1}}{(n-k-1)!} = \frac{(n-1)!}{(n-k-1)!} (1-S_k)^{n-k-1}$$ ✅

#### 4.2 联合超出概率

**定理 5**：对于 $c_1, \ldots, c_k \geq 0$ 满足 $\sum c_i \leq 1$：

$$P_n(x_1 > c_1, \ldots, x_k > c_k) = \left(1 - \sum_{i=1}^k c_i\right)^{n-1}$$

**证明**：

$$P_n(x_1 > c_1, \ldots, x_k > c_k) = \int_{c_1}^{\infty} \cdots \int_{c_k}^{\infty} p_n(x_1, \ldots, x_k) \prod_{i=1}^{k-1} dx_i$$

（注意由于约束，实际上限是有限的）

准确地：

$$= \int_{c_1}^{1-\sum_{j=2}^k c_j} \int_{c_2}^{1-x_1-\sum_{j=3}^k c_j} \cdots \int_{c_k}^{1-\sum_{i=1}^{k-1} x_i} p_n(x_1, \ldots, x_k) \, dx_k \cdots dx_1$$

**逐次积分（从内到外）**：

$$\int_{c_k}^{1-S_{k-1}} \frac{(n-1)!}{(n-k-1)!} (1-S_{k-1}-x_k)^{n-k-1} dx_k$$

令 $t = 1 - S_{k-1} - x_k$，则 $dt = -dx_k$：

$$= \frac{(n-1)!}{(n-k-1)!} \int_0^{1-S_{k-1}-c_k} t^{n-k-1} dt = \frac{(n-1)!}{(n-k-1)!} \cdot \frac{(1-S_{k-1}-c_k)^{n-k}}{n-k}$$

$$= \frac{(n-1)!}{(n-k)!} (1-S_{k-1}-c_k)^{n-k}$$

继续对 $x_{k-1}$ 积分，类似地可得：

$$\frac{(n-1)!}{(n-k+1)!} (1-S_{k-2}-c_{k-1}-c_k)^{n-k+1}$$

**最终结果**（经过 $k$ 次积分）：

$$P_n(x_1 > c_1, \ldots, x_k > c_k) = \frac{(n-1)!}{(n-1)!} \left(1 - \sum_{i=1}^k c_i\right)^{n-1} = \left(1 - \sum_{i=1}^k c_i\right)^{n-1}$$ ✅

**特例验证**：

$$P_n(x_1 > c_1) = (1 - c_1)^{n-1}$$

这符合直觉：第一段 $> c_1$ 意味着第一个点落在 $[c_1, 1]$，概率为 $1-c_1$，其余 $n-2$ 个点独立，总概率 $(1-c_1)^{n-1}$（精确计算更复杂，但结果一致）。

### 5. 容斥原理的应用

#### 5.1 容斥原理（Inclusion-Exclusion Principle）

**原理**：对于事件 $A_1, \ldots, A_n$：

$$P\left(\bigcup_{i=1}^n A_i\right) = \sum_{k=1}^n (-1)^{k-1} \sum_{1 \leq i_1 < \cdots < i_k \leq n} P(A_{i_1} \cap \cdots \cap A_{i_k})$$

#### 5.2 应用于最大值问题

定义事件：$A_i = \{x_i > c\}$。

则：

$$\{\max(x_1, \ldots, x_n) > c\} = A_1 \cup A_2 \cup \cdots \cup A_n$$

**容斥展开**：

$$P(\max(x_1, \ldots, x_n) > c) = \sum_{k=1}^n (-1)^{k-1} \sum_{1 \leq i_1 < \cdots < i_k \leq n} P(x_{i_1} > c, \ldots, x_{i_k} > c)$$

**对称性**：

由于联合分布对所有变量对称（都是从同一过程产生），任意选择 $k$ 个变量的联合概率都相同：

$$P(x_{i_1} > c, \ldots, x_{i_k} > c) = P(x_1 > c, \ldots, x_k > c) = (1 - kc)^{n-1}$$

（当 $kc \leq 1$ 时有效；否则概率为0）

**组合计数**：

从 $n$ 个变量中选 $k$ 个的方法数为 $\binom{n}{k}$。

#### 5.3 最终公式

**定理 6**（最大值分布）：

$$P_n(\max(x_1, \ldots, x_n) > c) = \sum_{k=1}^{\lfloor 1/c \rfloor} (-1)^{k-1} \binom{n}{k} (1 - kc)^{n-1}$$

其中求和上限 $\lfloor 1/c \rfloor$ 确保 $1 - kc > 0$。

**展开形式**：

$$= \binom{n}{1}(1-c)^{n-1} - \binom{n}{2}(1-2c)^{n-1} + \binom{n}{3}(1-3c)^{n-1} - \cdots$$

**特殊情况分析**：

**情况1**（$c > 1/2$）：

此时 $1 - 2c < 0$，只有第一项非零：

$$P_n(\max > c) = n(1-c)^{n-1}$$

**情况2**（$1/3 < c \leq 1/2$）：

$$P_n(\max > c) = n(1-c)^{n-1} - \binom{n}{2}(1-2c)^{n-1}$$

**情况3**（$c \leq 1/n$）：

所有 $n$ 项都存在。由抽屉原理，$\max \geq 1/n$ 必然成立，所以：

$$\sum_{k=1}^n (-1)^{k-1} \binom{n}{k} (1-kc)^{n-1} = 1 \quad \text{当 } c \leq 1/n$$

### 6. 回到原问题

#### 6.1 参数替换

回到圆内 $n$ 点在圆心角 $\theta$ 扇形内的问题：

$$c = 1 - \frac{\theta}{2\pi}$$

因此：

$$P_n(\text{存在扇形覆盖}) = P_n\left(\max > 1 - \frac{\theta}{2\pi}\right)$$

#### 6.2 "四鸭共半圆"的答案

$n = 4$，$\theta = \pi$，$c = 1 - \frac{\pi}{2\pi} = \frac{1}{2}$。

由于 $c = 1/2$，只有第一项：

$$P_4\left(\max > \frac{1}{2}\right) = 4 \left(1 - \frac{1}{2}\right)^{4-1} = 4 \cdot \frac{1}{8} = \frac{1}{2}$$

**答案：$\boxed{\frac{1}{2}}$**

#### 6.3 一般公式

$$P_n(\theta) = \sum_{k=1}^{\lfloor 2\pi/(2\pi - \theta) \rfloor} (-1)^{k-1} \binom{n}{k} \left(\frac{\theta}{2\pi}\right)^{n-1} \left(\frac{2\pi - k(2\pi - \theta)}{2\pi - \theta}\right)^{n-1}$$

简化：

$$P_n(\theta) = \sum_{k=1}^{K_\theta} (-1)^{k-1} \binom{n}{k} \left(1 - k\left(1 - \frac{\theta}{2\pi}\right)\right)^{n-1}$$

其中 $K_\theta = \lfloor 2\pi/(2\pi - \theta) \rfloor$。

**特殊情况**：

- $\theta < \pi$：$K_\theta = 1$，$P_n(\theta) = n \left(\frac{\theta}{2\pi}\right)^{n-1}$
- $\theta = \pi$：$K_\pi = 2$
- $\theta \to 2\pi$：$K_\theta \to n$（所有项都需要）

### 7. 顺序统计量理论

#### 7.1 最大值的累积分布函数（CDF）

对于 i.i.d. 随机变量 $X_1, \ldots, X_n \sim F(x)$：

$$F_{\max}(x) = P(\max(X_1, \ldots, X_n) \leq x) = \prod_{i=1}^n P(X_i \leq x) = [F(x)]^n$$

对于我们的问题，单个 $x_i$ 的边缘分布（通过积分联合分布得到）需要仔细计算。

实际上，由于联合分布的特殊性（Dirichlet 分布），直接推导更复杂，容斥原理提供了更简洁的路径。

#### 7.2 与 Dirichlet 分布的联系

**定理 7**：$(x_1, \ldots, x_n) \sim \text{Dirichlet}(1, 1, \ldots, 1) = \text{Dirichlet}(\mathbf{1}_n)$。

**Dirichlet 分布**的密度：

$$p(x_1, \ldots, x_n) = \frac{\Gamma(\sum \alpha_i)}{\prod \Gamma(\alpha_i)} \prod x_i^{\alpha_i - 1}$$

当 $\alpha_i = 1$ 时：

$$p(x_1, \ldots, x_n) = \frac{\Gamma(n)}{[\Gamma(1)]^n} \prod x_i^{0} = (n-1)!$$

（在单纯形上，考虑约束后实际是 $(n-1)$ 维）

这证实了我们之前的结果！

#### 7.3 期望值

**最大段的期望长度**：

$$\mathbb{E}[\max(x_1, \ldots, x_n)] = \int_0^1 P(\max > c) \, dc$$

$$= \int_0^1 \sum_{k=1}^{\lfloor 1/c \rfloor} (-1)^{k-1} \binom{n}{k} (1-kc)^{n-1} dc$$

这个积分比较复杂，但可以分段计算：

$$= \int_{1/2}^1 n(1-c)^{n-1} dc + \int_{1/3}^{1/2} \left[n(1-c)^{n-1} - \binom{n}{2}(1-2c)^{n-1}\right] dc + \cdots$$

**已知结果**（通过复杂计算）：

$$\mathbb{E}[\max] = H_n / n$$

其中 $H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n}$ 是调和数。

### 8. 数值验证与示例

#### 8.1 $n=4$, $\theta = \pi$（四鸭共半圆）

$$c = 1 - \frac{\pi}{2\pi} = \frac{1}{2}$$

$$P_4 = \binom{4}{1} \left(\frac{1}{2}\right)^3 = 4 \cdot \frac{1}{8} = \boxed{\frac{1}{2}}$$

#### 8.2 $n=4$, $\theta = \frac{2\pi}{3}$

$$c = 1 - \frac{1}{3} = \frac{2}{3} > \frac{1}{2}$$

$$P_4 = 4 \left(\frac{1}{3}\right)^3 = \frac{4}{27} \approx 0.148$$

#### 8.3 $n=4$, $\theta = \frac{3\pi}{2}$

$$c = 1 - \frac{3}{4} = \frac{1}{4}$$

$$\lfloor 1/c \rfloor = 4$$，所有项都存在：

$$P_4 = 4 \cdot \left(\frac{3}{4}\right)^3 - 6 \cdot \left(\frac{1}{2}\right)^3 + 4 \cdot \left(\frac{1}{4}\right)^3 - 0$$

$$= 4 \cdot \frac{27}{64} - 6 \cdot \frac{1}{8} + 4 \cdot \frac{1}{64}$$

$$= \frac{108}{64} - \frac{48}{64} + \frac{4}{64} = \frac{64}{64} = 1$$

符合预期（$\theta > \pi$ 意味着扇形超过半圆，必然能覆盖）。

### 9. 恒等式的代数验证

**定理 8**：对于 $c \leq 1/n$：

$$\sum_{k=1}^n (-1)^{k-1} \binom{n}{k} (1-kc)^{n-1} = 1$$

**证明（生成函数方法）**：

考虑多项式：

$$f(x) = \sum_{k=1}^n (-1)^{k-1} \binom{n}{k} (1-kx)^{n-1}$$

我们需要证明当 $x \leq 1/n$ 时，$f(x) = 1$。

**关键观察**：由于 $f(x)$ 是多项式，如果能证明它在某个区间恒为1，则它必是常数1（因为非常数多项式最多在有限个点等于常数）。

但这个论证不够严密。更严格的方法是直接展开计算（涉及二项式定理和复杂的组合恒等式）。

**另一角度（概率论证）**：

当 $c = 1/n$ 时，由抽屉原理，$n$ 段中至少有一段 $\geq 1/n$，所以：

$$P(\max \geq 1/n) = 1$$

连续性论证表明对 $c < 1/n$ 也成立。✅

### 10. 应用与扩展

#### 10.1 生日问题的推广

经典生日问题：$n$ 个人，至少两人同一天生日的概率。

本问题的对偶：$n$ 个点在圆周，最大间隔的分布。

#### 10.2 机器学习中的应用

- **熵估计**：最邻近估计法依赖顺序统计量
- **生成模型**：Gumbel-Max 技巧（参见本系列其他文章）
- **鲁棒性分析**：数据分布的极值行为

### 11. 总结

本文通过以下步骤解决了圆内随机 $n$ 点在扇形内的概率问题：

1. **等价转换**：圆内点 → 圆周点 → 线段分割
2. **联合分布**：证明间隔长度服从 Dirichlet$(1, \ldots, 1)$，密度为 $(n-1)!$
3. **边缘分布**：推导任意 $k$ 个变量的边缘分布和联合超出概率
4. **容斥原理**：应用于计算最大值超过阈值的概率
5. **最终公式**：

$$P_n(\max > c) = \sum_{k=1}^{\lfloor 1/c \rfloor} (-1)^{k-1} \binom{n}{k} (1-kc)^{n-1}$$

**关键洞察**：
- 均匀随机分割导致对称的 Dirichlet 分布
- 容斥原理优雅地处理了最大值的非独立性
- 几何概率问题可转化为组合优化问题

