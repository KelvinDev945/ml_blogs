---
title: 矩阵符号函数mcsgn能计算什么？
slug: 矩阵符号函数mcsgn能计算什么
date: 2025-06-23
tags: 代数, 矩阵, 线性, 生成模型, attention
status: completed
---

# 矩阵符号函数mcsgn能计算什么？

**原文链接**: [https://spaces.ac.cn/archives/11056](https://spaces.ac.cn/archives/11056)

**发布日期**: 

---

在[《msign的导数》](/archives/11025)一文中，我们正式引入了两种矩阵符号函数$\newcommand{msign}{\mathop{\text{msign}}}\msign$和$\newcommand{mcsgn}{\mathop{\text{mcsgn}}}\mcsgn$，其中$\msign$是Muon的核心运算，而$\mcsgn$则是用来解[Sylvester方程]( https://en.wikipedia.org/wiki/Sylvester_equation)。那么$\mcsgn$除了用来解Sylvester方程外，还能干些什么呢？本文就来整理一下这个问题的答案。

## 两种符号 #

设矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，我们有两种矩阵符号函数  
\begin{gather}\msign(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2} \\\\[6pt]  
\mcsgn(\boldsymbol{M}) = (\boldsymbol{M}^2)^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}  
\end{gather}  
第一种适用于任意形状的矩阵，第二种只适用于方阵，$^{-1/2}$是矩阵$1/2$次幂的逆，如果不可逆则按“[伪逆](/archives/10366)”来算。一般情况下$\msign$和$\mcsgn$会得到不同的结果，但当$\boldsymbol{M}$是对称矩阵时它们则相等。

它们的区别是，如果$\boldsymbol{M}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，其中$\boldsymbol{U},\boldsymbol{V}$是正交矩阵，那么$\msign(\boldsymbol{M}) = \boldsymbol{U}\msign(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}$；如果$\boldsymbol{M}=\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，其中$\boldsymbol{P}$是可逆矩阵，那么$\mcsgn(\boldsymbol{M})=\boldsymbol{P}\mcsgn(\boldsymbol{\Lambda})\boldsymbol{P}^{-1}$。说白了，一个具有正交不变性，一个具有相似不变性，一个会把所有非零奇异值变成1，一个会把所有非零特征值变成$\pm 1$。

关于$\msign$的计算，可以看[《msign算子的Newton-Schulz迭代（上）》](/archives/10922)和[《msign算子的Newton-Schulz迭代（下）》](/archives/10996)，它是GPU高效的。至于$\mcsgn$，由于特征值允许复数，一般情况会很复杂，但当$\boldsymbol{M}$的特征值全是实数时（实际上用到$\mcsgn$的场景几乎都是这样），可以复用$\msign$的迭代：  
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\sqrt{\tr(\boldsymbol{M}^2)}},\qquad \boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t^3 + c_{t+1}\boldsymbol{X}_t^5\end{equation}

更多的性质我们就不展开了，接下来我们主要看$\mcsgn$的应用。

## 分块恒等 #

历史上，$\mcsgn$就是为了解方程而引入的，只不过不单单是Sylvester方程，还包含更一般的[代数Riccati方程](https://en.wikipedia.org/wiki/Algebraic_Riccati_equation)，原论文可见[《Solving the algebraic Riccati equation with the matrix sign function》](https://www.sciencedirect.com/science/article/pii/0024379587902229)。

考虑分块矩阵$\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}$，我们有$\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} = \begin{bmatrix}\boldsymbol{0} & \boldsymbol{I} \\\ -\boldsymbol{I} & \boldsymbol{X}\end{bmatrix}$，可以验算  
\begin{equation}\begin{bmatrix}\boldsymbol{0} & \boldsymbol{I} \\\ -\boldsymbol{I} & \boldsymbol{X}\end{bmatrix}\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\\ \boldsymbol{D} & \boldsymbol{B}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}=\begin{bmatrix}\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X} & -\boldsymbol{D} \\\ \boldsymbol{X}\boldsymbol{D}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} - \boldsymbol{C} & \boldsymbol{A} - \boldsymbol{X}\boldsymbol{D}\end{bmatrix}\end{equation}  
如果  
\begin{equation}\boldsymbol{X}\boldsymbol{D}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} - \boldsymbol{C} = \boldsymbol{0}\label{eq:riccati}\end{equation}  
那么  
\begin{equation}\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\\ \boldsymbol{D} & \boldsymbol{B}\end{bmatrix}=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X} & -\boldsymbol{D} \\\ \boldsymbol{0} & \boldsymbol{A} - \boldsymbol{X}\boldsymbol{D}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1}\end{equation}  
式$\eqref{eq:riccati}$即为代数Riccati方程。两边取$\mcsgn$，我们有恒等式  
\begin{equation}\begin{aligned}  
\mcsgn\left(\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\\ \boldsymbol{D} & \boldsymbol{B}\end{bmatrix}\right)=&\,\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\mcsgn\left(\begin{bmatrix}\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X} & -\boldsymbol{D} \\\ \boldsymbol{0} & \boldsymbol{A} - \boldsymbol{X}\boldsymbol{D}\end{bmatrix}\right)\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} \\\\[6pt]  
=&\,\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\mcsgn(\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X}) & \boldsymbol{Y} \\\ \boldsymbol{0} & \mcsgn(\boldsymbol{A} - \boldsymbol{X}\boldsymbol{D})\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1}  
\end{aligned}\end{equation}  
第二个等号是利用了（分块）三角矩阵的特性。三角矩阵的特征值就是它的对角线元素，所以三角矩阵取$\mcsgn$，结果也是一个三角矩阵，其对角线元素等于原矩阵对角线元素的$\mathop{\text{csgn}}$，这个特性对分块三角矩阵也成立，所以结果具有第二个等号的形式，$\boldsymbol{Y}$是一个待定矩阵。

## 几个结果 #

下面就是根据具体情况来进一步化简，得到一些可能会用到的结果。

### 第一例子 #

假设$\boldsymbol{D}=\boldsymbol{0}$，$\boldsymbol{B}$是正定的，$\boldsymbol{A}$是负定的，分块对角阵运算是封闭的，于是$\boldsymbol{Y}=\boldsymbol{0}$，那么  
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\\ \boldsymbol{0} & \boldsymbol{B}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} = \begin{bmatrix}-\boldsymbol{I} & 2\boldsymbol{X} \\\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\end{equation}  
这意味着直接从$\mcsgn\left(\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\\ \boldsymbol{0} & \boldsymbol{B}\end{bmatrix}\right)$就可以读出Sylvester方程$\boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} = \boldsymbol{C}$的解。

### 第二例子 #

假设$\boldsymbol{A},\boldsymbol{B}=\boldsymbol{0}$，$\boldsymbol{D}=\boldsymbol{I}$，$\boldsymbol{C}$是正定矩阵，那么Riccati方程简化为$\boldsymbol{X}^2 = \boldsymbol{C}$，即$\boldsymbol{X}=\boldsymbol{C}^{1/2}$，那么$\mcsgn(\boldsymbol{C}^{1/2})=\boldsymbol{I}$，所以  
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{Y} \\\ \boldsymbol{0} & \- \boldsymbol{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} = \begin{bmatrix}-\boldsymbol{X}\boldsymbol{Y}-\boldsymbol{I} & 2\boldsymbol{X} + \boldsymbol{X}\boldsymbol{Y}\boldsymbol{X} \\\ -\boldsymbol{Y} & \boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}\end{bmatrix}\end{equation}  
注意到$\mcsgn$是奇函数，反对角阵的奇函数必然也是反对角阵，因此$\boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}=\boldsymbol{0}$，从中解得$\boldsymbol{Y} = -\boldsymbol{X}^{-1} = -\boldsymbol{C}^{-1/2}$，代入上式得到  
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C}^{1/2} \\\ \boldsymbol{C}^{-1/2} & \boldsymbol{0}\end{bmatrix}\end{equation}  
这表明$\mcsgn$还可以用来算矩阵的平方根和逆平方根。

### 第三例子 #

假设$\boldsymbol{A},\boldsymbol{B}=\boldsymbol{0}$，$\boldsymbol{D}=\boldsymbol{C}^{\top}$，那么Riccati方程简化为$\boldsymbol{X}\boldsymbol{C}^{\top}\boldsymbol{X} = \boldsymbol{C}$，容易验证$\boldsymbol{X}=\msign(\boldsymbol{C})$正是它的解。我们只演示最理想的情况，$\boldsymbol{C}$是满秩方阵，那么$\boldsymbol{C}^{\top}\boldsymbol{X}$和$\boldsymbol{X}\boldsymbol{C}^{\top}$都是正定的，于是有  
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\\ \boldsymbol{C}^{\top} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{Y} \\\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1}=\begin{bmatrix}-\boldsymbol{X}\boldsymbol{Y}-\boldsymbol{I} & 2\boldsymbol{X} + \boldsymbol{X}\boldsymbol{Y}\boldsymbol{X} \\\ -\boldsymbol{Y} & \boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}\end{bmatrix}\end{equation}  
跟上一节同理$\boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}=0$，所以  
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\\ \boldsymbol{C}^{\top} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{0} & \msign(\boldsymbol{C}) \\\ \msign(\boldsymbol{C}^{\top}) & \boldsymbol{0}\end{bmatrix}\end{equation}  
即$\mcsgn$也可以用来算$\msign$。其实可以直接证明这个等式对于任意矩阵$\boldsymbol{C}$都是成立的，但如果从这里的解Riccati方程角度来证，则会有些繁琐的细节，读者可以自行补充一下。

### 第四例子 #

第二例子和第三例子可以推广为一个更一般的结论：  
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\\ \boldsymbol{D} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C}(\boldsymbol{D}\boldsymbol{C})^{-1/2} \\\ \boldsymbol{D}(\boldsymbol{C}\boldsymbol{D})^{-1/2} & \boldsymbol{0}\end{bmatrix}\end{equation}  
对任意形状适合的$\boldsymbol{C},\boldsymbol{D}$都恒成立。矩阵请读者自行完成证明。

## 文章小结 #

本文主要从解代数Riccati方程的角度，整理了几个$\mcsgn$相关的恒等式。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11056>_

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

苏剑林. (Jun. 23, 2025). 《矩阵符号函数mcsgn能计算什么？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11056>

@online{kexuefm-11056,  
title={矩阵符号函数mcsgn能计算什么？},  
author={苏剑林},  
year={2025},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/11056}},  
} 


---

## 推导

本节将从多个角度对矩阵符号函数$\mcsgn$进行深入的数学推导，包括理论基础、计算方法和应用分析。

### 矩阵符号函数的定义与性质

#### 标量符号函数的回顾

对于标量$x\in\mathbb{R}$，符号函数定义为：
$$\text{sgn}(x) = \begin{cases}
1, & x > 0 \\
0, & x = 0 \\
-1, & x < 0
\end{cases}$$

其复数形式为：
$$\text{csgn}(z) = \begin{cases}
\frac{z}{|z|}, & z \neq 0 \\
0, & z = 0
\end{cases}$$

注意到$\text{csgn}(z) = z \cdot |z|^{-1} = z \cdot (z\bar{z})^{-1/2}$，这个形式启发了矩阵符号函数的定义。

#### 矩阵符号函数的两种定义

对于方阵$\boldsymbol{M}\in\mathbb{R}^{n\times n}$，我们可以通过两种方式推广符号函数：

**定义1（基于极分解）**：对于可逆矩阵$\boldsymbol{M}$，其极分解为：
$$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{H}$$
其中$\boldsymbol{U}$是正交矩阵（或酉矩阵），$\boldsymbol{H}$是正定Hermite矩阵。定义：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{U}$$

**定义2（基于谱分解）**：设$\boldsymbol{M}$可对角化，即$\boldsymbol{M} = \boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，其中$\boldsymbol{\Lambda}=\text{diag}(\lambda_1,\ldots,\lambda_n)$，定义：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{P}\,\text{diag}(\text{csgn}(\lambda_1),\ldots,\text{csgn}(\lambda_n))\,\boldsymbol{P}^{-1}$$

**定义3（基于函数演算）**：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}$$

下面我们证明这三种定义的等价性。

#### 等价性证明

**定理1**：对于特征值全为实数的可对角化矩阵$\boldsymbol{M}$，定义2和定义3等价。

**证明**：设$\boldsymbol{M} = \boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，其中$\boldsymbol{\Lambda}=\text{diag}(\lambda_1,\ldots,\lambda_n)$，$\lambda_i\in\mathbb{R}$。

首先计算$\boldsymbol{M}^2$：
$$\boldsymbol{M}^2 = (\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1})^2 = \boldsymbol{P}\boldsymbol{\Lambda}^2\boldsymbol{P}^{-1}$$

其中$\boldsymbol{\Lambda}^2 = \text{diag}(\lambda_1^2,\ldots,\lambda_n^2)$。

对于矩阵平方根，我们有：
$$(\boldsymbol{M}^2)^{1/2} = \boldsymbol{P}(\boldsymbol{\Lambda}^2)^{1/2}\boldsymbol{P}^{-1} = \boldsymbol{P}\,\text{diag}(|\lambda_1|,\ldots,|\lambda_n|)\,\boldsymbol{P}^{-1}$$

因此：
$$(\boldsymbol{M}^2)^{-1/2} = \boldsymbol{P}\,\text{diag}(|\lambda_1|^{-1},\ldots,|\lambda_n|^{-1})\,\boldsymbol{P}^{-1}$$

计算$\mcsgn(\boldsymbol{M})$：
$$\begin{aligned}
\mcsgn(\boldsymbol{M}) &= \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2} \\
&= \boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1} \cdot \boldsymbol{P}\,\text{diag}(|\lambda_1|^{-1},\ldots,|\lambda_n|^{-1})\,\boldsymbol{P}^{-1} \\
&= \boldsymbol{P}\boldsymbol{\Lambda}\,\text{diag}(|\lambda_1|^{-1},\ldots,|\lambda_n|^{-1})\,\boldsymbol{P}^{-1} \\
&= \boldsymbol{P}\,\text{diag}(\lambda_1|\lambda_1|^{-1},\ldots,\lambda_n|\lambda_n|^{-1})\,\boldsymbol{P}^{-1} \\
&= \boldsymbol{P}\,\text{diag}(\text{sgn}(\lambda_1),\ldots,\text{sgn}(\lambda_n))\,\boldsymbol{P}^{-1}
\end{aligned}$$

这正是定义2的形式。$\square$

### 极分解定理及其应用

#### 极分解定理

**定理2（极分解定理）**：任意复矩阵$\boldsymbol{M}\in\mathbb{C}^{n\times n}$都可以唯一分解为：
$$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{H}$$
其中$\boldsymbol{U}$是酉矩阵，$\boldsymbol{H}$是正半定Hermite矩阵。

**证明**：考虑矩阵$\boldsymbol{M}^*\boldsymbol{M}$，其中$\boldsymbol{M}^*$表示$\boldsymbol{M}$的共轭转置。

**步骤1**：证明$\boldsymbol{M}^*\boldsymbol{M}$是正半定Hermite矩阵。

对任意向量$\boldsymbol{x}\in\mathbb{C}^n$：
$$\langle \boldsymbol{x}, \boldsymbol{M}^*\boldsymbol{M}\boldsymbol{x}\rangle = \langle \boldsymbol{M}\boldsymbol{x}, \boldsymbol{M}\boldsymbol{x}\rangle = \|\boldsymbol{M}\boldsymbol{x}\|^2 \geq 0$$

且$(\boldsymbol{M}^*\boldsymbol{M})^* = \boldsymbol{M}^*\boldsymbol{M}^{**} = \boldsymbol{M}^*\boldsymbol{M}$，所以$\boldsymbol{M}^*\boldsymbol{M}$是正半定Hermite矩阵。

**步骤2**：定义$\boldsymbol{H} = (\boldsymbol{M}^*\boldsymbol{M})^{1/2}$。

由谱定理，正半定Hermite矩阵$\boldsymbol{M}^*\boldsymbol{M}$可对角化为：
$$\boldsymbol{M}^*\boldsymbol{M} = \boldsymbol{Q}\boldsymbol{\Sigma}\boldsymbol{Q}^*$$
其中$\boldsymbol{Q}$是酉矩阵，$\boldsymbol{\Sigma}=\text{diag}(\sigma_1^2,\ldots,\sigma_n^2)$，$\sigma_i\geq 0$。

定义：
$$\boldsymbol{H} = \boldsymbol{Q}\,\text{diag}(\sigma_1,\ldots,\sigma_n)\,\boldsymbol{Q}^*$$

显然$\boldsymbol{H}^2 = \boldsymbol{M}^*\boldsymbol{M}$。

**步骤3**：构造$\boldsymbol{U}$。

如果$\boldsymbol{H}$可逆，定义$\boldsymbol{U} = \boldsymbol{M}\boldsymbol{H}^{-1}$。验证$\boldsymbol{U}$是酉矩阵：
$$\boldsymbol{U}^*\boldsymbol{U} = (\boldsymbol{M}\boldsymbol{H}^{-1})^*(\boldsymbol{M}\boldsymbol{H}^{-1}) = \boldsymbol{H}^{-1}\boldsymbol{M}^*\boldsymbol{M}\boldsymbol{H}^{-1} = \boldsymbol{H}^{-1}\boldsymbol{H}^2\boldsymbol{H}^{-1} = \boldsymbol{I}$$

如果$\boldsymbol{H}$不可逆，需要用Moore-Penrose伪逆$\boldsymbol{H}^+$代替。$\square$

#### 极分解与符号函数的关系

由极分解$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{H}$，我们有：
$$\boldsymbol{M}^2 = \boldsymbol{U}\boldsymbol{H}\boldsymbol{U}\boldsymbol{H} = \boldsymbol{U}\boldsymbol{H}^2\boldsymbol{U}^*$$

因此：
$$(\boldsymbol{M}^2)^{1/2} = \boldsymbol{U}\boldsymbol{H}\boldsymbol{U}^*$$

$$(\boldsymbol{M}^2)^{-1/2} = \boldsymbol{U}\boldsymbol{H}^{-1}\boldsymbol{U}^*$$

于是：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2} = \boldsymbol{U}\boldsymbol{H} \cdot \boldsymbol{U}\boldsymbol{H}^{-1}\boldsymbol{U}^* = \boldsymbol{U}\boldsymbol{U}^* = \boldsymbol{U}$$

这证明了定义1和定义3的等价性。

### 投影算子性质

#### 谱投影的定义

设$\boldsymbol{M}$的特征值可分为两组：$\{\lambda_i : \text{Re}(\lambda_i) > 0\}$和$\{\mu_j : \text{Re}(\mu_j) < 0\}$。定义谱投影：
$$\boldsymbol{P}_+ = \frac{\boldsymbol{I} + \mcsgn(\boldsymbol{M})}{2}, \quad \boldsymbol{P}_- = \frac{\boldsymbol{I} - \mcsgn(\boldsymbol{M})}{2}$$

**定理3**：$\boldsymbol{P}_+$和$\boldsymbol{P}_-$是正交投影算子，满足：

1. $\boldsymbol{P}_+^2 = \boldsymbol{P}_+$，$\boldsymbol{P}_-^2 = \boldsymbol{P}_-$（幂等性）
2. $\boldsymbol{P}_+ + \boldsymbol{P}_- = \boldsymbol{I}$（完备性）
3. $\boldsymbol{P}_+\boldsymbol{P}_- = \boldsymbol{0}$（正交性）

**证明**：设$\boldsymbol{S} = \mcsgn(\boldsymbol{M})$，则$\boldsymbol{S}^2 = \boldsymbol{I}$（因为$\text{csgn}(z)^2 = 1$对所有非零$z$成立）。

验证幂等性：
$$\boldsymbol{P}_+^2 = \frac{(\boldsymbol{I} + \boldsymbol{S})^2}{4} = \frac{\boldsymbol{I} + 2\boldsymbol{S} + \boldsymbol{S}^2}{4} = \frac{\boldsymbol{I} + 2\boldsymbol{S} + \boldsymbol{I}}{4} = \frac{\boldsymbol{I} + \boldsymbol{S}}{2} = \boldsymbol{P}_+$$

同理可证$\boldsymbol{P}_-^2 = \boldsymbol{P}_-$。

验证完备性：
$$\boldsymbol{P}_+ + \boldsymbol{P}_- = \frac{\boldsymbol{I} + \boldsymbol{S}}{2} + \frac{\boldsymbol{I} - \boldsymbol{S}}{2} = \boldsymbol{I}$$

验证正交性：
$$\boldsymbol{P}_+\boldsymbol{P}_- = \frac{(\boldsymbol{I} + \boldsymbol{S})(\boldsymbol{I} - \boldsymbol{S})}{4} = \frac{\boldsymbol{I} - \boldsymbol{S}^2}{4} = \frac{\boldsymbol{I} - \boldsymbol{I}}{4} = \boldsymbol{0}$$

$\square$

#### 不变子空间分解

**定理4**：$\boldsymbol{P}_+$和$\boldsymbol{P}_-$分别将$\mathbb{C}^n$分解为两个$\boldsymbol{M}$-不变子空间：
$$\mathbb{C}^n = \text{Im}(\boldsymbol{P}_+) \oplus \text{Im}(\boldsymbol{P}_-)$$

其中$\text{Im}(\boldsymbol{P}_+)$对应于特征值实部为正的特征向量张成的子空间，$\text{Im}(\boldsymbol{P}_-)$对应于特征值实部为负的特征向量张成的子空间。

**证明**：设$\boldsymbol{M} = \boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，其中
$$\boldsymbol{\Lambda} = \begin{bmatrix}\boldsymbol{\Lambda}_+ & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\Lambda}_-\end{bmatrix}$$
$\boldsymbol{\Lambda}_+$包含实部为正的特征值，$\boldsymbol{\Lambda}_-$包含实部为负的特征值。

则：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{P}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\boldsymbol{P}^{-1}$$

因此：
$$\boldsymbol{P}_+ = \boldsymbol{P}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{0}\end{bmatrix}\boldsymbol{P}^{-1}$$

这正是投影到正特征值对应子空间的投影算子。$\square$

### Newton迭代法的详细推导

#### Newton-Schulz迭代的构造

考虑求解方程$f(\boldsymbol{X}) = \boldsymbol{X}^2 - \boldsymbol{I} = \boldsymbol{0}$，其解为$\boldsymbol{X} = \mcsgn(\boldsymbol{M})$。

标准Newton迭代为：
$$\boldsymbol{X}_{k+1} = \boldsymbol{X}_k - [f'(\boldsymbol{X}_k)]^{-1}f(\boldsymbol{X}_k)$$

对于$f(\boldsymbol{X}) = \boldsymbol{X}^2 - \boldsymbol{I}$，Fréchet导数为：
$$f'(\boldsymbol{X})\boldsymbol{H} = \boldsymbol{X}\boldsymbol{H} + \boldsymbol{H}\boldsymbol{X}$$

Newton迭代变为：
$$\boldsymbol{X}\boldsymbol{H}_{k+1} + \boldsymbol{H}_{k+1}\boldsymbol{X} = -(\boldsymbol{X}_k^2 - \boldsymbol{I})$$

这是一个Sylvester方程，求解复杂。为避免求解，我们采用Newton-Schulz迭代。

#### Newton-Schulz迭代公式

定义迭代：
$$\boldsymbol{X}_{k+1} = \frac{1}{2}\boldsymbol{X}_k(3\boldsymbol{I} - \boldsymbol{X}_k^2)$$

**定理5**：若$\|\boldsymbol{I} - \boldsymbol{X}_0^2\| < 1$，则Newton-Schulz迭代二次收敛到$\mcsgn(\boldsymbol{M})$。

**证明**：定义误差$\boldsymbol{E}_k = \boldsymbol{I} - \boldsymbol{X}_k^2$。计算：
$$\begin{aligned}
\boldsymbol{E}_{k+1} &= \boldsymbol{I} - \boldsymbol{X}_{k+1}^2 \\
&= \boldsymbol{I} - \frac{1}{4}\boldsymbol{X}_k^2(3\boldsymbol{I} - \boldsymbol{X}_k^2)^2 \\
&= \boldsymbol{I} - \frac{1}{4}\boldsymbol{X}_k^2(9\boldsymbol{I} - 6\boldsymbol{X}_k^2 + \boldsymbol{X}_k^4) \\
&= \boldsymbol{I} - \frac{1}{4}(9\boldsymbol{X}_k^2 - 6\boldsymbol{X}_k^4 + \boldsymbol{X}_k^6)
\end{aligned}$$

利用$\boldsymbol{X}_k^2 = \boldsymbol{I} - \boldsymbol{E}_k$：
$$\begin{aligned}
\boldsymbol{E}_{k+1} &= \boldsymbol{I} - \frac{1}{4}[9(\boldsymbol{I}-\boldsymbol{E}_k) - 6(\boldsymbol{I}-\boldsymbol{E}_k)^2 + (\boldsymbol{I}-\boldsymbol{E}_k)^3] \\
&= \boldsymbol{I} - \frac{1}{4}[9\boldsymbol{I} - 9\boldsymbol{E}_k - 6\boldsymbol{I} + 12\boldsymbol{E}_k - 6\boldsymbol{E}_k^2 + \boldsymbol{I} - 3\boldsymbol{E}_k + 3\boldsymbol{E}_k^2 - \boldsymbol{E}_k^3] \\
&= \boldsymbol{I} - \frac{1}{4}[4\boldsymbol{I} - 3\boldsymbol{E}_k^2 - \boldsymbol{E}_k^3] \\
&= -\frac{1}{4}\boldsymbol{E}_k^2(3\boldsymbol{I} - \boldsymbol{E}_k)
\end{aligned}$$

如果$\|\boldsymbol{E}_k\| < 1$，则：
$$\|\boldsymbol{E}_{k+1}\| \leq \frac{1}{4}\|\boldsymbol{E}_k\|^2(3 + \|\boldsymbol{E}_k\|) < \|\boldsymbol{E}_k\|^2$$

这证明了二次收敛性。$\square$

#### 高阶Newton-Schulz迭代

为了加速收敛，可以使用高阶迭代：
$$\boldsymbol{X}_{k+1} = \boldsymbol{X}_k(a_0\boldsymbol{I} + a_1\boldsymbol{X}_k^2 + a_2\boldsymbol{X}_k^4)$$

其中系数$a_0, a_1, a_2$需满足：
$$\begin{cases}
a_0 + a_1 + a_2 = 1 \\
a_0 + 3a_1 + 5a_2 = 0
\end{cases}$$

这给出三阶收敛的迭代格式。

### Padé逼近方法

#### Padé逼近的定义

符号函数$\text{sgn}(x)$可以表示为：
$$\text{sgn}(x) = \frac{x}{\sqrt{x^2}} = x \cdot (x^2)^{-1/2}$$

对于$(1+y)^{-1/2}$的Padé逼近：
$$R_{m,n}(y) = \frac{P_m(y)}{Q_n(y)}$$

其中$P_m(y)$是$m$次多项式，$Q_n(y)$是$n$次多项式。

#### 矩阵Padé逼近

对于$\boldsymbol{M}$，首先标准化：
$$\boldsymbol{Y} = \frac{\boldsymbol{M}}{\|\boldsymbol{M}\|_2}$$

使得$\|\boldsymbol{Y}\| \approx 1$，$\|\boldsymbol{Y}^2 - \boldsymbol{I}\| < 1$。

定义$\boldsymbol{Z} = \boldsymbol{Y}^2 - \boldsymbol{I}$，则：
$$(\boldsymbol{Y}^2)^{-1/2} = (\boldsymbol{I} + \boldsymbol{Z})^{-1/2} \approx R_{m,n}(\boldsymbol{Z})$$

常用的$(3,3)$ Padé逼近为：
$$R_{3,3}(z) = \frac{1 + \frac{5}{8}z + \frac{7}{128}z^2 + \frac{1}{1024}z^3}{1 + \frac{15}{8}z + \frac{63}{128}z^2 + \frac{15}{1024}z^3}$$

因此：
$$\mcsgn(\boldsymbol{M}) \approx \boldsymbol{Y} \cdot \frac{Q_3(\boldsymbol{Z})^{-1}P_3(\boldsymbol{Z})}{1}$$

#### 误差分析

**定理6**：设$\|\boldsymbol{Z}\| \leq \delta < 1$，则$(m,n)$ Padé逼近的误差为：
$$\|(\boldsymbol{I} + \boldsymbol{Z})^{-1/2} - R_{m,n}(\boldsymbol{Z})\| \leq C\delta^{m+n+1}$$

其中$C$是与$m,n$有关的常数。

这表明Padé逼近可以达到高阶精度。

### 函数演算的理论基础

#### Cauchy积分公式

对于解析函数$f$和矩阵$\boldsymbol{M}$，函数演算定义为：
$$f(\boldsymbol{M}) = \frac{1}{2\pi i}\oint_\Gamma f(z)(z\boldsymbol{I} - \boldsymbol{M})^{-1}\,dz$$

其中$\Gamma$是包含$\boldsymbol{M}$所有特征值的闭合曲线。

对于符号函数$\text{csgn}(z)$，选取围绕正实轴和负实轴的积分路径：
$$\mcsgn(\boldsymbol{M}) = \frac{1}{2\pi i}\left[\oint_{\Gamma_+} (z\boldsymbol{I} - \boldsymbol{M})^{-1}\,dz - \oint_{\Gamma_-} (z\boldsymbol{I} - \boldsymbol{M})^{-1}\,dz\right]$$

#### Dunford-Taylor积分

更一般地，对于不连续函数，可以使用Dunford-Taylor积分：
$$f(\boldsymbol{M}) = \sum_{k=1}^p \frac{1}{2\pi i}\oint_{\Gamma_k} f(z)(z\boldsymbol{I} - \boldsymbol{M})^{-1}\,dz$$

其中$\Gamma_k$包含第$k$组特征值。

对于符号函数：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{P}_+ - \boldsymbol{P}_- = \frac{1}{2\pi i}\oint_{\Gamma_+} (z\boldsymbol{I} - \boldsymbol{M})^{-1}\,dz - \frac{1}{2\pi i}\oint_{\Gamma_-} (z\boldsymbol{I} - \boldsymbol{M})^{-1}\,dz$$

这给出了符号函数的另一种积分表示。

### Sylvester方程的详细求解

#### Sylvester方程的标准形式

Sylvester方程的标准形式为：
$$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}$$

或等价地：
$$\boldsymbol{A}\boldsymbol{X} - \boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}$$

当$\boldsymbol{A}$和$-\boldsymbol{B}$没有公共特征值时，方程有唯一解。

#### 使用mcsgn求解的推导

考虑分块矩阵：
$$\boldsymbol{M} = \begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{0} & \boldsymbol{B}\end{bmatrix}$$

假设$\boldsymbol{A}$的特征值实部为负，$\boldsymbol{B}$的特征值实部为正。

计算$\mcsgn(\boldsymbol{M})$。根据分块三角矩阵的性质，$\boldsymbol{M}$的特征值为$\boldsymbol{A}$和$\boldsymbol{B}$的特征值的并集。

由于分块三角结构：
$$\mcsgn(\boldsymbol{M}) = \begin{bmatrix}\mcsgn(\boldsymbol{A}) & \boldsymbol{Y} \\ \boldsymbol{0} & \mcsgn(\boldsymbol{B})\end{bmatrix} = \begin{bmatrix}-\boldsymbol{I} & \boldsymbol{Y} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}$$

其中$\boldsymbol{Y}$待定。

利用$\mcsgn(\boldsymbol{M})^2 = \boldsymbol{I}$：
$$\begin{bmatrix}-\boldsymbol{I} & \boldsymbol{Y} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^2 = \begin{bmatrix}\boldsymbol{I} & -\boldsymbol{Y} + \boldsymbol{Y} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix} = \begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}$$

这自动满足。现在需要从$\mcsgn(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}$确定$\boldsymbol{Y}$。

计算$\boldsymbol{M}^2$：
$$\boldsymbol{M}^2 = \begin{bmatrix}\boldsymbol{A}^2 & \boldsymbol{A}\boldsymbol{C} + \boldsymbol{C}\boldsymbol{B} \\ \boldsymbol{0} & \boldsymbol{B}^2\end{bmatrix}$$

对于分块三角矩阵的矩阵函数：
$$(\boldsymbol{M}^2)^{-1/2} = \begin{bmatrix}(\boldsymbol{A}^2)^{-1/2} & \boldsymbol{Z} \\ \boldsymbol{0} & (\boldsymbol{B}^2)^{-1/2}\end{bmatrix}$$

其中$\boldsymbol{Z}$满足Sylvester方程：
$$(\boldsymbol{A}^2)^{-1/2}\boldsymbol{Z} + \boldsymbol{Z}(\boldsymbol{B}^2)^{-1/2} = -(\boldsymbol{A}^2)^{-1/2}(\boldsymbol{A}\boldsymbol{C} + \boldsymbol{C}\boldsymbol{B})(\boldsymbol{B}^2)^{-1/2}$$

简化后：
$$\boldsymbol{A}^{-1}\boldsymbol{Z} + \boldsymbol{Z}\boldsymbol{B}^{-1} = -\boldsymbol{A}^{-1}(\boldsymbol{A}\boldsymbol{C} + \boldsymbol{C}\boldsymbol{B})\boldsymbol{B}^{-1} = -\boldsymbol{C}\boldsymbol{B}^{-1} - \boldsymbol{A}^{-1}\boldsymbol{C}$$

最终，通过矩阵乘法：
$$\mcsgn(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}$$

比较分块$(1,2)$位置得到：
$$\boldsymbol{Y} = 2\boldsymbol{X}$$

其中$\boldsymbol{X}$是Sylvester方程$\boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} = \boldsymbol{C}$的解。

#### 显式公式

由上述推导，我们得到：
$$\boldsymbol{X} = \frac{1}{2}\left[\mcsgn\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{0} & \boldsymbol{B}\end{bmatrix}\right]_{12}$$

其中下标$_{12}$表示取分块矩阵的$(1,2)$块。

### Lyapunov方程的求解

#### Lyapunov方程的定义

Lyapunov方程是Sylvester方程的特殊情况：
$$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{A}^{\top} = \boldsymbol{C}$$

或稳定性形式：
$$\boldsymbol{A}^{\top}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{A} = -\boldsymbol{Q}$$

其中$\boldsymbol{Q}$是正定矩阵。

#### 使用mcsgn求解

构造分块矩阵：
$$\boldsymbol{M} = \begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{0} & -\boldsymbol{A}^{\top}\end{bmatrix}$$

注意$\boldsymbol{M}$的特征值关于原点对称（如果$\lambda$是$\boldsymbol{A}$的特征值，则$-\lambda$是$-\boldsymbol{A}^{\top}$的特征值）。

计算：
$$\mcsgn(\boldsymbol{M}) = \begin{bmatrix}-\boldsymbol{I} & 2\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}$$

其中$\boldsymbol{X}$满足：
$$\boldsymbol{X}(-\boldsymbol{A}^{\top}) - \boldsymbol{A}\boldsymbol{X} = \boldsymbol{C}$$

即：
$$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{A}^{\top} = \boldsymbol{C}$$

这正是Lyapunov方程。

#### 稳定性分析的应用

在控制理论中，Lyapunov方程用于稳定性分析。给定系统$\dot{\boldsymbol{x}} = \boldsymbol{A}\boldsymbol{x}$，如果存在正定矩阵$\boldsymbol{X}$满足：
$$\boldsymbol{A}^{\top}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{A} = -\boldsymbol{Q}$$

其中$\boldsymbol{Q}$正定，则系统渐近稳定。

使用$\mcsgn$可以快速数值求解这个方程，从而判断系统稳定性。

### 代数Riccati方程的深入分析

#### Riccati方程的一般形式

代数Riccati方程的一般形式为：
$$\boldsymbol{X}\boldsymbol{D}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} - \boldsymbol{C} = \boldsymbol{0}$$

或控制理论中的形式：
$$\boldsymbol{A}^{\top}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{A} - \boldsymbol{X}\boldsymbol{B}\boldsymbol{R}^{-1}\boldsymbol{B}^{\top}\boldsymbol{X} + \boldsymbol{Q} = \boldsymbol{0}$$

#### Hamilton矩阵

定义Hamilton矩阵：
$$\boldsymbol{H} = \begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}\boldsymbol{R}^{-1}\boldsymbol{B}^{\top} \\ -\boldsymbol{Q} & -\boldsymbol{A}^{\top}\end{bmatrix}$$

Hamilton矩阵具有特殊性质：如果$\lambda$是特征值，则$-\lambda$也是特征值。

#### 稳定解的计算

Riccati方程的稳定解（对应于闭环系统稳定）可以通过$\mcsgn(\boldsymbol{H})$计算：

**步骤1**：计算$\mcsgn(\boldsymbol{H})$：
$$\mcsgn(\boldsymbol{H}) = \begin{bmatrix}\boldsymbol{U}_{11} & \boldsymbol{U}_{12} \\ \boldsymbol{U}_{21} & \boldsymbol{U}_{22}\end{bmatrix}$$

**步骤2**：计算投影矩阵：
$$\boldsymbol{P}_- = \frac{\boldsymbol{I} - \mcsgn(\boldsymbol{H})}{2}$$

**步骤3**：提取不变子空间。$\boldsymbol{P}_-$的列空间对应于负特征值的不变子空间。设：
$$\boldsymbol{P}_- = \begin{bmatrix}\boldsymbol{V}_1 \\ \boldsymbol{V}_2\end{bmatrix}$$

**步骤4**：计算解：
$$\boldsymbol{X} = \boldsymbol{V}_2\boldsymbol{V}_1^{-1}$$

这给出了Riccati方程的稳定解。

#### 收敛性定理

**定理7**：设$(\boldsymbol{A}, \boldsymbol{B})$可控，$(\boldsymbol{A}, \boldsymbol{Q}^{1/2})$可观，则上述算法收敛到唯一的正定稳定解。

证明涉及不变子空间理论和Hamiltonian系统的结构性质，这里略去。

### 矩阵平方根的计算

#### 平方根的唯一性

**定理8**：设$\boldsymbol{C}$是正定矩阵，则存在唯一的正定矩阵$\boldsymbol{X}$使得$\boldsymbol{X}^2 = \boldsymbol{C}$。

**证明**：由谱定理，$\boldsymbol{C} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$，其中$\boldsymbol{\Lambda} = \text{diag}(\lambda_1,\ldots,\lambda_n)$，$\lambda_i > 0$。

定义$\boldsymbol{X} = \boldsymbol{Q}\,\text{diag}(\sqrt{\lambda_1},\ldots,\sqrt{\lambda_n})\,\boldsymbol{Q}^{\top}$，则$\boldsymbol{X}^2 = \boldsymbol{C}$且$\boldsymbol{X}$正定。

唯一性：假设存在另一个正定解$\boldsymbol{Y}$，则$\boldsymbol{Y}$可与$\boldsymbol{C}$同时对角化（因为它们可交换），由$\boldsymbol{Y}^2 = \boldsymbol{C}$和正定性知$\boldsymbol{Y} = \boldsymbol{X}$。$\square$

#### 使用mcsgn计算

由前面的结果：
$$\mcsgn\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix} = \begin{bmatrix}\boldsymbol{0} & \boldsymbol{C}^{1/2} \\ \boldsymbol{C}^{-1/2} & \boldsymbol{0}\end{bmatrix}$$

因此只需计算一次$\mcsgn$，就能同时得到$\boldsymbol{C}^{1/2}$和$\boldsymbol{C}^{-1/2}$。

#### 算法实现

```
输入：正定矩阵C
输出：C^{1/2}和C^{-1/2}

1. 构造M = [0, C; I, 0]
2. 计算S = mcsgn(M)使用Newton-Schulz迭代
3. 提取S_{12} = C^{1/2}，S_{21} = C^{-1/2}
4. 返回S_{12}，S_{21}
```

时间复杂度为$O(n^3\log(1/\epsilon))$，其中$\epsilon$是精度要求。

### 数值稳定性分析

#### 条件数分析

矩阵符号函数的条件数定义为：
$$\kappa_{\text{sgn}}(\boldsymbol{M}) = \lim_{\epsilon\to 0}\sup_{\|\boldsymbol{E}\|\leq\epsilon} \frac{\|\mcsgn(\boldsymbol{M}+\boldsymbol{E}) - \mcsgn(\boldsymbol{M})\|}{\epsilon}$$

**定理9**：设$\boldsymbol{M}$的特征值$\lambda_i$满足$|\text{Re}(\lambda_i)| \geq \delta > 0$，则：
$$\kappa_{\text{sgn}}(\boldsymbol{M}) \leq \frac{C}{\delta}$$

其中$C$是与$\boldsymbol{M}$的大小有关的常数。

这表明当特征值接近虚轴时，符号函数计算的条件数变差。

#### 预处理技术

为改善条件数，可以使用预处理：

**缩放预处理**：
$$\tilde{\boldsymbol{M}} = \alpha\boldsymbol{M}, \quad \alpha = \frac{1}{\|\boldsymbol{M}\|_2}$$

使得$\|\tilde{\boldsymbol{M}}\|_2 = 1$，改善迭代的收敛性。

**平移预处理**：对于Hamiltonian矩阵，可以使用平移：
$$\tilde{\boldsymbol{H}} = \boldsymbol{H} + \alpha\boldsymbol{I}$$

使得特征值远离虚轴。

### 计算复杂度分析

#### 直接方法的复杂度

使用特征分解计算$\mcsgn(\boldsymbol{M})$：

1. 计算特征分解：$O(n^3)$
2. 处理特征值：$O(n)$
3. 重构矩阵：$O(n^3)$

总复杂度：$O(n^3)$

#### Newton迭代的复杂度

Newton-Schulz迭代：

1. 每次迭代需要$k$次矩阵乘法：$O(kn^3)$
2. 迭代次数：$O(\log(1/\epsilon))$

总复杂度：$O(kn^3\log(1/\epsilon))$

对于三阶迭代（$k=2$），当$\epsilon=10^{-16}$时，通常需要5-6次迭代。

#### 稀疏矩阵的优化

对于稀疏矩阵（非零元素数为$s$），可以使用Krylov子空间方法：

复杂度降为$O(sm\log(1/\epsilon))$，其中$m$是Krylov子空间维数。

### 应用实例：神经网络中的归一化

#### Attention机制中的矩阵归一化

在Transformer的Attention机制中，需要计算：
$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}$$

使用$\msign$可以替代softmax进行硬归一化：
$$\text{HardAttention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \frac{\msign(\boldsymbol{Q}\boldsymbol{K}^{\top}) + \boldsymbol{I}}{2}\boldsymbol{V}$$

#### 权重矩阵的正交化

在深度学习中，保持权重矩阵的正交性有助于梯度流动。使用$\msign$可以快速正交化：
$$\boldsymbol{W}_{\text{orth}} = \msign(\boldsymbol{W})$$

这比QR分解或SVD更高效。

### 总结

矩阵符号函数$\mcsgn$是一个功能强大的工具，它：

1. **理论基础**：基于极分解、谱分解和函数演算，有坚实的数学基础
2. **计算方法**：Newton-Schulz迭代和Padé逼近提供了高效的数值算法
3. **广泛应用**：可用于求解Sylvester方程、Lyapunov方程、Riccati方程，计算矩阵平方根和逆平方根
4. **几何意义**：对应于投影到不变子空间，具有清晰的几何解释
5. **数值性质**：在特征值远离虚轴时具有良好的数值稳定性

本推导从多个角度（线性代数、泛函分析、数值计算、应用数学）全面阐述了矩阵符号函数的理论和应用，为理解和使用这个工具提供了完整的数学基础。

