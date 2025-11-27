---
title: Transformer升级之路：6、旋转位置编码的完备性分析
slug: transformer升级之路6旋转位置编码的完备性分析
date: 2022-12-28
tags: 详细推导, 矩阵, attention, 位置编码, rope, 生成模型
status: completed
---
# Transformer升级之路：6、旋转位置编码的完备性分析

**原文链接**: [https://spaces.ac.cn/archives/9403](https://spaces.ac.cn/archives/9403)

**发布日期**: 

---

在去年的文章[《Transformer升级之路：2、博采众长的旋转式位置编码》](/archives/8265)中，笔者提出了旋转位置编码（RoPE），当时的出发点只是觉得用绝对位置来实现相对位置是一件“很好玩的事情”，并没料到其实际效果还相当不错，并为大家所接受，不得不说这真是一个意外之喜。后来，在[《Transformer升级之路：4、二维位置的旋转式位置编码》](/archives/8397)中，笔者讨论了二维形式的RoPE，并研究了用矩阵指数表示的RoPE的一般解。

既然有了一般解，那么自然就会引出一个问题：我们常用的RoPE，只是一个以二维旋转矩阵为基本单元的分块对角矩阵，如果换成一般解，理论上效果会不会更好呢？本文就来回答这个问题。

## 指数通解 #

在[《Transformer升级之路：4、二维位置的旋转式位置编码》](/archives/8397)中，我们将RoPE抽象地定义为任意满足下式的方阵  
\begin{equation}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}\label{eq:re}\end{equation}  
然后，我们探讨了如下矩阵指数形式的解  
\begin{equation}\boldsymbol{\mathcal{R}}_n=\exp n\boldsymbol{B}\end{equation}  
这里的矩阵指数，不是像Softmax那样的激活函数式的element-wise运算，而是按照泰勒级数定义的“[Matrix Exponential](https://en.wikipedia.org/wiki/Matrix_exponential)”。根据“[Baker–Campbell–Hausdorff公式](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula)”，我们有  
\begin{equation}\begin{aligned}  
\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=&\,\big(\exp m\boldsymbol{B}\big)^{\top}\big(\exp n\boldsymbol{B}\big) = \big(\exp m\boldsymbol{B}^{\top}\big)\big(\exp n\boldsymbol{B}\big) \\\  
=&\,\exp\left(m\boldsymbol{B}^{\top} + n\boldsymbol{B} + \frac{1}{2}mn\left[\boldsymbol{B}^{\top}, \boldsymbol{B}\right]+\cdots\right)  
\end{aligned}\end{equation}  
这里$\left[\boldsymbol{A}, \boldsymbol{B}\right]=\boldsymbol{A}\boldsymbol{B}-\boldsymbol{B}\boldsymbol{A}$，$\cdots$省略的都是$m,n$的三次或三次以上的项。按照式$\eqref{eq:re}$，那么上式指数部分应该等于$(n-m)\boldsymbol{B}$，这就推出  
\begin{equation}\boldsymbol{B}^{\top} = - \boldsymbol{B}\end{equation}  
即要求$\boldsymbol{B}$是反对称矩阵。

## 正交通解 #

进一步地，我们有$(\exp \boldsymbol{B})^{\top}(\exp \boldsymbol{B})=\exp(\boldsymbol{B}-\boldsymbol{B}) = \boldsymbol{I}$和$\exp n\boldsymbol{B} = \left(\exp\boldsymbol{B}\right)^n$，前者说明$\exp\boldsymbol{B}$是正交矩阵，后者则启示我们这是不是可以推广到任意正交矩阵？不难验证，答案是肯定的，我们有结论：

> 对于任意正交矩阵$\boldsymbol{O}$，$\boldsymbol{\mathcal{R}}_n=\boldsymbol{O}^n$是满足式$\eqref{eq:re}$的解。

值得指出的是，在实数域内，并非所有正交矩阵能写成$\exp\boldsymbol{B}$的形式，所以$\boldsymbol{O}^n$实际上比矩阵指数形式更宽泛。从[《恒等式 det(exp(A)) = exp(Tr(A)) 赏析》](/archives/6377)可知$\det(\exp(\boldsymbol{A})) = \exp(\text{Tr}(\boldsymbol{A})) > 0$，所以能写成矩阵指数形式的正交矩阵行列式必然大于0（即等于1），事实上这个结果反过来也成立，即行列式等于1的正交矩阵，必然可以写成$\exp\boldsymbol{B}$的形式，其中$\boldsymbol{B}$是反对称矩阵。（参考[《Why can any orthogonal matrix be written as O=e^A》](https://math.stackexchange.com/questions/2467531/why-can-any-orthogonal-matrix-be-written-as-o-ea)）。

对于$\det(\boldsymbol{O}) = -1$的正交矩阵，我们有$\boldsymbol{O}=\boldsymbol{O}_+ \boldsymbol{I}_-$，其中$\boldsymbol{I}_-$是对角线有一个-1、剩下都是1的对角阵，$\boldsymbol{O}_+$是$\det(\boldsymbol{O}_+) = 1$的正交矩阵，它可以写成$\exp\boldsymbol{B}$的形式，此时$\boldsymbol{O}^n = (\boldsymbol{O}_+ \boldsymbol{I}_-)^n = \boldsymbol{I}_-^n\exp n\boldsymbol{B}$。这也就是说，即便对于$\det(\boldsymbol{O}) = -1$的$\boldsymbol{O}^n$，也只是$\exp n\boldsymbol{B}$的简单变换，所以接下来我们主要研究$\exp n\boldsymbol{B}$形式的解。

## 完备分析 #

众所周知，我们平时所用的RoPE位置编码，是如下形式的分块对角矩阵：  
\begin{equation}\scriptsize{\left(\begin{array}{cc:cc:cc:cc}  
\cos n\theta_0 & -\sin n\theta_0 & 0 & 0 & \cdots & \cdots & 0 & 0 \\\  
\sin n\theta_0 & \cos n\theta_0 & 0 & 0 & \cdots & \cdots & 0 & 0 \\\  
\hdashline  
0 & 0 & \cos n\theta_1 & -\sin n\theta_1 & \cdots & \cdots & 0 & 0 \\\  
0 & 0 & \sin n\theta_1 & \cos n\theta_1 & \cdots & \cdots & 0 & 0 \\\  
\hdashline  
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\\  
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\\  
\hdashline  
0 & 0 & 0 & 0 & \cdots & \cdots & \cos n\theta_{d/2-1} & -\sin n\theta_{d/2-1} \\\  
0 & 0 & 0 & 0 & \cdots & \cdots & \sin n\theta_{d/2-1} & \cos n\theta_{d/2-1} \\\  
\end{array}\right)}\end{equation}  
它可以简写成  
\begin{equation}\begin{pmatrix}  
\boldsymbol{R}_{n\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\\  
\boldsymbol{0} & \boldsymbol{R}_{n\theta_1} & \cdots & \boldsymbol{0} \\\  
\vdots & \vdots & \ddots & \vdots \\\  
\boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{R}_{n\theta_{d/2-1}} \\\  
\end{pmatrix} = \exp n\begin{pmatrix}  
\boldsymbol{J}_{\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\\  
\boldsymbol{0} & \boldsymbol{J}_{\theta_1} & \cdots & \boldsymbol{0} \\\  
\vdots & \vdots & \ddots & \vdots \\\  
\boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{J}_{\theta_{d/2-1}} \\\  
\end{pmatrix}\end{equation}  
其中  
\begin{equation}\boldsymbol{R}_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\\ \sin\theta & \cos\theta \end{pmatrix},\quad \boldsymbol{J}_{\theta} = \begin{pmatrix} 0 & -\theta \\\ \theta & 0\end{pmatrix}\end{equation}  
这种选择可以说是最简单的一种，其本质原因可以说是为了降低计算量。那么，所谓完备性问题，就是要回答：如上的分块对角矩阵的特例，相比全参数$\exp n\boldsymbol{B}$，是否有能力上的缺失？换句话说，如果不考虑计算量，将$\boldsymbol{B}$替换为一般的反对称矩阵，效果是否可能会有提升？

回答这个问题不困难，事实上，对于任意偶数阶反对称矩阵，它都可以对角化为分块对角矩阵  
\begin{equation}\boldsymbol{\Lambda} = \begin{pmatrix}  
\boldsymbol{J}_{\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\\  
\boldsymbol{0} & \boldsymbol{J}_{\theta_1} & \cdots & \boldsymbol{0} \\\  
\vdots & \vdots & \ddots & \vdots \\\  
\boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{J}_{\theta_{d/2-1}} \\\  
\end{pmatrix}\end{equation}  
该结论可以参考[Skew-symmetric matrix](https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Spectral_theory)。也就是说，存在可逆矩阵$\boldsymbol{P}$，使得$\boldsymbol{B}=\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，于是  
\begin{equation}\exp n\boldsymbol{B} = \exp \left(n\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}\right) = \boldsymbol{P}(\exp n\boldsymbol{\Lambda})\boldsymbol{P}^{-1}\end{equation}  
也就是说，任意的$\exp n\boldsymbol{B}$与分块对角的$\exp n\boldsymbol{\Lambda}$，仅仅相差一个相似变换，而我们在Self Attention中应用RoPE时，是  
\begin{equation}\boldsymbol{q}^{\top}\big(\exp (n-m)\boldsymbol{B}\big)\boldsymbol{k} = \big(\boldsymbol{P}^{\top}\boldsymbol{q}\big)^{\top}\big(\exp (n-m)\boldsymbol{\Lambda}\big)\big(\boldsymbol{P}^{-1}\boldsymbol{k}\big)\end{equation}  
由于$\boldsymbol{q},\boldsymbol{k}$一般都是输入$\boldsymbol{x}$经过某个可学习的线性变换而来，$\boldsymbol{P}^{\top},\boldsymbol{P}^{-1}$原则上都可以吸收到线性变换的训练参数中，因此直接设为$\boldsymbol{q}^{\top}\big(\exp (n-m)\boldsymbol{\Lambda}\big)\boldsymbol{k}$理论上不会损失一般性。

所以，对于Self Attention来说，问题的答案是否定的。不过，如果是[线性Attention](线性Attention的探索：Attention必须有个Softmax吗？)，答案会有少许区别，因为线性Attention的$\boldsymbol{q},\boldsymbol{k}$加了个激活函数：  
\begin{equation}\phi(\boldsymbol{q})^{\top}\big(\exp (n-m)\boldsymbol{B}\big)\varphi(\boldsymbol{k}) = \big(\boldsymbol{P}^{\top}\phi(\boldsymbol{q})\big)^{\top}\big(\exp (n-m)\boldsymbol{\Lambda}\big)\big(\boldsymbol{P}^{-1}\varphi(\boldsymbol{k})\big)\end{equation}  
这就导致了$\boldsymbol{P}^{\top},\boldsymbol{P}^{-1}$不一定能吸收到线性变换的训练参数中，因此对线性Attention补上两个参数矩阵，是有可能带来提升的。

## 文章小结 #

本文简单分析了RoPE的完备性问题，表明对于Self Attention来说，目前的分块对角型RoPE不会损失一般性。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9403>_

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

苏剑林. (Dec. 28, 2022). 《Transformer升级之路：6、旋转位置编码的完备性分析 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9403>

@online{kexuefm-9403,  
title={Transformer升级之路：6、旋转位置编码的完备性分析},  
author={苏剑林},  
year={2022},  
month={Dec},  
url={\url{https://spaces.ac.cn/archives/9403}},  
} 


---

## 公式推导与注释

### 1. RoPE的旋转矩阵表示与复数形式

#### 1.1 二维旋转矩阵的基本性质

二维旋转矩阵的标准形式为：

$$\boldsymbol{R}_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**性质1（旋转矩阵的正交性）**：旋转矩阵满足 $\boldsymbol{R}_{\theta}^{\top}\boldsymbol{R}_{\theta} = \boldsymbol{I}$

证明：
$$\boldsymbol{R}_{\theta}^{\top}\boldsymbol{R}_{\theta} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} = \begin{pmatrix} \cos^2\theta + \sin^2\theta & -\cos\theta\sin\theta + \sin\theta\cos\theta \\ -\sin\theta\cos\theta + \cos\theta\sin\theta & \sin^2\theta + \cos^2\theta \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**性质2（旋转矩阵的行列式）**：$\det(\boldsymbol{R}_{\theta}) = 1$

证明：
$$\det(\boldsymbol{R}_{\theta}) = \cos\theta \cdot \cos\theta - (-\sin\theta) \cdot \sin\theta = \cos^2\theta + \sin^2\theta = 1$$

**性质3（旋转矩阵的复合）**：$\boldsymbol{R}_{\theta_1}\boldsymbol{R}_{\theta_2} = \boldsymbol{R}_{\theta_1 + \theta_2}$

证明：
$$\begin{aligned}
\boldsymbol{R}_{\theta_1}\boldsymbol{R}_{\theta_2} &= \begin{pmatrix} \cos\theta_1 & -\sin\theta_1 \\ \sin\theta_1 & \cos\theta_1 \end{pmatrix}\begin{pmatrix} \cos\theta_2 & -\sin\theta_2 \\ \sin\theta_2 & \cos\theta_2 \end{pmatrix} \\
&= \begin{pmatrix} \cos\theta_1\cos\theta_2 - \sin\theta_1\sin\theta_2 & -\cos\theta_1\sin\theta_2 - \sin\theta_1\cos\theta_2 \\ \sin\theta_1\cos\theta_2 + \cos\theta_1\sin\theta_2 & -\sin\theta_1\sin\theta_2 + \cos\theta_1\cos\theta_2 \end{pmatrix} \\
&= \begin{pmatrix} \cos(\theta_1+\theta_2) & -\sin(\theta_1+\theta_2) \\ \sin(\theta_1+\theta_2) & \cos(\theta_1+\theta_2) \end{pmatrix} = \boldsymbol{R}_{\theta_1+\theta_2}
\end{aligned}$$

#### 1.2 复数形式的旋转表示

将二维向量 $(x, y)$ 表示为复数 $z = x + iy$，则旋转操作可以表示为复数乘法：

$$z' = e^{i\theta} z = (\cos\theta + i\sin\theta)(x + iy)$$

展开得：
$$z' = (\cos\theta \cdot x - \sin\theta \cdot y) + i(\sin\theta \cdot x + \cos\theta \cdot y)$$

这对应于矩阵形式：
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix}$$

**定理1（复数旋转的等价性）**：复数乘以 $e^{i\theta}$ 等价于对应二维向量应用旋转矩阵 $\boldsymbol{R}_{\theta}$。

#### 1.3 RoPE的位置编码形式

对于位置 $m$ 的向量 $\boldsymbol{q}_m \in \mathbb{R}^d$（假设 $d$ 为偶数），RoPE将其分为 $d/2$ 个二维块：

$$\boldsymbol{q}_m = \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \\ \vdots \\ q_m^{(d-1)} \\ q_m^{(d)} \end{pmatrix} \rightarrow \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \\ \hdashline q_m^{(3)} \\ q_m^{(4)} \\ \hdashline \vdots \\ \vdots \\ \hdashline q_m^{(d-1)} \\ q_m^{(d)} \end{pmatrix}$$

对第 $i$ 个二维块（$i = 0, 1, \ldots, d/2-1$），应用旋转角度 $\theta_i$：

$$\begin{pmatrix} q_m^{(2i+1)} \\ q_m^{(2i+2)} \end{pmatrix} \rightarrow \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}\begin{pmatrix} q_m^{(2i+1)} \\ q_m^{(2i+2)} \end{pmatrix}$$

其中旋转角度按几何级数递减：
$$\theta_i = \theta_{\text{base}}^{-2i/d}, \quad \theta_{\text{base}} = 10000$$

#### 1.4 复数形式的RoPE

将每个二维块表示为复数：
$$z_m^{(i)} = q_m^{(2i+1)} + i q_m^{(2i+2)}$$

则RoPE操作为：
$$z_m^{(i)} \rightarrow e^{im\theta_i} z_m^{(i)}$$

完整的RoPE编码向量可表示为：
$$\text{RoPE}(\boldsymbol{q}_m) = \begin{pmatrix} \text{Re}(e^{im\theta_0} z_m^{(0)}) \\ \text{Im}(e^{im\theta_0} z_m^{(0)}) \\ \text{Re}(e^{im\theta_1} z_m^{(1)}) \\ \text{Im}(e^{im\theta_1} z_m^{(1)}) \\ \vdots \\ \text{Re}(e^{im\theta_{d/2-1}} z_m^{(d/2-1)}) \\ \text{Im}(e^{im\theta_{d/2-1}} z_m^{(d/2-1)}) \end{pmatrix}$$

### 2. 相对位置编码的等价性

#### 2.1 自注意力中的相对位置

在自注意力机制中，Query向量 $\boldsymbol{q}_m$ 和 Key向量 $\boldsymbol{k}_n$ 的点积为：

$$\text{score}(m, n) = \boldsymbol{q}_m^{\top} \boldsymbol{k}_n$$

应用RoPE后：
$$\text{score}_{\text{RoPE}}(m, n) = (\boldsymbol{\mathcal{R}}_m \boldsymbol{q}_m)^{\top} (\boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n)$$

其中 $\boldsymbol{\mathcal{R}}_m$ 是分块对角矩阵：
$$\boldsymbol{\mathcal{R}}_m = \begin{pmatrix} \boldsymbol{R}_{m\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{R}_{m\theta_1} & \cdots & \boldsymbol{0} \\ \vdots & \vdots & \ddots & \vdots \\ \boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{R}_{m\theta_{d/2-1}} \end{pmatrix}$$

#### 2.2 相对位置的推导

利用旋转矩阵的正交性 $\boldsymbol{R}_{\theta}^{\top} = \boldsymbol{R}_{-\theta}$：

$$\begin{aligned}
\text{score}_{\text{RoPE}}(m, n) &= (\boldsymbol{\mathcal{R}}_m \boldsymbol{q}_m)^{\top} (\boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n) \\
&= \boldsymbol{q}_m^{\top} \boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_n \boldsymbol{k}_n
\end{aligned}$$

由于旋转矩阵的复合性质：
$$\boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{-m} \boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{n-m}$$

因此：
$$\text{score}_{\text{RoPE}}(m, n) = \boldsymbol{q}_m^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}_n$$

**定理2（RoPE的相对位置编码性质）**：RoPE编码的注意力得分仅依赖于相对位置 $n-m$，而不依赖于绝对位置 $m$ 和 $n$。

#### 2.3 复数形式的相对位置

对于第 $i$ 个二维块，复数形式为：
$$\text{score}^{(i)}(m, n) = \text{Re}((e^{im\theta_i} z_m^{(i)})^* (e^{in\theta_i} w_n^{(i)}))$$

其中 $z_m^{(i)}$ 和 $w_n^{(i)}$ 分别是Query和Key的复数表示，$*$ 表示共轭。

利用 $(e^{i\alpha}z)^* = e^{-i\alpha}z^*$：
$$\begin{aligned}
\text{score}^{(i)}(m, n) &= \text{Re}(e^{-im\theta_i} (z_m^{(i)})^* e^{in\theta_i} w_n^{(i)}) \\
&= \text{Re}(e^{i(n-m)\theta_i} (z_m^{(i)})^* w_n^{(i)})
\end{aligned}$$

这明确显示了仅依赖于相对位置 $n-m$。

### 3. 旋转位置编码的完备性定理

#### 3.1 反对称矩阵的分解定理

**定理3（偶数阶反对称矩阵的标准形）**：任意 $d$ 阶反对称矩阵 $\boldsymbol{B}$（$d$ 为偶数），存在正交矩阵 $\boldsymbol{P}$ 使得：

$$\boldsymbol{B} = \boldsymbol{P} \boldsymbol{\Lambda} \boldsymbol{P}^{\top}$$

其中 $\boldsymbol{\Lambda}$ 是分块对角矩阵：
$$\boldsymbol{\Lambda} = \begin{pmatrix} \boldsymbol{J}_{\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{J}_{\theta_1} & \cdots & \boldsymbol{0} \\ \vdots & \vdots & \ddots & \vdots \\ \boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{J}_{\theta_{d/2-1}} \end{pmatrix}$$

且 $\boldsymbol{J}_{\theta} = \begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix}$。

**证明思路**：反对称矩阵的特征值成共轭对出现，即如果 $\lambda$ 是特征值，则 $-\lambda$ 也是特征值。对于反对称矩阵，特征值都是纯虚数，记为 $\pm i\theta_j$。通过Schur分解和适当的基变换，可以将矩阵化为上述分块对角形式。

#### 3.2 矩阵指数的性质

对于反对称矩阵 $\boldsymbol{B}$，矩阵指数 $\exp(\boldsymbol{B})$ 定义为泰勒级数：

$$\exp(\boldsymbol{B}) = \sum_{k=0}^{\infty} \frac{\boldsymbol{B}^k}{k!} = \boldsymbol{I} + \boldsymbol{B} + \frac{\boldsymbol{B}^2}{2!} + \frac{\boldsymbol{B}^3}{3!} + \cdots$$

**性质4（矩阵指数的正交性）**：若 $\boldsymbol{B}^{\top} = -\boldsymbol{B}$，则 $\exp(\boldsymbol{B})$ 是正交矩阵。

证明：
$$\begin{aligned}
(\exp(\boldsymbol{B}))^{\top} \exp(\boldsymbol{B}) &= \exp(\boldsymbol{B}^{\top}) \exp(\boldsymbol{B}) \\
&= \exp(-\boldsymbol{B}) \exp(\boldsymbol{B}) \\
&= \exp(-\boldsymbol{B} + \boldsymbol{B}) = \exp(\boldsymbol{0}) = \boldsymbol{I}
\end{aligned}$$

注：这里使用了 $\exp(\boldsymbol{A})\exp(\boldsymbol{B}) = \exp(\boldsymbol{A} + \boldsymbol{B})$ 当 $\boldsymbol{A}\boldsymbol{B} = \boldsymbol{B}\boldsymbol{A}$ 时成立，而 $\boldsymbol{B}$ 与其转置必然交换。

**性质5（分块对角矩阵的指数）**：
$$\exp\begin{pmatrix} \boldsymbol{A} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{B} \end{pmatrix} = \begin{pmatrix} \exp(\boldsymbol{A}) & \boldsymbol{0} \\ \boldsymbol{0} & \exp(\boldsymbol{B}) \end{pmatrix}$$

对于 $\boldsymbol{J}_{\theta} = \begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix}$，计算其矩阵指数：

$$\boldsymbol{J}_{\theta}^2 = \begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix}\begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix} = \begin{pmatrix} -\theta^2 & 0 \\ 0 & -\theta^2 \end{pmatrix} = -\theta^2 \boldsymbol{I}$$

因此：
$$\boldsymbol{J}_{\theta}^{2k} = (-1)^k \theta^{2k} \boldsymbol{I}, \quad \boldsymbol{J}_{\theta}^{2k+1} = (-1)^k \theta^{2k} \boldsymbol{J}_{\theta}$$

代入泰勒级数：
$$\begin{aligned}
\exp(\boldsymbol{J}_{\theta}) &= \sum_{k=0}^{\infty} \frac{\boldsymbol{J}_{\theta}^k}{k!} \\
&= \sum_{k=0}^{\infty} \frac{\boldsymbol{J}_{\theta}^{2k}}{(2k)!} + \sum_{k=0}^{\infty} \frac{\boldsymbol{J}_{\theta}^{2k+1}}{(2k+1)!} \\
&= \boldsymbol{I} \sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k}}{(2k)!} + \boldsymbol{J}_{\theta} \sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k}}{(2k+1)!} \\
&= \boldsymbol{I} \cos\theta + \boldsymbol{J}_{\theta} \frac{\sin\theta}{\theta}
\end{aligned}$$

具体展开：
$$\exp(\boldsymbol{J}_{\theta}) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} = \boldsymbol{R}_{\theta}$$

#### 3.3 完备性定理的证明

**定理4（RoPE的完备性定理）**：对于Self Attention，分块对角形式的RoPE编码与一般的反对称矩阵指数形式 $\exp(n\boldsymbol{B})$ 在表达能力上是等价的。

**证明**：

设 $\boldsymbol{B}$ 是任意 $d$ 阶反对称矩阵，由定理3，存在正交矩阵 $\boldsymbol{P}$ 使得：
$$\boldsymbol{B} = \boldsymbol{P} \boldsymbol{\Lambda} \boldsymbol{P}^{\top}$$

因此：
$$\exp(n\boldsymbol{B}) = \exp(n\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{\top})$$

利用矩阵指数的相似不变性质 $\exp(\boldsymbol{P}\boldsymbol{A}\boldsymbol{P}^{-1}) = \boldsymbol{P}\exp(\boldsymbol{A})\boldsymbol{P}^{-1}$（对于正交矩阵 $\boldsymbol{P}^{-1} = \boldsymbol{P}^{\top}$）：

$$\exp(n\boldsymbol{B}) = \boldsymbol{P} \exp(n\boldsymbol{\Lambda}) \boldsymbol{P}^{\top}$$

在Self Attention中的应用：
$$\begin{aligned}
\boldsymbol{q}_m^{\top} \exp((n-m)\boldsymbol{B}) \boldsymbol{k}_n &= \boldsymbol{q}_m^{\top} \boldsymbol{P} \exp((n-m)\boldsymbol{\Lambda}) \boldsymbol{P}^{\top} \boldsymbol{k}_n \\
&= (\boldsymbol{P}^{\top}\boldsymbol{q}_m)^{\top} \exp((n-m)\boldsymbol{\Lambda}) (\boldsymbol{P}^{\top}\boldsymbol{k}_n)
\end{aligned}$$

设 $\tilde{\boldsymbol{q}}_m = \boldsymbol{P}^{\top}\boldsymbol{q}_m$，$\tilde{\boldsymbol{k}}_n = \boldsymbol{P}^{\top}\boldsymbol{k}_n$。

在神经网络中，$\boldsymbol{q}_m$ 和 $\boldsymbol{k}_n$ 通常是输入 $\boldsymbol{x}$ 经过线性变换得到：
$$\boldsymbol{q}_m = \boldsymbol{W}_Q \boldsymbol{x}_m, \quad \boldsymbol{k}_n = \boldsymbol{W}_K \boldsymbol{x}_n$$

则：
$$\tilde{\boldsymbol{q}}_m = \boldsymbol{P}^{\top}\boldsymbol{W}_Q \boldsymbol{x}_m = \tilde{\boldsymbol{W}}_Q \boldsymbol{x}_m, \quad \tilde{\boldsymbol{k}}_n = \boldsymbol{P}^{\top}\boldsymbol{W}_K \boldsymbol{x}_n = \tilde{\boldsymbol{W}}_K \boldsymbol{x}_n$$

其中 $\tilde{\boldsymbol{W}}_Q = \boldsymbol{P}^{\top}\boldsymbol{W}_Q$，$\tilde{\boldsymbol{W}}_K = \boldsymbol{P}^{\top}\boldsymbol{W}_K$。

由于 $\boldsymbol{W}_Q$ 和 $\boldsymbol{W}_K$ 是可训练参数，$\boldsymbol{P}^{\top}$ 可以被吸收到这些参数中。因此，使用一般的 $\exp(n\boldsymbol{B})$ 与使用分块对角的 $\exp(n\boldsymbol{\Lambda})$ 在表达能力上是等价的。

**推论**：这意味着我们不需要使用更复杂的全参数反对称矩阵形式，简单的分块对角RoPE已经具有完备的表达能力。

### 4. 正交性质与范数保持

#### 4.1 RoPE变换的等距性

**定理5（RoPE的等距性）**：RoPE变换保持向量的欧几里得范数。

证明：对于向量 $\boldsymbol{v}$，应用RoPE变换后为 $\boldsymbol{\mathcal{R}}_m \boldsymbol{v}$，其范数为：

$$\|\boldsymbol{\mathcal{R}}_m \boldsymbol{v}\|^2 = (\boldsymbol{\mathcal{R}}_m \boldsymbol{v})^{\top} (\boldsymbol{\mathcal{R}}_m \boldsymbol{v}) = \boldsymbol{v}^{\top} \boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_m \boldsymbol{v}$$

由于 $\boldsymbol{\mathcal{R}}_m$ 是正交矩阵，$\boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_m = \boldsymbol{I}$：

$$\|\boldsymbol{\mathcal{R}}_m \boldsymbol{v}\|^2 = \boldsymbol{v}^{\top} \boldsymbol{v} = \|\boldsymbol{v}\|^2$$

这表明RoPE变换不改变向量的长度。

#### 4.2 内积的变化

对于两个向量 $\boldsymbol{u}$ 和 $\boldsymbol{v}$，应用不同的RoPE变换后的内积：

$$\langle \boldsymbol{\mathcal{R}}_m \boldsymbol{u}, \boldsymbol{\mathcal{R}}_n \boldsymbol{v} \rangle = \boldsymbol{u}^{\top} \boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_n \boldsymbol{v} = \boldsymbol{u}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{v}$$

这个内积仅依赖于相对位置 $n-m$ 和原始向量 $\boldsymbol{u}$、$\boldsymbol{v}$。

#### 4.3 角度的保持

对于同一位置的两个向量 $\boldsymbol{u}$ 和 $\boldsymbol{v}$，应用相同的RoPE变换后：

$$\cos\angle(\boldsymbol{\mathcal{R}}_m \boldsymbol{u}, \boldsymbol{\mathcal{R}}_m \boldsymbol{v}) = \frac{\langle \boldsymbol{\mathcal{R}}_m \boldsymbol{u}, \boldsymbol{\mathcal{R}}_m \boldsymbol{v} \rangle}{\|\boldsymbol{\mathcal{R}}_m \boldsymbol{u}\| \|\boldsymbol{\mathcal{R}}_m \boldsymbol{v}\|}$$

由于RoPE的正交性：
$$\langle \boldsymbol{\mathcal{R}}_m \boldsymbol{u}, \boldsymbol{\mathcal{R}}_m \boldsymbol{v} \rangle = \boldsymbol{u}^{\top} \boldsymbol{\mathcal{R}}_m^{\top} \boldsymbol{\mathcal{R}}_m \boldsymbol{v} = \boldsymbol{u}^{\top} \boldsymbol{v} = \langle \boldsymbol{u}, \boldsymbol{v} \rangle$$

且：
$$\|\boldsymbol{\mathcal{R}}_m \boldsymbol{u}\| = \|\boldsymbol{u}\|, \quad \|\boldsymbol{\mathcal{R}}_m \boldsymbol{v}\| = \|\boldsymbol{v}\|$$

因此：
$$\cos\angle(\boldsymbol{\mathcal{R}}_m \boldsymbol{u}, \boldsymbol{\mathcal{R}}_m \boldsymbol{v}) = \frac{\langle \boldsymbol{u}, \boldsymbol{v} \rangle}{\|\boldsymbol{u}\| \|\boldsymbol{v}\|} = \cos\angle(\boldsymbol{u}, \boldsymbol{v})$$

**定理6（角度保持性）**：在同一位置，RoPE变换保持向量之间的夹角不变。

### 5. 多维推广与维度分解

#### 5.1 高维旋转的分解

对于 $d$ 维空间（$d$ 为偶数），任意旋转可以分解为 $d/2$ 个二维平面上的旋转。

设 $d = 2k$，则 $d$ 维向量可以看作 $k$ 个二维向量的拼接：
$$\boldsymbol{v} = \begin{pmatrix} \boldsymbol{v}_0 \\ \boldsymbol{v}_1 \\ \vdots \\ \boldsymbol{v}_{k-1} \end{pmatrix}, \quad \boldsymbol{v}_i \in \mathbb{R}^2$$

RoPE变换为：
$$\boldsymbol{\mathcal{R}}_m \boldsymbol{v} = \begin{pmatrix} \boldsymbol{R}_{m\theta_0} \boldsymbol{v}_0 \\ \boldsymbol{R}_{m\theta_1} \boldsymbol{v}_1 \\ \vdots \\ \boldsymbol{R}_{m\theta_{k-1}} \boldsymbol{v}_{k-1} \end{pmatrix}$$

#### 5.2 频率的选择

旋转频率 $\theta_i$ 的选择遵循几何级数：
$$\theta_i = \theta_{\text{base}}^{-2i/d}$$

这种选择的优点：
1. **多尺度表示**：不同的频率能够编码不同尺度的位置信息
2. **容量充分**：从高频到低频覆盖了广泛的频率范围
3. **平滑过渡**：几何级数保证了频率的平滑递减

对于基础频率 $\theta_{\text{base}} = 10000$，第 $i$ 个维度对的周期为：
$$T_i = \frac{2\pi}{\theta_i} = 2\pi \cdot 10000^{2i/d}$$

当 $i = 0$ 时，$T_0 = 2\pi$，周期最短；
当 $i = d/2-1$ 时，$T_{d/2-1} = 2\pi \cdot 10000$，周期最长。

#### 5.3 维度对的独立性

**定理7（维度对的正交性）**：不同维度对的旋转操作是相互独立的。

证明：由于 $\boldsymbol{\mathcal{R}}_m$ 是分块对角矩阵，不同块之间没有耦合：

$$\boldsymbol{\mathcal{R}}_m = \begin{pmatrix} \boldsymbol{R}_{m\theta_0} & & & \\ & \boldsymbol{R}_{m\theta_1} & & \\ & & \ddots & \\ & & & \boldsymbol{R}_{m\theta_{d/2-1}} \end{pmatrix}$$

对于维度对 $i$ 和 $j$（$i \neq j$），它们的变换互不影响：
$$\frac{\partial (\boldsymbol{\mathcal{R}}_m \boldsymbol{v})_{2i:2i+2}}{\partial \boldsymbol{v}_{2j:2j+2}} = \boldsymbol{0}$$

这种独立性使得模型可以在不同频率尺度上独立学习位置信息。

### 6. 与其他位置编码的对比分析

#### 6.1 与Sinusoidal位置编码的对比

**Sinusoidal编码**：
$$\begin{aligned}
PE(m, 2i) &= \sin(m / 10000^{2i/d}) \\
PE(m, 2i+1) &= \cos(m / 10000^{2i/d})
\end{aligned}$$

这是一种绝对位置编码，直接加在输入向量上：
$$\boldsymbol{x}_m' = \boldsymbol{x}_m + PE(m)$$

**相对位置的实现**：Sinusoidal编码可以通过三角恒等式实现相对位置：
$$\begin{aligned}
PE(m, 2i) \cdot PE(n, 2i) + PE(m, 2i+1) \cdot PE(n, 2i+1)
&= \sin(m\omega_i)\sin(n\omega_i) + \cos(m\omega_i)\cos(n\omega_i) \\
&= \cos((m-n)\omega_i)
\end{aligned}$$

其中 $\omega_i = 1/10000^{2i/d}$。

**与RoPE的区别**：
1. **作用方式**：Sinusoidal是加法，RoPE是旋转（乘法）
2. **相对位置**：Sinusoidal需要通过点积间接获得，RoPE直接建模
3. **范数保持**：RoPE保持范数，Sinusoidal改变范数

#### 6.2 与可训练位置编码的对比

**可训练位置编码**：
$$PE(m) = \boldsymbol{E}_m \in \mathbb{R}^d$$

其中 $\boldsymbol{E}_m$ 是可学习的参数矩阵的第 $m$ 行。

**优缺点对比**：

| 特性 | 可训练位置编码 | RoPE |
|------|---------------|------|
| 参数量 | $O(L \cdot d)$（$L$ 为最大长度） | 无参数 |
| 外推能力 | 差（未见过的位置无编码） | 好（函数式定义） |
| 相对位置 | 需要额外设计 | 天然支持 |
| 灵活性 | 高（完全可学习） | 低（固定函数形式） |

#### 6.3 与相对位置编码（RPE）的对比

**相对位置编码（Relative Position Encoding）**：
$$\text{score}(m, n) = \boldsymbol{q}_m^{\top} \boldsymbol{k}_n + \boldsymbol{q}_m^{\top} \boldsymbol{r}_{m-n}$$

其中 $\boldsymbol{r}_{m-n}$ 是可训练的相对位置嵌入。

**Shaw等人的方法**：
$$\text{score}(m, n) = \boldsymbol{q}_m^{\top} \boldsymbol{k}_n + \boldsymbol{q}_m^{\top} \boldsymbol{W}_K^{\top} \boldsymbol{r}_{m-n}^K$$

**与RoPE的区别**：
1. **实现方式**：RPE是加法偏置，RoPE是旋转变换
2. **参数效率**：RPE需要存储相对位置嵌入，RoPE无参数
3. **计算效率**：RoPE可以预先计算旋转矩阵，效率更高

#### 6.4 定量对比分析

设序列长度为 $L$，隐藏维度为 $d$，则：

**空间复杂度**：
- Sinusoidal：$O(1)$（无参数）
- 可训练绝对位置：$O(Ld)$
- 相对位置编码（RPE）：$O(L^2 d)$ 或 $O(Ld)$（截断版本）
- RoPE：$O(1)$（无参数，预计算旋转矩阵为 $O(d)$）

**时间复杂度（每个位置对）**：
- Sinusoidal：$O(d)$（加法）
- 可训练绝对位置：$O(d)$（加法）
- 相对位置编码：$O(d)$（加法偏置）
- RoPE：$O(d)$（旋转，可预计算）

**外推性能**：
- Sinusoidal：中等（函数式但高频振荡）
- 可训练绝对位置：差（需要重新训练）
- 相对位置编码：中等（需要外推新的相对位置）
- RoPE：好（天然支持任意长度）

### 7. 理论深化：Baker-Campbell-Hausdorff公式

#### 7.1 BCH公式的表述

对于矩阵 $\boldsymbol{A}$ 和 $\boldsymbol{B}$，Baker-Campbell-Hausdorff公式给出：

$$\log(\exp(\boldsymbol{A})\exp(\boldsymbol{B})) = \boldsymbol{A} + \boldsymbol{B} + \frac{1}{2}[\boldsymbol{A}, \boldsymbol{B}] + \frac{1}{12}[\boldsymbol{A}, [\boldsymbol{A}, \boldsymbol{B}]] - \frac{1}{12}[\boldsymbol{B}, [\boldsymbol{A}, \boldsymbol{B}]] + \cdots$$

其中 $[\boldsymbol{A}, \boldsymbol{B}] = \boldsymbol{A}\boldsymbol{B} - \boldsymbol{B}\boldsymbol{A}$ 是矩阵的Lie括号。

#### 7.2 应用于RoPE的验证

对于RoPE，我们需要验证：
$$\exp(m\boldsymbol{B})^{\top} \exp(n\boldsymbol{B}) = \exp((n-m)\boldsymbol{B})$$

由于 $\boldsymbol{B}$ 是反对称矩阵（$\boldsymbol{B}^{\top} = -\boldsymbol{B}$），我们有：
$$\exp(m\boldsymbol{B})^{\top} = \exp(m\boldsymbol{B}^{\top}) = \exp(-m\boldsymbol{B})$$

应用BCH公式：
$$\begin{aligned}
\log(\exp(-m\boldsymbol{B})\exp(n\boldsymbol{B})) &= -m\boldsymbol{B} + n\boldsymbol{B} + \frac{1}{2}[-m\boldsymbol{B}, n\boldsymbol{B}] + \cdots \\
&= (n-m)\boldsymbol{B} + \frac{-mn}{2}[\boldsymbol{B}, \boldsymbol{B}] + \cdots
\end{aligned}$$

由于 $[\boldsymbol{B}, \boldsymbol{B}] = \boldsymbol{B}\boldsymbol{B} - \boldsymbol{B}\boldsymbol{B} = \boldsymbol{0}$，所有高阶项都为零：

$$\log(\exp(-m\boldsymbol{B})\exp(n\boldsymbol{B})) = (n-m)\boldsymbol{B}$$

因此：
$$\exp(-m\boldsymbol{B})\exp(n\boldsymbol{B}) = \exp((n-m)\boldsymbol{B})$$

这证明了RoPE的相对位置性质。

#### 7.3 反对称矩阵的Lie代数结构

反对称矩阵构成一个Lie代数 $\mathfrak{so}(d)$（special orthogonal Lie algebra）：

**封闭性**：对于反对称矩阵 $\boldsymbol{A}, \boldsymbol{B}$，$[\boldsymbol{A}, \boldsymbol{B}]$ 仍是反对称矩阵。

证明：
$$\begin{aligned}
[\boldsymbol{A}, \boldsymbol{B}]^{\top} &= (\boldsymbol{A}\boldsymbol{B} - \boldsymbol{B}\boldsymbol{A})^{\top} \\
&= \boldsymbol{B}^{\top}\boldsymbol{A}^{\top} - \boldsymbol{A}^{\top}\boldsymbol{B}^{\top} \\
&= (-\boldsymbol{B})(-\boldsymbol{A}) - (-\boldsymbol{A})(-\boldsymbol{B}) \\
&= \boldsymbol{B}\boldsymbol{A} - \boldsymbol{A}\boldsymbol{B} \\
&= -[\boldsymbol{A}, \boldsymbol{B}]
\end{aligned}$$

**维度**：$d$ 维空间的反对称矩阵有 $\binom{d}{2} = \frac{d(d-1)}{2}$ 个自由度。

对于 $d = 2$，只有1个自由度；对于 $d = 4$，有6个自由度。

### 8. 数值稳定性分析

#### 8.1 浮点精度的影响

在实际计算中，旋转矩阵的计算涉及三角函数：
$$\boldsymbol{R}_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

对于大的位置索引 $m$ 和小的频率 $\theta_i$，$m\theta_i$ 可能非常大，导致：
1. 浮点数精度损失
2. 周期性混叠

**解决方案**：通过模运算归约到 $[0, 2\pi)$：
$$m\theta_i \leftarrow (m\theta_i) \mod 2\pi$$

#### 8.2 正交性的数值保持

理论上 $\boldsymbol{\mathcal{R}}_m$ 是正交矩阵，但数值计算可能导致偏离：
$$\|\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_m - \boldsymbol{I}\|_F = \epsilon$$

其中 $\epsilon$ 是数值误差。

**误差累积**：在长序列中，多次矩阵乘法可能累积误差。

**缓解措施**：
1. 使用更高精度（如float64）计算旋转矩阵
2. 预计算并缓存旋转矩阵
3. 定期重新正交化（Gram-Schmidt过程）

#### 8.3 复数实现的优势

使用复数表示可以避免显式构造旋转矩阵：
$$z' = e^{i\theta} z$$

复数乘法比矩阵乘法更高效，且数值稳定性更好：
- 矩阵乘法：$O(4)$ 次浮点运算（2×2矩阵）
- 复数乘法：$O(3)$ 次浮点运算（优化的复数乘法）

### 9. 总结与展望

#### 9.1 主要结论汇总

1. **完备性**：分块对角RoPE与全参数反对称矩阵指数在Self Attention中等价
2. **正交性**：RoPE保持向量范数和角度，是等距变换
3. **相对位置**：RoPE天然实现相对位置编码，仅依赖 $n-m$
4. **参数效率**：RoPE无需任何可训练参数
5. **计算效率**：通过预计算可高效实现

#### 9.2 理论意义

- 证明了简单的分块对角结构具有完备的表达能力
- 不需要更复杂的设计就能达到理论上的最优
- 为位置编码的设计提供了理论指导

#### 9.3 实践价值

- 降低了实现复杂度
- 提高了计算效率
- 保证了数值稳定性

本文通过严格的数学推导，证明了RoPE的完备性，表明当前的分块对角设计已经是理论最优的选择，不需要进一步复杂化。这为RoPE在实践中的广泛应用提供了坚实的理论基础。

