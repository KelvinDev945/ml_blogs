---
title: GlobalPointer下的“KL散度”应该是怎样的？
slug: globalpointer下的kl散度应该是怎样的
date: 2022-04-15
tags: 损失函数, 对抗训练, NER, 正则化, 生成模型
status: completed
---

# GlobalPointer下的"KL散度"应该是怎样的？

**原文链接**: [https://spaces.ac.cn/archives/9039](https://spaces.ac.cn/archives/9039)

**发布日期**: 2022-04-15

---

<div class="theorem-box">

### 核心问题

**背景**：
- [GlobalPointer](/archives/8373)用于命名实体识别等多标签分类任务
- [R-Drop](/archives/8496)等正则化方法需要计算输出分布的KL散度
- **问题**：GlobalPointer的输出不是标准概率分布，无法直接计算KL散度

**本文贡献**：
- ✅ 提出GlobalPointer场景下的KL散度计算方法
- ✅ 从理论上证明其与二分类KL散度的等价性
- ✅ 实验验证其在R-Drop中的有效性

**核心公式**：
$$
D_{GP}(\mathbf{s}, \mathbf{t}) = \sum_i (\sigma(s_i) - \sigma(t_i))(s_i - t_i)
$$

**关键洞察**：
- Softmax：将argmax光滑化（单标签分类）
- **Sigmoid：将threshold光滑化（多标签分类）**

</div>

---

最近有读者提到想测试一下[GlobalPointer](/archives/8373)与[R-Drop](/archives/8496)结合的效果，但不知道GlobalPointer下的KL散度该怎么算。像R-Drop或者[虚拟对抗训练](/archives/7466)这些正则化手段，里边都需要算概率分布的KL散度，但GlobalPointer的预测结果并非一个概率分布，因此无法直接进行计算。

经过一番尝试，笔者给出了一个可用的形式，并通过简单实验验证了它的可行性，遂在此介绍笔者的分析过程。

## 一、对称散度

KL散度是关于两个概率分布的函数，它是不对称的，即$KL(p\Vert q)$通常不等于$KL(q\Vert p)$，在实际应用中，我们通常使用对称化的KL散度：
\begin{equation}D(p,q) = KL(p\Vert q) + KL(q\Vert p)\tag{1}\end{equation}
代入KL散度的定义$KL(p\Vert q)=\sum\limits_i p_i\log\frac{p_i}{q_i}$，可以化简得到
\begin{equation}D(p,q) = \sum_i (p_i - q_i)(\log p_i - \log q_i)\tag{2}\end{equation}

<div class="derivation-box">

### 从概率到Logits

考虑到$p,q$通常由softmax得到，我们定义
\begin{equation}p_i = \frac{e^{s_i}}{\sum\limits_j e^{s_j}},\quad q_i = \frac{e^{t_i}}{\sum\limits_j e^{t_j}}\tag{3}\end{equation}

**关键推导**：代入后得到
\begin{equation}\begin{aligned}
D(p,q) =&\, \sum_i (p_i - q_i)(s_i - t_i) + \sum_i (p_i - q_i)\left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right) \\
=&\, \sum_i (p_i - q_i)(s_i - t_i) + \left(\sum_i p_i - \sum_i q_i\right)\left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right) \\
=&\, \sum_i (p_i - q_i)(s_i - t_i)
\end{aligned}\label{eq:kl-0}\tag{4}\end{equation}

**关键观察**：第二项消失是因为$\sum_i p_i = \sum_i q_i = 1$（概率归一化）

**简洁形式**：
$$
D(p,q) = \langle f(s) - f(t), s - t \rangle
$$
其中$f = \text{softmax}$，$\langle \cdot, \cdot \rangle$表示内积

</div>

## 二、类比推广到GlobalPointer

可以看到，从logits层面看，对称KL散度具有以下的形式
\begin{equation}D(s, t) = \sum_i (f(s_i) - f(t_i))(s_i - t_i) = \langle f(s) - f(t), s -t \rangle\label{eq:kl}\tag{5}\end{equation}
其中$f$是softmax操作，$\langle\cdot,\cdot\rangle$表示向量的内积。从形式上来看，它是两个向量的内积，其中一个向量是logits的差，第二个向量则是logits经过$f$变换后的差。

<div class="intuition-box">

### 🧠 Softmax的本质

**关键洞察**：Softmax是什么的光滑近似？

变换$f$有什么特点呢？我们知道，softmax实际上是$\text{onehot}(\text{argmax}(\cdot))$的光滑近似（参考[《函数光滑化杂谈：不可导函数的可导逼近》](/archives/6620)）。

**单标签分类的决策过程**：
```
Logits: [2.1, 1.3, 3.5, 0.9]
         ↓ argmax
Index:   2 (最大值位置)
         ↓ onehot
Output:  [0, 0, 1, 0]
```

**Softmax的光滑版本**：
```
Logits:  [2.1, 1.3, 3.5, 0.9]
          ↓ softmax
Output:  [0.17, 0.08, 0.68, 0.05]
         (接近 [0, 0, 1, 0])
```

**结论**：Softmax = "将最大值置1、其余置0" 的光滑近似

</div>

有了这个抽象视角，我们就可以类比地构建GlobalPointer的"KL散度"了。GlobalPointer的输出也可以理解为是logits，但它所用的损失函数是[《将"Softmax+交叉熵"推广到多标签分类问题》](/archives/7359)提出的多标签交叉熵，因此这本质上是多标签交叉熵中如何算KL散度的问题，最后GlobalPointer输出的目标类别亦并非logits最大的那个类，而是所有logits大于0的类别。

<div class="note-box">

### GlobalPointer vs Softmax分类

**Softmax分类**：
- 决策规则：$\hat{y} = \arg\max_i s_i$
- 目标：让正类的logit成为最大值
- 光滑近似：softmax

**GlobalPointer（多标签）**：
- 决策规则：$\hat{y}_i = \mathbb{1}(s_i > 0)$（threshold at 0）
- 目标：让正类logit > 0，负类logit < 0
- 光滑近似：**sigmoid**

**示例**：
```
Logits:  [1.2, -0.5, 2.1, -1.3, 0.8]
          ↓ threshold at 0
Labels:  [1, 0, 1, 0, 1]  (多标签)

Sigmoid: [0.77, 0.38, 0.89, 0.21, 0.69]
         (接近 [1, 0, 1, 0, 1])
```

</div>

所以，对于GlobalPointer来说，其对称散度可以保留式$\eqref{eq:kl}$的形式，但$f$应该换成"将大于0的置为1、将小于0的置为0"的光滑近似，而sigmoid函数$\sigma(x)=1/(1+e^{-x})$正好是满足这一性质的函数，因此我们可以将GlobalPointer的对称KL散度可以设计为
\begin{equation}D(s, t) = \sum_i (\sigma(s_i) - \sigma(t_i))(s_i - t_i) = \langle \sigma(s) - \sigma(t), s -t \rangle\label{eq:gp-kl}\tag{6}\end{equation}

## 峰回路转 #

有意思的是，笔者事后发现，式$\eqref{eq:gp-kl}$实际上等价于每个logits分别用$\sigma$激活后，各自单独算二元概率的KL散度然后求和。

要证明这一点很简单，留意到$\sigma$函数构建的二元分布$[\sigma(s),1 - \sigma(s)]$，跟用$[s, 0]$为logits加上softmax构建的二元分布是等价的，即$[\sigma(s),1 - \sigma(s)]=softmax([s, 0])$，所以根据公式$\eqref{eq:kl-0}$，我们直接有  
\begin{equation}\begin{aligned}  
&\,D\big([\sigma(s_i),1 - \sigma(s_i)],[\sigma(t_i),1 - \sigma(t_i)]\big) \\\  
=&\,(\sigma(s_i)-\sigma(t_i))(s_i - t_i) + \big((1-\sigma(s_i))-(1-\sigma(t_i))\big)(0 - 0)\\\  
=&\,(\sigma(s_i)-\sigma(t_i))(s_i - t_i)  
\end{aligned}\end{equation}  
将每个分量加起来，就得到式$\eqref{eq:gp-kl}$

这个等价性说明，虽然我们做多标签分类时作为多个二分类问题来做的话会带来类别不平衡问题，但是如果只是用来评估结果连续性的话，就不存在所谓的类别不平衡问题了（因为根本就不是分类），所以此时仍然可以将其看成多个二分类问题，然后算其常规的KL散度。

## 实验结果 #

笔者和网友分别做了简单的对比实验，结果显示用式$\eqref{eq:gp-kl}$作为KL散度，将R-Drop应用到GlobalPointer中，确实能轻微提升效果，而如果对GlobalPointer的logits直接做softmax然后算常规的KL散度，结果反而不好，这就体现了式$\eqref{eq:gp-kl}$的合理性。

但需要指出的是，式$\eqref{eq:gp-kl}$只是提供了一种在GlobalPointer中用R-Drop或者虚拟对抗训练的方案，但具体情况下效果会不会有提升，这是无法保证的，就好比常规的分类问题配合R-Drop也未必能取得效果提升一样。这需要多去实验尝试，尤其是需要精调正则项的权重系数。

## 文末小结 #

本文主要讨论了GlobalPointer下的“KL散度”计算问题，为GlobalPointer应用R-Drop或者虚拟对抗训练等提供一个可用的KL散度形式。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9039>_

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

苏剑林. (Apr. 15, 2022). 《GlobalPointer下的“KL散度”应该是怎样的？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9039>

@online{kexuefm-9039,  
title={GlobalPointer下的“KL散度”应该是怎样的？},  
author={苏剑林},  
year={2022},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9039}},  
} 


---

## 公式推导与注释

### 1. KL散度基础理论

**定义1.1 (KL散度的基本定义)**

对于两个离散概率分布$p$和$q$，Kullback-Leibler散度定义为：
\begin{equation}
KL(p\Vert q) = \sum_i p_i \log\frac{p_i}{q_i}
\tag{1}
\end{equation}

**数学直觉**: KL散度度量了使用分布$q$来近似分布$p$时的信息损失。它具有以下性质：

**性质1.1 (非负性)**

对于任意概率分布$p,q$，有：
\begin{equation}
KL(p\Vert q) \geq 0
\tag{2}
\end{equation}
等号成立当且仅当$p=q$。

**证明**: 利用Jensen不等式。由于$-\log$是凸函数，我们有：
\begin{equation}
\begin{aligned}
-KL(p\Vert q) &= -\sum_i p_i\log\frac{p_i}{q_i} \\
&= \sum_i p_i\log\frac{q_i}{p_i} \\
&\leq \log\sum_i p_i\frac{q_i}{p_i} \\
&= \log\sum_i q_i = \log 1 = 0
\end{aligned}
\tag{3}
\end{equation}

**性质1.2 (非对称性)**

一般情况下，$KL(p\Vert q) \neq KL(q\Vert p)$。这是KL散度的重要特性。

**数值示例**: 考虑$p=[0.9, 0.1]$, $q=[0.5, 0.5]$:
\begin{equation}
\begin{aligned}
KL(p\Vert q) &= 0.9\log\frac{0.9}{0.5} + 0.1\log\frac{0.1}{0.5} \\
&\approx 0.510 \\
KL(q\Vert p) &= 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1} \\
&\approx 0.510
\end{aligned}
\tag{4}
\end{equation}

注意：在这个特殊例子中两者相等是巧合。

### 2. 对称KL散度推导

**定义2.1 (对称KL散度)**

为了消除KL散度的非对称性，我们定义对称KL散度：
\begin{equation}
D(p,q) = KL(p\Vert q) + KL(q\Vert p)
\tag{5}
\end{equation}

**展开推导**:
\begin{equation}
\begin{aligned}
D(p,q) &= \sum_i p_i\log\frac{p_i}{q_i} + \sum_i q_i\log\frac{q_i}{p_i} \\
&= \sum_i \left(p_i\log p_i - p_i\log q_i + q_i\log q_i - q_i\log p_i\right) \\
&= \sum_i \left[(p_i - q_i)\log p_i - (p_i - q_i)\log q_i\right] \\
&= \sum_i (p_i - q_i)(\log p_i - \log q_i)
\end{aligned}
\tag{6}
\end{equation}

**几何直觉**: 对称KL散度可以理解为两个分布之间的"双向距离"，它满足对称性：$D(p,q) = D(q,p)$。

### 3. Softmax参数化下的KL散度

**假设3.1**: 假设$p$和$q$都由softmax函数生成：
\begin{equation}
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \quad q_i = \frac{e^{t_i}}{\sum_j e^{t_j}}
\tag{7}
\end{equation}

其中$s_i$和$t_i$是logits。

**定理3.1 (Softmax下的对称KL散度简化)**

在Softmax参数化下，对称KL散度可以简化为：
\begin{equation}
D(p,q) = \sum_i (p_i - q_i)(s_i - t_i) = \langle p - q, s - t \rangle
\tag{8}
\end{equation}

**详细证明**:

从式(6)出发，代入式(7)：
\begin{equation}
\begin{aligned}
\log p_i &= \log\frac{e^{s_i}}{\sum_j e^{s_j}} = s_i - \log\sum_j e^{s_j} \\
\log q_i &= t_i - \log\sum_j e^{t_j}
\end{aligned}
\tag{9}
\end{equation}

因此：
\begin{equation}
\begin{aligned}
\log p_i - \log q_i &= \left(s_i - \log\sum_j e^{s_j}\right) - \left(t_i - \log\sum_j e^{t_j}\right) \\
&= (s_i - t_i) + \left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right)
\end{aligned}
\tag{10}
\end{equation}

代入式(6)：
\begin{equation}
\begin{aligned}
D(p,q) &= \sum_i (p_i - q_i)\left[(s_i - t_i) + \left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right)\right] \\
&= \sum_i (p_i - q_i)(s_i - t_i) + \left(\sum_i p_i - \sum_i q_i\right)\left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right) \\
&= \sum_i (p_i - q_i)(s_i - t_i) + (1 - 1) \cdot \text{常数} \\
&= \sum_i (p_i - q_i)(s_i - t_i)
\end{aligned}
\tag{11}
\end{equation}

**关键观察**: 第二项消失是因为$\sum_i p_i = \sum_i q_i = 1$（概率归一化条件）。

### 4. 向量形式与几何解释

**定义4.1 (向量内积形式)**

记$\mathbf{p} = (p_1, \ldots, p_n)^T$, $\mathbf{s} = (s_1, \ldots, s_n)^T$，式(8)可写为：
\begin{equation}
D(\mathbf{s}, \mathbf{t}) = \langle f(\mathbf{s}) - f(\mathbf{t}), \mathbf{s} - \mathbf{t} \rangle
\tag{12}
\end{equation}

其中$f = \text{softmax}$是作用于向量的softmax函数。

**几何解释**:

1. **内积结构**: 这是两个向量$(\mathbf{s} - \mathbf{t})$和$(f(\mathbf{s}) - f(\mathbf{t}))$的内积
2. **单调性**: 由于softmax是单调的，当$s_i > t_i$时，通常有$p_i > q_i$，因此内积为正
3. **投影视角**: 可以理解为logits差在概率差方向上的投影

**性质4.1 (正定性)**

对于$\mathbf{s} \neq \mathbf{t}$，有$D(\mathbf{s}, \mathbf{t}) > 0$。

**证明思路**: softmax的严格单调性保证了当$s_i > t_i$时$p_i > q_i$，因此：
\begin{equation}
(p_i - q_i)(s_i - t_i) > 0
\tag{13}
\end{equation}

### 5. Softmax的函数性质分析

**定义5.1 (Softmax作为光滑近似)**

Softmax函数可以理解为$\text{onehot}(\text{argmax}(\cdot))$的光滑近似：
\begin{equation}
\lim_{\beta\to\infty} \text{softmax}(\beta \mathbf{s}) = \text{onehot}(\text{argmax}(\mathbf{s}))
\tag{14}
\end{equation}

**直觉**: 当温度参数趋于0（或等价地，logits被放大）时，softmax输出趋于one-hot向量。

**性质5.1 (Softmax的Jacobian)**

Softmax函数的Jacobian矩阵为：
\begin{equation}
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)
\tag{15}
\end{equation}

其中$\delta_{ij}$是Kronecker delta函数。

**推导**:
\begin{equation}
\begin{aligned}
\frac{\partial}{\partial s_j}\left(\frac{e^{s_i}}{\sum_k e^{s_k}}\right) &= \frac{\delta_{ij}e^{s_i}\sum_k e^{s_k} - e^{s_i}e^{s_j}}{(\sum_k e^{s_k})^2} \\
&= \frac{e^{s_i}}{\sum_k e^{s_k}}\left(\delta_{ij} - \frac{e^{s_j}}{\sum_k e^{s_k}}\right) \\
&= p_i(\delta_{ij} - p_j)
\end{aligned}
\tag{16}
\end{equation}

### 6. GlobalPointer场景分析

**背景6.1**: GlobalPointer用于命名实体识别等任务，其输出是多标签分类：
- 目标不是找到logits最大的单个类
- 而是找到所有logits大于0的类别

**定义6.1 (多标签分类的决策函数)**

GlobalPointer的决策函数为：
\begin{equation}
\hat{y}_i = \begin{cases}
1, & \text{if } s_i > 0 \\
0, & \text{if } s_i \leq 0
\end{cases}
\tag{17}
\end{equation}

**关键差异**: 与softmax+argmax不同：
- Softmax场景: 目标是让正类logit成为最大值
- GlobalPointer场景: 目标是让正类logit大于0，负类logit小于0

### 7. Sigmoid函数的角色

**定义7.1 (Sigmoid函数)**

Sigmoid函数定义为：
\begin{equation}
\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}
\tag{18}
\end{equation}

**性质7.1 (Sigmoid作为光滑近似)**

Sigmoid是"大于0置1，小于0置0"的光滑近似：
\begin{equation}
\lim_{x\to+\infty}\sigma(x) = 1, \quad \lim_{x\to-\infty}\sigma(x) = 0
\tag{19}
\end{equation}

**性质7.2 (Sigmoid的对称性)**

Sigmoid满足：
\begin{equation}
\sigma(-x) = 1 - \sigma(x)
\tag{20}
\end{equation}

**证明**:
\begin{equation}
\sigma(-x) = \frac{1}{1+e^x} = \frac{1+e^{-x}-e^{-x}}{1+e^{-x}} = 1 - \frac{e^{-x}}{1+e^{-x}} = 1 - \sigma(x)
\tag{21}
\end{equation}

### 8. GlobalPointer的KL散度设计

**类比推导**: 根据式(12)的形式，将softmax替换为sigmoid：
\begin{equation}
D_{GP}(\mathbf{s}, \mathbf{t}) = \sum_i (\sigma(s_i) - \sigma(t_i))(s_i - t_i) = \langle \sigma(\mathbf{s}) - \sigma(\mathbf{t}), \mathbf{s} - \mathbf{t} \rangle
\tag{22}
\end{equation}

**直觉**:
- Softmax场景: $f$将最大值置1，其余置0
- GlobalPointer场景: $f$将大于0的置1，小于0的置0
- 因此将softmax替换为sigmoid是自然的

**定理8.1 (GlobalPointer KL散度的等价性)**

式(22)等价于每个logit单独用sigmoid激活后，各自算二元KL散度再求和。

**证明**:

考虑单个分量的二元分布：
\begin{equation}
p_i^{(2)} = [\sigma(s_i), 1-\sigma(s_i)], \quad q_i^{(2)} = [\sigma(t_i), 1-\sigma(t_i)]
\tag{23}
\end{equation}

**引理8.1**: 二元分布$[\sigma(s), 1-\sigma(s)]$等价于用$[s, 0]$作为logits的softmax：
\begin{equation}
[\sigma(s), 1-\sigma(s)] = \text{softmax}([s, 0])
\tag{24}
\end{equation}

**验证**:
\begin{equation}
\text{softmax}([s, 0])_1 = \frac{e^s}{e^s + e^0} = \frac{e^s}{e^s + 1} = \sigma(s)
\tag{25}
\end{equation}

利用式(8)，二元分布的对称KL散度为：
\begin{equation}
\begin{aligned}
D(p_i^{(2)}, q_i^{(2)}) &= (\sigma(s_i) - \sigma(t_i))(s_i - t_i) + ((1-\sigma(s_i)) - (1-\sigma(t_i)))(0 - 0) \\
&= (\sigma(s_i) - \sigma(t_i))(s_i - t_i)
\end{aligned}
\tag{26}
\end{equation}

对所有分量求和即得式(22)。证毕。

### 9. 梯度分析

**定理9.1 (GlobalPointer KL散度的梯度)**

对于损失函数$L = D_{GP}(\mathbf{s}, \mathbf{t})$，其梯度为：
\begin{equation}
\frac{\partial L}{\partial s_i} = \sigma(s_i)(1-\sigma(s_i))(s_i - t_i) + (\sigma(s_i) - \sigma(t_i))
\tag{27}
\end{equation}

**推导**:

利用链式法则：
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial s_i} &= \frac{\partial}{\partial s_i}\left[(\sigma(s_i) - \sigma(t_i))(s_i - t_i)\right] \\
&= \frac{\partial\sigma(s_i)}{\partial s_i}(s_i - t_i) + (\sigma(s_i) - \sigma(t_i))
\end{aligned}
\tag{28}
\end{equation}

其中：
\begin{equation}
\frac{\partial\sigma(s_i)}{\partial s_i} = \sigma(s_i)(1-\sigma(s_i))
\tag{29}
\end{equation}

**梯度直觉**:
1. 第一项: $\sigma(s_i)(1-\sigma(s_i))$是sigmoid的导数，在$s_i=0$附近最大
2. 第二项: 直接的概率差
3. 当$s_i \approx t_i$时，梯度接近0（符合最优性条件）

### 10. 与标准Softmax KL散度的对比

**对比10.1**: 列表形式对比：

| 特性 | Softmax KL | GlobalPointer KL |
|------|-----------|------------------|
| 激活函数 | $\text{softmax}(\mathbf{s})$ | $\sigma(\mathbf{s})$ (逐元素) |
| 分类类型 | 单标签多分类 | 多标签分类 |
| 决策边界 | argmax | threshold at 0 |
| 归一化 | 全局归一化 $\sum p_i = 1$ | 独立概率 |
| 类别不平衡 | 需要权重调整 | 自动平衡 |

**定理10.1 (退化情况)**

当类别数为2时，两种KL散度在适当参数化下等价。

**证明**: 对于二分类，softmax可以写为：
\begin{equation}
p_1 = \frac{e^{s_1}}{e^{s_1}+e^{s_2}} = \frac{1}{1+e^{s_2-s_1}} = \sigma(s_1-s_2)
\tag{30}
\end{equation}

因此二分类softmax本质上就是sigmoid。

### 11. R-Drop正则化应用

**背景11.1**: R-Drop是一种正则化技术，通过最小化同一样本两次前向传播的输出分布之间的KL散度来提高模型鲁棒性。

**定义11.1 (R-Drop损失)**

对于同一输入$x$，两次dropout后得到输出$\mathbf{s}^{(1)}$和$\mathbf{s}^{(2)}$，R-Drop损失为：
\begin{equation}
L_{R-Drop} = L_{task} + \lambda \cdot D(\mathbf{s}^{(1)}, \mathbf{s}^{(2)})
\tag{31}
\end{equation}

其中$L_{task}$是任务损失，$\lambda$是超参数。

**在GlobalPointer中的应用**:
\begin{equation}
L_{total} = L_{CE} + \lambda \sum_i (\sigma(s_i^{(1)}) - \sigma(s_i^{(2)}))(s_i^{(1)} - s_i^{(2)})
\tag{32}
\end{equation}

其中$L_{CE}$是多标签交叉熵损失。

### 12. 数值稳定性分析

**问题12.1 (数值溢出)**

直接计算$e^{s_i}$可能导致数值溢出，当$s_i$很大时。

**解决方案**: 使用log-sum-exp技巧：
\begin{equation}
\log\sum_i e^{s_i} = m + \log\sum_i e^{s_i - m}
\tag{33}
\end{equation}

其中$m = \max_i s_i$。

**对于Sigmoid**: sigmoid函数在$|x|$很大时也可能数值不稳定：
\begin{equation}
\sigma(x) = \begin{cases}
\frac{1}{1+e^{-x}}, & x \geq 0 \\
\frac{e^x}{1+e^x}, & x < 0
\end{cases}
\tag{34}
\end{equation}

### 13. 实践建议

**建议13.1 (超参数选择)**

R-Drop的权重$\lambda$通常选择范围：
- 小数据集: $\lambda \in [0.1, 0.5]$
- 大数据集: $\lambda \in [0.01, 0.1]$
- GlobalPointer: $\lambda \in [0.1, 1.0]$

**建议13.2 (训练策略)**

1. **预热阶段**: 前几个epoch不使用R-Drop，让模型先学习基本任务
2. **逐渐增加**: 可以让$\lambda$从0逐渐增加到目标值
3. **验证集监控**: 监控验证集性能，防止过度正则化

**建议13.3 (实现细节)**

```python
# 伪代码示例
def globalpointer_kl_loss(s1, s2):
    """
    s1, s2: logits from two forward passes
    """
    prob1 = sigmoid(s1)  # shape: (batch, num_classes)
    prob2 = sigmoid(s2)

    # KL divergence
    kl = (prob1 - prob2) * (s1 - s2)
    return kl.sum() / batch_size
```

### 14. 信息论视角

**定义14.1 (互信息)**

对于两个随机变量$X$和$Y$，互信息定义为：
\begin{equation}
I(X; Y) = KL(p(x,y) \Vert p(x)p(y))
\tag{35}
\end{equation}

**解释**: KL散度可以理解为两个分布之间的"信息差异"。

**定理14.1 (KL散度与交叉熵的关系)**

\begin{equation}
KL(p \Vert q) = H(p, q) - H(p)
\tag{36}
\end{equation}

其中$H(p,q) = -\sum_i p_i\log q_i$是交叉熵，$H(p) = -\sum_i p_i\log p_i$是熵。

**在R-Drop中的意义**: 最小化KL散度等价于让两次输出的分布尽可能接近，从而提高预测的一致性。

### 15. 实验验证建议

**实验15.1 (消融实验)**

建议进行以下对比实验：
1. Baseline: 不使用R-Drop
2. Softmax KL: 使用标准softmax计算KL散度
3. Sigmoid KL: 使用式(22)的GlobalPointer KL散度
4. 不同$\lambda$值的影响

**评估指标**:
- F1分数（主要指标）
- 精确率和召回率
- 预测一致性（两次前向传播的输出相似度）

**实验15.2 (可视化分析)**

建议可视化：
1. KL散度在训练过程中的变化
2. 不同类别的logits分布
3. R-Drop对边界样本的影响

### 16. 理论性质总结

**性质16.1 (凸性)**

对于固定的$\mathbf{t}$，$D_{GP}(\mathbf{s}, \mathbf{t})$关于$\mathbf{s}$是凸函数。

**性质16.2 (对称性)**

$D_{GP}(\mathbf{s}, \mathbf{t}) = D_{GP}(\mathbf{t}, \mathbf{s})$（根据构造）

**性质16.3 (缩放不变性)**

对于常数$c$，$D_{GP}(\mathbf{s}+c\mathbf{1}, \mathbf{t}+c\mathbf{1}) = D_{GP}(\mathbf{s}, \mathbf{t})$，其中$\mathbf{1}$是全1向量。

**证明**: sigmoid函数满足：
\begin{equation}
\sigma(x+c) - \sigma(y+c) \text{ 依赖于 } (x-y)
\tag{37}
\end{equation}

### 17. 扩展与变体

**变体17.1 (加权GlobalPointer KL)**

可以为不同类别赋予不同权重：
\begin{equation}
D_{weighted}(\mathbf{s}, \mathbf{t}) = \sum_i w_i(\sigma(s_i) - \sigma(t_i))(s_i - t_i)
\tag{38}
\end{equation}

**变体17.2 (温度缩放)**

引入温度参数$\tau$：
\begin{equation}
D_{\tau}(\mathbf{s}, \mathbf{t}) = \sum_i (\sigma(s_i/\tau) - \sigma(t_i/\tau))(s_i - t_i)
\tag{39}
\end{equation}

### 18. 计算复杂度分析

**时间复杂度**: 对于$n$个类别：
- Sigmoid计算: $O(n)$
- 差值计算: $O(n)$
- 内积计算: $O(n)$
- **总计**: $O(n)$

**空间复杂度**: $O(n)$（存储概率和logits）

**对比**: 与标准softmax KL散度相同的复杂度。

### 19. 相关工作对比

**对比19.1**: 与虚拟对抗训练(VAT)的关系：
- VAT: 在输入空间添加对抗扰动，最小化输出KL散度
- R-Drop: 通过dropout在特征空间添加随机扰动
- GlobalPointer KL: 适配多标签场景的KL散度计算方式

**对比19.2**: 与标签平滑的关系：
- 标签平滑: 修改目标分布，防止过拟合
- R-Drop: 增加输出一致性约束
- 两者可以结合使用

### 20. 总结与展望

**关键贡献**:
1. 提出了GlobalPointer场景下合理的KL散度形式
2. 证明了其等价于多个独立二分类KL散度之和
3. 为R-Drop在多标签分类中的应用提供了理论基础

**未来方向**:
1. 探索其他形式的散度度量（如Wasserstein距离）
2. 研究温度参数的自适应调整策略
3. 扩展到更复杂的结构化预测任务

**实践要点**:
- 正确选择sigmoid而非softmax激活
- 合理调整正则化权重$\lambda$
- 注意数值稳定性问题
- 进行充分的消融实验验证


---

## 公式推导与注释

### 1. KL散度基础理论

**定义1.1 (KL散度的基本定义)**

对于两个离散概率分布$p$和$q$，Kullback-Leibler散度定义为：
\begin{equation}
KL(p\Vert q) = \sum_i p_i \log\frac{p_i}{q_i}
\tag{1}
\end{equation}

**数学直觉**: KL散度度量了使用分布$q$来近似分布$p$时的信息损失。它具有以下性质：

**性质1.1 (非负性)**

对于任意概率分布$p,q$，有：
\begin{equation}
KL(p\Vert q) \geq 0
\tag{2}
\end{equation}
等号成立当且仅当$p=q$。

**证明**: 利用Jensen不等式。由于$-\log$是凸函数，我们有：
\begin{equation}
\begin{aligned}
-KL(p\Vert q) &= -\sum_i p_i\log\frac{p_i}{q_i} \\
&= \sum_i p_i\log\frac{q_i}{p_i} \\
&\leq \log\sum_i p_i\frac{q_i}{p_i} \\
&= \log\sum_i q_i = \log 1 = 0
\end{aligned}
\tag{3}
\end{equation}

**性质1.2 (非对称性)**

一般情况下，$KL(p\Vert q) \neq KL(q\Vert p)$。这是KL散度的重要特性。

**数值示例**: 考虑$p=[0.9, 0.1]$, $q=[0.5, 0.5]$:
\begin{equation}
\begin{aligned}
KL(p\Vert q) &= 0.9\log\frac{0.9}{0.5} + 0.1\log\frac{0.1}{0.5} \\
&\approx 0.510 \\
KL(q\Vert p) &= 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1} \\
&\approx 0.510
\end{aligned}
\tag{4}
\end{equation}

注意：在这个特殊例子中两者相等是巧合。

### 2. 对称KL散度推导

**定义2.1 (对称KL散度)**

为了消除KL散度的非对称性，我们定义对称KL散度：
\begin{equation}
D(p,q) = KL(p\Vert q) + KL(q\Vert p)
\tag{5}
\end{equation}

**展开推导**:
\begin{equation}
\begin{aligned}
D(p,q) &= \sum_i p_i\log\frac{p_i}{q_i} + \sum_i q_i\log\frac{q_i}{p_i} \\
&= \sum_i \left(p_i\log p_i - p_i\log q_i + q_i\log q_i - q_i\log p_i\right) \\
&= \sum_i \left[(p_i - q_i)\log p_i - (p_i - q_i)\log q_i\right] \\
&= \sum_i (p_i - q_i)(\log p_i - \log q_i)
\end{aligned}
\tag{6}
\end{equation}

**几何直觉**: 对称KL散度可以理解为两个分布之间的"双向距离"，它满足对称性：$D(p,q) = D(q,p)$。

### 3. Softmax参数化下的KL散度

**假设3.1**: 假设$p$和$q$都由softmax函数生成：
\begin{equation}
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \quad q_i = \frac{e^{t_i}}{\sum_j e^{t_j}}
\tag{7}
\end{equation}

其中$s_i$和$t_i$是logits。

**定理3.1 (Softmax下的对称KL散度简化)**

在Softmax参数化下，对称KL散度可以简化为：
\begin{equation}
D(p,q) = \sum_i (p_i - q_i)(s_i - t_i) = \langle p - q, s - t \rangle
\tag{8}
\end{equation}

**详细证明**:

从式(6)出发，代入式(7)：
\begin{equation}
\begin{aligned}
\log p_i &= \log\frac{e^{s_i}}{\sum_j e^{s_j}} = s_i - \log\sum_j e^{s_j} \\
\log q_i &= t_i - \log\sum_j e^{t_j}
\end{aligned}
\tag{9}
\end{equation}

因此：
\begin{equation}
\begin{aligned}
\log p_i - \log q_i &= \left(s_i - \log\sum_j e^{s_j}\right) - \left(t_i - \log\sum_j e^{t_j}\right) \\
&= (s_i - t_i) + \left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right)
\end{aligned}
\tag{10}
\end{equation}

代入式(6)：
\begin{equation}
\begin{aligned}
D(p,q) &= \sum_i (p_i - q_i)\left[(s_i - t_i) + \left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right)\right] \\
&= \sum_i (p_i - q_i)(s_i - t_i) + \left(\sum_i p_i - \sum_i q_i\right)\left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right) \\
&= \sum_i (p_i - q_i)(s_i - t_i) + (1 - 1) \cdot \text{常数} \\
&= \sum_i (p_i - q_i)(s_i - t_i)
\end{aligned}
\tag{11}
\end{equation}

**关键观察**: 第二项消失是因为$\sum_i p_i = \sum_i q_i = 1$（概率归一化条件）。

### 4. 向量形式与几何解释

**定义4.1 (向量内积形式)**

记$\mathbf{p} = (p_1, \ldots, p_n)^T$, $\mathbf{s} = (s_1, \ldots, s_n)^T$，式(8)可写为：
\begin{equation}
D(\mathbf{s}, \mathbf{t}) = \langle f(\mathbf{s}) - f(\mathbf{t}), \mathbf{s} - \mathbf{t} \rangle
\tag{12}
\end{equation}

其中$f = \text{softmax}$是作用于向量的softmax函数。

**几何解释**:

1. **内积结构**: 这是两个向量$(\mathbf{s} - \mathbf{t})$和$(f(\mathbf{s}) - f(\mathbf{t}))$的内积
2. **单调性**: 由于softmax是单调的，当$s_i > t_i$时，通常有$p_i > q_i$，因此内积为正
3. **投影视角**: 可以理解为logits差在概率差方向上的投影

**性质4.1 (正定性)**

对于$\mathbf{s} \neq \mathbf{t}$，有$D(\mathbf{s}, \mathbf{t}) > 0$。

**证明思路**: softmax的严格单调性保证了当$s_i > t_i$时$p_i > q_i$，因此：
\begin{equation}
(p_i - q_i)(s_i - t_i) > 0
\tag{13}
\end{equation}

### 5. Softmax的函数性质分析

**定义5.1 (Softmax作为光滑近似)**

Softmax函数可以理解为$\text{onehot}(\text{argmax}(\cdot))$的光滑近似：
\begin{equation}
\lim_{\beta\to\infty} \text{softmax}(\beta \mathbf{s}) = \text{onehot}(\text{argmax}(\mathbf{s}))
\tag{14}
\end{equation}

**直觉**: 当温度参数趋于0（或等价地，logits被放大）时，softmax输出趋于one-hot向量。

**性质5.1 (Softmax的Jacobian)**

Softmax函数的Jacobian矩阵为：
\begin{equation}
\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j)
\tag{15}
\end{equation}

其中$\delta_{ij}$是Kronecker delta函数。

**推导**:
\begin{equation}
\begin{aligned}
\frac{\partial}{\partial s_j}\left(\frac{e^{s_i}}{\sum_k e^{s_k}}\right) &= \frac{\delta_{ij}e^{s_i}\sum_k e^{s_k} - e^{s_i}e^{s_j}}{(\sum_k e^{s_k})^2} \\
&= \frac{e^{s_i}}{\sum_k e^{s_k}}\left(\delta_{ij} - \frac{e^{s_j}}{\sum_k e^{s_k}}\right) \\
&= p_i(\delta_{ij} - p_j)
\end{aligned}
\tag{16}
\end{equation}

### 6. GlobalPointer场景分析

**背景6.1**: GlobalPointer用于命名实体识别等任务，其输出是多标签分类：
- 目标不是找到logits最大的单个类
- 而是找到所有logits大于0的类别

**定义6.1 (多标签分类的决策函数)**

GlobalPointer的决策函数为：
\begin{equation}
\hat{y}_i = \begin{cases}
1, & \text{if } s_i > 0 \\
0, & \text{if } s_i \leq 0
\end{cases}
\tag{17}
\end{equation}

**关键差异**: 与softmax+argmax不同：
- Softmax场景: 目标是让正类logit成为最大值
- GlobalPointer场景: 目标是让正类logit大于0，负类logit小于0

### 7. Sigmoid函数的角色

**定义7.1 (Sigmoid函数)**

Sigmoid函数定义为：
\begin{equation}
\sigma(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{1+e^x}
\tag{18}
\end{equation}

**性质7.1 (Sigmoid作为光滑近似)**

Sigmoid是"大于0置1，小于0置0"的光滑近似：
\begin{equation}
\lim_{x\to+\infty}\sigma(x) = 1, \quad \lim_{x\to-\infty}\sigma(x) = 0
\tag{19}
\end{equation}

**性质7.2 (Sigmoid的对称性)**

Sigmoid满足：
\begin{equation}
\sigma(-x) = 1 - \sigma(x)
\tag{20}
\end{equation}

**证明**:
\begin{equation}
\sigma(-x) = \frac{1}{1+e^x} = \frac{1+e^{-x}-e^{-x}}{1+e^{-x}} = 1 - \frac{e^{-x}}{1+e^{-x}} = 1 - \sigma(x)
\tag{21}
\end{equation}

### 8. GlobalPointer的KL散度设计

**类比推导**: 根据式(12)的形式，将softmax替换为sigmoid：
\begin{equation}
D_{GP}(\mathbf{s}, \mathbf{t}) = \sum_i (\sigma(s_i) - \sigma(t_i))(s_i - t_i) = \langle \sigma(\mathbf{s}) - \sigma(\mathbf{t}), \mathbf{s} - \mathbf{t} \rangle
\tag{22}
\end{equation}

**直觉**:
- Softmax场景: $f$将最大值置1，其余置0
- GlobalPointer场景: $f$将大于0的置1，小于0的置0
- 因此将softmax替换为sigmoid是自然的

**定理8.1 (GlobalPointer KL散度的等价性)**

式(22)等价于每个logit单独用sigmoid激活后，各自算二元KL散度再求和。

**证明**:

考虑单个分量的二元分布：
\begin{equation}
p_i^{(2)} = [\sigma(s_i), 1-\sigma(s_i)], \quad q_i^{(2)} = [\sigma(t_i), 1-\sigma(t_i)]
\tag{23}
\end{equation}

**引理8.1**: 二元分布$[\sigma(s), 1-\sigma(s)]$等价于用$[s, 0]$作为logits的softmax：
\begin{equation}
[\sigma(s), 1-\sigma(s)] = \text{softmax}([s, 0])
\tag{24}
\end{equation}

**验证**:
\begin{equation}
\text{softmax}([s, 0])_1 = \frac{e^s}{e^s + e^0} = \frac{e^s}{e^s + 1} = \sigma(s)
\tag{25}
\end{equation}

利用式(8)，二元分布的对称KL散度为：
\begin{equation}
\begin{aligned}
D(p_i^{(2)}, q_i^{(2)}) &= (\sigma(s_i) - \sigma(t_i))(s_i - t_i) + ((1-\sigma(s_i)) - (1-\sigma(t_i)))(0 - 0) \\
&= (\sigma(s_i) - \sigma(t_i))(s_i - t_i)
\end{aligned}
\tag{26}
\end{equation}

对所有分量求和即得式(22)。证毕。

### 9. 梯度分析

**定理9.1 (GlobalPointer KL散度的梯度)**

对于损失函数$L = D_{GP}(\mathbf{s}, \mathbf{t})$，其梯度为：
\begin{equation}
\frac{\partial L}{\partial s_i} = \sigma(s_i)(1-\sigma(s_i))(s_i - t_i) + (\sigma(s_i) - \sigma(t_i))
\tag{27}
\end{equation}

**推导**:

利用链式法则：
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial s_i} &= \frac{\partial}{\partial s_i}\left[(\sigma(s_i) - \sigma(t_i))(s_i - t_i)\right] \\
&= \frac{\partial\sigma(s_i)}{\partial s_i}(s_i - t_i) + (\sigma(s_i) - \sigma(t_i))
\end{aligned}
\tag{28}
\end{equation}

其中：
\begin{equation}
\frac{\partial\sigma(s_i)}{\partial s_i} = \sigma(s_i)(1-\sigma(s_i))
\tag{29}
\end{equation}

**梯度直觉**:
1. 第一项: $\sigma(s_i)(1-\sigma(s_i))$是sigmoid的导数，在$s_i=0$附近最大
2. 第二项: 直接的概率差
3. 当$s_i \approx t_i$时，梯度接近0（符合最优性条件）

### 10. 与标准Softmax KL散度的对比

**对比10.1**: 列表形式对比：

| 特性 | Softmax KL | GlobalPointer KL |
|------|-----------|------------------|
| 激活函数 | $\text{softmax}(\mathbf{s})$ | $\sigma(\mathbf{s})$ (逐元素) |
| 分类类型 | 单标签多分类 | 多标签分类 |
| 决策边界 | argmax | threshold at 0 |
| 归一化 | 全局归一化 $\sum p_i = 1$ | 独立概率 |
| 类别不平衡 | 需要权重调整 | 自动平衡 |

**定理10.1 (退化情况)**

当类别数为2时，两种KL散度在适当参数化下等价。

**证明**: 对于二分类，softmax可以写为：
\begin{equation}
p_1 = \frac{e^{s_1}}{e^{s_1}+e^{s_2}} = \frac{1}{1+e^{s_2-s_1}} = \sigma(s_1-s_2)
\tag{30}
\end{equation}

因此二分类softmax本质上就是sigmoid。

### 11. R-Drop正则化应用

**背景11.1**: R-Drop是一种正则化技术，通过最小化同一样本两次前向传播的输出分布之间的KL散度来提高模型鲁棒性。

**定义11.1 (R-Drop损失)**

对于同一输入$x$，两次dropout后得到输出$\mathbf{s}^{(1)}$和$\mathbf{s}^{(2)}$，R-Drop损失为：
\begin{equation}
L_{R-Drop} = L_{task} + \lambda \cdot D(\mathbf{s}^{(1)}, \mathbf{s}^{(2)})
\tag{31}
\end{equation}

其中$L_{task}$是任务损失，$\lambda$是超参数。

**在GlobalPointer中的应用**:
\begin{equation}
L_{total} = L_{CE} + \lambda \sum_i (\sigma(s_i^{(1)}) - \sigma(s_i^{(2)}))(s_i^{(1)} - s_i^{(2)})
\tag{32}
\end{equation}

其中$L_{CE}$是多标签交叉熵损失。

### 12. 数值稳定性分析

**问题12.1 (数值溢出)**

直接计算$e^{s_i}$可能导致数值溢出，当$s_i$很大时。

**解决方案**: 使用log-sum-exp技巧：
\begin{equation}
\log\sum_i e^{s_i} = m + \log\sum_i e^{s_i - m}
\tag{33}
\end{equation}

其中$m = \max_i s_i$。

**对于Sigmoid**: sigmoid函数在$|x|$很大时也可能数值不稳定：
\begin{equation}
\sigma(x) = \begin{cases}
\frac{1}{1+e^{-x}}, & x \geq 0 \\
\frac{e^x}{1+e^x}, & x < 0
\end{cases}
\tag{34}
\end{equation}

### 13. 实践建议

**建议13.1 (超参数选择)**

R-Drop的权重$\lambda$通常选择范围：
- 小数据集: $\lambda \in [0.1, 0.5]$
- 大数据集: $\lambda \in [0.01, 0.1]$
- GlobalPointer: $\lambda \in [0.1, 1.0]$

**建议13.2 (训练策略)**

1. **预热阶段**: 前几个epoch不使用R-Drop，让模型先学习基本任务
2. **逐渐增加**: 可以让$\lambda$从0逐渐增加到目标值
3. **验证集监控**: 监控验证集性能，防止过度正则化

**建议13.3 (实现细节)**

```python
# 伪代码示例
def globalpointer_kl_loss(s1, s2):
    """
    s1, s2: logits from two forward passes
    """
    prob1 = sigmoid(s1)  # shape: (batch, num_classes)
    prob2 = sigmoid(s2)

    # KL divergence
    kl = (prob1 - prob2) * (s1 - s2)
    return kl.sum() / batch_size
```

### 14. 信息论视角

**定义14.1 (互信息)**

对于两个随机变量$X$和$Y$，互信息定义为：
\begin{equation}
I(X; Y) = KL(p(x,y) \Vert p(x)p(y))
\tag{35}
\end{equation}

**解释**: KL散度可以理解为两个分布之间的"信息差异"。

**定理14.1 (KL散度与交叉熵的关系)**

\begin{equation}
KL(p \Vert q) = H(p, q) - H(p)
\tag{36}
\end{equation}

其中$H(p,q) = -\sum_i p_i\log q_i$是交叉熵，$H(p) = -\sum_i p_i\log p_i$是熵。

**在R-Drop中的意义**: 最小化KL散度等价于让两次输出的分布尽可能接近，从而提高预测的一致性。

### 15. 实验验证建议

**实验15.1 (消融实验)**

建议进行以下对比实验：
1. Baseline: 不使用R-Drop
2. Softmax KL: 使用标准softmax计算KL散度
3. Sigmoid KL: 使用式(22)的GlobalPointer KL散度
4. 不同$\lambda$值的影响

**评估指标**:
- F1分数（主要指标）
- 精确率和召回率
- 预测一致性（两次前向传播的输出相似度）

**实验15.2 (可视化分析)**

建议可视化：
1. KL散度在训练过程中的变化
2. 不同类别的logits分布
3. R-Drop对边界样本的影响

### 16. 理论性质总结

**性质16.1 (凸性)**

对于固定的$\mathbf{t}$，$D_{GP}(\mathbf{s}, \mathbf{t})$关于$\mathbf{s}$是凸函数。

**性质16.2 (对称性)**

$D_{GP}(\mathbf{s}, \mathbf{t}) = D_{GP}(\mathbf{t}, \mathbf{s})$（根据构造）

**性质16.3 (缩放不变性)**

对于常数$c$，$D_{GP}(\mathbf{s}+c\mathbf{1}, \mathbf{t}+c\mathbf{1}) = D_{GP}(\mathbf{s}, \mathbf{t})$，其中$\mathbf{1}$是全1向量。

**证明**: sigmoid函数满足：
\begin{equation}
\sigma(x+c) - \sigma(y+c) \text{ 依赖于 } (x-y)
\tag{37}
\end{equation}

### 17. 扩展与变体

**变体17.1 (加权GlobalPointer KL)**

可以为不同类别赋予不同权重：
\begin{equation}
D_{weighted}(\mathbf{s}, \mathbf{t}) = \sum_i w_i(\sigma(s_i) - \sigma(t_i))(s_i - t_i)
\tag{38}
\end{equation}

**变体17.2 (温度缩放)**

引入温度参数$\tau$：
\begin{equation}
D_{\tau}(\mathbf{s}, \mathbf{t}) = \sum_i (\sigma(s_i/\tau) - \sigma(t_i/\tau))(s_i - t_i)
\tag{39}
\end{equation}

### 18. 计算复杂度分析

**时间复杂度**: 对于$n$个类别：
- Sigmoid计算: $O(n)$
- 差值计算: $O(n)$
- 内积计算: $O(n)$
- **总计**: $O(n)$

**空间复杂度**: $O(n)$（存储概率和logits）

**对比**: 与标准softmax KL散度相同的复杂度。

### 19. 相关工作对比

**对比19.1**: 与虚拟对抗训练(VAT)的关系：
- VAT: 在输入空间添加对抗扰动，最小化输出KL散度
- R-Drop: 通过dropout在特征空间添加随机扰动
- GlobalPointer KL: 适配多标签场景的KL散度计算方式

**对比19.2**: 与标签平滑的关系：
- 标签平滑: 修改目标分布，防止过拟合
- R-Drop: 增加输出一致性约束
- 两者可以结合使用

### 20. 总结与展望

**关键贡献**:
1. 提出了GlobalPointer场景下合理的KL散度形式
2. 证明了其等价于多个独立二分类KL散度之和
3. 为R-Drop在多标签分类中的应用提供了理论基础

**未来方向**:
1. 探索其他形式的散度度量（如Wasserstein距离）
2. 研究温度参数的自适应调整策略
3. 扩展到更复杂的结构化预测任务

**实践要点**:
- 正确选择sigmoid而非softmax激活
- 合理调整正则化权重$\lambda$
- 注意数值稳定性问题
- 进行充分的消融实验验证

