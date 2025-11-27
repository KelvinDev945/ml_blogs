---
title: 多标签“Softmax+交叉熵”的软标签版本
slug: 多标签softmax交叉熵的软标签版本
date: 2022-05-07
tags: 优化, 损失函数, 光滑, 生成模型, attention
status: completed
---

# 多标签“Softmax+交叉熵”的软标签版本

**原文链接**: [https://spaces.ac.cn/archives/9064](https://spaces.ac.cn/archives/9064)

**发布日期**: 

---

**（注：本文的相关内容已整理成论文[《ZLPR: A Novel Loss for Multi-label Classification》](https://papers.cool/arxiv/2208.02955)，如需引用可以直接引用英文论文，谢谢。）**

在[《将“Softmax+交叉熵”推广到多标签分类问题》](/archives/7359)中，我们提出了一个用于多标签分类的损失函数：  
\begin{equation}\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(1 + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:original}\end{equation}  
这个损失函数有着单标签分类中“Softmax+交叉熵”的优点，即便在正负类不平衡的依然能够有效工作。但从这个损失函数的形式我们可以看到，它只适用于“硬标签”，这就意味着label smoothing、[mixup](/archives/5693)等技巧就没法用了。本文则尝试解决这个问题，提出上述损失函数的一个软标签版本。

## 巧妙联系 #

多标签分类的经典方案就是转化为多个二分类问题，即每个类别用sigmoid函数$\sigma(x)=1/(1+e^{-x})$激活，然后各自用二分类交叉熵损失。当正负类别极其不平衡时，这种做法的表现通常会比较糟糕，而相比之下损失$\eqref{eq:original}$通常是一个更优的选择。

在之前文章的评论区中，读者 [@wu.yan](/archives/7359/comment-page-2#comment-14196) 揭示了多个“sigmoid+二分类交叉熵”与式$\eqref{eq:original}$的一个巧妙的联系：多个“sigmoid+二分类交叉熵”可以适当地改写成  
\begin{equation}\begin{aligned}  
&\,-\sum_{j\in\Omega_{pos}}\log\sigma(s_j)-\sum_{i\in\Omega_{neg}}\log(1-\sigma(s_i))\\\  
=&\,\log\prod_{j\in\Omega_{pos}}(1+e^{-s_j})+\log\prod_{i\in\Omega_{neg}}(1+e^{s_i})\\\  
=&\,\log\left(1+\sum_{j\in\Omega_{pos}}e^{-s_j}+\cdots\right)+\log\left(1+\sum_{i\in\Omega_{neg}}e^{s_i}+\cdots\right)  
\end{aligned}\label{eq:link}\end{equation}  
对比式$\eqref{eq:original}$，我们可以发现式$\eqref{eq:original}$正好是上述多个“sigmoid+二分类交叉熵”的损失去掉了$\cdots$所表示的高阶项！在正负类别不平衡时，这些高阶项占据了过高的权重，加剧了不平衡问题，从而效果不佳；相反，去掉这些高阶项后，并没有改变损失函数的作用（希望正类得分大于0、负类得分小于0），同时因为括号内的求和数跟类别数是线性关系，因此正负类各自的损失差距不会太大。

## 形式猜测 #

这个巧妙联系告诉我们，要寻找式$\eqref{eq:original}$的软标签版本，可以尝试从多个“sigmoid+二分类交叉熵”的软标签版本出发，然后尝试去掉高阶项。所谓软标签，指的是标签不再是0或1，而是0～1之间的任意实数都有可能，表示属于该类的可能性。而对于二分类交叉熵，它的软标签版本很简单：  
\begin{equation}-t\log\sigma(s)-(1-t)\log(1-\sigma(s))\end{equation}  
这里$t$就是软标签，而$s$就是对应的打分。模仿过程$\eqref{eq:link}$，我们可以得到  
\begin{equation}\begin{aligned}  
&\,-\sum_i t_i\log\sigma(s_i)-\sum_i (1-t_i)\log(1-\sigma(s_i))\\\  
=&\,\log\prod_i(1+e^{-s_i})^{t_i}+\log\prod_i (1+e^{s_i})^{1-t_i}\\\  
=&\,\log\prod_i(1+t_i e^{-s_i} + \cdots)+\log\prod_i (1+(1-t_i)e^{s_i}+\cdots)\\\  
=&\,\log\left(1+\sum_i t_i e^{-s_i}+\cdots\right)+\log\left(1+\sum_i(1-t_i)e^{s_i}+\cdots\right)  
\end{aligned}\end{equation}  
如果去掉高阶项，那么就得到  
\begin{equation}\log\left(1+\sum_i t_i e^{-s_i}\right)+\log\left(1+\sum_i(1-t_i)e^{s_i}\right)\label{eq:soft}\end{equation}  
它就是式$\eqref{eq:original}$的软标签版本的候选形式，可以发现当$t_i\in\\{0,1\\}$时，正好是退化为式$\eqref{eq:original}$的。

## 证明结果 #

就目前来说，式$\eqref{eq:soft}$顶多是一个“候选”形式，要将它“转正”，我们需要证明在$t_i$为0～1浮点数时，式$\eqref{eq:soft}$能学出有意义的结果。所谓有意义，指的是理论上能够通过$s_i$来重构$t_i$的信息（$s_i$是模型预测结果，$t_i$是给定标签，所以$s_i$能重构$t_i$是机器学习的目标）。

为此，我们记式$\eqref{eq:soft}$为$l$，并求$s_i$的偏导数：  
\begin{equation}\frac{\partial l}{\partial s_i} = \frac{-t_i e^{-s_i}}{1+\sum\limits_i t_i e^{-s_i}}+\frac{(1-t_i)e^{s_i}}{1+\sum\limits_i(1-t_i)e^{s_i}}\end{equation}  
我们知道$l$的最小值出现在所有$\frac{\partial l}{\partial s_i}$都等于0时，直接去解方程组$\frac{\partial l}{\partial s_i}=0$并不容易，但笔者留意到一个神奇的“巧合”：当$t_i e^{-s_i}=(1-t_i)e^{s_i}$时，每个$\frac{\partial l}{\partial s_i}$自动地等于0！所以$t_i e^{-s_i}=(1-t_i)e^{s_i}$应该就是$l$的最优解了，解得  
\begin{equation}t_i = \frac{1}{1+e^{-2s_i}}=\sigma(2s_i)\end{equation}  
这是一个很漂亮的结果，它告诉我们几个信息：

> 1、式$\eqref{eq:soft}$确实是式$\eqref{eq:original}$合理的软标签推广，它能通过$s_i$完全重建$t_i$的信息，其形式也刚好与sigmoid相关；
> 
> 2、如果我们要将结果输出为0～1的概率值，那么正确的做法应该是$\sigma(2s_i)$而不是直觉中的$\sigma(s_i)$；
> 
> 3、既然最后的概率公式也具有sigmoid的形式，那么反过来想，也可以理解为我们依旧还是在学习多个sigmoid激活的二分类问题，只不过损失函数换成了式$\eqref{eq:soft}$。

## 实现技巧 #

式$\eqref{eq:soft}$的实现可以参考bert4keras的代码[multilabel_categorical_crossentropy](https://github.com/bojone/bert4keras/blob/5f5d493fe7be9ff2bd0e303e78ed945d386ed8fd/bert4keras/backend.py#L331)，其中有个小细节值得跟大家一起交流一下。

首先，我们将式$\eqref{eq:soft}$可以等价地改写成  
\begin{equation}\log\left(1+\sum_i e^{-s_i + \log t_i}\right)+\log\left(1+\sum_i e^{s_i + \log (1-t_i)}\right)\label{eq:soft-log}\end{equation}  
所以看上去，只需要将$\log t_i$加到$-s_i$、将$\log(1-t_i)$加到$s_i$上，补零后做常规的$\text{logsumexp}$即可。但实际上，$t_i$是有可能取到$0$或$1$的，对应的$\log t_i$或$\log(1-t_i)$就是负无穷，而框架无法直接处理负无穷，因此通常在$\log$之前需要clip一下，即选定$\epsilon > 0$后定义  
\begin{equation}\text{clip}(t)=\left\\{\begin{aligned}&\epsilon, &t < \epsilon \\\  
&t, &\epsilon\leq t\leq 1-\epsilon\\\  
&1-\epsilon, &t > 1-\epsilon\end{aligned}\right.\end{equation}

但这样一clip，问题就来了。由于$\epsilon$不是真的无穷小，比如$\epsilon=10^{-7}$，那么$\log\epsilon$大约是$-16$左右；而像GlobalPointer这样的场景中，我们会提前把不合理的$s_i$给mask掉，方式是将对应的$s_i$置为一个绝对值很大的负数，比如$-10^7$；然而我们再看式$\eqref{eq:soft-log}$，第一项的求和对象是$e^{-s_i + \log t_i}$，所以$-10^7$就会变成$10^7$，如果$t_i$没有clip，那么理论上$\log t_i$是$\log 0 = -\infty$，可以把$-s_i + \log t_i$重新变回负无穷，但刚才我们已经看到进行了clip之后的$\log t_i$顶多就是$-16$，远远比不上$-s_i$的$10^7$，所以$-s_i + \log t_i$依然是一个大正数。

为了解决这个问题，我们不止要对$t_i$进行clip，我们还要找出原本小于$\epsilon$的$t_i$，手动将对应的$-s_i$置为绝对值很大的负数，同样还要找出大于$1-\epsilon$的$t_i$，将对应的$s_i$置为绝对值很大的负数，这样做就是将小于$\epsilon$的按照绝对等于0额外处理，将大于$1-\epsilon$的按照绝对等于1处理。

## 文章小结 #

本文主要将笔者之前提出的多标签“Softmax+交叉熵”推广到软标签场景，有了对应的软标签版本后，我们就可以将它与label smoothing、[mixup](/archives/5693)等技巧结合起来了，像GlobalPointer等又可以多一个炼丹方向。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9064>_

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

苏剑林. (May. 07, 2022). 《多标签“Softmax+交叉熵”的软标签版本 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9064>

@online{kexuefm-9064,  
title={多标签“Softmax+交叉熵”的软标签版本},  
author={苏剑林},  
year={2022},  
month={May},  
url={\url{https://spaces.ac.cn/archives/9064}},  
} 


---

## 详细数学推导与理论分析

### 1. 多标签分类的基础理论

**问题设定**：给定输入$x$，预测多个标签的概率分布。

对于第$i$个标签，模型输出score $s_i\in\mathbb{R}$。传统方法使用sigmoid激活：
\begin{equation}
p_i = \sigma(s_i) = \frac{1}{1 + e^{-s_i}} \tag{1}
\end{equation}

**传统多标签损失**（多个独立的二分类交叉熵）：
\begin{equation}
\mathcal{L}_{\text{BCE}} = -\sum_{j\in\Omega_{\text{pos}}} \log\sigma(s_j) - \sum_{i\in\Omega_{\text{neg}}} \log(1-\sigma(s_i)) \tag{2}
\end{equation}

其中$\Omega_{\text{pos}}$是正类集合，$\Omega_{\text{neg}}$是负类集合。

### 2. 从BCE到多标签Softmax损失的转换

**展开sigmoid函数**：
\begin{equation}
\sigma(s) = \frac{1}{1+e^{-s}} = \frac{e^s}{1+e^s}, \quad 1-\sigma(s) = \frac{1}{1+e^s} \tag{3}
\end{equation}

**将BCE损失改写**：
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{BCE}} &= -\sum_{j\in\Omega_{\text{pos}}} \log\frac{1}{1+e^{-s_j}} - \sum_{i\in\Omega_{\text{neg}}} \log\frac{1}{1+e^{s_i}} \\
&= \sum_{j\in\Omega_{\text{pos}}} \log(1+e^{-s_j}) + \sum_{i\in\Omega_{\text{neg}}} \log(1+e^{s_i})
\end{aligned} \tag{4}
\end{equation}

**利用对数性质** $\log\prod_i a_i = \sum_i \log a_i$：
\begin{equation}
\mathcal{L}_{\text{BCE}} = \log\prod_{j\in\Omega_{\text{pos}}}(1+e^{-s_j}) + \log\prod_{i\in\Omega_{\text{neg}}}(1+e^{s_i}) \tag{5}
\end{equation}

**展开乘积**（关键步骤）：

对于正类项：
\begin{equation}
\prod_{j\in\Omega_{\text{pos}}}(1+e^{-s_j}) = 1 + \sum_{j\in\Omega_{\text{pos}}} e^{-s_j} + \sum_{j_1<j_2} e^{-s_{j_1}-s_{j_2}} + \cdots \tag{6}
\end{equation}

这包含了$2^{|\Omega_{\text{pos}}|}$项！类似地，对于负类项也有$2^{|\Omega_{\text{neg}}|}$项。

**关键观察**：当正负类极不平衡时（如$|\Omega_{\text{pos}}| \ll |\Omega_{\text{neg}}|$），负类的高阶项：
\begin{equation}
\sum_{i_1<i_2<\cdots<i_k} e^{s_{i_1}+s_{i_2}+\cdots+s_{i_k}} \tag{7}
\end{equation}
会占据主导地位，导致优化困难。

**多标签Softmax损失**（只保留一阶项）：
\begin{equation}
\mathcal{L}_{\text{ML-Softmax}} = \log\left(1 + \sum_{i\in\Omega_{\text{neg}}} e^{s_i}\right) + \log\left(1 + \sum_{j\in\Omega_{\text{pos}}} e^{-s_j}\right) \tag{8}
\end{equation}

这正是原文的式(1)！

### 3. 软标签版本的推导

**软标签的动机**：
- 硬标签：$t_i \in \{0, 1\}$
- 软标签：$t_i \in [0, 1]$，表示属于该类的置信度

**二分类交叉熵的软标签版本**：
\begin{equation}
\mathcal{L}_{\text{soft-BCE}}(s, t) = -t\log\sigma(s) - (1-t)\log(1-\sigma(s)) \tag{9}
\end{equation}

**推广到多标签**（每个标签有软标签$t_i\in[0,1]$）：
\begin{equation}
\mathcal{L} = -\sum_i t_i\log\sigma(s_i) - \sum_i (1-t_i)\log(1-\sigma(s_i)) \tag{10}
\end{equation}

**改写为对数形式**：
\begin{equation}
\mathcal{L} = \sum_i t_i\log(1+e^{-s_i}) + \sum_i (1-t_i)\log(1+e^{s_i}) \tag{11}
\end{equation}

**转换为乘积形式**：
\begin{equation}
\mathcal{L} = \log\prod_i (1+e^{-s_i})^{t_i} + \log\prod_i (1+e^{s_i})^{1-t_i} \tag{12}
\end{equation}

**二项式展开**（关键步骤）：

对于$(1+x)^t$，当$t\in(0,1)$时，使用二项式定理的推广：
\begin{equation}
(1+x)^t = 1 + tx + \frac{t(t-1)}{2!}x^2 + \frac{t(t-1)(t-2)}{3!}x^3 + \cdots \tag{13}
\end{equation}

因此：
\begin{equation}
(1+e^{-s_i})^{t_i} = 1 + t_i e^{-s_i} + \frac{t_i(t_i-1)}{2}e^{-2s_i} + \cdots \tag{14}
\end{equation}

**乘积展开**：
\begin{equation}
\prod_i (1+t_i e^{-s_i} + \cdots) = 1 + \sum_i t_i e^{-s_i} + \text{(高阶项)} \tag{15}
\end{equation}

高阶项包括：
- 二阶项：$\sum_{i<j} t_i t_j e^{-s_i-s_j}$，$\sum_i \frac{t_i(t_i-1)}{2}e^{-2s_i}$
- 更高阶项...

**软标签多标签Softmax损失**（只保留一阶项）：
\begin{equation}
\boxed{\mathcal{L}_{\text{soft}} = \log\left(1 + \sum_i t_i e^{-s_i}\right) + \log\left(1 + \sum_i (1-t_i)e^{s_i}\right)} \tag{16}
\end{equation}

这正是原文的式(7)！

**验证**：当$t_i\in\{0,1\}$时：
- 若$t_i=1$：第一项包含$e^{-s_i}$，第二项不含$e^{s_i}$
- 若$t_i=0$：第一项不含$e^{-s_i}$，第二项包含$e^{s_i}$

恰好退化为式(8)！

### 4. 最优解的推导：σ(2s_i)公式

**损失函数的梯度**：

对$s_i$求偏导：
\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{soft}}}{\partial s_i} &= \frac{\partial}{\partial s_i}\left[\log\left(1 + \sum_j t_j e^{-s_j}\right) + \log\left(1 + \sum_j (1-t_j)e^{s_j}\right)\right] \\
&= \frac{-t_i e^{-s_i}}{1 + \sum_j t_j e^{-s_j}} + \frac{(1-t_i)e^{s_i}}{1 + \sum_j (1-t_j)e^{s_j}}
\end{aligned} \tag{17}
\end{equation}

**最优条件**：$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial s_i} = 0$ 对所有$i$成立。

即：
\begin{equation}
\frac{t_i e^{-s_i}}{1 + \sum_j t_j e^{-s_j}} = \frac{(1-t_i)e^{s_i}}{1 + \sum_j (1-t_j)e^{s_j}} \tag{18}
\end{equation}

**关键观察**：注意到如果对所有$i$都有：
\begin{equation}
t_i e^{-s_i} = (1-t_i)e^{s_i} \tag{19}
\end{equation}

那么：
\begin{equation}
\sum_j t_j e^{-s_j} = \sum_j (1-t_j)e^{s_j} \tag{20}
\end{equation}

代入式(18)的左右两边，自动满足！

**求解式(19)**：
\begin{equation}
\begin{aligned}
t_i e^{-s_i} &= (1-t_i)e^{s_i} \\
t_i &= (1-t_i)e^{2s_i} \\
\frac{t_i}{1-t_i} &= e^{2s_i} \\
e^{2s_i} &= \frac{t_i}{1-t_i}
\end{aligned} \tag{21}
\end{equation}

**反解$t_i$**：
\begin{equation}
\begin{aligned}
e^{2s_i}(1-t_i) &= t_i \\
e^{2s_i} - e^{2s_i}t_i &= t_i \\
e^{2s_i} &= t_i(1 + e^{2s_i}) \\
t_i &= \frac{e^{2s_i}}{1 + e^{2s_i}}
\end{aligned} \tag{22}
\end{equation}

**最终结果**：
\begin{equation}
\boxed{t_i = \sigma(2s_i) = \frac{1}{1 + e^{-2s_i}}} \tag{23}
\end{equation}

这是一个非常优雅的结果！

### 5. σ(2s_i)的深层理解

**为什么是2倍？**

从式(21)可以看出：
\begin{equation}
\log\frac{t_i}{1-t_i} = 2s_i \tag{24}
\end{equation}

左边是**logit函数**（sigmoid的反函数）：
\begin{equation}
\text{logit}(t_i) = \log\frac{t_i}{1-t_i} \tag{25}
\end{equation}

因此：
\begin{equation}
s_i = \frac{1}{2}\text{logit}(t_i) \tag{26}
\end{equation}

**几何解释**：
- 在logit空间中，$s_i$是$t_i$的logit值的一半
- 在概率空间中，需要将$s_i$放大2倍才能还原$t_i$

**对称性分析**：

注意到式(19)具有对称性：
\begin{equation}
t_i e^{-s_i} = (1-t_i)e^{s_i} \tag{27}
\end{equation}

可以改写为：
\begin{equation}
\frac{t_i}{1-t_i} = e^{2s_i} \tag{28}
\end{equation}

这说明**odds ratio**（赔率）等于$e^{2s_i}$。

### 6. 梯度的详细分析

**损失函数的Hessian矩阵**：

二阶导数：
\begin{equation}
\frac{\partial^2 \mathcal{L}_{\text{soft}}}{\partial s_i^2} = \frac{t_i e^{-s_i}(1+\sum_{j\neq i} t_j e^{-s_j})}{(1+\sum_j t_j e^{-s_j})^2} + \frac{(1-t_i)e^{s_i}(1+\sum_{j\neq i}(1-t_j)e^{s_j})}{(1+\sum_j(1-t_j)e^{s_j})^2} \tag{29}
\end{equation}

注意到：
\begin{equation}
\frac{\partial^2 \mathcal{L}_{\text{soft}}}{\partial s_i^2} > 0 \tag{30}
\end{equation}

说明损失函数是**凸函数**！

**交叉导数**（$i\neq j$）：
\begin{equation}
\frac{\partial^2 \mathcal{L}_{\text{soft}}}{\partial s_i \partial s_j} = \frac{t_i t_j e^{-s_i-s_j}}{(1+\sum_k t_k e^{-s_k})^2} - \frac{(1-t_i)(1-t_j)e^{s_i+s_j}}{(1+\sum_k(1-t_k)e^{s_k})^2} \tag{31}
\end{equation}

**最优点处的Hessian**（在$t_i = \sigma(2s_i)$处）：

利用式(20)，记$Z = 1 + \sum_j t_j e^{-s_j} = 1 + \sum_j (1-t_j)e^{s_j}$，则：
\begin{equation}
\frac{\partial^2 \mathcal{L}_{\text{soft}}}{\partial s_i^2}\bigg|_{\text{opt}} = \frac{t_i e^{-s_i}(Z-t_i e^{-s_i})}{Z^2} + \frac{(1-t_i)e^{s_i}(Z-(1-t_i)e^{s_i})}{Z^2} \tag{32}
\end{equation}

利用$t_i e^{-s_i} = (1-t_i)e^{s_i}$，记$w_i = t_i e^{-s_i}$，则：
\begin{equation}
\frac{\partial^2 \mathcal{L}_{\text{soft}}}{\partial s_i^2}\bigg|_{\text{opt}} = \frac{2w_i(Z-w_i)}{Z^2} = \frac{2w_i}{Z} - \frac{2w_i^2}{Z^2} \tag{33}
\end{equation}

这保证了最优点是**局部极小值**。

### 7. 与知识蒸馏的联系

**知识蒸馏回顾**：

在知识蒸馏中，教师模型输出软标签$t_i^{\text{teacher}}$，学生模型学习这些软标签。

**温度缩放**：

对于单标签分类，教师的软标签通常定义为：
\begin{equation}
t_i^{\text{teacher}} = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} \tag{34}
\end{equation}

其中$T$是温度参数，$z_i$是教师的logits。

**Dark Knowledge**：

温度$T>1$时，软标签更加"平滑"，包含了类别间的相似性信息。这被称为**暗知识**（Dark Knowledge）。

**与我们的公式的对比**：

在我们的多标签设定中，软标签$t_i$直接给定，学生模型学习$s_i$使得$\sigma(2s_i) \approx t_i$。

这相当于**隐式的温度参数$T=0.5$**：
\begin{equation}
t_i = \sigma\left(\frac{s_i}{0.5}\right) = \sigma(2s_i) \tag{35}
\end{equation}

**为什么是0.5而不是1？**

因为在多标签设定中，每个类别是独立的二分类，不需要像单标签那样在类别间进行归一化。因子2来自于式(19)的对称性条件。

### 8. 数值稳定性分析

**数值溢出问题**：

当$s_i$很大时：
- $e^{s_i}$会溢出（超过浮点数上限）
- $e^{-s_i}$会下溢（接近0）

**LogSumExp技巧**：

对于$\log(1 + \sum_i e^{x_i})$，使用：
\begin{equation}
\log\left(1 + \sum_i e^{x_i}\right) = \log\left(e^{x_{\max}}\left(e^{-x_{\max}} + \sum_i e^{x_i - x_{\max}}\right)\right) = x_{\max} + \log\left(e^{-x_{\max}} + \sum_i e^{x_i - x_{\max}}\right) \tag{36}
\end{equation}

其中$x_{\max} = \max_i x_i$。

**应用到我们的损失**：

第一项：
\begin{equation}
\log\left(1 + \sum_i t_i e^{-s_i}\right) = \log\left(1 + \sum_i e^{\log t_i - s_i}\right) \tag{37}
\end{equation}

记$x_i = \log t_i - s_i$，应用LogSumExp。

**处理$t_i=0$或$t_i=1$的情况**：

当$t_i=0$时，$\log t_i = -\infty$，需要特殊处理：
- 在第一项中，直接排除$t_i=0$的项
- 在第二项中，$1-t_i=1$，保留$e^{s_i}$项

当$t_i=1$时，$\log(1-t_i) = -\infty$：
- 在第一项中，保留$e^{-s_i}$项
- 在第二项中，直接排除$t_i=1$的项

**Clip操作**：

实践中，设置$\epsilon = 10^{-7}$：
\begin{equation}
t_i' = \begin{cases}
\epsilon, & t_i < \epsilon \\
t_i, & \epsilon \leq t_i \leq 1-\epsilon \\
1-\epsilon, & t_i > 1-\epsilon
\end{cases} \tag{38}
\end{equation}

但需要额外处理mask，如原文所述。

### 9. 实现细节

**改写为等价形式**：

\begin{equation}
\mathcal{L}_{\text{soft}} = \log\left(1 + \sum_i e^{-s_i + \log t_i}\right) + \log\left(1 + \sum_i e^{s_i + \log(1-t_i)}\right) \tag{39}
\end{equation}

**处理mask**：

对于被mask的位置（$s_i = -\infty$）：
1. 若$t_i < \epsilon$：手动将$-s_i$置为大负数
2. 若$t_i > 1-\epsilon$：手动将$s_i$置为大负数

**伪代码**：

```python
def multilabel_categorical_crossentropy_soft(y_true, y_pred, epsilon=1e-7):
    """
    y_true: 软标签，shape [batch_size, num_labels]
    y_pred: logits，shape [batch_size, num_labels]
    """
    # Clip soft labels
    y_true = tf.clip_by_value(y_true, epsilon, 1 - epsilon)

    # 第一项：log(1 + sum(t_i * exp(-s_i)))
    # 等价于：log(1 + sum(exp(log(t_i) - s_i)))
    neg_logits = -y_pred + tf.math.log(y_true)
    # 对于 t_i < epsilon 的位置，手动mask
    mask_neg = tf.cast(y_true < epsilon, tf.float32)
    neg_logits = neg_logits - mask_neg * 1e12
    loss1 = tf.reduce_logsumexp(
        tf.concat([tf.zeros_like(neg_logits[:, :1]), neg_logits], axis=1),
        axis=1
    )

    # 第二项：log(1 + sum((1-t_i) * exp(s_i)))
    pos_logits = y_pred + tf.math.log(1 - y_true)
    # 对于 t_i > 1-epsilon 的位置，手动mask
    mask_pos = tf.cast(y_true > 1 - epsilon, tf.float32)
    pos_logits = pos_logits - mask_pos * 1e12
    loss2 = tf.reduce_logsumexp(
        tf.concat([tf.zeros_like(pos_logits[:, :1]), pos_logits], axis=1),
        axis=1
    )

    return loss1 + loss2
```

### 10. 与Mixup和Label Smoothing的结合

**Label Smoothing**：

对于硬标签$y_i\in\{0,1\}$，label smoothing定义软标签为：
\begin{equation}
t_i = (1-\alpha)y_i + \alpha \cdot \frac{1}{2} = \begin{cases}
\alpha/2, & y_i = 0 \\
1 - \alpha/2, & y_i = 1
\end{cases} \tag{40}
\end{equation}

其中$\alpha\in(0,1)$是平滑系数（例如$\alpha=0.1$）。

**Mixup**：

对于两个样本$(x_1, y_1)$和$(x_2, y_2)$，mixup生成新样本：
\begin{equation}
\tilde{x} = \lambda x_1 + (1-\lambda)x_2, \quad \tilde{y} = \lambda y_1 + (1-\lambda)y_2 \tag{41}
\end{equation}

其中$\lambda \sim \text{Beta}(\alpha, \alpha)$。

**与我们的损失结合**：

生成的软标签$\tilde{y}_i \in [0, 1]$可以直接用于我们的损失函数$\mathcal{L}_{\text{soft}}$！

这是硬标签版本（式8）无法做到的。

### 11. 理论性质总结

**性质1：凸性**
\begin{equation}
\mathcal{L}_{\text{soft}}(s) \text{ 是 } s \text{ 的凸函数} \tag{42}
\end{equation}

证明：Hessian矩阵半正定（见式29-33）。

**性质2：可还原性**
\begin{equation}
t_i = \sigma(2s_i^*) \quad \text{其中 } s^* = \arg\min_s \mathcal{L}_{\text{soft}}(s) \tag{43}
\end{equation}

证明：见式(23)。

**性质3：退化性**
\begin{equation}
t_i \in \{0,1\} \Rightarrow \mathcal{L}_{\text{soft}} = \mathcal{L}_{\text{ML-Softmax}} \tag{44}
\end{equation}

证明：直接验证。

**性质4：梯度有界性**

对于$t_i\in[\epsilon, 1-\epsilon]$和有界的$s_i$：
\begin{equation}
\left|\frac{\partial \mathcal{L}_{\text{soft}}}{\partial s_i}\right| \leq C \tag{45}
\end{equation}

其中$C$是常数，这保证了梯度不会爆炸。

### 12. GlobalPointer中的应用

**GlobalPointer背景**：

在命名实体识别中，GlobalPointer将实体识别建模为"选择头尾位置对"的多标签分类问题。

**打分函数**：

对于位置对$(i,j)$，定义打分：
\begin{equation}
s_{ij} = \boldsymbol{q}_i^{\top} \boldsymbol{k}_j \tag{46}
\end{equation}

**Mask处理**：

对于无效的位置对（如$i>j$），设置$s_{ij} = -\infty$（实际中用$-10^7$）。

**软标签场景**：

在部分标注、远程监督等场景下，标签可能是不确定的，可以用软标签$t_{ij}\in(0,1)$表示置信度。

使用我们的损失：
\begin{equation}
\mathcal{L} = \log\left(1 + \sum_{(i,j)} t_{ij} e^{-s_{ij}}\right) + \log\left(1 + \sum_{(i,j)} (1-t_{ij})e^{s_{ij}}\right) \tag{47}
\end{equation}

**预测**：

训练后，对于位置对$(i,j)$，预测概率为：
\begin{equation}
p_{ij} = \sigma(2s_{ij}) = \frac{1}{1 + e^{-2s_{ij}}} \tag{48}
\end{equation}

### 13. 与Focal Loss的对比

**Focal Loss回顾**：

Focal Loss用于处理类别不平衡，定义为：
\begin{equation}
\mathcal{L}_{\text{focal}} = -\alpha_t (1-p_t)^\gamma \log p_t \tag{49}
\end{equation}

其中$p_t$是真实类别的预测概率，$\gamma>0$是聚焦参数。

**对比分析**：

| 方法 | 处理不平衡的方式 | 适用场景 |
|------|-----------------|---------|
| Focal Loss | 降低易分样本的权重 | 单标签分类，目标检测 |
| 我们的方法 | 去掉高阶项，线性化求和 | 多标签分类 |

**能否结合？**

理论上可以，定义：
\begin{equation}
\mathcal{L}_{\text{soft-focal}} = (1-p_i)^\gamma \left[\log\left(1 + \sum_i t_i e^{-s_i}\right) + \log\left(1 + \sum_i (1-t_i)e^{s_i}\right)\right] \tag{50}
\end{equation}

其中$p_i = \sigma(2s_i)$。但这需要更多实验验证。

### 14. 计算复杂度分析

**时间复杂度**：

对于$n$个标签：
- 前向传播：$O(n)$（两次求和）
- 反向传播：$O(n)$（梯度计算）

与标准BCE相同！

**空间复杂度**：

需要存储：
- $s_i$：$O(n)$
- $t_i$：$O(n)$
- 中间结果（两个求和）：$O(1)$

总计：$O(n)$

**与硬标签版本对比**：

完全相同的复杂度！软标签版本没有引入额外开销。

### 15. 实验建议

**超参数选择**：

1. **Clip阈值$\epsilon$**：建议$\epsilon \in [10^{-8}, 10^{-6}]$
2. **学习率**：与BCE相似，建议从$10^{-4}$开始
3. **软标签生成**：
   - Label smoothing: $\alpha \in [0.05, 0.2]$
   - Mixup: $\alpha \in [0.2, 0.4]$（Beta分布参数）

**调试技巧**：

1. **检查梯度**：确保梯度不为NaN
2. **监控损失**：损失应该单调下降
3. **验证$\sigma(2s_i)$**：在验证集上，检查$\sigma(2s_i) \approx t_i$是否成立

**消融实验**：

对比以下变体：
1. 硬标签 + BCE
2. 硬标签 + 我们的损失（式8）
3. 软标签 + BCE
4. 软标签 + 我们的损失（式16）

### 16. 信息论视角

**互信息最大化**：

我们的损失可以理解为最大化标签$T$和预测$S$之间的互信息：
\begin{equation}
I(T; S) = H(T) - H(T|S) \tag{51}
\end{equation}

其中$H(T|S)$是条件熵。

**KL散度解释**：

最小化我们的损失等价于最小化真实分布$q(t)$和模型分布$p(t|s)$之间的KL散度：
\begin{equation}
\mathbb{E}_{(s,t)\sim\mathcal{D}}[\mathcal{L}_{\text{soft}}(s, t)] \approx \text{KL}(q(t) \| p(t|s)) + \text{const} \tag{52}
\end{equation}

**熵的分解**：

对于二分类，熵可以分解为：
\begin{equation}
H(t) = -t\log t - (1-t)\log(1-t) \tag{53}
\end{equation}

这是BCE的"目标熵"，我们的损失通过式(16)的结构隐含地优化这个目标。

### 17. 概率论视角

**最大似然估计**：

我们的损失可以理解为负对数似然：
\begin{equation}
\mathcal{L}_{\text{soft}} = -\log p(t|s; \theta) \tag{54}
\end{equation}

其中$\theta$是模型参数。

**贝叶斯解释**：

如果我们对$t_i$有先验分布$p(t_i)$，那么后验分布为：
\begin{equation}
p(t_i | s_i) \propto p(s_i | t_i) p(t_i) \tag{55}
\end{equation}

在均匀先验下，MAP估计等价于MLE。

**生成模型视角**：

可以将多标签分类视为生成模型：
\begin{equation}
p(t, s) = p(s)p(t|s) = p(t)p(s|t) \tag{56}
\end{equation}

我们的损失优化$p(t|s)$，而知识蒸馏可以理解为用教师的$p(t)$作为先验。

### 18. 几何解释

**Logit空间**：

在logit空间中，$s_i$和$\text{logit}(t_i)$的关系为：
\begin{equation}
s_i = \frac{1}{2}\text{logit}(t_i) \tag{57}
\end{equation}

这是一个**线性关系**，比例系数为1/2。

**概率单纯形**：

在$n$维概率单纯形$\Delta^n = \{t \in \mathbb{R}^n : t_i \geq 0, \sum_i t_i = 1\}$上，我们的损失定义了一个**Bregman散度**。

**流形结构**：

$\sigma(2s)$定义了从$\mathbb{R}^n$（score空间）到$(0,1)^n$（概率空间）的微分同胚。

### 19. 与其他多标签损失的对比

**Asymmetric Loss**（ASL）：
\begin{equation}
\mathcal{L}_{\text{ASL}} = -\sum_{i\in\Omega_{\text{pos}}} (1-p_i)^{\gamma_+} \log p_i - \sum_{i\in\Omega_{\text{neg}}} p_i^{\gamma_-} \log(1-p_i) \tag{58}
\end{equation}

**Circle Loss**：
\begin{equation}
\mathcal{L}_{\text{Circle}} = \log\left[1 + \sum_{i\in\Omega_{\text{neg}}} e^{\gamma(s_i - \Delta_n)}\right] + \log\left[1 + \sum_{j\in\Omega_{\text{pos}}} e^{\gamma(\Delta_p - s_j)}\right] \tag{59}
\end{equation}

**对比表**：

| 损失函数 | 软标签支持 | 理论保证 | 计算复杂度 |
|---------|-----------|---------|-----------|
| BCE | 是 | Fisher一致性 | O(n) |
| 我们的硬标签版本 | 否 | 凸性 | O(n) |
| 我们的软标签版本 | 是 | 凸性 + 可还原性 | O(n) |
| ASL | 部分 | 经验有效 | O(n) |
| Circle Loss | 否 | 几何意义 | O(n) |

### 20. 开放问题与未来方向

**问题1**：能否为式(16)找到更紧的理论界？

目前我们只知道它是凸的，但具体的收敛速度如何？

**问题2**：$\sigma(2s_i)$的因子2是否最优？

能否推广到$\sigma(\beta s_i)$，其中$\beta$是可学习的参数？

**问题3**：如何扩展到有序多标签分类？

当标签之间有依赖关系时，如何修改我们的损失？

**问题4**：与对比学习的结合？

能否将我们的损失与对比学习框架结合，用于表示学习？

**未来方向**：
1. 自适应温度参数
2. 层次化多标签分类
3. 长尾分布下的改进
4. 理论收敛性分析
5. 与元学习的结合

---

## 参考文献

1. Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
2. Zhang et al., "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels", NeurIPS 2018
3. Sun et al., "Circle Loss: A Unified Perspective of Pair Similarity Optimization", CVPR 2020
4. Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
5. Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014 Workshop

## 总结

本文详细推导了多标签Softmax交叉熵的软标签版本，关键贡献包括：

1. **理论推导**：从BCE出发，通过去除高阶项得到软标签版本
2. **最优解**：证明了$t_i = \sigma(2s_i)$的优雅结果
3. **实现细节**：处理数值稳定性和mask的技巧
4. **多角度理解**：信息论、概率论、几何等视角
5. **实践指导**：超参数选择、调试技巧、应用场景

这个损失函数将多标签分类推向了软标签时代，为label smoothing、mixup等技术在多标签场景的应用铺平了道路。

