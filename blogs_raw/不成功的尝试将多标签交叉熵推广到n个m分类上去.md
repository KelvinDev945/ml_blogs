---
title: 不成功的尝试：将多标签交叉熵推广到“n个m分类”上去
slug: 不成功的尝试将多标签交叉熵推广到n个m分类上去
date: 2022-07-15
tags: 优化, 损失函数, 生成模型, attention, 优化
status: completed
---

# 不成功的尝试：将多标签交叉熵推广到“n个m分类”上去

**原文链接**: [https://spaces.ac.cn/archives/9158](https://spaces.ac.cn/archives/9158)

**发布日期**: 

---

可能有读者留意到，这次更新相对来说隔得比较久了。事实上，在上周末时就开始准备这篇文章了，然而笔者低估了这个问题的难度，几乎推导了整整一周，仍然还没得到一个完善的结果出来。目前发出来的，仍然只是一个失败的结果，希望有经验的读者可以指点指点。

在文章[《将“Softmax+交叉熵”推广到多标签分类问题》](/archives/7359)中，我们提出了一个多标签分类损失函数，它能自动调节正负类的不平衡问题，后来在[《多标签“Softmax+交叉熵”的软标签版本》](/archives/9064)中我们还进一步得到了它的“软标签”版本。本质上来说，多标签分类就是“$n$个2分类”问题，那么相应的，“$n$个$m$分类”的损失函数又该是怎样的呢？

这就是本文所要探讨的问题。

## 类比尝试 #

在软标签推广的文章[《多标签“Softmax+交叉熵”的软标签版本》](/archives/9064)中，我们是通过直接将“$n$个2分类”的sigmoid交叉熵损失，在$\log$内做一阶截断来得到最终结果的。同样的过程确实也可以推广到“$n$个$m$分类”的softmax交叉熵损失，这是笔者的第一次尝试。

记$\text{softmax}(s_{i,j}) = \frac{e^{s_{i,j}}}{\sum\limits_j e^{s_{i,j}}}$，$s_{i,j}$为预测结果，而$t_{i,j}$则为标签，那么  
\begin{equation}\begin{aligned}-\sum_i\sum_j t_{i,j}\log \text{softmax}(s_{i,j}) =&\,\sum_i\sum_j t_{i,j}\log \left(1 + \sum_{k\neq j} e^{s_{i,k} - s_{i,j}}\right)\\\  
=&\,\sum_j \log \prod_i\left(1 + \sum_{k\neq j} e^{s_{i,k} - s_{i,j}}\right)^{t_{i,j}}\\\  
=&\,\sum_j \log \left(1 + \sum_i t_{i,j}\sum_{k\neq j} e^{s_{i,k} - s_{i,j}}+\cdots\right)\\\  
\end{aligned}\end{equation}  
对$i$的求和默认是$1\sim n$，对$j$的求和默认是$1\sim m$。截断$\cdots$的高阶项，得到  
\begin{equation}l = \sum_j \log \left(1 + \sum_{i,k\neq j} t_{i,j}e^{- s_{i,j} + s_{i,k}}\right)\label{eq:loss-1}\end{equation}  
这就是笔者开始得到的loss，它是之前的结果到“$n$个$m$分类”的自然推广。事实上，如果$t_{i,j}$是硬标签，那么该loss基本上没什么问题。但笔者希望它像[《多标签“Softmax+交叉熵”的软标签版本》](/archives/9064)一样，对于软标签也能得到推导出相应的解析解。为此，笔者对它进行求导：  
\begin{equation}\frac{\partial l}{\partial s_{i,j}} = \frac{- t_{i,j}e^{- s_{i,j}}\sum\limits_{k\neq j} e^{s_{i,k}}}{1 + \sum\limits_{i,k\neq j} t_{i,j}e^{- s_{i,j} + s_{i,k}}} + \sum_{h\neq j} \frac{t_{i,h}e^{- s_{i,h}}e^{s_{i,j}}}{1 + \sum\limits_{i,k\neq h} t_{i,h}e^{- s_{i,h} + s_{i,k}}}\end{equation}  
所谓解析解，就是通过方程$\frac{\partial l}{\partial s_{i,j}}=0$来解出。然而笔者尝试了好几天，都求不出方程的解，估计并没有简单的显式解，因此，第一次尝试失败。

## 结果倒推 #

尝试了几天实在没办法后，笔者又反过来想：既然直接类比出来的结果无法求解，那么我干脆从结果倒推好了，即先把解确定，然后再反推方程应该是怎样的。于是，笔者开始了第二次尝试。

首先，观察发现原来的多标签损失，或者前面得到的损失$\eqref{eq:loss-1}$，都具有如下的形式：  
\begin{equation}l = \sum_j \log \left(1 + \sum_i t_{i,j}e^{- f(s_{i,j})}\right)\label{eq:loss-2}\end{equation}  
我们就以这个形式为出发点，求导  
\begin{equation}\frac{\partial l}{\partial s_{i,k}} = \sum_j \frac{- t_{i,j}e^{- f(s_{i,j})}\frac{\partial f(s_{i,j})}{\partial s_{i,k}}}{1 + \sum\limits_i t_{i,j}e^{- f(s_{i,j})}}\end{equation}  
我们希望$t_{i,j}=\text{softmax}(f(s_{i,j}))=e^{f(s_{i,j})}/Z_i$就是$\frac{\partial l}{\partial s_{i,k}}=0$的解析解，其中$Z_i=\sum\limits_j e^{f(s_{i,j})}$。那么代入得到  
\begin{equation}0=\frac{\partial l}{\partial s_{i,k}} = \sum_j \frac{- (1/Z_i)\frac{\partial f(s_{i,j})}{\partial s_{i,k}}}{1 + \sum\limits_i 1/Z_i} = \frac{- (1/Z_i)\frac{\partial \left(\sum\limits_j f(s_{i,j})\right)}{\partial s_{i,k}}}{1 + \sum\limits_i 1/Z_i}\end{equation}  
所以要让上式自然成立，我们发现只需要让$\sum\limits_j f(s_{i,j})$等于一个跟$i,j$都无关的常数。简单起见，我们让  
\begin{equation}f(s_{i,j})=s_{i,j}-  
\bar{s}_i,\qquad \bar{s}_i=\frac{1}{m}\sum_j s_{i,j}\end{equation}  
这样自然地有$\sum\limits_j f(s_{i,j})=0$，对应的优化目标就是  
\begin{equation}l = \sum_j \log \left(1 + \sum_i t_{i,j}e^{- s_{i,j} + \bar{s}_i}\right)\label{eq:loss-3}\end{equation}  
$\bar{s}_i$不影响归一化结果，所以它的理论最优解是$t_{i,j}=\text{softmax}(s_{i,j})$。

然而，看上去很美好，然而它实际上的效果会比较糟糕，$t_{i,j}=\text{softmax}(s_{i,j})$确实是理论最优解，但实际上标签越接近硬标签，它的效果会越差。因为我们知道对于损失$\eqref{eq:loss-3}$来说，只要$s_{i,j} \gg \bar{s}_i$，损失就会很接近于0，而要达到$s_{i,j} \gg \bar{s}_i$，$s_{i,j}$不一定是$s_{i,1},s_{i,2},\cdots,s_{i,m}$中的最大者，这就无法实现分类目标了。

## 思考分析 #

现在我们得到了两个结果，式$\eqref{eq:loss-1}$是原来多标签交叉熵的类比推广，它在硬标签的情况下效果还是不错的，但是由于求不出软标签情况下的解析解，因此软标签的情况无法做理论评估；式$\eqref{eq:loss-3}$是从结果理论倒推出来的，理论上它的解析解就是简单的softmax，但由于实际优化算法的限制，硬标签的表现通常很差，甚至无法保证目标logits是最大值。特别地，当$m=2$时，式$\eqref{eq:loss-1}$和式$\eqref{eq:loss-3}$都能退化为多标签交叉熵。

我们知道，多标签交叉熵能够自动调节正负样本不平衡的问题，同样地，虽然我们目前还没能得到一个完美的推广，但理论上推广到“$n$个$m$分类”后依然能够自动调节$m$个类的不平衡问题。那么平衡的机制是怎样的呢？其实不难理解，不管是类比推广的式$\eqref{eq:loss-1}$，还是一般的假设式$\eqref{eq:loss-2}$，对$i$的求和都放在了$\log$里边，原本每个类的损失占比大体上是正比于“ _该类的样本数_ ”的，改为放在了$\log$里边求和后，每个类的损失占就大致等于“ _该类的样本数的 对数_”，从而缩小了每个类的损失差距，自动缓解了不平衡问题。

遗憾的是，本文还没有得出关于“$n$个$m$分类”的完美推广——它应该包含两个特性：1、通过$\log$的方法自动调节类别不平衡现象；2、能够求出软标签情况下的解析解。对于硬标签来说，直接用式$\eqref{eq:loss-1}$应该是足够了；而对于软标签来说，笔者实在是没辙了，欢迎有兴趣的读者一起思考交流。

## 文章小结 #

本文尝试将之前的多标签交叉熵推广到“$n$个$m$分类”上去，遗憾的是，这一次的推广并不算成功，暂且将结果分享在此，希望有兴趣的读者能一起参与改进。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9158>_

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

苏剑林. (Jul. 15, 2022). 《不成功的尝试：将多标签交叉熵推广到“n个m分类”上去 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9158>

@online{kexuefm-9158,  
title={不成功的尝试：将多标签交叉熵推广到“n个m分类”上去},  
author={苏剑林},  
year={2022},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/9158}},  
} 


---

## 公式推导与注释

### 1. 问题背景与动机

**问题1.1 (多标签分类回顾)**

多标签分类本质上是$n$个二分类问题。对于样本$x$，我们需要预测$n$个标签$\{y_1, \ldots, y_n\}$，每个$y_i \in \{0, 1\}$。

**标准方法**: 使用$n$个独立的sigmoid二分类：
\begin{equation}
L_{binary} = -\sum_{i=1}^n \left[y_i\log\sigma(s_i) + (1-y_i)\log(1-\sigma(s_i))\right]
\tag{1}
\end{equation}

其中$s_i$是第$i$个类别的logit，$\sigma(x) = 1/(1+e^{-x})$。

**问题**: 当正负样本严重不平衡时（例如$n=1000$，但只有3个正类），式(1)的表现会急剧下降。

**改进方案 (多标签交叉熵)**: 在之前的工作中，我们提出：
\begin{equation}
L_{multilabel} = \log\left(1 + \sum_{i \in \Omega_{neg}} e^{s_i}\right) + \log\left(1 + \sum_{j \in \Omega_{pos}} e^{-s_j}\right)
\tag{2}
\end{equation}

其中$\Omega_{pos}$是正类索引集，$\Omega_{neg}$是负类索引集。

### 2. 从二分类到m分类的推广

**问题2.1 (推广目标)**

现在考虑更一般的情况：有$n$个任务，每个任务是$m$分类问题（而不是二分类）。记：
- $s_{i,j}$: 第$i$个任务（$i=1,\ldots,n$）的第$j$个类别（$j=1,\ldots,m$）的logit
- $t_{i,j}$: 对应的标签（硬标签时为0或1，软标签时为$[0,1]$区间的实数）

**目标**: 设计一个损失函数，使得：
1. 具有类别不平衡自动调节能力（类似式(2)）
2. 支持软标签优化
3. 存在解析解

### 3. 第一次尝试：直接类比

**推导3.1 (从Softmax交叉熵出发)**

标准的$n$个$m$分类的softmax交叉熵为：
\begin{equation}
L_{softmax} = -\sum_{i=1}^n \sum_{j=1}^m t_{i,j} \log p_{i,j}
\tag{3}
\end{equation}

其中$p_{i,j} = \text{softmax}(s_{i,j}) = \frac{e^{s_{i,j}}}{\sum_{k=1}^m e^{s_{i,k}}}$。

**展开**:
\begin{equation}
\begin{aligned}
-\sum_i \sum_j t_{i,j} \log p_{i,j} &= -\sum_i \sum_j t_{i,j} \left(s_{i,j} - \log\sum_{k=1}^m e^{s_{i,k}}\right) \\
&= \sum_i \sum_j t_{i,j} \log\sum_{k=1}^m e^{s_{i,k}} - \sum_i \sum_j t_{i,j} s_{i,j} \\
&= \sum_i \log\sum_{k=1}^m e^{s_{i,k}} - \sum_i \sum_j t_{i,j} s_{i,j}
\end{aligned}
\tag{4}
\end{equation}

最后一步利用了$\sum_{j=1}^m t_{i,j} = 1$（概率归一化）。

**改写形式**:
\begin{equation}
-\sum_i \sum_j t_{i,j} \log p_{i,j} = \sum_i \sum_j t_{i,j} \log\left(1 + \sum_{k \neq j} e^{s_{i,k} - s_{i,j}}\right)
\tag{5}
\end{equation}

**证明**:
\begin{equation}
\begin{aligned}
\log p_{i,j}^{-1} &= \log\frac{\sum_k e^{s_{i,k}}}{e^{s_{i,j}}} \\
&= \log\left(\frac{e^{s_{i,j}} + \sum_{k \neq j} e^{s_{i,k}}}{e^{s_{i,j}}}\right) \\
&= \log\left(1 + \sum_{k \neq j} e^{s_{i,k} - s_{i,j}}\right)
\end{aligned}
\tag{6}
\end{equation}

### 4. 一阶截断近似

**引理4.1 (乘积的对数展开)**

对于小量$\epsilon$，有：
\begin{equation}
\log(1 + \epsilon) \approx \epsilon \quad \text{（一阶Taylor展开）}
\tag{7}
\end{equation}

**应用到多个乘积**:
\begin{equation}
\begin{aligned}
\log \prod_{i=1}^n (1 + a_i) &= \sum_{i=1}^n \log(1 + a_i) \\
&= \log\left(1 + \sum_i a_i + \sum_{i<j} a_i a_j + \cdots\right) \\
&\approx \log\left(1 + \sum_i a_i\right)
\end{aligned}
\tag{8}
\end{equation}

截断高阶项（交叉项）后得到一阶近似。

**定理4.1 (类比推广的损失函数)**

将式(5)改写并应用一阶截断：
\begin{equation}
\begin{aligned}
L_1 &= \sum_j \log \prod_i \left(1 + \sum_{k \neq j} e^{s_{i,k} - s_{i,j}}\right)^{t_{i,j}} \\
&= \sum_j \log\left(1 + \sum_i t_{i,j} \sum_{k \neq j} e^{s_{i,k} - s_{i,j}} + \text{高阶项}\right) \\
&\approx \sum_j \log\left(1 + \sum_i \sum_{k \neq j} t_{i,j} e^{s_{i,k} - s_{i,j}}\right)
\end{aligned}
\tag{9}
\end{equation}

**重新整理**:
\begin{equation}
L_1 = \sum_{j=1}^m \log\left(1 + \sum_{i=1}^n \sum_{k \neq j} t_{i,j} e^{s_{i,k} - s_{i,j}}\right)
\tag{10}
\end{equation}

### 5. 梯度计算与解析解分析

**定理5.1 (损失函数梯度)**

对$s_{i,j}$求偏导：
\begin{equation}
\frac{\partial L_1}{\partial s_{i,j}} = \sum_{h \neq j} \frac{t_{i,h} e^{s_{i,j} - s_{i,h}}}{1 + \sum_{i,k \neq h} t_{i,k} e^{s_{i,k} - s_{i,h}}} - \frac{t_{i,j} e^{-s_{i,j}} \sum_{k \neq j} e^{s_{i,k}}}{1 + \sum_{i,k \neq j} t_{i,k} e^{s_{i,k} - s_{i,j}}}
\tag{11}
\end{equation}

**推导**: 注意$s_{i,j}$出现在两个地方：
1. 作为分子：在第$j$项的$\sum_{k \neq j} t_{i,j} e^{s_{i,k} - s_{i,j}}$中
2. 作为分母：在所有$h \neq j$项的$\sum_{i,k \neq h} t_{i,k} e^{s_{i,k} - s_{i,h}}$中

对第一部分：
\begin{equation}
\frac{\partial}{\partial s_{i,j}} \log\left(1 + \sum_i \sum_{k \neq j} t_{i,k} e^{s_{i,k} - s_{i,j}}\right) = \frac{-t_{i,j} e^{-s_{i,j}} \sum_{k \neq j} e^{s_{i,k}}}{1 + \sum_{i,k \neq j} t_{i,k} e^{s_{i,k} - s_{i,j}}}
\tag{12}
\end{equation}

对第二部分（所有$h \neq j$）：
\begin{equation}
\sum_{h \neq j} \frac{\partial}{\partial s_{i,j}} \log\left(1 + \sum_i \sum_{k \neq h} t_{i,k} e^{s_{i,k} - s_{i,h}}\right) = \sum_{h \neq j} \frac{t_{i,h} e^{s_{i,j} - s_{i,h}}}{1 + \sum_{i,k \neq h} t_{i,k} e^{s_{i,k} - s_{i,h}}}
\tag{13}
\end{equation}

**问题**: 令$\frac{\partial L_1}{\partial s_{i,j}} = 0$得到的方程组极其复杂，没有简单的解析解。

### 6. 第二次尝试：从解倒推

**策略6.1**: 既然正向推导无法得到解析解，我们反向思考：
1. 先假设理想的解的形式
2. 反推对应的损失函数
3. 检验该损失函数是否合理

**假设6.1 (理想解的形式)**

假设最优解应该是：
\begin{equation}
t_{i,j} = \text{softmax}(f(s_{i,j})) = \frac{e^{f(s_{i,j})}}{\sum_k e^{f(s_{i,k})}}
\tag{14}
\end{equation}

其中$f: \mathbb{R} \to \mathbb{R}$是某个函数。

**一般损失形式**:
\begin{equation}
L_2 = \sum_{j=1}^m \log\left(1 + \sum_{i=1}^n t_{i,j} e^{-f(s_{i,j})}\right)
\tag{15}
\end{equation}

### 7. 梯度与最优性条件

**定理7.1 (一般形式的梯度)**

对式(15)求梯度：
\begin{equation}
\frac{\partial L_2}{\partial s_{i,k}} = \sum_{j=1}^m \frac{-t_{i,j} e^{-f(s_{i,j})} \frac{\partial f(s_{i,j})}{\partial s_{i,k}}}{1 + \sum_i t_{i,j} e^{-f(s_{i,j})}}
\tag{16}
\end{equation}

**化简**: 令$Z_j = 1 + \sum_i t_{i,j} e^{-f(s_{i,j})}$，则：
\begin{equation}
\frac{\partial L_2}{\partial s_{i,k}} = -\sum_{j=1}^m \frac{t_{i,j} e^{-f(s_{i,j})}}{Z_j} \cdot \frac{\partial f(s_{i,j})}{\partial s_{i,k}}
\tag{17}
\end{equation}

**最优性条件**: 如果$t_{i,j} = \frac{e^{f(s_{i,j})}}{\sum_k e^{f(s_{i,k})}}$是解，代入式(17)：
\begin{equation}
\frac{\partial L_2}{\partial s_{i,k}} = -\sum_{j=1}^m \frac{e^{f(s_{i,j})}/Z_i \cdot e^{-f(s_{i,j})}}{Z_j} \cdot \frac{\partial f(s_{i,j})}{\partial s_{i,k}}
\tag{18}
\end{equation}

其中$Z_i = \sum_k e^{f(s_{i,k})}$。

### 8. 函数f的确定

**要求**: 为了让式(18)对所有$k$都等于0，我们需要：
\begin{equation}
\sum_{j=1}^m f(s_{i,j}) = C \quad \text{（与$i,j$无关的常数）}
\tag{19}
\end{equation}

**最简单的选择**: 令$C=0$，即：
\begin{equation}
\sum_{j=1}^m f(s_{i,j}) = 0
\tag{20}
\end{equation}

**解**: 取平均中心化：
\begin{equation}
f(s_{i,j}) = s_{i,j} - \bar{s}_i, \quad \bar{s}_i = \frac{1}{m}\sum_{j=1}^m s_{i,j}
\tag{21}
\end{equation}

**验证**:
\begin{equation}
\sum_{j=1}^m f(s_{i,j}) = \sum_{j=1}^m (s_{i,j} - \bar{s}_i) = \sum_{j=1}^m s_{i,j} - m\bar{s}_i = 0
\tag{22}
\end{equation}

### 9. 倒推的损失函数

**定义9.1 (中心化损失函数)**

根据式(15)和(21)，得到：
\begin{equation}
L_3 = \sum_{j=1}^m \log\left(1 + \sum_{i=1}^n t_{i,j} e^{-(s_{i,j} - \bar{s}_i)}\right) = \sum_{j=1}^m \log\left(1 + \sum_{i=1}^n t_{i,j} e^{\bar{s}_i - s_{i,j}}\right)
\tag{23}
\end{equation}

**理论最优解**:
\begin{equation}
t_{i,j} = \frac{e^{s_{i,j} - \bar{s}_i}}{\sum_k e^{s_{i,k} - \bar{s}_i}} = \text{softmax}(s_{i,1}, \ldots, s_{i,m})_{j}
\tag{24}
\end{equation}

注意$\bar{s}_i$不影响softmax的结果。

### 10. 硬标签情况分析

**问题10.1**: 为什么式(23)在硬标签情况下表现不佳？

**分析**: 考虑第$i$个任务，假设真实标签是第$j^*$类（$t_{i,j^*}=1$，其他为0）。

损失函数简化为：
\begin{equation}
L_3 \supset \log\left(1 + e^{\bar{s}_i - s_{i,j^*}}\right)
\tag{25}
\end{equation}

**问题**: 只要$s_{i,j^*} \gg \bar{s}_i$，损失就会很小。但$s_{i,j^*}$不一定是$\{s_{i,1}, \ldots, s_{i,m}\}$中的最大值！

**具体例子**:
- 假设$s_{i,1} = 10, s_{i,2} = 5, s_{i,3} = -5$（真实类别是第2类）
- $\bar{s}_i = \frac{10+5-5}{3} = 3.33$
- $s_{i,2} - \bar{s}_i = 5 - 3.33 = 1.67 > 0$
- 但$s_{i,2}$不是最大值！模型会错误地预测第1类

### 11. 软标签vs硬标签的矛盾

**矛盾11.1**:
- **软标签情况**: $L_3$有美好的理论性质（解析解为softmax）
- **硬标签情况**: $L_3$无法保证正确分类（最大logit不一定对应最大概率）

**根本原因**: 平均中心化$\bar{s}_i$破坏了logits的相对大小关系。

**对比**: 标准softmax交叉熵不存在这个问题：
\begin{equation}
L_{standard} = -\sum_i t_{i,j^*} \log\frac{e^{s_{i,j^*}}}{\sum_k e^{s_{i,k}}}
\tag{26}
\end{equation}

这会强制$s_{i,j^*}$成为最大值。

### 12. 类别不平衡的自动调节机制

**机制12.1 (对数内求和)**

回顾多标签交叉熵（式2）和推广形式（式10、23），关键特征是：

**标准形式**:
\begin{equation}
L_{standard} = \sum_i L_i
\tag{27}
\end{equation}
每个类的损失权重正比于样本数。

**改进形式**:
\begin{equation}
L_{improved} = \sum_j \log\left(1 + \sum_i (\cdots)\right)
\tag{28}
\end{equation}
每个类的损失权重正比于样本数的对数。

**数学分析**: 假设第$j$类有$N_j$个样本：
- 标准方法: 贡献$\sim N_j$
- 改进方法: 贡献$\sim \log(1 + N_j) \approx \log N_j$

**效果**:
\begin{equation}
\frac{\log N_1}{\log N_2} < \frac{N_1}{N_2} \quad \text{当} \quad N_1 > N_2 > 1
\tag{29}
\end{equation}

缩小了类别间的权重差异。

### 13. 退化到二分类的验证

**定理13.1 (m=2时的退化)**

当$m=2$时，式(10)和式(23)都退化为多标签交叉熵。

**证明**: 对于二分类，不失一般性假设第2类是"负类"（$t_{i,2} = 1 - t_{i,1}$）。

式(23)变为：
\begin{equation}
\begin{aligned}
L_3 &= \log\left(1 + \sum_i t_{i,1} e^{\bar{s}_i - s_{i,1}}\right) + \log\left(1 + \sum_i (1-t_{i,1}) e^{\bar{s}_i - s_{i,2}}\right)
\end{aligned}
\tag{30}
\end{equation}

定义$\tilde{s}_i = s_{i,1} - s_{i,2}$（相对logit），则：
\begin{equation}
\bar{s}_i - s_{i,1} = \frac{s_{i,1} + s_{i,2}}{2} - s_{i,1} = \frac{s_{i,2} - s_{i,1}}{2} = -\frac{\tilde{s}_i}{2}
\tag{31}
\end{equation}

类似地：
\begin{equation}
\bar{s}_i - s_{i,2} = \frac{\tilde{s}_i}{2}
\tag{32}
\end{equation}

代入式(30)：
\begin{equation}
L_3 = \log\left(1 + \sum_i t_{i,1} e^{-\tilde{s}_i/2}\right) + \log\left(1 + \sum_i (1-t_{i,1}) e^{\tilde{s}_i/2}\right)
\tag{33}
\end{equation}

这与多标签交叉熵形式一致（相差一个常数因子1/2）。

### 14. 数值示例：硬标签失败

**例14.1**: 考虑3个任务，每个任务3分类。

**Logits**:
\begin{equation}
\mathbf{S} = \begin{pmatrix}
5 & 2 & 1 \\
3 & 6 & 0 \\
1 & 2 & 8
\end{pmatrix}
\tag{34}
\end{equation}

真实标签: 任务1选类1，任务2选类2，任务3选类3。

**平均值**: $\bar{s}_1 = 8/3 \approx 2.67$, $\bar{s}_2 = 3$, $\bar{s}_3 = 11/3 \approx 3.67$

**损失贡献**:
- 任务1: $\log(1 + e^{2.67-5}) = \log(1 + e^{-2.33}) \approx 0.091$
- 任务2: $\log(1 + e^{3-6}) = \log(1 + e^{-3}) \approx 0.048$
- 任务3: $\log(1 + e^{3.67-8}) = \log(1 + e^{-4.33}) \approx 0.013$

看起来损失很小！但用argmax预测：
- 任务1: argmax$(5,2,1) = 1$ ✓
- 任务2: argmax$(3,6,0) = 2$ ✓
- 任务3: argmax$(1,2,8) = 3$ ✓

这个例子中恰好正确。但考虑轻微扰动：

**扰动Logits**:
\begin{equation}
\mathbf{S'} = \begin{pmatrix}
5 & 2 & 4 \\
3 & 6 & 0 \\
1 & 2 & 8
\end{pmatrix}
\tag{35}
\end{equation}

现在$\bar{s}_1 = 11/3 \approx 3.67$，$s_{1,1} - \bar{s}_1 = 5 - 3.67 = 1.33 > 0$，损失仍然小，但argmax已经错了（如果第3类也很接近）。

### 15. 软标签的优势分析

**观察15.1**: 软标签情况下，式(23)的解析解是softmax，这意味着：

**收敛性**: 梯度下降会收敛到softmax分布
**光滑性**: 损失函数关于logits是光滑的
**唯一性**: 在凸优化框架下，解是唯一的

**但**: 这些优点在硬标签情况下都不重要，因为我们的评估指标是argmax正确率，而非概率分布的接近程度。

### 16. 组合优化视角

**视角16.1 (硬标签作为组合优化)**

硬标签分类本质上是组合优化问题：
\begin{equation}
\hat{y}_i = \arg\max_{j \in \{1,\ldots,m\}} s_{i,j}
\tag{36}
\end{equation}

**松弛**: softmax是这个组合优化的凸松弛：
\begin{equation}
p_{i,j} = \text{softmax}(s_{i,1}, \ldots, s_{i,m})_j \approx \mathbb{I}[j = \arg\max_k s_{i,k}]
\tag{37}
\end{equation}

**问题**: 平均中心化改变了argmax，因此破坏了这个松弛关系。

### 17. 信息论解释

**定义17.1 (交叉熵)**

标准softmax交叉熵可以写为：
\begin{equation}
H(t, p) = -\sum_{i,j} t_{i,j} \log p_{i,j}
\tag{38}
\end{equation}

这是真实分布$t$和预测分布$p$之间的交叉熵。

**式(10)的解释**:
\begin{equation}
L_1 = \sum_j \log\left(1 + \sum_i \sum_{k \neq j} t_{i,j} e^{s_{i,k} - s_{i,j}}\right)
\tag{39}
\end{equation}

可以理解为"加权几何平均"的对数形式，但缺乏清晰的信息论解释。

### 18. Sigmoid vs Softmax

**对比18.1**:

| 特性 | Sigmoid | Softmax |
|------|---------|---------|
| 输出范围 | $[0,1]$ | $[0,1]$ |
| 归一化 | 独立 | $\sum_j p_j = 1$ |
| 决策边界 | $s > 0$ | $s_j = \max_k s_k$ |
| 多标签 | 自然支持 | 需要推广 |
| 类别不平衡 | 严重问题 | 相对较好 |

**关键差异**: Sigmoid允许多个类别同时为正，Softmax强制单选。

**$n$个$m$分类的挑战**: 需要在每个任务内保持softmax的归一化约束，同时在任务间处理不平衡。

### 19. 实践建议与权衡

**建议19.1 (何时使用各种损失)**

1. **硬标签 + 类别平衡**: 使用标准softmax交叉熵（式3）
2. **硬标签 + 类别不平衡**: 使用式(10)，可能需要调整
3. **软标签 + 需要解析解**: 可以尝试式(23)，但注意硬标签测试时的问题
4. **多标签（真正的）**: 使用式(2)或GlobalPointer

**建议19.2 (混合策略)**

可以考虑加权组合：
\begin{equation}
L_{hybrid} = \alpha L_1 + (1-\alpha) L_{standard}
\tag{40}
\end{equation}

其中$\alpha \in [0,1]$平衡类别不平衡调节和分类正确性。

### 20. 未来方向与开放问题

**开放问题20.1**:
- 是否存在同时满足以下条件的损失函数：
  1. 自动类别不平衡调节
  2. 软标签解析解
  3. 硬标签情况下保证argmax正确

**可能方向**:
1. **自适应权重**: 动态调整每个类别的权重
2. **多目标优化**: 同时优化软标签和硬标签目标
3. **结构化预测**: 利用任务间的相关性

**理论挑战**: 可能需要证明不存在"完美"解，或者需要引入额外的假设。

### 21. 总结

**本文贡献**:
1. 提出了两种"$n$个$m$分类"损失函数的候选形式
2. 分析了各自的优缺点和适用场景
3. 揭示了软标签优化和硬标签评估之间的内在矛盾

**关键洞察**:
- 类别不平衡调节与分类正确性存在权衡
- 平均中心化虽然理论优美，但在实践中可能失效
- 二分类的成功经验不能直接推广到$m$分类（$m>2$）

**实践启示**:
- 硬标签情况：式(10)可以尝试，但需要实验验证
- 软标签情况：式(23)有理论保证
- 实际应用：可能需要针对具体问题定制损失函数

