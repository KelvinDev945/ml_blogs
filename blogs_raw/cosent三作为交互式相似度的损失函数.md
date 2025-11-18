---
title: CoSENT（三）：作为交互式相似度的损失函数
slug: cosent三作为交互式相似度的损失函数
date: 2022-11-09
tags: 语义, 语义相似度, 对比学习, 生成模型, attention
status: pending
---

# CoSENT（三）：作为交互式相似度的损失函数

**原文链接**: [https://spaces.ac.cn/archives/9341](https://spaces.ac.cn/archives/9341)

**发布日期**: 

---

在[《CoSENT（一）：比Sentence-BERT更有效的句向量方案》](/archives/8847)中，笔者提出了名为“CoSENT”的有监督句向量方案，由于它是直接训练cos相似度的，跟评测目标更相关，因此通常能有着比Sentence-BERT更好的效果以及更快的收敛速度。在[《CoSENT（二）：特征式匹配与交互式匹配有多大差距？》](/archives/8860)中我们还比较过它跟交互式相似度模型的差异，显示它在某些任务上的效果还能直逼交互式相似度模型。

然而，当时笔者是一心想找一个更接近评测目标的Sentence-BERT替代品，所以结果都是面向有监督句向量的，即特征式相似度模型。最近笔者突然反应过来，CoSENT其实也能作为交互式相似度模型的损失函数。那么它跟标准选择交叉熵相比孰优孰劣呢？本文来补充这部分实验。

## 基础回顾 #

CoSENT提出之初，是作为一个有监督句向量的损失函数：  
\begin{equation}\log \left(1 + \sum\limits_{\text{sim}(i,j) \gt \text{sim}(k,l)} e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))}\right)\end{equation}  
其中$i,j,k,l$是四个训练样本（比如四个句子），$u_i, u_j, u_k, u_l$是它们想要学习的句向量（比如它们经过BERT后的[CLS]向量），$\cos(\cdot,\cdot)$代表两个向量的余弦相似度，$\text{sim}(\cdot,\cdot)$则代表它们的相似度标签。所以这个损失函数的定义也很清晰，就是如果你认为$(i,j)$的相似度应该大于$\text{sim}(k,l)$的相似度，那么就往$\log$里边加入一项$e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))}$。

从这个形式就可以看出，当时CoSETN就是为了有监督训练余弦相似度的特征式模型的，包括“CoSENT”这个名字也是这样来的（Cosine Sentence）。然而，抛开余弦相似度这一层面不谈，CoSENT本质上是一个只依赖于标签相对顺序的损失函数，它跟余弦相似度没有必然联系，我们可以将它一般化为  
\begin{equation}\log \left(1 + \sum\limits_{\text{sim}(i,j) \gt \text{sim}(k,l)} e^{\lambda(f(k,l) - f(i,j))}\right)\end{equation}  
其中$f(\cdot,\cdot)$是任意标量输出函数（一般不需要加激活函数），代表要学习的相似度模型，包括将两个输入拼接成一个文本输入到BERT中的“交互式相似度”模型！

## 实验比较 #

训练交互式相似度的常规方式是最后构建一个两节点的输出，然后加上softmax，用交叉熵（下表简称CE）作为损失函数，这也等价于在前面的$f(\cdot,\cdot)$上加sigmoid激活，然后用单节点的二分类交叉熵。不过这种做法也就适合二分类形式的标签，如果连续型的打分（比如STS-B是1～5分），就不大适合了，此时通常要转化为回归问题。但CoSENT没有这个限制，因为它只需要标签的序信息，这个特点跟常用的评测指标spearman系数是一致的。

两者的对比实验，参考代码如下：

> [**https://github.com/bojone/CoSENT/blob/main/accuracy/interact_cosent.py**](https://github.com/bojone/CoSENT/blob/main/accuracy/interact_cosent.py)

实验结果为  
$$\begin{array}{c}  
\text{评测指标为spearman系数} \\\  
{\begin{array}{c|ccccc}  
\hline  
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{avg}\\\  
\hline  
\text{BERT + CE} & 48.01 & 71.96 & 78.53 & 68.59 & 66.77 \\\  
\text{BERT + CoSENT} & 48.09 & 72.25 & 78.70 & 69.34 & 67.10 \\\  
\hline  
\text{RoBERTa + CE} & 49.70 & 73.20 & 79.13 & 70.52 & 68.14 \\\  
\text{RoBERTa + CoSENT} & 49.82 & 73.09 & 78.78 & 70.54 & 68.06 \\\  
\hline  
\end{array}} \\\  
\\\  
\text{评测指标为accuracy} \\\  
{\begin{array}{c|ccccc}  
\hline  
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{avg}\\\  
\hline  
\text{BERT + CE} & 85.38 & 83.57 & 88.10 & 81.45 & 84.63 \\\  
\text{BERT + CoSENT} & 85.55 & 83.73 & 87.92 & 81.85 & 84.76 \\\  
\hline  
\text{RoBERTa + CE} & 85.97 & 84.67 & 88.14 & 82.85 & 85.41 \\\  
\text{RoBERTa + CoSENT} & 86.06 & 84.23 & 88.14 & 83.03 & 85.37 \\\  
\hline  
\end{array}}  
\end{array}$$

可以看到，没有惊喜，CE和CoSENT的效果基本一致。非要挖掘一些细致区别的话，可以看到在BERT中，CoSENT的效果相对好些，在RoBERTa中基本没区别了，以及在PAWSX这个任务上，CoSENT的提升相对明显些，其他任务基本持平。如此，可以“弱弱地”下一个结论：

> 当模型较弱（BERT弱于RoBERTa）或者任务较难（PAWSX相对来说比其他三个任务都难）时，CoSENT** _或许_** 能取得比CE更好的效果。

注意，是“或许”，笔者也不能保证。实事求是地说，我也不认为两者构成什么显著差异。不过可以猜测，因为两种损失函数的形式有明显的差异，所以哪怕最终指标上差不多，模型内部应该也有一定差异，这时候或许可以考虑模型融合？

## 文章小结 #

本文主要思考和实验了CoSENT在交互式相似度模型中的可行性，最终结论是“可行但效果没什么提升”。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9341>_

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

苏剑林. (Nov. 09, 2022). 《CoSENT（三）：作为交互式相似度的损失函数 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9341>

@online{kexuefm-9341,  
title={CoSENT（三）：作为交互式相似度的损失函数},  
author={苏剑林},  
year={2022},  
month={Nov},  
url={\url{https://spaces.ac.cn/archives/9341}},  
} 


---

## 公式推导与注释

### 1. CoSENT损失函数的完整数学定义

CoSENT (Cosine Sentence) 损失函数的核心思想是直接优化余弦相似度的排序关系。给定样本对集合，我们希望相似样本对的余弦相似度大于不相似样本对的余弦相似度。

**基础定义**：设有$N$个样本$\{x_1, x_2, \ldots, x_N\}$，每个样本$x_i$经过编码器得到表示向量$u_i \in \mathbb{R}^d$，样本对$(i,j)$的相似度标签为$\text{sim}(i,j) \in \mathbb{R}$。

\begin{equation}
\mathcal{L}_{\text{CoSENT}} = \log \left(1 + \sum_{\substack{(i,j), (k,l) \\ \text{sim}(i,j) > \text{sim}(k,l)}} e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))}\right) \tag{1}
\end{equation}

其中：
- $\cos(u_i, u_j) = \frac{\langle u_i, u_j \rangle}{\|u_i\|_2 \|u_j\|_2}$ 是余弦相似度
- $\lambda > 0$ 是温度参数（scale factor）
- 求和遍历所有满足 $\text{sim}(i,j) > \text{sim}(k,l)$ 的样本对

**数学直觉**：
- 当$\cos(u_k, u_l) > \cos(u_i, u_j)$时（预测错误），指数项$e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))} > 1$，损失增大
- 当$\cos(u_k, u_l) < \cos(u_i, u_j)$时（预测正确），指数项$e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))} < 1$，损失接近0
- $\log(1 + \cdot)$ 提供平滑的惩罚

### 2. 与Circle Loss的深层联系

**Circle Loss回顾**：Circle Loss是一个统一的对比学习框架，其形式为：

\begin{equation}
\mathcal{L}_{\text{Circle}} = \log\left[1 + \sum_{j \in \Omega_{\text{neg}}} e^{\gamma(s_j - \Delta_n)} \cdot \sum_{i \in \Omega_{\text{pos}}} e^{-\gamma(s_i - \Delta_p)}\right] \tag{2}
\end{equation}

其中$s_i, s_j$是相似度得分，$\Delta_p, \Delta_n$是margin参数，$\gamma$是scale参数。

**CoSENT与Circle Loss的关系**：

将Circle Loss改写为pairwise形式：

\begin{equation}
\mathcal{L}_{\text{Circle}} = \log\left[1 + \sum_{\substack{i \in \Omega_{\text{pos}} \\ j \in \Omega_{\text{neg}}}} e^{\gamma(s_j - s_i + \Delta_p - \Delta_n)}\right] \tag{3}
\end{equation}

当设置$\Delta_p = \Delta_n = 0$时，Circle Loss退化为：

\begin{equation}
\mathcal{L}_{\text{Circle}}^* = \log\left[1 + \sum_{\substack{i \in \Omega_{\text{pos}} \\ j \in \Omega_{\text{neg}}}} e^{\gamma(s_j - s_i)}\right] \tag{4}
\end{equation}

这正是CoSENT的形式（其中$s_i = \cos(u_i, u_j)$）。

**关键差异**：
1. **相似度度量**：Circle Loss可用任意相似度函数，CoSENT特化为余弦相似度
2. **Margin**：Circle Loss显式使用margin，CoSENT通过温度参数$\lambda$隐式控制
3. **应用场景**：Circle Loss偏向于分类，CoSENT专注于句子表示学习

### 3. 与InfoNCE的对比分析

**InfoNCE损失**：在对比学习中广泛使用，定义为：

\begin{equation}
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{e^{\text{sim}(u_i, u_i^+)/\tau}}{\sum_{j=1}^{K} e^{\text{sim}(u_i, u_j)/\tau}} \tag{5}
\end{equation}

其中$u_i^+$是正样本，$\{u_j\}_{j=1}^K$包含1个正样本和$K-1$个负样本。

**推导InfoNCE与CoSENT的关系**：

对于batch中的$N$个样本，InfoNCE可以写成：

\begin{equation}
\mathcal{L}_{\text{InfoNCE}} = -\sum_{i=1}^{N} \log \frac{e^{s_{i,i^+}/\tau}}{e^{s_{i,i^+}/\tau} + \sum_{j \neq i^+} e^{s_{i,j}/\tau}} \tag{6}
\end{equation}

变形得：

\begin{equation}
\mathcal{L}_{\text{InfoNCE}} = \sum_{i=1}^{N} \log\left(1 + \sum_{j \neq i^+} e^{(s_{i,j} - s_{i,i^+})/\tau}\right) \tag{7}
\end{equation}

**对比总结**：

| 特性 | InfoNCE | CoSENT |
|------|---------|---------|
| 形式 | Softmax交叉熵 | 排序损失 |
| 负样本 | 需要大量负样本 | 利用batch内配对 |
| 标签类型 | 二元（正/负） | 连续相似度 |
| 优化目标 | 最大化互信息 | 最小化排序误差 |
| 计算复杂度 | $O(NK)$ | $O(N^2)$ |

### 4. 对比学习的理论基础

**互信息最大化**：对比学习的核心是最大化表示$u$与输入$x$的互信息$I(u; x)$。

根据互信息的定义：

\begin{equation}
I(u; x) = H(u) - H(u|x) = H(x) - H(x|u) \tag{8}
\end{equation}

其中$H(\cdot)$表示熵。在对比学习中，我们希望：
- **最大化** $I(u; x)$：表示保留输入信息
- **最小化** $I(u; y)$ where $y$是增强视图：表示对增强的不变性

**InfoNCE作为互信息下界**：

\begin{equation}
I(u; x) \geq \log K - \mathcal{L}_{\text{InfoNCE}} \tag{9}
\end{equation}

这个下界随着负样本数$K$的增加而变紧。

**CoSENT的隐式互信息优化**：

CoSENT通过排序损失间接优化互信息。考虑两个样本对$(u_i, u_j)$和$(u_k, u_l)$，如果$\text{sim}(i,j) > \text{sim}(k,l)$，那么我们希望：

\begin{equation}
\cos(u_i, u_j) > \cos(u_k, u_l) \Rightarrow I(u_i; u_j) > I(u_k; u_l) \tag{10}
\end{equation}

虽然CoSENT不直接优化互信息，但通过保持相似度的序关系，间接实现了互信息的排序。

**度量学习视角**：

从度量学习角度，CoSENT学习一个嵌入空间$\mathcal{U}$，使得：

\begin{equation}
d(u_i, u_j) < d(u_k, u_l) \quad \text{if} \quad \text{sim}(i,j) > \text{sim}(k,l) \tag{11}
\end{equation}

其中$d(u, v) = 1 - \cos(u, v)$是余弦距离。

### 5. 梯度计算与优化性质

**梯度推导**：

对于单个样本对$(i,j)$，设$s_{ij} = \cos(u_i, u_j)$，损失函数关于$s_{ij}$的梯度为：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial s_{ij}} = -\lambda \sum_{k,l: \text{sim}(i,j) > \text{sim}(k,l)} \frac{e^{\lambda(s_{kl} - s_{ij})}}{1 + \sum e^{\lambda(s_{kl} - s_{ij})}} \tag{12}
\end{equation}

可以改写为：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial s_{ij}} = -\lambda \sum_{k,l: \text{sim}(i,j) > \text{sim}(k,l)} p_{ij,kl} \tag{13}
\end{equation}

其中：

\begin{equation}
p_{ij,kl} = \frac{e^{\lambda(s_{kl} - s_{ij})}}{\sum_{\text{all pairs}} e^{\lambda(s_{kl} - s_{ij})}} \tag{14}
\end{equation}

是一个概率分布，表示样本对$(k,l)$相对于$(i,j)$的"错误程度"。

**余弦相似度的梯度**：

\begin{equation}
\frac{\partial \cos(u_i, u_j)}{\partial u_i} = \frac{u_j}{\|u_i\| \|u_j\|} - \frac{\langle u_i, u_j \rangle}{\|u_i\|^3 \|u_j\|} u_i \tag{15}
\end{equation}

简化为（当向量归一化时$\|u_i\| = 1$）：

\begin{equation}
\frac{\partial \cos(u_i, u_j)}{\partial u_i} = u_j - \cos(u_i, u_j) \cdot u_i \tag{16}
\end{equation}

**链式法则应用**：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial u_i} = \sum_{j} \frac{\partial \mathcal{L}}{\partial s_{ij}} \frac{\partial s_{ij}}{\partial u_i} = -\lambda \sum_{j} p_{ij} (u_j - s_{ij} u_i) \tag{17}
\end{equation}

**优化性质分析**：

1. **凸性**：CoSENT损失关于$\{s_{ij}\}$是凸函数（log-sum-exp形式）

   证明：设$f(s) = \log(1 + \sum_k e^{\lambda(a_k - s)})$，其Hessian矩阵为：

   \begin{equation}
   H = \frac{\partial^2 f}{\partial s^2} = \lambda^2 \frac{\sum_k e^{\lambda(a_k - s)} \cdot \sum_k e^{\lambda(a_k - s)} - (\sum_k e^{\lambda(a_k - s)})^2}{(1 + \sum_k e^{\lambda(a_k - s)})^2} \geq 0 \tag{18}
   \end{equation}

2. **Lipschitz连续性**：梯度的Lipschitz常数为$L = O(\lambda^2 N^2)$

   \begin{equation}
   \|\nabla \mathcal{L}(s) - \nabla \mathcal{L}(s')\| \leq L \|s - s'\| \tag{19}
   \end{equation}

3. **收敛速率**：使用SGD时，收敛速率为$O(1/\sqrt{T})$；使用Adam时，实际收敛更快

### 6. 信息论解释

**熵的视角**：

定义"预测分布"：

\begin{equation}
q_{ij,kl} = \frac{e^{\lambda s_{kl}}}{\sum_{m,n} e^{\lambda s_{mn}}} \tag{20}
\end{equation}

"目标分布"（基于标签）：

\begin{equation}
p_{ij,kl} = \begin{cases}
\frac{1}{|\mathcal{P}_{ij}|} & \text{if } \text{sim}(k,l) < \text{sim}(i,j) \\
0 & \text{otherwise}
\end{cases} \tag{21}
\end{equation}

其中$\mathcal{P}_{ij} = \{(k,l): \text{sim}(k,l) < \text{sim}(i,j)\}$。

**KL散度分解**：

CoSENT可以理解为最小化KL散度的上界：

\begin{equation}
\text{KL}(p \| q) = \sum_{ij,kl} p_{ij,kl} \log \frac{p_{ij,kl}}{q_{ij,kl}} \tag{22}
\end{equation}

**互信息界**：

表示$u$与标签$\text{sim}$的互信息满足：

\begin{equation}
I(u; \text{sim}) \geq H(\text{sim}) - \mathcal{L}_{\text{CoSENT}}/\lambda \tag{23}
\end{equation}

**条件熵**：

给定表示$u$，标签的条件熵为：

\begin{equation}
H(\text{sim}|u) \approx -\sum_{ij} p(\text{sim}_{ij}) \log \frac{e^{\lambda s_{ij}}}{\sum_{kl} e^{\lambda s_{kl}}} \tag{24}
\end{equation}

### 7. 概率论视角

**似然函数构造**：

将CoSENT解释为排序概率模型。给定相似度标签，样本对$(i,j)$排序高于$(k,l)$的概率为：

\begin{equation}
P(\text{rank}_{ij} > \text{rank}_{kl}) = \sigma(\lambda(s_{ij} - s_{kl})) = \frac{1}{1 + e^{-\lambda(s_{ij} - s_{kl})}} \tag{25}
\end{equation}

其中$\sigma(\cdot)$是sigmoid函数。

**负对数似然**：

\begin{equation}
-\log P(\text{all correct rankings}) = -\sum_{\text{sim}(i,j) > \text{sim}(k,l)} \log \sigma(\lambda(s_{ij} - s_{kl})) \tag{26}
\end{equation}

利用恒等式$-\log \sigma(x) = \log(1 + e^{-x})$：

\begin{equation}
= \sum_{\text{sim}(i,j) > \text{sim}(k,l)} \log(1 + e^{-\lambda(s_{ij} - s_{kl})}) \tag{27}
\end{equation}

当所有项求和在$\log$内时（Jensen不等式的应用）：

\begin{equation}
\sum_{ij,kl} \log(1 + e^{-\lambda(s_{ij} - s_{kl})}) \geq \log\left(1 + \sum_{ij,kl} e^{-\lambda(s_{ij} - s_{kl})}\right) \tag{28}
\end{equation}

等号成立当且仅当所有项相等。实际中，右侧的紧upper bound更容易优化，这就是CoSENT的形式。

**贝叶斯解释**：

假设先验$p(\text{sim}) \sim \mathcal{N}(0, \sigma^2)$，后验为：

\begin{equation}
p(\text{sim}_{ij} | u_i, u_j) \propto \exp\left(-\frac{(\text{sim}_{ij} - s_{ij})^2}{2\sigma^2}\right) \tag{29}
\end{equation}

CoSENT相当于最大后验估计（MAP）的排序版本。

### 8. 几何理解

**超球面上的优化**：

当向量归一化时，$u_i \in \mathbb{S}^{d-1}$（单位超球面）。余弦相似度$\cos(u_i, u_j)$等于球面上的测地距离的余弦：

\begin{equation}
\text{geodesic\_dist}(u_i, u_j) = \arccos(\cos(u_i, u_j)) \tag{30}
\end{equation}

**流形结构**：

CoSENT在超球面流形$\mathbb{S}^{d-1}$上定义优化问题。流形上的梯度为：

\begin{equation}
\text{grad}_{\mathbb{S}^{d-1}} \mathcal{L} = \nabla \mathcal{L} - \langle \nabla \mathcal{L}, u \rangle u \tag{31}
\end{equation}

这正是式(16)的形式！

**黎曼度量**：

超球面的黎曼度量为：

\begin{equation}
g_{ij} = \delta_{ij} - u_i u_j \tag{32}
\end{equation}

在此度量下，CoSENT优化的是测地距离的保序性。

**角度分布**：

优化后，相似样本对的夹角$\theta_{ij} = \arccos(s_{ij})$服从：

\begin{equation}
p(\theta_{ij}) \propto e^{-\lambda \text{rank}(\text{sim}_{ij})} \tag{33}
\end{equation}

即高相似度样本聚集在小角度区域。

**von Mises-Fisher分布**：

在超球面上，CoSENT隐式地学习了von Mises-Fisher (vMF)分布族：

\begin{equation}
p(u | \mu, \kappa) = C_d(\kappa) e^{\kappa \mu^T u} \tag{34}
\end{equation}

其中$\mu$是均值方向，$\kappa$是集中度参数（类似于$\lambda$）。

### 9. 与其他损失函数的对比

#### 9.1 Triplet Loss

**定义**：

\begin{equation}
\mathcal{L}_{\text{Triplet}} = \max(0, d(u_a, u_p) - d(u_a, u_n) + m) \tag{35}
\end{equation}

**对比**：

| 维度 | Triplet Loss | CoSENT |
|------|--------------|---------|
| 样本选择 | 需要精心挖掘triplet | 利用batch内所有配对 |
| Margin | 硬margin $m$ | 软margin（隐式） |
| 梯度 | ReLU截断 | 平滑梯度 |
| 收敛性 | 容易陷入局部最优 | 更稳定 |

#### 9.2 Contrastive Loss

**定义**：

\begin{equation}
\mathcal{L}_{\text{Contrastive}} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2 \tag{36}
\end{equation}

**对比**：CoSENT不需要设置margin，自动适应数据分布。

#### 9.3 ArcFace Loss

**定义**：

\begin{equation}
\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s\cos\theta_j}} \tag{37}
\end{equation}

**对比**：ArcFace用于分类（固定类别中心），CoSENT用于度量学习（样本间关系）。

### 10. 理论性质深入分析

**性质1：排序一致性**

\begin{equation}
\text{If } \mathcal{L}_{\text{CoSENT}} = 0, \text{ then } \forall (i,j), (k,l): \text{sim}(i,j) > \text{sim}(k,l) \Rightarrow s_{ij} > s_{kl} \tag{38}
\end{equation}

证明：当损失为0时，$\sum e^{\lambda(s_{kl} - s_{ij})} = 0$，因此所有$s_{kl} - s_{ij} \to -\infty$，即$s_{ij} > s_{kl}$。

**性质2：温度参数的作用**

温度$\lambda$控制损失的"尖锐程度"：

\begin{equation}
\lim_{\lambda \to \infty} \frac{1}{\lambda}\mathcal{L}_{\text{CoSENT}} = \max_{\text{sim}(i,j) > \text{sim}(k,l)}(s_{kl} - s_{ij})^+ \tag{39}
\end{equation}

其中$(x)^+ = \max(0, x)$。即$\lambda \to \infty$时，CoSENT退化为max-margin损失。

**性质3：样本权重**

每个违反排序的样本对$(k,l)$的权重为：

\begin{equation}
w_{kl} = \frac{e^{\lambda s_{kl}}}{\sum_{m,n} e^{\lambda s_{mn}}} \propto e^{\lambda s_{kl}} \tag{40}
\end{equation}

即相似度越高的负样本对权重越大，符合hard negative mining的思想。

**性质4：界的刻画**

CoSENT损失的上下界：

\begin{equation}
0 \leq \mathcal{L}_{\text{CoSENT}} \leq \log(1 + N_{\text{viol}} \cdot e^{2\lambda}) \tag{41}
\end{equation}

其中$N_{\text{viol}}$是违反排序的样本对数量。

### 11. 数值稳定性分析

**问题1：指数溢出**

当$\lambda(s_{kl} - s_{ij})$很大时，$e^{\lambda(s_{kl} - s_{ij})}$会溢出。

**解决方案**：Log-sum-exp技巧

\begin{equation}
\log\sum_k e^{x_k} = M + \log\sum_k e^{x_k - M}, \quad M = \max_k x_k \tag{42}
\end{equation}

应用到CoSENT：

\begin{equation}
\mathcal{L} = \log\left(1 + e^M \sum e^{\lambda(s_{kl} - s_{ij}) - M}\right) \tag{43}
\end{equation}

其中$M = \max_{k,l} \lambda(s_{kl} - s_{ij})$。

**问题2：余弦相似度的数值精度**

当$u_i$和$u_j$几乎平行时，$\|u_i\| \|u_j\|$可能很小，导致除法不稳定。

**解决方案**：添加epsilon

\begin{equation}
\cos(u_i, u_j) = \frac{\langle u_i, u_j \rangle}{\max(\|u_i\| \|u_j\|, \epsilon)}, \quad \epsilon = 10^{-8} \tag{44}
\end{equation}

**问题3：梯度消失/爆炸**

当$\lambda$过大或过小时，梯度可能消失或爆炸。

**监控指标**：

\begin{equation}
\text{gradient\_norm} = \sqrt{\sum_i \|\nabla_{u_i} \mathcal{L}\|^2} \tag{45}
\end{equation}

建议保持在$[0.1, 10]$范围内。

### 12. 具体计算示例

**示例设置**：

- Batch size = 4
- Embedding dim = 3
- 相似度标签：$(1,2): 0.9$, $(1,3): 0.2$, $(2,3): 0.1$, $(1,4): 0.5$

**步骤1：计算嵌入**

假设归一化后的嵌入为：

\begin{equation}
\begin{aligned}
u_1 &= [0.577, 0.577, 0.577] \\
u_2 &= [0.707, 0.707, 0] \\
u_3 &= [0, 0, 1] \\
u_4 &= [0.5, 0.5, 0.707]
\end{aligned} \tag{46}
\end{equation}

**步骤2：计算余弦相似度**

\begin{equation}
\begin{aligned}
s_{12} &= \langle u_1, u_2 \rangle = 0.577 \times 0.707 + 0.577 \times 0.707 = 0.816 \\
s_{13} &= \langle u_1, u_3 \rangle = 0.577 \\
s_{14} &= \langle u_1, u_4 \rangle = 0.816 \\
s_{23} &= \langle u_2, u_3 \rangle = 0
\end{aligned} \tag{47}
\end{equation}

**步骤3：识别违反排序的对**

根据$\text{sim}(1,2) > \text{sim}(1,4) > \text{sim}(1,3) > \text{sim}(2,3)$：

- 违反项：$(1,4)$的得分$s_{14} = 0.816$不应该等于$s_{12}$
- 期望：$s_{12} > s_{14} > s_{13} > s_{23}$

**步骤4：计算损失**（设$\lambda = 20$）

\begin{equation}
\begin{aligned}
\mathcal{L} &= \log\left(1 + e^{20(s_{14} - s_{12})} + e^{20(s_{13} - s_{12})} + \cdots\right) \\
&= \log\left(1 + e^{20(0.816 - 0.816)} + e^{20(0.577 - 0.816)} + \cdots\right) \\
&= \log\left(1 + e^{0} + e^{-4.78} + \cdots\right) \\
&= \log(1 + 1 + 0.0084 + \cdots) \\
&\approx \log(2.01) \approx 0.698
\end{aligned} \tag{48}
\end{equation}

**步骤5：计算梯度**

对$u_1$的梯度（针对$(1,2)$对）：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial u_1} = -20 \times 0.5 \times (u_2 - 0.816 \cdot u_1) \approx -10 \times [0.236, 0.236, -0.472] \tag{49}
\end{equation}

### 13. 实践建议与超参数调优

**温度参数$\lambda$的选择**：

\begin{equation}
\lambda = \frac{\alpha}{\sigma_s} \tag{50}
\end{equation}

其中$\sigma_s$是余弦相似度的标准差，$\alpha \in [10, 30]$是可调因子。

**实践建议**：

1. **初始化**：
   - 使用预训练模型（如BERT）初始化编码器
   - $\lambda$从20开始，根据验证集调整

2. **Batch构造**：
   - 每个batch包含多样化的相似度级别
   - Batch size建议：64-256

3. **学习率调度**：
   \begin{equation}
   \text{lr}(t) = \text{lr}_0 \times \min\left(1, \frac{t}{T_{\text{warmup}}}\right) \times \frac{1}{\sqrt{\max(t, T_{\text{warmup}})}} \tag{51}
   \end{equation}

4. **正则化**：
   - Weight decay: $10^{-5}$ to $10^{-4}$
   - Dropout: 0.1-0.3
   - Layer normalization在编码器输出

5. **评估指标**：
   - Spearman相关系数（主要）
   - Pearson相关系数
   - 余弦相似度分布的KS检验

6. **调试技巧**：
   - 监控相似度分布的均值和方差
   - 检查是否存在坍缩（所有向量变得相似）
   - 可视化t-SNE/UMAP降维结果

**常见问题与解决**：

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 损失不下降 | $\lambda$过小 | 增大$\lambda$ |
| 梯度爆炸 | $\lambda$过大 | 减小$\lambda$或梯度裁剪 |
| 表示坍缩 | 缺乏多样性 | 增加负样本多样性 |
| 过拟合 | 数据不足 | 数据增强、正则化 |

**数据增强策略**：

1. **回译**（Back-translation）
2. **同义词替换**
3. **句子重排**（对于长文本）
4. **Cutoff**（随机mask部分token）

### 14. 理论扩展与变体

**加权CoSENT**：

为不同的违反对分配不同权重：

\begin{equation}
\mathcal{L}_{\text{weighted}} = \log\left(1 + \sum w_{ij,kl} e^{\lambda(s_{kl} - s_{ij})}\right) \tag{52}
\end{equation}

其中$w_{ij,kl} = |\text{sim}(i,j) - \text{sim}(k,l)|$。

**多任务CoSENT**：

结合分类任务：

\begin{equation}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CoSENT}} + \alpha \mathcal{L}_{\text{CE}} \tag{53}
\end{equation}

**层次化CoSENT**：

在多个粒度上优化：

\begin{equation}
\mathcal{L}_{\text{hierarchical}} = \sum_{l=1}^{L} \beta_l \mathcal{L}_{\text{CoSENT}}^{(l)} \tag{54}
\end{equation}

其中$l$表示不同的语义层次（词、句、段落）。

### 15. 收敛性分析

**定理（收敛性）**：设学习率满足$\sum \eta_t = \infty$且$\sum \eta_t^2 < \infty$，则CoSENT的SGD优化序列$\{u^{(t)}\}$以概率1收敛到损失函数的稳定点。

**证明草图**：

1. CoSENT损失下有界（$\geq 0$）
2. 梯度有界（紧集$\mathbb{S}^{d-1}$上）
3. 应用随机近似理论的Robbins-Monro定理

**收敛速率**：

在强凸假设下（局部），收敛速率为：

\begin{equation}
\mathbb{E}[\mathcal{L}(u^{(T)})] - \mathcal{L}^* \leq \frac{C}{\sqrt{T}} \tag{55}
\end{equation}

实际中，由于Adam等自适应优化器，收敛更快。

---

**总结**：本节提供了CoSENT损失函数的完整数学推导，涵盖了从基础定义到高级理论的方方面面。通过与Circle Loss、InfoNCE等经典方法的对比，阐明了CoSENT在对比学习框架中的独特位置。详细的梯度推导、数值稳定性分析和实践建议为实际应用提供了坚实的理论基础和操作指南。

