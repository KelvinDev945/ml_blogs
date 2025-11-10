---
title: MoE环游记：2、不患寡而患不均
slug: moe环游记2不患寡而患不均
date: 2025-02-21
tags: 损失函数, 梯度, 稀疏, moe, 生成模型
status: pending
---

# MoE环游记：2、不患寡而患不均

**原文链接**: [https://spaces.ac.cn/archives/10735](https://spaces.ac.cn/archives/10735)

**发布日期**: 

---

在上一篇文章[《MoE环游记：1、从几何意义出发》](/archives/10699)中，我们介绍了MoE的一个几何诠释，旨在通过Dense模型的最佳逼近出发来推导和理解MoE。同时在文末我们也说了，给出MoE的计算公式仅仅是开始，训练一个实际有效的MoE模型还有很多细节补，比如本文要讨论的负载均衡（Load Balance）问题。

负载均衡，即“不患寡而患不均”，说白了就是让每个Expert都在干活，并且都在干尽可能一样多的活，避免某些Expert浪费算力。负载均衡既是充分利用训练算力的需求，也是尽可能发挥MoE大参数量潜力的需求。

## 需求分析 #

我们知道，MoE的基本形式是  
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\end{equation}  
对于传统MoE，$\boldsymbol{\rho}$是一个概率分布（Router），$\boldsymbol{e}_i=\boldsymbol{v}_i$，$\boldsymbol{v}_i$是一个小型FFN（Expert）的输出；而对于我们上一篇推导的几何MoE，$\boldsymbol{\rho}$没有归一化的要求，它预测的是Expert的模长，而$\boldsymbol{e}_i=\boldsymbol{v}_i/\Vert\boldsymbol{v}_i\Vert$预测的是Expert的方向。

不管哪种格式的MoE，实际表现都差不多，只是理解视角的不同。但要注意，虽然MoE的公式给人的感觉是“每遇到一个Token，就去找相应的Expert来计算”，但实际训练时其实是反过来的：先给每个Expert分配好相应的算力，然后将Token分配（Route）到所属的Expert中并行计算，这也就为什么负责打分的$\boldsymbol{\rho}$被称为Router。

这样一来，如果Expert的分配不均衡，就可能出现如下局面：某些Expert（Dead Expert）几乎一直闲置，浪费算力；某些Expert要处理的Token太多，根本忙不过来，只能Token Drop（即放弃处理部分Token）。从理论上来说，出现Dead Expert意味着MoE没有达到预期的参数量，即花了大参数量的显存，结果只训出来小参数量的效果。

所以，不管是从训练还是性能角度看，我们都希望保证Expert的负载均衡。

## 辅助损失 #

促进负载均衡的常规思路是添加与之相关的损失函数，我们通常称之为“Aux Loss（Auxiliary Loss）”，目前主流用的Aux Loss最早可以追溯到2020年的[《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》](https://papers.cool/arxiv/2006.16668)。

介绍Aux Loss之前，我们需要先引入一些新概念。首先，我们已经提到对于一般的MoE来说，$\boldsymbol{\rho}$未必是概率分布，我们将归一化的$\boldsymbol{\rho}$记为$\boldsymbol{p}=[p_1,p_2,\cdots,p_n]$，以及它Top-$k$版为$\boldsymbol{f}=[f_1,f_2,\cdots,f_n]$，其中  
\begin{equation}p_i = \frac{\rho_i}{\sum_{i=1}^n \rho_i},\qquad f_i = \left\\{\begin{aligned}1/k, \quad i\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho} \\\  
0, \quad i\not\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}\end{aligned}\right.\end{equation}  
接着我们定义$\boldsymbol{P}=\mathbb{E}[\boldsymbol{p}],\boldsymbol{F}=\mathbb{E}[\boldsymbol{f}]$，这里的$\mathbb{E}$是指对所有样本的所有Token做平均。不难看出，$\boldsymbol{F}$就是Expert当前的负载分布，而$\boldsymbol{P}$则相当于$\boldsymbol{F}$的一个光滑近似。

有了这些记号，我们就可以写出Aux Loss为：  
\begin{equation}\mathcal{L}_{\text{aux}} = \boldsymbol{F}\cdot \boldsymbol{P} = \sum_{i=1}^n F_i P_i\label{eq:aux-loss}\end{equation}  
一般文献定义Aux Loss会多乘一个$n$，即它们的Aux Loss等于这里的$n \mathcal{L}_{\text{aux}}$。此外，有些大型MoE可能会按设备来算Aux Loss，以达到设备内的均衡，减少设备间的通信，这些就各自发挥了。但也有较新的实验显示，强行局部均衡极有可能影响模型最终效果。

## 直通估计 #

不知道大家有没有发现一个奇怪的现象：不管是最早出处、后续文献还是科普文章，总之笔者阅读过的资料中，对Aux Loss的引用都是不加证明的，似乎大家都公认上述Aux Loss能促进均衡是一件显然成立的事情。可真有这么显然易得吗？

反正笔者是没看出来，所以接下来笔者给出式$\eqref{eq:aux-loss}$的一种推导思路，由此思路我们还可以自定义其他形式的Aux Loss。首先，定义均匀分布$\boldsymbol{Q}=(1/n,1/n,\cdots,1/n)$，刚才我们说了$\boldsymbol{F}$就是当前负载分布，因此负载均衡等价于$\boldsymbol{F}=\boldsymbol{Q}$，那么下式就是一个比较直观的Aux Loss：  
\begin{equation}\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (F_i - 1/n)^2\label{eq:aux-loss-2}\end{equation}  
问题是$\boldsymbol{F}$是由$\mathop{\text{argtop}}_k$出来的，这意味着上式并不是一个能直接用的可导目标。怎么解决这个问题呢？答案是[STE（Straight-Through Estimator）](/archives/6760#%E8%87%AA%E8%A1%8C%E8%AE%BE%E8%AE%A1%E6%A2%AF%E5%BA%A6)技巧，分别设计前向传播和反向传播的函数。具体来说，$\boldsymbol{F}$不可导，$\boldsymbol{P}$作为它的光滑近似是可导的，那么我们在反向传播的时候将$\boldsymbol{F}$替换成$\boldsymbol{P}$就行了，即  
\begin{equation}\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert \boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}] - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n)^2\label{eq:aux-loss-3}\end{equation}  
其中$\text{sg}[]$是stop gradient算子，特点是保持前向输出不变，但强制梯度为零。这样改动之后，$\mathcal{L}_{\text{aux}}$就是一个切实可行的Aux Loss了，我们可以试求一下它的梯度：  
\begin{equation}\begin{aligned}  
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{aux}} =&\, \frac{1}{2}\nabla_{\boldsymbol{\theta}}\sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n)^2 \\\  
=&\, \sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n) \nabla_{\boldsymbol{\theta}}(P_i + \text{sg}[F_i - P_i] - 1/n)\\\  
=&\, \sum_{i=1}^n (F_i - 1/n) \nabla_{\boldsymbol{\theta}}P_i = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n (F_i - 1/n) P_i\\\  
=&\, \nabla_{\boldsymbol{\theta}}\left(\sum_{i=1}^n F_i P_i\right)  
\end{aligned}\end{equation}  
这里$\boldsymbol{\theta}$是模型参数。最后的结果表明式$\eqref{eq:aux-loss-3}$的梯度等于式$\eqref{eq:aux-loss}$梯度，这意味着用式$\eqref{eq:aux-loss}$作为Aux Loss跟式$\eqref{eq:aux-loss-3}$在梯度上是等价的，所以就出现了式$\eqref{eq:aux-loss}$的Aux Loss。

然而，式$\eqref{eq:aux-loss}$只有等效梯度的意义，但没有Loss的意义，不算一个真正的Loss，比如当$\boldsymbol{F} = \boldsymbol{P}$时我们可以算出式$\eqref{eq:aux-loss}$等于$1/n$，但实际上我们可以构造出一个不等于$\boldsymbol{P}$的$\boldsymbol{F}$让它小于$1/n$，所以式$\eqref{eq:aux-loss}$并不是像正常的Loss一样越小越好，最小值也不是$\boldsymbol{F} = \boldsymbol{P}$时取到。

## 一般形式 #

上述推导实际上提供了构建Aux Loss的一般思路：**首先基于$\boldsymbol{F}$构建符合要求的损失，然后在实现时将$\boldsymbol{F}$替换成$\boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}]$。** 比如，我们知道最大熵也可以将分布推向均衡，因此也可以用熵的相反数来构建Aux Loss：  
\begin{equation}\mathcal{L}_{\text{aux}} = \sum_{i=1}^n (P_i + \text{sg}[F_i - P_i])\log(P_i + \text{sg}[F_i - P_i])\end{equation}  
上式就可以直接用作代码实现，当然如果我们追求简化，也可以类似地求梯度，结果将是  
\begin{equation}\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{aux}} = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n(P_i + \text{sg}[F_i - P_i]) \log(P_i + \text{sg}[F_i - P_i]) = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n P_i \log F_i\end{equation}  
两次简化梯度的过程中，我们都用到了如下恒等式  
\begin{equation}\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}P_i = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n P_i = \nabla_{\boldsymbol{\theta}}1 = \boldsymbol{0}\end{equation}  
这依赖于$\boldsymbol{P}$是一个概率分布，以及目标分布$\boldsymbol{Q}$是均匀分布的事实。而如果我们不追求简化后的等价结果，而是直接用$\boldsymbol{F}\to \boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}]$形式的Aux Loss，那么可以不受这两个约束。

比如，$\boldsymbol{P}$作为$\boldsymbol{F}$光滑近似这一点，我们只用到了“$P_i$大$F_i$通常也大”的性质，所以用非归一化的$\mathbb{E}[\boldsymbol{\rho}]$作为$\boldsymbol{P}$通常也没问题，这一点在一些特殊场景（例如有正有负的$\boldsymbol{\rho}$）可能会比较关键，因为此时无法归一化为概率分布。又比如目标$\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2$，显然能将$\boldsymbol{F}$推向任意我们想要的、不一定是均匀的目标分布$\boldsymbol{Q}$。

## 文章小结 #

本文介绍了MoE的负载均衡问题，并给出了一种构建Aux Loss的一般思路。除了Aux Loss外，促进负载均衡还有一些其他方案，我们下回再谈。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10735>_

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

苏剑林. (Feb. 21, 2025). 《MoE环游记：2、不患寡而患不均 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10735>

@online{kexuefm-10735,  
title={MoE环游记：2、不患寡而患不均},  
author={苏剑林},  
year={2025},  
month={Feb},  
url={\url{https://spaces.ac.cn/archives/10735}},  
} 


---

## 公式推导与注释

本节将对MoE负载均衡问题进行深入的数学推导与分析，包括MoE的数学定义、负载均衡损失的理论基础、均匀分布的必要性证明、熵正则化的作用机制、不同平衡策略的数学对比，以及收敛性与稳定性的理论分析。

### 1. MoE的数学定义与路由机制

#### 1.1 基本MoE架构的数学表示

对于一个包含$n$个专家的MoE系统，给定输入$\boldsymbol{x} \in \mathbb{R}^d$，MoE的输出定义为：

$$
\boldsymbol{y} = \sum_{i=1}^{n} g_i(\boldsymbol{x}) \boldsymbol{e}_i(\boldsymbol{x})
$$

其中：
- $g_i(\boldsymbol{x})$是路由网络（Router）的输出，决定第$i$个专家的权重
- $\boldsymbol{e}_i(\boldsymbol{x})$是第$i$个专家网络的输出
- $\sum_{i=1}^{n} g_i(\boldsymbol{x}) = 1$且$g_i(\boldsymbol{x}) \geq 0$（概率分布约束）

**推导1：Router的Softmax表示**

Router通常通过一个线性变换后接Softmax实现：

$$
\boldsymbol{\rho} = \boldsymbol{W}_r \boldsymbol{x} + \boldsymbol{b}_r
$$

其中$\boldsymbol{W}_r \in \mathbb{R}^{n \times d}$，$\boldsymbol{b}_r \in \mathbb{R}^n$。然后对$\boldsymbol{\rho}$应用Softmax：

$$
g_i(\boldsymbol{x}) = \frac{e^{\rho_i}}{\sum_{j=1}^{n} e^{\rho_j}}
$$

这保证了$\sum_{i=1}^{n} g_i(\boldsymbol{x}) = 1$。

#### 1.2 Top-K稀疏化MoE

为了减少计算量，实际应用中采用Top-K稀疏化：

$$
\boldsymbol{y} = \sum_{i \in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i
$$

**推导2：Top-K操作的数学定义**

定义Top-K选择函数：

$$
\mathcal{I}_k(\boldsymbol{\rho}) = \{i_1, i_2, \ldots, i_k\} \quad \text{满足} \quad \rho_{i_1} \geq \rho_{i_2} \geq \cdots \geq \rho_{i_k} \geq \rho_j, \forall j \notin \mathcal{I}_k
$$

则Top-K稀疏MoE可以重写为：

$$
\boldsymbol{y} = \sum_{i=1}^{n} \mathbb{1}_{i \in \mathcal{I}_k(\boldsymbol{\rho})} \cdot \rho_i \boldsymbol{e}_i
$$

其中$\mathbb{1}_{i \in \mathcal{I}_k(\boldsymbol{\rho})}$是指示函数。

#### 1.3 几何MoE的解释

**推导3：几何视角下的MoE分解**

在几何MoE中，我们将专家输出分解为模长和方向：

$$
\boldsymbol{e}_i = \rho_i \frac{\boldsymbol{v}_i}{\|\boldsymbol{v}_i\|}
$$

其中：
- $\rho_i$预测的是专家贡献的幅度（模长）
- $\frac{\boldsymbol{v}_i}{\|\boldsymbol{v}_i\|}$预测的是专家贡献的方向（单位向量）

这样，MoE的输出变为：

$$
\boldsymbol{y} = \sum_{i \in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \frac{\boldsymbol{v}_i}{\|\boldsymbol{v}_i\|}
$$

**推导4：非归一化Router的有效性**

几何MoE不要求$\boldsymbol{\rho}$归一化，因为我们只关心相对大小：

$$
\mathop{\text{argtop}}_k \boldsymbol{\rho} = \mathop{\text{argtop}}_k (c\boldsymbol{\rho}) \quad \forall c > 0
$$

这给予了模型更大的表达自由度。

### 2. 负载均衡问题的数学分析

#### 2.1 负载分布的定义

**推导5：归一化Router分数$\boldsymbol{p}$**

对于任意Router输出$\boldsymbol{\rho}$，定义归一化版本：

$$
p_i = \frac{\rho_i}{\sum_{j=1}^{n} \rho_j}
$$

显然$\sum_{i=1}^{n} p_i = 1$，$\boldsymbol{p}$形成一个概率分布。

**推导6：Top-K离散分布$\boldsymbol{f}$**

定义Top-K后的离散分布：

$$
f_i = \begin{cases}
\frac{1}{k}, & i \in \mathop{\text{argtop}}_k \boldsymbol{\rho} \\
0, & i \notin \mathop{\text{argtop}}_k \boldsymbol{\rho}
\end{cases}
$$

验证归一化：$\sum_{i=1}^{n} f_i = k \cdot \frac{1}{k} = 1$ ✓

**推导7：全局负载分布**

对所有样本的所有Token求平均，得到期望负载分布：

$$
\boldsymbol{F} = \mathbb{E}_{\text{tokens}}[\boldsymbol{f}], \quad \boldsymbol{P} = \mathbb{E}_{\text{tokens}}[\boldsymbol{p}]
$$

其中$\boldsymbol{P}$是$\boldsymbol{F}$的光滑近似，因为：

$$
\lim_{\tau \to 0} \boldsymbol{P}(\boldsymbol{\rho}/\tau) = \boldsymbol{F}(\boldsymbol{\rho})
$$

#### 2.2 负载不均衡的代价

**推导8：Dead Expert的参数浪费**

如果专家$i$的负载$F_i \approx 0$，则该专家的参数$\boldsymbol{\theta}_i$几乎不参与训练：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_i} = \mathbb{E}\left[\frac{\partial \mathcal{L}}{\partial \boldsymbol{e}_i} \frac{\partial \boldsymbol{e}_i}{\partial \boldsymbol{\theta}_i}\right] \approx 0 \quad \text{当} \quad F_i \to 0
$$

设模型总参数为$M$，若$m$个专家处于Dead状态，则有效参数量降为：

$$
M_{\text{eff}} = M - m \cdot |\boldsymbol{\theta}_i| \ll M
$$

**推导9：Token Drop的损失**

当某个专家负载过高时，会发生Token Drop。设专家$i$的容量为$C_i$，但分配的Token数为$T_i > C_i$，则丢弃比例为：

$$
\text{Drop Rate}_i = \frac{T_i - C_i}{T_i} = 1 - \frac{C_i}{T_i}
$$

丢弃Token会导致信息损失，影响模型性能。

### 3. 辅助损失（Aux Loss）的数学推导

#### 3.1 标准Aux Loss的形式

**推导10：GShard Aux Loss**

GShard提出的辅助损失为：

$$
\mathcal{L}_{\text{aux}} = n \sum_{i=1}^{n} F_i P_i
$$

或不带系数$n$的版本（本文采用）：

$$
\mathcal{L}_{\text{aux}} = \sum_{i=1}^{n} F_i P_i = \boldsymbol{F} \cdot \boldsymbol{P}
$$

**推导11：为什么这个形式能促进均衡？**

定义均匀分布$\boldsymbol{Q} = (\frac{1}{n}, \frac{1}{n}, \ldots, \frac{1}{n})$。如果$\boldsymbol{F} = \boldsymbol{Q}$（完全均衡），则：

$$
\mathcal{L}_{\text{aux}} = \sum_{i=1}^{n} \frac{1}{n} P_i = \frac{1}{n} \sum_{i=1}^{n} P_i = \frac{1}{n}
$$

但这不是最小值！考虑极端不均衡情况，如$F_1 = 1, F_i = 0 (i>1)$，此时：

$$
\mathcal{L}_{\text{aux}} = P_1 \leq 1
$$

所以$\mathcal{L}_{\text{aux}}$本身不是标准的"越小越好"的损失。

#### 3.2 通过直通估计器（STE）理解Aux Loss

**推导12：基于均方误差的真实损失**

更自然的负载均衡损失应该是：

$$
\mathcal{L}_{\text{true}} = \frac{1}{2} \sum_{i=1}^{n} (F_i - Q_i)^2 = \frac{1}{2} \sum_{i=1}^{n} \left(F_i - \frac{1}{n}\right)^2
$$

这确实是"越小越好"的损失，最小值为0（当$\boldsymbol{F} = \boldsymbol{Q}$时）。

**推导13：STE替换$\boldsymbol{F} \to \boldsymbol{P}$**

由于$\boldsymbol{F}$包含不可微的$\text{argtop}_k$操作，我们用STE技巧：

$$
\tilde{F}_i = P_i + \text{sg}[F_i - P_i]
$$

其中$\text{sg}[\cdot]$是stop gradient算子：
- 前向传播：$\text{sg}[x] = x$
- 反向传播：$\frac{\partial \text{sg}[x]}{\partial x} = 0$

将$F_i$替换为$\tilde{F}_i$：

$$
\mathcal{L}_{\text{ste}} = \frac{1}{2} \sum_{i=1}^{n} \left(P_i + \text{sg}[F_i - P_i] - \frac{1}{n}\right)^2
$$

**推导14：梯度推导**

计算$\mathcal{L}_{\text{ste}}$关于$\boldsymbol{\theta}$的梯度：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{ste}} &= \sum_{i=1}^{n} \left(P_i + \text{sg}[F_i - P_i] - \frac{1}{n}\right) \nabla_{\boldsymbol{\theta}} (P_i + \text{sg}[F_i - P_i]) \\
&= \sum_{i=1}^{n} \left(P_i + \text{sg}[F_i - P_i] - \frac{1}{n}\right) \nabla_{\boldsymbol{\theta}} P_i \\
&= \sum_{i=1}^{n} \left(F_i - \frac{1}{n}\right) \nabla_{\boldsymbol{\theta}} P_i
\end{aligned}
$$

在第二步我们用了$\nabla_{\boldsymbol{\theta}} \text{sg}[x] = 0$。

**推导15：简化为点积形式**

继续推导：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{ste}} &= \sum_{i=1}^{n} F_i \nabla_{\boldsymbol{\theta}} P_i - \frac{1}{n} \sum_{i=1}^{n} \nabla_{\boldsymbol{\theta}} P_i \\
&= \sum_{i=1}^{n} F_i \nabla_{\boldsymbol{\theta}} P_i - \frac{1}{n} \nabla_{\boldsymbol{\theta}} \left(\sum_{i=1}^{n} P_i\right) \\
&= \sum_{i=1}^{n} F_i \nabla_{\boldsymbol{\theta}} P_i \quad (\text{因为} \sum_{i=1}^{n} P_i = 1) \\
&= \nabla_{\boldsymbol{\theta}} \sum_{i=1}^{n} F_i P_i
\end{aligned}
$$

这证明了$\mathcal{L}_{\text{aux}} = \sum_{i=1}^{n} F_i P_i$的梯度等价性！

### 4. 均匀分布的理论分析

#### 4.1 均匀分布的信息论意义

**推导16：熵与均匀分布**

分布$\boldsymbol{F}$的熵定义为：

$$
H(\boldsymbol{F}) = -\sum_{i=1}^{n} F_i \log F_i
$$

对于固定的$n$，熵在均匀分布时达到最大：

$$
H(\boldsymbol{Q}) = -\sum_{i=1}^{n} \frac{1}{n} \log \frac{1}{n} = \log n
$$

**推导17：熵最大性的证明**

使用拉格朗日乘数法，要最大化$H(\boldsymbol{F})$约束于$\sum_{i=1}^{n} F_i = 1$：

$$
\mathcal{L} = -\sum_{i=1}^{n} F_i \log F_i + \lambda \left(\sum_{i=1}^{n} F_i - 1\right)
$$

对$F_i$求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial F_i} = -\log F_i - 1 + \lambda = 0
$$

得到$\log F_i = \lambda - 1$，即$F_i = e^{\lambda - 1}$对所有$i$都相同。结合约束条件$\sum_{i=1}^{n} F_i = 1$，得到$F_i = \frac{1}{n}$。

**推导18：KL散度与均匀分布**

从当前分布$\boldsymbol{F}$到目标均匀分布$\boldsymbol{Q}$的KL散度：

$$
D_{KL}(\boldsymbol{F} \| \boldsymbol{Q}) = \sum_{i=1}^{n} F_i \log \frac{F_i}{Q_i} = \sum_{i=1}^{n} F_i \log(n F_i)
$$

展开：

$$
D_{KL}(\boldsymbol{F} \| \boldsymbol{Q}) = \log n + \sum_{i=1}^{n} F_i \log F_i = \log n - H(\boldsymbol{F})
$$

所以最大化熵等价于最小化KL散度。

#### 4.2 均匀分布的容量利用

**推导19：有效容量分析**

假设每个专家容量为$C$，总容量$C_{\text{total}} = nC$。在分布$\boldsymbol{F}$下，期望利用的容量为：

$$
C_{\text{used}} = \sum_{i=1}^{n} \min(F_i \cdot T, C)
$$

其中$T$是总Token数。当$\boldsymbol{F}$均匀时，$F_i = \frac{1}{n}$，如果$\frac{T}{n} \leq C$，则：

$$
C_{\text{used}} = \sum_{i=1}^{n} \frac{T}{n} = T
$$

所有Token都被处理，无Drop。

**推导20：不均匀分布的容量损失**

假设分布为$(F_1, F_2, \ldots, F_n) = (0.5, 0.5, 0, \ldots, 0)$（两个专家分担所有负载）。设$C = \frac{T}{n}$，则：

$$
C_{\text{used}} = 2 \cdot \min(0.5T, C) = 2C = \frac{2T}{n}
$$

浪费了$\frac{n-2}{n}$的容量！当$n$很大时，这是巨大的浪费。

### 5. 熵正则化的理论分析

#### 5.1 基于熵的Aux Loss

**推导21：熵正则化损失**

另一种促进均衡的方式是直接最大化熵（或最小化负熵）：

$$
\mathcal{L}_{\text{entropy}} = -H(\boldsymbol{F}) = \sum_{i=1}^{n} F_i \log F_i
$$

使用STE技巧：

$$
\mathcal{L}_{\text{entropy-ste}} = \sum_{i=1}^{n} (P_i + \text{sg}[F_i - P_i]) \log(P_i + \text{sg}[F_i - P_i])
$$

**推导22：熵损失的梯度**

计算梯度：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{entropy-ste}} &= \sum_{i=1}^{n} \nabla_{\boldsymbol{\theta}} [(P_i + \text{sg}[F_i - P_i]) \log(P_i + \text{sg}[F_i - P_i])] \\
&= \sum_{i=1}^{n} (\log(P_i + \text{sg}[F_i - P_i]) + 1) \nabla_{\boldsymbol{\theta}} P_i \\
&= \sum_{i=1}^{n} (\log F_i + 1) \nabla_{\boldsymbol{\theta}} P_i
\end{aligned}
$$

**推导23：梯度简化**

利用$\sum_{i=1}^{n} \nabla_{\boldsymbol{\theta}} P_i = 0$：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{entropy-ste}} &= \sum_{i=1}^{n} \log F_i \nabla_{\boldsymbol{\theta}} P_i + \sum_{i=1}^{n} \nabla_{\boldsymbol{\theta}} P_i \\
&= \sum_{i=1}^{n} \log F_i \nabla_{\boldsymbol{\theta}} P_i \\
&= \nabla_{\boldsymbol{\theta}} \sum_{i=1}^{n} P_i \log F_i
\end{aligned}
$$

所以可以使用简化形式：

$$
\mathcal{L}_{\text{entropy-simple}} = \sum_{i=1}^{n} P_i \log F_i
$$

#### 5.2 熵正则与点积形式的关系

**推导24：泰勒展开分析**

在$\boldsymbol{F} \approx \boldsymbol{Q}$附近，对$F_i = \frac{1}{n} + \epsilon_i$展开：

$$
\log F_i = \log\left(\frac{1}{n} + \epsilon_i\right) = \log \frac{1}{n} + \log\left(1 + n\epsilon_i\right) \approx \log \frac{1}{n} + n\epsilon_i
$$

代入熵损失：

$$
\begin{aligned}
\mathcal{L}_{\text{entropy-simple}} &= \sum_{i=1}^{n} P_i \log F_i \\
&\approx \sum_{i=1}^{n} P_i \left(\log \frac{1}{n} + n\epsilon_i\right) \\
&= -\log n + n \sum_{i=1}^{n} P_i \epsilon_i
\end{aligned}
$$

而点积形式在同样的展开下：

$$
\sum_{i=1}^{n} F_i P_i = \sum_{i=1}^{n} \left(\frac{1}{n} + \epsilon_i\right) P_i = \frac{1}{n} + \sum_{i=1}^{n} \epsilon_i P_i
$$

两者在一阶近似下本质相同！

### 6. 不同平衡策略的数学对比

#### 6.1 L2正则化策略

**推导25：L2距离损失**

$$
\mathcal{L}_{L2} = \frac{1}{2} \|\boldsymbol{F} - \boldsymbol{Q}\|^2 = \frac{1}{2} \sum_{i=1}^{n} \left(F_i - \frac{1}{n}\right)^2
$$

展开：

$$
\mathcal{L}_{L2} = \frac{1}{2} \sum_{i=1}^{n} F_i^2 - \frac{1}{n} \sum_{i=1}^{n} F_i + \frac{1}{2n}
$$

由于$\sum_{i=1}^{n} F_i = 1$，简化为：

$$
\mathcal{L}_{L2} = \frac{1}{2} \sum_{i=1}^{n} F_i^2 - \frac{1}{n} + \frac{1}{2n} = \frac{1}{2} \sum_{i=1}^{n} F_i^2 - \frac{1}{2n}
$$

**推导26：L2损失与点积形式的关系**

注意到：

$$
\sum_{i=1}^{n} F_i^2 = \sum_{i=1}^{n} F_i \cdot F_i
$$

如果$\boldsymbol{P} \approx \boldsymbol{F}$，则：

$$
\mathcal{L}_{L2} \approx \frac{1}{2} \sum_{i=1}^{n} F_i P_i - \frac{1}{2n}
$$

相差一个常数项！

#### 6.2 L1正则化策略

**推导27：L1距离损失**

$$
\mathcal{L}_{L1} = \|\boldsymbol{F} - \boldsymbol{Q}\|_1 = \sum_{i=1}^{n} \left|F_i - \frac{1}{n}\right|
$$

这个损失对离群专家（$F_i$远离$\frac{1}{n}$）的惩罚更强。

**推导28：L1的次梯度**

L1损失的次梯度为：

$$
\partial \mathcal{L}_{L1} = \sum_{i=1}^{n} \text{sign}\left(F_i - \frac{1}{n}\right) \nabla_{\boldsymbol{\theta}} P_i
$$

其中$\text{sign}(x) = \begin{cases} 1, & x > 0 \\ [-1, 1], & x = 0 \\ -1, & x < 0 \end{cases}$

#### 6.3 最大负载最小化

**推导29：Minimax目标**

另一种策略是最小化最大负载：

$$
\mathcal{L}_{\max} = \max_{i=1,\ldots,n} F_i
$$

这直接针对"最繁忙"的专家。可以用光滑近似：

$$
\mathcal{L}_{\max-soft} = \text{LogSumExp}(\boldsymbol{F}) = \log \sum_{i=1}^{n} e^{F_i}
$$

当温度系数很大时，这接近$\max F_i$。

### 7. 收敛性与稳定性分析

#### 7.1 Aux Loss的收敛性

**推导30：梯度下降更新**

设总损失为：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{aux}}
$$

其中$\alpha$是Aux Loss的权重。梯度下降更新：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{total}}
$$

**推导31：收敛条件分析**

假设$\mathcal{L}_{\text{task}}$是$L$-光滑的，即：

$$
\|\nabla \mathcal{L}_{\text{task}}(\boldsymbol{\theta}_1) - \nabla \mathcal{L}_{\text{task}}(\boldsymbol{\theta}_2)\| \leq L \|\boldsymbol{\theta}_1 - \boldsymbol{\theta}_2\|
$$

那么在学习率$\eta \leq \frac{1}{L}$下，梯度下降收敛。

对于Aux Loss项$\alpha \sum_{i=1}^{n} F_i P_i$，其梯度为：

$$
\alpha \sum_{i=1}^{n} F_i \nabla_{\boldsymbol{\theta}} P_i
$$

由于$P_i = \frac{e^{\rho_i}}{\sum_j e^{\rho_j}}$是Softmax，其Jacobian有界，因此Aux Loss也是光滑的。

**推导32：稳定性分析——方差界**

负载分布的方差定义为：

$$
\text{Var}(\boldsymbol{F}) = \sum_{i=1}^{n} \left(F_i - \frac{1}{n}\right)^2 = \sum_{i=1}^{n} F_i^2 - \frac{1}{n}
$$

最小化Aux Loss $\sum_{i=1}^{n} F_i P_i$时，如果$\boldsymbol{P} \approx \boldsymbol{F}$，则相当于最小化$\sum_{i=1}^{n} F_i^2$，即减小方差。

**推导33：Lyapunov稳定性**

定义Lyapunov函数：

$$
V(\boldsymbol{F}) = \frac{1}{2} \|\boldsymbol{F} - \boldsymbol{Q}\|^2
$$

其时间导数（沿梯度流）：

$$
\frac{dV}{dt} = (\boldsymbol{F} - \boldsymbol{Q})^T \frac{d\boldsymbol{F}}{dt}
$$

如果优化正确进行，$\frac{d\boldsymbol{F}}{dt}$应该指向减小$V$的方向，即：

$$
\frac{dV}{dt} < 0 \quad \text{当} \quad \boldsymbol{F} \neq \boldsymbol{Q}
$$

这保证了系统收敛到均匀分布。

#### 7.2 动态稳定性

**推导34：负载波动分析**

在训练过程中，负载$F_i^{(t)}$在时间步$t$可能有波动。定义波动幅度：

$$
\Delta F_i^{(t)} = F_i^{(t)} - F_i^{(t-1)}
$$

总波动为：

$$
\Omega^{(t)} = \sum_{i=1}^{n} |\Delta F_i^{(t)}|
$$

稳定的训练应该有$\Omega^{(t)} \to 0$当$t \to \infty$。

**推导35：指数移动平均的稳定效果**

为了减少波动，可以使用EMA更新$\boldsymbol{F}$：

$$
\boldsymbol{F}^{(t)} = \beta \boldsymbol{F}^{(t-1)} + (1-\beta) \boldsymbol{f}^{(t)}
$$

其中$\boldsymbol{f}^{(t)}$是第$t$步的瞬时负载，$\beta \in [0, 1)$是动量系数。

这个EMA的方差为：

$$
\text{Var}(\boldsymbol{F}^{(t)}) = (1-\beta)^2 \sum_{s=0}^{t} \beta^{2s} \text{Var}(\boldsymbol{f}^{(t-s)})
$$

当$t$足够大时，稳态方差为：

$$
\text{Var}(\boldsymbol{F}^{(\infty)}) = \frac{1-\beta}{1+\beta} \text{Var}(\boldsymbol{f})
$$

$\beta$越大，方差越小，越稳定。

### 8. 实验结果的理论解释

#### 8.1 Aux Loss权重的影响

**推导36：权重$\alpha$的权衡**

总损失为：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{aux}}
$$

过小的$\alpha$：负载不均衡，$\boldsymbol{F}$偏离$\boldsymbol{Q}$
过大的$\alpha$：过度强调均衡，牺牲任务性能

最优$\alpha^*$应平衡两者：

$$
\alpha^* = \arg\min_{\alpha} \mathbb{E}[\mathcal{L}_{\text{task}}(\alpha)] + \lambda \|\boldsymbol{F}(\alpha) - \boldsymbol{Q}\|^2
$$

其中$\lambda$是超参数，控制对均衡的重视程度。

**推导37：帕累托前沿分析**

定义两个目标：
- $J_1 = \mathcal{L}_{\text{task}}$（任务性能，越小越好）
- $J_2 = \|\boldsymbol{F} - \boldsymbol{Q}\|^2$（负载方差，越小越好）

帕累托最优解满足：不存在其他解能同时改善$J_1$和$J_2$。

不同的$\alpha$对应帕累托前沿上的不同点。

#### 8.2 训练动态的理论预测

**推导38：早期训练阶段**

初始时，Router权重接近随机，$\boldsymbol{P}^{(0)} \approx \boldsymbol{Q}$（近似均匀）。此时：

$$
\mathcal{L}_{\text{aux}}^{(0)} = \sum_{i=1}^{n} F_i^{(0)} P_i^{(0)} \approx \sum_{i=1}^{n} F_i^{(0)} \cdot \frac{1}{n} = \frac{1}{n}
$$

随着训练，Router开始专门化，$\boldsymbol{P}$偏离均匀，如果没有Aux Loss，$\boldsymbol{F}$也会偏离。

**推导39：中期训练阶段**

Aux Loss开始发挥作用，将$\boldsymbol{F}$拉向$\boldsymbol{Q}$。此时存在竞争：
- 任务损失希望Router专门化
- Aux Loss希望Router均匀化

平衡点取决于$\alpha$。

**推导40：后期训练阶段**

当模型接近收敛，$\boldsymbol{F}$应该稳定在接近$\boldsymbol{Q}$的位置。此时Aux Loss的梯度：

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{aux}} = \sum_{i=1}^{n} F_i \nabla P_i \approx \frac{1}{n} \sum_{i=1}^{n} \nabla P_i = 0
$$

Aux Loss的影响减弱，模型主要优化任务损失。

### 9. 总结与扩展

通过以上40个详细推导，我们从数学角度深入理解了MoE负载均衡问题：

1. **MoE架构**：建立了严格的数学定义，包括标准MoE和几何MoE
2. **负载问题**：量化了负载不均衡的代价（Dead Expert、Token Drop）
3. **Aux Loss设计**：通过STE技巧推导了标准Aux Loss的梯度等价性
4. **均匀分布**：从信息论（熵）和容量利用角度证明了均匀分布的必要性
5. **熵正则化**：展示了基于熵的Aux Loss与点积形式的联系
6. **策略对比**：比较了L1、L2、熵、Minimax等不同平衡策略
7. **收敛性**：分析了Aux Loss的收敛条件和Lyapunov稳定性
8. **实验理论**：解释了Aux Loss权重、训练动态等实验现象

这些推导为理解和改进MoE负载均衡提供了坚实的理论基础。

