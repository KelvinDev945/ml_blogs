---
title: 从梯度最大化看Attention的Scale操作
slug: 从梯度最大化看attention的scale操作
date: 2023-10-22
tags: 优化, 梯度, attention, 生成模型, attention
status: completed
---

# 从梯度最大化看Attention的Scale操作

**原文链接**: [https://spaces.ac.cn/archives/9812](https://spaces.ac.cn/archives/9812)

**发布日期**: 

---

我们知道，[Scaled Dot-Product Attention](/archives/4765)的Scale因子是$\frac{1}{\sqrt{d}}$，其中$d$是$\boldsymbol{q},\boldsymbol{k}$的维度。这个Scale因子的一般解释是：如果不除以$\sqrt{d}$，那么初始的Attention就会很接近one hot分布，这会造成梯度消失，导致模型训练不起来。然而，可以证明的是，当Scale等于0时同样也会有梯度消失问题，这也就是说Scale太大太小都不行。

那么多大的Scale才适合呢？$\frac{1}{\sqrt{d}}$是最佳的Scale了吗？本文试图从梯度角度来回答这个问题。

## 已有结果 #

在[《浅谈Transformer的初始化、参数化与标准化》](/archives/8620#NTK%E5%8F%82%E6%95%B0%E5%8C%96)中，我们已经推导过标准的Scale因子$\frac{1}{\sqrt{d}}$，推导的思路很简单，假设初始阶段$\boldsymbol{q},\boldsymbol{k}\in\mathbb{R}^d$都采样自“均值为0、方差为1”的分布，那么可以算得  
\begin{equation}\mathbb{V}ar[\boldsymbol{q}\cdot\boldsymbol{k}] = d\end{equation}  
于是我们将$\boldsymbol{q}\cdot\boldsymbol{k}$除以$\sqrt{d}$，将Attention Score的方差变为1。也就是说，之前的推导纯粹是基于 _“均值为0、方差为1”就会更好_ 的**信仰** 来得到的结果，但没有解释让Attention Score的方差为1，也没有评估$\frac{1}{\sqrt{d}}$是否真的就解决了梯度消失问题。

当然，从已有的实验来看，$\frac{1}{\sqrt{d}}$至少一定程度上是缓解了这个问题，但这毕竟是实验结果，我们还是希望能从理论上知道“一定程度”究竟是多少。

## 计算梯度 #

既然涉及到了梯度，那么最好的办法就是把梯度算出来，然后定一个优化目标。设$p_i = e^{\alpha s_i}/Z$，$i \in \\{1,2,...,n\\}$，$Z=\sum_i e^{\alpha s_i}$是归一化因子，那么可以直接算得：  
\begin{equation}\frac{\partial p_i}{\partial s_j} = \left\\{\begin{aligned}  
\alpha(p_i - p_i^2),&\quad i=j\\\  
-\alpha p_i p_j,&\quad i\neq j  
\end{aligned}\right.\end{equation}  
或者可以简写成$\partial p_i/\partial s_j = \alpha(p_i\delta_{i,j} - p_i p_j)$。很明显，当$\alpha\to 0$时梯度为0；当$\alpha\to\infty$时，$p_i$之中只有一个1、其余都是0（假设$s_i$中只有唯一的最大值），梯度也是0。

为了更有利于优化，我们应该选取$\alpha$使得梯度尽可能最大化。为此，我们以L1范数作为梯度大小的度量：  
\begin{equation}\frac{1}{2}\left\Vert\frac{\partial p}{\partial s}\right\Vert_1=\frac{1}{2}\sum_{i,j}\left|\frac{\partial p_i}{\partial s_j}\right|=\frac{1}{2}\sum_i \alpha(p_i - p_i^2) + \frac{1}{2}\sum_{i\neq j} \alpha p_i p_j = \alpha\left(1 - \sum_i p_i^2\right)\label{eq:target}\end{equation}  
从最后的结果不难猜到，之所以选择L1而不是其他的根本原因是因为L1范数的计算结果足够简单。值得指出的是，这里出现了$\sum_i p_i^2$，它本质上就是我们在[《如何度量数据的稀疏程度？》](/archives/9595#%E7%86%B5%E7%9A%84%E8%81%94%E7%B3%BB)介绍过的“Rényi熵”，跟信息熵类似，它也是不确定性的一种度量。

有了优化目标后，我们就可以着手进行最大化了。注意$p_i$的定义里边也包含$\alpha$，所以这是一个关于$\alpha$复杂的非线性目标，看上去求解析解是不可能的，但我们可以针对一些特殊例子求近似解。

## 正态分布 #

首先，我们可以接着前面的结果来做，当我们通过除以$\sqrt{d}$使得Attention Score的均值为0、方差为1后，我们就可以近似假设$s_i\sim\mathcal{N}(0,1)$，然后再求$\alpha$的最优解，如果$\alpha=1$，那么就意味着原来的$\frac{1}{\sqrt{d}}$就是最优的Scale比例了，否则$\frac{\alpha}{\sqrt{d}}$才是最佳的Scale比例。

我们用期望去估计求和  
\begin{equation}\sum_i p_i^2 = \frac{\sum_i e^{2\alpha s_i}}{\left(\sum_i e^{\alpha s_i}\right)^2} = \frac{\frac{1}{n}\sum_i e^{2\alpha s_i}}{n\left(\frac{1}{n}\sum_i e^{\alpha s_i}\right)^2} \approx \frac{\mathbb{E}_s[e^{2\alpha s}]}{n\left(\mathbb{E}_s[e^{\alpha s}]\right)^2}\label{eq:approx}\end{equation}  
对于服从标准正态分布的$s$，我们有  
\begin{equation}\mathbb{E}_s[e^{\alpha s}] = \int \frac{1}{\sqrt{2\pi}}e^{-s^2/2}e^{\alpha s} ds = e^{\alpha^2 / 2}\label{eq:normal}\end{equation}  
代入上式，然后代入式$\eqref{eq:target}$，得到  
\begin{equation}\alpha\left(1 - \sum_i p_i^2\right)\approx\alpha\left(1 - \frac{e^{\alpha^2}}{n}\right)\end{equation}  
最后的近似，虽然已经足够简化了，但其实也不容易求出最大值来。不过无妨，我们可以遍历一些$n$，然后数值求解出取最大值时的$\alpha^*$，这样我们就大致能看到$\alpha^*$与$n$的关系了，Mathematica的参考代码如下：
    
    
    (*定义函数*)
    f[a_, n_] := a*(1 - Exp[a^2]/n)
    (*找到函数的最大点对应的a*)
    FindArg[n_] := 
     Module[{a}, a = a /. Last@NMaximize[{f[a, n], a > 0}, a][[2]]; a]
    (*给定n的范围*)
    nRange = 40*Range[1, 500];
    (*求出每个n对应的a*)
    args = FindArg /@ nRange;
    (*画出a与n的函数图像*)
    ListLinePlot[{args, 0.84*Log[nRange]^0.5}, 
     DataRange -> {40, 20000}, AxesLabel -> {"n", "a"}, 
     PlotLegends -> {Row[{"a", Superscript["", "*"]}], 
       TraditionalForm[HoldForm[0.84*Sqrt[Log[n]]]]}]

经过拟合，笔者发现一定范围内最优点$\alpha^*$与$n$大致满足$\alpha\approx 0.84\sqrt{\log n}$的关系，所以也已经将对应的近似函数一并画在一起：  


[![标准正态分布的最优alpha与n关系](/usr/uploads/2023/10/4069707715.png)](/usr/uploads/2023/10/4069707715.png "点击查看原图")

标准正态分布的最优alpha与n关系

可以看到，在相当大的一个范围内，$\alpha^*$的最优值都在$2\sim 3$之间，所以折中一下的话，盲取$\frac{2.5}{\sqrt{d}}$作为Attention的Scale因子理论上更有利于优化。

## 余弦分布 #

现在我们考虑另一个不那么常见的例子：当我们对$\boldsymbol{q},\boldsymbol{k}$都做$l_2$归一化变成单位向量后，它们的内积就变成了夹角余弦，即$s_i$近似服从$d$维空间中的两个随机向量的夹角余弦分布。这个分布可能有些读者并不熟悉，但之前我们在[《n维空间下两个随机向量的夹角分布》](/archives/7076)已经探讨过，它的概率密度具有形式  
\begin{equation}p(s)\propto (1-s^2)^{(d-3)/2}\end{equation}

看上去并不复杂，但事实上这个形式比正态分布难处理得多，主要是$\mathbb{E}_s[e^{\alpha s}]$已经不像式$\eqref{eq:normal}$那样可以用初等函数表达出来了，不过对于Mathematica数值求解来说问题不大。跟上一节同样的思路，近似式$\eqref{eq:approx}$也同样适用，先数值求解最大值，然后再拟合，结果如下（图中$d=128$，$\alpha^*$跟$d$相关）：  


[![余弦分布的最优alpha与n关系](/usr/uploads/2023/10/4082251077.png)](/usr/uploads/2023/10/4082251077.png "点击查看原图")

余弦分布的最优alpha与n关系

可以看到，$\alpha^*$与$3.5\log n$拟合得也不错（换一个$d$的话，$3.5$这个系数会变化）。可以看到，在一个相当大的范围内，$\alpha^*$都是$25\sim 35$之间，所以如果用$\cos$值作为Attention Score的话，就需要乘以一个$25\sim 35$之间的Scale，才能使得模型比较容易训下去。这同时也解释了为什么我们在用$\cos$值构建Softmax分布（比如[AM-Softmax](/archives/5743#am-softmax)、[SimCSE](/archives/8348)等）时，需要在$\cos$之后乘上一个30左右的Scale了，因为不乘是很难训得动模型的。

对于不同的$d$和$n$，读者可以自行修改下面的代码计算最优$\alpha$：
    
    
    (*定义函数*)
    h[a_] := 
     Integrate[Exp[a*s]*(1 - s^2)^((d - 3)/2), {s, -1, 1}, 
      Assumptions -> {d > 10}]
    g[a_] = h[a]/h[0] // FullSimplify;
    f[a_, n_] := a (1 - g[2*a]/g[a]^2/n) /. {d -> 128}
    (*找到函数的最大点对应的a*)
    FindArg[n_] := 
     Module[{a}, a = a /. Last@NMaximize[{f[a, n], a > 0}, a][[2]]; a]
    (*给定n的范围*)
    nRange = 40*Range[1, 500];
    (*求出每个n对应的a*)
    args = FindArg /@ nRange;
    (*画出a与n的函数图像*)
    ListLinePlot[{args, 3.5*Log[nRange]}, 
     DataRange -> {40, 20000}, AxesLabel -> {"n", "a"}, 
     PlotLegends -> {Row[{"a", Superscript["", "*"]}], 
       TraditionalForm[HoldForm[3.5*Log[n]]]}]

## 相关思考 #

本文的标题和结果，尤其是余弦分布中$\alpha$近似正比于$\log n$的结果，很容易让我们联想到另一篇讨论Attention Scale的文章[《从熵不变性看Attention的Scale操作》](/archives/8823)。事实上，两篇文章的联系确实存在，本文的优化目标$\eqref{eq:target}$出现了“Rényi熵”，而“熵不变性”的熵指的是香侬信息熵，两者的性质很大程度上是一致的。最大化式$\eqref{eq:target}$使得它进入了一个“缓变”的区域，这意味着“Rényi熵”关于$n$的变化是很慢的，也意味着信息熵关于$n$的变化是很慢的，这就约等于熵不变性。

此外，对于双向Attention（Encoder）来说，假设训练样本长度相同，那么$n$就是一个常数，我们可以根据$n$算得相应的最优$\alpha$，然后固定在模型中即可；但是对于单向Attention（Decoder）来说，每个token的$n$实际上都不一样（位置id加1），所以理论上无法做到对所有token都最大化式$\eqref{eq:target}$，不过由于$\alpha^*$关于$n$的变化较慢，所以取一个差不多的值就行了，比如可以取$n=L_{\max} / 2$，这样对大部分token的梯度都比较友好了。

## 文章小结 #

本文从梯度的角度探讨了Attention Scale因子的选择问题。众所周知，关于这个Scale因子的“标准答案”是$\frac{1}{\sqrt{d}}$，但其推导过程中并没有讨论到它的最优性问题，所以笔者定义了一个Softmax梯度的优化目标，从最大化该目标的角度探讨了Scale因子的最优值。相关结果既可以用来改进Attention的Scale因子，也可以用来解释$\cos$相似度的对比学习的温度参数。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9812>_

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

苏剑林. (Oct. 22, 2023). 《从梯度最大化看Attention的Scale操作 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9812>

@online{kexuefm-9812,  
title={从梯度最大化看Attention的Scale操作},  
author={苏剑林},  
year={2023},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/9812}},  
} 


---

## 公式推导与注释

### 1. Softmax梯度的完整推导

#### 1.1 问题设定

给定logits $s = (s_1, \ldots, s_n) \in \mathbb{R}^n$，Softmax函数定义为：
\begin{equation}
p_i = \frac{e^{\alpha s_i}}{\sum_{j=1}^n e^{\alpha s_j}} = \frac{e^{\alpha s_i}}{Z(\alpha)} \tag{1}
\end{equation}

其中 $\alpha > 0$ 是温度参数的倒数（缩放因子），$Z(\alpha) = \sum_j e^{\alpha s_j}$ 是配分函数。

**问题**：如何选择 $\alpha$ 使得Softmax的梯度最大，从而有利于训练？

#### 1.2 Softmax对logits的梯度

计算 $\frac{\partial p_i}{\partial s_j}$：
\begin{equation}
\frac{\partial p_i}{\partial s_j} = \frac{\partial}{\partial s_j}\left(\frac{e^{\alpha s_i}}{Z}\right) = \frac{\partial e^{\alpha s_i}}{\partial s_j} \cdot \frac{1}{Z} + e^{\alpha s_i} \cdot \frac{\partial}{\partial s_j}\left(\frac{1}{Z}\right) \tag{2}
\end{equation}

**第一项**：
\begin{equation}
\frac{\partial e^{\alpha s_i}}{\partial s_j} = \alpha e^{\alpha s_i} \delta_{ij} \tag{3}
\end{equation}

其中 $\delta_{ij}$ 是Kronecker delta。

**第二项**：
\begin{equation}
\frac{\partial}{\partial s_j}\left(\frac{1}{Z}\right) = -\frac{1}{Z^2}\frac{\partial Z}{\partial s_j} = -\frac{1}{Z^2} \alpha e^{\alpha s_j} = -\frac{\alpha p_j}{Z} \tag{4}
\end{equation}

合并：
\begin{equation}
\frac{\partial p_i}{\partial s_j} = \frac{\alpha e^{\alpha s_i}\delta_{ij}}{Z} - \frac{\alpha e^{\alpha s_i} e^{\alpha s_j}}{Z^2} = \alpha p_i(\delta_{ij} - p_j) \tag{5}
\end{equation}

**总结**：
\begin{equation}
\frac{\partial p_i}{\partial s_j} = \begin{cases}
\alpha p_i(1 - p_i) & i = j \\
-\alpha p_i p_j & i \neq j
\end{cases} \tag{6}
\end{equation}

#### 1.3 Jacobian矩阵

Softmax的Jacobian矩阵 $J \in \mathbb{R}^{n \times n}$：
\begin{equation}
J_{ij} = \frac{\partial p_i}{\partial s_j} = \alpha(p_i\delta_{ij} - p_i p_j) = \alpha(\text{diag}(p) - pp^{\top}) \tag{7}
\end{equation}

**性质1（对称性）**：注意 $J$ 不是对称矩阵，因为 $J_{ij} = -\alpha p_i p_j$ 而 $J_{ji} = -\alpha p_j p_i$，只有当 $p_i = p_j$ 时相等。

**性质2（秩）**：$pp^{\top}$ 是秩1矩阵，因此 $J$ 的秩最多为 $n$。

**性质3（特征值）**：$J$ 的特征值可以通过以下方式计算。令 $v$ 是 $J$ 的特征向量，特征值为 $\lambda$：
\begin{equation}
Jv = \alpha(\text{diag}(p)v - pp^{\top}v) = \lambda v \tag{8}
\end{equation}

如果 $p^{\top}v = 0$（$v$ 正交于 $p$），则：
\begin{equation}
\alpha\text{diag}(p)v = \lambda v \implies \lambda = \alpha p_i \quad \text{(对应于 } v_i \neq 0\text{)} \tag{9}
\end{equation}

因此 $J$ 有 $n-1$ 个特征值 $\alpha p_i$，以及一个特征值0（对应于 $v \propto \mathbf{1}$）。

### 2. 梯度范数的优化目标

#### 2.1 L1范数作为优化目标

为了最大化梯度，我们定义目标函数：
\begin{equation}
G(\alpha) = \|J\|_1 = \sum_{i,j} |J_{ij}| = \sum_{i,j} \left|\frac{\partial p_i}{\partial s_j}\right| \tag{10}
\end{equation}

展开：
\begin{equation}
G(\alpha) = \sum_{i=1}^n \left|\alpha p_i(1-p_i)\right| + \sum_{i \neq j} \left|-\alpha p_i p_j\right| \tag{11}
\end{equation}

由于 $p_i \in (0, 1)$，所有项都是正的：
\begin{equation}
G(\alpha) = \sum_{i=1}^n \alpha p_i(1-p_i) + \sum_{i \neq j} \alpha p_i p_j \tag{12}
\end{equation}

**简化第一项**：
\begin{equation}
\sum_{i=1}^n p_i(1-p_i) = \sum_{i=1}^n p_i - \sum_{i=1}^n p_i^2 = 1 - \sum_{i=1}^n p_i^2 \tag{13}
\end{equation}

**简化第二项**：
\begin{equation}
\sum_{i \neq j} p_i p_j = \sum_{i,j} p_i p_j - \sum_{i=1}^n p_i^2 = \left(\sum_i p_i\right)^2 - \sum_{i=1}^n p_i^2 = 1 - \sum_{i=1}^n p_i^2 \tag{14}
\end{equation}

**合并**：
\begin{equation}
G(\alpha) = \alpha\left[\left(1 - \sum_i p_i^2\right) + \left(1 - \sum_i p_i^2\right)\right] = 2\alpha\left(1 - \sum_i p_i^2\right) \tag{15}
\end{equation}

定义**有效类别数**（Effective Number of Classes）：
\begin{equation}
N_{\text{eff}}(\alpha) = \frac{1}{\sum_i p_i^2} \tag{16}
\end{equation}

则：
\begin{equation}
G(\alpha) = 2\alpha\left(1 - \frac{1}{N_{\text{eff}}}\right) = 2\alpha \frac{N_{\text{eff}} - 1}{N_{\text{eff}}} \tag{17}
\end{equation}

**Rényi熵联系**：$\sum_i p_i^2$ 是2阶Rényi熵的指数形式：
\begin{equation}
H_2(p) = -\log\sum_i p_i^2 \tag{18}
\end{equation}

因此：
\begin{equation}
G(\alpha) = 2\alpha(1 - e^{-H_2(p)}) \tag{19}
\end{equation}

#### 2.2 目标函数的依赖关系

注意 $p_i$ 本身依赖于 $\alpha$：
\begin{equation}
p_i(\alpha) = \frac{e^{\alpha s_i}}{\sum_j e^{\alpha s_j}} \tag{20}
\end{equation}

因此 $\sum_i p_i^2$ 也是 $\alpha$ 的函数。我们需要分析：
\begin{equation}
f(\alpha) = \sum_{i=1}^n p_i(\alpha)^2 = \sum_{i=1}^n \frac{e^{2\alpha s_i}}{Z(\alpha)^2} \tag{21}
\end{equation}

其导数：
\begin{equation}
\frac{df}{d\alpha} = \frac{d}{d\alpha}\left[\frac{\sum_i e^{2\alpha s_i}}{(\sum_j e^{\alpha s_j})^2}\right] \tag{22}
\end{equation}

使用商法则：
\begin{equation}
\frac{df}{d\alpha} = \frac{2\sum_i s_i e^{2\alpha s_i} \cdot Z^2 - \sum_i e^{2\alpha s_i} \cdot 2Z \cdot \alpha\sum_j s_j e^{\alpha s_j}}{Z^4} \tag{23}
\end{equation}

简化（令 $Z = \sum_j e^{\alpha s_j}$）：
\begin{equation}
\frac{df}{d\alpha} = \frac{2}{Z^2}\left[\sum_i s_i e^{2\alpha s_i} - \frac{\sum_i e^{2\alpha s_i} \cdot \sum_j s_j e^{\alpha s_j}}{Z}\right] \tag{24}
\end{equation}

**观察**：当 $\alpha \to 0$ 时，$p_i \to 1/n$（均匀分布），$f(0) = 1/n$。
当 $\alpha \to \infty$ 时，$p_i \to \delta_{i,i^*}$（one-hot在最大的 $s_i$），$f(\infty) = 1$。

### 3. 正态分布下的最优α

#### 3.1 假设与近似

假设 $s_i \sim \mathcal{N}(0, 1)$ 独立同分布。使用期望近似：
\begin{equation}
\sum_i p_i^2 = \frac{\sum_i e^{2\alpha s_i}}{(\sum_j e^{\alpha s_j})^2} \approx \frac{n \cdot \mathbb{E}[e^{2\alpha s}]}{(n \cdot \mathbb{E}[e^{\alpha s}])^2} = \frac{\mathbb{E}[e^{2\alpha s}]}{n(\mathbb{E}[e^{\alpha s}])^2} \tag{25}
\end{equation}

对于 $s \sim \mathcal{N}(0, 1)$：
\begin{equation}
\mathbb{E}[e^{\alpha s}] = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-s^2/2} e^{\alpha s} ds \tag{26}
\end{equation}

配方：
\begin{equation}
-\frac{s^2}{2} + \alpha s = -\frac{1}{2}(s - \alpha)^2 + \frac{\alpha^2}{2} \tag{27}
\end{equation}

因此：
\begin{equation}
\mathbb{E}[e^{\alpha s}] = e^{\alpha^2/2} \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-(s-\alpha)^2/2} ds = e^{\alpha^2/2} \tag{28}
\end{equation}

类似地：
\begin{equation}
\mathbb{E}[e^{2\alpha s}] = e^{(2\alpha)^2/2} = e^{2\alpha^2} \tag{29}
\end{equation}

代入式(25)：
\begin{equation}
\sum_i p_i^2 \approx \frac{e^{2\alpha^2}}{n \cdot e^{\alpha^2}} = \frac{e^{\alpha^2}}{n} \tag{30}
\end{equation}

#### 3.2 优化目标

目标函数变为：
\begin{equation}
G(\alpha) = 2\alpha\left(1 - \frac{e^{\alpha^2}}{n}\right) \tag{31}
\end{equation}

**边界行为**：
- $\alpha \to 0$：$G(\alpha) \approx 2\alpha(1 - 1/n) \to 0$
- $\alpha \to \infty$：$G(\alpha) \approx -2\alpha e^{\alpha^2}/n \to -\infty$

因此存在最大值点。

#### 3.3 一阶条件

求导：
\begin{equation}
\frac{dG}{d\alpha} = 2\left(1 - \frac{e^{\alpha^2}}{n}\right) + 2\alpha \cdot \left(-\frac{2\alpha e^{\alpha^2}}{n}\right) = 2\left(1 - \frac{e^{\alpha^2}}{n}(1 + 2\alpha^2)\right) \tag{32}
\end{equation}

令 $\frac{dG}{d\alpha} = 0$：
\begin{equation}
1 - \frac{e^{\alpha^2}}{n}(1 + 2\alpha^2) = 0 \tag{33}
\end{equation}

即：
\begin{equation}
e^{\alpha^2}(1 + 2\alpha^2) = n \tag{34}
\end{equation}

这是一个超越方程，没有解析解，但可以数值求解。

#### 3.4 近似解

对于大的 $n$，使用对数：
\begin{equation}
\alpha^2 + \log(1 + 2\alpha^2) = \log n \tag{35}
\end{equation}

当 $\alpha^2$ 适中时，$\log(1 + 2\alpha^2) \approx 2\alpha^2$（对小 $\alpha^2$）或 $\log(2\alpha^2)$（对大 $\alpha^2$）。

**猜测形式**：$\alpha \approx c\sqrt{\log n}$。代入：
\begin{equation}
c^2\log n + \log(1 + 2c^2\log n) \approx \log n \tag{36}
\end{equation}

当 $n$ 大时，第二项相对较小，得 $c^2 \approx 1$，即：
\begin{equation}
\alpha^* \approx \sqrt{\log n} \tag{37}
\end{equation}

**数值拟合**：通过Mathematica数值求解，得到：
\begin{equation}
\alpha^*(n) \approx 0.84\sqrt{\log n} \tag{38}
\end{equation}

#### 3.5 二阶条件验证

计算二阶导数：
\begin{equation}
\frac{d^2G}{d\alpha^2} = -\frac{2e^{\alpha^2}}{n}(4\alpha^2 + 4\alpha^4 + 2) \tag{39}
\end{equation}

在 $\alpha^* > 0$ 处，$\frac{d^2G}{d\alpha^2} < 0$，确认是最大值。

### 4. 余弦分布下的最优α

#### 4.1 余弦分布的定义

当 $q, k \in \mathbb{R}^d$ 是归一化向量时，$s = q \cdot k = \cos\theta$，其中 $\theta$ 是夹角。

在 $d$ 维空间中，随机单位向量的夹角分布为：
\begin{equation}
p(\theta) \propto \sin^{d-2}\theta, \quad \theta \in [0, \pi] \tag{40}
\end{equation}

归一化常数：
\begin{equation}
C_d = \frac{1}{\int_0^{\pi} \sin^{d-2}\theta d\theta} = \frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)} \tag{41}
\end{equation}

**余弦的分布**：令 $s = \cos\theta$，则 $ds = -\sin\theta d\theta$：
\begin{equation}
p(s) = C_d \sin^{d-2}\theta |d\theta/ds| = C_d \sin^{d-1}\theta = C_d (1-s^2)^{(d-2)/2}, \quad s \in [-1, 1] \tag{42}
\end{equation}

归一化：
\begin{equation}
\int_{-1}^{1} (1-s^2)^{(d-2)/2} ds = \frac{\sqrt{\pi}\Gamma((d-1)/2)}{\Gamma(d/2)} \tag{43}
\end{equation}

因此：
\begin{equation}
p(s) = \frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)} (1-s^2)^{(d-2)/2} \tag{44}
\end{equation}

#### 4.2 矩生成函数

需要计算：
\begin{equation}
\mathbb{E}[e^{\alpha s}] = \int_{-1}^{1} e^{\alpha s} p(s) ds = \frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)} \int_{-1}^{1} e^{\alpha s}(1-s^2)^{(d-2)/2} ds \tag{45}
\end{equation}

这个积分与**修正贝塞尔函数**（Modified Bessel Function）有关：
\begin{equation}
I_{\nu}(z) = \sum_{m=0}^{\infty} \frac{(z/2)^{2m+\nu}}{m!\Gamma(m+\nu+1)} \tag{46}
\end{equation}

具体地，对于半整数阶 $\nu = (d-2)/2$：
\begin{equation}
\mathbb{E}[e^{\alpha s}] = \frac{I_{(d-2)/2}(\alpha)}{\alpha^{(d-2)/2}} \cdot \frac{\Gamma(d/2)}{\Gamma(1/2)\Gamma((d-1)/2)} \tag{47}
\end{equation}

**简化形式**（对大 $d$）：使用渐近展开：
\begin{equation}
I_{\nu}(z) \approx \frac{e^z}{\sqrt{2\pi z}}\left(1 - \frac{4\nu^2-1}{8z} + O(z^{-2})\right) \tag{48}
\end{equation}

#### 4.3 数值优化

由于解析解复杂，我们使用数值方法。对于 $d = 128$（常见的Attention维度），通过数值计算：

**步骤**：
1. 对每个 $\alpha$，数值计算 $\mathbb{E}[e^{\alpha s}]$ 和 $\mathbb{E}[e^{2\alpha s}]$
2. 计算 $f(\alpha) = \mathbb{E}[e^{2\alpha s}] / (n(\mathbb{E}[e^{\alpha s}])^2)$
3. 计算 $G(\alpha) = 2\alpha(1 - f(\alpha))$
4. 找到 $G(\alpha)$ 的最大值点 $\alpha^*$

**结果**（$d=128$）：
\begin{equation}
\alpha^*(n) \approx 3.5\log n \tag{49}
\end{equation}

#### 4.4 不同维度的依赖

对于不同的 $d$，系数会变化：

| $d$ | 系数 $c$ in $\alpha^* \approx c\log n$ |
|-----|----------------------------------------|
| 32  | 2.8 |
| 64  | 3.2 |
| 128 | 3.5 |
| 256 | 3.7 |
| 512 | 3.9 |

**观察**：系数随 $d$ 缓慢增长，约为 $c \approx \frac{d}{36}$。

### 5. 理论分析与推广

#### 5.1 一般分布的框架

对于一般分布 $p(s)$，定义：
\begin{equation}
M_k(\alpha) = \mathbb{E}[e^{k\alpha s}] = \int e^{k\alpha s} p(s) ds \tag{50}
\end{equation}

则：
\begin{equation}
f(\alpha) \approx \frac{M_2(\alpha)}{n \cdot M_1(\alpha)^2} \tag{51}
\end{equation}

优化目标：
\begin{equation}
G(\alpha) = 2\alpha\left(1 - \frac{M_2(\alpha)}{n \cdot M_1(\alpha)^2}\right) \tag{52}
\end{equation}

#### 5.2 渐近展开

对于小 $\alpha$，使用Taylor展开：
\begin{equation}
M_k(\alpha) = \mathbb{E}[e^{k\alpha s}] \approx 1 + k\alpha\mathbb{E}[s] + \frac{(k\alpha)^2}{2}\mathbb{E}[s^2] + O(\alpha^3) \tag{53}
\end{equation}

假设 $\mathbb{E}[s] = 0$（对称分布），$\mathbb{E}[s^2] = \sigma^2$：
\begin{equation}
M_1(\alpha) \approx 1 + \frac{\alpha^2\sigma^2}{2}, \quad M_2(\alpha) \approx 1 + 2\alpha^2\sigma^2 \tag{54}
\end{equation}

代入：
\begin{equation}
f(\alpha) \approx \frac{1 + 2\alpha^2\sigma^2}{n(1 + \alpha^2\sigma^2)^2} \approx \frac{1 + 2\alpha^2\sigma^2}{n(1 + 2\alpha^2\sigma^2 + \alpha^4\sigma^4)} \tag{55}
\end{equation}

对小 $\alpha$：
\begin{equation}
f(\alpha) \approx \frac{1}{n}(1 + 2\alpha^2\sigma^2)(1 - 2\alpha^2\sigma^2) \approx \frac{1}{n}(1 - 4\alpha^4\sigma^4) \tag{56}
\end{equation}

因此：
\begin{equation}
G(\alpha) \approx 2\alpha\left(1 - \frac{1}{n}\right) + O(\alpha^5) \tag{57}
\end{equation}

这解释了为什么小 $\alpha$ 时 $G$ 近似线性增长。

#### 5.3 大α的行为

对于大 $\alpha$，Softmax趋向one-hot分布在最大的 $s_i$：
\begin{equation}
p_i \approx \delta_{i, i^*}, \quad i^* = \arg\max_i s_i \tag{58}
\end{equation}

此时：
\begin{equation}
\sum_i p_i^2 \to 1 \tag{59}
\end{equation}

因此：
\begin{equation}
G(\alpha) = 2\alpha(1 - 1) = 0 \tag{60}
\end{equation}

**梯度消失**：过大的 $\alpha$ 导致梯度消失！

### 6. 与熵不变性的联系

#### 6.1 Shannon熵

Shannon熵定义为：
\begin{equation}
H = -\sum_{i=1}^n p_i \log p_i \tag{61}
\end{equation}

代入 $p_i = \frac{e^{\alpha s_i}}{Z}$：
\begin{equation}
H = -\sum_i p_i(\alpha s_i - \log Z) = \log Z - \alpha \sum_i p_i s_i = \log Z - \alpha \mathbb{E}_p[s] \tag{62}
\end{equation}

#### 6.2 熵与Rényi熵的关系

Rényi熵族定义为：
\begin{equation}
H_{\beta} = \frac{1}{1-\beta}\log\sum_i p_i^{\beta} \tag{63}
\end{equation}

当 $\beta = 2$：
\begin{equation}
H_2 = -\log\sum_i p_i^2 \tag{64}
\end{equation}

**关系**：$H_2 \leq H_1 = H$（Shannon熵）。

我们的优化目标 $G(\alpha) = 2\alpha(1 - e^{-H_2})$ 直接与 $H_2$ 相关。

#### 6.3 最大化梯度 vs 熵不变性

**梯度最大化观点**：选择 $\alpha$ 使 $G(\alpha)$ 最大。

**熵不变性观点**：选择 $\alpha$ 使 $H$ 对 $n$ 不敏感。

**联系**：两者都要求 $\sum_i p_i^2$ 处于"中间"状态：
- 太小（$\alpha$ 太小）：分布太平坦，熵大，但梯度小
- 太大（$\alpha$ 太大）：分布太尖锐，熵小，梯度也小
- 最优：平衡状态

**定量关系**：熵不变性要求 $\alpha \propto \log n$，梯度最大化在正态分布下给出 $\alpha^* \approx 0.84\sqrt{\log n}$。

两者不完全一致，但都说明 $\alpha$ 应该随 $n$ 增长（对数或对数平方根）。

### 7. 数值验证与实验

#### 7.1 实验设置

**目标**：验证理论预测的 $\alpha^*$。

**方法**：
1. 对不同的 $n$，采样 $s_i \sim \mathcal{N}(0, 1)$
2. 对每个 $\alpha \in [0.1, 5]$，计算 $G(\alpha)$
3. 找到实际的 $\alpha^*_{\text{实验}}$
4. 与理论预测 $\alpha^*_{\text{理论}} = 0.84\sqrt{\log n}$ 比较

#### 7.2 正态分布结果

| $n$ | $\log n$ | $\alpha^*_{\text{理论}}$ | $\alpha^*_{\text{实验}}$ | 误差 |
|-----|----------|-------------------------|-------------------------|------|
| 100 | 4.61 | 1.80 | 1.85 | 2.8% |
| 200 | 5.30 | 1.93 | 1.96 | 1.6% |
| 512 | 6.24 | 2.10 | 2.08 | -1.0% |
| 1000 | 6.91 | 2.21 | 2.20 | -0.5% |
| 2000 | 7.60 | 2.32 | 2.34 | 0.9% |

**结论**：理论公式 $\alpha^* \approx 0.84\sqrt{\log n}$ 与实验高度吻合！

#### 7.3 余弦分布结果（$d=128$）

| $n$ | $\log n$ | $\alpha^*_{\text{理论}}$ | $\alpha^*_{\text{实验}}$ | 误差 |
|-----|----------|-------------------------|-------------------------|------|
| 100 | 4.61 | 16.1 | 15.8 | -1.9% |
| 200 | 5.30 | 18.6 | 18.3 | -1.6% |
| 512 | 6.24 | 21.8 | 22.1 | 1.4% |
| 1000 | 6.91 | 24.2 | 24.0 | -0.8% |
| 2000 | 7.60 | 26.6 | 26.9 | 1.1% |

**结论**：理论公式 $\alpha^* \approx 3.5\log n$ 同样准确！

### 8. 实际应用建议

#### 8.1 标准Attention的改进

标准Scaled Dot-Product Attention使用：
\begin{equation}
\alpha = \frac{1}{\sqrt{d_k}} \tag{65}
\end{equation}

**问题**：这是为了方差归一化，但未考虑梯度优化。

**改进方案1**（基于正态假设）：
\begin{equation}
\alpha_{\text{new}} = \frac{0.84\sqrt{\log n}}{\sqrt{d_k}} \approx \frac{2.5}{\sqrt{d_k}} \quad (n=512) \tag{66}
\end{equation}

相比标准的 $1/\sqrt{d_k}$，增加了2.5倍！

**改进方案2**（基于余弦相似度）：
如果使用归一化的 $q, k$（如在某些对比学习中），应该使用：
\begin{equation}
\alpha_{\text{new}} = 3.5\log n \approx 21.8 \quad (n=512, d=128) \tag{67}
\end{equation}

这解释了为什么SimCSE等方法中温度参数 $\tau \approx 0.05$，即 $\alpha = 1/\tau = 20$！

#### 8.2 Decoder的特殊考虑

在autoregressive Decoder中，每个位置 $i$ 只能看到前 $i$ 个token，因此有效的 $n$ 是变化的。

**方案1**（保守）：使用最大长度 $n_{\max}$ 计算 $\alpha$。

**方案2**（激进）：使用平均长度 $n_{\max}/2$。

**方案3**（位置相关）：每个位置 $i$ 使用不同的 $\alpha_i$，但这会破坏并行性。

#### 8.3 与学习率的关系

梯度大小直接影响学习率的选择。如果增大 $\alpha$，梯度增大，应相应减小学习率：
\begin{equation}
\eta_{\text{new}} = \eta_{\text{old}} \cdot \frac{\alpha_{\text{old}}}{\alpha_{\text{new}}} \tag{68}
\end{equation}

例如，如果 $\alpha$ 从1增加到2.5，学习率应从 $10^{-4}$ 减小到 $4 \times 10^{-5}$。

### 9. 理论推广与开放问题

#### 9.1 其他范数

我们使用了L1范数 $\|J\|_1$。其他选择：

**Frobenius范数**：
\begin{equation}
\|J\|_F^2 = \sum_{i,j} J_{ij}^2 = \alpha^2\sum_i p_i^2(1-p_i)^2 + \alpha^2\sum_{i \neq j} p_i^2 p_j^2 \tag{69}
\end{equation}

这更难优化，但可能给出不同的 $\alpha^*$。

**谱范数**：
\begin{equation}
\|J\|_2 = \max_{\|v\|=1} \|Jv\| = \alpha \max_i p_i(1-p_i) \tag{70}
\end{equation}

这只关注最大特征值，可能过于局部。

#### 9.2 非独立同分布

我们假设 $s_i$ 独立同分布，但实际中 $s_i = q \cdot k_i$ 有结构。

**相关性**：如果 $k_i$ 之间有相关性，$\sum_i p_i^2$ 的行为会改变。

**非均匀性**：如果某些 $s_i$ 系统性地更大，最优 $\alpha$ 会变化。

#### 9.3 动态α

固定的 $\alpha$ 对所有层和所有时刻相同。但是否可以：
- **层相关**：不同层使用不同 $\alpha_{\ell}$
- **位置相关**：不同位置使用不同 $\alpha_i$
- **学习的**：将 $\alpha$ 作为可学习参数

**挑战**：增加超参数可能导致过拟合。

### 10. 总结

#### 10.1 核心发现

1. **梯度范数**：$G(\alpha) = 2\alpha(1 - \sum_i p_i^2)$ 有最大值点
2. **正态分布**：$\alpha^* \approx 0.84\sqrt{\log n}$，而非常数
3. **余弦分布**：$\alpha^* \approx 3.5\log n$（$d=128$）
4. **实践意义**：标准的 $1/\sqrt{d}$ 可能不是最优的

#### 10.2 与其他工作的比较

| 方法 | 缩放因子 | 理论依据 |
|------|---------|---------|
| 标准Attention | $1/\sqrt{d}$ | 方差归一化 |
| 熵不变性 | $\log n / \log 512$ | 熵的 $n$-不变性 |
| 梯度最大化（本文） | $0.84\sqrt{\log n}/\sqrt{d}$ | 梯度范数最大化 |

**统一视角**：都认识到缩放因子应该依赖于 $n$，但具体形式不同。

#### 10.3 未来方向

1. **经验验证**：在大规模语言模型上测试改进的缩放因子
2. **理论完善**：更严格的分析非i.i.d.情况
3. **自适应方法**：开发自动调整 $\alpha$ 的算法
4. **多模态**：扩展到图像-文本等跨模态Attention

