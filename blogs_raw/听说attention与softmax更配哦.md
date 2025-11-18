---
title: 听说Attention与Softmax更配哦～
slug: 听说attention与softmax更配哦
date: 2022-04-07
tags: 熵, 语言模型, attention, 预训练, 生成模型
status: pending
---

# 听说Attention与Softmax更配哦～

**原文链接**: [https://spaces.ac.cn/archives/9019](https://spaces.ac.cn/archives/9019)

**发布日期**: 

---

不知道大家留意到一个细节没有，就是当前NLP主流的预训练模式都是在一个固定长度（比如512）上进行，然后直接将预训练好的模型用于不同长度的任务中。大家似乎也没有对这种模式有过怀疑，仿佛模型可以自动泛化到不同长度是一个“理所应当”的能力。

当然，笔者此前同样也没有过类似的质疑，直到前几天笔者做了Base版的GAU实验后才发现GAU的长度泛化能力并不如想象中好。经过进一步分析后，笔者才明白原来这种长度泛化的能力并不是“理所当然”的......

## 模型回顾 #

在[《FLASH：可能是近来最有意思的高效Transformer设计》](/archives/8934)中，我们介绍了“门控注意力单元GAU”，它是一种融合了GLU和Attention的新设计。

除了效果，GAU在设计上给我们带来的冲击主要有两点：一是它显示了单头注意力未必就逊色于多头注意力，这奠定了它“快”、“省”的地位；二是它是显示了注意力未必需要Softmax归一化，可以换成简单的$\text{relu}^2$除以序列长度：  
\begin{equation}\boldsymbol{A}=\frac{1}{n}\text{relu}^2\left(\frac{\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}}{\sqrt{s}}\right)=\frac{1}{ns}\text{relu}^2\left(\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}\right)\end{equation}  
这个形式导致了一个有意思的问题：如果我们预训练的时候尽量将样本整理成同一长度（比如512），那么在预训练阶段$n$几乎一直就是512，也就是说$n$相当于一个常数，如果我们将它用于其他长度（比如64、128）微调，那么这个$n$究竟要自动改为样本长度，还是保持为512呢？

直觉应该是等于样本长度更加自适应一些，但答案很反直觉：$n$固定为512的微调效果比$n$取样本长度的效果要明显好！这就引人深思了......

## 问题定位 #

如果单看GAU的预训练效果，它是优于标准Attention的，所以GAU本身的拟合能力应该是没问题的，只是$\frac{1}{n}\text{relu}^2(\cdot)$在样本长度方面的迁移能力不好。为了确认这一点，笔者也尝试了混合不同长度的样本来做GAU的预训练，发现结果会有明显的改善。

那么，可能是GAU的什么地方出了问题呢？其实这不难猜测，GAU的整体运算可以简写成$\boldsymbol{O}=(\boldsymbol{U}\odot\boldsymbol{A}\boldsymbol{V})\boldsymbol{W}_o$，其中$\boldsymbol{U},\boldsymbol{V},\boldsymbol{W}_o$都是token-wise的，也就是说它们根本不会受到长度变化的影响，所以问题只能是出现在$\boldsymbol{A}$中。

以前我们用标准的Attention时，并没有出现类似的问题，以至于我们以前都无意识地觉得这是一个“理所当然”的性质。所以，我们需要从GAU的Attention与标准Attention的差异中发现问题。前面说了，两者不同的地方有两点，其一是多头Attention变成单头Attention，但是这顶多会让效果有一定波动，而我们测出来的结果是大幅下降，所以问题就只能出现在另一点，也就是归一化方式上，即Attention的$softmax$换成$\frac{1}{n}\text{relu}^2(\cdot)$所带来的。

验证这个猜测很简单，笔者将GAU中Attention的归一化方式换回Softmax后重新训练一个GAU模型，然后微调测试不同长度的任务，发现其效果比$\frac{1}{n}\text{relu}^2(\cdot)$时明显要好。所以，我们得出结论：Attention还是与Softmax更配～

## 原因分析 #

为什么更符合直觉的、自适应长度的$n$反而表现不如固定的$n$呢？既然我们已经以往用Softmax是没有这个问题的，所以我们不妨从Softmax出发找找灵感。Softmax的操作是：  
\begin{equation}a_{i,j} = \frac{1}{Z_i}\exp\left(\frac{\boldsymbol{q}_i\cdot\boldsymbol{k}_j}{\sqrt{d}}\right),\quad Z_i = \sum_{j=1}^n \exp\left(\frac{\boldsymbol{q}_i\cdot\boldsymbol{k}_j}{\sqrt{d}}\right)\end{equation}  
一个直接的问题就是：$Z_i$跟$n$的关系是怎样的呢？如果真有$Z_i=\mathcal{O}(n)$，那么理论上将$Z_i$换成$n$应该能取得相近的效果，至少不会是特别差的那种。

然而，我们知道注意力的重点是“注意”，它应该有能力“聚焦”到它认为比较重要的几个token上。同时，以往关于高效Transformer的一些实验结果显示，把标准Attention换成Local Attention后结果并不会明显下降，所以我们可以预计位置为$i$的Attention基本上就聚焦在$i$附近的若干token上，超出一定距离后就基本为0了。事实上，也有很多事后的可视化结果显示训练好的Attention矩阵其实是很稀疏的。

综合这些结果，我们可以得出，存在某个常数$k$，使得$|j-i|\geq k$时$\exp\left(\frac{\boldsymbol{q}_i\cdot\boldsymbol{k}_j}{\sqrt{d}}\right)$都相当接近于0，这样一来$Z_i$应该更接近$\mathcal{O}(k)$而不是$\mathcal{O}(n)$，这就意味着$Z_i$很可能跟$n$是无关的，或者说跟$n$的数量级关系至少是小于$\mathcal{O}(n)$的！因此，我们如果要将$Z_i$替换成别的东西，那应该是一个比$n$的一次方更低阶的函数，甚至是一个常数。

现在回看GAU，它的激活函数换成了$\text{relu}^2(\cdot)$时，其Attention情况是类似的，甚至会更稀疏。这是因为$\text{relu}$操作有直接置零的作用，不像$\exp(\cdot)$总是正的，同时GAU“标配”旋转位置编码RoPE，在[《Transformer升级之路：2、博采众长的旋转式位置编码》](/archives/8265)中我们就推导过，RoPE本身自带一定的远程衰减的能力。综合这些条件，GAU的归一化因子也应该是低于$\mathcal{O}(n)$的阶甚至是常数级别的。

## 熵不变性 #

由此，我们可以总结出GAU的三个解决方案，一是预训练和微调都用同一个固定的$n$；二是依然使用动态的样本长度$n$，但是预训练时需要用不同长度的样本来混合训练，不能只使用单一长度的样本；三就是像Softmax那样补充上一个归一化因子，让模型自己去学：  
\begin{equation}a_{i,j} = \frac{1}{Z_i}\text{relu}^2\left(\frac{\boldsymbol{q}_i\cdot\boldsymbol{k}_j}{\sqrt{d}}\right),\quad Z_i = \sum_{i=1}^n \text{relu}^2\left(\frac{\boldsymbol{q}_i\cdot\boldsymbol{k}_j}{\sqrt{d}}\right)\end{equation}

既然存在这些解决方案，那为什么我们还说“Attention与Softmax更配”呢？GAU的$\text{relu}^2(\cdot)$哪里不够配呢？首先，我们看GAU原论文的消融实验，显示出$\text{relu}^2(\cdot)$换成Softmax，效果基本是一致的：  


[![GAU的squared_relu换成softmax效果是相近的](/usr/uploads/2022/04/3734029708.png)](/usr/uploads/2022/04/3734029708.png "点击查看原图")

GAU的squared_relu换成softmax效果是相近的

有了这个基本保证之后，我们就可以看Softmax比$\text{relu}^2(\cdot)$好在哪里了。我们看刚才提到的GAU三个解决方案，方案一总让人感觉不够自适应，方案二必须用多种长度训练显得不够优雅，至于方案三补充了归一化因子后形式上相比Softmax反而显得“臃肿”了。所以，总体来说还是用Softmax显得更为优雅有效。

此外，泛化能力可以简单分为“内插”和“外推”两种，在这里内插（外推）指的是测试长度小于（大于）训练长度。我们刚才说归一化因子是常数量级，更多是在内插范围内说的。对于外推来说，如果长度足够长，$\boldsymbol{q}_i,\boldsymbol{k}_j$都“挤”在一起，所以很难保持距离超过某个范围就很接近于0的特性。而如果我们用Softmax的话，就是它可以推导出一个“熵不变性”的版本，来增强模型的外推能力：  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\log_{512} n}{\sqrt{d}}QK^{\top}\right)V\end{equation}  
在[《从熵不变性看Attention的Scale操作》](/archives/8823)中我们做过简单的对比实验，显示该版本确实能提高模型在超出训练长度外的效果。

那么，$\text{relu}^2(\cdot)$能否推一个“熵不变性”的版本呢？答案是不能，因为它相当于是通过温度参数来调节分布的熵，这要求激活函数不能是具备正齐次性，比如对于幂函数有$(\lambda \boldsymbol{q}_i\cdot\boldsymbol{k}_j)^n=\lambda^n (\boldsymbol{q}_i\cdot\boldsymbol{k}_j)^n$，归一化后$\lambda^n$就抵消了，不起作用。激活函数最好比幂函数高一阶，才比较好实现这个调控，而比幂函数高阶的函数，最常见就是指数函数了，而指数归一化正好就是Softmax。

## 本文小结 #

本文分析了GAU在微调效果不佳的原因，发现Attention的归一化因子应该是接近常数量级的，所以GAU用$n$或者$n^2$做归一化因子会表现不佳。总的来说，笔者认为Attention还是跟Softmax更配，它是一个不错的基准，并且还可以通过“熵不变性”的拓展来进一步增强外推能力。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9019>_

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

苏剑林. (Apr. 07, 2022). 《听说Attention与Softmax更配哦～ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9019>

@online{kexuefm-9019,  
title={听说Attention与Softmax更配哦～},  
author={苏剑林},  
year={2022},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9019}},  
} 


---

## 公式推导与注释

### 一、Softmax函数的数学基础

#### 1.1 Softmax函数定义与性质

**定义**: 给定向量 $\boldsymbol{x} = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$，Softmax函数定义为：
\begin{equation}
\text{softmax}(\boldsymbol{x})_i = \frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)} = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
\tag{1}
\end{equation}

**性质1（概率分布）**: Softmax输出满足概率分布的两个基本条件：
\begin{equation}
\text{softmax}(\boldsymbol{x})_i \in (0, 1), \quad \sum_{i=1}^n \text{softmax}(\boldsymbol{x})_i = 1
\tag{2}
\end{equation}

**证明**:
- 非负性显然成立：$\exp(x_i) > 0, \forall x_i \in \mathbb{R}$
- 归一性：$\sum_{i=1}^n \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{\sum_i e^{x_i}}{\sum_j e^{x_j}} = 1$

**性质2（平移不变性）**: 对任意常数 $c \in \mathbb{R}$，有：
\begin{equation}
\text{softmax}(\boldsymbol{x} + c\boldsymbol{1}) = \text{softmax}(\boldsymbol{x})
\tag{3}
\end{equation}

**证明**:
\begin{equation}
\text{softmax}(\boldsymbol{x} + c\boldsymbol{1})_i = \frac{e^{x_i+c}}{\sum_j e^{x_j+c}} = \frac{e^c \cdot e^{x_i}}{e^c \cdot \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}} = \text{softmax}(\boldsymbol{x})_i
\tag{4}
\end{equation}

这个性质对数值稳定性至关重要，实践中通常选择 $c = -\max_i x_i$ 以避免指数溢出。

**性质3（单调性）**: Softmax保持输入的相对顺序，即：
\begin{equation}
x_i > x_j \Rightarrow \text{softmax}(\boldsymbol{x})_i > \text{softmax}(\boldsymbol{x})_j
\tag{5}
\end{equation}

**性质4（温度参数）**: 引入温度参数 $\tau > 0$：
\begin{equation}
\text{softmax}_\tau(\boldsymbol{x})_i = \frac{\exp(x_i/\tau)}{\sum_{j=1}^n \exp(x_j/\tau)}
\tag{6}
\end{equation}

当 $\tau \to 0$ 时，Softmax退化为one-hot编码（选择最大值）；当 $\tau \to \infty$ 时，Softmax趋向均匀分布。

#### 1.2 Softmax梯度的完整推导

**定理**: Softmax函数的Jacobian矩阵为：
\begin{equation}
\frac{\partial \text{softmax}(\boldsymbol{x})_i}{\partial x_j} = \text{softmax}(\boldsymbol{x})_i \cdot (\delta_{ij} - \text{softmax}(\boldsymbol{x})_j)
\tag{7}
\end{equation}
其中 $\delta_{ij}$ 是Kronecker delta函数。

**详细证明**:

设 $s_i = \text{softmax}(\boldsymbol{x})_i = \frac{e^{x_i}}{Z}$，其中 $Z = \sum_{k=1}^n e^{x_k}$

**情况1**: 当 $i = j$ 时：
\begin{align}
\frac{\partial s_i}{\partial x_i} &= \frac{\partial}{\partial x_i}\left(\frac{e^{x_i}}{Z}\right) \tag{8}\\
&= \frac{e^{x_i} \cdot Z - e^{x_i} \cdot \frac{\partial Z}{\partial x_i}}{Z^2} \tag{9}\\
&= \frac{e^{x_i} \cdot Z - e^{x_i} \cdot e^{x_i}}{Z^2} \tag{10}\\
&= \frac{e^{x_i}}{Z} \cdot \frac{Z - e^{x_i}}{Z} \tag{11}\\
&= s_i \cdot (1 - s_i) \tag{12}
\end{align}

**情况2**: 当 $i \neq j$ 时：
\begin{align}
\frac{\partial s_i}{\partial x_j} &= \frac{\partial}{\partial x_j}\left(\frac{e^{x_i}}{Z}\right) \tag{13}\\
&= \frac{0 \cdot Z - e^{x_i} \cdot \frac{\partial Z}{\partial x_j}}{Z^2} \tag{14}\\
&= \frac{-e^{x_i} \cdot e^{x_j}}{Z^2} \tag{15}\\
&= -\frac{e^{x_i}}{Z} \cdot \frac{e^{x_j}}{Z} \tag{16}\\
&= -s_i \cdot s_j \tag{17}
\end{align}

综合两种情况，得到：
\begin{equation}
\frac{\partial s_i}{\partial x_j} = s_i \cdot (\delta_{ij} - s_j)
\tag{18}
\end{equation}

**矩阵形式**: 令 $\boldsymbol{s} = \text{softmax}(\boldsymbol{x})$，Jacobian矩阵为：
\begin{equation}
\boldsymbol{J} = \text{diag}(\boldsymbol{s}) - \boldsymbol{s}\boldsymbol{s}^{\top}
\tag{19}
\end{equation}

这是一个对称矩阵，且半正定（所有特征值非负）。

#### 1.3 反向传播中的梯度计算

在神经网络中，设损失函数为 $\mathcal{L}$，则通过链式法则：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{i=1}^n \frac{\partial \mathcal{L}}{\partial s_i} \cdot \frac{\partial s_i}{\partial x_j} = \sum_{i=1}^n \frac{\partial \mathcal{L}}{\partial s_i} \cdot s_i \cdot (\delta_{ij} - s_j)
\tag{20}
\end{equation}

展开得：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial x_j} = \frac{\partial \mathcal{L}}{\partial s_j} \cdot s_j - s_j \sum_{i=1}^n \frac{\partial \mathcal{L}}{\partial s_i} \cdot s_i
\tag{21}
\end{equation}

**特殊情况（交叉熵损失）**: 当 $\mathcal{L} = -\log s_y$（$y$是真实类别）时：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial s_i} = -\frac{\delta_{iy}}{s_y}
\tag{22}
\end{equation}

代入得：
\begin{align}
\frac{\partial \mathcal{L}}{\partial x_j} &= -\frac{\delta_{jy}}{s_y} \cdot s_j - s_j \sum_{i=1}^n \left(-\frac{\delta_{iy}}{s_y}\right) \cdot s_i \tag{23}\\
&= -\delta_{jy} + s_j \tag{24}\\
&= s_j - \delta_{jy} \tag{25}
\end{align}

这个简洁的形式是Softmax与交叉熵搭配的一个重要优势。

### 二、Attention机制中的Softmax分析

#### 2.1 标准Attention的数学形式

给定Query、Key、Value矩阵 $Q, K, V \in \mathbb{R}^{n \times d}$，标准的Scaled Dot-Product Attention定义为：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
\tag{26}
\end{equation}

对于第 $i$ 个位置，注意力权重为：
\begin{equation}
a_{i,j} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}{\sum_{j'=1}^n \exp\left(\frac{q_i \cdot k_{j'}}{\sqrt{d}}\right)}, \quad j = 1, \ldots, n
\tag{27}
\end{equation}

输出为：
\begin{equation}
o_i = \sum_{j=1}^n a_{i,j} v_j
\tag{28}
\end{equation}

#### 2.2 归一化因子的量级分析

**问题**: 归一化因子 $Z_i = \sum_{j=1}^n \exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)$ 与序列长度 $n$ 的关系？

**分析**: 如果Attention具有"聚焦"特性（即只关注少数几个token），存在常数 $K$ 使得：
\begin{equation}
\exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) \approx 0, \quad \text{当 } |i-j| > K
\tag{29}
\end{equation}

则：
\begin{equation}
Z_i \approx \sum_{j:|i-j|\leq K} \exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) = \mathcal{O}(K)
\tag{30}
\end{equation}

因此 $Z_i$ 与 $n$ **无关**或弱相关，这是一个关键观察。

**实证支持**:
1. Local Attention的成功表明大部分权重集中在局部
2. Attention矩阵的稀疏性可视化
3. RoPE等位置编码的远程衰减特性

#### 2.3 Softmax vs. 其他归一化方法

**方案1**: GAU使用的归一化（原文提到）：
\begin{equation}
\boldsymbol{A} = \frac{1}{n}\text{relu}^2\left(\frac{QK^{\top}}{\sqrt{d}}\right)
\tag{31}
\end{equation}

**问题分析**:
- 归一化因子 $n$ 是序列长度，与位置无关
- 但实际有效的注意力范围可能远小于 $n$
- 导致长度泛化能力差

**方案2**: 动态归一化：
\begin{equation}
a_{i,j} = \frac{\text{relu}^2\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}{\sum_{j'=1}^n \text{relu}^2\left(\frac{q_i \cdot k_{j'}}{同\sqrt{d}}\right)}
\tag{32}
\end{equation}

这与Softmax在形式上类似，但：
- $\text{relu}^2$ 具有正齐次性：$\text{relu}^2(\lambda x) = \lambda^2 \text{relu}^2(x)$
- 无法通过温度参数灵活调控

**方案3**: Softmax（标准方案）：
\begin{equation}
a_{i,j} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}{\sum_{j'=1}^n \exp\left(\frac{q_i \cdot k_{j'}}{\sqrt{d}}\right)}
\tag{33}
\end{equation}

优势：
- 归一化因子自适应于有效注意力范围
- 支持温度参数调控
- 与位置编码配合良好

### 三、熵不变性理论

#### 3.1 信息熵的定义

给定离散概率分布 $\boldsymbol{p} = (p_1, \ldots, p_n)$，Shannon熵定义为：
\begin{equation}
H(\boldsymbol{p}) = -\sum_{i=1}^n p_i \log p_i
\tag{34}
\end{equation}

对于Softmax输出的分布：
\begin{equation}
H(\text{softmax}(\boldsymbol{x})) = -\sum_{i=1}^n \frac{e^{x_i}}{\sum_j e^{x_j}} \cdot \log\frac{e^{x_i}}{\sum_j e^{x_j}}
\tag{35}
\end{equation}

化简：
\begin{align}
H &= -\sum_{i=1}^n \frac{e^{x_i}}{Z} \cdot \left(x_i - \log Z\right) \tag{36}\\
&= -\frac{1}{Z}\sum_{i=1}^n e^{x_i} x_i + \log Z \cdot \frac{1}{Z}\sum_{i=1}^n e^{x_i} \tag{37}\\
&= \log Z - \frac{1}{Z}\sum_{i=1}^n e^{x_i} x_i \tag{38}
\end{align}

其中 $Z = \sum_j e^{x_j}$ 是配分函数。

#### 3.2 温度参数对熵的影响

引入温度参数 $\tau$：
\begin{equation}
H_\tau = \log Z_\tau - \frac{1}{\tau Z_\tau}\sum_{i=1}^n e^{x_i/\tau} x_i
\tag{39}
\end{equation}

其中 $Z_\tau = \sum_j e^{x_j/\tau}$。

**定理（熵的单调性）**:
\begin{equation}
\frac{\partial H_\tau}{\partial \tau} > 0
\tag{40}
\end{equation}

即温度越高，熵越大（分布越平滑）。

**极限情况**:
- $\tau \to 0$: $H \to 0$（one-hot分布，熵最小）
- $\tau \to \infty$: $H \to \log n$（均匀分布，熵最大）

#### 3.3 长度外推的熵不变性方案

**问题**: 训练长度为 $n_0$，测试长度为 $n$，如何保持熵的一致性？

**方案**: 调整温度参数使得期望熵保持不变：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{\log_{n_0} n}{\sqrt{d}} QK^{\top}\right)V
\tag{41}
\end{equation}

**理论分析**: 假设注意力分数 $s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$ 的分布与长度无关，则：

当序列长度从 $n_0$ 变为 $n$ 时，有效的非零项数量从 $\mathcal{O}(K)$ 可能增加。为保持熵的量级，我们需要：
\begin{equation}
H_n \approx H_{n_0}
\tag{42}
\end{equation}

通过调整温度 $\tau = \frac{1}{\log_{n_0} n}$，可以部分补偿长度变化带来的熵变化。

**数值验证**: 设 $n_0 = 512, n = 2048$，则：
\begin{equation}
\tau = \frac{1}{\log_{512} 2048} = \frac{1}{\frac{\log 2048}{\log 512}} = \frac{\log 512}{\log 2048} = \frac{9\log 2}{11\log 2} \approx 0.818
\tag{43}
\end{equation}

即对于4倍长度，温度缩放到约0.82倍。

### 四、数值稳定性分析

#### 4.1 Softmax的数值稳定实现

**问题**: 直接计算 $e^{x_i}$ 可能导致：
- 上溢：$x_i$ 很大时，$e^{x_i} > \text{FLOAT_MAX}$
- 下溢：$x_i$ 很小时，$e^{x_i} \approx 0$

**解决方案**: 利用平移不变性 $\eqref{eq:3}$，减去最大值：
\begin{equation}
\tilde{x}_i = x_i - \max_j x_j
\tag{44}
\end{equation}

则：
\begin{equation}
\text{softmax}(\boldsymbol{x})_i = \frac{e^{\tilde{x}_i}}{\sum_j e^{\tilde{x}_j}}
\tag{45}
\end{equation}

**数值性质**:
- $\tilde{x}_i \leq 0, \forall i$，避免上溢
- $\max_i \tilde{x}_i = 0$，至少有一项为 $e^0 = 1$，避免全部下溢

**伪代码**:
```
def stable_softmax(x):
    x_max = max(x)
    x_shifted = x - x_max
    exp_x = exp(x_shifted)
    return exp_x / sum(exp_x)
```

#### 4.2 Log-Softmax的稳定计算

在计算交叉熵损失时，需要 $\log(\text{softmax}(\boldsymbol{x})_i)$：
\begin{align}
\log(\text{softmax}(\boldsymbol{x})_i) &= \log\frac{e^{x_i}}{\sum_j e^{x_j}} \tag{46}\\
&= x_i - \log\sum_j e^{x_j} \tag{47}
\end{align}

**Log-Sum-Exp技巧**:
\begin{align}
\log\sum_j e^{x_j} &= \log\left(e^{x_{\max}}\sum_j e^{x_j - x_{\max}}\right) \tag{48}\\
&= x_{\max} + \log\sum_j e^{x_j - x_{\max}} \tag{49}
\end{align}

其中 $x_{\max} = \max_j x_j$。

**完整的稳定实现**:
\begin{equation}
\log(\text{softmax}(\boldsymbol{x})_i) = (x_i - x_{\max}) - \log\sum_j e^{x_j - x_{\max}}
\tag{50}
\end{equation}

#### 4.3 梯度的数值稳定性

Softmax的梯度计算 $\eqref{eq:18}$ 本身是数值稳定的，因为：
- $s_i \in (0, 1)$，不会溢出
- $(1 - s_i) \in (0, 1)$，同样稳定

但在反向传播中，需要注意：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial x_j} = s_j - \delta_{jy}
\tag{51}
\end{equation}

这个形式非常稳定，因为：
- $s_j \in (0, 1)$
- $\delta_{jy} \in \{0, 1\}$
- 差值在 $(-1, 1)$ 范围内

### 五、复杂度分析

#### 5.1 标准Attention的复杂度

**时间复杂度**:
\begin{equation}
\mathcal{O}(n^2 d + n^2 + nd) = \mathcal{O}(n^2 d)
\tag{52}
\end{equation}

详细分解：
- 计算 $QK^{\top}$: $\mathcal{O}(n^2 d)$
- 计算Softmax: $\mathcal{O}(n^2)$
- 计算注意力输出: $\mathcal{O}(n^2 d)$

**空间复杂度**:
\begin{equation}
\mathcal{O}(n^2 + nd) = \mathcal{O}(n^2)
\tag{53}
\end{equation}

- 存储注意力矩阵: $\mathcal{O}(n^2)$
- 存储 $Q, K, V$: $\mathcal{O}(nd)$

#### 5.2 优化方案的复杂度对比

**Flash Attention**: 通过分块计算避免存储完整注意力矩阵
- 时间: $\mathcal{O}(n^2 d)$（不变）
- 空间: $\mathcal{O}(nd)$（降为线性！）

**Linear Attention**: 使用核方法近似
- 时间: $\mathcal{O}(nd^2)$
- 空间: $\mathcal{O}(nd)$

当 $n \gg d$ 时，Linear Attention更优。

**Sparse Attention**: 只计算部分注意力权重
- 时间: $\mathcal{O}(nkd)$，其中 $k \ll n$ 是每个位置关注的token数
- 空间: $\mathcal{O}(nk)$

### 六、激活函数对比实验

#### 6.1 理论对比

| 激活函数 | 公式 | 归一化性质 | 温度调控 | 梯度性质 |
|---------|------|-----------|---------|---------|
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | 自适应 | 支持 | 光滑 |
| ReLU² | $\frac{\text{relu}^2(x_i)}{\sum_j \text{relu}^2(x_j)}$ | 需手动设置 | 不支持 | 非光滑 |
| Sigmoid | $\sigma(x_i) = \frac{1}{1+e^{-x_i}}$ | 无（不归一） | 支持 | 光滑但饱和 |
| Tanh | $\tanh(x_i)$ | 无（不归一） | 支持 | 光滑但饱和 |

#### 6.2 齐次性分析

**Softmax**: 不具有齐次性
\begin{equation}
\text{softmax}(\lambda \boldsymbol{x}) \neq \lambda^k \text{softmax}(\boldsymbol{x})
\tag{54}
\end{equation}

这使得温度参数有效。

**ReLU²**: 具有2次齐次性
\begin{equation}
\text{relu}^2(\lambda x) = \lambda^2 \text{relu}^2(x)
\tag{55}
\end{equation}

归一化后：
\begin{equation}
\frac{\text{relu}^2(\lambda x_i)}{\sum_j \text{relu}^2(\lambda x_j)} = \frac{\lambda^2 \text{relu}^2(x_i)}{\sum_j \lambda^2 \text{relu}^2(x_j)} = \frac{\text{relu}^2(x_i)}{\sum_j \text{relu}^2(x_j)}
\tag{56}
\end{equation}

温度参数被抵消，无法调控！

#### 6.3 梯度特性对比

**Softmax梯度** $\eqref{eq:18}$:
- 处处可导
- 梯度有界：$|s_i(1-s_i)| \leq 1/4$
- 自动归一化（梯度和为0）

**ReLU²梯度**:
\begin{equation}
\frac{\partial \text{relu}^2(x)}{\partial x} = \begin{cases}
2x, & x > 0 \\
0, & x \leq 0
\end{cases}
\tag{57}
\end{equation}

- 在0点不可导
- 梯度无界（可能导致梯度爆炸）
- 稀疏性（负值梯度为0）

### 七、实践建议与超参数选择

#### 7.1 标准Attention的最佳实践

**1. Scale因子选择**:
\begin{equation}
\alpha = \frac{1}{\sqrt{d_k}}
\tag{58}
\end{equation}

**理论依据**: 假设 $q_i, k_j$ 的每个分量独立同分布，均值为0，方差为 $\sigma^2$，则：
\begin{equation}
\mathbb{E}[q_i \cdot k_j] = 0, \quad \text{Var}(q_i \cdot k_j) = d_k \sigma^2
\tag{59}
\end{equation}

除以 $\sqrt{d_k}$ 使方差归一化：
\begin{equation}
\text{Var}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) = \sigma^2
\tag{60}
\end{equation}

这避免了Softmax输入方差随维度增长。

**2. 温度参数调整**:

对于长度外推，使用：
\begin{equation}
\tau = \frac{\log n_{\text{train}}}{\log n_{\text{test}}}
\tag{61}
\end{equation}

**3. Dropout位置**:

推荐在Attention权重上应用Dropout：
\begin{equation}
\text{Attention}(Q,K,V) = \text{Dropout}(\text{softmax}(QK^{\top}/\sqrt{d}))V
\tag{62}
\end{equation}

典型值：$p_{\text{dropout}} = 0.1$

#### 7.2 数值稳定性检查清单

1. 使用Log-Softmax计算交叉熵损失
2. 梯度裁剪：`clip_grad_norm_(parameters, max_norm=1.0)`
3. 混合精度训练时注意Softmax精度
4. 检查Attention矩阵的最大值/最小值

#### 7.3 性能优化建议

**内存优化**:
- 使用Flash Attention减少显存占用
- Gradient Checkpointing牺牲计算换显存

**计算优化**:
- 使用fused kernels（Softmax与Scale合并）
- 利用硬件加速（Tensor Cores）

**多头注意力配置**:
\begin{equation}
h \times d_k = d_{\text{model}}
\tag{63}
\end{equation}

常见配置：
- 小模型：$h=8, d_k=64, d_{\text{model}}=512$
- 大模型：$h=16, d_k=64, d_{\text{model}}=1024$

### 八、总结与展望

#### 8.1 核心结论

1. **Softmax的理论优势**:
   - 自适应的归一化因子，与有效注意力范围匹配
   - 支持温度参数调控，实现熵的灵活控制
   - 数学性质优良（光滑、可导、有界）

2. **与其他激活函数的对比**:
   - ReLU²等幂函数因齐次性无法有效调控温度
   - Softmax的指数形式是实现温度调控的最自然选择

3. **长度泛化的关键**:
   - 归一化因子应与实际注意力范围匹配
   - 熵不变性提供了一个理论框架

#### 8.2 开放问题

1. 是否存在比Softmax更优的归一化方案？
2. 如何理论化地预测最优温度参数？
3. 熵不变性是否是长度外推的充分条件？

#### 8.3 扩展阅读

- Flash Attention: 高效的注意力计算
- Linformer: 线性复杂度的近似方法
- ALiBi: 基于偏置的位置编码方案

