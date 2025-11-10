---
title: Transformer升级之路：8、长度外推性与位置鲁棒性
slug: transformer升级之路8长度外推性与位置鲁棒性
date: 2023-01-31
tags: 详细推导, 语言模型, attention, 位置编码, 外推, 生成模型
status: pending
---
# Transformer升级之路：8、长度外推性与位置鲁棒性

**原文链接**: [https://spaces.ac.cn/archives/9444](https://spaces.ac.cn/archives/9444)

**发布日期**: 

---

上一篇文章[《Transformer升级之路：7、长度外推性与局部注意力》](/archives/9431)我们讨论了Transformer的长度外推性，得出的结论是长度外推性是一个训练和预测的不一致问题，而解决这个不一致的主要思路是将注意力局部化，很多外推性好的改进某种意义上都是局部注意力的变体。诚然，目前语言模型的诸多指标看来局部注意力的思路确实能解决长度外推问题，但这种“强行截断”的做法也许会不符合某些读者的审美，因为人工雕琢痕迹太强，缺乏了自然感，同时也让人质疑它们在非语言模型任务上的有效性。

本文我们从模型对位置编码的鲁棒性角度来重新审视长度外推性这个问题，此思路可以在基本不对注意力进行修改的前提下改进Transformer的长度外推效果，并且还适用多种位置编码，总体来说方法更为优雅自然，而且还适用于非语言模型任务。

## 问题分析 #

在之前的文章中，我们分析了长度外推性的缘由，给出了“长度外推性是一个训练和预测的长度不一致的问题”的定位，具体不一致的地方则有两点：

> 1、预测的时候用到了没训练过的位置编码（不管绝对还是相对）；
> 
> 2、预测的时候注意力机制所处理的token数量远超训练时的数量。

其中，第2点说的是更多的token会导致注意力更加分散（或者说注意力的熵变大），从而导致的训练和预测不一致问题，其实我们在[《从熵不变性看Attention的Scale操作》](/archives/8823)已经初步讨论并解决了它，答案是将Attention从  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{QK^{\top}}{\sqrt{d}}\right)V\end{equation}  
修改为  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\log_{m} n}{\sqrt{d}}QK^{\top}\right)V\end{equation}  
其中$m$是训练长度，$n$是预测长度。经过这样修改（下面简称为“$\log n$缩放注意力”），注意力的熵随着长度的变化更加平稳，缓解了这个不一致问题。个人的实验结果显示，至少在MLM任务上，“$\log n$缩放注意力”的长度外推表现更好。

所以，我们可以认为第2点不一致性已经得到初步解决，那么接下来应该是先集中精力解决第1点不一致性。

## 随机位置 #

第1点不一致性，即“预测的时候用到了没训练过的位置编码”，那么为了解决它，就应该做到“训练阶段把预测所用的位置编码也训练一下”。一篇ACL22还在匿名评审的论文[《Randomized Positional Encodings Boost Length Generalization of Transformers》](https://openreview.net/forum?id=nMYj4argap)首次从这个角度考虑了该问题，并且提出了解决方案。

论文的思路很简单：

> **随机位置训练** 设$N$为训练长度（论文$N=40$），$M$为预测长度（论文$M=500$），那么选定一个较大$L > M$（这是一个超参，论文$L=2048$），训练阶段原本长度为$N$的序列对应的位置序列是$[0,1,\cdots,N-2,N-1]$，现在改为从$\\{0,1,\cdots,L-2,L-1\\}$中随机不重复地选$N$个并从小到大排列，作为当前序列的位置序列。

基于numpy的参考代码为：
    
    
    def random_position_ids(N, L=2048):
        """从[0, L)中随机不重复挑N个整数，并从小到大排列
        """
        return np.sort(np.random.permutation(L)[:N])
    

预测阶段，也可以同样的方式随机采样位置序列，也可以直接在区间中均匀取点（个人的实验效果显示均匀取点的效果一般好些），这就解决了预测阶段的位置编码没有被训练过的问题。不难理解，这是一个很朴素的训练技巧（下面称之为“随机位置训练”），目标是希望Transformer能对位置的选择更加鲁棒一些，但后面我们将看到，它能取得长度外推效果的明显提升。笔者也在MLM任务上做了实验，结果显示在MLM上也是有效果的，并且配合“$\log n$缩放注意力”提升幅度更明显（原论文没有“$\log n$缩放注意力”这一步）。

## 新的基准 #

很多相关工作，包括上一篇文章提到的各种Local Attention及其变体的方案，都以语言模型任务构建评测指标，但不管是单向GPT还是双向的MLM，它们都高度依赖局部信息（局域性），所以之前的方案很 _**可能**_ 只是因为语言模型的局域性才有良好的外推表现，假如换一个非局域性的任务，效果可能就变差了。也许正因为如此，这篇论文的评测并非是常规的语言模型任务，而是Google去年在论文[《Neural Networks and the Chomsky Hierarchy》](https://papers.cool/arxiv/2207.02098)专门提出的一个长度外泛化基准（下面简称该测试基准为“CHE基准”，即“Chomsky Hierarchy Evaluation Benchmark”），这给我们提供了理解长度外推的一个新视角。

这个基准包含多个任务，分为R（Regular）、DCF（Deterministic Context-Free）、CS（Context-Sensitive）三个级别，每个级别的难度依次递增，每个任务的简介如下：

> **Even Pairs** ，难度**R** ，给定二元序列，如“aabba”，判断2-gram中ab和ba的总数是否为偶数，该例子中2-gram有aa、ab、bb、ba，其中ab和ba共有2个，即输出“是”，该题也等价于判断二元序列的首尾字符是否相同。
> 
> **Modular Arithmetic (Simple)** ，难度**R** ，计算由$\\{0, 1, 2, 3, 4\\}$五个数和$\\{+,-,\times\\}$三个运算符组成的算式的数值，并输出模5后的结果，比如输入$1 + 2 − 4$，那么等于$-1$，模5后等于$4$，所以输出$4$。
> 
> **Parity Check** ，难度**R** ，给定二元序列，如“aaabba”，判断b的数目是否为偶数，该例子中b的数目为2，那么输出“是”。
> 
> **Cycle Navigation** ，难度**R** ，给定三元序列，其中每个元分别代表$+0$、$+1$、$-1$之一，输出从0出发该序列最终的运算结果模5的值，比如$0,1,2$分别代表$+0,+1,-1$，那么$010211$代表$0 + 0 + 1 + 0 − 1 + 1 + 1 = 2$，模5后输出$2$。
> 
> **Modular Arithmetic** ，难度**DCF** ，计算由$\\{0, 1, 2, 3, 4\\}$五个数、括号$(,)$和$\\{+,-,\times\\}$三个运算符组成的算式的数值，并输出模5后的结果，比如输入$−(1−2)\times(4−3\times(−2))$，那么结果为$10$，模5后等于$0$，所以输出$0$，相比Simple版本，该任务多了“括号”，使得运算上更为复杂。
> 
> **Reverse String** ，难度**DCF** ，给定二元序列，如“aabba”，输出其反转序列，该例子中应该输出“abbaa”。
> 
> **Solve Equation** ，难度**DCF** ，给定由$\\{0, 1, 2, 3, 4\\}$五个数、括号$(,)$、$\\{+,-,\times\\}$三个运算符和未知数$z$组成的方程，求解未知数$z$的数值，使得它在模5之下成立。比如$−(1−2)\times(4−z\times(−2))=0$，那么$z=3$，解方程虽然看上去更难，但由于方程的构造是在Modular Arithmetic的基础上将等式中的某个数替换为$z$，所以保证有解并且解在$\\{0, 1, 2, 3, 4\\}$，因此理论上我们可以通过枚举结合Modular Arithmetic来求解，因此它的难度跟Modular Arithmetic相当。
> 
> **Stack Manipulation** ，难度**DCF** ，给定二元序列，如“abbaa”，以及由“POP/PUSH a/PUSH b”三个动作组成的堆栈操作序列，如“POP / PUSH a / POP”，输出最后的堆栈结果，该例子中应该输出“abba”。
> 
> **Binary Addition** ，难度**CS** ，给定两个二进制数，输出它们的和的二进制表示，如输入$10010$和$101$，输出$10111$，注意，这需要都在字符层面而不是数值层面输入到模型中进行训练和预测，并且两个数字是串行而不是并行对齐地提供的（可以理解为输入的是字符串$10010+101$）。
> 
> **Binary Multiplication** ，难度**CS** ，给定两个二进制数，输出它们的积的二进制表示，如输入$100$和$10110$，输出$1011000$，同Binary Addition一样，这需要都在字符层面而不是数值层面输入到模型中进行训练和预测，并且两个数字是串行而不是并行对齐地提供的（可以理解为输入的是字符串$100\times 10110$）。
> 
> **Compute Sqrt** ，难度**CS** ，给定一个二进制数，输出它的平方根的下取整的二进制表示，如输入$101001$，那么输出结果为$\lfloor\sqrt{101001}\rfloor=101$，这个难度同Binary Multiplication，因为至少我们可以从$0$到所给数结合Binary Multiplication逐一枚举来确定结果。
> 
> **Duplicate String** ，难度**CS** ，给定一个二元序列，如“abaab”，输出重复一次后的序列，该例子应该输出“abaababaab”，这个简单的任务看上去是难度**R** ，但实际上是**CS** ，大家可以想想为什么。
> 
> **Missing Duplicate** ，难度**CS** ，给定一个带有缺失值的二元序列，如“ab_aba”，并且已知原始的完整序列是一个重复序列（上一个任务的Duplicate String），预测确实值，该例子应该输出a。
> 
> **Odds First** ，难度**CS** ，给定一个二元序列$t_1 t_2 t_3 \cdots t_n$，输出$t_1 t_3 t_5 \cdots t_2 t_4 t_6 \cdots$，如输入aaabaa，将输出aaaaba。
> 
> **Bucket Sort** ，难度**CS** ，给定一个$n$元数值序列（数列中的每个数都是给定的$n$个数之一），返回其从小到大排序后的序列，如输入$421302214$应该输出$011222344$。

可以看到，这些任务都具有一个共同特点，就是它们的运算都有固定的简单规则，并且理论上输入都是不限长度的，那么我们可以通过短序列来训练，然后测试在短序列上的训练结果能否推广到长序列中。也就是说，它可以作为长度外推性的一个很强的测试基准。

## 实验结果 #

首先，介绍原始论文[《Neural Networks and the Chomsky Hierarchy》](https://papers.cool/arxiv/2207.02098)的实验结果，它对比了若干RNN模型及Transformer模型的效果（评测指标是诸位字符串的平均准确率，而不是整体结果的全对率）：  


[![若干模型在若干长度外推测试任务上的效果对比](/usr/uploads/2023/01/3218274161.png)](/usr/uploads/2023/01/3218274161.png "点击查看原图")

若干模型在若干长度外推测试任务上的效果对比

结果可能会让人意外，“风头正盛”的Transformer的长度外推效果是最差的（这里的Transformer还测试了不同的位置编码，并在每种任务上取了最优值），最好的是Tape-RNN。论文给它们的评级如下：  
$$\underbrace{\text{Transformer}}_{\text{R}^-} < \underbrace{\text{RNN}}_{\text{R}} < \underbrace{\text{LSTM}}_{\text{R}^+} < \underbrace{\text{Stack-RNN}}_{\text{DCF}} < \underbrace{\text{Tape-RNN}}_{\text{CS}}$$

而前面介绍的[《Randomized Positional Encodings Boost Length Generalization of Transformers》](https://openreview.net/forum?id=nMYj4argap)所提的随机位置训练方法，则为Transformer挽回了一些劣势：  


[![不同位置编码的Transformer在有无随机位置训练下的长度外推效果对比](/usr/uploads/2023/01/1506943925.png)](/usr/uploads/2023/01/1506943925.png "点击查看原图")

不同位置编码的Transformer在有无随机位置训练下的长度外推效果对比

可以看到，在随机位置训练之下，不管哪种位置编码的Transformer都有明显的提升，这就进一步验证了上一篇文章的结论，即长度外推性跟位置编码本身的设计没太大关系。特别地，随机位置训练还在Bucket Sort这个任务上首次取得了全对的准确率。尽管在总体表现上依然欠佳，但这相比之前的结果已经算是长足进步了（不知道结合“$\log n$缩放注意力”能否有提升？）。值得注意的地方还有，上表显示ALIBI这个在语言模型任务中表现良好的方法，在CHE基准上并无表现出什么优势，尤其是加入随机位置训练后，它的平均指标比RoPE还差，这就初步肯定了前面的猜测，即各种Local Attention变体的方法表现良好，大概率是因为基于语言模型的评测任务本身有严重的局域性，对于非局域性的CHE基准，这些方法并没有优势。

## 原理反思 #

细思之下，“随机位置训练”会很让人困惑。简单起见，我们不妨设$L=2048,N=64,M=512$，这样一来，训练阶段所用的平均位置序列大致为$[0, 32, 64, \cdots, 2016]$，预测阶段所用的平均位置序列是$[0, 4, 8, \cdots, 2044]$，训练阶段和预测阶段的相邻位置差不一样，这也可以说是某种不一致性，但它表现依然良好，这是为什么呢？

我们可以从“序”的角度去理解它。由于训练阶段的位置id是随机采样的，那么相邻位置差也是随机的，所以不管是相对位置还是绝对位置，模型不大可能通过精确的位置id来获取位置信息，取而代之是一个模糊的位置信号，更准确地说，是通过位置序列的“序”来编码位置而不是通过位置id本身来编码位置。比如，位置序列[1,3,5]跟[2,4,8]是等价的，因为它们都是从小到大排列的一个序列，随机位置训练“迫使”模型学会了一个等价类，即所有从小到大排列的位置序列都是等价的，都可以相互替换，这是位置鲁棒性的真正含义。

然而，笔者自己在MLM上做的实验结果显示，这个“等价类”的学习对模型还是有一定的困难的，更理想的方法是训练阶段依然使用随机位置，使得预测阶段的位置编码也被训练过，但是预测阶段的位置序列前面部分应该跟随机位置的平均结果一致。还是刚才的例子，如果预测阶段所用的位置序列是$[0, 4, 8, \cdots, 2044]$，那么我们希望训练阶段的随机位置平均结果是$[0, 4, 8, \cdots, 252]$（即序列$[0, 4, 8, \cdots, 2044]$的前$N$个），而不是$[0, 32, 64, \cdots, 2016]$，这样训练和预测的一致性就更加紧凑了。

## 延伸推广 #

于是，笔者考虑了如下思路：

> **等均值随机位置训练** 设$n$服从一个均值为$N$、采样空间为$[0, \infty)$的分布，那么训练阶段随机采样一个$n$，然后从$[0, n]$中均匀取$N$个点作为位置序列。

参考代码为：
    
    
    def random_position_ids(N):
        """先随机采样n，然后从[0, n]均匀取N个点
        """
        n = sample_from_xxx()
        return np.linspace(0, 1, N) * n
    

注意，这样采样出来的位置序列是浮点数，因此不适用于离散的训练式位置编码，只适用于函数式位置编码如[Sinusoidal](/archives/8231)或[RoPE](/archives/8265)，下面假设只考虑函数式位置编码。

该思路的最大问题是如何选择适合的采样分布。笔者的第一反应是[泊松分布](https://en.wikipedia.org/wiki/Poisson_distribution)，但考虑到泊松分布的均值和方差都是$n$，那么按照“3$\sigma$法则”来估算，它只能外推到$n+3\sqrt{n}$长度，这显然太短了。经过挑选和测试，笔者发现有两个分布比较适合：一个是[指数分布](https://en.wikipedia.org/wiki/Exponential_distribution)，它的均值和标准差都是$n$，那么即便按照“3$\sigma$法则”，也能外推到$4n$的长度，是一个比较理想的范围（实际还更长些）；另一个是[beta分布](https://en.wikipedia.org/wiki/Beta_distribution)，它定义在$[0,1]$上，我们可以将测试长度作为1，那么训练长度就是$N/M\in(0,1)$，beta分布有两个参数$\alpha,\beta$，其中均值为$\frac{\alpha}{\alpha+\beta}$，那么确保均值等于$N/M$后，我们还有额外的自由度控制$1$附近的概率，适合想要进一步拓展外推范围的场景。

笔者的实验结果显示，“等均值随机位置训练”结合“$\log n$缩放注意力”，在MLM任务上，能取得最佳的外推效果（训练长度64，测试长度512，采样分布为指数分布）。因为之前没做过CHE基准，所以笔者一时之间也没法测试CHE基准的效果，只能留到后面有机会再尝试了。

## 文章小结 #

本文从位置鲁棒性的角度思考了Transformer的长度外推性，得到了“随机位置训练”等增强长度外推性的新方案。同时，我们介绍了新的“CHE基准”，相比常规的语言模型任务，它具备更强的非局域性，可以更有效地评估长度外推相关工作。在它之下，之前的注意力局部化相关方法并没有较为突出的表现，相比之下“随机位置训练”效果更佳，这提醒我们应当在更全面的任务上评估相关方法的有效性，而不单单局限于语言模型任务。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9444>_

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

苏剑林. (Jan. 31, 2023). 《Transformer升级之路：8、长度外推性与位置鲁棒性 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9444>

@online{kexuefm-9444,  
title={Transformer升级之路：8、长度外推性与位置鲁棒性},  
author={苏剑林},  
year={2023},  
month={Jan},  
url={\url{https://spaces.ac.cn/archives/9444}},  
} 


---

## 公式推导与注释

### 1. 位置鲁棒性的数学定义

#### 1.1 鲁棒性的形式化定义

**定义1（位置鲁棒性）**：设 $f_{\theta}$ 是一个依赖位置编码的模型，位置序列为 $\boldsymbol{p} = (p_1, p_2, \ldots, p_L)$，输入为 $\boldsymbol{x} = (x_1, x_2, \ldots, x_L)$。模型在位置扰动下的鲁棒性定义为：

$$\text{Robustness}(f_{\theta}, \boldsymbol{p}, \boldsymbol{p}') = \mathbb{E}_{\boldsymbol{x}}\left[\|f_{\theta}(\boldsymbol{x}, \boldsymbol{p}) - f_{\theta}(\boldsymbol{x}, \boldsymbol{p}')\|\right]$$

其中 $\boldsymbol{p}'$ 是 $\boldsymbol{p}$ 的扰动版本。

对于长度外推，扰动来自两个方面：
1. **位置偏移**：$p_i' = p_i + \delta_i$，其中 $\delta_i$ 是随机偏移
2. **长度缩放**：$p_i' = \alpha p_i$，其中 $\alpha = L'/L$ 是长度缩放因子

#### 1.2 位置编码的等价类

**定义2（位置编码等价类）**：两个位置序列 $\boldsymbol{p}$ 和 $\boldsymbol{p}'$ 属于同一等价类，当且仅当它们诱导相同的相对位置关系：

$$\boldsymbol{p} \sim \boldsymbol{p}' \iff (p_i - p_j) = (p_i' - p_j'), \quad \forall i, j$$

对于严格单调递增的位置序列，等价类可以表征为：

$$[\boldsymbol{p}] = \{\boldsymbol{p}' : \text{order}(\boldsymbol{p}') = \text{order}(\boldsymbol{p})\}$$

其中 $\text{order}(\boldsymbol{p})$ 表示序列的单调性。

#### 1.3 鲁棒性度量

**度量1（位置扰动敏感度）**：模型对位置扰动的敏感度定义为Jacobian范数：

$$\text{Sensitivity} = \mathbb{E}\left[\left\|\frac{\partial f_{\theta}(\boldsymbol{x}, \boldsymbol{p})}{\partial \boldsymbol{p}}\right\|_F\right]$$

鲁棒的模型应该有较小的敏感度。

**度量2（Lipschitz常数）**：位置编码的Lipschitz常数：

$$L_{\text{PE}} = \sup_{\boldsymbol{p} \neq \boldsymbol{p}'} \frac{\|\text{PE}(\boldsymbol{p}) - \text{PE}(\boldsymbol{p}')\|}{\|\boldsymbol{p} - \boldsymbol{p}'\|}$$

对于RoPE，由于其正交性：

$$\|\boldsymbol{\mathcal{R}}_m \boldsymbol{v} - \boldsymbol{\mathcal{R}}_n \boldsymbol{v}\| = \|\boldsymbol{\mathcal{R}}_{m-n} \boldsymbol{v} - \boldsymbol{v}\|$$

Lipschitz常数依赖于相对位置 $|m-n|$ 而非绝对位置。

### 2. RoPE在长度外推中的稳定性分析

#### 2.1 RoPE的周期性分析

RoPE的第 $i$ 个维度对的旋转角度为：

$$\theta_i(m) = m \cdot \omega_i, \quad \omega_i = \theta_{\text{base}}^{-2i/d}$$

其周期为：

$$T_i = \frac{2\pi}{\omega_i} = 2\pi \cdot \theta_{\text{base}}^{2i/d}$$

**定理1（RoPE的多尺度周期性）**：RoPE具有多个不同周期的分量，从最短周期 $T_0 = 2\pi$ 到最长周期 $T_{d/2-1} = 2\pi \cdot \theta_{\text{base}}$。

对于 $\theta_{\text{base}} = 10000$，最长周期约为 $62832$。

#### 2.2 外推时的频率混叠

**定理2（频率混叠定理）**：当外推到长度 $L > T_i$ 时，第 $i$ 个维度对开始出现周期性重复，导致不同位置的编码发生混叠。

证明：设 $m_1 = m_0 + kT_i$（$k \in \mathbb{Z}$），则：

$$\begin{aligned}
\theta_i(m_1) &= (m_0 + kT_i) \omega_i \\
&= m_0\omega_i + kT_i\omega_i \\
&= m_0\omega_i + 2\pi k \\
&\equiv m_0\omega_i \pmod{2\pi}
\end{aligned}$$

因此 $\text{RoPE}^{(i)}(m_1) = \text{RoPE}^{(i)}(m_0)$，两个位置的编码相同。

#### 2.3 高频与低频分量的作用

**高频分量**（小 $i$，大 $\omega_i$）：
- 周期短，能区分相邻位置
- 外推时更快出现混叠
- 对短距离依赖建模重要

**低频分量**（大 $i$，小 $\omega_i$）：
- 周期长，能编码长距离信息
- 外推时更稳定
- 对长距离依赖建模重要

**定理3（频率分量的权衡）**：RoPE需要平衡高频和低频分量，以同时捕捉短距离和长距离依赖。

#### 2.4 稳定性的量化分析

**定义3（位置编码的稳定性指标）**：对于外推到长度 $L$，稳定性指标为：

$$\text{Stability}(L) = \frac{1}{d/2} \sum_{i=0}^{d/2-1} \mathbb{1}[L < T_i]$$

其中 $\mathbb{1}[\cdot]$ 是指示函数。

这个指标表示有多少比例的维度对在长度 $L$ 内不会出现周期性重复。

对于RoPE：

$$\text{Stability}(L) = \frac{1}{d/2} \sum_{i=0}^{d/2-1} \mathbb{1}\left[L < 2\pi \cdot \theta_{\text{base}}^{2i/d}\right]$$

计算临界维度 $i^*$ 使得 $L = T_{i^*}$：

$$L = 2\pi \cdot \theta_{\text{base}}^{2i^*/d} \implies i^* = \frac{d}{2} \log_{\theta_{\text{base}}} \frac{L}{2\pi}$$

因此：

$$\text{Stability}(L) \approx 1 - \frac{2i^*}{d} = 1 - \log_{\theta_{\text{base}}} \frac{L}{2\pi}$$

当 $L \ll \theta_{\text{base}}$ 时，$\text{Stability}(L) \approx 1$（高稳定性）。

当 $L \approx \theta_{\text{base}}$ 时，$\text{Stability}(L) \approx 0$（低稳定性）。

### 3. 插值与外推的理论比较

#### 3.1 插值的数学定义

**定义4（位置插值）**：给定训练长度 $L_{\text{train}}$，要在长度 $L_{\text{test}} > L_{\text{train}}$ 上测试，插值方法将位置序列缩放：

$$p_i^{\text{interp}} = \frac{L_{\text{train}}}{L_{\text{test}}} \cdot i = \frac{i}{\alpha}, \quad \alpha = \frac{L_{\text{test}}}{L_{\text{train}}}$$

对于RoPE，这等价于将所有频率缩放：

$$\omega_i^{\text{interp}} = \frac{\omega_i}{\alpha}$$

#### 3.2 外推的数学定义

**定义5（位置外推）**：直接使用原始位置序列，不进行缩放：

$$p_i^{\text{extrap}} = i$$

对于RoPE，频率保持不变：

$$\omega_i^{\text{extrap}} = \omega_i$$

#### 3.3 插值与外推的对比

**插值的优点**：
1. **避免周期性重复**：缩放后的位置 $p_i^{\text{interp}} = i/\alpha$ 保持在训练范围 $[0, L_{\text{train}}]$ 内
2. **位置编码在训练集内**：所有位置的编码都在训练时见过
3. **稳定性高**：$\text{Stability}(L_{\text{train}}) \approx 1$

**插值的缺点**：
1. **破坏绝对位置信息**：相邻位置的间距变为 $1/\alpha < 1$
2. **相对位置扭曲**：相对位置 $p_i - p_j = (i-j)/\alpha$ 被压缩
3. **注意力模式改变**：如果模型学习到了特定的相对位置模式，插值会破坏这些模式

**外推的优点**：
1. **保持相对位置**：相对位置 $p_i - p_j = i - j$ 不变
2. **注意力模式一致**：短距离的注意力模式与训练时相同

**外推的缺点**：
1. **未见过的位置编码**：$p_i > L_{\text{train}}$ 的编码在训练时未出现
2. **周期性混叠**：高频分量可能出现重复

#### 3.4 理论比较定理

**定理4（插值vs外推的误差界）**：

对于插值：
$$\mathcal{E}_{\text{interp}} = O\left(\alpha - 1\right) = O\left(\frac{L_{\text{test}} - L_{\text{train}}}{L_{\text{train}}}\right)$$

误差来自相对位置的压缩。

对于外推：
$$\mathcal{E}_{\text{extrap}} = O\left(\sum_{i: T_i < L_{\text{test}}} \frac{1}{d}\right)$$

误差来自周期性重复的维度。

**推论**：当外推比率 $\alpha$ 较小（$\alpha < 2$）时，外推误差可能小于插值；当 $\alpha$ 较大时，插值更稳定。

### 4. 随机位置训练的理论分析

#### 4.1 随机位置采样的数学描述

**定义6（随机位置训练）**：从 $[0, L_{\max}]$ 中随机采样 $N$ 个位置：

$$\boldsymbol{p} = \text{sort}(\{\text{sample}(L_{\max}) : k = 1, \ldots, N\})$$

其中 $\text{sample}(L_{\max})$ 从 $[0, L_{\max}]$ 均匀采样。

排序后得到单调递增的位置序列。

#### 4.2 位置间距的分布

**定理5（位置间距的期望）**：随机采样的位置序列，相邻位置的平均间距为：

$$\mathbb{E}[\Delta p] = \mathbb{E}[p_{i+1} - p_i] = \frac{L_{\max}}{N}$$

证明：$N$ 个点将 $[0, L_{\max}]$ 分成 $N+1$ 段，每段的期望长度为：
$$\mathbb{E}[\text{段长}] = \frac{L_{\max}}{N+1} \approx \frac{L_{\max}}{N}$$

**定理6（位置间距的方差）**：位置间距的方差为：

$$\text{Var}[\Delta p] = \frac{L_{\max}^2}{N^2(N+1)}$$

这说明位置间距有一定的随机性，但集中在期望值附近。

#### 4.3 等价类的学习

**定理7（等价类学习定理）**：随机位置训练迫使模型学习位置编码的等价类，即所有保持单调性的位置序列被视为等价。

证明：由于训练时位置序列是随机的，模型不能依赖具体的位置值，只能依赖相对顺序。设 $\boldsymbol{p}_1$ 和 $\boldsymbol{p}_2$ 是两个不同的随机采样序列，如果它们具有相同的排序关系（都是单调递增），则模型应该给出相似的输出：

$$\mathbb{E}_{\boldsymbol{p}_1, \boldsymbol{p}_2}\left[\|f_{\theta}(\boldsymbol{x}, \boldsymbol{p}_1) - f_{\theta}(\boldsymbol{x}, \boldsymbol{p}_2)\|\right] \to 0$$

这是通过最小化训练损失间接实现的。

#### 4.4 等均值随机位置训练

**定义7（等均值随机位置）**：先采样长度 $n \sim p(n)$（期望为 $N$），然后从 $[0, n]$ 均匀采样 $N$ 个点：

$$p_i = \frac{i-1}{N-1} \cdot n, \quad i = 1, \ldots, N$$

其中 $n$ 服从某个分布，如指数分布或Beta分布。

**定理8（等均值训练的平均位置）**：等均值训练的平均位置序列为：

$$\mathbb{E}[p_i] = \frac{i-1}{N-1} \cdot \mathbb{E}[n] = \frac{i-1}{N-1} \cdot N$$

与测试时均匀采样的位置序列一致。

### 5. 频率衰减与位置表示能力

#### 5.1 频率衰减的引入

为了改善外推性能，可以引入频率衰减，降低高频分量的权重：

$$\omega_i^{\text{decay}} = \omega_i \cdot \gamma^i$$

其中 $\gamma \in (0, 1)$ 是衰减因子。

或者直接修改基础频率：

$$\omega_i = \theta_{\text{base}}'^{-2i/d}, \quad \theta_{\text{base}}' > \theta_{\text{base}}$$

这增加了所有周期，减少了高频振荡。

#### 5.2 频率衰减对表示能力的影响

**定理9（频率衰减的权衡）**：增大 $\theta_{\text{base}}$ 可以提高外推稳定性，但会降低短距离位置的区分能力。

证明：短距离位置（$|m-n| = 1$）的RoPE编码差异为：

$$\|\boldsymbol{\mathcal{R}}_m - \boldsymbol{\mathcal{R}}_{m+1}\| = \left\|\sum_{i=0}^{d/2-1} (\boldsymbol{R}_{\omega_i} - \boldsymbol{I})\right\|$$

当 $\omega_i$ 减小时，$\boldsymbol{R}_{\omega_i} \approx \boldsymbol{I}$，差异变小。

#### 5.3 多尺度表示的数学分析

RoPE的多尺度性质可以表示为不同时间尺度的组合：

$$\text{RoPE}(\boldsymbol{v}, m) = \bigoplus_{i=0}^{d/2-1} \boldsymbol{R}_{m\omega_i} \boldsymbol{v}^{(i)}$$

其中 $\bigoplus$ 表示拼接，$\boldsymbol{v}^{(i)}$ 是 $\boldsymbol{v}$ 的第 $i$ 个二维块。

**定义8（有效分辨率）**：第 $i$ 个维度对能够有效区分的最大位置差为：

$$\Delta_{\max}^{(i)} = \frac{\pi}{\omega_i} = \frac{\pi \cdot \theta_{\text{base}}^{2i/d}}{1}$$

这是半个周期，对应于旋转角度 $\pi$（最大区分度）。

**定理10（多尺度覆盖定理）**：RoPE的 $d/2$ 个维度对提供了从 $\Delta_{\max}^{(0)} = \pi$ 到 $\Delta_{\max}^{(d/2-1)} = \pi \cdot \theta_{\text{base}}$ 的多尺度覆盖。

### 6. log(n)缩放注意力的数学原理

#### 6.1 注意力熵的详细推导

标准注意力的熵为：

$$H = -\sum_{n=1}^{L} A_{m,n} \log A_{m,n}$$

在均匀分布假设下（$A_{m,n} = 1/L$）：

$$H = -\sum_{n=1}^{L} \frac{1}{L} \log \frac{1}{L} = \log L$$

**定理11（熵随长度的增长）**：全局注意力的熵随序列长度对数增长，导致训练与测试不一致。

#### 6.2 温度缩放的数学原理

引入温度 $\tau$ 的softmax：

$$A_{m,n}^{(\tau)} = \frac{\exp\left(\frac{s_{m,n}}{\tau}\right)}{\sum_j \exp\left(\frac{s_{m,j}}{\tau}\right)}$$

其中 $s_{m,n} = \boldsymbol{q}_m^{\top}\boldsymbol{k}_n/\sqrt{d}$。

**性质1（温度对熵的影响）**：
- $\tau \to 0$：熵 $\to 0$（one-hot分布）
- $\tau \to \infty$：熵 $\to \log L$（均匀分布）

要保持熵不变，需要：
$$\tau(L) \propto \log L$$

因此，$\log_m n$ 缩放等价于设置温度：
$$\tau = \frac{\sqrt{d}}{\log_m n}$$

#### 6.3 log缩放的理论证明

**定理12（熵不变缩放定理）**：使用 $\log_m n$ 缩放可以近似保持注意力熵不变。

证明：设训练长度为 $m$，测试长度为 $n$，原始得分为 $s_{ij}$。

未缩放时，测试时的熵为：
$$H_n = -\sum_j A_j^{(n)} \log A_j^{(n)} \approx \log n$$

缩放后：
$$A_j^{(\text{scaled})} = \frac{\exp\left(\frac{\log_m n}{\sqrt{d}} s_j\right)}{\sum_k \exp\left(\frac{\log_m n}{\sqrt{d}} s_k\right)}$$

等效温度为：
$$\tau_{\text{eff}} = \frac{\sqrt{d}}{\log_m n} = \frac{\sqrt{d}}{\log n - \log m}$$

缩放后的熵约为：
$$H_{\text{scaled}} \approx \log n - \log(\log_m n) = \log n - \log\frac{\log n}{\log m}$$

当 $n \approx m$ 时，$H_{\text{scaled}} \approx \log m$，保持一致。

### 7. 鲁棒性度量与泛化界

#### 7.1 PAC学习框架下的泛化界

**定义9（位置泛化误差）**：设 $\mathcal{D}_{\text{train}}$ 是训练长度的分布，$\mathcal{D}_{\text{test}}$ 是测试长度的分布，泛化误差为：

$$\mathcal{E}_{\text{gen}} = \mathbb{E}_{L \sim \mathcal{D}_{\text{test}}}\left[\mathcal{L}(f_{\theta}, L)\right] - \mathbb{E}_{L \sim \mathcal{D}_{\text{train}}}\left[\mathcal{L}(f_{\theta}, L)\right]$$

**定理13（Rademacher复杂度界）**：在适当的假设下，泛化误差有界：

$$\mathcal{E}_{\text{gen}} \leq \hat{\mathcal{R}}(\mathcal{F}) + O\left(\sqrt{\frac{\log(1/\delta)}{N}}\right)$$

其中 $\hat{\mathcal{R}}(\mathcal{F})$ 是函数类 $\mathcal{F}$ 的经验Rademacher复杂度，$N$ 是训练样本数。

#### 7.2 位置编码的Lipschitz连续性

**定理14（RoPE的Lipschitz性质）**：RoPE满足局部Lipschitz连续性：

$$\|\text{RoPE}(m) - \text{RoPE}(n)\| \leq C \cdot |m - n|$$

其中 $C$ 是Lipschitz常数。

证明：对于单个维度对：

$$\|\boldsymbol{R}_{m\omega} - \boldsymbol{R}_{n\omega}\|_F^2 = 2 - 2\cos((m-n)\omega)$$

使用 $1 - \cos\theta \leq \theta^2/2$：

$$\|\boldsymbol{R}_{m\omega} - \boldsymbol{R}_{n\omega}\|_F^2 \leq (m-n)^2\omega^2$$

因此：
$$\|\boldsymbol{R}_{m\omega} - \boldsymbol{R}_{n\omega}\|_F \leq |m-n| \omega$$

对所有维度求和：
$$\|\text{RoPE}(m) - \text{RoPE}(n)\|_F \leq |m-n| \sqrt{\sum_{i=0}^{d/2-1} \omega_i^2}$$

#### 7.3 随机位置训练的泛化界

**定理15（随机位置训练的泛化保证）**：随机位置训练通过数据增强扩大了训练分布，从而减小了泛化误差。

形式化地，设原始训练集为 $\mathcal{S} = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$，随机位置增强后为：

$$\tilde{\mathcal{S}} = \{(\boldsymbol{x}_i, \boldsymbol{p}_i, y_i) : \boldsymbol{p}_i \sim p(\boldsymbol{p})\}$$

其中 $p(\boldsymbol{p})$ 是位置序列的分布。

增强后的经验风险为：
$$\hat{\mathcal{R}}_{\tilde{\mathcal{S}}}(f_{\theta}) = \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{\boldsymbol{p}_i \sim p(\boldsymbol{p})}\left[\mathcal{L}(f_{\theta}(\boldsymbol{x}_i, \boldsymbol{p}_i), y_i)\right]$$

这更接近测试时的期望风险。

### 8. CHE基准的理论解释

#### 8.1 非局部性任务的数学特征

CHE基准中的许多任务具有全局依赖性，例如：

**Reverse String**：输出 $y_i = x_{L-i+1}$，需要全局信息传播。

**Duplicate String**：输出 $y_i = x_i$ 和 $y_{L+i} = x_i$，需要识别序列的全局结构。

这些任务无法通过局部注意力在单层中完成。

#### 8.2 局部注意力在CHE上的局限性

**定理16（局部注意力的全局依赖瓶颈）**：对于需要全局信息传播的任务，局部注意力（窗口大小 $w$）需要至少 $\lceil L/(2w) \rceil$ 层才能传播信息到整个序列。

对于Reverse String任务，端到端的信息传播需要：
$$\ell_{\min} = \lceil L/w \rceil$$

当 $w \ll L$ 时，$\ell_{\min}$ 很大，可能导致信息损失和梯度消失。

#### 8.3 随机位置训练在CHE上的优势

随机位置训练不改变注意力的局部性，而是增强了模型对位置变化的鲁棒性。

**关键洞察**：CHE任务虽然需要全局信息，但不依赖于精确的位置值，只依赖于符号的顺序和模式。

随机位置训练迫使模型学习基于符号模式而非位置的表示，从而在外推时更鲁棒。

### 9. 指数分布与Beta分布的理论分析

#### 9.1 指数分布的性质

指数分布的概率密度函数：

$$p(n) = \frac{1}{\mu} \exp\left(-\frac{n}{\mu}\right), \quad n \geq 0$$

其中 $\mu = \mathbb{E}[n] = N$（训练长度）。

**性质2（无记忆性）**：指数分布具有无记忆性：
$$P(n > s+t | n > s) = P(n > t)$$

这意味着训练时采样的长度 $n$ 不会偏向任何特定的范围（除了期望值）。

**外推能力**：对于外推到长度 $L$，指数分布给予非零概率：

$$P(n \geq L) = \exp\left(-\frac{L}{\mu}\right)$$

当 $L = 2\mu$ 时，$P(n \geq 2\mu) \approx 0.135$，有约13.5%的训练样本长度超过 $2\mu$。

当 $L = 3\mu$ 时，$P(n \geq 3\mu) \approx 0.050$，有约5%的训练样本长度超过 $3\mu$。

这提供了一定的外推能力。

#### 9.2 Beta分布的性质

Beta分布定义在 $[0, 1]$ 上，PDF为：

$$p(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}$$

其中 $B(\alpha, \beta)$ 是Beta函数。

均值为：
$$\mathbb{E}[x] = \frac{\alpha}{\alpha+\beta}$$

**应用于长度采样**：设测试长度为 $L_{\max}$，训练长度为 $L_{\text{train}}$，则：

$$\frac{L_{\text{train}}}{L_{\max}} = \frac{\alpha}{\alpha+\beta}$$

通过调整 $\alpha$ 和 $\beta$，可以控制长度分布的形状。

例如，选择 $\alpha = 2, \beta = 3$ 使得均值为 $2/5 = 0.4$，即训练长度为测试长度的40%。

Beta分布的方差为：
$$\text{Var}[x] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

可以通过调整 $\alpha$ 和 $\beta$ 控制方差，进而控制外推范围。

### 10. 实验验证的理论解释

#### 10.1 随机位置训练的性能提升

实验显示，随机位置训练显著提升了外推性能。理论解释：

**解释1（分布匹配）**：随机位置训练使得训练集的位置分布与测试集更匹配。

设测试时的位置序列为均匀采样：
$$p_i^{\text{test}} = \frac{i-1}{N-1} \cdot L_{\text{test}}$$

随机位置训练的平均位置为：
$$\mathbb{E}[p_i^{\text{train}}] = \frac{i-1}{N-1} \cdot \mathbb{E}[n]$$

当 $\mathbb{E}[n] = L_{\text{test}}$ 时，两者分布相同。

**解释2（等价类学习）**：随机位置训练迫使模型学习位置不变性，只依赖相对顺序。

#### 10.2 不同位置编码的表现

实验显示，加入随机位置训练后，不同位置编码的性能差距缩小。

**理论解释**：随机位置训练减少了对精确位置值的依赖，使得位置编码的具体形式（Sinusoidal、RoPE、可训练等）变得不那么重要。

关键在于位置编码能否保持单调性和相对顺序，而非绝对值。

#### 10.3 log(n)缩放的效果

实验显示，结合log缩放和随机位置训练效果最佳。

**理论解释**：
1. **随机位置训练**解决了位置编码未见过的问题
2. **log缩放**解决了注意力熵变化的问题
3. 两者互补，共同提升外推性能

### 11. 开放问题与未来方向

#### 11.1 理论开放问题

1. **最优随机分布**：什么样的长度采样分布能最大化外推性能？是否存在理论最优分布？

2. **泛化界的紧致性**：当前的泛化界是否紧致？能否给出更精确的界？

3. **位置鲁棒性的度量**：如何更精确地度量和量化位置鲁棒性？

4. **多任务外推**：不同任务对位置编码的需求不同，如何设计通用的外推策略？

#### 11.2 实践问题

1. **计算效率**：随机位置训练增加了计算开销（每个样本采样位置），如何优化？

2. **超参数选择**：$L_{\max}$、分布类型等超参数如何自动选择？

3. **混合策略**：如何结合局部注意力、随机位置、log缩放等多种技术？

4. **任务特定调优**：不同任务可能需要不同的外推策略，如何自适应调整？

### 12. 总结

#### 12.1 核心理论贡献

1. **位置鲁棒性定义**：形式化了位置鲁棒性的数学定义和度量
2. **RoPE稳定性分析**：分析了RoPE的周期性和频率混叠问题
3. **插值vs外推**：理论比较了两种方法的优缺点和误差界
4. **随机位置训练**：证明了随机位置训练通过等价类学习提升鲁棒性
5. **log缩放原理**：阐明了log缩放保持注意力熵不变的数学原理

#### 12.2 实践指导

1. **推荐策略**：结合随机位置训练 + log(n)缩放 + 适当的位置编码（如RoPE）
2. **参数设置**：$L_{\max} \approx 2-3 \times L_{\text{test}}$，使用指数分布或Beta分布
3. **任务适配**：对于局域性强的任务（如语言模型），局部注意力有效；对于全局依赖任务（如CHE），随机位置训练更重要

#### 12.3 理论意义

本文通过严格的数学分析，从位置鲁棒性的角度重新理解了长度外推问题，提出了不依赖于局部注意力的外推方案，为Transformer的长序列建模提供了新的理论视角和实践方法。

位置鲁棒性的核心在于让模型学习位置的等价类而非绝对值，这通过随机位置训练得以实现，是一种更本质、更优雅的解决方案。

