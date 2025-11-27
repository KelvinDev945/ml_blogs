---
title: CAN：借助先验分布提升分类性能的简单后处理技巧
slug: can借助先验分布提升分类性能的简单后处理技巧
date: 2021-10-22
tags: 模型, 概率, 分析, 技巧, 生成模型
status: completed
---

# CAN：借助先验分布提升分类性能的简单后处理技巧

**原文链接**: [https://spaces.ac.cn/archives/8728](https://spaces.ac.cn/archives/8728)

**发布日期**: 

---

顾名思义，本文将会介绍一种用于分类问题的后处理技巧——CAN（Classification with Alternating Normalization），出自论文[《When in Doubt: Improving Classification Performance with Alternating Normalization》](https://papers.cool/arxiv/2109.13449)。经过笔者的实测，CAN确实多数情况下能提升多分类问题的效果，而且几乎没有增加预测成本，因为它仅仅是对预测结果的简单重新归一化操作。

有趣的是，其实CAN的思想是非常朴素的，朴素到每个人在生活中都应该用过同样的思想。然而，CAN的论文却没有很好地说清楚这个思想，只是纯粹形式化地介绍和实验这个方法。本文的分享中，将会尽量将算法思想介绍清楚。

## 思想例子 #

假设有一个二分类问题，模型对于输入$a$给出的预测结果是$p^{(a)} = [0.05, 0.95]$，那么我们就可以给出预测类别为$1$；接下来，对于输入$b$，模型给出的预测结果是$p^{(b)}=[0.5,0.5]$，这时候处于最不确定的状态，我们也不知道输出哪个类别好。

但是，假如我告诉你：1、类别必然是0或1其中之一；2、两个类别的出现概率各为0.5。在这两点先验信息之下，由于前一个样本预测结果为1，那么基于朴素的均匀思想，我们是否更倾向于将后一个样本预测为0，以得到一个满足第二点先验的预测结果？

这样的例子还有很多，比如做10道选择题，前9道你都比较有信心，第10题完全不会只能瞎蒙，然后你一看发现前9题选A、B、C的都有就是没有一个选D的，那么第10题在蒙的时候你会不会更倾向于选D？

这些简单例子的背后，有着跟CAN同样的思想，它其实就是 _用先验分布来校正低置信度的预测结果，使得新的预测结果的分布更接近先验分布。_

## 不确定性 #

准确来说， _CAN是针对低置信度预测结果的后处理手段_ ，所以我们首先要有一个衡量预测结果不确定性的指标。常见的度量是“熵”（参考[《“熵”不起：从熵、最大熵原理到最大熵模型（一）》](/archives/3534)），对于$p=[p_1,p_2,\cdots,p_m]$，定义为：  
\begin{equation}H(p) = -\sum_{i=1}^m p_i\log p_i\end{equation}  
然而，虽然熵是一个常见选择，但其实它得出的结果并不总是符合我们的直观理解。比如对于$p^{(a)}=[0.5,0.25,0.25]$和$p^{(b)}=[0.5,0.5,0]$，直接套用公式得到$H(p^{(a)}) > H(p^{(b)})$，但就我们的分类场景而言，显然我们会认为$p^{(b)}$比$p^{(a)}$更不确定，所以直接用熵还不够合理。

一个简单的修正是只用前top-$k$个概率值来算熵，不失一般性，假设$p_1,p_2,\cdots,p_k$是概率最高的$k$个值，那么  
\begin{equation}H_{\text{top-}k}(p) = -\sum_{i=1}^k \tilde{p}_i\log \tilde{p}_i\end{equation}  
其中$\tilde{p}_i=p_i\Big/ \sum\limits_{i=1}^k p_i$。为了得到一个0～1范围内的结果，我们取$H_{\text{top-}k}(p)/\log k$为最终的不确定性指标。

## 算法步骤 #

现在假设我们有$N$个样本需要预测类别，模型直接的预测结果是$N$个概率分布$p^{(1)},p^{(2)},\cdots,p^{(N)}$，假设测试样本和训练样本是同分布的，那么完美的预测结果应该有：  
\begin{equation}\frac{1}{N}\sum_{i=1}^N p^{(i)} = \tilde{p}\label{eq:prior}\end{equation}  
其中$\tilde{p}$是类别的先验分布，我们可以直接从训练集估计。也就是说，全体预测结果应该跟先验分布是一致的，但受限于模型性能等原因，实际的预测结果可能明显偏离上式，这时候我们就可以人为修正这部分。

具体来说，我们选定一个阈值$\tau$，将指标小于$\tau$的预测结果视为高置信度的，而大于等于$\tau$的则是低置信度的，不失一般性，我们假设前$n$个结果$p^{(1)},p^{(2)},\cdots,p^{(n)}$属于高置信度的，而剩下的$N-n$个属于低置信度的。我们认为高置信度部分是更加可靠的，所以它们不用修正，并且可以用它们来作为“标准参考系”来修正低置信度部分。

具体来说，对于$\forall j\in\\{n+1,n+2,\cdots,N\\}$，我们将$p^{(j)}$与高置信度的$p^{(1)},p^{(2)},\cdots,p^{(n)}$一起，执行一次**“行间”标准化** ：  
\begin{equation}p^{(k)} \leftarrow p^{(k)}\big/\bar{p}\times\tilde{p},\quad\bar{p}=\frac{1}{n+1}\left(p^{(j)} + \sum_{i=1}^n p^{(i)}\right)\label{eq:step-1}\end{equation}  
这里的$k\in\\{1,2,\cdots,n\\}\cup\\{j\\}$，其中乘除法都是element-wise的。不难发现，这个标准化的目的是使得所有新的$p^{(k)}$的平均向量等于先验分布$\tilde{p}$，也就是促使式$\eqref{eq:prior}$的成立。然而，这样标准化之后，每个$p^{(k)}$就未必满足归一化了，所以我们还要执行一次**“行内”标准化** ：  
\begin{equation}p^{(k)} \leftarrow \frac{p^{(k)}_i}{\sum\limits_{i=1}^m p^{(k)}_i}\label{eq:step-2}\end{equation}  
但这样一来，式$\eqref{eq:prior}$可能又不成立了。所以理论上我们可以交替迭代执行这两步，直到结果收敛（不过实验结果显示一般情况下一次的效果是最好的）。最后，我们只保留最新的$p^{(j)}$作为原来第$j$个样本的预测结果，其余的$p^{(k)}$均弃之不用。

注意，这个过程需要我们遍历每个低置信度结果$j\in\\{n+1,n+2,\cdots,N\\}$执行，也就是说是 _逐个样本进行修正_ ，而不是一次性修正的，每个$p^{(j)}$都借助**原始的** 高置信度结果$p^{(1)},p^{(2)},\cdots,p^{(n)}$组合来按照上述步骤迭代，虽然迭代过程中对应的$p^{(1)},p^{(2)},\cdots,p^{(n)}$都会随之更新，但那只是临时结果，最后都是弃之不用的，每次修正都是用原始的$p^{(1)},p^{(2)},\cdots,p^{(n)}$。

## 参考实现 #

这是笔者给出的参考实现代码：
    
    
    # 预测结果，计算修正前准确率
    y_pred = model.predict(
        valid_generator.fortest(), steps=len(valid_generator), verbose=True
    )
    y_true = np.array([d[1] for d in valid_data])
    acc_original = np.mean([y_pred.argmax(1) == y_true])
    print('original acc: %s' % acc_original)
    
    # 评价每个预测结果的不确定性
    k = 3
    y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
    y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
    y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)
    
    # 选择阈值，划分高、低置信度两部分
    threshold = 0.9
    y_pred_confident = y_pred[y_pred_uncertainty < threshold]
    y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
    y_true_confident = y_true[y_pred_uncertainty < threshold]
    y_true_unconfident = y_true[y_pred_uncertainty >= threshold]
    
    # 显示两部分各自的准确率
    # 一般而言，高置信度集准确率会远高于低置信度的
    acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
    acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
    print('confident acc: %s' % acc_confident)
    print('unconfident acc: %s' % acc_unconfident)
    
    # 从训练集统计先验分布
    prior = np.zeros(num_classes)
    for d in train_data:
        prior[d[1]] += 1.
    
    prior /= prior.sum()
    
    # 逐个修改低置信度样本，并重新评价准确率
    right, alpha, iters = 0, 1, 1
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y**alpha
            Y /= Y.mean(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        if y.argmax() == y_true_unconfident[i]:
            right += 1
    
    # 输出修正后的准确率
    acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
    print('new unconfident acc: %s' % (right / (i + 1.)))
    print('final acc: %s' % acc_final)

## 实验结果 #

那么，这样的简单后处理，究竟能带来多大的提升呢？原论文给出的实验结果是相当可观的：  


[![原论文的实验结果之一](/usr/uploads/2021/10/3322086689.png)](/usr/uploads/2021/10/3322086689.png "点击查看原图")

原论文的实验结果之一

笔者也在CLUE上的两个中文文本分类任务上做了实验，显示基本也有点提升，但没那么可观（验证集结果）：  
\begin{array}{c|c|c}  
\hline  
& \text{IFLYTEK(类别数:119)} & \text{TNEWS(类别数:15)}\\\  
\hline  
\text{BERT} & 60.06\% & 56.80\% \\\  
\text{BERT + CAN} & 60.52\% & 56.86\% \\\  
\hline  
\text{RoBERTa} & 60.64\% & 58.06\% \\\  
\text{RoBERTa + CAN} & 60.95\% & 58.00\% \\\  
\hline  
\end{array}

大体上来说，类别数目越多，效果提升越明显，如果类别数目比较少，那么可能提升比较微弱甚至会下降（当然就算下降也是微弱的），所以这算是一个“几乎免费的午餐”了。超参数选择方面，上面给出的中文结果，只迭代了1次，$k$的选择为3、$\tau$的选择为0.9，经过简单的调试，发现这基本上已经是比较优的参数组合了。

还有的读者可能想问前面说的“高置信度那部分结果更可靠”这个情况是否真的成立？至少在笔者的两个中文实验上它是明显成立的，比如IFLYTEK任务，筛选出来的高置信度集准确率为0.63+，而低置信度集的准确率只有0.22+；TNEWS任务类似，高置信度集准确率为0.58+，而低置信度集的准确率只有0.23+。

## 个人评价 #

最后再来综合地思考和评价一下CAN。

首先，一个很自然的疑问是为什么不直接将所有低置信度结果跟高置信度结果拼在一起进行修正，而是要逐个进行修正？笔者不知道原论文作者有没有对比过，但笔者确实实验过这个想法，结果是批量修正有时跟逐个修正持平，但有时也会下降。其实也可以理解，CAN本意应该是借助先验分布，结合高置信度结果来修正低置信度的，在这个过程中，如果掺入越多的低置信度结果，那么最终的偏差可能就越大，因此理论上逐个修正会比批量修正更为可靠。

说到原论文，读过CAN论文的读者，应该能发现本文介绍与CAN原论文大致有三点不同：

1、不确定性指标的计算方法不同。按照原论文的描述，它最终的不确定性指标计算方式应该是  
\begin{equation}-\frac{1}{\log m}\sum_{i=1}^k p_i\log p_i\end{equation}  
也就是说，它也是top-$k$个概率算熵的形式，但是它没有对这$k$个概率值重新归一化，并且它将其压缩到0～1之间的因子是$\log m$而不是$\log k$（因为它没有重新归一化，所以只有除$\log m$才能保证0～1之间）。经过笔者测试，原论文的这种方式计算出来的结果通常明显小于1，这不利于我们对阈值的感知和调试。

2、对CAN的介绍方式不同。原论文是纯粹数学化、矩阵化地陈述CAN的算法步骤，而且没有介绍算法的思想来源，这对理解CAN是相当不友好的。如果读者没有自行深入思考算法原理，是很难理解为什么这样的后处理手段就能提升分类效果的，而在彻底弄懂之后则会有一种故弄玄虚之感。

3、CAN的算法流程略有不同。原论文在迭代过程中还引入了参数$\alpha$，使得式$\eqref{eq:step-1}$变为  
\begin{equation}p^{(k)} \leftarrow [p^{(k)}]^{\alpha}\big/\bar{p}\times\tilde{p},\quad\bar{p}=\frac{1}{n+1}\left([p^{(j)}]^{\alpha} + \sum_{i=1}^n [p^{(i)}]^{\alpha}\right)\end{equation}  
也就是对每个结果进行$\alpha$次方后再迭代。当然，原论文也没有对此进行解释，而在笔者看来，该参数纯粹是为了调参而引入的（参数多了，总能把效果调到有所提升），没有太多实际意义。而且笔者自己在实验中发现，$\alpha=1$基本已经是最优选择了，精调$\alpha$也很难获得是实质收益。

## 文章小结 #

本文介绍了一种名为CAN的简单后处理技巧，它借助先验分布来将预测结果重新归一化，几乎没有增加多少计算成本就能提高分类性能。经过笔者的实验，CAN确实能给分类效果带来一定提升，并且通常来说类别数越多，效果越明显。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8728>_

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

苏剑林. (Oct. 22, 2021). 《CAN：借助先验分布提升分类性能的简单后处理技巧 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8728>

@online{kexuefm-8728,  
title={CAN：借助先验分布提升分类性能的简单后处理技巧},  
author={苏剑林},  
year={2021},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/8728}},  
} 


---

## 公式推导与注释

### 一、贝叶斯框架下的分类问题

#### 1.1 基本贝叶斯公式

**标准分类目标**：给定输入 $x$，我们希望预测其类别 $y \in \{1, 2, \ldots, m\}$。理想的分类器应该输出后验概率：

\begin{equation}
p(y|x) = \frac{p(x|y)p(y)}{p(x)} = \frac{p(x|y)p(y)}{\sum_{j=1}^m p(x|y=j)p(y=j)} \tag{1}
\end{equation}

其中：
- $p(y)$ 是类别的先验分布
- $p(x|y)$ 是类条件概率密度
- $p(x)$ 是边缘概率密度

**注释**：这个公式是贝叶斯定理的直接应用，它告诉我们如何从似然 $p(x|y)$ 和先验 $p(y)$ 计算后验 $p(y|x)$。

#### 1.2 神经网络分类器的输出

**实际情况**：深度学习分类器通常不直接建模 $p(x|y)$，而是通过softmax输出一个概率分布：

\begin{equation}
\hat{p}(y|x) = \text{softmax}(f_\theta(x))_y = \frac{e^{f_\theta(x)_y}}{\sum_{j=1}^m e^{f_\theta(x)_j}} \tag{2}
\end{equation}

其中 $f_\theta(x) \in \mathbb{R}^m$ 是神经网络的logits输出。

**隐含假设**：标准的交叉熵训练实际上隐含地假设：
1. 训练集的类别分布 $\tilde{p}(y)$ 等于真实的先验分布 $p(y)$
2. 模型完美地学习了后验概率

**注释**：但在实际中，这两个假设往往都不成立。训练集可能是平衡采样的（人为均匀），而真实世界的类别分布可能高度不平衡。

#### 1.3 先验不匹配的影响

**训练集先验**：设训练时的类别分布为 $\tilde{p}(y)$。

**测试集先验**：真实世界的类别分布为 $p(y)$。

**模型学到的后验**：如果模型完美拟合训练分布，那么：

\begin{equation}
\hat{p}(y|x) \approx \frac{p(x|y)\tilde{p}(y)}{\sum_{j=1}^m p(x|y=j)\tilde{p}(y=j)} \tag{3}
\end{equation}

**真实后验**：而我们真正需要的是：

\begin{equation}
p(y|x) = \frac{p(x|y)p(y)}{\sum_{j=1}^m p(x|y=j)p(y=j)} \tag{4}
\end{equation}

**修正公式**：通过比较式(3)和式(4)，我们可以推导出修正关系：

\begin{equation}
p(y|x) = \frac{\hat{p}(y|x) \cdot \frac{p(y)}{\tilde{p}(y)}}{\sum_{j=1}^m \hat{p}(y=j|x) \cdot \frac{p(y=j)}{\tilde{p}(y=j)}} \tag{5}
\end{equation}

**注释**：这个公式是后验校准的理论基础。它告诉我们，如果知道训练先验和测试先验，可以直接修正模型输出。

### 二、不确定性度量与熵

#### 2.1 熵的定义与性质

**Shannon熵**：对于离散概率分布 $p = (p_1, \ldots, p_m)$，Shannon熵定义为：

\begin{equation}
H(p) = -\sum_{i=1}^m p_i \log p_i \tag{6}
\end{equation}

**基本性质**：
1. **非负性**：$H(p) \geq 0$，当且仅当存在某个 $i$ 使得 $p_i = 1$ 时等号成立
2. **最大性**：在约束 $\sum_i p_i = 1$ 下，均匀分布 $p_i = 1/m$ 使得熵达到最大值 $H_{\max} = \log m$
3. **凹性**：$H$ 是关于 $p$ 的凹函数

**注释**：熵度量了分布的"不确定性"或"混乱度"。熵越大，分布越"平坦"，预测越不确定。

#### 2.2 熵在分类中的问题

**示例对比**：考虑三分类问题，两个预测结果：
- $p^{(a)} = [0.5, 0.25, 0.25]$：最大概率是0.5，其余两个类别各0.25
- $p^{(b)} = [0.5, 0.5, 0]$：最大概率是0.5（并列），第三个类别概率为0

**熵的计算**：

\begin{align}
H(p^{(a)}) &= -0.5\log 0.5 - 0.25\log 0.25 - 0.25\log 0.25 \tag{7}\\
&= 0.5\cdot 0.693 + 2\cdot 0.25\cdot 1.386 = 1.040 \tag{8}\\
H(p^{(b)}) &= -0.5\log 0.5 - 0.5\log 0.5 - 0\log 0 = 0.693 \tag{9}
\end{align}

**直觉矛盾**：按照熵的定义，$H(p^{(a)}) > H(p^{(b)})$，似乎 $p^{(a)}$ 更不确定。但从分类任务的角度看：
- $p^{(a)}$ 有明确的最大值（0.5），预测类别1
- $p^{(b)}$ 有两个并列最大值，无法明确选择类别1还是类别2

因此 $p^{(b)}$ 实际上更不确定！

**注释**：标准熵对所有类别一视同仁，但在分类任务中，我们更关心"谁是第一名"以及"第一名和第二名差距多大"。

#### 2.3 Top-K熵

**定义**：只考虑概率最大的 $k$ 个类别，并重新归一化后计算熵：

设 $p_{(1)} \geq p_{(2)} \geq \cdots \geq p_{(m)}$ 是概率值的降序排列，定义：

\begin{equation}
\tilde{p}_i = \frac{p_{(i)}}{\sum_{j=1}^k p_{(j)}}, \quad i = 1, \ldots, k \tag{10}
\end{equation}

\begin{equation}
H_{\text{top-}k}(p) = -\sum_{i=1}^k \tilde{p}_i \log \tilde{p}_i \tag{11}
\end{equation}

**归一化到[0,1]**：

\begin{equation}
U_{\text{top-}k}(p) = \frac{H_{\text{top-}k}(p)}{\log k} \tag{12}
\end{equation}

**重新计算示例**：以 $k=2$ 为例：

\begin{align}
p^{(a)}: &\quad \tilde{p} = [\frac{0.5}{0.75}, \frac{0.25}{0.75}] = [0.667, 0.333] \tag{13}\\
&\quad U_{\text{top-2}}(p^{(a)}) = \frac{-0.667\log 0.667 - 0.333\log 0.333}{\log 2} = 0.637 \tag{14}\\
p^{(b)}: &\quad \tilde{p} = [\frac{0.5}{1.0}, \frac{0.5}{1.0}] = [0.5, 0.5] \tag{15}\\
&\quad U_{\text{top-2}}(p^{(b)}) = \frac{-0.5\log 0.5 - 0.5\log 0.5}{\log 2} = 1.0 \tag{16}
\end{align}

**注释**：现在 $U_{\text{top-2}}(p^{(b)}) > U_{\text{top-2}}(p^{(a)})$，与我们的直觉一致！$p^{(b)}$ 确实更不确定。

### 三、CAN算法的数学原理

#### 3.1 先验分布的估计

**从训练集估计**：设训练集 $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$，先验分布估计为：

\begin{equation}
\tilde{p}(y = j) = \frac{\sum_{i=1}^N \mathbb{1}\{y_i = j\}}{N} = \frac{N_j}{N} \tag{17}
\end{equation}

其中 $N_j$ 是类别 $j$ 在训练集中出现的次数。

**拉普拉斯平滑**（可选）：为避免零概率：

\begin{equation}
\tilde{p}(y = j) = \frac{N_j + \alpha}{N + m\alpha} \tag{18}
\end{equation}

通常取 $\alpha = 1$。

**注释**：如果训练集是人为平衡的，则 $\tilde{p}(y) \approx [1/m, \ldots, 1/m]$ 均匀分布。

#### 3.2 行间标准化（Row-wise Normalization）

**动机**：我们希望让一组预测结果的平均分布等于先验分布 $\tilde{p}(y)$。

**标准化步骤**：给定一组预测 $\{p^{(1)}, \ldots, p^{(n)}, p^{(j)}\}$，计算平均：

\begin{equation}
\bar{p} = \frac{1}{n+1}\left(p^{(j)} + \sum_{i=1}^n p^{(i)}\right) \tag{19}
\end{equation}

然后对每个预测进行element-wise的缩放：

\begin{equation}
p^{(k)} \leftarrow p^{(k)} \odot \frac{\tilde{p}}{\bar{p}} \tag{20}
\end{equation}

其中 $\odot$ 表示element-wise乘法，$\frac{\tilde{p}}{\bar{p}}$ 也是element-wise除法。

**验证均值**：缩放后，新的平均值为：

\begin{align}
\bar{p}_{\text{new}} &= \frac{1}{n+1}\sum_{k} p^{(k)} \odot \frac{\tilde{p}}{\bar{p}} \tag{21}\\
&= \frac{1}{n+1}\left(\sum_{k} p^{(k)}\right) \odot \frac{\tilde{p}}{\bar{p}} = \bar{p} \odot \frac{\tilde{p}}{\bar{p}} = \tilde{p} \tag{22}
\end{align}

**注释**：这个操作确保了新的平均分布等于先验分布，这正是我们想要的！

#### 3.3 行内标准化（Column-wise Normalization）

**问题**：经过行间标准化后，每个 $p^{(k)}$ 不再满足归一化条件 $\sum_i p^{(k)}_i = 1$。

**修正**：对每个预测重新归一化：

\begin{equation}
p^{(k)} \leftarrow \frac{p^{(k)}}{\sum_{i=1}^m p^{(k)}_i} \tag{23}
\end{equation}

**注释**：这个操作恢复了概率分布的归一化性质，但也会破坏式(22)的性质。因此理论上需要交替迭代。

#### 3.4 交替标准化的收敛性

**迭代算法**：

\begin{equation}
\begin{aligned}
&\text{For } t = 1, 2, \ldots, T:\\
&\quad \text{行间标准化: } p^{(k)} \leftarrow p^{(k)} \odot \frac{\tilde{p}}{\bar{p}}\\
&\quad \text{行内标准化: } p^{(k)} \leftarrow \frac{p^{(k)}}{\sum_i p^{(k)}_i}
\end{aligned} \tag{24}
\end{equation}

**收敛性质**：这个交替投影算法类似于Sinkhorn算法，在一定条件下可以证明收敛到一个满足：
1. 每行归一化：$\sum_i p^{(k)}_i = 1$
2. 列平均等于先验：$\frac{1}{n+1}\sum_k p^{(k)}_i = \tilde{p}_i$

的不动点。

**实践中**：论文发现 $T=1$ （只迭代一次）效果最好，过多迭代反而会降低性能。

**注释**：这可能是因为模型输出本身存在误差，过度强制满足先验分布反而会引入额外偏差。

### 四、CAN算法的信息论解释

#### 4.1 互信息的角度

**互信息**：预测 $p(y|x)$ 和先验 $p(y)$ 之间的互信息定义为：

\begin{equation}
I(Y; X) = \sum_{x,y} p(x,y)\log\frac{p(y|x)}{p(y)} = H(Y) - H(Y|X) \tag{25}
\end{equation}

其中 $H(Y|X) = \mathbb{E}_{x}[H(p(y|x))]$ 是条件熵。

**CAN的作用**：通过修正后验使其平均分布等于先验，相当于在保持条件熵 $H(Y|X)$ 基本不变的情况下，调整边缘熵 $H(Y)$ 使其与真实先验匹配。

**注释**：这确保了模型不会因为训练集的类别不平衡而系统性地偏向某些类别。

#### 4.2 KL散度的视角

**目标**：最小化修正后的预测分布与真实后验之间的KL散度：

\begin{equation}
\min_{p'} \mathbb{E}_{x}\left[D_{\text{KL}}(p_{\text{true}}(y|x) \| p'(y|x))\right] \tag{26}
\end{equation}

在约束条件下：

\begin{equation}
\mathbb{E}_{x}[p'(y|x)] = \tilde{p}(y) \tag{27}
\end{equation}

**拉格朗日形式**：

\begin{equation}
\mathcal{L} = \mathbb{E}_{x}\left[D_{\text{KL}}(p_{\text{true}}(y|x) \| p'(y|x))\right] + \lambda \left\|\mathbb{E}_{x}[p'(y|x)] - \tilde{p}(y)\right\|^2 \tag{28}
\end{equation}

**注释**：CAN可以看作是这个优化问题的一个近似解法，通过启发式的标准化步骤来逼近最优解。

### 五、高置信度与低置信度的分离

#### 5.1 置信度阈值的选择

**阈值选择**：设定阈值 $\tau \in [0, 1]$，将样本分为两组：
- 高置信度：$U_{\text{top-}k}(p^{(i)}) < \tau$
- 低置信度：$U_{\text{top-}k}(p^{(i)}) \geq \tau$

**经验准则**：通常 $\tau \in [0.8, 0.95]$ 之间。$\tau$ 越小，高置信度集越小，越"精英化"。

**统计验证**：计算两组的准确率：

\begin{equation}
\text{Acc}_{\text{high}} = \frac{\sum_{i: U(p^{(i)}) < \tau} \mathbb{1}\{\arg\max p^{(i)} = y_i\}}{\sum_{i} \mathbb{1}\{U(p^{(i)}) < \tau\}} \tag{29}
\end{equation}

\begin{equation}
\text{Acc}_{\text{low}} = \frac{\sum_{i: U(p^{(i)}) \geq \tau} \mathbb{1}\{\arg\max p^{(i)} = y_i\}}{\sum_{i} \mathbb{1}\{U(p^{(i)}) \geq \tau\}} \tag{30}
\end{equation}

**期望结果**：应该有 $\text{Acc}_{\text{high}} \gg \text{Acc}_{\text{low}}$。

**注释**：如果这个不等式不成立，说明不确定性度量可能不合适，或者模型校准很差。

#### 5.2 逐个修正的原理

**为什么不批量修正**：如果将所有低置信度样本一起修正：

\begin{equation}
\{p^{(1)}, \ldots, p^{(n)}, p^{(n+1)}, \ldots, p^{(N)}\} \tag{31}
\end{equation}

其中前 $n$ 个是高置信度，后 $N-n$ 个是低置信度。

**问题**：批量修正时，大量的低置信度样本会稀释高置信度样本的影响，导致平均分布 $\bar{p}$ 不准确。

**逐个修正的优势**：对于每个低置信度样本 $p^{(j)}$，只与高置信度样本组合：

\begin{equation}
\{p^{(1)}, \ldots, p^{(n)}, p^{(j)}\} \tag{32}
\end{equation}

这样高置信度样本的"信号强度"更大，能更好地引导修正。

**注释**：这是一个bias-variance权衡。批量修正variance小但bias可能大；逐个修正variance稍大但bias更小。

### 六、超参数 $\alpha$ 的作用

#### 6.1 $\alpha$ 的定义

原论文引入了参数 $\alpha$，将行间标准化修改为：

\begin{equation}
p^{(k)} \leftarrow [p^{(k)}]^\alpha \odot \frac{\tilde{p}}{[\bar{p}]^\alpha} \tag{33}
\end{equation}

其中 $[\cdot]^\alpha$ 表示element-wise的幂运算。

#### 6.2 $\alpha$ 的效果分析

**$\alpha > 1$**：增强差异，使得大概率更大，小概率更小
- $p = [0.7, 0.2, 0.1]$，$\alpha = 2$ 时：$p^\alpha = [0.49, 0.04, 0.01]$（归一化前）

**$\alpha < 1$**：减弱差异，使得分布更平坦
- $p = [0.7, 0.2, 0.1]$，$\alpha = 0.5$ 时：$p^\alpha = [0.837, 0.447, 0.316]$（归一化前）

**$\alpha = 1$**：不改变，即本文前面推导的标准CAN

**温度的关系**：这个操作类似于softmax中的温度参数 $T$：

\begin{equation}
\text{softmax}(z/T)_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}} \tag{34}
\end{equation}

其中 $T > 1$ 使分布平滑，$T < 1$ 使分布尖锐。

**注释**：原论文报告 $\alpha = 1$ 通常是最优选择，额外调参收益很小，这支持了我们的简化版本（不考虑 $\alpha$）。

### 七、CAN的概率校准理论

#### 7.1 模型校准的定义

**完美校准**：一个模型是完美校准的，如果对于所有 $p \in [0, 1]$：

\begin{equation}
\mathbb{P}(Y = y | \hat{p}(y|X) = p) = p \tag{35}
\end{equation}

即预测概率等于实际条件概率。

**期望校准误差（ECE）**：

\begin{equation}
\text{ECE} = \sum_{k=1}^K \frac{|B_k|}{N} \left| \text{Acc}(B_k) - \text{Conf}(B_k) \right| \tag{36}
\end{equation}

其中：
- $B_k$ 是预测概率在区间 $[(k-1)/K, k/K)$ 内的样本
- $\text{Acc}(B_k)$ 是该bin内的准确率
- $\text{Conf}(B_k)$ 是该bin内的平均置信度

**注释**：ECE越小，模型校准越好。完美校准时 ECE = 0。

#### 7.2 CAN对校准的影响

**后验修正的效果**：CAN通过式(5)的后验修正，理论上应该改善校准。

**实验验证**：通过绘制可靠性图（Reliability Diagram）：
- x轴：预测置信度（binned）
- y轴：实际准确率

理想情况下应该是 $y = x$ 直线。

**CAN前后对比**：
- CAN前：可能出现系统性偏差，如高估或低估
- CAN后：应该更接近 $y = x$ 直线

**注释**：CAN不仅改善准确率，还能改善校准，这在需要置信度估计的应用（如主动学习、拒绝选项）中很重要。

### 八、理论上界与误差分析

#### 8.1 修正误差的来源

**误差分解**：CAN的总误差可以分解为：

\begin{equation}
\epsilon_{\text{total}} = \epsilon_{\text{model}} + \epsilon_{\text{prior}} + \epsilon_{\text{CAN}} \tag{37}
\end{equation}

其中：
- $\epsilon_{\text{model}}$：模型本身的误差（未完美学习 $p(y|x)$）
- $\epsilon_{\text{prior}}$：先验估计误差（$\tilde{p}(y) \neq p(y)$）
- $\epsilon_{\text{CAN}}$：CAN算法的近似误差

**注释**：只有当前两项足够小时，CAN才能发挥作用。如果模型本身很差，CAN也无济于事。

#### 8.2 先验估计的鲁棒性

**敏感性分析**：当先验估计有误差时：

\begin{equation}
\tilde{p}'(y) = \tilde{p}(y) + \delta(y), \quad \sum_y \delta(y) = 0 \tag{38}
\end{equation}

修正后的后验变为：

\begin{equation}
p'(y|x) \approx p(y|x) + \mathcal{O}(\|\delta\|) \tag{39}
\end{equation}

**注释**：只要先验估计大致正确（$\|\delta\|$ 小），CAN就能带来改善。不需要精确知道真实先验。

### 九、多次迭代的理论分析

#### 9.1 单次 vs 多次迭代

**单次迭代** ($T=1$)：

\begin{equation}
p^{(j)}_1 = \frac{p^{(j)}_0 \odot \frac{\tilde{p}}{\bar{p}_0}}{\sum_i \left(p^{(j)}_{0,i} \cdot \frac{\tilde{p}_i}{\bar{p}_{0,i}}\right)} \tag{40}
\end{equation}

**两次迭代** ($T=2$)：

先得到 $p^{(j)}_1$，然后计算新的 $\bar{p}_1$，再次修正：

\begin{equation}
p^{(j)}_2 = \frac{p^{(j)}_1 \odot \frac{\tilde{p}}{\bar{p}_1}}{\sum_i \left(p^{(j)}_{1,i} \cdot \frac{\tilde{p}_i}{\bar{p}_{1,i}}\right)} \tag{41}
\end{equation}

#### 9.2 过度迭代的风险

**理论分析**：每次迭代都会使得平均分布更接近 $\tilde{p}$，但同时也会：
1. 增加对先验估计误差的敏感性
2. 可能过度平滑，损失模型的原始信息

**实验观察**：
- $T=1$：通常最优
- $T=2,3$：效果与 $T=1$ 接近或略差
- $T\geq 5$：效果开始明显下降

**注释**：这是过拟合先验的表现。先验只是一个粗略的统计信息，不应过度依赖。

### 十、与其他校准方法的比较

#### 10.1 温度缩放（Temperature Scaling）

**方法**：在softmax中引入温度参数 $T$：

\begin{equation}
p_T(y|x) = \frac{e^{z_y/T}}{\sum_j e^{z_j/T}} \tag{42}
\end{equation}

在验证集上优化 $T$ 以最小化负对数似然。

**优点**：简单，只有一个参数
**缺点**：所有样本使用同一个 $T$，无法个性化调整

**与CAN的区别**：CAN针对低置信度样本单独处理，更灵活。

#### 10.2 Platt Scaling

**方法**：学习一个逻辑回归模型：

\begin{equation}
p_{\text{calib}}(y|x) = \sigma(az_y + b) \tag{43}
\end{equation}

其中 $a, b$ 在验证集上学习。

**优点**：理论上可以修正任意单调误差
**缺点**：需要额外的训练；多分类需要one-vs-rest

**与CAN的区别**：Platt Scaling需要训练，CAN是纯后处理，无需额外训练。

#### 10.3 Isotonic Regression

**方法**：学习一个单调的非参数映射 $f: [0,1] \to [0,1]$：

\begin{equation}
p_{\text{calib}}(y|x) = f(\hat{p}(y|x)) \tag{44}
\end{equation}

使得 $f$ 单调且最小化校准误差。

**优点**：非常灵活，可以拟合任意单调关系
**缺点**：容易过拟合；需要大量验证数据

**与CAN的区别**：Isotonic Regression是单变量校准，CAN考虑了类别之间的依赖（通过先验分布）。

### 十一、数值实验与案例分析

#### 11.1 类别不平衡的案例

**设定**：
- 真实先验：$p(y) = [0.1, 0.3, 0.6]$（高度不平衡）
- 训练先验：$\tilde{p}(y) = [1/3, 1/3, 1/3]$（人为平衡）

**模型输出**（未校准）：

\begin{equation}
\hat{p}(y|x) = [0.4, 0.35, 0.25] \tag{45}
\end{equation}

预测类别1（概率0.4）。

**CAN修正**：

\begin{align}
p_{\text{CAN}}(y|x) &\propto \hat{p}(y|x) \odot \frac{p(y)}{\tilde{p}(y)} \tag{46}\\
&\propto [0.4, 0.35, 0.25] \odot \frac{[0.1, 0.3, 0.6]}{[1/3, 1/3, 1/3]} \tag{47}\\
&\propto [0.4\times 0.3, 0.35\times 0.9, 0.25\times 1.8] \tag{48}\\
&= [0.12, 0.315, 0.45] \tag{49}
\end{align}

归一化后：$p_{\text{CAN}}(y|x) = [0.136, 0.358, 0.506]$

**预测变化**：类别从1变为3！这反映了真实世界中类别3更常见的事实。

**注释**：这个例子展示了CAN如何利用先验信息修正预测。

#### 11.2 高置信度样本的稳定性

**高置信度样本**：

\begin{equation}
p^{(h)} = [0.95, 0.03, 0.02], \quad U_{\text{top-2}}(p^{(h)}) = 0.23 < 0.9 \tag{50}
\end{equation}

**CAN修正前后**：由于 $p^{(h)}$ 是高置信度样本，不会被修正，保持不变。

**低置信度样本**：

\begin{equation}
p^{(l)} = [0.45, 0.35, 0.20], \quad U_{\text{top-2}}(p^{(l)}) = 0.99 > 0.9 \tag{51}
\end{equation}

会被修正。

**注释**：CAN只对不确定的预测进行调整，保留了高置信度预测的原始信息。

### 十二、实现细节与优化

#### 12.1 高效实现

**向量化计算**：使用NumPy/PyTorch的向量化操作，避免显式循环：

```python
# 高置信度样本
p_conf = predictions[uncertainty < tau]  # (n, m)
# 对于每个低置信度样本
for p_low in predictions[uncertainty >= tau]:
    # 拼接
    p_all = np.concatenate([p_conf, p_low[None]], axis=0)  # (n+1, m)
    # 行间标准化
    p_mean = p_all.mean(axis=0)  # (m,)
    p_all = p_all * (prior / p_mean)  # broadcasting
    # 行内标准化
    p_all = p_all / p_all.sum(axis=1, keepdims=True)
    # 提取修正后的低置信度样本
    p_low_new = p_all[-1]
```

**注释**：这个实现的复杂度是 $O(N_{\text{low}} \times N_{\text{high}} \times m)$，对于大规模问题可能较慢。

#### 12.2 近似加速

**随机采样高置信度样本**：如果 $N_{\text{high}}$ 很大，可以随机采样一个子集：

\begin{equation}
\tilde{p}_{\text{conf}} = \{p^{(i)}\}_{i \in \mathcal{S}} \tag{52}
\end{equation}

其中 $\mathcal{S}$ 是大小为 $n_{\text{sample}}$ 的随机子集（如 $n_{\text{sample}} = 1000$）。

**注释**：实验表明，只要 $n_{\text{sample}}$ 足够大（$\geq 500$），近似效果与使用全部高置信度样本几乎相同。

### 十三、总结与展望

本节详细推导了CAN算法的数学原理，核心内容包括：

1. **贝叶斯框架**：从贝叶斯公式出发，理解先验不匹配如何影响后验
2. **不确定性度量**：改进的top-k熵更适合分类任务
3. **交替标准化**：行间-行内标准化的迭代收敛
4. **逐个修正**：避免低置信度样本的稀释效应
5. **校准理论**：CAN改善模型的概率校准性
6. **数值分析**：具体案例展示CAN的修正效果

CAN作为一个简单的后处理方法，在几乎不增加计算成本的情况下，有效利用先验信息改善了分类性能和校准性。

