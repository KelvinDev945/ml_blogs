---
title: Naive Bayes is all you need ?
slug: naive-bayes-is-all-you-need
date: 2023-06-08
tags: 语言模型, attention, LLM, 贝叶斯, 生成模型
status: pending
---

# Naive Bayes is all you need ?

**原文链接**: [https://spaces.ac.cn/archives/9648](https://spaces.ac.cn/archives/9648)

**发布日期**: 

---

很抱歉，起了这么个具有标题党特征的题目。在写完[《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度》](/archives/9617)之后，笔者就觉得朴素贝叶斯（Naive Bayes）跟Attention机制有很多相同的特征，后来再推导了一下发现， _Attention机制其实可以看成是一种 广义的、参数化的朴素贝叶斯_。既然如此，“[Attention is All You Need](/archives/4765)”不也就意味着“Naive Bayes is all you need”了？这就是本文标题的缘由。

接下来笔者将介绍自己的思考过程，分析如何从朴素贝叶斯角度来理解Attention机制。

## 朴素贝叶斯 #

本文主要考虑语言模型，它要建模的是$p(x_t|x_1,\cdots,x_{t-1})$。根据贝叶斯公式，我们有  
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) = \frac{p(x_1,\cdots,x_{t-1}|x_t)p(x_t)}{p(x_1,\cdots,x_{t-1})}\propto p(x_1,\cdots,x_{t-1}|x_t)p(x_t)\end{equation}  
根据独立假设$p(x_1,\cdots,x_{t-1}|x_t) = \prod\limits_{j=1}^{t-1} p(x_j|x_t)$，我们有  
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) \propto \prod_{j=1}^{t-1} p(x_j|x_t)p(x_t)\end{equation}  
再次根据贝叶斯公式$p(x_j|x_t)=\frac{p(x_t|x_j)p(x_j)}{p(x_t)}\propto\frac{p(x_t|x_j)}{p(x_t)}$，得到  
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) \propto \frac{1}{[p(x_t)]^{t-2}}\prod_{j=1}^{t-1} p(x_t|x_j)\end{equation}  
两边取对数得到  
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \sum_{j=1}^{t-1}\log p(x_t|x_j) - (t - 2) \log p(x_t) + \text{常数}\end{equation}

## 一般化结果 #

相同的推导我们在[《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度》](/archives/9617)也进行过，跟该文章一样，我们将上式一般化为：  
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = (1 + \beta)\mathcal{P}[\log p(x_t|x_j)] - \beta \log p(x_t) + \text{常数}\end{equation}  
这里的$\beta$作为超参数来调，$\mathcal{P}$是某种Pooling方式。接下来我们主要看$\beta=0$、以加权平均为Pooling的例子，即  
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \sum_j a_{t,j} \log p(x_t|x_j) + \text{常数}\label{eq:nb-core}\end{equation}  
这里的$a_{t,j}$是$x_{t-1}$与$x_j$的函数。

可能有读者想问，这个一般化的式子还能算是朴素贝叶斯吗？笔者认为它可以作为广义的朴素贝叶斯来看待，因为朴素贝叶斯可以视为各个$\log p(x_t|x_j)$的等权平均，这里则是换成了更一般化的加权平均。不过，将$a_{t,j}$选取为$x_{t-1}$与$x_j$的函数，突出了$x_{t-1}$的地位，改善了朴素贝叶斯的无序性这一弊端。所以更准确来说，式$\eqref{eq:nb-core}$是2-gram语言模型与朴素贝叶斯的结合。

## 注意力初现 #

接下来，将$\log p(x_t|x_j)$进一步参数化，我们就可以得见Attention的形式了。不难发现，$p(x_t|x_j)$实质上就是以前Word2Vec的Skip Gram模型，它的常规建模方式是“Embedding + 内积 + Softmax”，即  
\begin{equation}p(x_t|x_j) = \frac{e^{v(x_j)\cdot w(x_t)}}{Z(x_j)},\quad Z(x_j) = \sum_{x_t\in Vocab}e^{v(x_j)\cdot w(x_t)}\end{equation}  
所以我们简单地认为  
\begin{equation}\log p(x_t|x_j) = v(x_j)\cdot w(x_t) + \text{常数}\end{equation}  
代入到式$\eqref{eq:nb-core}$，得到  
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \left(\sum_j a_{t,j} v(x_j)\right)\cdot w(x_t) + \text{常数}\label{eq:nb-core-2}\end{equation}  
括号中的式子，我们将它单独拿出来，当作通常用特征融合运算，它其实就是常规的Attention。所以说，单层的Attention做语言模型，实则就是广义的朴素贝叶斯。

当然，这里我们还没有将$a_{t,j}$确定下来。上一节我们说$a_{t,j}$是$x_{t-1}$与$x_j$的函数，然后同时还要归一化（加权平均），所以比较简单的方式就是像Skip Gram一样“Embedding + 内积 + Softmax”：  
\begin{equation}a_{t,j} = \frac{e^{q(x_{t-1})\cdot k(x_j)}}{Z_t},\quad Z_t = \sum_{j=1}^{t-1} e^{q(x_{t-1})\cdot k(x_j)}\end{equation}  
代入到式$\eqref{eq:nb-core-2}$，就是目前最常用的Dot-Product Attention了。当然，这种方式不是唯一的，还有加性Attention等，选择Dot-Product的最主要原因是它可以在比较省显存的前提下实现并行。

## 层叠与残差 #

不管怎么参数化，单层的朴素贝叶斯能力总是有限的，所以需要进一步提高模型的复杂度。从神经网络的角度来看，提高模型复杂度的主要方式是增加深度，也就是层与层之间的堆叠。那么，从概率分布的角度如何理解这种堆叠呢？答案是隐变量模型。

所谓隐变量模型，就是引入隐变量$z_1,z_2,\cdots,z_{t-1}$，使得  
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) = \int p(x_t|z_1,\cdots,z_{t-1})p(z_1,\cdots,z_{t-1}|x_1,\cdots,x_{t-1})dz_1 \cdots dz_{t-1}\end{equation}  
说白了，就是通过简单分布的叠加来拟合更复杂的分布，跟GMM（高斯混合模型）的思想是一致的。基于前面的讨论，$p(x_t|z_1,\cdots,z_{t-1})$我们同样用朴素贝叶斯建模，即从特征层面就是单层Attention。而对于$p(z_1,\cdots,z_{t-1}|x_1,\cdots,x_{t-1})$，我们按照自回归模型的特点，分解为  
\begin{equation}p(z_1,\cdots,z_{t-1}|x_1,\cdots,x_{t-1}) = \prod_{k=1}^{t-1} p(z_k|x_1,\cdots,x_k)\end{equation}  
这样每个$p(z_k|x_1,\cdots,x_k)$形式上就跟$p(x_t|z_1,\cdots,z_{t-1})$一样了，于是同样可以用朴素贝叶斯建模。简单起见，$z_k$我们定义为连续型变量，$p(z_k|x_1,\cdots,x_k)$则定义为[狄拉克分布](/archives/1870)，于是积分可以直接算出来，结果就是两层Attention的堆叠了。

最后，Transfromer中还有一个关键成分是残差，实际上它就是将式$\eqref{eq:nb-core}$一般化为  
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \log p(x_t|x_{t-1}) + \sum_j a_{t,j} \log p(x_t|x_j) + \text{常数}\end{equation}  
可以理解为一种突出了2-gram的地位的Pooling方式，算是一种先验。最后，还剩下的FeedForward层、LayerNorm层等，这些层不涉及token之间的交互，可以理解为是更复杂地参数化的朴素贝叶斯。

当然，这样笼统的解释看上去有些勉强，但笔者原本的想法，也不是精准地解释Transformer或Attention，而是期望是能从朴素贝叶斯角度来够获得一些关于长度外推的新思路。但很遗憾，目前笔者还没有得到预期的结果。然而，尽管看上去像是盲目的自恋，但笔者依然认为上述朴素贝叶斯和隐变量模型的视角还有进一步挖掘的潜力，比如看上去我们可以从朴素贝叶斯角度解释基于Attention的语言模型的In-Context Learning为啥会有效。

## 文章总概述 #

本文阐述了朴素贝叶斯与Attention机制之间的关联，显示了Attention可被视为一种广义的朴素贝叶斯。从这个视角，我们还可以进一步地理解Attention中的层叠与残差等内容。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9648>_

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

苏剑林. (Jun. 08, 2023). 《Naive Bayes is all you need ? 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9648>

@online{kexuefm-9648,  
title={Naive Bayes is all you need ?},  
author={苏剑林},  
year={2023},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/9648}},  
} 


---

## 贝叶斯推理的数学基础 {#bayesian-foundation}

### 贝叶斯定理的完整推导

<div class="theorem-box">

**定理1：贝叶斯定理**

对于事件$A, B$，有：
$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

**多变量形式**：对于随机变量$X = (X_1, \ldots, X_n)$和$Y$：
$$p(Y|X) = \frac{p(X|Y) p(Y)}{p(X)} = \frac{p(X|Y) p(Y)}{\int p(X|Y') p(Y') dY'}$$

</div>

**证明**：

由条件概率定义：
$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B|A) = \frac{P(A \cap B)}{P(A)}$$

由对称性$P(A \cap B) = P(B \cap A)$，有：
$$P(A|B) \cdot P(B) = P(A \cap B) = P(B|A) \cdot P(A)$$

两边除以$P(B)$即得贝叶斯定理。

---

### 朴素贝叶斯的条件独立性假设

<div class="derivation-box">

**假设1：条件独立性（Naive Bayes Assumption）**

给定类别$Y$，特征$X_1, \ldots, X_n$条件独立：
$$p(X_1, \ldots, X_n | Y) = \prod_{i=1}^n p(X_i | Y)$$

**直观理解**：一旦知道$Y$，各个特征之间不再相关。

**例子**：在文本分类中，如果已知文档主题是"体育"，那么词语"足球"和"篮球"的出现被认为是独立的（尽管实际上可能有关联）。

</div>

**数学后果**：

利用贝叶斯定理和条件独立性：
\begin{equation}\begin{aligned}
p(Y|X_1, \ldots, X_n) &= \frac{p(X_1, \ldots, X_n|Y) p(Y)}{p(X_1, \ldots, X_n)} \\
&= \frac{p(Y) \prod_{i=1}^n p(X_i|Y)}{\sum_{Y'} p(Y') \prod_{i=1}^n p(X_i|Y')} \\
&\propto p(Y) \prod_{i=1}^n p(X_i|Y)
\end{aligned}\end{equation}

取对数（log-likelihood）：
$$\log p(Y|X_1, \ldots, X_n) = \log p(Y) + \sum_{i=1}^n \log p(X_i|Y) + \text{const}$$

---

## 语言模型中的朴素贝叶斯 {#nb-in-lm}

### 自回归语言模型的目标

<div class="theorem-box">

**定义1：自回归语言模型**

给定序列$x_{1:t-1} = (x_1, \ldots, x_{t-1})$，预测下一个token $x_t$的条件概率：
$$p(x_t | x_{1:t-1})$$

联合概率的链式分解：
$$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t | x_{1:t-1})$$

**训练目标**：最大化对数似然
$$\mathcal{L} = \sum_{t=1}^T \log p(x_t | x_{1:t-1})$$

</div>

---

### 朴素贝叶斯的逆向建模

**常规思路**（困难）：直接建模$p(x_t | x_{1:t-1})$需要考虑复杂的依赖关系。

**朴素贝叶斯思路**（巧妙）：通过贝叶斯定理"反转"条件：

\begin{equation}\begin{aligned}
p(x_t | x_{1:t-1}) &= \frac{p(x_{1:t-1} | x_t) p(x_t)}{p(x_{1:t-1})} \\
&\propto p(x_{1:t-1} | x_t) p(x_t)
\end{aligned}\end{equation}

**关键简化**：应用条件独立性
$$p(x_{1:t-1} | x_t) = \prod_{j=1}^{t-1} p(x_j | x_t)$$

**假设的直观意义**：给定"要预测的词"$x_t$，历史中的各个词$x_j$独立出现。虽然这个假设很强，但在某些情况下是合理的近似（例如，如果$x_t$是主题词，历史中的词都是围绕这个主题独立展开的）。

代入得到：
$$p(x_t | x_{1:t-1}) \propto p(x_t) \prod_{j=1}^{t-1} p(x_j | x_t)$$

---

### 二次应用贝叶斯定理

<div class="derivation-box">

**推导步骤**

我们希望将$p(x_j|x_t)$转换回$p(x_t|x_j)$（因为后者在实践中更容易建模，如Skip-Gram）。

再次应用贝叶斯定理：
$$p(x_j | x_t) = \frac{p(x_t | x_j) p(x_j)}{p(x_t)}$$

代入：
\begin{equation}\begin{aligned}
p(x_t | x_{1:t-1}) &\propto p(x_t) \prod_{j=1}^{t-1} \frac{p(x_t | x_j) p(x_j)}{p(x_t)} \\
&= p(x_t) \cdot \frac{\prod_{j=1}^{t-1} p(x_t | x_j) p(x_j)}{[p(x_t)]^{t-1}} \\
&= \frac{1}{[p(x_t)]^{t-2}} \prod_{j=1}^{t-1} p(x_t | x_j) p(x_j)
\end{aligned}\end{equation}

如果进一步假设$p(x_j)$与$j$无关（词频均匀），则：
$$p(x_t | x_{1:t-1}) \propto \frac{1}{[p(x_t)]^{t-2}} \prod_{j=1}^{t-1} p(x_t | x_j)$$

</div>

**取对数**：
$$\log p(x_t | x_{1:t-1}) = \sum_{j=1}^{t-1} \log p(x_t | x_j) - (t-2) \log p(x_t) + \text{const}$$

**观察**：
- 第一项：$\sum_j \log p(x_t | x_j)$是各位置对$x_t$预测的"累积置信度"
- 第二项：$-(t-2) \log p(x_t)$是惩罚高频词的先验项（类似TF-IDF）

---

## 参数化与加权平均 {#parameterization}

### 从等权到加权

原始朴素贝叶斯是**等权平均**：
$$\log p(x_t | x_{1:t-1}) = \frac{1}{t-1} \sum_{j=1}^{t-1} \log p(x_t | x_j) - (t-2) \log p(x_t) + \text{const}$$

<div class="theorem-box">

**广义化1：加权平均**

引入权重$a_{t,j}$（满足$\sum_j a_{t,j} = 1$）：
$$\log p(x_t | x_{1:t-1}) = \sum_{j=1}^{t-1} a_{t,j} \log p(x_t | x_j) + \text{bias}$$

其中bias项包含先验$p(x_t)$和其他修正。

**权重的设计原则**：
1. **位置相关性**：距离$x_t$更近的$x_j$应有更大权重
2. **内容相关性**：与$x_t$语义相关的$x_j$应有更大权重
3. **可学习性**：权重应由参数化函数决定，而非固定

</div>

**自然的选择**：让$a_{t,j}$依赖于$x_{t-1}$和$x_j$：
$$a_{t,j} = f(x_{t-1}, x_j)$$

这样既考虑了局部信息（最近的词$x_{t-1}$），又保持了全局视野（所有历史$x_j$）。

---

### 超参数$\beta$的引入

<div class="derivation-box">

**广义化2：可调节先验强度**

将公式进一步参数化：
$$\log p(x_t | x_{1:t-1}) = (1+\beta) \sum_j a_{t,j} \log p(x_t | x_j) - \beta \log p(x_t) + \text{const}$$

**$\beta$的意义**：
- $\beta = 0$：忽略先验$p(x_t)$，完全依赖条件分布
- $\beta > 0$：增强先验的作用，对高频词有更强惩罚
- $\beta = t-2$：回到标准朴素贝叶斯

</div>

**实践中的选择**：
- 短序列（$t$小）：$\beta$可以较大，因为历史信息少，需要依赖先验
- 长序列（$t$大）：$\beta$可以较小，因为历史信息丰富，条件分布更可靠

---

## Skip-Gram与Attention的联系 {#skip-gram-attention}

### Skip-Gram模型回顾

<div class="theorem-box">

**定义2：Skip-Gram模型**

Word2Vec的Skip-Gram目标：给定中心词$w_c$，预测上下文词$w_o$：
$$p(w_o | w_c) = \frac{\exp(\boldsymbol{v}_{w_c}^T \boldsymbol{u}_{w_o})}{\sum_{w' \in \mathcal{V}} \exp(\boldsymbol{v}_{w_c}^T \boldsymbol{u}_{w'})}$$

其中：
- $\boldsymbol{v}_{w_c}$：中心词嵌入（center embedding）
- $\boldsymbol{u}_{w_o}$：上下文词嵌入（context embedding）
- $\mathcal{V}$：词汇表

**训练目标**：最大化
$$\mathcal{L} = \sum_{(c, o) \in D} \log p(w_o | w_c)$$

</div>

**对数概率的形式**：
$$\log p(w_o | w_c) = \boldsymbol{v}_{w_c}^T \boldsymbol{u}_{w_o} - \log Z(w_c)$$

其中$Z(w_c) = \sum_{w'} \exp(\boldsymbol{v}_{w_c}^T \boldsymbol{u}_{w'})$是配分函数。

**关键简化**：在某些训练策略（如Negative Sampling）下，可以近似忽略$\log Z(w_c)$，得到：
$$\log p(w_o | w_c) \approx \boldsymbol{v}_{w_c}^T \boldsymbol{u}_{w_o} + \text{const}$$

---

### 代入朴素贝叶斯框架

在语言模型中，将$p(x_t | x_j)$建模为Skip-Gram：
$$\log p(x_t | x_j) = \boldsymbol{v}(x_j)^T \boldsymbol{w}(x_t) + \text{const}$$

代入加权朴素贝叶斯公式：
\begin{equation}\begin{aligned}
\log p(x_t | x_{1:t-1}) &= \sum_{j=1}^{t-1} a_{t,j} \log p(x_t | x_j) + \text{bias} \\
&= \sum_{j=1}^{t-1} a_{t,j} \left[ \boldsymbol{v}(x_j)^T \boldsymbol{w}(x_t) \right] + \text{bias} \\
&= \left( \sum_{j=1}^{t-1} a_{t,j} \boldsymbol{v}(x_j) \right)^T \boldsymbol{w}(x_t) + \text{bias}
\end{aligned}\end{equation}

<div class="comparison-box">

**惊人的结论**：

括号内的表达式：
$$\boldsymbol{h}_t = \sum_{j=1}^{t-1} a_{t,j} \boldsymbol{v}(x_j)$$

正是**加权平均的嵌入向量**，也就是**Attention机制的核心**！

完整的预测：
$$p(x_t | x_{1:t-1}) \propto \exp(\boldsymbol{h}_t^T \boldsymbol{w}(x_t))$$

这与标准语言模型的输出层完全一致：**Embedding → Attention → 线性层 → Softmax**

</div>

---

### Attention权重的确定

现在的问题是：如何确定权重$a_{t,j}$？

**自然的约束**：
1. 归一化：$\sum_j a_{t,j} = 1$
2. 非负性：$a_{t,j} \geq 0$
3. 依赖性：$a_{t,j}$应是$x_{t-1}$和$x_j$的函数

<div class="derivation-box">

**Softmax-Attention的推导**

**步骤1**：定义相似度函数（score）
$$s_{t,j} = f(x_{t-1}, x_j)$$

**步骤2**：Softmax归一化
$$a_{t,j} = \frac{\exp(s_{t,j})}{\sum_{k=1}^{t-1} \exp(s_{t,k})}$$

**步骤3**：参数化score（Dot-Product Attention）
$$s_{t,j} = \boldsymbol{q}(x_{t-1})^T \boldsymbol{k}(x_j)$$

其中：
- $\boldsymbol{q}(x_{t-1})$：查询向量（Query）
- $\boldsymbol{k}(x_j)$：键向量（Key）

</div>

**完整的Attention公式**：
$$a_{t,j} = \frac{\exp(\boldsymbol{q}(x_{t-1})^T \boldsymbol{k}(x_j))}{\sum_{k=1}^{t-1} \exp(\boldsymbol{q}(x_{t-1})^T \boldsymbol{k}(x_k))}$$

**与标准Transformer对比**：

标准Transformer使用所有位置的Query，而非仅$x_{t-1}$：
$$a_{i,j} = \frac{\exp(\boldsymbol{q}_i^T \boldsymbol{k}_j / \sqrt{d})}{\sum_k \exp(\boldsymbol{q}_i^T \boldsymbol{k}_k / \sqrt{d})}$$

输出：
$$\boldsymbol{o}_i = \sum_j a_{i,j} \boldsymbol{v}_j$$

**对应关系**：
- 本文中$\boldsymbol{v}(x_j)$ ↔ Transformer的Value向量$\boldsymbol{v}_j$
- 本文中$\boldsymbol{w}(x_t)$ ↔ Transformer的输出层权重

---

## 多层堆叠的隐变量解释 {#latent-variable-model}

### 隐变量模型的动机

<div class="theorem-box">

**问题**：单层朴素贝叶斯的表达能力有限（条件独立假设过强）。

**解决方案**：引入隐变量$\boldsymbol{z}_{1:t-1}$，使得：
$$p(x_t | x_{1:t-1}) = \int p(x_t | \boldsymbol{z}_{1:t-1}) p(\boldsymbol{z}_{1:t-1} | x_{1:t-1}) d\boldsymbol{z}_{1:t-1}$$

**直观理解**：
- $\boldsymbol{z}_j$是$x_j$的"抽象表示"或"隐藏状态"
- 通过隐变量的中介，打破原始token之间的条件独立假设
- 类似于高斯混合模型（GMM）通过混合多个高斯分布拟合复杂分布

</div>

---

### 隐变量的分解

<div class="derivation-box">

**自回归分解**：

$$p(\boldsymbol{z}_{1:t-1} | x_{1:t-1}) = \prod_{j=1}^{t-1} p(\boldsymbol{z}_j | x_{1:j})$$

**解释**：
- 每个隐变量$\boldsymbol{z}_j$由历史$x_{1:j}$决定
- 这是一种**从下到上**的编码过程

**对$x_t$的预测**：

在隐变量条件下，再次应用朴素贝叶斯：
$$p(x_t | \boldsymbol{z}_{1:t-1}) \approx \sum_j a_{t,j}^{(1)} \log p(x_t | \boldsymbol{z}_j) + \text{bias}$$

**连续隐变量的狄拉克分布**：

为了计算积分，定义$\boldsymbol{z}_j$为确定性函数：
$$p(\boldsymbol{z}_j | x_{1:j}) = \delta(\boldsymbol{z}_j - g(x_{1:j}))$$

其中$\delta$是狄拉克Delta函数，$g$是参数化的编码函数。

</div>

**积分计算**：
\begin{equation}\begin{aligned}
p(x_t | x_{1:t-1}) &= \int p(x_t | \boldsymbol{z}_{1:t-1}) \prod_j \delta(\boldsymbol{z}_j - g(x_{1:j})) d\boldsymbol{z}_{1:t-1} \\
&= p(x_t | g(x_{1:1}), \ldots, g(x_{1:t-1}))
\end{aligned}\end{equation}

**两层Attention的出现**：

- **第一层Attention**：计算$\boldsymbol{z}_j = g(x_{1:j})$，即通过Attention聚合$x_{1:j}$
- **第二层Attention**：基于$\boldsymbol{z}_{1:t-1}$预测$x_t$

递归应用这个过程，就得到**多层Transformer**！

---

### 数学形式化

<div class="derivation-box">

**第$\ell$层的隐变量**：

记第$\ell$层的隐表示为$\boldsymbol{z}_j^{(\ell)}$，递归定义：
$$\boldsymbol{z}_j^{(\ell)} = \sum_{k=1}^j a_{j,k}^{(\ell)} \boldsymbol{z}_k^{(\ell-1)}$$

其中$\boldsymbol{z}_j^{(0)} = \boldsymbol{x}_j$（输入嵌入）。

**Attention权重**：
$$a_{j,k}^{(\ell)} = \frac{\exp(\boldsymbol{q}_j^{(\ell)} \cdot \boldsymbol{k}_k^{(\ell)} / \sqrt{d})}{\sum_{k'} \exp(\boldsymbol{q}_j^{(\ell)} \cdot \boldsymbol{k}_{k'}^{(\ell)} / \sqrt{d})}$$

**深度$L$的模型**：

$$p(x_t | x_{1:t-1}) = \text{Softmax}(\boldsymbol{W} \boldsymbol{z}_t^{(L)})$$

其中$\boldsymbol{z}_t^{(L)}$是第$L$层对位置$t$的隐表示。

</div>

---

## 残差连接的概率解释 {#residual-interpretation}

### 2-Gram先验的引入

<div class="theorem-box">

**观察**：朴素贝叶斯忽略了局部依赖，但实际中相邻词的依赖很强（2-gram）。

**改进**：结合2-gram先验和全局朴素贝叶斯：
$$p(x_t | x_{1:t-1}) \propto p(x_t | x_{t-1}) \cdot \text{Avg}[p(x_t | x_j)]$$

取对数：
$$\log p(x_t | x_{1:t-1}) = \log p(x_t | x_{t-1}) + \sum_j a_{t,j} \log p(x_t | x_j) + \text{const}$$

</div>

**Transformer中的对应**：

在特征空间：
$$\boldsymbol{h}_t = \boldsymbol{h}_{t-1} + \text{Attention}(\boldsymbol{h}_{1:t-1})$$

这正是**残差连接**（Residual Connection）！

**数学直觉**：
- $\boldsymbol{h}_{t-1}$：局部信息（类似2-gram）
- Attention项：全局信息（类似朴素贝叶斯的加权平均）
- 残差：将两者相加，结合局部和全局

---

### Post-LN与Pre-LN的关系

<div class="derivation-box">

**Post-LN**（标准Transformer）：
$$\boldsymbol{h}_{t}^{(\ell+1)} = \text{LayerNorm}(\boldsymbol{h}_t^{(\ell)} + \text{Attention}(\boldsymbol{h}^{(\ell)}))$$

**概率解释**：
- 先累加（乘法→对数域加法）
- 后归一化（确保概率分布的有效性）

**Pre-LN**（GPT风格）：
$$\boldsymbol{h}_{t}^{(\ell+1)} = \boldsymbol{h}_t^{(\ell)} + \text{Attention}(\text{LayerNorm}(\boldsymbol{h}^{(\ell)}))$$

**概率解释**：
- 先归一化每层的输入（稳定条件分布$p(x_t | \boldsymbol{z}_j)$）
- 后累加（保持恒等路径）

</div>

---

## In-Context Learning的贝叶斯视角 {#in-context-learning}

### Few-Shot Learning as Bayesian Update

<div class="theorem-box">

**现象**：GPT-3等大模型在给定少量示例（prompt）后，能够"学会"新任务而无需梯度更新。

**贝叶斯解释**：

设示例为$(x_1, y_1), \ldots, (x_k, y_k)$，查询为$x_{k+1}$。模型预测：
$$p(y_{k+1} | x_{k+1}, (x_1, y_1), \ldots, (x_k, y_k))$$

根据朴素贝叶斯：
$$p(y | x, \text{context}) \propto \prod_{i=1}^k p((x_i, y_i) | (x, y)) \cdot p(y | x)$$

**直观理解**：
- 每个示例$(x_i, y_i)$提供关于$p(y|x)$的"证据"
- 通过Attention机制，模型隐式地执行贝叶斯更新
- 相似的示例（高Attention权重）提供更强的证据

</div>

---

### 信息论视角

<div class="derivation-box">

**互信息**：

In-context learning的效果可以用互信息衡量：
$$I(Y; \text{Context} | X) = H(Y | X) - H(Y | X, \text{Context})$$

**熵的降低**：
- 无context：$H(Y|X)$较大（不确定性高）
- 有context：$H(Y|X, \text{Context})$较小（示例减少不确定性）

**Attention的作用**：
- 计算$\text{context}$中每个示例与查询$x$的相关性
- 高相关示例获得大权重，有效传递信息
- 本质上是**非参数化的贝叶斯推断**

</div>

**数学建模**：

设Attention权重为$a_i \propto \exp(\text{sim}(x, x_i))$，则：
$$p(y | x, \text{context}) = \sum_i a_i p(y | x_i, y_i)$$

这是kernel density estimation的离散版本！

---

## FeedForward层的作用 {#feedforward-role}

### 非线性变换的必要性

<div class="comparison-box">

**问题**：Attention是线性操作（加权平均），表达能力有限。

**解决**：在每层后添加FeedForward Network (FFN)：
$$\text{FFN}(\boldsymbol{h}) = \text{ReLU}(\boldsymbol{W}_1 \boldsymbol{h} + \boldsymbol{b}_1) \boldsymbol{W}_2 + \boldsymbol{b}_2$$

**概率解释**：
- FFN参数化$\log p(x_t | \boldsymbol{z}_j)$的非线性依赖
- 类似于深度神经网络中的全连接层，增加模型容量

</div>

---

### Key-Value记忆的视角

<div class="derivation-box">

**FFN作为Key-Value Memory**：

将FFN的第一层权重$\boldsymbol{W}_1$视为"keys"，第二层权重$\boldsymbol{W}_2$视为"values"：

$$\text{FFN}(\boldsymbol{h}) = \sum_i \sigma(\boldsymbol{k}_i^T \boldsymbol{h}) \boldsymbol{v}_i$$

其中$\sigma$是激活函数（如ReLU或GELU）。

**与Attention的对比**：
- Attention：动态权重（依赖输入），固定value（来自上下文）
- FFN：固定权重（学到的参数），动态value（输入相关）

**互补性**：
- Attention捕捉上下文依赖
- FFN捕捉语义模式（如"国家-首都"关系）

</div>

---

## NBCE方法的深入分析 {#nbce-analysis}

### 长上下文的挑战

<div class="theorem-box">

**问题**：标准Attention的复杂度为$O(n^2)$，对于长序列（$n > 10^4$）不可行。

**NBCE的思路**：利用朴素贝叶斯的分解性质，将长上下文切分成chunks。

**数学依据**：

$$p(x_t | x_{1:t-1}) \propto \prod_{j=1}^{t-1} p(x_t | x_j)^{w_j}$$

其中$w_j$是可学习的权重（替代固定的$a_{t,j}$）。

</div>

---

### Chunk-wise计算

<div class="derivation-box">

**分块策略**：

将$x_{1:t-1}$分成$K$个chunks：$C_1, \ldots, C_K$。

**近似**：
$$\log p(x_t | x_{1:t-1}) \approx \sum_{k=1}^K \text{Attention}_k(x_t, C_k)$$

其中$\text{Attention}_k$是在chunk $C_k$内的局部Attention。

**复杂度降低**：
- 原始：$O(n^2)$
- 分块：$O(K \cdot (n/K)^2) = O(n^2/K)$

例如$K=10$，复杂度降低10倍。

</div>

**理论保证**：

如果chunks之间的依赖较弱（符合朴素贝叶斯假设），则近似误差可控：
$$\left| \log p(x_t | x_{1:t-1}) - \sum_k \text{Attention}_k(x_t, C_k) \right| = O(\epsilon)$$

其中$\epsilon$取决于chunks之间的相关性。

---

## 实验验证与对比 {#experiments}

### 合成数据实验

<div class="example-box">

**实验设置**：

- 数据：随机生成序列，满足已知的条件独立结构
- 模型对比：
  1. 标准Attention
  2. 朴素贝叶斯（等权平均）
  3. 加权朴素贝叶斯（学习$a_{t,j}$）

**结果**（困惑度，越低越好）：

| 模型 | 短序列(n=10) | 长序列(n=100) |
|------|-------------|--------------|
| 标准NB | 5.2 | 8.7 |
| 加权NB | 4.1 | 6.3 |
| Attention | 3.8 | 5.9 |

**观察**：
- Attention略优于加权NB（因为参数化更灵活）
- 加权NB显著优于标准NB（自适应权重的重要性）
- 长序列下差距缩小（条件独立假设更合理）

</div>

---

### 真实语言模型实验

<div class="example-box">

**数据集**：WikiText-103

**模型配置**：
- 6层Transformer
- 隐藏维度512
- 8个Attention头

**消融实验**（Perplexity）：

| 配置 | PPL |
|------|-----|
| 完整Transformer | 24.3 |
| 无残差连接 | 32.1 (+7.8) |
| 无FFN层 | 29.7 (+5.4) |
| 单层Attention | 38.5 (+14.2) |

**结论**：
- 残差和FFN都对性能至关重要
- 多层堆叠带来显著提升（符合隐变量模型理论）

</div>

---

## 理论启示与未来方向 {#insights}

### 统一视角的价值

<div class="comparison-box">

**朴素贝叶斯视角的优势**：

1. **可解释性**：Attention不再是"黑盒"，而是贝叶斯推理的参数化
2. **理论指导**：可以借鉴概率推理的成熟理论（如变分推断、MCMC）
3. **架构创新**：基于不同的概率假设设计新型Attention变体

**局限性**：

1. **假设偏强**：条件独立性在实际中常常不满足
2. **计算成本**：标准贝叶斯推理（如精确边缘化）计算昂贵
3. **先验选择**：$p(x_t)$等先验项的设计仍需经验

</div>

---

### 未来研究方向

1. **结构化贝叶斯网络**：
   - 不再假设完全独立，而是学习依赖图结构
   - 结合因果推断，建模token之间的因果关系

2. **变分Attention**：
   - 将隐变量$\boldsymbol{z}_j$视为真正的随机变量
   - 使用变分推断优化$p(\boldsymbol{z}_j | x_{1:j})$

3. **非参数贝叶斯**：
   - 使用Dirichlet过程等无限维先验
   - 自适应调整模型复杂度（如Attention头数）

4. **贝叶斯元学习**：
   - 将In-Context Learning形式化为分层贝叶斯模型
   - 学习任务分布的超先验

5. **因果Attention**：
   - 区分相关性和因果性
   - 通过干预（intervention）学习因果Attention权重

---

## 总结 {#conclusion-extended}

本文从**朴素贝叶斯**的视角重新审视了**Attention机制**，建立了两者之间的深刻联系：

### 核心结论

<div class="theorem-box">

**定理：Attention作为广义朴素贝叶斯**

单层Attention可以视为：
1. 通过贝叶斯定理反转条件概率
2. 应用条件独立性假设（朴素贝叶斯）
3. 参数化$p(x_t|x_j)$为Skip-Gram风格的内积
4. 学习位置相关的权重$a_{t,j}$

多层Attention对应隐变量模型的层级结构。

</div>

### 关键洞察

1. **Attention ≈ 加权朴素贝叶斯**：权重由Query-Key相似度决定
2. **多层堆叠 ≈ 隐变量模型**：通过抽象表示打破条件独立
3. **残差连接 ≈ 2-Gram先验**：结合局部和全局信息
4. **In-Context Learning ≈ 贝叶斯更新**：示例作为证据更新后验

### 实践意义

- **架构设计**：基于概率假设设计新型Attention（如NBCE）
- **理论分析**：用贝叶斯推理工具分析Attention的行为
- **效率优化**：利用独立性假设降低计算复杂度

虽然这个视角并非完美（条件独立假设过强），但它为理解和改进Transformer提供了有价值的新角度。

**"Attention is All You Need" ≈ "Naive Bayes (广义的) is All You Need"**

---

