---
title: 训练1000层的Transformer究竟有什么困难？
slug: 训练1000层的transformer究竟有什么困难
date: 2022-03-09
tags: 优化, 梯度, attention, 生成模型, attention
status: completed
---

# 训练1000层的Transformer究竟有什么困难？

**原文链接**: [https://spaces.ac.cn/archives/8978](https://spaces.ac.cn/archives/8978)

**发布日期**: 

---

众所周知，现在的Transformer越做越大，但这个“大”通常是“宽”而不是“深”，像GPT-3虽然参数有上千亿，但也只是一个96层的Transformer模型，与我们能想象的深度相差甚远。是什么限制了Transformer往“深”发展呢？可能有的读者认为是算力，但“宽而浅”的模型所需的算力不会比“窄而深”的模型少多少，所以算力并非主要限制，归根结底还是Transformer固有的训练困难。一般的观点是，深模型的训练困难源于梯度消失或者梯度爆炸，然而实践显示，哪怕通过各种手段改良了梯度，深模型依然不容易训练。

近来的一些工作（如[Admin](https://papers.cool/arxiv/2004.08249)）指出，深模型训练的根本困难在于“增量爆炸”，即模型越深对输出的扰动就越大。上周的论文[《DeepNet: Scaling Transformers to 1,000 Layers》](https://papers.cool/arxiv/2203.00555)则沿着这个思路进行尺度分析，根据分析结果调整了模型的归一化和初始化方案，最终成功训练出了1000层的Transformer模型。整个分析过程颇有参考价值，我们不妨来学习一下。

## 增量爆炸 #

原论文的完整分析比较长，而且有些假设或者描述细酌之下是不够合理的。所以在本文的分享中，笔者会尽量修正这些问题，试图以一个更合理的方式来得到类似结果。

假设损失函数为$\mathcal{L}(\boldsymbol{\theta})$，$\boldsymbol{\theta}$是它的参数，考虑参数由$\boldsymbol{\theta}$变为$\boldsymbol{\theta}+\Delta\boldsymbol{\theta}$时损失函数的增量：  
\begin{equation}\Delta\mathcal{L} = \mathcal{L}(\boldsymbol{\theta}+\Delta\boldsymbol{\theta}) - \mathcal{L}(\boldsymbol{\theta}) \approx \langle\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}),\Delta\boldsymbol{\theta}\rangle\end{equation}  
对于SGD有$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$，那么$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2$。设模型有$N$层，每层有$K$个参数矩阵（$K$接近常数），配合Xavier初始化以及各种Normalization手段，我们可以使得每个参数矩阵的梯度模长是$\mathcal{O}(1)$量级，所以有$\Delta\mathcal{L}=\mathcal{O}(\eta NK)$。因此，模型每一步的更新量是正比于模型深度$N$的，如果模型越深，那么更新量就越大，这意味着初始阶段模型越容易进入不大好的局部最优点，然后训练停滞甚至崩溃，这就是“增量爆炸”问题。

这时候解决方法有两个，一是初始阶段用更小的学习率进行训练（不超过$\eta/N$量级），然后慢慢增大学习率，这就是Warmup技巧；二就是调整初始化方案，使得参数的梯度是$\mathcal{O}(1/\sqrt{N})$量级，这样就自动抵消掉模型深度的影响。

## 量级分析 #

怎么做到第二种方案呢？我们可以尝试分析Transformer的梯度。然而，精确的梯度求起来比较繁琐，并且事实上我们也不需要精确的梯度，而只是要对梯度做一个量级分析，所以我们可以用如下的“量级分解”技巧转化为标量的导数问题。

对于一个矩阵$\boldsymbol{W}$，我们将其分解为$\boldsymbol{W}=\lambda \boldsymbol{U}$的形式，其中  
\begin{equation}\lambda = \mathop{\text{argmin}}_{\kappa > 0} \Vert \boldsymbol{W}\boldsymbol{W}^{\top}/\kappa^2 - \boldsymbol{I}\Vert,\quad \end{equation}  
说白了，我们就是要将一个矩阵分解为一个标量$\lambda$与一个尽可能正交的矩阵$\boldsymbol{U}$之积。由于$\boldsymbol{U}$接近正交矩阵，它起到了一个标准参考系的作用，而对应的$\lambda$则代表了矩阵$\boldsymbol{W}$的量级。如果$\boldsymbol{W}$使用Xavier初始化，那么$\lambda$相当于其中的gain参数，即在Xavier初始化的基础上还要再乘一个$\lambda$。这是因为Xavier初始化的结果就接近一个正交矩阵，这一点可以参考[《从几何视角来理解模型参数的初始化策略》](/archives/7180)。

在此分解之下，我们有  
\begin{equation}\frac{\partial \mathcal{L}(\lambda \boldsymbol{U})}{\partial \lambda} = \left\langle\frac{\partial \mathcal{L}(\lambda \boldsymbol{U})}{\partial (\lambda \boldsymbol{U})}, \boldsymbol{U}\right\rangle = \left\langle\frac{\partial \mathcal{L}(\boldsymbol{W})}{\partial \boldsymbol{W}}, \boldsymbol{U}\right\rangle\end{equation}  
这意味着$\frac{\partial \mathcal{L}}{\partial \lambda}$跟$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}$在量级上是成正比的，所以对$\frac{\partial \mathcal{L}}{\partial \lambda}$做量级分析就相当于对$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}$做量级分析。这样$\frac{\partial \mathcal{L}}{\partial \lambda}$就相当于$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}$量级的一个简单的“探针”，原来的矩阵求导就可以转化为标量求导，降低了分析难度。

## 前馈梯度 #

很多实验结果都显示虽然Pre Norm比Post Norm更容易训练，但Post Norm的最终效果往往更好些，所以原论文保留了Post Norm结构，并考虑了更一般的形式（DeepNorm）：  
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\alpha\boldsymbol{x}_l + F(\boldsymbol{x}_l)) = \text{LN}(\boldsymbol{x}_l + F(\boldsymbol{x}_l)/\alpha)\end{equation}  
其中$\alpha > 0$是一个常数。简单起见，我们先考虑FFN层，此时  
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \phi(\boldsymbol{x}_l \boldsymbol{W}_1)\boldsymbol{W}_2/\alpha)\end{equation}  
这里的$\phi$是激活函数，一般为ReLU或其变体（Swish、GeLU等），它们（近似）满足$\phi(\lambda x) = \lambda \phi(x),\forall \lambda > 0$。使用前一节的量级分解探针，我们得到  
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\underbrace{\boldsymbol{x}_l + \lambda_1 \lambda_2 \phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2/\alpha}_{\text{记为}\boldsymbol{z}_{l+1}})\label{eq:ffn}\end{equation}  
求$\lambda$的梯度：  
\begin{equation}\begin{aligned}  
\frac{\partial \mathcal{L}}{\partial \lambda_1} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_1} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_2 \phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2}{\alpha} \\\  
\frac{\partial \mathcal{L}}{\partial \lambda_2} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_2} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_1 \phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2}{\alpha} \end{aligned}\end{equation}  
我们断言$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$都是$\mathcal{O}(1)$的，并且由于$\boldsymbol{U}_1$、$\boldsymbol{U}_2$都接近正交矩阵，所以$\phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2$也是$\mathcal{O}(1)$的，因此最终有  
\begin{equation}\frac{\partial \mathcal{L}}{\partial \lambda_1} = \mathcal{O}\left(\frac{\lambda_2}{\alpha}\right),\quad \frac{\partial \mathcal{L}}{\partial \lambda_2} = \mathcal{O}\left(\frac{\lambda_1}{\alpha}\right)\end{equation}

## 自注意力 #

现在考虑自Self Attention，作为量级分析，我们考虑单头注意力即可，其形式为  
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \sigma(\boldsymbol{x}_l \boldsymbol{W}_q\boldsymbol{W}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{W}_v\boldsymbol{W}_o/\alpha)\end{equation}  
其中$\sigma(\cdot)$是softmax操作的简写，这里省略了Attention的scale操作。对上式进行量级分解后的形式为  
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\underbrace{\boldsymbol{x}_l + \lambda_v\lambda_o \sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o/\alpha}_{\text{记为}\boldsymbol{z}_{l+1}})\label{eq:sa}\end{equation}  
现在我们可以对各个$\lambda$分别求梯度，而由于softmax的存在，事实上$\lambda_q,\lambda_k$的梯度本身会很小，不会明显影响最终的更新量，所以其实我们考虑$\lambda_v,\lambda_o$的更新量足矣：  
\begin{equation}\begin{aligned}  
\frac{\partial \mathcal{L}}{\partial \lambda_v} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_v} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_o \sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o}{\alpha} \\\  
\frac{\partial \mathcal{L}}{\partial \lambda_o} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_o} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_v \sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o}{\alpha} \end{aligned}\end{equation}  
同样断言$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$都是$\mathcal{O}(1)$的，并且注意softmax出来是一个概率分布，然后对$\boldsymbol{x}_l$的各个token做加权平均，通常而言，平均前后的向量会在同一数量级，所以我们认为$\sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o$也是$\mathcal{O}(1)$的，因此结果跟FFN层的类似：  
\begin{equation}\frac{\partial \mathcal{L}}{\partial \lambda_v} = \mathcal{O}\left(\frac{\lambda_o}{\alpha}\right),\quad \frac{\partial \mathcal{L}}{\partial \lambda_o} = \mathcal{O}\left(\frac{\lambda_v}{\alpha}\right)\end{equation}

## 初步结论 #

现在不管是FFN还是Self Attention，我们都得到了相似的结论，现在简单起见，假设每个参数的量级（至少在初始化阶段）是一致的，即所有的$\lambda$取同一个值，那么总的结论是  
\begin{equation}\frac{\partial \mathcal{L}}{\partial \lambda} = \mathcal{O}\left(\frac{\lambda}{\alpha}\right)\end{equation}  
即梯度的量级是$\mathcal{O}(\lambda/\alpha)$。另一方面，我们说$N$层的Transformer模型，一般是$N$层的Self Attention加$N$层的FFN，所以严格来说层数是$2N$。因此，按照“增量爆炸”一节的分析，我们需要将梯度调整到$\mathcal{O}(1/\sqrt{2N})$，上式告诉我们可以通过让$\lambda/\alpha=1/\sqrt{2N}$来实现。原论文的放缩更为宽松一些，得到的结果是$\lambda/\alpha = 1/\sqrt{4N}$，量级上是等价的。

现在我们得到了$\lambda$与$\alpha$的一个比例关系，但无法直接得到$\lambda$和$\alpha$的具体值。按照论文的说法，是从对称角度出发，让$\lambda=1/\alpha$，从而可以解得  
\begin{equation}\alpha = (2N)^{1/4},\quad \lambda = (2N)^{-1/4}\label{eq:result}\end{equation}  
然而，单纯对称的解释显然是不够说服力的，我们需要搞清楚不同的选择究竟有什么不同的结果。为此，我们可以比较另外两组解：

> **另解一：** $\alpha=1,\lambda=(2N)^{-1/2}$，此时参数的初始化缩小到原来的$(2N)^{-1/2}$倍，梯度也被缩小到原来的$(2N)^{-1/2}$倍，根据SGD的$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$得出每步的更新量也是原来的$(2N)^{-1/2}$倍，也就是说，调整前后的相对学习幅度是没有变化的，因此有可能刚开始$\lambda=\mathcal{O}((2N)^{-1/2})$级别，但训练集几步后就脱离了这个量级了。
> 
> **另解二：** $\alpha=(2N)^{1/2},\lambda=1$，此时参数的初始化没有缩小，但梯度也被缩小到原来的$(2N)^{-1/2}$倍，根据SGD的$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$得出每步的更新量也是原来的$(2N)^{-1/2}$倍，调整前后的相对学习幅度是明显缩小了，因此有可能出现学习得非常慢的情况。

这两种情况看上去都各有缺点，因此介乎两者之间的式$\eqref{eq:result}$似乎就能解释得通了，它就是保持梯度缩放到原来的$(2N)^{-1/2}$倍的同时，让初始学习步伐稍微慢一些，但又不至于太慢，隐式地起到了Warmup的作用。

## 多种优化 #

上面的分析都是基于SGD进行的，但事实上我们很少直接用SGD去训练NLP模型，我们更多是自适应学习率优化器，主要有两大类：一是用二阶矩来校正学习率，Adam、AdamW等都属此类；另一类是通过参数模长进一步校正学习率，比如[LAMB](/archives/7094)、[AdaFactor](/archives/7302)。原论文的说法是“我们在SGD上进行推导，然后在Adam上验证发现也还可以”，但从理论上来讲，它们并不完全通用，这一节我们就来针对性地做一下分析。

对于Adam类优化器来说，每一步的更新量大约为$\Delta\boldsymbol{\theta}=-\eta\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}))$，所以$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1$，它是正比于梯度的1次方而不是2次方，因此要过要让更新量跟层数无关，那么梯度应该缩小到原来的$1/(2N)$倍才对，即应该有$\lambda/\alpha=1/(2N)$，如果同样让$\lambda=1/\alpha$，那么有  
\begin{equation}\alpha = (2N)^{1/2},\quad \lambda = (2N)^{-1/2}\end{equation}

对于LAMB类优化器来说，每一步更新量大约为$\Delta\boldsymbol{\theta}=-\eta\Vert\theta\Vert\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}))$，所以$\Delta\mathcal{L} \approx -\eta\Vert\theta\Vert\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1$，注意到参数的缩放比例是$\lambda$、梯度的缩放比例是$\lambda/\alpha$，所以$\Delta\mathcal{L}=\mathcal{O}(2N\lambda^2/\alpha)$，从而是$\lambda^2/\alpha=1/(2N)$。注意这类优化器每步的相对更新量是一样的（等于学习率$\eta$），不管怎么调整$\alpha,\lambda$其相对更新大小都不会变化，所以我们可以直接取$\alpha=1,\lambda=(2N)^{-1/2}$。

结果汇总对比如下：  
\begin{array}{c|cc|cc}  
\hline  
\text{优化器} & \Delta\boldsymbol{\theta} & \Delta\mathcal{L} & \alpha & \lambda \\\  
\hline  
\text{SGD} & -\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}) & -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2 & (2N)^{1/4} & (2N)^{-1/4}\\\  
\text{Adam} & -\eta\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})) & -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1 & (2N)^{1/2}& (2N)^{-1/2}\\\  
\text{LAMB} & -\eta\Vert\theta\Vert\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})) & -\eta\Vert\theta\Vert\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1 & 1 & (2N)^{-1/2}\\\  
\hline  
\end{array}

## 事后分析 #

前面的两节推导过程都用到了断言“$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$都是$\mathcal{O}(1)$的”，那么它是否成立呢？这里我们事后分析一下。

其实也很简单，经过前述调整后，不管是FFN层$\eqref{eq:ffn}$还是Self Attention层$\eqref{eq:sa}$，初始阶段每个残差分支的权重被缩放到原来的$\lambda^2/\alpha$倍，不管是哪种优化器的结果，$\lambda^2/\alpha$都是一个比较小的数字，这意味着初始阶段整个模型其实接近一个恒等函数，因此$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$自然都是$\mathcal{O}(1)$的，所以结论和断言是自洽的。

另外，可能有读者想问同样的分析是否可以用到Pre Norm结构上呢？答案是可以的，并且结论是基本一致的，只是因为Norm放在了残差分支之前，所以就没必要设置$\alpha$参数了，所以结论就是上述关于Post Norm的结果中所有的$\alpha$都等于为1，然后重新计算相应的$\lambda$。

最后，读者可能有疑问的是花了那么多功夫讨论把模型做深，那么模型深度真有那么重要吗？有，原论文给出了一个漂亮的实验结果，用一个200层的“深而窄”的模型（32亿参数），战胜了之前48层“浅而宽”的SOTA模型（120亿参数）：  


[![“深而窄”的模型胜于“浅而宽”的模型](/usr/uploads/2022/03/2952207079.png)](/usr/uploads/2022/03/2952207079.png "点击查看原图")

“深而窄”的模型胜于“浅而宽”的模型

## 文章小结 #

本文分析了将Transformer做“深”的瓶颈所在并给出了相应的解决方案，文章的主要思路源于微软新出的DeepNet，并对原论文的分析过程做了一定的简化和完善。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8978>_

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

苏剑林. (Mar. 09, 2022). 《训练1000层的Transformer究竟有什么困难？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8978>

@online{kexuefm-8978,  
title={训练1000层的Transformer究竟有什么困难？},  
author={苏剑林},  
year={2022},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/8978}},  
} 


---

## 公式推导与注释

### 一、深层网络的梯度问题数学分析

#### 1.1 梯度消失与梯度爆炸的基础理论

考虑一个 $N$ 层的深度神经网络，损失函数为 $\mathcal{L}$。第 $l$ 层的参数 $\boldsymbol{\theta}_l$ 的梯度可以通过链式法则表示：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_N} \prod_{i=l+1}^{N} \frac{\partial \boldsymbol{x}_i}{\partial \boldsymbol{x}_{i-1}} \frac{\partial \boldsymbol{x}_l}{\partial \boldsymbol{\theta}_l}
\tag{1}
\end{equation}

其中 $\boldsymbol{x}_i$ 是第 $i$ 层的输出。

**关键观察**: 梯度包含一个连乘项 $\prod_{i=l+1}^{N} \frac{\partial \boldsymbol{x}_i}{\partial \boldsymbol{x}_{i-1}}$，这是梯度传播的雅可比矩阵链。

#### 1.2 雅可比矩阵的范数分析

设每层的雅可比矩阵 $\boldsymbol{J}_i = \frac{\partial \boldsymbol{x}_i}{\partial \boldsymbol{x}_{i-1}}$，其范数为 $\|\boldsymbol{J}_i\|$。则：

\begin{equation}
\left\|\prod_{i=l+1}^{N} \boldsymbol{J}_i\right\| \leq \prod_{i=l+1}^{N} \|\boldsymbol{J}_i\|
\tag{2}
\end{equation}

**情况1: 梯度消失** - 如果 $\|\boldsymbol{J}_i\| < 1$，设 $\|\boldsymbol{J}_i\| = \gamma < 1$:

\begin{equation}
\left\|\prod_{i=l+1}^{N} \boldsymbol{J}_i\right\| \leq \gamma^{N-l}
\tag{3}
\end{equation}

当 $N-l$ 很大时，$\gamma^{N-l} \to 0$，导致梯度消失。

**情况2: 梯度爆炸** - 如果 $\|\boldsymbol{J}_i\| > 1$，设 $\|\boldsymbol{J}_i\| = \beta > 1$:

\begin{equation}
\left\|\prod_{i=l+1}^{N} \boldsymbol{J}_i\right\| \geq \beta^{N-l}
\tag{4}
\end{equation}

当 $N-l$ 很大时，$\beta^{N-l} \to \infty$，导致梯度爆炸。

**数值示例**: 假设 $\gamma = 0.9$，$N = 100$，$l = 1$:

\begin{equation}
0.9^{99} \approx 2.65 \times 10^{-5}
\tag{5}
\end{equation}

梯度衰减了约 $10^5$ 倍！

#### 1.3 Transformer中的梯度传播

标准Transformer的一层包含两个子层：
1. 自注意力层: $\boldsymbol{y}_l = \boldsymbol{x}_l + \text{Attention}(\boldsymbol{x}_l)$
2. FFN层: $\boldsymbol{x}_{l+1} = \boldsymbol{y}_l + \text{FFN}(\boldsymbol{y}_l)$

不含残差时，雅可比矩阵为：

\begin{equation}
\boldsymbol{J}_l = \frac{\partial \text{FFN}(\text{Attention}(\boldsymbol{x}_l))}{\partial \boldsymbol{x}_l}
\tag{6}
\end{equation}

含残差时（Post-LN）：

\begin{equation}
\boldsymbol{J}_l = \boldsymbol{I} + \frac{\partial \text{FFN}(\boldsymbol{y}_l)}{\partial \boldsymbol{y}_l} \left(\boldsymbol{I} + \frac{\partial \text{Attention}(\boldsymbol{x}_l)}{\partial \boldsymbol{x}_l}\right)
\tag{7}
\end{equation}

**关键洞察**: 残差连接引入的 $\boldsymbol{I}$ 项确保 $\boldsymbol{J}_l$ 的特征值至少有1，防止梯度消失。

### 二、增量爆炸问题的深入分析

#### 2.1 增量爆炸的数学定义

对于参数更新 $\Delta \boldsymbol{\theta} = -\eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$，损失函数的一阶近似变化为：

\begin{equation}
\Delta \mathcal{L} = \mathcal{L}(\boldsymbol{\theta} + \Delta\boldsymbol{\theta}) - \mathcal{L}(\boldsymbol{\theta}) \approx \langle \nabla_{\boldsymbol{\theta}} \mathcal{L}, \Delta\boldsymbol{\theta} \rangle
\tag{8}
\end{equation}

对于SGD:

\begin{equation}
\Delta \mathcal{L} \approx -\eta \|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|^2
\tag{9}
\end{equation}

**问题**: 对于 $N$ 层Transformer，每层有 $K$ 个参数矩阵（Self-Attention有QKV和输出，FFN有两层，通常 $K \approx 6$）。

#### 2.2 深度模型的梯度范数

假设每个参数矩阵 $\boldsymbol{W}_i$ 的梯度范数为 $\|\nabla_{\boldsymbol{W}_i} \mathcal{L}\| = g$ (通过归一化和初始化控制)。

总梯度范数的平方：

\begin{equation}
\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|^2 = \sum_{i=1}^{NK} \|\nabla_{\boldsymbol{W}_i} \mathcal{L}\|^2 = NKg^2
\tag{10}
\end{equation}

因此：

\begin{equation}
\Delta \mathcal{L} \approx -\eta NK g^2 = \mathcal{O}(\eta N)
\tag{11}
\end{equation}

**关键结论**: 更新量正比于层数 $N$！

#### 2.3 增量爆炸的后果

对于深层网络（$N$ 很大）：
1. 初始阶段梯度步长过大
2. 容易跳过局部最优，进入次优区域
3. 训练不稳定，甚至发散

**数值示例**: $N=100$ 层 vs $N=10$ 层，相同学习率下更新量差 10 倍！

\begin{equation}
\frac{\Delta \mathcal{L}_{N=100}}{\Delta \mathcal{L}_{N=10}} = \frac{100}{10} = 10
\tag{12}
\end{equation}

### 三、LayerNorm的数学推导

#### 3.1 LayerNorm的定义

对于输入 $\boldsymbol{x} \in \mathbb{R}^d$，LayerNorm计算：

\begin{equation}
\text{LN}(\boldsymbol{x}) = \boldsymbol{\gamma} \odot \frac{\boldsymbol{x} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} + \boldsymbol{\beta}
\tag{13}
\end{equation}

其中：
- $\boldsymbol{\mu} = \frac{1}{d} \sum_{i=1}^{d} x_i$ (均值)
- $\boldsymbol{\sigma}^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$ (方差)
- $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ 是可学习参数
- $\epsilon$ 是数值稳定性常数（通常 $10^{-5}$）

#### 3.2 LayerNorm的梯度推导

设 $\boldsymbol{y} = \text{LN}(\boldsymbol{x})$，计算 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}}$：

首先计算归一化部分 $\hat{\boldsymbol{x}} = \frac{\boldsymbol{x} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}}$：

\begin{equation}
\frac{\partial \hat{x}_i}{\partial x_j} = \frac{1}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} \left(\delta_{ij} - \frac{1}{d}\right) - \frac{\hat{x}_i \cdot (x_j - \mu)}{d(\boldsymbol{\sigma}^2 + \epsilon)}
\tag{14}
\end{equation}

其中 $\delta_{ij}$ 是Kronecker delta。

完整的梯度为：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}} \left[\frac{\partial \mathcal{L}}{\partial y_i} - \frac{1}{d} \sum_{j=1}^{d} \frac{\partial \mathcal{L}}{\partial y_j} - \hat{x}_i \frac{1}{d} \sum_{j=1}^{d} \hat{x}_j \frac{\partial \mathcal{L}}{\partial y_j}\right]
\tag{15}
\end{equation}

**重要性质**: LayerNorm的梯度自动归一化，$\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}}\| = \mathcal{O}(1)$，有助于稳定训练。

#### 3.3 RMSNorm的简化推导

RMSNorm简化了LayerNorm，去掉了均值计算和偏置项：

\begin{equation}
\text{RMSNorm}(\boldsymbol{x}) = \boldsymbol{\gamma} \odot \frac{\boldsymbol{x}}{\text{RMS}(\boldsymbol{x})}
\tag{16}
\end{equation}

其中 RMS (Root Mean Square):

\begin{equation}
\text{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
\tag{17}
\end{equation}

**优势**:
1. 计算更快（少一次均值计算）
2. 梯度更简单
3. 实验效果与LayerNorm相当

RMSNorm的梯度：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\text{RMS}(\boldsymbol{x})} \left[\frac{\partial \mathcal{L}}{\partial y_i} - \frac{x_i}{d \cdot \text{RMS}(\boldsymbol{x})^2} \sum_{j=1}^{d} x_j \frac{\partial \mathcal{L}}{\partial y_j}\right]
\tag{18}
\end{equation}

### 四、Post-LN vs Pre-LN的理论分析

#### 4.1 Post-LN的结构

\begin{equation}
\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + F(\boldsymbol{x}_l))
\tag{19}
\end{equation}

其中 $F$ 是Self-Attention或FFN。

**梯度流**:

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \frac{\partial \text{LN}(\boldsymbol{x}_l + F(\boldsymbol{x}_l))}{\partial (\boldsymbol{x}_l + F(\boldsymbol{x}_l))} \left(\boldsymbol{I} + \frac{\partial F(\boldsymbol{x}_l)}{\partial \boldsymbol{x}_l}\right)
\tag{20}
\end{equation}

**问题**: LayerNorm的雅可比矩阵会重新缩放梯度，可能破坏残差连接的恒等映射。

#### 4.2 Pre-LN的结构

\begin{equation}
\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + F(\text{LN}(\boldsymbol{x}_l))
\tag{21}
\end{equation}

**梯度流**:

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \left(\boldsymbol{I} + \frac{\partial F(\text{LN}(\boldsymbol{x}_l))}{\partial \text{LN}(\boldsymbol{x}_l)} \frac{\partial \text{LN}(\boldsymbol{x}_l)}{\partial \boldsymbol{x}_l}\right)
\tag{22}
\end{equation}

**优势**: 恒等路径 $\boldsymbol{I}$ 不受LayerNorm影响，梯度传播更稳定。

#### 4.3 DeepNorm的改进

DeepNorm引入缩放因子 $\alpha$:

\begin{equation}
\boldsymbol{x}_{l+1} = \text{LN}(\alpha \boldsymbol{x}_l + F(\boldsymbol{x}_l))
\tag{23}
\end{equation}

等价形式：

\begin{equation}
\boldsymbol{x}_{l+1} = \text{LN}\left(\boldsymbol{x}_l + \frac{F(\boldsymbol{x}_l)}{\alpha}\right)
\tag{24}
\end{equation}

**设计原理**: 通过调整 $\alpha$ 控制残差分支的贡献，使初始阶段模型接近恒等函数。

### 五、量级分解与初始化策略

#### 5.1 矩阵的量级分解

将参数矩阵 $\boldsymbol{W}$ 分解为:

\begin{equation}
\boldsymbol{W} = \lambda \boldsymbol{U}
\tag{25}
\end{equation}

其中：
- $\lambda > 0$ 是标量增益
- $\boldsymbol{U}$ 接近正交矩阵 ($\boldsymbol{U}\boldsymbol{U}^\top \approx \boldsymbol{I}$)

**最优化问题**:

\begin{equation}
\lambda = \arg\min_{\kappa > 0} \|\boldsymbol{W}\boldsymbol{W}^\top / \kappa^2 - \boldsymbol{I}\|_F
\tag{26}
\end{equation}

解为：

\begin{equation}
\lambda = \sqrt{\frac{\text{tr}(\boldsymbol{W}\boldsymbol{W}^\top)}{d}} = \|\boldsymbol{W}\|_F / \sqrt{d}
\tag{27}
\end{equation}

#### 5.2 梯度的量级探针

对于损失函数 $\mathcal{L}(\lambda \boldsymbol{U})$，标量梯度为：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda} = \left\langle \frac{\partial \mathcal{L}}{\partial (\lambda \boldsymbol{U})}, \boldsymbol{U} \right\rangle = \left\langle \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}, \boldsymbol{U} \right\rangle
\tag{28}
\end{equation}

**关键性质**: $\frac{\partial \mathcal{L}}{\partial \lambda}$ 与 $\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}\|$ 在量级上成正比。

#### 5.3 Xavier初始化的数学原理

Xavier初始化要求前向传播时方差保持不变。对于线性层 $\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x}$:

\begin{equation}
\text{Var}(y_i) = \sum_{j=1}^{d_{\text{in}}} \text{Var}(W_{ij} x_j) = d_{\text{in}} \cdot \text{Var}(W_{ij}) \cdot \text{Var}(x_j)
\tag{29}
\end{equation}

要使 $\text{Var}(y_i) = \text{Var}(x_j)$，需要：

\begin{equation}
\text{Var}(W_{ij}) = \frac{1}{d_{\text{in}}}
\tag{30}
\end{equation}

考虑反向传播的对称性，最终：

\begin{equation}
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)
\tag{31}
\end{equation}

**矩阵视角**: Xavier初始化使 $\boldsymbol{W}$ 接近正交矩阵，即 $\lambda \approx 1$。

### 六、FFN层的梯度量级分析

#### 6.1 FFN的数学形式

\begin{equation}
\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \phi(\boldsymbol{x}_l \boldsymbol{W}_1) \boldsymbol{W}_2 / \alpha)
\tag{32}
\end{equation}

其中 $\phi$ 是激活函数（ReLU, GELU等）。

**齐次性**: 对于ReLU及其变体，满足 $\phi(\lambda \boldsymbol{x}) \approx \lambda \phi(\boldsymbol{x})$ (当 $\lambda > 0$)。

#### 6.2 量级分解后的形式

使用 $\boldsymbol{W}_i = \lambda_i \boldsymbol{U}_i$:

\begin{equation}
\boldsymbol{z}_{l+1} = \boldsymbol{x}_l + \lambda_1 \lambda_2 \phi(\boldsymbol{x}_l \boldsymbol{U}_1) \boldsymbol{U}_2 / \alpha
\tag{33}
\end{equation}

#### 6.3 梯度计算

对 $\lambda_1$ 的梯度：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda_1} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}} \frac{\lambda_2 \phi(\boldsymbol{x}_l \boldsymbol{U}_1) \boldsymbol{U}_2}{\alpha}
\tag{34}
\end{equation}

**量级估计**:
- $\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} = \mathcal{O}(1)$ (归一化作用)
- $\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}} = \mathcal{O}(1)$ (LayerNorm雅可比)
- $\phi(\boldsymbol{x}_l \boldsymbol{U}_1) \boldsymbol{U}_2 = \mathcal{O}(1)$ (正交性)

因此：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda_1} = \mathcal{O}\left(\frac{\lambda_2}{\alpha}\right)
\tag{35}
\end{equation}

同理：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda_2} = \mathcal{O}\left(\frac{\lambda_1}{\alpha}\right)
\tag{36}
\end{equation}

### 七、Self-Attention的梯度量级分析

#### 7.1 单头注意力的量级分解

\begin{equation}
\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \sigma(\boldsymbol{x}_l \boldsymbol{W}_q \boldsymbol{W}_k^\top \boldsymbol{x}_l^\top) \boldsymbol{x}_l \boldsymbol{W}_v \boldsymbol{W}_o / \alpha)
\tag{37}
\end{equation}

其中 $\sigma$ 表示softmax操作。

分解后：

\begin{equation}
\boldsymbol{z}_{l+1} = \boldsymbol{x}_l + \lambda_v \lambda_o \sigma(\lambda_q \lambda_k \boldsymbol{x}_l \boldsymbol{U}_q \boldsymbol{U}_k^\top \boldsymbol{x}_l^\top) \boldsymbol{x}_l \boldsymbol{U}_v \boldsymbol{U}_o / \alpha
\tag{38}
\end{equation}

#### 7.2 Softmax的量级不变性

**关键性质**: Softmax对输入的缩放不敏感（到某种程度）。

\begin{equation}
\sigma(c\boldsymbol{x}) = \sigma(\boldsymbol{x}), \quad \text{当缩放被温度吸收}
\tag{39}
\end{equation}

在Attention中，$\lambda_q \lambda_k$ 的缩放效果类似于调整温度，但softmax后的加权平均仍保持 $\mathcal{O}(1)$ 量级。

**数学直觉**:

\begin{equation}
\sum_{i} \text{softmax}(\boldsymbol{a})_i \boldsymbol{v}_i \approx \mathcal{O}(\|\boldsymbol{v}\|)
\tag{40}
\end{equation}

由于 $\boldsymbol{v}_i = \boldsymbol{x}_i \boldsymbol{U}_v$ 且 $\boldsymbol{U}_v$ 正交，有 $\|\boldsymbol{v}\| = \mathcal{O}(\|\boldsymbol{x}\|) = \mathcal{O}(1)$ (经过LayerNorm)。

#### 7.3 V和O的梯度

对 $\lambda_v$ 的梯度：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda_v} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}} \frac{\lambda_o \sigma(\cdots) \boldsymbol{x}_l \boldsymbol{U}_v \boldsymbol{U}_o}{\alpha} = \mathcal{O}\left(\frac{\lambda_o}{\alpha}\right)
\tag{41}
\end{equation}

对 $\lambda_o$ 的梯度：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda_o} = \mathcal{O}\left(\frac{\lambda_v}{\alpha}\right)
\tag{42}
\end{equation}

**统一结论**: 无论FFN还是Self-Attention：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \lambda} = \mathcal{O}\left(\frac{\lambda}{\alpha}\right)
\tag{43}
\end{equation}

### 八、深度归一化方案的推导

#### 8.1 抑制增量爆炸的目标

回顾增量爆炸：每个参数的梯度为 $\mathcal{O}(\lambda/\alpha)$，$N$ 层网络共 $2NK$ 个参数（每层K个Self-Attention参数，K个FFN参数）。

总梯度范数平方：

\begin{equation}
\|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|^2 = 2NK \cdot \mathcal{O}\left(\frac{\lambda^2}{\alpha^2}\right)
\tag{44}
\end{equation}

SGD的更新量：

\begin{equation}
\Delta \mathcal{L} = -\eta \|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|^2 = \mathcal{O}\left(\eta \cdot 2NK \cdot \frac{\lambda^2}{\alpha^2}\right)
\tag{45}
\end{equation}

**目标**: 使更新量与 $N$ 无关，即：

\begin{equation}
\frac{\lambda^2}{\alpha^2} = \mathcal{O}\left(\frac{1}{N}\right)
\tag{46}
\end{equation}

#### 8.2 三种优化器的分析

**SGD**: $\Delta \boldsymbol{\theta} = -\eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$

\begin{equation}
\Delta \mathcal{L} \propto -\eta \|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|^2 = \mathcal{O}\left(\eta NK \frac{\lambda^2}{\alpha^2}\right)
\tag{47}
\end{equation}

要求 $\frac{\lambda^2}{\alpha^2} = \frac{1}{2NK}$，即：

\begin{equation}
\frac{\lambda}{\alpha} = \frac{1}{\sqrt{2NK}} = \frac{1}{\sqrt{2N}} \quad (\text{假设} K \approx 1)
\tag{48}
\end{equation}

**Adam**: $\Delta \boldsymbol{\theta} \approx -\eta \cdot \text{sign}(\nabla_{\boldsymbol{\theta}} \mathcal{L})$

\begin{equation}
\Delta \mathcal{L} \propto -\eta \|\nabla_{\boldsymbol{\theta}} \mathcal{L}\|_1 = \mathcal{O}\left(\eta \sqrt{NK} \cdot \frac{\lambda}{\alpha}\right)
\tag{49}
\end{equation}

要求 $\frac{\lambda}{\alpha} = \frac{1}{\sqrt{NK}} = \frac{1}{\sqrt{N}}$。

**LAMB**: $\Delta \boldsymbol{\theta} \approx -\eta \|\boldsymbol{\theta}\| \cdot \text{sign}(\nabla_{\boldsymbol{\theta}} \mathcal{L})$

\begin{equation}
\Delta \mathcal{L} \propto -\eta \lambda \sqrt{NK} \cdot \frac{\lambda}{\alpha} = \mathcal{O}\left(\eta \sqrt{NK} \cdot \frac{\lambda^2}{\alpha}\right)
\tag{50}
\end{equation}

要求 $\frac{\lambda^2}{\alpha} = \frac{1}{\sqrt{NK}}$，即 $\lambda = 1, \alpha = \sqrt{N}$ (选择一种简单解)。

#### 8.3 $\alpha$ 和 $\lambda$ 的最优选择

**对称性原则**: 让 $\lambda = 1/\alpha$，使得残差分支和主路径在初始阶段平衡。

对于SGD:

\begin{equation}
\frac{\lambda}{\alpha} = \frac{1}{\sqrt{2N}} \quad \text{且} \quad \lambda = \frac{1}{\alpha}
\tag{51}
\end{equation}

解得：

\begin{equation}
\alpha = (2N)^{1/4}, \quad \lambda = (2N)^{-1/4}
\tag{52}
\end{equation}

对于Adam:

\begin{equation}
\alpha = N^{1/2}, \quad \lambda = N^{-1/2}
\tag{53}
\end{equation}

对于LAMB:

\begin{equation}
\alpha = 1, \quad \lambda = N^{-1/2}
\tag{54}
\end{equation}

### 九、初始化对训练的影响

#### 9.1 残差分支的初始权重

DeepNorm设置 $\lambda = (2N)^{-1/4}$ 意味着每个残差分支的初始贡献为：

\begin{equation}
\frac{\lambda^2}{\alpha} = \frac{(2N)^{-1/2}}{(2N)^{1/4}} = (2N)^{-3/4}
\tag{55}
\end{equation}

**数值示例**: $N=1000$ 时：

\begin{equation}
\frac{\lambda^2}{\alpha} = (2000)^{-3/4} \approx 0.0042
\tag{56}
\end{equation}

初始阶段每个残差分支仅贡献 $0.4\%$ 的变化，模型接近恒等函数！

#### 9.2 恒等初始化的重要性

**定理**: 如果初始阶段 $\boldsymbol{x}_{l+1} \approx \boldsymbol{x}_l$（恒等函数），则：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \approx \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l}
\tag{57}
\end{equation}

梯度几乎无衰减地传播到底层。

**证明**: 由链式法则：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{x}_l}
\tag{58}
\end{equation}

当 $\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + \epsilon F(\boldsymbol{x}_l)$ 且 $\epsilon \to 0$:

\begin{equation}
\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{x}_l} \approx \boldsymbol{I}
\tag{59}
\end{equation}

因此 $\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l} \approx \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$。□

### 十、Warmup的数学必要性

#### 10.1 学习率warmup的目的

在训练初期，即使有DeepNorm，参数的量级仍需要时间来稳定。Warmup通过逐渐增大学习率来适应这个过程。

**线性warmup**:

\begin{equation}
\eta(t) = \begin{cases}
\frac{t}{T_{\text{warmup}}} \eta_{\text{max}}, & t \leq T_{\text{warmup}} \\
\eta_{\text{max}}, & t > T_{\text{warmup}}
\end{cases}
\tag{60}
\end{equation}

#### 10.2 Warmup与增量爆炸的关系

即使梯度被归一化到 $\mathcal{O}(1/\sqrt{N})$，初始的大步长仍可能导致参数偏离最优初始化。

**自适应策略**: Warmup相当于动态调整 $\eta$ 使得：

\begin{equation}
\eta(t) \cdot \|\nabla_{\boldsymbol{\theta}} \mathcal{L}\| \approx \text{constant}
\tag{61}
\end{equation}

在初期梯度较大时用较小的 $\eta$，避免过度更新。

#### 10.3 数值示例

假设 $N=100$，使用Adam：
- 无warmup: 直接用 $\eta = 10^{-4}$ 可能导致初期震荡
- 有warmup: 前1000步从 $\eta = 10^{-7}$ 线性增加到 $10^{-4}$

初期更新量：

\begin{equation}
\Delta \boldsymbol{\theta}_{\text{warmup}} = \frac{t}{1000} \cdot 10^{-4} \cdot \text{sign}(\nabla) \approx 10^{-7} \text{sign}(\nabla) \quad (t \ll 1000)
\tag{62}
\end{equation}

相比无warmup减少了1000倍，让参数有时间"适应"。

### 十一、Pre-LN vs Post-LN的收敛性分析

#### 11.1 Pre-LN的梯度传播

Pre-LN: $\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + F(\text{LN}(\boldsymbol{x}_l))$

梯度：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \left[\boldsymbol{I} + \mathcal{O}(\lambda)\right]
\tag{63}
\end{equation}

跨 $N$ 层：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_1} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_N} \prod_{l=1}^{N-1} [\boldsymbol{I} + \mathcal{O}(\lambda)]
\tag{64}
\end{equation}

**范数估计**:

\begin{equation}
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_1}\right\| \approx \left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_N}\right\| (1 + \mathcal{O}(\lambda))^N
\tag{65}
\end{equation}

当 $\lambda$ 很小时，$(1 + \mathcal{O}(\lambda))^N \approx 1$，梯度传播稳定。

#### 11.2 Post-LN的梯度传播

Post-LN: $\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + F(\boldsymbol{x}_l))$

梯度包含LayerNorm的雅可比：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}} \cdot \boldsymbol{J}_{\text{LN}} \cdot [\boldsymbol{I} + \mathcal{O}(\lambda)]
\tag{66}
\end{equation}

**问题**: $\boldsymbol{J}_{\text{LN}}$ 会重新归一化，可能破坏梯度的恒等传播。

#### 11.3 DeepNorm的优化

DeepNorm在Post-LN基础上添加 $\alpha$ 缩放：

\begin{equation}
\boldsymbol{x}_{l+1} = \text{LN}(\alpha \boldsymbol{x}_l + F(\boldsymbol{x}_l))
\tag{67}
\end{equation}

当 $\alpha > 1$ 时，主路径被放大，减轻LayerNorm的破坏作用。

**最优 $\alpha$**: 使得主路径在LayerNorm后仍占主导：

\begin{equation}
\alpha = (2N)^{1/4}
\tag{68}
\end{equation}

### 十二、实践建议与数值验证

#### 12.1 不同深度的配置建议

| 层数 $N$ | 优化器 | $\alpha$ | $\lambda$ | Warmup步数 |
|----------|--------|----------|-----------|------------|
| 12       | Adam   | 2.5      | 0.4       | 1000       |
| 24       | Adam   | 3.5      | 0.29      | 2000       |
| 100      | Adam   | 7.1      | 0.14      | 4000       |
| 200      | Adam   | 10       | 0.1       | 8000       |
| 1000     | Adam   | 22.4     | 0.045     | 16000      |

#### 12.2 训练稳定性指标

**梯度范数监控**:

\begin{equation}
G(l) = \left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_l}\right\|, \quad l = 1, 2, \ldots, N
\tag{69}
\end{equation}

健康训练应满足：

\begin{equation}
0.1 \leq \frac{G(l)}{G(N)} \leq 10, \quad \forall l
\tag{70}
\end{equation}

**参数更新比例**:

\begin{equation}
R = \frac{\|\Delta \boldsymbol{\theta}\|}{\|\boldsymbol{\theta}\|} \approx 10^{-3} \sim 10^{-2}
\tag{71}
\end{equation}

#### 12.3 数值示例：1000层Transformer

配置：
- $d = 1024$, $h = 8$
- Adam优化器
- $\alpha = 22.4$, $\lambda = 0.045$
- 学习率 $10^{-4}$ (峰值)
- Warmup 16000步

初始梯度范数（第1层 vs 第1000层）：

\begin{equation}
\frac{G(1)}{G(1000)} = (1.045)^{1000} \approx 2.1
\tag{72}
\end{equation}

在可接受范围内！

### 十三、总结

本节详细推导了深层Transformer训练的数学原理：

**核心问题**:
1. 梯度消失/爆炸: $\|\nabla\| \propto \gamma^N$ 或 $\beta^N$
2. 增量爆炸: $\Delta \mathcal{L} \propto \eta N$

**解决方案**:
1. 残差连接: 确保 $\boldsymbol{J}_l \succeq \boldsymbol{I}$
2. LayerNorm: 归一化梯度到 $\mathcal{O}(1)$
3. DeepNorm: 通过 $\alpha, \lambda$ 缩放抑制增量爆炸
4. Warmup: 初期小学习率适应参数

**关键公式**:
\begin{equation}
\begin{cases}
\alpha = (2N)^{1/4}, \lambda = (2N)^{-1/4} & \text{(SGD)} \\
\alpha = N^{1/2}, \lambda = N^{-1/2} & \text{(Adam)} \\
\alpha = 1, \lambda = N^{-1/2} & \text{(LAMB)}
\end{cases}
\tag{73}
\end{equation}

这些理论保证了1000层甚至更深的Transformer能够稳定训练，打破了"浅而宽"的局限。

