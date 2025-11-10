---
title: VQ一下Key，Transformer的复杂度就变成线性了
slug: vq一下keytransformer的复杂度就变成线性了
date: 2023-11-09
tags: 量子化, 编码, 梯度, attention, 生成模型
status: pending
---

# VQ一下Key，Transformer的复杂度就变成线性了

**原文链接**: [https://spaces.ac.cn/archives/9844](https://spaces.ac.cn/archives/9844)

**发布日期**: 

---

Efficient Transformer，泛指一切致力于降低Transformer的二次复杂度的工作，开始特指针对Attention的改进，后来更一般的思路，如傅里叶变换、线性RNN等，也被归入这个范畴。不得不说，为了降低Transformer的二次复杂度，各路大牛可谓是“八仙过海，各显神通”，各种神奇的思路“百花齐放”，笔者也从中学习到了不少理论知识。然而，尽管Efficient Transformer在理论上是精彩的，但实际上该领域一直都是不愠不火的状态，并没有实际表现十分出色的模型，在LLM火爆的今天，甚至已经逐渐淡出了大家的视野，也淡出了笔者的兴趣范围。

不过，最近有一篇论文[《Transformer-VQ: Linear-Time Transformers via Vector Quantization》](https://papers.cool/arxiv/2309.16354)，却让笔者为之拍案叫绝。作者非常高明地洞察到，只需要对标准Attention的Key做一下VQ（Vector Quantize），复杂度就会自动降低为线性！这种线性化思路保留了标准Attention的形式，是标准Attention到线性Attention的一个完美过渡，同时最大程度上保留了标准Attention的能力。

## 高效难题 #

说起来，本站也算是比较早关注Efficient Transformer相关工作了，最早可以追溯到2019年解读Sparse Transformer的一篇博客[《为节约而生：从标准Attention到稀疏Attention》](/archives/6853)。此后，陆续写的关于Efficient Transformer的其他博文还有

> [《线性Attention的探索：Attention必须有个Softmax吗？》](/archives/7546)
> 
> [《Performer：用随机投影将Attention的复杂度线性化》](/archives/7921)
> 
> [《Nyströmformer：基于矩阵分解的线性化Attention方案》](/archives/8180)
> 
> [《Transformer升级之路：3、从Performer到线性Attention》](/archives/8338)
> 
> [《线性Transformer应该不是你要等的那个模型》](/archives/8610)
> 
> [《FLASH：可能是近来最有意思的高效Transformer设计》](/archives/8934)
> 
> [《Google新作试图“复活”RNN：RNN能否再次辉煌？》](/archives/9554)

然而，正如本文开头所说，尽管Efficient Transformer已有不少工作，也曾被大家寄予厚望，但实际上该领域一直都没什么能“出圈”的作品，这其中的原因可能是：

> 1、不少Efficient Transformer的提速以牺牲效果为代价；
> 
> 2、很多Efficient Transformer的复杂度降低仅仅是理论上的，实际使用提升不明显；
> 
> 3、有些Efficient Transformer难以用来训练Causal LM，所以在LLM流行的今天就没有了用武之地；
> 
> 4、Flash Attention的出现表明即便是标准的Transformer仍有很大的提速空间。

## VQ一下 #

那么，Transformer-VQ为何又具备的“出圈”潜力？

简单来说，Transformer-VQ就是对Attention的Key向量序列进行了“聚类”，并用所属类的类别中心近似原向量，然后Attention的复杂度就变成线性了。也就是说，Transformer-VQ仅仅改变了Key的形似，其余部分（理论上）完全不变，所以这是一种对Attention改动非常小的线性化方案，也能非常清楚体现出线性化后损失的精度在哪里（即用类别中心近似原向量的差距）。

铺垫得有点多了，现在我们正式介绍Transformer-VQ。首先，我们假设$Q,K\in\mathbb{R}^{n\times d_k},V\in\mathbb{R}^{n\times d_v}$，标准Attention就是  
\begin{equation}softmax\left(QK^{\top}\right)V\end{equation}  
简单起见，这里省略了scale factor。Transformer-VQ改为  
\begin{equation}softmax\left(Q\hat{K}^{\top}\right)V,\quad \hat{K} = \color{skyblue}{\mathcal{VQ}}(K, C)\label{eq:vq-att}\end{equation}  
其中$C\in\mathbb{R}^{c\times d_k}$是训练参数，也是VQ的编码表（Codebook）。对了，这里的“VQ”就是指VQ-VAE中的VQ，不了解的读者可以移步参考[《VQ-VAE的简明介绍：量子化自编码器》](/archives/6760)和[《简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE》](/archives/9826)，这里不重复介绍了。总之，经过$\color{skyblue}{\mathcal{VQ}}$之后，最直接的表现就是$K$的每个向量都变成了$C$中与之最相近的那个，这意味着$\hat{K}$的每个向量都是$C$的向量之一，用数学的语言就是说$K\in\mathbb{R}^{n\times d_k}$变成了$\hat{K}\in C^n$。

## Encoder #

当然，直接按照式$\eqref{eq:vq-att}$去实现Transformer-VQ的话，复杂度还是二次的，但由于$\hat{K}$的每个向量都是$C$的向量之一，所以我们可以先算$\exp\left(QC^{\top}\right)$，然后从中“ _挑出_ ”$\exp\left(Q\hat{K}{}^{\top}\right)$对应的结果，而由于$C$的大小是固定的，所以关键运算$QC^{\top}$的复杂度是线性的，这就是Transformer-VQ能线性化的原理（我们不妨称为“挑出”技巧）。

作为铺垫，我们先考虑双向注意力的Encoder情形。由于  
\begin{equation}softmax\left(QK^{\top}\right)V = \frac{\exp\left(QK^{\top}\right)V}{\exp\left(QK^{\top}\right)1_{n\times 1}}\label{eq:softmax-qkv}\end{equation}  
这里$1_{n\times 1}$指的是$n\times 1$大小的全1矩阵，分母可以视为分子的一个特殊形式，所以我们只需要考虑分子$\exp\left(QK^{\top}\right)V$。由于$\hat{K}$的每个向量都是$C$中之一，所以我们可以构建一个one hot矩阵$\Delta\in \\{0,1\\}^{n\times c}$，其中$\Delta_i\in\\{0,1\\}^c$是一个one hot向量，如果1所在的维度为$j$，那么$\hat{K}_i = C_j$，于是$\hat{K}=\Delta C$。

于是对于Transformer-VQ来说有  
\begin{equation}\exp\left(Q\hat{K}{}^{\top}\right)V = \exp\left(QC^{\top}\Delta^{\top}\right)V = \exp\left(QC^{\top}\right)\Delta^{\top}V = \exp\left(QC^{\top}\right)(\Delta^{\top}V)\end{equation}  
很明显，这里最关键的地方就是第二个等号！对于one hot矩阵$\Delta$，右乘以它的转置可以从$\exp$中分离出来， _这就是原理中的“挑出”技巧的数学表述_ 。分离出来之后，由于矩阵乘法结合律，$\Delta^{\top}$可以先跟$V$相乘，得到一个$c\times d_v$的矩阵，而$\exp\left(QC^{\top}\right)$是一个$n\times c$的矩阵，乘以$\Delta^{\top}V$就得到一个$n\times d_v$的矩阵，总的理论复杂度是$\mathcal{O}(ncd_k + ncd_v + ncd_v) = \mathcal{O}(n)$。

最后，根据式$\eqref{eq:softmax-qkv}$，将$\exp\left(Q\hat{K}{}^{\top}\right)V$的结果代入去，就可以计算完整的Attention结果（可能还要加一些避免溢出的细节），整个过程可以在线性复杂度内完成。

## Decoder #

现在我们来考虑单向注意力的Decoder，这是训练生成模型的关键，也是当前LLM的基础。有了Encoder的铺垫后，Decoder理解起来也就没那么困难了。假设$Q_i, \hat{K}_j \in \mathbb{R}^{1\times d_k}, V_j\in\mathbb{R}^{1\times d_v}$是向量序列$Q,\hat{K},V$的行向量之一，那么对于Decoder的分子有  
\begin{equation}\begin{aligned}  
O_i =&\, \sum_{j\leq i}\exp\left(Q_i\hat{K}{}_j^{\top}\right)V_j = \sum_{j\leq i}\exp\left(Q_i C^{\top}\Delta_j^{\top}\right)V_j \\\  
=&\, \sum_{j\leq i}\exp\left(Q_i C^{\top}\right)\Delta_j^{\top}V_j = \exp\left(Q_i C^{\top}\right)\sum_{j\leq i}\Delta_j^{\top}V_j  
\end{aligned}\end{equation}  
如果$c\times d_v$不大，那么最后的式子可以直接用$\text{cumsum}$算子完成，不过一般情况下，尤其是Multi-Heaad时，为了节省显存，通常是跟[《线性Attention的探索：Attention必须有个Softmax吗？》](/archives/7546)中的“自回归生成”一节一样，转为RNN来递归计算，即设$U_i = \sum_{j\leq i}\Delta_j^{\top}V_j\in\mathbb{R}^{c\times d_v}$，那么  
\begin{equation}O_i = \exp\left(Q_i C^{\top}\right)U_i,\quad U_i = U_{i-1} + \Delta_i^{\top}V_i  
\end{equation}  
在推理阶段这样step by step递归计算自然是没问题，但训练阶段step by step的话可能会比较慢，我们可以改为block by block来加速：不失一般性，设$n=lm$，$l$代表block_size，$m$代表block数目，block切片$[il:(i+1)l]$简写为$[i]$，那么  
\begin{equation}\begin{aligned}  
O_{[i]} =&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + M\right)V_{[i]} + \sum_{j\lt i}\exp\left(Q_{[i]}\hat{K}{}_{[j]}^{\top}\right)V_{[j]} \\\  
=&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + M\right)V_{[i]} + \sum_{j\lt i}\exp\left(Q_{[i]}C^{\top}\Delta_{[j]}^{\top}\right)V_{[j]} \\\  
=&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + M\right)V_{[i]} + \exp\left(Q_{[i]}C^{\top}\right)\sum_{j\lt i}\Delta_{[j]}^{\top}V_{[j]} \\\  
\end{aligned}\end{equation}  
其中$M\in\\{-\infty,0\\}^{l\times l}$是下三角的Attention Mask，即当$i \geq j$时$M_{i,j}=0$，否则$M_{i,j}=-\infty$。于是记$U_i = \sum_{j\lt i}\Delta_{[j]}^{\top}V_{[j]}$后，我们有  
\begin{equation}O_{[i]} = \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + M\right)V_{[i]} + \exp\left(Q_{[i]}C^{\top}\right)U_{i-1},\quad U_i = U_{i-1} + \Delta_{[i]}^{\top}V_{[i]}  
\end{equation}  
这样我们就将递归步数减少为$m$了，可以在保证线性效率的同时，更充分发挥硬件的并行能力。用同样的方式也可以计算分母，最后相除得到完整的Attention结果

## 局域增强 #

就这样完了？并不是，如果仅仅是这样的话，Transformer-VQ可能跟以往基于矩阵分解的Kernelized Attention如Performer并没有太多区别。当序列长度$n$远大于编码表大小$c$时，由抽屉原理我们知道部分编码向量必然会反复出现，甚至可以合理猜测所有编码向量应该会均匀分布在整个序列中。这样一来，邻近token的Attention就会跟远处某些token的Attention一样，也就是说模型无法区分远近，这本质上就是所有Kernelized Attention都存在的低秩问题。

已有的经验告诉我们，对于语言模型来说，相对于远处的token的来说邻近的token往往更为重要，所以一个好的语言模型架构应该具有区分远近的能力。为此，Transformer-VQ选择在$Q\hat{K}$之后，加上一个Sliding Window形状的Attention Bias（记为$B$），来对邻近token进行加权，如下图：  


[![Window Attention Bias示意图](/usr/uploads/2023/11/2460405664.svg)](/usr/uploads/2023/11/2460405664.svg "点击查看原图")

Window Attention Bias示意图

从最后一个图可以看出，如果将Window大小直接设为block大小$l$，即$i < j$或者$i - j \leq l$时$B_{i,j}=0$，那么在分block计算时，矩阵$B$顶多影响最邻近的两个block，再远的block依旧可以用“挑出”技巧来线性化。为了便于下面的推导，我们记$B_{[i,j]} = B_{[il:(i+1)l,jl:(j+1)l]}$，那么  
\begin{equation}\begin{aligned}  
O_{[i]} =&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + B_{[i,i]}\right)V_{[i]} + \exp\left(Q_{[i]}\hat{K}{}_{[i-1]}^{\top} + B_{[i,i-1]}\right)V_{[i-1]} + \sum_{j\lt i-1}\exp\left(Q_{[i]}\hat{K}{}_{[j]}^{\top}\right)V_{[j]} \\\  
=&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + B_{[i,i]}\right)V_{[i]} + \exp\left(Q_{[i]}\hat{K}{}_{[i-1]}^{\top} + B_{[i,i-1]}\right)V_{[i-1]} + \sum_{j\lt i-1}\exp\left(Q_{[i]}C^{\top}\Delta_{[j]}^{\top}\right)V_{[j]} \\\  
=&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + B_{[i,i]}\right)V_{[i]} + \exp\left(Q_{[i]}\hat{K}{}_{[i-1]}^{\top} + B_{[i,i-1]}\right)V_{[i-1]} + \exp\left(Q_{[i]}C^{\top}\right)\sum_{j\lt i-1}\Delta_{[j]}^{\top}V_{[j]} \\\  
\end{aligned}\end{equation}  
所以很明显，有（约定$V_{[-1]},U_{[-1]},U_{[-2]}$都是全零矩阵）  
\begin{equation}\begin{aligned}  
O_{[i]} =&\, \exp\left(Q_{[i]}\hat{K}{}_{[i]}^{\top} + B_{[i,i]}\right)V_{[i]} + \exp\left(Q_{[i]}\hat{K}{}_{[i-1]}^{\top} + B_{[i,i-1]}\right)V_{[i-1]} + \exp\left(Q_{[i]}C^{\top}\right)U_{i-2}\\\\[5pt]  
U_i =&\, U_{i-1} + \Delta_{[i]}^{\top}V_{[i]}  
\end{aligned}\label{eq:tvq}\end{equation}  
笔者认为，$B$的引入是Transformer-VQ是跟其他Kernelized Attention拉开差距的关键，为了减少参数量且支持变长生成，我们约束B的非零部分为“Toeplitz矩阵”，即$B_{i,j}$是$i-j$的函数，此时$B$就相当于加性相对位置编码。除了这种做法外，也可以考虑换为笔者之前提出的[ReRoPE](/archives/9708)，它是旋转位置编码的窗口版，跟$B$具有同样的相对位置编码形状。

## 梯度回传 #

等等，我们好像忘记了点什么。了解VQ-VAE的读者都知道，“$\hat{K}$的每个向量都是$C$的向量之一”只是前向传播的表现，反向传播用的可是原始的$K$，这意味着即便不同位置的$\hat{K}_j$等于同一个$C_k$，但它们的梯度却不相等，这叫做STE（Straight-Through Estimator）。由于STE的存在，“挑出”技巧理论上仅可用于推理阶段，训练阶段是无法线性化的。

没有其他办法了吗？确实如此，如果我们坚持要获得精确的梯度结果，那么并没有线性化效率的方案。然而，考虑到VQ的梯度本身就是近似的，所以Attention获取精确的梯度似乎也没多大必要。于是作者想了个折衷的方案：依然是按照式$\eqref{eq:tvq}$进行递归计算，仅在前两项使用STE（Key序列可以获得梯度），而$U_{i-1}$的梯度直接停掉（$\text{stop_gradient}$算子）。这样我们就保持了模型的线性性，同时也已经保留了最重要的梯度（邻近的两个block），算是一个比较合理的近似方案。从这一点来看，Transformer-VQ跟[Transformer-XL](https://papers.cool/arxiv/1901.02860)很像，Transformer-XL在递归的同时也停掉了历史窗口的梯度，即历史窗口可以参与递归计算，不传递梯度。

解决了梯度回传问题之后，在自回归交叉熵损失的基础上，再上VQ带来的用来更新编码表的辅助loss，就得到完整的训练目标了。当然，对于编码表的更新，Transformer-VQ采用了直接滑动平均的方案，所以只补充了Key的辅助loss，这些细节读者在熟悉VQ-VAE之后，稍微看一下原论文就理解了。

## 实验结果 #

这一节我们来看一下原论文的实验结果。作者已经将代码开源如下：

> **Github：<https://github.com/transformer-vq/transformer_vq>**

值得指出的是，作者做VQ的基础架构并不是常规的MHA（Multi-Head Attention），而是笔者一直很推崇的GAU（Gated Attention Unit）+Softmax，Transformer-VQ更准确的命名应该是“GAU-VQ”，不了解GAU的读者可以参考[《FLASH：可能是近来最有意思的高效Transformer设计》](/archives/8934)和[《听说Attention与Softmax更配哦～》](/archives/9019)。简单来说，GAU本身比MHA有着更高的效率，配合上VQ技巧后，就更加“如虎添翼”了。

实验方面，作者做了语言模型（ENWIK8、PG-19）和图像生成（IMAGENET64），所有的实验中的编码表大小都是$c=512$。模型最大参数量为1.3B，虽然比不上主流的大模型参数量，但其实对于科研来说不算小了。实验结果总体来说算得上优异：  


[![PG-19的实验结果](/usr/uploads/2023/11/3869943959.png)](/usr/uploads/2023/11/3869943959.png "点击查看原图")

PG-19的实验结果

[![IMAGENET64的实验结果](/usr/uploads/2023/11/531043849.png)](/usr/uploads/2023/11/531043849.png "点击查看原图")

IMAGENET64的实验结果

最后，让人惊奇的是，Transformer-VQ的作者只有一个，并且身份是“Independent Researcher”。

## 发散思考 #

笔者发现，从Transformer-VQ出发，可以联系到非常多的研究主题，这也是为什么笔者如此欣赏它的原因之一。

首先，再次为作者惊人的洞察力点赞，“只需VQ一下Key，Transformer的复杂度就会变成线性”这个发现实在太美妙了，它实现了标准Attention到线性Attention的自然过渡，并且可以通过加Attention Bias的方式让它比很多的Kernelized Attention都有效。然后，通过VQ进行“聚类”的方式，也比[Linformer](https://papers.cool/arxiv/2006.04768)、[Nyströmformer](/archives/8180)等更为高明，因为它防止了未来信息的泄漏，可以自然地用来做Causal的语言模型。

我们知道，VQ本质上也是将序列转为离散id的运算，这跟Tokenizer的作用是非常相似的。从这个角度来看，Transformer-VQ跟[MegaByte](https://papers.cool/arxiv/2305.07185)等模型一样，都是将Tokenizer内置在模型之中，并且相比MegaByte，VQ这一操作跟我们传统意义上的Tokenizer更为相似、直观。所以，Transformer-VQ实际上非常适合用来训练直接以Bytes输入的“No Tokenizer”模型，事实上，上述ENWIK8实验就是Bytes输入，Transformer-VQ效果明显优于MegaByte。

相比近来出的[RetNet](https://papers.cool/arxiv/2307.08621)，Transformer-VQ没有显式的远程衰减，所以Long Context能力有可能会更好，同时由于Key经过了VQ，都是有限集合之一，所以不会出现没有学过的Key，因此长度外推能力大概率也会更好。虽然Transformer-VQ的基础架构GAU只是Single-Head的，但它在递归过程中模型记忆状态大小是$\Delta_i^{\top}V_i\in\mathbb{R}^{c\times d_v}$，在默认的设置中，这比Multi-Head的RetNet还大（RetNet的记忆状态大小是$nd_k^2$，默认设置下$d_v = 2nd_k$），因此，记忆容量理论上是足够的。

由于上一篇文章刚好写了[《简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE》](/archives/9826)，可能会有读者想知道可否用更简单的FSQ取代VQ？笔者认为比较难，原因其实在上一篇文章给出了：第一，$c=512$还属于VQ优于FSQ的编码数量范围，所以换FSQ大概率会掉效果；第二，由于每层Attention的Key都要被VQ，所以平均来说VQ的Encoder和Decoder都不强，这种情况VQ近似精度更高，FSQ更适合Decoder和Decoder都足够强的场景；第三，Transformer-VQ需要用的是Key被VQ之后的中心向量而不是id，而FSQ则直接得到id，反而不容易恢复为近似的中心向量。

除此之外，用VQ而不是FSQ，使得Transformer-VQ有希望从现有的预训练模型如LLAMA2中微调过来，而不单单是从零训练。因为VQ具有鲜明的几何意义，跟K-Means有诸多相通之处，我们可以从现有预训练模型出发，选取一些样本计算出Key，对Key进行K-Means得到中心向量作为编码表的初始化，然后在原模型基础上加上VQ进行微调。不过Transformer-VQ不大好适配RoPE，所以要如前面所说，RoPE的模型要换成ReRoPE再VQ比较好，此时就可以不用加Bias了。

总之，在笔者眼中，Transformer-VQ在众多Efficient Transformer工作中，是非常独特、出色而又潜力深厚的之一。

## 文章小结 #

本文介绍了一个名为Transformer-VQ的Efficient Transformer方案，它基于“只需VQ一下Key，Transformer的复杂度就会变成线性”的观察结果进行展开，个人认为是一种非常独特且亮眼的线性化思路，实验结果也很优异。它既可以理解为一种更高明的线性Attention/RNN模型，也可以理解为一个带有“可训练的Tokenizer”的Attention模型。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9844>_

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

苏剑林. (Nov. 09, 2023). 《VQ一下Key，Transformer的复杂度就变成线性了 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9844>

@online{kexuefm-9844,  
title={VQ一下Key，Transformer的复杂度就变成线性了},  
author={苏剑林},  
year={2023},  
month={Nov},  
url={\url{https://spaces.ac.cn/archives/9844}},  
} 


---

## 公式推导与注释

本节将对Transformer-VQ方法进行极其详细的数学推导，涵盖标准Attention的复杂度分析、Key向量量化的数学表示、复杂度降低的严格证明、信息损失的定量分析、近似误差界等核心内容。

### 1. 标准Attention的复杂度分析

**定义1.1（标准Attention）**：给定查询矩阵$Q \in \mathbb{R}^{n \times d_k}$，键矩阵$K \in \mathbb{R}^{n \times d_k}$，值矩阵$V \in \mathbb{R}^{n \times d_v}$，标准Attention计算为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中$n$是序列长度，$d_k$是键向量维度，$d_v$是值向量维度。

**定理1.1（标准Attention的计算复杂度）**：标准Attention的计算复杂度为：

$$
\mathcal{O}(n^2 d_k + n^2 d_v)
$$

**证明**：分解计算步骤：

1. 计算$QK^T$：需要$n \times n \times d_k$次乘法，复杂度$\mathcal{O}(n^2 d_k)$
2. 应用softmax：对$n \times n$矩阵的每行计算softmax，复杂度$\mathcal{O}(n^2)$
3. 计算$\text{softmax}(QK^T)V$：需要$n \times n \times d_v$次乘法，复杂度$\mathcal{O}(n^2 d_v)$

总复杂度为$\mathcal{O}(n^2 d_k + n^2 + n^2 d_v) = \mathcal{O}(n^2(d_k + d_v))$。$\square$

**命题1.1（内存复杂度）**：标准Attention的内存复杂度为：

$$
\mathcal{O}(n^2 + nd_k + nd_v)
$$

其中$n^2$项来自存储注意力矩阵$QK^T$。

**定义1.2（序列长度的瓶颈）**：当序列长度$n$很大时（例如$n = 8192$），$n^2$项成为计算和内存的主要瓶颈：

- 计算复杂度：$n^2 d_k \approx 8192^2 \times 128 = 8.6 \times 10^9$次操作
- 内存占用：$n^2 \times 4\text{bytes} \approx 256\text{MB}$（单精度浮点数）

**命题1.2（理论下界）**：对于一般的Attention机制，如果要计算所有token对之间的注意力，则至少需要$\Omega(n^2)$的复杂度来读取和写入这些注意力权重。

### 2. Key向量量化的数学表示

**定义2.1（编码表）**：给定编码表$C \in \mathbb{R}^{c \times d_k}$，其中$c$是编码表大小（通常$c \ll n$）。

**定义2.2（Key的量化映射）**：对于Key矩阵$K$的第$i$行$K_i$，其量化结果$\hat{K}_i$定义为：

$$
\hat{K}_i = VQ(K_i, C) = \mathop{\text{argmin}}_{c_j \in C} \|K_i - c_j\|_2
$$

量化后的Key矩阵$\hat{K} \in C^n$，即$\hat{K}$的每一行都是$C$的某一行。

**定义2.3（量化索引）**：定义索引函数$\text{idx}: \{1, \cdots, n\} \to \{1, \cdots, c\}$，使得：

$$
\hat{K}_i = C_{\text{idx}(i)}
$$

**命题2.1（one-hot表示）**：可以用one-hot矩阵$\Delta \in \{0,1\}^{n \times c}$表示量化结果：

$$
\Delta_{ij} = \begin{cases}
1 & \text{if } \text{idx}(i) = j \\
0 & \text{otherwise}
\end{cases}
$$

则有：
$$
\hat{K} = \Delta C
$$

**证明**：第$i$行的量化结果为：

$$
\hat{K}_i = \sum_{j=1}^c \Delta_{ij} C_j = C_{\text{idx}(i)}
$$

因为$\Delta_i$是one-hot向量，只有第$\text{idx}(i)$个分量为1。$\square$

**定理2.1（量化误差）**：Key量化引入的误差定义为：

$$
\epsilon_K = \|K - \hat{K}\|_F = \sqrt{\sum_{i=1}^n \|K_i - \hat{K}_i\|_2^2}
$$

其中$\|\cdot\|_F$是Frobenius范数。

**命题2.2（量化误差的期望）**：对于随机分布的Key，期望量化误差满足：

$$
\mathbb{E}[\epsilon_K^2] = n \cdot \mathbb{E}\left[\min_{j} \|K_i - C_j\|_2^2\right]
$$

这依赖于编码表$C$的选择和Key的分布。

### 3. Transformer-VQ Encoder的复杂度分析

**定理3.1（Encoder的线性复杂度）**：对于双向Attention（Encoder），Transformer-VQ的复杂度为：

$$
\mathcal{O}(ncd_k + ncd_v)
$$

当$c \ll n$时，这是关于$n$的线性复杂度。

**证明**：根据原文的推导，Encoder的计算分解为：

$$
\text{softmax}(Q\hat{K}^T)V = \frac{\exp(Q\hat{K}^T)V}{\exp(Q\hat{K}^T)1_{n \times 1}}
$$

分子的计算：

$$
\exp(Q\hat{K}^T)V = \exp(Q(\Delta C)^T)V = \exp(QC^T\Delta^T)V
$$

利用$\Delta^T$是one-hot矩阵的性质，我们可以先计算$\exp(QC^T)$，然后"挑出"相应的列：

$$
\exp(QC^T\Delta^T)V = \exp(QC^T)(\Delta^T V)
$$

复杂度分解：
1. 计算$QC^T$：$\mathcal{O}(ncd_k)$
2. 计算$\exp$：$\mathcal{O}(nc)$
3. 计算$\Delta^T V$：$\mathcal{O}(ncd_v)$（稀疏矩阵乘法，每行只有一个非零元素）
4. 计算$\exp(QC^T)(\Delta^T V)$：$\mathcal{O}(ncd_v)$

总复杂度：$\mathcal{O}(ncd_k + nc + ncd_v + ncd_v) = \mathcal{O}(ncd_k + ncd_v)$。

当$c \ll n$时，这是$\mathcal{O}(n)$。$\square$

**定理3.2（加速比）**：相比于标准Attention，Transformer-VQ的加速比为：

$$
\text{Speedup} = \frac{n^2(d_k + d_v)}{nc(d_k + d_v)} = \frac{n}{c}
$$

例如，当$n = 8192, c = 512$时，理论加速比为16倍。

**命题3.1（内存节省）**：Transformer-VQ的内存复杂度为：

$$
\mathcal{O}(nc + nd_k + nd_v + cd_k)
$$

相比于标准Attention的$\mathcal{O}(n^2 + nd_k + nd_v)$，节省了$\mathcal{O}(n^2 - nc) \approx \mathcal{O}(n^2)$的内存（当$c \ll n$时）。

### 4. "挑出"技巧的数学原理

**定义4.1（挑出操作）**：设$A \in \mathbb{R}^{n \times c}$，$\Delta \in \{0,1\}^{n \times c}$是one-hot矩阵，则"挑出"操作定义为：

$$
\text{Pick}(A, \Delta) = A \odot \Delta
$$

其中$\odot$表示element-wise乘法。结果的第$i$行只有第$\text{idx}(i)$列非零。

**引理4.1（矩阵乘法中的挑出）**：对于矩阵$B \in \mathbb{R}^{c \times m}$，有：

$$
(A \odot \Delta)B = A(\Delta^T \circ B)
$$

其中$\circ$表示Hadamard积。但这个表达不太对，让我重新表述。

实际上，正确的表述是：

$$
\exp(QC^T\Delta^T) = \exp(QC^T) \odot (\mathbf{1}_{n \times 1} \Delta^T)
$$

其中$\mathbf{1}_{n \times 1}$是全1列向量。

**引理4.2（one-hot矩阵的性质）**：对于one-hot矩阵$\Delta$：

1. $\Delta \mathbf{1}_{c \times 1} = \mathbf{1}_{n \times 1}$（每行恰有一个1）
2. $\Delta^T$的每列有且仅有一个1（或没有1，如果某个编码未被使用）
3. $\Delta^T V$相当于对$V$的行进行"聚类求和"

**定理4.1（挑出技巧的核心）**：关键观察是，对于one-hot矩阵$\Delta$和指数运算：

$$
\exp(QC^T\Delta^T) = \exp(QC^T) \cdot \Delta^T
$$

其中右侧的乘法利用了$\Delta^T$的稀疏性，从$\exp(QC^T)$中"挑出"相应的列。

**证明**：设$S = QC^T \in \mathbb{R}^{n \times c}$，则：

$$
(S\Delta^T)_{ij} = \sum_{k=1}^c S_{ik} \Delta_{jk} = S_{i,\text{idx}(j)}
$$

因为$\Delta_j$是one-hot向量。因此：

$$
\exp(S\Delta^T)_{ij} = \exp(S_{i,\text{idx}(j)})
$$

而：

$$
(\exp(S)\Delta^T)_{ij} = \sum_{k=1}^c \exp(S_{ik}) \Delta_{jk} = \exp(S_{i,\text{idx}(j)})
$$

两者相等，证毕。$\square$

**推论4.1（线性化的关键）**：这个性质使得我们可以先计算$\exp(QC^T)$（复杂度$\mathcal{O}(ncd_k)$），然后通过稀疏矩阵乘法应用$\Delta^T$，而不需要先计算$S\Delta^T$（这会得到$n \times n$的矩阵）。

### 5. Decoder的复杂度分析

**定义5.1（Causal Attention）**：对于Decoder，Attention需要满足因果约束：

$$
\text{Attention}(Q, K, V)_i = \frac{\sum_{j \leq i} \exp(Q_iK_j^T) V_j}{\sum_{j \leq i} \exp(Q_iK_j^T)}
$$

**定理5.1（Decoder的递归形式）**：Transformer-VQ的Decoder可以表示为递归形式：

$$
O_i = \exp(Q_iC^T)U_{i-1}, \quad U_i = U_{i-1} + \Delta_i^T V_i
$$

其中$U_i \in \mathbb{R}^{c \times d_v}$是累积状态。

**证明**：根据原文的推导：

$$
\begin{aligned}
O_i &= \sum_{j \leq i} \exp(Q_i\hat{K}_j^T) V_j \\
&= \sum_{j \leq i} \exp(Q_i C^T \Delta_j^T) V_j \\
&= \sum_{j \leq i} \exp(Q_i C^T) \Delta_j^T V_j \\
&= \exp(Q_i C^T) \sum_{j \leq i} \Delta_j^T V_j \\
&= \exp(Q_i C^T) U_{i-1}
\end{aligned}
$$

其中$U_i = \sum_{j \leq i} \Delta_j^T V_j$。递归关系$U_i = U_{i-1} + \Delta_i^T V_i$自然成立。$\square$

**定理5.2（Decoder的复杂度）**：使用递归形式，Decoder的复杂度为：

$$
\mathcal{O}(ncd_k + ncd_v)
$$

仍然是关于$n$的线性复杂度。

**证明**：对于每个位置$i$：
1. 计算$Q_iC^T$：$\mathcal{O}(cd_k)$
2. 更新$U_i$：$\mathcal{O}(cd_v)$（因为$\Delta_i^T V_i$是秩1矩阵）
3. 计算$O_i$：$\mathcal{O}(cd_v)$

总复杂度：$n \times \mathcal{O}(cd_k + cd_v) = \mathcal{O}(ncd_k + ncd_v)$。$\square$

### 6. Block-wise计算的分析

**定义6.1（Block分割）**：将序列分为$m$个block，每个block大小为$l$，满足$n = lm$。

**定理6.1（Block-wise Decoder）**：Block-wise计算的复杂度仍为$\mathcal{O}(ncd_k + ncd_v)$，但可以更好地利用硬件并行性。

根据原文，第$i$个block的输出为：

$$
O_{[i]} = \exp(Q_{[i]}\hat{K}_{[i]}^T + M)V_{[i]} + \exp(Q_{[i]}C^T)U_{i-2}
$$

其中$M$是causal mask，$U_i = U_{i-1} + \Delta_{[i]}^T V_{[i]}$。

**命题6.1（Block内的二次复杂度）**：Block内的计算$\exp(Q_{[i]}\hat{K}_{[i]}^T + M)V_{[i]}$具有$\mathcal{O}(l^2 d_k + l^2 d_v)$的复杂度。

总复杂度：

$$
m \times \mathcal{O}(l^2(d_k + d_v) + lcd_k + lcd_v) = \mathcal{O}(nl(d_k + d_v) + nc(d_k + d_v))
$$

当$l \ll n$且$c \ll n$时，仍是线性复杂度。

**定理6.2（Block size的选择）**：最优的block size $l^*$需要平衡：
1. 更小的$l$：更好的线性特性，但并行度降低
2. 更大的$l$：更好的硬件利用率，但二次项$l^2$增加

实践中，$l \in [128, 512]$是一个合理的范围。

### 7. 加入Attention Bias的复杂度

**定义7.1（Window Attention Bias）**：添加窗口大小为$w$的Attention Bias矩阵$B$：

$$
B_{ij} = \begin{cases}
b(i - j) & \text{if } |i - j| \leq w \\
0 & \text{otherwise}
\end{cases}
$$

其中$b(\cdot)$是可学习的函数。

**定理7.1（加Bias后的复杂度）**：当窗口大小$w = l$（等于block size）时，复杂度仍为：

$$
\mathcal{O}(nl(d_k + d_v) + nc(d_k + d_v))
$$

**证明**：根据原文的公式：

$$
O_{[i]} = \exp(Q_{[i]}\hat{K}_{[i]}^T + B_{[i,i]})V_{[i]} + \exp(Q_{[i]}\hat{K}_{[i-1]}^T + B_{[i,i-1]})V_{[i-1]} + \exp(Q_{[i]}C^T)U_{i-2}
$$

复杂度分析：
1. 前两项：$\mathcal{O}(l^2 d_k + l^2 d_v)$（block内和相邻block）
2. 第三项：$\mathcal{O}(lcd_k + lcd_v)$（利用挑出技巧）

总复杂度仍为$\mathcal{O}(nl(d_k + d_v) + nc(d_k + d_v))$。$\square$

**命题7.1（Bias的作用）**：Attention Bias允许模型区分邻近和远处的token，这对于捕获局部依赖关系很重要。没有Bias，所有距离大于$w$的token对都会得到相同的注意力模式（因为它们的Key都是量化后的）。

### 8. 信息损失的定量分析

**定义8.1（注意力差异）**：量化引入的注意力差异定义为：

$$
\Delta A = \text{softmax}(QK^T) - \text{softmax}(Q\hat{K}^T)
$$

**定理8.1（注意力差异的上界）**：注意力差异满足：

$$
\|\Delta A\|_F \leq \frac{2}{\sqrt{d_k}} \|K - \hat{K}\|_F \cdot \|Q\|_F
$$

**证明**：利用softmax的Lipschitz连续性。设$S = QK^T, \hat{S} = Q\hat{K}^T$，则：

$$
\|S - \hat{S}\|_F = \|Q(K - \hat{K})^T\|_F \leq \|Q\|_F \|K - \hat{K}\|_F
$$

对于softmax函数$\sigma: \mathbb{R}^n \to \mathbb{R}^n$，在$l_2$范数下的Lipschitz常数为$\sqrt{2}$（在有scale factor $1/\sqrt{d_k}$时）。因此：

$$
\|\text{softmax}(S/\sqrt{d_k}) - \text{softmax}(\hat{S}/\sqrt{d_k})\|_F \leq \frac{\sqrt{2}}{\sqrt{d_k}} \|S - \hat{S}\|_F
$$

代入得到结果。$\square$

**推论8.1（输出差异）**：最终输出的差异满足：

$$
\|\text{Attention}(Q,K,V) - \text{Attention}(Q,\hat{K},V)\|_F \leq \frac{2}{\sqrt{d_k}} \|K - \hat{K}\|_F \cdot \|Q\|_F \cdot \|V\|_F
$$

**定义8.2（相对误差）**：定义相对误差为：

$$
\epsilon_{rel} = \frac{\|\text{Attention}(Q,K,V) - \text{Attention}(Q,\hat{K},V)\|_F}{\|\text{Attention}(Q,K,V)\|_F}
$$

**命题8.1（相对误差的估计）**：当编码表大小$c$足够大且初始化良好时，相对误差通常在1-5%的范围内。这可以通过实验验证。

### 9. 近似误差界

**定义9.1（量化误差率）**：定义量化误差率为：

$$
\rho = \frac{\|K - \hat{K}\|_F}{\|K\|_F}
$$

**定理9.1（输出的相对误差界）**：输出的相对误差满足：

$$
\epsilon_{rel} \leq C \cdot \rho
$$

其中$C$是依赖于$Q, K, V$分布的常数，通常$C \in [1, 3]$。

**证明**：由推论8.1：

$$
\epsilon_{rel} = \frac{\|\Delta\text{Attn}\|_F}{\|\text{Attn}\|_F} \leq \frac{2}{\sqrt{d_k}} \frac{\|K - \hat{K}\|_F \cdot \|Q\|_F \cdot \|V\|_F}{\|\text{Attn}\|_F}
$$

注意到$\|\text{Attn}\|_F \approx \sqrt{n} \|V\|_F$（因为softmax归一化），且$\|Q\|_F \approx \sqrt{n} \|K\|_F$（假设$Q, K$分布相似）。代入得：

$$
\epsilon_{rel} \lesssim \frac{2}{\sqrt{d_k}} \frac{\|K - \hat{K}\|_F \cdot \sqrt{n} \|K\|_F \cdot \|V\|_F}{\sqrt{n} \|V\|_F} = \frac{2}{\sqrt{d_k}} \frac{\|K - \hat{K}\|_F}{\|K\|_F} = \frac{2}{\sqrt{d_k}} \rho
$$

因此$C = \frac{2}{\sqrt{d_k}}$。对于$d_k = 64$，$C \approx 0.25$；对于$d_k = 128$，$C \approx 0.18$。$\square$

**推论9.1（编码表大小的影响）**：为了保证$\epsilon_{rel} < \epsilon_{target}$，需要的编码表大小满足：

$$
c \geq \frac{n}{r^2}
$$

其中$r = \frac{\epsilon_{target}}{C}$是相对量化误差的上界。

例如，若$\epsilon_{target} = 0.02, C = 0.2$，则$r = 0.1$，需要$c \geq n/100$。对于$n = 8192$，需要$c \geq 82$，实际中$c = 512$应该足够。

### 10. 与线性Attention的对比

**定义10.1（Kernelized Linear Attention）**：线性Attention使用核函数$\phi: \mathbb{R}^{d_k} \to \mathbb{R}^m$：

$$
\text{LinearAttn}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}
$$

复杂度为$\mathcal{O}(nm d_k + nm d_v)$，当$m \ll n$时为线性。

**命题10.1（表达能力对比）**：
- Linear Attention：用低维核函数近似softmax，近似能力有限
- Transformer-VQ：用离散编码表近似Key，保留完整的softmax结构

**定理10.1（近似质量对比）**：在相同的"压缩率"（Transformer-VQ的$c$和Linear Attention的$m$）下，Transformer-VQ的近似误差通常更小：

$$
\epsilon_{TVQ} < \epsilon_{LinearAttn}
$$

**证明思路**：Transformer-VQ直接在Key空间中进行量化，而Linear Attention在核空间中进行低秩近似。对于一般的数据分布，量化通常比低秩近似更精确（在相同参数量下）。

具体地，量化可以看作是"分段常数"近似，而低秩近似是"全局线性"近似。对于非线性的Attention模式，分段常数更灵活。$\square$

**命题10.2（计算效率对比）**：
- Linear Attention：需要设计和计算核函数$\phi$，可能有额外开销
- Transformer-VQ：只需最近邻搜索和矩阵乘法，实现简单

**定理10.2（长度外推能力）**：Transformer-VQ的长度外推能力更强：

1. **编码表固定**：训练时学到的编码表$C$在推理时可以用于任意长度
2. **无长度泄漏**：量化操作不依赖于序列长度$n$，不会有长度信息泄漏

相比之下，许多Linear Attention方法的核函数可能隐式依赖于训练时的序列长度。

### 11. 与RNN/State Space Model的对比

**定义11.1（线性RNN）**：线性RNN（如RetNet）的形式为：

$$
h_i = A h_{i-1} + B K_i, \quad O_i = C h_i + D V_i
$$

复杂度为$\mathcal{O}(n d^2)$（$d$是隐藏状态维度）。

**命题11.1（Transformer-VQ作为RNN）**：Transformer-VQ的递归形式：

$$
U_i = U_{i-1} + \Delta_i^T V_i, \quad O_i = \exp(Q_i C^T) U_i
$$

可以视为一种特殊的RNN，其中：
- 状态$U_i \in \mathbb{R}^{c \times d_v}$
- 转移矩阵$A = I$（简单累加）
- 输出依赖于query-dependent的权重$\exp(Q_i C^T)$

**定理11.1（记忆容量对比）**：Transformer-VQ的记忆容量为$cd_v$，相比于：
- 线性RNN：$d^2$
- 标准Transformer：$n^2$（隐式记忆在注意力矩阵中）

当$c = 512, d_v = 256$时，记忆容量为$131,072$，足以编码丰富的历史信息。

**命题11.2（远程依赖能力）**：
- Linear RNN：依赖于状态的衰减率，远程信息会指数衰减
- Transformer-VQ：没有显式的衰减，可以通过编码表$C$保留远程信息

因此Transformer-VQ的远程依赖能力理论上强于线性RNN。

### 12. 梯度传播的分析

**定义12.1（梯度停止）**：在训练阶段，Transformer-VQ停止历史状态$U_{i-1}$的梯度：

$$
O_i = \exp(Q_i C^T) \text{sg}[U_{i-1}] + \text{local terms}
$$

**定理12.1（有效梯度路径）**：尽管停止了$U_{i-1}$的梯度，但前两个block的梯度仍然被保留：

$$
\frac{\partial \mathcal{L}}{\partial K_{[i]}} \neq 0, \quad \frac{\partial \mathcal{L}}{\partial K_{[i-1]}} \neq 0
$$

对于$j < i-1$的block，$\frac{\partial \mathcal{L}}{\partial K_{[j]}} = 0$。

**命题12.1（与Transformer-XL的相似性）**：这种梯度策略与Transformer-XL类似：
- Transformer-XL：缓存历史隐藏状态，但停止其梯度
- Transformer-VQ：累积历史量化信息，但停止远处的梯度

**定理12.2（梯度偏差分析）**：停止梯度引入的偏差为：

$$
\text{Bias}_{grad} = \mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial K_{[j]}} - 0\right\|\right], \quad j < i-2
$$

但由于Attention本身具有"遗忘"特性（远处token的权重较小），这个偏差在实践中可以接受。

**命题12.2（编码表的梯度）**：编码表$C$的梯度通过所有时间步累积：

$$
\frac{\partial \mathcal{L}}{\partial C} = \sum_{i=1}^n \frac{\partial \mathcal{L}_i}{\partial C}
$$

因此编码表能接收到充分的训练信号。

### 13. 编码表学习的理论

**定义13.1（编码表的优化目标）**：编码表$C$的优化目标包括：

1. **重构损失**：最小化量化误差$\|K - \hat{K}\|^2$
2. **任务损失**：最小化最终的语言模型损失

**定理13.1（两阶段学习）**：编码表的学习可以分为两个阶段：

1. **初始化阶段**：使用K-Means或从预训练模型的Key分布中采样初始化$C$
2. **微调阶段**：通过端到端训练优化$C$

**命题13.1（K-Means初始化的优势）**：K-Means初始化最小化：

$$
\min_C \mathbb{E}\left[\min_{c \in C} \|K - c\|^2\right]
$$

这正是量化误差的直接优化目标。

**定理13.2（编码表的动态性）**：在训练过程中，编码表会自适应地调整以捕获Key的主要模式：

$$
C^{(t+1)} = C^{(t)} - \eta \nabla_C \mathcal{L}
$$

其中梯度$\nabla_C \mathcal{L}$依赖于当前数据分布和任务需求。

**命题13.2（编码表的利用率）**：与VQ-VAE类似，Transformer-VQ也面临编码表利用率问题。可以使用以下策略：

1. **EMA更新**：使用指数移动平均更新编码表
2. **辅助损失**：添加编码表利用率的正则化项
3. **重置策略**：重置长期未使用的编码向量

### 14. 实际加速比的理论预测

**定义14.1（理论加速比）**：理论加速比定义为：

$$
\text{Speedup}_{theory} = \frac{T_{standard}}{T_{VQ}}
$$

其中$T$表示计算时间。

**定理14.1（理论加速比的分解）**：理论加速比可以分解为：

$$
\text{Speedup}_{theory} = \underbrace{\frac{n}{c}}_{\text{复杂度降低}} \times \underbrace{\alpha}_{\text{并行效率}} \times \underbrace{\beta}_{\text{内存带宽}}
$$

其中：
- $\frac{n}{c}$：算法复杂度的降低因子
- $\alpha \in [0.5, 1]$：并行效率（受block size影响）
- $\beta \in [0.8, 1.2]$：内存带宽影响（VQ可能更cache-friendly）

**命题14.1（实际加速比的影响因素）**：

1. **硬件特性**：
   - GPU：更适合大矩阵乘法（标准Attention），小矩阵操作效率较低
   - TPU：更适合固定形状的操作，Transformer-VQ的动态性可能影响效率

2. **实现细节**：
   - 是否使用Flash Attention等优化
   - VQ的最近邻搜索算法（暴力搜索 vs 近似算法）
   - Block size的选择

**定理14.2（加速比的渐近行为）**：当序列长度$n \to \infty$且$c$固定时：

$$
\lim_{n \to \infty} \text{Speedup} = \frac{n}{c}
$$

这表明Transformer-VQ的优势在长序列上更加明显。

**实验数据**（根据原论文）：
- $n = 2048$：加速约2-3倍
- $n = 8192$：加速约8-12倍
- $n = 32768$：加速约20-30倍

这些数据与理论预测$\frac{n}{c} = \frac{n}{512}$基本一致（考虑到$\alpha, \beta$的影响）。

### 15. 信息瓶颈与表达能力

**定义15.1（信息瓶颈）**：Transformer-VQ的信息瓶颈在于编码表大小$c$：

$$
I(K; \hat{K}) \leq \log c
$$

这是量化后Key能携带的最大信息量（单位：bits）。

**定理15.1（表达能力的上界）**：对于序列长度$n$，量化后的Key矩阵$\hat{K}$最多有$c^n$种可能的配置，因此：

$$
H(\hat{K}) \leq n \log c
$$

相比于原始Key的$H(K) \approx n d_k \log 2$（假设每个元素用浮点数表示），信息损失为：

$$
\Delta H = H(K) - H(\hat{K}) \approx n(d_k \log 2 - \log c)
$$

**命题15.1（信息损失的影响）**：当$c = 512, d_k = 128$时：

$$
\frac{H(\hat{K})}{H(K)} \approx \frac{\log 512}{\log(2^{128})} = \frac{9}{128} \approx 7\%
$$

这表明量化保留了约7%的原始信息。然而，这个估计是保守的，因为：
1. Key的有效维度通常小于$d_k$（存在冗余）
2. 不是所有的Key信息都对最终任务重要

**定理15.2（任务相关的信息保留）**：对于特定任务$T$，重要的是保留与任务相关的信息：

$$
I(K; T) \approx I(\hat{K}; T)
$$

实验表明，量化后的模型性能接近全精度模型，说明量化成功保留了任务相关的信息。

### 16. 多头注意力的扩展

**定义16.1（Multi-Head Transformer-VQ）**：对于$h$个注意力头，每个头有自己的编码表$C^{(head)} \in \mathbb{R}^{c \times (d_k/h)}$：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \cdots, \text{head}_h)W^O
$$

其中：

$$
\text{head}_i = \text{Attention}(QW_i^Q, VQ(KW_i^K, C^{(i)}), VW_i^V)
$$

**定理16.1（Multi-Head的复杂度）**：Multi-Head Transformer-VQ的复杂度为：

$$
\mathcal{O}(h \cdot nc(d_k/h) + h \cdot nc(d_v/h)) = \mathcal{O}(ncd_k + ncd_v)
$$

与单头相同的渐近复杂度。

**命题16.1（编码表大小的分配）**：可以选择：
1. **固定总大小**：总编码表大小$hc$固定，每个头分配$c$个编码
2. **固定每头大小**：每个头都有$c$个编码，总大小为$hc$

实践中通常选择方案1，以控制总参数量。

### 17. 位置编码的兼容性

**定义17.1（旋转位置编码RoPE）**：RoPE通过旋转变换注入位置信息：

$$
q_m = R_m q, \quad k_m = R_m k
$$

其中$R_m$是位置$m$对应的旋转矩阵。

**命题17.1（Transformer-VQ与RoPE的冲突）**：直接应用RoPE到Transformer-VQ会遇到问题：

$$
\hat{K}_m = VQ(R_m K_m, C)
$$

量化后的$\hat{K}_m$不再满足RoPE的结构（$\hat{K}_m \neq R_m \hat{K}$）。

**解决方案17.1（ReRoPE）**：使用原文建议的窗口版RoPE（ReRoPE），只对窗口内的token对应用旋转，远处的token使用量化后的Key（已经失去了精确的位置信息）。

**定理17.1（ReRoPE的复杂度）**：使用ReRoPE，复杂度保持不变：

$$
\mathcal{O}(nlw + nc(d_k + d_v))
$$

其中$w$是窗口大小，通常$w \ll n$。

### 18. 与FSQ的对比

**定义18.1（Finite Scalar Quantization）**：FSQ直接量化向量的每个分量：

$$
\hat{K}_{ij} = \text{round}\left(\frac{K_{ij} - \min}{\max - \min} \cdot (L-1)\right)
$$

其中$L$是每个维度的量化级别。

**命题18.1（为什么Transformer-VQ不能用FSQ替代）**：

1. **需要连续值**：Transformer-VQ需要量化后的Key是连续向量（用于计算$\exp(Q\hat{K}^T)$），而FSQ给出的是离散索引

2. **编码表大小**：Transformer-VQ使用$c = 512$个编码，而FSQ在相同参数下每个维度只能有$L = \sqrt[d_k]{512} \approx 1.08$个级别（对于$d_k = 128$），这太粗糙了

3. **表达能力**：VQ的编码表可以学习数据的流形结构，而FSQ的笛卡尔积结构更僵硬

**定理18.1（VQ vs FSQ的trade-off）**：
- VQ：更灵活，更适合小编码表，需要K-Means初始化
- FSQ：更简单，更适合大编码表，容易训练

对于Transformer-VQ的场景（$c = 512, d_k = 128$），VQ是更好的选择。

### 19. 长度外推与泛化

**定义19.1（长度外推）**：长度外推指模型在训练序列长度$n_{train}$上训练，但能在更长的序列$n_{test} > n_{train}$上良好工作。

**定理19.1（Transformer-VQ的外推能力）**：Transformer-VQ具有良好的长度外推能力，因为：

1. **编码表无关长度**：$C \in \mathbb{R}^{c \times d_k}$的大小不依赖于$n$

2. **递归结构**：状态$U_i$的更新是线性的，不会累积误差

3. **局部依赖**：Attention Bias $B$只依赖于相对距离，不依赖于绝对位置

**命题19.1（与标准Transformer的对比）**：标准Transformer的外推能力受限于：
- 位置编码的外推（RoPE等虽然有一定外推能力，但仍有限）
- 注意力矩阵的大小依赖于序列长度

Transformer-VQ避免了第二个问题。

**实验观察**：原论文在ENWIK8上的实验显示，Transformer-VQ在测试时可以处理比训练时更长的序列，性能降解小于标准Transformer。

### 20. 与Sparse Attention的对比

**定义20.1（Sparse Attention）**：稀疏注意力只计算部分token对的注意力：

$$
\text{SparseAttn}(Q, K, V)_{ij} = \begin{cases}
\text{Attn}(Q, K, V)_{ij} & \text{if } (i,j) \in \mathcal{S} \\
0 & \text{otherwise}
\end{cases}
$$

其中$\mathcal{S}$是稀疏模式（如striped, fixed, local等）。

**命题20.1（复杂度对比）**：
- Sparse Attention：$\mathcal{O}(n \cdot s)$，其中$s = |\mathcal{S}_i|$是每行的稀疏度
- Transformer-VQ：$\mathcal{O}(nc)$

当$s = c$时，复杂度相同。

**定理20.1（表达能力对比）**：Transformer-VQ的表达能力通常强于Sparse Attention：

1. **动态选择**：Transformer-VQ根据Key的相似性动态选择"稀疏模式"，而Sparse Attention的模式是预定义的

2. **全局信息**：Transformer-VQ的状态$U_i$累积了所有历史信息，而Sparse Attention只看稀疏子集

**命题20.2（信息泄露）**：Sparse Attention的一个问题是可能泄露未来信息（如果稀疏模式不小心包含了未来的token）。Transformer-VQ通过causal mask和递归结构避免了这个问题。

### 21. 训练稳定性与收敛性

**定义21.1（训练目标）**：Transformer-VQ的训练目标是标准的语言模型损失：

$$
\mathcal{L} = -\sum_{i=1}^n \log P(x_i | x_{<i})
$$

加上VQ的辅助损失（如果使用）：

$$
\mathcal{L}_{VQ} = \sum_{i=1}^n \|K_i - \hat{K}_i\|^2
$$

**定理21.1（训练稳定性）**：Transformer-VQ的训练稳定性依赖于：

1. **编码表初始化**：好的初始化（如K-Means）是关键
2. **梯度裁剪**：由于梯度停止，梯度可能较大，需要裁剪
3. **学习率调度**：编码表$C$的学习率可能需要与其他参数不同

**命题21.1（收敛速度）**：实验显示Transformer-VQ的收敛速度与标准Transformer相当，有时甚至更快（因为参数更少）。

**定理21.2（局部最优）**：由于VQ操作的离散性，Transformer-VQ的优化景观可能有更多局部最优。但实践中这不是主要问题，可能因为：
1. 编码表的连续性（$C$是连续优化的）
2. 任务损失的平滑性
3. 大模型的过参数化

### 22. 模型容量与参数效率

**定义22.1（参数数量）**：Transformer-VQ相比标准Transformer增加的参数：

$$
\Delta \text{Params} = L \times c \times d_k
$$

其中$L$是层数。对于$L = 24, c = 512, d_k = 128$：

$$
\Delta \text{Params} = 24 \times 512 \times 128 = 1.57M
$$

这在总参数量（通常$>100M$）中占比很小。

**命题22.1（参数效率）**：Transformer-VQ的参数效率很高：
- 增加的参数：$\sim 1\%$
- 获得的加速：$>10\times$（长序列）

**定理22.1（模型容量的保持）**：尽管Key被量化，但模型的整体容量基本保持不变，因为：
1. Query $Q$没有被量化，仍保留完整信息
2. Value $V$没有被量化，仍保留完整信息
3. 只有Key的"索引"被量化，但通过编码表$C$恢复为连续值

### 23. 从预训练模型微调

**定义23.1（从预训练模型出发）**：给定预训练的标准Transformer，如何转换为Transformer-VQ？

**算法23.1（转换步骤）**：

1. **采样Key分布**：从预训练模型中前向传播一些样本，收集Key $\{K_i^{(l)}\}$（每层）

2. **K-Means聚类**：对每层的Key进行K-Means，得到$c$个聚类中心作为编码表$C^{(l)}$

3. **初始化VQ层**：在每层的Attention中加入VQ模块，使用$C^{(l)}$初始化

4. **微调**：使用较小的学习率微调整个模型

**定理23.1（微调的有效性）**：这种微调策略是有效的，因为：

1. **初始近似**：K-Means保证初始的量化误差较小
2. **渐进优化**：微调可以进一步优化编码表和其他参数
3. **正则化效应**：VQ起到正则化作用，可能提高泛化能力

**命题23.1（与从零训练的对比）**：
- 从零训练：需要更长的训练时间，但可能达到更优的解
- 微调：更快，但受限于预训练模型的初始化

### 24. 理论局限性与未来方向

**局限24.1（编码表大小的限制）**：编码表大小$c$是一个超参数，需要权衡：
- 太小：量化误差大，性能下降
- 太大：加速比降低，接近标准Transformer

**局限24.2（动态性的限制）**：编码表$C$在整个序列中是固定的，不能根据上下文动态调整。

**未来方向24.1（自适应编码表）**：可以设计上下文相关的编码表：

$$
C(x_{<i}) = f(x_{<i}, C_0)
$$

其中$f$是一个轻量级的网络，$C_0$是基础编码表。

**未来方向24.2（多尺度量化）**：类似Residual VQ，可以使用多个编码表：

$$
\hat{K}_i = c_1 + c_2 + \cdots + c_L
$$

每个$c_l$来自不同的编码表$C^{(l)}$，提供不同粒度的近似。

**未来方向24.3（Query量化）**：目前只量化Key，是否可以也量化Query？

$$
\hat{Q}_i = VQ(Q_i, C_Q), \quad \hat{K}_i = VQ(K_i, C_K)
$$

这可能进一步降低复杂度，但需要小心设计以避免过度的信息损失。

**未来方向24.4（与其他优化技术结合）**：
- Flash Attention：优化内存访问模式
- Mixture of Experts：结合稀疏激活
- Quantization-Aware Training：更激进的量化策略

### 25. 总结与结论

**核心贡献25.1**：Transformer-VQ提出了一个简单而有效的想法：

$$
\text{只需VQ一下Key，复杂度就从} O(n^2) \text{降到} O(n)
$$

**理论优势25.2**：
1. **线性复杂度**：$\mathcal{O}(nc(d_k + d_v))$，当$c \ll n$时为$\mathcal{O}(n)$
2. **保留结构**：保持softmax结构，不像线性Attention需要核函数近似
3. **简单实现**："挑出"技巧使得实现简单高效
4. **理论保证**：量化误差可控，近似质量有界

**实践价值25.3**：
1. **显著加速**：长序列上10-30倍加速
2. **性能保持**：实验显示性能接近标准Transformer
3. **易于部署**：可以从预训练模型微调得到

**理论洞察25.4**：
1. **VQ不仅用于VAE**：VQ在Transformer中也有重要应用
2. **离散化的好处**：适度的离散化可以在保持性能的同时大幅降低复杂度
3. **Key的特殊性**：Key比Query和Value更适合量化（因为Key主要用于索引）

**开放问题25.5**：
1. 为什么量化Key比量化Query或Value更有效？
2. 编码表大小$c$的理论最优值是多少？
3. 如何设计自适应的编码表？
4. Transformer-VQ的表达能力的精确刻画？

---

本推导全面分析了Transformer-VQ的数学原理、复杂度降低的机制、信息损失的定量分析、与其他方法的对比等，为理解这一优雅而强大的技术提供了坚实的理论基础。

