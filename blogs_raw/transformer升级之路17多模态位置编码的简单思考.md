---
title: Transformer升级之路：17、多模态位置编码的简单思考
slug: transformer升级之路17多模态位置编码的简单思考
date: 2024-03-29
tags: 详细推导, attention, 位置编码, rope, 多模态, 生成模型
status: pending
---
# Transformer升级之路：17、多模态位置编码的简单思考

**原文链接**: [https://spaces.ac.cn/archives/10040](https://spaces.ac.cn/archives/10040)

**发布日期**: 

---

在这个系列的第二篇文章[《Transformer升级之路：2、博采众长的旋转式位置编码》](/archives/8265)中，笔者提出了旋转位置编码（RoPE）——通过绝对位置的形式实现相对位置编码的方案。一开始RoPE是针对一维序列如文本、音频等设计的（RoPE-1D），后来在[《Transformer升级之路：4、二维位置的旋转式位置编码》](/archives/8397)中我们将它推广到了二维序列（RoPE-2D），这适用于图像的ViT。然而，不管是RoPE-1D还是RoPE-2D，它们的共同特点都是单一模态，即纯文本或者纯图像输入场景，那么对于多模态如图文混合输入场景，RoPE该做如何调整呢？

笔者搜了一下，发现鲜有工作讨论这个问题，主流的做法似乎都是直接展平所有输入，然后当作一维输入来应用RoPE-1D，因此连RoPE-2D都很少见。且不说这种做法会不会成为图像分辨率进一步提高时的效果瓶颈，它终究是显得不够优雅。所以，接下来我们试图探寻两者的一个自然结合。

## 旋转位置 #

RoPE名称中的“旋转”一词，来源于旋转矩阵$\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\\ \sin n\theta & \cos n\theta\end{pmatrix}$，它满足  
\begin{equation}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}\end{equation}  
这样一来对于$\boldsymbol{q},\boldsymbol{k}$（假设为列向量）的内积就有  
\begin{equation}\left(\boldsymbol{\mathcal{R}}_m\boldsymbol{q}\right)^{\top} \left(\boldsymbol{\mathcal{R}}_n\boldsymbol{k}\right)= \boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k}=\boldsymbol{q}^{\top}\boldsymbol{\mathcal{R}}_{n-m}\boldsymbol{k}\end{equation}  
最左边的式子中，$\boldsymbol{\mathcal{R}}_m\boldsymbol{q},\boldsymbol{\mathcal{R}}_n\boldsymbol{k}$是独立进行的，不涉及到$m,n$的交互，所以它形式上是绝对位置，但最右端的等价形式只依赖于相对位置$n-m$，所以跟Dot-Product的Attention结合之后，它实质表现为相对位置。这个特性也让RoPE具备平移不变性：因为$(n+c) - (m+c) = n-m$，所以在应用RoPE之前全体绝对位置都加上一个常数，那么Attention的结果理论上不会变化（实际上受限于计算精度，可能有微小误差）。

以上是$\boldsymbol{q},\boldsymbol{k}\in\mathbb{R}^2$的形式，对于$\boldsymbol{q},\boldsymbol{k}\in \mathbb{R}^d$（其中$d$是偶数），我们需要一个$d\times d$的旋转矩阵，为此我们引入$d/2$个不同的$\theta$，构造分块对角矩阵  
\begin{equation}\small{\boldsymbol{\mathcal{R}}_n^{(d\times d)} = \begin{pmatrix}  
\cos n\theta_0 & -\sin n\theta_0 & 0 & 0 & \cdots & 0 & 0 \\\  
\sin n\theta_0 & \cos n\theta_0 & 0 & 0 & \cdots & 0 & 0 \\\  
0 & 0 & \cos n\theta_1 & -\sin n\theta_1 & \cdots & 0 & 0 \\\  
0 & 0 & \sin n\theta_1 & \cos n\theta_1 & \cdots & 0 & 0 \\\  
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\  
0 & 0 & 0 & 0 & \cdots & \cos n\theta_{d/2-1} & -\sin n\theta_{d/2-1} \\\  
0 & 0 & 0 & 0 & \cdots & \sin n\theta_{d/2-1} & \cos n\theta_{d/2-1} \\\  
\end{pmatrix}}\end{equation}  
从实现上看，就是将$\boldsymbol{q},\boldsymbol{k}$两两分组，每组取不同的$\theta$进行二维的旋转变换，这些是已有的RoPE内容，就不再详细展开了。原则上来说，我们只需要找到一个最低维的解，就可以通过分块对角的方式推广到一般维度，因此下面的分析都只考虑最小维度。

## 二维位置 #

当我们谈到“维度”这个概念时，可能会有多种含义，比如刚才我们说$\boldsymbol{q},\boldsymbol{k}\in \mathbb{R}^d$，这就是说$\boldsymbol{q},\boldsymbol{k}$都是$d$维向量，但本文所聚焦的RoPE-1D、RoPE-2D，它并不是指这个维度，而是指记录一个位置所需要的维度。

[![文本及其位置ID](/usr/uploads/2024/03/1460426521.png)](/usr/uploads/2024/03/1460426521.png "点击查看原图")

文本及其位置ID

比如，我们要文本的某个token的位置，那么只需要一个标量$n$，记录它是第$n$个token。但对于图像来说，即便进行了patchify，它通常也会保留width和height两个方向维度，所以我们需要一对坐标$(x,y)$才能准确编码某个patch的位置：  


[![图片及其位置坐标](/usr/uploads/2024/03/3054926745.png)](/usr/uploads/2024/03/3054926745.png "点击查看原图")

图片及其位置坐标

上一节介绍$\boldsymbol{\mathcal{R}}_n$，它只编码了一个标量$n$，所以它是RoPE-1D，而为了更合理地处理图像输入，我们要推广到相应的RoPE-2D：  
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y}=\left(  
\begin{array}{cc:cc}  
\cos x\theta & -\sin x\theta & 0 & 0 \\\  
\sin x\theta & \cos x\theta & 0 & 0 \\\  
\hdashline  
0 & 0 & \cos y\theta & -\sin y\theta \\\  
0 & 0 & \sin y\theta & \cos y\theta \\\  
\end{array}\right) = \begin{pmatrix}\boldsymbol{\mathcal{R}}_x & 0 \\\ 0 & \boldsymbol{\mathcal{R}}_y\end{pmatrix}\end{equation}  
很明显，这只是$\boldsymbol{\mathcal{R}}_x$和$\boldsymbol{\mathcal{R}}_y$以分块对角的形式组合在一起，因此也很自然能将它推广到3D甚至更高维度。从实现上来理解就是更简单了，它就是将$\boldsymbol{q},\boldsymbol{k}$都切分为两半（3D就是三等分、4D就是四等分，依此类推），每一半都是$\mathbb{R}^{d/2}$的向量，然后一半做$x$的RoPE-1D，另一半做$y$的RoPE-1D，最后再拼起来。

需要指出的是，从对称性和简洁性考虑，上面构造的$\boldsymbol{\mathcal{R}}_{x,y}$中对$x,y$我们使用了相同的$\theta$，但这原则上是非必须的，在适当情况下我们分别给$x,y$配置略有不同的$\theta$。

## 强行降维 #

现在我们看到，文本的位置是一个标量$n$，图片的位置则是一个向量$(x,y)$，两者并不一致，因此在处理图文混合输入时就需要一些技巧，来调和两者之间的不一致性。

最直接的方案，文章开头已经说了，就是直接展平图片为一维向量序列，然后就当作普通文本来处理，文本怎么加位置编码它就怎么加位置编码。这种思路自然是非常通用的，不限于加RoPE，也可以加任何绝对位置编码，笔者目测已有的一些多模态模型，如Fuyu-8b、Deepseek-VL、Emu2等，都是这样做的，可能细节处理上会有所不同，比如遇到不同行的patch可以考虑加个表示[SEP]的special token来分隔：  


[![文本和图片都展平为一维来处理](/usr/uploads/2024/03/3697844644.png)](/usr/uploads/2024/03/3697844644.png "点击查看原图")

文本和图片都展平为一维来处理

这个方案也契合了当前主流的Decoder-Only架构，因为Decoder-Only意味着即便不加位置编码，它也不是置换不变的，因此必须人为指定我们认为最佳的输入顺序，而既然要指定输入顺序了，按照所指定的顺序使用一维的位置编码也是很自然的选择。此外，在纯文本时这种方案的模型跟普通纯文本LLM无异，所以这也允许我们将训练好的文本LLM来继续训练成一个多模态模型。

然而，从笔者的角度看，位置编码的概念本身不应该和Attention的用法绑定，它应该普适于Decoder、Encoder乃至任意的Attention Mask。另一方面，保持位置的二维性才能最大程度上保留我们关于相近位置的先验，比如我们认为位置$(x+1,y)$和$(x,y+1)$都应该跟$(x,y)$具有相近的距离，但如果（先水平后垂直）展平的话，$(x,y)$变为$xw + y$，而$(x+1,y)$和$(x,y+1)$分别变为了$xw+y+w$和$xw+y+1$，前者与$xw + y$的距离就依赖于$w$而后者是固定的$1$。当然，我们还可以指定其他制定顺序，但不管怎么指定顺序，都无法完全兼容所有邻近位置的相近性，毕竟少了一个维度，可表达的相似性就少了很多。

## 统一升维 #

从向量空间的角度看，一维的标量可以看成一个特殊的二维向量，因此相比于展平为一维，如果我们反过来将所有输入的位置都统一到二维，原则上有更大的操作空间。

为此，我们可以考虑一种常见的排版方式：以图片为分隔符，对文本进行分段，连续的文本都视为一行，图片则视为多行文本，那么整个图文混合输入就相当于一篇多行长文，每个文本token或者图片patch，都有自己所属的行数$x$以及行内的顺序$y$，这就给所有的输入单元（token或者patch）都赋予了一个二维位置$(x,y)$，于是可以统一用RoPE-2D（其他2D形式的位置编码理论上也可以）来编码位置，同时还保持了原本图片位置的二维性。

[![模拟排版统一构建二维位置坐标](/usr/uploads/2024/03/918639143.png)](/usr/uploads/2024/03/918639143.png "点击查看原图")

模拟排版统一构建二维位置坐标

很明显，该方案的主要优点是非常直观，它直接跟实际的视觉排版相对应，便于理解和推广。但它也有一个非常明显的缺点，那就是对于纯文本输入，它无法退化为RoPE-1D，而是变成了$x$始终为1的RoPE-2D，这样从已训练好的文本LLM出发来训练多模态LLM的可行性就值得怀疑。此外，以图片作为分割点的话，当图片比较多时，可能会让文本被分割得过于“支离破碎”，具体表现包括每一段文本的长度波动太大、本该连续的文本被强行换行等，这些都可能成为限制效果的瓶颈。

## 合二为一 #

如果要无损保留图片patch的位置信息，那么统一到二维然后用RoPE-2D（或者其他2D形式的位置编码）看上去是必然的选择，所以上一节的方案已经是走在了正确的方向上，我们需要进一步思考的是如何能够让它对于纯文本输入能够退化为RoPE-1D，以兼容已有的文本LLM。

首先，我们在前面已经提到过，$\boldsymbol{\mathcal{R}}_{x,y}$是$\boldsymbol{\mathcal{R}}_x$和$\boldsymbol{\mathcal{R}}_y$的分块对角组合，所以$\boldsymbol{\mathcal{R}}_{n,n}$是两个$\boldsymbol{\mathcal{R}}_n$的分块对角组合，而RoPE-1D的$\boldsymbol{\mathcal{R}}_n^{(d\times d)}$也是多个不同$\theta$的$\boldsymbol{\mathcal{R}}_n$的分块对角组合，由此可见，只要我们从$\boldsymbol{\mathcal{R}}_n^{(d\times d)}$选取不同的$\theta$给$x,y$，那么$\boldsymbol{\mathcal{R}}_{n,n}$就可以看成是RoPE-1D（即$\boldsymbol{\mathcal{R}}_n^{(d\times d)}$）的一部分。这样看来，要想RoPE-2D能退化为RoPE-1D，那么文本的位置应该采取$(n,n)$的形式，而不是像上一节那样用其他方式指定一个行号。

然后，在图片内部，我们则使用常规的RoPE-2D，对于单张$w\times h$个patch的图片来说，它的二维位置坐标展平后是  
\begin{array}{c|cccc|cccc|c|cccc}  
\hline  
x & 1 & 1 & \cdots & 1 & 2 & 2 & \cdots & 2 & \quad \cdots \quad & h & h & \cdots & h \\\  
\hline  
y & 1 & 2 & \cdots & w & 1 & 2 & \cdots & w & \quad \cdots \quad & 1 & 2 & \cdots & w \\\  
\hline  
\end{array}  
如果这张图片位于一个长度为$L$的句子后面，我们这个句子的最后一个token的位置编码就是$(L,L)$，于是这张接在句子后面的图片的位置编码看上去应该是  
\begin{array}{c|cccc|c|cccc}  
\hline  
x & L+1 & L+1 & \cdots & L+1 & \quad \cdots \quad & L+h & L+h & \cdots & L+h \\\  
\hline  
y & L+1 & L+2 & \cdots & L+w & \quad \cdots \quad & L+1 & L+2 & \cdots & L+w \\\  
\hline  
\end{array}  
但这并不完美，因为句子的最后一个token的位置是$(L,L)$，图片第一个patch的位置是$(L+1,L+1)$，它们相差$(1,1)$；假设这张图片后面再接一个句子，那么设该句子的第一个token的位置是$(K,K)$，图片的最后一个patch的位置则是$(L+h,L+w)$，当$w\neq h$时，不管我们怎么设置$K$，都不可能让$(K,K)$与$(L+h,L+w)$的差为$(1,1)$，即图片关于左右的句子存在不对称性，这就显得不够优雅。

为了改进这一点，我们可以将图片的$x,y$分别乘以正数$s,t$：  
\begin{array}{c|cccc|cccc|c|cccc}  
\hline  
x & s & s & \cdots & s & 2s & 2s & \cdots & 2s & \quad \cdots \quad & hs & hs & \cdots & hs \\\  
\hline  
y & t & 2t & \cdots & wt & t & 2t & \cdots & wt & \quad \cdots \quad & t & 2t & \cdots & wt \\\  
\hline  
\end{array}  
只要$s,t\neq 0$，那么这个缩放对位置信息是无损的，因此这样的操作是允许的。而引入scale之后，假设句子的最后一个token的位置依旧是$(L,L)$，那么图片的位置同样是上述序列都加上$L$，此时“句子的最后一个token的位置”与“图片第一个patch的位置”之差就是$(s,t)$，如果我们希望“图片后面的句子的第一个token的位置”与“图片最后一个patch的位置”之差也是$(s,t)$，那么就应该有  
\begin{equation}\begin{pmatrix}L + hs \\\ L + wt \end{pmatrix} + \begin{pmatrix}s \\\ t \end{pmatrix} = \begin{pmatrix}K \\\ K \end{pmatrix}\quad \Rightarrow \quad (h+1)s = (w+1)t\end{equation}  
考虑到$h,w$的任意性，并且希望保证位置ID都是整数的话，那么最简单的一个解自然是$s=w+1,t=h+1$，新句子第一个token的位置将会是$K=L+(w+1)(h+1)$。一个具体的例子如下图所示：  


[![支持退化为RoPE-1D的二维位置](/usr/uploads/2024/03/115077303.png)](/usr/uploads/2024/03/115077303.png "点击查看原图")

支持退化为RoPE-1D的二维位置

## 延伸思考 #

左边句子最后一个token的位置是$L$，右边句子第一个token的位置是$K=L+(w+1)(h+1)$，如果中间部分也是一个句子的话，那么可以推出该句子有$(w+1)(h+1)-1$个token，这也等价于说如果两个句子之间夹着一个$w\times h$的图片，那么对这两个句子的相对位置来说等价于隔着一个$(w+1)(h+1)-1$个token的句子。这个数字看起来有点不自然，因为看上去$wh$才是完美答案，但可惜这是保证所有位置ID都是整数的最简单解。如果允许非整数的位置ID，那么可以约定$w\times h$的图片等价于$wh$个token，反过来推出  
\begin{equation}s = \frac{wh + 1}{h+1}, \quad t = \frac{wh + 1}{w+1}\end{equation}

可能有读者要问：如果是两张不同大小的图片相邻，是不是就没有这样对称的方案了？这其实也不难，只要每张图片的前后，我们都加入special token来标记，如[IMG]、[/IMG]，并且special token当作普通文本token来编码位置，这样就直接避免了两张图片直接相邻的情况（因为按照约定，同一张图片的patch之间必然夹在[IMG]和[/IMG]，这两个token当作文本来处理，所以就等价于说每一张图片必然夹在两个文本之间）。此外，上述介绍中没有提及[SEP]，如果有需要自行引入即可，事实上只有用patch by patch的自回归方式做图片生成时，才有必要引入[SEP]，如果图片单纯是作为输入，或者图片生成用扩散模型来做，那么[SEP]则是多余的。

至此，我们关于将RoPE推广到图文混合输入的推导已经完成，如果需要一个名字，可以将最后的方案称之为“RoPE-Tie（RoPE for Text-image）”。不得不说的是，最后的RoPE-Tie并不算太漂亮，以至于给人一种“雕花”的感觉。从效果上来看，相比直接展平为一维用RoPE-1D，换用RoPE-Tie之后也不见得会有什么提升，它更多是笔者的强迫症的一个产物。所以，对于已经scale到了一定规模的多模态模型，就没有必要做出什么改动了，但如果还没有起步或者刚刚起步，那么不妨尝试一下RoPE-Tie。

## 文章小结 #

本文讨论了如何将RoPE-1D和RoPE-2D结合起来，来更好地处理图文混合的输入格式，主要思想是通过RoPE-2D支持图片的二维位置指标，并且通过适当的约束，使得在纯文本情况下能退化为常规的RoPE-1D。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10040>_

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

苏剑林. (Mar. 29, 2024). 《Transformer升级之路：17、多模态位置编码的简单思考 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10040>

@online{kexuefm-10040,  
title={Transformer升级之路：17、多模态位置编码的简单思考},  
author={苏剑林},  
year={2024},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/10040}},  
} 


---

## 公式推导与注释

### 1. 一维RoPE的数学基础

**定义1.1：一维旋转位置编码（RoPE-1D）**

对于文本序列，位置$m$的旋转矩阵定义为：
$$
\boldsymbol{\mathcal{R}}_m^{(2)} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

对于$d$维向量，分为$d/2$对，每对使用不同频率$\theta_i$：
$$
\boldsymbol{\mathcal{R}}_m^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_m^{(\theta_0)}, \boldsymbol{\mathcal{R}}_m^{(\theta_1)}, \ldots, \boldsymbol{\mathcal{R}}_m^{(\theta_{d/2-1})})
$$

**定理1.1：相对位置性质**

RoPE满足核心的相对位置性质：
$$
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^T (\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
$$

即内积只依赖于相对位置$\Delta = n - m$，而不依赖于绝对位置。

**推导1.1：旋转矩阵的性质**

旋转矩阵是正交矩阵：
$$
\boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_m = \boldsymbol{I}
$$

保持向量模长不变：
$$
\|\boldsymbol{\mathcal{R}}_m \boldsymbol{v}\| = \|\boldsymbol{v}\|
$$

满足群性质：
$$
\boldsymbol{\mathcal{R}}_m \boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{m+n}
$$

**推导1.2：频率的作用**

不同频率$\theta_i$对应不同的周期：
$$
T_i = \frac{2\pi}{\theta_i}
$$

标准设置中：
$$
\theta_i = 10000^{-2i/d}
$$

因此：
$$
T_i = 2\pi \cdot 10000^{2i/d}
$$

高频（$i$小）：短周期，适合编码局部位置
低频（$i$大）：长周期，适合编码远程位置

### 2. 二维RoPE的数学推广

**定义2.1：二维旋转位置编码（RoPE-2D）**

对于图像patch的二维位置$(x, y)$，旋转矩阵定义为分块对角形式：
$$
\boldsymbol{\mathcal{R}}_{x,y}^{(4)} = \begin{pmatrix} \boldsymbol{\mathcal{R}}_x & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\mathcal{R}}_y \end{pmatrix} = \begin{pmatrix}
\cos(x\theta) & -\sin(x\theta) & 0 & 0 \\
\sin(x\theta) & \cos(x\theta) & 0 & 0 \\
0 & 0 & \cos(y\theta) & -\sin(y\theta) \\
0 & 0 & \sin(y\theta) & \cos(y\theta)
\end{pmatrix}
$$

**推导2.1：二维相对位置性质**

对于两个位置$(x_1, y_1)$和$(x_2, y_2)$：
$$
\boldsymbol{\mathcal{R}}_{x_1,y_1}^T \boldsymbol{\mathcal{R}}_{x_2,y_2} = \begin{pmatrix} \boldsymbol{\mathcal{R}}_{x_2-x_1} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\mathcal{R}}_{y_2-y_1} \end{pmatrix}
$$

因此：
$$
(\boldsymbol{\mathcal{R}}_{x_1,y_1} \boldsymbol{q})^T (\boldsymbol{\mathcal{R}}_{x_2,y_2} \boldsymbol{k}) = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_{x_2-x_1, y_2-y_1} \boldsymbol{k}
$$

内积只依赖于相对位置$(\Delta_x, \Delta_y) = (x_2 - x_1, y_2 - y_1)$。

**定理2.1：二维位置编码的轴向分解**

二维RoPE可以分解为两个独立的一维RoPE：
$$
\text{RoPE-2D}(x, y) = \text{RoPE-1D}_x(x) \oplus \text{RoPE-1D}_y(y)
$$

其中$\oplus$表示分块对角组合。

**推导2.2：高维推广**

对于$d$维向量和$(k_1, k_2)$维的空间位置：
$$
\boldsymbol{\mathcal{R}}_{k_1,k_2}^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_{k_1,0}^{(d_1)}, \boldsymbol{\mathcal{R}}_{0,k_2}^{(d_2)})
$$

其中$d_1 + d_2 = d$，通常取$d_1 = d_2 = d/2$。

对于三维（视频）或更高维的情况，类似地分块对角组合。

### 3. 展平策略的数学分析

**定义3.1：图像展平映射**

将$H \times W$的图像展平为一维序列，常见的映射方式：

**行优先（Row-major）**：
$$
\phi_{row}(x, y) = x \cdot W + y, \quad x \in [0, H), \, y \in [0, W)
$$

**列优先（Column-major）**：
$$
\phi_{col}(x, y) = y \cdot H + x
$$

**Z-order（Morton order）**：
$$
\phi_{Z}(x, y) = \text{interleave}(\text{bits}(x), \text{bits}(y))
$$

**推导3.1：相邻性的损失**

在二维空间中，位置$(x, y)$有4个相邻位置：
$$
\mathcal{N}_{2D}(x, y) = \{(x \pm 1, y), (x, y \pm 1)\}
$$

其欧氏距离为：
$$
d_{2D}((x, y), (x', y')) = \sqrt{(x - x')^2 + (y - y')^2} = 1
$$

展平后，使用行优先映射：
$$
\begin{align}
d_{1D}(\phi(x, y), \phi(x+1, y)) &= |\phi(x+1, y) - \phi(x, y)| = W \\
d_{1D}(\phi(x, y), \phi(x, y+1)) &= |\phi(x, y+1) - \phi(x, y)| = 1
\end{align}
$$

垂直相邻的patch在一维序列中距离为$W$，远大于1，严重破坏了相邻性。

**定理3.1：展平无法保持全局相邻性**

**证明**：考虑$H \times W$的图像，任意位置$(x, y)$与$(x+1, y)$在二维空间中相邻，距离为1。

展平后：
$$
|\phi(x+1, y) - \phi(x, y)| = W
$$

要保持所有相邻关系，需要：
$$
W = 1
$$

但这要求图像宽度为1，即退化为一维。因此，展平策略必然损失部分相邻性信息。$\square$

**推导3.2：Z-order的改进**

Z-order（空间填充曲线）试图更好地保持局部性：
$$
\phi_Z(x, y) = \sum_{i=0}^{\log_2 \max(H,W)} (x_i \cdot 2^{2i+1} + y_i \cdot 2^{2i})
$$

其中$x_i, y_i$是$x, y$的第$i$位二进制数字。

Z-order的优势：相邻patch在一维序列中的距离更均衡，但仍无法完全保持二维相邻性。

### 4. 文本-图像混合输入的统一编码

**定义4.1：混合模态的位置表示**

文本token：位置为标量$n \in \mathbb{N}$
图像patch：位置为向量$(x, y) \in \mathbb{N}^2$

**策略1：展平为一维（强行降维）**

所有输入统一为一维位置$p \in \mathbb{N}$：
$$
p = \begin{cases}
n, & \text{文本token } n \\
\phi(x, y), & \text{图像patch } (x, y)
\end{cases}
$$

应用标准RoPE-1D：
$$
\boldsymbol{\mathcal{R}}_p^{(d)}
$$

**策略2：升维为二维（统一升维）**

所有输入统一为二维位置$(p_1, p_2) \in \mathbb{N}^2$：
$$
(p_1, p_2) = \begin{cases}
(n, n), & \text{文本token } n \\
(x, y), & \text{图像patch } (x, y)
\end{cases}
$$

应用RoPE-2D：
$$
\boldsymbol{\mathcal{R}}_{p_1, p_2}^{(d)}
$$

**推导4.1：策略1的问题**

使用行优先展平，两个水平相邻的patch：
$$
(x, y) \to p_1 = xW + y
$$
$$
(x, y+1) \to p_2 = xW + y + 1
$$

相对位置：$\Delta_1 = 1$

两个垂直相邻的patch：
$$
(x, y) \to p_1 = xW + y
$$
$$
(x+1, y) \to p_3 = (x+1)W + y
$$

相对位置：$\Delta_2 = W$

在二维空间中，两对patch的距离都是1，但展平后的相对位置差异为$W$倍，无法体现这种对称性。

**推导4.2：策略2对文本的退化**

对于文本token，位置表示为$(n, n)$：
$$
\boldsymbol{\mathcal{R}}_{n,n}^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_n^{(d/2)}, \boldsymbol{\mathcal{R}}_n^{(d/2)})
$$

两个文本token的相对位置编码：
$$
\boldsymbol{\mathcal{R}}_{m,m}^T \boldsymbol{\mathcal{R}}_{n,n} = \text{diag}(\boldsymbol{\mathcal{R}}_{n-m}^{(d/2)}, \boldsymbol{\mathcal{R}}_{n-m}^{(d/2)})
$$

这等价于两个相同的$d/2$维RoPE-1D的组合，而不是标准的$d$维RoPE-1D。

标准RoPE-1D使用$d/2$个不同频率$\{\theta_i\}_{i=0}^{d/2-1}$，而策略2只使用了其中一半$\{\theta_i\}_{i=0}^{d/4-1}$（每个重复两次）。

**定理4.1：策略2无法退化为RoPE-1D**

对于纯文本输入，策略2的位置编码：
$$
\boldsymbol{\mathcal{R}}_{n,n}^{(d)} \neq \boldsymbol{\mathcal{R}}_n^{(d)}
$$

因为前者只使用了后者一半的频率，信息容量降低。$\square$

### 5. RoPE-Tie：统一一维和二维的方案

**定义5.1：RoPE-Tie的核心思想**

**关键观察**：RoPE-2D的分块对角结构
$$
\boldsymbol{\mathcal{R}}_{x,y}^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_x^{(d/2)}, \boldsymbol{\mathcal{R}}_y^{(d/2)})
$$

可以看作两个独立的$d/2$维RoPE-1D的组合。

如果文本使用$(n, n)$作为位置，则：
$$
\boldsymbol{\mathcal{R}}_{n,n}^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_n^{(d/2)}, \boldsymbol{\mathcal{R}}_n^{(d/2)})
$$

要使其等价于标准RoPE-1D，需要$x$和$y$使用不同的频率集合。

**推导5.1：频率分配**

标准RoPE-1D的频率：
$$
\{\theta_i\}_{i=0}^{d/2-1} = \{10000^{-2i/d}\}_{i=0}^{d/2-1}
$$

RoPE-Tie的频率分配：
- $x$方向：使用$\{\theta_{2i}\}_{i=0}^{d/4-1}$（偶数索引）
- $y$方向：使用$\{\theta_{2i+1}\}_{i=0}^{d/4-1}$（奇数索引）

这样对于文本token$(n, n)$：
$$
\boldsymbol{\mathcal{R}}_{n,n}^{RoPE-Tie} = \text{diag}(\boldsymbol{\mathcal{R}}_n^{(\{\theta_{2i}\})}, \boldsymbol{\mathcal{R}}_n^{(\{\theta_{2i+1}\})})
$$

组合起来恰好覆盖了所有$d/2$个频率，等价于标准RoPE-1D。

**定理5.1：RoPE-Tie的退化性质**

使用适当的频率分配，RoPE-Tie满足：
$$
\boldsymbol{\mathcal{R}}_{n,n}^{RoPE-Tie} = \boldsymbol{\mathcal{R}}_n^{RoPE-1D}
$$

对于纯文本输入，RoPE-Tie完全退化为标准RoPE-1D。$\square$

### 6. 图像位置的缩放因子

**定义6.1：带缩放的二维位置**

对于$w \times h$的图像，位置$(x, y)$缩放为：
$$
(sx, ty)
$$

其中$s, t > 0$是缩放因子。

**定理6.1：缩放不改变位置信息**

由于旋转编码的性质，缩放只是重新参数化：
$$
\boldsymbol{\mathcal{R}}_{sx, ty} = \begin{pmatrix} \boldsymbol{\mathcal{R}}_{sx} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\mathcal{R}}_{ty} \end{pmatrix}
$$

相对位置：
$$
\Delta_{scaled} = ((x_2 - x_1)s, (y_2 - y_1)t)
$$

只要$(x_2 - x_1, y_2 - y_1)$不同，$\Delta_{scaled}$也不同，因此位置信息无损。

**推导6.1：左右对称性的数学表达**

设句子最后一个token位置为$L$，图像位置$(x, y)$缩放为$(sx + L, ty + L)$。

图像第一个patch（左上角）：
$$
(s + L, t + L)
$$

与句子最后token的距离：
$$
\Delta_{left} = (s + L, t + L) - (L, L) = (s, t)
$$

图像最后一个patch（右下角）：
$$
(hs + L, wt + L)
$$

后续句子第一个token位置为$K$，要求：
$$
(K, K) - (hs + L, wt + L) = (s, t)
$$

解得：
$$
K = L + hs + s = L + (h+1)s
$$
$$
K = L + wt + t = L + (w+1)t
$$

因此：
$$
(h+1)s = (w+1)t
$$

**推导6.2：缩放因子的选择**

**方案1：整数位置ID**

要求所有位置都是整数，最简单的解是：
$$
s = w + 1, \quad t = h + 1
$$

此时：
$$
K = L + (w+1)(h+1)
$$

图像"等价于"$(w+1)(h+1)$个文本token。

**方案2：图像等价于$wh$个token**

如果希望图像等价于$wh$个token：
$$
K = L + wh + 1
$$

则：
$$
(h+1)s = (w+1)t = wh + 1
$$

解得：
$$
s = \frac{wh + 1}{h + 1}, \quad t = \frac{wh + 1}{w + 1}
$$

此时位置ID不一定是整数。

**推导6.3：两种方案的比较**

方案1：
- 优点：所有位置都是整数，实现简单
- 缺点：图像"虚增"了一些等价token数

方案2：
- 优点：图像等价token数精确为$wh$，更符合直觉
- 缺点：位置ID可能非整数，需要处理浮点数

实践中，方案1更常用，因为"虚增"的影响很小（相对于整个序列）。

### 7. 特殊token的位置处理

**定义7.1：图像边界token**

引入特殊token标记图像边界：
- [IMG]：图像开始
- [/IMG]：图像结束

这些token作为普通文本token处理，位置为一维标量。

**推导7.1：完整的位置序列**

设句子有$L$个token，图像有$w \times h$个patch。

完整序列的位置：
$$
\begin{align}
&\text{Text}_1, \ldots, \text{Text}_L &&: (1,1), \ldots, (L,L) \\
&\text{[IMG]} &&: (L+1, L+1) \\
&\text{Patch}_{1,1}, \ldots, \text{Patch}_{h,w} &&: (s+L+1, t+L+1), \ldots, (hs+L+1, wt+L+1) \\
&\text{[/IMG]} &&: (K, K) \\
&\text{Text}_{L+1}, \ldots &&: (K+1, K+1), \ldots
\end{align}
$$

其中$K = L + 1 + (w+1)(h+1)$（使用方案1的缩放）。

**定理7.1：特殊token避免了模态直接相邻**

有了[IMG]和[/IMG]，任意两个图像patch之间都至少隔着这两个特殊token，因此：
- 不需要考虑两张不同图像直接相邻的情况
- 每张图像都"夹在"两个文本token之间（[IMG]和[/IMG]）
- 统一的位置编码方案适用于所有情况

### 8. 轴向位置编码的数学理论

**定义8.1：轴向分解（Axial Decomposition）**

二维位置编码$(x, y)$的轴向分解：
$$
\text{Pos}(x, y) = \text{Pos}_x(x) + \text{Pos}_y(y)
$$

或乘法形式：
$$
\text{Pos}(x, y) = \text{Pos}_x(x) \odot \text{Pos}_y(y)
$$

RoPE-2D使用的是分块对角（直和）形式：
$$
\boldsymbol{\mathcal{R}}_{x,y} = \boldsymbol{\mathcal{R}}_x \oplus \boldsymbol{\mathcal{R}}_y
$$

**定理8.1：轴向分解保持维度加性**

加法分解：
$$
\dim(\text{Pos}(x, y)) = \dim(\text{Pos}_x(x)) = \dim(\text{Pos}_y(y))
$$

直和分解（RoPE-2D）：
$$
\dim(\boldsymbol{\mathcal{R}}_{x,y}) = \dim(\boldsymbol{\mathcal{R}}_x) + \dim(\boldsymbol{\mathcal{R}}_y)
$$

**推导8.1：加法vs直和的比较**

**加法分解**：
$$
\boldsymbol{p}_{x,y} = \boldsymbol{p}_x + \boldsymbol{p}_y, \quad \boldsymbol{p}_x, \boldsymbol{p}_y \in \mathbb{R}^d
$$

优点：维度不增加
缺点：$x$和$y$的信息混合，可能相互干扰

**直和分解**：
$$
\boldsymbol{p}_{x,y} = \begin{pmatrix} \boldsymbol{p}_x \\ \boldsymbol{p}_y \end{pmatrix}, \quad \boldsymbol{p}_x \in \mathbb{R}^{d_1}, \boldsymbol{p}_y \in \mathbb{R}^{d_2}
$$

优点：$x$和$y$完全独立，信息不干扰
缺点：维度翻倍（$d_1 + d_2$）

**推导8.2：RoPE的选择**

RoPE选择直和形式的原因：
1. **独立性**：$x$和$y$方向完全独立，便于分析和优化
2. **相对位置**：旋转矩阵的群性质只在直和形式下保持
3. **可扩展性**：容易推广到3D（视频）或更高维

虽然维度增加，但通过合理分配（如$d_1 = d_2 = d/2$），总维度仍为$d$。

### 9. 多模态位置编码的信息论分析

**定义9.1：位置编码的熵**

位置编码$\boldsymbol{p}$的信息熵：
$$
H(\boldsymbol{p}) = -\int p(\boldsymbol{p}) \log p(\boldsymbol{p}) d\boldsymbol{p}
$$

对于离散位置集合$\mathcal{P}$：
$$
H(\mathcal{P}) = \log |\mathcal{P}|
$$

**定理9.1：二维位置的信息容量**

一维位置：$\mathcal{P}_{1D} = \{1, 2, \ldots, L\}$
$$
H_{1D} = \log L
$$

二维位置：$\mathcal{P}_{2D} = \{(x, y) : 1 \leq x \leq H, 1 \leq y \leq W\}$
$$
H_{2D} = \log(H \cdot W)
$$

展平后：$\mathcal{P}_{flat} = \{1, 2, \ldots, HW\}$
$$
H_{flat} = \log(HW) = H_{2D}
$$

**推导9.1：展平保持信息但损失结构**

虽然：
$$
H_{flat} = H_{2D}
$$

但二维结构信息丢失。定义结构熵：
$$
H_{struct} = H(\text{相邻关系})
$$

对于二维网格，每个内部位置有4个相邻位置，边界位置2-3个。平均相邻度：
$$
\bar{d}_{2D} \approx 4
$$

展平后，相邻关系变为不规则：
$$
\bar{d}_{flat} \approx 2 \quad \text{(一维序列)}
$$

结构熵损失：
$$
\Delta H_{struct} = H_{struct}^{2D} - H_{struct}^{flat} > 0
$$

**推导9.2：RoPE-2D保留结构信息**

使用RoPE-2D，相对位置$(\Delta_x, \Delta_y)$直接编码：
$$
\boldsymbol{\mathcal{R}}_{\Delta_x, \Delta_y}
$$

相邻位置的编码：
$$
\begin{align}
\text{水平：} \quad &\boldsymbol{\mathcal{R}}_{0,1} \\
\text{垂直：} \quad &\boldsymbol{\mathcal{R}}_{1,0} \\
\text{对角：} \quad &\boldsymbol{\mathcal{R}}_{1,1}
\end{align}
$$

这三种相邻关系有不同的编码，结构信息得以保留：
$$
\Delta H_{struct}^{RoPE-2D} = 0
$$

### 10. 视觉-语言位置对齐的理论

**定义10.1：跨模态位置对齐**

视觉patch $(x, y)$与文本token $n$的对齐度：
$$
A(n, (x, y)) = \text{Sim}(\boldsymbol{p}_n^{text}, \boldsymbol{p}_{x,y}^{vision})
$$

其中$\text{Sim}$是相似度函数，如余弦相似度。

**推导10.1：展平策略的对齐**

使用展平策略，图像patch位置为：
$$
p_{x,y}^{flat} = L + \phi(x, y)
$$

文本token位置为：
$$
p_n^{text} = n
$$

位置编码：
$$
\boldsymbol{p}_{p}^{RoPE-1D} = \boldsymbol{\mathcal{R}}_p \boldsymbol{e}
$$

相似度：
$$
\text{Sim}(n, (x, y)) = \boldsymbol{e}^T \boldsymbol{\mathcal{R}}_n^T \boldsymbol{\mathcal{R}}_{L+\phi(x,y)} \boldsymbol{e} = \boldsymbol{e}^T \boldsymbol{\mathcal{R}}_{\phi(x,y)+L-n} \boldsymbol{e}
$$

取决于一维相对位置$\Delta = \phi(x,y) + L - n$。

**推导10.2：RoPE-Tie的对齐**

使用RoPE-Tie，文本位置$(n, n)$，图像位置$(sx+L, ty+L)$。

位置编码的内积：
$$
\boldsymbol{\mathcal{R}}_{n,n}^T \boldsymbol{\mathcal{R}}_{sx+L, ty+L} = \begin{pmatrix} \boldsymbol{\mathcal{R}}_{sx+L-n} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\mathcal{R}}_{ty+L-n} \end{pmatrix}
$$

相似度取决于二维相对位置$(sx+L-n, ty+L-n)$，保留了$x$和$y$的独立信息。

**定理10.1：RoPE-Tie提供更精细的对齐**

对于相同的一维相对距离$\Delta = sx + L - n = ty + L - n$（当$s=t$时），RoPE-Tie能区分不同的$(x, y)$组合，而展平策略不能。

例如：$(x_1, y_1)$和$(x_2, y_2)$满足$x_1 + y_1 = x_2 + y_2$，则在展平后（行优先，$s=t=1$）有相同的一维位置，但RoPE-Tie能区分它们（除非$x_1=x_2$且$y_1=y_2$）。$\square$

### 11. 多图像场景的位置编码

**定义11.1：多图像输入**

输入包含$M$张图像，第$k$张图像大小为$w_k \times h_k$。

**推导11.1：逐图像的位置分配**

使用[IMG]和[/IMG]分隔各图像：
$$
\begin{align}
&\text{Text}_1, \ldots, \text{Text}_{L_0} &&: (1, 1), \ldots, (L_0, L_0) \\
&\text{[IMG]}_1 &&: (L_0 + 1, L_0 + 1) \\
&\text{Image}_1 &&: (s_1 x + L_0 + 1, t_1 y + L_0 + 1), \quad x \in [1, h_1], y \in [1, w_1] \\
&\text{[/IMG]}_1 &&: (L_1, L_1) \\
&\text{Text between} &&: (L_1 + 1, L_1 + 1), \ldots \\
&\text{[IMG]}_2 &&: \ldots \\
&\vdots
\end{align}
$$

其中：
$$
L_k = L_{k-1} + 1 + (w_k + 1)(h_k + 1)
$$

**定理11.1：多图像的位置一致性**

每张图像独立应用相同的位置编码规则，保证：
1. 所有patch位置唯一
2. 每张图像的局部结构保持
3. 图像间不会位置冲突

**推导11.2：不同大小图像的处理**

图像$k$的缩放因子：
$$
s_k = w_k + 1, \quad t_k = h_k + 1
$$

不同图像有不同的缩放因子，但这不影响位置编码的有效性，因为：
1. 每张图像的patch之间使用相同的$s_k, t_k$
2. 跨图像的相对位置不需要特殊处理（由绝对位置差自然确定）

### 12. 位置编码的频谱特性

**定义12.1：位置编码的傅里叶表示**

RoPE可以看作傅里叶级数：
$$
\boldsymbol{\mathcal{R}}_m = \sum_{i=0}^{d/2-1} \alpha_i e^{im\theta_i}
$$

其中$\alpha_i$是系数，$e^{im\theta_i} = \cos(m\theta_i) + i\sin(m\theta_i)$。

**推导12.1：一维位置的频谱**

对于RoPE-1D，频率集合：
$$
\Theta_{1D} = \{\theta_i\}_{i=0}^{d/2-1}
$$

功率谱：
$$
P_{1D}(\theta) = \sum_{i=0}^{d/2-1} \delta(\theta - \theta_i)
$$

其中$\delta$是Dirac delta函数。

**推导12.2：二维位置的频谱**

对于RoPE-2D，频率集合分为两组：
$$
\Theta_x = \{\theta_i^x\}_{i=0}^{d/4-1}, \quad \Theta_y = \{\theta_j^y\}_{j=0}^{d/4-1}
$$

二维功率谱：
$$
P_{2D}(\theta_x, \theta_y) = \sum_{i,j} \delta(\theta_x - \theta_i^x) \delta(\theta_y - \theta_j^y)
$$

**定理12.1：RoPE-Tie的频谱完整性**

当$\Theta_x \cup \Theta_y = \Theta_{1D}$且$\Theta_x \cap \Theta_y = \emptyset$时，RoPE-Tie在$(n, n)$处的频谱等价于RoPE-1D在$n$处的频谱。

**证明**：
$$
\boldsymbol{\mathcal{R}}_{n,n}^{RoPE-Tie} = \text{diag}(\boldsymbol{\mathcal{R}}_n^{\Theta_x}, \boldsymbol{\mathcal{R}}_n^{\Theta_y})
$$

其包含的频率为$\Theta_x \cup \Theta_y = \Theta_{1D}$，与RoPE-1D相同。$\square$

### 13. 注意力机制下的位置交互

**定义13.1：位置相关的注意力分数**

Attention分数：
$$
s(i, j) = (\boldsymbol{\mathcal{R}}_i \boldsymbol{q})^T (\boldsymbol{\mathcal{R}}_j \boldsymbol{k}) = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_{j-i} \boldsymbol{k}
$$

对于RoPE-2D：
$$
s((x_1, y_1), (x_2, y_2)) = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_{x_2-x_1, y_2-y_1} \boldsymbol{k}
$$

**推导13.1：文本-图像注意力**

文本token $n$（位置$(n, n)$）对图像patch $(x, y)$（位置$(sx+L, ty+L)$）的注意力：
$$
s(n, (x,y)) = \boldsymbol{q}_n^T \boldsymbol{\mathcal{R}}_{sx+L-n, ty+L-n} \boldsymbol{k}_{x,y}
$$

分解为：
$$
s(n, (x,y)) = \boldsymbol{q}_n^T \begin{pmatrix} \boldsymbol{\mathcal{R}}_{sx+L-n}^{(d/2)} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{\mathcal{R}}_{ty+L-n}^{(d/2)} \end{pmatrix} \boldsymbol{k}_{x,y}
$$

**推导13.2：图像内部注意力**

同一图像内，patch $(x_1, y_1)$对patch $(x_2, y_2)$的注意力：
$$
s((x_1,y_1), (x_2,y_2)) = \boldsymbol{q}_{x_1,y_1}^T \boldsymbol{\mathcal{R}}_{s(x_2-x_1), t(y_2-y_1)} \boldsymbol{k}_{x_2,y_2}
$$

由于$s = w+1, t = h+1$，相对位置被缩放，但对于同一图像内的所有patch对，缩放是一致的，因此相对关系保持。

**定理13.1：局部注意力模式的保持**

对于图像内部的相邻patch（二维相邻），其注意力分数由相对位置$(\pm s, 0)$或$(0, \pm t)$决定。

由于$s, t$对于同一图像是常数，相邻patch的注意力模式一致，不受图像在序列中位置的影响。$\square$

### 14. 位置插值与多模态

**定义14.1：多模态中的位置插值**

当测试时图像大小变化，从训练的$w \times h$变为$w' \times h'$：
$$
w' > w, \quad h' > h
$$

位置插值策略：
$$
(x', y') \to \left(\frac{w}{w'} x', \frac{h}{h'} y'\right)
$$

**推导14.1：插值对RoPE-Tie的影响**

原始位置：$(sx + L, ty + L)$
插值后：$\left(\frac{w}{w'} s x + L, \frac{h}{h'} t y + L\right)$

如果$s = w+1, t = h+1$，则：
$$
s' = \frac{w}{w'}(w+1), \quad t' = \frac{h}{h'}(h+1)
$$

**定理14.1：插值保持相对关系**

对于同一图像内的任意两个patch $(x_1, y_1)$和$(x_2, y_2)$，插值后的相对位置：
$$
\Delta' = \left(\frac{w}{w'}s(x_2-x_1), \frac{h}{h'}t(y_2-y_1)\right)
$$

虽然绝对位置被缩放，但相对位置的比例关系保持：
$$
\frac{\Delta_x'}{\Delta_y'} = \frac{s(x_2-x_1)}{t(y_2-y_1)} \cdot \frac{h'}{w'} = \frac{(w+1)(x_2-x_1)}{(h+1)(y_2-y_1)} \cdot \frac{h'}{w'}
$$

如果$w' / h' \approx w / h$（保持宽高比），则比例关系近似保持。$\square$

### 15. 多模态Transformer的位置编码实现

**算法15.1：RoPE-Tie的前向计算**

输入：混合序列$\{x_i\}_{i=1}^N$，每个$x_i$带有位置标签$(p_1^{(i)}, p_2^{(i)})$

对于每个位置$i$：
1. 计算旋转矩阵：
   $$
   \boldsymbol{\mathcal{R}}_i = \text{diag}(\boldsymbol{\mathcal{R}}_{p_1^{(i)}}^{(\Theta_x)}, \boldsymbol{\mathcal{R}}_{p_2^{(i)}}^{(\Theta_y)})
   $$

2. 应用旋转：
   $$
   \tilde{\boldsymbol{q}}_i = \boldsymbol{\mathcal{R}}_i \boldsymbol{q}_i, \quad \tilde{\boldsymbol{k}}_i = \boldsymbol{\mathcal{R}}_i \boldsymbol{k}_i
   $$

3. 计算Attention：
   $$
   \alpha_{ij} = \frac{\exp(\tilde{\boldsymbol{q}}_i^T \tilde{\boldsymbol{k}}_j / \sqrt{d})}{\sum_k \exp(\tilde{\boldsymbol{q}}_i^T \tilde{\boldsymbol{k}}_k / \sqrt{d})}
   $$

**推导15.1：计算复杂度**

对于序列长度$N$：
- RoPE应用：$O(Nd)$（线性于序列长度和维度）
- Attention计算：$O(N^2 d)$（标准Attention复杂度）

RoPE-Tie相比RoPE-1D没有额外的复杂度增加（只是频率分配不同）。

**推导15.2：内存占用**

位置信息存储：
- RoPE-1D：每个位置一个标量，$O(N)$
- RoPE-Tie：每个位置两个标量，$O(2N) = O(N)$

预计算的旋转矩阵（如果使用Cache）：
- RoPE-1D：$O(L_{max} \cdot d)$
- RoPE-Tie：$O(L_{max}^2 \cdot d)$（因为二维位置$(x, y)$的组合数）

但实际上不需要缓存所有可能的二维位置，只需动态计算，因此内存占用仍为$O(Nd)$。

### 16. 跨模态位置编码的泛化能力

**定义16.1：位置编码的泛化**

训练时图像大小分布：$\mathcal{D}_{train} = \{(w, h)\}$
测试时图像大小：$(w_{test}, h_{test}) \notin \mathcal{D}_{train}$

泛化能力：模型在新大小图像上的性能不显著下降。

**定理16.1：RoPE-Tie的尺度不变性**

由于RoPE的旋转性质，位置编码满足：
$$
\boldsymbol{\mathcal{R}}_{\lambda x, \lambda y} = \text{diag}(\boldsymbol{\mathcal{R}}_{\lambda x}, \boldsymbol{\mathcal{R}}_{\lambda y})
$$

对于不同的缩放因子$\lambda$（即不同图像大小），只要相对位置$(\Delta_x, \Delta_y)$的模式相似，注意力模式也相似。

**推导16.1：训练多尺度图像的好处**

如果训练时包含多种尺寸$(w_1, h_1), (w_2, h_2), \ldots$，则模型学到的是：
$$
\text{Attention}(\Delta_x, \Delta_y) \text{ for various } (\Delta_x, \Delta_y)
$$

测试时遇到新尺寸，只要$(\Delta_x, \Delta_y)$在训练过的范围内（通过插值等手段），就能泛化。

**推导16.2：展平策略的泛化问题**

使用展平策略，图像大小变化会改变展平的映射：
$$
\phi_{w \times h}(x, y) = xw + y
$$

不同的$w$导致完全不同的一维位置序列，模型很难泛化。

而RoPE-Tie通过缩放保持了相对位置的结构，更易泛化。

### 17. 位置编码与注意力Mask的交互

**定义17.1：Causal Mask**

Decoder中使用下三角Mask：
$$
M_{ij} = \begin{cases}
0, & j \leq i \\
-\infty, & j > i
\end{cases}
$$

Attention分数：
$$
s'_{ij} = s_{ij} + M_{ij}
$$

**推导17.1：多模态中的Causal Mask**

对于文本token $i$（位置$p_i$）和图像patch $j$（位置$(x_j, y_j)$转为$p_j$）：

如果图像出现在文本之前（$p_j < p_i$），则：
$$
M_{ij} = 0 \quad \text{（可见）}
$$

如果图像出现在文本之后（$p_j > p_i$），则：
$$
M_{ij} = -\infty \quad \text{（不可见）}
$$

**推导17.2：图像内部的Mask**

图像是同时输入的（非自回归），因此图像内部所有patch相互可见：
$$
M_{(x_1,y_1), (x_2,y_2)} = 0, \quad \forall (x_1,y_1), (x_2,y_2) \in \text{same image}
$$

这与文本的逐token生成不同。

**定理17.1：位置编码与Mask的独立性**

位置编码和Attention Mask是独立的机制：
- 位置编码决定相对位置的表示
- Mask决定哪些位置对可以相互作用

两者可以独立设计和优化，不冲突。$\square$

### 18. 三维位置编码（视频）

**定义18.1：三维RoPE（RoPE-3D）**

对于视频，位置为$(t, x, y)$，其中$t$是时间维度。

三维旋转矩阵：
$$
\boldsymbol{\mathcal{R}}_{t,x,y}^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_t^{(d/3)}, \boldsymbol{\mathcal{R}}_x^{(d/3)}, \boldsymbol{\mathcal{R}}_y^{(d/3)})
$$

将$d$维向量分为三等份，分别编码三个维度。

**推导18.1：视频-文本混合输入**

文本token：$(n, n, n)$
视频patch：$(t, x, y)$

统一为三维位置表示。

**定理18.1：RoPE的任意维度推广**

对于$K$维位置$(p_1, p_2, \ldots, p_K)$：
$$
\boldsymbol{\mathcal{R}}_{p_1, \ldots, p_K}^{(d)} = \text{diag}(\boldsymbol{\mathcal{R}}_{p_1}^{(d_1)}, \ldots, \boldsymbol{\mathcal{R}}_{p_K}^{(d_K)})
$$

其中$\sum_{i=1}^K d_i = d$。

退化性质：当所有$p_i$相等，即$(n, n, \ldots, n)$，且频率适当分配时，等价于$d$维RoPE-1D。$\square$

### 19. 位置编码的学习与固定

**定义19.1：可学习vs固定位置编码**

**固定位置编码**（如RoPE）：
$$
\boldsymbol{p}_i = f(i; \boldsymbol{\theta}), \quad \boldsymbol{\theta} \text{ 固定}
$$

**可学习位置编码**：
$$
\boldsymbol{p}_i = \boldsymbol{E}_i, \quad \boldsymbol{E} \in \mathbb{R}^{L_{max} \times d} \text{ 可学习}
$$

**推导19.1：两者的对比**

| 特性 | 固定（RoPE） | 可学习 |
|------|-------------|--------|
| 长度泛化 | 好（可外推） | 差（固定$L_{max}$） |
| 参数量 | 0 | $O(L_{max} d)$ |
| 表达能力 | 有限（函数形式固定） | 强（任意模式） |
| 训练速度 | 快（无需更新） | 慢（额外参数） |

**推导19.2：混合策略**

结合两者优点：
$$
\boldsymbol{p}_i = f(i; \boldsymbol{\theta}) + \boldsymbol{E}_i
$$

其中$f$是固定的RoPE，$\boldsymbol{E}$是可学习的偏置。

**定理19.1：混合策略的优势**

混合策略结合了：
1. RoPE的长度泛化能力
2. 可学习编码的灵活性

在有限的训练长度内，可学习部分补偿RoPE的不足；超出训练长度时，RoPE提供基础的位置信息。$\square$

### 20. 实践中的超参数设置

**建议20.1：频率分配**

对于RoPE-Tie，$d = 128$的情况：
- $x$方向：使用$\theta_{0}, \theta_{2}, \ldots, \theta_{62}$（32个频率）
- $y$方向：使用$\theta_{1}, \theta_{3}, \ldots, \theta_{63}$（32个频率）

或者根据任务特点调整：
- 如果$x$方向（如高度）变化更重要：分配更多频率
- 如果$x, y$同等重要：均分

**建议20.2：缩放因子选择**

对于常见的图像尺寸（如224×224，分patch后14×14）：
$$
s = t = 15 \quad \text{（方案1：} w+1=h+1=15\text{）}
$$

或：
$$
s = t = \frac{197}{15} \approx 13.13 \quad \text{（方案2：} wh+1=197\text{）}
$$

方案1更简单，推荐使用。

**建议20.3：特殊token的使用**

是否使用[IMG]和[/IMG]取决于任务：
- **图像理解**（输入）：可选，主要为了标记图像边界
- **图像生成**（输出）：必须，用于指示生成开始和结束
- **图文交错**：推荐，便于区分不同模态

**建议20.4：实现细节**

```python
# 伪代码示例
def apply_rope_tie(q, k, positions, d):
    # positions: (batch, seq_len, 2) 表示 (p1, p2)
    d_half = d // 2

    # 分别处理 x 和 y 方向
    q_x, q_y = q[..., :d_half], q[..., d_half:]
    k_x, k_y = k[..., :d_half], k[..., d_half:]

    # 应用旋转
    p1, p2 = positions[..., 0], positions[..., 1]
    q_x_rot = apply_rope_1d(q_x, p1, freqs_x)
    q_y_rot = apply_rope_1d(q_y, p2, freqs_y)
    k_x_rot = apply_rope_1d(k_x, p1, freqs_x)
    k_y_rot = apply_rope_1d(k_y, p2, freqs_y)

    # 拼接
    q_rot = concat([q_x_rot, q_y_rot], dim=-1)
    k_rot = concat([k_x_rot, k_y_rot], dim=-1)

    return q_rot, k_rot
```

### 总结：多模态位置编码的核心原则

**原则1：保持模态内的结构**
- 文本：一维序列结构
- 图像：二维网格结构
- 视频：三维时空结构

**原则2：统一的数学框架**
- 使用轴向分解（直和）统一不同维度
- 通过频率分配实现维度退化

**原则3：可扩展性**
- 易于推广到3D、4D或更高维
- 支持变长、变尺寸输入

**原则4：计算效率**
- 复杂度不超过标准Attention
- 与Flash Attention等优化技术兼容

RoPE-Tie作为一个统一多模态位置编码的方案，在理论和实践中都展现了良好的特性，值得在多模态大模型中进一步探索和应用。

