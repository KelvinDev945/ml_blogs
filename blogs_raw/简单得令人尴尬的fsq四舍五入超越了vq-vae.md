---
title: 简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE
slug: 简单得令人尴尬的fsq四舍五入超越了vq-vae
date: 2023-10-31
tags: 详细推导, 生成模型, 编码, 梯度, 离散化, 生成模型
status: pending
---
# 简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE

**原文链接**: [https://spaces.ac.cn/archives/9826](https://spaces.ac.cn/archives/9826)

**发布日期**: 

---

正如“XXX is all you need”一样，有不少论文都以“简单得令人尴尬”命名（An Embarrassingly Simple XXX），但在笔者看来，这些论文大多数都是噱头多于实力。不过，笔者最近阅读到的一篇论文，真的让人不由得发出“简单得令人尴尬”的感叹～

论文的标题是[《Finite Scalar Quantization: VQ-VAE Made Simple》](https://papers.cool/arxiv/2309.15505)，顾名思义，这是一篇旨在用FSQ（Finite Scalar Quantization）简化VQ-VAE的工作。随着生成模型、多模态LLM的逐渐流行，VQ-VAE及其后续工作也作为“图像的Tokenizer”而“水涨船高”。然而，VQ-VAE的训练本身也存在一些问题，而FSQ这篇论文则声称通过更简单的“四舍五入”就可以达到同样的目的，并且有着效果更好、收敛更快、训练更稳的优点。

FSQ真有这么神奇？接下来我们一起学习一下。

## VQ #

首先，我们来了解一下“VQ”。VQ全称是“Vector Quantize”，可以翻译为“向量量子化”或者“向量量化”，是指将无限、连续的编码向量映射为有限、离散的整数数字的一种技术。如果我们将VQ应用在自编码器的中间层，那么可以在压缩输入大小的同时，让编码结果成为一个离散的整数序列。

假设自编码器的重构损失能够让我们满意，那么这个整数序列就是原始图像的等价物，所有关于原始图像的操作都可以转化为整数序列上的操作。比如我们想训练图像生成模型，就只需要训练整数序列生成模型，而这跟本文生成等价，所以我们可以用它来训练一个GPT，模型和流程都跟文本一模一样，训练完成后，我们就可以从GPT模型中采样整数序列，然后送到解码器中得到图像，从而完完成了图像生成模型的构建。说白了，“VQ+自编码器”将任意输入都转化为跟文本一致的整数序列，统一了不同模态数据的输入形式，同时也统一了它们的处理和生成模型。

而这样的一个带有VQ功能的自编码器，就被称为“VQ-VAE”。

## AE #

早在四年前的文章[《VQ-VAE的简明介绍：量子化自编码器》](/archives/6760)中我们就介绍过了VQ-VAE，尽管被冠以“VAE（Variational AutoEncoder）”之名，但它实际上跟VAE没啥关系，如上一节所说，它只是一个带有VQ功能的AE（AutoEncoder）。

既然是AE，那么有encoder和decoder，一个普通的AE是这样的：  
\begin{equation}z = encoder(x),\quad \hat{x}=decoder(z),\quad \mathcal{L}=\Vert x - \hat{x}\Vert^2 \end{equation}  
VQ-VAE则稍微复杂一些：  
\begin{equation}\begin{aligned}  
z =&\, encoder(x)\\\\[5pt]  
z_q =&\, z + \text{sg}[e_k - z],\quad k = \mathop{\text{argmin}}_{i\in\\{1,2,\cdots,K\\}} \Vert z - e_i\Vert\\\  
\hat{x} =&\, decoder(z_q)\\\\[5pt]  
\mathcal{L} =&\, \Vert x - \hat{x}\Vert^2 + \beta\Vert e_k - \text{sg}[z]\Vert^2 + \gamma\Vert z - \text{sg}[e_k]\Vert^2  
\end{aligned}\label{eq:vqvae}\end{equation}

让我们来逐步解释一下。首先，第一步是相同的，输入$x$到encoder中，输出编码向量$z$。然而，我们并不是直接将$z$输入到decoder中，而是先维护一个编码向量表$\\{e_1,e_2,\cdots,e_K\\}$（Codebook），从中选取与$z$最相近的一个$e_k$送入到decoder中进行重构$x$。由于编码表是有限的，所以我们也可以将实际的编码结果理解为一个整数（即与$z$最相近的$e_k$的$k$），这就是VQ-VAE中“VQ”的含义。

当然，实际应用中，为了保证重构的清晰度，encoder的输出可能是多个向量，每个向量经历同样的量化步骤变成一个整数，所以结果就是一张原本在连续实数空间的图片，被编码VQ-VAE编码为了一个整数的序列，这跟文本Tokenizer的作用是类似的，所以就有了“图像的Tokenizer”的说法。

## 梯度 #

然而，由于整个前向计算的流程中出现了$\mathop{\text{argmin}}$，所以梯度无法回传到encoder，这意味着我们无法优化encoder。此时常见的手段是[Gumbel Softmax](/archives/6705)，但Gumbel Softmax的效果通常也是次优的，所以作者巧妙地借助了Straight-Through为VQ-VAE设计了更好的梯度。可以说，这是VQ-VAE最精彩的内容，它告诉我们什么“Attention is all you need”都是虚的，“Gradient”才是真正的“all we need”！

具体来说，VQ-VAE利用了深度学习框架基本上都自带的stop_gradient（即公式中的$\text{sg}$）函数来自定义梯度，所有经过$\text{sg}$的输入，都会保持同样的输出，但梯度被强迫为零。所以对于式$\eqref{eq:vqvae}$中的$z_q$，我们有  
\begin{equation}z_q = e_k,\quad \nabla z_q = \nabla z\label{eq:sg}\end{equation}  
这样一来，送入decoder的还是量化过后的$e_k$，但优化器求梯度时用的是$z$，而$z$是encoder出来的，所以encoder也能够被优化了。这个操作就叫做“Straight-Through Estimator（STE）”，是为神经网络的不可导模块设计梯度的常用技巧之一。

由于基于梯度的优化器依然是当前的主流，所以直接设计梯度往往比设计loss更贴近本质，当然通常也更难、更让人由衷地赞叹。

## Loss #

不顾，事情还没完，此时有两个问题：1、现在encoder是有梯度了，但是编码表$e_1,e_2,\cdots,e_K$却没了梯度；2、$\text{sg}$虽然可以随意定义梯度，但不是胡乱定义一个梯度都可以成功优化模型的。从$\eqref{eq:sg}$可以看成，要使它在数学上严格成立，那么唯一的解是$e_k=z$，这告诉我们如果STE是合理的，那么$e_k$与$z$至少是相近的。于是为了梯度的合理性，同时也为了优化编码表，我们还可以补充一项辅助loss：  
\begin{equation}\Vert e_k - z\Vert^2\label{eq:ez}\end{equation}  
这样既可以迫使$e_k$与$z$接近，又可以让$e_k$也拥有了梯度，一举两得！但细想之下，还是有点美中不足：理论上encoder和decoder的重构loss已经足够优化$z$了，所以额外引入的一项应该主要用来优化$e_k$，而不应该反过来明显影响$z$。为此，我们再次利用$\text{sg}$技巧，不难证明式$\eqref{eq:ez}$的梯度等价于  
\begin{equation}\Vert e_k - \text{sg}[z]\Vert^2 + \Vert z - \text{sg}[e_k]\Vert^2\end{equation}  
第一项把$z$的梯度停掉了，剩下$e_k$的梯度，第二项则反过来，目前两项是$1:1$的权重求和，意味着两项相同程度地相互影响，而刚才我们说了，这辅助loss应该主要用来优化$e_k$而不是$z$，所以我们引入$\beta > \gamma > 0$，将辅助loss改为  
\begin{equation}\beta\Vert e_k - \text{sg}[z]\Vert^2 + \gamma\Vert z - \text{sg}[e_k]\Vert^2\label{eq:ez2}\end{equation}  
然后再加到重构loss中，就得到了VQ-VAE总的loss了

除此之外，$e_k$的优化还有另外的方案：首先将式$\eqref{eq:ez2}$的$\beta$置零，这样一来$e_k$就又没有梯度了；然后我们观察到，VQ-VAE的VQ操作其实跟K-Means聚类是有点相似的，$e_1,e_2,\cdots,e_K$相当于是$K$个聚类中心。根据我们对K-Means的了解，聚类中心等于该类的所有向量的平均，所以$e_k$的一种优化方案就是$z$的滑动平均  
\begin{equation}e_k^{(t)} = \alpha e_k^{(t-1)} + (1-\alpha) z \end{equation}  
这等价于指定使用SGD优化$\Vert e_k - \text{sg}[z]\Vert^2$这一项loss（其他项可以用Adam等）。该方案被[VQ-VAE-2](https://papers.cool/arxiv/1906.00446)所使用。

## FSQ #

可能有些读者疑惑，本文的主题不是FSQ吗？前面介绍VQ-VAE的篇幅是不是有点多了？事实上，由于FSQ完全对得起“简单得令人尴尬”这个评价，相比VQ-VAE，介绍FSQ只需要“寥寥几行”，所以VQ-VAE不写长点，这篇博客就没几个字了哈～当然，将VQ-VAE写详细一点，也能让大家更深刻体会到FSQ的简单。

准确来说，FSQ只是用来替代VQ-VAE中的“VQ”的，它的离散化思路非常非常简单，就是“四舍五入”。首先，假设我们有一个标量$t\in\mathbb{R}$，我们定义：  
\begin{equation}\text{FSQ}(t)\triangleq \text{Round}[(L-1)\sigma(t)] \end{equation}  
这里的$L\in\mathbb{N}$是一个超参数，$\sigma(x)=1/(1+e^{-x})$就是sigmoid函数（原论文用了$\tanh$，笔者认为用sigmoid更科学），$\text{Round}$就是四舍五入为一个整数，所以不难看出$\text{FSQ}(t)\in\\{0,1,\cdots,L-1\\}$，即FSQ运算将输出限制在了$L$个整数之中，从而实现了离散化。当然，多数情况下一个标量还不够，对于$z\in\mathbb{R}^d$，每一维可以执行FSQ运行，于是  
\begin{equation}\text{FSQ}(z) = \text{Round}[(L-1)\sigma(z)]\in\\{0,1,\cdots,L-1\\}^d \end{equation}  
即$d$维向量$z$被离散为$L^d$个整数之一。但要注意，$\text{Round}$操作同样是没有梯度的（或者说梯度为零），不过经过VQ-VAE的铺垫，有些读者可能已经猜到接下来要做什么了：同样是利用STE技巧  
\begin{equation}\text{FSQ}(z) = (L-1)\sigma(z) + \text{sg}\big[\text{Round}[(L-1)\sigma(z)] - (L-1)\sigma(z)\big] \end{equation}  
即反向传播用$\text{Round}$之前的$(L-1)\sigma(z)$求梯度。由于$\text{Round}$前后本身是数值近似的，所以FSQ不需要额外loss来迫使近似的出现，也没有额外的编码表需要更新，FSQ的简洁可见一斑！

[![VQ与FSQ对比（来自原论文）](/usr/uploads/2023/10/1903966725.png)](/usr/uploads/2023/10/1903966725.png "点击查看原图")

VQ与FSQ对比（来自原论文）

## 实验 #

如果将VQ理解为直接将编码向量聚类为$K$个不同的类别，那么FSQ就是将编码向量归纳出$d$个属性，每个属性划分为了$L$个等级，从而直接表达了$L^d$个不同的整数。当然，从最一般的考虑，每个属性的等级数也可以是不相同的$L_1,L_2,\cdots,L_d$，从而不同的组合数为$L_1 L_2\cdots L_d$。

按照原论文的建议$L\geq 5$（之前的[LFQ](https://papers.cool/arxiv/2310.05737)则相当于$L=2$），所以如果要对齐VQ-VAE的编码数量$K$的话，对于FSQ来说应该有$d = \log_L K$，即FSQ对编码向量的维度$d$是有限制的（一般就只是个位数），并且通常是远小于VQ-VAE的编码维度（一般是三位数），这个直接后果是当编码总数$K$比较小（从而$d$也比较小）时，FSQ的效果通常不如VQ：  


[![不同编码表大小下VQ与FSQ的效果差异](/usr/uploads/2023/10/997991949.png)](/usr/uploads/2023/10/997991949.png "点击查看原图")

不同编码表大小下VQ与FSQ的效果差异

从图上可以看到，当编码表大小在1000左右时，FSQ与VQ的效果接近；当编码表大小明显超过1000时，FSQ占优；反之当编码表大小明显小于1000时，则VQ占优，这跟笔者自己的实验结果相近。笔者的参考代码为：

> **Github:<https://github.com/bojone/FSQ>**

其他实验就是比较常规的证明FSQ比VQ更优异的各种任务实验了，读者自行阅读原论文就好。

## 思考 #

从形式上来看，假设$K=L^d$，那么VQ就好比是“一个$L^d$类的分类器”，而FSQ则是“$d$个$L$级的打分器”，不管是从参数量、几何直观或者表达能力来看，其实FSQ都不如VQ，但为什么FSQ有机会取得比VQ更好的结果呢？笔者认为有两方面的原因。

第一个原因，是encoder和decoder太强。虽然FSQ本身弱一些，但是encoder和decoder都足够强了，所以基于神经网络的万能拟合能力假设，FSQ相对于VQ的劣势，完全可以在encoder和decoder中弥补过来。而在$K=L^d$的设定下，两者的离散化程度都是一样的，也就是说encoder与decoder之间的“信息瓶颈”是一样的，因此FSQ本身的问题就显得微不足道了。

第二个原因，是VQ的“队友”（梯度）太弱。VQ的经典问题是编码表坍缩：当编码表增大时，编码表并没有被充分利用起来，反而由于恶性竞争导致编码表聚集到一块了，经典表现就是一个5000的编码表，最终效果还不如500的编码表。归根结底，这是梯度不够合理所致，尽管VQ已经巧妙地设计了梯度，但对于$\mathop{\text{argmin}}$这种硬指派的运算，基于梯度的优化都存在“赢者通吃”问题，这是坍缩的根本原因，而FSQ的$\text{Round}$运算并不涉及到指派，它是直接取的近似值。当然，往大了讲，其实跟VQ类似的K-Means经常也有聚类中心坍缩的问题，可见$\mathop{\text{argmin}}$难优化已经是一个老大难的问题了。因此与其说FSQ太强，倒不如说是VQ的“队友”太弱。

从以上两点分析可以看出，FSQ要想超过VQ，除了编码表要足够大之外，还有encoder与decoder要足够复杂，但这并非总能满足，比如有些场景下，我们希望模型的每一层输出都被量化，这时候平摊下来的encoder和decoder未必足够复杂，此时FSQ本身的不足就成为效果的瓶颈了。此外，VQ之后的向量维度没有变化，可以是任意多维，而FSQ之前的向量必须投影到$d = \log_L K$维，这是很严重的降维，当我们需要用到投影之前的高维度近似向量时，就很难靠FSQ之后的低维向量简单恢复过来。

所以，如果单纯是作为“图像的Tokenzier”，那么FSQ或许已经可以取代VQ，但这并不意味着任意场景下VQ都可以被FSQ取代。

## 小结 #

本文介绍了VQ-VAE的“VQ”的一个及其简单的替代品——FSQ（Finite Scalar Quantization），它直接通过四舍五入来对连续向量进行离散化，并且不需要额外的loss进行辅助。实验结果表明，当编码表足够大时，FSQ比VQ更有优势。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9826>_

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

苏剑林. (Oct. 31, 2023). 《简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9826>

@online{kexuefm-9826,  
title={简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE},  
author={苏剑林},  
year={2023},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/9826}},  
} 


---

## 公式推导与注释

### 1. FSQ的数学定义与基本性质

**定义1.1（FSQ量化函数）**：对于标量 $t \in \mathbb{R}$，FSQ量化函数定义为：

$$
\text{FSQ}(t) = \text{Round}[(L-1)\sigma(t)]
$$

其中：
- $L \in \mathbb{N}$ 是量化等级数（超参数）
- $\sigma(x) = \frac{1}{1+e^{-x}}$ 是sigmoid激活函数
- $\text{Round}(\cdot)$ 是标准四舍五入函数

**性质1.1（值域性质）**：FSQ函数的值域为有限离散集合：

$$
\text{FSQ}(t) \in \{0, 1, 2, \ldots, L-1\}
$$

**证明**：由于 $\sigma(t) \in (0, 1)$，因此：

$$
(L-1)\sigma(t) \in (0, L-1)
$$

应用四舍五入操作后：

$$
\text{Round}[(L-1)\sigma(t)] \in \{0, 1, \ldots, L-1\}
$$

这确保了输出是 $L$ 个离散整数值之一。$\square$

### 2. 向量化的FSQ定义

**定义2.1（向量FSQ）**：对于向量 $z \in \mathbb{R}^d$，向量化FSQ定义为逐元素操作：

$$
\text{FSQ}(z) = [\text{FSQ}(z_1), \text{FSQ}(z_2), \ldots, \text{FSQ}(z_d)]^T
$$

其中 $z_i$ 是向量 $z$ 的第 $i$ 个分量。

**性质2.1（编码空间大小）**：向量FSQ的输出空间大小为：

$$
|\text{CodeSpace}| = L^d
$$

**证明**：每个维度有 $L$ 个可能的取值，共 $d$ 个独立维度，因此总的编码数量为：

$$
L \times L \times \cdots \times L = L^d
$$

这与VQ-VAE的编码表大小 $K$ 可以对应，即当 $L^d = K$ 时，两者的离散化程度相同。$\square$

### 3. FSQ的梯度分析：Straight-Through Estimator

**定义3.1（FSQ的前向传播）**：在前向传播中，FSQ的输出为：

$$
z_q = \text{Round}[(L-1)\sigma(z)]
$$

**定义3.2（FSQ的反向传播）**：在反向传播中，我们使用Straight-Through Estimator（STE）技巧：

$$
z_q = (L-1)\sigma(z) + \text{sg}[\text{Round}[(L-1)\sigma(z)] - (L-1)\sigma(z)]
$$

其中 $\text{sg}[\cdot]$ 表示stop_gradient操作。

**性质3.1（STE梯度）**：使用STE后，FSQ的梯度为：

$$
\frac{\partial z_q}{\partial z} = (L-1) \cdot \sigma'(z)
$$

**详细推导**：

令 $h(z) = (L-1)\sigma(z)$，$q(z) = \text{Round}[h(z)]$

根据STE的定义：
$$
z_q = h(z) + \text{sg}[q(z) - h(z)]
$$

在反向传播时，$\text{sg}[\cdot]$ 内的项梯度为0，因此：

$$
\frac{\partial z_q}{\partial z} = \frac{\partial h(z)}{\partial z} = (L-1) \frac{d\sigma(z)}{dz}
$$

而sigmoid函数的导数为：

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

因此：

$$
\frac{\partial z_q}{\partial z} = (L-1) \sigma(z)(1-\sigma(z))
$$

这个梯度是连续且可微的，能够有效地传播到encoder。$\square$

### 4. FSQ的近似误差分析

**定义4.1（量化误差）**：FSQ的量化误差定义为：

$$
\epsilon(z) = \text{Round}[h(z)] - h(z)
$$

其中 $h(z) = (L-1)\sigma(z)$。

**性质4.1（误差上界）**：量化误差满足：

$$
|\epsilon(z)| \leq \frac{1}{2}
$$

**证明**：四舍五入操作的性质保证了：

$$
|x - \text{Round}(x)| \leq \frac{1}{2}, \quad \forall x \in \mathbb{R}
$$

因此：

$$
|\epsilon(z)| = |\text{Round}[h(z)] - h(z)| \leq \frac{1}{2}
$$

这个上界是紧的（tight）。$\square$

**性质4.2（相对误差）**：当 $L$ 较大时，相对量化误差为：

$$
\frac{|\epsilon(z)|}{h(z)} \leq \frac{1}{2h(z)} = \frac{1}{2(L-1)\sigma(z)}
$$

当 $L$ 增大时，相对误差减小，量化精度提高。

### 5. FSQ与VQ-VAE的理论对比

**定义5.1（VQ-VAE的量化操作）**：在VQ-VAE中，量化定义为：

$$
z_q^{VQ} = e_k, \quad k = \arg\min_{i \in \{1,\ldots,K\}} \|z - e_i\|_2
$$

其中 $\{e_1, \ldots, e_K\}$ 是编码表（codebook）。

**定义5.2（FSQ的等价表示）**：FSQ可以看作隐式定义了一个编码表：

$$
\mathcal{E}_{FSQ} = \{[i_1, i_2, \ldots, i_d]^T : i_j \in \{0,1,\ldots,L-1\}, j=1,\ldots,d\}
$$

编码表大小 $|\mathcal{E}_{FSQ}| = L^d$。

**定理5.1（参数数量对比）**：

- VQ-VAE的编码表参数数量：$K \times d$
- FSQ的参数数量：$0$（不需要显式编码表）

当 $K = L^d$ 时，VQ-VAE需要 $L^d \times d$ 个参数，而FSQ不需要额外参数。

**证明**：VQ-VAE的编码表 $\{e_1, \ldots, e_K\}$ 中，每个 $e_i \in \mathbb{R}^d$ 是需要学习的参数，共 $K \times d$ 个参数。

FSQ的量化规则由公式 $\text{Round}[(L-1)\sigma(z)]$ 完全确定，不需要学习任何额外参数。这是FSQ相比VQ-VAE的重要优势。$\square$

### 6. FSQ的梯度传播完整分析

考虑一个完整的自编码器流程：

$$
x \xrightarrow{\text{encoder}} z \xrightarrow{\text{FSQ}} z_q \xrightarrow{\text{decoder}} \hat{x}
$$

损失函数为重构损失：

$$
\mathcal{L} = \|x - \hat{x}\|^2
$$

**定理6.1（FSQ的梯度链式法则）**：encoder参数 $\theta_e$ 的梯度为：

$$
\frac{\partial \mathcal{L}}{\partial \theta_e} = \frac{\partial \mathcal{L}}{\partial \hat{x}} \frac{\partial \hat{x}}{\partial z_q} \frac{\partial z_q}{\partial z} \frac{\partial z}{\partial \theta_e}
$$

**详细推导**：

第一项（重构损失对输出的梯度）：

$$
\frac{\partial \mathcal{L}}{\partial \hat{x}} = 2(x - \hat{x})^T
$$

第二项（decoder的梯度）：

$$
\frac{\partial \hat{x}}{\partial z_q} = J_{decoder}(z_q)
$$

其中 $J_{decoder}$ 是decoder的雅可比矩阵。

第三项（FSQ的STE梯度）：

$$
\frac{\partial z_q}{\partial z} = (L-1) \cdot \text{diag}[\sigma'(z_1), \ldots, \sigma'(z_d)]
$$

第四项（encoder的梯度）：

$$
\frac{\partial z}{\partial \theta_e} = J_{encoder}(x, \theta_e)
$$

将这些项相乘，得到完整的梯度链。关键是第三项的STE梯度是连续的，使得梯度能够顺畅地反向传播。$\square$

### 7. FSQ的STE合理性分析

**定理7.1（STE的近似性质）**：STE假设在数学上等价于近似假设：

$$
\text{Round}[h(z)] \approx h(z)
$$

这个近似在 $h(z)$ 接近整数时误差最小。

**推导**：根据STE定义：

$$
z_q = h(z) + \text{sg}[\text{Round}[h(z)] - h(z)]
$$

在前向传播中：$z_q = \text{Round}[h(z)]$

在反向传播中：梯度为 $\nabla z_q = \nabla h(z)$

这等价于假设 $\text{Round}[h(z)] = h(z)$，即：

$$
\text{Round}[h(z)] - h(z) = 0
$$

实际上 $|\text{Round}[h(z)] - h(z)| \leq 0.5$，因此STE引入的近似误差是有界的。

**性质7.1（STE误差的期望）**：假设 $h(z)$ 在区间 $[n, n+1)$ 上均匀分布（$n$ 是整数），则：

$$
\mathbb{E}[\text{Round}[h(z)] - h(z)] = 0
$$

**证明**：在区间 $[n, n+0.5)$ 上，$\text{Round}[h(z)] = n$；在 $[n+0.5, n+1)$ 上，$\text{Round}[h(z)] = n+1$。

$$
\mathbb{E}[\epsilon] = \int_n^{n+0.5} (n - x) dx + \int_{n+0.5}^{n+1} (n+1 - x) dx
$$

$$
= \left[-\frac{(x-n)^2}{2}\right]_n^{n+0.5} + \left[-\frac{(x-n-1)^2}{2}\right]_{n+0.5}^{n+1}
$$

$$
= -\frac{0.25}{2} + (-\frac{0}{2} + \frac{0.25}{2}) = 0
$$

因此STE在期望意义下是无偏的。$\square$

### 8. FSQ无需辅助损失的理论依据

**定理8.1（FSQ的自洽性）**：FSQ不需要额外的辅助损失，因为量化操作本身保证了：

$$
\|z_q - z\|_2 = \|\text{Round}[h(z)] - h(z)\|_2 \leq \frac{\sqrt{d}}{2}
$$

这个误差上界不依赖于可学习参数。

**对比VQ-VAE**：VQ-VAE需要辅助损失：

$$
\mathcal{L}_{aux} = \beta \|e_k - \text{sg}[z]\|^2 + \gamma \|z - \text{sg}[e_k]\|^2
$$

原因是：
1. VQ-VAE的 $e_k$ 是可学习的，需要损失来优化
2. $e_k$ 与 $z$ 之间的距离无先验保证，可能很大

**FSQ的优势**：

1. $\text{Round}[h(z)]$ 自动接近 $h(z)$（误差 $\leq 0.5$）
2. sigmoid函数 $\sigma(z)$ 自动将输出限制在 $(0, 1)$ 范围内
3. 不需要学习编码表，避免了编码表坍缩问题

### 9. FSQ的表达能力分析

**定义9.1（表达能力）**：模型的表达能力定义为其能够表示的不同编码的数量。

对于FSQ：$\text{Capacity}_{FSQ} = L^d$

对于VQ-VAE：$\text{Capacity}_{VQ} = K$

**定理9.1（等容量对比）**：当 $L^d = K$ 时，FSQ和VQ的离散化程度相同，但：

- FSQ的连续嵌入维度：$d$
- VQ的连续嵌入维度：$d_{VQ}$（通常 $d_{VQ} \gg d$）

**推导**：设 $K = L^d$，则：

$$
d = \frac{\log K}{\log L}
$$

例如，若 $K = 8192, L = 8$，则：

$$
d = \frac{\log 8192}{\log 8} = \frac{13}{3} \approx 4.33 \approx 4 \text{ 或 } 5
$$

而VQ-VAE通常使用 $d_{VQ} = 256$ 或更高。

**性质9.1（维度与表达的权衡）**：

FSQ的低维度 $d$ 意味着：
- 优势：参数效率高，计算快
- 劣势：连续空间的表达能力可能受限

这需要依靠encoder和decoder的强大能力来补偿。

### 10. FSQ的几何解释

**定义10.1（FSQ的量化网格）**：FSQ将连续空间 $\mathbb{R}^d$ 划分为网格：

$$
\mathcal{G} = \{[i_1, i_2, \ldots, i_d]^T : i_j \in \{0, 1, \ldots, L-1\}\}
$$

每个网格点对应一个编码。

**定理10.1（Voronoi分解）**：经过 $\sigma$ 和线性变换后，FSQ在原始 $z$ 空间定义了一个非均匀的Voronoi分解。

具体来说，编码 $[i_1, \ldots, i_d]^T$ 对应的Voronoi区域为：

$$
V_{i_1,\ldots,i_d} = \left\{z \in \mathbb{R}^d : \text{Round}[(L-1)\sigma(z_j)] = i_j, \forall j\right\}
$$

**性质10.1（区域形状）**：每个Voronoi区域是：

$$
V_{i_1,\ldots,i_d} = \prod_{j=1}^d \left\{z_j : \frac{i_j - 0.5}{L-1} \leq \sigma(z_j) < \frac{i_j + 0.5}{L-1}\right\}
$$

（边界情况需要特殊处理）

这些区域在原始 $z$ 空间是由sigmoid函数的反函数（logit函数）定义的超矩形：

$$
z_j \in \left[\text{logit}\left(\frac{i_j - 0.5}{L-1}\right), \text{logit}\left(\frac{i_j + 0.5}{L-1}\right)\right)
$$

其中 $\text{logit}(p) = \log\frac{p}{1-p}$。

### 11. FSQ的编码分布分析

**定义11.1（编码熵）**：编码的信息熵定义为：

$$
H = -\sum_{c \in \mathcal{C}} p(c) \log p(c)
$$

其中 $\mathcal{C}$ 是所有可能编码的集合，$p(c)$ 是编码 $c$ 的概率。

**定理11.1（最大熵）**：当编码均匀分布时，熵达到最大值：

$$
H_{max} = \log L^d = d \log L
$$

**FSQ的编码分布**：

假设 encoder 输出 $z$ 的分布为 $p(z)$，则编码 $c$ 的概率为：

$$
p(c) = \int_{V_c} p(z) dz
$$

其中 $V_c$ 是编码 $c$ 对应的Voronoi区域。

**性质11.1（均匀化趋势）**：当 encoder 输出 $z$ 的分布接近均匀时，FSQ自然地产生均匀的编码分布，不需要像VQ-VAE那样显式地优化编码表利用率。

### 12. FSQ的训练稳定性分析

**定理12.1（梯度范数上界）**：FSQ的梯度范数有上界：

$$
\left\|\frac{\partial z_q}{\partial z}\right\|_F = (L-1) \sqrt{\sum_{i=1}^d [\sigma'(z_i)]^2}
$$

由于 $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$，因此：

$$
\left\|\frac{\partial z_q}{\partial z}\right\|_F \leq \frac{(L-1)\sqrt{d}}{4}
$$

**推导**：

Frobenius范数的计算：

$$
\left\|\frac{\partial z_q}{\partial z}\right\|_F^2 = \sum_{i=1}^d \left[\frac{\partial z_{q,i}}{\partial z_i}\right]^2 = (L-1)^2 \sum_{i=1}^d [\sigma'(z_i)]^2
$$

由于 $\sigma'(z)$ 的最大值为 $0.25$（在 $z=0$ 处取得），因此：

$$
\left\|\frac{\partial z_q}{\partial z}\right\|_F \leq (L-1) \cdot \sqrt{d} \cdot 0.25 = \frac{(L-1)\sqrt{d}}{4}
$$

这个上界保证了梯度不会爆炸。$\square$

**定理12.2（梯度消失分析）**：当 $|z_i|$ 很大时，$\sigma'(z_i) \to 0$，可能导致梯度消失。

具体地：
- 当 $z \to +\infty$ 时，$\sigma(z) \to 1$，$\sigma'(z) \to 0$
- 当 $z \to -\infty$ 时，$\sigma(z) \to 0$，$\sigma'(z) \to 0$

**解决方案**：适当的网络初始化和归一化层可以保持 $z$ 在合理范围内，避免梯度消失。

### 13. FSQ与VQ的编码表坍缩对比

**定义13.1（编码表坍缩）**：在VQ-VAE中，编码表坍缩指大部分编码向量 $e_i$ 未被使用的现象。

**度量13.1（编码利用率）**：

$$
\text{Usage} = \frac{\text{实际使用的编码数}}{K}
$$

VQ-VAE常见问题：即使 $K = 8192$，实际利用率可能只有 $10\%$ 或更低。

**定理13.1（FSQ无坍缩）**：FSQ理论上不存在编码表坍缩问题。

**证明**：FSQ的编码空间是完全确定的网格 $\{0,1,\ldots,L-1\}^d$，每个编码都对应一个固定的Voronoi区域。只要训练数据足够多样化，所有编码都有机会被使用。

相比之下，VQ-VAE的编码表 $\{e_1, \ldots, e_K\}$ 是通过梯度学习的，存在"赢者通吃"效应：
- 某些 $e_i$ 吸引了大量 $z$
- 其他 $e_j$ 很少或从不被选中
- 这些 $e_j$ 的梯度很小，难以更新

FSQ避免了这个问题，因为它不需要学习编码表。$\square$

### 14. FSQ的计算复杂度分析

**定理14.1（前向传播复杂度）**：

- FSQ：$O(d)$
- VQ-VAE：$O(Kd)$

**证明**：

FSQ的前向传播：
1. 计算 $\sigma(z)$：$O(d)$
2. 计算 $(L-1)\sigma(z)$：$O(d)$
3. 计算 $\text{Round}$：$O(d)$
4. 总计：$O(d)$

VQ-VAE的前向传播：
1. 计算 $\|z - e_i\|$ 对所有 $i$：$O(Kd)$
2. 找最小距离：$O(K)$
3. 总计：$O(Kd)$

当 $K = L^d$ 很大时，FSQ的计算优势明显。$\square$

**定理14.2（反向传播复杂度）**：

- FSQ：$O(d)$
- VQ-VAE：$O(d)$（使用STE时）

两者在反向传播上复杂度相当，但FSQ不需要额外的编码表梯度计算。

### 15. FSQ的信息论视角

**定义15.1（编码的互信息）**：编码 $c$ 与原始输入 $x$ 的互信息：

$$
I(X; C) = H(C) - H(C|X)
$$

其中 $H(C)$ 是编码的熵，$H(C|X)$ 是给定输入时编码的条件熵。

**定理15.1（信息瓶颈）**：FSQ和VQ都实现了信息瓶颈：

$$
I(X; C) \leq H(C) \leq \log L^d = d \log L
$$

这个上界由编码空间的大小决定。

**推导**：

由于 $C$ 是确定性地由 $X$ 通过 encoder 和量化得到的：

$$
H(C|X) = 0
$$

因此：

$$
I(X; C) = H(C)
$$

而 $H(C)$ 的上界是均匀分布时的熵：

$$
H(C) \leq \log L^d
$$

这说明编码的信息容量受限于 $d \log L$。$\square$

### 16. 不同L值对FSQ性能的影响

**定理16.1（量化精度与L的关系）**：较大的 $L$ 提供更精细的量化：

$$
\Delta = \frac{1}{L-1}
$$

这是相邻量化级别在归一化空间 $[0, 1]$ 中的间距。

**分析不同L值**：

1. **$L = 2$**（二值化）：
   - 编码空间：$\{0, 1\}^d$，容量 $2^d$
   - 量化间距：$\Delta = 1$（最粗糙）
   - 对应LFQ（Lookup-Free Quantization）

2. **$L = 5$**（论文推荐）：
   - 编码空间：$\{0, 1, 2, 3, 4\}^d$，容量 $5^d$
   - 量化间距：$\Delta = 0.25$
   - 平衡了精度和计算效率

3. **$L = 8$**：
   - 编码空间：$\{0, 1, \ldots, 7\}^d$，容量 $8^d$
   - 量化间距：$\Delta = 0.143$
   - 更精细的量化

**定理16.2（L的选择准则）**：为了达到与VQ-VAE相当的编码容量 $K$，FSQ的参数应满足：

$$
L^d \approx K \Rightarrow L \approx K^{1/d}
$$

例如，若 $K = 8192, d = 8$，则：

$$
L = 8192^{1/8} = 2^{13/8} \approx 3.36 \approx 4
$$

### 17. FSQ的重参数化技巧详解

**定理17.1（STE的重参数化形式）**：FSQ可以写成重参数化形式：

$$
z_q = h(z) + \text{sg}[q(z) - h(z)]
$$

这等价于：

$$
z_q = \begin{cases}
q(z) & \text{前向传播} \\
h(z) & \text{反向传播（梯度计算）}
\end{cases}
$$

**实现细节**：在深度学习框架中：

```python
def fsq(z, L):
    h = (L - 1) * torch.sigmoid(z)
    q = torch.round(h)
    z_q = h + (q - h).detach()  # detach实现sg操作
    return z_q
```

**数学验证**：

前向传播：$z_q = h + (q - h) = q$ ✓

反向传播：
$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial z_q} \frac{\partial z_q}{\partial z} = \frac{\partial \mathcal{L}}{\partial z_q} \frac{\partial h}{\partial z}
$$

因为 $(q - h)$ 被detach，其梯度为0。✓

### 18. FSQ的多级量化扩展

**定义18.1（非均匀量化）**：允许不同维度使用不同的量化级别：

$$
\text{FSQ}(z) = [\text{FSQ}_{L_1}(z_1), \text{FSQ}_{L_2}(z_2), \ldots, \text{FSQ}_{L_d}(z_d)]^T
$$

其中 $L_i$ 可以不同。

**性质18.1（编码容量）**：总编码容量为：

$$
\text{Capacity} = \prod_{i=1}^d L_i = L_1 \times L_2 \times \cdots \times L_d
$$

**定理18.1（质数分解策略）**：给定目标容量 $K$，可以将其分解为质数幂的乘积：

$$
K = p_1^{a_1} \times p_2^{a_2} \times \cdots \times p_m^{a_m}
$$

选择 $d = a_1 + a_2 + \cdots + a_m$，令：
- $a_1$ 个维度使用 $L_i = p_1$
- $a_2$ 个维度使用 $L_i = p_2$
- 依此类推

例如，$K = 8192 = 2^{13}$，可以选择 $d = 13, L_i = 2$（全部二值化）。

### 19. FSQ的温度调节扩展

**定义19.1（温度参数）**：引入温度参数 $\tau$ 来控制量化的"硬度"：

$$
\text{FSQ}_\tau(t) = \text{Round}[(L-1)\sigma(t/\tau)]
$$

**性质19.1（温度的影响）**：

- $\tau \to 0$：量化变硬，sigmoid变陡峭，快速饱和到0或1
- $\tau \to \infty$：量化变软，sigmoid变平缓，接近线性
- $\tau = 1$：标准FSQ

**训练策略19.1（温度退火）**：训练过程中逐渐降低 $\tau$：

$$
\tau_t = \tau_0 \cdot \exp(-\lambda t)
$$

这类似于Gumbel-Softmax中的温度退火，可以改善训练稳定性。

### 20. FSQ的理论优势总结与证明

**定理20.1（FSQ的三大理论优势）**：

1. **参数效率**：FSQ不需要学习编码表，节省 $O(L^d \cdot d)$ 个参数
2. **计算效率**：FSQ的量化复杂度为 $O(d)$，而VQ为 $O(Kd)$
3. **训练稳定性**：FSQ避免了编码表坍缩，梯度有界

**综合证明**：

*参数效率*：已在定理5.1证明。

*计算效率*：已在定理14.1证明。

*训练稳定性*：
- 梯度有界（定理12.1）
- 无编码表坍缩（定理13.1）
- STE的无偏性（性质7.1）

这三个优势共同保证了FSQ在大规模应用中的可行性。$\square$

### 21. FSQ的局限性分析

**定理21.1（低维瓶颈）**：FSQ要求 $d = \log_L K$，当 $K$ 固定时，较大的 $L$ 导致较小的 $d$，这限制了连续表示的容量。

**例子**：若 $K = 8192, L = 8$，则 $d \approx 4.33$，实际取 $d = 4$ 或 $d = 5$。

如果VQ-VAE使用 $d_{VQ} = 256$，则FSQ的连续表示维度显著更低。

**影响**：当需要高维连续表示时（例如某些层级编码场景），FSQ可能不如VQ-VAE。

**定理21.2（表达能力权衡）**：FSQ将表达能力压缩到低维空间，依赖encoder和decoder的强大能力来补偿。

形式化地，FSQ的表示能力 $\mathcal{R}_{FSQ}$ 满足：

$$
\mathcal{R}_{FSQ} \leq \mathcal{R}_{encoder} \times \mathcal{R}_{FSQ,quant} \times \mathcal{R}_{decoder}
$$

其中 $\mathcal{R}_{FSQ,quant}$ 受限于低维 $d$，需要通过增强 $\mathcal{R}_{encoder}$ 和 $\mathcal{R}_{decoder}$ 来补偿。

### 22. FSQ在不同场景下的适用性分析

**场景1：大编码表（$K > 1000$）**

根据原论文实验，当 $K > 1000$ 时，FSQ性能优于或接近VQ-VAE。

**理论解释**：
- 大 $K$ 意味着 $d = \log_L K$ 不会太小（当 $L=5$ 时，$K=1024$ 对应 $d=4.4$）
- VQ-VAE在大 $K$ 时更容易出现编码表坍缩
- FSQ的计算优势更明显（$O(d)$ vs $O(Kd)$）

**场景2：小编码表（$K < 500$）**

当 $K$ 较小时，VQ-VAE可能更优。

**理论解释**：
- 小 $K$ 导致 $d$ 非常小，FSQ的表达受限
- VQ-VAE可以使用更高维的 $d_{VQ}$，保持表达能力
- 编码表坍缩问题相对较轻

**场景3：图像Tokenizer**

FSQ特别适合作为图像Tokenizer：
- 编码表通常很大（$K \geq 1024$）
- encoder和decoder可以设计得很强（ResNet、Transformer等）
- 训练效率和稳定性至关重要

### 23. FSQ与其他量化方法的联系

**定理23.1（FSQ与标量量化的关系）**：FSQ本质上是标量量化（Scalar Quantization）的向量化版本。

传统标量量化：

$$
Q(x) = \text{Round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \times (L-1)\right)
$$

FSQ：

$$
Q(x) = \text{Round}[(L-1)\sigma(x)]
$$

区别在于FSQ使用sigmoid自动处理范围归一化，而传统方法需要预先知道 $x_{min}, x_{max}$。

**定理23.2（FSQ与Gumbel-Softmax的对比）**：

Gumbel-Softmax用于分类分布的软化：

$$
\text{GumbelSoftmax}(z) = \frac{\exp((z_i + g_i)/\tau)}{\sum_j \exp((z_j + g_j)/\tau)}
$$

FSQ更简单直接，使用确定性的四舍五入而非概率采样。

**定理23.3（FSQ与Product Quantization的关系）**：

Product Quantization将向量分为 $m$ 个子向量，每个独立量化：

$$
z = [z^{(1)}, z^{(2)}, \ldots, z^{(m)}], \quad Q(z) = [Q_1(z^{(1)}), Q_2(z^{(2)}), \ldots, Q_m(z^{(m)})]
$$

FSQ可以看作特殊的Product Quantization，其中每个子向量是单个标量。

### 24. FSQ的数值稳定性分析

**定理24.1（数值溢出预防）**：sigmoid函数在极端输入时可能数值不稳定。

当 $z \to +\infty$ 时，$e^{-z} \to 0$，计算 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是安全的。

当 $z \to -\infty$ 时，$e^{-z} \to +\infty$，可能溢出。

**解决方案**：使用数值稳定的sigmoid实现：

$$
\sigma(z) = \begin{cases}
\frac{1}{1+e^{-z}} & \text{if } z \geq 0 \\
\frac{e^z}{1+e^z} & \text{if } z < 0
\end{cases}
$$

两种形式在数学上等价，但数值上更稳定。

### 25. FSQ的理论完备性总结

通过以上推导，我们完整地分析了FSQ的：

1. **数学定义**：基于四舍五入和sigmoid的确定性量化
2. **梯度机制**：Straight-Through Estimator及其合理性
3. **理论优势**：参数效率、计算效率、训练稳定性
4. **与VQ对比**：参数量、计算复杂度、编码表坍缩
5. **表达能力**：维度权衡与编码容量分析
6. **适用场景**：大编码表、图像Tokenizer等
7. **数值实现**：稳定性保证与实现技巧

FSQ的核心思想是用简单的数学运算（四舍五入+sigmoid）替代复杂的最近邻搜索，同时保持相当或更好的性能。这体现了"简单得令人尴尬"的设计哲学：有时最简单的方法反而最有效。

**最终定理**：在满足以下条件时，FSQ优于VQ-VAE：

1. 编码表足够大（$K > 1000$）
2. Encoder和Decoder足够强
3. 训练数据充足且多样化

这些条件在现代大规模视觉模型中通常都能满足，因此FSQ具有广泛的应用前景。

