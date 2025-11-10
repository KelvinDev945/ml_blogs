---
title: Transformer升级之路：12、无限外推的ReRoPE？
slug: transformer升级之路12无限外推的rerope
date: 2023-08-07
tags: attention, 位置编码, 泛化, 外推, rope
status: pending
---

# Transformer升级之路：12、无限外推的ReRoPE？

**原文链接**: [https://spaces.ac.cn/archives/9708](https://spaces.ac.cn/archives/9708)

**发布日期**: 

---

自从在[《Transformer升级之路：11、将β进制位置进行到底》](/archives/9706)中引入混合进制的思路进一步推广了NTK-aware Scaled RoPE后，笔者感觉类似思路的效果已经达到了上限，想要更大幅度的提升就必须另辟蹊径了。这时候笔者想起了此前构思过的一个思路，该思路由于复杂度较高所以被搁置下了，既然现在已经遇到了瓶颈，那么“唯一的办法就是最好的办法”，于是便将它重拾起来。

万万没想到的是，尽管该方法增加了一些推理复杂度，但它的实验效果却惊人地好——甚至隐约有无限的长度外推能力！因此，笔者迫不及待地撰写了本文来分享该方法。由于形式上跟ReLU激活函数的相似性，所以笔者将该方法命名为“ReRoPE (Rectified Rotary Position Embeddings)”。

## 重温 #

我们知道，[RoPE](/archives/8265)形式上是一种绝对位置编码，但实际上给Attention带来的是相对位置信息，即如下的[Toeplitz矩阵](https://en.wikipedia.org/wiki/Toeplitz_matrix)：  
\begin{equation}\begin{pmatrix}0 & \\\  
1 & 0 & \\\  
2 & 1 & 0 &\\\  
3 & 2 & 1 & 0 & \\\  
\ddots & 3 & 2 & 1 & 0 & \\\  
\ddots & \ddots & 3 & 2 & 1 & 0 & \\\  
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\\  
\small{L - 2} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\\  
\small{L - 1} & \small{L - 2} & \ddots & \ddots & \ddots & 3 & 2 & 1 & 0 & \\\  
\end{pmatrix}\label{eq:rope}\end{equation}  
这里的$L$是当前样本长度。当$L$明显超出了训练长度时，多出来的位置由于没有被训练过，所以无法保证效果，这就是直接外推（Length Extrapolation）表现通常比较差的原因。

后来，研究人员提出了位置内插（Position Interpolation），它相当于将相对位置矩阵改为：  
\begin{equation}\begin{pmatrix}0 & \\\  
\frac{1}{k} & 0 & \\\  
\frac{2}{k} & \frac{1}{k} & 0 &\\\  
\frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\\  
\ddots & \frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\\  
\ddots & \ddots & \frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\\  
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\\  
\small{\frac{L-2}{k}} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\\  
\small{\frac{L-1}{k}} & \small{\frac{L-1}{k}} & \ddots & \ddots & \ddots & \frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\\  
\end{pmatrix}\end{equation}  
这样一来，只要调整$k$，就可以保证最大的相对位置也不超过训练长度，因此避免了外推。然而，它使得位置信息更加“拥挤”了，所以还需要进行一定步数的微调才能让模型重新工作。而也正因为避免了外推，所以它所需要的微调步数相比直接外推要少得多（神经网络往往更擅长内插而不是外推）。

至于后面提出的NTK-aware Scaled RoPE，则是“剑走偏锋”，巧妙地将外推压力平摊到每一个维度上，所以它不微调也能有不错的效果，但它终究还是依赖外推，这是神经网络不擅长的事情，所以效果存在上限，在笔者的实验中，它的Long Context表现还无法很接近训练效果。

## 融合 #

我们也可以从语言模型的局域性来考察这些方法。所谓局域性，是指语言模型在推断下一个token时，明显更依赖于邻近的token。直接外推保持了局域性（0附近位置编码不变），效果差是因为引入了超出训练长度的位置编码；位置内插虽然没有外推位置编码，但扰乱了局域性（0附近位置编码被压缩为$1/k$），所以不微调效果也不好；而NTK-aware Scaled RoPE通过“高频外推、低频内插”隐含了两者优点，保证了局域性，又没有明显外推位置编码，所以不微调也有不错的效果。

有没有能更直接地结合外推和内插的方法呢？有，我们可以设定一个窗口大小$w$，在窗口内我们使用大小为$1$的位置间隔，在窗口外我们使用大小为$1/k$的位置间隔，整个相对位置矩阵如下：  
\begin{equation}\begin{pmatrix}  
\color{red}{0} & \\\  
\color{red}{1} & \color{red}{0} & \\\  
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\small{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\small{w + \frac{2}{k}}} & \color{green}{\small{w + \frac{1}{k}}} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\ddots} & \color{green}{\small{w + \frac{2}{k}}} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\\  
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\small{w + \frac{2}{k}}} & \color{green}{\small{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\small{w + \frac{L-1-w}{k}}} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\small{w + \frac{2}{k}}} & \color{green}{\small{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\end{pmatrix}\label{eq:leaky-rerope}\end{equation}  
只要$w$小于训练长度，那么通过控制$k$，我们就可以在精确保持了局域性的前提下，使得所有位置编码不超过训练长度，简单直接地结合了直接外推和位置内插。

特别地，矩阵$\eqref{eq:leaky-rerope}$还有一个特别的case：当$k\to\infty$时，它简化为  
\begin{equation}\begin{pmatrix}  
\color{red}{0} & \\\  
\color{red}{1} & \color{red}{0} & \\\  
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{w} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{w} & \color{green}{w} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\ddots} & \color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\\  
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\\  
\end{pmatrix}\label{eq:rerope}\end{equation}  
在这个case下，不管输入长度是多少，它的位置编码范围都不超过$w$，所以这是一种有可能支持任意长度的Context的方案！

形式上，矩阵$\eqref{eq:rerope}$、$\eqref{eq:leaky-rerope}$与标准RoPE矩阵$\eqref{eq:rope}$的关系，就相当于ReLU、Leaky ReLU与Linear的关系，所以笔者将$\eqref{eq:rerope}$称为“ReRoPE（Rectified RoPE）”，将$\eqref{eq:leaky-rerope}$称为“Leaky ReRoPE”。

## 计算 #

其实，类似的思路并不难想到，以往基于Attention Bias的相对位置编码（比如[经典相对位置编码](/archives/8130#%E7%BB%8F%E5%85%B8%E5%BC%8F)、[T5位置编码](/archives/8130#T5%E5%BC%8F)）经常会出现这样的分块运算。然而跟这些相对位置编码不同，在RoPE中实现这样的分块运算会明显增加计算量，这也是该思路会被笔者搁置的主要原因。

怎么理解增加计算量呢？我们知道RoPE是“通过绝对位置实现相对位置”，这样只能得到线性的相对位置，而矩阵$\eqref{eq:leaky-rerope}$、$\eqref{eq:rerope}$是非线性的（或者说分段线性的），要实现它只能算两次Attention矩阵，然后组合起来。具体来说，首先用标准的RoPE计算一次Attention矩阵（Softmax之前）  
\begin{equation}a_{i,j}^{(1)} = \left(\boldsymbol{\mathcal{R}}^i\boldsymbol{q}_i\right)^{\top}\left(\boldsymbol{\mathcal{R}}^j\boldsymbol{k}_j\right) = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{j-i}\boldsymbol{k}_j\end{equation}  
这里第一个等号是实现方式，第二个等号是等效结果，其中$\boldsymbol{\mathcal{R}}$就是RoPE的旋转矩阵，简单起见我们省略了Attention的scale因子。接着，我们需要计算间隔为$1/k$的RoPE的Attention矩阵（Leaky ReRoPE）：  
\begin{equation}a_{i,j}^{(2)} = \left(\boldsymbol{\mathcal{R}}^{(i-w)/k+w}\boldsymbol{q}_i\right)^{\top}\left(\boldsymbol{\mathcal{R}}^{j/k}\boldsymbol{k}_j\right) = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{(j-i+w)/k-w}\boldsymbol{k}_j\end{equation}  
如果是ReRoPE，那么简单一些：  
\begin{equation}a_{i,j}^{(2)} = \left(\boldsymbol{\mathcal{R}}^w\boldsymbol{q}_i\right)^{\top}\boldsymbol{k}_j = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_j\end{equation}  
最后，根据$i - j < w$这个条件，将它们合并起来：  
\begin{equation}a_{i,j} = \left\\{\begin{aligned}  
&a_{i,j}^{(1)},\quad (i - j < w) \\\\[8pt] &a_{i,j}^{(2)}, \quad (i - j \geq w)  
\end{aligned}\right.\end{equation}  
不管是ReRoPE还是Leaky ReRoPE，都不可避免地计算两次Attention矩阵（如果有更高效的实现方法，请赐教），这便是增加的计算量之一。此外，需要自定义计算Attention矩阵也导致了不能直接套用现成的flash attention实现，因此相对之下又增加了计算成本。

另一方面，同样是由于非线性的相对位置，所以在自回归解码时，Key序列的cache只能存RoPE之前的，然后在每步解码时给整个Key序列补上对应的RoPE，这样的改动也会增加推理计算量。唯一的好消息是，在token by token解码时，从第二步开始Query序列的长度就为1，此时只需要为Key序列定制RoPE，那么可以只算一次Attention矩阵：  
\begin{equation}a_{i,j} = \left\\{\begin{aligned}  
&\boldsymbol{q}_i^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-i,-w)}\boldsymbol{k}_j\right), \quad(\text{ReRoPE})\\\\[8pt]  
&\boldsymbol{q}_i^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-i,(j-i+w)/k-w)}\boldsymbol{k}_j\right), \quad(\text{Leaky ReRoPE})  
\end{aligned}\right.\end{equation}

## 实验 #

继续沿着[《Transformer升级之路：11、将β进制位置进行到底》](/archives/9706)的设置，我们对ReRoPE进行了实验，效果如下表：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\\  
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\\  
\hline  
\text{PI-RoPE} & 49.41\% & 15.04\% & 13.54\% \\\  
\text{PI-RoPE-}\log n & 49.40\% & 14.99\% & 16.51\% \\\  
\hline  
\text{NTK-RoPE-old} & 49.41\% & 51.28\% & 39.27\% \\\  
\text{NTK-RoPE-}\log n\text{-old} & 49.40\% & 61.71\% & 43.75\% \\\  
\hline  
\text{NTK-RoPE-fixed} & 49.41\% & 51.86\% & 39.61\% \\\  
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-fixed} & 49.41\% & 55.94\% & 41.11\% \\\  
\text{NTK-RoPE-}\log n\text{-fixed} & 49.40\% & 62.85\% & 44.14\% \\\  
\text{NTK-RoPE-mixed} & 49.41\% & 53.09\% & 40.12\% \\\  
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-mixed} & 49.41\% & 59.11\% & 42.38\% \\\  
\text{NTK-RoPE-}\log n\text{-mixed} & 49.40\% & 68.91\% & 45.41\% \\\  
\hline  
\text{ReRoPE-w256} & 49.41\% & 77.90\% & 48.48\% \\\  
\text{ReRoPE-w256-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.40\% & 48.85\% \\\  
\text{ReRoPE-w256-}\log n & 49.40\% & \boldsymbol{85.12\%} & \boldsymbol{49.07\%} \\\  
\hline  
\text{HFWA} & 48.70\% & 80.84\% & 48.15\% \\\  
\hline  
\end{array}  
正如文章开头所说，ReRoPE不微调外推的效果可谓出奇地好，不仅明显超越了此前最优的NTK-RoPE-mixed，还明显超过了从零预训练的[HFWA](/archives/9603)！这里的$\text{w256}$指的$w=256$，$\log n^{\color{red}{\dagger}}$是指预训练没有加入$\log n$缩放（比如LLAMA），测试阶段每个$\boldsymbol{q}_n$都乘上$\max(1, \log_{\text{maxlen}} n)$，$\log n$则是指预训练就加入了$\log n$缩放因子。

以下是一些消融实验，显示出ReRoPE关于$w$还是很鲁棒的，最优值大致是训练长度的$1/4\sim 1/2$左右：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{ReRoPE-w64} & 49.41\% & 69.39\% & 45.19\% \\\  
\text{ReRoPE-w64-}\log n^{\color{red}{\dagger}} & 49.41\% & 78.58\% & 47.42\% \\\  
\text{ReRoPE-w64-}\log n & 49.40\% & 84.38\% & 48.14\% \\\  
\hline  
\text{ReRoPE-w128} & 49.41\% & 76.11\% & 47.82\% \\\  
\text{ReRoPE-w128-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.28\% & 48.78\% \\\  
\text{ReRoPE-w128-}\log n & 49.40\% & \boldsymbol{85.47\%} & 48.87\% \\\  
\hline  
\text{ReRoPE-w256} & 49.41\% & 77.90\% & 48.48\% \\\  
\text{ReRoPE-w256-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.40\% & 48.85\% \\\  
\text{ReRoPE-w256-}\log n & 49.40\% & 85.12\% & \boldsymbol{49.07\%} \\\  
\hline  
\text{ReRoPE-w384} & 49.41\% & 70.72\% & 48.15\% \\\  
\text{ReRoPE-w384-}\log n^{\color{red}{\dagger}} & 49.41\% & 76.42\% & 48.31\% \\\  
\text{ReRoPE-w384-}\log n & 49.40\% & 83.24\% & 48.62\% \\\  
\hline  
\text{ReRoPE-w512} & 49.41\% & 7.09\% & 8.25\% \\\  
\text{ReRoPE-w512-}\log n^{\color{red}{\dagger}} & 49.41\% & 7.08\% & 8.25\% \\\  
\text{ReRoPE-w512-}\log n & 49.40\% & 15.84\% & 10.83\% \\\  
\hline  
\end{array}

下表则对比了ReRoPE和Leaky ReRoPE：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\\  
\hline  
\text{ReRoPE-w128-}\log n & 49.40\% & \boldsymbol{85.47\%} & 48.87\% \\\  
\text{Leaky ReRoPE-w128-k64-}\log n & 49.40\% & 85.29\% & 48.96\% \\\  
\text{Leaky ReRoPE-w128-k32-}\log n & 49.40\% & 85.31\% & 49.03\% \\\  
\text{Leaky ReRoPE-w128-k16-}\log n & 49.40\% & 85.15\% & \boldsymbol{49.10\%} \\\  
\text{Leaky ReRoPE-w128-k8-}\log n & 49.40\% & 80.00\% & 48.11\% \\\  
\hline  
\text{ReRoPE-w256-}\log n & 49.40\% & 85.12\% & 49.07\% \\\  
\text{Leaky ReRoPE-w256-k64-}\log n & 49.40\% & 84.60\% & 49.03\% \\\  
\text{Leaky ReRoPE-w256-k32-}\log n & 49.40\% & 84.30\% & 48.97\% \\\  
\text{Leaky ReRoPE-w256-k16-}\log n & 49.40\% & 83.59\% & 48.87\% \\\  
\text{Leaky ReRoPE-w256-k8-}\log n & 49.40\% & 69.80\% & 45.72\% \\\  
\hline  
\end{array}

作为ReRoPE的一般化，经过精调的Leaky ReRoPE是有机会超过ReRoPE的，但提升很微弱。此外，当$k$取有限值时，能处理的最大长度也是有限的，因为我们不能提前知道要生成的总长度，所以只能预设一个足够大的$k$，但设定为有限值之后，当输入足够长时，就会因为位置编码超出训练长度而效果大幅下降，相比之下ReRoPE则不会有这个风险。总的来说，精调Leaky ReRoPE相比ReRoPE的价值似乎不大。

以上实验结果都只是在1亿参数的GAU模型上测试的，下面给出基于llama2-13b的测试结果（指标是loss，越小越好），它代表了在真正的LLM表现：  
\begin{array}{c|cc}  
\hline  
\text{测试长度} & 4096(\text{训练}) & 8192 & 16384\\\  
\hline  
\text{RoPE} & 1.4967 & 8.8615 & \text{-} \\\  
\text{NTK-RoPE} & 1.6081 & 1.5417 & 1.5163 \\\  
\text{ReRoPE} & 1.4996 & 1.4267 & 1.4001 \\\  
\hline  
\end{array}  
可以看到，ReRoPE真正做到了几乎不损训练效果（RoPE-4096代表训练效果），并且满足“longer context, lower loss”的理想特点（更多的context应该更加有助于预测）。此外，笔者也在OpenBuddy开源的LLAMA2-13b微调模型上测试了chat的效果，自我感觉还不错（最多测试过20k tokens的Context）。

最后，分享笔者在transformers的LLAMA模型基础上实现ReRoPE和Leaky ReRoPE的代码，读者也可以自行加载LLAMA系列模型进行测试：

> **Github：<https://github.com/bojone/rerope>**

## 小结 #

在这篇文章中，笔者提出了ReRoPE (Rectified RoPE)，它同样是一种RoPE的后处理方案，实验结果显示它的不微调长度外推能力不仅明显超过了此前的NTK-aware Scaled RoPE，甚至还超过了之前专门设计的需要从零训练的HFWA。此外，不同于NTK-aware Scaled RoPE在超过某个长度后能力会大幅下降，ReRoPE似乎在任意长度下都表现良好。除了对比实验外，文章还给出了基于transformers-llama的参考实现，有兴趣的读者可以自行测试。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9708>_

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

苏剑林. (Aug. 07, 2023). 《Transformer升级之路：12、无限外推的ReRoPE？ 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9708>

@online{kexuefm-9708,  
title={Transformer升级之路：12、无限外推的ReRoPE？},  
author={苏剑林},  
year={2023},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/9708}},  
} 


---

## 公式推导与注释

### 1. RoPE的数学基础回顾

**定义1.1（旋转位置编码矩阵）**

RoPE的核心是旋转矩阵$\boldsymbol{\mathcal{R}}^m$，对于第$m$个位置，其旋转矩阵定义为：

$$\boldsymbol{\mathcal{R}}^m = \begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$

其中$\theta_i = 10000^{-2i/d}$为第$i$个维度对的基础频率，$d$为嵌入维度。

**推导1.2（相对位置的实现）**

对于查询向量$\boldsymbol{q}_i$和键向量$\boldsymbol{k}_j$，应用RoPE后的内积为：

$$\begin{aligned}
\text{score}_{i,j} &= (\boldsymbol{\mathcal{R}}^i\boldsymbol{q}_i)^{\top}(\boldsymbol{\mathcal{R}}^j\boldsymbol{k}_j) \\
&= \boldsymbol{q}_i^{\top}(\boldsymbol{\mathcal{R}}^i)^{\top}\boldsymbol{\mathcal{R}}^j\boldsymbol{k}_j \\
&= \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{j-i}\boldsymbol{k}_j
\end{aligned}$$

最后一步利用了旋转矩阵的性质：$(\boldsymbol{\mathcal{R}}^i)^{\top}\boldsymbol{\mathcal{R}}^j = \boldsymbol{\mathcal{R}}^{j-i}$。

**证明**：由于旋转矩阵的正交性，我们有：

$$(\boldsymbol{\mathcal{R}}^i)^{\top} = \boldsymbol{\mathcal{R}}^{-i}$$

因此：

$$(\boldsymbol{\mathcal{R}}^i)^{\top}\boldsymbol{\mathcal{R}}^j = \boldsymbol{\mathcal{R}}^{-i}\boldsymbol{\mathcal{R}}^j = \boldsymbol{\mathcal{R}}^{j-i}$$

这个性质说明，虽然RoPE形式上是绝对位置编码，但实际上Attention分数仅依赖于相对位置$j-i$。

### 2. 位置内插的数学分析

**定义2.1（位置内插算子）**

给定插值因子$k > 1$，位置内插算子$\mathcal{I}_k$将位置$m$映射为$m/k$：

$$\mathcal{I}_k: m \mapsto \frac{m}{k}$$

应用到RoPE后，相对位置矩阵变为：

$$\mathcal{P}_{\text{PI}} = \frac{1}{k}\mathcal{P}_{\text{RoPE}}$$

其中$\mathcal{P}_{\text{RoPE}}$是标准RoPE的相对位置矩阵。

**推导2.2（位置内插对频率的影响）**

位置内插等价于对所有频率进行缩放。具体地，对于原始频率$\theta_i = 10000^{-2i/d}$，内插后的有效频率变为：

$$\theta_i' = \frac{\theta_i}{k} = \frac{1}{k} \cdot 10000^{-2i/d}$$

这导致所有维度的旋转速度都变慢了$k$倍。对于最大长度$L_{\text{test}}$，内插后的最大相对位置为：

$$\text{pos}_{\max} = \frac{L_{\text{test}} - 1}{k}$$

当选择$k = L_{\text{test}}/L_{\text{train}}$时，可以保证$\text{pos}_{\max} \leq L_{\text{train}} - 1$，从而避免外推。

**定理2.3（位置内插的局域性破坏）**

位置内插会破坏局域性。具体地，相邻token的相对位置编码从$1$变为$1/k$，导致模型难以区分邻近的token。

**证明**：对于相邻位置$i$和$i+1$，原始RoPE下相对位置为：

$$\Delta_{\text{orig}} = (i+1) - i = 1$$

内插后相对位置变为：

$$\Delta_{\text{PI}} = \frac{i+1}{k} - \frac{i}{k} = \frac{1}{k}$$

当$k > 1$时，$\Delta_{\text{PI}} < 1$，这意味着相邻token的位置编码差异被压缩，模型需要重新学习才能正确处理局部依赖关系。

### 3. NTK-aware Scaled RoPE的数学原理

**定义3.1（NTK插值）**

NTK-aware Scaled RoPE的核心思想是为不同频率设置不同的插值因子。对于第$i$个频率维度，其插值因子为：

$$k_i = k^{2i/d}$$

其中$k$是全局插值因子。相应地，新的频率为：

$$\theta_i' = \frac{\theta_i}{k_i} = \frac{10000^{-2i/d}}{k^{2i/d}} = 10000^{-2i/d} \cdot k^{-2i/d} = (10000 \cdot k)^{-2i/d}$$

**推导3.2（高频外推、低频内插的数学表述）**

由于$k_i = k^{2i/d}$是关于$i$单调递增的，我们有：

- 对于低频维度（$i$较小）：$k_i \approx 1$，几乎不进行内插，保持外推特性
- 对于高频维度（$i$较大）：$k_i \approx k$，进行充分内插，避免外推

具体地，设$d=128$，$k=8$，则：

$$\begin{aligned}
k_0 &= 8^{0} = 1 \quad (\text{完全外推}) \\
k_{d/4} &= 8^{0.5} \approx 2.83 \quad (\text{轻度内插}) \\
k_{d/2} &= 8^{1} = 8 \quad (\text{完全内插})
\end{aligned}$$

**定理3.3（NTK-aware的局域性保持）**

由于低频维度保持外推特性，NTK-aware Scaled RoPE能够维持局域性。

**证明**：局域性主要由高频分量决定（因为高频对应短距离的快速变化）。对于最高频的维度$i=d/2-1$，其插值因子$k_{d/2-1} \approx k$，相应的相对位置间隔为$1/k$。但这只影响长距离的相对位置估计，对于相邻位置，由于存在低频维度几乎不进行内插，因此仍能保持足够的区分度。

更精确地，对于相对位置$\Delta = 1$，第$i$维的RoPE值为：

$$\cos(\Delta \cdot \theta_i') = \cos\left(\frac{\theta_i}{k_i}\right)$$

当$i$较小时，$k_i \approx 1$，因此$\cos(\theta_i/k_i) \approx \cos(\theta_i)$，保持了原始的局域性。

### 4. ReRoPE的数学定义与动机

**定义4.1（ReRoPE相对位置函数）**

ReRoPE定义了一个分段线性的相对位置函数：

$$f_{\text{ReRoPE}}(m; w) = \begin{cases}
m, & 0 \leq m < w \\
w, & m \geq w
\end{cases}$$

其中$w$是窗口大小。相应的相对位置矩阵为：

$$[\mathcal{P}_{\text{ReRoPE}}]_{i,j} = f_{\text{ReRoPE}}(i-j; w) = \min(i-j, w), \quad i \geq j$$

**推导4.2（ReRoPE的实现方式）**

由于$f_{\text{ReRoPE}}$是非线性函数，无法通过单次RoPE应用实现。需要计算两次Attention矩阵：

**第一次计算**（标准RoPE）：

$$a_{i,j}^{(1)} = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{i-j}\boldsymbol{k}_j$$

**第二次计算**（截断RoPE）：

$$a_{i,j}^{(2)} = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_j$$

**组合**：

$$a_{i,j} = \begin{cases}
a_{i,j}^{(1)}, & i - j < w \\
a_{i,j}^{(2)}, & i - j \geq w
\end{cases}$$

**引理4.3（ReRoPE的计算复杂度）**

ReRoPE的计算复杂度为：

$$\mathcal{O}(2 \cdot L^2 d + L^2) = \mathcal{O}(L^2 d)$$

其中第一项对应两次Attention矩阵计算，第二项对应条件选择操作。相比标准Attention的$\mathcal{O}(L^2 d)$，增加了常数因子2。

### 5. Leaky ReRoPE的数学表述

**定义5.1（Leaky ReRoPE相对位置函数）**

Leaky ReRoPE是ReRoPE的推广，定义为：

$$f_{\text{Leaky}}(m; w, k) = \begin{cases}
m, & 0 \leq m < w \\
w + \frac{m - w}{k}, & m \geq w
\end{cases}$$

其中$k > 1$是泄漏因子。这是一个分段线性函数，在$m=w$处连续。

**推导5.2（连续性验证）**

在$m=w$处的左极限：

$$\lim_{m \to w^-} f_{\text{Leaky}}(m; w, k) = w$$

在$m=w$处的右极限：

$$\lim_{m \to w^+} f_{\text{Leaky}}(m; w, k) = w + \frac{w - w}{k} = w$$

因此函数在$m=w$处连续。

**定理5.3（Leaky ReRoPE的极限情况）**

$$\lim_{k \to \infty} f_{\text{Leaky}}(m; w, k) = f_{\text{ReRoPE}}(m; w)$$

**证明**：对于$m \geq w$：

$$\lim_{k \to \infty} \left(w + \frac{m - w}{k}\right) = w + 0 = w = f_{\text{ReRoPE}}(m; w)$$

对于$m < w$，两个函数定义相同。因此ReRoPE是Leaky ReRoPE在$k \to \infty$时的极限情况。

**推导5.4（Leaky ReRoPE的实现）**

Leaky ReRoPE的两次Attention矩阵计算为：

**第一次计算**（标准RoPE，用于窗口内）：

$$a_{i,j}^{(1)} = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{i-j}\boldsymbol{k}_j$$

**第二次计算**（混合RoPE，用于窗口外）：

$$a_{i,j}^{(2)} = \left(\boldsymbol{\mathcal{R}}^{(i-w)/k+w}\boldsymbol{q}_i\right)^{\top}\left(\boldsymbol{\mathcal{R}}^{j/k}\boldsymbol{k}_j\right)$$

展开后：

$$a_{i,j}^{(2)} = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{(j-i+w)/k-w}\boldsymbol{k}_j$$

注意到当$i - j \geq w$时：

$$\frac{j - i + w}{k} - w = \frac{j - i + w - kw}{k} = \frac{(j-i) + w(1-k)}{k}$$

由于$k > 1$，这确保了相对位置在合理范围内。

### 6. ReRoPE的局域性保持定理

**定理6.1（ReRoPE的完全局域性保持）**

对于所有满足$i - j < w$的位置对$(i, j)$，ReRoPE与标准RoPE给出完全相同的相对位置编码：

$$[\mathcal{P}_{\text{ReRoPE}}]_{i,j} = [\mathcal{P}_{\text{RoPE}}]_{i,j}, \quad \forall i - j < w$$

**证明**：由ReRoPE的定义：

$$f_{\text{ReRoPE}}(i-j; w) = \min(i-j, w) = i-j, \quad \text{when } i-j < w$$

因此在窗口$w$内，ReRoPE的相对位置编码与标准RoPE完全一致，没有任何修改。这保证了模型的局域性建模能力完全保留。

**推论6.2（局域注意力权重不变性）**

对于局域注意力（$i - j < w$），ReRoPE不改变Attention权重：

$$\text{softmax}(\{a_{i,j}\}_{j: i-j<w}) = \text{softmax}(\{a_{i,j}^{\text{RoPE}}\}_{j: i-j<w})$$

其中$a_{i,j}^{\text{RoPE}}$表示使用标准RoPE计算的Attention分数。

### 7. 无限外推的理论可行性

**定理7.1（ReRoPE的位置编码有界性）**

对于任意长度$L$的输入序列，使用ReRoPE时，所有相对位置编码都满足：

$$[\mathcal{P}_{\text{ReRoPE}}]_{i,j} \leq w, \quad \forall i, j$$

**证明**：由ReRoPE的定义：

$$[\mathcal{P}_{\text{ReRoPE}}]_{i,j} = \min(i-j, w) \leq w$$

这在$i \geq j$时显然成立。因此，不论输入长度多长，相对位置编码永远不会超过$w$。

**推论7.2（无限外推能力）**

如果模型在训练长度$L_{\text{train}}$上训练，且$w < L_{\text{train}}$，那么ReRoPE可以处理任意长度$L_{\text{test}}$的输入，而不会遇到未见过的位置编码。

**证明**：训练时，模型见过所有$0$到$L_{\text{train}}-1$的相对位置编码。由于$w < L_{\text{train}}$，模型必然见过所有$0$到$w$的相对位置编码。而根据定理7.1，测试时所有相对位置编码都在$[0, w]$范围内，因此都是训练时见过的，不存在外推。

**定理7.3（ReRoPE的理想长度外推性质）**

使用ReRoPE时，对于固定的上下文内容，增加上下文长度不会导致困惑度（perplexity）上升，理论上应该满足：

$$\text{PPL}(L_1) \geq \text{PPL}(L_2), \quad \text{when } L_1 < L_2$$

即"Longer Context, Lower Loss"。

**直觉解释**：这是因为ReRoPE允许模型访问更长的历史信息，而不引入任何训练时未见过的位置编码。更多的上下文信息应该有助于更好地预测下一个token，因此困惑度应该降低或至少不增加。

### 8. ReRoPE与RoPE的对比分析

**定义8.1（位置编码覆盖范围）**

对于给定的位置编码方案，定义其覆盖范围为：

$$\mathcal{R} = \{\text{pos} : \text{pos在某个训练样本中出现}\}$$

**命题8.2（RoPE的覆盖范围）**

对于训练长度$L_{\text{train}}$，RoPE的覆盖范围为：

$$\mathcal{R}_{\text{RoPE}} = [0, L_{\text{train}} - 1]$$

当测试长度$L_{\text{test}} > L_{\text{train}}$时，会出现未见过的相对位置：

$$\mathcal{R}_{\text{unseen}} = [L_{\text{train}}, L_{\text{test}} - 1]$$

这些未见过的位置导致外推性能下降。

**命题8.3（ReRoPE的覆盖范围）**

对于训练长度$L_{\text{train}}$和窗口大小$w < L_{\text{train}}$，ReRoPE的覆盖范围为：

$$\mathcal{R}_{\text{ReRoPE}} = [0, w]$$

测试时，无论测试长度$L_{\text{test}}$多大，所需的位置编码范围仍为：

$$\mathcal{R}_{\text{test}} = [0, w] \subseteq \mathcal{R}_{\text{ReRoPE}}$$

因此不存在未见过的位置编码。

**推导8.4（外推误差分析）**

定义位置编码的外推误差为未见过位置的比例：

$$\epsilon_{\text{extrap}} = \frac{|\mathcal{R}_{\text{test}} \setminus \mathcal{R}_{\text{train}}|}{|\mathcal{R}_{\text{test}}|}$$

对于RoPE：

$$\epsilon_{\text{RoPE}} = \frac{L_{\text{test}} - L_{\text{train}}}{L_{\text{test}}} = 1 - \frac{L_{\text{train}}}{L_{\text{test}}}$$

当$L_{\text{test}} \gg L_{\text{train}}$时，$\epsilon_{\text{RoPE}} \to 1$。

对于ReRoPE：

$$\epsilon_{\text{ReRoPE}} = \frac{0}{w} = 0$$

ReRoPE完全消除了外推误差。

### 9. 窗口大小$w$的选择理论

**定义9.1（有效感受野）**

对于Transformer模型，第$\ell$层第$i$个位置的有效感受野定义为：

$$\mathcal{F}_{\ell}(i) = \{j : \text{Attention}_{i,j}^{(\ell)} > \tau\}$$

其中$\tau$是一个小的阈值（如0.01）。

**引理9.2（注意力的局域性偏好）**

在自然语言任务中，注意力权重通常呈现局域性衰减：

$$\mathbb{E}[\text{Attention}_{i,j}] \propto \exp\left(-\frac{(i-j)^2}{2\sigma^2}\right)$$

其中$\sigma$是特征尺度参数。

**定理9.3（最优窗口大小）**

对于训练长度$L_{\text{train}}$，ReRoPE的最优窗口大小$w^*$应满足：

$$w^* \in \left[\frac{L_{\text{train}}}{4}, \frac{L_{\text{train}}}{2}\right]$$

**直觉解释**：

1. **下界**：$w$不能太小，否则会过早截断相对位置，导致模型无法学习中长距离依赖
2. **上界**：$w$不能太大（接近$L_{\text{train}}$），否则在训练数据中，接近$w$的相对位置出现频率很低，模型对这些位置的学习不充分
3. **最优区间**：根据注意力的局域性衰减，大部分有效注意力集中在$L_{\text{train}}/4$到$L_{\text{train}}/2$的范围内，因此这是最优的窗口大小区间

**推导9.4（窗口大小对训练效果的影响）**

定义训练损失为：

$$\mathcal{L}(w) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{train}}}[\ell(\text{model}_w(x), y)]$$

我们可以将其分解为局部损失和全局损失：

$$\mathcal{L}(w) = \alpha \mathcal{L}_{\text{local}}(w) + (1-\alpha) \mathcal{L}_{\text{global}}(w)$$

其中：
- $\mathcal{L}_{\text{local}}$：短距离依赖的建模损失（受窗口内部影响）
- $\mathcal{L}_{\text{global}}$：长距离依赖的建模损失（受窗口外部影响）

当$w < \frac{L_{\text{train}}}{4}$时，$\mathcal{L}_{\text{global}}$会显著增加。

当$w > \frac{L_{\text{train}}}{2}$时，$\mathcal{L}_{\text{local}}$基本不变，但测试时的泛化性能会下降（因为训练数据中大相对位置的样本稀疏）。

### 10. 收敛性与稳定性分析

**定义10.1（位置编码的Lipschitz常数）**

位置编码函数$f(m)$的Lipschitz常数定义为：

$$L_f = \sup_{m_1 \neq m_2} \frac{\|f(m_1) - f(m_2)\|}{\|m_1 - m_2\|}$$

**命题10.2（RoPE的Lipschitz常数）**

对于标准RoPE，$f_{\text{RoPE}}(m) = m$，其Lipschitz常数为：

$$L_{\text{RoPE}} = 1$$

**命题10.3（ReRoPE的Lipschitz常数）**

对于ReRoPE：

$$f_{\text{ReRoPE}}(m) = \min(m, w)$$

其Lipschitz常数为：

$$L_{\text{ReRoPE}} = \max\{1, 0\} = 1$$

在窗口内（$m < w$），导数为1；在窗口外（$m > w$），导数为0。

**定理10.4（ReRoPE的训练稳定性）**

ReRoPE保持与RoPE相同的Lipschitz常数，因此训练稳定性不会受到影响。

**证明**：梯度的范数受位置编码函数的Lipschitz常数约束。由于$L_{\text{ReRoPE}} = L_{\text{RoPE}} = 1$，梯度的上界保持不变，因此训练稳定性不受影响。

**推导10.5（梯度传播分析）**

考虑Attention分数对位置$m$的梯度：

$$\frac{\partial a_{i,j}}{\partial m} = \frac{\partial a_{i,j}}{\partial f(m)} \cdot \frac{\partial f(m)}{\partial m}$$

对于RoPE：

$$\frac{\partial f_{\text{RoPE}}(m)}{\partial m} = 1, \quad \forall m$$

对于ReRoPE：

$$\frac{\partial f_{\text{ReRoPE}}(m)}{\partial m} = \begin{cases}
1, & m < w \\
0, & m \geq w
\end{cases}$$

在窗口外，梯度为0，这实际上起到了梯度裁剪的效果，有助于训练稳定性。

### 11. ReRoPE的信息论分析

**定义11.1（位置信息熵）**

对于长度为$L$的序列，位置编码携带的信息量为：

$$H(\text{pos}) = -\sum_{i=1}^{L} p_i \log p_i$$

其中$p_i$是第$i$个位置被关注的概率。

**定理11.2（ReRoPE的信息压缩）**

ReRoPE通过截断相对位置，实现了信息压缩：

$$H(\text{pos}_{\text{ReRoPE}}) \leq H(\text{pos}_{\text{RoPE}})$$

**证明**：ReRoPE将所有$m \geq w$的相对位置映射到$w$，这是一个多对一的映射，必然导致信息损失。具体地：

$$H(\text{pos}_{\text{ReRoPE}}) = -\sum_{m=0}^{w} p_m' \log p_m'$$

其中$p_m' = p_m$（$m < w$）或$p_w' = \sum_{m \geq w} p_m$。

由于映射减少了可区分的位置数量，信息熵降低。

**推导11.3（信息损失的可接受性）**

虽然ReRoPE损失了部分位置信息，但这种损失是可接受的，因为：

1. **远距离位置信息冗余**：在自然语言中，超过一定距离的位置信息对预测的贡献很小
2. **内容信息补偿**：Attention机制可以通过内容相似度来弥补位置信息的损失
3. **实际性能验证**：实验结果表明，ReRoPE在测试集上的性能甚至优于标准RoPE

### 12. 自回归解码的优化

**定义12.1（KV缓存策略）**

在自回归解码中，标准做法是缓存已计算的Key和Value：

$$\text{Cache}^{(t)} = \{(\boldsymbol{k}_1, \boldsymbol{v}_1), \ldots, (\boldsymbol{k}_t, \boldsymbol{v}_t)\}$$

**推导12.2（ReRoPE的KV缓存修改）**

由于ReRoPE对Key应用的旋转角度取决于Query的位置，标准的KV缓存策略需要修改：

**标准RoPE**：可以缓存$\boldsymbol{\mathcal{R}}^j\boldsymbol{k}_j$

**ReRoPE**：只能缓存$\boldsymbol{k}_j$，然后在每步解码时重新计算旋转

具体地，在第$t$步解码时（$t > w$），对于位置$j$的Key：

$$\boldsymbol{k}_j' = \begin{cases}
\boldsymbol{\mathcal{R}}^{t-j}\boldsymbol{k}_j, & t - j < w \\
\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_j, & t - j \geq w
\end{cases}$$

**命题12.3（解码阶段的计算复杂度）**

对于第$t$步解码（$t > w$），ReRoPE的额外计算量为：

$$\mathcal{O}(t \cdot d)$$

这是因为需要为$t$个历史Key重新计算旋转。

**优化12.4（Query为单个token时的简化）**

当Query序列长度为1时（逐token解码），可以只计算一次Attention矩阵：

$$a_{t,j} = \boldsymbol{q}_t^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-t,-w)}\boldsymbol{k}_j\right)$$

其中$\max(j-t, -w)$确保相对位置不超过$w$。注意这里$j \leq t$，所以$j-t \leq 0$，因此：

$$\max(j-t, -w) = \begin{cases}
j-t, & j > t - w \\
-w, & j \leq t - w
\end{cases}$$

这样只需要对Key序列应用一次旋转即可。

### 13. ReRoPE与注意力掩码的关系

**定义13.1（因果掩码）**

在自回归语言模型中，因果掩码定义为：

$$M_{i,j} = \begin{cases}
0, & i \geq j \\
-\infty, & i < j
\end{cases}$$

**定义13.2（窗口掩码）**

窗口注意力使用窗口掩码：

$$M_{i,j}^{(w)} = \begin{cases}
0, & i - w < j \leq i \\
-\infty, & \text{otherwise}
\end{cases}$$

**推导13.3（ReRoPE的隐式掩码）**

虽然ReRoPE不使用显式的窗口掩码，但其效果类似于软窗口掩码。对于$i - j \geq w$的位置对，ReRoPE使用相同的相对位置编码$w$，这使得Attention权重主要由内容相似度决定，而非位置信息。

具体地，对于$j_1, j_2 < i - w$：

$$\frac{a_{i,j_1}}{a_{i,j_2}} = \frac{\exp(\boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_{j_1})}{\exp(\boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_{j_2})} = \frac{\exp(\boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_{j_1})}{\exp(\boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_{j_2})}$$

位置编码部分$\boldsymbol{\mathcal{R}}^w$对两者相同，因此相对权重完全由内容决定。

### 14. 多层Transformer中的ReRoPE效果

**定义14.1（层级感受野）**

在$L$层Transformer中，第$\ell$层的有效感受野为：

$$\mathcal{F}^{(\ell)}(i) = \bigcup_{j \in \mathcal{F}^{(\ell-1)}(i)} \mathcal{F}^{(1)}(j)$$

其中$\mathcal{F}^{(1)}(i)$是单层的感受野。

**定理14.2（多层ReRoPE的有效覆盖范围）**

对于$L$层Transformer，每层使用窗口大小为$w$的ReRoPE，第$L$层的理论感受野覆盖范围为：

$$\text{Range}^{(L)} = \min(L \cdot w, \text{序列长度})$$

**证明**（归纳法）：

**基础步骤**：第1层的覆盖范围为$w$（显然）。

**归纳步骤**：假设第$\ell$层的覆盖范围为$\min(\ell \cdot w, \text{序列长度})$。在第$\ell+1$层，每个位置可以关注到前$w$个位置，每个这样的位置又可以关注到其前$\ell \cdot w$个位置，因此总覆盖范围为：

$$w + \ell \cdot w = (\ell + 1) \cdot w$$

**推论14.3（所需窗口大小）**

要使$L$层Transformer能够覆盖整个训练长度$L_{\text{train}}$，窗口大小应满足：

$$w \geq \frac{L_{\text{train}}}{L}$$

对于典型的设置（如$L=24$，$L_{\text{train}}=512$），需要$w \geq 21$。实际上，由于注意力权重衰减，通常选择更大的$w$（如128或256）以确保有效的信息传播。

### 15. ReRoPE的频率域分析

**定义15.1（RoPE的频率表示）**

RoPE可以视为多个频率分量的组合。对于维度$d$，有$d/2$个频率：

$$\Omega = \{\theta_0, \theta_1, \ldots, \theta_{d/2-1}\}$$

其中$\theta_i = 10000^{-2i/d}$。

**推导15.2（ReRoPE在频率域的效果）**

ReRoPE对每个频率分量的影响可以分别分析。对于频率$\theta_i$，相对位置$m$对应的相位为：

$$\phi_i(m) = m \cdot \theta_i$$

ReRoPE后的相位为：

$$\phi_i^{\text{ReRoPE}}(m) = \min(m, w) \cdot \theta_i = \begin{cases}
m \cdot \theta_i, & m < w \\
w \cdot \theta_i, & m \geq w
\end{cases}$$

**引理15.3（高频分量的截断效应）**

对于高频分量（$\theta_i$较大），窗口$w$内可能包含多个完整的周期。设周期为$T_i = 2\pi/\theta_i$，则窗口内的周期数为：

$$N_i = \frac{w}{T_i} = \frac{w \cdot \theta_i}{2\pi}$$

当$N_i \gg 1$时，高频分量在窗口内已经完成了多次振荡，截断对高频信息的影响较小。

**引理15.4（低频分量的信息保留）**

对于低频分量（$\theta_i$较小），窗口$w$内可能只包含部分周期。但低频分量对应长距离依赖，其主要作用在于区分不同的长距离位置。ReRoPE通过保持窗口内的完整低频信息，同时统一窗口外的低频表示，实现了长距离信息的有效压缩。

### 16. ReRoPE的实验验证理论

**定义16.1（困惑度指标）**

对于测试序列$\boldsymbol{x} = (x_1, \ldots, x_L)$，困惑度定义为：

$$\text{PPL} = \exp\left(-\frac{1}{L}\sum_{i=1}^{L} \log p(x_i | x_{<i})\right)$$

**定理16.2（ReRoPE的困惑度单调性）**

在理想情况下，使用ReRoPE时，困惑度应该关于序列长度单调递减：

$$\frac{\partial \text{PPL}}{\partial L} \leq 0$$

**直觉证明**：更长的上下文提供更多信息，有助于更准确地预测下一个token，因此困惑度应该降低。由于ReRoPE不引入未见过的位置编码，模型能够有效利用所有上下文信息，因此应该满足这个单调性。

**推导16.3（实验结果的理论解释）**

实验结果显示，ReRoPE在4096长度上的困惑度（85.12%）远高于训练长度512的困惑度（49.40%）。注意这里的指标是准确率而非困惑度，准确率越高越好。

准确率从49.40%提升到85.12%，说明模型在长上下文下的预测能力大幅提升，这验证了"Longer Context, Lower Loss"的理论预测。

### 17. 与其他长度外推方法的统一框架

**定义17.1（通用位置编码函数族）**

定义参数化的位置编码函数族：

$$\mathcal{F} = \{f(m; \boldsymbol{\theta}) : \mathbb{Z}_{\geq 0} \to \mathbb{R}_{\geq 0}\}$$

其中$\boldsymbol{\theta}$是参数向量。

**命题17.2（各种方法在统一框架下的表示）**

1. **标准RoPE**：$f_{\text{RoPE}}(m) = m$
2. **位置内插**：$f_{\text{PI}}(m; k) = m/k$
3. **ReRoPE**：$f_{\text{ReRoPE}}(m; w) = \min(m, w)$
4. **Leaky ReRoPE**：$f_{\text{Leaky}}(m; w, k) = \begin{cases} m, & m < w \\ w + \frac{m-w}{k}, & m \geq w \end{cases}$

**定理17.3（理想位置编码函数的性质）**

一个理想的位置编码函数应满足：

1. **局域性保持**：对于小的$m$，$f(m) \approx m$
2. **外推避免**：$\sup_m f(m) < L_{\text{train}}$
3. **单调性**：$f(m_1) \leq f(m_2)$ when $m_1 < m_2$
4. **连续性**：$f$是连续函数

ReRoPE满足性质1（当$m < w$）、2、3，但不满足性质4（在$m=w$处不可导）。Leaky ReRoPE满足所有四个性质。

### 18. ReRoPE的Transformer架构集成

**推导18.1（多头注意力中的ReRoPE）**

在多头注意力（Multi-Head Attention, MHA）中，每个头独立应用ReRoPE：

$$\text{MHA}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\boldsymbol{W}^O$$

其中：

$$\text{head}_i = \text{Attention}_{\text{ReRoPE}}(\boldsymbol{Q}\boldsymbol{W}_i^Q, \boldsymbol{K}\boldsymbol{W}_i^K, \boldsymbol{V}\boldsymbol{W}_i^V)$$

所有头使用相同的窗口大小$w$。

**推导18.2（Layer Normalization的影响）**

Layer Normalization对Attention输出进行归一化：

$$\text{LayerNorm}(\boldsymbol{x}) = \gamma \cdot \frac{\boldsymbol{x} - \mu}{\sigma} + \beta$$

ReRoPE不影响Layer Normalization的计算，因为它只修改了Attention内部的计算，输出维度和分布特性保持不变。

**推导18.3（残差连接的兼容性）**

残差连接：

$$\boldsymbol{x}^{(\ell+1)} = \boldsymbol{x}^{(\ell)} + \text{Attention}_{\text{ReRoPE}}(\boldsymbol{x}^{(\ell)})$$

ReRoPE与残差连接完全兼容，不需要任何修改。

### 19. ReRoPE的数值稳定性

**定义19.1（数值稳定性指标）**

定义Attention分数的动态范围：

$$\Delta = \max_{i,j} a_{i,j} - \min_{i,j} a_{i,j}$$

**命题19.2（ReRoPE的动态范围控制）**

ReRoPE通过限制相对位置的最大值，隐式地控制了Attention分数的动态范围：

$$\Delta_{\text{ReRoPE}} \leq \Delta_{\text{base}} + \|q\|_2 \|k\|_2 \cdot (\text{effect of max pos} = w)$$

相比之下，标准RoPE在长序列上的动态范围会持续增大：

$$\Delta_{\text{RoPE}} \leq \Delta_{\text{base}} + \|q\|_2 \|k\|_2 \cdot (\text{effect of max pos} = L-1)$$

当$L \gg w$时，$\Delta_{\text{ReRoPE}} \ll \Delta_{\text{RoPE}}$，因此ReRoPE具有更好的数值稳定性。

**推导19.3（Softmax计算的稳定性）**

Softmax计算：

$$\text{softmax}(a_i)_j = \frac{\exp(a_{i,j})}{\sum_{k} \exp(a_{i,k})}$$

当动态范围$\Delta$很大时，会出现数值上溢或下溢问题。ReRoPE通过限制$\Delta$，提高了Softmax计算的数值稳定性。

### 20. 训练策略与ReRoPE的配合

**定义20.1（课程学习策略）**

在训练过程中逐渐增加序列长度：

$$L_t = \min(L_0 + \alpha t, L_{\text{max}})$$

其中$t$是训练步数，$\alpha$是增长率。

**推导20.2（ReRoPE与课程学习的协同）**

使用ReRoPE时，课程学习的效果更加显著，因为：

1. 初期使用短序列（$L < w$），ReRoPE等价于标准RoPE，模型可以充分学习局部依赖
2. 后期使用长序列（$L > w$），ReRoPE的截断机制开始发挥作用，模型学习如何在有限的位置信息下利用内容信息

**推导20.3（Dropout与ReRoPE）**

Attention Dropout：

$$\text{Attention}_{i,j}' = \text{Attention}_{i,j} \cdot \text{Bernoulli}(1-p)$$

ReRoPE与Dropout完全兼容。有趣的是，ReRoPE的截断效应可以视为一种结构化的Dropout，因为它减少了远距离位置的位置信息，迫使模型更多地依赖内容信息。

### 21. Flash Attention兼容性分析

**定义21.1（Flash Attention算法）**

Flash Attention通过分块计算和重计算来减少内存访问：

```
对于每个Query块 Q_i:
    对于每个Key-Value块 (K_j, V_j):
        计算 S_ij = Q_i @ K_j^T
        计算 P_ij = softmax(S_ij)
        累积 O_i += P_ij @ V_j
```

**命题21.2（ReRoPE的Flash Attention实现挑战）**

标准ReRoPE难以直接集成到Flash Attention中，因为：

1. 需要根据$i-j < w$的条件选择不同的RoPE旋转角度
2. 这种条件选择在分块计算中难以高效实现

**推导21.3（可能的解决方案）**

一种可能的解决方案是修改Flash Attention的kernel，在计算$S_{ij}$时应用ReRoPE的条件逻辑：

```python
if i - j < w:
    apply standard RoPE with angle (i-j)
else:
    apply truncated RoPE with angle w
```

这需要自定义CUDA kernel，增加了实现复杂度。

### 22. ReRoPE的变体与扩展

**定义22.1（软ReRoPE）**

定义平滑版本的ReRoPE，使用软截断函数：

$$f_{\text{soft}}(m; w, \beta) = w \cdot \tanh\left(\frac{m}{w} \cdot \beta\right) / \tanh(\beta)$$

当$\beta \to \infty$时，$f_{\text{soft}} \to f_{\text{ReRoPE}}$。

**推导22.2（软ReRoPE的可导性）**

软ReRoPE的导数为：

$$\frac{\partial f_{\text{soft}}}{\partial m} = \frac{\beta}{w \tanh(\beta)} \cdot \text{sech}^2\left(\frac{m}{w} \cdot \beta\right)$$

在所有点都连续可导，这可能有助于训练稳定性。

**定义22.3（多尺度ReRoPE）**

对不同的频率分量使用不同的窗口大小：

$$f_{\text{multi}}(m; \{w_i\}_i) = \{f_{\text{ReRoPE}}(m; w_i)\}_i$$

高频分量使用较小的$w_i$（因为高频对应短距离依赖），低频分量使用较大的$w_i$（因为低频对应长距离依赖）。

### 23. ReRoPE的理论局限性

**定理23.1（全局信息传递的限制）**

ReRoPE通过截断相对位置编码，限制了全局信息的直接传递。对于距离超过$L \cdot w$的两个位置（$L$为层数），信息传递需要经过多次间接路径。

**推导23.2（信息传递延迟）**

在标准Transformer中，任意两个位置之间的信息可以在1层内直接交换。在使用ReRoPE的Transformer中，距离为$d$的两个位置需要至少$\lceil d/w \rceil$层才能交换信息。

**推论23.3（所需层数的增加）**

要处理长度为$L$的序列并确保全局信息传递，使用ReRoPE的Transformer需要至少：

$$L_{\text{layers}} \geq \frac{L}{w}$$

这可能需要比标准Transformer更深的网络。

**反驳23.4（实际影响有限）**

虽然存在理论上的信息传递延迟，但实际中：

1. 大部分任务不需要所有位置之间的直接交互
2. 现代LLM通常已经足够深（24层或更多）
3. 实验结果表明ReRoPE的性能优于标准RoPE

因此，这个理论局限性在实际中影响不大。

### 24. 总结与展望

**总结24.1（ReRoPE的核心贡献）**

1. **理论创新**：提出分段线性的相对位置编码，统一了局部和全局位置信息
2. **实践突破**：实现了几乎无限的长度外推能力，同时保持训练效果
3. **实验验证**：大量实验证明了ReRoPE的有效性

**展望24.2（未来研究方向）**

1. **高效实现**：开发与Flash Attention兼容的ReRoPE实现
2. **理论深化**：研究ReRoPE在不同任务上的理论保证
3. **自适应窗口**：探索根据任务和数据自动调整窗口大小$w$的方法
4. **与其他技术结合**：研究ReRoPE与其他长序列技术（如Sparse Attention、Linear Attention）的结合

**结语**：ReRoPE代表了位置编码研究的一个重要进展，通过巧妙的设计实现了理论优雅性和实践有效性的统一。随着进一步的研究和优化，ReRoPE有望成为长序列处理的标准方法之一。

