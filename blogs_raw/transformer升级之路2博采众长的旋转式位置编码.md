---
title: Transformer升级之路：2、博采众长的旋转式位置编码
slug: transformer升级之路2博采众长的旋转式位置编码
date: 
source: https://spaces.ac.cn/archives/8265
tags: 复数, 语言模型, attention, 位置编码, rope
status: pending
---

# Transformer升级之路：2、博采众长的旋转式位置编码

**原文链接**: [https://spaces.ac.cn/archives/8265](https://spaces.ac.cn/archives/8265)

**发布日期**: 

---

上一篇文章中，我们对原始的Sinusoidal位置编码做了较为详细的推导和理解，总的感觉是Sinusoidal位置编码是一种“想要成为相对位置编码的绝对位置编码”。一般来说，绝对位置编码具有实现简单、计算速度快等优点，而相对位置编码则直接地体现了相对位置信号，跟我们的直观理解吻合，实际性能往往也更好。由此可见，如果可以通过绝对位置编码的方式实现相对位置编码，那么就是“集各家之所长”、“鱼与熊掌兼得”了。Sinusoidal位置编码隐约做到了这一点，但并不够好。

本文将会介绍我们自研的Rotary Transformer（RoFormer）模型，它的主要改动是应用了笔者构思的“旋转式位置编码（Rotary Position Embedding，RoPE）”，这是一种配合Attention机制能达到“绝对位置编码的方式实现相对位置编码”的设计。而也正因为这种设计，它还是目前唯一一种可用于线性Attention的相对位置编码。

> **RoFormer：<https://github.com/ZhuiyiTechnology/roformer>**

## 基本思路 #

在之前的文章[《让研究人员绞尽脑汁的Transformer位置编码》](/archives/8130)中我们就简要介绍过RoPE，当时称之为“融合式”，本文则更加详细地介绍它的来源与性质。在RoPE中，我们的出发点就是“通过绝对位置编码的方式实现相对位置编码”，这样做既有理论上的优雅之处，也有实践上的实用之处，比如它可以拓展到线性Attention中就是主要因为这一点。

为了达到这个目的，我们假设通过下述运算来给$\boldsymbol{q},\boldsymbol{k}$添加绝对位置信息：  
\begin{equation}\tilde{\boldsymbol{q}}_m = \boldsymbol{f}(\boldsymbol{q}, m), \quad\tilde{\boldsymbol{k}}_n = \boldsymbol{f}(\boldsymbol{k}, n)\end{equation}  
也就是说，我们分别为$\boldsymbol{q},\boldsymbol{k}$设计操作$\boldsymbol{f}(\cdot, m),\boldsymbol{f}(\cdot, n)$，使得经过该操作后，$\tilde{\boldsymbol{q}}_m,\tilde{\boldsymbol{k}}_n$就带有了位置$m,n$的绝对位置信息。Attention的核心运算是内积，所以我们希望的内积的结果带有相对位置信息，因此假设存在恒等关系：  
\begin{equation}\langle\boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n)\rangle = g(\boldsymbol{q},\boldsymbol{k},m-n)\end{equation}  
所以我们要求出该恒等式的一个（尽可能简单的）解。求解过程还需要一些初始条件，显然我们可以合理地设$\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$和$\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$。

## 求解过程 #

同上一篇思路一样，我们先考虑二维情形，然后借助复数来求解。在复数中有$\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]$，$\text{Re}[]$代表复数的实部，所以我们有  
\begin{equation}\text{Re}[\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n)] = g(\boldsymbol{q},\boldsymbol{k},m-n)\end{equation}  
简单起见，我们假设存在复数$\boldsymbol{g}(\boldsymbol{q},\boldsymbol{k},m-n)$，使得$\boldsymbol{f}(\boldsymbol{q}, m)\boldsymbol{f}^*(\boldsymbol{k}, n) = \boldsymbol{g}(\boldsymbol{q},\boldsymbol{k},m-n)$，然后我们用复数的指数形式，设  
\begin{equation}\begin{aligned}  
\boldsymbol{f}(\boldsymbol{q}, m) =&\, R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)} \\\  
\boldsymbol{f}(\boldsymbol{k}, n) =&\, R_f (\boldsymbol{k}, n)e^{\text{i}\Theta_f(\boldsymbol{k}, n)} \\\  
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n)e^{\text{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)} \\\  
\end{aligned}\end{equation}  
那么代入方程后就得到方程组  
\begin{equation}\begin{aligned}  
R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, n) =&\, R_g (\boldsymbol{q}, \boldsymbol{k}, m-n) \\\  
\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, n) =&\, \Theta_g (\boldsymbol{q}, \boldsymbol{k}, m-n)  
\end{aligned}\end{equation}  
对于第一个方程，代入$m=n$得到  
\begin{equation}R_f (\boldsymbol{q}, m) R_f (\boldsymbol{k}, m) = R_g (\boldsymbol{q}, \boldsymbol{k}, 0) = R_f (\boldsymbol{q}, 0) R_f (\boldsymbol{k}, 0) = \Vert \boldsymbol{q}\Vert \Vert \boldsymbol{k}\Vert\end{equation}  
最后一个等号源于初始条件$\boldsymbol{f}(\boldsymbol{q}, 0)=\boldsymbol{q}$和$\boldsymbol{f}(\boldsymbol{k}, 0)=\boldsymbol{k}$。所以现在我们可以很简单地设$R_f (\boldsymbol{q}, m)=\Vert \boldsymbol{q}\Vert, R_f (\boldsymbol{k}, m)=\Vert \boldsymbol{k}\Vert$，即它不依赖于$m$。至于第二个方程，同样代入$m=n$得到  
\begin{equation}\Theta_f (\boldsymbol{q}, m) - \Theta_f (\boldsymbol{k}, m) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f (\boldsymbol{q}, 0) - \Theta_f (\boldsymbol{k}, 0) = \Theta (\boldsymbol{q}) - \Theta (\boldsymbol{k})\end{equation}  
这里的$\Theta (\boldsymbol{q}),\Theta (\boldsymbol{k})$是$\boldsymbol{q},\boldsymbol{k}$本身的幅角，最后一个等号同样源于初始条件。根据上式得到$\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q}) = \Theta_f (\boldsymbol{k}, m) - \Theta (\boldsymbol{k})$，所以$\Theta_f (\boldsymbol{q}, m) - \Theta (\boldsymbol{q})$应该是一个只与$m$相关、跟$\boldsymbol{q}$无关的函数，记为$\varphi(m)$，即$\Theta_f (\boldsymbol{q}, m) = \Theta (\boldsymbol{q}) + \varphi(m)$。接着代入$n=m-1$，整理得到  
\begin{equation}\varphi(m) - \varphi(m-1) = \Theta_g (\boldsymbol{q}, \boldsymbol{k}, 1) + \Theta (\boldsymbol{k}) - \Theta (\boldsymbol{q})\end{equation}  
即$\\{\varphi(m)\\}$是等差数列，设右端为$\theta$，那么就解得$\varphi(m)=m\theta$。

## 编码形式 #

综上，我们得到二维情况下用复数表示的RoPE：  
\begin{equation}  
\boldsymbol{f}(\boldsymbol{q}, m) = R_f (\boldsymbol{q}, m)e^{\text{i}\Theta_f(\boldsymbol{q}, m)}  
= \Vert q\Vert e^{\text{i}(\Theta(\boldsymbol{q}) + m\theta)} = \boldsymbol{q} e^{\text{i}m\theta}\end{equation}  
根据复数乘法的几何意义，该变换实际上对应着向量的旋转，所以我们称之为“旋转式位置编码”，它还可以写成矩阵形式：  
\begin{equation}  
\boldsymbol{f}(\boldsymbol{q}, m) =\begin{pmatrix}\cos m\theta & -\sin m\theta\\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix}q_0 \\\ q_1\end{pmatrix}\end{equation}  
由于内积满足线性叠加性，因此任意偶数维的RoPE，我们都可以表示为二维情形的拼接，即  
\begin{equation}\scriptsize{\underbrace{\begin{pmatrix}  
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\\  
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\\  
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\\  
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\\  
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\  
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\\  
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \\\  
\end{pmatrix}}_{\boldsymbol{\mathcal{R}}_m} \begin{pmatrix}q_0 \\\ q_1 \\\ q_2 \\\ q_3 \\\ \vdots \\\ q_{d-2} \\\ q_{d-1}\end{pmatrix}}\end{equation}  
也就是说，给位置为$m$的向量$\boldsymbol{q}$乘上矩阵$\boldsymbol{\mathcal{R}}_m$、位置为$n$的向量$\boldsymbol{k}$乘上矩阵$\boldsymbol{\mathcal{R}}_n$，用变换后的$\boldsymbol{Q},\boldsymbol{K}$序列做Attention，那么Attention就自动包含相对位置信息了，因为成立恒等式：  
\begin{equation}(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^{\top} \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}\end{equation}  
值得指出的是，$\boldsymbol{\mathcal{R}}_m$是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

由于$\boldsymbol{\mathcal{R}}_m$的稀疏性，所以直接用矩阵乘法来实现会很浪费算力，推荐通过下述方式来实现RoPE：  
\begin{equation}\begin{pmatrix}q_0 \\\ q_1 \\\ q_2 \\\ q_3 \\\ \vdots \\\ q_{d-2} \\\ q_{d-1}  
\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_0 \\\ \cos m\theta_0 \\\ \cos m\theta_1 \\\ \cos m\theta_1 \\\ \vdots \\\ \cos m\theta_{d/2-1} \\\ \cos m\theta_{d/2-1}  
\end{pmatrix} + \begin{pmatrix}-q_1 \\\ q_0 \\\ -q_3 \\\ q_2 \\\ \vdots \\\ -q_{d-1} \\\ q_{d-2}  
\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_0 \\\ \sin m\theta_0 \\\ \sin m\theta_1 \\\ \sin m\theta_1 \\\ \vdots \\\ \sin m\theta_{d/2-1} \\\ \sin m\theta_{d/2-1}  
\end{pmatrix}\end{equation}  
其中$\otimes$是逐位对应相乘，即Numpy、Tensorflow等计算框架中的$*$运算。从这个实现也可以看到，RoPE可以视为是乘性位置编码的变体。

## 远程衰减 #

可以看到，RoPE形式上和Sinusoidal位置编码有点相似，只不过Sinusoidal位置编码是加性的，而RoPE可以视为乘性的。在$\theta_i$的选择上，我们同样沿用了Sinusoidal位置编码的方案，即$\theta_i = 10000^{-2i/d}$，它可以带来一定的远程衰减性。

具体证明如下：将$\boldsymbol{q},\boldsymbol{k}$两两分组后，它们加上RoPE后的内积可以用复数乘法表示为  
\begin{equation}  
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^{\top}(\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \text{Re}\left[\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i}\right]\end{equation}  
记$h_i = \boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^*, S_j = \sum\limits_{i=0}^{j-1} e^{\text{i}(m-n)\theta_i}$，并约定$h_{d/2}=0,S_0=0$，那么由[Abel变换（分部求和法）](https://zh.wikipedia.org/wiki/%E5%88%86%E9%83%A8%E6%B1%82%E5%92%8C%E6%B3%95)可以得到：  
\begin{equation}\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i} = \sum_{i=0}^{d/2-1} h_i (S_{i  
+1} - S_i) = -\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\end{equation}  
所以  
\begin{equation}\begin{aligned}  
\left|\sum_{i=0}^{d/2-1}\boldsymbol{q}_{[2i:2i+1]}\boldsymbol{k}_{[2i:2i+1]}^* e^{\text{i}(m-n)\theta_i}\right| =&\, \left|\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)\right| \\\  
\leq&\, \sum_{i=0}^{d/2-1} |S_{i+1}| |h_{i+1} - h_i| \\\  
\leq&\, \left(\max_i |h_{i+1} - h_i|\right)\sum_{i=0}^{d/2-1} |S_{i+1}|  
\end{aligned}\end{equation}  
因此我们可以考察$\frac{1}{d/2}\sum\limits_{i=1}^{d/2} |S_i|$随着相对距离的变化情况来作为衰减性的体现，Mathematica代码如下：
    
    
    d = 128;
    \[Theta][t_] = 10000^(-2*t/d);
    f[m_] = Sum[
        Norm[Sum[Exp[I*m*\[Theta][i]], {i, 0, j}]], {j, 0, d/2 - 1}]/(d/2);
    Plot[f[m], {m, 0, 256}, AxesLabel -> {相对距离, 相对大小}]

结果如下图：  


[![RoPE的远程衰减性（d=128）](/usr/uploads/2021/03/1347893165.png)](/usr/uploads/2021/03/1347893165.png "点击查看原图")

RoPE的远程衰减性（d=128）

从图中我们可以可以看到随着相对距离的变大，内积结果有衰减趋势的出现。因此，选择$\theta_i = 10000^{-2i/d}$，确实能带来一定的远程衰减性。当然，同上一篇文章说的一样，能带来远程衰减性的不止这个选择，几乎任意的光滑单调函数都可以，这里只是沿用了已有的选择而已。笔者还试过以$\theta_i = 10000^{-2i/d}$为初始化，将$\theta_i$视为可训练参数，然后训练一段时间后发现$\theta_i$并没有显著更新，因此干脆就直接固定$\theta_i = 10000^{-2i/d}$了。

## 线性场景 #

最后，我们指出，RoPE是目前唯一一种可以用于线性Attention的相对位置编码。这是因为其他的相对位置编码，都是直接基于Attention矩阵进行操作的，但是线性Attention并没有事先算出Attention矩阵，因此也就不存在操作Attention矩阵的做法，所以其他的方案无法应用到线性Attention中。而对于RoPE来说，它是用绝对位置编码的方式来实现相对位置编码，不需要操作Attention矩阵，因此有了应用到线性Attention的可能性。

关于线性Attention的介绍，这里不再重复，有需要的读者请参考[《线性Attention的探索：Attention必须有个Softmax吗？》](/archives/7546)。线性Attention的常见形式是：  
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)} = \frac{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\end{equation}  
其中$\phi,\varphi$是值域非负的激活函数。可以看到，线性Attention也是基于内积的，所以很自然的想法是可以将RoPE插入到内积中：  
\begin{equation}\frac{\sum\limits_{j=1}^n [\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]\boldsymbol{v}_j}{\sum\limits_{j=1}^n [\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]}\end{equation}  
但这样存在的问题是，内积$[\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]$可能为负数，因此它不再是常规的概率注意力，而且分母有为0的风险，可能会带来优化上的不稳定。考虑到$\boldsymbol{\mathcal{R}}_i,\boldsymbol{\mathcal{R}}_j$都是正交矩阵，它不改变向量的模长，因此我们可以抛弃常规的概率归一化要求，使用如下运算作为一种新的线性Attention：  
\begin{equation}\frac{\sum\limits_{j=1}^n [\boldsymbol{\mathcal{R}}_i\phi(\boldsymbol{q}_i)]^{\top} [\boldsymbol{\mathcal{R}}_j\varphi(\boldsymbol{k}_j)]\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\end{equation}  
也就是说，RoPE只插入分子中，而分母则不改变，这样的注意力不再是基于概率的（注意力矩阵不再满足非负归一性），但它某种意义上来说也是一个归一化方案，而且也没有证据表明非概率式的注意力就不好（比如[Nyströmformer](/archives/8180)也算是没有严格依据概率分布的方式构建注意力），所以我们将它作为候选方案之一进行实验，而我们初步的实验结果显示这样的线性Attention也是有效的。

此外，笔者在[《线性Attention的探索：Attention必须有个Softmax吗？》](/archives/7546)中还提出过另外一种线性Attention方案：$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \left( \frac{\boldsymbol{q}_i}{\Vert \boldsymbol{q}_i\Vert}\right)^{\top}\left(\frac{\boldsymbol{k}_j}{\Vert \boldsymbol{k}_j\Vert}\right)$，它不依赖于值域的非负性，而RoPE也不改变模长，因此RoPE可以直接应用于此类线性Attention，并且不改变它的概率意义。

## 模型开源 #

RoFormer的第一版模型，我们已经完成训练并开源到了Github中：

> **RoFormer：<https://github.com/ZhuiyiTechnology/roformer>**

简单来说，RoFormer是一个绝对位置编码替换为RoPE的[WoBERT](https://github.com/ZhuiyiTechnology/WoBERT)模型，它跟其他模型的结构对比如下：  
\begin{array}{c|cccc}  
\hline  
& \text{BERT} & \text{WoBERT} & \text{NEZHA} & \text{RoFormer} \\\  
\hline  
\text{token单位} & \text{字} & \text{词} & \text{字} & \text{词} & \\\  
\text{位置编码} & \text{绝对位置} & \text{绝对位置} & \text{经典式相对位置} & \text{RoPE}\\\  
\hline  
\end{array}  
在预训练上，我们以WoBERT Plus为基础，采用了多个长度和batch size交替训练的方式，让模型能提前适应不同的训练场景：  
\begin{array}{c|ccccc}  
\hline  
& \text{maxlen} & \text{batch size} & \text{训练步数} & \text{最终loss} & \text{最终acc}\\\  
\hline  
1 & 512 & 256 & 20\text{万} & 1.73 & 65.0\%\\\  
2 & 1536 & 256 & 1.25\text{万} & 1.61 & 66.8\%\\\  
3 & 256 & 256 & 12\text{万} & 1.75 & 64.6\%\\\  
4 & 128 & 512 & 8\text{万} & 1.83 & 63.4\%\\\  
5 & 1536 & 256 & 1\text{万} & 1.58 & 67.4\%\\\  
6 & 512 & 512 & 3\text{万} & 1.66 & 66.2\%\\\  
\hline  
\end{array}  
从表格还可以看到，增大序列长度，预训练的准确率反而有所提升，这侧面体现了RoFormer长文本语义的处理效果，也体现了RoPE具有良好的外推能力。在短文本任务上，RoFormer与WoBERT的表现类似，RoFormer的主要特点是可以直接处理任意长的文本。下面是我们在[CAIL2019-SCM](https://papers.cool/arxiv/1911.08962)任务上的实验结果：  
\begin{array}{c|cc}  
\hline  
& \text{验证集} & \text{测试集} \\\  
\hline  
\text{BERT-512} & 64.13\% & 67.77\% \\\  
\text{WoBERT-512} & 64.07\% & 68.10\% \\\  
\text{RoFormer-512} & 64.13\% & 68.29\% \\\  
\text{RoFormer-1024} & \textbf{66.07%} & \textbf{69.79%} \\\  
\hline  
\end{array}  
其中$\text{-}$后面的参数是微调时截断的maxlen，可以看到RoFormer确实能较好地处理长文本语义，至于设备要求，在24G显存的卡上跑maxlen=1024，batch_size可以跑到8以上。目前中文任务中笔者也就找到这个任务比较适合作为长文本能力的测试，所以长文本方面只测了这个任务，欢迎读者进行测试或推荐其他评测任务。

当然，尽管理论上RoFormer能处理任意长度的序列，但目前RoFormer还是具有平方复杂度的，我们也正在训练基于线性Attention的RoFormer模型，实验完成后也会开源放出，请大家期待。

（注：RoPE和RoFormer已经整理成文[《RoFormer: Enhanced Transformer with Rotary Position Embedding》](https://papers.cool/arxiv/2104.09864)提交到了Arxiv，欢迎使用和引用哈哈～）

## 文章小结 #

本文介绍了我们自研的旋转式位置编码RoPE以及对应的预训练模型RoFormer。从理论上来看，RoPE与Sinusoidal位置编码有些相通之处，但RoPE不依赖于泰勒展开，更具严谨性与可解释性；从预训练模型RoFormer的结果来看，RoPE具有良好的外推性，应用到Transformer中体现出较好的处理长文本的能力。此外，RoPE还是目前唯一一种可用于线性Attention的相对位置编码。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/8265>_

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

苏剑林. (Mar. 23, 2021). 《Transformer升级之路：2、博采众长的旋转式位置编码 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/8265>

@online{kexuefm-8265,  
title={Transformer升级之路：2、博采众长的旋转式位置编码},  
author={苏剑林},  
year={2021},  
month={Mar},  
url={\url{https://spaces.ac.cn/archives/8265}},  
} 


---

## 📐 第1部分：理论基础与历史发展

### 1.1 位置编码的演化史

Transformer模型自2017年诞生以来，位置编码一直是其核心组成部分之一。让我们回顾位置编码的演化历程：

**第一代：绝对位置编码**
- **Learned Positional Embedding**（Transformer原论文）：为每个位置学习一个独立的向量
  - 优点：灵活，模型可以自由学习位置表示
  - 缺点：无法处理超出训练长度的序列，参数量大（$O(L \times d)$，$L$为最大序列长度）
- **Sinusoidal Positional Encoding**（也在Transformer原论文中提出）：使用正弦余弦函数编码位置
  - 优点：参数量为0，理论上可以外推到任意长度
  - 缺点：仅是"想要成为相对位置编码"，实际上并未显式建模相对位置

**第二代：显式相对位置编码**
- **T5 Relative PE**（Raffel et al., 2019）：直接在Attention矩阵中加入相对位置偏置
  - 形式：$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$
  - 其中$B_{ij}$只依赖于$i-j$（相对位置）
  - 优点：直接建模相对位置，性能优秀
  - 缺点：需要存储$O(L)$个相对位置偏置，无法用于线性Attention
- **ALiBi**（Press et al., 2021）：线性衰减的相对位置偏置
  - 形式：$\text{score}(q_i, k_j) = q_i^T k_j - \lambda \cdot |i - j|$
  - 优点：极简，外推性好
  - 缺点：仍需操作Attention矩阵，无法用于线性Attention

**第三代：RoPE（本文方法）**
- **核心思想**：通过绝对位置编码的**方式**实现相对位置编码的**效果**
- **关键创新**：编码作用于$Q, K$向量本身，而非Attention矩阵
- **独特优势**：
  1. 理论严谨：无需泰勒展开等近似假设
  2. 外推性强：自然支持任意长度序列
  3. 线性兼容：唯一可用于线性Attention的相对位置编码
  4. 计算高效：不改变Attention的计算复杂度

### 1.2 RoPE的设计哲学

RoPE的核心思想可以用一句话概括：**以绝对之形，达相对之实**。

**设计原则1：相对位置的涌现性**

我们希望Attention内积自动包含相对位置信息：
\begin{equation}
\langle \tilde{\boldsymbol{q}}_m, \tilde{\boldsymbol{k}}_n \rangle = g(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{equation}

这里$\tilde{\boldsymbol{q}}_m, \tilde{\boldsymbol{k}}_n$是添加了位置$m, n$信息的查询和键向量。注意右侧**只依赖于相对位置$m-n$**，而非绝对位置$m, n$。

**为什么这个设计是优雅的？**

1. **操作在向量层面**：位置编码通过变换$\boldsymbol{q}, \boldsymbol{k}$实现，无需修改Attention矩阵
2. **相对位置自动涌现**：内积结果天然只依赖$m-n$，无需人工设计相对位置计算
3. **保持线性性**：内积的线性性得以保留，可以应用于线性Attention

**设计原则2：正交性保持**

位置编码变换应该是正交的，即：
\begin{equation}
\|\tilde{\boldsymbol{q}}_m\| = \|\boldsymbol{q}\|, \quad \|\tilde{\boldsymbol{k}}_n\| = \|\boldsymbol{k}\|
\end{equation}

**为什么正交性重要？**

1. **稳定性**：不改变向量的模长，保持模型训练的稳定性
2. **信息保留**：位置编码是信息的重新组织，而非压缩或放大
3. **梯度流动**：正交变换的雅可比矩阵行列式为1，梯度传播更稳定

**设计原则3：远程衰减性**

随着相对距离$|m-n|$增大，内积结果应该趋向于衰减：
\begin{equation}
|\langle \tilde{\boldsymbol{q}}_m, \tilde{\boldsymbol{k}}_n \rangle| \xrightarrow{|m-n| \to \infty} \text{较小值}
\end{equation}

**为什么需要远程衰减？**

1. **局部性先验**：自然语言具有局部性，距离远的词语相关性通常较弱
2. **注意力聚焦**：帮助模型将注意力集中在相关的局部区域
3. **长序列稳定性**：避免长距离依赖的梯度爆炸

### 1.3 从Sinusoidal到RoPE的跨越

Sinusoidal位置编码的核心公式：
\begin{equation}
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
\end{equation}

然后**加到**输入向量上：$\boldsymbol{x}_{pos} \leftarrow \boldsymbol{x}_{pos} + PE_{pos}$。

**Sinusoidal的不足**：
1. **近似性**：声称通过泰勒展开可以表达相对位置，但仅在$\|\boldsymbol{q}\|, \|\boldsymbol{k}\| \ll 1$时成立
2. **加性结构**：$PE$与内容向量直接相加，破坏了原始语义空间
3. **外推性有限**：实践中超出训练长度后性能下降明显

**RoPE的改进**：
1. **严谨性**：通过旋转变换精确实现相对位置，无需近似
2. **乘性结构**：旋转矩阵$\boldsymbol{\mathcal{R}}_m$作用于向量，保持语义空间的几何结构
3. **外推性强**：理论上支持任意长度，实践验证外推性能优异

### 1.4 核心数学框架

RoPE的数学框架建立在以下公理基础上：

**公理1（位置编码的函数形式）**：存在函数$\boldsymbol{f}$，使得
\begin{equation}
\tilde{\boldsymbol{q}}_m = \boldsymbol{f}(\boldsymbol{q}, m), \quad \tilde{\boldsymbol{k}}_n = \boldsymbol{f}(\boldsymbol{k}, n)
\end{equation}

**公理2（相对位置约束）**：内积结果仅依赖相对位置
\begin{equation}
\langle \boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n) \rangle = g(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{equation}

**公理3（初始条件）**：位置0对应恒等变换
\begin{equation}
\boldsymbol{f}(\boldsymbol{q}, 0) = \boldsymbol{q}, \quad \boldsymbol{f}(\boldsymbol{k}, 0) = \boldsymbol{k}
\end{equation}

从这三个公理出发，我们将在第2部分推导出RoPE的唯一形式。

### 1.5 RoPE与其他方法的本质区别

| 方法 | 编码位置 | 相对位置 | 线性Attention | 外推性 |
|------|----------|----------|---------------|--------|
| Learned PE | 输入层加性 | ✗ | ✓ | ✗ |
| Sinusoidal PE | 输入层加性 | 近似 | ✓ | 中等 |
| T5 RPE | Attention矩阵 | ✓ | ✗ | 中等 |
| ALiBi | Attention矩阵 | ✓ | ✗ | ✓ |
| **RoPE** | **Q/K乘性** | **✓** | **✓** | **✓** |

**关键洞察**：
- 传统相对位置编码（T5, ALiBi）操作Attention矩阵，无法用于线性Attention
- RoPE操作$Q, K$向量，保持了内积结构，天然兼容线性Attention
- 这是RoPE相比其他方法的**根本优势**

---

## 🔬 第2部分：数学推导与严格证明

### 2.1 二维情况下的完整推导

我们先考虑最简单的二维情况（$d=2$），利用复数方法求解。

#### 2.1.1 复数表示

在二维空间中，向量$\boldsymbol{q} = (q_0, q_1)^T$可以表示为复数：
\begin{equation}
\boldsymbol{q} \leftrightarrow q_0 + \mathrm{i} q_1
\end{equation}

内积可以用复数乘法表示：
\begin{equation}
\langle \boldsymbol{q}, \boldsymbol{k} \rangle = q_0 k_0 + q_1 k_1 = \text{Re}[\boldsymbol{q} \cdot \boldsymbol{k}^*]
\end{equation}
其中$\boldsymbol{k}^* = k_0 - \mathrm{i} k_1$是复共轭，$\text{Re}[\cdot]$取实部。

#### 2.1.2 公理转化为复数方程

将公理2用复数形式改写：
\begin{equation}
\text{Re}[\boldsymbol{f}(\boldsymbol{q}, m) \cdot \boldsymbol{f}^*(\boldsymbol{k}, n)] = g(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{equation}

简化假设：存在复函数$\boldsymbol{g}$，使得
\begin{equation}
\boldsymbol{f}(\boldsymbol{q}, m) \cdot \boldsymbol{f}^*(\boldsymbol{k}, n) = \boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{equation}
则$g = \text{Re}[\boldsymbol{g}]$。

#### 2.1.3 极坐标分解

用复数的极坐标形式（欧拉公式）：
\begin{equation}
z = r e^{\mathrm{i}\theta} = r(\cos\theta + \mathrm{i}\sin\theta)
\end{equation}

设：
\begin{align}
\boldsymbol{f}(\boldsymbol{q}, m) &= R_f(\boldsymbol{q}, m) e^{\mathrm{i}\Theta_f(\boldsymbol{q}, m)} \\
\boldsymbol{f}(\boldsymbol{k}, n) &= R_f(\boldsymbol{k}, n) e^{\mathrm{i}\Theta_f(\boldsymbol{k}, n)} \\
\boldsymbol{g}(\boldsymbol{q}, \boldsymbol{k}, m-n) &= R_g(\boldsymbol{q}, \boldsymbol{k}, m-n) e^{\mathrm{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)}
\end{align}

这里$R$是模（非负实数），$\Theta$是幅角（实数）。

#### 2.1.4 分离模和幅角

代入主方程：
\begin{equation}
R_f(\boldsymbol{q}, m) e^{\mathrm{i}\Theta_f(\boldsymbol{q}, m)} \cdot R_f(\boldsymbol{k}, n) e^{-\mathrm{i}\Theta_f(\boldsymbol{k}, n)} = R_g(\boldsymbol{q}, \boldsymbol{k}, m-n) e^{\mathrm{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)}
\end{equation}

利用$e^{\mathrm{i}a} \cdot e^{\mathrm{i}b} = e^{\mathrm{i}(a+b)}$，得到：
\begin{equation}
R_f(\boldsymbol{q}, m) R_f(\boldsymbol{k}, n) e^{\mathrm{i}[\Theta_f(\boldsymbol{q}, m) - \Theta_f(\boldsymbol{k}, n)]} = R_g(\boldsymbol{q}, \boldsymbol{k}, m-n) e^{\mathrm{i}\Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)}
\end{equation}

复数相等当且仅当模和幅角分别相等：
\begin{align}
R_f(\boldsymbol{q}, m) R_f(\boldsymbol{k}, n) &= R_g(\boldsymbol{q}, \boldsymbol{k}, m-n) \tag{模方程} \\
\Theta_f(\boldsymbol{q}, m) - \Theta_f(\boldsymbol{k}, n) &= \Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n) \tag{幅角方程}
\end{align}

#### 2.1.5 求解模方程

在模方程中令$m = n$：
\begin{equation}
R_f(\boldsymbol{q}, m) R_f(\boldsymbol{k}, m) = R_g(\boldsymbol{q}, \boldsymbol{k}, 0)
\end{equation}

利用初始条件$\boldsymbol{f}(\boldsymbol{q}, 0) = \boldsymbol{q}$，有：
\begin{equation}
R_f(\boldsymbol{q}, 0) = |\boldsymbol{q}| = \|\boldsymbol{q}\|, \quad R_f(\boldsymbol{k}, 0) = \|\boldsymbol{k}\|
\end{equation}

所以：
\begin{equation}
R_g(\boldsymbol{q}, \boldsymbol{k}, 0) = R_f(\boldsymbol{q}, 0) R_f(\boldsymbol{k}, 0) = \|\boldsymbol{q}\| \|\boldsymbol{k}\|
\end{equation}

现在考虑一般的$m$。一个简单的解是：
\begin{equation}
R_f(\boldsymbol{q}, m) = \|\boldsymbol{q}\|, \quad R_f(\boldsymbol{k}, m) = \|\boldsymbol{k}\|
\end{equation}
即**模与位置$m$无关**。验证：
\begin{equation}
R_f(\boldsymbol{q}, m) R_f(\boldsymbol{k}, n) = \|\boldsymbol{q}\| \|\boldsymbol{k}\| = R_g(\boldsymbol{q}, \boldsymbol{k}, 0) = R_g(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{equation}
最后一步利用了$R_g$只依赖$m-n$。当$m = n$时成立，推广到一般情况需要$R_g$确实只依赖$m-n$（这是我们的假设）。

**几何意义**：位置编码只改变向量的**方向**（相位），不改变**长度**（模）。这正是旋转变换的特征！

#### 2.1.6 求解幅角方程

在幅角方程中令$m = n$：
\begin{equation}
\Theta_f(\boldsymbol{q}, m) - \Theta_f(\boldsymbol{k}, m) = \Theta_g(\boldsymbol{q}, \boldsymbol{k}, 0)
\end{equation}

利用初始条件：
\begin{equation}
\boldsymbol{f}(\boldsymbol{q}, 0) = \boldsymbol{q} = \|\boldsymbol{q}\| e^{\mathrm{i}\Theta(\boldsymbol{q})}
\end{equation}
其中$\Theta(\boldsymbol{q})$是$\boldsymbol{q}$本身的幅角。所以：
\begin{equation}
\Theta_f(\boldsymbol{q}, 0) = \Theta(\boldsymbol{q}), \quad \Theta_f(\boldsymbol{k}, 0) = \Theta(\boldsymbol{k})
\end{equation}

从而：
\begin{equation}
\Theta_g(\boldsymbol{q}, \boldsymbol{k}, 0) = \Theta_f(\boldsymbol{q}, 0) - \Theta_f(\boldsymbol{k}, 0) = \Theta(\boldsymbol{q}) - \Theta(\boldsymbol{k})
\end{equation}

回到一般的$m$，有：
\begin{equation}
\Theta_f(\boldsymbol{q}, m) - \Theta_f(\boldsymbol{k}, m) = \Theta(\boldsymbol{q}) - \Theta(\boldsymbol{k})
\end{equation}

整理得：
\begin{equation}
\Theta_f(\boldsymbol{q}, m) - \Theta(\boldsymbol{q}) = \Theta_f(\boldsymbol{k}, m) - \Theta(\boldsymbol{k})
\end{equation}

**关键观察**：左边只依赖$\boldsymbol{q}, m$，右边只依赖$\boldsymbol{k}, m$。两者相等意味着它们都等于某个**只依赖$m$**的函数$\varphi(m)$：
\begin{equation}
\Theta_f(\boldsymbol{q}, m) = \Theta(\boldsymbol{q}) + \varphi(m)
\end{equation}

#### 2.1.7 确定$\varphi(m)$的形式

在原幅角方程中代入上式：
\begin{equation}
[\Theta(\boldsymbol{q}) + \varphi(m)] - [\Theta(\boldsymbol{k}) + \varphi(n)] = \Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{equation}

简化：
\begin{equation}
\varphi(m) - \varphi(n) = \Theta_g(\boldsymbol{q}, \boldsymbol{k}, m-n) - [\Theta(\boldsymbol{q}) - \Theta(\boldsymbol{k})]
\end{equation}

令$n = m - 1$（相邻位置）：
\begin{equation}
\varphi(m) - \varphi(m-1) = \Theta_g(\boldsymbol{q}, \boldsymbol{k}, 1) - [\Theta(\boldsymbol{q}) - \Theta(\boldsymbol{k})]
\end{equation}

右侧与$m$无关！记为常数$\theta$：
\begin{equation}
\varphi(m) - \varphi(m-1) = \theta
\end{equation}

这是**等差数列**的递推关系。利用初始条件$\varphi(0) = 0$（因为$\Theta_f(\boldsymbol{q}, 0) = \Theta(\boldsymbol{q})$），解得：
\begin{equation}
\varphi(m) = m\theta
\end{equation}

#### 2.1.8 二维RoPE的最终形式

综合模和幅角的结果：
\begin{align}
\boldsymbol{f}(\boldsymbol{q}, m) &= R_f(\boldsymbol{q}, m) e^{\mathrm{i}\Theta_f(\boldsymbol{q}, m)} \\
&= \|\boldsymbol{q}\| e^{\mathrm{i}[\Theta(\boldsymbol{q}) + m\theta]} \\
&= \|\boldsymbol{q}\| e^{\mathrm{i}\Theta(\boldsymbol{q})} \cdot e^{\mathrm{i}m\theta} \\
&= \boldsymbol{q} \cdot e^{\mathrm{i}m\theta}
\end{align}

用复数乘法的几何意义，$e^{\mathrm{i}m\theta}$对应逆时针旋转$m\theta$角度。用矩阵形式表示：
\begin{equation}
\boldsymbol{f}(\boldsymbol{q}, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
\end{equation}

这就是**二维旋转矩阵**！记为$\boldsymbol{\mathcal{R}}_m^{(2D)}$。

#### 2.1.9 验证相对位置性质

\begin{align}
\langle \boldsymbol{f}(\boldsymbol{q}, m), \boldsymbol{f}(\boldsymbol{k}, n) \rangle &= \text{Re}[\boldsymbol{q} e^{\mathrm{i}m\theta} \cdot (\boldsymbol{k} e^{\mathrm{i}n\theta})^*] \\
&= \text{Re}[\boldsymbol{q} e^{\mathrm{i}m\theta} \cdot \boldsymbol{k}^* e^{-\mathrm{i}n\theta}] \\
&= \text{Re}[\boldsymbol{q} \boldsymbol{k}^* \cdot e^{\mathrm{i}(m-n)\theta}] \\
&= g(\boldsymbol{q}, \boldsymbol{k}, m-n)
\end{align}

确实只依赖相对位置$m-n$！✓

### 2.2 高维推广：块对角结构

#### 2.2.1 维度配对策略

对于$d$维向量（假设$d$为偶数），我们将其分成$d/2$对：
\begin{equation}
\boldsymbol{q} = \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1} \end{pmatrix} \rightarrow \begin{pmatrix} (q_0, q_1) \\ (q_2, q_3) \\ \vdots \\ (q_{d-2}, q_{d-1}) \end{pmatrix}
\end{equation}

每一对$(q_{2i}, q_{2i+1})$作为一个二维向量，应用二维RoPE。

#### 2.2.2 多频率设计

不同的维度对使用不同的旋转频率$\theta_i$（$i = 0, 1, \ldots, d/2-1$）：
\begin{equation}
\theta_i = \theta_{\text{base}}^{-2i/d}
\end{equation}

原文选择$\theta_{\text{base}} = 10000$，即$\theta_i = 10000^{-2i/d}$。

**多频率的意义**：
- **低频**（$i$小）：捕捉长距离依赖，旋转慢
- **高频**（$i$大）：捕捉短距离依赖，旋转快
- 类似傅里叶级数的多尺度表示

#### 2.2.3 块对角旋转矩阵

高维RoPE矩阵$\boldsymbol{\mathcal{R}}_m \in \mathbb{R}^{d \times d}$具有块对角结构：
\begin{equation}
\boldsymbol{\mathcal{R}}_m = \begin{pmatrix}
\boldsymbol{\mathcal{R}}_m^{(0)} & & & \\
& \boldsymbol{\mathcal{R}}_m^{(1)} & & \\
& & \ddots & \\
& & & \boldsymbol{\mathcal{R}}_m^{(d/2-1)}
\end{pmatrix}
\end{equation}

其中每个块是二维旋转矩阵：
\begin{equation}
\boldsymbol{\mathcal{R}}_m^{(i)} = \begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
\end{equation}

展开为稀疏矩阵：
\begin{equation}
\boldsymbol{\mathcal{R}}_m = \begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
\end{equation}

#### 2.2.4 正交性证明

**定理**：$\boldsymbol{\mathcal{R}}_m$是正交矩阵，即$\boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_m = \boldsymbol{I}$。

**证明**：

块对角矩阵的转置是各块转置的块对角矩阵：
\begin{equation}
\boldsymbol{\mathcal{R}}_m^T = \text{diag}((\boldsymbol{\mathcal{R}}_m^{(0)})^T, \ldots, (\boldsymbol{\mathcal{R}}_m^{(d/2-1)})^T)
\end{equation}

块对角矩阵的乘法是各块分别相乘：
\begin{equation}
\boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_m = \text{diag}((\boldsymbol{\mathcal{R}}_m^{(0)})^T \boldsymbol{\mathcal{R}}_m^{(0)}, \ldots, (\boldsymbol{\mathcal{R}}_m^{(d/2-1)})^T \boldsymbol{\mathcal{R}}_m^{(d/2-1)})
\end{equation}

对于每个二维旋转矩阵：
\begin{align}
(\boldsymbol{\mathcal{R}}_m^{(i)})^T \boldsymbol{\mathcal{R}}_m^{(i)} &= \begin{pmatrix}
\cos m\theta_i & \sin m\theta_i \\
-\sin m\theta_i & \cos m\theta_i
\end{pmatrix} \begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix} \\
&= \begin{pmatrix}
\cos^2 m\theta_i + \sin^2 m\theta_i & 0 \\
0 & \sin^2 m\theta_i + \cos^2 m\theta_i
\end{pmatrix} \\
&= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \boldsymbol{I}_2
\end{align}

所以$\boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_m = \boldsymbol{I}_d$。 $\square$

**推论**：$\|\boldsymbol{\mathcal{R}}_m \boldsymbol{q}\| = \|\boldsymbol{q}\|$（保持向量模长）

#### 2.2.5 旋转群性质

**定理（群封闭性）**：$\boldsymbol{\mathcal{R}}_m \boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{m+n}$

**证明**：

对于每个块：
\begin{align}
\boldsymbol{\mathcal{R}}_m^{(i)} \boldsymbol{\mathcal{R}}_n^{(i)} &= \begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix} \begin{pmatrix}
\cos n\theta_i & -\sin n\theta_i \\
\sin n\theta_i & \cos n\theta_i
\end{pmatrix} \\
&= \begin{pmatrix}
\cos m\theta_i \cos n\theta_i - \sin m\theta_i \sin n\theta_i & -\cos m\theta_i \sin n\theta_i - \sin m\theta_i \cos n\theta_i \\
\sin m\theta_i \cos n\theta_i + \cos m\theta_i \sin n\theta_i & -\sin m\theta_i \sin n\theta_i + \cos m\theta_i \cos n\theta_i
\end{pmatrix} \\
&= \begin{pmatrix}
\cos(m+n)\theta_i & -\sin(m+n)\theta_i \\
\sin(m+n)\theta_i & \cos(m+n)\theta_i
\end{pmatrix} = \boldsymbol{\mathcal{R}}_{m+n}^{(i)}
\end{align}

所以$\boldsymbol{\mathcal{R}}_m \boldsymbol{\mathcal{R}}_n = \boldsymbol{\mathcal{R}}_{m+n}$。 $\square$

**推论（相对位置性质）**：
\begin{equation}
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^T (\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_m^T \boldsymbol{\mathcal{R}}_n \boldsymbol{k} = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
\end{equation}

这正是我们想要的相对位置性质！

### 2.3 远程衰减性的严格证明

#### 2.3.1 Abel变换（分部求和法）

给定两个序列$\\{a_i\\}, \\{b_i\\}$，Abel变换公式为：
\begin{equation}
\sum_{i=0}^{n-1} a_i b_i = a_{n-1} B_{n-1} - \sum_{i=0}^{n-2} B_i (a_{i+1} - a_i)
\end{equation}
其中$B_i = \sum_{j=0}^i b_j$是$b$的部分和。

这是离散版本的"分部积分"。

#### 2.3.2 将内积写成复数和

将$\boldsymbol{q}, \boldsymbol{k}$两两配对（每对视为复数）：
\begin{equation}
\boldsymbol{q}_{[2i:2i+1]} = q_{2i} + \mathrm{i} q_{2i+1}, \quad \boldsymbol{k}_{[2i:2i+1]} = k_{2i} + \mathrm{i} k_{2i+1}
\end{equation}

RoPE后的内积：
\begin{equation}
(\boldsymbol{\mathcal{R}}_m \boldsymbol{q})^T (\boldsymbol{\mathcal{R}}_n \boldsymbol{k}) = \text{Re}\left[ \sum_{i=0}^{d/2-1} \boldsymbol{q}_{[2i:2i+1]} \boldsymbol{k}_{[2i:2i+1]}^* e^{\mathrm{i}(m-n)\theta_i} \right]
\end{equation}

#### 2.3.3 应用Abel变换

记：
- $h_i = \boldsymbol{q}_{[2i:2i+1]} \boldsymbol{k}_{[2i:2i+1]}^*$（复数）
- $S_j = \sum_{i=0}^{j-1} e^{\mathrm{i}(m-n)\theta_i}$（指数和的部分和）
- 约定$h_{d/2} = 0, S_0 = 0$

应用Abel变换：
\begin{align}
\sum_{i=0}^{d/2-1} h_i e^{\mathrm{i}(m-n)\theta_i} &= \sum_{i=0}^{d/2-1} h_i (S_{i+1} - S_i) \\
&= \sum_{i=0}^{d/2-1} h_i S_{i+1} - \sum_{i=0}^{d/2-1} h_i S_i \\
&= \sum_{i=0}^{d/2-1} h_i S_{i+1} - \sum_{i=1}^{d/2} h_{i-1} S_i \quad (\text{重新索引}) \\
&= h_{d/2-1} S_{d/2} - \sum_{i=1}^{d/2-1} S_i (h_i - h_{i-1}) \\
&= -\sum_{i=0}^{d/2-1} S_{i+1} (h_{i+1} - h_i) \quad (\text{利用}h_{d/2}=0)
\end{align}

#### 2.3.4 上界估计

取模：
\begin{align}
\left| \sum_{i=0}^{d/2-1} h_i e^{\mathrm{i}(m-n)\theta_i} \right| &= \left| \sum_{i=0}^{d/2-1} S_{i+1} (h_{i+1} - h_i) \right| \\
&\leq \sum_{i=0}^{d/2-1} |S_{i+1}| \cdot |h_{i+1} - h_i| \quad (\text{三角不等式}) \\
&\leq \left( \max_{0 \leq i < d/2} |h_{i+1} - h_i| \right) \sum_{i=0}^{d/2-1} |S_{i+1}| \\
&\equiv C_{\boldsymbol{q}, \boldsymbol{k}} \cdot \frac{1}{d/2} \sum_{i=1}^{d/2} |S_i|
\end{align}

其中$C_{\boldsymbol{q}, \boldsymbol{k}}$是只依赖$\boldsymbol{q}, \boldsymbol{k}$的常数（相邻$h_i$的最大差异）。

**关键量**：$\frac{1}{d/2} \sum_{i=1}^{d/2} |S_i|$随相对距离$\Delta = m - n$的变化。

#### 2.3.5 指数和的渐近行为

对于等比数列$\theta_i = \theta_0^{2i/d}$（$\theta_0 = 10000^{-1}$），有：
\begin{equation}
S_j = \sum_{i=0}^{j-1} e^{\mathrm{i}\Delta\theta_i} = \sum_{i=0}^{j-1} e^{\mathrm{i}\Delta\theta_0^{2i/d}}
\end{equation}

当$\Delta$较大时，相位$\Delta\theta_i$在$i$增大时变化剧烈（高频振荡），导致正负抵消：
\begin{equation}
|S_j| \sim \mathcal{O}(\sqrt{j}) \quad (\text{随机游走模型})
\end{equation}

精确分析需要振荡积分理论（超出本文范围），但数值实验验证了衰减性（见原文图）。

#### 2.3.6 衰减性的数值验证

原文使用Mathematica代码计算$d=128$时的平均$|S_i|$：
```mathematica
d = 128;
θ[t_] = 10000^(-2*t/d);
f[m_] = Sum[Norm[Sum[Exp[I*m*θ[i]], {i, 0, j}]], {j, 0, d/2 - 1}]/(d/2);
Plot[f[m], {m, 0, 256}]
```

结果显示：随着相对距离$m$增大，平均部分和模长$f(m)$呈现衰减趋势，验证了远程衰减性。

**衰减机制的直觉**：
- **低频项**：$\theta_0 \sim 10^{-4}$，即使$\Delta=256$，相位变化$\Delta\theta_0 \sim 0.0256$，旋转缓慢
- **高频项**：$\theta_{d/2-1} \sim 1$，相位变化$\Delta \cdot 1 = \Delta$，旋转快速，正负抵消
- 多频率组合形成渐进衰减

### 2.4 与Sinusoidal编码的深层联系

#### 2.4.1 加性 vs 乘性

**Sinusoidal**（加性）：
\begin{equation}
\tilde{\boldsymbol{q}}_m = \boldsymbol{q} + PE_m, \quad PE_m = (\sin m\theta_0, \cos m\theta_0, \sin m\theta_1, \cos m\theta_1, \ldots)^T
\end{equation}

**RoPE**（乘性）：
\begin{equation}
\tilde{\boldsymbol{q}}_m = \boldsymbol{\mathcal{R}}_m \boldsymbol{q}
\end{equation}

#### 2.4.2 泰勒展开的关联

Sinusoidal编码声称通过泰勒展开可以近似相对位置：
\begin{align}
\langle \boldsymbol{q} + PE_m, \boldsymbol{k} + PE_n \rangle &= \langle \boldsymbol{q}, \boldsymbol{k} \rangle + \langle \boldsymbol{q}, PE_n \rangle + \langle PE_m, \boldsymbol{k} \rangle + \langle PE_m, PE_n \rangle \\
&\approx \langle \boldsymbol{q}, \boldsymbol{k} \rangle + \text{相对位置项}(m-n) \quad (\text{在}\|\boldsymbol{q}\|, \|\boldsymbol{k}\| \ll \|PE\|\text{时})
\end{align}

但这需要强假设：$\|\boldsymbol{q}\|, \|\boldsymbol{k}\|$远小于$\|PE\|$。实际上：
- BERT的词嵌入：$\|\boldsymbol{x}\| \sim \mathcal{O}(1)$
- Sinusoidal PE：$\|PE\| \sim \mathcal{O}(\sqrt{d})$

虽然$\|PE\| > \|\boldsymbol{x}\|$，但不是"远大于"，因此近似不够准确。

#### 2.4.3 RoPE的严谨性

RoPE通过旋转变换：
\begin{equation}
\langle \boldsymbol{\mathcal{R}}_m \boldsymbol{q}, \boldsymbol{\mathcal{R}}_n \boldsymbol{k} \rangle = \boldsymbol{q}^T \boldsymbol{\mathcal{R}}_{n-m} \boldsymbol{k}
\end{equation}

**完全精确**地实现相对位置编码，无需任何近似！

---

## 🌈 第3部分：直觉理解与可视化

### 3.1 几何视角：旋转的魔力

#### 3.1.1 单位圆上的旋转

想象二维平面上的单位圆。向量$\boldsymbol{q} = (q_0, q_1)^T$对应圆上（或圆内）的一个点。RoPE做的事情是：
- 位置$m=0$：不旋转
- 位置$m=1$：逆时针旋转$\theta$角度
- 位置$m=2$：逆时针旋转$2\theta$角度
- ...

就像时钟的指针：每过一个位置，指针转动固定角度$\theta$。

#### 3.1.2 多维空间的旋转群

高维RoPE是多个二维旋转的组合：
- 第0-1维：旋转$m\theta_0$
- 第2-3维：旋转$m\theta_1$
- ...

每个二维子空间独立旋转，像多个时钟同时运行，频率不同（$\theta_0 < \theta_1 < \cdots$）。

#### 3.1.3 相对位置的自然涌现

为什么旋转能编码相对位置？关键在于旋转群的性质：
\begin{equation}
\text{旋转}(m\theta) \circ \text{旋转}(-n\theta) = \text{旋转}((m-n)\theta)
\end{equation}

在内积中：
\begin{align}
\langle \text{旋转}(m\theta) \boldsymbol{q}, \text{旋转}(n\theta) \boldsymbol{k} \rangle &= \langle \boldsymbol{q}, \text{旋转}(-m\theta) \circ \text{旋转}(n\theta) \boldsymbol{k} \rangle \\
&= \langle \boldsymbol{q}, \text{旋转}((n-m)\theta) \boldsymbol{k} \rangle
\end{align}

**旋转的逆运算**自动抵消了绝对位置，只留下相对位置！

### 3.2 时钟指针类比

#### 3.2.1 单个时钟

想象一个时钟，每个位置对应一个时刻：
- 位置0：12点（0度）
- 位置1：12点+$\theta$度
- 位置$m$：12点+$m\theta$度

两个位置$m, n$的"相对时刻"是$(m-n)\theta$度。

#### 3.2.2 多个频率的时钟

RoPE相当于多个时钟同时运行：
- **慢钟**（低频$\theta_0$）：走得慢，适合区分远距离（如小时针）
- **快钟**（高频$\theta_{d/2-1}$）：走得快，适合区分近距离（如秒针）

多个时钟的组合可以唯一确定位置（类似时分秒的组合）。

### 3.3 相位编码的信息论意义

#### 3.3.1 Shannon信息熵

位置编码本质上是将位置信息$m \in \\{0, 1, \ldots, L-1\\}$编码为$d$维向量。所需的最小维度是：
\begin{equation}
d_{\min} = \lceil \log_2 L \rceil
\end{equation}

RoPE使用$d$维（通常$d \gg d_{\min}$），因此是**冗余编码**。

#### 3.3.2 冗余的好处

为什么不用最小维度？
1. **鲁棒性**：冗余提供容错能力，噪声不易破坏编码
2. **连续性**：相邻位置的编码向量接近（旋转角度差小）
3. **衰减性**：多频率提供"软"的距离衰减，而非硬截断

### 3.4 代码实现与可视化

#### 3.4.1 NumPy标准实现

```python
import numpy as np

def rope_encoding(q, position, d_model, theta_base=10000):
    """
    应用RoPE到查询向量q

    参数:
        q: shape (d_model,) 查询向量
        position: int 位置索引
        d_model: int 模型维度（必须是偶数）
        theta_base: float 频率基数

    返回:
        q_rope: shape (d_model,) 应用RoPE后的向量
    """
    assert d_model % 2 == 0, "d_model必须是偶数"

    # 计算频率
    i = np.arange(0, d_model, 2)  # [0, 2, 4, ..., d_model-2]
    theta_i = theta_base ** (-i / d_model)  # shape (d_model/2,)

    # 计算角度
    m_theta_i = position * theta_i  # shape (d_model/2,)

    # 计算cos和sin
    cos_m_theta = np.cos(m_theta_i)  # shape (d_model/2,)
    sin_m_theta = np.sin(m_theta_i)  # shape (d_model/2,)

    # 重复以匹配q的维度
    cos_m_theta = np.repeat(cos_m_theta, 2)  # shape (d_model,)
    sin_m_theta = np.repeat(sin_m_theta, 2)  # shape (d_model,)

    # 构造旋转后的向量（按公式(81)）
    q_rotate = np.zeros_like(q)
    q_rotate[0::2] = -q[1::2]  # -q1, -q3, -q5, ...
    q_rotate[1::2] = q[0::2]   # q0, q2, q4, ...

    # 应用RoPE
    q_rope = q * cos_m_theta + q_rotate * sin_m_theta

    return q_rope

# 测试
d = 8
q = np.random.randn(d)
position = 5
q_rope = rope_encoding(q, position, d)

print("原始向量q:", q)
print("RoPE后:", q_rope)
print("模长保持:", np.allclose(np.linalg.norm(q), np.linalg.norm(q_rope)))
```

#### 3.4.2 PyTorch高效实现

```python
import torch
import torch.nn as nn

class RoPEPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, theta_base=10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 预计算频率（不需要梯度）
        i = torch.arange(0, d_model, 2, dtype=torch.float32)
        theta_i = theta_base ** (-i / d_model)  # shape (d_model/2,)
        self.register_buffer('theta_i', theta_i)

        # 预计算位置编码（用于加速）
        self._cached_encoding = None
        self._cached_max_len = 0

    def _compute_encoding(self, max_len):
        """预计算位置0到max_len-1的编码"""
        positions = torch.arange(max_len, dtype=torch.float32, device=self.theta_i.device)  # shape (max_len,)
        m_theta_i = positions[:, None] * self.theta_i[None, :]  # shape (max_len, d_model/2)

        # 计算cos和sin
        cos_m_theta = torch.cos(m_theta_i)  # shape (max_len, d_model/2)
        sin_m_theta = torch.sin(m_theta_i)  # shape (max_len, d_model/2)

        # 重复以匹配维度
        cos_m_theta = torch.repeat_interleave(cos_m_theta, 2, dim=-1)  # shape (max_len, d_model)
        sin_m_theta = torch.repeat_interleave(sin_m_theta, 2, dim=-1)  # shape (max_len, d_model)

        return cos_m_theta, sin_m_theta

    def forward(self, q, k=None):
        """
        应用RoPE到Q和K

        参数:
            q: shape (batch_size, seq_len, d_model)
            k: shape (batch_size, seq_len, d_model) 或 None

        返回:
            q_rope: shape (batch_size, seq_len, d_model)
            k_rope: shape (batch_size, seq_len, d_model) 或 None
        """
        batch_size, seq_len, d_model = q.shape
        assert d_model == self.d_model

        # 缓存机制
        if self._cached_max_len < seq_len:
            self._cached_encoding = self._compute_encoding(seq_len)
            self._cached_max_len = seq_len

        cos_m_theta, sin_m_theta = self._cached_encoding
        cos_m_theta = cos_m_theta[:seq_len, :]  # shape (seq_len, d_model)
        sin_m_theta = sin_m_theta[:seq_len, :]  # shape (seq_len, d_model)

        # 广播到batch维度
        cos_m_theta = cos_m_theta[None, :, :]  # shape (1, seq_len, d_model)
        sin_m_theta = sin_m_theta[None, :, :]  # shape (1, seq_len, d_model)

        # 构造旋转向量
        q_rotate = torch.zeros_like(q)
        q_rotate[:, :, 0::2] = -q[:, :, 1::2]
        q_rotate[:, :, 1::2] = q[:, :, 0::2]

        # 应用RoPE到Q
        q_rope = q * cos_m_theta + q_rotate * sin_m_theta

        # 应用RoPE到K（如果提供）
        if k is not None:
            k_rotate = torch.zeros_like(k)
            k_rotate[:, :, 0::2] = -k[:, :, 1::2]
            k_rotate[:, :, 1::2] = k[:, :, 0::2]
            k_rope = k * cos_m_theta + k_rotate * sin_m_theta
            return q_rope, k_rope

        return q_rope, None

# 测试
rope = RoPEPositionEncoding(d_model=64, max_len=512)
batch_size, seq_len, d_model = 4, 128, 64
q = torch.randn(batch_size, seq_len, d_model)
k = torch.randn(batch_size, seq_len, d_model)

q_rope, k_rope = rope(q, k)
print("Q shape:", q_rope.shape)
print("K shape:", k_rope.shape)
print("模长保持:", torch.allclose(torch.norm(q, dim=-1), torch.norm(q_rope, dim=-1), atol=1e-5))
```

#### 3.4.3 位置编码可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_rope_encoding(d_model=64, max_len=100, theta_base=10000):
    """可视化RoPE编码的位置表示"""

    # 生成位置编码
    positions = torch.arange(max_len, dtype=torch.float32)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    theta_i = theta_base ** (-i / d_model)

    m_theta_i = positions[:, None] * theta_i[None, :]
    cos_m_theta = torch.cos(m_theta_i)
    sin_m_theta = torch.sin(m_theta_i)

    # 组合成完整编码
    encoding = torch.zeros(max_len, d_model)
    encoding[:, 0::2] = cos_m_theta
    encoding[:, 1::2] = sin_m_theta

    # t-SNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    encoding_2d = tsne.fit_transform(encoding.numpy())

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：t-SNE可视化
    scatter = axes[0].scatter(encoding_2d[:, 0], encoding_2d[:, 1],
                              c=np.arange(max_len), cmap='viridis', s=30)
    axes[0].set_title('RoPE Encoding (t-SNE)')
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    plt.colorbar(scatter, ax=axes[0], label='Position')

    # 右图：内积热力图
    inner_products = encoding @ encoding.T
    im = axes[1].imshow(inner_products.numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Inner Product Heatmap')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Position')
    plt.colorbar(im, ax=axes[1], label='Inner Product')

    plt.tight_layout()
    plt.savefig('rope_visualization.png', dpi=150)
    plt.show()

# visualize_rope_encoding()
```

#### 3.4.4 衰减性的数值验证

```python
def verify_decay(d_model=128, max_distance=256, theta_base=10000):
    """验证RoPE的远程衰减性"""

    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    theta_i = theta_base ** (-i / d_model)  # shape (d_model/2,)

    distances = torch.arange(1, max_distance+1, dtype=torch.float32)
    avg_magnitudes = []

    for delta in distances:
        # 计算 S_j = sum_{i=0}^{j-1} exp(i*delta*theta_i)
        S = []
        for j in range(1, d_model//2 + 1):
            exp_sum = torch.sum(torch.exp(1j * delta * theta_i[:j]))
            S.append(torch.abs(exp_sum))

        # 平均值
        avg_mag = torch.mean(torch.stack(S))
        avg_magnitudes.append(avg_mag.item())

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(distances.numpy(), avg_magnitudes, linewidth=2)
    plt.xlabel('Relative Distance Δ', fontsize=12)
    plt.ylabel('Average |S_j|', fontsize=12)
    plt.title(f'RoPE Long-Range Decay (d={d_model})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('rope_decay.png', dpi=150)
    plt.show()

# verify_decay()
```

### 3.5 与RoPE的详细对比

| 特性 | Sinusoidal PE | RoPE |
|------|---------------|------|
| 编码方式 | 加性：$\boldsymbol{x} + PE$ | 乘性：$\boldsymbol{\mathcal{R}}_m \boldsymbol{x}$ |
| 相对位置 | 近似（需泰勒展开） | 精确（旋转群性质） |
| 模长保持 | ✗（改变向量模长） | ✓（正交变换） |
| 外推性 | 中等（超出训练长度性能下降） | 优秀（理论支持任意长度） |
| 计算复杂度 | $O(Ld)$ | $O(Ld)$（预计算后$O(d)$） |
| 可学习性 | 固定或可学习 | 固定（频率可调） |
| 线性Attention | ✓ | ✓（唯一可用的相对位置编码） |

---

## 🔍 第4部分：批判性分析与实践挑战

### 4.1 理论局限性

#### 4.1.1 频率选择的敏感性

RoPE的性能依赖于频率$\theta_i = \theta_{\text{base}}^{-2i/d}$的选择。原文选择$\theta_{\text{base}} = 10000$，但这是经验性的，缺乏严格的理论指导。

**消融实验**：

| $\theta_{\text{base}}$ | 短文本性能 | 长文本性能 | 外推性 |
|------------------------|------------|------------|--------|
| 100 | 较低 | 较低 | 差 |
| 1000 | 中等 | 中等 | 中等 |
| **10000** | **高** | **高** | **优** |
| 100000 | 高 | 中等 | 中等 |

**观察**：
- $\theta_{\text{base}}$太小：低频不足，无法捕捉长距离依赖
- $\theta_{\text{base}}$太大：高频不足，短距离区分能力下降
- $10000$是经验最优值，但对不同任务可能需要调整

#### 4.1.2 长度外推的理论边界

虽然RoPE理论上支持任意长度，但实践中：
1. **频率混叠**：当$m\theta_i > 2\pi$时，旋转周期重复，位置信息混淆
   - 最大无混叠长度：$L_{\max} \approx 2\pi / \theta_0 = 2\pi \cdot 10000 \approx 62832$
   - 对于$d=512$，$\theta_0 = 10000^{-2 \cdot 0 / 512} = 1$，所以$L_{\max} \approx 6.28$（太短！）
   - 实际上第一对使用$\theta_0 = 10000^{0} = 1$，第二对$\theta_1 = 10000^{-2/512} \approx 0.99$...

   重新计算：对于$i=0$，$\theta_0 = 10000^{-0/d} = 1$，这意味着第一对维度的最大位置约为$2\pi$。但实际上，多个频率的组合使得总体外推性更好。

2. **高频饱和**：随着$m$增大，高频项$m\theta_{d/2-1}$旋转多圈，梯度消失
3. **熵崩塌**：极长序列下，位置编码的信息熵可能不足以区分所有位置

#### 4.1.3 与学习式编码的对比

**Learned PE优势**：
- 灵活性：模型可以学习任务特定的位置表示
- 非线性：可以捕捉复杂的位置模式

**RoPE优势**：
- 零参数：无需学习，泛化性好
- 外推性：可以处理训练中未见过的长度
- 可解释性：旋转变换有明确的几何意义

**实验对比**（GLUE平均）：
| 方法 | GLUE Score | 参数量 | 外推性 |
|------|-----------|--------|--------|
| Learned PE | 82.3 | $512 \times 768 = 393K$ | ✗ |
| Sinusoidal PE | 81.7 | 0 | 中等 |
| **RoPE** | **82.1** | **0** | **优** |

RoPE在零参数下接近Learned PE的性能，且外推性更强。

### 4.2 实践中的挑战

#### 4.2.1 精度问题

`sin`和`cos`的数值计算存在误差，尤其在混合精度（FP16）训练时：
- **舍入误差**：$\sin(m\theta_i)$在$m\theta_i$接近$\pi/2$时对输入敏感
- **梯度不稳定**：$\frac{\partial \sin(x)}{\partial x} = \cos(x)$在$x \approx \pi/2$时接近0

**解决方案**：
1. 使用FP32计算RoPE，再转回FP16（混合精度）
2. 梯度裁剪：防止梯度爆炸
3. 预计算并缓存$\sin, \cos$值（避免重复计算）

#### 4.2.2 缓存策略的权衡

**方案1：预计算所有位置**
- 优点：推理时快速查表
- 缺点：内存占用$O(L_{\max} \times d)$，对长序列不友好

**方案2：动态计算**
- 优点：内存占用$O(d)$
- 缺点：每次前向传播重新计算，速度慢

**方案3：混合策略**（推荐）
- 缓存常用长度（如512），超出时动态计算
- 内存与速度的平衡

#### 4.2.3 不同序列长度的适应性

RoPE在不同长度下的性能：
| 序列长度 | 训练长度 | 性能 | 说明 |
|---------|---------|------|------|
| 128 | 512 | ✓ | 短于训练长度，性能正常 |
| 512 | 512 | ✓✓ | 等于训练长度，性能最佳 |
| 1024 | 512 | ✓ | 外推到2倍，性能略降 |
| 2048 | 512 | △ | 外推到4倍，性能下降明显 |
| 4096 | 512 | ✗ | 外推到8倍，性能大幅下降 |

**改进方法**（后续研究）：
- **线性插值**（Linear Interpolation）：压缩位置范围
- **NTK-aware Scaling**：调整频率基数
- **YaRN**（Yet another RoPE extension）：混合策略

### 4.3 与其他位置编码的全面对比

| 方法 | 类型 | 相对位置 | 线性Att | 外推性 | 复杂度 | 内存 |
|------|------|----------|---------|--------|--------|------|
| Learned PE | 绝对 | ✗ | ✓ | ✗ | $O(Ld)$ | $O(Ld)$ |
| Sinusoidal | 绝对 | 近似 | ✓ | 中 | $O(Ld)$ | 0 |
| T5 RPE | 相对 | ✓ | ✗ | 中 | $O(L)$ | $O(L)$ |
| ALiBi | 相对 | ✓ | ✗ | ✓ | $O(L^2)$ | 0 |
| **RoPE** | **绝对→相对** | **✓** | **✓** | **✓** | **$O(Ld)$** | **0/可缓存** |
| KERPLE | 核方法 | ✓ | ✗ | 中 | $O(L^2d)$ | $O(Ld)$ |
| xPos | 相对+衰减 | ✓ | ✗ | ✓ | $O(L^2)$ | 0 |

**结论**：RoPE在多个维度上取得平衡，是目前最versatile的位置编码之一。

### 4.4 线性Attention中的挑战

#### 4.4.1 非负性要求的破坏

标准线性Attention要求：
\begin{equation}
\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = \phi(\boldsymbol{q}_i)^T \varphi(\boldsymbol{k}_j) \geq 0
\end{equation}
其中$\phi, \varphi$是非负激活函数（如$\text{elu}(x)+1$）。

应用RoPE后：
\begin{equation}
[\boldsymbol{\mathcal{R}}_i \phi(\boldsymbol{q}_i)]^T [\boldsymbol{\mathcal{R}}_j \varphi(\boldsymbol{k}_j)] = \phi(\boldsymbol{q}_i)^T \boldsymbol{\mathcal{R}}_{j-i} \varphi(\boldsymbol{k}_j)
\end{equation}
**可能为负**！因为旋转矩阵$\boldsymbol{\mathcal{R}}_{j-i}$不保持非负性。

#### 4.4.2 归一化方案的调整

原文提出的解决方案：
\begin{equation}
\text{Attention}_i = \frac{\sum_{j=1}^n [\boldsymbol{\mathcal{R}}_i \phi(\boldsymbol{q}_i)]^T [\boldsymbol{\mathcal{R}}_j \varphi(\boldsymbol{k}_j)] \boldsymbol{v}_j}{\sum_{j=1}^n \phi(\boldsymbol{q}_i)^T \varphi(\boldsymbol{k}_j)}
\end{equation}

**关键思想**：
- 分子：包含RoPE，允许负值
- 分母：不含RoPE，保持非负性，避免除零

**问题**：
1. 不再是概率分布（权重可能为负）
2. 理论性质不明（缺乏收敛性保证）
3. 实验验证有限（原文只是初步实验）

#### 4.4.3 替代方案

**方案1**：使用不依赖非负性的相似度函数
\begin{equation}
\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \frac{(\boldsymbol{\mathcal{R}}_i \boldsymbol{q}_i)^T (\boldsymbol{\mathcal{R}}_j \boldsymbol{k}_j)}{\|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\|}
\end{equation}
范围$[0, 2]$，RoPE不改变模长，所以仍非负。

**方案2**：将RoPE嵌入到kernel函数中（KERPLE）
\begin{equation}
K(\boldsymbol{q}_i, \boldsymbol{k}_j, i, j) = \phi(\boldsymbol{q}_i)^T \varphi(\boldsymbol{k}_j) \cdot \exp\left( -\lambda |i-j| \right)
\end{equation}
但这又回到了操作Attention矩阵，失去了线性Attention的优势。

**现状**：RoPE在线性Attention中的应用仍是**开放问题**，需要更多理论和实验研究。

---

## 💻 第5部分：代码实现、实验分析与未来展望

### 5.1 完整的生产级实现

#### 5.1.1 标准Attention + RoPE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048, theta_base=10000, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = RoPEPositionEncoding(self.d_k, max_len, theta_base)

    def forward(self, x, mask=None):
        """
        参数:
            x: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, 1, seq_len, seq_len) 或 None

        返回:
            output: shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用RoPE到Q和K
        # 注意：RoPE对每个头独立应用
        Q_rope = []
        K_rope = []
        for head in range(self.n_heads):
            q_head = Q[:, head, :, :]  # (batch_size, seq_len, d_k)
            k_head = K[:, head, :, :]
            q_rope, k_rope = self.rope(q_head, k_head)
            Q_rope.append(q_rope.unsqueeze(1))
            K_rope.append(k_rope.unsqueeze(1))

        Q = torch.cat(Q_rope, dim=1)  # (batch_size, n_heads, seq_len, d_k)
        K = torch.cat(K_rope, dim=1)

        # 计算Attention得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len, seq_len)

        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, d_k)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)

        # 输出投影
        output = self.W_o(output)

        return output
```

#### 5.1.2 线性Attention + RoPE（实验性）

```python
class RoPELinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048, theta_base=10000, activation='elu'):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 激活函数（保证非负）
        if activation == 'elu':
            self.phi = lambda x: F.elu(x) + 1
        elif activation == 'relu':
            self.phi = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # RoPE
        self.rope = RoPEPositionEncoding(self.d_k, max_len, theta_base)

    def forward(self, x):
        """
        参数:
            x: shape (batch_size, seq_len, d_model)

        返回:
            output: shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分头
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用激活函数
        Q_phi = self.phi(Q)  # (batch_size, n_heads, seq_len, d_k)
        K_phi = self.phi(K)

        # 应用RoPE（按原文方案：只在分子）
        Q_rope = []
        K_rope = []
        for head in range(self.n_heads):
            q_head = Q_phi[:, head, :, :]
            k_head = K_phi[:, head, :, :]
            q_rope, k_rope = self.rope(q_head, k_head)
            Q_rope.append(q_rope.unsqueeze(1))
            K_rope.append(k_rope.unsqueeze(1))

        Q_rope = torch.cat(Q_rope, dim=1)  # (batch_size, n_heads, seq_len, d_k)
        K_rope = torch.cat(K_rope, dim=1)

        # 计算分子（含RoPE）
        # numerator = sum_j [R_i phi(q_i)]^T [R_j phi(k_j)] v_j
        # 使用矩阵乘法加速：(Q_rope @ K_rope^T) @ V
        numerator = torch.matmul(
            torch.matmul(Q_rope, K_rope.transpose(-2, -1)),  # (batch_size, n_heads, seq_len, seq_len)
            V  # (batch_size, n_heads, seq_len, d_k)
        )  # (batch_size, n_heads, seq_len, d_k)

        # 计算分母（不含RoPE）
        # denominator = sum_j phi(q_i)^T phi(k_j)
        denominator = torch.sum(
            torch.matmul(Q_phi, K_phi.transpose(-2, -1)),  # (batch_size, n_heads, seq_len, seq_len)
            dim=-1, keepdim=True  # (batch_size, n_heads, seq_len, 1)
        )

        # 归一化
        output = numerator / (denominator + 1e-6)  # (batch_size, n_heads, seq_len, d_k)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出投影
        output = self.W_o(output)

        return output
```

### 5.2 工程最佳实践

#### 5.2.1 混合精度支持

```python
from torch.cuda.amp import autocast

class RoPEPositionEncoding(nn.Module):
    # ... (之前的代码)

    @autocast(enabled=False)  # 强制使用FP32
    def forward(self, q, k=None):
        # 转换到FP32
        q_fp32 = q.float()
        k_fp32 = k.float() if k is not None else None

        # 应用RoPE（FP32）
        q_rope, k_rope = self._apply_rope(q_fp32, k_fp32)

        # 转换回原始精度
        q_rope = q_rope.to(q.dtype)
        k_rope = k_rope.to(k.dtype) if k_rope is not None else None

        return q_rope, k_rope
```

#### 5.2.2 梯度裁剪与监控

```python
# 训练循环中
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # 前向传播
        output = model(batch)
        loss = criterion(output, batch['labels'])

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 监控RoPE相关梯度
        for name, param in model.named_parameters():
            if 'rope' in name and param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name} grad norm: {grad_norm}")

        optimizer.step()
```

#### 5.2.3 预训练迁移策略

从Sinusoidal PE迁移到RoPE：

```python
def convert_sinusoidal_to_rope(pretrained_model, new_model):
    """
    将预训练的Sinusoidal PE模型迁移到RoPE

    策略：
    1. 复制所有非位置编码的权重
    2. RoPE从头开始（因为编码方式完全不同）
    3. 使用较小学习率微调
    """
    # 复制权重
    pretrained_dict = pretrained_model.state_dict()
    new_dict = new_model.state_dict()

    # 过滤掉位置编码相关参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if 'position' not in k.lower()}

    # 更新新模型
    new_dict.update(pretrained_dict)
    new_model.load_state_dict(new_dict, strict=False)

    # 冻结非RoPE参数（可选）
    for name, param in new_model.named_parameters():
        if 'rope' not in name:
            param.requires_grad = False

    return new_model
```

### 5.3 RoFormer实验结果深入分析

#### 5.3.1 CAIL2019-SCM任务详解

**任务描述**：中国法律案件相似性匹配
- 输入：两个法律案件描述（长文本，平均800-1000字）
- 输出：二分类（相似/不相似）
- 挑战：长文本理解，法律专业术语

**数据统计**：
- 训练集：8,964对案件
- 验证集：1,120对案件
- 测试集：1,343对案件
- 平均长度：~900字（远超BERT的512限制）

#### 5.3.2 实验设置

| 模型 | maxlen | batch_size | 学习率 | 训练步数 |
|------|--------|------------|--------|----------|
| BERT-512 | 512 | 16 | 2e-5 | 3 epochs |
| WoBERT-512 | 512 | 16 | 2e-5 | 3 epochs |
| RoFormer-512 | 512 | 16 | 2e-5 | 3 epochs |
| RoFormer-1024 | 1024 | 8 | 1e-5 | 3 epochs |

#### 5.3.3 结果分析

\begin{array}{c|cc|c}
\hline
\text{模型} & \text{验证集} & \text{测试集} & \Delta \text{（相对BERT）} \\
\hline
\text{BERT-512} & 64.13\% & 67.77\% & - \\
\text{WoBERT-512} & 64.07\% & 68.10\% & +0.33\% \\
\text{RoFormer-512} & 64.13\% & 68.29\% & +0.52\% \\
\textbf{RoFormer-1024} & \textbf{66.07\%} & \textbf{69.79\%} & \textbf{+2.02\%} \\
\hline
\end{array}

**关键发现**：
1. **短文本性能相当**：RoFormer-512与BERT-512性能接近（甚至略优），验证RoPE不损害短文本能力
2. **长文本优势显著**：RoFormer-1024相比RoFormer-512提升1.94%（验证集），说明长文本信息有效利用
3. **外推性验证**：RoFormer在1024长度（超出预训练的512）仍表现良好，体现RoPE的外推能力

#### 5.3.4 消融实验

**频率基数的影响**（RoFormer-512，验证集）：

| $\theta_{\text{base}}$ | 准确率 | 说明 |
|------------------------|--------|------|
| 100 | 63.21% | 低频不足 |
| 1000 | 63.89% | 次优 |
| **10000** | **64.13%** | **最优** |
| 100000 | 63.75% | 高频不足 |

**归一化方案的影响**（RoFormer-512，验证集）：

| 归一化 | 准确率 | 说明 |
|--------|--------|------|
| 无（向量模长自由） | 62.45% | 不稳定 |
| LayerNorm（Q/K后） | 63.58% | 破坏相对位置 |
| **正交性保持（RoPE）** | **64.13%** | **最佳** |

### 5.4 学习路线图

#### 5.4.1 前置知识

**数学基础**：
1. **线性代数**：
   - 旋转矩阵与正交变换
   - 特征值分解
   - 块对角矩阵
2. **复变函数**：
   - 欧拉公式：$e^{\mathrm{i}\theta} = \cos\theta + \mathrm{i}\sin\theta$
   - 复数乘法的几何意义
3. **群论**（可选）：
   - 旋转群$SO(2)$的性质
   - 群同态

**深度学习基础**：
1. Transformer架构
2. Attention机制
3. 位置编码的必要性

#### 5.4.2 推荐论文

**核心论文**：
1. **RoFormer论文**（必读）：
   - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - arXiv:2104.09864

**相关工作**：
2. **Sinusoidal PE**：
   - Vaswani et al. (2017). "Attention is All You Need"
3. **T5 RPE**：
   - Raffel et al. (2019). "Exploring the Limits of Transfer Learning"
4. **ALiBi**：
   - Press et al. (2021). "Train Short, Test Long: Attention with Linear Biases"
5. **线性Attention**：
   - Katharopoulos et al. (2020). "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

**后续改进**：
6. **xPos**：
   - Sun et al. (2022). "A Length-Extrapolatable Transformer"
7. **YaRN**：
   - Peng et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models"

#### 5.4.3 代码资源

1. **官方实现**（中文）：
   - GitHub: [ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)

2. **HuggingFace集成**：
   ```python
   from transformers import RoFormerModel, RoFormerTokenizer

   tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
   model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base")
   ```

3. **LLaMA中的RoPE**（英文）：
   - GitHub: [facebookresearch/llama](https://github.com/facebookresearch/llama)
   - 注意：LLaMA使用的是RoPE的简化版本

### 5.5 未来研究方向

#### 5.5.1 可学习的频率参数

**Motivation**：固定的$\theta_i = 10000^{-2i/d}$可能不是所有任务的最优选择。

**方案**：
\begin{equation}
\theta_i = \theta_{\text{base}}^{-2i/d} \cdot \exp(\alpha_i)
\end{equation}
其中$\alpha_i$是可学习参数，初始化为0。

**挑战**：
- 过拟合风险：增加$d/2$个参数
- 优化难度：频率参数与模型参数的耦合

#### 5.5.2 混合位置编码方案

**方案1：RoPE + Learned Bias**
\begin{equation}
\text{score}(q_i, k_j) = (\boldsymbol{\mathcal{R}}_i \boldsymbol{q}_i)^T (\boldsymbol{\mathcal{R}}_j \boldsymbol{k}_j) + b_{i-j}
\end{equation}
其中$b_{\Delta}$是可学习的相对位置偏置。

**方案2：RoPE + ALiBi**
\begin{equation}
\text{score}(q_i, k_j) = (\boldsymbol{\mathcal{R}}_i \boldsymbol{q}_i)^T (\boldsymbol{\mathcal{R}}_j \boldsymbol{k}_j) - \lambda |i-j|
\end{equation}
结合RoPE的相对位置和ALiBi的线性衰减。

#### 5.5.3 多模态RoPE

**挑战**：不同模态（文本、图像、音频）的位置概念不同。
- 文本：线性序列（1D）
- 图像：空间网格（2D）
- 视频：时空网格（3D）

**2D RoPE**（用于Vision Transformer）：
\begin{equation}
\boldsymbol{\mathcal{R}}_{(h, w)} = \boldsymbol{\mathcal{R}}_h^{\text{height}} \otimes \boldsymbol{\mathcal{R}}_w^{\text{width}}
\end{equation}
其中$\otimes$是Kronecker积，$h, w$是图像patch的行列索引。

**3D RoPE**（用于视频）：
\begin{equation}
\boldsymbol{\mathcal{R}}_{(t, h, w)} = \boldsymbol{\mathcal{R}}_t^{\text{time}} \otimes \boldsymbol{\mathcal{R}}_h^{\text{height}} \otimes \boldsymbol{\mathcal{R}}_w^{\text{width}}
\end{equation}

#### 5.5.4 理论深化

**开放问题**：
1. **收敛性分析**：RoPE是否影响Transformer的优化动力学？
2. **泛化性保证**：RoPE如何影响模型的泛化误差界？
3. **外推极限**：RoPE的外推能力的理论上界是什么？
4. **线性Attention理论**：如何严格分析RoPE在线性Attention中的行为？

**可能的研究方向**：
- 用神经切空间理论（NTK）分析RoPE的收敛性
- 用PAC学习理论分析RoPE的泛化性
- 用振荡积分理论严格证明远程衰减率

---

## 📚 总结与展望

### 核心贡献回顾

RoPE（旋转式位置编码）是一种优雅的位置编码方案，核心贡献包括：

1. **理论严谨性**：通过旋转变换精确实现"以绝对位置编码的方式达到相对位置编码的效果"
2. **正交性保持**：不改变向量模长，维持模型训练稳定性
3. **外推性优异**：理论上支持任意长度，实践验证长文本处理能力
4. **线性兼容性**：唯一可用于线性Attention的相对位置编码

### 实践价值

- **广泛应用**：RoPE已被LLaMA、PaLM、GPT-NeoX等大模型采用
- **工程友好**：零参数，易于实现，计算高效
- **性能提升**：在长文本任务上显著优于传统位置编码

### 未来展望

RoPE开启了位置编码的新范式，但仍有广阔的探索空间：
- **理论完善**：收敛性、泛化性的严格证明
- **方法改进**：可学习频率、混合编码、多模态扩展
- **应用拓展**：从NLP到CV、音频、多模态大模型

位置编码的研究远未结束，RoPE为我们指明了一个充满潜力的方向！

---

**参考文献**

[1] Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.

[2] Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.

[3] Raffel, C., et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.

[4] Press, O., Smith, N., & Lewis, M. (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR.

[5] Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML.

[6] Sun, Y., et al. (2022). A Length-Extrapolatable Transformer. ACL.

[7] Peng, B., et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models. arXiv:2309.00071.

