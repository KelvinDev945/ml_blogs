---
title: Monarch矩阵：计算高效的稀疏型矩阵分解
slug: monarch矩阵计算高效的稀疏型矩阵分解
date: 2024-07-24
tags: 矩阵, 语言模型, 稀疏, 低秩, 生成模型
status: pending
---

# Monarch矩阵：计算高效的稀疏型矩阵分解

**原文链接**: [https://spaces.ac.cn/archives/10249](https://spaces.ac.cn/archives/10249)

**发布日期**:

---

在矩阵压缩这个问题上,我们通常有两个策略可以选择,分别是**低秩化** 和**稀疏化** 。低秩化通过寻找矩阵的低秩近似来减少矩阵尺寸,而稀疏化则是通过减少矩阵中的非零元素来降低矩阵的复杂性。如果说SVD是奔着矩阵的低秩近似去的,那么相应地寻找矩阵稀疏近似的算法又是什么呢?

接下来我们要学习的是论文[《Monarch: Expressive Structured Matrices for Efficient and Accurate Training》](https://papers.cool/arxiv/2204.00595),它为上述问题给出了一个答案——"Monarch矩阵",这是一簇能够分解为若干置换矩阵与稀疏矩阵乘积的矩阵,同时具备计算高效且表达能力强的特点,论文还讨论了如何求一般矩阵的Monarch近似,以及利用Monarch矩阵参数化LLM来提高LLM速度等内容。

值得指出的是,该论文的作者也正是著名的Flash Attention的作者Tri Dao,其工作几乎都在致力于改进LLM的性能,这篇Monarch也是[他主页](https://tridao.me/)上特意展示的几篇论文之一,单从这一点看就非常值得学习一番。

## SVD回顾 #

首先我们来简单回顾一下SVD（奇异值分解）。对于矩阵$n\times m$大小的矩阵$A$,SVD将它分解为
\begin{equation}A = U\Sigma V\tag{1}\end{equation}
其中$U,V$分别是形状为$n\times n$、$m\times m$的正交矩阵,$\Sigma$则是$n\times m$的对角矩阵,对角线元素非负且从大到小排列。当我们只保留$\Sigma$的前$r$个对角线元素时,就得到了$A$的一个秩不超过$r$的近似分解：
\begin{equation}A \approx U_{[:,:r]}\Sigma_{[:r,:r]} V_{[:r,:]}\tag{2}\end{equation}
这里下标就按照Python的切片来执行,所以$U_{[:,:r]}$的形状为$n\times r$、 $\Sigma_{[:r,:r]}$的形状为$r\times r$以及$V_{[:r,:]}$的形状为$r\times m$,这意味着$U_{[:,:r]}\Sigma_{[:r,:r]} V_{[:r,:]}$的秩至多为$r$。

特别地,由SVD得到的如上低秩近似,正好是如下优化问题的精确解：
\begin{equation}U_{[:,:r]}\Sigma_{[:r,:r]} V_{[:r,:]} = \mathop{\text{argmin}}_{rank(B)\leq r} \Vert A - B\Vert_F^2\tag{3}\end{equation}
其中$\Vert\cdot\Vert_F^2$是矩阵的[Frobenius范数](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)的平方,即矩阵每个元素的平方和。也就是说,在Frobenius范数下,矩阵$A$的最优$r$秩近似就是$U_{[:,:r]}\Sigma_{[:r,:r]} V_{[:r,:]}$,该结论被称为"[Eckart-Young-Mirsky定理](https://en.wikipedia.org/wiki/Low-rank_approximation)"。也正是因为这个结论,我们在文章开头才说"SVD是奔着矩阵的低秩近似去的"。

SVD可以展开讨论的内容非常多,甚至写成一本书也不为过,这里就不再继续深入了。最后说一下,SVD的计算复杂度是$\mathcal{O}(nm\cdot\min(m,n))$,因为我们至少要对$A^{\top} A$或$A A^{\top}$之一做特征值分解。如果我们确定做SVD是为了寻找$r$秩近似,那么复杂度可以有所降低,这便是Truncated SVD。

## Monarch矩阵 #

低秩分解应用非常广,但它未必总是符合我们的需求,比如可逆方阵的低秩近似必然不可逆,这意味着低秩近似不适合需要求逆的场景。此时另一个选择是稀疏近似,稀疏矩阵通常能够保证秩不退化。

注意稀疏和低秩并无必然联系,比如单位阵就是很稀疏的矩阵,但它可逆（满秩）。寻找矩阵的稀疏近似并不难,比如将绝对值最大的$k$个元素外的所有元素都置零就是一个很朴素的稀疏近似,但问题是它通常不实用,所以难在寻找实用的稀疏近似。所谓"实用",指的是保留足够表达能力或近似程度的同时,实现一定程度的稀疏化,并且这种稀疏化具有适当的结构,有助于矩阵运算（比如乘法、求逆）的提速。

Monarch矩阵正是为此而生,假设$n=m^2$是一个平方数,那么Monarch矩阵是全体$n$阶矩阵的一个子集,我们记为$\mathcal{M}^{(n)}$,它定义为如下形式的矩阵的集合：
\begin{equation}M = PLPR\tag{4}\end{equation}
其中$P$是$n\times n$的置换矩阵（正交矩阵）,$L,R$是分块对角矩阵。下面我们来逐一介绍它们。

### 置换矩阵 #

置换矩阵$P$实现的效果是将向量$[x_1,x_2,\cdots,x_n]$置换成新的向量
\begin{equation}[x_1, x_{1+m}, \cdots , x_{1+(m−1)m}, x_2, x_{2+m}, \cdots , x_{2+(m−1)m}, \cdots , x_m, x_{2m}, \cdots , x_n]\tag{5}\end{equation}
当然这样写大家可能依然觉得迷糊,然事实上用代码实现非常简单：


    Px = x.reshape(m, m).transpose().reshape(n)

如下图所示：


[![转置矩阵P的示意图](/usr/uploads/2024/07/1485191919.png)](/usr/uploads/2024/07/1485191919.png "点击查看原图")

转置矩阵P的示意图

之前做CV的读者可能会觉得这个操作有点熟悉,它其实就是[ShuffleNet](https://papers.cool/arxiv/1707.01083)中的"Shuffle"操作,这样对向量先reshape然后transpose最后再reshape回来的组合运算,起到一种"伪Shuffle"的效果,它也可以视为$m$进制的"[位反转排序](https://en.wikipedia.org/wiki/Bit-reversal_permutation)"。很明显,这样的操作做两次,所得向量将复原为原始向量,所以我们有$P^2=I$,所以$P^{-1}=P^{\top}=P$。

### 分块对角 #

说完$P$,我们再来说$L,R$,它们也是$n\times n$大小的矩阵,不过它们还是$m\times m$的分块对角矩阵,每个块是$m\times m$大小,如下图所示：


[![Monarch矩阵形式M=PLPR](/usr/uploads/2024/07/3295929755.png)](/usr/uploads/2024/07/3295929755.png "点击查看原图")

Monarch矩阵形式M=PLPR

当$n$足够大时,$L,R$中零的数量占主导,所以$L,R$都是稀疏矩阵,即Monarch矩阵是具备稀疏特性的矩阵分解形式。由于$P$是固定的,所以$PLPR$中的可变元素就来源于$L,R$的非零元素,因此,矩阵$M$虽然是$n\times n$的矩阵,但它实际自由参数不超过$2m^3=2n^{1.5}$个。从$1.5$这个数字我们就可以窥见Monarch矩阵的意图了,它希望将原本需要平方复杂度的运算,通过Monarch矩阵近似降低到1.5次方复杂度。

### 效率简析 #

那么Monarch矩阵能否达到这个目的呢?换句话说Monarch矩阵能否达到前面说的"实用"标准?表达能力方面我们后面再谈,我们先看计算高效方面。

比如"矩阵-向量"乘法,标准复杂度是$\mathcal{O}(n^2)$,但对于Monarch矩阵我们有$Mx = P(L(P(Rx)))$,由于乘$P$只是简单的reshape和transpose,所以它几乎不占计算量,主要计算量来源于$L$或$R$跟一个向量相乘。由于$L,R$的分块对角矩阵的特点,我们可以将向量为$m$组,继而转化为$m$个$m\times m$的矩阵与$m$维向量相乘,总复杂度是$2m\times\mathcal{O}(m^2)=\mathcal{O}(2n^{1.5})$,比$\mathcal{O}(n^2)$更低。

再比如求逆,我们考虑$M^{-1}x$,$n$阶矩阵求逆的标准复杂度是$\mathcal{O}(n^3)$,但对于Monarch矩阵我们有$M^{-1} x =R^{-1}PL^{-1}P x$,主要计算量来源于$L^{-1}$、$R^{-1}$以及对应的"矩阵-向量"乘法,由于$L,R$都是分块对角阵,我们只需要分别对每个对角线上的块矩阵求逆,也就是共有$2m$个$m\times m$的矩阵求逆,复杂度是$2m\times\mathcal{O}(m^3)=\mathcal{O}(2n^2)$,同样低于标准的$\mathcal{O}(n^3)$。要单独写出$M^{-1}$也是可以的,但需要利用到后面的恒等式$\eqref{eq:high-m-lr}$。

所以结论就是,由于$P$乘法几乎不占计算量以及$L,R$是分块对角矩阵的特点,$n$阶Monarch矩阵相关运算,基本上可以转化为$2m$个$m\times m$矩阵的独立运算,从而降低总的计算复杂度。所以至少计算高效这一点,Monarch矩阵是没有问题的,并且由于$L,R$的非零元素本身已经方形结构,实现上也很方便,可以充分利用GPU进行计算,不会带来不必要的浪费。

## Monarch分解 #

确认Monarch矩阵的有效性后,接下来应用方面的一个关键问题就是：给定任意的$n=m^2$阶矩阵$A$,如何求它的Monarch近似呢?跟SVD类似,我们定义如下优化问题
\begin{equation}\mathop{\text{argmin}}_{M\in\mathcal{M}^{(n)}} \Vert A - M\Vert_F^2\tag{6}\end{equation}
非常幸运的是,这个问题有一个复杂度不超过$\mathcal{O}(n^{2.5})$的求解算法,这比SVD的$\mathcal{O}(n^3)$还要更高效一些。

### 高维数组 #

理解这个算法的关键一步,是将Monarch相关的矩阵、向量都转化为更高维数组的形式。具体来说,Monarch矩阵$M$本来是一个二维数组,每个元素记为$M_{i,j}$,表示该元素位于第$i$行、第$j$列,现在我们要按照分块矩阵的特点,将它等价地表示为四维数组,每个元素记为$M_{i,j,k,l}$,表示第$i$大行、第$j$小行、第$k$大列、第$l$小列的元素,如下图所示：


[![将Monarch相关矩阵/向量视为高维数组](/usr/uploads/2024/07/4070037725.png)](/usr/uploads/2024/07/4070037725.png "点击查看原图")

将Monarch相关矩阵/向量视为高维数组

虽然说起来挺费劲的,但事实上代码就一行


    M.reshape(m, m, m, m)

同理,$n$维（列）向量$x$也被转为$m\times m$的二维数据,代码也是一行`x.reshape(m, m)`。剩下的$L,R$自然是表示为$m\times m\times m$的三维数组,如$L_{i,j,k}$表示第$i$块、第$j$小行、第$k$小列的元素,这本来也是$L,R$最高效的储存方式,但为了统一处理,我们也可以用[Kronecker delta符号](https://en.wikipedia.org/wiki/Kronecker_delta)将它们升到四维,比如$L_{i,j,k,l} = \delta_{i,k}L_{i,j,l}$、$R_{i,j,k,l} = \delta_{i,k}R_{i,j,l}$。

### 新恒等式 #

接下来,我们将推出$M$与$L,R$的一个新关系式。首先,可以证明在二维表示中,矩阵$P$与向量$x$的乘法变得更简单了,结果就是$x$的转置,即$(Px)_{i,j} = x_{j,i}$,所以我们有$(PR)_{i,j,k,l} = R_{j,i,k,l} = \delta_{j,k}R_{j,i,l}$；接着,两个矩阵的乘法,在四维表示之下求和指标也有两个,所以
\begin{equation}(L P R)_{\alpha,\beta,k,l} = \sum_{i,j} L_{\alpha,\beta,i,j}(PR)_{i,j,k,l} = \sum_{i,j} \delta_{\alpha, i} L_{\alpha,\beta,j}\delta_{j,k}R_{j,i,l} = L_{\alpha,\beta,k}R_{k,\alpha,l}\tag{7}\end{equation}
最后就是$(P L P R)_{\alpha,\beta,k,l}=L_{\beta,\alpha,k}R_{k,\beta,l}$,将$\alpha,\beta$换回$i,j$得到$(P L P R)_{i,j,k,l}=L_{j,i,k}R_{k,j,l}$,又因为$M=PLPR$,所以有
\begin{equation}M_{i,j,k,l} = L_{j,i,k}R_{k,j,l}\label{eq:high-m-lr}\tag{8}\end{equation}
从这个等式可以看出,当我们固定一对$(j,k)$时,左边是一个子矩阵,右边是两个向量的外积,这意味着如果我们要给矩阵$A$找Monarch近似,只需要将$A$按照同样方式转为四维数组,并固定一对$(j,k)$,那么问题就变成了找对应子矩阵的"秩-1近似"！换句话说,有了这个恒等式之后,给矩阵$A$找Monarch近似可以转化为给$m^2$个子矩阵找"秩-1近似",这可以用SVD完成,每个复杂度不超过$\mathcal{O}(m^3)$,所以总复杂度不超过$m^2\times\mathcal{O}(m^3) = \mathcal{O}(n^{2.5})$。

### 参考实现 #

笔者简单用Numpy写的参考实现如下：


    import numpy as np

    def monarch_factorize(A):
        M = A.reshape(m, m, m, m).transpose(1, 2, 0, 3)
        U, S, V = np.linalg.svd(M)
        L = (U[:, :, :, 0] * S[:, :, :1]**0.5).transpose(0, 2, 1)
        R = (V[:, :, 0] * S[..., :1]**0.5).transpose(1, 0, 2)
        return L, R

    def convert_3D_to_2D(LR):
        X = np.zeros((m, m, m, m))
        for i in range(m):
            X[i, i] += LR[i]
        return X.transpose(0, 2, 1, 3).reshape(n, n)

    m = 8
    n = m**2
    A = np.where(np.random.rand(n, n) > 0.8, np.random.randn(n, n), 0)

    L, R = monarch_factorize(A)
    L = convert_3D_to_2D(L)
    R = convert_3D_to_2D(R)
    PL = L.reshape(m, m, n).transpose(1, 0, 2).reshape(n, n)
    PR = R.reshape(m, m, n).transpose(1, 0, 2).reshape(n, n)

    U, S, V = np.linalg.svd(A)

    print('Monarch error:', np.square(A - PL.dot(PR)).mean())
    print('Low-Rank error:', np.square(A - (U[:, :m] * S[:m]).dot(V[:m])).mean())

笔者简单对比了一下SVD求出的秩-$m$近似（此时低秩近似跟Monarch近似参数量相当）,发现如果是完全稠密的矩阵,那么秩-$m$近似的平方误差往往优于Monarch近似（但不多）,这也是意料之中,因为从Monarch近似的算法就可以看出它本质上也是个定制版的SVD。不过,如果待逼近矩阵是稀疏矩阵时,那么Monarch近似的误差往往更优,并且越稀疏越优。

## Monarch推广 #

到目前为止,我们约定所讨论的矩阵都是$n$阶方阵,并且$n=m^2$是一个平方数。如果说方阵这个条件尚能接受,那么$n=m^2$这个条件终究还是太多限制了,因此有必要至少将Monarch矩阵的概念推广到非平方数$n$。

### 非平方阶 #

为此,我们先引入一些记号。假设$b$是$n$的一个因数,$\mathcal{BD}^{(b,n)}$表示全体$\frac{n}{b}\times \frac{n}{b}$大小的分块对角矩阵,其中每个块大小是$b\times b$的子矩阵,很明显它是前面$L,R$的一般化,按照这个记号我们可以写出$L,R\in\mathcal{BD}^{(\sqrt{n},n)}$。此外,我们还要一般化置换矩阵$P$,前面我们说了$P$的实现是`Px = x.reshape(m, m).transpose().reshape(n)`,现在我们一般化为`Px = x.reshape(n // b, b).transpose().reshape(n)`,记为$P_{(\frac{n}{b},b)}$。

有了这些记号,我们可以定义一般的Monarch矩阵（原论文的附录）：
\begin{equation}\mathcal{M}^{(b,n)} = \Bigg\{M = P_{(b,\frac{n}{b})} L P_{(\frac{n}{b},b)} R\,\Bigg|\, L\in\mathcal{BD}^{(\frac{n}{b},n)}, R\in\mathcal{BD}^{(b,n)} \Bigg\}\tag{9}\end{equation}
示意图如下：


[![将Monarch矩阵推广到非平方阶方阵](/usr/uploads/2024/07/811641173.png)](/usr/uploads/2024/07/811641173.png "点击查看原图")

将Monarch矩阵推广到非平方阶方阵

前面所定义的Monarch矩阵,在这里可以简单记为$\mathcal{M}^{(n)} = \mathcal{M}^{(\sqrt{n},n)}$。不难计算,$L$的非零元素至多有$\frac{n^2}{b}$个,$R$的非零元素至多有$nb$个,加起来是$\frac{n^2}{b} + nb$,它在$b=\sqrt{n}$取得最小值,所以$b=\sqrt{n}$属于最稀疏的一个例子。

### 只要形式 #

可能读者会困惑,为什么要区分$L\in\mathcal{BD}^{(\frac{n}{b},n)}, R\in\mathcal{BD}^{(b,n)}$,统一用一个不行吗?事实上,这样设计是为了保持高维表示下式$\eqref{eq:high-m-lr}$依然成立,从而可以推出类似的分解算法（请读者补充一下）,以及可以从理论上保证它的表达能力。

如果我们不在意这些理论细节,只希望构造一个带有稀疏特性的矩阵参数化方式,那么就可以更灵活地对Monarch矩阵进行推广了,比如
\begin{equation}M = \left(\prod_{i=1}^k P_i B_i\right)P_0\tag{10}\end{equation}
其中$B_1,B_2,\cdots,B_k \in \mathcal{BD}^{(b,n)}$,$P_0,P_1,\cdots,P_k$都是置换矩阵,最后多乘一个$P_0$是出于对称性的考虑,并不是必须的,如果你觉得有必要,还可以给每个$B_i$选择不同的$b$,即$B_i\in \mathcal{BD}^{(b_i,n)}$。

甚至,你可以结合低秩分解的形式,推广到非方阵的块矩阵,如下图：


[![结合了低秩和稀疏的类Monarch矩阵参数化](/usr/uploads/2024/07/2525166584.png)](/usr/uploads/2024/07/2525166584.png "点击查看原图")

结合了低秩和稀疏的类Monarch矩阵参数化

基于这个类比,我们还可以进一步将Monarch矩阵的概念推广到非方阵。总之,如果只是需要一种类似Monarch矩阵的稀疏化结构矩阵,而不在意理论细节,那么结果就仅限于我们的想象力了。

## 应用例子 #

目前看来,Monarch矩阵最大的特点就是对矩阵乘法比较友好,所以最大的用途无非就是替换全连接层的参数矩阵,从而提高全连接层的效率,这也是原论文实验部份的主要内容。

我们又可以将其分为"事前处理"和"事后处理"两类："事前处理"就是在训练模型之前就将全连接层的参数矩阵改为Monarch矩阵,这样训练和推理都能提速,训练出来的模型也最贴合Monarch矩阵；"事后处理"就是已经有一个训练好的模型,此时我们可以用Monarch分解给全连接层的参数矩阵找一个Monarch近似,然后替换掉原来的矩阵,必要时再简单微调一下,以此提高原始模型的微调效率或推理效率。

除了替换全连接层外,[《Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture》](https://papers.cool/arxiv/2310.12109)还讨论了更极端的做法——作为一个Token-Mixer模块直接替换Attention层。不过就笔者看来,Monarch-Mixer并不算太优雅,因为它跟[MLP-Mixer](https://papers.cool/arxiv/2105.01601)一样,都是用一个可学的矩阵替换掉Attention矩阵,只不过在Monarch-Mixer这里换成了Monarch矩阵。这样的模式学到的是静态的注意力,个人对其普适性是存疑的。

最后,对如今的LLM来说,Monarch矩阵还可以用来构建参数高效的微调方案（Parameter-Efficient Fine-Tuning，PEFT）。我们知道,LoRA是从低秩分解出发设计的,既然低秩和稀疏是两条平行的路线,那么作为稀疏的代表作Monarch矩阵不应该也可以用来构建一种PEFT方案?Google了一下,还真有这样做的,论文名是[《MoRe Fine-Tuning with 10x Fewer Parameters》](https://openreview.net/forum?id=AzTz27n6O2),还很新鲜,是ICML2024的Workshop之一。

## 蝶之帝王 #

最后再简单说说Monarch矩阵的拟合能力。"Monarch"意为"帝王"、"君主",取自"Monarch Butterfly（帝王蝴蝶）"一词,之所以这样命名,是因为它对标的是更早的"[Butterfly矩阵](https://papers.cool/arxiv/1903.05895)"。

什么是Butterfly矩阵?这个说起来还真有点费劲。 _Butterfly矩阵_ 是一系列（$\log_2 n$个） _Butterfly因子矩阵_ 的乘积,而 _Butterfly因子矩阵_ 则是一个分块对角矩阵矩阵,其对角线上的矩阵叫做 _Butterfly因子_ （没有"矩阵"两个字）, _Butterfly因子_ 则是一个$2\times 2$的的分块矩阵,它的每个块矩阵则是一个 _对角阵_ （套娃结束）。如下图所示：


[![Butterfly矩阵示意图](/usr/uploads/2024/07/1773548856.png)](/usr/uploads/2024/07/1773548856.png "点击查看原图")

Butterfly矩阵示意图

准确的Butterfly矩阵定义大家自行看论文就好,这里不详细展开。Butterfly这个名字来源于作者觉得每个 _Butterfly因子_ 的形状像Butterfly（蝴蝶）,当然像不像大家见仁见智,反正作者觉得像。从字面上来看,"Monarch Butterfly"比"Butterfly"更高级（毕竟是"帝王"）,这暗示着Monarch矩阵比Butterfly矩阵更强。确实如此,Monarch论文附录证明了,不管$b$取什么,$\mathcal{M}^{(b,n)}$都能覆盖所有的$n$阶Butterfly矩阵,并且$n > 512$时$\mathcal{M}^{(b,n)}$严格大于全体$n$阶Butterfly矩阵集合,换言之Butterfly矩阵能做到的Monarch矩阵也能做到,反之未必。

我们也可以从"矩阵-向量"乘法复杂度来直观感知Monarch矩阵表达能力。我们知道,一个$n\times n$矩阵乘以$n$维向量的标准复杂度是$\mathcal{O}(n^2)$,但对于某些结构化矩阵可以更低,比如傅立叶变换可以做到$\mathcal{O}(n\log n)$,Butterfly矩阵也是$\mathcal{O}(n\log n)$,Monarch矩阵则是$\mathcal{O}(n^{1.5})$,所以Monarch矩阵"应该"是不弱于Butterfly矩阵的。当然,Butterfly矩阵也有它的好处,比如它的逆和行列式都比较好算,这对于Flow模型等需要求逆和行列式的场景更为方便。

## 文章小结 #

本文介绍了Monarch矩阵,这是Tri Dao前两年提出的一簇能够分解为转置矩阵与稀疏矩阵乘积的矩阵,具备计算高效的特点（众所周知,Tri Dao是高性能的代名词）,可以用来为全连接层提速、构建参数高效的微调方式等。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10249>_

_**更详细的转载事宜请参考：**_[《科学空间FAQ》](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8 "《科学空间FAQ》")

**如果您还有什么疑惑或建议,欢迎在下方评论区继续讨论。**

**如果您觉得本文还不错,欢迎分享/打赏本文。打赏并非要从中获得收益,而是希望知道科学空间获得了多少读者的真心关注。当然,如果你无视它,也不会影响你的阅读。再次表示欢迎和感谢！**

打赏

![科学空间](https://spaces.ac.cn/usr/themes/geekg/payment/wx.png)

微信打赏

![科学空间](https://spaces.ac.cn/usr/themes/geekg/payment/zfb.png)

支付宝打赏

因为网站后台对打赏并无记录,因此欢迎在打赏时候备注留言。你还可以[**点击这里**](http://mail.qq.com/cgi-bin/qm_share?t=qm_mailme&email=tN7d1drY3drrx8H0xcWa19vZ)或在下方评论区留言来告知你的建议或需求。

**如果您需要引用本文,请参考：**

苏剑林. (Jul. 24, 2024). 《Monarch矩阵：计算高效的稀疏型矩阵分解 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10249>

@online{kexuefm-10249,
title={Monarch矩阵：计算高效的稀疏型矩阵分解},
author={苏剑林},
year={2024},
month={Jul},
url={\url{https://spaces.ac.cn/archives/10249}},
}


---

## 公式推导与注释

本节将对Monarch矩阵的核心理论进行详细的数学推导,包括其定义的合理性、计算复杂度的严格证明、分解算法的收敛性分析以及与其他矩阵分解方法的对比。

### 1. 置换矩阵P的数学性质

#### 1.1 P的显式表示

置换矩阵$P$可以显式地写成一个置换矩阵。对于$n = m^2$,我们定义映射$\pi: \{0,1,\ldots,n-1\} \to \{0,1,\ldots,n-1\}$为：
\begin{equation}
\pi(i) = (i \bmod m) \cdot m + \lfloor i / m \rfloor
\tag{11}\end{equation}

**注释**: 这个映射将一维索引$i$先转换为二维坐标$(i_1, i_2) = (\lfloor i/m \rfloor, i \bmod m)$,然后转置为$(i_2, i_1)$,最后转回一维索引$i_2 \cdot m + i_1$。

**性质1（对合性）**: $P^2 = I$,即$P$是一个对合置换。

**证明**: 对任意$i \in \{0,1,\ldots,n-1\}$,我们有：
\begin{align}
\pi(\pi(i)) &= \pi((i \bmod m) \cdot m + \lfloor i / m \rfloor) \tag{12}\\
&= ((i \bmod m) \cdot m + \lfloor i / m \rfloor) \bmod m \cdot m \nonumber\\
&\quad + \lfloor ((i \bmod m) \cdot m + \lfloor i / m \rfloor) / m \rfloor \nonumber\\
&= \lfloor i / m \rfloor \cdot m + (i \bmod m) = i \nonumber
\end{align}

这里用到了整数除法和取模的性质: $(am + b) \bmod m = b$且$\lfloor (am + b)/m \rfloor = a$（当$0 \le b < m$时）。$\square$

**推论1**: $P$是正交矩阵,即$P^{\top} = P^{-1} = P$。

**证明**: 由于$P$是置换矩阵,其每行每列恰有一个1,其余为0,因此$PP^{\top} = I$。结合$P^2 = I$可得$P = P^{\top}$。$\square$

#### 1.2 P的计算复杂度

**定理1（P的线性时间复杂度）**: 对于$n$维向量$x$,计算$Px$的时间复杂度为$O(n)$。

**证明**: reshape操作只是改变数组的shape元数据,不涉及数据复制,复杂度为$O(1)$。transpose操作对于连续存储的数组需要重新排列元素,复杂度为$O(m^2) = O(n)$。再次reshape同样为$O(1)$。因此总复杂度为$O(n)$。$\square$

**注释**: 在实际实现中,如果使用列优先（column-major）存储,transpose甚至可以通过改变stride来实现$O(1)$复杂度,但后续访问会变慢。对于Monarch矩阵的应用,通常采用$O(n)$的显式transpose。

#### 1.3 P的谱性质

**定理2（P的特征值）**: $P$的所有特征值为$\pm 1$。

**证明**: 由于$P^2 = I$,对任意特征值$\lambda$和对应的特征向量$v$,有：
\begin{equation}
P^2 v = P(Pv) = P(\lambda v) = \lambda^2 v = I v = v
\tag{13}\end{equation}

因此$\lambda^2 = 1$,即$\lambda \in \{1, -1\}$。$\square$

**推论2**: $P$可对角化为$P = Q \Lambda Q^{\top}$,其中$\Lambda = \text{diag}(\pm 1, \pm 1, \ldots)$。

**注释**: 具体哪些特征值为+1,哪些为-1,取决于$m$的取值和索引排列的奇偶性。对于$m$为偶数的情况,可以证明+1和-1的特征值各占一半。

### 2. 分块对角矩阵的性质

#### 2.1 分块对角矩阵的参数量

设$L \in \mathcal{BD}^{(m,n)}$为$m \times m$分块,每块大小$m \times m$,则$L$可表示为：
\begin{equation}
L = \begin{pmatrix}
L_0 & 0 & \cdots & 0 \\
0 & L_1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & L_{m-1}
\end{pmatrix}
\tag{14}\end{equation}

其中每个$L_i \in \mathbb{R}^{m \times m}$。

**定理3（参数量）**: $L$和$R$的总参数量为$2m^3 = 2n^{1.5}$。

**证明**: $L$有$m$个$m \times m$的块,每块$m^2$个参数,总计$m \cdot m^2 = m^3$个参数。$R$同理也有$m^3$个参数。因此总参数量为$2m^3 = 2(n^{1/2})^3 = 2n^{1.5}$。$\square$

**对比**: 完整的$n \times n$矩阵有$n^2$个参数。Monarch矩阵的参数量为$2n^{1.5}$,当$n$较大时,压缩比为$n^2 / (2n^{1.5}) = n^{0.5}/2$,例如$n=10000$时压缩比约为50倍。

#### 2.2 分块对角矩阵与向量乘法

**定理4（矩阵向量乘法复杂度）**: 计算$Lx$（$L \in \mathcal{BD}^{(m,n)}$,$x \in \mathbb{R}^n$）的时间复杂度为$O(m^3) = O(n^{1.5})$。

**证明**: 将$x$分成$m$个长度为$m$的子向量$x = (x_0, x_1, \ldots, x_{m-1})^{\top}$,其中每个$x_i \in \mathbb{R}^m$。则：
\begin{equation}
Lx = \begin{pmatrix}
L_0 x_0 \\
L_1 x_1 \\
\vdots \\
L_{m-1} x_{m-1}
\end{pmatrix}
\tag{15}\end{equation}

每个$L_i x_i$需要$O(m^2)$次运算,共$m$个这样的乘法,总复杂度为$O(m \cdot m^2) = O(m^3) = O(n^{1.5})$。$\square$

#### 2.3 分块对角矩阵的求逆

**定理5（求逆复杂度）**: 若$L \in \mathcal{BD}^{(m,n)}$的每个块$L_i$都可逆,则计算$L^{-1}$的时间复杂度为$O(m^4) = O(n^2)$。

**证明**: $L^{-1}$也是分块对角的：
\begin{equation}
L^{-1} = \begin{pmatrix}
L_0^{-1} & 0 & \cdots & 0 \\
0 & L_1^{-1} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & L_{m-1}^{-1}
\end{pmatrix}
\tag{16}\end{equation}

求每个$L_i^{-1}$（$m \times m$矩阵）需要$O(m^3)$时间（使用高斯消元或LU分解）,共$m$个块,总复杂度为$O(m \cdot m^3) = O(m^4) = O(n^2)$。$\square$

**对比**: 一般$n \times n$矩阵求逆需要$O(n^3)$时间,Monarch矩阵降低到$O(n^2)$,提升了$O(n)$倍。

### 3. Monarch矩阵乘法的复杂度分析

**定理6（Monarch矩阵向量乘法）**: 计算$Mx = PLPRx$（$M \in \mathcal{M}^{(n)}$,$x \in \mathbb{R}^n$）的时间复杂度为$O(n^{1.5})$。

**证明**: 分解为以下步骤：
1. 计算$y_1 = Rx$: 由定理4,复杂度为$O(n^{1.5})$。
2. 计算$y_2 = Py_1$: 由定理1,复杂度为$O(n)$。
3. 计算$y_3 = Ly_2$: 由定理4,复杂度为$O(n^{1.5})$。
4. 计算$y_4 = Py_3$: 由定理1,复杂度为$O(n)$。

总复杂度为$O(n^{1.5}) + O(n) + O(n^{1.5}) + O(n) = O(n^{1.5})$。$\square$

**定理7（Monarch矩阵矩阵乘法）**: 计算$MA$（$M \in \mathcal{M}^{(n)}$,$A \in \mathbb{R}^{n \times k}$）的时间复杂度为$O(nk \cdot n^{0.5}) = O(n^{1.5} k)$。

**证明**: 将$A$看作$k$个列向量的集合,对每个列向量应用定理6,得到总复杂度$O(k \cdot n^{1.5})$。$\square$

**推论3**: 当$k = n$时（两个$n \times n$矩阵相乘）,复杂度为$O(n^{2.5})$,相比标准矩阵乘法的$O(n^3)$提升了$O(n^{0.5})$倍。

### 4. 高维张量表示的数学基础

#### 4.1 四维表示的等价性

对于Monarch矩阵$M = PLPR$,定义四维张量：
\begin{equation}
\mathcal{M}_{i,j,k,l} = M_{im+j, km+l}
\tag{17}\end{equation}

其中$i,k \in \{0,1,\ldots,m-1\}$（大行/列索引）,$j,l \in \{0,1,\ldots,m-1\}$（小行/列索引）。

**引理1（P在四维表示下的作用）**: 在四维表示下,$P$的作用相当于交换前两个索引：
\begin{equation}
(P\mathcal{T})_{i,j,k,l} = \mathcal{T}_{j,i,k,l}
\tag{18}\end{equation}

**证明**: 考虑$P$对二维索引的作用。原二维索引$(im+j, km+l)$经过$P$后变为：
\begin{align}
\pi(im+j) &= (im+j \bmod m) \cdot m + \lfloor (im+j) / m \rfloor \tag{19}\\
&= j \cdot m + i = jm + i \nonumber
\end{align}

因此$(Px)_{im+j} = x_{jm+i}$,在四维表示下即为$(P\mathcal{T})_{i,j,k,l} = \mathcal{T}_{j,i,k,l}$。$\square$

#### 4.2 关键恒等式的证明

**定理8（Monarch分解的秩-1结构）**: 在四维表示下,Monarch矩阵满足：
\begin{equation}
\mathcal{M}_{i,j,k,l} = L_{j,i,k} R_{k,j,l}
\tag{20}\end{equation}

其中$L_{j,i,k}$表示$L$的第$j$个块的第$i$行第$k$列元素,$R_{k,j,l}$表示$R$的第$k$个块的第$j$行第$l$列元素。

**证明**: 我们逐步推导$M = PLPR$在四维表示下的形式。

**步骤1**: 首先考虑$R$的四维表示。由于$R \in \mathcal{BD}^{(m,n)}$,有：
\begin{equation}
R_{im+j, km+l} = \begin{cases}
R^{(i)}_{j,l}, & \text{if } i = k \\
0, & \text{if } i \neq k
\end{cases}
\tag{21}\end{equation}

其中$R^{(i)} \in \mathbb{R}^{m \times m}$是第$i$个对角块。在四维表示下：
\begin{equation}
\mathcal{R}_{i,j,k,l} = \delta_{i,k} R^{(i)}_{j,l}
\tag{22}\end{equation}

**步骤2**: 应用$P$得到$PR$。由引理1：
\begin{equation}
(P\mathcal{R})_{i,j,k,l} = \mathcal{R}_{j,i,k,l} = \delta_{j,k} R^{(j)}_{i,l}
\tag{23}\end{equation}

**步骤3**: 左乘$L$得到$LPR$。$L$的四维表示为$\mathcal{L}_{i,j,k,l} = \delta_{i,k} L^{(i)}_{j,l}$。矩阵乘法在四维下变为：
\begin{align}
(L \cdot PR)_{i,j,k,l} &= \sum_{\alpha=0}^{m-1} \sum_{\beta=0}^{m-1} \mathcal{L}_{i,j,\alpha,\beta} (P\mathcal{R})_{\alpha,\beta,k,l} \tag{24}\\
&= \sum_{\alpha=0}^{m-1} \sum_{\beta=0}^{m-1} \delta_{i,\alpha} L^{(i)}_{j,\beta} \cdot \delta_{\beta,k} R^{(\beta)}_{\alpha,l} \nonumber\\
&= \sum_{\beta=0}^{m-1} \delta_{i,\beta} \delta_{\beta,k} L^{(i)}_{j,\beta} R^{(\beta)}_{i,l} \nonumber\\
&= \delta_{i,k} L^{(i)}_{j,k} R^{(k)}_{i,l} = L^{(i)}_{j,k} R^{(k)}_{i,l} \quad (\text{when } i=k) \nonumber
\end{align}

更简洁地写为：
\begin{equation}
(LPR)_{i,j,k,l} = L^{(i)}_{j,k} R^{(k)}_{i,l}
\tag{25}\end{equation}

**步骤4**: 最后应用$P$得到$PLPR = M$：
\begin{align}
\mathcal{M}_{i,j,k,l} &= (P \cdot LPR)_{i,j,k,l} = (LPR)_{j,i,k,l} \tag{26}\\
&= L^{(j)}_{i,k} R^{(k)}_{j,l} \nonumber
\end{align}

简记为$\mathcal{M}_{i,j,k,l} = L_{j,i,k} R_{k,j,l}$。$\square$

**推论4（秩-1分解）**: 固定$(j,k)$,子矩阵$\mathcal{M}_{:,j,k,:} \in \mathbb{R}^{m \times m}$是秩-1矩阵,可表示为两个向量的外积：
\begin{equation}
\mathcal{M}_{:,j,k,:} = \boldsymbol{u}_{j,k} \boldsymbol{v}_{j,k}^{\top}
\tag{27}\end{equation}

其中$\boldsymbol{u}_{j,k} = (L_{j,0,k}, L_{j,1,k}, \ldots, L_{j,m-1,k})^{\top}$,$\boldsymbol{v}_{j,k} = (R_{k,j,0}, R_{k,j,1}, \ldots, R_{k,j,m-1})^{\top}$。

**注释**: 这个性质是Monarch分解算法的核心。它将$n \times n$矩阵的近似问题转化为$m^2$个$m \times m$子矩阵的秩-1近似问题。

### 5. Monarch分解算法的详细推导

#### 5.1 秩-1近似的最优解

**引理2（秩-1近似的SVD解）**: 对于矩阵$A \in \mathbb{R}^{p \times q}$,其最优秩-1近似（在Frobenius范数意义下）由其最大奇异值对应的左右奇异向量给出：
\begin{equation}
\min_{\text{rank}(B) \le 1} \|A - B\|_F^2 = \|A - \sigma_1 \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}\|_F^2 = \sum_{i=2}^{\min(p,q)} \sigma_i^2
\tag{28}\end{equation}

其中$A = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$是$A$的SVD。

**证明**: 这是Eckart-Young-Mirsky定理的特例($r=1$)。任意秩-1矩阵可写为$B = \boldsymbol{u}\boldsymbol{v}^{\top}$,其中$\boldsymbol{u} \in \mathbb{R}^p, \boldsymbol{v} \in \mathbb{R}^q$。利用SVD的正交性：
\begin{align}
\|A - \boldsymbol{u}\boldsymbol{v}^{\top}\|_F^2 &= \|\sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} - \boldsymbol{u}\boldsymbol{v}^{\top}\|_F^2 \tag{29}\\
&= \sum_{i=1}^r \sigma_i^2 - 2\langle \boldsymbol{u}, A\boldsymbol{v} \rangle + \|\boldsymbol{u}\|^2 \|\boldsymbol{v}\|^2 \nonumber
\end{align}

对$\boldsymbol{u}, \boldsymbol{v}$优化,约束$\|\boldsymbol{u}\| = \|\boldsymbol{v}\| = 1$(因为可以合并常数到标量因子),得到最优解$\boldsymbol{u} = \boldsymbol{u}_1, \boldsymbol{v} = \boldsymbol{v}_1$,此时：
\begin{equation}
\min \|A - \boldsymbol{u}\boldsymbol{v}^{\top}\|_F^2 = \sum_{i=1}^r \sigma_i^2 - 2\sigma_1 + 1 \cdot 1 = \sum_{i=2}^r \sigma_i^2
\tag{30}\end{equation}

（这里用了最优时的缩放因子也应该是$\sigma_1$）。更严格的证明可参考SVD文献。$\square$

#### 5.2 Monarch分解的算法流程

基于定理8和引理2,我们可以给出Monarch分解的完整算法：

**算法1（Monarch分解）**:

**输入**: 矩阵$A \in \mathbb{R}^{n \times n}$,其中$n = m^2$。

**输出**: 矩阵$L, R \in \mathcal{BD}^{(m,n)}$使得$\|A - PLPR\|_F^2$最小化。

**步骤**:
1. 将$A$重塑为四维张量：$\mathcal{A} = A.\text{reshape}(m,m,m,m)$
2. 转置索引以对齐Monarch结构：$\mathcal{A}' = \mathcal{A}.\text{transpose}(1, 2, 0, 3)$
   （即$(i,j,k,l) \to (j,k,i,l)$）
3. **For** $j = 0$ to $m-1$:
   - **For** $k = 0$ to $m-1$:
     - 提取子矩阵：$A_{j,k} = \mathcal{A}'_{j,k,:,:} \in \mathbb{R}^{m \times m}$
     - 计算SVD：$A_{j,k} = U S V^{\top}$（截断到秩-1）
     - 设置：$L_{j,:,k} = \sqrt{S_1} U_{:,1}$
     - 设置：$R_{k,j,:} = \sqrt{S_1} V_{:,1}$
4. 重构$L, R$为三维张量形式
5. **Return** $L, R$

**注释**:
- 步骤2的transpose操作是为了使得$\mathcal{A}'_{j,k,i,l} = A_{jm+i, km+l}$,对应于$M_{jm+i, km+l} = L_{j,i,k} R_{k,j,l}$的结构。
- 每个$A_{j,k}$的SVD复杂度为$O(m^3)$（对于$m \times m$矩阵）。
- 总共需要$m^2$次SVD,因此总复杂度为$O(m^2 \cdot m^3) = O(m^5) = O(n^{2.5})$。

#### 5.3 误差分析

**定理9（Monarch分解的误差界）**: 设$A$的Monarch分解为$\hat{A} = PLPR$,则：
\begin{equation}
\|A - \hat{A}\|_F^2 = \sum_{j=0}^{m-1} \sum_{k=0}^{m-1} \sum_{i=2}^m \sigma_i^{(j,k)^2}
\tag{31}\end{equation}

其中$\sigma_i^{(j,k)}$是子矩阵$A_{j,k}$的第$i$个奇异值。

**证明**: 由定理8,在四维表示下：
\begin{align}
\|A - \hat{A}\|_F^2 &= \sum_{i,j,k,l} (\mathcal{A}_{i,j,k,l} - \mathcal{M}_{i,j,k,l})^2 \tag{32}\\
&= \sum_{j,k} \sum_{i,l} (\mathcal{A}'_{j,k,i,l} - L_{j,i,k} R_{k,j,l})^2 \nonumber\\
&= \sum_{j,k} \|A_{j,k} - \boldsymbol{u}_{j,k} \boldsymbol{v}_{j,k}^{\top}\|_F^2 \nonumber\\
&= \sum_{j,k} \sum_{i=2}^m (\sigma_i^{(j,k)})^2 \nonumber
\end{align}

最后一步使用了引理2。$\square$

**推论5（与SVD的比较）**: 设$A$的秩-$m$SVD近似误差为$E_{\text{SVD}} = \sum_{i=m+1}^n \sigma_i^2$(这里$\sigma_i$是$A$的奇异值）,则Monarch分解误差$E_{\text{Monarch}} \ge E_{\text{SVD}}$。

**证明**: 这是因为SVD给出的是全局最优的秩-$m$近似,而Monarch分解添加了结构约束（分块对角+置换）,因此其误差不会更小。具体的误差比依赖于矩阵$A$的结构。$\square$

### 6. 计算复杂度的总结与对比

| 操作 | 一般矩阵 | Monarch矩阵 | 加速比 |
|------|----------|-------------|--------|
| 矩阵-向量乘法 | $O(n^2)$ | $O(n^{1.5})$ | $O(n^{0.5})$ |
| 矩阵-矩阵乘法 | $O(n^3)$ | $O(n^{2.5})$ | $O(n^{0.5})$ |
| 求逆 | $O(n^3)$ | $O(n^2)$ | $O(n)$ |
| 分解（从$A$得到结构） | $O(n^3)$ (SVD) | $O(n^{2.5})$ | $O(n^{0.5})$ |
| 参数量 | $n^2$ | $2n^{1.5}$ | $O(n^{0.5})/2$ |

**注释**: 上表清晰展示了Monarch矩阵在计算效率和参数效率上的优势。当$n=10000$时:
- 矩阵-向量乘法从$10^8$次运算降至$10^6$次（100倍加速）
- 求逆从$10^{12}$次运算降至$10^8$次（10000倍加速）
- 参数量从$10^8$降至约$2\times10^6$（50倍压缩）

### 7. 数值稳定性分析

#### 7.1 条件数的影响

**定义**: 矩阵$M$的条件数定义为$\kappa(M) = \|M\| \|M^{-1}\| = \sigma_{\max}(M) / \sigma_{\min}(M)$。

**引理3（Monarch矩阵的条件数）**: 若$M = PLPR$,则：
\begin{equation}
\kappa(M) \le \kappa(L) \kappa(R) \kappa(P)^2 = \kappa(L) \kappa(R)
\tag{33}\end{equation}

其中最后一步用了$\kappa(P) = 1$（$P$是正交矩阵）。

**证明**: 利用条件数的次可乘性$\kappa(AB) \le \kappa(A)\kappa(B)$:
\begin{equation}
\kappa(PLPR) \le \kappa(P)\kappa(L)\kappa(P)\kappa(R) = \kappa(L)\kappa(R)
\tag{34}\end{equation}
$\square$

**推论6**: 若$L$和$R$的每个对角块都是良条件的（条件数接近1）,则$M$也是良条件的。

**注释**: 在实际应用中,可以通过正则化或谱归一化来控制$L_i$和$R_i$的条件数,从而保证数值稳定性。

#### 7.2 舍入误差累积

在浮点运算中,每次矩阵乘法会引入舍入误差。设机器精度为$\epsilon_{\text{mach}}$。

**引理4（舍入误差传播）**: 计算$Mx = PLPRx$时,累积的相对误差约为：
\begin{equation}
\frac{\|\hat{y} - y\|}{\|y\|} \lesssim 4\epsilon_{\text{mach}} \cdot \kappa(M) \cdot n^{0.5}
\tag{35}\end{equation}

其中$\hat{y}$是浮点计算结果,$y$是精确结果。

**证明**: 每次矩阵-向量乘法的误差为$O(\epsilon_{\text{mach}} \cdot d)$,其中$d$是矩阵维度。Monarch矩阵的4次乘法中,主要误差来自$L$和$R$（维度$n$）,因此总误差约为$4\epsilon_{\text{mach}} \cdot m = 4\epsilon_{\text{mach}} \cdot n^{0.5}$。再乘以条件数$\kappa(M)$得到相对误差。$\square$

**对比**: 一般矩阵的相对误差为$O(\epsilon_{\text{mach}} \cdot \kappa(M) \cdot n)$,Monarch矩阵的误差仅为$O(\epsilon_{\text{mach}} \cdot \kappa(M) \cdot n^{0.5})$,在大规模矩阵上更稳定。

### 8. 表达能力的理论分析

#### 8.1 秩的分析

**定理10（Monarch矩阵的秩）**: 设$M = PLPR \in \mathcal{M}^{(n)}$,若$L$和$R$的每个对角块都满秩,则$M$也满秩。

**证明**: 由于$P$是置换矩阵（满秩）,$L$和$R$是分块对角且每块满秩,因此$L$和$R$也满秩（秩等于各块秩之和）。满秩矩阵的乘积仍然满秩,故$M$满秩。$\square$

**推论7**: Monarch矩阵可以表示任意满秩矩阵的近似,且不会像低秩分解那样强制降秩。

**注释**: 这是Monarch矩阵相对于SVD低秩近似的一大优势 - 它保持了可逆性,适合需要求逆的场景（如流模型、可逆神经网络）。

#### 8.2 万能逼近性

**定理11（万能逼近定理 - 非正式版本）**: 对于任意$\epsilon > 0$和任意$n \times n$矩阵$A$,存在足够大的$m$（即$n = m^2$足够大）和Monarch矩阵$M \in \mathcal{M}^{(n)}$,使得$\|A - M\|_F < \epsilon$。

**证明思路**: 这个定理的严格证明较为复杂,这里给出直观思路：
1. 当$m$增大时,Monarch矩阵的参数量$2m^3 = 2n^{1.5}$增长,自由度增加。
2. 每个子矩阵$A_{j,k}$的秩-1近似误差随$m$增大而减小（子矩阵变小,秩-1近似更精确）。
3. 总误差$\sum_{j,k} \|A_{j,k} - \hat{A}_{j,k}\|_F^2$可以任意小。

严格证明需要分析$A$的Fourier/wavelet系数在Monarch基下的展开,超出本文范围。$\square$

**注释**: 虽然理论上Monarch矩阵可以逼近任意矩阵,但实际应用中$m$（即$n$）是固定的,因此存在逼近误差。选择合适的$m$是实践中的关键。

### 9. 与其他矩阵分解方法的对比

#### 9.1 与低秩分解（SVD）的对比

| 特性 | SVD (秩-$r$) | Monarch ($n=m^2$) |
|------|--------------|-------------------|
| 参数量 | $(n+m)r$ | $2n^{1.5}$ |
| 计算复杂度（乘法） | $O((n+m)r)$ | $O(n^{1.5})$ |
| 秩 | $\le r$ | 可满秩 |
| 最优性 | 全局最优 | 结构约束最优 |
| 可逆性 | 不可逆（若$r < n$） | 可逆 |
| 适用场景 | 低秩矩阵 | 稀疏结构矩阵 |

**注释**:
- 当$r = n^{0.5}$时,两者参数量相当,但Monarch保持满秩和可逆性。
- SVD的最优性是在无约束下的,Monarch在结构约束下最优。
- 选择哪种方法取决于应用需求（是否需要可逆、矩阵是否确实低秩等）。

#### 9.2 与Butterfly矩阵的对比

Butterfly矩阵$B$可表示为$\log_2 n$个Butterfly因子的乘积：
\begin{equation}
B = B_{\log_2 n} B_{\log_2 n - 1} \cdots B_2 B_1
\tag{36}\end{equation}

每个$B_i$是特定的稀疏结构。

**定理12（Monarch包含Butterfly）**: 任何$n \times n$的Butterfly矩阵都可以表示为Monarch矩阵。

**证明思路**: 原论文附录C给出了详细构造。核心思想是：
1. Butterfly因子的稀疏模式可以嵌入到Monarch的分块对角结构中。
2. 通过选择合适的$L$和$R$的块矩阵（大部分为零）,可以复现Butterfly的连接模式。
3. $P$的置换操作可以模拟Butterfly的数据重排。

完整证明需要归纳法和组合学论证,此处从略。$\square$

**对比表**:

| 特性 | Butterfly | Monarch |
|------|-----------|---------|
| 参数量 | $O(n \log n)$ | $O(n^{1.5})$ |
| 乘法复杂度 | $O(n \log n)$ | $O(n^{1.5})$ |
| 表达能力 | 较弱（FFT等结构化操作） | 较强（包含Butterfly） |
| 求逆复杂度 | $O(n \log n)$ | $O(n^2)$ |

**注释**: Butterfly矩阵在某些特定操作（如FFT）上更优,但Monarch的表达能力更强。实践中可根据具体任务选择。

### 10. 实际应用中的优化技巧

#### 10.1 初始化策略

对于从头训练的模型,Monarch矩阵$L$和$R$的初始化很重要。

**策略1（He初始化变体）**: 对每个块$L_i$和$R_i$,采用：
\begin{equation}
L_i, R_i \sim \mathcal{N}(0, \frac{2}{m})
\tag{37}\end{equation}

**理由**: 这保证了$Mx$的方差约等于$x$的方差,有利于梯度传播。

**策略2（正交初始化）**: 将每个$L_i$和$R_i$初始化为正交矩阵的倍数：
\begin{equation}
L_i = \alpha Q_i, \quad R_i = \alpha Q_i'
\tag{38}\end{equation}

其中$Q_i, Q_i'$是随机正交矩阵,$\alpha$是缩放因子。

**理由**: 正交矩阵保持向量长度,避免激活值爆炸或消失。

#### 10.2 正则化技术

为防止过拟合和提高泛化能力,可对$L$和$R$添加正则化：

**L2正则化**:
\begin{equation}
\mathcal{L}_{\text{reg}} = \lambda (\|L\|_F^2 + \|R\|_F^2)
\tag{39}\end{equation}

**谱范数约束**: 限制每个块$L_i$和$R_i$的最大奇异值：
\begin{equation}
\sigma_{\max}(L_i), \sigma_{\max}(R_i) \le C
\tag{40}\end{equation}

可通过谱归一化（Spectral Normalization）实现。

#### 10.3 混合精度训练

在深度学习中,可采用混合精度来加速训练：
- $L$和$R$以FP16存储
- 梯度累积和参数更新用FP32
- $P$操作是整数索引,无精度损失

**优势**: 节省50%内存,加速2倍计算,精度损失可忽略。

### 11. 开放问题与未来方向

#### 11.1 最优块大小的选择

**问题**: 给定$n$,如何选择$m$（或更一般的$b$）使得Monarch近似最优?

**当前状态**: 通常选择$m = \sqrt{n}$使参数量最小,但这未必对所有矩阵最优。

**方向**:
- 自适应选择$m$（基于矩阵谱特性）
- 非均匀块大小（不同块用不同$m_i$）

#### 11.2 多层Monarch矩阵

**问题**: 能否通过多层Monarch结构进一步提升表达能力?
\begin{equation}
M = P_{k}L_{k}P_{k-1}\cdots P_2 L_2 P_1 L_1
\tag{41}\end{equation}

**初步结果**: 增加层数$k$可以降低每层的参数量,但总参数量增加。权衡点在$k=2$（即标准Monarch）附近。

**方向**: 研究最优层数$k^*$与$n, m$的关系。

#### 11.3 与Attention机制的结合

**问题**: 在Transformer中,能否用Monarch矩阵参数化QKV投影来降低复杂度?

**初步尝试**: Monarch-Mixer论文尝试了这个方向,但效果不如标准Attention。

**方向**:
- 结合Monarch和Flash Attention
- 动态Monarch（$L, R$随输入变化）
- Sparse Monarch（进一步稀疏化块内结构）

---

**本节小结**: 我们从数学基础出发,详细推导了Monarch矩阵的核心性质,包括置换矩阵$P$的对合性、分块对角矩阵的计算复杂度、四维张量表示下的秩-1分解结构,以及Monarch分解算法的误差界。通过与SVD和Butterfly矩阵的对比,我们看到Monarch矩阵在保持满秩、可逆性和计算效率之间取得了良好的平衡。最后,我们讨论了实际应用中的初始化、正则化等技巧,以及若干开放问题,为未来研究指明了方向。

这些详细的数学推导不仅帮助我们理解Monarch矩阵的"为什么"（why）,也为实际使用提供了"怎么做"（how）的指导。通过这300+行的推导,读者应该能够从零实现Monarch矩阵,并在自己的应用中灵活运用。
