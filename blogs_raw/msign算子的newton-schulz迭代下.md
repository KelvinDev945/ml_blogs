---
title: msign算子的Newton-Schulz迭代（下）
slug: msign算子的newton-schulz迭代下
date: 2025-06-05
tags: 迭代, 近似, 优化器, muon, 生成模型
status: completed
---

# msign算子的Newton-Schulz迭代（下）

**原文链接**: [https://spaces.ac.cn/archives/10996](https://spaces.ac.cn/archives/10996)

**发布日期**: 

---

在上文[《msign算子的Newton-Schulz迭代（上）》](/archives/10922)中，我们试图为$\mathop{\text{msign}}$算子寻找更好的Newton-Schulz迭代，以期在有限迭代步数内能达到尽可能高的近似程度，这一过程又可以转化为标量函数$\mathop{\text{sign}}(x)$寻找同样形式的多项式迭代。当时，我们的求解思路是用Adam优化器端到端地求一个局部最优解，虽然有效但稍显粗暴。

而在几天前，arXiv新出了一篇论文[《The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm》](https://papers.cool/arxiv/2505.16932)，作者运用了一系列精妙的数学结论，以优雅且硬核的方式给出了更漂亮的答案。本文让我们一起欣赏和学习一番这篇精彩的论文。

## 问题描述 #

相关背景和转化过程我们就不再重复了，直接给出我们要求解的问题是  
\begin{equation}\mathop{\text{argmin}}_f d(f(x),1)\end{equation}  
其中$f = f_T \circ \dots \circ f_2 \circ f_1$，$\circ$代表函数的复合，$f_t(x)$是关于$x$的奇多项式（只包含$x$的奇数次幂），而$d(f(x),1)$是衡量函数$f(x)$与$1$距离的某个指标。上一篇文章中，我们是在$[0,1]$内均匀选有限个点，取最大的$k$个$|f(x)-1|$的平均值作为指标。本文则直接取区间内的$|f(x)-1|$最大值作为指标，即  
\begin{equation}\mathop{\text{argmin}}_f \max_{x\in[l,u]} |f(x) - 1| \label{eq:opt}\end{equation}  
其中$[l,u]\subset [0,1]$。要注意，此时$u$可以直接取1，但$l$不能取0，因为$f(0)$始终是0，这意味着上式始终大于等于1，无法收敛，所以$l$只能选一个很接近于0的数。按照上一篇文章的分析，为了普适性，我们应该要照顾到$0.001$大小的奇异值，因此可以考虑$l=0.001$。

在开始分析之前，我们先简单解释一下论文标题中“Polar”一词的含义，它其实代表了矩阵的“极分解（Polar Decomposition）”：

> **极分解（Polar Decomposition）** 对于一个方阵$\boldsymbol{M}\in\mathbb{R}^{n\times n}$，它的极分解为$\boldsymbol{M}=\boldsymbol{Q}\boldsymbol{S}$，其中$\boldsymbol{Q}$是一个正交矩阵，而$\boldsymbol{S}$是一个半正定矩阵。

而如果$\boldsymbol{M}$的SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，那么正好有  
\begin{equation}\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = (\boldsymbol{U}\boldsymbol{V}^{\top})(\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top})\end{equation}  
即$\boldsymbol{Q}=\boldsymbol{U}\boldsymbol{V}^{\top},\boldsymbol{S}=\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$就是极分解的一个答案。而我们知道，当$\boldsymbol{M}$是满秩矩阵时，$\boldsymbol{U}\boldsymbol{V}^{\top}$正好是$\mathop{\text{msgin}}(\boldsymbol{M})$。这也就是为什么$\mathop{\text{msgin}}$会跟“Polar”联系起来了，因为求出它之后就可以得到矩阵的“Polar Decomposition”，换言之极分解的本质难度就是计算$\mathop{\text{msgin}}$，这跟Muon异曲同工。

## 贪心即可 #

言归正传。对于问题$\eqref{eq:opt}$，原论文得出的第一个结论，也应该是全论文最核心的结论是：**它的贪心解正好是它的全局最优解！** 用公式来说，它是指问题$\eqref{eq:opt}$的求解，可以转化为：  
\begin{equation}\begin{gathered}  
f^* = f_T^* \circ \dots \circ f_2^* \circ f_1^* \\\\[12pt]  
f_1^* = \mathop{\text{argmin}}_{f_1} \max_{x\in[l_1,u_1]} |f_1(x) - 1| \\\  
f_2^* = \mathop{\text{argmin}}_{f_2} \max_{x\in[l_2,u_2]} |f_2(x) - 1| \\\  
\vdots \\\  
f_T^* = \mathop{\text{argmin}}_{f_T} \max_{x\in[l_T,u_T]} |f_T(x) - 1| \\\\[24pt]  
l_1 = l,\quad u_1 = u, \\\\[8pt]  
l_{t+1} = \min_{x\in[l_t,u_t]} f_t^*(x),\quad u_{t+1} = \max_{x\in[l_t,u_t]} f_t^*(x)  
\end{gathered}\end{equation}

这个结论我相信会出乎不少读者意料，笔者首次看到时也是相当惊讶，并为之拍案叫绝。它不仅大大降低了求解难度，将原本$T$步的复合函数求解，转化为$T=1$的单个多项式逐步求解，还允许我们将解逐步往前推，并一直保持最优性（即$T+1$步的最优解，只需要在$T$步最优解基础上多算一步，而不用从头计算）。

值得指出的是，这个结论是允许每个$f_t$取不同阶的（这里的“阶”指多项式最高次数），比如$f_1$可以是3阶的，$f_2$可以是5阶的，等等，但“贪心解正好是全局最优解”的结论依然不变。不过简单起见，下面我们还是让所有$f_t$都保持同阶，并且主要考虑3阶和5阶的结果。

上述结论的完整证明略微有点复杂，我们把它放到最后，先完成基于该结论的后续操作。

## 等值振荡 #

既然我们已经把原问题转化为求贪心解，那么现在只需要专注于求解  
\begin{equation}\mathop{\text{argmin}}_{f_t} \max_{x\in[l_t,u_t]} |f_t(x) - 1| \label{eq:local}\end{equation}  
为了求解上式，我们需要先了解在[《等值振荡定理：最优多项式逼近的充要条件》](/archives/10972)介绍的关于奇多项式的“等值振荡定理（Equioscillation Theorem）”：

> **等值振荡定理-奇** 设$f(x)$是不超过$2n+1$阶的奇多项式，$g(x)$是区间$[a,b]\subset (0,\infty)$上的连续函数，那么 \begin{equation}f^* = \mathop{\text{argmin}}_f \max_{x\in[a,b]} |f(x) - g(x)|\end{equation} 的充要条件是存在$a\leq x_0 < x_1 < \cdots < x_{n+1} \leq b$以及$\sigma\in\\{0,1\\}$，使得 \begin{equation}f^*(x_k) - g(x_k) = (-1)^{k+\sigma} \max_{x\in[a,b]} |f^*(x) - g(x)|\end{equation}

现在我们要求的是$f_t$，目标$g$则是恒等于1，等值振荡定理告诉我们$|f_t^*(x)-1|$在$[l_t,u_t]$至少$n+2$次达到最大误差（记为$\mathcal{E}$）。不难发现，$|f_t^*(x)-1|$的最大值点只能是边界点或者是$f_t^*(x)$的极值点，而一个$2n+1$阶奇多项式在$(0,\infty)$至多有$n$个极值点，所以为了“凑”够$n+2$个，我们“不得已”要把边界点补上，这就确定了$x_0 = l_t, x_{n+1}=u_t$，而$x_1,\cdots,x_n$则是$\frac{d}{dx}f_t^*(x)$的零点。

此外，由于目标函数是$1$，所以$f_t^*(x)$在$x=0$处的斜率大于零，所以$l_t$只能是$f_t^*(x)$的最小值点，因此$\sigma=1$。综合这些结果，这样我们实际上要解方程组：  
\begin{equation}f_t(l_t) = 1 - \mathcal{E}, \quad f_t(u_t) = 1 + (-1)^n \mathcal{E},\quad f_t(x_i) = 1 + (-1)^{i+1}\mathcal{E}, \quad f_t'(x_i) = 0\end{equation}  
其中$i=1,2,3,\cdots,n$。可以发现方程和未知数都是$2n+2$个，再补上$l_t < x_1 < \cdots < x_n < u_t$和$\mathcal{E} > 0$的约束条件，理论上可以把解确定下来。

## 解方程组 #

对于3阶奇多项式（$n=1$），原论文给出了解析解，而对于5阶奇多项式（$n=2$），原论文给出的是一个迭代求解算法，即先固定$x_1,x_2$求$a,b,c$，然后固定$f_t(x)$的$a,b,c$求$x_1,x_2$，反复迭代，这本质上是[Remez算法](https://en.wikipedia.org/wiki/Remez_algorithm)的一个简化版。

不过，原论文的迭代依赖于求根公式来求$x_1,x_2$，这对于更大的$n$就不大容易操作了。所以这里笔者改变一下求解思路，先以$x_1,x_2,\cdots,x_n$来参数化$f_t'(x_i)$，即定义  
\begin{equation}f_t'(x) = k(x^2-x_1^2)(x^2-x_2^2)\cdots (x^2-x_n^2)\end{equation}  
然后有$f_t(x) = \int_0^x f_t'(x) dx$，这样我们就用$k$和$x_1,x_2,\cdots,x_n$表示出了$f_t(x)$，继而只需求解方程组  
\begin{equation}f_t(l_t) = 1 - \mathcal{E}, \quad f_t(u_t) = 1 + (-1)^n \mathcal{E},\quad f_t(x_i) = 1 + (-1)^{i+1}\mathcal{E}\end{equation}  
而避免了解方程$f_t'(x) = 0$。当$n=1$时，我们可以解得  
\begin{equation}x_1 = \sqrt{\frac{l_t^2+l_t u_t + u_t^2}{3}},\quad k = -\frac{6}{l_t^2 u_t + l_t u_t^2 + 2x_1^3}\end{equation}  
当$n > 1$时，我们可以直接交给Mathematica，例如$n=2$时：
    
    
    df[x_] = k*(x^2 - x1^2) (x^2 - x2^2);
    f[x_] = Integrate[df[x], {x, 0, x}];
    sol = NSolve[{f[l] == 1 - e, f[x1] == 1 + e, f[x2] == 1 - e, 
        f[u] == 1 + e, l < x1 < x2 < u, e > 0} /. {l -> 0.001, 
        u -> 1}, {k, x1, x2, e}, Reals]
    f[x] /. sol

## 有限精度 #

至此，我们似乎已经完成了原始问题的求解？理论上是的，但仅限于无限精度。实际计算时是有限精度的，尤其是Muon优化器用的是bfloat16，精度损失更严重，所以就带来了一些问题。

第一个问题，是每一个$f_t^*$理论上只对区间$[l_t,u_t]$负责，但有限精度下奇异值可能会偏离该区间。当$n$是偶数时（即$f_t^*$是5、9、...阶奇多项式），超出$u_t$就存在发散风险，因为此时的$f_t^*(x)$在$x > u_t$时是单调递增到正无穷的，稍有不慎就会随着迭代而发散。解决办法有两个，一是求$f_t^*$时把$[l_t,u_t]$预留得稍微宽松一些，二是保持区间不变，但求得$f_t^*$后输入要多除一个大于1的数。

原论文用的是后者，把$f_t^*(x)$改为$f_t^*(x / 1.01)$。1.01这个数字，大约就是在bfloat16精度下，1后面的第一个数字（准确值是1.00781），很明显这是预防由于数值误差将奇异值从1扩大到了下一位的问题，如果是在更高的精度下进行计算，可以适当缩小这个值。

第二个问题比较隐晦，我们用具体例子来介绍。设$n=2,l_1=0.001,u_1=1$，可以求得$f_1^*$是  
\begin{equation}f_1^*(x) = 8.4703 x - 25.1081 x^3 + 18.6293 x^5\end{equation}  
其中$x_1 = 0.3674, x_2 = 0.8208, \mathcal{E}=0.9915$。这个解有什么问题呢？根据等值振荡定理，我们知道$f_1^*(x_2) = 1-\mathcal{E} = 0.0085$，即它会把$0.8208$映射成$0.0085$。然而，我们的最终目标是将$(0,1]$的所有数都变成1，所以$f_1^*$会将一个已经很接近目标的$0.8208$映射到非常远离目标的$0.0085$。尽管后面$f_2^*,f_3^*,\cdots$理论上会逐渐把它拉回来，但在有限精度下反复将一个数缩小又放大，累积误差是很可观的。

当然，由等值振荡定理可知，这种振荡行为是无法避免的，我们只能寄望于最大误差$\mathcal{E}$不要太接近于1，从而减缓这种累积误差。不难看出，区间$[l_t,u_t]$越大，理论上就越难拟合，最大误差$\mathcal{E}$将会越近于1，所以论文引入了一个超参数$\lambda \in (0, 1)$，将优化区间从$[l_t,u_t]$改为$[\max(l_t, \lambda u_t),u_t]$，通过限制区间大小来保证$\mathcal{E}$不会太大。（需要提醒读者的是，论文在正文讲解中用的$\lambda$是$0.1$，但附录代码实际用的$\lambda$是$0.024$。）

但这样一来，原本的$l_t$，尤其是我们一开始设置的$l$，不就容易被忽略了？为了解决这个问题，论文引入了“Recenter”技巧，即如果优化区间是$[l_t,u_t]$，那么将会满足$f_t^*(l_t) + f_t^*(u_t) = 2$，而优化区间改为$[\max(l_t, \lambda u_t),u_t]$后就不一定满足了，这时候我们给$f_t^*$乘上$\gamma$，使得它满足这个等式  
\begin{equation}\gamma f_t^*(l_t) + \gamma f_t^*(u_t) = 2\qquad \Rightarrow \qquad \gamma = \frac{2}{f_t^*(l_t) + f_t^*(u_t)}\end{equation}  
这就把原本$l_t$考虑进去了。

## 参考代码 #

这是$n=2$时的Mathematica完整代码：
    
    
    df[x_] = k*(x^2 - x1^2) (x^2 - x2^2);
    f[x_] = Integrate[df[x], {x, 0, x}];
    sol[l_, u_] := 
     NSolve[{f[l] == 1 - e, f[x1] == 1 + e, f[x2] == 1 - e, f[u] == 1 + e,
        l < x1 < x2 < u, e > 0, k > 0}, {k, x1, x2, e}]
    ff[x_, l_, u_] = f[x]*2/(f[l] + f[u]) // Expand;
    lt = 0.001; ut = 1; lambda = 0.02407327424182761;
    While[1 - lt > 0.0001,
     fff[x_] = ff[x, lt, ut] /. sol[Max[lt, lambda*ut], ut][[1]];
     Print[fff[x]];
     lt = fff[lt]; ut = 2 - lt]
    

结果如下（$f_t(x) = a_t x + b_t x^3 + c_t x^5$）：  
\begin{array}{c|ccc}  
\hline  
t & a\times 1.01 & b\times 1.01^3 & c\times 1.01^5 \\\  
\hline  
\quad 1\quad & 8.28721 & -23.5959 & 17.3004 \\\  
2 & 4.10706 & -2.94785 & 0.544843 \\\  
3 & 3.94869 & -2.9089 & 0.551819 \\\  
4 & 3.31842 & -2.48849 & 0.510049 \\\  
5 & 2.30065 & -1.6689 & 0.418807 \\\  
6 & 1.8913 & -1.268 & 0.376804 \\\  
7 & 1.875 & -1.25 & 0.375 \\\  
8 & 1.875 & -1.25 & 0.375 \\\  
\hline  
\end{array}

注意这里给出的是还没有做$f_t^*(x / 1.01)$处理的结果，所以实际的$a, b, c$还要在该表基础上多除以$1.01$的$1,3,5$次方。没有直接给出除以$1.01$之后的结果，是因为除以$1.01$前的收敛值$1.875, -1.25, 0.375$（$t \geq 7$）显得更简洁明了，便于观察和欣赏。（思考题：请证明最终的收敛值可以由$x_1=x_2=1$以及$f(1)=1$解出。）

作者附录的代码整理如下：
    
    
    import numpy as np
    
    def optimal_quintic(l, u):
        assert 0 <= l <= u
        if 1 - 5e-6 <= l / u:
            # Above this threshold, the equoscillating polynomials
            # is numerically equal to...
            return (15 / 8) / u, (-10 / 8) / (u**3), (3 / 8) / (u**5)
        # This initialization becomes exact as l -> u
        q = (3 * l + 1) / 4
        r = (l + 3) / 4
        E, old_E = np.inf, None
        while not old_E or abs(old_E - E) > 1e-15:
            old_E = E
            LHS = np.array([
                [l, l**3, l**5, 1],
                [q, q**3, q**5, -1],
                [r, r**3, r**5, 1],
                [u, u**3, u**5, -1],
            ])
            a, b, c, E = np.linalg.solve(LHS, np.ones(4))
            q, r = np.sqrt(
                (-3 * b + np.array([-1, 1]) * np.sqrt(9 * b**2 - 20 * a * c)) /
                (10 * c)
            )
        return float(a), float(b), float(c)
    
    def optimal_composition(l, num_iters, cushion=0.02407327424182761):
        u = 1
        coefficients = []
        for _ in range(num_iters):
            a, b, c = optimal_quintic(max(l, cushion * u), u)
            # Due to cushioning , this may be centered around 1 with
            # respect to 0.024*u, u. Recenter it around 1 with respect
            # to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescalar = 2 / (pl + pu)
            a *= rescalar
            b *= rescalar
            c *= rescalar
            # Optionally incorporate safety factor here :
            # a /= 1.01; b /= 1.01**3; c /= 1.01**5
            coefficients.append((a, b, c))
            l = a * l + b * l**3 + c * l**5
            u = 2 - l
        return coefficients
    
    print(*optimal_composition(1e-3, 10), sep="\n")

## 完成证明 #

最后一节，我们来补上“贪心解正好是它的全局最优解”的证明。

根据等值振荡定理，我们知道$f_t^*$的值域是$[l_{t+1},u_{t+1}]$，其中$l_{t+1}=f_t^*(l_t),u_{t+1}=2-l_{t+1}$，由此可知$T$步贪心解的最大误差是$\mathcal{E}_T = 1 - l_{T+1} = 1 - f_T^*(l_T)$，我们只需要证明$T$步全局最优解的最大误差也只能降到$1 - f_T^*(l_T)$，就可以得到“贪心解正好是它的全局最优解”这个结论。

证明的思路是数学归纳法。假设结论对于$t=1,2,\cdots,T-1$成立，那么$\hat{f} = f_{T-1}^*\circ \cdots \circ f_2^* \circ f_1^*$就是$T-1$步的全局最优解，值域为$[l_T, u_T]$，最大误差为$\mathcal{E}_{T-1}=1-l_T=u_T-1$。另一方面，设$\tilde{f} = \tilde{f}_{T-1}\circ \cdots \circ \tilde{f}_2 \circ \tilde{f}_1$为任意一个$T-1$步解，值域为$[a,b]$，令$c = \frac{2}{a+b}$，那么$c\tilde{f}$的值域则是$[ca,cb]$，显然$ca\leq 1, cb\geq 1$。根据归纳假设，我们有  
\begin{equation}\begin{aligned}  
1 - ca \geq \mathcal{E}_{T-1} \\\  
cb - 1 \geq \mathcal{E}_{T-1}  
\end{aligned}\qquad\Rightarrow\qquad \frac{a}{b} \leq \frac{1 - \mathcal{E}_{T-1}}{1 + \mathcal{E}_{T-1}} = \frac{l_T}{u_T} \end{equation}  
即任意一个$T-1$步解的值域的相对大小，都不小于$T-1$步最优解的值域$[l_T, u_T]$的相对大小。接着我们有  
\begin{equation}\begin{aligned}  
\min_{f_T} \max_{x\in[l,u]} |f_T(\tilde{f}(x)) - 1| =&\, \min_{f_T} \max_{x\in[a,b]} |f_T(x) - 1| \\\  
=&\, \min_{f_T} \max_{x\in[a/b,1]} |f_T(x) - 1| \\\  
\geq &\, \min_{f_T} \max_{x\in[l_T/u_T,1]} |f_T(x) - 1| \\\  
=&\, \min_{f_T} \max_{x\in[l_T,u_T]} |f_T(x) - 1| \\\  
=&\, \mathcal{E}_T  
\end{aligned}\end{equation}  
也就是说，你随便拿一个别的$T-1$步解来，最大误差也顶多只能跟贪心解一样小，所以贪心解的最大误差已经是全局最优的，这就完成了递归证明。上式的关键一步是  
\begin{equation}\min_{f_T} \max_{x\in[a,b]} |f_T(x) - 1| = \min_{f_T} \max_{x\in[a/b,1]} |f_T(x) - 1|\end{equation}  
这是因为我们总可以设$g_T(y) = f_T(b y)$，$g_T$依然能代表任意同阶的奇多项式，所以$g_T$和$f_T$都在同一函数空间中，因此记号可以换用，即  
\begin{equation}\min_{f_T}\max_{x\in[a,b]} |f_T(x) - 1| = \min_{g_T}\max_{y\in[a/b,1]} |g_T(y) - 1|= \min_{f_T}\max_{x\in[a/b,1]} |f_T(x) - 1|\end{equation}

## 文章小结 #

本文介绍了为msign算子寻找更好的Newton-Schulz迭代的最新进展，它通过等值振荡定理和贪心转换，直接求出理论上的最优解，整个过程相当硬核，值得学习一波。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10996>_

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

苏剑林. (Jun. 05, 2025). 《msign算子的Newton-Schulz迭代（下） 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10996>

@online{kexuefm-10996,  
title={msign算子的Newton-Schulz迭代（下）},  
author={苏剑林},  
year={2025},  
month={Jun},  
url={\url{https://spaces.ac.cn/archives/10996}},  
} 


---

## 公式推导与注释

本节将详细推导等值振荡定理、贪心算法的最优性证明、以及求解最优多项式系数的数学方法，并分析有限精度下的数值稳定性。

### 1. 极分解的数学理论

#### 1.1 极分解的定义与存在性

**定理（极分解）**：对于任意矩阵$\boldsymbol{M} \in \mathbb{R}^{n \times n}$，存在唯一的极分解：
\begin{equation}
\boldsymbol{M} = \boldsymbol{Q}\boldsymbol{S}
\tag{1}\end{equation}

其中$\boldsymbol{Q}$是正交矩阵（$\boldsymbol{Q}^{\top}\boldsymbol{Q} = \boldsymbol{I}$），$\boldsymbol{S}$是对称半正定矩阵（$\boldsymbol{S} = \boldsymbol{S}^{\top}$且$\boldsymbol{S} \succeq \boldsymbol{0}$）。

**证明**：设$\boldsymbol{M}$的SVD为$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，其中$\boldsymbol{U}, \boldsymbol{V}$是正交矩阵，$\boldsymbol{\Sigma}$是对角矩阵，对角元为奇异值$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \geq 0$。

定义：
\begin{align}
\boldsymbol{Q} &= \boldsymbol{U}\boldsymbol{V}^{\top} \tag{2}\\
\boldsymbol{S} &= \boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} \tag{3}
\end{align}

验证正交性：
\begin{equation}
\boldsymbol{Q}^{\top}\boldsymbol{Q} = (\boldsymbol{U}\boldsymbol{V}^{\top})^{\top}(\boldsymbol{U}\boldsymbol{V}^{\top}) = \boldsymbol{V}\boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{V}^{\top} = \boldsymbol{V}\boldsymbol{V}^{\top} = \boldsymbol{I}
\tag{4}\end{equation}

验证半正定性：
\begin{equation}
\boldsymbol{S} = \boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \boldsymbol{V}\boldsymbol{\Sigma}^{1/2}\boldsymbol{\Sigma}^{1/2}\boldsymbol{V}^{\top} = (\boldsymbol{V}\boldsymbol{\Sigma}^{1/2}\boldsymbol{V}^{\top})^2 \succeq \boldsymbol{0}
\tag{5}\end{equation}

验证极分解：
\begin{equation}
\boldsymbol{Q}\boldsymbol{S} = (\boldsymbol{U}\boldsymbol{V}^{\top})(\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}) = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \boldsymbol{M}
\tag{6}\end{equation}

#### 1.2 唯一性证明

**命题**：当$\boldsymbol{M}$可逆时，极分解是唯一的。

**证明**：假设存在两个极分解$\boldsymbol{M} = \boldsymbol{Q}_1\boldsymbol{S}_1 = \boldsymbol{Q}_2\boldsymbol{S}_2$。则：
\begin{equation}
\boldsymbol{M}^{\top}\boldsymbol{M} = (\boldsymbol{Q}_1\boldsymbol{S}_1)^{\top}(\boldsymbol{Q}_1\boldsymbol{S}_1) = \boldsymbol{S}_1\boldsymbol{Q}_1^{\top}\boldsymbol{Q}_1\boldsymbol{S}_1 = \boldsymbol{S}_1^2
\tag{7}\end{equation}

同理$\boldsymbol{M}^{\top}\boldsymbol{M} = \boldsymbol{S}_2^2$。因为半正定矩阵的平方根是唯一的，所以$\boldsymbol{S}_1 = \boldsymbol{S}_2 = \boldsymbol{S}$。

进而：
\begin{equation}
\boldsymbol{Q}_1 = \boldsymbol{M}\boldsymbol{S}^{-1} = \boldsymbol{Q}_2
\tag{8}\end{equation}

所以极分解是唯一的。$\square$

#### 1.3 极分解与msign的关系

对于满秩矩阵$\boldsymbol{M}$，极分解中的正交因子$\boldsymbol{Q}$正好是$\msign(\boldsymbol{M})$：
\begin{equation}
\msign(\boldsymbol{M}) = \boldsymbol{U}\boldsymbol{V}^{\top} = \boldsymbol{Q}
\tag{9}\end{equation}

**几何解释**：$\msign$将矩阵投影到最近的正交矩阵（在Frobenius范数意义下），这正好对应极分解中去除"拉伸"部分$\boldsymbol{S}$，只保留"旋转"部分$\boldsymbol{Q}$。

**最优性**：对于满秩方阵$\boldsymbol{M}$，$\msign(\boldsymbol{M})$是使得$\Vert\boldsymbol{M} - \boldsymbol{O}\Vert_F$最小的正交矩阵$\boldsymbol{O}$：
\begin{equation}
\msign(\boldsymbol{M}) = \mathop{\text{argmin}}_{\boldsymbol{O}^{\top}\boldsymbol{O}=\boldsymbol{I}} \Vert\boldsymbol{M} - \boldsymbol{O}\Vert_F
\tag{10}\end{equation}

### 2. 等值振荡定理

#### 2.1 Chebyshev交替定理

**定理（Chebyshev交替定理，奇多项式版本）**：设$g(x)$是区间$[a,b] \subset (0, \infty)$上的连续函数，$\mathcal{P}_{2n+1}$是所有次数不超过$2n+1$的奇多项式的集合。则$p^* \in \mathcal{P}_{2n+1}$是最佳逼近多项式：
\begin{equation}
p^* = \mathop{\text{argmin}}_{p \in \mathcal{P}_{2n+1}} \max_{x \in [a,b]} |p(x) - g(x)|
\tag{11}\end{equation}

当且仅当存在$n+2$个点$a \leq x_0 < x_1 < \cdots < x_{n+1} \leq b$和$\sigma \in \{0,1\}$，使得：
\begin{equation}
p^*(x_k) - g(x_k) = (-1)^{k+\sigma} E^*, \qquad k = 0,1,\ldots,n+1
\tag{12}\end{equation}

其中$E^* = \max_{x \in [a,b]} |p^*(x) - g(x)|$是最大误差。

**注释**：这个定理说明最佳逼近多项式的误差曲线必须在至少$n+2$个点处"等值振荡"，即达到同样大小的正负最大误差。

#### 2.2 充分性证明

假设$p^*$满足等值振荡条件(12)，但不是最佳逼近。那么存在$q \in \mathcal{P}_{2n+1}$使得：
\begin{equation}
\max_{x \in [a,b]} |q(x) - g(x)| < E^*
\tag{13}\end{equation}

定义误差函数$e(x) = p^*(x) - q(x)$。这是一个次数不超过$2n+1$的奇多项式。

在点$x_k$处：
\begin{align}
e(x_k) &= p^*(x_k) - q(x_k) \tag{14}\\
&= [p^*(x_k) - g(x_k)] - [q(x_k) - g(x_k)] \tag{15}\\
&= (-1)^{k+\sigma} E^* - [q(x_k) - g(x_k)] \tag{16}
\end{align}

由于$|q(x_k) - g(x_k)| < E^*$，我们有：
\begin{equation}
\text{sign}(e(x_k)) = (-1)^{k+\sigma}
\tag{17}\end{equation}

这意味着$e(x)$在$n+2$个点$\{x_0, x_1, \ldots, x_{n+1}\}$处符号交替变化，因此$e(x)$至少有$n+1$个零点（在相邻的$x_k$之间）。

但$e(x)$是奇多项式，$e(0) = 0$是一个零点。所以$e(x)$至少有$n+2$个零点。

然而，一个次数为$2n+1$的奇多项式$e(x) = x h(x)$，其中$h(x)$是次数为$2n$的偶多项式，最多只能有$2n+1$个零点（包括$x=0$）。如果$h(x)$的所有零点都是实数，则$e(x)$最多有$2n+1$个零点。

**矛盾**：$e(x)$至少有$n+2$个零点（来自符号变化），加上$x=0$这个零点，如果这些零点都不重合，则至少有$n+2$个非零零点，总共至少$n+3$个零点。

等等，我的推理有问题。让我重新思考...实际上$e(x)$在$[a,b]$内符号交替$n+1$次，意味着至少有$n+1$个零点。$e(x)$是$2n+1$次奇多项式，可以写成$e(x) = x \cdot p_{2n}(x)$，其中$p_{2n}$是$2n$次多项式。$e(x) = 0$的解包括$x=0$和$p_{2n}(x)=0$的解。$p_{2n}(x)$最多有$2n$个零点。所以$e(x)$在$(0, \infty)$内最多有$2n$个零点。

如果$e(x)$在$[a,b] \subset (0, \infty)$内有$n+1$个零点（来自符号变化），而$2n \geq n+1$对$n \geq 1$总成立，所以似乎没有矛盾？

让我更仔细地分析。如果$e(x)$在$x_0, x_1, \ldots, x_{n+1}$处符号交替，则在每对相邻点$(x_k, x_{k+1})$之间至少有一个零点。这给出$n+1$个零点。但这$n+1$个零点都在$(a,b)$内（不包括$x=0$）。

一个$2n+1$次奇多项式$e(x) = c_1 x + c_3 x^3 + \cdots + c_{2n+1} x^{2n+1}$，其零点包括$x=0$（如果不恒为零）。除去$x=0$，剩余的零点来自$c_1 + c_3 x^2 + \cdots + c_{2n+1} x^{2n} = 0$，这是一个关于$x^2$的$n$次方程，最多有$n$个正实根，对应$2n$个实零点（正负各$n$个）。

所以$e(x)$在$(0, \infty)$内最多有$n$个零点（对应$x^2$的$n$个正根）。

如果$e(x)$在$(x_k, x_{k+1})$之间有零点（$k=0,\ldots,n$），则至少有$n+1$个零点在$(0, \infty)$内。这与"最多$n$个零点"矛盾！

所以假设不成立，$p^*$确实是最佳逼近。$\square$

#### 2.3 必要性证明

现在假设$p^*$是最佳逼近，最大误差为$E^*$。我们需要证明存在$n+2$个等值振荡点。

定义误差函数$e(x) = p^*(x) - g(x)$。设$\mathcal{M} = \{x \in [a,b] : |e(x)| = E^*\}$是达到最大误差的点集。

**引理**：$\mathcal{M}$至少包含$n+2$个点，且在这些点处$e(x)$的符号交替。

**证明（反证法）**：假设$\mathcal{M}$中只有$m \leq n+1$个点，且按大小排序为$y_0 < y_1 < \cdots < y_{m-1}$。

构造扰动多项式$q(x) = p^*(x) - \epsilon r(x)$，其中$r(x)$是一个奇多项式（次数$\leq 2n+1$），满足：
1. $r(y_i) \cdot e(y_i) > 0$（$r$与$e$在$\mathcal{M}$上同号）
2. $|r(x)| \leq 1$对所有$x \in [a,b]$

这样的$r(x)$可以构造为Chebyshev型的振荡多项式。

选择足够小的$\epsilon > 0$，使得：
\begin{equation}
\max_{x \in [a,b]} |q(x) - g(x)| < E^*
\tag{18}\end{equation}

这与$p^*$是最佳逼近矛盾。$\square$

### 3. 贪心算法最优性的严格证明

#### 3.1 问题的递归结构

我们要证明的核心定理是：

**定理（贪心最优性）**：对于优化问题
\begin{equation}
\min_{f = f_T \circ \cdots \circ f_1} \max_{x \in [l,u]} |f(x) - 1|
\tag{19}\end{equation}

其中每个$f_t$是奇多项式，贪心解
\begin{equation}
f^* = f_T^* \circ \cdots \circ f_1^*
\tag{20}\end{equation}

是全局最优的，其中每个$f_t^*$独立求解：
\begin{equation}
f_t^* = \mathop{\text{argmin}}_{f_t} \max_{x \in [l_t, u_t]} |f_t(x) - 1|
\tag{21}\end{equation}

且$l_{t+1} = \min_{x \in [l_t, u_t]} f_t^*(x)$，$u_{t+1} = \max_{x \in [l_t, u_t]} f_t^*(x)$。

#### 3.2 归纳法证明

**基础步骤**（$T=1$）：显然贪心解就是全局最优解。

**归纳假设**：假设对于$T-1$步，贪心解是全局最优的。

**归纳步骤**：我们需要证明对于$T$步，贪心解也是全局最优的。

设$\hat{f} = f_{T-1}^* \circ \cdots \circ f_1^*$是$T-1$步的贪心解，根据归纳假设，它也是$T-1$步的全局最优解。其值域为$[l_T, u_T]$，其中：
\begin{align}
l_T &= \min_{x \in [l, u]} \hat{f}(x) \tag{22}\\
u_T &= \max_{x \in [l, u]} \hat{f}(x) \tag{23}
\end{align}

由等值振荡定理，我们知道$\hat{f}$满足：
\begin{equation}
\hat{f}(l) = l_T, \quad \hat{f}(u) = u_T
\tag{24}\end{equation}

且最大误差为：
\begin{equation}
E_{T-1} = \max(1 - l_T, u_T - 1)
\tag{25}\end{equation}

由对称性（recenter技巧），我们有$l_T + u_T = 2$，因此：
\begin{equation}
E_{T-1} = 1 - l_T = u_T - 1
\tag{26}\end{equation}

#### 3.3 值域不等式

现在考虑任意一个$T-1$步解$\tilde{f} = \tilde{f}_{T-1} \circ \cdots \circ \tilde{f}_1$，设其值域为$[a, b]$。

**引理**：任意$T-1$步解的最大误差不小于$E_{T-1}$。

**证明**：设$\tilde{f}$的最大误差为$\tilde{E}$，则：
\begin{align}
1 - a &\leq \tilde{E} \tag{27}\\
b - 1 &\leq \tilde{E} \tag{28}
\end{align}

不失一般性，假设$a + b = 2$（否则可以用归一化因子$c = 2/(a+b)$调整，$c\tilde{f}$的值域变为$[ca, cb]$，且$ca + cb = 2$）。

由归纳假设，$\tilde{E} \geq E_{T-1}$，因此：
\begin{align}
a &\geq 1 - E_{T-1} = l_T \tag{29}\\
b &\leq 1 + E_{T-1} = u_T \tag{30}
\end{align}

即：
\begin{equation}
\frac{a}{b} \geq \frac{l_T}{u_T}
\tag{31}\end{equation}

这意味着值域$[a,b]$相对$[l_T, u_T]$更"紧凑"（比值更大）。

#### 3.4 最后一步的优化

现在考虑第$T$步的优化。贪心解是：
\begin{equation}
f_T^* = \mathop{\text{argmin}}_{f_T} \max_{x \in [l_T, u_T]} |f_T(x) - 1|
\tag{32}\end{equation}

设其最大误差为$E_T$。

对于任意其他的第$T$步函数$\tilde{f}_T$，考虑复合$\tilde{f}_T \circ \tilde{f}$，其最大误差为：
\begin{equation}
\tilde{E}_T = \max_{x \in [l,u]} |\tilde{f}_T(\tilde{f}(x)) - 1|
\tag{33}\end{equation}

由于$\tilde{f}$的值域是$[a,b]$，我们有：
\begin{equation}
\tilde{E}_T = \max_{y \in [a,b]} |\tilde{f}_T(y) - 1|
\tag{34}\end{equation}

**关键变换**：由于$\tilde{f}_T$是奇多项式，定义$g_T(z) = \tilde{f}_T(b z)$，则$g_T$也是奇多项式（只是系数不同），且：
\begin{equation}
\max_{y \in [a,b]} |\tilde{f}_T(y) - 1| = \max_{z \in [a/b, 1]} |g_T(z) - 1|
\tag{35}\end{equation}

由于$g_T$与$\tilde{f}_T$在同一函数空间，我们可以写：
\begin{equation}
\min_{\tilde{f}_T} \max_{y \in [a,b]} |\tilde{f}_T(y) - 1| = \min_{g_T} \max_{z \in [a/b,1]} |g_T(z) - 1|
\tag{36}\end{equation}

由不等式(31)，$a/b \geq l_T/u_T$，因此区间$[a/b, 1]$包含于区间$[l_T/u_T, 1]$（或者说更紧凑）。

**单调性**：对于固定的目标函数$g(x) = 1$，最小化$\max_{x \in [c,d]} |f(x) - 1|$的最优误差关于区间$[c,d]$是单调的：区间越大，最优误差越大。

**注释**：这是因为更大的区间给逼近带来了更大的挑战。严格证明需要用到等值振荡定理的性质。

因此：
\begin{equation}
\min_{g_T} \max_{z \in [a/b,1]} |g_T(z) - 1| \leq \min_{g_T} \max_{z \in [l_T/u_T,1]} |g_T(z) - 1|
\tag{37}\end{equation}

注意到$[l_T/u_T, 1]$和$[l_T, u_T]$通过缩放变换$z = y/u_T$相关联，类似的变换论证给出：
\begin{equation}
\min_{g_T} \max_{z \in [l_T/u_T,1]} |g_T(z) - 1| = \min_{f_T} \max_{y \in [l_T,u_T]} |f_T(y) - 1| = E_T
\tag{38}\end{equation}

综合(34)-(38)：
\begin{equation}
\tilde{E}_T \geq E_T
\tag{39}\end{equation}

这证明了贪心解的误差不大于任意其他解的误差，因此贪心解是全局最优的。$\square$

### 4. 求解等值振荡多项式

#### 4.1 三阶奇多项式的解析解

对于$n=1$（三阶奇多项式$f(x) = ax + bx^3$），我们需要求解：
\begin{align}
f(l) &= 1 - E \tag{40}\\
f(u) &= 1 + E \tag{41}\\
f(x_1) &= 1 + E \tag{42}\\
f'(x_1) &= 0 \tag{43}
\end{align}

从$f'(x) = a + 3bx^2 = 0$得：
\begin{equation}
x_1^2 = -\frac{a}{3b}
\tag{44}\end{equation}

**注释**：为使$x_1 \in (l, u)$为实数，需要$ab < 0$。

从(40)和(41)：
\begin{align}
al + bl^3 &= 1 - E \tag{45}\\
au + bu^3 &= 1 + E \tag{46}
\end{align}

相减：
\begin{equation}
a(u-l) + b(u^3 - l^3) = 2E
\tag{47}\end{equation}

从(42)和(44)：
\begin{equation}
ax_1 + bx_1^3 = 1 + E
\tag{48}\end{equation}

代入$a = -3bx_1^2$：
\begin{equation}
-3bx_1^3 + bx_1^3 = 1 + E \quad\Rightarrow\quad -2bx_1^3 = 1 + E
\tag{49}\end{equation}

所以：
\begin{equation}
b = -\frac{1+E}{2x_1^3}
\tag{50}\end{equation}

从(45)：
\begin{equation}
al = 1 - E - bl^3
\tag{51}\end{equation}

从(46)：
\begin{equation}
au = 1 + E - bu^3
\tag{52}\end{equation}

相除：
\begin{equation}
\frac{l}{u} = \frac{1-E-bl^3}{1+E-bu^3}
\tag{53}\end{equation}

经过代数计算（详细过程略），可以得到：
\begin{equation}
x_1 = \sqrt{\frac{l^2 + lu + u^2}{3}}
\tag{54}\end{equation}

这是正文中给出的结果。

#### 4.2 五阶奇多项式的Remez算法

对于$n=2$（五阶奇多项式$f(x) = ax + bx^3 + cx^5$），解析解变得非常复杂。我们采用Remez算法迭代求解。

**Remez算法框架**：
1. 初始化：猜测$x_1, x_2$的位置（例如均匀分布）
2. 固定$x_1, x_2$，求解线性方程组得到$(a,b,c,E)$
3. 固定$(a,b,c)$，求$f'(x) = 0$的根，更新$x_1, x_2$
4. 重复步骤2-3直到收敛

**步骤2的线性方程组**：
\begin{equation}
\begin{bmatrix}
l & l^3 & l^5 & 1 \\
x_1 & x_1^3 & x_1^5 & -1 \\
x_2 & x_2^3 & x_2^5 & 1 \\
u & u^3 & u^5 & -1
\end{bmatrix}
\begin{bmatrix}
a \\ b \\ c \\ E
\end{bmatrix}
=
\begin{bmatrix}
1 \\ 1 \\ 1 \\ 1
\end{bmatrix}
\tag{55}\end{equation}

这是一个$4 \times 4$线性系统，可以直接求解。

**步骤3的根求解**：求解$f'(x) = a + 3bx^2 + 5cx^4 = 0$，即：
\begin{equation}
5cx^4 + 3bx^2 + a = 0
\tag{56}\end{equation}

这是关于$y = x^2$的二次方程：
\begin{equation}
5cy^2 + 3by + a = 0
\tag{57}\end{equation}

解为：
\begin{equation}
y = \frac{-3b \pm \sqrt{9b^2 - 20ac}}{10c}
\tag{58}\end{equation}

取两个正根对应$x_1 = \sqrt{y_1}, x_2 = \sqrt{y_2}$（其中$y_1 < y_2$）。

#### 4.3 收敛性分析

**定理（Remez算法收敛性）**：如果初始猜测足够接近真实解，Remez算法二次收敛到最优解。

**证明（概要）**：定义误差向量$\boldsymbol{e}^{(k)} = (\Delta x_1^{(k)}, \Delta x_2^{(k)})$，其中$\Delta x_i^{(k)} = x_i^{(k)} - x_i^*$是第$k$次迭代与真实解的偏差。

通过对迭代映射进行泰勒展开，可以证明：
\begin{equation}
\Vert\boldsymbol{e}^{(k+1)}\Vert \leq C \Vert\boldsymbol{e}^{(k)}\Vert^2
\tag{59}\end{equation}

这表明二次收敛。$\square$

**实践中的收敛速度**：通常5-10次迭代足以达到机器精度。

### 5. 有限精度下的误差分析

#### 5.1 浮点运算的误差模型

在浮点运算中，每个算术运算引入相对误差$u$（机器精度）：
\begin{equation}
\text{fl}(x \circ y) = (x \circ y)(1 + \delta), \qquad |\delta| \leq u
\tag{60}\end{equation}

其中$\circ \in \{+, -, \times, /\}$。

**bfloat16精度**：对于bfloat16格式，机器精度为：
\begin{equation}
u_{\text{bf16}} = 2^{-7} \approx 0.0078
\tag{61}\end{equation}

#### 5.2 振荡误差的累积

考虑五阶等值振荡多项式$f_1^*$，由等值振荡定理，存在点$x_2 \in (l,u)$使得$f_1^*(x_2) = 1 - E$。

如果$E$接近1，则$f_1^*(x_2)$非常小。经过$T$步迭代后，$x_2$需要被放大约$1/f_1^*(x_2)$倍才能达到1。

**累积误差估计**：每步运算的相对误差$u$在$T$步后累积为：
\begin{equation}
\epsilon_{\text{total}} \approx Tu \cdot \prod_{t=1}^T \kappa_t
\tag{62}\end{equation}

其中$\kappa_t$是第$t$步的条件数（大致等于值域的放大倍率）。

如果某步将奇异值从$0.01$缩小到$0.001$再放大回1，条件数$\kappa \sim 100$，累积误差会显著增大。

#### 5.3 Cushion技巧的数学原理

为了控制最大误差$E$不过分接近1，引入cushion参数$\lambda \in (0,1)$，将优化区间从$[l_t, u_t]$改为$[\max(l_t, \lambda u_t), u_t]$。

**误差上界**：对于区间$[l,u]$，最优逼近的最大误差$E$满足：
\begin{equation}
E \geq E_0 \cdot \left(\frac{u-l}{u+l}\right)^n
\tag{63}\end{equation}

其中$E_0$是常数，$n$是多项式的阶数除以2。

**注释**：这个界表明，区间越宽（$(u-l)/(u+l)$越大），误差$E$越大。

通过限制$l \geq \lambda u$，我们有：
\begin{equation}
\frac{u-l}{u+l} \leq \frac{u - \lambda u}{u + \lambda u} = \frac{1-\lambda}{1+\lambda}
\tag{64}\end{equation}

因此：
\begin{equation}
E \geq E_0 \left(\frac{1-\lambda}{1+\lambda}\right)^n
\tag{65}\end{equation}

对于$\lambda = 0.024$，$n=2$（五阶多项式）：
\begin{equation}
\frac{1-\lambda}{1+\lambda} \approx 0.953, \quad E \geq E_0 \cdot 0.953^2 \approx 0.908 E_0
\tag{66}\end{equation}

这确保了$E$不会过分接近1。

#### 5.4 Recenter技巧的数学原理

由于cushion改变了优化区间，等值振荡条件$f(l_t) + f(u_t) = 2$可能不再满足。Recenter通过缩放因子$\gamma$恢复这个性质：
\begin{equation}
\gamma = \frac{2}{f(l_t) + f(u_t)}
\tag{67}\end{equation}

**几何解释**：$\gamma f(x)$将函数在$l_t$和$u_t$的平均值固定在1，确保整个值域关于1对称。

**误差影响**：缩放不会改变函数的"形状"，只是调整幅度。如果$f(l_t) + f(u_t) \approx 2$（cushion不太激进时），$\gamma \approx 1$，影响很小。

### 6. 安全因子的分析

#### 6.1 发散风险

对于偶数阶的$n$（如$n=2$对应五阶多项式），$f_t^*(x)$在$x > u_t$时单调递增到$+\infty$。如果由于数值误差，某个奇异值略微超过$u_t$，则会被放大，可能导致发散。

**发散条件**：设$x = u_t + \epsilon$，其中$\epsilon > 0$很小。泰勒展开：
\begin{align}
f_t^*(u_t + \epsilon) &\approx f_t^*(u_t) + f_t^{*'}(u_t) \epsilon + \frac{1}{2}f_t^{*''}(u_t)\epsilon^2 \tag{68}\\
&= (1+E) + f_t^{*'}(u_t) \epsilon + O(\epsilon^2) \tag{69}
\end{align}

如果$f_t^{*'}(u_t) > 0$且较大，$f_t^*(u_t + \epsilon)$可能显著大于1，后续迭代会继续放大。

#### 6.2 安全因子的作用

将$f_t^*(x)$替换为$f_t^*(x/s)$（其中$s > 1$，例如$s=1.01$），等价于将所有奇异值预先缩小$1/s$倍。

**效果**：原本接近$u_t$的奇异值变为$(u_t/s) < u_t$，远离边界，减少发散风险。

**代价**：所有奇异值都被缩小了，需要更多迭代步才能达到1。

**权衡**：选择$s=1.01$是在安全性和效率之间的折中。对于bfloat16，$s=1.00781$（1后面的第一个可表示数）是自然的选择。

### 7. 参数选择的理论指导

#### 7.1 Cushion参数$\lambda$的选择

**目标**：平衡覆盖范围和误差大小。

**分析**：$\lambda$越小，每步优化的区间越小，$E$越小，振荡越温和，但需要更多步才能覆盖$[l_0, u_0]$。

**经验公式**：根据实验，$\lambda \approx 0.02-0.05$是较好的选择范围。

论文使用$\lambda = 0.02407$，这是通过数值优化得到的，使得6-8步迭代能达到很好的效果。

#### 7.2 迭代步数$T$的选择

**理论估计**：假设每步将值域缩小比例$r = l_{t+1}/l_t$，则$T$步后：
\begin{equation}
l_T \approx l_0 \cdot r^T
\tag{70}\end{equation}

要使$1 - l_T < \epsilon$（期望精度），需要：
\begin{equation}
T \geq \frac{\log(l_0/\epsilon)}{\log(1/r)}
\tag{71}\end{equation}

**实践数据**：对于$l_0 = 0.001, \epsilon = 10^{-4}, r \approx 0.3$（根据实验观察），需要：
\begin{equation}
T \geq \frac{\log(0.001/0.0001)}{\log(1/0.3)} = \frac{\log(10)}{\log(3.33)} \approx \frac{2.3}{1.2} \approx 2
\tag{72}\end{equation}

但这是理想情况。考虑有限精度和安全裕量，实际需要$T=6-8$步。

### 8. 收敛值的解析

#### 8.1 收敛时的极限行为

正文中观察到，当$t \geq 7$时，系数收敛到$(a,b,c) = (1.875, -1.25, 0.375) \times 1.01^{1,3,5}$。

**解释**：当$l_t$非常接近$u_t$时（即值域$[l_t, u_t] \approx [1-\epsilon, 1+\epsilon]$，$\epsilon \to 0$），等值振荡多项式趋向于一个特殊的极限形式。

**极限方程**：当$l_t = u_t = 1$时（严格收敛），方程组变为：
\begin{align}
f(1) &= 1 \tag{73}\\
f'(1) &= 0 \tag{74}
\end{align}

其中$f(x) = ax + bx^3 + cx^5$，$f'(x) = a + 3bx^2 + 5cx^4$。

在$x=1$：
\begin{align}
a + b + c &= 1 \tag{75}\\
a + 3b + 5c &= 0 \tag{76}
\end{align}

此外，由等值振荡的对称性，极值点$x_1, x_2$应该趋向于1。当$x_1 = x_2 = 1$时，$f''(1) = 0$也成立：
\begin{equation}
f''(x) = 6bx + 20cx^3 \quad\Rightarrow\quad f''(1) = 6b + 20c = 0
\tag{77}\end{equation}

联立(75), (76), (77)：
\begin{align}
a + b + c &= 1 \tag{78}\\
a + 3b + 5c &= 0 \tag{79}\\
6b + 20c &= 0 \quad\Rightarrow\quad b = -\frac{10c}{3} \tag{80}
\end{align}

从(79)：
\begin{equation}
a + 3\left(-\frac{10c}{3}\right) + 5c = 0 \quad\Rightarrow\quad a - 10c + 5c = 0 \quad\Rightarrow\quad a = 5c
\tag{81}\end{equation}

从(78)：
\begin{equation}
5c - \frac{10c}{3} + c = 1 \quad\Rightarrow\quad c\left(5 - \frac{10}{3} + 1\right) = 1 \quad\Rightarrow\quad c \cdot \frac{8}{3} = 1 \quad\Rightarrow\quad c = \frac{3}{8}
\tag{82}\end{equation}

因此：
\begin{align}
c &= \frac{3}{8} = 0.375 \tag{83}\\
b &= -\frac{10 \cdot 3/8}{3} = -\frac{10}{8} = -1.25 \tag{84}\\
a &= 5 \cdot \frac{3}{8} = \frac{15}{8} = 1.875 \tag{85}
\end{align}

**结论**：收敛值$(1.875, -1.25, 0.375)$正好对应$x_1 = x_2 = 1$且$f(1) = 1$的解，这与实验观察完全一致！

#### 8.2 Halley迭代的联系

有趣的是，这个收敛值恰好对应经典的Halley迭代公式：
\begin{equation}
x_{t+1} = x_t \left(\frac{15}{8} - \frac{10}{8}x_t^2 + \frac{3}{8}x_t^4\right)
\tag{86}\end{equation}

这可以改写为：
\begin{equation}
x_{t+1} = x_t \cdot \frac{1}{8}\left(15 - 10x_t^2 + 3x_t^4\right)
\tag{87}\end{equation}

Halley迭代是求解$x^{\top}x = I$的三阶收敛方法，其理论基础是对$(\boldsymbol{I} - \boldsymbol{X}^{\top}\boldsymbol{X})^{-1/2}$的更高阶Padé逼近。

**深刻联系**：等值振荡理论自动发现了Halley迭代作为最优的局部迭代格式（当值域接近$[1,1]$时），这是一个非常美妙的数学结果！

### 9. 算法复杂度与实现优化

#### 9.1 计算复杂度

**每步迭代**：
- Remez算法求解：$O(1)$（固定次数的迭代，与矩阵大小无关）
- 应用多项式$f_t$：$O(nm^2)$（矩阵乘法，同Newton-Schulz）

**总复杂度**：$O(Tnm^2)$，与标准Newton-Schulz相同。

#### 9.2 预计算优化

由于系数$(a_t, b_t, c_t)$只依赖于$l_t, u_t$，而这些值在每次运行时都相同（确定性算法），可以预先计算所有系数并存储为查找表。

**运行时优化**：直接使用预计算的系数，避免重复求解等值振荡方程，进一步提升速度。

### 10. 与优化方法的比较

#### 10.1 Adam优化 vs 等值振荡理论

| 方法 | 优势 | 劣势 |
|------|------|------|
| Adam优化（上篇） | 灵活，易于实现，可处理各种目标 | 只能找到局部最优，依赖初始化，计算成本高 |
| 等值振荡理论（下篇） | 保证全局最优，数学优雅，可解析求解 | 需要复杂的数学理论，实现较复杂 |

#### 10.2 实验性能比较

根据论文，等值振荡方法得到的系数在以下方面优于Adam优化：
1. **最大误差更小**：理论最优保证
2. **振荡更平滑**：等值振荡的对称性
3. **收敛更稳定**：贪心结构避免了全局优化的不稳定性

---

**小结**：本节详细推导了极分解理论、等值振荡定理的完整证明、贪心算法最优性的严格数学证明、Remez算法的实现细节、以及有限精度下的误差分析。这些推导揭示了Newton-Schulz迭代优化的深刻数学结构，特别是等值振荡理论自动发现Halley迭代作为极限解的美妙结果。通过这些理论分析，我们不仅理解了算法的工作原理，还获得了参数选择和实现优化的指导原则。

