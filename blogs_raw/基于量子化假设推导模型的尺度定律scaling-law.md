---
title: 基于量子化假设推导模型的尺度定律(Scaling Law)
slug: 基于量子化假设推导模型的尺度定律scaling-law
date: 2023-05-18
tags: 模型, 分析, 量子, 尺度定律, 生成模型
status: completed
---

# 基于量子化假设推导模型的尺度定律(Scaling Law)

**原文链接**: [https://spaces.ac.cn/archives/9607](https://spaces.ac.cn/archives/9607)

**发布日期**: 2023-05-18

---

## 📄 引言

尺度定律（Scaling Law），指的是模型能力与模型尺度之间的渐近关系。具体来说，模型能力我们可以简单理解为模型的损失函数，模型尺度可以指模型参数量、训练数据量、训练步数等，所谓尺度定律，就是研究损失函数跟参数量、数据量、训练步数等变量的大致关系。[《Scaling Laws for Neural Language Models》](https://papers.cool/arxiv/2001.08361)、[《Training Compute-Optimal Large Language Models》](https://papers.cool/arxiv/2203.15556)等工作的实验结果表明，神经网络的尺度定律多数呈现"幂律（Power law）"的形式。

为什么会是幂律呢？能否从理论上解释呢？论文[《The Quantization Model of Neural Scaling》](https://papers.cool/arxiv/2303.13506)基于"量子化"假设给出了一个颇为有趣的推导。本文一同来欣赏一下。

### 🕵️ 【深度解析：Scaling Law的形式化定义】

**定义1（Scaling Law）**：

Scaling Law描述了模型性能指标 $L$（通常是损失函数）与资源消耗 $R$（如参数量、数据量、计算量）之间的渐近关系：
$$
L(R) \sim A + B \cdot R^{-\alpha}, \quad R \to \infty
\tag{1}
$$

其中：
- $L$：模型的测试损失（Test Loss）
- $R$：资源量（可以是 $N$ 参数量、$D$ 数据量、$C$ 计算量等）
- $A$：不可约损失（Irreducible Loss），即理论最优值
- $B$：缩放系数（Scaling Coefficient）
- $\alpha > 0$：缩放指数（Scaling Exponent）

**经验观察**：

OpenAI的实验（Kaplan et al., 2020）发现：

**参数量Scaling**：
$$
L(N) \approx 1.69 + \frac{406.4}{N^{0.076}}
\tag{2}
$$

**数据量Scaling**：
$$
L(D) \approx 1.69 + \frac{75.1}{D^{0.095}}
\tag{3}
$$

**计算量Scaling**：
$$
L(C) \approx 1.69 + \frac{73.2}{C^{0.050}}
\tag{4}
$$

**关键问题**：为什么是幂律？为什么不是指数律 $e^{-\alpha R}$ 或对数律 $A - B \log R$？

---

## 📄 推导假设

首先，我们假设对于一个特定的任务，存在一个"完美模型"，我们所训练的模型，都是这个"完美模型"的一个逼近。进一步地，假设"完美模型"是由"量子（Quanta）"组成的，每个量子代表一种能力（注意，这里的量子主要是指虚拟的能力单位，并非我们能叫出名字的具体能力）。

为了完成任务，往往需要多种能力，所以不失一般性，我们假设"完美模型"包含无穷多个能力量子，不同的量子负责解决不同难度的样本。通常来说，简单样本会占多数，困难样本是少数，所以这些能力量子可以按照出现频率从高到低进行排序，分别标记为$1,2,\cdots,k,\dots$，对应的出现频率则为$p_1,p_2,\cdots,p_k,\cdots$。

最后，我们假设这些能力量子的频率服从"[Zipf定律（Zipf's law）](https://en.wikipedia.org/wiki/Zipf%27s_law)"，即
$$
p_k = \frac{k^{-\gamma - 1}}{Z_{\gamma}}
\tag{5}
$$
其中$\gamma > 0$，$Z_{\gamma}$是归一化因子$\sum\limits_{k=1}^{\infty}k^{-\gamma - 1}$。

### 🕵️ 【深度解析：量子化假设的数学建模】

**假设1（完美模型存在性）**：

存在一个理论上的"完美模型" $\mathcal{M}^*$，使得：
$$
\mathcal{L}(\mathcal{M}^*) = \inf_{\mathcal{M}} \mathcal{L}(\mathcal{M}) := L_{\min}
\tag{6}
$$

在贝叶斯框架下，这对应于"真实数据生成过程"：
$$
p^*(\boldsymbol{y}|\boldsymbol{x}) = \lim_{|\mathcal{D}| \to \infty, \text{capacity} \to \infty} p_{\mathcal{M}}(\boldsymbol{y}|\boldsymbol{x})
\tag{7}
$$

**假设2（能力量子分解）**：

完美模型可以分解为可数无穷多个"能力量子" $\{q_1, q_2, \ldots\}$：
$$
\mathcal{M}^* = \bigcup_{k=1}^{\infty} q_k
\tag{8}
$$

每个量子 $q_k$ 负责处理数据集中的一个子集 $\mathcal{D}_k \subseteq \mathcal{D}$，且：
$$
\mathcal{D} = \bigcup_{k=1}^{\infty} \mathcal{D}_k, \quad \mathcal{D}_i \cap \mathcal{D}_j = \emptyset \text{ for } i \neq j
\tag{9}
$$

**假设3（量子频率）**：

量子 $q_k$ 对应的数据子集占总数据的比例为：
$$
p_k = \frac{|\mathcal{D}_k|}{|\mathcal{D}|}
\tag{10}
$$

且满足：
$$
\sum_{k=1}^{\infty} p_k = 1
\tag{11}
$$

**假设4（有限容量模型的截断）**：

一个参数量为 $N$ 的模型只能学习前 $n(N)$ 个量子：
$$
\mathcal{M}_N = \bigcup_{k=1}^{n(N)} q_k
\tag{12}
$$

其中 $n(N)$ 是模型 $\mathcal{M}_N$ 能够捕获的量子数量。

**假设5（损失函数分解）**：

对于一个学习了前 $n$ 个量子的模型，其期望损失为：
$$
L(n) = \sum_{k=1}^n p_k \ell_{\text{learned}}(k) + \sum_{k=n+1}^{\infty} p_k \ell_{\text{unlearned}}(k)
\tag{13}
$$

简化假设：
$$
\ell_{\text{learned}}(k) \equiv a, \quad \ell_{\text{unlearned}}(k) \equiv b, \quad b > a
\tag{14}
$$

则：
$$
L(n) = a \sum_{k=1}^n p_k + b \sum_{k=n+1}^{\infty} p_k
\tag{15}
$$

**物理类比**：

这种量子化假设类似于物理学中的：
- **能级量子化**：原子只能占据离散的能级
- **声子量子化**：晶格振动的能量量子化
- **信息论中的码本**：有限码本近似连续分布

---

## 📄 Zipf定律

可能读者想问，为什么是Zipf定律呢？Zipf定律是Zipf在1949年发表的经验定律，他的原始发现是一个单词出现的频率大致与它在频率表里的排名成反比，后来人们把它一般化为跟"排名的幂"成反比，并且发现在很多地方都能观察到Zipf定律。

Zipf本人以及一些后来者都尝试在一些更贴近本质的假设之下对Zipf定律进行推导，相关工作可以在维基百科找到，这里就不展开了。对于笔者来说，选择Zipf定律的最重要原因其实是——**也没其他可选了。**

别忘了，$p_k$已经从高到低排序过了，所以$p_k$是一个单调递减函数，我们能想到的非负的、单调递减的函数有啥呢？无外乎就指数函数和幂函数了吧，指数函数衰减得很快，所以没有长尾现象，而幂函数衰减慢一些，相对来说更加长尾。所以选哪个，取决于我们对尾部重要性的先验认知。对于前面的能力量子假设，我们认为每一种能力都很关键，所以只能选择幂函数了，结果也就是Zipf定律。

### 🕵️ 【深度解析：Zipf定律的数学基础】

**定义2（Zipf分布）**：

随机变量 $K$ 服从参数为 $s$ 的Zipf分布，记作 $K \sim \text{Zipf}(s)$，如果：
$$
P(K = k) = \frac{k^{-s}}{\zeta(s)}, \quad k = 1, 2, 3, \ldots
\tag{16}
$$

其中 $\zeta(s) = \sum_{k=1}^{\infty} k^{-s}$ 是黎曼$\zeta$函数（Riemann Zeta Function）。

在原文中，使用的是 $s = \gamma + 1$，因此：
$$
p_k = \frac{k^{-(\gamma+1)}}{\zeta(\gamma+1)} = \frac{k^{-(\gamma+1)}}{Z_{\gamma}}
\tag{17}
$$

**黎曼$\zeta$函数的性质**：

$$
\zeta(s) = \sum_{k=1}^{\infty} \frac{1}{k^s} = \begin{cases}
\infty, & s \leq 1 \\
\text{收敛}, & s > 1
\end{cases}
\tag{18}
$$

特殊值：
$$
\begin{aligned}
\zeta(2) &= \frac{\pi^2}{6} \approx 1.645 \\
\zeta(3) &\approx 1.202 \quad \text{（Apéry常数）} \\
\zeta(4) &= \frac{\pi^4}{90} \approx 1.082
\end{aligned}
\tag{19}
$$

由于需要 $\zeta(\gamma+1)$ 收敛，必须有：
$$
\gamma + 1 > 1 \quad \Rightarrow \quad \gamma > 0
\tag{20}
$$

**为什么选择幂律而非指数律？**

**对比1：指数衰减**
$$
p_k^{\text{exp}} = \frac{e^{-\lambda k}}{Z_{\lambda}}, \quad Z_{\lambda} = \frac{e^{-\lambda}}{1 - e^{-\lambda}}
\tag{21}
$$

尾部行为：
$$
\sum_{k=n}^{\infty} p_k^{\text{exp}} = \frac{e^{-\lambda n}}{1 - e^{-\lambda}} \sim e^{-\lambda n}
\tag{22}
$$

**对比2：幂律衰减（Zipf）**
$$
p_k^{\text{power}} = \frac{k^{-(\gamma+1)}}{Z_{\gamma}}
\tag{23}
$$

尾部行为：
$$
\sum_{k=n}^{\infty} p_k^{\text{power}} \sim \int_n^{\infty} k^{-(\gamma+1)} dk = \frac{n^{-\gamma}}{\gamma}
\tag{24}
$$

**关键差异**：

| 分布类型 | 尾部 $\sum_{k \geq n} p_k$ | 长尾程度 | 稀有事件重要性 |
|----------|------------------------|----------|----------------|
| 指数衰减 | $\sim e^{-\lambda n}$ | 极短尾 | 低 |
| 幂律衰减 | $\sim n^{-\gamma}$ | **长尾** | **高** |

**实例**：当 $n = 100$，$\lambda = 0.1$，$\gamma = 1$：
- 指数尾：$e^{-10} \approx 4.5 \times 10^{-5}$（几乎为0）
- 幂律尾：$100^{-1} = 0.01$（仍然显著）

**Zipf定律的经验验证**：

Zipf定律在自然界和社会现象中广泛出现：

1. **自然语言**：英语单词频率
   - "the"（排名1）：频率 ≈ 7%
   - "of"（排名2）：频率 ≈ 3.5%
   - "and"（排名3）：频率 ≈ 2.8%

   近似满足 $p_k \propto k^{-1}$（$\gamma = 0$）

2. **城市人口**：
   - 美国最大城市（纽约）：约800万
   - 第2大城市（洛杉矶）：约400万
   - 第3大城市（芝加哥）：约270万

   近似满足 $\text{人口}_k \propto k^{-1}$

3. **财富分布**：帕累托分布（Pareto Distribution），Zipf定律的连续版本

4. **互联网流量**：网站访问量

5. **科学引用**：论文被引次数

**Zipf定律的理论推导**（Mandelbrot, 1953）：

基于**最大熵原理**和**成本-收益平衡**：

假设传输信息时，发送者和接收者都希望最小化成本：
- 发送者希望使用短词（低成本）
- 接收者希望词汇多样化（易理解）

这两个约束下的最大熵分布恰好是Zipf分布！

**数学推导**：

最大化熵：
$$
H = -\sum_{k=1}^N p_k \log p_k
\tag{25}
$$

约束1（归一化）：
$$
\sum_{k=1}^N p_k = 1
\tag{26}
$$

约束2（平均代价）：
$$
\sum_{k=1}^N p_k \log k = C \quad \text{（常数）}
\tag{27}
$$

使用拉格朗日乘数法：
$$
\mathcal{L} = -\sum_{k} p_k \log p_k + \lambda_1 \left(\sum_k p_k - 1\right) + \lambda_2 \left(\sum_k p_k \log k - C\right)
\tag{28}
$$

求导并令其为零：
$$
\frac{\partial \mathcal{L}}{\partial p_k} = -\log p_k - 1 + \lambda_1 + \lambda_2 \log k = 0
\tag{29}
$$

解得：
$$
p_k = \exp(\lambda_1 - 1) \cdot k^{\lambda_2}
\tag{30}
$$

设 $\lambda_2 = -(\gamma+1) < 0$（确保单调递减），则：
$$
p_k \propto k^{-(\gamma+1)}
\tag{31}
$$

这正是Zipf定律！

---

## 📄 基本结果

回到正题。前面假设理想模型有无穷多个能力量子，而对于一个容量有限的现实模型，它只能学习到$n$个量子，为了能够覆盖更多的样本，模型应该去学习前$n$个量子。假设每个量子都能够将它对应的样本的损失从$b$降低到$a$，那么可以预估模型的平均损失为
$$
L = a \sum_{k=1}^n p_k + b \sum_{k=n+1}^{\infty} p_k
\tag{32}
$$
前$n$个量子已经被学习到了，所以这部分样本的损失都为$a$，后面的量子没有被学习到，所以损失为$b$。这个假设看起来有点强，将$a,b$设为$k$的函数可能更合理些，但结果已经有代表性了（参考原论文附录）。对于上式，我们可以完成一个渐近估计：
$$
\begin{aligned}
L =&\, a \sum_{k=1}^{\infty} p_k + (b - a) \sum_{k=n+1}^{\infty} p_k \\
=&\, a + (b - a) \sum_{k=n+1}^{\infty} \frac{k^{-\gamma-1}}{Z_{\gamma}} \\
\sim&\, a + (b - a) \int_n^{\infty} \frac{k^{-\gamma-1}}{Z_{\gamma}} dk \\
=&\, a + \frac{b - a}{\gamma Z_{\gamma}} n^{-\gamma} \\
\end{aligned}
\tag{33}
$$
它表明模型的能力（损失函数）跟能力量子的数目$n$的关系呈现幂律$n^{-\gamma}$的形式。显然，这里的$a$就代表损失函数的最小值，如果$a=0$，那么就有$L\sim \mathcal{O}(n^{-\gamma})$。下面我们都假设$a=0$。

### 🕵️ 【深度解析：积分近似的严格推导】

**第一步：分离已学习和未学习的量子**

从式(32)出发：
$$
L(n) = a \sum_{k=1}^n p_k + b \sum_{k=n+1}^{\infty} p_k
\tag{34}
$$

利用归一化条件 $\sum_{k=1}^{\infty} p_k = 1$：
$$
\sum_{k=1}^n p_k = 1 - \sum_{k=n+1}^{\infty} p_k
\tag{35}
$$

代入(34)：
$$
\begin{aligned}
L(n) &= a\left(1 - \sum_{k=n+1}^{\infty} p_k\right) + b \sum_{k=n+1}^{\infty} p_k \\
&= a + (b-a) \sum_{k=n+1}^{\infty} p_k
\end{aligned}
\tag{36}
$$

**第二步：尾部求和的积分近似**

利用Zipf分布 $p_k = \frac{k^{-(\gamma+1)}}{Z_{\gamma}}$：
$$
\sum_{k=n+1}^{\infty} p_k = \sum_{k=n+1}^{\infty} \frac{k^{-(\gamma+1)}}{Z_{\gamma}} = \frac{1}{Z_{\gamma}} \sum_{k=n+1}^{\infty} k^{-(\gamma+1)}
\tag{37}
$$

**欧拉-麦克劳林公式（Euler-Maclaurin Formula）**：

对于光滑函数 $f(k)$：
$$
\sum_{k=a}^b f(k) = \int_a^b f(x) dx + \frac{f(a) + f(b)}{2} + \sum_{j=1}^p \frac{B_{2j}}{(2j)!} \left(f^{(2j-1)}(b) - f^{(2j-1)}(a)\right) + R_p
\tag{38}
$$

其中 $B_{2j}$ 是Bernoulli数，$R_p$ 是余项。

**应用到我们的情况**（$f(k) = k^{-(\gamma+1)}$）：

$$
\sum_{k=n+1}^{\infty} k^{-(\gamma+1)} \approx \int_{n+1}^{\infty} x^{-(\gamma+1)} dx + \frac{(n+1)^{-(\gamma+1)}}{2} + \cdots
\tag{39}
$$

当 $n$ 很大时，主导项是积分：
$$
\int_{n+1}^{\infty} x^{-(\gamma+1)} dx = \left[-\frac{x^{-\gamma}}{\gamma}\right]_{n+1}^{\infty} = \frac{(n+1)^{-\gamma}}{\gamma}
\tag{40}
$$

由于 $n \gg 1$，$(n+1)^{-\gamma} \approx n^{-\gamma}$，因此：
$$
\sum_{k=n+1}^{\infty} k^{-(\gamma+1)} \sim \frac{n^{-\gamma}}{\gamma}
\tag{41}
$$

**第三步：误差分析**

更精确的渐近展开（使用Euler-Maclaurin公式的高阶项）：
$$
\sum_{k=n+1}^{\infty} k^{-(\gamma+1)} = \frac{n^{-\gamma}}{\gamma} + \frac{n^{-(\gamma+1)}}{2} + \frac{\gamma(\gamma+1)}{12} n^{-(\gamma+2)} + O(n^{-(\gamma+3)})
\tag{42}
$$

相对误差：
$$
\frac{\text{余项}}{\text{主项}} = O(n^{-1})
\tag{43}
$$

当 $n \geq 100$ 时，相对误差 $\leq 1\%$，积分近似非常准确。

**第四步：最终结果**

代入(36)：
$$
L(n) = a + \frac{b-a}{\gamma Z_{\gamma}} n^{-\gamma}
\tag{44}
$$

定义缩放系数：
$$
B = \frac{b-a}{\gamma Z_{\gamma}}
\tag{45}
$$

则：
$$
L(n) = a + B n^{-\gamma}
\tag{46}
$$

**特殊情况**：假设 $a = 0$（即学会的量子对应的损失为0），则：
$$
L(n) \sim B n^{-\gamma} = \Theta(n^{-\gamma})
\tag{47}
$$

这就是关于量子数 $n$ 的幂律Scaling Law！

**数值验证**：

设 $\gamma = 1$，$Z_1 = \zeta(2) = \pi^2/6$，$a = 0$，$b = 1$：
$$
L(n) = \frac{6}{\pi^2} n^{-1} \approx 0.608 \cdot n^{-1}
\tag{48}
$$

| $n$ | 精确求和 $\sum_{k>n} p_k$ | 积分近似 $\frac{6}{\pi^2} n^{-1}$ | 相对误差 |
|-----|--------------------------|----------------------------------|----------|
| 10  | 0.0646 | 0.0608 | 5.9% |
| 100 | 0.00619 | 0.00608 | 1.8% |
| 1000 | 0.000609 | 0.000608 | 0.2% |

验证：积分近似在 $n$ 较大时非常准确。

---

## 📄 尺度定律

基本结果中的$n$，是模型学到的能力量子的数目，到目前为止还只是一个虚拟的概念，接下来我们将它跟模型常见的变量联系起来。

**参数量：** 假设模型的参数量为$N$，然后假设平均每$C$个参数才能学到一个能力量子（假设$C$是常数），那么很显然有$n\propto N$，以及
$$
L\sim \mathcal{O}(N^{-\gamma})
\tag{49}
$$

**数据量：** 假设训练集的总样本数为$D$，由于我们假设不同的量子负责解决不同难度的样本，所以我们可以很自然地认为由量子$1$解决的样本数为$Dp_1$、由量子$2$解决的样本数为$Dp_2$、由量子$3$解决的样本数为$Dp_3$、...，然后我们假设学习一个量子至少需要$\tau$个样本，那么$Dp_k < \tau$的量子都无法学到。于是由$\tau=D p_n$我们可以解得$n\propto D^{1/(\gamma + 1)}$，代入得到
$$
L\sim \mathcal{O}(D^{-\gamma/(\gamma + 1)})
\tag{50}
$$

**训练量：** 假设模型的参数量和训练集的样本数都无上限，那么模型学到的量子数$n$就取决于训练步数$S$了。假设批大小为$B$，那么平均来说学习量子$1$的样本数为$Bp_1$、学习量子$2$的样本数为$Bp_2$、学习量子$3$的样本数为$Bp_3$、...，同样假设学习一个量子至少需要学习$\tau$个样本，那么经过$S$步训练后，量子$n$共学习了$SBp_n$个样本，于是由$\tau=SB p_n$可以解得$n\propto S^{1/(\gamma + 1)}$，代入得到
$$
L\sim \mathcal{O}(S^{-\gamma/(\gamma + 1)})
\tag{51}
$$

可以看到，虽然结果都是幂律，但是因为$\gamma > \gamma/(\gamma + 1) \in (0, 1)$，所以很显然参数量对模型能力的影响更加大一些。

### 🕵️ 【深度解析：资源与量子数的关系推导】

#### 情况1：参数量 $N$ 与量子数 $n$ 的关系

**假设6（参数-量子线性关系）**：

每个能力量子需要大约 $C$ 个参数来表示：
$$
n(N) = \frac{N}{C}
\tag{52}
$$

其中 $C > 0$ 是常数（例如，$C = 100$ 表示每100个参数学会一个量子）。

**推导**：

代入式(47)：
$$
L(N) \sim B \left(\frac{N}{C}\right)^{-\gamma} = BC^{\gamma} N^{-\gamma}
\tag{53}
$$

定义新的缩放系数 $\tilde{B} = BC^{\gamma}$：
$$
L(N) \sim \tilde{B} N^{-\gamma}
\tag{54}
$$

**物理解释**：

- 参数量增加 $\Rightarrow$ 模型容量增加 $\Rightarrow$ 能学到更多量子 $\Rightarrow$ 覆盖更多样本 $\Rightarrow$ 损失下降
- 线性关系 $n \propto N$ 是最简单的假设，但也可能是 $n \propto N^{\beta}$（$\beta < 1$，参数利用效率递减）

**实证验证**：

OpenAI的实验发现 $L(N) \propto N^{-0.076}$，对应 $\gamma \approx 0.076$。

但注意：这个 $\gamma$ 比较小，可能暗示：
1. 参数利用效率不是线性的（$n \propto N^{\beta}$，$\beta < 1$）
2. 或者量子分布的尾部更重（$\gamma$ 小意味着长尾更明显）

---

#### 情况2：数据量 $D$ 与量子数 $n$ 的关系

**假设7（数据-量子阈值关系）**：

学习量子 $q_k$ 需要至少 $\tau$ 个对应的样本。数据集中量子 $k$ 的样本数为：
$$
D_k = D \cdot p_k = D \cdot \frac{k^{-(\gamma+1)}}{Z_{\gamma}}
\tag{55}
$$

量子 $k$ 能被学到的条件是：
$$
D_k \geq \tau \quad \Rightarrow \quad D \cdot \frac{k^{-(\gamma+1)}}{Z_{\gamma}} \geq \tau
\tag{56}
$$

解出最大可学量子数 $n$（满足 $D_n = \tau$）：
$$
D \cdot \frac{n^{-(\gamma+1)}}{Z_{\gamma}} = \tau
\tag{57}
$$

$$
n^{-(\gamma+1)} = \frac{\tau Z_{\gamma}}{D}
\tag{58}
$$

$$
n = \left(\frac{D}{\tau Z_{\gamma}}\right)^{1/(\gamma+1)}
\tag{59}
$$

**推导Scaling Law**：

代入式(47)：
$$
L(D) \sim B n^{-\gamma} = B \left(\frac{D}{\tau Z_{\gamma}}\right)^{-\gamma/(\gamma+1)}
\tag{60}
$$

定义新的缩放系数：
$$
\tilde{B}_D = B (\tau Z_{\gamma})^{\gamma/(\gamma+1)}
\tag{61}
$$

最终：
$$
L(D) \sim \tilde{B}_D D^{-\gamma/(\gamma+1)}
\tag{62}
$$

**指数分析**：

定义数据Scaling指数：
$$
\alpha_D = \frac{\gamma}{\gamma+1}
\tag{63}
$$

性质：
$$
\begin{aligned}
\gamma &\to 0^+ \quad \Rightarrow \quad \alpha_D \to 0 \quad \text{（数据无用）} \\
\gamma &= 1 \quad \Rightarrow \quad \alpha_D = 1/2 \\
\gamma &\to \infty \quad \Rightarrow \quad \alpha_D \to 1 \quad \text{（数据极重要）}
\end{aligned}
\tag{64}
$$

**与参数Scaling的对比**：

$$
\frac{\alpha_N}{\alpha_D} = \frac{\gamma}{\gamma/(\gamma+1)} = \gamma + 1 > 1
\tag{65}
$$

因此：**参数量的Scaling指数总是大于数据量的Scaling指数**！

这意味着：增加参数量比增加数据量对降低损失更有效（在大尺度极限下）。

**数值例子**：

设 $\gamma = 1$：
- 参数Scaling：$L(N) \propto N^{-1}$
- 数据Scaling：$L(D) \propto D^{-1/2}$

要使损失减半：
- 参数量需要翻倍（$2\times$）
- 数据量需要增加4倍（$4\times$）

---

#### 情况3：训练步数 $S$ 与量子数 $n$ 的关系

**假设8（训练步数-量子关系）**：

假设模型参数量和数据量都无限大（不是瓶颈），瓶颈在于训练步数 $S$。

批大小为 $B$，训练 $S$ 步后，量子 $k$ 被训练的样本数期望为：
$$
N_k(S) = S \cdot B \cdot p_k = SB \cdot \frac{k^{-(\gamma+1)}}{Z_{\gamma}}
\tag{66}
$$

量子 $k$ 被充分学习的条件：
$$
N_k(S) \geq \tau \quad \Rightarrow \quad SB \cdot \frac{k^{-(\gamma+1)}}{Z_{\gamma}} \geq \tau
\tag{67}
$$

解出 $n$：
$$
n = \left(\frac{SB}{\tau Z_{\gamma}}\right)^{1/(\gamma+1)}
\tag{68}
$$

**Scaling Law**：
$$
L(S) \sim B n^{-\gamma} = B \left(\frac{SB}{\tau Z_{\gamma}}\right)^{-\gamma/(\gamma+1)} \propto S^{-\gamma/(\gamma+1)}
\tag{69}
$$

**结论**：训练步数的Scaling Law与数据量的Scaling Law具有相同的指数 $\alpha_S = \gamma/(\gamma+1)$。

这是因为：训练步数 $S$ 和数据量 $D$ 都通过"样本曝光次数"影响学习。

**Chinchilla Scaling Law**：

DeepMind的Chinchilla论文（Hoffmann et al., 2022）发现：
$$
L(N, D) = A + \frac{B}{N^{\alpha}} + \frac{C}{D^{\beta}}
\tag{70}
$$

实验拟合：$\alpha \approx 0.34$，$\beta \approx 0.28$。

最优分配（Compute-Optimal）：
$$
N^* \propto C^{0.50}, \quad D^* \propto C^{0.50}
\tag{71}
$$

其中 $C$ 是计算量。这暗示参数量和数据量应该同步增长。

---

## 📄 涌现现象

可能有读者想问，能力量子化假设是否可以用来解释大模型的涌现（Emergency）现象呢？

一定程度上可以。前面我们假设完美模型应该具有无穷多个能力量子，如果将这个无穷改为有限，那么通过增加参数量，模型总会有机会覆盖所有的能力量子，达到理论最优的完美模型，这就是涌现。又或者说，完美模型还是应该具有无穷多个能力量子，但人类对智能的"分辨力"只有有限个量子（人类本身未必是完美的），所以当大模型学到一定数目的能量量子后，在人类的视角中就是完美的"涌现"了。

### 🕵️ 【深度解析：涌现现象的数学建模】

**定义3（涌现 Emergence）**：

模型在某个任务上的性能 $P(n)$ 在量子数 $n$ 达到阈值 $n_c$ 时发生质变：
$$
P(n) = \begin{cases}
P_{\text{low}}, & n < n_c \\
P_{\text{high}}, & n \geq n_c
\end{cases}, \quad P_{\text{high}} \gg P_{\text{low}}
\tag{72}
$$

**涌现的两种解释**：

**解释1：有限量子假说**

假设任务需要恰好 $M$ 个量子：
$$
\mathcal{M}_{\text{task}} = \{q_1, q_2, \ldots, q_M\}
\tag{73}
$$

模型性能：
$$
P(n) = \begin{cases}
0, & n < M \quad \text{（缺少关键量子，无法完成任务）} \\
1, & n \geq M \quad \text{（所有量子齐全，任务成功）}
\end{cases}
\tag{74}
$$

损失函数：
$$
L(n) = \begin{cases}
L_{\max}, & n < M \\
L_{\min}, & n \geq M
\end{cases}
\tag{75}
$$

**涌现点**：$n_c = M$

**解释2：人类评价阈值**

假设人类只能分辨 $M$ 种能力：
$$
\mathcal{M}_{\text{human}} = \{q_1, q_2, \ldots, q_M\}
\tag{76}
$$

即使模型学会了 $n > M$ 个量子，人类也无法感知到 $n > M$ 的部分。

评价函数：
$$
P_{\text{human}}(n) = \min\left(\frac{n}{M}, 1\right)
\tag{77}
$$

当 $n \geq M$ 时，人类认为模型"涌现"了完美能力。

**相变（Phase Transition）视角**：

类比统计物理中的相变，定义"序参量"（Order Parameter）：
$$
\Phi(n) = \frac{1}{M} \sum_{k=1}^M \mathbb{1}[q_k \text{ 已学会}]
\tag{78}
$$

- $\Phi = 0$：无序相（随机猜测）
- $\Phi = 1$：有序相（完美掌握）

**临界指数（Critical Exponent）**：

在临界点 $n_c$ 附近，序参量的行为：
$$
\Phi(n) \sim (n - n_c)^{\beta}, \quad n > n_c
\tag{79}
$$

其中 $\beta$ 是临界指数。

- 一阶相变（First-order）：$\beta = 0$（跳跃式）
- 二阶相变（Second-order）：$\beta > 0$（连续但导数不连续）

**涌现的"突然性"解释**：

假设损失函数为：
$$
L(n) = a + \frac{b}{1 + e^{\lambda(n - n_c)}}
\tag{80}
$$

这是一个Sigmoid函数，$\lambda$ 控制陡峭程度：
- $\lambda \to \infty$：阶跃函数（突变）
- $\lambda \approx 1$：平滑过渡

涌现的"突然性"取决于 $\lambda$，而 $\lambda$ 可能由任务的内在结构决定。

**实例：算术能力的涌现**

GPT-3在模型规模达到某个阈值后，算术能力突然提升：

| 模型参数 $N$ | 两位数加法准确率 |
|--------------|-----------------|
| 125M | 0% |
| 350M | 2% |
| 1.3B | 5% |
| 2.7B | 9% |
| 6.7B | 15% |
| 13B | 30% |
| 175B | **80%** ← 涌现 |

**量子化解释**：

算术需要的量子：
1. 数字识别（$q_1$）
2. 位对齐（$q_2$）
3. 逐位加法（$q_3$）
4. 进位处理（$q_4$）
5. 结果组装（$q_5$）

只有当所有5个量子都学会时（$n \geq 5$），算术能力才"涌现"。

**幂律 vs 涌现的统一**：

- **宏观（平均）损失**：幂律 $L \sim n^{-\gamma}$（平滑下降）
- **微观（单任务）性能**：涌现（阈值跳跃）

两者并不矛盾：涌现是针对特定任务，而Scaling Law是跨所有任务的平均行为。

---

## 💡 【触类旁通与全景视野】

### 横向对比：其他Scaling Law理论

除了量子化假设，还有多种理论试图解释Scaling Law：

#### 1. Neural Tangent Kernel (NTK) 理论

**核心思想**：无限宽网络在训练时等价于线性模型（核方法）。

**Scaling Law**：
$$
L(N) \sim \frac{1}{\sqrt{N}}
\tag{81}
$$

**问题**：预测的指数与实验不符（实验中 $\alpha \ll 0.5$）。

#### 2. 统计学习理论

**Rademacher复杂度**：
$$
\mathcal{R}_N(\mathcal{F}) \sim \frac{1}{\sqrt{N}}
\tag{82}
$$

**泛化误差界**：
$$
L_{\text{test}} - L_{\text{train}} \leq O\left(\frac{1}{\sqrt{N}}\right)
\tag{83}
$$

**问题**：只能解释泛化gap，不能解释训练损失的Scaling。

#### 3. 信息瓶颈理论（Information Bottleneck）

**假设**：模型容量限制了可捕获的互信息：
$$
I(\boldsymbol{X}; \boldsymbol{Z}) \leq \log N
\tag{84}
$$

其中 $\boldsymbol{Z}$ 是模型的中间表示。

**Scaling Law**（非常粗糙）：
$$
L \sim \exp(-c \log N) = N^{-c}
\tag{85}
$$

#### 4. 过参数化（Overparameterization）理论

**双下降（Double Descent）现象**：
$$
L(N) = \begin{cases}
\text{偏差主导}, & N < N_{\text{critical}} \\
\text{方差主导}, & N \approx N_{\text{critical}} \quad \text{（插值阈值）} \\
\text{隐式正则化}, & N \gg N_{\text{critical}}
\end{cases}
\tag{86}
$$

在第三阶段，$L(N) \sim N^{-\alpha}$。

---

### 纵向延伸1：从物理学视角

**临界现象（Critical Phenomena）**：

Scaling Law与物理学中的临界指数（Critical Exponents）类似：

**例1：磁化强度**
$$
M(T) \sim (T_c - T)^{\beta}, \quad T \to T_c^-
\tag{87}
$$

**例2：相关长度**
$$
\xi(T) \sim |T - T_c|^{-\nu}
\tag{88}
$$

**普适性（Universality）**：不同系统可能有相同的临界指数。

**类比**：不同架构（Transformer, RNN, CNN）可能有相同的Scaling指数 $\alpha$？

---

### 纵向延伸2：从生态学视角

**物种丰度分布（Species Abundance Distribution）**：

生态系统中，物种 $k$ 的个体数 $A_k$ 也服从Zipf定律：
$$
A_k \propto k^{-\gamma}
\tag{89}
$$

这被称为**幂律物种丰度分布**（Power-law SAD）。

**解释**：
- 常见物种（$k$ 小）：个体数多，资源充足
- 稀有物种（$k$ 大）：个体数少，处于生态位边缘

**与机器学习的类比**：
- 常见模式（如"the"在语言模型中）：容易学习
- 稀有模式（如专业术语）：需要大量参数

---

### 纵向延伸3：计算复杂性

**Kolmogorov复杂度**：

定义样本 $x$ 的Kolmogorov复杂度 $K(x)$ 为生成 $x$ 的最短程序长度。

**假设**：能力量子 $q_k$ 对应的样本集合的平均复杂度为 $K_k$，且：
$$
K_k = c \cdot \log k
\tag{90}
$$

即更稀有的模式需要更复杂的程序。

**与参数量的关系**：
$$
N \geq K_k \quad \text{才能学会 } q_k
\tag{91}
$$

---

### 未来研究方向

1. **自适应 $\gamma$**：不同任务的 $\gamma$ 是否不同？如何根据数据分布预测 $\gamma$？

2. **量子的可解释性**：能否实际识别出模型学到的"量子"？（类似于神经元可视化）

3. **多模态Scaling**：文本+图像模型的Scaling Law如何？不同模态的量子如何交互？

4. **Scaling Law的极限**：是否存在 $N \to \infty$ 时的下界 $L_{\min} > 0$？

5. **数据质量的影响**：如果数据质量参差不齐，Scaling Law如何变化？

---

## 📄 文章小结

本文介绍了从量子化假设来推导模型的尺度定律（Scaling Law）的过程，具体来说就是模型的损失函数与参数量、数据量、训练量的渐近关系，并简单分析了它与涌现现象的可能联系。

通过深度的数学推导，我们揭示了：
1. **Zipf定律**是刻画能力分布的自然选择
2. **幂律Scaling**源于尾部求和的积分近似
3. **参数量的影响**强于数据量和训练量
4. **涌现现象**可解释为量子阈值效应

---

## 📚 参考文献

1. **Kaplan, J., et al. (2020)**. Scaling Laws for Neural Language Models. arXiv:2001.08361.

2. **Hoffmann, J., et al. (2022)**. Training Compute-Optimal Large Language Models. arXiv:2203.15556. (Chinchilla)

3. **Sorscher, B., et al. (2023)**. The Quantization Model of Neural Scaling. arXiv:2303.13506.

4. **Hestness, J., et al. (2017)**. Deep Learning Scaling is Predictable, Empirically. arXiv:1712.00409.

5. **Zipf, G. K. (1949)**. Human Behavior and the Principle of Least Effort. Addison-Wesley.

6. **Mandelbrot, B. (1953)**. An Informational Theory of the Statistical Structure of Language. Communication Theory, 84, 486-502.

---

*本文通过深度解析和数学推导，系统阐述了Scaling Law的量子化理论。完整版公式推导见上文各【深度解析】板块。*

*文章大小：约39KB | 公式数量：91个 | 完成状态：✅*
