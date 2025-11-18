---
title: GAU-α：尝鲜体验快好省的下一代Attention
slug: gau-α尝鲜体验快好省的下一代attention
date: 2022-04-22
tags: 语言模型, attention, 预训练, 生成模型, attention
status: pending
---

# GAU-α：尝鲜体验快好省的下一代Attention

**原文链接**: [https://spaces.ac.cn/archives/9052](https://spaces.ac.cn/archives/9052)

**发布日期**: 

---

在[《FLASH：可能是近来最有意思的高效Transformer设计》](/archives/8934)中，我们介绍了GAU（Gated Attention Unit，门控线性单元），在这里笔者愿意称之为“目前最有潜力的下一代Attention设计”，因为它真正达到了“更快（速度）、更好（效果）、更省（显存）”的特点。

然而，有些读者在自己的测试中得到了相反的结果，比如收敛更慢、效果更差等，这与笔者的测试结果大相径庭。本文就来分享一下笔者自己的训练经验，并且放出一个尝鲜版“GAU-α”供大家测试。

> **开源地址：<https://github.com/ZhuiyiTechnology/GAU-alpha>**

## GAU-α #

首先介绍一下开源出来的“GAU-α”在CLUE任务上的成绩单：  
$$\small{\begin{array}{c|ccccccccccc}  
\hline  
& \text{iflytek} & \text{tnews} & \text{afqmc} & \text{cmnli} & \text{ocnli} & \text{wsc} & \text{csl} & \text{cmrc2018} & \text{c3} & \text{chid} & \text{cluener}\\\  
\hline  
\text{BERT} & 60.06 & 56.80 & 72.41 & 79.56 & 73.93 & 78.62 & 83.93 & 56.17 & 60.54 & 85.69 & 79.45 \\\  
\text{RoBERTa} & 60.64 & \textbf{58.06} & 74.05 & 81.24 & 76.00 & \textbf{87.50} & 84.50 & 56.54 & 67.66 & 86.71 & 79.47\\\  
\text{RoFormer} & 60.91 & 57.54 & 73.52 & 80.92 & \textbf{76.07} & 86.84 & 84.63 & 56.26 & 67.24 & 86.57 & 79.72\\\  
\text{RoFormerV2}^* & 60.87 & 56.54 & 72.75 & 80.34 & 75.36 & 80.92 & 84.67 & 57.91 & 64.62 & 85.09 & \textbf{81.08}\\\  
\hline  
\text{GAU-}\alpha & \textbf{61.41} & 57.76 & \textbf{74.17} & \textbf{81.82} & 75.86 & 79.93 & \textbf{85.67} & \textbf{58.09} & \textbf{68.24} & \textbf{87.91} & 80.01\\\  
\hline  
\end{array}}$$

所有的模型都是Base版，上表显示的是CLUE任务上验证集上的结果，大家的运行方式和比较都是公平的，作为一个相对比较来说是合理的。另外，这里的RoFormerV2*并非[《RoFormerV2：自然语言理解的极限探索》](/archives/8998)中的多任务版本，而是仅仅进行了MLM预训练的版本（该版本没开源），这样对比是因为GAU-α也仅仅进行了MLM预训练。

从表中可以看出，除了WSC这个数据量极少的“异类”外，GAU-α在多数任务上都有优势，并且除了WSC外的平均成绩是最好的。其中，RoFormerV2*与GAU-α的比较是最为公平的，因为它们的训练脚本、训练数据、整体结构都是一样的，唯一不同就是GAU-α是将RoFormerV2*中的Attention+FFN组合换成了两层GAU，两者对比充分显示出了GAU设计“更好”的特点。

此外，我们在[《RoFormerV2：自然语言理解的极限探索》](/archives/8998)介绍过RoFormerV2对结构进行了简化，从而获得更快的速度，具有同样整体结构的GAU-α也是如此，所以GAU-α的速度是比表中的BERT、RoBERTa、RoFormer都要快的，但平均效果却更胜一筹。更进一步的测试显示，当序列长度超过512时，GAU-α的速度开始超过同样精简过的RoFormerV2，并且显存占用更低，越长则对GAU-α更有利。

## 训练 #

现在介绍一下模型的训练细节，完整的代码已经开源到Github中，如有疑惑可以对照着代码来读。

**模型架构** ： GAU-α就是将RoFormerV2的Attention+FFN换成了两层GAU，在[之前的文章](/archives/8934)中我们比较过两层GAU的计算量和参数量大致相当于Attention+FFN组合，所以这样的替换是合理的；RoFormerV2的特点是保留了Post Norm结构，去掉了所有的Bias项，并且Layer Norm换成了RMS Norm的最简单变体，在GAU-α中也是如此。

**归一化** ： 在[《听说Attention与Softmax更配哦～》](/archives/9019)中我们讨论过Attention的归一化问题，GAU-α的Attention归一化选取了其中笔者自行提出的具有较好外推能力的[熵不变性Softmax](/archives/8823)（在bert4keras中暂称为softmax_plus）。

**训练方式** ： 在初始化方面笔者按照[《训练1000层的Transformer究竟有什么困难？》](/archives/8978)进行了调整，因此无须Wamrup就可以直接训练，优化器用的是LAMB，学习率分段线性衰减；预训练任务用的是全词MLM，分词工具用百度的LAC，这些跟RoFormerV2都是对齐的。

好像值得一提的也就这么多了，确实没进行多大的改变。除了在归一化方式上花了点时间进行测试，其他方面也没多费时间，直接训练就得到了不错的效果。

## 小结 #

GAU是笔者认为的“目前最有潜力的下一代Attention设计”，本文分享了GAU的一些训练经验，并开源了一个尝鲜版“GAU-α”。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9052>_

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

苏剑林. (Apr. 22, 2022). 《GAU-α：尝鲜体验快好省的下一代Attention 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9052>

@online{kexuefm-9052,  
title={GAU-α：尝鲜体验快好省的下一代Attention},  
author={苏剑林},  
year={2022},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9052}},  
} 


---

## 公式推导与注释

### 1. GAU架构基础

#### 1.1 标准Attention回顾

标准的Scaled Dot-Product Attention定义为：
\begin{equation}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V \tag{1}
\end{equation}

其中 $Q, K, V \in \mathbb{R}^{n \times d}$ 分别是查询、键、值矩阵，$n$ 是序列长度，$d$ 是特征维度。

#### 1.2 GAU的核心设计

GAU（Gated Attention Unit）将注意力机制与门控机制结合，其核心形式为：
\begin{equation}
\text{GAU}(X) = (X \odot \text{Attention}(X)) W_O \tag{2}
\end{equation}

其中 $\odot$ 表示逐元素乘法（Hadamard积），这是GAU的关键创新点。

### 2. GAU完整数学推导

#### 2.1 输入变换

给定输入 $X \in \mathbb{R}^{n \times d}$，GAU首先通过三个线性变换：
\begin{equation}
U = XW_U + b_U \in \mathbb{R}^{n \times e} \tag{3}
\end{equation}
\begin{equation}
V = XW_V + b_V \in \mathbb{R}^{n \times e} \tag{4}
\end{equation}
\begin{equation}
\text{Base} = XW_{\text{base}} + b_{\text{base}} \in \mathbb{R}^{n \times e} \tag{5}
\end{equation}

其中 $e$ 是中间维度，通常 $e = 2d$ 以保持计算量平衡。

**推导注释**：这三个变换的作用不同：
- $U$ 用于生成门控信号和注意力的查询/键
- $V$ 用于生成注意力的值
- $\text{Base}$ 作为基础表示，类似于残差连接的主路径

#### 2.2 门控机制

门控信号通过以下方式计算：
\begin{equation}
Z = \phi(U) \in \mathbb{R}^{n \times e} \tag{6}
\end{equation}

其中 $\phi$ 通常是 Swish 激活函数或 GELU：
\begin{equation}
\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}} \tag{7}
\end{equation}

**推导分析**：为什么使用 Swish？
计算 Swish 的梯度：
\begin{equation}
\frac{d\text{Swish}(x)}{dx} = \sigma(\beta x) + \beta x \sigma(\beta x)(1-\sigma(\beta x)) \tag{8}
\end{equation}

这个梯度形式在 $x>0$ 时近似为1（类似ReLU），但在 $x<0$ 时有小的非零值，避免了"神经元死亡"问题。

#### 2.3 注意力分数计算

GAU使用单头注意力，查询和键都来自同一个矩阵：
\begin{equation}
Q = K = \gamma(U) \in \mathbb{R}^{n \times s} \tag{9}
\end{equation}

其中 $\gamma$ 是归一化函数（如RMSNorm），$s$ 是注意力维度。

**RMSNorm推导**：
\begin{equation}
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \tag{10}
\end{equation}

RMSNorm的梯度为：
\begin{equation}
\frac{\partial \text{RMSNorm}(x_i)}{\partial x_j} = \frac{1}{\text{RMS}(x)}\left(\delta_{ij} - \frac{x_i x_j}{\text{RMS}(x)^2}\right) \tag{11}
\end{equation}

注意力矩阵计算：
\begin{equation}
A_{raw} = QK^{\top} = \gamma(U)\gamma(U)^{\top} \in \mathbb{R}^{n \times n} \tag{12}
\end{equation}

#### 2.4 相对位置编码

GAU采用RoPE（Rotary Position Embedding）：
\begin{equation}
\text{RoPE}(x, m) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_2) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \otimes \begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_2) \\ \vdots \end{pmatrix} \tag{13}
\end{equation}

其中频率定义为：
\begin{equation}
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, \frac{d}{2}-1 \tag{14}
\end{equation}

**RoPE性质推导**：
对于位置 $m$ 和 $n$ 的两个向量，其内积满足：
\begin{equation}
\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \text{Re}\left(\sum_{i=1}^{d/2} (q_{2i-1} + iq_{2i})(k_{2i-1} - ik_{2i})e^{i(m-n)\theta_i}\right) \tag{15}
\end{equation}

这意味着内积只依赖于相对位置 $m-n$，具有平移不变性。

#### 2.5 熵不变性Softmax

这是GAU-α的关键创新，标准Softmax为：
\begin{equation}
a_{ij} = \frac{e^{s_{ij}}}{\sum_{k=1}^n e^{s_{ik}}} \tag{16}
\end{equation}

熵不变性Softmax引入对数缩放：
\begin{equation}
a_{ij} = \frac{e^{\lambda(n) s_{ij}}}{\sum_{k=1}^n e^{\lambda(n) s_{ik}}} \tag{17}
\end{equation}

其中缩放因子：
\begin{equation}
\lambda(n) = \frac{\log n}{\log 512} \tag{18}
\end{equation}

**熵不变性推导**：
Shannon熵定义为：
\begin{equation}
H = -\sum_{j=1}^n a_{ij} \log a_{ij} \tag{19}
\end{equation}

代入式(17)：
\begin{equation}
H = -\sum_{j=1}^n a_{ij}\left(\lambda s_{ij} - \log\sum_k e^{\lambda s_{ik}}\right) = \log\sum_k e^{\lambda s_{ik}} - \lambda\sum_j a_{ij}s_{ij} \tag{20}
\end{equation}

假设 $s_{ij}$ 是独立同分布的随机变量，使用对数求和指数（LSE）的性质：
\begin{equation}
\log\sum_{k=1}^n e^{\lambda s_k} \approx \log n + \lambda \max_k s_k \quad \text{(当 } \lambda \text{ 足够大时)} \tag{21}
\end{equation}

更精确地，使用拉普拉斯近似：
\begin{equation}
\log\sum_{k=1}^n e^{\lambda s_k} \approx \log n + \mathbb{E}[\lambda s] + \frac{\lambda^2}{2}\text{Var}[s] \tag{22}
\end{equation}

为了使熵 $H$ 对 $n$ 不敏感，需要 $\log n$ 项被抵消，即：
\begin{equation}
\lambda \propto \log n \tag{23}
\end{equation}

#### 2.6 完整的注意力输出

结合位置编码和熵不变性Softmax：
\begin{equation}
A = \text{softmax}\left(\lambda(n) \frac{\text{RoPE}(Q)\text{RoPE}(K)^{\top}}{\sqrt{s}}\right) \in \mathbb{R}^{n \times n} \tag{24}
\end{equation}

注意力值：
\begin{equation}
O = AV \in \mathbb{R}^{n \times e} \tag{25}
\end{equation}

#### 2.7 门控融合

GAU的最终输出结合了门控和注意力：
\begin{equation}
Y = (Z \odot O + \text{Base})W_O \tag{26}
\end{equation}

**展开推导**：
\begin{equation}
Y_i = \sum_{j=1}^e \left(\sum_{k=1}^n Z_{ik} \cdot A_{ik} \cdot V_{kj} + \text{Base}_{ij}\right) W_{O,j\ell} \tag{27}
\end{equation}

其中 $Z \odot O$ 是逐元素门控，$\text{Base}$ 提供直接路径。

### 3. 参数量和计算量分析

#### 3.1 参数量计算

GAU单层的参数包括：
- $W_U, b_U$: $(d \times e) + e = de + e$
- $W_V, b_V$: $(d \times e) + e = de + e$
- $W_{\text{base}}, b_{\text{base}}$: $(d \times e) + e = de + e$
- $W_O$: $e \times d = ed$

总参数量：
\begin{equation}
P_{GAU} = 3(de + e) + ed = 4de + 3e \tag{28}
\end{equation}

取 $e = 2d$：
\begin{equation}
P_{GAU} = 4d \cdot 2d + 3 \cdot 2d = 8d^2 + 6d \approx 8d^2 \tag{29}
\end{equation}

**对比标准Transformer**：
标准Attention+FFN的参数量：
- Attention (Q, K, V, O): $4(d \times d_k) + d_k \times d = 4dd_k + d_kd = 5dd_k$（取$d_k=d$）
- FFN: $d \times 4d + 4d \times d = 8d^2$

总计：
\begin{equation}
P_{\text{Transformer}} = 5d^2 + 8d^2 = 13d^2 \tag{30}
\end{equation}

两层GAU的参数量：
\begin{equation}
P_{2\times GAU} = 2 \times 8d^2 = 16d^2 \tag{31}
\end{equation}

相比标准Transformer，参数量比例：
\begin{equation}
\frac{P_{2\times GAU}}{P_{\text{Transformer}}} = \frac{16d^2}{13d^2} \approx 1.23 \tag{32}
\end{equation}

**结论**：两层GAU参数量略多于Attention+FFN，但由于去除了多头机制和某些归一化层，实际可比。

#### 3.2 计算量分析（FLOPs）

单个GAU的前向传播FLOPs：

**步骤1**：线性变换 $U, V, \text{Base}$
\begin{equation}
\text{FLOPs}_1 = 3 \times (2n \times d \times e) = 6nde \tag{33}
\end{equation}

**步骤2**：激活函数（忽略，相对较小）

**步骤3**：注意力矩阵 $QK^{\top}$
\begin{equation}
\text{FLOPs}_2 = 2n^2s \tag{34}
\end{equation}

**步骤4**：Softmax（忽略，相对较小）

**步骤5**：注意力值 $AV$
\begin{equation}
\text{FLOPs}_3 = 2n^2e \tag{35}
\end{equation}

**步骤6**：门控 $Z \odot O$（逐元素，忽略）

**步骤7**：输出投影 $W_O$
\begin{equation}
\text{FLOPs}_4 = 2ned \tag{36}
\end{equation}

总FLOPs（取 $e=2d, s=d/2$）：
\begin{equation}
\text{FLOPs}_{GAU} = 6n \cdot d \cdot 2d + 2n^2 \cdot \frac{d}{2} + 2n^2 \cdot 2d + 2n \cdot 2d \cdot d \tag{37}
\end{equation}
\begin{equation}
= 12nd^2 + n^2d + 4n^2d + 4nd^2 = 16nd^2 + 5n^2d \tag{38}
\end{equation}

**对比标准Attention**：
\begin{equation}
\text{FLOPs}_{\text{Attention}} = 4nd^2 + 2n^2d + 2n^2d + 2nd^2 = 6nd^2 + 4n^2d \tag{39}
\end{equation}

FFN的FLOPs：
\begin{equation}
\text{FLOPs}_{\text{FFN}} = 2n \cdot d \cdot 4d + 2n \cdot 4d \cdot d = 16nd^2 \tag{40}
\end{equation}

标准Transformer总计：
\begin{equation}
\text{FLOPs}_{\text{Transformer}} = 22nd^2 + 4n^2d \tag{41}
\end{equation}

两层GAU：
\begin{equation}
\text{FLOPs}_{2\times GAU} = 2(16nd^2 + 5n^2d) = 32nd^2 + 10n^2d \tag{42}
\end{equation}

**复杂度对比**：
- 当 $n \ll d$ 时，$\text{FLOPs}_{2\times GAU} \approx 32nd^2$ vs $\text{FLOPs}_{\text{Transformer}} \approx 22nd^2$，GAU约慢1.45倍
- 当 $n \approx d$ 时，两者相当
- 当 $n \gg d$ 时，GAU由于 $10n^2d$ vs $4n^2d$ 的优势，在长序列上更优

#### 3.3 显存占用分析

**激活值显存**（需要保存用于反向传播）：
- $U, V, \text{Base}$: $3ne$
- $Z$: $ne$
- $Q, K$: $2ns$
- $A$: $n^2$（注意力矩阵）
- $O$: $ne$

总激活显存：
\begin{equation}
M_{GAU} = 5ne + 2ns + n^2 \tag{43}
\end{equation}

取 $e=2d, s=d/2$：
\begin{equation}
M_{GAU} = 10nd + nd + n^2 = 11nd + n^2 \tag{44}
\end{equation}

**标准Attention**：
\begin{equation}
M_{\text{Attention}} = 4nd + n^2 \quad (\text{Q, K, V, A}) \tag{45}
\end{equation}

FFN显存（假设 $d_{\text{ffn}}=4d$）：
\begin{equation}
M_{\text{FFN}} = 4nd \tag{46}
\end{equation}

总计：
\begin{equation}
M_{\text{Transformer}} = 8nd + n^2 \tag{47}
\end{equation}

两层GAU：
\begin{equation}
M_{2\times GAU} = 22nd + 2n^2 \tag{48}
\end{equation}

显存比例：
\begin{equation}
\frac{M_{2\times GAU}}{M_{\text{Transformer}}} = \frac{22nd + 2n^2}{8nd + n^2} \tag{49}
\end{equation}

当 $n$ 较大时，$\frac{M_{2\times GAU}}{M_{\text{Transformer}}} \to 2$，显存约2倍。

**优化**：通过梯度检查点（Gradient Checkpointing）可以权衡计算和显存：
- 不保存中间激活 $U, V, Z$ 等
- 反向传播时重新计算
- 显存减少至：$M_{GAU}^{\text{opt}} \approx n^2 + 2nd$

### 4. 训练技巧和稳定性

#### 4.1 权重初始化

GAU使用修改的Xavier初始化。对于权重矩阵 $W \in \mathbb{R}^{m \times n}$：
\begin{equation}
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{m + n}\right) \tag{50}
\end{equation}

**推导依据**：假设输入 $x \sim \mathcal{N}(0, \sigma_x^2)$，输出 $y = Wx$：
\begin{equation}
\text{Var}[y_i] = \text{Var}\left[\sum_{j=1}^m W_{ij}x_j\right] = \sum_{j=1}^m \text{Var}[W_{ij}]\text{Var}[x_j] = m \cdot \frac{2}{m+n} \cdot \sigma_x^2 \tag{51}
\end{equation}

当 $m=n$ 时，$\text{Var}[y_i] = \sigma_x^2$，保持方差稳定。

**深层调整**：对于第 $\ell$ 层，缩放初始化：
\begin{equation}
W^{(\ell)} \sim \mathcal{N}\left(0, \frac{2}{(m+n)\sqrt{\ell}}\right) \tag{52}
\end{equation}

这基于以下观察：深层网络中，梯度会随层数累积，除以 $\sqrt{\ell}$ 可以稳定训练。

#### 4.2 学习率调度

GAU-α使用分段线性衰减，定义为：
\begin{equation}
\eta(t) = \begin{cases}
\eta_{\max} & t \leq t_0 \\
\eta_{\max} \cdot \frac{T - t}{T - t_0} & t_0 < t \leq T \\
0 & t > T
\end{cases} \tag{53}
\end{equation}

其中 $\eta_{\max}$ 是最大学习率，$t_0$ 是开始衰减的步数，$T$ 是总步数。

**与Warmup对比**：标准Warmup为：
\begin{equation}
\eta_{\text{warmup}}(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{t_{\text{warmup}}} & t \leq t_{\text{warmup}} \\
\eta_{\max} & t > t_{\text{warmup}}
\end{cases} \tag{54}
\end{equation}

GAU-α由于更好的初始化，可以省略Warmup直接使用恒定学习率再衰减。

#### 4.3 优化器：LAMB

LAMB（Layer-wise Adaptive Moments optimizer for Batch training）是Adam的扩展：
\begin{equation}
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \tag{55}
\end{equation}
\begin{equation}
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \tag{56}
\end{equation}
\begin{equation}
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \tag{57}
\end{equation}

Adam更新：
\begin{equation}
\theta_t^{\text{Adam}} = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \tag{58}
\end{equation}

LAMB额外的层归一化：
\begin{equation}
r_1 = \|\theta_{t-1}\|_2, \quad r_2 = \left\|\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\right\|_2 \tag{59}
\end{equation}
\begin{equation}
\theta_t^{\text{LAMB}} = \theta_{t-1} - \eta \frac{r_1}{r_2} \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \tag{60}
\end{equation}

**推导意义**：比值 $\frac{r_1}{r_2}$ 使得更新步长相对于参数范数进行调整，防止大参数层更新过快。

**收敛性分析**：在凸优化设置下，LAMB的收敛率为：
\begin{equation}
\mathbb{E}[f(\theta_T) - f(\theta^*)] \leq \mathcal{O}\left(\frac{1}{\sqrt{T}}\right) \tag{61}
\end{equation}

#### 4.4 梯度裁剪

全局梯度范数裁剪：
\begin{equation}
g_{\text{clip}} = \begin{cases}
g & \|g\|_2 \leq \tau \\
\frac{\tau}{\|g\|_2} g & \|g\|_2 > \tau
\end{cases} \tag{62}
\end{equation}

其中 $\tau$ 是裁剪阈值（通常取1.0）。

**梯度爆炸分析**：在深度为 $L$ 的网络中，梯度传播满足：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \theta^{(1)}} = \frac{\partial \mathcal{L}}{\partial \theta^{(L)}} \prod_{\ell=1}^{L-1} \frac{\partial \theta^{(\ell+1)}}{\partial \theta^{(\ell)}} \tag{63}
\end{equation}

若 $\left\|\frac{\partial \theta^{(\ell+1)}}{\partial \theta^{(\ell)}}\right\| > 1$，则梯度指数增长。裁剪确保：
\begin{equation}
\|g_{\text{clip}}\|_2 \leq \tau \tag{64}
\end{equation}

#### 4.5 Post Norm vs Pre Norm

GAU-α使用Post Norm结构：
\begin{equation}
X_{\ell+1} = \text{Norm}(X_{\ell} + \text{GAU}(X_{\ell})) \tag{65}
\end{equation}

对比Pre Norm：
\begin{equation}
X_{\ell+1}^{\text{Pre}} = X_{\ell} + \text{GAU}(\text{Norm}(X_{\ell})) \tag{66}
\end{equation}

**梯度流分析**：Post Norm的梯度：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial X_{\ell}} = \frac{\partial \mathcal{L}}{\partial X_{\ell+1}} \frac{\partial \text{Norm}(X_{\ell} + \text{GAU}(X_{\ell}))}{\partial X_{\ell}} \tag{67}
\end{equation}

Pre Norm的梯度：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial X_{\ell}} = \frac{\partial \mathcal{L}}{\partial X_{\ell+1}} \left(I + \frac{\partial \text{GAU}(\text{Norm}(X_{\ell}))}{\partial X_{\ell}}\right) \tag{68}
\end{equation}

Pre Norm由于 $I$ 的存在，梯度更容易直接传播，但Post Norm在配合好的初始化后，训练更稳定且效果更好。

#### 4.6 RMS Norm细节

RMSNorm相比LayerNorm省略了减均值操作：
\begin{equation}
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta \tag{69}
\end{equation}
\begin{equation}
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma \tag{70}
\end{equation}

其中 $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$。

**计算优势**：
- LayerNorm需要两遍扫描（一遍计算均值，一遍计算方差）
- RMSNorm只需一遍扫描计算平方和

**理论分析**：对于零均值输入（通过前一层归一化保证），有：
\begin{equation}
\text{Var}[x] = \mathbb{E}[x^2] - (\mathbb{E}[x])^2 \approx \mathbb{E}[x^2] = \text{RMS}(x)^2 \tag{71}
\end{equation}

因此RMSNorm近似LayerNorm，但计算更快。

**参数量对比**：
- LayerNorm: $2d$ 参数 ($\gamma, \beta$)
- RMSNorm: $d$ 参数 (仅 $\gamma$)

### 5. 熵不变性Softmax详细推导

#### 5.1 问题设定

给定注意力分数 $s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}$，标准Softmax：
\begin{equation}
a_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^n \exp(s_{ik})} \tag{72}
\end{equation}

熵定义：
\begin{equation}
H_i = -\sum_{j=1}^n a_{ij} \log a_{ij} \tag{73}
\end{equation}

**问题**：当序列长度 $n$ 增加时，$H_i$ 也会增加，导致注意力更分散。

#### 5.2 熵的期望值

展开熵的表达式：
\begin{equation}
H_i = -\sum_{j=1}^n a_{ij} \left(s_{ij} - \log \sum_k e^{s_{ik}}\right) = \log \sum_k e^{s_{ik}} - \sum_j a_{ij} s_{ij} \tag{74}
\end{equation}

定义配分函数：
\begin{equation}
Z_i = \sum_{k=1}^n e^{s_{ik}} \tag{75}
\end{equation}

则：
\begin{equation}
H_i = \log Z_i - \mathbb{E}_{j \sim a_i}[s_{ij}] \tag{76}
\end{equation}

#### 5.3 独立同分布假设

假设 $s_{ij}$ 独立同分布，服从某分布 $p(s)$。则：
\begin{equation}
\mathbb{E}[\log Z_i] = \mathbb{E}\left[\log \sum_{k=1}^n e^{s_k}\right] \tag{77}
\end{equation}

使用Jensen不等式：
\begin{equation}
\mathbb{E}[\log Z] = \mathbb{E}\left[\log \sum_{k=1}^n e^{s_k}\right] \geq \log \mathbb{E}\left[\sum_{k=1}^n e^{s_k}\right] = \log(n \mathbb{E}[e^s]) \tag{78}
\end{equation}

更精确的估计使用鞍点近似：
\begin{equation}
\log \sum_{k=1}^n e^{s_k} \approx \log n + \log \mathbb{E}[e^s] + \mathcal{O}(1/n) \tag{79}
\end{equation}

#### 5.4 高斯假设下的推导

假设 $s_{ij} \sim \mathcal{N}(\mu, \sigma^2)$，则：
\begin{equation}
\mathbb{E}[e^s] = e^{\mu + \sigma^2/2} \tag{80}
\end{equation}

因此：
\begin{equation}
\mathbb{E}[H_i] \approx \log n + \mu + \frac{\sigma^2}{2} - \mathbb{E}[\mathbb{E}_{j \sim a_i}[s_{ij}]] \tag{81}
\end{equation}

假设注意力集中在top-k个位置，则第二项约为：
\begin{equation}
\mathbb{E}[\mathbb{E}_{j \sim a_i}[s_{ij}]] \approx \mu + c\sigma \tag{82}
\end{equation}

其中 $c$ 是常数（约为1）。代入得：
\begin{equation}
\mathbb{E}[H_i] \approx \log n + \frac{\sigma^2}{2} - c\sigma \tag{83}
\end{equation}

**关键观察**：熵的主要 $n$ 依赖项是 $\log n$。

#### 5.5 缩放因子设计

为了抵消 $\log n$ 的影响，引入缩放因子 $\lambda(n)$：
\begin{equation}
a_{ij}^{\text{new}} = \frac{\exp(\lambda(n) s_{ij})}{\sum_k \exp(\lambda(n) s_{ik})} \tag{84}
\end{equation}

新的熵：
\begin{equation}
H_i^{\text{new}} = \log \sum_k e^{\lambda s_k} - \lambda \mathbb{E}_{j \sim a_i^{\text{new}}}[s_{ij}] \tag{85}
\end{equation}

使用前面的近似：
\begin{equation}
\log \sum_k e^{\lambda s_k} \approx \log n + \lambda\mu + \frac{\lambda^2\sigma^2}{2} \tag{86}
\end{equation}

第二项：
\begin{equation}
\lambda \mathbb{E}_{j \sim a_i^{\text{new}}}[s_{ij}] \approx \lambda(\mu + c\sigma\sqrt{\lambda}) \tag{87}
\end{equation}

其中 $\sqrt{\lambda}$ 来自于softmax在缩放后的锐化效应。

代入得：
\begin{equation}
H_i^{\text{new}} \approx \log n + \lambda\mu + \frac{\lambda^2\sigma^2}{2} - \lambda\mu - c\sigma\lambda^{3/2} \tag{88}
\end{equation}
\begin{equation}
= \log n + \frac{\lambda^2\sigma^2}{2} - c\sigma\lambda^{3/2} \tag{89}
\end{equation}

为了使 $H_i^{\text{new}}$ 对 $n$ 不敏感，需要：
\begin{equation}
\frac{\partial H_i^{\text{new}}}{\partial n} \approx 0 \tag{90}
\end{equation}

即：
\begin{equation}
\frac{1}{n} + \left(\lambda\sigma^2 - \frac{3c\sigma}{2}\lambda^{1/2}\right)\frac{d\lambda}{dn} \approx 0 \tag{91}
\end{equation}

若 $\lambda$ 主导项与 $\log n$ 成正比，设 $\lambda = \alpha \log n$：
\begin{equation}
\frac{d\lambda}{dn} = \frac{\alpha}{n} \tag{92}
\end{equation}

代入：
\begin{equation}
\frac{1}{n} + \frac{\alpha}{n}\left(\alpha\sigma^2\log n - \frac{3c\sigma}{2}\sqrt{\alpha\log n}\right) \approx 0 \tag{93}
\end{equation}

当 $n$ 足够大时，第一项可忽略，得：
\begin{equation}
\alpha\sigma^2\log n \approx \frac{3c\sigma}{2}\sqrt{\alpha\log n} \tag{94}
\end{equation}

解得：
\begin{equation}
\alpha \approx \frac{9c^2}{4\sigma^2 \log n} \cdot \log n = \frac{9c^2}{4\sigma^2} \tag{95}
\end{equation}

**实践选择**：归一化后 $\sigma \approx 1$，$c \approx 1$，因此：
\begin{equation}
\lambda(n) = \kappa \log n \tag{96}
\end{equation}

其中 $\kappa$ 是可调超参数，实验中取 $\kappa = \frac{1}{\log 512}$ 使得 $n=512$ 时退化为标准Softmax。

#### 5.6 信息论解释

从信息论角度，熵 $H$ 度量分布的"不确定性"或"信息量"。

**互信息**：注意力机制可视为 $Q$ 和 $K$ 之间的信息传递，互信息：
\begin{equation}
I(Q; K) = H(K) - H(K|Q) \tag{97}
\end{equation}

其中：
\begin{equation}
H(K|Q=q_i) = H_i = -\sum_j a_{ij} \log a_{ij} \tag{98}
\end{equation}

当 $n$ 增加时，$H(K)$ 增加（更多选择），但我们希望 $H(K|Q)$ 保持不变（给定查询后，关键token的不确定性不变），从而：
\begin{equation}
I(Q; K) \propto \log n \tag{99}
\end{equation}

互信息随 $n$ 增长，这是合理的（更多token提供更多信息）。

### 6. 与标准Attention的对比

#### 6.1 公式对比

| 方面 | 标准Attention | GAU-α |
|------|--------------|-------|
| 查询/键 | $Q \neq K$ | $Q = K$ |
| 多头 | 是 | 否（单头） |
| 门控 | 无 | 有 ($Z \odot O$) |
| 缩放 | $1/\sqrt{d}$ | $\frac{\log n}{\sqrt{d}\log 512}$ |
| 归一化 | LayerNorm | RMSNorm |
| 位置编码 | 可选 | RoPE |

#### 6.2 性能对比推导

**标准Attention的有效秩**：
注意力矩阵 $A$ 的秩最多为 $\min(n, d)$，但实际有效秩由奇异值分布决定：
\begin{equation}
r_{\text{eff}} = \frac{(\sum_i \sigma_i)^2}{\sum_i \sigma_i^2} \tag{100}
\end{equation}

**GAU的有效秩**：由于门控机制，GAU的输出可以表示为：
\begin{equation}
Y = Z \odot (AV) + \text{Base} \tag{101}
\end{equation}

有效秩增加到：
\begin{equation}
r_{\text{eff}}^{\text{GAU}} \geq r_{\text{eff}}^{\text{Attention}} \tag{102}
\end{equation}

因为 $Z \odot (AV)$ 允许逐位置的不同缩放，增加了表达能力。

#### 6.3 长度外推性对比

**标准Attention**：设训练长度 $n_{\text{train}} = 512$，测试长度 $n_{\text{test}} = 1024$。

熵变化：
\begin{equation}
\Delta H = H(n_{\text{test}}) - H(n_{\text{train}}) \approx \log \frac{n_{\text{test}}}{n_{\text{train}}} = \log 2 \approx 0.693 \tag{103}
\end{equation}

这意味着注意力更分散，性能下降。

**GAU-α**：使用 $\lambda(n) = \frac{\log n}{\log 512}$：
\begin{equation}
H^{\text{GAU}}(n) \approx \log n + C - \lambda(n) \cdot f(n) \tag{104}
\end{equation}

其中 $f(n)$ 是关于 $n$ 缓慢变化的函数。代入 $\lambda(n)$：
\begin{equation}
H^{\text{GAU}}(n) \approx \log n + C - \frac{\log n}{\log 512} \cdot f(n) \approx \text{const} \tag{105}
\end{equation}

因此熵基本不变，长度外推性更好。

**实验验证**（论文数据）：
\begin{equation}
\text{Accuracy}_{\text{Attention-O}}(n=256) = 23.02\% \tag{106}
\end{equation}
\begin{equation}
\text{Accuracy}_{\text{Attention-E}}(n=256) = 34.04\% \tag{107}
\end{equation}

提升：
\begin{equation}
\Delta = \frac{34.04 - 23.02}{23.02} \approx 47.8\% \tag{108}
\end{equation}

### 7. 高级话题

#### 7.1 GAU的梯度流分析

考虑损失 $\mathcal{L}$ 对输入 $X$ 的梯度：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Y} \frac{\partial Y}{\partial X} \tag{109}
\end{equation}

展开：
\begin{equation}
\frac{\partial Y}{\partial X} = W_O^{\top} \frac{\partial}{\partial X}(Z \odot O + \text{Base}) \tag{110}
\end{equation}

包含三条路径：
1. **门控路径**：$\frac{\partial Z}{\partial X} = \frac{\partial \phi(U)}{\partial U} \frac{\partial U}{\partial X}$
2. **注意力路径**：$\frac{\partial O}{\partial X} = \frac{\partial (AV)}{\partial X}$
3. **残差路径**：$\frac{\partial \text{Base}}{\partial X} = W_{\text{base}}$

总梯度：
\begin{equation}
\frac{\partial \mathcal{L}}{\partial X} = W_O^{\top}\left(O \odot \frac{\partial Z}{\partial X} + Z \odot \frac{\partial O}{\partial X} + W_{\text{base}}\right) \tag{111}
\end{equation}

**梯度范数估计**：假设各项独立：
\begin{equation}
\mathbb{E}\left[\left\|\frac{\partial \mathcal{L}}{\partial X}\right\|^2\right] \approx \|W_O\|^2 \left(\|O\|^2\|\nabla Z\|^2 + \|Z\|^2\|\nabla O\|^2 + \|W_{\text{base}}\|^2\right) \tag{112}
\end{equation}

由于 $Z, O, \text{Base}$ 都有 $\mathcal{O}(ne)$ 的元素，梯度范数为 $\mathcal{O}(\sqrt{ne})$，与标准Attention的 $\mathcal{O}(\sqrt{nd})$ 相当（$e \approx 2d$）。

#### 7.2 理论收敛性

**假设**：损失函数 $\mathcal{L}$ 是 $L$-光滑的，即：
\begin{equation}
\|\nabla \mathcal{L}(\theta_1) - \nabla \mathcal{L}(\theta_2)\| \leq L\|\theta_1 - \theta_2\| \tag{113}
\end{equation}

使用LAMB优化器，学习率 $\eta < \frac{1}{L}$，经过 $T$ 步后：
\begin{equation}
\min_{t \leq T} \mathbb{E}[\|\nabla \mathcal{L}(\theta_t)\|^2] \leq \frac{2(\mathcal{L}(\theta_0) - \mathcal{L}^*)}{\eta T} + \eta L \sigma^2 \tag{114}
\end{equation}

其中 $\sigma^2$ 是梯度方差。

**GAU的优势**：由于门控机制提供了更平滑的损失面（实验观察），$L$ 更小，允许更大的学习率。

#### 7.3 表达能力分析

**定理（非正式）**：两层GAU可以近似任意Attention+FFN组合。

**证明思路**：
1. FFN可以表示为：$\text{FFN}(x) = W_2 \sigma(W_1 x)$
2. GAU的 $Z \odot O$ 可以模拟门控FFN：$Z$ 对应 $\sigma(W_1 x)$，$O$ 对应 $W_2$
3. Attention部分直接对应
4. 通过两层GAU，可以分别处理Attention和FFN功能

**通用逼近**：GAU作为非线性变换，满足通用逼近定理的条件（包含非线性激活和足够宽度）。

### 8. 实现细节和优化

#### 8.1 融合算子

**Softmax融合**：将缩放、指数、求和、除法融合为单个CUDA kernel：
\begin{equation}
\text{FusedSoftmax}(s, \lambda, n) = \frac{\exp(\lambda s)}{\sum_k \exp(\lambda s_k)} \tag{115}
\end{equation}

**加速比**：通过减少内存访问，融合算子可达到 $2\times$ 加速。

#### 8.2 混合精度训练

使用FP16存储权重和激活，但累积梯度用FP32：
\begin{equation}
\theta_t^{\text{FP32}} = \theta_{t-1}^{\text{FP32}} - \eta \cdot \text{FP32}(\nabla \mathcal{L}(\text{FP16}(\theta_{t-1}))) \tag{116}
\end{equation}

**动态缩放**：为避免下溢，梯度乘以缩放因子 $s$：
\begin{equation}
g_{\text{scaled}} = s \cdot g \tag{117}
\end{equation}

更新时除以 $s$：
\begin{equation}
\theta_t = \theta_{t-1} - \eta \frac{g_{\text{scaled}}}{s} \tag{118}
\end{equation}

$s$ 动态调整：若梯度溢出（出现inf/nan），则 $s \leftarrow s/2$；若连续1000步无溢出，则 $s \leftarrow 2s$。

#### 8.3 分布式训练

**数据并行**：将batch分割到 $N$ 个GPU，每个GPU计算局部梯度 $g_i$，然后AllReduce求平均：
\begin{equation}
g = \frac{1}{N}\sum_{i=1}^N g_i \tag{119}
\end{equation}

**梯度累积**：当GPU显存不足时，累积 $K$ 个mini-batch的梯度：
\begin{equation}
g_{\text{accum}} = \frac{1}{K}\sum_{k=1}^K g_k \tag{120}
\end{equation}

有效batch size：$B_{\text{eff}} = N \times K \times B_{\text{local}}$

### 9. 总结

GAU-α通过以下创新实现了"快好省"：
1. **门控机制**（式26）：增强表达能力
2. **熵不变性Softmax**（式17-18）：改善长度外推
3. **单头设计**：减少参数和计算
4. **优化的归一化**（式70）：提高效率
5. **更好的初始化**（式52）：稳定训练

这些设计共同作用，使GAU-α成为高效的Attention替代方案。

