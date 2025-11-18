---
title: BytePiece：更纯粹、更高压缩率的Tokenizer
slug: bytepiece更纯粹更高压缩率的tokenizer
date: 2023-09-07
tags: 最小熵, 分词, 无监督, 新词发现, 生成模型
status: pending
---

# BytePiece：更纯粹、更高压缩率的Tokenizer

**原文链接**: [https://spaces.ac.cn/archives/9752](https://spaces.ac.cn/archives/9752)

**发布日期**: 

---

目前在LLM中最流行的Tokenizer（分词器）应该是Google的[SentencePiece](https://github.com/google/sentencepiece)了，因为它符合Tokenizer的一些理想特性，比如语言无关、数据驱动等，并且由于它是C++写的，所以Tokenize（分词）的速度很快，非常适合追求效率的场景。然而，它也有一些明显的缺点，比如训练速度慢（BPE算法）、占用内存大等，同时也正因为它是C++写的，对于多数用户来说它就是黑箱，也不方便研究和二次开发。

事实上，Tokenizer的训练就相当于以往的“新词发现”，而笔者之前也写过[中文分词](/search/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E7%B3%BB%E5%88%97/)和[最小熵](/tag/%E6%9C%80%E5%B0%8F%E7%86%B5/)系列文章，对新词发现也有一定的积累，所以很早之前就有自己写一版Tokenizer的想法。这几天总算腾出了时间初步完成了这件事情，东施效颦SentencePiece，命名为“BytePiece”。

> **Github：<https://github.com/bojone/bytepiece>**

## 理想特性 #

既然要重写Tokenizer，那么我们就要思考一个理想的Tokenizer应该是怎样的，这样才能判断最终是否达到了预期。照笔者看来，Tokenizer至少应该具备如下基本特性：

> 1、**无损重构** ：分词结果应该可以无损还原为输入；
> 
> 2、**高压缩率** ：词表大小相同时，同一批数据的tokens数应该尽可能少；
> 
> 3、**语言无关** ：基于统计，训练和分词过程都不应引入语言特性；
> 
> 4、**数据驱动** ：可以直接基于原始语料进行无监督训练；
> 
> 5、**训练友好** ：能够在合理的时间和配置上完成训练过程。

最后，还有一些加分项，比如分词速度快、代码易读、方便二次拓展等，这些满足自然最好，但笔者认为可以不列入基本特性里边。

对于笔者来说，SentencePiece最大的槽点就是“无损重构”和“训练友好”。首先，SentencePiece默认会进行[NFKC normalization](https://github.com/google/sentencepiece/blob/master/doc/normalization.md)，这会导致“全角逗号转半角逗号”等不可逆变化，所以默认情况下它连“无损重构”都不满足，所以很长时间里它都不在笔者的候选名单中，直到后来发现，在训练时添加参数`--normalization_rule_name=identity`就可以让它不做任何转换。所以SentencePiece算是支持无损重构，只不过要特别设置。

至于训练方面，就更让人抓狂了。SentencePiece支持BPE和Unigram两种主流算法，Unigram训练速度尚可，但压缩率会稍低一些，BPE的压缩率更高，但是训练速度要比Unigram慢上一个数量级！而且不管是BPE还是Unigram，训练过程都极费内存。总而言之，用较大的语料去训练一个SentencePiece模型真不是一种好的体验。

## 模型构思 #

一个新Tokenizer的构建，可以分解为三个部分：1、基本单元；2、分词算法；3、训练算法。确定这三个部分后，剩下的就只是编程技巧问题了。下面逐一介绍BytePiece对这些问题的思考。

### 基本单元 #

我们知道，Python3的默认字符串类型是Unicode，如果以Unicode为基本单位，我们称之为Char-based。Char-based很直观方便，汉字表现为长度为1的单个字符，但不同语言的Char实在太多，即便只是覆盖单字都需要消耗非常大的vocab_size，更不用说引入Word。所以BytePiece跟主流的Tokenizer一样，以Byte为基本单位。

回到Byte之后，很多问题都“豁然开朗”了。因为不同的单Byte只有256个，所以只要词表里包含了这256个单Byte，那么就可以杜绝OOV（Out of Vocabulary），这是它显而易见的好处。此外，我们知道汉字的平均信息熵要比英文字母的平均信息熵要大，如果我们选择Char-based，那么虽然每个Char表面看起来长度都是1，但“内在”的颗粒度不一样，这会导致统计结果有所偏置。相比之下，每个Byte的信息熵则更加均匀【比如，大部分汉字的UTF-8编码对应3个Byte，而汉字的平均信息熵正好是英文字母（对应一个Byte）的2～3倍左右】，因此用Byte的统计结果会更加无偏，这将会使得模型更加“语言无关”。

在Byte-based方面，BytePiece比SentencePiece更彻底，SentencePiece是先以Char-based进行处理，然后遇到OOV再以Byte-based处理，BytePiece则是在一开始就将文本通过`text.encode()`转为Bytes，然后才进行后续操作，相比之下更加纯粹。

### 分词算法 #

基于词典进行分词的算法无非就那几种，比如最大匹配、最短路径、最大概率路径等，有兴趣追溯的读者，可以参考Matrix67之前写的[《漫话中文自动分词和语义识别（上）：中文分词算法》](http://www.matrix67.com/blog/archives/4212)，

跟jieba等中文分词工具一样，BytePiece选择的是最大概率路径分词，也称“一元文法模型”，即Unigram。选择Unigram有三方面的考虑：第一，Unigram的最大概率换言之就是最大似然，而LLM的训练目标也是最大似然，两者更加一致；第二，从压缩的角度看，最大概率实际上就是最短编码长度（也叫最小描述长度），是压缩率最大化的体现，这也跟“压缩就是智能”的信仰一致；第三，Unigram求最优分词方案可以通过Viterbi算法在线性复杂度内完成，这是理论最优的复杂度了。

当然，既然有“一元文法模型”，自然也有更复杂的“二元文法模型”、“三元文法模型”等，但它们的复杂度增加远大于它能带来的收益，所以我们通常不考虑这些高阶模型。

### 训练算法 #

之所以先讨论分词算法在讨论训练算法，是因为只有分词算法确定下来后，才能确定训练的优化目标，从而研究对应的训练算法。

开头就提到，Tokenizer的训练本质上就是以往的“新词发现”，而笔者之前也提了好几种新词发现算法，如[《基于切分的新词发现》](/archives/3913)、[《基于语言模型的无监督分词》](/archives/3956)、[《更好的新词发现算法》](/archives/4256)。现在看来，跟Unigram分词算法最契合、最有潜力的，应该是[《基于语言模型的无监督分词》](/archives/3956)，BytePiece的训练就是基于它实现的，这里称之为**Byte-based N-gram Language Model（BNLM）** 。

具体来说，对于Unigram分词，如果一个长度为$l$的字节串$c_1, c_2, \dots, c_l$，最优分词结果为$w_1, w_2, \dots, w_m$，那么概率乘积$p(w_1)p(w_2)\dots p(w_m)$应该是所有切分中最大的。设$w_1,w_2,\cdots,w_m$的长度分别为$l_1,l_2,\cdots,l_m$，那么根据条件分解公式  
\begin{equation}\prod_{i=1}^m p(w_i) = \prod_{i=1}^m \prod_{j=L_{i-1} + 1}^{j=L_{i-1} + l_i} p(c_j|c_{L_{i-1} + 1},\cdots,c_{j-1})\end{equation}  
这里$L_i=l_1+l_2+\cdots+l_i$。只考虑$n$-gram模型，将$j\gt L_{i-1} + n$的$p(c_j|c_{L_{i-1} + 1},\cdots,c_{j-1})$统一用$p(c_j|c_{j - n + 1},\cdots,c_{j-1})$近似，那么Unigram分词就转化为一个字（节）标注问题，而Tokenizer的训练则转化为$n$-gram语言模型的训练（推荐$n=6$），可以直接无监督完成。更详细的介绍请读者移步原文[《基于语言模型的无监督分词》](/archives/3956)。

（注意：$n=6$只是说BytePiece的统计信息最多到6-gram，但并非最大只能生成长度为6的piece，因为大于$6$的$n$-gram条件概率我们会用6-gram的近似，所以它是可以做到任意阶的，即理论上可以生成任意长度piece。）

## 代码实现 #

原理确定之后，剩下的就是枯燥的开发工作了。幸不辱命，勉强写出了一套可用的代码：

> **Github：<https://github.com/bojone/bytepiece>**

代码很简单，单文件，里边就`Trainer`和`Tokenizer`两个类，分别对应分词两部分。分词借助[pyahocorasick](https://github.com/WojciechMula/pyahocorasick)来构建AC自动机来稍微提了一下速，能凑合用，但还是会比SentencePiece慢不少，毕竟速度方面纯Python跟C++确实没法比。训练则分为四个主要步骤：1、$n$-gram计数；2、$n$-gram剪枝；3、预分词；4、预分词结果剪枝。其中1、3、4都是计算密集型，并且都是可并行的，所以编写了相应的多进程实现。在开足够多的进程（笔者开了64进程，每个进程的使用率基本上都是满的）下，训练速度能媲美SentencePiece的Unigram训练速度。

这里特别要提一下结果剪枝方面。剪枝最基本的依据自然是频数和vocab_size，但这还不够，因为有时候会出现$p(w_1)p(w_2) > p(w_1\circ w_2)$（$w_1\circ w_2$指两个词拼接）且$w_1,w_2,w_1\circ w_2$三个词都在词表中，这种情况下$w_1\circ w_2$这个词永远不会切分出来，所以将它放在词表中是纯粹浪费空间的，因此剪枝过程也包含了这类结果的排除。

## 效果测试 #

到了大家喜闻乐见的测试环节，是骡子是马总要拉出来遛遛。首先做个小规模的测试，从悟道之前开源的数据集里边随机采样10万条作为训练集（导出来的文件大概330MB），然后另外采样1千作为测试集，训练一个vocab_size=50k的词表，结果对比如下：  
\begin{array}{c|ccc}  
\hline  
& \text{训练时间}\downarrow & \text{最大内存占用}\downarrow & \text{压缩率}\uparrow \\\  
\hline  
\text{SP-BPE} & \text{55.3分钟} & \text{5.2GB} & 4.80 \\\  
\text{SP-Unigram} & \text{1.6分钟} & \text{2.5GB} & 4.73 \\\  
\text{BytePiece} & \text{6.5分钟} & \text{4.3GB} & 5.05 \\\  
\hline  
\end{array}  
解释一下，这里SP-BPE、SP-Unigram分别指SentencePiece的model_type设为BPE和Unigram，训练代码分别是
    
    
    spm.SentencePieceTrainer.train('--input=wudao.txt --model_prefix=wudao_m --vocab_size=50000 --model_type=bpe --train_extremely_large_corpus=true --normalization_rule_name=identity')
    
    spm.SentencePieceTrainer.train('--input=wudao.txt --model_prefix=wudao_m2 --vocab_size=50000 --model_type=unigram --train_extremely_large_corpus=true --normalization_rule_name=identity')

压缩率的单位是“bytes/token”，即平均每个token对应的字节数。可见，BytePiece能够在训练时间和内存都比较折中的情况下，获得最大的压缩率。

接下来进行一个更大规模的测试。从中英比例大致为3:5的混合语料库中，抽取出10万条样本训练vocab_size=100k的Tokenizer。这个语料库的文本都比较长，所以这时候10万条导出来的文件已经13GB了，测试集包含两部分，一部分是同样的语料库中采样出1000条（即同源），另一部分是刚才采样出来的1000条悟道数据集（代表不同源）。结果如下：  
\begin{array}{c|cccc}  
\hline  
& \text{训练时间}\downarrow & \text{最大内存占用}\downarrow & \text{压缩率(同源)}\uparrow & \text{压缩率(异源)}\uparrow \\\  
\hline  
\text{SP-BPE} & \text{19.21小时} & \text{97GB} & 4.52 & 4.46 \\\  
\text{SP-Unigram} & \text{2.02小时} & \text{384GB} & 4.51 & 4.48 \\\  
\text{BytePiece} & \text{2.24小时} & \text{51GB} & 5.39 & 4.51\\\  
\hline  
\end{array}

不管是训练时间、内存还是压缩率，看起来训练数据量越大，BytePiece越有优势！

## 未完待续 #

就目前的结果看来，BytePiece在训练方面是有一定优势的，分词效果也尚可，不过吃了纯Python的亏，分词速度只有SentencePiece的1/10左右，这是未来的一个优化方向之一，期待有C/C++大牛能参与进来，帮助提升BytePiece的分词速度。（注：从0.2.0版开始，使用Cython加速了分词函数，目前BytePiece的分词速度已经接近BPE，并且在文本足够长时能优于BPE。）

实际上，如果采用随机采样、动态剪枝等技术，BytePiece的训练速度和内存都还可以进一步优化。目前BytePiece为了保证结果的确定性，直到所有结果都统计完毕才进行剪枝，这样不管是单进程还是多进程，都能保证结果的一致性。如果随机打乱输入，并且定时进行剪枝，那么可以进一步控制内存的占用量，同时还能加快统计速度，并且可以预期对最终效果的影响也不大。这部分工作，也在后面根据用户体验进一步引入。

除了以上这些，BytePiece细节之处还有不少需要完善的地方，以及可能还有未发现的错漏之处，敬请大家海涵且反馈

## 文章小结 #

本文介绍了笔者自行开发的Tokenizer——BytePiece，它是Byte-based的Unigram分词器，纯Python实现，更加易读和易拓展。由于采用了新的训练算法，所以压缩率通常比现有tokenizer更高，同时支持多进程加速训练。此外，它直接操作文本的utf-8 bytes，几乎不进行任何的预处理，所以更加纯粹和语言无关。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9752>_

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

苏剑林. (Sep. 07, 2023). 《BytePiece：更纯粹、更高压缩率的Tokenizer 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9752>

@online{kexuefm-9752,  
title={BytePiece：更纯粹、更高压缩率的Tokenizer},  
author={苏剑林},  
year={2023},  
month={Sep},  
url={\url{https://spaces.ac.cn/archives/9752}},  
} 


---

## 公式推导与注释

### 一、信息论基础

#### 1.1 熵与压缩率的关系

**香农信息熵定义**：对于离散随机变量$X$，其信息熵定义为
\begin{equation}
H(X) = -\sum_{i} p(x_i) \log_2 p(x_i) \tag{1}
\end{equation}

**数学直觉**：熵度量了随机变量的不确定性。$\log_2 p(x_i)$表示事件$x_i$的自信息（单位：bit），负号使其为正值，期望即为平均信息量。

**压缩率定义**：假设词表大小为$V$，平均每个token对应$L$个bytes，则压缩率定义为
\begin{equation}
R = \frac{L}{\log_2 V} \tag{2}
\end{equation}

**推导过程**：
- 若每个token等概出现，需要$\log_2 V$ bits编码一个token
- 该token实际对应$L$ bytes = $8L$ bits的原始数据
- 因此压缩比为$\frac{8L}{\log_2 V}$，压缩率即为上式

**实际考虑**：token分布通常不均匀，理论最优编码长度由熵给出
\begin{equation}
H(T) = -\sum_{t \in V} p(t) \log_2 p(t) \tag{3}
\end{equation}

**香农第一定理**：对于信源$T$，最优无损压缩的平均码长不小于其熵$H(T)$，即
\begin{equation}
\mathbb{E}[L_{code}] \geq H(T) \tag{4}
\end{equation}

#### 1.2 Byte级别的信息熵

**Byte熵计算**：设byte序列为$B = b_1, b_2, \ldots, b_n$，每个byte取值范围为$[0, 255]$，则
\begin{equation}
H(B) = -\sum_{i=0}^{255} p(b=i) \log_2 p(b=i) \tag{5}
\end{equation}

**经验观察**：
- 英文字符（1 byte/char）：熵约为$4-5$ bits
- 中文字符（3 bytes/char in UTF-8）：熵约为$10-13$ bits
- 因此中文平均信息熵约为英文的2-3倍

**信息密度**：定义单位byte的信息密度为
\begin{equation}
\rho = \frac{H(T)}{L_{avg}} \tag{6}
\end{equation}
其中$L_{avg}$是token的平均byte长度。

### 二、N-gram语言模型

#### 2.1 条件概率分解

**完整概率分解**：对于byte序列$c_1, c_2, \ldots, c_l$，根据链式法则
\begin{equation}
p(c_1, c_2, \ldots, c_l) = \prod_{i=1}^{l} p(c_i | c_1, \ldots, c_{i-1}) \tag{7}
\end{equation}

**马尔可夫假设**：n-gram模型假设每个byte只依赖于前$n-1$个byte
\begin{equation}
p(c_i | c_1, \ldots, c_{i-1}) \approx p(c_i | c_{i-n+1}, \ldots, c_{i-1}) \tag{8}
\end{equation}

**推导说明**：
- 当$i \leq n$时，使用所有历史：$p(c_i | c_1, \ldots, c_{i-1})$
- 当$i > n$时，截断历史：$p(c_i | c_{i-n+1}, \ldots, c_{i-1})$

#### 2.2 词概率的n-gram分解

**关键推导**：设词$w = c_1 c_2 \ldots c_m$，其概率可分解为
\begin{equation}
p(w) = \prod_{j=1}^{m} p(c_j | c_{\max(1, j-n+1)}, \ldots, c_{j-1}) \tag{9}
\end{equation}

**具体示例**（以$n=3$, $w=$"深度学习"为例）：
\begin{align}
p(\text{"深度学习"}) &= p(\text{深}) \cdot p(\text{度}|\text{深}) \cdot p(\text{学}|\text{深度}) \cdot p(\text{习}|\text{度学}) \tag{10}\\
&= p(c_1) \cdot p(c_2|c_1) \cdot p(c_3|c_1,c_2) \cdot p(c_4|c_2,c_3) \tag{11}
\end{align}

**边界处理**：
- $j=1$: $p(c_1)$ 为unigram概率
- $j=2$: $p(c_2|c_1)$ 为bigram概率
- $j \geq n$: $p(c_j|c_{j-n+1}, \ldots, c_{j-1})$ 为完整n-gram

### 三、Unigram分词算法

#### 3.1 最大概率路径

**问题定义**：给定byte序列$B = b_1, \ldots, b_n$和词表$V$，找到最优切分$W^* = w_1, w_2, \ldots, w_k$使得
\begin{equation}
W^* = \argmax_{W} \prod_{i=1}^{k} p(w_i) \tag{12}
\end{equation}

**对数形式**（避免下溢）：
\begin{equation}
W^* = \argmax_{W} \sum_{i=1}^{k} \log p(w_i) \tag{13}
\end{equation}

**组合爆炸**：对于长度$n$的序列，可能的切分数为$2^{n-1}$（每个间隙可切可不切），暴力枚举不可行。

#### 3.2 Viterbi动态规划

**状态定义**：令$dp[i]$表示前$i$个byte的最优切分的对数概率
\begin{equation}
dp[i] = \max_{1 \leq j < i, b_{j+1:i} \in V} \left( dp[j] + \log p(b_{j+1:i}) \right) \tag{14}
\end{equation}

**初始化**：
\begin{equation}
dp[0] = 0 \tag{15}
\end{equation}

**递推公式详解**：
- 枚举所有可能的最后一个词：$b_{j+1:i}$（从位置$j+1$到$i$）
- 要求该词在词表中：$b_{j+1:i} \in V$
- 前$j$个byte的最优解为$dp[j]$
- 加上当前词的对数概率$\log p(b_{j+1:i})$

**回溯路径**：同时维护指针数组$ptr[i]$记录切分点
\begin{equation}
ptr[i] = \argmax_{j} \left( dp[j] + \log p(b_{j+1:i}) \right) \tag{16}
\end{equation}

**完整算法**：
```
输入：byte序列 B[1..n]，词表 V，词概率 p(·)
输出：最优切分 W*

1. 初始化 dp[0] = 0
2. For i = 1 to n:
     For j = 0 to i-1:
       if B[j+1:i] ∈ V:
         score = dp[j] + log p(B[j+1:i])
         if score > dp[i]:
           dp[i] = score
           ptr[i] = j
3. 回溯：从 ptr[n] 开始恢复切分路径
```

**复杂度分析**：
- 时间复杂度：$O(n^2 \cdot T_{lookup})$，其中$T_{lookup}$为词表查询时间
- 使用Trie树可优化为$O(n \cdot L_{max})$，其中$L_{max}$为最长词长度

### 四、BytePiece训练算法

#### 4.1 词频统计与剪枝

**n-gram计数**：对于语料库$C$，统计所有n-gram的频次
\begin{equation}
count(w) = \sum_{s \in C} \sum_{i} \mathbb{I}(s[i:i+|w|] = w) \tag{17}
\end{equation}
其中$\mathbb{I}(\cdot)$是指示函数。

**最大似然估计**：词概率的MLE为
\begin{equation}
p(w) = \frac{count(w)}{\sum_{w' \in V} count(w')} \tag{18}
\end{equation}

**平滑处理**：为避免零概率，采用Add-k平滑
\begin{equation}
p_{smooth}(w) = \frac{count(w) + k}{Nsum + k \cdot |V|} \tag{19}
\end{equation}
其中$N = \sum_{w'} count(w')$，通常取$k=1$（Laplace平滑）。

#### 4.2 冗余词剪枝

**冗余性判断**：若词$w = w_1 \circ w_2$（拼接），且满足
\begin{equation}
p(w_1) \cdot p(w_2) > p(w) \tag{20}
\end{equation}
则$w$永远不会在Unigram分词中被选中（因为分开切分概率更大），可以剪枝。

**数学证明**：假设序列包含$w$，比较两种切分：
- 保留$w$：贡献$\log p(w)$
- 拆分为$w_1, w_2$：贡献$\log p(w_1) + \log p(w_2)$

由于
\begin{equation}
\log p(w_1) + \log p(w_2) = \log(p(w_1) \cdot p(w_2)) > \log p(w) \tag{21}
\end{equation}
拆分总是更优，因此$w$冗余。

**递归剪枝**：考虑词$w = w_1 \circ w_2 \circ w_3$
\begin{equation}
p(w_1) \cdot p(w_2) \cdot p(w_3) > p(w) \Rightarrow w\text{冗余} \tag{22}
\end{equation}

### 五、压缩率理论分析

#### 5.1 理论压缩上界

**熵率**：定义语言的熵率为
\begin{equation}
H_{\infty} = \lim_{n \to \infty} \frac{1}{n} H(B_1, B_2, \ldots, B_n) \tag{23}
\end{equation}

**香农源编码定理**：对于平稳遍历信源，最优压缩率趋近于熵率
\begin{equation}
\lim_{n \to \infty} \frac{L_{compressed}}{L_{original}} = \frac{H_{\infty}}{8} \tag{24}
\end{equation}

**实际估计**：对于中文文本，经验值为
- 熵率$H_{\infty} \approx 5-6$ bits/byte
- 理论压缩极限约为$\frac{5.5}{8} \approx 0.69$

#### 5.2 Tokenizer压缩率

**tokens数统计**：对于长度$N$（bytes）的文本，分词后得到$M$个tokens
\begin{equation}
R_{actual} = \frac{N}{M} \tag{25}
\end{equation}

**与词表大小的关系**：根据Zipf定律，词频分布为
\begin{equation}
p(w_r) \propto \frac{1}{r^{\alpha}} \tag{26}
\end{equation}
其中$r$是词的频率排名，$\alpha \approx 1$。

**期望词长**：在Zipf分布下，期望token长度为
\begin{equation}
\mathbb{E}[L] = \sum_{r=1}^{|V|} p(w_r) \cdot |w_r| \tag{27}
\end{equation}

**优化目标**：最大化
\begin{equation}
\max_{V} \frac{\mathbb{E}[L]}{\log_2 |V|} \tag{28}
\end{equation}
在固定$|V|$下获得最长的平均token长度。

### 六、数值稳定性技巧

#### 6.1 对数空间计算

**问题**：直接计算$\prod_i p(w_i)$会导致数值下溢（概率连乘快速趋近0）。

**解决方案**：在对数空间计算
\begin{equation}
\log \prod_i p(w_i) = \sum_i \log p(w_i) \tag{29}
\end{equation}

**LogSumExp技巧**：计算$\log(\sum_i e^{x_i})$时避免溢出
\begin{equation}
\log \sum_i e^{x_i} = x_{max} + \log \sum_i e^{x_i - x_{max}} \tag{30}
\end{equation}
其中$x_{max} = \max_i x_i$。

#### 6.2 Laplace平滑的数值考虑

**极端情况**：当$count(w) = 0$时，不加平滑会导致$p(w) = 0$，进而$\log p(w) = -\infty$。

**平滑后的对数概率**：
\begin{equation}
\log p_{smooth}(w) = \log(count(w) + k) - \log(N + k \cdot |V|) \tag{31}
\end{equation}

**下界保证**：即使$count(w) = 0$，仍有
\begin{equation}
\log p_{smooth}(w) \geq \log k - \log(N + k \cdot |V|) \tag{32}
\end{equation}
避免了负无穷。

### 七、BytePiece vs BPE比较

#### 7.1 BPE合并规则

**BPE迭代过程**：每次选择频次最高的byte pair合并
\begin{equation}
(b_i, b_j)^* = \argmax_{(b_i, b_j)} count(b_i, b_j) \tag{33}
\end{equation}

**贪心近似**：BPE是贪心算法，不保证全局最优。每次合并后需重新统计：
\begin{equation}
count_{new}(b_i b_j) = count_{old}(b_i, b_j) \tag{34}
\end{equation}

#### 7.2 复杂度对比

**BPE训练复杂度**：
- 每轮需扫描全部语料：$O(N)$
- 需要$|V|$轮迭代
- 总复杂度：$O(N \cdot |V|)$

**BytePiece训练复杂度**：
- n-gram统计：$O(N \cdot n)$（单次扫描）
- 排序与剪枝：$O(|V| \log |V|)$
- 总复杂度：$O(N \cdot n + |V| \log |V|)$

**空间复杂度**：
- BPE：需存储所有pair频次，约$O(|V|^2)$
- BytePiece：存储n-gram频次，约$O(|V|)$

### 八、实践建议

#### 8.1 超参数选择

**n-gram阶数$n$**：
- $n=6$的理论依据：英文单词平均长度$\approx 4-5$，加上词缀，$n=6$足够覆盖
- 更大的$n$：提升准确性，但增加统计复杂度和稀疏性
- 建议：$n \in [5, 7]$

**词表大小$|V|$**：
- 小词表（$<10k$）：压缩率低，训练快
- 中词表（$30-50k$）：平衡点，适合多数应用
- 大词表（$>100k$）：压缩率高，但增加模型参数

**平滑系数$k$**：
- $k=0$：无平滑，可能出现零概率
- $k=1$：Laplace平滑，经典选择
- $k \in (0,1)$：Lidstone平滑，更温和

#### 8.2 工程优化

**Trie树加速**：词表查询从$O(|V|)$优化到$O(L_{max})$
\begin{equation}
T_{Viterbi} = O(n \cdot L_{max} \cdot T_{Trie}) \tag{35}
\end{equation}
其中$T_{Trie} = O(L_{max})$。

**并行化**：n-gram统计天然支持Map-Reduce
\begin{equation}
count(w) = \sum_{partition} count_{partition}(w) \tag{36}
\end{equation}

**内存优化**：使用布隆过滤器预过滤低频n-gram
\begin{equation}
\text{filter}(w) = \begin{cases}
1, & \text{if } count(w) \geq \theta \\
0, & \text{otherwise}
\end{cases} \tag{37}
\end{equation}

### 九、理论拓展

#### 9.1 信息论视角

**互信息**：词$w$的信息增益为
\begin{equation}
I(w) = \log \frac{p(w)}{\prod_{i} p(c_i)} \tag{38}
\end{equation}
高互信息的词更应该保留在词表中。

**困惑度（Perplexity）**：评估语言模型质量
\begin{equation}
PPL = 2^{H(T)} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(w_i)\right) \tag{39}
\end{equation}
越低越好。

#### 9.2 与其他Tokenizer的联系

**WordPiece**：使用似然增益而非频次
\begin{equation}
score(w_i, w_j) = \frac{p(w_i w_j)}{p(w_i) p(w_j)} \tag{40}
\end{equation}

**Unigram LM**（SentencePiece）：本质上与BytePiece相同，但使用EM算法迭代优化
\begin{equation}
p^{(t+1)}(w) = \frac{\sum_{s} \mathbb{E}_{p^{(t)}}[count(w|s)]}{\sum_{w'} \sum_{s} \mathbb{E}_{p^{(t)}}[count(w'|s)]} \tag{41}
\end{equation}

### 十、总结

BytePiece通过n-gram语言模型和Viterbi算法实现了高效的Tokenizer训练：

1. **信息论基础**：压缩率$R = \frac{L}{\log_2 V}$，受熵率$H_{\infty}$限制
2. **Unigram分词**：动态规划求解最大概率路径，复杂度$O(n \cdot L_{max})$
3. **训练算法**：一次扫描统计n-gram，复杂度$O(N \cdot n)$，远快于BPE的$O(N \cdot |V|)$
4. **剪枝策略**：移除冗余词$p(w_1)p(w_2) > p(w)$，减少词表大小
5. **数值稳定**：对数空间计算+Laplace平滑避免下溢

**核心优势**：纯Python实现、训练高效、压缩率高、无需NFKC归一化。

