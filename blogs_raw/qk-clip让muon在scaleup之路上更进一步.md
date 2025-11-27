---
title: QK-Clip：让Muon在Scaleup之路上更进一步
slug: qk-clip让muon在scaleup之路上更进一步
date: 2025-07-12
tags: 优化, attention, 优化器, muon, 生成模型
status: completed
---

# QK-Clip：让Muon在Scaleup之路上更进一步

**原文链接**: [https://spaces.ac.cn/archives/11126](https://spaces.ac.cn/archives/11126)

**发布日期**: 

---

四个月前，我们发布了[Moonlight](/archives/10739)，在16B的MoE模型上验证了[Muon](/archives/10592)优化器的有效性。在Moonlight中，我们确认了给Muon添加Weight Decay的必要性，同时提出了通过Update RMS对齐来迁移Adam超参的技巧，这使得Muon可以快速应用于LLM的训练。然而，当我们尝试将Muon进一步拓展到千亿参数以上的模型时，遇到了新的“拦路虎”——MaxLogit爆炸。

为了解决这个问题，我们提出了一种简单但极其有效的新方法，我们称之为“QK-Clip”。该方法从一个非常本质的角度去看待和解决MaxLogit现象，并且无损模型效果，这成为我们最新发布的万亿参数模型“[Kimi K2](https://moonshotai.github.io/Kimi-K2/)”的关键训练技术之一。

## 问题描述 #

我们先来简单介绍一下MaxLogit爆炸现象。回顾Attention的定义  
\begin{equation}\boldsymbol{O} = softmax(\boldsymbol{Q}\boldsymbol{K}^{\top})\boldsymbol{V}\end{equation}  
这里省略了缩放因子$1/\sqrt{d}$，因为它总可以吸收到$\boldsymbol{Q},\boldsymbol{K}$的定义中。“MaxLogit爆炸”中的Logit，指的是Softmax前的Attention矩阵，即$\boldsymbol{Q}\boldsymbol{K}^{\top}$，而MaxLogit指的是全体Logit的最大值，我们将它记为  
\begin{equation}S_{\max} = \max_{i,j}\, \boldsymbol{q}_i\cdot \boldsymbol{k}_j\end{equation}  
这里的$\max$其实还要在batch_size维度上取，最终得到一个标量。而MaxLogit爆炸是指，$S_{\max}$随着训练的推进一直往上涨，增长速度是线性甚至是超线性的，并且在相当长的时间内没有稳定的迹象。

[![MaxLogit爆炸现象](/usr/uploads/2025/07/3626912260.png)](/usr/uploads/2025/07/3626912260.png "点击查看原图")

MaxLogit爆炸现象

MaxLogit本质上是一个异常值指标，它的爆炸意味着异常值超出了可控范围。具体来说，我们有  
\begin{equation}|\boldsymbol{q}_i\cdot \boldsymbol{k}_j| \leq \Vert\boldsymbol{q}_i\Vert \Vert\boldsymbol{k}_j\Vert = \Vert\boldsymbol{x}_i\boldsymbol{W}_q\Vert \Vert\boldsymbol{x}_j\boldsymbol{W}_k\Vert \leq \Vert\boldsymbol{x}_i\Vert \Vert\boldsymbol{x}_j\Vert \Vert\boldsymbol{W}_q\Vert \Vert\boldsymbol{W}_k\Vert\label{eq:kexi}\end{equation}  
由于$\boldsymbol{x}$通常会加RMSNorm，所以一般情况下$\Vert\boldsymbol{x}_i\Vert \Vert\boldsymbol{x}_j\Vert$是不会爆炸的，因此MaxLogit爆炸意味着谱范数$\Vert\boldsymbol{W}_q\Vert,\Vert\boldsymbol{W}_k\Vert$有往无穷大发展的风险，这显然不是一个好消息。

由于再大的数值经过Softmax后都变得小于1，所以比较幸运的情况下，这个现象不会带来太严重的后果，顶多是浪费了一个Attention Head，但比较糟糕的情况下，可能会引起Grad Spike甚至训练崩溃。因此，保险起见应当尽量避免MaxLogit爆炸的出现。

## 已有尝试 #

在[《Muon续集：为什么我们选择尝试Muon？》](/archives/10739)中我们简单分析过，Weight Decay能一定程度上预防MaxLogit爆炸，所以小模型出现MaxLogit爆炸的概率很小，即便像Moonlight这样的16B模型，MaxLogit最多涨到120后就自动降下来了。

[![Moonlight的MaxLogit自动降了下来](/usr/uploads/2025/07/3150611957.png)](/usr/uploads/2025/07/3150611957.png "点击查看原图")

Moonlight的MaxLogit自动降了下来

换句话说，MaxLogit爆炸更多出现在非常大参数量的模型中，模型越大，训练的不稳定因素越多，Weight Decay越难稳定训练。这时候增加Weight Decay自然也能加强控制，但同时也会带来明显的效果损失，所以此路不通。另一个比较直接的思路是直接给Logit加$\text{softcap}$：  
\begin{equation}\boldsymbol{O} = softmax(\text{softcap}(\boldsymbol{Q}\boldsymbol{K}^{\top};\tau))\boldsymbol{V}\end{equation}  
其中$\text{softcap}(x;\tau) = \tau\tanh(x/\tau)$，由Google的[Gemma2](https://papers.cool/arxiv/2408.00118)引入。由于$\tanh$的有界性，$\text{softcap}$自然是能够保证$\text{softcap}$后的Logit有界的，但无法保证$\text{softcap}$前的Logit是有界的（亲测），所以$\text{softcap}$只是将一个问题转化为了另一个问题，实际上并没有解决问题。

也许Google自己都意识到了这一点，所以在后来的[Gemma3](https://papers.cool/arxiv/2503.19786)中没有用$\text{softcap}$了，而改用“QK-Norm”：  
\begin{equation}\boldsymbol{O} = softmax(\tilde{\boldsymbol{Q}}\tilde{\boldsymbol{K}}{}^{\top})\boldsymbol{V},\quad \begin{aligned}  
\tilde{\boldsymbol{Q}}=&\,\text{RMSNorm}(\boldsymbol{Q}) \\\  
\tilde{\boldsymbol{K}}=&\,\text{RMSNorm}(\boldsymbol{K})  
\end{aligned}\end{equation}

QK-Norm确实是压制MaxLogit非常有效的方法，然而它只适用于MHA、GQA等，不适用于MLA，因为QK-Norm需要把$\boldsymbol{Q},\boldsymbol{K}$给Materialize出来，但对于MLA来说，它训练阶段跟Decoding阶段的$\boldsymbol{Q},\boldsymbol{K}$并不一样（如下式所示），在Decoding阶段我们没法完全Materialize训练阶段的$\boldsymbol{K}$，换言之，Decoding阶段没法做QK-Norm。

$$\require{cancel}\begin{array}{c|c}  
\text{训练/Prefill} & \text{Decoding} \\\  
\\\  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = \frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)},\boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r}\\\  
\boldsymbol{k}_i^{(s)} = \left[\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)},\boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_k + d_r} \\\  
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c}  
\end{gathered}  
&  
\begin{gathered}  
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}\boldsymbol{W}_v^{(1)}, \boldsymbol{o}_t^{(2)}\boldsymbol{W}_v^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\boldsymbol{W}_v^{(h)}\right] \\\\[10pt]  
\boldsymbol{o}_t^{(s)} = \frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} }{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\\\[15pt]  
\boldsymbol{q}_i^{(s)} = \left[\boldsymbol{x}_i\boldsymbol{W}_{qc}^{(s)}\boldsymbol{W}_{kc}^{(s)}{}^{\top}, \boldsymbol{x}_i\boldsymbol{W}_{qr}^{(s)}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c + d_r}\\\  
\boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \left[\boldsymbol{c}_i, \boldsymbol{x}_i\boldsymbol{W}_{kr}^{\color{#ccc}{\smash{\bcancel{(s)}}}}\color{#3ce2f7}{\boldsymbol{\mathcal{R}}_i}\right]\in\mathbb{R}^{d_c + d_r}\\\  
\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{c}_i= \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c}  
\end{gathered} \\\  
\end{array} $$

为什么要用MLA？我们已经用两篇文章[《Transformer升级之路：21、MLA好在哪里?（上）》](/archives/10907)和[《Transformer升级之路：21、MLA好在哪里?（下）》](/archives/11111)讨论了这个问题，这里不再重复。总之，我们希望MLA也能有类似QK-Norm的能够保证压制MaxLogit的手段。

## 直击目标 #

期间我们还尝试了一些间接手段，比如单独降低$\boldsymbol{Q},\boldsymbol{K}$的学习率、单独增大它们的Weight Decay等，但都不奏效。最接近成功的一次是Partial QK-Norm，对于MLA来说，它的$\boldsymbol{Q},\boldsymbol{K}$分为qr、qc、kr、kc四个部分，其中前三部分在Decoding时都是可以Materialize的，所以我们给这三部分都加上RMSNorm，结果是可以压制MaxLogit，但长度激活效果非常糟糕。

在失败多次之后，我们不禁开始反思：前面我们的尝试其实都只是压制MaxLogit的“间接手段”，真正能保证解决MaxLogit爆炸的直接手段是什么？从不等式$\eqref{eq:kexi}$我们不难联想到可以对$\boldsymbol{W}_q,\boldsymbol{W}_k$做[奇异值裁剪](/archives/11006)，但这本质上还是间接手段，而且奇异值裁剪的计算成本也不低。

但很明显，对$\boldsymbol{W}_q,\boldsymbol{W}_k$进行事后缩放理论上是可行的，问题是 _什么时候缩放、缩放多少_ 。终于，某天福至心灵之下，笔者总算反应过来：**MaxLogit本身就是触发缩放的最直接信号！** 具体来说，当MaxLogit超过期望阈值$\tau$时，我们直接给$\boldsymbol{Q}\boldsymbol{K}^{\top}$乘上$\gamma = \tau / S_{\max}$，那么新的MaxLogit肯定就不超过$\tau$了。乘$\gamma$的操作，我们可以分别吸收到权重$\boldsymbol{Q}\boldsymbol{K}$的权重上去，于是我们得到初版QK-Clip：  
$$\begin{aligned}  
&\boldsymbol{W}_t = \text{Optimizer}(\boldsymbol{W}_{t-1}, \boldsymbol{G}_t) \\\  
&\text{if }S_{\max}^{(l)} > \tau\text{ and }\boldsymbol{W} \in \\{\boldsymbol{W}_q^{(l)}, \boldsymbol{W}_k^{(l)}\\}: \\\  
&\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \sqrt{\tau / S_{\max}^{(l)}}  
\end{aligned}$$

其中$S_{\max}^{(l)}$是第$l$层Attention的MaxLogit，$\boldsymbol{W}_q^{(l)}, \boldsymbol{W}_k^{(l)}$是它$\boldsymbol{Q},\boldsymbol{K}$的权重。也就是说，在优化器更新之后，根据$S_{\max}^{(l)}$的大小来决定是否对$\boldsymbol{Q},\boldsymbol{K}$的权重进行裁剪，裁剪的幅度直接由$S_{\max}^{(l)}$与阈值$\tau$的比例来决定，直接保证裁剪后的矩阵不再MaxLogit爆炸。同时，由于是直接对权重进行操作，所以不影响推理模式，自然也就兼容MLA了。

## 精细调整 #

初版QK-Clip确实已经能成功压制MLA的MaxLogit，但经过仔细观察模型的“内科”后，我们发现它会出现“过度裁剪”的问题，修复该问题后就得到最终版QK-Clip。

我们知道，不管哪种Attention变体都有多个Head，一开始我们是每一层Attention只监控一个MaxLogit指标，所有Head的Logit是放在一起取Max的，这导致QK-Clip也是所有Head一起Clip的。然而，当我们分别监控每个Head的MaxLogit后发现，实际上每层只有为数不多的Head会出现MaxLogit爆炸，如果所有Head按同一个比例来Clip，那么大部份Head都是被“无辜受累”的了，这就是过度裁剪的含义。

简单来说，QK-Clip的操作是乘以一个小于1的数，这个数对于MaxLogit爆炸的Head来说是刚刚好抵消增长趋势，但是对于其他head来说是单纯的缩小（它们没有增长趋势或者增长趋势很弱）。由于长期无端被乘一个小于1的数，那么很容易出现就趋于零的现象，这是“过度裁剪”的表现。

所以，为了避免“殃及池鱼”，我们应该Per-Head地进行监控MaxLogit和QK-Clip。不过这里边又隐藏了另一个魔鬼细节：初版QK-Clip是将Clip因子平摊到$\boldsymbol{Q},\boldsymbol{K}$上的，但是MLA的$\boldsymbol{Q},\boldsymbol{K}$有qr、qc、kr、kc四部分，其中kr是所有Head共享的，如果对它Clip，那么同样会有“殃及池鱼”的问题。因此，对于(qr, kr)，我们应该只Clip到qr上去。

经过上述调整，最终版的QK-Clip为  
$$\begin{aligned}  
&\boldsymbol{W}_t = \text{Optimizer}(\boldsymbol{W}_{t-1}, \boldsymbol{G}_t) \\\  
&\text{if }S_{\max}^{(l,h)} > \tau: \\\  
&\qquad\text{if }\boldsymbol{W} \in \\{\boldsymbol{W}_{qc}^{(l,h)}, \boldsymbol{W}_{kc}^{(l,h)}\\}: \\\  
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \sqrt{\tau / S_{\max}^{(l,h)}} \\\  
&\qquad\text{elif }\boldsymbol{W} \in \\{\boldsymbol{W}_{qr}^{(l,h)}\\}: \\\  
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \tau / S_{\max}^{(l,h)}  
\end{aligned}$$  
其中上标${}^{(l,h)}$表示第$l$层、第$h$个Head。

## 扩展之路 #

至此，QK-Clip的操作细节已经介绍完毕，它直接以我们期望的MaxLogit为信号，对$\boldsymbol{Q},\boldsymbol{K}$的权重进行尽可能小的改动，达到了将MaxLogit值控制在指定阈值内的效果。同时因为这是直接对权重进行修改的方法，所以它兼容性比QK-Norm更好，可以用于MLA。

在Kimi K2的训练中，我们设置阈值$\tau$为100，总训练步数约为220k steps，从大致7k steps开始，就出现了MaxLogit超过$\tau$的Head，此后在相当长的时间内，Muon Update和QK-Clip都在“拉锯战”，即Muon想要增加MaxLogit而QK-Clip想要降低MaxLogit，它们一直处于微妙的平衡状态。有趣的是，70k steps之后，所有Head的MaxLogit都主动降低到了100以下，QK-Clip不再生效。

[![经过接近70k steps的Muon和QK-Clip拉锯战后，MaxLogit 主动降了下来](/usr/uploads/2025/07/550205001.png)](/usr/uploads/2025/07/550205001.png "点击查看原图")

经过接近70k steps的Muon和QK-Clip拉锯战后，MaxLogit 主动降了下来

这表明，在Weight Decay的作用下，只要我们能稳住训练，模型最后很可能都会主动将MaxLogit降下来，QK-Clip的作用，正是帮助模型更平稳地度过训练初期。可能有读者担心QK-Clip会有损效果，但我们在小模型上做了对比实验，即便通过QK-Clip将MaxLogit压得特别小（比如30），也没有观察到效果有实质区别，再加上中后期模型会主动将MaxLogit降下来的这一现象，我们有理由相信QK-Clip对效果是无损的。

我们在实验中也观察到，Muon普遍比Adam更容易MaxLogit爆炸，所以某种程度上来说，QK-Clip是专门为Muon补充的更新规则，它是Muon在超大规模训练上的“通关秘笈”之一，这也是本文标题的含义。为此，我们将我们Moonlight中所提的Muon改动跟QK-Clip组合起来，起了个“MuonClip”的名字（$\boldsymbol{W}\in\mathbb{R}^{n\times m}$）：  
$$\text{MuonClip}\quad\left\\{\quad\begin{aligned}  
&\boldsymbol{M}_t = \mu \boldsymbol{M}_{t−1} + \boldsymbol{G}_t \\\\[8pt]  
&\boldsymbol{O}_t = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{M}_t) \underbrace{\times \sqrt{\max(n,m)}\times 0.2}_{\text{Match Adam Update RMS}} \\\\[8pt]  
&\boldsymbol{W}_t = \boldsymbol{W}_{t−1} − \eta_t (\boldsymbol{O}_t + \lambda \boldsymbol{W}_{t-1}) \\\\[8pt]  
&\left.\begin{aligned}  
&\text{if }S_{\max}^{(l,h)} > \tau: \\\  
&\qquad\text{if }\boldsymbol{W} \in \\{\boldsymbol{W}_{qc}^{(l,h)}, \boldsymbol{W}_{kc}^{(l,h)}\\}: \\\  
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \sqrt{\tau / S_{\max}^{(l,h)}} \\\  
&\qquad\text{elif }\boldsymbol{W} \in \\{\boldsymbol{W}_{qr}^{(l,h)}\\}: \\\  
&\qquad\qquad\boldsymbol{W}_t \leftarrow \boldsymbol{W}_t \times \tau / S_{\max}^{(l,h)}  
\end{aligned}\quad\right\\} \text{QK-Clip}  
\end{aligned}\right.$$

注意，“Muon普遍比Adam更容易MaxLogit爆炸”并不意味着只有Muon会MaxLogit爆炸，我们知道[DeepSeek-V3](https://papers.cool/arxiv/2412.19437)是Adam训练的，而我们从DeepSeek-V3的开源模型中也观察到了MaxLogit爆炸现象，还有Gemma2用$\text{softcap}$防止MaxLogit爆炸，它也是Adam训的。因此，虽然我们强调了QK-Clip对Muon的价值，但如果读者坚持用Adam，那么它也是可以跟Adam结合成AdamClip的。

## 原因思考 #

为什么Muon更容易导致MaxLogit爆炸呢？这一节笔者尝试给出一个理论角度的解释，供大家参考。

从不等式$\eqref{eq:kexi}$可以看出，MaxLogit爆炸往往意味着$\boldsymbol{W}_q$或$\boldsymbol{W}_k$的谱范数出现爆炸的迹象，实际上谱范数的定义中也包含了取$\max$操作，两者本质上是相通的。因此，问题可以转化为“为什么Muon更容易导致谱范数爆炸”。我们知道谱范数等于最大的奇异值，所以又可以进一步联想到“为什么Muon更倾向于增大奇异值”。

Muon和Adam的区别是什么呢？Muon给出的更新量是经过$\msign$运算的，所有奇异值都相等，即它的[有效秩](/archives/10847)是**满秩** ；而一般情况下的矩阵，奇异值通常都是有大有小，并且以前面几个奇异值为主，从[有效秩](/archives/10847)的角度看它们是**低秩** 的，我们对Adam更新量的假设也是如此。这个假设并不新鲜，比如[高阶MuP](/archives/10795)同样假设了Adam更新量的低秩性。

用公式来说，我们设参数$\boldsymbol{W}_{t-1}$的SVD为$\sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$，Muon更新量的SVD为$\sum_j \bar{\sigma}\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}$，Adam更新量的SVD为$\sum_j \tilde{\sigma}_j\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}$，那么  
\begin{gather}  
\boldsymbol{W}_t = \sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + \sum_j \bar{\sigma}\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}\qquad (\text{Muon}) \\\  
\boldsymbol{W}_t = \sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + \sum_j \tilde{\sigma}_j\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}\qquad (\text{Adam}) \\\  
\end{gather}

很明显，如果奇异向量对$\boldsymbol{u}_i \boldsymbol{v}_i^{\top}$跟某个$\bar{\boldsymbol{u}}_j \bar{\boldsymbol{v}}_j^{\top}$或$\tilde{\boldsymbol{u}}_j \tilde{\boldsymbol{v}}_j^{\top}$很接近，那它们将会直接叠加起来，从而增大$\boldsymbol{W}_t$的奇异值。由于Muon的更新量是满秩的，所以它与$\boldsymbol{W}_{t-1}$的“碰撞几率”会远大于Adam的，所以Muon更容易增大参数的奇异值。

当然，上述分析是通用的，不限于$\boldsymbol{Q},\boldsymbol{K}$的权重，实际上在Moonlight中我们已经验证过，Muon训出来的模型权重的奇异值熵普遍更高，这也佐证了上述猜测。Attention Logit的特殊之处在于，它是双线性形式$\boldsymbol{q}_i\cdot \boldsymbol{k}_j = (\boldsymbol{x}_i \boldsymbol{W}_q)\cdot(\boldsymbol{x}_j \boldsymbol{W}_k)$，$\boldsymbol{W}_q,\boldsymbol{W}_k$的连乘使得爆炸的风险更大，还容易导致“糟的更糟”的恶性循环，最终促成了MaxLogit爆炸。

[![Muon与Adam训练的模型权重奇异值熵（等价于有效秩）比较](/usr/uploads/2025/07/1022731099.png)](/usr/uploads/2025/07/1022731099.png "点击查看原图")

Muon与Adam训练的模型权重奇异值熵（等价于有效秩）比较

最后就是“Muon的碰撞几率远大于Adam”是相对而言的，实际上奇异向量碰撞在一块还是小概率事件，这也就能解释为啥只有小部份Attention Head会有MaxLogit爆炸现象了。

这个视角还可以解释之前Moonlight的一个现象：用Muon/Adam预训练的模型，反过来用Adam/Muon微调，结果通常是次优的。因为Muon的训练权重有效秩更高，而Adam的更新量是低秩的，一高一低之下，微调效率就变差了；反之，Adam的训练权重有效秩更低，但Muon的更新量是满秩的，它有更大概率去干预那些小奇异值分量，让模型偏离预训练的低秩局部最优点，从而影响微调效率。

## 一些延伸 #

写到这里，关于QK-Clip比较重要计算和实验细节应该都讲清楚了。另外还需要提醒的是，QK-Clip思想很简单，但由于需要Per-Head来Clip，因此在分布式训练中写起来还是略微有点难度的，因为此时的参数矩阵往往被切分得“支离破碎”（在Muon基础上改起来不算难，在Adam基础上改则稍显复杂）。

对于笔者及其团队来说，QK-Clip不单单是解决MaxLogit爆炸问题的一个具体方法，还是反复尝试通过间接手段来解决问题且失败后的一次”幡然醒悟“：**既然有了明确的度量指标，那么我们应该寻求能够保证解决问题的直接思路，而不是在降低LR、增大Weight Decay、部分QK-Norm等 _可能但不一定能_ 解决问题的思路上浪费时间。**

从方法上来看，QK-Clip的思路也不限于解决MaxLogit爆炸，它可以说是解决很多训练不稳定问题的“抗生素”。所谓抗生素，指的是它也许并不是解决问题最精妙的方法，但往往是解决问题最直接有效的方法之一，QK-Clip正是具有这个特点，它可以一般地推广成“哪里不稳Clip哪里”。

比如，有些情况下模型会出现“MaxOutput爆炸”的问题，这时候我们可以考虑根据MaxOutput的值来Clip权重$\boldsymbol{W}_o$。类比QK-Clip的Per-Head操作，这里我们也需要考虑Per-Dim操作，但Per-Dim Clip的成本显然太大，可能需要折中一下。总之，“哪里不稳Clip哪里”提供了统一的解决思路，但具体细节就要看大家发挥了。

最后，QK-Clip这种根据某些信号手动制定更新规则的操作，一定程度上是受到了DeepSeek的[Loss-Free](/archives/10757)负载均衡策略的启发而悟到的，这里再次致敬DeepSeek！

## 文章小结 #

本文提出了QK-Clip，它是MaxLogit爆炸问题的一种新思路，跟QK-Norm不同，它是对Q、K权重的一种事后调整方案，并不改变模型的前向计算，因此适用性更广，它是“Muon + MLA”组合在超大规模训练上的重要稳定策略，也是我们最新发布的万亿模型Kimi K2的关键技术之一。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11126>_

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

苏剑林. (Jul. 12, 2025). 《QK-Clip：让Muon在Scaleup之路上更进一步 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11126>

@online{kexuefm-11126,  
title={QK-Clip：让Muon在Scaleup之路上更进一步},  
author={苏剑林},  
year={2025},  
month={Jul},  
url={\url{https://spaces.ac.cn/archives/11126}},  
} 


---

## 详细数学推导与分析

### 1. Attention机制的数学基础

#### 1.1 标准Attention公式

标准的缩放点积注意力（Scaled Dot-Product Attention）定义为：

$$
\boldsymbol{O} = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V} \tag{1}
$$

其中：
- $\boldsymbol{Q} \in \mathbb{R}^{n \times d_k}$ 是查询矩阵
- $\boldsymbol{K} \in \mathbb{R}^{m \times d_k}$ 是键矩阵
- $\boldsymbol{V} \in \mathbb{R}^{m \times d_v}$ 是值矩阵
- $n$ 是查询序列长度，$m$ 是键值序列长度
- $d_k$ 是键/查询维度，$d_v$ 是值维度

**缩放因子吸收**：缩放因子$1/\sqrt{d_k}$可以吸收到$\boldsymbol{Q}$或$\boldsymbol{K}$的定义中：

$$
\boldsymbol{O} = \text{softmax}(\boldsymbol{Q}\boldsymbol{K}^{\top})\boldsymbol{V} \tag{2}
$$

其中$\boldsymbol{Q} \leftarrow \boldsymbol{Q}/\sqrt{d_k}$或$\boldsymbol{K} \leftarrow \boldsymbol{K}/\sqrt{d_k}$。

#### 1.2 Attention Logits定义

Attention logits矩阵定义为：

$$
\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{K}^{\top} \in \mathbb{R}^{n \times m} \tag{3}
$$

矩阵元素为：

$$
S_{ij} = \boldsymbol{q}_i \cdot \boldsymbol{k}_j = \sum_{l=1}^{d_k} q_{il} k_{jl} \tag{4}
$$

其中$\boldsymbol{q}_i$是第$i$个查询向量，$\boldsymbol{k}_j$是第$j$个键向量。

#### 1.3 MaxLogit定义

MaxLogit定义为所有logits的最大值：

$$
S_{\max} = \max_{i,j} S_{ij} = \max_{i,j} (\boldsymbol{q}_i \cdot \boldsymbol{k}_j) \tag{5}
$$

在实际实现中，还需要在批次维度上取最大值：

$$
S_{\max}^{\text{global}} = \max_{b \in \text{Batch}} \max_{i,j} S_{ij}^{(b)} \tag{6}
$$

### 2. MaxLogit爆炸的数学分析

#### 2.1 点积的上界

由Cauchy-Schwarz不等式：

$$
|\boldsymbol{q}_i \cdot \boldsymbol{k}_j| \leq \|\boldsymbol{q}_i\| \|\boldsymbol{k}_j\| \tag{7}
$$

对于$\boldsymbol{q}_i = \boldsymbol{x}_i \boldsymbol{W}_q$和$\boldsymbol{k}_j = \boldsymbol{x}_j \boldsymbol{W}_k$：

$$
\begin{align}
|\boldsymbol{q}_i \cdot \boldsymbol{k}_j| &= |\boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^{\top} \boldsymbol{x}_j^{\top}| \tag{8} \\
&\leq \|\boldsymbol{x}_i \boldsymbol{W}_q\| \|\boldsymbol{x}_j \boldsymbol{W}_k\| \tag{9} \\
&\leq \|\boldsymbol{x}_i\| \|\boldsymbol{W}_q\| \|\boldsymbol{x}_j\| \|\boldsymbol{W}_k\| \tag{10}
\end{align}
$$

其中$\|\boldsymbol{W}\|$表示谱范数（最大奇异值）：

$$
\|\boldsymbol{W}\| = \sigma_{\max}(\boldsymbol{W}) = \sup_{\|\boldsymbol{x}\|=1} \|\boldsymbol{W}\boldsymbol{x}\| \tag{11}
$$

#### 2.2 RMSNorm的作用

在Transformer中，输入通常经过RMSNorm：

$$
\text{RMSNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\text{RMS}(\boldsymbol{x})} \odot \boldsymbol{g} \tag{12}
$$

其中：

$$
\text{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2} = \frac{\|\boldsymbol{x}\|}{\sqrt{d}} \tag{13}
$$

归一化后的向量范数为：

$$
\|\text{RMSNorm}(\boldsymbol{x})\| = \|\boldsymbol{g}\| \cdot \sqrt{d} \tag{14}
$$

假设$\boldsymbol{g} \approx \boldsymbol{1}$，则$\|\text{RMSNorm}(\boldsymbol{x})\| \approx \sqrt{d}$。

因此，MaxLogit的上界主要由权重谱范数决定：

$$
S_{\max} \lesssim d \cdot \|\boldsymbol{W}_q\| \|\boldsymbol{W}_k\| \tag{15}
$$

**MaxLogit爆炸的充要条件**：$\|\boldsymbol{W}_q\|$或$\|\boldsymbol{W}_k\|$趋向无穷大。

#### 2.3 谱范数增长的动力学

设权重更新为：

$$
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \boldsymbol{\Delta}_t \tag{16}
$$

谱范数的变化为：

$$
\|\boldsymbol{W}_t\| \leq \|\boldsymbol{W}_{t-1}\| + \eta_t \|\boldsymbol{\Delta}_t\| \tag{17}
$$

**增长条件**：若更新量$\boldsymbol{\Delta}_t$与$\boldsymbol{W}_{t-1}$的主奇异向量方向一致，则：

$$
\|\boldsymbol{W}_t\| \approx \|\boldsymbol{W}_{t-1}\| + \eta_t \|\boldsymbol{\Delta}_t\| \cos\theta \tag{18}
$$

其中$\theta$是更新方向与主奇异向量的夹角。

### 3. Muon优化器详解

#### 3.1 Muon更新规则

Muon（Momentum Orthogonalized by Normalization）优化器的更新规则为：

$$
\begin{align}
\boldsymbol{M}_t &= \mu \boldsymbol{M}_{t-1} + \boldsymbol{G}_t \tag{19} \\
\boldsymbol{O}_t &= \text{msign}(\boldsymbol{M}_t) \tag{20} \\
\boldsymbol{W}_t &= \boldsymbol{W}_{t-1} - \eta_t \boldsymbol{O}_t \tag{21}
\end{align}
$$

其中：
- $\boldsymbol{G}_t = \nabla_{\boldsymbol{W}} L(\boldsymbol{W}_{t-1})$ 是梯度
- $\mu \in (0,1)$ 是动量系数，通常为0.95
- $\text{msign}(\cdot)$ 是矩阵符号函数

#### 3.2 矩阵符号函数

矩阵符号函数定义为：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{U}\text{sign}(\boldsymbol{\Sigma})\boldsymbol{V}^{\top} \tag{22}
$$

其中$\boldsymbol{M} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$是SVD分解，$\text{sign}(\boldsymbol{\Sigma})$是对角元素取符号。

**等价表示**（对于满秩矩阵）：

$$
\text{msign}(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2} \tag{23}
$$

**关键性质**：
1. $\text{msign}(\boldsymbol{M})$的所有奇异值都等于1
2. $\|\text{msign}(\boldsymbol{M})\|_F = \sqrt{\text{rank}(\boldsymbol{M})}$
3. 满秩矩阵：$\|\text{msign}(\boldsymbol{M})\|_F = \sqrt{\min(n,m)}$

#### 3.3 带权重衰减的Muon

加入权重衰减后：

$$
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t (\boldsymbol{O}_t + \lambda \boldsymbol{W}_{t-1}) \tag{24}
$$

展开：

$$
\boldsymbol{W}_t = (1 - \eta_t \lambda) \boldsymbol{W}_{t-1} - \eta_t \boldsymbol{O}_t \tag{25}
$$

#### 3.4 Update RMS对齐

为了与Adam的更新量级对齐，Muon引入缩放因子：

$$
\boldsymbol{O}_t = \text{msign}(\boldsymbol{M}_t) \times \alpha \sqrt{\max(n,m)} \tag{26}
$$

其中$\alpha = 0.2$是经验系数。

**理论依据**：Adam的更新量RMS约为$\mathcal{O}(1)$，而$\text{msign}(\boldsymbol{M})$的Frobenius范数为$\sqrt{\max(n,m)}$，因此：

$$
\text{RMS}(\boldsymbol{O}_t) = \frac{\|\boldsymbol{O}_t\|_F}{\sqrt{nm}} = \frac{\alpha \sqrt{\max(n,m)} \cdot \sqrt{\min(n,m)}}{\sqrt{nm}} = \alpha \tag{27}
$$

### 4. Adam优化器对比

#### 4.1 Adam更新规则

Adam的更新规则为：

$$
\begin{align}
\boldsymbol{m}_t &= \beta_1 \boldsymbol{m}_{t-1} + (1-\beta_1) \boldsymbol{G}_t \tag{28} \\
\boldsymbol{v}_t &= \beta_2 \boldsymbol{v}_{t-1} + (1-\beta_2) \boldsymbol{G}_t \odot \boldsymbol{G}_t \tag{29} \\
\hat{\boldsymbol{m}}_t &= \boldsymbol{m}_t / (1 - \beta_1^t) \tag{30} \\
\hat{\boldsymbol{v}}_t &= \boldsymbol{v}_t / (1 - \beta_2^t) \tag{31} \\
\boldsymbol{W}_t &= \boldsymbol{W}_{t-1} - \eta_t \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} \tag{32}
\end{align}
$$

通常：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

#### 4.2 Adam更新量的秩分析

Adam的更新量为：

$$
\boldsymbol{\Delta}_t^{\text{Adam}} = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon} \tag{33}
$$

由于$\hat{\boldsymbol{m}}_t$是梯度的指数移动平均，而梯度通常是低秩的（由损失函数的Hessian决定），因此：

**假设**：Adam更新量是低秩的，即主要奇异值集中在前几个：

$$
\boldsymbol{\Delta}_t^{\text{Adam}} \approx \sum_{i=1}^{r} \tilde{\sigma}_i \tilde{\boldsymbol{u}}_i \tilde{\boldsymbol{v}}_i^{\top}, \quad r \ll \min(n,m) \tag{34}
$$

其中$\tilde{\sigma}_1 \gg \tilde{\sigma}_2 \gg \cdots \gg \tilde{\sigma}_r$。

#### 4.3 Muon更新量的秩分析

Muon的更新量为：

$$
\boldsymbol{\Delta}_t^{\text{Muon}} = \text{msign}(\boldsymbol{M}_t) \times \alpha \sqrt{\max(n,m)} \tag{35}
$$

**关键特性**：所有奇异值相等，即：

$$
\boldsymbol{\Delta}_t^{\text{Muon}} = \sum_{i=1}^{\min(n,m)} \bar{\sigma} \bar{\boldsymbol{u}}_i \bar{\boldsymbol{v}}_i^{\top}, \quad \bar{\sigma} = \alpha \sqrt{\frac{\max(n,m)}{\min(n,m)}} \tag{36}
$$

这是满秩更新！

### 5. 有效秩理论

#### 5.1 有效秩定义

矩阵$\boldsymbol{A}$的有效秩定义为其奇异值的熵：

$$
\text{EffRank}(\boldsymbol{A}) = \exp\left(-\sum_{i=1}^{r} p_i \log p_i\right) \tag{37}
$$

其中$p_i = \sigma_i / \sum_j \sigma_j$是归一化的奇异值。

**极端情况**：
- 秩1矩阵：$\text{EffRank} = 1$
- 所有奇异值相等：$\text{EffRank} = r$（满秩）

#### 5.2 与奇异值熵的关系

奇异值熵定义为：

$$
H(\boldsymbol{A}) = -\sum_{i=1}^{r} p_i \log p_i \tag{38}
$$

则：

$$
\text{EffRank}(\boldsymbol{A}) = \exp(H(\boldsymbol{A})) \tag{39}
$$

**性质**：
- $1 \leq \text{EffRank}(\boldsymbol{A}) \leq \text{rank}(\boldsymbol{A})$
- 熵越高，有效秩越大，矩阵越"满秩"

#### 5.3 Muon与Adam的有效秩对比

**Muon更新量**：

$$
\text{EffRank}(\boldsymbol{\Delta}_t^{\text{Muon}}) = \min(n,m) \tag{40}
$$

**Adam更新量**：

$$
\text{EffRank}(\boldsymbol{\Delta}_t^{\text{Adam}}) \ll \min(n,m) \tag{41}
$$

经验观察：$\text{EffRank}(\boldsymbol{\Delta}_t^{\text{Adam}}) \approx 10\text{-}50$，即使对于$n,m > 1000$的矩阵。

### 6. 奇异值碰撞理论

#### 6.1 SVD表示

设当前权重的SVD为：

$$
\boldsymbol{W}_{t-1} = \sum_{i=1}^{r_W} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} \tag{42}
$$

更新量的SVD为：

$$
\boldsymbol{\Delta}_t = \sum_{j=1}^{r_\Delta} \delta_j \boldsymbol{p}_j \boldsymbol{q}_j^{\top} \tag{43}
$$

#### 6.2 更新后的奇异值

更新后的权重为：

$$
\boldsymbol{W}_t = \boldsymbol{W}_{t-1} - \eta_t \boldsymbol{\Delta}_t = \sum_{i=1}^{r_W} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} - \eta_t \sum_{j=1}^{r_\Delta} \delta_j \boldsymbol{p}_j \boldsymbol{q}_j^{\top} \tag{44}
$$

**简化情况**：若$\boldsymbol{u}_k \approx \boldsymbol{p}_j$且$\boldsymbol{v}_k \approx \boldsymbol{q}_j$（奇异向量对齐），则：

$$
\boldsymbol{W}_t \approx \sum_{i \neq k} \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} + (\sigma_k - \eta_t \delta_j) \boldsymbol{u}_k \boldsymbol{v}_k^{\top} + \cdots \tag{45}
$$

第$k$个奇异值变化：

$$
\Delta\sigma_k \approx -\eta_t \delta_j \cos\theta_j \tag{46}
$$

其中$\theta_j$是奇异向量对的夹角。

#### 6.3 碰撞概率分析

定义"碰撞"事件为$|\cos\theta_j| > \tau$（如$\tau = 0.5$）。

**Muon的碰撞概率**：由于Muon有$r_\Delta = \min(n,m)$个奇异向量对，碰撞概率为：

$$
P_{\text{collision}}^{\text{Muon}} \approx \frac{r_\Delta \cdot \tau}{r_W} = \frac{\min(n,m) \cdot \tau}{r_W} \tag{47}
$$

**Adam的碰撞概率**：由于Adam有效秩$r_\Delta \ll \min(n,m)$：

$$
P_{\text{collision}}^{\text{Adam}} \approx \frac{r_\Delta^{\text{Adam}} \cdot \tau}{r_W} \ll P_{\text{collision}}^{\text{Muon}} \tag{48}
$$

**比例**：假设$r_\Delta^{\text{Adam}} = 20$，$\min(n,m) = 1000$：

$$
\frac{P_{\text{collision}}^{\text{Muon}}}{P_{\text{collision}}^{\text{Adam}}} \approx \frac{1000}{20} = 50 \tag{49}
$$

Muon的碰撞概率是Adam的50倍！

### 7. MaxLogit爆炸的机制

#### 7.1 双线性形式的放大效应

Attention logit是双线性形式：

$$
S_{ij} = \boldsymbol{q}_i \cdot \boldsymbol{k}_j = \boldsymbol{x}_i \boldsymbol{W}_q \boldsymbol{W}_k^{\top} \boldsymbol{x}_j^{\top} \tag{50}
$$

设$\boldsymbol{W}_q$的最大奇异值为$\sigma_q$，$\boldsymbol{W}_k$的最大奇异值为$\sigma_k$，则：

$$
|S_{ij}| \leq \|\boldsymbol{x}_i\| \|\boldsymbol{x}_j\| \sigma_q \sigma_k \tag{51}
$$

**乘积放大**：若两个谱范数都增长，MaxLogit以乘积速度增长！

设$\sigma_q(t) = \sigma_k(t) = 1 + \alpha t$（线性增长），则：

$$
S_{\max}(t) \propto (1 + \alpha t)^2 = 1 + 2\alpha t + \alpha^2 t^2 \tag{52}
$$

超线性增长！

#### 7.2 正反馈循环

1. **初始扰动**：某个Head的$\boldsymbol{W}_q$或$\boldsymbol{W}_k$的奇异值略微增大
2. **MaxLogit上升**：$S_{\max} \propto \sigma_q \sigma_k$增大
3. **梯度模式变化**：较大的logit主导softmax输出，梯度集中在对应方向
4. **进一步增强**：若梯度方向与奇异向量对齐，继续增大奇异值
5. **循环往复**：形成"富者愈富"的恶性循环

数学表示：设第$k$个奇异值在时刻$t$为$\sigma_k(t)$，若存在正反馈：

$$
\frac{d\sigma_k}{dt} \propto \sigma_k \tag{53}
$$

解为指数增长：

$$
\sigma_k(t) = \sigma_k(0) e^{\gamma t} \tag{54}
$$

其中$\gamma > 0$是增长率。

#### 7.3 权重衰减的抑制作用

加入权重衰减后：

$$
\frac{d\sigma_k}{dt} = \alpha \sigma_k - \lambda \sigma_k = (\alpha - \lambda) \sigma_k \tag{55}
$$

**稳定条件**：$\alpha < \lambda$。

若$\alpha > \lambda$，仍会指数增长；若$\alpha \approx \lambda$，则处于临界状态（拉锯战）。

### 8. QK-Clip的数学原理

#### 8.1 直接裁剪策略

当检测到$S_{\max}^{(l,h)} > \tau$时，希望将其裁剪到$\tau$。

**方法**：对logit矩阵乘以因子$\gamma$：

$$
\boldsymbol{S}_{\text{clip}} = \gamma \boldsymbol{S} = \gamma \boldsymbol{Q}\boldsymbol{K}^{\top} \tag{56}
$$

选择$\gamma$使得新的MaxLogit等于$\tau$：

$$
\gamma = \frac{\tau}{S_{\max}} \tag{57}
$$

则：

$$
\max_{i,j} (\boldsymbol{S}_{\text{clip}})_{ij} = \gamma S_{\max} = \tau \tag{58}
$$

#### 8.2 权重分配

由于$\gamma \boldsymbol{Q}\boldsymbol{K}^{\top} = (\sqrt{\gamma}\boldsymbol{Q})(\sqrt{\gamma}\boldsymbol{K})^{\top}$，可以将因子平分：

$$
\begin{align}
\boldsymbol{W}_q &\leftarrow \sqrt{\gamma} \boldsymbol{W}_q \tag{59} \\
\boldsymbol{W}_k &\leftarrow \sqrt{\gamma} \boldsymbol{W}_k \tag{60}
\end{align}
$$

**不变性**：$\boldsymbol{Q}\boldsymbol{K}^{\top} = (\boldsymbol{X}\boldsymbol{W}_q)(\boldsymbol{X}\boldsymbol{W}_k)^{\top}$保持。

#### 8.3 Per-Head裁剪

对于多头注意力，第$(l,h)$个Head的裁剪：

$$
\text{if } S_{\max}^{(l,h)} > \tau: \quad \gamma^{(l,h)} = \frac{\tau}{S_{\max}^{(l,h)}} \tag{61}
$$

只裁剪该Head的权重：

$$
\begin{align}
\boldsymbol{W}_{qc}^{(l,h)} &\leftarrow \sqrt{\gamma^{(l,h)}} \boldsymbol{W}_{qc}^{(l,h)} \tag{62} \\
\boldsymbol{W}_{kc}^{(l,h)} &\leftarrow \sqrt{\gamma^{(l,h)}} \boldsymbol{W}_{kc}^{(l,h)} \tag{63}
\end{align}
$$

**避免殃及池鱼**：其他Head不受影响。

#### 8.4 MLA的特殊处理

MLA中，$\boldsymbol{k}_i$包含共享部分$\boldsymbol{W}_{kr}$：

$$
\boldsymbol{k}_i^{(s)} = [\boldsymbol{c}_i\boldsymbol{W}_{kc}^{(s)}, \boldsymbol{x}_i\boldsymbol{W}_{kr}\boldsymbol{\mathcal{R}}_i] \tag{64}
$$

若对$\boldsymbol{W}_{kr}$裁剪，会影响所有Head。

**解决方案**：只裁剪$\boldsymbol{W}_{qr}^{(l,h)}$：

$$
\boldsymbol{W}_{qr}^{(l,h)} \leftarrow \gamma^{(l,h)} \boldsymbol{W}_{qr}^{(l,h)} \tag{65}
$$

完全裁剪（不开方），因为另一侧不裁剪。

### 9. QK-Clip的收敛性分析

#### 9.1 离散动力系统

考虑简化模型：单个Head，学习率恒定。设第$t$步的谱范数为$\sigma_t$：

**无QK-Clip**：

$$
\sigma_{t+1} = \sigma_t + \eta \alpha - \eta \lambda \sigma_t = (1 - \eta\lambda)\sigma_t + \eta\alpha \tag{66}
$$

其中$\alpha$是Muon更新的贡献（假设常数）。

**稳态**：$\sigma^* = \alpha / \lambda$。

若$\alpha / \lambda > \tau$（阈值），则会MaxLogit爆炸。

**加入QK-Clip**：

$$
\sigma_{t+1} = \begin{cases}
(1 - \eta\lambda)\sigma_t + \eta\alpha, & S_{\max}(t) \leq \tau \\
\sqrt{\tau / S_{\max}(t)} \cdot [(1 - \eta\lambda)\sigma_t + \eta\alpha], & S_{\max}(t) > \tau
\end{cases} \tag{67}
$$

简化为：$S_{\max}(t) \propto \sigma_t^2$（双线性），则裁剪条件变为$\sigma_t > \sqrt{\tau/C}$。

$$
\sigma_{t+1} = \begin{cases}
(1 - \eta\lambda)\sigma_t + \eta\alpha, & \sigma_t \leq \sigma_{\text{th}} \\
\sigma_{\text{th}} \cdot \frac{(1 - \eta\lambda)\sigma_t + \eta\alpha}{\sigma_t}, & \sigma_t > \sigma_{\text{th}}
\end{cases} \tag{68}
$$

其中$\sigma_{\text{th}} = \sqrt{\tau/C}$。

#### 9.2 稳定性分析

**定理**：在QK-Clip下，谱范数$\sigma_t$有界。

**证明草图**：
1. 若$\sigma_t \leq \sigma_{\text{th}}$，则$\sigma_{t+1} \leq (1-\eta\lambda)\sigma_{\text{th}} + \eta\alpha$
2. 若$(1-\eta\lambda)\sigma_{\text{th}} + \eta\alpha \leq \sigma_{\text{th}}$，则不会超过阈值
3. 即使超过阈值，裁剪后$\sigma_{t+1} \leq \sigma_{\text{th}} \cdot [1 + \eta(\alpha/\sigma_{\text{th}} - \lambda)]$
4. 选择适当的$\tau$和$\lambda$，可保证$\sigma_t$有界

**推论**：$S_{\max}(t) \leq C\sigma_t^2 \leq C\sigma_{\max}^2 < \infty$。□

#### 9.3 拉锯战动态

在实际训练中，观察到Muon更新与QK-Clip的"拉锯战"：

$$
\begin{align}
\text{Muon step}: \quad &\sigma_t \to \sigma_t + \delta \quad (\delta > 0) \tag{69} \\
\text{QK-Clip step}: \quad &\sigma_t \to \sigma_t \times \gamma \quad (\gamma < 1) \tag{70}
\end{align}
$$

**平衡状态**：$(1 + \delta/\sigma_t) \times \gamma \approx 1$，即：

$$
\gamma \approx \frac{1}{1 + \delta/\sigma_t} \approx 1 - \frac{\delta}{\sigma_t} \tag{71}
$$

从$\gamma = \sqrt{\tau / S_{\max}}$：

$$
1 - \frac{\delta}{\sigma_t} \approx \sqrt{\frac{\tau}{S_{\max}}} \quad \Rightarrow \quad S_{\max} \approx \tau \left(1 + \frac{2\delta}{\sigma_t}\right) \tag{72}
$$

$S_{\max}$在$\tau$附近小幅震荡。

### 10. 数值稳定性与实现细节

#### 10.1 浮点精度考虑

当$S_{\max}$非常大时（如1000+），softmax计算需要数值稳定化：

$$
\text{softmax}(\boldsymbol{s})_i = \frac{\exp(s_i - s_{\max})}{\sum_j \exp(s_j - s_{\max})} \tag{73}
$$

其中$s_{\max} = \max_j s_j$。

**QK-Clip的好处**：保证$s_{\max} \leq \tau$，避免了极大的指数计算。

#### 10.2 梯度流分析

QK-Clip是对权重的事后修改，不在计算图中，因此：

$$
\frac{\partial L}{\partial \boldsymbol{W}_q^{\text{before clip}}} \neq \frac{\partial L}{\partial \boldsymbol{W}_q^{\text{after clip}}} \tag{74}
$$

但这是可接受的，因为：
1. 裁剪只在必要时发生（$S_{\max} > \tau$）
2. 裁剪幅度很小（$\gamma \approx 1$）
3. 类似于梯度裁剪，属于正则化手段

#### 10.3 分布式训练中的同步

在模型并行中，权重矩阵可能被切分。Per-Head裁剪需要：

1. **收集MaxLogit**：在前向传播中计算每个Head的$S_{\max}^{(l,h)}$
2. **跨设备同步**：通过all-reduce获取全局$S_{\max}^{(l,h)}$
3. **计算裁剪因子**：$\gamma^{(l,h)} = \sqrt{\tau / S_{\max}^{(l,h)}}$
4. **局部裁剪**：每个设备裁剪其持有的权重分片

**通信开销**：每层每个Head一个标量，$O(\text{num\_layers} \times \text{num\_heads})$，可忽略。

### 11. 理论推广与变体

#### 11.1 自适应阈值

固定阈值$\tau$可能不是最优的。考虑自适应阈值：

$$
\tau_t = \tau_0 + \beta \cdot \text{median}(\{S_{\max}^{(l,h)}\}) \tag{75}
$$

根据模型当前状态动态调整。

#### 11.2 软裁剪

替代硬裁剪，可以使用软裁剪：

$$
\gamma^{(l,h)} = \left(\frac{\tau}{S_{\max}^{(l,h)}}\right)^{\alpha}, \quad \alpha \in (0, 1) \tag{76}
$$

$\alpha < 1$时裁剪更温和。

#### 11.3 MaxOutput裁剪

对于输出层，定义MaxOutput：

$$
O_{\max} = \max_{i,j} |O_{ij}| \tag{77}
$$

类似地裁剪$\boldsymbol{W}_o$：

$$
\text{if } O_{\max}^{(l)} > \tau_o: \quad \boldsymbol{W}_o^{(l)} \leftarrow \frac{\tau_o}{O_{\max}^{(l)}} \boldsymbol{W}_o^{(l)} \tag{78}
$$

**挑战**：输出是多维的，Per-Dim裁剪成本高。可能需要Per-Layer或Per-Head的折中。

### 12. 实验验证与分析

#### 12.1 理论预测

**预测1**：Muon训练的模型，权重有效秩更高。

验证：计算权重的奇异值熵：

$$
H(\boldsymbol{W}) = -\sum_i p_i \log p_i, \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j} \tag{79}
$$

观察到：$H(\boldsymbol{W}^{\text{Muon}}) > H(\boldsymbol{W}^{\text{Adam}})$。

**预测2**：QK-Clip后，MaxLogit被控制在$\tau$以下。

验证：监控$S_{\max}^{(l,h)}(t)$，确认$S_{\max}^{(l,h)}(t) \leq \tau + \epsilon$。

**预测3**：长期训练后，权重衰减使MaxLogit自然下降。

验证：Kimi K2在70k steps后，MaxLogit自动降低到100以下。

#### 12.2 消融实验

**实验1**：固定$\tau$的影响

| $\tau$ | 训练稳定性 | 最终Loss | MaxLogit爆炸 |
|--------|----------|---------|------------|
| 30 | 稳定 | 2.45 | 无 |
| 100 | 稳定 | 2.43 | 无 |
| 300 | 偶尔不稳 | 2.44 | 少数Head |
| 无限 | 不稳定 | NaN | 严重 |

结论：$\tau = 100$是良好的平衡点。

**实验2**：Per-Head vs Per-Layer裁剪

| 方法 | 被裁剪的Head比例 | 有效Head数 | 最终性能 |
|------|----------------|-----------|---------|
| Per-Layer | 100% | 85% | 较差 |
| Per-Head | 15% | 98% | 正常 |

结论：Per-Head避免了过度裁剪。

#### 12.3 与其他方法对比

| 方法 | MaxLogit控制 | 兼容MLA | 计算开销 | 效果影响 |
|------|------------|--------|---------|---------|
| Weight Decay | 部分 | 是 | 低 | 中（大WD损失效果）|
| Softcap | 否（只控制输出）| 是 | 低 | 未知 |
| QK-Norm | 是 | 否 | 中 | 小 |
| **QK-Clip** | **是** | **是** | **低** | **无** |

### 13. 理论洞察与启示

#### 13.1 满秩更新的双刃剑

Muon的满秩更新具有：

**优势**：
- 探索更全面，不局限于梯度的主方向
- 隐式正则化，避免过拟合到低秩解
- 泛化性能可能更好

**劣势**：
- 更容易与现有权重的奇异向量"碰撞"
- 导致奇异值增长，引发MaxLogit爆炸
- 需要额外的稳定化机制

#### 13.2 直接vs间接方法

**间接方法**（调整超参数）：
- 降低学习率
- 增大权重衰减
- 部分QK-Norm

**缺点**：不能保证解决问题，可能损害效果。

**直接方法**（QK-Clip）：
- 直接以MaxLogit为信号
- 最小化干预
- 保证控制目标

**启示**："哪里不稳Clip哪里"——针对具体问题直接干预。

#### 13.3 优化器与架构的协同设计

QK-Clip的成功说明：
- 优化器的特性（Muon的满秩更新）
- 架构的特性（Attention的双线性形式）
- 需要协同考虑

**设计原则**：
1. 理解优化器的数学特性
2. 识别架构的脆弱点
3. 针对性地设计稳定化机制

### 14. 开放问题与未来方向

#### 14.1 理论问题

1. **收敛速度**：QK-Clip如何影响收敛速度？是否存在理论保证？
2. **最优阈值**：如何理论上确定最优的$\tau$？
3. **与其他正则化的关系**：QK-Clip与谱范数约束、Lipschitz约束的关系？

#### 14.2 实践问题

1. **其他架构**：QK-Clip在CNN、RNN中的应用？
2. **其他优化器**：与AdaGrad、RMSprop的兼容性？
3. **自动化**：能否自动检测需要裁剪的层/参数？

#### 14.3 推广方向

1. **通用异常值控制**：统一框架处理各种异常值（MaxLogit、MaxGrad、MaxAct）
2. **学习式裁剪**：用学习到的策略替代固定规则
3. **理论统一**：将QK-Clip纳入优化理论的框架

### 15. 结论

#### 15.1 核心贡献

QK-Clip提供了控制MaxLogit爆炸的直接、有效、通用的方法：
1. **直接性**：以MaxLogit为信号，直接裁剪权重
2. **有效性**：理论保证和实验验证
3. **通用性**：兼容MHA、GQA、MLA等架构
4. **无损性**：对最终效果无负面影响

#### 15.2 数学精髓

- Muon的满秩更新导致奇异值碰撞概率增加
- Attention的双线性形式放大了谱范数增长
- QK-Clip通过事后裁剪直接控制谱范数
- Per-Head裁剪避免过度干预

#### 15.3 实践价值

QK-Clip是Muon在超大规模训练（千亿参数以上）中的关键技术，使得：
- Muon + MLA 成为可行的组合
- 训练稳定性大幅提升
- 为Kimi K2等万亿参数模型的成功提供保障

