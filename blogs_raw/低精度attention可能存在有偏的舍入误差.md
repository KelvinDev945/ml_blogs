---
title: 低精度Attention可能存在有偏的舍入误差
slug: 低精度attention可能存在有偏的舍入误差
date: 2025-10-27
source: https://spaces.ac.cn/archives/11371
tags: 机器学习
status: pending
---

# 低精度Attention可能存在有偏的舍入误差

**原文链接**: [https://spaces.ac.cn/archives/11371](https://spaces.ac.cn/archives/11371)

**发布日期**: 2025-10-27

---

前段时间笔者在arXiv上刷到了论文[《Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention》](https://arxiv.org/abs/2510.04212)，里面描述的实验现象跟我们在训练[Kimi K2](https://arxiv.org/abs/2507.20534)时出现的一些现象很吻合，比如都是第二层Attention开始出现问题。论文将其归因为低精度Attention固有的有偏误差，这个分析角度是比较出乎笔者意料的，所以饶有兴致地阅读了一番。

然而，论文的表述似乎比较让人费解——当然也有笔者本就不大熟悉低精度运算的原因。总之，经过多次向作者请教后，笔者才勉强看懂论文，遂将自己的理解记录在此，供大家参考。

## 结论简述

要指出的是，论文标题虽然点名了“Flash Attention”，但按照论文的描述，即便block_size取到训练长度那么大，相同的问题依然会出现，所以Flash Attention的分块计算并不是引起问题的原因，因此我们可以按照朴素的低精度Attention实现来简化分析。

[[...]](https://spaces.ac.cn/archives/11371 "低精度Attention可能存在有偏的舍入误差")


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

