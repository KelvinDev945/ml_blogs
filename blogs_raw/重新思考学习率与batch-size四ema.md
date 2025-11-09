---
title: 重新思考学习率与Batch Size（四）：EMA
slug: 重新思考学习率与batch-size四ema
date: 2025-09-22
source: https://spaces.ac.cn/archives/11301
tags: 优化
status: pending
---

# 重新思考学习率与Batch Size（四）：EMA

**原文链接**: [https://spaces.ac.cn/archives/11301](https://spaces.ac.cn/archives/11301)

**发布日期**: 2025-09-22

---

我们在[《重新思考学习率与Batch Size（二）：平均场》](https://kexue.fm/archives/11280)中提到，关注SignSGD的原因之一是我们通常将它作为Adam的理论近似，这是Adam做理论分析时常用的简化策略。除了分析学习率的场景外，在[《配置不同的学习率，LoRA还能再涨一点？》](https://kexue.fm/archives/10001)、[《初探MuP：超参数的跨模型尺度迁移规律》](https://kexue.fm/archives/10770)等地方我们也用了这个简化。

然而，SignSGD真是Adam的良好近似吗？一个明显差异是SignSGD的Update RMS总是1，而Adam并非如此。笔者发现，导致这一差异的核心原因是动量，它普遍存在于Adam、Lion、Muon等优化器中。所以，本文我们来考察动量——更广义地说是EMA——的影响。

## 问题分析

从Adam的视角看，SignSGD对应$\beta_1=\beta_2=0$这个特例，或者对应于Adam的第一步更新量（不管$\beta_1,\beta_2$如何）。因此，我们认为它跟Adam肯定有一些共性，能够捕捉到一些通用的规律。

[[...]](https://spaces.ac.cn/archives/11301 "重新思考学习率与Batch Size（四）：EMA")


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

