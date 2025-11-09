---
title: AdamW的Weight RMS的渐近估计
slug: adamw的weight-rms的渐近估计
date: 2025-10-01
source: https://spaces.ac.cn/archives/11307
tags: 优化
status: pending
---

# AdamW的Weight RMS的渐近估计

**原文链接**: [https://spaces.ac.cn/archives/11307](https://spaces.ac.cn/archives/11307)

**发布日期**: 2025-10-01

---

在[《为什么Adam的Update RMS是0.2？》](https://kexue.fm/archives/11267)中，我们用平均场近似估计了Adam的Update RMS。不久后，读者 [@EIFY](https://x.com/EIFY/status/1965888629814988984) 指出相同的结果已经出现在论文[《Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks》](https://arxiv.org/abs/2305.17212)中。阅读后，笔者发现其中不仅包含了Update RMS的估计，还包含了Weight RMS的估计。

也就是说，AdamW训出来的模型，其权重的RMS是可以事先估计出来一个渐近结果的。大家会不会觉得这个结论有点意外？反正笔者第一次看到它是颇为意外的，直觉上权重模长是模型根据训练集自己学出来的，结果它告诉我这已经隐藏在优化器的超参中，可谓很反直觉了。

这篇文章我们还是用平均场近似方法，来复现对Weight RMS的渐近估计。

[[...]](https://spaces.ac.cn/archives/11307 "AdamW的Weight RMS的渐近估计")


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

