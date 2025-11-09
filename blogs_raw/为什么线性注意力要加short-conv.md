---
title: 为什么线性注意力要加Short Conv？
slug: 为什么线性注意力要加short-conv
date: 2025-10-05
source: https://spaces.ac.cn/archives/11320
tags: 机器学习
status: pending
---

# 为什么线性注意力要加Short Conv？

**原文链接**: [https://spaces.ac.cn/archives/11320](https://spaces.ac.cn/archives/11320)

**发布日期**: 2025-10-05

---

如果读者有关注模型架构方面的进展，那么就会发现，比较新的线性Attention（参考[《线性注意力简史：从模仿、创新到反哺》](https://kexue.fm/archives/11033)）模型都给$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$加上了Short Conv，比如下图所示的[DeltaNet](https://arxiv.org/abs/2406.06484)：  
[![DeltaNet中的Short Conv.png](https://kexue.fm/usr/uploads/2025/10/175536171.png)](https://kexue.fm/usr/uploads/2025/10/175536171.png "点击查看原图")

为什么要加这个Short Conv呢？直观理解可能是增加模型深度、增强模型的Token-Mixing能力等，说白了就是补偿线性化导致的表达能力下降。这个说法当然是大差不差，但它属于“万能模版”式的回答，我们更想对它的生效机制有更准确的认知。

接下来，笔者将给出自己的一个理解（更准确说应该是猜测）。

[[...]](https://spaces.ac.cn/archives/11320 "为什么线性注意力要加Short Conv？")


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

