---
title: DiVeQ：一种非常简洁的VQ训练方案
slug: diveq一种非常简洁的vq训练方案
date: 2025-10-08
source: https://spaces.ac.cn/archives/11328
tags: 机器学习
status: pending
---

# DiVeQ：一种非常简洁的VQ训练方案

**原文链接**: [https://spaces.ac.cn/archives/11328](https://spaces.ac.cn/archives/11328)

**发布日期**: 2025-10-08

---

对于坚持离散化路线的研究人员来说，VQ（Vector Quantization）是视觉理解和生成的关键部分，担任着视觉中的“Tokenizer”的角色。它提出在2017年的论文[《Neural Discrete Representation Learning》](https://arxiv.org/abs/1711.00937)，笔者在2019年的博客[《VQ-VAE的简明介绍：量子化自编码器》](https://kexue.fm/archives/6760)也介绍过它。

然而，这么多年过去了，我们可以发现VQ的训练技术几乎没有变化，都是STE（Straight-Through Estimator）加额外的Aux Loss。STE倒是没啥问题，它可以说是给离散化运算设计梯度的标准方式了，但Aux Loss的存在总让人有种不够端到端的感觉，同时还引入了额外的超参要调。

幸运的是，这个局面可能要结束了，上周的论文[《DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick》](https://arxiv.org/abs/2509.26469)提出了一个新的STE技巧，它最大亮点是不需要Aux Loss，这让它显得特别简洁漂亮！

[[...]](https://spaces.ac.cn/archives/11328 "DiVeQ：一种非常简洁的VQ训练方案")


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

