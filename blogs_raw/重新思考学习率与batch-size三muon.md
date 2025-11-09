---
title: 重新思考学习率与Batch Size（三）：Muon
slug: 重新思考学习率与batch-size三muon
date: 2025-09-15
source: https://spaces.ac.cn/archives/11285
tags: 优化
status: pending
---

# 重新思考学习率与Batch Size（三）：Muon

**原文链接**: [https://spaces.ac.cn/archives/11285](https://spaces.ac.cn/archives/11285)

**发布日期**: 2025-09-15

---

前两篇文章[《重新思考学习率与Batch Size（一）：现状》](https://kexue.fm/archives/11260)和[《重新思考学习率与Batch Size（二）：平均场》](https://kexue.fm/archives/11280)中，我们主要是提出了平均场方法，用以简化学习率与Batch Size的相关计算。当时我们分析的优化器是SGD、SignSGD和SoftSignSGD，并且主要目的是简化，本质上没有新的结论。

然而，在如今的优化器盛宴中，怎能少得了Muon的一席之地呢？所以，这篇文章我们就来尝试计算Muon的相关结论，看看它的学习率与Batch Size的关系是否会呈现出新的规律。

## 基本记号

众所周知，[Muon](https://kexue.fm/archives/10592)的主要特点就是非Element-wise的更新规则，所以之前在[《当Batch Size增大时，学习率该如何随之变化？》](https://kexue.fm/archives/10542)和[《Adam的epsilon如何影响学习率的Scaling Law？》](https://kexue.fm/archives/10563)的Element-wise的计算方法将完全不可用。但幸运的是，上篇文章介绍的平均场依然好使，只需要稍微调整一下细节。

[[...]](https://spaces.ac.cn/archives/11285 "重新思考学习率与Batch Size（三）：Muon")


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

