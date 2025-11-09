---
title: 熵不变性Softmax的一个快速推导
slug: 熵不变性softmax的一个快速推导
date: 
source: https://spaces.ac.cn/archives/9034
tags: 近似, 熵, attention, 生成模型, attention
status: pending
---

# 熵不变性Softmax的一个快速推导

**原文链接**: [https://spaces.ac.cn/archives/9034](https://spaces.ac.cn/archives/9034)

**发布日期**: 

---

在文章[《从熵不变性看Attention的Scale操作》](/archives/8823)中，我们推导了一版具有熵不变性质的注意力机制：  
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\kappa \log n}{d}QK^{\top}\right)V\label{eq:a}\end{equation}  
可以观察到，它主要是往Softmax里边引入了长度相关的缩放因子$\log n$来实现的。原来的推导比较繁琐，并且做了较多的假设，不利于直观理解，本文为其补充一个相对简明快速的推导。

## 推导过程 #

我们可以抛开注意力机制的背景，直接设有$s_1,s_2,\cdots,s_n\in\mathbb{R}$，定义  
$$p_i = \frac{e^{\lambda s_i}}{\sum\limits_{i=1}^n e^{\lambda s_i}}$$  
显然这就是$s_1,s_2,\cdots,s_n$同时乘上缩放因子$\lambda$后做Softmax的结果。现在我们算它的熵  
\begin{equation}\begin{aligned}H =&\, -\sum_{i=1}^n p_i \log p_i = \log\sum_{i=1}^n e^{\lambda s_i} - \lambda\sum_{i=1}^n p_i s_i \\\  
=&\, \log n + \log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i} - \lambda\sum_{i=1}^n p_i s_i  
\end{aligned}\end{equation}  
第一项的$\log$里边是“先指数后平均”，我们用“先平均后指数”（平均场）来近似它：  
\begin{equation}  
\log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i}\approx \log\exp\left(\frac{1}{n}\sum_{i=1}^n \lambda s_i\right) = \lambda \bar{s}  
\end{equation}  
然后我们知道Softmax是会侧重于$\max$的那个（参考[《函数光滑化杂谈：不可导函数的可导逼近》](/archives/6620#softmax)），所以有近似  
\begin{equation}\lambda\sum_{i=1}^n p_i s_i \approx \lambda s_{\max}\end{equation}  
所以  
\begin{equation}H\approx \log n - \lambda(s_{\max} - \bar{s})\end{equation}  
所谓熵不变性，就是希望尽可能地消除长度$n$的影响，所以根据上式我们需要有$\lambda\propto \log n$。如果放到注意力机制中，那么$s$的形式为$\langle \boldsymbol{q}, \boldsymbol{k}\rangle\propto d$（$d$是向量维度），所以需要有$\lambda\propto \frac{1}{d}$，综合起来就是  
\begin{equation}\lambda\propto \frac{\log n}{d}\end{equation}  
这就是文章开头式$\eqref{eq:a}$的结果。

## 文章小结 #

为之前提出的“熵不变性Softmax”构思了一个简单明快的推导。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9034>_

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

苏剑林. (Apr. 11, 2022). 《熵不变性Softmax的一个快速推导 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9034>

@online{kexuefm-9034,  
title={熵不变性Softmax的一个快速推导},  
author={苏剑林},  
year={2022},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9034}},  
} 


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

