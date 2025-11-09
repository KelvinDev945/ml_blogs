---
title: ReLU/GeLU/Swish的一个恒等式
slug: relugeluswish的一个恒等式
date: 
source: https://spaces.ac.cn/archives/11233
tags: 分析, 神经网络, 恒等式, 生成模型, attention
status: pending
---

# ReLU/GeLU/Swish的一个恒等式

**原文链接**: [https://spaces.ac.cn/archives/11233](https://spaces.ac.cn/archives/11233)

**发布日期**: 

---

今天水一点轻松的内容，它基于笔者这两天意识到的一个恒等式。这个恒等式实际上很简单，但初看之下会有点意料之外的感觉，所以来记录一下。

## 基本结果 #

我们知道$\newcommand{relu}{\mathop{\text{relu}}}\relu(x) = \max(x, 0)$，容易证明如下恒等式  
\begin{equation}x = \relu(x) - \relu(-x)\end{equation}  
如果$x$是一个向量，那么上式就更直观了，$\relu(x)$是提取出$x$的正分量，$- \relu(-x)$是提取出$x$的负分量，两者相加就得到原本的向量。

## 一般结论 #

接下来的问题是[GeLU](/archives/7309)、[Swish](https://papers.cool/arxiv/1710.05941)等激活函数成立类似的恒等式吗？初看之下并不成立，然而事实上是成立的！我们甚至还有更一般的结论：

> 设$\phi(x)$是任意奇函数，$f(x)=\frac{1}{2}(\phi(x) + 1)x$，那么恒成立 \begin{equation}x = f(x) - f(-x)\end{equation} 

证明该结论也是一件很轻松的事，这里就不展开了。对于Swish来说我们有$\phi(x) = \tanh(\frac{x}{2})$，对于GeLU来说则有$\phi(x)=\mathop{\text{erf}}(\frac{x}{\sqrt{2}})$，它们都是奇函数，所以成立同样的恒等式。

## 意义思考 #

上述恒等式写成矩阵形式是  
\begin{equation}x = f(x) - f(-x) = f(x[1, -1])\begin{bmatrix}1 \\\ -1\end{bmatrix}\end{equation}  
这表明以ReLU、GeLU、Swish等为激活函数时，两层神经网络有退化为一层的能力，这意味着它们可以自适应地调节模型的实际深度，这与ResNet的工作原理异曲同工，这也许是这些激活函数为什么比传统的Tanh、Sigmoid等更好的原因之一。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11233>_

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

苏剑林. (Aug. 16, 2025). 《ReLU/GeLU/Swish的一个恒等式 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11233>

@online{kexuefm-11233,  
title={ReLU/GeLU/Swish的一个恒等式},  
author={苏剑林},  
year={2025},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/11233}},  
} 


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

