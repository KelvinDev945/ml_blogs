---
title: Cool Papers浏览器扩展升级至v0.2.0
slug: cool-papers浏览器扩展升级至v020
date: 2024-10-16
tags: 网站, 论文, 酷论文, 生成模型, attention
status: pending
---

# Cool Papers浏览器扩展升级至v0.2.0

**原文链接**: [https://spaces.ac.cn/archives/10480](https://spaces.ac.cn/archives/10480)

**发布日期**: 

---

年初，我们在[《更便捷的Cool Papers打开方式：Chrome重定向扩展》](/archives/9978)中发布了一个Chrome浏览器插件（Cool Papers Redirector v0.1.0），可以通过右击菜单从任意页面中重定向到Cool Papers中，让大家更方便地获取Kimi对论文的理解。前几天我们把该插件升级到了v0.2.0，并顺利上架到了Chrome应用商店中，遂在此向大家推送一下。

## 更新汇总 #

相比旧版v0.1.0，当前版v0.2.0的主要更新内容如下：

> 1、右键菜单跳转改为在新标签页打开；
> 
> 2、右键菜单支持同时访问多个论文ID；
> 
> 3、右键菜单支持PDF页面；
> 
> 4、右键菜单新增更多论文源（arXiv、OpenReview、ACL、IJCAI、PMLR）；
> 
> 5、右键菜单在搜索不到论文ID时，转入站内搜索（即划词搜索）；
> 
> 6、在某些网站的适当位置插入快捷跳转链接（arXiv、OpenReview，ACL）。

下面对更新内容展开介绍一下。

## 右键菜单 #

前面五点更新都是关于右键菜单的，这也是Cool Papers扩展程序最主要的功能，它的工作逻辑是：依次检查用户右击鼠标时的所选文字、超链接、页面路径中是否包含论文ID，如果包含则跳转到Cool Papers相应页面。

v0.1.0版写在Cool Papers建立之初，当时只支持识别单一的arXiv论文ID，现在支持识别多个论文ID在同一页打开，并且支持arXiv、OpenReview、ACL、IJCAI、PMLR五个论文源（后面再陆续增加，这几个是图方便先补充上去）。还有一点比较关键的改进，就是之前在PDF页面是无法进行跳转的，比如你访问了arXiv的PDF如“https://arxiv/org/pdf/xxxx.xxxxx”，那么右击时就会发现“Redirect to Cool Papers”根本不显示，这是旧版遗留的一个问题，新版把它解决了。

最后，如果在所选文字、超链接、页面路径中都检测不到论文ID，那么就直接以所选文字为Query，跳转到Cool Papers的站内搜索页，这时候就变成了一个划词搜索功能了。

## 页面链接 #

最后一点变化，是新引入的除右键菜单外的功能，它在某些页面的适当位置插入了可以跳转到Cool Papers的超链接，这些页面包括arXiv列表页、arXiv详情页、OpenReView全站、ACL详情页。希望这个改动不会影响大家浏览原本的页面，如有打扰，请及时指正。

[![arXiv列表页的快捷链接](/usr/uploads/2024/10/3920073027.png)](/usr/uploads/2024/10/3920073027.png "点击查看原图")

arXiv列表页的快捷链接

[![arXiv详情页的快捷链接](/usr/uploads/2024/10/1893966955.png)](/usr/uploads/2024/10/1893966955.png "点击查看原图")

arXiv详情页的快捷链接

[![OpenReview的快捷链接](/usr/uploads/2024/10/3091733289.png)](/usr/uploads/2024/10/3091733289.png "点击查看原图")

OpenReview的快捷链接

[![ACL详情页的快捷链接](/usr/uploads/2024/10/3569796965.png)](/usr/uploads/2024/10/3569796965.png "点击查看原图")

ACL详情页的快捷链接

## 文章小结 #

本文根据作者自己的使用经验简单升级了一下Cool Papers Redirector浏览器拓展，欢迎大家测试。如果有更多改进意见，欢迎留言指出。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/10480>_

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

苏剑林. (Oct. 16, 2024). 《Cool Papers浏览器扩展升级至v0.2.0 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/10480>

@online{kexuefm-10480,  
title={Cool Papers浏览器扩展升级至v0.2.0},  
author={苏剑林},  
year={2024},  
month={Oct},  
url={\url{https://spaces.ac.cn/archives/10480}},  
} 


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

