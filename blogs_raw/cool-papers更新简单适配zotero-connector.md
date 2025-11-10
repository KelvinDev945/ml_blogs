---
title: Cool Papers更新：简单适配Zotero Connector
slug: cool-papers更新简单适配zotero-connector
date: 2025-08-25
tags: 网站, 论文, 酷论文, 生成模型, attention
status: pending
---

# Cool Papers更新：简单适配Zotero Connector

**原文链接**: [https://spaces.ac.cn/archives/11250](https://spaces.ac.cn/archives/11250)

**发布日期**: 

---

很早之前就有读者提出希望可以给[Cool Papers](https://papers.cool/)增加导入Zotero的功能，但由于笔者没用Zotero，加上又比较懒，所以一直没提上日程。这个周末刚好有点时间，研究了一下，做了个简单的适配。

## 单篇导入 #

首先，我们需要安装[Zotero](https://www.zotero.org/)（这是废话），然后需要给所用浏览器安装[Zotero Connector](https://www.zotero.org/download/connectors)插件。安装完成后，我们访问Cool Papers的单篇论文页面，如 <https://papers.cool/arxiv/2104.09864> 或 <https://papers.cool/venue/2024.naacl-long.431@ACL> ，然后点击Zotero Connector的图标，就会自动把论文导入了，包括PDF文件。

[![单篇论文导入到Zotero](/usr/uploads/2025/08/2053963362.png)](/usr/uploads/2025/08/2053963362.png "点击查看原图")

单篇论文导入到Zotero

保存的信息包括论文标题、作者、摘要、日期、所属主类别（arXiv）或所属会议（会议论文）。这是通过在网页头部嵌入Metadata实现的，不需要用户做额外配置，缺点是只支持单页面单论文的导入。

## 批量导入 #

如果想要支持批量导入，那就只能通过[Translator](https://github.com/zotero/translators)来实现了，它是实现复杂导入的必要条件。当我们访问arXiv官网时，Zotero Connector能够自动识别多篇论文让我们选择导入，就是通过Translator实现的，只不过arXiv的Translator已经内置到Zotero中，而Cool Papers的自然要自己写了。

Translator实际上就是一段JS代码，笔者已经写好并放到了Github上（[链接](https://github.com/bojone/papers.cool/blob/main/Zotero/CoolPapers.js)），读者只需要将它保存下来，并放到Zotero的translators目录中（MacOS是“用户/Zotero/translators”），重启Zotero就生效了。（注意，Translator是配置到Zotero而不是Zotero Connector中的。）

这时候我们访问列表页，如 <https://papers.cool/arxiv/cs.AI> 或 <https://papers.cool/venue/ACL.2025> ，正常来说Zotero Connector的图标就会变成文件夹形状，点击它，就会出现如下的多选框：  


[![批量导入到Zotero](/usr/uploads/2025/08/168196517.png)](/usr/uploads/2025/08/168196517.png "点击查看原图")

批量导入到Zotero

这里显示的论文，并不是当前列表页的所有论文，而是当前页面下我们曾点击过“[PDF]”或“[Kimi]”的论文，这意味着我们可能对这些论文感兴趣，因此可能会需要进一步收藏到Zotero，这便是Cool Papers适配Zotero的逻辑。

## 文章小结 #

对Cool Papers做了一些简单的调整，并写了一个Translator，从而可以配合Zotero Connector，将Cool Papers的论文导入到Zotero上。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/11250>_

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

苏剑林. (Aug. 25, 2025). 《Cool Papers更新：简单适配Zotero Connector 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/11250>

@online{kexuefm-11250,  
title={Cool Papers更新：简单适配Zotero Connector},  
author={苏剑林},  
year={2025},  
month={Aug},  
url={\url{https://spaces.ac.cn/archives/11250}},  
} 


---

## 公式推导与注释

TODO: 添加详细的数学公式推导和注释

