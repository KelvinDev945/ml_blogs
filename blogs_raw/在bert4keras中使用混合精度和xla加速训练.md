---
title: 在bert4keras中使用混合精度和XLA加速训练
slug: 在bert4keras中使用混合精度和xla加速训练
date: 2022-04-28
tags: 模型, 优化, 梯度, 生成模型, attention
status: pending
---

# 在bert4keras中使用混合精度和XLA加速训练

**原文链接**: [https://spaces.ac.cn/archives/9059](https://spaces.ac.cn/archives/9059)

**发布日期**: 

---

之前笔者一直都是聚焦于模型的构思和实现，鲜有关注模型的训练加速，像混合精度和XLA这些技术，虽然也有听过，但没真正去实践过。这两天折腾了一番，成功在bert4keras中使用了混合精度和XLA来加速训练，在此做个简单的总结，供大家参考。

本文的多数经验结论并不只限于bert4keras中使用，之所以在标题中强调bert4keras，只不过bert4keras中的模型实现相对较为规整，因此启动这些加速技巧所要做的修改相对更少。

## 实验环境 #

本文的实验显卡为3090，使用的docker镜像为nvcr.io/nvidia/tensorflow:21.09-tf1-py3，其中自带的tensorflow版本为1.15.5。另外，实验所用的bert4keras版本为0.11.3。其他环境也可以参考着弄，要注意有折腾精神，不要指望着无脑调用。

顺便提一下，3090、A100等卡只能用cuda11，而tensorflow官网的1.15版本是不支持cuda11的，如果还想用tensorflow 1.x，那么只能用nvidia亲自维护的[nvidia-tensorflow](https://github.com/NVIDIA/tensorflow)，或者用其构建的[docker镜像](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html)。用nvidia而不是google维护的tensorflow，除了能让你在最新的显卡用上1.x版本外，还有nvidia专门做的一些额外优化，具体文档可以参考[这里](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html)。

不要说“tensorflow都出到2.8了，怎么还用1.15”这样的话，你的显卡是nvidia产的，所以哪个版本的tensorflow最好用，你我说了不算，甚至Google说了都不算，nvidia说的才算，nvidia还在维护着1.15，那说明1.15才是yyds。

## 混合精度 #

首先我们来看混合精度训练，简单来说就是模型计算用FP16、参数更新和存储用FP32，FP16的表示范围大致是$6\times 10^{-8}\sim 65504$，其上下界都是我们在实现模型时有可能触碰到的，所以引入FP16后最大的问题就是溢出和精度损失。更详细的原理介绍大家自行搜索就好，本文主要关注怎么用。

nvidia-tensorflow的帮助文档中对混合精度训练的介绍可见[这里](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#tfamp)，其中启动混合精度训练最简单的方法是脚本的开头添加环境变量：
    
    
    import os
    os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'  # 混合精度训练

读者或许留意到，多数教程介绍的是 TF_ENABLE_AUTO_MIXED_PRECISION 而我这里是 TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE ，它们的区别在于前者会自动添加“动态损失放大（Loss Scaling）”而后者不会，但笔者测试发现“动态损失放大”并不能替代手动调整损失，因此干脆不要这个功能了。

添加完环境变量后，可以重新启动训练脚本看看情况。如果训练开始就出现了NaN，那么可以调整一下infinity和epsilon：
    
    
    from bert4keras.backend import K
    K.set_infinity(1e4)
    K.set_epsilon(1e-5)
    

调整完后通常不会一开始就NaN了（如果还有，那就检查一下模型其他地方有没有用到不受这这两个函数控制的 infinity 和 epsilon 并修改过来），但有可能出现的是loss先降后升最后NaN，这是因为初始化不好，或者是像[DeepNet](/archives/8994)那样刻意为之，使得模型存在部分参数的梯度极小（小于$10^{-8}$），这时候在FP16的精度内它就直接等于0了，于是这部分参数不会得到更新，或者等价说梯度是不准的，长时间用不准的梯度更新，就容易不收敛。

这时候解决方案就是“损失放大”了。我们可以直接在损失函数上乘上一个放大因子（比如1000，可以自行调试，不出现NaN的前提下越大越好），使得原本很小的梯度就得以放大到FP16范围内，不至于直接置零，避免了梯度的精度损失。而对于我们平时用的Adam、[LAMB](/archives/8978)等优化器来说，损失函数乘上一个常数并不会改变这些优化器的训练过程，也就是它们完全是兼容“损失放大”的。

事实上，笔者发现“损失放大”技巧不仅仅在混合精度训练场景下有效，即便是全FP32精度训练也会有一定作用：在全FP32精度训练时，如果不进行损失放大，开始阶段模型会停留在某个损失值一段时间，然后才慢慢下降；而如果进行了损失放大，那么开始阶段模型就一直保持缓慢下降趋势，相对来说收敛更快了。

## 代数加速 #

现在我们来看XLA，全称为“Accelerated Linear Algebra”，即专门用来加速线性代数运算的。简单来说，XLA就是对计算图提前进行编译优化，将能合并的算子进行合并（减少缓存变量以节省内存），将能并行的算子进行并行（提高计算速度）。

在nvidia-tensorflow中，启动XLA的最简单方式依旧是添加环境变量：
    
    
    import os
    os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'  # 启用XLA

但要注意，XLA不是保证有提升的，刚才我们说到，XLA会将能并行的算子尽量并行，很明显这是通过空间换时间的方案，因此启用XLA后可能会消耗更多的显存以导致OOM，甚至并行簇过大时反而会导致性能下降。[官方文档](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-best-practices)对有可能出现的异常做了比较详尽的分析并提出了相应的建议，其中笔者推荐的解决方法是补充`--tf_xla_enable_lazy_compilation=false`参数：
    
    
    import os
    os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'  # 启用XLA
    os.environ['TF_XLA_FLAGS'] += ' --tf_xla_enable_lazy_compilation=false'  # 优化XLA

如果这都不能解决，那就换成XLA Lite：
    
    
    import os
    os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'  # 启用XLA Lite

如果换成XLA Lite都无法解决，那基本就说明XLA不适合你的模型了。

## 性能比较 #

在3090上，启动混合精度训练带来的加速大概是10%多一点。这个幅度可能不如大家想象的那么快，笔者猜测这是因为3090、A100等新卡上面，默认的FP32格式实际上用的是一种名为TF32的格式（参考[这里](https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/)），TF32某种意义来说本身就是一种“半精度格式”，比FP32更快。换句话说，3090上的FP32本身就相当于已经做过一定的半精度优化了，速度本来就更快，因此换成混合精度后的提升相对来说就小了。

至于XLA带来的提升，大致是15%左右。在笔者的训练脚本中，直接设置环境变量 TF_XLA_FLAGS 为`--tf_xla_auto_jit=1`会OOM，补充`--tf_xla_enable_lazy_compilation=false`依旧，而改为`--tf_xla_auto_jit=fusible`则可以正常训练。

最后，最关键的是，混合精度与XLA可以叠加使用！两者一起使用带来的加速大概是30%左右，并且混合精度训练的加入基本上可以抵消XLA带来的显存消耗增加，两者真可谓是相得益彰了。

## 文章小结 #

本文介绍了在bert4keras中使用混合精度和XLA加速训练的尝试，两者同时启用大概能在3090上加速30%左右。

_**转载到请包括本文地址：**<https://spaces.ac.cn/archives/9059>_

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

苏剑林. (Apr. 28, 2022). 《在bert4keras中使用混合精度和XLA加速训练 》[Blog post]. Retrieved from <https://spaces.ac.cn/archives/9059>

@online{kexuefm-9059,  
title={在bert4keras中使用混合精度和XLA加速训练},  
author={苏剑林},  
year={2022},  
month={Apr},  
url={\url{https://spaces.ac.cn/archives/9059}},  
} 


---

## 公式推导与注释

### 一、浮点数表示与数值范围

#### 1.1 IEEE 754浮点数标准

浮点数采用科学计数法表示：

\begin{equation}
x = (-1)^s \times m \times 2^e
\tag{1}
\end{equation}

其中：
- $s$：符号位（0为正，1为负）
- $m$：尾数（mantissa），范围为$[1, 2)$
- $e$：指数（exponent）

#### 1.2 FP32（单精度）表示

**位分配**：总32位
- 符号位：1位
- 指数位：8位
- 尾数位：23位

**数值范围**：

\begin{equation}
\begin{aligned}
\text{最小正规数} &= 2^{-126} \approx 1.175 \times 10^{-38} \\
\text{最大正规数} &= (2-2^{-23}) \times 2^{127} \approx 3.403 \times 10^{38} \\
\text{最小非规数} &= 2^{-149} \approx 1.401 \times 10^{-45}
\end{aligned}
\tag{2}
\end{equation}

**精度**：尾数23位意味着相对精度约为$2^{-24} \approx 5.96 \times 10^{-8}$。

#### 1.3 FP16（半精度）表示

**位分配**：总16位
- 符号位：1位
- 指数位：5位
- 尾数位：10位

**数值范围**：

\begin{equation}
\begin{aligned}
\text{最小正规数} &= 2^{-14} \approx 6.104 \times 10^{-5} \\
\text{最大正规数} &= (2-2^{-10}) \times 2^{15} = 65504 \\
\text{最小非规数} &= 2^{-24} \approx 5.96 \times 10^{-8}
\end{aligned}
\tag{3}
\end{equation}

**精度**：尾数10位意味着相对精度约为$2^{-11} \approx 4.88 \times 10^{-4}$。

**关键观察**：
- FP16的表示范围仅为$[6\times 10^{-8}, 65504]$，远小于FP32
- FP16的精度比FP32低约3个数量级

#### 1.4 范围对比表

\begin{equation}
\begin{array}{|c|c|c|c|}
\hline
\text{类型} & \text{符号位} & \text{指数位} & \text{尾数位} & \text{范围} & \text{相对精度} \\
\hline
\text{FP32} & 1 & 8 & 23 & \pm 3.4\times 10^{38} & 2^{-24} \\
\text{FP16} & 1 & 5 & 10 & \pm 6.5\times 10^{4} & 2^{-11} \\
\text{BF16} & 1 & 8 & 7 & \pm 3.4\times 10^{38} & 2^{-8} \\
\hline
\end{array}
\tag{4}
\end{equation}

**注**：BF16（Brain Float 16）保持与FP32相同的指数范围，但精度降低。

### 二、混合精度训练的数学原理

#### 2.1 前向传播的数值流

在混合精度训练中，前向传播使用FP16计算：

\begin{equation}
\begin{aligned}
\boldsymbol{h}_l^{(\text{FP16})} &= f_l(\boldsymbol{h}_{l-1}^{(\text{FP16})}, \boldsymbol{W}_l^{(\text{FP16})}) \\
&= \text{FP16}(f_l(\boldsymbol{h}_{l-1}^{(\text{FP16})}, \boldsymbol{W}_l^{(\text{FP16})}))
\end{aligned}
\tag{5}
\end{equation}

其中$f_l$是第$l$层的变换函数。

**数值误差分析**：每次运算引入舍入误差：

\begin{equation}
\text{FP16}(x) = x(1 + \delta),\quad |\delta| \leq 2^{-11}
\tag{6}
\end{equation}

经过$L$层后，累积误差为：

\begin{equation}
\boldsymbol{h}_L = \boldsymbol{h}_L^{\text{exact}} \prod_{l=1}^L (1 + \delta_l) \approx \boldsymbol{h}_L^{\text{exact}} (1 + \sum_{l=1}^L \delta_l)
\tag{7}
\end{equation}

当$L$很大时，误差累积可能导致精度损失。

#### 2.2 损失函数计算

损失函数通常在FP32精度下计算以保证准确性：

\begin{equation}
\mathcal{L} = \text{FP32}(\text{loss}(\boldsymbol{h}_L^{(\text{FP16})}, \boldsymbol{y}))
\tag{8}
\end{equation}

**示例**：交叉熵损失

\begin{equation}
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^n y_i \log p_i = -\sum_{i=1}^n y_i \log \frac{\exp(z_i)}{\sum_j \exp(z_j)}
\tag{9}
\end{equation}

若$z_i$接近65504（FP16上界），$\exp(z_i)$会溢出为`Inf`。

**解决方案**：数值稳定的Softmax计算

\begin{equation}
\log p_i = z_i - \log \sum_j \exp(z_j) = z_i - \max_k z_k - \log \sum_j \exp(z_j - \max_k z_k)
\tag{10}
\end{equation}

通过减去最大值避免指数溢出。

#### 2.3 反向传播与梯度

反向传播计算梯度：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_l} \frac{\partial \boldsymbol{h}_l}{\partial \boldsymbol{W}_l}
\tag{11}
\end{equation}

**梯度下溢问题**：梯度通常比激活值小几个数量级。例如，对于ReLU激活：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_l} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_{l+1}} \cdot \boldsymbol{W}_{l+1}^T \cdot \mathbb{1}[\boldsymbol{h}_l > 0]
\tag{12}
\end{equation}

如果$\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_{l+1}}\| \ll 1$，且$\|\boldsymbol{W}_{l+1}\| \approx 1$，则经过多层后：

\begin{equation}
\left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_1}\right\| \approx \left\|\frac{\partial \mathcal{L}}{\partial \boldsymbol{h}_L}\right\| \prod_{l=1}^{L-1} \|\boldsymbol{W}_l\|
\tag{13}
\end{equation}

当$\|\boldsymbol{W}_l\| < 1$时，梯度呈指数衰减。

**FP16下溢**：若梯度小于$6 \times 10^{-8}$（FP16最小非规数），则会被截断为0：

\begin{equation}
\text{FP16}(g) = \begin{cases}
g, & |g| \geq 6 \times 10^{-8} \\
0, & |g| < 6 \times 10^{-8}
\end{cases}
\tag{14}
\end{equation}

### 三、Loss Scaling原理

#### 3.1 基本思想

通过放大损失函数，间接放大梯度：

\begin{equation}
\tilde{\mathcal{L}} = S \cdot \mathcal{L}
\tag{15}
\end{equation}

其中$S > 1$是缩放因子（通常为$2^{10}$到$2^{15}$）。

**梯度缩放**：根据链式法则：

\begin{equation}
\frac{\partial \tilde{\mathcal{L}}}{\partial \boldsymbol{W}} = S \cdot \frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}
\tag{16}
\end{equation}

**参数更新**：在FP32下更新参数时，需要将梯度缩小回原尺度：

\begin{equation}
\boldsymbol{W}^{\text{FP32}}_{t+1} = \boldsymbol{W}^{\text{FP32}}_t - \eta \cdot \frac{1}{S} \cdot \frac{\partial \tilde{\mathcal{L}}}{\partial \boldsymbol{W}}\Bigg|_{\text{FP16}}
\tag{17}
\end{equation}

#### 3.2 最优缩放因子选择

**目标**：选择$S$使得梯度充分利用FP16的动态范围。

设原始梯度的分布为：

\begin{equation}
g \sim \mathcal{N}(0, \sigma_g^2)
\tag{18}
\end{equation}

**下溢约束**：希望缩放后的梯度$S \cdot g$大于FP16最小值：

\begin{equation}
P(|S \cdot g| \geq 6 \times 10^{-8}) \geq 1 - \epsilon
\tag{19}
\end{equation}

对于正态分布，这要求：

\begin{equation}
S \cdot \sigma_g \geq \Phi^{-1}\left(\frac{1+1-\epsilon}{2}\right) \cdot 6 \times 10^{-8}
\tag{20}
\end{equation}

其中$\Phi^{-1}$是标准正态分布的逆累积分布函数。

**上溢约束**：同时避免缩放后超过FP16最大值65504：

\begin{equation}
P(|S \cdot g| \leq 65504) \geq 1 - \epsilon
\tag{21}
\end{equation}

综合两个约束，最优缩放因子为：

\begin{equation}
S^* \approx \frac{65504}{k \cdot \max(|g_i|)}
\tag{22}
\end{equation}

其中$k \in [2, 4]$是安全系数。

#### 3.3 静态 vs 动态Loss Scaling

**静态Loss Scaling**：

\begin{equation}
S = \text{const},\quad \text{如 } S = 2^{12} = 4096
\tag{23}
\end{equation}

优点：简单，无额外计算开销
缺点：可能不适应训练过程中梯度分布的变化

**动态Loss Scaling**：

初始化$S_0 = 2^{15}$，然后根据梯度是否溢出动态调整：

\begin{equation}
S_{t+1} = \begin{cases}
S_t \times 2, & \text{若连续 } N \text{ 步无溢出} \\
S_t / 2, & \text{若检测到溢出} \\
S_t, & \text{否则}
\end{cases}
\tag{24}
\end{equation}

其中$N$通常取1000-2000。

**溢出检测**：检查梯度中是否有`Inf`或`NaN`：

\begin{equation}
\text{overflow} = \bigvee_{i} (\text{isnan}(g_i) \vee \text{isinf}(g_i))
\tag{25}
\end{equation}

#### 3.4 数学证明：Loss Scaling不改变优化方向

**定理**：对于Adam等自适应优化器，Loss Scaling不改变参数更新的方向和相对大小。

**证明**（以Adam为例）：

Adam的更新公式为：

\begin{equation}
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\Delta \boldsymbol{W}_t &= -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
\tag{26}
\end{equation}

应用Loss Scaling后，$g_t' = S \cdot g_t$：

\begin{equation}
\begin{aligned}
m_t' &= \beta_1 m_{t-1}' + (1-\beta_1) S g_t = S m_t \\
v_t' &= \beta_2 v_{t-1}' + (1-\beta_2) S^2 g_t^2 = S^2 v_t \\
\hat{m}_t' &= S \hat{m}_t,\quad \hat{v}_t' = S^2 \hat{v}_t
\end{aligned}
\tag{27}
\end{equation}

因此更新量为：

\begin{equation}
\Delta \boldsymbol{W}_t' = -\eta \frac{S \hat{m}_t}{\sqrt{S^2 \hat{v}_t} + \epsilon} = -\eta \frac{S \hat{m}_t}{S\sqrt{\hat{v}_t} + \epsilon}
\tag{28}
\end{equation}

当$\epsilon \ll S\sqrt{\hat{v}_t}$时（实践中通常成立）：

\begin{equation}
\Delta \boldsymbol{W}_t' \approx -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}} = \Delta \boldsymbol{W}_t
\tag{29}
\end{equation}

因此Loss Scaling对Adam更新几乎无影响。$\square$

**注意**：对于SGD，Loss Scaling会改变更新量，需要手动除以$S$：

\begin{equation}
\Delta \boldsymbol{W}_t^{\text{SGD}} = -\eta \frac{g_t'}{S} = -\eta g_t
\tag{30}
\end{equation}

### 四、数值稳定性分析

#### 4.1 Batch Normalization的稳定性

Batch Normalization计算：

\begin{equation}
\begin{aligned}
\mu_B &= \frac{1}{B}\sum_{i=1}^B x_i \\
\sigma_B^2 &= \frac{1}{B}\sum_{i=1}^B (x_i - \mu_B)^2 \\
\hat{x}_i &= \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\end{aligned}
\tag{31}
\end{equation}

**FP16下的问题**：若$\sigma_B^2 \ll \epsilon$，则$\sqrt{\sigma_B^2 + \epsilon} \approx \sqrt{\epsilon}$，归一化效果减弱。

**稳定版本**：

\begin{equation}
\hat{x}_i = \frac{x_i - \mu_B}{\max(\sigma_B, \epsilon)}
\tag{32}
\end{equation}

#### 4.2 Layer Normalization的稳定性

Layer Normalization对单个样本归一化：

\begin{equation}
\begin{aligned}
\mu &= \frac{1}{d}\sum_{j=1}^d x_j \\
\sigma^2 &= \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2 \\
\hat{x}_j &= \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
\end{aligned}
\tag{33}
\end{equation}

**推荐epsilon值**：
- FP32：$\epsilon = 10^{-5}$
- FP16：$\epsilon = 10^{-3}$（更大的epsilon提高稳定性）

#### 4.3 Softmax的数值稳定性

标准Softmax：

\begin{equation}
p_i = \frac{\exp(z_i)}{\sum_{j=1}^n \exp(z_j)}
\tag{34}
\end{equation}

**问题1**：$z_i$过大导致$\exp(z_i)$溢出
**问题2**：所有$z_i$都很大时，分子分母同时溢出

**稳定版本**（减去最大值）：

\begin{equation}
p_i = \frac{\exp(z_i - z_{\max})}{\sum_{j=1}^n \exp(z_j - z_{\max})}
\tag{35}
\end{equation}

其中$z_{\max} = \max_j z_j$。

**证明**：

\begin{equation}
\frac{\exp(z_i - z_{\max})}{\sum_j \exp(z_j - z_{\max})} = \frac{\exp(z_i)/\exp(z_{\max})}{\sum_j \exp(z_j)/\exp(z_{\max})} = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
\tag{36}
\end{equation}

**范围保证**：$z_i - z_{\max} \leq 0$，因此$\exp(z_i - z_{\max}) \in (0, 1]$，不会溢出。

#### 4.4 梯度裁剪

防止梯度爆炸：

\begin{equation}
\boldsymbol{g}_{\text{clip}} = \begin{cases}
\boldsymbol{g}, & \|\boldsymbol{g}\| \leq \tau \\
\tau \frac{\boldsymbol{g}}{\|\boldsymbol{g}\|}, & \|\boldsymbol{g}\| > \tau
\end{cases}
\tag{37}
\end{equation}

其中$\tau$是阈值（如$\tau = 1.0$）。

**与Loss Scaling的交互**：应在缩放回原尺度后再裁剪：

\begin{equation}
\boldsymbol{g}_{\text{clip}} = \text{clip}\left(\frac{\boldsymbol{g}_{\text{scaled}}}{S}\right)
\tag{38}
\end{equation}

### 五、XLA编译优化原理

#### 5.1 计算图融合

**示例**：连续的逐元素操作

\begin{equation}
\boldsymbol{y} = \sigma(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b})
\tag{39}
\end{equation}

**未融合**：需要3次内存读写
1. $\boldsymbol{z}_1 = \boldsymbol{W}\boldsymbol{x}$（写入）
2. $\boldsymbol{z}_2 = \boldsymbol{z}_1 + \boldsymbol{b}$（读取$\boldsymbol{z}_1$，写入$\boldsymbol{z}_2$）
3. $\boldsymbol{y} = \sigma(\boldsymbol{z}_2)$（读取$\boldsymbol{z}_2$，写入$\boldsymbol{y}$）

**融合后**：1次内存写入

\begin{equation}
\boldsymbol{y}_i = \sigma((\boldsymbol{W}\boldsymbol{x})_i + b_i),\quad i = 1, \cdots, n
\tag{40}
\end{equation}

直接计算并写入，无需中间存储。

**加速比**：假设计算时间$T_{\text{comp}}$，内存访问时间$T_{\text{mem}}$，未融合时间：

\begin{equation}
T_{\text{unfused}} = T_{\text{comp}} + 3T_{\text{mem}}
\tag{41}
\end{equation}

融合后时间：

\begin{equation}
T_{\text{fused}} = T_{\text{comp}} + T_{\text{mem}}
\tag{42}
\end{equation}

加速比为：

\begin{equation}
\text{Speedup} = \frac{T_{\text{comp}} + 3T_{\text{mem}}}{T_{\text{comp}} + T_{\text{mem}}}
\tag{43}
\end{equation}

当$T_{\text{mem}} \gg T_{\text{comp}}$（内存瓶颈）时，加速比接近3。

#### 5.2 自动微分优化

**正向模式自动微分**（Forward-mode AD）：

\begin{equation}
\frac{d f(\boldsymbol{x})}{d x_i} = \lim_{\epsilon \to 0} \frac{f(\boldsymbol{x} + \epsilon \boldsymbol{e}_i) - f(\boldsymbol{x})}{\epsilon}
\tag{44}
\end{equation}

计算复杂度：$O(n)$，其中$n$是输入维度。

**反向模式自动微分**（Reverse-mode AD，即反向传播）：

\begin{equation}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}} \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{W}}
\tag{45}
\end{equation}

计算复杂度：$O(1)$相对于输出维度。

**XLA优化**：自动选择最优微分模式，并融合梯度计算。

#### 5.3 内存布局优化

**行主序 vs 列主序**：

对于矩阵$\boldsymbol{A} \in \mathbb{R}^{m \times n}$：

**行主序**（Row-major，C风格）：

\begin{equation}
\boldsymbol{A}[i, j] \to \text{index} = i \cdot n + j
\tag{46}
\end{equation}

**列主序**（Column-major，Fortran风格）：

\begin{equation}
\boldsymbol{A}[i, j] \to \text{index} = j \cdot m + i
\tag{47}
\end{equation}

**缓存友好性**：连续访问同一行时，行主序更高效；连续访问同一列时，列主序更高效。

**XLA优化**：根据访问模式自动转换内存布局。

#### 5.4 并行化分析

**数据并行**：将batch分割到多个设备：

\begin{equation}
\boldsymbol{X} = [\boldsymbol{X}_1, \boldsymbol{X}_2, \cdots, \boldsymbol{X}_K]
\tag{48}
\end{equation}

每个设备计算：

\begin{equation}
\mathcal{L}_k = \text{loss}(f(\boldsymbol{X}_k; \boldsymbol{W}))
\tag{49}
\end{equation}

梯度聚合：

\begin{equation}
\boldsymbol{g} = \frac{1}{K}\sum_{k=1}^K \frac{\partial \mathcal{L}_k}{\partial \boldsymbol{W}}
\tag{50}
\end{equation}

**通信成本**：AllReduce操作复杂度为$O(P \cdot M)$，其中$P$是设备数，$M$是参数量。

**XLA优化**：重叠通信与计算，减少同步开销。

### 六、混合精度训练的实践策略

#### 6.1 模型层分类

**可安全使用FP16的层**：
- 卷积层：$\boldsymbol{y} = \boldsymbol{W} * \boldsymbol{x}$
- 全连接层：$\boldsymbol{y} = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}$
- Attention：$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V})$

**需保持FP32的操作**：
- Batch Normalization的统计量更新
- Loss计算
- Softmax（logits可以FP16，但计算应在FP32）

#### 6.2 梯度累积

小批量训练时，通过累积梯度模拟大批量：

\begin{equation}
\begin{aligned}
\boldsymbol{g}_{\text{accum}} &= \sum_{k=1}^K \frac{1}{K} \frac{\partial \mathcal{L}_k}{\partial \boldsymbol{W}} \\
\boldsymbol{W}_{t+1} &= \boldsymbol{W}_t - \eta \boldsymbol{g}_{\text{accum}}
\end{aligned}
\tag{51}
\end{equation}

**与Loss Scaling的配合**：

\begin{equation}
\boldsymbol{g}_{\text{accum}} = \frac{1}{K \cdot S} \sum_{k=1}^K \frac{\partial (S \cdot \mathcal{L}_k)}{\partial \boldsymbol{W}}
\tag{52}
\end{equation}

#### 6.3 学习率调整

混合精度训练可能需要调整学习率。理论上，由于Loss Scaling，有效学习率不变：

\begin{equation}
\eta_{\text{eff}} = \eta \cdot \frac{1}{S} \cdot S = \eta
\tag{53}
\end{equation}

但实践中，由于舍入误差，可能需要微调：

\begin{equation}
\eta_{\text{mixed}} \approx (1 \pm 0.1) \eta_{\text{FP32}}
\tag{54}
\end{equation}

### 七、性能分析与理论加速比

#### 7.1 理论计算加速

**FLOPS对比**：
- FP32：现代GPU约10-20 TFLOPS
- FP16：现代GPU约100-300 TFLOPS（Tensor Core）

**理论加速比**：

\begin{equation}
\text{Speedup}_{\text{compute}} = \frac{\text{FLOPS}_{\text{FP16}}}{\text{FLOPS}_{\text{FP32}}} \approx 8\sim 15\times
\tag{55}
\end{equation}

#### 7.2 内存带宽加速

FP16数据量是FP32的一半：

\begin{equation}
\text{Bandwidth}_{\text{saved}} = \frac{\text{Size}_{\text{FP32}}}{\text{Size}_{\text{FP16}}} = 2\times
\tag{56}
\end{equation}

对于内存受限（memory-bound）的操作，加速接近2倍。

#### 7.3 实际加速分析

实际加速受限于：
1. **Amdahl定律**：必须保持FP32的部分限制整体加速

\begin{equation}
\text{Speedup}_{\text{actual}} = \frac{1}{(1-p) + p/S}
\tag{57}
\end{equation}

其中$p$是可并行/加速部分的比例，$S$是该部分的加速比。

2. **数据转换开销**：FP16↔FP32转换需要时间

\begin{equation}
T_{\text{total}} = T_{\text{compute}}^{\text{FP16}} + T_{\text{convert}} + T_{\text{update}}^{\text{FP32}}
\tag{58}
\end{equation}

3. **Loss Scaling检查开销**：动态Loss Scaling需要检测溢出

**实测加速比**：通常在1.3-2.0倍之间（本文实验为1.1倍，可能因TF32已有优化）。

### 八、TF32格式详解

#### 8.1 TF32定义

TF32（TensorFloat-32）是NVIDIA Ampere架构引入的格式：

**位分配**：
- 符号位：1位
- 指数位：8位（与FP32相同）
- 尾数位：10位（与FP16相同）

\begin{equation}
\text{TF32} = \text{FP32}_{\text{range}} + \text{FP16}_{\text{precision}}
\tag{59}
\end{equation}

#### 8.2 TF32的优势

**范围**：与FP32相同，避免溢出
**精度**：$2^{-11}$，略低于FP32但足够

**计算性能**：Tensor Core加速，接近FP16速度

\begin{equation}
\text{FLOPS}_{\text{TF32}} \approx 0.5 \times \text{FLOPS}_{\text{FP16}}
\tag{60}
\end{equation}

#### 8.3 为什么3090上混合精度加速有限

在Ampere架构（3090、A100）上，默认FP32计算已自动转换为TF32：

\begin{equation}
\text{FP32}_{\text{input}} \xrightarrow{\text{自动}} \text{TF32}_{\text{compute}} \xrightarrow{\text{自动}} \text{FP32}_{\text{output}}
\tag{61}
\end{equation}

因此"FP32训练"实际已享受部分加速，混合精度相对提升变小。

### 九、实践建议与调试技巧

#### 9.1 渐进式启用

1. **先FP32基线**：确保模型正常收敛
2. **加入Loss Scaling**：$S = 2^{10}$开始
3. **逐步增大$S$**：观察是否有NaN
4. **启用动态Loss Scaling**：自动调整

#### 9.2 调试NaN问题

**检查点1**：是否为数值溢出？

\begin{equation}
\max_i |z_i| > 65504 \Rightarrow \text{降低初始化方差或添加梯度裁剪}
\tag{62}
\end{equation}

**检查点2**：是否为梯度下溢？

\begin{equation}
\min_i |g_i| < 10^{-7} \Rightarrow \text{增大Loss Scaling}
\tag{63}
\end{equation}

**检查点3**：是否为不稳定的数值操作？

\begin{equation}
\log(0),\, \frac{1}{0},\, \sqrt{x<0} \Rightarrow \text{添加epsilon保护}
\tag{64}
\end{equation}

#### 9.3 监控指标

训练过程中监控：

\begin{equation}
\begin{aligned}
\text{激活值范围}: &\quad [\min(\boldsymbol{h}), \max(\boldsymbol{h})] \\
\text{梯度范数}: &\quad \|\nabla_{\boldsymbol{W}} \mathcal{L}\| \\
\text{参数范数}: &\quad \|\boldsymbol{W}\| \\
\text{Loss Scaling因子}: &\quad S_t
\end{aligned}
\tag{65}
\end{equation}

异常信号：
- 激活值突然变大：可能即将溢出
- 梯度范数突降为0：梯度下溢
- Loss Scaling频繁减半：模型不稳定

### 十、总结与理论洞察

混合精度训练的核心思想可总结为：

\begin{equation}
\boxed{
\begin{aligned}
&\text{前向传播：FP16节省内存与计算} \\
&\text{反向传播：Loss Scaling防止梯度下溢} \\
&\text{参数更新：FP32保证累积精度}
\end{aligned}
}
\tag{66}
\end{equation}

**数学本质**：利用神经网络的冗余性和误差容忍性，在不损失最终性能的前提下降低计算精度。

**适用条件**：
1. 模型足够大，计算占主导
2. 梯度分布合理，不过度稀疏
3. 使用自适应优化器（Adam系列）

通过理论分析与实践技巧的结合，混合精度训练成为现代大规模模型训练的标准配置。

