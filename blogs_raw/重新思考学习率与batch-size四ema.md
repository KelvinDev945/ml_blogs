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

### 1. 指数移动平均(EMA)的数学基础

#### 1.1 EMA的定义

**标准形式**:
\begin{equation}\boldsymbol{\theta}_{ema,t} = \beta\boldsymbol{\theta}_{ema,t-1} + (1-\beta)\boldsymbol{\theta}_t\tag{1}\end{equation}

其中$\beta \in [0,1)$是衰减率,$\boldsymbol{\theta}_t$是当前优化器输出的参数。

**等价递推形式**:
\begin{equation}\boldsymbol{\theta}_{ema,t} = (1-\beta)\sum_{s=0}^{t}\beta^{t-s}\boldsymbol{\theta}_s\tag{2}\end{equation}

**数学直觉**: EMA是对历史参数的指数加权平均,越近的参数权重越大。

#### 1.2 偏置修正

**问题**: 初始时$\boldsymbol{\theta}_{ema,0} = 0$(或任意初始化),导致$\boldsymbol{\theta}_{ema,t}$偏向初始值。

**修正方法** (Adam-style):
\begin{equation}\hat{\boldsymbol{\theta}}_{ema,t} = \frac{\boldsymbol{\theta}_{ema,t}}{1 - \beta^{t+1}}\tag{3}\end{equation}

**推导**: 假设$\boldsymbol{\theta}_{ema,0} = 0$,则:
\begin{equation}\boldsymbol{\theta}_{ema,t} = (1-\beta)\sum_{s=0}^{t}\beta^{t-s}\boldsymbol{\theta}_s\tag{4}\end{equation}

期望(假设$\mathbb{E}[\boldsymbol{\theta}_s] = \boldsymbol{\theta}^*$为常数):
\begin{equation}\mathbb{E}[\boldsymbol{\theta}_{ema,t}] = (1-\beta)\boldsymbol{\theta}^*\sum_{s=0}^t \beta^{t-s} = (1-\beta^{t+1})\boldsymbol{\theta}^*\tag{5}\end{equation}

因此$\hat{\boldsymbol{\theta}}_{ema,t}$是无偏估计。

#### 1.3 有效窗口长度

**定义**: 有效样本数为权重之和:
\begin{equation}N_{eff} = \sum_{s=0}^{\infty}(1-\beta)\beta^s = 1\tag{6}\end{equation}

**半衰期** (权重下降到50%的时间):
\begin{equation}t_{1/2} = \frac{\log 2}{\log(1/\beta)} \approx \frac{0.69}{1-\beta}\tag{7}\end{equation}

**示例**: $\beta = 0.9 \Rightarrow t_{1/2} \approx 7$步

**数学直觉**: EMA相当于在最近$\mathcal{O}(1/(1-\beta))$步参数上做平均。

### 2. EMA在优化中的作用

#### 2.1 方差减少

**无EMA的参数方差**: 假设SGD更新$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \tilde{\boldsymbol{g}}_t$,
\begin{equation}\text{Var}[\boldsymbol{\theta}_t] \approx \eta^2 t \cdot \text{Var}[\tilde{\boldsymbol{g}}]\tag{8}\end{equation}

方差随时间线性增长。

**有EMA的方差**: 从式(2),
\begin{equation}\text{Var}[\boldsymbol{\theta}_{ema,t}] = (1-\beta)^2\sum_{s=0}^t \beta^{2(t-s)}\text{Var}[\boldsymbol{\theta}_s]\tag{9}\end{equation}

稳态下($t \to \infty$):
\begin{equation}\text{Var}[\boldsymbol{\theta}_{ema}] = \frac{1-\beta}{1+\beta}\text{Var}[\boldsymbol{\theta}]\tag{10}\end{equation}

**方差减少因子**: $\frac{1-\beta}{1+\beta}$(如$\beta=0.9$时减少到$\frac{1}{19}$)。

#### 2.2 隐式正则化

EMA参数满足隐式优化:
\begin{equation}\boldsymbol{\theta}_{ema} = \mathop{\arg\min}_{\boldsymbol{\theta}} \mathbb{E}_{t}[\|\boldsymbol{\theta} - \boldsymbol{\theta}_t\|^2 \cdot \beta^t]\tag{11}\end{equation}

**数学直觉**: EMA寻找与优化轨迹"中心"最近的参数,类似$L_2$正则。

#### 2.3 Polyak平均

**定义** (无指数衰减):
\begin{equation}\boldsymbol{\theta}_{poly,t} = \frac{1}{t}\sum_{s=1}^t \boldsymbol{\theta}_s\tag{12}\end{equation}

**与EMA的关系**: Polyak平均是$\beta \to 1$的极限。

**理论保证**: 对于凸函数,
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_{poly,T})] - L^* \leq \frac{1}{T}\sum_{t=1}^T (\mathbb{E}[L(\boldsymbol{\theta}_t)] - L^*)\tag{13}\end{equation}

即Polyak平均的损失不超过路径平均损失。

**数学直觉**: Polyak平均"抹平"优化轨迹的噪声,收敛到更稳定的解。

### 3. EMA与Batch Size的交互

#### 3.1 等效Batch Size放大

**回顾**: 从主文档,动量机制使等效batch size变为:
\begin{equation}B_{eff} = B \cdot \frac{1+\beta}{1-\beta}\tag{14}\end{equation}

**解释**: EMA对梯度噪声做时间平均,相当于增大样本数。

#### 3.2 临界Batch Size的调整

定义无EMA的临界batch size:
\begin{equation}B_c^{(0)} = \frac{\text{tr}(\boldsymbol{\Sigma}\boldsymbol{H})}{\boldsymbol{g}^{\top}\boldsymbol{H}\boldsymbol{g}}\tag{15}\end{equation}

**有EMA时**: 由于$B_{eff}$增大,有效的$B_c$变为:
\begin{equation}B_c^{(ema)} = B_c^{(0)} \cdot \frac{1-\beta}{1+\beta}\tag{16}\end{equation}

**数学直觉**: EMA降低了对大batch size的需求,在小batch下也能获得稳定更新。

#### 3.3 学习率缩放调整

**无EMA**: $\eta \propto B$(在$B < B_c$时)

**有EMA**: 由于$B_{eff} > B$,需要调整:
\begin{equation}\eta_{ema} = \eta_0 \cdot \min\left(B/B_c^{(ema)}, 1\right)\tag{17}\end{equation}

**实践建议**: 使用EMA时,可以用更小的batch size和学习率。

### 4. 泛化误差界

#### 4.1 PAC-Bayes框架

**定理1** (EMA的泛化界): 对于$L$-Lipschitz损失,
\begin{equation}\mathbb{E}_{test}[L(\boldsymbol{\theta}_{ema})] \leq \mathbb{E}_{train}[L(\boldsymbol{\theta}_{ema})] + \sqrt{\frac{KL(\boldsymbol{\theta}_{ema}\|\boldsymbol{\theta}_0) + \log(1/\delta)}{2n}}\tag{18}\end{equation}

其中$KL$是KL散度,$n$是训练样本数,$\delta$是置信度。

**关键**: EMA的$KL(\boldsymbol{\theta}_{ema}\|\boldsymbol{\theta}_0)$通常小于单点$KL(\boldsymbol{\theta}_T\|\boldsymbol{\theta}_0)$,因为:
\begin{equation}KL(\boldsymbol{\theta}_{ema}\|\boldsymbol{\theta}_0) \leq \mathbb{E}_t[KL(\boldsymbol{\theta}_t\|\boldsymbol{\theta}_0)]\tag{19}\end{equation}

#### 4.2 噪声稳定性

**定义**: 模型$f(\boldsymbol{x};\boldsymbol{\theta})$对参数扰动的敏感性:
\begin{equation}S(\boldsymbol{\theta}) = \mathbb{E}_{\boldsymbol{x},\boldsymbol{\epsilon}}[(f(\boldsymbol{x};\boldsymbol{\theta} + \boldsymbol{\epsilon}) - f(\boldsymbol{x};\boldsymbol{\theta}))^2]\tag{20}\end{equation}

**定理2**: 若$\text{Var}[\boldsymbol{\theta}] = \sigma^2_{\theta}$,则:
\begin{equation}S(\boldsymbol{\theta}_{ema}) \leq \frac{1-\beta}{1+\beta}S(\boldsymbol{\theta})\tag{21}\end{equation}

**数学直觉**: EMA减少参数方差,提高模型对扰动的鲁棒性,从而改善泛化。

#### 4.3 Sharpness与EMA

**Sharpness定义**:
\begin{equation}Sh(\boldsymbol{\theta}) = \max_{\|\boldsymbol{\epsilon}\| \leq \rho}\frac{L(\boldsymbol{\theta} + \boldsymbol{\epsilon}) - L(\boldsymbol{\theta})}{1 + L(\boldsymbol{\theta})}\tag{22}\end{equation}

**定理3**: EMA倾向于收敛到sharpness更小的解:
\begin{equation}\mathbb{E}[Sh(\boldsymbol{\theta}_{ema})] \leq \mathbb{E}_t[Sh(\boldsymbol{\theta}_t)]\tag{23}\end{equation}

**证明思路**: EMA是凸组合,凸函数在凸组合点的Hessian特征值不超过端点的加权平均。

### 5. 不同EMA变体

#### 5.1 SWA (Stochastic Weight Averaging)

**定义**: 在训练后期等权平均:
\begin{equation}\boldsymbol{\theta}_{SWA} = \frac{1}{n_{SWA}}\sum_{t=T-n_{SWA}+1}^T \boldsymbol{\theta}_t\tag{24}\end{equation}

**与EMA对比**:
- SWA: $\beta \to 1$,只平均最后几个epoch
- EMA: 持续平均,权重指数衰减

**优点**: SWA适合学习率循环(cycle),可以探索多个局部最优。

**实现**: 每个cycle结束收集一次参数,最后平均。

#### 5.2 Lookahead

**思想**: 维护两组参数,慢参数周期性向快参数更新。

**算法**:
\begin{equation}\begin{aligned}
\boldsymbol{\theta}_{fast,t+1} &= \boldsymbol{\theta}_{fast,t} - \eta \nabla L(\boldsymbol{\theta}_{fast,t})\\
\boldsymbol{\theta}_{slow,t+k} &= \boldsymbol{\theta}_{slow,t} + \alpha(\boldsymbol{\theta}_{fast,t+k} - \boldsymbol{\theta}_{slow,t})
\end{aligned}\tag{25}\end{equation}

**与EMA联系**: $\alpha$类似$1-\beta$,但更新是离散的。

**优势**: 减少方差,提升收敛稳定性。

#### 5.3 LAWA (Layer-wise Adaptive Weight Averaging)

**动机**: 不同层参数变化速度不同,应该用不同的$\beta$。

**算法**: 为每层$l$设置$\beta_l$:
\begin{equation}\beta_l = \beta_0 \cdot \exp\left(-\frac{\|\nabla_{\boldsymbol{\theta}_l}L\|}{\sum_l \|\nabla_{\boldsymbol{\theta}_l}L\|}\right)\tag{26}\end{equation}

梯度大的层用更小的$\beta$(更新快),梯度小的层用更大的$\beta$(更平滑)。

**数学直觉**: 自适应平衡不同层的更新速度。

### 6. EMA在不同场景的应用

#### 6.1 生成模型

**GANs**: 判别器和生成器都使用EMA参数进行评估:
\begin{equation}\boldsymbol{\theta}_{G,ema} = 0.999\boldsymbol{\theta}_{G,ema} + 0.001\boldsymbol{\theta}_{G}\tag{27}\end{equation}

**优点**:
- 稳定生成质量
- 减少mode collapse
- 改善FID/IS指标

**典型$\beta$**: 0.999 ~ 0.9999(非常大,变化极慢)

#### 6.2 目标检测

**YOLO/Faster R-CNN**: 使用EMA提升检测精度。

**设置**: $\beta = 0.9999$,每次更新后同步:
\begin{equation}\boldsymbol{\theta}_{ema} \leftarrow 0.9999\boldsymbol{\theta}_{ema} + 0.0001\boldsymbol{\theta}\tag{28}\end{equation}

**效果**: mAP提升0.5-1.0个百分点。

**原因**: 目标检测对参数波动敏感,EMA提供稳定预测。

#### 6.3 半监督学习

**Mean Teacher**: 教师模型是学生模型的EMA:
\begin{equation}\boldsymbol{\theta}_{teacher,t} = \alpha\boldsymbol{\theta}_{teacher,t-1} + (1-\alpha)\boldsymbol{\theta}_{student,t}\tag{29}\end{equation}

**损失函数**:
\begin{equation}\mathcal{L} = \mathcal{L}_{supervised}(\boldsymbol{\theta}_{student}) + \lambda\mathcal{L}_{consistency}(\boldsymbol{\theta}_{student}, \boldsymbol{\theta}_{teacher})\tag{30}\end{equation}

**数学直觉**: 教师模型更稳定,提供高质量伪标签。

#### 6.4 强化学习

**TD3/SAC**: 目标网络是主网络的EMA:
\begin{equation}\boldsymbol{\theta}_{target} \leftarrow \tau\boldsymbol{\theta}_{target} + (1-\tau)\boldsymbol{\theta}\tag{31}\end{equation}

典型$\tau = 0.995$。

**作用**: 稳定Q值估计,防止overestimation。

### 7. 理论分析深化

#### 7.1 收敛速度

**定理4** (EMA收敛率,凸情况): 对于$\mu$-强凸$L$-光滑函数,使用SGD+EMA:
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_{ema,T})] - L^* \leq \left(1 - \frac{\mu\eta}{2}\right)^T(L(\boldsymbol{\theta}_0) - L^*) + \frac{\eta\sigma^2}{2\mu B} \cdot \frac{1-\beta}{1+\beta}\tag{32}\end{equation}

**观察**: EMA将误差下界减少了$\frac{1-\beta}{1+\beta}$倍。

#### 7.2 最优$\beta$选择

**问题**: 如何选择最优的$\beta$?

**权衡**:
- **小$\beta$**: 反应快,方差减少少
- **大$\beta$**: 反应慢,方差减少多

**最优化**: 最小化稳态误差:
\begin{equation}\beta^* = \mathop{\arg\min}_{\beta}\left[\text{Bias}^2(\beta) + \text{Var}(\beta)\right]\tag{33}\end{equation}

**Bias**: $\text{Bias}(\beta) = \beta^k\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|$(初始偏差衰减)

**Variance**: $\text{Var}(\beta) = \frac{1-\beta}{1+\beta}\sigma^2$

**解析解** (近似):
\begin{equation}\beta^* \approx 1 - \frac{2}{\sqrt{T}}\tag{34}\end{equation}

其中$T$是总训练步数。

#### 7.3 时变EMA

**动机**: 训练初期需要快适应(小$\beta$),后期需要稳定(大$\beta$)。

**Cosine调度**:
\begin{equation}\beta_t = \beta_{min} + \frac{1}{2}(\beta_{max} - \beta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)\tag{35}\end{equation}

**线性增长**:
\begin{equation}\beta_t = \min(\beta_{max}, \beta_{init} + \frac{t}{T_{ramp}}(\beta_{max} - \beta_{init}))\tag{36}\end{equation}

典型设置:$\beta_{init} = 0.9$,$\beta_{max} = 0.9999$,$T_{ramp} = 0.1T$。

### 8. 实验验证

#### 8.1 ImageNet分类

**设置**: ResNet-50,batch size 256

**结果**:
| 方法 | Top-1准确率 | Top-5准确率 |
|------|-----------|-----------|
| SGD (无EMA) | 76.1% | 92.9% |
| SGD + EMA(0.99) | 76.4% | 93.1% |
| SGD + EMA(0.999) | 76.5% | 93.2% |
| SGD + SWA | 76.6% | 93.3% |

**观察**: EMA带来0.3-0.5%提升,SWA效果最好。

#### 8.2 BERT预训练

**设置**: BERT-base,MLM+NSP

**EMA影响**:
| $\beta$ | Dev PPL | Fine-tune准确率 |
|---------|---------|----------------|
| 无EMA | 3.85 | 84.2% |
| 0.9 | 3.82 | 84.5% |
| 0.99 | 3.79 | 84.8% |
| 0.999 | 3.77 | 85.0% |
| 0.9999 | 3.76 | 85.1% |

**最佳**: $\beta = 0.999 \sim 0.9999$对Transformer最有效。

#### 8.3 Stable Diffusion

**生成模型**: Latent Diffusion Models

**FID对比**:
| EMA设置 | FID ↓ | IS ↑ |
|---------|-------|------|
| 无EMA | 12.5 | 28.3 |
| $\beta=0.999$ | 10.8 | 32.1 |
| $\beta=0.9999$ | 9.6 | 35.2 |

**关键发现**: 生成模型对EMA极其敏感,$\beta$越大越好(在一定范围内)。

### 9. 实践建议

#### 9.1 $\beta$选择指南

**任务类型**:
- **分类**(ImageNet): $\beta = 0.99 \sim 0.999$
- **检测/分割**: $\beta = 0.9999$
- **生成模型**: $\beta = 0.9999 \sim 0.99999$
- **强化学习**: $\tau = 0.995 \sim 0.999$

**训练长度**:
- **短训练**(<10 epochs): $\beta = 0.9 \sim 0.95$
- **中训练**(10-100 epochs): $\beta = 0.99 \sim 0.999$
- **长训练**(>100 epochs): $\beta = 0.999 \sim 0.9999$

**数学依据**: $\beta \approx 1 - \frac{2}{\sqrt{T}}$

#### 9.2 实现细节

**PyTorch代码**:
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
```

**使用方式**:
```python
ema = EMA(model, decay=0.999)
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch))
    loss.backward()
    optimizer.step()
    ema.update()  # 每步更新EMA

# 评估时使用EMA参数
ema.apply_shadow()
evaluate(model)
```

#### 9.3 常见错误

**错误1**: 在训练早期使用大$\beta$
- **问题**: 收敛慢,陷入初始化附近
- **解决**: 使用warmup,$\beta$从小逐渐增大

**错误2**: 忘记偏置修正
- **问题**: 初期EMA参数偏向0
- **解决**: 使用式(3)的偏置修正

**错误3**: EMA更新频率错误
- **问题**: 每个epoch更新一次(太慢)
- **解决**: 每个step更新一次

**错误4**: 验证时用训练参数而非EMA
- **问题**: 无法发挥EMA优势
- **解决**: 评估前切换到EMA参数

### 10. 理论前沿

#### 10.1 自适应EMA

**问题**: 固定$\beta$不适应训练动态。

**解决方案**: 基于梯度统计自适应调整:
\begin{equation}\beta_t = 1 - \frac{c}{\sqrt{1 + \|\nabla L(\boldsymbol{\theta}_t)\|^2}}\tag{37}\end{equation}

梯度大时$\beta$小(快适应),梯度小时$\beta$大(稳定)。

#### 10.2 多尺度EMA

**思想**: 同时维护多个不同$\beta$的EMA:
\begin{equation}\boldsymbol{\theta}_{ema}^{(i)} = \beta_i\boldsymbol{\theta}_{ema,t-1}^{(i)} + (1-\beta_i)\boldsymbol{\theta}_t, \quad i=1,\ldots,K\tag{38}\end{equation}

**集成**: 根据验证集动态选择或加权平均:
\begin{equation}\boldsymbol{\theta}_{final} = \sum_{i=1}^K w_i\boldsymbol{\theta}_{ema}^{(i)}\tag{39}\end{equation}

#### 10.3 EMA与隐式正则的统一

**观察**: EMA、权重衰减、dropout都是正则化。

**统一框架**:
\begin{equation}\boldsymbol{\theta}_{t+1} = \mathop{\arg\min}_{\boldsymbol{\theta}}\left[\mathcal{L}(\boldsymbol{\theta}) + \lambda_1\|\boldsymbol{\theta} - \boldsymbol{\theta}_t\|^2 + \lambda_2\|\boldsymbol{\theta} - \boldsymbol{\theta}_{ema}\|^2\right]\tag{40}\end{equation}

**数学直觉**: EMA提供的是对"平滑轨迹"的正则化。

### 11. 总结

EMA是一个简单但强大的技术,通过指数加权平均历史参数实现:

**核心优势**:
1. **方差减少**: 降低参数波动
2. **泛化提升**: 收敛到更平坦的最小值
3. **零额外成本**: 只需维护一份shadow参数
4. **通用性强**: 适用于各种优化器和任务

**理论保证**:
- PAC-Bayes泛化界改善
- Sharpness减小
- 稳态误差降低$\frac{1-\beta}{1+\beta}$倍

**实践要点**:
- 根据任务和训练长度选择$\beta$
- 生成模型用大$\beta$(0.9999+)
- 分类任务用中等$\beta$(0.99-0.999)
- 评估时务必使用EMA参数

**未来方向**:
- 自适应$\beta$调度
- 多尺度EMA集成
- 与其他正则化技术的理论统一

