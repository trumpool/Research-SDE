# 《用波动率调制神经 SDE 建模危机动力学》教学指南

> 本指南基于论文 *Modeling Crisis Dynamics with Volatility-Modulated Neural SDEs*，分两部分：
> 1. 逐步还原研究是如何从动机到实验方案被构建出来的
> 2. 将论文每一部分精确对应到 `src/sv_nsde/` 下的代码

---

## 第一部分：研究思路的构建过程

### 第一步：发现问题——找到研究的"缺口"

任何研究都从一个真实的、尚未被解决的问题开始。

**场景观察：** COVID-19 疫情期间，微博上的话题讨论出现了一种特殊现象——某些话题会在沉寂数小时后突然爆炸式传播，这种爆发与官方发布具体数据无关，而是发生在"信息极度混乱、谣言四起"的时期。

**已有工具的局限：** 经典的神经时序点过程（Neural TPPs），如 RMTPP 和 Neural Hawkes，有一个核心假设：**事件的发生率由系统的确定性状态驱动**（比如"确诊人数增加 → 发帖量增加"）。

**这个假设在危机初期失效了**，因为：

> 危机初期，舆论的"均值漂移"（Mean Drift）可能还很稳定，但"扩散项"（Volatility）已经在剧烈飙升。换言之，**爆发是由"不知道该信什么"触发的，而不是由某条具体的坏消息触发的**。

**核心洞察（这是论文最重要的一句话）：**
> 波动率不是背景噪声，而是一个能够**预示结构性突变**的主动信号。

---

### 第二步：找灵感——跨领域借鉴

找到了问题，下一步是找解决方案。研究者把目光投向了**金融数学**。

**为什么借鉴金融模型？**

金融市场和社交媒体舆论有惊人的相似性：
- 股票价格也有"趋势"和"恐慌性暴跌"
- 金融学家早就建立了一套专门处理"随机波动率"的数学框架

**核心借鉴：Heston 模型（1993）**

Heston 模型将资产价格分解为两个耦合过程：
1. **价格主过程**：描述资产价格的趋势演化
2. **波动率子过程**：描述不确定性本身的随机演化（CIR 过程）

研究者的创新在于：**把这套金融工具"移植"到社交媒体语义建模中**，用"语义状态"替代"资产价格"，用"舆论不确定性"替代"资产波动率"。

---

### 第三步：数据探索——用数据验证动机

在搭建任何模型之前，研究者先做了**探索性数据分析（EDA）**来确认自己的直觉是对的。

**数据集：Weibo-COV**（2020 年 1 月至 12 月的微博文本流）

**分析一：级联规模分布**

绘制所有转发级联的规模分布图，发现：
- 绝大多数级联极小（大家都没转）
- 极少数级联达到 $10^6$ 量级（病毒式传播）
- 这是典型的**幂律分布（Power-law）**，说明背后存在"自激"或"突变"机制，而不是简单的线性增长

**分析二：大级联的生长轨迹**

追踪 Top 10 大级联的增长曲线，发现：
- 不是平滑增长，而是**"阶梯状"跃迁**
- 某些话题会沉寂数小时后突然爆发

**结论：** 这两个发现共同支持了引入 **Jump-Diffusion 机制**（跳跃-扩散）的必要性。数据本身在说话。

---

### 第四步：语义编码——把文字变成数字

在建模之前，需要把微博文本转换为模型可以处理的数值向量。

**为什么选 RoBERTa-wwm-ext？**

- 专门针对**中文**优化的预训练语言模型
- 采用**全词掩码（Whole Word Masking, WWM）**：掩码时以整个词（而非单个汉字）为单位，能学到更完整的语义单元
- 对于"气溶胶"、"方舱医院"这类疫情专有名词有更好的理解

**编码流程：**

```
微博文本 T_i
    ↓  Tokenizer（截断/填充至 L=128 个 token）
    ↓  RoBERTa 编码器
    → H_i ∈ R^{L × 768}（每个 token 的向量）
    → 取 [CLS] 标记的向量 h_i^{CLS} ∈ R^{768}（整句的表示）
    ↓  线性投影层（降维）
    → x_i ∈ R^{d_in}（例如 d_in = 32 或 64）
```

**为什么要降维？** 768 维太高了，直接输入 SDE 会造成"维度灾难"，增加训练难度。线性投影层是一个可训练的参数矩阵，在训练中自动学习最有用的压缩方式。

---

### 第五步：构建模型——SV-NSDE 的数学骨架

#### 5.1 两个状态变量的含义

| 变量 | 含义 | 对应现实 |
|------|------|----------|
| $z(t)$ | 语义趋势状态 | 舆论当前讨论的主流话题方向 |
| $v(t)$ | 瞬时波动率 | 舆论的混乱程度、不确定性强度 |

#### 5.2 耦合 SDE 方程组

$$
dz(t) = \underbrace{\mu_\theta(z(t), t)dt}_{\text{趋势项（神经网络）}} + \underbrace{\sqrt{v(t)} \odot dW_z(t)}_{\text{随机扩散（由波动率调制）}} + \underbrace{J_\phi(z(t^-), x_i)dN(t)}_{\text{事件跳跃}}
$$

$$
dv(t) = \underbrace{\kappa(\theta - v(t))dt}_{\text{均值回归（恐慌终将冷却）}} + \underbrace{\xi\sqrt{v(t)} \odot dW_v(t)}_{\text{波动率的波动}}
$$

**逐项解读：**

- **趋势项** $\mu_\theta$：一个神经网络，学习舆论"正常状态"下的演化规律
- **随机扩散项** $\sqrt{v(t)} \odot dW_z$：**关键设计！** 扩散强度不是固定的，而是由 $v(t)$ 动态调制。波动率高 → 语义轨迹更不稳定
- **跳跃项** $J_\phi dN(t)$：每当一条新微博 $(t_i, x_i)$ 到达，立即对潜在状态施加一个冲击，更新舆论轨迹
- **均值回归** $\kappa(\theta - v(t))$：CIR 过程的核心特性，保证波动率不会无限增长，最终回归到均值 $\theta$（恐慌会消退）

#### 5.3 双通道强度函数

这是区分"热点"与"恐慌"的关键机制：

$$
\lambda(t) = \text{Softplus}\Big(\underbrace{w_{tr}^\top m(z(t))}_{\text{通道 1：趋势驱动}} + \underbrace{w_{vol}^\top g(v(t))}_{\text{通道 2：波动率驱动}} + \mu_{base}\Big)
$$

- **通道 1** 负责建模：官方发布数据 → 讨论量上升（常规热点）
- **通道 2** 负责建模：信息混乱度飙升 → 讨论量爆发（恐慌性爆发）

即使语义状态 $z(t)$ 完全不变，只要 $v(t)$ 激增，$\lambda(t)$ 也会增大。这就是该模型能捕捉"不确定性驱动爆发"的核心原理。

#### 5.4 语义解码器

为了防止 $z(t)$ 与实际文本语义"脱钩"，引入生成式解码器：

$$
p(x | z(t)) = \mathcal{N}\big(x;\ \mu_{dec}(z(t)),\ \sigma_{obs}^2 I\big)
$$

一个 MLP 网络从潜在状态 $z(t)$ 重构出原始的文本语义向量 $x$。这迫使 $z(t)$ 必须"记住"语义信息，防止模型退化（后崩塌问题）。

---

### 第六步：训练目标——ELBO 损失函数

采用变分推断框架，最大化**证据下界（ELBO）**：

$$
\mathcal{L} = \underbrace{\sum_i \log \lambda(t_i) - \int_0^T \lambda(t)dt}_{\text{事件对数似然（时间预测）}} + \underbrace{\sum_i \log p(x_i | z(t_i))}_{\text{语义重构（内容预测）}} - \underbrace{\text{KL}(Q \| P)}_{\text{正则化项}}
$$

**三项的作用：**

| 项 | 作用 |
|----|------|
| 事件对数似然 | 让模型学会准确预测**下一条微博何时出现** |
| 语义重构 | 让模型学会准确预测**下一条微博说什么** |
| KL 散度 | 约束学到的路径不要偏离 Heston 先验太远，起正则化作用 |

---

### 第七步：实验设计——如何验证模型有效

#### 7.1 数据切分策略

按危机阶段切分，而非随机切分：

```
爆发期（1-2 月）→ 高波动
平台期（3 月）   → 中波动
衰退期（4 月）   → 低波动
```

**目的：** 测试模型在不同阶段的泛化能力，尤其是爆发期的预测性能。

#### 7.2 基线模型选择（逐层递进）

| 基线 | 特点 | 缺什么 |
|------|------|--------|
| RMTPP / Neural Hawkes | 确定性 RNN | 没有随机性 |
| Latent ODE | 连续确定性演化 | 没有随机性，没有跳跃 |
| Neural Jump SDE | 有随机性，有跳跃 | **没有解耦波动率与强度**（这是本文的贡献所在） |

选基线要选"最强的邻居"——Neural Jump SDE 已经很强了，本文模型比它多的只有"波动率通道"，这样才能干净地证明这个设计的价值。

#### 7.3 评估指标

- **时间预测**：RMSE（预测下一条微博的发帖时间）
- **语义预测**：Cosine Similarity / MSE（预测下一条微博的语义内容）
- **整体拟合**：Log-Likelihood（对数似然，越高越好）

#### 7.4 消融实验——证明每个设计都有用

| 变体 | 去掉什么 | 验证什么 |
|------|----------|----------|
| w/o Volatility Channel | 去掉强度函数的第二通道 | 波动率对预测爆发到底有没有用？ |
| Deterministic Volatility | 把 $v(t)$ 换成 $z(t)$ 的确定性函数 | SDE 的随机建模是否必要？ |

消融实验的逻辑：**把你认为重要的模块一个个拆掉，如果性能下降了，就证明那个模块是有贡献的。**

---

### 研究路径总结

```
发现现象（危机初期的爆发性）
    ↓
识别现有方法的假设缺陷（忽略不确定性驱动）
    ↓
提炼核心洞察（波动率是主动信号）
    ↓
跨领域借鉴（金融 Heston 模型）
    ↓
数据驱动验证（EDA 确认幂律分布和阶梯跃迁）
    ↓
语义编码（RoBERTa → 降维向量）
    ↓
模型设计（耦合 SDE + 双通道强度函数 + 解码器）
    ↓
变分训练目标（ELBO = 时间预测 + 语义重构 + KL 正则）
    ↓
对照实验（逐层递进的基线 + 消融实验）
```

---

## 第二部分：论文每一部分 → 对应代码精确讲解

### Section 3.1 — 语义编码 → `encoder.py`

**论文公式：**
$$x_i = W_p h_i^{[CLS]} + b_p$$

**对应文件：** `src/sv_nsde/encoder.py`

```python
# encoder.py:41 — 加载中文 RoBERTa
self.bert = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

# encoder.py:55 — 线性投影层，对应公式(2) x_i = W_p h + b_p
self.projection = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(self.bert_dim, d_latent),   # 768 -> 32
)

# encoder.py:94 — 提取 [CLS] token，对应公式(1)
cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取第 0 个位置

# encoder.py:98 — 执行投影降维
x = self.projection(cls_embedding)
```

**一句话：** 文本从 `input_ids` 进去，经过 RoBERTa 出来 768 维，再被 `self.projection` 压缩成 32 维的 `x_i`。

---

### Section 3.3 — 耦合 SDE 动力学 → `sde.py`

**论文公式（方程 3）：**
$$dz = \mu_\theta dt + \sqrt{v(t)} \odot dW_z + J_\phi dN$$
$$dv = \kappa(\theta - v)dt + \xi\sqrt{v} \odot dW_v$$

**对应文件：** `src/sv_nsde/sde.py`

**SDE 参数（可学习）：**
```python
# sde.py:62-64 — κ、θ、ξ 存成 log 再 exp 保证为正
self.log_kappa = nn.Parameter(torch.tensor(config.kappa_init).log())
self.log_theta = nn.Parameter(torch.tensor(config.theta_init).log())
self.log_xi    = nn.Parameter(torch.tensor(config.xi_init).log())
```

**三项分别对应代码：**

| 论文中的项 | 代码位置 | 具体实现 |
|---|---|---|
| 趋势项 $\mu_\theta(z,t)$ | `sde.py:52` | `self.drift_net`：3层MLP，输入`[z, t]` |
| 扩散项 $\sqrt{v(t)} \odot dW_z$ | `sde.py:139` | `diffusion_z(v)` 返回 `sqrt(clamp(v))` |
| 跳跃项 $J_\phi(z^-, x_i)$ | `sde.py:68` | `self.jump_net`：拼接 `[z, x]` 后过 MLP+Tanh |
| CIR 均值回归 $\kappa(\theta-v)$ | `sde.py:125` | `drift_v(v)` 直接返回 `kappa * (theta - v)` |
| CIR 扩散 $\xi\sqrt{v} \odot dW_v$ | `sde.py:153` | `diffusion_v(v)` 返回 `xi * sqrt(clamp(v))` |

**Euler-Maruyama 数值积分（把公式真正跑起来）：**
```python
# sde.py:271-283 — 每步离散化更新
dW_z = torch.randn_like(z) * sqrt_dt      # 布朗运动增量
z_next = z + drift_z * dt + diff_z * dW_z  # dz 方程
v_next = v + drift_v * dt + diff_v * dW_v  # dv 方程
v_next = torch.clamp(v_next, min=1e-6)     # 保证 v > 0（反射边界）
```

**事件跳跃的触发逻辑（`solve` 函数里）：**
```python
# sde.py:341-356 — 当事件到来，先步进到事件时刻，再施加跳跃
if event_times[event_idx] <= next_t:
    z, v = self.euler_maruyama_step(...)   # 步进到事件时刻
    z_at_events.append(z.clone())          # 记录跳前状态
    z = z + self.sde_func.jump(z, x_i)    # 施加跳跃 J_φ
```

---

### Section 3.4 — 双通道强度函数 → `intensity.py`

**论文公式（方程 4）：**
$$\lambda(t) = \text{Softplus}\big(w_{tr}^\top m(z) + w_{vol}^\top g(v) + \mu_{base}\big)$$

**对应文件：** `src/sv_nsde/intensity.py`

```python
# intensity.py:39-45 — 通道1：趋势网络 m(z)
self.trend_net = nn.Sequential(
    nn.Linear(d_latent, d_hidden), nn.SiLU(),
    nn.Linear(d_hidden, d_hidden//2), nn.SiLU(),
    nn.Linear(d_hidden//2, 1),
)

# intensity.py:48-55 — 通道2：波动率网络 g(v)
self.vol_net = nn.Sequential(
    nn.Linear(d_latent, d_hidden), nn.SiLU(),
    nn.Linear(d_hidden, d_hidden//2), nn.SiLU(),
    nn.Linear(d_hidden//2, 1),
)

# intensity.py:103-124 — forward 里的合并与 Softplus
trend_contrib = self.trend_net(z)
vol_contrib   = self.vol_net(v)
pre_activation = w_trend * trend_contrib + w_vol * vol_contrib + self.mu_base
intensity = F.softplus(pre_activation).squeeze(-1)
```

**论文还提到"门控机制"（代码里有扩展实现）：**
```python
# intensity.py:59-65 — 可选的动态门控：由 [z, v] 联合决定两个通道的权重
if use_gating:
    self.gate_net = nn.Sequential(
        nn.Linear(d_latent * 2, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, 2),
        nn.Softmax(dim=-1),     # 两个通道的权重加和为1
    )
```

**区分恐慌与热点的核心接口：**
```python
# intensity.py:126-134 — return_components=True 时返回每个通道的贡献值
components = {
    "trend_contrib": ...,   # 趋势通道的贡献
    "vol_contrib":   ...,   # 波动率通道的贡献
}
```

这就是 `model.py` 里 `get_volatility_decomposition()` 能区分"热点"与"恐慌"的底层依据。

---

### Section 3.5 — 语义解码器 → `decoder.py`

**论文公式（方程 5）：**
$$p(x|z(t)) = \mathcal{N}(x;\ \mu_{dec}(z(t)),\ \sigma_{obs}^2 I)$$

**对应文件：** `src/sv_nsde/decoder.py`

```python
# decoder.py:42-48 — MLP 解码器 μ_dec(z)
self.decoder = nn.Sequential(
    nn.Linear(d_latent, d_hidden), nn.SiLU(),
    nn.Linear(d_hidden, d_hidden), nn.SiLU(),
    nn.Linear(d_hidden, d_latent),   # 输出与输入同维，重构 x_i
)

# decoder.py:110-115 — 计算高斯对数似然 log p(x|z)
log_prob_per_dim = (
    -0.5 * log(2π)
    - self.log_sigma_obs           # -log σ
    - 0.5 * (x - mu)**2 / var      # 马氏距离项
)
```

`sigma_obs` 默认固定为 0.1（超参），但可以设 `learn_variance=True` 让它也被训练。

---

### Section 3.6 — ELBO 训练目标 → `model.py`

**论文公式（方程 6）：**
$$\mathcal{L} = \underbrace{\sum_i \log\lambda(t_i) - \int\lambda\, dt}_{\text{点过程似然}} + \underbrace{\sum_i \log p(x_i|z(t_i))}_{\text{重构}} - \underbrace{KL}_{\text{正则}}$$

**对应文件：** `src/sv_nsde/model.py`，`compute_loss()` 方法

```python
# model.py:217-220 — 第1项：Σ log λ(t_i)
for i in range(n_events):
    lam = self.intensity(z_events[i], v_events[i])
    log_intensity_sum += torch.log(lam + 1e-8)

# model.py:225-226 — 第2项：-∫λ(t)dt（梯形积分）
integral = self.intensity.compute_integral(z_traj, v_traj, times)

# model.py:231-235 — 第3项：Σ log p(x_i|z(t_i))
log_p = self.decoder.log_prob(x_i, z_events[i])
recon_loss += log_p

# model.py:241 — 第4项：KL 散度（简化版）
kl_loss = self.kl_weight * (z_traj ** 2).mean()

# model.py:244-245 — 组合 ELBO，取负值作为 loss
elbo = log_intensity_sum - integral + recon_loss - kl_loss
loss = -elbo
```

> **注意：** 论文中 KL 项说要用 Girsanov 定理，代码里用了简化版（对 z 轨迹做 L2 正则），这是工程近似。

---

### Section 4.2 — 基线模型 → `baselines.py`

**对应文件：** `src/sv_nsde/baselines.py`

| 论文基线 | 代码中的类 |
|---|---|
| RMTPP | `RMTPP` |
| Neural Hawkes | `NeuralHawkes` |
| Latent ODE | `LatentODE` |
| Neural Jump SDE（最强基线） | `NeuralJumpSDE` |

**两个消融实验也直接实现了：**
```python
# baselines.py — 消融1：去掉波动率通道
class SVNSDENoVolatilityChannel(...)     # w/o Volatility Channel

# baselines.py — 消融2：波动率改为确定性
class SVNSDEDeterministicVolatility(...)  # Deterministic v(t) = h(z(t))
```

---

### Section 4 — 实验流程 → `experiment.py` + `evaluate.py`

`experiment.py` 是最高层的"指挥官"：

| 函数 | 作用 |
|---|---|
| `precompute_bert_embeddings()` | 批量跑 RoBERTa，结果存盘避免重复计算 |
| `train_model()` | 训练单个模型 |
| `train_all_baselines()` | 批量训练所有基线 |
| `run_comparison()` | 统一评估，输出对比报告 |

`evaluate.py` 负责三类指标的计算：

```python
# evaluate.py — 对应论文 4.3 评价指标
class MetricComputer:
    def time_prediction_rmse(...)      # 时间预测 RMSE
    def semantic_prediction_mse(...)   # 语义预测 MSE / Cosine Similarity
    def log_likelihood(...)            # Log-Likelihood

class VolatilityAnalyzer:
    def identify_panic_events(...)     # 识别哪些事件是"恐慌驱动"的
```

---

## 整体数据流（代码串联图）

```
微博文本 T_i
    │
    ▼  encoder.py: SemanticEncoder
    x_i ∈ R^{32}   ← RoBERTa [CLS] → Linear 投影
    │
    ▼  sde.py: NeuralHestonSDE.solve()
    z(t), v(t)      ← Euler-Maruyama + 事件跳跃
    │
    ├──▶  intensity.py: DualChannelIntensity
    │         trend_net(z) + vol_net(v) → λ(t)
    │
    └──▶  decoder.py: SemanticDecoder
              μ_dec(z) → 重构 x_i
    │
    ▼  model.py: SVNSDE.compute_loss()
    ELBO = Σlog λ − ∫λdt + Σlog p(x|z) − KL
    │
    ▼  experiment.py + evaluate.py
    训练 / 评估 / 基线对比
```

---

## 关键学习要点

1. **好的研究问题来自观察具体现象**，不是凭空想象。危机初期的"不确定性爆发"是整篇论文的出发点。

2. **跨领域借鉴是重要的创新来源**。把金融的 Heston 模型移植到社交媒体，本质上是发现了两个领域问题结构的相似性。

3. **EDA 不是可有可无的**。数据探索不仅验证了动机，还直接指导了模型设计（幂律分布 → 需要跳跃机制；阶梯跃迁 → 需要 Jump-Diffusion）。

4. **每个设计决策都要能被实验验证**。双通道强度函数、随机波动率建模，都对应了明确的消融实验。

5. **基线选择要有层次感**，从简单到复杂，最后一个基线应该只比你的模型少"你声称的核心贡献"。

6. **代码与论文公式一一对应**。每个公式都能在代码里找到直接对应的 `nn.Module` 或函数，这种对应关系是工程实现研究的最佳实践。
