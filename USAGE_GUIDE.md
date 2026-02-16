# SV-NSDE 使用指南

**Semantic Volatility-Modulated Neural SDE for Crisis Dynamics**

一个用于危机动态建模的神经点过程模型，基于论文《Modeling Crisis Dynamics with Volatility-Modulated Neural SDEs: Distinguishing Panic from Trends in COVID-19 Information Diffusion》。

---

## 项目概述

本项目实现了 **SV-NSDE** 模型及多个基线模型，用于分析社交媒体（微博）信息传播的动态，特别是在危机期间区分恐慌驱动的爆发与趋势驱动的热点。

**核心创新:**
- 将舆论建模为两个耦合的随机过程：语义趋势 z(t) 和波动率 v(t)
- 波动率作为主动信号预测事件强度，而非被动噪声
- 能够识别 "panic-driven bursts" 与常规热点

---

## 快速开始 (5分钟)

### 1. 克隆项目

```bash
git clone <项目地址>
cd Research
```

### 2. 创建虚拟环境

```bash
# 使用 uv (推荐，快速)
uv venv
source .venv/bin/activate

# 或使用 conda/venv
conda create -n sv_nsde python=3.10
conda activate sv_nsde
```

### 3. 安装依赖

```bash
# 使用 uv (快速)
uv pip install -e ".[dev,viz]"

# 或使用 pip
pip install -e ".[dev,viz]"
```

### 4. 验证安装

```bash
python -c "from sv_nsde import SVNSDELite; print('✓ Installation successful!')"
```

### 5. 运行示例

```bash
cd examples
python quick_start.py
```

---

## 详细设置指南

### 环境要求

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **内存**: 至少 8GB RAM
- **硬盘**: 至少 5GB (用于数据)

### 完整依赖

```
torch>=2.0.0
numpy>=1.21.0
transformers>=4.30.0  # 用于RoBERTa编码器
tqdm>=4.60.0
datasets>=4.0.0       # 用于数据处理
```

### 可选依赖

```bash
# 可视化
uv pip install matplotlib seaborn

# 实验跟踪
uv pip install wandb tensorboard

# 开发工具
uv pip install pytest black isort
```

---

## 数据下载

### 方案 A: 使用合成数据 (推荐用于快速测试)

项目已内置合成数据生成器，无需下载：

```python
from sv_nsde import generate_synthetic_weibo_data

# 生成1000个级联的数据
df = generate_synthetic_weibo_data(
    n_cascades=1000,
    output_path="data/weibo_synthetic.csv",
    seed=42
)
```

### 方案 B: 使用真实 Weibo-COV 数据

#### 下载步骤

1. **访问官方数据源**

   论文链接: https://arxiv.org/abs/2005.09174

   GitHub: https://github.com/nghuyong/weibo-cov

2. **下载数据**

   数据托管在百度网盘 (需百度账号):

   - **Weibo-COV 1.0**: https://pan.baidu.com/s/1SwbkEnuXrUFmRj1lx_AQlg?pwd=r8gn
   - **Weibo-COV 2.0** (推荐): https://pan.baidu.com/s/1mxU5RbnGBNRvR4Ci-9d0Hg?pwd=jffm

3. **提取文件**

   ```bash
   # 解压到 data/ 目录
   unzip weibo-cov.zip -d data/
   ```

#### 数据格式

CSV文件包含以下列:

```
_id          - 推文ID
user_id      - 用户ID (哈希)
created_at   - 发布时间 (格式: YYYY-MM-DD HH:MM:SS)
content      - 推文内容 (中文)
like_num     - 点赞数
repost_num   - 转发数
comment_num  - 评论数
origin_weibo - 原推文ID (转发链)
geo_info     - 地理信息
```

**数据规模:**

| 版本 | 时间范围 | 推文数 | 源推文数 |
|------|---------|--------|---------|
| 1.0 | 2019-12-01 ~ 2020-04-30 | 4089万 | 6.93亿 |
| 2.0 | 2019-12-01 ~ 2020-12-30 | 6518万 | 26.15亿 |

---

## 使用方式

### 1. 加载数据并构建级联

```python
from sv_nsde import WeiboCOVLoader

# 加载真实数据
loader = WeiboCOVLoader("data/weibo_cov.csv")
loader.load(nrows=1000000)  # 可选: 限制行数

# 构建级联 (转发链)
cascades = loader.build_cascades(
    min_size=5,      # 最小级联大小
    max_size=500,    # 最大级联大小
    time_unit="hours"
)

# 获取统计信息
stats = loader.get_statistics()
print(f"级联数: {stats['num_cascades']}")
print(f"平均大小: {stats['size_mean']:.1f}")

# 按时间分割 (按危机阶段)
train, val, test = loader.split_by_time(
    train_end="2020-02-29",  # 爆发期 (高波动)
    val_end="2020-03-31"     # 平台期
)
```

### 2. 模型训练

```python
from sv_nsde import SVNSDELite
from sv_nsde.train import Trainer, CascadeDataset
from torch.utils.data import DataLoader

# 创建模型
model = SVNSDELite(
    d_input=768,      # 输入维度 (BERT)
    d_latent=32,      # 潜在空间维度
    d_hidden=64       # 隐藏层维度
)

# 创建数据集
dataset = CascadeDataset(
    cascades=train,
    precomputed_embeddings=embeddings,  # 可选: 预计算的嵌入
    max_events=100,
    max_seq_length=128
)

# 创建训练器
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=100,
    device="cuda"  # 使用 GPU
)

# 训练
history = trainer.train()
```

### 3. 评估与对比

```python
from sv_nsde import Evaluator, get_baseline

# 初始化模型
models = {
    "SV-NSDE": model,
    "RMTPP": get_baseline("rmtpp", d_input=768, d_latent=32),
    "Neural Hawkes": get_baseline("neural_hawkes", d_input=768, d_latent=32),
    "Latent ODE": get_baseline("latent_ode", d_input=768, d_latent=32),
    "Neural Jump SDE": get_baseline("neural_jump_sde", d_input=768, d_latent=32),
}

# 评估
evaluator = Evaluator(device="cuda")
results = evaluator.evaluate_all(models, test, embeddings)

# 查看结果
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  时间预测 RMSE: {metrics.time_rmse:.4f}")
    print(f"  语义预测 Cosine: {metrics.semantic_cosine:.4f}")
    print(f"  Log-Likelihood: {metrics.log_likelihood:.2f}")
```

### 4. 波动率分析 (核心功能)

```python
from sv_nsde import VolatilityAnalyzer

analyzer = VolatilityAnalyzer(model, device="cuda")

# 分析单个级联
analysis = analyzer.analyze_cascade(cascade, event_embeddings)
print(f"恐慌驱动的事件: {analysis['num_panic_events']}/{len(cascade)}")
print(f"恐慌比例: {analysis['panic_ratio']:.2%}")

# 找出所有恐慌事件
burst_events = analyzer.find_burst_events(
    test_cascades,
    embeddings_dict,
    volatility_threshold=0.6  # 波动率贡献度 > 60%
)

# 分析结果
for event in burst_events[:5]:
    print(f"时间: {event['time']:.2f}h, 波动率比: {event['volatility_ratio']:.2f}")
    print(f"内容: {event['text']}")
```

---

## 完整工作流程示例

### 端到端示例 (使用合成数据)

```python
import torch
from sv_nsde import (
    generate_synthetic_weibo_data,
    WeiboCOVLoader,
    SVNSDELite,
    Trainer,
    CascadeDataset,
    Evaluator,
)

# 1. 生成数据
print("Step 1: 生成合成数据...")
df = generate_synthetic_weibo_data(n_cascades=500, output_path="data/syn.csv")

# 2. 加载并处理
print("Step 2: 加载数据...")
loader = WeiboCOVLoader("data/syn.csv")
loader.load()
cascades = loader.build_cascades(min_size=5, max_size=200)
train, val, test = loader.split_by_time()

# 3. 创建数据集
print("Step 3: 准备数据集...")
train_dataset = CascadeDataset(
    train,
    precomputed_embeddings={c.cascade_id: torch.randn(c.size, 32) for c in train}
)
val_dataset = CascadeDataset(
    val,
    precomputed_embeddings={c.cascade_id: torch.randn(c.size, 32) for c in val}
)

# 4. 训练
print("Step 4: 训练模型...")
model = SVNSDELite(d_input=32, d_latent=32, d_hidden=64)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-3,
    batch_size=8,
    num_epochs=50,
    device="cpu"
)
trainer.train()

# 5. 评估
print("Step 5: 评估...")
test_embeddings = {c.cascade_id: torch.randn(c.size, 32) for c in test}
evaluator = Evaluator(device="cpu")
metrics = evaluator.evaluate_model(model, test, test_embeddings)

print(f"\n最终性能:")
print(f"  Semantic Cosine: {metrics.semantic_cosine:.4f}")
print(f"  Log-Likelihood: {metrics.log_likelihood:.2f}")
```

### 运行评估脚本

```bash
# 快速评估 (小数据集)
python scripts/run_evaluation.py --quick

# 完整评估
python scripts/run_evaluation.py \
    --data data/weibo_cov.csv \
    --output results/ \
    --device cuda

# 自定义参数
python scripts/run_evaluation.py \
    --data data/weibo_cov.csv \
    --d_latent 64 \
    --d_hidden 128 \
    --max_rows 2000000
```

---

## 项目结构

```
Research/
├── README.md                     # 项目简介
├── USAGE_GUIDE.md               # 本文档
├── requirements.txt              # 依赖列表
├── pyproject.toml               # 项目配置
├── setup.py                     # 安装脚本
│
├── data/
│   ├── synthetic_weibo_cov.csv  # 合成数据 (预生成)
│   └── weibo_cov.csv            # 真实数据 (待下载)
│
├── results/
│   ├── report.txt               # 评估报告
│   └── results.json             # 详细结果
│
├── examples/
│   └── quick_start.py           # 快速示例
│
├── scripts/
│   └── run_evaluation.py        # 评估脚本
│
└── src/sv_nsde/
    ├── __init__.py              # 包初始化
    ├── encoder.py               # RoBERTa编码器
    ├── sde.py                   # Neural Heston SDE
    ├── intensity.py             # 双通道强度函数
    ├── decoder.py               # 语义解码器
    ├── model.py                 # 完整SV-NSDE模型
    ├── baselines.py             # 6个基线模型
    ├── data.py                  # 数据加载
    ├── train.py                 # 训练脚本
    └── evaluate.py              # 评估指标
```

---

## 常见问题

### Q1: 如何使用 GPU 加速?

```python
# 检查 GPU 可用性
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# 在模型中指定GPU
model = SVNSDELite(...)
model = model.to("cuda")

# 在Trainer中指定
trainer = Trainer(..., device="cuda")
```

### Q2: 内存不足怎么办?

```python
# 减少批大小
trainer = Trainer(..., batch_size=4)  # 默认16

# 减少最大事件数
dataset = CascadeDataset(..., max_events=50)  # 默认100

# 减少数据行数
loader.load(nrows=100000)  # 而不是全部
```

### Q3: 如何使用预训练的模型权重?

```python
import torch

# 保存模型
torch.save(model.state_dict(), "model_weights.pt")

# 加载模型
model = SVNSDELite(...)
model.load_state_dict(torch.load("model_weights.pt"))
model.eval()
```

### Q4: 数据应该如何预处理?

```python
from transformers import AutoTokenizer

# 使用RoBERTa进行文本编码
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 预计算嵌入 (加快训练)
from sv_nsde.data import precompute_embeddings
from sv_nsde import SemanticEncoder

encoder = SemanticEncoder(pretrained_model="hfl/chinese-roberta-wwm-ext")
embeddings = precompute_embeddings(
    cascades=train,
    encoder=encoder,
    tokenizer=tokenizer,
    output_path="embeddings_train.pt"
)
```

### Q5: 如何复现论文结果?

1. 下载完整的 Weibo-COV 2.0 数据
2. 使用推荐的超参数:
   - d_latent = 64
   - d_hidden = 128
   - learning_rate = 1e-4
   - batch_size = 16
   - num_epochs = 100

3. 运行完整评估:
   ```bash
   python scripts/run_evaluation.py \
       --data data/weibo_cov.csv \
       --d_latent 64 \
       --d_hidden 128 \
       --device cuda
   ```

### Q6: 如何调试模型?

```bash
# 启用日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 运行小数据集测试
python scripts/run_evaluation.py --quick

# 检查模型架构
from sv_nsde import SVNSDELite
model = SVNSDELite()
print(model)  # 打印所有层

# 检查输入/输出形状
import torch
x = torch.randn(10, 32)  # 10个事件, 32维嵌入
output = model(...)
print(output.shape)
```

---

## 模型架构概览

### SV-NSDE 的5大组件

1. **SemanticEncoder** (encoder.py)
   - 使用 RoBERTa 编码微博文本
   - 输出: 768维语义向量 → 投影到d_latent维

2. **NeuralHestonSDE** (sde.py)
   - 耦合SDE系统:
     - dz(t) = μ(z,t)dt + √v(t)dW_z(t) + J(z,x)dN(t)
     - dv(t) = κ(θ-v)dt + ξ√v dW_v(t)
   - z(t): 语义趋势
   - v(t): 波动率过程 (CIR)

3. **DualChannelIntensity** (intensity.py)
   - λ(t) = Softplus(w_tr·m(z(t)) + w_vol·g(v(t)) + μ_base)
   - Channel 1: 趋势驱动
   - Channel 2: 波动率驱动 (关键创新!)

4. **SemanticDecoder** (decoder.py)
   - p(x|z(t)) = N(x; μ_dec(z), σ²_obs·I)
   - 防止后验坍塌 (Posterior Collapse)

5. **Full SVNSDE** (model.py)
   - 组合所有组件
   - ELBO 损失: L = log λ - ∫λdt + log p(x|z) - KL

### 基线模型

| 名称 | 核心特性 | 代码 |
|------|---------|------|
| RMTPP | RNN + 指数强度 | baselines.py:L45 |
| Neural Hawkes | LSTM + 衰减 | baselines.py:L105 |
| Latent ODE | 确定性ODE | baselines.py:L225 |
| Neural Jump SDE | SDE+跳跃(无v分解) | baselines.py:L345 |

---

## 论文引用

```bibtex
@inproceedings{chen2026svnsde,
  title={Modeling Crisis Dynamics with Volatility-Modulated Neural SDEs:
         Distinguishing Panic from Trends in COVID-19 Information Diffusion},
  author={Chen, Zirui},
  booktitle={Proceedings of [Conference]},
  year={2026}
}
```

---

## 许可证

MIT License

---

## 技术支持

如有问题，请:

1. **查看文档**: 本文档中的常见问题部分
2. **查看示例**: `examples/quick_start.py`
3. **查看源码**: 每个模块都有详细注释
4. **查看论文**: 论文中有完整的方法论描述

---

## 致谢

- Weibo-COV 数据集: https://github.com/nghuyong/weibo-cov
- RoBERTa: https://github.com/ymcui/Chinese-BERT-wwm
- PyTorch: https://pytorch.org/
- HuggingFace Transformers: https://huggingface.co/

---

**最后更新**: 2026-02-11
