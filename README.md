# SV-NSDE: Semantic Volatility-Modulated Neural SDE

> 一个用于危机动态建模的神经点过程模型，能够区分恐慌驱动的爆发与趋势驱动的热点

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 概览

本项目实现了论文《Modeling Crisis Dynamics with Volatility-Modulated Neural SDEs: Distinguishing Panic from Trends in COVID-19 Information Diffusion》中的 **SV-NSDE** 模型。

### 核心特性

- **双通道强度函数**: 显式区分趋势驱动 vs 波动率驱动的事件
- **耦合随机微分方程**: z(t) 和 v(t) 的Heston风格建模
- **语义感知**: 利用RoBERTa编码器处理中文微博文本
- **6个基线模型**: RMTPP, Neural Hawkes, Latent ODE, Neural Jump SDE + 消融变体
- **完整评估框架**: 时间预测、语义预测、似然度量

### 应用场景

- 🔍 识别社交媒体上的恐慌驱动事件
- 📊 建模信息扩散的时间动态
- 🎯 危机预警和舆情监测
- 📈 COVID-19期间的信息传播分析

---

## 🚀 快速开始

### 1. 安装 (2分钟)

```bash
# 克隆项目
git clone <项目地址>
cd Research

# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装依赖
uv pip install -e ".[dev,viz]"

# 验证
python -c "from sv_nsde import SVNSDELite; print('✓ Ready!')"
```

### 2. 运行示例 (1分钟)

```bash
cd examples && python quick_start.py
```

### 3. 评估基线 (5分钟)

```bash
python scripts/run_evaluation.py --quick
```

---

## 📚 文档

| 文档 | 内容 |
|------|------|
| **[QUICKSTART.md](QUICKSTART.md)** | 5分钟快速入门 + 常用命令 |
| **[USAGE_GUIDE.md](USAGE_GUIDE.md)** | 完整中文使用文档 (596行) |
| **examples/** | 5个完整代码示例 |

---

## 📥 数据

### 使用合成数据 (推荐：快速测试)

项目内置数据生成器，无需下载：

```python
from sv_nsde import generate_synthetic_weibo_data

df = generate_synthetic_weibo_data(
    n_cascades=1000,
    output_path="data/weibo_synthetic.csv"
)
```

### 使用真实 Weibo-COV 数据

Weibo-COV 是一个包含6500万条COVID-19相关微博的真实数据集：

**论文**: https://arxiv.org/abs/2005.09174 | https://github.com/nghuyong/weibo-cov

**下载链接** (需百度账号):
- [Weibo-COV 2.0](https://pan.baidu.com/s/1mxU5RbnGBNRvR4Ci-9d0Hg?pwd=jffm) (推荐, 12GB)
- [Weibo-COV 1.0](https://pan.baidu.com/s/1SwbkEnuXrUFmRj1lx_AQlg?pwd=r8gn) (7GB)

**使用真实数据**:

```python
from sv_nsde import WeiboCOVLoader

# 下载并解压到 data/ 目录后
loader = WeiboCOVLoader("data/weibo_cov.csv")
loader.load()
cascades = loader.build_cascades(min_size=10, max_size=500)
train, val, test = loader.split_by_time()
```

---

## 💻 使用示例

### 基础用法

```python
from sv_nsde import SVNSDELite

# 创建模型
model = SVNSDELite(d_input=768, d_latent=32, d_hidden=64)

# 前向传播
outputs = model(event_times, event_embeddings, T=1.0)
# outputs['z_events']: 事件时的语义状态
# outputs['z_trajectory']: 完整轨迹
```

### 训练模型

```python
from sv_nsde import SVNSDELite, Trainer, CascadeDataset

model = SVNSDELite()
dataset = CascadeDataset(cascades)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=100,
    device="cuda"
)

trainer.train()
```

### 评估与对比

```python
from sv_nsde import Evaluator, get_baseline

models = {
    "SV-NSDE": SVNSDELite(),
    "RMTPP": get_baseline("rmtpp"),
    "Neural Hawkes": get_baseline("neural_hawkes"),
}

evaluator = Evaluator(device="cuda")
results = evaluator.evaluate_all(models, test_cascades)
```

### 波动率分析 (核心功能)

```python
from sv_nsde import VolatilityAnalyzer

analyzer = VolatilityAnalyzer(model, device="cuda")

# 分析单个级联
analysis = analyzer.analyze_cascade(cascade, embeddings)
print(f"恐慌事件: {analysis['num_panic_events']}")

# 找出恐慌爆发
bursts = analyzer.find_burst_events(
    cascades,
    embeddings_dict,
    volatility_threshold=0.6
)
```

---

## 🏗️ 项目结构

```
Research/
├── QUICKSTART.md              # 快速入门 (推荐先看)
├── USAGE_GUIDE.md             # 详细文档
├── examples/
│   └── quick_start.py         # 5个完整示例
├── scripts/
│   └── run_evaluation.py      # 完整评估脚本
├── src/sv_nsde/
│   ├── model.py               # SV-NSDE主模型
│   ├── baselines.py           # 6个基线模型
│   ├── evaluate.py            # 评估指标
│   ├── encoder.py             # RoBERTa编码器
│   ├── sde.py                 # Neural Heston SDE
│   ├── intensity.py           # 双通道强度函数
│   ├── decoder.py             # 语义解码器
│   ├── train.py               # 训练脚本
│   └── data.py                # 数据加载
└── data/
    └── synthetic_weibo_cov.csv # 合成测试数据
```

---

## 📊 模型对比

| 模型 | 类型 | 参数 | 特点 |
|------|------|------|------|
| **SV-NSDE** | Neural SDE + Heston | 44K | 双通道强度 (本文) |
| RMTPP | RNN | 59K | 指数强度衰减 |
| Neural Hawkes | RNN | 50K | 连续时间LSTM |
| Latent ODE | ODE | 22K | 确定性演化 |
| Neural Jump SDE | SDE | 22K | 有跳跃无波动率分解 |
| SV-NSDE (no vol) | SDE + 消融 | 47K | 去掉波动率通道 |
| SV-NSDE (det vol) | SDE + 消融 | 53K | 确定性波动率 |

---

## 🔬 评估指标

根据论文Section 4.3:

- **时间预测**: RMSE (Root Mean Square Error)
- **语义预测**: Cosine Similarity / MSE
- **模型拟合**: Log-Likelihood
- **波动率分析**: Panic ratio / volatility decomposition

运行完整评估:

```bash
python scripts/run_evaluation.py --data data/weibo_cov.csv --device cuda
```

---

## 🎯 主要创新

1. **显式波动率建模**: v(t) 不仅是噪声，而是主动预测信号
2. **双通道机制**: λ(t) = Softplus(trend + volatility)
3. **语义感知**: RoBERTa编码 + VAE风格的重构
4. **危机阶段分析**: 对爆发、平台、衰退期的区分建模

---

## 📖 参考文献

```bibtex
@article{chen2026svnsde,
  title={Modeling Crisis Dynamics with Volatility-Modulated Neural SDEs:
         Distinguishing Panic from Trends in COVID-19 Information Diffusion},
  author={Chen, Zirui},
  year={2026}
}

@inproceedings{hu2020weibo,
  title={Weibo-COV: A Large-Scale COVID-19 Social Media Dataset from Weibo},
  author={Hu, Yong and Huang, Heyan and Chen, Anfan and Mao, Xian-Ling},
  booktitle={Proceedings of NLP4COVID@EMNLP 2020},
  year={2020}
}
```

---

## ⚙️ 环境要求

- Python 3.10+
- PyTorch 2.0+
- 8GB+ RAM (推荐16GB+)
- 可选: CUDA 11.8+ (GPU加速)

---

## 📞 支持

- **快速问题**: 查看 [QUICKSTART.md](QUICKSTART.md)
- **详细问题**: 查看 [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **代码示例**: 查看 `examples/`
- **源代码注释**: 每个模块都有详细文档

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

---

**推荐阅读顺序**: [QUICKSTART.md](QUICKSTART.md) → `examples/quick_start.py` → [USAGE_GUIDE.md](USAGE_GUIDE.md)
