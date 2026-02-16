# 快速入门 (5分钟)

## 🚀 一键启动

### 1️⃣ 准备环境

```bash
# 克隆项目
git clone <项目地址>
cd Research

# 创建虚拟环境
uv venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv pip install -e ".[dev,viz]"
```

### 2️⃣ 验证安装

```bash
python -c "from sv_nsde import SVNSDELite; print('✓ Ready to use!')"
```

### 3️⃣ 运行示例

```bash
cd examples
python quick_start.py
```

---

## 📊 5个常用命令

### 生成测试数据
```bash
python -c "
from sv_nsde import generate_synthetic_weibo_data
generate_synthetic_weibo_data(n_cascades=500, output_path='data/test.csv')
"
```

### 训练模型
```bash
python -c "
from sv_nsde import SVNSDELite, Trainer, CascadeDataset
from sv_nsde.data import WeiboCOVLoader

# 加载数据
loader = WeiboCOVLoader('data/test.csv')
loader.load()
cascades = loader.build_cascades(min_size=5, max_size=100)
train = cascades[:50]

# 创建数据集
dataset = CascadeDataset(train)

# 模型和训练
model = SVNSDELite(d_input=32, d_latent=32)
trainer = Trainer(model, dataset, learning_rate=1e-3, batch_size=8, num_epochs=5)
trainer.train()
"
```

### 评估基线模型
```bash
python scripts/run_evaluation.py --quick
```

### 分析波动率
```python
from sv_nsde import SVNSDELite, VolatilityAnalyzer

model = SVNSDELite()
analyzer = VolatilityAnalyzer(model)
analysis = analyzer.analyze_cascade(cascade, embeddings)
print(f"恐慌事件占比: {analysis['panic_ratio']:.2%}")
```

### 加载真实数据
```python
from sv_nsde import WeiboCOVLoader

# 下载后放在 data/ 目录
loader = WeiboCOVLoader("data/weibo_cov.csv")
loader.load()
cascades = loader.build_cascades(min_size=10, max_size=500)
train, val, test = loader.split_by_time()
```

---

## 📥 下载真实数据

### 步骤 1: 获取数据

数据托管于百度网盘（需要百度账号）：

- **版本 2.0** (推荐): https://pan.baidu.com/s/1mxU5RbnGBNRvR4Ci-9d0Hg?pwd=jffm
- **版本 1.0**: https://pan.baidu.com/s/1SwbkEnuXrUFmRj1lx_AQlg?pwd=r8gn

### 步骤 2: 解压文件

```bash
# 下载后解压
unzip weibo-cov.zip -d data/

# 查看数据
ls -lh data/weibo_cov*.csv
```

### 步骤 3: 使用数据

```python
from sv_nsde import WeiboCOVLoader

loader = WeiboCOVLoader("data/weibo_cov.csv")
loader.load()  # 可选: load(nrows=1000000) 限制行数

cascades = loader.build_cascades(min_size=10, max_size=500)
print(f"共 {len(cascades)} 个级联")

# 按时间分割
train, val, test = loader.split_by_time(
    train_end="2020-02-29",  # 爆发期
    val_end="2020-03-31"     # 平台期
)
```

---

## 📚 数据格式

**Weibo-COV 数据**包含以下列：

```
_id (推文ID)
user_id (用户ID)
created_at (发布时间: YYYY-MM-DD HH:MM:SS)
content (推文内容)
like_num (点赞)
repost_num (转发)
comment_num (评论)
origin_weibo (原推ID)
geo_info (地理位置)
```

**数据规模**:
- Weibo-COV 1.0: 4089万条推文 (~7GB)
- Weibo-COV 2.0: 6518万条推文 (~12GB)

---

## ⚡ 性能建议

| 设备 | 推荐配置 |
|------|---------|
| CPU | batch_size=4, 小数据 (<100K) |
| GPU (8GB) | batch_size=16, d_latent=32 |
| GPU (24GB) | batch_size=32, d_latent=64 |
| 多GPU | 使用 DataParallel |

---

## 🔧 常见问题

**Q: 导入失败？**
```bash
# 重新安装
pip install --upgrade -e .
```

**Q: CUDA 相关错误？**
```bash
# 使用 CPU
python scripts/run_evaluation.py --device cpu
```

**Q: 内存不足？**
```python
# 减少 batch 和 max_events
trainer = Trainer(..., batch_size=4)
dataset = CascadeDataset(..., max_events=50)
```

**Q: 如何使用 GPU？**
```python
import torch
print(torch.cuda.is_available())  # 检查GPU
model = SVNSDELite().to("cuda")
```

---

## 📖 详细文档

完整使用说明请查看 `USAGE_GUIDE.md`

主要章节：
- 详细环境设置
- 端到端工作流程
- 所有模型和基线
- 评估指标解释
- 论文引用

---

## 文件结构

```
.
├── USAGE_GUIDE.md          # 详细文档 (本文件)
├── examples/
│   └── quick_start.py      # 5个快速示例
├── scripts/
│   └── run_evaluation.py   # 完整评估脚本
├── data/
│   └── synthetic_weibo_cov.csv  # 合成测试数据
└── src/sv_nsde/
    ├── model.py            # 主模型
    ├── baselines.py        # 6个基线
    └── evaluate.py         # 评估指标
```

---

## 核心命令速查

```bash
# 验证安装
python -c "from sv_nsde import *; print('OK')"

# 查看所有模型
python -c "from sv_nsde.baselines import BASELINE_MODELS; print(BASELINE_MODELS.keys())"

# 快速评估
python scripts/run_evaluation.py --quick

# 完整评估 (GPU)
python scripts/run_evaluation.py --device cuda --data data/weibo_cov.csv

# 生成数据
python -c "from sv_nsde import generate_synthetic_weibo_data; generate_synthetic_weibo_data()"

# 查看项目结构
find . -name "*.py" -path "*/sv_nsde/*" | head -20
```

---

**需要帮助？** 查看 `USAGE_GUIDE.md` 的完整文档！
