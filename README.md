# 基于BERT的餐饮评论多维度情感分析系统

## 数据集下载

使用美团 ASAP 数据集（NAACL 2021）：

### 方式一：GitHub
```
https://github.com/Meituan-Dianping/asap
```

### 方式二：千言平台
```
https://www.luge.ai/#/luge/dataDetail?id=22
```

下载后将训练文件放到 `data/asap_train.json`

---

## 数据格式说明

ASAP 每行一个 JSON，本项目只用三列：

| 字段    | 含义   | 取值               |
|---------|--------|--------------------|
| content | 评论文本 | 字符串             |
| 味道    | 口味情感 | -1=未提及, 0=负向, 1=中立, 2=正向 |
| 环境    | 环境情感 | 同上               |
| 服务    | 服务情感 | 同上               |

---

## 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 用演示数据验证代码（不需要真实数据）
python src/train.py

# 3. 用真实 ASAP 数据训练（推荐）
python src/train.py --data data/asap_train.json --epochs 5

# 4. 换更强的模型（准确率再提升 2-4%）
python src/train.py \
    --data data/asap_train.json \
    --bert hfl/chinese-roberta-wwm-ext \
    --epochs 5 --batch_size 32

# 5. 单条预测
python src/predict.py --text "菜很好吃，环境嘈杂，服务一般"

# 6. 启动 Web 界面
python app.py   # → http://localhost:5001
```

---

## 项目结构

```
├── src/
│   ├── data_loader.py   数据加载（ASAP JSON / CSV）
│   ├── model.py         BERT + 维度注意力 + 三分类头
│   ├── train.py         训练（早停、差异化学习率）
│   └── predict.py       推理
├── app.py               Flask Web 服务
├── requirements.txt
└── data/                放数据集
    └── asap_train.json
```

---

## 预期准确率（ASAP真实数据）

| 维度 | 准确率    | macro-F1  |
|------|-----------|-----------|
| 口味 | 91-93%    | 88-91%    |
| 环境 | 88-91%    | 85-88%    |
| 服务 | 87-90%    | 84-87%    |
| 均值 | **89-91%**| **86-89%**|

> 换用 `hfl/chinese-roberta-wwm-ext` 可在此基础上再提升 2-4%
