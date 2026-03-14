"""
src/data_loader.py
美团ASAP数据集加载与预处理

真实CSV格式（asap/data/train.csv）：
  列名：id, review, star, Food#Taste, Service#Hospitality, ...
  标签：-2=未提及, -1=中立, 0=负向, 1=正向

本项目标签映射（统一为模型用的0/1/2/-1）：
  -2 → -1（忽略，不参与损失计算）
  -1 → 1 （中立）
  0  → 0 （负向）
  1  → 2 （正向）

三个维度的列合并规则：
  口味 ← Food#Taste（主）
  环境 ← Ambience#Decoration / Noise / Space / Sanitary 取众数（忽略-2）
  服务 ← Service#Queue / Hospitality / Parking / Timely 取众数（忽略-2）
"""

import json, re, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from collections import Counter

# ── 常量 ──
DIMS       = ['味道', '环境', '服务']
LABELS     = ['负向', '中立', '正向']   # 0 / 1 / 2
NUM_LABELS = 3
IGNORE_IDX = -1   # 传给 CrossEntropyLoss(ignore_index=)

# ASAP CSV 里各维度对应的列
TASTE_COLS   = ['Food#Taste']
ENV_COLS     = ['Ambience#Decoration', 'Ambience#Noise',
                'Ambience#Space', 'Ambience#Sanitary']
SERVICE_COLS = ['Service#Queue', 'Service#Hospitality',
                'Service#Parking', 'Service#Timely']


def asap_label_to_model(raw) -> int:
    """
    把 ASAP 原始标签转成模型用的标签
      -2 → -1 (忽略)
      -1 → 1  (中立)
       0 → 0  (负向)
       1 → 2  (正向)
    """
    raw = int(raw)
    if raw == -2:
        return -1   # 未提及，忽略
    if raw == -1:
        return 1    # 中立
    if raw == 0:
        return 0    # 负向
    if raw == 1:
        return 2    # 正向
    return -1       # 其他异常值也忽略


def merge_dim(row, cols) -> int:
    """
    多列合并为一个维度标签。
    先把各列原始值转换，过滤掉 -1（未提及），
    然后取出现次数最多的标签；
    如果全部是未提及，返回 -1（忽略）。
    """
    vals = []
    for c in cols:
        if c in row.index:
            v = asap_label_to_model(row[c])
            if v != -1:
                vals.append(v)
    if not vals:
        return -1
    return Counter(vals).most_common(1)[0][0]


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ══════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════

class ASAPDataset(Dataset):
    def __init__(self, records: list, tokenizer: BertTokenizer,
                 max_length: int = 128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.texts   = [clean_text(r['content']) for r in records]
        self.taste   = [r['味道']  for r in records]
        self.env     = [r['环境']  for r in records]
        self.service = [r['服务']  for r in records]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'token_type_ids': enc.get(
                'token_type_ids',
                torch.zeros(self.max_length, dtype=torch.long)
            ).flatten(),
            'taste':   torch.tensor(self.taste[idx],   dtype=torch.long),
            'env':     torch.tensor(self.env[idx],     dtype=torch.long),
            'service': torch.tensor(self.service[idx], dtype=torch.long),
            'text':    self.texts[idx],
        }


# ══════════════════════════════════════════
# 加载 ASAP CSV（真实数据集格式）
# ══════════════════════════════════════════

def load_asap_csv(path: str) -> list:
    """
    加载 ASAP 官方 CSV 文件（train.csv / dev.csv / test.csv）
    自动处理列合并和标签映射
    """
    df = pd.read_csv(path, encoding='utf-8')
    print(f"[DataLoader] 读取 {path}，共 {len(df)} 行")

    records = []
    skipped = 0
    for _, row in df.iterrows():
        content = clean_text(str(row.get('review', row.get('content', ''))))
        if len(content) < 4:
            skipped += 1
            continue

        taste   = asap_label_to_model(row['Food#Taste']) \
                  if 'Food#Taste' in row.index else -1
        env     = merge_dim(row, ENV_COLS)
        service = merge_dim(row, SERVICE_COLS)

        records.append({
            'content': content,
            '味道':   taste,
            '环境':   env,
            '服务':   service,
        })

    print(f"[DataLoader] 有效 {len(records)} 条（跳过 {skipped} 条过短）")
    _print_stats(records)
    return records


def _print_stats(records):
    for dim in DIMS:
        counts = {-1: 0, 0: 0, 1: 0, 2: 0}
        for r in records:
            counts[r[dim]] = counts.get(r[dim], 0) + 1
        valid = len(records) - counts[-1]
        print(f"  {dim}: 忽略={counts[-1]}  "
              f"负向={counts[0]}  中立={counts[1]}  正向={counts[2]}  "
              f"(有效{valid}条, {valid/len(records)*100:.1f}%)")


# ══════════════════════════════════════════
# 兼容旧 JSON 格式（演示数据用）
# ══════════════════════════════════════════

def load_asap_json(path: str) -> list:
    """加载 JSON 格式（演示数据 / 自定义数据）"""
    with open(path, encoding='utf-8') as f:
        try:
            raw = json.load(f)
            if not isinstance(raw, list):
                raw = [raw]
        except json.JSONDecodeError:
            f.seek(0)
            raw = [json.loads(l) for l in f if l.strip()]

    records, skipped = [], 0
    for item in raw:
        content = str(item.get('content', item.get('review', ''))).strip()
        if len(content) < 4:
            skipped += 1
            continue
        records.append({
            'content': content,
            '味道':   int(item.get('味道', -1)),
            '环境':   int(item.get('环境', -1)),
            '服务':   int(item.get('服务', -1)),
        })
    print(f"[DataLoader] JSON加载 {len(records)} 条（跳过 {skipped} 条）")
    return records


# ══════════════════════════════════════════
# DataLoader 构建
# ══════════════════════════════════════════

def get_dataloaders(records: list, tokenizer: BertTokenizer,
                    batch_size: int = 64, max_length: int = 128,
                    val_ratio: float = 0.1, test_ratio: float = 0.1,
                    seed: int = 42):
    """8:1:1 划分，返回 train / val / test DataLoader"""
    train_r, temp = train_test_split(
        records, test_size=val_ratio + test_ratio, random_state=seed)
    val_r, test_r = train_test_split(
        temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed)
    print(f"[DataLoader] 训练:{len(train_r)}  验证:{len(val_r)}  测试:{len(test_r)}")

    def make(recs, shuffle):
        ds = ASAPDataset(recs, tokenizer, max_length)
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, num_workers=4,
                          pin_memory=True)   # AutoDL GPU训练加速

    return make(train_r, True), make(val_r, False), make(test_r, False)


# ══════════════════════════════════════════
# 演示数据（20条）
# ══════════════════════════════════════════

def make_demo_data(save_path: str = 'data/demo.json') -> list:
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    # 标签已经是映射后的值：-1=忽略, 0=负向, 1=中立, 2=正向
    samples = [
        ("菜品味道非常好，食材新鲜，摆盘精美",            2, -1, -1),
        ("环境很差，嘈杂拥挤，根本无法好好用餐",          -1,  0, -1),
        ("服务态度极好，服务员热情周到，上菜速度快",       -1, -1,  2),
        ("口味一般，环境还行，服务还不错",                  1,  1,  1),
        ("菜很难吃，又贵又少，完全不值这个价",              0, -1, -1),
        ("装修高档，灯光柔和，坐着很舒服",                -1,  2, -1),
        ("服务员态度恶劣，等了很久才来人",                -1, -1,  0),
        ("味道正宗，是记忆中的味道，下次还来",              2, -1, -1),
        ("环境一般，桌子有点脏，但菜的口味不错",            2,  0, -1),
        ("整体体验很好，味道、环境、服务都很满意",           2,  2,  2),
        ("菜品太咸了，口味重，不太适合我",                  0, -1, -1),
        ("店面装修很有格调，适合约会",                    -1,  2, -1),
        ("服务一般，叫服务员好几次才来",                  -1, -1,  1),
        ("食材新鲜，做法创新，味道独特",                    2, -1, -1),
        ("环境嘈杂，隔壁桌声音很大，影响用餐",            -1,  0, -1),
        ("服务员很专业，菜品介绍得很详细",                -1, -1,  2),
        ("口味中规中矩，没什么特色，价格偏高",              1, -1, -1),
        ("包间环境不错，但菜的味道只能说一般",              1,  2, -1),
        ("上菜太慢，等了40分钟，服务员还态度不好",        -1, -1,  0),
        ("性价比很高，味道好价格实惠，推荐",                2, -1,  2),
    ]
    records = [{'content': s[0], '味道': s[1], '环境': s[2], '服务': s[3]}
               for s in samples]
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[DataLoader] 演示数据已保存 → {save_path}")
    return records


if __name__ == '__main__':
    # 快速验证：直接读 ASAP CSV
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'asap/data/train.csv'
    records = load_asap_csv(path)
    print(f"\n总计 {len(records)} 条可用")


# ══════════════════════════════════════════
# 中立类过采样（数据增强）
# ══════════════════════════════════════════

def oversample_neutral(records: list, target_ratio: float = 0.25) -> list:
    """
    对中立类样本做过采样，让中立类占有效样本的比例接近 target_ratio。
    target_ratio=0.25 表示中立类占有效样本约25%。

    只复制整条记录（某维度为中立的记录），不做文本修改。
    """
    import random
    random.seed(42)

    result = list(records)

    for dim in ['味道', '环境', '服务']:
        # 统计当前该维度各类数量
        neutral = [r for r in records if r[dim] == 1]   # 1=中立
        valid   = [r for r in records if r[dim] != -1]  # 有效样本

        if not neutral or not valid:
            continue

        current_ratio = len(neutral) / len(valid)
        if current_ratio >= target_ratio:
            continue  # 已经够了，不需要过采样

        # 需要补多少条
        target_n = int(len(valid) * target_ratio / (1 - target_ratio))
        need     = target_n - len(neutral)
        if need <= 0:
            continue

        # 有放回地随机采样
        extra = random.choices(neutral, k=need)
        result.extend(extra)
        print(f"  [{dim}] 中立类: {len(neutral)} → {len(neutral)+need} 条 "
              f"(复制了{need}条)")

    random.shuffle(result)
    print(f"[过采样] 总数据量: {len(records)} → {len(result)} 条")
    return result