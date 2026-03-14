"""
src/train.py
训练脚本 — AutoDL · PyTorch 2.7.0 · Python 3.12 · CUDA 12.8

优化策略：类别重加权 + 标签平滑（针对中立类样本不足问题）

用法：
  python train.py --data data/asap_train.csv --epochs 15 --batch_size 32
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultiDimSentimentModel
from data_loader import load_asap_csv, load_asap_json, make_demo_data, get_dataloaders, oversample_neutral, DIMS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',           default=None)
    p.add_argument('--bert',           default='bert-base-chinese')
    p.add_argument('--epochs',         type=int,   default=15)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=2e-5)
    p.add_argument('--max_len',        type=int,   default=128)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--patience',       type=int,   default=4)
    p.add_argument('--save_dir',       default='checkpoints')
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--gpu',            type=int,   default=0)
    p.add_argument('--label_smoothing',type=float, default=0.1)
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if not torch.cuda.is_available():
        print('[WARNING] 没有检测到 CUDA，使用 CPU')
        return torch.device('cpu')
    n = torch.cuda.device_count()
    if gpu_id < 0:
        free = [torch.cuda.get_device_properties(i).total_memory -
                torch.cuda.memory_reserved(i) for i in range(n)]
        gpu_id = int(np.argmax(free))
    elif gpu_id >= n:
        gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    props = torch.cuda.get_device_properties(gpu_id)
    free_gb = (props.total_memory - torch.cuda.memory_reserved(gpu_id)) / 1e9
    print(f'[GPU] {props.name}  总显存={props.total_memory/1e9:.1f}GB  可用≈{free_gb:.1f}GB')
    return device


def compute_weights(records, dim_key, device):
    """
    计算某个维度的类别权重（反比于样本数）
    跳过 -1（忽略标签）
    """
    labels = [r[dim_key] for r in records if r[dim_key] != -1]
    if len(set(labels)) < 2:
        return None
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2]),
        y=np.array(labels)
    )
    print(f'  {dim_key} 类别权重: 负向={weights[0]:.2f} 中立={weights[1]:.2f} 正向={weights[2]:.2f}')
    return torch.tensor(weights, dtype=torch.float).to(device)


def make_loss_fns(records, device, label_smoothing=0.1):
    """为三个维度分别构建带权重的损失函数"""
    dim_map = {'味道': 'taste', '环境': 'env', '服务': 'service'}
    loss_fns = {}
    print('[INFO] 计算类别权重...')
    for dim_cn, dim_key in dim_map.items():
        w = compute_weights(records, dim_cn, device)
        loss_fns[dim_key] = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=w,
            label_smoothing=label_smoothing
        )
    return loss_fns


def train_epoch(model, loader, optimizer, scheduler, device, loss_fns):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='  训练', leave=False):
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        tids = batch['token_type_ids'].to(device)
        t_lb = batch['taste'].to(device)
        e_lb = batch['env'].to(device)
        s_lb = batch['service'].to(device)

        optimizer.zero_grad()
        out = model(ids, mask, tids)   # 不传标签，手动算损失

        loss = (loss_fns['taste'](out['taste_logits'],   t_lb) +
                loss_fns['env']  (out['env_logits'],     e_lb) +
                loss_fns['service'](out['service_logits'], s_lb))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    dim_keys   = ['taste', 'env', 'service']
    all_preds  = {k: [] for k in dim_keys}
    all_labels = {k: [] for k in dim_keys}

    with torch.no_grad():
        for batch in tqdm(loader, desc='  评估', leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tids = batch['token_type_ids'].to(device)
            out  = model(ids, mask, tids)
            for k in dim_keys:
                preds  = out[f'{k}_logits'].argmax(dim=-1).cpu().numpy()
                labels = batch[k].numpy()
                valid  = labels != -1
                all_preds[k].extend(preds[valid].tolist())
                all_labels[k].extend(labels[valid].tolist())

    metrics = {}
    for k, name in zip(dim_keys, DIMS):
        p = np.array(all_preds[k])
        t = np.array(all_labels[k])
        if len(t) == 0:
            metrics[name] = {'acc': 0.0, 'f1': 0.0, 'n': 0}
            continue
        metrics[name] = {
            'acc': float((p == t).mean()),
            'f1':  float(f1_score(t, p, average='macro', zero_division=0)),
            'n':   len(t)
        }
    total_n = sum(v['n'] for v in metrics.values())
    metrics['avg'] = {
        'acc': sum(v['acc'] * v['n'] for v in metrics.values()) / max(total_n, 1),
        'f1':  sum(v['f1']  * v['n'] for v in metrics.values()) / max(total_n, 1),
        'n':   total_n
    }
    return metrics


def fmt(metrics):
    parts = [f"{n} acc={metrics[n]['acc']:.3f} F1={metrics[n]['f1']:.3f}" for n in DIMS]
    parts.append(f"均值 acc={metrics['avg']['acc']:.3f} F1={metrics['avg']['f1']:.3f}")
    return '  '.join(parts)


def main():
    args = get_args()
    set_seed(args.seed)
    device = get_device(args.gpu)

    print(f"\n{'='*60}")
    print(f"  BERT: {args.bert}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  label_smoothing={args.label_smoothing}  patience={args.patience}")
    print(f"{'='*60}\n")

    # 加载数据
    if args.data and os.path.exists(args.data):
        if args.data.endswith('.csv'):
            records = load_asap_csv(args.data)
        else:
            records = load_asap_json(args.data)
    else:
        print('[INFO] 未指定数据，使用演示数据')
        records = make_demo_data('data/demo.json')

    # 对训练集中立类过采样
    records = oversample_neutral(records)

    tokenizer = BertTokenizer.from_pretrained(args.bert)
    train_loader, val_loader, test_loader = get_dataloaders(
        records, tokenizer, batch_size=args.batch_size, max_length=args.max_len)

    # 构建带权重的损失函数（用全部records统计权重）
    loss_fns = make_loss_fns(records, device, args.label_smoothing)

    model = MultiDimSentimentModel(bert_model_name=args.bert, dropout=args.dropout).to(device)

    bert_params = list(model.bert.parameters())
    head_params = [p for n, p in model.named_parameters() if 'bert' not in n]
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': args.lr,      'weight_decay': 0.01},
        {'params': head_params, 'lr': args.lr * 5,  'weight_decay': 0.0},
    ])
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1, no_improve, history = 0.0, 0, []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fns)
        val_m      = evaluate(model, val_loader, device)
        val_f1     = val_m['avg']['f1']

        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  {fmt(val_m)}  显存={used_gb:.2f}GB")
        else:
            print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  {fmt(val_m)}")

        history.append({'epoch': epoch, 'loss': train_loss,
                        'val_f1': val_f1, 'val_acc': val_m['avg']['acc']})

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            ckpt_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_metrics': val_m, 'args': vars(args)}, ckpt_path)
            print(f"  ✓ 保存最优模型 avg_F1={best_f1:.4f}")
        else:
            no_improve += 1
            print(f"  patience {no_improve}/{args.patience}")
            if no_improve >= args.patience:
                print(f"\n[早停] 连续{args.patience}轮未提升，停止训练")
                break

    # 测试集评估
    print('\n[INFO] 加载最优模型，测试集评估...')
    ckpt = torch.load(os.path.join(args.save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_m = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    for name in DIMS:
        m = test_m[name]
        print(f"  {name}:  准确率={m['acc']:.4f}  macro-F1={m['f1']:.4f}  (样本{m['n']}条)")
    m = test_m['avg']
    print(f"  均值:  准确率={m['acc']:.4f}  macro-F1={m['f1']:.4f}")
    print(f"{'='*50}\n")

    with open(os.path.join(args.save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'history': history, 'test': test_m}, f, ensure_ascii=False, indent=2)
    print('[INFO] 完成！结果保存至 checkpoints/results.json')


if __name__ == '__main__':
    main()