"""
src/predict.py
推理模块

用法：
  python src/predict.py --text "菜很好吃，环境一般，服务还不错"
"""

import os, sys
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from model import MultiDimSentimentModel
from data_loader import clean_text, DIMS, LABELS

LABEL_CN   = ['负向', '中立', '正向']
LABEL_EMOJI= ['😞', '😐', '😊']
DIM_KEYS   = ['taste', 'env', 'service']


class SentimentPredictor:
    """
    餐饮评论多维度情感预测器
    输出三个维度的情感极性和置信度
    """

    def __init__(self, checkpoint_path: str = None,
                 bert_model_name: str = 'bert-base-chinese',
                 max_length: int = 128):
        self.max_length = max_length
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer  = BertTokenizer.from_pretrained(bert_model_name)
        self.model      = MultiDimSentimentModel(bert_model_name=bert_model_name)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            print(f'[Predictor] 已加载模型: {checkpoint_path}')
        else:
            print('[Predictor] 未找到checkpoint，使用初始权重（仅测试）')

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts) -> list:
        if isinstance(texts, str):
            texts = [texts]
        texts = [clean_text(t) for t in texts]

        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ids  = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)
        tids = enc.get('token_type_ids',
                       torch.zeros_like(ids)).to(self.device)

        with torch.no_grad():
            out = self.model(ids, mask, tids)

        results = []
        for i, text in enumerate(texts):
            item = {'text': text, 'dimensions': {}}
            for dim_cn, dim_key in zip(DIMS, DIM_KEYS):
                logits = out[f'{dim_key}_logits'][i]     # (3,)
                probs  = F.softmax(logits, dim=-1)
                pred   = int(probs.argmax())
                conf   = float(probs.max())
                item['dimensions'][dim_key] = {
                    'name':       dim_cn,
                    'sentiment':  LABEL_CN[pred],
                    'emoji':      LABEL_EMOJI[pred],
                    'label':      pred,           # 0/1/2
                    'confidence': round(conf, 4),
                    'probs': {
                        '负向': round(float(probs[0]), 4),
                        '中立': round(float(probs[1]), 4),
                        '正向': round(float(probs[2]), 4),
                    }
                }
            results.append(item)
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--text',  required=True)
    p.add_argument('--model', default='checkpoints/best_model.pt')
    p.add_argument('--bert',  default='bert-base-chinese')
    args = p.parse_args()

    predictor = SentimentPredictor(args.model, args.bert)
    result    = predictor.predict_one(args.text)

    print(f"\n评论：{result['text']}\n")
    for dim_key in DIM_KEYS:
        d = result['dimensions'][dim_key]
        bar = '█' * int(d['confidence'] * 20)
        print(f"  {d['name']:>4}：{d['emoji']} {d['sentiment']:>2}  "
              f"置信度 {d['confidence']*100:.1f}%  [{bar:<20}]")
        print(f"        负向={d['probs']['负向']:.3f}  "
              f"中立={d['probs']['中立']:.3f}  "
              f"正向={d['probs']['正向']:.3f}")
