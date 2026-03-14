"""
src/model.py
基于BERT的多维度情感分析模型（三分类版）

输入：餐饮评论文本
输出：口味 / 环境 / 服务 各自的情感极性
      0=负向  1=中立  2=正向  （未提及维度不参与损失计算）

核心改进（对比旧版）：
  - 标签从1-5星改为 负/中/正 三分类，更贴合ASAP数据
  - 损失计算用 ignore_index=-1，未提及的维度自动跳过
  - 每个维度独立注意力池化，让模型聚焦不同关键词
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class MultiDimSentimentModel(nn.Module):
    """
    三维度三分类情感分析模型

    架构：
      BERT编码器（共享）
        ↓
      维度注意力池化 × 3（各维度独立关注相关词）
        ↓
      Dropout
        ↓
      分类头 × 3  →  各输出3维logits（负/中/正）
    """

    def __init__(self, bert_model_name: str = 'bert-base-chinese',
                 num_labels: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.bert      = BertModel.from_pretrained(bert_model_name)
        hidden         = self.bert.config.hidden_size   # 768
        self.dropout   = nn.Dropout(dropout)
        self.num_labels = num_labels

        # 每个维度独立的注意力权重向量（用于从全部token中加权求和）
        self.taste_attn   = nn.Linear(hidden, 1)
        self.env_attn     = nn.Linear(hidden, 1)
        self.service_attn = nn.Linear(hidden, 1)

        # 三个独立分类头
        self.taste_cls   = nn.Linear(hidden, num_labels)
        self.env_cls     = nn.Linear(hidden, num_labels)
        self.service_cls = nn.Linear(hidden, num_labels)

    def _attn_pool(self, hidden_states: torch.Tensor,
                   attention_mask: torch.Tensor,
                   attn_layer: nn.Linear) -> torch.Tensor:
        """
        注意力池化：为每个维度从全部token中加权求和
        hidden_states: (batch, seq_len, 768)
        返回: (batch, 768)
        """
        scores  = attn_layer(hidden_states).squeeze(-1)      # (batch, seq_len)
        scores  = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (hidden_states * weights).sum(dim=1)            # (batch, 768)

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                taste_labels:   torch.Tensor = None,
                env_labels:     torch.Tensor = None,
                service_labels: torch.Tensor = None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state   # (batch, seq_len, 768)

        # 各维度独立注意力池化
        taste_vec   = self.dropout(self._attn_pool(hidden, attention_mask, self.taste_attn))
        env_vec     = self.dropout(self._attn_pool(hidden, attention_mask, self.env_attn))
        service_vec = self.dropout(self._attn_pool(hidden, attention_mask, self.service_attn))

        taste_logits   = self.taste_cls(taste_vec)      # (batch, 3)
        env_logits     = self.env_cls(env_vec)          # (batch, 3)
        service_logits = self.service_cls(service_vec)  # (batch, 3)

        # 计算损失（ignore_index=-1 自动跳过"未提及"）
        loss = None
        if taste_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = (loss_fn(taste_logits,   taste_labels) +
                    loss_fn(env_logits,     env_labels)   +
                    loss_fn(service_logits, service_labels))

        return {
            'loss':           loss,
            'taste_logits':   taste_logits,
            'env_logits':     env_logits,
            'service_logits': service_logits,
        }
