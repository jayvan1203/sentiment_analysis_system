"""
models.py  ——  SQLAlchemy ORM 数据库模型定义
数据库：SQLite（文件 sentiment.db，自动创建）

表结构：
  User        用户表
  Merchant    餐厅表
  Tag         标签表（多对多中间表 merchant_tags）
  Review      评论表（含情感分析结果 JSON）
  SentAgg     情感聚合缓存表（每个餐厅一行，实时维护）
  AnalysisLog 分析日志表（对应旧的 history_records）
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# ── 餐厅-标签 多对多中间表 ──────────────────────────────────────
merchant_tags = db.Table(
    'merchant_tags',
    db.Column('merchant_id', db.Integer, db.ForeignKey('merchant.id', ondelete='CASCADE'), primary_key=True),
    db.Column('tag_id',      db.Integer, db.ForeignKey('tag.id',      ondelete='CASCADE'), primary_key=True),
)


class User(db.Model):
    """用户表"""
    __tablename__ = 'user'

    id         = db.Column(db.Integer,     primary_key=True)
    username   = db.Column(db.String(64),  unique=True, nullable=False)
    password   = db.Column(db.String(128), nullable=False)
    name       = db.Column(db.String(64),  nullable=False)
    avatar     = db.Column(db.String(16),  default='👤')
    role       = db.Column(db.String(16),  default='user')   # 'admin' | 'user'
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {'id': self.id, 'username': self.username,
                'name': self.name, 'avatar': self.avatar, 'role': self.role}


class Tag(db.Model):
    """标签表（去重复用）"""
    __tablename__ = 'tag'

    id   = db.Column(db.Integer,    primary_key=True)
    name = db.Column(db.String(32), unique=True, nullable=False)


class Merchant(db.Model):
    """餐厅表"""
    __tablename__ = 'merchant'

    id         = db.Column(db.Integer,     primary_key=True)
    name       = db.Column(db.String(128), nullable=False)
    category   = db.Column(db.String(64),  default='其他')
    rating     = db.Column(db.Float,       default=4.5)
    price      = db.Column(db.Integer,     default=88)
    cover      = db.Column(db.String(16),  default='🍽️')
    address    = db.Column(db.String(256), default='')
    open_hours = db.Column(db.String(64),  default='')
    phone      = db.Column(db.String(32),  default='')
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

    tags    = db.relationship('Tag', secondary=merchant_tags, lazy='subquery',
                              backref=db.backref('merchants', lazy=True))
    reviews = db.relationship('Review', backref='merchant', lazy='dynamic',
                              cascade='all, delete-orphan')
    agg     = db.relationship('SentAgg', backref='merchant', uselist=False,
                              cascade='all, delete-orphan')

    def tag_names(self):
        return [t.name for t in self.tags]

    def to_dict(self, with_reviews=False, with_agg=False):
        d = {
            'id':         self.id,
            'name':       self.name,
            'category':   self.category,
            'rating':     self.rating,
            'price':      self.price,
            'cover':      self.cover,
            'address':    self.address,
            'open_hours': self.open_hours,
            'phone':      self.phone,
            'tags':       self.tag_names(),
        }
        if with_reviews:
            d['reviews'] = [r.to_dict() for r in
                            self.reviews.order_by(Review.created_at.desc()).all()]
        if with_agg and self.agg:
            d['agg'] = self.agg.to_summary()
        return d


class Review(db.Model):
    """评论表"""
    __tablename__ = 'review'

    id          = db.Column(db.Integer,     primary_key=True)
    merchant_id = db.Column(db.Integer,     db.ForeignKey('merchant.id', ondelete='CASCADE'), nullable=False)
    user_name   = db.Column(db.String(64),  nullable=False)          # 冗余存储，方便展示
    user_avatar = db.Column(db.String(16),  default='👤')
    stars       = db.Column(db.Integer,     default=3)
    text        = db.Column(db.Text,        nullable=False)
    status      = db.Column(db.String(16),  default='approved')      # approved | pending | rejected
    # 情感分析结果（JSON 字符串，None 表示未分析）
    analysis    = db.Column(db.Text,        nullable=True)
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)

    def get_analysis(self):
        """返回解析后的 analysis dict，或 None"""
        if self.analysis:
            try:
                return __import__('json').loads(self.analysis)
            except Exception:
                return None
        return None

    def set_analysis(self, data: dict):
        self.analysis = __import__('json').dumps(data, ensure_ascii=False)

    def to_dict(self):
        return {
            'id':      self.id,
            'user':    self.user_name,
            'avatar':  self.user_avatar,
            'stars':   self.stars,
            'date':    self.created_at.strftime('%Y-%m-%d'),
            'text':    self.text,
            'status':  self.status,
            '_analysis': self.get_analysis(),
        }


class SentAgg(db.Model):
    """
    情感聚合缓存表——每家餐厅一行，实时维护。
    各维度存为 neg/neu/pos 三个计数列。
    """
    __tablename__ = 'sent_agg'

    id          = db.Column(db.Integer, primary_key=True)
    merchant_id = db.Column(db.Integer, db.ForeignKey('merchant.id', ondelete='CASCADE'),
                            unique=True, nullable=False)

    taste_neg  = db.Column(db.Integer, default=0)
    taste_neu  = db.Column(db.Integer, default=0)
    taste_pos  = db.Column(db.Integer, default=0)

    env_neg    = db.Column(db.Integer, default=0)
    env_neu    = db.Column(db.Integer, default=0)
    env_pos    = db.Column(db.Integer, default=0)

    service_neg = db.Column(db.Integer, default=0)
    service_neu = db.Column(db.Integer, default=0)
    service_pos = db.Column(db.Integer, default=0)

    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── helpers ───────────────────────────────────────────────
    SCLS = ['neg', 'neu', 'pos']
    STXT = ['负向', '中立', '正向']
    SEMJ = ['😞', '😐', '😊']

    def _dim(self, dk: str):
        counts = [
            getattr(self, f'{dk}_neg'),
            getattr(self, f'{dk}_neu'),
            getattr(self, f'{dk}_pos'),
        ]
        total = sum(counts)
        if total == 0:
            return None
        best = int(max(range(3), key=lambda i: counts[i]))
        return {
            'label': best,
            'cls':   self.SCLS[best],
            'text':  self.STXT[best],
            'emoji': self.SEMJ[best],
            'pct':   [round(c / total * 100) for c in counts],
            'total': total,
        }

    def to_summary(self):
        """返回与旧代码 agg_summary() 完全相同格式的 dict"""
        return {
            'taste':   self._dim('taste'),
            'env':     self._dim('env'),
            'service': self._dim('service'),
        }

    def add_result(self, result: dict):
        """把一条 predict_one() 结果累加进聚合"""
        dims = result.get('dimensions', {})
        for dk in ('taste', 'env', 'service'):
            label = dims.get(dk, {}).get('label', -1)
            if label == 0:
                setattr(self, f'{dk}_neg', getattr(self, f'{dk}_neg') + 1)
            elif label == 1:
                setattr(self, f'{dk}_neu', getattr(self, f'{dk}_neu') + 1)
            elif label == 2:
                setattr(self, f'{dk}_pos', getattr(self, f'{dk}_pos') + 1)
        self.updated_at = datetime.utcnow()

    def reset(self):
        for dk in ('taste', 'env', 'service'):
            for suffix in ('neg', 'neu', 'pos'):
                setattr(self, f'{dk}_{suffix}', 0)
        self.updated_at = datetime.utcnow()

    def recompute_from_reviews(self):
        """从已审核且已分析的评论重新计算聚合（用于状态变更后重算）"""
        self.reset()
        for rv in self.merchant.reviews.filter_by(status='approved').all():
            ana = rv.get_analysis()
            if ana:
                self.add_result(ana)


class AnalysisLog(db.Model):
    """
    情感分析日志——对应旧的 history_records。
    每次调用 /api/predict 写入一条。
    """
    __tablename__ = 'analysis_log'

    id          = db.Column(db.Integer,  primary_key=True)
    username    = db.Column(db.String(64), nullable=False)
    text        = db.Column(db.Text,     nullable=False)
    result      = db.Column(db.Text,     nullable=False)   # JSON
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def get_result(self):
        try:
            return __import__('json').loads(self.result)
        except Exception:
            return {}

    def to_dict(self):
        r = self.get_result()
        r['time'] = self.created_at.strftime('%m-%d %H:%M')
        return r