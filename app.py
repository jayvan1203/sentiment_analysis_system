"""
app.py  ——  Flask 主程序（数据库版）
依赖：flask, flask-sqlalchemy, sqlalchemy
新增依赖安装：pip install flask-sqlalchemy --break-system-packages
"""
import os, sys, json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import SentimentPredictor
from models import db, User, Merchant, Tag, Review, SentAgg, AnalysisLog

app = Flask(__name__)
app.secret_key = 'sentiment-app-secret-2024'

# ── 数据库配置 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'sentiment.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ── 模型 ───────────────────────────────────────────────────────
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor(
            checkpoint_path=os.environ.get('MODEL_PATH', 'checkpoints/best_model.pt'),
            bert_model_name=os.environ.get('BERT_NAME', 'bert-base-chinese'),
        )
    return _predictor

# ── 情感聚合工具 ───────────────────────────────────────────────
def get_or_create_agg(mid: int) -> SentAgg:
    agg = SentAgg.query.filter_by(merchant_id=mid).first()
    if not agg:
        agg = SentAgg(merchant_id=mid)
        db.session.add(agg)
    return agg

def build_agg_dict(mid: int) -> dict | None:
    """返回与前端兼容的聚合 dict（格式同旧版 agg_summary）"""
    agg = SentAgg.query.filter_by(merchant_id=mid).first()
    return agg.to_summary() if agg else None

def all_agg_dict() -> dict:
    return {str(m.id): build_agg_dict(m.id) for m in Merchant.query.all()}

# ── Auth 装饰器 ────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def d(*a, **kw):
        if 'username' not in session:
            return redirect(url_for('login_page'))
        return f(*a, **kw)
    return d

def api_auth(f):
    @wraps(f)
    def d(*a, **kw):
        if 'username' not in session:
            return jsonify({'error': '请先登录'}), 401
        return f(*a, **kw)
    return d

def api_admin(f):
    @wraps(f)
    def d(*a, **kw):
        if 'username' not in session:
            return jsonify({'error': '请先登录'}), 401
        u = User.query.filter_by(username=session['username']).first()
        if not u or u.role != 'admin':
            return jsonify({'error': '权限不足'}), 403
        return f(*a, **kw)
    return d


# ══════════════════════════════════════════════════════════════
# 种子数据（首次运行自动写入数据库）
# ══════════════════════════════════════════════════════════════
SEED_USERS = [
    {'username': 'admin', 'password': '123456', 'name': '管理员', 'avatar': '👨\u200d💼', 'role': 'admin'},
    {'username': 'alice', 'password': '123456', 'name': '爱丽丝', 'avatar': '👩',          'role': 'user'},
    {'username': 'bob',   'password': '123456', 'name': '小明',   'avatar': '👦',          'role': 'user'},
]

SEED_MERCHANTS = [
    {'name':'外婆家·江南私房菜','category':'江浙菜','rating':4.7,'price':88,'cover':'🏮',
     'address':'上海市黄浦区南京东路168号','open_hours':'10:00-22:00','phone':'021-12345678',
     'tags':['老字号','人均88','必吃榜'],
     'reviews':[
      ('美食达人_小红','👩\u200d🦰',5,'外婆家真的名不虚传！红烧肉肥而不腻，入口即化，汤汁浓郁。环境布置得很有江南风情，木质桌椅，窗帘轻盈，坐在里面感觉很舒适。服务员小姐姐很热情，主动介绍菜品，上菜速度也很快。'),
      ('吃货本货','🧔',4,'龙井虾仁和清蒸鲈鱼都是招牌，味道确实不错，食材新鲜。环境稍微有点吵，因为人很多。但总体来说性价比很高，推荐！'),
      ('周末食客','👩',3,'菜的味道还可以，但是等位等了将近一个小时，服务员态度比较冷漠，叫了好几次才来。价格中规中矩，不算便宜。'),
      ('gourmet_li','👨',5,'每次来上海都要来这里打卡！腌笃鲜鲜美无比，笋香四溢。包厢环境很好，适合商务宴请，服务也非常到位，给满分！'),
      ('素食主义者','🧑',2,'作为素食者，选择非常有限，只有几道蔬菜可以吃。上菜很慢，等了半天才上齐。不太推荐素食者来，荤菜爱好者应该会喜欢。'),
     ]},
    {'name':'海底捞火锅（旗舰店）','category':'火锅','rating':4.9,'price':120,'cover':'🫕',
     'address':'北京市朝阳区三里屯太古里南区B1层','open_hours':'00:00-24:00','phone':'010-87654321',
     'tags':['服务满分','24小时','生日惊喜'],
     'reviews':[
      ('火锅控','🧑\u200d🍳',5,'海底捞的服务真的没话说！等位的时候有零食和美甲，进去之后服务员全程笑脸相迎。锅底选了番茄锅，酸甜适口，食材新鲜。唯一遗憾是价格偏贵，但物有所值。'),
      ('Lucy张','👩\u200d🦱',5,'过生日来的，服务员给唱了生日歌还送了长寿面，感动到哭！食材超级新鲜，虾滑弹弹的，毛肚脆脆的。环境宽敞明亮，强烈推荐！'),
      ('理性消费者','👨\u200d💻',3,'味道是不错的，但是价格真的有点高，人均下来要150左右。环境还好，服务确实很好，但感觉有点过于热情，有些不自在。'),
      ('吃辣小能手','👧',5,'辣锅底太带劲了！配上鸭肠和黄喉，完美！服务员一直帮忙涮菜，非常贴心。店内环境干净整洁，没有异味，赞一个！'),
      ('慢吞吞先生','🧓',4,'食材新鲜度在连锁火锅里算顶尖的了。就是高峰期需要等位，大概等了40分钟，建议提前预约。'),
     ]},
    {'name':'和牛職人·日式烤肉','category':'日式料理','rating':4.5,'price':380,'cover':'🥩',
     'address':'深圳市南山区海岸城购物中心L4','open_hours':'11:30-22:30','phone':'0755-11223344',
     'tags':['和牛','高端','约会首选'],
     'reviews':[
      ('牛肉鉴赏家','🧑\u200d🍽️',5,'A5和牛雪花纹理清晰，在炭火上烤出来油脂四溢，入口即化，简直是人间极品！环境非常有格调，灯光昏黄温馨，很适合约会。服务员懂行，会指导最佳烤法，专业度很高。'),
      ('价格敏感用户','😅',3,'味道确实很好，但是价格实在太贵了，两个人吃了将近800块。环境不错，服务也很好，就是心疼钱包。偶尔犒劳自己还是可以的。'),
      ('日料爱好者小林','👩\u200d🎓',5,'这里的和牛品质真的可以媲美日本本土了！厚切牛舌外焦里嫩，配上柠檬汁太绝了。空间私密，每张桌子都有隔断，服务周到不打扰，完美！'),
      ('普通食客','👨\u200d🦳',2,'肉的质量一般，感觉和标榜的A5有差距。服务态度还可以，但上菜速度太慢了。环境装修不错，但实际性价比很低，不会再来了。'),
      ('商务宴请常客','👔',4,'用来接待外地客户非常合适，档次够高，菜品质量稳定。私人包间很安静，服务也很专业。价格较高但在高端餐饮里算合理。'),
     ]},
    {'name':'喜茶·旗舰店','category':'茶饮甜品','rating':4.3,'price':35,'cover':'🧋',
     'address':'广州市天河区珠江新城花城汇L1','open_hours':'09:00-22:00','phone':'020-99887766',
     'tags':['网红','奶茶','排队王'],
     'reviews':[
      ('奶茶爱好者','👩\u200d🦰',5,'多肉葡萄真的绝了！葡萄皮薄肉厚，茶底清爽，奶盖绵密，甜度刚好。店内环境很时尚，适合拍照打卡。'),
      ('排队苦手','😤',2,'味道是好的，但排队排了将近一个小时，服务员做单太慢，效率很低。而且店里面很吵，环境嘈杂，体验感很差。'),
      ('Cathy_饮品控','👩',4,'芝芝莓莓味道清新，水果颗粒满满。点单后等了20分钟，还算正常。店员态度还不错，偶尔会推荐新品。'),
      ('健康生活家','🏃\u200d♀️',3,'饮品整体口感不错，但含糖量偏高，不适合经常喝。环境还行，但位置不够，节假日根本没地方坐。'),
      ('甜品达人','🧁',5,'限定款草莓系列太香了！每次出新品都会来打卡。店员服务很热情，会介绍各款产品的特点，推荐指数满分！'),
     ]},
    {'name':'老北京炸酱面馆','category':'北京菜','rating':4.6,'price':45,'cover':'🍜',
     'address':'北京市西城区鼓楼大街88号','open_hours':'07:00-21:00','phone':'010-55667788',
     'tags':['老字号','地道','平价实惠'],
     'reviews':[
      ('北京土著','👴',5,'这才是正宗的北京炸酱面！面条劲道爽滑，炸酱浓郁香醇，配上黄瓜丝、豆芽、青豆，七八种小料拌在一起，这口感真是没话说。店里装修复古，有种老北京的市井气息。'),
      ('外地游客小王','🧳',4,'慕名而来，果然不负盛名！炸酱面分量很足，价格也实惠。店里环境有点拥挤，但服务很热情，老板很健谈，给我们普及了很多老北京饮食文化。'),
      ('面食爱好者','🍝',5,'手擀面非常好吃，面条韧劲十足。炸酱里有猪肉丁，肥瘦相间，咸香适口。店内虽小，但卫生干净，老板娘很亲切，像回家吃饭一样温暖。'),
      ('匆匆过客','🚶',3,'中规中矩，味道还可以但不算惊艳。高峰期服务有点跟不上，等了比较久。价格公道，是正常的快餐面馆水准。'),
      ('美食博主Linda','📸',5,'拍摄美食视频专门来打卡！面条手感和口感都是一流的，炸酱的配方应该有几十年历史了。环境虽然简朴，但充满了老北京的生活气息，服务也很接地气，强烈推荐！'),
     ]},
    {'name':'麻辣江湖·川菜私厨','category':'川菜','rating':4.4,'price':95,'cover':'🌶️',
     'address':'成都市锦江区春熙路附近','open_hours':'11:00-22:00','phone':'028-33445566',
     'tags':['麻辣鲜香','私厨','网红店'],
     'reviews':[
      ('辣椒小王子','🌶️',5,'水煮鱼辣得过瘾，麻辣鲜香四味俱全！鱼肉嫩滑，花椒和辣椒的比例很完美。环境很有成都风情，竹编装饰，茶香弥漫。服务员很会聊天，推荐的菜都对我们口味。'),
      ('不能吃辣的我','😰',2,'可能是我太不能吃辣了，点了微辣还是辣得受不了。菜品种类少，环境还可以，但价格不便宜，对于不吃辣的人不太友好。'),
      ('川菜资深食客','🧑\u200d🍳',4,'口水鸡和夫妻肺片都做得很地道，调料配比很到位。就是上菜有点慢，等了将近20分钟。服务态度不错，多次询问是否需要加汤加料。'),
      ('成都本地人','👩\u200d🦱',5,'作为成都人，这家川菜还是很正宗的！麻婆豆腐嫩而入味，锅巴肉片酥脆可口。环境接地气，价格合理，老板也是个实在人，经常在店里招呼客人。'),
      ('外省游客','🗺️',4,'第一次来成都，专门来体验正宗川菜。麻辣味确实很正宗，吃得大汗淋漓但停不下来！环境有特色，服务热情，就是价格比想象中贵一点。'),
     ]},
]


def init_db():
    """建表 + 写入种子数据（仅首次运行，已有数据则跳过）"""
    db.create_all()

    # 用户
    if User.query.count() == 0:
        for ud in SEED_USERS:
            db.session.add(User(**ud))
        db.session.commit()
        print('[DB] 用户种子数据已写入')

    # 餐厅 & 评论
    if Merchant.query.count() == 0:
        pred = get_predictor()
        for md in SEED_MERCHANTS:
            tag_objs = []
            for tn in md.get('tags', []):
                t = Tag.query.filter_by(name=tn).first()
                if not t:
                    t = Tag(name=tn)
                    db.session.add(t)
                tag_objs.append(t)

            m = Merchant(
                name=md['name'], category=md['category'],
                rating=md['rating'], price=md['price'],
                cover=md['cover'], address=md['address'],
                open_hours=md['open_hours'], phone=md['phone'],
                tags=tag_objs,
            )
            db.session.add(m)
            db.session.flush()

            agg = SentAgg(merchant_id=m.id)
            db.session.add(agg)

            print(f'[DB] 分析 {m.name} 评论…')
            for (uname, uav, stars, text) in md['reviews']:
                rv = Review(
                    merchant_id=m.id,
                    user_name=uname, user_avatar=uav,
                    stars=stars, text=text,
                    status='approved',
                )
                try:
                    result = get_predictor().predict_one(text)
                    rv.set_analysis(result)
                    agg.add_result(result)
                except Exception as e:
                    print(f'  warn: {e}')
                db.session.add(rv)

        db.session.commit()
        print('[DB] 餐厅种子数据已写入')

    # ── 修复：若 sent_agg 存在但全部为零（旧库升级场景），重新计算 ──
    all_zero = all(
        (a.taste_neg + a.taste_neu + a.taste_pos +
         a.env_neg + a.env_neu + a.env_pos +
         a.service_neg + a.service_neu + a.service_pos) == 0
        for a in SentAgg.query.all()
    )
    if SentAgg.query.count() > 0 and all_zero:
        print('[DB] 检测到 sent_agg 全为零，正在重新计算…')
        pred = get_predictor()
        for m in Merchant.query.all():
            agg = get_or_create_agg(m.id)
            agg.reset()
            for rv in m.reviews.filter_by(status='approved').all():
                ana = rv.get_analysis()
                if ana is None:
                    # 还没分析过，现在跑一遍
                    try:
                        ana = pred.predict_one(rv.text)
                        rv.set_analysis(ana)
                    except Exception as e:
                        print(f'  warn: {e}')
                        continue
                agg.add_result(ana)
            print(f'  {m.name} ✓')
        db.session.commit()
        print('[DB] sent_agg 重建完成')

    # ── 修复：若有评论但缺少 sent_agg 行，补建 ──
    for m in Merchant.query.all():
        if not SentAgg.query.filter_by(merchant_id=m.id).first():
            db.session.add(SentAgg(merchant_id=m.id))
    db.session.commit()


# ══════════════════════════════════════════════════════════════
# 从数据库构建前端所需的 merchants list（格式与旧版兼容）
# ══════════════════════════════════════════════════════════════
def merchants_for_frontend():
    result = []
    for m in Merchant.query.order_by(Merchant.id).all():
        md = m.to_dict()
        # 前端需要 reviews 列表（含 _analysis）
        md['reviews'] = [r.to_dict() for r in
                         m.reviews.order_by(Review.created_at.desc()).all()]
        result.append(md)
    return result

LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>饮食情报 · 登录</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--red:#D94F3D;--rdk:#B03A2C;--rl:#FDECEA;--ink:#1A1008;--i3:#8A7265;--cream:#FBF7F2;--paper:#F5EFE6;--bd:#E8DDD2;--gold:#C9943A}
body{font-family:'Noto Sans SC',sans-serif;background:var(--cream);min-height:100vh;display:flex;align-items:center;justify-content:center;overflow:hidden}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(217,79,61,.09) 0,transparent 55%),radial-gradient(ellipse at 80% 20%,rgba(201,148,58,.08) 0,transparent 50%);pointer-events:none}
.pat{position:fixed;inset:0;opacity:.025;background-image:repeating-linear-gradient(45deg,var(--red) 0,var(--red) 1px,transparent 0,transparent 50%);background-size:18px 18px;pointer-events:none}
.wrap{width:100%;max-width:440px;padding:20px;animation:up .5s ease both}
@keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:translateY(0)}}
.brand{text-align:center;margin-bottom:28px}
.brand-icon{font-size:52px;display:block;margin-bottom:10px;filter:drop-shadow(0 6px 16px rgba(217,79,61,.28))}
.brand-name{font-family:'Noto Serif SC',serif;font-size:30px;font-weight:700;color:var(--red);letter-spacing:7px}
.brand-sub{font-size:11px;color:var(--i3);letter-spacing:3px;margin-top:5px}
.card{background:#fff;border-radius:18px;padding:34px;box-shadow:0 10px 48px rgba(26,16,8,.10);border:1px solid var(--bd)}
.card-title{font-family:'Noto Serif SC',serif;font-size:16px;font-weight:600;color:var(--ink);margin-bottom:20px;text-align:center}
.field{margin-bottom:14px}
.field label{display:block;font-size:11px;color:var(--i3);letter-spacing:1.5px;margin-bottom:5px;font-weight:500}
.field input{width:100%;padding:11px 14px;border:1.5px solid var(--bd);border-radius:10px;font-size:14px;font-family:inherit;color:var(--ink);background:var(--paper);transition:all .2s;outline:none}
.field input:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(217,79,61,.11);background:#fff}
.btn-login{width:100%;padding:13px;background:var(--red);color:#fff;border:none;border-radius:10px;font-size:14px;font-family:'Noto Serif SC',serif;font-weight:600;letter-spacing:5px;cursor:pointer;transition:all .2s;margin-top:6px;box-shadow:0 4px 18px rgba(217,79,61,.38)}
.btn-login:hover{background:var(--rdk)}.btn-login:active{transform:scale(.98)}
.divider{display:flex;align-items:center;gap:12px;margin:18px 0;color:var(--i3);font-size:11px;letter-spacing:1px}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:var(--bd)}
.demo-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.demo-item{padding:12px 8px;background:var(--paper);border:1.5px solid var(--bd);border-radius:11px;text-align:center;cursor:pointer;transition:all .2s}
.demo-item:hover{border-color:var(--red);background:var(--rl)}
.demo-av{font-size:22px;display:block;margin-bottom:4px}
.demo-name{font-size:12px;color:var(--ink);font-weight:500}
.demo-role{font-size:10px;color:var(--i3);margin-top:2px}
.admin-item{border-color:rgba(201,148,58,.4);background:#fffbf3}
.admin-item:hover{border-color:var(--gold);background:#fff8e7}
.admin-role{color:var(--gold);font-weight:600}
.err{color:var(--red);font-size:12px;text-align:center;margin-top:10px;display:none;padding:8px;background:var(--rl);border-radius:8px}
</style></head><body>
<div class="pat"></div>
<div class="wrap">
  <div class="brand">
    <span class="brand-icon">🍽️</span>
    <div class="brand-name">饮食情报</div>
    <div class="brand-sub">餐饮评论情感分析平台</div>
  </div>
  <div class="card">
    <div class="card-title">登录账户</div>
    <div class="field"><label>账号</label><input type="text" id="u" placeholder="请输入账号" autocomplete="username"></div>
    <div class="field"><label>密码</label><input type="password" id="p" placeholder="请输入密码"></div>
    <button class="btn-login" onclick="go()">登 录</button>
    <div class="err" id="err"></div>
    <div class="divider">快速体验演示账号</div>
    <div class="demo-grid">
      <div class="demo-item admin-item" onclick="fill('admin')"><span class="demo-av">👨‍💼</span><div class="demo-name">admin</div><div class="demo-role admin-role">管理员</div></div>
      <div class="demo-item" onclick="fill('alice')"><span class="demo-av">👩</span><div class="demo-name">alice</div><div class="demo-role">普通用户</div></div>
      <div class="demo-item" onclick="fill('bob')"><span class="demo-av">👦</span><div class="demo-name">bob</div><div class="demo-role">普通用户</div></div>
    </div>
  </div>
</div>
<script>
function fill(u){document.getElementById('u').value=u;document.getElementById('p').value='123456';document.getElementById('err').style.display='none'}
function go(){
  const u=document.getElementById('u').value.trim(),p=document.getElementById('p').value;
  if(!u||!p){show('请填写账号和密码');return}
  fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})})
    .then(r=>r.json()).then(d=>{if(d.ok)location.href=d.redirect;else show(d.error||'登录失败')}).catch(()=>show('网络错误'))
}
function show(m){const e=document.getElementById('err');e.textContent=m;e.style.display='block'}
document.addEventListener('keydown',e=>{if(e.key==='Enter')go()})
</script></body></html>"""

# ══════════════════════════════════════════════════════════════
# USER HTML
# ══════════════════════════════════════════════════════════════
USER_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>饮食情报</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--red:#D94F3D;--rdk:#B03A2C;--rl:#FDECEA;--gold:#C9943A;--ink:#1A1008;--i2:#4A3728;--i3:#8A7265;--cream:#FBF7F2;--paper:#F5EFE6;--bd:#E8DDD2;--green:#2E8B57;--gl:#E6F5EC;--orange:#E07830;--ol:#FEF3E8}
body{font-family:'Noto Sans SC',sans-serif;background:var(--cream);color:var(--ink);min-height:100vh}
.nb{background:#fff;border-bottom:2px solid var(--red);padding:0 22px;display:flex;align-items:center;height:54px;gap:14px;position:sticky;top:0;z-index:100;box-shadow:0 2px 12px rgba(26,16,8,.08)}
.nb-brand{font-family:'Noto Serif SC',serif;font-size:17px;font-weight:700;color:var(--red);letter-spacing:3px;white-space:nowrap}
.nb-tabs{display:flex;gap:3px;flex:1}
.nt{padding:5px 13px;border-radius:8px;font-size:13px;color:var(--i3);cursor:pointer;transition:all .2s;font-weight:500}
.nt:hover{background:var(--paper);color:var(--ink)}.nt.active{background:var(--rl);color:var(--red);font-weight:600}
.nu{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--i2);margin-left:auto}
.nav-av{width:26px;height:26px;border-radius:50%;background:var(--rl);display:flex;align-items:center;justify-content:center;font-size:15px}
.lbtn{padding:4px 10px;border:1.5px solid var(--bd);border-radius:7px;font-size:11px;color:var(--i3);cursor:pointer;background:transparent;font-family:inherit;transition:all .2s}
.lbtn:hover{border-color:var(--red);color:var(--red)}
.page{display:none}.page.active{display:block}
.con{max-width:1060px;margin:0 auto;padding:20px 18px}
.st{font-family:'Noto Serif SC',serif;font-size:17px;font-weight:600;color:var(--ink);margin-bottom:14px;display:flex;align-items:center;gap:7px}
/* Cards */
.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.mc{background:#fff;border-radius:13px;overflow:hidden;border:1px solid var(--bd);cursor:pointer;transition:transform .2s,box-shadow .2s;box-shadow:0 2px 8px rgba(26,16,8,.05)}
.mc:hover{transform:translateY(-3px);box-shadow:0 8px 22px rgba(26,16,8,.11)}
.mc-cov{height:84px;background:linear-gradient(135deg,var(--paper),var(--cream));display:flex;align-items:center;justify-content:center;font-size:40px;border-bottom:1px solid var(--bd);position:relative}
.mc-cov-bg{position:absolute;inset:0;opacity:.04;background-image:repeating-linear-gradient(-45deg,var(--red) 0,var(--red) 1px,transparent 0,transparent 8px)}
.mc-body{padding:12px}
.mc-name{font-family:'Noto Serif SC',serif;font-size:14px;font-weight:600;color:var(--ink);margin-bottom:3px}
.mc-meta{display:flex;align-items:center;gap:9px;font-size:11px;color:var(--i3);margin-bottom:6px}
.rat{color:var(--gold);font-weight:600;font-size:12px}
.tags{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:7px}
.tag{padding:2px 6px;background:var(--paper);border:1px solid var(--bd);border-radius:20px;font-size:10px;color:var(--i2)}
.mc-foot{display:flex;justify-content:space-between;align-items:center;padding-top:6px;border-top:1px solid var(--bd)}
.price{font-size:11px;color:var(--i2)}.price strong{color:var(--red);font-size:13px}
/* Agg strip */
.astrip{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:7px;min-height:22px}
.ab{display:inline-flex;align-items:center;gap:2px;padding:2px 7px;border-radius:20px;font-size:10px;font-weight:500;border:1px solid}
.ab.pos{background:var(--gl);border-color:#a8d8bb;color:var(--green)}.ab.neu{background:var(--ol);border-color:#f0d79a;color:var(--orange)}.ab.neg{background:var(--rl);border-color:#f5b8b3;color:var(--red)}.ab.none{background:var(--paper);border-color:var(--bd);color:var(--i3)}
/* Detail */
.bk{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--i3);cursor:pointer;padding:5px 0;margin-bottom:13px;transition:color .2s}.bk:hover{color:var(--red)}
.dh{background:#fff;border-radius:13px;padding:18px;margin-bottom:12px;border:1px solid var(--bd);display:flex;gap:14px;align-items:flex-start}
.dh-cov{width:68px;height:68px;border-radius:12px;background:var(--paper);display:flex;align-items:center;justify-content:center;font-size:34px;flex-shrink:0;border:1px solid var(--bd)}
.dh-name{font-family:'Noto Serif SC',serif;font-size:18px;font-weight:700;color:var(--ink);margin-bottom:3px}
.dh-addr{font-size:11px;color:var(--i3);margin-top:4px}
.dh-info{font-size:11px;color:var(--i3);margin-top:2px;display:flex;gap:10px}
/* Agg panel */
.ap{background:#fff;border-radius:12px;padding:14px 16px;margin-bottom:12px;border:1px solid var(--bd)}
.ap-title{font-size:11px;color:var(--i3);letter-spacing:1px;margin-bottom:10px;font-weight:500}
.adims{display:flex;gap:8px}
.ad{flex:1;padding:10px;border-radius:9px;text-align:center;border:1.5px solid}
.ad-name{font-size:10px;color:var(--i3);margin-bottom:3px}
.ad-emoji{font-size:22px}
.ad-text{font-size:12px;font-weight:700;margin-top:2px}
.ad-count{font-size:9px;color:var(--i3);margin-top:1px}
.ad.pos{border-color:#a8d8bb;background:var(--gl)}.ad.pos .ad-text{color:var(--green)}
.ad.neu{border-color:#f0d79a;background:var(--ol)}.ad.neu .ad-text{color:var(--orange)}
.ad.neg{border-color:#f5b8b3;background:var(--rl)}.ad.neg .ad-text{color:var(--red)}
.sbar{display:flex;height:4px;border-radius:3px;overflow:hidden;margin-top:5px;gap:1px}
.sb{height:100%}
/* Reviews */
.rv{background:#fff;border-radius:11px;padding:14px 16px;margin-bottom:9px;border:1px solid var(--bd);box-shadow:0 1px 4px rgba(26,16,8,.04)}
.rv-hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:7px}
.rv-user{display:flex;align-items:center;gap:8px}
.rv-av{width:30px;height:30px;border-radius:50%;background:var(--paper);display:flex;align-items:center;justify-content:center;font-size:16px;border:1px solid var(--bd)}
.rv-name{font-size:12px;font-weight:500;color:var(--ink)}
.rv-date{font-size:10px;color:var(--i3);margin-top:1px}
.stars{color:var(--gold);font-size:12px}
.rv-text{font-size:13px;line-height:1.75;color:var(--i2);margin-bottom:10px}
.new-badge{display:inline-block;padding:1px 5px;background:var(--rl);border:1px solid rgba(217,79,61,.3);border-radius:20px;font-size:9px;color:var(--red);margin-left:6px;vertical-align:middle}
.ana-btn{display:inline-flex;align-items:center;gap:4px;padding:6px 12px;background:var(--rl);border:1.5px solid rgba(217,79,61,.3);border-radius:7px;font-size:11px;color:var(--red);cursor:pointer;font-family:inherit;font-weight:500;transition:all .2s}
.ana-btn:hover{background:var(--red);color:#fff;border-color:var(--red)}.ana-btn:disabled{opacity:.6;cursor:not-allowed}
.ar{margin-top:10px;padding:10px 12px;background:var(--paper);border-radius:9px;border:1px solid var(--bd);animation:si .3s ease}
@keyframes si{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
.res-t{font-size:10px;color:var(--i3);letter-spacing:1px;margin-bottom:7px;font-weight:500}
.dr{display:flex;gap:6px}
.dp{flex:1;padding:7px;border-radius:8px;text-align:center;border:1.5px solid}
.dp-n{font-size:9px;color:var(--i3);margin-bottom:2px}.dp-e{font-size:17px}.dp-s{font-size:10px;font-weight:700;margin-top:1px}.dp-c{font-size:9px;color:var(--i3)}
.dp.pos{border-color:#a8d8bb;background:var(--gl)}.dp.pos .dp-s{color:var(--green)}
.dp.neu{border-color:#f0d79a;background:var(--ol)}.dp.neu .dp-s{color:var(--orange)}
.dp.neg{border-color:#f5b8b3;background:var(--rl)}.dp.neg .dp-s{color:var(--red)}
.pm{display:flex;gap:1px;margin-top:4px;height:3px;border-radius:2px;overflow:hidden}.ps{height:100%}
/* Write review */
.wc{background:#fff;border-radius:13px;padding:20px;border:1px solid var(--bd);box-shadow:0 2px 8px rgba(26,16,8,.05);margin-bottom:13px}
.wc h3{font-family:'Noto Serif SC',serif;font-size:15px;font-weight:600;color:var(--ink);margin-bottom:12px}
.msel{width:100%;padding:10px 12px;border:1.5px solid var(--bd);border-radius:9px;font-size:12px;font-family:inherit;color:var(--ink);background:var(--paper);outline:none;cursor:pointer;appearance:none;transition:border-color .2s}
.msel:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(217,79,61,.10)}
.fl{font-size:11px;color:var(--i3);letter-spacing:1px;font-weight:500;margin-bottom:4px}
.si2{display:flex;gap:4px;margin:8px 0 2px}
.sb2{font-size:21px;cursor:pointer;opacity:.25;transition:opacity .15s,transform .1s;border:none;background:transparent;padding:0;line-height:1}
.sb2.lit{opacity:1}.sb2:hover{transform:scale(1.2)}
textarea{width:100%;padding:10px 12px;border:1.5px solid var(--bd);border-radius:9px;font-size:13px;font-family:'Noto Sans SC',sans-serif;resize:vertical;min-height:80px;color:var(--ink);background:var(--paper);outline:none;transition:all .2s;margin-top:6px}
textarea:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(217,79,61,.10);background:#fff}
.btn-sub{padding:9px 22px;background:var(--red);color:#fff;border:none;border-radius:9px;font-size:12px;font-family:'Noto Sans SC',sans-serif;font-weight:500;cursor:pointer;transition:all .2s;box-shadow:0 3px 12px rgba(217,79,61,.3);margin-top:10px}
.btn-sub:hover{background:var(--rdk)}.btn-sub:disabled{opacity:.5;cursor:not-allowed}
.sr{margin-top:13px;padding:14px;background:var(--paper);border-radius:11px;border:1px solid var(--bd);animation:si .3s ease;display:none}
.sr-t{font-size:11px;color:var(--i3);letter-spacing:1px;margin-bottom:10px;font-weight:500}
.bdims{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.bd{padding:12px;border-radius:11px;text-align:center;border:1.5px solid}
.bd-n{font-size:10px;color:var(--i3);margin-bottom:3px}.bd-e{font-size:24px}.bd-s{font-size:12px;font-weight:700;margin:3px 0 2px}.bd-c{font-size:9px;color:var(--i3)}
.bd.pos{border-color:#a8d8bb;background:var(--gl)}.bd.pos .bd-s{color:var(--green)}
.bd.neu{border-color:#f0d79a;background:var(--ol)}.bd.neu .bd-s{color:var(--orange)}
.bd.neg{border-color:#f5b8b3;background:var(--rl)}.bd.neg .bd-s{color:var(--red)}
.pbs{display:flex;flex-direction:column;gap:3px;margin-top:6px;text-align:left}
.pbr{display:flex;align-items:center;gap:4px;font-size:9px;color:var(--i3)}
.pbt{flex:1;height:4px;background:rgba(0,0,0,.07);border-radius:3px;overflow:hidden}
.pbf{height:100%;border-radius:3px;transition:width .5s}
.ok-tip{display:flex;align-items:center;gap:6px;padding:8px 12px;background:var(--gl);border:1px solid #a8d8bb;border-radius:9px;font-size:12px;color:var(--green);margin-bottom:10px}
/* history */
.hc{background:#fff;border-radius:11px;padding:12px 16px;margin-bottom:8px;border:1px solid var(--bd)}
.ht{color:var(--ink);margin-bottom:6px;line-height:1.6;font-size:12px}
.hf{display:flex;justify-content:space-between;align-items:center}
.htime{font-size:10px;color:var(--i3)}.hbs{display:flex;gap:4px}
.badge{padding:2px 7px;border-radius:20px;font-size:10px;font-weight:500}
.badge.pos{background:var(--gl);color:var(--green)}.badge.neu{background:var(--ol);color:var(--orange)}.badge.neg{background:var(--rl);color:var(--red)}
.spin{display:inline-block;width:11px;height:11px;border:2px solid rgba(217,79,61,.2);border-top-color:var(--red);border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle;margin-right:3px}
@keyframes sp{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:32px;color:var(--i3)}.empty .ei{font-size:32px;margin-bottom:7px}.empty p{font-size:12px}
</style></head><body>
<nav class="nb">
  <div class="nb-brand">🍽️ 饮食情报</div>
  <div class="nb-tabs">
    <div class="nt active" id="tab-m" onclick="sp2('m')">🏪 发现餐厅</div>
    <div class="nt" id="tab-r" onclick="sp2('r')">✍️ 写点评</div>
    <div class="nt" id="tab-h" onclick="sp2('h')">📋 历史</div>
  </div>
  <div class="nu">
    <div class="nav-av" id="navAv">👤</div>
    <span id="navNm"></span>
    <button class="lbtn" onclick="logout()">退出</button>
  </div>
</nav>
<div class="page active" id="pg-m">
  <div class="con">
    <div id="dv" style="display:none">
      <div class="bk" onclick="closeD()">← 返回列表</div>
      <div class="dh" id="dhdr"></div>
      <div class="ap"><div class="ap-title">📊 综合情感评分（AI 全量分析）</div><div class="adims" id="dagg"></div></div>
      <div class="st" style="margin-bottom:10px">💬 用户评论</div>
      <div id="rvlist"></div>
    </div>
    <div id="lv"><div class="st">🔥 热门餐厅</div><div class="mgrid" id="mgrid"></div></div>
  </div>
</div>
<div class="page" id="pg-r">
  <div class="con" style="max-width:640px">
    <div class="st">✍️ 写点评</div>
    <div class="wc">
      <h3>选择餐厅 &amp; 填写评论</h3>
      <div class="fl" style="margin-bottom:4px">选择餐厅</div>
      <select class="msel" id="wsel" onchange="onsc()"><option value="">— 请选择 —</option></select>
      <div id="wprev" style="display:none;margin-top:10px;padding:10px;background:var(--paper);border-radius:9px;border:1px solid var(--bd);display:none;gap:10px;align-items:center">
        <span id="wpc" style="font-size:24px"></span>
        <div><div id="wpn" style="font-weight:600;font-size:12px;color:var(--ink)"></div><div id="wpt" style="font-size:10px;color:var(--i3);margin-top:2px"></div></div>
      </div>
      <div style="margin-top:12px">
        <div class="fl">打分</div>
        <div class="si2" id="si2"><button class="sb2 lit" onclick="ss(1)">★</button><button class="sb2 lit" onclick="ss(2)">★</button><button class="sb2 lit" onclick="ss(3)">★</button><button class="sb2" onclick="ss(4)">★</button><button class="sb2" onclick="ss(5)">★</button></div>
        <div id="slbl" style="font-size:11px;color:var(--i3);margin-top:2px">3星 — 一般</div>
      </div>
      <div style="margin-top:12px">
        <div class="fl">评论内容</div>
        <textarea id="wtxt" placeholder="分享用餐体验，口味、环境、服务（至少10字）" rows="4"></textarea>
      </div>
      <button class="btn-sub" id="subBtn" onclick="subRev()">提交并分析</button>
    </div>
    <div id="sres" class="sr">
      <div class="ok-tip" id="oktip">✅ 提交成功</div>
      <div class="sr-t">AI 情感分析结果</div>
      <div class="bdims" id="sdims"></div>
      <div style="margin-top:12px;font-size:11px;color:var(--i3);text-align:center">综合评分已更新 · <span style="color:var(--red);cursor:pointer" onclick="goM()">查看餐厅详情 →</span></div>
    </div>
  </div>
</div>
<div class="page" id="pg-h">
  <div class="con" style="max-width:660px">
    <div class="st">📋 分析历史</div>
    <div id="hlist"><div class="empty"><div class="ei">📭</div><p>暂无记录</p></div></div>
  </div>
</div>
<script>
const DIMS=[{k:'taste',n:'口味'},{k:'env',n:'环境'},{k:'service',n:'服务'}];
const SC=['neg','neu','pos'],SL=['负向','中立','正向'],SE=['😞','😐','😊'];
const PC=['#D94F3D','#E07830','#2E8B57'];
const SLB=['','1星—很差','2星—较差','3星—一般','4星—不错','5星—很棒'];
const UI={{ user_info|tojson }};
let MS={{ merchants|tojson }};
let AGG={{ agg_data|tojson }};
document.getElementById('navAv').textContent=UI.avatar;
document.getElementById('navNm').textContent=UI.name;
function sp2(n){['m','r','h'].forEach(x=>{document.getElementById('pg-'+x).classList.toggle('active',x===n);document.getElementById('tab-'+x).classList.toggle('active',x===n)});if(n==='r')initWR();if(n==='h')loadH();}
/* merchants */
function rndMs(){
  document.getElementById('mgrid').innerHTML=MS.map(m=>{
    const ag=AGG[String(m.id)];
    const strip=ag?DIMS.map(d=>{const s=ag[d.k];return s?`<span class="ab ${s.cls}">${d.n} ${s.emoji} ${s.text}</span>`:`<span class="ab none">${d.n} —</span>`;}).join(''):`<span class="ab none" style="font-size:9px">暂无评分</span>`;
    return`<div class="mc" onclick="openM(${m.id})">
      <div class="mc-cov"><div class="mc-cov-bg"></div><span style="position:relative">${m.cover}</span></div>
      <div class="mc-body">
        <div class="mc-name">${m.name}</div>
        <div class="mc-meta"><span class="rat">★ ${m.rating}</span><span>${m.category}</span></div>
        <div class="tags">${m.tags.map(t=>`<span class="tag">${t}</span>`).join('')}</div>
        <div class="astrip" id="ca-${m.id}">${strip}</div>
        <div class="mc-foot"><div class="price">人均 <strong>¥${m.price}</strong></div><div id="cc-${m.id}" style="font-size:10px;color:var(--i3)">${m.reviews.filter(r=>r.status==='approved').length} 条评论</div></div>
      </div></div>`;
  }).join('');
}
rndMs();
function rfCard(mid){
  const el=document.getElementById('ca-'+mid);if(!el)return;
  const ag=AGG[String(mid)];
  el.innerHTML=ag?DIMS.map(d=>{const s=ag[d.k];return s?`<span class="ab ${s.cls}">${d.n} ${s.emoji} ${s.text}</span>`:`<span class="ab none">${d.n} —</span>`;}).join(''):`<span class="ab none" style="font-size:9px">暂无评分</span>`;
  const m=MS.find(x=>x.id===mid);const cc=document.getElementById('cc-'+mid);
  if(m&&cc)cc.textContent=m.reviews.filter(r=>r.status==='approved').length+' 条评论';
}
let curMid=null;
function openM(id){
  curMid=id;const m=MS.find(x=>x.id===id);if(!m)return;
  document.getElementById('lv').style.display='none';document.getElementById('dv').style.display='block';
  document.getElementById('dhdr').innerHTML=`<div class="dh-cov">${m.cover}</div><div style="flex:1"><div class="dh-name">${m.name}</div><div style="display:flex;align-items:center;gap:10px;margin-top:3px"><span class="rat" style="font-size:13px">★ ${m.rating}</span><span style="font-size:11px;color:var(--i3)">${m.category}</span><span style="font-size:11px;color:var(--red)">人均 ¥${m.price}</span></div><div class="dh-addr">📍 ${m.address}</div><div class="dh-info"><span>⏰ ${m.open_hours||'—'}</span><span>📞 ${m.phone||'—'}</span></div><div class="tags" style="margin-top:6px">${m.tags.map(t=>`<span class="tag">${t}</span>`).join('')}</div></div>`;
  rfAgg(id);rndRvs(id);
}
function closeD(){document.getElementById('lv').style.display='';document.getElementById('dv').style.display='none';curMid=null;}
function rfAgg(mid){
  const ag=AGG[String(mid)];const el=document.getElementById('dagg');
  if(!ag||DIMS.every(d=>!ag[d.k])){el.innerHTML=`<div style="color:var(--i3);font-size:12px;padding:5px">暂无数据——评论分析后自动更新</div>`;return;}
  el.innerHTML=DIMS.map(d=>{
    const s=ag[d.k];if(!s)return`<div class="ad"><div class="ad-name">${d.n}</div><div class="ad-emoji">—</div><div class="ad-text">暂无</div></div>`;
    const bar=s.pct.map((p,i)=>`<div class="sb" style="width:${p}%;background:${PC[i]}"></div>`).join('');
    return`<div class="ad ${s.cls}"><div class="ad-name">${d.n}</div><div class="ad-emoji">${s.emoji}</div><div class="ad-text">${s.text}</div><div class="ad-count">${s.total} 条</div><div class="sbar">${bar}</div></div>`;
  }).join('');
}
function rndRvs(mid){
  const m=MS.find(x=>x.id===mid);
  const approved=[...m.reviews].filter(r=>r.status==='approved').reverse();
  document.getElementById('rvlist').innerHTML=approved.map((r,i)=>{
    const ri=m.reviews.indexOf(r);
    const nb=r._new?`<span class="new-badge">新</span>`:'';
    const body=r._analysis?renderInline(r._analysis):`<button class="ana-btn" id="ab-${mid}-${ri}" onclick="anaRv(${mid},${ri},this)">✨ 分析</button><div id="ar-${mid}-${ri}"></div>`;
    return`<div class="rv" id="rv-${mid}-${ri}"><div class="rv-hd"><div class="rv-user"><div class="rv-av">${r.avatar}</div><div><div class="rv-name">${r.user}${nb}</div><div class="rv-date">${r.date}</div></div></div><div class="stars">${'★'.repeat(r.stars)}${'☆'.repeat(5-r.stars)}</div></div><div class="rv-text">${r.text}</div>${body}</div>`;
  }).join('');
}
function anaRv(mid,idx,btn){
  const m=MS.find(x=>x.id===mid);const text=m.reviews[idx].text;
  const rel=document.getElementById('ar-'+mid+'-'+idx);
  btn.disabled=true;btn.innerHTML='<span class="spin"></span>分析中…';
  predict(text).then(d=>{m.reviews[idx]._analysis=d;btn.style.display='none';rel.innerHTML=renderInline(d);
    fetch('/api/add_agg/'+mid,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(r=>r.json()).then(x=>{AGG[String(mid)]=x.agg;rfAgg(mid);rfCard(mid);});
  }).catch(e=>{btn.disabled=false;btn.innerHTML='✨ 分析';alert(e.message);});
}
function renderInline(d){
  const pills=DIMS.map(dk=>{const i=d.dimensions[dk.k],c=SC[i.label];const segs=[i.probs['负向'],i.probs['中立'],i.probs['正向']].map((p,x)=>`<div class="ps" style="width:${Math.round(p*100)}%;background:${PC[x]}"></div>`).join('');
    return`<div class="dp ${c}"><div class="dp-n">${dk.n}</div><div class="dp-e">${i.emoji}</div><div class="dp-s">${i.sentiment}</div><div class="dp-c">${(i.confidence*100).toFixed(0)}%</div><div class="pm">${segs}</div></div>`;}).join('');
  return`<div class="ar"><div class="res-t">AI 情感分析</div><div class="dr">${pills}</div></div>`;
}
/* write review */
let cstar=3,lastMid=null;
function initWR(){document.getElementById('wsel').innerHTML='<option value="">— 请选择 —</option>'+MS.map(m=>`<option value="${m.id}">${m.cover} ${m.name}</option>`).join('');document.getElementById('sres').style.display='none';}
function onsc(){const mid=parseInt(document.getElementById('wsel').value);const p=document.getElementById('wprev');if(!mid){p.style.display='none';return;}const m=MS.find(x=>x.id===mid);document.getElementById('wpc').textContent=m.cover;document.getElementById('wpn').textContent=m.name;document.getElementById('wpt').textContent=m.category+' · 人均 ¥'+m.price;p.style.display='flex';}
function ss(n){cstar=n;document.querySelectorAll('.sb2').forEach((b,i)=>b.classList.toggle('lit',i<n));document.getElementById('slbl').textContent=SLB[n];}
function subRev(){
  const mid=parseInt(document.getElementById('wsel').value),txt=document.getElementById('wtxt').value.trim();
  if(!mid){alert('请选择餐厅');return;}if(txt.length<10){alert('至少10个字');return;}
  const btn=document.getElementById('subBtn');btn.disabled=true;btn.innerHTML='<span class="spin"></span>提交中…';document.getElementById('sres').style.display='none';
  predict(txt).then(d=>{
    const m=MS.find(x=>x.id===mid);
    const nr={user:UI.name,avatar:UI.avatar,stars:cstar,date:new Date().toISOString().slice(0,10),text:txt,status:'approved',_analysis:d,_new:true};
    m.reviews.push(nr);lastMid=mid;
    return fetch('/api/submit_review/'+mid,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({review:nr,analysis:d})}).then(r=>r.json()).then(x=>{AGG[String(mid)]=x.agg;rfCard(mid);if(curMid===mid){rfAgg(mid);rndRvs(mid);}showSR(d,mid);});
  }).catch(e=>alert(e.message)).finally(()=>{btn.disabled=false;btn.innerHTML='提交并分析';document.getElementById('wtxt').value='';ss(3);});
}
function showSR(d,mid){
  const m=MS.find(x=>x.id===mid);document.getElementById('oktip').textContent='✅ 已提交到「'+m.name+'」';
  document.getElementById('sdims').innerHTML=DIMS.map(dk=>{const i=d.dimensions[dk.k],c=SC[i.label];const bars=[{l:'负向',v:i.probs['负向'],c:PC[0]},{l:'中立',v:i.probs['中立'],c:PC[1]},{l:'正向',v:i.probs['正向'],c:PC[2]}].map(p=>`<div class="pbr"><span style="width:22px">${p.l}</span><div class="pbt"><div class="pbf" style="width:${Math.round(p.v*100)}%;background:${p.c}"></div></div><span style="width:26px;text-align:right">${Math.round(p.v*100)}%</span></div>`).join('');
    return`<div class="bd ${c}"><div class="bd-n">${dk.n}</div><div class="bd-e">${i.emoji}</div><div class="bd-s">${i.sentiment}</div><div class="bd-c">${(i.confidence*100).toFixed(1)}%</div><div class="pbs">${bars}</div></div>`;}).join('');
  document.getElementById('sres').style.display='block';
}
function goM(){if(lastMid){sp2('m');openM(lastMid);}}
/* history */
function loadH(){fetch('/api/history').then(r=>r.json()).then(recs=>{const el=document.getElementById('hlist');if(!recs.length){el.innerHTML='<div class="empty"><div class="ei">📭</div><p>暂无记录</p></div>';return;}el.innerHTML=[...recs].reverse().slice(0,40).map(it=>{const bs=DIMS.map(d=>{const i=it.dimensions[d.k];return`<span class="badge ${SC[i.label]}">${d.n}:${i.sentiment}</span>`;}).join('');return`<div class="hc"><div class="ht">${it.text}</div><div class="hf"><div class="hbs">${bs}</div><div class="htime">${it.time}</div></div></div>`;}).join('');});}
function predict(t){return fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})}).then(r=>r.json().then(d=>{if(!r.ok)throw new Error(d.error||'请求失败');return d;}));}
function logout(){fetch('/api/logout',{method:'POST'}).then(()=>location.href='/login');}
</script></body></html>"""

# ══════════════════════════════════════════════════════════════
# ADMIN HTML
# ══════════════════════════════════════════════════════════════
ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>饮食情报 · 管理后台</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--red:#D94F3D;--rdk:#B03A2C;--rl:#FDECEA;--gold:#C9943A;--ink:#1A1008;--i2:#4A3728;--i3:#8A7265;--cream:#FBF7F2;--paper:#F5EFE6;--bd:#E8DDD2;--green:#2E8B57;--gl:#E6F5EC;--orange:#E07830;--ol:#FEF3E8;--blue:#2563EB;--bl:#EEF2FF;--sidebar:#1C1410}
body{font-family:'Noto Sans SC',sans-serif;background:var(--cream);color:var(--ink);min-height:100vh;display:flex}
/* Sidebar */
.sidebar{width:220px;background:var(--sidebar);min-height:100vh;flex-shrink:0;display:flex;flex-direction:column;position:sticky;top:0}
.sb-brand{padding:20px 18px 16px;border-bottom:1px solid rgba(255,255,255,.08)}
.sb-logo{font-size:22px;margin-bottom:4px}
.sb-title{font-family:'Noto Serif SC',serif;font-size:14px;font-weight:700;color:#fff;letter-spacing:2px}
.sb-sub{font-size:10px;color:rgba(255,255,255,.4);margin-top:2px;letter-spacing:1px}
.sb-nav{flex:1;padding:12px 10px}
.sn{display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:8px;font-size:13px;color:rgba(255,255,255,.6);cursor:pointer;transition:all .2s;margin-bottom:2px}
.sn:hover{background:rgba(255,255,255,.08);color:rgba(255,255,255,.9)}
.sn.active{background:var(--red);color:#fff;font-weight:500}
.sn-icon{font-size:15px;width:18px;text-align:center}
.sb-footer{padding:14px 16px;border-top:1px solid rgba(255,255,255,.08)}
.sb-user{display:flex;align-items:center;gap:8px;font-size:12px;color:rgba(255,255,255,.6)}
.sb-av{width:28px;height:28px;border-radius:50%;background:rgba(217,79,61,.4);display:flex;align-items:center;justify-content:center;font-size:15px}
.lbtn2{padding:4px 10px;border:1px solid rgba(255,255,255,.2);border-radius:6px;font-size:11px;color:rgba(255,255,255,.5);cursor:pointer;background:transparent;font-family:inherit;transition:all .2s;margin-left:auto}
.lbtn2:hover{border-color:var(--red);color:var(--red)}
/* Main */
.main{flex:1;overflow-x:hidden}
.topbar{background:#fff;border-bottom:1px solid var(--bd);padding:0 24px;height:52px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 1px 6px rgba(26,16,8,.05)}
.topbar-title{font-family:'Noto Serif SC',serif;font-size:16px;font-weight:600;color:var(--ink)}
.topbar-badge{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;background:#FEF3C7;border:1px solid #F59E0B;border-radius:20px;font-size:11px;color:#92400E}
.page{display:none;padding:22px 24px}.page.active{display:block}
/* Stats row */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.stat{background:#fff;border-radius:12px;padding:16px;border:1px solid var(--bd);box-shadow:0 1px 6px rgba(26,16,8,.04)}
.stat-val{font-family:'Noto Serif SC',serif;font-size:26px;font-weight:700;color:var(--ink);margin-bottom:3px}
.stat-label{font-size:11px;color:var(--i3)}
.stat-icon{font-size:22px;float:right;margin-top:-4px}
/* Merchant table */
.tbl-wrap{background:#fff;border-radius:12px;border:1px solid var(--bd);overflow:hidden;box-shadow:0 1px 6px rgba(26,16,8,.04)}
.tbl-head{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--bd)}
.tbl-title{font-size:14px;font-weight:600;color:var(--ink)}
.btn-add{display:inline-flex;align-items:center;gap:5px;padding:7px 14px;background:var(--red);color:#fff;border:none;border-radius:8px;font-size:12px;font-family:inherit;cursor:pointer;transition:all .2s;box-shadow:0 2px 8px rgba(217,79,61,.3)}
.btn-add:hover{background:var(--rdk)}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:var(--paper);padding:9px 14px;text-align:left;color:var(--i3);font-weight:500;font-size:11px;letter-spacing:.5px;border-bottom:1px solid var(--bd)}
td{padding:10px 14px;border-bottom:1px solid rgba(232,221,210,.5);vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(251,247,242,.6)}
.mc-cell{display:flex;align-items:center;gap:9px}
.mc-icon{width:32px;height:32px;border-radius:8px;background:var(--paper);display:flex;align-items:center;justify-content:center;font-size:18px;border:1px solid var(--bd)}
.mc-cname{font-weight:500;color:var(--ink)}
.mc-cat{font-size:10px;color:var(--i3)}
.ab2{display:inline-flex;align-items:center;gap:2px;padding:2px 6px;border-radius:20px;font-size:10px;font-weight:500;border:1px solid}
.ab2.pos{background:var(--gl);border-color:#a8d8bb;color:var(--green)}.ab2.neu{background:var(--ol);border-color:#f0d79a;color:var(--orange)}.ab2.neg{background:var(--rl);border-color:#f5b8b3;color:var(--red)}.ab2.none{background:var(--paper);border-color:var(--bd);color:var(--i3)}
.act-btns{display:flex;gap:5px}
.btn-sm{padding:4px 10px;border-radius:6px;font-size:11px;cursor:pointer;border:1.5px solid;font-family:inherit;transition:all .2s}
.btn-edit{background:#fff;border-color:var(--bd);color:var(--i2)}.btn-edit:hover{border-color:var(--blue);color:var(--blue)}
.btn-del{background:#fff;border-color:var(--bd);color:var(--i2)}.btn-del:hover{border-color:var(--red);color:var(--red);background:var(--rl)}
/* Review management */
.rv-row{background:#fff;border-radius:10px;padding:12px 15px;margin-bottom:8px;border:1px solid var(--bd)}
.rv-meta{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.rv-who{display:flex;align-items:center;gap:7px;font-size:12px}
.rv-av2{width:26px;height:26px;border-radius:50%;background:var(--paper);display:flex;align-items:center;justify-content:center;font-size:14px}
.rv-mname{font-size:10px;color:var(--i3);margin-top:1px}
.stars2{color:var(--gold);font-size:11px}
.rv-txt{font-size:12px;line-height:1.65;color:var(--i2);margin-bottom:8px}
.rv-agg{display:flex;gap:5px;margin-bottom:8px;flex-wrap:wrap}
.rv-acts{display:flex;gap:5px}
.btn-approve{padding:3px 9px;border-radius:5px;font-size:10px;cursor:pointer;border:1.5px solid;font-family:inherit;background:var(--gl);border-color:#a8d8bb;color:var(--green)}
.btn-approve:hover{background:#c3ead2}
.btn-reject{padding:3px 9px;border-radius:5px;font-size:10px;cursor:pointer;border:1.5px solid;font-family:inherit;background:var(--rl);border-color:#f5b8b3;color:var(--red)}
.btn-reject:hover{background:#fad3cf}
.status-badge{padding:2px 7px;border-radius:20px;font-size:10px;font-weight:500}
.s-approved{background:var(--gl);color:var(--green)}.s-pending{background:#FEF3C7;color:#92400E}.s-rejected{background:var(--rl);color:var(--red)}
/* Modal */
.modal-bg{position:fixed;inset:0;background:rgba(26,16,8,.5);z-index:200;display:none;align-items:center;justify-content:center;backdrop-filter:blur(4px)}
.modal-bg.open{display:flex}
.modal{background:#fff;border-radius:16px;padding:28px;width:100%;max-width:520px;box-shadow:0 20px 60px rgba(26,16,8,.2);animation:mup .25s ease}
@keyframes mup{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.modal-title{font-family:'Noto Serif SC',serif;font-size:16px;font-weight:600;color:var(--ink);margin-bottom:18px}
.mfield{margin-bottom:13px}
.mfield label{display:block;font-size:11px;color:var(--i3);letter-spacing:.8px;margin-bottom:5px;font-weight:500}
.mfield input,.mfield select,.mfield textarea{width:100%;padding:9px 12px;border:1.5px solid var(--bd);border-radius:8px;font-size:13px;font-family:inherit;color:var(--ink);background:var(--paper);outline:none;transition:all .2s}
.mfield input:focus,.mfield select:focus,.mfield textarea:focus{border-color:var(--red);box-shadow:0 0 0 3px rgba(217,79,61,.10);background:#fff}
.mfield .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.modal-foot{display:flex;justify-content:flex-end;gap:8px;margin-top:20px;padding-top:16px;border-top:1px solid var(--bd)}
.btn-cancel{padding:8px 18px;border:1.5px solid var(--bd);border-radius:8px;font-size:12px;color:var(--i3);cursor:pointer;background:#fff;font-family:inherit;transition:all .2s}.btn-cancel:hover{border-color:var(--ink);color:var(--ink)}
.btn-save{padding:8px 20px;background:var(--red);color:#fff;border:none;border-radius:8px;font-size:12px;font-family:inherit;cursor:pointer;transition:all .2s;box-shadow:0 2px 8px rgba(217,79,61,.3)}.btn-save:hover{background:var(--rdk)}
/* Charts */
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
.chart-card{background:#fff;border-radius:12px;padding:16px;border:1px solid var(--bd)}
.chart-title{font-size:13px;font-weight:600;color:var(--ink);margin-bottom:12px}
.bar-chart{display:flex;flex-direction:column;gap:6px}
.bar-row{display:flex;align-items:center;gap:8px;font-size:11px}
.bar-label{width:60px;color:var(--i3);text-align:right;flex-shrink:0}
.bar-track{flex:1;background:var(--paper);border-radius:4px;height:16px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:4px;transition:width .6s ease;display:flex;align-items:center;padding-left:6px;font-size:10px;color:#fff;font-weight:500;white-space:nowrap}
.bar-count{width:28px;color:var(--i3);text-align:right;flex-shrink:0}
.donut-wrap{display:flex;align-items:center;gap:16px}
.donut-legend{display:flex;flex-direction:column;gap:5px}
.dl-item{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--i2)}
.dl-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
/* Misc */
.spin{display:inline-block;width:11px;height:11px;border:2px solid rgba(217,79,61,.2);border-top-color:var(--red);border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle;margin-right:3px}
@keyframes sp{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:30px;color:var(--i3)}.empty .ei{font-size:30px;margin-bottom:6px}.empty p{font-size:12px}
.filter-row{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.fsel{padding:6px 10px;border:1.5px solid var(--bd);border-radius:7px;font-size:12px;font-family:inherit;color:var(--ink);background:#fff;outline:none;cursor:pointer;transition:border-color .2s}
.fsel:focus{border-color:var(--red)}
.tag-input{display:flex;gap:6px;flex-wrap:wrap;padding:7px 10px;border:1.5px solid var(--bd);border-radius:8px;background:var(--paper);min-height:38px;align-items:center;cursor:text;transition:all .2s}
.tag-input:focus-within{border-color:var(--red);background:#fff}
.tag-chip{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;background:var(--rl);border:1px solid rgba(217,79,61,.3);border-radius:20px;font-size:11px;color:var(--red)}
.tag-chip-del{cursor:pointer;opacity:.6;font-size:12px;line-height:1}.tag-chip-del:hover{opacity:1}
.tag-input-field{border:none;outline:none;background:transparent;font-size:12px;font-family:inherit;color:var(--ink);min-width:80px}
</style></head><body>
<aside class="sidebar">
  <div class="sb-brand"><div class="sb-logo">🍽️</div><div class="sb-title">饮食情报</div><div class="sb-sub">管理后台</div></div>
  <nav class="sb-nav">
    <div class="sn active" id="sn-dash" onclick="sp('dash')"><span class="sn-icon">📊</span>数据看板</div>
    <div class="sn" id="sn-merchants" onclick="sp('merchants')"><span class="sn-icon">🏪</span>餐厅管理</div>
    <div class="sn" id="sn-reviews" onclick="sp('reviews')"><span class="sn-icon">💬</span>评论管理<span id="pend-cnt" style="margin-left:auto;background:var(--red);color:#fff;padding:1px 6px;border-radius:10px;font-size:10px;display:none"></span></div>
    <div class="sn" id="sn-sentiment" onclick="sp('sentiment')"><span class="sn-icon">🧠</span>情感分析</div>
    <div class="sn" id="sn-users" onclick="sp('users')"><span class="sn-icon">👥</span>用户管理</div>
    <div class="sn" id="sn-db" onclick="sp('db')"><span class="sn-icon">🗄️</span>数据库状态</div>
    <div class="sn" id="sn-bert" onclick="sp('bert')"><span class="sn-icon">🤖</span>BERT 模型</div>
  </nav>
  <div class="sb-footer">
    <div class="sb-user">
      <div class="sb-av" id="sbAv">👤</div>
      <span id="sbNm"></span>
      <button class="lbtn2" onclick="logout()">退出</button>
    </div>
  </div>
</aside>
<div class="main">
  <div class="topbar">
    <div class="topbar-title" id="topbar-title">数据看板</div>
    <div class="topbar-badge">👨‍💼 管理员模式</div>
  </div>

  <!-- Dashboard -->
  <div class="page active" id="pg-dash">
    <div class="stats">
      <div class="stat"><span class="stat-icon">🏪</span><div class="stat-val" id="s-mc">—</div><div class="stat-label">餐厅总数</div></div>
      <div class="stat"><span class="stat-icon">💬</span><div class="stat-val" id="s-rv">—</div><div class="stat-label">评论总数</div></div>
      <div class="stat"><span class="stat-icon">🧠</span><div class="stat-val" id="s-ana">—</div><div class="stat-label">已分析评论</div></div>
      <div class="stat"><span class="stat-icon">⏳</span><div class="stat-val" id="s-pend">—</div><div class="stat-label">待审核评论</div></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-title">各餐厅综合口味情感</div>
        <div class="bar-chart" id="chart-taste"></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">各餐厅综合服务情感</div>
        <div class="bar-chart" id="chart-service"></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">各餐厅综合环境情感</div>
        <div class="bar-chart" id="chart-env"></div>
      </div>
      <div class="chart-card">
        <div class="chart-title">整体情感分布</div>
        <div class="donut-wrap"><canvas id="donut" width="100" height="100"></canvas><div class="donut-legend" id="donut-leg"></div></div>
      </div>
    </div>
  </div>

  <!-- Merchants -->
  <div class="page" id="pg-merchants">
    <div class="tbl-wrap">
      <div class="tbl-head">
        <div class="tbl-title">餐厅列表</div>
        <button class="btn-add" onclick="openAdd()">+ 添加餐厅</button>
      </div>
      <table>
        <thead><tr><th>餐厅</th><th>分类</th><th>评分</th><th>人均</th><th>评论数</th><th>口味</th><th>环境</th><th>服务</th><th>操作</th></tr></thead>
        <tbody id="mc-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- Reviews -->
  <div class="page" id="pg-reviews">
    <div class="filter-row">
      <select class="fsel" id="rv-fmc" onchange="loadRvs()"><option value="">全部餐厅</option></select>
      <select class="fsel" id="rv-fst" onchange="loadRvs()"><option value="">全部状态</option><option value="approved">已通过</option><option value="pending">待审核</option><option value="rejected">已拒绝</option></select>
    </div>
    <div id="rv-list"></div>
  </div>

  <!-- Sentiment analysis tool -->
  <div class="page" id="pg-sentiment">
    <div style="max-width:600px">
      <div style="background:#fff;border-radius:12px;padding:20px;border:1px solid var(--bd);margin-bottom:14px">
        <div style="font-size:14px;font-weight:600;color:var(--ink);margin-bottom:12px">🔍 自由情感分析</div>
        <textarea id="sa-txt" rows="4" style="width:100%;padding:10px 12px;border:1.5px solid var(--bd);border-radius:9px;font-size:13px;font-family:inherit;color:var(--ink);background:var(--paper);outline:none;resize:vertical;min-height:80px" placeholder="输入任意餐饮评论进行分析…"></textarea>
        <button onclick="doSA()" style="margin-top:8px;padding:8px 20px;background:var(--red);color:#fff;border:none;border-radius:8px;font-size:12px;font-family:inherit;cursor:pointer" id="sabtn">分析情感</button>
      </div>
      <div id="sa-res" style="display:none;background:#fff;border-radius:12px;padding:16px;border:1px solid var(--bd)">
        <div style="font-size:11px;color:var(--i3);letter-spacing:1px;margin-bottom:10px">分析结果</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px" id="sa-dims"></div>
      </div>
    </div>
  </div>

  <!-- Users -->
  <div class="page" id="pg-users">
    <div class="tbl-wrap">
      <div class="tbl-head"><div class="tbl-title">用户列表</div></div>
      <table>
        <thead><tr><th>用户</th><th>账号</th><th>角色</th><th>提交评论数</th></tr></thead>
        <tbody id="u-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

  <!-- ══ DB Page ══ -->
  <div class="page" id="pg-db">
    <div class="stats" style="grid-template-columns:repeat(5,1fr)">
      <div class="stat"><span class="stat-icon">🗄️</span><div class="stat-val" id="db-file">—</div><div class="stat-label">数据库文件大小</div></div>
      <div class="stat"><span class="stat-icon">👥</span><div class="stat-val" id="db-users">—</div><div class="stat-label">用户表 (user)</div></div>
      <div class="stat"><span class="stat-icon">🏪</span><div class="stat-val" id="db-merchants">—</div><div class="stat-label">餐厅表 (merchant)</div></div>
      <div class="stat"><span class="stat-icon">💬</span><div class="stat-val" id="db-reviews">—</div><div class="stat-label">评论表 (review)</div></div>
      <div class="stat"><span class="stat-icon">📋</span><div class="stat-val" id="db-logs">—</div><div class="stat-label">日志表 (analysis_log)</div></div>
    </div>

    <!-- Table schema cards -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">
      <div class="chart-card">
        <div class="chart-title">📐 数据库表结构</div>
        <div style="font-size:11px;color:var(--i3);line-height:1.8;font-family:monospace">
          <div style="margin-bottom:8px"><span style="color:var(--red);font-weight:600">user</span>　id · username · password · name · avatar · role · created_at</div>
          <div style="margin-bottom:8px"><span style="color:var(--red);font-weight:600">merchant</span>　id · name · category · rating · price · cover · address · open_hours · phone · created_at</div>
          <div style="margin-bottom:8px"><span style="color:var(--red);font-weight:600">tag</span>　id · name</div>
          <div style="margin-bottom:8px"><span style="color:var(--red);font-weight:600">merchant_tags</span>　merchant_id · tag_id　<span style="color:var(--i3)">(多对多中间表)</span></div>
          <div style="margin-bottom:8px"><span style="color:var(--red);font-weight:600">review</span>　id · merchant_id · user_name · user_avatar · stars · text · status · analysis · created_at</div>
          <div style="margin-bottom:8px"><span style="color:var(--red);font-weight:600">sent_agg</span>　id · merchant_id · taste/env/service × neg/neu/pos · updated_at</div>
          <div><span style="color:var(--red);font-weight:600">analysis_log</span>　id · username · text · result · created_at</div>
        </div>
      </div>
      <div class="chart-card">
        <div class="chart-title">🔗 表关系示意</div>
        <div style="font-size:11px;color:var(--i3);line-height:2;font-family:monospace">
          <div>merchant <span style="color:var(--red)">──1:N──</span> review</div>
          <div>merchant <span style="color:var(--red)">──1:1──</span> sent_agg</div>
          <div>merchant <span style="color:var(--red)">──M:N──</span> tag　(通过 merchant_tags)</div>
          <div style="margin-top:10px;color:var(--ink);font-family:'Noto Sans SC',sans-serif">
            <div>• 删除餐厅 → 级联删除评论 + 聚合行</div>
            <div>• 评论状态变更 → 重算 sent_agg</div>
            <div>• 新评论提交 → 增量更新 sent_agg</div>
            <div>• 每次推理 → 写入 analysis_log</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent logs table -->
    <div class="tbl-wrap">
      <div class="tbl-head">
        <div class="tbl-title">📋 最近分析日志 (analysis_log)</div>
        <button class="btn-add" style="background:var(--i2)" onclick="loadDbPage()">🔄 刷新</button>
      </div>
      <table>
        <thead><tr><th>ID</th><th>用户</th><th style="min-width:200px">评论文本</th><th>口味</th><th>环境</th><th>服务</th><th>时间</th></tr></thead>
        <tbody id="log-tbody"></tbody>
      </table>
    </div>

    <!-- sent_agg table -->
    <div class="tbl-wrap" style="margin-top:14px">
      <div class="tbl-head"><div class="tbl-title">📊 情感聚合缓存 (sent_agg)</div></div>
      <table>
        <thead><tr><th>餐厅</th><th>口味-负</th><th>口味-中</th><th>口味-正</th><th>环境-负</th><th>环境-中</th><th>环境-正</th><th>服务-负</th><th>服务-中</th><th>服务-正</th><th>更新时间</th></tr></thead>
        <tbody id="agg-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ══ BERT Page ══ -->
  <div class="page" id="pg-bert">
    <!-- Model info cards -->
    <div class="stats" style="grid-template-columns:repeat(4,1fr);margin-bottom:18px">
      <div class="stat"><span class="stat-icon">🏗️</span><div class="stat-val" style="font-size:16px">bert-base-chinese</div><div class="stat-label">预训练模型</div></div>
      <div class="stat"><span class="stat-icon">🔢</span><div class="stat-val">102M</div><div class="stat-label">模型参数量</div></div>
      <div class="stat"><span class="stat-icon">📐</span><div class="stat-val">768</div><div class="stat-label">隐藏层维度</div></div>
      <div class="stat"><span class="stat-icon">🎯</span><div class="stat-val">3 类</div><div class="stat-label">每维度输出（负/中/正）</div></div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px">
      <!-- Architecture -->
      <div class="chart-card">
        <div class="chart-title">🏗️ 模型架构</div>
        <div style="font-size:12px;color:var(--i2);line-height:1.9">
          <div style="display:flex;flex-direction:column;gap:6px">
            <div style="padding:8px 12px;background:var(--bl);border-radius:8px;border-left:3px solid var(--blue);font-size:11px">
              <strong style="color:var(--blue)">输入层</strong><br>
              评论文本 → BertTokenizer 分词 → input_ids / attention_mask / token_type_ids<br>
              <span style="color:var(--i3)">max_length=128，padding / truncation</span>
            </div>
            <div style="text-align:center;color:var(--i3);font-size:11px">↓</div>
            <div style="padding:8px 12px;background:#FEF3C7;border-radius:8px;border-left:3px solid #F59E0B;font-size:11px">
              <strong style="color:#92400E">BERT 编码器（共享）</strong><br>
              12 层 Transformer · 12 注意力头 · hidden=768<br>
              输出 last_hidden_state: (batch, 128, 768)
            </div>
            <div style="text-align:center;color:var(--i3);font-size:11px">↓ 分三路独立处理</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">
              <div style="padding:7px;background:var(--gl);border-radius:8px;font-size:10px;text-align:center;border:1px solid #a8d8bb">
                <strong style="color:var(--green)">口味分支</strong><br>
                注意力池化层<br>Linear(768→1)<br>↓<br>加权求和 → 768维
              </div>
              <div style="padding:7px;background:var(--ol);border-radius:8px;font-size:10px;text-align:center;border:1px solid #f0d79a">
                <strong style="color:var(--orange)">环境分支</strong><br>
                注意力池化层<br>Linear(768→1)<br>↓<br>加权求和 → 768维
              </div>
              <div style="padding:7px;background:var(--rl);border-radius:8px;font-size:10px;text-align:center;border:1px solid #f5b8b3">
                <strong style="color:var(--red)">服务分支</strong><br>
                注意力池化层<br>Linear(768→1)<br>↓<br>加权求和 → 768维
              </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px">
              <div style="padding:6px;background:var(--paper);border-radius:8px;font-size:10px;text-align:center">Dropout(0.1)<br>↓<br>Linear(768→3)<br>↓<br>logits[3]</div>
              <div style="padding:6px;background:var(--paper);border-radius:8px;font-size:10px;text-align:center">Dropout(0.1)<br>↓<br>Linear(768→3)<br>↓<br>logits[3]</div>
              <div style="padding:6px;background:var(--paper);border-radius:8px;font-size:10px;text-align:center">Dropout(0.1)<br>↓<br>Linear(768→3)<br>↓<br>logits[3]</div>
            </div>
            <div style="text-align:center;color:var(--i3);font-size:11px">↓ Softmax</div>
            <div style="padding:7px 12px;background:var(--paper);border-radius:8px;font-size:11px;text-align:center">
              口味概率[负/中/正] · 环境概率[负/中/正] · 服务概率[负/中/正]
            </div>
          </div>
        </div>
      </div>

      <!-- Training details -->
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="chart-card">
          <div class="chart-title">⚙️ 训练配置</div>
          <table style="font-size:11px;width:100%">
            <tr><td style="color:var(--i3);padding:4px 0">数据集</td><td style="font-weight:500">美团 ASAP（NAACL 2021）</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">训练轮数</td><td>10 epochs（早停 patience=4）</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">批次大小</td><td>32</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">BERT 学习率</td><td>2e-5</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">分类头学习率</td><td>1e-4（BERT × 5）</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">优化器</td><td>AdamW + 线性预热调度</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">损失函数</td><td>CrossEntropy（ignore_index=-1）</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">标签平滑</td><td>0.1</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">类别重加权</td><td>balanced（缓解中立类不足）</td></tr>
            <tr><td style="color:var(--i3);padding:4px 0">中立类过采样</td><td>target_ratio=0.25</td></tr>
          </table>
        </div>
        <div class="chart-card">
          <div class="chart-title">📊 测试集指标</div>
          <table style="font-size:11px;width:100%;border-collapse:collapse">
            <thead><tr><th style="background:var(--paper);padding:5px 8px;text-align:left;color:var(--i3)">维度</th><th style="background:var(--paper);padding:5px 8px;color:var(--i3)">准确率</th><th style="background:var(--paper);padding:5px 8px;color:var(--i3)">macro-F1</th><th style="background:var(--paper);padding:5px 8px;color:var(--i3)">样本数</th></tr></thead>
            <tbody>
              <tr><td style="padding:5px 8px">口味</td><td style="padding:5px 8px;text-align:center"><span style="color:var(--green);font-weight:600">81.5%</span></td><td style="padding:5px 8px;text-align:center">82.6%</td><td style="padding:5px 8px;text-align:center;color:var(--i3)">5,243</td></tr>
              <tr><td style="padding:5px 8px">环境</td><td style="padding:5px 8px;text-align:center"><span style="color:var(--green);font-weight:600">87.2%</span></td><td style="padding:5px 8px;text-align:center">83.4%</td><td style="padding:5px 8px;text-align:center;color:var(--i3)">3,293</td></tr>
              <tr><td style="padding:5px 8px">服务</td><td style="padding:5px 8px;text-align:center"><span style="color:var(--green);font-weight:600">81.8%</span></td><td style="padding:5px 8px;text-align:center">78.7%</td><td style="padding:5px 8px;text-align:center;color:var(--i3)">3,995</td></tr>
              <tr style="border-top:1px solid var(--bd)"><td style="padding:5px 8px;font-weight:600">平均</td><td style="padding:5px 8px;text-align:center"><span style="color:var(--red);font-weight:700">83.1%</span></td><td style="padding:5px 8px;text-align:center;font-weight:600">81.6%</td><td style="padding:5px 8px;text-align:center;color:var(--i3)">12,531</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Training loss chart -->
    <div class="chart-card" style="margin-bottom:14px">
      <div class="chart-title">📉 训练过程（Loss & val F1）</div>
      <canvas id="train-chart" height="60"></canvas>
    </div>

    <!-- Live inference test -->
    <div class="chart-card">
      <div class="chart-title">⚡ 实时推理测试</div>
      <div style="display:flex;gap:10px;align-items:flex-start">
        <textarea id="bert-test-txt" rows="2" style="flex:1;padding:9px 12px;border:1.5px solid var(--bd);border-radius:9px;font-size:12px;font-family:inherit;resize:none;outline:none;transition:all .2s;background:var(--paper)" placeholder="输入任意评论，观察模型原始输出…" onfocus="this.style.borderColor='var(--red)'" onblur="this.style.borderColor='var(--bd)'"></textarea>
        <button onclick="doBertTest()" style="padding:9px 16px;background:var(--red);color:#fff;border:none;border-radius:9px;font-size:12px;font-family:inherit;cursor:pointer;white-space:nowrap" id="bert-test-btn">推理</button>
      </div>
      <div id="bert-test-res" style="display:none;margin-top:12px">
        <div style="font-size:10px;color:var(--i3);letter-spacing:1px;margin-bottom:8px">模型输出（Softmax 概率分布）</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px" id="bert-test-dims"></div>
        <div style="margin-top:12px;padding:10px 12px;background:var(--paper);border-radius:8px;font-size:10px;font-family:monospace;color:var(--i3);line-height:1.7" id="bert-raw-out"></div>
      </div>
    </div>
  </div>

<!-- Add/Edit Modal -->
<div class="modal-bg" id="modal">
  <div class="modal">
    <div class="modal-title" id="modal-title">添加餐厅</div>
    <div class="mfield"><label>餐厅名称</label><input id="f-name" placeholder="例如：外婆家·江南私房菜"></div>
    <div class="mfield"><div class="row"><div><label>封面 Emoji</label><input id="f-cover" placeholder="🍜" maxlength="4"></div><div><label>分类</label><input id="f-cat" placeholder="例如：川菜"></div></div></div>
    <div class="mfield"><div class="row"><div><label>评分 (1-5)</label><input id="f-rat" type="number" min="1" max="5" step="0.1" placeholder="4.5"></div><div><label>人均消费 (¥)</label><input id="f-price" type="number" min="1" placeholder="88"></div></div></div>
    <div class="mfield"><label>地址</label><input id="f-addr" placeholder="城市 + 详细地址"></div>
    <div class="mfield"><div class="row"><div><label>营业时间</label><input id="f-hours" placeholder="10:00-22:00"></div><div><label>联系电话</label><input id="f-phone" placeholder="021-12345678"></div></div></div>
    <div class="mfield"><label>标签（回车添加）</label><div class="tag-input" id="tag-input" onclick="document.getElementById('tag-field').focus()"><div id="tag-chips"></div><input id="tag-field" class="tag-input-field" placeholder="输入标签按回车..." onkeydown="addTag(event)"></div></div>
    <div class="modal-foot"><button class="btn-cancel" onclick="closeModal()">取消</button><button class="btn-save" onclick="saveM()">保存</button></div>
  </div>
</div>

<script>
const UI={{ user_info|tojson }};
let MS={{ merchants|tojson }};
let AGG={{ agg_data|tojson }};
const SC=['neg','neu','pos'],SL=['负向','中立','正向'],SE=['😞','😐','😊'];
const PC=['#D94F3D','#E07830','#2E8B57'];
const TITLES={'dash':'数据看板','merchants':'餐厅管理','reviews':'评论管理','sentiment':'情感分析','users':'用户管理','db':'数据库状态','bert':'BERT 模型'};

document.getElementById('sbAv').textContent=UI.avatar;
document.getElementById('sbNm').textContent=UI.name;

function sp(n){
  ['dash','merchants','reviews','sentiment','users','db','bert'].forEach(x=>{
    document.getElementById('pg-'+x).classList.toggle('active',x===n);
    document.getElementById('sn-'+x).classList.toggle('active',x===n);
  });
  document.getElementById('topbar-title').textContent=TITLES[n];
  if(n==='dash')loadDash();
  if(n==='merchants')loadMcTbl();
  if(n==='reviews')loadRvs();
  if(n==='users')loadUsers();
  if(n==='db')loadDbPage();
  if(n==='bert')loadBertPage();
}

/* ─── Dashboard ─── */
function loadDash(){
  const totalRvs=MS.reduce((s,m)=>s+m.reviews.length,0);
  const anaRvs=MS.reduce((s,m)=>s+m.reviews.filter(r=>r._analysis).length,0);
  const pendRvs=MS.reduce((s,m)=>s+m.reviews.filter(r=>r.status==='pending').length,0);
  document.getElementById('s-mc').textContent=MS.length;
  document.getElementById('s-rv').textContent=totalRvs;
  document.getElementById('s-ana').textContent=anaRvs;
  document.getElementById('s-pend').textContent=pendRvs;
  if(pendRvs>0){const el=document.getElementById('pend-cnt');el.textContent=pendRvs;el.style.display='';}
  // Bar charts
  ['taste','env','service'].forEach(dk=>{
    const el=document.getElementById('chart-'+dk);
    el.innerHTML=MS.map(m=>{
      const ag=AGG[String(m.id)];const s=ag&&ag[dk];
      if(!s){return`<div class="bar-row"><div class="bar-label" style="font-size:9px">${m.name.slice(0,5)}</div><div class="bar-track"><div class="bar-fill" style="width:10%;background:var(--bd);color:var(--i3)">无数据</div></div><div class="bar-count">—</div></div>`;}
      const pct=s.pct[s.label];const clr=PC[s.label];
      return`<div class="bar-row"><div class="bar-label" style="font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${m.name.slice(0,5)}</div><div class="bar-track"><div class="bar-fill" style="width:${Math.max(pct,15)}%;background:${clr}">${SL[s.label]}</div></div><div class="bar-count">${pct}%</div></div>`;
    }).join('');
  });
  // Donut
  let tots=[0,0,0];
  MS.forEach(m=>{['taste','env','service'].forEach(dk=>{const ag=AGG[String(m.id)];if(ag&&ag[dk]){ag[dk].pct.forEach((p,i)=>tots[i]+=p);}});});
  const total=tots.reduce((a,b)=>a+b,0)||1;
  const pcts=tots.map(v=>Math.round(v/total*100));
  const cv=document.getElementById('donut');const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,100,100);
  let ang=-Math.PI/2;
  pcts.forEach((p,i)=>{const slice=p/100*Math.PI*2;ctx.beginPath();ctx.moveTo(50,50);ctx.arc(50,50,40,ang,ang+slice);ctx.closePath();ctx.fillStyle=PC[i];ctx.fill();ang+=slice;});
  ctx.beginPath();ctx.arc(50,50,22,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();
  document.getElementById('donut-leg').innerHTML=SL.map((l,i)=>`<div class="dl-item"><div class="dl-dot" style="background:${PC[i]}"></div>${l} ${pcts[i]}%</div>`).join('');
}

/* ─── Merchant table ─── */
function loadMcTbl(){
  document.getElementById('mc-tbody').innerHTML=MS.map(m=>{
    const ag=AGG[String(m.id)];
    const dim=(dk)=>{const s=ag&&ag[dk];return s?`<span class="ab2 ${s.cls}">${s.emoji}${s.text}</span>`:`<span class="ab2 none">—</span>`;};
    return`<tr>
      <td><div class="mc-cell"><div class="mc-icon">${m.cover}</div><div><div class="mc-cname">${m.name}</div><div class="mc-cat">${m.address.slice(0,12)}…</div></div></div></td>
      <td>${m.category}</td>
      <td><span style="color:var(--gold);font-weight:600">★ ${m.rating}</span></td>
      <td>¥${m.price}</td>
      <td>${m.reviews.filter(r=>r.status==='approved').length}</td>
      <td>${dim('taste')}</td><td>${dim('env')}</td><td>${dim('service')}</td>
      <td><div class="act-btns">
        <button class="btn-sm btn-edit" onclick="openEdit(${m.id})">编辑</button>
        <button class="btn-sm btn-del" onclick="delM(${m.id})">删除</button>
      </div></td>
    </tr>`;
  }).join('');
}

/* ─── Review management ─── */
function loadRvs(){
  const filterMid=parseInt(document.getElementById('rv-fmc').value)||0;
  const filterSt=document.getElementById('rv-fst').value;
  // populate restaurant filter
  const sel=document.getElementById('rv-fmc');
  if(sel.options.length<=1)MS.forEach(m=>{const o=document.createElement('option');o.value=m.id;o.textContent=m.name;sel.appendChild(o);});
  let rows=[];
  MS.forEach(m=>{if(filterMid&&m.id!==filterMid)return;
    m.reviews.forEach((r,i)=>{if(filterSt&&r.status!==filterSt)return;rows.push({m,r,i});});
  });
  rows.sort((a,b)=>b.r.date.localeCompare(a.r.date));
  if(!rows.length){document.getElementById('rv-list').innerHTML='<div class="empty"><div class="ei">💬</div><p>没有符合条件的评论</p></div>';return;}
  document.getElementById('rv-list').innerHTML=rows.map(({m,r,i})=>{
    const sb=r.status==='approved'?'s-approved':r.status==='pending'?'s-pending':'s-rejected';
    const agg=r._analysis?`<div class="rv-agg">`+['taste','env','service'].map((dk,di)=>{const d=r._analysis.dimensions[dk];return`<span class="ab2 ${SC[d.label]}">${['口味','环境','服务'][di]} ${d.emoji}${d.sentiment}</span>`;}).join('')+'</div>':'';
    const acts=r.status!=='approved'?`<button class="btn-approve" onclick="setRvSt(${m.id},${i},'approved')">✓ 通过</button>`:'';
    const rej=r.status!=='rejected'?`<button class="btn-reject" onclick="setRvSt(${m.id},${i},'rejected')">✕ 拒绝</button>`:'';
    return`<div class="rv-row">
      <div class="rv-meta">
        <div class="rv-who"><div class="rv-av2">${r.avatar}</div><div><div style="font-size:12px;font-weight:500">${r.user}</div><div class="rv-mname">📍 ${m.name} · ${r.date}</div></div></div>
        <div style="display:flex;align-items:center;gap:7px"><div class="stars2">${'★'.repeat(r.stars)}${'☆'.repeat(5-r.stars)}</div><span class="status-badge ${sb}">${r.status==='approved'?'已通过':r.status==='pending'?'待审核':'已拒绝'}</span></div>
      </div>
      <div class="rv-txt">${r.text}</div>
      ${agg}
      <div class="rv-acts">${acts}${rej}${r.status==='approved'&&!r._analysis?`<button class="btn-sm btn-edit" onclick="anaAdminRv(${m.id},${i})">🧠 分析</button>`:''}</div>
    </div>`;
  }).join('');
}

function setRvSt(mid,idx,st){
  const m=MS.find(x=>x.id===mid);if(!m)return;
  m.reviews[idx].status=st;
  fetch('/api/admin/review_status',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mid,idx,status:st})}).then(r=>r.json()).then(d=>{if(d.agg)AGG[String(mid)]=d.agg;loadRvs();loadDash();});
}

function anaAdminRv(mid,idx){
  const m=MS.find(x=>x.id===mid);const text=m.reviews[idx].text;
  predict(text).then(d=>{m.reviews[idx]._analysis=d;fetch('/api/add_agg/'+mid,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(r=>r.json()).then(x=>{AGG[String(mid)]=x.agg;loadRvs();loadDash();});});
}

/* ─── Sentiment page ─── */
function doSA(){
  const txt=document.getElementById('sa-txt').value.trim();if(!txt){alert('请输入文本');return;}
  const btn=document.getElementById('sabtn');btn.disabled=true;btn.innerHTML='<span class="spin"></span>分析中…';
  predict(txt).then(d=>{
    document.getElementById('sa-dims').innerHTML=['taste','env','service'].map((dk,i)=>{const info=d.dimensions[dk],c=SC[info.label];const bars=[info.probs['负向'],info.probs['中立'],info.probs['正向']].map((p,j)=>`<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:var(--i3);margin-top:4px"><span style="width:22px">${SL[j]}</span><div style="flex:1;height:4px;background:rgba(0,0,0,.07);border-radius:3px;overflow:hidden"><div style="width:${Math.round(p*100)}%;height:100%;background:${PC[j]};border-radius:3px"></div></div><span style="width:26px;text-align:right">${Math.round(p*100)}%</span></div>`).join('');
      return`<div style="padding:12px;border-radius:10px;text-align:center;border:1.5px solid;${c==='pos'?'border-color:#a8d8bb;background:var(--gl)':c==='neu'?'border-color:#f0d79a;background:var(--ol)':'border-color:#f5b8b3;background:var(--rl)'}"><div style="font-size:10px;color:var(--i3);margin-bottom:3px">${['口味','环境','服务'][i]}</div><div style="font-size:22px">${info.emoji}</div><div style="font-size:12px;font-weight:700;margin-top:2px;color:${c==='pos'?'var(--green)':c==='neu'?'var(--orange)':'var(--red)'}">${info.sentiment}</div><div style="font-size:10px;color:var(--i3)">${(info.confidence*100).toFixed(1)}%</div>${bars}</div>`;
    }).join('');
    document.getElementById('sa-res').style.display='block';
  }).catch(e=>alert(e.message)).finally(()=>{btn.disabled=false;btn.innerHTML='分析情感';});
}

/* ─── Users ─── */
function loadUsers(){
  const users=[{id:'admin',name:'管理员',avatar:'👨‍💼',role:'admin'},{id:'alice',name:'爱丽丝',avatar:'👩',role:'user'},{id:'bob',name:'小明',avatar:'👦',role:'user'}];
  document.getElementById('u-tbody').innerHTML=users.map(u=>{
    const cnt=MS.reduce((s,m)=>s+m.reviews.filter(r=>r.user===u.name).length,0);
    return`<tr><td><div style="display:flex;align-items:center;gap:8px"><div style="width:28px;height:28px;border-radius:50%;background:var(--paper);display:flex;align-items:center;justify-content:center;font-size:16px">${u.avatar}</div><span style="font-weight:500">${u.name}</span></div></td><td style="color:var(--i3)">${u.id}</td><td><span style="padding:2px 8px;border-radius:20px;font-size:10px;font-weight:500;${u.role==='admin'?'background:#fff8e7;border:1px solid rgba(201,148,58,.4);color:var(--gold)':'background:var(--gl);border:1px solid #a8d8bb;color:var(--green)'}">${u.role==='admin'?'管理员':'普通用户'}</span></td><td>${cnt}</td></tr>`;
  }).join('');
}

/* ─── Add/Edit Modal ─── */
let editId=null,editTags=[];
function openAdd(){editId=null;editTags=[];document.getElementById('modal-title').textContent='添加餐厅';['name','cover','cat','addr','hours','phone'].forEach(f=>document.getElementById('f-'+f).value='');document.getElementById('f-rat').value='4.5';document.getElementById('f-price').value='';renderTags();document.getElementById('modal').classList.add('open');}
function openEdit(id){
  const m=MS.find(x=>x.id===id);if(!m)return;editId=id;editTags=[...m.tags];
  document.getElementById('modal-title').textContent='编辑餐厅';
  document.getElementById('f-name').value=m.name;document.getElementById('f-cover').value=m.cover;
  document.getElementById('f-cat').value=m.category;document.getElementById('f-rat').value=m.rating;
  document.getElementById('f-price').value=m.price;document.getElementById('f-addr').value=m.address;
  document.getElementById('f-hours').value=m.open_hours||'';document.getElementById('f-phone').value=m.phone||'';
  renderTags();document.getElementById('modal').classList.add('open');
}
function closeModal(){document.getElementById('modal').classList.remove('open');}
function renderTags(){
  document.getElementById('tag-chips').innerHTML=editTags.map((t,i)=>`<span class="tag-chip">${t}<span class="tag-chip-del" onclick="rmTag(${i})">×</span></span>`).join('');
}
function addTag(e){if(e.key!=='Enter'&&e.key!==',')return;e.preventDefault();const v=e.target.value.trim();if(v&&!editTags.includes(v)){editTags.push(v);renderTags();}e.target.value='';}
function rmTag(i){editTags.splice(i,1);renderTags();}
function saveM(){
  const data={name:document.getElementById('f-name').value.trim(),cover:document.getElementById('f-cover').value.trim()||'🍽️',category:document.getElementById('f-cat').value.trim(),rating:parseFloat(document.getElementById('f-rat').value)||4.5,price:parseInt(document.getElementById('f-price').value)||88,address:document.getElementById('f-addr').value.trim(),open_hours:document.getElementById('f-hours').value.trim(),phone:document.getElementById('f-phone').value.trim(),tags:editTags};
  if(!data.name){alert('请填写餐厅名称');return;}
  const url=editId?'/api/admin/merchant/'+editId:'/api/admin/merchant';
  const method=editId?'PUT':'POST';
  fetch(url,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(r=>r.json()).then(d=>{
    if(d.ok){if(editId){const i=MS.findIndex(x=>x.id===editId);if(i>=0)MS[i]={...MS[i],...data};}else{MS.push(d.merchant);AGG[d.merchant.id]=null;}loadMcTbl();closeModal();}else alert(d.error||'保存失败');
  });
}
function delM(id){
  if(!confirm('确定删除该餐厅及其所有评论？'))return;
  fetch('/api/admin/merchant/'+id,{method:'DELETE'}).then(r=>r.json()).then(d=>{if(d.ok){MS=MS.filter(x=>x.id!==id);delete AGG[String(id)];loadMcTbl();loadDash();}else alert(d.error||'删除失败');});
}

/* ─── Shared ─── */
function predict(t){return fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})}).then(r=>r.json().then(d=>{if(!r.ok)throw new Error(d.error||'失败');return d;}));}

/* ─── DB Page ─── */
function loadDbPage(){
  fetch('/api/admin/db_stats').then(r=>r.json()).then(d=>{
    document.getElementById('db-file').textContent=d.file_size;
    document.getElementById('db-users').textContent=d.users;
    document.getElementById('db-merchants').textContent=d.merchants;
    document.getElementById('db-reviews').textContent=d.reviews;
    document.getElementById('db-logs').textContent=d.logs;
    document.getElementById('log-tbody').innerHTML=(d.recent_logs||[]).map(l=>`<tr><td style="color:var(--i3)">#${l.id}</td><td>${l.username}</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${l.text}">${l.text}</td><td>${l.taste}</td><td>${l.env}</td><td>${l.service}</td><td style="color:var(--i3);white-space:nowrap">${l.time}</td></tr>`).join('')||'<tr><td colspan="7" style="text-align:center;color:var(--i3);padding:16px">暂无记录</td></tr>';
    document.getElementById('agg-tbody').innerHTML=(d.agg_rows||[]).map(a=>{const c=(v)=>`<td style="text-align:center;color:var(--i3)">${v}</td>`;return`<tr><td style="font-weight:500">${a.name}</td>${c(a.taste_neg)}${c(a.taste_neu)}${c(a.taste_pos)}${c(a.env_neg)}${c(a.env_neu)}${c(a.env_pos)}${c(a.service_neg)}${c(a.service_neu)}${c(a.service_pos)}<td style="color:var(--i3);font-size:10px;white-space:nowrap">${a.updated_at}</td></tr>`;}).join('');
  });
}

/* ─── BERT Page ─── */
let _bertChart=null;
function loadBertPage(){
  const history={{ training_history|tojson }};
  if(!history.length)return;
  const cv=document.getElementById('train-chart');if(!cv)return;
  function drawChart(){
    const W=cv.parentElement.offsetWidth-32||600,H=100;cv.width=W;cv.height=H;
    const ctx=cv.getContext('2d');ctx.clearRect(0,0,W,H);
    const pad={l:38,r:16,t:12,b:26};const cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
    ctx.strokeStyle='#E8DDD2';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+ch);ctx.lineTo(pad.l+cw,pad.t+ch);ctx.stroke();
    const n=history.length;
    function line(vals,color){
      const mx=Math.max(...vals),mn=Math.min(...vals),rng=mx-mn||1;
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
      vals.forEach((v,i)=>{const x=pad.l+i/(n-1)*cw,y=pad.t+ch-(v-mn)/rng*ch;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
      ctx.stroke();
      vals.forEach((v,i)=>{const x=pad.l+i/(n-1)*cw,y=pad.t+ch-(v-mn)/rng*ch;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill();});
    }
    line(history.map(h=>h.loss),'#D94F3D');
    line(history.map(h=>h.val_f1*3),'#2E8B57');
    ctx.fillStyle='#8A7265';ctx.font='9px sans-serif';ctx.textAlign='center';
    history.forEach((h,i)=>{ctx.fillText('E'+h.epoch,pad.l+i/(n-1)*cw,H-6);});
    ctx.fillStyle='#D94F3D';ctx.fillRect(pad.l,2,12,3);ctx.fillStyle='#1A1008';ctx.font='10px sans-serif';ctx.textAlign='left';ctx.fillText('Loss',pad.l+16,8);
    ctx.fillStyle='#2E8B57';ctx.fillRect(pad.l+56,2,12,3);ctx.fillText('val F1 (×3)',pad.l+72,8);
  }
  drawChart();_bertChart=drawChart;
  window.addEventListener('resize',drawChart);
}
function doBertTest(){
  const txt=document.getElementById('bert-test-txt').value.trim();if(!txt){alert('请输入文本');return;}
  const btn=document.getElementById('bert-test-btn');btn.disabled=true;btn.innerHTML='<span class="spin"></span>';
  predict(txt).then(d=>{
    document.getElementById('bert-test-dims').innerHTML=['taste','env','service'].map((dk,i)=>{
      const info=d.dimensions[dk],c=SC[info.label];
      const bars=[info.probs['负向'],info.probs['中立'],info.probs['正向']].map((p,j)=>`<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:var(--i3);margin-top:3px"><span style="width:22px">${SL[j]}</span><div style="flex:1;height:4px;background:rgba(0,0,0,.07);border-radius:3px;overflow:hidden"><div style="width:${Math.round(p*100)}%;height:100%;background:${PC[j]};border-radius:3px"></div></div><span style="width:28px;text-align:right">${Math.round(p*100)}%</span></div>`).join('');
      return`<div style="padding:11px;border-radius:10px;border:1.5px solid;${c==='pos'?'border-color:#a8d8bb;background:var(--gl)':c==='neu'?'border-color:#f0d79a;background:var(--ol)':'border-color:#f5b8b3;background:var(--rl)'}"><div style="font-size:10px;color:var(--i3);margin-bottom:3px">${['口味','环境','服务'][i]}</div><div style="font-size:20px">${info.emoji}</div><div style="font-size:12px;font-weight:700;color:${c==='pos'?'var(--green)':c==='neu'?'var(--orange)':'var(--red)'}">${info.sentiment}</div><div style="font-size:10px;color:var(--i3)">置信度 ${(info.confidence*100).toFixed(1)}%</div>${bars}</div>`;
    }).join('');
    const raw=['--- 模型原始输出 ---',...['taste','env','service'].map((dk,i)=>{const info=d.dimensions[dk];return`[${['口味','环境','服务'][i]}]  负向=${info.probs['负向'].toFixed(4)}  中立=${info.probs['中立'].toFixed(4)}  正向=${info.probs['正向'].toFixed(4)}  → ${info.sentiment}(${(info.confidence*100).toFixed(1)}%)`;})];
    document.getElementById('bert-raw-out').textContent=raw.join('\n');
    document.getElementById('bert-test-res').style.display='block';
  }).catch(e=>alert(e.message)).finally(()=>{btn.disabled=false;btn.innerHTML='推理';});
}

function logout(){fetch('/api/logout',{method:'POST'}).then(()=>location.href='/login');}

// init
loadDash();
</script></body></html>"""

# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════

@app.route('/login')
def login_page():
    if 'username' in session:
        return redirect(url_for('index'))
    return render_template_string(LOGIN_HTML)

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(force=True)
    username = data.get('username', '').strip()
    u = User.query.filter_by(username=username).first()
    if u and u.password == data.get('password', ''):
        session['username'] = username
        redirect_to = '/admin' if u.role == 'admin' else '/'
        return jsonify({'ok': True, 'redirect': redirect_to})
    return jsonify({'ok': False, 'error': '账号或密码错误'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'ok': True})

@app.route('/')
@login_required
def index():
    u = User.query.filter_by(username=session['username']).first()
    if u.role == 'admin':
        return redirect(url_for('admin_page'))
    return render_template_string(
        USER_HTML,
        user_info={'name': u.name, 'avatar': u.avatar},
        merchants=merchants_for_frontend(),
        agg_data=all_agg_dict(),
    )

@app.route('/admin')
@login_required
def admin_page():
    u = User.query.filter_by(username=session['username']).first()
    if u.role != 'admin':
        return redirect(url_for('index'))
    # 读取训练历史（results.json）
    training_history = []
    results_path = os.path.join(BASE_DIR, 'checkpoints', 'results.json')
    if os.path.exists(results_path):
        try:
            with open(results_path, encoding='utf-8') as f:
                training_history = json.load(f).get('history', [])
        except Exception:
            pass
    return render_template_string(
        ADMIN_HTML,
        user_info={'name': u.name, 'avatar': u.avatar},
        merchants=merchants_for_frontend(),
        agg_data=all_agg_dict(),
        training_history=training_history,
    )

# ── 情感预测（核心接口）────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@api_auth
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请提供 text 字段'}), 400

    result = get_predictor().predict_one(text)

    # 写入分析日志
    log = AnalysisLog(
        username=session['username'],
        text=text,
        result=json.dumps(result, ensure_ascii=False),
    )
    db.session.add(log)
    db.session.commit()

    return jsonify(result)

# ── 单条评论分析后更新聚合 ──────────────────────────────────────
@app.route('/api/add_agg/<int:mid>', methods=['POST'])
@api_auth
def add_agg(mid):
    if not Merchant.query.get(mid):
        return jsonify({'error': 'not found'}), 404

    agg = get_or_create_agg(mid)
    agg.add_result(request.get_json(force=True))
    db.session.commit()
    return jsonify({'agg': agg.to_summary()})

# ── 用户提交新评论 ─────────────────────────────────────────────
@app.route('/api/submit_review/<int:mid>', methods=['POST'])
@api_auth
def submit_review(mid):
    m = Merchant.query.get(mid)
    if not m:
        return jsonify({'error': 'not found'}), 404

    data     = request.get_json(force=True)
    rv_data  = data.get('review', {})
    analysis = data.get('analysis', {})

    u = User.query.filter_by(username=session['username']).first()
    rv = Review(
        merchant_id=mid,
        user_name=u.name,
        user_avatar=u.avatar,
        stars=rv_data.get('stars', 3),
        text=rv_data.get('text', ''),
        status='approved',
    )
    if analysis:
        rv.set_analysis(analysis)

    db.session.add(rv)

    # 更新聚合
    if analysis:
        agg = get_or_create_agg(mid)
        agg.add_result(analysis)
        # 同时写分析日志
        log = AnalysisLog(
            username=session['username'],
            text=rv_data.get('text', ''),
            result=json.dumps(analysis, ensure_ascii=False),
        )
        db.session.add(log)

    db.session.commit()
    return jsonify({'ok': True, 'agg': build_agg_dict(mid)})

# ── 分析历史 ───────────────────────────────────────────────────
@app.route('/api/history')
@api_auth
def history():
    logs = AnalysisLog.query.order_by(AnalysisLog.created_at.desc()).limit(100).all()
    return jsonify([l.to_dict() for l in logs])

# ══════════════════════════════════════════════════════════════
# Admin APIs
# ══════════════════════════════════════════════════════════════

def _get_or_create_tag(name: str) -> Tag:
    t = Tag.query.filter_by(name=name).first()
    if not t:
        t = Tag(name=name)
        db.session.add(t)
    return t

@app.route('/api/admin/merchant', methods=['POST'])
@api_admin
def admin_add_merchant():
    data = request.get_json(force=True)
    tag_objs = [_get_or_create_tag(tn) for tn in data.get('tags', [])]
    m = Merchant(
        name=data.get('name', '新餐厅'),
        category=data.get('category', '其他'),
        rating=float(data.get('rating', 4.5)),
        price=int(data.get('price', 88)),
        cover=data.get('cover', '🍽️'),
        address=data.get('address', ''),
        open_hours=data.get('open_hours', ''),
        phone=data.get('phone', ''),
        tags=tag_objs,
    )
    db.session.add(m)
    db.session.flush()
    db.session.add(SentAgg(merchant_id=m.id))
    db.session.commit()
    return jsonify({'ok': True, 'merchant': m.to_dict()})

@app.route('/api/admin/merchant/<int:mid>', methods=['PUT'])
@api_admin
def admin_edit_merchant(mid):
    m = Merchant.query.get(mid)
    if not m:
        return jsonify({'error': 'not found'}), 404
    data = request.get_json(force=True)
    for k in ('name', 'category', 'rating', 'price', 'cover', 'address', 'open_hours', 'phone'):
        if k in data:
            setattr(m, k, data[k])
    if 'tags' in data:
        m.tags = [_get_or_create_tag(tn) for tn in data['tags']]
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/admin/merchant/<int:mid>', methods=['DELETE'])
@api_admin
def admin_del_merchant(mid):
    m = Merchant.query.get(mid)
    if not m:
        return jsonify({'error': 'not found'}), 404
    db.session.delete(m)   # cascade 自动删除 reviews + sent_agg
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/admin/review_status', methods=['POST'])
@api_admin
def admin_review_status():
    data = request.get_json(force=True)
    rv = Review.query.get(data.get('idx'))     # idx 现在是 Review.id
    if not rv:
        return jsonify({'error': 'not found'}), 404
    rv.status = data['status']
    db.session.flush()
    # 重算该餐厅的聚合
    agg = get_or_create_agg(rv.merchant_id)
    agg.recompute_from_reviews()
    db.session.commit()
    return jsonify({'ok': True, 'agg': agg.to_summary()})

# ── 管理员：查看所有日志 ────────────────────────────────────────
@app.route('/api/admin/logs')
@api_admin
def admin_logs():
    logs = AnalysisLog.query.order_by(AnalysisLog.created_at.desc()).limit(200).all()
    return jsonify([{
        'id':       l.id,
        'username': l.username,
        'text':     l.text[:60] + ('…' if len(l.text) > 60 else ''),
        'time':     l.created_at.strftime('%m-%d %H:%M'),
    } for l in logs])

@app.route('/api/admin/db_stats')
@api_admin
def admin_db_stats():
    """数据库状态 API：表行数 + 文件大小 + 最近日志 + 聚合缓存明细"""
    # 文件大小
    db_path = os.path.join(BASE_DIR, 'sentiment.db')
    size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    if size_bytes < 1024:
        file_size = f'{size_bytes} B'
    elif size_bytes < 1024 ** 2:
        file_size = f'{size_bytes/1024:.1f} KB'
    else:
        file_size = f'{size_bytes/1024**2:.2f} MB'

    # 表行数
    from models import AnalysisLog, SentAgg
    users_cnt     = User.query.count()
    merchants_cnt = Merchant.query.count()
    reviews_cnt   = Review.query.count()
    logs_cnt      = AnalysisLog.query.count()

    # 最近 20 条日志（含各维度结果）
    SCLS = ['neg','neu','pos']; STXT = ['负向','中立','正向']; SEMJ = ['😞','😐','😊']
    recent_logs = []
    for log in AnalysisLog.query.order_by(AnalysisLog.created_at.desc()).limit(20).all():
        r = log.get_result()
        dims = r.get('dimensions', {})
        def fmt(dk):
            d = dims.get(dk, {}); lbl = d.get('label', -1)
            if lbl < 0: return '—'
            return f'<span style="color:{["#D94F3D","#E07830","#2E8B57"][lbl]}">{SEMJ[lbl]}{STXT[lbl]}</span>'
        recent_logs.append({
            'id':       log.id,
            'username': log.username,
            'text':     log.text[:50] + ('…' if len(log.text) > 50 else ''),
            'taste':    fmt('taste'),
            'env':      fmt('env'),
            'service':  fmt('service'),
            'time':     log.created_at.strftime('%m-%d %H:%M'),
        })

    # sent_agg 明细
    agg_rows = []
    for agg in SentAgg.query.all():
        m = Merchant.query.get(agg.merchant_id)
        agg_rows.append({
            'name':        m.name if m else f'#{agg.merchant_id}',
            'taste_neg':   agg.taste_neg,  'taste_neu': agg.taste_neu,  'taste_pos': agg.taste_pos,
            'env_neg':     agg.env_neg,    'env_neu':   agg.env_neu,    'env_pos':   agg.env_pos,
            'service_neg': agg.service_neg,'service_neu':agg.service_neu,'service_pos':agg.service_pos,
            'updated_at':  agg.updated_at.strftime('%m-%d %H:%M') if agg.updated_at else '—',
        })

    return jsonify({
        'file_size':   file_size,
        'users':       users_cnt,
        'merchants':   merchants_cnt,
        'reviews':     reviews_cnt,
        'logs':        logs_cnt,
        'recent_logs': recent_logs,
        'agg_rows':    agg_rows,
    })


def health():
    mc  = Merchant.query.count()
    rv  = Review.query.count()
    log = AnalysisLog.query.count()
    return jsonify({'status': 'ok', 'merchants': mc, 'reviews': rv, 'logs': log})

# ══════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    with app.app_context():
        init_db()   # 建表 + 种子数据（幂等，可重复运行）

    print('=' * 60)
    print('  饮食情报 · 餐饮评论情感分析平台（SQLite 版）')
    print('  用户界面:   http://localhost:5001/')
    print('  管理后台:   http://localhost:5001/admin')
    print('  数据库文件: sentiment.db')
    print('  账号: admin(管理员) / alice / bob   密码: 123456')
    print('=' * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)