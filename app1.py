import os, sys, io, json
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from functools import wraps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import SentimentPredictor

app = Flask(__name__)
app.secret_key = 'sentiment-app-secret-2024'

# ── Mock Users ──────────────────────────────────────────────
USERS = {
    'admin':  {'password': '123456', 'name': '管理员', 'avatar': '👨‍💼'},
    'alice':  {'password': '123456', 'name': '爱丽丝', 'avatar': '👩'},
    'bob':    {'password': '123456', 'name': '小明',   'avatar': '👦'},
}

# ── Mock Merchants ───────────────────────────────────────────
MERCHANTS = [
    {
        'id': 1,
        'name': '外婆家·江南私房菜',
        'category': '江浙菜',
        'rating': 4.7,
        'price': 88,
        'address': '上海市黄浦区南京东路168号',
        'cover': '🏮',
        'tags': ['老字号', '人均88', '必吃榜'],
        'reviews': [
            {'user': '美食达人_小红', 'avatar': '👩‍🦰', 'stars': 5, 'date': '2024-12-01', 'text': '外婆家真的名不虚传！红烧肉肥而不腻，入口即化，汤汁浓郁。环境布置得很有江南风情，木质桌椅，窗帘轻盈，坐在里面感觉很舒适。服务员小姐姐很热情，主动介绍菜品，上菜速度也很快。'},
            {'user': '吃货本货', 'avatar': '🧔', 'stars': 4, 'date': '2024-11-28', 'text': '龙井虾仁和清蒸鲈鱼都是招牌，味道确实不错，食材新鲜。环境稍微有点吵，因为人很多。但总体来说性价比很高，推荐！'},
            {'user': '周末食客', 'avatar': '👩', 'stars': 3, 'date': '2024-11-20', 'text': '菜的味道还可以，但是等位等了将近一个小时，服务员态度比较冷漠，叫了好几次才来。价格中规中矩，不算便宜。'},
            {'user': 'gourmet_li', 'avatar': '👨', 'stars': 5, 'date': '2024-11-15', 'text': '每次来上海都要来这里打卡！腌笃鲜鲜美无比，笋香四溢。包厢环境很好，适合商务宴请，服务也非常到位，给满分！'},
            {'user': '素食主义者', 'avatar': '🧑', 'stars': 2, 'date': '2024-11-10', 'text': '作为素食者，选择非常有限，只有几道蔬菜可以吃。上菜很慢，等了半天才上齐。不太推荐素食者来，荤菜爱好者应该会喜欢。'},
        ]
    },
    {
        'id': 2,
        'name': '海底捞火锅（旗舰店）',
        'category': '火锅',
        'rating': 4.9,
        'price': 120,
        'address': '北京市朝阳区三里屯太古里南区B1层',
        'cover': '🫕',
        'tags': ['服务满分', '24小时', '生日惊喜'],
        'reviews': [
            {'user': '火锅控', 'avatar': '🧑‍🍳', 'stars': 5, 'date': '2024-12-03', 'text': '海底捞的服务真的没话说！等位的时候有零食和美甲，进去之后服务员全程笑脸相迎。锅底选了番茄锅，酸甜适口，食材新鲜。唯一遗憾是价格偏贵，但物有所值。'},
            {'user': 'Lucy张', 'avatar': '👩‍🦱', 'stars': 5, 'date': '2024-12-01', 'text': '过生日来的，服务员给唱了生日歌还送了长寿面，感动到哭！食材超级新鲜，虾滑弹弹的，毛肚脆脆的。环境宽敞明亮，强烈推荐！'},
            {'user': '理性消费者', 'avatar': '👨‍💻', 'stars': 3, 'date': '2024-11-25', 'text': '味道是不错的，但是价格真的有点高，人均下来要150左右。环境还好，服务确实很好，但感觉有点过于热情，有些不自在。'},
            {'user': '吃辣小能手', 'avatar': '👧', 'stars': 5, 'date': '2024-11-22', 'text': '辣锅底太带劲了！配上鸭肠和黄喉，完美！服务员一直帮忙涮菜，非常贴心。店内环境干净整洁，没有异味，赞一个！'},
            {'user': '慢吞吞先生', 'avatar': '🧓', 'stars': 4, 'date': '2024-11-18', 'text': '食材新鲜度在连锁火锅里算顶尖的了。就是高峰期需要等位，大概等了40分钟，建议提前预约。'},
        ]
    },
    {
        'id': 3,
        'name': '和牛職人·日式烤肉',
        'category': '日式料理',
        'rating': 4.5,
        'price': 380,
        'address': '深圳市南山区海岸城购物中心L4',
        'cover': '🥩',
        'tags': ['和牛', '高端', '约会首选'],
        'reviews': [
            {'user': '牛肉鉴赏家', 'avatar': '🧑‍🍽️', 'stars': 5, 'date': '2024-12-02', 'text': 'A5和牛雪花纹理清晰，在炭火上烤出来油脂四溢，入口即化，简直是人间极品！环境非常有格调，灯光昏黄温馨，很适合约会。服务员懂行，会指导最佳烤法，专业度很高。'},
            {'user': '价格敏感用户', 'avatar': '😅', 'stars': 3, 'date': '2024-11-30', 'text': '味道确实很好，但是价格实在太贵了，两个人吃了将近800块。环境不错，服务也很好，就是心疼钱包。偶尔犒劳自己还是可以的。'},
            {'user': '日料爱好者小林', 'avatar': '👩‍🎓', 'stars': 5, 'date': '2024-11-26', 'text': '这里的和牛品质真的可以媲美日本本土了！厚切牛舌外焦里嫩，配上柠檬汁太绝了。空间私密，每张桌子都有隔断，服务周到不打扰，完美！'},
            {'user': '普通食客', 'avatar': '👨‍🦳', 'stars': 2, 'date': '2024-11-20', 'text': '肉的质量一般，感觉和标榜的A5有差距。服务态度还可以，但上菜速度太慢了。环境装修不错，但实际性价比很低，不会再来了。'},
            {'user': '商务宴请常客', 'avatar': '👔', 'stars': 4, 'date': '2024-11-15', 'text': '用来接待外地客户非常合适，档次够高，菜品质量稳定。私人包间很安静，服务也很专业。价格较高但在高端餐饮里算合理。'},
        ]
    },
    {
        'id': 4,
        'name': '喜茶·旗舰店',
        'category': '茶饮甜品',
        'rating': 4.3,
        'price': 35,
        'address': '广州市天河区珠江新城花城汇L1',
        'cover': '🧋',
        'tags': ['网红', '奶茶', '排队王'],
        'reviews': [
            {'user': '奶茶爱好者', 'avatar': '👩‍🦰', 'stars': 5, 'date': '2024-12-04', 'text': '多肉葡萄真的绝了！葡萄皮薄肉厚，茶底清爽，奶盖绵密，甜度刚好。店内环境很时尚，适合拍照打卡。'},
            {'user': '排队苦手', 'avatar': '😤', 'stars': 2, 'date': '2024-12-02', 'text': '味道是好的，但排队排了将近一个小时，服务员做单太慢，效率很低。而且店里面很吵，环境嘈杂，体验感很差。'},
            {'user': 'Cathy_饮品控', 'avatar': '👩', 'stars': 4, 'date': '2024-11-29', 'text': '芝芝莓莓味道清新，水果颗粒满满。点单后等了20分钟，还算正常。店员态度还不错，偶尔会推荐新品。'},
            {'user': '健康生活家', 'avatar': '🏃‍♀️', 'stars': 3, 'date': '2024-11-24', 'text': '饮品整体口感不错，但含糖量偏高，不适合经常喝。环境还行，但位置不够，节假日根本没地方坐。'},
            {'user': '甜品达人', 'avatar': '🧁', 'stars': 5, 'date': '2024-11-20', 'text': '限定款草莓系列太香了！每次出新品都会来打卡。店员服务很热情，会介绍各款产品的特点，推荐指数满分！'},
        ]
    },
    {
        'id': 5,
        'name': '老北京炸酱面馆',
        'category': '北京菜',
        'rating': 4.6,
        'price': 45,
        'address': '北京市西城区鼓楼大街88号',
        'cover': '🍜',
        'tags': ['老字号', '地道', '平价实惠'],
        'reviews': [
            {'user': '北京土著', 'avatar': '👴', 'stars': 5, 'date': '2024-12-03', 'text': '这才是正宗的北京炸酱面！面条劲道爽滑，炸酱浓郁香醇，配上黄瓜丝、豆芽、青豆，七八种小料拌在一起，这口感真是没话说。店里装修复古，有种老北京的市井气息。'},
            {'user': '外地游客小王', 'avatar': '🧳', 'stars': 4, 'date': '2024-11-30', 'text': '慕名而来，果然不负盛名！炸酱面分量很足，价格也实惠。店里环境有点拥挤，但服务很热情，老板很健谈，给我们普及了很多老北京饮食文化。'},
            {'user': '面食爱好者', 'avatar': '🍝', 'stars': 5, 'date': '2024-11-25', 'text': '手擀面非常好吃，面条韧劲十足。炸酱里有猪肉丁，肥瘦相间，咸香适口。店内虽小，但卫生干净，老板娘很亲切，像回家吃饭一样温暖。'},
            {'user': '匆匆过客', 'avatar': '🚶', 'stars': 3, 'date': '2024-11-18', 'text': '中规中矩，味道还可以但不算惊艳。高峰期服务有点跟不上，等了比较久。价格公道，是正常的快餐面馆水准。'},
            {'user': '美食博主Linda', 'avatar': '📸', 'stars': 5, 'date': '2024-11-10', 'text': '拍摄美食视频专门来打卡！面条手感和口感都是一流的，炸酱的配方应该有几十年历史了。环境虽然简朴，但充满了老北京的生活气息，服务也很接地气，强烈推荐！'},
        ]
    },
    {
        'id': 6,
        'name': '麻辣江湖·川菜私厨',
        'category': '川菜',
        'rating': 4.4,
        'price': 95,
        'address': '成都市锦江区春熙路附近',
        'cover': '🌶️',
        'tags': ['麻辣鲜香', '私厨', '网红店'],
        'reviews': [
            {'user': '辣椒小王子', 'avatar': '🌶️', 'stars': 5, 'date': '2024-12-01', 'text': '水煮鱼辣得过瘾，麻辣鲜香四味俱全！鱼肉嫩滑，花椒和辣椒的比例很完美。环境很有成都风情，竹编装饰，茶香弥漫。服务员很会聊天，推荐的菜都对我们口味。'},
            {'user': '不能吃辣的我', 'avatar': '😰', 'stars': 2, 'date': '2024-11-28', 'text': '可能是我太不能吃辣了，点了微辣还是辣得受不了。菜品种类少，环境还可以，但价格不便宜，对于不吃辣的人不太友好。'},
            {'user': '川菜资深食客', 'avatar': '🧑‍🍳', 'stars': 4, 'date': '2024-11-22', 'text': '口水鸡和夫妻肺片都做得很地道，调料配比很到位。就是上菜有点慢，等了将近20分钟。服务态度不错，多次询问是否需要加汤加料。'},
            {'user': '成都本地人', 'avatar': '👩‍🦱', 'stars': 5, 'date': '2024-11-17', 'text': '作为成都人，这家川菜还是很正宗的！麻婆豆腐嫩而入味，锅巴肉片酥脆可口。环境接地气，价格合理，老板也是个实在人，经常在店里招呼客人。'},
            {'user': '外省游客', 'avatar': '🗺️', 'stars': 4, 'date': '2024-11-12', 'text': '第一次来成都，专门来体验正宗川菜。麻辣味确实很正宗，吃得大汗淋漓但停不下来！环境有特色，服务热情，就是价格比想象中贵一点。'},
        ]
    },
]

history_records = []

# ── Lazy model loader ──────────────────────────────────────────
_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor(
            checkpoint_path=os.environ.get('MODEL_PATH', 'checkpoints/best_model.pt'),
            bert_model_name=os.environ.get('BERT_NAME', 'bert-base-chinese'),
        )
    return _predictor

# ── Auth helper ────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'error': '请先登录', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════
# HTML — Full SPA
# ══════════════════════════════════════════════════════════════
LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>饮食情报 · 登录</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --red: #D94F3D;
  --red-dark: #B03A2C;
  --red-light: #FDECEA;
  --gold: #C9943A;
  --ink: #1A1008;
  --ink-2: #4A3728;
  --ink-3: #8A7265;
  --cream: #FBF7F2;
  --paper: #F5EFE6;
  --border: #E8DDD2;
}

body {
  font-family: 'Noto Sans SC', sans-serif;
  background: var(--cream);
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

/* Decorative background */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse at 20% 50%, rgba(217,79,61,0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 80% 20%, rgba(201,148,58,0.07) 0%, transparent 50%),
    radial-gradient(ellipse at 60% 80%, rgba(217,79,61,0.05) 0%, transparent 40%);
  pointer-events: none;
}

.bg-pattern {
  position: fixed;
  inset: 0;
  opacity: 0.03;
  background-image: repeating-linear-gradient(
    45deg,
    var(--red) 0, var(--red) 1px,
    transparent 0, transparent 50%
  );
  background-size: 20px 20px;
  pointer-events: none;
}

.login-wrap {
  width: 100%;
  max-width: 420px;
  padding: 20px;
  animation: fadeUp .5s ease both;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

.brand {
  text-align: center;
  margin-bottom: 36px;
}

.brand-logo {
  font-size: 48px;
  display: block;
  margin-bottom: 8px;
  filter: drop-shadow(0 4px 12px rgba(217,79,61,0.3));
}

.brand-name {
  font-family: 'Noto Serif SC', serif;
  font-size: 28px;
  font-weight: 700;
  color: var(--red);
  letter-spacing: 6px;
}

.brand-sub {
  font-size: 12px;
  color: var(--ink-3);
  letter-spacing: 3px;
  margin-top: 4px;
}

.card {
  background: #fff;
  border-radius: 16px;
  padding: 36px;
  box-shadow: 0 8px 40px rgba(26,16,8,0.10), 0 2px 8px rgba(26,16,8,0.06);
  border: 1px solid var(--border);
}

.card-title {
  font-family: 'Noto Serif SC', serif;
  font-size: 18px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 24px;
  text-align: center;
}

.field {
  margin-bottom: 16px;
}

.field label {
  display: block;
  font-size: 12px;
  color: var(--ink-3);
  letter-spacing: 1px;
  margin-bottom: 6px;
  font-weight: 500;
}

.field input {
  width: 100%;
  padding: 12px 16px;
  border: 1.5px solid var(--border);
  border-radius: 10px;
  font-size: 14px;
  font-family: inherit;
  color: var(--ink);
  background: var(--paper);
  transition: border-color .2s, box-shadow .2s;
  outline: none;
}

.field input:focus {
  border-color: var(--red);
  box-shadow: 0 0 0 3px rgba(217,79,61,0.12);
  background: #fff;
}

.btn-login {
  width: 100%;
  padding: 14px;
  background: var(--red);
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 15px;
  font-family: 'Noto Serif SC', serif;
  font-weight: 600;
  letter-spacing: 4px;
  cursor: pointer;
  transition: background .2s, transform .1s, box-shadow .2s;
  margin-top: 8px;
  box-shadow: 0 4px 16px rgba(217,79,61,0.35);
}

.btn-login:hover {
  background: var(--red-dark);
  box-shadow: 0 6px 20px rgba(217,79,61,0.45);
}

.btn-login:active { transform: scale(.98); }

.demo-accounts {
  margin-top: 20px;
  padding: 14px;
  background: var(--red-light);
  border-radius: 10px;
  font-size: 12px;
  color: var(--ink-2);
}

.demo-accounts strong {
  display: block;
  color: var(--red);
  margin-bottom: 6px;
  font-size: 11px;
  letter-spacing: 1px;
}

.demo-item {
  display: flex;
  justify-content: space-between;
  margin: 4px 0;
  cursor: pointer;
  padding: 3px 6px;
  border-radius: 6px;
  transition: background .15s;
}

.demo-item:hover { background: rgba(217,79,61,0.1); }

.err-msg {
  color: var(--red);
  font-size: 12px;
  text-align: center;
  margin-top: 10px;
  display: none;
  padding: 8px;
  background: var(--red-light);
  border-radius: 8px;
}
</style>
</head>
<body>
<div class="bg-pattern"></div>

<div class="login-wrap">
  <div class="brand">
    <span class="brand-logo">🍽️</span>
    <div class="brand-name">饮食情报</div>
    <div class="brand-sub">餐饮评论情感分析平台</div>
  </div>

  <div class="card">
    <div class="card-title">欢迎回来</div>

    <div class="field">
      <label>账号</label>
      <input type="text" id="uname" placeholder="请输入账号" autocomplete="username">
    </div>
    <div class="field">
      <label>密码</label>
      <input type="password" id="pwd" placeholder="请输入密码" autocomplete="current-password">
    </div>

    <button class="btn-login" onclick="doLogin()">登 录</button>
    <div class="err-msg" id="errMsg"></div>

    <div class="demo-accounts">
      <strong>📋 演示账号（密码均为 123456）</strong>
      <div class="demo-item" onclick="fillUser('admin')"><span>👨‍💼 admin</span><span style="color:var(--ink-3)">管理员</span></div>
      <div class="demo-item" onclick="fillUser('alice')"><span>👩 alice</span><span style="color:var(--ink-3)">爱丽丝</span></div>
      <div class="demo-item" onclick="fillUser('bob')"><span>👦 bob</span><span style="color:var(--ink-3)">小明</span></div>
    </div>
  </div>
</div>

<script>
function fillUser(u) {
  document.getElementById('uname').value = u;
  document.getElementById('pwd').value = '123456';
}

function doLogin() {
  const u = document.getElementById('uname').value.trim();
  const p = document.getElementById('pwd').value;
  const errEl = document.getElementById('errMsg');
  if (!u || !p) { showErr('请填写账号和密码'); return; }

  fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: u, password: p })
  }).then(r => r.json()).then(d => {
    if (d.ok) {
      window.location.href = '/';
    } else {
      showErr(d.error || '登录失败');
    }
  }).catch(() => showErr('网络错误，请重试'));
}

function showErr(msg) {
  const el = document.getElementById('errMsg');
  el.textContent = msg;
  el.style.display = 'block';
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter') doLogin();
});
</script>
</body>
</html>"""

MAIN_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>饮食情报</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --red: #D94F3D;
  --red-dark: #B03A2C;
  --red-light: #FDECEA;
  --gold: #C9943A;
  --ink: #1A1008;
  --ink-2: #4A3728;
  --ink-3: #8A7265;
  --cream: #FBF7F2;
  --paper: #F5EFE6;
  --border: #E8DDD2;
  --green: #2E8B57;
  --green-light: #E6F5EC;
  --orange: #E07830;
  --orange-light: #FEF3E8;
}

body {
  font-family: 'Noto Sans SC', sans-serif;
  background: var(--cream);
  color: var(--ink);
  min-height: 100vh;
}

/* ─── Navbar ─────────────────────────────────────────────── */
.navbar {
  background: #fff;
  border-bottom: 2px solid var(--red);
  padding: 0 24px;
  display: flex;
  align-items: center;
  height: 56px;
  gap: 20px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 12px rgba(26,16,8,0.08);
}

.nav-brand {
  font-family: 'Noto Serif SC', serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--red);
  letter-spacing: 3px;
  white-space: nowrap;
  display: flex;
  align-items: center;
  gap: 6px;
}

.nav-tabs {
  display: flex;
  gap: 4px;
  flex: 1;
}

.nav-tab {
  padding: 6px 14px;
  border-radius: 8px;
  font-size: 13px;
  color: var(--ink-3);
  cursor: pointer;
  transition: all .2s;
  font-weight: 500;
}

.nav-tab:hover { background: var(--paper); color: var(--ink); }
.nav-tab.active { background: var(--red-light); color: var(--red); font-weight: 600; }

.nav-user {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--ink-2);
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 8px;
  transition: background .2s;
}

.nav-user:hover { background: var(--paper); }

.nav-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--red-light);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
}

.logout-btn {
  padding: 5px 12px;
  border: 1.5px solid var(--border);
  border-radius: 8px;
  font-size: 12px;
  color: var(--ink-3);
  cursor: pointer;
  transition: all .2s;
  background: transparent;
  font-family: inherit;
}

.logout-btn:hover { border-color: var(--red); color: var(--red); }

/* ─── Layout ─────────────────────────────────────────────── */
.page { display: none; }
.page.active { display: block; }

.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px;
}

/* ─── Merchant List ───────────────────────────────────────── */
.section-title {
  font-family: 'Noto Serif SC', serif;
  font-size: 18px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.merchant-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}

.merchant-card {
  background: #fff;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
  cursor: pointer;
  transition: transform .2s, box-shadow .2s;
  box-shadow: 0 2px 8px rgba(26,16,8,0.06);
}

.merchant-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(26,16,8,0.12);
}

.merchant-cover {
  height: 100px;
  background: linear-gradient(135deg, var(--paper), var(--cream));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  border-bottom: 1px solid var(--border);
  position: relative;
}

.merchant-cover-bg {
  position: absolute;
  inset: 0;
  opacity: 0.05;
  background-image: repeating-linear-gradient(
    -45deg,
    var(--red) 0, var(--red) 1px,
    transparent 0, transparent 8px
  );
}

.merchant-body { padding: 14px; }

.merchant-name {
  font-family: 'Noto Serif SC', serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 4px;
}

.merchant-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--ink-3);
  margin-bottom: 10px;
}

.rating {
  display: flex;
  align-items: center;
  gap: 3px;
  color: var(--gold);
  font-weight: 600;
  font-size: 13px;
}

.tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.tag {
  padding: 2px 8px;
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 11px;
  color: var(--ink-2);
}

.merchant-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}

.price {
  font-size: 13px;
  color: var(--ink-2);
}

.price strong { color: var(--red); font-size: 15px; }

.review-count {
  font-size: 11px;
  color: var(--ink-3);
}

/* ─── Merchant Detail ─────────────────────────────────────── */
.back-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--ink-3);
  cursor: pointer;
  padding: 6px 0;
  margin-bottom: 16px;
  transition: color .2s;
}

.back-btn:hover { color: var(--red); }

.detail-header {
  background: #fff;
  border-radius: 14px;
  padding: 24px;
  margin-bottom: 16px;
  border: 1px solid var(--border);
  display: flex;
  gap: 20px;
  align-items: center;
}

.detail-cover {
  width: 80px;
  height: 80px;
  border-radius: 14px;
  background: var(--paper);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 40px;
  flex-shrink: 0;
  border: 1px solid var(--border);
}

.detail-name {
  font-family: 'Noto Serif SC', serif;
  font-size: 22px;
  font-weight: 700;
  color: var(--ink);
  margin-bottom: 6px;
}

.detail-address {
  font-size: 12px;
  color: var(--ink-3);
  margin-top: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}

/* ─── Reviews ────────────────────────────────────────────── */
.review-card {
  background: #fff;
  border-radius: 12px;
  padding: 18px;
  margin-bottom: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 1px 4px rgba(26,16,8,0.04);
}

.review-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.reviewer {
  display: flex;
  align-items: center;
  gap: 10px;
}

.reviewer-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--paper);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  border: 1px solid var(--border);
}

.reviewer-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--ink);
}

.reviewer-date {
  font-size: 11px;
  color: var(--ink-3);
  margin-top: 2px;
}

.stars {
  color: var(--gold);
  font-size: 14px;
}

.review-text {
  font-size: 13.5px;
  line-height: 1.75;
  color: var(--ink-2);
  margin-bottom: 14px;
}

.analyze-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: var(--red-light);
  border: 1.5px solid rgba(217,79,61,0.3);
  border-radius: 8px;
  font-size: 12px;
  color: var(--red);
  cursor: pointer;
  font-family: inherit;
  font-weight: 500;
  transition: all .2s;
}

.analyze-btn:hover {
  background: var(--red);
  color: #fff;
  border-color: var(--red);
}

.analyze-btn:disabled {
  opacity: .6;
  cursor: not-allowed;
}

.analysis-result {
  margin-top: 14px;
  padding: 14px;
  background: var(--paper);
  border-radius: 10px;
  border: 1px solid var(--border);
  animation: slideIn .3s ease;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}

.result-title {
  font-size: 11px;
  color: var(--ink-3);
  letter-spacing: 1px;
  margin-bottom: 10px;
  font-weight: 500;
}

.dim-row {
  display: flex;
  gap: 10px;
}

.dim-pill {
  flex: 1;
  padding: 10px;
  border-radius: 10px;
  text-align: center;
  border: 1.5px solid;
}

.dim-pill-name { font-size: 11px; color: var(--ink-3); margin-bottom: 4px; }
.dim-pill-emoji { font-size: 22px; }
.dim-pill-sent { font-size: 12px; font-weight: 700; margin-top: 2px; }
.dim-pill-conf { font-size: 10px; color: var(--ink-3); margin-top: 1px; }

.dim-pill.pos { border-color: #a8d8bb; background: var(--green-light); }
.dim-pill.pos .dim-pill-sent { color: var(--green); }
.dim-pill.neu { border-color: #f0d79a; background: var(--orange-light); }
.dim-pill.neu .dim-pill-sent { color: var(--orange); }
.dim-pill.neg { border-color: #f5b8b3; background: var(--red-light); }
.dim-pill.neg .dim-pill-sent { color: var(--red); }

.prob-mini {
  display: flex;
  gap: 2px;
  margin-top: 6px;
  height: 3px;
  border-radius: 2px;
  overflow: hidden;
}

.prob-seg { height: 100%; transition: width .4s; }

/* ─── Single Analyze Page ──────────────────────────────── */
.analyze-card {
  background: #fff;
  border-radius: 14px;
  padding: 24px;
  border: 1px solid var(--border);
  box-shadow: 0 2px 8px rgba(26,16,8,0.06);
  margin-bottom: 16px;
}

.analyze-card h3 {
  font-family: 'Noto Serif SC', serif;
  font-size: 16px;
  font-weight: 600;
  color: var(--ink);
  margin-bottom: 14px;
}

textarea {
  width: 100%;
  padding: 12px 14px;
  border: 1.5px solid var(--border);
  border-radius: 10px;
  font-size: 13.5px;
  font-family: 'Noto Sans SC', sans-serif;
  resize: vertical;
  min-height: 90px;
  color: var(--ink);
  background: var(--paper);
  outline: none;
  transition: border-color .2s, box-shadow .2s;
}

textarea:focus {
  border-color: var(--red);
  box-shadow: 0 0 0 3px rgba(217,79,61,0.10);
  background: #fff;
}

.btn-primary {
  padding: 10px 24px;
  background: var(--red);
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 13px;
  font-family: 'Noto Sans SC', sans-serif;
  font-weight: 500;
  cursor: pointer;
  transition: all .2s;
  box-shadow: 0 3px 12px rgba(217,79,61,0.3);
  margin-top: 12px;
}

.btn-primary:hover { background: var(--red-dark); }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; }

.examples-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 10px 0 0;
}

.ex-chip {
  padding: 4px 10px;
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: 20px;
  font-size: 11px;
  cursor: pointer;
  color: var(--ink-2);
  transition: all .15s;
}

.ex-chip:hover { background: var(--red-light); border-color: var(--red); color: var(--red); }

.big-result {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  margin-top: 16px;
}

.big-dim {
  padding: 18px;
  border-radius: 14px;
  text-align: center;
  border: 1.5px solid;
}

.big-dim-name { font-size: 12px; color: var(--ink-3); margin-bottom: 6px; }
.big-dim-emoji { font-size: 32px; }
.big-dim-sent { font-size: 15px; font-weight: 700; margin: 6px 0 4px; }
.big-dim-conf { font-size: 11px; color: var(--ink-3); }
.big-dim.pos { border-color: #a8d8bb; background: var(--green-light); }
.big-dim.pos .big-dim-sent { color: var(--green); }
.big-dim.neu { border-color: #f0d79a; background: var(--orange-light); }
.big-dim.neu .big-dim-sent { color: var(--orange); }
.big-dim.neg { border-color: #f5b8b3; background: var(--red-light); }
.big-dim.neg .big-dim-sent { color: var(--red); }

.prob-bars {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-top: 10px;
  text-align: left;
}

.prob-bar-row { display: flex; align-items: center; gap: 6px; font-size: 10px; color: var(--ink-3); }
.prob-track { flex: 1; height: 5px; background: rgba(0,0,0,0.07); border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; transition: width .5s ease; }

/* ─── History Page ────────────────────────────────────────── */
.hist-card {
  background: #fff;
  border-radius: 12px;
  padding: 14px 18px;
  margin-bottom: 10px;
  border: 1px solid var(--border);
  font-size: 13px;
}

.hist-text { color: var(--ink); margin-bottom: 8px; line-height: 1.6; }
.hist-meta { display: flex; justify-content: space-between; align-items: center; }
.hist-time { font-size: 11px; color: var(--ink-3); }
.hist-badges { display: flex; gap: 6px; }
.badge {
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 500;
}
.badge.pos { background: var(--green-light); color: var(--green); }
.badge.neu { background: var(--orange-light); color: var(--orange); }
.badge.neg { background: var(--red-light); color: var(--red); }

.loading-spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(217,79,61,0.2);
  border-top-color: var(--red);
  border-radius: 50%;
  animation: spin .7s linear infinite;
  vertical-align: middle;
  margin-right: 4px;
}

@keyframes spin { to { transform: rotate(360deg); } }

.empty-state {
  text-align: center;
  padding: 40px;
  color: var(--ink-3);
}

.empty-state .empty-icon { font-size: 40px; margin-bottom: 8px; }
.empty-state p { font-size: 13px; }
</style>
</head>
<body>

<nav class="navbar">
  <div class="nav-brand">🍽️ 饮食情报</div>
  <div class="nav-tabs">
    <div class="nav-tab active" onclick="showPage('merchants')" id="tab-merchants">🏪 发现餐厅</div>
    <div class="nav-tab" onclick="showPage('analyze')" id="tab-analyze">🔍 情感分析</div>
    <div class="nav-tab" onclick="showPage('history')" id="tab-history">📋 分析历史</div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-left:auto">
    <div class="nav-user">
      <div class="nav-avatar" id="navAvatar">👤</div>
      <span id="navName">用户</span>
    </div>
    <button class="logout-btn" onclick="doLogout()">退出</button>
  </div>
</nav>

<!-- ─── Page: Merchant List ─── -->
<div class="page active" id="page-merchants">
  <div class="container">
    <!-- Merchant Detail (hidden by default) -->
    <div id="detail-view" style="display:none">
      <div class="back-btn" onclick="closeDetail()">← 返回列表</div>
      <div class="detail-header" id="detail-header"></div>
      <div class="section-title">💬 用户评论</div>
      <div id="reviews-list"></div>
    </div>

    <!-- Merchant Grid -->
    <div id="list-view">
      <div class="section-title">🔥 热门餐厅</div>
      <div class="merchant-grid" id="merchant-grid"></div>
    </div>
  </div>
</div>

<!-- ─── Page: Single Analyze ─── -->
<div class="page" id="page-analyze">
  <div class="container" style="max-width:700px">
    <div class="section-title">🔍 自由情感分析</div>
    <div class="analyze-card">
      <h3>输入评论文本</h3>
      <textarea id="inp" placeholder="输入任意餐饮评论，例如：菜品味道很好，环境有点嘈杂，服务态度一般…"></textarea>
      <div class="examples-row">
        <span class="ex-chip" onclick="setInp('菜品味道非常好，食材新鲜，环境优雅，服务一流！')">全好评示例</span>
        <span class="ex-chip" onclick="setInp('菜很难吃，环境脏乱差，服务员态度极差')">全差评示例</span>
        <span class="ex-chip" onclick="setInp('口味还不错，就是环境太嘈杂了，上菜速度也很慢')">混合型示例</span>
        <span class="ex-chip" onclick="setInp('装修高档有情调，但菜的口味一般，价格偏贵')">部分好评</span>
      </div>
      <button class="btn-primary" id="analyzeBtn" onclick="doAnalyze()">🔍 分析情感</button>
    </div>

    <div id="analyze-result" style="display:none">
      <div class="analyze-card">
        <h3>📊 分析结果</h3>
        <div class="big-result" id="big-result-grid"></div>
      </div>
    </div>
  </div>
</div>

<!-- ─── Page: History ─── -->
<div class="page" id="page-history">
  <div class="container" style="max-width:700px">
    <div class="section-title">📋 分析历史</div>
    <div id="hist-list">
      <div class="empty-state"><div class="empty-icon">📭</div><p>暂无历史记录，去分析一些评论吧！</p></div>
    </div>
  </div>
</div>

<script>
// ── Constants ─────────────────────────────────────────────
const DIMS = [
  { k: 'taste',   name: '口味' },
  { k: 'env',     name: '环境' },
  { k: 'service', name: '服务' },
];
const SENT_CLS  = ['neg', 'neu', 'pos'];
const SENT_LABEL = ['负向', '中立', '正向'];
const PROB_COLORS = ['#D94F3D', '#E07830', '#2E8B57'];

// ── Init ──────────────────────────────────────────────────
const USER_INFO = {{ user_info | tojson }};
const MERCHANTS = {{ merchants | tojson }};

document.getElementById('navAvatar').textContent = USER_INFO.avatar;
document.getElementById('navName').textContent   = USER_INFO.name;

renderMerchants();

// ── Page switch ───────────────────────────────────────────
function showPage(name) {
  ['merchants','analyze','history'].forEach(n => {
    document.getElementById('page-'+n).classList.toggle('active', n===name);
    document.getElementById('tab-'+n).classList.toggle('active', n===name);
  });
  if (name === 'history') loadHistory();
}

// ── Merchant list ─────────────────────────────────────────
function renderMerchants() {
  const grid = document.getElementById('merchant-grid');
  grid.innerHTML = MERCHANTS.map(m => `
    <div class="merchant-card" onclick="openMerchant(${m.id})">
      <div class="merchant-cover">
        <div class="merchant-cover-bg"></div>
        <span style="position:relative">${m.cover}</span>
      </div>
      <div class="merchant-body">
        <div class="merchant-name">${m.name}</div>
        <div class="merchant-meta">
          <span class="rating">★ ${m.rating}</span>
          <span>${m.category}</span>
        </div>
        <div class="tags">
          ${m.tags.map(t => `<span class="tag">${t}</span>`).join('')}
        </div>
        <div class="merchant-footer">
          <div class="price">人均 <strong>¥${m.price}</strong></div>
          <div class="review-count">${m.reviews.length} 条评论</div>
        </div>
      </div>
    </div>
  `).join('');
}

// ── Merchant detail ───────────────────────────────────────
function openMerchant(id) {
  const m = MERCHANTS.find(x => x.id === id);
  if (!m) return;

  document.getElementById('list-view').style.display = 'none';
  document.getElementById('detail-view').style.display = 'block';

  document.getElementById('detail-header').innerHTML = `
    <div class="detail-cover">${m.cover}</div>
    <div style="flex:1">
      <div class="detail-name">${m.name}</div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:4px">
        <span class="rating" style="font-size:15px">★ ${m.rating}</span>
        <span style="font-size:13px;color:var(--ink-3)">${m.category}</span>
        <span style="font-size:13px;color:var(--red)">人均 ¥${m.price}</span>
      </div>
      <div class="detail-address">📍 ${m.address}</div>
      <div class="tags" style="margin-top:8px">
        ${m.tags.map(t => `<span class="tag">${t}</span>`).join('')}
      </div>
    </div>
  `;

  document.getElementById('reviews-list').innerHTML = m.reviews.map((r, idx) => `
    <div class="review-card" id="review-${id}-${idx}">
      <div class="review-header">
        <div class="reviewer">
          <div class="reviewer-avatar">${r.avatar}</div>
          <div>
            <div class="reviewer-name">${r.user}</div>
            <div class="reviewer-date">${r.date}</div>
          </div>
        </div>
        <div class="stars">${'★'.repeat(r.stars)}${'☆'.repeat(5-r.stars)}</div>
      </div>
      <div class="review-text">${r.text}</div>
      <button class="analyze-btn" id="btn-${id}-${idx}" onclick="analyzeReview(${id}, ${idx}, this)">
        ✨ 情感分析
      </button>
      <div id="result-${id}-${idx}"></div>
    </div>
  `).join('');
}

function closeDetail() {
  document.getElementById('list-view').style.display = '';
  document.getElementById('detail-view').style.display = 'none';
}

// ── Analyze a review in merchant page ────────────────────
function analyzeReview(merchantId, reviewIdx, btn) {
  const m = MERCHANTS.find(x => x.id === merchantId);
  if (!m) return;
  const text = m.reviews[reviewIdx].text;
  const resultEl = document.getElementById(`result-${merchantId}-${reviewIdx}`);

  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span> 分析中…';
  resultEl.innerHTML = '';

  callPredict(text).then(data => {
    btn.style.display = 'none';
    resultEl.innerHTML = renderInlineResult(data);
  }).catch(err => {
    btn.disabled = false;
    btn.innerHTML = '✨ 情感分析';
    alert('分析失败：' + err.message);
  });
}

function renderInlineResult(data) {
  const dims = DIMS.map(d => {
    const info = data.dimensions[d.k];
    const cls  = SENT_CLS[info.label];
    const probs = [info.probs['负向'], info.probs['中立'], info.probs['正向']];
    const segs = probs.map((p, i) => `<div class="prob-seg" style="width:${Math.round(p*100)}%;background:${PROB_COLORS[i]}"></div>`).join('');
    return `
      <div class="dim-pill ${cls}">
        <div class="dim-pill-name">${d.name}</div>
        <div class="dim-pill-emoji">${info.emoji}</div>
        <div class="dim-pill-sent">${info.sentiment}</div>
        <div class="dim-pill-conf">${(info.confidence*100).toFixed(0)}%</div>
        <div class="prob-mini">${segs}</div>
      </div>
    `;
  }).join('');
  return `
    <div class="analysis-result">
      <div class="result-title">AI 情感分析结果</div>
      <div class="dim-row">${dims}</div>
    </div>
  `;
}

// ── Single analyze page ───────────────────────────────────
function setInp(t) { document.getElementById('inp').value = t; }

function doAnalyze() {
  const text = document.getElementById('inp').value.trim();
  if (!text) { alert('请输入评论文本'); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.textContent = '分析中…';
  document.getElementById('analyze-result').style.display = 'none';

  callPredict(text).then(data => {
    const grid = document.getElementById('big-result-grid');
    grid.innerHTML = DIMS.map(d => {
      const info  = data.dimensions[d.k];
      const cls   = SENT_CLS[info.label];
      const probs = [
        { label: '负向', val: info.probs['负向'], color: PROB_COLORS[0] },
        { label: '中立', val: info.probs['中立'], color: PROB_COLORS[1] },
        { label: '正向', val: info.probs['正向'], color: PROB_COLORS[2] },
      ];
      const bars = probs.map(p => `
        <div class="prob-bar-row">
          <span style="width:24px">${p.label}</span>
          <div class="prob-track">
            <div class="prob-fill" style="width:${Math.round(p.val*100)}%;background:${p.color}"></div>
          </div>
          <span style="width:30px;text-align:right">${Math.round(p.val*100)}%</span>
        </div>
      `).join('');
      return `
        <div class="big-dim ${cls}">
          <div class="big-dim-name">${d.name}</div>
          <div class="big-dim-emoji">${info.emoji}</div>
          <div class="big-dim-sent">${info.sentiment}</div>
          <div class="big-dim-conf">置信度 ${(info.confidence*100).toFixed(1)}%</div>
          <div class="prob-bars">${bars}</div>
        </div>
      `;
    }).join('');
    document.getElementById('analyze-result').style.display = 'block';
  }).catch(err => {
    alert('分析失败：' + err.message);
  }).finally(() => {
    btn.disabled = false;
    btn.textContent = '🔍 分析情感';
  });
}

// ── History ───────────────────────────────────────────────
function loadHistory() {
  fetch('/api/history').then(r => r.json()).then(records => {
    const el = document.getElementById('hist-list');
    if (!records.length) {
      el.innerHTML = '<div class="empty-state"><div class="empty-icon">📭</div><p>暂无历史记录</p></div>';
      return;
    }
    el.innerHTML = [...records].reverse().slice(0, 30).map(item => {
      const badges = DIMS.map(d => {
        const info = item.dimensions[d.k];
        return `<span class="badge ${SENT_CLS[info.label]}">${d.name}：${info.sentiment}</span>`;
      }).join('');
      return `
        <div class="hist-card">
          <div class="hist-text">${item.text}</div>
          <div class="hist-meta">
            <div class="hist-badges">${badges}</div>
            <div class="hist-time">${item.time}</div>
          </div>
        </div>
      `;
    }).join('');
  });
}

// ── Shared predict call ───────────────────────────────────
function callPredict(text) {
  return fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  }).then(r => r.json().then(d => {
    if (!r.ok) throw new Error(d.error || '请求失败');
    return d;
  }));
}

// ── Logout ────────────────────────────────────────────────
function doLogout() {
  fetch('/api/logout', { method: 'POST' }).then(() => {
    window.location.href = '/login';
  });
}
</script>
</body>
</html>"""

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
    password = data.get('password', '')
    user = USERS.get(username)
    if user and user['password'] == password:
        session['username'] = username
        return jsonify({'ok': True, 'name': user['name']})
    return jsonify({'ok': False, 'error': '账号或密码错误'}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'ok': True})

@app.route('/')
@login_required
def index():
    username = session['username']
    user = USERS[username]
    return render_template_string(
        MAIN_HTML,
        user_info={'name': user['name'], 'avatar': user['avatar'], 'username': username},
        merchants=MERCHANTS,
    )

@app.route('/api/predict', methods=['POST'])
@api_login_required
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请提供 text 字段'}), 400
    result = get_predictor().predict_one(text)
    history_records.append({**result, 'time': datetime.now().strftime('%m-%d %H:%M')})
    return jsonify(result)

@app.route('/api/history')
@api_login_required
def history():
    return jsonify(history_records[-100:])

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print('=' * 50)
    print('  饮食情报 · 餐饮评论情感分析平台')
    print('  http://localhost:5001')
    print('  演示账号: admin / alice / bob  密码: 123456')
    print('=' * 50)
    app.run(debug=True, host='0.0.0.0', port=5001)
