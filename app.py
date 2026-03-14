import os, sys, io, json
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import SentimentPredictor

app = Flask(__name__)
history_records = []

_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor(
            checkpoint_path=os.environ.get('MODEL_PATH', 'checkpoints/best_model.pt'),
            bert_model_name=os.environ.get('BERT_NAME', 'bert-base-chinese'),
        )
    return _predictor

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>餐饮评论情感分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f0eb}
.header{background:linear-gradient(135deg,#c0392b,#e67e22);color:#fff;padding:16px 28px;display:flex;align-items:center;gap:12px}
.header h1{font-size:1.2rem;font-weight:700}
.header p{font-size:.8rem;opacity:.85;margin-top:2px}
.tabs{display:flex;background:#fff;border-bottom:2px solid #f0e8e0;padding:0 28px}
.tab{padding:12px 20px;cursor:pointer;color:#888;font-size:.9rem;border-bottom:3px solid transparent;margin-bottom:-2px}
.tab.active{color:#c0392b;border-bottom-color:#c0392b;font-weight:600}
.page{display:none;padding:24px 28px;max-width:820px;margin:0 auto}
.page.active{display:block}
.card{background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.card h3{color:#c0392b;margin-bottom:12px;font-size:.92rem}
textarea{width:100%;padding:10px;border:1.5px solid #e8d8d0;border-radius:8px;font-size:.9rem;resize:vertical;min-height:80px;font-family:inherit}
textarea:focus{outline:none;border-color:#c0392b}
.examples{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px}
.ex{padding:4px 10px;background:#fdf2ee;border:1px solid #f0c8b8;border-radius:12px;cursor:pointer;font-size:.78rem;color:#c0392b}
.ex:hover{background:#c0392b;color:#fff}
.btn{padding:9px 22px;background:#c0392b;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:.9rem}
.btn:hover{background:#962d22}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-out{background:#fff;color:#c0392b;border:1.5px solid #c0392b}
.btn-out:hover{background:#fdf2ee}
.dim-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}
.dim-card{border-radius:10px;padding:14px;text-align:center;border:1.5px solid}
.dim-name{font-size:.76rem;color:#999;margin-bottom:4px}
.dim-emoji{font-size:1.8rem;margin:4px 0}
.dim-sent{font-size:1rem;font-weight:700;margin-bottom:4px}
.dim-conf{font-size:.72rem;color:#bbb}
.neg{border-color:#ffd0cc;background:#fff8f7}.neg .dim-sent{color:#c0392b}
.neu{border-color:#ffe8b0;background:#fffdf5}.neu .dim-sent{color:#d68910}
.pos{border-color:#c8f0d8;background:#f5fff8}.pos .dim-sent{color:#1a8a3c}
.prob-row{display:flex;gap:4px;margin-top:8px;font-size:.68rem;text-align:center}
.prob-item{flex:1}
.prob-bar-wrap{background:#f0ece8;border-radius:3px;height:5px;margin-bottom:2px}
.prob-fill{height:5px;border-radius:3px}
.upload-zone{border:2px dashed #e8d8d0;border-radius:10px;padding:32px;text-align:center;cursor:pointer;background:#fdf9f7}
.upload-zone:hover{border-color:#c0392b;background:#fdf2ee}
.batch-table{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:12px}
.batch-table th{background:#c0392b;color:#fff;padding:8px;text-align:center}
.batch-table td{padding:7px 8px;border-bottom:1px solid #f0e8e0;text-align:center}
.badge{display:inline-block;padding:2px 8px;border-radius:8px;font-size:.72rem;font-weight:600}
.b-neg{background:#ffd0cc;color:#8b1a1a}
.b-neu{background:#ffe8b0;color:#7a5c00}
.b-pos{background:#c8f0d8;color:#0d5c2e}
.chart-wrap{position:relative;height:240px}
.hist-item{padding:9px 0;border-bottom:1px solid #f0e8e0;font-size:.83rem}
.hist-meta{color:#bbb;font-size:.72rem;margin-top:2px}
#err{display:none;color:#c0392b;padding:9px;background:#fff8f7;border-radius:7px;margin-bottom:10px}
#loading{display:none;text-align:center;padding:18px;color:#c0392b}
</style>
</head>
<body>
<div class="header">
  <div style="font-size:1.6rem">🍜</div>
  <div>
    <h1>餐饮评论情感分析系统</h1>
    <p>基于BERT多任务学习 · 口味 / 环境 / 服务 三维度情感分析</p>
  </div>
</div>
<div class="tabs">
  <div class="tab active" id="tab-single" onclick="sw('single')">单条分析</div>
  <div class="tab" id="tab-batch" onclick="sw('batch')">批量分析</div>
  <div class="tab" id="tab-visual" onclick="sw('visual')">可视化</div>
</div>

<div class="page active" id="page-single">
  <div class="card">
    <h3>📝 输入评论</h3>
    <textarea id="inp" placeholder="输入餐饮评论，例如：菜品味道很好，环境有点吵，服务态度不错"></textarea>
    <div class="examples">
      <span style="font-size:.73rem;color:#ccc;line-height:2.2">示例：</span>
      <span class="ex" onclick="si('菜品味道非常好，食材新鲜，环境优雅，服务一流')">全好评</span>
      <span class="ex" onclick="si('菜很难吃，环境脏乱差，服务态度极差')">全差评</span>
      <span class="ex" onclick="si('口味还不错，就是环境太嘈杂了，上菜也慢')">混合型</span>
      <span class="ex" onclick="si('装修很有特色，但菜的口味一般，价格偏贵')">部分好评</span>
    </div>
    <button class="btn" id="abtn" onclick="analyze()">🔍 分析情感</button>
    <span style="font-size:.76rem;color:#ccc;margin-left:8px">Ctrl+Enter</span>
  </div>
  <div id="loading">⏳ 模型分析中…</div>
  <div id="err"></div>
  <div id="result" style="display:none">
    <div class="card">
      <h3>📊 分析结果</h3>
      <div id="dimGrid" class="dim-grid"></div>
    </div>
  </div>
</div>

<div class="page" id="page-batch">
  <div class="card">
    <h3>📂 批量上传</h3>
    <div class="upload-zone" onclick="document.getElementById('fi').click()"
         ondragover="event.preventDefault();this.style.borderColor='#c0392b'"
         ondragleave="this.style.borderColor=''"
         ondrop="handleDrop(event)">
      <div style="font-size:1.6rem">📄</div>
      <div style="font-weight:600;color:#888;margin-top:6px">点击或拖拽上传 CSV / Excel</div>
      <div style="color:#bbb;font-size:.8rem;margin-top:3px">需包含 content 或 text 列</div>
    </div>
    <input type="file" id="fi" style="display:none" accept=".csv,.xlsx,.xls" onchange="handleFile(this.files[0])">
    <div style="margin-top:8px">
      <button class="btn btn-out" style="font-size:.78rem" onclick="dlTemplate()">⬇ 下载模板</button>
    </div>
  </div>
  <div class="card" id="batchCard" style="display:none">
    <h3>📋 批量结果</h3>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <span id="batchSum" style="font-size:.82rem;color:#888"></span>
      <button class="btn btn-out" style="font-size:.76rem" onclick="exportCSV()">⬇ 导出</button>
    </div>
    <div style="overflow-x:auto">
      <table class="batch-table" id="bt">
        <thead><tr><th>#</th><th style="text-align:left;min-width:160px">评论</th><th>口味</th><th>环境</th><th>服务</th></tr></thead>
        <tbody id="bb"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="page" id="page-visual">
  <div class="card">
    <h3>📈 情感分布（近100条）</h3>
    <div class="chart-wrap"><canvas id="chart"></canvas></div>
  </div>
  <div class="card">
    <h3>📋 历史记录</h3>
    <div id="histList" style="color:#bbb;font-size:.83rem">暂无记录</div>
  </div>
</div>

<script>
var DIMS = [{k:'taste',n:'口味'},{k:'env',n:'环境'},{k:'service',n:'服务'}];
var SENT_CLS = ['neg','neu','pos'];
var SENT_LBL = ['负向','中立','正向'];
var SENT_EMJ = ['😞','😐','😊'];
var PROB_CLR = ['#e74c3c','#f39c12','#27ae60'];
var batchData = [];
var chartInst = null;

function sw(name) {
  ['single','batch','visual'].forEach(function(n) {
    document.getElementById('tab-'+n).classList.toggle('active', n===name);
    document.getElementById('page-'+n).classList.toggle('active', n===name);
  });
  if (name === 'visual') loadViz();
}

function si(t) { document.getElementById('inp').value = t; }

function analyze() {
  var text = document.getElementById('inp').value.trim();
  if (!text) { alert('请输入评论'); return; }
  document.getElementById('abtn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('result').style.display = 'none';
  document.getElementById('err').style.display = 'none';
  fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: text})
  }).then(function(r) {
    return r.json().then(function(d) { return {ok: r.ok, d: d}; });
  }).then(function(res) {
    if (!res.ok) throw new Error(res.d.error || '服务器错误');
    showResult(res.d);
  }).catch(function(e) {
    var el = document.getElementById('err');
    el.style.display = 'block';
    el.textContent = '⚠️ ' + e.message;
  }).finally(function() {
    document.getElementById('abtn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  });
}

function showResult(d) {
  var grid = document.getElementById('dimGrid');
  grid.innerHTML = '';
  DIMS.forEach(function(dim) {
    var info = d.dimensions[dim.k];
    var cls  = SENT_CLS[info.label];
    var bars = ['负向','中立','正向'].map(function(lb, i) {
      var pct = Math.round(info.probs[lb] * 100);
      return '<div class="prob-item">'
        + '<div class="prob-bar-wrap"><div class="prob-fill" style="width:'+pct+'%;background:'+PROB_CLR[i]+'"></div></div>'
        + lb+' '+pct+'%</div>';
    }).join('');
    grid.innerHTML += '<div class="dim-card '+cls+'">'
      + '<div class="dim-name">'+dim.n+'</div>'
      + '<div class="dim-emoji">'+info.emoji+'</div>'
      + '<div class="dim-sent">'+info.sentiment+'</div>'
      + '<div class="dim-conf">置信度 '+(info.confidence*100).toFixed(1)+'%</div>'
      + '<div class="prob-row">'+bars+'</div>'
      + '</div>';
  });
  document.getElementById('result').style.display = 'block';
}

function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.style.borderColor = '';
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
}

function handleFile(file) {
  var fd = new FormData();
  fd.append('file', file);
  fetch('/api/batch', {method:'POST', body:fd}).then(function(r) {
    return r.json().then(function(d) { return {ok:r.ok, d:d}; });
  }).then(function(res) {
    if (!res.ok) { alert(res.d.error || '上传失败'); return; }
    batchData = res.d.results;
    document.getElementById('batchSum').textContent = '共 '+res.d.results.length+' 条';
    document.getElementById('batchCard').style.display = 'block';
    var html = res.d.results.map(function(r, i) {
      var cells = ['taste','env','service'].map(function(k) {
        var s = r.dimensions[k].label;
        return '<td><span class="badge b-'+SENT_CLS[s]+'">'+SENT_LBL[s]+'</span></td>';
      }).join('');
      return '<tr><td>'+(i+1)+'</td><td style="text-align:left;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+r.text+'">'+r.text+'</td>'+cells+'</tr>';
    }).join('');
    document.getElementById('bb').innerHTML = html;
  }).catch(function(e) { alert('错误：'+e.message); });
}

function dlTemplate() {
  var a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,\uFEFF' + encodeURIComponent('content\n菜品味道很好，环境一般\n服务很差，等了很久\n');
  a.download = 'template.csv';
  a.click();
}

function exportCSV() {
  if (!batchData.length) { alert('无数据'); return; }
  var csv = '评论,口味,环境,服务\n';
  batchData.forEach(function(r) {
    var dims = ['taste','env','service'].map(function(k) { return SENT_LBL[r.dimensions[k].label]; });
    csv += '"'+r.text+'",'+dims.join(',')+'\n';
  });
  var a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,\uFEFF' + encodeURIComponent(csv);
  a.download = 'results.csv';
  a.click();
}

function loadViz() {
  fetch('/api/history').then(function(r) { return r.json(); }).then(function(data) {
    if (!data.length) { document.getElementById('histList').textContent = '暂无记录'; return; }
    var counts = [[0,0,0],[0,0,0],[0,0,0]];
    data.forEach(function(item) {
      ['taste','env','service'].forEach(function(k, i) {
        counts[i][item.dimensions[k].label]++;
      });
    });
    if (chartInst) chartInst.destroy();
    chartInst = new Chart(document.getElementById('chart'), {
      type: 'bar',
      data: {
        labels: ['负向','中立','正向'],
        datasets: ['口味','环境','服务'].map(function(n, i) {
          return {label:n, data:counts[i],
            backgroundColor:['#e74c3c88','#f39c1288','#27ae6088'][i],
            borderColor:['#e74c3c','#f39c12','#27ae60'][i], borderWidth:1};
        })
      },
      options: {responsive:true, maintainAspectRatio:false,
        plugins:{legend:{position:'top'}},
        scales:{y:{beginAtZero:true, ticks:{stepSize:1}}}}
    });
    var html = data.slice().reverse().slice(0,8).map(function(item) {
      var tags = ['taste','env','service'].map(function(k, i) {
        var s = item.dimensions[k].label;
        return '<span class="badge b-'+SENT_CLS[s]+'" style="margin-right:4px">'+['口味','环境','服务'][i]+':'+SENT_LBL[s]+'</span>';
      }).join('');
      return '<div class="hist-item"><div>'+item.text+'</div><div style="margin-top:3px">'+tags+'</div><div class="hist-meta">'+item.time+'</div></div>';
    }).join('');
    document.getElementById('histList').innerHTML = html;
  });
}

document.getElementById('inp').addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'Enter') analyze();
});
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': '请提供 text 字段'}), 400
    result = get_predictor().predict_one(text)
    history_records.append({**result, 'time': datetime.now().strftime('%m-%d %H:%M')})
    return jsonify(result)

@app.route('/api/batch', methods=['POST'])
def batch():
    if 'file' not in request.files:
        return jsonify({'error': '请上传文件'}), 400
    f = request.files['file']
    try:
        if f.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(f.read()), encoding='utf-8')
        else:
            df = pd.read_excel(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({'error': f'解析失败: {e}'}), 400
    col = next((c for c in ['content','text','评论'] if c in df.columns), None)
    if not col:
        return jsonify({'error': '需要 content 或 text 列'}), 400
    texts = [t for t in df[col].fillna('').astype(str).tolist() if len(t.strip()) > 2]
    if not texts:
        return jsonify({'error': '没有有效文本'}), 400
    pred = get_predictor()
    results = []
    for i in range(0, len(texts), 32):
        results.extend(pred.predict(texts[i:i+32]))
    return jsonify({'total': len(results), 'results': results})

@app.route('/api/history')
def history():
    return jsonify(history_records[-100:])

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print('=' * 50)
    print('  餐饮评论情感分析系统')
    print('  http://localhost:5001')
    print('=' * 50)
    app.run(debug=True, host='0.0.0.0', port=5001)