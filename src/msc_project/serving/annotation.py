"""Human-only local annotation storage and loopback HTTP interface."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

SENTIMENTS = {"negative", "neutral", "positive"}
ZH = {
    "Account management: Account access": ("账户访问", "登录、密码、注册、验证、账户被锁定，以及恢复账户访问。"),
    "Company brand: Competitor": ("竞品比较", "提到、比较、偏好或考虑其他公司、品牌、替代服务；按评论对该比较方面表达的态度判断情感。"),
    "Company brand: General satisfaction": ("整体满意度", "对整个公司、服务或整体体验的评价；不要把单一功能评价自动扩大为整体满意度。"),
    "Company brand: Reviews": ("评论与评分", "阅读、发布、索取或回复评价，以及评分、口碑；不是因为当前文字本身是一条评论就选择。"),
    "Logistics rides: Speed": ("物流或行程速度", "行程、乘车、配送、取件或物流的快慢、延误、等待与准时性；一般银行转账速度不自动归入此类。"),
    "Online experience: App website": ("App 或网站体验", "应用、网站、网页或在线界面的导航、加载、崩溃、故障及在线功能。"),
    "Purchase booking experience: Ease of use": ("交易或预订便利性", "购买、下单、预订、结账或完成交易流程的容易、困难、清晰与便利程度。"),
    "Staff support: Attitude of staff": ("工作人员态度", "员工或客服的行为、帮助程度、友善、粗鲁、专业、知识或同理心。"),
    "Staff support: Email": ("邮件客服", "通过邮件联系或获得支持，包括发送、回复、响应时间、未回复及交流质量。"),
    "Staff support: Phone": ("电话客服", "通过电话联系或获得支持，包括通话、呼叫中心、等候、回拨及接听。"),
    "Value: Discounts promotions": ("折扣与促销", "折扣、促销、优惠券、代金券、特别优惠、会员优惠及宣传的降价。"),
    "Value: Price value for money": ("价格与性价比", "通常价格、成本、收费、手续费、贵便宜、负担能力及是否物有所值。"),
}


class AnnotationStore:
    def __init__(self, dataset: Path, taxonomy: Path, storage: Path) -> None:
        rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = {row["request_id"]: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("Duplicate annotation IDs")
        self.order = list(self.rows)
        self.taxonomy = json.loads(taxonomy.read_text(encoding="utf-8"))
        self.allowed = {x["aspect"] for x in self.taxonomy}
        self.storage = storage
        self.storage.mkdir(parents=True, exist_ok=True)
        self.path = storage / "annotations.json"
        self.lock = threading.RLock()
        self.annotations = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if set(self.annotations) - set(self.rows):
            raise ValueError("Stored annotation IDs do not belong to this dataset")

    def group_ids(self, group: str) -> list[str]:
        if group not in {"pilot", "adaptation", "calibration", "locked", "all"}:
            raise ValueError("Unknown annotation group")
        return [i for i in self.order if group == "all" or
                (group == "pilot" and self.rows[i].get("pilot")) or self.rows[i]["split"] == group]

    def bootstrap(self, group: str) -> dict:
        with self.lock:
            ids = self.group_ids(group)
            return {"group": group, "ids": ids, "taxonomy": self.taxonomy,
                    "total": len(ids), "confirmed": sum(self.annotations.get(i, {}).get("status") == "confirmed" for i in ids),
                    "statuses": {i: self.annotations.get(i, {}).get("status", "unlabelled") for i in ids},
                    "scope": "Human labels only; ratings and model predictions are hidden"}

    def item(self, identity: str) -> dict:
        with self.lock:
            if identity not in self.rows:
                raise KeyError(identity)
            row = self.rows[identity]
            return {"request_id": identity, "text": row["text"], "app": row["app"],
                    "split": row["split"], "pilot": row.get("pilot", False),
                    "annotation": self.annotations.get(identity)}

    def save(self, value: dict[str, Any]) -> dict:
        allowed_keys = {"request_id", "expected_revision", "reviewer", "status", "labels", "no_defined_theme",
                        "ambiguous", "out_of_scope", "incomplete_text", "english_confirmed", "note"}
        if not isinstance(value, dict) or set(value) - allowed_keys:
            raise ValueError("Unexpected annotation fields")
        identity = value.get("request_id")
        if identity not in self.rows:
            raise ValueError("Unknown review ID")
        if value.get("status") not in {"confirmed", "needs_review"}:
            raise ValueError("Invalid status")
        if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
            raise ValueError("请填写标注人昵称")
        labels = value.get("labels")
        if not isinstance(labels, list):
            raise ValueError("Invalid labels")
        pairs = set()
        for pair in labels:
            if not isinstance(pair, list) or len(pair) != 2 or pair[0] not in self.allowed or pair[1] not in SENTIMENTS:
                raise ValueError("Invalid aspect/sentiment pair")
            if tuple(pair) in pairs:
                raise ValueError("Duplicate label pair")
            pairs.add(tuple(pair))
        for flag in ("no_defined_theme", "ambiguous", "out_of_scope", "incomplete_text", "english_confirmed"):
            if not isinstance(value.get(flag), bool):
                raise ValueError(f"Missing boolean flag: {flag}")
        if labels and value["no_defined_theme"]:
            raise ValueError("选择了主题时不能勾选没有12类主题")
        if value["status"] == "confirmed":
            if not labels and not value["no_defined_theme"]:
                raise ValueError("请标主题，或明确勾选没有12类主题")
            if value["ambiguous"] or not value["english_confirmed"]:
                raise ValueError("有歧义或非英文条目请先保存为待复核")
        if not isinstance(value.get("note", ""), str) or len(value.get("note", "")) > 5000:
            raise ValueError("Invalid note")
        with self.lock:
            old = self.annotations.get(identity)
            revision = old["revision"] if old else 0
            if value.get("expected_revision") != revision:
                raise ValueError("此条已在另一窗口更新，请重新加载后再保存")
            saved = {key: val for key, val in value.items() if key != "expected_revision"}
            saved.update({"reviewer": value["reviewer"].strip(), "revision": revision + 1,
                          "saved_at": datetime.now(timezone.utc).isoformat(),
                          "source": "human_annotation", "split": self.rows[identity]["split"]})
            updated = dict(self.annotations, **{identity: saved})
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
            # History is append-only and contains no original review text.
            with (self.storage / "annotation_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"previous": old, "current": saved}, ensure_ascii=False) + "\n")
            self.annotations = updated
            return saved


def make_server(store: AnnotationStore, port: int = 8769) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def reply(self, status: int, body: object, content_type: str = "application/json; charset=utf-8") -> None:
            data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.reply(200, PAGE, "text/html; charset=utf-8")
                elif parsed.path == "/api/bootstrap":
                    self.reply(200, store.bootstrap(query.get("group", ["pilot"])[0]))
                elif parsed.path == "/api/item":
                    self.reply(200, store.item(query.get("id", [""])[0]))
                elif parsed.path == "/health":
                    self.reply(200, {"ready": True, "kind": "human_annotation"})
                else:
                    self.reply(404, {"error": "Not found"})
            except (KeyError, ValueError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            valid_origin = f"http://127.0.0.1:{self.server.server_port}"
            if origin != valid_origin or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.reply(403, {"error": "Only this local annotation page may save labels"})
                return
            if self.path != "/api/save":
                self.reply(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid body size")
                value = json.loads(self.rfile.read(size))
                self.reply(200, store.save(value))
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chattermill 评论人工标注</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#172434;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1250px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 10px}.sub{color:#506477}.toolbar,.card{background:white;border:1px solid #dde4ec;border-radius:12px;padding:18px;margin:14px 0}.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap}button,select,input[type=text]{font:inherit;border:1px solid #bccad8;border-radius:7px;padding:7px 11px;background:white}button{cursor:pointer}button.primary{background:#1966b3;color:white;border-color:#1966b3}button:disabled{opacity:.45}.grid{display:grid;grid-template-columns: minmax(300px, .8fr) minmax(500px,1.3fr);gap:18px}.review{white-space:pre-wrap;overflow-wrap:anywhere;font-size:18px;line-height:1.8}.sticky{position:sticky;top:16px;align-self:start}.theme{padding:12px 0;border-bottom:1px solid #e2e8ef}.theme:last-child{border:0}.theme strong{font-size:17px}.hint{font-size:13px;color:#56687b}.options{display:flex;gap:18px;padding-top:6px}.options label{cursor:pointer}input[type=checkbox]{width:17px;height:17px;vertical-align:middle}textarea{width:100%;min-height:75px;font:inherit;border:1px solid #bac9d6;border-radius:6px;padding:8px}.flags label{display:block;margin:7px 0}.message{padding:10px;border-radius:7px;color:#255736;background:#e7f5eb}.message.error{background:#ffecec;color:#9a2828}.footer{position:sticky;bottom:0;background:#fff;padding:12px;border-top:1px solid #d7e0e9;display:flex;gap:10px;flex-wrap:wrap}.badge{background:#edf3fa;padding:3px 8px;border-radius:6px;font-size:13px}details{padding:8px 0}summary{cursor:pointer;color:#235c92}@media(max-width:900px){.grid{grid-template-columns:1fr}.sticky{position:static}main{padding:12px}}
</style><main><h1>Chattermill 评论人工标注</h1><div class="sub">先完成首批 50 条，确定标注口径。每条检查全部 12 个主题，只给实际出现的主题选情感。进度保存在本机。</div>
<details><summary>展开标注说明</summary><p>一条评论可以涉及多个主题；同一主题明确包含两种不同情感时可多选。正面＝赞扬或满意；负面＝抱怨或不满；中性＝提到该主题但没有明确褒贬，不能用“中性”表示“不确定”。没有出现的主题三个选项都不选。</p><p>不要根据星级、来源应用或常识补充原文没有表达的内容。本页隐藏星级与模型预测。泛泛的“很好”通常属于整体满意度；不能因为来自 App 商店就自动选择 App 体验。邮箱、电话等渠道只有原文提到才选择。</p><p>不确定时勾选“有歧义”，写短备注，保存待复核。纯范围外评论勾选“没有12类主题”和“含范围外内容”；范围外内容也可能与体系内主题同时存在。非英文条目保存待复核。</p></details>
<div class="toolbar"><label>批次 <select id="group"><option value="pilot">首批 50 条</option><option value="adaptation">改进集 200 条</option><option value="calibration">校准集 200 条</option><option value="locked">最终评估集 400 条</option></select></label><label>标注人昵称 <input id="reviewer" type="text" placeholder="例如 Jialin"></label><span id="progress"></span><button id="resume">跳到首条未确认</button></div>
<div id="message" class="message">正在读取…</div><div class="grid"><div class="card sticky"><span id="position" class="badge"></span> <span id="source" class="badge"></span><div id="review" class="review"></div><div id="saved" class="hint"></div></div><div class="card"><div id="themes"></div><div class="flags"><label><input type="checkbox" id="english">确认原文可作为英文评论理解</label><label><input type="checkbox" id="none">没有涉及上述 12 类主题</label><label><input type="checkbox" id="outside">含有主题体系以外的内容</label><label><input type="checkbox" id="ambiguous">有歧义，需要讨论</label><label><input type="checkbox" id="incomplete">原文信息不完整</label></div><textarea id="note" placeholder="可选：简短说明判断依据或疑问"></textarea></div></div>
<div class="footer"><button id="prev">上一条</button><button id="next">跳过／下一条</button><button id="draft">保存待复核</button><button id="confirm" class="primary">确认标签并保存下一条</button></div></main>
<script>
let boot=null,item=null,index=0,dirty=false,busy=false;const $=id=>document.getElementById(id);function msg(t,error=false){$('message').textContent=t;$('message').className='message'+(error?' error':'')}
async function api(path,body){const r=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const v=await r.json();if(!r.ok)throw Error(v.error||r.status);return v}
function renderThemes(){const root=$('themes');root.replaceChildren();boot.taxonomy.forEach((t,i)=>{const d=document.createElement('div');d.className='theme';const title=document.createElement('strong');title.textContent=t.zh_name;d.append(title);const en=document.createElement('div');en.className='hint';en.textContent=t.aspect;d.append(en);const desc=document.createElement('div');desc.className='hint';desc.textContent=t.zh_description;d.append(desc);const details=document.createElement('details'),summary=document.createElement('summary'),def=document.createElement('div');summary.textContent='英文定义';def.textContent=t.definition;def.className='hint';details.append(summary,def);d.append(details);const opts=document.createElement('div');opts.className='options';[['negative','负面'],['neutral','中性'],['positive','正面']].forEach(([s,n])=>{const l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.dataset.aspect=t.aspect;c.dataset.sentiment=s;c.onchange=()=>{dirty=true};l.append(c,document.createTextNode(' '+n));opts.append(l)});d.append(opts);root.append(d)})}
async function load(){if(!boot.ids.length){msg('当前批次没有条目');return}item=await api('/api/item?id='+encodeURIComponent(boot.ids[index]));$('review').textContent=item.text;$('position').textContent=(index+1)+' / '+boot.total;$('source').textContent=item.app+' · '+item.split;const a=item.annotation||{};document.querySelectorAll('[data-aspect]').forEach(c=>c.checked=(a.labels||[]).some(p=>p[0]===c.dataset.aspect&&p[1]===c.dataset.sentiment));for(const [id,key] of [['english','english_confirmed'],['none','no_defined_theme'],['outside','out_of_scope'],['ambiguous','ambiguous'],['incomplete','incomplete_text']])$(id).checked=Boolean(a[key]);$('note').value=a.note||'';$('saved').textContent=a.saved_at?'已保存：'+a.saved_at+' · '+a.status:'尚未标注';$('progress').textContent='本批已确认 '+boot.confirmed+' / '+boot.total;$('prev').disabled=index===0;$('next').disabled=index===boot.total-1;dirty=false;msg('请阅读左侧原文，检查全部主题后保存。')}
async function init(){boot=await api('/api/bootstrap?group='+$('group').value);renderThemes();index=Math.max(0,boot.ids.findIndex(i=>boot.statuses[i]!=='confirmed'));await load()}
async function move(delta){if(dirty&&!confirm('当前修改未保存。确定离开这条吗？'))return;index=Math.max(0,Math.min(boot.ids.length-1,index+delta));await load()}
async function save(status){if(busy||!item)return;busy=true;try{const labels=Array.from(document.querySelectorAll('[data-aspect]:checked')).map(c=>[c.dataset.aspect,c.dataset.sentiment]);const value={request_id:item.request_id,expected_revision:item.annotation?.revision||0,reviewer:$('reviewer').value,status,labels,no_defined_theme:$('none').checked,ambiguous:$('ambiguous').checked,out_of_scope:$('outside').checked,incomplete_text:$('incomplete').checked,english_confirmed:$('english').checked,note:$('note').value};const saved=await api('/api/save',value);localStorage.setItem('chattermillReviewer',$('reviewer').value);item.annotation=saved;boot=await api('/api/bootstrap?group='+$('group').value);dirty=false;if(status==='confirmed'&&index<boot.ids.length-1){index++;await load();msg('上一条已保存。继续下一条。')}else{await load();msg(status==='confirmed'?'已保存。本批最后一条已完成；可检查是否有跳过的条目。':'已保存为待复核，可以继续下一条。')}}catch(e){msg(e.message,true)}finally{busy=false}}
$('reviewer').value=localStorage.getItem('chattermillReviewer')||'';$('group').onchange=()=>{if(dirty&&!confirm('未保存的修改将丢失，继续切换？'))return;init().catch(e=>msg(e.message,true))};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('draft').onclick=()=>save('needs_review');$('confirm').onclick=()=>save('confirmed');$('resume').onclick=async()=>{if(dirty&&!confirm('未保存的修改将丢失，继续？'))return;boot=await api('/api/bootstrap?group='+$('group').value);index=Math.max(0,boot.ids.findIndex(i=>boot.statuses[i]!=='confirmed'));await load()};document.querySelectorAll('.flags input,#note').forEach(e=>e.onchange=()=>dirty=true);window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});init().catch(e=>msg(e.message,true));
</script></html>'''
