"""多策略 RAG 文件問答系統 · PWA + RWD 版
Gradio (UI) + FastAPI (PWA 路由) + ChromaDB + Groq，可直接部署到 Render。
"""
import csv, functools, io, json, os, re, shutil, tempfile, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import uvicorn
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ═══════════ 設定（皆可用環境變數覆寫）═══════════
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")  # 可換 multilingual-e5-base、BAAI/bge-m3（較吃記憶體）
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
ENV_KEY = os.getenv("GROQ_API_KEY", "").strip()           # 伺服器預設 Key（可不設，讓使用者自行輸入）
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(tempfile.gettempdir(), "chroma_db"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
PORT = int(os.getenv("PORT", "7860"))
APP_NAME, APP_SHORT, THEME_COLOR = "RAG 文件問答", "RAG 問答", "#059669"

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf); buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p; fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


# ═══════════ 全站共用：嵌入模型 + ChromaDB（延遲載入，讓服務先開 port）═══════════
class Shared:
    lock, enc_lock = threading.Lock(), threading.Lock()
    embedder = db = col = ccol = None
    e5 = "e5" in EMBED_MODEL.lower()

    @classmethod
    def init(cls):
        with cls.lock:
            if cls.embedder is None:
                import chromadb
                from sentence_transformers import SentenceTransformer
                print(f"[init] 載入嵌入模型 {EMBED_MODEL} …", flush=True)
                shutil.rmtree(CHROMA_DIR, ignore_errors=True)  # 容器重啟後舊的 session 資料已無用
                emb = SentenceTransformer(EMBED_MODEL, device="cpu")
                emb.max_seq_length = min(512, emb.max_seq_length or 512)
                cls.db = chromadb.PersistentClient(path=CHROMA_DIR)
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40]
                cls.col = cls.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
                cls.ccol = cls.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
                cls.embedder = emb  # 最後才設定，確保其他欄位都就緒
                print("[init] 完成", flush=True)
        return cls

    @classmethod
    def enc(cls, texts, query=False):
        cls.init()
        if cls.e5: texts = [("query: " if query else "passage: ") + t for t in texts]
        with cls.enc_lock:
            return cls.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()


# ═══════════ 每位使用者（session）各自的 RAG 狀態 ═══════════
class RAG:
    def __init__(self):
        self.client, self.model, self.sid, self.name = None, MODELS[0], None, ""
        self.chunks, self.labels, self.n_child, self.tfidf, self.tm = [], [], 0, None, None
        self.set_key(ENV_KEY)

    def close(self):  # session 結束或換檔時清掉向量
        if self.sid and Shared.col is not None:
            for c in (Shared.col, Shared.ccol):
                try: c.delete(where={"sid": self.sid})
                except Exception: pass
        self.sid, self.chunks, self.labels, self.n_child = None, [], [], 0

    # ── Groq ──
    def set_key(self, key):
        self.client = Groq(api_key=key.strip()) if key and key.strip() else None

    def list_models(self):
        if not self.client: return []
        try:
            ids = [m.id for m in self.client.models.list().data if getattr(m, "active", True) is not False]
        except Exception: return []
        return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))

    def llm(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client: return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try: r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower(): raise
                kw.pop("reasoning_effort", None); kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err: raise
            return None

    # ── 文件載入 ──
    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"): return "✗ 僅支援 PDF / DOCX"
        if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024: return f"✗ 檔案超過 {MAX_UPLOAD_MB:g} MB 上限"
        try:
            segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
            pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
            if not pairs: return "✗ 無可擷取文字（可能是掃描檔，需先 OCR）"
            Shared.init(); self.close()
            self.sid, self.name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}", path.name
            self.chunks, self.labels = [p[0] for p in pairs], [p[1] for p in pairs]
            self._add(Shared.col, [f"{self.sid}_{i}" for i in range(len(pairs))], self.chunks,
                      [{"sid": self.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{self.sid}_{p}_{c}", t, p) for p, par in enumerate(self.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            self.n_child = len(kids)
            self._add(Shared.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": self.sid, "idx": k[2]} for k in kids])
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            self.tm = self.tfidf.fit_transform(self.chunks)
            return f"✓ {self.name}\n{len(self.chunks)} 片段 · {time.time() - t0:.1f}s"
        except Exception as e:
            return f"✗ 載入失敗：{type(e).__name__}: {e}"

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc); txt = para.text.strip()
                if not txt: continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush(); head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1; rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t): cells.append(t)
                    if cells: rows.append(" | ".join(cells))
                if rows: flush(); segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=Shared.enc(docs[i:i + b]))

    def vec(self, qs, k, child=False):
        total = self.n_child if child else len(self.chunks)
        if not total: return [[] for _ in qs]
        res = (Shared.ccol if child else Shared.col).query(query_embeddings=Shared.enc(qs, True), n_results=max(1, min(k, total)),
                                                           where={"sid": self.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    def kw(self, q, k):
        s = (self.tm @ self.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)): sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, q, strategy, k):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec([q], n * 2)[0], self.kw(q, n * 2)], n)
        if strategy == "semantic": idx = self.vec([q], k)[0]
        elif strategy == "tfidf": idx = self.kw(q, k)
        elif strategy == "parent_child": idx = list(dict.fromkeys(self.vec([q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = self.llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec([q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = self.llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec([q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12)); idx = cands[:k]
            if self.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {self.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = self.llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                             15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else: idx = hybrid(k)
        hits = [(i, self.chunks[i]) for i in idx]
        if strategy == "compress" and self.client and hits:
            def comp(h):
                r = self.llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex: hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model):
        if not self.chunks: return "請先上傳 PDF 或 DOCX 文件。", ""
        self.model, t0 = model or MODELS[0], time.time()
        hits = self.retrieve(q, strategy, int(k)); t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{self.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX: break
            blocks.append(b); used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {self.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits: return "找不到相關片段，請換個問法或策略。", src
        if not self.client: return "⚠ 尚未設定 Groq API Key（檢索結果請看下方片段）", src
        try:
            ans = self.llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                           f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                           1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        except Exception as e:
            return f"✗ 生成失敗：{type(e).__name__}: {e}", src
        return f"{ans or '（模型未回傳內容）'}\n\n<sub>{self.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s</sub>", src


# ═══════════ 匯出：Word / CSV ═══════════
def _out_path(ext):
    return os.path.join(tempfile.gettempdir(), f"RAG問答紀錄_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}.{ext}")


def export_csv(records):
    if not records: gr.Warning("目前沒有問答紀錄可匯出"); return None
    path = _out_path("csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟中文不亂碼
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    return path


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part: par.add_run(part).bold = bool(i % 2)


def export_docx(records, with_src=True):
    if not records: gr.Warning("目前沒有問答紀錄可匯出"); return None
    doc = Document()
    st = doc.styles["Normal"]; st.font.name, st.font.size = "Microsoft JhengHei", Pt(11)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip(): continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line): doc.add_heading(m.group(1).replace("**", ""), level=2); continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"); _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number"); _md_runs(par, m.group(1))
            else: _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph(); run = h.add_run("參考片段"); run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip()); p.paragraph_format.left_indent = Pt(12)
                for run in p.runs: run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    path = _out_path("docx"); doc.save(path)
    return path


# ═══════════ PWA：manifest、Service Worker、圖示、離線頁 ═══════════
MANIFEST = {
    "id": "/", "name": APP_NAME, "short_name": APP_SHORT, "lang": "zh-TW",
    "description": "上傳 PDF / DOCX，選擇檢索策略，直接提問",
    "start_url": "/", "scope": "/", "display": "standalone", "orientation": "any",
    "background_color": "#ffffff", "theme_color": THEME_COLOR, "categories": ["productivity", "education"],
    "icons": [
        {"src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
        {"src": "/icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
    ],
}

# 只攔截「頁面導覽」與圖示；Gradio 的 API / SSE / 上傳一律不碰，避免快取造成錯誤
SW_JS = """const VER='rag-pwa-v1', SHELL=['/offline','/icons/icon-192.png','/icons/icon-512.png'];
self.addEventListener('install',e=>{e.waitUntil(caches.open(VER).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==VER).map(k=>caches.delete(k)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{
  const r=e.request; if(r.method!=='GET') return;
  const u=new URL(r.url); if(u.origin!==location.origin) return;
  if(r.mode==='navigate'){ e.respondWith(fetch(r).catch(()=>caches.match('/offline'))); return; }
  if(u.pathname.startsWith('/icons/')){ e.respondWith(caches.match(r).then(c=>c||fetch(r))); }
});
"""

OFFLINE_HTML = f"""<!doctype html><html lang="zh-TW"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="{THEME_COLOR}">
<title>離線中 · {APP_NAME}</title>
<style>body{{margin:0;min-height:100dvh;display:grid;place-items:center;font-family:system-ui,"Noto Sans TC",sans-serif;background:#f8fafc;color:#334155;text-align:center;padding:24px}}
h1{{font-size:22px;margin:0 0 8px}}p{{margin:0 0 20px;color:#64748b}}
button{{padding:12px 24px;border:0;border-radius:999px;background:{THEME_COLOR};color:#fff;font-size:16px}}</style></head>
<body><main><div style="font-size:48px">📡</div><h1>目前沒有網路連線</h1>
<p>RAG 問答需要連線到伺服器才能檢索與生成。</p><button onclick="location.reload()">重新連線</button></main></body></html>"""


@functools.lru_cache(maxsize=8)
def make_icon(name):
    from PIL import Image, ImageDraw
    size, rounded, scale = {"icon-192": (192, True, 1.0), "icon-512": (512, True, 1.0),
                            "icon-maskable-512": (512, False, 0.78), "apple-touch-icon": (180, False, 0.9)}[name]
    s, bg = size * 4, (5, 150, 105, 255)
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, s, s), radius=int(s * .22), fill=bg) if rounded else d.rectangle((0, 0, s, s), fill=bg)
    w, h = s * scale * .46, s * scale * .60
    x0, y0 = (s - w) / 2, (s - h) / 2
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=s * .04, fill=(255, 255, 255, 255))
    for i in range(4):
        ly, lw = y0 + h * (.2 + .18 * i), w * (.5 if i == 3 else .76)
        d.rounded_rectangle((x0 + w * .12, ly, x0 + w * .12 + lw, ly + h * .06), radius=s * .01, fill=bg)
    buf = io.BytesIO()
    im.resize((size, size), Image.LANCZOS).convert("RGB" if not rounded else "RGBA").save(buf, "PNG")
    return buf.getvalue()


HEAD = f"""
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="{THEME_COLOR}">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="{APP_SHORT}">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<link rel="apple-touch-icon" href="/icons/apple-touch-icon.png">
<link rel="icon" type="image/png" href="/icons/icon-192.png">
<script>
if ('serviceWorker' in navigator) {{
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js', {{scope: '/'}}).catch(() => {{}}));
}}
(function () {{
  let dp = null;
  const rm = () => document.getElementById('pwa-install')?.remove();
  window.addEventListener('beforeinstallprompt', e => {{
    e.preventDefault(); dp = e;
    if (document.getElementById('pwa-install')) return;
    const b = document.createElement('button');
    b.id = 'pwa-install'; b.textContent = '📲 安裝 App';
    b.onclick = async () => {{ rm(); dp.prompt(); await dp.userChoice; dp = null; }};
    (document.body || document.documentElement).appendChild(b);
  }});
  window.addEventListener('appinstalled', rm);
}})();
</script>
"""

# ═══════════ RWD 樣式 ═══════════
CSS = """
.gradio-container{max-width:1200px!important;margin:auto!important;
  padding-left:max(12px,env(safe-area-inset-left))!important;padding-right:max(12px,env(safe-area-inset-right))!important;
  padding-bottom:calc(12px + env(safe-area-inset-bottom))!important}
#hdr{padding:14px 4px 4px}
#hdr h1{font-size:clamp(20px,4.6vw,28px);margin:0;font-weight:700}
#hdr p{color:var(--body-text-color-subdued);margin:4px 0 0;font-size:clamp(12px,3.2vw,14px)}
.side{border-radius:14px!important}
#chat{height:clamp(320px,62dvh,680px)!important}
#send{min-width:90px}
button{min-height:40px}
footer{display:none!important}
#pwa-install{position:fixed;right:16px;bottom:calc(16px + env(safe-area-inset-bottom));z-index:9999;padding:12px 18px;
  border:0;border-radius:999px;background:#059669;color:#fff;font-size:14px;box-shadow:0 4px 14px rgba(0,0,0,.25)}

/* 平板 / 手機：側欄與聊天區改為上下堆疊 */
@media (max-width:860px){
  #main{flex-direction:column!important;gap:12px!important}
  #main>.column{width:100%!important;min-width:0!important;flex:none!important}
  #chat{height:clamp(300px,56dvh,560px)!important}
  textarea,input{font-size:16px!important}          /* 避免 iOS 聚焦時自動放大 */
  #actions{flex-wrap:wrap!important}
  #actions>*{flex:1 1 calc(50% - 8px)!important;min-width:calc(50% - 8px)!important}
  #actions>label{flex-basis:100%!important}
  button{min-height:44px}                            /* 觸控目標至少 44px */
}
@media (max-width:480px){ #hdr h1{font-size:20px} #send{min-width:70px} }

/* 以 PWA 獨立視窗開啟時 */
@media (display-mode:standalone){ body{padding-top:env(safe-area-inset-top)} #pwa-install{display:none} }
"""
THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Noto Sans TC"), "sans-serif"])


# ═══════════ Gradio 事件 ═══════════
def _r(st): return st if isinstance(st, RAG) else RAG()


def on_session_start(): return RAG()


def on_key(key, cur, st):
    r, own = _r(st), (key or "").strip()
    r.set_key(own or ENV_KEY)
    if not r.client: return "⚠ 尚未設定 Key", gr.update(), r
    ms = r.list_models()
    if not ms: return "✗ Key 無效或無法取得模型", gr.update(), r
    tag = "你的 Key" if own else "伺服器預設 Key"
    return f"✓ {tag} · {len(ms)} 個模型", gr.update(choices=ms, value=cur if cur in ms else (MODELS[0] if MODELS[0] in ms else ms[0])), r


def on_upload(f, st):
    r = _r(st)
    return ("⚠ 請選擇檔案" if f is None else r.load(f if isinstance(f, str) else f.name)), r


def on_ask(q, history, records, strategy, k, model, st):
    r, q = _r(st), (q or "").strip()
    if not q: return "", history, gr.update(), records, r
    ans, src = r.answer(q, strategy, k, model)
    records = records + [{"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文件": r.name, "問題": q,
                          "回答": re.sub(r"\n*<sub>.*?</sub>", "", ans, flags=re.S).strip(),
                          "策略": STRATEGIES[strategy][0], "模型": r.model,
                          "參考片段": src.split("─" * 40, 1)[-1].strip()}]
    return "", history + [{"role": "user", "content": q}, {"role": "assistant", "content": ans}], src, records, r


def build_ui():
    with gr.Blocks(title=APP_NAME, theme=THEME, css=CSS, head=HEAD) as demo:
        gr.HTML(f"<div id='hdr'><h1>📚 {APP_NAME}</h1><p>上傳 PDF / DOCX，選擇檢索策略，直接提問 · Groq × ChromaDB</p></div>")
        st = gr.State(None, delete_callback=lambda r: r.close() if isinstance(r, RAG) else None, time_to_live=4 * 3600)
        records = gr.State([])
        with gr.Row(equal_height=False, elem_id="main"):
            with gr.Column(scale=1, min_width=280, elem_classes="side"):
                with gr.Accordion("🔑 Groq API Key", open=not ENV_KEY):
                    key = gr.Textbox(type="password", show_label=False, placeholder="gsk_…（Enter 套用）")
                    key_status = gr.Markdown("✓ 使用伺服器預設 Key（也可在上方輸入自己的）" if ENV_KEY else "⚠ 尚未設定")
                model = gr.Dropdown(MODELS, value=MODELS[0], allow_custom_value=True, label="模型")
                file = gr.File(label="上傳文件", file_types=[".pdf", ".docx"], type="filepath", height=120)
                status = gr.Textbox(show_label=False, interactive=False, lines=2, placeholder="尚未載入文件")
                strategy = gr.Radio([(v[0], k) for k, v in STRATEGIES.items()], value="hybrid", label="檢索策略")
                desc = gr.Markdown(f"<small>{STRATEGIES['hybrid'][1]}</small>")
                topk = gr.Slider(1, 10, 4, step=1, label="片段數量 Top-K")
            with gr.Column(scale=2, min_width=0):
                chat = gr.Chatbot(elem_id="chat", show_label=False, type="messages", show_copy_button=True,
                                  placeholder="**上傳文件後開始提問**<br>例如：這份文件的重點是什麼？")
                with gr.Row():
                    q = gr.Textbox(show_label=False, placeholder="輸入問題，Enter 送出", scale=6, container=False)
                    send = gr.Button("送出", variant="primary", scale=1, elem_id="send")
                with gr.Row():
                    gr.Examples([["這份文件的重點是什麼？"], ["文件提到哪些規定或限制？"], ["有哪些風險或注意事項？"]], q, label=None)
                    clear = gr.Button("🗑 清除對話", size="sm", variant="secondary")
                with gr.Accordion("🔎 檢索到的片段", open=False):
                    src = gr.Textbox(show_label=False, lines=12, interactive=False)
                with gr.Row(elem_id="actions"):
                    with_src = gr.Checkbox(True, label="Word 包含參考片段", scale=2)
                    btn_docx = gr.Button("📄 匯出 Word", size="sm", scale=1)
                    btn_csv = gr.Button("📊 匯出 CSV", size="sm", scale=1)
                out_file = gr.File(label="下載檔案", visible=False)

        demo.load(on_session_start, None, st)
        key.submit(on_key, [key, model, st], [key_status, model, st], concurrency_limit=4)
        key.blur(on_key, [key, model, st], [key_status, model, st], concurrency_limit=4)
        file.upload(on_upload, [file, st], [status, st], concurrency_limit=2)
        strategy.change(lambda s: f"<small>{STRATEGIES[s][1]}</small>", strategy, desc)
        for ev in (q.submit, send.click):
            ev(on_ask, [q, chat, records, strategy, topk, model, st], [q, chat, src, records, st], concurrency_limit=4)
        clear.click(lambda: ([], "", [], gr.update(value=None, visible=False)), None, [chat, src, records, out_file])
        show = lambda p: gr.update(value=p, visible=bool(p))
        btn_docx.click(lambda r, w: show(export_docx(r, w)), [records, with_src], out_file)
        btn_csv.click(lambda r: show(export_csv(r)), records, out_file)
    return demo.queue(default_concurrency_limit=4, max_size=32)


# ═══════════ FastAPI：PWA 路由 + 掛載 Gradio ═══════════
app = FastAPI(title=APP_NAME, docs_url=None, redoc_url=None)
NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/healthz")
def healthz(): return {"ok": True, "embedder_ready": Shared.embedder is not None}


@app.get("/manifest.webmanifest")
def manifest(): return JSONResponse(MANIFEST, media_type="application/manifest+json", headers=NO_CACHE)


@app.get("/sw.js")
def sw(): return Response(SW_JS, media_type="application/javascript", headers={**NO_CACHE, "Service-Worker-Allowed": "/"})


@app.get("/offline")
def offline(): return HTMLResponse(OFFLINE_HTML)


@app.get("/icons/{name}.png")
def icon(name: str):
    try: data = make_icon(name)
    except KeyError: return Response(status_code=404)
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


_pw = os.getenv("APP_PASSWORD", "")
app = gr.mount_gradio_app(app, build_ui(), path="/", auth=(os.getenv("APP_USER", "admin"), _pw) if _pw else None)

if __name__ == "__main__":
    threading.Thread(target=Shared.init, daemon=True).start()  # 背景預載模型，不阻擋服務啟動
    uvicorn.run(app, host="0.0.0.0", port=PORT, proxy_headers=True, forwarded_allow_ips="*")
