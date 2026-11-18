# 📚 RAG 文件問答（PWA + RWD 版）

上傳 PDF / DOCX，選擇檢索策略後直接提問。支援 **8 種檢索策略**、**Word / CSV 匯出**，
可**安裝成手機或桌面 App（PWA）**，介面隨螢幕寬度自動調整（RWD）。

**技術組合**：Gradio 5 · FastAPI · ChromaDB · sentence-transformers（multilingual-e5）· Groq

## 功能

| 類別 | 說明 |
|---|---|
| 檢索策略 | 混合搜尋（RRF）、語意、關鍵字（TF-IDF）、重新排序、多查詢、壓縮、父子文檔、HyDE |
| 文件格式 | PDF（依頁）、DOCX（依標題與表格分段） |
| 匯出 | Word（保留條列與粗體）、CSV（UTF-8 BOM，Excel 不亂碼） |
| PWA | Web App Manifest、Service Worker、離線提示頁、可安裝圖示（含 maskable / iOS） |
| RWD | 手機 / 平板 / 桌面三段版型、`dvh` 高度、44px 觸控目標、iOS 安全區域、避免 iOS 輸入框放大 |
| 多人使用 | 每位使用者獨立的文件與對話，互不覆蓋；閒置 4 小時自動清除向量資料 |

## 專案結構

```
.
├── app.py             # 主程式：FastAPI（PWA 路由）+ Gradio（介面）+ RAG
├── requirements.txt
├── render.yaml        # Render Blueprint
├── .gitignore
└── README.md
```

PWA 所需的 `manifest`、`sw.js`、圖示都由 `app.py` 動態提供，不需另外放靜態檔：

| 路徑 | 用途 |
|---|---|
| `/manifest.webmanifest` | 應用程式資訊 |
| `/sw.js` | Service Worker（只處理頁面導覽與圖示，不干擾 Gradio API） |
| `/offline` | 離線時顯示的頁面 |
| `/icons/*.png` | 192 / 512 / maskable / apple-touch 圖示（啟動時以 Pillow 產生） |
| `/healthz` | Render 健康檢查 |

## 本機執行

```bash
python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
pip install -r requirements.txt
export GROQ_API_KEY=gsk_xxx        # 選填；Windows PowerShell：$env:GROQ_API_KEY="gsk_xxx"
python app.py
```

開啟 http://localhost:7860 。第一次啟動會下載嵌入模型（約 470 MB）。

> PWA 的安裝與 Service Worker 需要 **HTTPS**（`localhost` 例外），所以本機可測試，部署後在 Render 的 https 網址上也能正常使用。

## 上傳到 GitHub

```bash
git init
git add .
git commit -m "RAG PWA"
git branch -M main
git remote add origin https://github.com/<你的帳號>/<repo名稱>.git
git push -u origin main
```

## 部署到 Render

### 方式 A：Blueprint（推薦）
1. 登入 [Render](https://render.com) → **New → Blueprint**，選擇你的 GitHub repo。
2. Render 會讀取 `render.yaml`，依提示填入 `GROQ_API_KEY`（可留空）、`APP_PASSWORD`（可留空）。
3. 按 **Apply**，等待建置完成（首次約 5–10 分鐘，含下載模型）。

### 方式 B：手動建立 Web Service
| 欄位 | 值 |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"` |
| Start Command | `python app.py` |
| Health Check Path | `/healthz` |
| Instance Type | **Standard（2 GB）以上** |

環境變數：`PYTHON_VERSION=3.11.9`、`HF_HOME=/opt/render/project/src/.hf_cache`，其餘見下表。

### 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `GROQ_API_KEY` | 空 | 伺服器預設 Key。**公開網站請留空**，否則任何人都會消耗你的額度 |
| `APP_PASSWORD` | 空 | 設定後整站需要帳密登入 |
| `APP_USER` | `admin` | 登入帳號 |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | 嵌入模型；改用其他模型時請同步修改 build command 的預載名稱 |
| `MAX_UPLOAD_MB` | `20` | 單檔大小上限 |
| `CHROMA_DIR` | 系統暫存目錄 | 向量資料位置（重啟後會清空） |
| `PORT` | Render 自動注入 | 服務埠號 |

## 安裝成 App

- **Android / Chrome / Edge**：開啟網站後點右下角「📲 安裝 App」，或瀏覽器選單 →「安裝應用程式」。
- **iPhone / iPad（Safari）**：分享按鈕 →「加入主畫面」。
- **桌面 Chrome / Edge**：網址列右側的安裝圖示。

## 注意事項

- **記憶體**：torch 加嵌入模型執行時約佔 1–1.5 GB。Render 512 MB 的方案很可能 OOM，請選 2 GB 以上，或改用較小的嵌入模型。
- **資料不持久**：Render 的檔案系統是暫時性的，重新部署或重啟後向量資料會清空，使用者需重新上傳文件（程式本來就是以單次 session 設計）。
- **休眠**：免費或低階方案閒置後可能休眠，喚醒後需要重新載入模型，首次請求會較慢；`/healthz` 的 `embedder_ready` 欄位可看模型是否已就緒。
- **離線**：PWA 只快取圖示與離線提示頁。檢索與生成都需要連線到伺服器與 Groq，因此離線時只會看到提示。
- **掃描版 PDF**：沒有文字層的檔案無法擷取內容，需先 OCR。

## 疑難排解

| 現象 | 處理 |
|---|---|
| 部署後被重啟、日誌出現 `Killed` | 記憶體不足，升級方案 |
| 建置逾時或太久 | 確認 `requirements.txt` 第一行的 CPU 版 torch index 有生效 |
| 「模型未回傳內容」 | 換模型，或把 Top-K 調小 |
| 手機看不到安裝按鈕 | 需 HTTPS；iOS 沒有安裝按鈕，請用「加入主畫面」 |
| 更新後畫面沒變 | Service Worker 版本在 `app.py` 的 `SW_JS`（`VER`），改版號即可強制更新 |
