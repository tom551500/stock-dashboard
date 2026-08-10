# -*- coding: utf-8 -*-
"""
台股前50大市值 正逆價差 vs 大盤漲跌 分析工具
================================================
延伸自 investor_conference_analysis.py 的邏輯：
  1. 即時用「股價 x 股本」算出市值，取前 N 大市值個股
  2. 對照台灣期交所「股票期貨標的」清單，找出每檔個股對應的股票期貨代碼
  3. 抓取現貨收盤價 與 個股期貨（近月）收盤價，計算「正逆價差」
  4. 兩個分頁：
     - 每日快照儀表板：像原本截圖那樣，看今天前50大市值個股的正逆價差狀況
     - 歷史回測分析：正逆價差擴散指標 vs 隔日加權指數漲跌，統計勝率/相關性

資料來源：FinMind API (https://finmindtrade.com)
          台灣期貨交易所 股票期貨標的清單 (https://www.taifex.com.tw/cht/2/stockLists)

執行方式：
  streamlit run top50_futures_spread_app.py

需要的 API Token：
  在 Streamlit Cloud 的 App -> Settings -> Secrets 加入：
  FINMIND_TOKEN = "你的 FinMind token"cd ..
  （免費註冊 https://finmindtrade.com 即可取得，可提升 API 上限到 600 次/小時）
"""

import time
import datetime as dt
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------------
st.set_page_config(page_title="台股前50大市值 正逆價差 vs 大盤", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
TAIFEX_STOCKLIST_URL = "https://www.taifex.com.tw/cht/2/stockLists"
TWSE_STOCK_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TWSE_CAPITAL_CSV_URL = "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}

TOKEN = st.secrets.get("FINMIND_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

REQUEST_SLEEP = 0.4  # 放慢節奏，避免觸發瞬間流量限制（跟總額度是兩回事）


# ------------------------------------------------------------------
# 基礎資料抓取函式
# ------------------------------------------------------------------
def finmind_get(dataset: str, data_id: str = None, start_date: str = None,
                 end_date: str = None, silent: bool = False, max_retries: int = 3) -> pd.DataFrame:
    """呼叫 FinMind /v4/data，回傳 DataFrame（失敗回傳空表）。
    對 402(額度用盡)/403(暫時被擋)/429(太頻繁) 會自動退避重試。
    """
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(FINMIND_URL, headers=HEADERS, params=params, timeout=20)
            if resp.status_code in (402, 403, 429):
                wait = 3 * (attempt + 1)
                time.sleep(wait)
                last_err = requests.exceptions.HTTPError(
                    f"{resp.status_code} {'額度用盡/太頻繁' if resp.status_code != 403 else '暫時被拒絕(可能觸發流量限制)'} "
                    f"for url: {resp.url}"
                )
                continue
            resp.raise_for_status()
            js = resp.json()
            data = js.get("data", [])
            time.sleep(REQUEST_SLEEP)
            return pd.DataFrame(data)
        except Exception as e:
            last_err = e
            time.sleep(REQUEST_SLEEP)

    if not silent and last_err is not None:
        st.warning(f"抓取 {dataset} ({data_id}) 失敗（已重試{max_retries}次）：{last_err}")
    return pd.DataFrame()


TAIFEX_HEADERS = {
    # 台灣期交所會擋掉沒有瀏覽器標頭的請求，這裡模擬一般瀏覽器
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Referer": "https://www.taifex.com.tw/cht/2/stockLists",
}


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def get_stock_futures_mapping() -> pd.DataFrame:
    """
    抓取台灣期交所「股票期貨/股票選擇權 交易標的」清單，
    整理出 股票代號 <-> 股票期貨商品代碼 對照表（只取標準型 2,000 股合約，排除 ETF）。
    """
    # 注意：這個函式故意用 raise 而不是 return 空表 + st.error 來處理失敗，
    # 因為 @st.cache_data 只有在「正常 return」時才會快取結果；
    # 用 raise 的話失敗不會被快取，下次按「重新整理」才會真的重新連線，
    # 而不是一直吃到12小時前失敗時存下來的空結果。
    try:
        resp = requests.get(TAIFEX_STOCKLIST_URL, headers=TAIFEX_HEADERS, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        raise RuntimeError(
            f"無法連線台灣期交所取得股票期貨標的清單（連線層失敗）：{type(e).__name__}: {e}\n\n"
            "常見原因：公司/校園網路防火牆或防毒軟體擋掉 taifex.com.tw、VPN、或該網站暫時維護中。"
        ) from e

    try:
        tables = pd.read_html(StringIO(resp.text))
    except Exception as e:
        raise RuntimeError(
            f"有連上台灣期交所（HTTP {resp.status_code}），但網頁表格解析失敗：{type(e).__name__}: {e}\n\n"
            "可能是網站改版導致表格結構變了，或缺少 lxml 套件（請確認 requirements.txt 已安裝 lxml，"
            "並執行過 pip install -r requirements.txt）。"
        ) from e

    if not tables:
        raise RuntimeError(
            f"有連上台灣期交所（HTTP {resp.status_code}，內容長度 {len(resp.text)} 字元），"
            "但頁面裡完全沒有偵測到表格，可能是網站改版了，或是被導向到驗證/錯誤頁面。"
        )

    # 找出欄位包含「商品代碼」的那張表
    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("商品代碼" in c for c in cols):
            target = t
            break
    if target is None:
        raise RuntimeError(
            "有抓到表格，但找不到包含「商品代碼」欄位的表格，網站版面可能改了。\n"
            f"目前抓到的表格欄位分別是：{[list(t.columns) for t in tables]}"
        )

    df = target.copy()
    df.columns = [str(c) for c in df.columns]

    def _find_col(keyword_options):
        for kw in keyword_options:
            hits = [c for c in df.columns if kw in c]
            if hits:
                return hits[0]
        return None

    sec_col = _find_col(["證券代號", "股票代號"])
    name_col = _find_col(["簡稱", "證券簡稱", "股票名稱"])
    code_col = _find_col(["商品代碼"])
    lot_col = _find_col(["股數", "契約單位", "受益權單位"])
    is_futures_col = _find_col(["是否為股票期貨", "是否為期貨"])

    # 除錯用：如果關鍵欄位找不到，或篩完是空的，把整張原始表印出來，方便你貼給我對照修正
    debug_info = (
        f"偵測到的欄位對應：\n"
        f"  證券代號欄 = {sec_col}\n  簡稱欄 = {name_col}\n"
        f"  商品代碼欄 = {code_col}\n  股數/契約單位欄 = {lot_col}\n"
        f"  是否為股票期貨欄 = {is_futures_col}\n\n"
        f"原始表格欄位：{list(df.columns)}\n\n"
        f"原始表格前15列：\n{df.head(15).to_string()}"
    )

    if not all([sec_col, name_col, code_col]):
        raise RuntimeError(
            "有抓到表格，但關鍵欄位（證券代號/簡稱/商品代碼）對不齊，網站版面可能改了。\n\n"
            + debug_info
        )

    out = df[[sec_col, name_col, code_col] + ([lot_col] if lot_col else [])].copy()
    out.columns = ["stock_id", "stock_name", "futures_root"] + (["lot"] if lot_col else [])

    # 只保留 4 碼股票代號（排除 00開頭的 ETF、以及非數字的權證代號）
    out = out[out["stock_id"].astype(str).str.match(r"^\d{4}$", na=False)]
    out = out[~out["stock_id"].astype(str).str.startswith("00")]

    if is_futures_col:
        # 排除選擇權列（該欄位若明確標示「否」則排除；沒有標「否」字樣的都保留，避免誤刪）
        out_flag = df.loc[out.index, is_futures_col].astype(str)
        out = out[~out_flag.str.contains("否", na=False)]

    if lot_col:
        # 每個股票代號若有多筆合約（例如標準2,000股 + 小型100股），取股數最大的那一筆
        out["lot_num"] = out["lot"].astype(str).str.extract(r"([\d,]+)")[0].str.replace(",", "", regex=False)
        out["lot_num"] = pd.to_numeric(out["lot_num"], errors="coerce").fillna(0)
        out = out.sort_values("lot_num", ascending=False).drop_duplicates(subset="stock_id")
        out = out.drop(columns=["lot", "lot_num"])
    else:
        out = out.drop_duplicates(subset="stock_id")

    # 期交所這個「商品代碼」欄位其實是股票期貨/選擇權共用的2碼根代碼（例如台積電是"CD"），
    # 不是完整的期貨合約代碼。標準型股票期貨的完整代碼 = 根代碼 + "F"（例如"CD"+"F"="CDF"）。
    out["futures_root"] = out["futures_root"].astype(str).str.strip()
    out["futures_id"] = out["futures_root"] + "F"

    out = out.reset_index(drop=True)

    if out.empty:
        raise RuntimeError(
            "有抓到表格也對到欄位，但篩選完是空的，篩選條件可能太嚴格。\n\n" + debug_info
        )

    return out


@st.cache_data(ttl=3600 * 24, show_spinner=False)
def get_top_n_by_market_cap(stock_ids: tuple, n: int = 50) -> pd.DataFrame:
    """
    市值 = 收盤價 x (實收資本額 / 10)（台股面額多為每股10元，股數 = 資本額/10）。

    完全改用台灣證交所官方、免金鑰的公開資料，不會動用到任何 FinMind 額度：
      - 全市場當日收盤價：https://www.twse.com.tw/exchangeReport/MI_INDEX
      - 公司實收資本額：https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv
        （公開資訊觀測站 MOPS 官方開放資料）
    這樣市值排名這一步 0 次 FinMind 呼叫，額度全部留給後面的正逆價差分析。
    """
    # 1. 抓實收資本額（公司基本資料，全市場一次性資料，不分日期）
    try:
        cap_resp = requests.get(TWSE_CAPITAL_CSV_URL, headers=BROWSER_HEADERS, timeout=30)
        cap_resp.raise_for_status()
        cap_resp.encoding = cap_resp.apparent_encoding or "utf-8"
    except Exception as e:
        raise RuntimeError(
            f"無法連線公開資訊觀測站取得實收資本額資料：{type(e).__name__}: {e}"
        ) from e

    try:
        cap_df = pd.read_csv(StringIO(cap_resp.text))
    except Exception as e:
        raise RuntimeError(f"實收資本額 CSV 解析失敗：{type(e).__name__}: {e}") from e

    cap_df.columns = [str(c).strip() for c in cap_df.columns]
    id_col = next((c for c in cap_df.columns if "公司代號" in c), None)
    capital_col = next((c for c in cap_df.columns if "實收資本額" in c), None)
    if id_col is None or capital_col is None:
        raise RuntimeError(
            f"實收資本額 CSV 欄位對不到「公司代號」或「實收資本額」，網站格式可能改了。\n"
            f"目前欄位：{list(cap_df.columns)}"
        )
    cap_df = cap_df[[id_col, capital_col]].rename(columns={id_col: "stock_id", capital_col: "capital"})
    cap_df["stock_id"] = cap_df["stock_id"].astype(str).str.strip()
    cap_df["capital"] = pd.to_numeric(cap_df["capital"], errors="coerce")
    cap_df["shares"] = cap_df["capital"] / 10.0  # 面額10元 -> 股數
    cap_df = cap_df.dropna(subset=["shares"]).drop_duplicates(subset="stock_id")

    # 2. 抓全市場當日收盤價：改用證交所新版 OpenAPI（格式單純的flat陣列，不用逐日試探）
    try:
        price_resp = requests.get(TWSE_STOCK_DAY_ALL_URL, headers=BROWSER_HEADERS, timeout=30)
        price_resp.raise_for_status()
        price_json = price_resp.json()
    except Exception as e:
        raise RuntimeError(
            f"無法連線證交所 OpenAPI (STOCK_DAY_ALL) 取得全市場收盤價：{type(e).__name__}: {e}"
        ) from e

    if not isinstance(price_json, list) or not price_json:
        raise RuntimeError(
            f"證交所 OpenAPI (STOCK_DAY_ALL) 回傳的資料格式不符預期（預期是陣列）。"
            f"實際回傳型別：{type(price_json).__name__}，內容前500字：{str(price_json)[:500]}"
        )

    price_df = pd.DataFrame(price_json)
    id_col2 = next((c for c in price_df.columns if c in ("Code", "證券代號") or "代號" in c), None)
    close_col2 = next((c for c in price_df.columns if c in ("ClosingPrice", "收盤價") or "收盤價" in c), None)
    if id_col2 is None or close_col2 is None:
        raise RuntimeError(
            f"STOCK_DAY_ALL 欄位對不到代號/收盤價，目前欄位：{list(price_df.columns)}"
        )
    used_date = "最新交易日（openapi.twse.com.tw 只提供最新一天資料）"

    price_df = price_df[[id_col2, close_col2]].rename(columns={id_col2: "stock_id", close_col2: "latest_close"})
    price_df["stock_id"] = price_df["stock_id"].astype(str).str.strip()
    price_df["latest_close"] = pd.to_numeric(
        price_df["latest_close"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    price_df = price_df.dropna(subset=["latest_close"])

    if price_df.empty:
        raise RuntimeError("證交所 OpenAPI 有回應，但整理後的收盤價資料是空的。")

    # 3. 合併算市值，限縮在「有股票期貨」的候選池內，取前 N 大
    merged = pd.merge(price_df, cap_df, on="stock_id", how="inner")
    merged = merged[merged["stock_id"].isin(stock_ids)].copy()
    merged["market_value"] = merged["latest_close"] * merged["shares"]

    if merged.empty:
        raise RuntimeError(
            f"證交所資料抓到了（{used_date} 收盤價 {len(price_df)} 檔、資本額 {len(cap_df)} 檔），"
            f"但跟股票期貨候選池({len(stock_ids)}檔)兜不起來，股票代號格式可能不一致。"
        )

    df = merged.sort_values("market_value", ascending=False).head(n).reset_index(drop=True)

    info = finmind_get("TaiwanStockInfo", silent=True)
    if not info.empty:
        info = info.drop_duplicates(subset="stock_id")[["stock_id", "stock_name"]]
        df = df.merge(info, on="stock_id", how="left")
    else:
        df["stock_name"] = df["stock_id"]

    st.caption(f"✅ 市值排名完成（證交所官方資料，{used_date} 收盤價），完全沒有消耗 FinMind 額度。")
    return df


def pick_near_month_futures(fut_df: pd.DataFrame, ref_date: str) -> pd.DataFrame:
    """
    每個 date 只留下「近月」合約（contract_date 最接近但 >= 當月）
    並優先使用一般交易時段(非夜盤)的資料。

    注意：刻意不用 df.groupby(...).apply(...) 寫法，因為不同版本的 pandas
    在groupby分組欄位是否會被保留在傳入apply的子DataFrame裡，行為不一致，
    容易出現 KeyError。改用手動迴圈，明確、穩定、不依賴pandas內部細節。
    """
    if fut_df.empty:
        return fut_df
    df = fut_df.copy()
    if "trading_session" in df.columns:
        normal = df[df["trading_session"].astype(str).str.lower().isin(["position", "regular", ""])]
        if not normal.empty:
            df = normal
    df["date"] = pd.to_datetime(df["date"])
    df["contract_date"] = df["contract_date"].astype(str)

    picked_rows = []
    for date_value in df["date"].unique():
        group = df[df["date"] == date_value]
        cur_month = pd.Timestamp(date_value).strftime("%Y%m")
        candidates = group[group["contract_date"] >= cur_month]
        if candidates.empty:
            candidates = group
        picked_rows.append(candidates.sort_values("contract_date").iloc[0])

    if not picked_rows:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(picked_rows).reset_index(drop=True)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def build_spread_table(stock_ids: tuple, futures_map: dict, start_date: str,
                        end_date: str) -> pd.DataFrame:
    """
    對每一檔股票抓現貨 + 期貨近月價格，計算每日正逆價差。
    回傳長表：date, stock_id, stock_name, spot_close, futures_close, spread, sign(+1/-1)

    一樣改用「不帶 data_id，抓整個期間全市場」的免費批次模式，
    一次抓完全部股票的現貨價，再一次抓完全部期貨合約，本地端再篩選/合併，
    把原本 2 x N 檔 API 呼叫，壓縮成 2 次。
    """
    fut_ids_needed = {futures_map[sid] for sid in stock_ids if sid in futures_map}

    if not fut_ids_needed:
        st.caption("（診斷：傳入的股票清單裡，沒有任何一檔能對應到股票期貨代碼，請檢查 futures_map 是否正確。）")
        return pd.DataFrame()

    spot_all = finmind_get("TaiwanStockPrice", start_date=start_date, end_date=end_date, silent=True)
    fut_all = finmind_get("TaiwanFuturesDaily", start_date=start_date, end_date=end_date, silent=True)

    if spot_all.empty or fut_all.empty or "close" not in spot_all.columns:
        st.caption(
            f"（診斷：正逆價差全市場批次查詢沒拿到可用資料：spot_all行數={len(spot_all)}, "
            f"fut_all行數={len(fut_all)}，改用逐檔查詢備援方案...）"
        )
        return _build_spread_table_fallback(stock_ids, futures_map, start_date, end_date)

    spot_all = spot_all[spot_all["stock_id"].isin(stock_ids)].copy()
    fut_all = fut_all[fut_all["futures_id"].isin(fut_ids_needed)].copy() if "futures_id" in fut_all.columns else pd.DataFrame()

    if spot_all.empty or fut_all.empty:
        st.caption(
            f"（診斷：正逆價差全市場批次查詢篩選後是空的：spot篩選後={len(spot_all)}列, "
            f"fut篩選後={len(fut_all)}列（futures_id欄位是否存在：{'是' if 'futures_id' in fut_all.columns else '否'}）。"
            "改用逐檔查詢備援方案...）"
        )
        return _build_spread_table_fallback(stock_ids, futures_map, start_date, end_date)

    spot_all["date"] = pd.to_datetime(spot_all["date"])

    records = []
    reasons = {"無期貨代碼": 0, "現貨無資料": 0, "期貨無資料": 0, "近月合約篩選後為空": 0, "日期對不上(merge為空)": 0}
    for sid in stock_ids:
        fut_id = futures_map.get(sid)
        if not fut_id:
            reasons["無期貨代碼"] += 1
            continue
        spot = spot_all[spot_all["stock_id"] == sid]
        fut = fut_all[fut_all["futures_id"] == fut_id]
        if spot.empty:
            reasons["現貨無資料"] += 1
            continue
        if fut.empty:
            reasons["期貨無資料"] += 1
            continue
        fut_near = pick_near_month_futures(fut, end_date)
        if fut_near.empty:
            reasons["近月合約篩選後為空"] += 1
            continue
        oi_col = "open_interest" if "open_interest" in fut_near.columns else None
        fut_cols = ["date", "close"] + ([oi_col] if oi_col else [])
        fut_rename = {"close": "futures_close"}
        if oi_col:
            fut_rename[oi_col] = "open_interest"
        merged = pd.merge(
            spot[["date", "close"]].rename(columns={"close": "spot_close"}),
            fut_near[fut_cols].rename(columns=fut_rename),
            on="date", how="inner",
        )
        if merged.empty:
            reasons["日期對不上(merge為空)"] += 1
            continue
        merged["stock_id"] = sid
        merged["spread"] = merged["futures_close"] - merged["spot_close"]
        merged["spread_pct"] = merged["spread"] / merged["spot_close"] * 100
        merged["sign"] = np.where(merged["spread"] >= 0, 1, -1)
        if "open_interest" not in merged.columns:
            merged["open_interest"] = np.nan
        records.append(merged)

    if not records:
        st.caption(f"（診斷：批次模式逐檔合併後全部失敗，失敗原因統計：{reasons}）")
        return pd.DataFrame()

    st.caption(f"✅ 正逆價差批次查詢成功，共處理出 {len(records)} 檔個股的資料"
               + (f"（另有 {sum(reasons.values())} 檔因故略過：{reasons}）" if sum(reasons.values()) else ""))
    return pd.concat(records, ignore_index=True)


def _build_spread_table_fallback(stock_ids: tuple, futures_map: dict, start_date: str,
                                  end_date: str) -> pd.DataFrame:
    """備援方案：逐檔查詢（免費方案下較慢）"""
    records = []
    reasons = {"無期貨代碼": 0, "現貨無資料": 0, "期貨無資料": 0, "近月合約篩選後為空": 0, "日期對不上(merge為空)": 0}
    sample_fut_fail = []  # 記錄前幾筆「期貨無資料」的實際 (stock_id, futures_id, repr)，方便除錯
    for sid in stock_ids:
        fut_id = futures_map.get(sid)
        if not fut_id:
            reasons["無期貨代碼"] += 1
            continue
        spot = finmind_get("TaiwanStockPrice", data_id=sid, start_date=start_date, end_date=end_date, silent=True)
        fut = finmind_get("TaiwanFuturesDaily", data_id=fut_id, start_date=start_date, end_date=end_date, silent=True)
        if spot.empty:
            reasons["現貨無資料"] += 1
            continue
        if fut.empty:
            reasons["期貨無資料"] += 1
            if len(sample_fut_fail) < 5:
                sample_fut_fail.append(f"stock_id={sid!r}, futures_id={fut_id!r} (repr顯示看得出多餘空白等字元)")
            continue
        spot["date"] = pd.to_datetime(spot["date"])
        fut_near = pick_near_month_futures(fut, end_date)
        if fut_near.empty:
            reasons["近月合約篩選後為空"] += 1
            continue
        oi_col = "open_interest" if "open_interest" in fut_near.columns else None
        fut_cols = ["date", "close"] + ([oi_col] if oi_col else [])
        fut_rename = {"close": "futures_close"}
        if oi_col:
            fut_rename[oi_col] = "open_interest"
        merged = pd.merge(
            spot[["date", "close"]].rename(columns={"close": "spot_close"}),
            fut_near[fut_cols].rename(columns=fut_rename),
            on="date", how="inner",
        )
        if merged.empty:
            reasons["日期對不上(merge為空)"] += 1
            continue
        merged["stock_id"] = sid
        merged["spread"] = merged["futures_close"] - merged["spot_close"]
        merged["spread_pct"] = merged["spread"] / merged["spot_close"] * 100
        merged["sign"] = np.where(merged["spread"] >= 0, 1, -1)
        if "open_interest" not in merged.columns:
            merged["open_interest"] = np.nan
        records.append(merged)
    if not records:
        st.caption(f"（診斷：逐檔備援方案也全部失敗，失敗原因統計：{reasons}）")
        if sample_fut_fail:
            st.code("期貨代碼樣本（前5筆「期貨無資料」的實際查詢代碼）：\n" + "\n".join(sample_fut_fail))
        return pd.DataFrame()
    if sum(reasons.values()):
        st.caption(f"（逐檔備援方案：成功 {len(records)} 檔，略過 {sum(reasons.values())} 檔，原因：{reasons}）")
    return pd.concat(records, ignore_index=True)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def get_taiex(start_date: str, end_date: str) -> pd.DataFrame:
    df = finmind_get("TaiwanStockTotalReturnIndex", data_id="TAIEX",
                      start_date=start_date, end_date=end_date, silent=True)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    price_col = "price" if "price" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        return pd.DataFrame()
    df["taiex_close"] = df[price_col]
    df["taiex_return_pct"] = df["taiex_close"].pct_change() * 100
    return df[["date", "taiex_close", "taiex_return_pct"]]


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
def check_finmind_quota():
    """查詢目前 FinMind API 額度使用狀況（若有設定 token）"""
    if not TOKEN:
        return None
    try:
        resp = requests.get("https://api.web.finmindtrade.com/v2/user_info",
                             headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            js = resp.json()
            return js.get("user_count"), js.get("api_request_limit")
    except Exception:
        pass
    return None


st.title("📊 台股前50大市值 正逆價差 vs 大盤漲跌")
st.caption("市值排序（證交所官方資料計算，即時） + 個股期貨正逆價差 + 與加權指數的領先/落後關係分析（市值排名侷限在「有掛牌股票期貨」的個股範圍內，約300檔）")

if not TOKEN:
    st.info("尚未設定 FINMIND_TOKEN，將以免費額度（300次/小時，較低流量上限）呼叫 API。建議在 Secrets 中加入 FINMIND_TOKEN 以提升到600次/小時。")
else:
    quota = check_finmind_quota()
    if quota:
        used, limit = quota
        if used is not None and limit is not None:
            st.caption(f"目前 FinMind API 額度使用狀況：{used} / {limit}（每小時重置）")
            if isinstance(used, (int, float)) and isinstance(limit, (int, float)) and used >= limit * 0.9:
                st.warning("⚠️ API 額度快用完了，如果剛剛已經跑過一次「抓取資料」，可能就是因為額度不足才失敗，建議等額度重置（每小時整點重置）後再試。")

with st.sidebar:
    st.header("⚙️ 參數設定")
    top_n = st.slider("取前 N 大市值個股", 20, 80, 50, step=5)
    lookback_days = st.slider("歷史回測回溯天數（交易日，含快取抓取範圍）", 20, 250, 60, step=10)
    st.divider()
    st.caption("策略疊加參數（用於歷史回測分頁的訊號測試）")
    trend_direction = st.radio(
        "訊號方向", ["多（廣度回升）", "空（廣度下降）"], index=0,
        help="多＝前一天→當天廣度指標增加，隔日做多；空＝前一天→當天廣度指標減少，隔日做空",
    )
    trend_ma_days = st.slider("趨勢均線天數", 5, 60, 20, step=5,
                               help="多單只在加權指數收盤價高於此均線（多頭趨勢）時才算訊號；空單只在低於此均線（空頭趨勢）時才算訊號")
    breadth_change_threshold = st.slider("廣度變化門檻（前一天→當天，絕對值，>=此值才算訊號）", 10, 150, 80, step=10,
                                          help="廣度指標單日變化量的絕對值要超過這個門檻才算訊號，數字越大代表要求變化越劇烈")
    run_btn = st.button("🚀 抓取資料 / 重新整理", type="primary")

tab1, tab2 = st.tabs(["📌 每日快照儀表板", "📈 歷史回測分析"])

if run_btn or "spread_df" not in st.session_state:
    with st.spinner("抓取股票期貨標的對照表..."):
        try:
            fut_map_df = get_stock_futures_mapping()
        except Exception as e:
            st.error("❌ 無法取得股票期貨標的對照表，詳細除錯資訊如下（麻煩把這整段複製貼給我）：")
            st.code(str(e))
            st.stop()
    if fut_map_df.empty:
        st.error("台灣期交所回傳的表格是空的，請稍後再試，或把上面的錯誤訊息回報給我。")
        st.stop()
    futures_map = dict(zip(fut_map_df["stock_id"], fut_map_df["futures_id"]))

    with st.spinner(f"依市值排序取前 {top_n} 大個股（用證交所官方資料計算，完全不消耗FinMind額度，通常幾秒~十幾秒完成）..."):
        try:
            top_df = get_top_n_by_market_cap(tuple(futures_map.keys()), top_n)
        except Exception as e:
            st.error("❌ 無法取得市值排名：")
            st.code(str(e))
            st.stop()
    if top_df.empty:
        st.error("市值排名資料是空的，請稍後再試，或把上面的錯誤訊息回報給我。")
        st.stop()

    top_df["has_futures"] = True  # 由於候選池本身就是「有股票期貨」的個股，這裡必然為 True
    covered = top_df.copy()
    missing = pd.DataFrame(columns=top_df.columns)

    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=int(lookback_days * 1.6) + 10)).isoformat()

    with st.spinner(f"抓取 {len(covered)} 檔個股的現貨/期貨價格中（改用批次抓取，通常很快）..."):
        spread_df = build_spread_table(tuple(covered["stock_id"]), futures_map, start_date, end_date)

    with st.spinner("抓取加權指數資料..."):
        taiex_df = get_taiex(start_date, end_date)

    st.session_state["spread_df"] = spread_df
    st.session_state["taiex_df"] = taiex_df
    st.session_state["top_df"] = top_df
    st.session_state["covered"] = covered
    st.session_state["missing"] = missing
    st.session_state["fut_map_df"] = fut_map_df

spread_df = st.session_state.get("spread_df", pd.DataFrame())
taiex_df = st.session_state.get("taiex_df", pd.DataFrame())
top_df = st.session_state.get("top_df", pd.DataFrame())
covered = st.session_state.get("covered", pd.DataFrame())
missing = st.session_state.get("missing", pd.DataFrame())

if spread_df.empty:
    st.warning("目前沒有資料，請按左側「抓取資料」按鈕。")
    st.stop()

name_map = dict(zip(top_df["stock_id"], top_df["stock_name"]))
mv_map = dict(zip(top_df["stock_id"], top_df["market_value"]))

# ------------------------------------------------------------------
# Tab 1：每日快照儀表板
# ------------------------------------------------------------------
with tab1:
    latest_date = spread_df["date"].max()
    snap = spread_df[spread_df["date"] == latest_date].copy()
    snap["stock_name"] = snap["stock_id"].map(name_map)
    snap["market_value"] = snap["stock_id"].map(mv_map)
    snap["正逆"] = np.where(snap["sign"] == 1, "正", "逆")
    snap = snap.sort_values("market_value", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("快照日期", latest_date.strftime("%Y-%m-%d"))
    c2.metric("涵蓋個股數", f"{len(snap)} / {top_n}")
    n_pos = int((snap["sign"] == 1).sum())
    n_neg = int((snap["sign"] == -1).sum())
    c3.metric("正價家數", n_pos)
    c4.metric("逆價家數", n_neg)

    breadth = (n_pos - n_neg) / max(len(snap), 1) * 100
    st.metric("正逆價差廣度指標（正-逆家數佔比）", f"{breadth:+.1f} %",
              help="正值代表偏多（正價家數較多），負值代表偏空（逆價家數較多）")

    st.divider()
    st.subheader("個股正逆價差明細")
    show_cols = ["stock_id", "stock_name", "market_value", "spot_close",
                 "futures_close", "spread", "spread_pct", "正逆"]
    st.dataframe(
        snap[show_cols].rename(columns={
            "stock_id": "股票代號", "stock_name": "股票名稱", "market_value": "0050權重%",
            "spot_close": "股價", "futures_close": "期貨價格", "spread": "價差",
            "spread_pct": "價差%",
        }).style.format({"0050權重%": "{:.2f}%", "股價": "{:.1f}", "期貨價格": "{:.1f}",
                          "價差": "{:+.2f}", "價差%": "{:+.2f}%"}),
        use_container_width=True, height=500,
    )

    fig = px.bar(
        snap.sort_values("spread_pct"), x="spread_pct", y="stock_name",
        orientation="h", color="正逆", color_discrete_map={"正": "#d62728", "逆": "#2ca02c"},
        title="各股正逆價差（%）",
    )
    fig.update_layout(height=max(400, 18 * len(snap)))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("💡 市值排名僅在「有掛牌股票期貨」的個股範圍內計算（並非全市場絕對前N大），因此這裡不會有缺漏股票期貨的個股。")

# ------------------------------------------------------------------
# Tab 2：歷史回測分析
# ------------------------------------------------------------------
with tab2:
    st.subheader("正逆價差擴散指標 vs 隔日加權指數漲跌")

    daily = spread_df.groupby("date").agg(
        n_pos=("sign", lambda s: (s == 1).sum()),
        n_neg=("sign", lambda s: (s == -1).sum()),
        avg_spread_pct=("spread_pct", "mean"),
        total_oi=("open_interest", "sum"),
    ).reset_index()
    daily["breadth"] = (daily["n_pos"] - daily["n_neg"]) / (daily["n_pos"] + daily["n_neg"]) * 100
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["oi_change_pct"] = daily["total_oi"].pct_change() * 100

    merged = pd.merge(daily, taiex_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["taiex_return_next"] = merged["taiex_return_pct"].shift(-1)
    merged["taiex_ma"] = merged["taiex_close"].rolling(trend_ma_days).mean()
    merged["downtrend"] = merged["taiex_close"] < merged["taiex_ma"]

    if len(merged) < 5:
        st.warning("資料筆數太少，無法進行有意義的回測，請拉長回溯天數。")
    else:
        # 前 1~5 天正逆價差廣度指標（比照原本 Excel 的邏輯）
        for lag in range(1, 6):
            merged[f"前{lag}天breadth"] = merged["breadth"].shift(lag)

        colA, colB = st.columns(2)
        with colA:
            corr = merged[["breadth", "taiex_return_next"]].dropna().corr().iloc[0, 1]
            st.metric("今日廣度指標 vs 隔日大盤報酬 相關係數", f"{corr:.3f}")
        with colB:
            hit = merged.dropna(subset=["breadth", "taiex_return_next"])
            same_dir = np.sign(hit["breadth"]) == np.sign(hit["taiex_return_next"])
            win_rate = same_dir.mean() * 100 if len(hit) else np.nan
            st.metric("方向一致勝率（廣度轉正/轉負 vs 隔日大盤漲跌）", f"{win_rate:.1f}%")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=merged["date"], y=merged["breadth"], name="正逆價差廣度指標",
                                   yaxis="y1", line=dict(color="#1f77b4")))
        fig2.add_trace(go.Scatter(x=merged["date"], y=merged["taiex_close"], name="加權指數",
                                   yaxis="y2", line=dict(color="#ff7f0e")))
        fig2.update_layout(
            title="正逆價差廣度指標 與 加權指數 走勢對照",
            yaxis=dict(title="廣度指標 (%)"),
            yaxis2=dict(title="加權指數", overlaying="y", side="right"),
            legend=dict(orientation="h"),
            height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("前1~5天 正逆價差廣度指標 對照表")
        lag_cols = [f"前{lag}天breadth" for lag in range(1, 6)]
        show = merged[["date", "breadth"] + lag_cols + ["taiex_return_pct", "taiex_return_next"]].copy()
        show = show.rename(columns={
            "date": "日期", "breadth": "當日廣度指標", "taiex_return_pct": "當日大盤漲跌%",
            "taiex_return_next": "隔日大盤漲跌%",
        })
        st.dataframe(
            show.sort_values("日期", ascending=False).style.format(
                {c: "{:+.2f}" for c in show.columns if c != "日期"}, na_rep="-"
            ),
            use_container_width=True, height=450,
        )

        st.divider()
        is_long = trend_direction.startswith("多")
        merged["breadth_change"] = merged["breadth"] - merged["breadth"].shift(1)

        if is_long:
            st.subheader("🎯 策略測試：廣度回升 → 隔日做多")
            st.caption(
                f"訊號定義：當天廣度指標 - 前一天廣度指標 ≥ {breadth_change_threshold}"
                f"（廣度明顯回升） 且 加權指數收盤價 > {trend_ma_days}日均線（多頭趨勢），隔日做多台指"
            )
            signal_mask = (merged["breadth_change"] >= breadth_change_threshold) & (merged["taiex_close"] > merged["taiex_ma"])
            win_label, win_col = "隔日上漲勝率", (lambda s: s > 0)
        else:
            st.subheader("🎯 策略測試：廣度下降 → 隔日做空")
            st.caption(
                f"訊號定義：當天廣度指標 - 前一天廣度指標 ≤ -{breadth_change_threshold}"
                f"（廣度明顯下降/逆價差擴大） 且 加權指數收盤價 < {trend_ma_days}日均線（空頭趨勢），隔日做空台指"
            )
            signal_mask = (merged["breadth_change"] <= -breadth_change_threshold) & (merged["taiex_close"] < merged["taiex_ma"])
            win_label, win_col = "隔日下跌勝率", (lambda s: s < 0)

        signal_df = merged[signal_mask].dropna(subset=["taiex_return_next"])
        all_valid = merged.dropna(subset=["taiex_return_next"])

        if signal_df.empty:
            st.warning("目前參數組合下，回測期間內沒有任何一天符合訊號條件。可以試著調低「廣度變化門檻」，或拉長回溯天數增加樣本。")
        else:
            signal_win = win_col(signal_df["taiex_return_next"])
            baseline_win = win_col(all_valid["taiex_return_next"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("符合訊號的天數", f"{len(signal_df)} / {len(all_valid)}")
            c2.metric(f"訊號日{win_label}", f"{signal_win.mean()*100:.1f}%")
            c3.metric(f"同期基準{win_label}", f"{baseline_win.mean()*100:.1f}%",
                      help="不加任何條件，回測期間內所有交易日的同方向勝率，當作對照組")
            c4.metric("訊號日平均隔日報酬", f"{signal_df['taiex_return_next'].mean():+.2f}%")

            edge = signal_win.mean() * 100 - baseline_win.mean() * 100
            if edge > 0:
                st.success(f"訊號日勝率比基準高 {edge:+.1f} 個百分點。")
            else:
                st.info(f"訊號日勝率比基準低 {edge:+.1f} 個百分點，這組參數目前沒有看出優勢。")

            if len(signal_df) < 15:
                st.warning(
                    f"⚠️ 符合訊號的天數只有 {len(signal_df)} 天，樣本數太少，這個勝率統計上很不可靠"
                    "（隨便換一組參數或換一段期間，數字可能差很多），建議拉長「歷史回測回溯天數」"
                    "或調低「廣度變化門檻」，累積到至少30~50個訊號日再參考。"
                )

            st.markdown("**符合訊號的日期明細**")
            show_signal = signal_df[["date", "breadth", "breadth_change", "taiex_close", "taiex_ma",
                                      "taiex_return_pct", "taiex_return_next"]].copy()
            show_signal = show_signal.rename(columns={
                "date": "日期", "breadth": "當日廣度指標", "breadth_change": "廣度變化(當天-前一天)",
                "taiex_close": "加權指數收盤", "taiex_ma": f"{trend_ma_days}日均線",
                "taiex_return_pct": "當日漲跌%", "taiex_return_next": "隔日漲跌%",
            })
            st.dataframe(
                show_signal.sort_values("日期", ascending=False).style.format(
                    {c: "{:+.2f}" for c in show_signal.columns if c != "日期"}, na_rep="-"
                ),
                use_container_width=True, height=300,
            )

        st.download_button(
            "下載回測資料 CSV",
            merged.to_csv(index=False).encode("utf-8-sig"),
            file_name="top50_spread_backtest.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "⚠️ 免責聲明：本工具資料來源為 FinMind 與台灣期貨交易所公開資訊，僅供研究參考，"
    "不構成任何投資建議。正逆價差與大盤漲跌之間的統計關係不代表因果關係或未來績效保證。"
)
