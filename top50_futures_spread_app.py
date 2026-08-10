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
  FINMIND_TOKEN = "你的 FinMind token"
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

TOKEN = st.secrets.get("FINMIND_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}

REQUEST_SLEEP = 0.15  # 避免超過 FinMind 流量限制，每次呼叫間隔


# ------------------------------------------------------------------
# 基礎資料抓取函式
# ------------------------------------------------------------------
def finmind_get(dataset: str, data_id: str = None, start_date: str = None,
                 end_date: str = None) -> pd.DataFrame:
    """呼叫 FinMind /v4/data，回傳 DataFrame（失敗回傳空表）"""
    params = {"dataset": dataset}
    if data_id:
        params["data_id"] = data_id
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    try:
        resp = requests.get(FINMIND_URL, headers=HEADERS, params=params, timeout=20)
        resp.raise_for_status()
        js = resp.json()
        data = js.get("data", [])
        time.sleep(REQUEST_SLEEP)
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"抓取 {dataset} ({data_id}) 失敗：{e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600 * 12, show_spinner=False)
def get_stock_futures_mapping() -> pd.DataFrame:
    """
    抓取台灣期交所「股票期貨/股票選擇權 交易標的」清單，
    整理出 股票代號 <-> 股票期貨商品代碼 對照表（只取標準型 2,000 股合約，排除 ETF）。
    """
    try:
        tables = pd.read_html(TAIFEX_STOCKLIST_URL)
    except Exception as e:
        st.error(f"無法連線台灣期交所取得股票期貨標的清單：{e}")
        return pd.DataFrame(columns=["stock_id", "stock_name", "futures_id"])

    # 找出欄位包含「股票期貨」商品代碼的那張表
    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("商品代碼" in c for c in cols):
            target = t
            break
    if target is None:
        return pd.DataFrame(columns=["stock_id", "stock_name", "futures_id"])

    df = target.copy()
    df.columns = [str(c) for c in df.columns]
    code_col = [c for c in df.columns if "商品代碼" in c][0]
    sec_col = [c for c in df.columns if "證券代號" in c][0]
    name_col = [c for c in df.columns if "簡稱" in c][0]
    is_futures_col = [c for c in df.columns if "是否為" in c and "股票期貨" in c][0]
    lot_col = [c for c in df.columns if "股數" in c or "受益權單位" in c][0]

    df = df[df[is_futures_col].astype(str).str.contains("是股票期貨標的", na=False)]
    df = df[df[sec_col].astype(str).str.match(r"^\d{4,6}$", na=False)]  # 排除 ETF 代號如 0050
    df[lot_col] = df[lot_col].astype(str)
    df = df[df[lot_col].str.contains("2,000", na=False)]  # 只取標準型合約，排除小型100股版本

    out = df[[sec_col, name_col, code_col]].rename(
        columns={sec_col: "stock_id", name_col: "stock_name", code_col: "futures_id"}
    )
    out = out.drop_duplicates(subset="stock_id").reset_index(drop=True)
    return out


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def get_top_n_by_market_cap(n: int = 50, as_of_date: str = None) -> pd.DataFrame:
    """用 TaiwanStockMarketValue 抓最近一個交易日的市值，排出前 N 大"""
    if as_of_date is None:
        as_of_date = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    start = (pd.to_datetime(as_of_date) - pd.Timedelta(days=10)).date().isoformat()
    df = finmind_get("TaiwanStockMarketValue", start_date=start, end_date=as_of_date)
    if df.empty:
        return pd.DataFrame(columns=["stock_id", "market_value", "date"])
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    df = df[df["date"] == latest_date]
    df = df.sort_values("market_value", ascending=False).head(n).reset_index(drop=True)

    # 補上股票名稱
    info = finmind_get("TaiwanStockInfo")
    if not info.empty:
        info = info.drop_duplicates(subset="stock_id")[["stock_id", "stock_name"]]
        df = df.merge(info, on="stock_id", how="left")
    return df


def pick_near_month_futures(fut_df: pd.DataFrame, ref_date: str) -> pd.DataFrame:
    """
    每個 date 只留下「近月」合約（contract_date 最接近但 >= 當月）
    並優先使用一般交易時段(非夜盤)的資料。
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

    def _pick(group):
        cur_month = group["date"].iloc[0].strftime("%Y%m")
        candidates = group[group["contract_date"] >= cur_month]
        if candidates.empty:
            candidates = group
        return candidates.sort_values("contract_date").iloc[0]

    near = df.groupby("date", group_keys=False).apply(_pick)
    return near.reset_index(drop=True)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def build_spread_table(stock_ids: tuple, futures_map: dict, start_date: str,
                        end_date: str) -> pd.DataFrame:
    """
    對每一檔股票抓現貨 + 期貨近月價格，計算每日正逆價差。
    回傳長表：date, stock_id, stock_name, spot_close, futures_close, spread, sign(+1/-1)
    """
    records = []
    for sid in stock_ids:
        fut_id = futures_map.get(sid)
        if not fut_id:
            continue
        spot = finmind_get("TaiwanStockPrice", data_id=sid, start_date=start_date, end_date=end_date)
        fut = finmind_get("TaiwanFuturesDaily", data_id=fut_id, start_date=start_date, end_date=end_date)
        if spot.empty or fut.empty:
            continue
        spot["date"] = pd.to_datetime(spot["date"])
        fut_near = pick_near_month_futures(fut, end_date)
        if fut_near.empty:
            continue
        merged = pd.merge(
            spot[["date", "close"]].rename(columns={"close": "spot_close"}),
            fut_near[["date", "close"]].rename(columns={"close": "futures_close"}),
            on="date", how="inner",
        )
        if merged.empty:
            continue
        merged["stock_id"] = sid
        merged["spread"] = merged["futures_close"] - merged["spot_close"]
        merged["spread_pct"] = merged["spread"] / merged["spot_close"] * 100
        merged["sign"] = np.where(merged["spread"] >= 0, 1, -1)
        records.append(merged)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def get_taiex(start_date: str, end_date: str) -> pd.DataFrame:
    df = finmind_get("TaiwanStockTotalReturnIndex", data_id="TAIEX",
                      start_date=start_date, end_date=end_date)
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
st.title("📊 台股前50大市值 正逆價差 vs 大盤漲跌")
st.caption("即時市值排序 + 個股期貨正逆價差 + 與加權指數的領先/落後關係分析")

if not TOKEN:
    st.info("尚未設定 FINMIND_TOKEN，將以免費額度（較低流量上限）呼叫 API。建議在 Secrets 中加入 FINMIND_TOKEN 以提升穩定度。")

with st.sidebar:
    st.header("⚙️ 參數設定")
    top_n = st.slider("取前 N 大市值個股", 20, 80, 50, step=5)
    lookback_days = st.slider("歷史回測回溯天數（交易日，含快取抓取範圍）", 20, 250, 60, step=10)
    run_btn = st.button("🚀 抓取資料 / 重新整理", type="primary")

tab1, tab2 = st.tabs(["📌 每日快照儀表板", "📈 歷史回測分析"])

if run_btn or "spread_df" not in st.session_state:
    with st.spinner("抓取股票期貨標的對照表..."):
        fut_map_df = get_stock_futures_mapping()
    if fut_map_df.empty:
        st.error("無法取得股票期貨標的對照表，請稍後再試。")
        st.stop()
    futures_map = dict(zip(fut_map_df["stock_id"], fut_map_df["futures_id"]))

    with st.spinner(f"依市值排序取前 {top_n} 大個股..."):
        top_df = get_top_n_by_market_cap(top_n)
    if top_df.empty:
        st.error("無法取得市值資料，請稍後再試。")
        st.stop()

    top_df["has_futures"] = top_df["stock_id"].isin(futures_map.keys())
    covered = top_df[top_df["has_futures"]].copy()
    missing = top_df[~top_df["has_futures"]].copy()

    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=int(lookback_days * 1.6) + 10)).isoformat()

    with st.spinner(f"抓取 {len(covered)} 檔個股的現貨/期貨價格中，這可能需要一點時間..."):
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
            "stock_id": "股票代號", "stock_name": "股票名稱", "market_value": "市值",
            "spot_close": "股價", "futures_close": "期貨價格", "spread": "價差",
            "spread_pct": "價差%",
        }).style.format({"市值": "{:,.0f}", "股價": "{:.1f}", "期貨價格": "{:.1f}",
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

    if not missing.empty:
        with st.expander(f"⚠️ 前{top_n}大市值中有 {len(missing)} 檔沒有對應股票期貨（已排除）"):
            st.dataframe(missing[["stock_id", "stock_name", "market_value"]], use_container_width=True)

# ------------------------------------------------------------------
# Tab 2：歷史回測分析
# ------------------------------------------------------------------
with tab2:
    st.subheader("正逆價差擴散指標 vs 隔日加權指數漲跌")

    daily = spread_df.groupby("date").agg(
        n_pos=("sign", lambda s: (s == 1).sum()),
        n_neg=("sign", lambda s: (s == -1).sum()),
        avg_spread_pct=("spread_pct", "mean"),
    ).reset_index()
    daily["breadth"] = (daily["n_pos"] - daily["n_neg"]) / (daily["n_pos"] + daily["n_neg"]) * 100
    daily = daily.sort_values("date").reset_index(drop=True)

    merged = pd.merge(daily, taiex_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["taiex_return_next"] = merged["taiex_return_pct"].shift(-1)

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
