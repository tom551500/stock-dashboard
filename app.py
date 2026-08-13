import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(page_title="個股事件獲利王", page_icon="📈", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    white-space: nowrap !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
}
.stDataFrame th { text-align: center !important; }
[data-testid="metric-container"] {
    padding: 6px 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 輔助函式與資料載入
# ==========================================
def get_stars(rate):
    if rate >= 80: return "⭐⭐⭐"
    if rate >= 70: return "⭐⭐"
    if rate >= 60: return "⭐"
    return ""

def get_discount_stars(d):
    if d is None or pd.isna(d): return ""
    if d < 8.5: return "⭐⭐⭐"
    if d < 9.0: return "⭐⭐"
    if d < 9.5: return "⭐"
    return ""

def badge_html(text, kind="red"):
    styles = {
        "red":   "background:#fdecea;color:#c0392b;",
        "green": "background:#e8f5e9;color:#1b7e4b;",
        "amber": "background:#fff8e1;color:#b7790a;",
    }
    s = styles.get(kind, styles["red"])
    return f"<span style='{s}font-size:11px;font-weight:600;padding:2px 8px;border-radius:4px;white-space:nowrap'>{text}</span>"

@st.cache_data(ttl=30)
def get_realtime_quote(stock_id):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_id}.TW", headers=headers, timeout=5).json()
        if res.get('chart', {}).get('error'):
            res = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{stock_id}.TWO", headers=headers, timeout=5).json()
            
        result = res['chart']['result'][0]['meta']
        price = result['regularMarketPrice']
        prev_close = result['chartPreviousClose']
        
        change = price - prev_close
        pct_change = (change / prev_close) * 100
        return price, change, pct_change
    except:
        return None, None, None

@st.cache_data(ttl=30)
def get_realtime_futures_quote(stock_id, spot_price):
    DEMO_MODE = True
    if DEMO_MODE and spot_price is not None:
        return round(spot_price * 0.995, 2)
    return None

if not os.path.exists('all_taiwan_dividend_10years.csv'):
    st.error("⚠️ 找不到 `all_taiwan_dividend_10years.csv`，請確認已上傳到 GitHub！")
    st.stop()

@st.cache_data(ttl=3600)
def load_local_database(token):
    """
    改用 FinMind 的「股利政策表」(TaiwanStockDividend) 批次查詢，取代直接連線
    TWSE/TPEx 官方網站。原因：TPEx 官網對雲端主機 IP 有反機器人防護，Streamlit Cloud
    這類雲端環境常被擋下（回傳假的200狀態但內容是攔截頁面），造成上櫃股票資料抓不到；
    FinMind 是統一的第三方資料源，同時涵蓋上市(TWSE)和上櫃(TPEx)，且本App其他功能
    都已證實它在雲端運作穩定，不會被擋。

    這裡故意抓「較寬的日期區間」(往前抓 200 天) 再讓外層自行篩選未來60天，
    是因為現金股利公告日期跟實際除息日通常會差幾個月，用太窄的區間可能篩不到。
    """
    debug_msgs = []
    today = pd.to_datetime('today').normalize()
    query_start = (today - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
    query_end = (today + pd.Timedelta(days=60)).strftime('%Y-%m-%d')

    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={
            "dataset": "TaiwanStockDividend",
            "start_date": query_start,
            "end_date": query_end,
            "token": token
        }, timeout=30)
        debug_msgs.append(f"FinMind TaiwanStockDividend status={r.status_code}")
        df = pd.DataFrame(r.json().get("data", []))
        debug_msgs.append(f"FinMind 回傳筆數={len(df)}")
        if df.empty:
            return pd.DataFrame(columns=['CashExDividendTradingDate', 'stock_id']), debug_msgs

        debug_msgs.append(f"FinMind 欄位={list(df.columns)}")
        df['stock_id'] = df['stock_id'].astype(str).str.strip()
        df['CashExDividendTradingDate'] = pd.to_datetime(df.get('CashExDividendTradingDate'), errors='coerce')
        if 'ExRightTradingDate' in df.columns:
            df['ExRightTradingDate'] = pd.to_datetime(df['ExRightTradingDate'], errors='coerce')
            df['CashExDividendTradingDate'] = df['CashExDividendTradingDate'].fillna(df['ExRightTradingDate'])

        df = df.dropna(subset=['CashExDividendTradingDate'])
        df = df.drop_duplicates(subset=['stock_id', 'CashExDividendTradingDate'])
        debug_msgs.append(f"日期解析成功筆數={len(df)}（涵蓋上市+上櫃，來源:FinMind）")
        return df[['CashExDividendTradingDate', 'stock_id']], debug_msgs
    except Exception as e:
        debug_msgs.append(f"FinMind error: {e}")
        return pd.DataFrame(columns=['CashExDividendTradingDate', 'stock_id']), debug_msgs

@st.cache_data(ttl=86400)
def load_stock_names(api_token):
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={"dataset": "TaiwanStockInfo", "token": api_token}, timeout=10)
        df = pd.DataFrame(r.json().get("data", []))
        if not df.empty:
            return dict(zip(df['stock_id'].astype(str).str.strip(), df['stock_name']))
        return {}
    except:
        return {}

@st.cache_data(ttl=3600 * 12)
def get_stock_futures():
    """
    抓取台灣期交所「股票期貨標的」清單，回傳有股票期貨的4碼股票代號清單。
    邏輯移植自 top50_futures_spread_app.py 裡驗證過可用的 get_stock_futures_mapping()：
    改用尋找「商品代碼」欄位來定位正確的表格，比單純找「代號」欄位更準確
    （期交所頁面裡很多表格欄位都含「代號」兩字，容易抓錯表）。
    """
    fallback = ['2330', '2317', '2454', '2603', '2308', '1319', '1904', '8299', '2881', '2882']
    TAIFEX_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-TW,zh;q=0.9",
        "Referer": "https://www.taifex.com.tw/cht/2/stockLists",
    }

    try:
        resp = requests.get("https://www.taifex.com.tw/cht/2/stockLists", headers=TAIFEX_HEADERS, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        return fallback, f"TAIFEX連線失敗({type(e).__name__}: {e})，改用內建10檔預設清單（不含2520/2540等小型股）"

    try:
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:
        return fallback, f"TAIFEX表格解析失敗({type(e).__name__}: {e})，可能缺少lxml套件，改用內建10檔預設清單"

    if not tables:
        return fallback, f"TAIFEX有回應(HTTP {resp.status_code})但頁面裡沒有偵測到任何表格，改用內建10檔預設清單"

    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("商品代碼" in c for c in cols):
            target = t
            break

    if target is None:
        return fallback, f"抓到{len(tables)}張表但找不到含「商品代碼」的表，網站版面可能改了，改用內建10檔預設清單"

    df = target.copy()
    df.columns = [str(c) for c in df.columns]
    sec_col = next((c for c in df.columns if "證券代號" in c or "股票代號" in c), None)

    if sec_col is None:
        return fallback, f"找到商品代碼表但沒有證券代號欄位，欄位有：{list(df.columns)}，改用內建10檔預設清單"

    stock_ids = df[sec_col].astype(str).str.strip()
    stock_ids = stock_ids[stock_ids.str.match(r'^\d{4}$', na=False)]
    stock_ids = stock_ids[~stock_ids.str.startswith('00')]
    futures_set = sorted(set(stock_ids.tolist()))

    if futures_set:
        return futures_set, f"TAIFEX抓取成功，共{len(futures_set)}檔有股票期貨的個股"
    else:
        return fallback, "表格解析完但篩選後是空清單，改用內建10檔預設清單"

# ==========================================
# 2-1. 【新增】基本面因子：月營收模組
# ==========================================
@st.cache_data(ttl=3600)
def load_monthly_revenue(stock_id, token):
    """
    抓取月營收資料，並計算 YoY(年增率) 與 MoM(月增率)。
    announce_date 為概估公告日 (營收所屬月份的次月10日)，
    僅用來確保後續分析『除息當下市場已公開此筆營收』，避免使用未來資訊。
    """
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={
            "dataset": "TaiwanStockMonthRevenue",
            "data_id": str(stock_id),
            "start_date": "2013-01-01",
            "end_date": "2026-12-31",
            "token": token
        }, timeout=15)
        df = pd.DataFrame(r.json().get("data", []))
        if df.empty:
            return pd.DataFrame()

        df['revenue'] = pd.to_numeric(df.get('revenue', np.nan), errors='coerce')
        df['revenue_year'] = pd.to_numeric(df.get('revenue_year', np.nan), errors='coerce')
        df['revenue_month'] = pd.to_numeric(df.get('revenue_month', np.nan), errors='coerce')
        df = df.dropna(subset=['revenue_year', 'revenue_month', 'revenue'])
        if df.empty:
            return pd.DataFrame()

        df['period'] = pd.to_datetime(
            df['revenue_year'].astype(int).astype(str) + '-' +
            df['revenue_month'].astype(int).astype(str).str.zfill(2) + '-01'
        )
        df['announce_date'] = df['period'] + pd.DateOffset(months=1, days=9)
        df = df.sort_values('period').drop_duplicates(subset=['period']).reset_index(drop=True)

        df['yoy'] = df['revenue'].pct_change(periods=12) * 100
        df['mom'] = df['revenue'].pct_change(periods=1) * 100
        return df
    except:
        return pd.DataFrame()

def get_revenue_snapshot_before(rev_df, ref_date):
    """取得 ref_date 當下，市場已公開的最新一筆營收資料列 (含 yoy/mom)"""
    if rev_df is None or rev_df.empty or pd.isna(ref_date):
        return None
    avail = rev_df[rev_df['announce_date'] <= ref_date]
    if avail.empty:
        return None
    return avail.iloc[-1]

def get_revenue_momentum_badge(rev_df, ref_date, n=3):
    """比較最近n個月 YoY 平均 vs 前n個月 YoY 平均，判斷營收動能是加速還是趨緩"""
    if rev_df is None or rev_df.empty or pd.isna(ref_date):
        return "資料不足", "amber"
    avail = rev_df[rev_df['announce_date'] <= ref_date].tail(n * 2)
    if len(avail) < n * 2:
        return "資料不足", "amber"
    recent = avail['yoy'].tail(n).mean()
    prior = avail['yoy'].head(n).mean()
    if pd.isna(recent) or pd.isna(prior):
        return "資料不足", "amber"
    if recent > prior + 2:
        return "🚀 動能加速", "red"
    elif recent < prior - 2:
        return "⚠️ 動能趨緩", "green"
    else:
        return "➖ 動能持平", "amber"

@st.cache_data(ttl=3600)
def load_quarterly_eps(stock_id, token):
    """
    抓取季度財報中的 EPS 資料 (FinMind: TaiwanStockFinancialStatements, type == 'EPS')
    回傳含 year / quarter / value 的乾淨表，若某季尚未公布則該季查不到資料。
    """
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={
            "dataset": "TaiwanStockFinancialStatements",
            "data_id": str(stock_id),
            "start_date": "2022-01-01",
            "end_date": "2026-12-31",
            "token": token
        }, timeout=15)
        df = pd.DataFrame(r.json().get("data", []))
        if df.empty or 'type' not in df.columns:
            return pd.DataFrame()

        df = df[df['type'] == 'EPS'].copy()
        if df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['date', 'value'])
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        return df[['date', 'year', 'quarter', 'value']].sort_values('date').drop_duplicates(subset=['year', 'quarter'], keep='last')
    except:
        return pd.DataFrame()

def get_eps_value(eps_df, year, quarter):
    """取得指定年度/季度的EPS，若尚未公布則回傳 None"""
    if eps_df is None or eps_df.empty:
        return None
    row = eps_df[(eps_df['year'] == year) & (eps_df['quarter'] == quarter)]
    if row.empty:
        return None
    return row.iloc[0]['value']

# ==========================================
# 2-2. 【新增】總體面因子：大盤同步性 + 除息旺季效應
# ==========================================
@st.cache_data(ttl=3600)
def load_taiex_index(token):
    """
    抓取加權報酬指數 (TAIEX Total Return Index) 歷史資料，作為大盤同步性分析的基準。
    用報酬指數而非單純的加權指數，是因為報酬指數有把除權息還原計入，
    走勢更能反映「大盤整體多空氛圍」，不受成分股除權息干擾。
    """
    try:
        r = requests.get("https://api.finmindtrade.com/api/v4/data", params={
            "dataset": "TaiwanStockTotalReturnIndex",
            "data_id": "TAIEX",
            "start_date": "2013-01-01",
            "end_date": "2026-12-31",
            "token": token
        }, timeout=15)
        df = pd.DataFrame(r.json().get("data", []))
        if df.empty:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df.get('date'), errors='coerce')
        price_col = 'price' if 'price' in df.columns else ('close' if 'close' in df.columns else None)
        if price_col is None:
            return pd.DataFrame()
        df = df[['date', price_col]].rename(columns={price_col: 'taiex_close'})
        df['taiex_close'] = pd.to_numeric(df['taiex_close'], errors='coerce')
        df = df.dropna().sort_values('date').reset_index(drop=True)
        return df
    except:
        return pd.DataFrame()

def get_taiex_week_return(taiex_df, ex_date, lookback_days=5):
    """計算『除息當週』大盤表現：除息日往前lookback_days個交易日到除息日當天的大盤漲跌幅%"""
    if taiex_df is None or taiex_df.empty or pd.isna(ex_date):
        return None
    sub = taiex_df[taiex_df['date'] <= ex_date].tail(lookback_days + 1)
    if len(sub) < 2:
        return None
    start_price = sub.iloc[0]['taiex_close']
    end_price = sub.iloc[-1]['taiex_close']
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None
    return (end_price - start_price) / start_price * 100

# ==========================================
# 3. Token 設定
# ==========================================
# 【已修改】優先從 st.secrets 讀取（部署到 Streamlit Cloud 時使用）
# 若本機測試沒有設定 secrets.toml，則退回下方預設值（僅供本機測試，正式部署前請改用 secrets）
try:
    api_token_str = st.secrets["FINMIND_TOKEN"]
except Exception:
    api_token_str = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoidG9tODg4NSIsImVtYWlsIjoidG9tNjQ2ZkBnbWFpbC5jb20iLCJ0b2tlbl92ZXJzaW9uIjowfQ.MJL4mTzQEbYSavhBTYM3GCBstqGJThASMo9iTQbbCxQ'

if not api_token_str or api_token_str == '您的_TOKEN':
    st.error("⚠️ 尚未設定 FinMind Token！請於 Streamlit Cloud 後台 Secrets 新增 FINMIND_TOKEN。")
    st.stop()

stock_names_dict = load_stock_names(api_token_str)

# ==========================================
# 4. 左側邊欄
# ==========================================
with st.sidebar:
    st.title("⚙️ 戰略控制台")
    st.markdown("---")
    st.markdown("### 🔍 進階篩選")
    col_f, col_w = st.columns(2)
    with col_f: filter_futures = st.checkbox("☑️ 有股期")
    with col_w: filter_warrants = st.checkbox("☑️ 有權證")

    st.markdown("---")
    st.markdown("### 🏆 近期除權息日程雷達")
    st.markdown("---")

    try:
        dividend_db, debug_msgs = load_local_database(api_token_str)
        df_local = dividend_db.copy()
        df_local['ExDate'] = df_local['CashExDividendTradingDate']
        df_local = df_local.dropna(subset=['ExDate'])
        today_norm = pd.to_datetime('today').normalize()
        next_days = today_norm + pd.Timedelta(days=60)
        
        upcoming_df = df_local[(df_local['ExDate'] >= today_norm) & (df_local['ExDate'] <= next_days)].copy()
        if filter_futures:
            futures_list, futures_debug = get_stock_futures()
            upcoming_df = upcoming_df[upcoming_df['stock_id'].isin(futures_list)]
            debug_msgs.append(futures_debug)
            debug_msgs.append(f"股期代號範例(前10檔): {futures_list[:10]}")
        
        upcoming_df = upcoming_df[['ExDate', 'stock_id']].drop_duplicates().sort_values(['ExDate', 'stock_id'])
        upcoming_df['stock_name'] = upcoming_df['stock_id'].map(stock_names_dict).fillna("")
        upcoming_df['display_text'] = (upcoming_df['ExDate'].dt.strftime('%m/%d') + " | " + upcoming_df['stock_id'] + " " + upcoming_df['stock_name'])
        upcoming_list = upcoming_df['display_text'].tolist()
        debug_msgs.append(f"篩選後可顯示筆數={len(upcoming_list)}")
    except Exception as e:
        upcoming_list = []
        debug_msgs = [f"整體流程發生例外: {e}"]

    with st.expander("🔧 除錯資訊（近期除權息清單抓取狀況）"):
        for m in debug_msgs:
            st.caption(m)

    selected_option = st.selectbox("📅 近期除權息清單：", ["--- 請選擇或手動輸入 ---"] + upcoming_list)
    manual_input    = st.text_input("🔍 手動輸入代號 (例: 1904)", "")

# ==========================================
# 5. 股票選擇
# ==========================================
if manual_input: target_stock_id = manual_input.strip()
elif selected_option != "--- 請選擇或手動輸入 ---": target_stock_id = selected_option.split(" | ")[1].split(" ")[0]
else: target_stock_id = None

# ==========================================
# 6. 主畫面
# ==========================================
if not target_stock_id:
    st.title("📈 股利與報酬統計系統")
    st.markdown("### 👈 請從左側選單挑選股票，系統將為您生成多維度時間軸報表。")
else:
    target_stock_name = stock_names_dict.get(str(target_stock_id), "")

    with st.spinner(f"正在建立時間軸，即時抓取 {target_stock_id} {target_stock_name} 資料..."):
        try:
            my_div = pd.DataFrame(requests.get("https://api.finmindtrade.com/api/v4/data", params={
                "dataset": "TaiwanStockDividend", "data_id": str(target_stock_id),
                "start_date": "2015-01-01", "end_date": "2026-12-31", "token": api_token_str
            }, timeout=15).json().get("data", []))

            if my_div.empty: st.warning(f"❌ 找不到 {target_stock_id} 的配息紀錄。"); st.stop()

            kline = pd.DataFrame(requests.get("https://api.finmindtrade.com/api/v4/data", params={
                "dataset": "TaiwanStockPrice", "data_id": str(target_stock_id),
                "start_date": "2015-01-01", "end_date": "2026-12-31", "token": api_token_str
            }, timeout=15).json().get("data", []))

            if kline.empty: st.error("❌ 獲取歷史股價失敗。請確認 API 額度。"); st.stop()

            if 'max' in kline.columns and 'high' not in kline.columns:
                kline['high'] = kline['max']
            if 'min' in kline.columns and 'low' not in kline.columns:
                kline['low'] = kline['min']

            latest_close = float(kline['close'].iloc[-1])
            latest_open  = float(kline['open'].iloc[-1])
            day_chg      = latest_close - latest_open
            day_chg_pct  = day_chg / latest_open * 100

            # 【新增】載入月營收資料 + 季度EPS資料 + 大盤指數資料
            rev_df = load_monthly_revenue(target_stock_id, api_token_str)
            eps_df = load_quarterly_eps(target_stock_id, api_token_str)
            taiex_df = load_taiex_index(api_token_str)

            my_div['TotalCashDividend']  = (pd.to_numeric(my_div.get('CashEarningsDistribution',  0), errors='coerce').fillna(0) + pd.to_numeric(my_div.get('CashStatutorySurplus',      0), errors='coerce').fillna(0))
            my_div['TotalStockDividend'] = (pd.to_numeric(my_div.get('StockEarningsDistribution', 0), errors='coerce').fillna(0) + pd.to_numeric(my_div.get('StockStatutorySurplus',      0), errors='coerce').fillna(0))
            my_div['ExDate'] = pd.to_datetime(my_div.get('CashExDividendTradingDate'), errors='coerce')
            if 'ExRightTradingDate' in my_div.columns: my_div['ExDate'] = my_div['ExDate'].fillna(pd.to_datetime(my_div['ExRightTradingDate'], errors='coerce'))
            my_div = my_div.dropna(subset=['ExDate'])
            my_div = my_div[(my_div['TotalCashDividend'] > 0) | (my_div['TotalStockDividend'] > 0)]
            my_div = my_div.sort_values(['ExDate','TotalCashDividend'], ascending=[False,False]).drop_duplicates(subset=['ExDate'])

            kline['date'] = pd.to_datetime(kline['date'])
            kline = kline.sort_values('date').reset_index(drop=True)
            valid_trading_dates = kline['date'].dropna().sort_values().unique()

            def find_actual(d):
                if pd.isna(d): return pd.NaT
                f = valid_trading_dates[valid_trading_dates >= d]
                return f[0] if len(f) > 0 else d

            my_div['ExDate'] = my_div['ExDate'].apply(find_actual)

            kline['close_T_minus_5'] = kline['close'].shift(5)
            kline['close_T_minus_4'] = kline['close'].shift(4)
            kline['close_T_minus_3'] = kline['close'].shift(3)
            kline['close_T_minus_2'] = kline['close'].shift(2)
            kline['close_T_minus_1'] = kline['close'].shift(1)
            kline['open_T_minus_1']  = kline['open'].shift(1)
            kline['open_T_plus_1']   = kline['open'].shift(-1)
            kline['close_T_plus_1']  = kline['close'].shift(-1)

            merged = pd.merge(my_div, kline, left_on='ExDate', right_on='date', how='inner')
            merged['ref_price'] = (merged['close_T_minus_1'] - merged['TotalCashDividend']) / (1 + merged['TotalStockDividend']/10)
            merged['yield']     = merged['TotalCashDividend'] / merged['close_T_minus_1'] * 100
            merged['pre_4']     = (merged['close_T_minus_4'] - merged['close_T_minus_5']) / merged['close_T_minus_5'] * 100
            merged['pre_3']     = (merged['close_T_minus_3'] - merged['close_T_minus_4']) / merged['close_T_minus_4'] * 100
            merged['pre_2']     = (merged['close_T_minus_2'] - merged['close_T_minus_3']) / merged['close_T_minus_3'] * 100
            merged['pre_1']     = (merged['close_T_minus_1'] - merged['close_T_minus_2']) / merged['close_T_minus_2'] * 100
            merged['pre_1_open']= (merged['open_T_minus_1'] - merged['close_T_minus_2']) / merged['close_T_minus_2'] * 100
            merged['day_open']     = (merged['open']         - merged['ref_price']) / merged['ref_price'] * 100
            merged['day_close']    = (merged['close']        - merged['ref_price']) / merged['ref_price'] * 100
            merged['post_1_open']  = (merged['open_T_plus_1'] - merged['close'])     / merged['close']     * 100
            merged['post_1_close'] = (merged['close_T_plus_1']- merged['close'])     / merged['close']     * 100
            merged['pre5_trend']   = merged['close_T_minus_1'] > merged['close_T_minus_5']

            total_events         = len(merged)
            open_win_rate      = len(merged[merged['day_open']  > 0]) / total_events * 100 if total_events > 0 else 0
            close_win_rate     = len(merged[merged['day_close'] > 0]) / total_events * 100 if total_events > 0 else 0
            avg_open_return  = merged['day_open'].mean()  if total_events > 0 else 0
            avg_close_return = merged['day_close'].mean() if total_events > 0 else 0
            pre1_open_win_rate  = len(merged[merged['pre_1_open'] > 0]) / total_events * 100 if total_events > 0 else 0
            pre1_close_win_rate = len(merged[merged['pre_1'] > 0]) / total_events * 100 if total_events > 0 else 0
            avg_pre1_close_ret  = merged['pre_1'].mean() if total_events > 0 else 0

            up_avg   = merged.loc[merged['day_open'] > 0, 'day_open'].mean()
            down_avg = merged.loc[merged['day_open'] < 0, 'day_open'].mean()
            risk_reward = abs(up_avg / down_avg) if (not pd.isna(up_avg) and not pd.isna(down_avg) and down_avg != 0) else None

            trend_up_mask       = merged['pre5_trend'] == True
            trend_down_mask     = merged['pre5_trend'] == False
            up_trend_win_rate   = len(merged[trend_up_mask   & (merged['day_open'] > 0)]) / max(trend_up_mask.sum(),   1) * 100
            down_trend_win_rate = len(merged[trend_down_mask & (merged['day_open'] > 0)]) / max(trend_down_mask.sum(), 1) * 100

            today_dt    = pd.to_datetime('today').normalize()
            future_divs = my_div[my_div['ExDate'] >= today_dt].sort_values('ExDate')
            if not future_divs.empty:
                next_div_date = future_divs.iloc[0]['ExDate']
                days_left     = (next_div_date - today_dt).days
                next_cash     = future_divs.iloc[0]['TotalCashDividend']
                next_stock    = future_divs.iloc[0]['TotalStockDividend']
                date_str      = next_div_date.strftime('%Y/%m/%d')
                discount_val  = ((latest_close - next_cash) / (1 + next_stock/10)) / latest_close * 10
                discount_str  = f"{discount_val:.1f}折"
            else:
                date_str = "暫無公告"; days_left = "-"; next_cash = 0.0; next_stock = 0.0; discount_str = "-"; discount_val = None

            rt_price, rt_change, rt_pct = get_realtime_quote(target_stock_id)
            if rt_price is None: rt_price, rt_change, rt_pct = latest_close, day_chg, day_chg_pct
            fut_price = get_realtime_futures_quote(target_stock_id, rt_price)

            if fut_price is not None and rt_price is not None:
                basis = fut_price - rt_price
                basis_str = f"{basis:+.2f}"
                basis_color = "#ef5350" if basis > 0 else "#26a69a"
                basis_label = "正價差" if basis > 0 else "逆價差"
                fut_disp = f"{fut_price:.2f}"
            else:
                basis_str = "--"; basis_color = "#888"; basis_label = "價差"; fut_disp = "--"

            chg_color = "#ef5350" if rt_change > 0 else ("#26a69a" if rt_change < 0 else "#9e9e9e")
            chg_sign  = "+" if rt_change > 0 else ""
            nc_disp   = f"{next_cash:.2f}"
            ns_disp   = f"{next_stock:.2f}"

            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap; padding-bottom:14px;margin-bottom:18px; border-bottom:1px solid rgba(128,128,128,0.18);">
  <span style="font-size:13px;color:#888;font-weight:500; background:rgba(128,128,128,0.1);padding:2px 9px;border-radius:6px">{target_stock_id}</span>
  <span style="font-size:22px;font-weight:600">{target_stock_name}</span>
  <span style="font-size:26px;font-weight:700;color:{chg_color};margin-left:5px;">{rt_price:.2f}</span>
  <span style="font-size:13px;color:{chg_color};font-weight:600;">{chg_sign}{rt_change:.2f} ({chg_sign}{rt_pct:.2f}%)</span>
  <span style="font-size:12px;color:#555; background:rgba(255,165,0,0.12);padding:4px 10px;border-radius:6px;margin-left:10px">
    FUT 期貨 <b>{fut_disp}</b>
  </span>
  <span style="font-size:12px;color:{basis_color}; background:rgba(128,128,128,0.07);padding:4px 10px;border-radius:6px">
    {basis_label} <b>{basis_str}</b>
  </span>
  <span style="margin-left:auto;font-size:12px;color:#777; background:rgba(128,128,128,0.07);padding:6px 14px;border-radius:8px; border:0.5px solid rgba(128,128,128,0.18);white-space:nowrap">
    📅 下次除息 <b style="color:#333">{date_str}</b> &nbsp;·&nbsp; 距今 <b style="color:#333">{days_left}</b> 天 &nbsp;·&nbsp; 現金 <b style="color:#ef5350">{nc_disp}</b>
  </span>
</div>
""", unsafe_allow_html=True)

            # ==========================================
            # 6-1. 五大核心指標卡片（第5張為新增的基本面診斷卡）
            # ==========================================
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f"""
                <div style="background:white;padding:16px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);height:150px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px">⬅️ 除息前一日</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
                        <span style="font-size:13px;color:#666">開高機率</span><span style="font-size:14px;font-weight:600">{pre1_open_win_rate:.0f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:13px;color:#666">收高機率</span><span style="font-size:14px;font-weight:600">{pre1_close_win_rate:.0f}% {get_stars(pre1_close_win_rate)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;color:#666">預期報酬</span>
                        <div style="display:flex;align-items:center;gap:6px">
                            <span style="font-size:14px;font-weight:700;color:{'#ef5350' if avg_pre1_close_ret>0 else '#26a69a'}">{avg_pre1_close_ret:+.2f}%</span>
                            {badge_html('偏強', 'red') if avg_pre1_close_ret > 0 else badge_html('偏弱', 'green')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div style="background:white;padding:16px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);height:150px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px">🎯 除息當日開盤</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
                        <span style="font-size:13px;color:#666">開高機率</span><span style="font-size:14px;font-weight:600">{open_win_rate:.0f}% {get_stars(open_win_rate)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:13px;color:#666">開盤折扣</span><span style="font-size:14px;font-weight:700;color:#333">{discount_str} {get_discount_stars(discount_val)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;color:#666">預期報酬</span>
                        <div style="display:flex;align-items:center;gap:6px">
                            <span style="font-size:14px;font-weight:700;color:{'#ef5350' if avg_open_return>0 else '#26a69a'}">{avg_open_return:+.2f}%</span>
                            {badge_html('值得博', 'red') if discount_val is not None and discount_val < 9.0 else badge_html('普通', 'amber')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div style="background:white;padding:16px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);height:150px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:12px">📊 除息當日收盤</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;margin-top:8px">
                        <span style="font-size:13px;color:#666">收高機率</span><span style="font-size:16px;font-weight:800;color:#111">{close_win_rate:.0f}% {get_stars(close_win_rate)}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;color:#666">預期報酬</span>
                        <div style="display:flex;align-items:center;gap:6px">
                            <span style="font-size:14px;font-weight:700;color:{'#ef5350' if avg_close_return>0 else '#26a69a'}">{avg_close_return:+.2f}%</span>
                            {badge_html('偏強', 'red') if avg_close_return > 0 else badge_html('偏弱', 'green')}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c4:
                risk_str = f"{risk_reward:.2f}" if risk_reward is not None else "--"
                st.markdown(f"""
                <div style="background:white;padding:16px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);height:150px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px">🎯 勝率增強儀表板</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
                        <span style="font-size:13px;color:#666">開盤風報酬比</span><span style="font-size:14px;font-weight:700">{risk_str}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:13px;color:#666">上漲均獲利</span><span style="font-size:14px;font-weight:700;color:#ef5350">+{up_avg:.2f}%</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;color:#666">前日趨勢勝率</span><span style="font-size:14px;font-weight:700">{up_trend_win_rate:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 【新增】c5：基本面診斷卡
            with c5:
                latest_rev_row = get_revenue_snapshot_before(rev_df, pd.to_datetime('today'))
                if latest_rev_row is not None:
                    yoy_val = latest_rev_row['yoy']
                    mom_val = latest_rev_row['mom']
                    yoy_disp = f"{yoy_val:+.1f}%" if not pd.isna(yoy_val) else "-"
                    mom_disp = f"{mom_val:+.1f}%" if not pd.isna(mom_val) else "-"
                    yoy_color = '#ef5350' if (not pd.isna(yoy_val) and yoy_val > 0) else '#26a69a'
                    rev_period_str = latest_rev_row['period'].strftime('%Y/%m')
                else:
                    yoy_disp = "無資料"; mom_disp = "無資料"; yoy_color = '#888'; rev_period_str = "-"

                momentum_text, momentum_kind = get_revenue_momentum_badge(rev_df, pd.to_datetime('today'))
                struct_text = "全現金股利" if next_stock == 0 else "含股票股利"
                struct_kind = "red" if next_stock == 0 else "amber"

                st.markdown(f"""
                <div style="background:white;padding:16px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);height:150px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px">🔬 基本面診斷 <span style="font-size:10px;color:#999;font-weight:400">({rev_period_str})</span></div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
                        <span style="font-size:13px;color:#666" title="與去年同月相比的營收成長率，排除季節性，適合看長期趨勢方向">最新營收YoY ⓘ</span><span style="font-size:14px;font-weight:700;color:{yoy_color}">{yoy_disp}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:13px;color:#666" title="與上個月相比的營收成長率，容易受季節性影響，適合看短期變化速度">最新營收MoM ⓘ</span><span style="font-size:14px;font-weight:600">{mom_disp}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;color:#666">股利結構</span>
                        <div style="display:flex;align-items:center;gap:6px">
                            {badge_html(momentum_text, momentum_kind)}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ==========================================
            # 6-1b. 【新增】去年 vs 今年 Q1 / Q2 EPS 對比
            # ==========================================
            this_year = datetime.now().year
            last_year = this_year - 1

            eps_last_q1 = get_eps_value(eps_df, last_year, 1)
            eps_last_q2 = get_eps_value(eps_df, last_year, 2)
            eps_this_q1 = get_eps_value(eps_df, this_year, 1)
            eps_this_q2 = get_eps_value(eps_df, this_year, 2)

            def eps_disp(v):
                return f"{v:.2f}" if v is not None else "—"

            def eps_yoy_disp(this_v, last_v):
                if this_v is None:
                    return "尚未公布", "#999"
                if last_v is None or last_v == 0:
                    return "無去年基期", "#999"
                yoy = (this_v - last_v) / abs(last_v) * 100
                color = '#ef5350' if yoy > 0 else '#26a69a'
                return f"{yoy:+.1f}%", color

            q1_yoy_txt, q1_yoy_color = eps_yoy_disp(eps_this_q1, eps_last_q1)
            q2_yoy_txt, q2_yoy_color = eps_yoy_disp(eps_this_q2, eps_last_q2)

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.markdown(f"""
                <div style="background:white;padding:14px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);text-align:center;">
                    <div style="font-size:12px;color:#888;margin-bottom:6px" title="去年第一季每股盈餘，作為YoY比較基期">{last_year} Q1 EPS</div>
                    <div style="font-size:20px;font-weight:700;color:#333">{eps_disp(eps_last_q1)}</div>
                </div>
                """, unsafe_allow_html=True)
            with e2:
                st.markdown(f"""
                <div style="background:white;padding:14px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);text-align:center;">
                    <div style="font-size:12px;color:#888;margin-bottom:6px" title="與去年同季相比的EPS成長率，正值代表獲利成長、負值代表獲利衰退">{this_year} Q1 EPS <span style="color:{q1_yoy_color};font-weight:600">({q1_yoy_txt})</span></div>
                    <div style="font-size:20px;font-weight:700;color:#333">{eps_disp(eps_this_q1)}</div>
                </div>
                """, unsafe_allow_html=True)
            with e3:
                st.markdown(f"""
                <div style="background:white;padding:14px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);text-align:center;">
                    <div style="font-size:12px;color:#888;margin-bottom:6px" title="去年第二季每股盈餘，作為YoY比較基期">{last_year} Q2 EPS</div>
                    <div style="font-size:20px;font-weight:700;color:#333">{eps_disp(eps_last_q2)}</div>
                </div>
                """, unsafe_allow_html=True)
            with e4:
                st.markdown(f"""
                <div style="background:white;padding:14px;border-radius:10px;border:1px solid rgba(128,128,128,0.2);text-align:center;">
                    <div style="font-size:12px;color:#888;margin-bottom:6px" title="與去年同季相比的EPS成長率；若今年該季尚未公布財報則留空">{this_year} Q2 EPS <span style="color:{q2_yoy_color};font-weight:600">({q2_yoy_txt})</span></div>
                    <div style="font-size:20px;font-weight:700;color:#333">{eps_disp(eps_this_q2)}</div>
                </div>
                """, unsafe_allow_html=True)

            # 【新增】名詞解釋備註（收合式，不佔版面但點開就能查）
            with st.expander("ℹ️ 名詞解釋：YoY / MoM 怎麼看？"):
                st.markdown("""
| 指標 | 全稱 | 比較對象 | 用途 |
|---|---|---|---|
| **YoY** | Year over Year（年增率） | 與**去年同月／同季**比較 | 排除季節性，適合看**長期趨勢方向** |
| **MoM** | Month over Month（月增率） | 與**上個月**比較 | 容易受季節性影響，適合看**短期變化速度** |

**顏色與正負號規則（跟台股慣例一致：紅漲綠跌）**
- 🔴 **紅色 / 正號（+%）**：比較基準期成長，營收或獲利變多
- 🟢 **綠色 / 負號（−%）**：比較基準期衰退，營收或獲利變少

**判讀技巧**
- 只看 MoM 容易被「季節性回升」誤導（例如淡季後回溫），建議搭配 YoY 一起看。
- YoY 若基期（去年同期）特別低或特別高，會讓成長率數字失真（例如去年同季獲利很差，今年稍微好一點就變成 YoY +300%），要留意「低基期效應」。
- 「營收動能」badge（🚀加速 / ➖持平 / ⚠️趨緩）是拿「最近3個月YoY平均」跟「再前3個月YoY平均」比較，用來判斷短期MoM回升是否只是曇花一現，還是中期趨勢真的在轉強。
- **EPS 尚未公布時會顯示「—」**，YoY 欄位會標示「尚未公布」，不會誤植為0或估算值。
                """)

            st.markdown("<br>", unsafe_allow_html=True)

            # 圖表繪製
            calc_cols_chart = ['pre_4','pre_3','pre_2','pre_1','day_open','day_close','post_1_close']
            labels_chart    = ["4天前","3天前","2天前","1天前","除息開盤","除息收盤","1天後"]
            timeline_avg    = [merged[c].mean() for c in calc_cols_chart]
            timeline_wr     = [len(merged[merged[c]>0]) / max(len(merged[c].dropna()),1) * 100 for c in calc_cols_chart]
            bar_colors      = ['#ef5350' if x > 0 else '#26a69a' for x in timeline_avg]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=labels_chart, y=timeline_avg, marker_color=bar_colors, text=[f"{x:+.2f}%" for x in timeline_avg], textposition='auto', name="平均漲跌幅"), secondary_y=False)
            fig.add_trace(go.Scatter(x=labels_chart, y=timeline_wr, mode='lines+markers+text', line=dict(color='#FFA500', width=2, dash='dot'), marker=dict(size=8, color='#FFA500'), text=[f"{x:.0f}%" for x in timeline_wr], textposition='top center', name="上漲勝率%"), secondary_y=True)
            fig.update_yaxes(title_text="平均漲跌幅 (%)", secondary_y=False, gridcolor='rgba(128,128,128,0.1)')
            fig.update_yaxes(title_text="上漲機率 (%)", secondary_y=True, range=[0, 115], showgrid=False)
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', y=1.15, x=0), hovermode='x unified', xaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{target_stock_id}")
            
            st.markdown("---")
            st.markdown(f"#### 📜 歷年詳細數據報表 <span style='font-size:13px;color:#888;font-weight:400;margin-left:10px'>共 {total_events} 筆 · 2015–2025</span>", unsafe_allow_html=True)

            # 填權息天數計算 + 【新增】除息前營收YoY 計算
            filled_days_list = []
            is_filled_list = []
            rev_yoy_before_list = []
            taiex_week_ret_list = []
            
            for idx, row in merged.iterrows():
                ex_date = row['ExDate']
                base_price = row['close_T_minus_1']
                
                sub_kline = kline[kline['date'] > ex_date].sort_values('date')
                
                filled = False
                days_to_fill = None
                
                if not pd.isna(base_price) and not sub_kline.empty:
                    for i, (_, k_row) in enumerate(sub_kline.iterrows(), start=1):
                        h_val = k_row.get('high')
                        if h_val is None or pd.isna(h_val):
                            h_val = k_row.get('max')
                            
                        if h_val is not None and not pd.isna(h_val) and h_val >= base_price:
                            filled = True
                            days_to_fill = i
                            break
                            
                is_filled_list.append(filled)
                filled_days_list.append(days_to_fill if filled else "-")

                # 【新增】取得該次除息日前，市場已公開的最新營收YoY
                rev_snap = get_revenue_snapshot_before(rev_df, ex_date)
                rev_yoy_before_list.append(rev_snap['yoy'] if rev_snap is not None else np.nan)

                # 【新增】計算除息當週大盤表現
                taiex_week_ret_list.append(get_taiex_week_return(taiex_df, ex_date))

            merged['is_filled'] = is_filled_list
            merged['filled_days'] = filled_days_list
            merged['rev_yoy_before'] = rev_yoy_before_list
            merged['taiex_week_ret'] = taiex_week_ret_list
            merged['is_peak_season'] = merged['ExDate'].dt.month.isin([7, 8, 9])

            # 組合報表輸出
            merged['year'] = merged['ExDate'].dt.year.astype(str)
            merged['date_str'] = merged['ExDate'].dt.strftime('%m/%d')
            final_data = []
            
            for _, row in merged.sort_values('ExDate', ascending=False).iterrows():
                yield_flag = " 🔥" if row['yield'] >= 5 else (" ✅" if row['yield'] >= 3 else "")
                
                if row['is_filled']:
                    fill_status = f"✅ 已填權息 ({row['filled_days']}天)"
                else:
                    fill_status = "❌ 未填權息"

                rev_yoy_disp = f"{row['rev_yoy_before']:+.1f}%" if not pd.isna(row['rev_yoy_before']) else "-"
                taiex_wr_disp = f"{row['taiex_week_ret']:+.1f}%" if row['taiex_week_ret'] is not None and not pd.isna(row['taiex_week_ret']) else "-"
                season_disp = "🔥旺季" if row['is_peak_season'] else "淡季"
                
                final_data.append({
                    '年度': row['year'], 
                    '除權息日': row['date_str'], 
                    '現金股利': f"{row['TotalCashDividend']:.2f}", 
                    '股票股利': f"{row['TotalStockDividend']:.2f}", 
                    '殖利率': f"{row['yield']:.2f}%{yield_flag}",
                    '除息前營收YoY': rev_yoy_disp,
                    '除息當週大盤%': taiex_wr_disp,
                    '季節': season_disp,
                    '填權息表現': fill_status,
                    '填權息天數': row['filled_days'],
                    '3天前': f"{row['pre_3']:+.2f}%", 
                    '2天前': f"{row['pre_2']:+.2f}%", 
                    '1天前': f"{row['pre_1']:+.2f}%", 
                    '開盤': f"{row['day_open']:+.2f}%", 
                    '收盤': f"{row['day_close']:+.2f}%"
                })

            total_cnt = len(merged)
            filled_cnt = sum(merged['is_filled'])
            fill_win_rate = (filled_cnt / total_cnt * 100) if total_cnt > 0 else 0
            
            valid_days = [d for d in merged['filled_days'] if isinstance(d, int)]
            avg_fill_days = (sum(valid_days) / len(valid_days)) if valid_days else 0

            stats_rows = [
                {
                    '年度': '📊 歷史填權息統計', 
                    '除權息日': '-', 
                    '現金股利': f"成功率: {fill_win_rate:.0f}%", 
                    '股票股利': f"平均 {avg_fill_days:.1f} 天", 
                    '殖利率': f"總計 {filled_cnt}/{total_cnt} 次",
                    '除息前營收YoY': '-',
                    '除息當週大盤%': '-',
                    '季節': '-',
                    '填權息表現': f"整體填權息勝率 {fill_win_rate:.1f}%",
                    '填權息天數': f"{avg_fill_days:.1f}天(平均)",
                    '3天前': '-', '2天前': '-', '1天前': '-', '開盤': '-', '收盤': '-'
                }
            ]

            df_final = pd.concat([pd.DataFrame(final_data), pd.DataFrame(stats_rows)], ignore_index=True)

            def style_cell(val):
                base = 'text-align: center; '
                if isinstance(val, str):
                    try:
                        num = float(val.replace('%','').replace('+','').strip().split(' ')[0])
                        if '+' not in val and num >= 0 and val.strip().endswith('%'): return base
                        if num > 0: return base + 'color: #ef5350; font-weight: 600;'
                        if num < 0: return base + 'color: #26a69a; font-weight: 600;'
                    except: pass
                elif isinstance(val, (int, float)):
                    if val > 0: return base + 'color: #ef5350; font-weight: 600;'
                    if val < 0: return base + 'color: #26a69a; font-weight: 600;'
                return base

            st.dataframe(
                df_final.style.map(style_cell)
                .set_table_styles([
                    dict(selector="th", props=[("text-align","center"),("font-size","12px")]), 
                    dict(selector="td", props=[("font-size","12px")])
                ]), 
                use_container_width=True, 
                height=620, 
                hide_index=True
            )

            # ==========================================
            # 6-2. 【新增】營收 vs 填權息 關聯性分析
            # ==========================================
            rev_valid = merged.dropna(subset=['rev_yoy_before'])
            if len(rev_valid) >= 3:
                pos_mask = rev_valid['rev_yoy_before'] > 0
                neg_mask = rev_valid['rev_yoy_before'] <= 0
                pos_cnt, neg_cnt = pos_mask.sum(), neg_mask.sum()
                pos_rate = (rev_valid.loc[pos_mask, 'is_filled'].mean() * 100) if pos_cnt > 0 else None
                neg_rate = (rev_valid.loc[neg_mask, 'is_filled'].mean() * 100) if neg_cnt > 0 else None

                pos_txt = f"{pos_rate:.0f}% ({pos_cnt}次)" if pos_rate is not None else "無樣本"
                neg_txt = f"{neg_rate:.0f}% ({neg_cnt}次)" if neg_rate is not None else "無樣本"

                if pos_rate is not None and neg_rate is not None:
                    diff = pos_rate - neg_rate
                    if diff > 10:
                        insight = f"營收動能對此股填權息有明顯正向影響（相差 {diff:+.0f} 個百分點），除息前留意最新一期營收公告。"
                    elif diff < -10:
                        insight = f"此股的填權息表現與除息前營收YoY呈反向關係（相差 {diff:+.0f} 個百分點），可能受股價位階、除息缺口大小等其他因素主導，營收非主要驅動力。"
                    else:
                        insight = "此股填權息表現與除息前營收YoY的關聯性不明顯，建議搭配籌碼面或大盤同步性等其他因子綜合判斷。"
                else:
                    insight = "樣本數不足以判斷關聯性，僅供參考。"

                st.markdown(f"""
                <div style="background:#f7f9fc;padding:14px 18px;border-radius:10px;border:1px solid rgba(128,128,128,0.15);margin-top:12px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:8px">📊 基本面關聯性分析 — 除息前營收YoY vs 填權息成功率</div>
                    <div style="display:flex;gap:24px;margin-bottom:8px;flex-wrap:wrap;">
                        <span style="font-size:13px;color:#666">營收YoY為正時填權息成功率：<b style="color:#ef5350">{pos_txt}</b></span>
                        <span style="font-size:13px;color:#666">營收YoY為負時填權息成功率：<b style="color:#26a69a">{neg_txt}</b></span>
                    </div>
                    <div style="font-size:12.5px;color:#888;line-height:1.5;">💡 {insight}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("⚠️ 月營收資料樣本不足（可能為新上市股或 FinMind 額度限制），暫無法進行營收關聯性分析。")

            # ==========================================
            # 6-3. 【新增】總體面關聯性分析：大盤同步性 + 除息旺季效應
            # ==========================================
            taiex_valid = merged.dropna(subset=['taiex_week_ret'])
            if len(taiex_valid) >= 3:
                bull_mask = taiex_valid['taiex_week_ret'] > 0
                bear_mask = taiex_valid['taiex_week_ret'] <= 0
                bull_cnt, bear_cnt = bull_mask.sum(), bear_mask.sum()
                bull_rate = (taiex_valid.loc[bull_mask, 'is_filled'].mean() * 100) if bull_cnt > 0 else None
                bear_rate = (taiex_valid.loc[bear_mask, 'is_filled'].mean() * 100) if bear_cnt > 0 else None

                bull_txt = f"{bull_rate:.0f}% ({bull_cnt}次)" if bull_rate is not None else "無樣本"
                bear_txt = f"{bear_rate:.0f}% ({bear_cnt}次)" if bear_rate is not None else "無樣本"

                if bull_rate is not None and bear_rate is not None:
                    mkt_diff = bull_rate - bear_rate
                    if mkt_diff > 10:
                        mkt_insight = f"除息當週大盤走勢對此股填權息有明顯正向影響（相差 {mkt_diff:+.0f} 個百分點），大盤同步性可能是比個股基本面更強的解釋變數。"
                    elif mkt_diff < -10:
                        mkt_insight = f"此股填權息表現與除息當週大盤走勢呈反向關係（相差 {mkt_diff:+.0f} 個百分點），較不受大盤氛圍主導，個股本身籌碼或基本面可能是更關鍵的因素。"
                    else:
                        mkt_insight = "此股填權息表現與除息當週大盤走勢的關聯性不明顯。"
                else:
                    mkt_insight = "樣本數不足以判斷關聯性，僅供參考。"

                st.markdown(f"""
                <div style="background:#f7f9fc;padding:14px 18px;border-radius:10px;border:1px solid rgba(128,128,128,0.15);margin-top:12px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:8px">🌐 總體面關聯性分析 — 大盤同步性 vs 填權息成功率</div>
                    <div style="display:flex;gap:24px;margin-bottom:8px;flex-wrap:wrap;">
                        <span style="font-size:13px;color:#666">除息當週大盤上漲時填權息成功率：<b style="color:#ef5350">{bull_txt}</b></span>
                        <span style="font-size:13px;color:#666">除息當週大盤下跌時填權息成功率：<b style="color:#26a69a">{bear_txt}</b></span>
                    </div>
                    <div style="font-size:12.5px;color:#888;line-height:1.5;">💡 {mkt_insight}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("⚠️ 大盤指數資料樣本不足，暫無法進行大盤同步性分析。")

            season_valid = merged.copy()
            if len(season_valid) >= 3:
                peak_mask = season_valid['is_peak_season'] == True
                offpeak_mask = season_valid['is_peak_season'] == False
                peak_cnt, offpeak_cnt = peak_mask.sum(), offpeak_mask.sum()
                peak_rate = (season_valid.loc[peak_mask, 'is_filled'].mean() * 100) if peak_cnt > 0 else None
                offpeak_rate = (season_valid.loc[offpeak_mask, 'is_filled'].mean() * 100) if offpeak_cnt > 0 else None

                peak_txt = f"{peak_rate:.0f}% ({peak_cnt}次)" if peak_rate is not None else "無樣本"
                offpeak_txt = f"{offpeak_rate:.0f}% ({offpeak_cnt}次)" if offpeak_rate is not None else "無樣本"

                if peak_rate is not None and offpeak_rate is not None:
                    season_diff = peak_rate - offpeak_rate
                    if season_diff > 10:
                        season_insight = f"7-9月除息旺季的填權息成功率明顯較高（相差 {season_diff:+.0f} 個百分點），可能與除息旺季資金集中回補、族群同步性效應有關。"
                    elif season_diff < -10:
                        season_insight = f"7-9月除息旺季的填權息成功率反而較低（相差 {season_diff:+.0f} 個百分點），旺季籌碼分散、資金排擠效應可能是原因之一。"
                    else:
                        season_insight = "旺季與非旺季的填權息成功率差異不明顯，此股的填權息表現較不受除息旺季效應影響。"
                else:
                    season_insight = "樣本數不足以判斷季節性效應，僅供參考。"

                st.markdown(f"""
                <div style="background:#f7f9fc;padding:14px 18px;border-radius:10px;border:1px solid rgba(128,128,128,0.15);margin-top:12px;">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:8px">📅 除息旺季效應 — 7-9月 vs 其他月份填權息成功率</div>
                    <div style="display:flex;gap:24px;margin-bottom:8px;flex-wrap:wrap;">
                        <span style="font-size:13px;color:#666">🔥 旺季(7-9月)填權息成功率：<b style="color:#ef5350">{peak_txt}</b></span>
                        <span style="font-size:13px;color:#666">其他月份填權息成功率：<b style="color:#26a69a">{offpeak_txt}</b></span>
                    </div>
                    <div style="font-size:12.5px;color:#888;line-height:1.5;">💡 {season_insight}</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"資料處理或繪圖發生錯誤: {e}")
