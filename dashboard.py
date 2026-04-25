"""
dashboard.py — 주식 스크리너 대시보드
실행: streamlit run dashboard.py
"""

import glob
import io as _io
import json
import os
import requests
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta

import numpy as np
try:
    from scipy.signal import argrelextrema
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

import FinanceDataReader as fdr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st

# ================================================
# 페이지 설정
# ================================================
st.set_page_config(page_title="주식 스크리너 대시보드", page_icon="📈", layout="wide")

st.markdown("""<style>
    [data-testid="stSidebar"] { min-width:220px !important; max-width:220px !important; }
    [data-testid="stSidebar"] .stSlider { margin-bottom:4px; }
    [data-testid="stSidebar"] label { font-size:0.78rem !important; }
    .last-updated { color:#555e7a; font-size:0.82rem; }
    .indicator-card {
        background:#f8faff; border-radius:12px; padding:16px 18px;
        border-left:4px solid #4a7afa; margin-bottom:4px;
        box-shadow:0 2px 8px rgba(74,122,250,0.08);
    }
    .indicator-title { font-size:0.9rem; font-weight:700; color:#1a1f36; }
    .indicator-value { font-size:1.5rem; font-weight:800; color:#4a7afa; }
    .indicator-desc  { font-size:0.78rem; color:#555e7a; margin-top:4px; }
    .macro-card {
        background:#ffffff; border-radius:14px; padding:14px 16px 12px;
        border:1px solid #e8ecf4; box-shadow:0 2px 12px rgba(0,0,0,0.06);
        height:130px; display:flex; flex-direction:column; justify-content:space-between;
    }
    .macro-label   { font-size:0.72rem; font-weight:600; color:#888ea8; letter-spacing:0.03em; }
    .macro-value   { font-size:1.25rem; font-weight:800; color:#1a1f36; margin:2px 0; }
    .macro-change  { font-size:0.8rem;  font-weight:700; }
    .macro-benefit { font-size:0.78rem; font-weight:600; color:#3d4a6b; margin-top:4px; }
</style>""", unsafe_allow_html=True)

# ================================================
# 상수
# ================================================
TRACKING_FILE       = "signal_tracking.json"
SIGNAL_HISTORY_FILE = "signal_history.csv"
GEMINI_MODEL        = "gemini-2.0-flash"
# .env 파일 자동 로드 (python-dotenv 없어도 동작)
def _load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
_load_dotenv()

DART_KEY = (
    os.environ.get("DART_KEY")
    or os.environ.get("DART_API_KEY")
    or os.environ.get("crtfc_key", "")
)

_PERIOD_MAP = {"3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y"}

_MACRO_BENEFIT = {
    "환율(KRW/USD)":        {True: "반도체/IT · 자동차 · 조선", False: "소비재 · 통신"},
    "WTI유가":              {True: "에너지/화학",               False: "소비재 · 항공"},
    "미국금리(10Y)":        {True: "금융/보험",                 False: "바이오 · IT · 건설"},
    "미국금리(2Y)":         {True: "금융/보험",                 False: "성장주"},
    "장단기금리차(10Y-2Y)": {True: "금융/보험",                 False: "방어주"},
    "미국CPI(YoY)":         {True: "에너지/화학 · 소재",        False: "소비재 · IT"},
    "한국CPI(YoY)":         {True: "에너지/화학",               False: "소비재"},
    "VIX":                  {True: "방어주 · 헬스케어",         False: "반도체/IT · 성장주"},
    "달러인덱스":           {True: "반도체/IT · 자동차",        False: "신흥시장 · 원자재"},
    "한국기준금리":         {True: "금융/보험",                 False: "건설 · 바이오"},
}

# ================================================
# Gemini AI 헬퍼
# ================================================
_KR_ONLY_PREFIX = (
    "당신은 한국어 전문 투자 분석가입니다. 반드시 한국어로만 답변하세요. "
    "다른 언어(중국어, 영어 등)는 절대 사용하지 마세요.\n\n"
)
_GEMINI_SYSTEM = (
    "당신은 한국어 전문 투자 분석가입니다. 반드시 한국어로만 답변하세요. "
    "다른 언어(중국어, 영어, 일본어 등)는 절대 사용하지 마세요. "
    "한자, 중국어 간체·번체 문자를 답변에 포함하지 마세요. "
    "모든 전문 용어도 한국어로 표현하세요."
)

def copy_button(text: str, label: str = "📋 결과 복사", key: str = ""):
    """st.code 내장 복사 버튼 활용 — expander 접힌 상태로 표시"""
    with st.expander(label):
        st.code(text, language=None)


def call_ollama(prompt: str) -> str:
    """Gemini API 호출 (함수명은 하위 호환성 유지)"""
    gemini_key = os.environ.get("GEMINI_KEY", "")
    if not gemini_key:
        return "❌ GEMINI_KEY 환경변수가 설정되지 않았습니다. .env 파일에 GEMINI_KEY=... 를 추가하세요."
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=_GEMINI_SYSTEM,
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini 오류: {e}"


@st.cache_data(ttl=1800)
def search_kr_stock(name: str) -> pd.DataFrame:
    """종목명(또는 코드)으로 KOSPI/KOSDAQ 검색"""
    try:
        kospi  = fdr.StockListing("KOSPI")[["Code", "Name"]].copy();  kospi["시장"]  = "KOSPI"
        kosdaq = fdr.StockListing("KOSDAQ")[["Code", "Name"]].copy(); kosdaq["시장"] = "KOSDAQ"
        all_s  = pd.concat([kospi, kosdaq], ignore_index=True)
        mask   = (all_s["Name"].str.contains(name, na=False) |
                  all_s["Code"].str.contains(name, na=False))
        return all_s[mask].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ================================================
# 데이터 로드
# ================================================
@st.cache_data(ttl=300)
def load_latest_csv(prefix: str):
    files = sorted(glob.glob(f"{prefix}_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame(), None
    try:
        df = pd.read_csv(files[0], encoding="utf-8-sig")
        if df.empty or len(df.columns) == 0:
            return pd.DataFrame(), None
        if "PER"    in df.columns: df = df[df["PER"]    > 0]
        if "PBR"    in df.columns: df = df[df["PBR"]    > 0]
        if "ROE(%)" in df.columns: df = df[(df["ROE(%)"] > 0) & (df["ROE(%)"] <= 500)]
        if "이론PBR" not in df.columns and all(c in df.columns for c in ["PER","PBR","ROE(%)"]):
            df["이론PBR"] = (df["PER"] * df["ROE(%)"] / 100).round(2)
            valid = df["이론PBR"] > 0
            df.loc[valid, "괴리율(%)"] = (
                (df.loc[valid,"이론PBR"] - df.loc[valid,"PBR"]) / df.loc[valid,"이론PBR"] * 100
            ).round(1)
        mtime = os.path.getmtime(files[0])
        return df.reset_index(drop=True), datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pd.DataFrame(), None


@st.cache_data(ttl=300)
def load_macro_csv():
    files = sorted(glob.glob("macro_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    try:
        return pd.read_csv(files[0], encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_us_sector_map():
    try:
        sp500 = fdr.StockListing("S&P500")
        if "Symbol" in sp500.columns and "Sector" in sp500.columns:
            return dict(zip(sp500["Symbol"], sp500["Sector"]))
    except Exception:
        pass
    return {}


# ================================================
# DART 분기 실적 헬퍼
# ================================================
@st.cache_data(ttl=86400)
def _load_dart_corp_map(dart_key: str) -> dict:
    """corpCode.xml 다운로드 → {stock_code: corp_code} 전체 맵 (1일 캐시)"""
    if not dart_key:
        return {}
    try:
        resp = requests.get(
            "https://opendart.fss.or.kr/api/corpCode.xml",
            params={"crtfc_key": dart_key},
            timeout=30,
        )
        zf   = zipfile.ZipFile(_io.BytesIO(resp.content))
        root = ET.fromstring(zf.read("CORPCODE.xml"))
        return {
            item.findtext("stock_code"): item.findtext("corp_code")
            for item in root.findall("list")
            if item.findtext("stock_code")
        }
    except Exception:
        return {}


def fetch_dart_corp_code(stock_code: str, dart_key: str) -> str:
    """주식코드 6자리 → DART corp_code (corpCode.xml 캐시 활용)"""
    corp_map = _load_dart_corp_map(dart_key)
    return corp_map.get(stock_code, "")


def _parse_dart_amount(s) -> int:
    if not s or str(s).strip() in ("-", ""):
        return 0
    try:
        return int(str(s).replace(",", "").replace(" ", ""))
    except Exception:
        return 0


@st.cache_data(ttl=7200)
def fetch_dart_quarterly(stock_code: str, dart_key: str) -> pd.DataFrame:
    """최근 8분기 매출액·영업이익 조회 (DART API). 연결재무 우선, 없으면 별도."""
    if not dart_key:
        return pd.DataFrame()
    corp_code = fetch_dart_corp_code(stock_code, dart_key)
    if not corp_code:
        return pd.DataFrame()

    from datetime import date
    current_year = date.today().year

    # reprt_code → (축약명, YTD 누적 성격)
    RCODES = [("11013", "Q1"), ("11012", "Q2"), ("11014", "Q3"), ("11011", "Q4")]

    def _fetch_ytd(year: int, reprt_code: str):
        for fs_div in ("CFS", "OFS"):
            try:
                resp = requests.get(
                    "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                    params={
                        "crtfc_key": dart_key, "corp_code": corp_code,
                        "bsns_year": str(year), "reprt_code": reprt_code,
                        "fs_div": fs_div,
                    },
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if data.get("status") != "000":
                    continue
                rev = op = None
                for item in data.get("list", []):
                    nm  = item.get("account_nm", "")
                    val = _parse_dart_amount(item.get("thstrm_amount"))
                    if rev is None and ("매출액" in nm or "수익(매출액)" in nm):
                        rev = val
                    if op is None and "영업이익" in nm and "손실" not in nm:
                        op = val
                if rev is not None:
                    return rev, (op or 0)
            except Exception:
                continue
        return None, None

    all_rows = []
    for year in (current_year - 1, current_year):
        ytd = {}
        for rc, _ in RCODES:
            rev, op = _fetch_ytd(year, rc)
            if rev is not None:
                ytd[rc] = (rev, op)

        q1_r, q1_o = ytd.get("11013", (None, None))
        h1_r, h1_o = ytd.get("11012", (None, None))
        nm_r, nm_o = ytd.get("11014", (None, None))
        fy_r, fy_o = ytd.get("11011", (None, None))

        if q1_r is not None:
            all_rows.append({"분기": f"{year}Q1", "매출액": q1_r, "영업이익": q1_o})
        if h1_r is not None:
            q2_r = h1_r - (q1_r or 0)
            q2_o = h1_o - (q1_o or 0)
            all_rows.append({"분기": f"{year}Q2", "매출액": q2_r, "영업이익": q2_o})
        if nm_r is not None and h1_r is not None:
            all_rows.append({"분기": f"{year}Q3", "매출액": nm_r - h1_r, "영업이익": nm_o - h1_o})
        if fy_r is not None and nm_r is not None:
            all_rows.append({"분기": f"{year}Q4", "매출액": fy_r - nm_r, "영업이익": fy_o - nm_o})
        elif fy_r is not None and q1_r is None and h1_r is None and nm_r is None:
            # 연간 데이터만 존재
            all_rows.append({"분기": f"{year}FY", "매출액": fy_r, "영업이익": fy_o})

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    valid = df["매출액"] > 0
    df.loc[valid, "영업이익률(%)"] = (
        df.loc[valid, "영업이익"] / df.loc[valid, "매출액"] * 100
    ).round(1)
    df["매출액(억)"]   = (df["매출액"]   / 1e8).round(1)
    df["영업이익(억)"] = (df["영업이익"] / 1e8).round(1)
    return df


@st.cache_data(ttl=86400)
def fetch_dart_annual(stock_code: str, dart_key: str) -> pd.DataFrame:
    """최근 5개년 연간 매출액·영업이익 조회 (DART API). 연결재무 우선, 없으면 별도."""
    if not dart_key:
        return pd.DataFrame()
    corp_code = fetch_dart_corp_code(stock_code, dart_key)
    if not corp_code:
        return pd.DataFrame()

    from datetime import date
    current_year = date.today().year

    def _fetch_annual(year: int):
        for fs_div in ("CFS", "OFS"):
            try:
                resp = requests.get(
                    "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                    params={
                        "crtfc_key": dart_key, "corp_code": corp_code,
                        "bsns_year": str(year), "reprt_code": "11011",
                        "fs_div": fs_div,
                    },
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if data.get("status") != "000":
                    continue
                rev = op = None
                for item in data.get("list", []):
                    nm  = item.get("account_nm", "")
                    val = _parse_dart_amount(item.get("thstrm_amount"))
                    if rev is None and ("매출액" in nm or "수익(매출액)" in nm):
                        rev = val
                    if op is None and "영업이익" in nm and "손실" not in nm:
                        op = val
                if rev is not None:
                    return rev, (op or 0)
            except Exception:
                continue
        return None, None

    all_rows = []
    for year in range(current_year - 5, current_year):
        rev, op = _fetch_annual(year)
        if rev is not None and rev > 0:
            all_rows.append({"연도": str(year), "매출액": rev, "영업이익": op})

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    valid = df["매출액"] > 0
    df.loc[valid, "영업이익률(%)"] = (
        df.loc[valid, "영업이익"] / df.loc[valid, "매출액"] * 100
    ).round(1)
    df["매출액(억)"]   = (df["매출액"]   / 1e8).round(1)
    df["영업이익(억)"] = (df["영업이익"] / 1e8).round(1)
    return df


# ================================================
# 추적검증
# ================================================
def load_tracking():
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_tracking(records):
    try:
        with open(TRACKING_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_signal_history() -> pd.DataFrame:
    _cols = ["날짜", "종목명", "코드", "시장", "주가",
             "펀더멘탈신호", "기술적신호", "종합신호",
             "단기목표가", "중기목표가", "손절기준"]
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            df = pd.read_csv(SIGNAL_HISTORY_FILE, encoding="utf-8-sig")
            # 구형 파일에 새 컬럼 없으면 추가
            for c in ["단기목표가", "중기목표가", "손절기준"]:
                if c not in df.columns:
                    df[c] = ""
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=_cols)


def save_signal_history(row_dict: dict):
    df = load_signal_history()
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(SIGNAL_HISTORY_FILE, index=False, encoding="utf-8-sig")


_TRACK_HORIZONS = [5, 10, 20, 60, 120]


def add_tracking_record(종목명, 코드, 시장, 가격, 펀더멘탈, 기술적):
    records = load_tracking()
    cutoff  = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if any(r["코드"] == 코드 and r["기록일"] >= cutoff for r in records):
        return False
    rec = {
        "기록일": datetime.now().strftime("%Y-%m-%d"),
        "종목명": 종목명, "코드": 코드, "시장": 시장,
        "기록가격": 가격, "펀더멘탈": 펀더멘탈, "기술적": 기술적,
    }
    for h in _TRACK_HORIZONS:
        rec[f"{h}일수익률"] = None
    save_tracking(records + [rec])
    return True


def update_tracking_prices(records):
    updated = False
    for r in records:
        try:
            rec_date_str = r["기록일"]
            시장  = r.get("시장", "")
            코드  = str(r.get("코드", ""))
            sfx  = ".KS" if 시장 == "KOSPI" else ".KQ" if 시장 == "KOSDAQ" else ""
            ticker = (코드.zfill(6) if 시장 in ("KOSPI","KOSDAQ") else 코드) + sfx
            base = r.get("기록가격")
            if not base:
                continue

            # need up to 120 trading days → ~180 calendar days
            start_dt = (datetime.strptime(rec_date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            data = yf.download(ticker, start=start_dt, progress=False, auto_adjust=True)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.reset_index()
            date_col = "Date" if "Date" in data.columns else data.columns[0]
            data_after = data[data[date_col] >= pd.Timestamp(rec_date_str)].reset_index(drop=True)

            for h in _TRACK_HORIZONS:
                col = f"{h}일수익률"
                if r.get(col) is None and len(data_after) > h:
                    price = float(data_after.iloc[h]["Close"])
                    r[col] = round((price - base) / base * 100, 2)
                    updated = True
        except Exception:
            continue
    if updated:
        save_tracking(records)
    return records

# ================================================
# 점수 계산
# ================================================
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not all(c in out.columns for c in ["ROE(%)", "PER", "PBR"]):
        return out
    has_sect = "섹터" in out.columns and out["섹터"].nunique() > 1

    def _sr(col, asc=True):
        # 섹터 내 3개 이상이면 섹터 상대랭크, 미만이면 전체 글로벌랭크
        # (단독 섹터에서 rank(pct=True)=1.0 → 1-1.0=0.0 되는 버그 방지)
        global_r = out[col].rank(pct=True)
        if has_sect:
            sect_sizes = out.groupby("섹터")[col].transform("count")
            sect_r    = out.groupby("섹터")[col].rank(pct=True)
            r = sect_r.where(sect_sizes >= 3, global_r)
        else:
            r = global_r
        return r if asc else 1 - r

    if "섹터평균PBR" in out.columns:
        # 한국: 업종내 상대PBR 40% + ROE업종대비 30% + PER업종대비 30%
        out["종합점수"] = (_sr("PBR", False)*0.4 + _sr("ROE(%)", True)*0.3 + _sr("PER", False)*0.3).mul(100).round(1)
    else:
        # 미국: 이론PBR괴리율 40% + 업종내PBR 30% + ROE업종대비 30%
        if "이론PBR" not in out.columns:
            out["이론PBR"] = (out["PER"] * out["ROE(%)"] / 100).round(2)
        if "괴리율(%)" not in out.columns:
            valid = out["이론PBR"] > 0
            out.loc[valid, "괴리율(%)"] = (
                (out.loc[valid,"이론PBR"] - out.loc[valid,"PBR"]) / out.loc[valid,"이론PBR"] * 100
            ).round(1)
        gap_s = out["괴리율(%)"].fillna(out["괴리율(%)"].median()).rank(pct=True)
        out["종합점수"] = (gap_s*0.4 + _sr("PBR", False)*0.3 + _sr("ROE(%)", True)*0.3).mul(100).round(1)
    return out

# ================================================
# 기술적 지표
# ================================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA20"]  = d["Close"].rolling(20).mean()
    d["MA60"]  = d["Close"].rolling(60).mean()
    d["MA120"] = d["Close"].rolling(120).mean()
    d["BB_mid"]   = d["MA20"]
    d["BB_std"]   = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - 2 * d["BB_std"]
    delta = d["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"]   = d["MACD"] - d["MACD_signal"]
    d["Vol_MA20"]    = d["Volume"].rolling(20).mean()
    # OBV: 상승일 +거래량, 하락일 -거래량 누적
    direction = d["Close"].diff().fillna(0)
    signed_vol = d["Volume"] * direction.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    d["OBV"] = signed_vol.cumsum()
    return d

# ================================================
# 신호 계산
# ================================================
def _fund_signal(row) -> str:
    """펀더멘탈 신호: 종합점수 기반 — 70↑ 매수, 45~70 중립, 45↓ 매도
    (스크리너 통과 종목 중 전체 글로벌랭크 기반 → 상위 30% 매수 / 중간 40% 중립 / 하위 30% 매도)
    """
    try:
        score = float(row.get("종합점수") or 0)
        if score >= 70: return "🟢 매수"
        if score >= 45: return "🟡 중립"
        return "🔴 매도"
    except Exception:
        return "🟡 중립"


@st.cache_data(ttl=3600)
def get_tech_signal(ticker: str, period: str = "1y") -> str:
    """기술적 신호: MA·RSI·MACD·거래량 기반"""
    try:
        df = fetch_stock_data(ticker, period)
        if df.empty or len(df) < 26:
            return "—"
        df     = calc_indicators(df)
        last   = df.iloc[-1]
        close  = float(last["Close"])
        ma20   = float(last["MA20"])
        ma20_p = float(df["MA20"].iloc[-6]) if len(df) >= 6 else ma20
        rsi    = float(last["RSI"])
        macd   = float(last["MACD"])
        sig    = float(last["MACD_signal"])
        vol    = float(last["Volume"])
        vol_ma = float(last["Vol_MA20"]) if float(last["Vol_MA20"]) > 0 else 1
        pos = sum([close > ma20, ma20 > ma20_p, 30 < rsi < 70, macd > sig, vol >= vol_ma])
        if rsi >= 70: pos -= 1
        if pos >= 4: return "🟢 매수"
        if pos >= 2: return "🟡 중립"
        return "🔴 매도"
    except Exception:
        return "—"


@st.cache_data(ttl=3600)
def run_backtest(ticker: str, years: int) -> dict:
    """선택 기간 내 매수 신호 발생 시점 탐지 및 forward return 계산"""
    HORIZONS = [5, 10, 20, 60, 120]
    dl_period = f"{years + 2}y"
    df = fetch_stock_data(ticker, dl_period)
    if df.empty or len(df) < 60:
        return {}

    df = calc_indicators(df)
    df = df.dropna(subset=["MA20", "RSI", "MACD", "MACD_signal", "Vol_MA20"])
    if df.empty:
        return {}

    cutoff = df.index[-1] - pd.DateOffset(years=years)
    df_test = df[df.index >= cutoff].copy()
    df_test = df_test.reset_index()
    df_full = df.reset_index()

    signals = []
    for i in range(len(df_test)):
        row     = df_test.iloc[i]
        prev_ma = df_test["MA20"].iloc[i - 6] if i >= 6 else df_test["MA20"].iloc[0]
        close   = float(row["Close"])
        ma20    = float(row["MA20"])
        rsi     = float(row["RSI"])
        macd    = float(row["MACD"])
        sig_v   = float(row["MACD_signal"])
        vol     = float(row["Volume"])
        vol_ma  = float(row["Vol_MA20"]) if float(row["Vol_MA20"]) > 0 else 1

        pos = sum([close > ma20, ma20 > prev_ma, 30 < rsi < 70, macd > sig_v, vol >= vol_ma])
        if rsi >= 70:
            pos -= 1
        if pos < 4:
            continue

        entry_date  = row["Date"] if "Date" in row else row.name
        entry_price = close

        # forward return: df_full 기준 인덱스 탐색
        full_mask = df_full["Date"] == entry_date if "Date" in df_full.columns else None
        if full_mask is not None and full_mask.any():
            fi = int(df_full.index[full_mask][0])
        else:
            continue

        rets = {}
        for h in HORIZONS:
            fj = fi + h
            if fj < len(df_full):
                rets[h] = (float(df_full.iloc[fj]["Close"]) - entry_price) / entry_price * 100
            else:
                rets[h] = None

        signals.append({
            "날짜":   entry_date,
            "진입가": round(entry_price, 2),
            "RSI":    round(rsi, 1),
            "pos":    pos,
            **{f"{h}일수익률": (round(rets[h], 2) if rets[h] is not None else None) for h in HORIZONS},
        })

    ohlc_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    available = [c for c in ohlc_cols if c in df_test.columns]
    return {
        "signals":  signals,
        "df_ohlc":  df_test[available],
        "ticker":   ticker,
        "years":    years,
    }


def generate_target_prices(name: str, code: str, ticker: str,
                           per=None, pbr=None, roe=None, period: str = "1y") -> dict:
    """Ollama로 단기/중기 목표가·손절기준 자동 생성. 실패 시 빈 dict 반환."""
    try:
        df = fetch_stock_data(ticker, period)
        if df.empty or len(df) < 26:
            return {}
        df = calc_indicators(df)
        last    = df.iloc[-1]
        close   = float(last["Close"])
        ma20    = float(last["MA20"])
        rsi_v   = float(last["RSI"])
        macd_v  = float(last["MACD"])
        macd_s  = float(last["MACD_signal"])
        bb_up   = float(last["BB_upper"])
        vol_r   = float(last["Volume"]) / max(float(last["Vol_MA20"]), 1)
        ma_up   = ma20 > float(df["MA20"].iloc[-6]) if len(df) >= 6 else False

        fund_str = ""
        if any(v is not None for v in [per, pbr, roe]):
            fund_str = f"PER={per}, PBR={pbr}, ROE={roe}%"

        prompt = _KR_ONLY_PREFIX + f"""당신은 한국 주식 전문 애널리스트입니다. 아래 데이터를 바탕으로 {name}({code}) 종목의 목표가를 제시해주세요.

[기술적 지표]
현재가: {close:,.0f} | MA20: {ma20:,.0f}({'상승' if ma_up else '하락'}추세) | RSI: {rsi_v:.1f}
MACD: {'골든크로스' if macd_v > macd_s else '데드크로스'} | 볼린저밴드 상단: {bb_up:,.0f} | 거래량: 평균대비 {vol_r:.1f}배
{f'[펀더멘탈] {fund_str}' if fund_str else ''}

아래 형식으로만 답변해주세요 (각 항목 숫자 위주 1~2문장):
[단기목표가] 볼린저밴드 상단({bb_up:,.0f}) 기준 단기 시나리오 (구체적 가격 제시)
[중기목표가] 적정PER 기준 중기 목표가 (구체적 가격 제시)
[손절기준] MA20({ma20:,.0f}) 이탈 시 대응"""

        result = call_ollama(prompt)
        sections = {"단기목표가": "", "중기목표가": "", "손절기준": ""}
        current = None
        for line in result.split("\n"):
            for key_s in sections:
                if f"[{key_s}]" in line:
                    current = key_s
                    line = line.replace(f"[{key_s}]", "").strip()
                    break
            if current:
                sections[current] += line + " "
        return {k: v.strip() for k, v in sections.items()}
    except Exception:
        return {}


def add_signals(df: pd.DataFrame, ticker_suffix: str, is_kr_flag: bool,
                period: str = "1y") -> pd.DataFrame:
    out   = df.copy()
    codes = df.head(30)["코드"].astype(str).tolist()

    # 기술적 신호 (상위 30개까지 API 호출)
    tech = {}
    for code in codes:
        t = (code.zfill(6) if is_kr_flag else code) + ticker_suffix
        tech[code] = get_tech_signal(t, period)
    out["기술적신호"] = out["코드"].astype(str).map(tech).fillna("—")

    # 펀더멘탈 신호 (종합점수 기반 80/60 기준)
    out["펀더멘탈신호"] = out.apply(_fund_signal, axis=1)

    # 종합 신호 — 두 신호의 조합
    def _combined(row):
        f = row["펀더멘탈신호"]
        t = row["기술적신호"]
        f_buy  = "매수" in f
        f_sell = "매도" in f
        t_buy  = "매수" in t
        t_sell = "매도" in t
        if   f_buy  and t_buy:              return "⭐ 강력매수"
        elif f_buy  and not t_sell:         return "🟢 매수검토"   # 펀더멘탈매수 + 기술중립
        elif f_buy  and t_sell:             return "🟡 관망"       # 펀더멘탈매수 + 기술매도
        elif not f_sell and t_buy:          return "🔵 기술적↑"   # 펀더멘탈중립 + 기술매수
        elif f_sell and t_buy:              return "🟡 관망"       # 펀더멘탈매도 + 기술매수
        elif f_sell and t_sell:             return "🔴 매도"
        elif f_sell:                        return "🔴 매도주의"   # 펀더멘탈매도 + 기술중립
        else:                               return "🟡 중립"
    out["종합신호"] = out.apply(_combined, axis=1)
    return out

# ================================================
# 차트 렌더러
# ================================================
def render_stock_chart(code: str, name: str, ticker_suffix: str, is_kr_chart: bool,
                       stock_data: dict = None, df_macro: pd.DataFrame = None,
                       period: str = "1y"):
    ticker = f"{code}{ticker_suffix}" if ticker_suffix else code
    with st.spinner(f"{name} 데이터 불러오는 중..."):
        raw = fetch_stock_data(ticker, period)
    if raw.empty:
        st.error(f"데이터를 가져올 수 없습니다 ({ticker})")
        return

    df      = calc_indicators(raw)
    display = df          # 선택한 기간 전체 표시
    last    = df.iloc[-1]
    close   = float(last["Close"])

    # ── 5행 서브플롯 (캔들 / 거래량 / OBV / RSI / MACD) ──
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.42, 0.12, 0.13, 0.165, 0.165],
        vertical_spacing=0.022,
        subplot_titles=(
            "주가 · 이동평균 · 볼린저밴드",
            "거래량 (주황=평균200%↑)",
            "OBV (On-Balance Volume)",
            "RSI (14)",
            "MACD",
        ),
    )

    # ── Row 1: 캔들스틱 + 이동평균 + 볼린저밴드 ──
    fig.add_trace(go.Candlestick(
        x=display.index, open=display["Open"], high=display["High"],
        low=display["Low"], close=display["Close"], name="주가",
        increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
        increasing_fillcolor="#ef5350", decreasing_fillcolor="#26a69a",
    ), row=1, col=1)
    for ma, color in [("MA20","#f7b731"), ("MA60","#4a7afa"), ("MA120","#9b59b6")]:
        fig.add_trace(go.Scatter(
            x=display.index, y=display[ma], name=ma,
            line=dict(color=color, width=1.5), opacity=0.9,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=display.index, y=display["BB_upper"],
        line=dict(color="#adb5bd", width=1, dash="dot"), showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=display.index, y=display["BB_lower"],
        fill="tonexty", fillcolor="rgba(173,181,189,0.08)",
        line=dict(color="#adb5bd", width=1, dash="dot"), showlegend=False,
    ), row=1, col=1)

    # ── 지지선/저항선 자동 탐지 (scipy argrelextrema, order=5) ──
    if _SCIPY_OK and len(display) > 12:
        hi_vals = display["High"].values
        lo_vals = display["Low"].values
        hi_idx  = argrelextrema(hi_vals, np.greater, order=5)[0]
        lo_idx  = argrelextrema(lo_vals, np.less,    order=5)[0]
        resist_levels = sorted(hi_vals[hi_idx], reverse=True)[:3] if len(hi_idx) >= 1 else []
        support_levels= sorted(lo_vals[lo_idx])[:3]               if len(lo_idx) >= 1 else []
        x_end = display.index[-1]
        for i, lvl in enumerate(resist_levels):
            fig.add_hline(
                y=lvl, line_dash="dot",
                line_color="rgba(239,83,80,0.7)", line_width=1.5,
                row=1, col=1,
            )
            fig.add_annotation(
                x=x_end, y=lvl, xref="x", yref="y",
                text=f" R{i+1}: {lvl:,.0f}",
                showarrow=False, xanchor="left",
                font=dict(size=9, color="#ef5350"),
            )
        for i, lvl in enumerate(support_levels):
            fig.add_hline(
                y=lvl, line_dash="dot",
                line_color="rgba(38,166,154,0.7)", line_width=1.5,
                row=1, col=1,
            )
            fig.add_annotation(
                x=x_end, y=lvl, xref="x", yref="y",
                text=f" S{i+1}: {lvl:,.0f}",
                showarrow=False, xanchor="left",
                font=dict(size=9, color="#26a69a"),
            )

    # ── Row 2: 거래량 (200% 이상 주황색 강조) + 20일 평균 빨간 점선 ──
    vol_colors = []
    for c, o, v, vm in zip(
        display["Close"], display["Open"],
        display["Volume"], display["Vol_MA20"].fillna(0)
    ):
        if vm > 0 and v >= vm * 2:
            vol_colors.append("#ff7800")   # 주황: 평균 200% 이상
        elif c >= o:
            vol_colors.append("#ef5350")   # 빨강: 양봉
        else:
            vol_colors.append("#26a69a")   # 초록: 음봉
    fig.add_trace(go.Bar(
        x=display.index, y=display["Volume"],
        marker_color=vol_colors, opacity=0.75, name="거래량", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=display.index, y=display["Vol_MA20"],
        line=dict(color="#e64553", width=1.5, dash="dot"),
        name="거래량MA20", showlegend=False,
    ), row=2, col=1)

    # ── Row 3: OBV ──
    fig.add_trace(go.Scatter(
        x=display.index, y=display["OBV"],
        line=dict(color="#4a7afa", width=1.5),
        name="OBV", showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888ea8", row=3, col=1)

    # ── Row 4: RSI ──
    fig.add_trace(go.Scatter(x=display.index, y=display["RSI"],
        line=dict(color="#4a7afa", width=1.5), showlegend=False), row=4, col=1)
    for lvl, clr in [(70,"rgba(239,83,80,0.4)"), (30,"rgba(38,166,154,0.4)")]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=clr, row=4, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(173,181,189,0.05)", line_width=0, row=4, col=1)

    # ── Row 5: MACD ──
    hist_colors = ["#ef5350" if v >= 0 else "#26a69a" for v in display["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=display.index, y=display["MACD_hist"],
        marker_color=hist_colors, opacity=0.6, showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=display.index, y=display["MACD"],
        line=dict(color="#4a7afa", width=1.5), showlegend=False), row=5, col=1)
    fig.add_trace(go.Scatter(x=display.index, y=display["MACD_signal"],
        line=dict(color="#f7b731", width=1.5), showlegend=False), row=5, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888ea8", row=5, col=1)

    fig.update_layout(
        height=900, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=80, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 6):
        fig.update_xaxes(showgrid=True, gridcolor="#e8ecf4", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#e8ecf4", row=i, col=1)
    fig.update_yaxes(range=[20, 80], row=4, col=1)
    st.plotly_chart(fig, width="stretch")

    # ── 지표 계산 ──
    ma20      = float(last["MA20"])
    rsi       = float(last["RSI"])
    macd_v    = float(last["MACD"])
    macd_sig  = float(last["MACD_signal"])
    vol       = float(last["Volume"])
    vol_ma    = float(last["Vol_MA20"]) if float(last["Vol_MA20"]) > 0 else 1
    bb_up     = float(last["BB_upper"])
    bb_lo     = float(last["BB_lower"])
    bb_w      = bb_up - bb_lo
    bb_pct    = (close - bb_lo) / bb_w * 100 if bb_w > 0 else 50
    ma20_prev = float(df["MA20"].iloc[-6]) if len(df) >= 6 else ma20
    hist_prev = float(df["MACD_hist"].iloc[-2]) if len(df) >= 2 else 0
    vol_ratio = vol / vol_ma
    ma_up     = ma20 > ma20_prev

    ma_txt = (
        f"20일선({ma20:,.0f}) **상승 추세** | 현재가 MA20 {'위 ✓' if close > ma20 else '아래'}"
        if ma_up else
        f"20일선({ma20:,.0f}) **하락 추세** | 현재가 MA20 {'위' if close > ma20 else '아래'}"
    )
    bb_txt = (
        f"상단 근접 ({bb_pct:.0f}%) — 과열 주의"   if bb_pct > 80 else
        f"하단 근접 ({bb_pct:.0f}%) — 반등 가능성" if bb_pct < 20 else
        f"중간 구간 ({bb_pct:.0f}%) — 중립"
    )
    rsi_txt = (
        f"**{rsi:.1f}** — 🔴 과매수 (매도 신호)" if rsi >= 70 else
        f"**{rsi:.1f}** — 🟢 과매도 (반등 기대)"  if rsi <= 30 else
        f"**{rsi:.1f}** — 🟡 중립 구간"
    )
    if   macd_v > macd_sig and hist_prev <= 0: macd_txt = "**골든크로스** 발생 🟢 — 상승 전환"
    elif macd_v < macd_sig and hist_prev >= 0: macd_txt = "**데드크로스** 발생 🔴 — 하락 전환"
    elif macd_v > macd_sig:                    macd_txt = "시그널 **상회** — 상승 모멘텀 유지"
    else:                                      macd_txt = "시그널 **하회** — 하락 압력 지속"
    vol_txt = (
        f"평균 대비 **{vol_ratio:.1f}배** — 강한 거래" if vol_ratio >= 1.5 else
        f"평균 대비 **{vol_ratio:.1f}배** — 보통 수준" if vol_ratio >= 1.0 else
        f"평균 대비 **{vol_ratio:.1f}배** — 거래 위축"
    )

    signals = [close > ma20, ma_up, 30 < rsi < 70, macd_v > macd_sig, vol_ratio >= 1.0]
    pos = sum(signals)
    if rsi >= 70: pos -= 1
    if pos >= 4:   sig_color, sig_text, sig_bg = "#40a02b", "🟢 매수 우호적", "#f0fff4"
    elif pos >= 2: sig_color, sig_text, sig_bg = "#df8e1d", "🟡 중립",        "#fffdf0"
    else:          sig_color, sig_text, sig_bg = "#e64553", "🔴 매도 우호적", "#fff5f5"

    # 52주 고저 (복사 텍스트용 + 카드용 공통 계산)
    _w52_h   = float(df["High"].max())
    _w52_l   = float(df["Low"].min())
    _w52_pct = (close - _w52_l) / (_w52_h - _w52_l) * 100 if _w52_h > _w52_l else 50
    _fmt     = (lambda v: f"₩{v:,.0f}") if is_kr_chart else (lambda v: f"${v:.2f}")

    # 가치평가 텍스트 (stock_data 있을 때)
    _sd = stock_data or {}
    def _sv(k):
        try: return float(_sd[k]) if _sd.get(k) not in (None, "", "nan") else None
        except: return None
    _cp_per = _sv("PER"); _cp_pbr = _sv("PBR"); _cp_roe = _sv("ROE(%)")
    def _pg(v, t):
        if v is None: return "—"
        if t == "per":  return "저평가" if v<=10 else "적정" if v<=20 else "고평가주의" if v<=30 else "고평가"
        if t == "pbr":  return "저평가" if v<=1  else "적정" if v<=2  else "고평가주의" if v<=4  else "고평가"
        if t == "roe":  return "우수" if v>=15 else "보통" if v>=8 else "미흡" if v>=0 else "적자"
    _val_line = "  ".join(filter(None, [
        f"PER {_cp_per:.1f}({_pg(_cp_per,'per')})" if _cp_per is not None else "",
        f"PBR {_cp_pbr:.2f}({_pg(_cp_pbr,'pbr')})" if _cp_pbr is not None else "",
        f"ROE {_cp_roe:.1f}%({_pg(_cp_roe,'roe')})" if _cp_roe is not None else "",
    ]))

    # ── 해설 카드 ──
    st.markdown(f"#### 📊 {name} 기술적 분석 해설")
    # ── 복사 버튼 ──
    _rsi_st_pre  = ("과매수" if rsi >= 70 else "과매도" if rsi <= 30 else "중립")
    _macd_st_pre = ("골든크로스" if macd_v > macd_sig else "데드크로스" if macd_v < macd_sig else "시그널상회" if macd_v > macd_sig else "시그널하회")
    _chart_copy_text = "\n".join([
        f"[차트 분석] {name} ({code})",
        f"분석일: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "[ 기술적 지표 ]",
        f"  현재가: {_fmt(close)}  MA20: {_fmt(ma20)} ({'상승' if ma_up else '하락'}추세)",
        f"  RSI: {rsi:.1f} ({_rsi_st_pre})",
        f"  MACD: {_macd_st_pre}",
        f"  볼린저밴드: {bb_pct:.0f}% ({'상단 근접' if bb_pct>80 else '하단 근접' if bb_pct<20 else '중간'})",
        f"  거래량: 평균 대비 {vol_ratio:.1f}배",
        "",
        f"[ 기술적 종합 ] RSI {rsi:.1f}({_rsi_st_pre}) / MACD {_macd_st_pre} → {sig_text} (긍정신호 {pos}/5)",
        "",
        "[ 가치평가 ]",
        f"  {_val_line}" if _val_line else "  데이터 없음",
        "",
        "[ 52주 고저 ]",
        f"  최저 {_fmt(_w52_l)} / 현재 {_fmt(close)} ({_w52_pct:.0f}%) / 최고 {_fmt(_w52_h)}",
    ])
    copy_button(_chart_copy_text, "📋 분석 결과 복사", key=f"chart_{code}")
    card = lambda icon, title, body, border: f"""
<div style="background:#f8faff;border-radius:10px;padding:12px 16px;margin-bottom:8px;
            border-left:3px solid {border};font-size:0.87rem;line-height:1.5">
  {icon} <b>{title}</b><br>{body}
</div>"""
    ca, cb = st.columns(2)
    with ca:
        st.markdown(card("📈","이동평균선", ma_txt,  "#4a7afa"), unsafe_allow_html=True)
        st.markdown(card("📊","볼린저밴드", bb_txt,  "#adb5bd"), unsafe_allow_html=True)
    with cb:
        st.markdown(card("⚡","RSI",        rsi_txt, "#f7b731"), unsafe_allow_html=True)
        st.markdown(card("📉","MACD",       macd_txt,"#9b59b6"), unsafe_allow_html=True)
    st.markdown(card("🔊","거래량", vol_txt, "#26a69a"), unsafe_allow_html=True)

    # ── 기술적 신호 종합 요약 (RSI · MACD · OBV 한 줄) ──
    _rsi_st  = ("과매수" if rsi >= 70 else "과매도" if rsi <= 30 else "중립")
    _macd_st = ("골든크로스" if macd_v > macd_sig and float(df["MACD_hist"].iloc[-2] if len(df) >= 2 else 0) <= 0
                else "데드크로스" if macd_v < macd_sig and float(df["MACD_hist"].iloc[-2] if len(df) >= 2 else 0) >= 0
                else "상승 중" if macd_v > macd_sig else "하락 중")
    # OBV 추세: 최근 5봉 OBV 기울기
    _obv_vals = df["OBV"].dropna().tail(10)
    if len(_obv_vals) >= 5:
        _obv_slope = float(_obv_vals.iloc[-1]) - float(_obv_vals.iloc[-5])
        _obv_st = "상승" if _obv_slope > 0 else "하락"
    else:
        _obv_st = "—"
    # 종합 판단
    _bull = sum([rsi < 70 and rsi > 40, macd_v > macd_sig, _obv_st == "상승"])
    _bear = sum([rsi < 30, macd_v < macd_sig, _obv_st == "하락"])
    if _bull >= 2 and _bear == 0:
        _overall_st, _ov_color = "단기 상승 우위, 매수 검토", "#40a02b"
    elif _bear >= 2 and _bull == 0:
        _overall_st, _ov_color = "단기 하락 우위, 관망 권고", "#e64553"
    else:
        _overall_st, _ov_color = "추세 혼조, 관망 추천", "#df8e1d"
    _summary = (
        f"RSI {rsi:.1f} ({_rsi_st}) &nbsp;·&nbsp; "
        f"MACD {_macd_st} &nbsp;·&nbsp; "
        f"OBV {_obv_st} &nbsp;→&nbsp; "
        f"<span style='color:{_ov_color};font-weight:700'>{_overall_st}</span>"
    )
    st.markdown(f"""
<div style="background:#1a1f36;border-radius:10px;padding:12px 18px;margin-bottom:8px;
            border-left:3px solid {_ov_color};font-size:0.87rem;color:#e0e4ef;line-height:1.5">
  🔭 <b style="color:#fff">기술적 종합 요약</b><br>{_summary}
</div>""", unsafe_allow_html=True)

    # ── 가치평가 카드 ──
    if stock_data:
        def _safe(v):
            try:
                return float(v) if v is not None and str(v) not in ("","nan") else None
            except Exception:
                return None
        actual_p  = _safe(stock_data.get("PBR"))
        theory_p  = _safe(stock_data.get("이론PBR"))
        gap       = _safe(stock_data.get("괴리율(%)"))
        stock_per = _safe(stock_data.get("PER"))
        stock_roe = _safe(stock_data.get("ROE(%)"))

        # 절대값 배지 함수 (한국/미국 공통)
        def _per_badge(v):
            if v is None: return "—"
            if v <= 10:  return "🟢 저평가"
            if v <= 20:  return "🟡 적정"
            if v <= 30:  return "🟠 고평가주의"
            return "🔴 고평가"
        def _pbr_badge(v):
            if v is None: return "—"
            if v <= 1:   return "🟢 저평가"
            if v <= 2:   return "🟡 적정"
            if v <= 4:   return "🟠 고평가주의"
            return "🔴 고평가"
        def _roe_badge(v):
            if v is None: return "—"
            if v >= 15:  return "🟢 우수"
            if v >= 8:   return "🟡 보통"
            if v >= 0:   return "🟠 미흡"
            return "🔴 적자"

        def _render_abs_val_card():
            if not any(v is not None for v in [stock_per, actual_p, stock_roe]):
                return
            _border = "#40a02b" if (
                (stock_per is not None and stock_per <= 20) and
                (actual_p  is not None and actual_p  <= 2)  and
                (stock_roe is not None and stock_roe >= 8)
            ) else "#4a7afa"
            per_str = f"PER &nbsp;<b>{stock_per:.1f}</b>&nbsp;{_per_badge(stock_per)}" if stock_per is not None else ""
            pbr_str = f"PBR &nbsp;<b>{actual_p:.2f}</b>&nbsp;{_pbr_badge(actual_p)}"   if actual_p  is not None else ""
            roe_str = f"ROE &nbsp;<b>{stock_roe:.1f}%</b>&nbsp;{_roe_badge(stock_roe)}" if stock_roe is not None else ""
            parts   = " &nbsp;|&nbsp; ".join(x for x in [per_str, pbr_str, roe_str] if x)
            st.markdown(f"""
<div style="background:#f8faff;border-radius:10px;padding:12px 16px;margin-bottom:8px;
            border-left:3px solid {_border};font-size:0.87rem;line-height:2.0">
  📊 <b>절대 가치 판정</b>
  <span style="color:#888ea8;font-size:0.73rem">&nbsp;PER: ~10🟢/~20🟡/~30🟠/30↑🔴 &nbsp;·&nbsp; PBR: ~1🟢/~2🟡/~4🟠/4↑🔴 &nbsp;·&nbsp; ROE: 15↑🟢/8~🟡/0~🟠/↓0🔴</span><br>
  {parts}
</div>""", unsafe_allow_html=True)

        if is_kr_chart:
            _render_abs_val_card()
            # ── Ollama 가치 분석 코멘트 (버튼 클릭 후 표시) ──
            _val_key = f"val_comment_{code}"
            if _val_key in st.session_state:
                st.markdown(f"""
<div style="background:#f0f4ff;border-radius:10px;padding:10px 16px;margin-bottom:8px;
            border-left:3px solid #4a7afa;font-size:0.84rem;line-height:1.6;color:#1a1f36">
  💬 <b>AI 가치 분석</b>&nbsp;<span style="font-size:0.72rem;color:#888ea8">(Ollama)</span><br>
  {st.session_state[_val_key]}
</div>""", unsafe_allow_html=True)
        else:
            if gap is not None and theory_p is not None and actual_p is not None:
                gap_color = "#40a02b" if gap > 10 else "#e64553" if gap < -10 else "#df8e1d"
                gap_label = "🟢 저평가" if gap > 10 else "🔴 고평가" if gap < -10 else "🟡 적정"
                sign = "+" if gap >= 0 else ""
                st.markdown(f"""
<div style="background:#f8faff;border-radius:10px;padding:12px 16px;margin-bottom:8px;
            border-left:3px solid {gap_color};font-size:0.87rem;line-height:1.6">
  💰 <b>이론PBR 괴리율 분석</b>
  <span style="color:#888ea8;font-size:0.78rem">&nbsp;(이론PBR = PER × ROE)</span><br>
  이론PBR <b>{theory_p:.2f}</b> vs 실제PBR <b>{actual_p:.2f}</b>
  &nbsp;→&nbsp;
  <span style="color:{gap_color};font-weight:700">{gap_label} ({sign}{gap:.1f}%)</span>
</div>""", unsafe_allow_html=True)
            _render_abs_val_card()

    # ── 종합 기술적 신호등 + 52주 위치 ──
    sc, wc = st.columns([1, 2])
    with sc:
        st.markdown(f"""
<div style="background:{sig_bg};border-radius:12px;padding:18px;
            border:2px solid {sig_color};text-align:center;margin-top:4px">
  <div style="font-size:0.75rem;color:#555e7a;font-weight:600">기술적 신호등</div>
  <div style="font-size:1.5rem;font-weight:800;color:{sig_color};margin:6px 0">{sig_text}</div>
  <div style="font-size:0.78rem;color:#888ea8">긍정 신호 {pos} / 5</div>
</div>""", unsafe_allow_html=True)
    with wc:
        filled  = max(0, min(20, int(_w52_pct / 5)))
        bar     = "█" * filled + "░" * (20 - filled)
        st.markdown(f"""
<div style="background:#f8faff;border-radius:12px;padding:16px 20px;
            border:1px solid #e8ecf4;margin-top:4px">
  <div style="font-size:0.78rem;color:#555e7a;font-weight:600;margin-bottom:8px">52주 고저 대비 현재가</div>
  <div style="font-family:monospace;font-size:0.8rem;color:#4a7afa;letter-spacing:0.05em">{bar}</div>
  <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:0.82rem">
    <span style="color:#26a69a">최저 {_fmt(_w52_l)}</span>
    <span style="color:#1a1f36;font-weight:800">{_fmt(close)} ({_w52_pct:.0f}%)</span>
    <span style="color:#ef5350">최고 {_fmt(_w52_h)}</span>
  </div>
</div>""", unsafe_allow_html=True)


    # ── Ollama AI 분석 ──
    st.markdown("---")
    st.markdown("#### 🤖 AI 종목 분석 (Ollama)")
    if st.button("✨ AI 해설 생성", key=f"ai_{code}"):
        # 거시환경 요약
        macro_summary = ""
        if df_macro is not None and not df_macro.empty:
            rows = []
            for _, mr in df_macro.iterrows():
                rows.append(f"{mr['지표']} {mr['현재값']:.2f} (전월대비 {mr['변화량']:+.2f})")
            macro_summary = "\n".join(rows)

        fund_info = ""
        if stock_data:
            per  = stock_data.get("PER", "N/A")
            pbr  = stock_data.get("PBR", "N/A")
            roe  = stock_data.get("ROE(%)", "N/A")
            scr  = stock_data.get("종합점수", "N/A")
            gap  = stock_data.get("괴리율(%)", "N/A")
            fund_info = (
                f"PER={per}, PBR={pbr}, ROE={roe}%, 이론PBR괴리율={gap}%, 종합점수={scr}/100"
            )

        prompt = _KR_ONLY_PREFIX + f"""당신은 한국 주식 전문 애널리스트입니다. 아래 데이터를 바탕으로 {name}({code}) 종목을 분석해주세요.

[기술적 지표]
현재가: {close:,.0f} | MA20: {ma20:,.0f}({'상승' if ma_up else '하락'}추세)
RSI: {rsi:.1f} | MACD: {'골든크로스' if macd_v > macd_sig else '데드크로스'} | 볼린저밴드 위치: {bb_pct:.0f}%
거래량: 평균대비 {vol_ratio:.1f}배

[펀더멘탈]
{fund_info}

[거시환경]
{macro_summary if macro_summary else '데이터 없음'}

아래 형식으로 한국어로 간결하게 답변해주세요 (각 항목 2~3문장):
[모멘텀] 현재 기술적 흐름과 매매 타이밍
[펀더멘탈] 가치평가 수준과 업종 내 위치
[거시환경] 현재 거시지표가 이 종목에 미치는 영향
[단기목표가] 볼린저밴드 상단({float(last['BB_upper']):,.0f}) 기준 단기 시나리오
[중기목표가] 적정PER 기준 중기 목표가 (숫자로 제시)
[손절기준] MA20({ma20:,.0f}) 이탈 시 대응 방안"""

        # ── Ollama 가치 분석 코멘트 (절대값 카드 아래 표시용) ──
        if is_kr_chart and stock_data:
            _sector_nm = stock_data.get("섹터", "해당 업종")
            _per_v  = stock_data.get("PER",    "N/A")
            _pbr_v  = stock_data.get("PBR",    "N/A")
            _roe_v  = stock_data.get("ROE(%)", "N/A")
            _val_prompt = (
                "당신은 한국어 전문 투자 분석가입니다. 반드시 한국어로만 답변하세요. "
                "다른 언어(중국어, 영어 등)는 절대 사용하지 마세요.\n\n"
                f"이 종목은 {_sector_nm}에 속하며 PER {_per_v}, PBR {_pbr_v}, ROE {_roe_v}%입니다. "
                f"업종 특성을 고려해서 이 수치들이 적정한지 2~3문장으로 설명해줘."
            )
            with st.spinner("가치 분석 코멘트 생성 중..."):
                _val_result = call_ollama(_val_prompt)
            st.session_state[f"val_comment_{code}"] = _val_result

        with st.spinner("AI 분석 중... (약 20~40초)"):
            ai_result = call_ollama(prompt)

        # 파싱해서 카드로 표시
        sections = {"모멘텀": "", "펀더멘탈": "", "거시환경": "", "단기목표가": "", "중기목표가": "", "손절기준": ""}
        current  = None
        for line in ai_result.split("\n"):
            for key_s in sections:
                if f"[{key_s}]" in line:
                    current = key_s
                    line = line.replace(f"[{key_s}]", "").strip()
                    break
            if current:
                sections[current] += line + " "
        # session_state에 저장 (📌 체크 시 signal_history.csv에 기록하기 위해)
        st.session_state[f"ai_sections_{code}"] = sections

        card_ai = lambda icon, title, body, border: f"""
<div style="background:#f8faff;border-radius:10px;padding:12px 16px;margin-bottom:8px;
            border-left:3px solid {border};font-size:0.87rem;line-height:1.55">
  {icon} <b>{title}</b><br>{body.strip() or '—'}
</div>"""
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(card_ai("📈", "모멘텀",    sections["모멘텀"],    "#4a7afa"), unsafe_allow_html=True)
            st.markdown(card_ai("💰", "펀더멘탈",  sections["펀더멘탈"],  "#40a02b"), unsafe_allow_html=True)
        with col_b:
            st.markdown(card_ai("🌐", "거시환경",  sections["거시환경"],  "#9b59b6"), unsafe_allow_html=True)
            col_t1, col_t2, col_t3 = st.columns(3)
            _tgt_card = lambda label, body, color: f"""
<div style="background:#f8faff;border-radius:8px;padding:8px 10px;
            border-left:3px solid {color};font-size:0.72rem;line-height:1.55;
            word-break:break-all;overflow-wrap:break-word;white-space:normal">
  <div style="font-size:0.67rem;color:#888ea8;font-weight:600;margin-bottom:3px">{label}</div>
  <div style="color:#1a1f36;font-weight:500">{body.strip() or '—'}</div>
</div>"""
            with col_t1:
                st.markdown(_tgt_card("📈 단기 목표가", sections["단기목표가"], "#4a7afa"),
                            unsafe_allow_html=True)
            with col_t2:
                st.markdown(_tgt_card("🎯 중기 목표가", sections["중기목표가"], "#40a02b"),
                            unsafe_allow_html=True)
            with col_t3:
                st.markdown(_tgt_card("🛑 손절 기준",   sections["손절기준"],   "#e64553"),
                            unsafe_allow_html=True)

    # ── 분기별 실적 추이 (한국 종목 + DART 키 있을 때만) ──
    if is_kr_chart:
        st.markdown("---")
        st.markdown("#### 📊 실적 추이")
        if not DART_KEY:
            st.info(
                "**DART API 키가 설정되지 않았습니다.** "
                "실적 차트를 사용하려면 아래 중 하나로 설정하세요.\n\n"
                "**방법 1 — `.env` 파일 (권장)**\n"
                "```\n"
                "DART_API_KEY=여기에_API키_입력\n"
                "```\n"
                "**방법 2 — PowerShell 환경변수 (세션)**\n"
                "```powershell\n"
                '$env:DART_KEY = "여기에_API키_입력"\n'
                "```\n"
                "DART API 키는 [금융감독원 DART 오픈API](https://opendart.fss.or.kr) 에서 무료 발급"
            )
        else:
            _earn_tab_annual, _earn_tab_qtr = st.tabs(["📅 연간", "📊 분기"])

            def _build_earn_chart(df_earn: pd.DataFrame, x_col: str, title: str) -> go.Figure:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=df_earn[x_col], y=df_earn["매출액(억)"],
                    name="매출액(억)", marker_color="#4a7afa", opacity=0.85,
                ), secondary_y=False)
                fig.add_trace(go.Bar(
                    x=df_earn[x_col], y=df_earn["영업이익(억)"],
                    name="영업이익(억)", marker_color="#40a02b", opacity=0.85,
                ), secondary_y=False)
                if "영업이익률(%)" in df_earn.columns:
                    fig.add_trace(go.Scatter(
                        x=df_earn[x_col], y=df_earn["영업이익률(%)"],
                        name="영업이익률(%)",
                        line=dict(color="#f7b731", width=2),
                        mode="lines+markers",
                        marker=dict(size=6),
                    ), secondary_y=True)
                fig.update_layout(
                    title=title,
                    barmode="group",
                    height=380,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=10, r=10, t=60, b=10),
                )
                fig.update_xaxes(showgrid=True, gridcolor="#e8ecf4", type="category")
                fig.update_yaxes(
                    title_text="금액 (억원)", showgrid=True, gridcolor="#e8ecf4",
                    secondary_y=False,
                )
                fig.update_yaxes(
                    title_text="영업이익률 (%)", showgrid=False,
                    secondary_y=True,
                )
                return fig

            with _earn_tab_annual:
                with st.spinner("연간 실적 데이터 조회 중..."):
                    df_a = fetch_dart_annual(code, DART_KEY)
                if df_a.empty:
                    st.warning("연간 실적 데이터를 가져올 수 없습니다. (DART 미공시 또는 API 오류)")
                else:
                    st.plotly_chart(
                        _build_earn_chart(df_a, "연도", f"{name} 연간 실적 추이 (최근 5년)"),
                        width="stretch",
                    )

            with _earn_tab_qtr:
                with st.spinner("분기 실적 데이터 조회 중..."):
                    df_q = fetch_dart_quarterly(code, DART_KEY)
                if df_q.empty:
                    st.warning("분기 실적 데이터를 가져올 수 없습니다. (DART 미공시 또는 API 오류)")
                else:
                    df_q8 = df_q.tail(8).copy()
                    st.plotly_chart(
                        _build_earn_chart(df_q8, "분기", f"{name} 분기별 실적 추이 (최근 8분기)"),
                        width="stretch",
                    )


def show_chart_picker(df_top: pd.DataFrame, key: str, ticker_suffix: str,
                      is_kr_chart: bool, mkt_label: str = "", df_macro: pd.DataFrame = None,
                      period: str = "1y"):
    if df_top.empty or "종목명" not in df_top.columns:
        return
    # 스크리너 테이블 전체 종목 표시 (기존 head(10) 제거)
    pool  = df_top.reset_index(drop=True)
    names = pool["종목명"].tolist()
    codes = pool["코드"].astype(str).tolist()
    n_total = len(pool)
    st.markdown("<br>", unsafe_allow_html=True)
    sel = st.selectbox(
        f"📈 차트 분석 — 종목 선택 (전체 {n_total}개)",
        ["선택 안 함"] + names, key=key,
        help="종목을 선택하면 기술적 분석 차트가 펼쳐집니다",
    )
    if sel != "선택 안 함":
        idx        = names.index(sel)
        code       = codes[idx].zfill(6) if is_kr_chart else codes[idx]
        stock_data = pool.iloc[idx].to_dict()
        with st.expander(f"📊 {sel} 기술적 분석", expanded=True):
            render_stock_chart(code, sel, ticker_suffix, is_kr_chart, stock_data, df_macro,
                               period=period)
            # 추적 추가 버튼
            fund_sig = stock_data.get("펀더멘탈신호", "—")
            tech_sig = stock_data.get("기술적신호",   "—")
            시장_key  = ("KOSPI"  if "코스피" in mkt_label else
                        "KOSDAQ" if "코스닥" in mkt_label else "US")
            가격       = stock_data.get("현재주가")
            if 가격:
                if st.button(f"📌 {sel} 추적 추가", key=f"track_{key}_{idx}"):
                    ok = add_tracking_record(sel, codes[idx], 시장_key,
                                             float(가격), fund_sig, tech_sig)
                    st.success(f"✅ {sel} 추적 추가 완료" if ok
                               else "이미 최근 7일 내 추적 중인 종목입니다.")

# ================================================
# TOP 10 렌더러
# ================================================
TABLE_COLS = ["순위", "종목명", "코드", "현재주가", "PER", "ROE(%)", "종합점수",
              "펀더멘탈신호", "기술적신호", "종합신호"]


def make_col_config(is_kr: bool) -> dict:
    price_fmt = "₩%.0f" if is_kr else "$%.2f"
    return {
        "순위":          st.column_config.TextColumn(width="small"),
        "현재주가":      st.column_config.NumberColumn(format=price_fmt),
        "PER":           st.column_config.NumberColumn(format="%.1f"),
        "ROE(%)":        st.column_config.NumberColumn("ROE(%)", format="%.1f"),
        "종합점수":      st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
        "매출성장률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "영업이익률(%)": st.column_config.NumberColumn(format="%.1f%%"),
        "펀더멘탈신호":  st.column_config.TextColumn("펀더멘탈", width="small"),
        "기술적신호":    st.column_config.TextColumn("기술적",   width="small"),
        "종합신호":      st.column_config.TextColumn("종합",     width="medium"),
    }


def render_top10(df_ranked: pd.DataFrame, chart_x: str, chart_color: str,
                 chart_title: str, is_kr: bool = False, n: int = 10):
    df_show    = df_ranked.head(n).copy()
    df_show.insert(0, "순위", [f"#{i+1}" for i in range(len(df_show))])
    cols_avail = [c for c in TABLE_COLS if c in df_show.columns]

    tbl_col, chart_col = st.columns([3, 1], gap="large")
    with tbl_col:
        st.dataframe(
            df_show[cols_avail].reset_index(drop=True),
            width="stretch", hide_index=True,
            column_config=make_col_config(is_kr),
        )
    with chart_col:
        if chart_x not in df_show.columns:
            return
        fig = px.bar(
            df_show[["종목명", chart_x]].reset_index(drop=True),
            x=chart_x, y="종목명", orientation="h",
            color=chart_x, color_continuous_scale=chart_color,
            text=chart_x, title=chart_title,
        )
        sfx = "%" if "%" in chart_x else ""
        fig.update_traces(texttemplate=f"%{{text:.1f}}{sfx}", textposition="outside")
        fig.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
            yaxis_title="", xaxis_title=chart_x, margin=dict(l=0, r=70, t=40, b=20),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e0e4ef")
        st.plotly_chart(fig, width="stretch")


def sector_dropdown(key: str, sectors: list) -> str:
    return st.selectbox("섹터 필터", ["전체"] + sorted(sectors), key=key)

# ================================================
# 사이드바 — 마켓 선택 (필터는 데이터 로드 후)
# ================================================
with st.sidebar:
    st.markdown("### 🌏 마켓")
    market_label = st.radio(
        "시장 선택",
        ["🇺🇸 미국 S&P500", "🇰🇷 코스피", "🇰🇷 코스닥"],
        index=0, label_visibility="collapsed",
    )
    is_kr      = "코스피" in market_label or "코스닥" in market_label
    suffix     = ".KS" if "코스피" in market_label else ".KQ" if "코스닥" in market_label else ""
    csv_prefix = ("us"           if "미국"   in market_label else
                  "kospi_섹터별" if "코스피" in market_label else "kosdaq_섹터별")

# ================================================
# 데이터 로드 (사이드바 필터에 필요)
# ================================================
df_raw, updated_at = load_latest_csv(csv_prefix)

if not is_kr and not df_raw.empty:
    sector_map = load_us_sector_map()
    if sector_map and "코드" in df_raw.columns:
        df_raw["섹터"] = df_raw["코드"].map(sector_map).fillna("기타")
    elif "섹터" not in df_raw.columns:
        df_raw["섹터"] = "기타"

if is_kr and not df_raw.empty and "섹터" not in df_raw.columns:
    df_raw["섹터"] = "기타"

# ── 사이드바 필터 (데이터 로드 후) ──
with st.sidebar:
    st.divider()
    st.markdown("### 🔧 필터")
    if df_raw.empty:
        st.warning("데이터 없음")
        per_range = (0.0, 50.0)
        roe_min   = 0.0
        pbr_max_v = 20.0
    else:
        per_max   = round(float(df_raw["PER"].max()), 1) if "PER" in df_raw.columns else 50.0
        per_range = st.slider("PER 범위", 0.0, per_max, (0.0, per_max), step=0.5)
        roe_min   = st.slider("ROE 최소 (%)", 0.0, 100.0, 0.0, step=1.0)
        pbr_max   = round(float(df_raw["PBR"].max()), 1) if "PBR" in df_raw.columns else 20.0
        pbr_max_v = st.slider("PBR 최대", 0.0, pbr_max, pbr_max, step=0.5)
    st.divider()
    st.markdown("### 📅 차트 기간")
    _period_label = st.radio(
        "차트 기간",
        ["3개월", "6개월", "1년", "2년"],
        index=2,
        label_visibility="collapsed",
        key="chart_period_radio",
    )
    chart_period = _PERIOD_MAP[_period_label]
    st.divider()
    if st.button("🔄 새로고침", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# ================================================
# 데이터 없을 때
# ================================================
if df_raw.empty:
    st.title("📈 주식 스크리너 대시보드")
    st.warning(f"**{market_label.split(' ',1)[-1]}** CSV 파일이 없습니다.")
    st.info("`python screener.py 분석실행해줘` 를 먼저 실행하세요.")
    st.stop()

# 필터 적용
df_base = df_raw.copy()
if "PER"    in df_base.columns: df_base = df_base[df_base["PER"].between(*per_range)]
if "PBR"    in df_base.columns: df_base = df_base[df_base["PBR"] <= pbr_max_v]
if "ROE(%)" in df_base.columns: df_base = df_base[df_base["ROE(%)"] >= roe_min]
df_base = df_base.reset_index(drop=True)

if "종합점수" not in df_base.columns:
    df_base = compute_scores(df_base)

sectors_all = sorted(df_base["섹터"].dropna().unique().tolist()) if "섹터" in df_base.columns else []

# ================================================
# 거시지표 카드
# ================================================
df_macro = load_macro_csv()
if not df_macro.empty:
    st.subheader("🌐 거시지표")
    chunks = [df_macro.iloc[i:i+5] for i in range(0, len(df_macro), 5)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            label   = row["지표"]
            cur     = row["현재값"]
            chg     = row["변화량"]
            unit    = str(row.get("단위", "") or "")
            up      = chg >= 0
            arrow   = "▲" if up else "▼"
            color   = "#e64553" if up else "#40a02b"
            benefit = _MACRO_BENEFIT.get(label, {}).get(up, "—")
            if "%" in unit:
                val_str, chg_str = f"{cur:.2f}%", f"{arrow} {abs(chg):.2f}%p"
            elif unit == "원":
                val_str, chg_str = f"{cur:,.0f}원", f"{arrow} {abs(chg):,.0f}"
            elif unit == "달러":
                val_str, chg_str = f"${cur:.1f}", f"{arrow} {abs(chg):.2f}"
            else:
                val_str, chg_str = f"{cur:.2f}", f"{arrow} {abs(chg):.2f}"
            col.markdown(f"""
<div class="macro-card">
  <div class="macro-label">{label}</div>
  <div class="macro-value">{val_str}</div>
  <div class="macro-change" style="color:{color}">{chg_str}</div>
  <div class="macro-benefit">▸ {benefit}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    # 거시환경 Ollama 분석 버튼
    st.markdown("#### 🤖 거시환경 AI 종합 해설")
    if st.button("🌐 거시환경 분석", key="macro_ai"):
        rows = []
        for _, mr in df_macro.iterrows():
            rows.append(f"- {mr['지표']}: {mr['현재값']:.2f} (전월대비 {mr['변화량']:+.2f})")
        macro_text = "\n".join(rows)

        prompt = _KR_ONLY_PREFIX + f"""당신은 한국 주식시장 거시경제 전문가입니다. 아래 최신 거시지표를 분석해주세요.

[현재 거시지표]
{macro_text}

다음 형식으로 한국어로 답변해주세요:

[종합해설] 현재 거시환경의 전반적인 특징과 주식시장 영향 (3~4문장)
[주목섹터🟢] 현재 환경에서 유리한 섹터 2~3개와 이유 (한 줄씩)
[주의섹터🔴] 현재 환경에서 불리한 섹터 2~3개와 이유 (한 줄씩)
[투자전략] 현재 거시환경에 맞는 단기 투자 전략 (2~3문장)"""

        with st.spinner("거시환경 분석 중... (약 20~40초)"):
            macro_ai_text = call_ollama(prompt)

        sec = {"종합해설": "", "주목섹터🟢": "", "주의섹터🔴": "", "투자전략": ""}
        cur_s = None
        for line in macro_ai_text.split("\n"):
            for k in sec:
                if f"[{k}]" in line:
                    cur_s = k
                    line = line.replace(f"[{k}]", "").strip()
                    break
            if cur_s:
                sec[cur_s] += line + "\n"

        macro_card = lambda icon, title, body, border: f"""
<div style="background:#f8faff;border-radius:10px;padding:12px 16px;margin-bottom:8px;
            border-left:3px solid {border};font-size:0.87rem;line-height:1.6;white-space:pre-line">
  {icon} <b>{title}</b><br>{body.strip() or '—'}
</div>"""
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(macro_card("📊", "종합 해설", sec["종합해설"],   "#4a7afa"), unsafe_allow_html=True)
            st.markdown(macro_card("🎯", "투자 전략", sec["투자전략"],   "#9b59b6"), unsafe_allow_html=True)
        with m2:
            st.markdown(macro_card("🟢", "주목 섹터", sec["주목섹터🟢"], "#40a02b"), unsafe_allow_html=True)
            st.markdown(macro_card("🔴", "주의 섹터", sec["주의섹터🔴"], "#e64553"), unsafe_allow_html=True)

    st.divider()

# ================================================
# 헤더 + 요약 카드
# ================================================
h_left, h_right = st.columns([4, 1])
with h_left:
    st.title("📈 주식 스크리너 대시보드")
    st.caption(market_label)
with h_right:
    if updated_at:
        st.markdown(f'<p class="last-updated">🕐 마지막 분석: {updated_at}</p>',
                    unsafe_allow_html=True)

med_per = df_base["PER"].median()    if "PER"    in df_base.columns else 0
med_roe = df_base["ROE(%)"].median() if "ROE(%)" in df_base.columns else 0
n_pbr1  = int((df_base["PBR"] < 1).sum()) if "PBR" in df_base.columns else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="indicator-card">
        <div class="indicator-title">PER (주가수익비율)</div>
        <div class="indicator-value">{med_per:.1f} <span style="font-size:1rem">중앙값</span></div>
        <div class="indicator-desc">📉 낮을수록 → 시장이 이익 대비 싸게 평가 중</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="indicator-card" style="border-left-color:#40a02b;">
        <div class="indicator-title">PBR 1 이하 종목</div>
        <div class="indicator-value" style="color:#40a02b;">{n_pbr1}개</div>
        <div class="indicator-desc">📦 PBR &lt; 1 → 장부가보다 싸게 살 수 있음</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="indicator-card" style="border-left-color:#df8e1d;">
        <div class="indicator-title">ROE (자기자본이익률)</div>
        <div class="indicator-value" style="color:#df8e1d;">{med_roe:.1f}% <span style="font-size:1rem">중앙값</span></div>
        <div class="indicator-desc">📈 높을수록 → 자본 대비 수익성 우수</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption(f"필터 후 분석 대상: {len(df_base)}개 종목 | 이상치(ROE>500%, PBR/PER/ROE 음수) 자동 제거")
st.divider()

# ================================================
# 종목 직접 검색 + 차트 분석
# ================================================
with st.expander("🔍 종목 직접 검색 & 차트 분석", expanded=False):
    srch_col, mkt_col = st.columns([3, 1])
    with srch_col:
        search_query = st.text_input("종목명 또는 코드 입력", placeholder="예: 삼성전자, 005930, AAPL",
                                     key="stock_search_input")
    with mkt_col:
        search_market = st.selectbox("시장", ["자동(KR)", "코스피", "코스닥", "미국"],
                                      key="stock_search_market")

    if search_query:
        if search_market == "미국":
            # 미국 종목: 입력을 ticker로 직접 사용
            s_ticker = search_query.upper().strip()
            s_code   = s_ticker
            s_suffix = ""
            s_is_kr  = False
            st.info(f"미국 종목 **{s_ticker}** 차트 분석")
            with st.expander(f"📊 {s_ticker} 기술적 분석", expanded=True):
                render_stock_chart(s_code, s_ticker, s_suffix, s_is_kr,
                                   df_macro=df_macro, period=chart_period)
        else:
            # 한국 종목: FinanceDataReader로 검색
            with st.spinner("종목 검색 중..."):
                results = search_kr_stock(search_query.strip())

            if results.empty:
                st.warning("검색 결과가 없습니다.")
            else:
                # 시장 필터
                if search_market in ("코스피", "코스닥"):
                    results = results[results["시장"] == search_market]

                if results.empty:
                    st.warning(f"{search_market}에서 '{search_query}' 종목을 찾을 수 없습니다.")
                else:
                    options = [f"{r['Name']} ({r['Code']}) [{r['시장']}]"
                               for _, r in results.iterrows()]
                    chosen  = st.selectbox("검색 결과 — 종목 선택", options, key="search_result_sel")
                    if chosen:
                        idx_ch   = options.index(chosen)
                        row_ch   = results.iloc[idx_ch]
                        s_code   = str(row_ch["Code"]).zfill(6)
                        s_name   = row_ch["Name"]
                        s_mkt    = row_ch["시장"]
                        s_suffix = ".KS" if s_mkt == "KOSPI" else ".KQ"
                        s_is_kr  = True

                        # CSV에서 펀더멘탈 데이터 조회
                        stock_extra = {}
                        for pfx in ["kospi_섹터별", "kosdaq_섹터별"]:
                            tmp_df, _ = load_latest_csv(pfx)
                            if not tmp_df.empty and "코드" in tmp_df.columns:
                                hit = tmp_df[tmp_df["코드"].astype(str).str.zfill(6) == s_code]
                                if not hit.empty:
                                    stock_extra = hit.iloc[0].to_dict()
                                    break

                        with st.expander(f"📊 {s_name} 기술적 분석", expanded=True):
                            render_stock_chart(s_code, s_name, s_suffix, s_is_kr,
                                               stock_extra or None, df_macro,
                                               period=chart_period)

# ================================================
# 탭
# ================================================
tab_main, tab_track, tab_bt = st.tabs(["📋 통합 스크리닝", "🔍 추적검증", "🧪 백테스팅"])

# ── 탭 1: 통합 스크리닝 ──
with tab_main:
    st.subheader("📋 통합 종목 스크리닝")
    st.caption("한국: 업종내상대PBR 40% + ROE업종대비 30% + PER업종대비 30% | "
               "미국: 이론PBR괴리율 40% + 업종내PBR 30% + ROE업종대비 30%")

    # ── 필터 & 정렬 컨트롤 ──
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        sort_by = st.selectbox("📊 정렬 기준",
                               ["종합점수순", "펀더멘탈순", "기술적순"],
                               key="sort_by")
    with ctrl2:
        sel_sec = st.selectbox("🏭 섹터 필터",
                               ["전체"] + sorted(sectors_all),
                               key="sec_main")
    with ctrl3:
        sig_filter = st.selectbox("🚦 종합신호 필터",
                                  ["전체", "강력매수만", "매수검토이상"],
                                  key="sig_filter")

    # ── 데이터 준비 ──
    df_main = df_base.copy()
    if sel_sec != "전체" and "섹터" in df_main.columns:
        df_main = df_main[df_main["섹터"] == sel_sec].copy()
        df_main = compute_scores(df_main)

    if df_main.empty:
        st.info("해당 조건에 맞는 종목이 없습니다.")
    else:
        df_main = add_signals(df_main, suffix, is_kr, chart_period)

        # 종합신호 필터
        if sig_filter == "강력매수만":
            df_main = df_main[df_main["종합신호"].str.contains("강력매수", na=False)]
        elif sig_filter == "매수검토이상":
            df_main = df_main[df_main["종합신호"].str.contains("강력매수|매수검토", na=False)]

        # 정렬
        _SCORE_MAP = {"🟢 매수": 2, "🟡 중립": 1, "🔴 매도": 0, "—": 0}
        if sort_by == "종합점수순" and "종합점수" in df_main.columns:
            df_main = df_main.sort_values("종합점수", ascending=False)
        elif sort_by == "펀더멘탈순":
            df_main["_fo"] = df_main["펀더멘탈신호"].map(_SCORE_MAP).fillna(0)
            df_main = df_main.sort_values(["_fo", "종합점수"], ascending=[False, False])
            df_main = df_main.drop(columns=["_fo"])
        elif sort_by == "기술적순":
            df_main["_to"] = df_main["기술적신호"].map(_SCORE_MAP).fillna(0)
            df_main = df_main.sort_values(["_to", "종합점수"], ascending=[False, False])
            df_main = df_main.drop(columns=["_to"])

        df_main = df_main.reset_index(drop=True)

        if df_main.empty:
            st.info("해당 신호 조건에 맞는 종목이 없습니다.")
        else:
            # ── 세션 상태 초기화 ──
            if "tracked_codes" not in st.session_state:
                _hist = load_signal_history()
                st.session_state.tracked_codes = (
                    set(_hist["코드"].astype(str).tolist()) if not _hist.empty else set()
                )

            # ── 표시용 컬럼 구성 ──
            price_fmt = "₩%.0f" if is_kr else "$%.2f"
            SHOW_COLS = [c for c in
                ["종목명", "코드", "현재주가", "종합점수",
                 "펀더멘탈신호", "기술적신호", "종합신호"]
                if c in df_main.columns]

            df_show = df_main[SHOW_COLS].copy()
            df_show.insert(0, "순위", [f"#{i+1}" for i in range(len(df_show))])
            df_show["추적📌"] = df_show["코드"].astype(str).isin(st.session_state.tracked_codes)
            prev_tracked_col = df_show["추적📌"].copy()

            col_cfg = {
                "순위":       st.column_config.TextColumn(width="small"),
                "현재주가":   st.column_config.NumberColumn(format=price_fmt),
                "종합점수":   st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "펀더멘탈신호": st.column_config.TextColumn("펀더멘탈", width="small"),
                "기술적신호":   st.column_config.TextColumn("기술적",   width="small"),
                "종합신호":     st.column_config.TextColumn("종합신호", width="medium"),
                "추적📌":       st.column_config.CheckboxColumn("추적📌", default=False),
            }
            disabled_cols = ["순위"] + SHOW_COLS  # 추적📌만 편집 가능

            edited = st.data_editor(
                df_show,
                width="stretch",
                hide_index=True,
                column_config=col_cfg,
                disabled=disabled_cols,
                key="main_table_editor",
            )

            # ── 체크박스 처리 → signal_history.csv + signal_tracking.json ──
            newly_checked = edited.index[edited["추적📌"] & ~prev_tracked_col]
            if len(newly_checked) > 0:
                today_str = datetime.now().strftime("%Y-%m-%d")
                시장_key = ("KOSPI"  if "코스피" in market_label else
                            "KOSDAQ" if "코스닥" in market_label else "US")
                with st.spinner(f"📌 {len(newly_checked)}개 종목 추적 추가 중 (AI 목표가 생성)..."):
                    for idx in newly_checked:
                        row_data = df_main.loc[idx]
                        code     = str(row_data.get("코드", ""))
                        name_v   = row_data.get("종목명", "")
                        가격     = row_data.get("현재주가")
                        ticker_v = (code.zfill(6) + suffix) if is_kr else code
                        st.session_state.tracked_codes.add(code)

                        # session_state에 이미 AI 해설 있으면 재사용, 없으면 자동 생성
                        _ai_sec = st.session_state.get(f"ai_sections_{code}")
                        if not _ai_sec:
                            _ai_sec = generate_target_prices(
                                name_v, code, ticker_v,
                                per=row_data.get("PER"),
                                pbr=row_data.get("PBR"),
                                roe=row_data.get("ROE(%)"),
                                period=chart_period,
                            )
                            if _ai_sec:
                                st.session_state[f"ai_sections_{code}"] = _ai_sec

                        save_signal_history({
                            "날짜":         today_str,
                            "종목명":       name_v,
                            "코드":         code,
                            "시장":         시장_key,
                            "주가":         가격,
                            "펀더멘탈신호": row_data.get("펀더멘탈신호", "—"),
                            "기술적신호":   row_data.get("기술적신호",   "—"),
                            "종합신호":     row_data.get("종합신호",     "—"),
                            "단기목표가":   _ai_sec.get("단기목표가", "").strip() if _ai_sec else "",
                            "중기목표가":   _ai_sec.get("중기목표가", "").strip() if _ai_sec else "",
                            "손절기준":     _ai_sec.get("손절기준",   "").strip() if _ai_sec else "",
                        })
                        if 가격:
                            add_tracking_record(
                                name_v, code, 시장_key, float(가격),
                                row_data.get("펀더멘탈신호", "—"),
                                row_data.get("기술적신호",   "—"),
                            )
                st.success(f"✅ {len(newly_checked)}개 종목 추적 추가 완료!")
                st.rerun()

            st.caption(f"총 {len(df_main)}개 종목 | 추적📌 체크 시 signal_history.csv + 수익률 추적 자동 저장")

            # ── 차트 분석 ──
            show_chart_picker(df_main, "chart_main", suffix, is_kr, market_label, df_macro,
                              period=chart_period)

# ── 탭 2: 추적검증 ──
with tab_track:
    st.subheader("🔍 추적검증 — 매수 신호 적중률")
    st.caption("통합 스크리닝 테이블에서 📌 체크한 종목 → 5일/10일/20일/60일/120일 후 수익률 자동 계산")

    records = load_tracking()

    act1, act2, act3, _ = st.columns([1, 1, 1, 2])
    with act1:
        if st.button("🔄 가격 업데이트", width="stretch"):
            with st.spinner("가격 업데이트 중..."):
                records = update_tracking_prices(records)
            st.success("업데이트 완료!")
            st.rerun()
    with act2:
        if st.button("🗑️ 추적 전체 삭제", width="stretch"):
            if os.path.exists(TRACKING_FILE):
                os.remove(TRACKING_FILE)
            st.session_state.pop("tracked_codes", None)
            st.success("추적 기록을 모두 삭제했습니다.")
            st.rerun()
    with act3:
        if st.button("🗑️ 히스토리 삭제", width="stretch"):
            if os.path.exists(SIGNAL_HISTORY_FILE):
                os.remove(SIGNAL_HISTORY_FILE)
            st.success("signal_history.csv 삭제 완료")
            st.rerun()

    # ── 수익률 추적 섹션 ──
    if not records:
        st.info("추적 중인 종목이 없습니다. 통합 스크리닝 탭에서 📌 체크박스를 클릭하세요.")
    else:
        df_track = pd.DataFrame(records)
        # 구형 2주후/4주후 포맷 마이그레이션: 새 컬럼 없으면 None으로 추가
        for _h in _TRACK_HORIZONS:
            _hcol = f"{_h}일수익률"
            if _hcol not in df_track.columns:
                df_track[_hcol] = None

        # ── 기간별 적중률 통계 ──
        st.markdown("#### 📊 기간별 적중률 통계")
        _stat_cols = st.columns(len(_TRACK_HORIZONS) + 1)
        _stat_cols[0].metric("전체 추적", f"{len(df_track)}개")
        for ci, h in enumerate(_TRACK_HORIZONS):
            _col = f"{h}일수익률"
            if _col in df_track.columns:
                _done = df_track[df_track[_col].notna()]
                if len(_done):
                    _wr  = int((_done[_col] > 0).sum())
                    _avg = float(_done[_col].mean())
                    _stat_cols[ci + 1].metric(
                        f"{h}일 승률",
                        f"{_wr/len(_done)*100:.0f}%",
                        f"평균 {_avg:+.1f}% ({len(_done)}건)",
                    )
                else:
                    _stat_cols[ci + 1].metric(f"{h}일 승률", "—")

        # ── 추적 상세 테이블 ──
        st.markdown("#### 📋 추적 상세")
        _base_cols = ["기록일", "종목명", "코드", "시장", "기록가격", "펀더멘탈", "기술적"]
        _ret_cols  = [f"{h}일수익률" for h in _TRACK_HORIZONS]
        disp_cols  = [c for c in _base_cols + _ret_cols if c in df_track.columns]
        df_disp    = df_track[disp_cols].copy()

        def _fmt_ret(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "—"
            return f"{'🟢' if val > 0 else '🔴'} {val:+.1f}%"

        for col in _ret_cols:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].apply(_fmt_ret)
        st.dataframe(df_disp.reset_index(drop=True), width="stretch", hide_index=True)

        # ── 기간 선택 수익률 바차트 ──
        _chart_h = st.selectbox(
            "📈 수익률 분포 보기",
            [h for h in _TRACK_HORIZONS if f"{h}일수익률" in df_track.columns
             and df_track[f"{h}일수익률"].notna().any()],
            format_func=lambda x: f"{x}일 후",
            key="track_chart_horizon",
        ) if any(
            df_track[f"{h}일수익률"].notna().any()
            for h in _TRACK_HORIZONS if f"{h}일수익률" in df_track.columns
        ) else None

        if _chart_h:
            _chart_col = f"{_chart_h}일수익률"
            _done_chart = df_track[df_track[_chart_col].notna()].copy()
            if len(_done_chart) >= 2:
                fig_ret = px.bar(
                    _done_chart.sort_values(_chart_col, ascending=False),
                    x="종목명", y=_chart_col,
                    color=_chart_col, color_continuous_scale="RdYlGn",
                    text=_chart_col, title=f"{_chart_h}일 후 수익률 (%)",
                )
                fig_ret.update_traces(texttemplate="%{text:+.1f}%", textposition="outside")
                fig_ret.add_hline(y=0, line_dash="dot", line_color="#888ea8")
                fig_ret.update_layout(
                    height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, margin=dict(l=0, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_ret, width="stretch")

    # ── signal_history.csv 섹션 ──
    df_hist = load_signal_history()
    if not df_hist.empty:
        st.markdown("---")
        st.markdown("#### 📂 신호 기록 히스토리 (signal_history.csv)")
        st.dataframe(df_hist.sort_values("날짜", ascending=False).reset_index(drop=True),
                     width="stretch", hide_index=True)

# ── 탭 3: 백테스팅 ──
with tab_bt:
    st.subheader("🧪 기술적 신호 백테스팅")
    st.caption("매수 신호(🟢) 발생 시점의 이후 수익률을 검증합니다. 신호 기준: 현재 차트와 동일 (MA·RSI·MACD·거래량)")

    # ── 종목 선택 ──
    bt_c1, bt_c2, bt_c3 = st.columns([2, 1, 1])
    with bt_c1:
        # 스크리너 목록 + 직접 입력
        if not df_base.empty and "종목명" in df_base.columns:
            _bt_names = df_base["종목명"].tolist()
            _bt_codes = df_base["코드"].astype(str).tolist()
            _bt_opts  = [f"{n} ({c})" for n, c in zip(_bt_names, _bt_codes)]
        else:
            _bt_names, _bt_codes, _bt_opts = [], [], []

        bt_mode = st.radio("종목 입력", ["스크리너 목록", "직접 입력"], horizontal=True, key="bt_mode")

    with bt_c2:
        bt_years = st.selectbox("백테스트 기간", [1, 2, 3], format_func=lambda x: f"{x}년", key="bt_years")

    with bt_c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        bt_run = st.button("▶ 백테스트 실행", key="bt_run", type="primary", width="stretch")

    if bt_mode == "스크리너 목록":
        if _bt_opts:
            bt_sel = st.selectbox("종목 선택", _bt_opts, key="bt_sel_list")
            _idx   = _bt_opts.index(bt_sel)
            bt_code = _bt_codes[_idx]
            bt_name = _bt_names[_idx]
            bt_ticker = (bt_code.zfill(6) + suffix) if is_kr else bt_code
        else:
            st.info("스크리너 데이터를 먼저 로드해주세요.")
            bt_ticker, bt_name = "", ""
    else:
        bt_manual = st.text_input("티커 직접 입력", placeholder="예: 005930.KS  /  AAPL", key="bt_manual")
        bt_ticker = bt_manual.strip()
        bt_name   = bt_ticker

    if bt_run and bt_ticker:
        with st.spinner(f"{bt_name or bt_ticker} 백테스트 실행 중..."):
            bt_result = run_backtest(bt_ticker, int(bt_years))

        if not bt_result or not bt_result.get("signals"):
            st.warning("해당 기간 내 매수 신호가 발생하지 않았습니다.")
        else:
            _signals  = bt_result["signals"]
            _df_ohlc  = bt_result["df_ohlc"]
            _horizons = [5, 10, 20, 60, 120]

            # ── 요약 메트릭 ──
            n_sig = len(_signals)
            st.markdown(f"#### 📊 백테스트 결과 — {bt_name or bt_ticker} ({bt_years}년)")

            # ── 복사 버튼 ──
            _bt_lines = [
                f"[백테스팅] {bt_name or bt_ticker}  기간: {bt_years}년",
                f"분석일: {datetime.now().strftime('%Y-%m-%d')}",
                f"신호 발생 횟수: {n_sig}회",
                "",
                "[ 기간별 성과 ]",
            ]
            for _h in _horizons:
                _cn = f"{_h}일수익률"
                _hv = [s[_cn] for s in _signals if s.get(_cn) is not None]
                if _hv:
                    _bt_lines.append(
                        f"  {_h:3d}일  평균 {sum(_hv)/len(_hv):+.2f}%  "
                        f"승률 {sum(1 for v in _hv if v>0)/len(_hv)*100:.0f}%  "
                        f"최고 {max(_hv):+.2f}%  최대손실 {min(_hv):+.2f}%  "
                        f"({len(_hv)}건)"
                    )
                else:
                    _bt_lines.append(f"  {_h:3d}일  —")
            _bt_lines += ["", "[ 신호 발생 내역 ]"]
            for _i, _s in enumerate(_signals):
                _dt = str(_s["날짜"])[:10] if hasattr(_s["날짜"], "__str__") else str(_s["날짜"])
                _rets = "  ".join(
                    f"{_h}d:{_s.get(f'{_h}일수익률'):+.1f}%" if _s.get(f"{_h}일수익률") is not None else f"{_h}d:—"
                    for _h in _horizons
                )
                _bt_lines.append(f"  #{_i+1}  {_dt}  진입가:{_s['진입가']:,.0f}  RSI:{_s['RSI']}  pos:{_s['pos']}  {_rets}")
            copy_button("\n".join(_bt_lines), "📋 백테스트 결과 복사", key=f"bt_{bt_ticker}")

            _m_cols = st.columns(len(_horizons) + 1)
            _m_cols[0].metric("신호 발생 횟수", f"{n_sig}회")

            for ci, h in enumerate(_horizons):
                _col_name = f"{h}일수익률"
                _vals = [s[_col_name] for s in _signals if s.get(_col_name) is not None]
                if _vals:
                    _avg = sum(_vals) / len(_vals)
                    _wr  = sum(1 for v in _vals if v > 0) / len(_vals) * 100
                    _m_cols[ci + 1].metric(
                        f"{h}일",
                        f"{_avg:+.2f}%",
                        f"승률 {_wr:.0f}% ({len(_vals)}건)",
                    )
                else:
                    _m_cols[ci + 1].metric(f"{h}일", "—")

            # ── 승률 테이블 ──
            st.markdown("##### 기간별 평균수익률 & 승률")
            _tbl_rows = []
            for h in _horizons:
                _col_name = f"{h}일수익률"
                _vals = [s[_col_name] for s in _signals if s.get(_col_name) is not None]
                if _vals:
                    _avg = sum(_vals) / len(_vals)
                    _wr  = sum(1 for v in _vals if v > 0) / len(_vals) * 100
                    _best  = max(_vals)
                    _worst = min(_vals)
                    _tbl_rows.append({
                        "보유기간": f"{h}일",
                        "평균수익률": f"{_avg:+.2f}%",
                        "승률": f"{_wr:.0f}%",
                        "최고수익": f"{_best:+.2f}%",
                        "최대손실": f"{_worst:+.2f}%",
                        "표본수": len(_vals),
                    })
            if _tbl_rows:
                st.dataframe(pd.DataFrame(_tbl_rows), hide_index=True, width="stretch")

            # ── 캔들차트 + 신호 마커 ──
            st.markdown("##### 신호 발생 시점 차트")
            _date_col = "Date" if "Date" in _df_ohlc.columns else _df_ohlc.columns[0]
            _fig_bt = go.Figure()

            _fig_bt.add_trace(go.Candlestick(
                x=_df_ohlc[_date_col],
                open=_df_ohlc["Open"], high=_df_ohlc["High"],
                low=_df_ohlc["Low"],   close=_df_ohlc["Close"],
                name="주가",
                increasing_line_color="#40a02b", decreasing_line_color="#e64553",
            ))

            # 신호 마커
            _sig_dates  = [s["날짜"]  for s in _signals]
            _sig_prices = [s["진입가"] for s in _signals]
            _sig_texts  = [
                f"신호#{i+1}<br>진입: {s['진입가']:,.0f}<br>RSI: {s['RSI']}"
                for i, s in enumerate(_signals)
            ]
            _fig_bt.add_trace(go.Scatter(
                x=_sig_dates, y=_sig_prices,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=14, color="#4a7afa",
                            line=dict(color="#ffffff", width=1)),
                text=[f"#{i+1}" for i in range(len(_signals))],
                textposition="top center",
                textfont=dict(size=9, color="#4a7afa"),
                hovertext=_sig_texts, hoverinfo="text",
                name="매수 신호",
            ))

            _fig_bt.update_layout(
                height=500,
                xaxis_rangeslider_visible=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=10, t=30, b=20),
                legend=dict(orientation="h", y=1.05),
                xaxis=dict(showgrid=True, gridcolor="#e8ecf3"),
                yaxis=dict(showgrid=True, gridcolor="#e8ecf3"),
            )
            st.plotly_chart(_fig_bt, width="stretch")

            # ── 신호 상세 테이블 ──
            with st.expander("📋 신호 발생 상세 내역"):
                _df_sig = pd.DataFrame(_signals)
                _df_sig.insert(0, "#", range(1, len(_df_sig) + 1))
                st.dataframe(_df_sig, hide_index=True, width="stretch")
