"""
글로벌 시황 슬랙 봇 — 4대 핵심 매크로 지표
매일 오전 7시(KST) GitHub Actions 또는 로컬에서 실행

[지표]
  1. 장단기금리차 (10Y-3M)  — yfinance: ^TNX, ^IRX
  2. 구리 가격              — yfinance: HG=F
  3. 하이일드 채권 스프레드 — FRED: BAMLH0A0HYM2
  4. 다우존스 운송지수      — yfinance: ^DJT
"""
import os
import sys
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp949", "cp1252", "ascii"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() in ("cp949", "cp1252", "ascii"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── 환경변수 로드 ─────────────────────────────────────────────────────────────
def _load_dotenv(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

SLACK_BOT_TOKEN  = os.getenv("SLACK_BOT_TOKEN", "")
_raw_ch          = os.getenv("SLACK_CHANNEL_ID", "")
SLACK_CHANNEL_ID = _raw_ch.split("/")[-1] if "/" in _raw_ch else _raw_ch
GEMINI_KEY       = os.getenv("GEMINI_KEY", "") or os.getenv("GEMINI_API_KEY", "")
FRED_API_KEY     = os.getenv("FRED_API_KEY", "")

GEMINI_MODEL    = "gemini-2.0-flash-lite"
GEMINI_FALLBACK = ["gemini-2.0-flash", "gemini-2.5-flash"]


# ── 추세 라벨 헬퍼 ────────────────────────────────────────────────────────────
def _ma_label(pct: float, inverted: bool = False) -> str:
    """MA50 대비 이격도 → 라벨. inverted=True 이면 스프레드 류 (높을수록 위험)."""
    if inverted:
        if pct > 10:   return "🚨위험확대"
        if pct > 2:    return "⚠️주의"
        if pct > -2:   return "➡️보통"
        if pct > -10:  return "📉안정화"
        return "🟢안정권"
    else:
        if pct > 10:   return "🔥과열"
        if pct > 2:    return "📈상승추세"
        if pct > -2:   return "➡️횡보"
        if pct > -10:  return "📉조정"
        return "🔻하락추세"

def _high_label(pct: float) -> str:
    """52주 고점 대비 % → 라벨."""
    if pct > -3:   return "🏔️고점근처"
    if pct > -10:  return "⚡고점권"
    if pct > -20:  return "⚠️조정구간"
    return "🔻급락구간"

def _arrow(chg: float, threshold: float = 0.3) -> str:
    if chg > threshold:  return "🔺"
    if chg < -threshold: return "🔻"
    return "➖"


# ── FRED 데이터 조회 ──────────────────────────────────────────────────────────
def _fetch_fred(series_id: str, limit: int = 300) -> pd.Series:
    """FRED API → pd.Series (날짜 인덱스, 오름차순)."""
    if not FRED_API_KEY:
        return pd.Series(dtype=float)
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "sort_order": "desc", "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        rows = {}
        for o in resp.json().get("observations", []):
            if o["value"] != ".":
                try:
                    rows[o["date"]] = float(o["value"])
                except ValueError:
                    pass
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series(rows)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"[WARN] FRED 조회 실패 ({series_id}): {e}", file=sys.stderr)
        return pd.Series(dtype=float)


# ── 시계열 통계 계산 ──────────────────────────────────────────────────────────
def _stats(series: pd.Series) -> dict:
    """현재가, 전일비, MA50 이격도, 52주 고점 대비 계산."""
    s = series.dropna()
    if len(s) < 2:
        return {}
    curr = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    ma50 = float(s.rolling(50).mean().iloc[-1]) if len(s) >= 50 else None
    high52 = float(s.max())
    low52  = float(s.min())
    pct_ma50      = (curr - ma50)   / abs(ma50)   * 100 if ma50   else None
    pct_from_high = (curr - high52) / abs(high52) * 100
    return {
        "curr": curr, "prev": prev,
        "pct_ma50": pct_ma50,
        "pct_from_high": pct_from_high,
        "high52": high52, "low52": low52,
    }


# ── 핵심 데이터 수집 ──────────────────────────────────────────────────────────
def fetch_market_data() -> dict:
    results = {}

    # yfinance: 1년치 (MA50 + 52주 계산용)
    yf_tickers = ["^TNX", "^IRX", "HG=F", "^DJT"]
    try:
        raw   = yf.download(yf_tickers, period="1y", progress=False, auto_adjust=True)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    except Exception as e:
        print(f"[ERROR] yfinance 실패: {e}", file=sys.stderr)
        close = pd.DataFrame()

    # 1. 장단기금리차 (10Y - 3M)
    if "^TNX" in close.columns and "^IRX" in close.columns:
        spread = close["^TNX"].dropna().align(close["^IRX"].dropna(), join="inner")[0] - \
                 close["^IRX"].dropna().align(close["^TNX"].dropna(), join="inner")[0]
        # 정렬 재처리
        tnx = close["^TNX"].dropna()
        irx = close["^IRX"].dropna()
        idx = tnx.index.intersection(irx.index)
        if len(idx) >= 2:
            sp = tnx.loc[idx] - irx.loc[idx]
            st = _stats(sp)
            if st:
                chg_abs = st["curr"] - st["prev"]  # %p 변화
                results["장단기금리차(10Y-3M)"] = {
                    **st,
                    "chg": chg_abs,
                    "chg_unit": "%p",
                    "fmt": lambda v: f"{v:+.3f}%p",
                    "inverted_ma": False,
                }

    # 2. 구리
    if "HG=F" in close.columns:
        st = _stats(close["HG=F"])
        if st:
            results["구리(HG=F)"] = {
                **st,
                "chg": (st["curr"] - st["prev"]) / st["prev"] * 100,
                "chg_unit": "%",
                "fmt": lambda v: f"${v:.3f}",
                "inverted_ma": False,
            }

    # 3. 다우존스 운송지수
    if "^DJT" in close.columns:
        st = _stats(close["^DJT"])
        if st:
            results["다우운송지수(^DJT)"] = {
                **st,
                "chg": (st["curr"] - st["prev"]) / st["prev"] * 100,
                "chg_unit": "%",
                "fmt": lambda v: f"{v:,.1f}",
                "inverted_ma": False,
            }

    # 4. 하이일드 스프레드 (FRED)
    hy = _fetch_fred("BAMLH0A0HYM2")
    if not hy.empty:
        st = _stats(hy)
        if st:
            chg_abs = st["curr"] - st["prev"]
            results["하이일드스프레드"] = {
                **st,
                "chg": chg_abs,
                "chg_unit": "%p",
                "fmt": lambda v: f"{v:.2f}%p",
                "inverted_ma": True,  # 스프레드 확대 = 위험
            }

    return results


# ── Gemini AI 코멘트 ──────────────────────────────────────────────────────────
def get_gemini_comment(data: dict) -> str:
    if not GEMINI_KEY or not data:
        return ""
    try:
        import google.genai as genai
        client = genai.Client(api_key=GEMINI_KEY)

        lines = []
        for name, v in data.items():
            unit = v["chg_unit"]
            chg  = v["chg"]
            ma   = v.get("pct_ma50")
            hi   = v.get("pct_from_high")
            line = f"{name}: 전일대비 {chg:+.3f}{unit}"
            if ma is not None:
                line += f" / MA50 대비 {ma:+.1f}%"
            if hi is not None:
                line += f" / 52주고점比 {hi:+.1f}%"
            lines.append(line)

        prompt = (
            "아래는 오늘의 글로벌 매크로 핵심 4대 지표야:\n"
            + "\n".join(lines) + "\n\n"
            "한국 주식 투자자 관점에서 '돈과 물건의 흐름' 측면으로 "
            "이 4개 지표가 현재 시장에 보내는 종합 신호를 한 문장으로 완성해줘. "
            "이모지 포함, 경기 사이클 위치와 리스크 온/오프 판단 위주로. "
            "반드시 완전한 문장으로 끝내야 하며 절대 중간에 끊기지 않아야 함."
        )
        for model in [GEMINI_MODEL] + GEMINI_FALLBACK:
            try:
                cfg  = genai.types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024)
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                return resp.text.strip()
            except Exception as e:
                if any(x in str(e) for x in ("429", "404", "NOT_FOUND")):
                    continue
                break
    except Exception as e:
        print(f"[WARN] Gemini 오류: {e}", file=sys.stderr)
    return ""


# ── 슬랙 블록 빌드 ────────────────────────────────────────────────────────────
_ICONS = {
    "장단기금리차(10Y-3M)": "💸",
    "하이일드스프레드":      "⚠️",
    "구리(HG=F)":           "🔴",
    "다우운송지수(^DJT)":   "🚛",
}

def build_blocks(data: dict, comment: str) -> list:
    today = datetime.now().strftime("%Y년 %m월 %d일 (%a)")
    now   = datetime.now().strftime("%H:%M KST")

    blocks: list = [
        {
            "type": "header",
            "text": {"type": "plain_text",
                     "text": f"📊 4대 핵심 매크로 — {today}", "emoji": True},
        },
        {"type": "divider"},
    ]

    for name, v in data.items():
        chg      = v["chg"]
        unit     = v["chg_unit"]
        inverted = v.get("inverted_ma", False)
        icon     = _ICONS.get(name, "📌")
        arrow    = _arrow(chg, threshold=0.05 if unit == "%p" else 0.3)
        price    = v["fmt"](v["curr"])
        chg_str  = f"{chg:+.3f}{unit}" if unit == "%p" else f"{chg:+.2f}{unit}"

        # 추세 라인
        parts = []
        if v.get("pct_ma50") is not None:
            pct  = v["pct_ma50"]
            lbl  = _ma_label(pct, inverted)
            parts.append(f"MA50 {pct:+.1f}% {lbl}")
        if v.get("pct_from_high") is not None:
            pct  = v["pct_from_high"]
            lbl  = _high_label(pct)
            parts.append(f"52주고점 {pct:+.1f}% {lbl}")
        trend = "  ·  ".join(parts)

        text = f"{icon} *{name}*: {price}  {arrow} `({chg_str})`"
        if trend:
            text += f"\n    └ {trend}"

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})

    blocks.append({"type": "divider"})

    if comment:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"💬 {comment}"},
        })
        blocks.append({"type": "divider"})

    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"_업데이트: {now}_"}],
    })
    return blocks


# ── 슬랙 전송 ─────────────────────────────────────────────────────────────────
def send_slack(blocks: list, fallback: str) -> bool:
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        print("[ERROR] SLACK_BOT_TOKEN 또는 SLACK_CHANNEL_ID 미설정", file=sys.stderr)
        return False
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": SLACK_CHANNEL_ID, "text": fallback, "blocks": blocks},
            timeout=15,
        )
        body = resp.json()
        if not body.get("ok"):
            print(f"[ERROR] 슬랙 전송 실패: {body.get('error')}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[ERROR] 슬랙 요청 오류: {e}", file=sys.stderr)
        return False


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 슬랙 시황 봇 실행 시작")

    data = fetch_market_data()
    if not data:
        print("[ERROR] 시장 데이터 수집 실패", file=sys.stderr)
        sys.exit(1)

    print(f"수집 완료: {list(data.keys())}")

    comment = get_gemini_comment(data)
    if comment:
        print(f"AI 코멘트: {comment}")

    blocks  = build_blocks(data, comment)
    fallback = "📊 4대 매크로 | " + " | ".join(
        f"{k}: {v['fmt'](v['curr'])} ({v['chg']:+.3f}{v['chg_unit']})"
        for k, v in data.items()
    )

    ok = send_slack(blocks, fallback)
    print("✅ 슬랙 전송 완료" if ok else "❌ 슬랙 전송 실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
