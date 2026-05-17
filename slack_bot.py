"""
글로벌 시황 슬랙 봇
매일 오전 7시(KST) GitHub Actions 또는 로컬에서 실행
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
_raw_channel     = os.getenv("SLACK_CHANNEL_ID", "")
SLACK_CHANNEL_ID = _raw_channel.split("/")[-1] if "/" in _raw_channel else _raw_channel

GEMINI_KEY      = os.getenv("GEMINI_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.0-flash-lite"
GEMINI_FALLBACK = ["gemini-2.0-flash", "gemini-2.5-flash"]

# (표시명, ticker, 포맷함수)
# 장단기금리차는 별도 계산 — SPREAD_TICKERS 참고
INDICATORS = [
    # 미국
    ("S&P500",        "^GSPC",     lambda v: f"{v:,.1f}"),
    ("나스닥100",     "^NDX",      lambda v: f"{v:,.1f}"),
    ("VIX",           "^VIX",      lambda v: f"{v:.2f}"),
    ("WTI유가",       "CL=F",      lambda v: f"${v:.2f}"),
    ("브렌트유",      "BZ=F",      lambda v: f"${v:.2f}"),
    # 한국
    ("코스피",        "^KS11",     lambda v: f"{v:,.2f}"),
    ("코스닥",        "^KQ11",     lambda v: f"{v:,.2f}"),
    ("원달러환율",    "USDKRW=X",  lambda v: f"₩{v:,.1f}"),
    # 원자재·지수
    ("금",            "GC=F",      lambda v: f"${v:,.1f}"),
    ("구리",          "HG=F",      lambda v: f"${v:.3f}"),
    ("필라델피아반도체", "^SOX",   lambda v: f"{v:,.1f}"),
]

# 장단기금리차: 미국10년(^TNX) - 미국3개월(^IRX)
SPREAD_LONG  = "^TNX"
SPREAD_SHORT = "^IRX"
SPREAD_NAME  = "장단기금리차(10Y-3M)"

GROUPS = {
    "🇺🇸 미국": ["S&P500", "나스닥100", "VIX", "WTI유가", "브렌트유", SPREAD_NAME],
    "🇰🇷 한국": ["코스피", "코스닥", "원달러환율"],
    "🪨 원자재·지수": ["금", "구리", "필라델피아반도체"],
}


def _trend_label(pct_vs_ma: float) -> str:
    """MA50 대비 괴리율 → 추세 라벨"""
    if pct_vs_ma > 10:  return "🔥과열"
    if pct_vs_ma > 2:   return "📈상승추세"
    if pct_vs_ma > -2:  return "➡️횡보"
    if pct_vs_ma > -10: return "📉조정"
    return "🔻하락추세"


def _high_label(pct_from_high: float) -> str:
    """52주 고점 대비 % → 상태 라벨"""
    if pct_from_high > -3:   return "🏔️고점근처"
    if pct_from_high > -10:  return "⚡고점권"
    if pct_from_high > -20:  return "⚠️조정구간"
    return "🔻급락구간"


def fetch_market_data() -> dict:
    results = {}
    tickers = [t for _, t, _ in INDICATORS] + [SPREAD_LONG, SPREAD_SHORT]
    try:
        # 1년 데이터: MA50 + 52주 고저점 계산용
        raw = yf.download(tickers, period="1y", progress=False, auto_adjust=True)
        if raw.empty:
            return results
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = tickers[:1]
    except Exception as e:
        print(f"[ERROR] yfinance 다운로드 실패: {e}", file=sys.stderr)
        return results

    # 개별 지표
    for name, ticker, fmt_fn in INDICATORS:
        if ticker not in close.columns:
            continue
        series = close[ticker].dropna()
        if len(series) < 2:
            continue
        curr = float(series.iloc[-1])
        prev = float(series.iloc[-2])
        chg  = (curr - prev) / prev * 100 if prev else 0.0

        # MA50 대비
        ma50 = float(series.rolling(50).mean().iloc[-1]) if len(series) >= 50 else None
        pct_ma50 = (curr - ma50) / ma50 * 100 if ma50 else None

        # 52주 고점/저점 대비
        high52 = float(series.max())
        low52  = float(series.min())
        pct_from_high = (curr - high52) / high52 * 100
        pct_from_low  = (curr - low52)  / low52  * 100

        results[name] = {
            "current": curr, "prev": prev, "chg": chg, "fmt": fmt_fn,
            "pct_ma50": pct_ma50,
            "pct_from_high": pct_from_high,
            "pct_from_low":  pct_from_low,
        }

    # 장단기금리차 (spread = 10Y - 3M, 단위: %p)
    if SPREAD_LONG in close.columns and SPREAD_SHORT in close.columns:
        s_long  = close[SPREAD_LONG].dropna()
        s_short = close[SPREAD_SHORT].dropna()
        idx     = s_long.index.intersection(s_short.index)
        if len(idx) >= 2:
            spreads = s_long.loc[idx] - s_short.loc[idx]
            curr = float(spreads.iloc[-1])
            prev = float(spreads.iloc[-2])
            chg  = curr - prev
            ma50_sp  = float(spreads.rolling(50).mean().iloc[-1]) if len(spreads) >= 50 else None
            pct_ma50 = (curr - ma50_sp) / abs(ma50_sp) * 100 if ma50_sp else None
            high52   = float(spreads.max())
            low52    = float(spreads.min())
            pct_from_high = (curr - high52) / abs(high52) * 100 if high52 else None
            results[SPREAD_NAME] = {
                "current": curr, "prev": prev, "chg": chg,
                "fmt": lambda v: f"{v:+.3f}%p",
                "chg_is_abs": True,
                "pct_ma50": pct_ma50,
                "pct_from_high": pct_from_high,
                "pct_from_low": None,
            }

    return results


def get_gemini_comment(data: dict) -> str:
    if not GEMINI_KEY or not data:
        return ""
    try:
        import google.genai as genai
        client  = genai.Client(api_key=GEMINI_KEY)
        us_part = "  /  ".join(
            f"{k} {v['chg']:+.2f}{'%p' if v.get('chg_is_abs') else '%'}"
            for k, v in data.items()
            if k in GROUPS["🇺🇸 미국"]
        )
        kr_part = "  /  ".join(
            f"{k} {v['chg']:+.2f}{'%p' if v.get('chg_is_abs') else '%'}"
            for k, v in data.items()
            if k in GROUPS["🇰🇷 한국"]
        )
        commodity_part = "  /  ".join(
            f"{k} {v['chg']:+.2f}%"
            for k, v in data.items()
            if k in GROUPS["🪨 원자재·지수"]
        )
        prompt = (
            f"[미국] {us_part}\n"
            f"[한국] {kr_part}\n"
            f"[원자재·지수] {commodity_part}\n\n"
            "한국 주식 투자자 관점에서 미국과 한국 시장을 종합해 오늘의 핵심 인사이트를 한 문장으로 완성해줘. "
            "이모지 포함, 글로벌 흐름이 국내 시장에 미치는 영향과 오늘의 투자 시사점 위주로. "
            "반드시 완전한 문장으로 끝내야 하며, 절대 중간에 끊기지 않아야 함."
        )
        for model in [GEMINI_MODEL] + GEMINI_FALLBACK:
            try:
                cfg  = genai.types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024)
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "404" in err or "NOT_FOUND" in err:
                    continue
                break
    except Exception as e:
        print(f"[WARN] Gemini 오류: {e}", file=sys.stderr)
    return ""


def _arrow(chg: float, is_abs: bool = False) -> str:
    threshold = 0.05 if is_abs else 0.5
    if chg > threshold:
        return "🔺"
    if chg < -threshold:
        return "🔻"
    return "➖"


def build_blocks(data: dict, comment: str) -> list:
    today = datetime.now().strftime("%Y년 %m월 %d일 (%a)")
    now   = datetime.now().strftime("%H:%M KST")

    blocks: list = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📊 글로벌 시황 — {today}", "emoji": True},
        },
        {"type": "divider"},
    ]

    for group_label, names in GROUPS.items():
        lines = []
        for name in names:
            if name not in data:
                continue
            v        = data[name]
            chg      = v["chg"]
            is_abs   = v.get("chg_is_abs", False)
            arrow    = _arrow(chg, is_abs)
            price    = v["fmt"](v["current"])
            chg_str  = f"{chg:+.3f}%p" if is_abs else f"{chg:+.2f}%"

            # 추세 정보 (MA50 + 52주 고점)
            extra_parts = []
            pct_ma50 = v.get("pct_ma50")
            pct_high = v.get("pct_from_high")
            if pct_ma50 is not None:
                extra_parts.append(f"MA50 {pct_ma50:+.1f}% {_trend_label(pct_ma50)}")
            if pct_high is not None:
                extra_parts.append(f"52주고점 {pct_high:+.1f}% {_high_label(pct_high)}")
            extra = "  ·  ".join(extra_parts)

            line = f"{arrow} *{name}*: {price}  `({chg_str})`"
            if extra:
                line += f"\n    └ {extra}"
            lines.append(line)

        if lines:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{group_label}*\n" + "\n".join(lines)},
            })
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


def send_slack(blocks: list, fallback_text: str) -> bool:
    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        print("[ERROR] SLACK_BOT_TOKEN 또는 SLACK_CHANNEL_ID 미설정", file=sys.stderr)
        return False
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"channel": SLACK_CHANNEL_ID, "text": fallback_text, "blocks": blocks},
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

    blocks   = build_blocks(data, comment)
    fallback = "📊 글로벌 시황 " + "  ".join(
        f"{k} {v['chg']:+.2f}{'%p' if v.get('chg_is_abs') else '%'}"
        for k, v in data.items()
    )

    ok = send_slack(blocks, fallback)
    print("✅ 슬랙 전송 완료" if ok else "❌ 슬랙 전송 실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
