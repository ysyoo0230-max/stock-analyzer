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


def fetch_market_data() -> dict:
    results = {}
    tickers = [t for _, t, _ in INDICATORS] + [SPREAD_LONG, SPREAD_SHORT]
    try:
        raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
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
        results[name] = {"current": curr, "prev": prev, "chg": chg, "fmt": fmt_fn}

    # 장단기금리차 (spread = 10Y - 3M, 단위: %p)
    if SPREAD_LONG in close.columns and SPREAD_SHORT in close.columns:
        s_long  = close[SPREAD_LONG].dropna()
        s_short = close[SPREAD_SHORT].dropna()
        idx     = s_long.index.intersection(s_short.index)
        if len(idx) >= 2:
            curr = float(s_long.loc[idx[-1]]) - float(s_short.loc[idx[-1]])
            prev = float(s_long.loc[idx[-2]]) - float(s_short.loc[idx[-2]])
            chg  = curr - prev  # %p 변화량
            results[SPREAD_NAME] = {
                "current": curr, "prev": prev, "chg": chg,
                "fmt": lambda v: f"{v:+.3f}%p",
                "chg_is_abs": True,   # 변화량이 % 아닌 절댓값
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
            v       = data[name]
            chg     = v["chg"]
            is_abs  = v.get("chg_is_abs", False)
            arrow   = _arrow(chg, is_abs)
            price   = v["fmt"](v["current"])
            chg_str = f"{chg:+.3f}%p" if is_abs else f"{chg:+.2f}%"
            lines.append(f"{arrow} *{name}*: {price}  `({chg_str})`")

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
