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

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
_raw_channel    = os.getenv("SLACK_CHANNEL_ID", "")
# "T0B0KLJTNRH/C0B0VKT1A76" 형식이면 채널 ID만 추출
SLACK_CHANNEL_ID = _raw_channel.split("/")[-1] if "/" in _raw_channel else _raw_channel

GEMINI_KEY      = os.getenv("GEMINI_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.0-flash-lite"
GEMINI_FALLBACK = ["gemini-2.0-flash", "gemini-2.5-flash"]

INDICATORS = [
    ("S&P500",     "^GSPC",    lambda v: f"{v:,.1f}"),
    ("나스닥100",  "^NDX",     lambda v: f"{v:,.1f}"),
    ("금",         "GC=F",     lambda v: f"${v:,.1f}"),
    ("WTI유가",    "CL=F",     lambda v: f"${v:.2f}"),
    ("미10년채권", "^TNX",     lambda v: f"{v:.3f}%"),
    ("달러인덱스", "DX-Y.NYB", lambda v: f"{v:.2f}"),
]


def fetch_market_data() -> dict:
    results = {}
    tickers = [t for _, t, _ in INDICATORS]
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

    return results


def get_gemini_comment(data: dict) -> str:
    if not GEMINI_KEY or not data:
        return ""
    try:
        import google.genai as genai
        client = genai.Client(api_key=GEMINI_KEY)
        summary = "  /  ".join(f"{k} {v['chg']:+.2f}%" for k, v in data.items())
        prompt = (
            f"오늘 글로벌 시장 변동: {summary}\n"
            "한국 주식 투자자 관점에서 이 시황을 한 문장으로 요약해줘. "
            "이모지 포함, 핵심 흐름과 투자 시사점 위주로. "
            "반드시 완전한 문장으로 끝내야 하며, 절대 중간에 끊기지 않아야 함."
        )
        for model in [GEMINI_MODEL] + GEMINI_FALLBACK:
            try:
                cfg = genai.types.GenerateContentConfig(temperature=0.7, max_output_tokens=1024)
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


def build_blocks(data: dict, comment: str) -> list:
    today = datetime.now().strftime("%Y년 %m월 %d일 (%a)")
    now   = datetime.now().strftime("%H:%M KST")

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📊 글로벌 시황 — {today}", "emoji": True},
        },
        {"type": "divider"},
    ]

    indicator_lines = []
    for name, v in data.items():
        chg = v["chg"]
        arrow = "🔺" if chg > 0.5 else ("🔻" if chg < -0.5 else "➖")
        price_str = v["fmt"](v["current"])
        indicator_lines.append(f"{arrow} *{name}*: {price_str}  `({chg:+.2f}%)`")

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(indicator_lines)},
    })

    if comment:
        blocks.append({"type": "divider"})
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

    blocks = build_blocks(data, comment)
    fallback = "📊 글로벌 시황 " + "  ".join(
        f"{k} {v['chg']:+.2f}%" for k, v in data.items()
    )

    ok = send_slack(blocks, fallback)
    print("✅ 슬랙 전송 완료" if ok else "❌ 슬랙 전송 실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
