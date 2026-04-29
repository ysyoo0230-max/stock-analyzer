"""
글로벌 시황 텔레그램 봇
매일 오전 7시(KST) GitHub Actions 또는 로컬에서 실행
"""
import os
import sys
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime

# Windows 콘솔 인코딩 대응
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp949", "cp1252", "ascii"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() in ("cp949", "cp1252", "ascii"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── .env 로드 ──────────────────────────────────────────────────────────────
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

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_KEY      = os.getenv("GEMINI_KEY", "") or os.getenv("GEMINI_API_KEY", "")

GEMINI_MODEL         = "gemini-2.0-flash-lite"
GEMINI_FALLBACK      = ["gemini-2.0-flash", "gemini-2.5-flash"]

# ── 수집 지표 ─────────────────────────────────────────────────────────────
# (이름, yfinance ticker, 단위 포맷 함수)
INDICATORS = [
    ("S&P500",      "^GSPC",     lambda v: f"{v:,.1f}"),
    ("나스닥100",   "^NDX",      lambda v: f"{v:,.1f}"),
    ("금",          "GC=F",      lambda v: f"${v:,.1f}"),
    ("WTI유가",     "CL=F",      lambda v: f"${v:.2f}"),
    ("미10년채권",  "^TNX",      lambda v: f"{v:.3f}%"),
    ("달러인덱스",  "DX-Y.NYB",  lambda v: f"{v:.2f}"),
]


# ── 시장 데이터 수집 ───────────────────────────────────────────────────────
def fetch_market_data() -> dict:
    results = {}
    tickers = [t for _, t, _ in INDICATORS]
    try:
        raw = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        if raw.empty:
            return results
        # Close 레벨 추출
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
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


# ── Gemini 한 줄 코멘트 ────────────────────────────────────────────────────
def get_gemini_comment(data: dict) -> str:
    if not GEMINI_KEY or not data:
        return ""
    try:
        import google.genai as genai
        client = genai.Client(api_key=GEMINI_KEY)
        summary = "  /  ".join(
            f"{k} {v['chg']:+.2f}%" for k, v in data.items()
        )
        prompt = (
            f"오늘 글로벌 시장 변동: {summary}\n"
            "한국 주식 투자자 관점에서 이 시황을 한 문장으로 요약해줘. "
            "이모지 포함, 80자 이내, 핵심 흐름과 투자 시사점 위주로."
        )
        for model in [GEMINI_MODEL] + GEMINI_FALLBACK:
            try:
                cfg = genai.types.GenerateContentConfig(
                    temperature=0.7, max_output_tokens=200
                )
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=cfg
                )
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "404" in err or "NOT_FOUND" in err:
                    continue
                break
    except Exception as e:
        print(f"[WARN] Gemini 오류: {e}", file=sys.stderr)
    return ""


# ── 메시지 포맷팅 ─────────────────────────────────────────────────────────
def format_message(data: dict, comment: str) -> str:
    today = datetime.now().strftime("%Y년 %m월 %d일 (%a)")
    lines = [f"📊 *글로벌 시황* — {today}", ""]

    for name, v in data.items():
        chg = v["chg"]
        if chg > 0.5:
            arrow = "🔺"
        elif chg < -0.5:
            arrow = "🔻"
        else:
            arrow = "➖"
        price_str = v["fmt"](v["current"])
        lines.append(f"{arrow} *{name}*: {price_str}  `({chg:+.2f}%)`")

    if comment:
        lines += ["", f"💬 {comment}"]

    lines += ["", f"_업데이트: {datetime.now().strftime('%H:%M KST')}_"]
    return "\n".join(lines)


# ── 텔레그램 전송 ─────────────────────────────────────────────────────────
def send_telegram(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[ERROR] TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID 미설정", file=sys.stderr)
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
        if not resp.ok:
            print(f"[ERROR] 텔레그램 전송 실패: {resp.status_code} {resp.text}", file=sys.stderr)
        return resp.ok
    except Exception as e:
        print(f"[ERROR] 텔레그램 요청 오류: {e}", file=sys.stderr)
        return False


# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 시황 봇 실행 시작")

    data = fetch_market_data()
    if not data:
        print("[ERROR] 시장 데이터 수집 실패", file=sys.stderr)
        sys.exit(1)

    print(f"수집 완료: {list(data.keys())}")

    comment = get_gemini_comment(data)
    if comment:
        print(f"AI 코멘트: {comment}")

    msg = format_message(data, comment)
    print("\n── 전송 메시지 ──")
    print(msg)
    print("─────────────────")

    ok = send_telegram(msg)
    print("✅ 전송 완료" if ok else "❌ 전송 실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
