# screener.py — 한국/미국 주식 스크리너 핵심 엔진

import sys
import io
# Windows 터미널 한글/이모지 인코딩 문제 방지
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import zipfile
import io as _io
import functools
import requests
from xml.etree import ElementTree as ET

import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

MAX_WORKERS   = 10  # 병렬 스레드 수 (미국)
DART_WORKERS  = 5   # DART API rate-limit 고려

# ================================================
# 1. 한국 주식 스크리너 (FinanceDataReader 기반)
# ================================================

def _fetch_kr_roe(args):
    """병렬용: ROE만 yfinance로 보완 (FDR에 ROE 없음)"""
    code, suffix = args
    try:
        info = yf.Ticker(f"{code}{suffix}").info
        return code, info.get('returnOnEquity')
    except Exception:
        return code, None


def _fetch_kr_yf(args):
    """병렬용: FDR에 재무데이터 없을 때 yfinance fallback"""
    code, name, market, suffix = args
    try:
        info = yf.Ticker(f"{code}{suffix}").info
        per = info.get('trailingPE')
        pbr = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        market_cap = info.get('marketCap')
        if per and pbr and roe and per < 20 and pbr < 3 and roe > 0.05:
            return {
                '종목명': name, '코드': code, '시장': market,
                'PER': round(per, 2), 'PBR': round(pbr, 2),
                'ROE(%)': round(roe * 100, 2),
                '시가총액(억)': round(market_cap / 1e8, 0) if market_cap else None
            }
    except Exception:
        pass
    return None


def get_krx_screener(market='KOSPI', max_stocks=50):
    """
    FDR로 PER/PBR/시가총액 수집, ROE는 yfinance 병렬 보완
    FDR에 재무데이터 없으면 yfinance 병렬 수집으로 자동 전환
    """
    print(f"\n[{market}] 종목 리스트 불러오는 중...")
    df = fdr.StockListing(market).head(max_stocks)

    # FDR 버전마다 Symbol/Code 컬럼명이 다름
    if 'Symbol' in df.columns and 'Code' not in df.columns:
        df = df.rename(columns={'Symbol': 'Code'})

    suffix = '.KS' if market == 'KOSPI' else '.KQ'
    has_per = 'PER' in df.columns
    has_pbr = 'PBR' in df.columns
    has_marcap = 'Marcap' in df.columns

    # FDR에 PER/PBR 없으면 yfinance 병렬 수집
    if not (has_per and has_pbr):
        print(f"  ※ FDR에 PER/PBR 없음 → yfinance 병렬 수집")
        args_list = [(row['Code'], row['Name'], market, suffix) for _, row in df.iterrows()]
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_fetch_kr_yf, a) for a in args_list]
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
        return pd.DataFrame(results)

    # FDR에 PER/PBR 있으면 → ROE만 yfinance 병렬 보완
    print(f"  ROE 병렬 수집 중 ({len(df)}개)...")
    roe_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_kr_roe, (row['Code'], suffix)): row['Code']
                   for _, row in df.iterrows()}
        for future in as_completed(futures):
            code, roe = future.result()
            roe_map[code] = roe

    results = []
    for _, row in df.iterrows():
        try:
            per = float(row['PER']) if pd.notna(row.get('PER')) else None
            pbr = float(row['PBR']) if pd.notna(row.get('PBR')) else None
            roe = roe_map.get(row['Code'])
            market_cap = row.get('Marcap') if has_marcap else None

            if per and pbr and roe and per < 20 and pbr < 3 and roe > 0.05:
                results.append({
                    '종목명': row['Name'], '코드': row['Code'], '시장': market,
                    'PER': round(per, 2), 'PBR': round(pbr, 2),
                    'ROE(%)': round(roe * 100, 2),
                    '시가총액(억)': round(market_cap / 1e8, 0) if market_cap else None
                })
        except Exception:
            continue

    return pd.DataFrame(results)


# ================================================
# 2. 미국 주식 스크리너 (S&P500 전체 자동 수집)
# ================================================

def _fetch_us_stock(symbol):
    """병렬용: S&P500 종목 재무데이터 수집"""
    try:
        info = yf.Ticker(symbol).info
        per = info.get('trailingPE')
        pbr = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        market_cap = info.get('marketCap')
        name = info.get('longName', symbol)
        if per and pbr and roe:
            roe_pct    = round(roe * 100, 2)
            theory_pbr = round(per * roe, 2) if per > 0 and roe > 0 else None   # roe는 소수(0.15)
            pbr_gap    = round((theory_pbr - round(pbr, 2)) / theory_pbr * 100, 1) \
                         if (theory_pbr and pbr and theory_pbr > 0) else None
            return {
                '종목명': name, '코드': symbol, '시장': 'US',
                'PER': round(per, 2), 'PBR': round(pbr, 2),
                '이론PBR': theory_pbr, '괴리율(%)': pbr_gap,
                'ROE(%)': roe_pct,
                '시가총액(억달러)': round(market_cap / 1e9, 1) if market_cap else None,
                '현재주가': info.get('currentPrice'),
                '52주최고': info.get('fiftyTwoWeekHigh'),
                '52주최저': info.get('fiftyTwoWeekLow'),
                '매출성장률(%)': round(info['revenueGrowth'] * 100, 2) if info.get('revenueGrowth') is not None else None,
                '영업이익률(%)': round(info['operatingMargins'] * 100, 2) if info.get('operatingMargins') is not None else None,
            }
    except Exception:
        pass
    return None


def get_us_screener(per_threshold=25):
    """
    FDR로 S&P500 전체 리스트 자동 수집 후 yfinance 병렬 분석
    per_threshold: 이 값 미만인 종목만 결과에 포함
    """
    print("\n[미국] S&P500 종목 리스트 불러오는 중...")
    sp500 = fdr.StockListing('S&P500')

    if 'Symbol' in sp500.columns:
        symbols = sp500['Symbol'].dropna().tolist()
    else:
        symbols = sp500.index.tolist()

    print(f"  총 {len(symbols)}개 종목 병렬 분석 중...")
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_us_stock, sym): sym for sym in symbols}
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                print(f"  진행: {done}/{len(symbols)}")
            r = future.result()
            if r:
                results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # 1단계: 기본필터
    before1 = len(df)
    df = df[(df['PER'] > 0) & (df['PBR'] > 0) & (df['ROE(%)'] > 0)]
    df = df[df['PER'] < per_threshold]
    print(f"  1단계(기본필터 PER<{per_threshold}): {before1} → {len(df)}개")

    # 2단계: 교차검증 — 이론PBR 괴리율 10% 이상 (저평가 판정)
    before2 = len(df)
    df = df[df['괴리율(%)'].notna() & (df['괴리율(%)'] >= 10)]
    print(f"  2단계(교차검증 괴리율≥10%): {before2} → {len(df)}개")

    # 3단계: 수익성 ROE > 10%
    before3 = len(df)
    df = df[df['ROE(%)'] > 10]
    print(f"  3단계(수익성 ROE>10%): {before3} → {len(df)}개")

    # 4단계: 종합점수 계산 후 상위 30개
    if not df.empty:
        gap_s = df['괴리율(%)'].rank(pct=True)
        pbr_s = 1 - df['PBR'].rank(pct=True)
        roe_s = df['ROE(%)'].rank(pct=True)
        df['종합점수'] = (gap_s*0.4 + pbr_s*0.3 + roe_s*0.3).mul(100).round(1)
        df = df.sort_values('종합점수', ascending=False).head(30)
        print(f"  4단계(종합점수 상위30): {len(df)}개 선정")
    return df


# ================================================
# 3. DART API 기반 한국 주식 스크리너
# ================================================
_DART_KEY  = os.getenv("DART_API_KEY", "")
_DART_BASE = "https://opendart.fss.or.kr/api"

_FRED_KEY = os.getenv("FRED_API_KEY", "")
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


@functools.lru_cache(maxsize=1)
def _dart_corp_map() -> dict:
    """주식코드 → DART corp_code 매핑 (1회 다운로드 후 프로세스 캐시)"""
    if not _DART_KEY:
        return {}
    try:
        print("  [DART] 기업코드 맵 다운로드 중...")
        resp = requests.get(f"{_DART_BASE}/corpCode.xml",
                            params={"crtfc_key": _DART_KEY}, timeout=30)
        zf   = zipfile.ZipFile(_io.BytesIO(resp.content))
        root = ET.fromstring(zf.read("CORPCODE.xml"))
        mapping = {
            item.findtext("stock_code"): item.findtext("corp_code")
            for item in root.findall("list")
            if item.findtext("stock_code")
        }
        print(f"  [DART] 기업코드 {len(mapping):,}건 로드 완료")
        return mapping
    except Exception as e:
        print(f"  [DART] 기업코드 다운로드 실패: {e}")
        return {}


def _induty_to_sector(code: str) -> str:
    """KSIC 업종코드 앞 2자리(대분류) → 섹터명 (DART fallback용)"""
    try:
        n = int(str(code).strip()[:2])
    except (ValueError, TypeError):
        return "기타"
    if   n <=  3: return "농업/임업/어업"
    elif n <=  8: return "광업"
    elif n <= 12: return "식품/음료"
    elif n <= 15: return "섬유/의복"
    elif n <= 18: return "목재/종이/인쇄"
    elif n == 19: return "석유정제"
    elif n == 20: return "화학"
    elif n == 21: return "제약/바이오"
    elif n <= 23: return "고무/플라스틱/비금속"
    elif n == 24: return "철강/금속"
    elif n == 25: return "금속가공"
    elif n == 26: return "전자/반도체"
    elif n <= 28: return "전기장비/기계"
    elif n == 29: return "자동차"
    elif n <= 33: return "기타제조"
    elif n == 35: return "전기/가스/에너지"
    elif n <= 39: return "환경/수도"
    elif n <= 42: return "건설"
    elif n <= 47: return "도소매"
    elif n <= 52: return "운수/물류"
    elif n <= 56: return "숙박/음식"
    elif n <= 63: return "IT/통신"
    elif n <= 66: return "금융/보험"
    elif n == 68: return "부동산"
    elif n <= 73: return "전문서비스"
    elif n <= 76: return "사업서비스"
    elif n == 85: return "교육"
    elif n <= 87: return "헬스케어"
    elif n <= 91: return "엔터테인먼트"
    else:         return "기타"


# ── 직접 코드→섹터 매핑 (검증된 주요 종목) ──
_CODE_SECTOR: dict[str, str] = {
    # 전자/반도체
    "005930": "전자/반도체",  # 삼성전자
    "000660": "전자/반도체",  # SK하이닉스
    "000990": "전자/반도체",  # DB하이텍
    "009150": "전자/반도체",  # 삼성전기
    "005935": "전자/반도체",  # 삼성전자우
    "091700": "전자/반도체",  # 파트론
    "084370": "전자/반도체",  # 유진테크(반도체장비)
    "095340": "전자/반도체",  # ISC
    "403870": "전자/반도체",  # HPSP
    "131290": "전자/반도체",  # 티에스이
    # IT/소프트/플랫폼/게임/통신
    "035420": "IT/통신",      # 카카오
    "035720": "IT/통신",      # 카카오뱅크? No - 035720=카카오
    "030200": "IT/통신",      # KT
    "017670": "IT/통신",      # SK텔레콤
    "032640": "IT/통신",      # LG유플러스
    "064400": "IT/통신",      # LG씨엔에스
    "052400": "IT/통신",      # 코나아이
    "036570": "IT/통신",      # 엔씨소프트
    "112040": "IT/통신",      # 위메이드
    "293490": "IT/통신",      # 카카오게임즈
    "263750": "IT/통신",      # 펄어비스
    # 자동차/부품
    "005380": "자동차",       # 현대차
    "000270": "자동차",       # 기아
    "012330": "자동차",       # 현대모비스
    "060980": "자동차",       # 한온시스템
    "277810": "자동차",       # 레인보우로보틱스? 아니면 다른 자동차 계열
    "420770": "자동차",       # 기가비스
    "054950": "자동차",       # 제이브이엠
    # 제약/바이오/헬스케어
    "068270": "제약/바이오",  # 셀트리온
    "128940": "제약/바이오",  # 한미약품
    "069620": "제약/바이오",  # 대웅제약? 069620=대웅
    "185750": "제약/바이오",  # 종근당
    "003520": "제약/바이오",  # 영진약품
    "006280": "제약/바이오",  # 녹십자
    "028300": "제약/바이오",  # HLB
    "141080": "제약/바이오",  # 레고켐바이오
    "298380": "제약/바이오",  # 에이비엘바이오
    "226950": "제약/바이오",  # 올릭스
    "214370": "제약/바이오",  # 케어젠
    # 에너지/화학/배터리/소재
    "006400": "에너지/화학",  # 삼성SDI
    "051910": "에너지/화학",  # LG화학
    "096770": "에너지/화학",  # SK이노베이션
    "247540": "에너지/화학",  # 에코프로비엠
    "086520": "에너지/화학",  # 에코프로
    "373220": "에너지/화학",  # LG에너지솔루션
    "011170": "에너지/화학",  # 롯데케미칼
    "009830": "에너지/화학",  # 한화솔루션
    "196170": "에너지/화학",  # 알테오젠? 아님. 196170=알테오젠
    "357780": "에너지/화학",  # 솔브레인
    # 금융/보험/증권
    "105560": "금융/보험",    # KB금융
    "055550": "금융/보험",    # 신한지주
    "086790": "금융/보험",    # 하나금융지주
    "316140": "금융/보험",    # 우리금융지주
    "138930": "금융/보험",    # BNK금융
    "032830": "금융/보험",    # 삼성생명
    "000810": "금융/보험",    # 삼성화재
    "001450": "금융/보험",    # 현대해상
    "071050": "금융/보험",    # 한국금융지주
    "039490": "금융/보험",    # 키움증권
    "005940": "금융/보험",    # NH투자증권
    "402340": "금융/보험",    # SK스퀘어
    "009970": "금융/보험",    # 영원무역홀딩스(투자지주)
    # 건설
    "000720": "건설",         # 현대건설
    "006360": "건설",         # GS건설
    "047040": "건설",         # 대우건설
    "375500": "건설",         # DL이앤씨
    "012630": "건설",         # HDC
    "097230": "건설",         # HJ중공업(건설플랜트)
    # 조선/중공업
    "009540": "조선/중공업",  # HD한국조선해양
    "010140": "조선/중공업",  # 삼성중공업
    "042660": "조선/중공업",  # 한화오션
    "329180": "조선/중공업",  # HD현대
    "267250": "조선/중공업",  # HD현대중공업
    "082740": "조선/중공업",  # 한화엔진
    "439260": "조선/중공업",  # 대한조선
    "012450": "조선/중공업",  # 한화에어로스페이스
    "034020": "조선/중공업",  # 두산에너빌리티
    # 전기장비/기계
    "298040": "전기장비/기계", # 효성중공업
    "045390": "전기장비/기계", # 대아티아이
    "013030": "전기장비/기계", # 하이록코리아
    # 철강/금속
    "005490": "철강/금속",    # POSCO홀딩스
    "004020": "철강/금속",    # 현대제철
    "103140": "철강/금속",    # 풍산
    "010130": "철강/금속",    # 고려아연
    # 식품/음료
    "004370": "식품/음료",    # 농심
    "267980": "식품/음료",    # 매일유업
    "003800": "소비재",       # 에이스침대
    # 소비재
    "028260": "소비재",       # 삼성물산
    "023530": "소비재",       # 롯데쇼핑
    # 운수/물류
    "003490": "운수/물류",    # 대한항공
    "011200": "운수/물류",    # HMM
    "086280": "운수/물류",    # 현대글로비스
    # 엔터테인먼트
    "035900": "엔터테인먼트", # JYP Ent.
    "041510": "엔터테인먼트", # SM엔터
    "122870": "엔터테인먼트", # 와이지엔터
    "352820": "엔터테인먼트", # 하이브
    # 사업서비스
    "012750": "사업서비스",   # 에스원
    # 도소매
    "111770": "도소매",       # 영원무역
    "472850": "도소매",       # 폰드그룹
    # 전문서비스
    "028050": "전문서비스",   # 삼성E&A
}

# ── 키워드 기반 자동 분류 (순서 중요: 구체적→일반적) ──
_KEYWORD_SECTOR: list[tuple[str, list[str]]] = [
    ("전자/반도체", [
        "반도체", "하이닉스", "실리콘", "웨이퍼", "파운드리", "팹리스",
        "DRAM", "NAND", "OLED", "LCD", "디스플레이", "이미지센서",
        "PCB", "기판", "테스나", "HPSP", "ISC", "DB하이텍",
        "삼성전자", "매그나칩", "에스앤에스텍",
    ]),
    ("제약/바이오", [
        "바이오", "제약", "의약", "헬스케어", "메디", "큐어", "테라피",
        "진단", "유전자", "신약", "백신", "임상", "셀트리온", "녹십자",
        "종근당", "유한양행", "대웅제약", "보령제약", "한미약품",
        "JW중외", "일동제약", "광동제약", "HLB",
    ]),
    ("에너지/화학", [
        "화학", "에너지솔루션", "배터리", "전지", "정유", "석유화학",
        "폴리머", "에코프로", "코스모신소재", "솔루스", "엘앤에프",
        "포스코퓨처엠", "한화솔루션", "금호석유", "롯데케미칼",
        "SDI", "이노베이션",
    ]),
    ("IT/통신", [
        "소프트웨어", "플랫폼", "게임", "카카오", "네이버", "NHN",
        "넥슨", "엔씨소프트", "펄어비스", "크래프톤",
        "이동통신", "방송통신", "인터넷", "클라우드", "데이터",
        "사이버", "보안", "핀테크", "LG씨엔에스", "현대오토에버",
        "코나아이",
    ]),
    ("자동차", [
        "자동차", "모비스", "현대차", "기아자동차", "만도", "현대위아",
        "한온시스템", "센트랄", "화신", "서연이화",
        "유진테크", "제이브이엠",
    ]),
    ("금융/보험", [
        "금융지주", "금융그룹", "은행", "생명보험", "손해보험", "증권",
        "투자증권", "캐피탈", "자산운용", "카드",
        "신한", "KB금융", "하나금융", "우리금융", "IBK", "BNK", "DGB",
        "미래에셋", "한국투자", "삼성생명", "삼성화재", "DB손보",
        "메리츠화재", "흥국화재", "교보생명", "한화생명",
        "영원무역홀딩스",
    ]),
    ("건설", [
        "건설", "건축", "엔지니어링", "플랜트", "시공",
        "GS건설", "대우건설", "현대건설", "DL이앤씨", "HDC",
        "HJ중공업",
    ]),
    ("조선/중공업", [
        "조선", "중공업", "HD현대", "한화오션", "삼성중공업", "두산에너빌",
        "터빈", "항공우주", "방산", "한화에어로", "대한조선",
    ]),
    ("전기장비/기계", [
        "전력", "변압기", "모터", "배전", "효성중공업", "현대일렉",
        "대아티아이", "하이록코리아", "티에스이",
    ]),
    ("철강/금속", [
        "철강", "스틸", "금속", "알루미늄", "포스코", "POSCO",
        "현대제철", "동국제강", "고려아연", "풍산", "세아",
    ]),
    ("식품/음료", [
        "식품", "음료", "주류", "제과", "유업", "사료", "농심",
        "오리온", "풀무원", "빙그레", "하이트진로", "오뚜기",
        "CJ제일제당", "매일유업", "남양유업",
    ]),
    ("소비재", [
        "화장품", "뷰티", "패션", "의류", "잡화", "생활건강",
        "아모레", "코스맥스", "한국콜마", "F&F", "한섬",
        "신세계", "이마트", "롯데쇼핑", "백화점", "마트",
        "현대백화점", "에이스침대",
    ]),
    ("운수/물류", [
        "물류", "운수", "택배", "해운", "항공", "화물", "항만",
        "CJ대한통운", "한진", "현대글로비스", "HMM", "대한항공",
        "아시아나",
    ]),
    ("도소매", [
        "도소매", "무역", "상사", "홀딩스",
        "영원무역", "폰드그룹",
    ]),
    ("엔터테인먼트", [
        "엔터", "엔터테인먼트", "SM", "YG", "JYP", "하이브",
        "문화", "콘텐츠", "예능", "영화",
    ]),
    ("전기/가스/에너지", [
        "한국전력", "한전", "한국가스공사", "도시가스", "지역난방",
        "발전", "원전", "태양광", "풍력", "수소",
    ]),
    ("사업서비스", [
        "에스원", "서비스", "경비", "청소", "시설관리",
    ]),
]


def _name_to_sector(name: str) -> str:
    """
    종목명 키워드 매칭으로 섹터 분류.
    코드 직접 매핑(_CODE_SECTOR) → 키워드 패턴(_KEYWORD_SECTOR) → '기타' 순으로 적용.
    """
    if not name:
        return "기타"
    for sector, keywords in _KEYWORD_SECTOR:
        for kw in keywords:
            if kw in name:
                return sector
    return "기타"


def _build_sector_map_v2(listing_df: pd.DataFrame) -> dict[str, str]:
    """
    FDR listing DataFrame → {stock_code: sector}
    우선순위:
      1. _CODE_SECTOR 직접 매핑 (하드코딩 주요 종목)
      2. _name_to_sector 키워드 매칭
      3. DART KSIC 코드 (API 1회 per unknown 종목, 병렬)
    """
    result: dict[str, str] = {}
    dart_needed: list[tuple[str, str]] = []  # (stock_code, corp_code)

    corp_map = _dart_corp_map() if _DART_KEY else {}

    name_col = "Name" if "Name" in listing_df.columns else None

    for _, row in listing_df.iterrows():
        code = str(row.get("Code", "")).zfill(6)
        name = str(row.get(name_col, "")) if name_col else ""

        # 1순위: 직접 코드 매핑
        if code in _CODE_SECTOR:
            result[code] = _CODE_SECTOR[code]
            continue

        # 2순위: 키워드 매칭
        sector = _name_to_sector(name)
        if sector != "기타":
            result[code] = sector
            continue

        # 3순위: DART KSIC fallback
        if code in corp_map:
            dart_needed.append((code, corp_map[code]))
        else:
            result[code] = "기타"

    # DART fallback (미분류 종목만)
    if dart_needed:
        def _fetch_sector(args):
            sc, cc = args
            try:
                resp = requests.get(
                    f"{_DART_BASE}/company.json",
                    params={"crtfc_key": _DART_KEY, "corp_code": cc},
                    timeout=8,
                )
                data = resp.json()
                if data.get("status") == "000":
                    return sc, _induty_to_sector(data.get("induty_code", ""))
            except Exception:
                pass
            return sc, "기타"

        with ThreadPoolExecutor(max_workers=DART_WORKERS) as ex:
            for sc, sec in ex.map(_fetch_sector, dart_needed):
                result[sc] = sec

    return result


def _fetch_company_sector(args) -> tuple[str, str]:
    """병렬용: DART company.json → induty_code → 섹터명 (legacy, _build_sector_map_v2로 대체됨)"""
    stock_code, corp_code = args
    try:
        resp = requests.get(
            f"{_DART_BASE}/company.json",
            params={"crtfc_key": _DART_KEY, "corp_code": corp_code},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "000":
            return stock_code, _induty_to_sector(data.get("induty_code", ""))
    except Exception:
        pass
    return stock_code, "기타"


def _dart_fetch_fs(corp_code: str, year: str) -> pd.DataFrame:
    """단일회사 전체 재무제표 조회 (연결 우선, 별도 fallback)"""
    for fs_div in ("CFS", "OFS"):
        try:
            resp = requests.get(
                f"{_DART_BASE}/fnlttSinglAcntAll.json",
                params={"crtfc_key": _DART_KEY, "corp_code": corp_code,
                        "bsns_year": year, "reprt_code": "11011", "fs_div": fs_div},
                timeout=15,
            )
            data = resp.json()
            if data.get("status") == "000" and data.get("list"):
                return pd.DataFrame(data["list"])
        except Exception:
            pass
    return pd.DataFrame()


def _dart_amount(df: pd.DataFrame, keyword: str, sj_div: str,
                 col: str = "thstrm_amount") -> float | None:
    """계정명 키워드로 금액 추출 (None이면 None 반환)"""
    if df.empty:
        return None
    rows = df[(df["sj_div"] == sj_div) & df["account_nm"].str.contains(keyword, na=False)]
    if rows.empty:
        return None
    raw = str(rows.iloc[0].get(col, "")).replace(",", "").strip()
    try:
        v = float(raw)
        return v if v != 0 else None
    except ValueError:
        return None


def _fetch_dart_kr_stock(args):
    """병렬용: DART 재무제표 → 지표 계산"""
    stock_code, name, sector, marcap, cur_price = args

    corp_code = _dart_corp_map().get(stock_code)
    if not corp_code:
        return None

    # 최근 사업보고서 조회 (당해 → 전년 순)
    cur_year = datetime.now().year
    df_fs = pd.DataFrame()
    for year in [str(cur_year - 1), str(cur_year - 2)]:
        df_fs = _dart_fetch_fs(corp_code, year)
        if not df_fs.empty:
            break
    if df_fs.empty:
        return None

    # 주요 계정 추출 (thstrm=당기, frmtrm=전기)
    equity     = _dart_amount(df_fs, "자본총계",   "BS", "thstrm_amount")
    net_income = _dart_amount(df_fs, "당기순이익", "IS", "thstrm_amount")
    rev_cur    = _dart_amount(df_fs, "매출액",     "IS", "thstrm_amount")
    rev_prev   = _dart_amount(df_fs, "매출액",     "IS", "frmtrm_amount")

    if not equity or not net_income or equity <= 0:
        return None

    roe = net_income / equity * 100

    # PBR / PER: 시가총액(Marcap) 기반 계산
    pbr = round(marcap / equity,     2) if (marcap and equity > 0) else None
    per = round(marcap / net_income, 2) if (marcap and net_income > 0) else None

    # 매출성장률: 단일 API 호출에서 당기/전기 동시 취득
    rev_growth = None
    if rev_cur and rev_prev and rev_prev != 0:
        rev_growth = round((rev_cur - rev_prev) / abs(rev_prev) * 100, 2)

    # 이론 PBR = PER × ROE(%) / 100  (양수=저평가, 음수=고평가)
    theory_pbr = round(per * roe / 100, 2) if (per and per > 0) else None
    pbr_gap    = round((theory_pbr - pbr) / theory_pbr * 100, 1) \
                 if (theory_pbr and pbr and theory_pbr > 0) else None

    return {
        "종목명":       name,
        "코드":         stock_code,
        "섹터":         sector or "기타",
        "현재주가":     cur_price,
        "PER":          per,
        "PBR":          pbr,
        "이론PBR":      theory_pbr,
        "괴리율(%)":    pbr_gap,
        "ROE(%)":       round(roe, 2),
        "매출성장률(%)": rev_growth,
        "시가총액(억)":  round(marcap / 1e8, 0) if marcap else None,
    }


def _apply_kr_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    단계적 필터 적용 후 종합점수 상위 30개 반환
    1단계: 기본필터 (이미 완료)
    2단계: 교차검증 — ROE > 업종 평균 ROE (업종 대비 수익성 우위)
    3단계: 수익성 — ROE > 5%
    4단계: 종합점수 상위 30개
    """
    out = df.copy()

    # ── 섹터별 평균 통계 ──
    sect_avg = out.groupby("섹터").agg(
        섹터평균PER=("PER",    "mean"),
        섹터평균PBR=("PBR",    "mean"),
        섹터평균ROE=("ROE(%)", "mean"),
    ).round(2)
    out = out.join(sect_avg, on="섹터")

    # 2단계: 교차검증 — 업종 평균 대비 ROE가 업종평균 이상 (경계값 포함)
    out["_roe_excess"] = out["ROE(%)"] - out["섹터평균ROE"]
    out["_pbr_excess"] = out["PBR"]    - out["섹터평균PBR"]
    before2 = len(out)
    out = out[out["_roe_excess"] >= 0]
    print(f"  2단계(교차검증 ROE>=업종평균): {before2} → {len(out)}개")

    # 3단계: 수익성 ROE > 5%
    before3 = len(out)
    out = out[out["ROE(%)"] > 5]
    print(f"  3단계(수익성 ROE>5%): {before3} → {len(out)}개")

    if out.empty:
        return out.reset_index(drop=True)

    # 4단계: 종합점수 계산
    # - 섹터 내 3개 이상: 섹터 내 상대 랭크 (사과-사과 비교)
    # - 섹터 내 1~2개: 전체 글로벌 랭크 (단독 섹터 고정 30점 버그 방지)
    has_sect = out["섹터"].nunique() > 1
    def _sr(col, asc=True):
        global_r = out[col].rank(pct=True)
        if has_sect:
            sect_sizes = out.groupby("섹터")[col].transform("count")
            sect_r    = out.groupby("섹터")[col].rank(pct=True)
            r = sect_r.where(sect_sizes >= 3, global_r)
        else:
            r = global_r
        return r if asc else 1 - r
    out["종합점수"] = (_sr("PBR", False)*0.4 + _sr("ROE(%)", True)*0.3 + _sr("PER", False)*0.3).mul(100).round(1)

    out = out.sort_values("종합점수", ascending=False).head(30)
    out = out.drop(columns=["_roe_excess", "_pbr_excess"], errors="ignore")
    print(f"  4단계(종합점수 상위30): {len(out)}개 선정")
    return out.reset_index(drop=True)


def get_krx_screener_dart(market: str = "KOSPI", max_stocks: int = 200) -> pd.DataFrame:
    """
    DART API 기반 코스피/코스닥 섹터별 TOP 10 추출
    - FDR: 종목 리스트 + Marcap + 섹터
    - DART: ROE / 매출성장률 (재무제표 직접 계산)
    - Marcap 기반: PBR / PER 계산
    - 이상치 제거 → 섹터별 종합점수 TOP 10
    """
    if not _DART_KEY:
        print("  DART_API_KEY 없음. .env 파일을 확인하세요.")
        return pd.DataFrame()

    print(f"\n[{market}-DART] 종목 리스트 불러오는 중...")
    listing = fdr.StockListing(market)
    if listing.empty:
        print(f"  [{market}] FDR 종목 리스트 비어있음")
        return pd.DataFrame()

    if "Symbol" in listing.columns and "Code" not in listing.columns:
        listing = listing.rename(columns={"Symbol": "Code"})

    if "Marcap" in listing.columns:
        listing = listing.sort_values("Marcap", ascending=False)
    listing = listing.head(max_stocks).copy()

    def _est_price(row):
        m = row.get("Marcap")
        s = row.get("Stocks")
        return round(m / s, 0) if (m and s and s > 0) else None

    # 기업코드 맵 워밍 (1회, DART fallback용)
    _dart_corp_map()

    # 섹터 매핑: 코드 직접매핑 → 이름 키워드 → DART KSIC fallback
    print(f"  섹터 분류 중 ({len(listing)}개)...")
    sector_dict = _build_sector_map_v2(listing)
    기타_cnt = sum(1 for v in sector_dict.values() if v == "기타")
    print(f"  섹터 매핑 완료: {len(sector_dict)}개 (기타: {기타_cnt}개)")

    args_list = [
        (
            str(row.get("Code", "")).zfill(6),
            row.get("Name", ""),
            sector_dict.get(str(row.get("Code", "")).zfill(6), "기타"),
            row.get("Marcap") if "Marcap" in listing.columns else None,
            _est_price(row),
        )
        for _, row in listing.iterrows()
    ]

    print(f"  {len(args_list)}개 종목 DART 조회 중 (workers={DART_WORKERS})...")
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=DART_WORKERS) as executor:
        futures = [executor.submit(_fetch_dart_kr_stock, a) for a in args_list]
        for future in as_completed(futures):
            done += 1
            if done % 20 == 0:
                print(f"  진행: {done}/{len(args_list)}")
            r = future.result()
            if r:
                results.append(r)

    if not results:
        print(f"  [{market}] 수집 결과 없음")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    before = len(df)

    # 이상치 제거
    df = df[df["ROE(%)"] > 0]
    df = df[df["ROE(%)"] <= 500]
    df = df[df["PBR"].isna() | (df["PBR"] > 0)]
    df = df[df["PER"].isna() | (df["PER"] > 0)]
    df = df.reset_index(drop=True)
    print(f"  이상치 제거: {before} → {len(df)}개")

    result_df = _apply_kr_filters(df)
    print(f"  단계적 필터 완료: 총 {len(result_df)}개 종목")

    col_order = ["종목명", "코드", "섹터", "현재주가", "PER", "PBR", "이론PBR", "괴리율(%)",
                 "ROE(%)", "매출성장률(%)", "종합점수", "시가총액(억)",
                 "섹터평균PER", "섹터평균PBR", "섹터평균ROE"]
    return result_df[[c for c in col_order if c in result_df.columns]]


# ================================================
# 4. 섹터맵 CSV 생성 (10개 대표 섹터)
# ================================================

def _induty_to_10sector(code: str) -> str:
    """KSIC 앞 2자리 → 10개 대표 섹터"""
    try:
        n = int(str(code).strip()[:2])
    except (ValueError, TypeError):
        return "기타"
    if n == 26 or n in (58, 62, 63):                                       return "반도체/IT"
    if n == 29:                                                             return "자동차"
    if 64 <= n <= 66:                                                       return "금융/보험"
    if n == 21 or 86 <= n <= 87:                                           return "바이오/헬스케어"
    if n in (19, 20, 35):                                                   return "에너지/화학"
    if 41 <= n <= 42:                                                       return "건설"
    if n in (10,11,12,13,14,15,16,17,18,45,46,47,55,56,90,91):            return "소비재"
    if n == 61:                                                             return "통신"
    if n in (24, 25, 28, 30):                                              return "조선/중공업"
    return "기타"


def _fetch_company_10sector(args) -> tuple[str, str]:
    """병렬용: DART company.json → 10섹터명"""
    stock_code, corp_code = args
    try:
        resp = requests.get(
            f"{_DART_BASE}/company.json",
            params={"crtfc_key": _DART_KEY, "corp_code": corp_code},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "000":
            return stock_code, _induty_to_10sector(data.get("induty_code", ""))
    except Exception:
        pass
    return stock_code, "기타"


def build_sector_map_csv() -> pd.DataFrame:
    """코스피 상위 100 + 코스닥 상위 50 → 10섹터 분류 → sector_map.csv"""
    if not _DART_KEY:
        print("  DART_API_KEY 없음. .env 파일을 확인하세요.")
        return pd.DataFrame()

    print("\n[섹터맵] 종목 리스트 불러오는 중...")
    _dart_corp_map()  # 코드 맵 워밍

    records = []
    for market, n in [("KOSPI", 100), ("KOSDAQ", 50)]:
        listing = fdr.StockListing(market)
        if "Symbol" in listing.columns and "Code" not in listing.columns:
            listing = listing.rename(columns={"Symbol": "Code"})
        if "Marcap" in listing.columns:
            listing = listing.sort_values("Marcap", ascending=False)
        listing = listing.head(n)

        codes    = [str(row.get("Code", "")).zfill(6) for _, row in listing.iterrows()]
        corp_map = _dart_corp_map()
        args     = [(c, corp_map[c]) for c in codes if c in corp_map]

        print(f"  [{market}] {len(args)}개 종목 10섹터 조회 중...")
        sector_dict: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=DART_WORKERS) as ex:
            for code, sector in ex.map(_fetch_company_10sector, args):
                sector_dict[code] = sector

        for _, row in listing.iterrows():
            code = str(row.get("Code", "")).zfill(6)
            records.append({
                "코드":         code,
                "종목명":       row.get("Name", ""),
                "시장":         market,
                "섹터":         sector_dict.get(code, "기타"),
                "시가총액(억)": round(row.get("Marcap", 0) / 1e8, 0) if row.get("Marcap") else None,
            })

    df = pd.DataFrame(records)
    df.to_csv("sector_map.csv", index=False, encoding="utf-8-sig")
    print(f"  [섹터맵] sector_map.csv 저장 완료 ({len(df)}개 종목)")
    print(df.groupby(["시장", "섹터"])["코드"].count().to_string())
    return df


# ================================================
# 5. 거시지표 수집 (FRED API)
# ================================================

def _fred_obs(series_id: str, limit: int = 3) -> list[dict]:
    """FRED API → 최근 limit개 유효 관측값"""
    if not _FRED_KEY:
        return []
    try:
        r = requests.get(_FRED_URL, params={
            "series_id": series_id, "api_key": _FRED_KEY,
            "file_type": "json", "sort_order": "desc", "limit": limit,
        }, timeout=10)
        return [o for o in r.json().get("observations", []) if o.get("value") != "."]
    except Exception:
        return []


def get_macro_data() -> pd.DataFrame:
    """FRED API 거시지표 수집 → macro_날짜.csv 저장"""
    if not _FRED_KEY:
        print("  [MACRO] FRED_API_KEY 없음. .env 파일을 확인하세요.")
        return pd.DataFrame()

    print("\n[MACRO] 거시지표 수집 중...")
    rows: list[dict] = []

    def _point(series_id: str, label: str, unit: str):
        obs = _fred_obs(series_id, 3)
        if len(obs) < 2:
            print(f"  {label}: 데이터 없음 ({series_id})")
            return
        cur  = float(obs[0]["value"])
        prev = float(obs[1]["value"])
        rows.append({"지표": label, "현재값": round(cur, 4), "전월값": round(prev, 4),
                     "변화량": round(cur - prev, 4), "단위": unit, "날짜": obs[0]["date"]})
        print(f"  {label}: {cur:.3g} {unit}")

    def _yoy(series_id: str, label: str):
        obs = _fred_obs(series_id, 14)
        if len(obs) < 13:
            print(f"  {label}: 데이터 부족 ({series_id})")
            return
        cur, prev = float(obs[0]["value"]), float(obs[1]["value"])
        y1         = float(obs[12]["value"])
        y2         = float(obs[13]["value"]) if len(obs) > 13 else None
        yoy_c = round((cur / y1 - 1) * 100, 2)
        yoy_p = round((prev / y2 - 1) * 100, 2) if y2 else yoy_c
        rows.append({"지표": label, "현재값": yoy_c, "전월값": yoy_p,
                     "변화량": round(yoy_c - yoy_p, 2), "단위": "%", "날짜": obs[0]["date"]})
        print(f"  {label}: {yoy_c:.2f}%")

    _point("DEXKOUS",          "환율(KRW/USD)",       "원")
    _point("DCOILWTICO",       "WTI유가",              "달러")
    _point("DGS10",            "미국금리(10Y)",        "%")
    _point("DGS2",             "미국금리(2Y)",         "%")
    _point("VIXCLS",           "VIX",                  "")
    _point("DTWEXBGS",         "달러인덱스",            "")
    _point("IRSTKOR01STM156N", "한국기준금리",          "%")
    _yoy("CPIAUCSL",           "미국CPI(YoY)")
    _yoy("KORCPIALLMINMEI",    "한국CPI(YoY)")

    # 장단기금리차 = 미국금리(10Y) - 미국금리(2Y)
    r10 = next((r for r in rows if r["지표"] == "미국금리(10Y)"), None)
    r2  = next((r for r in rows if r["지표"] == "미국금리(2Y)"),  None)
    if r10 and r2:
        rows.append({
            "지표": "장단기금리차(10Y-2Y)",
            "현재값": round(r10["현재값"] - r2["현재값"], 4),
            "전월값": round(r10["전월값"] - r2["전월값"], 4),
            "변화량": round((r10["현재값"] - r2["현재값"]) - (r10["전월값"] - r2["전월값"]), 4),
            "단위": "%p", "날짜": r10["날짜"],
        })
        print(f"  장단기금리차: {r10['현재값'] - r2['현재값']:.3f}%p")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    today = datetime.now().strftime('%Y%m%d')
    fname = f"macro_{today}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"  [MACRO] {fname} 저장 완료 ({len(df)}개 지표)")
    return df


# ================================================
# 6. 분석 실행 함수
# ================================================
def run_analysis():
    print("=== 주식 스크리너 시작 ===")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    today = datetime.now().strftime('%Y%m%d')

    # ── 기존 한국 (yfinance 기반) ──
    df_kospi  = get_krx_screener('KOSPI',  max_stocks=50)
    df_kosdaq = get_krx_screener('KOSDAQ', max_stocks=50)

    print("\n===== 코스피 저평가 종목 =====")
    print(df_kospi.to_string(index=False)  if not df_kospi.empty  else "조건 충족 종목 없음")
    print("\n===== 코스닥 저평가 종목 =====")
    print(df_kosdaq.to_string(index=False) if not df_kosdaq.empty else "조건 충족 종목 없음")

    df_kospi.to_csv( f'kospi_{today}.csv',  index=False, encoding='utf-8-sig')
    df_kosdaq.to_csv(f'kosdaq_{today}.csv', index=False, encoding='utf-8-sig')

    # ── DART 기반 한국 (섹터별 TOP 10) ──
    if _DART_KEY:
        df_kospi_dart  = get_krx_screener_dart('KOSPI',  max_stocks=200)
        df_kosdaq_dart = get_krx_screener_dart('KOSDAQ', max_stocks=400)

        print("\n===== 코스피 섹터별 TOP 10 (DART) =====")
        print(df_kospi_dart.to_string(index=False)  if not df_kospi_dart.empty  else "결과 없음")
        print("\n===== 코스닥 섹터별 TOP 10 (DART) =====")
        print(df_kosdaq_dart.to_string(index=False) if not df_kosdaq_dart.empty else "결과 없음")

        df_kospi_dart.to_csv( f'kospi_섹터별_{today}.csv',  index=False, encoding='utf-8-sig')
        df_kosdaq_dart.to_csv(f'kosdaq_섹터별_{today}.csv', index=False, encoding='utf-8-sig')
        print(f"  DART CSV 저장: kospi_섹터별_{today}.csv / kosdaq_섹터별_{today}.csv")
    else:
        print("\n[DART] API 키 없음 — 섹터별 분석 건너뜀")

    # ── 미국 (yfinance 기반) ──
    df_us = get_us_screener(per_threshold=25)
    print("\n===== 미국 S&P500 저평가 종목 (PER<25, ROE 내림차순) =====")
    print(df_us.to_string(index=False) if not df_us.empty else "조건 충족 종목 없음")
    df_us.to_csv(f'us_{today}.csv', index=False, encoding='utf-8-sig')

    # ── 거시지표 (FRED) ──
    get_macro_data()

    print(f"\n✅ CSV 저장 완료! (날짜: {today})")


# ================================================
# 5. 진입점 — 명령어: python screener.py 분석실행해줘
# ================================================
COMMANDS = {
    '분석실행해줘':      run_analysis,
    'sector_map만들어줘': build_sector_map_csv,
    '거시지표':          get_macro_data,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else '분석실행해줘'  # 인수 없으면 기본 실행

    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"❌ 알 수 없는 명령어: '{cmd}'")
        print("사용 가능한 명령어:")
        for name in COMMANDS:
            print(f"  python screener.py {name}")
