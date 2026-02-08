#!/usr/bin/env python3
# janpara_scrape_geforce.py  (Ubuntu / Python3.12 / 標準ライブラリのみ)
#
# 目的:
# - じゃんぱら検索結果をページ巡回して商品/価格/在庫/型番(RTXxxxx)を抽出
# - 指定RTX世代(20/30/40/50)でフィルタ（既定は全部）
# - CSV保存 + 統計(最安/平均/最大)を全体＆型番別に表示
#
# 使い方（例）:
#   #!/usr/bin/env python3
# janpara_scrape_geforce.py  (Ubuntu / Python3.12 / 標準ライブラリのみ)
#
# 目的:
# - じゃんぱら検索結果をページ巡回して商品/価格/在庫/型番(RTXxxxx)を抽出
# - 指定RTX世代(20/30/40/50)でフィルタ（既定は全部）
# - CSV保存 + 統計(最安/平均/最大)を全体＆型番別に表示
#
# 使い方（例）:
#   python3 janpara_scrape_geforce.py "RTX 8GB" --all-pages
#   python3 janpara_scrape_geforce.py "RTX 12GB" --all-pages
#   python3 janpara_scrape_geforce.py "RTX 8GB" --max-pages 13
#   python3 janpara_scrape_geforce.py "RTX 8GB" --sleep 7 --timeout 40
#
# よく使うオプション:
#   --all-pages         : 1ページ目から最終ページ推定して回す（hard-cap上限あり）
#   --max-pages N       : 最大Nページまで（--all-pagesなしの場合の上限）
#   --sleep SEC         : ページ間待機（既定: 7秒）
#   --timeout SEC       : HTTPタイムアウト（既定: 40秒）
#   --hard-cap N        : --all-pages時の最大上限（既定: 50）
#   --series "20,30,40,50" : 対象RTX世代。空文字("")ならフィルタ無し
#   --debug-save-all    : 各ページHTMLを保存（調査用。ファイルが増える）
#
# 注意:
# - SiteGuard/WAFの影響で 408/429/503/504 が出ることがあるため
#   Cookie保持 + トップ踏み + ヘッダ寄せ + 控えめリトライを入れている。
# - 抽出はリンクテキスト頼みなので、ページ構造変化で0件になることがある。
#   その場合は debug HTML を見て調整する。
#
# 方針:
# - 余計な自動調整や“賢い推測”は入れない（壊れ方を追えることを優先）。
# - 例外は握りつぶさず、ログを残して止まる。

import re
import csv
import time
import sys
import math
import argparse
import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar
from html.parser import HTMLParser

VERSION = "2026-01-25c-geforce"

BASE = "https://www.janpara.co.jp/sale/search/result/"
TOP  = "https://www.janpara.co.jp/"

# --- 解析用 正規表現 ---
PRICE_RE = re.compile(r"中古\s*¥\s*([\d,]+)")
STOCK_RE = re.compile(r"(\d+)個の在庫")

# RTXのみ（GTXは対象外。検索語が混ざっても、ここで拾わない）
MODEL_RE = re.compile(r"RTX\s*?(\d{4})", re.IGNORECASE)

HIT_RE   = re.compile(r"該当件数：\s*(\d+)\s*商品")
PAGE_RE  = re.compile(r"[?&]PAGE=(\d+)")

# --- HTMLから <a href=...> とそのテキストを集める ---
class ATagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_a = False
        self._href = None
        self._buf = []
        self.links = []  # list[(href, text)]

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self._in_a = True
                self._href = href
                self._buf = []

    def handle_data(self, data):
        if self._in_a and data:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = " ".join("".join(self._buf).split())
            self.links.append((self._href, text))
            self._in_a = False
            self._href = None
            self._buf = []

# --- Cookie保持 opener ---
CJ = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CJ))

COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Connection": "close",
}

_COOKIE_WARMED = False

def log_progress(msg: str):
    """進行ログ（コメント風）"""
    print("# " + msg, flush=True)

def sleep_with_indicator(sec: float, label: str = "sleep"):
    """待機中に動いていることが分かるように '.' を出す"""
    s = int(sec)
    if s <= 0:
        return
    sys.stdout.write(f"# {label}: {s}s ")
    sys.stdout.flush()
    for _ in range(s):
        time.sleep(1)
        sys.stdout.write(".")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

def warm_cookies(timeout_sec: int):
    """トップを1回踏んでCookieを確定させる"""
    global _COOKIE_WARMED
    if _COOKIE_WARMED:
        return
    log_progress("warming cookies (GET /)")
    req = urllib.request.Request(TOP, headers=COMMON_HEADERS, method="GET")
    with OPENER.open(req, timeout=timeout_sec) as resp:
        resp.read()
    _COOKIE_WARMED = True
    log_progress("cookie warm done")

def build_search_url(keywords: str, order: str, page: int) -> str:
    params = {
        "cache_key": "/sale/search/result/",
        "KEYWORDS": keywords,
        "OUTCLSCODE": "",
        "SSHPCODE": "",
        "CHKOUTCOM": "1",
        "ORDER": order,
        "PAGE": str(page),
    }
    return BASE + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)

def fetch_html(url: str, timeout_sec: int, sleep_retry=(0, 2, 5, 9), page_hint: str = "") -> str:
    """HTMLを取る。408/429/503/504 だけ控えめに再試行。"""
    headers = dict(COMMON_HEADERS)
    headers["Referer"] = TOP
    req = urllib.request.Request(url, headers=headers, method="GET")

    last_err = None
    for i, wait in enumerate(sleep_retry, start=1):
        if wait:
            log_progress(f"{page_hint}retry wait {wait}s (attempt {i}/{len(sleep_retry)})")
            sleep_with_indicator(wait, label="retry-sleep")

        try:
            log_progress(f"{page_hint}fetching (attempt {i}/{len(sleep_retry)})")
            with OPENER.open(req, timeout=timeout_sec) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                body = resp.read().decode(charset, errors="replace")
                return body
        except urllib.error.HTTPError as e:
            last_err = e
            log_progress(f"{page_hint}HTTPError {e.code}")
            if e.code in (408, 429, 503, 504):
                continue
            raise
        except Exception as e:
            last_err = e
            log_progress(f"{page_hint}Exception {type(e).__name__}")
            continue

    raise last_err

def parse_hit_count(html: str):
    m = HIT_RE.search(html)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def detect_last_page(html: str):
    """HTML内のリンクから PAGE=数字 を拾い、最大値を最終ページ候補として返す。"""
    p = ATagCollector()
    p.feed(html)
    max_page = None
    for href, _text in p.links:
        m = PAGE_RE.search(href)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if (max_page is None) or (n > max_page):
            max_page = n
    return max_page

def parse_items(html: str):
    """検索結果から商品相当のリンクテキストを抽出（RTXのみ）"""
    p = ATagCollector()
    p.feed(html)

    rows = []
    for href, text in p.links:
        if "/sale/search/detail/" not in href:
            continue
        if ("中古" not in text) or ("¥" not in text):
            continue

        m_price = PRICE_RE.search(text)
        m_stock = STOCK_RE.search(text)
        m_model = MODEL_RE.search(text)

        price = int(m_price.group(1).replace(",", "")) if m_price else None
        stock = int(m_stock.group(1)) if m_stock else None
        model = m_model.group(1) if m_model else None
        series = model[:2] if model else None  # "20","30","40","50"

        # RTXが取れない＝対象外（GTX等）として捨てる
        if model is None:
            continue

        rows.append({
            "series": series,
            "model": model,
            "price_yen": price,
            "stock": stock,
            "href": href,
            "text": text,
        })
    return rows

def sanitize_out_name(q: str) -> str:
    """例: 'RTX 8GB' -> 'janpara_geforce_rtx_8gb.csv'"""
    s = q.strip().lower().replace("+", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if not s:
        s = "query"
    return f"janpara_geforce_{s}.csv"

def stats_summary(prices):
    if not prices:
        return None
    mn = min(prices)
    mx = max(prices)
    avg = int(round(sum(prices) / len(prices)))
    return mn, avg, mx

def save_debug_html(page: int, html: str) -> str:
    fn = f"janpara_geforce_page{page:02d}_debug.html"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(html)
    return fn

def main():
    epilog = (
        "例:\n"
        "  python3 janpara_scrape_geforce.py \"RTX 8GB\" --all-pages\n"
        "  python3 janpara_scrape_geforce.py \"RTX 12GB\" --all-pages\n"
        "  python3 janpara_scrape_geforce.py \"RTX 8GB\" --max-pages 13\n"
        "  python3 janpara_scrape_geforce.py \"RTX 8GB\" --series \"30,40,50\" --all-pages\n"
        "\n"
        "メモ:\n"
        "- Amazon等と違い、じゃんぱらはWAFの気分で408/429が出る。\n"
        "- 0件になったら debug HTML を見て、リンク文言/構造変化を疑う。\n"
    )

    ap = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog
    )
    ap.add_argument("keywords", nargs="?", default="RTX 8GB",
                    help='検索語 (例: "RTX 8GB" / "RTX 12GB")')
    ap.add_argument("--series", default="20,30,40,50",
                    help='対象RTX世代 (例: "20,30,40,50") / 空("")ならフィルタ無し')
    ap.add_argument("--max-pages", type=int, default=7,
                    help="巡回する最大ページ数（既定: 7）")
    ap.add_argument("--all-pages", action="store_true",
                    help="1ページ目のリンクから最終ページ推定して回す（hard-capで上限）")
    ap.add_argument("--sleep", type=float, default=7.0,
                    help="ページ間ウェイト秒（既定: 7.0）")
    ap.add_argument("--timeout", type=int, default=40,
                    help="HTTPタイムアウト秒（既定: 40）")
    ap.add_argument("--hard-cap", type=int, default=50,
                    help="--all-pages時の最大上限（既定: 50）")
    ap.add_argument("--debug-save-all", action="store_true",
                    help="各ページHTMLを保存（調査用。ファイルが増える）")
    args = ap.parse_args()

    # series 指定の解釈
    series_list = []
    if args.series.strip():
        series_list = [x.strip() for x in args.series.split(",") if x.strip()]
    series_set = set(series_list)

    log_progress(f"version: {VERSION}")
    warm_cookies(args.timeout)

    # 1ページ目
    url1 = build_search_url(args.keywords, order="3", page=1)
    html1 = fetch_html(url1, timeout_sec=args.timeout, page_hint="page 1/?: ")
    if args.debug_save_all:
        save_debug_html(1, html1)

    hit_count = parse_hit_count(html1)
    rows1 = parse_items(html1)

    if not rows1:
        fn = save_debug_html(1, html1)
        print("query:", args.keywords)
        print("page1: parsed 0 items (debug saved:", fn + ")")
        if hit_count is not None:
            print("hit_count:", hit_count)
        print("version:", VERSION)
        return

    per_page = len(rows1)
    expected_pages = None
    if hit_count is not None and per_page > 0:
        expected_pages = int(math.ceil(hit_count / per_page))

    # 最大ページ決定
    if args.all_pages:
        last = detect_last_page(html1)
        if last is None:
            max_pages = args.max_pages
        else:
            max_pages = min(last, args.hard_cap)
    else:
        max_pages = args.max_pages

    all_rows = []
    pages_ok = 0
    stop_reason = "reached max-pages"
    debug_file = None

    # 1ページ目
    all_rows.extend(rows1)
    pages_ok += 1
    log_progress(f"page 1 parsed {len(rows1)} items")

    # 2ページ目以降
    for page in range(2, max_pages + 1):
        log_progress(f"page {page}/{max_pages}: waiting before fetch")
        sleep_with_indicator(args.sleep, label=f"page-sleep {page}/{max_pages}")

        url = build_search_url(args.keywords, order="3", page=page)
        html = fetch_html(url, timeout_sec=args.timeout, page_hint=f"page {page}/{max_pages}: ")
        if args.debug_save_all:
            save_debug_html(page, html)

        rows = parse_items(html)
        log_progress(f"page {page}/{max_pages} parsed {len(rows)} items (total so far {len(all_rows) + len(rows)})")

        if not rows:
            debug_file = save_debug_html(page, html)
            stop_reason = f"parsed 0 items at page {page} (debug saved)"
            break

        all_rows.extend(rows)
        pages_ok += 1

    # フィルタ（RTX世代）
    if series_list:
        filtered = [r for r in all_rows if r["series"] in series_set]
    else:
        filtered = list(all_rows)

    # CSV出力
    out_csv = sanitize_out_name(args.keywords)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series", "model", "price_yen", "stock", "href", "text"])
        w.writeheader()
        w.writerows(filtered)

    # 統計（全体）
    prices_all = [r["price_yen"] for r in filtered if isinstance(r["price_yen"], int)]
    s_all = stats_summary(prices_all)

    # 結果出力（あなたの形式に寄せる）
    print("query:", args.keywords)
    print("saved:", out_csv, "rows=" + str(len(filtered)))

    # 追加の診断ログ
    if hit_count is not None:
        print("hit_count:", hit_count, "per_page(observed):", per_page)
    if expected_pages is not None:
        print("expected_pages:", expected_pages)
    print("pages fetched:", pages_ok, "/", max_pages, "stop:", stop_reason)
    if debug_file:
        print("debug_html:", debug_file)
    print("version:", VERSION)

    if s_all is not None:
        mn, avg, mx = s_all
        print("ALL min", mn, "avg", avg, "max", mx, "count", len(prices_all))

    # 型番別（最安/平均/最大/件数）
    by_model = {}
    for r in filtered:
        m = r["model"] or "unknown"
        p = r["price_yen"]
        if not isinstance(p, int):
            continue
        by_model.setdefault(m, []).append(p)

    for m in sorted(by_model.keys()):
        ps = by_model[m]
        sm = stats_summary(ps)
        if sm is None:
            continue
        mn, avg, mx = sm
        print(m, "min", mn, "avg", avg, "max", mx, "count", len(ps))

if __name__ == "__main__":
    main()

#   python3 janpara_scrape_geforce.py "RTX 12GB" --all-pages
#   python3 janpara_scrape_geforce.py "RTX 8GB" --max-pages 13
#   python3 janpara_scrape_geforce.py "RTX 8GB" --sleep 7 --timeout 40
#
# よく使うオプション:
#   --all-pages         : 1ページ目から最終ページ推定して回す（hard-cap上限あり）
#   --max-pages N       : 最大Nページまで（--all-pagesなしの場合の上限）
#   --sleep SEC         : ページ間待機（既定: 7秒）
#   --timeout SEC       : HTTPタイムアウト（既定: 40秒）
#   --hard-cap N        : --all-pages時の最大上限（既定: 50）
#   --series "20,30,40,50" : 対象RTX世代。空文字("")ならフィルタ無し
#   --debug-save-all    : 各ページHTMLを保存（調査用。ファイルが増える）
#
# 注意:
# - SiteGuard/WAFの影響で 408/429/503/504 が出ることがあるため
#   Cookie保持 + トップ踏み + ヘッダ寄せ + 控えめリトライを入れている。
# - 抽出はリンクテキスト頼みなので、ページ構造変化で0件になることがある。
#   その場合は debug HTML を見て調整する。
#
# 方針:
# - 余計な自動調整や“賢い推測”は入れない（壊れ方を追えることを優先）。
# - 例外は握りつぶさず、ログを残して止まる。

import re
import csv
import time
import sys
import math
import argparse
import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar
from html.parser import HTMLParser

VERSION = "2026-01-25c-geforce"

BASE = "https://www.janpara.co.jp/sale/search/result/"
TOP  = "https://www.janpara.co.jp/"

# --- 解析用 正規表現 ---
PRICE_RE = re.compile(r"中古\s*¥\s*([\d,]+)")
STOCK_RE = re.compile(r"(\d+)個の在庫")

# RTXのみ（GTXは対象外。検索語が混ざっても、ここで拾わない）
MODEL_RE = re.compile(r"RTX\s*?(\d{4})", re.IGNORECASE)

HIT_RE   = re.compile(r"該当件数：\s*(\d+)\s*商品")
PAGE_RE  = re.compile(r"[?&]PAGE=(\d+)")

# --- HTMLから <a href=...> とそのテキストを集める ---
class ATagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in_a = False
        self._href = None
        self._buf = []
        self.links = []  # list[(href, text)]

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self._in_a = True
                self._href = href
                self._buf = []

    def handle_data(self, data):
        if self._in_a and data:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = " ".join("".join(self._buf).split())
            self.links.append((self._href, text))
            self._in_a = False
            self._href = None
            self._buf = []

# --- Cookie保持 opener ---
CJ = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CJ))

COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Connection": "close",
}

_COOKIE_WARMED = False

def log_progress(msg: str):
    """進行ログ（コメント風）"""
    print("# " + msg, flush=True)

def sleep_with_indicator(sec: float, label: str = "sleep"):
    """待機中に動いていることが分かるように '.' を出す"""
    s = int(sec)
    if s <= 0:
        return
    sys.stdout.write(f"# {label}: {s}s ")
    sys.stdout.flush()
    for _ in range(s):
        time.sleep(1)
        sys.stdout.write(".")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

def warm_cookies(timeout_sec: int):
    """トップを1回踏んでCookieを確定させる"""
    global _COOKIE_WARMED
    if _COOKIE_WARMED:
        return
    log_progress("warming cookies (GET /)")
    req = urllib.request.Request(TOP, headers=COMMON_HEADERS, method="GET")
    with OPENER.open(req, timeout=timeout_sec) as resp:
        resp.read()
    _COOKIE_WARMED = True
    log_progress("cookie warm done")

def build_search_url(keywords: str, order: str, page: int) -> str:
    params = {
        "cache_key": "/sale/search/result/",
        "KEYWORDS": keywords,
        "OUTCLSCODE": "",
        "SSHPCODE": "",
        "CHKOUTCOM": "1",
        "ORDER": order,
        "PAGE": str(page),
    }
    return BASE + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)

def fetch_html(url: str, timeout_sec: int, sleep_retry=(0, 2, 5, 9), page_hint: str = "") -> str:
    """HTMLを取る。408/429/503/504 だけ控えめに再試行。"""
    headers = dict(COMMON_HEADERS)
    headers["Referer"] = TOP
    req = urllib.request.Request(url, headers=headers, method="GET")

    last_err = None
    for i, wait in enumerate(sleep_retry, start=1):
        if wait:
            log_progress(f"{page_hint}retry wait {wait}s (attempt {i}/{len(sleep_retry)})")
            sleep_with_indicator(wait, label="retry-sleep")

        try:
            log_progress(f"{page_hint}fetching (attempt {i}/{len(sleep_retry)})")
            with OPENER.open(req, timeout=timeout_sec) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                body = resp.read().decode(charset, errors="replace")
                return body
        except urllib.error.HTTPError as e:
            last_err = e
            log_progress(f"{page_hint}HTTPError {e.code}")
            if e.code in (408, 429, 503, 504):
                continue
            raise
        except Exception as e:
            last_err = e
            log_progress(f"{page_hint}Exception {type(e).__name__}")
            continue

    raise last_err

def parse_hit_count(html: str):
    m = HIT_RE.search(html)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def detect_last_page(html: str):
    """HTML内のリンクから PAGE=数字 を拾い、最大値を最終ページ候補として返す。"""
    p = ATagCollector()
    p.feed(html)
    max_page = None
    for href, _text in p.links:
        m = PAGE_RE.search(href)
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if (max_page is None) or (n > max_page):
            max_page = n
    return max_page

def parse_items(html: str):
    """検索結果から商品相当のリンクテキストを抽出（RTXのみ）"""
    p = ATagCollector()
    p.feed(html)

    rows = []
    for href, text in p.links:
        if "/sale/search/detail/" not in href:
            continue
        if ("中古" not in text) or ("¥" not in text):
            continue

        m_price = PRICE_RE.search(text)
        m_stock = STOCK_RE.search(text)
        m_model = MODEL_RE.search(text)

        price = int(m_price.group(1).replace(",", "")) if m_price else None
        stock = int(m_stock.group(1)) if m_stock else None
        model = m_model.group(1) if m_model else None
        series = model[:2] if model else None  # "20","30","40","50"

        # RTXが取れない＝対象外（GTX等）として捨てる
        if model is None:
            continue

        rows.append({
            "series": series,
            "model": model,
            "price_yen": price,
            "stock": stock,
            "href": href,
            "text": text,
        })
    return rows

def sanitize_out_name(q: str) -> str:
    """例: 'RTX 8GB' -> 'janpara_geforce_rtx_8gb.csv'"""
    s = q.strip().lower().replace("+", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    if not s:
        s = "query"
    return f"janpara_geforce_{s}.csv"

def stats_summary(prices):
    if not prices:
        return None
    mn = min(prices)
    mx = max(prices)
    avg = int(round(sum(prices) / len(prices)))
    return mn, avg, mx

def save_debug_html(page: int, html: str) -> str:
    fn = f"janpara_geforce_page{page:02d}_debug.html"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(html)
    return fn

def main():
    epilog = (
        "例:\n"
        "  python3 janpara_scrape_geforce.py \"RTX 8GB\" --all-pages\n"
        "  python3 janpara_scrape_geforce.py \"RTX 12GB\" --all-pages\n"
        "  python3 janpara_scrape_geforce.py \"RTX 8GB\" --max-pages 13\n"
        "  python3 janpara_scrape_geforce.py \"RTX 8GB\" --series \"30,40,50\" --all-pages\n"
        "\n"
        "メモ:\n"
        "- Amazon等と違い、じゃんぱらはWAFの気分で408/429が出る。\n"
        "- 0件になったら debug HTML を見て、リンク文言/構造変化を疑う。\n"
    )

    ap = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=epilog
    )
    ap.add_argument("keywords", nargs="?", default="RTX 8GB",
                    help='検索語 (例: "RTX 8GB" / "RTX 12GB")')
    ap.add_argument("--series", default="20,30,40,50",
                    help='対象RTX世代 (例: "20,30,40,50") / 空("")ならフィルタ無し')
    ap.add_argument("--max-pages", type=int, default=7,
                    help="巡回する最大ページ数（既定: 7）")
    ap.add_argument("--all-pages", action="store_true",
                    help="1ページ目のリンクから最終ページ推定して回す（hard-capで上限）")
    ap.add_argument("--sleep", type=float, default=7.0,
                    help="ページ間ウェイト秒（既定: 7.0）")
    ap.add_argument("--timeout", type=int, default=40,
                    help="HTTPタイムアウト秒（既定: 40）")
    ap.add_argument("--hard-cap", type=int, default=50,
                    help="--all-pages時の最大上限（既定: 50）")
    ap.add_argument("--debug-save-all", action="store_true",
                    help="各ページHTMLを保存（調査用。ファイルが増える）")
    args = ap.parse_args()

    # series 指定の解釈
    series_list = []
    if args.series.strip():
        series_list = [x.strip() for x in args.series.split(",") if x.strip()]
    series_set = set(series_list)

    log_progress(f"version: {VERSION}")
    warm_cookies(args.timeout)

    # 1ページ目
    url1 = build_search_url(args.keywords, order="3", page=1)
    html1 = fetch_html(url1, timeout_sec=args.timeout, page_hint="page 1/?: ")
    if args.debug_save_all:
        save_debug_html(1, html1)

    hit_count = parse_hit_count(html1)
    rows1 = parse_items(html1)

    if not rows1:
        fn = save_debug_html(1, html1)
        print("query:", args.keywords)
        print("page1: parsed 0 items (debug saved:", fn + ")")
        if hit_count is not None:
            print("hit_count:", hit_count)
        print("version:", VERSION)
        return

    per_page = len(rows1)
    expected_pages = None
    if hit_count is not None and per_page > 0:
        expected_pages = int(math.ceil(hit_count / per_page))

    # 最大ページ決定
    if args.all_pages:
        last = detect_last_page(html1)
        if last is None:
            max_pages = args.max_pages
        else:
            max_pages = min(last, args.hard_cap)
    else:
        max_pages = args.max_pages

    all_rows = []
    pages_ok = 0
    stop_reason = "reached max-pages"
    debug_file = None

    # 1ページ目
    all_rows.extend(rows1)
    pages_ok += 1
    log_progress(f"page 1 parsed {len(rows1)} items")

    # 2ページ目以降
    for page in range(2, max_pages + 1):
        log_progress(f"page {page}/{max_pages}: waiting before fetch")
        sleep_with_indicator(args.sleep, label=f"page-sleep {page}/{max_pages}")

        url = build_search_url(args.keywords, order="3", page=page)
        html = fetch_html(url, timeout_sec=args.timeout, page_hint=f"page {page}/{max_pages}: ")
        if args.debug_save_all:
            save_debug_html(page, html)

        rows = parse_items(html)
        log_progress(f"page {page}/{max_pages} parsed {len(rows)} items (total so far {len(all_rows) + len(rows)})")

        if not rows:
            debug_file = save_debug_html(page, html)
            stop_reason = f"parsed 0 items at page {page} (debug saved)"
            break

        all_rows.extend(rows)
        pages_ok += 1

    # フィルタ（RTX世代）
    if series_list:
        filtered = [r for r in all_rows if r["series"] in series_set]
    else:
        filtered = list(all_rows)

    # CSV出力
    out_csv = sanitize_out_name(args.keywords)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series", "model", "price_yen", "stock", "href", "text"])
        w.writeheader()
        w.writerows(filtered)

    # 統計（全体）
    prices_all = [r["price_yen"] for r in filtered if isinstance(r["price_yen"], int)]
    s_all = stats_summary(prices_all)

    # 結果出力（あなたの形式に寄せる）
    print("query:", args.keywords)
    print("saved:", out_csv, "rows=" + str(len(filtered)))

    # 追加の診断ログ
    if hit_count is not None:
        print("hit_count:", hit_count, "per_page(observed):", per_page)
    if expected_pages is not None:
        print("expected_pages:", expected_pages)
    print("pages fetched:", pages_ok, "/", max_pages, "stop:", stop_reason)
    if debug_file:
        print("debug_html:", debug_file)
    print("version:", VERSION)

    if s_all is not None:
        mn, avg, mx = s_all
        print("ALL min", mn, "avg", avg, "max", mx, "count", len(prices_all))

    # 型番別（最安/平均/最大/件数）
    by_model = {}
    for r in filtered:
        m = r["model"] or "unknown"
        p = r["price_yen"]
        if not isinstance(p, int):
            continue
        by_model.setdefault(m, []).append(p)

    for m in sorted(by_model.keys()):
        ps = by_model[m]
        sm = stats_summary(ps)
        if sm is None:
            continue
        mn, avg, mx = sm
        print(m, "min", mn, "avg", avg, "max", mx, "count", len(ps))

if __name__ == "__main__":
    main()

