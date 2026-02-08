#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
janpara_gpu_scrape.py
=====================
じゃんぱら (janpara.co.jp) の GPU (GeForce / Radeon) 価格スクレイピング＋グラフ生成ツール。
既存の janpara_scrape_geforce.py と janpara_scrape_radeon.py を統合・強化したもの。

[Version 1.2]
  - GeForceとRadeonのVRAM容量別一括調査機能
  - Bot対策強化（ページ遷移10-15秒、容量切り替え20秒待機）
  - グラフ生成機能（ブランド別にPNG出力）
  - ユーザ指定による柔軟なターゲット追加
  - グラフの性能を最適化 (ver1.1)
  - Quadroを除外 (ver1.2)

============================================================
[コマンド例]
  python3 janpara_gpu_scrape.py --scrape
  python3 janpara_gpu_scrape.py --scrape --target geforce
  python3 janpara_gpu_scrape.py --scrape --target radeon --capacity 16GB
  python3 janpara_gpu_scrape.py gauge
  python3 janpara_gpu_scrape.py gauge --target geforce

============================================================
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
import time
import random
import csv
import math
import statistics
import urllib.request
import urllib.parse
import urllib.error
import http.cookiejar
from html.parser import HTMLParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

# ============================================================
# LAYOUT CONFIGURATION SECTION (Graph)
# ============================================================
FIGURE_WIDTH = 12              # グラフ全体の幅,default,12
FIGURE_HEIGHT_BASE = 4.0       # 最小の高さ,default,4.0
FIGURE_HEIGHT_PER_ROW = 0.65   # 1行あたりの追加高さ,default,0.65
FIGURE_HEIGHT_OFFSET = 1.2     # 高さ計算のオフセット,default,1.2
FIGURE_DPI = 100               # 解像度(DPI),default,120

MARGIN_TOP = 0.90              # 上マージン(0.0-1.0),default,0.90
MARGIN_BOTTOM = 0.05           # 下マージン(0.0-1.0),default,0.05
MARGIN_RIGHT = 0.98           # 右マージン(0.0-1.0),default,0.995
MARGIN_LEFT = 0.12             # 左マージン(0.0-1.0),default,0.05

BAR_HEIGHT = 0.62              # バーの高さ (MAX),default,0.62
BAR_AVG_RATIO = 0.70            # AVGバーの高さ比率(MAXバーに対する),default,0.7
BAR_MIN_RATIO = 0.40           # MINバーの高さ比率(MAXバーに対する),default,0.45

# --- Bar Color Settings ---
COLOR_MAX = "#FF6666"          # MAX価格の色(赤系),default,#FF6666
COLOR_AVG = "#66CC66"          # AVG価格の色(緑系),default,#66CC66
COLOR_MIN = "#6666FF"          # MIN価格の色(青系),default,#6666FF

# --- Bar Alpha (Transparency) Settings ---
ALPHA_MAX = 0.38                # MAX価格の透明度(0.0-1.0),default,0.8
ALPHA_AVG = 0.39                # AVG価格の透明度(0.0-1.0),default,0.9
ALPHA_MIN = 0.40                # MIN価格の透明度(0.0-1.0),default,1.0

# --- X-Axis Settings ---
X_AXIS_MAX_BAR_POSITION = 0.92 # 最大バーがX軸上で表示される位置(0.0-1.0)
                               # 例: 0.60 = グラフ幅の60%位置に最大値が来る,default,0.60

# --- Text Area Layout Settings ---
TEXT_START_MULTIPLIER = 0.10   # テキスト開始位置の倍率(global_max * この値),default,1.00
TEXT_START_MIN_RATIO = 0.10    # テキスト開始位置の最小比率(xlim_right * この値),default,0.60

# --- Column Width Settings (Text Area) ---
# 各列の幅の比率。合計値で正規化されます。
COLUMN_WIDTH_MIN = 1.1         # MIN列の幅,default,1.2
COLUMN_WIDTH_AVG = 1.1         # AVG列の幅,default,1.2
COLUMN_WIDTH_MAX = 1.1         # MAX列の幅,default,1.2
COLUMN_WIDTH_CNT = 0.3         # Count列の幅(他より狭く設定),default,0.5

# --- Title Settings ---
TITLE_FONTSIZE = 16            # タイトルのフォントサイズ,default,16
TITLE_PAD = 20                 # タイトルと図の間隔,default,20

# --- Legend Settings ---
LEGEND_LOCATION = "lower right"        # 凡例の位置,default,
LEGEND_BBOX_X = 1.00                   # 凡例のX位置(bbox_to_anchor),default,1.00
LEGEND_BBOX_Y = 1.01                   # 凡例のY位置(bbox_to_anchor),default,1.01
LEGEND_NCOL = 3                        # 凡例の列数,default,3

# --- Header Row Settings (Column Labels) ---
LEGEND_FONTSIZE = 10                   # 凡例のフォントサイズ,default,10
HEADER_Y_POSITION = -0.70       # ヘッダー行のY位置(負の値で上に配置),default,-0.70
HEADER_FONTSIZE = 10            # ヘッダーのフォントサイズ,default,10
DATA_TEXT_FONTSIZE = 10         # データテキストのフォントサイズ,default,11
DATA_COUNT_FONTSIZE = 11        # Count列のフォントサイズ,default,11
Y_LABEL_FONTSIZE = 12           # Y軸ラベルのフォントサイズ,default,11

# ============================================================
# Global Settings
# ============================================================
VERSION = "1.2"

# デフォルトの調査対象
DEFAULT_TARGETS = {
    "GeForce": ["8GB", "10GB", "11GB", "12GB", "16GB", "32GB"],
    "Radeon": ["8GB", "12GB", "16GB", "24GB"],
}

BASE_URL = "https://www.janpara.co.jp/sale/search/result/"
TOP_URL = "https://www.janpara.co.jp/"

# --- 解析用 正規表現 ---
# 価格は "中古" の後に任意の文字（空白や全角記号など）を挟んで "¥" が来る
PRICE_RE = re.compile(r"中古.*?¥\s*([\d,]+)")
STOCK_RE = re.compile(r"(\d+)個の在庫")
HIT_RE   = re.compile(r"該当件数：\s*(\d+)\s*商品")
PAGE_RE  = re.compile(r"[?&]PAGE=(\d+)")

# RTXのみ（GTXは対象外）
RTX_MODEL_RE = re.compile(r"RTX\s*(\d{4})", re.IGNORECASE)

# Radeon抽出（RX系、Vega系）
RX_RE = re.compile(r"\bRX\s*(\d{3,4})\s*(XTX|XT)?\b", re.IGNORECASE)
VEGA_RE = re.compile(r"\b(RADEON\s*)?RX\s*VEGA\s*(56|64)\b", re.IGNORECASE)

# Radeon除外キーワード
RADEON_EXCLUDE_RE = re.compile(r"\b(Ryzen|iMac|Apple)\b", re.IGNORECASE)

# ============================================================
# HTTP & Utils
# ============================================================
CJ = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CJ))

COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Connection": "close",
    "Referer": TOP_URL,
}

_COOKIE_WARMED = False

def log_progress(msg: str):
    print("# " + msg, flush=True)

def sleep_with_indicator(sec: float, label: str = "sleep"):
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

def sleep_random(min_sec: float, max_sec: float, label: str = "sleep"):
    """min_sec 〜 max_sec の間でランダムに待機"""
    wait = random.uniform(min_sec, max_sec)
    sleep_with_indicator(wait, label)

def warm_cookies(timeout_sec: int):
    global _COOKIE_WARMED
    if _COOKIE_WARMED:
        return
    log_progress("warming cookies (GET /)")
    req = urllib.request.Request(TOP_URL, headers=COMMON_HEADERS, method="GET")
    try:
        with OPENER.open(req, timeout=timeout_sec) as resp:
            resp.read()
        _COOKIE_WARMED = True
        log_progress("cookie warm done")
    except Exception as e:
        log_progress(f"cookie warming failed: {e}")

def fetch_html(url: str, timeout_sec: int, sleep_retry=(2, 5, 10), page_hint: str = "") -> str:
    headers = dict(COMMON_HEADERS)
    headers["Referer"] = TOP_URL
    req = urllib.request.Request(url, headers=headers, method="GET")

    last_err = None
    for i, wait in enumerate(sleep_retry, start=0):
        if i > 0:
            log_progress(f"{page_hint}retry wait {wait}s (attempt {i}/{len(sleep_retry)})")
            sleep_with_indicator(wait, label="retry-sleep")

        try:
            # log_progress(f"{page_hint}fetching...")
            with OPENER.open(req, timeout=timeout_sec) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                body = resp.read().decode(charset, errors="replace")
                return body
        except urllib.error.HTTPError as e:
            last_err = e
            log_progress(f"{page_hint}HTTPError {e.code}")
            try:
                err_body = e.read().decode("utf-8", errors="replace")
                # log_progress(f"Error Body: {err_body[:1000]}")
            except Exception:
                pass

            if e.code in (408, 429, 503, 504):
                continue
            raise
        except Exception as e:
            last_err = e
            log_progress(f"{page_hint}Exception {type(e).__name__}")
            continue

    raise last_err

# ============================================================
# HTML Parsing Classes
# ============================================================
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

def parse_hit_count(html: str):
    m = HIT_RE.search(html)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def detect_last_page(html: str):
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

def normalize_radeon_model(text: str):
    m = VEGA_RE.search(text)
    if m:
        num = m.group(2)
        return f"RX VEGA {num}"

    m = RX_RE.search(text)
    if m:
        base = m.group(1)
        suf = m.group(2) or ""
        return f"RX{base}{suf.upper()}"
    return None

# ============================================================
# Scraping Logic
# ============================================================
@dataclass
class ScrapeResult:
    model: str
    price_yen: int
    stock: int
    href: str
    text: str

class GPUScraper:
    def __init__(self, brand: str, capacity: str, timeout: int = 40):
        self.brand = brand
        self.capacity = capacity
        self.timeout = timeout

    def get_search_keywords(self) -> str:
        if self.brand.lower() == "geforce":
            return f"RTX {self.capacity}"
        elif self.brand.lower() == "radeon":
            # Radeonの場合は "Radeon 8GB" のように検索して後でRXを絞り込む
            return f"Radeon {self.capacity}"
        else:
            return f"{self.brand} {self.capacity}"

    def build_search_url(self, page: int) -> str:
        keywords = self.get_search_keywords()
        params = {
            "KEYWORDS": keywords,
            "OUTCLSCODE": "",
            "ORDER": "2",
            "PAGE": str(page),
        }
        url = BASE_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
        log_progress(f"Built URL: {url}")
        return url

    def parse_items(self, html: str) -> List[ScrapeResult]:
        p = ATagCollector()
        p.feed(html)

        results = []
        for href, text in p.links:
            if "/sale/search/detail/" not in href:
                continue

            if ("中古" not in text) or ("¥" not in text):
                continue

            # 価格・在庫抽出
            m_price = PRICE_RE.search(text)
            m_stock = STOCK_RE.search(text)
            price = int(m_price.group(1).replace(",", "")) if m_price else None
            stock = int(m_stock.group(1)) if m_stock else None

            if price is None:
                continue

            model = None
            if self.brand.lower() == "geforce":
                # GeForce (RTXのみ)
                m_model = RTX_MODEL_RE.search(text)

                # 予備の手動抽出: 正規表現が効かないケースへのフェイルセーフ
                if not m_model and "RTX" in text:
                     parts = text.upper().split("RTX")
                     for part in parts[1:]:
                         m_num = re.match(r"^\s*(\d{4})", part)
                         if m_num:
                             model = m_num.group(1)
                             break

                if model:
                     model = f"RTX{model}"
                elif m_model:
                    model = m_model.group(1)
                    model = f"RTX{model}"

                # Quadroを除外
                if re.search(r"Quadro", text, re.IGNORECASE):
                    model = None

            elif self.brand.lower() == "radeon":
                # Radeon (RX/Vegaのみ, Ryzen/Apple除外)
                if RADEON_EXCLUDE_RE.search(text):
                    continue
                model = normalize_radeon_model(text)

            else:
                # その他 (汎用)
                model = text # 仮

            if model:
                results.append(ScrapeResult(
                    model=model,
                    price_yen=price,
                    stock=stock,
                    href=href,
                    text=text
                ))
        return results

def sanitize_filename(brand: str, capacity: str) -> str:
    s = f"{brand}_{capacity}".strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return f"janpara_{s}.csv"

def run_scrape(targets: Dict[str, List[str]], args):
    warm_cookies(args.timeout)

    total_targets = sum(len(caps) for caps in targets.values())
    processed_count = 0

    for brand, capacities in targets.items():
        for i, cap in enumerate(capacities):
            processed_count += 1
            log_progress(f"=== Target ({processed_count}/{total_targets}): {brand} {cap} ===")

            scraper = GPUScraper(brand, cap, args.timeout)

            # 1ページ目
            url1 = scraper.build_search_url(page=1)
            try:
                html1 = fetch_html(url1, args.timeout, page_hint="page 1: ")
            except Exception as e:
                log_progress(f"Failed to fetch page 1: {e}")
                continue

            hit_count = parse_hit_count(html1)
            rows = scraper.parse_items(html1)

            max_pages = 1
            if args.all_pages:
                last = detect_last_page(html1)
                if last:
                    max_pages = min(last, args.hard_cap)
            else:
                max_pages = args.max_pages

            all_results = []
            all_results.extend(rows)
            log_progress(f"page 1: parsed {len(rows)} items (Hits: {hit_count})")

            # 2ページ目以降
            for page in range(2, max_pages + 1):
                # ページ間ウェイト (10-15秒)
                sleep_random(10.0, 15.0, label=f"page-sleep {page}/{max_pages}")

                url = scraper.build_search_url(page)
                try:
                    html = fetch_html(url, args.timeout, page_hint=f"page {page}: ")
                    page_rows = scraper.parse_items(html)
                    if not page_rows:
                        log_progress(f"page {page}: no items found, stopping")
                        break
                    all_results.extend(page_rows)
                    log_progress(f"page {page}: parsed {len(page_rows)} items")
                except Exception as e:
                    log_progress(f"Failed to fetch page {page}: {e}")
                    break

            # 保存
            # ディレクトリは run_scrape 呼び出し前に作成済み
            out_csv = args.base_dir / sanitize_filename(brand, cap)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["model", "price_yen", "stock", "href", "text"])
                w.writeheader()
                for r in all_results:
                    w.writerow({
                        "model": r.model,
                        "price_yen": r.price_yen,
                        "stock": r.stock,
                        "href": r.href,
                        "text": r.text
                    })
            log_progress(f"Saved {len(all_results)} items to {out_csv}")

            # 次のターゲットへのウェイト (最後以外)
            if processed_count < total_targets:
                log_progress("Waiting 20s before next target...")
                sleep_with_indicator(20.0, "target-sleep")

# ============================================================
# Statistics & Graph
# ============================================================
def extract_model_number(model: str) -> int:
    m = re.search(r"(\d+)", model)
    if m:
        return int(m.group(1))
    return 0

def extract_capacity_number(capacity: str) -> int:
    m = re.search(r"(\d+)", capacity)
    if m:
        return int(m.group(1))
    return 0

@dataclass
class StatRow:
    brand: str
    capacity: str
    model: str # 代表モデル名 (例: RTX3080, RX6800XT)
    label: str # グラフ表示用ラベル
    min_price: int
    avg_price: int
    max_price: int
    count: int

    @property
    def sort_key(self):
        # グラフでの並び順: ブランド -> 容量(数値) -> モデル名(数値)
        return (self.brand, extract_model_number(self.model), extract_capacity_number(self.capacity))

def load_stats_from_dir(data_dir: Path, target_brand: str = None) -> List[StatRow]:
    rows = []
    # janpara_{brand}_{capacity}.csv を探す
    # ただし、モデル別集計を行う必要がある

    for csv_path in sorted(data_dir.glob("janpara_*.csv")):
        stem = csv_path.stem # janpara_geforce_8gb
        parts = stem.split("_")
        if len(parts) < 3:
            continue

        brand = parts[1] # geforce
        capacity = parts[2] # 8gb

        if target_brand and brand.lower() != target_brand.lower():
            continue

        # CSV読み込み
        prices_by_model = {} # model -> list[price]
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    m = r.get("model")
                    p_str = r.get("price_yen")
                    if m and p_str:
                        try:
                            p = int(p_str)
                            prices_by_model.setdefault(m, []).append(p)
                        except ValueError:
                            pass
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        # 集計
        for model, prices in prices_by_model.items():
            if not prices:
                continue
            rows.append(StatRow(
                brand=brand.title(),
                capacity=capacity.upper(),
                model=model,
                label=f"{model}, {capacity.upper()}",
                min_price=min(prices),
                avg_price=int(statistics.mean(prices)),
                max_price=max(prices),
                count=len(prices)
            ))

    # ソート: モデル名でソートしておく
    rows.sort(key=lambda x: x.sort_key)
    return rows

def plot_price_gauge(rows: List[StatRow], out_path: Path, title: str, show: bool = False):
    if not rows:
        return
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    labels = [r.label for r in rows]
    mins = [r.min_price for r in rows]
    avgs = [r.avg_price for r in rows]
    maxs = [r.max_price for r in rows]

    global_max = max(maxs) if maxs else 10000
    xlim_right = global_max / X_AXIS_MAX_BAR_POSITION

    # Layout calc
    col_start_x = global_max * TEXT_START_MULTIPLIER
    if col_start_x < xlim_right * TEXT_START_MIN_RATIO:
        col_start_x = xlim_right * TEXT_START_MIN_RATIO

    text_width = xlim_right - col_start_x
    total_w = COLUMN_WIDTH_MIN + COLUMN_WIDTH_AVG + COLUMN_WIDTH_MAX + COLUMN_WIDTH_CNT
    w_unit = text_width / total_w

    x_min_txt = col_start_x + w_unit * COLUMN_WIDTH_MIN
    x_avg_txt = x_min_txt + w_unit * COLUMN_WIDTH_AVG
    x_max_txt = x_avg_txt + w_unit * COLUMN_WIDTH_MAX
    x_cnt_txt = x_max_txt + w_unit * COLUMN_WIDTH_CNT

    fig_h = max(FIGURE_HEIGHT_BASE, FIGURE_HEIGHT_PER_ROW * len(rows) + FIGURE_HEIGHT_OFFSET)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, fig_h), dpi=FIGURE_DPI)

    plt.subplots_adjust(top=MARGIN_TOP, bottom=MARGIN_BOTTOM, right=MARGIN_RIGHT, left=MARGIN_LEFT)

    y_pos = list(range(len(rows)))

    ax.barh(y_pos, maxs, color=COLOR_MAX, alpha=ALPHA_MAX, height=BAR_HEIGHT, label="MAX")
    ax.barh(y_pos, avgs, color=COLOR_AVG, alpha=ALPHA_AVG, height=BAR_HEIGHT*BAR_AVG_RATIO, label="AVG")
    ax.barh(y_pos, mins, color=COLOR_MIN, alpha=ALPHA_MIN, height=BAR_HEIGHT*BAR_MIN_RATIO, label="MIN")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=Y_LABEL_FONTSIZE, fontweight='bold')
    ax.invert_yaxis()

    ax.set_xlim(0, xlim_right)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax.grid(axis='x', linestyle=':', alpha=0.5)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=TITLE_PAD)

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc=LEGEND_LOCATION, bbox_to_anchor=(LEGEND_BBOX_X, LEGEND_BBOX_Y), ncol=LEGEND_NCOL, frameon=False, fontsize=LEGEND_FONTSIZE)

    # Header
    head_y = HEADER_Y_POSITION
    ax.text(x_min_txt, head_y, "MIN", color='blue', fontweight='bold', ha='right')
    ax.text(x_avg_txt, head_y, "AVG", color='green', fontweight='bold', ha='right')
    ax.text(x_max_txt, head_y, "MAX", color='red', fontweight='bold', ha='right')
    ax.text(x_cnt_txt, head_y, "Cnt", color='black', fontweight='bold', ha='right')

    def fmt_val(v): return f"{v:,}"

    for i, r in enumerate(rows):
        ax.text(x_min_txt, i, fmt_val(r.min_price), va='center', ha='right', fontsize=DATA_TEXT_FONTSIZE, fontfamily='monospace')
        ax.text(x_avg_txt, i, fmt_val(r.avg_price), va='center', ha='right', fontsize=DATA_TEXT_FONTSIZE, fontfamily='monospace')
        ax.text(x_max_txt, i, fmt_val(r.max_price), va='center', ha='right', fontsize=DATA_TEXT_FONTSIZE, fontfamily='monospace')
        ax.text(x_cnt_txt, i, str(r.count), va='center', ha='right', fontsize=DATA_COUNT_FONTSIZE)

    plt.savefig(out_path)
    if show: plt.show()
    plt.close()

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=f"Janpara GPU Scraper v{VERSION}")

    # Argparse config to support multiple modes
    parser.add_argument("mode", nargs="?", choices=["gauge"], help="Mode: gauge (or omit for scrape/help)")

    parser.add_argument("--scrape", action="store_true", help="Execute scraping")

    parser.add_argument("--target", choices=["geforce", "radeon", "all"], default="all", help="Target brand")
    parser.add_argument("--capacity", help="Specific capacity (e.g. 16GB) to scrape/graph")

    parser.add_argument("--add-target", action="append", help="Add custom target (format: 'Brand:Capacity', e.g. 'GeForce:4090')")

    parser.add_argument("--date", default=_dt.date.today().strftime("%Y-%m-%d"), help="Target date YYYY-MM-DD")
    parser.add_argument("--base-dir", default=".", help="Base directory")

    parser.add_argument("--max-pages", type=int, default=7, help="Max pages to scrape")
    parser.add_argument("--all-pages", action="store_true", help="Scrape all pages")
    parser.add_argument("--hard-cap", type=int, default=50, help="Hard cap for all-pages")
    parser.add_argument("--timeout", type=int, default=40, help="HTTP timeout")

    parser.add_argument("--show", action="store_true", help="Show graph GUI")

    args = parser.parse_args()

    # Mode determination logic
    run_mode = "help"
    if args.scrape:
        run_mode = "scrape"
    elif args.mode == "gauge":
        run_mode = "gauge"

    if run_mode == "help":
        parser.print_help()
        return 0

    base_path = Path(args.base_dir).resolve()
    data_dir = base_path / f"janpara_gpu_{args.date}"

    # --- Prepare Targets ---
    targets = {}

    # Default targets
    if args.target == "all":
        targets = {k: v[:] for k, v in DEFAULT_TARGETS.items()}
    elif args.target == "geforce":
        targets["GeForce"] = DEFAULT_TARGETS["GeForce"][:]
    elif args.target == "radeon":
        targets["Radeon"] = DEFAULT_TARGETS["Radeon"][:]

    # Capacity filter (if specified, filter defaults)
    if args.capacity:
        for brand in list(targets.keys()):
            # Only keep specified capacity
            targets[brand] = [args.capacity]

    # Add custom targets
    if args.add_target:
        for t in args.add_target:
            parts = t.split(":")
            if len(parts) == 2:
                b, c = parts
                targets.setdefault(b, []).append(c)
            else:
                print(f"Invalid format for --add-target: {t}")

    # Remove empty brands
    targets = {k: v for k, v in targets.items() if v}

    # --- Execute Scrape ---
    if run_mode == "scrape":
        if not targets:
            print("No targets specified.")
            return 1

        data_dir.mkdir(parents=True, exist_ok=True)
        # Pass data_dir to run_scrape via args object injection or modifying function
        args.base_dir = data_dir # Override base_dir for scraping function to point to date dir

        print(f"=== Scraping Start [{args.date}] (v{VERSION}) ===")
        print(f"Targets: {targets}")
        run_scrape(targets, args)
        print("=== Scraping Finished ===")

    # --- Execute Gauge ---
    if run_mode == "gauge" or args.scrape:
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            return 1

        print(f"=== Generating Graph [{args.date}] ===")

        # GeForce
        if "GeForce" in targets or args.target in ["all", "geforce"]:
            rows = load_stats_from_dir(data_dir, "geforce")
            if rows:
                out_png = base_path / f"janpara_geforce_{args.date}.png"
                plot_price_gauge(rows, out_png, f"GeForce Price Gauge {args.date} (Janpara)", args.show)
                print(f"Saved: {out_png}")
            else:
                print("No GeForce data found (or filtered out).")

        # Radeon
        if "Radeon" in targets or args.target in ["all", "radeon"]:
            rows = load_stats_from_dir(data_dir, "radeon")
            if rows:
                out_png = base_path / f"janpara_radeon_{args.date}.png"
                plot_price_gauge(rows, out_png, f"Radeon Price Gauge {args.date} (Janpara)", args.show)
                print(f"Saved: {out_png}")
            else:
                print("No Radeon data found (or filtered out).")

    return 0

if __name__ == "__main__":
    sys.exit(main())
