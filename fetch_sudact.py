"""
Загрузчик судебных решений.
Поддерживаемые источники:
  sudact  — sudact.ru (арбитражные суды, HTML-парсинг)
  gas     — ГАС Правосудие bsr.sudrf.ru (суды общей юрисдикции, JSON API)

Запуск из корня проекта:
    python fetch_sudact.py                        # интерактивное меню
    python fetch_sudact.py --source sudact --n 200 --query "налог НДС"
    python fetch_sudact.py --source gas    --n 200 --query "налог НДФЛ"

Результат: data/<source>/raw/docs.jsonl
"""

import argparse
import json
import re
import time
import uuid
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Установи зависимости: pip install requests beautifulsoup4")
    raise SystemExit(1)

# ── Общие настройки ────────────────────────────────────────────────────────────

DELAY     = 2.0
PAGE_SIZE = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)


def _get(url: str, params: dict = None, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = DELAY * (attempt + 1)
            print(f"  [retry {attempt + 1}/{retries}] {e} — ждём {wait:.0f}с")
            time.sleep(wait)
    return None


def _post(url: str, payload: dict, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = session.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = DELAY * (attempt + 1)
            print(f"  [retry {attempt + 1}/{retries}] {e} — ждём {wait:.0f}с")
            time.sleep(wait)
    return None


def _clean_text(raw: str) -> str:
    raw = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", raw)
    return re.sub(r"\s{2,}", " ", raw).strip()


# ══════════════════════════════════════════════════════════════════════════════
# ИСТОЧНИК 1: sudact.ru
# ══════════════════════════════════════════════════════════════════════════════

_SUDACT_BASE      = "https://sudact.ru"
_SUDACT_SEARCH    = f"{_SUDACT_BASE}/arbitral/doc/"
_SUDACT_NOISE     = [
    "script", "style", "nav", "header", "footer",
    ".sidebar", ".breadcrumb", ".doc-navigation",
    ".doc-info", ".act-meta", ".act-controls",
]


def _sudact_parse_search(html: str, debug: bool = False) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    if debug:
        # Печатаем все ссылки на странице чтобы найти нужный паттерн
        all_links = [(a.get("href", ""), a.get_text(strip=True)[:60]) for a in soup.find_all("a")]
        print(f"  [debug] всего ссылок на странице: {len(all_links)}")
        for href, text in all_links[:30]:
            print(f"    {href!r:60s}  {text}")

    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Ищем любые ссылки на документы арбитражных судов
        if re.search(r"/(arbitral|regular|vlasimov)/doc/", href):
            if not href.startswith("http"):
                href = _SUDACT_BASE + href
            urls.append(href)
    return list(dict.fromkeys(urls))  # убираем дубли, сохраняем порядок


def _sudact_parse_doc(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    for sel in _SUDACT_NOISE:
        for tag in soup.select(sel):
            tag.decompose()

    body = (
        soup.select_one(".act-text")
        or soup.select_one("#documentText")
        or soup.select_one(".doc-text")
        or soup.select_one("article")
        or soup.select_one("main")
    )
    if body is None:
        return None

    text = _clean_text(body.get_text(separator=" ", strip=True))
    if len(text) < 300:
        return None

    parts  = url.rstrip("/").split("/")
    doc_id = next((p for p in reversed(parts) if p.isdigit()), None) or re.sub(r"[^\w]", "_", url[-40:])
    return {"doc_id": f"sudact_{doc_id}", "text": text, "url": url}


def fetch_sudact(query: str, n: int, out_file: Path, existing_ids: set,
                 debug: bool = False) -> list[dict]:
    docs = []
    page = 1
    seen: set[str] = set()
    empty_pages = 0

    while len(docs) < n:
        params = {
            "page":         page,
            "count":        PAGE_SIZE,
            "arbitral-txt": query,
        }
        print(f"  sudact: страница {page} (найдено {len(docs)}/{n})...")
        resp = _get(_SUDACT_SEARCH, params=params)
        if resp is None:
            break

        if debug and page == 1:
            Path("debug_sudact_search.html").write_text(resp.text, encoding="utf-8")
            print("  [debug] первая страница сохранена → debug_sudact_search.html")

        urls = _sudact_parse_search(resp.text, debug=(debug and page == 1))

        if not urls:
            empty_pages += 1
            print(f"  Нет ссылок на документы (пустых страниц подряд: {empty_pages})")
            if empty_pages >= 3:
                print("  Останавливаемся — сайт не возвращает результаты.")
                break
            page += 1
            time.sleep(DELAY)
            continue

        empty_pages = 0
        for url in urls:
            if url in seen:
                continue
            seen.add(url)

            r = _get(url)
            if r is None:
                continue

            doc = _sudact_parse_doc(r.text, url)
            if doc is None or doc["doc_id"] in existing_ids:
                continue

            existing_ids.add(doc["doc_id"])
            docs.append(doc)
            print(f"    [{len(docs):3d}/{n}] {doc['doc_id']}  {len(doc['text'])} симв.")
            time.sleep(DELAY)

            if len(docs) >= n:
                break

        page += 1
        time.sleep(DELAY)

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# ИСТОЧНИК 2: ГАС Правосудие (bsr.sudrf.ru) — JSON API
# ══════════════════════════════════════════════════════════════════════════════

_GAS_SEARCH = "https://bsr.sudrf.ru/bigs/s.action"
_GAS_DOC    = "https://bsr.sudrf.ru/bigs/ui.action"


def _gas_search_payload(query: str, start: int) -> dict:
    return {
        "type": "MULTIPLEQUERY",
        "multiqueryRequest": {
            "queryRequests": [{
                "type": "Q",
                "queryString": query,
                "fieldRequests": [{
                    "field":    "fullText",
                    "values":   [query],
                    "operator": "CONTAINS",
                }],
            }],
        },
        "start":      start,
        "rows":       PAGE_SIZE,
        "uid":        str(uuid.uuid4()),
        "senderType": "MULTIQUERY",
    }


def _gas_fetch_doc_text(doc_id: str) -> str | None:
    resp = _get(_GAS_DOC, params={"id": doc_id})
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup.select("script, style, nav, header, footer, .sidebar"):
        tag.decompose()

    body = (
        soup.select_one("#documentBody")
        or soup.select_one(".decision-text")
        or soup.select_one(".act-text")
        or soup.select_one("article")
        or soup.select_one("main")
    )
    if body is None:
        return None

    return _clean_text(body.get_text(separator=" ", strip=True))


def fetch_gas(query: str, n: int, out_file: Path, existing_ids: set) -> list[dict]:
    docs  = []
    start = 0

    gas_session = requests.Session()
    gas_session.headers.update({
        **HEADERS,
        "Content-Type": "application/json",
        "Referer":      "https://bsr.sudrf.ru/",
        "Origin":       "https://bsr.sudrf.ru",
    })

    while len(docs) < n:
        print(f"  ГАС: записи {start}–{start + PAGE_SIZE - 1} (найдено {len(docs)}/{n})...")
        try:
            resp = gas_session.post(
                _GAS_SEARCH,
                json=_gas_search_payload(query, start),
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [ошибка поиска] {e}")
            break

        items = data.get("response", {}).get("docs", [])
        if not items:
            print("  Результаты кончились.")
            break

        for item in items:
            doc_id = str(item.get("id") or item.get("uid") or "")
            if not doc_id or f"gas_{doc_id}" in existing_ids:
                continue

            text = item.get("fullText") or item.get("text") or ""
            if not text:
                # Полный текст не в ответе — запрашиваем страницу документа
                text = _gas_fetch_doc_text(doc_id) or ""
                time.sleep(DELAY)

            text = _clean_text(text)
            if len(text) < 300:
                continue

            full_id = f"gas_{doc_id}"
            existing_ids.add(full_id)
            doc = {
                "doc_id": full_id,
                "text":   text,
                "url":    f"{_GAS_DOC}?id={doc_id}",
            }
            docs.append(doc)
            print(f"    [{len(docs):3d}/{n}] {full_id}  {len(text)} симв.")

            if len(docs) >= n:
                break

        start += PAGE_SIZE
        time.sleep(DELAY)

    return docs


# ══════════════════════════════════════════════════════════════════════════════
# Общий I/O и main
# ══════════════════════════════════════════════════════════════════════════════

def _load_existing(out_file: Path) -> tuple[list[dict], set[str]]:
    docs, ids = [], set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                doc = json.loads(line)
                ids.add(doc["doc_id"])
                docs.append(doc)
    return docs, ids


def _save_new(docs: list[dict], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


SOURCES = {
    "sudact": ("СудАкт (sudact.ru — арбитражные суды)",        fetch_sudact),
    "gas":    ("ГАС Правосудие (bsr.sudrf.ru — общая юрисдикция)", fetch_gas),
}


def interactive_setup(args) -> tuple[str, str, int]:
    print("\n" + "=" * 52)
    print("   Загрузчик судебных решений")
    print("=" * 52)

    if args.source:
        source = args.source
    else:
        print("\nИсточник:")
        keys = list(SOURCES.keys())
        for i, (k, (label, _)) in enumerate(SOURCES.items(), 1):
            print(f"  {i}. {label}")
        while True:
            raw = input("Выбор [1]: ").strip() or "1"
            if raw.isdigit() and 1 <= int(raw) <= len(keys):
                source = keys[int(raw) - 1]
                break

    query = args.query or input("\nПоисковый запрос [налог НДС НДФЛ]: ").strip() or "налог НДС НДФЛ"
    n     = args.n     or int(input("Количество документов [100]: ").strip() or "100")
    return source, query, n


def main():
    parser = argparse.ArgumentParser(
        description="Загрузчик судебных решений (sudact.ru / ГАС Правосудие)",
    )
    parser.add_argument("--source", choices=list(SOURCES.keys()), default=None)
    parser.add_argument("--query",  default=None,  help="Поисковый запрос")
    parser.add_argument("--n",      type=int, default=None, help="Количество документов")
    parser.add_argument("--debug",  action="store_true",
                        help="Сохранить первую страницу поиска в debug_*.html и напечатать все ссылки")
    args = parser.parse_args()

    if not all([args.source, args.query, args.n]):
        source, query, n = interactive_setup(args)
    else:
        source, query, n = args.source, args.query, args.n

    out_file = Path(f"data/{source}/raw/docs.jsonl")
    existing_docs, existing_ids = _load_existing(out_file)

    if existing_docs:
        print(f"Уже скачано: {len(existing_docs)} документов")

    need = n - len(existing_docs)
    if need <= 0:
        print(f"Уже есть {len(existing_docs)} ≥ {n} документов, ничего не делаем.")
        return

    label, fetch_fn = SOURCES[source]
    print(f"\nИсточник: {label}")
    print(f"Запрос:   '{query}'")
    print(f"Нужно ещё: {need} документов\n")

    kwargs = {"debug": args.debug} if source == "sudact" else {}
    new_docs = fetch_fn(query, need, out_file, existing_ids, **kwargs)
    _save_new(new_docs, out_file)

    print(f"\nГотово: скачано {len(new_docs)} новых документов → {out_file}")
    print(f"Итого в файле: {len(existing_docs) + len(new_docs)}")


if __name__ == "__main__":
    main()
