"""Build the lore corpus (data/gta_bible.txt) from a fixed list of GTA Wiki articles.

  python -m rag_gta.lore_corpus [--out PATH] [--delay SECONDS]

The text comes from the MediaWiki API of gta.fandom.com, whose content is under CC BY-SA. Each article is
written as its URL on one line followed by its prose, the format that sources.split_articles expects, so every
answer can cite the article it came from. Only paragraphs and section headings are kept: infoboxes, tables,
galleries, navigation boxes and references are dropped. Standard library only.
"""
import argparse, json, sys, time
from html.parser import HTMLParser
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from . import config

API = "https://gta.fandom.com/api.php"
WIKI = "https://gta.fandom.com/wiki/"
USER_AGENT = "rag-gta-corpus-builder/1.0 (+https://github.com/ddmdros/rag-gta)"

# What the lore eval set (eval/lore.jsonl) asks about: the protagonists, the people around them, the cities and
# the games. Fixed on purpose, so anyone can rebuild the same corpus and compare evaluation runs.
ARTICLES = [
    "Grand Theft Auto: San Andreas", "Carl Johnson", "Sweet Johnson", "Big Smoke", "Ryder", "Los Santos",
    "Grand Theft Auto IV", "Niko Bellic", "Roman Bellic", "Liberty City",
    "Grand Theft Auto: Vice City", "Tommy Vercetti", "Vice City",
    "Grand Theft Auto V", "Michael De Santa", "Franklin Clinton", "Trevor Philips",
]

SKIP_TAGS = {"aside", "table", "figure", "sup", "style", "script", "nav", "gallery"}
HEADINGS = {"h2", "h3"}
# Sections after which an article holds no more prose worth indexing.
STOP_SECTIONS = {"references", "see also", "external links", "navigation", "gallery", "videos"}


class _Prose(HTMLParser):
    """Collects paragraphs and h2/h3 headings from rendered wiki HTML, outside infoboxes and tables."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self._buf, self._in, self._skip, self._stopped = [], [], None, 0, False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif not self._skip and tag in HEADINGS | {"p"}:
            self._in, self._buf = tag, []

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == self._in:
            text = " ".join("".join(self._buf).split())
            self._in = None
            if not text or self._stopped:
                return
            if tag in HEADINGS:
                if text.lower() in STOP_SECTIONS:
                    self._stopped = True
                else:
                    self.blocks.append(f"== {text} ==")
            else:
                self.blocks.append(text)

    def handle_data(self, data):
        if self._in and not self._skip:
            self._buf.append(data)


def prose(html):
    """Rendered article HTML -> plain text, one paragraph per line block."""
    parser = _Prose()
    parser.feed(html)
    parser.close()
    blocks = parser.blocks
    while blocks and blocks[-1].startswith("== "):  # a heading with nothing under it
        blocks.pop()
    return "\n\n".join(blocks)


def article_url(title):
    # Parentheses and quotes are percent-encoded so the URL pattern in sources.py keeps the whole link.
    return WIKI + quote(title.replace(" ", "_"), safe=":")


def fetch_html(title):
    """(final title, rendered HTML) of one article, following redirects. None if the page does not exist."""
    params = {"action": "parse", "page": title, "prop": "text", "redirects": 1, "format": "json",
              "formatversion": 2, "disableeditsection": 1, "disablelimitreport": 1, "disabletoc": 1}
    req = Request(f"{API}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    if "error" in data:
        return None
    return data["parse"]["title"], data["parse"]["text"]


def build(titles=ARTICLES, fetch=fetch_html, delay=1.0):
    """(corpus text, report rows). report rows: (asked title, final title or None, characters)."""
    parts, report = [], []
    for i, asked in enumerate(titles):
        if i and delay:
            time.sleep(delay)  # be polite to the wiki
        got = fetch(asked)
        if got is None:
            report.append((asked, None, 0))
            continue
        title, html = got
        text = prose(html)
        report.append((asked, title, len(text)))
        if text:
            parts.append(f"{article_url(title)}\n{text}")
    return "\n\n".join(parts) + "\n", report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag_gta.lore_corpus", description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(config.CORPORA["lore"].path))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    text, report = build(delay=a.delay)
    missing = [asked for asked, title, _ in report if title is None]
    for asked, title, chars in report:
        status = "MISSING" if title is None else f"{chars:>7,} chars" + (f"  (-> {title})" if title != asked else "")
        print(f"{asked:<32} {status}")
    found = len(report) - len(missing)
    if not found:
        print("No article could be fetched.", file=sys.stderr)
        return 1
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n{found}/{len(report)} articles, {len(text):,} characters -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
