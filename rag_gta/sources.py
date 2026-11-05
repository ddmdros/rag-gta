"""Find which wiki article each piece of the lore text came from.

The lore file is a concatenation of GTA Wiki (gta.fandom.com) articles, and each article carries its URL.
An article starts at every line that holds a wiki URL and runs until the next one. Text before the first URL,
or a file with no URL at all, becomes one article called «Registro geral».
"""
import re
from dataclasses import dataclass
from urllib.parse import unquote

WIKI_URL = re.compile(r"https?://gta\.fandom\.com/wiki/[^\s|)\]>\"']+")
GENERAL = "Registro geral"


@dataclass
class Article:
    title: str
    url: str
    text: str


def title_from_url(url):
    """'https://gta.fandom.com/wiki/Carl_Johnson' -> 'Carl Johnson'."""
    slug = url.rstrip("/").rsplit("/wiki/", 1)[-1]
    return unquote(slug).replace("_", " ").strip() or GENERAL


def clean_url(url):
    return url.rstrip(".,;:-")


def split_articles(text):
    """Split the lore text into articles at each line that holds a wiki URL."""
    articles, title, url, buf = [], GENERAL, "", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            articles.append(Article(title, url, body))

    for line in text.splitlines():
        m = WIKI_URL.search(line)
        if m:
            flush()
            url = clean_url(m.group(0))
            title, buf = title_from_url(url), []
            rest = (line[:m.start()] + line[m.end():]).strip(" -|:")
            if rest:
                buf.append(rest)
            continue
        buf.append(line)
    flush()
    return articles


def unique_sources(docs, preview=120):
    """[(title, url, preview)] of the retrieved documents, one per article, in retrieval order."""
    seen, out = set(), []
    for d in docs:
        title = d.metadata.get("title") or GENERAL
        if title in seen:
            continue
        seen.add(title)
        out.append((title, d.metadata.get("source", ""), " ".join(d.page_content.split())[:preview]))
    return out
