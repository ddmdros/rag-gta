"""Turn the raw files into LangChain documents.

- Glossary (CSV): one document per term. The file starts with a preamble (title, license, usage notes) and
  only then the header row `English, Definition, Brazilian Portuguese, Usage and Notes`; the preamble is skipped.
  Terms are short, so they are not split: a chunk that cuts a term from its translation is useless.
- Lore (TXT): split into overlapping chunks. Each chunk keeps the wiki article it came from (title and URL) in
  its metadata, so answers can cite their sources.
"""
import csv
from pathlib import Path

from langchain_core.documents import Document

from . import sources

GLOSSARY_HEADER = "english"


def read_text(path):
    """Text of a file in UTF-8, falling back to Latin-1 (files saved by older Windows tools)."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def glossary_rows(path):
    """(english, definition, portuguese, notes) for each term, after the header row."""
    rows = list(csv.reader(read_text(path).splitlines(keepends=True)))
    start = next((i for i, r in enumerate(rows) if r and r[0].strip().lower() == GLOSSARY_HEADER), None)
    if start is None:
        raise ValueError(f"{path}: header row starting with 'English' not found")
    out = []
    for r in rows[start + 1:]:
        r = [c.strip() for c in r] + [""] * (4 - len(r))
        en, definition, pt, notes = r[:4]
        if en and pt:
            out.append((en, definition, pt.rstrip(", "), notes))
    return out


def load_glossary(path):
    docs = []
    for en, definition, pt, notes in glossary_rows(path):
        lines = [f"Termo em inglês: {en}", f"Tradução em português: {pt}"]
        if definition:
            lines.append(f"Definição: {definition}")
        if notes:
            lines.append(f"Notas de uso: {notes}")
        docs.append(Document(page_content="\n".join(lines),
                             metadata={"title": en, "source": "Video Games Localization Glossary (CC BY-NC 4.0)"}))
    return docs


def load_lore(path, chunk_size, chunk_overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " "])
    docs = []
    for article in sources.split_articles(read_text(path)):
        for chunk in splitter.split_text(article.text):
            docs.append(Document(page_content=chunk, metadata={"title": article.title, "source": article.url}))
    return docs


def load(corpus):
    if not corpus.path.exists():
        raise FileNotFoundError(f"{corpus.path} not found. See the README section on the {corpus.name} corpus.")
    if corpus.kind == "glossary":
        return load_glossary(corpus.path)
    return load_lore(corpus.path, corpus.chunk_size, corpus.chunk_overlap)
