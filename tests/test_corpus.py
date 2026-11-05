"""Glossary parsing and lore splitting (no models needed)."""
import tempfile, unittest
from pathlib import Path

from rag_gta import config, corpus, sources


class GlossaryTest(unittest.TestCase):
    def test_preamble_is_skipped(self):
        rows = corpus.glossary_rows(config.CORPORA["glossary"].path)
        self.assertGreater(len(rows), 100)
        self.assertFalse(any("Attribution" in r[0] or "Rules of use" in r[0] for r in rows))
        self.assertEqual(rows[0][0].lower(), rows[0][0].lower().strip())

    def test_one_document_per_term(self):
        docs = corpus.load_glossary(config.CORPORA["glossary"].path)
        by_term = {d.metadata["title"].lower(): d.page_content for d in docs}
        self.assertIn("Tradução em português: dano por segundo", by_term["damage per second"])
        self.assertTrue(by_term["crash"].splitlines()[1].endswith("fecha sozinho"))  # trailing ", " stripped

    def test_latin1_fallback(self):
        p = Path(tempfile.mkdtemp()) / "g.csv"
        p.write_bytes("English,Definition,Brazilian Portuguese,Usage and Notes\nheal,,cura|r,ação\n".encode("latin-1"))
        self.assertEqual(corpus.glossary_rows(p), [("heal", "", "cura|r", "ação")])


class LoreTest(unittest.TestCase):
    TEXT = ("Introdução sem link.\n"
            "https://gta.fandom.com/wiki/Carl_Johnson - Carl Johnson\n"
            "Carl Johnson, também chamado CJ, é o protagonista.\n\n"
            "Ele volta para Los Santos.\n"
            "https://gta.fandom.com/wiki/Tommy_Vercetti\n"
            "Tommy é o protagonista de Vice City.\n")

    def test_split_articles(self):
        arts = sources.split_articles(self.TEXT)
        self.assertEqual([a.title for a in arts], ["Registro geral", "Carl Johnson", "Tommy Vercetti"])
        self.assertEqual(arts[1].url, "https://gta.fandom.com/wiki/Carl_Johnson")
        self.assertIn("Ele volta para Los Santos.", arts[1].text)
        self.assertEqual(arts[1].text.splitlines()[0], "Carl Johnson")

    def test_title_from_url(self):
        self.assertEqual(sources.title_from_url("https://gta.fandom.com/wiki/Big_Smoke%27s_Order"), "Big Smoke's Order")

    def test_chunks_keep_their_article(self):
        p = Path(tempfile.mkdtemp()) / "bible.txt"
        p.write_text(self.TEXT * 1, encoding="utf-8")
        docs = corpus.load_lore(p, chunk_size=40, chunk_overlap=0)
        self.assertGreater(len(docs), 3)
        cj = [d for d in docs if "Los Santos" in d.page_content][0]
        self.assertEqual(cj.metadata["title"], "Carl Johnson")

    def test_missing_corpus_says_where_to_look(self):
        c = config.Corpus("lore", "lore", Path("/nope/bible.txt"), "", k=1, search="mmr", chunk_size=10)
        with self.assertRaisesRegex(FileNotFoundError, "README"):
            corpus.load(c)


if __name__ == "__main__":
    unittest.main()
