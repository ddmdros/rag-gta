"""Lore corpus builder, with a fake wiki (no network)."""
import unittest

from rag_gta import lore_corpus, sources

HTML = """
<div class="mw-parser-output">
<aside class="portable-infobox"><p>Infobox text that must not appear</p></aside>
<p><b>Carl Johnson</b>, also known as <b>CJ</b>, is the protagonist<sup>[1]</sup> of
   <i>Grand Theft Auto: San Andreas</i>.</p>
<table><tr><td><p>Table cell</p></td></tr></table>
<h2>History</h2>
<p>He returns to Los Santos &amp; meets Sweet.</p>
<h2>Empty section</h2>
<h2>References</h2>
<p>Text after the references</p>
</div>
"""

PAGES = {
    "Carl Johnson": ("Carl Johnson", HTML),
    "Los Santos": ("Los Santos (HD Universe)", "<p>Los Santos is based on Los Angeles.</p>"),
}


def fake_fetch(title):
    return PAGES.get(title)


class ProseTest(unittest.TestCase):
    def test_keeps_prose_and_drops_boxes_tables_and_references(self):
        text = lore_corpus.prose(HTML)
        self.assertEqual(text.splitlines()[0],
                         "Carl Johnson, also known as CJ, is the protagonist of Grand Theft Auto: San Andreas.")
        self.assertIn("== History ==", text)
        self.assertIn("He returns to Los Santos & meets Sweet.", text)
        for gone in ("Infobox", "Table cell", "[1]", "References", "after the references"):
            self.assertNotIn(gone, text)


class BuildTest(unittest.TestCase):
    def test_build_writes_the_format_sources_reads(self):
        text, report = lore_corpus.build(["Carl Johnson", "Los Santos", "Nowhere"], fetch=fake_fetch, delay=0)
        self.assertEqual(report[2], ("Nowhere", None, 0))
        self.assertEqual(report[1][1], "Los Santos (HD Universe)")

        arts = sources.split_articles(text)
        self.assertEqual([a.title for a in arts], ["Carl Johnson", "Los Santos (HD Universe)"])
        self.assertEqual(arts[1].url, "https://gta.fandom.com/wiki/Los_Santos_%28HD_Universe%29")
        self.assertIn("Los Angeles", arts[1].text)

    def test_url_keeps_colons_readable(self):
        self.assertEqual(lore_corpus.article_url("Grand Theft Auto: San Andreas"),
                         "https://gta.fandom.com/wiki/Grand_Theft_Auto:_San_Andreas")


if __name__ == "__main__":
    unittest.main()
