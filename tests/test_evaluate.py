"""Scoring rules of the evaluation (no models needed)."""
import unittest

from langchain_core.documents import Document

from rag_gta import config, evaluate as ev


class ScoreTest(unittest.TestCase):
    def test_normalization_ignores_case_and_accents(self):
        self.assertTrue(ev.contains_any("Use SEQUENCIA narrativa", ["sequência narrativa"]))
        self.assertFalse(ev.contains_any("dano", ["dano por segundo"]))

    def test_answer_item(self):
        item = {"id": "x", "type": "answer", "question": "q", "expect": ["dano por segundo"]}
        docs = [Document(page_content="Tradução em português: dano por segundo", metadata={"title": "DPS"})]
        r = ev.score(item, "Use «dano por segundo».", docs)
        self.assertEqual((r["retrieval_hit"], r["answer_hit"], r["refused"]), (True, True, False))
        r = ev.score(item, config.REFUSAL, docs)
        self.assertEqual((r["retrieval_hit"], r["answer_hit"], r["refused"]), (True, False, True))

    def test_refusal_item(self):
        item = {"id": "y", "type": "refusal", "question": "q"}
        self.assertTrue(ev.score(item, "Não encontrei essa informação nos registros.", [])["refusal_ok"])
        self.assertFalse(ev.score(item, "Canberra.", [])["refusal_ok"])

    def test_summary(self):
        rows = [{"type": "answer", "retrieval_hit": True, "answer_hit": True, "refused": False, "seconds": 1},
                {"type": "answer", "retrieval_hit": True, "answer_hit": False, "refused": True, "seconds": 3},
                {"type": "refusal", "refusal_ok": True, "refused": True, "seconds": 2}]
        self.assertEqual(ev.summarize(rows), {"questions": 3, "retrieval_hit": 1.0, "answer_hit": 0.5,
                                              "false_refusal": 0.5, "refusal_ok": 1.0, "avg_seconds": 2.0})

    def test_eval_sets_are_valid(self):
        for name in config.CORPORA:
            items = ev.load_set(ev.EVAL_DIR / f"{name}.jsonl")
            self.assertTrue(items)
            self.assertEqual(len({i["id"] for i in items}), len(items))
            self.assertTrue(all(i["expect"] for i in items if i["type"] == "answer"))

    def test_glossary_expectations_exist_in_the_glossary(self):
        # Every expected answer of the glossary set must really be in the CSV: the eval checks the system,
        # not the author's memory.
        from rag_gta import corpus
        text = "\n".join(d.page_content for d in corpus.load_glossary(config.CORPORA["glossary"].path))
        for i in ev.load_set(ev.EVAL_DIR / "glossary.jsonl"):
            if i["type"] == "answer":
                self.assertTrue(ev.contains_any(text, i["expect"]), i["id"])


if __name__ == "__main__":
    unittest.main()
