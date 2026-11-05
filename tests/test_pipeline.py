"""The whole chain on the real glossary, with Chroma on a temporary folder and fake models (no Ollama).

Fake embeddings make retrieval quality meaningless here; these tests check the plumbing: one retrieval per
question, sources returned with the answer, history rewriting the question, and the eval loop end to end.
"""
import shutil, tempfile, unittest

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from rag_gta import config, evaluate, pipeline


class CountingRetriever:
    def __init__(self, inner):
        self.inner, self.queries = inner, []

    def invoke(self, query):
        self.queries.append(query)
        return self.inner.invoke(query)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="chroma_")
        cls.emb = DeterministicFakeEmbedding(size=64)
        cls.store = pipeline.index(config.CORPORA["glossary"], cls.emb, persist_dir=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make(self, answers, mode="similarity"):
        retr = CountingRetriever(pipeline.retriever(self.store, config.CORPORA["glossary"], mode))
        llm = FakeListChatModel(responses=answers)
        return pipeline.chain(retr, llm, config.GLOSSARY_PROMPT), retr

    def test_index_is_built_once(self):
        n = len(self.store.get()["ids"])
        self.assertGreater(n, 100)
        again = pipeline.index(config.CORPORA["glossary"], self.emb, persist_dir=self.tmp)
        self.assertEqual(len(again.get()["ids"]), n)

    def test_one_retrieval_and_sources(self):
        rag, retr = self.make(["Use «dano por segundo»."])
        out = rag.invoke({"question": "Como traduzo DPS?"})
        self.assertEqual(out["answer"], "Use «dano por segundo».")
        self.assertEqual(len(out["docs"]), config.CORPORA["glossary"].k)
        self.assertEqual(retr.queries, ["Como traduzo DPS?"])

    def test_mmr_mode(self):
        rag, retr = self.make(["ok"], mode="mmr")
        self.assertEqual(len(rag.invoke({"question": "buff"})["docs"]), config.CORPORA["glossary"].k)

    def test_history_rewrites_the_query(self):
        rag, retr = self.make(["Como se traduz 'nerf'?", "enfraquecimento"])
        history = [HumanMessage("Como traduzo buff?"), AIMessage("bônus")]
        out = rag.invoke({"question": "E o contrário?", "history": history})
        self.assertEqual(retr.queries, ["Como se traduz 'nerf'?"])
        self.assertEqual(out["answer"], "enfraquecimento")

    def test_eval_loop(self):
        items = evaluate.load_set(evaluate.EVAL_DIR / "glossary.jsonl")
        items = [items[0], items[-1]]
        rag, _ = self.make(["dano por segundo", config.REFUSAL])
        report = evaluate.run(rag, items, "fake")
        self.assertEqual(report["summary"]["answer_hit"], 1.0)
        self.assertEqual(report["summary"]["refusal_ok"], 1.0)


if __name__ == "__main__":
    unittest.main()
