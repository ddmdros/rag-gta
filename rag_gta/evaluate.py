"""Evaluation: does the system find the right passage, answer correctly, and admit when it does not know?

Eval set: eval/<corpus>.jsonl, one question per line:
  {"id": "g01", "type": "answer",  "question": "...", "expect": ["dano por segundo"]}
  {"id": "g90", "type": "refusal", "question": "..."}
Metrics (accents and case ignored):
  retrieval hit   answer questions: some expected string appears in the retrieved documents
  answer hit      answer questions: some expected string appears in the answer, and the answer is not a refusal
  refusal ok      refusal questions: the answer is the refusal sentence (the corpus does not have it)
  false refusal   answer questions: the model refused although the answer was retrievable
Retrieval and answer are scored apart on purpose: a miss in retrieval and a miss in generation have
different fixes (chunking/k/mode vs. prompt/model).
"""
import json, time, unicodedata
from datetime import datetime
from pathlib import Path

from . import config

EVAL_DIR = config.ROOT / "eval"


def norm(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def contains_any(text, expected):
    t = norm(text)
    return any(norm(e) in t for e in expected)


def is_refusal(answer):
    return norm(config.REFUSAL.rstrip(".")) in norm(answer) or "nao encontrei" in norm(answer)


def load_set(path):
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            if item.get("type") not in ("answer", "refusal"):
                raise ValueError(f"{path}:{n}: type must be 'answer' or 'refusal'")
            items.append(item)
    return items


def score(item, answer, docs):
    context = "\n".join(d.page_content for d in docs)
    refused = is_refusal(answer)
    r = {"id": item["id"], "type": item["type"], "question": item["question"], "answer": answer,
         "sources": [d.metadata.get("title", "") for d in docs], "refused": refused}
    if item["type"] == "answer":
        r["retrieval_hit"] = contains_any(context, item["expect"])
        r["answer_hit"] = contains_any(answer, item["expect"]) and not refused
    else:
        r["refusal_ok"] = refused
    return r


def summarize(results):
    ans = [r for r in results if r["type"] == "answer"]
    ref = [r for r in results if r["type"] == "refusal"]

    def rate(rows, key):
        return round(sum(bool(r[key]) for r in rows) / len(rows), 3) if rows else None

    return {"questions": len(results), "retrieval_hit": rate(ans, "retrieval_hit"),
            "answer_hit": rate(ans, "answer_hit"), "false_refusal": rate(ans, "refused"),
            "refusal_ok": rate(ref, "refusal_ok"),
            "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 2) if results else None}


def run(rag, items, label=""):
    """Run every item through the chain. Returns {"label", "summary", "results"}."""
    results = []
    for item in items:
        t0 = time.perf_counter()
        out = rag.invoke({"question": item["question"]})
        r = score(item, out["answer"], out["docs"])
        r["seconds"] = round(time.perf_counter() - t0, 2)
        results.append(r)
        mark = "ok " if r.get("answer_hit", r.get("refusal_ok")) else "ERR"
        print(f"  [{mark}] {item['id']}: {item['question'][:70]}")
    return {"label": label, "summary": summarize(results), "results": results}


def save(report, corpus_name, mode):
    out_dir = EVAL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{corpus_name}-{mode}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path
