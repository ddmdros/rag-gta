"""Command line.

  python -m rag_gta index {lore,glossary} [--rebuild]
  python -m rag_gta ask {lore,glossary} "pergunta" [--mode similarity|mmr|multiquery] [--k N] [--show-chunks]
  python -m rag_gta chat {lore,glossary} [--mode ...] [--k N]
  python -m rag_gta eval {lore,glossary} [--mode ...] [--k N] [--limit N]
"""
import argparse, sys

from . import config, evaluate, pipeline, sources


def print_answer(out, show_chunks=False):
    print(f"\n{out['answer']}\n")
    print("Fontes:")
    for i, (title, url, preview) in enumerate(sources.unique_sources(out["docs"]), 1):
        print(f"  {i}. {title}" + (f" — {url}" if url.startswith("http") else ""))
        if show_chunks:
            print(f"     «{preview}...»")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag_gta", description="RAG local sobre o universo GTA e um glossário de games.")
    ap.add_argument("command", choices=["index", "ask", "chat", "eval"])
    ap.add_argument("corpus", choices=sorted(config.CORPORA))
    ap.add_argument("question", nargs="?")
    ap.add_argument("--mode", choices=pipeline.MODES)
    ap.add_argument("--k", type=int)
    ap.add_argument("--rebuild", action="store_true", help="drop and rebuild the vector index")
    ap.add_argument("--show-chunks", action="store_true")
    ap.add_argument("--limit", type=int, help="eval: only the first N questions")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if a.command == "index":
        pipeline.build(a.corpus, rebuild=a.rebuild)
        return 0
    rag, corpus = pipeline.build(a.corpus, a.mode, a.k)
    if a.command == "ask":
        if not a.question:
            ap.error("ask needs a question")
        print_answer(rag.invoke({"question": a.question}), a.show_chunks)
    elif a.command == "chat":
        from langchain_core.messages import AIMessage, HumanMessage

        history = []
        print("Digite 'sair' para encerrar.")
        while True:
            q = input("\nVocê: ").strip()
            if q.lower() in ("sair", "exit"):
                break
            if not q:
                continue
            out = rag.invoke({"question": q, "history": history})
            print_answer(out, a.show_chunks)
            history += [HumanMessage(q), AIMessage(out["answer"])]
            history = history[-8:]  # last 4 turns: enough for follow-ups, keeps the prompt small
    else:
        items = evaluate.load_set(evaluate.EVAL_DIR / f"{a.corpus}.jsonl")[: a.limit]
        mode = a.mode or corpus.search
        report = evaluate.run(rag, items, label=f"{a.corpus} {mode} k={a.k or corpus.k} {config.LLM_MODEL}")
        print("\nResumo:", report["summary"])
        print("Salvo em", evaluate.save(report, a.corpus, mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
