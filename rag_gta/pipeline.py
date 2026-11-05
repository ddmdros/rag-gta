"""Index, retriever and the RAG chain.

The chain retrieves once per question and returns the answer together with the documents it used, so the
caller can show sources and the evaluation can score retrieval and generation separately. (The earlier version
ran the retriever twice per question: once for the prompt context and once for the source list.)
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from . import config, corpus as corpus_mod

MODES = ("similarity", "mmr", "multiquery")

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Dado o histórico da conversa e a última pergunta, reescreva a pergunta para que ela possa ser "
               "entendida sozinha, sem o histórico. Não responda: devolva só a pergunta reescrita."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def default_models():
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    return ChatOllama(model=config.LLM_MODEL, temperature=0), OllamaEmbeddings(model=config.EMBED_MODEL)


def index(corpus, embeddings, rebuild=False, persist_dir=None):
    """Chroma collection for the corpus, persisted on disk. Built once; `rebuild` drops and rebuilds it."""
    from langchain_chroma import Chroma

    persist = str(persist_dir or config.INDEX_DIR / corpus.name)
    store = Chroma(collection_name=corpus.name, embedding_function=embeddings, persist_directory=persist)
    if rebuild:
        store.reset_collection()
    if not store.get(limit=1)["ids"]:
        docs = corpus_mod.load(corpus)
        store.add_documents(docs)
        print(f"Indexed {len(docs)} documents from {corpus.path.name} into {persist}")
    return store


def retriever(store, corpus, mode=None, llm=None, k=None):
    mode, k = mode or corpus.search, k or corpus.k
    if mode == "similarity":
        return store.as_retriever(search_kwargs={"k": k})
    if mode == "mmr":  # relevant but diverse: avoids k near-identical chunks of the same paragraph
        return store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(30, 3 * k),
                                                                    "lambda_mult": 0.5})
    if mode == "multiquery":  # the LLM rewrites the question in several ways and the results are merged
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        return MultiQueryRetriever.from_llm(retriever=store.as_retriever(search_kwargs={"k": k}), llm=llm)
    raise ValueError(f"mode must be one of {MODES}")


def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


def chain(retr, llm, system_prompt):
    """Runnable: {"question", "history"?} -> {"question", "answer", "docs"}.

    With a non-empty history, the question is first rewritten to stand alone (so "e o irmão dele?" becomes a
    real search query); the rewritten question is what gets retrieved.
    """
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ])
    condense = CONDENSE_PROMPT | llm | StrOutputParser()

    def standalone(x):
        return condense.invoke(x).strip() if x.get("history") else x["question"]

    return (
        RunnablePassthrough.assign(search_query=RunnableLambda(standalone))
        | RunnablePassthrough.assign(docs=RunnableLambda(lambda x: retr.invoke(x["search_query"])))
        | RunnablePassthrough.assign(answer=RunnableLambda(lambda x: {**x, "context": format_docs(x["docs"])})
                                     | answer_prompt | llm | StrOutputParser())
    )


def build(corpus_name, mode=None, k=None, llm=None, embeddings=None, rebuild=False, persist_dir=None):
    """(chain, corpus) ready to invoke. Models default to the local Ollama ones."""
    corpus = config.CORPORA[corpus_name]
    if llm is None or embeddings is None:
        default_llm, default_emb = default_models()
        llm, embeddings = llm or default_llm, embeddings or default_emb
    store = index(corpus, embeddings, rebuild, persist_dir)
    return chain(retriever(store, corpus, mode, llm, k), llm, corpus.prompt), corpus
