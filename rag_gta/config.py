"""Settings: models, paths and the two corpora (lore and glossary)."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INDEX_DIR = DATA / "chroma"

# Local models served by Ollama. Override with environment variables.
LLM_MODEL = os.environ.get("RAG_LLM", "llama3.1:8b")
EMBED_MODEL = os.environ.get("RAG_EMBED", "nomic-embed-text")

REFUSAL = "Não encontrei essa informação nos registros."

LORE_PROMPT = (
    "Você é um historiador da franquia Grand Theft Auto.\n"
    "Responda em português, usando APENAS os trechos abaixo, tirados de artigos do GTA Wiki.\n"
    "Regras:\n"
    "1. Cite nomes de personagens, lugares e jogos como aparecem nos trechos.\n"
    "2. Não use conhecimento de fora dos trechos e não invente fatos.\n"
    f"3. Se os trechos não trazem a resposta, diga exatamente: «{REFUSAL}»\n\n"
    "TRECHOS:\n{context}"
)

GLOSSARY_PROMPT = (
    "Você é um especialista em localização de jogos do inglês para o português do Brasil.\n"
    "Responda usando APENAS as entradas de glossário abaixo.\n"
    "Regras:\n"
    "1. Dê a tradução como está no glossário; se houver mais de uma opção, liste todas.\n"
    "2. Se a entrada tiver definição ou nota de uso, resuma em uma frase.\n"
    f"3. Se o termo não está nas entradas, diga exatamente: «{REFUSAL}»\n\n"
    "ENTRADAS DO GLOSSÁRIO:\n{context}"
)


@dataclass(frozen=True)
class Corpus:
    name: str
    kind: str            # "lore" (long text, split in chunks) | "glossary" (one document per term)
    path: Path
    prompt: str
    k: int               # chunks handed to the model
    search: str          # default retrieval mode: "similarity" | "mmr" | "multiquery"
    chunk_size: int = 0
    chunk_overlap: int = 0


CORPORA = {
    "lore": Corpus("lore", "lore", DATA / "gta_bible.txt", LORE_PROMPT, k=8, search="mmr",
                   chunk_size=1500, chunk_overlap=300),
    "glossary": Corpus("glossary", "glossary", DATA / "glossario_gamer_ptbr.csv", GLOSSARY_PROMPT, k=4,
                       search="similarity"),
}
