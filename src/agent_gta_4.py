import os
import re
import logging
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Importação do pacote classic para o MultiQuery
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# --- 1. CONFIGURAÇÕES DE AMBIENTE ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "gta_bible.txt")
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_gta_db")

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- 2. INICIALIZAÇÃO DE MODELOS ---
llm = ChatOllama(model="llama3.1:8b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 3. BANCO VETORIAL (PROCESSAMENTO E PERSISTÊNCIA) ---
if not os.path.exists(DB_DIR):
    print("🚀 Processando a Bíblia GTA pela primeira vez...")
    loader = TextLoader(FILE_PATH, encoding="latin-1")
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = text_splitter.split_documents(raw_docs)
    
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"✅ Sucesso! {len(documents)} blocos indexados.")
else:
    print("📂 Carregando banco vetorial existente...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- 4. CONFIGURAÇÃO DO RETRIEVER (MULTI-QUERY) ---
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# --- 5. LÓGICA DE EXTRAÇÃO DE METADADOS (TÍTULO E LINK) ---

def extrair_dados_fonte(texto):
    """Extrai o título formatado e a URL original da Wiki."""
    # 1. Extrai a URL completa
    match_url = re.search(r'https://gta\.fandom\.com/wiki/[^\s\-]+', texto)
    url = match_url.group(0) if match_url else "Link não disponível"
    
    # 2. Extrai o Título (baseado na parte final da URL)
    match_titulo = re.search(r'wiki/([^ \-\n]+)', texto)
    if match_titulo:
        titulo = match_titulo.group(1).replace('_', ' ').replace('%27', "'").title()
    else:
        titulo = "Registro Geral"
        
    return titulo, url

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 6. CONSTRUÇÃO DA CHAIN (LCEL) ---
system_prompt = (
    "Você é um historiador sênior da franquia Grand Theft Auto.\n"
    "Use APENAS o contexto abaixo para responder. Ignore metadados técnicos.\n"
    "Se não souber, diga que a informação não consta nos registros.\n\n"
    "CONTEXTO:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs, 
        "input": RunnablePassthrough(),
        "docs": retriever
    })
    | {
        "answer": (lambda x: {"context": x["context"], "input": x["input"]}) | prompt | llm | StrOutputParser(),
        "sources": lambda x: x["docs"]
    }
)

# --- 7. INTERFACE ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📖 ORÁCULO GTA - COM LINKS ORIGINAIS DA WIKI")
    print("="*60)
    
    while True:
        pergunta = input("\nPergunta: ").strip()
        if pergunta.lower() in ['sair', 'exit']: break
        if not pergunta: continue
        
        print("🔍 Consultando registros e gerando referências...")
        
        try:
            resultado = rag_chain.invoke(pergunta)
            
            print(f"\n🤖 RESPOSTA:\n{resultado['answer']}")
            
            print("\n" + "─"*40)
            print("📑 FONTES CONSULTADAS")
            print("─"*40)
            
            fontes_unicas = {}
            for doc in resultado['sources']:
                nome, link = extrair_dados_fonte(doc.page_content)
                
                # Limpa o texto para o resumo
                conteudo_limpo = doc.page_content.split('---')[-1].strip()
                resumo = conteudo_limpo[:120].replace('\n', ' ')
                
                if nome not in fontes_unicas:
                    fontes_unicas[nome] = {"link": link, "resumo": resumo}

            for i, (nome, dados) in enumerate(fontes_unicas.items(), 1):
                print(f"  {i}. {nome.upper()}")
                print(f"     🔗 Link: {dados['link']}")
                print(f"     📄 Contexto: \"{dados['resumo']}...\"")
                print("") # Espaço entre fontes
            
            print("─"*40)
        except Exception as e:
            print(f"❌ Erro: {e}")