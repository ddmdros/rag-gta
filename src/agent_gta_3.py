import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURAÇÕES DE AMBIENTE E CAMINHOS ---
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "gta_bible.txt")
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_gta_db")

# --- 2. INICIALIZAÇÃO DE MODELOS ---
llm = ChatOllama(model="llama3.1:8b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 3. BANCO VETORIAL (CARREGAMENTO OU CRIAÇÃO) ---
if not os.path.exists(DB_DIR):
    print("🚀 Processando a Bíblia GTA pela primeira vez (600 páginas)...")
    # Usamos latin-1 para evitar erros de encoding comuns em TXTs grandes no Windows
    loader = TextLoader(FILE_PATH, encoding="latin-1")
    raw_docs = loader.load()

    # Aumentamos o chunk para 3000 para capturar biografias completas
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
    print(f"✅ Banco criado com {len(documents)} chunks.")
else:
    print("📂 Carregando banco vetorial existente...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Configurado para buscar os 6 trechos mais relevantes
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# --- 4. CONSTRUÇÃO DA CHAIN (LCEL COM FONTES) ---

def format_docs(docs):
    """Junta o conteúdo dos documentos para o prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

system_prompt = (
    "Você é um historiador sênior da franquia Grand Theft Auto.\n"
    "Use APENAS o contexto abaixo para responder. Ignore metadados técnicos ou URLs.\n\n"
    "DIRETRIZES:\n"
    "1. Extraia detalhes sobre personagens, gêneros e relações.\n"
    "2. Use pistas contextuais (he/she/ele/ela) para identificar gêneros.\n"
    "3. Se não encontrar a informação, diga: 'Informação não disponível nos registros'.\n\n"
    "CONTEXTO DOS ARQUIVOS:\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Esta é a 'mágica' do LCEL para retornar fontes:
# Criamos um dicionário que contém a resposta (answer) e os documentos (docs)
rag_chain_with_sources = (
    RunnableParallel({
        "context": retriever | format_docs, 
        "input": RunnablePassthrough(),
        "docs": retriever # Mantemos os documentos originais aqui
    })
    | {
        "answer": (lambda x: {"context": x["context"], "input": x["input"]}) | prompt | llm | StrOutputParser(),
        "sources": lambda x: x["docs"]
    }
)

# --- 5. INTERFACE DE INTERAÇÃO ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📖 ORÁCULO GTA - CONSULTA COM CITAÇÃO DE FONTES")
    print("="*60)
    
    while True:
        pergunta = input("\nPergunta: ").strip()
        if pergunta.lower() in ['sair', 'exit']: break
        if not pergunta: continue
        
        print("🔍 Consultando registros históricos...")
        
        # Executa a chain
        resultado = rag_chain_with_sources.invoke(pergunta)
        
        # Exibe a Resposta
        print(f"\n🤖 RESPOSTA:\n{resultado['answer']}")
        
        # Exibe as Fontes (opcional, mas ótimo para o portfólio)
        print("\n📚 FONTES CONSULTADAS:")
        fontes_exibidas = set()
        for doc in resultado['sources']:
            # Extrai os primeiros 80 caracteres como prova de origem
            snippet = doc.page_content[:100].replace('\n', ' ')
            # Evita repetir o mesmo snippet se o retriever trouxer blocos redundantes
            if snippet not in fontes_exibidas:
                print(f"   - [Registro]: \"{snippet}...\"")
                fontes_exibidas.add(snippet)