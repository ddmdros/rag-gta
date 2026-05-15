import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURAÇÃO DE AMBIENTE ---
# Desativa telemetria para maior privacidade e performance
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# Caminhos dinâmicos (Garante que funcione em qualquer pasta)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data", "gta_bible.txt")
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_gta_db")

# --- 1. MODELOS (OLLAMA) ---
# Usamos temperature 0 para que a IA seja um historiador rígido e não invente fatos
llm = ChatOllama(model="llama3.1:8b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 2. BANCO VETORIAL PERSISTENTE ---
if not os.path.exists(DB_DIR):
    print("🚀 Iniciando processamento da Bíblia GTA (600 páginas)...")
    print("Isso será feito apenas uma vez e os vetores serão salvos em disco.")
    
    # Carregamento com latin-1 para evitar erros de caracteres especiais do Windows
    loader = TextLoader(FILE_PATH, encoding="latin-1")
    raw_documents = loader.load()

    # Divisão do texto em blocos (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = text_splitter.split_documents(raw_documents)
    
    # Criação e salvamento do banco vetorial
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"✅ Sucesso! {len(documents)} blocos de conhecimento indexados.")
else:
    print("📂 Carregando banco de dados vetorial de: " + DB_DIR)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Configura o buscador para trazer os 5 trechos mais relevantes
# retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 18,              # Aumentamos para 10 pedaços de informação
        "fetch_k": 30,        # Ele analisa 30 e escolhe os 10 mais diversos
        "lambda_mult": 0.5    # Equilíbrio entre relevância e diversidade
    }
)

# --- 3. DEFINIÇÃO DA LÓGICA RAG (MODO MODERNO LCEL) ---

# Função auxiliar para formatar os documentos recuperados
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Template do sistema (O "caráter" da IA)
# system_prompt = (
#     "Você é um historiador sênior da franquia Grand Theft Auto.\n"
#     "Use APENAS o contexto fornecido abaixo para responder às perguntas.\n\n"
#     "REGRAS DE OURO:\n"
#     "1. Se a informação não estiver no texto, responda: 'Lamentavelmente, essa informação não consta nos meus registros'.\n"
#     "2. Seja direto, mas detalhado quando o documento permitir.\n"
#     "3. Cite nomes de personagens, datas e locais conforme aparecem no registro.\n\n"
#     "CONTEXTO DISPONÍVEL:\n"
#     "{context}"
# )
system_prompt = (
    "Você é um historiador especialista na franquia GTA.\n"
    "Abaixo estão trechos extraídos da bíblia oficial do jogo.\n"
    "Seu objetivo é extrair o máximo de informação possível desses trechos.\n\n"
    "DIRETRIZES:\n"
    "1. Se os trechos mencionarem os personagens ou suas relações, explique-os detalhadamente.\n"
    "2. Se a informação realmente não estiver nos trechos abaixo, diga que não encontrou nos registros específicos, mas tente inferir se houver pistas.\n"
    "3. Não invente fatos que não tenham base no texto fornecido.\n\n"
    "4. Use pistas contextuais e pronomes (como 'he', 'she', 'ele', 'ela') presentes nos trechos para identificar características como gênero e relações, sem inventar fatos externos."
    "TRECHOS DA BÍBLIA:\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# --- MONTAGEM DA CORRENTE (PIPE |) ---
# Este é o fluxo de dados: Pergunta -> Busca -> Prompt -> LLM -> Texto
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

            # --- LOOP DE INTERAÇÃO COM DEBUG ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📖 ORÁCULO GTA - MODO DIAGNÓSTICO")
    print("="*60)
    
    while True:
        pergunta = input("\nSua dúvida (ou 'sair'): ").strip()
        if pergunta.lower() in ['sair', 'exit']: break
        
        # DEBUG: Vamos ver o que o buscador está achando antes de mandar pra IA
        print("🔍 Buscando evidências no banco vetorial...")
        docs_encontrados = retriever.invoke(pergunta)
        
        print(f"--- DEBUG: O buscador encontrou {len(docs_encontrados)} trechos ---")
        for i, d in enumerate(docs_encontrados):
            # Imprime os primeiros 100 caracteres de cada trecho recuperado
            print(f"Trecho [{i+1}]: {d.page_content[:150].replace('\n', ' ')}...")
        print("-" * 40)

        try:
            resposta = rag_chain.invoke(pergunta)
            print(f"\n🤖 RESPOSTA FINAL:\n{resposta}")
        except Exception as e:
            print(f"❌ Erro: {e}")