import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configurações de Telemetria e LLM
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

llm = LLM(model="ollama/llama3.1:8b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- PARTE 1: INGESTÃO DA BÍBLIA GTA ---
# Pega o caminho absoluto da pasta onde o agent_gta.py está (que é a pasta 'src')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Monta o caminho correto para chegar na pasta 'data' e pegar o arquivo
FILE_PATH = os.path.join(BASE_DIR, "data", "gta_bible.txt")

DB_DIR = os.path.join(BASE_DIR, "data", "chroma_gta_db")
# Só processa o arquivo se o banco de dados ainda não existir
if not os.path.exists(DB_DIR):
    print("Processando a bíblia pela primeira vez... Isso pode demorar um pouco.")
    loader = TextLoader(FILE_PATH, encoding="latin-1")
    raw_documents = loader.load()

    # Splitter ajustado para textos longos (chunks maiores para manter contexto narrativo)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = text_splitter.split_documents(raw_documents)
    
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print(f"Indexação concluída. {len(documents)} chunks criados.")
else:
    print("Carregando banco de dados vetorial existente...")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- PARTE 2: FERRAMENTA DE BUSCA ---
@tool("Consultar Lore do GTA")
def search_gta_bible(query: str) -> str:
    """
    Busca informações detalhadas sobre o universo de Grand Theft Auto 
    dentro do documento gta_bible.txt.
    """
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# --- PARTE 3: AGENTE E TAREFA ---
GTA_Expert = Agent(
    role="Historiador Sênior da franquia Grand Theft Auto",
    goal="Responder perguntas com precisão absoluta baseando-se apenas na gta_bible",
    backstory=(
        "Você é o maior arquivista da Rockstar Games. Você conhece cada rua de Los Santos "
        "e cada segredo de Liberty City. Sua integridade é sua marca: se a informação "
        "não estiver nos seus registros (documentos), você admite que não sabe. "
        "Você nunca inventa lore e nunca mistura o seu conhecimento externo com o que está no arquivo."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_gta_bible],
    llm=llm
)

def realizar_pergunta(pergunta_usuario):
    task = Task(
        description=(
            f"Analise a seguinte pergunta: '{pergunta_usuario}'. "
            "Use a ferramenta de busca para encontrar fatos na gta_bible. "
            "Se a informação for encontrada, forneça uma resposta detalhada. "
            "IMPORTANTE: Se após a busca você não encontrar evidências diretas na ferramenta, "
            "responda exatamente: 'Lamentavelmente, essa informação não consta nos meus registros atuais sobre o GTA'."
        ),
        expected_output="Uma resposta clara e informativa baseada no documento, ou a negação de conhecimento.",
        agent=GTA_Expert
    )

    crew = Crew(
        agents=[GTA_Expert],
        tasks=[task],
        process=Process.sequential,
        verbose=False # Reduzi o verbose aqui para focar na resposta
    )
    
    return crew.kickoff()

# --- PARTE 4: INTERAÇÃO ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("ORÁCULO GTA - BÍBLIA DE 600 PÁGINAS ONLINE")
    print("="*50)
    
    while True:
        pergunta = input("\nO que deseja saber sobre o universo GTA? (ou 'sair'): ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
        
        resultado = realizar_pergunta(pergunta)
        print(f"\n✨ RESPOSTA:\n{resultado}")