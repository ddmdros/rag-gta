import os
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


#LLM and Ollama settings
llm = LLM(
    model="ollama/llama3.1:8b"
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

#Rag with ChromaDB

loader = CSVLoader("./src/data/glossario_gamer_ptbr.csv")
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

documents = text_splitter.split_documents(raw_documents)

print(f"Total de chunks criados: {len(documents)}")
print(f"Exemplo do primeiro chunk (primeiros 300 chars):\n{documents[0].page_content[:300]}")

vectorstore = Chroma.from_documents(documents, embedding=embeddings)

#busca por similaridade
retriever = vectorstore.as_retriever(
    search_kwargs = {"k":15}
)

@tool("Busca no glossario local")
def search_local_knowledge(query:str) -> str:
    """
    Busca traduções de termos de jogos.
    ATENÇÃO: Você deve passar APENAS UMA ÚNICA PALAVRA (string simples) no parâmetro 'query'. 
    NÃO USE listas (colchetes) ou formatos JSON.
    """
    docs = retriever.invoke(query)

    print("\nChunks recuperados")
    for i, doc in enumerate(docs, 1):
        print(f"\nChunk {i}")
        print(f"Tamanho: {len(doc.page_content)} caracteres")
        print(f"Conteúdo: {doc.page_content[:200]}...")
    print("="*50 + "\n")
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


#Parte 3: criando o agente e a tarefa com CrewAi

ResearcherAgent = Agent(
    role="Você é um especialista em localização de jogos nos pares EN-PTBR",
    goal="Fornecer traduções precisas de termos utilizados nos jogos com base no glossário local",
    backstory="Você é um tradutor sênior da indústria de games. Seu objetivo é ajudar desenvolvedores a encontrar o termo oficial em português brasileiro para termos em inglês usando estritamente a sua base de dados fornecida.",
    verbose=True,
    allow_delegation=False,
    tools=[search_local_knowledge],
    llm=llm
)

topic = "DPS"

task = Task(
    description=(
        f"Pesquise EXATAMENTE a palavra '{topic}' usando a ferramenta de busca. "
        "Leia o resultado retornado pela ferramenta."
        "IMPORTANTE: Se a ferramenta não retornar a palavra '{topic}' após duas tentativas, pare de pesquisar e responda que não encontrou a resposta."
    ),
    expected_output=(
        "Um texto curto em português informando a tradução oficial de '{topic}' "
        "conforme encontrado no documento e, se houver, um exemplo de uso."
    ),
    agent=ResearcherAgent
)

#Parte 4: montando e executando a crew

crew = Crew(
    agents=[ResearcherAgent],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

print("Resultado:")
print(result)