import os
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_classic.chains import create_retrieval_chain

# Parte 1: Configuração do LLM e Embeddings
# No LangChain direto, usamos ChatOllama em vez do LLM do CrewAI
llm = ChatOllama(model="llama3.1:8b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Parte 2: Carregamento do CSV e Criação do Banco Vetorial
print("Carregando documento e criando banco de dados vetorial...")
loader = CSVLoader("./src/data/glossario_gamer_ptbr.csv")
documents = loader.load()

vectorstore = Chroma.from_documents(documents, embedding=embeddings)

# Trazemos os 10 resultados mais parecidos para não perder nada
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Parte 3: Criando o Prompt e a Corrente (Chain) de Resposta
system_prompt = (
    "Você é um especialista em localização de jogos nos pares EN-PTBR. "
    "Use APENAS o seguinte glossário para responder à pergunta do usuário. "
    "Não invente respostas. Se o termo não estiver no glossário, diga que não encontrou.\n\n"
    "Glossário encontrado:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Aqui montamos a tubulação: O RAG busca no banco, e passa para a cadeia de perguntas
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Parte 4: Executando a Busca Direta
pergunta = "Como devo traduzir o termo 'cooldown'? O glossário possui algum exemplo prático de uso para ele?"

print(f"\nPesquisando: '{pergunta}'\n" + "="*50)
response = rag_chain.invoke({"input": pergunta})

# Opcional: imprimir o que o banco encontrou para debug
print("Documentos usados pelo LLM:")
for doc in response['context']:
    if "DPS" in doc.page_content or "DoT" in doc.page_content:
        print(f"- {doc.page_content.strip()}")
print("="*50 + "\n")

# A resposta final do LLM
print("Resposta Final:")
print(response["answer"])