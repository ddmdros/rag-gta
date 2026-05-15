import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURAÇÃO DE AMBIENTE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_gta_db")

class GTAOracle:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 7})
        self.history_store = {} # Simula um banco de dados de sessões

    def get_session_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        # 1. Prompt de contextualização (Trata o histórico)
        contextualize_system_prompt = (
            "Dado o histórico de conversa e a pergunta mais recente do usuário, "
            "formule uma pergunta independente que possa ser entendida sem o histórico."
        )
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 2. Prompt de Resposta (O Historiador)
        system_prompt = (
            "Você é o Historiador Sênior da franquia GTA. Use os trechos abaixo para responder.\n\n"
            "CONTEXTO:\n{context}\n\n"
            "DIRETRIZ: Se não souber, diga honestamente que a informação não consta nos registros."
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 3. Definição da Cadeia (Pipe)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(self.retriever.invoke(x["input"]))
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

        # 4. Wrap com gerenciamento de histórico
        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

# --- EXECUÇÃO ---
if __name__ == "__main__":
    oracle = GTAOracle()
    chain = oracle.build_chain()
    
    print("🎮 Oráculo GTA Online (Com Memória)")
    
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit"]: break
        
        # Simulamos uma session_id para manter o histórico
        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "diogo_session_01"}}
        )
        
        print(f"\n🤖 Oráculo: {response}")