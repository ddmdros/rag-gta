# Dependências do Projeto RAG-GTA

Este projeto utiliza as seguintes dependências Python:

## Pacotes Python
- langchain-ollama: Integração com Ollama para LLMs
- langchain-community: Componentes comunitários do LangChain
- langchain-core: Núcleo do LangChain
- chromadb: Banco de dados vetorial para embeddings
- langchain-chroma: Integração Chroma com LangChain
- ollama: Cliente Python para Ollama
- numpy: Computação numérica
- pydantic: Validação de dados
- requests: HTTP requests
- SQLAlchemy: ORM para bancos de dados
- tenacity: Retry logic
- PyYAML: Parsing YAML
- typing-extensions: Extensões de tipagem

## Instalação
Para instalar todas as dependências:
```
pip install -r requirements.txt
```

## Requisitos Externos
- **Ollama**: Deve estar instalado e rodando
  - Modelos necessários: `llama3.1:8b`, `nomic-embed-text`
  - Comando para baixar: `ollama pull llama3.1:8b && ollama pull nomic-embed-text`

## Ambiente Virtual
Recomenda-se usar um ambiente virtual:
```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r dependencies/requirements.txt
```