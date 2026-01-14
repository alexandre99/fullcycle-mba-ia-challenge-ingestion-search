import os_util
from factory import get_llm, get_embeddings, get_vector_store
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA À "PERGUNTA DO USUÁRIO" EM FRASE COMPLETA E NATURAL
"""

def retrieve_context(question: str) -> str:
    embeddings = get_embeddings()
    store = get_vector_store(embeddings)
    results = store.similarity_search_with_score(question, k=10)
    return "\n".join([doc.page_content for doc, _ in results])

def generate_answer(question: str, contexto: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": contexto})

def search_prompt(question: str) -> str:
    contexto = retrieve_context(question)
    return generate_answer(question, contexto)
