import os_util
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def get_llm():
    """
    Factory to get LLM instance.
    """
    provider = os_util.get_env("LLM_PROVIDER", "google").lower()
    
    if provider == "google":
        model = os_util.get_env("GOOGLE_LLM_MODEL")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os_util.get_env("GOOGLE_API_KEY"),
            temperature=0
        )
    elif provider == "openai":
        model =  os_util.get_env("OPENAI_LLM_MODEL")
        return ChatOpenAI(
            model=model,
            openai_api_key=os_util.get_env("OPENAI_API_KEY"),
            temperature=0
        )
    else:
        raise ValueError(f"Provider '{provider}' not supported.")

def get_embeddings():
    """
    Factory to get Embeddings instance.
    """
    provider = os_util.get_env("LLM_PROVIDER").lower()

    if provider == "google":
        model = os_util.get_env("GOOGLE_EMBEDDING_MODEL")
        return GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os_util.get_env("GOOGLE_API_KEY")
        )
    elif provider == "openai":
        model = os_util.get_env("OPENAI_EMBEDDING_MODEL")
        return OpenAIEmbeddings(
            model=model,
            openai_api_key=os_util.get_env("OPENAI_API_KEY")
        )
    elif provider == "openai-openrouter":
        model = os_util.get_env("OPENAI_EMBEDDING_MODEL")
        return OpenAIEmbeddings(
            model=model,
            api_key=os_util.get_env("OPENAI_API_KEY_EMBEDDING_FROM_OPENROUTER_KEY"),
            base_url=os_util.get_env("OPENROUTER_URL"),
            dimensions=1536,
        )  
    else:
        raise ValueError(f"Provider '{provider}' not supported.")
