# core/llm_clients.py

from langchain_openai import ChatOpenAI
from core.data_types import Settings, Parameters

def get_llm(settings: Settings, parameters: Parameters, model_name: str, temperature: float) -> ChatOpenAI:
    """
    Создает и настраивает экземпляр LLM клиента.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=settings.openai_api_key,
        base_url="https://api.vsegpt.ru:7090/v1" # Ваш кастомный URL
    )