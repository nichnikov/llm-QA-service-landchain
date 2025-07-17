# chains/classifier.py

import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm_clients import get_llm
from core.data_types import PromtsChain, Settings, Parameters

def get_classifier_chain(prompts: PromtsChain, settings: Settings, parameters: Parameters):
    """
    Создает цепочку для классификации запроса.
    """
    llm = get_llm(settings, parameters, parameters.ai_model_classifier, 0.5)
    
    prompt = ChatPromptTemplate.from_template(prompts.classication)
    
    # Парсер для извлечения только цифры из ответа
    def _parse_classification(text: str) -> int:
        match = re.search(r"\d", text)
        if not match:
            return 3 # По умолчанию считаем, что это вопрос, требующий поиска
        return int(match.group(0))

    # Собираем цепочку
    chain = prompt | llm | StrOutputParser() | _parse_classification
    
    return chain