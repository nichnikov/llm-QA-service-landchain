# chains/qa_chain.py

import re
from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from core.data_types import PromtsChain, Settings, Parameters
from core.llm_clients import get_llm
from services.retriever import CustomApiRetriever

def format_docs(docs: list[Document]) -> str:
    """Форматирует найденные документы для подачи в промпт."""
    return "\n\n".join(f"Заголовок текста: {doc.metadata.get('title', '')} ссылка на текст: {doc.metadata.get('source', '')} Фрагмент: {doc.page_content}" for doc in docs)

def get_qa_chain(prompts: PromtsChain, settings: Settings, parameters: Parameters, voting_enabled: bool, queries_generate: bool):
    """
    Собирает основной конвейер для ответа на вопрос.
    """
    # --- 1. Инициализация LLM-моделей ---
    analisys_llm = get_llm(settings, parameters, parameters.ai_model_analisys_note, 0.1)
    voting_llm = get_llm(settings, parameters, parameters.ai_model_voting, 0.2)
    answer_llm = get_llm(settings, parameters, parameters.ai_model_answer_generator, 0.1)
    query_gen_llm = get_llm(settings, parameters, parameters.ai_model_queries_generate, 1.0)

    # --- 2. Шаг: Генерация поисковых запросов (опционально) ---
    queries_gen_prompt = ChatPromptTemplate.from_template(prompts.query_generation)
    queries_generator_chain = queries_gen_prompt | query_gen_llm | StrOutputParser()

    # --- 3. Шаг: Поиск документов (Retrieval) ---
    def get_retrieved_docs(input_dict: dict) -> list[Document]:
        # Эта функция будет асинхронной, но для простоты цепочки используем sync-обертку
        # В FastAPI это будет работать корректно через chain.ainvoke
        import asyncio
        
        initial_query = input_dict['query']
        generated_queries_text = input_dict.get('generated_queries', '')
        
        queries = [initial_query] + generated_queries_text.split("\n")
        clean_queries = [re.sub(r"Вопрос\d+:", "", q).strip() for q in queries if q.strip()]

        # Создаем ретривер "на лету" с нужным alias
        retriever = CustomApiRetriever(
            base_url=parameters.retrieval_base_url,
            endpoint=parameters.retrieval_endpoint,
            alias=input_dict['alias']
        )
        
        async def fetch_all():
            tasks = [retriever.ainvoke(q) for q in clean_queries]
            results = await asyncio.gather(*tasks)
            # Объединяем и убираем дубликаты
            all_docs = {}
            for doc_list in results:
                for doc in doc_list:
                    all_docs[doc.page_content] = doc # Простая дедупликация по контенту
            return list(all_docs.values())

        return asyncio.run(fetch_all())

    # --- 4. Шаг: Создание аналитической записки ---
    analysis_prompt = ChatPromptTemplate.from_template(prompts.validation_plan)
    analysis_chain = (
        {"query": itemgetter("query"), "best_fragments_str": itemgetter("context") | RunnableLambda(format_docs)}
        | analysis_prompt
        | analisys_llm
        | StrOutputParser()
    )

    # --- 5. Шаг: Голосование (опционально) ---
    def _parse_voting_result(text: str) -> bool:
        return bool(re.search(r"общее\s+мнение:\s+есть\s+ответ", text, re.IGNORECASE))

    voting_prompt = ChatPromptTemplate.from_template(prompts.validation_voting)
    voting_chain = voting_prompt | voting_llm | StrOutputParser() | RunnableLambda(_parse_voting_result)
    
    # --- 6. Шаг: Генерация ответа ---
    answer_prompt_template = prompts.answer_generation if voting_enabled else prompts.answer_generation_with_votin
    answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)
    answer_chain = answer_prompt | answer_llm | StrOutputParser()
    
    # --- 7. Собираем всё в единую цепочку с ветвлением ---

    # Цепочка, которая запускается после получения документов
    processing_chain = RunnablePassthrough.assign(
        analysis_note=analysis_chain
    )
    
    if voting_enabled:
        final_branch = RunnableBranch(
            (
                # Условие для ветки: результат голосования
                RunnablePassthrough.assign(
                    vote_result=lambda x: voting_chain.invoke({
                        "query": x["query"], "analysis_note": x["analysis_note"], "best_fragments": format_docs(x["context"])
                    })
                ) | itemgetter("vote_result"),
                # Если голосование успешно, запускаем генерацию ответа
                RunnablePassthrough.assign(
                    final_answer=lambda x: answer_chain.invoke({
                        "query": x["query"], "analysis_note": x["analysis_note"], "best_fragments": format_docs(x["context"])
                    })
                )
            ),
            # Ветка по умолчанию, если голосование провалилось
            RunnableLambda(lambda x: {"final_answer": "НЕТ ОТВЕТА"})
        )
        processing_chain = processing_chain | final_branch
    else:
        # Без голосования сразу генерируем ответ
        processing_chain = processing_chain.assign(
             final_answer=lambda x: answer_chain.invoke({
                "query": x["query"], "analysis_note": x["analysis_note"], "best_fragments": format_docs(x["context"])
            })
        )

    # Основная цепочка
    if queries_generate:
        full_chain = RunnablePassthrough.assign(generated_queries=queries_generator_chain)
    else:
        full_chain = RunnablePassthrough.assign(generated_queries=lambda x: "")

    full_chain = full_chain | RunnablePassthrough.assign(context=RunnableLambda(get_retrieved_docs)) | processing_chain
    
    return full_chain