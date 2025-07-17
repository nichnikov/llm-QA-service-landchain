# main.py

import uuid
from fastapi import FastAPI, HTTPException, Depends
from typing import Annotated

from core.data_types import QueryRequest, AnswerResponse, Settings, Parameters, PromtsChain
from core.callbacks import FileCallbackHandler
from chains.classifier import get_classifier_chain
from chains.qa_chain import get_qa_chain

# --- Создание зависимостей ---

def get_settings() -> Settings:
    return Settings()

def get_parameters() -> Parameters:
    return Parameters()

def get_prompts() -> PromtsChain:
    PROMPTS_FILE_PATH = "configs/prompts.json"
    return PromtsChain.from_file(PROMPTS_FILE_PATH)

# Создаем один раз и кешируем
prompts = get_prompts()
settings = get_settings()
parameters = get_parameters()

# Создаем цепочки, которые будут использоваться в эндпоинте
classifier_chain = get_classifier_chain(prompts, settings, parameters)
qa_chain_with_voting = get_qa_chain(prompts, settings, parameters, voting_enabled=True, queries_generate=False)


# Создаем экземпляр FastAPI
app = FastAPI(title="LangChain QA Service")


@app.post("/expert_bot/", response_model=AnswerResponse)
async def process_query(request: QueryRequest):
    """
    Основной эндпоинт для обработки запросов пользователя.
    """
    query = request.query
    alias = request.alias
    
    # --- Логирование и трассировка ---
    run_id = uuid.uuid4()
    file_callback = FileCallbackHandler(memory_path=parameters.memory_path, query=query)
    config = {"callbacks": [file_callback], "run_id": run_id}

    # --- 1. Классификация запроса ---
    query_type = await classifier_chain.ainvoke({"query": query}, config=config)

    # --- 2. Маршрутизация ---
    answ_dict = {
        1: "Рады приветствовать вас на нашем сайте",
        2: "Рады, что смогли вам помочь",
    }
    if query_type in answ_dict:
        return AnswerResponse(answer=answ_dict[query_type], answer_text=answ_dict[query_type], run_id=str(run_id))

    if query_type in [3, 4]: # Бухгалтерский вопрос
        # Вызываем основной конвейер
        result = await qa_chain_with_voting.ainvoke({"query": query, "alias": alias}, config=config)
        answer_text = result.get("final_answer", "НЕТ ОТВЕТА")
    else: # Другое
        answer_text = "Не удалось определить тип вашего запроса. Пожалуйста, переформулируйте его."

    if not answer_text or answer_text == "НЕТ ОТВЕТА":
        raise HTTPException(status_code=404, detail="No answer found")

    return AnswerResponse(answer=answer_text, answer_text=answer_text, run_id=str(run_id))

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)