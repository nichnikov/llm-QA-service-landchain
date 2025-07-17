# services/retriever.py

from typing import List, Dict, Any
import aiohttp
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

class CustomApiRetriever(BaseRetriever):
    """
    Кастомный ретривер для обращения к вашему внутреннему API поиска.
    """
    base_url: str
    endpoint: str
    alias: str

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Синхронная версия не используется, но должна быть реализована
        raise NotImplementedError("CustomApiRetriever does not support synchronous calls.")

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Асинхронно получает документы из вашего поискового API.
        """
        url = f"{self.base_url.rstrip('/')}{self.endpoint}"
        request_body = {"query": query, "alias": self.alias}
        
        documents = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_body, headers={"Authorization": "Bearer token123"}, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Преобразуем ответ API в список объектов Document
                        for d in data.get("ranking_dicts", []):
                            # Объединяем лучшие фрагменты в один текст для page_content
                            content = "\n\n".join([f[0] for f in d.get("best_fragments_scores", [])])
                            metadata = {
                                "source": d.get("link", ""),
                                "title": d.get("title", ""),
                                "doc_id": d.get("doc_id"),
                                "mod_id": d.get("mod_id")
                            }
                            documents.append(Document(page_content=content, metadata=metadata))
                    else:
                        # Логирование или обработка ошибок
                        print(f"Error fetching documents: {response.status}")
        except Exception as e:
            print(f"An exception occurred in retriever: {e}")
        
        return documents