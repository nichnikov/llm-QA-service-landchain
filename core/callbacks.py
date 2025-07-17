# core/callbacks.py

import os
import json
import uuid
import datetime
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class FileCallbackHandler(BaseCallbackHandler):
    """
    Обработчик, который записывает полную трассировку выполнения цепочки в JSON-файл.
    """
    def __init__(self, memory_path: str, query: str):
        if not os.path.exists(memory_path):
            os.makedirs(memory_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        query_part = re.sub(r'[\\/*?:"<>|]', "", query)[:50].replace(" ", "_")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{query_part}_{unique_id}.json"
        
        self.log_file_path = os.path.join(memory_path, filename)
        self.run_data = {"initial_query": query, "steps": []}

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.run_data["steps"].append({
            "type": "llm_start",
            "prompts": prompts,
            "llm_config": serialized,
        })

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.run_data["steps"].append({
            "type": "llm_end",
            "response": response.generations[0][0].text
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Вызывается в конце выполнения всей цепочки."""
        self.run_data["final_output"] = outputs
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.run_data, f, ensure_ascii=False, indent=4, default=str)
        except Exception as e:
            print(f"Failed to write log to {self.log_file_path}: {e}")