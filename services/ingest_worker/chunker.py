# services/ingest_worker/chunker.py
from typing import List, Dict

class SemanticChunker:
    def __init__(self):
        # Здесь напарник инициализирует свою модель из Kaggle
        # self.tokenizer = ...
        pass

    async def create_chunks(self, text: str, doc_title: str) -> List[Dict]:
        """
        Вход: чистый текст и название документа.
        Выход: список словарей с текстом чанка и метаданными.
        """
        # ТУТ БУДЕТ МАГИЯ ИЗ KAGGLE (Semantic Splitting)
        # А пока — простая нарезка для теста:
        words = text.split()
        simple_chunks = []
        for i in range(0, len(words), 100):
            chunk_text = " ".join(words[i:i+100])
            # Header Injection: вклеиваем название документа
            final_text = f"Документ: {doc_title}\nКонтент: {chunk_text}"
            simple_chunks.append({
                "text": final_text,
                "meta": {"start_word": i}
            })
        return simple_chunks
