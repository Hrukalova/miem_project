# services/api/retrieval.py
"""
Vector Retrieval через pgvector
=================================
Принимает вектор запроса и возвращает top-K наиболее близких чанков.

Использует оператор pgvector  <=>  (cosine distance):
  distance = embedding <=> query_vector
  similarity = 1 - distance

Требования к таблице chunks (из core/models.py):
  id, document_id, text, embedding (Vector(1024)), chunk_metadata
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

TOP_K_DEFAULT = 5


async def vector_search(
    session: AsyncSession,
    query_embedding: List[float],
    top_k: int = TOP_K_DEFAULT,
) -> List[dict]:
    """
    Выполняет ANN-поиск в pgvector по косинусному расстоянию.

    Параметры
    ----------
    session         : активная AsyncSession SQLAlchemy
    query_embedding : вектор вопроса (list[float], dim=1024)
    top_k           : сколько чанков вернуть (default=5)

    Возвращает
    ----------
    Список словарей:
        {
            "chunk_id"   : UUID,
            "text"       : str,        # текст чанка (уже с Header Injection)
            "similarity" : float,      # косинусное сходство [0..1]
            "metadata"   : dict,       # chunk_metadata из БД
        }
    """
    # pgvector: <=> это cosine_distance, поэтому similarity = 1 - distance
    # ::vector приводит JSON-массив к типу vector
    sql = text("""
        SELECT
            id::text          AS chunk_id,
            text              AS text,
            chunk_metadata    AS metadata,
            1 - (embedding <=> CAST(:vec AS vector)) AS similarity
        FROM chunks
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)

    # Преодразуем вектор в строку формата [1.1,2.2,...] для pgvector
    vec_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"

    result = await session.execute(sql, {"vec": vec_str, "k": top_k})
    rows = result.mappings().all()

    return [
        {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "similarity": float(row["similarity"]),
            "metadata": row["metadata"] or {},
        }
        for row in rows
    ]
