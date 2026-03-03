from sqlalchemy import text

async def hybrid_search(db_session, query_text, query_vector, limit=5):
    # Упрощенный гибридный поиск через SQL
    sql = text("""
        SELECT c.id, c.content,
               (1 - (ce.embedding <=> :vector)) as similarity
        FROM chunks c
        JOIN chunk_embeddings ce ON c.id = ce.chunk_id
        ORDER BY similarity DESC
        LIMIT :limit
    """)
    result = await db_session.execute(sql, {"vector": str(query_vector), "limit": limit})
    return result.all()
