# rag_service/retrieval.py
"""
Vector Retrieval Service: library schema + Profile Filtering
=============================================================
Implementation of the core retrieval logic including:
- Joining library.chunk_embeddings, library.chunks, and library.documents
- Filtering by document status (published) and ingest status
- Filtering by user profile (university, campus, faculty, program, year, role)
  based on documents.scope_json.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("Retrieval")

TOP_K_DEFAULT = 5


async def vector_search(
    session: AsyncSession,
    query_embedding: List[float],
    user_profile: Dict[str, Any],
    top_k: int = TOP_K_DEFAULT,
) -> List[Dict[str, Any]]:
    """
    Performs vector search in library schema with profile scoping.
    
    :param user_profile: dict containing university_id, campus_id, etc.
    """
    
    # SQL query joining library tables and applying filters
    # User profile fields:
    # university_id, campus_id, faculty_id, program_id, year, role
    
    # scope_json check logic:
    # (field is MISSING OR EMPTY) OR (user_value IS IN array)
    
    sql = text("""
        SELECT
            c.id::text                         AS chunk_id,
            c.text                             AS text,
            c.metadata_json                    AS chunk_metadata,
            d.id::text                         AS document_id,
            d.title                            AS document_title,
            d.scope_json                       AS doc_scope,
            1 - (e.embedding <=> CAST(:vec AS vector)) AS similarity
        FROM library.chunk_embeddings e
        JOIN library.chunks c     ON c.id = e.chunk_id
        JOIN library.documents d  ON d.id = c.document_id
        WHERE d.status = 'published'
          AND d.ingest_status = 'indexed'
          
          -- Profile Scoping Filters
          AND (
              d.scope_json = '{}'::jsonb OR
              (
                  (d.scope_json -> 'university_ids' IS NULL OR jsonb_array_length(d.scope_json -> 'university_ids') = 0 OR d.scope_json -> 'university_ids' @> jsonb_build_array(:u_id))
                  AND (d.scope_json -> 'campus_ids' IS NULL OR jsonb_array_length(d.scope_json -> 'campus_ids') = 0 OR d.scope_json -> 'campus_ids' @> jsonb_build_array(:c_id))
                  AND (d.scope_json -> 'faculty_ids' IS NULL OR jsonb_array_length(d.scope_json -> 'faculty_ids') = 0 OR d.scope_json -> 'faculty_ids' @> jsonb_build_array(:f_id))
                  AND (d.scope_json -> 'program_ids' IS NULL OR jsonb_array_length(d.scope_json -> 'program_ids') = 0 OR d.scope_json -> 'program_ids' @> jsonb_build_array(:p_id))
                  AND (d.scope_json -> 'years' IS NULL OR jsonb_array_length(d.scope_json -> 'years') = 0 OR d.scope_json -> 'years' @> jsonb_build_array(:year))
                  AND (d.scope_json -> 'roles' IS NULL OR jsonb_array_length(d.scope_json -> 'roles') = 0 OR d.scope_json -> 'roles' @> jsonb_build_array(:role))
              )
          )
        ORDER BY e.embedding <=> CAST(:vec AS vector)
        LIMIT :k
    """)

    # Prepare vector string
    vec_str = "[" + ",".join(f"{x:.8f}" for x in query_embedding) + "]"
    
    # Extract params from profile
    params = {
        "vec": vec_str,
        "k": top_k,
        "u_id": str(user_profile.get("university_id")) if user_profile.get("university_id") else None,
        "c_id": str(user_profile.get("campus_id")) if user_profile.get("campus_id") else None,
        "f_id": str(user_profile.get("faculty_id")) if user_profile.get("faculty_id") else None,
        "p_id": str(user_profile.get("program_id")) if user_profile.get("program_id") else None,
        "year": user_profile.get("year"),
        "role": user_profile.get("role"),
    }

    try:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "similarity": float(row["similarity"]),
                "metadata": row["chunk_metadata"] or {},
                "document_id": row["document_id"],
                "document_title": row["document_title"]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Database retrieval error: {e}")
        return []
