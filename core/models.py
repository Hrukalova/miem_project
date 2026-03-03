# core/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector # Убедись, что pip install pgvector сделан
import uuid
from .db import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(512), nullable=False)
    source_type = Column(String(50))  # 'file', 'url', 'faq'
    raw_content_link = Column(String(1024), nullable=True)
    content_text = Column(Text, nullable=True)

    status = Column(String(20), default="pending")

    # ПЕРЕИМЕНОВАЛИ ТУТ:
    doc_metadata = Column(JSONB, default={})

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))

    chunk_metadata = Column(JSONB, default={}) # И тут на всякий случай переименовал

# Индекс HNSW для pgvector
Index(
    "idx_chunks_embedding_hnsw",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
