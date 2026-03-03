# tests/test_indexer.py
"""
Тесты для Новичка №2 — "The Indexer"
=======================================
Запуск:
  python -m pytest tests/test_indexer.py -v
  # или напрямую:
  python tests/test_indexer.py

Тесты не требуют запущенной БД или ML-модели (используют mock/fallback).
"""

import asyncio
import sys
import os

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ingest_worker.chunker import SemanticChunker, _split_sentences, _cosine_similarity
from services.ingest_worker.topic_helper import TopicHelper

import numpy as np


# ===========================================================================
# Тесты: _split_sentences
# ===========================================================================

def test_split_sentences_basic():
    text = "Стипендия назначается приказом ректора. Размер составляет 10 000 рублей. Выплата — ежемесячно."
    sentences = _split_sentences(text)
    assert len(sentences) >= 2, f"Ожидали >= 2 предложений, получили: {sentences}"
    print(f"  ✅ test_split_sentences_basic: {sentences}")


def test_split_sentences_empty():
    result = _split_sentences("")
    assert result == [], f"Ожидали [], получили: {result}"
    print("  ✅ test_split_sentences_empty")


def test_split_sentences_paragraphs():
    text = "Первый абзац с информацией.\n\nВторой абзац с другой темой. Ещё одно предложение."
    result = _split_sentences(text)
    assert len(result) >= 2
    print(f"  ✅ test_split_sentences_paragraphs: {len(result)} предложений")


# ===========================================================================
# Тесты: _cosine_similarity
# ===========================================================================

def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0])
    sim = _cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-6
    print("  ✅ test_cosine_similarity_identical")


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    sim = _cosine_similarity(a, b)
    assert abs(sim) < 1e-6
    print("  ✅ test_cosine_similarity_orthogonal")


def test_cosine_similarity_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    sim = _cosine_similarity(a, b)
    assert sim == 0.0
    print("  ✅ test_cosine_similarity_zero_vector")


# ===========================================================================
# Тесты: SemanticChunker (fallback-режим без модели)
# ===========================================================================

def test_chunker_fallback_basic():
    """Chunker без модели (fallback) должен нарезать текст."""
    chunker = SemanticChunker(embedder=None, max_tokens=50, min_tokens=5)
    text = " ".join([f"Предложение {i} о стипендиях МИЭМ." for i in range(30)])
    chunks = asyncio.run(chunker.create_chunks(text, "Тест", topic_breadcrumb=None))
    assert len(chunks) >= 2, f"Должно быть >= 2 чанков, получили: {len(chunks)}"
    print(f"  ✅ test_chunker_fallback_basic: {len(chunks)} чанков")


def test_chunker_header_injection():
    """Каждый чанк должен начинаться с имени документа."""
    chunker = SemanticChunker(embedder=None)
    text = "Студент получает стипендию ежемесячно. Размер зависит от успеваемости."
    chunks = asyncio.run(chunker.create_chunks(text, "Положение о стипендиях", topic_breadcrumb=None))
    for c in chunks:
        assert "Документ: Положение о стипендиях" in c["text"], f"Header не найден: {c['text'][:80]}"
    print(f"  ✅ test_chunker_header_injection: {len(chunks)} чанков с заголовком")


def test_chunker_breadcrumb_injection():
    """Если breadcrumb передан, он должен быть в тексте чанка."""
    chunker = SemanticChunker(embedder=None)
    text = "Правительственная стипендия назначается лучшим студентам страны согласно приказу."
    breadcrumb = "Про деньги > Стипендии > Правительства РФ"
    chunks = asyncio.run(chunker.create_chunks(text, "Стипендии", topic_breadcrumb=breadcrumb))
    for c in chunks:
        assert breadcrumb in c["text"], f"Breadcrumb не найден: {c['text'][:100]}"
    print(f"  ✅ test_chunker_breadcrumb_injection: breadcrumb вклеен")


def test_chunker_empty_text():
    """Пустой текст → пустой список."""
    chunker = SemanticChunker(embedder=None)
    chunks = asyncio.run(chunker.create_chunks("", "Документ"))
    assert chunks == [], f"Ожидали [], получили: {chunks}"
    print("  ✅ test_chunker_empty_text")


def test_chunker_meta_fields():
    """Проверяем наличие обязательных полей в метаданных."""
    chunker = SemanticChunker(embedder=None)
    text = " ".join([f"Предложение номер {i}." for i in range(20)])
    chunks = asyncio.run(chunker.create_chunks(text, "Мой документ", topic_breadcrumb="A > B"))
    required_keys = {"chunk_index", "doc_title", "topic_breadcrumb", "token_count"}
    for c in chunks:
        missing = required_keys - set(c["meta"].keys())
        assert not missing, f"Отсутствуют ключи в meta: {missing}"
    print(f"  ✅ test_chunker_meta_fields: все поля присутствуют в {len(chunks)} чанках")


# ===========================================================================
# Тесты: TopicHelper
# ===========================================================================

SAMPLE_RECORDS = [
    {"topic_id": 1, "parent_id": None, "name": "Про деньги"},
    {"topic_id": 2, "parent_id": 1,    "name": "Стипендии"},
    {"topic_id": 3, "parent_id": 2,    "name": "Правительства РФ"},
    {"topic_id": 4, "parent_id": 1,    "name": "Льготы"},
]


def test_topic_helper_breadcrumb():
    helper = TopicHelper()
    helper.load_from_records(SAMPLE_RECORDS)
    result = helper.get_breadcrumb(3)
    assert result == "Про деньги > Стипендии > Правительства РФ", f"Неверный breadcrumb: {result}"
    print(f"  ✅ test_topic_helper_breadcrumb: '{result}'")


def test_topic_helper_root():
    helper = TopicHelper()
    helper.load_from_records(SAMPLE_RECORDS)
    result = helper.get_breadcrumb(1)
    assert result == "Про деньги", f"Неверный breadcrumb: {result}"
    print(f"  ✅ test_topic_helper_root: '{result}'")


def test_topic_helper_none_id():
    helper = TopicHelper()
    helper.load_from_records(SAMPLE_RECORDS)
    result = helper.get_breadcrumb(None)
    assert result is None
    print("  ✅ test_topic_helper_none_id")


def test_topic_helper_unknown_id():
    helper = TopicHelper()
    helper.load_from_records(SAMPLE_RECORDS)
    result = helper.get_breadcrumb(999)
    assert result is None
    print("  ✅ test_topic_helper_unknown_id")


def test_topic_helper_sibling():
    helper = TopicHelper()
    helper.load_from_records(SAMPLE_RECORDS)
    result = helper.get_breadcrumb(4)
    assert result == "Про деньги > Льготы", f"Неверный breadcrumb: {result}"
    print(f"  ✅ test_topic_helper_sibling: '{result}'")


def test_topic_helper_not_loaded():
    helper = TopicHelper()
    assert not helper.is_loaded()
    helper.load_from_records(SAMPLE_RECORDS)
    assert helper.is_loaded()
    print("  ✅ test_topic_helper_not_loaded")


# ===========================================================================
# Точка запуска
# ===========================================================================

if __name__ == "__main__":
    tests = [
        # _split_sentences
        test_split_sentences_basic,
        test_split_sentences_empty,
        test_split_sentences_paragraphs,
        # _cosine_similarity
        test_cosine_similarity_identical,
        test_cosine_similarity_orthogonal,
        test_cosine_similarity_zero_vector,
        # SemanticChunker
        test_chunker_fallback_basic,
        test_chunker_header_injection,
        test_chunker_breadcrumb_injection,
        test_chunker_empty_text,
        test_chunker_meta_fields,
        # TopicHelper
        test_topic_helper_breadcrumb,
        test_topic_helper_root,
        test_topic_helper_none_id,
        test_topic_helper_unknown_id,
        test_topic_helper_sibling,
        test_topic_helper_not_loaded,
    ]

    print("\n" + "="*60)
    print("  Тесты Новичка №2 — The Indexer")
    print("="*60)
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"  ❌ {t.__name__}: {exc}")
            failed += 1

    print("="*60)
    print(f"  Итого: {passed} пройдено, {failed} провалено")
    print("="*60 + "\n")
    sys.exit(0 if failed == 0 else 1)
