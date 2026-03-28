"""Тесты для умного ассистента."""

import pytest
from pydantic import ValidationError

from main import (
    AssistantResponse,
    Classification,
    MemoryManager,
    RequestType,
    CHARACTER_PROMPTS,
    HANDLER_INSTRUCTIONS,
    build_handlers,
)


def test_request_type_values():
    assert RequestType.QUESTION.value == "question"
    assert RequestType.SMALL_TALK.value == "small_talk"
    assert len(RequestType) == 5


def test_classification_valid():
    c = Classification(request_type=RequestType.QUESTION, confidence=0.95, reasoning="Это вопрос")
    assert c.request_type == RequestType.QUESTION
    assert c.confidence == 0.95


def test_classification_invalid_confidence():
    with pytest.raises(ValidationError):
        Classification(request_type=RequestType.QUESTION, confidence=1.5, reasoning="test")
    with pytest.raises(ValidationError):
        Classification(request_type=RequestType.QUESTION, confidence=-0.1, reasoning="test")


def test_assistant_response():
    r = AssistantResponse(content="Привет!", request_type=RequestType.SMALL_TALK, confidence=0.9, tokens_used=10)
    assert r.content == "Привет!"
    assert r.tokens_used == 10


def test_memory_buffer():
    mem = MemoryManager(strategy="buffer", max_messages=4)
    mem.add("msg1", "reply1")
    mem.add("msg2", "reply2")
    mem.add("msg3", "reply3")
    assert mem.count == 4  # trimmed to last 4 (2 pairs)
    history = mem.get_history()
    assert history[0].content == "msg2"
    assert history[-1].content == "reply3"


def test_memory_clear():
    mem = MemoryManager(strategy="buffer")
    mem.add("hello", "hi")
    assert mem.count == 2
    mem.clear()
    assert mem.count == 0
    assert mem.get_history() == []


def test_all_request_types_have_instructions():
    for rt in RequestType:
        assert rt in HANDLER_INSTRUCTIONS


def test_all_characters_defined():
    assert set(CHARACTER_PROMPTS.keys()) == {"friendly", "professional", "sarcastic", "pirate"}
