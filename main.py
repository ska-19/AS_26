"""Умный ассистент с характером — CLI чат-бот на LangChain."""

import argparse
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
FALLBACK_MODEL = "google/gemma-3-4b-it:free"


# ── Часть 1: Pydantic-модели ──


class RequestType(str, Enum):
    """Перечисление типов пользовательских запросов."""
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class Classification(BaseModel):
    """Результат классификации запроса."""
    request_type: RequestType = Field(description="Тип запроса")
    confidence: float = Field(ge=0, le=1, description="Уверенность классификации")
    reasoning: str = Field(description="Краткое обоснование")


class AssistantResponse(BaseModel):
    """Итоговый ответ бота с метаданными."""
    content: str
    request_type: RequestType
    confidence: float
    tokens_used: int


# ── Часть 4: Характеры ──

CHARACTER_PROMPTS = {
    "friendly": "Ты — дружелюбный ассистент. Общаешься тепло и позитивно, можешь использовать эмодзи.",
    "professional": "Ты — деловой ассистент. Отвечаешь сдержанно, по делу, без панибратства.",
    "sarcastic": "Ты — саркастичный ассистент. Отвечаешь с лёгкой иронией и юмором, но по делу.",
    "pirate": "Ты — пират-ассистент. Говоришь как пират: «Арр!», «Тысяча чертей!», называешь пользователя «матрос».",
}

HANDLER_INSTRUCTIONS = {
    RequestType.QUESTION: "Это вопрос пользователя. Дай информативный и полезный ответ. Если не знаешь — честно скажи.",
    RequestType.TASK: "Пользователь просит выполнить задачу. Сделай качественно.",
    RequestType.SMALL_TALK: "Поддержи беседу, будь приветлив. Если пользователь представился — запомни имя.",
    RequestType.COMPLAINT: "Пользователь жалуется. Прояви эмпатию, попробуй понять проблему и предложить решение.",
    RequestType.UNKNOWN: "Запрос непонятен. Вежливо попроси уточнить, что имел в виду пользователь.",
}


# ── Часть 2: Классификатор ──

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — классификатор запросов. Определи тип запроса пользователя.

Типы:
- question — вопрос, требующий информации
- task — просьба что-то сделать
- small_talk — приветствие, болтовня
- complaint — жалоба, недовольство
- unknown — бессмыслица или нераспознанный запрос

Примеры:
"Привет!" → small_talk
"Как дела?" → small_talk
"Меня зовут Алексей" → small_talk
"Что такое Python?" → question
"Как работает GIL?" → question
"Напиши стих про кота" → task
"Расскажи анекдот" → task
"Это ужасно работает!" → complaint
"Почему так долго отвечаешь?" → complaint
"asdfghjkl" → unknown

{format_instructions}"""),
    ("human", "Запрос: {query}"),
])


def build_classifier(model):
    """LCEL-цепочка: вход → промпт → модель → PydanticOutputParser → Classification."""
    parser = PydanticOutputParser(pydantic_object=Classification)
    chain = (
        {"query": RunnablePassthrough(), "format_instructions": lambda _: parser.get_format_instructions()}
        | CLASSIFICATION_PROMPT
        | model
        | parser
    )

    def classify(query: str) -> Classification:
        try:
            return chain.invoke(query)
        except Exception:
            return Classification(
                request_type=RequestType.UNKNOWN,
                confidence=0.5,
                reasoning="Ошибка парсинга ответа модели",
            )

    return classify


# ── Часть 3: Обработчики и роутинг ──


def build_handler(model, request_type: RequestType, character: str):
    system = CHARACTER_PROMPTS[character] + "\n\n" + HANDLER_INSTRUCTIONS[request_type]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    return prompt | model | StrOutputParser()


def build_handlers(model, character: str):
    return {rt: build_handler(model, rt, character) for rt in RequestType}


# ── Часть 5: Память ──


class MemoryManager:
    """Хранит историю диалога. Стратегии: buffer (последние N) и summary (суммаризация старых)."""

    def __init__(self, strategy: str = "buffer", model=None, max_messages: int = 20):
        self.strategy = strategy
        self.model = model
        self.max_messages = max_messages
        self.messages: list = []
        self.summary: str | None = None

    def get_history(self) -> list:
        if self.strategy == "summary" and self.summary:
            return [SystemMessage(content=f"Краткое содержание предыдущего разговора: {self.summary}")] + self.messages
        return self.messages

    def add(self, human_text: str, ai_text: str):
        self.messages.append(HumanMessage(content=human_text))
        self.messages.append(AIMessage(content=ai_text))
        self._trim()

    def _trim(self):
        if len(self.messages) <= self.max_messages:
            return
        if self.strategy == "buffer":
            self.messages = self.messages[-self.max_messages:]
        elif self.strategy == "summary" and self.model:
            old = self.messages[:-self.max_messages]
            self.messages = self.messages[-self.max_messages:]
            self._summarize(old)

    def _summarize(self, old_messages: list):
        text = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in old_messages)
        prev = f"Предыдущее резюме: {self.summary}\n\n" if self.summary else ""
        prompt = f"{prev}Сделай краткое содержание этого диалога, сохрани ключевые факты (имена, предпочтения):\n\n{text}"
        self.summary = self.model.invoke([HumanMessage(content=prompt)]).content

    def clear(self):
        self.messages.clear()
        self.summary = None

    @property
    def count(self) -> int:
        return len(self.messages)


# ── Модель с fallback ──


def build_model(model_name: str):
    """Основная модель + fallback на запасную через .with_fallbacks()."""
    main_model = ChatOpenAI(model=model_name, temperature=0.7, base_url=OPENROUTER_BASE_URL)
    fallback_model = ChatOpenAI(model=FALLBACK_MODEL, temperature=0.7, base_url=OPENROUTER_BASE_URL)
    return main_model.with_fallbacks([fallback_model])


# ── Часть 6: CLI + стриминг ──


class SmartAssistant:
    """Классификация → роутинг → ответ (с памятью и характером)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, character: str = "friendly", memory_strategy: str = "buffer"):
        self.model_name = model_name
        self.model = build_model(model_name)
        self.character = character
        self.memory = MemoryManager(strategy=memory_strategy, model=self.model)
        self.classify = build_classifier(self.model)
        self.handlers = build_handlers(self.model, self.character)

    def set_character(self, character: str):
        self.character = character
        self.handlers = build_handlers(self.model, self.character)

    def set_memory_strategy(self, strategy: str):
        self.memory.strategy = strategy

    def process(self, user_input: str) -> AssistantResponse:
        classification = self.classify(user_input)
        handler = self.handlers[classification.request_type]
        response_text = handler.invoke({"query": user_input, "history": self.memory.get_history()})
        self.memory.add(user_input, response_text)
        return AssistantResponse(
            content=response_text,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=len(response_text) // 4,
        )

    def process_stream(self, user_input: str):
        classification = self.classify(user_input)
        handler = self.handlers[classification.request_type]
        print(f"[{classification.request_type.value}] ", end="", flush=True)
        chunks = []
        for chunk in handler.stream({"query": user_input, "history": self.memory.get_history()}):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        response_text = "".join(chunks)
        self.memory.add(user_input, response_text)
        print(f"\nconfidence: {classification.confidence:.2f} | tokens: ~{len(response_text) // 4}\n")


def main():
    parser = argparse.ArgumentParser(description="Умный ассистент с характером")
    parser.add_argument("--character", default="friendly", choices=CHARACTER_PROMPTS.keys())
    parser.add_argument("--memory", default="buffer", choices=["buffer", "summary"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    set_llm_cache(InMemoryCache())
    assistant = SmartAssistant(model_name=args.model, character=args.character, memory_strategy=args.memory)

    print("🤖 Умный ассистент с характером")
    print(f"Характер: {assistant.character} | Память: {assistant.memory.strategy}")
    print("─" * 40)
    print("Введите /help для справки\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("До свидания!")
                break
            elif cmd == "/clear":
                assistant.memory.clear()
                print("✓ История очищена")
            elif cmd == "/character":
                name = parts[1].strip() if len(parts) > 1 else ""
                if name in CHARACTER_PROMPTS:
                    assistant.set_character(name)
                    print(f"✓ Характер изменён на: {name}")
                else:
                    print(f"✗ Доступные характеры: {', '.join(CHARACTER_PROMPTS.keys())}")
            elif cmd == "/memory":
                strategy = parts[1].strip() if len(parts) > 1 else ""
                if strategy in ("buffer", "summary"):
                    assistant.set_memory_strategy(strategy)
                    print(f"✓ Стратегия памяти: {strategy}")
                else:
                    print("✗ Доступные стратегии: buffer, summary")
            elif cmd == "/status":
                print(f"Характер: {assistant.character}")
                print(f"Память: {assistant.memory.strategy} ({assistant.memory.count} сообщений)")
                print(f"Модель: {assistant.model_name}")
            elif cmd == "/help":
                print("/clear            — очистить историю")
                print("/character <name> — сменить характер (friendly, professional, sarcastic, pirate)")
                print("/memory <type>    — сменить память (buffer, summary)")
                print("/status           — текущие настройки")
                print("/help             — эта справка")
                print("/quit             — выход")
            else:
                print("✗ Неизвестная команда. Введите /help")
            continue

        if args.no_stream:
            result = assistant.process(user_input)
            print(f"[{result.request_type.value}] {result.content}")
            print(f"confidence: {result.confidence:.2f} | tokens: ~{result.tokens_used}\n")
        else:
            assistant.process_stream(user_input)


if __name__ == "__main__":
    main()
