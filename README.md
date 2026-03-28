# AS_26
Agent Systems HSE & VK 2026 course

Умный ассистент с характером

CLI чат-бот на LangChain с классификацией запросов, роутингом, памятью и сменой характера.

## Запуск

```bash
uv sync
cp .env.example .env  # add your OPENAI_API_KEY (OpenRouter.ai)
uv run python main.py
uv run python main.py --character sarcastic --memory summary
```

## Команды

- `/character <name>` — сменить характер (friendly, professional, sarcastic, pirate)
- `/memory <type>` — сменить память (buffer, summary)
- `/clear` — очистить историю
- `/status` — текущие настройки
- `/help` — справка
- `/quit` — выход

## Пример

```
➜  AS_26 git:(master) ✗ uv run python main.py

🤖 Умный ассистент с характером
Характер: friendly | Память: buffer
────────────────────────────────────────
Введите /help для справки

> Привет! Меня зовут Алексей
[small_talk] Привет, Алексей! 😊  
Счастлив, что ты заходил! Чем я могу тебя помочь сегодня? 🌟
confidence: 1.00 | tokens: ~20

> /character sarcastic
✓ Характер изменён на: sarcastic
>  Как меня зовут?
[question] 

О,Алексей, ты спрашиваешь, как тебя звать? Я вроде уже сказал, что тебя зовут Алексей, но если ты хочешь, чтобы я тебя звал по-другому, то... ну, я не умею меняться. Но если хочешь, я могу называть тебя "господин Алексей" или "милый юный пользователь". Или просто "Алексей" — это тоже нормально. Что тебе больше нравится?  

Чем я могу тебе помочь сегодня?
confidence: 0.95 | tokens: ~89

> ^C
До свидания!
```
