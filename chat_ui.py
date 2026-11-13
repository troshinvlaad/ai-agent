"""
Gradio UI для Travel Planner агента.
Запуск: python chat_ui.py
Откроется: http://localhost:7860

Предварительно запустите FastAPI-сервер:
    python -m uvicorn app:app --reload --port 8000
"""

import httpx
import gradio as gr

API_URL = "http://localhost:8000/chat"
PLAN_CREATE_URL = "http://localhost:8000/plan_create"
PLAN_UPDATE_URL = "http://localhost:8000/plan_update"


def chat(message, history, api_key):
    """
    Вызов FastAPI-сервера и возврат ответа агента.
    
    Args:
        message: текст сообщения от пользователя
        history: история чата [{"role": "user", "content": "..."}, ...]
        api_key: ключ GigaChat, введённый пользователем
    
    Returns:
        Ответ агента — ChatInterface сам форматирует в messages
    """
    if not api_key or api_key.strip() == "":
        return "⚠️ Введите GigaChat API-ключ в поле ниже перед отправкой сообщения."
    
    # Определяем команду
    if message.startswith("/plan_create"):
        url = PLAN_CREATE_URL
        payload = {
            "user_id": "local_user",
            "session_id": "local_session",
            "api_key": api_key,
        }
        try:
            import json
            params = json.loads(message.split(None, 1)[1])
            for k, v in params.items():
                payload[k] = v
        except (json.JSONDecodeError, IndexError):
            return "⚠️ Неверный формат JSON. Пример:\n```json\n{\"destination\": \"Сочи\", \"dates\": \"2026-04-15,2026-04-22\", \"budget\": 100000, \"passengers\": 2, \"comfort_level\": \"стандарт\"}\n```"
    
    elif message.startswith("/plan_update"):
        url = PLAN_UPDATE_URL
        payload = {
            "user_id": "local_user",
            "session_id": "local_session",
            "api_key": api_key,
        }
        try:
            import json
            params = json.loads(message.split(None, 1)[1])
            for k, v in params.items():
                payload[k] = v
        except (json.JSONDecodeError, IndexError):
            return "⚠️ Неверный формат JSON. Пример:\n```json\n{\"budget\": 150000, \"passengers\": 3}\n```"
    
    else:
        url = API_URL
        context_messages = []
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        context_messages.append(f"Пользователь: {msg.get('content', '')}")
                    elif msg.get("role") == "assistant":
                        context_messages.append(f"Агент: {msg.get('content', '')}")
                elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                    user_part = str(msg[0]) if msg[0] else ""
                    assistant_part = str(msg[1]) if msg[1] else ""
                    if user_part:
                        context_messages.append(f"Пользователь: {user_part}")
                    if assistant_part:
                        context_messages.append(f"Агент: {assistant_part}")
        
        context = "\n".join(context_messages) if context_messages else "Новый диалог"
        
        payload = {
            "user_id": "local_user",
            "session_id": "local_session",
            "message": message,
            "context": context,
            "api_key": api_key,
        }
    
    try:
        resp = httpx.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        # Приоритет: itinerary (план) > response (текст)
        response = data.get("itinerary") or data.get("response", "Нет ответа от агента")
        return response
        
    except httpx.ConnectError:
        return (
            "❌ Не удалось подключиться к FastAPI-серверу.\n\n"
            "1. Запустите FastAPI-сервер в отдельном терминале:\n"
            "   ```bash\n"
            "   python -m uvicorn app:app --reload --port 8000\n"
            "   ```\n\n"
            "2. Убедитесь, что сервер запущен и порт 8000 свободен."
        )
    except (httpx.ReadTimeout, httpx.TimeoutException):
        return (
            "⏰ Превышено время ожидания.\n\n"
            "Агент обрабатывает запрос — это может занять до 2 минут.\n"
            "Попробуйте ещё раз или уточните запрос (укажите даты, бюджет, направление)."
        )
    except httpx.HTTPStatusError as e:
        return f"⚠️ Ошибка сервера ({e.response.status_code}). Попробуйте ещё раз."
    except Exception as e:
        return f"⚠️ Ошибка: {e}"


if __name__ == "__main__":
    # Подсказки команд при вводе /
    command_hints = [
        "/plan_create",
        "/plan_update",
    ]
    
    demo = gr.ChatInterface(
        chat,
        title="🌍 Travel Planner — Планировщик путешествий",
        description="""
        **ИИ-агент для планирования путешествий**
        
        Агент учитывает:
        - 🌤️ Погоду (Open-Meteo API)
        - 💱 Курсы валют (exchangerate-api.com)
        - 📍 Геокодинг (Nominatim / OpenStreetMap)
        - 🗺️ LLM-генерация плана с бюджетом, жильём и транспортом
        
        **Важно:** сначала запустите FastAPI-сервер
        
        **Команды:**
        - `/plan_create` — создать новый план
        - `/plan_update` — обновить существующий план
        """,
        additional_inputs=[
            gr.Textbox(
                label="🔑 GigaChat API-ключ",
                placeholder="Вставьте ваш GigaChat API-ключ сюда",
                type="password",
            )
        ],
        examples=[],
        textbox=gr.Textbox(
            placeholder="Напишите сообщение или введите / для команд...",
            show_label=False,
        ),
    )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
