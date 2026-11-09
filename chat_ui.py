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
        return "⚠️ Введите GigaChat API-ключ в поле выше перед отправкой сообщения."
    
    # Формируем контекст из истории чата
    context_messages = []
    if history:
        for msg in history:
            if isinstance(msg, dict):
                # Формат Gradio 6: {"role": "user", "content": "..."}
                if msg.get("role") == "user":
                    context_messages.append(f"Пользователь: {msg.get('content', '')}")
                elif msg.get("role") == "assistant":
                    context_messages.append(f"Агент: {msg.get('content', '')}")
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                # Формат (user_msg, assistant_msg)
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
        "context": context,  # передаём историю
        "api_key": api_key
    }
    
    try:
        resp = httpx.post(API_URL, json=payload, timeout=120)
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
    demo = gr.ChatInterface(
        chat,
        title="🌍 Travel Planner — Планировщик путешествий",
        description="""
        **ИИ-агент для планирования путешествий**
        
        Агент учитывает:
        - 🌤️ Погоду (Open-Meteo API)
        - 💱 Курсы валют (exchangerate-api.com)
        - 🗺️ LLM-генерация плана с бюджетом, жильём и транспортом
        
        **Важно:** сначала запустите FastAPI-сервер (см. инструкцию ниже)
        """,
        additional_inputs=[
            gr.Textbox(
                label="🔑 GigaChat API-ключ",
                placeholder="Вставьте ваш GigaChat API-ключ сюда",
                type="password",
            )
        ],
        examples=[
            [["Хочу на горнолыжный курорт в начале апреля из Нижнего Новгорода"]],
            [["Пляжный отдых в феврале, бюджет 70 000 рублей"]],
            [["Тропики и море на неделю, премиум уровень"]],
            [["Бюджетно и недалеко — куда поехать в марте?"]],
            [["Хочу в Иорданию на 5 дней, стандартный комфорт"]],
        ],
    )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
