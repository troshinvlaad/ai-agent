"""
FastAPI-сервис для Travel Planner агента.
Запуск: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated, Literal
from typing_extensions import TypedDict
import json
import os
import httpx
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# --- Конфигурация ---
BASE_URL = "https://foundation-models.api.cloud.ru/v1"
DEFAULT_MODEL = "GigaChat/GigaChat-3-Pro"
REASONING_MODEL = "ai-sage/GigaChat3.5-432B-A28B-Reasoning"

# Ключ можно задать через переменную окружения ИЛИ передать в запросе
GIGACHAT_KEY = os.environ.get("GIGACHAT_KEY", "")

# Глобальный клиент по умолчанию (будет перезаписан при каждом запросе с новым ключом)
client = OpenAI(api_key=GIGACHAT_KEY, base_url=BASE_URL) if GIGACHAT_KEY else None


# --- State ---
class TravelState(TypedDict, total=False):
    user_id: str
    session_id: str
    messages: Annotated[List[dict], add_messages]
    intent: Literal["plan", "refine", "recall", "unknown"]
    destination: Optional[str]
    dates: Optional[str]
    budget: Optional[float]
    comfort_level: Optional[str]
    weather: Optional[str]
    costs: Optional[str]
    weather_ok: Optional[bool]
    budget_ok: Optional[bool]
    alternatives: Optional[str]
    itinerary: Optional[str]
    user_query: str
    agent_response: str
    asked_missing_info: bool  # флаг: уже задавали уточняющие вопросы
    api_key: Optional[str]  # ключ GigaChat для текущего запроса


# --- Tools ---
def get_weather_forecast(location: str, dates: str) -> str:
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=ru"
        with httpx.Client(timeout=10.0) as http:
            geo_resp = http.get(geo_url)
            geo_data = geo_resp.json()
            if "results" not in geo_data or not geo_data["results"]:
                return f"Город '{location}' не найден."
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            name = geo_data["results"][0]["name"]
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
                f"&timezone=auto"
            )
            weather_resp = http.get(weather_url)
            weather_data = weather_resp.json()
            daily = weather_data.get("daily", {})
            temps_max = daily.get("temperature_2m_max", [])
            temps_min = daily.get("temperature_2m_min", [])
            precip = daily.get("precipitation_sum", [])
            codes = daily.get("weathercode", [])
            days = daily.get("time", [])
            if not temps_max:
                return "Сервис погоды недоступен."
            avg_temp = sum(temps_max) / len(temps_max)
            total_precip = sum(precip) if precip else 0
            def weather_desc(code):
                descriptions = {
                    0: "Ясно", 1: "Преимущественно ясно", 2: "Переменная облачность",
                    3: "Пасмурно", 61: "Дождь", 71: "Снег", 95: "Гроза"
                }
                return descriptions.get(code, "Неизвестно")
            result = (
                f"**Погода в {name}**: "
                f"Средняя макс. {avg_temp:.1f}°C, мин. {min(temps_min):.1f}°C, "
                f"осадки {total_precip:.1f} мм, погода: {weather_desc(codes[0] if codes else 0)}"
            )
            return result
    except Exception as e:
        return f"Ошибка погоды: {e}."


def get_travel_costs(location: str, days: int, comfort_level: str) -> str:
    """
    Возвращает placeholder — реальная стоимость формируется LLM в build_itinerary.

    Args:
        location: Название направления.
        days: Количество дней поездки.
        comfort_level: Уровень комфорта.

    Returns:
        Строка-заглушка, стоимость рассчитывается LLM.
    """
    return f"Стоимость поездки в {location} ({days} дн.) будет рассчитана в итоговом плане."


def get_exchange_rates(from_currency="USD", to_currency="RUB") -> str:
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        with httpx.Client(timeout=10.0) as http:
            resp = http.get(url)
            resp.raise_for_status()
            data = resp.json()
            rate = data["rates"].get(to_currency)
            if rate is None:
                return f"Валюта {to_currency} не найдена."
            return f"1 {from_currency} = {rate} {to_currency}"
    except Exception as e:
        return f"Ошибка курса: {e}."


# --- LLM Call ---
def llm_call(dialog: list, structure=None, model: str = DEFAULT_MODEL,
             temperature: float = 0.0, max_tokens: int = 4096,
             timeout: float = 120.0, api_key: Optional[str] = None) -> dict | str:
    """
    Универсальная функция вызова LLM.

    Args:
        dialog: Список сообщений для диалога.
        structure: Pydantic-модель для структурного вывода.
        model: Название модели.
        temperature: Температура генерации.
        max_tokens: Максимальное количество токенов (по умолчанию 4096).
        timeout: Таймаут ожидания ответа в секундах (по умолчанию 120).
        api_key: Ключ GigaChat (если не передан — используется глобальный).

    Returns:
        Структурированные данные (dict) или текстовый ответ (str).

    Raises:
        httpx.TimeoutException: Если модель не ответила за timeout секунд.
    """
    local_client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=timeout) if api_key else (client if client else OpenAI(api_key=GIGACHAT_KEY, base_url=BASE_URL, timeout=timeout))

    request_params = {
        "model": model,
        "messages": dialog,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if structure is not None:
        response = local_client.chat.completions.parse(
            model=model, messages=dialog, temperature=temperature,
            max_tokens=max_tokens, response_format=structure,
        )
        return response.choices[0].message.parsed.model_dump()
    else:
        response = local_client.chat.completions.create(**request_params)
        return response.choices[0].message.content


# --- Node functions ---
def classify_intent(state: TravelState) -> dict:
    """
    Классифицирует намерение пользователя: plan / refine / recall / unknown.
    Парсит даты из запроса регулярными выражениями.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с intent, user_query, messages и распарсенными датами.
    """
    from pydantic import BaseModel
    import re
    class IntentFormat(BaseModel):
        intent: Literal["plan", "refine", "recall", "unknown"]
        missing_fields: List[str] = []
    user_query = state.get("user_query", "")
    user_query_lower = user_query.lower()
    messages = state.get("messages", [])
    api_key = state.get("api_key")
    
    # Парсим даты из запроса
    parsed_dates = None
    date_patterns = [
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})",  # 01.04.2027
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{2})",   # 01.04.27
        r"(\d{4}-\d{2}-\d{2})",                # 2027-04-01
    ]
    for pattern in date_patterns:
        match = re.search(pattern, user_query)
        if match:
            parsed_dates = match.group(1)
            break
    
    # Парсим количество дней
    parsed_days = None
    days_match = re.search(r"по\s+(\d{1,2}[./-]\d{1,2}[./-]\d{4})", user_query)
    if days_match:
        end_date = days_match.group(1)
        if parsed_dates:
            parsed_dates = f"{parsed_dates},{end_date}"
    
    # Общие вопросы — определяем без LLM
    general_keywords = ["что ты умеешь", "что ты можешь", "помоги", "как работает", "что это", "help"]
    if any(kw in user_query_lower for kw in general_keywords):
        return {"intent": "unknown", "user_query": user_query,
                "agent_response": "Я — Travel Planner, агент для планирования путешествий. Могу помочь:\n\n"
                                   "• Подобрать направление по вашим предпочтениям (погода, бюджет, сезон)\n"
                                   "• Проверить погоду в выбранном месте\n"
                                   "• Рассчитать стоимость поездки (перелёт, отель, питание)\n"
                                   "• Предложить варианты жилья и транспорта\n"
                                   "• Составить детальный план поездки по дням\n\n"
                                   "Примеры запросов:\n"
                                   "• «Хочу на горнолыжный курорт в апреле»\n"
                                   "• «Пляжный отдых в феврале, бюджет 70 000 рублей»\n"
                                   "• «Куда поехать в марте бюджетно?»\n"
                                   "• «Составь план поездки в Иорданию на 5 дней»"}
    
    # Формируем полный диалог с историей
    dialog = [
        {"role": "system", "content": """Ты — Travel Planner. Определи намерение:
plan = хочет спланировать поездку (есть запрос про путешествие, курорт, отель, даты, бюджет и т.д.)
refine = хочет уточнить или изменить существующий план (есть слова: измени, другие даты, перенеси, хочу в апреле, хочу в мае)
recall = хочет восстановить сохранённый план
unknown = не относится к путешествиям

Примеры:
- "Хочу на горнолыжный курорт" → plan
- "Куда поехать в феврале?" → plan
- "Измени даты на март" → refine
- "Хочу в апреле" → refine
- "Покажи мой план" → recall
- "Что ты умеешь?" → unknown"""},
    ]
    
    # Добавляем историю чата
    if messages:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
                dialog.append(msg)
    
    # Добавляем текущий запрос
    dialog.append({"role": "user", "content": user_query})
    
    result = llm_call(dialog, structure=IntentFormat, max_tokens=256, api_key=api_key)
    
    # Формируем ответ
    response = {
        "intent": result["intent"],
        "user_query": user_query,
        "messages": [{"role": "assistant", "content": f"Намерение: {result['intent']}"}],
    }
    
    # Если есть распарсенные даты — добавляем в state
    if parsed_dates:
        response["dates"] = parsed_dates
    
    return response


def select_destination(state: TravelState) -> dict:
    """
    Подбирает направление для поездки. LLM самостоятельно определяет тип отдыха
    и рекомендует конкретный курорт/город на основе запроса, бюджета и дат.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с destination и messages.
    """
    from pydantic import BaseModel
    class DestinationFormat(BaseModel):
        destination: str
        reason: str
    user_query = state.get("user_query", "")
    budget = state.get("budget")
    dates = state.get("dates")
    api_key = state.get("api_key")
    
    dialog = [
        {"role": "system", "content": (
            "Ты — эксперт по путешествиям. Подбери конкретное направление (город/курорт/страну) "
            "на основе запроса пользователя. Учитывай сезон, бюджет и активность. "
            "Ответь строго в формате JSON: {destination: 'название', reason: 'краткая причина до 1 предложения'}."
        )},
        {"role": "user", "content": user_query +
            (f"\nБюджет: {budget} руб." if budget else "") +
            (f"\nДаты: {dates}" if dates else "")},
    ]
    result = llm_call(dialog, structure=DestinationFormat, max_tokens=512, api_key=api_key)
    return {"destination": result["destination"],
            "messages": [{"role": "assistant", "content": f"Рекомендую: {result['destination']}"}]}


def ask_missing_info(state: TravelState) -> dict:
    """
    Определяет недостающие параметры и формирует уточняющий вопрос.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с agent_response, messages и asked_missing_info.
    """
    from pydantic import BaseModel
    class MissingInfoFormat(BaseModel):
        need_clarification: bool
        missing_fields: List[str]
        query_to_user: str
    user_query = state.get("user_query", "")
    api_key = state.get("api_key")
    
    # Проверяем какие поля уже есть
    missing = []
    if not state.get("destination"):
        missing.append("направление")
    if not state.get("dates"):
        missing.append("даты")
    if not state.get("budget"):
        missing.append("бюджет")
    if not state.get("comfort_level"):
        missing.append("комфорт")
    
    dialog = [
        {"role": "system", "content": f"Travel Planner. Задай 1 вопрос. Нужно узнать: {', '.join(missing)}."},
        {"role": "user", "content": user_query},
    ]
    result = llm_call(dialog, structure=MissingInfoFormat, max_tokens=512, api_key=api_key)
    return {
        "agent_response": result["query_to_user"],
        "messages": [{"role": "assistant", "content": result["query_to_user"]}],
        "asked_missing_info": True,
    }


def load_state(state: TravelState) -> dict:
    """
    Восстанавливает состояние из JSON-файла по session_id.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с agent_response и messages.
    """
    session_id = state.get("session_id", "")
    states_dir = Path("states")
    state_file = states_dir / f"{state.get('user_id', 'default')}_{session_id}.json"
    if state_file.exists():
        with open(state_file) as f:
            saved = json.load(f)
        response = f"Данные загружены! Направление: {saved.get('destination', 'не указано')}"
    else:
        response = f"Состояние '{session_id}' не найдено."
    return {"agent_response": response,
            "messages": [{"role": "assistant", "content": response}]}


def _parse_days(dates_str: str) -> int:
    """
    Определяет количество дней из строки дат.

    Args:
        dates_str: Строка в формате 'YYYY-MM-DD,YYYY-MM-DD'.

    Returns:
        Количество дней между датами.
    """
    if not dates_str or "," not in dates_str:
        return 7
    try:
        parts = dates_str.split(",")
        start = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
        end = datetime.strptime(parts[1].strip(), "%Y-%m-%d")
        delta = (end - start).days
        return max(delta, 1)
    except (ValueError, IndexError):
        return 7


def check_weather_budget(state: TravelState) -> dict:
    """
    Проверяет погоду и стоимость поездки, сравнивает с бюджетом.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с weather, costs, weather_ok, budget_ok, comfort_level.
    """
    destination = state.get("destination", "")
    dates = state.get("dates", "")
    budget = state.get("budget")
    comfort = state.get("comfort_level", "standard")
    
    # Определяем реальное количество дней
    days = _parse_days(dates)
    
    weather_result = get_weather_forecast(destination, dates)
    costs_result = get_travel_costs(destination, days, comfort)
    weather_ok = "ошибка" not in weather_result.lower()
    
    # Проверяем бюджет: извлекаем итоговую сумму из costs_result
    budget_ok = True
    if budget is not None:
        import re
        cost_match = re.search(r"итого ~\$(\d+)", costs_result)
        if cost_match:
            total_usd = int(cost_match.group(1))
            # Получаем реальный курс через API
            rate_info = get_exchange_rates("USD", "RUB")
            try:
                rate = float(rate_info.split("=")[1].split()[0])
            except (IndexError, ValueError):
                rate = 95  # fallback
            rub_budget = total_usd * rate
            if rub_budget > budget:
                budget_ok = False
    
    return {"weather": weather_result, "costs": costs_result,
            "weather_ok": weather_ok, "budget_ok": budget_ok,
            "comfort_level": comfort,
            "messages": [{"role": "assistant", "content": f"Погода OK={weather_ok}, бюджет OK={budget_ok}"}]}


def suggest_alternatives(state: TravelState) -> dict:
    """
    Предлагает альтернативные направления при плохой погоде или превышении бюджета.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с alternatives и messages.
    """
    destination = state.get("destination", "")
    user_query = state.get("user_query", "")
    api_key = state.get("api_key")
    dialog = [
        {"role": "system", "content": """Ты — Travel Planner. Предложи 2-3 альтернативные направления для поездки, которые лучше соответствуют запросу пользователя.
Укажи: название, почему подходит, примерная стоимость, лучший сезон. Кратко."""},
        {"role": "user", "content": f"Выбрано: {destination}. Запрос: {user_query}"},
    ]
    result = llm_call(dialog, model=DEFAULT_MODEL, max_tokens=4096, api_key=api_key)
    return {"alternatives": result,
            "messages": [{"role": "assistant", "content": result}]}


def build_itinerary(state: TravelState) -> dict:
    """
    Составляет детальный план поездки с использованием reasoning-модели.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с itinerary и messages.
    """
    destination = state.get("destination", "")
    weather = state.get("weather", "")
    costs = state.get("costs", "")
    budget = state.get("budget")
    dates = state.get("dates", "")
    comfort = state.get("comfort_level", "standard")
    user_query = state.get("user_query", "")
    api_key = state.get("api_key")
    dialog = [
        {"role": "system", "content": """Ты — профессиональный туристический планировщик.
Составь детальный план поездки на основе ВСЕХ данных. Используй точные даты из запроса.
Включи: транспорт, проживание, достопримечательности, питание, бюджет по дням.
Будь конкретным и практичным."""},
        {"role": "user", "content": f"Направление: {destination}\nДаты: {dates}\nБюджет: {budget} руб.\nУровень комфорта: {comfort}\nПогода: {weather}\nСтоимость: {costs}\nЗапрос: {user_query}"},
    ]
    result = llm_call(dialog, model=REASONING_MODEL, max_tokens=4096, api_key=api_key)
    return {"itinerary": result,
            "messages": [{"role": "assistant", "content": result}]}


def save_state(state: TravelState) -> dict:
    """
    Сериализует состояние в JSON-файл для долгосрочного хранения.

    Args:
        state: Текущее состояние графа.

    Returns:
        Словарь с agent_response и messages.
    """
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "session_1")
    states_dir = Path("states")
    states_dir.mkdir(exist_ok=True)
    state_file = states_dir / f"{user_id}_{session_id}.json"
    data_to_save = {
        "user_id": user_id, "session_id": session_id,
        "destination": state.get("destination"),
        "budget": state.get("budget"),
        "itinerary": state.get("itinerary"),
        "saved_at": datetime.now().isoformat(),
    }
    with open(state_file, "w") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    return {"agent_response": "План сохранён.",
            "messages": [{"role": "assistant", "content": "План сохранён."}]}


# --- Routing ---
def route_intent(state: TravelState) -> str:
    """
    Определяет следующий узел после classify_intent.

    Args:
        state: Текущее состояние графа.

    Returns:
        Название следующего узла.
    """
    intent = state.get("intent", "unknown")
    if intent == "recall":
        return "load_state"
    elif intent == "refine":
        # При уточнении — если есть направление, идём пересчитывать
        if state.get("destination"):
            return "check_weather_budget"
        # Иначе уточняем
        return "ask_missing_info"
    elif intent == "plan":
        # Если уже задавали уточняющие — идём подбирать направление
        if state.get("asked_missing_info"):
            return "select_destination"
        # Если все поля уже есть — проверяем, конкретное ли направление
        if (state.get("destination") and 
            state.get("dates") and 
            state.get("budget") and 
            state.get("comfort_level")):
            return "check_weather_budget"
        # Иначе сначала уточняем
        return "ask_missing_info"
    elif intent == "unknown":
        # Если есть готовый ответ (общий вопрос) — сразу END
        if state.get("agent_response"):
            return "__end__"
        return "ask_missing_info"
    else:
        return "ask_missing_info"


def route_check(state: TravelState) -> str:
    """
    Определяет следующий узел после check_weather_budget.

    Args:
        state: Текущее состояние графа.

    Returns:
        'build_itinerary' если всё OK, 'suggest_alternatives' иначе.
    """
    weather_ok = state.get("weather_ok", True)
    budget_ok = state.get("budget_ok", True)
    if not weather_ok or not budget_ok:
        return "suggest_alternatives"
    return "build_itinerary"


# --- Graph ---
graph_builder = StateGraph(TravelState)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("select_destination", select_destination)
graph_builder.add_node("ask_missing_info", ask_missing_info)
graph_builder.add_node("load_state", load_state)
graph_builder.add_node("check_weather_budget", check_weather_budget)
graph_builder.add_node("suggest_alternatives", suggest_alternatives)
graph_builder.add_node("build_itinerary", build_itinerary)
graph_builder.add_node("save_state", save_state)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent, {
    "select_destination": "select_destination",
    "ask_missing_info": "ask_missing_info",
    "load_state": "load_state",
    "__end__": END,
})
graph_builder.add_edge("select_destination", "check_weather_budget")
graph_builder.add_edge("ask_missing_info", "check_weather_budget")
graph_builder.add_edge("load_state", END)
graph_builder.add_conditional_edges("check_weather_budget", route_check, {
    "build_itinerary": "build_itinerary",
    "suggest_alternatives": "suggest_alternatives",
})
graph_builder.add_edge("suggest_alternatives", "ask_missing_info")
graph_builder.add_edge("build_itinerary", "save_state")
graph_builder.add_edge("save_state", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# --- Pydantic models ---
class ChatRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str = Field(default="session_1")
    message: str
    context: str = ""  # история чата
    budget: Optional[float] = None  # без дефолта!
    dates: Optional[str] = None  # без дефолта!
    comfort_level: Optional[str] = None  # без дефолта!
    api_key: Optional[str] = None  # ключ GigaChat из UI


class ChatResponse(BaseModel):
    response: str
    destination: Optional[str] = None
    itinerary: Optional[str] = None
    session_id: str


# --- FastAPI app ---
app = FastAPI(title="Travel Planner Agent", version="1.0.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    """
    Основной эндпоинт для синхронного вызова агента.

    Args:
        payload: Запрос с message, session_id, api_key и др.

    Returns:
        ChatResponse с response, destination, itinerary.

    Raises:
        HTTPException: При ошибке обработки запроса.
    """
    try:
        initial_state: TravelState = {
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "user_query": payload.message,
            "budget": payload.budget,
            "dates": payload.dates,
            "comfort_level": payload.comfort_level,
            "messages": [{"role": "user", "content": payload.context}],
            "api_key": payload.api_key,
        }
        config = {"configurable": {"thread_id": f"{payload.user_id}_{payload.session_id}"}}
        result = graph.invoke(initial_state, config)
        return ChatResponse(
            response=result.get("agent_response", ""),
            destination=result.get("destination"),
            itinerary=result.get("itinerary"),
            session_id=payload.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка агента: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    """
    Эндпоинт для стримингового вызова агента через SSE.

    Args:
        payload: Запрос с message, session_id, api_key и др.

    Returns:
        StreamingResponse с токенами в формате SSE.

    Raises:
        HTTPException: При ошибке обработки запроса.
    """
    try:
        initial_state: TravelState = {
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "user_query": payload.message,
            "budget": payload.budget,
            "dates": payload.dates,
            "comfort_level": payload.comfort_level,
            "messages": [{"role": "user", "content": payload.context}],
            "api_key": payload.api_key,
        }
        config = {"configurable": {"thread_id": f"{payload.user_id}_{payload.session_id}"}}

        async def event_generator():
            async for event in graph.astream_events(initial_state, config, version="v2"):
                event_type = event.get("event", "")
                if event_type == "on_chain_end":
                    data = event.get("data", {})
                    yield f"data: {json.dumps({'type': 'end', 'output': data}, ensure_ascii=False)}\n\n"
                elif event_type == "on_chat_model_stream":
                    content = event.get("data", {}).get("chunk", {}).get("content", "")
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка агента: {str(e)}")


@app.get("/state/{session_id}")
async def get_state(session_id: str, user_id: str = "default"):
    """
    Получение сохранённого состояния сессии из JSON-файла.

    Args:
        session_id: ID сессии.
        user_id: ID пользователя.

    Returns:
        Словарь со статусом и данными.
    """
    try:
        states_dir = Path("states")
        state_file = states_dir / f"{user_id}_{session_id}.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            return {"status": "found", "data": data}
        else:
            return {"status": "not_found", "message": f"Сессия {session_id} не найдена"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/reset/{session_id}")
async def reset_state(session_id: str, user_id: str = "default"):
    """
    Очистка сохранённого состояния сессии.

    Args:
        session_id: ID сессии.
        user_id: ID пользователя.

    Returns:
        Словарь со статусом операции.
    """
    try:
        states_dir = Path("states")
        state_file = states_dir / f"{user_id}_{session_id}.json"
        if state_file.exists():
            state_file.unlink()
            return {"status": "reset", "message": f"Сессия {session_id} очищена"}
        else:
            return {"status": "not_found", "message": f"Сессия {session_id} не найдена"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health():
    """Health-check эндпоинт для проверки работоспособности сервиса."""
    return {"status": "ok", "model": DEFAULT_MODEL}
