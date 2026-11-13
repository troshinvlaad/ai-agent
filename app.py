"""
FastAPI-сервис для Travel Planner агента на базе GigaChat + LangGraph.

Агент использует LLM GigaChat для принятия решений, вызывает внешние API
через инструменты (tools), сохраняет состояние диалога и маршрутизирует
запросы через граф LangGraph с 4 путями выполнения.

Запуск: uvicorn app:app --reload --port 8000
API-ключ: GIGACHAT_KEY (переменная окружения или .env файл)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated, Literal
from typing_extensions import TypedDict
import json
import os
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Конфигурация ---
BASE_URL = "https://foundation-models.api.cloud.ru/v1"
DEFAULT_MODEL = "GigaChat/GigaChat-3-Pro"
GIGACHAT_KEY = os.environ.get("GIGACHAT_KEY", "")

# --- Импорт LangGraph ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI, pydantic_function_tool


# --- State ---
class TravelState(TypedDict, total=False):
    """Состояние Travel Planner агента. Хранит все данные сессии: параметры поездки,
    историю сообщений, флаги статуса внешних API и счётчики уточнений."""
    user_id: str
    session_id: str
    messages: Annotated[List[dict], add_messages]
    intent: Literal["plan", "refine", "recall", "unknown"]
    destination: Optional[str]
    dates: Optional[str]
    budget: Optional[float]
    comfort_level: Optional[str]
    passengers: Optional[int]
    departure_city: Optional[str]
    interests: Optional[str]
    special_requests: Optional[str]
    transport: Optional[str]
    weather: Optional[str]
    costs: Optional[str]
    exchange_rate: Optional[str]
    alternatives: Optional[str]
    itinerary: Optional[str]
    user_query: str
    agent_response: str
    asked_missing_info: bool
    clarification_count: int
    api_key: Optional[str]


# --- Tools ---
def get_weather_forecast(lat: float, lon: float, dates: str = "", location: str = "локация") -> str:
    """Получает прогноз погоды для координат через Open-Meteo API.

    Args:
        lat: Широта (из Nominatim).
        lon: Долгота (из Nominatim).
        dates: Даты поездки в формате YYYY-MM-DD,YYYY-MM-DD.
        location: Название места для отображения в ответе.

    Returns:
        Строка с прогнозом погоды или сообщение о недоступности.
    """
    try:
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
            f"&timezone=auto"
        )
        weather_resp = requests.get(weather_url, timeout=10.0)
        weather_data = weather_resp.json()
        daily = weather_data.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes = daily.get("weathercode", [])
        if not temps_max:
            return "Сервис погоды недоступен."
        avg_temp = sum(temps_max) / len(temps_max)
        total_precip = sum(precip) if precip else 0
        def weather_desc(code):
            return {0: "Ясно", 1: "Преимущественно ясно", 2: "Переменная облачность",
                    3: "Пасмурно", 61: "Дождь", 71: "Снег", 95: "Гроза"}.get(code, "Неизвестно")
        return (f"**Погода в {location}**: Средняя макс. {avg_temp:.1f}°C, "
                f"мин. {min(temps_min):.1f}°C, осадки {total_precip:.1f} мм, "
                f"погода: {weather_desc(codes[0] if codes else 0)}")
    except (requests.Timeout, requests.ConnectionError, requests.exceptions.RequestException):
        return "Сервис погоды временно недоступен (таймаут/ошибка соединения)."
    except Exception as e:
        return f"Сервис погоды недоступен: {e}."


def get_travel_costs(location: str, days: int, comfort_level: str) -> str:
    return f"Стоимость в {location} ({days} дн., {comfort_level}) будет рассчитана LLM."


def get_exchange_rates(from_currency="USD", to_currency="RUB") -> str:
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        resp = requests.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        rate = data["rates"].get(to_currency)
        if rate is None:
            return f"Валюта {to_currency} не найдена."
        return f"1 {from_currency} = {rate} {to_currency}"
    except (requests.Timeout, requests.ConnectionError, requests.exceptions.RequestException):
        return f"Сервис курсов валют временно недоступен."
    except Exception as e:
        return f"Сервис курсов валют недоступен: {e}."


def get_geolocation(location: str) -> str:
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1&addressdetails=1"
        headers = {"User-Agent": "TravelPlanner/1.0"}
        resp = requests.get(url, headers=headers, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return f"Место '{location}' не найдено."
        item = data[0]
        addr = item.get("address", {})
        country = addr.get("country", "неизвестно")
        city = addr.get("city") or addr.get("town") or location
        return f"**{city}, {country}**: {item['lat']}, {item['lon']}"
    except (requests.Timeout, requests.ConnectionError, requests.exceptions.RequestException):
        return f"Сервис геокодинга временно недоступен."
    except Exception as e:
        return f"Сервис геокодинга недоступен: {e}."


# --- LLM Call через OpenAI SDK (паттерн из 6-го занятия) ---
def llm_call(dialog, structure=None, model: str = DEFAULT_MODEL,
             temperature: float = 0.0, max_tokens: int = 4096,
             timeout: float = 60.0, api_key: Optional[str] = None) -> dict | str:
    """Вызов GigaChat через OpenAI SDK. LLM всегда доступна."""
    key = api_key or GIGACHAT_KEY
    client = OpenAI(
        api_key=key,
        base_url="https://foundation-models.api.cloud.ru/v1",
    )

    if structure:
        response = client.chat.completions.parse(
            model=model,
            messages=dialog,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=dialog,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# --- Node functions ---
def show_template(state: TravelState) -> dict:
    """Показать шаблон с командами для заполнения."""
    template = (
        "🌍 **Travel Planner — заполните параметры поездки:**\n\n"
        "Используйте команды с JSON:\n\n"
        "**/plan_create** — создать новый план:\n"
        "```json\n"
        '{"destination": "Шерегеш", "departure_city": "Нижний Новгород", "dates": "2027-04-01,2027-04-06", "budget": 150000, "passengers": 2, "comfort_level": "стандарт", "transport": "авиа", "interests": "сноубординг", "special_requests": "нужна аренда сноубордов и ботинок, остальное снаряжение есть"}\n'
        "```\n\n"
        "**/plan_update** — обновить конкретные параметры:\n"
        "```json\n"
        '{"budget": 150000, "passengers": 3}\n'
        "```\n\n"
        "Доступные поля:\n"
        "- destination: куда ехать (Сочи, Шерегеш, Турция)\n"
        "- departure_city: откуда (Москва, СПб)\n"
        "- dates: даты (2026-04-15,2026-04-22)\n"
        "- budget: общий бюджет на всех (число)\n"
        "- passengers: количество пассажиров (число)\n"
        "- comfort_level: бюджет/стандарт/премиум\n"
        "- transport: авиа/жд/авто\n"
        "- interests: интересы (пляж, экскурсии)\n"
        "- special_requests: пожелания"
    )
    return {
        "agent_response": template,
        "messages": [{"role": "assistant", "content": template}],
    }


def classify_intent(state: TravelState) -> dict:
    """Классифицирует намерение и извлекает параметры.
    
    Поддерживает только команды:
    1. /plan_create — создать новый план
    2. /plan_update — обновить существующий план
    """
    user_query = state.get("user_query", "")

    # === КОМАНДЫ: /plan_create и /plan_update ===
    # Парсим JSON напрямую, без LLM — ПЕРВЕЕ всего
    if user_query.startswith("/"):
        parts = user_query.split(None, 1)  # split by whitespace
        command = parts[0].lower()
        json_str = parts[1] if len(parts) > 1 else ""

        # Пытаемся распарсить JSON напрямую
        try:
            # LLM может вернуть кириллические кавычки — чистим ТОЛЬКО кавычки, НЕ кириллицу
            cleaned = json_str.replace("\u201c", '"').replace("\u201d", '"')
            cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")
            params = json.loads(cleaned)
            
            if not isinstance(params, dict):
                params = {}
        except (json.JSONDecodeError, ValueError):
            params = {}

        # Нормализуем ключи (разные варианты написания)
        param_map = {
            "destination": ["destination", "направление", "куда", "курорт", "country", "city"],
            "departure_city": ["departure_city", "откуда", "from", "departure", "city_from"],
            "dates": ["dates", "даты", "date", "start_date", "end_date", "period"],
            "budget": ["budget", "бюджет", "cost", "amount", "price"],
            "comfort_level": ["comfort_level", "комфорт", "comfort", "class", "standard"],
            "passengers": ["passengers", "пассажиры", "passenger_count", "people", "adults", "count"],
            "interests": ["interests", "интересы", "interest", "activities", "hobbies"],
            "special_requests": ["special_requests", "пожелания", "requests", "preferences", "wishes"],
            "transport": ["transport", "транспорт", "transport_type", "travel_by"],
        }

        normalized = {}
        for std_key, variants in param_map.items():
            for variant in variants:
                if variant in params:
                    normalized[std_key] = params[variant]
                    break
        
        # Преобразуем budget в float
        if "budget" in normalized:
            try:
                normalized["budget"] = float(str(normalized["budget"]).replace(" ", "").replace(",", ""))
            except (ValueError, TypeError):
                normalized.pop("budget", None)
        
        # Преобразуем passengers в int
        if "passengers" in normalized:
            try:
                normalized["passengers"] = int(normalized["passengers"])
            except (ValueError, TypeError):
                normalized.pop("passengers", None)

        if command == "/plan_create":
            # Создаём новый план — игнорируем старое состояние
            response = {
                "intent": "plan",
                "user_query": user_query,
                "messages": [],
                "clarification_count": 0,
            }
            # Очищаем все параметры
            for key in ("destination", "departure_city", "dates", "budget",
                        "comfort_level", "passengers", "interests", "special_requests",
                        "transport", "weather", "costs", "exchange_rate", "itinerary"):
                response[key] = None
            for key, value in normalized.items():
                if value is not None:
                    response[key] = value
            return response

        elif command == "/plan_update":
            # Обновляем существующий план — сохраняем старое состояние
            response = {
                "intent": "refine",
                "user_query": user_query,
                "messages": state.get("messages", []),
            }
            # Копируем существующие параметры
            for key in ("destination", "departure_city", "dates", "budget", 
                        "comfort_level", "passengers", "interests", "special_requests", "transport"):
                val = state.get(key)
                if val is not None:
                    response[key] = val
            # Переопределяем новыми значениями
            for key, value in normalized.items():
                if value is not None:
                    response[key] = value

            # Если refine, но destination не задан — показываем шаблон
            if not response.get("destination"):
                return show_template(state)

            return response

        # Неизвестная команда — показываем шаблоны
        return show_template(state)

    # Быстрый ответ на общие вопросы — показываем команды
    if any(kw in user_query.lower() for kw in ["что ты умеешь", "что ты можешь", "помоги", "как работает"]):
        return show_template(state)

    # === ФИЛЬТР: только вопросы про путешествия ===
    travel_keywords = [
        "отдых", "поездк", "море", "горы", "курорт", "путешеств", "тур",
        "отель", "билет", "авиа", "поезд", "виза", "бюджет", "маршрут",
        "проживан", "трансфер", "экскурси", "пляж", "сноуборд", "лыж",
        "зимн", "летн", "отпуск", "каникул", "направлен", "стран",
        "город", "загород", "база", "хостел", "ресторан", "кухн",
        "погода", "сезон", "климат", "активн", "развлечен", "достопримечательност",
        "авиабилет", "ж/д", "железнодорожн", "автобус", "машина", "авто",
        "снять", "аренд", "жильё", "жилье", "номер", "завтрак", "питание",
        "сочи", "шерегеш", "антальи", "стамбул", "москв", "питер", "петербург",
        "кемеров", "новосибирск", "краснодар", "крым", "дагестан", "азов",
        "белгород", "нижний", "казань", "самара", "уфа", "екатеринбург",
        "россия", "россии", "российск", "турция", "турецк", "египет",
        "таиланд", "тайланд", "оаэ", "оаэ", "дубай", "дубаи", "абудаби",
        "грузия", "тбилиси", "батуми", "армения", "армянск", "азербайджан",
        "беларусь", "казахстан", "узбекистан", "кипр", "китай", "япония",
        "италия", "франция", "испания", "германия", "франция", "греция",
    ]
    kw_lower = user_query.lower()
    kw_count = sum(1 for kw in travel_keywords if kw in kw_lower)

    if kw_count < 1:
        # Не похоже на вопрос про путешествия — показываем команды
        return show_template(state)

    # Сообщение про путешествия — передаём в LLM для ответа
    travel_dialog = [
        {"role": "system", "content": (
            "Ты — Travel Planner. Отвечай ТОЛЬКО на вопросы, связанные с путешествиями, "
            "отдыхом, туризмом, курортами, визами, транспортом, проживанием, бюджетом поездки.\n\n"
            "Если вопрос НЕ про путешествия — вежливо откажи и предложи использовать команды:\n"
            "/plan_create — создать новый план\n"
            "/plan_update — обновить существующий план\n\n"
            "ВАЖНО: Не задавай вопросов и не проси дополнительную информацию.\n"
            "Отвечай кратко и по делу на русском языке.\n"
            "Не используй фразы типа 'Хотите добавить детали?', 'Могу предложить...' и т.д."
        )},
        {"role": "user", "content": user_query},
    ]
    api_key = state.get("api_key")
    try:
        answer = llm_call(travel_dialog, structure=None, max_tokens=1024, api_key=api_key)
        return {
            "agent_response": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "user_query": user_query,
        }
    except Exception:
        return show_template(state)





def load_state(state: TravelState) -> dict:
    session_id = state.get("session_id", "")
    states_dir = Path("states")
    state_file = states_dir / f"{state.get('user_id', 'default')}_{session_id}.json"
    if state_file.exists():
        with open(state_file) as f:
            saved = json.load(f)
        # Восстанавливаем все параметры
        response_parts = [f"Данные загружены!"]
        if saved.get("destination"):
            response_parts.append(f"Направление: {saved['destination']}")
        if saved.get("departure_city"):
            response_parts.append(f"Откуда: {saved['departure_city']}")
        if saved.get("dates"):
            response_parts.append(f"Даты: {saved['dates']}")
        if saved.get("budget"):
            response_parts.append(f"Бюджет: {saved['budget']} руб.")
        if saved.get("comfort_level"):
            response_parts.append(f"Комфорт: {saved['comfort_level']}")
        if saved.get("passengers"):
            response_parts.append(f"Пассажиров: {saved['passengers']}")
        if saved.get("itinerary"):
            response_parts.append(f"План: {saved['itinerary'][:200]}...")
        response = "\n".join(response_parts)
        # Восстанавливаем параметры в состоянии
        restore_fields = ["destination", "departure_city", "dates", "budget",
                          "comfort_level", "passengers", "interests", "special_requests",
                          "itinerary", "weather", "costs", "exchange_rate"]
        restore = {}
        for field in restore_fields:
            if field in saved:
                restore[field] = saved[field]
        restore["agent_response"] = response
        restore["messages"] = [{"role": "assistant", "content": response}]
        return restore
    else:
        response = f"Состояние '{session_id}' не найдено."
        return {"agent_response": response, "messages": [{"role": "assistant", "content": response}]}


def _parse_days(dates_str: str) -> int:
    if not dates_str or "," not in dates_str:
        return 7
    try:
        parts = dates_str.split(",")
        start = datetime.strptime(parts[0].strip(), "%Y-%m-%d")
        end = datetime.strptime(parts[1].strip(), "%Y-%m-%d")
        return max((end - start).days, 1)
    except (ValueError, IndexError):
        return 7


# --- Описание инструментов через pydantic_function_tool (паттерн из 2-го и 6-го занятий) ---
# LLM получает ТОЛЬКО описания инструментов, реализация вызывается Python-кодом

class GetWeatherTool(BaseModel):
    """Получить прогноз погоды по координатам на даты поездки."""
    lat: float = Field(description="Широта (из геокодинга Nominatim)")
    lon: float = Field(description="Долгота (из геокодинга Nominatim)")
    dates: str = Field(default="", description="Даты поездки в формате YYYY-MM-DD,YYYY-MM-DD")
    location: str = Field(default="локация", description="Название места для отображения в ответе")

class GetCostsTool(BaseModel):
    """Оценить стоимость поездки с учётом комфортного проживания."""
    location: str = Field(description="Город или курорт")
    days: int = Field(description="Количество дней поездки")
    comfort_level: str = Field(description="Уровень комфорта: бюджет, стандарт, премиум")

class GetExchangeRateTool(BaseModel):
    """Получить курс валют (USD/EUR к RUB)."""
    from_currency: str = Field(default="USD", description="Исходная валюта (например: USD, EUR)")
    to_currency: str = Field(default="RUB", description="Целевая валюта (например: RUB)")

class GetGeolocationTool(BaseModel):
    """Получить координаты и страну по названию города."""
    location: str = Field(description="Город или место для геокодирования")


# Создаём описания инструментов для LLM через pydantic_function_tool
_get_weather_desc = pydantic_function_tool(GetWeatherTool, name="get_weather")
_get_costs_desc = pydantic_function_tool(GetCostsTool, name="get_costs")
_get_exchange_rate_desc = pydantic_function_tool(GetExchangeRateTool, name="get_exchange_rate")
_get_geolocation_desc = pydantic_function_tool(GetGeolocationTool, name="get_geolocation")


def check_weather_budget(state: TravelState) -> dict:
    """Агентный цикл: LLM сама решает, какие инструменты вызвать.

    Паттерн из 6-го занятия (строки 1018-1064):
    - LLM получает описания инструментов через pydantic_function_tool
    - LLM вызывает инструменты через tool_calls
    - Python исполняет вызовы и возвращает результат с ролью "tool"
    - Цикл продолжается пока LLM не вернёт финальный ответ

    Если внешний сервис недоступен — LLM видит ошибку в результате
    и решает, как действовать дальше (другой инструмент или альтернатива).
    """
    destination = state.get("destination", "")
    dates = state.get("dates", "")
    budget = state.get("budget")
    comfort = state.get("comfort_level", "стандарт")
    passengers = state.get("passengers", 1)
    departure_city = state.get("departure_city", "Москва")
    api_key = state.get("api_key")
    days = _parse_days(dates) if dates else 7

    # Описание инструментов (только текстовая часть — LLM решает какие вызвать)
    tools = [
        _get_weather_desc,
        _get_geolocation_desc,
        _get_costs_desc,
        _get_exchange_rate_desc,
    ]

    # Маппинг названий инструментов на Python-функции
    TOOL_MAP = {
        "get_weather": get_weather_forecast,
        "get_geolocation": get_geolocation,
        "get_costs": get_travel_costs,
        "get_exchange_rate": get_exchange_rates,
    }

    # Системный промпт — даём LLM контекст поездки
    agent_system = (
        "Ты — Travel Planner. Тебе нужно собрать данные для составления плана поездки.\n\n"
        "Доступные инструменты:\n"
        "- get_geolocation: координаты города и страна (нужен location)\n"
        "- get_weather: прогноз погоды по координатам (нужны lat, lon из geolocation)\n"
        "- get_costs: оценка стоимости поездки (нужны location, days, comfort_level)\n"
        "- get_exchange_rate: курс валют (нужны from_currency, to_currency)\n\n"
        "Порядок вызовов:\n"
        "1. Сначала вызови get_geolocation для получения координат.\n"
        "2. Затем вызови get_weather с полученными lat и lon.\n"
        "3. Затем вызови get_costs для оценки бюджета.\n"
        "4. Вызови get_exchange_rate для курса USD→RUB.\n\n"
        "Информация о поездке:\n"
        f"- Направление: {destination}\n"
        f"- Даты: {dates if dates else 'не указаны'}\n"
        f"- Бюджет: {budget if budget else 'не указан'} руб.\n"
        f"- Пассажиров: {passengers}\n"
        f"- Откуда: {departure_city if departure_city else 'не указано'}\n"
        f"- Комфорт: {comfort}\n\n"
        "Правила:\n"
        "1. Сначала вызови get_geolocation для получения координат.\n"
        "2. Затем вызови get_weather с полученными lat и lon.\n"
        "3. Затем вызови get_costs для оценки бюджета.\n"
        "4. Вызови get_exchange_rate для курса USD→RUB.\n"
        "5. После всех вызовов верни финальный ответ с итогами каждого инструмента.\n"
        "6. Если сервис недоступен — сообщи об этом и предложи продолжить без него."
    )

    # Agent loop — точный паттерн из 6-го занятия (строки 1032-1064)
    messages = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": f"Собери данные для поездки в {destination}."}
    ]

    # Собираем результаты инструментов
    weather_result = ""
    geolocation_result = ""
    costs_result = ""
    exchange_result = ""

    max_iterations = 10
    for _ in range(max_iterations):
        key = api_key or GIGACHAT_KEY
        client = OpenAI(
            api_key=key,
            base_url="https://foundation-models.api.cloud.ru/v1",
        )
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=tools,
            temperature=0.0,
            max_tokens=1024
        )
        message = resp.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                fn = TOOL_MAP.get(name)
                if fn:
                    function_result = fn(**arguments)
                else:
                    function_result = f"Неизвестный инструмент: {name}"

                # Сохраняем результаты инструментов в state
                if name == "get_weather":
                    weather_result = function_result
                elif name == "get_geolocation":
                    geolocation_result = function_result
                elif name == "get_costs":
                    costs_result = function_result
                elif name == "get_exchange_rate":
                    exchange_result = function_result

                # Результат возвращаем как json.dumps({"result": ...}) — паттерн из 6-го занятия
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"result": function_result}, ensure_ascii=False)
                })
        else:
            # LLM вернула финальный ответ — выходим из цикла
            final_answer = message.content or ""
            break

    return {
        "agent_response": final_answer,
        "weather": weather_result,
        "costs": costs_result,
        "exchange_rate": exchange_result,
        "messages": [{"role": "assistant", "content": final_answer}]
    }


def build_itinerary(state: TravelState) -> dict:
    destination = state.get("destination", "")
    weather = state.get("weather", "")
    costs = state.get("costs", "")
    exchange_rate = state.get("exchange_rate", "")
    budget = state.get("budget")
    dates = state.get("dates", "")
    comfort = state.get("comfort_level", "стандарт")
    passengers = state.get("passengers", 1)
    departure_city = state.get("departure_city", "Москва")
    interests = state.get("interests", "")
    special_requests = state.get("special_requests", "")
    api_key = state.get("api_key")

    # Предупреждение если погода недоступна
    weather_warning = ""
    if weather and ("недоступен" in weather.lower() or "ошибка" in weather.lower() or "временно недоступен" in weather.lower()):
        weather_warning = "\n⚠️ ВНИМАНИЕ: Прогноз погоды недоступен. План составлен без учёта погодных условий."
        weather = ""

    # Предупреждение если курс валют недоступен
    exchange_warning = ""
    if exchange_rate and ("недоступен" in exchange_rate.lower() or "ошибка" in exchange_rate.lower()):
        exchange_warning = "\n⚠️ ВНИМАНИЕ: Курс валют недоступен. План составлен без учёта актуального курса."
        exchange_rate = ""

    # Формируем контекст — пропускаем отсутствующие параметры
    context_parts = [
        f"Направление: {destination}",
        f"Даты: {dates}" if dates else "Даты: не указаны",
        f"Бюджет: {budget} руб. (общий на {passengers} чел.)" if budget else f"Бюджет: не указан (на {passengers} чел.)",
        f"Пассажиров: {passengers}",
        f"Откуда: {departure_city}" if departure_city else "Откуда: не указано",
        f"Комфорт: {comfort}",
        f"Погода: {weather}" if weather else "Погода: не проверена",
        f"Курс валют: {exchange_rate}" if exchange_rate else "Курс валют: не проверен",
        f"Интересы: {interests}" if interests else "Интересы: не указаны",
        f"Пожелания: {special_requests}" if special_requests else "Пожелания: не указаны",
    ]
    context = "\n".join(context_parts)

    dialog = [
        {"role": "system", "content": (
            "Ты — профессиональный туристический планировщик. Составь детальный план поездки.\n"
            "Включи: транспорт, проживание, достопримечательности, питание, бюджет по дням.\n"
            "Укажи ссылки на Aviasales, RZD, Booking.\n"
            "Используй год 2026 для всех дат.\n"
            "Если погода недоступна — укажи это в предупреждении.\n"
            "Бюджет — ОБЩИЙ на всех пассажиров. Цены должны быть реалистичными.\n\n"
            "Ориентиры цен (руб., 2026):\n"
            "- Авиабилеты Москва-Сочи: 8000-15000 туда-обратно\n"
            "- Авиабилеты Москва-Шерегеш (через Новосибирск): 12000-20000 туда-обратно\n"
            "- Проживание в Сочи (стандарт): 3000-5000/ночь\n"
            "- Проживание в Шерегеш (стандарт): 3500-5000/ночь\n"
            "- Проживание в Турции (all inclusive): 8000-15000/ночь\n"
            "- Ски-пасс Шерегеш: 2000-3000/день\n"
            "- Прокат сноуборда: 1500/день\n"
            "- Средний чек в кафе: 1000-1500/чел\n"
            "- Экскурсия в Луксор: 5000-8000/чел\n"
            "Не указывай цены, которых не существует. Используй реалистичные диапазоны."
        )},
        {"role": "user", "content": context + weather_warning},
    ]
    result = llm_call(dialog, max_tokens=1024, api_key=api_key)
    return {"itinerary": result, "messages": [{"role": "assistant", "content": result}]}


def save_state(state: TravelState) -> dict:
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "session_1")
    states_dir = Path("states")
    states_dir.mkdir(exist_ok=True)
    state_file = states_dir / f"{user_id}_{session_id}.json"
    data_to_save = {
        "user_id": user_id,
        "session_id": session_id,
        "destination": state.get("destination"),
        "departure_city": state.get("departure_city"),
        "dates": state.get("dates"),
        "budget": state.get("budget"),
        "comfort_level": state.get("comfort_level"),
        "passengers": state.get("passengers"),
        "interests": state.get("interests"),
        "special_requests": state.get("special_requests"),
        "itinerary": state.get("itinerary"),
        "weather": state.get("weather"),
        "costs": state.get("costs"),
        "exchange_rate": state.get("exchange_rate"),
        "saved_at": datetime.now().isoformat(),
        "conversation_history": [
            msg for msg in state.get("messages", [])
            if isinstance(msg, dict) and msg.get("role") in ("user", "assistant")
        ],
    }
    with open(state_file, "w") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    # Возвращаем itinerary в ответе
    itinerary = state.get("itinerary", "")
    if itinerary:
        agent_response = f"✅ **План поездки сохранён!**\n\n{itinerary}"
    else:
        agent_response = "План сохранён."
    
    return {
        "agent_response": agent_response,
        "itinerary": itinerary,
        "messages": [{"role": "assistant", "content": agent_response}]
    }


# --- Routing ---
def route_intent(state: TravelState) -> str:
    """Динамический роутинг с минимум 3 путями. БЕЗ ЦИКЛОВ.
    
    classify_intent уже обработал refine и обновил state.
    Здесь только маршрутизация по intent.
    """
    intent = state.get("intent", "unknown")
    clarification_count = state.get("clarification_count", 0)
    destination = state.get("destination")

    # Путь 1: recall — восстановить сохранённый план
    if intent == "recall":
        return "load_state"

    # Путь 2: unknown — общие вопросы, ответ и завершение
    if intent == "unknown":
        return "__end__"

    # Путь 3: refine — изменить существующий план
    if intent == "refine":
        if destination:
            return "check_weather_budget"
        return "__end__"

    # Путь 4: plan — спланировать новую поездку
    if intent == "plan":
        if destination:
            return "check_weather_budget"

    return "__end__"


# --- Graph ---
graph_builder = StateGraph(TravelState)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("load_state", load_state)
graph_builder.add_node("check_weather_budget", check_weather_budget)
graph_builder.add_node("build_itinerary", build_itinerary)
graph_builder.add_node("save_state", save_state)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent, {
    "load_state": "load_state",
    "check_weather_budget": "check_weather_budget",
    "__end__": END,
})
graph_builder.add_edge("load_state", END)
graph_builder.add_edge("check_weather_budget", "build_itinerary")
graph_builder.add_edge("build_itinerary", "save_state")
graph_builder.add_edge("save_state", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# --- Pydantic models ---
class ChatRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str = Field(default="session_1")
    message: str
    context: str = ""
    budget: Optional[float] = None
    dates: Optional[str] = None
    comfort_level: Optional[str] = None
    passengers: Optional[int] = None
    departure_city: Optional[str] = None
    interests: Optional[str] = None
    special_requests: Optional[str] = None
    transport: Optional[str] = None
    api_key: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    destination: Optional[str] = None
    itinerary: Optional[str] = None
    session_id: str


class PlanCreateRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str = Field(default="session_1")
    destination: Optional[str] = None
    departure_city: Optional[str] = None
    dates: Optional[str] = None
    budget: Optional[float] = None
    comfort_level: Optional[str] = None
    passengers: Optional[int] = None
    interests: Optional[str] = None
    special_requests: Optional[str] = None
    transport: Optional[str] = None
    api_key: Optional[str] = None


class PlanUpdateRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str = Field(default="session_1")
    destination: Optional[str] = None
    departure_city: Optional[str] = None
    dates: Optional[str] = None
    budget: Optional[float] = None
    comfort_level: Optional[str] = None
    passengers: Optional[int] = None
    interests: Optional[str] = None
    special_requests: Optional[str] = None
    transport: Optional[str] = None
    api_key: Optional[str] = None


# --- FastAPI app ---
app = FastAPI(title="Travel Planner Agent", version="1.0.0")


@app.post("/plan_create", response_model=ChatResponse)
async def plan_create(payload: PlanCreateRequest):
    """Создать новый план поездки — игнорирует старое состояние."""
    # Уникальный thread_id для каждого plan_create — чекпоинт будет свежим
    import time
    thread_id = f"{payload.user_id}_{payload.session_id}_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    # Формируем параметры — только те, что переданы
    params_dict = {k: v for k, v in payload.__dict__.items() if v is not None}
    user_query = f"/plan_create {json.dumps(params_dict, ensure_ascii=False)}"

    # Полное начальное состояние — граф выполнится с нуля
    initial_state: TravelState = {
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "intent": "plan",
        "user_query": user_query,
        "messages": [{"role": "user", "content": user_query}],
        "clarification_count": 0,
        "destination": payload.destination,
        "departure_city": payload.departure_city,
        "dates": payload.dates,
        "budget": payload.budget,
        "comfort_level": payload.comfort_level,
        "passengers": payload.passengers,
        "interests": payload.interests,
        "special_requests": payload.special_requests,
        "transport": payload.transport,
        "api_key": payload.api_key,
        # Явно сбрасываем — новые значения None
        "weather": None,
        "costs": None,
        "exchange_rate": None,
        "itinerary": None,
        "agent_response": "",
        "alternatives": None,
    }

    result = graph.invoke(initial_state, config)
    return ChatResponse(
        response=result.get("agent_response", ""),
        destination=result.get("destination"),
        itinerary=result.get("itinerary"),
        session_id=payload.session_id,
    )


@app.post("/plan_update", response_model=ChatResponse)
async def plan_update(payload: PlanUpdateRequest):
    """Обновить существующий план — сохраняет не изменённые параметры."""
    thread_id = f"{payload.user_id}_{payload.session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # Загружаем сохранённое состояние из JSON-файла
    states_dir = Path("states")
    state_file = states_dir / f"{payload.user_id}_{payload.session_id}.json"
    saved = {}
    if state_file.exists():
        with open(state_file) as f:
            saved = json.load(f)
        # Убираем поля, которых нет в TravelState
        saved = {k: v for k, v in saved.items() if k in TravelState.__annotations__}

    # Копируем ВСЕ параметры из сохранённого состояния
    # Переопределяем ТОЛЬКО переданные параметры
    initial_state: TravelState = {
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "intent": "refine",
        "user_query": f"/plan_update {json.dumps({k: v for k, v in payload.__dict__.items() if v is not None}, ensure_ascii=False)}",
        "messages": [{"role": "user", "content": f"/plan_update {json.dumps({k: v for k, v in payload.__dict__.items() if v is not None}, ensure_ascii=False)}"}],
        "clarification_count": 0,
        "destination": payload.destination if payload.destination is not None else saved.get("destination"),
        "departure_city": payload.departure_city if payload.departure_city is not None else saved.get("departure_city"),
        "dates": payload.dates if payload.dates is not None else saved.get("dates"),
        "budget": payload.budget if payload.budget is not None else saved.get("budget"),
        "comfort_level": payload.comfort_level if payload.comfort_level is not None else saved.get("comfort_level"),
        "passengers": payload.passengers if payload.passengers is not None else saved.get("passengers"),
        "interests": payload.interests if payload.interests is not None else saved.get("interests"),
        "special_requests": payload.special_requests if payload.special_requests is not None else saved.get("special_requests"),
        "transport": payload.transport if payload.transport is not None else saved.get("transport"),
        "api_key": payload.api_key if payload.api_key is not None else saved.get("api_key"),
        # Явно сбрасываем — будут пересчитаны
        "weather": None,
        "costs": None,
        "exchange_rate": None,
        "itinerary": None,
        "agent_response": "",
        "alternatives": None,
    }

    # Явно сбрасываем сообщения в чекпоинте — add_messages объединяет с историей,
    # поэтому без этого старые сообщения остаются и LLM не регенерирует itinerary
    graph.update_state(config, {"messages": []})

    result = graph.invoke(initial_state, config)
    return ChatResponse(
        response=result.get("agent_response", ""),
        destination=result.get("destination"),
        itinerary=result.get("itinerary"),
        session_id=payload.session_id,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    thread_id = f"{payload.user_id}_{payload.session_id}"
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    if snapshot.next:
        # Продолжаем с места остановки (уточнение параметров)
        current_values = snapshot.values
        current_values["user_query"] = payload.message
        current_values["messages"].append({"role": "user", "content": payload.message})
        if payload.budget is not None:
            current_values["budget"] = payload.budget
        if payload.dates:
            current_values["dates"] = payload.dates
        if payload.comfort_level:
            current_values["comfort_level"] = payload.comfort_level
        if payload.passengers:
            current_values["passengers"] = payload.passengers
        if payload.departure_city:
            current_values["departure_city"] = payload.departure_city
        if payload.interests:
            current_values["interests"] = payload.interests
        if payload.special_requests:
            current_values["special_requests"] = payload.special_requests
        if payload.api_key:
            current_values["api_key"] = payload.api_key
        graph.update_state(config, current_values)
        result = graph.invoke(None, config)
    else:
        # Граф завершён — читаем последнее состояние из checkpointer
        # и merge с новыми параметрами из payload
        saved_state = dict(snapshot.values) if snapshot.values else {}
        
        initial_state: TravelState = {
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "user_query": payload.message,
            "messages": [{"role": "user", "content": payload.message}],
            "api_key": payload.api_key,
        }
        # Merge: берём сохранённые параметры, переопределяем новыми
        for key in ("budget", "dates", "comfort_level", "passengers",
                    "departure_city", "interests", "special_requests"):
            if payload.__dict__.get(key) is not None:
                initial_state[key] = payload.__dict__[key]
            elif key in saved_state and saved_state[key] is not None:
                initial_state[key] = saved_state[key]
        # destination и другие поля
        for key in ("destination",):
            if key in saved_state and saved_state[key] is not None:
                initial_state[key] = saved_state[key]
        
        result = graph.invoke(initial_state, config)

    return ChatResponse(
        response=result.get("agent_response", ""),
        destination=result.get("destination"),
        itinerary=result.get("itinerary"),
        session_id=payload.session_id,
    )


@app.get("/state/{session_id}")
async def get_state(session_id: str, user_id: str = "default"):
    session_id = session_id
    states_dir = Path("states")
    state_file = states_dir / f"{user_id}_{session_id}.json"
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"error": "Состояние не найдено"}


@app.post("/reset/{session_id}")
async def reset_state(session_id: str, user_id: str = "default"):
    states_dir = Path("states")
    state_file = states_dir / f"{user_id}_{session_id}.json"
    if state_file.exists():
        state_file.unlink()
    return {"status": "Состояние сброшено"}


@app.get("/health")
async def health():
    resp = requests.get(f"{BASE_URL}/models", timeout=5.0)
    models = resp.json().get("data", [])
    available = [m["id"] for m in models if "GigaChat" in m["id"]]
    return {"status": "ok", "model": DEFAULT_MODEL, "llm": f"available models: {len(available)}"}