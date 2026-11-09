"""
FastAPI-сервис для Travel Planner агента.
Запуск: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI
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

# Глобальный клиент (будет перезаписан при каждом запросе с новым ключом)
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
    COSTS_DB = {
        "Turkey": {"flight": 350, "hotel_per_day": 40, "food_per_day": 15, "activities_per_day": 10},
        "Jordan": {"flight": 450, "hotel_per_day": 50, "food_per_day": 20, "activities_per_day": 30},
        "Thailand": {"flight": 500, "hotel_per_day": 25, "food_per_day": 10, "activities_per_day": 15},
        "Egypt": {"flight": 400, "hotel_per_day": 35, "food_per_day": 15, "activities_per_day": 20},
        "UAE": {"flight": 450, "hotel_per_day": 60, "food_per_day": 25, "activities_per_day": 20},
        "Georgia": {"flight": 250, "hotel_per_day": 30, "food_per_day": 12, "activities_per_day": 10},
    }
    try:
        matched_key = None
        for key in COSTS_DB:
            if key.lower() in location.lower():
                matched_key = key
                break
        if not matched_key:
            return f"Нет данных для '{location}'."
        costs = COSTS_DB[matched_key]
        flight = costs["flight"]
        hotel = costs["hotel_per_day"] * days
        food = costs["food_per_day"] * days
        activities = costs["activities_per_day"] * days
        total = flight + hotel + food + activities
        return f"**{matched_key}** ({days} дн.): перелёт ${flight}, отель ${hotel}, питание ${food}, развлечения ${activities}, **итого ~${total}**"
    except Exception as e:
        return f"Ошибка стоимости: {e}."


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


def get_accommodation_prices(location, check_in, check_out, guests=1) -> str:
    try:
        from datetime import datetime
        start = datetime.strptime(check_in, "%Y-%m-%d")
        end = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (end - start).days
        accommodation_types = {
            "hostel": 15, "budget_hotel": 40, "standard_hotel": 80,
            "premium_hotel": 180, "apartment": 60, "villa": 250,
        }
        result = f"**Жильё в {location}** ({nights} ночей):\n"
        for key, price in accommodation_types.items():
            total = price * nights
            result += f"- {key}: ${price}/ночь (всего: ${total})\n"
        return result
    except Exception as e:
        return f"Ошибка жилья: {e}."


def get_transport_options(from_city, to_city, travel_date, passengers=1) -> str:
    try:
        transport_types = {
            "flight": {"price": 150, "duration": "2-4 часа", "url": "https://www.aviasales.ru"},
            "train": {"price": 80, "duration": "6-12 часов", "url": "https://www.rzd.ru"},
            "bus": {"price": 40, "duration": "8-15 часов", "url": "https://www.pulscen.ru"},
            "car_rental": {"price": 50, "duration": "гибко", "url": "https://www.rentalcars.com"},
        }
        result = f"**Транспорт {from_city} -> {to_city}** ({travel_date}):\n"
        for key, info in transport_types.items():
            result += f"- {key}: ${info['price']}/чел, {info['duration']}, {info['url']}\n"
        return result
    except Exception as e:
        return f"Ошибка транспорта: {e}."


# --- LLM Call ---
def llm_call(dialog, structure=None, model=DEFAULT_MODEL, temperature=0.0, max_tokens=4096):
    request_params = {
        "model": model,
        "messages": dialog,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if structure is not None:
        response = client.chat.completions.parse(
            model=model, messages=dialog, temperature=temperature,
            max_tokens=max_tokens, response_format=structure,
        )
        return response.choices[0].message.parsed.model_dump()
    else:
        response = client.chat.completions.create(**request_params)
        return response.choices[0].message.content


# --- Node functions ---
def classify_intent(state: TravelState) -> dict:
    from pydantic import BaseModel
    import re
    class IntentFormat(BaseModel):
        intent: Literal["plan", "refine", "recall", "unknown"]
        missing_fields: List[str] = []
    user_query = state.get("user_query", "")
    user_query_lower = user_query.lower()
    messages = state.get("messages", [])
    
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
    
    result = llm_call(dialog, structure=IntentFormat)
    
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
    from pydantic import BaseModel
    class DestinationFormat(BaseModel):
        destination: str
        reason: str
        temperature: str
        best_season: str
    user_query = state.get("user_query", "").lower()
    
    # Проверяем тип активности
    is_ski = any(kw in user_query for kw in ["горнолыж", "лыж", "сноуборд", "ski", "снег"])
    is_beach = any(kw in user_query for kw in ["пляж", "море", "отдых", "тропик", "солнц"])
    is_culture = any(kw in user_query for kw in ["экскурс", "культ", "истор", "музей"])
    
    context = ""
    if is_ski:
        context = (
            "Горнолыжные направления (декабрь-март):\n"
            "- Россия: Домбай, Архыз, Красная Поляна, Эльтон, УдГУ\n"
            "- Грузия: Гудаури, Бакуриани\n"
            "- Казахстан: Шымбулак, Медеу\n\n"
            "НЕ подходит: ОАЭ, Египет, Турция (там нет горнолыжных курортов для отдыха)!"
        )
    elif is_beach:
        context = (
            "Пляжные направления:\n"
            "- Египет: Хургада, Шарм-эш-Шейх (март-октябрь)\n"
            "- Турция: Анталья, Бодрум (май-октябрь)\n"
            "- Таиланд: Пхукет, Самуи (ноябрь-март)\n"
            "- Мальдивы: круглый год\n"
            "- ОАЭ: Дубай, Шарджа (ноябрь-март)"
        )
    elif is_culture:
        context = (
            "Культурные направления:\n"
            "- Иордания: Амман, Петра, Вади-Рам\n"
            "- Грузия: Тбилиси, Казбеги\n"
            "- Турция: Стамбул, Каппадокия\n"
            "- Египет: Каир, Луксор, Александрия"
        )
    else:
        context = (
            "Доступные направления:\n"
            "- Горнолыжные (Россия, Грузия): декабрь-март\n"
            "- Пляжные (Египет, Турция): март-октябрь\n"
            "- Тропики (Таиланд, Мальдивы): ноябрь-март\n"
            "- Ближний Восток (Иордания, ОАЭ): март-май, сентябрь-ноябрь\n"
            "- Кавказ (Грузия): круглый год\n\n"
            "Выбери направление, которое соответствует запросу пользователя по сезону и активности."
        )
    
    dialog = [
        {"role": "system", "content": "Подбери направление. Ответь 1 предложением. " + context},
        {"role": "user", "content": user_query},
    ]
    result = llm_call(dialog, structure=DestinationFormat)
    return {"destination": result["destination"],
            "messages": [{"role": "assistant", "content": f"Рекомендую: {result['destination']}"}]}


# --- Быстрый выбор курорта (без LLM) ---
SKI_RESORTS_RUSSIA = {
    "домбай": {"name": "Домбай, Карачаево-Черкесия", "lat": 43.44, "lon": 41.64, "note": "Кавказ, хороший снег до мая"},
    "архыз": {"name": "Архыз, Карачаево-Черкесия", "lat": 43.95, "lon": 41.10, "note": "Кавказ, новый курорт"},
    "красная поляна": {"name": "Красная Поляна (Роза Хутор), Сочи", "lat": 43.40, "lon": 39.85, "note": "Кавказ, близко к аэропорту"},
    "эштер": {"name": "Эстосадок/Роза Хутор, Сочи", "lat": 43.40, "lon": 39.85, "note": "Кавказ"},
    "шерегеш": {"name": "Шерегеш, Кемеровская область", "lat": 51.90, "lon": 87.20, "note": "Горная Шория, сухой снег, сезон до июня"},
    "манжерок": {"name": "Манжерок, Горный Алтай", "lat": 51.67, "lon": 86.40, "note": "Алтай, хороший сезон"},
    "сокольская": {"name": "Сокольская, Ленинградская область", "lat": 60.35, "lon": 31.15, "note": "Под Санкт-Петербургом"},
}

BEACH_RESORTS = {
    "хургада": {"name": "Хургада, Египет", "lat": 27.25, "lon": 33.80},
    "шарм": {"name": "Шарм-эш-Шейх, Египет", "lat": 27.90, "lon": 34.33},
    "анталья": {"name": "Анталья, Турция", "lat": 36.89, "lon": 30.71},
    "бодрум": {"name": "Бодрум, Турция", "lat": 37.03, "lon": 27.43},
    "дубай": {"name": "Дубай, ОАЭ", "lat": 25.20, "lon": 55.27},
    "тбилиси": {"name": "Тбилиси, Грузия", "lat": 41.72, "lon": 44.79},
    "сочи": {"name": "Сочи, Россия", "lat": 43.59, "lon": 39.72},
    "казань": {"name": "Казань, Россия", "lat": 55.79, "lon": 49.11},
    "москва": {"name": "Москва, Россия", "lat": 55.76, "lon": 37.62},
}


def select_or_validate_destination(state: TravelState) -> dict:
    """
    Быстрый выбор/валидация направления без вызова LLM.
    Если пользователь написал страну + активность — подбирает конкретный курорт.
    """
    user_query = state.get("user_query", "").lower()
    destination = state.get("destination", "").lower()
    
    # Проверяем активность
    is_ski = any(kw in user_query for kw in ["сноуборд", "горнолыж", "лыж", "ski", "снег", "катание"])
    is_beach = any(kw in user_query for kw in ["пляж", "море", "отдых", "тропик", "солнц", "купание"])
    is_culture = any(kw in user_query for kw in ["экскурс", "культ", "истор", "музей", "город"])
    
    # Если destination уже конкретный курорт — возвращаем как есть
    for resort_name in SKI_RESORTS_RUSSIA:
        if resort_name in destination or destination in resort_name:
            return {"destination": SKI_RESORTS_RUSSIA[resort_name]["name"],
                    "messages": [{"role": "assistant", "content": f"Направление: {SKI_RESORTS_RUSSIA[resort_name]['name']}"}]}
    
    for resort_name in BEACH_RESORTS:
        if resort_name in destination or destination in resort_name:
            return {"destination": BEACH_RESORTS[resort_name]["name"],
                    "messages": [{"role": "assistant", "content": f"Направление: {BEACH_RESORTS[resort_name]['name']}"}]}
    
    # Если пользователь написал страну + активность — подбираем курорт
    if "россия" in destination or "россии" in destination:
        if is_ski:
            # Для горнолыжки в апреле лучше всего Шерегеш (снег до июня) или Домбай
            selected = "Шерегеш, Кемеровская область"
            return {"destination": selected,
                    "messages": [{"role": "assistant", "content": f"Для горнолыжки в апреле рекомендую: {selected} (снег до июня)"}]}
        elif is_beach:
            selected = "Сочи, Россия"
            return {"destination": selected,
                    "messages": [{"role": "assistant", "content": f"Для пляжного отдыха: {selected}"}]}
        elif is_culture:
            selected = "Казань, Россия"
            return {"destination": selected,
                    "messages": [{"role": "assistant", "content": f"Для культурной поездки: {selected}"}]}
        else:
            # По умолчанию — Сочи
            selected = "Сочи, Россия"
            return {"destination": selected,
                    "messages": [{"role": "assistant", "content": f"Рекомендую: {selected}"}]}
    
    # Если destination — это конкретный город/курорт
    if destination and len(destination) > 3:
        return {"destination": state.get("destination", ""),
                "messages": [{"role": "assistant", "content": f"Направление: {state.get('destination', '')}"}]}
    
    # Если ничего не подошло — возвращаем как есть
    return {"destination": state.get("destination", ""),
            "messages": [{"role": "assistant", "content": f"Направление: {state.get('destination', '')}"}]}


def ask_missing_info(state: TravelState) -> dict:
    from pydantic import BaseModel
    class MissingInfoFormat(BaseModel):
        need_clarification: bool
        missing_fields: List[str]
        query_to_user: str
    user_query = state.get("user_query", "")
    
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
    result = llm_call(dialog, structure=MissingInfoFormat, max_tokens=1024)
    return {
        "agent_response": result["query_to_user"],
        "messages": [{"role": "assistant", "content": result["query_to_user"]}],
        "asked_missing_info": True,
    }


def load_state(state: TravelState) -> dict:
    session_id = state.get("session_id", "")
    states_dir = Path("states")
    state_file = states_dir / f"{state.get('user_id', 'default')}_{session_id}.json"
    if state_file.exists():
        with open(state_file) as f:
            saved = json.load(f)
        response = f"Данные загружены! Направление: {saved.get('destination', 'N/A')}"
    else:
        response = f"Состояние '{session_id}' не найдено."
    return {"agent_response": response,
            "messages": [{"role": "assistant", "content": response}]}


def check_weather_budget(state: TravelState) -> dict:
    destination = state.get("destination", "")
    dates = state.get("dates", "2026-02-10,2026-02-17")
    budget = state.get("budget", 70000)
    comfort = state.get("comfort_level", "standard")
    weather_result = get_weather_forecast(destination, dates)
    costs_result = get_travel_costs(destination, 7, comfort)
    weather_ok = "ошибка" not in weather_result.lower()
    budget_ok = True
    return {"weather": weather_result, "costs": costs_result,
            "weather_ok": weather_ok, "budget_ok": budget_ok,
            "comfort_level": comfort,
            "messages": [{"role": "assistant", "content": f"Погода OK={weather_ok}, бюджет OK={budget_ok}"}]}


def suggest_alternatives(state: TravelState) -> dict:
    destination = state.get("destination", "")
    user_query = state.get("user_query", "")
    dialog = [
        {"role": "system", "content": """Ты — Travel Planner. Предложи 2-3 альтернативные направления для поездки, которые лучше соответствуют запросу пользователя.
Укажи: название, почему подходит, примерная стоимость, лучший сезон. Кратко."""},
        {"role": "user", "content": f"Выбрано: {destination}. Запрос: {user_query}"},
    ]
    result = llm_call(dialog, model=DEFAULT_MODEL)
    return {"alternatives": result,
            "messages": [{"role": "assistant", "content": result}]}


def build_itinerary(state: TravelState) -> dict:
    destination = state.get("destination", "")
    weather = state.get("weather", "")
    costs = state.get("costs", "")
    budget = state.get("budget", 70000)
    dates = state.get("dates", "")
    comfort = state.get("comfort_level", "standard")
    user_query = state.get("user_query", "")
    dialog = [
        {"role": "system", "content": """Ты — профессиональный туристический планировщик.
Составь детальный план поездки на основе ВСЕХ данных. Используй точные даты из запроса.
Включи: транспорт, проживание, достопримечательности, питание, бюджет по дням.
Будь конкретным и практичным."""},
        {"role": "user", "content": f"Направление: {destination}\nДаты: {dates}\nБюджет: {budget} руб.\nУровень комфорта: {comfort}\nПогода: {weather}\nСтоимость: {costs}\nЗапрос: {user_query}"},
    ]
    result = llm_call(dialog, model=REASONING_MODEL, max_tokens=8192)
    return {"itinerary": result,
            "messages": [{"role": "assistant", "content": result}]}


def save_state(state: TravelState) -> dict:
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
    return {"agent_response": f"План сохранён в {state_file}",
            "messages": [{"role": "assistant", "content": "План сохранён."}]}


# --- Routing ---
def route_intent(state: TravelState) -> str:
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
        # Если все поля уже есть — пропускаем ask_missing_info и select_destination
        if (state.get("destination") and 
            state.get("dates") and 
            state.get("budget") and 
            state.get("comfort_level")):
            # Если destination уже конкретный — сразу проверяем
            destination = state.get("destination", "").lower()
            is_specific = any(kw in destination for kw in ["домбай", "архыз", "шерегеш", "красн", "сочи", "хурга", "шарм", "антал", "дубай", "тбилиси", "казань"])
            if is_specific:
                return "check_weather_budget"
            # Если destination — страна (Россия) — выбираем курорт
            return "select_or_validate_destination"
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
    weather_ok = state.get("weather_ok", True)
    budget_ok = state.get("budget_ok", True)
    if not weather_ok or not budget_ok:
        return "suggest_alternatives"
    return "build_itinerary"


# --- Graph ---
graph_builder = StateGraph(TravelState)
graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("select_destination", select_destination)
graph_builder.add_node("select_or_validate_destination", select_or_validate_destination)
graph_builder.add_node("ask_missing_info", ask_missing_info)
graph_builder.add_node("load_state", load_state)
graph_builder.add_node("check_weather_budget", check_weather_budget)
graph_builder.add_node("suggest_alternatives", suggest_alternatives)
graph_builder.add_node("build_itinerary", build_itinerary)
graph_builder.add_node("save_state", save_state)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent", route_intent, {
    "select_destination": "select_destination",
    "select_or_validate_destination": "select_or_validate_destination",
    "ask_missing_info": "ask_missing_info",
    "load_state": "load_state",
    "__end__": END,
})
graph_builder.add_edge("select_destination", "check_weather_budget")
graph_builder.add_edge("select_or_validate_destination", "check_weather_budget")
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
    # Используем ключ из запроса, если передан
    if payload.api_key:
        global client
        client = OpenAI(api_key=payload.api_key, base_url=BASE_URL)
    
    initial_state: TravelState = {
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "user_query": payload.message,
        "budget": payload.budget,
        "dates": payload.dates,
        "comfort_level": payload.comfort_level,
        "messages": [{"role": "user", "content": payload.context}],  # история в messages
    }
    config = {"configurable": {"thread_id": f"{payload.user_id}_{payload.session_id}"}}
    result = graph.invoke(initial_state, config)
    return ChatResponse(
        response=result.get("agent_response", ""),
        destination=result.get("destination"),
        itinerary=result.get("itinerary"),
        session_id=payload.session_id,
    )


@app.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    # Используем ключ из запроса, если передан
    if payload.api_key:
        global client
        client = OpenAI(api_key=payload.api_key, base_url=BASE_URL)
    
    initial_state: TravelState = {
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "user_query": payload.message,
        "budget": payload.budget,
        "dates": payload.dates,
        "comfort_level": payload.comfort_level,
        "messages": [{"role": "user", "content": payload.context}],  # история в messages
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


@app.get("/state/{session_id}")
async def get_state(session_id: str, user_id: str = "default"):
    states_dir = Path("states")
    state_file = states_dir / f"{user_id}_{session_id}.json"
    if state_file.exists():
        with open(state_file) as f:
            data = json.load(f)
        return {"status": "found", "data": data}
    else:
        return {"status": "not_found", "message": f"Сессия {session_id} не найдена"}


@app.post("/reset/{session_id}")
async def reset_state(session_id: str, user_id: str = "default"):
    states_dir = Path("states")
    state_file = states_dir / f"{user_id}_{session_id}.json"
    if state_file.exists():
        state_file.unlink()
        return {"status": "reset", "message": f"Сессия {session_id} очищена"}
    else:
        return {"status": "not_found", "message": f"Сессия {session_id} не найдена"}


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}
