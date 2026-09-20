"""Read-only configured-location weather answers from Open-Meteo, spoken by the client."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
import time
import requests
from .observability import timed


@dataclass(frozen=True)
class WeatherQuery:
    period: str = 'current'


def parse_weather(text, config=None):
    location = (config or {}).get('weather', {})
    names = [location.get('name', ''), *location.get('aliases', [])]
    location_words = set(re.findall(r'[a-z0-9]+', ' '.join(names).lower()))
    words = re.findall(r'[a-z0-9]+', text.lower())
    if not set(words) & {'weather', 'forecast'} and not ('temperature' in words and bool(location_words.intersection(words))):
        return None
    if set(words) & {'turn', 'switch', 'set', 'start', 'stop', 'cancel', 'open', 'close'}:
        return None
    allowed = set('what whats s is the weather forecast report outlook temperature like in for at today tomorrow current currently now right outside here local tell me about please give us how will be going to it look looks does look can you get show this day'.split()) | location_words
    if set(words) - allowed:
        return WeatherQuery('unsupported')
    if 'tomorrow' in words:
        return WeatherQuery('tomorrow')
    if 'today' in words or 'forecast' in words:
        return WeatherQuery('today')
    return WeatherQuery()


CONDITIONS = {0: 'clear skies', 1: 'mainly clear skies', 2: 'partly cloudy skies', 3: 'overcast skies',
    45: 'fog', 48: 'freezing fog', 51: 'light drizzle', 53: 'drizzle', 55: 'heavy drizzle',
    56: 'light freezing drizzle', 57: 'freezing drizzle', 61: 'light rain', 63: 'rain', 65: 'heavy rain',
    66: 'light freezing rain', 67: 'freezing rain', 71: 'light snow', 73: 'snow', 75: 'heavy snow',
    77: 'snow grains', 80: 'light rain showers', 81: 'rain showers', 82: 'heavy rain showers',
    85: 'light snow showers', 86: 'heavy snow showers', 95: 'thunderstorms',
    96: 'thunderstorms with hail', 99: 'severe thunderstorms with hail'}


def number(value, low=-100, high=100):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Missing or invalid weather value')
    return str(round(value))


class WeatherService:
    def __init__(self, config, now=time.time):
        self.location = config.get('weather', {})
        self.name = self.location.get('name', 'your configured location')
        self.now = now

    def answer(self, query):
        if query.period == 'unsupported':
            return f'I can give you the current weather, or the forecast for today or tomorrow, in {self.name}.'
        if any(self.location.get(key) is None for key in ('latitude', 'longitude')):
            return 'Please configure a weather location first.'
        with timed('weather.GET.open_meteo'):
            response = requests.get('https://api.open-meteo.com/v1/forecast', params={
                'latitude': self.location['latitude'],
                'longitude': self.location['longitude'],
                'timezone': self.location.get('timezone', 'auto'), 'forecast_days': 2,
                'temperature_unit': 'celsius', 'wind_speed_unit': 'kmh',
                'current': 'temperature_2m,apparent_temperature,weather_code,wind_speed_10m',
                'daily': 'weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max'
            }, timeout=(4, 10))
            response.raise_for_status()
            data = response.json()
        return self.format(data, query.period)

    def format(self, data, period):
        current, daily = data['current'], data['daily']
        tz = timezone(timedelta(seconds=data['utc_offset_seconds']))
        observed = datetime.fromisoformat(current['time']).replace(tzinfo=tz)
        age = self.now() - observed.timestamp()
        if age < -1800 or age > 7200:
            raise ValueError('Weather data is stale')
        date = datetime.fromtimestamp(self.now(), tz).date() + timedelta(days=period == 'tomorrow')
        index = daily['time'].index(date.isoformat())
        high = number(daily['temperature_2m_max'][index])
        low = number(daily['temperature_2m_min'][index])
        chance = number(daily['precipitation_probability_max'][index], 0, 100)
        when = 'Tomorrow' if period == 'tomorrow' else 'Today'
        if period == 'current':
            temperature = number(current['temperature_2m'])
            feel = number(current['apparent_temperature'])
            wind = number(current['wind_speed_10m'], 0, 400)
            condition = CONDITIONS.get(current['weather_code'])
            reply = f'In {self.name}, it is currently {temperature} degrees Celsius'
            reply += f' with {condition}.' if condition else '.'
            if feel != temperature:
                reply += f' It feels like {feel} degrees.'
            reply += f' Wind is {wind} kilometres per hour.'
            reply += f' Today, expect a high of {high} and a low of {low} degrees, with a {chance} percent chance of precipitation.'
        else:
            condition = CONDITIONS.get(daily['weather_code'][index])
            reply = f'{when} in {self.name}, expect ' + (condition+', ' if condition else '')
            reply += f'a high of {high} and a low of {low} degrees Celsius, with a {chance} percent chance of precipitation.'
        return reply
