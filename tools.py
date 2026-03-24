import os
import json
import requests

def get_current_weather(location):
    api_key = os.environ.get("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
    
    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    weather_info = data["current"]

    return json.dumps({
        "location": data["location"]["name"],
        "temperature_c": weather_info["temp_c"],
        "temperature_f": weather_info["temp_f"],
        "condition": weather_info["condition"]["text"],
        "humidity": weather_info["humidity"],
        "wind_kph": weather_info["wind_kph"],
    })


def get_weather_forecast(location, days=3):
    api_key = os.environ.get("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days={days}&aqi=no"

    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return f"Error: {data['error']['message']}"

    forecast_days = data["forecast"]["forecastday"]
    forecast_data = []

    for day in forecast_days:
        forecast_data.append({
            "date": day["date"],
            "max_temp_c": day["day"]["maxtemp_c"],
            "min_temp_c": day["day"]["mintemp_c"],
            "condition": day["day"]["condition"]["text"],
            "chance_of_rain": day["day"]["daily_chance_of_rain"],
        })

    return json.dumps({
        "location": data["location"]["name"],
        "forecast": forecast_data,
    })


def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)