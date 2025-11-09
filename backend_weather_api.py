"""
Geospatial Intelligence Layer: Weather & Hazard API Integration
Uses open-source weather APIs to detect environmental hazards
"""
import os
import time
import requests
from typing import Dict, Any, Optional
import json


class WeatherHazardService:
    """
    Integrates weather data to compute hazard index.
    Uses OpenWeatherMap API (free tier) or Open-Meteo (no API key needed)
    """
    
    def __init__(self):
        # Open-Meteo (no API key required, good for MVP)
        self.open_meteo_url = "https://api.open-meteo.com/v1/forecast"
        # OpenWeatherMap (requires API key, better data)
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "")
        self.openweather_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_hazard_index(self, lat: float, lon: float, timestamp: int) -> Dict[str, Any]:
        """
        Compute hazard index (0.0-1.0) based on weather conditions.
        Returns: {
            "hazard_index": float,
            "raw": {...weather data...},
            "hazards": [...list of active hazards...]
        }
        """
        try:
            # Try OpenWeatherMap first if API key available
            if self.openweather_api_key:
                weather_data = self._fetch_openweather(lat, lon)
            else:
                # Fallback to Open-Meteo (no API key needed)
                weather_data = self._fetch_openmeteo(lat, lon)
            
            # Compute hazard index from weather data
            hazard_index, hazards = self._compute_hazard_index(weather_data)
            
            return {
                "hazard_index": hazard_index,
                "raw": weather_data,
                "hazards": hazards,
                "timestamp": timestamp
            }
        except Exception as e:
            # Fallback: return low hazard if API fails
            return {
                "hazard_index": 0.1,
                "raw": {"error": str(e)},
                "hazards": [],
                "timestamp": timestamp
            }
    
    def _fetch_openweather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch from OpenWeatherMap API"""
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.openweather_api_key,
            "units": "metric"
        }
        response = requests.get(self.openweather_url, params=params, timeout=5.0)
        response.raise_for_status()
        return response.json()
    
    def _fetch_openmeteo(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch from Open-Meteo API (no API key required)"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto"
        }
        response = requests.get(self.open_meteo_url, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        # Transform to similar format as OpenWeatherMap
        current = data.get("current", {})
        return {
            "main": {
                "temp": current.get("temperature_2m", 20),
                "humidity": current.get("relative_humidity_2m", 50),
                "pressure": 1013  # Default
            },
            "weather": [{
                "main": self._weather_code_to_main(current.get("weather_code", 0)),
                "description": "unknown"
            }],
            "wind": {
                "speed": current.get("wind_speed_10m", 0)
            },
            "rain": {
                "1h": current.get("precipitation", 0)
            }
        }
    
    def _weather_code_to_main(self, code: int) -> str:
        """Convert WMO weather code to main weather type"""
        # WMO Weather codes: https://open-meteo.com/en/docs
        if code in [71, 73, 75, 77, 85, 86]:  # Snow
            return "Snow"
        elif code in [51, 52, 53, 54, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:  # Rain
            return "Rain"
        elif code in [95, 96, 99]:  # Thunderstorm
            return "Thunderstorm"
        elif code in [45, 48]:  # Fog
            return "Fog"
        else:
            return "Clear"
    
    def _compute_hazard_index(self, weather_data: Dict[str, Any]) -> tuple[float, list]:
        """
        Compute hazard index (0.0-1.0) and list of active hazards.
        Higher values = more dangerous conditions.
        """
        hazards = []
        hazard_index = 0.1  # Base level
        
        main = weather_data.get("main", {})
        weather = weather_data.get("weather", [{}])[0]
        wind = weather_data.get("wind", {})
        rain = weather_data.get("rain", {})
        
        # Temperature extremes
        temp = main.get("temp", 20)
        if temp > 40:  # Extreme heat (>40°C)
            hazards.append("extreme_heat")
            hazard_index = max(hazard_index, 0.7)
        elif temp < -20:  # Extreme cold (<-20°C)
            hazards.append("extreme_cold")
            hazard_index = max(hazard_index, 0.6)
        
        # Precipitation (flood risk)
        precip_1h = rain.get("1h", 0)
        if precip_1h > 50:  # Heavy rain (>50mm/hr)
            hazards.append("flash_flood_warning")
            hazard_index = max(hazard_index, 0.95)
        elif precip_1h > 20:  # Moderate rain
            hazards.append("flood_warning")
            hazard_index = max(hazard_index, 0.7)
        
        # Wind (hurricane/cyclone risk)
        wind_speed = wind.get("speed", 0)
        if wind_speed > 25:  # Strong wind (>25 m/s ≈ 90 km/h)
            hazards.append("extreme_wind")
            hazard_index = max(hazard_index, 0.8)
        
        # Weather type
        weather_main = weather.get("main", "").upper()
        if "THUNDERSTORM" in weather_main:
            hazards.append("thunderstorm")
            hazard_index = max(hazard_index, 0.6)
        if "SNOW" in weather_main and temp < 0:
            hazards.append("snow_ice")
            hazard_index = max(hazard_index, 0.5)
        
        # Wildfire risk (high temp + low humidity)
        humidity = main.get("humidity", 50)
        if temp > 30 and humidity < 30:
            hazards.append("wildfire_risk_high")
            hazard_index = max(hazard_index, 0.8)
        
        return float(hazard_index), hazards
