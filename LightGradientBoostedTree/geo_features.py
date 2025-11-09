import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import requests  # Optional dependency; used only when Grok integration is configured
except ImportError:  # pragma: no cover - fallback when requests is unavailable
    requests = None  # type: ignore


class GeoHazardFeatureStore:
    """
    Lightweight offline cache that approximates satellite / space-based hazard intelligence.

    The cache is optional; if it is missing the store reverts to analytical defaults that
    estimate hazard proximity using latitude bands. This makes it safe to deploy completely
    offline while still providing meaningful priors for the risk model.
    """

    def __init__(self, cache_path: Optional[str] = None) -> None:
        self.cache_path = cache_path or os.getenv(
            "GEO_HAZARD_CACHE",
            "data/geospatial/offline_hazard_tiles.parquet",
        )
        self.cache_df: Optional[pd.DataFrame] = self._load_cache(self.cache_path)

    def _load_cache(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path or not os.path.exists(path):
            return None

        try:
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"WARNING: Failed to load geospatial cache {path}: {exc}")
            return None

    def lookup(self, lat: float, lon: float, timestamp: Optional[float] = None) -> Dict[str, float]:
        """
        Return heuristically smoothed hazard metrics for the provided coordinate.
        When no cache is present, fall back to latitude-derived priors.
        """
        if self.cache_df is None or self.cache_df.empty:
            return self._default_response(lat, lon)

        coords = self.cache_df[["lat", "lon"]].to_numpy(dtype=float)
        deltas = coords - np.array([[lat, lon]])
        dist_sq = np.sum(np.square(deltas), axis=1)
        nearest_idx = int(np.argmin(dist_sq))
        row = self.cache_df.iloc[nearest_idx]

        return {
            "geo_disaster_distance_km": float(row.get("distance_km", 150.0)),
            "geo_space_weather_kp": float(row.get("space_weather_kp", 2.5)),
            "geo_population_density_norm": float(row.get("population_norm", 0.35)),
            "geo_satellite_latency_ms": float(row.get("sat_latency_ms", 220.0)),
        }

    @staticmethod
    def _default_response(lat: float, lon: float) -> Dict[str, float]:
        lat_clip = float(np.clip(lat, -90.0, 90.0))
        # wrap longitude to [-180, 180]
        lon_norm = ((lon + 180.0) % 360.0) - 180.0
        _ = lon_norm  # longitude currently unused in the heuristic

        equatorial_factor = 1.0 - abs(lat_clip) / 90.0

        hazard_distance = float(np.clip(140.0 - 80.0 * equatorial_factor, 30.0, 180.0))
        kp = float(np.clip(2.0 + 2.5 * equatorial_factor, 1.0, 7.0))
        density_norm = float(np.clip(0.25 + 0.5 * equatorial_factor, 0.15, 0.95))
        latency_ms = float(np.clip(240.0 - 80.0 * equatorial_factor, 120.0, 260.0))

        return {
            "geo_disaster_distance_km": hazard_distance,
            "geo_space_weather_kp": kp,
            "geo_population_density_norm": density_norm,
            "geo_satellite_latency_ms": latency_ms,
        }


class GrokGeoContextClient:
    """
    Minimal Grok API wrapper. The actual HTTP call is deferred so that hackathon teams can
    wire in their credentials without shipping secrets. When no API key is present the
    client simply returns ``None`` and the downstream feature engineer sticks to offline
    priors.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.x.ai/v1"):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.base_url = base_url.rstrip("/")

    def request_context(
        self,
        lat: float,
        lon: float,
        timestamp: Optional[float] = None,
        hazard_summary: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.api_key or requests is None:
            return None

        prompt = self._build_prompt(lat, lon, timestamp, hazard_summary)
        try:
            response = requests.post(  # type: ignore[operator]
                f"{self.base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": "grok-2-latest",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a geospatial risk analyst helping an autonomous medical dispatch agent. "
                                "Respond with concise JSON containing estimated disaster severity and responder ETA adjustments."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": 0.1,
                },
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not message:
                return None
            return self._safe_parse_json(message)
        except Exception as exc:  # pragma: no cover - network errors expected offline
            print(f"WARNING: Grok geospatial request failed, falling back to offline priors: {exc}")
            return None

    @staticmethod
    def _build_prompt(
        lat: float,
        lon: float,
        timestamp: Optional[float],
        hazard_summary: Optional[str],
    ) -> str:
        summary_line = hazard_summary or "No on-device hazard notes."
        ts_text = f"{timestamp}" if timestamp is not None else "unknown"
        return (
            f"Latitude: {lat:.6f}\n"
            f"Longitude: {lon:.6f}\n"
            f"Timestamp: {ts_text}\n"
            f"Local observations: {summary_line}\n"
            "Respond with JSON: {\"hazard_score\": <0-1>, \"hazard_label\": str, "
            "\"responder_delay_factor\": <float minutes>, \"confidence\": <0-1>}."
        )

    @staticmethod
    def _safe_parse_json(payload: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(payload)
        except Exception:
            # Grok might return JSON inside code fences; strip simple wrappers
            payload = payload.strip().strip("`")
            if payload.startswith("json"):
                payload = payload[4:].strip()
            try:
                return json.loads(payload)
            except Exception:
                return None


class GeoFeatureEngineer:
    """
    Converts geospatial intelligence from Grok + offline priors into numerical features that
    can be consumed by the LightGBM risk classifier. Also provides synthetic generation
    utilities so the model can be trained before the Grok integration is wired up.
    """

    FEATURE_NAMES = [
        "geo_lat",
        "geo_lon",
        "geo_disaster_distance_km",
        "geo_space_weather_kp",
        "geo_population_density_norm",
        "geo_satellite_latency_ms",
        "grok_hazard_score",
        "grok_confidence",
        "geo_alert_weight",
    ]

    def __init__(
        self,
        hazard_store: Optional[GeoHazardFeatureStore] = None,
        grok_client: Optional[GrokGeoContextClient] = None,
    ) -> None:
        self.hazard_store = hazard_store or GeoHazardFeatureStore()
        self.grok_client = grok_client
        self._fallback_defaults: Dict[str, float] = {
            "geo_lat": 0.0,
            "geo_lon": 0.0,
            "geo_disaster_distance_km": 120.0,
            "geo_space_weather_kp": 2.5,
            "geo_population_density_norm": 0.4,
            "geo_satellite_latency_ms": 200.0,
            "grok_hazard_score": 0.25,
            "grok_confidence": 0.45,
            "geo_alert_weight": 1.0,
        }

    def compute_features(
        self,
        lat: float,
        lon: float,
        timestamp: Optional[float] = None,
        hazard_summary: Optional[str] = None,
    ) -> Dict[str, float]:
        offline = self.hazard_store.lookup(lat, lon, timestamp)
        grok_payload = None
        if self.grok_client is not None:
            grok_payload = self.grok_client.request_context(lat, lon, timestamp, hazard_summary)

        if grok_payload:
            hazard_score = float(np.clip(grok_payload.get("hazard_score", 0.35), 0.0, 1.0))
            grok_confidence = float(np.clip(grok_payload.get("confidence", 0.6), 0.0, 1.0))
        else:
            hazard_score = self._fallback_defaults["grok_hazard_score"]
            grok_confidence = self._fallback_defaults["grok_confidence"]

        alert_weight = float(
            np.clip(
                0.9 + 0.7 * hazard_score + 0.15 * (1.0 - offline["geo_disaster_distance_km"] / 200.0),
                0.5,
                3.5,
            )
        )

        result = {
            "geo_lat": float(lat),
            "geo_lon": float(lon),
            **offline,
            "grok_hazard_score": hazard_score,
            "grok_confidence": grok_confidence,
            "geo_alert_weight": alert_weight,
        }

        self._fallback_defaults.update(
            {
                "geo_disaster_distance_km": offline["geo_disaster_distance_km"],
                "geo_space_weather_kp": offline["geo_space_weather_kp"],
                "geo_population_density_norm": offline["geo_population_density_norm"],
                "geo_satellite_latency_ms": offline["geo_satellite_latency_ms"],
            }
        )

        return result

    def generate_synthetic_geo_features(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        rng = rng or np.random.RandomState()
        lat = float(rng.uniform(-55.0, 55.0))
        lon = float(rng.uniform(-180.0, 180.0))
        offline = self.hazard_store._default_response(lat, lon)

        disaster_distance = float(
            np.clip(
                offline["geo_disaster_distance_km"] + rng.normal(loc=0.0, scale=18.0),
                5.0,
                220.0,
            )
        )
        space_weather = float(np.clip(offline["geo_space_weather_kp"] + rng.normal(0.0, 0.6), 0.0, 9.0))
        density_norm = float(
            np.clip(offline["geo_population_density_norm"] + rng.normal(0.0, 0.1), 0.05, 1.0)
        )
        latency_ms = float(np.clip(offline["geo_satellite_latency_ms"] + rng.normal(0.0, 25.0), 80.0, 320.0))

        hazard_score = float(
            np.clip(
                0.5 * (1.0 - disaster_distance / 250.0) + 0.35 * (density_norm) + 0.1 * (space_weather / 9.0),
                0.0,
                1.0,
            )
        )
        grok_confidence = float(np.clip(0.55 + 0.35 * (1.0 - disaster_distance / 300.0), 0.2, 0.95))
        alert_weight = float(np.clip(0.85 + 0.8 * hazard_score + 0.2 * grok_confidence, 0.5, 3.5))

        return {
            "geo_lat": lat,
            "geo_lon": lon,
            "geo_disaster_distance_km": disaster_distance,
            "geo_space_weather_kp": space_weather,
            "geo_population_density_norm": density_norm,
            "geo_satellite_latency_ms": latency_ms,
            "grok_hazard_score": hazard_score,
            "grok_confidence": grok_confidence,
            "geo_alert_weight": alert_weight,
        }

    def get_feature_defaults(self) -> Dict[str, float]:
        return dict(self._fallback_defaults)


def enrich_dataframe_with_geo(
    df: pd.DataFrame,
    engineer: GeoFeatureEngineer,
    random_seed: int = 42,
    lat_column: Optional[str] = None,
    lon_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Attach geospatial features to every row of ``df``. If latitude / longitude columns are
    supplied they will be used; otherwise synthetic-yet-realistic coordinates are drawn.
    """
    if df is None or df.empty:
        return df

    result = df.copy()
    rng = np.random.RandomState(random_seed)
    geo_rows = []

    use_columns = lat_column in result.columns and lon_column in result.columns if lat_column and lon_column else False

    for idx in range(len(result)):
        if use_columns:
            lat = float(result.iloc[idx][lat_column])
            lon = float(result.iloc[idx][lon_column])
            features = engineer.compute_features(lat, lon)
        else:
            features = engineer.generate_synthetic_geo_features(rng=rng)
        geo_rows.append(features)

    geo_df = pd.DataFrame(geo_rows)
    for col in GeoFeatureEngineer.FEATURE_NAMES:
        result[col] = geo_df[col].values

    return result
