 # grok_agent.py
import json
import time
from typing import Dict, Any

import numpy as np
import joblib
import requests

from sos_agent import extract_ecg, extract_acc, compute_hazard_weighted_risk


MODEL_PATH = "sos_gbt_model.joblib"

GROK_API_URL = "https://api.x.ai/v1/chat/completions"  # update to actual endpoint
GROK_API_KEY = "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"

HAZARD_API_URL = "https://your-hazard-api.example.com/alerts"  # update
FIRST_RESPONDER_WEBHOOK_URL = "https://your-backend.example.com/first_responders"
NEARBY_BROADCAST_URL = "https://your-backend.example.com/nearby_broadcast"


class GrokDistressAgent:
    """
    Hybrid AI agent that:
      - Uses LightGBM SOS model for physiology + kinematics.
      - Uses geo-hazard API for environment risk.
      - Uses Grok xAI for reasoning, triage, and cluster intelligence.
      - Issues autonomous SOS & local bystander alerts.
      - Falls back to offline thresholds if network / Grok is unavailable.
    """

class GrokDistressAgent:
    """
    Hybrid AI agent that:
      - Uses LightGBM SOS model for physiology + kinematics.
      - Uses geo-hazard API for environment risk.
      - Uses Grok xAI for reasoning, triage, and cluster intelligence.
      - Issues autonomous SOS & local bystander alerts.
      - Falls back to offline thresholds if network / Grok is unavailable.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        grok_api_url: str = GROK_API_URL,
        grok_api_key: str = GROK_API_KEY,
        hazard_api_url: str = HAZARD_API_URL,
        first_responder_webhook: str = FIRST_RESPONDER_WEBHOOK_URL,
        nearby_broadcast_url: str = NEARBY_BROADCAST_URL,
    ):
        saved = joblib.load(model_path)

        # --- Handle older model files gracefully ---
        if isinstance(saved, dict):
            # Old version: {"model": lgbm, "features": feature_cols}
            # New version: {"model": lgbm, "logistic": lr, "features": ..., "meta": {...}}
            self.model = saved.get("model", None)
            if self.model is None:
                # If someone saved just the model as "saved"
                self.model = saved

            self.logistic = saved.get("logistic", None)
            self.features = saved.get("features", [])

            # Provide defaults if 'meta' missing
            default_meta = {
                "alpha_default": 0.5,
                "sos_threshold": 0.5,
                "prealert_threshold": 0.3,
            }
            self.meta = saved.get("meta", default_meta)
        else:
            # Very old: joblib.dump(model, "sos_gbt_model.joblib")
            self.model = saved
            self.logistic = None
            # You MUST set this list to match training features if you’re in this case
            raise RuntimeError(
                "Model file has no feature metadata. Retrain with the new training script "
                "or save {'model': model, 'features': feature_cols}."
            )

        if not self.features:
            raise RuntimeError("No feature list found in model file. Cannot build feature vector.")

        # --- Config / URLs ---
        self.grok_api_url = grok_api_url
        self.grok_api_key = grok_api_key
        self.hazard_api_url = hazard_api_url
        self.first_responder_webhook = first_responder_webhook
        self.nearby_broadcast_url = nearby_broadcast_url

        # --- Thresholds + fusion param ---
        self.alpha = self.meta.get("alpha_default", 0.5)
        self.sos_threshold = self.meta.get("sos_threshold", 0.5)
        self.prealert_threshold = self.meta.get("prealert_threshold", 0.3)


    # -----------------------------
    # PUBLIC ENTRYPOINT
    # -----------------------------
    def analyze_window(
        self,
        ecg_signal: np.ndarray,
        acc_xyz: np.ndarray,
        fs_ecg: float,
        fs_acc: float,
        lat: float,
        lon: float,
        device_id: str,
        extra_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Main call per time window.

        Returns a dict:
        {
          "P_SOS": float,
          "hazard_index": float,
          "risk": float,
          "grok_triage": {...} or None,
          "decision": "NO_ACTION"|"PREALERT"|"SOS",
          "offline_fallback": bool
        }
        """
        if extra_context is None:
            extra_context = {}

        ts = extra_context.get("timestamp", time.time())

        # 1) Build feature vector (must match training)
        feat_dict = self._build_features(ecg_signal, acc_xyz, fs_ecg, fs_acc)
        feature_vec = np.array([feat_dict[name] for name in self.features], dtype=float)

        # 2) Base SOS probability from LightGBM
        p_sos = float(self.model.predict_proba(feature_vec.reshape(1, -1))[:, 1])

        # 3) Hazard index from geo API
        try:
            hazard_resp = self._query_hazard_api(lat, lon, ts)
            hazard_index = float(hazard_resp.get("hazard_index", 0.0))
            hazard_meta = hazard_resp.get("raw", {})
            offline = False
        except Exception as e:
            # Network or API failure -> offline path
            hazard_index = 0.0
            hazard_meta = {"error": str(e)}
            offline = True

        # 4) Fuse into risk
        risk = compute_hazard_weighted_risk(p_sos, hazard_index, alpha=self.alpha)

        base_payload = {
            "device_id": device_id,
            "timestamp": ts,
            "lat": lat,
            "lon": lon,
            "P_SOS": p_sos,
            "hazard_index": hazard_index,
            "risk": risk,
            "features": feat_dict,
            "env_meta": hazard_meta,
            "extra_context": extra_context,
        }

        if offline:
            # Completely offline: no Grok. Pure thresholds + rules.
            decision = self._decide_offline(risk, feat_dict)
            self._dispatch_actions(decision, base_payload, triage=None)
            return {
                "P_SOS": p_sos,
                "hazard_index": hazard_index,
                "risk": risk,
                "grok_triage": None,
                "decision": decision,
                "offline_fallback": True,
            }

        # 5) Online: call Grok for triage + reasoning
        triage = self._call_grok_triage(base_payload)
        decision = self._decide_online(risk, triage)

        # 6) Dispatch actions (webhooks / push / BLE)
        self._dispatch_actions(decision, base_payload, triage=triage)

        return {
            "P_SOS": p_sos,
            "hazard_index": hazard_index,
            "risk": risk,
            "grok_triage": triage,
            "decision": decision,
            "offline_fallback": False,
        }

    # -----------------------------
    # FEATURE BUILDING
    # -----------------------------
    def _build_features(
        self,
        ecg_signal: np.ndarray,
        acc_xyz: np.ndarray,
        fs_ecg: float,
        fs_acc: float,
    ) -> Dict[str, float]:
        f_ecg = extract_ecg(ecg_signal, fs_ecg)
        f_acc = extract_acc(acc_xyz, fs_acc)
        feat = {**f_ecg, **f_acc}

        # Ensure all model features exist; fill missing with 0.0
        for name in self.features:
            if name not in feat:
                feat[name] = 0.0
        return feat

    # -----------------------------
    # HAZARD INDEX
    # -----------------------------
    def _query_hazard_api(self, lat: float, lon: float, ts: float) -> Dict[str, Any]:
        """
        Hit your hazard / disaster API and compress to a single 0..1 hazard_index.
        THIS IS WHERE YOU HOOK IN SPACE / EO / WEATHER INTELLIGENCE.
        """
        r = requests.get(
            self.hazard_api_url,
            params={"lat": lat, "lon": lon, "ts": int(ts)},
            timeout=3.0,
        )
        j = r.json()

        # Example mapping (replace with your real schema):
        #   - flood/flash flood warning -> 0.9
        #   - hurricane / cyclone / extreme wind -> 0.9
        #   - extreme heat advisory -> 0.7
        #   - no active hazard -> 0.1
        hazard_index = 0.1
        if j.get("flood_warning"):
            hazard_index = max(hazard_index, 0.9)
        if j.get("flash_flood_warning"):
            hazard_index = max(hazard_index, 0.95)
        if j.get("extreme_heat"):
            hazard_index = max(hazard_index, 0.7)
        if j.get("wildfire_risk_high"):
            hazard_index = max(hazard_index, 0.8)

        return {"hazard_index": float(hazard_index), "raw": j}

    # -----------------------------
    # GROK TRIAGE CALL
    # -----------------------------
    def _call_grok_triage(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call Grok to reason over:
          - vitals & accelerometer features,
          - hazard context,
          - geolocation,
          - nearby events (if you add them in extra_context).

        EXPECTED JSON FROM GROK (design this as your response_format):

        {
          "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "FALSE_ALERT",
          "confidence": 0.0-1.0,
          "explanation": "...short human explanation...",
          "actions": {
            "notify_first_responders": true/false,
            "notify_nearby_bystanders": true/false,
            "play_local_audio": true/false
          },
          "label_override": 0 | 1 | null
        }
        """
        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are a disaster triage AI running on top of a fall/trauma detection sensor. "
            "Given sensor features, a geo-hazard index, GPS, and device metadata, "
            "decide if this is a true SOS event, and what actions to take. "
            "Return ONLY a compact JSON object with fields: severity, confidence, explanation, "
            "actions{notify_first_responders, notify_nearby_bystanders, play_local_audio}, "
            "and label_override (0, 1, or null)."
        )

        # We pass the structured telemetry payload as JSON in the user message
        user_msg = json.dumps(payload)

        body = {
            "model": "grok-2-latest",   # or whatever HackPrinceton / xAI specify
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }

        resp = requests.post(
            self.grok_api_url,
            headers=headers,
            json=body,
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        # Grok (OpenAI-style) -> choices[0].message.content (JSON string)
        try:
            content = data["choices"][0]["message"]["content"]
            triage = json.loads(content)
        except Exception:
            # If anything weird, fall back to a conservative neutral response
            triage = {
                "severity": "UNKNOWN",
                "confidence": 0.0,
                "explanation": "Parsing error in Grok response.",
                "actions": {
                    "notify_first_responders": False,
                    "notify_nearby_bystanders": False,
                    "play_local_audio": False,
                },
                "label_override": None,
            }

        return triage

    # -----------------------------
    # DECISION LOGIC
    # -----------------------------
    def _decide_online(self, risk: float, triage: Dict[str, Any]) -> str:
        """
        Combine numeric risk + Grok's judgment into final decision.
        """
        severity = str(triage.get("severity", "UNKNOWN")).upper()
        actions = triage.get("actions", {}) or {}
        conf = float(triage.get("confidence", 0.0))

        # Direct actions if Grok explicitly requests them
        if actions.get("notify_first_responders"):
            return "SOS"
        if actions.get("notify_nearby_bystanders"):
            return "PREALERT"

        # Otherwise use our thresholds + severity
        if risk >= self.sos_threshold or severity in ("CRITICAL", "HIGH"):
            return "SOS"
        if risk >= self.prealert_threshold or severity == "MEDIUM":
            return "PREALERT"
        return "NO_ACTION"

    def _decide_offline(self, risk: float, feat: Dict[str, float]) -> str:
        """
        Offline-only fallback: no Grok, no hazard API.
        Use only ML + hard rules.
        """
        # Hard rule: extremely concerning stillness + high impact
        if feat.get("after_impact_still_flag", 0.0) > 0.5 and feat.get("acc_max", 0.0) > 3.0:
            return "SOS"

        if risk >= self.sos_threshold:
            return "SOS"
        if risk >= self.prealert_threshold:
            return "PREALERT"
        return "NO_ACTION"

    # -----------------------------
    # DISPATCH ACTIONS
    # -----------------------------
    def _dispatch_actions(
        self,
        decision: str,
        payload: Dict[str, Any],
        triage: Dict[str, Any] = None,
    ):
        """
        For the hackathon, just call your backend webhooks.
        Your backend can:
          - hit 911 / EMS integration,
          - broadcast to nearby users via FCM/WebPush/Bluetooth,
          - log to database.
        """
        if triage is None:
            triage = {}

        summary = {
            "decision": decision,
            "P_SOS": payload["P_SOS"],
            "hazard_index": payload["hazard_index"],
            "risk": payload["risk"],
            "triage": triage,
            "location": {"lat": payload["lat"], "lon": payload["lon"]},
            "device_id": payload["device_id"],
            "timestamp": payload["timestamp"],
        }

        if decision == "SOS":
            # 1) Notify first responders
            try:
                requests.post(
                    self.first_responder_webhook,
                    json=summary,
                    timeout=3.0,
                )
            except Exception as e:
                print(f"[WARN] First responder webhook failed: {e}")

            # 2) Notify nearby users as well
            try:
                requests.post(
                    self.nearby_broadcast_url,
                    json={**summary, "scope": "SOS"},
                    timeout=3.0,
                )
            except Exception as e:
                print(f"[WARN] Nearby broadcast webhook failed: {e}")

        elif decision == "PREALERT":
            # Only broadcast to nearby app users
            try:
                requests.post(
                    self.nearby_broadcast_url,
                    json={**summary, "scope": "PREALERT"},
                    timeout=3.0,
                )
            except Exception as e:
                print(f"[WARN] Nearby broadcast webhook failed: {e}")

        else:
            # NO_ACTION -> maybe just log on your backend
            try:
                # Optional: change URL to log endpoint
                requests.post(
                    self.nearby_broadcast_url,
                    json={**summary, "scope": "LOG_ONLY"},
                    timeout=2.0,
                )
            except Exception:
                pass


# -----------------------------
# QUICK MANUAL TEST (optional)
# -----------------------------
if __name__ == "__main__":
    agent = GrokDistressAgent()

    # Fake 20s test window
    fs_ecg = 5
    fs_acc = 50
    t_ecg = np.linspace(0, 20, 20 * fs_ecg)
    t_acc = np.linspace(0, 20, 20 * fs_acc)

    # Simple normal HR + mild motion
    ecg = 80 + 3 * np.sin(0.5 * t_ecg) + 2 * np.random.randn(len(t_ecg))
    acc = 0.1 * np.random.randn(len(t_acc), 3)

    result = agent.analyze_window(
        ecg_signal=ecg,
        acc_xyz=acc,
        fs_ecg=fs_ecg,
        fs_acc=fs_acc,
        lat=40.35,
        lon=-74.65,
        device_id="demo-device-1",
        extra_context={"timestamp": time.time()},
    )
    print(json.dumps(result, indent=2))
