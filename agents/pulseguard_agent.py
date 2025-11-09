import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import joblib
import numpy as np

from LightGradientBoostedTree.LGBMAlgo import extract_accel, extract_ecg
from LightGradientBoostedTree.geo_features import (
    GeoFeatureEngineer,
    GeoHazardFeatureStore,
    GrokGeoContextClient,
)


class ResponderNetworkClient:
    """
    Handles communication with remote first responder systems. For hackathon use this can be
    backed by Twilio, a webhook, or a simple mock HTTP endpoint. The send_alert method
    purposefully returns a boolean so the agent can degrade gracefully when connectivity is
    lost.
    """

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        self.webhook_url = webhook_url or os.getenv("RESPONDER_WEBHOOK_URL")

    def send_alert(self, payload: Dict[str, Any]) -> bool:
        if not self.webhook_url:
            return False

        try:
            import requests

            response = requests.post(self.webhook_url, json=payload, timeout=8)
            response.raise_for_status()
            return True
        except Exception as exc:  # pragma: no cover - network failures expected during offline demos
            print(f"WARNING: Failed to reach responder webhook: {exc}")
            return False


class BluetoothMeshBroadcaster:
    """
    Broadcasts SOS packets over Bluetooth Low Energy to nearby peers. The actual BLE stack can
    be implemented with libraries such as bleak, but for portability this class records packets
    locally so the demo can simulate a mesh push even when Bluetooth APIs are unavailable.
    """

    def __init__(self, log_path: str = "data/offline/bt_mesh_log.jsonl") -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def broadcast_alert(self, payload: Dict[str, Any]) -> None:
        entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "type": "bluetooth_mesh_alert",
            "payload": payload,
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


class LocalEventLogger:
    """
    Persists every decision locally so the agent can reconcile once connectivity returns.
    """

    def __init__(self, path: str = "data/offline/pulseguard_events.jsonl") -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", datetime.utcnow().isoformat() + "Z")
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class PulseGuardAgent:
    """
    Orchestrates the hybrid AI workflow:

    1. Extract physiological + motion features on device.
    2. Blend LightGBM distress probability with Grok-powered geospatial intelligence.
    3. Autonomously coordinate human + machine responders with online/offline failover.
    """

    RISK_EMERGENCY = 0.7
    RISK_ATTENTION = 0.4

    def __init__(
        self,
        model_path: str = "sos_gbt_model.joblib",
        hazard_store: Optional[GeoHazardFeatureStore] = None,
        grok_client: Optional[GrokGeoContextClient] = None,
        responder_client: Optional[ResponderNetworkClient] = None,
        mesh_broadcaster: Optional[BluetoothMeshBroadcaster] = None,
        event_logger: Optional[LocalEventLogger] = None,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bundle not found at {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["features"]
        self.geo_defaults = bundle.get("geo_feature_defaults", {})

        self.geo_engineer = GeoFeatureEngineer(
            hazard_store=hazard_store,
            grok_client=grok_client or GrokGeoContextClient(),
        )
        self.responder_client = responder_client or ResponderNetworkClient()
        self.mesh_broadcaster = mesh_broadcaster or BluetoothMeshBroadcaster()
        self.event_logger = event_logger or LocalEventLogger()

    def assess_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a multimodal sensor packet into a calibrated risk score and recommended actions.
        """
        features = self._build_feature_dictionary(packet)
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names], dtype=float)
        distress_prob = float(self.model.predict_proba(feature_vector.reshape(1, -1))[0, 1])

        geo_context = {
            key: features.get(key, self.geo_defaults.get(key, 0.0))
            for key in self.geo_defaults.keys()
        }

        hazard_boost = float(
            np.clip(
                0.18 * geo_context.get("grok_hazard_score", 0.0)
                + 0.12 * (1.0 - geo_context.get("geo_disaster_distance_km", 120.0) / 220.0)
                + 0.08 * geo_context.get("geo_population_density_norm", 0.4),
                0.0,
                0.35,
            )
        )
        risk_score = float(np.clip(distress_prob + hazard_boost, 0.0, 1.0))

        severity = self._derive_severity_label(risk_score, packet)
        guidance = self._craft_guidance(severity, packet)

        assessment = {
            "distress_probability": distress_prob,
            "hazard_boost": hazard_boost,
            "risk_score": risk_score,
            "severity": severity,
            "guidance": guidance,
            "features": features,
            "geo_context": geo_context,
        }
        return assessment

    def orchestrate(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full decision loop: evaluate risk, dispatch communications, and persist the outcome.
        """
        assessment = self.assess_packet(packet)
        dispatch_report = self._dispatch(assessment, packet)
        record = {
            "assessment": assessment,
            "dispatch": dispatch_report,
        }
        self.event_logger.log(record)
        return record

    def _build_feature_dictionary(self, packet: Dict[str, Any]) -> Dict[str, float]:
        ecg = np.asarray(packet.get("ecg", []), dtype=float)
        accel = np.asarray(packet.get("accel", []), dtype=float)

        fs_ecg = float(packet.get("fs_ecg", 125.0))
        fs_accel = float(packet.get("fs_accel", 125.0))

        ecg_features = extract_ecg(ecg, fs_ecg)
        accel_features = extract_accel(accel, fs_accel)

        lat = float(packet.get("location", {}).get("lat", self.geo_defaults.get("geo_lat", 0.0)))
        lon = float(packet.get("location", {}).get("lng", self.geo_defaults.get("geo_lon", 0.0)))
        hazard_summary = packet.get("hazard_summary")
        geo_features = self.geo_engineer.compute_features(lat, lon, packet.get("ts"), hazard_summary)

        symptoms = packet.get("symptoms", {})
        symptom_score = float(
            0.2 * symptoms.get("chest_pain", False)
            + 0.15 * symptoms.get("short_breath", False)
            + 0.12 * symptoms.get("sweat", False)
            + 0.1 * symptoms.get("nausea", False)
            + 0.1 * symptoms.get("left_arm_pain", False)
        )

        immobile_seconds = float(packet.get("immobile_s", accel_features.get("stillness_duration", 0.0)))
        post_impact_flag = float(accel_features.get("post_impact_still_flag", 0.0))

        feature_map: Dict[str, float] = {}
        feature_map.update(ecg_features)
        feature_map.update(accel_features)
        feature_map.update(geo_features)
        feature_map["symptom_score"] = symptom_score
        feature_map["immobile_seconds"] = immobile_seconds
        feature_map["post_impact_flag"] = post_impact_flag
        feature_map["geo_lat"] = geo_features.get("geo_lat", lat)
        feature_map["geo_lon"] = geo_features.get("geo_lon", lon)

        return feature_map

    def _dispatch(self, assessment: Dict[str, Any], packet: Dict[str, Any]) -> Dict[str, Any]:
        risk = assessment["risk_score"]
        mode = "monitor"
        responders_alerted = False
        local_broadcast = False

        alert_payload = {
            "timestamp": packet.get("ts", datetime.utcnow().timestamp()),
            "severity": assessment["severity"],
            "risk_score": risk,
            "location": packet.get("location", {}),
            "guidance": assessment["guidance"],
            "geo_context": assessment["geo_context"],
        }

        if risk >= self.RISK_EMERGENCY:
            mode = "emergency"
            responders_alerted = self.responder_client.send_alert(alert_payload)
            if not responders_alerted:
                mode = "offline_emergency"
            self.mesh_broadcaster.broadcast_alert(alert_payload)
            local_broadcast = True
        elif risk >= self.RISK_ATTENTION:
            mode = "attention"
            self.mesh_broadcaster.broadcast_alert(alert_payload)
            local_broadcast = True

        return {
            "mode": mode,
            "responders_alerted": responders_alerted,
            "bluetooth_alert": local_broadcast,
        }

    @staticmethod
    def _derive_severity_label(risk_score: float, packet: Dict[str, Any]) -> str:
        if risk_score >= PulseGuardAgent.RISK_EMERGENCY:
            return "EMERGENCY"
        if risk_score >= PulseGuardAgent.RISK_ATTENTION:
            return "CHECK_NOW"

        accel = np.asarray(packet.get("accel", []), dtype=float)
        if accel.size > 0 and np.max(np.linalg.norm(accel.reshape(-1, 3), axis=1)) > 3.5:
            return "CHECK_NOW"
        return "NORMAL"

    @staticmethod
    def _craft_guidance(severity: str, packet: Dict[str, Any]) -> str:
        if severity == "EMERGENCY":
            return "Possible cardiac emergency detected. Call emergency services, start hands-only CPR, and stay with the patient."
        if severity == "CHECK_NOW":
            return "Assess responsiveness, gather symptoms, and prepare to contact emergency services if condition worsens."
        symptom_flags = packet.get("symptoms", {})
        if symptom_flags.get("chest_pain"):
            return "Chest discomfort monitored. Sit or lie down, keep calm, and continue monitoring."
        return "Vitals stable. Continue monitoring and ensure good sensor contact."
