const API_BASE = "http://127.0.0.1:5000"; // Flask backend

export async function getVitals() {
  const res = await fetch(`${API_BASE}/api/vitals`);
  const data = await res.json();
  return data;
}

export async function getPrediction(features) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features }),
  });
  const data = await res.json();
  return data;
}
