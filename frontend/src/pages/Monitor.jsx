import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { io } from "socket.io-client";
import About from "../components/About";
import LiveECGChart from "../components/LiveECGChart";
import LiveAccelChart from "../components/LiveAccelChart";

function VitalsCard({ label, value, accent }) {
  return (
    <div className="bg-card rounded-xl px-6 py-4 shadow-lg border border-white/10 w-44 min-w-[10rem]">
      <p className="text-sm opacity-60">{label}</p>
      <p className={`text-2xl font-bold ${accent ?? "text-accent"}`}>{value}</p>
    </div>
  );
}

const socketURL = import.meta.env.VITE_SOCKET_URL ?? "http://localhost:5100";

export default function Monitor() {
  const [ecg, setEcg] = useState(0);
  const [ax, setAx] = useState("...");
  const [ay, setAy] = useState("...");
  const [az, setAz] = useState("...");

  const [riskProbability, setRiskProbability] = useState(null);
  const [riskLevel, setRiskLevel] = useState("low");
  const [lastModelUpdate, setLastModelUpdate] = useState(null);
  const [featureSnapshot, setFeatureSnapshot] = useState({});

  useEffect(() => {
    const socket = io(socketURL, { transports: ["websocket"] });

    socket.on("connect", () => {
      console.log("✅ Connected to SocketIO server");
    });

    socket.on("sensor_data", (data) => {
      setEcg(parseFloat(data.ecg.toFixed(2)));
      setAx(data.ax.toFixed(2));
      setAy(data.ay.toFixed(2));
      setAz(data.az.toFixed(2));
    });

    socket.on("prediction", (data) => {
      if (typeof data.probability === "number") {
        setRiskProbability(data.probability);
      }
      if (typeof data.risk_level === "string") {
        setRiskLevel(data.risk_level);
      }
      if (data.timestamp) {
        const date = new Date(data.timestamp * 1000);
        setLastModelUpdate(date);
      } else {
        setLastModelUpdate(new Date());
      }
      if (data.features && typeof data.features === "object") {
        setFeatureSnapshot(data.features);
      }
    });

    socket.on("disconnect", () => {
      console.log("⚠️ SocketIO disconnected");
    });

    return () => socket.disconnect();
  }, []);

  const riskDisplay = useMemo(() => {
    if (riskProbability == null) {
      return "Analyzing…";
    }

    const pct = `${(riskProbability * 100).toFixed(0)}%`;
    return `${riskLevel.toUpperCase()} (${pct})`;
  }, [riskProbability, riskLevel]);

  const riskAccent = useMemo(() => {
    if (riskProbability == null) {
      return "text-white";
    }
    if (riskLevel === "high") {
      return "text-red-400";
    }
    if (riskLevel === "medium") {
      return "text-yellow-300";
    }
    return "text-emerald-300";
  }, [riskProbability, riskLevel]);

  const featureEntries = useMemo(() => {
    return Object.entries(featureSnapshot ?? {}).sort(([a], [b]) =>
      a.localeCompare(b)
    );
  }, [featureSnapshot]);

  return (
    <div className="min-h-screen w-full bg-background text-text flex flex-col items-center justify-center px-4 py-12 gap-10">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-accent mb-2">Monitoring</h1>
        <p className="text-lg opacity-70">
          Live vitals and AI alerts will appear here.
        </p>
      </div>

      <div className="flex flex-wrap gap-6 justify-center text-center">
        <VitalsCard label="ECG" value={ecg.toFixed(2)} />
        <VitalsCard label="Accel X" value={ax} />
        <VitalsCard label="Accel Y" value={ay} />
        <VitalsCard label="Accel Z" value={az} />
        <VitalsCard label="SOS Risk" value={riskDisplay} accent={riskAccent} />
      </div>

      {lastModelUpdate && (
        <div className="bg-card border border-white/10 rounded-lg px-4 py-2 text-sm text-white/70">
          Last model update:{" "}
          <span className="text-white">
            {lastModelUpdate.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
              second: "2-digit",
            })}
          </span>
        </div>
      )}

      <div className="w-full max-w-3xl bg-card p-4 rounded-xl shadow-xl border border-white/10">
        <h2 className="text-xl font-semibold text-accent mb-4 text-center">
          Real-Time ECG Signal
        </h2>
        <LiveECGChart ecg={ecg} />
      </div>

      <div className="w-full max-w-3xl bg-card p-4 rounded-xl mt-6 border border-white/10">
        <h2 className="text-xl font-semibold text-accent mb-4 text-center">
          Real-Time Accelerometer
        </h2>
        <LiveAccelChart
          ax={parseFloat(ax)}
          ay={parseFloat(ay)}
          az={parseFloat(az)}
        />
      </div>

      {featureEntries.length > 0 && (
        <div className="w-full max-w-3xl bg-card p-4 rounded-xl border border-white/10">
          <h2 className="text-xl font-semibold text-accent mb-4 text-center">
            Model Feature Snapshot
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-white/80">
            {featureEntries.map(([key, value]) => (
              <div
                key={key}
                className="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2"
              >
                <span className="capitalize">{key.replace(/_/g, " ")}</span>
                <span className="font-mono text-white">
                  {Number.isFinite(value)
                    ? Number(value).toFixed(2)
                    : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <Link
        to="/alert"
        className="mt-6 px-6 py-3 bg-accent text-background rounded-full font-semibold hover:bg-secondary transition"
      >
        Simulate Emergency ➜
      </Link>

      <About />
    </div>
  );
}
