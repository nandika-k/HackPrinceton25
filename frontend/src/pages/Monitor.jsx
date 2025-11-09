import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { io } from "socket.io-client";
import About from "../components/About";
import LiveECGChart from "../components/LiveECGChart"; // ⬅️ import the chart component
import LiveAccelChart from "../components/LiveAccelChart";


function VitalsCard({ label, value }) {
  return (
    <div className="bg-card rounded-xl px-6 py-4 shadow-lg border border-white/10 w-44 min-w-[10rem]">
      <p className="text-sm opacity-60">{label}</p>
      <p className="text-2xl font-bold text-accent">{value}</p>
    </div>
  );
}

export default function Monitor() {
  const [ecg, setEcg] = useState(0);     // ⬅️ initialized as number for chart
  const [ax, setAx] = useState("...");
  const [ay, setAy] = useState("...");
  const [az, setAz] = useState("...");
  const [risk, setRisk] = useState(0);
  const [status, setStatus] = useState("...");
  const [windowSeconds, setWindowSeconds] = useState(null);
  const [syntheticLabel, setSyntheticLabel] = useState(null);

  useEffect(() => {
    const socket = io("http://localhost:5100");

    socket.on("connect", () => {
      console.log("✅ Connected to SocketIO server");
    });

      socket.on("sensor_data", (data) => {
        if (typeof data.ecg === "number") {
          setEcg(parseFloat(data.ecg.toFixed(2)));  // ⬅️ pass number to chart
        }
        if (typeof data.ax === "number") {
          setAx(data.ax.toFixed(2));
        }
        if (typeof data.ay === "number") {
          setAy(data.ay.toFixed(2));
        }
        if (typeof data.az === "number") {
          setAz(data.az.toFixed(2));
        }

        if (typeof data.riskScore === "number") {
          setRisk(data.riskScore);
        }
        if (typeof data.featureWindowSeconds === "number") {
          setWindowSeconds(data.featureWindowSeconds);
        }
        if (typeof data.syntheticLabel === "boolean") {
          setSyntheticLabel(data.syntheticLabel);
        } else {
          setSyntheticLabel(null);
        }
        if (typeof data.status === "string") {
          setStatus(data.status);
        }
      });

    return () => socket.disconnect();
  }, []);

  return (
    <div className="min-h-screen w-full bg-background text-text flex flex-col items-center justify-center px-4 py-12 gap-10">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-5xl font-bold text-accent mb-2">Monitoring</h1>
        <p className="text-lg opacity-70">Live vitals will appear here.</p>
      </div>

      {/* Cards */}
      <div className="flex flex-wrap gap-6 justify-center text-center">
        <VitalsCard label="ECG" value={ecg.toFixed(2)} />
        <VitalsCard label="Accel X" value={ax} />
        <VitalsCard label="Accel Y" value={ay} />
        <VitalsCard label="Accel Z" value={az} />
        <VitalsCard label="Risk %" value={(risk * 100).toFixed(0)} />
        <VitalsCard label="Status" value={status} />
      </div>

      {/* Real-Time ECG Waveform */}
      <div className="w-full max-w-3xl bg-card p-4 rounded-xl shadow-xl border border-white/10">
        <h2 className="text-xl font-semibold text-accent mb-4 text-center">Real-Time ECG Signal</h2>
        <LiveECGChart ecg={ecg} />
      </div>

      <div className="w-full max-w-3xl bg-card p-4 rounded-xl border border-white/10">
        <h2 className="text-xl font-semibold text-accent mb-4 text-center">Real-Time Accelerometer</h2>
        <LiveAccelChart ax={parseFloat(ax)} ay={parseFloat(ay)} az={parseFloat(az)} />
      </div>

      {/* Model Context */}
      {windowSeconds !== null && (
        <p className="text-sm opacity-60">
          Model window: {windowSeconds.toFixed(1)}s
          {syntheticLabel !== null && (
            <span className="ml-2 opacity-80">
              ({syntheticLabel ? "Simulated Fall" : "Simulated Normal"})
            </span>
          )}
        </p>
      )}

      {/* Emergency Sim Button */}
      <Link
        to="/alert"
        className="mt-6 px-6 py-3 bg-accent text-background rounded-full font-semibold hover:bg-secondary transition"
      >
        Simulate Emergency ➜
      </Link>

      {/* About */}
      <About />
    </div>
  );
}
