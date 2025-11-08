import { Link } from "react-router-dom";
import About from "../components/About"; // ⬅️ Import the new component

export default function Monitor() {
  return (
    <div className="min-h-screen w-full bg-background text-text flex flex-col items-center justify-center px-4 py-12 gap-10">

      {/* Title */}
      <div className="text-center">
        <h1 className="text-5xl font-bold text-accent mb-2">Monitoring</h1>
        <p className="text-lg opacity-70">Live vitals will appear here.</p>
      </div>

      {/* Vitals */}
      <div className="flex flex-wrap gap-6 justify-center text-center">
        <VitalsCard label="Heart Rate" value="82 bpm" />
        <VitalsCard label="O₂ Level" value="97%" />
        <VitalsCard label="Temperature" value="98.6°F" />
      </div>

      {/* Emergency Button */}
      <Link
        to="/alert"
        className="mt-6 px-6 py-3 bg-accent text-background rounded-full font-semibold hover:bg-secondary transition"
      >
        Simulate Emergency ➜
      </Link>

      {/* About Section */}
      <About />
    </div>
  );
}

function VitalsCard({ label, value }) {
  return (
    <div className="bg-card rounded-xl px-6 py-4 shadow-lg border border-white/10 w-44 min-w-[10rem]">
      <p className="text-sm opacity-60">{label}</p>
      <p className="text-2xl font-bold text-accent">{value}</p>
    </div>
  );
}
