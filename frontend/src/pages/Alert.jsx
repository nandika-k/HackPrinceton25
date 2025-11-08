import { Link } from "react-router-dom";
import { useEffect, useState } from "react";

export default function Alert() {
  const [timestamp, setTimestamp] = useState("");

  useEffect(() => {
    const now = new Date();
    const formatted = now.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setTimestamp(formatted);
  }, []);

  return (
    <div className="h-screen w-full bg-accent text-background flex flex-col items-center justify-center px-4 text-center gap-8">
      {/* Emergency Heading */}
      <h1 className="text-6xl font-extrabold animate-pulse drop-shadow-lg">
        🚨 EMERGENCY
      </h1>

      {/* Description */}
      <p className="text-xl font-medium">
        Possible cardiac distress detected.
      </p>

      {/* Timestamp */}
      <p className="text-sm opacity-80">
        Alert triggered at <span className="font-bold">{timestamp}</span>
      </p>

      {/* CTA */}
      <Link
        to="/cpr"
        className="px-6 py-3 rounded-full bg-background text-accent font-semibold hover:bg-secondary transition"
      >
        Start CPR Guide ➜
      </Link>
    </div>
  );
}
