import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const { pathname } = useLocation();

  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur-lg bg-background/40 border-b border-white/10 shadow-sm flex justify-center">
      <nav className="w-full max-w-6xl px-10 h-14 flex justify-between items-center">

        {/* PulseGuard logo with ECG icon */}
        <Link to="/" className="flex items-center gap-1">
          <svg
  xmlns="http://www.w3.org/2000/svg"
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  strokeWidth="2"
  strokeLinecap="round"
  strokeLinejoin="round"
  className="w-5 h-5 text-accent animate-ecg-pulse"
>
  <polyline points="3 12 6 12 9 6 13 18 16 12 21 12" />
</svg>

          <h1 className="text-2xl font-semibold tracking-tight">
            <span className="text-accent font-bold">Pulse</span>Guard
          </h1>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-10 text-sm font-medium">
          {[
            { name: "Monitor", path: "/" },
            { name: "Alert", path: "/alert" },
          ].map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`transition hover:text-accent ${
                pathname === item.path ? "text-accent" : "text-text"
              }`}
            >
              {item.name}
            </Link>
          ))}

          {/* CTA Button */}
          <Link
            to="/cpr"
            className="inline-block px-5 py-2 rounded-full bg-accent text-background font-medium hover:bg-secondary transition"
          >
            CPR Guide
          </Link>
        </div>

      </nav>
    </header>
  );
}
