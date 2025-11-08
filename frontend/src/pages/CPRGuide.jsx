import { Link } from "react-router-dom";

export default function CPRGuide() {
  const steps = [
    {
      title: "Check Responsiveness",
      description: "Tap the person and shout. Check for movement or breathing.",
    },
    {
      title: "Call Emergency Services",
      description: "Call 911 or instruct someone nearby to call immediately.",
    },
    {
      title: "Open Airway",
      description: "Tilt the head back slightly and lift the chin.",
    },
    {
      title: "Check Breathing",
      description: "Look, listen, and feel for breathing for no more than 10 seconds.",
    },
    {
      title: "Begin Chest Compressions",
      description:
        "Push hard and fast in the center of the chest (100–120 compressions/min).",
    },
    {
      title: "Continue CPR",
      description: "Alternate 30 compressions and 2 rescue breaths if trained. Continue until help arrives.",
    },
  ];

  return (
    <div className="min-h-screen bg-background text-text px-6 py-12 flex flex-col items-center gap-10">
      <div className="text-center">
        <h1 className="text-5xl font-bold text-accent mb-2">CPR Guide</h1>
        <p className="opacity-70 max-w-xl mx-auto text-lg">
          Follow these steps immediately if someone collapses and is unresponsive.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl w-full">
        {steps.map((step, index) => (
          <div
            key={index}
            className="bg-card border border-white/10 rounded-xl p-6 shadow-md flex flex-col gap-2"
          >
            <div className="text-accent font-bold text-lg">
              Step {index + 1}
            </div>
            <h2 className="text-xl font-semibold">{step.title}</h2>
            <p className="opacity-70 text-sm">{step.description}</p>
          </div>
        ))}
      </div>

      <Link
        to="/"
        className="px-6 py-3 rounded-full bg-accent text-background font-semibold hover:bg-secondary transition mt-8"
      >
        Back to Monitor
      </Link>
    </div>
  );
}
