export default function About() {
  return (
    <section className="w-full max-w-4xl text-center px-6 py-10 pb-0">
      <h2 className="text-3xl font-bold text-accent mb-2">About PulseGuard</h2>

      {/* Horizontal accent line */}
      <div className="w-150 h-0.5 bg-accent mx-auto mb-6 rounded-full" />

      <p className="text-md opacity-70 leading-relaxed">
        <strong>PulseGuard</strong> is a real-time health emergency simulator built during HackPrinceton ’25.
        It mimics vital monitoring in critical situations and provides instant access to CPR instructions during
        cardiac distress. Designed for awareness, training, and responsiveness, this tool showcases how technology
        can enhance emergency readiness and save lives.
      </p>
    </section>
  );
}
