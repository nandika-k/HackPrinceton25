export default function Footer() {
  return (
    <footer className="w-full h-auto py-4 flex justify-center items-center backdrop-blur-lg bg-background/20 border-t border-white/10">
      <div className="text-center text-sm opacity-80">
        <p className="font-semibold">
          &copy; {new Date().getFullYear()} PulseGuard
        </p>
        <p className="mt-1">
          Built with ❤️ at{" "}
          <span className="text-accent font-semibold">HackPrinceton '25</span>
        </p>
        <div className="inline-block bg-accent text-background text-xs font-semibold px-4 py-1 rounded-full mt-2">
          v1.0
        </div>
      </div>
    </footer>
  );
}
