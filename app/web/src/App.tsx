import { NavLink, Route, Routes } from "react-router-dom";
import { Brain } from "lucide-react";
import Analyze from "./pages/Analyze";
import About from "./pages/About";
import Data from "./pages/Data";
import Results from "./pages/Results";
import Explain from "./pages/Explain";
import Reliability from "./pages/Reliability";

const NAV = [
  ["/", "Analyse scan"], ["/results", "Performance"], ["/explain", "Explainability"], ["/reliability", "Reliability"], ["/data", "Data"], ["/about", "About"],
];

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-ink-300/60 bg-white/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-8 px-6 lg:px-10">
          <div className="flex items-center gap-2.5">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-brand-600 text-white"><Brain size={20} /></div>
            <div className="leading-tight">
              <div className="text-sm font-semibold text-ink-900">Cross-Cohort AD / MCI</div>
              <div className="text-[11px] muted">Robust &amp; explainable sMRI deep learning</div>
            </div>
          </div>
          <nav className="flex gap-1 overflow-x-auto">
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"}
                className={({ isActive }) => `whitespace-nowrap rounded-lg px-3 py-2 text-sm font-medium transition ${isActive ? "bg-brand-50 text-brand-700" : "text-ink-500 hover:text-ink-900"}`}>
                {label}
              </NavLink>
            ))}
          </nav>
          
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Analyze />} />
          <Route path="/about" element={<About />} />
          <Route path="/data" element={<Data />} />
          <Route path="/results" element={<Results />} />
          <Route path="/explain" element={<Explain />} />
          <Route path="/reliability" element={<Reliability />} />
        </Routes>
      </main>
      <footer className="mx-auto max-w-7xl px-6 pb-10 pt-4 text-xs muted lg:px-10">
        Research software, not a medical device. All numbers are computed from local result files (ADNI, OASIS-3); no subject-level data is displayed.
      </footer>
    </div>
  );
}
