import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App.jsx";
import Monitor from "./pages/Monitor.jsx";
import Alert from "./pages/Alert.jsx";
import CPRGuide from "./pages/CPRGuide.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route path="/" element={<Monitor />} />
          <Route path="/alert" element={<Alert />} />
          <Route path="/cpr" element={<CPRGuide />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
