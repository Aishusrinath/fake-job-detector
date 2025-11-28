import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import UrlDetector from "./components/UrlDetector";
import EmailDetector from "./components/EmailDetector";
import FakeJobs from "./components/FakeJobs";
import Educate from "./components/Educate";
function App() {
  return (

    <div>
      <nav style={{ padding: "10px", background: "#222", color: "#fff" }}>
        <Link to="/" style={{ margin: "10px", color: "#fff" }}>Educate</Link>
        <Link to="/url" style={{ margin: "10px", color: "#fff" }}>URL Detector</Link>
        <Link to="/email" style={{ margin: "10px", color: "#fff" }}>Email Detector</Link>
        <Link to="/jobs" style={{ margin: "10px", color: "#fff" }}>Fake Jobs</Link>
      </nav>

      <Routes>
        <Route path="/" element={<Educate />} />
        <Route path="/url" element={<UrlDetector />} />
        <Route path="/email" element={<EmailDetector />} />
        <Route path="/jobs" element={<FakeJobs />} />
      </Routes>
    </div>
  );
}

export default App;

