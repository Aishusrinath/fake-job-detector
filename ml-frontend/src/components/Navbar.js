import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Educate from "./components/Educate";
import UrlDetector from "./components/UrlDetector";
import EmailDetector from "./components/EmailDetector";
import FakeJobs from "./components/FakeJobs";

function App() {
  return (
    <Router>
      <Navbar />
      <div className="container mt-4">
        <Routes>
          <Route path="/" element={<Educate />} />
          <Route path="/url-detector" element={<UrlDetector />} />
          <Route path="/email-detector" element={<EmailDetector />} />
          <Route path="/fake-jobs" element={<FakeJobs />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
