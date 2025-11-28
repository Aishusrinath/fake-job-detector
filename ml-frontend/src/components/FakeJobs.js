import React, { useState } from "react";

function FakeJobs() {
  const [jobDesc, setJobDesc] = useState("");
  const [result, setResult] = useState("");

  const checkJob = (e) => {
    e.preventDefault();
    if (jobDesc.toLowerCase().includes("wire money") || jobDesc.toLowerCase().includes("no experience high pay")) {
      setResult("⚠️ This job post looks suspicious!");
    } else {
      setResult("✅ This job post seems safe.");
    }
  };

  return (
    <div className="detector-box text-center">
      <h3>Fake Jobs Detection</h3>
      <p>Paste a job description to check if it’s suspicious.</p>
      <form onSubmit={checkJob}>
        <div className="input-group mb-3">
          <textarea 
            className="form-control" 
            rows="4"
            value={jobDesc}
            onChange={(e) => setJobDesc(e.target.value)}
            placeholder="Paste job description here..." 
            required 
          />
        </div>
        <button className="btn btn-primary" type="submit">Check</button>
      </form>
      {result && <p>{result}</p>}
    </div>
  );
}

export default FakeJobs;
