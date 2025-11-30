// import React, { useState } from "react";

// function FakeJobs() {
//   const [jobDesc, setJobDesc] = useState("");
//   const [result, setResult] = useState("");

//   const checkJob = (e) => {
//     e.preventDefault();
//     if (jobDesc.toLowerCase().includes("wire money") || jobDesc.toLowerCase().includes("no experience high pay")) {
//       setResult("⚠️ This job post looks suspicious!");
//     } else {
//       setResult("✅ This job post seems safe.");
//     }
//   };

//   return (
//     <div className="detector-box text-center">
//       <h3>Fake Jobs Detection</h3>
//       <p>Paste a job description to check if it’s suspicious.</p>
//       <form onSubmit={checkJob}>
//         <div className="input-group mb-3">
//           <textarea 
//             className="form-control" 
//             rows="4"
//             value={jobDesc}
//             onChange={(e) => setJobDesc(e.target.value)}
//             placeholder="Paste job description here..." 
//             required 
//           />
//         </div>
//         <button className="btn btn-primary" type="submit">Check</button>
//       </form>
//       {result && <p>{result}</p>}
//     </div>
//   );
// }

// export default FakeJobs;

import React, { useState } from "react";

function FakeJobDetector() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCheck = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("https://fake-job-detect.onrender.com/predict_job", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      // Convert FAKE/REAL to user-friendly SAFE/FAKE
      const displayResult = data.prediction === "FAKE" ? "❌ FAKE JOB" : "✅ SAFE JOB";
      setResult(displayResult);

    } catch (error) {
      console.error("Error:", error);
      setResult("⚠️ Error checking job");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto" }}>
      <h2>Fake Job Detector</h2>

      <textarea
        rows="6"
        placeholder="Paste job description here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ width: "100%", padding: "10px", fontSize: "16px" }}
      />

      <button
        onClick={handleCheck}
        style={{ marginTop: "10px", padding: "10px 20px", fontSize: "16px" }}
        disabled={loading}
      >
        {loading ? "Checking..." : "Check Job"}
      </button>

      {result && (
        <div style={{ marginTop: "20px", fontSize: "18px", fontWeight: "bold" }}>
          {result}
        </div>
      )}
    </div>
  );
}

export default FakeJobDetector;




