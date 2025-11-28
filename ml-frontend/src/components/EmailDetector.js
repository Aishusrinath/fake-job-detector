import React, { useState } from "react";

function EmailDetector() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");

  const checkEmail = (e) => {
    e.preventDefault();

    if (!file) {
      setResult("⚠️ Please upload an email screenshot.");
      return;
    }

    // Placeholder logic — replace with real AI image classification
    if (file.name.toLowerCase().includes("phish") || file.name.toLowerCase().includes("spam")) {
      setResult("⚠️ This email looks suspicious!");
    } else {
      setResult("✅ This email seems safe.");
    }
  };

  return (
    <div className="detector-box text-center" style={{ padding: "20px", maxWidth: "500px", margin: "auto" }}>
      <h3>Email Detector</h3>
      <p>Upload an email screenshot to check if it’s phishing.</p>

      <form onSubmit={checkEmail}>
        <div className="input-group mb-3">
          <input
            type="file"
            className="form-control"
            accept="image/*"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
          <button className="btn btn-primary" type="submit">Check</button>
        </div>
      </form>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <p>{result}</p>
          {file && (
            <img
              src={URL.createObjectURL(file)}
              alt="Uploaded Email"
              style={{ maxWidth: "100%", border: "1px solid #ccc", padding: "10px" }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default EmailDetector;


