import React, { useState } from "react";

function UrlDetector() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState("");

  const checkURL = async (e) => {
  e.preventDefault();

  try {
    const response = await fetch("https://your-backend.onrender.com/predict_url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url }),
    });

    const data = await response.json();

    if (data.prediction === "PHISHING") {
      setResult("⚠️ Phishing URL Detected!");
    } else {
      setResult("✅ This URL seems legitimate.");
    }
  } catch (error) {
    setResult("❌ Error connecting to server.");
  }
};


  return (
    <div className="detector-box text-center">
      <h3>URL Detector</h3>
      <p>Paste a URL below to check if it’s safe.</p>
      <form onSubmit={checkURL}>
        <div className="input-group mb-3">
          <input 
            type="text" 
            className="form-control" 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter website URL" 
            required 
          />
          <button className="btn btn-primary" type="submit">Check</button>
        </div>
      </form>
      {result && <p>{result}</p>}
    </div>
  );
}

export default UrlDetector;

