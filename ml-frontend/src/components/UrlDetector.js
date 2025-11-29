import React, { useState } from "react";

function UrlDetector() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState("");

  const checkURL = (e) => {
    e.preventDefault();
    if (url.includes("phish") || url.includes("scam")) {
      setResult("⚠️ This link looks suspicious!");
    } else {
      setResult("✅ This link seems safe (but always be cautious).");
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
