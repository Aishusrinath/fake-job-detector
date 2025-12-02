import React, { useState } from "react";

function UrlDetector() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const checkURL = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult("");

    try {
      const response = await fetch("https://fake-job-detect.onrender.com/predict_url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();

      // Display prediction exactly as returned by backend
      setResult(data.prediction);

    } catch (error) {
      console.error("Error fetching prediction:", error);
      setResult("❌ Error connecting to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="detector-box text-center p-4 border rounded shadow-sm">
      <h3 className="mb-2">URL Detector</h3>
      <p>Paste a URL below to check if it’s safe.</p>

      <form onSubmit={checkURL} className="mb-3">
        <div className="input-group">
          <input
            type="text"
            className="form-control"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Enter website URL"
            required
          />
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Checking..." : "Check"}
          </button>
        </div>
      </form>

      {result && <p className="mt-2 fw-bold">{result}</p>}
    </div>
  );
}

export default UrlDetector;







