import React, { useState } from "react";

function EmailDetector() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    setSelectedImage(file);
    setResult(null);

    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const res = await fetch("https://fake-job-detect.onrender.com/predict-image", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "Failed to connect to server." });
    }

    setLoading(false);
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>Email Scam Image Detector</h2>

      <input type="file" accept="image/*" onChange={handleFileUpload} />

      {selectedImage && (
        <img
          src={URL.createObjectURL(selectedImage)}
          alt="preview"
          width="300"
          style={{ marginTop: "20px", border: "1px solid #ddd", padding: "10px" }}
        />
      )}

      {loading && <p>Analyzing image...</p>}

      {result && !loading && (
        <div style={{ marginTop: "20px" }}>
          {result.error ? (
            <p style={{ color: "red" }}>{result.error}</p>
          ) : (
            <>
              <h3>Prediction: {result.label}</h3>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
              <pre>{JSON.stringify(result.probabilities, null, 2)}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default EmailDetector;

