import React, { useEffect, useState } from "react";

function Educate() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const response = await fetch(
          `https://newsapi.org/v2/everything?q=phishing+Canada&language=en&pageSize=5&apiKey=66a21a149d374c229abc8dfec6dd54a3`
        );

        const data = await response.json();

        if (data.status === "ok") {
          setNews(data.articles);
        } else {
          console.error("Error fetching news:", data.message);
          setNews([]);
        }
      } catch (error) {
        console.error("Error fetching news:", error);
        setNews([]);
      } finally {
        setLoading(false);
      }
    };

    fetchNews();
  }, []);

  const tips = [
    "🔗 Always verify the URL before clicking.",
    "📧 Be cautious of unsolicited emails asking for personal information.",
    "🔐 Use strong, unique passwords.",
    "🛡 Enable two-factor authentication (2FA).",
    "🧠 Stay informed about cybersecurity threats."
  ];

  return (
    <div className="container mt-4">
      <div className="row">
        {/* Cybersecurity Tips */}
        <div className="col-md-6">
          <div className="card shadow p-3 mb-4">
            <h5>Cybersecurity Tips</h5>
            <ul>
              {tips.map((tip, index) => (
                <li key={index}>{tip}</li>
              ))}
            </ul>
          </div>
        </div>

        {/* Latest Cybersecurity News */}
        <div className="col-md-6">
          <div className="card shadow p-3 mb-4">
            <h5>Latest Cybersecurity News</h5>
            {loading ? (
              <p>Loading news...</p>
            ) : news.length > 0 ? (
              <ul className="list-unstyled">
                {news.map((article, index) => (
                  <li key={index} className="mb-3">
                    <a href={article.url} target="_blank" rel="noopener noreferrer">
                      <strong>{article.title}</strong>
                    </a>
                    <p style={{ fontSize: "0.9rem", color: "#555" }}>
                      {article.description}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No recent phishing news available in Canada.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Educate;
