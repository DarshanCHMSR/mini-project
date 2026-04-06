import { useEffect, useState } from "react";
import PredictionForm from "../components/PredictionForm";
import PredictionResult from "../components/PredictionResult";
import StatusPill from "../components/StatusPill";
import { getHealth, predictDefect } from "../services/api";

export default function PredictionPage() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [healthy, setHealthy] = useState(false);

  useEffect(() => {
    getHealth()
      .then((response) => setHealthy(response.status === "ok"))
      .catch(() => setHealthy(false));
  }, []);

  const handleSubmit = async (payload) => {
    setLoading(true);
    setError("");
    try {
      const response = await predictDefect(payload);
      setResult(response);
    } catch (requestError) {
      setError("Prediction request failed. Check backend status, payload shape, and network access.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="page">
      <header className="hero-card compact">
        <div>
          <p className="hero-kicker">Interactive Inference</p>
          <h2>Live Defect Prediction</h2>
          <p className="hero-copy">
            Submit process features and selected sensor arrays to the FastAPI backend and inspect
            the returned prediction confidence in real time.
          </p>
        </div>
        <StatusPill healthy={healthy} />
      </header>

      {error ? <div className="form-error">{error}</div> : null}

      <section className="content-grid prediction-layout">
        <PredictionForm onSubmit={handleSubmit} loading={loading} />
        <PredictionResult result={result} />
      </section>
    </section>
  );
}
