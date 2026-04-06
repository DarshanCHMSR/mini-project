import { useEffect, useState } from "react";
import PlotGallery from "../components/PlotGallery";
import { getSummary } from "../services/api";

export default function VisualizationPage() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getSummary()
      .then(setSummary)
      .catch(() => setError("Unable to load backend plots. Make sure FastAPI is serving `/plots`."));
  }, []);

  return (
    <section className="page">
      <header className="hero-card compact">
        <div>
          <p className="hero-kicker">Training Artifacts</p>
          <h2>Visual Model Diagnostics</h2>
          <p className="hero-copy">
            Explore saved confusion matrices, training curves, ROC performance, feature importance,
            and model comparison plots produced by the backend pipeline.
          </p>
        </div>
      </header>

      {error ? <div className="form-error">{error}</div> : null}

      <PlotGallery plots={summary?.plots ?? []} />
    </section>
  );
}
