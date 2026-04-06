import ConfidenceGauge from "./ConfidenceGauge";
import { formatPercent } from "../utils/formatters";

export default function PredictionResult({ result }) {
  if (!result) {
    return (
      <section className="result-card empty">
        <p className="section-kicker">Prediction Output</p>
        <h3>No prediction yet</h3>
        <p>Submit a sample from the form to see the predicted label, probability, and confidence.</p>
      </section>
    );
  }

  const isDefect = result.predicted_label === 1;

  return (
    <section className="result-card">
      <p className="section-kicker">Prediction Output</p>
      <div className="result-headline">
        <div>
          <h3>{isDefect ? "Defect Detected" : "Part Looks OK"}</h3>
          <p>{isDefect ? "The model flags this cycle as defective." : "The model flags this cycle as non-defective."}</p>
        </div>
        <span className={`prediction-badge ${isDefect ? "danger" : "safe"}`}>
          {isDefect ? "Defect" : "OK"}
        </span>
      </div>

      <div className="result-metrics">
        <div>
          <span>Probability</span>
          <strong>{formatPercent(result.defect_probability)}</strong>
        </div>
        <div>
          <span>Threshold</span>
          <strong>{formatPercent(result.threshold)}</strong>
        </div>
      </div>

      <ConfidenceGauge probability={result.defect_probability} />
    </section>
  );
}
