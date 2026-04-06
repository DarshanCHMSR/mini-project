import { formatPercent } from "../utils/formatters";

export default function ConfidenceGauge({ probability = 0 }) {
  const width = Math.max(4, Math.min(100, probability * 100));
  const tone = probability >= 0.7 ? "danger" : probability >= 0.4 ? "warning" : "safe";

  return (
    <div className="confidence-card">
      <div className="confidence-header">
        <div>
          <p className="section-kicker">Confidence</p>
          <h3>Predicted Defect Probability</h3>
        </div>
        <strong>{formatPercent(probability)}</strong>
      </div>
      <div className="confidence-track">
        <div className={`confidence-fill ${tone}`} style={{ width: `${width}%` }} />
      </div>
      <p className="confidence-caption">
        Higher values indicate stronger model belief that the current sample is defective.
      </p>
    </div>
  );
}
