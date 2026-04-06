import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function MetricsBarChart({ data }) {
  return (
    <div className="chart-card">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Metrics</p>
          <h3>Precision, Recall, F1, and ROC-AUC</h3>
        </div>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(140, 160, 183, 0.18)" />
            <XAxis dataKey="metric" stroke="#7f8ea3" />
            <YAxis domain={[0, 1]} stroke="#7f8ea3" />
            <Tooltip />
            <Legend />
            <Bar dataKey="XGBoost" fill="#1f7aec" radius={[8, 8, 0, 0]} />
            <Bar dataKey="LSTM Fusion" fill="#19b394" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
