import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ModelComparisonChart({ data }) {
  return (
    <div className="chart-card">
      <div className="section-heading">
        <div>
          <p className="section-kicker">Comparison</p>
          <h3>Model Accuracy Snapshot</h3>
        </div>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(140, 160, 183, 0.18)" />
            <XAxis dataKey="model" stroke="#7f8ea3" />
            <YAxis domain={[0, 1]} stroke="#7f8ea3" />
            <Tooltip />
            <Bar dataKey="accuracy" fill="#ff7a59" radius={[10, 10, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
