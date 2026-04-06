export default function StatCard({ title, value, subtitle, accent = "teal" }) {
  return (
    <article className={`stat-card ${accent}`}>
      <p className="stat-title">{title}</p>
      <h3 className="stat-value">{value}</h3>
      <p className="stat-subtitle">{subtitle}</p>
    </article>
  );
}
