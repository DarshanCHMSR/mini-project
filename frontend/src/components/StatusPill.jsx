export default function StatusPill({ healthy }) {
  return (
    <span className={`status-pill ${healthy ? "healthy" : "down"}`}>
      <span className="status-dot" />
      {healthy ? "API Healthy" : "API Unreachable"}
    </span>
  );
}
