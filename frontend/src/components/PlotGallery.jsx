import { buildAssetUrl } from "../services/api";

export default function PlotGallery({ plots }) {
  return (
    <section className="plot-grid">
      {plots.map((plot) => (
        <article key={plot.key} className="plot-card">
          <div className="plot-copy">
            <p className="section-kicker">Artifact</p>
            <h3>{plot.title}</h3>
            <p>{plot.description}</p>
          </div>
          <img src={buildAssetUrl(plot.path)} alt={plot.title} className="plot-image" />
        </article>
      ))}
    </section>
  );
}
