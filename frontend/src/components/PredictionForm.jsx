import { useEffect, useMemo, useState } from "react";
import { getSchema } from "../services/api";
import { prettyJson } from "../utils/formatters";

const defaultSequenceTemplate = {
  DXP_Inj1PrsAct: [0.12, 0.28, 0.31, 0.45, 0.53],
  DXP_Inj1VelAct: [0.0, 0.05, 0.1, 0.08, 0.02],
};

export default function PredictionForm({ onSubmit, loading }) {
  const [schema, setSchema] = useState(null);
  const [error, setError] = useState("");
  const [tabularFields, setTabularFields] = useState({});
  const [extraTabularJson, setExtraTabularJson] = useState("{}");
  const [sequenceJson, setSequenceJson] = useState(prettyJson(defaultSequenceTemplate));

  useEffect(() => {
    getSchema()
      .then((response) => {
        setSchema(response);
        const nextFields = {};
        response.suggested_tabular_fields.forEach((field) => {
          nextFields[field] = "";
        });
        setTabularFields(nextFields);
      })
      .catch(() => {
        setError("Unable to load backend schema. Check whether the FastAPI server is running.");
      });
  }, []);

  const helperText = useMemo(() => {
    if (!schema) {
      return "Loading schema from backend...";
    }
    return `Backend expects ${schema.tabular_columns.length} tabular fields and ${schema.sequence_columns.length} sequence channels.`;
  }, [schema]);

  const handleInputChange = (field, value) => {
    setTabularFields((current) => ({ ...current, [field]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    try {
      const parsedExtraTabular = JSON.parse(extraTabularJson || "{}");
      const parsedSequences = JSON.parse(sequenceJson || "{}");

      const curatedTabular = Object.fromEntries(
        Object.entries(tabularFields)
          .filter(([, value]) => value !== "")
          .map(([key, value]) => [key, Number(value)]),
      );

      await onSubmit({
        tabular: { ...parsedExtraTabular, ...curatedTabular },
        sequences: parsedSequences,
      });
    } catch (submitError) {
      setError("Invalid JSON payload. Please correct the tabular or sequence JSON blocks.");
    }
  };

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <div className="section-heading">
        <div>
          <p className="section-kicker">Live Inference</p>
          <h3>Send a Sample to `/predict`</h3>
        </div>
      </div>

      <p className="helper-copy">{helperText}</p>

      <div className="input-grid">
        {Object.keys(tabularFields).map((field) => (
          <label key={field} className="field">
            <span>{field}</span>
            <input
              type="number"
              step="any"
              placeholder="Enter value"
              value={tabularFields[field]}
              onChange={(event) => handleInputChange(field, event.target.value)}
            />
          </label>
        ))}
      </div>

      <div className="editor-grid">
        <label className="field">
          <span>Additional Tabular JSON</span>
          <textarea
            rows={9}
            value={extraTabularJson}
            onChange={(event) => setExtraTabularJson(event.target.value)}
          />
        </label>

        <label className="field">
          <span>Time-Series JSON</span>
          <textarea
            rows={9}
            value={sequenceJson}
            onChange={(event) => setSequenceJson(event.target.value)}
          />
        </label>
      </div>

      {error ? <div className="form-error">{error}</div> : null}

      <button type="submit" className="primary-button" disabled={loading}>
        {loading ? "Running prediction..." : "Predict"}
      </button>
    </form>
  );
}
