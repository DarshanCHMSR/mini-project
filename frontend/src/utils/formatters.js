export const formatPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${(value * 100).toFixed(2)}%`;
};

export const formatMetricLabel = (value) =>
  value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export const formatCount = (value) => Intl.NumberFormat("en-US").format(value || 0);

export const prettyJson = (value) => JSON.stringify(value, null, 2);
