import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
});

export const getHealth = async () => {
  const { data } = await apiClient.get("/health");
  return data;
};

export const getSummary = async () => {
  const { data } = await apiClient.get("/artifacts/summary");
  return data;
};

export const getSchema = async () => {
  const { data } = await apiClient.get("/artifacts/schema");
  return data;
};

export const predictDefect = async (payload) => {
  const { data } = await apiClient.post("/predict", payload);
  return data;
};

export const buildAssetUrl = (path) => {
  if (!path) {
    return "";
  }
  return `${API_BASE_URL}${path}`;
};

export default apiClient;
