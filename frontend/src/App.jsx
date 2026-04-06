import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";

const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const PredictionPage = lazy(() => import("./pages/PredictionPage"));
const VisualizationPage = lazy(() => import("./pages/VisualizationPage"));

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<div className="loading-state">Loading dashboard...</div>}>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/predict" element={<PredictionPage />} />
          <Route path="/visualizations" element={<VisualizationPage />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
