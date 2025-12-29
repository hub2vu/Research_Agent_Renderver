import React from 'react';
import { BrowserRouter, Routes, Route, useParams } from 'react-router-dom';
import GlobalGraphPage from './pages/GlobalGraphPage';
import PaperGraphPage from './pages/PaperGraphPage';

function PaperGraphWrapper() {
  const { paperId } = useParams<{ paperId: string }>();
  return <PaperGraphPage paperId={paperId || ''} />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<GlobalGraphPage />} />
        <Route path="/paper/:paperId" element={<PaperGraphWrapper />} />
      </Routes>
    </BrowserRouter>
  );
}
