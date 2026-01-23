import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, useParams } from 'react-router-dom';
import GlobalGraphPage from './pages/GlobalGraphPage';
import PaperGraphPage from './pages/PaperGraphPage';
import NeurIPS2025Page from './pages/NeurIPS2025Page';
import NavBar from './components/NavBar';
import LLMChatPopup from './components/LLMChatPopup';
import NotePage from './pages/NotePage';

function PaperGraphWrapper() {
  const { paperId } = useParams<{ paperId: string }>();
  return <PaperGraphPage paperId={paperId || ''} />;
}

function NotePageWrapper() {
  const { noteId } = useParams<{ noteId: string }>();
  return <NotePage noteId={noteId || ''} />;
}

function AppContent() {
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#0d1117' }}>
      <NavBar
        onOpenChat={() => setIsChatOpen(true)}
      />

      <Routes>
        <Route path="/" element={<GlobalGraphPage />} />
        <Route path="/paper/:paperId" element={<PaperGraphWrapper />} />
        <Route path="/neurips2025" element={<NeurIPS2025Page />} />
        <Route path="/note/:noteId" element={<NotePage />} />
      </Routes>

      <LLMChatPopup
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
      />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}
