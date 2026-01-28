/**
 * NavBar Component
 *
 * Navigation bar displayed at the top of all pages.
 * Provides links to Global Paper Graph and NeurIPS 2025 pages,
 * plus a button to open the LLM Chat popup.
 */

import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

interface NavBarProps {
  onOpenChat: () => void;
  onOpenPipeline: () => void;
}

export default function NavBar({ onOpenChat, onOpenPipeline }: NavBarProps) {
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  const buttonStyle = (path: string): React.CSSProperties => ({
    padding: '8px 16px',
    marginRight: '8px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    transition: 'all 0.2s ease',
    backgroundColor: isActive(path) ? '#4a90d9' : '#2d3748',
    color: isActive(path) ? '#fff' : '#cbd5e0',
  });

  const chatButtonStyle: React.CSSProperties = {
    padding: '8px 16px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    backgroundColor: '#48bb78',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  };

  const pipelineButtonStyle: React.CSSProperties = {
    padding: '8px 16px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 500,
    backgroundColor: '#9f7aea',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  };

  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '12px 20px',
      backgroundColor: '#1a202c',
      borderBottom: '1px solid #2d3748',
    }}>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <span style={{
          fontSize: '18px',
          fontWeight: 'bold',
          color: '#fff',
          marginRight: '24px'
        }}>
          Research Agent
        </span>

        <button
          style={buttonStyle('/')}
          onClick={() => navigate('/')}
          onMouseEnter={(e) => {
            if (!isActive('/')) e.currentTarget.style.backgroundColor = '#3d4a5c';
          }}
          onMouseLeave={(e) => {
            if (!isActive('/')) e.currentTarget.style.backgroundColor = '#2d3748';
          }}
        >
          Global Paper Graph
        </button>

        <button
          style={buttonStyle('/neurips2025')}
          onClick={() => navigate('/neurips2025')}
          onMouseEnter={(e) => {
            if (!isActive('/neurips2025')) e.currentTarget.style.backgroundColor = '#3d4a5c';
          }}
          onMouseLeave={(e) => {
            if (!isActive('/neurips2025')) e.currentTarget.style.backgroundColor = '#2d3748';
          }}
        >
          NeurIPS 2025
        </button>

        <button
          style={buttonStyle('/iclr2025')}
          onClick={() => navigate('/iclr2025')}
          onMouseEnter={(e) => {
            if (!isActive('/iclr2025')) e.currentTarget.style.backgroundColor = '#3d4a5c';
          }}
          onMouseLeave={(e) => {
            if (!isActive('/iclr2025')) e.currentTarget.style.backgroundColor = '#2d3748';
          }}
        >
          ICLR 2025
        </button>
      </div>

      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <button
          style={pipelineButtonStyle}
          onClick={onOpenPipeline}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#805ad5'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#9f7aea'}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14l-5-5 1.41-1.41L12 14.17l4.59-4.58L18 11l-6 6z"/>
          </svg>
          Pipeline
        </button>
        <button
          style={chatButtonStyle}
          onClick={onOpenChat}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#38a169'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#48bb78'}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
          </svg>
          LLM Chat
        </button>
      </div>
    </nav>
  );
}
