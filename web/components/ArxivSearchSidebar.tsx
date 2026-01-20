/**
 * ArxivSearchSidebar Component
 *
 * Sidebar for arXiv search with profile settings and search input.
 */

import React, { useState } from 'react';
import UserProfileSettingsModal from './UserProfileSettingsModal';

interface ArxivSearchSidebarProps {
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onSearch: () => void;
  isSearching: boolean;
  onProfileSave?: () => void;
}

export default function ArxivSearchSidebar({
  searchQuery,
  onSearchQueryChange,
  onSearch,
  isSearching,
  onProfileSave,
}: ArxivSearchSidebarProps) {
  const [showProfileModal, setShowProfileModal] = useState(false);

  return (
    <>
      <div style={{
        position: 'absolute',
        top: '16px',
        right: '16px',
        zIndex: 5,
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        padding: '16px',
        borderRadius: '8px',
        width: '320px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
        maxHeight: 'calc(100vh - 120px)',
        overflowY: 'auto',
      }}>
        {/* Profile Settings Button */}
        <div style={{ marginBottom: '16px' }}>
          <button
            onClick={() => setShowProfileModal(true)}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '6px',
              border: '1px solid #e2e8f0',
              backgroundColor: '#f7fafc',
              color: '#4a5568',
              fontSize: '13px',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            ⚙️ Profile Settings
          </button>
        </div>

        {/* Search Input */}
        <div style={{ marginBottom: '12px' }}>
          <label style={{
            display: 'block',
            color: '#4a5568',
            fontSize: '12px',
            marginBottom: '8px',
            textTransform: 'uppercase',
            fontWeight: 500,
          }}>
            Search Query
          </label>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !isSearching) {
                e.preventDefault();
                onSearch();
              }
            }}
            placeholder="Enter research topic..."
            disabled={isSearching}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '6px',
              border: '1px solid #e2e8f0',
              backgroundColor: '#fff',
              color: '#1a202c',
              fontSize: '14px',
              outline: 'none',
            }}
          />
        </div>

        {/* Search Button */}
        <button
          onClick={onSearch}
          disabled={isSearching || !searchQuery.trim()}
          style={{
            width: '100%',
            padding: '12px',
            borderRadius: '6px',
            border: 'none',
            backgroundColor: (isSearching || !searchQuery.trim()) ? '#cbd5e0' : '#4299e1',
            color: '#fff',
            fontSize: '14px',
            fontWeight: 500,
            cursor: (isSearching || !searchQuery.trim()) ? 'not-allowed' : 'pointer',
          }}
        >
          {isSearching ? 'Analyzing...' : '분석 실행'}
        </button>

        {isSearching && (
          <div style={{
            marginTop: '12px',
            padding: '8px',
            backgroundColor: '#f7fafc',
            borderRadius: '4px',
            color: '#718096',
            fontSize: '12px',
            textAlign: 'center',
            border: '1px solid #e2e8f0',
          }}>
            Searching arXiv and ranking papers...
          </div>
        )}
      </div>

      {/* Profile Settings Modal */}
      <UserProfileSettingsModal
        isOpen={showProfileModal}
        onClose={() => setShowProfileModal(false)}
        onSave={() => {
          setShowProfileModal(false);
          if (onProfileSave) {
            onProfileSave();
          }
        }}
      />
    </>
  );
}
