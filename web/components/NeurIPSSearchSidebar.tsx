/**
 * NeurIPSSearchSidebar Component
 *
 * Sidebar for NeurIPS search with profile settings and search input.
 */

import React, { useState } from 'react';
import UserProfileSettingsModal from './UserProfileSettingsModal';

interface NeurIPSSearchSidebarProps {
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onSearch: () => void;
  isSearching: boolean;
  onProfileSave?: () => void;
}

export default function NeurIPSSearchSidebar({
  searchQuery,
  onSearchQueryChange,
  onSearch,
  isSearching,
  onProfileSave,
}: NeurIPSSearchSidebarProps) {
  const [showProfileModal, setShowProfileModal] = useState(false);

  return (
    <>
      <div style={{
        position: 'absolute',
        top: '16px',
        right: '16px',
        zIndex: 5,
        backgroundColor: 'rgba(26, 32, 44, 0.95)',
        padding: '16px',
        borderRadius: '8px',
        width: '320px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
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
              border: '1px solid #4a5568',
              backgroundColor: '#2d3748',
              color: '#e2e8f0',
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
            color: '#a0aec0',
            fontSize: '12px',
            marginBottom: '8px',
            textTransform: 'uppercase',
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
              border: '1px solid #2d3748',
              backgroundColor: '#2d3748',
              color: '#fff',
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
            backgroundColor: (isSearching || !searchQuery.trim()) ? '#4a5568' : '#4a90d9',
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
            backgroundColor: '#2d3748',
            borderRadius: '4px',
            color: '#a0aec0',
            fontSize: '12px',
            textAlign: 'center',
          }}>
            Searching and ranking papers...
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
