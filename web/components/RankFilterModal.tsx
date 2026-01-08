/**
 * RankFilterModal Component
 *
 * Main modal for executing the rank and filter pipeline.
 * Allows users to search for papers and see ranked results.
 */

import React, { useState, useEffect } from 'react';
import { executeRankFilterPipeline, getUserProfile, UserProfile } from '../lib/mcp';
import PaperResultCard, { ScoredPaper } from './PaperResultCard';
import UserProfileSettingsModal from './UserProfileSettingsModal';

interface RankFilterModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type PipelinePhase = 'idle' | 'searching' | 'filtering' | 'scoring' | 'ranking' | 'complete' | 'error';

export default function RankFilterModal({ isOpen, onClose }: RankFilterModalProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ScoredPaper[]>([]);
  const [phase, setPhase] = useState<PipelinePhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<Partial<UserProfile>>({});
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [progress, setProgress] = useState<string>('');

  useEffect(() => {
    if (isOpen) {
      loadProfile();
    }
  }, [isOpen]);

  const loadProfile = async () => {
    try {
      const loadedProfile = await getUserProfile();
      setProfile(loadedProfile);
    } catch (err) {
      // Profile might not exist, use defaults
      setProfile({
        purpose: 'general',
        ranking_mode: 'balanced',
        top_k: 5,
        include_contrastive: false,
        contrastive_type: 'method',
      });
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setError(null);
    setResults([]);
    setPhase('searching');
    setProgress('Searching arXiv...');

    try {
      setPhase('filtering');
      setProgress('Applying filters...');

      const result = await executeRankFilterPipeline({
        query: query.trim(),
        max_results: 50,
        purpose: profile.purpose,
        ranking_mode: profile.ranking_mode,
        top_k: profile.top_k,
        include_contrastive: profile.include_contrastive,
        contrastive_type: profile.contrastive_type,
      });

      if (result.success) {
        setResults(result.ranked_papers || []);
        setPhase('complete');
        setProgress(`Found ${result.ranked_papers?.length || 0} papers`);
      } else {
        throw new Error(result.error || 'Pipeline execution failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute pipeline');
      setPhase('error');
      setProgress('');
    }
  };

  const handleDownloadPdf = async (paperId: string) => {
    // This would call the download API
    window.open(`https://arxiv.org/pdf/${paperId}.pdf`, '_blank');
  };

  if (!isOpen) return null;

  return (
    <>
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1000,
      }}>
        <div style={{
          width: '900px',
          maxWidth: '95vw',
          maxHeight: '95vh',
          backgroundColor: '#1a202c',
          borderRadius: '12px',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
        }}>
          {/* Header */}
          <div style={{
            padding: '16px 20px',
            borderBottom: '1px solid #2d3748',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}>
            <h2 style={{ margin: 0, color: '#fff', fontSize: '18px' }}>
              Rank & Filter Papers
            </h2>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setShowProfileModal(true)}
                style={{
                  padding: '6px 12px',
                  borderRadius: '4px',
                  border: '1px solid #2d3748',
                  backgroundColor: 'transparent',
                  color: '#a0aec0',
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                Profile Settings
              </button>
              <button
                onClick={onClose}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#a0aec0',
                  fontSize: '24px',
                  cursor: 'pointer',
                  lineHeight: 1,
                }}
              >
                &times;
              </button>
            </div>
          </div>

          {/* Content */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '20px',
          }}>
            {/* Search Input */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', gap: '10px', marginBottom: '12px' }}>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSearch();
                    }
                  }}
                  placeholder="Enter search query (e.g., 'transformer attention mechanism')"
                  disabled={phase === 'searching' || phase === 'filtering' || phase === 'scoring' || phase === 'ranking'}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    borderRadius: '6px',
                    border: '1px solid #2d3748',
                    backgroundColor: '#2d3748',
                    color: '#fff',
                    fontSize: '14px',
                    outline: 'none',
                  }}
                />
                <button
                  onClick={handleSearch}
                  disabled={phase === 'searching' || phase === 'filtering' || phase === 'scoring' || phase === 'ranking'}
                  style={{
                    padding: '12px 24px',
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: (phase === 'searching' || phase === 'filtering' || phase === 'scoring' || phase === 'ranking') ? '#4a5568' : '#4a90d9',
                    color: '#fff',
                    fontSize: '14px',
                    fontWeight: 500,
                    cursor: (phase === 'searching' || phase === 'filtering' || phase === 'scoring' || phase === 'ranking') ? 'not-allowed' : 'pointer',
                  }}
                >
                  {phase === 'searching' || phase === 'filtering' || phase === 'scoring' || phase === 'ranking' ? 'Processing...' : 'Search & Rank'}
                </button>
              </div>

              {/* Progress */}
              {progress && (
                <div style={{
                  padding: '8px 12px',
                  backgroundColor: '#2d3748',
                  borderRadius: '4px',
                  color: '#a0aec0',
                  fontSize: '12px',
                }}>
                  {progress}
                </div>
              )}

              {/* Error */}
              {error && (
                <div style={{
                  marginTop: '12px',
                  padding: '12px',
                  backgroundColor: '#f56565',
                  color: '#fff',
                  borderRadius: '4px',
                  fontSize: '13px',
                }}>
                  {error}
                </div>
              )}

              {/* Score Calculation Info */}
              {phase === 'idle' && results.length === 0 && (
                <div style={{
                  marginTop: '12px',
                  padding: '12px',
                  backgroundColor: '#2d3748',
                  borderRadius: '6px',
                  border: '1px solid #4a5568',
                  fontSize: '12px',
                  color: '#a0aec0',
                }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '8px', 
                    marginBottom: '8px',
                    color: '#e2e8f0',
                    fontWeight: 500,
                  }}>
                    <span>💡</span>
                    <span>Score Calculation Tips</span>
                  </div>
                  <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: 1.8 }}>
                    <li>
                      <strong>Semantic Score (30%):</strong> Set your research interests in Profile Settings to get higher relevance scores
                    </li>
                    <li>
                      <strong>Keywords (10%):</strong> Papers matching your "must include" keywords get bonus points
                    </li>
                    <li>
                      <strong>Recency (20%):</strong> Recent papers (published within 1 year) score higher
                    </li>
                    <li>
                      <strong>Practicality (15%):</strong> Papers with GitHub code or local PDFs get bonus points
                    </li>
                    <li>
                      <strong>Author/Institution (25%):</strong> Preferred authors and institutions add trust scores
                    </li>
                  </ul>
                  <div style={{ marginTop: '8px', fontSize: '11px', fontStyle: 'italic', color: '#718096' }}>
                    💡 Tip: Configure your interests and preferences in Profile Settings for better ranking results!
                  </div>
                </div>
              )}

              {/* Current Settings Display */}
              {profile && (profile.purpose || profile.ranking_mode) && (
                <div style={{
                  marginTop: '12px',
                  padding: '8px 12px',
                  backgroundColor: '#2d3748',
                  borderRadius: '4px',
                  fontSize: '11px',
                  color: '#718096',
                }}>
                  Using: Purpose={profile.purpose || 'general'}, Mode={profile.ranking_mode || 'balanced'}, Top-K={profile.top_k || 5}
                  {profile.include_contrastive && `, Contrastive=${profile.contrastive_type || 'method'}`}
                </div>
              )}
            </div>

            {/* Results */}
            {results.length > 0 && (
              <div>
                {/* Score Range Info */}
                {results.some(p => p.score.final < 0.3) && (
                  <div style={{
                    marginBottom: '16px',
                    padding: '12px',
                    backgroundColor: '#fbbf2420',
                    border: '1px solid #fbbf2440',
                    borderRadius: '6px',
                    fontSize: '12px',
                    color: '#fbbf24',
                  }}>
                    <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                      💡 Low scores detected
                    </div>
                    <div>
                      Some papers have low scores. To improve ranking:
                      <ul style={{ margin: '6px 0 0 20px', lineHeight: 1.6 }}>
                        <li>Configure your research interests in Profile Settings (30% of score)</li>
                        <li>Set preferred authors/institutions for trust bonuses</li>
                        <li>Recent papers (within 1 year) score higher in recency</li>
                      </ul>
                    </div>
                  </div>
                )}

                <h3 style={{ color: '#fff', fontSize: '16px', marginBottom: '16px' }}>
                  Ranked Results ({results.length})
                </h3>
                {results.map((paper) => (
                  <PaperResultCard
                    key={paper.paper_id}
                    paper={paper}
                    onDownloadPdf={handleDownloadPdf}
                  />
                ))}
              </div>
            )}

            {phase === 'idle' && results.length === 0 && (
              <div style={{
                textAlign: 'center',
                color: '#718096',
                padding: '40px 20px',
              }}>
                Enter a search query and click "Search & Rank" to find and rank papers
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Profile Settings Modal */}
      <UserProfileSettingsModal
        isOpen={showProfileModal}
        onClose={() => {
          setShowProfileModal(false);
          loadProfile(); // Reload profile after saving
        }}
        onSave={() => {
          loadProfile();
        }}
      />
    </>
  );
}

