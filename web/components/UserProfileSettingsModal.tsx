/**
 * UserProfileSettingsModal Component
 *
 * Modal for configuring user profile settings including purpose, ranking mode,
 * exclude_local_papers, contrastive_type, top_k, and other preferences.
 */

import React, { useState, useEffect } from 'react';
import { getUserProfile, updateUserProfile, UserProfile } from '../lib/mcp';

interface UserProfileSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave?: () => void;
}

export default function UserProfileSettingsModal({
  isOpen,
  onClose,
  onSave,
}: UserProfileSettingsModalProps) {
  const [profile, setProfile] = useState<Partial<UserProfile>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadProfile();
    }
  }, [isOpen]);

  const loadProfile = async () => {
    setLoading(true);
    setError(null);
    try {
      const loadedProfile = await getUserProfile();
      setProfile(loadedProfile);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load profile');
      // Set defaults if profile doesn't exist
      setProfile({
        purpose: 'general',
        ranking_mode: 'balanced',
        top_k: 5,
        include_contrastive: false,
        contrastive_type: 'method',
        exclude_local_papers: false,
        interests: { primary: [], secondary: [], exploratory: [] },
        keywords: { must_include: [], exclude: { hard: [], soft: [] } },
        preferred_authors: [],
        preferred_institutions: [],
        constraints: { min_year: 2000, require_code: false, exclude_local_papers: false },
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await updateUserProfile(profile);
      if (onSave) {
        onSave();
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  const updateField = (path: string[], value: any) => {
    setProfile((prev) => {
      const newProfile = { ...prev };
      let current: any = newProfile;
      for (let i = 0; i < path.length - 1; i++) {
        if (!current[path[i]]) {
          current[path[i]] = {};
        }
        current = current[path[i]];
      }
      current[path[path.length - 1]] = value;
      return newProfile;
    });
  };

  if (!isOpen) return null;

  return (
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
        width: '700px',
        maxWidth: '90vw',
        maxHeight: '90vh',
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
            User Profile Settings
          </h2>
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

        {/* Content */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
        }}>
          {loading ? (
            <div style={{ color: '#a0aec0', textAlign: 'center' }}>Loading...</div>
          ) : (
            <>
              {error && (
                <div style={{
                  backgroundColor: '#f56565',
                  color: '#fff',
                  padding: '12px',
                  borderRadius: '4px',
                  marginBottom: '16px',
                }}>
                  {error}
                </div>
              )}

              {/* Basic Settings */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#fff', fontSize: '14px', marginBottom: '12px' }}>Basic Settings</h3>
                
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Purpose
                  </label>
                  <select
                    value={profile.purpose || 'general'}
                    onChange={(e) => updateField(['purpose'], e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  >
                    <option value="general">General</option>
                    <option value="literature_review">Literature Review</option>
                    <option value="implementation">Implementation</option>
                    <option value="idea_generation">Idea Generation</option>
                  </select>
                </div>

                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Ranking Mode
                  </label>
                  <select
                    value={profile.ranking_mode || 'balanced'}
                    onChange={(e) => updateField(['ranking_mode'], e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  >
                    <option value="balanced">Balanced</option>
                    <option value="novelty">Novelty</option>
                    <option value="practicality">Practicality</option>
                    <option value="diversity">Diversity</option>
                  </select>
                </div>

                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Top K (Number of papers to return)
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={profile.top_k || 5}
                    onChange={(e) => updateField(['top_k'], parseInt(e.target.value, 10))}
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  />
                </div>

                <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={profile.constraints?.exclude_local_papers || false}
                    onChange={(e) => updateField(['constraints', 'exclude_local_papers'], e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <label style={{ color: '#a0aec0', fontSize: '12px', cursor: 'pointer' }}>
                    Exclude local papers (treat downloaded PDFs as already read)
                  </label>
                </div>

                <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <input
                    type="checkbox"
                    checked={profile.include_contrastive || false}
                    onChange={(e) => updateField(['include_contrastive'], e.target.checked)}
                    style={{ cursor: 'pointer' }}
                  />
                  <label style={{ color: '#a0aec0', fontSize: '12px', cursor: 'pointer' }}>
                    Include contrastive paper
                  </label>
                </div>

                {profile.include_contrastive && (
                  <div style={{ marginBottom: '12px', marginLeft: '24px' }}>
                    <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                      Contrastive Type
                    </label>
                    <select
                      value={profile.contrastive_type || 'method'}
                      onChange={(e) => updateField(['contrastive_type'], e.target.value)}
                      style={{
                        width: '100%',
                        padding: '8px',
                        borderRadius: '4px',
                        border: '1px solid #2d3748',
                        backgroundColor: '#2d3748',
                        color: '#fff',
                        fontSize: '14px',
                      }}
                    >
                      <option value="method">Method</option>
                      <option value="assumption">Assumption</option>
                      <option value="domain">Domain</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Interests */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#fff', fontSize: '14px', marginBottom: '12px' }}>Interests</h3>
                {['primary', 'secondary', 'exploratory'].map((level) => (
                  <div key={level} style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px', textTransform: 'capitalize' }}>
                      {level} (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={(profile.interests?.[level as keyof typeof profile.interests] as string[])?.join(', ') || ''}
                      onChange={(e) => {
                        const values = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                        updateField(['interests', level], values);
                      }}
                      placeholder={`Enter ${level} interests...`}
                      style={{
                        width: '100%',
                        padding: '8px',
                        borderRadius: '4px',
                        border: '1px solid #2d3748',
                        backgroundColor: '#2d3748',
                        color: '#fff',
                        fontSize: '14px',
                      }}
                    />
                  </div>
                ))}
              </div>

              {/* Keywords */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#fff', fontSize: '14px', marginBottom: '12px' }}>Keywords</h3>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Must Include (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={profile.keywords?.must_include?.join(', ') || ''}
                    onChange={(e) => {
                      const values = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                      updateField(['keywords', 'must_include'], values);
                    }}
                    placeholder="Enter must-include keywords..."
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Hard Exclude (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={profile.keywords?.exclude?.hard?.join(', ') || ''}
                    onChange={(e) => {
                      const values = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                      updateField(['keywords', 'exclude', 'hard'], values);
                    }}
                    placeholder="Enter hard exclude keywords..."
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Soft Exclude (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={profile.keywords?.exclude?.soft?.join(', ') || ''}
                    onChange={(e) => {
                      const values = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                      updateField(['keywords', 'exclude', 'soft'], values);
                    }}
                    placeholder="Enter soft exclude keywords..."
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  />
                </div>
              </div>

              {/* Preferred Authors/Institutions */}
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ color: '#fff', fontSize: '14px', marginBottom: '12px' }}>Preferences</h3>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Preferred Authors (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={profile.preferred_authors?.join(', ') || ''}
                    onChange={(e) => {
                      const values = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                      updateField(['preferred_authors'], values);
                    }}
                    placeholder="Enter preferred authors..."
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Preferred Institutions (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={profile.preferred_institutions?.join(', ') || ''}
                    onChange={(e) => {
                      const values = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                      updateField(['preferred_institutions'], values);
                    }}
                    placeholder="Enter preferred institutions..."
                    style={{
                      width: '100%',
                      padding: '8px',
                      borderRadius: '4px',
                      border: '1px solid #2d3748',
                      backgroundColor: '#2d3748',
                      color: '#fff',
                      fontSize: '14px',
                    }}
                  />
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 20px',
          borderTop: '1px solid #2d3748',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '10px',
        }}>
          <button
            onClick={onClose}
            disabled={saving}
            style={{
              padding: '10px 20px',
              borderRadius: '6px',
              border: '1px solid #2d3748',
              backgroundColor: 'transparent',
              color: '#a0aec0',
              fontSize: '14px',
              cursor: saving ? 'not-allowed' : 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || loading}
            style={{
              padding: '10px 20px',
              borderRadius: '6px',
              border: 'none',
              backgroundColor: saving ? '#4a5568' : '#4a90d9',
              color: '#fff',
              fontSize: '14px',
              fontWeight: 500,
              cursor: saving || loading ? 'not-allowed' : 'pointer',
            }}
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

