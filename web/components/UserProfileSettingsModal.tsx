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
  // Input values state to allow typing spaces and commas freely
  const [inputValues, setInputValues] = useState<Record<string, string>>({});

  useEffect(() => {
    if (isOpen) {
      loadProfile();
    }
  }, [isOpen]);

  // Helper function to parse comma-separated values
  const parseCommaSeparated = (value: string): string[] => {
    return value.split(',').map(s => s.trim()).filter(s => s);
  };

  // Handle input change - update both input value and profile
  const handleInputChange = (key: string, value: string) => {
    // Update input value (raw string)
    setInputValues(prev => ({ ...prev, [key]: value }));
    
    // Parse and update profile
    const parsed = parseCommaSeparated(value);
    
    if (key.startsWith('interests.')) {
      const level = key.split('.')[1];
      updateField(['interests', level], parsed);
    } else if (key === 'keywords.must_include') {
      updateField(['keywords', 'must_include'], parsed);
    } else if (key === 'keywords.exclude.hard') {
      updateField(['keywords', 'exclude', 'hard'], parsed);
    } else if (key === 'keywords.exclude.soft') {
      updateField(['keywords', 'exclude', 'soft'], parsed);
    } else if (key === 'preferred_authors') {
      updateField(['preferred_authors'], parsed.length > 0 ? parsed : []);
    } else if (key === 'preferred_institutions') {
      updateField(['preferred_institutions'], parsed.length > 0 ? parsed : []);
    }
  };

  const loadProfile = async () => {
    setLoading(true);
    setError(null);
    try {
      const loadedProfile = await getUserProfile();
      // Filter out empty arrays for preferred_authors and preferred_institutions
      const cleanedProfile = { ...loadedProfile };
      if (Array.isArray(cleanedProfile.preferred_authors) && cleanedProfile.preferred_authors.length === 0) {
        cleanedProfile.preferred_authors = undefined;
      }
      if (Array.isArray(cleanedProfile.preferred_institutions) && cleanedProfile.preferred_institutions.length === 0) {
        cleanedProfile.preferred_institutions = undefined;
      }
      setProfile(cleanedProfile);
      
      // Initialize input values from profile
      setInputValues({
        'interests.primary': cleanedProfile.interests?.primary?.join(', ') || '',
        'interests.secondary': cleanedProfile.interests?.secondary?.join(', ') || '',
        'interests.exploratory': cleanedProfile.interests?.exploratory?.join(', ') || '',
        'keywords.must_include': cleanedProfile.keywords?.must_include?.join(', ') || '',
        'keywords.exclude.hard': cleanedProfile.keywords?.exclude?.hard?.join(', ') || '',
        'keywords.exclude.soft': cleanedProfile.keywords?.exclude?.soft?.join(', ') || '',
        'preferred_authors': cleanedProfile.preferred_authors?.join(', ') || '',
        'preferred_institutions': cleanedProfile.preferred_institutions?.join(', ') || '',
      });
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
        preferred_authors: undefined,
        preferred_institutions: undefined,
        constraints: { min_year: 2000, require_code: false, exclude_local_papers: false },
      });
      // Initialize input values with defaults
      setInputValues({
        'interests.primary': '',
        'interests.secondary': '',
        'interests.exploratory': '',
        'keywords.must_include': '',
        'keywords.exclude.hard': '',
        'keywords.exclude.soft': '',
        'preferred_authors': '',
        'preferred_institutions': '',
      });
    } finally {
      setLoading(false);
    }
  };

  // Helper function to remove undefined values from object recursively
  const removeUndefined = (obj: any): any => {
    if (obj === null || obj === undefined) {
      return obj;
    }
    if (Array.isArray(obj)) {
      return obj.map(removeUndefined);
    }
    if (typeof obj === 'object') {
      const cleaned: any = {};
      for (const [key, value] of Object.entries(obj)) {
        if (value !== undefined) {
          cleaned[key] = removeUndefined(value);
        }
      }
      return cleaned;
    }
    return obj;
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      // Deep clone profile to avoid mutating state
      const cleanedProfile = JSON.parse(JSON.stringify(profile));
      
      // Extract exclude_local_papers from constraints and add as top-level field
      // Backend accepts exclude_local_papers both as top-level and inside constraints
      if (cleanedProfile.constraints?.exclude_local_papers !== undefined) {
        cleanedProfile.exclude_local_papers = cleanedProfile.constraints.exclude_local_papers;
      }
      
      // Ensure preferred_authors and preferred_institutions are arrays (not undefined)
      // Backend needs to receive them even if empty to process updates correctly
      if (!Array.isArray(cleanedProfile.preferred_authors)) {
        cleanedProfile.preferred_authors = [];
      }
      if (!Array.isArray(cleanedProfile.preferred_institutions)) {
        cleanedProfile.preferred_institutions = [];
      }
      
      // Remove all undefined fields recursively to prevent them from being sent as null
      // But keep empty arrays for preferred_authors and preferred_institutions
      const profileToSend = removeUndefined(cleanedProfile);
      
      console.log('[UserProfileSettingsModal] Saving profile:', profileToSend);
      
      await updateUserProfile(profileToSend);
      
      console.log('[UserProfileSettingsModal] Profile saved successfully');
      
      if (onSave) {
        onSave();
      }
      onClose();
    } catch (err) {
      console.error('[UserProfileSettingsModal] Failed to save profile:', err);
      setError(err instanceof Error ? err.message : 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  const updateField = (path: string[], value: any) => {
    setProfile((prev) => {
      // Deep clone to ensure immutability
      const newProfile = JSON.parse(JSON.stringify(prev));
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

                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px' }}>
                    Minimum Year
                  </label>
                  <input
                    type="number"
                    min="1900"
                    max={new Date().getFullYear()}
                    value={profile.constraints?.min_year || 2000}
                    onChange={(e) => updateField(['constraints', 'min_year'], parseInt(e.target.value, 10))}
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
                
                {/* Info Box */}
                <div style={{
                  marginBottom: '12px',
                  padding: '10px',
                  backgroundColor: '#2d3748',
                  borderRadius: '4px',
                  fontSize: '11px',
                  color: '#a0aec0',
                  border: '1px solid #4a5568',
                }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '6px', 
                    marginBottom: '6px',
                    color: '#e2e8f0',
                    fontWeight: 500,
                  }}>
                    <span>📊</span>
                    <span>This affects 30% of your paper scores!</span>
                  </div>
                  <div style={{ lineHeight: 1.6 }}>
                    • <strong>Primary:</strong> Main research topics (e.g., "transformer", "attention")
                    <br />
                    • <strong>Secondary:</strong> Related areas (e.g., "NLP", "computer vision")
                    <br />
                    • <strong>Exploratory:</strong> Areas you want to explore (e.g., "reinforcement learning")
                    <br />
                    <div style={{ marginTop: '6px', fontStyle: 'italic', color: '#718096' }}>
                      ⚠️ Leaving interests empty will result in lower semantic scores (0-30% range)
                    </div>
                  </div>
                </div>

                {['primary', 'secondary', 'exploratory'].map((level) => (
                  <div key={level} style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', color: '#a0aec0', fontSize: '12px', marginBottom: '4px', textTransform: 'capitalize' }}>
                      {level} (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={inputValues[`interests.${level}`] || ''}
                      onChange={(e) => handleInputChange(`interests.${level}`, e.target.value)}
                      onBlur={(e) => {
                        // Format on blur: parse and rejoin with proper spacing
                        const parsed = parseCommaSeparated(e.target.value);
                        setInputValues(prev => ({ ...prev, [`interests.${level}`]: parsed.join(', ') }));
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
                    value={inputValues['keywords.must_include'] || ''}
                    onChange={(e) => handleInputChange('keywords.must_include', e.target.value)}
                    onBlur={(e) => {
                      const parsed = parseCommaSeparated(e.target.value);
                      setInputValues(prev => ({ ...prev, 'keywords.must_include': parsed.join(', ') }));
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
                    value={inputValues['keywords.exclude.hard'] || ''}
                    onChange={(e) => handleInputChange('keywords.exclude.hard', e.target.value)}
                    onBlur={(e) => {
                      const parsed = parseCommaSeparated(e.target.value);
                      setInputValues(prev => ({ ...prev, 'keywords.exclude.hard': parsed.join(', ') }));
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
                    value={inputValues['keywords.exclude.soft'] || ''}
                    onChange={(e) => handleInputChange('keywords.exclude.soft', e.target.value)}
                    onBlur={(e) => {
                      const parsed = parseCommaSeparated(e.target.value);
                      setInputValues(prev => ({ ...prev, 'keywords.exclude.soft': parsed.join(', ') }));
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
                    value={inputValues['preferred_authors'] || ''}
                    onChange={(e) => handleInputChange('preferred_authors', e.target.value)}
                    onBlur={(e) => {
                      const parsed = parseCommaSeparated(e.target.value);
                      setInputValues(prev => ({ ...prev, 'preferred_authors': parsed.join(', ') }));
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
                    value={inputValues['preferred_institutions'] || ''}
                    onChange={(e) => handleInputChange('preferred_institutions', e.target.value)}
                    onBlur={(e) => {
                      const parsed = parseCommaSeparated(e.target.value);
                      setInputValues(prev => ({ ...prev, 'preferred_institutions': parsed.join(', ') }));
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

