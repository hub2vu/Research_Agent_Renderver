/**
 * useNodeColors Hook
 *
 * File-based node color persistence with real-time sync.
 * Colors are saved to /output/graph/node_colors.json via Vite middleware API.
 *
 * Features:
 * - Load colors from file on mount
 * - Save colors to file on change
 * - Poll for file changes to sync across tabs/sessions
 * - Fallback to localStorage for offline/error cases
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface NodeColorsData {
  colors: Record<string, string>;
  timestamp: number;
}

interface UseNodeColorsReturn {
  nodeColorMap: Record<string, string>;
  setNodeColor: (nodeKey: string, color: string) => void;
  resetNodeColor: (nodeKey: string) => void;
  isLoading: boolean;
  error: string | null;
}

const API_URL = '/api/node-colors';
const CHECK_URL = '/api/node-colors/check';
const POLL_INTERVAL = 1000; // 1 second
const SAVE_DEBOUNCE = 300; // 300ms debounce for saves
const LOCALSTORAGE_KEY = 'nodeColorMap';

export function useNodeColors(): UseNodeColorsReturn {
  const [nodeColorMap, setNodeColorMap] = useState<Record<string, string>>(() => {
    // Initialize from localStorage as fallback
    try {
      return JSON.parse(localStorage.getItem(LOCALSTORAGE_KEY) || '{}');
    } catch {
      return {};
    }
  });

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Track last known mtime to detect external changes
  const lastMtimeRef = useRef<number>(0);
  // Track if we're the source of the latest change (to avoid re-fetching our own saves)
  const pendingSaveRef = useRef<boolean>(false);
  // Debounce timer for saves
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load colors from API
  const loadColors = useCallback(async () => {
    try {
      const res = await fetch(API_URL, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data: NodeColorsData = await res.json();
      const mtime = parseFloat(res.headers.get('X-Last-Modified') || '0');

      lastMtimeRef.current = mtime || data.timestamp || 0;
      setNodeColorMap(data.colors || {});
      setError(null);

      // Sync to localStorage as backup
      try {
        localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(data.colors || {}));
      } catch {
        // Ignore localStorage errors
      }
    } catch (err) {
      console.error('Failed to load node colors:', err);
      setError(err instanceof Error ? err.message : 'Failed to load colors');
      // Keep using localStorage fallback
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Save colors to API (debounced)
  const saveColors = useCallback((colors: Record<string, string>) => {
    // Clear any pending save
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }

    saveTimerRef.current = setTimeout(async () => {
      pendingSaveRef.current = true;

      try {
        const res = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ colors })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const result = await res.json();
        if (result.timestamp) {
          lastMtimeRef.current = result.timestamp;
        }

        setError(null);
      } catch (err) {
        console.error('Failed to save node colors:', err);
        setError(err instanceof Error ? err.message : 'Failed to save colors');
      } finally {
        // Small delay before allowing poll updates again
        setTimeout(() => {
          pendingSaveRef.current = false;
        }, 200);
      }
    }, SAVE_DEBOUNCE);
  }, []);

  // Set a single node's color
  const setNodeColor = useCallback((nodeKey: string, color: string) => {
    if (!nodeKey || !color) return;

    setNodeColorMap(prev => {
      const next = { ...prev, [nodeKey]: color };

      // Sync to localStorage immediately
      try {
        localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(next));
      } catch {
        // Ignore
      }

      // Save to file (debounced)
      saveColors(next);

      return next;
    });
  }, [saveColors]);

  // Reset a single node's color
  const resetNodeColor = useCallback((nodeKey: string) => {
    if (!nodeKey) return;

    setNodeColorMap(prev => {
      const next = { ...prev };
      delete next[nodeKey];

      // Sync to localStorage immediately
      try {
        localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(next));
      } catch {
        // Ignore
      }

      // Save to file (debounced)
      saveColors(next);

      return next;
    });
  }, [saveColors]);

  // Initial load
  useEffect(() => {
    loadColors();
  }, [loadColors]);

  // Poll for external changes
  useEffect(() => {
    const checkForUpdates = async () => {
      // Skip if we just saved
      if (pendingSaveRef.current) return;

      try {
        const res = await fetch(CHECK_URL, { cache: 'no-store' });
        if (!res.ok) return;

        const data = await res.json();
        const mtime = data.mtime || 0;

        // If mtime changed and it's not from our own save, reload
        if (mtime > 0 && mtime !== lastMtimeRef.current) {
          console.log('Node colors file changed externally, reloading...');
          await loadColors();
        }
      } catch {
        // Ignore poll errors
      }
    };

    const interval = setInterval(checkForUpdates, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [loadColors]);

  // Cleanup save timer on unmount
  useEffect(() => {
    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }
    };
  }, []);

  return {
    nodeColorMap,
    setNodeColor,
    resetNodeColor,
    isLoading,
    error
  };
}

export default useNodeColors;
