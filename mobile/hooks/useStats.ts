import { useState, useEffect, useRef, useCallback } from 'react';
import { apiStats } from '../lib/api';
import { STATS_POLL_MS } from '../lib/constants';

export interface Stats {
  count: number;
  density: string;
  fps: number;
  is_warming_up: boolean;
  camera_id: string;
}

export function useStats() {
  const [stats, setStats]   = useState<Stats | null>(null);
  const [error, setError]   = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const intervalRef          = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetch = useCallback(async () => {
    try {
      const data = await apiStats();
      setStats(data);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to fetch stats');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetch();
    intervalRef.current = setInterval(fetch, STATS_POLL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetch]);

  return { stats, error, loading, refetch: fetch };
}
