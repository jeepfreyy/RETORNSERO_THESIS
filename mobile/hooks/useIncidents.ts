import { useState, useCallback } from 'react';
import { apiIncidents, Incident } from '../lib/api';
import { useAppStore } from '../store/useAppStore';

export function useIncidents() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);

  const setAllIncidentIds = useAppStore((s) => s.setAllIncidentIds);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await apiIncidents();
      setIncidents(data);
      setAllIncidentIds(data.map((i) => i.id)); // keep store in sync for unread count
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load incidents');
    } finally {
      setLoading(false);
    }
  }, [setAllIncidentIds]);

  return { incidents, loading, error, refresh };
}
