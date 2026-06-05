/**
 * useNotifications
 *
 * Polls /api/mobile/notifications every 5 seconds while the user is logged in.
 * Returns the list of pending assignment notifications and a dismiss helper
 * that optimistically removes a notification from the list after the user
 * responds (so the modal disappears instantly without waiting for the next poll).
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { AssignmentNotification, apiPendingNotifications } from '../lib/api';
import { useAppStore } from '../store/useAppStore';

const POLL_MS = 5_000; // 5 seconds

export function useNotifications() {
  const { token } = useAppStore();
  const [notifications, setNotifications] = useState<AssignmentNotification[]>([]);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef  = useRef(true);

  const fetchNotifications = useCallback(async () => {
    if (!token || !mountedRef.current) return;
    try {
      const data = await apiPendingNotifications();
      if (mountedRef.current) setNotifications(data);
    } catch {
      // Silent — network may be temporarily unavailable
    }
  }, [token]);

  useEffect(() => {
    mountedRef.current = true;
    if (token) {
      fetchNotifications();
      intervalRef.current = setInterval(fetchNotifications, POLL_MS);
    }
    return () => {
      mountedRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [token, fetchNotifications]);

  /** Optimistically remove a notification by ID after the user responds. */
  const dismiss = useCallback((id: number) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  return {
    notifications,
    /** The oldest pending notification, shown one at a time. */
    current: notifications[0] ?? null,
    refetch: fetchNotifications,
    dismiss,
  };
}
