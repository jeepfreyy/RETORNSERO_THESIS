import axios from 'axios';
import * as SecureStore from 'expo-secure-store';
import { BASE_URL } from './constants';

// ─── Axios instance ───────────────────────────────────────────────────────────
export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 12000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Request interceptor: inject Bearer token from SecureStore ─────────────────
api.interceptors.request.use(async (config) => {
  try {
    const token = await SecureStore.getItemAsync('auth_token');
    if (token) {
      config.headers = config.headers ?? {};
      config.headers['Authorization'] = `Bearer ${token}`;
    }
  } catch {
    // SecureStore unavailable in web emulation — ignore
  }
  return config;
});

// ── Typed helpers ─────────────────────────────────────────────────────────────
export async function apiLogin(email: string, password: string) {
  const res = await api.post('/api/mobile/login', { email, password });
  return res.data as { success: boolean; token: string; username: string; role: string; message?: string };
}

export async function apiRegister(
  username: string,
  email: string,
  password: string,
) {
  const res = await api.post('/api/register', { username, email, password });
  return res.data as { success: boolean; message: string };
}

/** Step 1 — request a 6-digit OTP be sent to the email address. */
export async function apiForgotRequest(email: string) {
  const res = await api.post('/api/forgot/request', { email });
  return res.data as { success: boolean; message: string };
}

/** Step 2 — verify the OTP; returns a signed reset_ticket on success. */
export async function apiForgotVerify(email: string, code: string) {
  const res = await api.post('/api/forgot/verify', { email, code });
  return res.data as { success: boolean; reset_ticket?: string; message: string };
}

/** Step 3 — submit the new password using the signed ticket. */
export async function apiForgotReset(reset_ticket: string, new_password: string) {
  const res = await api.post('/api/forgot/reset', { reset_ticket, new_password });
  return res.data as { success: boolean; message: string };
}

export async function apiLogout(token: string) {
  await api.post('/api/mobile/logout', null, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function apiStats() {
  const res = await api.get('/api/stats');
  return res.data as {
    count: number;
    density: string;
    fps: number;
    is_warming_up: boolean;
    camera_id: string;
  };
}

export async function apiIncidents() {
  const res = await api.get('/api/incidents');
  return res.data as Incident[];
}

export async function apiAssignSelf(incidentId: number, username: string, role: string) {
  const res = await api.post(`/api/incidents/${incidentId}/respond`, {
    responder_name: username,
    responder_role: role,
    note: 'Responding via mobile app',
  });
  return res.data;
}

export async function apiUpdateStatus(
  incidentId: number,
  status: string,
  resolution_note?: string
) {
  const res = await api.patch(`/api/incidents/${incidentId}/status`, {
    status,
    resolution_note: resolution_note ?? '',
  });
  return res.data;
}

export async function apiRemoveResponder(incidentId: number, responderId: number) {
  const res = await api.delete(`/api/incidents/${incidentId}/responders/${responderId}`);
  return res.data;
}

// ─── Assignment notification helpers ─────────────────────────────────────────

export async function apiPendingNotifications(): Promise<AssignmentNotification[]> {
  const res = await api.get<AssignmentNotification[]>('/api/mobile/notifications');
  return res.data;
}

export async function apiRespondNotification(
  notifId: number,
  action: 'accept' | 'decline',
  reason?: string,
): Promise<void> {
  await api.post(`/api/mobile/notifications/${notifId}/respond`, {
    action,
    reason: reason ?? '',
  });
}

export async function apiMyTasks(): Promise<MyTask[]> {
  const res = await api.get<MyTask[]>('/api/mobile/my_tasks');
  return res.data;
}

// ─── Shared types ─────────────────────────────────────────────────────────────
export interface Responder {
  id: number;
  responder_name: string;
  responder_role: string;
  assigned_at: string;
  note: string | null;
}

export interface PendingAssignment {
  id: number;               // AssignmentNotification id
  assigned_to_name: string;
  assigned_by: string;
  created_at: string;
}

export interface Incident {
  id: number;
  timestamp: string;
  camera_id: string;
  title: string | null;
  density_tag: string;
  threat_level: number;
  incident_status: string;
  reporter_name: string;
  resolution_note: string | null;
  people_count: number;
  location: string | null;
  clip_filename: string | null;
  clip_url: string | null;        // Full relative path to archived clip, e.g. /archive_media/...
  thumbnail_url: string | null;
  responders: Responder[];
  pending_assignments: PendingAssignment[];  // awaiting mobile-app acceptance
}

/** Incident summary embedded in a notification / task record. */
export interface NotificationIncident {
  id: number;
  title: string;
  density_tag: string;
  incident_status: string;
  location: string | null;
  people_count: number;
  camera_id: string;
  timestamp: string;
  threat_level: number;
  reporter_name: string | null;
  thumbnail_url: string | null;
}

/** A pending assignment notification (operator assigned this user). */
export interface AssignmentNotification {
  id: number;
  incident_id: number;
  assigned_by: string;
  created_at: string;
  incident: NotificationIncident;
}

/** One entry in the personal task history (all statuses). */
export interface MyTask {
  id: number;
  incident_id: number;
  assigned_by: string;
  status: 'pending' | 'accepted' | 'declined';
  decline_reason: string | null;
  created_at: string;
  responded_at: string | null;
  incident: NotificationIncident;
}
