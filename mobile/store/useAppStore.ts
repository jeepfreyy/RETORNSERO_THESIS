import { create } from 'zustand';
import * as SecureStore from 'expo-secure-store';

type Theme = 'dark' | 'light';
type Lang  = 'en'   | 'tl';

const READ_IDS_KEY    = 'read_incident_ids';
const MAX_READ_STORED = 300; // cap stored entries to stay within SecureStore limits

interface AppState {
  // ── Auth ──────────────────────────────────────────────────────────────────
  token:    string | null;
  username: string | null;
  role:     string | null;
  isLoaded: boolean;
  setAuth:    (token: string, username: string, role: string) => Promise<void>;
  clearAuth:  () => Promise<void>;
  loadAuth:   () => Promise<void>;

  // ── Theme ─────────────────────────────────────────────────────────────────
  theme:     Theme;
  setTheme:  (t: Theme) => Promise<void>;

  // ── Language ──────────────────────────────────────────────────────────────
  lang:     Lang;
  setLang:  (l: Lang)  => Promise<void>;

  // ── Onboarding ────────────────────────────────────────────────────────────
  onboardingDone:    boolean;
  completeOnboarding: () => Promise<void>;

  // ── Read / Unread tracking ────────────────────────────────────────────────
  // readIds: persisted set of incident IDs the user has opened
  readIds:           number[];
  markRead:          (id: number) => Promise<void>;
  // allIncidentIds: the full list from the last successful fetch (not persisted)
  allIncidentIds:    number[];
  setAllIncidentIds: (ids: number[]) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // ── Auth defaults ──────────────────────────────────────────────────────────
  token:    null,
  username: null,
  role:     null,
  isLoaded: false,

  setAuth: async (token, username, role) => {
    await SecureStore.setItemAsync('auth_token',    token);
    await SecureStore.setItemAsync('auth_username', username);
    await SecureStore.setItemAsync('auth_role',     role);
    set({ token, username, role });
  },

  clearAuth: async () => {
    await SecureStore.deleteItemAsync('auth_token');
    await SecureStore.deleteItemAsync('auth_username');
    await SecureStore.deleteItemAsync('auth_role');
    await SecureStore.deleteItemAsync(READ_IDS_KEY);
    set({ token: null, username: null, role: null, readIds: [], allIncidentIds: [] });
  },

  loadAuth: async () => {
    try {
      const token    = await SecureStore.getItemAsync('auth_token');
      const username = await SecureStore.getItemAsync('auth_username');
      const role     = await SecureStore.getItemAsync('auth_role');
      const theme    = (await SecureStore.getItemAsync('app_theme'))  as Theme | null;
      const lang     = (await SecureStore.getItemAsync('app_lang'))   as Lang  | null;
      const rawReadIds    = await SecureStore.getItemAsync(READ_IDS_KEY);
      const readIds: number[] = rawReadIds ? JSON.parse(rawReadIds) : [];
      const onboarding    = await SecureStore.getItemAsync('onboarding_done');
      set({
        token, username, role, isLoaded: true,
        theme:          theme ?? 'dark',
        lang:           lang  ?? 'en',
        readIds,
        onboardingDone: onboarding === 'true',
      });
    } catch {
      set({ isLoaded: true });
    }
  },

  // ── Theme defaults ─────────────────────────────────────────────────────────
  theme: 'dark',
  setTheme: async (theme) => {
    await SecureStore.setItemAsync('app_theme', theme);
    set({ theme });
  },

  // ── Language defaults ──────────────────────────────────────────────────────
  lang: 'en',
  setLang: async (lang) => {
    await SecureStore.setItemAsync('app_lang', lang);
    set({ lang });
  },

  // ── Onboarding defaults ───────────────────────────────────────────────────
  onboardingDone: false,
  completeOnboarding: async () => {
    await SecureStore.setItemAsync('onboarding_done', 'true');
    set({ onboardingDone: true });
  },

  // ── Read / Unread defaults ─────────────────────────────────────────────────
  readIds:    [],
  allIncidentIds: [],

  markRead: async (id) => {
    const current = useAppStore.getState().readIds;
    if (current.includes(id)) return;           // already read — skip
    // Keep newest MAX_READ_STORED entries (drop oldest if over cap)
    const updated = [...current, id].slice(-MAX_READ_STORED);
    try {
      await SecureStore.setItemAsync(READ_IDS_KEY, JSON.stringify(updated));
    } catch { /* non-critical */ }
    set({ readIds: updated });
  },

  setAllIncidentIds: (ids) => set({ allIncidentIds: ids }),
}));
