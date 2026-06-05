/**
 * theme/index.ts — Barangay Sentinel color palettes
 *
 * Two palettes: DarkColors (default) and LightColors.
 * Every screen imports Colors type and uses useTheme() to get the active one.
 */

export type Colors = {
  /** Full-screen background */
  background: string;
  /** Card / surface backgrounds */
  surface: string;
  /** Input field backgrounds (sit inside a card, so slightly different) */
  surfaceInput: string;
  /** Primary border */
  border: string;
  /** Lighter / hairline border */
  borderLight: string;
  /** Primary text (headings, values) */
  text: string;
  /** Secondary text (subtitles, labels) */
  textSec: string;
  /** Muted text (meta, captions) */
  textMuted: string;
  /** Very muted text / placeholders */
  textHint: string;
  /** Brand green accent */
  accent: string;
  /** Soft accent fill for chips and active states */
  accentSoft: string;
  /** Expo StatusBar style prop */
  statusBar: 'light' | 'dark';
  /** Bottom tab-bar background */
  tabBar: string;
  /** Bottom tab-bar top border */
  tabBarBorder: string;
};

export const DarkColors: Colors = {
  background:   '#0f172a',
  surface:      '#1e293b',
  surfaceInput: '#0f172a',
  border:       '#334155',
  borderLight:  '#1e293b',
  text:         '#f1f5f9',
  textSec:      '#94a3b8',
  textMuted:    '#64748b',
  textHint:     '#475569',
  accent:       '#10b981',
  accentSoft:   'rgba(16,185,129,0.15)',
  statusBar:    'light',
  tabBar:       '#0f172a',
  tabBarBorder: '#1e293b',
};

export const LightColors: Colors = {
  background:   '#f1f5f9',
  surface:      '#ffffff',
  surfaceInput: '#f8fafc',
  border:       '#cbd5e1',
  borderLight:  '#e2e8f0',
  text:         '#0f172a',
  textSec:      '#334155',
  textMuted:    '#475569',
  textHint:     '#94a3b8',
  accent:       '#059669',
  accentSoft:   'rgba(5,150,105,0.12)',
  statusBar:    'dark',
  tabBar:       '#ffffff',
  tabBarBorder: '#e2e8f0',
};
