// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// SERVER CONFIGURATION
//
// CLOUDFLARE TUNNEL (active) â€” works from any network / mobile data
// Tunnel URL: https://craig-significant-hanging-metabolism.trycloudflare.com
//
// To switch back to LAN (same Wi-Fi), comment out BASE_URL below and
// uncomment the LAN block:
//   export const SERVER_IP   = 'existing-hereby-prediction-bingo.trycloudflare.com';
//   export const SERVER_PORT = '443';
//   export const BASE_URL    = `http://${SERVER_IP}:${SERVER_PORT}`;
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export const SERVER_IP   = 'existing-hereby-prediction-bingo.trycloudflare.com';
export const SERVER_PORT = '443';
export const BASE_URL    = 'https://existing-hereby-prediction-bingo.trycloudflare.com';

// Polling intervals (milliseconds)
export const STATS_POLL_MS     = 5000;   // how often Home refreshes density
export const INCIDENTS_POLL_MS = 15000;  // how often Incidents list refreshes

// Density colour mapping (matches the web dashboard)
export const DENSITY_COLORS: Record<string, string> = {
  HIGH:   '#ef4444',   // red-500
  MEDIUM: '#f59e0b',   // amber-500
  LOW:    '#10b981',   // emerald-500
  NORMAL: '#10b981',
  MANUAL: '#a855f7',   // purple-500
};

export const DENSITY_BG: Record<string, string> = {
  HIGH:   'rgba(239,68,68,0.15)',
  MEDIUM: 'rgba(245,158,11,0.15)',
  LOW:    'rgba(16,185,129,0.15)',
  NORMAL: 'rgba(16,185,129,0.15)',
  MANUAL: 'rgba(168,85,247,0.15)',
};

export const STATUS_COLORS: Record<string, string> = {
  OPEN:       '#ef4444',
  RESPONDING: '#f59e0b',
  RESOLVED:   '#10b981',
  CLOSED:     '#64748b',
};



