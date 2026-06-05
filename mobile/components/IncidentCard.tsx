import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Incident } from '../lib/api';
import { DensityBadge } from './DensityBadge';
import { STATUS_COLORS } from '../lib/constants';
import { useTheme } from '../hooks/useTheme';
import { useAppStore } from '../store/useAppStore';

interface Props {
  incident: Incident;
}

function timeAgo(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60)    return `${diff}s ago`;
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export function IncidentCard({ incident: inc }: Props) {
  const router   = useRouter();
  const { colors } = useTheme();
  const { readIds, markRead } = useAppStore();

  const statusColor = STATUS_COLORS[inc.incident_status] ?? '#94a3b8';
  const isUnread    = !readIds.includes(inc.id);

  function handlePress() {
    markRead(inc.id);
    router.push({ pathname: '/incident/[id]', params: { id: String(inc.id) } });
  }

  return (
    <TouchableOpacity
      style={[
        styles.card,
        {
          backgroundColor: colors.surface,
          borderLeftColor: isUnread ? colors.accent : statusColor,
          borderLeftWidth: isUnread ? 4 : 3,
          borderColor: isUnread
            ? `${colors.accent}40`
            : colors.borderLight,
        },
      ]}
      activeOpacity={0.75}
      onPress={handlePress}
    >
      {/* Top row */}
      <View style={styles.topRow}>
        <View style={styles.badges}>
          <View style={[styles.statusBadge, { backgroundColor: `${statusColor}22`, borderColor: statusColor }]}>
            <Text style={[styles.statusText, { color: statusColor }]}>{inc.incident_status}</Text>
          </View>
          <DensityBadge density={inc.density_tag} size="sm" />
        </View>

        {/* Time + unread indicator */}
        <View style={styles.timeRow}>
          {isUnread && (
            <View style={[styles.unreadDot, { backgroundColor: colors.accent }]} />
          )}
          <Text style={[styles.timeText, { color: isUnread ? colors.textSec : colors.textMuted }]}>
            {timeAgo(inc.timestamp)}
          </Text>
        </View>
      </View>

      {/* Title — bolder when unread */}
      <Text
        style={[
          styles.title,
          {
            color:      isUnread ? colors.text : colors.textSec,
            fontWeight: isUnread ? '800' : '700',
          },
        ]}
        numberOfLines={1}
      >
        {inc.title ?? 'Untitled Incident'}
      </Text>

      <Text style={[styles.meta, { color: colors.textMuted }]}>
        {inc.people_count > 0 ? `${inc.people_count} people · ` : ''}
        {inc.responders.length} responder{inc.responders.length !== 1 ? 's' : ''}
        {inc.location ? ` · ${inc.location}` : ''}
      </Text>

      {/* Unread badge — corner pill */}
      {isUnread && (
        <View style={[styles.unreadBadge, { backgroundColor: colors.accent }]}>
          <Text style={styles.unreadBadgeText}>NEW</Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 12, padding: 14, marginBottom: 10,
    borderWidth: 1, position: 'relative',
  },
  topRow: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between', marginBottom: 6,
  },
  badges:    { flexDirection: 'row', gap: 6, alignItems: 'center' },
  timeRow:   { flexDirection: 'row', alignItems: 'center', gap: 5 },
  statusBadge: { borderRadius: 6, borderWidth: 1, paddingHorizontal: 7, paddingVertical: 2 },
  statusText:  { fontSize: 10, fontWeight: '700', letterSpacing: 0.5 },
  timeText:    { fontSize: 11 },

  // Small filled dot next to the time label
  unreadDot: {
    width: 7, height: 7, borderRadius: 4,
  },

  title: { fontSize: 14, marginBottom: 4 },
  meta:  { fontSize: 11 },

  // "NEW" pill badge — top-right corner
  unreadBadge: {
    position: 'absolute', top: 10, right: 10,
    paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: 6,
  },
  unreadBadgeText: {
    fontSize: 9, fontWeight: '800', color: '#fff', letterSpacing: 0.8,
  },
});
