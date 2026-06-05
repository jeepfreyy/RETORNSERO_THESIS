import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, FlatList, TouchableOpacity,
  ActivityIndicator, RefreshControl, Alert,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { MyTask, apiMyTasks, apiLeaveIncident } from '../../lib/api';
import { useAppStore } from '../../store/useAppStore';
import { DENSITY_COLORS, DENSITY_BG, STATUS_COLORS } from '../../lib/constants';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

const TASK_STATUS_COLOR: Record<string, string> = {
  pending:  '#f59e0b',
  accepted: '#10b981',
  declined: '#ef4444',
  removed:  '#64748b',   // slate — operator removed
  left:     '#64748b',   // slate — self left
};

const TASK_STATUS_LABEL: Record<string, string> = {
  pending:  'PENDING',
  accepted: 'ACCEPTED',
  declined: 'DECLINED',
  removed:  'REMOVED',
  left:     'LEFT',
};

const INCIDENT_ACTIVE_STATUSES = new Set(['OPEN', 'RESPONDING']);

export default function TasksScreen() {
  const { token } = useAppStore();
  const router    = useRouter();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s      = useMemo(() => makeStyles(colors), [colors]);
  const insets = useSafeAreaInsets();

  const [tasks,     setTasks]     = useState<MyTask[]>([]);
  const [loading,   setLoading]   = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error,     setError]     = useState<string | null>(null);

  const fetchTasks = useCallback(async (isRefresh = false) => {
    if (!token) return;
    if (isRefresh) setRefreshing(true);
    else           setLoading(true);
    setError(null);
    try {
      const data = await apiMyTasks();
      setTasks(data);
    } catch {
      setError(t('tasks.error'));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [token]);

  useFocusEffect(useCallback(() => { fetchTasks(); }, [fetchTasks]));

  // ── Leave incident ──────────────────────────────────────────────────────────
  function handleLeave(task: MyTask) {
    Alert.alert(
      'Remove Assignment',
      `Remove yourself from "${task.incident.title ?? 'this incident'}"?\n\nThis assignment will be deleted from your task list.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            try {
              await apiLeaveIncident(task.incident_id);
              // Optimistically remove from local list while refresh loads
              setTasks(prev => prev.filter(t => t.incident_id !== task.incident_id || t.id !== task.id));
              fetchTasks(true);
            } catch (e: any) {
              const msg = e?.response?.data?.error ?? 'Could not remove the assignment. Please try again.';
              Alert.alert('Error', msg);
            }
          },
        },
      ],
    );
  }

  // Active: accepted by user AND incident still open/responding AND NOT removed by operator
  const active  = tasks.filter(
    t => t.status === 'accepted' && INCIDENT_ACTIVE_STATUSES.has(t.incident.incident_status),
  );
  const history = tasks.filter(t => !active.includes(t));

  if (loading) {
    return (
      <View style={s.centered}>
        <ActivityIndicator color={colors.accent} size="large" />
        <Text style={s.loadingText}>{t('tasks.loading')}</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={s.centered}>
        <Text style={s.errorText}>⚠️ {error}</Text>
        <TouchableOpacity onPress={() => fetchTasks()} style={s.retryBtn}>
          <Text style={s.retryText}>{t('inc.retry')}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={s.screen}>
      <View style={[s.header, { paddingTop: insets.top + 16 }]}>
        <Text style={s.headerTitle}>{t('tasks.title')}</Text>
        <Text style={s.headerSub}>{t('tasks.subtitle')}</Text>
      </View>

      <FlatList
        data={[]}
        renderItem={null}
        ListHeaderComponent={() => (
          <>
            <SectionHeader title={t('tasks.active')}  count={active.length}  s={s} />
            {active.length === 0
              ? <EmptyState message={t('tasks.noActive')} s={s} />
              : active.map(task => (
                  <TaskCard
                    key={task.id}
                    task={task}
                    onPress={() => router.push(`/incident/${task.incident_id}`)}
                    onLeave={() => handleLeave(task)}
                    colors={colors} s={s} t={t}
                  />
                ))
            }

            <SectionHeader title={t('tasks.history')} count={history.length} s={s} style={{ marginTop: 24 }} />
            {history.length === 0
              ? <EmptyState message={t('tasks.noHistory')} s={s} />
              : history.map(task => (
                  <TaskCard
                    key={task.id}
                    task={task}
                    onPress={() => router.push(`/incident/${task.incident_id}`)}
                    colors={colors} s={s} t={t}
                  />
                ))
            }
            <View style={{ height: 32 }} />
          </>
        )}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={() => fetchTasks(true)} tintColor={colors.accent} />
        }
        contentContainerStyle={{ paddingBottom: 24 }}
      />
    </View>
  );
}

function SectionHeader({ title, count, s, style }: {
  title: string; count: number;
  s: ReturnType<typeof makeStyles>; style?: object;
}) {
  return (
    <View style={[s.sectionHeader, style]}>
      <Text style={s.sectionTitle}>{title}</Text>
      <View style={s.countBadge}>
        <Text style={s.countBadgeText}>{count}</Text>
      </View>
    </View>
  );
}

function TaskCard({ task, onPress, onLeave, colors, s, t }: {
  task: MyTask; onPress: () => void; onLeave?: () => void;
  colors: Colors;
  s: ReturnType<typeof makeStyles>;
  t: (key: string, vars?: any) => string;
}) {
  const { incident } = task;
  const densityColor    = DENSITY_COLORS[incident.density_tag]    ?? '#64748b';
  const densityBg       = DENSITY_BG[incident.density_tag]        ?? colors.surface;
  const statusColor     = STATUS_COLORS[incident.incident_status] ?? '#64748b';
  const taskStatusColor = TASK_STATUS_COLOR[task.status] ?? '#64748b';
  const taskStatusLabel = TASK_STATUS_LABEL[task.status] ?? task.status.toUpperCase();
  const isRemoved       = task.status === 'removed';

  const assignedDate  = new Date(task.created_at).toLocaleDateString('en-PH', { month: 'short', day: 'numeric', year: 'numeric' });
  const respondedDate = task.responded_at
    ? new Date(task.responded_at).toLocaleDateString('en-PH', { month: 'short', day: 'numeric' })
    : null;

  return (
    <TouchableOpacity
      style={[s.card, isRemoved && s.cardRemoved]}
      onPress={onPress}
      activeOpacity={0.75}
    >
      <View style={s.cardTopRow}>
        <Text style={[s.cardTitle, isRemoved && { color: colors.textMuted }]} numberOfLines={2}>
          {incident.title}
        </Text>
        <View style={[s.taskStatusChip, { backgroundColor: `${taskStatusColor}22`, borderColor: taskStatusColor }]}>
          <Text style={[s.taskStatusText, { color: taskStatusColor }]}>
            {taskStatusLabel}
          </Text>
        </View>
      </View>

      <View style={s.cardMetaRow}>
        <View style={[s.densityChip, { backgroundColor: densityBg, borderColor: densityColor }]}>
          <Text style={[s.densityText, { color: densityColor }]}>{incident.density_tag}</Text>
        </View>
        <View style={[s.incidentStatusChip, { borderColor: statusColor }]}>
          <Text style={[s.incidentStatusText, { color: statusColor }]}>{incident.incident_status}</Text>
        </View>
        {incident.people_count > 0 && (
          <Text style={s.metaChip}>👥 {incident.people_count}</Text>
        )}
      </View>

      {incident.location && (
        <Text style={s.cardLocation} numberOfLines={1}>📍 {incident.location}</Text>
      )}

      <View style={s.cardFooter}>
        <Text style={s.footerText}>
          {t('tasks.assignedBy')} <Text style={s.footerName}>{task.assigned_by}</Text>
          {' · '}{assignedDate}
        </Text>
        {respondedDate && (
          <Text style={s.footerText}>{t('tasks.responded')} {respondedDate}</Text>
        )}
      </View>

      {/* Decline reason */}
      {task.status === 'declined' && task.decline_reason && (
        <View style={s.declineReasonBox}>
          <Text style={s.declineReasonLabel}>{t('tasks.yourReason')}</Text>
          <Text style={s.declineReasonText}>"{task.decline_reason}"</Text>
        </View>
      )}

      {/* Removed by operator notice */}
      {isRemoved && (
        <View style={s.removedBox}>
          <Text style={s.removedIcon}>⚠️</Text>
          <Text style={s.removedText}>
            You were removed from this incident by the operator.
          </Text>
        </View>
      )}

      {/* Left notice */}
      {task.status === 'left' && (
        <View style={s.removedBox}>
          <Text style={s.removedIcon}>🚪</Text>
          <Text style={s.removedText}>You left this incident.</Text>
        </View>
      )}

      {/* Remove button — only shown on active accepted tasks */}
      {onLeave && (
        <TouchableOpacity
          style={s.leaveBtn}
          onPress={onLeave}
          activeOpacity={0.8}
        >
          <Text style={s.leaveBtnText}>Remove</Text>
        </TouchableOpacity>
      )}

      <Text style={s.tapHint}>{t('tasks.tapHint')}</Text>
    </TouchableOpacity>
  );
}

function EmptyState({ message, s }: { message: string; s: ReturnType<typeof makeStyles> }) {
  return (
    <View style={s.emptyState}>
      <Text style={s.emptyText}>{message}</Text>
    </View>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    screen: { flex: 1, backgroundColor: c.background },

    centered: {
      flex: 1, alignItems: 'center', justifyContent: 'center',
      backgroundColor: c.background, gap: 12, padding: 24,
    },
    loadingText: { color: c.textMuted, fontSize: 13 },
    errorText:   { color: '#ef4444', fontSize: 13, textAlign: 'center' },
    retryBtn: {
      backgroundColor: c.surface, borderRadius: 10,
      paddingHorizontal: 24, paddingVertical: 10,
      borderWidth: 1, borderColor: c.border,
    },
    retryText: { color: c.accent, fontSize: 14, fontWeight: '700' },

    header: {
      paddingHorizontal: 18, paddingBottom: 12,
      borderBottomWidth: 1, borderBottomColor: c.borderLight,
    },
    headerTitle: { fontSize: 22, fontWeight: '800', color: c.text },
    headerSub:   { fontSize: 12, color: c.textMuted, marginTop: 2 },

    sectionHeader: {
      flexDirection: 'row', alignItems: 'center', gap: 8,
      paddingHorizontal: 18, paddingTop: 20, paddingBottom: 10,
    },
    sectionTitle: { fontSize: 14, fontWeight: '700', color: c.textSec },
    countBadge: {
      backgroundColor: c.surface, borderRadius: 10,
      paddingHorizontal: 7, paddingVertical: 2,
      borderWidth: 1, borderColor: c.border,
    },
    countBadgeText: { fontSize: 11, color: c.textMuted, fontWeight: '700' },

    emptyState: {
      marginHorizontal: 18, paddingVertical: 20,
      backgroundColor: c.surface, borderRadius: 12,
      alignItems: 'center', borderWidth: 1, borderColor: c.border,
      borderStyle: 'dashed',
    },
    emptyText: { fontSize: 13, color: c.textMuted },

    card: {
      marginHorizontal: 16, marginBottom: 12,
      backgroundColor: c.surface, borderRadius: 14,
      padding: 16, borderWidth: 1, borderColor: c.border, gap: 8,
    },
    cardTopRow: {
      flexDirection: 'row', alignItems: 'flex-start',
      justifyContent: 'space-between', gap: 10,
    },
    cardTitle:  { flex: 1, fontSize: 15, fontWeight: '700', color: c.text },
    taskStatusChip: {
      borderRadius: 6, borderWidth: 1,
      paddingHorizontal: 7, paddingVertical: 3, alignSelf: 'flex-start',
    },
    taskStatusText: { fontSize: 9, fontWeight: '800', letterSpacing: 0.5 },

    cardMetaRow:  { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
    densityChip:  { borderRadius: 6, borderWidth: 1, paddingHorizontal: 8, paddingVertical: 3 },
    densityText:  { fontSize: 10, fontWeight: '700' },
    incidentStatusChip: {
      borderRadius: 6, borderWidth: 1, borderColor: c.textMuted,
      paddingHorizontal: 8, paddingVertical: 3,
    },
    incidentStatusText: { fontSize: 10, fontWeight: '700' },
    metaChip: { fontSize: 11, color: c.textMuted },

    cardLocation: { fontSize: 12, color: c.textMuted },

    cardFooter: { borderTopWidth: 1, borderTopColor: c.borderLight, paddingTop: 8, gap: 2 },
    footerText: { fontSize: 11, color: c.textMuted },
    footerName: { color: c.textSec, fontWeight: '700' },

    declineReasonBox: {
      backgroundColor: 'rgba(239,68,68,0.08)', borderRadius: 8, padding: 10,
      borderWidth: 1, borderColor: 'rgba(239,68,68,0.2)', gap: 3,
    },
    declineReasonLabel: { fontSize: 10, color: '#ef4444', fontWeight: '700' },
    declineReasonText:  { fontSize: 12, color: c.textSec, fontStyle: 'italic' },

    // ── Removed-by-operator card styles ──────────────────────────────────────
    cardRemoved: {
      opacity: 0.7,
      borderLeftColor: '#475569',
      borderColor: c.borderLight,
    },
    removedBox: {
      flexDirection: 'row', alignItems: 'flex-start', gap: 8,
      backgroundColor: 'rgba(71,85,105,0.12)', borderRadius: 8, padding: 10,
      borderWidth: 1, borderColor: 'rgba(71,85,105,0.25)',
    },
    removedIcon: { fontSize: 13 },
    removedText: { flex: 1, fontSize: 12, color: c.textMuted, lineHeight: 17 },

    // ── Leave button ──────────────────────────────────────────────────────────
    leaveBtn: {
      marginTop: 10,
      paddingVertical: 10,
      borderRadius: 10,
      alignItems: 'center',
      borderWidth: 1,
      borderColor: 'rgba(239,68,68,0.35)',
      backgroundColor: 'rgba(239,68,68,0.07)',
    },
    leaveBtnText: {
      fontSize: 13, fontWeight: '700', color: '#ef4444',
    },

    tapHint: { fontSize: 10, color: c.border, textAlign: 'right' },
  });
}
