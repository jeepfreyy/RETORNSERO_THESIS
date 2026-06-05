import React, { useEffect, useMemo } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, RefreshControl, ActivityIndicator,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { useStats } from '../../hooks/useStats';
import { useIncidents } from '../../hooks/useIncidents';
import { DensityBadge } from '../../components/DensityBadge';
import { IncidentCard } from '../../components/IncidentCard';
import { useAppStore } from '../../store/useAppStore';
import { DENSITY_COLORS, DENSITY_BG } from '../../lib/constants';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

export default function HomeScreen() {
  const { username }   = useAppStore();
  const { stats, loading: sLoading, refetch: refetchStats } = useStats();
  const { incidents, loading: iLoading, refresh } = useIncidents();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s = useMemo(() => makeStyles(colors), [colors]);
  const router  = useRouter();
  const insets  = useSafeAreaInsets();

  useEffect(() => { refresh(); }, []);

  const activeIncidents = incidents.filter(
    (i) => i.incident_status === 'OPEN' || i.incident_status === 'RESPONDING',
  );
  const density      = stats?.density ?? '—';
  const densityColor = DENSITY_COLORS[density] ?? '#94a3b8';
  const densityBg    = DENSITY_BG[density]    ?? 'rgba(148,163,184,0.1)';
  const isRefreshing = sLoading || iLoading;

  function onRefresh() { refetchStats(); refresh(); }

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={[s.content, { paddingTop: insets.top + 16 }]}
      refreshControl={<RefreshControl refreshing={isRefreshing} onRefresh={onRefresh} tintColor={colors.accent} />}
    >
      {/* Header */}
      <View style={s.header}>
        <Text style={s.greeting}>{t('home.greeting', { name: username ?? 'Officer' })}</Text>
        <Text style={s.subtitle}>{t('home.subtitle')}</Text>
      </View>

      {/* Density Status Card */}
      <View style={[s.densityCard, { backgroundColor: densityBg, borderColor: densityColor }]}>
        <Text style={s.densityLabel}>{t('home.density')}</Text>
        {sLoading && !stats
          ? <ActivityIndicator color={colors.accent} style={{ marginVertical: 8 }} />
          : (
            <>
              <Text style={[s.densityValue, { color: densityColor }]}>{density}</Text>
              {stats && (
                <Text style={s.densityMeta}>
                  {stats.count === 1
                    ? t('home.densityMeta1', { count: stats.count })
                    : t('home.densityMeta',  { count: stats.count })}
                  {stats.is_warming_up ? t('home.warmingUp') : ''}
                  {'  ·  '}{Math.round(stats.fps)} fps
                </Text>
              )}
            </>
          )
        }
      </View>

      {/* Active Incidents */}
      <View style={s.section}>
        <View style={s.sectionHeader}>
          <Text style={s.sectionTitle}>
            {t('home.activeInc')}
            {activeIncidents.length > 0 && (
              <Text style={s.badge}>  {activeIncidents.length}</Text>
            )}
          </Text>
          <TouchableOpacity onPress={() => router.push('/(tabs)/incidents')}>
            <Text style={s.seeAll}>{t('home.seeAll')}</Text>
          </TouchableOpacity>
        </View>

        {iLoading && activeIncidents.length === 0 && (
          <ActivityIndicator color={colors.accent} style={{ marginTop: 12 }} />
        )}

        {!iLoading && activeIncidents.length === 0 && (
          <View style={s.emptyBox}>
            <Text style={s.emptyText}>{t('home.noActive')}</Text>
          </View>
        )}

        {activeIncidents.slice(0, 3).map((inc) => (
          <IncidentCard key={inc.id} incident={inc} />
        ))}

        {activeIncidents.length > 3 && (
          <TouchableOpacity style={s.moreBtn} onPress={() => router.push('/(tabs)/incidents')}>
            <Text style={s.moreBtnText}>{t('home.moreInc', { n: activeIncidents.length - 3 })}</Text>
          </TouchableOpacity>
        )}
      </View>
    </ScrollView>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    screen:   { flex: 1, backgroundColor: c.background },
    content:  { paddingHorizontal: 20, paddingBottom: 40 },
    header:   { marginBottom: 20 },
    greeting: { fontSize: 22, fontWeight: '800', color: c.text, marginBottom: 2 },
    subtitle: { fontSize: 12, color: c.textMuted },

    densityCard: {
      borderRadius: 16, borderWidth: 1, padding: 20,
      marginBottom: 24, alignItems: 'center',
    },
    densityLabel: {
      fontSize: 11, fontWeight: '600', color: c.textSec,
      textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8,
    },
    densityValue: { fontSize: 44, fontWeight: '900', letterSpacing: 2, marginBottom: 6 },
    densityMeta:  { fontSize: 12, color: c.textMuted, textAlign: 'center' },

    section:       { marginBottom: 20 },
    sectionHeader: {
      flexDirection: 'row', justifyContent: 'space-between',
      alignItems: 'center', marginBottom: 12,
    },
    sectionTitle: { fontSize: 16, fontWeight: '700', color: c.text },
    badge:        { fontSize: 14, fontWeight: '800', color: '#ef4444' },
    seeAll:       { fontSize: 13, color: c.accent, fontWeight: '600' },

    emptyBox: { backgroundColor: c.surface, borderRadius: 12, padding: 20, alignItems: 'center' },
    emptyText:{ fontSize: 13, color: c.textMuted },

    moreBtn:     { backgroundColor: c.surface, borderRadius: 10, padding: 12, alignItems: 'center', marginTop: 4 },
    moreBtnText: { color: c.accent, fontSize: 13, fontWeight: '600' },
  });
}
