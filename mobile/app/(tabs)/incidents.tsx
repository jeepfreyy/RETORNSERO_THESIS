import React, { useEffect, useMemo, useState } from 'react';
import {
  View, Text, FlatList, TouchableOpacity,
  StyleSheet, RefreshControl, ActivityIndicator,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useIncidents } from '../../hooks/useIncidents';
import { IncidentCard } from '../../components/IncidentCard';
import { Incident } from '../../lib/api';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

type Filter = 'ALL' | 'OPEN' | 'RESPONDING' | 'RESOLVED';

export default function IncidentsScreen() {
  const { incidents, loading, error, refresh } = useIncidents();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s      = useMemo(() => makeStyles(colors), [colors]);
  const insets = useSafeAreaInsets();

  const [filter, setFilter] = useState<Filter>('ALL');

  useEffect(() => { refresh(); }, []);

  const FILTERS: { key: Filter; label: string }[] = [
    { key: 'ALL',        label: t('inc.all') },
    { key: 'OPEN',       label: t('inc.open') },
    { key: 'RESPONDING', label: t('inc.responding') },
    { key: 'RESOLVED',   label: t('inc.resolved') },
  ];

  const filtered: Incident[] =
    filter === 'ALL' ? incidents : incidents.filter((i) => i.incident_status === filter);

  return (
    <View style={s.screen}>
      <View style={[s.header, { paddingTop: insets.top + 16 }]}>
        <Text style={s.title}>{t('inc.title')}</Text>
        <Text style={s.subtitle}>{t('inc.total', { n: incidents.length })}</Text>
      </View>

      <View style={s.filterRow}>
        {FILTERS.map(({ key, label }) => (
          <TouchableOpacity
            key={key}
            style={[s.filterBtn, filter === key && s.filterBtnActive]}
            onPress={() => setFilter(key)}
          >
            <Text style={[s.filterText, filter === key && s.filterTextActive]}>
              {label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {loading && incidents.length === 0 ? (
        <ActivityIndicator color={colors.accent} style={{ marginTop: 40 }} />
      ) : error ? (
        <View style={s.errorBox}>
          <Text style={s.errorText}>⚠️ {error}</Text>
          <TouchableOpacity onPress={refresh} style={s.retryBtn}>
            <Text style={s.retryText}>{t('inc.retry')}</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={filtered}
          keyExtractor={(item) => String(item.id)}
          renderItem={({ item }) => <IncidentCard incident={item} />}
          contentContainerStyle={s.list}
          refreshControl={<RefreshControl refreshing={loading} onRefresh={refresh} tintColor={colors.accent} />}
          ListEmptyComponent={
            <View style={s.emptyBox}>
              <Text style={s.emptyText}>{t('inc.empty')}</Text>
            </View>
          }
        />
      )}
    </View>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    screen:   { flex: 1, backgroundColor: c.background },
    header:   { paddingHorizontal: 20, paddingBottom: 12 },
    title:    { fontSize: 24, fontWeight: '800', color: c.text },
    subtitle: { fontSize: 12, color: c.textMuted, marginTop: 2 },

    filterRow: { flexDirection: 'row', paddingHorizontal: 16, gap: 8, marginBottom: 12 },
    filterBtn: {
      flex: 1, paddingVertical: 7, borderRadius: 8,
      backgroundColor: c.surface, alignItems: 'center',
      borderWidth: 1, borderColor: c.borderLight,
    },
    filterBtnActive: {
      backgroundColor: c.accentSoft,
      borderColor: c.accent,
    },
    filterText:       { fontSize: 11, fontWeight: '600', color: c.textHint },
    filterTextActive: { color: c.accent, fontWeight: '700' },

    list:     { paddingHorizontal: 16, paddingBottom: 40 },
    errorBox: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
    errorText:{ color: '#ef4444', fontSize: 14, textAlign: 'center', marginBottom: 16 },
    retryBtn: { backgroundColor: c.surface, borderRadius: 8, paddingHorizontal: 20, paddingVertical: 10 },
    retryText:{ color: c.accent, fontWeight: '600' },
    emptyBox: { padding: 40, alignItems: 'center' },
    emptyText:{ color: c.textMuted, fontSize: 13 },
  });
}
