import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  View, Text, StyleSheet, TouchableOpacity,
  ActivityIndicator, Image,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useStats } from '../../hooks/useStats';
import { DensityBadge } from '../../components/DensityBadge';
import { BASE_URL } from '../../lib/constants';
import { useAppStore } from '../../store/useAppStore';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

const POLL_MS      = 200;
const SLOW_POLL_MS = 2000;
const MAX_ERRORS   = 5;

export default function LiveFeedScreen() {
  const { token }  = useAppStore();
  const { stats, refetch: refetchStats } = useStats();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s      = useMemo(() => makeStyles(colors), [colors]);
  const insets = useSafeAreaInsets();

  const [frameUri,    setFrameUri]    = useState('');
  const [hasFrame,    setHasFrame]    = useState(false);
  const [errorStreak, setErrorStreak] = useState(0);
  const [isSlowMode,  setIsSlowMode]  = useState(false);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef  = useRef(true);

  const tick = useCallback(() => {
    if (!mountedRef.current) return;
    setFrameUri(`${BASE_URL}/cam1_frame?t=${Date.now()}`);
  }, []);

  const startFast = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsSlowMode(false);
    tick();
    intervalRef.current = setInterval(tick, POLL_MS);
  }, [tick]);

  const startSlow = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsSlowMode(true);
    tick();
    intervalRef.current = setInterval(tick, SLOW_POLL_MS);
  }, [tick]);

  useEffect(() => {
    mountedRef.current = true;
    if (token) startFast();
    return () => {
      mountedRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [token]);

  useEffect(() => {
    if (errorStreak >= MAX_ERRORS && !isSlowMode) startSlow();
  }, [errorStreak, isSlowMode, startSlow]);

  function onFrameLoad()  {
    if (!mountedRef.current) return;
    setHasFrame(true);
    setErrorStreak(0);
    if (isSlowMode) startFast();
  }
  function onFrameError() {
    if (!mountedRef.current) return;
    setErrorStreak((n) => n + 1);
  }
  function onReload() {
    setHasFrame(false);
    setErrorStreak(0);
    refetchStats();
    startFast();
  }

  const showError   = errorStreak >= MAX_ERRORS;
  const isWarmingUp = stats?.is_warming_up ?? false;
  const density     = stats?.density ?? '—';
  const count       = stats?.count   ?? 0;

  return (
    <View style={s.screen}>
      {/* Header */}
      <View style={[s.header, { paddingTop: insets.top + 12 }]}>
        <View style={s.liveRow}>
          <View style={[s.liveDot, showError && s.liveDotOff]} />
          <Text style={[s.liveLabel, showError && s.liveLabelOff]}>
            {showError ? t('live.offline') : t('live.live')}
          </Text>
        </View>

        <View style={s.headerRight}>
          {stats && !isWarmingUp && <DensityBadge density={density} size="sm" />}
          {isWarmingUp && <Text style={s.warmupChip}>{t('live.warmup')}</Text>}
          {count > 0 && (
            <View style={s.countChip}>
              <Text style={s.countText}>{count} {count === 1 ? 'person' : 'people'}</Text>
            </View>
          )}
          <TouchableOpacity onPress={onReload} style={s.reloadBtn} activeOpacity={0.7}>
            <Text style={s.reloadIcon}>↻</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Feed */}
      <View style={s.feedArea}>
        {token && frameUri !== '' && (
          <Image
            key={token}
            source={{
              uri: frameUri,
              headers: { Authorization: `Bearer ${token}` },
              cache: 'reload' as any,
            }}
            style={[s.frameImage, !hasFrame && { opacity: 0 }]}
            resizeMode="contain"
            onLoad={onFrameLoad}
            onError={onFrameError}
            fadeDuration={0}
          />
        )}

        {!hasFrame && !showError && (
          <View style={s.overlay}>
            <ActivityIndicator color={colors.accent} size="large" />
            <Text style={s.overlayTitle}>
              {isWarmingUp ? t('live.warming') : t('live.connecting')}
            </Text>
            {isWarmingUp && <Text style={s.overlaySubtitle}>{t('live.warmSub')}</Text>}
          </View>
        )}

        {showError && (
          <View style={s.overlay}>
            <Text style={s.errorEmoji}>📷</Text>
            <Text style={s.errorTitle}>{t('live.unavailable')}</Text>
            <Text style={s.errorSubtitle}>
              {isWarmingUp ? t('live.errWarm') : t('live.errConn')}
            </Text>
            <TouchableOpacity style={s.retryBtn} onPress={onReload} activeOpacity={0.8}>
              <Text style={s.retryText}>{t('live.retry')}</Text>
            </TouchableOpacity>
          </View>
        )}
      </View>

    </View>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    screen: { flex: 1, backgroundColor: c.background },

    header: {
      paddingHorizontal: 16, paddingBottom: 10,
      flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    },
    liveRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
    liveDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#ef4444' },
    liveDotOff: { backgroundColor: c.textHint },
    liveLabel:    { fontSize: 12, fontWeight: '800', color: '#ef4444', letterSpacing: 1.5 },
    liveLabelOff: { color: c.textHint },

    headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    warmupChip: {
      fontSize: 11, fontWeight: '600', color: '#f59e0b',
      backgroundColor: 'rgba(245,158,11,0.12)',
      paddingHorizontal: 8, paddingVertical: 3,
      borderRadius: 6, borderWidth: 1, borderColor: 'rgba(245,158,11,0.3)',
    },
    countChip: {
      backgroundColor: c.surface, borderRadius: 8,
      paddingHorizontal: 10, paddingVertical: 4,
      borderWidth: 1, borderColor: c.border,
    },
    countText: { fontSize: 12, color: c.textSec, fontWeight: '600' },
    reloadBtn: {
      backgroundColor: c.surface, borderRadius: 8,
      paddingHorizontal: 11, paddingVertical: 4,
      borderWidth: 1, borderColor: c.border,
    },
    reloadIcon: { fontSize: 17, color: c.textMuted },

    feedArea: {
      flex: 1, backgroundColor: '#020617',
      position: 'relative', overflow: 'hidden',
    },
    frameImage: {
      position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
      width: '100%', height: '100%',
    },

    overlay: {
      ...StyleSheet.absoluteFillObject,
      backgroundColor: '#020617',
      alignItems: 'center', justifyContent: 'center',
      paddingHorizontal: 32, gap: 12, zIndex: 10,
    },
    overlayTitle:    { fontSize: 14, color: '#64748b', textAlign: 'center', fontWeight: '600' },
    overlaySubtitle: { fontSize: 12, color: '#334155', textAlign: 'center', lineHeight: 18 },

    errorEmoji:    { fontSize: 36, marginBottom: 4 },
    errorTitle:    { fontSize: 16, fontWeight: '700', color: '#ef4444', textAlign: 'center' },
    errorSubtitle: { fontSize: 12, color: '#64748b', textAlign: 'center', lineHeight: 20 },
    retryBtn: {
      marginTop: 8, backgroundColor: c.surface,
      borderRadius: 10, paddingHorizontal: 28, paddingVertical: 11,
      borderWidth: 1, borderColor: c.border,
    },
    retryText: { color: c.accent, fontSize: 14, fontWeight: '700' },

  });
}
