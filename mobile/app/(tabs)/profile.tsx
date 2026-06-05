import React, { useMemo, useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  Alert, ActivityIndicator, ScrollView, Modal,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAppStore } from '../../store/useAppStore';
import { OnboardingModal } from '../../components/OnboardingModal';
import { apiLogout } from '../../lib/api';
import { BASE_URL, SERVER_IP, SERVER_PORT } from '../../lib/constants';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

const ABOUT_TEXT =
  'Barangay Sentinel is a real-time crowd monitoring and incident response system ' +
  'designed for barangay security operations. It uses CCTV-based computer vision to ' +
  'detect crowd density levels and automatically alerts operators to potential threats. ' +
  'Field responders receive assignments through this mobile app and can accept, decline, ' +
  'and track incidents in real time — keeping the entire team coordinated and informed.';

export default function ProfileScreen() {
  const { username, role, token, clearAuth, theme, setTheme, lang, setLang } =
    useAppStore();
  const { colors, isDark } = useTheme();
  const { t } = useTranslation();
  const s      = useMemo(() => makeStyles(colors), [colors]);
  const insets = useSafeAreaInsets();

  const [loading,       setLoading]       = useState(false);
  const [showGuide,     setShowGuide]     = useState(false);
  const [showAbout,     setShowAbout]     = useState(false);

  async function handleLogout() {
    Alert.alert(
      t('profile.signOut'),
      t('profile.signOutQ'),
      [
        { text: t('profile.cancel'), style: 'cancel' },
        {
          text: t('profile.signOut'),
          style: 'destructive',
          onPress: async () => {
            setLoading(true);
            try {
              if (token) await apiLogout(token);
            } catch {}
            await clearAuth();
            setLoading(false);
          },
        },
      ],
    );
  }

  // ── Segmented-control helpers (need live colors, can't be in StyleSheet) ──
  const segActive = {
    backgroundColor: colors.accentSoft,
    borderRadius: 9,
    margin: 2,
  };
  const segActiveText = { color: colors.accent, fontWeight: '700' as const };

  return (
    <ScrollView
      style={s.screen}
      contentContainerStyle={s.content}
      showsVerticalScrollIndicator={false}
    >
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <View style={[s.header, { paddingTop: insets.top + 16 }]}>
        <Text style={s.title}>{t('profile.title')}</Text>
      </View>

      {/* ── Avatar ────────────────────────────────────────────────────────── */}
      <View style={s.avatarSection}>
        <View style={s.avatar}>
          <Text style={s.avatarText}>
            {(username ?? 'U').charAt(0).toUpperCase()}
          </Text>
        </View>
        <Text style={s.username}>{username ?? 'Unknown'}</Text>
        <View style={s.roleBadge}>
          <Text style={s.roleText}>{role ?? 'Tanod'}</Text>
        </View>
      </View>

      {/* ── App Settings ──────────────────────────────────────────────────── */}
      <View style={s.card}>
        <Text style={s.cardTitle}>{t('profile.settings')}</Text>

        {/* Language */}
        <View style={s.settingRow}>
          <View style={s.settingLeft}>
            <Ionicons name="language" size={18} color={colors.accent} style={{ marginRight: 8 }} />
            <Text style={s.settingLabel}>{t('profile.language')}</Text>
          </View>
          <View style={s.segmented}>
            <TouchableOpacity
              style={[s.segBtn, lang === 'en' && segActive]}
              onPress={() => setLang('en')}
              activeOpacity={0.8}
            >
              <Text style={[s.segText, lang === 'en' && segActiveText]}>English</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[s.segBtn, lang === 'tl' && segActive]}
              onPress={() => setLang('tl')}
              activeOpacity={0.8}
            >
              <Text style={[s.segText, lang === 'tl' && segActiveText]}>Tagalog</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Theme */}
        <View style={s.settingRow}>
          <View style={s.settingLeft}>
            <Ionicons
              name={isDark ? 'moon' : 'sunny'}
              size={18}
              color={colors.accent}
              style={{ marginRight: 8 }}
            />
            <Text style={s.settingLabel}>{t('profile.theme')}</Text>
          </View>
          <View style={s.segmented}>
            <TouchableOpacity
              style={[s.segBtn, theme === 'dark' && segActive]}
              onPress={() => setTheme('dark')}
              activeOpacity={0.8}
            >
              <Ionicons name="moon" size={12} color={theme === 'dark' ? colors.accent : colors.textHint} style={{ marginRight: 3 }} />
              <Text style={[s.segText, theme === 'dark' && segActiveText]}>{t('profile.dark')}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[s.segBtn, theme === 'light' && segActive]}
              onPress={() => setTheme('light')}
              activeOpacity={0.8}
            >
              <Ionicons name="sunny" size={12} color={theme === 'light' ? colors.accent : colors.textHint} style={{ marginRight: 3 }} />
              <Text style={[s.segText, theme === 'light' && segActiveText]}>{t('profile.light')}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      {/* ── Server info ───────────────────────────────────────────────────── */}
      <View style={s.card}>
        <Text style={s.cardTitle}>{t('profile.serverConn')}</Text>
        <View style={s.row}>
          <Text style={s.rowLabel}>{t('profile.address')}</Text>
          <Text style={s.rowValue}>{BASE_URL}</Text>
        </View>
        <View style={s.row}>
          <Text style={s.rowLabel}>{t('profile.ip')}</Text>
          <Text style={s.rowValue}>{SERVER_IP}</Text>
        </View>
        <View style={s.row}>
          <Text style={s.rowLabel}>{t('profile.port')}</Text>
          <Text style={s.rowValue}>{SERVER_PORT}</Text>
        </View>
      </View>

      {/* ── App info ──────────────────────────────────────────────────────── */}
      <View style={s.card}>
        <Text style={s.cardTitle}>{t('profile.appInfo')}</Text>
        <View style={s.row}>
          <Text style={s.rowLabel}>{t('profile.version')}</Text>
          <Text style={s.rowValue}>1.0.0</Text>
        </View>
        <View style={s.row}>
          <Text style={s.rowLabel}>{t('profile.system')}</Text>
          <Text style={s.rowValue}>Barangay Sentinel</Text>
        </View>
      </View>

      {/* ── Help & Guide ──────────────────────────────────────────────────── */}
      <View style={s.card}>
        <Text style={s.cardTitle}>HELP</Text>

        <TouchableOpacity style={s.helpRow} onPress={() => setShowGuide(true)} activeOpacity={0.75}>
          <View style={[s.helpIcon, { backgroundColor: `${colors.accent}18` }]}>
            <Ionicons name="book-outline" size={18} color={colors.accent} />
          </View>
          <View style={s.helpInfo}>
            <Text style={s.helpTitle}>User Guide</Text>
            <Text style={s.helpSub}>Step-by-step walkthrough of all features</Text>
          </View>
          <Ionicons name="chevron-forward" size={16} color={colors.textHint} />
        </TouchableOpacity>

        <TouchableOpacity style={s.helpRow} onPress={() => setShowAbout(true)} activeOpacity={0.75}>
          <View style={[s.helpIcon, { backgroundColor: 'rgba(99,102,241,0.15)' }]}>
            <Ionicons name="information-circle-outline" size={18} color="#6366f1" />
          </View>
          <View style={s.helpInfo}>
            <Text style={s.helpTitle}>About Barangay Sentinel</Text>
            <Text style={s.helpSub}>System description and purpose</Text>
          </View>
          <Ionicons name="chevron-forward" size={16} color={colors.textHint} />
        </TouchableOpacity>
      </View>

      {/* ── Logout ────────────────────────────────────────────────────────── */}
      <TouchableOpacity
        style={[s.logoutBtn, loading && { opacity: 0.6 }]}
        onPress={handleLogout}
        disabled={loading}
      >
        {loading
          ? <ActivityIndicator color="#ef4444" />
          : <Text style={s.logoutText}>{t('profile.signOut')}</Text>
        }
      </TouchableOpacity>

      <View style={{ height: 32 }} />

      {/* ── Guide replay modal ────────────────────────────────────────────── */}
      <OnboardingModal
        visible={showGuide}
        onComplete={() => setShowGuide(false)}
        isReplay
      />

      {/* ── About modal ───────────────────────────────────────────────────── */}
      <Modal visible={showAbout} transparent animationType="fade" statusBarTranslucent>
        <View style={s.aboutBackdrop}>
          <View style={s.aboutCard}>
            {/* Icon */}
            <View style={s.aboutIconWrap}>
              <View style={s.aboutIconRing}>
                <Ionicons name="videocam" size={32} color={colors.accent} />
              </View>
            </View>
            <Text style={s.aboutAppName}>BARANGAY SENTINEL</Text>
            <Text style={[s.aboutVersion, { color: colors.textMuted }]}>Version 1.0.0  ·  Mobile Field App</Text>
            <View style={s.aboutDivider} />
            <ScrollView style={{ maxHeight: 180 }} showsVerticalScrollIndicator={false}>
              <Text style={s.aboutBody}>{ABOUT_TEXT}</Text>
            </ScrollView>
            <View style={s.aboutDivider} />
            <View style={s.aboutMeta}>
              <Text style={s.aboutMetaLabel}>System</Text>
              <Text style={s.aboutMetaValue}>Barangay Sentinel CCTV</Text>
            </View>
            <View style={s.aboutMeta}>
              <Text style={s.aboutMetaLabel}>Developed for</Text>
              <Text style={s.aboutMetaValue}>Barangay Security Operations</Text>
            </View>
            <TouchableOpacity style={[s.aboutCloseBtn, { backgroundColor: colors.accent }]} onPress={() => setShowAbout(false)} activeOpacity={0.85}>
              <Text style={s.aboutCloseBtnText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

    </ScrollView>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    screen:  { flex: 1, backgroundColor: c.background },
    content: { paddingHorizontal: 20 },

    header:      { paddingBottom: 16 },
    title:       { fontSize: 24, fontWeight: '800', color: c.text },

    avatarSection: { alignItems: 'center', paddingVertical: 28 },
    avatar: {
      width: 80, height: 80, borderRadius: 40,
      backgroundColor: `${c.accent}22`,
      borderWidth: 2, borderColor: `${c.accent}80`,
      alignItems: 'center', justifyContent: 'center', marginBottom: 12,
    },
    avatarText:  { fontSize: 32, fontWeight: '800', color: c.accent },
    username:    { fontSize: 20, fontWeight: '700', color: c.text, marginBottom: 8 },
    roleBadge: {
      backgroundColor: `${c.accent}18`,
      borderWidth: 1, borderColor: `${c.accent}50`,
      borderRadius: 20, paddingHorizontal: 14, paddingVertical: 4,
    },
    roleText: { fontSize: 12, color: c.accent, fontWeight: '600' },

    card: {
      backgroundColor: c.surface,
      borderRadius: 14, padding: 16, marginBottom: 12, gap: 14,
      borderWidth: 1, borderColor: c.borderLight,
    },
    cardTitle: {
      fontSize: 11, fontWeight: '700', color: c.textMuted,
      textTransform: 'uppercase', letterSpacing: 1, marginBottom: 2,
    },

    settingRow:  { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
    settingLeft: { flexDirection: 'row', alignItems: 'center' },
    settingLabel:{ fontSize: 14, color: c.text, fontWeight: '600' },

    // Segmented control pill container
    segmented: {
      flexDirection: 'row',
      backgroundColor: c.background,
      borderRadius: 10, borderWidth: 1, borderColor: c.border,
      overflow: 'hidden',
    },
    // Each button inside the segmented control
    segBtn: {
      flexDirection: 'row', alignItems: 'center',
      paddingHorizontal: 12, paddingVertical: 7,
    },
    segText: { fontSize: 12, fontWeight: '600', color: c.textHint },

    row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    rowLabel: { fontSize: 13, color: c.textMuted },
    rowValue: {
      fontSize: 13, color: c.textSec, fontWeight: '600',
      flexShrink: 1, textAlign: 'right', marginLeft: 12,
    },

    logoutBtn: {
      borderWidth: 1, borderColor: 'rgba(239,68,68,0.4)',
      backgroundColor: 'rgba(239,68,68,0.08)',
      borderRadius: 12, paddingVertical: 14,
      alignItems: 'center', marginTop: 8,
    },
    logoutText: { color: '#ef4444', fontSize: 15, fontWeight: '700' },

    // ── Help card rows ──────────────────────────────────────────────────────
    helpRow: {
      flexDirection: 'row', alignItems: 'center', gap: 12,
      paddingVertical: 10,
      borderBottomWidth: 1, borderBottomColor: c.borderLight,
    },
    helpIcon: {
      width: 36, height: 36, borderRadius: 10,
      alignItems: 'center', justifyContent: 'center',
    },
    helpInfo:  { flex: 1 },
    helpTitle: { fontSize: 14, fontWeight: '700', color: c.text },
    helpSub:   { fontSize: 11, color: c.textMuted, marginTop: 1 },

    // ── About modal ─────────────────────────────────────────────────────────
    aboutBackdrop: {
      flex: 1, backgroundColor: 'rgba(0,0,0,0.82)',
      justifyContent: 'center', alignItems: 'center', padding: 24,
    },
    aboutCard: {
      width: '100%', backgroundColor: c.surface,
      borderRadius: 22, borderWidth: 1, borderColor: `${c.accent}40`,
      padding: 24, alignItems: 'center', gap: 10,
    },
    aboutIconWrap: {
      width: 80, height: 80, borderRadius: 40,
      backgroundColor: `${c.accent}10`, borderWidth: 1,
      borderColor: `${c.accent}30`,
      alignItems: 'center', justifyContent: 'center',
    },
    aboutIconRing: {
      width: 62, height: 62, borderRadius: 31,
      backgroundColor: `${c.accent}18`, borderWidth: 2,
      borderColor: `${c.accent}60`,
      alignItems: 'center', justifyContent: 'center',
    },
    aboutAppName: {
      fontSize: 18, fontWeight: '800', color: c.accent,
      letterSpacing: 2, textAlign: 'center',
    },
    aboutVersion: { fontSize: 11, letterSpacing: 0.5 },
    aboutDivider: { width: '100%', height: 1, backgroundColor: c.borderLight, marginVertical: 2 },
    aboutBody:    { fontSize: 13, color: c.textSec, lineHeight: 21, textAlign: 'center' },
    aboutMeta: {
      flexDirection: 'row', justifyContent: 'space-between',
      width: '100%',
    },
    aboutMetaLabel: { fontSize: 12, color: c.textMuted },
    aboutMetaValue: { fontSize: 12, color: c.text, fontWeight: '600' },
    aboutCloseBtn: {
      width: '100%', paddingVertical: 13,
      borderRadius: 12, alignItems: 'center', marginTop: 4,
    },
    aboutCloseBtnText: { color: '#fff', fontSize: 14, fontWeight: '800' },
  });
}
