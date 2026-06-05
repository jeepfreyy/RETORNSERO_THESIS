import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, Alert, ActivityIndicator, Image,
  TextInput, Modal,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { Video, ResizeMode, AVPlaybackStatus } from 'expo-av';
import {
  apiIncidents, apiAssignSelf, apiUpdateStatus, apiRemoveResponder,
  Incident,
} from '../../lib/api';
import { DensityBadge } from '../../components/DensityBadge';
import { useAppStore } from '../../store/useAppStore';
import { BASE_URL, STATUS_COLORS } from '../../lib/constants';
import { useTheme } from '../../hooks/useTheme';
import { Colors } from '../../theme';

const STATUS_OPTIONS = ['OPEN', 'RESPONDING', 'RESOLVED', 'CLOSED'];

function timeAgo(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60)    return `${diff}s ago`;
  if (diff < 3600)  return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function IncidentDetailScreen() {
  const { id }             = useLocalSearchParams<{ id: string }>();
  const router             = useRouter();
  const { username, role, token } = useAppStore();
  const { colors }         = useTheme();
  const styles             = useMemo(() => makeStyles(colors), [colors]);
  const insets             = useSafeAreaInsets();

  const [incident, setIncident]   = useState<Incident | null>(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);

  // ── Video state ─────────────────────────────────────────────────────────────
  const videoRef = useRef<Video | null>(null);
  const [videoStatus,   setVideoStatus]   = useState<AVPlaybackStatus | null>(null);
  const [videoReady,    setVideoReady]    = useState(false);
  const [videoError,    setVideoError]    = useState(false);

  // ── Status update state ──────────────────────────────────────────────────────
  const [statusModalVisible, setStatusModalVisible] = useState(false);
  const [selectedStatus,     setSelectedStatus]     = useState('');
  const [resolutionNote,     setResolutionNote]     = useState('');
  const [savingStatus,       setSavingStatus]       = useState(false);

  // ── Responder action state ───────────────────────────────────────────────────
  const [takingIncident, setTakingIncident] = useState(false);
  const [removing,       setRemoving]       = useState<number | null>(null);

  // ── Load incident ────────────────────────────────────────────────────────────
  const load = useCallback(async () => {
    setLoading(true);
    try {
      const all   = await apiIncidents();
      const found = all.find((i) => i.id === Number(id));
      if (!found) { setError('Incident not found.'); return; }
      setIncident(found);
      setSelectedStatus(found.incident_status);
      setError(null);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to load incident');
    } finally {
      setLoading(false);
    }
  }, [id]);

  const { markRead } = useAppStore();

  useEffect(() => { load(); }, [load]);

  // Mark incident as read as soon as it finishes loading
  useEffect(() => {
    if (incident) markRead(incident.id);
  }, [incident?.id]);

  // Reset video state whenever the incident changes
  useEffect(() => {
    setVideoReady(false);
    setVideoError(false);
    setVideoStatus(null);
  }, [incident?.id]);

  // ── Handlers ─────────────────────────────────────────────────────────────────
  async function handleTakeIncident() {
    if (!username) return;
    setTakingIncident(true);
    try {
      await apiAssignSelf(Number(id), username, role ?? 'Tanod');
      await load();
    } catch (e: any) {
      Alert.alert('Error', e?.response?.data?.error ?? e?.message ?? 'Failed to assign');
    } finally {
      setTakingIncident(false);
    }
  }

  async function handleUpdateStatus() {
    setSavingStatus(true);
    try {
      await apiUpdateStatus(Number(id), selectedStatus, resolutionNote);
      setStatusModalVisible(false);
      setResolutionNote('');
      await load();
    } catch (e: any) {
      Alert.alert('Error', e?.message ?? 'Failed to update status');
    } finally {
      setSavingStatus(false);
    }
  }

  async function handleRemoveResponder(rid: number) {
    Alert.alert('Remove Responder', 'Are you sure you want to remove this responder?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Remove', style: 'destructive',
        onPress: async () => {
          setRemoving(rid);
          try {
            await apiRemoveResponder(Number(id), rid);
            await load();
          } catch (e: any) {
            Alert.alert('Error', e?.message ?? 'Failed to remove');
          } finally {
            setRemoving(null);
          }
        },
      },
    ]);
  }

  async function togglePlayPause() {
    if (!videoRef.current || !videoReady) return;
    const status = videoStatus as any;
    if (status?.isPlaying) {
      await videoRef.current.pauseAsync();
    } else {
      // If finished, replay from start
      if (status?.didJustFinish || status?.positionMillis >= status?.durationMillis - 100) {
        await videoRef.current.setPositionAsync(0);
      }
      await videoRef.current.playAsync();
    }
  }

  // ── Loading / error screens ──────────────────────────────────────────────────
  if (loading && !incident) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator color={colors.accent} size="large" />
        <Text style={styles.loadingText}>Loading incident…</Text>
      </View>
    );
  }

  if (error || !incident) {
    return (
      <View style={styles.centered}>
        <Text style={styles.errorText}>{error ?? 'Not found'}</Text>
        <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
          <Text style={styles.backBtnText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Derived values ────────────────────────────────────────────────────────────
  const statusColor       = STATUS_COLORS[incident.incident_status] ?? '#94a3b8';
  // clip_url is a relative path like /archive_media/2026/01/clip.mp4 — prepend BASE_URL
  const clipUrl           = incident.clip_url
    ? `${BASE_URL}${incident.clip_url}`
    : null;
  const thumbUrl          = incident.thumbnail_url
    ? `${BASE_URL}${incident.thumbnail_url}`
    : null;
  const alreadyResponding = incident.responders.some(
    (r) => r.responder_name.toLowerCase() === (username ?? '').toLowerCase(),
  );
  // If the user has a pending assignment for this incident they haven't responded to yet
  const hasPendingForMe = (incident.pending_assignments ?? []).some(
    (pa) => pa.assigned_to_name.toLowerCase() === (username ?? '').toLowerCase(),
  );

  const isPlaying = (videoStatus as any)?.isPlaying === true;
  const hasClip   = !!clipUrl;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <View style={styles.screen}>

      {/* ── Nav bar ───────────────────────────────────────────────────────── */}
      <View style={[styles.navbar, { paddingTop: insets.top + 8 }]}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backIconBtn}>
          <Ionicons name="chevron-back" size={22} color={colors.textSec} />
          <Text style={styles.backLabel}>Incidents</Text>
        </TouchableOpacity>
        <View style={[styles.statusChip, { borderColor: statusColor, backgroundColor: `${statusColor}22` }]}>
          <Text style={[styles.statusChipText, { color: statusColor }]}>
            {incident.incident_status}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>

        {/* ── Title & meta ──────────────────────────────────────────────── */}
        <Text style={styles.incidentTitle}>{incident.title ?? 'Untitled Incident'}</Text>
        <View style={styles.metaRow}>
          <DensityBadge density={incident.density_tag} size="sm" />
          <Text style={styles.metaText}>TL-{incident.threat_level}</Text>
          <Text style={styles.metaText}>{timeAgo(incident.timestamp)}</Text>
        </View>
        <Text style={styles.metaDetail}>
          📍 {incident.location ?? incident.camera_id}
          {incident.people_count > 0 ? `  ·  👥 ${incident.people_count} people` : ''}
        </Text>

        {/* ── Video / Clip player ───────────────────────────────────────── */}
        <View style={styles.videoBox}>
          {hasClip ? (
            <>
              {/* expo-av Video — sends auth header so the server authorises it */}
              <Video
                ref={videoRef}
                source={{
                  uri: clipUrl!,
                  headers: token ? { Authorization: `Bearer ${token}` } : undefined,
                }}
                style={styles.video}
                resizeMode={ResizeMode.CONTAIN}
                shouldPlay={false}
                isLooping={false}
                onPlaybackStatusUpdate={(status) => {
                  setVideoStatus(status);
                  if (status.isLoaded && !videoReady) setVideoReady(true);
                  if (!status.isLoaded && (status as any).error) setVideoError(true);
                }}
                onError={() => setVideoError(true)}
              />

              {/* Overlay: loading spinner */}
              {!videoReady && !videoError && (
                <View style={styles.videoOverlay}>
                  <ActivityIndicator color={colors.accent} size="large" />
                  <Text style={styles.videoOverlayText}>Loading clip…</Text>
                </View>
              )}

              {/* Overlay: error */}
              {videoError && (
                <View style={styles.videoOverlay}>
                  <Ionicons name="videocam-off" size={32} color={colors.textHint} />
                  <Text style={styles.videoOverlayText}>Clip unavailable</Text>
                </View>
              )}

              {/* Overlay: play/pause button (shown when not playing or when ready) */}
              {videoReady && !videoError && (
                <TouchableOpacity
                  style={[
                    styles.playBtn,
                    isPlaying && styles.playBtnPlaying,
                  ]}
                  onPress={togglePlayPause}
                  activeOpacity={0.75}
                >
                  <Ionicons
                    name={isPlaying ? 'pause' : 'play'}
                    size={28}
                    color="#fff"
                  />
                </TouchableOpacity>
              )}

              {/* Bottom label bar */}
              <View style={styles.videoLabelBar}>
                <View style={styles.videoLabelDot} />
                <Text style={styles.videoLabelText}>
                  {isPlaying ? 'Playing clip' : 'Incident clip'}
                </Text>
                {incident.clip_filename && (
                  <Text style={styles.videoLabelFile} numberOfLines={1}>
                    {incident.clip_filename}
                  </Text>
                )}
              </View>
            </>
          ) : thumbUrl ? (
            /* Fallback: static thumbnail when no clip file */
            <>
              <Image source={{ uri: thumbUrl }} style={styles.video} resizeMode="cover" />
              <View style={styles.videoLabelBar}>
                <Ionicons name="image" size={12} color={colors.textMuted} style={{ marginRight: 6 }} />
                <Text style={styles.videoLabelText}>Thumbnail only — no clip recorded</Text>
              </View>
            </>
          ) : (
            /* No clip at all */
            <View style={styles.noClip}>
              <Ionicons name="videocam-off" size={36} color={colors.textHint} />
              <Text style={styles.noClipTitle}>No clip available</Text>
              <Text style={styles.noClipSub}>This incident has no recorded video</Text>
            </View>
          )}
        </View>

        {/* ── Resolution note ───────────────────────────────────────────── */}
        {incident.resolution_note ? (
          <View style={styles.resolutionBox}>
            <Text style={styles.resolutionLabel}>Resolution</Text>
            <Text style={styles.resolutionNote}>{incident.resolution_note}</Text>
          </View>
        ) : null}

        {/* ── Assigned Responders ───────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>👮 Assigned Responders</Text>

          {/* Confirmed responders */}
          {incident.responders.length === 0 && (incident.pending_assignments ?? []).length === 0 ? (
            <Text style={styles.emptyText}>No one assigned yet.</Text>
          ) : (
            incident.responders.map((r) => (
              <View key={r.id} style={styles.responderRow}>
                <View style={styles.responderAvatar}>
                  <Text style={styles.responderInitial}>
                    {r.responder_name.charAt(0).toUpperCase()}
                  </Text>
                </View>
                <View style={styles.responderInfo}>
                  <View style={styles.responderNameRow}>
                    <Text style={styles.responderName}>{r.responder_name}</Text>
                    <View style={styles.confirmedBadge}>
                      <Text style={styles.confirmedBadgeText}>CONFIRMED</Text>
                    </View>
                  </View>
                  <Text style={styles.responderMeta}>
                    {r.responder_role} · {timeAgo(r.assigned_at)}
                    {r.note ? `\n"${r.note}"` : ''}
                  </Text>
                </View>
                <TouchableOpacity
                  onPress={() => handleRemoveResponder(r.id)}
                  disabled={removing === r.id}
                >
                  {removing === r.id
                    ? <ActivityIndicator color="#ef4444" size="small" />
                    : <Ionicons name="close-circle-outline" size={22} color={colors.textHint} />
                  }
                </TouchableOpacity>
              </View>
            ))
          )}

          {/* Pending assignments (awaiting mobile-app response) */}
          {(incident.pending_assignments ?? []).map((pa) => (
            <View key={`pa-${pa.id}`} style={styles.pendingRow}>
              <View style={styles.pendingAvatar}>
                <Text style={styles.pendingInitial}>
                  {pa.assigned_to_name.charAt(0).toUpperCase()}
                </Text>
              </View>
              <View style={styles.responderInfo}>
                <View style={styles.responderNameRow}>
                  <Text style={styles.responderName}>{pa.assigned_to_name}</Text>
                  <View style={styles.pendingBadge}>
                    <Text style={styles.pendingBadgeText}>PENDING</Text>
                  </View>
                </View>
                <Text style={styles.responderMeta}>
                  Sent by {pa.assigned_by} · {timeAgo(pa.created_at)}
                </Text>
              </View>
            </View>
          ))}

          {/* Action buttons */}
          {!alreadyResponding && !hasPendingForMe && (
            <TouchableOpacity
              style={[styles.takeBtn, takingIncident && { opacity: 0.6 }]}
              onPress={handleTakeIncident}
              disabled={takingIncident}
            >
              {takingIncident
                ? <ActivityIndicator color="#fff" size="small" />
                : <Text style={styles.takeBtnText}>✋ I'll Take This Incident</Text>
              }
            </TouchableOpacity>
          )}
          {hasPendingForMe && !alreadyResponding && (
            <View style={styles.pendingForMeBox}>
              <Text style={styles.pendingForMeText}>
                ⏳ You have a pending assignment request for this incident
              </Text>
            </View>
          )}
          {alreadyResponding && (
            <View style={styles.alreadyBox}>
              <Text style={styles.alreadyText}>✅ You are assigned to this incident</Text>
            </View>
          )}
        </View>

        {/* ── Update Status ─────────────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔄 Update Status</Text>
          <TouchableOpacity
            style={styles.changeStatusBtn}
            onPress={() => setStatusModalVisible(true)}
          >
            <View style={styles.changeStatusLeft}>
              <View style={[styles.changeStatusDot, { backgroundColor: statusColor }]} />
              <Text style={styles.changeStatusText}>
                Current: <Text style={[styles.changeStatusValue, { color: statusColor }]}>
                  {incident.incident_status}
                </Text>
              </Text>
            </View>
            <Ionicons name="chevron-forward" size={18} color={colors.textSec} />
          </TouchableOpacity>
        </View>

        <Text style={styles.reporter}>
          Reported by <Text style={styles.reporterName}>{incident.reporter_name}</Text>
        </Text>

      </ScrollView>

      {/* ── Status Update Modal ───────────────────────────────────────────── */}
      <Modal
        visible={statusModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setStatusModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalSheet}>
            <View style={styles.modalHandle} />
            <Text style={styles.modalTitle}>Update Incident Status</Text>

            {/* Status option rows — NOTE: param renamed to 'opt' to avoid shadowing styles */}
            {STATUS_OPTIONS.map((opt) => {
              const sc     = STATUS_COLORS[opt] ?? '#94a3b8';
              const active = selectedStatus === opt;
              return (
                <TouchableOpacity
                  key={opt}
                  style={[
                    styles.statusOption,
                    active
                      ? { backgroundColor: `${sc}22`, borderColor: sc }
                      : styles.statusOptionInactive,
                  ]}
                  onPress={() => setSelectedStatus(opt)}
                  activeOpacity={0.75}
                >
                  {/* Colour dot */}
                  <View style={[styles.statusDot, { backgroundColor: sc }]} />

                  {/* Label */}
                  <Text style={[
                    styles.statusOptionText,
                    active ? { color: sc, fontWeight: '800' } : null,
                  ]}>
                    {opt}
                  </Text>

                  {/* Active checkmark */}
                  {active && (
                    <View style={[styles.statusCheckBadge, { backgroundColor: `${sc}22`, borderColor: `${sc}60` }]}>
                      <Ionicons name="checkmark" size={14} color={sc} />
                    </View>
                  )}
                </TouchableOpacity>
              );
            })}

            {/* Resolution note */}
            <Text style={styles.noteLabel}>Resolution note (optional)</Text>
            <TextInput
              style={styles.noteInput}
              value={resolutionNote}
              onChangeText={setResolutionNote}
              placeholder="Describe what action was taken…"
              placeholderTextColor={colors.textHint}
              multiline
              numberOfLines={3}
              textAlignVertical="top"
            />

            {/* Buttons */}
            <View style={styles.modalBtns}>
              <TouchableOpacity
                style={styles.cancelBtn}
                onPress={() => setStatusModalVisible(false)}
              >
                <Text style={styles.cancelBtnText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.saveBtn, savingStatus && { opacity: 0.6 }]}
                onPress={handleUpdateStatus}
                disabled={savingStatus}
              >
                {savingStatus
                  ? <ActivityIndicator color="#fff" size="small" />
                  : <Text style={styles.saveBtnText}>Save</Text>
                }
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

// ── Style factory ──────────────────────────────────────────────────────────────

function makeStyles(c: Colors) {
  return StyleSheet.create({
    // ── Scaffold ──────────────────────────────────────────────────────────────
    screen:      { flex: 1, backgroundColor: c.background },
    centered:    {
      flex: 1, backgroundColor: c.background,
      alignItems: 'center', justifyContent: 'center', gap: 12,
    },
    loadingText: { color: c.textMuted, fontSize: 13 },
    errorText:   { color: '#ef4444', fontSize: 14, textAlign: 'center', padding: 24 },
    backBtn:     {
      backgroundColor: c.surface, borderRadius: 8,
      paddingHorizontal: 20, paddingVertical: 10,
    },
    backBtnText: { color: c.accent, fontWeight: '600' },

    // ── Nav ───────────────────────────────────────────────────────────────────
    navbar: {
      flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
      paddingHorizontal: 16, paddingBottom: 12,
      borderBottomWidth: 1, borderBottomColor: c.borderLight,
    },
    backIconBtn:    { flexDirection: 'row', alignItems: 'center', gap: 4 },
    backLabel:      { fontSize: 15, color: c.textSec },
    statusChip:     { borderWidth: 1.5, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 4 },
    statusChipText: { fontSize: 11, fontWeight: '800', letterSpacing: 0.8 },

    // ── Content ───────────────────────────────────────────────────────────────
    content:       { paddingHorizontal: 20, paddingBottom: 48 },
    incidentTitle: { fontSize: 22, fontWeight: '800', color: c.text, marginBottom: 8, marginTop: 16 },
    metaRow:       { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 6 },
    metaText:      { fontSize: 12, color: c.textMuted, fontWeight: '600' },
    metaDetail:    { fontSize: 13, color: c.textMuted, marginBottom: 18, lineHeight: 18 },

    // ── Video box ─────────────────────────────────────────────────────────────
    videoBox: {
      borderRadius: 14, overflow: 'hidden',
      backgroundColor: '#000', marginBottom: 18,
      minHeight: 210,
      borderWidth: 1, borderColor: c.borderLight,
      position: 'relative',
    },
    video: { width: '100%', height: 210 },

    // Overlays (loading / error / play button)
    videoOverlay: {
      ...StyleSheet.absoluteFillObject,
      backgroundColor: 'rgba(0,0,0,0.65)',
      alignItems: 'center', justifyContent: 'center', gap: 10,
    },
    videoOverlayText: { fontSize: 13, color: '#94a3b8', fontWeight: '600' },

    playBtn: {
      position: 'absolute',
      bottom: 38,        // sits above the label bar
      right: 14,
      width: 48, height: 48, borderRadius: 24,
      backgroundColor: 'rgba(0,0,0,0.55)',
      borderWidth: 1.5, borderColor: 'rgba(255,255,255,0.25)',
      alignItems: 'center', justifyContent: 'center',
    },
    playBtnPlaying: {
      backgroundColor: 'rgba(16,185,129,0.35)',
      borderColor: 'rgba(16,185,129,0.6)',
    },

    // Label bar at bottom of video box
    videoLabelBar: {
      position: 'absolute', bottom: 0, left: 0, right: 0,
      flexDirection: 'row', alignItems: 'center',
      backgroundColor: 'rgba(0,0,0,0.7)',
      paddingHorizontal: 12, paddingVertical: 7, gap: 6,
    },
    videoLabelDot: {
      width: 7, height: 7, borderRadius: 4,
      backgroundColor: '#ef4444',
    },
    videoLabelText: { fontSize: 11, color: '#e2e8f0', fontWeight: '600', flex: 1 },
    videoLabelFile: { fontSize: 10, color: '#64748b', maxWidth: 140 },

    // No clip state
    noClip: {
      height: 210, alignItems: 'center', justifyContent: 'center', gap: 8,
      backgroundColor: c.surface,
    },
    noClipTitle: { fontSize: 14, color: c.textMuted, fontWeight: '700' },
    noClipSub:   { fontSize: 12, color: c.textHint },

    // ── Resolution box ────────────────────────────────────────────────────────
    resolutionBox: {
      backgroundColor: `${c.accent}18`, borderWidth: 1,
      borderColor: `${c.accent}40`, borderRadius: 12, padding: 14, marginBottom: 18,
    },
    resolutionLabel: {
      fontSize: 11, fontWeight: '700', color: c.accent,
      marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1,
    },
    resolutionNote: { fontSize: 13, color: c.accent, lineHeight: 19 },

    // ── Section card ──────────────────────────────────────────────────────────
    section: {
      backgroundColor: c.surface, borderRadius: 14, padding: 16,
      marginBottom: 14, gap: 10, borderWidth: 1, borderColor: c.borderLight,
    },
    sectionTitle: { fontSize: 15, fontWeight: '700', color: c.text, marginBottom: 2 },
    emptyText:    { fontSize: 13, color: c.textMuted, fontStyle: 'italic' },

    // ── Responders ────────────────────────────────────────────────────────────
    responderRow: {
      flexDirection: 'row', alignItems: 'center', gap: 12,
      paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: c.borderLight,
    },
    responderAvatar: {
      width: 38, height: 38, borderRadius: 19,
      backgroundColor: `${c.accent}22`, borderWidth: 1.5, borderColor: `${c.accent}60`,
      alignItems: 'center', justifyContent: 'center',
    },
    responderInitial:  { fontSize: 15, fontWeight: '800', color: c.accent },
    responderInfo:     { flex: 1 },
    responderNameRow:  { flexDirection: 'row', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
    responderName:     { fontSize: 14, fontWeight: '700', color: c.text },
    responderMeta:     { fontSize: 11, color: c.textMuted, marginTop: 2, lineHeight: 16 },

    // Confirmed badge (green)
    confirmedBadge:     {
      backgroundColor: `${c.accent}18`, borderWidth: 1, borderColor: `${c.accent}40`,
      borderRadius: 6, paddingHorizontal: 6, paddingVertical: 2,
    },
    confirmedBadgeText: { fontSize: 9, fontWeight: '800', color: c.accent, letterSpacing: 0.5 },

    // Pending assignment row (amber)
    pendingRow: {
      flexDirection: 'row', alignItems: 'center', gap: 12,
      paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: c.borderLight,
    },
    pendingAvatar: {
      width: 38, height: 38, borderRadius: 19,
      backgroundColor: 'rgba(245,158,11,0.15)', borderWidth: 1.5, borderColor: 'rgba(245,158,11,0.40)',
      alignItems: 'center', justifyContent: 'center',
    },
    pendingInitial:    { fontSize: 15, fontWeight: '800', color: '#f59e0b' },
    pendingBadge:      {
      backgroundColor: 'rgba(245,158,11,0.15)', borderWidth: 1,
      borderColor: 'rgba(245,158,11,0.35)', borderRadius: 6,
      paddingHorizontal: 6, paddingVertical: 2,
    },
    pendingBadgeText:  { fontSize: 9, fontWeight: '800', color: '#f59e0b', letterSpacing: 0.5 },

    // "You have a pending assignment" info box
    pendingForMeBox: {
      backgroundColor: 'rgba(245,158,11,0.10)', borderRadius: 10, paddingVertical: 11,
      paddingHorizontal: 14, alignItems: 'center',
      borderWidth: 1, borderColor: 'rgba(245,158,11,0.28)', marginTop: 4,
    },
    pendingForMeText: { color: '#f59e0b', fontSize: 13, fontWeight: '600', textAlign: 'center' },

    takeBtn: {
      backgroundColor: c.accent, borderRadius: 10,
      paddingVertical: 13, alignItems: 'center', marginTop: 4,
    },
    takeBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
    alreadyBox: {
      backgroundColor: `${c.accent}12`, borderRadius: 10, paddingVertical: 11,
      alignItems: 'center', borderWidth: 1, borderColor: `${c.accent}30`, marginTop: 4,
    },
    alreadyText: { color: c.accent, fontSize: 13, fontWeight: '700' },

    // ── Change-status button ──────────────────────────────────────────────────
    changeStatusBtn: {
      flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
      backgroundColor: c.surfaceInput, borderRadius: 10,
      paddingHorizontal: 14, paddingVertical: 13,
      borderWidth: 1, borderColor: c.border,
    },
    changeStatusLeft:  { flexDirection: 'row', alignItems: 'center', gap: 10 },
    changeStatusDot:   { width: 10, height: 10, borderRadius: 5 },
    changeStatusText:  { fontSize: 14, color: c.text, fontWeight: '600' },
    changeStatusValue: { fontWeight: '800' },

    reporter:     { fontSize: 12, color: c.textHint, textAlign: 'center', marginTop: 4 },
    reporterName: { color: c.accent, fontWeight: '700' },

    // ── Status modal ──────────────────────────────────────────────────────────
    modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.65)', justifyContent: 'flex-end' },
    modalSheet: {
      backgroundColor: c.surface,
      borderTopLeftRadius: 24, borderTopRightRadius: 24,
      padding: 24, gap: 10,
      borderTopWidth: 1, borderColor: c.border,
    },
    modalHandle: {
      width: 44, height: 5, backgroundColor: c.border,
      borderRadius: 3, alignSelf: 'center', marginBottom: 6,
    },
    modalTitle: { fontSize: 18, fontWeight: '800', color: c.text, marginBottom: 6 },

    // Status option row — inactive state
    statusOption: {
      flexDirection: 'row', alignItems: 'center', gap: 14,
      paddingHorizontal: 16, paddingVertical: 14,
      borderRadius: 12, borderWidth: 1.5,
    },
    statusOptionInactive: {
      backgroundColor: c.surfaceInput,
      borderColor: c.border,
    },
    statusDot:        { width: 11, height: 11, borderRadius: 6 },
    // Text is always fully readable — active gets the status colour, inactive gets c.text
    statusOptionText: { flex: 1, fontSize: 15, fontWeight: '600', color: c.text },
    statusCheckBadge: {
      borderRadius: 8, borderWidth: 1,
      paddingHorizontal: 6, paddingVertical: 2,
      alignItems: 'center', justifyContent: 'center',
    },

    noteLabel: {
      fontSize: 11, fontWeight: '700', color: c.textMuted,
      textTransform: 'uppercase', letterSpacing: 0.8, marginTop: 4,
    },
    noteInput: {
      backgroundColor: c.surfaceInput, borderWidth: 1, borderColor: c.border,
      borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12,
      fontSize: 14, color: c.text, minHeight: 88, textAlignVertical: 'top',
    },

    modalBtns:    { flexDirection: 'row', gap: 10, marginTop: 4 },
    cancelBtn: {
      flex: 1, backgroundColor: c.surfaceInput, borderRadius: 12,
      paddingVertical: 14, alignItems: 'center',
      borderWidth: 1, borderColor: c.border,
    },
    cancelBtnText: { color: c.text, fontWeight: '600', fontSize: 15 },
    saveBtn:       {
      flex: 1, backgroundColor: c.accent, borderRadius: 12,
      paddingVertical: 14, alignItems: 'center',
    },
    saveBtnText:   { color: '#fff', fontWeight: '800', fontSize: 15 },
  });
}
