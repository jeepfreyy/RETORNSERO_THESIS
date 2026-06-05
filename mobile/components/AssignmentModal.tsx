import React, { useMemo, useState } from 'react';
import {
  Modal, View, Text, StyleSheet, TouchableOpacity,
  ScrollView, TextInput, ActivityIndicator, Alert,
  KeyboardAvoidingView, Platform,
} from 'react-native';
import { AssignmentNotification, apiRespondNotification } from '../lib/api';
import { DensityBadge } from './DensityBadge';
import { DENSITY_COLORS } from '../lib/constants';
import { useTheme } from '../hooks/useTheme';
import { useTranslation } from '../hooks/useTranslation';
import { Colors } from '../theme';

interface Props {
  notification: AssignmentNotification | null;
  onDismiss: (id: number) => void;
}

type Phase = 'view' | 'declining' | 'loading';

export function AssignmentModal({ notification, onDismiss }: Props) {
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s = useMemo(() => makeStyles(colors), [colors]);

  const [phase,  setPhase]  = useState<Phase>('view');
  const [reason, setReason] = useState('');

  React.useEffect(() => {
    if (notification) { setPhase('view'); setReason(''); }
  }, [notification?.id]);

  if (!notification) return null;
  const { incident } = notification;

  async function handleAccept() {
    setPhase('loading');
    try {
      await apiRespondNotification(notification!.id, 'accept');
      onDismiss(notification!.id);
    } catch {
      setPhase('view');
      Alert.alert('Error', 'Could not accept the assignment. Please try again.');
    }
  }

  async function handleDeclineConfirm() {
    if (!reason.trim()) {
      Alert.alert(t('modal.reasonLabel'), 'Please state why you cannot handle this incident.');
      return;
    }
    setPhase('loading');
    try {
      await apiRespondNotification(notification!.id, 'decline', reason.trim());
      onDismiss(notification!.id);
    } catch {
      setPhase('declining');
      Alert.alert('Error', 'Could not submit your response. Please try again.');
    }
  }

  const densityColor = DENSITY_COLORS[incident.density_tag] ?? '#64748b';
  const detectedAt   = new Date(incident.timestamp).toLocaleString('en-PH', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  const assignedAt   = new Date(notification.created_at).toLocaleString('en-PH', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

  return (
    <Modal visible transparent animationType="fade" statusBarTranslucent>
      <KeyboardAvoidingView
        style={s.backdrop}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <View style={s.card}>

          {/* Banner */}
          <View style={s.banner}>
            <Text style={s.bannerIcon}>🔔</Text>
            <View>
              <Text style={s.bannerTitle}>{t('modal.title')}</Text>
              <Text style={s.bannerSub}>
                {t('modal.sub', { name: notification.assigned_by })}
              </Text>
            </View>
          </View>

          <ScrollView style={s.scroll} contentContainerStyle={s.scrollContent} showsVerticalScrollIndicator={false}>

            {/* Title + density */}
            <View style={s.titleRow}>
              <DensityBadge density={incident.density_tag} size="sm" />
              <Text style={s.incidentTitle} numberOfLines={2}>{incident.title}</Text>
            </View>

            {/* Info grid */}
            <View style={s.infoGrid}>
              <InfoBox label={t('modal.location')} value={incident.location ?? t('modal.notSpecified')} s={s} />
              <InfoBox label={t('modal.people')} value={t('modal.peopleDet', { n: incident.people_count })} s={s} />
              <InfoBox label={t('modal.camera')} value={incident.camera_id} s={s} />
              <InfoBox
                label={t('modal.threat')}
                value={t('modal.threatVal', { n: incident.threat_level })}
                valueColor={incident.threat_level >= 7 ? '#ef4444' : incident.threat_level >= 4 ? '#f59e0b' : colors.accent}
                s={s}
              />
              <InfoBox label={t('modal.detected')}   value={detectedAt} fullWidth s={s} />
              <InfoBox label={t('modal.assignedAt')} value={assignedAt} fullWidth s={s} />
              {incident.reporter_name && (
                <InfoBox label={t('modal.reportedBy')} value={incident.reporter_name} fullWidth s={s} />
              )}
            </View>

            {/* Decline reason field */}
            {phase === 'declining' && (
              <View style={s.reasonBox}>
                <Text style={s.reasonLabel}>{t('modal.reasonLabel')}</Text>
                <TextInput
                  style={s.reasonInput}
                  placeholder={t('modal.reasonPh')}
                  placeholderTextColor={colors.textHint}
                  multiline
                  numberOfLines={4}
                  value={reason}
                  onChangeText={setReason}
                  autoFocus
                  textAlignVertical="top"
                />
              </View>
            )}
          </ScrollView>

          {/* Buttons */}
          {phase === 'loading' ? (
            <View style={s.loadingRow}>
              <ActivityIndicator color={colors.accent} />
              <Text style={s.loadingText}>{t('modal.submitting')}</Text>
            </View>
          ) : phase === 'view' ? (
            <View style={s.buttonRow}>
              <TouchableOpacity style={[s.btn, s.btnDecline]} onPress={() => setPhase('declining')} activeOpacity={0.8}>
                <Text style={s.btnDeclineText}>{t('modal.decline')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[s.btn, s.btnAccept]} onPress={handleAccept} activeOpacity={0.8}>
                <Text style={s.btnAcceptText}>{t('modal.accept')}</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={s.buttonRow}>
              <TouchableOpacity style={[s.btn, s.btnCancel]} onPress={() => setPhase('view')} activeOpacity={0.8}>
                <Text style={s.btnCancelText}>{t('modal.back')}</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[s.btn, s.btnConfirmDecline]} onPress={handleDeclineConfirm} activeOpacity={0.8}>
                <Text style={s.btnSubmitText}>{t('modal.confirmDec')}</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      </KeyboardAvoidingView>
    </Modal>
  );
}

function InfoBox({ label, value, fullWidth = false, valueColor, s }: {
  label: string; value: string; fullWidth?: boolean;
  valueColor?: string;
  s: ReturnType<typeof makeStyles>;
}) {
  return (
    <View style={[s.infoBox, fullWidth && s.infoBoxFull]}>
      <Text style={s.infoLabel}>{label}</Text>
      <Text style={[s.infoValue, valueColor ? { color: valueColor } : null]}>{value}</Text>
    </View>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    backdrop: {
      flex: 1, backgroundColor: 'rgba(0,0,0,0.75)',
      justifyContent: 'center', alignItems: 'center', padding: 16,
    },
    card: {
      width: '100%', maxHeight: '90%',
      backgroundColor: c.background, borderRadius: 20,
      borderWidth: 1, borderColor: '#ef4444', overflow: 'hidden',
    },

    banner: {
      flexDirection: 'row', alignItems: 'center', gap: 12,
      backgroundColor: 'rgba(239,68,68,0.12)',
      paddingHorizontal: 18, paddingVertical: 14,
      borderBottomWidth: 1, borderBottomColor: 'rgba(239,68,68,0.2)',
    },
    bannerIcon:  { fontSize: 26 },
    bannerTitle: { fontSize: 15, fontWeight: '800', color: '#ef4444', letterSpacing: 0.3 },
    bannerSub:   { fontSize: 12, color: c.textSec, marginTop: 2 },

    scroll:        { flexGrow: 0 },
    scrollContent: { padding: 18, gap: 14 },

    titleRow:      { gap: 8 },
    incidentTitle: { fontSize: 18, fontWeight: '800', color: c.text, lineHeight: 24 },

    infoGrid:    { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
    infoBox:     {
      width: '47%', backgroundColor: c.surface,
      borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10,
      borderWidth: 1, borderColor: c.border,
    },
    infoBoxFull: { width: '100%' },
    infoLabel:   { fontSize: 10, color: c.textMuted, fontWeight: '600', marginBottom: 4 },
    infoValue:   { fontSize: 13, color: c.text, fontWeight: '700' },

    reasonBox:   { gap: 8 },
    reasonLabel: { fontSize: 12, color: '#f59e0b', fontWeight: '700' },
    reasonInput: {
      backgroundColor: c.surface, borderRadius: 10,
      borderWidth: 1, borderColor: '#f59e0b',
      color: c.text, fontSize: 13,
      paddingHorizontal: 14, paddingVertical: 12, minHeight: 100,
    },

    buttonRow: {
      flexDirection: 'row', gap: 10, padding: 16,
      borderTopWidth: 1, borderTopColor: c.borderLight,
    },
    btn:              { flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center', justifyContent: 'center' },
    btnAccept:        { backgroundColor: c.accent },
    btnAcceptText:    { color: '#fff', fontSize: 15, fontWeight: '800' },
    btnDecline:       { backgroundColor: c.surface, borderWidth: 1, borderColor: '#ef4444' },
    btnDeclineText:   { color: '#ef4444', fontSize: 15, fontWeight: '700' },
    btnCancel:        { backgroundColor: c.surface, borderWidth: 1, borderColor: c.textHint },
    btnCancelText:    { color: c.textSec, fontSize: 15, fontWeight: '700' },
    btnConfirmDecline:{ backgroundColor: '#ef4444' },
    btnSubmitText:    { color: '#fff', fontSize: 15, fontWeight: '800' },

    loadingRow: {
      flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
      gap: 10, padding: 20, borderTopWidth: 1, borderTopColor: c.borderLight,
    },
    loadingText: { color: c.textMuted, fontSize: 13 },
  });
}
