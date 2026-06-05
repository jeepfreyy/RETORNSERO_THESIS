import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  ActivityIndicator, KeyboardAvoidingView, Platform,
  ScrollView, Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { apiForgotRequest, apiForgotVerify, apiForgotReset } from '../../lib/api';
import { AppLogo } from '../../components/AppLogo';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

type Step = 1 | 2 | 3;
const RESEND_COOLDOWN = 60;

function validatePassword(pw: string): string | null {
  if (pw.length < 8)      return 'At least 8 characters required.';
  if (!/[A-Z]/.test(pw))  return 'Must include at least one uppercase letter.';
  if (!/[\d\W]/.test(pw)) return 'Must include at least one number or symbol.';
  return null;
}

export default function ForgotPasswordScreen() {
  const router = useRouter();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s = useMemo(() => makeStyles(colors), [colors]);

  const [step,      setStep]     = useState<Step>(1);
  const [loading,   setLoading]  = useState(false);
  const [email,     setEmail]    = useState('');
  const [code,      setCode]     = useState('');
  const [ticket,    setTicket]   = useState('');
  const [resendCd,  setResendCd] = useState(0);
  const resendTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const [newPw,     setNewPw]    = useState('');
  const [confirmPw, setConfirmPw]= useState('');

  useEffect(() => () => {
    if (resendTimer.current) clearInterval(resendTimer.current);
  }, []);

  function startResendCooldown() {
    setResendCd(RESEND_COOLDOWN);
    resendTimer.current = setInterval(() => {
      setResendCd(prev => {
        if (prev <= 1) { clearInterval(resendTimer.current!); return 0; }
        return prev - 1;
      });
    }, 1000);
  }

  async function handleRequestCode() {
    const normalizedEmail = email.trim().toLowerCase();
    if (!normalizedEmail) { Alert.alert('Missing field', 'Please enter your email address.'); return; }
    setLoading(true);
    try {
      await apiForgotRequest(normalizedEmail);
      setEmail(normalizedEmail);
      setStep(2);
      startResendCooldown();
    } catch (e: any) {
      const status = e?.response?.status;
      const msg    = e?.response?.data?.message ?? e?.message ?? '';
      if (!msg || msg.includes('reach') || status == null) {
        Alert.alert('Cannot Reach Server', 'Make sure the Flask server is running and your phone is on the same Wi-Fi network as the PC.');
      } else {
        Alert.alert('Error', msg);
      }
    } finally { setLoading(false); }
  }

  async function handleResend() {
    if (resendCd > 0) return;
    setLoading(true);
    try {
      const data = await apiForgotRequest(email);
      Alert.alert('Code Resent', data.message);
      startResendCooldown();
    } catch (e: any) {
      Alert.alert('Error', e?.response?.data?.message ?? e?.message ?? 'Cannot reach server.');
    } finally { setLoading(false); }
  }

  async function handleVerifyCode() {
    if (code.length !== 6) { Alert.alert('Invalid code', 'Please enter the 6-digit code from your email.'); return; }
    setLoading(true);
    try {
      const data = await apiForgotVerify(email, code);
      if (data.success && data.reset_ticket) {
        setTicket(data.reset_ticket);
        setStep(3);
      } else {
        Alert.alert('Verification Failed', data.message ?? 'Invalid or expired code.');
      }
    } catch (e: any) {
      Alert.alert('Error', e?.response?.data?.message ?? e?.message ?? 'Cannot reach server.');
    } finally { setLoading(false); }
  }

  async function handleReset() {
    const pwErr = validatePassword(newPw);
    if (pwErr) { Alert.alert('Weak password', pwErr); return; }
    if (newPw !== confirmPw) { Alert.alert('Password mismatch', 'Passwords do not match.'); return; }
    setLoading(true);
    try {
      const data = await apiForgotReset(ticket, newPw);
      if (data.success) {
        Alert.alert(
          'Password Reset!',
          'Your password has been updated. You can now sign in with your new password.',
          [{ text: t('reg.signIn'), onPress: () => router.replace('/(auth)') }],
        );
      } else {
        Alert.alert('Reset Failed', data.message ?? 'Could not reset password.');
      }
    } catch (e: any) {
      Alert.alert('Error', e?.response?.data?.message ?? e?.message ?? 'Cannot reach server.');
    } finally { setLoading(false); }
  }

  const rules = {
    length:    newPw.length >= 8,
    uppercase: /[A-Z]/.test(newPw),
    numSym:    /[\d\W]/.test(newPw),
  };
  const pwMatch = newPw === confirmPw && confirmPw.length > 0;

  // Step label names
  const stepLabels = [t('fp.stepEmail'), t('fp.stepVerify'), t('fp.stepReset')];

  return (
    <KeyboardAvoidingView style={s.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        contentContainerStyle={s.container}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={s.logoWrap}>
          <AppLogo subtitle={t('fp.subtitle')} />
        </View>

        {/* ── Step progress indicator ──────────────────────────────────────── */}
        <View style={s.stepRow}>
          {([1, 2, 3] as Step[]).map((sn, idx) => (
            <View key={sn} style={s.stepItem}>
              <View style={[s.stepDot, step >= sn && s.stepDotActive]}>
                {step > sn
                  ? <Text style={s.stepDotCheck}>✓</Text>
                  : <Text style={[s.stepDotNum, step === sn && s.stepDotNumActive]}>{sn}</Text>
                }
              </View>
              <Text style={[s.stepLabel, step >= sn && s.stepLabelActive]}>
                {stepLabels[idx]}
              </Text>
              {sn < 3 && <View style={[s.stepLine, step > sn && s.stepLineActive]} />}
            </View>
          ))}
        </View>

        {/* ── Card ─────────────────────────────────────────────────────────── */}
        <View style={s.card}>

          {/* STEP 1 */}
          {step === 1 && (
            <>
              <Text style={s.cardTitle}>{t('fp.step1Title')}</Text>
              <Text style={s.cardDesc}>{t('fp.step1Desc')}</Text>
              <Text style={s.label}>{t('fp.emailLabel')}</Text>
              <TextInput
                style={s.input}
                value={email}
                onChangeText={setEmail}
                placeholder="your@email.com"
                placeholderTextColor={colors.textHint}
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
                onSubmitEditing={handleRequestCode}
                returnKeyType="send"
              />
              <TouchableOpacity style={[s.btn, loading && s.btnDisabled]} onPress={handleRequestCode} disabled={loading} activeOpacity={0.85}>
                {loading ? <ActivityIndicator color="#fff" /> : <Text style={s.btnText}>{t('fp.sendBtn')}</Text>}
              </TouchableOpacity>
            </>
          )}

          {/* STEP 2 */}
          {step === 2 && (
            <>
              <Text style={s.cardTitle}>{t('fp.step2Title')}</Text>
              <Text style={s.cardDesc}>
                {t('fp.step2Desc')}
                <Text style={s.emailHighlight}>{email}</Text>.
              </Text>
              <Text style={s.label}>{t('fp.codeLabel')}</Text>
              <TextInput
                style={[s.input, s.otpInput]}
                value={code}
                onChangeText={v => setCode(v.replace(/\D/g, '').slice(0, 6))}
                placeholder="000000"
                placeholderTextColor={colors.textHint}
                keyboardType="number-pad"
                maxLength={6}
                onSubmitEditing={handleVerifyCode}
                returnKeyType="done"
                autoFocus
              />
              <TouchableOpacity style={[s.btn, (loading || code.length !== 6) && s.btnDisabled]} onPress={handleVerifyCode} disabled={loading || code.length !== 6} activeOpacity={0.85}>
                {loading ? <ActivityIndicator color="#fff" /> : <Text style={s.btnText}>{t('fp.verifyBtn')}</Text>}
              </TouchableOpacity>
              <TouchableOpacity onPress={handleResend} disabled={resendCd > 0 || loading} activeOpacity={0.7} style={s.resendWrap}>
                <Text style={[s.resendText, resendCd > 0 && s.resendTextDisabled]}>
                  {resendCd > 0 ? t('fp.resendCd', { n: resendCd }) : t('fp.resend')}
                </Text>
              </TouchableOpacity>
              <View style={s.tipBox}>
                <Text style={s.tipTitle}>{t('fp.tipTitle')}</Text>
                <Text style={s.tipItem}>{t('fp.tip1')}</Text>
                <Text style={s.tipItem}>{t('fp.tip2')}</Text>
                <Text style={s.tipItem}>{t('fp.tip3')}</Text>
              </View>
              <TouchableOpacity onPress={() => { setCode(''); setStep(1); }} activeOpacity={0.7} style={s.changeEmailWrap}>
                <Text style={s.changeEmailText}>{t('fp.changeEmail')}</Text>
              </TouchableOpacity>
            </>
          )}

          {/* STEP 3 */}
          {step === 3 && (
            <>
              <Text style={s.cardTitle}>{t('fp.step3Title')}</Text>
              <Text style={s.cardDesc}>{t('fp.step3Desc')}</Text>
              <Text style={s.label}>{t('fp.newPwLabel')}</Text>
              <TextInput
                style={s.input}
                value={newPw}
                onChangeText={setNewPw}
                placeholder="••••••••"
                placeholderTextColor={colors.textHint}
                secureTextEntry
              />
              {newPw.length > 0 && (
                <View style={s.rulesBox}>
                  <RuleRow met={rules.length}    label={t('reg.rule8char')}  colors={colors} />
                  <RuleRow met={rules.uppercase} label={t('reg.ruleUpper')}  colors={colors} />
                  <RuleRow met={rules.numSym}    label={t('reg.ruleNumSym')} colors={colors} />
                </View>
              )}
              <Text style={s.label}>{t('fp.confirmPwLabel')}</Text>
              <TextInput
                style={[s.input, confirmPw.length > 0 && (pwMatch ? s.inputOk : s.inputErr)]}
                value={confirmPw}
                onChangeText={setConfirmPw}
                placeholder="••••••••"
                placeholderTextColor={colors.textHint}
                secureTextEntry
                onSubmitEditing={handleReset}
                returnKeyType="done"
              />
              {confirmPw.length > 0 && !pwMatch && <Text style={s.errorHint}>{t('reg.pwMismatch')}</Text>}
              <TouchableOpacity style={[s.btn, loading && s.btnDisabled]} onPress={handleReset} disabled={loading} activeOpacity={0.85}>
                {loading ? <ActivityIndicator color="#fff" /> : <Text style={s.btnText}>{t('fp.resetBtn')}</Text>}
              </TouchableOpacity>
            </>
          )}
        </View>

        {/* ── Back to login ────────────────────────────────────────────────── */}
        <TouchableOpacity onPress={() => router.replace('/(auth)')} activeOpacity={0.7} style={s.backWrap}>
          <Text style={s.backText}>{t('fp.backToLogin')}</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function RuleRow({ met, label, colors }: { met: boolean; label: string; colors: Colors }) {
  return (
    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
      <Text style={{ fontSize: 13, width: 16, textAlign: 'center', color: met ? colors.accent : colors.textHint }}>
        {met ? '✓' : '○'}
      </Text>
      <Text style={{ fontSize: 12, color: met ? colors.accent : colors.textMuted }}>{label}</Text>
    </View>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    flex:      { flex: 1, backgroundColor: c.background },
    container: {
      flexGrow: 1, justifyContent: 'center',
      paddingHorizontal: 24, paddingVertical: 40, gap: 24,
    },
    logoWrap: { alignItems: 'center' },

    // Step indicator
    stepRow: { flexDirection: 'row', justifyContent: 'center', alignItems: 'flex-start', gap: 0 },
    stepItem: { alignItems: 'center', position: 'relative', flex: 1 },
    stepDot: {
      width: 32, height: 32, borderRadius: 16,
      backgroundColor: c.surface, borderWidth: 2, borderColor: c.border,
      alignItems: 'center', justifyContent: 'center', zIndex: 1,
    },
    stepDotActive:    { borderColor: c.accent, backgroundColor: `${c.accent}22` },
    stepDotCheck:     { fontSize: 14, color: c.accent, fontWeight: '800' },
    stepDotNum:       { fontSize: 13, color: c.textMuted, fontWeight: '700' },
    stepDotNumActive: { color: c.accent },
    stepLabel:        { fontSize: 10, color: c.textMuted, marginTop: 6, textAlign: 'center' },
    stepLabelActive:  { color: c.accent, fontWeight: '700' },
    stepLine: {
      position: 'absolute', top: 16, left: '50%', right: '-50%', height: 2,
      backgroundColor: c.border, zIndex: 0,
    },
    stepLineActive: { backgroundColor: c.accent },

    // Card
    card: {
      backgroundColor: c.surface, borderRadius: 18,
      padding: 24, borderWidth: 1, borderColor: c.border, gap: 8,
    },
    cardTitle: { fontSize: 20, fontWeight: '800', color: c.text, marginBottom: 4 },
    cardDesc:  { fontSize: 13, color: c.textMuted, lineHeight: 20 },
    emailHighlight: { color: c.accent, fontWeight: '700' },

    label: {
      fontSize: 11, fontWeight: '700', color: c.textMuted,
      textTransform: 'uppercase', letterSpacing: 0.8,
      marginTop: 6, marginBottom: 2,
    },
    input: {
      backgroundColor: c.surfaceInput, borderWidth: 1, borderColor: c.border,
      borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12,
      fontSize: 15, color: c.text,
    },
    otpInput: { fontSize: 22, fontWeight: '800', letterSpacing: 8, textAlign: 'center' },
    inputOk:  { borderColor: c.accent },
    inputErr: { borderColor: '#ef4444' },
    errorHint:{ fontSize: 11, color: '#ef4444', marginTop: 2 },

    btn:         { backgroundColor: c.accent, borderRadius: 12, paddingVertical: 14, alignItems: 'center', marginTop: 8 },
    btnDisabled: { opacity: 0.5 },
    btnText:     { color: '#fff', fontSize: 16, fontWeight: '700', letterSpacing: 0.4 },

    resendWrap:         { alignSelf: 'center', marginTop: 4 },
    resendText:         { fontSize: 13, color: c.accent, fontWeight: '600' },
    resendTextDisabled: { color: c.textMuted },

    tipBox: {
      backgroundColor: `${c.accent}10`,
      borderRadius: 10, padding: 14, gap: 5,
      borderWidth: 1, borderColor: `${c.accent}30`,
    },
    tipTitle: { fontSize: 12, color: c.accent,   fontWeight: '700', marginBottom: 4 },
    tipItem:  { fontSize: 12, color: c.textMuted, lineHeight: 18 },

    changeEmailWrap: { alignSelf: 'center' },
    changeEmailText: { fontSize: 12, color: c.textMuted, fontWeight: '600' },

    rulesBox: {
      backgroundColor: c.surfaceInput, borderRadius: 8,
      padding: 10, marginTop: 4, gap: 5,
      borderWidth: 1, borderColor: c.borderLight,
    },

    backWrap: { alignSelf: 'center' },
    backText: { fontSize: 13, color: c.textMuted, fontWeight: '600' },
  });
}
