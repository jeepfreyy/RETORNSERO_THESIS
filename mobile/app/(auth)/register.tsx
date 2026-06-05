import React, { useMemo, useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, ActivityIndicator, KeyboardAvoidingView,
  Platform, ScrollView, Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { apiRegister } from '../../lib/api';
import { AppLogo } from '../../components/AppLogo';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

function validatePassword(pw: string): string | null {
  if (pw.length < 8)         return 'At least 8 characters required.';
  if (!/[A-Z]/.test(pw))     return 'Must include at least one uppercase letter.';
  if (!/[\d\W]/.test(pw))    return 'Must include at least one number or symbol.';
  return null;
}

export default function RegisterScreen() {
  const router = useRouter();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s = useMemo(() => makeStyles(colors), [colors]);

  const [username, setUsername] = useState('');
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [confirm,  setConfirm]  = useState('');
  const [loading,  setLoading]  = useState(false);

  const rules = {
    length:    password.length >= 8,
    uppercase: /[A-Z]/.test(password),
    numSym:    /[\d\W]/.test(password),
  };
  const passwordMatch = password === confirm && confirm.length > 0;

  async function handleRegister() {
    if (!username.trim() || !email.trim() || !password || !confirm) {
      Alert.alert('Missing fields', 'Please fill in all fields.');
      return;
    }
    const pwErr = validatePassword(password);
    if (pwErr) { Alert.alert('Weak password', pwErr); return; }
    if (password !== confirm) {
      Alert.alert('Password mismatch', 'Passwords do not match.');
      return;
    }
    setLoading(true);
    try {
      const data = await apiRegister(username.trim(), email.trim().toLowerCase(), password);
      if (data.success) {
        Alert.alert(
          'Account Created!',
          'Your account has been registered. You can now sign in.',
          [{ text: t('reg.signIn'), onPress: () => router.replace('/(auth)') }],
        );
      } else {
        Alert.alert('Registration Failed', data.message ?? 'Could not create account.');
      }
    } catch (e: any) {
      Alert.alert('Error', e?.response?.data?.message ?? e?.message ?? 'Cannot reach server.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <KeyboardAvoidingView
      style={s.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        contentContainerStyle={s.container}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={s.logoWrap}>
          <AppLogo subtitle={t('reg.subtitle')} />
        </View>

        <View style={s.card}>
          <Text style={s.cardTitle}>{t('reg.title')}</Text>

          <Text style={s.label}>{t('reg.usernameLabel')}</Text>
          <TextInput
            style={s.input}
            value={username}
            onChangeText={setUsername}
            placeholder={t('reg.usernamePh')}
            placeholderTextColor={colors.textHint}
            autoCapitalize="none"
            autoCorrect={false}
          />

          <Text style={s.label}>{t('reg.emailLabel')}</Text>
          <TextInput
            style={s.input}
            value={email}
            onChangeText={setEmail}
            placeholder="your@email.com"
            placeholderTextColor={colors.textHint}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
          />

          <Text style={s.label}>{t('reg.pwLabel')}</Text>
          <TextInput
            style={s.input}
            value={password}
            onChangeText={setPassword}
            placeholder="••••••••"
            placeholderTextColor={colors.textHint}
            secureTextEntry
          />

          {password.length > 0 && (
            <View style={s.rulesBox}>
              <RuleRow met={rules.length}    label={t('reg.rule8char')}  colors={colors} />
              <RuleRow met={rules.uppercase} label={t('reg.ruleUpper')}  colors={colors} />
              <RuleRow met={rules.numSym}    label={t('reg.ruleNumSym')} colors={colors} />
            </View>
          )}

          <Text style={s.label}>{t('reg.confirmLabel')}</Text>
          <TextInput
            style={[
              s.input,
              confirm.length > 0 && (passwordMatch ? s.inputOk : s.inputErr),
            ]}
            value={confirm}
            onChangeText={setConfirm}
            placeholder="••••••••"
            placeholderTextColor={colors.textHint}
            secureTextEntry
            onSubmitEditing={handleRegister}
            returnKeyType="done"
          />
          {confirm.length > 0 && !passwordMatch && (
            <Text style={s.errorHint}>{t('reg.pwMismatch')}</Text>
          )}

          <TouchableOpacity
            style={[s.btn, loading && s.btnDisabled]}
            onPress={handleRegister}
            disabled={loading}
            activeOpacity={0.85}
          >
            {loading
              ? <ActivityIndicator color="#fff" />
              : <Text style={s.btnText}>{t('reg.createBtn')}</Text>
            }
          </TouchableOpacity>
        </View>

        <View style={s.footRow}>
          <Text style={s.footPrompt}>{t('reg.haveAccount')}</Text>
          <TouchableOpacity onPress={() => router.back()} activeOpacity={0.7}>
            <Text style={s.footLink}>{t('reg.signIn')}</Text>
          </TouchableOpacity>
        </View>
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
      <Text style={{ fontSize: 12, color: met ? colors.accent : colors.textMuted }}>
        {label}
      </Text>
    </View>
  );
}

function makeStyles(c: Colors) {
  return StyleSheet.create({
    flex:      { flex: 1, backgroundColor: c.background },
    container: {
      flexGrow: 1, justifyContent: 'center',
      paddingHorizontal: 24, paddingVertical: 48, gap: 28,
    },
    logoWrap: { alignItems: 'center' },

    card: {
      backgroundColor: c.surface, borderRadius: 18,
      padding: 24, borderWidth: 1, borderColor: c.border, gap: 4,
    },
    cardTitle: { fontSize: 20, fontWeight: '800', color: c.text, marginBottom: 12 },

    label: {
      fontSize: 11, fontWeight: '700', color: c.textMuted,
      textTransform: 'uppercase', letterSpacing: 0.8,
      marginTop: 10, marginBottom: 4,
    },
    input: {
      backgroundColor: c.surfaceInput, borderWidth: 1, borderColor: c.border,
      borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12,
      fontSize: 15, color: c.text,
    },
    inputOk:   { borderColor: c.accent },
    inputErr:  { borderColor: '#ef4444' },
    errorHint: { fontSize: 11, color: '#ef4444', marginTop: 2 },

    rulesBox: {
      backgroundColor: c.surfaceInput,
      borderRadius: 8, padding: 10, marginTop: 6, gap: 5,
      borderWidth: 1, borderColor: c.borderLight,
    },

    btn:         { backgroundColor: c.accent, borderRadius: 12, paddingVertical: 14, alignItems: 'center', marginTop: 18 },
    btnDisabled: { opacity: 0.6 },
    btnText:     { color: '#fff', fontSize: 16, fontWeight: '700', letterSpacing: 0.4 },

    footRow:    { flexDirection: 'row', justifyContent: 'center', alignItems: 'center' },
    footPrompt: { fontSize: 13, color: c.textMuted },
    footLink:   { fontSize: 13, color: c.accent, fontWeight: '700' },
  });
}
