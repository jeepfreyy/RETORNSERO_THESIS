import React, { useMemo, useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  StyleSheet, ActivityIndicator, KeyboardAvoidingView,
  Platform, ScrollView, Alert,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useAppStore } from '../../store/useAppStore';
import { apiLogin } from '../../lib/api';
import { AppLogo } from '../../components/AppLogo';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { Colors } from '../../theme';

export default function LoginScreen() {
  const { setAuth } = useAppStore();
  const router = useRouter();
  const { colors } = useTheme();
  const { t } = useTranslation();
  const s = useMemo(() => makeStyles(colors), [colors]);

  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [loading,  setLoading]  = useState(false);

  async function handleLogin() {
    if (!email.trim() || !password) {
      Alert.alert(t('auth.loginFailed'), t('auth.missingFields'));
      return;
    }
    setLoading(true);
    try {
      const data = await apiLogin(email.trim().toLowerCase(), password);
      if (data.success) {
        await setAuth(data.token, data.username, data.role);
      } else {
        Alert.alert(t('auth.loginFailed'), data.message ?? 'Invalid email or password.');
      }
    } catch (e: any) {
      const msg = e?.response?.data?.message ?? e?.message ?? 'Cannot reach server.';
      Alert.alert(t('auth.connError'), msg + '\n\nMake sure the server is running and SERVER_IP is correct in lib/constants.ts.');
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
        {/* ── Logo ──────────────────────────────────────────────────────────── */}
        <View style={s.logoWrap}>
          <AppLogo subtitle={t('app.field')} />
        </View>

        {/* ── Form ──────────────────────────────────────────────────────────── */}
        <View style={s.card}>
          <Text style={s.cardTitle}>{t('auth.signIn')}</Text>

          <Text style={s.label}>{t('auth.emailLabel')}</Text>
          <TextInput
            style={s.input}
            value={email}
            onChangeText={setEmail}
            placeholder={t('auth.emailPh')}
            placeholderTextColor={colors.textHint}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
          />

          <Text style={s.label}>{t('auth.passwordLabel')}</Text>
          <TextInput
            style={s.input}
            value={password}
            onChangeText={setPassword}
            placeholder={t('auth.passwordPh')}
            placeholderTextColor={colors.textHint}
            secureTextEntry
            onSubmitEditing={handleLogin}
            returnKeyType="done"
          />

          <TouchableOpacity
            onPress={() => router.push('/(auth)/forgot-password' as any)}
            activeOpacity={0.7}
            style={s.forgotWrap}
          >
            <Text style={s.forgotText}>{t('auth.forgotPw')}</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[s.btn, loading && s.btnDisabled]}
            onPress={handleLogin}
            disabled={loading}
            activeOpacity={0.85}
          >
            {loading
              ? <ActivityIndicator color="#fff" />
              : <Text style={s.btnText}>{t('auth.signInBtn')}</Text>
            }
          </TouchableOpacity>
        </View>

        {/* ── Register link ──────────────────────────────────────────────── */}
        <View style={s.registerRow}>
          <Text style={s.registerPrompt}>{t('auth.noAccount')}</Text>
          <TouchableOpacity onPress={() => router.push('/(auth)/register' as any)} activeOpacity={0.7}>
            <Text style={s.registerLink}>{t('auth.createAccount')}</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
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
      padding: 24, borderWidth: 1, borderColor: c.border, gap: 6,
    },
    cardTitle: { fontSize: 20, fontWeight: '800', color: c.text, marginBottom: 10 },

    label: {
      fontSize: 11, fontWeight: '700', color: c.textMuted,
      textTransform: 'uppercase', letterSpacing: 0.8,
      marginTop: 8, marginBottom: 4,
    },
    input: {
      backgroundColor: c.surfaceInput, borderWidth: 1, borderColor: c.border,
      borderRadius: 10, paddingHorizontal: 14, paddingVertical: 12,
      fontSize: 15, color: c.text,
    },

    forgotWrap: { alignSelf: 'flex-end', marginTop: 6 },
    forgotText: { fontSize: 12, color: c.accent, fontWeight: '600' },

    btn:         { backgroundColor: c.accent, borderRadius: 12, paddingVertical: 14, alignItems: 'center', marginTop: 16 },
    btnDisabled: { opacity: 0.6 },
    btnText:     { color: '#fff', fontSize: 16, fontWeight: '700', letterSpacing: 0.4 },

    registerRow:    { flexDirection: 'row', justifyContent: 'center', alignItems: 'center' },
    registerPrompt: { fontSize: 13, color: c.textMuted },
    registerLink:   { fontSize: 13, color: c.accent, fontWeight: '700' },
  });
}
