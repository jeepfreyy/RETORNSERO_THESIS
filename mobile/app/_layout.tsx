import { useEffect } from 'react';
import { Stack, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useAppStore } from '../store/useAppStore';
import { useNotifications } from '../hooks/useNotifications';
import { AssignmentModal } from '../components/AssignmentModal';
import { OnboardingModal } from '../components/OnboardingModal';
import { useTheme } from '../hooks/useTheme';

export default function RootLayout() {
  const { token, isLoaded, loadAuth, onboardingDone, completeOnboarding } = useAppStore();
  const { colors } = useTheme();
  const segments = useSegments();
  const router   = useRouter();

  // ── Restore persisted auth + preferences on first mount ─────────────────
  useEffect(() => { loadAuth(); }, []);

  // ── Auth guard: redirect to login or app based on token ─────────────────
  useEffect(() => {
    if (!isLoaded) return;
    const inAuth = segments[0] === '(auth)';
    if (!token && !inAuth) {
      router.replace('/(auth)');
    } else if (token && inAuth) {
      router.replace('/(tabs)');
    }
  }, [token, isLoaded, segments]);

  // ── Global assignment notification polling ───────────────────────────────
  const { current: pendingNotification, dismiss } = useNotifications();

  return (
    <>
      <StatusBar style={colors.statusBar} backgroundColor={colors.background} />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.background } }}>
        <Stack.Screen name="(auth)" />
        <Stack.Screen name="(tabs)" />
        <Stack.Screen name="incident/[id]" options={{ presentation: 'card' }} />
      </Stack>

      {/* Assignment notification popup — renders on top of everything */}
      <AssignmentModal
        notification={pendingNotification}
        onDismiss={dismiss}
      />

      {/* First-time onboarding guide — shown once per device after first login */}
      <OnboardingModal
        visible={!!token && isLoaded && !onboardingDone}
        onComplete={completeOnboarding}
      />
    </>
  );
}
