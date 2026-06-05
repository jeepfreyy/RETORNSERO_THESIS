import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../hooks/useTheme';

interface Props {
  subtitle?: string;
}

export function AppLogo({ subtitle = 'Mobile Incident Response' }: Props) {
  const { colors } = useTheme();

  return (
    <View style={styles.wrapper}>
      {/* Outer glow halo */}
      <View style={[styles.glow, { borderColor: `${colors.accent}30` }]}>
        {/* Solid ring */}
        <View style={[styles.ring, { borderColor: colors.accent, backgroundColor: `${colors.accent}18` }]}>
          <Ionicons name="videocam" size={36} color={colors.accent} />
        </View>
      </View>

      <Text style={[styles.name, { color: colors.accent }]}>BARANGAY SENTINEL</Text>
      {subtitle ? <Text style={[styles.sub, { color: colors.textMuted }]}>{subtitle}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: { alignItems: 'center', gap: 12 },

  glow: {
    width: 100, height: 100, borderRadius: 50,
    backgroundColor: 'rgba(16,185,129,0.06)',
    borderWidth: 1,
    alignItems: 'center', justifyContent: 'center',
    shadowColor: '#10b981',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.55,
    shadowRadius: 18,
    elevation: 12,
  },

  ring: {
    width: 78, height: 78, borderRadius: 39,
    borderWidth: 2,
    alignItems: 'center', justifyContent: 'center',
  },

  name: {
    fontSize: 22, fontWeight: '800',
    letterSpacing: 2.5, textAlign: 'center',
  },
  sub: {
    fontSize: 12, letterSpacing: 0.5, textAlign: 'center',
  },
});
