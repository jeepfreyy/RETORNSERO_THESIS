import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { DENSITY_COLORS, DENSITY_BG } from '../lib/constants';

interface Props {
  density: string;
  size?: 'sm' | 'md' | 'lg';
}

export function DensityBadge({ density, size = 'md' }: Props) {
  const color = DENSITY_COLORS[density] ?? '#94a3b8';
  const bg    = DENSITY_BG[density]     ?? 'rgba(148,163,184,0.15)';

  const fontSize = size === 'sm' ? 10 : size === 'lg' ? 16 : 12;
  const px       = size === 'sm' ? 6  : size === 'lg' ? 14 : 10;
  const py       = size === 'sm' ? 2  : size === 'lg' ? 6  : 4;

  return (
    <View style={[styles.badge, { backgroundColor: bg, borderColor: color, paddingHorizontal: px, paddingVertical: py }]}>
      <Text style={[styles.text, { color, fontSize }]}>{density}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    borderRadius: 6,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  text: {
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});
