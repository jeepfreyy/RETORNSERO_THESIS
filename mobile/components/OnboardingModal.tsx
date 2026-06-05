import React, { useState, useMemo } from 'react';
import {
  Modal, View, Text, TouchableOpacity, StyleSheet,
  Dimensions, ScrollView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useTheme } from '../hooks/useTheme';
import { Colors } from '../theme';

const { width } = Dimensions.get('window');

// ── Slide definitions ─────────────────────────────────────────────────────────
const SLIDES = [
  {
    icon:  'shield-checkmark' as const,
    color: '#10b981',
    title: 'Welcome to\nBarangay Sentinel',
    body:  'Your mobile field response companion. Stay connected with the barangay command center and respond to incidents faster and smarter.',
  },
  {
    icon:  'home' as const,
    color: '#10b981',
    title: 'Home Dashboard',
    body:  'The Home screen shows the current live crowd density level and your active incidents at a glance. Pull down to refresh anytime.',
  },
  {
    icon:  'alert-circle' as const,
    color: '#ef4444',
    title: 'Incident Reports',
    body:  'The Incidents tab lists all reported incidents. Unread ones are highlighted with a NEW badge. Tap any card to see full details, view the recorded clip, check assigned responders, and update the status.',
  },
  {
    icon:  'notifications' as const,
    color: '#f59e0b',
    title: 'Assignment Requests',
    body:  'When a barangay operator assigns you to an incident, a popup will appear. Tap Accept to confirm, or Decline and provide a reason. Your response is sent directly to the operator.',
  },
  {
    icon:  'checkmark-circle' as const,
    color: '#6366f1',
    title: 'My Tasks',
    body:  'My Tasks keeps a complete history of all assignments sent to you — active, accepted, and declined — so you always have a clear picture of your workload.',
  },
  {
    icon:  'videocam' as const,
    color: '#3b82f6',
    title: 'Live CCTV Feed',
    body:  'The Live tab streams the barangay CCTV camera in real time. Use it to monitor crowd density and verify details of reported incidents before responding.',
  },
  {
    icon:  'rocket' as const,
    color: '#10b981',
    title: "You're All Set!",
    body:  'You can revisit this guide anytime from Profile → Help & Guide. Stay safe, respond with confidence, and protect your community.',
  },
];

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  visible:    boolean;
  onComplete: () => void;
  /** When true the modal is a replay (from Profile); completing it just closes */
  isReplay?:  boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────
export function OnboardingModal({ visible, onComplete, isReplay = false }: Props) {
  const { colors } = useTheme();
  const insets     = useSafeAreaInsets();
  const s          = useMemo(() => makeStyles(colors), [colors]);

  const [step, setStep] = useState(0);
  const slide           = SLIDES[step];
  const isLast          = step === SLIDES.length - 1;

  function handleNext() {
    if (isLast) { onComplete(); setStep(0); }
    else         setStep((n) => n + 1);
  }
  function handleSkip() { onComplete(); setStep(0); }

  return (
    <Modal visible={visible} animationType="fade" statusBarTranslucent transparent>
      <View style={[s.backdrop, { paddingTop: insets.top, paddingBottom: insets.bottom + 16 }]}>

        {/* Skip button — hidden on last slide */}
        {!isLast && (
          <TouchableOpacity style={s.skipBtn} onPress={handleSkip} activeOpacity={0.7}>
            <Text style={s.skipText}>Skip</Text>
          </TouchableOpacity>
        )}

        {/* Card */}
        <View style={s.card}>

          {/* Icon halo */}
          <View style={[s.iconHalo, { borderColor: `${slide.color}30`, backgroundColor: `${slide.color}10` }]}>
            <View style={[s.iconRing, { borderColor: slide.color, backgroundColor: `${slide.color}18` }]}>
              <Ionicons name={slide.icon} size={40} color={slide.color} />
            </View>
          </View>

          {/* Step count */}
          <Text style={[s.stepCount, { color: slide.color }]}>
            {step + 1} of {SLIDES.length}
          </Text>

          {/* Title */}
          <Text style={s.title}>{slide.title}</Text>

          {/* Body */}
          <ScrollView style={s.bodyScroll} showsVerticalScrollIndicator={false}>
            <Text style={s.body}>{slide.body}</Text>
          </ScrollView>

          {/* Progress dots */}
          <View style={s.dots}>
            {SLIDES.map((_, i) => (
              <View
                key={i}
                style={[
                  s.dot,
                  i === step
                    ? { backgroundColor: slide.color, width: 20 }
                    : { backgroundColor: colors.border },
                ]}
              />
            ))}
          </View>

          {/* Action button */}
          <TouchableOpacity
            style={[s.nextBtn, { backgroundColor: slide.color }]}
            onPress={handleNext}
            activeOpacity={0.85}
          >
            <Text style={s.nextText}>
              {isLast ? (isReplay ? 'Close' : 'Get Started') : 'Next  →'}
            </Text>
          </TouchableOpacity>
        </View>

      </View>
    </Modal>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
function makeStyles(c: Colors) {
  return StyleSheet.create({
    backdrop: {
      flex: 1,
      backgroundColor: 'rgba(0,0,0,0.88)',
      justifyContent: 'center',
      alignItems: 'center',
      paddingHorizontal: 24,
    },

    skipBtn: {
      position: 'absolute', top: 56, right: 28,
    },
    skipText: { fontSize: 13, color: c.textMuted, fontWeight: '600' },

    card: {
      width: '100%',
      backgroundColor: c.surface,
      borderRadius: 24,
      borderWidth: 1,
      borderColor: c.border,
      padding: 28,
      alignItems: 'center',
      gap: 14,
      maxHeight: '85%',
    },

    iconHalo: {
      width: 110, height: 110, borderRadius: 55,
      borderWidth: 1,
      alignItems: 'center', justifyContent: 'center',
    },
    iconRing: {
      width: 84, height: 84, borderRadius: 42,
      borderWidth: 2,
      alignItems: 'center', justifyContent: 'center',
    },

    stepCount: { fontSize: 11, fontWeight: '700', letterSpacing: 1 },

    title: {
      fontSize: 22, fontWeight: '800', color: c.text,
      textAlign: 'center', lineHeight: 30,
    },

    bodyScroll: { maxHeight: 110, width: '100%' },
    body: {
      fontSize: 14, color: c.textSec, textAlign: 'center',
      lineHeight: 22,
    },

    dots: { flexDirection: 'row', gap: 6, alignItems: 'center' },
    dot:  { height: 6, borderRadius: 3 },

    nextBtn: {
      width: '100%', paddingVertical: 15,
      borderRadius: 14, alignItems: 'center',
      marginTop: 4,
    },
    nextText: { color: '#fff', fontSize: 15, fontWeight: '800', letterSpacing: 0.3 },
  });
}
