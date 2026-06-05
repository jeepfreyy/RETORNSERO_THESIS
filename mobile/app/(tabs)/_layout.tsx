import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useTheme } from '../../hooks/useTheme';
import { useTranslation } from '../../hooks/useTranslation';
import { useAppStore } from '../../store/useAppStore';

export default function TabsLayout() {
  const { colors } = useTheme();
  const { t } = useTranslation();
  const { readIds, allIncidentIds } = useAppStore();

  // Count how many known incidents the user hasn't opened yet
  const unreadCount = allIncidentIds.filter((id) => !readIds.includes(id)).length;

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: colors.tabBar,
          borderTopColor:  colors.tabBarBorder,
          borderTopWidth:  1,
          height:          62,
          paddingBottom:   8,
        },
        tabBarActiveTintColor:   colors.accent,
        tabBarInactiveTintColor: colors.textHint,
        tabBarLabelStyle: { fontSize: 10, fontWeight: '600' },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="home" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="incidents"
        options={{
          title: t('inc.title'),
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="alert-circle" size={size} color={color} />
          ),
          tabBarBadge:          unreadCount > 0 ? (unreadCount > 99 ? '99+' : unreadCount) : undefined,
          tabBarBadgeStyle:     { backgroundColor: '#ef4444', fontSize: 9, minWidth: 16, height: 16 },
        }}
      />
      <Tabs.Screen
        name="tasks"
        options={{
          title: t('tasks.title'),
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="checkmark-circle" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="live"
        options={{
          title: 'Live',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="videocam" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: t('profile.title'),
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="person" size={size} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
