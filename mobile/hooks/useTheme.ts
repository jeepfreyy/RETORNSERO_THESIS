import { useAppStore } from '../store/useAppStore';
import { DarkColors, LightColors, Colors } from '../theme';

export function useTheme(): { colors: Colors; isDark: boolean } {
  const theme = useAppStore((s) => s.theme);
  const isDark = theme === 'dark';
  return { colors: isDark ? DarkColors : LightColors, isDark };
}
