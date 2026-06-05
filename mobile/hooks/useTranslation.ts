import { useAppStore } from '../store/useAppStore';
import { translations, Lang } from '../i18n';

/**
 * useTranslation — returns a `t(key, vars?)` function for the current language.
 *
 * Usage:
 *   const { t } = useTranslation();
 *   t('home.greeting', { name: 'Juan' })  // → "Hello, Juan 👋"  (EN)
 *                                          // → "Kamusta, Juan 👋"  (TL)
 */
export function useTranslation() {
  const lang = useAppStore((s) => s.lang) as Lang;

  function t(key: string, vars?: Record<string, string | number>): string {
    const map = translations[lang] ?? translations['en'];
    let str = map[key] ?? translations['en'][key] ?? key;
    if (vars) {
      Object.entries(vars).forEach(([k, v]) => {
        str = str.replace(`{${k}}`, String(v));
      });
    }
    return str;
  }

  return { t, lang };
}
