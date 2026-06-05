Place the following image files in this folder before building a production APK:

  icon.png              — 1024x1024 px app icon (used for iOS and Android)
  adaptive-icon.png     — 1024x1024 px foreground for Android adaptive icon
  splash.png            — 1284x2778 px (or 1242x2688 px) splash screen image
  favicon.png           — 32x32 or 64x64 px favicon for web
  notification-icon.png — 96x96 px white silhouette icon for push notifications

For development with Expo Go, these files are not required.
Expo will use a default placeholder icon if they are missing.

Quick way to generate placeholder icons:
  npx expo install expo-asset
  Run: npx generate-expo-app-icons  (if you have a source image)
