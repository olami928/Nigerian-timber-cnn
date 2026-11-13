# Offline testing report (Android APK)

**Date:** 2026-11-13  
**Status:** NOT EXECUTED (no APK built; no Android device/emulator/SDK in this environment)

## Required tests (real Android device, airplane mode)

Perform and record pass/fail + observations:

1. Install the APK.
2. Enable airplane mode.
3. Launch the application (cold start).
4. Navigate through all screens (Identify/Home, Species directory, Guide/Photograph, About).
5. Capture a photograph via camera.
6. Select a photograph from the device gallery.
7. Run timber identification (local inference).
8. View prediction and species information (top-3, patch votes/agreement, uses, disclaimers).
9. Close and fully reopen the application while remaining offline.
10. Repeat identification.

**Pass criteria:** all steps succeed with no network access and no runtime downloads (model/WASM/assets/species DB all local).

## Additional checks

- No external CDN/font/API requests (verify via network inspector/logcat/proxy if available).
- Model loads from packaged assets; ORT WASM resolves locally in WebView.
- Large photographs do not crash; EXIF/orientation handled correctly.
- Portrait/small-large screens, touch targets, loading/error states, long names.

## Environment blockers

- `java/javac`, Android SDK (`sdkmanager`, build-tools, platform 34), `adb`, and device/emulator not available.
- Debug/release APK not compiled; on-device behavior unverified.

## Evidence to attach after execution

- Device model, Android version, APK build/version, date/time.
- Screenshots/screen recording (airplane mode visible) and notes per step.
- Any failures, logcat excerpts, and fixes applied.

**Result:** No successful offline test claimed.
